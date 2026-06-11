"""Run a full Dynare model through MATLAB and return a structured verdict.

Unlike :mod:`dynare_lsp.preprocessor` (which only runs the Dynare *front-end*
to validate parsing/structure) this module **executes the model end-to-end**:
it drives the real Dynare toolbox inside a headless MATLAB process, so the
caller gets back what Dynare itself computes — the deterministic steady state,
the Blanchard-Kahn eigenvalue verdict, and Dynare's own error text when a run
fails.

.. warning::

   Running a ``.mod`` file **executes arbitrary code**.  Dynare ``.mod`` files
   can carry trailing MATLAB statements after the model blocks, and those run
   in the MATLAB session this module spawns.  Only run model text you trust.

The MATLAB side is **not** re-implemented here.  This module reuses the same
ground-truth runner the oracle cache builder uses
(``dynare_lsp/oracle/run_dynare_model.m``) and the same
``matlab -batch "run_dynare_model('mod','out','dynare')"`` invocation pattern as
``dynare_lsp/oracle/build_cache.py`` (run with ``cwd`` set to the oracle
directory so MATLAB finds the ``.m`` file, child stdout captured to a log,
``stdin`` set to ``DEVNULL``, and a per-run timeout).  This module does *not*
run the corpus-wide cache sweep — it executes exactly one model per call.

MATLAB and the Dynare toolbox are discovered explicitly so the behaviour is
predictable and overridable:

* MATLAB: the ``matlab_path`` argument, else the ``DYNARE_LSP_MATLAB``
  environment variable, else ``matlab`` on ``PATH``, else the common Windows
  install location and common macOS/Linux install locations.
* Dynare ``matlab/`` dir: the ``dynare_path`` argument, else the
  ``DYNARE_LSP_DYNARE`` environment variable, else common Windows/macOS/Linux
  install locations. Explicit and environment paths may point either to the
  Dynare install root or directly to its ``matlab/`` directory.

If neither MATLAB nor Dynare can be located the functions here **never raise** —
they return a result dict with ``matlab_available: False`` (or
``dynare_available: False``) and an explanatory ``message`` so callers (notably
the ``dynare_run_dynare`` MCP tool) degrade gracefully.
"""

from __future__ import annotations

import glob
import json
import math
import os
import platform
import signal
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .include_resolver import _uri_to_path


# The MATLAB runner scripts ship under ``matlab/`` in published builds; the
# in-repo dev tree keeps them next to the oracle cache builder.  Reuse the
# script verbatim rather than shipping a second copy of the Dynare invocation.
_PKG_DIR = Path(__file__).resolve().parent
_ORACLE_DIR = _PKG_DIR / "oracle"
_MATLAB_DIR = _PKG_DIR / "matlab"


def runner_dir() -> Path:
    """Directory holding the bundled MATLAB runner scripts.

    Published builds place ``run_dynare_model.m`` and
    ``dynare_session_server.m`` under ``matlab/``; the dev tree keeps them
    under ``oracle/`` next to the cache builder.  Prefer ``matlab/`` and fall
    back to ``oracle/`` so both layouts work.
    """
    if (_MATLAB_DIR / "run_dynare_model.m").is_file():
        return _MATLAB_DIR
    return _ORACLE_DIR


_RUNNER_M = runner_dir() / "run_dynare_model.m"

# Same default install locations the oracle cache builder pins (build_cache.py).
_DEFAULT_MATLAB_WINDOWS = r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"
_DEFAULT_DYNARE_WINDOWS = "C:/Program Files/dynare/7.1/matlab"

_DEFAULT_TIMEOUT = 300


def _matlab_char_literal(value: str) -> str:
    """Return *value* as a MATLAB single-quoted character vector literal."""
    return "'" + value.replace("'", "''") + "'"


def _matlab_popen_kwargs() -> Dict[str, Any]:
    """Start MATLAB in a killable process group where the platform supports it."""
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": creationflags} if creationflags else {}
    return {"start_new_session": True}


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Best-effort termination of MATLAB plus descendants after a timeout."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if completed.returncode == 0:
                return
        except OSError:
            pass
    else:
        try:
            os.killpg(proc.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            return
        except OSError:
            pass
    try:
        proc.kill()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# MATLAB / Dynare discovery (explicit + overridable)
# ---------------------------------------------------------------------------

def find_matlab(matlab_path: Optional[str] = None) -> Optional[str]:
    """Locate a MATLAB executable.

    Search order:

    1. Explicit *matlab_path* argument.
    2. ``DYNARE_LSP_MATLAB`` environment variable.
    3. ``matlab`` on ``PATH`` (``shutil.which``).
    4. Common platform install locations (newest version first), falling back
       to the pinned Windows default the oracle uses.

    Returns the path to use, or ``None`` if nothing usable was found.  A
    caller may deliberately point *matlab_path* at a nonexistent binary to
    force the "MATLAB unavailable" branch, so a missing explicit path yields
    ``None`` rather than being passed through.
    """
    # 1. Explicit override.
    if matlab_path:
        return matlab_path if _is_file(matlab_path) else None

    # 2. Environment variable.
    env_path = os.environ.get("DYNARE_LSP_MATLAB")
    if env_path:
        return env_path if _is_file(env_path) else None

    # 3. PATH.
    which_result = shutil.which("matlab")
    if which_result:
        return which_result

    # 4. Common platform install locations (newest version first).
    if os.name == "nt":
        roots = sorted(
            glob.glob(r"C:\Program Files\MATLAB\R*\bin\matlab.exe"),
            reverse=True,
        )
        for candidate in roots:
            if _is_file(candidate):
                return candidate
        if _is_file(_DEFAULT_MATLAB_WINDOWS):
            return _DEFAULT_MATLAB_WINDOWS
    else:
        system = platform.system()
        patterns: List[str] = []
        if system == "Darwin":
            patterns.extend([
                "/Applications/MATLAB_R*.app/bin/matlab",
                "/Applications/MATLAB*.app/bin/matlab",
            ])
        elif system == "Linux":
            patterns.extend([
                "/usr/local/MATLAB/R*/bin/matlab",
                "/opt/MATLAB/R*/bin/matlab",
                "/usr/lib/matlab/bin/matlab",
            ])
        for candidate in sorted(
            [path for pattern in patterns for path in glob.glob(pattern)],
            reverse=True,
        ):
            if _is_file(candidate):
                return candidate

    return None


def find_dynare(dynare_path: Optional[str] = None) -> Optional[str]:
    """Locate the Dynare ``matlab/`` directory (the one containing ``dynare.m``).

    Search order:

    1. Explicit *dynare_path* argument.
    2. ``DYNARE_LSP_DYNARE`` environment variable.
    3. Common platform install locations, falling back to the pinned 7.1
       Windows default the oracle uses.

    A directory is accepted only if it actually contains ``dynare.m`` (so a
    stale/empty path doesn't silently produce a ``no_dynare`` run). Explicit
    and environment paths may point either to the install root or directly to
    the ``matlab/`` directory. Returns the directory to put on the MATLAB path,
    or ``None``.
    """
    if dynare_path:
        return _resolve_dynare_matlab_dir(dynare_path)

    env_path = os.environ.get("DYNARE_LSP_DYNARE")
    if env_path:
        return _resolve_dynare_matlab_dir(env_path)

    candidates: List[str] = []
    if os.name == "nt":
        candidates.extend(sorted(
            glob.glob(r"C:\Program Files\dynare\*"),
            reverse=True,
        ))
        candidates.append(_DEFAULT_DYNARE_WINDOWS)
    else:
        system = platform.system()
        if system == "Linux":
            candidates.extend([
                "/usr/share/dynare",
                "/usr/local/share/dynare",
                "/usr/lib/dynare",
                "/usr/local/lib/dynare",
            ])
            candidates.extend(sorted(glob.glob("/opt/dynare/*"), reverse=True))
        elif system == "Darwin":
            candidates.extend([
                "/Applications/Dynare.app/Contents/Resources/dynare",
                "/Applications/Dynare.app/Contents/Resources",
                "/Applications/Dynare",
                "/opt/homebrew/share/dynare",
                "/usr/local/share/dynare",
            ])

    for candidate in candidates:
        resolved = _resolve_dynare_matlab_dir(candidate)
        if resolved is not None:
            return resolved

    return None


def _is_file(path: str) -> bool:
    try:
        return os.path.isfile(path)
    except OSError:
        return False


def _has_dynare_m(directory: str) -> bool:
    try:
        return os.path.isfile(os.path.join(directory, "dynare.m"))
    except OSError:
        return False


def _resolve_dynare_matlab_dir(path: str) -> Optional[str]:
    """Resolve either a Dynare install root or its ``matlab/`` directory."""
    for candidate in (path, os.path.join(path, "matlab")):
        if _has_dynare_m(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Result shaping
# ---------------------------------------------------------------------------

def _bk_unavailable(message: str) -> Dict[str, Any]:
    return {
        "satisfied": None,
        "message": message,
        "n_explosive": None,
        "n_forward_looking": None,
    }


def _unavailable(message: str, *, matlab_available: bool) -> Dict[str, Any]:
    """Build the graceful-degradation result for a run that never started."""
    return {
        "success": False,
        "matlab_available": matlab_available,
        "dynare_available": False,
        "status": "no_matlab" if not matlab_available else "no_dynare",
        "steady_state": {},
        "blanchard_kahn": _bk_unavailable(message),
        "errors": [message],
        "message": message,
        "raw_log": "",
    }


def _shape_from_record(rec: Any, raw_log: str) -> Dict[str, Any]:
    """Translate ``run_dynare_model.m``'s JSON record into the public contract.

    The MATLAB runner emits ``status`` (``success`` / ``dynare_error`` /
    ``no_dynare`` / ``harness_error``), the endogenous names + steady state,
    and an eigenvalue/Blanchard-Kahn summary (``n_explosive``, ``nfwrd``,
    ``nboth``, ``bk_rank_ok``).  Map those into the structured dict callers
    expect, keeping non-finite steady-state entries (written as JSON ``null``)
    out of the map.
    """
    if not isinstance(rec, dict):
        message = "MATLAB result JSON was not an object."
        return {
            "success": False,
            "matlab_available": True,
            "dynare_available": True,
            "status": "bad_json",
            "steady_state": {},
            "blanchard_kahn": _bk_unavailable(message),
            "errors": [message],
            "raw_log": raw_log,
            "message": message,
        }

    status = rec.get("status", "unknown")
    success = status == "success"

    # Steady state: pair endo names with values, dropping nulls (non-finite).
    # A successful Dynare run should report one steady-state slot per
    # endogenous variable.  If it does not, treat the record as malformed
    # instead of publishing a truncated prefix as a successful run.
    names_raw = rec.get("endo_names")
    values_raw = rec.get("steady_state")
    names_are_arrays = isinstance(names_raw, list) and isinstance(values_raw, list)
    names = names_raw if isinstance(names_raw, list) else []
    values = values_raw if isinstance(values_raw, list) else []
    malformed_message = ""
    if success:
        if not names_are_arrays:
            malformed_message = (
                "Malformed Dynare record: steady-state names and values "
                "must be arrays."
            )
        elif len(names) != len(values):
            malformed_message = (
                "Malformed Dynare record: "
                f"{len(names)} endogenous name(s) but "
                f"{len(values)} steady-state value(s)."
            )
    if malformed_message:
        status = "harness_error"
        success = False
    steady_state: Dict[str, float] = {}
    if names_are_arrays and len(names) == len(values):
        for idx, name in enumerate(names):
            value = values[idx]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                steady_state[str(name)] = float(value)

    # Blanchard-Kahn: the runner reports a rank verdict (bk_rank_ok) only when
    # it has a decision-rule eigenvalue decomposition.  Translate to the same
    # {satisfied, message} shape the MCP BK tool uses.
    bk_ok = rec.get("bk_rank_ok")
    # The MATLAB runner initialises unset scalar fields as ``[]`` (an empty
    # numeric matrix), which ``jsonencode`` renders as a JSON empty array.
    # Normalise those to ``None`` so the contract's ``int | None`` holds.
    n_explosive = _opt_int(rec.get("n_explosive"))
    nfwrd = _opt_int(rec.get("nfwrd"))
    nboth = _opt_int(rec.get("nboth"))
    n_forward = None if nfwrd is None or nboth is None else nfwrd + nboth
    if bk_ok is True:
        if n_explosive is None or n_forward is None:
            bk_message = "Blanchard-Kahn satisfied: count details unavailable."
        else:
            bk_message = (
                f"Blanchard-Kahn satisfied: {n_explosive} explosive eigenvalue(s) "
                f"match {n_forward} forward-looking variable(s)."
            )
        bk_satisfied: Optional[bool] = True
    elif bk_ok is False:
        if n_explosive is None or n_forward is None:
            bk_message = (
                "Blanchard-Kahn condition not satisfied: "
                "count details unavailable."
            )
        else:
            bk_message = (
                f"Blanchard-Kahn condition not satisfied: {n_explosive} explosive "
                f"eigenvalue(s) vs {n_forward} forward-looking variable(s)."
            )
        bk_satisfied = False
    elif not rec.get("has_dr"):
        bk_satisfied = None
        bk_message = (
            "No Blanchard-Kahn verdict: Dynare produced no decision rule "
            "(model did not solve, or is not a stochastic/perturbation run)."
        )
    else:
        bk_satisfied = None
        bk_message = "Blanchard-Kahn verdict unavailable."

    # Errors: surface Dynare's own message verbatim when the run failed.
    errors: List[str] = []
    error_raw = rec.get("error")
    err_text = error_raw.strip() if isinstance(error_raw, str) else ""
    error_id_raw = rec.get("error_id", "")
    error_id = error_id_raw if isinstance(error_id_raw, str) else ""
    if malformed_message:
        errors.append(malformed_message)
    elif err_text:
        errors.append(err_text)
    elif not success and status not in ("success",):
        errors.append(f"Dynare run did not succeed (status: {status}).")

    return {
        "success": success,
        "matlab_available": True,
        "dynare_available": status != "no_dynare",
        "status": status,
        "error_id": error_id,
        "steady_state": steady_state,
        "blanchard_kahn": {
            "satisfied": bk_satisfied,
            "message": bk_message,
            "n_explosive": n_explosive,
            "n_forward_looking": n_forward,
        },
        "errors": errors,
        "raw_log": raw_log,
    }


def _opt_int(value: Any) -> Optional[int]:
    """Coerce a scalar to int, or ``None`` for missing / empty-matrix fields.

    ``run_dynare_model.m`` initialises unset scalar fields as ``[]``, which
    ``jsonencode`` renders as a JSON empty list (``[]`` -> Python ``list``);
    those, and any non-numeric value, collapse to ``None``.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


# ---------------------------------------------------------------------------
# Run one model
# ---------------------------------------------------------------------------

def _workspace_relative_path(
    fname: str,
    normalized: Dict[str, Path],
    common_parent: Path,
) -> Path:
    path = normalized[fname]
    try:
        rel = path.relative_to(common_parent)
    except ValueError:
        rel = Path(path.name)
    if rel.is_absolute() or any(part == ".." for part in rel.parts):
        return Path(path.name)
    return rel


def _materialize_workspace_files(
    work_dir: Path,
    active_file: str,
    files: Dict[str, str],
    active_content: str,
) -> Path:
    workspace_files = dict(files)
    workspace_files[active_file] = active_content
    normalized = {
        fname: _uri_to_path(fname)
        for fname in workspace_files
    }
    parents = [str(path.parent) for path in normalized.values()]
    try:
        common_parent = Path(os.path.commonpath(parents))
    except ValueError:
        common_parent = normalized[active_file].parent

    entry_parent = (
        work_dir
        / _workspace_relative_path(active_file, normalized, common_parent).parent
    )
    entry_parent.mkdir(parents=True, exist_ok=True)

    basename_counts: Dict[str, int] = {}
    materialized: List[Tuple[str, Path]] = []
    for fname, content in workspace_files.items():
        rel = _workspace_relative_path(fname, normalized, common_parent)
        target = work_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        materialized.append((content, target))
        basename_counts[target.name] = basename_counts.get(target.name, 0) + 1

    for content, target in materialized:
        if basename_counts.get(target.name) != 1:
            continue
        alias = entry_parent / target.name
        if alias.exists():
            continue
        alias.write_text(content, encoding="utf-8")

    return work_dir / _workspace_relative_path(
        active_file,
        normalized,
        common_parent,
    )


def run_dynare_matlab(
    mod_text: str,
    timeout: int = _DEFAULT_TIMEOUT,
    matlab_path: Optional[str] = None,
    dynare_path: Optional[str] = None,
    active_file: Optional[str] = None,
    files: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Execute one ``.mod`` model end-to-end with MATLAB + Dynare.

    .. warning::

       This **executes arbitrary code**.  Trailing MATLAB statements in
       *mod_text* run inside the spawned MATLAB session.  Only pass trusted
       model text.

    Writes *mod_text* to a private temp directory, invokes MATLAB in batch mode
    with the Dynare toolbox on the path, runs the model via the bundled
    ``run_dynare_model.m`` (the same runner the oracle cache uses), and returns
    a structured verdict.  The function never raises for an unavailable
    toolchain, a model error, a timeout, or a MATLAB crash — every outcome maps
    to a result dict.

    Args:
        mod_text: Full text of a Dynare ``.mod`` file.
        timeout: Seconds before the MATLAB run is abandoned (the process is
            killed and ``status`` becomes ``"timeout"``).
        matlab_path: Explicit ``matlab`` executable.  Falls back to
            ``DYNARE_LSP_MATLAB`` / ``PATH`` / common install dir.  Point this
            at a nonexistent path to force the "MATLAB unavailable" branch.
        dynare_path: Explicit Dynare ``matlab/`` directory (the one with
            ``dynare.m``).  Falls back to ``DYNARE_LSP_DYNARE`` / common
            install dir.
        active_file: Optional filename for *mod_text* when *files* supplies
            related workspace files such as ``@#include`` siblings.
        files: Optional ``filename -> content`` map to materialize in the
            private runner temp directory.  The active file's content is
            always overwritten with *mod_text* so unsaved edits are honored.

    Returns:
        ``{
            "success": bool,                # True iff Dynare ran end-to-end
            "matlab_available": bool,       # False if no MATLAB was found/ran
            "dynare_available": bool,       # False if MATLAB ran but no Dynare
            "status": str,                  # success | dynare_error | timeout |
                                            #   matlab_crash | bad_json |
                                            #   no_matlab | no_dynare |
                                            #   harness_error | runner_missing
            "steady_state": {var: value},   # finite entries only
            "blanchard_kahn": {
                "satisfied": bool | None,
                "message": str,
                "n_explosive": int | None,
                "n_forward_looking": int | None,
            },
            "errors": [str],                # Dynare's own error text on failure
            "raw_log": str,                 # full MATLAB stdout/stderr
            "message": str,                 # present on degraded/error paths
        }``
    """
    matlab = find_matlab(matlab_path)
    if not matlab:
        return _unavailable(
            "MATLAB executable not found. Set DYNARE_LSP_MATLAB or pass "
            "matlab_path. (This tool executes the model in MATLAB + Dynare.)",
            matlab_available=False,
        )

    dynare = find_dynare(dynare_path)
    if not dynare:
        return _unavailable(
            "Dynare toolbox not found (need the matlab/ dir containing "
            "dynare.m). Set DYNARE_LSP_DYNARE or pass dynare_path.",
            matlab_available=True,
        )

    if not _RUNNER_M.is_file():
        result = _unavailable(
            f"MATLAB runner script missing: {_RUNNER_M}",
            matlab_available=True,
        )
        result["status"] = "runner_missing"
        return result

    work_dir = Path(tempfile.mkdtemp(prefix="dynare_run_"))
    work_model = work_dir / "model.mod"
    out_json = work_dir / "result.json"
    log_path = work_dir / "matlab.log"
    raw_log = ""
    timed_out = False
    proc_rc: Optional[int] = None

    try:
        if active_file and files:
            work_model = _materialize_workspace_files(
                work_dir,
                active_file,
                files,
                mod_text,
            )
        else:
            work_model.write_text(mod_text, encoding="utf-8")

        # Mirror build_cache.run_one's invocation exactly: matlab -batch with a
        # single run_dynare_model(mod, out_json, dynare_root) call, cwd set to
        # the oracle dir so MATLAB resolves run_dynare_model.m, stdout to a file
        # (not a pipe, which a detached MATLAB can deadlock on), stdin DEVNULL.
        cmd = [
            matlab,
            "-batch",
            "run_dynare_model({},{},{})".format(
                _matlab_char_literal(work_model.as_posix()),
                _matlab_char_literal(out_json.as_posix()),
                _matlab_char_literal(Path(dynare).as_posix()),
            ),
        ]
        t0 = time.time()
        try:
            with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(_RUNNER_M.parent),
                    stdin=subprocess.DEVNULL,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    **_matlab_popen_kwargs(),
                )
                try:
                    proc.wait(timeout=timeout)
                    proc_rc = proc.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _kill_process_tree(proc)
                    try:
                        proc.wait(timeout=30)
                    except (subprocess.TimeoutExpired, OSError):
                        pass
        except OSError as exc:
            # MATLAB resolved to a path but could not be launched (e.g. an
            # explicit nonexistent binary the discovery step let through is
            # impossible now, but a non-executable file or perms issue is).
            return _unavailable(
                f"Failed to launch MATLAB '{matlab}': {exc}",
                matlab_available=False,
            )

        try:
            raw_log = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            raw_log = ""

        if out_json.exists():
            try:
                rec = json.loads(out_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return {
                    "success": False,
                    "matlab_available": True,
                    "dynare_available": True,
                    "status": "bad_json",
                    "steady_state": {},
                    "blanchard_kahn": _bk_unavailable(
                        "MATLAB result file was not valid JSON."
                    ),
                    "errors": [f"Could not parse MATLAB result JSON: {exc}"],
                    "raw_log": raw_log,
                    "message": f"Could not parse MATLAB result JSON: {exc}",
                }
            return _shape_from_record(rec, raw_log)

        # No result file: the runner never reached its write step.
        status = "timeout" if timed_out else "matlab_crash"
        message = (
            f"MATLAB run timed out after {timeout}s."
            if timed_out
            else "MATLAB exited without writing a result "
            f"(exit code {proc_rc}). See raw_log."
        )
        return {
            "success": False,
            "matlab_available": True,
            "dynare_available": True,
            "status": status,
            "exit_code": proc_rc,
            "seconds": round(time.time() - t0, 1),
            "steady_state": {},
            "blanchard_kahn": _bk_unavailable(message),
            "errors": [message],
            "raw_log": raw_log,
            "message": message,
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
