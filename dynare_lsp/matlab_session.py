"""Persistent MATLAB + Dynare session: start one MATLAB, reuse it for many runs.

:mod:`matlab_runner` spawns a fresh ``matlab -batch`` per model, paying MATLAB's
~7 s startup every call.  This module instead launches MATLAB **once** running
``oracle/dynare_session_server.m`` (a small watch-loop), then feeds it models
via a work directory and reads back the same JSON verdict the one-shot runner
produces.  Startup is paid once; each subsequent run costs only Dynare's solve
time.

Robustness: a model that hangs never appears as a finished ``<out>`` file, so
``run`` kills and restarts the session after the timeout and returns a
``timeout`` verdict — the next model gets a clean session.  Discovery,
result-shaping, and the graceful-degradation contract are reused verbatim from
:mod:`matlab_runner`, so a session verdict is drop-in compatible with
``run_dynare_matlab``.

    sess = DynareSession()
    if sess.start():
        v1 = sess.run(mod_text_1)   # pays startup
        v2 = sess.run(mod_text_2)   # solve-time only
        sess.close()
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from . import matlab_runner as mr

# Runner scripts live under ``matlab/`` in published builds and ``oracle/`` in
# the dev tree; reuse matlab_runner's resolver so both layouts work.
_SERVER_M = mr.runner_dir() / "dynare_session_server.m"


class DynareSession:
    """A reused MATLAB+Dynare process driven through a work directory."""

    def __init__(
        self,
        matlab_path: Optional[str] = None,
        dynare_path: Optional[str] = None,
        startup_timeout: int = 180,
    ):
        self._matlab = mr.find_matlab(matlab_path)
        self._dynare = mr.find_dynare(dynare_path)
        self.startup_timeout = startup_timeout
        self.workdir: Optional[Path] = None
        self.proc: Optional[subprocess.Popen] = None
        self._logf = None

    @property
    def available(self) -> bool:
        return bool(self._matlab and self._dynare and _SERVER_M.is_file())

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> bool:
        """Launch the session; return True once it signals READY."""
        matlab = self._matlab
        dynare = self._dynare
        if not matlab or not dynare or not _SERVER_M.is_file():
            return False
        self.workdir = Path(tempfile.mkdtemp(prefix="dynare_sess_"))
        ready = self.workdir / "READY"
        log_path = self.workdir / "session.log"
        self._logf = open(log_path, "w", encoding="utf-8", errors="replace")
        cmd = [
            matlab, "-batch",
            "dynare_session_server({},{})".format(
                mr._matlab_char_literal(self.workdir.as_posix()),
                mr._matlab_char_literal(Path(dynare).as_posix()),
            ),
        ]
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=str(_SERVER_M.parent), stdin=subprocess.DEVNULL,
                stdout=self._logf, stderr=subprocess.STDOUT,
                **mr._matlab_popen_kwargs(),
            )
        except OSError:
            self._cleanup()
            return False

        t0 = time.time()
        while time.time() - t0 < self.startup_timeout:
            if self.proc.poll() is not None:  # died during startup
                self._cleanup()
                return False
            if ready.exists():
                return True
            time.sleep(0.1)
        self.close()
        return False

    def _alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # -- run one model ------------------------------------------------------

    def run(
        self,
        mod_text: str,
        timeout: int = 180,
        active_file: Optional[str] = None,
        files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Run one ``.mod`` in the live session; return the verdict dict.

        Same contract as ``matlab_runner.run_dynare_matlab``.  Restarts the
        session on a hang so subsequent runs stay clean.
        """
        if not self._alive() and not self.start():
            return mr._unavailable(
                "MATLAB session unavailable.", matlab_available=bool(self._matlab),
            )
        if self.workdir is None:
            message = "MATLAB session work directory unavailable."
            return {
                "success": False, "matlab_available": bool(self._matlab),
                "dynare_available": bool(self._dynare), "status": "harness_error",
                "steady_state": {},
                "blanchard_kahn": mr._bk_unavailable(message),
                "errors": [message],
                "raw_log": self._tail_log(),
                "message": message,
            }

        # Leading letter: Dynare turns the .mod stem into a MATLAB function
        # name, which must not start with a digit.
        jid = "m" + uuid.uuid4().hex[:12]
        has_workspace_context = active_file is not None and files is not None
        job_dir = self.workdir / jid if has_workspace_context else None
        if job_dir is not None:
            assert active_file is not None
            assert files is not None
            mod_path = mr._materialize_workspace_files(
                job_dir,
                active_file,
                files,
                mod_text,
            )
        else:
            mod_path = self.workdir / f"{jid}.mod"
            mod_path.write_text(mod_text, encoding="utf-8")
        out_path = self.workdir / f"{jid}.out"
        job_tmp = self.workdir / f"{jid}.job.tmp"
        job_path = self.workdir / f"{jid}.job"
        job_tmp.write_text(
            json.dumps({"mod": mod_path.as_posix(), "out": out_path.as_posix()}),
            encoding="utf-8",
        )
        job_tmp.rename(job_path)  # atomic publish so the server never reads a partial job

        t0 = time.time()
        while time.time() - t0 < timeout:
            if out_path.exists():
                try:
                    rec = json.loads(out_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    for p in (mod_path, out_path):
                        try:
                            p.unlink()
                        except OSError:
                            pass
                    if job_dir is not None:
                        shutil.rmtree(job_dir, ignore_errors=True)
                    message = f"Could not parse MATLAB result JSON: {exc}"
                    return {
                        "success": False,
                        "matlab_available": True,
                        "dynare_available": True,
                        "status": "bad_json",
                        "steady_state": {},
                        "blanchard_kahn": mr._bk_unavailable(
                            "MATLAB result file was not valid JSON."
                        ),
                        "errors": [message],
                        "raw_log": self._tail_log(),
                        "message": message,
                    }
                for p in (mod_path, out_path):
                    try:
                        p.unlink()
                    except OSError:
                        pass
                if job_dir is not None:
                    shutil.rmtree(job_dir, ignore_errors=True)
                return mr._shape_from_record(rec, "")
            if not self._alive():
                message = "MATLAB session died during the run."
                return {
                    "success": False, "matlab_available": True,
                    "dynare_available": True, "status": "matlab_crash",
                    "steady_state": {},
                    "blanchard_kahn": mr._bk_unavailable("session died"),
                    "errors": [message],
                    "raw_log": self._tail_log(),
                    "message": message,
                }
            time.sleep(0.03)

        # Hang: kill + restart so the next job gets a clean session.
        self.close()
        self.start()
        return {
            "success": False, "matlab_available": True, "dynare_available": True,
            "status": "timeout", "steady_state": {},
            "blanchard_kahn": mr._bk_unavailable(f"timed out after {timeout}s"),
            "errors": [f"Model run timed out after {timeout}s (session restarted)."],
            "raw_log": "",
            "message": f"Model run timed out after {timeout}s (session restarted).",
        }

    # -- teardown -----------------------------------------------------------

    def _tail_log(self) -> str:
        workdir = self.workdir
        if workdir is None:
            return ""
        try:
            return (workdir / "session.log").read_text(
                encoding="utf-8", errors="replace")[-2000:]
        except OSError:
            return ""

    def close(self):
        try:
            if self.workdir is not None:
                (self.workdir / "STOP").write_text("", encoding="utf-8")
        except OSError:
            pass
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mr._kill_process_tree(self.proc)
                try:
                    self.proc.wait(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    pass
        self._cleanup()

    def _cleanup(self):
        if self._logf is not None:
            try:
                self._logf.close()
            except OSError:
                pass
            self._logf = None
        if self.workdir is not None:
            shutil.rmtree(self.workdir, ignore_errors=True)
            self.workdir = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()
