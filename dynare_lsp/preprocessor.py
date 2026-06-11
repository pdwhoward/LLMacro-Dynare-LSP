"""Dynare preprocessor integration for deeper model validation.

Shells out to the ``dynare-preprocessor`` binary (bundled with Dynare)
for macro expansion, command compatibility, unused variables, etc.

Degrades gracefully if the preprocessor binary is not installed.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import List, Optional, Tuple

from .parser import (
    Position,
    SourceRange,
    _macro_branch_state,
    _mask_inactive_macro_lines,
    _parse_includes,
    _parse_macro_directives,
    _strip_non_macro_comments,
)
from .diagnostics import Diagnostic, Severity


@dataclass
class PreprocessorResult:
    """Result of running the Dynare preprocessor.

    ``raw_output`` is the legacy combined ``stderr + stdout`` text used by
    the diagnostic-reconciliation path.  ``stdout`` / ``stderr`` /
    ``exit_code`` expose the underlying process streams separately so a
    caller (e.g. the MCP ``dynare_run_preprocessor`` tool) can surface the
    preprocessor's real output verbatim.  They default to empty / ``None``
    so the timeout and exception constructors below stay valid.
    """

    success: bool
    diagnostics: List[Diagnostic]
    raw_output: str
    preprocessor_path: str
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None


# ---------------------------------------------------------------------------
# Preprocessor discovery
# ---------------------------------------------------------------------------


def _is_executable_file(path: str) -> bool:
    if not os.path.isfile(path):
        return False

    if platform.system() == "Windows":
        executable_exts = {
            ext.lower()
            for ext in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";")
            if ext
        }
        return os.path.splitext(path)[1].lower() in executable_exts

    return os.access(path, os.X_OK)


def _bundled_preprocessor() -> Optional[str]:
    """Return the dynare-preprocessor binary bundled inside this package, if any."""
    bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
    for name in ("dynare-preprocessor.exe", "dynare-preprocessor"):
        candidate = os.path.join(bin_dir, name)
        if _is_executable_file(candidate):
            return candidate
    return None


def find_preprocessor(configured_path: Optional[str] = None) -> Optional[str]:
    """Find the dynare-preprocessor binary.

    Search order:
      1. Explicit configured_path (from VS Code setting)
      2. DYNARE_PREPROCESSOR environment variable
      3. Bundled copy shipped inside this package (dynare_lsp/bin/)
      4. PATH via shutil.which
      5. Common install locations

    The bundled copy is preferred over a PATH hit so the LSP uses its own
    pinned, known-good preprocessor by default; users can still override
    explicitly via the setting or the environment variable.
    """
    # 1. Explicit path
    if configured_path and _is_executable_file(configured_path):
        return configured_path

    # 2. Environment variable
    env_path = os.environ.get("DYNARE_PREPROCESSOR")
    if env_path and _is_executable_file(env_path):
        return env_path

    # 3. Bundled copy shipped with the package (pinned, known-good)
    bundled = _bundled_preprocessor()
    if bundled:
        return bundled

    # 4. PATH
    which_result = shutil.which("dynare-preprocessor")
    if which_result:
        return which_result

    # 5. Common install locations
    system = platform.system()
    candidates: List[str] = []

    if system == "Windows":
        # Dynare installs under C:\Program Files\dynare\X.Y or C:\dynare\X.Y
        for dynare_root in ("C:\\Program Files\\dynare", "C:\\dynare"):
            if not os.path.isdir(dynare_root):
                continue
            try:
                for entry in sorted(os.listdir(dynare_root), reverse=True):
                    candidates.append(
                        os.path.join(
                            dynare_root,
                            entry,
                            "preprocessor",
                            "dynare-preprocessor.exe",
                        )
                    )
            except OSError:
                pass

    elif system == "Linux":
        candidates.extend(
            [
                "/usr/lib/dynare/preprocessor/dynare-preprocessor",
                "/usr/local/lib/dynare/preprocessor/dynare-preprocessor",
            ]
        )

    elif system == "Darwin":
        candidates.extend(
            [
                "/Applications/Dynare/preprocessor/dynare-preprocessor",
                "/usr/local/lib/dynare/preprocessor/dynare-preprocessor",
            ]
        )

    for c in candidates:
        if _is_executable_file(c):
            return c

    return None


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

# Match preprocessor error/warning lines:
# ERROR: foo.mod: line 10, col 5: some message
# ERROR: foo.mod: line 10, cols 5-12: some message
# ERROR: foo.mod: line 10, col 5 - line 11, col 3: some message
# WARNING: foo.mod: line 10: some message
# ERROR: /path with spaces/foo.mod: line 10: some message
# The filename is captured non-greedily up to ``: line N`` so paths with
# spaces are preserved.  The filename is reported back to the caller so
# diagnostics anchored on a transitively-included file can be rerouted
# rather than dumped onto the active document.
_PREPROC_LINE = re.compile(
    r"(ERROR|WARNING):\s*(.+?):\s*line\s+(\d+)"
    r"(?:,\s*(?:"
    r"col\s+(\d+)(?:\s*-\s*line\s+(\d+),\s*col\s+(\d+))?"
    r"|cols\s+(\d+)\s*-\s*(\d+)"
    r"))?"
    r":\s*(.+)"
)
_PREPROC_MACRO_PROCESSOR_LINE = re.compile(
    r"(?:"
    r"(ERROR|WARNING)\s+in\s+macro-processor:\s*(.+)"
    r"|Macro-processing error:\s*(.+)"
    r")",
    re.IGNORECASE,
)
_PREPROC_DIAG_LABEL_RE = re.compile(r"^\[(?P<label>.+):\d+:\d+\]\s")
# Backtrace bullets emitted after ``Macro-processing error: backtrace...``:
#   - @#error: "path" line 1, col 1-14
#   - binary operation: "path" line 1, col 8-10
#   - @#for: "path" line 1, col 1 to line 3, col 8
_PREPROC_BACKTRACE_LOCATION = re.compile(
    r'"([^"]+)"\s+line\s+(\d+),\s+col\s+(\d+)'
    r"(?:\s*-\s*(\d+)|\s+to\s+line\s+(\d+),\s+col\s+(\d+))?"
)
# ``ERROR in macro-processor: [@#directive: ]path:LINE.COL[-COL][: ]message``
# The separator after the location is a colon for most macro errors but a
# bare space for @#include failures ("...mod:1.1-27 Could not open f.inc").
_PREPROC_MACRO_INLINE_LOCATION = re.compile(
    r"^(?:(@#\w+):\s*)?(.*?):(\d+)\.(\d+)(?:-(\d+))?(?::|\s)\s*(.+)$"
)
_INCLUDE_DIRECTIVE_RE = re.compile(
    r"(@#\s*include\s*)([\"'])([^\"'\n]+)(\2)",
    re.IGNORECASE,
)


_MACRO_DEFINE_LINE_RE = re.compile(
    r"^[ \t]*@#\s*define\b.*$",
    re.IGNORECASE | re.MULTILINE,
)
_DOUBLE_QUOTED_RE = re.compile(r'"([^"\n]*)"')


def rewrite_supplied_absolute_includes(
    content: str,
    target_by_norm: "dict[str, str]",
) -> str:
    """Point absolute ``@#include`` paths at their temp-mirror copies.

    A caller-supplied workspace file can be keyed by an absolute path that
    ALSO exists on disk with different (stale) content.  The temp mirror
    materializes the supplied content elsewhere, but an absolute include
    directive would still make the preprocessor read the real disk file —
    so any absolute include whose normalized target has a mirror copy is
    rewritten to that copy's path.  ``@#include`` also accepts string
    EXPRESSIONS built from ``@#define``d paths (``@#include F`` or
    ``@#include P + "/x.mod"``), so absolute path literals inside
    ``@#define`` values are remapped too — both exact supplied-file paths
    and the supplied files' parent directories.
    """
    dir_by_norm: "dict[str, str]" = {}
    for norm_key, target in target_by_norm.items():
        parent = os.path.dirname(norm_key)
        if parent:
            dir_by_norm.setdefault(os.path.normcase(parent), os.path.dirname(target))

    def _map_absolute(raw: str) -> Optional[str]:
        raw = raw.strip()
        if not (PureWindowsPath(raw).is_absolute() or PurePosixPath(raw).is_absolute()):
            return None
        try:
            key = os.path.normcase(os.path.abspath(raw))
        except (OSError, ValueError):
            return None
        target = target_by_norm.get(key)
        if target is not None:
            return str(target).replace("\\", "/")
        mirror_dir = dir_by_norm.get(key)
        if mirror_dir is not None:
            return str(mirror_dir).replace("\\", "/")
        return None

    def _include_repl(m: "re.Match[str]") -> str:
        mapped = _map_absolute(m.group(3))
        if mapped is None:
            return m.group(0)
        return m.group(1) + m.group(2) + mapped + m.group(4)

    def _define_repl(m: "re.Match[str]") -> str:
        line = m.group(0)

        def _quoted_repl(qm: "re.Match[str]") -> str:
            raw_value = qm.group(1)
            mapped = _map_absolute(raw_value)
            if mapped is None:
                return qm.group(0)
            # ``@#define P = "C:/dir/"`` concatenated as P + "file.mod"
            # needs the trailing separator preserved (abspath strips it).
            if raw_value.rstrip()[-1:] in ("/", "\\") and not mapped.endswith("/"):
                mapped += "/"
            return f'"{mapped}"'

        return _DOUBLE_QUOTED_RE.sub(_quoted_repl, line)

    content = _INCLUDE_DIRECTIVE_RE.sub(_include_repl, content)
    return _MACRO_DEFINE_LINE_RE.sub(_define_repl, content)


def _slash_path(path: str) -> str:
    return path.replace("\\", "/")


def _strip_leading_relative_segments(path: str) -> str:
    normalized = _slash_path(path)
    while normalized.startswith("./"):
        normalized = normalized[2:]
    while normalized.startswith("../"):
        normalized = normalized[3:]
    return normalized


def _diagnostic_file_label(filename: str, synthetic_path: Optional[str]) -> str:
    """Return a compact, path-aware label for non-active preprocessor output."""
    if not filename:
        return ""
    if synthetic_path is None or filename == os.path.basename(filename):
        return _slash_path(os.path.basename(filename))

    filename_is_abs = (
        PureWindowsPath(filename).is_absolute() or PurePosixPath(filename).is_absolute()
    )
    filename_abs = os.path.abspath(filename) if filename_is_abs else None
    synthetic_dir = os.path.abspath(os.path.dirname(synthetic_path))
    if filename_abs is not None:
        try:
            return _slash_path(os.path.relpath(filename_abs, synthetic_dir))
        except ValueError:
            return _slash_path(filename_abs)
    return _slash_path(filename)


def diagnostic_message_matches_file(message: str, active_file: str) -> bool:
    """Whether a prefixed preprocessor diagnostic belongs to *active_file*.

    Directory-bearing prefixes must match by path suffix so two different
    included files with the same basename do not get conflated.  Bare-basename
    prefixes remain supported because some preprocessor builds only emit
    basenames for included files.
    """
    match = _PREPROC_DIAG_LABEL_RE.match(message)
    if match is None:
        return False
    label = _slash_path(match.group("label")).strip()
    if not label:
        return False

    active = _slash_path(active_file).strip()
    active_tail = _strip_leading_relative_segments(active)
    label_tail = _strip_leading_relative_segments(label)

    label_has_dir = "/" in label_tail
    if not label_has_dir:
        return os.path.normcase(os.path.basename(active_tail)) == os.path.normcase(
            label_tail,
        )

    active_norm = os.path.normcase(active_tail)
    label_norm = os.path.normcase(label_tail)
    return active_norm == label_norm or active_norm.endswith(f"/{label_norm}")


def _is_absolute_macro_path(raw_path: str) -> bool:
    """Whether a macro include path is clearly absolute before expansion."""
    path = raw_path.strip()
    if len(path) >= 2 and path[0] == path[-1] and path[0] in {"'", '"'}:
        path = path[1:-1].strip()
    if not path or "@{" in path:
        return False
    return PureWindowsPath(path).is_absolute() or PurePosixPath(path).is_absolute()


def _split_includepath_argument(argument: str) -> List[str]:
    raw = argument.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        raw = raw[1:-1]
    if not raw:
        return []
    parts: List[str] = []
    start = 0
    for index, char in enumerate(raw):
        if char != ":":
            continue
        # Match WorkspaceIndex behavior: split path lists on ":" while
        # preserving Windows drive prefixes such as C:/models.
        if (
            index == start + 1
            and raw[start].isalpha()
            and index + 1 < len(raw)
            and raw[index + 1] in "/\\"
        ):
            continue
        parts.append(raw[start:index].strip())
        start = index + 1
    parts.append(raw[start:].strip())
    return [part for part in parts if part]


def _requires_source_dir_file(mod_text: str) -> bool:
    """Whether Dynare needs the synthetic model beside source-relative includes."""
    macro_scan = _strip_non_macro_comments(mod_text)
    _defines, active_lines, _line_defines = _macro_branch_state(macro_scan)
    macro_scan = _mask_inactive_macro_lines(macro_scan, active_lines)
    if any(
        not _is_absolute_macro_path(include.filename)
        for include in _parse_includes(macro_scan)
    ):
        return True
    for directive in _parse_macro_directives(macro_scan):
        if directive.kind != "includepath" or directive.argument is None:
            continue
        parts = _split_includepath_argument(directive.argument)
        if not parts or any(not _is_absolute_macro_path(part) for part in parts):
            return True
    return False


def _macro_path_is_synthetic(
    path: str,
    synthetic_path: Optional[str],
    synthetic_abs: Optional[str],
) -> bool:
    """Whether a macro-error location refers to the synthetic tmp model."""
    base = os.path.basename(path)
    if path == base:
        # Bare basename — mirror the main-path heuristic: treat it as the
        # active document only when it matches the synthetic filename.
        if synthetic_abs is None:
            return base == "model.mod"
        return os.path.normcase(base) == os.path.normcase(
            os.path.basename(synthetic_path or "")
        )
    if synthetic_abs is None:
        return base == "model.mod"
    try:
        return os.path.normcase(os.path.abspath(path)) == synthetic_abs
    except (OSError, ValueError):
        return False


def _macro_location_to_range(
    loc: "re.Match[str]",
    message: str,
    synthetic_path: Optional[str],
    synthetic_abs: Optional[str],
) -> Tuple[str, SourceRange]:
    """Convert a backtrace ``"file" line N, col A[-B| to line M, col B]``."""
    line0 = max(int(loc.group(2)) - 1, 0)
    col0 = max(int(loc.group(3)) - 1, 0)
    if loc.group(5):
        end_line0 = max(int(loc.group(5)) - 1, 0)
        end_col0 = max(int(loc.group(6)), 0)
    elif loc.group(4):
        end_line0 = line0
        end_col0 = max(int(loc.group(4)), col0 + 1)
    else:
        end_line0 = line0
        end_col0 = col0 + 1
    if _macro_path_is_synthetic(loc.group(1), synthetic_path, synthetic_abs):
        return message, SourceRange(
            Position(line0, col0), Position(end_line0, end_col0)
        )
    label = _diagnostic_file_label(loc.group(1), synthetic_path)
    return (
        f"[{label}:{line0 + 1}:{col0 + 1}] {message}",
        SourceRange(Position(0, 0), Position(0, 1)),
    )


def _macro_inline_location_to_range(
    inline: "re.Match[str]",
    synthetic_path: Optional[str],
    synthetic_abs: Optional[str],
) -> Tuple[str, SourceRange]:
    """Convert ``path:LINE.COL[-COL]: msg`` from ``ERROR in macro-processor``."""
    directive = inline.group(1)
    path = inline.group(2).strip()
    line0 = max(int(inline.group(3)) - 1, 0)
    col0 = max(int(inline.group(4)) - 1, 0)
    end_col0 = int(inline.group(5)) if inline.group(5) else col0 + 1
    message = inline.group(6).strip()
    # The model is checked from an in-memory temp copy, so the searched-
    # directories tail names only meaningless temp paths.
    message = re.sub(r"\s*The following directories were searched:\s*$", "", message)
    if directive:
        message = f"{directive}: {message}"
    if _macro_path_is_synthetic(path, synthetic_path, synthetic_abs):
        return message, SourceRange(
            Position(line0, col0), Position(line0, max(end_col0, col0 + 1))
        )
    label = _diagnostic_file_label(path, synthetic_path)
    return (
        f"[{label}:{line0 + 1}:{col0 + 1}] {message}",
        SourceRange(Position(0, 0), Position(0, 1)),
    )


def _parse_preprocessor_output(
    output: str,
    synthetic_path: Optional[str] = None,
) -> List[Diagnostic]:
    """Parse preprocessor stderr into Diagnostic objects.

    *synthetic_path*, when provided, is the absolute path of the tmp
    ``model.mod`` file the runner created.  Diagnostics from any other
    file (including a real user file that happens to be named
    ``model.mod`` somewhere else on disk) get a basename prefix so the
    user sees which file owns the error.  Without this disambiguation,
    a real ``/project/sub/model.mod`` error would be silently
    anchored to the active document.
    """
    diagnostics: List[Diagnostic] = []
    code_counter = 1
    import os as _os

    synthetic_abs = (
        _os.path.normcase(_os.path.abspath(synthetic_path)) if synthetic_path else None
    )

    output_lines = output.splitlines()
    idx = 0
    while idx < len(output_lines):
        line = output_lines[idx]
        idx += 1
        stripped = line.strip()
        m = _PREPROC_LINE.match(stripped)
        if not m:
            macro_m = _PREPROC_MACRO_PROCESSOR_LINE.match(stripped)
            if not macro_m:
                continue
            level = (macro_m.group(1) or "ERROR").upper()
            message = (macro_m.group(2) or macro_m.group(3) or "").strip()
            rng = SourceRange(Position(0, 0), Position(0, 1))
            if message.lower().startswith("backtrace"):
                # ``Macro-processing error: backtrace...`` is followed by
                # ``- <cause>`` and ``- <frame>: "file" line N, col A-B``
                # bullet lines; fold them into one diagnostic carrying the
                # real cause and the innermost source location instead of
                # the literal message "backtrace...".
                causes: List[str] = []
                located: Optional["re.Match[str]"] = None
                inline_rng: Optional[SourceRange] = None
                while idx < len(output_lines):
                    bullet = output_lines[idx].strip()
                    if not bullet.startswith("-"):
                        break
                    idx += 1
                    bullet_text = bullet.lstrip("-").strip()
                    loc = _PREPROC_BACKTRACE_LOCATION.search(bullet_text)
                    if loc is not None:
                        if located is None:
                            located = loc
                        continue
                    # @#include failures emit inline-format bullets
                    # (``- @#include: path:1.1-27 Could not open f.inc``);
                    # parse them so the cause is the clean message, not the
                    # raw bullet with the synthetic temp path baked in.
                    inline_bullet = _PREPROC_MACRO_INLINE_LOCATION.match(bullet_text)
                    if inline_bullet is not None:
                        cause, bullet_rng = _macro_inline_location_to_range(
                            inline_bullet, synthetic_path, synthetic_abs
                        )
                        if inline_rng is None:
                            inline_rng = bullet_rng
                        if cause:
                            causes.append(cause)
                        continue
                    if bullet_text:
                        causes.append(bullet_text)
                if causes:
                    message = "; ".join(causes)
                if located is not None:
                    message, rng = _macro_location_to_range(
                        located, message, synthetic_path, synthetic_abs
                    )
                elif inline_rng is not None:
                    rng = inline_rng
            else:
                inline = _PREPROC_MACRO_INLINE_LOCATION.match(message)
                if inline is not None:
                    message, rng = _macro_inline_location_to_range(
                        inline, synthetic_path, synthetic_abs
                    )
            severity = Severity.ERROR if level == "ERROR" else Severity.WARNING
            code = f"P{code_counter:03d}"
            code_counter += 1
            diagnostics.append(
                Diagnostic(
                    range=rng,
                    severity=severity,
                    message=message,
                    source="dynare-preprocessor",
                    code=code,
                )
            )
            continue

        level = m.group(1)
        filename = m.group(2).strip()
        line_no = max(int(m.group(3)) - 1, 0)  # convert to 0-based
        if m.group(7):
            col = int(m.group(7)) - 1
            end_line = line_no
            end_col = int(m.group(8))
        else:
            col = int(m.group(4)) - 1 if m.group(4) else 0
            end_line = max(int(m.group(5)) - 1, 0) if m.group(5) else line_no
            end_col = int(m.group(6)) if m.group(6) else col + 1
        col = max(col, 0)
        end_col = max(end_col, col + 1) if end_line == line_no else max(end_col, 0)
        message = m.group(9).strip()
        # If the error lives in a file other than the synthetic tmp
        # model, prefix the message so the user sees which file the
        # diagnostic targets — anchor remains on the active document
        # since we don't have a way to publish to a non-open file from
        # a single-doc check.  We use the FULL synthetic path (when
        # supplied) to disambiguate so a real user file that happens to
        # be named ``model.mod`` somewhere on disk doesn't get treated
        # as the synthetic temp file.  Fallback: basename match for
        # callers that don't supply the synthetic path.
        is_synthetic = True
        if filename:
            from pathlib import Path as _Path

            basename = _Path(filename).name
            if synthetic_abs is not None:
                synthetic_name = _Path(synthetic_path).name if synthetic_path else ""
                if filename == basename:
                    # Some preprocessor builds report only ``model.mod``
                    # even when invoked with an absolute temp path.  A
                    # bare basename has no cross-file location to route,
                    # so treat the synthetic temp filename as the active
                    # document while keeping directory-bearing paths
                    # fully disambiguated below.
                    is_synthetic = _os.path.normcase(basename) == _os.path.normcase(
                        synthetic_name
                    )
                else:
                    is_synthetic = (
                        _os.path.normcase(_os.path.abspath(filename)) == synthetic_abs
                    )
            else:
                is_synthetic = basename == "model.mod"
            if not is_synthetic:
                label = _diagnostic_file_label(filename, synthetic_path)
                message = f"[{label}:{line_no + 1}:{col + 1}] {message}"

        severity = Severity.ERROR if level == "ERROR" else Severity.WARNING
        code = f"P{code_counter:03d}"
        code_counter += 1

        # When the error comes from an included file, the line/col in
        # the preprocessor's output refer to THAT file, not the active
        # document we publish diagnostics against.  Anchor the
        # diagnostic at (0, 0) of the active document and keep the
        # real source location in the message text instead — otherwise
        # an editor would point users to the wrong line number in the
        # wrong file.
        if not is_synthetic:
            rng = SourceRange(
                Position(0, 0),
                Position(0, 1),
            )
        else:
            rng = SourceRange(
                Position(line_no, col),
                Position(end_line, end_col),
            )
        diagnostics.append(
            Diagnostic(
                range=rng,
                severity=severity,
                message=message,
                source="dynare-preprocessor",
                code=code,
            )
        )

    return diagnostics


# ---------------------------------------------------------------------------
# Run preprocessor
# ---------------------------------------------------------------------------


def run_preprocessor(
    mod_text: str,
    preprocessor_path: str,
    timeout: int = 30,
    source_dir: Optional[str] = None,
) -> PreprocessorResult:
    """Run the Dynare preprocessor on model text.

    Writes text to a temp file, runs the preprocessor in check mode,
    parses output, and cleans up.

    *source_dir* is the directory of the actual document being checked,
    if known.  When the model has active source-relative ``@#include`` or
    ``@#includepath`` directives, the synthetic file is placed there and the
    preprocessor runs with ``cwd=source_dir`` so those paths resolve against
    the user's real workspace.  Models with no source-relative include search
    needs stay entirely in the system temp directory to avoid source-tree
    artifact churn.
    """
    if mod_text.startswith("\ufeff"):
        mod_text = mod_text[1:]

    tmp_dir = None
    tmp_file = None
    generated_root = None
    source_dir_abs: Optional[str] = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="dynare_lsp_")
        source_dir_abs = (
            os.path.abspath(source_dir)
            if source_dir and os.path.isdir(source_dir)
            else None
        )
        use_source_dir_file = source_dir_abs is not None and _requires_source_dir_file(
            mod_text
        )
        run_cwd = source_dir_abs if use_source_dir_file else tmp_dir
        if use_source_dir_file:
            assert source_dir_abs is not None
            fd, tmp_file = tempfile.mkstemp(
                prefix=".dynare_lsp_",
                suffix=".mod",
                dir=source_dir_abs,
            )
            os.close(fd)
            generated_root = os.path.join(
                source_dir_abs,
                os.path.splitext(os.path.basename(tmp_file))[0],
            )
        else:
            tmp_file = os.path.join(tmp_dir, "model.mod")
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(mod_text)

        cmd = [
            preprocessor_path,
            tmp_file,
            "json=check",
            "onlyjson",
            "nopreprocessoroutput",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=run_cwd,
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        raw_output = stderr + stdout
        diagnostics = _parse_preprocessor_output(
            raw_output,
            synthetic_path=tmp_file,
        )

        return PreprocessorResult(
            success=result.returncode == 0,
            diagnostics=diagnostics,
            raw_output=raw_output,
            preprocessor_path=preprocessor_path,
            stdout=stdout,
            stderr=stderr,
            exit_code=result.returncode,
        )

    except subprocess.TimeoutExpired:
        return PreprocessorResult(
            success=False,
            diagnostics=[
                Diagnostic(
                    range=SourceRange(Position(0, 0), Position(0, 1)),
                    severity=Severity.WARNING,
                    message=f"Dynare preprocessor timed out after {timeout}s",
                    source="dynare-preprocessor",
                    code="P000",
                )
            ],
            raw_output="",
            preprocessor_path=preprocessor_path,
        )
    except Exception as e:
        message = f"Could not run Dynare preprocessor '{preprocessor_path}': {e}"
        return PreprocessorResult(
            success=False,
            diagnostics=[
                Diagnostic(
                    range=SourceRange(Position(0, 0), Position(0, 1)),
                    severity=Severity.WARNING,
                    message=message,
                    source="dynare-preprocessor",
                    code="P000",
                )
            ],
            raw_output=str(e),
            preprocessor_path=preprocessor_path,
        )
    finally:
        if generated_root:
            try:
                if os.path.isdir(generated_root):
                    shutil.rmtree(generated_root, ignore_errors=True)
                elif os.path.exists(generated_root):
                    os.remove(generated_root)
            except OSError:
                pass
        if (
            tmp_file
            and source_dir_abs
            and os.path.normcase(os.path.abspath(os.path.dirname(tmp_file)))
            == os.path.normcase(source_dir_abs)
        ):
            try:
                os.remove(tmp_file)
            except OSError:
                pass
        if tmp_dir:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass


def _diagnostic_to_struct(d: Diagnostic) -> dict:
    """Render a Diagnostic as the machine-readable dict the MCP tool returns.

    Mirrors the LSP convention of 1-based line/column numbers and the
    severity *name* (``ERROR`` / ``WARNING`` / ``INFORMATION`` / ``HINT``)
    so the shape matches the other dynare-mcp diagnostic surfaces.
    """
    return {
        "line": d.range.start.line + 1,
        "column": d.range.start.character + 1,
        "severity": d.severity.name,
        "message": d.message,
        "code": d.code,
    }


def run_preprocessor_structured(
    mod_text: str,
    source_dir: Optional[str] = None,
    timeout: int = 30,
    configured_path: Optional[str] = None,
) -> dict:
    """Execute the bundled Dynare preprocessor and return a structured result.

    This is the reusable entry point behind the ``dynare_run_preprocessor``
    MCP tool.  It locates the preprocessor with :func:`find_preprocessor`
    (which prefers the pinned binary bundled under ``dynare_lsp/bin/``) and
    runs it through :func:`run_preprocessor`, so process spawning, the
    ``json=check onlyjson nopreprocessoroutput`` invocation, workspace
    artifact cleanup, and stderr parsing are *not* re-implemented here.

    Args:
        mod_text: Full text of a Dynare ``.mod`` file.
        source_dir: Directory the document lives in, when known, so
            ``@#include`` directives resolve against the real workspace.
        timeout: Seconds before the preprocessor run is abandoned.
        configured_path: Explicit preprocessor path override; falls back to
            the bundled / environment / PATH discovery in
            :func:`find_preprocessor`.

    Returns:
        On success or a clean failure (the binary ran and emitted
        diagnostics)::

            {
                "success": bool,        # preprocessor exit code == 0
                "exit_code": int|None,  # raw process exit code
                "diagnostics": [        # parsed from preprocessor stderr
                    {"line", "column", "severity", "message", "code"}, ...
                ],
                "raw_stdout": str,
                "raw_stderr": str,
            }

        When the bundled/discoverable preprocessor binary is unavailable::

            {"success": False, "message": "...", "diagnostics": []}
    """
    preprocessor_path = find_preprocessor(configured_path)
    if not preprocessor_path:
        return {
            "success": False,
            "message": (
                "Dynare preprocessor binary not found; it ships bundled "
                "under dynare_lsp/bin/ but could not be located. Set the "
                "DYNARE_PREPROCESSOR environment variable to override."
            ),
            "diagnostics": [],
        }

    result = run_preprocessor(
        mod_text,
        preprocessor_path,
        timeout=timeout,
        source_dir=source_dir,
    )
    return {
        "success": bool(result.success),
        "exit_code": result.exit_code,
        "diagnostics": [_diagnostic_to_struct(d) for d in result.diagnostics],
        "raw_stdout": result.stdout,
        "raw_stderr": result.stderr,
    }


# ---------------------------------------------------------------------------
# Diagnostic reconciliation (defer to the authoritative Dynare parser)
# ---------------------------------------------------------------------------

# Diagnostic codes the Dynare preprocessor authoritatively owns: parsing,
# identifiers, declarations, and equation/variable structure.  When the
# preprocessor reports errors we defer to its precise messages for these
# classes rather than our own parser's.
_PREPROCESSOR_AUTHORITATIVE_CODES = frozenset(
    {
        "E001",
        "E010",
        "E020",
        "E030",
        "E052",
        "E061",
    }
)

# Structural ERROR codes the preprocessor does NOT enforce at parse time, so its
# acceptance does not make them false positives. Equation-count balance (E010) is
# only checked by Dynare at the solve / ``check`` stage -- exactly like the
# clean-room Blanchard-Kahn verdict, which we already keep on success. E010 is
# additionally suppressed upstream (run_diagnostics) for OccBin, optimal-policy,
# and macro-template/branch models, so when it reaches here our static count is
# faithful and a surviving E010 is a true "this model is non-square; Dynare will
# fail at solve time" signal, not a parser false positive.
_SURVIVES_PREPROCESSOR_SUCCESS = frozenset({"E010"})


def reconcile_diagnostics(
    own_diagnostics: List[Diagnostic],
    preproc_result: Optional["PreprocessorResult"],
) -> List[Diagnostic]:
    """Combine our diagnostics with the Dynare preprocessor's, deferring to it.

    The preprocessor is Dynare's authoritative front-end:

    * If it ACCEPTS the model, any hard error from our parser is a false
      positive (Dynare compiles the file), so our ``ERROR``-severity
      diagnostics are dropped, keeping only value-add (warnings, hints,
      steady-state, Blanchard-Kahn) plus any preprocessor warnings.  The one
      exception is ``_SURVIVES_PREPROCESSOR_SUCCESS`` (equation-count balance,
      E010): the preprocessor accepts non-square models and only Dynare's
      solve/``check`` stage rejects them, so that ERROR is a true positive even
      on success.
    * If it REJECTS the model, its precise parse/declaration errors supersede
      our structural-class diagnostics.

    With no preprocessor available the caller's diagnostics pass through
    unchanged -- the while-typing / graceful-degradation fallback.
    """
    if preproc_result is None:
        return list(own_diagnostics)

    if preproc_result.success:
        kept = [
            d
            for d in own_diagnostics
            if d.severity != Severity.ERROR or d.code in _SURVIVES_PREPROCESSOR_SUCCESS
        ]
    elif any(d.code != "P000" for d in preproc_result.diagnostics):
        # The preprocessor genuinely REJECTED the model and produced precise
        # parse/declaration errors -> defer to it for the structural classes.
        kept = [
            d
            for d in own_diagnostics
            if d.code not in _PREPROCESSOR_AUTHORITATIVE_CODES
        ]
    else:
        # success=False but the ONLY diagnostic is the synthetic ``P000`` sentinel
        # (timeout / spawn failure / crash): the preprocessor never actually ran,
        # so there is no authoritative verdict to defer to. Keep ALL our own
        # diagnostics -- dropping them here would silently clear a broken model
        # (e.g. a large file that times out the preprocessor) and show "no issues".
        kept = list(own_diagnostics)
    return kept + list(preproc_result.diagnostics)
