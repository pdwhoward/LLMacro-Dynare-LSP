"""Versioned, snapshot-based analysis. Never executes MATLAB or writes source."""
from __future__ import annotations

import hashlib
import json
import math
import re
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from numbers import Integral, Real
from typing import Any

from .steady_state import DYNARE_SOLVE_TOLF

SCHEMA_VERSION = "dynare-analysis/1"
STAGES = ("diagnostics", "preprocessor", "steady_state", "residuals", "jacobian", "blanchard_kahn")
STATUSES = frozenset({"passed", "failed", "not_run", "unsupported", "unavailable"})
DEFAULT_CONFIG = {"tolerance": DYNARE_SOLVE_TOLF, "solve_budget": 10.0, "numerical": False, "preprocessor": False, "search_paths": []}


def json_safe(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, complex):
        return {"real": json_safe(value.real), "imaginary": json_safe(value.imag)}
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def fingerprint(files: dict[str, str], config: dict[str, Any], entry_file: str) -> str:
    """Hash the exact UTF-8 text, filenames, entrypoint, and effective settings."""
    payload = {"schema": SCHEMA_VERSION, "entry_file": entry_file, "files": files, "config": config}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def effective_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    values = dict(DEFAULT_CONFIG)
    supplied = dict(config or {})
    unknown = set(supplied) - set(values)
    if unknown:
        raise ValueError(f"Unknown analysis settings: {', '.join(sorted(unknown))}")
    values.update(supplied)
    for name, maximum in (("tolerance", 1.0), ("solve_budget", 60.0)):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError(f"{name} must be finite and in (0, {maximum}]")
        values[name] = float(value)
    for name in ("numerical", "preprocessor"):
        if not isinstance(values[name], bool):
            raise ValueError(f"{name} must be boolean")
    paths = values["search_paths"]
    if not isinstance(paths, list) or not all(isinstance(p, str) and p for p in paths):
        raise ValueError("search_paths must be a list of nonempty path strings")
    values["search_paths"] = list(dict.fromkeys(paths))
    return values


def stage(status: str, **evidence: Any) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"Invalid analysis stage status: {status}")
    return {"status": status, **json_safe(evidence)}


def diagnostic_row(item: Any) -> dict[str, Any]:
    return {"code": item.code, "severity": item.severity.name, "message": item.message,
            "range": {"start": {"line": item.range.start.line, "character": item.range.start.character},
                      "end": {"line": item.range.end.line, "character": item.range.end.character}}}


@dataclass
class AnalysisSnapshot:
    entry_file: str
    files: dict[str, str]
    config: dict[str, Any]
    index: Any
    root_model: Any
    model: Any
    include_models: dict[str, Any]
    unresolved: list[Any]
    cycles: list[Any]
    # Normalized snapshot key -> the filename exactly as the caller supplied it.
    display_names: dict[str, str] = field(default_factory=dict)

    def display_name(self, key: str) -> str:
        return self.display_names.get(key, key)

    @property
    def snapshot_id(self) -> str:
        return fingerprint(self.files, self.config, self.entry_file)

    def manifest(self) -> dict[str, Any]:
        from . import __version__
        return {"id": self.snapshot_id, "entry_file": self.entry_file, "configuration": self.config,
                "source_hashes": {p: hashlib.sha256(t.encode("utf-8", "surrogatepass")).hexdigest() for p, t in sorted(self.files.items())},
                "engine_version": __version__, "file_policy": "supplied_only"}


def snapshot(file_content: str, active_file: str = "model.mod", files: dict[str, str] | None = None,
             config: dict[str, Any] | None = None, *,
             preserve_include_ranges: bool = False) -> AnalysisSnapshot:
    """Freeze supplied buffers; missing includes never silently read local disk.

    active_file is the model entrypoint, not an ambiguous include fragment.
    Absolute entrypoints anchor relative map keys to their own directory.
    Context retrieval can preserve include-local ranges instead of the
    synthetic expansion-order ranges required by ordinary analysis.
    """
    from .diagnostics import model_with_include_context, _with_model_editing_commands
    from .workspace import WorkspaceIndex, _normalize_uri
    from .mcp_server import _rebase_relative_file_keys

    if not isinstance(file_content, str) or not isinstance(active_file, str) or not active_file:
        raise ValueError("file_content and a nonempty active_file are required")
    config = effective_config(config)
    raw = dict(files or {})
    for filename, content in raw.items():
        if not isinstance(filename, str) or not isinstance(content, str):
            raise ValueError("files must map filenames to source strings")
    # Rebase an identity map so every anchored key remembers its caller name.
    identity = {name: name for name in raw}
    rebased = _rebase_relative_file_keys(active_file, identity) or identity
    normalized: dict[str, str] = {}
    display: dict[str, str] = {}
    for filename, original in rebased.items():
        content = raw[original]
        key = _normalize_uri(filename)
        if key in normalized and normalized[key] != content:
            raise ValueError(f"Conflicting aliases for {original}")
        normalized[key] = content
        display.setdefault(key, original)
    entry = _normalize_uri(active_file)
    normalized[entry] = file_content  # The unsaved active buffer always wins.
    display[entry] = active_file

    class SuppliedIndex(WorkspaceIndex):
        def _contextualize_include(self, *args, **kwargs):
            if preserve_include_ranges:
                kwargs["preserve_ranges"] = True
            return super()._contextualize_include(*args, **kwargs)

        def _read_and_parse_from_disk(self, path):
            return self.get_model(str(path))

        def _resolve_directive(self, including_key, filename, active_search_paths=None,
                               *, root_key=None, strict=False):
            from .include_resolver import resolve_include_path
            search_paths, known_paths = self._directive_resolution_inputs(
                including_key, active_search_paths
            )
            return resolve_include_path(
                filename, including_key, search_paths, known_paths, known_only=True,
                working_dir=self._working_dir_for_root(root_key), strict=strict,
            )

    paths = [Path(p) if Path(p).is_absolute() else Path(entry).parent / p for p in config["search_paths"]]
    index = SuppliedIndex(search_paths=paths)
    for filename, content in normalized.items():
        index.update_document(filename, content)
    includes = index.resolve_all_includes(entry)
    root = index.get_effective_model(entry)
    if root is None:
        raise ValueError("No parsed model for entrypoint")
    model = _with_model_editing_commands(model_with_include_context(root, list(includes.values()), include_model_equations=True))
    return AnalysisSnapshot(entry, normalized, config, index, root, model, includes,
                            index.find_unresolved_includes(entry), index.find_circular_includes(entry),
                            display_names=display)


_STATIC_INCLUDE = re.compile(r'^([ \t]*@#\s*include\s*)["\']([^"\'\r\n]+)["\'][ \t]*(?=\r?$)', re.MULTILINE)
_MACRO_DIRECTIVE = re.compile(r"(?:^|(?<=[\r\n]))[ \t]*@#\s*([A-Za-z_]\w*)")
# NUL and other C0/DEL controls (tab, LF, VT, FF and CR are whitespace) make
# the bundled preprocessor stall until its timeout; reject them up front.
_DISALLOWED_CONTROL = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")
_UNLOCATED_ERROR = re.compile(r"^\s*ERROR:\s*(.+?)\s*$")
_SENTINEL_RANGE = ((0, 0), (0, 1))


def _macro_scan(content: str) -> str:
    from .parser import _strip_non_macro_comments
    scan = _strip_non_macro_comments(content)
    # Keep offsets stable while recognizing a directive after an initial BOM.
    return " " + scan[1:] if scan.startswith("\ufeff") else scan


def dynamic_macro_io(content: str) -> str | None:
    """Name the first macro directive whose file IO is not statically known.

    ``@#define``, ``@#if``/``@#ifdef``/``@#ifndef``, ``@#for``, ``@#echo``,
    ``@#error`` and friends are evaluated in memory by the preprocessor and
    are safe. Only ``@#includepath`` and ``@#include`` directives whose
    filename is not a plain string literal (or is macro-expanded with
    ``@{...}``) can reach files outside the frozen snapshot.
    """
    scan = _macro_scan(content)
    static = {m.start() for m in _STATIC_INCLUDE.finditer(scan) if "@{" not in m.group(2)}
    for match in _MACRO_DIRECTIVE.finditer(scan):
        name = match.group(1).lower()
        if name == "includepath":
            return "@#includepath"
        if name == "include" and match.start() not in static:
            return "macro-expanded @#include"
    return None


def _position(text: str, offset: int) -> tuple[int, int]:
    line = text.count("\n", 0, offset)
    return line, offset - (text.rfind("\n", 0, offset) + 1)


def _entry_include_anchors(snap: AnalysisSnapshot) -> dict[str, Any]:
    """Map every reachable include to the entry-file directive that pulls it in."""
    anchors: dict[str, Any] = {}
    for directive, resolved, _context in snap.index.resolve_direct_includes(snap.entry_file):
        if resolved is None:
            continue
        pending, seen = [resolved], set()
        while pending:
            key = pending.pop()
            if key in seen:
                continue
            seen.add(key)
            anchors.setdefault(key, directive.range)
            pending.extend(r for _d, r, _c in snap.index.resolve_direct_includes(key) if r is not None)
    return anchors


def _control_character_result(snap: AnalysisSnapshot, sources: dict[str, str]) -> Any:
    """Return a failed preprocessor result locating disallowed control chars."""
    from .diagnostics import Diagnostic, Severity
    from .parser import Position, SourceRange
    from .preprocessor import PreprocessorResult
    anchors = None
    diagnostics = []
    for name in [snap.entry_file] + [n for n in sources if n != snap.entry_file]:
        match = _DISALLOWED_CONTROL.search(sources[name])
        if match is None:
            continue
        line, character = _position(sources[name], match.start())
        message = (f"Invalid control character U+{ord(match.group()):04X} in source; "
                   "Dynare cannot preprocess this file.")
        if name == snap.entry_file:
            source_range = SourceRange(Position(line, character), Position(line, character + 1))
        else:
            anchors = _entry_include_anchors(snap) if anchors is None else anchors
            message = f"[{snap.display_name(name)}:{line + 1}:{character + 1}] {message}"
            source_range = anchors.get(name) or SourceRange(Position(0, 0), Position(0, 1))
        diagnostics.append(Diagnostic(range=source_range, severity=Severity.ERROR, message=message,
                                      source="dynare-preprocessor", code=f"P{len(diagnostics) + 1:03d}"))
    if not diagnostics:
        return None
    return PreprocessorResult(success=False, diagnostics=diagnostics, raw_output="", preprocessor_path="")


def _relabel_preprocessor_output(snap: AnalysisSnapshot, result: Any, folder: str,
                                 targets: dict[str, Path]) -> None:
    """Replace private mirror paths with caller names, before reconciliation.

    The preprocessor reports mirror files as absolute or run-directory
    relative paths with either separator. Findings it could not locate in
    the entry file are anchored at the entry directive that includes the
    reported file, so output is deterministic across calls. A rejected run
    whose only complaint is an unlocated ``ERROR:`` line keeps that line.
    """
    from .diagnostics import Diagnostic, Severity
    from .parser import Position, SourceRange
    by_target = {target.name.lower(): name for name, target in targets.items()}
    mirror_path = re.compile(r"(?:[A-Za-z]:)?[^\s\[\]\"'<>|:]*?" + re.escape(Path(folder).name)
                             + r"[\\/]+(source_\d+\.mod)", re.IGNORECASE)
    exact = sorted(((form, name) for name, target in targets.items()
                    for form in {str(target), target.as_posix(), os.path.normcase(str(target))}),
                   key=lambda item: -len(item[0]))
    anchors: dict[str, Any] | None = None
    diagnostics = list(result.diagnostics)
    if (not result.success and getattr(result, "exit_code", None) not in (None, 0)
            and all(d.code == "P000" for d in diagnostics)):
        unlocated = [m.group(1) for line in (getattr(result, "raw_output", "") or "").splitlines()
                     if (m := _UNLOCATED_ERROR.match(line))]
        if unlocated:
            diagnostics = [Diagnostic(range=SourceRange(Position(0, 0), Position(0, 1)),
                                      severity=Severity.ERROR, message=message,
                                      source="dynare-preprocessor", code=f"P{i:03d}")
                           for i, message in enumerate(unlocated, 1)]
    for diag in diagnostics:
        mentioned: list[str] = []
        message = diag.message
        for form, name in exact:
            if form in message:
                message = message.replace(form, snap.display_name(name))
                mentioned.append(name)

        def replace(match: re.Match[str]) -> str:
            name = by_target.get(match.group(1).lower())
            if name is None:
                return match.group(0)
            mentioned.append(name)
            return snap.display_name(name)

        diag.message = mirror_path.sub(replace, message)
        start, end = diag.range.start, diag.range.end
        unlocated_range = ((start.line, start.character), (end.line, end.character)) == _SENTINEL_RANGE
        included = next((name for name in mentioned if name != snap.entry_file), None)
        if unlocated_range and included is not None:
            anchors = _entry_include_anchors(snap) if anchors is None else anchors
            if included in anchors:
                diag.range = anchors[included]
    result.diagnostics = diagnostics


def _preprocess(snap: AnalysisSnapshot) -> tuple[dict[str, Any], Any]:
    """Preprocess a static supplied include tree in a private temporary mirror.

    Macro directives evaluated in memory (``@#define``, ``@#if``, ``@#for``,
    ...) are supported. Dynamic macro IO (``@#includepath`` or a
    macro-expanded ``@#include``) is conservatively unsupported: never read a
    dependency outside the fingerprint. Only supplied files are mirrored and
    every static include is rewritten to its mirror, so an unsupplied include
    stays unresolved (E061) instead of being read from disk.
    """
    from .preprocessor import find_preprocessor
    from . import preprocessor
    # All supplied buffers stay in the snapshot fingerprint, but only the
    # entry's resolved include closure participates in preprocessing.
    reachable = {snap.entry_file}
    reachable.update(
        name if name in snap.files else name.rsplit("#", 1)[0]
        for name in snap.include_models
    )
    sources = {name: snap.files[name] for name in sorted(reachable)}
    for name, content in sources.items():
        dynamic = dynamic_macro_io(content)
        if dynamic:
            return stage("unsupported", reason=f"Dynamic macro IO ({dynamic} in {snap.display_name(name)}) "
                         "requires an explicit external workspace; no untracked dependencies are read."), None
    if snap.unresolved or snap.cycles:
        return stage("not_run", reason="Resolve supplied include dependencies first."), None
    invalid = _control_character_result(snap, sources)
    if invalid is not None:
        return stage("failed", reason="Source contains characters the Dynare preprocessor cannot read.",
                     executable=None, exit_code=None,
                     findings=[diagnostic_row(d) for d in invalid.diagnostics]), invalid
    executable = find_preprocessor()
    if not executable:
        return stage("unavailable", reason="Dynare preprocessor not found."), None
    with tempfile.TemporaryDirectory(prefix="dynare_analysis_") as folder:
        targets = {name: Path(folder) / f"source_{i}.mod" for i, name in enumerate(sources)}
        for name, content in sources.items():
            mappings = {}
            for directive, resolved, _context in snap.index.resolve_direct_includes(name):
                mappings[directive.filename] = targets.get(resolved)

            def rewrite(match):
                target = mappings.get(match.group(2))
                if target is None:
                    raise ValueError("Include is not part of the frozen snapshot")
                return f'{match.group(1)}"{target.as_posix()}"'

            # Only replace active include statements, not comments or quoted text.
            scan = _macro_scan(content)
            edits = [(m.start(), m.end(), rewrite(m)) for m in _STATIC_INCLUDE.finditer(scan)]
            for start, end, replacement in reversed(edits):
                content = content[:start] + replacement + content[end:]
            targets[name].write_text(content, encoding="utf-8", newline="")
        text = targets[snap.entry_file].read_text(encoding="utf-8")
        # Runtime installation adds the explicit cache-bypass keyword.
        run_preprocessor = getattr(preprocessor, "run_preprocessor")
        result = run_preprocessor(text, executable, source_dir=folder, timeout=30, use_cache=False)
        # Relabel before reconciliation so no private path reaches any stage.
        _relabel_preprocessor_output(snap, result, folder, targets)
    # P000 alone means the preprocessor never produced a verdict (spawn
    # failure or timeout). A non-zero exit from a run that happened is a
    # rejection, even when its message could not be located.
    ran = getattr(result, "exit_code", None) is not None
    unavailable = not ran and any(d.code == "P000" for d in result.diagnostics)
    info = stage("unavailable" if unavailable else "passed" if result.success else "failed",
                 executable=executable, exit_code=getattr(result, "exit_code", None),
                 findings=[diagnostic_row(d) for d in result.diagnostics])
    return info, result


def bk_skipped(bk: Any) -> bool:
    """A BK result without a verdict: unsatisfied yet no eigenvalues computed.

    Mirrors bk_check.bk_to_diagnostics (I071), so scipy absence or a failed
    eigenvalue computation is never reported as a genuine BK violation.
    """
    eigenvalues = getattr(bk, "eigenvalues", None)
    return not bk.satisfied and (eigenvalues is None or len(eigenvalues) == 0)


FINDING_LIMITS = {"ERROR": 200, "WARNING": 100}
OTHER_FINDING_LIMIT = 20


def limit_findings(rows: list[dict[str, Any]], limits: dict[str, int] | None = None,
                   other_limit: int = OTHER_FINDING_LIMIT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep findings in order up to a per-severity cap and summarize the rest."""
    limits = FINDING_LIMITS if limits is None else limits
    kept: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    omitted: Counter[str] = Counter()
    for row in rows:
        severity = str(row.get("severity"))
        counts[severity] += 1
        if counts[severity] <= limits.get(severity, other_limit):
            kept.append(row)
        else:
            omitted[str(row.get("code") or "")] += 1
    return kept, {"total": len(rows), "returned": len(kept), "truncated": len(kept) < len(rows),
                  "by_severity": dict(sorted(counts.items())), "omitted_by_code": dict(sorted(omitted.items())),
                  "limits": {**limits, "other": other_limit}}


def bound_report(report: dict[str, Any]) -> dict[str, Any]:
    """Cap every stage's findings in place, recording totals and truncation."""
    for evidence in report.get("stages", {}).values():
        rows = evidence.get("findings")
        if isinstance(rows, list):
            evidence["findings"], evidence["findings_summary"] = limit_findings(rows)
    return report


def unique_endogenous_model(model: Any) -> Any:
    """Return *model* with repeated endogenous declarations collapsed.

    Dynare only warns about ``var y c; var y;``; the variable is one unknown.
    A shallow copy keeps the snapshot's parsed model untouched.
    """
    import copy
    seen: set[str] = set()
    unique = [item for item in model.endogenous if not (item.name in seen or seen.add(item.name))]
    if len(unique) == len(model.endogenous):
        return model
    clone = copy.copy(model)
    clone.endogenous = unique
    return clone


def numerical_stages(model: Any, tolerance: float, solve_budget: float) -> dict[str, Any]:
    """Required numerical stages fail closed; an empty warning list is not proof."""
    model = unique_endogenous_model(model)
    output = {name: stage("not_run") for name in STAGES[2:]}
    current = "steady_state"
    try:
        from .solver import compute_steady_state
        steady = compute_steady_state(model, time_budget=solve_budget)
        values = {k: float(v) for k, v in steady.values.items()}
        valid = steady.success and bool(values) and all(math.isfinite(value) for value in values.values())
        output[current] = stage("passed" if valid else "failed", message=steady.message,
                                values=values if valid else {}, method=getattr(steady, "method_used", None))
        if not valid:
            return output
        current = "residuals"
        from .steady_state import validate_computed_steady_state
        residual_report = validate_computed_steady_state(model, values)
        equations = [r for r in residual_report.results if not r.is_local_var]
        expected = len(model.static_model_equations())
        valid = bool(equations) and len(equations) == expected and all(
            r.residual is not None and math.isfinite(float(r.residual)) and abs(float(r.residual)) <= tolerance for r in equations)
        output[current] = stage("passed" if valid else "failed", tolerance=tolerance,
                                n_expected=expected, n_evaluated=len(equations),
                                residuals=[json_safe(r.residual) for r in equations])
        if not valid:
            return output
        current = "jacobian"
        import numpy as np
        from .bk_check import _compute_jacobian
        from .model_diagnostics import _static_jacobian_model
        lagged, current_matrix, leading = _compute_jacobian(_static_jacobian_model(model), values)
        matrix = lagged + current_matrix + leading
        n = len({item.name for item in model.endogenous})
        if matrix.shape != (n, n) or n == 0:
            output[current] = stage("unsupported", reason="Static Jacobian is not a nonempty square system.")
            return output
        if not np.all(np.isfinite(matrix)):
            output[current] = stage("unavailable", reason="Static Jacobian contains non-finite entries.")
            return output
        singular = np.linalg.svd(matrix, compute_uv=False)
        cutoff = 1e-8 * float(max(singular))
        rank = int(np.count_nonzero(singular > cutoff))
        output[current] = stage("passed" if rank == n else "failed", rank=rank, dimension=n,
                                singular_values=singular.tolist(), relative_tolerance=1e-8, cutoff=cutoff)
        if rank != n:
            return output
        current = "blanchard_kahn"
        from .bk_check import check_blanchard_kahn
        bk = check_blanchard_kahn(model, values)
        skipped = bk_skipped(bk)
        output[current] = stage("unsupported" if skipped else "passed" if bk.satisfied else "failed",
                                satisfied=None if skipped else bool(bk.satisfied), message=bk.message,
                                n_unstable=bk.n_unstable, n_forward=bk.n_forward,
                                forward_variables=list(bk.forward_variables), predetermined_variables=list(bk.predetermined_variables))
    except Exception as exc:
        output[current] = stage("unavailable", reason=f"{type(exc).__name__}: {exc}")
    return output


def analyze_snapshot(snap: AnalysisSnapshot) -> dict[str, Any]:
    from .diagnostics import run_diagnostics
    requested = ["diagnostics"] + (["preprocessor"] if snap.config["preprocessor"] else [])
    if snap.config["numerical"]:
        requested.extend(STAGES[2:])
    stages = {name: stage("not_run", reason="Not requested or blocked by an earlier stage.") for name in STAGES}
    try:
        findings = run_diagnostics(snap.root_model, include_models=list(snap.include_models.values()),
                                  include_symbols=snap.index.collect_symbols(snap.entry_file),
                                  include_cycles=snap.cycles, unresolved_includes=snap.unresolved)
        if snap.config["preprocessor"]:
            try:
                stages["preprocessor"], pp = _preprocess(snap)
                if pp is not None:
                    from .preprocessor import reconcile_diagnostics
                    findings = reconcile_diagnostics(findings, pp)
            except Exception as exc:
                stages["preprocessor"] = stage("unavailable", reason=f"{type(exc).__name__}: {exc}")
        rows = [diagnostic_row(d) for d in findings]
        stages["diagnostics"] = stage("failed" if any(r["severity"] == "ERROR" for r in rows) else "passed", findings=rows,
                                      range_encoding="zero-based Unicode code points", source_file=snap.entry_file)
        ready = all(stages[name]["status"] == "passed" for name in requested if name in STAGES[:2])
        if snap.config["numerical"] and ready:
            stages.update(numerical_stages(snap.model, snap.config["tolerance"], snap.config["solve_budget"]))
    except Exception as exc:
        stages["diagnostics"] = stage("unavailable", reason=f"{type(exc).__name__}: {exc}")
    return finish_report(snap.manifest(), stages, requested)


def finish_report(manifest: dict[str, Any], stages: dict[str, Any], requested: list[str]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "snapshot": manifest, "freshness": "snapshot",
            "requested_checks": requested, "checks_passed": all(stages[s]["status"] == "passed" for s in requested),
            "blocking_stage": next((s for s in requested if stages[s]["status"] != "passed"), None),
            "stages": stages, "execution_performed": False,
            "disclaimer": "Passing the requested checks does not guarantee Dynare execution or economic fidelity."}


def analyze(file_content: str, active_file: str = "model.mod", files: dict[str, str] | None = None,
            config: dict[str, Any] | None = None) -> dict[str, Any]:
    return analyze_snapshot(snapshot(file_content, active_file, files, config))
