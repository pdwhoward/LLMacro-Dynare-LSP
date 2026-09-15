"""Versioned, snapshot-based analysis. Never executes MATLAB or writes source."""
from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from numbers import Integral, Real
from typing import Any

SCHEMA_VERSION = "dynare-analysis/1"
STAGES = ("diagnostics", "preprocessor", "steady_state", "residuals", "jacobian", "blanchard_kahn")
STATUSES = frozenset({"passed", "failed", "not_run", "unsupported", "unavailable"})
DEFAULT_CONFIG = {"tolerance": 1e-6, "solve_budget": 10.0, "numerical": False, "preprocessor": False, "search_paths": []}


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

    @property
    def snapshot_id(self) -> str:
        return fingerprint(self.files, self.config, self.entry_file)

    def manifest(self) -> dict[str, Any]:
        from . import __version__
        return {"id": self.snapshot_id, "entry_file": self.entry_file, "configuration": self.config,
                "source_hashes": {p: hashlib.sha256(t.encode("utf-8", "surrogatepass")).hexdigest() for p, t in sorted(self.files.items())},
                "engine_version": __version__, "file_policy": "supplied_only"}


def snapshot(file_content: str, active_file: str = "model.mod", files: dict[str, str] | None = None,
             config: dict[str, Any] | None = None) -> AnalysisSnapshot:
    """Freeze supplied buffers; missing includes never silently read local disk.

    active_file is the model entrypoint, not an ambiguous include fragment.
    Absolute entrypoints anchor relative map keys to their own directory.
    """
    from .diagnostics import model_with_include_context, _with_model_editing_commands
    from .workspace import WorkspaceIndex, _normalize_uri
    from .mcp_server import _rebase_relative_file_keys

    if not isinstance(file_content, str) or not isinstance(active_file, str) or not active_file:
        raise ValueError("file_content and a nonempty active_file are required")
    config = effective_config(config)
    supplied = _rebase_relative_file_keys(active_file, dict(files or {})) or dict(files or {})
    normalized: dict[str, str] = {}
    for filename, content in supplied.items():
        if not isinstance(filename, str) or not isinstance(content, str):
            raise ValueError("files must map filenames to source strings")
        key = _normalize_uri(filename)
        if key in normalized and normalized[key] != content:
            raise ValueError(f"Conflicting aliases for {filename}")
        normalized[key] = content
    entry = _normalize_uri(active_file)
    normalized[entry] = file_content  # The unsaved active buffer always wins.

    class SuppliedIndex(WorkspaceIndex):
        def _read_and_parse_from_disk(self, path):
            return self.get_model(str(path))

        def _resolve_directive(self, including_key, filename, active_search_paths=None):
            from .include_resolver import resolve_include_path
            search_paths, known_paths = self._directive_resolution_inputs(
                including_key, active_search_paths
            )
            return resolve_include_path(
                filename, including_key, search_paths, known_paths, known_only=True
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
                            index.find_unresolved_includes(entry), index.find_circular_includes(entry))


def _preprocess(snap: AnalysisSnapshot) -> tuple[dict[str, Any], Any]:
    """Preprocess a static supplied include tree in a private temporary mirror.

    Dynamic macro IO is conservatively unsupported: never read a dependency
    outside the fingerprint. Ordinary native analysis remains available.
    """
    from .parser import _strip_non_macro_comments
    from .preprocessor import find_preprocessor
    from . import preprocessor
    include_re = re.compile(r'^([ \t]*@#\s*include\s*)["\']([^"\'\r\n]+)["\'][ \t]*(?=\r?$)', re.MULTILINE)
    def macro_scan(content: str) -> str:
        scan = _strip_non_macro_comments(content)
        # Keep offsets stable while recognizing a directive after an initial BOM.
        return " " + scan[1:] if scan.startswith("\ufeff") else scan
    for content in snap.files.values():
        scan = macro_scan(content)
        remaining = include_re.sub("", scan)
        if re.search(r"(?:^|[\r\n])\s*@#", remaining):
            return stage("unsupported", reason="Dynamic macro preprocessing requires an explicit external workspace; no untracked dependencies are read."), None
    if snap.unresolved or snap.cycles:
        return stage("not_run", reason="Resolve supplied include dependencies first."), None
    executable = find_preprocessor()
    if not executable:
        return stage("unavailable", reason="Dynare preprocessor not found."), None
    with tempfile.TemporaryDirectory(prefix="dynare_analysis_") as folder:
        targets = {name: Path(folder) / f"source_{i}.mod" for i, name in enumerate(sorted(snap.files))}
        for name, content in snap.files.items():
            mappings = {}
            for directive, resolved, _context in snap.index.resolve_direct_includes(name):
                mappings[directive.filename] = targets.get(resolved)
            def rewrite(match):
                target = mappings.get(match.group(2))
                if target is None:
                    raise ValueError("Include is not part of the frozen snapshot")
                return f'{match.group(1)}"{target.as_posix()}"'
            # Only replace active include statements, not comments or quoted text.
            scan = macro_scan(content)
            edits = [(m.start(), m.end(), rewrite(m)) for m in include_re.finditer(scan)]
            for start, end, replacement in reversed(edits):
                content = content[:start] + replacement + content[end:]
            targets[name].write_text(content, encoding="utf-8", newline="")
        text = targets[snap.entry_file].read_text(encoding="utf-8")
        # Runtime installation adds the explicit cache-bypass keyword.
        run_preprocessor = getattr(preprocessor, "run_preprocessor")
        result = run_preprocessor(text, executable, source_dir=folder, timeout=30, use_cache=False)
        unavailable = any(d.code == "P000" for d in result.diagnostics)
        info = stage("unavailable" if unavailable else "passed" if result.success else "failed",
                     executable=executable, exit_code=result.exit_code,
                     findings=[diagnostic_row(d) for d in result.diagnostics])
        # Temporary paths are implementation details; attach stable source labels.
        for row in info["findings"]:
            for original, target in targets.items():
                row["message"] = row["message"].replace(str(target), original).replace(target.name, original)
        return info, result


def numerical_stages(model: Any, tolerance: float, solve_budget: float) -> dict[str, Any]:
    """Required numerical stages fail closed; an empty warning list is not proof."""
    output = {name: stage("not_run") for name in STAGES[2:]}
    current = "steady_state"
    try:
        from .solver import compute_steady_state
        steady = compute_steady_state(model, time_budget=solve_budget)
        values = {k: float(v) for k, v in steady.values.items()}
        valid = steady.success and bool(values) and all(math.isfinite(v) for v in values.values())
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
        n = len(model.endogenous)
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
        skipped = "check skipped" in (bk.message or "").lower()
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
