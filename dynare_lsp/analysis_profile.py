"""Deterministic profiling for a complete workspace diagnostic pass."""

from __future__ import annotations

from typing import Any, Dict

from .diagnostics import run_diagnostics
from .parser import parse
from .performance import capture_work
from .workspace import WorkspaceIndex, _normalize_uri


def _active_file_key(active_file: str, files: Dict[str, str]) -> str:
    if active_file in files:
        return active_file
    target = _normalize_uri(active_file)
    for candidate in files:
        if _normalize_uri(candidate) == target:
            return candidate
    raise ValueError(f"active_file {active_file!r} must appear in files keys")


def profile_workspace_analysis(
    active_file: str,
    files: Dict[str, str],
) -> Dict[str, Any]:
    """Run the parser/workspace/diagnostic pipeline and return stable counters.

    The report intentionally excludes wall-clock duration and preprocessor or
    solver execution.  It is suitable for regression budgets across machines
    because every metric is a deterministic count of analysis work.
    """
    active_file = _active_file_key(active_file, files)
    with capture_work() as profile:
        index = WorkspaceIndex()
        for filename, content in files.items():
            index.update_document(filename, content)

        include_symbols = index.collect_symbols(active_file)
        include_models = index.resolve_all_includes(active_file)
        include_cycles = index.find_circular_includes(active_file)
        unresolved = index.find_unresolved_includes(active_file)
        model = index.get_effective_model(active_file) or parse(files[active_file])
        diagnostics = run_diagnostics(
            model,
            include_symbols=include_symbols,
            include_models=list(include_models.values()),
            include_cycles=include_cycles,
            unresolved_includes=unresolved,
        )

    counters = profile.to_dict()
    operation_counters = {
        name: value
        for name, value in counters.items()
        if not name.endswith(".characters") and not name.endswith(".equations")
    }
    return {
        "active_file": active_file,
        "n_files_supplied": len(files),
        "n_files_included": len(include_models),
        "n_diagnostics": len(diagnostics),
        "diagnostic_codes": sorted(
            str(diagnostic.code) for diagnostic in diagnostics if diagnostic.code
        ),
        "work_units": counters,
        "total_operation_units": sum(operation_counters.values()),
        "scope": "parser, workspace, and native diagnostics; no preprocessor or solver",
    }
