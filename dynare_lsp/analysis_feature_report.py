"""Shared analysis report transport bindings."""
from __future__ import annotations
from typing import Any
from .analysis_service import analyze


def dynare_analysis_report(file_content: str, active_file: str = "model.mod",
                           files: dict[str, str] | None = None,
                           config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Analyze a frozen supplied workspace without executing MATLAB.

    config accepts numerical/preprocessor booleans, tolerance, solve_budget
    (seconds, at most 60), and search_paths. Expensive stages are opt-in.
    active_file selects the entry model. Missing includes must be supplied.
    Results include a versioned schema, exact source/configuration fingerprint,
    and explicit passed/failed/not_run/unsupported/unavailable stage states.
    """
    return analyze(file_content, active_file, files, config)


TOOL = dynare_analysis_report
COMMAND = "dynare/analysisReport"
CLI = "report"
