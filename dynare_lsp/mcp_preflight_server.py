"""Analysis-only MCP entrypoint with consolidated, snapshot-based preflight.

MATLAB execution is an explicit launch-time opt-in: --profile execution.
The legacy granular catalog remains available only through mcp_server.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

_DISCLAIMER = (
    "Preflight runs static analysis, the Dynare preprocessor when available, "
    "and LLMacro's local numerical checks. It does not execute Dynare/MATLAB "
    "and cannot guarantee that a full Dynare run will succeed."
)


def dynare_preflight(
    file_content: str,
    active_file: Optional[str] = None,
    files: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Check a frozen supplied-only workspace without executing MATLAB.

    Relative file keys are anchored to an absolute active_file's directory.
    Supply every include: missing buffers never silently fall back to disk.
    Numerical success requires explicit steady-state, residual, Jacobian and
    Blanchard-Kahn evidence. An installed preprocessor is also required to pass;
    an absent preprocessor is reported as unavailable, not as successful.
    Findings are capped per severity (200 errors, 100 warnings, 20 of each
    other severity); each stage's findings_summary gives totals, counts by
    severity, omitted counts by code and a truncated flag. ``configuration``
    is the analysis config fingerprinted in snapshot.id; pass it unchanged
    to dynare_validate_patch (which also infers it when config is omitted).
    """
    from .analysis_service import analyze, bound_report, json_safe
    from .preprocessor import find_preprocessor

    # Preserve preflight's optional-toolchain behavior while making its scope
    # explicit in requested_checks. A detected but failing/skipped preprocessor
    # may not be mistaken for a successful check.
    use_preprocessor = find_preprocessor() is not None
    config = {"numerical": True, "preprocessor": use_preprocessor}
    report = analyze(
        file_content,
        active_file or "model.mod",
        files,
        config,
    )
    if not use_preprocessor:
        report["stages"]["preprocessor"] = {
            "status": "unavailable",
            "reason": "Dynare preprocessor not found; not included in requested_checks.",
        }

    # Older preflight clients read ``message`` for environmental failures.
    # Retain it alongside the versioned service's more explicit ``reason``.
    for evidence in report["stages"].values():
        if evidence.get("status") == "unavailable" and "reason" in evidence:
            evidence.setdefault("message", evidence["reason"])

    # Keep the legacy one-based location fields alongside the versioned range.
    findings = report["stages"]["diagnostics"].get("findings", [])
    for row in findings:
        source_range = row.get("range")
        if source_range:
            row.update(
                line=source_range["start"]["line"] + 1,
                column=source_range["start"]["character"] + 1,
                end_line=source_range["end"]["line"] + 1,
                end_column=source_range["end"]["character"] + 1,
            )
    residuals = report["stages"]["residuals"]
    values = residuals.get("residuals", [])
    residuals["max_abs_residual"] = (
        max(abs(value) for value in values)
        if values and all(value is not None for value in values)
        else None
    )
    report["stages"]["jacobian"].setdefault("findings", [])
    preprocessor = report["stages"]["preprocessor"]
    preprocessor["n_diagnostics"] = len(preprocessor.get("findings", []))
    bound_report(report)
    report.update(
        preflight_passed=report["checks_passed"],
        warnings=[row for row in report["stages"]["diagnostics"].get("findings", [])
                  if row["severity"] == "WARNING"],
        configuration=config,
        disclaimer=_DISCLAIMER,
    )
    return json_safe(report)


def build_server(profile: str = "analysis", **limits):
    """Build the enforced profile, not the unrestricted legacy tool catalog."""
    from .mcp_profiles import build_server as build_profile

    return build_profile(profile, **limits)


def main(argv=None) -> None:
    """Run the bounded profile over stdio (analysis-only unless opted in)."""
    from .mcp_profiles import main as profile_main

    profile_main(argv)


if __name__ == "__main__":
    main()
