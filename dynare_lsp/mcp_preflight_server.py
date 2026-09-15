"""MCP entrypoint that adds one consolidated Dynare preflight tool.

The base MCP server remains the source of all granular tools.  This module adds
``dynare_preflight`` as an orchestration layer and is used by the console script
and bundled Claude Code plugin.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .diagnostics import (
    _with_model_editing_commands,
    model_with_include_context,
    run_diagnostics,
)
from .parser import parse
from .workspace import WorkspaceIndex

_DISCLAIMER = (
    "Preflight runs static analysis, the bundled Dynare preprocessor when "
    "available, and LLMacro's local numerical checks. It does not execute "
    "Dynare/MATLAB and cannot guarantee that a full Dynare run will succeed."
)


def _diagnostic_dict(diagnostic) -> Dict[str, Any]:
    return {
        "line": diagnostic.range.start.line + 1,
        "column": diagnostic.range.start.character + 1,
        "end_line": diagnostic.range.end.line + 1,
        "end_column": diagnostic.range.end.character + 1,
        "severity": diagnostic.severity.name,
        "code": diagnostic.code,
        "message": diagnostic.message,
    }


def _workspace_model(
    file_content: str,
    active_file: Optional[str],
    files: Optional[Dict[str, str]],
):
    if not active_file or not files:
        model = parse(file_content)
        return model, [], [], [], None

    workspace_files = dict(files)
    workspace_files[active_file] = file_content
    index = WorkspaceIndex()
    for filename, content in workspace_files.items():
        index.update_document(filename, content)
    model = index.get_effective_model(active_file) or parse(file_content)
    include_models_map = index.resolve_all_includes(active_file)
    include_models = list(include_models_map.values())
    include_symbols = index.collect_symbols(active_file)
    cycles = index.find_circular_includes(active_file)
    unresolved = index.find_unresolved_includes(active_file)
    return model, include_models, cycles, unresolved, include_symbols


def dynare_preflight(
    file_content: str,
    active_file: Optional[str] = None,
    files: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run the available non-execution analysis pipeline and return one result."""
    model, include_models, cycles, unresolved, include_symbols = _workspace_model(
        file_content, active_file, files
    )
    diagnostics = run_diagnostics(
        model,
        include_symbols=include_symbols,
        include_models=include_models,
        include_cycles=cycles,
        unresolved_includes=unresolved,
    )
    # Diagnostics own their include merge. Numerical stages need a separate,
    # once-merged view with the same editing commands as the granular tools.
    model = _with_model_editing_commands(
        model_with_include_context(model, include_models)
    )

    # Match the single-file LSP/CLI surface by reconciling with Dynare's pinned
    # preprocessor whenever that binary is available.  Multi-file preflight
    # still uses the same workspace resolver above; materializing arbitrary MCP
    # workspace paths merely to invoke the external binary would make this tool
    # stateful, so workspace preprocessor reconciliation is left to
    # dynare_diagnose_workspace.
    preprocessor_stage: Dict[str, Any] = {"status": "unavailable"}
    if not active_file or not files:
        try:
            from .preprocessor import (
                find_preprocessor,
                reconcile_diagnostics,
                run_preprocessor,
            )

            pp = find_preprocessor()
            if pp:
                pp_result = run_preprocessor(file_content, pp)
                diagnostics = reconcile_diagnostics(diagnostics, pp_result)
                preprocessor_stage = {
                    "status": "passed" if pp_result.success else "failed",
                    "exit_code": pp_result.exit_code,
                    "n_diagnostics": len(pp_result.diagnostics),
                }
        except Exception as exc:  # environmental/toolchain issue, not model failure
            preprocessor_stage = {"status": "unavailable", "message": str(exc)}

    diagnostic_rows = [_diagnostic_dict(d) for d in diagnostics]
    blocking = [row for row in diagnostic_rows if row["severity"] == "ERROR"]
    stages: Dict[str, Any] = {
        "diagnostics": {
            "status": "failed" if blocking else "passed",
            "findings": diagnostic_rows,
        },
        "preprocessor": preprocessor_stage,
    }
    if blocking:
        return {
            "preflight_passed": False,
            "blocking_stage": "diagnostics",
            "stages": stages,
            "disclaimer": _DISCLAIMER,
        }

    try:
        from .solver import compute_steady_state, default_solve_budget

        steady = compute_steady_state(model, time_budget=default_solve_budget())
    except Exception as exc:
        stages["steady_state"] = {"status": "unavailable", "message": str(exc)}
        return {
            "preflight_passed": False,
            "blocking_stage": "steady_state",
            "stages": stages,
            "disclaimer": _DISCLAIMER,
        }

    stages["steady_state"] = {
        "status": "passed" if steady.success else "failed",
        "message": steady.message,
        "values": dict(steady.values) if steady.success else {},
    }
    if not steady.success:
        return {
            "preflight_passed": False,
            "blocking_stage": "steady_state",
            "stages": stages,
            "disclaimer": _DISCLAIMER,
        }

    values = dict(steady.values)
    try:
        from .steady_state import validate_computed_steady_state

        residual_report = validate_computed_steady_state(model, values)
        max_residual = max(
            (
                abs(item.residual)
                for item in residual_report.results
                if item.residual is not None
            ),
            default=0.0,
        )
        residual_ok = all(
            item.is_satisfied
            for item in residual_report.results
            if not item.is_local_var
        )
        stages["residuals"] = {
            "status": "passed" if residual_ok else "failed",
            "max_abs_residual": max_residual,
        }
    except Exception as exc:
        stages["residuals"] = {"status": "unavailable", "message": str(exc)}
        residual_ok = False
    if not residual_ok:
        return {
            "preflight_passed": False,
            "blocking_stage": "residuals",
            "stages": stages,
            "disclaimer": _DISCLAIMER,
        }

    try:
        from .model_diagnostics import check_model_diagnostics

        jacobian_diags = check_model_diagnostics(model, values)
        jacobian_ok = not jacobian_diags
        stages["jacobian"] = {
            "status": "failed" if jacobian_diags else "passed",
            "findings": [_diagnostic_dict(d) for d in jacobian_diags],
        }
    except Exception as exc:
        jacobian_ok = False
        stages["jacobian"] = {"status": "unavailable", "message": str(exc)}
    if not jacobian_ok:
        return {
            "preflight_passed": False,
            "blocking_stage": "jacobian",
            "stages": stages,
            "disclaimer": _DISCLAIMER,
        }

    try:
        from .bk_check import check_blanchard_kahn

        bk = check_blanchard_kahn(model, values)
        skipped = (bk.message or "").lower().startswith("blanchard-kahn check skipped")
        stages["blanchard_kahn"] = {
            "status": "unavailable"
            if skipped
            else "passed"
            if bk.satisfied
            else "failed",
            "satisfied": bk.satisfied,
            "n_unstable": bk.n_unstable,
            "n_forward": bk.n_forward,
            "message": bk.message,
            "forward_variables": list(bk.forward_variables),
            "predetermined_variables": list(bk.predetermined_variables),
        }
        bk_ok = bk.satisfied and not skipped
    except Exception as exc:
        stages["blanchard_kahn"] = {"status": "unavailable", "message": str(exc)}
        bk_ok = False
    if not bk_ok:
        return {
            "preflight_passed": False,
            "blocking_stage": "blanchard_kahn",
            "stages": stages,
            "disclaimer": _DISCLAIMER,
        }

    warnings = [row for row in diagnostic_rows if row["severity"] == "WARNING"]
    return {
        "preflight_passed": True,
        "blocking_stage": None,
        "warnings": warnings,
        "stages": stages,
        "disclaimer": _DISCLAIMER,
    }


def build_server():
    """Build the normal MCP server and register the aggregate preflight tool."""
    from .mcp_server import build_server as build_base_server

    server = build_base_server()
    server.tool()(dynare_preflight)
    return server


def main() -> None:
    """Run the extended MCP server over stdio."""
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
