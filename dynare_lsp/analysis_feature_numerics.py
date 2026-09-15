"""Inspectable numerical evidence; no automatic economic-model rewriting."""
from __future__ import annotations
import math
from typing import Any
from .analysis_service import json_safe, snapshot, stage


def eigenvalue_rows(values, criterion: float | None = None) -> list[dict[str, Any]]:
    rows = []
    for value in values:
        value = complex(value)
        finite = math.isfinite(value.real) and math.isfinite(value.imag)
        modulus = abs(value)
        rows.append({"real": json_safe(value.real), "imaginary": json_safe(value.imag),
                     "modulus": json_safe(modulus), "finite": finite,
                     "above_reported_cutoff": bool(modulus > criterion) if finite and criterion is not None else None})
    return rows


def matrix_evidence(matrix, variables: list[str], equations: list[str], relative_tolerance: float = 1e-8,
                    max_relations: int = 8) -> dict[str, Any]:
    """Expose the SVD basis, not a claim of uniquely identified causal relations."""
    import numpy as np
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape != (len(equations), len(variables)) or not array.size:
        return stage("unsupported", reason="Jacobian dimensions do not match the equation/variable labels.")
    if not np.all(np.isfinite(array)):
        return stage("unavailable", reason="Jacobian has non-finite entries.")
    if not math.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError("relative_tolerance must be finite and between 0 and 1")
    if isinstance(max_relations, bool) or not isinstance(max_relations, int) or max_relations < 1:
        raise ValueError("max_relations must be a positive integer")
    u, singular, vh = np.linalg.svd(array, full_matrices=True)
    cutoff = relative_tolerance * float(max(singular))
    rank = int(np.count_nonzero(singular > cutoff))
    def relations(vectors, labels, operator):
        output = []
        for vector in vectors[:max_relations]:
            # Remove arbitrary sign flips from the display; repeated singular
            # values still permit different, equally valid SVD bases.
            vector = vector.copy()
            if vector[int(np.argmax(np.abs(vector)))] < 0:
                vector *= -1
            output.append({"coefficients": dict(zip(labels, vector.tolist())),
                           "verification_residual_norm": float(np.linalg.norm(operator @ vector))})
        return output
    full_rank = array.shape[0] == array.shape[1] == rank
    return stage("passed" if full_rank else "failed", shape=list(array.shape), rank=rank,
                 singular_values=singular.tolist(), relative_tolerance=relative_tolerance, cutoff=cutoff,
                 variable_relations=relations(vh[rank:], variables, array),
                 equation_relations=relations(u[:, rank:].T, equations, array.T),
                 n_variable_relations=len(variables) - rank, n_equation_relations=len(equations) - rank,
                 relations_truncated=max(len(variables), len(equations)) - rank > max_relations,
                 interpretation="Numerical null-space basis at the stated values; not a unique causal explanation.")


def _source_location(snap, equation) -> dict[str, Any]:
    candidates = []
    for key, model in [(snap.entry_file, snap.root_model)] + list(snap.include_models.items()):
        physical = key if key in snap.files else key.rsplit("#", 1)[0]
        raw = snap.files.get(physical)
        if raw is None or (getattr(model, "original_text", "") or model.text) != raw:
            continue
        matches = [item for item in model.model_equations if item.text == equation.text and
                   getattr(item, "name", None) == getattr(equation, "name", None)]
        for item in matches:
            def offset(pos):
                lines = model.text.split("\n")
                return sum(len(s) + 1 for s in lines[:pos.line]) + pos.character
            start, end = offset(item.range.start), offset(item.range.end)
            mapping = getattr(model, "source_map", None)
            if mapping:
                start, end = mapping[min(start, len(mapping) - 1)], mapping[min(end, len(mapping) - 1)]
            if not 0 <= start <= end <= len(raw):
                continue
            def position(index):
                return {"line": raw.count("\n", 0, index), "character": index - raw.rfind("\n", 0, index) - 1}
            candidates.append({"file": physical, "start": position(start), "end": position(end),
                               "range_encoding": "zero-based Unicode code points"})
    unique = {str(item): item for item in candidates}
    return {"status": "mapped", **next(iter(unique.values()))} if len(unique) == 1 else {
        "status": "unmapped", "reason": "Original equation location is absent or ambiguous."}


def inspect_snapshot(snap, values: dict[str, float] | None, max_rows: int) -> dict[str, Any]:
    model = snap.model
    names = [item.name for item in model.endogenous]
    equations = list(model.static_model_equations())
    output = {name: stage("not_run") for name in ("steady_state", "residuals", "jacobian", "blanchard_kahn")}
    report = {"schema_version": "dynare-numerics/1", "snapshot": snap.manifest(), "stages": output,
              "execution_performed": False, "economic_fidelity": "not_assessed"}
    if not names or len(names) > 2000 or snap.unresolved or snap.cycles:
        output["steady_state"] = stage("unsupported", reason="Requires a resolved, nonempty model with at most 2000 endogenous variables.")
        return {**report, "success": False}
    current = "steady_state"
    try:
        if values is None:
            from .solver import compute_steady_state
            solved = compute_steady_state(model, time_budget=snap.config["solve_budget"])
            if not solved.success:
                output[current] = stage("failed", message=solved.message)
                return {**report, "success": False}
            values = dict(solved.values)
            provenance = "computed by LLMacro solver"
        else:
            provenance = "caller supplied; residual validation required"
        if set(values) != set(names):
            raise ValueError("Supply exactly the endogenous variables; no missing or unknown values")
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in values.values()):
            raise ValueError("Steady-state values must be finite real numbers")
        values = {name: float(values[name]) for name in names}
        output[current] = stage("passed", values=values, provenance=provenance, validation="See residual stage")
        current = "residuals"
        from .steady_state import validate_computed_steady_state
        residual_report = validate_computed_steady_state(model, values)
        residual_results = [item for item in residual_report.results if not item.is_local_var]
        rows = []
        for ordinal, item in enumerate(residual_results, 1):
            residual = item.residual
            finite = residual is not None and math.isfinite(float(residual))
            rows.append({"equation_id": f"static:{ordinal}", "name": item.equation.name,
                         "text": item.equation.text, "source": _source_location(snap, item.equation),
                         "residual": json_safe(residual), "finite": finite,
                         "satisfied": residual is not None and finite and abs(residual) <= snap.config["tolerance"],
                         "message": getattr(item, "error_message", "")})
        valid = len(rows) == len(equations) and bool(rows) and all(r["satisfied"] for r in rows)
        output[current] = stage("passed" if valid else "failed", tolerance=snap.config["tolerance"],
                                n_expected=len(equations), n_evaluated=len(rows), values_used=values,
                                rows=rows[:max_rows], rows_truncated=len(rows) > max_rows)
        output["steady_state"]["validation"] = "passed residual validation" if valid else "failed residual validation"
        if not valid:
            output["steady_state"]["status"] = "failed"
            return {**report, "success": False}
    except Exception as exc:
        output[current] = stage("unavailable", reason=f"{type(exc).__name__}: {exc}")
        return {**report, "success": False}
    # Inspect Jacobian and BK independently once residuals establish the point.
    # A rank-deficient static Jacobian can coincide with a deliberate unit root.
    try:
        from .bk_check import _compute_jacobian
        from .model_diagnostics import _static_jacobian_model
        matrix = sum(_compute_jacobian(_static_jacobian_model(model), values))
        output["jacobian"] = matrix_evidence(matrix, names, [f"static:{i + 1}" for i in range(len(equations))])
    except Exception as exc:
        output["jacobian"] = stage("unavailable", reason=f"{type(exc).__name__}: {exc}")
    try:
        from .bk_check import check_blanchard_kahn
        bk = check_blanchard_kahn(model, values)
        skipped = "check skipped" in (bk.message or "").lower()
        criterion = getattr(bk, "qz_criterium", None)
        eigenvalues = eigenvalue_rows(bk.eigenvalues, criterion)
        output["blanchard_kahn"] = stage("unsupported" if skipped else "passed" if bk.satisfied else "failed",
            message=bk.message, satisfied=None if skipped else bk.satisfied,
            n_unstable=bk.n_unstable, n_forward=bk.n_forward,
            qz_criterium=criterion, qz_zero_threshold=getattr(bk, "qz_zero_threshold", None),
            singular_qz=getattr(bk, "singular_qz", None),
            eigenvalues=eigenvalues[:max_rows], n_eigenvalues=len(eigenvalues), rows_truncated=len(eigenvalues) > max_rows,
            near_boundary_eigenvalues=eigenvalue_rows(getattr(bk, "near_boundary_eigenvalues", []), criterion),
            forward_variables=list(bk.forward_variables), predetermined_variables=list(bk.predetermined_variables))
    except Exception as exc:
        output["blanchard_kahn"] = stage("unavailable", reason=f"{type(exc).__name__}: {exc}")
    return json_safe({**report, "success": all(s["status"] == "passed" for s in output.values()),
                     "disclaimer": "Numerical evidence is local to these values and tolerances, not an instruction to change the model's economics."})


def dynare_numerical_evidence(file_content: str, active_file: str = "model.mod", files: dict[str, str] | None = None,
                              values: dict[str, float] | None = None, config: dict[str, Any] | None = None,
                              max_rows: int = 100) -> dict[str, Any]:
    """Inspect equation residuals, static-Jacobian SVD and BK eigenvalue evidence.

    Values may be supplied or computed once. Missing/unavailable evidence never
    means a pass. max_rows caps displayed residual/eigenvalue rows (1..500);
    validation always checks the complete model. Original source links are
    returned only when unambiguous. This tool never runs Dynare/MATLAB.
    """
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= 500:
        raise ValueError("max_rows must be an integer in [1, 500]")
    return inspect_snapshot(snapshot(file_content, active_file, files, config), values, max_rows)


TOOL = dynare_numerical_evidence
COMMAND = "dynare/numericalEvidence"
CLI = "numerics"
EXAMPLE = {"max_rows": 100}
