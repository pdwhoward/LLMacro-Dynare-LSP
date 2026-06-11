"""First-order structural identification checks for Dynare models.

This module checks whether calibrated parameters have distinguishable local
effects on steady-state residuals and the first-order structural Jacobian.
It is intentionally narrower than data/moment-based identification tests.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, cast

from .bk_check import _compute_jacobian
from .diagnostics import Diagnostic, Severity
from .parser import ParsedModel, Position, SourceRange
from .steady_state import validate_computed_steady_state


IDENTIFICATION_CODE = "W081"
_PARAM_STEP = 1e-6
_NULL_LOADING_TOL = 1e-5


def check_identification(
    model: ParsedModel,
    ss_values: dict,
    tol: float = 1e-7,
) -> List[Diagnostic]:
    """Check first-order structural parameter identification.

    Returns W081 warnings for parameters with no first-order effect and for
    groups whose local effect vectors are collinear or linearly dependent.
    The check uses only model structure at the supplied steady state.
    """
    try:
        import numpy as np
    except ImportError:
        return []

    endogenous = list(model.endogenous)
    equations = model.dynamic_model_equations()
    n_endo = len(endogenous)
    if n_endo == 0 or len(equations) != n_endo:
        return []

    try:
        from .solver import _effective_param_values

        param_values = _effective_param_values(model)
    except Exception:
        param_values = model.param_values()

    parameter_names = [
        parameter.name
        for parameter in model.parameters
        if _is_known_numeric(param_values.get(parameter.name))
    ]
    # Restrict to parameters that appear in the model equations.  Parameters
    # absent from the model (e.g. shock-variance params used only in the shocks
    # block) have no first-order structural effect by construction and are
    # identified from data, not model structure -- flagging them would be noise
    # and overlaps the unused-parameter check (W022).
    from .diagnostics import _extract_references
    declared = model.all_declared_names()
    model_referenced: set = set()
    for equation in model.dynamic_model_equations():
        model_referenced.update(_extract_references(equation.text, declared))
    parameter_names = [name for name in parameter_names if name in model_referenced]
    if not parameter_names:
        return []

    try:
        baseline = _effect_basis(model, ss_values, None, np, n_endo)
        columns = []
        for name in parameter_names:
            value = float(param_values[name])
            h = max(abs(value), 1.0) * _PARAM_STEP
            perturbed = _effect_basis(
                model,
                ss_values,
                {name: value + h},
                np,
                n_endo,
            )
            if perturbed.shape != baseline.shape:
                return []
            columns.append((perturbed - baseline) / h)
    except Exception:
        return []

    if not columns:
        return []

    matrix = np.column_stack(columns)
    if matrix.shape[1] != len(parameter_names):
        return []
    if not np.all(np.isfinite(matrix)):
        return []

    norms = np.linalg.norm(matrix, axis=0)
    if not np.all(np.isfinite(norms)):
        return []
    max_norm = float(np.max(norms)) if norms.size else 0.0
    zero_cutoff = tol * max(1.0, max_norm)

    diagnostics: List[Diagnostic] = []
    no_effect_names = [
        parameter_names[idx]
        for idx, norm in enumerate(norms)
        if float(norm) <= zero_cutoff
    ]
    if no_effect_names:
        diagnostics.append(_make_no_effect_diagnostic(model, no_effect_names))

    nonzero_indices = [
        idx
        for idx, norm in enumerate(norms)
        if float(norm) > zero_cutoff
    ]
    if len(nonzero_indices) < 2:
        return diagnostics

    normalized = matrix[:, nonzero_indices] / norms[nonzero_indices]
    if _column_rank(normalized, np, tol) == normalized.shape[1]:
        return diagnostics

    nonzero_names = [parameter_names[idx] for idx in nonzero_indices]
    groups = _collinear_groups(normalized, nonzero_names, np, tol)
    for group in groups:
        diagnostics.append(_make_collinearity_diagnostic(model, group))

    return diagnostics


def _is_known_numeric(value: object) -> bool:
    if value is None:
        return False
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def _effect_basis(
    model: ParsedModel,
    ss_values: dict,
    param_overrides: Optional[Dict[str, float]],
    np,
    n_endo: int,
):
    residuals = _steady_state_residuals(
        model,
        ss_values,
        param_overrides,
        np,
    )
    f_yp, f_y0, f_ym = _compute_jacobian(
        model,
        ss_values,
        param_overrides=param_overrides,
    )
    expected_shape = (len(model.dynamic_model_equations()), n_endo)
    if (
        f_yp.shape != expected_shape
        or f_y0.shape != expected_shape
        or f_ym.shape != expected_shape
    ):
        raise ValueError("Jacobian shape mismatch")
    pieces = [
        residuals.ravel(),
        f_yp.ravel(),
        f_y0.ravel(),
        f_ym.ravel(),
    ]
    vector = np.concatenate(pieces)
    if not np.all(np.isfinite(vector)):
        raise ValueError("non-finite identification effect basis")
    return vector


def _steady_state_residuals(
    model: ParsedModel,
    ss_values: dict,
    param_overrides: Optional[Dict[str, float]],
    np,
):
    report = validate_computed_steady_state(
        model,
        ss_values,
        param_overrides=param_overrides,
    )
    residuals = []
    for result in report.results:
        if result.is_local_var:
            continue
        if result.residual is None:
            raise ValueError("steady-state residual could not be evaluated")
        residuals.append(float(result.residual))
    return np.asarray(residuals, dtype=float)


def _column_rank(matrix, np, tol: float) -> int:
    if matrix.size == 0:
        return 0
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0:
        return 0
    if not np.all(np.isfinite(singular_values)):
        raise ValueError("non-finite singular values")
    cutoff = tol * max(matrix.shape) * max(float(singular_values[0]), 1.0)
    return int(np.count_nonzero(singular_values > cutoff))


def _collinear_groups(matrix, names: List[str], np, tol: float) -> List[List[str]]:
    pairwise_groups = _pairwise_collinear_groups(matrix, names, np, tol)
    if pairwise_groups:
        return pairwise_groups

    groups = _nullspace_groups(matrix, names, np, tol)
    if groups:
        return groups
    return [names]


def _pairwise_collinear_groups(
    matrix,
    names: List[str],
    np,
    tol: float,
) -> List[List[str]]:
    parent = list(range(len(names)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            subset = matrix[:, [left, right]]
            singular_values = np.linalg.svd(subset, compute_uv=False)
            if singular_values.size < 2:
                union(left, right)
                continue
            cutoff = tol * max(subset.shape) * max(float(singular_values[0]), 1.0)
            if float(singular_values[1]) <= cutoff:
                union(left, right)

    grouped: dict[int, List[str]] = {}
    for idx, name in enumerate(names):
        grouped.setdefault(find(idx), []).append(name)
    return [group for group in grouped.values() if len(group) > 1]


def _nullspace_groups(matrix, names: List[str], np, tol: float) -> List[List[str]]:
    _u, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    if not np.all(np.isfinite(singular_values)):
        raise ValueError("non-finite singular values")
    if singular_values.size == 0:
        return []
    cutoff = tol * max(matrix.shape) * max(float(singular_values[0]), 1.0)
    rank = int(np.count_nonzero(singular_values > cutoff))
    groups: List[List[str]] = []
    seen: set[tuple[str, ...]] = set()

    for row_idx in range(rank, vh.shape[0]):
        vector = vh[row_idx, :]
        active = [
            names[idx]
            for idx, weight in enumerate(vector)
            if abs(float(weight)) > _NULL_LOADING_TOL
        ]
        if len(active) < 2:
            continue
        key = tuple(active)
        if key not in seen:
            seen.add(key)
            groups.append(active)
    return groups


def _diagnostic_range(model: ParsedModel) -> SourceRange:
    return model.model_block_range or SourceRange(
        Position(0, 0),
        Position(0, 1),
    )


def _make_no_effect_diagnostic(
    model: ParsedModel,
    parameter_names: List[str],
) -> Diagnostic:
    parameter_text = ", ".join(parameter_names)
    return Diagnostic(
        range=_diagnostic_range(model),
        severity=Severity.WARNING,
        message=(
            "First-order structural identification: parameter(s) "
            f"{parameter_text} have no first-order effect on steady-state "
            "residuals or the first-order Jacobian. This structural check "
            "does not test full data/moment identification."
        ),
        source="dynare",
        code=IDENTIFICATION_CODE,
    )


def _make_collinearity_diagnostic(
    model: ParsedModel,
    parameter_names: List[str],
) -> Diagnostic:
    parameter_text = ", ".join(parameter_names)
    return Diagnostic(
        range=_diagnostic_range(model),
        severity=Severity.WARNING,
        message=(
            "First-order structural identification: parameters "
            f"{parameter_text} are not separately identifiable at first "
            "order because their effect vectors on steady-state residuals "
            "and the first-order Jacobian are collinear or linearly "
            "dependent. This structural check does not test full "
            "data/moment identification."
        ),
        source="dynare",
        code=IDENTIFICATION_CODE,
    )
