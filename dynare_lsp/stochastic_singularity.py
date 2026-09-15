"""First-order model-implied stochastic-singularity diagnostics.

The cheap W092 setup check catches the necessary count condition
``n_observables <= n_shocks``.  This module goes one step further after a valid
steady state and determinate first-order solution are available: it derives the
contemporaneous innovation loading implied by the linearized model and checks
whether observed variables receive enough *independent* innovations.
"""

from __future__ import annotations

from copy import copy
from typing import Any, Dict, List, cast

from .diagnostics import Diagnostic, Severity
from .parser import ParsedModel, Position, SourceRange

CODE = "W098"


def _anchor(model: ParsedModel) -> SourceRange:
    if model.varobs_range is not None:
        return model.varobs_range
    if model.model_block_range is not None:
        return model.model_block_range
    return SourceRange(Position(0, 0), Position(0, 1))


def _shock_impact_matrix(model: ParsedModel, ss_values: Dict[str, float]):
    """Return endogenous-by-structural-shock first-order impact matrix."""
    import numpy as np
    from scipy.linalg import ordqz

    from .bk_check import (
        _compute_jacobian,
        _extract_variable_timing,
        _form_minimal_dynamic_pencil,
        check_blanchard_kahn,
    )

    bk = check_blanchard_kahn(model, ss_values)
    if not bk.satisfied:
        raise ValueError("a determinate Blanchard-Kahn solution is required")

    endo_names = [item.name for item in model.endogenous]
    timing = _extract_variable_timing(model)
    forward = {name for name, offsets in timing.items() if any(o > 0 for o in offsets)}
    predetermined = {name for name, offsets in timing.items() if any(o < 0 for o in offsets)}
    forward_indices = [i for i, name in enumerate(endo_names) if name in forward]
    predetermined_indices = [i for i, name in enumerate(endo_names) if name in predetermined]
    static_indices = [
        i for i, name in enumerate(endo_names)
        if name not in forward and name not in predetermined
    ]

    f_yp, f_y0, f_ym = _compute_jacobian(model, ss_values)
    a, b = _form_minimal_dynamic_pencil(
        f_yp,
        f_y0,
        f_ym,
        predetermined_indices,
        forward_indices,
        static_indices,
    )
    qz_criterium = 1.000001

    def explosive(alpha, beta):
        return np.abs(alpha) > qz_criterium * np.abs(beta)

    if a.shape[0]:
        _aa, _bb, _alpha, _beta, _q, z = cast(Any, ordqz)(
            a, b, sort=explosive, output="complex"
        )
    else:
        z = np.zeros((0, 0), dtype=complex)

    n_pred = len(predetermined_indices)
    n_forward = len(forward_indices)
    if n_pred:
        stable = z[:, n_forward:]
        z_pred = stable[:n_pred, :]
        z_forward = stable[n_pred:, :]
        if z_pred.shape != (n_pred, n_pred) or np.linalg.matrix_rank(z_pred) < n_pred:
            raise ValueError("stable invariant subspace is rank deficient")
        h_forward = z_forward @ np.linalg.inv(z_pred)
    else:
        h_forward = np.zeros((n_forward, 0), dtype=complex)

    selector_pred = np.zeros((n_pred, len(endo_names)))
    for row, idx in enumerate(predetermined_indices):
        selector_pred[row, idx] = 1.0
    expectational_feedback = (
        f_yp[:, forward_indices] @ h_forward @ selector_pred
        if n_forward and n_pred
        else np.zeros_like(f_y0, dtype=complex)
    )
    current_system = np.asarray(f_y0, dtype=complex) + expectational_feedback
    if current_system.shape[0] != current_system.shape[1]:
        raise ValueError("equation/endogenous count mismatch")
    if np.linalg.matrix_rank(current_system) < current_system.shape[0]:
        raise ValueError("current first-order system is rank deficient")

    shocks = list(model.exogenous)
    if not shocks:
        return np.zeros((len(endo_names), 0), dtype=float)

    # Reuse the thoroughly-tested endogenous Jacobian evaluator for shocks by
    # temporarily treating stochastic exogenous variables as current-period
    # endogenous columns. Their steady-state value is zero.
    augmented = copy(model)
    augmented.endogenous = list(model.endogenous) + shocks
    augmented.exogenous = []
    augmented_ss = dict(ss_values)
    augmented_ss.update({shock.name: 0.0 for shock in shocks})
    _fyp_aug, fy0_aug, _fym_aug = _compute_jacobian(augmented, augmented_ss)
    f_e = np.asarray(fy0_aug[:, len(endo_names):], dtype=complex)
    impact = np.linalg.solve(current_system, -f_e)
    return np.real_if_close(impact, tol=1000).astype(float)


def check_stochastic_singularity(
    model: ParsedModel,
    ss_values: Dict[str, float],
) -> List[Diagnostic]:
    """Return W098 when first-order observable innovations are rank deficient."""
    observables = [
        name for name in dict.fromkeys(model.varobs_vars)
        if name in model.endogenous_names()
    ]
    if not observables:
        return []

    structural_shocks = [item.name for item in model.exogenous]
    measurement_errors = {
        entry.name
        for entry in model.estimated_params
        if entry.kind == "stderr" and entry.name in observables
    }
    measurement_errors |= {name for name in model.shocks_vars if name in observables}
    n_available = len(structural_shocks) + len(measurement_errors)
    if len(observables) > n_available:
        # W092 owns the simpler count failure.
        return []

    try:
        import numpy as np
        impact = _shock_impact_matrix(model, ss_values)
    except Exception:
        return []

    endo_index = {item.name: i for i, item in enumerate(model.endogenous)}
    observed_impact = impact[[endo_index[name] for name in observables], :]
    if measurement_errors:
        extra = np.zeros((len(observables), len(measurement_errors)))
        for col, name in enumerate(sorted(measurement_errors)):
            extra[observables.index(name), col] = 1.0
        observed_impact = np.hstack([observed_impact, extra])

    if observed_impact.size == 0:
        return []
    singular_values = np.linalg.svd(observed_impact, compute_uv=False)
    scale = float(singular_values[0]) if singular_values.size else 0.0
    tol = max(observed_impact.shape) * max(scale, 1.0) * 1e-10
    rank = int(np.count_nonzero(singular_values > tol))
    if rank >= len(observables):
        return []

    smallest = float(singular_values[-1]) if singular_values.size else 0.0
    return [Diagnostic(
        range=_anchor(model),
        severity=Severity.WARNING,
        message=(
            "First-order stochastic singularity: the model has enough shocks "
            f"by count ({n_available} for {len(observables)} observables), but "
            f"the implied observable innovation matrix has rank {rank}; "
            f"smallest singular value {smallest:.3g}. This is a local "
            "first-order rank result, not a data/posterior Hessian diagnostic."
        ),
        source="dynare",
        code=CODE,
    )]
