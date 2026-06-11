"""Steady-state model diagnostics for Dynare models.

This module checks the static Jacobian implied by the model linearized at the
steady state.  Rank deficiency indicates collinear equations or variables that
can prevent a unique steady-state solution.

The collinearity analysis is ported from Dynare's ``matlab/model_diagnostics.m``
(``Singularity in Static Jacobian`` branch).  For a rank-deficient steady-state
Jacobian ``J`` it reports, per collinear relation:

* the collinear *variables* (right null space of ``J``), and
* the collinear *equations* (left null space of ``J``),

each reduced to the minimal set of participants via the progressive
weight-threshold sweep Dynare uses (``for j=1:10``).  When the singularity
coincides with an eigenvalue of modulus one, it adds Dynare's unit-root note so
an intentionally nonstationary model is not mistaken for a modelling error.
"""

from __future__ import annotations

from typing import List

from .bk_check import _compute_jacobian
from .diagnostics import Diagnostic, Severity
from .parser import ParsedModel, Position, SourceRange


MODEL_DIAGNOSTICS_CODE = "W080"
# Residual below which a reduced index set is accepted as a valid collinear
# relation (matches the 1e-6 acceptance test in model_diagnostics.m).
_VERIFY_TOL = 1e-6


def _rank_and_null_space(matrix, np, rel_tol: float):
    """Return ``(rank, null_vectors)`` for *matrix* from a single SVD.

    Mirrors MATLAB ``null(matrix, tol)``: the null space is spanned by the
    right-singular vectors whose singular value is at or below
    ``rel_tol * max(singular value)``.  For an ``m x n`` matrix these vectors
    live in ``R^n`` -- so calling with ``J`` yields the right null space
    (collinear *columns*/variables) and calling with ``J.T`` yields the left
    null space (collinear *rows*/equations).
    """
    _u, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    n_cols = matrix.shape[1]
    if singular_values.size == 0:
        return 0, [vh[i, :] for i in range(n_cols)]
    cutoff = rel_tol * float(np.max(singular_values))
    rank = int(np.count_nonzero(singular_values > cutoff))
    null_vectors = [vh[i, :] for i in range(rank, n_cols)]
    return rank, null_vectors


def _minimal_relation(matrix, vector, np) -> List[int]:
    """Isolate the minimal set of indices that still forms a collinear relation.

    Ports the ``for j=1:10`` refinement in ``model_diagnostics.m``: progressively
    loosen the weight threshold (``10^-1`` .. ``10^-10``) and keep the smallest
    index set ``k`` such that ``matrix[:, k] @ vector[k]`` is still numerically
    zero.  Without this, every faintly-weighted entry of the null-space vector
    would be reported as "involved".
    """
    last_k: List[int] = []
    for j in range(1, 11):
        threshold = 10.0 ** (-j)
        k = [i for i, weight in enumerate(vector) if abs(weight) > threshold]
        if not k:
            continue
        last_k = k
        residual = matrix[:, k] @ vector[k]
        if residual.size == 0 or float(np.max(np.abs(residual))) < _VERIFY_TOL:
            break
    if not last_k:
        last_k = [i for i, weight in enumerate(vector) if abs(weight) > 0.0]
    return sorted(last_k)


def _equation_label(equations, eq_index_zero_based: int) -> str:
    """Human label for the equation at row *eq_index_zero_based* of the Jacobian.

    Uses Dynare's 1-based equation numbering and appends the ``[name=...]`` tag
    when the equation carries one (Dynare prints ``Equation %d: %s``).
    """
    ordinal = eq_index_zero_based + 1
    if 0 <= eq_index_zero_based < len(equations):
        name = (getattr(equations[eq_index_zero_based], "name", "") or "").strip()
        if name:
            return f"equation {ordinal} ({name})"
    return f"equation {ordinal}"


def _unit_root_note(model: ParsedModel, ss_values: dict) -> str:
    """Dynare's note flagging a singularity that is (partly) a unit root.

    Best-effort: reuses the Blanchard-Kahn eigenvalue computation and stays
    silent if it is unavailable or fails, so it can never suppress the warning.
    """
    try:
        import numpy as np

        from .bk_check import check_blanchard_kahn

        result = check_blanchard_kahn(model, ss_values)
        for eigenvalue in result.eigenvalues:
            modulus = abs(eigenvalue)
            if np.isfinite(modulus) and abs(modulus - 1.0) < 1e-6:
                return (
                    " The singularity may be (partly) caused by a unit root: "
                    "an eigenvalue has modulus near 1. This is expected for an "
                    "intentionally nonstationary model; otherwise check for a "
                    "missing or redundant equation."
                )
    except Exception:
        pass
    return ""


def check_model_diagnostics(
    model: ParsedModel,
    ss_values: dict,
    tol: float = 1e-8,
) -> List[Diagnostic]:
    """Check for steady-state Jacobian rank deficiency.

    Returns a single warning diagnostic when the static steady-state Jacobian
    is rank deficient -- reporting the collinear variables and the collinear
    equations involved in each relation -- otherwise an empty list.
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
        f_yp, f_y0, f_ym = _compute_jacobian(model, ss_values)
    except Exception:
        return []

    try:
        steady_state_jacobian = f_yp + f_y0 + f_ym
        if steady_state_jacobian.shape != (n_endo, n_endo):
            return []
        if not np.all(np.isfinite(steady_state_jacobian)):
            return []

        rank, var_vectors = _rank_and_null_space(steady_state_jacobian, np, tol)
        if rank == n_endo:
            return []

        # Left null space (collinear equations) from the transpose.
        _eq_rank, eq_vectors = _rank_and_null_space(
            steady_state_jacobian.T, np, tol,
        )

        variable_names = [var.name for var in endogenous]

        variable_relations: List[List[str]] = []
        for vector in var_vectors:
            indices = _minimal_relation(steady_state_jacobian, vector, np)
            names = [
                variable_names[idx]
                for idx in indices
                if 0 <= idx < len(variable_names)
            ]
            if names:
                variable_relations.append(names)

        equation_relations: List[List[str]] = []
        for vector in eq_vectors:
            indices = _minimal_relation(steady_state_jacobian.T, vector, np)
            labels = [_equation_label(equations, idx) for idx in indices]
            if labels:
                equation_relations.append(labels)

        n_relations = n_endo - rank
        parts: List[str] = [
            "Model diagnostics: steady-state Jacobian is rank deficient "
            f"(rank {rank} for {n_endo} endogenous variable(s)). "
            f"Found {n_relations} collinear relationship(s)."
        ]

        if variable_relations:
            total = len(variable_relations)
            for i, names in enumerate(variable_relations, start=1):
                parts.append(
                    f"Collinear variables (relation {i} of {total}): "
                    f"{', '.join(names)}."
                )
        else:
            parts.append("Collinear variables: none identified.")

        if equation_relations:
            total = len(equation_relations)
            for i, labels in enumerate(equation_relations, start=1):
                parts.append(
                    f"Collinear equations (relation {i} of {total}): "
                    f"{', '.join(labels)}."
                )

        message = " ".join(parts) + _unit_root_note(model, ss_values)

        return [Diagnostic(
            range=model.model_block_range or SourceRange(
                Position(0, 0),
                Position(0, 1),
            ),
            severity=Severity.WARNING,
            message=message,
            source="dynare",
            code=MODEL_DIAGNOSTICS_CODE,
        )]
    except Exception:
        return []
