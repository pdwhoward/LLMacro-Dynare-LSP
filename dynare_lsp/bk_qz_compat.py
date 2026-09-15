"""Dynare-compatible QZ zero-threshold evidence for the BK checker.

Dynare treats a generalized Schur pair with both |alpha| and |beta| below
``qz_zero_threshold`` (default 1e-6) as a singular 0/0 root.  SciPy's QZ
routine can return tiny nonzero values for the same structure, so exact-zero
tests can miss a genuinely singular pencil.  This compatibility layer keeps
the existing BK implementation as the source of timing/Jacobian/rank logic and
adds Dynare's numerical classification consistently to CLI, LSP, and MCP users.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple, cast

DEFAULT_QZ_ZERO_THRESHOLD = 1e-6
DEFAULT_BOUNDARY_TOL = 1e-5


def _qz_evidence(
    model,
    ss_values: Dict[str, float],
    *,
    unit_root_tol: float,
    qz_zero_threshold: float,
    boundary_tol: float,
) -> Optional[Tuple[bool, float, List[complex]]]:
    """Return (singular_0_over_0, qz_criterium, near-boundary roots)."""
    try:
        import numpy as np
        from scipy.linalg import ordqz

        from . import bk_check as bk

        timing = bk._extract_variable_timing(model)
        endo_names = [item.name for item in model.endogenous]
        n_endo = len(endo_names)
        real_eqs = model.dynamic_model_equations()
        if n_endo == 0 or len(real_eqs) != n_endo:
            return None
        if any(bk._EXPECTATION_OPERATOR_RE.search(eq.text) for eq in real_eqs):
            return None
        if bk._higher_order_timing_reference(model, timing) is not None:
            return None
        if bk._unresolved_timed_builtin_constant(model) is not None:
            return None

        forward = {
            name
            for name, offsets in timing.items()
            if any(offset > 0 for offset in offsets)
        }
        predetermined = {
            name
            for name, offsets in timing.items()
            if any(offset < 0 for offset in offsets)
        }
        forward_indices = [i for i, name in enumerate(endo_names) if name in forward]
        predetermined_indices = [
            i for i, name in enumerate(endo_names) if name in predetermined
        ]
        static_indices = [
            i
            for i, name in enumerate(endo_names)
            if name not in forward and name not in predetermined
        ]

        f_yp, f_y0, f_ym = bk._compute_jacobian(model, ss_values)
        linearization = np.hstack([f_yp, f_y0, f_ym])
        if int(np.linalg.matrix_rank(linearization)) < n_endo:
            return None
        a, b = bk._form_minimal_dynamic_pencil(
            f_yp,
            f_y0,
            f_ym,
            predetermined_indices,
            forward_indices,
            static_indices,
        )
        qz_criterium = 1.0 + unit_root_tol

        def explosive(alpha, beta):
            return np.abs(alpha) > qz_criterium * np.abs(beta)

        if a.shape[0] == 0:
            alpha = np.array([], dtype=complex)
            beta = np.array([], dtype=complex)
        else:
            _aa, _bb, alpha, beta, _q, _z = cast(Any, ordqz)(
                a, b, sort=explosive, output="complex"
            )

        singular = any(
            abs(complex(av)) < qz_zero_threshold
            and abs(complex(bv)) < qz_zero_threshold
            for av, bv in zip(alpha, beta)
        )
        near: List[complex] = []
        for av, bv in zip(alpha, beta):
            av = complex(av)
            bv = complex(bv)
            if abs(av) < qz_zero_threshold and abs(bv) < qz_zero_threshold:
                continue
            if abs(bv) < qz_zero_threshold:
                continue
            root = av / bv
            if np.isfinite(abs(root)) and abs(abs(root) - qz_criterium) <= boundary_tol:
                near.append(root)
        return singular, qz_criterium, near
    except Exception:
        return None


def install() -> None:
    """Patch the public BK API once, preserving all existing solver logic."""
    from . import bk_check as bk

    if getattr(bk, "_dynare_qz_compat_installed", False):
        return

    original_check = bk.check_blanchard_kahn
    original_to_diagnostics = bk.bk_to_diagnostics

    def check_blanchard_kahn(
        model,
        ss_values,
        unit_root_tol: float = 1e-6,
        *,
        qz_zero_threshold: float = DEFAULT_QZ_ZERO_THRESHOLD,
        boundary_tol: float = DEFAULT_BOUNDARY_TOL,
        _allow_auxiliary_transform: bool = True,
    ):
        result = original_check(
            model,
            ss_values,
            unit_root_tol,
            _allow_auxiliary_transform=_allow_auxiliary_transform,
        )
        evidence = _qz_evidence(
            model,
            ss_values,
            unit_root_tol=unit_root_tol,
            qz_zero_threshold=qz_zero_threshold,
            boundary_tol=boundary_tol,
        )
        setattr(result, "qz_zero_threshold", qz_zero_threshold)
        setattr(result, "qz_criterium", 1.0 + unit_root_tol)
        setattr(result, "singular_qz", False)
        setattr(result, "near_boundary_eigenvalues", [])
        if evidence is None:
            return result
        singular, qz_criterium, near = evidence
        setattr(result, "qz_criterium", qz_criterium)
        setattr(result, "near_boundary_eigenvalues", near)
        legacy_exact_singularity = (
            not result.satisfied
            and not result.eigenvalues
            and "0/0" in (result.message or "")
        )
        if singular and not legacy_exact_singularity:
            singular_result = replace(
                result,
                satisfied=False,
                eigenvalues=[],
                message=(
                    "Blanchard-Kahn conditions cannot be uniquely classified: "
                    "the generalized Schur decomposition contains a singular "
                    "0/0 eigenvalue whose numerator and denominator are both "
                    f"below Dynare's qz_zero_threshold={qz_zero_threshold:g}. "
                    "The linearized model does not admit a unique local solution."
                ),
            )
            setattr(singular_result, "qz_zero_threshold", qz_zero_threshold)
            setattr(singular_result, "qz_criterium", qz_criterium)
            setattr(singular_result, "singular_qz", True)
            setattr(singular_result, "near_boundary_eigenvalues", near)
            return singular_result
        return result

    def bk_to_diagnostics(result, model):
        if getattr(result, "singular_qz", False):
            rng = model.model_block_range or bk.SourceRange(
                bk.Position(0, 0), bk.Position(0, 1)
            )
            diagnostics = [
                bk.Diagnostic(
                    range=rng,
                    severity=bk.Severity.WARNING,
                    message=result.message,
                    source="dynare",
                    code="W072",
                )
            ]
        else:
            diagnostics = original_to_diagnostics(result, model)

        near = list(getattr(result, "near_boundary_eigenvalues", []) or [])
        if near:
            rng = model.model_block_range or bk.SourceRange(
                bk.Position(0, 0), bk.Position(0, 1)
            )
            criterion = float(getattr(result, "qz_criterium", 1.000001))
            preview = ", ".join(f"{abs(value):.8g}" for value in near[:4])
            diagnostics.append(
                bk.Diagnostic(
                    range=rng,
                    severity=bk.Severity.INFORMATION,
                    message=(
                        "Blanchard-Kahn classification is numerically sensitive: "
                        f"eigenvalue modulus {preview} lies very near the QZ "
                        f"stability cutoff {criterion:.8g}."
                    ),
                    source="dynare",
                    code="I072",
                )
            )
        return diagnostics

    bk.check_blanchard_kahn = check_blanchard_kahn
    bk.bk_to_diagnostics = bk_to_diagnostics
    setattr(bk, "_dynare_qz_compat_installed", True)
