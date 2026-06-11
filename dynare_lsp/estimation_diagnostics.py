"""Diagnostics for Dynare estimation blocks.

Mirrors the setup checks Dynare runs before estimation
(``initial_estimation_checks.m``, ``check_stochastic_singularity.m``): an
estimation is ill-posed if observed variables are not declared endogenous, if
there are more observables than shocks (stochastic singularity), if an
``estimated_params`` entry names an undeclared symbol, or if a prior's bounds
are inconsistent.  All findings are warnings so they never block a model that
is still being written.

Codes:
  W090  varobs variable is not a declared endogenous variable
  W091  duplicate varobs variable
  W092  stochastic singularity (more observables than shocks)
  W093  estimated_params references an undeclared symbol
  W094  estimated_params bound/initial-value inconsistency
  W095  observation_trends variable not listed in varobs
"""

from __future__ import annotations

from typing import List, Optional, Set

from .diagnostics import Diagnostic, Severity
from .parser import ParsedModel, Position, SourceRange

_FALLBACK_RANGE = SourceRange(Position(0, 0), Position(0, 1))


def _rng(rng: Optional[SourceRange]) -> SourceRange:
    return rng if rng is not None else _FALLBACK_RANGE


def check_estimation(
    model: ParsedModel,
    endogenous: Set[str],
    exogenous: Set[str],
    parameters: Set[str],
) -> List[Diagnostic]:
    """Validate the estimation blocks of *model*.

    ``endogenous``/``exogenous``/``parameters`` are the declared-name sets
    (typically taken from the include-merged context model so symbols declared
    in ``@#include``d files count as declared).
    """
    diagnostics: List[Diagnostic] = []

    has_context = bool(
        model.varobs_vars
        or model.observation_trends_vars
        or model.estimated_params
        or model.estimated_params_range is not None
        or model.varobs_range is not None
    )
    if not has_context:
        return diagnostics

    # --- varobs: declared-endogenous and duplicate checks ---
    seen: Set[str] = set()
    for name in model.varobs_vars:
        if name in seen:
            diagnostics.append(Diagnostic(
                range=_rng(model.varobs_range),
                severity=Severity.WARNING,
                message=f"Observed variable '{name}' is listed more than once in varobs.",
                source="dynare",
                code="W091",
            ))
            continue
        seen.add(name)
        if name not in endogenous:
            if name in exogenous:
                why = " (it is an exogenous variable)"
            elif name in parameters:
                why = " (it is a parameter)"
            else:
                why = ""
            diagnostics.append(Diagnostic(
                range=_rng(model.varobs_range),
                severity=Severity.WARNING,
                message=(
                    f"varobs variable '{name}' is not a declared endogenous "
                    f"variable{why}. Observed variables must be endogenous."
                ),
                source="dynare",
                code="W090",
            ))

    # --- stochastic singularity: #observables must be <= #shocks ---
    observables = [v for v in dict.fromkeys(model.varobs_vars) if v in endogenous]
    n_obs = len(observables)
    if n_obs:
        stochastic_exogenous = exogenous - {
            shock.name for shock in model.deterministic_exogenous
        }
        measurement_errors = {
            e.name for e in model.estimated_params
            if e.kind == "stderr" and e.name in endogenous
        }
        measurement_errors |= {s for s in model.shocks_vars if s in endogenous}
        n_shocks = len(stochastic_exogenous) + len(measurement_errors)
        if n_obs > n_shocks:
            diagnostics.append(Diagnostic(
                range=_rng(model.varobs_range),
                severity=Severity.WARNING,
                message=(
                    f"Stochastic singularity: {n_obs} observed variable(s) but only "
                    f"{n_shocks} shock(s) (structural shocks plus measurement errors). "
                    "The likelihood is stochastically singular; add measurement errors "
                    "or shocks, or reduce the number of observed variables."
                ),
                source="dynare",
                code="W092",
            ))

    # --- estimated_params: declared symbols and bound sanity ---
    for entry in model.estimated_params:
        rng = _rng(entry.range)
        if entry.kind == "param":
            if entry.name not in parameters:
                if entry.name in endogenous:
                    where = " (it is an endogenous variable)"
                elif entry.name in exogenous:
                    where = " (it is an exogenous variable)"
                else:
                    where = ""
                diagnostics.append(Diagnostic(
                    range=rng,
                    severity=Severity.WARNING,
                    message=f"estimated_params: '{entry.name}' is not a declared parameter{where}.",
                    source="dynare",
                    code="W093",
                ))
        elif entry.kind == "stderr":
            if entry.name not in exogenous and entry.name not in endogenous:
                diagnostics.append(Diagnostic(
                    range=rng,
                    severity=Severity.WARNING,
                    message=(
                        f"estimated_params: stderr '{entry.name}' is not a declared "
                        "shock or observed variable."
                    ),
                    source="dynare",
                    code="W093",
                ))
        elif entry.kind == "corr":
            for symbol in (entry.name, entry.corr_with):
                if symbol and symbol not in exogenous and symbol not in endogenous:
                    diagnostics.append(Diagnostic(
                        range=rng,
                        severity=Severity.WARNING,
                        message=(
                            f"estimated_params: corr references '{symbol}', which is not "
                            "a declared shock or variable."
                        ),
                        source="dynare",
                        code="W093",
                    ))

        if entry.lower is not None and entry.upper is not None and entry.lower >= entry.upper:
            diagnostics.append(Diagnostic(
                range=rng,
                severity=Severity.WARNING,
                message=(
                    f"estimated_params: '{entry.name}' has lower bound {entry.lower:g} "
                    f">= upper bound {entry.upper:g}."
                ),
                source="dynare",
                code="W094",
            ))
        if (
            entry.init is not None
            and entry.lower is not None
            and entry.upper is not None
            and not (entry.lower <= entry.init <= entry.upper)
        ):
            diagnostics.append(Diagnostic(
                range=rng,
                severity=Severity.WARNING,
                message=(
                    f"estimated_params: '{entry.name}' initial value {entry.init:g} is "
                    f"outside its bounds [{entry.lower:g}, {entry.upper:g}]."
                ),
                source="dynare",
                code="W094",
            ))

    # --- observation_trends variables must be observed ---
    varobs_set = set(model.varobs_vars)
    for name in model.observation_trends_vars:
        if name not in varobs_set:
            diagnostics.append(Diagnostic(
                range=_rng(
                    model.observation_trends_ranges.get(name)
                    or model.varobs_range
                ),
                severity=Severity.WARNING,
                message=f"observation_trends: '{name}' is not listed in varobs.",
                source="dynare",
                code="W095",
            ))

    return diagnostics
