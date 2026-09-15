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
  W096  estimated-parameter value/prior outside mathematical support
  W097  invalid prior hyperparameters (mean/standard deviation)
"""

from __future__ import annotations

import math
from typing import List, Optional, Set

from .diagnostics import Diagnostic, Severity
from .parser import (
    _PRIOR_SHAPES,
    _strip_comments,
    EstimatedParam,
    ParsedModel,
    Position,
    SourceRange,
)

_FALLBACK_RANGE = SourceRange(Position(0, 0), Position(0, 1))


def _rng(rng: Optional[SourceRange]) -> SourceRange:
    return rng if rng is not None else _FALLBACK_RANGE


def _format_bounds(lower: Optional[float], upper: Optional[float]) -> str:
    """Render inclusive estimated-parameter bounds, including one-sided ones."""
    if lower is None or (math.isinf(lower) and lower < 0):
        left = "(-inf"
    else:
        left = f"[{lower:g}"
    if upper is None or (math.isinf(upper) and upper > 0):
        right = "+inf)"
    else:
        right = f"{upper:g}]"
    return f"{left}, {right}"


def _source_slice(text: str, rng: Optional[SourceRange]) -> str:
    """Return the source text covered by a position range."""
    if rng is None:
        return ""
    lines = text.splitlines(keepends=True)
    if not lines or rng.start.line >= len(lines):
        return ""
    last_line = min(rng.end.line, len(lines) - 1)
    if rng.start.line == last_line:
        return lines[rng.start.line][rng.start.character : rng.end.character]
    pieces = [lines[rng.start.line][rng.start.character :]]
    pieces.extend(lines[rng.start.line + 1 : last_line])
    pieces.append(lines[last_line][: rng.end.character])
    return "".join(pieces)


def _parse_explicit_bound(token: str) -> Optional[float]:
    """Parse a literal bound while treating an empty field as unspecified."""
    token = token.strip()
    if not token:
        return None
    try:
        value = float(token)
    except ValueError:
        return None
    return None if math.isnan(value) else value


def _explicit_source_bounds(
    model: ParsedModel,
    entry: EstimatedParam,
) -> tuple[Optional[float], Optional[float]]:
    """Recover explicit bounds that the parser drops when one field is empty.

    Dynare permits empty numeric fields in ``estimated_params``.  The parser's
    compact numeric-prefix representation intentionally stops at the first
    empty field, so a source line with only a lower or only an upper bound can
    leave both ``entry.lower`` and ``entry.upper`` unset.  The entry range still
    preserves the original fields, which is sufficient for this diagnostic.
    """
    raw = _strip_comments(_source_slice(model.text, entry.range)).strip()
    if raw.endswith(";"):
        raw = raw[:-1]
    fields = [field.strip() for field in raw.split(",")]
    rest_start = 2 if entry.kind == "corr" else 1
    shape_index = next(
        (
            index
            for index in range(rest_start, len(fields))
            if fields[index].lower() in _PRIOR_SHAPES
        ),
        len(fields),
    )

    def bound_at(index: int) -> Optional[float]:
        if index >= len(fields) or index >= shape_index:
            return None
        return _parse_explicit_bound(fields[index])

    return bound_at(rest_start + 1), bound_at(rest_start + 2)



def _prior_fields(
    model: ParsedModel,
    entry: EstimatedParam,
) -> tuple[str, Optional[float], Optional[float]]:
    """Recover prior shape, mean, and standard deviation from source fields."""
    raw = _strip_comments(_source_slice(model.text, entry.range)).strip()
    if raw.endswith(";"):
        raw = raw[:-1]
    fields = [field.strip() for field in raw.split(",")]
    rest_start = 2 if entry.kind == "corr" else 1
    for index in range(rest_start, len(fields)):
        shape = fields[index].lower()
        if shape not in _PRIOR_SHAPES:
            continue
        def value_at(offset: int) -> Optional[float]:
            j = index + offset
            if j >= len(fields) or not fields[j]:
                return None
            try:
                value = float(fields[j])
            except ValueError:
                return None
            return value
        return shape, value_at(1), value_at(2)
    return entry.prior_shape, None, None


def _prior_support_diagnostics(
    model: ParsedModel,
    entry: EstimatedParam,
    lower: Optional[float],
    upper: Optional[float],
) -> List[Diagnostic]:
    """W096/W097 -- objective support and hyperparameter validity checks."""
    rng = _rng(entry.range)
    shape, prior_mean, prior_std = _prior_fields(model, entry)
    diagnostics: List[Diagnostic] = []

    if prior_std is not None and (not math.isfinite(prior_std) or prior_std <= 0):
        diagnostics.append(Diagnostic(
            range=rng,
            severity=Severity.WARNING,
            message=(
                f"estimated_params: '{entry.name}' prior standard deviation "
                f"must be finite and positive; got {prior_std:g}."
            ),
            source="dynare",
            code="W097",
        ))

    if prior_mean is not None and not math.isfinite(prior_mean):
        diagnostics.append(Diagnostic(
            range=rng,
            severity=Severity.WARNING,
            message=(
                f"estimated_params: '{entry.name}' prior mean must be finite; "
                f"got {prior_mean:g}."
            ),
            source="dynare",
            code="W097",
        ))
        prior_mean = None

    positive_support = entry.kind == "stderr" or shape in {
        "gamma_pdf", "inv_gamma_pdf", "inv_gamma1_pdf", "inv_gamma2_pdf", "weibull_pdf"
    }
    if positive_support:
        candidates = [
            ("initial value", entry.init),
            ("lower bound", lower),
            ("prior mean", prior_mean),
        ]
        for label, value in candidates:
            if value is not None and math.isfinite(value) and value < 0:
                diagnostics.append(Diagnostic(
                    range=rng,
                    severity=Severity.WARNING,
                    message=(
                        f"estimated_params: '{entry.name}' {label} {value:g} "
                        "is outside the non-negative support required by this "
                        "standard-deviation/positive prior specification."
                    ),
                    source="dynare",
                    code="W096",
                ))
                break

    if entry.kind == "corr":
        candidates = [
            ("initial value", entry.init),
            ("lower bound", lower),
            ("upper bound", upper),
            ("prior mean", prior_mean),
        ]
        for label, value in candidates:
            if value is not None and math.isfinite(value) and not -1 <= value <= 1:
                diagnostics.append(Diagnostic(
                    range=rng,
                    severity=Severity.WARNING,
                    message=(
                        f"estimated_params: correlation '{entry.name}, "
                        f"{entry.corr_with}' {label} {value:g} lies outside "
                        "the mathematical correlation support [-1, 1]."
                    ),
                    source="dynare",
                    code="W096",
                ))
                break

    if shape == "beta_pdf" and prior_mean is not None:
        if not 0 < prior_mean < 1:
            diagnostics.append(Diagnostic(
                range=rng,
                severity=Severity.WARNING,
                message=(
                    f"estimated_params: beta prior for '{entry.name}' has mean "
                    f"{prior_mean:g}, outside its open support (0, 1)."
                ),
                source="dynare",
                code="W096",
            ))
        elif prior_std is not None and math.isfinite(prior_std) and prior_std > 0:
            max_variance = prior_mean * (1.0 - prior_mean)
            if prior_std * prior_std >= max_variance:
                diagnostics.append(Diagnostic(
                    range=rng,
                    severity=Severity.WARNING,
                    message=(
                        f"estimated_params: beta prior for '{entry.name}' has "
                        f"mean {prior_mean:g} and standard deviation "
                        f"{prior_std:g}; these imply non-positive beta shape "
                        "parameters because variance must be below "
                        "mean*(1-mean)."
                    ),
                    source="dynare",
                    code="W097",
                ))

    return diagnostics

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

        source_lower, source_upper = _explicit_source_bounds(model, entry)
        lower = entry.lower if entry.lower is not None else source_lower
        upper = entry.upper if entry.upper is not None else source_upper
        if lower is not None and upper is not None and lower >= upper:
            diagnostics.append(Diagnostic(
                range=rng,
                severity=Severity.WARNING,
                message=(
                    f"estimated_params: '{entry.name}' has lower bound {lower:g} "
                    f">= upper bound {upper:g}."
                ),
                source="dynare",
                code="W094",
            ))
        if entry.init is not None:
            below_lower = lower is not None and entry.init < lower
            above_upper = upper is not None and entry.init > upper
            if below_lower or above_upper:
                diagnostics.append(Diagnostic(
                    range=rng,
                    severity=Severity.WARNING,
                    message=(
                        f"estimated_params: '{entry.name}' initial value {entry.init:g} "
                        f"is outside its bounds {_format_bounds(lower, upper)}."
                    ),
                    source="dynare",
                    code="W094",
                ))
        diagnostics.extend(_prior_support_diagnostics(model, entry, lower, upper))

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
