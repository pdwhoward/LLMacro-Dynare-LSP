"""Conventional bounds for common DSGE parameters.

This is a static, opinionated table mapping parameter-name patterns to the
theoretical bounds under which the parameter has its standard
interpretation. Used by ``diagnostics._check_parameter_bounds`` to warn when
a calibration falls outside the expected range — a soft check that catches
typos like ``beta = 99`` (should be 0.99) or ``sigma = -1``.

Patterns are case-insensitive substring matches against the parameter name,
so the table can be small while still covering the most common naming
conventions in the literature (``betA``, ``BETA``, ``beta_hh`` all match
the ``beta`` pattern). Substring matching does not cover every spelling:
``betta`` does not contain ``beta``, so close alternate spellings get
their own table entries. When a parameter name matches more than one
pattern, the narrowest applicable bound wins.

A bound of the form ``(low, high, strict_low, strict_high)`` means the
admissible range is ``low <= x <= high`` with strictness flags. The
``rationale`` field is appended to the warning message so the user can
understand why the bound exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class ParameterBound:
    """Bound on a single parameter."""
    pattern: str            # case-insensitive substring match
    low: Optional[float]    # None = -inf
    high: Optional[float]   # None = +inf
    strict_low: bool        # True if x must be strictly greater than low
    strict_high: bool       # True if x must be strictly less than high
    rationale: str          # human-readable reason


# Conservative, well-attested bounds. Each is the *theoretical* admissible
# range; if a published paper deliberately uses a calibration outside the
# range, the user can ignore the warning. The point is to catch typos and
# unit errors, not to second-guess deliberate modeling choices.
_BOUNDS: List[ParameterBound] = [
    # Discount factor
    ParameterBound(
        pattern="beta",
        low=0.0, high=1.0, strict_low=True, strict_high=True,
        rationale="discount factor must lie in (0,1)",
    ),
    # Common alternate spelling of the discount factor (e.g. Schmitt-Grohe/
    # Uribe-derived code). "beta" is not a substring of "betta", so it needs
    # its own entry.
    ParameterBound(
        pattern="betta",
        low=0.0, high=1.0, strict_low=True, strict_high=True,
        rationale="discount factor must lie in (0,1)",
    ),
    # Capital share / output elasticity
    ParameterBound(
        pattern="alpha",
        low=0.0, high=1.0, strict_low=True, strict_high=True,
        rationale="capital share / output elasticity is typically in (0,1)",
    ),
    # Depreciation rate
    ParameterBound(
        pattern="delta",
        low=0.0, high=1.0, strict_low=False, strict_high=True,
        rationale="depreciation rate must lie in [0,1)",
    ),
    # Inverse Frisch elasticity (labor)
    ParameterBound(
        pattern="frisch",
        low=0.0, high=20.0, strict_low=False, strict_high=False,
        rationale="inverse Frisch elasticity is typically in [0,20]",
    ),
    # Risk aversion / inverse intertemporal elasticity
    ParameterBound(
        pattern="sigma_c",
        low=0.0, high=20.0, strict_low=True, strict_high=False,
        rationale="risk aversion / 1/IES is typically in (0,20]",
    ),
    # Generic standard deviation (e.g. sigma_a, sigma_e, sigma_g)
    ParameterBound(
        pattern="sigma_",
        low=0.0, high=None, strict_low=False, strict_high=False,
        rationale="standard deviation must be non-negative",
    ),
    ParameterBound(
        pattern="std_",
        low=0.0, high=None, strict_low=False, strict_high=False,
        rationale="standard deviation must be non-negative",
    ),
    # Variance
    ParameterBound(
        pattern="var_",
        low=0.0, high=None, strict_low=False, strict_high=False,
        rationale="variance must be non-negative",
    ),
    # AR(1) persistence — usually rho_*
    ParameterBound(
        pattern="rho",
        low=-1.0, high=1.0, strict_low=True, strict_high=True,
        rationale="AR(1) persistence must lie in (-1,1) for stationarity",
    ),
    # Taylor rule inflation coefficient (Taylor principle: > 1)
    ParameterBound(
        pattern="phi_pi",
        low=0.0, high=10.0, strict_low=False, strict_high=False,
        rationale="Taylor rule inflation coefficient should be non-negative; "
                  ">1 satisfies the Taylor principle",
    ),
    # Taylor rule output coefficient
    ParameterBound(
        pattern="phi_y",
        low=-1.0, high=5.0, strict_low=False, strict_high=False,
        rationale="Taylor rule output-gap coefficient is typically in [-1,5]",
    ),
    # Calvo price-stickiness probability
    ParameterBound(
        pattern="theta",
        low=0.0, high=1.0, strict_low=False, strict_high=True,
        rationale="Calvo / share parameter must lie in [0,1)",
    ),
    # Habit persistence
    ParameterBound(
        pattern="habit",
        low=0.0, high=1.0, strict_low=False, strict_high=True,
        rationale="habit persistence is typically in [0,1)",
    ),
    # Indexation
    ParameterBound(
        pattern="iota",
        low=0.0, high=1.0, strict_low=False, strict_high=False,
        rationale="indexation parameter is typically in [0,1]",
    ),
    # Markup / elasticity
    ParameterBound(
        pattern="epsilon",
        low=1.0, high=None, strict_low=True, strict_high=False,
        rationale="demand elasticity must exceed 1 for positive markup",
    ),
    # Inflation rate (steady-state).  Accept both net (-10%..50%) and
    # gross (0.9..1.5) conventions — Dynare models use both, and we
    # have no reliable way to tell them apart from the name alone.  The
    # union range covers gross-factor calibrations like ``pi_bar=1.005``
    # which we would otherwise reject as out of bounds.
    ParameterBound(
        pattern="pi_bar",
        low=-0.1, high=1.5, strict_low=False, strict_high=False,
        rationale="steady-state inflation rate is typically in [-10%,50%] "
                  "(net) or [0.9,1.5] (gross); accept both conventions.",
    ),
    # Real interest rate.  Same dual-convention story: net (-5%..50%)
    # or gross (0.9..1.5).
    ParameterBound(
        pattern="r_bar",
        low=-0.05, high=1.5, strict_low=False, strict_high=False,
        rationale="steady-state real interest rate is typically in "
                  "[-5%,50%] (net) or [0.9,1.5] (gross); accept both.",
    ),
]


def lookup(name: str) -> Optional[ParameterBound]:
    """Return the narrowest applicable bound for a parameter name, or None.

    Match is case-insensitive substring. If multiple patterns match, the one
    declared first in ``_BOUNDS`` wins — patterns are ordered from most
    specific to least specific.
    """
    if not name:
        return None
    lower = name.lower()
    for b in _BOUNDS:
        if b.pattern.lower() in lower:
            return b
    return None


def is_in_bounds(
    value: float, bound: ParameterBound,
) -> bool:
    """Check whether ``value`` lies inside the bound's admissible range."""
    if bound.low is not None:
        if bound.strict_low and not (value > bound.low):
            return False
        if not bound.strict_low and not (value >= bound.low):
            return False
    if bound.high is not None:
        if bound.strict_high and not (value < bound.high):
            return False
        if not bound.strict_high and not (value <= bound.high):
            return False
    return True


def format_range(bound: ParameterBound) -> str:
    """Render the bound as a printable interval, e.g. ``(0, 1)`` or ``[0, +inf)``."""
    if bound.low is None:
        left_bracket, left_val = "(-inf", ""
    else:
        left_bracket = "(" if bound.strict_low else "["
        left_val = f"{bound.low:g}"
    if bound.high is None:
        right_bracket, right_val = "", "+inf)"
    else:
        right_bracket = ")" if bound.strict_high else "]"
        right_val = f"{bound.high:g}"
    return f"{left_bracket}{left_val},{right_val}{right_bracket}"


def known_patterns() -> List[Tuple[str, str, str]]:
    """Return ``(pattern, range, rationale)`` triples for all known patterns."""
    return [(b.pattern, format_range(b), b.rationale) for b in _BOUNDS]
