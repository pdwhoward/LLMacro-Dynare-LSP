"""Diagnostics for Dynare optimal-policy constructs.

Mirrors the setup requirements Dynare enforces for optimal policy
(``ComputingTasks.cc`` / ``ramsey_*``, ``discretionary_policy``, ``osr``): a
Ramsey or discretionary problem needs a ``planner_objective``; instruments must
be endogenous; the planner discount factor must be a valid discount factor; and
``osr`` needs both the parameters to optimise (``osr_params``) and an objective
(``optim_weights``).  All findings are warnings.

Codes:
  W100  optimal-policy command requires a planner_objective
  W101  policy instrument is not a declared endogenous variable
  W102  planner_discount is not a valid discount factor in (0, 1]
  W103  osr is missing osr_params or optim_weights
"""

from __future__ import annotations

from typing import List, Optional, Set

from .diagnostics import Diagnostic, Severity
from .parser import ParsedModel, Position, SourceRange

_FALLBACK_RANGE = SourceRange(Position(0, 0), Position(0, 1))

_PLANNER_COMMANDS = {"ramsey_model", "ramsey_policy", "discretionary_policy"}


def _rng(rng: Optional[SourceRange]) -> SourceRange:
    return rng if rng is not None else _FALLBACK_RANGE


def check_policy(model: ParsedModel, endogenous: Set[str]) -> List[Diagnostic]:
    """Validate optimal-policy constructs of *model*."""
    diagnostics: List[Diagnostic] = []
    if not model.policy_commands:
        return diagnostics

    anchor = _rng(model.policy_command_range)

    # W100 -- Ramsey / discretionary policy needs a planner_objective.
    planner_command = next(
        (c for c in model.policy_commands if c in _PLANNER_COMMANDS), None,
    )
    if planner_command is not None and model.planner_objective_range is None:
        diagnostics.append(Diagnostic(
            range=anchor,
            severity=Severity.WARNING,
            message=(
                f"{planner_command} requires a planner_objective statement, "
                "which is missing."
            ),
            source="dynare",
            code="W100",
        ))

    # W101 -- declared instruments must be endogenous.
    for instrument in model.instruments:
        if instrument not in endogenous:
            diagnostics.append(Diagnostic(
                range=anchor,
                severity=Severity.WARNING,
                message=(
                    f"Policy instrument '{instrument}' is not a declared "
                    "endogenous variable."
                ),
                source="dynare",
                code="W101",
            ))

    # W102 -- planner_discount must be a valid discount factor.
    discount = model.planner_discount
    if discount is not None and not (0.0 < discount <= 1.0):
        diagnostics.append(Diagnostic(
            range=anchor,
            severity=Severity.WARNING,
            message=(
                f"planner_discount = {discount:g} should be a discount factor "
                "in the interval (0, 1]."
            ),
            source="dynare",
            code="W102",
        ))

    # W103 -- osr needs osr_params and an optim_weights objective.
    if "osr" in model.policy_commands:
        if not model.osr_params:
            diagnostics.append(Diagnostic(
                range=anchor,
                severity=Severity.WARNING,
                message=(
                    "osr requires an osr_params statement listing the "
                    "parameters to optimize."
                ),
                source="dynare",
                code="W103",
            ))
        if not model.has_optim_weights:
            diagnostics.append(Diagnostic(
                range=anchor,
                severity=Severity.WARNING,
                message=(
                    "osr requires an optim_weights block defining the "
                    "objective (the weights on the target variables)."
                ),
                source="dynare",
                code="W103",
            ))

    return diagnostics
