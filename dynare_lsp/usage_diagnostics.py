"""Static usage/calibration diagnostics for Dynare models.

Cheap, gated checks that mirror runtime errors Dynare raises once a model is
executed, surfaced statically so the LSP can flag them before a run:

  W120  a stochastic command (stoch_simul / estimation) with no exogenous
  W121  a parameter used with a lead/lag, e.g. ``beta(+1)``
  W122  a deep parameter assigned a non-finite value (NaN / Inf) before a run

All findings are warnings.  Each check returns nothing for a model that does
not use the relevant construct, so an unrelated model gets zero new diagnostics.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Dict, List, Optional, Set, Tuple

from .diagnostics import Diagnostic, Severity
from .parser import (
    ParamAssignment,
    ParsedModel,
    Position,
    SourceRange,
    _offset_to_position,
    _safe_eval,
    _strip_comments,
)

_FALLBACK_RANGE = SourceRange(Position(0, 0), Position(0, 1))

# Stochastic commands that require at least one stochastic exogenous (varexo).
_STOCH_COMMAND_RE = re.compile(r"(?<!\w)(stoch_simul|estimation)\b", re.IGNORECASE)

# Commands whose execution reads the deep parameters; they must be finite.
_RUN_COMMAND_RE = re.compile(
    r"(?<!\w)(steady|check|stoch_simul|simul|estimation|"
    r"perfect_foresight_setup|perfect_foresight_solver|"
    r"ramsey_policy|discretionary_policy|osr|calib_smoother|forecast)\b",
    re.IGNORECASE,
)

# ``name(<signed int>)`` with nothing but whitespace between name and ``(``
# (so ``p*(1)`` multiplication and ``log(p)`` calls are not matched).
_LEAD_LAG_RE = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*\(\s*([+-]?\d+)\s*\)")

# Parameters Dynare auto-populates; never flagged as uninitialised.
_AUTO_PARAMS = {"optimal_policy_discount_factor"}


def _rng(rng: Optional[SourceRange]) -> SourceRange:
    return rng if rng is not None else _FALLBACK_RANGE


def _first_command(
    stripped: str,
    pattern: re.Pattern,
) -> Optional[Tuple[str, SourceRange]]:
    """Return ``(name, range)`` for the first command matching *pattern*."""
    match = pattern.search(stripped)
    if match is None:
        return None
    rng = SourceRange(
        _offset_to_position(stripped, match.start(1)),
        _offset_to_position(stripped, match.end(1)),
    )
    return match.group(1), rng


def _all_commands(
    stripped: str,
    pattern: re.Pattern,
) -> List[Tuple[str, SourceRange]]:
    commands: List[Tuple[str, SourceRange]] = []
    for match in pattern.finditer(stripped):
        commands.append((
            match.group(1),
            SourceRange(
                _offset_to_position(stripped, match.start(1)),
                _offset_to_position(stripped, match.end(1)),
            ),
        ))
    return commands


def _position_key(pos: Position) -> Tuple[int, int]:
    return (pos.line, pos.character)


OrderKey = Tuple[int, int, int, int, int]
CommandEvent = Tuple[str, SourceRange, OrderKey]
AssignmentEvent = Tuple[ParamAssignment, SourceRange, OrderKey]


def _usage_text(model: ParsedModel) -> str:
    """Return comment/string-stripped text for command scans."""
    return re.sub(
        r"'[^'\n]*'|\"[^\"\n]*\"",
        lambda m: " " * (m.end() - m.start()),
        _strip_comments(model.text),
    )


def _active_order_key(rng: SourceRange, sequence: int) -> OrderKey:
    return (
        rng.start.line,
        0,
        rng.start.line,
        rng.start.character,
        sequence,
    )


def _include_order_key(
    include_model: ParsedModel,
    rng: SourceRange,
    sequence: int,
) -> OrderKey:
    anchor = include_model.include_anchor_range
    parent_line = anchor.start.line if anchor is not None else rng.start.line
    return (
        parent_line,
        1 if anchor is not None else 0,
        rng.start.line,
        rng.start.character,
        sequence,
    )


def _anchor_include_diagnostics(
    diagnostics: List[Diagnostic],
    anchor: Optional[SourceRange],
) -> List[Diagnostic]:
    if anchor is None:
        return diagnostics
    return [replace(diagnostic, range=anchor, fix=None) for diagnostic in diagnostics]


def _all_command_events(
    model: ParsedModel,
    include_models: List[ParsedModel],
) -> List[CommandEvent]:
    events: List[CommandEvent] = []
    sequence = 0
    for command, rng in _all_commands(_usage_text(model), _RUN_COMMAND_RE):
        events.append((command, rng, _active_order_key(rng, sequence)))
        sequence += 1
    for include_model in include_models:
        anchor = include_model.include_anchor_range
        for command, rng in _all_commands(_usage_text(include_model), _RUN_COMMAND_RE):
            events.append((
                command,
                anchor or rng,
                _include_order_key(include_model, rng, sequence),
            ))
            sequence += 1
    return events


def _all_assignment_events(
    model: ParsedModel,
    include_models: List[ParsedModel],
    parameters: Set[str],
) -> List[AssignmentEvent]:
    events: List[AssignmentEvent] = []
    sequence = 0
    for assignment in model.param_assignments + model.helper_assignments:
        if assignment.name in parameters:
            events.append((
                assignment,
                assignment.range,
                _active_order_key(assignment.range, sequence),
            ))
            sequence += 1
    for include_model in include_models:
        anchor = include_model.include_anchor_range
        for assignment in (
            include_model.param_assignments + include_model.helper_assignments
        ):
            if assignment.name not in parameters:
                continue
            events.append((
                assignment,
                anchor or assignment.range,
                _include_order_key(include_model, assignment.range, sequence),
            ))
            sequence += 1
    return events


def check_usage(
    model: ParsedModel,
    exogenous: Set[str],
    deterministic_exogenous: Set[str],
    parameters: Set[str],
    include_models: Optional[List[ParsedModel]] = None,
) -> List[Diagnostic]:
    """Run the static usage checks (W120 / W121 / W122).

    The name sets are the *merged* (include-aware) symbol sets so a declaration
    in an ``@#include``'d file does not trigger a false positive.
    """
    include_models = include_models or []
    stripped = _usage_text(model)
    diagnostics: List[Diagnostic] = []
    diagnostics.extend(
        _check_no_exogenous(stripped, exogenous, deterministic_exogenous),
    )
    diagnostics.extend(_check_param_lead_lag(model, parameters))
    for include_model in include_models:
        include_stripped = _usage_text(include_model)
        diagnostics.extend(_anchor_include_diagnostics(
            _check_no_exogenous(
                include_stripped,
                exogenous,
                deterministic_exogenous,
            ),
            include_model.include_anchor_range,
        ))
        diagnostics.extend(_anchor_include_diagnostics(
            _check_param_lead_lag(include_model, parameters),
            include_model.include_anchor_range,
        ))
    diagnostics.extend(_check_nonfinite_params(
        model,
        stripped,
        parameters,
        include_models=include_models,
    ))
    return diagnostics


def _check_no_exogenous(
    stripped: str,
    exogenous: Set[str],
    deterministic_exogenous: Set[str],
) -> List[Diagnostic]:
    """W120 -- a stochastic command but no stochastic exogenous is declared."""
    stochastic_exogenous = exogenous - deterministic_exogenous
    if stochastic_exogenous:
        return []
    found = _first_command(stripped, _STOCH_COMMAND_RE)
    if found is None:
        return []
    name, rng = found
    return [Diagnostic(
        range=rng,
        severity=Severity.WARNING,
        message=(
            f"'{name}' is a stochastic command but the model declares no "
            "stochastic exogenous variable. Dynare requires at least one "
            "'varexo'; add a (dummy) shock and a shocks-block entry for it."
        ),
        source="dynare",
        code="W120",
    )]


def _check_param_lead_lag(
    model: ParsedModel,
    parameters: Set[str],
) -> List[Diagnostic]:
    """W121 -- a parameter written with a lead/lag, e.g. ``beta(+1)``."""
    if not parameters:
        return []
    diagnostics: List[Diagnostic] = []
    seen: Set[Tuple[str, int]] = set()
    for equation in model.dynamic_model_equations():
        for match in _LEAD_LAG_RE.finditer(equation.text):
            name = match.group(1)
            if name not in parameters:
                continue
            key = (name, equation.range.start.line)
            if key in seen:
                continue
            seen.add(key)
            diagnostics.append(Diagnostic(
                range=_rng(equation.range),
                severity=Severity.WARNING,
                message=(
                    f"Parameter '{name}' is used with a lead/lag "
                    f"('{name}({match.group(2)})'). Parameters are "
                    "time-invariant; this is usually a variable mis-declared "
                    "as a parameter, or a stray time index."
                ),
                source="dynare",
                code="W121",
            ))
    return diagnostics


def _check_nonfinite_params(
    model: ParsedModel,
    stripped: str,
    parameters: Set[str],
    *,
    include_models: Optional[List[ParsedModel]] = None,
) -> List[Diagnostic]:
    """W122 -- a deep parameter assigned a non-finite value before a run."""
    include_models = include_models or []
    commands = _all_command_events(model, include_models)
    if not commands:
        return []

    # Deep parameters: those actually referenced by the model equations
    # (Dynare's ``test_for_deep_parameters_calibration`` scope).  An ``inf``
    # bound parameter that never enters the model is intentionally not flagged.
    used = _names_used_in_model(model)
    for include_model in include_models:
        used.update(_names_used_in_model(include_model))
    assignments = sorted(
        _all_assignment_events(model, include_models, parameters),
        key=lambda event: event[2],
    )

    diagnostics: List[Diagnostic] = []
    reported: Set[Tuple[str, int, int]] = set()

    for command, _command_range, command_key in sorted(commands, key=lambda event: event[2]):
        latest: Dict[str, Tuple[ParamAssignment, SourceRange, Optional[float]]] = {}
        known: Dict[str, float] = {}
        for assignment, publish_range, assignment_key in assignments:
            if assignment_key >= command_key:
                continue
            value = _safe_eval(assignment.expression, known)
            if value is None:
                value = assignment.value
            if value is not None:
                known[assignment.name] = value
            else:
                known.pop(assignment.name, None)
            latest[assignment.name] = (assignment, publish_range, value)

        for name, (assignment, publish_range, value) in latest.items():
            if name in _AUTO_PARAMS or name not in used:
                continue
            kind: Optional[str] = None
            if value is not None and not math.isfinite(value):
                kind = "Inf" if math.isinf(value) else "a non-finite value"
            elif value is None:
                # ``NaN`` and capitalised ``Inf``/``Infinity`` are not in the
                # numeric eval environment, so they surface as an unevaluated
                # literal rather than a float -- match them textually.
                literal = re.fullmatch(
                    r"[+-]?(nan|inf|infinity)",
                    assignment.expression.strip(),
                    re.IGNORECASE,
                )
                if literal is not None:
                    kind = "NaN" if literal.group(1).lower() == "nan" else "Inf"
            if kind is None:
                continue
            report_key = (
                name,
                assignment.range.start.line,
                assignment.range.start.character,
            )
            if report_key in reported:
                continue
            reported.add(report_key)
            diagnostics.append(Diagnostic(
                range=_rng(publish_range),
                severity=Severity.WARNING,
                message=(
                    f"Parameter '{name}' is assigned {kind}. Dynare requires every "
                    f"deep parameter used in the model to be finite before "
                    f"running '{command}'."
                ),
                source="dynare",
                code="W122",
            ))
    return diagnostics


def _names_used_in_model(model: ParsedModel) -> Set[str]:
    """Identifiers that appear in the (non-``#``) model equations."""
    names: Set[str] = set()
    for equation in model.model_equations:
        if equation.text.strip().startswith("#"):
            continue
        names.update(re.findall(r"[A-Za-z_]\w*", equation.text))
    return names
