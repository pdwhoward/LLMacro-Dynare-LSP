"""Blanchard-Kahn condition checker for Dynare models.

After computing steady state values, linearizes the model around the
steady state and checks whether the number of unstable eigenvalues
equals the number of forward-looking variables -- the Blanchard-Kahn
condition for a unique, stable rational expectations solution.

Requires scipy (optional dependency).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from types import CodeType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, cast

from .parser import Equation, ParsedModel, Position, SourceRange, VarDeclaration
from .diagnostics import Diagnostic, Severity
from .steady_state import (
    _TIME_SUBSCRIPT,
    _TIME_SUBSCRIPT_FUNCTION_NAMES,
    _STEADY_STATE_OP,
    _LOCAL_VAR_DEF,
    _build_eval_env,
    _escape_expr,
    _escape_reserved,
    _exogenous_values,
    _safe_eval_expr,
    _split_equation,
    _strip_equation_tags,
)

if TYPE_CHECKING:
    import numpy


_EXPECTATION_OPERATOR_RE = re.compile(
    r"\b(?:expectation|pac_expectation)\s*\(",
    re.IGNORECASE,
)
_DIFF_OPERATOR_RE = re.compile(r"\bdiff\s*\(", re.IGNORECASE)


def _mask_steady_state_operator(text: str) -> str:
    """Blank ``steady_state(...)`` calls while preserving offsets."""
    chars = list(text)
    opener = re.compile(r"\bsteady_state\s*\(", re.IGNORECASE)
    i = 0
    n = len(text)
    while i < n:
        match = opener.search(text, i)
        if match is None:
            break
        j = match.end()
        depth = 1
        while j < n and depth > 0:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        if depth != 0:
            break
        for idx in range(match.start(), j):
            if chars[idx] != "\n":
                chars[idx] = " "
        i = j
    return "".join(chars)


@dataclass
class BKResult:
    """Result of Blanchard-Kahn eigenvalue analysis."""
    satisfied: bool
    n_unstable: int         # eigenvalues with |lambda| > 1
    n_forward: int          # variables appearing with (+1)
    eigenvalues: list       # List[complex]
    message: str
    forward_variables: List[str] = field(default_factory=list)
    predetermined_variables: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Variable timing extraction
# ---------------------------------------------------------------------------

def _extract_variable_timing(model: ParsedModel) -> Dict[str, Set[int]]:
    """Scan model equations to find which endogenous vars appear at which offsets.

    Returns {var_name: {offsets}} where offsets are integers like -1, 0, +1.
    Only endogenous variables are tracked.
    """
    endo_names = model.endogenous_names()
    predetermined = {
        v.name for v in getattr(model, "predetermined_variables", [])
    }
    timing: Dict[str, Set[int]] = {name: set() for name in endo_names}
    local_defs: Dict[str, str] = {}

    for eq in model.model_equations:
        text = eq.text.strip()
        if text.startswith("#"):
            match = _LOCAL_VAR_DEF.match(text)
            if match:
                local_defs[match.group(1)] = match.group(2).strip()

    def _shift(name: str, offset: int) -> int:
        if name in predetermined:
            return offset - 1
        return offset

    def _merge_from_text(
        text: str,
        visiting: Optional[Set[str]] = None,
        timing_offset: int = 0,
    ) -> None:
        visiting = set(visiting or set())

        # Strip equation tags
        text = _strip_equation_tags(text)
        text = _mask_steady_state_operator(text)
        if _DIFF_OPERATOR_RE.search(text):
            text = _expand_diff_operator(text, endo_names)

        # Find all time-subscripted occurrences: name(+1), name(-1), etc.
        subscripted_spans = set()
        for m in _TIME_SUBSCRIPT.finditer(text):
            name = m.group(1)
            if name.lower() in _TIME_SUBSCRIPT_FUNCTION_NAMES:
                continue
            offset_str = m.group(2).replace(" ", "")
            offset = int(offset_str) + timing_offset
            if name in endo_names:
                timing[name].add(_shift(name, offset))
            elif name in local_defs and name not in visiting:
                _merge_from_text(
                    local_defs[name], visiting | {name}, timing_offset=offset,
                )
            subscripted_spans.add((m.start(), m.end()))

        # Replace steady_state(x) so those don't show as bare identifiers
        text_no_ss = _STEADY_STATE_OP.sub("__SS__", text)

        # Remove subscripted matches to find bare identifiers (time 0)
        # Work backwards to preserve positions
        chars = list(text_no_ss)
        for start, end in sorted(subscripted_spans, reverse=True):
            if end <= len(chars):
                for i in range(start, min(end, len(chars))):
                    chars[i] = " "
        bare_text = "".join(chars)

        # Find remaining bare identifiers that are endogenous
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", bare_text):
            name = m.group(1)
            if name in endo_names:
                timing[name].add(_shift(name, timing_offset))
            elif name in local_defs and name not in visiting:
                _merge_from_text(
                    local_defs[name],
                    visiting | {name},
                    timing_offset=timing_offset,
                )

    for eq in model.dynamic_model_equations():
        text = eq.text.strip()
        _merge_from_text(text)

    return timing


def _format_time_offset(offset: int) -> str:
    if offset > 0:
        return f"+{offset}"
    return str(offset)


def _timed_parameter_reference(model: ParsedModel) -> Optional[str]:
    parameter_names = model.parameter_names()
    if not parameter_names:
        return None
    for eq in model.dynamic_model_equations():
        text = _mask_steady_state_operator(_strip_equation_tags(eq.text))
        for match in _TIME_SUBSCRIPT.finditer(text):
            name = match.group(1)
            if name.lower() in _TIME_SUBSCRIPT_FUNCTION_NAMES:
                continue
            if name in parameter_names:
                return name
    return None


_TIMED_BUILTIN_CONSTANT_RE = re.compile(
    r"\b(pi|inf|nan)\s*\(\s*[+-]?\s*\d+\s*\)",
    re.IGNORECASE,
)


def _unresolved_timed_builtin_constant(model: ParsedModel) -> Optional[str]:
    declared = model.all_declared_names()
    for eq in model.dynamic_model_equations():
        text = _strip_equation_tags(eq.text)
        for match in _TIMED_BUILTIN_CONSTANT_RE.finditer(text):
            name = match.group(1)
            if name not in declared:
                return name
    return None


def _higher_order_timing_reference(
    model: ParsedModel,
    timing: Optional[Dict[str, Set[int]]] = None,
) -> Optional[Tuple[str, int]]:
    endo_names = model.endogenous_names()
    if not endo_names:
        return None
    if timing is None:
        timing = _extract_variable_timing(model)
    for name in sorted(timing):
        for offset in sorted(timing[name], key=lambda value: (abs(value), value)):
            if abs(offset) > 1:
                return name, offset
    predetermined = {
        v.name for v in getattr(model, "predetermined_variables", [])
    }
    for eq in model.dynamic_model_equations():
        text = _mask_steady_state_operator(_strip_equation_tags(eq.text))
        for match in _TIME_SUBSCRIPT.finditer(text):
            name = match.group(1)
            if name.lower() in _TIME_SUBSCRIPT_FUNCTION_NAMES:
                continue
            if name in endo_names:
                raw_offset = int(match.group(2).replace(" ", ""))
                offset = raw_offset - 1 if name in predetermined else raw_offset
                if abs(offset) > 1:
                    return name, offset
    return None


def _synthetic_source_range(model: ParsedModel) -> SourceRange:
    if model.model_block_range is not None:
        return model.model_block_range
    if model.model_equations:
        return model.model_equations[0].range
    return SourceRange(Position(0, 0), Position(0, 0))


def _unique_auxiliary_name(
    kind: str,
    base_name: str,
    order: int,
    used_names: Set[str],
) -> str:
    stem = f"bk_aux_{kind}_{base_name}_{order}"
    candidate = stem
    disambiguator = 2
    while candidate in used_names:
        candidate = f"{stem}_{disambiguator}"
        disambiguator += 1
    used_names.add(candidate)
    return candidate


def _reference_for_effective_offset(
    name: str,
    offset: int,
    predetermined_names: Set[str],
) -> str:
    raw_offset = offset + 1 if name in predetermined_names else offset
    if raw_offset == 0:
        return name
    return f"{name}({_format_time_offset(raw_offset)})"


def _steady_state_operator_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    opener = re.compile(r"\bsteady_state\s*\(", re.IGNORECASE)
    i = 0
    n = len(text)
    while i < n:
        match = opener.search(text, i)
        if match is None:
            break
        j = match.end()
        depth = 1
        while j < n and depth > 0:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        if depth != 0:
            break
        spans.append((match.start(), j))
        i = j
    return spans


def _inside_spans(start: int, end: int, spans: List[Tuple[int, int]]) -> bool:
    return any(span_start <= start and end <= span_end for span_start, span_end in spans)


def _replace_higher_order_timing_refs(
    text: str,
    endo_names: Set[str],
    predetermined_names: Set[str],
    lead_auxiliaries: Dict[Tuple[str, int], str],
    lag_auxiliaries: Dict[Tuple[str, int], str],
) -> str:
    steady_state_spans = _steady_state_operator_spans(text)

    def _time_repl(match: re.Match) -> str:
        if _inside_spans(match.start(), match.end(), steady_state_spans):
            return match.group(0)
        name = match.group(1)
        if name.lower() in _TIME_SUBSCRIPT_FUNCTION_NAMES or name not in endo_names:
            return match.group(0)
        raw_offset = int(match.group(2).replace(" ", ""))
        offset = raw_offset - 1 if name in predetermined_names else raw_offset
        if offset > 1:
            return f"{lead_auxiliaries[(name, offset - 1)]}(+1)"
        if offset < -1:
            return lag_auxiliaries[(name, -offset - 1)]
        return match.group(0)

    return _TIME_SUBSCRIPT.sub(_time_repl, text)


def _auxiliary_transformed_model(
    model: ParsedModel,
    ss_values: Dict[str, float],
    timing: Optional[Dict[str, Set[int]]] = None,
) -> Optional[Tuple[ParsedModel, Dict[str, float]]]:
    endo_names = model.endogenous_names()
    if not endo_names:
        return None
    if timing is None:
        timing = _extract_variable_timing(model)
    max_leads = {
        name: max(offset for offset in offsets if offset > 1)
        for name, offsets in timing.items()
        if any(offset > 1 for offset in offsets)
    }
    max_lags = {
        name: max(-offset for offset in offsets if offset < -1)
        for name, offsets in timing.items()
        if any(offset < -1 for offset in offsets)
    }
    if not max_leads and not max_lags:
        return None

    predetermined_names = {
        v.name for v in getattr(model, "predetermined_variables", [])
    }
    local_raw_defs: Dict[str, str] = {}
    for eq in model.model_equations:
        text = eq.text.strip()
        if not text.startswith("#"):
            continue
        match = _LOCAL_VAR_DEF.match(text)
        if match:
            local_raw_defs[match.group(1)] = match.group(2).strip()

    used_names = set(model.all_declared_names())
    used_names.update(v.name for v in model.predetermined_variables)
    used_names.update(assignment.name for assignment in model.helper_assignments)
    used_names.update(local_raw_defs)

    source_range = _synthetic_source_range(model)
    aux_declarations: List[VarDeclaration] = []
    aux_predetermined_declarations: List[VarDeclaration] = []
    aux_source_names: Dict[str, str] = {}
    lead_auxiliaries: Dict[Tuple[str, int], str] = {}
    lag_auxiliaries: Dict[Tuple[str, int], str] = {}

    for name in sorted(max_leads):
        for order in range(1, max_leads[name]):
            aux_name = _unique_auxiliary_name("lead", name, order, used_names)
            lead_auxiliaries[(name, order)] = aux_name
            aux_source_names[aux_name] = name
            aux_declarations.append(VarDeclaration(aux_name, source_range))

    for name in sorted(max_lags):
        for order in range(1, max_lags[name]):
            aux_name = _unique_auxiliary_name("lag", name, order, used_names)
            lag_auxiliaries[(name, order)] = aux_name
            aux_source_names[aux_name] = name
            declaration = VarDeclaration(aux_name, source_range)
            aux_declarations.append(declaration)
            aux_predetermined_declarations.append(declaration)

    transformed_equations: List[Equation] = []
    for eq in model.dynamic_model_equations():
        expanded = _expand_model_local_refs(
            eq.text.strip(),
            local_raw_defs,
            endo_names,
        )
        expanded = _expand_diff_operator(expanded, endo_names)
        transformed_equations.append(replace(
            eq,
            text=_replace_higher_order_timing_refs(
                expanded,
                endo_names,
                predetermined_names,
                lead_auxiliaries,
                lag_auxiliaries,
            ),
        ))

    auxiliary_equations: List[Equation] = []
    for name in sorted(max_leads):
        for order in range(1, max_leads[name]):
            aux_name = lead_auxiliaries[(name, order)]
            if order == 1:
                rhs = _reference_for_effective_offset(
                    name,
                    1,
                    predetermined_names,
                )
            else:
                rhs = f"{lead_auxiliaries[(name, order - 1)]}(+1)"
            auxiliary_equations.append(Equation(
                text=f"{aux_name} = {rhs}",
                name="",
                range=source_range,
            ))

    for name in sorted(max_lags):
        for order in range(1, max_lags[name]):
            aux_name = lag_auxiliaries[(name, order)]
            if order == 1:
                rhs = _reference_for_effective_offset(
                    name,
                    -1,
                    predetermined_names,
                )
            else:
                rhs = lag_auxiliaries[(name, order - 1)]
            auxiliary_equations.append(Equation(
                text=f"{aux_name}(+1) = {rhs}",
                name="",
                range=source_range,
            ))

    transformed_ss = dict(ss_values)
    for aux_name, source_name in aux_source_names.items():
        transformed_ss[aux_name] = ss_values.get(source_name, 0.0)

    transformed = replace(
        model,
        endogenous=list(model.endogenous) + aux_declarations,
        predetermined_variables=(
            list(model.predetermined_variables) + aux_predetermined_declarations
        ),
        model_equations=transformed_equations + auxiliary_equations,
    )
    return transformed, transformed_ss


def _shift_expression_timing(expr: str, offset: int, endo_names: Set[str]) -> str:
    """Shift every endogenous reference in *expr* by *offset* periods."""
    if offset == 0:
        return expr

    placeholders: Dict[str, str] = {}

    def _time_repl(m: re.Match) -> str:
        name = m.group(1)
        if name.lower() in _TIME_SUBSCRIPT_FUNCTION_NAMES or name not in endo_names:
            return m.group(0)
        shifted = int(m.group(2).replace(" ", "")) + offset
        token = f"__BK_TIMED_{len(placeholders)}__"
        if shifted == 0:
            placeholders[token] = name
        else:
            placeholders[token] = f"{name}({_format_time_offset(shifted)})"
        return token

    shifted_expr = _TIME_SUBSCRIPT.sub(_time_repl, expr)

    def _bare_repl(m: re.Match) -> str:
        name = m.group(1)
        if name in endo_names:
            return f"{name}({_format_time_offset(offset)})"
        return name

    shifted_expr = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", _bare_repl, shifted_expr)
    for token, replacement in placeholders.items():
        shifted_expr = shifted_expr.replace(token, replacement)
    return shifted_expr


def _expand_model_local_refs(
    text: str,
    local_defs: Dict[str, str],
    endo_names: Set[str],
    visiting: Optional[Set[str]] = None,
) -> str:
    """Inline model-local references so timed local uses preserve endogenous timing."""
    visiting = set(visiting or set())

    def _expand_local(name: str, offset: int) -> str:
        if name in visiting or name not in local_defs:
            return name
        expanded = _expand_model_local_refs(
            local_defs[name],
            local_defs,
            endo_names,
            visiting | {name},
        )
        return f"({_shift_expression_timing(expanded, offset, endo_names)})"

    def _time_repl(m: re.Match) -> str:
        name = m.group(1)
        if name.lower() in _TIME_SUBSCRIPT_FUNCTION_NAMES:
            return m.group(0)
        if name in local_defs:
            return _expand_local(name, int(m.group(2).replace(" ", "")))
        return m.group(0)

    expanded = _TIME_SUBSCRIPT.sub(_time_repl, text)

    def _bare_repl(m: re.Match) -> str:
        name = m.group(1)
        if name in local_defs:
            return _expand_local(name, 0)
        return name

    return re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", _bare_repl, expanded)


def _expand_diff_operator(text: str, endo_names: Set[str]) -> str:
    """Expand Dynare ``diff(expr)`` as ``expr - expr(-1)`` for BK checks."""
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        match = _DIFF_OPERATOR_RE.search(text, i)
        if match is None:
            out.append(text[i:])
            break
        out.append(text[i:match.start()])
        j = match.end()
        depth = 1
        while j < n and depth > 0:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        if depth != 0:
            out.append(text[match.start():])
            break
        inner = _expand_diff_operator(text[match.end():j - 1].strip(), endo_names)
        lagged = _shift_expression_timing(inner, -1, endo_names)
        out.append(f"(({inner}) - ({lagged}))")
        i = j
    return "".join(out)


# ---------------------------------------------------------------------------
# BK expression preparation (preserves timing)
# ---------------------------------------------------------------------------

def _prepare_bk_expression(
    eq_text: str,
    endo_names: set,
    predetermined_names: Optional[Set[str]] = None,
) -> str:
    """Transform equation text preserving timing as separate symbols.

    C(+1) -> C__lead, K(-1) -> K__lag, bare C -> C__curr
    Parameters and exogenous stay as-is (no suffix).
    STEADY_STATE(expr) is handled as a steady-state constant.
    ^ -> **

    Also collapses Dynare's ``EXPECTATION(h)(expr)`` operator to ``(expr)``
    BEFORE time-subscript rewriting — otherwise the ``(h)`` parameter
    would be mistaken for a time subscript and the operator would become
    an undefined function (silently zeroing residuals at jacobian
    evaluation time).
    """
    from .steady_state import _collapse_expectation_operator
    result = _collapse_expectation_operator(_strip_equation_tags(eq_text))
    result = _expand_diff_operator(result, set(endo_names))
    predetermined = predetermined_names or set()
    placeholders: Dict[str, str] = {}

    def _placeholder(replacement: str) -> str:
        token = f"__BK_TIMED_PLACEHOLDER_{len(placeholders)}__"
        placeholders[token] = replacement
        return token

    def _suffix_for_offset(name: str, offset: int) -> str:
        if name in predetermined:
            offset -= 1
        escaped = _escape_reserved(name)
        if offset > 0:
            return f"{escaped}__lead"
        if offset < 0:
            return f"{escaped}__lag"
        return f"{escaped}__curr"

    result = _replace_steady_state_operator_with_constants(result, endo_names)

    # Replace time-subscripted variables
    def _time_repl(m: re.Match) -> str:
        name = m.group(1)
        offset_str = m.group(2).replace(" ", "")
        if name.lower() in _TIME_SUBSCRIPT_FUNCTION_NAMES:
            return m.group(0)
        offset = int(offset_str)
        if name not in endo_names:
            return name  # non-endogenous: drop timing
        # Escape Python-reserved Dynare identifiers like ``lambda`` before
        # appending the lead/lag/curr suffix; otherwise the suffixed
        # name lands in the expression as a literal ``lambda__lead``
        # which the eval env (which uses _escape_reserved) doesn't know
        # about, zeroing the Jacobian row.
        return _placeholder(_suffix_for_offset(name, offset))
    result = _TIME_SUBSCRIPT.sub(_time_repl, result)

    # Replace bare endogenous identifiers with __curr suffix
    # Must be careful not to re-replace already-suffixed names.  Use the
    # same _escape_reserved rule so ``lambda`` -> ``_dyn_lambda`` -> ``_dyn_lambda__curr``.
    def _bare_repl(m: re.Match) -> str:
        name = m.group(1)
        if name in endo_names:
            return _suffix_for_offset(name, 0)
        return name
    result = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", _bare_repl, result)

    # Convert ^ to **
    result = result.replace("^", "**")
    for token, replacement in placeholders.items():
        result = result.replace(token, replacement)

    return result


def _replace_steady_state_operator_with_constants(
    text: str,
    endo_names: Set[str],
) -> str:
    """Map ``steady_state(expr)`` to variables that are never perturbed."""
    out: List[str] = []
    i = 0
    n = len(text)
    endo_set = set(endo_names)
    opener = re.compile(r"\bsteady_state\s*\(", re.IGNORECASE)
    while i < n:
        match = opener.search(text, i)
        if match is None:
            out.append(text[i:])
            break
        out.append(text[i:match.start()])
        j = match.end()
        depth = 1
        while j < n and depth > 0:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        if depth != 0:
            i = j
            continue
        inner = text[match.end():j - 1]

        def _const_repl(name_match: re.Match) -> str:
            name = name_match.group(1)
            if name in endo_set:
                return f"{_escape_reserved(name)}__ss"
            return name

        const_inner = _TIME_SUBSCRIPT.sub(_const_repl, inner)
        const_inner = re.sub(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\b",
            lambda name_match: (
                f"{_escape_reserved(name_match.group(1))}__ss"
                if name_match.group(1) in endo_set
                else name_match.group(1)
            ),
            const_inner,
        )
        out.append(f"({const_inner})")
        i = j
    return "".join(out)


# ---------------------------------------------------------------------------
# Jacobian computation (numerical, central differences)
# ---------------------------------------------------------------------------

def _compute_jacobian(
    model: ParsedModel,
    ss_values: Dict[str, float],
    param_overrides: Optional[Dict[str, float]] = None,
    dense: bool = False,
) -> Tuple["numpy.ndarray", "numpy.ndarray", "numpy.ndarray"]:
    """Compute Jacobian matrices f_yp, f_y0, f_ym via finite differences.

    Returns three n_eq x n_var matrices:
      f_yp: derivatives w.r.t. y(+1)  (lead)
      f_y0: derivatives w.r.t. y(t)   (current)
      f_ym: derivatives w.r.t. y(t-1) (lag)

    Each equation references only a handful of the model's variables, so the
    default **sparse** path perturbs a variable and re-evaluates only the
    equations that actually contain it -- every other row of that column is a
    structural zero.  On a 437-equation model this is ~90x fewer equation
    evaluations than the dense path (~57s -> ~1s).  The result is numerically
    identical to the dense path (perturbing a variable cannot change the
    residual of an equation that does not contain it); ``dense=True`` forces the
    reference all-equations loop, kept as that equivalence oracle and a fallback
    (see ``test_bk_jacobian_sparse_equals_dense``).
    """
    import numpy as np

    endo_names = [v.name for v in model.endogenous]
    endo_set = set(endo_names)
    predetermined = {
        v.name for v in getattr(model, "predetermined_variables", [])
    }
    exogenous = model.exogenous_names()
    exogenous_values = _exogenous_values(model, param_overrides=param_overrides)
    # Match the solver/diagnostics contract: parameters assigned inside
    # steady_state_model are valid and must be visible during linearization.
    from .solver import _effective_param_values
    params = _effective_param_values(model, param_overrides=param_overrides)
    helper_values = {
        assignment.name: assignment.value
        for assignment in model.helper_assignments
        if assignment.value is not None
    }
    n_var = len(endo_names)

    # Pre-process equations: separate local var defs from real equations
    local_var_defs: List[Tuple[str, str]] = []  # (name, bk_expr)
    local_raw_defs: Dict[str, str] = {}
    prepared_equations: List[Tuple[str, Optional[str]]] = []

    for eq in model.model_equations:
        text = eq.text.strip()
        if not text.startswith("#"):
            continue
        match = _LOCAL_VAR_DEF.match(text)
        if not match:
            continue
        local_name = match.group(1)
        local_expr = match.group(2).strip()
        local_raw_defs[local_name] = local_expr
        bk_expr = _escape_expr(
            _prepare_bk_expression(local_expr, endo_set, predetermined)
        )
        local_var_defs.append((local_name, bk_expr))

    for eq in model.dynamic_model_equations():
        text = eq.text.strip()

        text = _expand_model_local_refs(text, local_raw_defs, endo_set)
        bk_text = _escape_expr(_prepare_bk_expression(text, endo_set, predetermined))
        try:
            lhs, rhs = _split_equation(bk_text)
            prepared_equations.append((lhs, rhs))
        except ValueError:
            prepared_equations.append((bk_text, None))

    n_eq = len(prepared_equations)

    def _build_bk_env(var_vals: Dict[str, float]) -> dict:
        """Build eval environment with __lead/__curr/__lag keys."""
        base_vals: Dict[str, float] = dict(params)
        base_vals.update(helper_values)
        for exo in exogenous:
            base_vals[exo] = exogenous_values.get(exo, 0.0)
        env = _build_eval_env(base_vals)
        for exo in exogenous:
            value = exogenous_values.get(exo, 0.0)
            env[exo] = value
            env[_escape_reserved(exo)] = value
        # Add timed endogenous variables
        env.update(var_vals)
        return env

    def _eval_residuals(var_vals: Dict[str, float]) -> "numpy.ndarray":
        """Evaluate all equation residuals given timed variable values."""
        env = _build_bk_env(var_vals)

        # Evaluate local var defs
        for local_name, local_expr in local_var_defs:
            val, err = _safe_eval_expr(local_expr, env)
            if val is None or err:
                raise ValueError(
                    f"failed to evaluate model-local variable '{local_name}': {err}",
                )
            env[_escape_reserved(local_name)] = val

        residuals = np.zeros(n_eq)
        for i, (lhs, rhs) in enumerate(prepared_equations):
            if rhs is not None:
                lhs_val, err1 = _safe_eval_expr(lhs, env)
                rhs_val, err2 = _safe_eval_expr(rhs, env)
                if err1 or err2 or lhs_val is None or rhs_val is None:
                    detail = err1 or err2 or "non-finite residual"
                    raise ValueError(
                        f"failed to evaluate equation {i + 1}: {detail}",
                    )
                else:
                    residuals[i] = lhs_val - rhs_val
            else:
                val, err = _safe_eval_expr(lhs, env)
                if val is None or err:
                    raise ValueError(
                        f"failed to evaluate equation {i + 1}: {err}",
                    )
                residuals[i] = val
        return residuals

    # Build baseline timed values at steady state
    base_vals: Dict[str, float] = {}
    for name in endo_names:
        ss_val = ss_values.get(name, 0.0)
        esc = _escape_reserved(name)
        base_vals[f"{esc}__lead"] = ss_val
        base_vals[f"{esc}__curr"] = ss_val
        base_vals[f"{esc}__lag"] = ss_val
        base_vals[f"{esc}__ss"] = ss_val

    # Initialize Jacobian matrices
    f_yp = np.zeros((n_eq, n_var))
    f_y0 = np.zeros((n_eq, n_var))
    f_ym = np.zeros((n_eq, n_var))

    # Central finite differences, with a one-sided fallback at domain
    # boundaries (e.g. a variable at SS=0 inside log()/sqrt(), where the -h
    # step leaves the function's domain).  Falling back to a one-sided
    # difference there avoids spuriously skipping the whole BK check on an
    # otherwise well-posed model; models away from a boundary are unaffected.
    f_base = _eval_residuals(base_vals)

    if dense:
        # Reference path: perturb every variable and re-evaluate all equations.
        for j, name in enumerate(endo_names):
            ss_val = ss_values.get(name, 0.0)
            h = max(abs(ss_val) * 1e-7, 1e-8)
            esc = _escape_reserved(name)

            for suffix, target_mat in [("__lead", f_yp), ("__curr", f_y0), ("__lag", f_ym)]:
                key = f"{esc}{suffix}"

                vals_plus = dict(base_vals)
                vals_plus[key] = ss_val + h
                try:
                    f_plus = _eval_residuals(vals_plus)
                except Exception:
                    f_plus = None

                vals_minus = dict(base_vals)
                vals_minus[key] = ss_val - h
                try:
                    f_minus = _eval_residuals(vals_minus)
                except Exception:
                    f_minus = None

                if f_plus is not None and f_minus is not None:
                    target_mat[:, j] = (f_plus - f_minus) / (2 * h)
                elif f_plus is not None:
                    target_mat[:, j] = (f_plus - f_base) / h
                elif f_minus is not None:
                    target_mat[:, j] = (f_base - f_minus) / h
                else:
                    raise ValueError(
                        f"could not evaluate Jacobian column for '{name}'"
                    )
        return f_yp, f_y0, f_ym

    # --- sparse path: re-evaluate only the equations a perturbed token reaches ---
    import re as _re

    _TIMED = _re.compile(r"\b([A-Za-z_]\w*__(?:lead|curr|lag))\b")
    _WORD = _re.compile(r"\b([A-Za-z_]\w*)\b")
    esc_local_index = {
        _escape_reserved(lname): li
        for li, (lname, _expr) in enumerate(local_var_defs)
    }
    # Each model-local's transitive set of timed tokens (defs are topologically
    # ordered: a local may only reference earlier locals).
    local_closure: List[set] = []
    for li, (lname, lexpr) in enumerate(local_var_defs):
        cl = set(_TIMED.findall(lexpr))
        for w in _WORD.findall(lexpr):
            lj = esc_local_index.get(w)
            if lj is not None and lj < li:
                cl |= local_closure[lj]
        local_closure.append(cl)
    # token -> equation rows that depend on it, and -> locals to re-evaluate.
    token_rows: Dict[str, List[int]] = {}
    token_locals: Dict[str, set] = {}
    for i, (lhs, rhs) in enumerate(prepared_equations):
        body = lhs if rhs is None else f"{lhs} {rhs}"
        toks = set(_TIMED.findall(body))
        used_locals = {esc_local_index[w] for w in _WORD.findall(body)
                       if w in esc_local_index}
        for li in used_locals:
            toks |= local_closure[li]
        for t in toks:
            token_rows.setdefault(t, []).append(i)
    for li, cl in enumerate(local_closure):
        for t in cl:
            token_locals.setdefault(t, set()).add(li)

    base_env = _build_bk_env(base_vals)
    for lname, lexpr in local_var_defs:
        val, err = _safe_eval_expr(lexpr, base_env)
        if val is None or err:
            raise ValueError(
                f"failed to evaluate model-local variable '{lname}': {err}",
            )
        base_env[_escape_reserved(lname)] = val

    # Pre-compile all equation expressions and local-var expressions so that
    # _eval_rows can use direct eval(code_obj, ...) instead of routing through
    # _safe_eval_expr (which calls _validate_ast + ast.parse on every call).
    # The expressions were already AST-validated during _prepare_bk_expression /
    # _escape_expr; we re-validate once here and fall back to _safe_eval_expr
    # for any expression that cannot be compiled.
    _compiled_lhs: List[Optional[CodeType]] = []
    _compiled_rhs: List[Optional[CodeType]] = []
    for lhs, rhs in prepared_equations:
        try:
            _compiled_lhs.append(compile(lhs, "<bk-lhs>", "eval"))
        except Exception:
            _compiled_lhs.append(None)
        if rhs is not None:
            try:
                _compiled_rhs.append(compile(rhs, "<bk-rhs>", "eval"))
            except Exception:
                _compiled_rhs.append(None)
        else:
            _compiled_rhs.append(None)

    _compiled_local_exprs: List[Optional[CodeType]] = []
    for _lname, lexpr in local_var_defs:
        try:
            _compiled_local_exprs.append(compile(lexpr, "<bk-local>", "eval"))
        except Exception:
            _compiled_local_exprs.append(None)

    _NO_BUILTINS: dict = {"__builtins__": {}}

    def _eval_rows(key: str, key_val: float, aff_locals, rows):
        env = dict(base_env)
        env[key] = key_val
        for li in aff_locals:
            lname, lexpr = local_var_defs[li]
            code = _compiled_local_exprs[li]
            if code is not None:
                try:
                    v = eval(code, _NO_BUILTINS, env)  # noqa: S307
                    if isinstance(v, complex):
                        if abs(v.imag) < 1e-10:
                            v = float(v.real)
                        else:
                            return None
                    env[_escape_reserved(lname)] = float(v)
                except Exception:
                    return None
            else:
                val, err = _safe_eval_expr(lexpr, env)
                if val is None or err:
                    return None
                env[_escape_reserved(lname)] = val
        out = np.empty(len(rows))
        for ridx, i in enumerate(rows):
            lhs_code = _compiled_lhs[i]
            rhs_code = _compiled_rhs[i]
            lhs_str, rhs_str = prepared_equations[i]
            if rhs_str is not None:
                # Try fast compiled path first.
                if lhs_code is not None and rhs_code is not None:
                    try:
                        lv = eval(lhs_code, _NO_BUILTINS, env)  # noqa: S307
                        rv = eval(rhs_code, _NO_BUILTINS, env)  # noqa: S307
                        if isinstance(lv, complex):
                            if abs(lv.imag) < 1e-10:
                                lv = float(lv.real)
                            else:
                                return None
                        if isinstance(rv, complex):
                            if abs(rv.imag) < 1e-10:
                                rv = float(rv.real)
                            else:
                                return None
                        out[ridx] = float(lv) - float(rv)
                        continue
                    except Exception:
                        pass
                # Fall back to _safe_eval_expr for this row.
                lv2, e1 = _safe_eval_expr(lhs_str, env)
                rv2, e2 = _safe_eval_expr(rhs_str, env)
                if lv2 is None or rv2 is None or e1 or e2:
                    return None
                out[ridx] = lv2 - rv2
            else:
                if lhs_code is not None:
                    try:
                        v = eval(lhs_code, _NO_BUILTINS, env)  # noqa: S307
                        if isinstance(v, complex):
                            if abs(v.imag) < 1e-10:
                                v = float(v.real)
                            else:
                                return None
                        out[ridx] = float(v)
                        continue
                    except Exception:
                        pass
                v2, e = _safe_eval_expr(lhs_str, env)
                if v2 is None or e:
                    return None
                out[ridx] = v2
        return out

    for j, name in enumerate(endo_names):
        ss_val = ss_values.get(name, 0.0)
        h = max(abs(ss_val) * 1e-7, 1e-8)
        esc = _escape_reserved(name)
        for suffix, target_mat in [("__lead", f_yp), ("__curr", f_y0), ("__lag", f_ym)]:
            key = f"{esc}{suffix}"
            rows = token_rows.get(key)
            if not rows:
                continue  # column is structurally zero
            aff_locals = sorted(token_locals.get(key, ()))
            rows_arr = np.asarray(rows, dtype=int)
            base_rows = f_base[rows_arr]

            f_plus = _eval_rows(key, ss_val + h, aff_locals, rows)
            f_minus = _eval_rows(key, ss_val - h, aff_locals, rows)
            if f_plus is not None and f_minus is not None:
                target_mat[rows_arr, j] = (f_plus - f_minus) / (2 * h)
            elif f_plus is not None:
                target_mat[rows_arr, j] = (f_plus - base_rows) / h
            elif f_minus is not None:
                target_mat[rows_arr, j] = (base_rows - f_minus) / h
            else:
                raise ValueError(
                    f"could not evaluate Jacobian column for '{name}'"
                )

    return f_yp, f_y0, f_ym


# ---------------------------------------------------------------------------
# Minimal dynamic generalized eigenvalue problem
# ---------------------------------------------------------------------------

def _static_reduction_projection(
    f_y0: "numpy.ndarray",
    static_indices: List[int],
) -> "numpy.ndarray":
    """Return a row projection that eliminates static current variables."""
    import numpy as np

    n_eq = f_y0.shape[0]
    n_static = len(static_indices)
    if n_static == 0:
        return np.eye(n_eq)

    static_block = f_y0[:, static_indices]
    static_rank = int(np.linalg.matrix_rank(static_block))
    if static_rank < n_static:
        raise ValueError(
            "singular or rank-deficient static block "
            f"(rank {static_rank} for {n_static} static variable(s))",
        )

    q, _r = np.linalg.qr(static_block, mode="complete")
    return q[:, n_static:].T


def _form_minimal_dynamic_pencil(
    f_yp: "numpy.ndarray",
    f_y0: "numpy.ndarray",
    f_ym: "numpy.ndarray",
    predetermined_indices: List[int],
    forward_indices: List[int],
    static_indices: List[int],
) -> Tuple["numpy.ndarray", "numpy.ndarray"]:
    """Form the minimal first-order BK pencil.

    The reduced equations are arranged as

        B * [y_P(t), E_t y_F(t+1)]' =
        A * [y_P(t-1), y_F(t)]'

    where P are variables appearing with a lag and F are variables
    appearing with a lead.  Variables appearing with both a lead and a
    lag need one identity row because their current value appears in both
    coordinate systems.
    """
    import numpy as np

    projection = _static_reduction_projection(f_y0, static_indices)
    g_yp = projection @ f_yp
    g_y0 = projection @ f_y0
    g_ym = projection @ f_ym

    n_predetermined = len(predetermined_indices)
    n_forward = len(forward_indices)
    n_dynamic = n_predetermined + n_forward
    predetermined_set = set(predetermined_indices)
    forward_set = set(forward_indices)
    mixed_indices = [
        idx for idx in predetermined_indices
        if idx in forward_set
    ]

    # For mixed variables, the current column is represented by the
    # predetermined-current coordinate; the identity row below links it to
    # the forward-current coordinate.  Including it in both places doubles
    # the current coefficient and changes the characteristic roots.
    forward_current = -g_y0[:, forward_indices]
    for col, idx in enumerate(forward_indices):
        if idx in predetermined_set:
            forward_current[:, col] = 0.0

    lhs_blocks = [
        g_y0[:, predetermined_indices],
        g_yp[:, forward_indices],
    ]
    rhs_blocks = [
        -g_ym[:, predetermined_indices],
        forward_current,
    ]
    lhs = np.hstack(lhs_blocks)
    rhs = np.hstack(rhs_blocks)

    if mixed_indices:
        pred_pos = {
            idx: pos for pos, idx in enumerate(predetermined_indices)
        }
        forward_pos = {
            idx: n_predetermined + pos
            for pos, idx in enumerate(forward_indices)
        }
        lhs_id = np.zeros((len(mixed_indices), n_dynamic))
        rhs_id = np.zeros((len(mixed_indices), n_dynamic))
        for row, idx in enumerate(mixed_indices):
            lhs_id[row, pred_pos[idx]] = 1.0
            rhs_id[row, forward_pos[idx]] = 1.0
        lhs = np.vstack([lhs, lhs_id])
        rhs = np.vstack([rhs, rhs_id])

    expected_shape = (n_dynamic, n_dynamic)
    if lhs.shape != expected_shape or rhs.shape != expected_shape:
        raise ValueError(
            "singular or rank-deficient dynamic reduction "
            f"(got pencil shapes {rhs.shape} and {lhs.shape}, "
            f"expected {expected_shape})",
        )

    return rhs, lhs


def _generalized_eigenvalues(
    alpha: "numpy.ndarray",
    beta: "numpy.ndarray",
) -> List[complex]:
    """Convert QZ alpha/beta pairs to Python complex eigenvalues."""
    import numpy as np

    eigenvalues: List[complex] = []
    for a, b in zip(alpha, beta):
        if b == 0:
            if a == 0:
                eigenvalues.append(complex(np.nan, np.nan))
            else:
                eigenvalues.append(complex(np.inf, 0.0))
        else:
            eigenvalues.append(complex(a / b))
    return eigenvalues


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_blanchard_kahn(
    model: ParsedModel,
    ss_values: Dict[str, float],
    unit_root_tol: float = 1e-6,
    *,
    _allow_auxiliary_transform: bool = True,
) -> BKResult:
    """Check the Blanchard-Kahn condition for a linearized model.

    Requires scipy for eigenvalue computation.

    Parameters:
        model: Parsed Dynare model
        ss_values: Steady state values (from solver)
        unit_root_tol: Tolerance for classifying eigenvalues (|lambda| > 1 + tol)

    Returns:
        BKResult with eigenvalue counts and pass/fail status
    """
    try:
        import numpy as np
        from scipy.linalg import ordqz
    except ImportError:
        return BKResult(
            satisfied=False, n_unstable=0, n_forward=0,
            eigenvalues=[], message="scipy required for BK check",
        )

    # Extract variable timing once; thread it into helpers to avoid recomputation.
    timing = _extract_variable_timing(model)
    endo_names = [v.name for v in model.endogenous]

    forward_vars = sorted(
        name for name, offsets in timing.items()
        if any(o > 0 for o in offsets)
    )
    predetermined_vars = sorted(
        name for name, offsets in timing.items()
        if any(o < 0 for o in offsets)
    )
    n_forward = len(forward_vars)
    forward_set = set(forward_vars)
    predetermined_set = set(predetermined_vars)
    forward_indices = [
        i for i, name in enumerate(endo_names)
        if name in forward_set
    ]
    predetermined_indices = [
        i for i, name in enumerate(endo_names)
        if name in predetermined_set
    ]
    static_indices = [
        i for i, name in enumerate(endo_names)
        if name not in forward_set and name not in predetermined_set
    ]

    timed_constant = _unresolved_timed_builtin_constant(model)
    if timed_constant is not None:
        return BKResult(
            satisfied=False, n_unstable=0, n_forward=n_forward,
            eigenvalues=[],
            message=(
                f"Blanchard-Kahn check skipped: '{timed_constant}' is used "
                "with a lead or lag but is not declared. Dynare time "
                "subscripts apply to endogenous variables."
            ),
            forward_variables=forward_vars,
            predetermined_variables=predetermined_vars,
        )

    # Reject obviously-malformed inputs up front so the companion-matrix
    # construction (which assumes n_equations == n_variables) doesn't
    # raise an unhelpful ValueError from numpy.block().
    n_endo = len(endo_names)
    real_eqs = model.dynamic_model_equations()
    if any(_EXPECTATION_OPERATOR_RE.search(eq.text) for eq in real_eqs):
        return BKResult(
            satisfied=False, n_unstable=0, n_forward=n_forward,
            eigenvalues=[],
            message=(
                "Blanchard-Kahn check skipped: EXPECTATION operators require "
                "Dynare's auxiliary-variable transformation, which this LSP "
                "does not implement."
            ),
            forward_variables=forward_vars,
            predetermined_variables=predetermined_vars,
        )

    higher_order = _higher_order_timing_reference(model, timing)
    if higher_order is not None:
        if _allow_auxiliary_transform:
            transformed = _auxiliary_transformed_model(model, ss_values, timing)
            if transformed is not None:
                transformed_model, transformed_ss = transformed
                if _higher_order_timing_reference(transformed_model) is None:
                    _result = check_blanchard_kahn(
                        transformed_model,
                        transformed_ss,
                        unit_root_tol,
                        _allow_auxiliary_transform=False,
                    )
                    # Replace the transformed model's variable lists with those
                    # from the original model so internal auxiliary variable names
                    # (e.g. 'bk_aux_lead_y_1') never leak to callers.
                    return replace(
                        _result,
                        forward_variables=forward_vars,
                        predetermined_variables=predetermined_vars,
                    )
        name, offset = higher_order
        return BKResult(
            satisfied=False, n_unstable=0, n_forward=n_forward,
            eigenvalues=[],
            message=(
                "Blanchard-Kahn check skipped: higher-order lead/lag "
                f"{name}({offset:+d}) requires Dynare's auxiliary-variable "
                "transformation, which this LSP does not implement."
            ),
            forward_variables=forward_vars,
            predetermined_variables=predetermined_vars,
        )
    if n_endo == 0:
        return BKResult(
            satisfied=False, n_unstable=0, n_forward=n_forward,
            eigenvalues=[],
            message="Blanchard-Kahn check skipped: model has no endogenous variables.",
            forward_variables=forward_vars,
            predetermined_variables=predetermined_vars,
        )
    if len(real_eqs) != n_endo:
        return BKResult(
            satisfied=False, n_unstable=0, n_forward=n_forward,
            eigenvalues=[],
            message=(
                f"Blanchard-Kahn check skipped: {len(real_eqs)} equation(s) "
                f"vs {n_endo} endogenous variable(s).  Fix the count mismatch "
                f"(see E010) before running BK analysis."
            ),
            forward_variables=forward_vars,
            predetermined_variables=predetermined_vars,
        )

    # Compute Jacobian
    try:
        f_yp, f_y0, f_ym = _compute_jacobian(model, ss_values)
    except ValueError as e:
        return BKResult(
            satisfied=False, n_unstable=0, n_forward=n_forward,
            eigenvalues=[],
            message=(
                "Blanchard-Kahn check skipped: steady state could not be "
                f"evaluated for linearization ({e})."
            ),
            forward_variables=forward_vars,
            predetermined_variables=predetermined_vars,
        )

    linearization = np.hstack([f_yp, f_y0, f_ym])
    rank = int(np.linalg.matrix_rank(linearization))
    if rank < n_endo:
        return BKResult(
            satisfied=False, n_unstable=0, n_forward=n_forward,
            eigenvalues=[],
            message=(
                "Blanchard-Kahn check skipped: singular or rank-deficient "
                f"linearization (rank {rank} for {n_endo} endogenous "
                "variable(s))."
            ),
            forward_variables=forward_vars,
            predetermined_variables=predetermined_vars,
        )

    try:
        A, B = _form_minimal_dynamic_pencil(
            f_yp,
            f_y0,
            f_ym,
            predetermined_indices,
            forward_indices,
            static_indices,
        )
    except Exception as e:
        return BKResult(
            satisfied=False, n_unstable=0, n_forward=n_forward,
            eigenvalues=[],
            message=(
                "Blanchard-Kahn check skipped: singular or rank-deficient "
                f"dynamic linearization ({e})."
            ),
            forward_variables=forward_vars,
            predetermined_variables=predetermined_vars,
        )

    qz_criterium = 1 + unit_root_tol

    def _is_explosive(alpha, beta):
        return np.abs(alpha) > qz_criterium * np.abs(beta)

    # Compute generalized eigenvalues.  A zero beta is a genuine infinite
    # generalized root in the minimal pencil and counts as explosive.
    try:
        if A.shape[0] == 0:
            alpha = np.array([], dtype=complex)
            beta = np.array([], dtype=complex)
            z = np.zeros((0, 0), dtype=complex)
        else:
            _aa, _bb, alpha, beta, _q, z = ordqz(
                A, B, sort=cast(Any, _is_explosive), output="complex",
            )
    except Exception as e:
        return BKResult(
            satisfied=False, n_unstable=0, n_forward=n_forward,
            eigenvalues=[], message=f"Eigenvalue computation failed: {e}",
            forward_variables=forward_vars,
            predetermined_variables=predetermined_vars,
        )

    eigenvalues = _generalized_eigenvalues(alpha, beta)
    n_unstable = int(np.count_nonzero(_is_explosive(alpha, beta)))

    # BK condition: n_unstable == n_forward
    satisfied = (n_unstable == n_forward)
    rank_message: Optional[str] = None

    if satisfied and n_forward > 0 and n_unstable > 0:
        try:
            n_predetermined = len(predetermined_indices)
            forward_rows = list(range(
                n_predetermined,
                n_predetermined + n_forward,
            ))
            unstable_current_rows = z[forward_rows, :n_unstable]
            forward_rank = int(np.linalg.matrix_rank(
                unstable_current_rows,
            ))
            if forward_rank < n_forward:
                satisfied = False
                rank_message = (
                    "Blanchard & Kahn conditions are not satisfied: "
                    "indeterminacy due to rank failure. The order condition "
                    f"holds ({n_unstable} eigenvalue(s) larger than 1 in "
                    f"modulus for {n_forward} forward-looking variable(s)), "
                    "but the rank condition is NOT verified (unstable-root "
                    f"rank {forward_rank} is below {n_forward})."
                )
        except Exception as e:
            satisfied = False
            rank_message = f"Blanchard-Kahn rank condition check failed: {e}"

    if satisfied:
        message = (
            "Blanchard-Kahn conditions are satisfied: "
            f"{n_unstable} eigenvalue(s) larger than 1 in modulus for "
            f"{n_forward} forward-looking variable(s). "
            "The order and rank conditions are verified."
        )
    elif rank_message is not None:
        message = rank_message
    elif n_unstable < n_forward:
        message = (
            "Blanchard & Kahn conditions are not satisfied: indeterminacy. "
            f"There are {n_unstable} eigenvalue(s) larger than 1 in modulus "
            f"for {n_forward} forward-looking variable(s); too few unstable "
            "roots, so the order condition is NOT verified."
        )
    else:
        message = (
            "Blanchard & Kahn conditions are not satisfied: no stable "
            f"equilibrium. There are {n_unstable} eigenvalue(s) larger than 1 "
            f"in modulus for {n_forward} forward-looking variable(s); too "
            "many unstable roots, so the order condition is NOT verified."
        )

    return BKResult(
        satisfied=satisfied,
        n_unstable=n_unstable,
        n_forward=n_forward,
        eigenvalues=eigenvalues,
        message=message,
        forward_variables=forward_vars,
        predetermined_variables=predetermined_vars,
    )


# ---------------------------------------------------------------------------
# Diagnostic conversion
# ---------------------------------------------------------------------------

def bk_to_diagnostics(result: BKResult, model: ParsedModel) -> List[Diagnostic]:
    """Convert a BKResult into LSP Diagnostics on the model block."""
    rng = model.model_block_range
    if rng is None:
        rng = SourceRange(Position(0, 0), Position(0, 1))

    if result.satisfied:
        return [Diagnostic(
            range=rng,
            severity=Severity.INFORMATION,
            message=result.message,
            source="dynare",
            code="I070",
        )]
    elif not result.eigenvalues:
        # satisfied=False but NO eigenvalues were computed == the BK check was
        # SKIPPED, not failed: EXPECTATION operators, higher-order lead/lag,
        # singular/rank-deficient reduction, an unsolvable steady state, a
        # count mismatch, or scipy missing. These are tool limitations on a
        # possibly-valid model, so emit them as INFORMATION (I071) -- NOT the
        # W071 WARNING reserved for a genuine "conditions are not satisfied"
        # verdict -- so a skip is never conflated with a real BK violation.
        return [Diagnostic(
            range=rng,
            severity=Severity.INFORMATION,
            message=result.message,
            source="dynare",
            code="I071",
        )]
    else:
        # W070 was previously reused here, colliding with the parameter-
        # bounds warning emitted by diagnostics._check_param_bounds and
        # explain.py's W070 entry.  Use W071 for BK failures so each
        # code has a single canonical meaning and explain.py can route
        # the user to the right documentation.
        return [Diagnostic(
            range=rng,
            severity=Severity.WARNING,
            message=result.message,
            source="dynare",
            code="W071",
        )]
