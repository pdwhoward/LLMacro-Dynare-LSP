"""Model-form diagnostics: steady_state_model order, linearity, deprecation.

Mirrors checks the Dynare preprocessor performs on the *shape* of a model:

  W130  a variable used in steady_state_model before it is assigned
  W131  a variable assigned more than once in steady_state_model
  W140  a model declared ``linear`` uses a nonlinear operator on a variable
  W150  a deprecated command (``simul``, ``ramsey_policy``)
  E066-E069 / W054-W055  histval/perfect-foresight initialization checks

All findings are warnings.  W150 carries the LSP ``Deprecated`` tag so editors
render the deprecated token struck through.  Each check is gated: a model that
uses none of the relevant constructs gets nothing.
"""

from __future__ import annotations

import ast
import re
from dataclasses import replace
from typing import List, Optional, Set, Tuple

from .diagnostics import Diagnostic, Severity
from .parser import (
    ParsedModel,
    Position,
    SourceRange,
    _find_all_blocks,
    _offset_to_position,
    _strip_comments,
)
from .steady_state import _LOCAL_VAR_DEF, _prepare_ss_expression

_FALLBACK_RANGE = SourceRange(Position(0, 0), Position(0, 1))

# LSP DiagnosticTag.Deprecated (kept numeric; the server maps it for the wire).
_TAG_DEPRECATED = 2

# Nonlinear unary/binary operators Dynare rejects in a ``linear`` model
# (``isUnaryOpUsedOnType`` / ``isBinaryOpUsedOnType``).
_NONLINEAR_FUNC_RE = re.compile(r"(?<![\w.])(max|min|abs|sign)\s*\(", re.IGNORECASE)

# Deprecated computation commands -> their modern replacements.
_DEPRECATED_SIMUL_RE = re.compile(r"(?<!\w)simul(?!\w)\s*[(;]")

# Deprecated command *options*, detected as option keywords (the bundled
# preprocessor emits a deprecation warning and still accepts the model).
_DEPRECATED_OPTIONS = {
    "aim_solver": "The 'aim_solver' option is deprecated; use 'dr = aim' instead.",
    "bytecode": (
        "The 'bytecode' option is deprecated and will be removed in a future "
        "Dynare release."
    ),
}
_DEPRECATED_OPTION_RE = re.compile(
    r"(?<!\w)(" + "|".join(_DEPRECATED_OPTIONS) + r")(?!\w)",
)


def _rng(rng: Optional[SourceRange]) -> SourceRange:
    return rng if rng is not None else _FALLBACK_RANGE


def check_model_form(
    model: ParsedModel,
    endogenous: Set[str],
    exogenous: Set[str],
    deterministic_exogenous: Set[str],
    parameters: Set[str],
    *,
    include_models: Optional[List[ParsedModel]] = None,
) -> List[Diagnostic]:
    """Run the model-form checks (W130 / W131 / W140 / W150)."""
    diagnostics: List[Diagnostic] = []
    diagnostics.extend(
        _check_steady_state_model(
            model,
            endogenous,
            parameters,
            exogenous,
            deterministic_exogenous,
        )
    )
    diagnostics.extend(
        _check_linear_model(
            model,
            endogenous | exogenous | deterministic_exogenous,
        )
    )
    diagnostics.extend(_check_deprecations(model))
    diagnostics.extend(_check_histval(model))
    for include_model in include_models or []:
        diagnostics.extend(
            _check_deprecations(
                include_model,
                include_model.include_anchor_range,
                include_policy_commands=False,
            )
        )
        include_histval = _check_histval(include_model)
        if include_model.include_anchor_range is not None:
            include_histval = [
                replace(item, range=include_model.include_anchor_range, fix=None)
                for item in include_histval
            ]
        diagnostics.extend(include_histval)
    return diagnostics


# ---------------------------------------------------------------------------
# W130 / W131 -- steady_state_model block ordering
# ---------------------------------------------------------------------------


def _rhs_identifiers(rhs: str) -> List[str]:
    """Identifiers in *rhs* that are not function calls (not followed by ``(``)."""
    names: List[str] = []
    for match in re.finditer(r"(?<![\w.])([A-Za-z_]\w*)\s*(\(?)", rhs):
        if match.group(2) == "(":
            continue  # function application, not a variable reference
        names.append(match.group(1))
    return names


def _check_steady_state_model(
    model: ParsedModel,
    endogenous: Set[str],
    parameters: Set[str],
    exogenous: Set[str],
    deterministic_exogenous: Set[str],
) -> List[Diagnostic]:
    """W130 (use-before-assign) and W131 (double-assign) in steady_state_model."""
    equations = model.steady_state_equations
    if not equations:
        return []

    diagnostics: List[Diagnostic] = []
    assigned_anywhere = {eq.lhs.strip() for eq in equations}
    # Parameters and exogenous (= 0 at the steady state) are known up front.
    assigned_so_far: Set[str] = (
        set(parameters) | set(exogenous) | set(deterministic_exogenous)
    )
    assigned_once: Set[str] = set()
    flagged_use_before: Set[str] = set()

    for equation in equations:
        lhs = equation.lhs.strip()

        for name in _rhs_identifiers(equation.rhs):
            # Only an endogenous that *is* assigned later in the block (so it is
            # not the separate "missing from steady_state_model" case, W042) and
            # is referenced before that assignment is an ordering error.
            if (
                name in endogenous
                and name not in assigned_so_far
                and name in assigned_anywhere
                and name not in flagged_use_before
            ):
                flagged_use_before.add(name)
                diagnostics.append(
                    Diagnostic(
                        range=_rng(equation.range),
                        severity=Severity.WARNING,
                        message=(
                            f"'{name}' is used in the steady_state_model block "
                            "before it is assigned. The block is evaluated top to "
                            "bottom, so each variable must be assigned before use."
                        ),
                        source="dynare",
                        code="W130",
                    )
                )

        if lhs in assigned_once and lhs in endogenous:
            # An in-place transformation that reuses the prior value (the
            # ubiquitous ``A = log(A)`` / ``tb = exp(tb)`` log-model idiom, or a
            # lag form like ``A = A(-1) + ...``) is intentional; only a silent
            # override that ignores the earlier value is flagged.  Use a plain
            # word search so a self-reference written as ``A(-1)`` counts too.
            reuses_self = (
                re.search(r"(?<![\w.])" + re.escape(lhs) + r"\b", equation.rhs)
                is not None
            )
            if not reuses_self:
                diagnostics.append(
                    Diagnostic(
                        range=_rng(equation.range),
                        severity=Severity.WARNING,
                        message=(
                            f"'{lhs}' is assigned more than once in the "
                            "steady_state_model block; the later assignment "
                            "silently overrides the earlier one."
                        ),
                        source="dynare",
                        code="W131",
                    )
                )
        assigned_once.add(lhs)
        assigned_so_far.add(lhs)

    return diagnostics


# ---------------------------------------------------------------------------
# E066-E069 / W054-W055 -- histval and deterministic initialization
# ---------------------------------------------------------------------------

_HIST_ASSIGN_RE = re.compile(
    r"(?m)^\s*([A-Za-z_]\w*)\s*(?:\(\s*([+-]?\d+)\s*\))?\s*=\s*([^;]+);"
)


def _check_histval(model: ParsedModel) -> List[Diagnostic]:
    stripped = _strip_comments(model.text)
    hist_blocks = _find_all_blocks(stripped, "histval")
    if not hist_blocks:
        return []

    diagnostics: List[Diagnostic] = []
    endval_blocks = _find_all_blocks(stripped, "endval")
    first_hist = hist_blocks[0]
    hist_anchor = SourceRange(
        _offset_to_position(model.text, first_hist.start()),
        _offset_to_position(model.text, first_hist.start() + len("histval")),
    )

    if endval_blocks:
        diagnostics.append(Diagnostic(
            range=hist_anchor,
            severity=Severity.ERROR,
            message=(
                "A model cannot use both histval and endval for the same "
                "deterministic simulation setup. Choose one initialization "
                "scheme before running Dynare."
            ),
            source="dynare",
            code="E066",
        ))

    after_hist = stripped[first_hist.end():]
    steady_match = re.search(r"(?<!\w)steady(?!\w)\s*(?:\([^;]*\))?\s*;", after_hist, re.I)
    if steady_match is not None:
        start = first_hist.end() + steady_match.start()
        diagnostics.append(Diagnostic(
            range=SourceRange(
                _offset_to_position(model.text, start),
                _offset_to_position(model.text, start + len("steady")),
            ),
            severity=Severity.ERROR,
            message=(
                "A steady command after histval is incompatible with histval "
                "initialization because steady would overwrite the historical "
                "conditions used by the deterministic simulation."
            ),
            source="dynare",
            code="E067",
        ))

    declared = model.endogenous_names() | model.exogenous_names() | model.deterministic_exogenous_names()
    state_offsets: dict[str, set[int]] = {}
    try:
        from .bk_check import _extract_variable_timing
        state_offsets = _extract_variable_timing(model)
    except Exception:
        pass
    state_names = {
        name for name, offsets in state_offsets.items() if any(offset < 0 for offset in offsets)
    }

    supplied: dict[str, set[int]] = {}
    for block in hist_blocks:
        body = block.group(2)
        body_start = block.start(2)
        for match in _HIST_ASSIGN_RE.finditer(body):
            name = match.group(1)
            offset = int(match.group(2) or "0")
            start = body_start + match.start(1)
            rng = SourceRange(
                _offset_to_position(model.text, start),
                _offset_to_position(model.text, start + len(name)),
            )
            if name not in declared:
                diagnostics.append(Diagnostic(
                    range=rng,
                    severity=Severity.ERROR,
                    message=f"histval assigns undeclared symbol '{name}'.",
                    source="dynare",
                    code="E068",
                ))
                continue
            if offset > 0:
                diagnostics.append(Diagnostic(
                    range=rng,
                    severity=Severity.ERROR,
                    message=(
                        f"histval entry '{name}({offset:+d})' is in a future "
                        "period. Historical initialization accepts period 0 "
                        "and negative lags, not positive leads."
                    ),
                    source="dynare",
                    code="E069",
                ))
            supplied.setdefault(name, set()).add(offset)
            if name in model.endogenous_names() and name not in state_names:
                diagnostics.append(Diagnostic(
                    range=rng,
                    severity=Severity.WARNING,
                    message=(
                        f"histval specifies endogenous variable '{name}', but "
                        "the dynamic model does not classify it as a state "
                        "variable with a lag. Dynare uses histval to supply "
                        "historical state values."
                    ),
                    source="dynare",
                    code="W054",
                ))

    all_values_required = bool(re.search(
        r"(?<!\w)histval\s*\(\s*all_values_required\s*\)", stripped, re.I
    ))
    for name in sorted(state_names):
        required_lags = sorted(
            offset for offset in state_offsets.get(name, set()) if offset < 0
        )
        missing = [offset for offset in required_lags if offset not in supplied.get(name, set())]
        if not missing:
            continue
        rendered = ", ".join(f"{name}({offset})" for offset in missing)
        severity = Severity.ERROR if all_values_required else Severity.WARNING
        diagnostics.append(Diagnostic(
            range=hist_anchor,
            severity=severity,
            message=(
                f"histval does not provide required historical state value(s): "
                f"{rendered}. " + (
                    "all_values_required makes missing values an error."
                    if all_values_required
                    else "Dynare will initialize missing historical values to zero."
                )
            ),
            source="dynare",
            code="W055",
        ))
    return diagnostics


# ---------------------------------------------------------------------------
# W140 -- nonlinear operator in a model declared ``linear``
# ---------------------------------------------------------------------------


def _balanced_arg(text: str, open_idx: int) -> Optional[str]:
    """Return the substring inside the parentheses opened at *open_idx*."""
    close_idx = _balanced_close(text, open_idx)
    if close_idx is None:
        return None
    return text[open_idx + 1 : close_idx]


def _balanced_close(text: str, open_idx: int) -> Optional[int]:
    """Return the matching close-parenthesis index for *open_idx*."""
    depth = 0
    for i in range(open_idx, len(text)):
        char = text[i]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


_STEADY_STATE_CALL_RE = re.compile(r"\bsteady_state\s*\(", re.IGNORECASE)


def _mask_steady_state_calls(expr: str) -> str:
    """Replace steady-state operators by opaque constants for linearity checks."""
    pieces: List[str] = []
    cursor = 0
    while True:
        match = _STEADY_STATE_CALL_RE.search(expr, cursor)
        if match is None:
            break
        close_idx = _balanced_close(expr, match.end() - 1)
        if close_idx is None:
            break
        pieces.append(expr[cursor : match.start()])
        pieces.append("None")
        cursor = close_idx + 1
    pieces.append(expr[cursor:])
    return "".join(pieces)


def _check_linear_model(
    model: ParsedModel,
    variables: Set[str],
) -> List[Diagnostic]:
    """W140 -- a ``linear`` model uses a nonlinear operator on a variable."""
    if not model.is_linear or not variables:
        return []

    diagnostics: List[Diagnostic] = []
    seen: Set[Tuple[int, str]] = set()
    variable_aliases = set(variables)
    for equation in model.model_equations:
        local_match = _LOCAL_VAR_DEF.match(equation.text.strip())
        if local_match:
            exprs = [local_match.group(2)]
        elif equation.rhs:
            exprs = [equation.lhs, equation.rhs]
        else:
            exprs = [equation.text]

        for expr in exprs:
            operator = _linear_nonlinear_operator(expr, variable_aliases)
            if operator is None:
                continue
            key = (equation.range.start.line, operator)
            if key in seen:
                continue
            seen.add(key)
            diagnostics.append(
                Diagnostic(
                    range=_rng(equation.range),
                    severity=Severity.WARNING,
                    message=(
                        f"Model is declared 'linear' but applies the nonlinear "
                        f"operator '{operator}' to a variable. Dynare requires the "
                        "equations of a 'linear' model to be linear in the "
                        "variables; drop 'linear' or linearise the equation."
                    ),
                    source="dynare",
                    code="W140",
                )
            )
            break
        if local_match and _expression_has_variable(
            local_match.group(2),
            variable_aliases,
        ):
            variable_aliases.add(local_match.group(1))
    return diagnostics


def _linear_nonlinear_operator(expr: str, variables: Set[str]) -> Optional[str]:
    """Return the first nonlinear operator applied to model variables."""
    analysis_expr = _mask_steady_state_calls(expr)
    for match in _NONLINEAR_FUNC_RE.finditer(analysis_expr):
        arg = _balanced_arg(analysis_expr, match.end() - 1)
        if arg is None:
            continue
        if not any(tok in variables for tok in re.findall(r"[A-Za-z_]\w*", arg)):
            continue
        return match.group(1).lower()

    prepared = _prepare_ss_expression(analysis_expr.rstrip(";").strip())
    try:
        tree = ast.parse(prepared, mode="eval")
    except SyntaxError:
        return None
    return _nonlinear_operator_in_node(tree.body, variables)


def _expression_has_variable(expr: str, variables: Set[str]) -> bool:
    prepared = _prepare_ss_expression(
        _mask_steady_state_calls(expr).rstrip(";").strip()
    )
    try:
        tree = ast.parse(prepared, mode="eval")
    except SyntaxError:
        return False
    return _node_has_variable(tree.body, variables)


def _nonlinear_operator_in_node(
    node: ast.AST,
    variables: Set[str],
) -> Optional[str]:
    if isinstance(node, ast.Call):
        if any(_node_has_variable(arg, variables) for arg in node.args):
            if isinstance(node.func, ast.Name):
                return node.func.id
            return "function"
    if isinstance(node, ast.Compare):
        if _node_has_variable(node, variables):
            return "comparison"
    if isinstance(node, ast.BinOp):
        left_has = _node_has_variable(node.left, variables)
        right_has = _node_has_variable(node.right, variables)
        if isinstance(node.op, ast.Mult) and left_has and right_has:
            return "*"
        if isinstance(node.op, ast.Div) and right_has:
            return "/"
        if isinstance(node.op, ast.Pow) and (left_has or right_has):
            if not (
                left_has and not right_has and _is_linear_safe_exponent(node.right)
            ):
                return "^"

    for child in ast.iter_child_nodes(node):
        operator = _nonlinear_operator_in_node(child, variables)
        if operator is not None:
            return operator
    return None


def _node_has_variable(node: ast.AST, variables: Set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in variables
    if isinstance(node, ast.Call):
        return any(_node_has_variable(arg, variables) for arg in node.args)
    return any(
        _node_has_variable(child, variables) for child in ast.iter_child_nodes(node)
    )


def _is_linear_safe_exponent(node: ast.AST) -> bool:
    """``x^1`` is linear and ``x^0`` is constant — neither is nonlinear."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value) in (0.0, 1.0)
    return False


# ---------------------------------------------------------------------------
# W150 -- deprecated commands
# ---------------------------------------------------------------------------


def _check_deprecations(
    model: ParsedModel,
    range_override: Optional[SourceRange] = None,
    *,
    include_policy_commands: bool = True,
) -> List[Diagnostic]:
    """W150 -- deprecated computation commands (``simul``, ``ramsey_policy``)."""
    diagnostics: List[Diagnostic] = []
    stripped = re.sub(
        r"'[^'\n]*'|\"[^\"\n]*\"",
        lambda m: " " * (m.end() - m.start()),
        _strip_comments(model.text),
    )

    for match in _DEPRECATED_SIMUL_RE.finditer(stripped):
        rng = range_override or SourceRange(
            _offset_to_position(stripped, match.start()),
            _offset_to_position(stripped, match.start() + len("simul")),
        )
        diagnostics.append(
            Diagnostic(
                range=rng,
                severity=Severity.WARNING,
                message=(
                    "'simul' is deprecated. Use 'perfect_foresight_setup' followed "
                    "by 'perfect_foresight_solver'."
                ),
                source="dynare",
                code="W150",
                tags=[_TAG_DEPRECATED],
            )
        )

    if include_policy_commands and "ramsey_policy" in model.policy_commands:
        diagnostics.append(
            Diagnostic(
                range=_rng(model.policy_command_range),
                severity=Severity.WARNING,
                message=(
                    "'ramsey_policy' is deprecated. Use 'ramsey_model' followed by "
                    "'stoch_simul'."
                ),
                source="dynare",
                code="W150",
                tags=[_TAG_DEPRECATED],
            )
        )

    declared = set(model.all_declared_names())
    for equation in model.model_equations:
        local = re.match(r"#\s*([A-Za-z][A-Za-z0-9_]*)\s*=", equation.text.strip())
        if local:
            declared.add(local.group(1))
    for match in _DEPRECATED_OPTION_RE.finditer(stripped):
        name = match.group(1)
        if name in declared:
            continue
        diagnostics.append(
            Diagnostic(
                range=range_override
                or SourceRange(
                    _offset_to_position(stripped, match.start()),
                    _offset_to_position(stripped, match.end()),
                ),
                severity=Severity.WARNING,
                message=_DEPRECATED_OPTIONS[name],
                source="dynare",
                code="W150",
                tags=[_TAG_DEPRECATED],
            )
        )

    return diagnostics
