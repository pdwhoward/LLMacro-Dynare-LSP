"""Steady state equation validator for Dynare models.

Validates that the steady state values (from ``steady_state_model`` or
``initval`` blocks) actually satisfy the model equations.

At steady state:
  - All leads and lags collapse: y(+1) = y(-1) = y  (the steady state value)
  - All exogenous shocks are zero
  - The ``steady_state(x)`` operator returns the steady state value of x

The validator computes all steady state values, then plugs them into every
model equation and checks that LHS = RHS (residual ≈ 0).
"""

from __future__ import annotations

import ast
import logging
import math
import re
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Dict, List, Optional, Tuple

from .parser import (
    Equation,
    ParsedModel,
    SourceRange,
    _mask_string_literals,
    _strip_comments,
    _strip_equation_tags_from_text,
)

logger = logging.getLogger(__name__)


@dataclass
class SteadyStateResult:
    """Result of evaluating one equation at steady state."""

    equation: Equation
    residual: Optional[float]
    error_message: str = ""
    is_satisfied: bool = False
    is_local_var: bool = False  # True for #name = expr definitions
    missing_vars: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SteadyStateAssignmentEvaluation:
    """Sequential evaluation result for one steady_state_model assignment."""

    name: str
    expression: str
    value: Optional[float]
    range: SourceRange
    error_message: str = ""
    is_local_var: bool = False


@dataclass
class SteadyStateReport:
    """Complete report of steady state validation."""

    values: Dict[str, float]
    results: List[SteadyStateResult]
    value_errors: List[Tuple[str, str, SourceRange]]  # (var_name, error_msg, range)

    @property
    def all_satisfied(self) -> bool:
        return (
            not self.value_errors
            and not self.missing_endogenous
            and all(r.is_satisfied for r in self.results)
        )

    @property
    def n_satisfied(self) -> int:
        return sum(1 for r in self.results if r.is_satisfied)

    @property
    def n_failed(self) -> int:
        return sum(
            1 for r in self.results if not r.is_satisfied and r.residual is not None
        )

    @property
    def n_unevaluable(self) -> int:
        return sum(
            1
            for r in self.results
            if not r.is_satisfied and r.residual is None and not r.is_local_var
        )

    @property
    def missing_endogenous(self) -> List[str]:
        """Endogenous variables that have no steady state value."""
        return self._missing_endo

    _missing_endo: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary of steady state consistency."""
        lines = []
        total = len(self.results)
        lines.append(
            f"Steady state check: {self.n_satisfied}/{total} equations satisfied"
        )

        if self.n_failed > 0:
            lines.append(f"  {self.n_failed} equation(s) have non-zero residuals:")
            for r in self.results:
                if not r.is_satisfied and r.residual is not None:
                    label = (
                        f"'{r.equation.name}'"
                        if r.equation.name
                        else r.equation.text[:50]
                    )
                    lines.append(f"    - {label}: residual = {r.residual:.2e}")

        if self.n_unevaluable > 0:
            lines.append(f"  {self.n_unevaluable} equation(s) could not be evaluated")

        if self._missing_endo:
            lines.append(f"  Missing steady state for: {', '.join(self._missing_endo)}")

        if self.value_errors:
            lines.append(
                f"  {len(self.value_errors)} error(s) computing steady state values"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Expression preparation
# ---------------------------------------------------------------------------

# Match time-subscripted variables: name(+1), name(-1), name(+2), etc.
_TIME_SUBSCRIPT = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*([+-]?\s*\d+)\s*\)")

_TIME_SUBSCRIPT_FUNCTION_NAMES = frozenset(
    {
        "exp",
        "log",
        "ln",
        "log2",
        "log10",
        "sqrt",
        "cbrt",
        "abs",
        "sign",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "floor",
        "ceil",
        "round",
        "min",
        "max",
        "erf",
        "erfc",
        "normpdf",
        "normcdf",
        "norminv",
        "logncdf",
    }
)

# Match steady_state(name) or STEADY_STATE(name) operator for callers
# that only need to mask the simple identifier form.
_STEADY_STATE_OP = re.compile(
    r"\bsteady_state\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
    re.IGNORECASE,
)
_STEADY_STATE_OPENER = re.compile(r"\bsteady_state\s*\(", re.IGNORECASE)

# Match model-local variable definitions: #name = expr
_LOCAL_VAR_DEF = re.compile(r"^\s*#\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$")

_BUILTIN_FUNCTION_CALL_NAMES = _TIME_SUBSCRIPT_FUNCTION_NAMES | {
    "steady_state",
    "expectation",
    "pac_expectation",
    "diff",
}


def _strip_equation_tags(eq_text: str) -> str:
    """Remove equation tags like [name='Euler equation'] from text."""
    return _strip_equation_tags_from_text(eq_text)


def _prepare_ss_expression(eq_text: str) -> str:
    """Transform a model equation into its steady state form.

    - Strip equation tags: [name='...'] -> removed
    - Collapse Dynare's ``EXPECTATION(h)(expr)`` operator to ``expr``
      (at steady state every variable equals its lead/lag, so the
      conditional expectation is just the inner expression itself)
    - Expand Dynare's ``diff(expr)`` operator as ``expr - expr``
    - Remove time subscripts: y(+1), y(-1), y(+2), etc. -> y
    - Replace steady_state(x) -> x
    - Convert ^ to ** for Python evaluation
    """
    result = eq_text

    # Strip equation tags
    result = _strip_equation_tags(result)

    # Collapse EXPECTATION(h)(expr) -> expr.  Dynare lets the inner
    # ``expr`` contain arbitrary content including nested parens, so
    # do this with a manual scanner rather than a regex.  Run BEFORE
    # the time-subscript stripper so the EXPECTATION's own ``(h)``
    # parameter doesn't get mis-parsed as a time subscript.
    result = _collapse_expectation_operator(result)

    # ``diff(expr)`` is ``expr - expr(-1)``; at steady state the two
    # operands have the same value.  Keep both operands instead of
    # replacing the call by literal zero so invalid inner expressions
    # such as ``diff(1/y)`` still surface their domain errors.
    result = _collapse_diff_operator(result)

    # Replace steady_state(expr) with expr.
    result = _collapse_steady_state_operator(result)

    result = _canonicalize_builtin_function_calls(result)

    # Remove time subscripts.  Do not rewrite known function calls with
    # integer literal arguments (e.g. exp(0)); those are valid functions,
    # not Dynare leads/lags.
    def _time_repl(m: re.Match) -> str:
        name = m.group(1)
        if name.lower() in _TIME_SUBSCRIPT_FUNCTION_NAMES:
            return m.group(0)
        return name

    result = _TIME_SUBSCRIPT.sub(_time_repl, result)

    # Convert ^ to ** for Python
    result = result.replace("^", "**")

    return result


def _canonicalize_builtin_function_calls(text: str) -> str:
    """Lowercase known Dynare built-in function calls before Python eval."""

    def _replace(match: re.Match) -> str:
        name = match.group(1)
        lowered = name.lower()
        return lowered if lowered in _BUILTIN_FUNCTION_CALL_NAMES else name

    return re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?=\()", _replace, text)


_EXPECTATION_OPENER = re.compile(
    r"\b(?:expectation|pac_expectation)\s*\(",
    re.IGNORECASE,
)

_DIFF_OPENER = re.compile(r"\bdiff\s*\(", re.IGNORECASE)


def _collapse_expectation_operator(text: str) -> str:
    """Replace ``EXPECTATION(h)(expr)`` (or ``pac_expectation``) with ``(expr)``.

    At steady state every variable equals its own lead/lag, so the
    conditional expectation operator becomes a no-op.  Without this
    collapse the steady-state evaluator sees the operator as an
    undefined function and the equation evaluation fails silently,
    masking real diagnostics about the model.

    Match is case-insensitive (Dynare's preprocessor accepts any
    capitalisation), and covers both ``expectation`` and the PAC-style
    ``pac_expectation`` variant.
    """
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        m = _EXPECTATION_OPENER.search(text, i)
        if m is None:
            out.append(text[i:])
            break
        idx = m.start()
        out.append(text[i:idx])
        # Skip the EXPECTATION(...) parameter group.
        j = m.end()  # position just past the opening "("
        depth = 1
        while j < n and depth > 0:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        # Now j points past the closing ')' of the parameter group.
        # If an inner-expression paren immediately follows, strip it
        # too: ``EXPECTATION(h)(expr)`` -> ``(expr)`` (preserving the
        # parentheses keeps operator precedence intact for callers).
        if j < n and text[j] == "(":
            inner_start = j  # keep the leading '(' in the output
            depth = 1
            k = j + 1
            while k < n and depth > 0:
                if text[k] == "(":
                    depth += 1
                elif text[k] == ")":
                    depth -= 1
                k += 1
            out.append(text[inner_start:k])
            i = k
        else:
            # No inner expression — odd but leave the result blank.
            i = j
    return "".join(out)


def _collapse_diff_operator(text: str) -> str:
    """Replace Dynare ``diff(expr)`` calls with ``(expr) - (expr)``."""
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        m = _DIFF_OPENER.search(text, i)
        if m is None:
            out.append(text[i:])
            break
        out.append(text[i : m.start()])
        j = m.end()
        depth = 1
        while j < n and depth > 0:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        inner = _collapse_diff_operator(text[m.end() : j - 1])
        out.append(f"(({inner}) - ({inner}))")
        i = j
    return "".join(out)


def _collapse_steady_state_operator(text: str) -> str:
    """Replace ``steady_state(expr)`` with ``(expr)`` for SS evaluation."""
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        m = _STEADY_STATE_OPENER.search(text, i)
        if m is None:
            out.append(text[i:])
            break
        out.append(text[i : m.start()])
        j = m.end()
        depth = 1
        while j < n and depth > 0:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        if depth == 0:
            inner = text[m.end() : j - 1]
            out.append(f"({inner})")
        i = j
    return "".join(out)


_PYTHON_RESERVED = frozenset(
    {
        "False",
        "None",
        "True",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
    }
)


def _escape_reserved(name: str) -> str:
    """Prefix Python reserved words so they can be used as identifiers."""
    if name in _PYTHON_RESERVED:
        return f"_dyn_{name}"
    return name


def _escape_expr(expr: str) -> str:
    """Escape Python reserved words in an expression string."""

    def _replace(m: re.Match) -> str:
        word = m.group(0)
        return _escape_reserved(word)

    return re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", _replace, expr)


def _logncdf(x, mu=0, sigma=1):
    try:
        x = float(x)
        mu = float(mu)
        sigma = float(sigma)
    except (TypeError, ValueError):
        return math.nan
    if sigma <= 0:
        return math.nan
    if x <= 0:
        return 0.0
    return 0.5 * (1 + math.erf((math.log(x) - mu) / (sigma * math.sqrt(2))))


def _norminv(p, mu=0, sigma=1):
    try:
        p = float(p)
        mu = float(mu)
        sigma = float(sigma)
    except (TypeError, ValueError):
        return math.nan
    if sigma <= 0 or not 0 < p < 1:
        return math.nan
    return NormalDist(mu, sigma).inv_cdf(p)


def _cbrt(x):
    x = float(x)
    return math.copysign(abs(x) ** (1.0 / 3.0), x)


def _build_eval_env(values: Dict[str, float]) -> dict:
    """Build the evaluation environment with math functions and variable values.

    Python reserved words used as Dynare identifiers are escaped with a
    ``_dyn_`` prefix so that ``eval()`` doesn't raise a SyntaxError.
    """
    env = {
        "exp": math.exp,
        "log": math.log,
        "ln": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "sqrt": math.sqrt,
        "cbrt": getattr(math, "cbrt", _cbrt),
        "abs": abs,
        "sign": lambda x: float((x > 0) - (x < 0)),
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "asinh": math.asinh,
        "acosh": math.acosh,
        "atanh": math.atanh,
        "floor": math.floor,
        "ceil": math.ceil,
        "round": round,
        "min": min,
        "max": max,
        "erf": math.erf,
        "erfc": math.erfc,
        "pi": math.pi,
        "inf": math.inf,
        "normpdf": lambda x, mu=0, sigma=1: (
            math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
        ),
        "normcdf": lambda x, mu=0, sigma=1: (
            0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))
        ),
        "norminv": _norminv,
        "logncdf": _logncdf,
    }
    env.update(
        {
            name.upper(): value
            for name, value in list(env.items())
            if name.upper() != name
        }
    )
    # Variable values override math constants (e.g. a model variable named 'pi')
    env.update({_escape_reserved(k): v for k, v in values.items()})
    return env


def _find_undefined_vars(expr: str, env: dict) -> List[str]:
    """Find variable identifiers in expr that are not defined in env."""
    ids = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", expr))
    undefined = []
    for name in ids:
        escaped = _escape_reserved(name)
        if escaped not in env and name not in env:
            undefined.append(name)
    return sorted(undefined)


def _validate_ast(expr_str: str) -> bool:
    """Validate that an expression contains only safe AST constructs.

    Rejects attribute access (blocks ``().__class__.__bases__`` etc.),
    subscript operations (blocks ``__subclasses__()[0]``), and calls to
    anything other than simple named functions (blocks chained method calls).

    Returns True if the expression is safe, False otherwise.
    """
    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.NamedExpr):
            return False
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                node.value,
                (int, float, complex),
            ):
                return False
        if isinstance(node, ast.Attribute):
            return False
        if isinstance(node, ast.Subscript):
            return False
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return False
    return True


def _safe_eval_expr(expr: str, env: dict) -> Tuple[Optional[float], str]:
    """Evaluate a Python expression, returning (value, error_message).

    Validates the expression AST before evaluation to reject dangerous
    constructs (attribute access, subscripts, non-simple calls).

    Handles complex number results by taking the real part when the
    imaginary component is negligible.
    """
    if not _validate_ast(expr):
        logger.debug("Rejected unsafe expression: %s", expr)
        return None, f"Unsafe expression rejected: {expr}"
    try:
        result = eval(expr, {"__builtins__": {}}, env)
        # Handle complex results from fractional exponentiation
        if isinstance(result, complex):
            if abs(result.imag) < 1e-10:
                return float(result.real), ""
            return None, f"Expression produced complex result: {result}"
        return float(result), ""
    except NameError as e:
        return None, f"Undefined variable: {e}"
    except ZeroDivisionError:
        return None, "Division by zero"
    except (ValueError, OverflowError) as e:
        return None, f"Math error: {e}"
    except SyntaxError as e:
        return None, f"Syntax error: {e}"
    except Exception as e:
        return None, f"Evaluation error: {e}"


# ---------------------------------------------------------------------------
# Steady state value computation
# ---------------------------------------------------------------------------


def _compute_ss_values_from_block(
    model: ParsedModel,
    param_overrides: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], List[Tuple[str, str, SourceRange]]]:
    """Compute steady state values from the steady_state_model block.

    Evaluates assignments top-to-bottom, building up the namespace
    sequentially (since later assignments may depend on earlier ones).
    """
    values, evaluations = evaluate_steady_state_model_assignments(
        model,
        param_overrides=param_overrides,
    )
    errors = [
        (item.name, item.error_message, item.range)
        for item in evaluations
        if item.error_message
    ]
    return values, errors


def evaluate_steady_state_model_assignments(
    model: ParsedModel,
    param_overrides: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], List[SteadyStateAssignmentEvaluation]]:
    """Evaluate ``steady_state_model`` assignments with Dynare steady-state semantics."""
    values: Dict[str, float] = {}
    evaluations: List[SteadyStateAssignmentEvaluation] = []

    # Start with parameter and top-level helper values.
    values.update(model.param_values())
    declared_params = model.parameter_names()
    if param_overrides:
        for name, value in param_overrides.items():
            if name in declared_params:
                values[name] = float(value)
    for assignment in model.helper_assignments:
        if assignment.value is not None:
            values[assignment.name] = assignment.value
        else:
            values.pop(assignment.name, None)
    values.update(_exogenous_values(model))
    declared_names = model.all_declared_names()

    # Build the eval env once before the loop rather than once per equation.
    # When a new variable is evaluated successfully, we add it to the shared
    # env dict directly (O(1)) instead of rebuilding the entire environment
    # from scratch (O(n)) on every iteration.  On a 291-equation model this
    # turns O(n^2) env-construction into O(n).
    env = _build_eval_env(values)

    for eq in model.steady_state_equations:
        text = eq.text.strip()
        if not text:
            continue

        # Skip model-local variable definitions (shouldn't appear here but be safe)
        if text.startswith("#"):
            local_match = _LOCAL_VAR_DEF.match(text)
            if local_match:
                name = local_match.group(1)
                if name in declared_names:
                    evaluations.append(
                        SteadyStateAssignmentEvaluation(
                            name=name,
                            expression=local_match.group(2).strip(),
                            value=None,
                            range=eq.range,
                            error_message=(
                                "Model-local variable shadows a declared Dynare symbol"
                            ),
                            is_local_var=True,
                        )
                    )
                    continue
                expr = local_match.group(2).strip()
                expr_py = _escape_expr(_prepare_ss_expression(expr))
                val, err = _safe_eval_expr(expr_py, env)
                if val is not None:
                    values[name] = val
                    env[_escape_reserved(name)] = val
                else:
                    err = f"Cannot evaluate: {err}"
                evaluations.append(
                    SteadyStateAssignmentEvaluation(
                        name=name,
                        expression=expr,
                        value=val,
                        range=eq.range,
                        error_message="" if val is not None else err,
                        is_local_var=True,
                    )
                )
            continue

        # Strip equation tags
        text = _strip_equation_tags(text)
        if not text:
            continue

        # Parse as assignment: name = expr
        if "=" not in text:
            continue

        parts = text.split("=", 1)
        name = parts[0].strip()
        expr = parts[1].strip()

        # Remove trailing semicolons
        if expr.endswith(";"):
            expr = expr[:-1].strip()

        # Validate name is an identifier
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            continue

        # Convert to Python expression and escape reserved words
        expr_py = _escape_expr(_prepare_ss_expression(expr))

        val, err = _safe_eval_expr(expr_py, env)
        if val is not None:
            values[name] = val
            env[_escape_reserved(name)] = val
        else:
            err = f"Cannot evaluate expression '{expr}': {err}"
        evaluations.append(
            SteadyStateAssignmentEvaluation(
                name=name,
                expression=expr,
                value=val,
                range=eq.range,
                error_message="" if val is not None else err,
            )
        )

    return values, evaluations


def _compute_ss_values_from_initval(
    model: ParsedModel,
    param_overrides: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], List[Tuple[str, str, SourceRange]]]:
    """Get steady state values from the initval block.

    Skips ``initval`` entries that target a declared parameter.  Dynare's
    ``initval`` / ``endval`` blocks are documented as setting endogenous
    and exogenous values only; a parameter-name entry would otherwise
    override the calibration during steady-state evaluation and produce
    spurious W041 residuals when the model equations reference the
    real parameter value.
    """
    values: Dict[str, float] = {}
    errors: List[Tuple[str, str, SourceRange]] = []

    # Start with parameter values and default exogenous steady-state values,
    # so initval expressions such as ``y = eps`` can use Dynare's eps=0
    # default before any explicit exogenous entry overrides it.
    values.update(model.param_values())
    declared_params = model.parameter_names()
    if param_overrides:
        for name, value in param_overrides.items():
            if name in declared_params:
                values[name] = float(value)
    for assignment in model.helper_assignments:
        if assignment.value is not None:
            values[assignment.name] = assignment.value
    values.update({name: 0.0 for name in model.exogenous_names()})
    for entry in model.initval_entries:
        if entry.name in declared_params:
            # Don't overwrite calibrated parameter values from initval —
            # initval is for endo/exo only.  We don't emit a warning here
            # because there's already a separate parsing-side check for
            # this, and a noisy duplicate would distract during repair.
            continue
        if entry.value is not None:
            values[entry.name] = entry.value
        else:
            # Try evaluating with current known values
            expr_py = _escape_expr(_prepare_ss_expression(entry.expression))
            env = _build_eval_env(values)
            val, err = _safe_eval_expr(expr_py, env)
            if val is not None:
                values[entry.name] = val
            else:
                values.pop(entry.name, None)
                errors.append(
                    (
                        entry.name,
                        f"Cannot evaluate initval expression '{entry.expression}': {err}",
                        entry.range,
                    )
                )

    return values, errors


def _exogenous_values(
    model: ParsedModel,
    param_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Return steady-state exogenous values from initval/endval, defaulting to 0."""
    exo_names = set(model.exogenous_names())
    values: Dict[str, float] = {name: 0.0 for name in exo_names}
    entries = [*model.initval_entries, *model.endval_entries]
    eval_values: Dict[str, float] = {}
    eval_values.update(model.param_values())
    declared_params = model.parameter_names()
    if param_overrides:
        for name, value in param_overrides.items():
            if name in declared_params:
                eval_values[name] = float(value)
    for assignment in model.helper_assignments:
        if assignment.value is not None:
            eval_values[assignment.name] = assignment.value
        else:
            eval_values.pop(assignment.name, None)
    eval_values.update(values)

    # Evaluate ALL initval/endval entries sequentially, like Dynare does:
    # an exogenous expression may reference an endogenous assigned earlier
    # in the block (``y = 2; e = y;``), so endogenous values must flow into
    # the evaluation environment even though only exogenous names are
    # returned.
    for entry in entries:
        if entry.value is not None:
            value = entry.value
        else:
            expr_py = _escape_expr(_prepare_ss_expression(entry.expression))
            env = _build_eval_env(eval_values)
            value, _err = _safe_eval_expr(expr_py, env)
            if value is None:
                if entry.name in exo_names:
                    values.pop(entry.name, None)
                eval_values.pop(entry.name, None)
                continue
        if entry.name in exo_names:
            values[entry.name] = value
        eval_values[entry.name] = value

    return values


def _var_log_value_errors(
    model: ParsedModel,
    values: Dict[str, float],
) -> List[Tuple[str, str, SourceRange]]:
    """Return domain errors for ``var(log)`` steady-state levels."""
    errors: List[Tuple[str, str, SourceRange]] = []
    for declaration in model.endogenous:
        if not getattr(declaration, "log_transform", False):
            continue
        if declaration.name not in values:
            continue
        value = values[declaration.name]
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = math.nan
        if not math.isfinite(numeric) or numeric <= 0:
            errors.append(
                (
                    declaration.name,
                    (
                        "var(log) requires a positive steady-state level "
                        f"for '{declaration.name}'"
                    ),
                    declaration.range,
                )
            )
    return errors


# ---------------------------------------------------------------------------
# Equation evaluation
# ---------------------------------------------------------------------------


def _split_equation(ss_text: str) -> Tuple[str, str]:
    """Split an equation on ``=``, ignoring ``==``, ``<=``, ``>=``, ``~=``.

    Returns (lhs, rhs) or raises ValueError if no valid ``=`` found.
    """
    # Find all standalone = signs (not part of ==, <=, >=, ~=)
    i = 0
    eq_positions = []
    while i < len(ss_text):
        ch = ss_text[i]
        if ch == "=" and i + 1 < len(ss_text) and ss_text[i + 1] == "=":
            i += 2  # skip ==
        elif (
            ch in ("<", ">", "~", "!")
            and i + 1 < len(ss_text)
            and ss_text[i + 1] == "="
        ):
            i += 2  # skip <=, >=, ~=, !=
        elif ch == "=":
            eq_positions.append(i)
            i += 1
        else:
            i += 1

    if not eq_positions:
        raise ValueError("No '=' found")

    # Use the first = sign
    pos = eq_positions[0]
    return ss_text[:pos].strip(), ss_text[pos + 1 :].strip()


def _evaluate_equation(
    eq: Equation,
    values: Dict[str, float],
    exogenous_names: set,
    tolerance: float = 1e-6,
    exogenous_values: Optional[Dict[str, float]] = None,
    protected_names: Optional[set] = None,
    _prebuilt_env: Optional[dict] = None,
) -> SteadyStateResult:
    """Evaluate a single model equation at steady state.

    For an equation ``LHS = RHS``, the residual is ``LHS - RHS``.
    For a bare expression (no ``=``), the residual is the expression value
    (should be zero).

    *_prebuilt_env* is an optional pre-constructed evaluation environment
    (from :func:`_build_eval_env` called once before a loop over equations).
    When provided the function skips rebuilding the env dict — giving O(1)
    setup instead of O(n) per equation.  The dict is **mutated** to add any
    model-local variables (``# name = expr``), so the same env can be reused
    for subsequent equations in the same loop.
    """
    text = eq.text.strip()

    # Skip model-local variable definitions — evaluate them and store
    if text.startswith("#"):
        protected = protected_names or set()
        local_match = _LOCAL_VAR_DEF.match(text)
        if local_match:
            name = local_match.group(1)
            expr = local_match.group(2).strip()
            expr = _escape_expr(_prepare_ss_expression(expr))
            if _prebuilt_env is not None:
                env = _prebuilt_env
            else:
                env = _build_eval_env(values)
                exo_values = exogenous_values or {}
                for exo in exogenous_names:
                    value = values.get(exo, exo_values.get(exo, 0.0))
                    env.setdefault(exo, value)
                    env.setdefault(_escape_reserved(exo), value)
            val, err = _safe_eval_expr(expr, env)
            if val is not None:
                if name not in protected:
                    values[name] = val
                    if _prebuilt_env is not None:
                        _prebuilt_env[_escape_reserved(name)] = val
            else:
                return SteadyStateResult(
                    equation=eq,
                    residual=None,
                    is_satisfied=False,
                    is_local_var=True,
                    error_message=(
                        f"Cannot evaluate model-local variable '{name}': {err}"
                    ),
                )
            # Model-local vars are always "satisfied" (they're definitions)
        return SteadyStateResult(
            equation=eq,
            residual=0.0,
            is_satisfied=True,
            is_local_var=True,
        )

    # Prepare steady state form (strips tags, removes time subscripts, ^ -> **)
    ss_text = _escape_expr(_prepare_ss_expression(text))

    if _prebuilt_env is not None:
        # Fast path: use the shared pre-built env (already contains values +
        # exogenous zero-overrides), avoiding a 700-entry dict copy and a
        # full _build_eval_env call per equation.
        env = _prebuilt_env
    else:
        # Slow path (single-call usage without a prebuilt env).
        # Set exogenous variables to their declared steady-state values, or 0.
        ss_values = dict(values)
        exo_values = exogenous_values or {}
        for exo in exogenous_names:
            ss_values[exo] = values.get(exo, exo_values.get(exo, 0.0))
        env = _build_eval_env(ss_values)
        for exo in exogenous_names:
            value = ss_values[exo]
            env[exo] = value
            env[_escape_reserved(exo)] = value

    # Check for missing variables before attempting evaluation
    missing = _find_undefined_vars(ss_text, env)
    if missing:
        return SteadyStateResult(
            equation=eq,
            residual=None,
            error_message=f"Cannot evaluate: undefined variable(s) {', '.join(missing)}",
            missing_vars=missing,
        )

    try:
        lhs_expr, rhs_expr = _split_equation(ss_text)
    except ValueError:
        # Bare expression — should evaluate to zero
        val, err = _safe_eval_expr(ss_text, env)
        if err:
            return SteadyStateResult(
                equation=eq,
                residual=None,
                error_message=f"Cannot evaluate expression: {err}",
            )
        if val is None:
            return SteadyStateResult(
                equation=eq,
                residual=None,
                error_message="Cannot evaluate expression: no numeric value",
            )
        is_satisfied = abs(val) < tolerance
        return SteadyStateResult(
            equation=eq,
            residual=val,
            is_satisfied=is_satisfied,
            error_message=""
            if is_satisfied
            else f"Expression value = {val:.2e} (expected 0)",
        )

    lhs_val, lhs_err = _safe_eval_expr(lhs_expr, env)
    if lhs_err:
        return SteadyStateResult(
            equation=eq,
            residual=None,
            error_message=f"Cannot evaluate LHS: {lhs_err}",
        )
    if lhs_val is None:
        return SteadyStateResult(
            equation=eq,
            residual=None,
            error_message="Cannot evaluate LHS: no numeric value",
        )

    rhs_val, rhs_err = _safe_eval_expr(rhs_expr, env)
    if rhs_err:
        return SteadyStateResult(
            equation=eq,
            residual=None,
            error_message=f"Cannot evaluate RHS: {rhs_err}",
        )
    if rhs_val is None:
        return SteadyStateResult(
            equation=eq,
            residual=None,
            error_message="Cannot evaluate RHS: no numeric value",
        )

    residual = lhs_val - rhs_val

    # Use relative tolerance for large values
    scale = max(abs(lhs_val), abs(rhs_val), 1.0)
    is_satisfied = abs(residual) / scale < tolerance

    return SteadyStateResult(
        equation=eq,
        residual=residual,
        is_satisfied=is_satisfied,
        error_message=""
        if is_satisfied
        else (f"Residual = {residual:.2e} (LHS = {lhs_val:.6g}, RHS = {rhs_val:.6g})"),
    )


def _pre_evaluate_model_local_variables(
    model: ParsedModel,
    values: Dict[str, float],
    exogenous_names: set,
    exogenous_values: Optional[Dict[str, float]] = None,
) -> None:
    """Populate model-local values before checking equations that may use them."""
    protected_names = model.all_declared_names()
    local_defs = []
    for eq in model.model_equations:
        text = eq.text.strip()
        if not text.startswith("#"):
            continue
        match = _LOCAL_VAR_DEF.match(text)
        if match:
            name = match.group(1)
            if name not in protected_names:
                local_defs.append((name, match.group(2).strip()))

    if not local_defs:
        return

    exo_values = exogenous_values or {}
    pending = list(local_defs)
    for _ in range(len(local_defs)):
        progressed = False
        next_pending = []
        for name, expr in pending:
            expr_py = _escape_expr(_prepare_ss_expression(expr))
            env = _build_eval_env(values)
            for exo in exogenous_names:
                value = values.get(exo, exo_values.get(exo, 0.0))
                env.setdefault(exo, value)
                env.setdefault(_escape_reserved(exo), value)
            val, _err = _safe_eval_expr(expr_py, env)
            if val is not None:
                values[name] = val
                progressed = True
            else:
                next_pending.append((name, expr))
        if not progressed:
            break
        pending = next_pending


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_computed_steady_state(
    model: ParsedModel,
    computed_values: Dict[str, float],
    tolerance: float = 1e-6,
    param_overrides: Optional[Dict[str, float]] = None,
) -> SteadyStateReport:
    """Validate externally-computed values against model equations.

    Like ``validate_steady_state`` but uses *provided* values instead of
    reading from ``steady_state_model`` or ``initval`` blocks.  This
    bridges solver output into the existing validation/display pipeline.
    """
    exogenous = model.exogenous_names()
    exo_values = _exogenous_values(model, param_overrides=param_overrides)
    endo_names = model.endogenous_names()

    values: Dict[str, float] = {}
    try:
        from .solver import _effective_param_values

        values.update(
            _effective_param_values(model, param_overrides=param_overrides),
        )
    except Exception:
        values.update(model.param_values())
        if param_overrides:
            for name, value in param_overrides.items():
                if name in model.parameter_names():
                    values[name] = float(value)
    values.update(exo_values)
    values.update(computed_values)

    missing_endo = sorted(name for name in endo_names if name not in values)

    results: List[SteadyStateResult] = []
    eval_values = dict(
        values
    )  # local copy; _evaluate_equation may add model-local vars
    _pre_evaluate_model_local_variables(model, eval_values, exogenous, exo_values)
    shared_env = _build_eval_env(eval_values)
    for exo in exogenous:
        v = eval_values.get(exo, exo_values.get(exo, 0.0))
        shared_env[exo] = v
        shared_env[_escape_reserved(exo)] = v
    for eq in model.steady_state_check_equations():
        result = _evaluate_equation(
            eq,
            eval_values,
            exogenous,
            tolerance,
            exo_values,
            model.all_declared_names(),
            _prebuilt_env=shared_env,
        )
        results.append(result)

    report = SteadyStateReport(
        values=values,
        results=results,
        value_errors=_var_log_value_errors(model, values),
    )
    report._missing_endo = missing_endo
    return report


def _command_scan_text(model: ParsedModel) -> str:
    return _mask_string_literals(_strip_comments(model.text or ""))


def _is_deterministic_transition(model: ParsedModel) -> bool:
    """A perfect-foresight / deterministic model supplies ``initval`` as the
    INITIAL CONDITION of a transition path (deliberately off the steady state)
    and uses ``endval`` for the terminal steady state, so its ``initval`` must
    NOT be validated as a steady state. Detect via an ``endval`` block or a
    perfect-foresight / deterministic ``simul`` run command. ``stoch_simul`` is
    excluded by the word-boundary lookbehind (its 'simul' is preceded by '_')."""
    if model.endval_entries:
        return True
    command_text = _command_scan_text(model)
    return bool(
        re.search(
            r"(?<!\w)(?:perfect_foresight_setup|perfect_foresight_solver|"
            r"extended_path|simul)\s*[(;]",
            command_text,
            re.IGNORECASE,
        )
    )


def _steady_command_uses_nocheck(model: ParsedModel) -> bool:
    """Return True when a top-level ``steady(nocheck);`` asks Dynare not to check."""
    stripped = _command_scan_text(model)
    return bool(
        re.search(
            r"(?<!\w)steady\s*\([^;\)]*\bnocheck\b[^;\)]*\)\s*;",
            stripped,
            re.IGNORECASE,
        )
    )


def validate_steady_state(
    model: ParsedModel,
    tolerance: float = 1e-6,
    param_overrides: Optional[Dict[str, float]] = None,
) -> Optional[SteadyStateReport]:
    """Validate that steady state values satisfy the model equations.

    Returns None if there is no steady state information (no steady_state_model
    or initval block), otherwise returns a SteadyStateReport with:
      - Computed steady state values for all variables
      - Per-equation evaluation results (satisfied, residual, or error)
      - A summary() method for human-readable output
    """
    if not model.static_model_equations():
        return None

    if _steady_command_uses_nocheck(model):
        return None

    # Determine source of steady state values
    if model.steady_state_equations:
        values, value_errors = _compute_ss_values_from_block(
            model,
            param_overrides=param_overrides,
        )
    elif model.initval_entries and not _is_deterministic_transition(model):
        values, value_errors = _compute_ss_values_from_initval(
            model,
            param_overrides=param_overrides,
        )
    else:
        # No steady-state block, and either no initval or a deterministic
        # (perfect-foresight) model whose initval is a transition start rather
        # than a steady state -- there is nothing to validate as a steady state.
        return None

    exogenous = model.exogenous_names()
    exo_values = _exogenous_values(model, param_overrides=param_overrides)
    values.update(exo_values)
    value_errors = list(value_errors)
    value_errors.extend(_var_log_value_errors(model, values))

    # Check which endogenous variables are missing
    endo_names = model.endogenous_names()
    syntactically_assigned: set = set()
    multivariate_assigned: set = set()
    for eq in model.steady_state_equations:
        lhs = eq.lhs.strip()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", lhs):
            syntactically_assigned.add(lhs)
            continue
        # Dynare's multivariate assignment ``[r, w] = f(...);`` assigns
        # every bracketed name; the values come from an (often external
        # MATLAB) function the LSP cannot evaluate.
        if re.match(r"^\[[A-Za-z0-9_,\s]+\]$", lhs):
            names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", lhs)
            syntactically_assigned.update(names)
            multivariate_assigned.update(names)
    missing_endo = sorted(
        name
        for name in endo_names
        if name not in values and name not in syntactically_assigned
    )

    # Evaluate each model equation.  Build the eval environment once and share
    # it across all _evaluate_equation calls; this avoids an O(n) dict-copy +
    # _build_eval_env rebuild per equation (was O(n^2) on large models).
    results: List[SteadyStateResult] = []
    # Multivariate-assigned names without a computed value stay out of the
    # eval environment: defaulting them to 0.0 would turn every equation
    # that references them into a false non-zero residual (W040/W041).
    # Leaving them out routes those equations to the silent
    # missing-variable path instead.
    eval_values = {
        name: 0.0
        for name in endo_names
        if name not in multivariate_assigned or name in values
    }
    eval_values.update(
        values
    )  # local copy; _evaluate_equation may add model-local vars
    _pre_evaluate_model_local_variables(model, eval_values, exogenous, exo_values)
    shared_env = _build_eval_env(eval_values)
    for exo in exogenous:
        v = eval_values.get(exo, exo_values.get(exo, 0.0))
        shared_env[exo] = v
        shared_env[_escape_reserved(exo)] = v
    for eq in model.steady_state_check_equations():
        result = _evaluate_equation(
            eq,
            eval_values,
            exogenous,
            tolerance,
            exo_values,
            model.all_declared_names(),
            _prebuilt_env=shared_env,
        )
        results.append(result)

    report = SteadyStateReport(
        values=values,
        results=results,
        value_errors=value_errors,
    )
    report._missing_endo = missing_endo

    return report
