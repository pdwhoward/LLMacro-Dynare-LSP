"""Automatic deterministic steady state solver for Dynare models.

Computes steady state values by solving the system of nonlinear equations
F(x) = 0, where x is the vector of endogenous variable values and F maps
to equation residuals at steady state.

At steady state:
  - All leads and lags collapse: y(+1) = y(-1) = y
  - All exogenous shocks are zero
  - Each model equation must hold (residual = 0)

Solver strategy chain (each step feeds into the next):
  1. Build initial guess from initval / steady_state_model / heuristics
  2. Gauss-Seidel pre-conditioning: iteratively solve single-variable
     subproblems to fix definitional equations and improve the guess
  2b. Dulmage-Mendelsohn block-triangular decomposition (Dynare's default
     solve_algo=4): permute the system to block-triangular form and solve the
     recursive blocks in topological order; declines for irreducible systems
  3. scipy.optimize.least_squares with trust-region methods (trf, dogbox)
  4. scipy.optimize.root with hybr, lm, broyden1, krylov
  5. Homotopy continuation: gradually morph from x0 to the true solution
  6. Random restarts with least_squares

Requires scipy (optional dependency). Install with:
  pip install dynare-lsp[solver]
"""

from __future__ import annotations

import ast
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, cast

from .parser import ParsedModel
from .steady_state import (
    _build_eval_env,
    evaluate_steady_state_model_assignments,
    _escape_expr,
    _escape_reserved,
    _exogenous_values,
    _logncdf,
    _LOCAL_VAR_DEF,
    _norminv,
    _prepare_ss_expression,
    _safe_eval_expr,
    _split_equation,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy


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
                node.value, (int, float, complex),
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


@dataclass
class SolverResult:
    """Result of attempting to compute steady state values."""
    success: bool
    values: Dict[str, float]
    residual_norm: float
    method_used: str
    iterations: int
    message: str
    equation_residuals: List[float] = field(default_factory=list)
    initial_guess: Dict[str, float] = field(default_factory=dict)
    symbolic_steps: List[str] = field(default_factory=list)
    n_symbolic: int = 0
    n_numerical: int = 0
    # True when the solve stopped because its wall-clock budget elapsed before
    # convergence -- distinct from a genuine non-convergence failure, so callers
    # can treat it as "inconclusive" rather than "no steady state exists".
    timed_out: bool = False


def _past_deadline(deadline: Optional[float]) -> bool:
    """True once a solve's monotonic wall-clock budget has elapsed.

    ``deadline`` is an absolute ``time.monotonic()`` value, or ``None`` for an
    unbounded solve (the default), in which case this is always ``False``.
    """
    return deadline is not None and time.monotonic() >= deadline


def default_solve_budget() -> Optional[float]:
    """Wall-clock budget (seconds) for an interactive LSP/MCP steady-state solve.

    Reads ``DYNARE_LSP_SOLVE_BUDGET_SECONDS`` (default 5s; set <= 0 to disable).
    An LSP must stay responsive: normal models solve in well under a second, so
    this is a tight backstop that stops a pathological or oversized model from
    making the editor wait.  Pass the return value as
    ``compute_steady_state(..., time_budget=default_solve_budget())``.
    """
    try:
        budget = float(os.environ.get("DYNARE_LSP_SOLVE_BUDGET_SECONDS", "5"))
    except ValueError:
        return 5.0
    if not math.isfinite(budget):
        return 5.0
    return None if budget <= 0 else budget


# Symbolic steady-state reduction (sympy) helps SMALL models only: it returns
# exact closed-form solutions, but sympy's expression growth is super-linear, so
# on larger systems it can grind for minutes (a 70-equation model was measured
# still running past 135 CPU-seconds with no result).  Large DSGE steady states
# are not closed-form anyway, so above this many endogenous variables the solver
# skips symbolic reduction and goes straight to the numeric cascade.  Tunable
# via DYNARE_LSP_SYMBOLIC_MAX_VARS.
try:
    _SYMBOLIC_MAX_VARS = int(os.environ.get("DYNARE_LSP_SYMBOLIC_MAX_VARS", "30"))
except ValueError:
    _SYMBOLIC_MAX_VARS = 30


class _SkipSymbolicReduction(Exception):
    """Internal sentinel: skip symbolic reduction (size or deadline guard)."""


class _SolveDeadlineExceeded(Exception):
    """Raised from the residual function when the solve's time budget elapses.

    This is what actually bounds a single in-flight SciPy/LAPACK call: the
    optimisers evaluate the residual on every step, so raising here aborts the
    call the instant the budget is gone -- something ``max_nfev`` and
    between-stage deadline checks cannot do.
    """


def _deadline_guarded(fn, deadline):
    """Wrap a residual function so it raises ``_SolveDeadlineExceeded`` once the
    monotonic ``deadline`` has passed.

    A ``None`` deadline returns ``fn`` unchanged, so unbounded solves pay no
    per-evaluation overhead and keep their original behaviour.
    """
    if deadline is None:
        return fn

    def guarded(x):
        if time.monotonic() >= deadline:
            raise _SolveDeadlineExceeded
        return fn(x)

    return guarded


# ---------------------------------------------------------------------------
# Solver-safe math environment
# ---------------------------------------------------------------------------

def _build_solver_env(values: Dict[str, float]) -> dict:
    """Build eval environment with overflow-safe math functions.

    Unlike _build_eval_env (which uses math.* and throws on overflow),
    this uses clipped/guarded versions that return finite values or nan,
    keeping the solver's Jacobian well-defined even at extreme points.
    """
    import numpy as _np

    def _safe_exp(x):
        try:
            return float(_np.exp(_np.clip(float(x), -700, 700)))
        except (TypeError, ValueError):
            return float('nan')

    def _safe_log(x):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return float('nan')
        if x <= 0:
            return -1e10
        return float(_np.log(x))

    def _safe_sqrt(x):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return float('nan')
        if x < 0:
            return float('nan')
        return float(_np.sqrt(x))

    def _safe_cbrt(x):
        try:
            return float(_np.cbrt(float(x)))
        except (TypeError, ValueError):
            return float('nan')

    env: dict = {
        "exp": _safe_exp,
        "log": _safe_log,
        "ln": _safe_log,
        "log2": lambda x: _safe_log(x) / _safe_log(2),
        "log10": lambda x: _safe_log(x) / _safe_log(10),
        "sqrt": _safe_sqrt,
        "cbrt": _safe_cbrt,
        "abs": lambda x: abs(float(x)),
        "sign": lambda x: float((float(x) > 0) - (float(x) < 0)),
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
            _safe_exp(-0.5 * ((x - mu) / sigma) ** 2)
            / (sigma * math.sqrt(2 * math.pi))
        ),
        "normcdf": lambda x, mu=0, sigma=1: (
            0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))
        ),
        "norminv": _norminv,
        "logncdf": _logncdf,
    }
    env.update({
        name.upper(): value
        for name, value in list(env.items())
        if name.upper() != name
    })
    env.update({_escape_reserved(k): v for k, v in values.items()})
    return env


def _build_static_solver_env(
    params: Dict[str, float],
    exogenous: set,
    exogenous_values: Dict[str, float],
) -> dict:
    """Build the *static* part of the solver eval environment.

    Returns a dict containing math functions, their uppercase aliases, all
    parameter values, and all exogenous values — everything that does NOT
    change between equations within a Gauss-Seidel sweep.  Variable values
    are NOT included; callers must copy this dict and inject them separately.

    Calling this once and then doing ``env = dict(base); env[esc_var] = val``
    is ~10× faster than calling ``_build_solver_env`` on every equation
    because it avoids rebuilding 50+ function objects and their uppercase
    aliases on each call.
    """
    # Reuse _build_solver_env with an empty values dict to get the math env,
    # then layer in params and exo with pre-escaped keys.
    env = _build_solver_env({})
    for k, v in params.items():
        env[_escape_reserved(k)] = v
    for exo in exogenous:
        val = exogenous_values.get(exo, 0.0)
        env[exo] = val
        env[_escape_reserved(exo)] = val
    return env


def _make_env_from_base(
    base_env: dict,
    x,
    escaped_var_names: List[str],
    locals_list: List[_PrepLocal],
) -> dict:
    """Build a full eval environment by copying *base_env* and injecting vars.

    This is the fast path for Gauss-Seidel: the static env (math + params +
    exo) is pre-built once outside the sweep loop; here we shallow-copy it
    and update only the variable entries.  ``escaped_var_names[i]`` must be
    ``_escape_reserved(var_names[i])`` for each index ``i``.

    The locals loop is identical to ``_make_env`` — it must run after the
    variable values are in place because locals may depend on variables.
    """
    import numpy as _np

    env = dict(base_env)
    for j, esc_name in enumerate(escaped_var_names):
        env[esc_name] = float(x[j])

    # Evaluate local var definitions until forward references settle.
    pending = list(locals_list)
    for _ in range(len(locals_list)):
        progressed = False
        remaining: List[_PrepLocal] = []
        for loc in pending:
            val = _safe_eval_compiled(loc.code, env)
            if _np.isnan(val):
                remaining.append(loc)
                continue
            env[loc.escaped_name] = val
            progressed = True
        if not progressed:
            break
        pending = remaining

    return env


def _safe_eval_compiled(code, env: dict) -> float:
    """Evaluate a compiled code object, returning nan on any error."""
    try:
        result = eval(code, {"__builtins__": {}}, env)
        if isinstance(result, complex):
            if abs(result.imag) < 1e-10:
                return float(result.real)
            return float('nan')
        r = float(result)
        # Treat extreme values as nan to avoid polluting Jacobians
        if not (-1e200 < r < 1e200):
            return float('nan')
        return r
    except Exception:
        return float('nan')


# ---------------------------------------------------------------------------
# Equation preprocessing  (compile once, evaluate many times)
# ---------------------------------------------------------------------------

@dataclass
class _PrepEq:
    """A pre-processed model equation ready for fast evaluation."""
    lhs_code: object          # compiled Python code
    rhs_code: Optional[object]  # compiled Python code or None
    lhs_str: str
    rhs_str: Optional[str]
    var_indices: List[int]    # indices into var_names of variables in this eq


@dataclass
class _PrepLocal:
    """A pre-processed model-local variable definition."""
    name: str
    escaped_name: str
    code: object              # compiled Python code
    expr_str: str


def _preprocess_model(
    model: ParsedModel,
    var_names: List[str],
) -> Tuple[List[_PrepEq], List[_PrepLocal]]:
    """Parse and compile all model equations and local variable definitions.

    Returns (equations, local_var_defs).
    """
    locals_list: List[_PrepLocal] = []
    equations: List[_PrepEq] = []
    declared_names = model.all_declared_names()

    for eq in model.model_equations:
        text = eq.text.strip()

        # --- local variable definition ---
        if text.startswith("#"):
            match = _LOCAL_VAR_DEF.match(text)
            if match:
                name = match.group(1)
                if name in declared_names:
                    continue
                expr_str = _escape_expr(
                    _prepare_ss_expression(match.group(2).strip()))
                if not _validate_ast(expr_str):
                    logger.debug(
                        "Rejected unsafe local-var expression: %s", expr_str)
                    continue
                try:
                    code = compile(expr_str, "<local>", "eval")
                except SyntaxError:
                    continue
                locals_list.append(_PrepLocal(
                    name=name,
                    escaped_name=_escape_reserved(name),
                    code=code,
                    expr_str=expr_str,
                ))
            continue
        if "dynamic" in eq.tags:
            continue

        # --- normal equation ---
        ss_text = _escape_expr(_prepare_ss_expression(text))
        try:
            lhs_str, rhs_str = _split_equation(ss_text)
        except ValueError:
            lhs_str, rhs_str = ss_text, None

        if not _validate_ast(lhs_str):
            logger.debug("Rejected unsafe LHS expression: %s", lhs_str)
            continue
        if rhs_str is not None and not _validate_ast(rhs_str):
            logger.debug("Rejected unsafe RHS expression: %s", rhs_str)
            continue

        try:
            lhs_code = compile(lhs_str, "<eq-lhs>", "eval")
        except SyntaxError:
            continue
        rhs_code = None
        if rhs_str is not None:
            try:
                rhs_code = compile(rhs_str, "<eq-rhs>", "eval")
            except SyntaxError:
                continue

        # Identify which endogenous variables appear.
        # Extract all identifiers from the equation text once with a single
        # findall, then check set membership -- O(len(eq_text) + n_vars) instead
        # of the O(n_vars * len(eq_text)) per-variable re.search loop (~20x
        # faster on a 361-equation model).
        full_text = lhs_str + " " + (rhs_str if rhs_str else "")
        names_in_eq = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', full_text))
        var_indices = [
            j for j, name in enumerate(var_names)
            if _escape_reserved(name) in names_in_eq
        ]

        equations.append(_PrepEq(
            lhs_code=lhs_code, rhs_code=rhs_code,
            lhs_str=lhs_str, rhs_str=rhs_str,
            var_indices=var_indices,
        ))

    return equations, locals_list


# ---------------------------------------------------------------------------
# Fast residual evaluation (solver-safe, compiled)
# ---------------------------------------------------------------------------

def _make_env(
    x, var_names: List[str],
    params: Dict[str, float], exogenous: set,
    exogenous_values: Dict[str, float],
    locals_list: List[_PrepLocal],
) -> dict:
    """Build a solver-safe eval environment from a variable vector."""
    values: Dict[str, float] = dict(params)
    for j, name in enumerate(var_names):
        values[name] = float(x[j])
    for exo in exogenous:
        values[exo] = exogenous_values.get(exo, 0.0)

    env = _build_solver_env(values)
    for exo in exogenous:
        value = exogenous_values.get(exo, 0.0)
        env[exo] = value
        env[_escape_reserved(exo)] = value

    # Evaluate local var definitions until forward references settle.
    import numpy as _np
    pending = list(locals_list)
    for _ in range(len(locals_list)):
        progressed = False
        remaining: List[_PrepLocal] = []
        for loc in pending:
            val = _safe_eval_compiled(loc.code, env)
            if _np.isnan(val):
                remaining.append(loc)
                continue
            env[loc.escaped_name] = val
            progressed = True
        if not progressed:
            break
        pending = remaining

    return env


def _eval_eq(eq: _PrepEq, env: dict) -> float:
    """Evaluate a single equation residual."""
    import numpy as _np
    if eq.rhs_code is not None:
        lv = _safe_eval_compiled(eq.lhs_code, env)
        rv = _safe_eval_compiled(eq.rhs_code, env)
        if _np.isnan(lv) or _np.isnan(rv):
            return float('nan')
        return lv - rv
    else:
        return _safe_eval_compiled(eq.lhs_code, env)


def _eval_all(
    equations: List[_PrepEq], var_names: List[str],
    x, params: Dict[str, float], exogenous: set,
    exogenous_values: Dict[str, float],
    locals_list: List[_PrepLocal],
) -> "numpy.ndarray":
    """Evaluate all equation residuals at x, returning a numpy array."""
    import numpy as np
    env = _make_env(
        x, var_names, params, exogenous, exogenous_values, locals_list,
    )
    residuals = np.zeros(len(equations))
    for i, eq in enumerate(equations):
        r = _eval_eq(eq, env)
        if np.isnan(r):
            residuals[i] = 1e6 * (1 + np.sum(np.abs(x)))
        else:
            residuals[i] = r
    return residuals


def _build_compiled_residual_fn(
    equations: List[_PrepEq], locals_list: List[_PrepLocal],
    var_names: List[str], params: Dict[str, float], exogenous: set,
    exogenous_values: Dict[str, float],
) -> Optional[Callable]:
    """Compile ALL residuals into one function (one call instead of two
    ``eval``s per equation -- ~8x faster than the loop on a 400-equation model).

    Returns ``None`` if code generation fails, so the caller falls back to the
    loop.  The compiled function itself falls back to ``_eval_all`` on any
    per-point edge case (a complex sub-result, a raised exception, or a value
    past the ``1e200`` clamp), so it is numerically identical to the loop at
    every point.
    """
    import numpy as np

    static_vals: Dict[str, float] = dict(params)
    for exo in exogenous:
        static_vals[exo] = exogenous_values.get(exo, 0.0)
    g = _build_solver_env(static_vals)
    for exo in exogenous:
        val = exogenous_values.get(exo, 0.0)
        g[exo] = val
        g[_escape_reserved(exo)] = val

    lines = ["def _F(x):"]
    for i, name in enumerate(var_names):
        # float() so a negative base to a fractional power yields a Python
        # complex (handled by the fallback) exactly as the loop does -- not a
        # NumPy nan -- keeping the two paths bit-for-bit equivalent.
        lines.append(f"    {_escape_reserved(name)} = float(x[{i}])")
    for loc in locals_list:
        lines.append(f"    {loc.escaped_name} = {loc.expr_str}")
    lhs_terms: List[str] = []
    rhs_terms: List[str] = []
    has_rhs: List[bool] = []
    for eq in equations:
        lhs_terms.append(f"({eq.lhs_str})")
        rhs_terms.append(f"({eq.rhs_str})" if eq.rhs_str is not None else "0.0")
        has_rhs.append(eq.rhs_str is not None)
    lines.append(
        "    return ([" + ",".join(lhs_terms) + "],[" + ",".join(rhs_terms) + "])"
    )
    ns: dict = {}
    try:
        exec(compile("\n".join(lines), "<vec-residual>", "exec"), g, ns)  # noqa: S102
    except Exception:
        return None
    _F = ns["_F"]
    has_rhs_arr = np.array(has_rhs)

    def F(x):
        try:
            lhs_list, rhs_list = _F(x)
            lhs = np.asarray(lhs_list, dtype=float)
            rhs = np.asarray(rhs_list, dtype=float)
        except Exception:
            return _eval_all(equations, var_names, x, params, exogenous,
                             exogenous_values, locals_list)
        # Fast path: all sub-results finite and within the clamp -> no clamp or
        # penalty is needed, so the result equals the loop's exactly.
        if (np.all(np.isfinite(lhs)) and np.all(np.isfinite(rhs))
                and float(np.max(np.abs(lhs))) < 1e200
                and float(np.max(np.abs(rhs))) < 1e200):
            return np.where(has_rhs_arr, lhs - rhs, lhs)
        # Edge values: reproduce the per-equation clamp + nan->penalty exactly.
        lhs = np.where(np.abs(lhs) < 1e200, lhs, np.nan)
        rhs = np.where(np.abs(rhs) < 1e200, rhs, np.nan)
        res = np.where(has_rhs_arr, lhs - rhs, lhs)
        if np.any(np.isnan(res)):
            pen = 1e6 * (1.0 + float(np.sum(np.abs(x))))
            res = np.where(np.isnan(res), pen, res)
        return res

    return F


def _build_solver_residual_fn(
    equations: List[_PrepEq], locals_list: List[_PrepLocal],
    var_names: List[str], params: Dict[str, float], exogenous: set,
    exogenous_values: Dict[str, float],
) -> Callable:
    """Return a vectorised residual function F(x) -> ndarray for scipy.

    Prefers a single compiled-all-equations function; falls back to the
    per-equation loop when code generation is unavailable.
    """
    compiled = _build_compiled_residual_fn(
        equations, locals_list, var_names, params, exogenous, exogenous_values,
    )
    if compiled is not None:
        return compiled

    def F(x):
        return _eval_all(
            equations, var_names, x, params, exogenous,
            exogenous_values, locals_list,
        )
    return F


# ---------------------------------------------------------------------------
# Residual function construction  (original, for validation)
# ---------------------------------------------------------------------------

def _detect_invalid_domain(
    model_equations: List,
    var_names: List[str],
    solution_values: Dict[str, float],
    param_values: Dict[str, float],
    exogenous_names: set,
    exogenous_values: Dict[str, float],
) -> Optional[str]:
    """Check whether the proposed steady-state values land in an invalid
    domain for any ``log(...)`` or ``sqrt(...)`` operand.

    Returns ``None`` when all operands sit in their domains, or a short
    diagnostic string identifying the offending expression otherwise.

    Without this guard, ``_safe_log`` / ``_safe_sqrt`` clamp invalid
    inputs to ``-1e10`` or ``nan`` and the solver can declare success
    on mathematically nonsensical roots — e.g. two equations
    ``log(y) = log(z)`` and ``y = z`` both touching ``y < 0`` cancel
    each other in the residual.
    """
    import re as _re
    # Build a substitution environment that includes everything the
    # equation might reference.  We don't escape reserved names because
    # the eval here is on the ORIGINAL Dynare expression text, not the
    # internal escaped form the solver uses.
    env: Dict[str, float] = dict(param_values)
    env.update(solution_values)
    for exo in exogenous_names:
        env[exo] = exogenous_values.get(exo, 0.0)

    # Math functions we care about validating.  Each entry maps the
    # function name to a predicate for "argument is in the valid
    # domain"; the message is what we surface on failure.
    domain_checks = {
        "log": (lambda v: v > 0, "log of non-positive value"),
        "ln":  (lambda v: v > 0, "ln of non-positive value"),
        "log2": (lambda v: v > 0, "log2 of non-positive value"),
        "log10": (lambda v: v > 0, "log10 of non-positive value"),
        "sqrt": (lambda v: v >= 0, "sqrt of negative value"),
    }

    for eq in model_equations:
        text = eq.text.strip()
        if not text:
            continue
        if text.startswith("#"):
            match = _LOCAL_VAR_DEF.match(text)
            if not match:
                continue
            text = match.group(2).strip()
        # Strip equation tags and steady-state-style mark-ups so the
        # plain expression remains for argument extraction.
        text = _re.sub(r"\[[^\]]*\]", "", text)
        # Find each math function and extract its parenthesized argument.
        for fname, (predicate, msg) in domain_checks.items():
            for m in _re.finditer(rf"\b{fname}\s*\(", text):
                # Walk to the matching ``)``.
                depth = 1
                j = m.end()
                while j < len(text) and depth > 0:
                    if text[j] == "(":
                        depth += 1
                    elif text[j] == ")":
                        depth -= 1
                    j += 1
                arg = text[m.end():j - 1].strip()
                # Strip time subscripts at the top level so ``y(+1)`` -> ``y``.
                arg_ss = _re.sub(r"\(\s*[+\-]?\s*\d+\s*\)", "", arg)
                # Evaluate the argument in a guarded environment.
                try:
                    safe_env = _build_eval_env(env)
                    arg_expr = _escape_expr(arg_ss.replace("^", "**"))
                    if not _validate_ast(arg_expr):
                        continue
                    arg_value = eval(  # noqa: S307 — AST-validated, controlled env
                        arg_expr,
                        {"__builtins__": {}},
                        safe_env,
                    )
                    if not predicate(float(arg_value)):
                        return f"{msg} (in equation '{text[:80]}')"
                except Exception:
                    # Argument may reference a name we don't have or use
                    # syntax we can't evaluate — that's a softer failure
                    # already covered by other diagnostics.  Skip silently.
                    continue
        for denominator in _division_denominators(text):
            denominator_ss = _re.sub(
                r"\(\s*[+\-]?\s*\d+\s*\)",
                "",
                denominator,
            )
            try:
                safe_env = _build_eval_env(env)
                denom_expr = _escape_expr(denominator_ss.replace("^", "**"))
                if not _validate_ast(denom_expr):
                    continue
                value = eval(  # noqa: S307 - AST-validated, controlled env
                    denom_expr,
                    {"__builtins__": {}},
                    safe_env,
                )
                if abs(float(value)) <= 1e-12:
                    return f"division by zero (in equation '{text[:80]}')"
            except Exception:
                continue
    return None


def _division_denominators(text: str) -> List[str]:
    """Return denominator expressions from simple division operators."""
    out: List[str] = []
    i = 0
    while i < len(text):
        if text[i] != "/":
            i += 1
            continue
        if i + 1 < len(text) and text[i + 1] == "/":
            break
        if i > 0 and text[i - 1] == ".":
            i += 1
            continue
        j = i + 1
        while j < len(text) and text[j].isspace():
            j += 1
        if j >= len(text):
            break
        if text[j] == "(":
            depth = 1
            k = j + 1
            while k < len(text) and depth > 0:
                if text[k] == "(":
                    depth += 1
                elif text[k] == ")":
                    depth -= 1
                k += 1
            out.append(text[j + 1:k - 1].strip())
            i = k
            continue
        k = j
        while k < len(text) and re.match(r"[A-Za-z0-9_.]", text[k]):
            if text[k] in "+-" and k > j:
                break
            k += 1
        denominator = text[j:k].strip()
        if denominator:
            out.append(denominator)
        i = max(k, i + 1)
    return out


def _detect_var_log_domain(
    model: ParsedModel,
    solution_values: Dict[str, float],
) -> Optional[str]:
    """Reject non-positive steady-state levels for ``var(log)`` variables."""
    for declaration in model.endogenous:
        if not getattr(declaration, "log_transform", False):
            continue
        if declaration.name not in solution_values:
            continue
        value = solution_values[declaration.name]
        if not math.isfinite(value) or value <= 0:
            return (
                "var(log) endogenous variable "
                f"'{declaration.name}' has non-positive steady-state level"
            )
    return None


def _build_residual_function(
    model: ParsedModel,
    var_names: List[str],
    param_values: Dict[str, float],
    exogenous_names: set,
    exogenous_values: Optional[Dict[str, float]] = None,
) -> Callable:
    """Build the F(x) = 0 residual function from model equations.

    Returns a function that maps a numpy array of endogenous variable
    values to a numpy array of equation residuals.  Pre-processes all
    equations at construction time so per-call overhead is minimal.
    """
    import numpy as np
    exo_values = exogenous_values or {}

    # Pre-process equations: separate local var defs from real equations
    local_var_defs: List[Tuple[str, str]] = []  # (name, escaped_expr)
    prepared_equations: List[Tuple[str, Optional[str]]] = []  # (lhs, rhs|None)
    declared_names = model.all_declared_names()
    helper_values = {
        assignment.name: assignment.value
        for assignment in model.helper_assignments
        if assignment.value is not None
    }

    for eq in model.model_equations:
        text = eq.text.strip()
        if text.startswith("#"):
            match = _LOCAL_VAR_DEF.match(text)
            if match and match.group(1) not in declared_names:
                local_var_defs.append((
                    match.group(1),
                    _escape_expr(_prepare_ss_expression(match.group(2).strip())),
                ))
            continue
        if "dynamic" in eq.tags:
            continue

        ss_text = _escape_expr(_prepare_ss_expression(text))
        try:
            lhs, rhs = _split_equation(ss_text)
            prepared_equations.append((lhs, rhs))
        except ValueError:
            prepared_equations.append((ss_text, None))

    n_eqs = len(prepared_equations)

    def residual_fn(x):
        # Map vector to variable names
        values = dict(param_values)
        values.update(helper_values)
        for i, name in enumerate(var_names):
            values[name] = float(x[i])

        for exo in exogenous_names:
            values[exo] = exo_values.get(exo, 0.0)

        # Build eval environment and evaluate local var defs
        env = _build_eval_env(values)
        for exo in exogenous_names:
            value = exo_values.get(exo, 0.0)
            env[exo] = value
            env[_escape_reserved(exo)] = value

        pending_locals = list(local_var_defs)
        for _ in range(len(local_var_defs)):
            progressed = False
            remaining_locals: List[Tuple[str, str]] = []
            for local_name, local_expr in pending_locals:
                val, _ = _safe_eval_expr(local_expr, env)
                if val is None:
                    remaining_locals.append((local_name, local_expr))
                    continue
                env[_escape_reserved(local_name)] = val
                values[local_name] = val
                progressed = True
            if not progressed:
                break
            pending_locals = remaining_locals

        # Rebuild env with local vars included
        env = _build_eval_env(values)
        for exo in exogenous_names:
            value = exo_values.get(exo, 0.0)
            env[exo] = value
            env[_escape_reserved(exo)] = value

        residuals = np.zeros(n_eqs)
        for i, (lhs, rhs) in enumerate(prepared_equations):
            try:
                if rhs is not None:
                    lhs_val, err1 = _safe_eval_expr(lhs, env)
                    rhs_val, err2 = _safe_eval_expr(rhs, env)
                    if err1 or err2 or lhs_val is None or rhs_val is None:
                        residuals[i] = 1e6 * (1 + np.sum(np.abs(x)))
                    elif not np.isfinite(lhs_val) or not np.isfinite(rhs_val):
                        residuals[i] = 1e6 * (1 + np.sum(np.abs(x)))
                    else:
                        residuals[i] = lhs_val - rhs_val
                else:
                    val, err = _safe_eval_expr(lhs, env)
                    if err or val is None or not np.isfinite(val):
                        residuals[i] = 1e6 * (1 + np.sum(np.abs(x)))
                    else:
                        residuals[i] = val
            except Exception:
                residuals[i] = 1e6 * (1 + np.sum(np.abs(x)))

        return residuals

    return residual_fn


# ---------------------------------------------------------------------------
# Initial guess construction
# ---------------------------------------------------------------------------

def _build_initial_guess(
    model: ParsedModel,
    var_names: List[str],
    param_values: Dict[str, float],
    warm_start_guess: Optional[Dict[str, float]] = None,
) -> "numpy.ndarray":
    """Construct initial guess vector from available information.

    Layers (later overrides earlier):
      1. Default: all ones plus heuristics for common variable name patterns
      2. Warm-start values from the previous solve
      3. initval block values
      4. endval terminal steady-state values
      5. steady_state_model evaluable values
    """
    import numpy as np

    n = len(var_names)
    x0 = np.ones(n)

    # Layer 1: Heuristics for common variable types
    for i, name in enumerate(var_names):
        name_lower = name.lower()
        if name_lower.startswith("log_") or name_lower.startswith("ln_"):
            x0[i] = 0.0
        elif name_lower in ("l", "n", "hours", "labor"):
            x0[i] = 0.33

    # Layer 2: previous successful solve, used only as a starting point.
    if warm_start_guess:
        for name, value in warm_start_guess.items():
            if name in var_names:
                x0[var_names.index(name)] = float(value)

    def _apply_entries(entries):
        pre_entry_x0 = x0.copy()
        for entry in entries:
            value = entry.value
            if value is None:
                expr_py = _escape_expr(_prepare_ss_expression(entry.expression))
                env = _build_eval_env(init_values)
                value, _err = _safe_eval_expr(expr_py, env)
            if value is None:
                init_values.pop(entry.name, None)
                if entry.name in var_names:
                    idx = var_names.index(entry.name)
                    x0[idx] = pre_entry_x0[idx]
                continue
            init_values[entry.name] = value
            if entry.name in var_names:
                idx = var_names.index(entry.name)
                x0[idx] = value

    # Layer 3: initval values
    init_values: Dict[str, float] = dict(param_values)
    for assignment in model.helper_assignments:
        if assignment.value is not None:
            init_values[assignment.name] = assignment.value
        else:
            init_values.pop(assignment.name, None)
    _apply_entries(model.initval_entries)
    _apply_entries(model.endval_entries)

    # Layer 5: steady_state_model values
    if model.steady_state_equations:
        _ss_values, ss_assignments = evaluate_steady_state_model_assignments(model)
        for assignment in ss_assignments:
            if assignment.value is None or assignment.name not in var_names:
                continue
            x0[var_names.index(assignment.name)] = assignment.value

    return x0


def _effective_param_values(
    model: ParsedModel,
    param_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Return parameter values, including evaluable steady_state_model assignments."""
    params = model.param_values()
    declared_params = model.parameter_names()
    if param_overrides:
        for name, value in param_overrides.items():
            if name in declared_params:
                params[name] = float(value)
    if not declared_params:
        return params

    if not model.steady_state_equations:
        return params
    _values, ss_assignments = evaluate_steady_state_model_assignments(
        model,
        param_overrides=param_overrides,
    )
    for assignment in ss_assignments:
        if assignment.value is not None and assignment.name in declared_params:
            params[assignment.name] = assignment.value
    if param_overrides:
        for name, value in param_overrides.items():
            if name in declared_params:
                params[name] = float(value)

    return params


def _preferred_symbolic_values(
    model: ParsedModel,
    var_names: List[str],
    param_values: Dict[str, float],
    warm_start_guess: Optional[Dict[str, float]],
    user_initial_guess: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """Return explicit user/model guesses for symbolic root selection."""
    preferred: Dict[str, float] = {}
    values: Dict[str, float] = dict(param_values)
    for assignment in model.helper_assignments:
        if assignment.value is not None:
            values[assignment.name] = assignment.value
        else:
            values.pop(assignment.name, None)
    values.update(_exogenous_values(model))

    if warm_start_guess:
        for name, value in warm_start_guess.items():
            if name in var_names:
                preferred[name] = float(value)

    for entry in [*model.initval_entries, *model.endval_entries]:
        value = entry.value
        if value is None:
            expr_py = _escape_expr(_prepare_ss_expression(entry.expression))
            env = _build_eval_env(values)
            value, _err = _safe_eval_expr(expr_py, env)
        if value is None:
            values.pop(entry.name, None)
            preferred.pop(entry.name, None)
            continue
        values[entry.name] = value
        if entry.name in var_names:
            preferred[entry.name] = value

    if model.steady_state_equations:
        _ss_values, ss_assignments = evaluate_steady_state_model_assignments(model)
        for assignment in ss_assignments:
            if assignment.value is None:
                continue
            values[assignment.name] = assignment.value
            if assignment.name in var_names:
                preferred[assignment.name] = assignment.value

    if user_initial_guess:
        for name, value in user_initial_guess.items():
            if name in var_names:
                preferred[name] = float(value)

    return preferred


# ---------------------------------------------------------------------------
# Gauss-Seidel pre-conditioning
# ---------------------------------------------------------------------------

def _scalar_newton(f: Callable[[float], float], x0: float,
                   maxiter: int = 80, tol: float = 1e-12) -> Tuple[float, bool]:
    """Damped Newton's method for scalar root-finding.

    Returns (solution, converged).
    """
    import numpy as np
    x = float(x0)
    fx = f(x)
    if not np.isfinite(fx):
        return x0, False

    for _ in range(maxiter):
        if abs(fx) < tol:
            return x, True

        # Central finite difference derivative
        h = max(abs(x) * 1e-7, 1e-10)
        fxp = f(x + h)
        fxm = f(x - h)
        if not (np.isfinite(fxp) and np.isfinite(fxm)):
            # Fall back to forward difference
            if np.isfinite(fxp):
                deriv = (fxp - fx) / h
            else:
                return x, abs(fx) < tol
        else:
            deriv = (fxp - fxm) / (2 * h)

        if abs(deriv) < 1e-30:
            return x, abs(fx) < tol

        step = -fx / deriv
        # Limit step size to prevent wild jumps
        max_step = max(abs(x) * 5.0, 2.0)
        step = max(min(step, max_step), -max_step)

        # Backtracking line search
        best_x, best_fx = x, abs(fx)
        alpha = 1.0
        for _ in range(12):
            x_try = x + alpha * step
            fx_try = f(x_try)
            if np.isfinite(fx_try) and abs(fx_try) < best_fx:
                best_x = x_try
                best_fx = abs(fx_try)
                break
            alpha *= 0.5
        else:
            # No improvement found
            if best_fx < abs(fx):
                x, fx = best_x, f(best_x)
            return x, abs(fx) < tol

        x = best_x
        fx = f(x)

    return x, abs(fx) < tol


def _gauss_seidel_improve(
    equations: List[_PrepEq],
    locals_list: List[_PrepLocal],
    var_names: List[str],
    x0: "numpy.ndarray",
    params: Dict[str, float],
    exogenous: set,
    exogenous_values: Dict[str, float],
    max_sweeps: int = 15,
    tol: float = 1e-10,
) -> "numpy.ndarray":
    """Iteratively solve single-variable subproblems to improve initial guess.

    For each equation with a large residual, try solving it for each of
    its constituent variables using damped scalar Newton.  Accept the
    variable update that most reduces the residual.

    This is nonlinear Gauss-Seidel iteration -- it fixes "definitional"
    equations (u = f(C,L), etc.) quickly and also makes incremental
    progress on coupled equations.
    """
    import numpy as np

    x = x0.copy()

    # Sort equations by number of variables (solve simpler equations first)
    eq_order = sorted(range(len(equations)),
                      key=lambda i: len(equations[i].var_indices))

    # --- Fast-path: build the static part of the eval env once --------------
    # Math functions, uppercase aliases, params, and exogenous values never
    # change within or across sweeps.  Pre-compute escaped variable names so
    # the per-equation env construction is a single dict.copy() + a list of
    # direct dict assignments — no calls to _build_solver_env or
    # _escape_reserved inside the hot loop.
    base_env = _build_static_solver_env(params, exogenous, exogenous_values)
    escaped_var_names = [_escape_reserved(n) for n in var_names]
    # -------------------------------------------------------------------------

    for sweep in range(max_sweeps):
        sweep_improved = False

        for eq_idx in eq_order:
            eq = equations[eq_idx]
            if not eq.var_indices:
                continue

            # Build environment with current x (fast path: copy base + vars)
            env = _make_env_from_base(
                base_env, x, escaped_var_names, locals_list,
            )
            current_res = _eval_eq(eq, env)
            if not np.isfinite(current_res) or abs(current_res) < tol:
                continue

            best_var_idx: Optional[int] = None
            best_val: float = 0.0
            best_res: float = abs(current_res)

            for var_idx in eq.var_indices:
                esc_var = escaped_var_names[var_idx]
                current_val = x[var_idx]

                # Build scalar function: vary only this variable
                def _scalar_f(t, _esc=esc_var, _env=env,
                              _eq=eq, _ll=locals_list):
                    test_env = dict(_env)
                    test_env[_esc] = t
                    for loc in _ll:
                        lv = _safe_eval_compiled(loc.code, test_env)
                        if not np.isnan(lv):
                            test_env[loc.escaped_name] = lv
                    return _eval_eq(_eq, test_env)

                # Try damped Newton
                sol, converged = _scalar_newton(_scalar_f, current_val)
                if converged or abs(_scalar_f(sol)) < best_res * 0.5:
                    res = abs(_scalar_f(sol))
                    if res < best_res:
                        best_var_idx = var_idx
                        best_val = sol
                        best_res = res
                        if res < tol:
                            break

                # If Newton didn't fully converge, try bracket search
                if best_res > tol:
                    try:
                        from scipy.optimize import brentq
                        f0 = _scalar_f(current_val)
                        if np.isfinite(f0):
                            for scale in [0.01, 0.05, 0.1, 0.5, 1.0,
                                          2.0, 5.0, 10.0, 50.0]:
                                delta = max(abs(current_val) * scale,
                                            scale * 0.01)
                                brackets = [
                                    (current_val - delta,
                                     current_val + delta),
                                    (current_val,
                                     current_val + 2 * delta),
                                    (current_val - 2 * delta,
                                     current_val),
                                ]
                                for a, b in brackets:
                                    fa = _scalar_f(a)
                                    fb = _scalar_f(b)
                                    if (np.isfinite(fa)
                                            and np.isfinite(fb)
                                            and fa * fb < 0):
                                        try:
                                            sol2 = cast(float, brentq(
                                                _scalar_f, a, b,
                                                xtol=1e-14, maxiter=200))
                                            sol2_value = float(sol2)
                                            res2 = abs(_scalar_f(sol2_value))
                                            if res2 < best_res:
                                                best_var_idx = var_idx
                                                best_val = sol2_value
                                                best_res = res2
                                        except Exception:
                                            pass
                                if best_res < tol:
                                    break
                    except ImportError:
                        pass

            # Accept improvement if significant
            if (best_var_idx is not None
                    and best_res < abs(current_res) * 0.9):
                x[best_var_idx] = best_val
                sweep_improved = True

        if not sweep_improved:
            break

    return x


# ---------------------------------------------------------------------------
# Solver strategies
# ---------------------------------------------------------------------------

_ROOT_METHODS = [
    ("hybr", "Modified Powell hybrid method"),
    ("lm", "Levenberg-Marquardt"),
    ("broyden1", "Broyden's first method"),
    ("krylov", "Krylov approximation"),
]


def _try_root(
    residual_fn: Callable, x0: "numpy.ndarray",
    method: str, tolerance: float,
) -> Optional[Tuple["numpy.ndarray", str, int]]:
    """Attempt scipy.optimize.root with a specific method."""
    import numpy as np
    from scipy.optimize import root

    import warnings
    if method in ("hybr", "lm"):
        options: dict = {"maxfev": 10000}
    else:
        options = {"maxiter": 1000}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = root(residual_fn, x0, method=method, tol=tolerance,
                          options=options)
        if result.success and np.max(np.abs(result.fun)) < tolerance:
            niter = getattr(result, "nit", getattr(result, "nfev", 0))
            return (result.x, method, niter)
    except Exception:
        pass
    return None


def _try_least_squares(
    residual_fn: Callable, x0: "numpy.ndarray", tolerance: float,
    deadline: Optional[float] = None,
) -> Optional[Tuple["numpy.ndarray", str, int]]:
    """Try scipy.optimize.least_squares with trust-region methods.

    least_squares minimises 0.5 * ||F(x)||^2 and is significantly more
    robust than root-finding for ill-conditioned or poorly-initialised
    nonlinear systems, because the trust-region globalisation prevents
    catastrophic overshooting.
    """
    import numpy as np
    from scipy.optimize import least_squares

    for method in ("trf", "dogbox"):
        if _past_deadline(deadline):
            return None
        try:
            result = least_squares(
                residual_fn, x0, method=method,
                ftol=tolerance, xtol=tolerance, gtol=tolerance,
                max_nfev=200000,
                x_scale="jac",  # auto-scale by Jacobian column norms
            )
            max_res = float(np.max(np.abs(result.fun)))
            if max_res < tolerance:
                return (result.x, f"least_squares ({method})", result.nfev)
        except Exception:
            pass

    # Also try least_squares with 'lm' (Levenberg-Marquardt variant)
    if _past_deadline(deadline):
        return None
    try:
        result = least_squares(
            residual_fn, x0, method="lm",
            ftol=tolerance, xtol=tolerance, gtol=tolerance,
            max_nfev=200000,
        )
        max_res = float(np.max(np.abs(result.fun)))
        if max_res < tolerance:
            return (result.x, "least_squares (lm)", result.nfev)
    except Exception:
        pass

    return None


def _try_homotopy(
    residual_fn: Callable, x0: "numpy.ndarray", tolerance: float,
    n_steps: int = 50, deadline: Optional[float] = None,
) -> Optional[Tuple["numpy.ndarray", str, int]]:
    """Natural-parameter homotopy continuation.

    Defines H(x, lam) = F(x) - (1 - lam) * F(x0).
    At lam=0: H(x,0) = F(x) - F(x0) = 0  =>  x = x0 (trivial)
    At lam=1: H(x,1) = F(x) = 0            =>  true solution

    We march lam from 0 to 1 in small steps.  At each step the previous
    solution is an excellent initial guess, so even a simple solver
    converges reliably.
    """
    import numpy as np
    from scipy.optimize import least_squares

    f0 = residual_fn(x0)
    if not np.all(np.isfinite(f0)):
        return None

    x = x0.copy()
    total_nfev = 0

    for step in range(1, n_steps + 1):
        if _past_deadline(deadline):
            break
        lam = step / n_steps
        target = (1.0 - lam) * f0  # residual target for this step

        def homotopy_fn(x_inner, _tgt=target):
            return residual_fn(x_inner) - _tgt

        try:
            result = least_squares(
                homotopy_fn, x, method="trf",
                ftol=tolerance * 10, xtol=tolerance * 10,
                gtol=tolerance * 10,
                max_nfev=20000, x_scale="jac",
            )
            total_nfev += result.nfev
            # Accept even rough convergence for intermediate steps
            max_res = float(np.max(np.abs(result.fun)))
            if max_res < tolerance * 1000:
                x = result.x.copy()
            else:
                # Stuck -- try smaller steps from here
                break
        except Exception:
            break

    # Final solve at lam=1 (F(x)=0)
    try:
        result = least_squares(
            residual_fn, x, method="trf",
            ftol=tolerance, xtol=tolerance, gtol=tolerance,
            max_nfev=100000, x_scale="jac",
        )
        total_nfev += result.nfev
        max_res = float(np.max(np.abs(result.fun)))
        if max_res < tolerance:
            return (result.x, "homotopy continuation", total_nfev)
    except Exception:
        pass

    # Try final solve with dogbox too
    try:
        result = least_squares(
            residual_fn, x, method="dogbox",
            ftol=tolerance, xtol=tolerance, gtol=tolerance,
            max_nfev=100000, x_scale="jac",
        )
        total_nfev += result.nfev
        max_res = float(np.max(np.abs(result.fun)))
        if max_res < tolerance:
            return (result.x, "homotopy continuation (dogbox)", total_nfev)
    except Exception:
        pass

    return None


def _structural_incidence(
    residual_fn: Callable, x_start: "numpy.ndarray", n: int, np,
) -> "Optional[numpy.ndarray]":
    """Estimate which variable appears in which equation.

    Returns an ``n x n`` boolean matrix where entry ``(i, j)`` is True when
    equation ``i`` depends on variable ``j``, or None if the pattern could not
    be estimated reliably.  The pattern is the union of nonzero finite-
    difference Jacobian entries sampled at several points, so a derivative that
    happens to vanish at one point (e.g. ``x**2`` at ``x = 0``) is still picked
    up elsewhere.  This is the structural incidence Dynare reads symbolically;
    sampling is a robust numerical proxy and any error is caught later by the
    full-residual check, so it can never manufacture a false solution.
    """
    rng = np.random.default_rng(0)
    samples = [np.asarray(x_start, dtype=float)]
    for _ in range(4):
        samples.append(rng.uniform(0.3, 1.7, size=n))

    pattern = np.zeros((n, n), dtype=bool)
    usable_samples = 0
    for x in samples:
        try:
            f0 = np.asarray(residual_fn(x), dtype=float)
        except Exception:
            continue
        if f0.shape != (n,) or not np.all(np.isfinite(f0)):
            continue
        step = 1e-6 * (np.abs(x) + 1.0)
        sample_ok = True
        contribution = np.zeros((n, n), dtype=bool)
        for j in range(n):
            xj = x.copy()
            xj[j] += step[j]
            try:
                fj = np.asarray(residual_fn(xj), dtype=float)
            except Exception:
                sample_ok = False
                break
            if fj.shape != (n,) or not np.all(np.isfinite(fj)):
                sample_ok = False
                break
            derivative = np.abs(fj - f0) / step[j]
            contribution[:, j] = derivative > 1e-7
        if sample_ok:
            pattern |= contribution
            usable_samples += 1

    if usable_samples == 0:
        return None
    return pattern


def _solve_block(
    block_residual: Callable, x0: "numpy.ndarray", tolerance: float, np,
) -> Optional[Tuple["numpy.ndarray", int]]:
    """Solve a single block of a block-triangular system on a tight budget.

    A correctly-ordered recursive block converges in a handful of iterations,
    so each solver gets only a small evaluation budget; if it does not converge
    quickly the decomposition is abandoned (the caller falls back to the full
    cascade).  Tries trust-region least squares first, then a hybrid Powell
    step.  Returns ``(x, nfev)`` or None.
    """
    from scipy.optimize import least_squares, root

    k = int(x0.size)
    budget = max(100, 40 * k)
    # Drive the block solve with tolerances much tighter than the acceptance
    # threshold: least_squares' relative stopping criteria otherwise halt a
    # small block while its residual is still well above ``tolerance``.
    inner = 1e-13

    # Try the given guess first, then a copy with (near-)zero components
    # nudged away from the origin: trust-region steps can degenerate when a
    # block starts at exactly zero or a denormal value.
    seeded = np.where(np.abs(x0) < 1e-8, 0.5, x0)
    starts = [x0]
    if not np.array_equal(seeded, x0):
        starts.append(seeded)

    for start in starts:
        try:
            result = least_squares(
                block_residual, start, method="trf",
                ftol=inner, xtol=inner, gtol=inner,
                max_nfev=budget,
                x_scale="jac",
            )
            if float(np.max(np.abs(result.fun))) < tolerance:
                return result.x, int(result.nfev)
        except Exception:
            pass

        try:
            result = root(
                block_residual, start, method="hybr", tol=inner,
                options={"maxfev": budget},
            )
            if getattr(result, "success", False) and \
                    float(np.max(np.abs(result.fun))) < tolerance:
                return result.x, int(getattr(result, "nfev", budget))
        except Exception:
            pass

    return None


def _block_structure(pattern, n: int, np):
    """Block-triangular (Dulmage-Mendelsohn) decomposition of an incidence.

    ``pattern`` is an n x n boolean matrix where ``pattern[i, j]`` is True when
    equation ``i`` depends on variable ``j``.  Returns ``(eq_of_var, blocks)``
    where ``blocks`` is a list of sorted variable-index lists in topological
    (dependencies-first) order, or None if the system is structurally singular
    (no perfect equation/variable matching).  An irreducible system yields a
    single block; callers decide what to do with that.
    """
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import (
            connected_components,
            maximum_bipartite_matching,
        )
    except Exception:
        return None

    if n == 0 or not pattern.any(axis=1).all() or not pattern.any(axis=0).all():
        return None

    incidence = csr_matrix(pattern.astype(np.int8))
    try:
        var_of_eq = maximum_bipartite_matching(incidence, perm_type="column")
    except Exception:
        return None
    if np.any(var_of_eq < 0):
        return None
    eq_of_var = np.empty(n, dtype=int)
    eq_of_var[var_of_eq] = np.arange(n)

    # Dependency digraph on variables: the variable matched to equation e
    # depends on every other variable appearing in e.  Edge v -> u.
    rows: List[int] = []
    cols: List[int] = []
    for eq in range(n):
        v = int(var_of_eq[eq])
        for u in range(n):
            if u != v and pattern[eq, u]:
                rows.append(v)
                cols.append(u)
    if rows:
        adjacency = csr_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)),
            shape=(n, n),
        )
    else:
        adjacency = csr_matrix((n, n), dtype=np.int8)

    n_blocks, labels = connected_components(
        adjacency, directed=True, connection="strong",
    )

    block_vars: List[List[int]] = [[] for _ in range(n_blocks)]
    for v in range(n):
        block_vars[labels[v]].append(v)
    block_deps: List[set] = [set() for _ in range(n_blocks)]
    for eq in range(n):
        cv = labels[var_of_eq[eq]]
        for u in range(n):
            if pattern[eq, u] and labels[u] != cv:
                block_deps[cv].add(int(labels[u]))

    order: List[int] = []
    solved_blocks: set = set()
    while len(order) < n_blocks:
        progressed = False
        for c in range(n_blocks):
            if c not in solved_blocks and block_deps[c] <= solved_blocks:
                order.append(c)
                solved_blocks.add(c)
                progressed = True
        if not progressed:
            return None  # unexpected cycle across blocks

    blocks = [sorted(block_vars[c]) for c in order]
    return eq_of_var, blocks


def _try_block_decomposition(
    residual_fn: Callable, x_start: "numpy.ndarray", n: int, tolerance: float,
    incidence: "Optional[numpy.ndarray]" = None,
) -> Optional[Tuple["numpy.ndarray", str, int]]:
    """Solve a recursive system block-by-block (Dulmage-Mendelsohn ordering).

    Mirrors Dynare's default ``solve_algo=4``: permute the system to
    block-triangular form (maximum bipartite matching of equations to
    variables, then strongly-connected components of the induced dependency
    graph) and solve the blocks in topological order, substituting each solved
    block forward.  Solving a recursive steady state one small block at a time
    is far more robust than throwing the whole nonlinear system at a general
    solver.

    ``incidence`` is the n x n boolean equation/variable structure; callers
    pass the pattern already computed during preprocessing (free), so the hot
    path does no extra residual evaluations.  When omitted it is estimated by
    sampling finite-difference Jacobians (used only by direct unit tests).

    Returns ``(x, method, nfev)`` if the full residual is within tolerance,
    otherwise None.  Returns None immediately when scipy is unavailable, the
    system is structurally singular, or it is irreducible (a single block) --
    in which case the caller's full-system cascade runs unchanged, so this
    routine can only ever help.
    """
    if n < 2:
        return None

    try:
        import numpy as np
    except Exception:
        return None

    if incidence is None:
        pattern = _structural_incidence(residual_fn, x_start, n, np)
    else:
        pattern = np.asarray(incidence, dtype=bool)
        if pattern.shape != (n, n):
            return None
    if pattern is None:
        return None

    structure = _block_structure(pattern, n, np)
    if structure is None:
        return None
    eq_of_var, blocks = structure
    if len(blocks) <= 1:
        return None  # irreducible -- the general cascade handles it

    x_current = np.asarray(x_start, dtype=float).copy()
    total_nfev = 0

    for vidx in blocks:
        eidx = [int(eq_of_var[v]) for v in vidx]

        def block_residual(x_block, _vidx=vidx, _eidx=eidx):
            trial = x_current.copy()
            trial[_vidx] = x_block
            return np.asarray(residual_fn(trial))[_eidx]

        solved = _solve_block(block_residual, x_current[vidx], tolerance, np)
        if solved is None:
            # A correctly-ordered recursive block solves within a small
            # budget; if it doesn't, the decomposition is not helping, so
            # bail and let the full-system cascade run instead.
            return None

        x_block_sol, nfev = solved
        x_current[vidx] = x_block_sol
        total_nfev += int(nfev)

    final = np.asarray(residual_fn(x_current))
    if np.all(np.isfinite(final)) and float(np.max(np.abs(final))) < tolerance:
        return (x_current, f"block decomposition ({len(blocks)} blocks)", total_nfev)
    return None


def _try_minimize_ssr(
    residual_fn: Callable, x0: "numpy.ndarray", tolerance: float,
    deadline: Optional[float] = None,
) -> Optional[Tuple["numpy.ndarray", str, int]]:
    """Minimise sum-of-squared-residuals using L-BFGS-B and Nelder-Mead.

    This is a last-resort approach that treats the problem as unconstrained
    optimisation rather than root-finding.
    """
    import numpy as np
    from scipy.optimize import minimize

    def ssr(x):
        r = residual_fn(x)
        return float(0.5 * np.sum(r ** 2))

    for method in ("L-BFGS-B", "Nelder-Mead", "Powell"):
        if _past_deadline(deadline):
            break
        try:
            opts: dict = {"maxiter": 100000}
            if method == "Nelder-Mead":
                opts["maxfev"] = 500000
                opts["xatol"] = tolerance
                opts["fatol"] = tolerance ** 2
            result = minimize(ssr, x0, method=method, options=opts)
            # Check if residuals are actually small enough
            r = residual_fn(result.x)
            max_res = float(np.max(np.abs(r)))
            if max_res < tolerance:
                niter = getattr(result, "nit", getattr(result, "nfev", 0))
                return (result.x, f"minimise SSR ({method})", niter)
        except Exception:
            pass
    return None


def _jacobian_rank_issue(
    residual_fn: Callable,
    x: "numpy.ndarray",
    n_vars: int,
    tolerance: float,
    deadline: Optional[float] = None,
) -> Optional[str]:
    """Return a message when a zero-residual solution is locally underdetermined.

    This builds an n_vars-column finite-difference Jacobian (O(n) residual
    evaluations) and takes its rank, so it is the dominant post-solve cost on
    large models.  ``deadline`` bounds it: once the solve's wall-clock budget is
    spent it stops and accepts the candidate (which already had a residual
    within tolerance) instead of overshooting.
    """
    import numpy as np

    if n_vars == 0:
        return None

    x = np.asarray(x, dtype=float)
    try:
        f0 = np.asarray(residual_fn(x), dtype=float)
    except Exception as exc:
        return f"could not validate residual Jacobian rank: {exc}"
    if f0.size == 0:
        return "residual Jacobian has no equations"

    jac = np.zeros((f0.size, n_vars), dtype=float)
    eps = np.sqrt(np.finfo(float).eps)
    for j in range(n_vars):
        if _past_deadline(deadline):
            return None  # out of budget: accept the candidate, skip the rank check
        h = eps * max(1.0, abs(float(x[j])))
        xp = x.copy()
        xm = x.copy()
        xp[j] += h
        xm[j] -= h
        try:
            fp = np.asarray(residual_fn(xp), dtype=float)
            fm = np.asarray(residual_fn(xm), dtype=float)
        except Exception as exc:
            return f"could not validate residual Jacobian rank: {exc}"
        if fp.shape != f0.shape or fm.shape != f0.shape:
            return "residual Jacobian changed dimension near the solution"
        if not (np.all(np.isfinite(fp)) and np.all(np.isfinite(fm))):
            return "residual Jacobian has non-finite entries near the solution"
        jac[:, j] = (fp - fm) / (2 * h)

    rank_tol = max(1e-8, tolerance * 0.1)
    rank = int(np.linalg.matrix_rank(jac, tol=rank_tol))
    if rank < n_vars:
        return (
            f"residual Jacobian rank {rank} is below {n_vars} endogenous "
            "variable(s); the steady state is underdetermined"
        )
    return None


_TIMED_BUILTIN_CONSTANT_RE = re.compile(
    r"\b(pi|inf|nan)\s*\(\s*[+-]?\s*\d+\s*\)",
    re.IGNORECASE,
)
_TIMED_IDENTIFIER_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*[+-]?\s*\d+\s*\)"
)


def _unresolved_timed_builtin_constant(model: ParsedModel) -> Optional[str]:
    declared = model.all_declared_names()
    for eq in model.model_equations:
        for match in _TIMED_BUILTIN_CONSTANT_RE.finditer(eq.text):
            name = match.group(1)
            if name not in declared:
                return name
    return None


def _timed_parameter_reference(model: ParsedModel) -> Optional[str]:
    parameter_names = {p.name for p in model.parameters}
    if not parameter_names:
        return None
    for eq in model.model_equations:
        text = re.sub(r"\[[^\]]*\]", "", eq.text)
        for match in _TIMED_IDENTIFIER_RE.finditer(text):
            name = match.group(1)
            if name in parameter_names:
                return name
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _identifier_names(text: str) -> set[str]:
    return set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text))


def _unassigned_parameters_used_by_static_model(
    model: ParsedModel,
    unassigned: List[str],
) -> List[str]:
    """Return unassigned parameters used directly or via model-local vars."""
    unassigned_set = set(unassigned)
    local_defs: Dict[str, str] = {}
    for eq in model.model_equations:
        text = eq.text.strip()
        if not text.startswith("#"):
            continue
        match = _LOCAL_VAR_DEF.match(text)
        if match:
            local_defs[match.group(1)] = match.group(2)

    used_params: set[str] = set()
    pending_locals: List[str] = []
    seen_locals: set[str] = set()

    def _scan(text: str) -> None:
        for name in _identifier_names(text):
            if name in unassigned_set:
                used_params.add(name)
            elif name in local_defs and name not in seen_locals:
                seen_locals.add(name)
                pending_locals.append(name)

    for eq in model.static_model_equations():
        _scan(eq.text)

    while pending_locals:
        local_name = pending_locals.pop()
        _scan(local_defs[local_name])

    return sorted(used_params)


def _model_local_shadowing_names(model: ParsedModel) -> List[str]:
    declared = model.all_declared_names()
    if not declared:
        return []

    names: List[str] = []
    seen: set[str] = set()
    for eq in list(model.model_equations) + list(model.steady_state_equations):
        match = _LOCAL_VAR_DEF.match(eq.text.strip())
        if not match:
            continue
        name = match.group(1)
        if name in declared and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def compute_steady_state(
    model: ParsedModel,
    tolerance: float = 1e-8,
    max_attempts: int = 10,
    user_initial_guess: Optional[Dict[str, float]] = None,
    warm_start_guess: Optional[Dict[str, float]] = None,
    time_budget: Optional[float] = None,
) -> SolverResult:
    """Compute steady state values for a parsed model.

    Tries multiple solver strategies in sequence until one succeeds.
    Returns a SolverResult with the computed values or failure info.

    ``time_budget`` optionally bounds the wall-clock seconds spent in the
    iterative solver cascade.  When set, the expensive stages (root finding,
    homotopy, SSR minimisation, random restarts) stop once the budget elapses
    and the result comes back with ``timed_out=True`` instead of grinding for
    minutes on a large or non-converging system.  ``None`` (the default) leaves
    the solve unbounded, preserving the original behaviour for existing callers.

    Requires scipy (optional dependency).
    """
    deadline = (
        time.monotonic() + time_budget
        if time_budget is not None and time_budget > 0
        else None
    )
    try:
        import numpy as np
        from scipy.optimize import root as _  # noqa: F401
    except ImportError:
        return SolverResult(
            success=False, values={}, residual_norm=float("inf"),
            method_used="", iterations=0,
            message="scipy is required for steady state computation. "
                    "Install with: pip install dynare-lsp[solver]",
        )

    # Validate preconditions
    if not model.static_model_equations():
        return SolverResult(
            success=False, values={}, residual_norm=float("inf"),
            method_used="", iterations=0,
            message="No model equations found.",
        )

    shadowing_names = _model_local_shadowing_names(model)
    if shadowing_names:
        name = shadowing_names[0]
        return SolverResult(
            success=False,
            values={},
            residual_norm=float("inf"),
            method_used="none",
            iterations=0,
            message=(
                f"Cannot solve: model-local variable '{name}' shadows a "
                "declared Dynare symbol. Rename the model-local variable first."
            ),
        )

    timed_constant = _unresolved_timed_builtin_constant(model)
    if timed_constant is not None:
        return SolverResult(
            success=False,
            values={},
            residual_norm=float("inf"),
            method_used="",
            iterations=0,
            message=(
                f"Timed identifier '{timed_constant}' is not declared; "
                "built-in constants cannot be used with leads or lags."
            ),
        )

    var_names = [v.name for v in model.endogenous]
    real_equations = model.static_model_equations()
    n_vars = len(var_names)
    n_eqs = len(real_equations)

    if n_vars != n_eqs:
        return SolverResult(
            success=False, values={}, residual_norm=float("inf"),
            method_used="", iterations=0,
            message=f"Cannot solve: {n_eqs} equations but {n_vars} unknowns.",
        )

    params = _effective_param_values(model)
    exogenous = model.exogenous_names()
    exogenous_values = _exogenous_values(model)

    # Refuse to solve if any declared parameter wasn't assigned a value.
    # Without this guard the solver runs with the parameter missing
    # from the eval env and returns a generic "convergence failed"
    # error — masking the actual issue (the user has W010 in the
    # diagnostics but the solver result they see says nothing about
    # the unassigned parameter).
    #
    # ``steady_state_model`` equations of the form ``beta = 0.9;`` are a
    # legitimate place to define a parameter; include those names in the
    # "assigned" set so the pre-check doesn't reject models that
    # diagnostics already accept.
    declared_params = {v.name for v in model.parameters}
    unassigned = sorted(declared_params - set(params))
    # Only complain about parameters that are actually USED in the
    # model equations; an unused unassigned parameter is harmless.
    if unassigned:
        offending = _unassigned_parameters_used_by_static_model(model, unassigned)
        if offending:
            return SolverResult(
                success=False, values={},
                residual_norm=float("nan"),
                method_used="none", iterations=0,
                message=(
                    "Cannot solve: parameter(s) declared but never "
                    f"assigned a value: {', '.join(offending)}. "
                    "Add a parameter assignment line like "
                    f"`{offending[0]} = <value>;` before the model block."
                ),
            )

    helper_values = {
        assignment.name: assignment.value
        for assignment in model.helper_assignments
        if assignment.value is not None
    }
    eval_params = dict(params)
    eval_params.update(helper_values)

    symbolic_steps: List[str] = []
    symbolic_solved_count = 0
    preferred_symbolic_values = _preferred_symbolic_values(
        model,
        var_names,
        eval_params,
        warm_start_guess,
        user_initial_guess,
    )
    try:
        if n_vars > _SYMBOLIC_MAX_VARS or _past_deadline(deadline):
            # Back off symbolic reduction on larger systems (see _SYMBOLIC_MAX_VARS):
            # it can grind for minutes, and the numeric cascade below solves these.
            raise _SkipSymbolicReduction
        from .symbolic import reduce_symbolically

        symbolic_result = reduce_symbolically(
            model,
            eval_params,
            preferred_values=preferred_symbolic_values,
            deadline=deadline,
        )
        symbolic_steps = list(symbolic_result.symbolic_steps)
        symbolic_solved_count = len(set(var_names) & set(symbolic_result.solved_values))
        if var_names and set(var_names).issubset(symbolic_result.solved_values):
            values = {
                name: float(symbolic_result.solved_values[name])
                for name in var_names
            }
            x_symbolic = np.asarray(
                [values[name] for name in var_names],
                dtype=float,
            )
            validation_residual_fn = _build_residual_function(
                model, var_names, eval_params, exogenous, exogenous_values,
            )
            residuals = validation_residual_fn(x_symbolic)
            residual_max = float(np.max(np.abs(residuals)))
            domain_issue = _detect_var_log_domain(model, values)
            if domain_issue is None:
                domain_issue = _detect_invalid_domain(
                    model.steady_state_check_equations(), var_names, values, eval_params,
                    exogenous, exogenous_values,
                )
            if (
                domain_issue is None
                and np.isfinite(residual_max)
                and residual_max <= max(1e-6, tolerance * 1000)
            ):
                return SolverResult(
                    success=True,
                    values=values,
                    residual_norm=residual_max,
                    method_used="symbolic",
                    iterations=0,
                    message="Solved symbolically.",
                    equation_residuals=residuals.tolist(),
                    initial_guess=values,
                    symbolic_steps=symbolic_steps,
                    n_symbolic=len(values),
                    n_numerical=0,
                )
    except _SkipSymbolicReduction:
        pass
    except Exception:
        logger.debug("Symbolic steady-state reduction failed", exc_info=True)

    symbolic_numerical_count = max(0, n_vars - symbolic_solved_count)

    # ------------------------------------------------------------------
    # Step 0: Preprocess equations (compile once)
    # ------------------------------------------------------------------
    equations, locals_list = _preprocess_model(model, var_names)

    # Refuse to solve if any equations were silently dropped during
    # preprocessing (malformed RHS/LHS, validation rejection, compile
    # errors).  Continuing with a shorter equation list yields a system
    # that nominally "solves" but doesn't represent the user's model —
    # the LLM benchmark would see a false positive.
    real_eqs = model.static_model_equations()
    if len(equations) < len(real_eqs):
        n_dropped = len(real_eqs) - len(equations)
        return SolverResult(
            success=False,
            values={},
            residual_norm=float("nan"),
            method_used="none",
            iterations=0,
            message=(
                f"Cannot solve: {n_dropped} of {len(real_eqs)} model "
                f"equation(s) could not be compiled (malformed LHS/RHS, "
                f"empty equation, or unsafe expression). Fix the model "
                f"first; running the solver on the truncated equation "
                f"set would produce misleading results."
            ),
        )

    solver_residual_fn = _build_solver_residual_fn(
        equations, locals_list, var_names, eval_params, exogenous, exogenous_values,
    )
    # Bound a single in-flight SciPy call by time: the cascade uses a residual
    # that raises once the budget elapses (max_nfev and between-stage checks
    # cannot interrupt one long least_squares/root call).  Keep the raw residual
    # for the Gauss-Seidel check and final-fallback reporting, which must not raise.
    _raw_solver_residual_fn = solver_residual_fn
    solver_residual_fn = _deadline_guarded(solver_residual_fn, deadline)

    # Also keep original residual fn for final validation
    validation_residual_fn = _build_residual_function(
        model, var_names, eval_params, exogenous, exogenous_values,
    )

    # ------------------------------------------------------------------
    # Step 1: Build initial guess
    # ------------------------------------------------------------------
    x0 = _build_initial_guess(model, var_names, params, warm_start_guess)
    if user_initial_guess:
        for name, val in user_initial_guess.items():
            if name in var_names:
                x0[var_names.index(name)] = val

    initial_guess_dict = {name: float(x0[i])
                          for i, name in enumerate(var_names)}

    # ------------------------------------------------------------------
    # Step 2: Gauss-Seidel pre-conditioning
    # ------------------------------------------------------------------
    x_gs = _gauss_seidel_improve(
        equations, locals_list, var_names, x0, eval_params,
        exogenous, exogenous_values,
        max_sweeps=15, tol=tolerance)

    # Check if Gauss-Seidel already solved it
    gs_residuals = _raw_solver_residual_fn(x_gs)
    gs_max = float(np.max(np.abs(gs_residuals)))
    if gs_max < tolerance:
        values = {name: float(x_gs[i]) for i, name in enumerate(var_names)}
        # Domain validation also applies on the Gauss-Seidel early-out
        # path — without this the solver claims success on solutions
        # that send ``log(y)`` arguments negative (silently clamped to
        # ``-1e10`` by ``_safe_log``).
        domain_issue = _detect_var_log_domain(model, values)
        if domain_issue is None:
            domain_issue = _detect_invalid_domain(
                model.steady_state_check_equations(), var_names, values, eval_params,
                exogenous, exogenous_values,
            )
        validation_residuals = validation_residual_fn(x_gs)
        validation_max = float(np.max(np.abs(validation_residuals)))
        # Use the compiled solver residual for the rank-Jacobian build: same
        # mathematical function as validation_residual_fn but ~200x faster per
        # call, so the O(n) finite-difference rank check actually COMPLETES
        # within the time budget instead of timing out and being skipped (which
        # silently accepted rank-deficient steady states on large models).
        rank_issue = _jacobian_rank_issue(
            _raw_solver_residual_fn,
            x_gs,
            n_vars,
            tolerance,
            deadline,
        )
        if (
            domain_issue is None
            and (not np.isfinite(validation_max)
                 or validation_max > max(1e-6, tolerance * 1000))
        ):
            return SolverResult(
                success=False,
                values=values,
                residual_norm=validation_max,
                method_used="Gauss-Seidel",
                iterations=0,
                message=(
                    "Converged using Gauss-Seidel but rejected: original "
                    f"model residual {validation_max:.2e} exceeds tolerance."
                ),
                equation_residuals=validation_residuals.tolist(),
                initial_guess=initial_guess_dict,
                symbolic_steps=symbolic_steps,
                n_symbolic=symbolic_solved_count,
                n_numerical=symbolic_numerical_count,
            )
        if domain_issue is None and rank_issue is None:
            # If the rank check was skipped because the deadline elapsed, we
            # cannot confirm the system is full-rank: return an inconclusive
            # timed-out result rather than a false success.
            if _past_deadline(deadline):
                return SolverResult(
                    success=False, values=values,
                    residual_norm=validation_max,
                    method_used="Gauss-Seidel",
                    iterations=0,
                    message=(
                        "Converged using Gauss-Seidel but rank check was skipped "
                        "(time budget elapsed); result is inconclusive."
                    ),
                    equation_residuals=validation_residuals.tolist(),
                    initial_guess=initial_guess_dict,
                    symbolic_steps=symbolic_steps,
                    n_symbolic=symbolic_solved_count,
                    n_numerical=symbolic_numerical_count,
                    timed_out=True,
                )
            return SolverResult(
                success=True, values=values,
                residual_norm=validation_max,
                method_used="Gauss-Seidel",
                iterations=0,
                message="Converged using Gauss-Seidel pre-conditioning alone.",
                equation_residuals=validation_residuals.tolist(),
                initial_guess=initial_guess_dict,
                symbolic_steps=symbolic_steps,
                n_symbolic=symbolic_solved_count,
                n_numerical=symbolic_numerical_count,
            )
        if domain_issue is None and rank_issue is not None:
            return SolverResult(
                success=False,
                values=values,
                residual_norm=gs_max,
                method_used="Gauss-Seidel",
                iterations=0,
                message=(
                    "Converged using Gauss-Seidel but rejected: "
                    f"{rank_issue}."
                ),
                equation_residuals=gs_residuals.tolist(),
                initial_guess=initial_guess_dict,
                symbolic_steps=symbolic_steps,
                n_symbolic=symbolic_solved_count,
                n_numerical=symbolic_numerical_count,
            )
        # Otherwise fall through and let the downstream solvers try.
        logger.info("Gauss-Seidel solution rejected: %s", domain_issue)

    # Use the improved guess for all subsequent solvers
    x_start = x_gs

    # Helper to package a successful result
    def _success(x_sol, method_name, niter, extra_msg=""):
        vals = {name: float(x_sol[i]) for i, name in enumerate(var_names)}
        r = validation_residual_fn(x_sol)
        # Domain validation: ``_safe_log`` and ``_safe_sqrt`` mask out-of-
        # domain values with ``-1e10`` or ``nan`` rather than raising.
        # Two equations both touching invalid domain can cancel to a
        # residual of 0 — the solver would otherwise claim success on
        # mathematically nonsensical roots (e.g. ``log(-1) = log(-2)``).
        # Reject the result if any solution component sits where the
        # original mathematical expression is undefined.
        domain_issue = _detect_var_log_domain(model, vals)
        if domain_issue is None:
            domain_issue = _detect_invalid_domain(
                model.steady_state_check_equations(), var_names, vals, eval_params,
                exogenous, exogenous_values,
            )
        if domain_issue is not None:
            return SolverResult(
                success=False, values=vals,
                residual_norm=float(np.max(np.abs(r))),
                method_used=method_name, iterations=niter,
                message=(
                    f"Converged using {method_name} but rejected: "
                    f"{domain_issue}"
                ),
                equation_residuals=r.tolist(),
                initial_guess=initial_guess_dict,
                symbolic_steps=symbolic_steps,
                n_symbolic=symbolic_solved_count,
                n_numerical=symbolic_numerical_count,
            )
        validation_max = float(np.max(np.abs(r)))
        if (
            not np.isfinite(validation_max)
            or validation_max > max(1e-6, tolerance * 1000)
        ):
            return SolverResult(
                success=False, values=vals,
                residual_norm=validation_max,
                method_used=method_name, iterations=niter,
                message=(
                    f"Converged using {method_name} but rejected: original "
                    f"model residual {validation_max:.2e} exceeds tolerance."
                ),
                equation_residuals=r.tolist(),
                initial_guess=initial_guess_dict,
                symbolic_steps=symbolic_steps,
                n_symbolic=symbolic_solved_count,
                n_numerical=symbolic_numerical_count,
            )
        # Compiled solver residual (see the GS rank-check site above) -- same
        # Jacobian, ~200x faster, so the rank check completes within budget.
        rank_issue = _jacobian_rank_issue(
            _raw_solver_residual_fn,
            x_sol,
            n_vars,
            tolerance,
            deadline,
        )
        if rank_issue is not None:
            return SolverResult(
                success=False, values=vals,
                residual_norm=float(np.max(np.abs(r))),
                method_used=method_name, iterations=niter,
                message=(
                    f"Converged using {method_name} but rejected: "
                    f"{rank_issue}."
                ),
                equation_residuals=r.tolist(),
                initial_guess=initial_guess_dict,
                symbolic_steps=symbolic_steps,
                n_symbolic=symbolic_solved_count,
                n_numerical=symbolic_numerical_count,
            )
        # If rank_issue is None because the deadline elapsed (not because the
        # system is genuinely full-rank), treat the result as inconclusive.
        if _past_deadline(deadline):
            return SolverResult(
                success=False, values=vals,
                residual_norm=float(np.max(np.abs(r))),
                method_used=method_name, iterations=niter,
                message=(
                    f"Converged using {method_name} but rank check was skipped "
                    "(time budget elapsed); result is inconclusive."
                ),
                equation_residuals=r.tolist(),
                initial_guess=initial_guess_dict,
                symbolic_steps=symbolic_steps,
                n_symbolic=symbolic_solved_count,
                n_numerical=symbolic_numerical_count,
                timed_out=True,
            )
        return SolverResult(
            success=True, values=vals,
            residual_norm=float(np.max(np.abs(r))),
            method_used=method_name, iterations=niter,
            message=f"Converged using {method_name}.{extra_msg}",
            equation_residuals=r.tolist(),
            initial_guess=initial_guess_dict,
            symbolic_steps=symbolic_steps,
            n_symbolic=symbolic_solved_count,
            n_numerical=symbolic_numerical_count,
        )

    # ------------------------------------------------------------------
    # Step 2.5: Dulmage-Mendelsohn block-triangular decomposition
    # ------------------------------------------------------------------
    # Solve recursive systems block-by-block (Dynare's default solve_algo=4)
    # before throwing the whole system at a general solver.  Returns None for
    # irreducible or structurally singular systems, leaving the cascade below
    # unchanged, and any block solution is still re-validated by ``_success``.
    # The equation/variable incidence was already computed during
    # preprocessing (``_PrepEq.var_indices``), so this adds no residual
    # evaluations on the hot path.
    if len(equations) == n_vars and not _past_deadline(deadline):
        block_incidence = np.zeros((n_vars, n_vars), dtype=bool)
        for i, eq in enumerate(equations):
            for j in eq.var_indices:
                if 0 <= j < n_vars:
                    block_incidence[i, j] = True
        try:
            result = _try_block_decomposition(
                solver_residual_fn, x_start, n_vars, tolerance,
                incidence=block_incidence,
            )
        except _SolveDeadlineExceeded:
            result = None
        if result is not None:
            return _success(*result)

    # ------------------------------------------------------------------
    # Step 3: least_squares (trust-region methods)
    # ------------------------------------------------------------------
    if not _past_deadline(deadline):
        result = _try_least_squares(
            solver_residual_fn, x_start, tolerance, deadline=deadline)
        if result is not None:
            return _success(*result)

    # ------------------------------------------------------------------
    # Step 4: scipy.optimize.root methods
    # ------------------------------------------------------------------
    # Order: LM first for large systems, hybr first for small
    if n_vars > 15:
        methods = [m for m in _ROOT_METHODS if m[0] == "lm"] + \
                  [m for m in _ROOT_METHODS if m[0] != "lm"]
    else:
        methods = list(_ROOT_METHODS)

    for method, description in methods:
        if _past_deadline(deadline):
            break
        res = _try_root(solver_residual_fn, x_start, method, tolerance)
        if res is not None:
            return _success(*res)

    # ------------------------------------------------------------------
    # Step 5: Homotopy continuation
    # ------------------------------------------------------------------
    if not _past_deadline(deadline):
        result = _try_homotopy(
            solver_residual_fn, x_start, tolerance, deadline=deadline)
        if result is not None:
            return _success(*result)

    # ------------------------------------------------------------------
    # Step 6: Minimise SSR  (L-BFGS-B, Nelder-Mead, Powell)
    # ------------------------------------------------------------------
    if not _past_deadline(deadline):
        result = _try_minimize_ssr(
            solver_residual_fn, x_start, tolerance, deadline=deadline)
        if result is not None:
            return _success(*result)

    # ------------------------------------------------------------------
    # Step 7: Random restarts with least_squares
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    for attempt in range(max_attempts):
        if _past_deadline(deadline):
            break
        perturbation = rng.normal(0, 0.05 * (1 + attempt), size=n_vars)
        x_perturbed = x_start * (1 + perturbation)
        # Avoid exact zeros
        x_perturbed = np.where(np.abs(x_perturbed) < 1e-10,
                               0.01, x_perturbed)

        # Gauss-Seidel on perturbed guess
        x_perturbed = _gauss_seidel_improve(
            equations, locals_list, var_names, x_perturbed,
            eval_params, exogenous, exogenous_values,
            max_sweeps=5, tol=tolerance)

        result = _try_least_squares(
            solver_residual_fn, x_perturbed, tolerance, deadline=deadline)
        if result is not None:
            return _success(*result, f" (restart #{attempt + 1})")

        # Also try root methods on perturbed
        for method, description in methods[:2]:
            res = _try_root(
                solver_residual_fn, x_perturbed, method, tolerance)
            if res is not None:
                return _success(*res)

    # ------------------------------------------------------------------
    # Step 8: Homotopy with random restarts
    # ------------------------------------------------------------------
    for attempt in range(3):
        if _past_deadline(deadline):
            break
        perturbation = rng.normal(0, 0.02, size=n_vars)
        x_perturbed = x_start * (1 + perturbation)
        x_perturbed = np.where(np.abs(x_perturbed) < 1e-10,
                               0.01, x_perturbed)
        result = _try_homotopy(
            solver_residual_fn, x_perturbed, tolerance, n_steps=100,
            deadline=deadline)
        if result is not None:
            return _success(*result, f" (restart #{attempt + 1})")

    # ------------------------------------------------------------------
    # All strategies failed (or the time budget elapsed mid-cascade)
    # ------------------------------------------------------------------
    # Return best result found (lowest max residual)
    final_residuals = _raw_solver_residual_fn(x_start).tolist()
    budget_hit = _past_deadline(deadline)
    return SolverResult(
        success=False,
        values={name: float(x_start[i])
                for i, name in enumerate(var_names)},
        residual_norm=float(np.max(np.abs(np.array(final_residuals)))),
        method_used="none", iterations=0,
        message=(
            f"Steady-state solve stopped after its {time_budget:g}s time "
            "budget elapsed before convergence."
            if budget_hit else
            "All solver strategies failed to converge. "
            "Try providing better initial values in an initval block."
        ),
        equation_residuals=final_residuals,
        initial_guess=initial_guess_dict,
        symbolic_steps=symbolic_steps,
        n_symbolic=symbolic_solved_count,
        n_numerical=symbolic_numerical_count,
        timed_out=budget_hit,
    )
