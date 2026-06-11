"""Hybrid symbolic-numerical steady state solver for Dynare models.

Uses sympy to analytically solve as many steady state variables as possible,
then hands the remaining coupled system to scipy for numerical solving.

Algorithm:
  1. Convert Dynare equations to sympy expressions (steady state form)
  2. Pattern detection: AR(1), growth rates, expectations, exp inversions
  3. Iterative forward substitution: solve single-unknown equations
  4. SCC block solving: solve small coupled blocks with sympy.solve
  5. Build analytic Jacobian for the remaining numerical system

Requires sympy (optional dependency). Install with:
  pip install dynare-lsp[solver]
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .parser import ParsedModel
from .steady_state import (
    _LOCAL_VAR_DEF,
    _exogenous_values,
    _prepare_ss_expression,
    _split_equation,
)

try:
    import sympy as _sympy
    from sympy.parsing.sympy_parser import parse_expr as _parse_expr

    sympy: Any = _sympy
    Symbol: Any = getattr(_sympy, "Symbol")
    parse_expr: Any = _parse_expr

    _HAS_SYMPY = True
except ImportError:
    sympy = None
    Symbol = None
    parse_expr = None
    _HAS_SYMPY = False

_SAFE_SYMPY_GLOBALS: Dict[str, object] = {}

# Mapping of Dynare math functions to sympy equivalents
_DYNARE_TO_SYMPY_FUNCS: Dict[str, object] = {}
_DYNARE_TO_SYMPY_CONSTANTS: Dict[str, object] = {}

if _HAS_SYMPY:
    _SAFE_SYMPY_GLOBALS = {
        "__builtins__": {},
        "Float": sympy.Float,
        "Integer": sympy.Integer,
        "Rational": sympy.Rational,
        "Symbol": sympy.Symbol,
    }
    _DYNARE_TO_SYMPY_FUNCS = {
        "exp": sympy.exp,
        "log": sympy.log,
        "ln": sympy.log,
        "log2": lambda x: sympy.log(x, 2),
        "log10": lambda x: sympy.log(x, 10),
        "sqrt": sympy.sqrt,
        "cbrt": lambda x: x ** sympy.Rational(1, 3),
        "abs": sympy.Abs,
        "sign": sympy.sign,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
        "asin": sympy.asin,
        "acos": sympy.acos,
        "atan": sympy.atan,
        "sinh": sympy.sinh,
        "cosh": sympy.cosh,
        "tanh": sympy.tanh,
        "asinh": sympy.asinh,
        "acosh": sympy.acosh,
        "atanh": sympy.atanh,
        "min": sympy.Min,
        "max": sympy.Max,
        "erf": sympy.erf,
        "erfc": sympy.erfc,
    }
    _DYNARE_TO_SYMPY_CONSTANTS = {
        "pi": sympy.pi,
        "inf": sympy.oo,
    }


@dataclass
class SymbolicReductionResult:
    """Result of attempting symbolic reduction of the steady state system."""

    solved_vars: Dict[str, object]  # var_name -> sympy.Expr
    solved_values: Dict[str, float]  # var_name -> numerical value
    unsolved_var_names: List[str]  # truly independent unknowns for numerical solver
    unsolved_equations: List[object]  # residual sympy expressions (== 0)
    jacobian_fn: Optional[Callable]  # lambdified Jacobian or None
    scc_blocks: List[List[str]]  # strongly connected components
    symbolic_steps: List[str]  # human-readable log
    # Variables eliminated symbolically but not evaluable to numbers
    # (they depend on unsolved vars). Maps name -> sympy.Expr in terms
    # of unsolved vars only (chains resolved).
    eliminated_exprs: Dict[str, object] = field(default_factory=dict)
    # Callable: fn(x_unsolved_array) -> dict{name: float} for all eliminated vars
    elimination_fn: Optional[Callable] = None


# ---------------------------------------------------------------------------
# Expression conversion: Dynare -> sympy
# ---------------------------------------------------------------------------


def _dynare_expr_to_sympy(
    expr_text: str,
    var_symbols: Dict[str, Any],
    param_symbols: Dict[str, Any],
    local_var_exprs: Dict[str, Any],
) -> Any:
    """Convert a Dynare steady-state expression string to a sympy expression.

    The input should already have time subscripts removed and ``^`` converted
    to ``**`` via ``_prepare_ss_expression()``.  Python keywords that appear
    as Dynare identifiers (``lambda`` is the canonical example, common as a
    Lagrange multiplier or wage markup) would otherwise crash ``parse_expr``
    and the whole equation would be silently dropped.  Escape them on both
    sides — the expression text and the local-dict keys — using the same
    ``_escape_reserved`` rule that the numeric solver applies.
    """
    from .solver import _escape_reserved

    # Build local_dict: maps identifier strings to sympy objects
    local_dict: Dict[str, object] = {}

    # Add functions first
    local_dict.update(_DYNARE_TO_SYMPY_FUNCS)
    local_dict.update(_DYNARE_TO_SYMPY_CONSTANTS)

    # Add parameter / variable / local symbols under their escaped keys.
    local_dict.update({_escape_reserved(k): v for k, v in param_symbols.items()})
    local_dict.update({_escape_reserved(k): v for k, v in var_symbols.items()})
    local_dict.update({_escape_reserved(k): v for k, v in local_var_exprs.items()})

    # Escape Python-reserved identifiers in the expression text itself.
    import re as _re
    escape_names = (
        set(param_symbols) | set(var_symbols) | set(local_var_exprs)
    )
    for name in escape_names:
        esc = _escape_reserved(name)
        if esc != name:
            expr_text = _re.sub(
                r"\b" + _re.escape(name) + r"\b", esc, expr_text,
            )
    if _re.search(r"""['"]""", expr_text):
        raise ValueError("String literals are not supported in symbolic expressions")

    import ast as _ast

    try:
        tree = _ast.parse(expr_text, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid symbolic expression syntax") from exc

    allowed_calls = set(_DYNARE_TO_SYMPY_FUNCS)
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.Attribute, _ast.Subscript)):
            raise ValueError(
                "Attribute and subscript access are not supported in symbolic expressions"
            )
        if isinstance(
            node,
            (
                _ast.List,
                _ast.Tuple,
                _ast.Dict,
                _ast.Set,
                _ast.ListComp,
                _ast.SetComp,
                _ast.DictComp,
                _ast.GeneratorExp,
                _ast.Lambda,
            ),
        ):
            raise ValueError("Container expressions are not supported symbolically")
        if isinstance(node, _ast.Constant) and isinstance(node.value, str):
            raise ValueError("String literals are not supported in symbolic expressions")
        if isinstance(node, _ast.Name) and node.id.startswith("__"):
            raise ValueError("Private Python names are not supported symbolically")
        if isinstance(node, _ast.Call):
            if not isinstance(node.func, _ast.Name) or node.func.id not in allowed_calls:
                raise ValueError(
                    "Only Dynare math functions are supported in symbolic expressions"
                )

    # parse_expr handles ** correctly and respects local_dict.  Keep the
    # global namespace restricted so unsupported calls such as __import__()
    # fail before any Python builtins can be reached.
    return parse_expr(
        expr_text,
        local_dict=local_dict,
        global_dict=_SAFE_SYMPY_GLOBALS,
    )


def _prepare_sympy_system(
    model: ParsedModel,
    param_values: Dict[str, float],
) -> Tuple[
    List[Any],  # residual expressions (LHS - RHS = 0)
    Dict[str, Any],  # var_name -> Symbol
    Dict[str, Any],  # param_name -> Symbol
    List[str],  # equation texts for logging
]:
    """Convert all model equations to sympy residuals.

    Returns the system as a list of expressions that should equal zero,
    along with the symbol dictionaries.
    """
    var_names = [v.name for v in model.endogenous]
    exo_names = model.exogenous_names()
    exo_values = _exogenous_values(model)

    # Create sympy symbols for all variables and parameters
    var_symbols = {name: Symbol(name) for name in var_names}
    param_symbols = {name: Symbol(name) for name in param_values}

    # Substitute parameter numerical values
    param_subs = {param_symbols[k]: sympy.Rational(v).limit_denominator(10**12)
                  if v == int(v) else sympy.Float(v, 15)
                  for k, v in param_values.items()}

    # Process local variable definitions first
    local_var_exprs: Dict[str, Any] = {}
    model_equations_text: List[str] = []
    equation_texts: List[str] = []

    for eq in model.model_equations:
        text = eq.text.strip()
        if text.startswith("#"):
            match = _LOCAL_VAR_DEF.match(text)
            if match:
                name = match.group(1)
                expr_str = _prepare_ss_expression(match.group(2).strip())
                try:
                    expr = _dynare_expr_to_sympy(
                        expr_str, var_symbols, param_symbols, local_var_exprs
                    )
                    # Substitute parameter values
                    expr = expr.subs(param_subs)
                    # Set exogenous to its deterministic steady-state value.
                    for exo in exo_names:
                        exo_value = exo_values.get(exo, 0.0)
                        if exo in var_symbols:
                            expr = expr.subs(var_symbols[exo], exo_value)
                        elif Symbol(exo) in expr.free_symbols:
                            expr = expr.subs(Symbol(exo), exo_value)
                    local_var_exprs[name] = expr
                except Exception:
                    pass  # Skip unparseable local vars
            continue
        if "dynamic" in eq.tags:
            continue
        model_equations_text.append(text)

    # Convert each real equation to a sympy residual (LHS - RHS = 0)
    residuals: List[Any] = []

    for text in model_equations_text:
        ss_text = _prepare_ss_expression(text)
        try:
            lhs_str, rhs_str = _split_equation(ss_text)
        except ValueError:
            # Bare expression: should be zero
            try:
                expr = _dynare_expr_to_sympy(
                    ss_text, var_symbols, param_symbols, local_var_exprs
                )
                expr = expr.subs(param_subs)
                for exo in exo_names:
                    exo_value = exo_values.get(exo, 0.0)
                    if exo in var_symbols:
                        expr = expr.subs(var_symbols[exo], exo_value)
                    elif Symbol(exo) in expr.free_symbols:
                        expr = expr.subs(Symbol(exo), exo_value)
                residuals.append(expr)
                equation_texts.append(ss_text)
            except Exception:
                pass
            continue

        try:
            lhs = _dynare_expr_to_sympy(
                lhs_str, var_symbols, param_symbols, local_var_exprs
            )
            rhs = _dynare_expr_to_sympy(
                rhs_str, var_symbols, param_symbols, local_var_exprs
            )
            residual = lhs - rhs
            # Substitute parameter values
            residual = residual.subs(param_subs)
            # Set exogenous to its deterministic steady-state value.
            for exo in exo_names:
                exo_value = exo_values.get(exo, 0.0)
                if exo in var_symbols:
                    residual = residual.subs(var_symbols[exo], exo_value)
                elif Symbol(exo) in residual.free_symbols:
                    residual = residual.subs(Symbol(exo), exo_value)
            residuals.append(residual)
            equation_texts.append(f"{lhs_str} = {rhs_str}")
        except Exception:
            pass  # Skip unparseable equations

    return residuals, var_symbols, param_symbols, equation_texts


# ---------------------------------------------------------------------------
# Dependency graph and SCC detection
# ---------------------------------------------------------------------------


def _build_dependency_graph(
    equations: List[Any],
    var_symbols: Dict[str, Any],
) -> Dict[str, Set[str]]:
    """Build a variable-to-variable dependency graph.

    For each equation, finds which endogenous variables appear.
    Returns adjacency dict: var -> set of vars it co-occurs with.
    """
    sym_to_name = {s: n for n, s in var_symbols.items()}
    all_syms = set(var_symbols.values())

    # Which vars appear in each equation?
    eq_vars: List[Set[str]] = []
    for eq in equations:
        vs = eq.free_symbols & all_syms
        eq_vars.append({sym_to_name[s] for s in vs})

    # Build adjacency: for each var, what other vars does it co-occur with?
    adj: Dict[str, Set[str]] = {n: set() for n in var_symbols}
    for vs in eq_vars:
        for v in vs:
            adj[v] |= vs - {v}

    return adj


def _tarjan_scc(adj: Dict[str, Set[str]]) -> List[List[str]]:
    """Compute strongly connected components using iterative Tarjan's algorithm.

    Returns SCCs in reverse topological order (leaves first).
    """
    index_counter = [0]
    stack: List[str] = []
    on_stack: Set[str] = set()
    index: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    result: List[List[str]] = []

    def strongconnect(v: str) -> None:
        # Iterative version using explicit call stack
        call_stack: List[Tuple[str, list, int]] = []
        call_stack.append((v, list(adj.get(v, set())), 0))
        index[v] = lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        while call_stack:
            node, neighbors, ni = call_stack[-1]

            if ni < len(neighbors):
                call_stack[-1] = (node, neighbors, ni + 1)
                w = neighbors[ni]
                if w not in index:
                    index[w] = lowlink[w] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(w)
                    on_stack.add(w)
                    call_stack.append((w, list(adj.get(w, set())), 0))
                elif w in on_stack:
                    lowlink[node] = min(lowlink[node], index[w])
            else:
                # Done with this node
                if lowlink[node] == index[node]:
                    scc: List[str] = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        scc.append(w)
                        if w == node:
                            break
                    result.append(scc)

                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[node])

    for v in adj:
        if v not in index:
            strongconnect(v)

    return result


def _topological_sort_sccs(
    sccs: List[List[str]],
    adj: Dict[str, Set[str]],
) -> List[List[str]]:
    """Sort SCCs in dependency order (solvable blocks first).

    An SCC A should come before SCC B if B depends on A.
    """
    # Map each var to its SCC index
    var_to_scc: Dict[str, int] = {}
    for i, scc in enumerate(sccs):
        for v in scc:
            var_to_scc[v] = i

    # Build SCC-level DAG
    n = len(sccs)
    scc_adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for i, scc in enumerate(sccs):
        for v in scc:
            for w in adj.get(v, set()):
                j = var_to_scc.get(w, i)
                if j != i:
                    scc_adj[i].add(j)

    # Topological sort via Kahn's algorithm
    in_degree = {i: 0 for i in range(n)}
    for i in range(n):
        for j in scc_adj[i]:
            in_degree[j] += 1

    queue = deque(i for i in range(n) if in_degree[i] == 0)
    order: List[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for j in scc_adj[node]:
            in_degree[j] -= 1
            if in_degree[j] == 0:
                queue.append(j)

    # If there are cycles at SCC level (shouldn't happen), append remaining
    for i in range(n):
        if i not in order:
            order.append(i)

    return [sccs[i] for i in order]


# ---------------------------------------------------------------------------
# Solution selection
# ---------------------------------------------------------------------------


def _select_best_solution(
    solutions: list,
    var_sym: Any,
    known_values: Dict[Any, float],
    preferred_value: Optional[float] = None,
) -> Optional[Any]:
    """Select the most plausible solution from multiple candidates.

    Prefers: real > complex, evaluable > symbolic, closer to 1.0.
    """
    if not solutions:
        return None
    if len(solutions) == 1:
        return solutions[0]

    scored: List[Tuple[float, Any]] = []
    for sol in solutions:
        try:
            val = complex(sol.subs(known_values))
            if abs(val.imag) > 1e-10:
                continue  # skip complex solutions
            real_val = val.real
            if preferred_value is not None:
                score = abs(real_val - preferred_value)
            else:
                # Prefer values near 1 (typical SS), penalize very large/small
                score = abs(real_val - 1.0) if real_val > 0 else abs(real_val) + 10
            scored.append((score, sol))
        except (TypeError, ValueError, AttributeError):
            # Can't evaluate numerically; accept if it's the only option
            scored.append((100.0, sol))

    if not scored:
        return solutions[0]  # fallback

    scored.sort(key=lambda x: x[0])
    return scored[0][1]


# ---------------------------------------------------------------------------
# Core symbolic reduction
# ---------------------------------------------------------------------------


def _past(deadline: Optional[float]) -> bool:
    """True once a monotonic wall-clock deadline has passed (None = no limit)."""
    return deadline is not None and time.monotonic() >= deadline


def reduce_symbolically(
    model: ParsedModel,
    param_values: Optional[Dict[str, float]] = None,
    preferred_values: Optional[Dict[str, float]] = None,
    deadline: Optional[float] = None,
) -> SymbolicReductionResult:
    """Attempt to symbolically reduce the steady state system.

    Returns a SymbolicReductionResult with solved variables, unsolved
    equations, and optionally an analytic Jacobian for the numerical solver.
    """
    if not _HAS_SYMPY:
        return SymbolicReductionResult(
            solved_vars={},
            solved_values={},
            unsolved_var_names=[v.name for v in model.endogenous],
            unsolved_equations=[],
            jacobian_fn=None,
            scc_blocks=[],
            symbolic_steps=["sympy not available; skipping symbolic reduction"],
        )

    if param_values is None:
        from .solver import _effective_param_values
        param_values = _effective_param_values(model)

    var_names = [v.name for v in model.endogenous]
    steps: List[str] = []

    # Phase 1: Convert equations to sympy
    try:
        residuals, var_symbols, param_syms, eq_texts = _prepare_sympy_system(
            model, param_values
        )
    except Exception as e:
        steps.append(f"Failed to convert equations to sympy: {e}")
        return SymbolicReductionResult(
            solved_vars={},
            solved_values={},
            unsolved_var_names=var_names,
            unsolved_equations=[],
            jacobian_fn=None,
            scc_blocks=[],
            symbolic_steps=steps,
        )

    if not residuals:
        steps.append("No equations could be converted to sympy")
        return SymbolicReductionResult(
            solved_vars={},
            solved_values={},
            unsolved_var_names=var_names,
            unsolved_equations=[],
            jacobian_fn=None,
            scc_blocks=[],
            symbolic_steps=steps,
        )

    steps.append(f"Converted {len(residuals)}/{len(var_names)} equations to sympy")

    all_var_syms = set(var_symbols.values())
    sym_to_name = {s: n for n, s in var_symbols.items()}

    solved: Dict[str, Any] = {}  # var_name -> sympy expression
    remaining_eqs = list(residuals)
    remaining_eq_texts = list(eq_texts) if len(eq_texts) == len(residuals) else [
        str(r) for r in residuals
    ]
    remaining_vars = set(var_names)

    # Phase 2 + 3: Iterative forward substitution
    # (Pattern detection is handled naturally by sympy simplification +
    #  single-var solving. e.g., `c - c + ng` simplifies to `ng` automatically)

    max_iterations = len(var_names) + 5  # safety bound
    iteration = 0

    while iteration < max_iterations:
        if _past(deadline):
            break
        iteration += 1
        changed = False

        i = 0
        while i < len(remaining_eqs):
            if _past(deadline):
                break
            eq = remaining_eqs[i]

            # Substitute all solved vars
            subs_dict = {var_symbols[n]: v for n, v in solved.items()
                         if n in var_symbols}
            eq_sub = eq.subs(subs_dict)

            # Note: avoid sympy.simplify/expand -- too slow for DSGE models
            # with exp/log. The substitution + free_symbols check is enough.

            # What unsolved vars remain?
            free = eq_sub.free_symbols & all_var_syms
            unsolved_in_eq = {sym_to_name[s] for s in free
                              if sym_to_name.get(s) in remaining_vars}

            if len(unsolved_in_eq) == 0:
                # Equation is fully determined -- check if it's satisfied
                try:
                    val = complex(eq_sub.evalf())
                    if abs(val) < 1e-8:
                        remaining_eqs.pop(i)
                        if i < len(remaining_eq_texts):
                            remaining_eq_texts.pop(i)
                        changed = True
                        continue
                except (TypeError, ValueError):
                    pass
                # If not satisfied or can't evaluate, skip
                i += 1

            elif len(unsolved_in_eq) == 1:
                # Single unknown -- try to solve
                var_name = next(iter(unsolved_in_eq))
                var_sym = var_symbols[var_name]

                try:
                    solutions = sympy.solve(eq_sub, var_sym)
                    if solutions:
                        # Build known values for solution selection
                        known = {}
                        for n, v in solved.items():
                            try:
                                known[var_symbols[n]] = float(v.evalf())
                            except (TypeError, ValueError, AttributeError):
                                pass

                        preferred = (
                            preferred_values.get(var_name)
                            if preferred_values is not None else None
                        )
                        sol = _select_best_solution(
                            solutions,
                            var_sym,
                            known,
                            preferred,
                        )
                        if sol is not None:
                            solved[var_name] = sol
                            remaining_vars.discard(var_name)
                            remaining_eqs.pop(i)
                            if i < len(remaining_eq_texts):
                                eq_label = remaining_eq_texts.pop(i)
                            else:
                                eq_label = str(eq_sub)[:60]
                            steps.append(
                                f"Solved {var_name} = {sol} "
                                f"(from: {eq_label[:80]})"
                            )
                            changed = True
                            continue
                except Exception:
                    pass
                i += 1
            else:
                i += 1

        if not changed:
            break

    # Phase 3b: Variable elimination -- solve for variables that appear
    # linearly in multi-variable equations (express one var in terms of others)
    if remaining_vars and remaining_eqs:
        elim_changed = True
        while elim_changed:
            if _past(deadline):
                break
            elim_changed = False
            i = 0
            while i < len(remaining_eqs):
                if _past(deadline):
                    break
                eq = remaining_eqs[i]
                subs_dict = {var_symbols[n]: v for n, v in solved.items()
                             if n in var_symbols}
                eq_sub = eq.subs(subs_dict)

                free = eq_sub.free_symbols & all_var_syms
                unsolved_in_eq = {sym_to_name[s] for s in free
                                  if sym_to_name.get(s) in remaining_vars}

                if len(unsolved_in_eq) == 0:
                    # Fully determined after substitution
                    try:
                        val = complex(eq_sub.evalf())
                        if abs(val) < 1e-8:
                            remaining_eqs.pop(i)
                            if i < len(remaining_eq_texts):
                                remaining_eq_texts.pop(i)
                            elim_changed = True
                            continue
                    except (TypeError, ValueError):
                        pass
                    i += 1
                elif len(unsolved_in_eq) == 1:
                    # Single unknown after substitution
                    var_name = next(iter(unsolved_in_eq))
                    var_sym = var_symbols[var_name]
                    try:
                        solutions = sympy.solve(eq_sub, var_sym)
                        if solutions:
                            preferred = (
                                preferred_values.get(var_name)
                                if preferred_values is not None else None
                            )
                            sol = _select_best_solution(
                                solutions, var_sym,
                                {var_symbols[n]: v for n, v in solved.items()
                                 if n in var_symbols},
                                preferred,
                            )
                            if sol is not None:
                                solved[var_name] = sol
                                remaining_vars.discard(var_name)
                                remaining_eqs.pop(i)
                                if i < len(remaining_eq_texts):
                                    eq_label = remaining_eq_texts.pop(i)
                                else:
                                    eq_label = str(eq_sub)[:60]
                                steps.append(
                                    f"Solved {var_name} = {sol} "
                                    f"(elimination, from: {eq_label[:80]})"
                                )
                                elim_changed = True
                                continue
                    except Exception:
                        pass
                    i += 1
                elif len(unsolved_in_eq) >= 2:
                    # Try to isolate a variable that appears linearly.
                    # A variable is linear if d(eq)/d(var) is independent of var
                    # (i.e. diff is fast and structural, not algebraic).
                    found_linear = False
                    for vn in sorted(unsolved_in_eq):
                        vs = var_symbols[vn]
                        try:
                            d1 = sympy.diff(eq_sub, vs)
                            if vs in d1.free_symbols:
                                continue  # nonlinear in this var
                            # Also check d2 == 0 to confirm linearity
                            d2 = sympy.diff(d1, vs)
                            if d2 != 0:
                                continue
                            # var appears linearly: eq = d1*var + rest => var = -rest/d1
                            rest = eq_sub - d1 * vs
                            sol = -rest / d1
                            # Check for circularity: if var still appears in sol
                            # (e.g. log(exp(x)) not simplified), skip this var
                            if vs in sol.free_symbols:
                                continue
                            solved[vn] = sol
                            remaining_vars.discard(vn)
                            remaining_eqs.pop(i)
                            if i < len(remaining_eq_texts):
                                eq_label = remaining_eq_texts.pop(i)
                            else:
                                eq_label = str(eq_sub)[:60]
                            steps.append(
                                f"Eliminated {vn} = {sol} "
                                f"(linear in eq: {eq_label[:80]})"
                            )
                            elim_changed = True
                            found_linear = True
                            break
                        except Exception:
                            pass
                    if not found_linear:
                        i += 1
                    # If found_linear, the eq was popped; don't increment
                else:
                    i += 1

    # Phase 4: SCC block solving for remaining coupled equations
    if remaining_vars and remaining_eqs:
        # Substitute solved vars into remaining equations
        subs_dict = {var_symbols[n]: v for n, v in solved.items()
                     if n in var_symbols}
        remaining_eqs = [eq.subs(subs_dict) for eq in remaining_eqs]

        # Build dependency graph on remaining vars/equations
        remaining_var_syms = {n: var_symbols[n] for n in remaining_vars
                              if n in var_symbols}
        adj = _build_dependency_graph(remaining_eqs, remaining_var_syms)

        # Filter adjacency to only remaining vars
        adj = {k: v & remaining_vars for k, v in adj.items()
               if k in remaining_vars}

        sccs = _tarjan_scc(adj)
        sccs = _topological_sort_sccs(sccs, adj)

        for scc in sccs:
            if len(scc) == 0:
                continue

            # Find equations involving this SCC's variables
            scc_eqs = []
            scc_eq_indices = []

            for idx, eq in enumerate(remaining_eqs):
                eq_free = eq.free_symbols & all_var_syms
                eq_vars = {sym_to_name[s] for s in eq_free
                           if sym_to_name.get(s) in remaining_vars}
                if eq_vars & set(scc):
                    scc_eqs.append(eq)
                    scc_eq_indices.append(idx)

            # Only try singleton blocks -- multi-var sympy.solve can hang
            # on nonlinear DSGE equations with fractional exponents
            if len(scc) == 1 and len(scc_eqs) >= 1:
                try:
                    scc_var_syms = [var_symbols[n] for n in scc
                                    if n in var_symbols]
                    solve_eqs = scc_eqs[: len(scc)]
                    solutions = sympy.solve(solve_eqs, scc_var_syms, dict=True)

                    if solutions:
                        sol_dict = solutions[0]
                        all_solved_in_scc = True
                        for sym in scc_var_syms:
                            if sym not in sol_dict:
                                all_solved_in_scc = False
                                break

                        if all_solved_in_scc:
                            for sym in scc_var_syms:
                                name = sym_to_name[sym]
                                solved[name] = sol_dict[sym]
                                remaining_vars.discard(name)
                                steps.append(
                                    f"Solved {name} = {sol_dict[sym]} "
                                    f"(SCC block: {scc})"
                                )

                            for idx in sorted(scc_eq_indices, reverse=True):
                                if idx < len(remaining_eqs):
                                    remaining_eqs.pop(idx)
                                    if idx < len(remaining_eq_texts):
                                        remaining_eq_texts.pop(idx)

                except Exception:
                    pass

    # Phase 5: Numerical evaluation of symbolic solutions
    # Separate into: fully evaluated (number), eliminated (depends on unsolved)
    solved_values: Dict[str, float] = {}
    unevaluated: Dict[str, Any] = dict(solved)  # name -> sympy expr

    # First pass: iterative evaluation for chains that resolve to numbers
    for _ in range(len(unevaluated) + 1):
        progress = False
        eval_subs: Dict[Any, object] = {}
        for name, val in solved_values.items():
            eval_subs[var_symbols[name]] = sympy.Float(val, 15)
        for name, expr in unevaluated.items():
            eval_subs[var_symbols[name]] = expr

        still_unevaluated: Dict[str, Any] = {}
        for name, expr in unevaluated.items():
            try:
                val = expr.subs(eval_subs)
                free = val.free_symbols & all_var_syms
                remaining_free = {sym_to_name.get(s) for s in free} & remaining_vars
                if not remaining_free and not (val.free_symbols - set(eval_subs.keys())):
                    numeric = complex(val.evalf())
                    if abs(numeric.imag) < 1e-10:
                        solved_values[name] = float(numeric.real)
                        remaining_vars.discard(name)
                        progress = True
                        continue
                still_unevaluated[name] = val
            except (TypeError, ValueError, AttributeError):
                still_unevaluated[name] = expr

        unevaluated = still_unevaluated
        if not progress:
            break

    # Phase 5b: Resolve chains in unevaluated expressions
    # e.g. cg -> ng -> yg -> zg becomes cg -> zg, ng -> zg, yg -> zg
    # so each eliminated expr is in terms of truly unsolved vars only.
    eliminated_exprs: Dict[str, Any] = {}
    if unevaluated:
        # Iteratively substitute eliminated expressions into each other
        resolved = dict(unevaluated)
        for _ in range(len(resolved) + 1):
            changed = False
            for name, expr in list(resolved.items()):
                subs_dict = {
                    var_symbols[n]: resolved[n]
                    for n in resolved
                    if n != name and var_symbols.get(n) in expr.free_symbols
                }
                if subs_dict:
                    new_expr = expr.subs(subs_dict)
                    if new_expr != expr:
                        resolved[name] = new_expr
                        changed = True
            if not changed:
                break

        # Verify each resolved expression is in terms of truly unsolved vars only
        unsolved_syms = {var_symbols[n] for n in remaining_vars}
        for name, expr in resolved.items():
            expr_vars = expr.free_symbols & all_var_syms
            non_unsolved = expr_vars - unsolved_syms
            if non_unsolved:
                # Still depends on another eliminated var (shouldn't happen
                # after chain resolution, but be safe) -- treat as unsolved
                remaining_vars.add(name)
                steps.append(f"WARNING: {name} depends on unresolved vars, "
                             f"moving to numerical solver")
            else:
                eliminated_exprs[name] = expr
                steps.append(f"Eliminated (chain-resolved): {name} = {expr}")

    # Build lambdified elimination function
    # fn(x_unsolved_array) -> dict{name: float} for all eliminated vars
    elimination_fn = None
    if eliminated_exprs:
        unsolved_var_list = sorted(remaining_vars)
        unsolved_sym_list = [var_symbols[n] for n in unsolved_var_list
                             if n in var_symbols]
        try:
            elim_exprs_list = [(name, expr) for name, expr in
                               sorted(eliminated_exprs.items())]
            elim_fns_individual = []
            for name, expr in elim_exprs_list:
                fn = sympy.lambdify(unsolved_sym_list, expr, modules="numpy")
                elim_fns_individual.append((name, fn))

            def _elimination_fn(x_array):
                result = {}
                for ename, efn in elim_fns_individual:
                    try:
                        result[ename] = float(efn(*x_array))
                    except Exception:
                        result[ename] = 1.0  # fallback
                return result

            elimination_fn = _elimination_fn
        except Exception as e:
            steps.append(f"WARNING: Could not build elimination function: {e}")
            # Fall back: move eliminated vars to unsolved
            for name in eliminated_exprs:
                remaining_vars.add(name)
            eliminated_exprs = {}

    # Phase 6: Build Jacobian for unsolved system
    jacobian_fn = None
    unsolved_eq_list = []

    if remaining_vars and remaining_eqs:
        final_subs = {var_symbols[n]: sympy.Float(v, 15)
                      for n, v in solved_values.items()
                      if n in var_symbols}
        unsolved_eq_list = [eq.subs(final_subs) for eq in remaining_eqs]
        unsolved_var_list = sorted(remaining_vars)

        jacobian_fn = _build_jacobian(
            unsolved_eq_list,
            [var_symbols[n] for n in unsolved_var_list if n in var_symbols],
        )

    n_sym = len(solved_values)
    n_elim = len(eliminated_exprs)
    n_unsolved = len(remaining_vars)
    steps.append(f"Summary: {n_sym} solved symbolically, {n_elim} eliminated, "
                 f"{n_unsolved} remaining for numerical solver")

    return SymbolicReductionResult(
        solved_vars=dict(solved),
        solved_values=solved_values,
        unsolved_var_names=sorted(remaining_vars),
        unsolved_equations=unsolved_eq_list,
        jacobian_fn=jacobian_fn,
        scc_blocks=_tarjan_scc(
            _build_dependency_graph(remaining_eqs, {n: var_symbols[n]
                                                     for n in remaining_vars
                                                     if n in var_symbols})
        ) if remaining_vars and remaining_eqs else [],
        symbolic_steps=steps,
        eliminated_exprs=eliminated_exprs,
        elimination_fn=elimination_fn,
    )


# ---------------------------------------------------------------------------
# Symbolic Jacobian
# ---------------------------------------------------------------------------


def _build_jacobian(
    equations: List[Any],
    variables: List[Any],
) -> Optional[Callable]:
    """Build an analytic Jacobian function using sympy.lambdify.

    Returns a function f(x: ndarray) -> ndarray of shape (n_eq, n_var),
    or None if construction fails.
    """
    if not _HAS_SYMPY or not equations or not variables:
        return None

    try:
        import numpy as np

        n_eq = len(equations)
        n_var = len(variables)

        # Compute symbolic Jacobian matrix
        jac_exprs = []
        for eq in equations:
            row = [sympy.diff(eq, v) for v in variables]
            jac_exprs.append(row)

        # Lambdify to numpy function
        flat_exprs = [expr for row in jac_exprs for expr in row]
        flat_fn = sympy.lambdify(variables, flat_exprs, modules="numpy")

        def jacobian_fn(x):
            vals = flat_fn(*x)
            return np.array(vals, dtype=float).reshape(n_eq, n_var)

        return jacobian_fn
    except Exception:
        return None
