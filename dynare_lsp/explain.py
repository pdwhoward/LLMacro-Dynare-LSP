"""Diagnostic code documentation.

Maps each diagnostic code emitted by ``diagnostics.py`` to a longer-form
markdown explanation with examples. Consumed by:

  - The LSP server (hover / code action for "explain this diagnostic")
  - The ``dynare-lsp explain <CODE>`` CLI subcommand
  - LLM agents that want to look up what a code means without leaving the
    editing session

Keep entries terse and example-rich; this is meant to be read inside a
tooltip or a single CLI invocation, not as long-form documentation.
"""

from __future__ import annotations

from typing import Dict, Optional

# Each entry: short title + multi-paragraph markdown explanation.
# Markdown is the lingua franca of LSP clients (VS Code, Neovim) and is
# rendered cleanly in chat UIs by LLM clients as well.

_ENTRIES: Dict[str, Dict[str, str]] = {
    "E001": {
        "title": "Parse error",
        "body": (
            "The Dynare parser could not interpret the source. The diagnostic "
            "range points at the offending token or the nearest recoverable "
            "position.\n\n"
            "**Common causes**\n\n"
            "- Missing semicolon at the end of a declaration or equation\n"
            "- Unbalanced parentheses, braces, or block keywords\n"
            "- Malformed time subscript such as `y(1)` where `y(+1)` was meant\n"
            "- A reserved keyword used as an identifier\n\n"
            "**Fix**\n\n"
            "Inspect the line cited and the line immediately preceding it. "
            "Dynare's preprocessor frequently flags the *next* line after a "
            "missing semicolon."
        ),
    },
    "E010": {
        "title": "Equation count does not match endogenous variable count",
        "body": (
            "The number of equations inside the `model` block must equal the "
            "number of endogenous variables declared in the `var` block. The "
            "LSP catches this within milliseconds of editing, before Dynare "
            "is invoked.\n\n"
            "**Fix**\n\n"
            "- Add a missing equation, or remove a duplicate one\n"
            "- Declare the missing endogenous variable in `var`, or remove an "
            "  extra declaration\n"
            "- Check whether a commented-out equation was intended to be "
            "  active"
        ),
    },
    "E023": {
        "title": "Predetermined variable not declared endogenous",
        "body": (
            "A name listed in `predetermined_variables` must also be declared "
            "as an endogenous variable in the `var` block. Dynare requires the "
            "variable to exist before it can be marked predetermined.\n\n"
            "**Fix**\n\n"
            "- Add the variable to the `var` declaration, or\n"
            "- Remove it from `predetermined_variables` if it is not actually "
            "  endogenous"
        ),
    },
    "I071": {
        "title": "Blanchard-Kahn check skipped",
        "body": (
            "The clean-room Blanchard-Kahn check could not be performed on this "
            "model, so this is an INFORMATION note, **not** a `W071` "
            "Blanchard-Kahn violation. The model may still be perfectly valid; "
            "the LSP simply could not analyze it.\n\n"
            "**Common reasons a check is skipped**\n\n"
            "- `EXPECTATION(...)` operators, which need Dynare's "
            "auxiliary-variable transformation\n"
            "- Leads/lags of order >= 2 that require the same transformation\n"
            "- A singular or rank-deficient linearization\n"
            "- A steady state that could not be evaluated for linearization\n"
            "- An equation/endogenous count mismatch (see `E010`)\n"
            "- SciPy not installed\n\n"
            "**Fix**\n\n"
            "Nothing is required if the model is otherwise valid. Run it through "
            "Dynare to get the authoritative Blanchard-Kahn verdict."
        ),
    },
    "E020": {
        "title": "Undeclared identifier in model block",
        "body": (
            "An identifier appears in the `model` block but is not declared "
            "as a `var`, `varexo`, or `parameters` symbol. The diagnostic "
            "names the exact identifier and the equation it appears in.\n\n"
            "**Fix**\n\n"
            "- Add the identifier to the appropriate declaration block\n"
            "- Correct a typo (the LSP suggests close matches when available)\n"
            "- If the symbol is a local helper, define it in the parameter "
            "  section before use"
        ),
    },
    "E024": {
        "title": "Unsupported time subscript",
        "body": (
            "A deterministic exogenous variable is used with a lead or lag in "
            "a context where the LSP cannot safely interpret the dated value. "
            "Parameter leads/lags are accepted because Dynare treats them as "
            "fixed scalars.\n\n"
            "**Fix**\n\n"
            "- Remove the time subscript if the symbol is meant to be a fixed "
            "  scalar\n"
            "- If the dated quantity is state-dependent, model it as an "
            "  endogenous variable instead"
        ),
    },
    "E025": {
        "title": "Model-local variable shadows a declared symbol",
        "body": (
            "A model-local variable defined with `#` uses the same name as a "
            "declared `var`, `varexo`, or `parameters` symbol. That hides the "
            "declared steady-state value inside the model block and can make "
            "solver diagnostics misleading.\n\n"
            "**Fix**\n\n"
            "- Rename the model-local helper, for example `#y_local = ...`\n"
            "- Or remove the declaration if the name was intended to be only "
            "  a model-local helper"
        ),
    },
    "E030": {
        "title": "Duplicate declaration across blocks",
        "body": (
            "The same identifier is declared in more than one block "
            "(for example, in both `var` and `varexo`, or twice in "
            "`parameters`). Dynare requires each name to belong to exactly "
            "one symbol class.\n\n"
            "**Fix**\n\n"
            "Remove the duplicate declaration. If you intended two related "
            "but distinct symbols, rename one (the LSP's rename action "
            "propagates the change across the file)."
        ),
    },
    "E040": {
        "title": "Steady-state computation error",
        "body": (
            "The automatic steady-state solver raised an exception while "
            "evaluating the equilibrium conditions. The named variable is "
            "the one the solver was working on when the error occurred.\n\n"
            "**Common causes**\n\n"
            "- Division by zero at the candidate steady state\n"
            "- `log(.)` or `sqrt(.)` applied to a negative quantity\n"
            "- A parameter that has not yet been assigned a value\n\n"
            "**Fix**\n\n"
            "Verify the parameter calibration. Consider providing an "
            "explicit `steady_state_model` block or `initval` guess to "
            "steer the solver toward the correct basin."
        ),
    },
    "E050": {
        "title": "Duplicate equation",
        "body": (
            "Two equations inside the `model` block are textually identical. "
            "The diagnostic cites the line where the duplicate appears and "
            "the line where the first occurrence was found.\n\n"
            "**Fix**\n\n"
            "Remove the duplicate. Dynare would otherwise report an "
            "equation-count mismatch (E010) downstream."
        ),
    },
    "E051": {
        "title": "Contradictory equation (always false)",
        "body": (
            "An equation reduces to a tautological falsehood, for example "
            "`0 = 1`. The LSP detects this by symbolic simplification of "
            "constant-only equations.\n\n"
            "**Fix**\n\n"
            "Remove the equation, or restore a variable reference that was "
            "accidentally simplified away."
        ),
    },
    "E052": {
        "title": "Duplicate parameter assignment",
        "body": (
            "The same parameter is assigned a value more than once in the "
            "parameter section. Only the last assignment takes effect at "
            "runtime, so the earlier one is silently ignored — usually a "
            "bug.\n\n"
            "**Fix**\n\n"
            "Remove one of the assignments, or rename if two distinct "
            "parameters were intended."
        ),
    },
    "E053": {
        "title": "Stray equation outside model block",
        "body": (
            "A line that looks like a model equation appears outside the "
            "`model` ... `end;` block. Dynare's parser will reject this.\n\n"
            "**Fix**\n\n"
            "Move the equation inside the `model` block, or convert it to "
            "a parameter assignment if it belongs at the top level."
        ),
    },
    "E060": {
        "title": "Circular @#include detected",
        "body": (
            "Two or more files reach themselves through the chain of "
            "`@#include` directives. Dynare's macro preprocessor expands "
            "includes inline, so a cycle would either loop forever or — "
            "in the real preprocessor — be rejected with a hard error. "
            "The language server reports the cycle as a chain of file "
            "names: `a.mod -> b.mod -> a.mod`.\n\n"
            "**Common causes**\n\n"
            "- A submodel was refactored and now includes its parent\n"
            "- Two helper files cross-include each other for shared\n"
            "  parameters or steady-state definitions\n"
            "- A copy-paste mistake duplicated the include in the wrong\n"
            "  direction\n\n"
            "**Fix**\n\n"
            "Break the cycle by removing one `@#include` along the chain. "
            "If both files genuinely need a shared block, extract that "
            "block into a third file and have both parents include it."
        ),
    },
    "E061": {
        "title": "Cannot resolve @#include target",
        "body": (
            "An `@#include` directive names a file that the language "
            "server could not find. It looked in the directory of the "
            "including file first, then in each configured workspace "
            "search path, and ran out of candidates.\n\n"
            "**Common causes**\n\n"
            "- A typo in the filename\n"
            "- The included file lives in a directory that isn't on the "
            "  language server's search paths\n"
            "- The file was renamed or moved without updating the "
            "  directive\n\n"
            "**Fix**\n\n"
            "Correct the filename, add the missing file, or extend the "
            "search paths so the directory containing the include is "
            "visible to the LSP."
        ),
    },
    "E062": {
        "title": "Unmatched macro block",
        "body": (
            "A Dynare macro `@#if` block has no matching `@#endif`, or "
            "a `@#for` block has no matching `@#endfor`. Dynare's "
            "preprocessor expands these directives at build time and "
            "will reject unbalanced control flow.\n\n"
            "**Common causes**\n\n"
            "- A copy-paste deleted the closing directive\n"
            "- Mismatched closers — `@#endif` accidentally written for "
            "  a `@#for`, or vice versa\n"
            "- A nested block missing its inner closer\n\n"
            "**Fix**\n\n"
            "Add the missing `@#endif` or `@#endfor` at the appropriate "
            "scope, or remove the stray closer. Each `@#if` needs its "
            "own `@#endif`; each `@#for` its own `@#endfor`."
        ),
    },
    "E063": {
        "title": "Undefined macro interpolation",
        "body": (
            "An active line still contains an unresolved `@{NAME}` macro "
            "interpolation after the language server applied the simple "
            "`@#define` substitutions it can evaluate. Dynare's macro "
            "preprocessor cannot produce valid model code unless that "
            "macro name is defined in scope.\n\n"
            "**Fix**\n\n"
            "Define the macro with `@#define NAME = value` before the line "
            "that uses it, correct the macro name, or remove the interpolation."
        ),
    },
    "E064": {
        "title": "Macro error directive",
        "body": (
            "An active Dynare macro `@#error` directive was reached. Dynare's "
            "macro preprocessor stops when this directive is active.\n\n"
            "**Fix**\n\n"
            "Remove the `@#error` directive, or guard it behind a macro "
            "condition that is false for this model variant."
        ),
    },
    "E065": {
        "title": "Invalid steady_state operand",
        "body": (
            "The `steady_state(...)` operator must refer to model endogenous "
            "or parameter expressions, not exogenous shocks. Exogenous "
            "variables are not valid operands for this Dynare operator.\n\n"
            "**Fix**\n\n"
            "Remove the exogenous variable from `steady_state(...)`, replace "
            "it with the intended endogenous or parameter expression, or "
            "rewrite the equation so the shock enters outside the operator."
        ),
    },
    "E999": {
        "title": "Additional errors truncated",
        "body": (
            "More diagnostics were produced than the server displays at "
            "once. Fix the visible errors first; the next analysis pass "
            "will surface anything that was previously hidden."
        ),
    },
    "W010": {
        "title": "Parameter declared but never assigned",
        "body": (
            "A name appears in the `parameters` block but no assignment was "
            "found in the parameter section or in `steady_state_model`. At "
            "runtime, the parameter will be undefined and most computations "
            "will fail.\n\n"
            "**Fix**\n\n"
            "Assign a numerical value, or remove the declaration if the "
            "parameter is no longer used."
        ),
    },
    "W011": {
        "title": "Parameter assignment cannot be evaluated",
        "body": (
            "An assignment like `phi = 1/(1-beta)` could not be evaluated "
            "because one or more right-hand-side symbols are not yet "
            "defined. The parameter falls back to undefined.\n\n"
            "**Fix**\n\n"
            "Reorder the parameter section so that dependencies appear "
            "before dependents."
        ),
    },
    "W012": {
        "title": "Undeclared helper variable in parameter section",
        "body": (
            "An identifier appears on the right-hand side of a parameter "
            "assignment but is not declared as a parameter or known helper "
            "variable.\n\n"
            "**Fix**\n\n"
            "Add a declaration, or replace the helper with an explicit "
            "numeric value."
        ),
    },
    "W020": {
        "title": "Endogenous variable never referenced in model",
        "body": (
            "An endogenous variable is declared in `var` but does not "
            "appear in any equation. Either remove the declaration or add "
            "the missing equation that uses the variable."
        ),
    },
    "W021": {
        "title": "Exogenous variable never referenced in model",
        "body": (
            "A shock declared in `varexo` does not appear in any equation. "
            "It will have no effect at runtime."
        ),
    },
    "W022": {
        "title": "Parameter declared but never referenced in model equations",
        "body": (
            "A parameter is declared and assigned but does not appear in "
            "any model equation. Often the result of stripping an equation "
            "but forgetting to remove the parameter."
        ),
    },
    "W040": {
        "title": "Steady-state summary",
        "body": (
            "A summary of the steady-state residual check: of the *n* "
            "equations evaluated, *k* are satisfied within tolerance and "
            "the remainder report non-zero residuals (W041 cites each "
            "individually).\n\n"
            "**Fix**\n\n"
            "Adjust the `steady_state_model` expressions, or supply a "
            "better `initval` so the solver can find the consistent "
            "values."
        ),
    },
    "W041": {
        "title": "Steady-state equation has a non-zero residual",
        "body": (
            "Substituting the supplied steady-state values into the cited "
            "equation gives a non-zero result. Either the analytical "
            "formula is wrong, or the parameter calibration is "
            "inconsistent with the equation.\n\n"
            "**Fix**\n\n"
            "Recompute the steady state symbolically, or invoke the "
            "automatic solver via the Compute Steady State code action."
        ),
    },
    "W042": {
        "title": "Endogenous variable missing from steady_state_model",
        "body": (
            "The `steady_state_model` block does not assign a value for "
            "every endogenous variable. Dynare will fall back to the "
            "`initval` value (or zero), which usually produces an "
            "inconsistent steady state.\n\n"
            "**Fix**\n\n"
            "Add the missing assignments. The Compute Steady State action "
            "fills them in automatically when the solver converges."
        ),
    },
    "W050": {
        "title": "Undeclared variable in initval",
        "body": (
            "An entry in the `initval` block refers to a name that is not "
            "declared as a variable. Dynare's preprocessor will reject "
            "this.\n\n"
            "**Fix**\n\n"
            "Declare the variable, or remove the stray `initval` entry."
        ),
    },
    "W051": {
        "title": "Exogenous variable set in initval",
        "body": (
            "Setting an exogenous variable in `initval` has no effect on "
            "the steady-state computation. Shocks are zero at the "
            "deterministic steady state by construction."
        ),
    },
    "W053": {
        "title": "Parameter assigned in initval/endval is ignored",
        "body": (
            "A parameter is assigned a value inside an `initval`/`endval` "
            "block, where Dynare ignores it. Assign parameters before the "
            "model block, or inside `steady_state_model`, instead."
        ),
    },
    "W052": {
        "title": "Endogenous variable missing from initval",
        "body": (
            "The `initval` block does not provide an initial guess for "
            "every endogenous variable. The solver will start from zero "
            "for the missing entries, which may slow or prevent "
            "convergence on nonlinear models."
        ),
    },
    "W060": {
        "title": "Exogenous variables declared but no shocks block",
        "body": (
            "One or more exogenous variables are declared in `varexo` but "
            "the file contains no `shocks` block specifying their "
            "variance-covariance structure. The model is then "
            "deterministic.\n\n"
            "**Fix**\n\n"
            "Add a `shocks` block to define the shock processes, or remove "
            "the unused `varexo` declarations."
        ),
    },
    "W061": {
        "title": "Ambiguous include parent context",
        "body": (
            "The active include file is reachable from more than one parent "
            "model, so the language server cannot safely infer which parent "
            "declarations and block context should apply.\n\n"
            "**Fix**\n\n"
            "Open or run the intended parent `.mod` file, or provide only "
            "that parent and its include closure when calling workspace tools."
        ),
    },
    "I041": {
        "title": "Steady-state evaluation failed (non-residual reason)",
        "body": (
            "Evaluating the steady-state equation raised an error other "
            "than a missing variable — for example a domain error in "
            "`log(.)` or a division by zero. The diagnostic is "
            "informational because the equation may still be correct at "
            "a feasible steady state."
        ),
    },
    "I050": {
        "title": "No steady state provided; automatic solver available",
        "body": (
            "The file declares variables and equations but does not include "
            "an `initval` or `steady_state_model` block. The Compute Steady "
            "State code action can derive one automatically using the "
            "multi-strategy solver chain (Gauss-Seidel → trust-region "
            "least squares → root finders → homotopy → random restarts).\n\n"
            "**Fix**\n\n"
            "Invoke the code action, or provide an initial guess manually."
        ),
    },
    "W070": {
        "title": "Parameter outside its conventional range",
        "body": (
            "A parameter assignment falls outside the theoretically "
            "admissible range for its standard interpretation. The bounds "
            "table in `dynare_lsp.bounds` is opinionated but conservative: "
            "it flags values that violate the *theoretical* admissible "
            "range under the parameter's conventional meaning, not values "
            "that simply look unusual.\n\n"
            "**Common causes**\n\n"
            "- Unit error: e.g. `beta = 99` when 0.99 was meant\n"
            "- Sign error on a quantity that must be non-negative "
            "  (variance, standard deviation, depreciation rate)\n"
            "- Gross-vs-net confusion on a rate parameter\n\n"
            "**Fix**\n\n"
            "Correct the value, or — if the calibration is intentional — "
            "ignore the warning. This is a soft check, not a structural "
            "error: Dynare will accept any numeric value."
        ),
    },
    "I070": {
        "title": "Blanchard-Kahn condition satisfied",
        "body": (
            "The number of explosive (|lambda| > 1) eigenvalues equals "
            "the number of forward-looking variables — Dynare's "
            "saddle-path stability condition.  The model admits a "
            "unique stable rational-expectations solution at the "
            "current calibration.  No action required; this is "
            "informational."
        ),
    },
    "P000": {
        "title": "Dynare preprocessor diagnostic",
        "body": (
            "A message emitted by the external Dynare preprocessor "
            "(``dynare-preprocessor`` binary) when it parses the file. "
            "Codes prefixed ``P`` come from the preprocessor, not from "
            "this language server's static analysis.  The message text "
            "and source location are passed through directly.\n\n"
            "**Common reasons**\n\n"
            "- A genuine syntax error the preprocessor caught\n"
            "- A semantic error involving ``check``, ``stoch_simul``, "
            "  or other runtime statements\n"
            "- The preprocessor binary timed out (``P000``: timeout)\n\n"
            "**Fix**\n\n"
            "Read the underlying message; the preprocessor's error "
            "reporting points at the source location.  For timeouts, "
            "consider whether the model is too large for the configured "
            "timeout or whether a non-terminating macro loop is present."
        ),
    },
    "DYNR": {
        "title": "Dynare run diagnostic",
        "body": (
            "A message produced by actually running the model through "
            "Dynare in MATLAB/Octave (the editor's 'Run Dynare' action), "
            "not by this language server's static analysis.  Errors are "
            "passed through from the Dynare/MATLAB run; a warning with "
            "this code is also used when the run reports that the "
            "Blanchard-Kahn conditions are not satisfied.\n\n"
            "**Common reasons**\n\n"
            "- The model fails at the ``steady`` / ``check`` / "
            "``stoch_simul`` stage even though it parses cleanly\n"
            "- Missing data files or toolboxes in the MATLAB environment\n"
            "- Blanchard-Kahn rank or order conditions failing at solve "
            "time\n\n"
            "**Fix**\n\n"
            "Read the underlying Dynare/MATLAB message; it describes the "
            "run, so re-run after editing the model.  The diagnostic "
            "clears on the next successful run."
        ),
    },
    "W071": {
        "title": "Blanchard-Kahn condition not satisfied",
        "body": (
            "The numerical Blanchard-Kahn check found that the number of "
            "explosive (|lambda| > 1) eigenvalues of the linearised model "
            "does not equal the number of forward-looking variables. "
            "Dynare requires this equality for a unique stable rational-"
            "expectations equilibrium.\n\n"
            "**Common causes**\n\n"
            "- A timing error in the model equations (variable typed "
            "  with the wrong lead/lag)\n"
            "- An infeasible calibration that makes the model "
            "  indeterminate or explosive\n"
            "- A missing equation that pins down one of the forward-"
            "  looking variables\n\n"
            "**Fix**\n\n"
            "Re-check the timing of each forward-looking variable in the "
            "model block, verify the parameter calibration, and confirm "
            "the equation count is correct (see E010). If the model is "
            "intentionally indeterminate (e.g. policy analysis), this "
            "warning is informational."
        ),
    },
    "W080": {
        "title": "Model diagnostics found collinearity",
        "body": (
            "The model diagnostics check linearised the model at the "
            "steady state and found that the steady-state Jacobian is "
            "rank deficient. This means one or more equations or variables "
            "are locally collinear, so the steady state may not be uniquely "
            "pinned down. The warning lists, for each collinear relation, the "
            "minimal set of collinear variables (right null space) and the "
            "collinear equations (left null space). If an eigenvalue has "
            "modulus near one, it also notes that the singularity may be a "
            "unit root, which is expected for an intentionally nonstationary "
            "model.\n\n"
            "**Fix**\n\n"
            "Inspect the listed variables and the equations that define "
            "them. Look for duplicated equations, redundant identities, "
            "missing normalizations, or an equation that is an exact linear "
            "combination of another equation at the steady state."
        ),
    },
    "W081": {
        "title": "First-order structural identification warning",
        "body": (
            "The first-order structural identification check perturbed "
            "each calibrated parameter and compared its local effect on "
            "steady-state residuals and the first-order structural "
            "Jacobian. A parameter with no effect, or a group of "
            "parameters with collinear effect vectors, may not be "
            "separately identifiable from first-order model structure.\n\n"
            "This is not a full Iskrev-style identification test: it "
            "does not use data, moments, observables, or an information "
            "matrix.\n\n"
            "**Fix**\n\n"
            "Inspect the listed parameters and equations. Look for "
            "parameters that never enter the model at first order, "
            "parameters that only appear through a product or ratio, or "
            "normalizations that leave two calibrations doing the same "
            "local job."
        ),
    },
    "W090": {
        "title": "Observed variable is not a declared endogenous variable",
        "body": (
            "A name listed in ``varobs`` is not a declared endogenous "
            "variable. Dynare requires every observed variable to be an "
            "endogenous variable of the model.\n\n"
            "**Fix**\n\n"
            "Declare the variable in ``var``, or remove it from ``varobs`` if "
            "it was a typo or an exogenous/parameter name."
        ),
    },
    "W091": {
        "title": "Duplicate observed variable",
        "body": (
            "A variable is listed more than once in ``varobs``. Each observed "
            "variable should appear exactly once.\n\n"
            "**Fix**\n\n"
            "Remove the duplicate entry."
        ),
    },
    "W092": {
        "title": "Stochastic singularity",
        "body": (
            "There are more observed variables (``varobs``) than shocks "
            "(structural shocks plus measurement errors). The likelihood is "
            "then stochastically singular and estimation cannot proceed: the "
            "model cannot generate enough independent variation to match the "
            "observed series.\n\n"
            "**Fix**\n\n"
            "Add structural shocks, add measurement errors on the observed "
            "variables (an ``stderr`` on an observed variable), or reduce the "
            "number of observed variables so that observables ≤ shocks."
        ),
    },
    "W093": {
        "title": "estimated_params references an undeclared symbol",
        "body": (
            "An ``estimated_params`` entry names a symbol that is not declared "
            "with the expected role: a plain entry must name a parameter, an "
            "``stderr`` entry must name a shock or observed variable, and a "
            "``corr`` entry must name two declared shocks or variables.\n\n"
            "**Fix**\n\n"
            "Declare the symbol, or correct the name / entry type."
        ),
    },
    "W094": {
        "title": "estimated_params bound or initial-value inconsistency",
        "body": (
            "An ``estimated_params`` entry has a lower bound that is not below "
            "its upper bound, or an initial value that lies outside the "
            "``[lower, upper]`` interval. Dynare needs a non-empty bound "
            "interval containing the starting value.\n\n"
            "**Fix**\n\n"
            "Order the bounds so that lower < upper and place the initial "
            "value inside them."
        ),
    },
    "W095": {
        "title": "observation_trends variable not in varobs",
        "body": (
            "A variable given a trend in ``observation_trends`` is not listed "
            "in ``varobs``. Trends are only meaningful for observed "
            "variables.\n\n"
            "**Fix**\n\n"
            "Add the variable to ``varobs`` or remove its trend specification."
        ),
    },
    "W100": {
        "title": "Optimal-policy command requires a planner_objective",
        "body": (
            "``ramsey_model``, ``ramsey_policy``, and ``discretionary_policy`` "
            "optimise a planner's loss function, so they require a "
            "``planner_objective`` statement, which is missing.\n\n"
            "**Fix**\n\n"
            "Add a ``planner_objective <expression>;`` statement before the "
            "policy command."
        ),
    },
    "W101": {
        "title": "Policy instrument is not a declared endogenous variable",
        "body": (
            "An ``instruments=(...)`` entry names a symbol that is not a "
            "declared endogenous variable. Policy instruments must be "
            "endogenous variables of the model.\n\n"
            "**Fix**\n\n"
            "Declare the instrument in ``var``, or correct the instrument name."
        ),
    },
    "W102": {
        "title": "planner_discount is not a valid discount factor",
        "body": (
            "``planner_discount`` must be a discount factor in the interval "
            "(0, 1]. A value outside this range is almost certainly a "
            "mistake (for example, entering a discount rate instead of a "
            "factor).\n\n"
            "**Fix**\n\n"
            "Set ``planner_discount`` to a value such as 0.99."
        ),
    },
    "W103": {
        "title": "osr is missing osr_params or optim_weights",
        "body": (
            "Optimal simple rules (``osr``) optimise the values of chosen "
            "parameters to minimise a weighted objective, so they require an "
            "``osr_params`` statement (the parameters to optimise) and an "
            "``optim_weights`` block (the objective weights). One of these is "
            "missing.\n\n"
            "**Fix**\n\n"
            "Add the missing ``osr_params`` statement and/or ``optim_weights`` "
            "block."
        ),
    },
    "W110": {
        "title": "Shock correlation outside [-1, 1]",
        "body": (
            "A ``corr`` entry in the shocks block sets a correlation whose "
            "magnitude exceeds one. A correlation coefficient must lie in "
            "[-1, 1], and the implied covariance matrix would not be positive "
            "semidefinite.\n\n"
            "**Fix**\n\n"
            "Set the correlation to a value in [-1, 1]."
        ),
    },
    "W111": {
        "title": "Shock variance or correlation specified more than once",
        "body": (
            "A shock's variance / standard error, or a correlation pair, is "
            "specified more than once in the shocks block. The repeated entry "
            "silently overrides the earlier one and is usually a mistake.\n\n"
            "**Fix**\n\n"
            "Keep a single specification per shock variance and per "
            "correlation pair."
        ),
    },
    "W112": {
        "title": "Negative shock variance",
        "body": (
            "A shocks-block ``var e = ...`` entry sets a variance that folds to "
            "a negative constant. A variance is a squared quantity and must be "
            'non-negative; Dynare errors with "You have specified negative '
            'shock variances."\n\n'
            "(The ``stderr`` form is not flagged: Dynare squares the standard "
            "error, so a negative ``stderr`` still yields a valid variance.)\n\n"
            "**Fix**\n\n"
            "Use a non-negative value. Recall the ``var`` form sets the "
            "variance, i.e. the standard error *squared* (e.g. "
            "``var e = 0.01^2;``)."
        ),
    },
    "W120": {
        "title": "Stochastic command with no stochastic exogenous variable",
        "body": (
            "``stoch_simul`` / ``estimation`` drive the model with stochastic "
            "shocks, but the model declares no stochastic ``varexo``. "
            "``varexo_det`` declarations are deterministic and do not satisfy "
            'that runtime requirement. Dynare stops with "stoch_simul:: does '
            'not support having no varexo in the model."'
            "\n\n"
            "**Fix**\n\n"
            "Declare at least one stochastic exogenous variable (a dummy "
            "``varexo`` plus a shocks-block entry is enough if the model is otherwise "
            "deterministic)."
        ),
    },
    "W121": {
        "title": "Parameter used with a lead or lag",
        "body": (
            "A declared parameter appears with a time subscript such as "
            "``beta(+1)`` or ``rho(-1)`` in the model block. Parameters are "
            "time-invariant constants, so a lead/lag on one is meaningless and "
            "almost always means the symbol should have been declared as a "
            "variable, or that the time index is stray.\n\n"
            "**Fix**\n\n"
            "Remove the time index, or declare the symbol with ``var`` / "
            "``varexo`` if it really is a variable."
        ),
    },
    "W122": {
        "title": "Deep parameter assigned a non-finite value",
        "body": (
            "A parameter that is used in the model equations is assigned a "
            "non-finite value (``NaN`` or ``Inf``) while a run command "
            "(``steady``, ``stoch_simul``, ``perfect_foresight_*``, "
            "``estimation``, ...) is present. Dynare's deep-parameter "
            "calibration check rejects ``NaN`` / ``Inf`` parameters.\n\n"
            "**Fix**\n\n"
            "Assign a finite numeric value before the run command."
        ),
    },
    "W130": {
        "title": "Variable used before assignment in steady_state_model",
        "body": (
            "The ``steady_state_model`` block is evaluated top to bottom as a "
            "sequence of assignments, so every variable on a right-hand side "
            "must already have been assigned above. A variable is referenced "
            "before its own assignment, which the Dynare preprocessor "
            "rejects.\n\n"
            "**Fix**\n\n"
            "Reorder the assignments so each variable is computed before it is "
            "used."
        ),
    },
    "W131": {
        "title": "Variable silently overwritten in steady_state_model",
        "body": (
            "A variable is assigned more than once in the "
            "``steady_state_model`` block and the later assignment does not "
            "use the earlier value, so the first assignment is dead. (An "
            "in-place transformation that reuses the value, such as the "
            "``A = log(A)`` log-model idiom, is intentional and is not "
            "flagged.)\n\n"
            "**Fix**\n\n"
            "Remove the redundant assignment, or fold the two into one."
        ),
    },
    "W140": {
        "title": "Nonlinear operator in a linear model",
        "body": (
            "The model is declared ``linear`` (``model(linear);``) but an "
            "equation applies a nonlinear operator to a variable. Examples "
            "include variable-dependent functions (such as ``log(y)`` or "
            "``abs(e)``), products or ratios involving multiple variables "
            "(``c*k`` or ``a/y``), powers involving variables (``k^2``), and "
            "comparisons. Dynare requires the equations of a ``linear`` model "
            "to be linear in the variables and errors otherwise.\n\n"
            "**Fix**\n\n"
            "Remove the ``linear`` option, or rewrite the equation without the "
            "nonlinear operator."
        ),
    },
    "W150": {
        "title": "Deprecated command or option",
        "body": (
            "A deprecated command or option is used; current Dynare warns and "
            "may remove it in a future release. Commands: ``simul`` → "
            "``perfect_foresight_setup`` + ``perfect_foresight_solver``; "
            "``ramsey_policy`` → ``ramsey_model`` + ``stoch_simul``. Options: "
            "``aim_solver`` → ``dr = aim``; ``bytecode`` (being removed).\n\n"
            "**Fix**\n\n"
            "Switch to the modern command or option form."
        ),
    },
}


def explain(code: str) -> Optional[Dict[str, str]]:
    """Return ``{'title', 'body'}`` for a diagnostic code, or ``None``.

    Code lookup is case-insensitive.  ``P###`` codes (P001, P002, ...)
    are routed to the generic ``P000`` preprocessor-passthrough entry —
    the preprocessor synthesises codes one-per-emitted-line and they
    all share the same "this came from the external Dynare binary"
    semantics.
    """
    key = code.upper()
    import re as _re

    if _re.fullmatch(r"P\d+", key):
        key = "P000"
    return _ENTRIES.get(key)


def render_markdown(code: str) -> Optional[str]:
    """Render the explanation as a single markdown string.

    Returns ``None`` if the code is unknown.
    """
    entry = explain(code)
    if entry is None:
        return None
    return f"### {code}: {entry['title']}\n\n{entry['body']}\n"


def known_codes() -> list[str]:
    """Sorted list of all codes that have explanations."""
    return sorted(_ENTRIES.keys())
