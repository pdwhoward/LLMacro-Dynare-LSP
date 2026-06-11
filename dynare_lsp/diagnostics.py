"""Diagnostics engine for Dynare .mod files.

Generates LSP-compatible diagnostics from a parsed model, covering:
  - Structural errors (unmatched blocks, missing semicolons)
  - Variable reference validation (undeclared variables in equations)
  - Equation count vs. endogenous variable count
  - Unassigned parameters
  - Steady state equation residuals
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from .parser import (
    IncludeDirective,
    MacroDirective,
    ParamAssignment,
    ParsedModel,
    Position,
    SourceRange,
    VarDeclaration,
    _DYNARE_COMMANDS,
    _block_ranges,
    _find_all_blocks,
    _inside_block,
    _iter_equation_tag_spans,
    _macro_branch_state,
    _macro_truth_value,
    _mask_string_literals,
    _safe_eval,
    _strip_comments,
    _strip_non_macro_comments,
    _unresolved_macro_for_template_ranges,
    parse,
)
from .steady_state import evaluate_steady_state_model_assignments, validate_steady_state


class Severity(IntEnum):
    """LSP DiagnosticSeverity values."""

    ERROR = 1
    WARNING = 2
    INFORMATION = 3
    HINT = 4


@dataclass
class TextEdit:
    """A text edit to auto-fix a diagnostic.

    Represents either an insertion or a replacement on the source text.
    Line numbers are 0-based.
    """

    start_line: int
    start_char: int
    end_line: int
    end_char: int
    new_text: str

    @staticmethod
    def insert_at(line: int, char: int, text: str) -> "TextEdit":
        """Insert text at the given position."""
        return TextEdit(line, char, line, char, text)

    @staticmethod
    def insert_line_after(line: int, text: str) -> "TextEdit":
        """Insert a new line after the given 0-based line number."""
        return TextEdit(line, 999999, line, 999999, "\n" + text)


@dataclass
class Diagnostic:
    """A single diagnostic (error / warning / info) with source location."""

    range: SourceRange
    severity: Severity
    message: str
    source: str = "dynare"
    code: str = ""
    fix: Optional[TextEdit] = None
    # LSP DiagnosticTag values (1 = Unnecessary, 2 = Deprecated).  Kept as bare
    # ints so this module stays free of any lsprotocol dependency; the server
    # maps them to ``lsp.DiagnosticTag`` when converting for the wire.
    tags: Optional[List[int]] = None


def _merge_declarations(
    local: List[VarDeclaration],
    include_models: List[ParsedModel],
    attr: str,
) -> List[VarDeclaration]:
    merged = list(local)
    seen = {decl.name for decl in merged}
    for include_model in include_models:
        for decl in getattr(include_model, attr):
            if decl.name not in seen:
                merged.append(
                    replace(
                        decl,
                        range=_include_visible_range(include_model, decl.range),
                    )
                )
                seen.add(decl.name)
    return merged


def _append_unique_strings(local: List[str], included: List[str]) -> List[str]:
    merged = list(local)
    seen = set(merged)
    for item in included:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged


def _include_visible_range(
    include_model: ParsedModel,
    rng: SourceRange,
) -> SourceRange:
    return include_model.include_anchor_range or rng


def _source_ordered_assignments(
    assignments: List[ParamAssignment],
) -> List[ParamAssignment]:
    return [
        assignment
        for _index, assignment in sorted(
            enumerate(assignments),
            key=lambda item: (
                item[1].range.start.line,
                item[1].range.start.character,
                item[0],
            ),
        )
    ]


def _reevaluate_assignments(
    param_assignments: List[ParamAssignment],
    helper_assignments: List[ParamAssignment],
) -> Tuple[List[ParamAssignment], List[ParamAssignment]]:
    """Evaluate parameter/helper assignments in textual include order."""
    known: Dict[str, float] = {}
    updated: Dict[int, ParamAssignment] = {}
    for assignment in _source_ordered_assignments(
        param_assignments + helper_assignments,
    ):
        value = _safe_eval(assignment.expression, known)
        if value is not None:
            known[assignment.name] = value
        else:
            known.pop(assignment.name, None)
        updated[id(assignment)] = replace(assignment, value=value)

    return (
        [updated.get(id(a), a) for a in param_assignments],
        [updated.get(id(a), a) for a in helper_assignments],
    )


def model_with_include_context(
    model: ParsedModel,
    include_models: Optional[List[ParsedModel]] = None,
    *,
    include_model_equations: bool = True,
    include_steady_state: bool = True,
    include_initval: bool = True,
) -> ParsedModel:
    """Return a shallow diagnostic view with include-visible model context.

    The active model remains the source of publishable locations. Included
    declarations and calibration values are added so whole-model semantic
    checks do not report false positives when a file relies on ``@#include``.
    """
    if not include_models:
        return model

    context = replace(model)
    context.endogenous = _merge_declarations(
        model.endogenous, include_models, "endogenous"
    )
    context.exogenous = _merge_declarations(
        model.exogenous, include_models, "exogenous"
    )
    context.deterministic_exogenous = _merge_declarations(
        model.deterministic_exogenous,
        include_models,
        "deterministic_exogenous",
    )
    context.parameters = _merge_declarations(
        model.parameters, include_models, "parameters"
    )
    context.predetermined_variables = _merge_declarations(
        model.predetermined_variables,
        include_models,
        "predetermined_variables",
    )
    context.policy_commands = _append_unique_strings(
        list(model.policy_commands),
        [
            command
            for include_model in include_models
            for command in include_model.policy_commands
        ],
    )
    context.policy_command_range = model.policy_command_range
    if context.policy_command_range is None:
        for include_model in include_models or []:
            if include_model.policy_command_range is not None:
                context.policy_command_range = _include_visible_range(
                    include_model,
                    include_model.policy_command_range,
                )
                break
    context.planner_objective_range = model.planner_objective_range
    if context.planner_objective_range is None:
        for include_model in include_models or []:
            if include_model.planner_objective_range is not None:
                context.planner_objective_range = _include_visible_range(
                    include_model,
                    include_model.planner_objective_range,
                )
                break
    context.instruments = _append_unique_strings(
        list(model.instruments),
        [
            instrument
            for include_model in include_models
            for instrument in include_model.instruments
        ],
    )
    context.planner_discount = model.planner_discount
    if context.planner_discount is None:
        context.planner_discount = next(
            (
                include_model.planner_discount
                for include_model in include_models
                if include_model.planner_discount is not None
            ),
            None,
        )
    context.osr_params = _append_unique_strings(
        list(model.osr_params),
        [
            param
            for include_model in include_models
            for param in include_model.osr_params
        ],
    )
    context.has_optim_weights = model.has_optim_weights or any(
        include_model.has_optim_weights for include_model in include_models
    )
    context.has_occbin = model.has_occbin or any(
        include_model.has_occbin for include_model in include_models
    )

    context.varobs_vars = list(model.varobs_vars) + [
        name for include_model in include_models for name in include_model.varobs_vars
    ]
    context.varobs_range = model.varobs_range
    if context.varobs_range is None:
        for include_model in include_models:
            if include_model.varobs_range is not None:
                context.varobs_range = _include_visible_range(
                    include_model,
                    include_model.varobs_range,
                )
                break
    context.varexobs_vars = list(model.varexobs_vars) + [
        name for include_model in include_models for name in include_model.varexobs_vars
    ]
    context.varexobs_range = model.varexobs_range
    if context.varexobs_range is None:
        for include_model in include_models:
            if include_model.varexobs_range is not None:
                context.varexobs_range = _include_visible_range(
                    include_model,
                    include_model.varexobs_range,
                )
                break
    context.estimated_params = list(model.estimated_params)
    for include_model in include_models:
        for entry in include_model.estimated_params:
            entry_range = (
                _include_visible_range(include_model, entry.range)
                if entry.range is not None
                else None
            )
            context.estimated_params.append(replace(entry, range=entry_range))
    context.estimated_params_range = model.estimated_params_range
    if context.estimated_params_range is None:
        for include_model in include_models:
            if include_model.estimated_params_range is not None:
                context.estimated_params_range = _include_visible_range(
                    include_model,
                    include_model.estimated_params_range,
                )
                break
    context.observation_trends_vars = _append_unique_strings(
        list(model.observation_trends_vars),
        [
            name
            for include_model in include_models
            for name in include_model.observation_trends_vars
        ],
    )
    context.observation_trends_ranges = dict(model.observation_trends_ranges)
    for include_model in include_models:
        for name, rng in include_model.observation_trends_ranges.items():
            context.observation_trends_ranges.setdefault(
                name,
                _include_visible_range(include_model, rng),
            )

    context_param_names = {p.name for p in context.parameters}
    included_param_assignments = [
        assignment
        for include_model in include_models
        for assignment in (
            include_model.param_assignments + include_model.helper_assignments
        )
        if assignment.name in context_param_names
    ]
    included_helper_assignments = [
        assignment
        for include_model in include_models
        for assignment in include_model.helper_assignments
        if assignment.name not in context_param_names
    ]
    active_param_assignments = [
        assignment
        for assignment in model.param_assignments + model.helper_assignments
        if assignment.name in context_param_names
    ]
    active_helper_assignments = [
        assignment
        for assignment in model.helper_assignments
        if assignment.name not in context_param_names
    ]
    context.param_assignments = _source_ordered_assignments(
        included_param_assignments + active_param_assignments,
    )
    context.helper_assignments = _source_ordered_assignments(
        included_helper_assignments + active_helper_assignments,
    )
    context.param_assignments, context.helper_assignments = _reevaluate_assignments(
        context.param_assignments,
        context.helper_assignments,
    )

    if include_model_equations:
        included_model_equations = [
            equation
            for include_model in include_models
            for equation in include_model.model_equations
        ]
        context.model_equations = sorted(
            included_model_equations + list(model.model_equations),
            key=lambda eq: _range_key(eq.range),
        )

    # steady_state_model / initval / endval statements are SEQUENTIAL and
    # loop expansions are iteration-major with shared template-line ranges,
    # so a per-item positional sort would regroup them statement-major
    # (false W041 on cross-iteration recursions).  Keep each source list's
    # internal order and splice the sources by their first item's position.
    def _splice_sequential(source_lists):
        # Split each source into RUNS of non-decreasing line positions: a
        # decrease marks the next iteration of a loop expansion (emission
        # lines L1,L2,L1,L2...).  Runs interleave positionally like the old
        # per-item sort (so textually-nested includes keep their order),
        # while items inside a run never regroup.
        def _key(item):
            return (item.range.start.line, item.range.start.character)

        def _runs(items):
            # Positions occurring more than once mark loop-expansion
            # territory: every iteration re-emits the same template
            # positions (whether real source coordinates or the synthetic
            # include-ordering coordinates).  Glue consecutive ascending
            # items on duplicated positions into one run (a single
            # iteration); everything else stays a singleton so textually
            # nested includes still interleave per-item like before.
            keys = [_key(item) for item in items]
            duplicated = {key for key in keys if keys.count(key) > 1}
            runs, current = [], []
            prev_key = None
            for item, key in zip(items, keys):
                glue = (
                    bool(current)
                    and prev_key is not None
                    and key > prev_key
                    and key in duplicated
                    and prev_key in duplicated
                )
                if current and not glue:
                    runs.append(current)
                    current = []
                current.append(item)
                prev_key = key
            if current:
                runs.append(current)
            return runs

        all_runs = [
            run for items in source_lists if items for run in _runs(list(items))
        ]
        all_runs.sort(key=lambda run: _range_key(run[0].range))
        return [item for run in all_runs for item in run]

    if include_steady_state:
        context.steady_state_equations = _splice_sequential(
            [include_model.steady_state_equations for include_model in include_models]
            + [model.steady_state_equations]
        )
    if include_initval:
        context.initval_entries = _splice_sequential(
            [include_model.initval_entries for include_model in include_models]
            + [model.initval_entries]
        )
        context.endval_entries = _splice_sequential(
            [include_model.endval_entries for include_model in include_models]
            + [model.endval_entries]
        )

    context.model_remove_names = [
        name
        for source_model in [*include_models, model]
        for name in source_model.model_remove_names
    ]
    context.model_replacements = [
        replacement
        for source_model in [*include_models, model]
        for replacement in source_model.model_replacements
    ]
    context.var_removed_names = [
        name
        for source_model in [*include_models, model]
        for name in source_model.var_removed_names
    ]

    shocks_vars = list(model.shocks_vars)
    for include_model in include_models:
        for name in include_model.shocks_vars:
            if name not in shocks_vars:
                shocks_vars.append(name)
    context.shocks_vars = shocks_vars

    return context


def _with_model_editing_commands(model: ParsedModel) -> ParsedModel:
    """Return a diagnostic view after model_remove/model_replace/var_remove."""
    if (
        not model.model_remove_names
        and not model.model_replacements
        and not model.var_removed_names
    ):
        return model

    edited = replace(model)
    removed_equation_names = set(model.model_remove_names)
    for replacement in model.model_replacements:
        removed_equation_names.update(replacement.names)
    replacement_equations = [
        equation
        for replacement in model.model_replacements
        for equation in replacement.equations
    ]
    if removed_equation_names or replacement_equations:
        edited.model_equations = sorted(
            [
                equation
                for equation in model.model_equations
                if not equation.name or equation.name not in removed_equation_names
            ]
            + replacement_equations,
            key=lambda eq: _range_key(eq.range),
        )

    removed_vars = set(model.var_removed_names)
    if removed_vars:
        edited.endogenous = [
            decl for decl in model.endogenous if decl.name not in removed_vars
        ]
        edited.parameters = [
            decl for decl in model.parameters if decl.name not in removed_vars
        ]
        edited.predetermined_variables = [
            decl
            for decl in model.predetermined_variables
            if decl.name not in removed_vars
        ]

    return edited


def _range_key(rng: SourceRange) -> Tuple[int, int, int, int]:
    return (
        rng.start.line,
        rng.start.character,
        rng.end.line,
        rng.end.character,
    )


def _diagnostic_quoted_name(diagnostic: Diagnostic) -> Optional[str]:
    match = re.search(r"'([A-Za-z_][A-Za-z0-9_]*)'", diagnostic.message)
    return match.group(1) if match else None


def _used_parameter_names(model: ParsedModel) -> set[str]:
    param_names = model.parameter_names()
    if not param_names:
        return set()
    declared_names = model.all_declared_names()
    used: set[str] = set()
    for equation in model.model_equations:
        for ref in _extract_equation_references(equation, declared_names):
            if ref in param_names:
                used.add(ref)
    return used


def _anchor_to_include(
    diagnostic: Diagnostic,
    anchor: SourceRange,
) -> Diagnostic:
    return replace(diagnostic, range=anchor, fix=None)


def _macro_for_template_ranges(model: ParsedModel) -> List[Tuple[int, int]]:
    """Return @#for line ranges whose body contains unresolved interpolation.

    The result is cached on the model instance: the underlying scan calls
    ``_macro_branch_state`` and several regex sweeps (~8 ms on large models),
    so caching avoids re-running it on every keystroke.
    """
    if model._cached_macro_for_template_ranges is None:
        source = (
            model.text
            if getattr(model, "source_map", None)
            else (model.original_text or model.text)
        )
        model._cached_macro_for_template_ranges = _unresolved_macro_for_template_ranges(
            source
        )
    return model._cached_macro_for_template_ranges


def _line_in_ranges(line: int, ranges: List[Tuple[int, int]]) -> bool:
    return any(start <= line <= end for start, end in ranges)


def _without_macro_for_template_equations(
    model: ParsedModel,
    ranges: List[Tuple[int, int]],
) -> ParsedModel:
    if not ranges:
        return model
    filtered = replace(model)
    filtered.model_equations = [
        eq
        for eq in model.model_equations
        if not _line_in_ranges(eq.range.start.line, ranges)
    ]
    return filtered


def _build_line_starts(text: str) -> List[int]:
    """Compute the byte offset at which each line starts.

    Returns a list ``starts`` where ``starts[i]`` is the character offset of
    line *i* (0-based).  A single precomputed call replaces the repeated
    ``text.split("\\n")`` that made :func:`_position_to_offset` O(n) per call
    when invoked hundreds of times on the same large text.
    """
    starts: List[int] = [0]
    i = text.find("\n")
    while i != -1:
        starts.append(i + 1)
        i = text.find("\n", i + 1)
    return starts


def _position_to_offset_with_index(
    line_starts: List[int],
    text_len: int,
    pos: Position,
) -> int:
    """O(1) :func:`_position_to_offset` using a precomputed line-starts index."""
    line = pos.line
    char = max(pos.character, 0)
    if line <= 0:
        line_end = line_starts[1] - 1 if len(line_starts) > 1 else text_len
        return min(char, line_end)
    if line >= len(line_starts):
        return text_len
    start = line_starts[line]
    next_start = line_starts[line + 1] if line + 1 < len(line_starts) else text_len + 1
    line_len = next_start - start - 1  # exclude the newline
    return min(start + char, start + line_len)


def _position_to_offset(text: str, pos: Position) -> int:
    """Convert a Position in *text* to a clamped character offset.

    For single-call use.  When the same *text* will be converted many times
    (e.g. once per assignment in a 300+ parameter model), call
    :func:`_build_line_starts` once and use
    :func:`_position_to_offset_with_index` directly to avoid the repeated
    ``text.split("\\n")`` overhead.
    """
    if pos.line <= 0:
        line_text = text.split("\n", 1)[0] if text else ""
        return min(max(pos.character, 0), len(line_text))

    offset = 0
    lines = text.split("\n")
    if pos.line >= len(lines):
        return len(text)
    for line in lines[: pos.line]:
        offset += len(line) + 1
    return min(offset + max(pos.character, 0), offset + len(lines[pos.line]))


def _offset_to_position_local(text: str, offset: int) -> Position:
    """Convert a clamped character offset into a Position."""
    offset = max(0, min(offset, len(text)))
    line = text.count("\n", 0, offset)
    last_nl = text.rfind("\n", 0, offset)
    character = offset if last_nl == -1 else offset - last_nl - 1
    return Position(line, character)


def _ranges_nested_or_equal(outer: SourceRange, inner: SourceRange) -> bool:
    outer_start = (outer.start.line, outer.start.character)
    outer_end = (outer.end.line, outer.end.character)
    inner_start = (inner.start.line, inner.start.character)
    inner_end = (inner.end.line, inner.end.character)
    return outer_start <= inner_start and inner_end <= outer_end


def _map_diagnostic_to_original_source(
    model: ParsedModel,
    diagnostic: Diagnostic,
) -> Diagnostic:
    """Map substituted-macro diagnostic ranges back to the user's source."""
    source_map = getattr(model, "source_map", None)
    original_text = getattr(model, "original_text", "") or model.text
    if not source_map:
        return diagnostic

    macro_spans = [
        (m.start(), m.end())
        for m in re.finditer(r"@\{[A-Za-z_][A-Za-z0-9_]*\}", original_text)
    ]

    def _map_pos(pos: Position) -> Position:
        offset = _position_to_offset(model.text, pos)
        if offset >= len(source_map):
            mapped = source_map[-1]
        else:
            mapped = source_map[offset]
        return _offset_to_position_local(original_text, mapped)

    def _map_range(rng: SourceRange) -> SourceRange:
        return SourceRange(_map_pos(rng.start), _map_pos(rng.end))

    def _range_overlaps_macro_interpolation(start: Position, end: Position) -> bool:
        if not macro_spans:
            return False
        start_offset = _position_to_offset(original_text, start)
        end_offset = _position_to_offset(original_text, end)
        if start_offset == end_offset:
            return any(s < start_offset < e for s, e in macro_spans)
        return any(start_offset < e and s < end_offset for s, e in macro_spans)

    fix = diagnostic.fix
    mapped_fix = None
    if fix is not None:
        start = _map_pos(Position(fix.start_line, fix.start_char))
        end = _map_pos(Position(fix.end_line, fix.end_char))
        # A diagnostic produced from generated macro text may map back
        # onto the user's ``@{NAME}`` interpolation.  Replacing that span
        # with the generated identifier would erase the macro use, so keep
        # the diagnostic but suppress the edit.
        if not _range_overlaps_macro_interpolation(start, end):
            mapped_fix = TextEdit(
                start_line=start.line,
                start_char=start.character,
                end_line=end.line,
                end_char=end.character,
                new_text=fix.new_text,
            )

    return Diagnostic(
        range=_map_range(diagnostic.range),
        severity=diagnostic.severity,
        message=diagnostic.message,
        source=diagnostic.source,
        code=diagnostic.code,
        fix=mapped_fix,
    )


# ---------------------------------------------------------------------------
# Identifier extraction from equations
# ---------------------------------------------------------------------------

# Identifiers that are built-in functions or operators, not variables.
# Membership is checked via :func:`_is_builtin` below, which lower-cases
# the candidate before lookup — Dynare's preprocessor is documented as
# not case-sensitive, so ``EXPECTATION``, ``Expectation``, and
# ``expectation`` all refer to the same operator.
_BUILTINS = frozenset(
    {
        # Math functions
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
        "normpdf",
        "normcdf",
        "norminv",
        "logncdf",
        "erf",
        "erfc",
        # Constants
        "pi",
        "inf",
        "nan",
        # Dynare operators / keywords used inside equations
        "steady_state",
        "expectation",
        "pac_expectation",
        "diff",
        "adl",
    }
)
_TIMED_IDENTIFIER_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*[+-]?\s*\d+\s*\)")
_TIMED_BUILTIN_FUNCTIONS = frozenset(
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
        "normpdf",
        "normcdf",
        "norminv",
        "logncdf",
        "erf",
        "erfc",
        "steady_state",
        "expectation",
        "pac_expectation",
    }
)


def _is_builtin(name: str) -> bool:
    """Case-insensitive membership check against ``_BUILTINS``."""
    return name.lower() in _BUILTINS


_RESERVED_BLOCK_KEYWORDS = frozenset(
    {
        "var",
        "varexo",
        "varexo_det",
        "parameters",
        "predetermined_variables",
        "model",
        "end",
        "initval",
        "endval",
        "shocks",
        "steady_state_model",
        "estimated_params",
        "varobs",
    }
)


# Names that this LSP treats as math constants/functions when evaluating
# expressions, but that Dynare does NOT reserve in its lexer and therefore
# permits as user identifiers.  ``pi`` is the canonical case: it is the
# inflation variable across the New-Keynesian literature, and Dynare has no
# built-in ``pi`` constant, so declaring ``var pi;`` is valid.
_DECLARABLE_BUILTINS = {"pi"}

# Function-like operators Dynare's lexer reserves inside model expressions:
# declaring or assigning them parses as a malformed function call
# ("syntax error, unexpected EQUAL, expecting '('").
_EXPRESSION_OPERATOR_RESERVED = {"var_expectation", "pac_target_nonstationary"}


def _reserved_identifier_reason(name: str) -> Optional[str]:
    lowered = name.lower()
    if lowered in _DECLARABLE_BUILTINS:
        return None
    if lowered in _BUILTINS:
        return "Dynare built-in function or operator"
    if lowered in _EXPRESSION_OPERATOR_RESERVED:
        return "Dynare reserved model-expression operator"
    if lowered in _DYNARE_COMMANDS or lowered in _RESERVED_BLOCK_KEYWORDS:
        return "Dynare command or block keyword"
    return None


# ---------------------------------------------------------------------------
# Auto-fix helpers
# ---------------------------------------------------------------------------


def _find_declaration_insert_point(
    text: str, keyword: str, model: ParsedModel
) -> Optional[TextEdit]:
    """Find where to insert a new name into a var/varexo/parameters declaration.

    Returns a TextEdit that inserts " name" before the semicolon of the declaration.
    The caller should set new_text to " <name>" (with leading space).
    """
    lines = text.split("\n")
    # Find the declaration keyword line
    decl_vars = {
        "var": model.endogenous,
        "varexo": model.exogenous,
        "parameters": model.parameters,
    }
    vars_list = decl_vars.get(keyword, [])
    if not vars_list:
        # Declaration doesn't exist yet — can't auto-fix
        return None

    # For ``varexo``, model.exogenous merges both ``varexo`` and
    # ``varexo_det`` declarations.  An auto-fix insertion should land
    # in a real ``varexo`` declaration, NOT in a ``varexo_det`` line.
    # Walk backward from each variable's line to find the introducing
    # keyword; keep only variables whose owning declaration is ``varexo``
    # (and not ``varexo_det``).
    if keyword == "varexo":

        def _belongs_to_varexo(v: VarDeclaration) -> bool:
            start = v.range.start.line
            for k in range(min(start, len(lines) - 1), -1, -1):
                line = lines[k]
                m_det = re.search(r"\bvarexo_det\b", line, re.IGNORECASE)
                m_exo = re.search(r"\bvarexo\b(?!_det)", line, re.IGNORECASE)
                if m_det and (not m_exo or m_det.start() < m_exo.start()):
                    return False
                if m_exo:
                    return True
            return False

        filtered = [v for v in vars_list if _belongs_to_varexo(v)]
        if filtered:
            vars_list = filtered
        else:
            # No real ``varexo`` declaration exists — only ``varexo_det``.
            # Falling back to the unfiltered list would anchor the
            # insertion on a ``varexo_det`` line, contradicting the
            # diagnostic's "add to varexo" guidance.  Return None so
            # the caller decides (typically: synthesize a new varexo
            # declaration at the top).
            return None

    # Find the last declared variable in this declaration to locate the semicolon
    last_var = vars_list[-1]
    last_line = last_var.range.end.line

    # Search from last_line forward for the semicolon
    for i in range(last_line, min(last_line + 3, len(lines))):
        line = lines[i]
        # Strip comments (// and %).  Without the % case, ``var y % see
        # below ;\nz;`` would have the semicolon found INSIDE the
        # comment and an auto-inserted name would land mid-comment.
        line_no_comment = re.sub(r"(?://|%).*$", "", line)
        semi_idx = line_no_comment.find(";")
        if semi_idx >= 0:
            return TextEdit(
                start_line=i,
                start_char=semi_idx,
                end_line=i,
                end_char=semi_idx,
                new_text="",  # caller fills in
            )
    return None


def _find_name_in_declaration(
    text: str,
    keyword: str,
    name: str,
    model: ParsedModel,
    target_range: Optional[SourceRange] = None,
) -> Optional[TextEdit]:
    """Find a name in a var/varexo/parameters declaration and return a TextEdit to remove it."""
    lines = text.split("\n")
    decl_vars = {
        "var": model.endogenous,
        "varexo": model.exogenous,
        "varexo_det": model.deterministic_exogenous,
        "parameters": model.parameters,
    }
    vars_list = decl_vars.get(keyword, [])
    decl_keywords = ("varexo", "varexo_det") if keyword == "varexo" else (keyword,)

    def _empty_declaration_remainder(value: str) -> bool:
        """True when only a declaration keyword/options would remain."""
        return any(
            re.fullmatch(rf"{re.escape(kw)}(?:\s*\([^)]*\))?", value, re.IGNORECASE)
            for kw in decl_keywords
        )

    def _line_starts_declaration(value: str) -> bool:
        return any(
            re.match(rf"\s*{re.escape(kw)}\b", value, re.IGNORECASE)
            for kw in decl_keywords
        )

    for v in vars_list:
        if v.name == name:
            if target_range is not None and v.range != target_range:
                continue
            # Found it — remove this variable from the declaration
            line_idx = v.range.start.line
            line = lines[line_idx]
            # Strip comments to work with code only
            line_no_comment = re.sub(r"(?://|%).*$", "", line)
            # Find the variable name in the line
            name_match = re.search(rf"\b{re.escape(name)}\b", line_no_comment)
            if name_match:
                start = name_match.start()
                end = name_match.end()
                # Include leading whitespace
                while start > 0 and line_no_comment[start - 1] in " \t":
                    start -= 1
                # Check if removing leaves the line empty (just keyword or empty)
                remaining = line_no_comment[:start] + line_no_comment[end:]
                remaining_stripped = remaining.strip().rstrip(";").strip()
                # If the line would be just the keyword or empty, the
                # declaration becomes invalid (``var;`` / ``varexo;`` is
                # not legal Dynare).  Two cases:
                #
                # 1) The keyword is on THIS line — the entire ``keyword
                #    name ;`` declaration must be removed, not just the
                #    name.  Find the trailing semicolon on this line so
                #    we delete the whole declaration as a unit.
                # 2) The keyword is on a PRIOR line (multi-line decl) —
                #    just removing this var-only line is fine, but we
                #    must also leave the prior line's keyword intact;
                #    the prior line still has other names.  Old code
                #    handles this correctly.
                if not remaining_stripped or _empty_declaration_remainder(
                    remaining_stripped
                ):
                    if _line_starts_declaration(line):
                        # Remove the entire declaration: from start of
                        # the line through the trailing ``;`` (which we
                        # locate via the comment-stripped view).
                        semi_idx = line_no_comment.find(";")
                        if semi_idx >= 0:
                            return TextEdit(
                                start_line=line_idx,
                                start_char=0,
                                end_line=line_idx + 1,
                                end_char=0,
                                new_text="",
                            )
                        # No semicolon on this line — fall back to old
                        # behavior of removing just the name.
                        return TextEdit(
                            start_line=line_idx,
                            start_char=start,
                            end_line=line_idx,
                            end_char=end,
                            new_text="",
                        )
                    else:
                        # Line only has this variable — remove entire line
                        return TextEdit(
                            start_line=line_idx,
                            start_char=0,
                            end_line=line_idx + 1,
                            end_char=0,
                            new_text="",
                        )
                else:
                    return TextEdit(
                        start_line=line_idx,
                        start_char=start,
                        end_line=line_idx,
                        end_char=end,
                        new_text="",
                    )
    return None


def _extract_references(
    eq_text: str,
    declared_names: Optional[set] = None,
) -> List[str]:
    """Extract variable/parameter identifiers referenced in an equation."""
    declared_names = declared_names or set()
    # Remove entire equation tag blocks [name='...', mcp='...']
    cleaned = re.sub(r"\[[^\]]*\]", "", eq_text)
    # Remove string literals
    cleaned = re.sub(r"'[^']*'", "", cleaned)
    cleaned = re.sub(r'"[^"]*"', "", cleaned)
    timed_identifiers = {
        name
        for name in _TIMED_IDENTIFIER_RE.findall(cleaned)
        if name.lower() not in _TIMED_BUILTIN_FUNCTIONS
    }
    # Find all identifiers (must start with a letter, not just underscore)
    ids = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", cleaned)
    # Filter builtins and name-like keywords
    return [
        i
        for i in ids
        if (
            (not _is_builtin(i) or i in declared_names or i in timed_identifiers)
            and i.lower() != "end"
        )
    ]


def _extract_equation_references(
    equation,
    declared_names: Optional[set] = None,
) -> List[str]:
    """Extract references from equation text and MCP constraint tags."""
    refs = list(_extract_references(equation.text, declared_names))
    for constraint in getattr(equation, "mcp_constraints", []):
        refs.extend(_extract_references(constraint, declared_names))
    return refs


_STRING_LITERAL_RE = re.compile(r"\"[^\"\n]*\"|'[^'\n]*'")
_MCP_TAG_VALUE_RE = re.compile(r"\bmcp\s*=\s*(['\"])(.*?)\1", re.IGNORECASE)
_TAG_ATTRIBUTE_KEY_RE = re.compile(r"\b(?:name|mcp)\b(?=\s*=)", re.IGNORECASE)


def _inside_quoted_string(text: str, offset: int) -> bool:
    quote: Optional[str] = None
    for ch in text[:offset]:
        if quote is not None:
            if ch == quote:
                quote = None
            continue
        if ch in ('"', "'"):
            quote = ch
    return quote is not None


def _mask_strings_preserving_mcp_values(text: str) -> str:
    """Mask quoted strings, except the expression payload of ``mcp=`` tags."""
    keep = [False] * len(text)
    for match in _MCP_TAG_VALUE_RE.finditer(text):
        if _inside_quoted_string(text, match.start()):
            continue
        for idx in range(match.start(2), match.end(2)):
            keep[idx] = True

    chars = list(text)
    for match in _STRING_LITERAL_RE.finditer(text):
        for idx in range(match.start(), match.end()):
            if not keep[idx] and chars[idx] != "\n":
                chars[idx] = " "
    for tag_start, tag_end in _iter_equation_tag_spans(text):
        tag_text = text[tag_start:tag_end]
        for key in _TAG_ATTRIBUTE_KEY_RE.finditer(tag_text):
            start = tag_start + key.start()
            end = tag_start + key.end()
            for idx in range(start, end):
                chars[idx] = " "
    return "".join(chars)


def _mask_non_code_for_reference_search(text: str) -> str:
    """Mask comments and strings while preserving source offsets."""

    def _blank(m: re.Match) -> str:
        return re.sub(r"[^\n]", " ", m.group(0))

    masked = _mask_strings_preserving_mcp_values(text)
    masked = re.sub(r"/\*.*?\*/", _blank, masked, flags=re.DOTALL)
    masked = re.sub(r"(?://|%).*$", _blank, masked, flags=re.MULTILINE)
    return masked


def _mask_macro_directive_lines(text: str) -> str:
    """Blank macro directive lines while preserving source offsets."""
    lines: List[str] = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        newline = line[len(body) :]
        if body.lstrip().startswith("@#"):
            lines.append((" " * len(body)) + newline)
        else:
            lines.append(line)
    return "".join(lines)


def _find_reference_range_in_equation(
    text: str,
    equation_range: SourceRange,
    ref: str,
) -> Optional[SourceRange]:
    """Return the first code occurrence of *ref* within an equation range."""
    if not ref:
        return None
    masked_lines = _mask_non_code_for_reference_search(text).split("\n")
    if equation_range.start.line >= len(masked_lines):
        return None
    last_line = min(equation_range.end.line, len(masked_lines) - 1)
    pattern = re.compile(r"\b" + re.escape(ref) + r"\b")
    for line_idx in range(equation_range.start.line, last_line + 1):
        line_masked = masked_lines[line_idx]
        start_char = (
            equation_range.start.character
            if line_idx == equation_range.start.line
            else 0
        )
        end_char = (
            equation_range.end.character
            if line_idx == equation_range.end.line
            else len(line_masked)
        )
        if end_char < start_char:
            continue
        match = pattern.search(line_masked[start_char:end_char])
        if match is None:
            continue
        start = Position(line_idx, start_char + match.start())
        end = Position(line_idx, start_char + match.end())
        return SourceRange(start, end)
    return None


def _find_timed_identifier_ranges_in_equation(
    text: str,
    equation_range: SourceRange,
) -> List[Tuple[str, SourceRange]]:
    """Return timed identifier occurrences within an equation range."""
    masked_lines = _mask_non_code_for_reference_search(text).split("\n")
    if equation_range.start.line >= len(masked_lines):
        return []
    occurrences: List[Tuple[str, SourceRange]] = []
    last_line = min(equation_range.end.line, len(masked_lines) - 1)
    for line_idx in range(equation_range.start.line, last_line + 1):
        line_masked = masked_lines[line_idx]
        start_char = (
            equation_range.start.character
            if line_idx == equation_range.start.line
            else 0
        )
        end_char = (
            equation_range.end.character
            if line_idx == equation_range.end.line
            else len(line_masked)
        )
        if end_char < start_char:
            continue
        segment = line_masked[start_char:end_char]
        for match in _TIMED_IDENTIFIER_RE.finditer(segment):
            name = match.group(1)
            if name.lower() in _TIMED_BUILTIN_FUNCTIONS:
                continue
            occurrences.append(
                (
                    name,
                    SourceRange(
                        Position(line_idx, start_char + match.start()),
                        Position(line_idx, start_char + match.end()),
                    ),
                )
            )
    return occurrences


def _find_timed_identifier_ranges_in_equation_text(
    equation_text: str,
    equation_range: SourceRange,
    raw_equation_text: Optional[str] = None,
) -> List[Tuple[str, SourceRange]]:
    """Return timed identifiers using equation-local text and source anchor."""
    masked = _mask_non_code_for_reference_search(equation_text)
    base = Position(0, 0)
    if raw_equation_text:
        raw_start = raw_equation_text.find(equation_text)
        if raw_start >= 0:
            base = _offset_to_position_local(raw_equation_text, raw_start)
    occurrences: List[Tuple[str, SourceRange]] = []
    for match in _TIMED_IDENTIFIER_RE.finditer(masked):
        name = match.group(1)
        if name.lower() in _TIMED_BUILTIN_FUNCTIONS:
            continue
        local_start = _offset_to_position_local(equation_text, match.start())
        local_end = _offset_to_position_local(equation_text, match.end())

        def _anchor(pos: Position) -> Position:
            anchored_line = equation_range.start.line + base.line + pos.line
            if base.line + pos.line == 0:
                anchored_char = (
                    equation_range.start.character + base.character + pos.character
                )
            elif pos.line == 0:
                anchored_char = base.character + pos.character
            else:
                anchored_char = pos.character
            return Position(
                anchored_line,
                anchored_char,
            )

        occurrences.append(
            (
                name,
                SourceRange(_anchor(local_start), _anchor(local_end)),
            )
        )
    return occurrences


# ---------------------------------------------------------------------------
# Individual diagnostic checks
# ---------------------------------------------------------------------------


def _check_parse_errors(model: ParsedModel) -> List[Diagnostic]:
    """Convert parse errors into diagnostics."""
    diagnostics = []
    for err in model.errors:
        msg, rng = err[0], err[1]
        fix_dict = err[2] if len(err) > 2 else None
        fix = None
        if fix_dict:
            fix = TextEdit(
                start_line=fix_dict["start_line"],
                start_char=fix_dict["start_char"],
                end_line=fix_dict["end_line"],
                end_char=fix_dict["end_char"],
                new_text=fix_dict["new_text"],
            )
        diagnostics.append(
            Diagnostic(
                range=rng,
                severity=Severity.ERROR,
                message=msg,
                code="E001",
                fix=fix,
            )
        )
    return diagnostics


def _missing_end_block_kind(message: str) -> Optional[str]:
    match = re.search(r"Missing 'end;' for '([A-Za-z_][A-Za-z0-9_]*)' block", message)
    return match.group(1) if match else None


def _include_closes_block(
    include_models: Optional[List[ParsedModel]],
    block_kind: str,
) -> bool:
    return any(
        getattr(include_model, "context_closing_block", None) == block_kind
        for include_model in include_models or []
    )


def _filter_include_closed_parse_errors(
    parse_errors: List[Diagnostic],
    include_models: Optional[List[ParsedModel]],
) -> List[Diagnostic]:
    if not include_models:
        return parse_errors
    filtered: List[Diagnostic] = []
    for diagnostic in parse_errors:
        block_kind = _missing_end_block_kind(diagnostic.message)
        if block_kind and _include_closes_block(include_models, block_kind):
            continue
        filtered.append(diagnostic)
    return filtered


def _check_included_parse_errors(
    include_models: Optional[List[ParsedModel]],
) -> List[Diagnostic]:
    diagnostics: List[Diagnostic] = []
    for include_model in include_models or []:
        for diagnostic in _check_parse_errors(include_model):
            diagnostics.append(
                replace(
                    diagnostic,
                    message=f"Included file error: {diagnostic.message}",
                    fix=None,
                )
            )
    return diagnostics


_DECL_INVALID_IDENT_PATTERN = re.compile(
    r"(?<!\w)(var|varexo_det|varexo|parameters|predetermined_variables)"
    r"\b(?:\s*\([^)]*\))?\s+([^;]*);",
    re.IGNORECASE | re.DOTALL,
)


def _check_invalid_identifier_declarations(model: ParsedModel) -> List[Diagnostic]:
    diagnostics: List[Diagnostic] = []
    # Use ``model.text`` rather than ``original_text`` because the parser
    # blanks known-inactive macro branches there while preserving offsets.
    text = model.text or getattr(model, "original_text", "")

    # Reuse the doubly-masked text and block-exclusion ranges if already
    # computed for this model instance (e.g. on repeated keystroke calls).
    # The masking is expensive (~4.5 ms) on large models such as US_FRB03.
    if model._cached_diag_stripped is None:
        model._cached_diag_stripped = _mask_macro_directive_lines(
            _mask_non_code_for_reference_search(text),
        )
    if model._cached_diag_block_exclusions is None:
        # _block_ranges (regex over the masked text) finds ALL blocks -- crucially
        # including REPEATED shocks/estimated_params blocks that the single-value
        # model.*_block_range attributes miss.  Missing a second shocks block
        # caused false E001s (e.g. Ascari_Sbordone_2014, Gali_2015_chapter_4,
        # which have a "var e_a = 0.45^2;" shock-variance line outside the first
        # shocks block).  Cached per ParsedModel, so it still runs only once.
        model._cached_diag_block_exclusions = _block_ranges(
            model._cached_diag_stripped,
        )
    stripped = model._cached_diag_stripped
    block_exclusions = model._cached_diag_block_exclusions
    for match in _DECL_INVALID_IDENT_PATTERN.finditer(stripped):
        if _inside_block(match.start(), block_exclusions):
            continue
        body = match.group(2)
        masked_body = re.sub(r"\$[^$\n]*\$", lambda m: " " * len(m.group(0)), body)
        masked_body = re.sub(
            r"\([^()]*\)", lambda m: " " * len(m.group(0)), masked_body
        )
        if re.search(
            r"(?<!\w)(?:varexo_det|var|varexo|parameters|"
            r"predetermined_variables|model|initval|endval|"
            r"shocks|steady_state_model)\b",
            masked_body,
            re.IGNORECASE,
        ):
            continue
        for bad in re.finditer(r"[^,\s]+", masked_body):
            token = bad.group(0)
            if "@{" in token:
                continue
            if not re.search(r"[A-Za-z_]", token):
                continue
            if re.match(r"^[A-Za-z][A-Za-z0-9_]*$", token):
                continue
            start = match.start(2) + bad.start()
            end = match.start(2) + bad.end()
            diagnostics.append(
                Diagnostic(
                    range=SourceRange(
                        _offset_to_position_local(text, start),
                        _offset_to_position_local(text, end),
                    ),
                    severity=Severity.ERROR,
                    message=(
                        f"Invalid Dynare identifier '{token}'. "
                        "Identifiers must start with a letter and contain only "
                        "letters, digits, and underscores."
                    ),
                    code="E001",
                )
            )
    declarations = (
        model.endogenous
        + model.exogenous
        + model.deterministic_exogenous
        + model.parameters
        + model.predetermined_variables
    )
    seen_reserved: set[str] = set()
    for declaration in declarations:
        reason = _reserved_identifier_reason(declaration.name)
        if reason is None or declaration.name.lower() in seen_reserved:
            continue
        line = text.split("\n")[declaration.range.start.line]
        leading = len(line) - len(line.lstrip())
        if declaration.range.start.character == leading:
            continue
        seen_reserved.add(declaration.name.lower())
        diagnostics.append(
            Diagnostic(
                range=declaration.range,
                severity=Severity.ERROR,
                message=(
                    f"Invalid Dynare identifier '{declaration.name}': reserved "
                    f"{reason.lower()}. Choose a different name."
                ),
                code="E001",
            )
        )
    return diagnostics


def _check_predetermined_variables(
    model: ParsedModel,
    endogenous_names: Optional[set] = None,
) -> List[Diagnostic]:
    """Check that predetermined variables are declared endogenous names."""
    diagnostics: List[Diagnostic] = []
    endo_names = (
        endogenous_names if endogenous_names is not None else model.endogenous_names()
    )
    seen: set = set()
    for var in model.predetermined_variables:
        if var.name in seen:
            continue
        seen.add(var.name)
        if var.name not in endo_names:
            diagnostics.append(
                Diagnostic(
                    range=var.range,
                    severity=Severity.ERROR,
                    message=(
                        f"Predetermined variable '{var.name}' is not declared as an "
                        f"endogenous variable. Fix: add '{var.name}' to the 'var' "
                        f"declaration or remove it from 'predetermined_variables'."
                    ),
                    code="E023",
                )
            )
    return diagnostics


def _check_timed_parameters(model: ParsedModel) -> List[Diagnostic]:
    """Dynare accepts parameter time subscripts and treats them as no-ops."""
    return []


def _check_timed_deterministic_exogenous(model: ParsedModel) -> List[Diagnostic]:
    """Reject leads/lags applied to deterministic exogenous variables."""
    diagnostics: List[Diagnostic] = []
    det_exo_names = model.deterministic_exogenous_names()
    if not det_exo_names:
        return diagnostics

    seen: set[str] = set()
    for eq in model.model_equations:
        start = _position_to_offset(model.text, eq.range.start)
        end = _position_to_offset(model.text, eq.range.end)
        raw_equation_text = model.text[start:end] if start < end else None
        for name, rng in _find_timed_identifier_ranges_in_equation_text(
            eq.text,
            eq.range,
            raw_equation_text,
        ):
            if name not in det_exo_names or name in seen:
                continue
            seen.add(name)
            diagnostics.append(
                Diagnostic(
                    range=rng,
                    severity=Severity.ERROR,
                    message=(
                        f"Deterministic exogenous variable '{name}' cannot be "
                        "used with a lead or lag. Fix: remove the time subscript "
                        f"from '{name}'."
                    ),
                    code="E024",
                )
            )
    return diagnostics


def _check_model_local_shadowing(model: ParsedModel) -> List[Diagnostic]:
    """Reject model-local ``#`` names that shadow declared Dynare symbols."""
    diagnostics: List[Diagnostic] = []
    declared = model.all_declared_names()
    if not declared:
        return diagnostics

    seen: set[str] = set()
    for eq in list(model.model_equations) + list(model.steady_state_equations):
        match = re.match(r"#\s*([A-Za-z_][A-Za-z0-9_]*)", eq.text.strip())
        if not match:
            continue
        name = match.group(1)
        if name not in declared or name in seen:
            continue
        seen.add(name)
        rng = _find_reference_range_in_equation(model.text, eq.range, name) or eq.range
        diagnostics.append(
            Diagnostic(
                range=rng,
                severity=Severity.ERROR,
                message=(
                    f"Model-local variable '{name}' shadows a declared Dynare "
                    "symbol. Fix: rename the model-local variable or remove the "
                    "duplicate declaration."
                ),
                code="E025",
            )
        )
    return diagnostics


def _check_equation_count(model: ParsedModel) -> List[Diagnostic]:
    """Check that number of equations matches number of endogenous variables."""
    diagnostics: List[Diagnostic] = []

    if model.model_block_range is None and not model.model_equations:
        return diagnostics

    real_equations = model.dynamic_model_equations()
    n_eq = len(real_equations)
    n_endo = len(model.endogenous)
    if n_eq == 0 and n_endo == 0:
        return diagnostics

    if n_eq != n_endo:
        rng = model.model_block_range
        if rng is None:
            rng = SourceRange(Position(0, 0), Position(0, 1))

        if n_eq > n_endo:
            fix_msg = (
                f"Fix: remove {n_eq - n_endo} duplicate/extra equation(s) from the model block, "
                f"or add {n_eq - n_endo} missing variable(s) to the 'var' declaration."
            )
        else:
            fix_msg = (
                f"Fix: add {n_endo - n_eq} missing equation(s) to the model block, "
                f"or remove {n_endo - n_eq} extra variable(s) from the 'var' declaration."
            )
        diagnostics.append(
            Diagnostic(
                range=rng,
                severity=Severity.ERROR if abs(n_eq - n_endo) > 0 else Severity.WARNING,
                message=(
                    f"Equation count mismatch: {n_eq} equation(s) "
                    f"but {n_endo} endogenous variable(s). "
                    f"{fix_msg}"
                ),
                code="E010",
            )
        )

    return diagnostics


def _find_similar_names(name: str, declared: set, max_results: int = 3) -> List[str]:
    """Find declared names similar to the given undeclared name."""
    candidates = []
    name_lower = name.lower()
    for d in sorted(declared):
        d_lower = d.lower()
        # Check for common typo patterns
        if d_lower == name_lower:
            continue  # exact match (shouldn't happen)
        # Levenshtein-like: check if names differ by 1-2 characters
        if abs(len(d) - len(name)) <= 2:
            # Simple character-diff count
            diffs = sum(1 for a, b in zip(name_lower, d_lower) if a != b)
            diffs += abs(len(name) - len(d))
            if diffs <= 2:
                candidates.append((diffs, d))
        # Also check prefix match (e.g., 'rhoe' vs 'rho')
        elif name_lower.startswith(d_lower) or d_lower.startswith(name_lower):
            candidates.append((abs(len(name) - len(d)), d))
    candidates.sort()
    return [c[1] for c in candidates[:max_results]]


def _check_undeclared_references(
    model: ParsedModel,
    include_symbols: Optional[Dict[str, List[VarDeclaration]]] = None,
) -> List[Diagnostic]:
    """Check that all identifiers in model equations are declared.

    When *include_symbols* is provided, names declared in transitively
    included files are treated as visible (i.e. they don't trigger
    E020).  The include-visible names are not used for typo suggestions
    because the editor user can't always jump to the declaration in an
    unopened file; suggestions stay focused on local declarations.
    """
    diagnostics: List[Diagnostic] = []
    local_declared = model.all_declared_names()
    declared = set(local_declared)
    if include_symbols is not None:
        for kind in ("endogenous", "exogenous", "parameters"):
            for v in include_symbols.get(kind, []):
                declared.add(v.name)
    shocks_var_set = set(model.shocks_vars)
    # Names with value assignments (param or helper) → likely parameters
    assigned_names = {a.name for a in model.param_assignments} | {
        a.name for a in model.helper_assignments
    }

    # Model-local variables are scoped to the whole model block, not only
    # equations textually after the ``#`` definition.
    local_vars: set = set()
    for eq in model.model_equations:
        match = re.match(r"#\s*([A-Za-z_][A-Za-z0-9_]*)", eq.text.strip())
        if match:
            local_vars.add(match.group(1))
    seen_undeclared: set = set()  # avoid duplicate reports for same identifier

    for eq in model.model_equations:
        refs = _extract_equation_references(eq)
        for ref in refs:
            if ref not in declared and ref not in local_vars:
                ref_range = _find_reference_range_in_equation(
                    model.text,
                    eq.range,
                    ref,
                )
                diagnostic_range = ref_range or eq.range
                # Skip identifiers that look like macro-expansion fragments
                # (lone underscore or double-underscore suffix from @{...})
                if ref == "_" or ref.endswith("__"):
                    continue
                # Only report each undeclared identifier once
                if ref in seen_undeclared:
                    continue
                seen_undeclared.add(ref)

                # Build actionable message with similar name suggestions
                msg = f"Undeclared identifier '{ref}' in equation"
                if eq.name:
                    msg += f" '{eq.name}'"
                msg += "."

                similar = _find_similar_names(ref, local_declared)
                if similar:
                    msg += f" Did you mean: {', '.join(similar)}?"

                # Determine the most likely declaration type
                # Check if naming pattern matches exogenous variables
                exo_names = {v.name for v in model.exogenous}
                ref_matches_exo_pattern = False
                if ref.endswith("_") and any(e.endswith("_") for e in exo_names):
                    ref_matches_exo_pattern = True

                # Determine target declaration and build fix
                target_decl = None
                fix = None
                is_typo_replace = False

                # First: check if the identifier is in the shocks block or
                # has a value assignment — these are definite declarations,
                # not typos. Only try typo-replace if neither applies.
                definite_decl = (
                    ref in shocks_var_set
                    or ref in assigned_names
                    or ref_matches_exo_pattern
                )

                # Typo-replace: if there's a very close match (edit distance ≤ 2)
                # and the identifier isn't known to be a real name, replace it.
                if similar and not definite_decl:
                    best = similar[0]  # sorted by edit distance
                    diffs = sum(1 for a, b in zip(ref.lower(), best.lower()) if a != b)
                    diffs += abs(len(ref) - len(best))
                    if diffs <= 2:
                        is_typo_replace = True
                        msg = f"Undeclared identifier '{ref}' in equation"
                        if eq.name:
                            msg += f" '{eq.name}'"
                        msg += (
                            f". This is likely a typo for '{best}'. "
                            f"Fix: replace '{ref}' with '{best}'."
                        )
                        if ref_range is not None:
                            fix = TextEdit(
                                start_line=ref_range.start.line,
                                start_char=ref_range.start.character,
                                end_line=ref_range.end.line,
                                end_char=ref_range.end.character,
                                new_text=best,
                            )

                if is_typo_replace:
                    pass  # fix already set above
                elif ref in shocks_var_set:
                    target_decl = "varexo"
                    msg += (
                        f" Since '{ref}' is referenced in the shocks block, "
                        f"it is an exogenous variable. "
                        f"Fix: add '{ref}' to the 'varexo' declaration."
                    )
                elif ref_matches_exo_pattern:
                    target_decl = "varexo"
                    msg += (
                        f" Since '{ref}' follows the naming pattern of exogenous "
                        f"shock variables (like {', '.join(sorted(exo_names)[:2])}), "
                        f"it is likely an exogenous variable. "
                        f"Fix: add '{ref}' to the 'varexo' declaration."
                    )
                elif ref in assigned_names:
                    target_decl = "parameters"
                    msg += (
                        f" Since '{ref}' has a value assignment in the file, "
                        f"it is likely a parameter. "
                        f"Fix: add '{ref}' to the 'parameters' declaration."
                    )
                elif similar:
                    # If all similar names are in the same declaration type,
                    # classify this identifier as the same type
                    sim_in_exo = all(
                        s in {v.name for v in model.exogenous} for s in similar
                    )
                    sim_in_endo = all(
                        s in {v.name for v in model.endogenous} for s in similar
                    )
                    sim_in_params = all(
                        s in {p.name for p in model.parameters} for s in similar
                    )
                    if sim_in_exo:
                        target_decl = "varexo"
                        msg += (
                            f" Similar declared names ({', '.join(similar)}) are "
                            f"all exogenous variables. "
                            f"Fix: add '{ref}' to the 'varexo' declaration."
                        )
                    elif sim_in_params:
                        target_decl = "parameters"
                        msg += (
                            f" Similar declared names ({', '.join(similar)}) are "
                            f"all parameters. "
                            f"Fix: add '{ref}' to the 'parameters' declaration."
                        )
                    elif sim_in_endo:
                        target_decl = "var"
                        msg += (
                            f" Similar declared names ({', '.join(similar)}) are "
                            f"all endogenous variables. "
                            f"Fix: add '{ref}' to the 'var' declaration."
                        )
                    else:
                        msg += f" Fix: replace '{ref}' with the correct name, or add '{ref}' to a var/varexo/parameters declaration."
                else:
                    msg += f" Fix: add '{ref}' to a var, varexo, or parameters declaration."

                # Compute auto-fix if we have a confident target
                # (skip if typo replacement already set the fix)
                if target_decl and not is_typo_replace:
                    insert_edit = _find_declaration_insert_point(
                        model.text, target_decl, model
                    )
                    if insert_edit:
                        fix = TextEdit(
                            start_line=insert_edit.start_line,
                            start_char=insert_edit.start_char,
                            end_line=insert_edit.end_line,
                            end_char=insert_edit.end_char,
                            new_text=f" {ref}",
                        )

                diagnostics.append(
                    Diagnostic(
                        range=diagnostic_range,
                        severity=Severity.ERROR,
                        message=msg,
                        code="E020",
                        fix=fix,
                    )
                )

    return diagnostics


def _check_unassigned_parameters(model: ParsedModel) -> List[Diagnostic]:
    """Warn about parameters that are declared but never assigned a value."""
    diagnostics: List[Diagnostic] = []
    assigned = {a.name for a in model.param_assignments}

    # Parameters can also be assigned inside steady_state_model
    for eq in model.steady_state_equations:
        text = eq.text.strip()
        if "=" in text:
            name = text.split("=", 1)[0].strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                assigned.add(name)

    for p in model.parameters:
        if p.name not in assigned:
            diagnostics.append(
                Diagnostic(
                    range=p.range,
                    severity=Severity.WARNING,
                    message=f"Parameter '{p.name}' is declared but never assigned a value.",
                    code="W010",
                )
            )

    return diagnostics


def _check_duplicate_declarations(
    model: ParsedModel,
    include_symbols: Optional[Dict[str, List[VarDeclaration]]] = None,
) -> List[Diagnostic]:
    """Check for variables declared more than once.

    *include_symbols* is accepted for forward compatibility (so cross-file
    duplicate detection can be added later without changing the signature
    again).  Today the check only inspects the local file: cross-file
    duplicates aren't flagged because central-bank model packages
    deliberately re-declare some names per submodel.
    """
    diagnostics: List[Diagnostic] = []
    seen: dict = {}
    branch_signatures = _macro_branch_signatures_by_line(model)
    assigned_params = {a.name for a in model.param_assignments}

    # Check which identifiers appear with time subscripts in model equations
    timed_vars: set = set()
    # Check which identifiers appear on the LHS of model equations (solved-for → endogenous)
    lhs_vars: set = set()
    for eq in model.model_equations:
        for m_timing in re.finditer(
            r"\b([A-Za-z][A-Za-z0-9_]*)\s*\(\s*[+-]?\s*\d+\s*\)", eq.text
        ):
            timed_vars.add(m_timing.group(1))
        # Check if equation has form "varname = ..." (LHS is a single variable)
        lhs = eq.lhs.strip()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", lhs):
            lhs_vars.add(lhs)
    shocks_var_set = set(model.shocks_vars)

    deterministic_keys = {
        (
            v.name,
            v.range.start.line,
            v.range.start.character,
            v.range.end.line,
            v.range.end.character,
        )
        for v in model.deterministic_exogenous
    }
    all_vars = (
        [(v, "var") for v in model.endogenous]
        + [
            (v, "varexo")
            for v in model.exogenous
            if (
                v.name,
                v.range.start.line,
                v.range.start.character,
                v.range.end.line,
                v.range.end.character,
            )
            not in deterministic_keys
        ]
        + [(v, "varexo_det") for v in model.deterministic_exogenous]
        + [(v, "parameters") for v in model.parameters]
    )
    all_vars.sort(
        key=lambda item: (
            item[0].range.start.line,
            item[0].range.start.character,
            item[0].range.end.line,
            item[0].range.end.character,
        )
    )

    for v, kind in all_vars:
        if v.name in seen:
            prev_kind, prev_sig = seen[v.name]
            sig = branch_signatures.get(v.range.start.line, tuple())
            if _macro_signatures_are_mutually_exclusive(prev_sig, sig):
                continue
            if prev_kind != kind:
                # A name in two different declaration kinds (var vs varexo vs
                # parameters …) is a genuine Dynare error.
                severity = Severity.ERROR
                # Try to determine which declaration is likely correct
                hint = ""
                remove_from = None
                if v.name in assigned_params:
                    # Has a parameter assignment → likely should be 'parameters'
                    if prev_kind == "parameters" and kind != "parameters":
                        wrong_kind = kind
                    elif kind == "parameters" and prev_kind != "parameters":
                        wrong_kind = prev_kind
                    else:
                        wrong_kind = "var" if "var" in (prev_kind, kind) else prev_kind
                    remove_from = wrong_kind
                    hint = (
                        f" Since '{v.name}' has a value assignment (like a parameter), "
                        f"it likely belongs in 'parameters'. "
                        f"Fix: remove '{v.name}' from the '{wrong_kind}' declaration."
                    )
                elif v.name in timed_vars and {"var", "varexo"} == {prev_kind, kind}:
                    # Used with time subscripts → likely endogenous (var)
                    remove_from = "varexo"
                    hint = (
                        f" Since '{v.name}' appears with time subscripts in equations "
                        f"(e.g. {v.name}(-1) or {v.name}(+1)), it is likely an endogenous "
                        f"variable. Fix: remove '{v.name}' from the 'varexo' declaration."
                    )
                elif {"var", "varexo"} == {prev_kind, kind}:
                    # var vs varexo: use LHS and shocks heuristics
                    if v.name in lhs_vars and v.name not in shocks_var_set:
                        remove_from = "varexo"
                        hint = (
                            f" Since '{v.name}' appears on the LHS of a model equation "
                            f"(it is solved for), it is likely an endogenous variable. "
                            f"Fix: remove '{v.name}' from the 'varexo' declaration."
                        )
                    elif v.name in shocks_var_set:
                        remove_from = "var"
                        hint = (
                            f" Since '{v.name}' is referenced in the shocks block, "
                            f"it is likely an exogenous variable. "
                            f"Fix: remove '{v.name}' from the 'var' declaration."
                        )
                    else:
                        # Fallback: if the variable appears in model equations
                        # and is not a shock, it is most likely endogenous
                        refs_in_eqs = set()
                        for eq in model.model_equations:
                            refs_in_eqs.update(_extract_equation_references(eq))
                        if v.name in refs_in_eqs:
                            remove_from = "varexo"
                            hint = (
                                f" Since '{v.name}' appears in model equations "
                                f"and is not referenced in the shocks block, "
                                f"it is likely an endogenous variable. "
                                f"Fix: remove '{v.name}' from the 'varexo' declaration."
                            )
                        else:
                            hint = (
                                f" Fix: '{v.name}' must appear in exactly one of "
                                f"'{prev_kind}' or '{kind}'. Remove it from whichever "
                                f"declaration is incorrect."
                            )
                elif {"varexo", "varexo_det"} == {prev_kind, kind}:
                    remove_from = kind
                    hint = (
                        f" Fix: '{v.name}' cannot appear in both stochastic and "
                        f"deterministic exogenous declarations. Remove the "
                        f"duplicate from the '{kind}' declaration."
                    )
                else:
                    hint = (
                        f" Fix: '{v.name}' must appear in exactly one of "
                        f"'{prev_kind}' or '{kind}'. Remove it from whichever "
                        f"declaration is incorrect."
                    )
            else:
                # The same name re-declared in the *same* kind is redundant,
                # which Dynare tolerates (the later wins) -- a warning, not an
                # error.  Remove the second occurrence.
                severity = Severity.WARNING
                remove_from = kind
                hint = (
                    f" Fix: remove the redundant '{v.name}' from the "
                    f"{kind} declaration."
                )
            # Compute auto-fix if we know which declaration to remove from
            fix = None
            if remove_from:
                target_range = v.range if remove_from == kind else None
                fix = _find_name_in_declaration(
                    model.text,
                    remove_from,
                    v.name,
                    model,
                    target_range=target_range,
                )
            if prev_kind == kind:
                message = f"'{v.name}' is declared more than once in '{kind}'.{hint}"
            else:
                message = (
                    f"'{v.name}' is declared in both '{prev_kind}' and '{kind}'.{hint}"
                )
            diagnostics.append(
                Diagnostic(
                    range=v.range,
                    severity=severity,
                    message=message,
                    code="E030",
                    fix=fix,
                )
            )
        else:
            seen[v.name] = (kind, branch_signatures.get(v.range.start.line, tuple()))

    return diagnostics


def _check_steady_state(model: ParsedModel) -> List[Diagnostic]:
    """Validate steady state equations against the model.

    Reports three categories of issues:
      1. Errors computing steady state values (E040)
      2. Equations with non-zero residuals (W041) — the SS is inconsistent
      3. Equations that cannot be evaluated (E041) — missing variables
    Only reports eval failures if there are no missing endogenous variables,
    since missing vars cascade into many downstream failures.
    """
    diagnostics: List[Diagnostic] = []
    diagnostics.extend(_check_steady_state_operator_operands(model))

    report = validate_steady_state(model)
    if report is None:
        return diagnostics

    # Report value computation errors in steady_state_model block
    for var_name, error_msg, rng in report.value_errors:
        diagnostics.append(
            Diagnostic(
                range=rng,
                severity=Severity.WARNING,
                message=f"Steady state could not be verified for '{var_name}': {error_msg}",
                code="E040",
            )
        )

    # Report missing endogenous variables
    if report.missing_endogenous and model.steady_state_equations:
        rng = model.steady_state_block_range
        if rng is None:
            rng = SourceRange(Position(0, 0), Position(0, 1))
        missing = ", ".join(report.missing_endogenous[:5])
        n_missing = len(report.missing_endogenous)
        suffix = f" (and {n_missing - 5} more)" if n_missing > 5 else ""
        diagnostics.append(
            Diagnostic(
                range=rng,
                severity=Severity.WARNING,
                message=(
                    f"{n_missing} endogenous variable(s) missing from "
                    f"steady_state_model: {missing}{suffix}"
                ),
                code="W042",
            )
        )

    # Report equations with non-zero residuals (actual SS inconsistencies)
    for result in report.results:
        if result.is_satisfied:
            continue

        if result.is_local_var and result.residual is None:
            msg = f"Steady state check failed for {result.error_message}"
            diagnostics.append(
                Diagnostic(
                    range=result.equation.range,
                    severity=Severity.INFORMATION,
                    message=msg,
                    code="I041",
                )
            )
        elif result.residual is not None:
            # This is a real inconsistency — the equation doesn't hold
            eq_label = f" '{result.equation.name}'" if result.equation.name else ""
            msg = (
                f"Steady state inconsistency in equation{eq_label}: "
                f"{result.error_message}"
            )
            diagnostics.append(
                Diagnostic(
                    range=result.equation.range,
                    severity=Severity.WARNING,
                    message=msg,
                    code="W041",
                )
            )
        elif not result.missing_vars:
            # Eval error that isn't just a missing variable
            eq_label = f"'{result.equation.name}'" if result.equation.name else ""
            msg = f"Steady state check failed for equation {eq_label}: {result.error_message}".strip()
            diagnostics.append(
                Diagnostic(
                    range=result.equation.range,
                    severity=Severity.INFORMATION,
                    message=msg,
                    code="I041",
                )
            )

    # Add a summary diagnostic if there are real residual failures
    if report.n_failed > 0:
        rng = model.model_block_range
        if rng is None:
            rng = SourceRange(Position(0, 0), Position(0, 1))
        total = len([r for r in report.results if not r.is_local_var])
        # Subtract local-var rows from the numerator so the "k/n satisfied"
        # message is consistent: both sides count only real equations.
        # ``report.n_satisfied`` historically counted ``#``-prefixed local
        # variable definitions, which produced ``1/1 satisfied, 1 failed``
        # for a model with one local def and one failed equation.
        satisfied_real = len(
            [r for r in report.results if not r.is_local_var and r.is_satisfied]
        )
        diagnostics.append(
            Diagnostic(
                range=rng,
                severity=Severity.WARNING,
                message=(
                    f"Steady state check: {satisfied_real}/{total} equations "
                    f"satisfied, {report.n_failed} failed with non-zero residuals. "
                    f"Fix the steady state values before running Dynare."
                ),
                code="W040",
            )
        )

    return diagnostics


def _check_ss_block_coverage(model: ParsedModel) -> List[Diagnostic]:
    """Check that all endogenous variables have a steady state value assigned.

    Note: This check is now also done as part of _check_steady_state via the
    SteadyStateReport.missing_endogenous property, which provides a more
    consolidated message. This function remains for cases where the steady
    state validator returns None (no model block) but there IS a
    steady_state_model block.
    """
    # Skip if _check_steady_state will handle it
    if model.model_equations and model.steady_state_equations:
        return []

    diagnostics: List[Diagnostic] = []

    if not model.steady_state_equations:
        return diagnostics

    # Collect variables assigned in the steady_state_model block
    assigned: set = set()
    for eq in model.steady_state_equations:
        text = eq.text.strip()
        if "=" in text and not text.startswith("#"):
            name = text.split("=", 1)[0].strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                assigned.add(name)

    # Check that all endogenous variables are assigned
    for v in model.endogenous:
        if v.name not in assigned:
            rng = model.steady_state_block_range
            if rng is None:
                rng = v.range
            diagnostics.append(
                Diagnostic(
                    range=rng,
                    severity=Severity.WARNING,
                    message=(
                        f"Endogenous variable '{v.name}' has no assignment "
                        f"in the steady_state_model block."
                    ),
                    code="W042",
                )
            )

    return diagnostics


def _check_missing_steady_state(model: ParsedModel) -> List[Diagnostic]:
    """Suggest computing steady state when none is provided."""
    diagnostics: List[Diagnostic] = []

    if not model.static_model_equations():
        return diagnostics

    if _check_model_local_shadowing(model):
        return diagnostics

    # Only trigger if there's no SS info at all
    if model.steady_state_equations or model.initval_entries:
        return diagnostics

    # Check equation count matches (solver needs N equations = N unknowns)
    real_equations = model.static_model_equations()
    if len(real_equations) != len(model.endogenous):
        return diagnostics

    rng = model.model_block_range
    if rng is None:
        rng = SourceRange(Position(0, 0), Position(0, 1))

    diagnostics.append(
        Diagnostic(
            range=rng,
            severity=Severity.INFORMATION,
            message="No steady state values provided. "
            "Use the 'Compute Steady State' code action to solve automatically.",
            code="I050",
        )
    )

    return diagnostics


def _check_initval_references(model: ParsedModel) -> List[Diagnostic]:
    """Check that initval/endval entries reference declared variables."""
    diagnostics: List[Diagnostic] = []
    declared = model.all_declared_names()
    parameters = model.parameter_names()

    for block_name, entries in (
        ("initval", model.initval_entries),
        ("endval", model.endval_entries),
    ):
        for entry in entries:
            if entry.name in parameters:
                diagnostics.append(
                    Diagnostic(
                        range=entry.range,
                        severity=Severity.WARNING,
                        message=(
                            f"Parameter '{entry.name}' assigned in {block_name} "
                            "is ignored. Assign parameters before the model block "
                            "or inside steady_state_model instead."
                        ),
                        code="W053",
                    )
                )
                continue
            if entry.name not in declared:
                diagnostics.append(
                    Diagnostic(
                        range=entry.range,
                        severity=Severity.WARNING,
                        message=f"Variable '{entry.name}' in {block_name} is not declared.",
                        code="W050",
                    )
                )

    return diagnostics


_STEADY_STATE_CALL_OPENER = re.compile(r"\bsteady_state\s*\(", re.IGNORECASE)
_UNEVALUABLE_PARAM_RUN_COMMAND_RE = re.compile(
    r"(?<!\w)(steady|check|stoch_simul|simul|estimation|"
    r"perfect_foresight_setup|perfect_foresight_solver|"
    r"ramsey_policy|discretionary_policy|osr|calib_smoother|forecast)\b",
    re.IGNORECASE,
)


def _check_steady_state_operator_operands(
    model: ParsedModel,
) -> List[Diagnostic]:
    """Reject exogenous variables inside Dynare's steady_state() operator."""
    exogenous = model.exogenous_names()
    if not exogenous:
        return []
    diagnostics: List[Diagnostic] = []
    for equation in model.model_equations:
        text = equation.text
        i = 0
        while i < len(text):
            match = _STEADY_STATE_CALL_OPENER.search(text, i)
            if match is None:
                break
            depth = 1
            j = match.end()
            while j < len(text) and depth > 0:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                j += 1
            if depth != 0:
                break
            operand = text[match.end() : j - 1]
            bad_names = sorted(
                {
                    name
                    for name in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", operand)
                    if name in exogenous
                }
            )
            if bad_names:
                names = ", ".join(bad_names)
                diagnostics.append(
                    Diagnostic(
                        range=equation.range,
                        severity=Severity.ERROR,
                        message=(
                            "Invalid steady_state() operand: exogenous variable(s) "
                            f"{names} cannot be used inside steady_state()."
                        ),
                        code="E065",
                    )
                )
            i = j
    return diagnostics


def _run_command_positions(model: ParsedModel) -> List[Position]:
    """Return source positions for commands that execute calibrated values."""
    stripped = re.sub(
        r"'[^'\n]*'|\"[^\"\n]*\"",
        lambda match: " " * (match.end() - match.start()),
        _strip_non_macro_comments(model.text),
    )
    return [
        _offset_to_position_local(stripped, match.start(1))
        for match in _UNEVALUABLE_PARAM_RUN_COMMAND_RE.finditer(stripped)
    ]


def _position_sort_key(pos: Position) -> Tuple[int, int]:
    return (pos.line, pos.character)


def _add_unevaluable_param_diagnostic(
    diagnostics: List[Diagnostic],
    reported: set[Tuple[str, int, int]],
    assignment: ParamAssignment,
) -> None:
    if assignment.value is not None:
        return
    report_key = (
        assignment.name,
        assignment.range.start.line,
        assignment.range.start.character,
    )
    if report_key in reported:
        return
    reported.add(report_key)
    diagnostics.append(
        Diagnostic(
            range=assignment.range,
            severity=Severity.WARNING,
            message=(
                f"Parameter '{assignment.name}' assignment could not be evaluated: "
                f"{assignment.name} = {assignment.expression}. "
                f"Check that all referenced names are declared and assigned."
            ),
            code="W011",
        )
    )


def _check_unevaluable_params(model: ParsedModel) -> List[Diagnostic]:
    """Warn about parameter assignments that could not be evaluated."""
    diagnostics: List[Diagnostic] = []
    assignments: List[ParamAssignment] = list(model.param_assignments)
    parameter_names = model.parameter_names()
    if parameter_names:
        _values, ss_assignments = evaluate_steady_state_model_assignments(model)
        for ss_assignment in ss_assignments:
            name = ss_assignment.name
            if name not in parameter_names:
                continue
            expr = ss_assignment.expression.strip().rstrip(";").strip()
            if not expr:
                continue
            assignments.append(
                ParamAssignment(
                    name=name,
                    expression=expr,
                    value=ss_assignment.value,
                    range=ss_assignment.range,
                )
            )

    latest: Dict[str, ParamAssignment] = {}
    ordered_assignments = _source_ordered_assignments(assignments)
    reported: set[Tuple[str, int, int]] = set()

    for run_pos in (_position_sort_key(pos) for pos in _run_command_positions(model)):
        latest_before_run: Dict[str, ParamAssignment] = {}
        for assignment in ordered_assignments:
            if _position_sort_key(assignment.range.start) >= run_pos:
                continue
            latest_before_run[assignment.name] = assignment
        for assignment in latest_before_run.values():
            _add_unevaluable_param_diagnostic(diagnostics, reported, assignment)

    for assignment in ordered_assignments:
        latest[assignment.name] = assignment

    for assignment in latest.values():
        _add_unevaluable_param_diagnostic(diagnostics, reported, assignment)
    return diagnostics


def _check_helper_variables(
    model: ParsedModel,
    declared_names: Optional[set[str]] = None,
    range_override: Optional[SourceRange] = None,
) -> List[Diagnostic]:
    """Flag undeclared helper variables assigned in the parameter section.

    Only flags assignments that appear before the model block, since
    assignments after blocks are typically Dynare commands or MATLAB code.
    """
    diagnostics: List[Diagnostic] = []
    declared = declared_names or model.all_declared_names()

    # Only flag helpers before the first major block (model, initval, etc.)
    first_block_line = None
    for rng in [
        model.model_block_range,
        model.initval_block_range,
        model.endval_block_range,
        model.steady_state_block_range,
        model.shocks_block_range,
    ]:
        if rng is not None:
            bl = rng.start.line
            if first_block_line is None or bl < first_block_line:
                first_block_line = bl

    for a in model.helper_assignments:
        if a.name in declared:
            continue
        # Skip assignments after the first block
        if first_block_line is not None and a.range.start.line >= first_block_line:
            continue
        diagnostics.append(
            Diagnostic(
                range=range_override or a.range,
                severity=Severity.INFORMATION,
                message=(
                    f"'{a.name}' is assigned but not declared as a parameter. "
                    f"If it is used to compute other parameters, consider declaring "
                    f"it in the parameters block."
                ),
                code="W012",
            )
        )
    return diagnostics


def _collect_model_equation_references(
    model: ParsedModel,
    declared_names: Optional[set] = None,
) -> set:
    """Return the set of identifiers referenced in ``model.model_equations``.

    This is the hot inner loop shared by :func:`_check_unused_endogenous`,
    :func:`_check_unused_exogenous`, and :func:`_check_unused_parameters`.
    Computing it once and passing the result in avoids scanning the same
    equation list ~3 times per :func:`run_diagnostics` call.
    """
    if declared_names is None:
        declared_names = model.all_declared_names()
    refs: set = set()
    for eq in model.model_equations:
        refs.update(_extract_equation_references(eq, declared_names))
    return refs


def _check_unused_endogenous(
    model: ParsedModel,
    _model_eq_refs: Optional[set] = None,
) -> List[Diagnostic]:
    """Warn about endogenous variables not referenced in model equations."""
    diagnostics: List[Diagnostic] = []
    if not model.static_model_equations():
        return diagnostics

    # Collect all identifiers referenced in model equations (shared when caller
    # has already computed the set for the same model instance).
    if _model_eq_refs is None:
        referenced = _collect_model_equation_references(model)
    else:
        referenced = _model_eq_refs

    for v in model.endogenous:
        if v.name not in referenced:
            diagnostics.append(
                Diagnostic(
                    range=v.range,
                    severity=Severity.WARNING,
                    message=(
                        f"Endogenous variable '{v.name}' is declared but "
                        f"never referenced in the model block."
                    ),
                    code="W020",
                )
            )
    return diagnostics


def _check_unused_parameters(
    model: ParsedModel,
    _model_eq_refs: Optional[set] = None,
) -> List[Diagnostic]:
    """Warn about parameters declared but never referenced in model equations."""
    diagnostics: List[Diagnostic] = []
    if not model.model_equations:
        return diagnostics

    declared_names = model.all_declared_names()

    # Start from the pre-computed model-equation reference set when available;
    # otherwise compute it here.  Either way we extend it with the additional
    # sources (steady_state_model, shocks, assignments, initval/endval) that
    # only this check needs.
    if _model_eq_refs is None:
        referenced: set = _collect_model_equation_references(model, declared_names)
    else:
        referenced = set(_model_eq_refs)  # copy so we can extend in place

    # Also check steady_state_model equations
    for eq in model.steady_state_equations:
        referenced.update(_extract_equation_references(eq, declared_names))

    # Also check the shocks block: a parameter setting a shock variance or
    # stderr (e.g. "var e = sigma_e^2;") is genuinely referenced even though
    # it never appears in a model equation.
    for shocks_m in _find_all_blocks(_strip_comments(model.text), "shocks"):
        referenced.update(_extract_references(shocks_m.group(2), declared_names))

    # Parameters used in other parameter/helper assignments should not be flagged.
    for a in model.param_assignments + model.helper_assignments:
        for ref in _extract_references(a.expression, declared_names):
            referenced.add(ref)

    # Also check initval/endval entries: a parameter used to set an initial
    # guess (e.g. "initval; k = kss; end;") is genuinely referenced even though
    # it never appears in a model equation.
    for entry in model.initval_entries + model.endval_entries:
        referenced.update(_extract_references(entry.expression, declared_names))

    for p in model.parameters:
        if p.name not in referenced:
            diagnostics.append(
                Diagnostic(
                    range=p.range,
                    severity=Severity.INFORMATION,
                    message=(
                        f"Parameter '{p.name}' is declared but never referenced "
                        f"in model equations."
                    ),
                    code="W022",
                )
            )
    return diagnostics


def _check_unused_exogenous(
    model: ParsedModel,
    _model_eq_refs: Optional[set] = None,
) -> List[Diagnostic]:
    """Warn about exogenous variables not referenced in model equations."""
    diagnostics: List[Diagnostic] = []
    if not model.model_equations:
        return diagnostics

    # Collect all identifiers referenced in model equations (shared when caller
    # has already computed the set for the same model instance).
    if _model_eq_refs is None:
        referenced = _collect_model_equation_references(model)
    else:
        referenced = _model_eq_refs

    for v in model.exogenous:
        if v.name not in referenced:
            diagnostics.append(
                Diagnostic(
                    range=v.range,
                    severity=Severity.WARNING,
                    message=(
                        f"Exogenous variable '{v.name}' is declared but "
                        f"never referenced in the model block."
                    ),
                    code="W021",
                )
            )
    return diagnostics


def _check_exogenous_in_initval(model: ParsedModel) -> List[Diagnostic]:
    """Flag exogenous variables set in the initval block."""
    diagnostics: List[Diagnostic] = []
    exo_names = model.exogenous_names() - model.deterministic_exogenous_names()

    for entry in model.initval_entries:
        if entry.name in exo_names:
            diagnostics.append(
                Diagnostic(
                    range=entry.range,
                    severity=Severity.INFORMATION,
                    message=(
                        f"Exogenous variable '{entry.name}' is set in initval. "
                        f"This is unusual -- exogenous shocks are typically zero "
                        f"at steady state."
                    ),
                    code="W051",
                )
            )
    return diagnostics


def _check_missing_initval(model: ParsedModel) -> List[Diagnostic]:
    """Report endogenous variables missing from the initval block."""
    diagnostics: List[Diagnostic] = []

    if not model.initval_entries:
        return diagnostics

    initval_names = {e.name for e in model.initval_entries}
    endo_names = model.endogenous_names()
    missing = sorted(endo_names - initval_names)

    if missing:
        rng = model.initval_block_range
        if rng is None:
            rng = SourceRange(Position(0, 0), Position(0, 1))

        listed = ", ".join(missing[:10])
        suffix = f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""
        diagnostics.append(
            Diagnostic(
                range=rng,
                severity=Severity.INFORMATION,
                message=(
                    f"{len(missing)} endogenous variable(s) missing from initval "
                    f"(will default to 0): {listed}{suffix}"
                ),
                code="W052",
            )
        )
    return diagnostics


def _check_duplicate_equations(model: ParsedModel) -> List[Diagnostic]:
    """Detect duplicate equations in the model block."""
    diagnostics: List[Diagnostic] = []
    if not model.model_equations:
        return diagnostics

    # Normalize equations for comparison
    seen: dict = {}  # normalized_text -> list[(equation, macro branch signature)]
    branch_signatures = _macro_branch_signatures_by_line(model)
    for eq in model.model_equations:
        if eq.text.strip().startswith("#"):
            continue  # Skip model-local variable definitions
        # Normalize: ignore formatting-only operator spacing differences.
        # Dynare identifiers are case-sensitive (``lAGG`` and ``LAGG`` are
        # different variables), so normalization must NOT fold case.
        normalized = re.sub(r"\s*([=+\-*/^,()\[\]<>])\s*", r"\1", eq.text)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        tag_class = tuple(sorted(set(eq.tags) & {"static", "dynamic"}))
        duplicate_key = (normalized, tag_class)
        sig = branch_signatures.get(eq.range.start.line, tuple())
        duplicate_of = None
        for first_eq, first_sig in seen.get(duplicate_key, []):
            if not _macro_signatures_are_mutually_exclusive(first_sig, sig):
                duplicate_of = first_eq
                break
        if duplicate_of is not None:
            first_eq = duplicate_of
            first_line = first_eq.range.start.line + 1
            # Build autofix: remove the duplicate equation line(s) when
            # they hold ONLY this equation.  Compact one-liners like
            # ``var y; model; y = y; end;`` must not get a whole-line
            # delete (which would wipe declarations + block delimiters);
            # in that case, skip the fix entirely and let the user
            # resolve manually.
            fix = None
            if _whole_line_delete_safe(
                model.text,
                eq.range.start.line,
                eq.range.end.line,
            ):
                fix = TextEdit(
                    start_line=eq.range.start.line,
                    start_char=0,
                    end_line=eq.range.end.line + 1,
                    end_char=0,
                    new_text="",
                )
            diagnostics.append(
                Diagnostic(
                    range=eq.range,
                    severity=Severity.ERROR,
                    message=(
                        f"Duplicate equation (same as line {first_line}). "
                        f"Fix: remove this duplicate equation."
                    ),
                    code="E050",
                    fix=fix,
                )
            )
        else:
            seen.setdefault(duplicate_key, []).append((eq, sig))

    return diagnostics


MacroBranchSignature = Tuple[Tuple[int, int], ...]


def _macro_branch_signatures_by_line(
    model: ParsedModel,
) -> Dict[int, MacroBranchSignature]:
    """Map source lines to active macro control-flow alternatives.

    Two lines are mutually exclusive only when they share the same
    conditional frame id but are in different branches of that frame.
    Separate ``@#if`` blocks receive different frame ids and are not
    treated as mutually exclusive.
    """
    directives = sorted(
        model.macro_directives,
        key=lambda d: (d.range.start.line, d.range.start.character),
    )
    if not directives:
        return {}

    max_line = max(
        len(
            model.original_text.split("\n")
            if model.original_text
            else model.text.split("\n")
        )
        - 1,
        max(d.range.start.line for d in directives),
    )
    by_line: Dict[int, MacroBranchSignature] = {}
    stack: List[Tuple[int, int, str]] = []
    next_frame_id = 0
    idx = 0

    for line_no in range(max_line + 1):
        while idx < len(directives) and directives[idx].range.start.line == line_no:
            kind = directives[idx].kind
            if kind in {"if", "ifdef", "ifndef", "for"}:
                next_frame_id += 1
                stack.append((next_frame_id, 0, kind))
            elif kind in {"elseif", "else"}:
                for stack_idx in range(len(stack) - 1, -1, -1):
                    frame_id, branch, open_kind = stack[stack_idx]
                    if open_kind in {"if", "ifdef", "ifndef"}:
                        stack[stack_idx] = (frame_id, branch + 1, open_kind)
                        break
            elif kind in {"endif", "endfor"}:
                expected = {"endif": {"if", "ifdef", "ifndef"}, "endfor": {"for"}}[kind]
                for stack_idx in range(len(stack) - 1, -1, -1):
                    if stack[stack_idx][2] in expected:
                        del stack[stack_idx]
                        break
            idx += 1

        by_line[line_no] = tuple(
            (frame_id, branch) for frame_id, branch, _kind in stack
        )

    return by_line


def _macro_signatures_are_mutually_exclusive(
    left: MacroBranchSignature,
    right: MacroBranchSignature,
) -> bool:
    left_by_frame = dict(left)
    for frame_id, right_branch in right:
        left_branch = left_by_frame.get(frame_id)
        if left_branch is not None and left_branch != right_branch:
            return True
    return False


def _has_mutually_exclusive_macro_declarations(model: ParsedModel) -> bool:
    """True when declarations occur in mutually-exclusive macro branches."""
    branch_signatures = _macro_branch_signatures_by_line(model)
    signatures = [
        branch_signatures.get(decl.range.start.line, tuple())
        for decl in (
            list(model.endogenous) + list(model.exogenous) + list(model.parameters)
        )
    ]
    signatures = [sig for sig in signatures if sig]
    for i, left in enumerate(signatures):
        for right in signatures[i + 1 :]:
            if _macro_signatures_are_mutually_exclusive(left, right):
                return True
    return False


def _has_mutually_exclusive_macro_equations(model: ParsedModel) -> bool:
    """True when model equations occur in mutually-exclusive macro branches."""
    branch_signatures = _macro_branch_signatures_by_line(model)
    signatures = [
        branch_signatures.get(eq.range.start.line, tuple())
        for eq in model.model_equations
        if not eq.text.strip().startswith("#")
    ]
    signatures = [sig for sig in signatures if sig]
    for i, left in enumerate(signatures):
        for right in signatures[i + 1 :]:
            if _macro_signatures_are_mutually_exclusive(left, right):
                return True
    return False


def _has_macro_branch_equations(model: ParsedModel) -> bool:
    branch_signatures = _macro_branch_signatures_by_line(model)
    return any(
        bool(branch_signatures.get(eq.range.start.line, tuple()))
        for eq in model.model_equations
        if not eq.text.strip().startswith("#")
    )


def _macro_branch_equation_counts(model: ParsedModel) -> Optional[set[int]]:
    """Return possible real-equation counts for simple independent macro branches."""
    branch_signatures = _macro_branch_signatures_by_line(model)
    branch_alternatives = _macro_branch_alternatives(model)
    original_lines = (getattr(model, "original_text", "") or model.text).splitlines()
    seen_equation_texts: Dict[str, int] = {}

    def _signature_line(eq_text: str, pos: Position) -> int:
        normalized = eq_text.strip().rstrip(";").strip()
        if normalized and original_lines:
            seen_idx = seen_equation_texts.get(normalized, 0)
            seen_equation_texts[normalized] = seen_idx + 1
            candidates = [
                line_no
                for line_no, line in enumerate(original_lines)
                if line.strip().rstrip(";").strip() == normalized
                and not line.lstrip().startswith("@#")
            ]
            if seen_idx < len(candidates):
                return candidates[seen_idx]
        source_map = getattr(model, "source_map", None)
        if not source_map:
            return pos.line
        original_text = getattr(model, "original_text", "") or model.text
        offset = _position_to_offset(model.text, pos)
        mapped = source_map[-1] if offset >= len(source_map) else source_map[offset]
        return _offset_to_position_local(original_text, mapped).line

    base_count = 0
    branch_counts: Dict[int, Dict[int, int]] = {}
    for eq in model.model_equations:
        if eq.text.strip().startswith("#"):
            continue
        sig = branch_signatures.get(_signature_line(eq.text, eq.range.start), tuple())
        if not sig:
            base_count += 1
            continue
        if len(sig) != 1:
            return None
        frame_id, branch = sig[0]
        branch_counts.setdefault(frame_id, {})
        branch_counts[frame_id][branch] = branch_counts[frame_id].get(branch, 0) + 1

    counts = {base_count}
    frame_ids = set(branch_counts) | set(branch_alternatives)
    for frame_id in frame_ids:
        per_branch = branch_counts.get(frame_id, {})
        alternatives = branch_alternatives.get(frame_id, set(per_branch))
        counts = {
            count + per_branch.get(branch, 0)
            for count in counts
            for branch in alternatives
        }
    return counts


def _macro_branch_alternatives(model: ParsedModel) -> Dict[int, set[int]]:
    """Return branch ids that may be selected for each simple conditional frame."""
    directives = sorted(
        model.macro_directives,
        key=lambda d: (d.range.start.line, d.range.start.character),
    )
    source_text = model.original_text or model.text
    _defines, active_lines, line_defines = _macro_branch_state(source_text)

    def _truth_for(directive: MacroDirective) -> Optional[bool]:
        defines = line_defines.get(directive.range.start.line, {})
        if directive.kind == "if":
            return _macro_truth_value(directive.argument, defines)
        name = (directive.argument or "").strip()
        truth = name in defines
        return not truth if directive.kind == "ifndef" else truth

    alternatives: Dict[int, set[int]] = {}
    stack: List[dict] = []
    next_frame_id = 0
    for directive in directives:
        kind = directive.kind
        if kind in {"if", "ifdef", "ifndef"}:
            next_frame_id += 1
            truth = _truth_for(directive)
            is_active = (
                directive.range.start.line < len(active_lines)
                and active_lines[directive.range.start.line]
            )
            frame = {
                "id": next_frame_id,
                "branch": 0,
                "kind": kind,
                "has_else": False,
                "known_taken": bool(truth)
                if truth is not None and is_active
                else False,
                "unknown_possible": truth is None and is_active,
            }
            stack.append(frame)
            alternatives[next_frame_id] = (
                {0} if truth is not False and is_active else set()
            )
        elif kind == "for":
            next_frame_id += 1
            stack.append(
                {
                    "id": next_frame_id,
                    "branch": 0,
                    "kind": kind,
                    "has_else": True,
                }
            )
        elif kind in {"elseif", "else"}:
            for frame in reversed(stack):
                if frame["kind"] in {"if", "ifdef", "ifndef"}:
                    frame["branch"] += 1
                    if kind == "else":
                        frame["has_else"] = True
                    possible = frame["unknown_possible"] or not frame["known_taken"]
                    if kind == "elseif" and not frame["unknown_possible"]:
                        truth = _macro_truth_value(
                            directive.argument,
                            line_defines.get(directive.range.start.line, {}),
                        )
                        possible = truth is None or bool(truth)
                        if truth is None:
                            frame["unknown_possible"] = True
                        elif truth:
                            frame["known_taken"] = True
                    elif kind == "else" and possible:
                        frame["known_taken"] = True
                    if possible:
                        alternatives.setdefault(frame["id"], set()).add(frame["branch"])
                    break
        elif kind in {"endif", "endfor"}:
            expected = {"endif": {"if", "ifdef", "ifndef"}, "endfor": {"for"}}[kind]
            for idx in range(len(stack) - 1, -1, -1):
                frame = stack[idx]
                if frame["kind"] not in expected:
                    continue
                if (
                    kind == "endif"
                    and not frame["has_else"]
                    and (frame["unknown_possible"] or not frame["known_taken"])
                ):
                    alternatives.setdefault(frame["id"], set()).add(
                        frame["branch"] + 1,
                    )
                del stack[idx]
                break
    return alternatives


def _check_macro_branch_equation_count(model: ParsedModel) -> List[Diagnostic]:
    """Check equation count when simple macro alternatives are present."""
    if not model.endogenous:
        return []
    counts = _macro_branch_equation_counts(model)
    if not counts:
        return []
    n_endo = len(model.endogenous)
    # When the resolved (active) configuration balances and the branch
    # arithmetic yields a single count, trust the resolved model: a lone count
    # that disagrees with a balanced model is a counting artifact of
    # fully-defined ``@#define`` branches (Dynare itself accepts the model).
    # Genuine mismatches (resolved count != n_endo) and true branch-dependent
    # ambiguity (multiple possible counts) still flag below.
    if len(model.dynamic_model_equations()) == n_endo and len(counts) <= 1:
        return []
    if all(count == n_endo for count in counts):
        return []

    if len(counts) == 1:
        count_text = f"{next(iter(counts))} equation(s)"
    else:
        count_text = (
            f"{min(counts)}-{max(counts)} equation(s) depending on macro branch"
        )
    rng = model.model_block_range or SourceRange(Position(0, 0), Position(0, 1))
    return [
        Diagnostic(
            range=rng,
            severity=Severity.ERROR,
            message=(
                f"Equation count mismatch: {count_text} but {n_endo} endogenous "
                "variable(s). Fix: adjust equations or declarations so every "
                "macro branch has one equation per endogenous variable."
            ),
            code="E010",
        )
    ]


def _check_duplicate_param_assignments(model: ParsedModel) -> List[Diagnostic]:
    """Detect duplicate parameter assignments."""
    diagnostics: List[Diagnostic] = []
    seen: dict = {}  # name -> list[(assignment, macro branch signature)]
    branch_signatures = _macro_branch_signatures_by_line(model)

    for a in model.param_assignments:
        sig = branch_signatures.get(a.range.start.line, tuple())
        duplicate_of = None
        for first, first_sig in seen.get(a.name, []):
            if (
                first.value is not None
                and a.value is not None
                and first.value == a.value
                and not _macro_signatures_are_mutually_exclusive(first_sig, sig)
            ):
                duplicate_of = first
                break
        if duplicate_of is not None:
            first = duplicate_of
            fix = None
            if _whole_line_delete_safe(
                model.text,
                a.range.start.line,
                a.range.start.line,
            ):
                fix = TextEdit(
                    start_line=a.range.start.line,
                    start_char=0,
                    end_line=a.range.start.line + 1,
                    end_char=0,
                    new_text="",
                )
            diagnostics.append(
                Diagnostic(
                    range=a.range,
                    # Dynare permits re-assigning a parameter (the later value
                    # wins), so this is a redundancy warning, not a hard error.
                    severity=Severity.WARNING,
                    message=(
                        f"Duplicate parameter assignment '{a.name} = {a.expression}' "
                        f"(same value as line {first.range.start.line + 1}); the "
                        f"later assignment is redundant. Remove it to avoid confusion."
                    ),
                    code="E052",
                    fix=fix,
                )
            )
        seen.setdefault(a.name, []).append((a, sig))

    return diagnostics


def _whole_line_delete_safe(
    model_text: str,
    start_line: int,
    end_line: int,
) -> bool:
    """Return True if removing lines start_line..end_line is safe.

    A whole-line delete is safe only when those lines don't share their
    content with surrounding code on the same line.  For compact
    Dynare like ``var y; model; y = y; end;`` on one line, the safe
    answer is False — deleting the whole line wipes the surrounding
    declaration / block delimiters too.
    """
    lines = model_text.split("\n")
    if start_line < 0 or end_line >= len(lines):
        return False
    # If the start line has any content BEFORE the offset where the
    # equation begins, we can't safely whole-line-delete.  Likewise if
    # the end line has content AFTER.  This conservative check rejects
    # compact one-liners while still permitting "equation on its own
    # line" — the common case.  We approximate by checking that all
    # lines in the range contain only the equation text + whitespace +
    # an optional trailing ``;``.
    for i in range(start_line, end_line + 1):
        stripped = lines[i].strip()
        # Allow lines that are entirely the equation content (including
        # block-keyword absence): if the line is just a single
        # ``thing = thing;`` style fragment with no other statements,
        # accept.  Reject if multiple ``;`` separators are present —
        # that signals same-line compact code.
        if stripped.count(";") > 1:
            return False
        # Reject if a Dynare block-opener or closer appears on the
        # same line.
        block_kws = re.compile(
            r"\b(?:model|end|initval|endval|shocks|steady_state_model|var|varexo|parameters)\b",
            re.IGNORECASE,
        )
        # Mask the equation's own occurrence so we don't false-positive
        # on its identifier.
        if block_kws.search(stripped):
            # A var/end/etc keyword on the same line means whole-line
            # delete would corrupt structure.
            return False
    return True


def _check_trivial_equations(model: ParsedModel) -> List[Diagnostic]:
    """Detect trivially true or contradictory equations in the model block."""
    diagnostics: List[Diagnostic] = []
    if not model.model_equations:
        return diagnostics

    for eq in model.model_equations:
        text = eq.text.strip()
        if text.startswith("#"):
            continue  # Skip model-local variable definitions

        # Check for "0 = 0", "0 = 1", etc. (constant = constant contradictions)
        m = re.match(
            r"^\s*([+-]?\s*\d+(?:\.\d+)?)\s*=\s*([+-]?\s*\d+(?:\.\d+)?)\s*$", text
        )
        if m:
            lhs_val = float(m.group(1).replace(" ", ""))
            rhs_val = float(m.group(2).replace(" ", ""))
            if lhs_val != rhs_val:
                fix = None
                if _whole_line_delete_safe(
                    model.text,
                    eq.range.start.line,
                    eq.range.end.line,
                ):
                    fix = TextEdit(
                        start_line=eq.range.start.line,
                        start_char=0,
                        end_line=eq.range.end.line + 1,
                        end_char=0,
                        new_text="",
                    )
                diagnostics.append(
                    Diagnostic(
                        range=eq.range,
                        severity=Severity.ERROR,
                        message=(
                            f"Contradictory equation '{text}' (always false). "
                            f"Fix: remove this equation."
                        ),
                        code="E051",
                        fix=fix,
                    )
                )
                continue

        # Check for "X = X" (trivially true)
        if "=" in text:
            parts = text.split("=", 1)
            lhs_norm = re.sub(r"\s+", "", parts[0]).strip()
            rhs_norm = re.sub(r"\s+", "", parts[1]).strip()
            if lhs_norm and lhs_norm == rhs_norm:
                fix = None
                if _whole_line_delete_safe(
                    model.text,
                    eq.range.start.line,
                    eq.range.end.line,
                ):
                    fix = TextEdit(
                        start_line=eq.range.start.line,
                        start_char=0,
                        end_line=eq.range.end.line + 1,
                        end_char=0,
                        new_text="",
                    )
                diagnostics.append(
                    Diagnostic(
                        range=eq.range,
                        severity=Severity.ERROR,
                        message=(
                            f"Trivially true equation '{text}' (LHS = RHS). "
                            f"Fix: remove this equation."
                        ),
                        code="E051",
                        fix=fix,
                    )
                )

    return diagnostics


_ASSIGNMENT_LHS_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)")


def _blank_equation_tags(text: str, masked: str) -> str:
    chars = list(masked)
    for start, end in _iter_equation_tag_spans(text):
        for idx in range(start, min(end, len(chars))):
            if chars[idx] != "\n":
                chars[idx] = " "
    return "".join(chars)


def _merged_assignment_diagnostic(
    model: ParsedModel,
    rng: SourceRange,
    lhs_name: str,
    context: str,
    _line_starts: Optional[List[int]] = None,
) -> Optional[Diagnostic]:
    """Check a single assignment range for a missing-semicolon merge.

    *_line_starts* is an optional precomputed line-starts index from
    :func:`_build_line_starts`.  When provided (as :func:`_check_merged_assignments`
    always does), the two :func:`_position_to_offset` calls become O(1) instead
    of O(n) in the length of ``model.text``, which matters on 300+ parameter models.
    """
    text_len = len(model.text)
    if _line_starts is not None:
        start = _position_to_offset_with_index(_line_starts, text_len, rng.start)
        end = _position_to_offset_with_index(_line_starts, text_len, rng.end)
    else:
        start = _position_to_offset(model.text, rng.start)
        end = _position_to_offset(model.text, rng.end)
    if start >= end:
        return None

    source_segment = model.text[start:end]
    if context == "top-level":
        # Trailing MATLAB statements in a .mod file (e.g.
        # ``x = csolve('f', 0, [], M_, oo_, options_)``) are captured as
        # helper assignments but are not merged Dynare assignments: omitting
        # the semicolon is legal MATLAB.  A genuine top-level Dynare
        # assignment is scalar math, so string literals, matrix/cell
        # brackets, or the runtime structs M_/oo_/options_ mark MATLAB code.
        if (
            "'" in source_segment
            or '"' in source_segment
            or "[" in source_segment
            or "]" in source_segment
            or "{" in source_segment
            or "}" in source_segment
            or re.search(r"\b(?:M_|oo_|options_)\b", source_segment)
        ):
            return None
        line_start = model.text.rfind("\n", 0, start) + 1
        prefix = model.text[line_start:start]
        prefix_code = _mask_non_code_for_reference_search(prefix).strip()
        if prefix_code and not prefix_code.endswith(";"):
            return None

    masked_segment = _blank_equation_tags(
        source_segment,
        _mask_non_code_for_reference_search(source_segment),
    )
    matches = list(_ASSIGNMENT_LHS_RE.finditer(masked_segment))
    if len(matches) < 2:
        return None

    split_match = matches[1]
    split_rel = split_match.start(1)
    fix_start_rel = split_rel
    while fix_start_rel > 0 and source_segment[fix_start_rel - 1] in " \t":
        fix_start_rel -= 1

    split_abs = start + split_rel
    fix_start_abs = start + fix_start_rel
    split_pos = _offset_to_position_local(model.text, split_abs)
    fix_start_pos = _offset_to_position_local(model.text, fix_start_abs)
    second_name = split_match.group(1)
    message = (
        f"Statement in '{context}' appears to contain multiple assignments "
        "merged due to a missing semicolon. "
        f"Fix: add ';' before '{second_name} = ...'."
    )
    if context == "top-level":
        message = (
            f"Parameter/helper assignment '{lhs_name}' appears to contain "
            "multiple assignments merged due to a missing semicolon. "
            f"Fix: add ';' before '{second_name} = ...'."
        )

    return Diagnostic(
        range=rng,
        severity=Severity.ERROR,
        message=message,
        code="E001",
        fix=TextEdit(
            start_line=fix_start_pos.line,
            start_char=fix_start_pos.character,
            end_line=split_pos.line,
            end_char=split_pos.character,
            new_text="; ",
        ),
    )


def _check_merged_assignments(model: ParsedModel) -> List[Diagnostic]:
    """Detect assignment statements merged by a missing semicolon."""
    from .parser import _strip_comments, _trailing_code_line

    diagnostics: List[Diagnostic] = []

    # Precompute line starts once so that _merged_assignment_diagnostic's two
    # _position_to_offset calls are O(1) rather than O(len(text)) each.
    # On a 300+ parameter model this reduces ~680 O(n) splits to a single scan.
    line_starts = _build_line_starts(model.text)

    # Assignments after the first computation command are trailing MATLAB, not
    # Dynare parameter assignments, so they must not be flagged as merged.
    trailing_line = _trailing_code_line(_strip_comments(model.text))

    for assignment in list(model.param_assignments) + list(model.helper_assignments):
        if trailing_line is not None and assignment.range.start.line > trailing_line:
            continue
        diagnostic = _merged_assignment_diagnostic(
            model,
            assignment.range,
            assignment.name,
            "top-level",
            _line_starts=line_starts,
        )
        if diagnostic is not None:
            diagnostics.append(diagnostic)

    for block_name, entries in (
        ("initval", model.initval_entries),
        ("endval", model.endval_entries),
    ):
        for entry in entries:
            diagnostic = _merged_assignment_diagnostic(
                model,
                entry.range,
                entry.name,
                block_name,
                _line_starts=line_starts,
            )
            if diagnostic is not None:
                diagnostics.append(diagnostic)

    for eq in model.steady_state_equations:
        if eq.text.strip().startswith("#"):
            continue
        diagnostic = _merged_assignment_diagnostic(
            model,
            eq.range,
            eq.lhs or eq.text.split("=", 1)[0].strip(),
            "steady_state_model",
            _line_starts=line_starts,
        )
        if diagnostic is not None:
            diagnostics.append(diagnostic)

    return diagnostics


def _check_unbalanced_parens(model: ParsedModel) -> List[Diagnostic]:
    """Detect model equations whose parentheses cannot balance (e.g. a
    mangled timed reference leaving an orphan ``)``).

    The preprocessor rejects these as syntax errors; this native check keeps
    that sensitivity when the bundled preprocessor is unavailable.  On a
    preprocessor SUCCESS the reconciliation drops this E001 like any other
    native parse error, so it cannot create user-visible false positives.
    """
    diagnostics: List[Diagnostic] = []
    for eq in model.model_equations:
        masked = _mask_macro_directive_lines(
            _mask_non_code_for_reference_search(eq.text),
        )
        # Equation tags ([name='...']) may legally contain parens.
        masked = re.sub(r"\[[^\]]*\]", " ", masked)
        depth = 0
        orphan_close = False
        for ch in masked:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    orphan_close = True
                    break
        if orphan_close or depth != 0:
            unmatched = "')'" if orphan_close else "'('"
            diagnostics.append(
                Diagnostic(
                    range=eq.range,
                    severity=Severity.ERROR,
                    message=f"Unbalanced parentheses in equation: unmatched {unmatched}.",
                    code="E001",
                )
            )
    return diagnostics


def _check_merged_equations(model: ParsedModel) -> List[Diagnostic]:
    """Detect model equations that appear to be two equations merged due to
    a missing semicolon (contain multiple '=' signs).

    For example: ``y = alpha * y(-1) + e c = beta * y`` should be two equations
    separated by a semicolon.
    """
    diagnostics: List[Diagnostic] = []
    if not model.model_equations:
        return diagnostics

    for eq in model.model_equations:
        text = eq.text.strip()
        if text.startswith("#"):
            continue  # Skip model-local variable definitions

        # Remove equation tags [name='...']
        cleaned = re.sub(r"\[[^\]]*\]", "", text)
        # Count equals signs
        eq_count = cleaned.count("=")
        if eq_count >= 2:
            # This is likely two equations merged due to a missing semicolon
            # Find the position of the second '=' to suggest where the split should be
            parts = cleaned.split("=")
            # The split point is where a new identifier starts on the RHS
            # Pattern: "... expr IDENTIFIER = ..."
            # We want to insert ";" before the identifier that precedes the second "="
            rhs = parts[1]  # everything after first =
            # Find the last identifier before the second = in the rhs
            m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*$", rhs.rstrip())
            split_var = m.group(1) if m else None

            # Build auto-fix: insert semicolon before the second equation
            fix = None
            source_start = _position_to_offset(model.text, eq.range.start)
            source_end = _position_to_offset(model.text, eq.range.end)
            source_segment = model.text[source_start:source_end]
            masked_segment = _mask_non_code_for_reference_search(source_segment)

            def _blank_preserve_newlines(m: re.Match) -> str:
                return re.sub(r"[^\n]", " ", m.group(0))

            # Equation tags can contain ``name=`` / ``mcp=`` attributes.
            # Blank them before locating the second equation separator.
            masked_segment = re.sub(
                r"\[[^\]]*\]", _blank_preserve_newlines, masked_segment
            )

            eq_positions: List[int] = []
            i = 0
            while i < len(masked_segment):
                ch = masked_segment[i]
                if (
                    ch == "="
                    and i + 1 < len(masked_segment)
                    and masked_segment[i + 1] == "="
                ):
                    i += 2
                elif (
                    ch in ("<", ">", "~", "!")
                    and i + 1 < len(masked_segment)
                    and masked_segment[i + 1] == "="
                ):
                    i += 2
                elif ch == "=":
                    eq_positions.append(i)
                    i += 1
                else:
                    i += 1

            if len(eq_positions) >= 2:
                # Walk left from the second standalone "=" to find the
                # identifier starting the second merged equation. This
                # avoids matching the first equation's LHS when both
                # equations use the same variable name.
                k = eq_positions[1]
                while k > 0 and masked_segment[k - 1].isspace():
                    k -= 1
                ident_end = k
                while k > 0 and re.match(r"[A-Za-z0-9_]", masked_segment[k - 1]):
                    k -= 1
                candidate = masked_segment[k:ident_end]
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate):
                    if split_var is None:
                        split_var = candidate
                    split_offset = source_start + k
                    split_pos = _offset_to_position_local(model.text, split_offset)
                    lines = model.text.split("\n")
                    line_prefix = lines[split_pos.line][: split_pos.character]
                    if split_pos.line > eq.range.start.line and not line_prefix.strip():
                        # Second equation starts on its own line; insert
                        # the missing semicolon at the end of the previous
                        # CODE line — skipping blank/comment-only separator
                        # lines (a ';' inside a comment is inert) and
                        # stopping before any trailing inline comment.
                        stripped_src_lines = _strip_non_macro_comments(
                            model.text
                        ).split("\n")
                        prev_line_idx = split_pos.line - 1
                        while (
                            prev_line_idx > eq.range.start.line
                            and not stripped_src_lines[prev_line_idx].strip()
                        ):
                            prev_line_idx -= 1
                        prev_char = len(stripped_src_lines[prev_line_idx].rstrip())
                        fix = TextEdit(
                            start_line=prev_line_idx,
                            start_char=prev_char,
                            end_line=prev_line_idx,
                            end_char=prev_char,
                            new_text=";",
                        )
                    else:
                        fix = TextEdit(
                            start_line=split_pos.line,
                            start_char=split_pos.character,
                            end_line=split_pos.line,
                            end_char=split_pos.character,
                            new_text=";\n",
                        )

            msg = (
                "Equation appears to contain multiple equations merged "
                "due to a missing semicolon."
            )
            if split_var:
                msg += (
                    f" It looks like '{split_var} = ...' should be a "
                    f"separate equation. Fix: add ';' before '{split_var}'."
                )
            else:
                msg += " Fix: add ';' between the two equations."

            diagnostics.append(
                Diagnostic(
                    range=eq.range,
                    severity=Severity.ERROR,
                    message=msg,
                    code="E001",
                    fix=fix,
                )
            )

    return diagnostics


def _check_stray_equations(model: ParsedModel) -> List[Diagnostic]:
    """Detect equation-like statements outside model/steady_state_model blocks.

    Catches things like ``0 = 1;`` or ``x = y + z;`` placed between declarations,
    which cause Dynare syntax errors at runtime.
    """
    diagnostics: List[Diagnostic] = []
    text = model.text

    # Build line ranges for all known blocks
    block_lines: set = set()
    for rng in [
        model.model_block_range,
        model.initval_block_range,
        model.endval_block_range,
        model.steady_state_block_range,
        model.shocks_block_range,
    ]:
        if rng is not None:
            for ln in range(rng.start.line, rng.end.line + 1):
                block_lines.add(ln)

    # Also exclude parameter assignment lines and declaration lines
    param_lines = {a.range.start.line for a in model.param_assignments}
    param_lines.update(a.range.start.line for a in model.helper_assignments)
    decl_lines: set = set()
    for v in model.endogenous + model.exogenous + model.parameters:
        decl_lines.add(v.range.start.line)

    from .parser import _strip_comments

    stripped = _strip_comments(text)
    stripped_lines = stripped.split("\n")

    for i, sline in enumerate(stripped_lines):
        if i in block_lines or i in param_lines or i in decl_lines:
            continue
        sline_stripped = sline.strip()
        if not sline_stripped:
            continue
        # Look for "number = number" patterns (contradictory equations)
        m = re.match(
            r"([+-]?\s*\d+(?:\.\d+)?)\s*=\s*([+-]?\s*\d+(?:\.\d+)?)\s*;?\s*$",
            sline_stripped,
        )
        if m:
            try:
                float(m.group(1).replace(" ", ""))
                float(m.group(2).replace(" ", ""))
            except ValueError:
                continue
            fix = TextEdit(
                start_line=i,
                start_char=0,
                end_line=i + 1,
                end_char=0,
                new_text="",
            )
            start_col = len(sline) - len(sline.lstrip())
            end_col = start_col + len(sline_stripped)
            diagnostics.append(
                Diagnostic(
                    range=SourceRange(Position(i, start_col), Position(i, end_col)),
                    severity=Severity.ERROR,
                    message=(
                        f"Stray equation '{sline_stripped.rstrip(';').strip()}' outside model block "
                        f"(line {i + 1}). This will cause a Dynare syntax error. "
                        f"Fix: remove this line."
                    ),
                    code="E053",
                    fix=fix,
                )
            )

    return diagnostics


def _check_included_stray_equations(
    include_models: Optional[List[ParsedModel]],
) -> List[Diagnostic]:
    diagnostics: List[Diagnostic] = []
    for include_model in include_models or []:
        if getattr(include_model, "include_context", None) in {
            "model",
            "steady_state_model",
            "initval",
            "endval",
            "shocks",
        }:
            continue
        anchor = getattr(include_model, "include_anchor_range", None)
        for diagnostic in _check_stray_equations(include_model):
            if anchor is not None:
                diagnostic = _anchor_to_include(diagnostic, anchor)
            else:
                diagnostic = replace(diagnostic, fix=None)
            diagnostics.append(diagnostic)
    return diagnostics


def _check_circular_includes(
    model: ParsedModel,
    cycles: List[List[str]],
) -> List[Diagnostic]:
    """Emit ``E060`` for each cycle in the include graph touching this file.

    *cycles* is the output of :meth:`WorkspaceIndex.find_circular_includes`
    for the active document.  Each cycle is anchored at the first
    ``@#include`` directive in the source so the editor can place the
    squiggle on the offending line.  When no directives are present (e.g.
    the cycle is discovered via files we transitively read from disk) the
    cycle is reported at the top of the file.
    """
    diagnostics: List[Diagnostic] = []
    if not cycles:
        return diagnostics

    for cycle in cycles:
        # Render the cycle in a compact, human-readable form.
        from pathlib import Path as _P

        names = [_P(p).name for p in cycle]
        chain = " -> ".join(names)
        directive_range = SourceRange(Position(0, 0), Position(0, 1))
        source_text = getattr(model, "original_text", "") or model.text
        _defines, active_lines, _line_defines = _macro_branch_state(source_text)
        for directive in model.includes:
            if (
                directive.range.start.line < len(active_lines)
                and not active_lines[directive.range.start.line]
            ):
                continue
            target_parts = _P(directive.filename.replace("\\", "/")).parts
            if any(
                len(target_parts) <= len(_P(path).parts)
                and _P(path).parts[-len(target_parts) :] == target_parts
                for path in cycle
            ):
                directive_range = directive.range
                break
        else:
            if model.includes:
                directive_range = model.includes[0].range
        diagnostics.append(
            Diagnostic(
                range=directive_range,
                severity=Severity.ERROR,
                message=(
                    f"Circular @#include detected: {chain}. "
                    f"Fix: break the cycle by removing one of the @#include "
                    f"directives along this chain."
                ),
                code="E060",
            )
        )

    return diagnostics


def _check_unresolved_includes(
    unresolved: List[IncludeDirective],
) -> List[Diagnostic]:
    """Emit ``E061`` for each ``@#include`` whose target file isn't on disk.

    *unresolved* is the output of
    :meth:`WorkspaceIndex.find_unresolved_includes` for the active
    document.  Each diagnostic is anchored at the directive's own source
    range so the editor can place the squiggle on the bad path.
    """
    diagnostics: List[Diagnostic] = []
    for directive in unresolved:
        diagnostics.append(
            Diagnostic(
                range=directive.range,
                severity=Severity.ERROR,
                message=(
                    f"Cannot resolve @#include target '{directive.filename}'. "
                    f"Searched the directory of the including file and the "
                    f"workspace search paths. Fix: correct the path, add the "
                    f"missing file, or add its containing directory to the "
                    f"language server's search paths."
                ),
                code="E061",
            )
        )
    return diagnostics


def _anchor_synthetic_include_diagnostics(
    model: ParsedModel,
    diagnostics: List[Diagnostic],
) -> List[Diagnostic]:
    """Move diagnostics from synthetic include expansion ranges to include lines."""
    if not model.includes:
        return diagnostics
    source_text = getattr(model, "original_text", "") or model.text
    lines = source_text.split("\n")
    include_by_line = {
        directive.range.start.line: directive for directive in model.includes
    }
    anchored: List[Diagnostic] = []
    for diagnostic in diagnostics:
        line_no = diagnostic.range.start.line
        line_too_short = (
            line_no < 0
            or line_no >= len(lines)
            or diagnostic.range.start.character > len(lines[line_no])
        )
        directive = include_by_line.get(line_no)
        if directive is not None and (
            line_too_short or diagnostic.code not in {"E060", "E061"}
        ):
            diagnostic = replace(diagnostic, range=directive.range, fix=None)
        anchored.append(diagnostic)
    return anchored


def _check_unmatched_macro_blocks(model: ParsedModel) -> List[Diagnostic]:
    """Emit ``E062`` for unmatched or crossed Dynare macro blocks.

    Conditional openers (``@#if``, ``@#ifdef``, ``@#ifndef``) all close
    with ``@#endif``.  ``@#for`` closes with ``@#endfor``.  Use one stack
    so crossed closers are reported instead of hidden by per-kind stacks.
    """
    diagnostics: List[Diagnostic] = []
    stack: List[MacroDirective] = []
    seen_else_stack: List[bool] = []
    conditional_openers = {"if", "ifdef", "ifndef"}
    had_mismatch = False

    def _label(kind: str) -> str:
        return f"@#{kind}"

    def _closer_for(kind: str) -> str:
        return "@#endif" if kind in conditional_openers else "@#endfor"

    def _emit_stray(directive, opener_label: str) -> None:
        diagnostics.append(
            Diagnostic(
                range=directive.range,
                severity=Severity.ERROR,
                message=(
                    f"Stray {_label(directive.kind)} with no matching {opener_label}. "
                    f"Fix: remove this directive or add a matching {opener_label} above."
                ),
                code="E062",
            )
        )

    def _emit_mismatch(directive, open_kind: str) -> None:
        nonlocal had_mismatch
        had_mismatch = True
        expected = _closer_for(open_kind)
        diagnostics.append(
            Diagnostic(
                range=directive.range,
                severity=Severity.ERROR,
                message=(
                    f"Mismatched {_label(directive.kind)} while {_label(open_kind)} "
                    f"block is still open. Fix: close the {_label(open_kind)} block "
                    f"with {expected} before {_label(directive.kind)}."
                ),
                code="E062",
            )
        )

    def _emit_invalid_branch(directive, message: str) -> None:
        diagnostics.append(
            Diagnostic(
                range=directive.range,
                severity=Severity.ERROR,
                message=f"{message}. Fix: remove or reorder this macro branch.",
                code="E062",
            )
        )

    for directive in model.macro_directives:
        if directive.kind in conditional_openers or directive.kind == "for":
            stack.append(directive)
            seen_else_stack.append(False)
        elif directive.kind in ("else", "elseif"):
            if not stack:
                _emit_stray(directive, "@#if")
            elif stack[-1].kind not in conditional_openers:
                _emit_mismatch(directive, stack[-1].kind)
            elif directive.kind == "else":
                if seen_else_stack[-1]:
                    _emit_invalid_branch(directive, "Duplicate @#else in @#if block")
                else:
                    seen_else_stack[-1] = True
            elif seen_else_stack[-1]:
                _emit_invalid_branch(directive, "@#elseif after @#else in @#if block")
        elif directive.kind == "endif":
            if not stack:
                _emit_stray(directive, "@#if")
            elif stack[-1].kind in conditional_openers:
                stack.pop()
                seen_else_stack.pop()
            else:
                _emit_mismatch(directive, stack[-1].kind)
        elif directive.kind == "endfor":
            if not stack:
                _emit_stray(directive, "@#for")
            elif stack[-1].kind == "for":
                stack.pop()
                seen_else_stack.pop()
            else:
                _emit_mismatch(directive, stack[-1].kind)

    if had_mismatch:
        return diagnostics

    # A closer may sit inside a ``/* */`` block comment, which comment
    # stripping removed before macro-directive extraction.  Dynare's macro
    # processor pairs ``@#for``/``@#endfor`` (and ``@#if``/``@#endif``)
    # regardless of comments, so if the raw source is balanced for an opener's
    # kind, the block is in fact closed -- don't report it as unterminated.
    raw = getattr(model, "original_text", "") or model.text
    raw_for = len(re.findall(r"(?<!\w)@#\s*for\b", raw))
    raw_endfor = len(re.findall(r"(?<!\w)@#\s*endfor\b", raw))
    raw_if = len(re.findall(r"(?<!\w)@#\s*(?:if|ifdef|ifndef)\b", raw))
    raw_endif = len(re.findall(r"(?<!\w)@#\s*endif\b", raw))
    for_balanced = raw_endfor >= raw_for
    if_balanced = raw_endif >= raw_if

    for opener in stack:
        if opener.kind == "for" and for_balanced:
            continue
        if opener.kind in conditional_openers and if_balanced:
            continue
        expected = _closer_for(opener.kind)
        diagnostics.append(
            Diagnostic(
                range=opener.range,
                severity=Severity.ERROR,
                message=(
                    f"Unterminated {_label(opener.kind)} block -- no matching "
                    f"{expected} before end of file. Fix: add {expected} to close "
                    f"this block."
                ),
                code="E062",
            )
        )
    return diagnostics


def _check_macro_error_directives(model: ParsedModel) -> List[Diagnostic]:
    """Emit an error for active Dynare ``@#error`` directives."""
    source = model.text
    # Reuse the per-line activity vector cached at parse time rather than
    # re-running the expensive _macro_branch_state scan (~4.9 ms) on every
    # keystroke.  Fall back to a fresh computation when the cache is absent
    # (e.g. a context model built via dataclasses.replace from a partial
    # parse, or a model loaded without calling parse()).
    cached_active = model._cached_active_macro_lines
    if cached_active is not None:
        active_lines = cached_active
    else:
        _defines, active_lines, _line_defines = _macro_branch_state(source)
    lines = source.splitlines()
    diagnostics: List[Diagnostic] = []
    for directive in model.macro_directives:
        if directive.kind != "error":
            continue
        line = directive.range.start.line
        # Dynare allows whitespace between ``@#`` and the directive name
        # (``@# error "msg"``), so a literal "@#error" containment test
        # would silently skip the spaced spelling.
        if line < len(lines) and not re.search(
            r"@#\s*error\b", lines[line], re.IGNORECASE
        ):
            continue
        if line < len(active_lines) and not active_lines[line]:
            continue
        message = (directive.argument or "").strip().strip("\"'")
        diagnostics.append(
            Diagnostic(
                range=directive.range,
                severity=Severity.ERROR,
                message=(
                    f"Macro @#error triggered{f': {message}' if message else ''}."
                ),
                code="E064",
            )
        )
    return diagnostics


_MACRO_INTERPOLATION_PATTERN = re.compile(r"@\{([A-Za-z_][A-Za-z0-9_]*)\}")
_MACRO_INTERPOLATION_EXPR_PATTERN = re.compile(r"@\{([^{}\n]+)\}")
_MACRO_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _known_macro_names(model: ParsedModel) -> set:
    """Names introduced by ``@#define`` / ``@#for`` anywhere in the file."""
    known: set = set()
    for directive in model.macro_directives:
        argument = directive.argument or ""
        if directive.kind == "define":
            m = re.match(r"\s*([A-Za-z_]\w*)", argument)
            if m:
                known.add(m.group(1))
        elif directive.kind == "for":
            m = re.match(
                r"\s*\(?\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*\)?\s+in\b",
                argument,
            )
            if m:
                known.update(_MACRO_IDENTIFIER_PATTERN.findall(m.group(1)))
    return known


# Built-in functions/constants of Dynare's macro language: calling these in
# an interpolation is valid even though no @#define introduces them.
_MACRO_BUILTIN_NAMES = frozenset(
    {
        "true",
        "false",
        "inf",
        "nan",
        "length",
        "isempty",
        "isboolean",
        "isreal",
        "isstring",
        "isarray",
        "istuple",
        "isdefined",
        "defined",
        "exp",
        "log",
        "ln",
        "log10",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sqrt",
        "cbrt",
        "sign",
        "floor",
        "ceil",
        "trunc",
        "round",
        "mod",
        "max",
        "min",
        "sum",
        "erf",
        "erfc",
        "gamma",
        "lgamma",
        "abs",
        "normpdf",
        "normcdf",
    }
)


_MACRO_DEFINED_CALL_PATTERN = re.compile(
    r"\b(?:is)?defined\s*\(\s*([A-Za-z_]\w*)\s*\)",
    re.IGNORECASE,
)


def _has_unresolved_interpolations(model: ParsedModel) -> bool:
    """Whether macro constructs beyond the simple engine survive in the text.

    True when a ``@{...}`` interpolation survives substitution on a
    non-directive line — OUTSIDE string literals (a tag like
    ``[name='eq @{V}']`` cannot change counts) and OUTSIDE unresolved
    ``@#for`` template ranges (those are already excluded from the
    counting models) — or when an expression-form ``@#include`` resisted
    constant folding.  Either way the static declaration/equation counts
    are not a faithful picture of what Dynare compiles.
    """
    if getattr(model, "has_unfoldable_expression_include", False):
        return True
    text = _mask_string_literals(_strip_non_macro_comments(model.text))
    lines = text.splitlines()
    template_ranges = _macro_for_template_ranges(model)
    for match in _MACRO_INTERPOLATION_EXPR_PATTERN.finditer(text):
        start = _offset_to_position_local(text, match.start())
        if any(s <= start.line <= e for s, e in template_ranges):
            continue
        line = lines[start.line] if 0 <= start.line < len(lines) else ""
        if line.lstrip().startswith("@#"):
            continue
        return True
    return False


def _check_unresolved_macro_interpolations(
    model: ParsedModel,
    include_models: Optional[List[ParsedModel]] = None,
) -> List[Diagnostic]:
    """Emit ``E063`` for active ``@{...}`` interpolations left unresolved.

    The scan runs on :func:`_strip_non_macro_comments` output, which has
    already substituted every interpolation the LSP's macro engine can
    resolve — anything still matching ``@{...}`` here is unresolved.
    Identifiers inside string literals and Dynare macro built-ins are not
    names at all; a name introduced by a ``@#define`` / ``@#for`` in this
    file is treated as known (the directive may simply be beyond the
    simple evaluator — the real preprocessor still validates on save).
    An interpolation is flagged when it contains at least one definitely
    unknown name; expression interpolations additionally require the file
    to have no ``@#include`` that could provide unseen defines.
    """
    diagnostics: List[Diagnostic] = []
    text = _strip_non_macro_comments(model.text)
    lines = text.splitlines()
    known = _known_macro_names(model)
    # Names @#define'd / @#for-bound in included files are visible at the
    # point of use even when their VALUES are beyond the simple evaluator.
    for include_model in include_models or []:
        known |= _known_macro_names(include_model)
    has_includes = bool(model.includes)

    for match in _MACRO_INTERPOLATION_EXPR_PATTERN.finditer(text):
        start = _offset_to_position_local(text, match.start())
        line = lines[start.line] if 0 <= start.line < len(lines) else ""
        if line.lstrip().startswith("@#"):
            continue
        expr = match.group(1).strip()
        # ``defined(NAME)`` queries whether NAME exists — its argument is
        # legitimately allowed to be undefined, and a guarded short-circuit
        # use (``defined(F) && F``) keeps the OTHER occurrences of NAME
        # valid too.
        masked_expr = _mask_string_literals(expr)
        guarded = {
            m.group(1) for m in _MACRO_DEFINED_CALL_PATTERN.finditer(masked_expr)
        }
        scannable = _MACRO_DEFINED_CALL_PATTERN.sub("", masked_expr)
        identifiers = [
            ident
            for ident in _MACRO_IDENTIFIER_PATTERN.findall(scannable)
            if ident.lower() not in _MACRO_BUILTIN_NAMES
        ]
        if not identifiers:
            continue
        unknown_names = [
            ident
            for ident in dict.fromkeys(identifiers)
            if ident not in known and ident not in guarded
        ]
        if not unknown_names:
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expr):
            message = (
                f"Undefined macro interpolation '@{{{expr}}}'. "
                f"Fix: define '{expr}' with @#define before this line, "
                "or remove the macro interpolation."
            )
        else:
            if has_includes:
                # An @#include may provide defines this single-file scan
                # can't see; leave expression validation to the
                # preprocessor.
                continue
            unknown = ", ".join(unknown_names)
            message = (
                f"Undefined macro name(s) in interpolation "
                f"'@{{{expr}}}': {unknown}. Fix: define them with "
                "@#define before this line, or remove the macro "
                "interpolation."
            )
        diagnostics.append(
            Diagnostic(
                range=SourceRange(
                    start,
                    _offset_to_position_local(text, match.end()),
                ),
                severity=Severity.ERROR,
                message=message,
                code="E063",
            )
        )

    return diagnostics


def _check_shocks_block_var_declared(
    model: ParsedModel,
    exogenous_names: Optional[set] = None,
    observed_endogenous: Optional[set] = None,
) -> List[Diagnostic]:
    """Flag ``shocks; var X; ...`` where ``X`` isn't declared ``varexo``.

    Dynare requires every name appearing as a ``var`` inside a shocks
    block to also be declared as an exogenous variable (``varexo``).
    Without this check, a typo'd or missing ``varexo`` declaration is
    silently accepted by the LSP even though Dynare's preprocessor
    would reject the file.

    Exception: ``var <observed endogenous>; stderr ...;`` is Dynare's
    calibrated measurement-error syntax — the preprocessor accepts an
    endogenous name there when the variable is also listed in ``varobs``.
    """
    diagnostics: List[Diagnostic] = []
    if model.shocks_block_range is None or not model.shocks_vars:
        return diagnostics
    exo_names = (
        exogenous_names
        if exogenous_names is not None
        else {v.name for v in model.exogenous}
    )
    observed = (
        observed_endogenous
        if observed_endogenous is not None
        else {v.name for v in model.endogenous} & set(model.varobs_vars)
    )
    for name in model.shocks_vars:
        if name in exo_names:
            continue
        if name in observed:
            continue
        # Anchor the diagnostic at the shocks block opener; we don't
        # track per-name ranges inside the shocks block.
        diagnostics.append(
            Diagnostic(
                range=model.shocks_block_range,
                severity=Severity.ERROR,
                message=(
                    f"Shock '{name}' referenced in shocks block but not "
                    f"declared in 'varexo'. Fix: add '{name}' to a 'varexo' "
                    f"declaration."
                ),
                code="E020",
            )
        )
    return diagnostics


def _check_included_shocks_block_var_declared(
    model: ParsedModel,
    include_models: Optional[List[ParsedModel]],
    exogenous_names: set,
    observed_endogenous: Optional[set] = None,
) -> List[Diagnostic]:
    diagnostics: List[Diagnostic] = []
    if not include_models:
        return diagnostics
    if model.includes:
        rng = model.includes[0].range
    elif model.model_block_range is not None:
        rng = model.model_block_range
    else:
        rng = SourceRange(Position(0, 0), Position(0, 1))

    observed = observed_endogenous if observed_endogenous is not None else set()
    seen: set[str] = set()
    for include_model in include_models:
        for name in include_model.shocks_vars:
            if name in exogenous_names or name in seen:
                continue
            if name in observed:
                # Measurement error on an observed endogenous — valid.
                continue
            seen.add(name)
            diagnostics.append(
                Diagnostic(
                    range=rng,
                    severity=Severity.ERROR,
                    message=(
                        f"Shock '{name}' referenced in an included shocks block "
                        f"but not declared in 'varexo'. Fix: add '{name}' to a "
                        "'varexo' declaration."
                    ),
                    code="E020",
                )
            )
    return diagnostics


def _check_missing_shocks_block(
    model: ParsedModel,
    has_shocks_block: Optional[bool] = None,
) -> List[Diagnostic]:
    """Warn when exogenous variables are declared but no shocks block exists."""
    diagnostics: List[Diagnostic] = []
    stochastic_exogenous = [
        v
        for v in model.exogenous
        if v.name not in model.deterministic_exogenous_names()
    ]
    if not stochastic_exogenous:
        return diagnostics
    if has_shocks_block is None:
        has_shocks_block = model.shocks_block_range is not None or bool(
            model.shocks_vars
        )
    if has_shocks_block:
        return diagnostics
    # Exogenous variables declared but no shocks block
    exo_names = [v.name for v in stochastic_exogenous]
    diagnostics.append(
        Diagnostic(
            range=stochastic_exogenous[0].range,
            severity=Severity.WARNING,
            message=(
                f"Exogenous variable(s) declared ({', '.join(exo_names[:5])}) "
                f"but no 'shocks' block found. Add a shocks block to define "
                f"the shock processes."
            ),
            code="W060",
        )
    )
    return diagnostics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_diagnostics(
    model: ParsedModel,
    include_symbols: Optional[Dict[str, List[VarDeclaration]]] = None,
    include_models: Optional[List[ParsedModel]] = None,
    include_cycles: Optional[List[List[str]]] = None,
    unresolved_includes: Optional[List[IncludeDirective]] = None,
) -> List[Diagnostic]:
    """Run all diagnostic checks on a parsed model.

    Uses cascade suppression: when structural errors (E001) are present,
    downstream checks that would produce noisy/misleading results are
    suppressed.  Instead, the E001 message includes actionable fix guidance.

    When *include_symbols* is provided, names declared in transitively
    included files are treated as visible — so an identifier declared
    only in ``foo.mod`` won't trigger ``E020`` in a file that
    ``@#include`` s it.  When *include_models* is provided, whole-model
    checks such as equation counts and steady-state validation also see
    declarations and calibration values from included files.  When
    *include_cycles* is provided, each cycle touching this file is
    reported as ``E060`` on the first ``@#include`` directive.  When
    *unresolved_includes* is provided, each directive whose target file
    isn't on disk is reported as ``E061`` on the directive's own source
    range.  All optional arguments default to ``None`` so the historical
    single-file signature stays backwards-compatible.
    """
    diagnostics: List[Diagnostic] = []
    context_model = model_with_include_context(
        model,
        include_models,
        include_model_equations=False,
    )
    equation_count_model = model_with_include_context(model, include_models)
    reference_model = model_with_include_context(model, include_models)
    steady_state_context_model = model_with_include_context(model, include_models)
    if include_models and model.model_block_range is not None:
        # Keep only THIS file's model-block equations.  Dynare concatenates
        # multiple model blocks, so filter against the union of all the
        # file's block ranges — testing only the first block silently
        # dropped later blocks' equations from steady-state validation.
        own_block_ranges = model.model_block_ranges or [model.model_block_range]
        steady_state_context_model.model_equations = [
            eq
            for eq in steady_state_context_model.model_equations
            if any(_ranges_nested_or_equal(rng, eq.range) for rng in own_block_ranges)
        ]
    context_model = _with_model_editing_commands(context_model)
    equation_count_model = _with_model_editing_commands(equation_count_model)
    reference_model = _with_model_editing_commands(reference_model)
    steady_state_context_model = _with_model_editing_commands(
        steady_state_context_model
    )
    usage_model = replace(equation_count_model)
    usage_model.model_equations = list(equation_count_model.model_equations)
    macro_for_template_ranges = _macro_for_template_ranges(model)
    has_macro_for_templates = bool(macro_for_template_ranges)
    # An UNRESOLVED @#for overlapping the steady_state_model block means the
    # static view holds only loop templates, so per-variable coverage and
    # residual checks (W040/W041/W042) would be false alarms.
    ss_block_has_macro_templates = model.steady_state_block_range is not None and any(
        start <= model.steady_state_block_range.end.line
        and end >= model.steady_state_block_range.start.line
        for start, end in macro_for_template_ranges
    )
    # Tripwire: a macro directive leaking into a parsed steady-state
    # equation (e.g. an @#if inside a RESOLVED @#for body, which the
    # expander does not branch-evaluate) means the static SS view is
    # mangled — treat exactly like an unresolved template.
    if not ss_block_has_macro_templates and any(
        "@#" in eq.text for eq in model.steady_state_equations
    ):
        ss_block_has_macro_templates = True
    if has_macro_for_templates:
        equation_count_model = _without_macro_for_template_equations(
            equation_count_model,
            macro_for_template_ranges,
        )
        reference_model = _without_macro_for_template_equations(
            reference_model,
            macro_for_template_ranges,
        )
        usage_model.model_equations = [
            eq
            for eq in usage_model.model_equations
            if not _line_in_ranges(eq.range.start.line, macro_for_template_ranges)
        ]
    has_macro_branch_decl_alternatives = _has_mutually_exclusive_macro_declarations(
        model
    )
    has_macro_branch_equation_alternatives = _has_mutually_exclusive_macro_equations(
        model
    ) or _has_macro_branch_equations(model)
    usage_model.steady_state_equations = list(
        equation_count_model.steady_state_equations
    )

    # Phase 0: Cross-file structural — circular includes always reported,
    # regardless of the single-file structural state.
    if include_cycles:
        diagnostics.extend(_check_circular_includes(model, include_cycles))
    if unresolved_includes:
        diagnostics.extend(_check_unresolved_includes(unresolved_includes))
    # Unmatched @#if / @#for run from the parsed macro directives — no
    # cross-file information needed, so always run.
    diagnostics.extend(_check_unmatched_macro_blocks(model))
    macro_errors = _check_macro_error_directives(model)
    macro_errors.extend(_check_unresolved_macro_interpolations(model, include_models))
    diagnostics.extend(macro_errors)

    # Phase 1: Structural errors (E001) — always run
    parse_errors = _check_parse_errors(model)
    parse_errors = _filter_include_closed_parse_errors(parse_errors, include_models)
    parse_errors.extend(_check_invalid_identifier_declarations(model))
    parse_errors.extend(_check_included_parse_errors(include_models))
    diagnostics.extend(parse_errors)
    # Also check for merged equations (missing semicolons within model block),
    # including model fragments that came from textual includes.
    merged_eq_errors = _check_merged_equations(equation_count_model)
    diagnostics.extend(merged_eq_errors)
    merged_assignment_errors = _check_merged_assignments(context_model)
    diagnostics.extend(merged_assignment_errors)
    unbalanced_paren_errors = _check_unbalanced_parens(equation_count_model)
    diagnostics.extend(unbalanced_paren_errors)

    has_structural_error = (
        len(parse_errors) > 0
        or len(merged_eq_errors) > 0
        or len(merged_assignment_errors) > 0
        or len(unbalanced_paren_errors) > 0
        or len(macro_errors) > 0
    )

    # Phase 2: Declaration-level checks and downstream checks
    if has_structural_error:
        # When there are structural errors (missing end;, missing semicolons),
        # the parser may produce incomplete/incorrect declarations, equation
        # counts, and references.  Suppress cascading errors to keep focus on
        # the structural fix.
        #
        # E030 (duplicates) — suppress: missing semicolons merge declarations.
        # E010 (equation count) — suppress: missing end; corrupts equation list.
        # E020 (undeclared refs) — suppress: partial parsing may include
        #   Dynare commands or text outside blocks as equations.
        # Warnings — suppress all downstream warnings to reduce noise.
        pass
    else:
        # No structural errors — run all checks normally
        diagnostics.extend(_check_duplicate_declarations(model, include_symbols))
        diagnostics.extend(
            _check_predetermined_variables(
                model,
                context_model.endogenous_names(),
            ),
        )
        diagnostics.extend(_check_timed_parameters(reference_model))
        diagnostics.extend(_check_timed_deterministic_exogenous(reference_model))
        diagnostics.extend(_check_model_local_shadowing(reference_model))

        # Optimal-policy models (Ramsey / discretionary) declare a
        # ``planner_objective``; Dynare then *adds* the planner's first-order
        # conditions, so the user writes fewer equations than there are
        # endogenous variables.  An equation-count check would false-positive,
        # so skip it for these models (Dynare balances them internally).
        _policy_commands = set(model.policy_commands) | set(
            context_model.policy_commands
        )
        is_optimal_policy = (
            model.planner_objective_range is not None
            or context_model.planner_objective_range is not None
            or bool(
                _policy_commands
                & {"ramsey_model", "ramsey_policy", "discretionary_policy"}
            )
        )
        # OccBin models declare regime constraints that Dynare turns into extra
        # equations, so the written model need not balance on its own.
        is_occbin = model.has_occbin or context_model.has_occbin
        # When any of these hold, our static equation count is not a faithful
        # picture of what Dynare compiles -- suppress both the count check and
        # the E010+E020 linking heuristic below.  (``@#for`` is intentionally
        # NOT suppressed: the project resolves loop expansions and still checks
        # their body diagnostics.)
        # A surviving @{...} interpolation (valid for Dynare, beyond the
        # simple substitution engine — e.g. ``var y_@{length(L)};``) means
        # declarations/equations were dropped or mangled, so the static
        # count is meaningless; E010 would otherwise survive even a
        # preprocessor SUCCESS.
        has_unresolved_interpolations = _has_unresolved_interpolations(model)
        equation_count_unreliable = (
            is_optimal_policy
            or is_occbin
            or has_macro_branch_decl_alternatives
            or has_macro_for_templates
            or has_unresolved_interpolations
        )
        if equation_count_unreliable:
            eq_count_diags = []
        elif has_macro_branch_equation_alternatives:
            eq_count_diags = _check_macro_branch_equation_count(
                equation_count_model,
            )
        else:
            eq_count_diags = _check_equation_count(equation_count_model)
        undeclared_diags = _check_undeclared_references(
            reference_model,
            include_symbols,
        )
        if has_unresolved_interpolations:
            # Dropped/mangled declarations make undeclared-name results
            # guesses (``var z@{"a"};`` would flag za as a typo); the
            # preprocessor arbitrates on save.
            undeclared_diags = []

        # Link E010 + E020 when they point to the same root cause:
        # If #equations > #endo_vars AND there are undeclared identifiers,
        # the fix is to add those identifiers to `var` declaration.
        real_eqs = equation_count_model.dynamic_model_equations()
        n_eq = len(real_eqs)
        n_endo = len(equation_count_model.endogenous)
        if n_eq > n_endo and undeclared_diags and not equation_count_unreliable:
            n_extra = n_eq - n_endo
            n_undecl = len(undeclared_diags)
            if n_undecl <= n_extra:
                # The undeclared identifiers likely ARE the missing variables
                names = []
                for d in undeclared_diags:
                    m = re.search(r"'(\w+)'", d.message)
                    if m:
                        names.append(m.group(1))
                if names:
                    helper_like = {
                        assignment.name
                        for assignment in (
                            model.helper_assignments
                            + context_model.helper_assignments
                            + equation_count_model.helper_assignments
                        )
                    }
                    if helper_like.intersection(names):
                        rng = (
                            eq_count_diags[0].range
                            if eq_count_diags
                            else undeclared_diags[0].range
                        )
                        diagnostics.append(
                            Diagnostic(
                                range=rng,
                                severity=Severity.ERROR,
                                message=(
                                    f"Equation count mismatch: {n_eq} equation(s) but "
                                    f"{n_endo} endogenous variable(s). No automatic "
                                    "declaration fix is offered because one or more "
                                    "undeclared identifier(s) also appear as top-level "
                                    "helper assignments."
                                ),
                                code="E010",
                                fix=None,
                            )
                        )
                        diagnostics.extend(undeclared_diags)
                    else:
                        combined_msg = (
                            f"Equation count mismatch: {n_eq} equation(s) but {n_endo} endogenous "
                            f"variable(s). The undeclared identifier(s) {', '.join(names)} are likely "
                            f"the missing variable(s). Fix: add {', '.join(names)} to the 'var' declaration."
                        )
                        rng = (
                            eq_count_diags[0].range
                            if eq_count_diags
                            else undeclared_diags[0].range
                        )
                        # Compute auto-fix: add all names to var declaration
                        insert_edit = _find_declaration_insert_point(
                            model.text, "var", model
                        )
                        combined_fix = None
                        if insert_edit:
                            combined_fix = TextEdit(
                                start_line=insert_edit.start_line,
                                start_char=insert_edit.start_char,
                                end_line=insert_edit.end_line,
                                end_char=insert_edit.end_char,
                                new_text=" " + " ".join(names),
                            )
                        diagnostics.append(
                            Diagnostic(
                                range=rng,
                                severity=Severity.ERROR,
                                message=combined_msg,
                                code="E010",
                                fix=combined_fix,
                            )
                        )
                else:
                    diagnostics.extend(eq_count_diags)
                    diagnostics.extend(undeclared_diags)
            else:
                diagnostics.extend(eq_count_diags)
                diagnostics.extend(undeclared_diags)
        else:
            diagnostics.extend(eq_count_diags)
            diagnostics.extend(undeclared_diags)

        # Link E010 + W020 when they point to the same root cause:
        # If #endo_vars > #equations AND there are unreferenced endogenous vars,
        # the fix is to add the missing equation(s) for those variables.
        #
        # Pre-compute the equation-reference set once; all three unused-symbol
        # checks scan the same usage_model.model_equations loop, so sharing
        # this result eliminates 2 redundant scans (1395 → 837 total
        # _extract_equation_references calls in run_diagnostics on a
        # 279-equation model like US_FRB03).  Set to None when the checks
        # are suppressed so _collect_model_equation_references is never
        # called unnecessarily.
        _usage_eq_refs: Optional[set] = (
            None
            if (
                equation_count_unreliable
                or has_macro_branch_decl_alternatives
                or has_macro_branch_equation_alternatives
                or has_macro_for_templates
            )
            else _collect_model_equation_references(usage_model)
        )
        unused_endo_diags = (
            []
            if _usage_eq_refs is None
            else _check_unused_endogenous(usage_model, _model_eq_refs=_usage_eq_refs)
        )
        if n_endo > n_eq and unused_endo_diags:
            n_missing = n_endo - n_eq
            # Extract names of unreferenced endogenous variables
            unreferenced = []
            for d in unused_endo_diags:
                m = re.search(r"'(\w+)'", d.message)
                if m:
                    unreferenced.append(m.group(1))
            if unreferenced and len(unreferenced) <= n_missing + 2:
                # Remove the separate E010 diagnostics
                diagnostics = [d for d in diagnostics if d.code != "E010"]

                # If the number of unreferenced vars exactly matches the
                # surplus, these extra vars are the problem — remove them.
                # But skip the auto-fix if E030 already has a fix that
                # removes from var (those fixes will change the count too).
                e030_removes_from_var = any(
                    d.code == "E030" and d.fix is not None for d in diagnostics
                )
                combined_fix = None
                if (
                    len(unreferenced) == n_missing
                    and len(unreferenced) == 1
                    and not e030_removes_from_var
                ):
                    combined_msg = (
                        f"Equation count mismatch: {n_eq} equation(s) but {n_endo} endogenous "
                        f"variable(s) ({n_missing} extra variable(s)). "
                        f"The unreferenced variable(s) {', '.join(unreferenced)} "
                        f"should be removed from the 'var' declaration. "
                        f"Fix: remove {', '.join(unreferenced)} from 'var'."
                    )
                    combined_fix = _find_name_in_declaration(
                        model.text, "var", unreferenced[0], model
                    )
                elif len(unreferenced) == n_missing and not e030_removes_from_var:
                    combined_msg = (
                        f"Equation count mismatch: {n_eq} equation(s) but {n_endo} endogenous "
                        f"variable(s) ({n_missing} extra variable(s)). "
                        f"The unreferenced variable(s) {', '.join(unreferenced)} "
                        f"should be removed from the 'var' declaration. "
                        f"Fix: remove {', '.join(unreferenced)} from 'var'."
                    )
                else:
                    combined_msg = (
                        f"Equation count mismatch: {n_eq} equation(s) but {n_endo} endogenous "
                        f"variable(s) ({n_missing} equation(s) missing). "
                        f"The unreferenced variable(s) {', '.join(unreferenced)} "
                        f"likely need equation(s). "
                        f"Fix: look for commented-out or deleted equations involving "
                        f"{', '.join(unreferenced)}, and restore or re-add them to the model block."
                    )

                rng = (
                    eq_count_diags[0].range
                    if eq_count_diags
                    else unused_endo_diags[0].range
                )
                diagnostics.append(
                    Diagnostic(
                        range=rng,
                        severity=Severity.ERROR,
                        message=combined_msg,
                        code="E010",
                        fix=combined_fix,
                    )
                )
                # Don't add the W020 diagnostics separately
            else:
                diagnostics.extend(unused_endo_diags)
        else:
            diagnostics.extend(unused_endo_diags)

        diagnostics.extend(_check_duplicate_equations(equation_count_model))
        diagnostics.extend(_check_duplicate_param_assignments(context_model))
        diagnostics.extend(_check_trivial_equations(equation_count_model))
        diagnostics.extend(_check_stray_equations(model))
        diagnostics.extend(_check_included_stray_equations(include_models))
        diagnostics.extend(
            _check_missing_shocks_block(
                model,
                context_model.shocks_block_range is not None
                or bool(context_model.shocks_vars),
            ),
        )
        diagnostics.extend(
            _check_shocks_block_var_declared(
                model,
                context_model.exogenous_names(),
                set(context_model.endogenous_names()) & set(context_model.varobs_vars),
            ),
        )
        diagnostics.extend(
            _check_included_shocks_block_var_declared(
                model,
                include_models,
                context_model.exogenous_names(),
                set(context_model.endogenous_names()) & set(context_model.varobs_vars),
            ),
        )
        # Estimation-block diagnostics (varobs / estimated_params /
        # observation_trends).  Imported lazily to avoid an import cycle.
        from .estimation_diagnostics import check_estimation

        diagnostics.extend(
            check_estimation(
                context_model,
                context_model.endogenous_names(),
                context_model.exogenous_names(),
                context_model.parameter_names(),
            )
        )
        # Optimal-policy diagnostics (planner_objective / ramsey / osr).
        from .policy_diagnostics import check_policy

        diagnostics.extend(
            check_policy(context_model, context_model.endogenous_names())
        )
        # shocks-block value/consistency checks (|corr|<=1, duplicate specs,
        # negative variance).
        from .shocks_diagnostics import check_shocks

        diagnostics.extend(
            check_shocks(
                model,
                include_models,
                context_model.param_values(),
            )
        )
        # Static usage/calibration checks (no varexo, param lead/lag,
        # non-finite deep parameter).
        from .usage_diagnostics import check_usage

        diagnostics.extend(
            check_usage(
                model,
                context_model.exogenous_names(),
                context_model.deterministic_exogenous_names(),
                context_model.parameter_names(),
                include_models=include_models,
            )
        )
        # Model-form checks (steady_state_model order, linear-model operators,
        # deprecated commands).
        from .model_form_diagnostics import check_model_form

        diagnostics.extend(
            check_model_form(
                context_model,
                context_model.endogenous_names(),
                context_model.exogenous_names(),
                context_model.deterministic_exogenous_names(),
                context_model.parameter_names(),
                include_models=include_models,
            )
        )
        local_parameter_ranges = {_range_key(p.range) for p in model.parameters}
        used_parameters = _used_parameter_names(reference_model)
        diagnostics.extend(
            d
            for d in _check_unassigned_parameters(context_model)
            if (
                _range_key(d.range) in local_parameter_ranges
                or _diagnostic_quoted_name(d) in used_parameters
            )
        )
        local_assignment_ranges = {
            _range_key(a.range)
            for a in (model.param_assignments + model.helper_assignments)
        }
        for eq in model.steady_state_equations:
            if eq.lhs.strip() in context_model.parameter_names():
                local_assignment_ranges.add(_range_key(eq.range))
        diagnostics.extend(
            d
            for d in _check_unevaluable_params(context_model)
            if (
                _range_key(d.range) in local_assignment_ranges
                or _diagnostic_quoted_name(d) in used_parameters
            )
        )
        declared_names = context_model.all_declared_names()
        diagnostics.extend(_check_helper_variables(model, declared_names))
        for include_model in include_models or []:
            diagnostics.extend(
                _check_helper_variables(
                    include_model,
                    declared_names,
                    include_model.include_anchor_range,
                )
            )
        if not has_macro_branch_decl_alternatives and not has_macro_for_templates:
            # Pass the pre-computed model-equation reference set so these two
            # checks don't repeat the usage_model.model_equations scan that
            # _check_unused_endogenous already performed above.
            diagnostics.extend(
                _check_unused_exogenous(
                    usage_model,
                    _model_eq_refs=_usage_eq_refs,
                )
            )
            diagnostics.extend(
                _check_unused_parameters(
                    usage_model,
                    _model_eq_refs=_usage_eq_refs,
                )
            )
        if not ss_block_has_macro_templates:
            diagnostics.extend(_check_ss_block_coverage(context_model))
        if not is_occbin and not ss_block_has_macro_templates:
            # OccBin models declare bind=/relax= regime-equation pairs (one
            # structural slot, two regime variants); both variants get fed to the
            # residual check, so the binding-regime equation produces a spurious
            # residual at the reference (relaxed) steady state. This is the same
            # reason E010 is suppressed for OccBin above, so skip the steady-state
            # residual / missing-steady-state checks too rather than emit a false
            # W040/W041 on a model Dynare solves.
            diagnostics.extend(_check_steady_state(steady_state_context_model))
            diagnostics.extend(_check_missing_steady_state(equation_count_model))
        diagnostics.extend(_check_initval_references(context_model))
        diagnostics.extend(_check_exogenous_in_initval(context_model))
        diagnostics.extend(_check_missing_initval(context_model))
        diagnostics.extend(_check_parameter_bounds(context_model))

    # Cap total errors to avoid overwhelming LLM consumers
    MAX_ERRORS = 10
    errors = [d for d in diagnostics if d.severity == Severity.ERROR]
    warnings = [d for d in diagnostics if d.severity != Severity.ERROR]
    if len(errors) > MAX_ERRORS:
        kept = errors[:MAX_ERRORS]
        kept.append(
            Diagnostic(
                range=errors[MAX_ERRORS].range,
                severity=Severity.INFORMATION,
                message=f"... and {len(errors) - MAX_ERRORS} more error(s). Fix the errors above first.",
                code="E999",
            )
        )
        diagnostics = kept + warnings

    if getattr(model, "source_map", None):
        diagnostics = [
            d
            if d.code in {"E060", "E061"}
            else _map_diagnostic_to_original_source(model, d)
            for d in diagnostics
        ]
    diagnostics = _anchor_synthetic_include_diagnostics(model, diagnostics)

    return diagnostics


def _check_parameter_bounds(model: ParsedModel) -> List[Diagnostic]:
    """Warn when a parameter assignment falls outside its conventional bounds.

    Catches typos and unit errors (e.g. ``beta = 99`` when 0.99 was meant;
    ``sigma_e = -0.1`` when a standard deviation was meant). The bounds table
    in ``bounds.py`` is opinionated but conservative — it flags only values
    that violate the *theoretical* admissible range for the standard
    interpretation of the parameter.

    Emits W070.
    """
    from . import bounds as _bounds_module

    diagnostics: List[Diagnostic] = []
    assignments: List[ParamAssignment] = list(model.param_assignments)
    parameter_names = model.parameter_names()
    if parameter_names:
        _values, ss_assignments = evaluate_steady_state_model_assignments(model)
        for ss_assignment in ss_assignments:
            name = ss_assignment.name
            if name not in parameter_names:
                continue
            expr = ss_assignment.expression.strip().rstrip(";").strip()
            if not expr:
                continue
            assignments.append(
                ParamAssignment(
                    name=name,
                    expression=expr,
                    value=ss_assignment.value,
                    range=ss_assignment.range,
                )
            )

    latest: Dict[str, ParamAssignment] = {}
    for assignment in assignments:
        latest[assignment.name] = assignment

    for a in latest.values():
        # Try to interpret the RHS as a literal number. We deliberately
        # do not evaluate expressions — that's W011's job. A typo like
        # ``beta = 99`` is almost always a bare numeric literal, and
        # evaluating arbitrary expressions here would be both expensive
        # and prone to false positives.
        rhs = (a.expression or "").strip().rstrip(";").strip()
        try:
            value = a.value if a.value is not None else float(rhs)
        except (ValueError, TypeError):
            continue

        bound = _bounds_module.lookup(a.name)
        if bound is None:
            continue
        if _bounds_module.is_in_bounds(value, bound):
            continue

        rng = format_range_str(bound)
        msg = (
            f"Parameter '{a.name}' = {value:g} is outside the conventional "
            f"range {rng}: {bound.rationale}. "
            f"This is a warning, not an error — override if intentional."
        )
        diagnostics.append(
            Diagnostic(
                range=a.range,
                severity=Severity.WARNING,
                message=msg,
                code="W070",
            )
        )
    return diagnostics


def format_range_str(bound) -> str:
    """Thin local re-export of bounds.format_range so callers don't need an extra import."""
    from . import bounds as _bounds_module

    return _bounds_module.format_range(bound)


def analyze_text(text: str) -> List[Diagnostic]:
    """Parse text and run all diagnostics. Convenience function."""
    model = parse(text)
    return run_diagnostics(model)


def _apply_edits(text: str, edits: List[TextEdit]) -> str:
    """Apply a list of TextEdits to the text, in reverse order to preserve positions.

    Edits whose ranges overlap an already-applied edit are skipped — when
    two diagnostics target the same equation (e.g. E050 + E051 on a
    duplicate trivial equation) the second edit would otherwise land
    in the wrong place after the first one shifted line content, and
    could delete an unrelated structural line like ``end;``.  Pure
    insertions (start == end) only conflict if their position falls
    strictly inside the consumed region of a prior edit.
    """
    lines = text.split("\n")

    # Sort edits by position, descending (last edit first)
    sorted_edits = sorted(
        edits,
        key=lambda e: (e.start_line, e.start_char),
        reverse=True,
    )

    def _ranges_overlap(a: TextEdit, b: TextEdit) -> bool:
        # Returns True if a's [start, end) and b's [start, end) overlap
        # as line/column pairs.  Equal points count as overlap when the
        # candidate is a deletion (start != end); pure insertions at the
        # same point are not flagged.
        a_start = (a.start_line, a.start_char)
        a_end = (a.end_line, a.end_char)
        b_start = (b.start_line, b.start_char)
        b_end = (b.end_line, b.end_char)
        return a_start < b_end and b_start < a_end

    applied: List[TextEdit] = []
    for edit in sorted_edits:
        if any(_ranges_overlap(edit, prior) for prior in applied):
            continue
        sl, sc = edit.start_line, edit.start_char
        el, ec = edit.end_line, edit.end_char

        # Clamp to actual line lengths
        if sl < 0 or sl >= len(lines) or el < 0:
            continue
        sc = min(sc, len(lines[sl]))
        if el >= len(lines):
            el = len(lines) - 1
            ec = len(lines[el])
        else:
            ec = min(ec, len(lines[el]))

        # Build the replacement
        before = lines[sl][:sc]
        after = lines[el][ec:]
        new_content = before + edit.new_text + after

        # Replace the affected lines
        new_lines = new_content.split("\n")
        lines[sl : el + 1] = new_lines
        applied.append(edit)

    return "\n".join(lines)


def auto_fix(text: str) -> str:
    """Apply all safe, deterministic auto-fixes to a .mod file.

    Uses a two-pass approach:
      1. Fix structural errors (E001: missing end;, missing semicolons,
         keyword typos) — always applied (these are definitionally correct).
      2. Fix semantic errors (E020: undeclared refs, E030: duplicates,
         E010+E020 combined, E050/E051/E052: duplicate/trivial equations).

    Pass 2 verifies the fix doesn't increase total error count.
    Returns the fixed text.

    Macro-substitution safeguard
    ----------------------------
    If the source contains ``@#define`` directives, parsing rewrites
    ``@{VAR}`` interpolations and ``model.text`` becomes the substituted
    version.  Diagnostic ranges then index into the substituted text and
    would not align with positions in the user's raw source.  Applying
    such edits to the raw text via _apply_edits would corrupt the file
    (e.g. ``y_@{C} = alph;`` could become ``y_@{C} alphaph;``).  Refuse
    auto-fix in that case — the user's source is returned unchanged.
    The LSP server still publishes diagnostics; only the auto-rewrite is
    suppressed.
    """

    def _count_errors(t: str) -> int:
        return len(
            [d for d in run_diagnostics(parse(t)) if d.severity == Severity.ERROR]
        )

    # Detect macro substitution upfront — any ``@#define`` directive
    # (or a ``@#for`` template, which exposes the loop variable to the
    # body via ``@{X}``) PLUS a ``@{VAR}`` interpolation means offsets
    # will shift between the user's source and the parser's view, so
    # auto-fix would corrupt the file.
    _has_define_or_for = (
        re.search(
            r"@\#\s*(?:define|for)\b",
            text,
        )
        is not None
    )
    _has_interp = re.search(r"@\{[A-Za-z_][A-Za-z0-9_]*\}", text) is not None
    if _has_define_or_for and _has_interp:
        return text

    # Pass 1: Structural fixes (E001) — iterate until stable
    # (fixing one structural error can reveal another, e.g., missing end;
    #  reveals a keyword typo in the next block)
    for _ in range(3):  # max 3 iterations to prevent infinite loops
        model = parse(text)
        diags = run_diagnostics(model)
        structural_fixes = [
            d.fix for d in diags if d.code == "E001" and d.fix is not None
        ]
        if not structural_fixes:
            break
        text = _apply_edits(text, structural_fixes)

    # Re-parse after all structural fixes
    model = parse(text)
    diags = run_diagnostics(model)

    # Pass 2: Semantic fixes (E020, E030, E010, E050, E051, E052)
    semantic_codes = ("E020", "E030", "E010", "E050", "E051", "E052", "E053")
    semantic_fixes = [
        d.fix for d in diags if d.code in semantic_codes and d.fix is not None
    ]
    if semantic_fixes:
        candidate = _apply_edits(text, semantic_fixes)
        if _count_errors(candidate) <= _count_errors(text):
            text = candidate

    return text
