"""Dynare .mod file parser with position tracking for LSP diagnostics.

Parses the major blocks of a Dynare .mod file:
  - var / varexo / parameters declarations
  - parameter assignments
  - model block (equations with optional names)
  - steady_state_model block
  - initval / endval blocks
  - shocks block

Every parsed element carries a SourceRange so the LSP can report
precise diagnostic locations.
"""

from __future__ import annotations

import ast
import bisect
import keyword
import logging
import math
import re
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source location helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Position:
    """0-based line and character offset."""

    line: int
    character: int


@dataclass(frozen=True)
class SourceRange:
    """Half-open range [start, end) in the source text."""

    start: Position
    end: Position


# Single-slot cache for _offset_to_position line-start index.
# Stores the text object itself (not id()) so stale reuse is impossible:
# the cache is only valid when the stored reference IS the same object.
# This is safe on the single-threaded parse path and requires no changes
# to any caller signatures.
_LINE_STARTS_CACHE: "tuple[str, list[int]] | None" = None


def _build_line_starts(text: str) -> "list[int]":
    """Return a sorted list of character offsets where each line begins.

    Index 0 is always 0 (the start of the first line).  Index k is the
    offset of the first character of line k (0-based).  Used by
    _offset_to_position for O(log n) lookup via bisect.
    """
    starts = [0]
    pos = 0
    while True:
        nl = text.find("\n", pos)
        if nl == -1:
            break
        starts.append(nl + 1)
        pos = nl + 1
    return starts


def _offset_to_position(text: str, offset: int) -> Position:
    """Convert a character offset into a (line, character) Position.

    Uses a per-call bisect over a precomputed line-start index that is
    cached in a single-slot store keyed by object identity (``is``).
    The first call for a given text object pays the O(n) build cost;
    every subsequent call for the *same* object is O(log n).
    """
    global _LINE_STARTS_CACHE
    if _LINE_STARTS_CACHE is not None and _LINE_STARTS_CACHE[0] is text:
        line_starts = _LINE_STARTS_CACHE[1]
    else:
        line_starts = _build_line_starts(text)
        _LINE_STARTS_CACHE = (text, line_starts)
    line = bisect.bisect_right(line_starts, offset) - 1
    character = offset - line_starts[line]
    return Position(line, character)


def _range_from_match(text: str, m: re.Match, group: int = 0) -> SourceRange:
    return SourceRange(
        _offset_to_position(text, m.start(group)),
        _offset_to_position(text, m.end(group)),
    )


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------


@dataclass
class VarDeclaration:
    """A single variable name from a var / varexo / parameters block."""

    name: str
    range: SourceRange
    long_name: str = ""
    log_transform: bool = False
    options: str = ""


@dataclass
class ParamAssignment:
    """A parameter value assignment: ``name = expr ;``."""

    name: str
    expression: str
    value: Optional[float]
    range: SourceRange


@dataclass
class Equation:
    """A single equation from the model or steady_state_model block."""

    text: str
    name: str
    range: SourceRange
    lhs: str = ""
    rhs: str = ""
    mcp_constraints: List[str] = field(default_factory=list)
    on_the_fly_declarations: List[Tuple[str, str, SourceRange]] = field(
        default_factory=list,
    )
    tags: List[str] = field(default_factory=list)


@dataclass
class ModelReplacement:
    """A Dynare ``model_replace`` block and the equations it contributes."""

    names: List[str]
    equations: List[Equation] = field(default_factory=list)


@dataclass
class InitvalEntry:
    """An entry from an initval / endval block."""

    name: str
    expression: str
    value: Optional[float]
    range: SourceRange


@dataclass
class EstimatedParam:
    """An entry from an ``estimated_params`` block.

    ``kind`` is ``"param"`` for a parameter, ``"stderr"`` for a shock standard
    error (``stderr e``), or ``"corr"`` for a shock correlation
    (``corr e1, e2``).  Bound/init fields are populated only when the entry
    clearly uses the ``name, init, lower, upper, ...`` numeric form.
    """

    name: str
    kind: str
    corr_with: str = ""
    init: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    prior_shape: str = ""
    range: Optional[SourceRange] = None


@dataclass
class BlockRange:
    """Source range covering a whole block keyword … end;"""

    keyword: str
    range: SourceRange


@dataclass
class IncludeDirective:
    """A Dynare macro ``@#include`` directive.

    Dynare accepts two surface forms — quoted (``@#include "foo.mod"``) and
    bare (``@#include foo.mod``).  Only the filename string is captured here;
    path resolution against the workspace search paths is the responsibility
    of :mod:`dynare_lsp.include_resolver`.
    """

    filename: str
    range: SourceRange
    macro_defines: Dict[str, str] = field(default_factory=dict)


@dataclass
class MacroDirective:
    """A Dynare macro preprocessor directive other than ``@#include``.

    Captures the ``kind`` (``define``, ``includepath``, ``if``, ``elseif``,
    ``else``, ``endif``, ``for``, ``endfor``, ``echo``, ``error``) and the
    raw argument text following it (``None`` for control-flow tokens like
    ``endif``).  Stored on :class:`ParsedModel` so the LSP can introspect
    macro structure for diagnostics (e.g. unmatched ``@#if``) and the
    workspace index can register ``@#includepath`` entries.
    """

    kind: str
    argument: Optional[str]
    range: SourceRange


@dataclass
class ParsedModel:
    """Complete parsed representation of a .mod file."""

    text: str
    original_text: str = ""
    source_map: Optional[List[int]] = None
    nostrict: bool = False

    # declarations
    endogenous: List[VarDeclaration] = field(default_factory=list)
    exogenous: List[VarDeclaration] = field(default_factory=list)
    deterministic_exogenous: List[VarDeclaration] = field(default_factory=list)
    parameters: List[VarDeclaration] = field(default_factory=list)
    predetermined_variables: List[VarDeclaration] = field(default_factory=list)

    # macro @#include directives
    includes: List[IncludeDirective] = field(default_factory=list)
    # An expression-form @#include whose argument could not be constant-
    # folded — its contents are invisible, so static counts are unreliable.
    has_unfoldable_expression_include: bool = False

    # other macro preprocessor directives (@#define, @#if, @#for, ...)
    macro_directives: List[MacroDirective] = field(default_factory=list)

    # parameter values
    param_assignments: List[ParamAssignment] = field(default_factory=list)
    # helper variable assignments (non-declared names assigned in param section)
    helper_assignments: List[ParamAssignment] = field(default_factory=list)

    # model block
    is_linear: bool = False
    model_equations: List[Equation] = field(default_factory=list)
    model_block_range: Optional[SourceRange] = None
    # All ``model; ... end;`` block ranges (Dynare concatenates them);
    # ``model_block_range`` stays the first block for back-compat anchors.
    model_block_ranges: List[SourceRange] = field(default_factory=list)
    model_remove_names: List[str] = field(default_factory=list)
    model_replacements: List[ModelReplacement] = field(default_factory=list)
    var_removed_names: List[str] = field(default_factory=list)

    # steady_state_model block
    steady_state_equations: List[Equation] = field(default_factory=list)
    steady_state_block_range: Optional[SourceRange] = None

    # initval
    initval_entries: List[InitvalEntry] = field(default_factory=list)
    initval_block_range: Optional[SourceRange] = None

    # endval
    endval_entries: List[InitvalEntry] = field(default_factory=list)
    endval_block_range: Optional[SourceRange] = None

    # shocks block range (for exclusion during param parsing)
    shocks_block_range: Optional[SourceRange] = None
    # variable names referenced in the shocks block (e.g. "var e_a; stderr ...")
    shocks_vars: List[str] = field(default_factory=list)

    # estimation blocks
    varobs_vars: List[str] = field(default_factory=list)
    varobs_range: Optional[SourceRange] = None
    varexobs_vars: List[str] = field(default_factory=list)
    varexobs_range: Optional[SourceRange] = None
    estimated_params: List["EstimatedParam"] = field(default_factory=list)
    estimated_params_range: Optional[SourceRange] = None
    observation_trends_vars: List[str] = field(default_factory=list)
    observation_trends_ranges: Dict[str, SourceRange] = field(default_factory=dict)

    # optimal-policy constructs
    policy_commands: List[str] = field(default_factory=list)
    policy_command_range: Optional[SourceRange] = None
    planner_objective_range: Optional[SourceRange] = None
    instruments: List[str] = field(default_factory=list)
    planner_discount: Optional[float] = None
    osr_params: List[str] = field(default_factory=list)
    has_optim_weights: bool = False
    # OccBin (occasionally-binding constraints): Dynare handles the regime
    # equations, so equation-count balance does not apply to the written model.
    has_occbin: bool = False

    # blocks detected
    blocks: List[BlockRange] = field(default_factory=list)

    # parse errors accumulated during parsing
    # Each error is (message, range) or (message, range, fix_dict)
    # fix_dict: {"start_line": int, "start_char": int, "end_line": int, "end_char": int, "new_text": str}
    errors: List[Tuple] = field(default_factory=list)

    # Context supplied by textual include expansion.  For example, a file
    # included inside a parent ``model;`` block can contain ``end;`` to close
    # that parent block even though the included file has no local opener.
    include_context: Optional[str] = None
    context_closing_block: Optional[str] = None
    include_anchor_range: Optional[SourceRange] = None

    # ---------------------------------------------------------------------------
    # Diagnostic-performance caches (not part of the semantic model; excluded
    # from equality, hashing, and repr so they are transparent to tests).
    # ---------------------------------------------------------------------------

    # Per-line macro activity vector from the parse-time _macro_branch_state
    # call.  Stored once and reused by _check_macro_error_directives (which
    # previously re-ran _macro_branch_state on every keystroke).
    _cached_active_macro_lines: Optional[List[bool]] = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )

    # Doubly-masked text used by _check_invalid_identifier_declarations
    # (mask_non_code_for_reference_search + mask_macro_directive_lines).
    # Re-computing this on every keystroke costs ~4.5 ms on large models.
    _cached_diag_stripped: Optional[str] = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )

    # _block_ranges result computed on _cached_diag_stripped.  Caching saves
    # ~8.7 ms on a 44-KB model (the dominant cost in the check).
    _cached_diag_block_exclusions: Optional[List[Tuple[int, int]]] = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )

    # Cached result of _unresolved_macro_for_template_ranges, which calls
    # _macro_branch_state + several scans.  Re-used across repeated
    # run_diagnostics calls on the same parse result (~8 ms saved).
    _cached_macro_for_template_ranges: Optional[List[Tuple[int, int]]] = field(
        default=None,
        init=False,
        compare=False,
        repr=False,
    )

    # helpers ---------------------------------------------------------------
    def all_declared_names(self) -> set:
        names: set = set()
        for v in self.endogenous:
            names.add(v.name)
        for v in self.exogenous:
            names.add(v.name)
        for v in self.parameters:
            names.add(v.name)
        return names

    def endogenous_names(self) -> set:
        return {v.name for v in self.endogenous}

    def exogenous_names(self) -> set:
        return {v.name for v in self.exogenous}

    def deterministic_exogenous_names(self) -> set:
        return {v.name for v in self.deterministic_exogenous}

    def parameter_names(self) -> set:
        return {v.name for v in self.parameters}

    def param_values(self) -> Dict[str, float]:
        latest: Dict[str, Optional[float]] = {}
        for a in self.param_assignments:
            latest[a.name] = a.value
        return {name: value for name, value in latest.items() if value is not None}

    def dynamic_model_equations(self) -> List[Equation]:
        return [
            eq
            for eq in self.model_equations
            if not eq.text.strip().startswith("#") and "static" not in eq.tags
        ]

    def static_model_equations(self) -> List[Equation]:
        return [
            eq
            for eq in self.model_equations
            if not eq.text.strip().startswith("#") and "dynamic" not in eq.tags
        ]

    def steady_state_check_equations(self) -> List[Equation]:
        return [
            eq
            for eq in self.model_equations
            if eq.text.strip().startswith("#") or "dynamic" not in eq.tags
        ]

    def initval_values(self) -> Dict[str, float]:
        latest: Dict[str, Optional[float]] = {}
        for e in self.initval_entries:
            latest[e.name] = e.value
        return {name: value for name, value in latest.items() if value is not None}


# ---------------------------------------------------------------------------
# Comment stripping
# ---------------------------------------------------------------------------

_COMMENT_START = re.compile(
    r"//|%|/\*|@#",  # // line comment, % line comment, /* block comment, @# macro
    re.MULTILINE,
)


def _strip_non_macro_comments(text: str) -> str:
    """Mask ``/* */``, ``//``, and ``%`` comments while preserving ``@#`` directives.

    Returns a string of the same length so character offsets remain valid.
    Used by the macro-directive scanners so that a commented-out
    ``@#include`` or ``@#if`` is NOT extracted as a real directive — a
    common idiom is to leave an old include behind a ``//`` or inside a
    ``/* */`` block as documentation.

    Also skips characters inside string literals (``"..."`` and ``'...'``)
    so that e.g. ``@#include "foo//bar.mod"`` doesn't get its filename
    truncated at ``//`` mid-string.
    """
    result = list(text)
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        # Skip over the body of a string literal so comment / macro
        # tokens inside the string are left untouched.  Only single-line
        # strings (no embedded newlines) — that matches the Dynare and
        # macro-language usage.
        if ch in ('"', "'"):
            quote = ch
            j = i + 1
            while j < n and text[j] != quote and text[j] != "\n":
                j += 1
            if j < n and text[j] == quote:
                i = j + 1
            else:
                # Unterminated string: skip just past the opening quote
                # so we don't loop forever; treat the rest of the line
                # as outside any string.
                i = j
            continue
        if ch == "/" and nxt == "*":
            end = text.find("*/", i + 2)
            if end == -1:
                end = n
            else:
                end += 2
            for j in range(i, end):
                if result[j] != "\n":
                    result[j] = " "
            i = end
            continue
        if ch == "/" and nxt == "/":
            end = text.find("\n", i)
            if end == -1:
                end = n
            for j in range(i, end):
                result[j] = " "
            i = end
            continue
        # ``%`` is a line comment only when it's not part of a Matlab
        # block-comment pair %{ / %}; we don't actually support that,
        # but match the simple ``%`` to end-of-line shape that Dynare
        # accepts in practice.
        if ch == "%":
            end = text.find("\n", i)
            if end == -1:
                end = n
            for j in range(i, end):
                result[j] = " "
            i = end
            continue
        i += 1
    return "".join(result)


def _strip_comments(text: str) -> str:
    """Remove block comments, line comments (// and %), and macro directives.

    Returns a string of the *same length* so that character offsets remain valid,
    with comment characters replaced by spaces.

    Comments are processed in document order so that ``//`` prevents a subsequent
    ``/*`` on the same line from starting a block comment, and vice versa.

    Skips characters inside ``"..."`` and ``'...'`` string literals so that
    e.g. ``[name='eq // one']`` doesn't have its tag text truncated at the
    ``//`` and the equation parser sees the full content.
    """
    result = list(text)
    i = 0
    n = len(text)

    while i < n:
        # Step through string literals so embedded ``//`` / ``%`` / ``@#``
        # tokens inside a quoted string don't trigger comment masking.
        if text[i] in ('"', "'"):
            quote = text[i]
            j = i + 1
            while j < n and text[j] != quote and text[j] != "\n":
                j += 1
            if j < n and text[j] == quote:
                i = j + 1
            else:
                i = j  # unterminated string — bail
            continue
        m = _COMMENT_START.search(text, i)
        if m is None:
            break

        # Skip if this position was already blanked (inside a prior comment)
        start = m.start()
        token = m.group()

        # If a string literal opened between ``i`` and the matched token,
        # rewind to the string and let the next loop iteration handle it.
        scan_segment = text[i:start]
        for q in ('"', "'"):
            qpos = scan_segment.find(q)
            if qpos != -1:
                i = i + qpos
                break
        else:
            if token == "/*":
                # Block comment: find matching */
                end = text.find("*/", start + 2)
                if end == -1:
                    end = n  # unterminated block comment extends to EOF
                else:
                    end += 2  # include the */
                for j in range(start, end):
                    if result[j] != "\n":
                        result[j] = " "
                i = end
            elif token in ("//", "%", "@#"):
                # Line comment / macro directive: extends to end of line
                end = text.find("\n", start)
                if end == -1:
                    end = n
                for j in range(start, end):
                    result[j] = " "
                i = end
            else:
                i = m.end()
            continue

    # Also blank out macro interpolation sequences @{...} so they don't
    # leave partial identifiers (e.g. ``y_@{X}`` → ``y_   ``).
    text2 = "".join(result)
    for m in re.finditer(r"@\{[^}]*\}", text2):
        for j in range(m.start(), m.end()):
            if result[j] != "\n":
                result[j] = " "

    return "".join(result)


# ---------------------------------------------------------------------------
# Block extraction (preserves offsets)
# ---------------------------------------------------------------------------

_STRING_LITERAL_STRUCTURAL = r"(?:'[^'\n]*'|\"[^\"\n]*\")"
_NON_END_TOKEN = rf"(?:{_STRING_LITERAL_STRUCTURAL}|(?!(?<!\w)end\s*;).)"


def _mask_string_literals(text: str) -> str:
    """Blank string literals while preserving line/column offsets."""
    return re.sub(
        r'"[^"\n]*"|\'[^\'\n]*\'',
        lambda m: " " * (m.end() - m.start()),
        text,
    )


def _find_block(stripped: str, keyword: str) -> Optional[re.Match]:
    """Find ``keyword(...) ; … end ;`` returning a match whose group(1) is
    any options string and group(2) is the body."""
    pattern = rf"(?<!\w){keyword}\s*(\([^)]*\))?\s*;({_NON_END_TOKEN}*?)(?<!\w)end\s*;"
    return re.search(pattern, stripped, re.DOTALL | re.IGNORECASE)


def _find_all_blocks(stripped: str, keyword: str) -> List[re.Match]:
    pattern = rf"(?<!\w){keyword}\s*(\([^)]*\))?\s*;({_NON_END_TOKEN}*?)(?<!\w)end\s*;"
    return list(re.finditer(pattern, stripped, re.DOTALL | re.IGNORECASE))


def _mask_ranges_preserving_lines(text: str, ranges: List[Tuple[int, int]]) -> str:
    result = list(text)
    for start, end in ranges:
        for idx in range(max(start, 0), min(end, len(result))):
            if result[idx] != "\n":
                result[idx] = " "
    return "".join(result)


def _mask_verbatim_blocks(text: str) -> str:
    """Blank ``verbatim; ... end;`` contents before Dynare parsing."""
    comment_masked = _strip_non_macro_comments(text)
    ranges = [
        (m.start(), m.end()) for m in _find_all_blocks(comment_masked, "verbatim")
    ]
    if not ranges:
        return text
    return _mask_ranges_preserving_lines(text, ranges)


# ---------------------------------------------------------------------------
# Declaration parsing
# ---------------------------------------------------------------------------


def _valid_declaration_identifier_boundaries(body: str, start: int, end: int) -> bool:
    """Return true when a scanned identifier is a standalone declaration token."""
    separators = " \t\r\n,"
    before = body[start - 1] if start > 0 else ""
    after = body[end] if end < len(body) else ""
    if before and before not in separators:
        return False
    # A declaration name may be immediately followed (no separating space) by a
    # ``$LaTeX$`` name or a ``(long_name=...)`` option group, e.g.
    # ``Pi_CPI_lag1_flex${\Pi}^{CPI}$(long_name='...')`` -- both are valid Dynare.
    if after and after not in separators and after not in "$(":
        return False
    return True


def _find_declaration_terminator(source: str, start: int) -> Optional[int]:
    """Return the first declaration-ending semicolon at or after *start*."""
    quote: Optional[str] = None
    paren_depth = 0
    in_latex = False
    i = start
    while i < len(source):
        ch = source[i]
        if in_latex:
            if ch == "$":
                in_latex = False
            i += 1
            continue
        if quote is not None:
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch == "$":
            in_latex = True
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == "(":
            paren_depth += 1
            i += 1
            continue
        if ch == ")" and paren_depth:
            paren_depth -= 1
            i += 1
            continue
        if ch == ";" and paren_depth == 0:
            return i
        i += 1
    return None


def _parse_declarations(
    stripped: str,
    text: str,
    keyword: str,
    precomputed_exclusions: "Optional[List[Tuple[int, int]]]" = None,
) -> List[VarDeclaration]:
    """Parse ``keyword name1 name2 … ;`` declarations.

    Only matches declarations that are *outside* of block structures
    (model, initval, shocks, etc.) to avoid misinterpreting block-internal
    uses of ``var`` (e.g. ``var eps_z;`` inside a shocks block).

    Accepts Dynare's parenthesised option syntax (``var(log) y;``,
    ``var(deflator=A) y;``) by tolerating an optional ``(...)`` group
    between the keyword and the body.

    *precomputed_exclusions* may be supplied to avoid recomputing block ranges
    when the same *stripped* text is processed for multiple keywords.
    """
    # Optional ``(...)`` options group between keyword and body so
    # ``var(log) y;`` / ``var(deflator=A) y;`` are recognised.  Inside
    # the options group we match anything except ``)`` (no nested
    # parens supported — Dynare's option syntax doesn't use them).
    pattern = (
        rf"(?<!\w){keyword}\s*"
        rf"(?:\(((?:[^()]|\([^()]*\))*)\)\s*)?"
        rf"\s+"
    )
    results: List[VarDeclaration] = []

    # Compute block ranges to exclude (or reuse a precomputed value)
    exclusions = (
        precomputed_exclusions
        if precomputed_exclusions is not None
        else _block_ranges(stripped)
    )

    scan = _mask_string_literals(stripped)
    for m in re.finditer(pattern, scan, re.DOTALL | re.IGNORECASE):
        if _inside_block(m.start(), exclusions):
            continue

        option_text = stripped[m.start(1) : m.end(1)] if m.group(1) is not None else ""
        terminator = _find_declaration_terminator(scan, m.end())
        if terminator is None:
            continue
        body = stripped[m.end() : terminator]
        body_start = m.end()
        log_transform = keyword.lower() == "var" and any(
            part.strip().lower() == "log" for part in option_text.split(",")
        )

        # Find individual identifiers inside the body, skipping LaTeX ${...}$ and (long_name=...)
        # We iterate character-by-character to skip $ blocks and parenthesized blocks
        i = 0
        while i < len(body):
            ch = body[i]
            # Skip $...$
            if ch == "$":
                j = body.find("$", i + 1)
                if j != -1:
                    i = j + 1
                else:
                    i += 1
                continue
            # Skip (...)
            if ch == "(":
                depth = 1
                j = i + 1
                while j < len(body) and depth > 0:
                    if body[j] == "(":
                        depth += 1
                    elif body[j] == ")":
                        depth -= 1
                    j += 1
                i = j
                continue
            # Try to match an identifier
            id_match = re.match(r"[A-Za-z][A-Za-z0-9_]*", body[i:])
            if id_match:
                name = id_match.group(0)
                # Skip Dynare sub-keywords that appear in declarations
                abs_start = body_start + i
                abs_end = abs_start + len(name)
                if (
                    name.lower() not in ("long_name", "latex_name", "long")
                    and text[abs_end : abs_end + 2] != "@{"
                    and _valid_declaration_identifier_boundaries(
                        body,
                        i,
                        i + len(name),
                    )
                ):
                    results.append(
                        VarDeclaration(
                            name=name,
                            range=SourceRange(
                                _offset_to_position(text, abs_start),
                                _offset_to_position(text, abs_end),
                            ),
                            log_transform=log_transform,
                            options=option_text,
                        )
                    )
                i += len(name)
            else:
                i += 1

    return results


_SIMPLE_FOR_BLOCK_RE = re.compile(
    r"^[ \t]*@#for\s+([^\r\n]+)\r?\n(?P<body>.*?)^[ \t]*@#endfor[^\r\n]*(?:\r?\n|$)",
    re.MULTILINE | re.DOTALL,
)
_FOR_BLOCK_DIRECTIVE_RE = re.compile(
    r"^\ufeff?[ \t]*@#[ \t]*(for|endfor)\b(?:[ \t]+([^\n]*))?",
    re.IGNORECASE | re.MULTILINE,
)
_DECL_TOKEN_WITH_MACRO_RE = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9_]*)?"
    r"(?:@\{[A-Za-z_][A-Za-z0-9_]*\}[A-Za-z0-9_]*)+"
)


def _after_macro_directive_line(text: str, offset: int) -> int:
    if text.startswith("\r\n", offset):
        return offset + 2
    if text.startswith("\n", offset):
        return offset + 1
    return offset


def _simple_for_blocks(
    text: str,
) -> List[Tuple[int, int, int, int, str]]:
    """Return top-level ``@#for`` blocks as offsets plus raw argument."""
    blocks: List[Tuple[int, int, int, int, str]] = []
    stack: List[Tuple[int, int, str]] = []
    for match in _FOR_BLOCK_DIRECTIVE_RE.finditer(text):
        kind = match.group(1).lower()
        if kind == "for":
            argument = (match.group(2) or "").strip()
            stack.append(
                (
                    match.start(),
                    _after_macro_directive_line(text, match.end()),
                    argument,
                )
            )
            continue
        if kind != "endfor" or not stack:
            continue
        start, body_start, argument = stack.pop()
        if stack:
            continue
        blocks.append(
            (
                start,
                _after_macro_directive_line(text, match.end()),
                body_start,
                match.start(),
                argument,
            )
        )
    return blocks


def _parse_macro_for_declarations(
    text: str,
    keyword: str,
    line_defines: Optional[Dict[int, Dict[str, str]]] = None,
    precomputed_exclusions: "Optional[List[Tuple[int, int]]]" = None,
    precomputed_non_macro_stripped: "Optional[str]" = None,
) -> List[VarDeclaration]:
    """Expand simple ``@#for`` declaration templates such as ``var y_@{C};``."""
    results: List[VarDeclaration] = []
    seen: set = set()
    exclusions = (
        precomputed_exclusions
        if precomputed_exclusions is not None
        else _block_ranges(_strip_comments(text))
    )
    decl_pattern = re.compile(
        (
            rf"(?<!\w){keyword}\s*"
            rf"(?:\(((?:[^()]|\([^()]*\))*)\)\s*)?"
            rf"(?:\s+(.*?))\s*;"
        ),
        re.DOTALL | re.IGNORECASE,
    )
    line_defines = line_defines or {}
    macro_masked_text = (
        precomputed_non_macro_stripped
        if precomputed_non_macro_stripped is not None
        else _strip_non_macro_comments(text)
    )

    def _walk(scan_text: str, scan_offset: int, defines: Dict[str, str]) -> None:
        nested_blocks = _simple_for_blocks(scan_text)
        for start, _end, body_start_rel, body_end_rel, argument in nested_blocks:
            loop_line = text.count("\n", 0, scan_offset + start)
            scoped_defines = dict(defines)
            for name, value in line_defines.get(loop_line, {}).items():
                scoped_defines.setdefault(name, value)
            loop_values = _simple_macro_for_values(argument, scoped_defines)
            if loop_values is None:
                continue
            loop_name, values = loop_values
            if not values:
                continue
            body = scan_text[body_start_rel:body_end_rel]
            body_offset = scan_offset + body_start_rel
            for value in values:
                loop_defines = dict(scoped_defines)
                loop_defines[loop_name] = value
                _collect_direct_declarations(
                    body,
                    body_offset,
                    loop_defines,
                )
                _walk(body, body_offset, loop_defines)

    def _collect_direct_declarations(
        body: str,
        body_offset: int,
        defines: Dict[str, str],
    ) -> None:
        child_ranges = [
            (start, end)
            for start, end, _body_start, _body_end, _argument in _simple_for_blocks(
                body
            )
        ]
        for decl in decl_pattern.finditer(body):
            if any(start <= decl.start() < end for start, end in child_ranges):
                continue
            _defines, active_lines, _line_defines = _macro_branch_state(
                body,
                defines,
            )
            decl_line = body.count("\n", 0, decl.start())
            if decl_line < len(active_lines) and not active_lines[decl_line]:
                continue
            decl_start = body_offset + decl.start()
            if any(start <= decl_start < end for start, end in exclusions):
                continue
            option_text = decl.group(1) or ""
            body_text = decl.group(2)
            body_start = body_offset + decl.start(2)
            log_transform = keyword.lower() == "var" and any(
                part.strip().lower() == "log" for part in option_text.split(",")
            )
            for token in _DECL_TOKEN_WITH_MACRO_RE.finditer(body_text):
                raw = token.group(0)
                if "@{" not in raw:
                    continue
                raw_start = body_start + token.start()
                raw_end = body_start + token.end()
                expanded, _source_map = _substitute_macro_vars_with_map(
                    raw,
                    defines,
                )
                if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", expanded):
                    continue
                key = (keyword.lower(), expanded, raw_start, raw_end)
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    VarDeclaration(
                        name=expanded,
                        range=SourceRange(
                            _offset_to_position(text, raw_start),
                            _offset_to_position(text, raw_end),
                        ),
                        log_transform=log_transform,
                        options=option_text,
                    )
                )

    _walk(macro_masked_text, 0, {})
    return results


def _parse_statement_macro_for_declarations(
    text: str,
    keyword: str,
    line_defines: Optional[Dict[int, Dict[str, str]]] = None,
    precomputed_exclusions: "Optional[List[Tuple[int, int]]]" = None,
    active_macro_lines: Optional[List[bool]] = None,
) -> List[VarDeclaration]:
    """Expand ``@#for`` loops nested *inside* a ``keyword … ;`` declaration.

    Handles the term-structure pattern where bare ``name@{var}`` tokens sit
    inside a single declaration that spans the loop, e.g.::

        var ln_p1
            @#for j in 2:40
            ln_p@{j}   (long_name='...')
            @#endfor
        ;

    :func:`_parse_declarations` skips ``@{`` tokens and
    :func:`_parse_macro_for_declarations` only matches a ``keyword`` inside the
    loop body, so neither expands this form.  Here the enclosing keyword is
    outside the loop, so we find each ``keyword … ;`` statement, then expand any
    ``@#for`` blocks within its body over all loop values.
    """
    line_defines = line_defines or {}
    # Preserve ``@#`` directives (only strip real comments): a declaration whose
    # body begins directly with ``@#for`` (e.g. ``var\n@#for v in [...]\n@{v}\n
    # @#endfor;``) would otherwise have the blanked loop-opener line swallowed by
    # the head pattern's trailing ``\s+``, losing the ``@#for`` so no names expand
    # (a false E010). Keeping the directive makes ``\s+`` stop at the ``@``.
    stripped = _strip_non_macro_comments(text)
    exclusions = (
        precomputed_exclusions
        if precomputed_exclusions is not None
        else _block_ranges(stripped)
    )
    head_pattern = re.compile(
        rf"(?<!\w){keyword}\s*(?:\(((?:[^()]|\([^()]*\))*)\)\s*)?\s+",
        re.IGNORECASE | re.DOTALL,
    )
    results: List[VarDeclaration] = []
    seen: set = set()

    for head in head_pattern.finditer(stripped):
        head_line = stripped.count("\n", 0, head.start())
        if (
            active_macro_lines is not None
            and head_line < len(active_macro_lines)
            and not active_macro_lines[head_line]
        ):
            continue
        if _inside_block(head.start(), exclusions):
            continue
        terminator = _find_declaration_terminator(stripped, head.end())
        if terminator is None:
            continue
        option_text = head.group(1) or ""
        log_transform = keyword.lower() == "var" and any(
            p.strip().lower() == "log" for p in option_text.split(",")
        )

        def _add_declaration(name: str, abs_start: int, abs_end: int) -> None:
            if (keyword.lower(), name) in seen:
                return
            seen.add((keyword.lower(), name))
            results.append(
                VarDeclaration(
                    name=name,
                    range=SourceRange(
                        _offset_to_position(text, abs_start),
                        _offset_to_position(text, abs_end),
                    ),
                    log_transform=log_transform,
                    options=option_text,
                )
            )

        # ``stripped`` (comment-blanked, @# directives preserved, offsets
        # 1:1) — scanning raw text would register comment words like
        # ``// total wealth`` as declarations.
        body = stripped[head.end() : terminator]
        body_offset = head.end()

        def _expand_segment(
            segment: str,
            segment_offset: int,
            defines: Dict[str, str],
        ) -> None:
            """Register declarations in *segment* under *defines*.

            Recurses into nested ``@#for`` blocks with their loop variables
            bound (mirroring ``_parse_macro_for_equations``), so statement-
            spanning nested loops expand every combination instead of only
            the first inner iteration.
            """
            _branch_defines, seg_active, _seg_line_defines = _macro_branch_state(
                segment,
                defines,
            )
            active_segment = _mask_inactive_macro_lines(segment, seg_active)
            child_blocks = list(_simple_for_blocks(active_segment))
            # Mask child loops for the direct scans — their tokens only make
            # sense once the child loop variable is bound (recursion below).
            scan_segment = _mask_ranges_preserving_newlines(
                active_segment,
                [(c_start, c_end) for c_start, c_end, _bs, _be, _arg in child_blocks],
            )
            line_offset = 0
            for line in scan_segment.splitlines(keepends=True):
                if line.lstrip().startswith("@#"):
                    line_offset += len(line)
                    continue
                i = 0
                while i < len(line):
                    ch = line[i]
                    if ch == "$":
                        j = line.find("$", i + 1)
                        i = j + 1 if j != -1 else i + 1
                        continue
                    if ch == "(":
                        depth = 1
                        j = i + 1
                        while j < len(line) and depth > 0:
                            if line[j] == "(":
                                depth += 1
                            elif line[j] == ")":
                                depth -= 1
                            j += 1
                        i = j
                        continue
                    id_match = re.match(r"[A-Za-z][A-Za-z0-9_]*", line[i:])
                    if id_match is None:
                        i += 1
                        continue
                    name = id_match.group(0)
                    token_start = line_offset + i
                    token_end = token_start + len(name)
                    abs_start = segment_offset + token_start
                    abs_end = segment_offset + token_end
                    if (
                        name.lower() not in ("long_name", "latex_name", "long")
                        and text[abs_end : abs_end + 2] != "@{"
                        and _valid_declaration_identifier_boundaries(
                            scan_segment,
                            token_start,
                            token_end,
                        )
                    ):
                        _add_declaration(name, abs_start, abs_end)
                    i += len(name)
                line_offset += len(line)

            for token in _DECL_TOKEN_WITH_MACRO_RE.finditer(scan_segment):
                raw = token.group(0)
                if "@{" not in raw:
                    continue
                expanded, _source_map = _substitute_macro_vars_with_map(raw, defines)
                if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", expanded):
                    continue
                abs_start = segment_offset + token.start()
                abs_end = segment_offset + token.end()
                _add_declaration(expanded, abs_start, abs_end)

            for _cs, _ce, c_bs, c_be, child_argument in child_blocks:
                child_values = _simple_macro_for_values(child_argument, defines)
                if child_values is None:
                    continue
                child_name, child_items = child_values
                for child_item in child_items:
                    child_defines = dict(defines)
                    child_defines[child_name] = child_item
                    # Recurse on the UNMASKED slice: parent-level branch
                    # masking binds the child loop variable to its first
                    # value only, which would wrongly blank @#if branches
                    # that depend on the child variable.  The recursion
                    # re-evaluates branch state with the child bound.
                    _expand_segment(
                        segment[c_bs:c_be],
                        segment_offset + c_bs,
                        child_defines,
                    )

        for start, _end, b_start, b_end, argument in _simple_for_blocks(body):
            loop_line = text.count("\n", 0, body_offset + start)
            scoped = dict(line_defines.get(loop_line, {}))
            loop_values = _simple_macro_for_values(argument, scoped)
            if loop_values is None:
                continue
            loop_name, values = loop_values
            loop_body = body[b_start:b_end]
            loop_body_offset = body_offset + b_start
            for value in values:
                defines = dict(scoped)
                defines[loop_name] = value
                _expand_segment(loop_body, loop_body_offset, defines)
    return results


def _mask_ranges_preserving_newlines(
    value: str,
    ranges: List[Tuple[int, int]],
) -> str:
    chars = list(value)
    for start, end in ranges:
        for i in range(max(0, start), min(len(chars), end)):
            if chars[i] not in "\r\n":
                chars[i] = " "
    return "".join(chars)


def _parse_macro_for_equations(
    body: str,
    body_offset: int,
    text: str,
    line_defines: Optional[Dict[int, Dict[str, str]]] = None,
) -> Tuple[List[Equation], List[Tuple[int, int]]]:
    """Expand simple ``@#for`` equation templates inside a model block."""
    results: List[Equation] = []
    resolved_ranges: List[Tuple[int, int]] = []
    line_defines = line_defines or {}

    def _walk(scan_text: str, scan_offset: int, defines: Dict[str, str]) -> None:
        for start, end, body_start_rel, body_end_rel, argument in _simple_for_blocks(
            scan_text,
        ):
            loop_line = text.count("\n", 0, scan_offset + start)
            scoped_defines = dict(defines)
            for name, value in line_defines.get(loop_line, {}).items():
                scoped_defines.setdefault(name, value)
            loop_values = _simple_macro_for_values(argument, scoped_defines)
            if loop_values is None:
                continue
            loop_name, values = loop_values
            if not values:
                continue

            loop_body = scan_text[body_start_rel:body_end_rel]
            loop_body_offset = scan_offset + body_start_rel

            # A loop body with no ``;`` is an in-equation fragment (a sum term
            # like ``+EXPECTATION(-@{lag})(z)``), not a sequence of complete
            # statements.  The enclosing equation parser already handles it
            # inline, so emitting separate equations here would over-count.
            if ";" not in loop_body:
                continue

            start_line = _offset_to_position(text, scan_offset + start).line
            # Use the body end (the ``@#endfor`` line), NOT ``end`` -- ``end`` is the
            # offset just past the ``@#endfor`` newline, i.e. the FOLLOWING line, so
            # the inclusive consumer filter would drop a real equation that sits on
            # the line immediately after ``@#endfor`` (e.g. a market-clearing line).
            end_line = _offset_to_position(text, scan_offset + body_end_rel).line
            resolved_ranges.append((start_line, end_line))
            child_ranges = [
                (child_start, child_end)
                for child_start, child_end, _child_body_start, _child_body_end, _arg in _simple_for_blocks(
                    loop_body
                )
            ]
            direct_body = _mask_ranges_preserving_newlines(loop_body, child_ranges)
            for value in values:
                loop_defines = dict(scoped_defines)
                loop_defines[loop_name] = value
                substituted = _substitute_macro_arg(direct_body, loop_defines)
                results.extend(
                    _parse_equations(
                        substituted,
                        loop_body_offset,
                        text,
                        filter_commands=True,
                    )
                )
                _walk(loop_body, loop_body_offset, loop_defines)

    _walk(body, body_offset, {})
    return results, resolved_ranges


def _unresolved_macro_for_template_ranges(text: str) -> List[Tuple[int, int]]:
    """Return ``@#for`` ranges whose interpolations cannot be resolved simply."""
    _defines, _active_lines, line_defines = _macro_branch_state(text)
    macro_masked_text = _strip_non_macro_comments(text)
    ranges: List[Tuple[int, int]] = []

    def _interpolation_names(value: str) -> List[str]:
        return re.findall(r"@\{([A-Za-z_][A-Za-z0-9_]*)\}", value)

    def _walk(scan_text: str, scan_offset: int, defines: Dict[str, str]) -> None:
        for start, end, body_start_rel, body_end_rel, argument in _simple_for_blocks(
            scan_text,
        ):
            loop_body = scan_text[body_start_rel:body_end_rel]
            loop_line = text.count("\n", 0, scan_offset + start)
            scoped_defines = dict(defines)
            for name, value in line_defines.get(loop_line, {}).items():
                scoped_defines.setdefault(name, value)
            loop_values = _simple_macro_for_values(argument, scoped_defines)
            start_line = _offset_to_position(text, scan_offset + start).line
            end_line = _offset_to_position(text, scan_offset + end).line
            if loop_values is None:
                if "@{" in loop_body:
                    ranges.append((start_line, end_line))
                continue
            loop_name, values = loop_values
            if not values:
                continue

            child_ranges = [
                (child_start, child_end)
                for child_start, child_end, _child_body_start, _child_body_end, _arg in _simple_for_blocks(
                    loop_body
                )
            ]
            direct_body = _mask_ranges_preserving_newlines(loop_body, child_ranges)
            for value in values:
                loop_defines = dict(scoped_defines)
                loop_defines[loop_name] = value
                names = _interpolation_names(direct_body)
                if any(name not in loop_defines for name in names):
                    ranges.append((start_line, end_line))
                    break
                _walk(loop_body, scan_offset + body_start_rel, loop_defines)

    _walk(macro_masked_text, 0, {})
    return ranges


def _merge_declarations_by_name(
    base: List[VarDeclaration],
    extra: List[VarDeclaration],
) -> List[VarDeclaration]:
    out = list(base)
    seen = {decl.name for decl in out}
    for decl in extra:
        if decl.name in seen:
            continue
        seen.add(decl.name)
        out.append(decl)
    return out


# ---------------------------------------------------------------------------
# Parameter assignment parsing
# ---------------------------------------------------------------------------


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


_PYTHON_RESERVED = frozenset(keyword.kwlist)
_PARAM_EVAL_BUILTINS = frozenset(
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


def _escape_reserved_identifier(name: str) -> str:
    if name in _PYTHON_RESERVED:
        return f"_dyn_{name}"
    return name


def _escape_known_reserved_identifiers(expr: str, known: Dict[str, float]) -> str:
    for name in known:
        escaped = _escape_reserved_identifier(name)
        if escaped != name:
            expr = re.sub(r"\b" + re.escape(name) + r"\b", escaped, expr)
    return expr


_NUMERIC_LITERAL_RE = re.compile(r"^\s*-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*$")


def _safe_eval(expr: str, known: Dict[str, float]) -> Optional[float]:
    # Fast path: pure numeric literals (the common case for Dynare parameter
    # assignments like ``0.99`` or ``1.5e-3``).  Skip the ``^``->``**`` rewrite,
    # reserved-name escaping, and AST validation below -- they dominate parse
    # time on parameter-heavy models where nearly every assignment is a number.
    if _NUMERIC_LITERAL_RE.match(expr):
        try:
            return float(expr)
        except ValueError:
            pass

    if "'" in expr or '"' in expr:
        return None

    # In Dynare ``^`` is exponentiation; in Python ``^`` is bitwise XOR
    # on ints and a type error on floats.  Rewrite ``^`` to ``**`` before
    # AST parsing so e.g. ``2^3 = 8`` (Dynare) instead of ``1`` (Python).
    # Leave string literals alone — Dynare assignments never embed them
    # since ``=`` here is a numeric assignment context.
    expr = expr.replace("^", "**")
    expr = _escape_known_reserved_identifiers(expr, known)
    expr = re.sub(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?=\()",
        lambda match: (
            match.group(1).lower()
            if match.group(1).lower() in _PARAM_EVAL_BUILTINS
            else match.group(1)
        ),
        expr,
    )
    if not _validate_ast(expr):
        logger.debug("Rejected unsafe expression: %s", expr)
        return None

    def _cbrt(x):
        x = float(x)
        return math.copysign(abs(x) ** (1.0 / 3.0), x)

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

    env = {
        "exp": math.exp,
        "log": math.log,
        "ln": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "sqrt": math.sqrt,
        "cbrt": getattr(math, "cbrt", _cbrt),
        "abs": abs,
        "sign": lambda x: (x > 0) - (x < 0),
        "pi": math.pi,
        "inf": math.inf,
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
    env.update({_escape_reserved_identifier(k): v for k, v in known.items()})
    try:
        return float(eval(expr, {"__builtins__": {}}, env))
    except (
        NameError,
        ValueError,
        TypeError,
        ZeroDivisionError,
        OverflowError,
        SyntaxError,
    ):
        return None
    except Exception:
        logger.warning(
            "Unexpected error evaluating expression: %s", expr, exc_info=True
        )
        return None


def _block_ranges(stripped: str) -> List[Tuple[int, int]]:
    """Return offset ranges of model/initval/endval/shocks/steady_state_model blocks
    and Dynare command statements like steady(...), stoch_simul(...), etc."""
    ranges = []
    # ``matched_irfs`` (and similar analysis blocks) contain ``var NAME;`` /
    # ``varexo NAME;`` statements that refer to ALREADY-declared symbols, not new
    # declarations; excluding the block keeps the declaration scanner from
    # double-counting them as endogenous/exogenous (which otherwise yields false
    # E030 duplicate-declaration warnings and inflates the E010 equation count).
    for kw in (
        "model",
        "model_replace",
        "initval",
        "endval",
        "shocks",
        "steady_state_model",
        "matched_irfs",
        "conditional_forecast_paths",
        "heteroskedastic_shocks",
        "mshocks",
        "shock_paths",
    ):
        for m in _find_all_blocks(stripped, kw):
            ranges.append((m.start(), m.end()))

    # Exclude Dynare command statements with parenthesized arguments
    # e.g. steady(solve_algo=4,...); stoch_simul(order=3,...);
    _CMD_PAT = re.compile(
        r"\b(?:steady|check|resid|stoch_simul|simul|estimation|"
        r"osr|optim_weights|calib_smoother|forecast|"
        r"identification|dynasave|dynatype)\s*\([^)]*\)\s*;",
        re.DOTALL | re.IGNORECASE,
    )
    for m in _CMD_PAT.finditer(stripped):
        ranges.append((m.start(), m.end()))

    return ranges


# Pattern for var / varexo / parameters declarations, used as an
# additional exclusion zone by the helper-assignment scanner so it
# doesn't pick up ``(long_name='...')`` metadata inside declarations as
# helper assignments.
_DECL_RANGE_PATTERN = re.compile(
    r"(?<!\w)(?:varexo_det|var|varexo|parameters|predetermined_variables)\b[^;]*;",
    re.DOTALL | re.IGNORECASE,
)


def _declaration_ranges(stripped: str) -> List[Tuple[int, int]]:
    """Offset ranges of ``var`` / ``varexo`` / ``parameters`` declarations."""
    return [(m.start(), m.end()) for m in _DECL_RANGE_PATTERN.finditer(stripped)]


def _inside_block(offset: int, ranges: List[Tuple[int, int]]) -> bool:
    return any(s <= offset < e for s, e in ranges)


def _parse_param_assignments(
    stripped: str,
    text: str,
    param_names: set,
    block_exclusions: List[Tuple[int, int]],
) -> Tuple[List[ParamAssignment], List[ParamAssignment]]:
    """Parse ``name = expr ;`` outside of block structures.

    Returns (param_assignments, helper_assignments) where helpers are
    assignments to names not declared in the ``parameters`` block.
    """
    results: List[ParamAssignment] = []
    helpers: List[ParamAssignment] = []
    known: Dict[str, float] = {}

    # Declaration parens like ``var c ${c}$ (long_name='consumption');``
    # contain ``name='value'`` syntactic sugar that the assignment regex
    # would otherwise pick up.  Skip anything inside a declaration.
    decl_ranges = _declaration_ranges(stripped)

    pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);")
    for m in pattern.finditer(stripped):
        name = m.group(1)
        if m.start(1) > 0 and stripped[m.start(1) - 1] == ".":
            continue
        if _inside_block(m.start(), block_exclusions):
            continue
        if _inside_block(m.start(), decl_ranges):
            continue

        expr = m.group(2).strip()
        val = _safe_eval(expr, known)
        if val is not None:
            known[name] = val
        else:
            known.pop(name, None)

        assignment = ParamAssignment(
            name=name,
            expression=expr,
            value=val,
            range=_range_from_match(text, m),
        )

        if name in param_names:
            results.append(assignment)
        else:
            helpers.append(assignment)

    return results, helpers


_MACRO_ASSIGN_LHS = (
    r"(?:[A-Za-z][A-Za-z0-9_]*|@\{[A-Za-z_][A-Za-z0-9_]*\})"
    r"(?:[A-Za-z0-9_]+|@\{[A-Za-z_][A-Za-z0-9_]*\})*"
)


def _macro_param_assignment_range(
    text: str,
    start: int,
    end: int,
) -> SourceRange:
    return SourceRange(
        _offset_to_position(text, start),
        _offset_to_position(text, end),
    )


def _parse_macro_for_param_assignments(
    text: str,
    param_names: set,
    line_defines: Optional[Dict[int, Dict[str, str]]] = None,
    initial_known: Optional[Dict[str, float]] = None,
    precomputed_exclusions: "Optional[List[Tuple[int, int]]]" = None,
    precomputed_non_macro_stripped: "Optional[str]" = None,
) -> Tuple[List[ParamAssignment], List[ParamAssignment]]:
    """Expand simple ``@#for`` assignment templates outside Dynare blocks."""
    results: List[ParamAssignment] = []
    helpers: List[ParamAssignment] = []
    known: Dict[str, float] = dict(initial_known or {})
    seen: set = set()
    line_defines = line_defines or {}
    scan_text = (
        precomputed_non_macro_stripped
        if precomputed_non_macro_stripped is not None
        else _strip_non_macro_comments(text)
    )
    _stripped_for_ranges = _strip_comments(text)
    block_exclusions = (
        precomputed_exclusions
        if precomputed_exclusions is not None
        else _block_ranges(_stripped_for_ranges)
    )
    decl_ranges = _declaration_ranges(_stripped_for_ranges)
    assign_lhs_pattern = re.compile(rf"^{_MACRO_ASSIGN_LHS}$")

    def _collect_direct_assignments(
        body: str,
        body_offset: int,
        defines: Dict[str, str],
    ) -> None:
        child_ranges = [
            (start, end)
            for start, end, _body_start, _body_end, _argument in _simple_for_blocks(
                body
            )
        ]
        _defines, active_lines, _line_defines = _macro_branch_state(body, defines)
        line_offset = 0
        for line in body.splitlines(keepends=True):
            if "=" not in line or ";" not in line:
                line_offset += len(line)
                continue
            eq_index = line.find("=")
            if (
                eq_index < 0
                or (eq_index > 0 and line[eq_index - 1] in "<>=!")
                or (eq_index + 1 < len(line) and line[eq_index + 1] == "=")
            ):
                line_offset += len(line)
                continue
            lhs_candidate = line[:eq_index].strip()
            # Macro-expanded parameter assignments have simple identifiers on
            # the left.  MATLAB calls/properties such as ``options_.x=...`` or
            # ``foo(bar=...)`` are runtime code and should not enter this scan.
            if (
                not lhs_candidate
                or any(ch in lhs_candidate for ch in ".(),[]")
                or assign_lhs_pattern.fullmatch(lhs_candidate) is None
            ):
                line_offset += len(line)
                continue
            semi_index = line.find(";", eq_index + 1)
            if semi_index < 0:
                line_offset += len(line)
                continue

            lhs_start = line.find(lhs_candidate)
            lhs_end = lhs_start + len(lhs_candidate)
            statement_start = line_offset + lhs_start
            statement_end = line_offset + semi_index + 1
            match_start_in_body = statement_start
            if any(start <= match_start_in_body < end for start, end in child_ranges):
                line_offset += len(line)
                continue
            match_start = body_offset + match_start_in_body
            if lhs_start > 0 and line[lhs_start - 1] == ".":
                line_offset += len(line)
                continue
            if lhs_end < len(line) and line[lhs_end] == ".":
                line_offset += len(line)
                continue
            if _inside_block(match_start, block_exclusions):
                line_offset += len(line)
                continue
            if _inside_block(match_start, decl_ranges):
                line_offset += len(line)
                continue
            assign_line = body.count("\n", 0, match_start_in_body)
            if assign_line < len(active_lines) and not active_lines[assign_line]:
                line_offset += len(line)
                continue

            raw_name = lhs_candidate
            name = _substitute_macro_arg(raw_name, defines)
            if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", name):
                line_offset += len(line)
                continue
            expression = _substitute_macro_arg(
                line[eq_index + 1 : semi_index].strip(),
                defines,
            )
            value = _safe_eval(expression, known)
            if value is not None:
                known[name] = value

            raw_start = body_offset + statement_start
            raw_end = body_offset + statement_end
            key = (name, raw_start, raw_end)
            if key in seen:
                line_offset += len(line)
                continue
            seen.add(key)
            assignment = ParamAssignment(
                name=name,
                expression=expression,
                value=value,
                range=_macro_param_assignment_range(text, raw_start, raw_end),
            )
            if name in param_names:
                results.append(assignment)
            else:
                helpers.append(assignment)
            line_offset += len(line)

    def _walk(body: str, body_offset: int, defines: Dict[str, str]) -> None:
        for start, _end, body_start_rel, body_end_rel, argument in _simple_for_blocks(
            body,
        ):
            loop_line = text.count("\n", 0, body_offset + start)
            scoped_defines = dict(defines)
            for name, value in line_defines.get(loop_line, {}).items():
                scoped_defines.setdefault(name, value)
            loop_values = _simple_macro_for_values(argument, scoped_defines)
            if loop_values is None:
                continue
            loop_name, values = loop_values
            loop_body = body[body_start_rel:body_end_rel]
            loop_body_offset = body_offset + body_start_rel
            for value in values:
                loop_defines = dict(scoped_defines)
                loop_defines[loop_name] = value
                _collect_direct_assignments(loop_body, loop_body_offset, loop_defines)
                _walk(loop_body, loop_body_offset, loop_defines)

    _walk(scan_text, 0, {})
    return results, helpers


def _merge_macro_param_assignments(
    existing: List[ParamAssignment],
    generated: List[ParamAssignment],
) -> List[ParamAssignment]:
    """Merge macro-expanded assignments, skipping the first-iteration duplicate."""
    seen = {(item.name, item.range.start.line) for item in existing}
    merged = list(existing)
    for assignment in generated:
        key = (assignment.name, assignment.range.start.line)
        if key in seen:
            continue
        seen.add(key)
        merged.append(assignment)
    merged.sort(
        key=lambda item: (
            item.range.start.line,
            item.range.start.character,
            item.range.end.line,
            item.range.end.character,
            item.name,
        )
    )
    return merged


# ---------------------------------------------------------------------------
# Equation parsing (model block & steady_state_model block)
# ---------------------------------------------------------------------------

_EQ_NAME_PATTERN = re.compile(r"""\bname\s*=\s*['"]([^'"]*)['"]""")
_EQ_MCP_PATTERN = re.compile(r"""\bmcp\s*=\s*(['"])(.*?)\1""", re.IGNORECASE)
_EQ_ENDOGENOUS_PATTERN = re.compile(
    r"""\bendogenous\s*=\s*(['"])([A-Za-z][A-Za-z0-9_]*)\1""",
    re.IGNORECASE,
)


def _iter_equation_tag_spans(text: str) -> List[Tuple[int, int]]:
    """Return quote-aware ``[ ... ]`` equation-tag spans in *text*.

    Dynare equation tags can contain quoted attribute values, and those
    values may themselves contain ``]``.  Regexes stop too early in that
    case, so use the same simple quote state model used by equation
    statement splitting.
    """
    spans: List[Tuple[int, int]] = []
    i = 0
    while i < len(text):
        if text[i] != "[":
            i += 1
            continue

        start = i
        i += 1
        quote: Optional[str] = None
        while i < len(text):
            ch = text[i]
            if quote is not None:
                if ch == quote:
                    quote = None
                i += 1
                continue
            if ch in ("'", '"'):
                quote = ch
                i += 1
                continue
            if ch == "]":
                end = i + 1
                look = end
                while look < len(text) and text[look] in " \t\r\n":
                    look += 1
                # ``[a, b] = f(...)`` is Dynare's multivariate assignment
                # (steady_state_model), not an equation tag — a tag is
                # followed by the equation text, never directly by ``=``.
                if not (look < len(text) and text[look] == "="):
                    spans.append((start, end))
                i = end
                break
            i += 1
        else:
            break
    return spans


def _strip_equation_tags_from_text(text: str) -> str:
    """Remove quote-aware Dynare equation tags while preserving other text."""
    spans = _iter_equation_tag_spans(text)
    if not spans:
        return text.strip()
    parts: List[str] = []
    last = 0
    for start, end in spans:
        parts.append(text[last:start])
        last = end
    parts.append(text[last:])
    return "".join(parts).strip()


def _inside_quoted_string(text: str, offset: int) -> bool:
    quote: Optional[str] = None
    i = 0
    while i < min(offset, len(text)):
        ch = text[i]
        if quote is not None:
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ('"', "'"):
            quote = ch
        i += 1
    return quote is not None


def _tag_mcp_constraints(tag_text: str) -> List[str]:
    return [
        match.group(2)
        for match in _EQ_MCP_PATTERN.finditer(tag_text)
        if not _inside_quoted_string(tag_text, match.start())
    ]


def _tag_endogenous_declarations(
    tag_text: str,
    tag_offset: int,
    text: str,
) -> List[Tuple[str, SourceRange]]:
    """Return Dynare ``[endogenous='name']`` on-the-fly declarations."""
    declarations: List[Tuple[str, SourceRange]] = []
    for match in _EQ_ENDOGENOUS_PATTERN.finditer(tag_text):
        if _inside_quoted_string(tag_text, match.start()):
            continue
        name_start = tag_offset + match.start(2)
        name_end = tag_offset + match.end(2)
        declarations.append(
            (
                match.group(2),
                SourceRange(
                    _offset_to_position(text, name_start),
                    _offset_to_position(text, name_end),
                ),
            )
        )
    return declarations


def _tag_names(tag_text: str) -> List[str]:
    """Return bare equation tags such as ``static`` and ``dynamic``."""
    masked = list(tag_text)
    quote: Optional[str] = None
    for i, ch in enumerate(tag_text):
        if quote is not None:
            if ch == quote:
                quote = None
            elif ch != "\n":
                masked[i] = " "
            continue
        if ch in ('"', "'"):
            quote = ch
            continue
    return [
        match.group(1).lower()
        for match in re.finditer(r"\b(static|dynamic)\b", "".join(masked), re.I)
    ]


_DYNARE_COMMANDS = frozenset(
    {
        "steady",
        "check",
        "resid",
        "stoch_simul",
        "simul",
        "estimation",
        "osr",
        "calib_smoother",
        "forecast",
        "identification",
        "dynasave",
        "dynatype",
        "model_diagnostics",
        "model_info",
        "perfect_foresight_setup",
        "perfect_foresight_solver",
    }
)


_PARAM_ASSIGNMENT_FOLLOWER_RE = re.compile(
    r"(?:"
    r"model|model_remove|model_replace|var|var_remove|varexo|varexo_det|"
    r"parameters|predetermined_variables|initval|endval|shocks|"
    r"steady_state_model|"
    + "|".join(
        re.escape(cmd) for cmd in sorted(_DYNARE_COMMANDS, key=len, reverse=True)
    )
    + r")\b",
    re.IGNORECASE,
)


def _split_equation_statements(body: str) -> List[Tuple[str, int]]:
    """Split a Dynare block body on statement semicolons.

    Semicolons inside quoted equation tags are tag text, not statement
    terminators.  Return each raw statement with its relative start
    offset in *body* so source ranges stay exact.
    """
    statements: List[Tuple[str, int]] = []
    start = 0
    quote: Optional[str] = None
    i = 0
    while i < len(body):
        ch = body[i]
        if quote is not None:
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == ";":
            statements.append((body[start:i], start))
            start = i + 1
        i += 1
    statements.append((body[start:], start))
    return statements


_ON_THE_FLY_DECL_PATTERN = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]*)\s*\|\s*([exp])\b",
    re.IGNORECASE,
)
_NOSTRICT_OPTION_RE = re.compile(
    r"^\s*(?://|%)\s*--\+\s*options:[^\r\n]*\bnostrict\b[^\r\n]*\+--",
    re.IGNORECASE | re.MULTILINE,
)
_NOSTRICT_REFERENCE_SKIP_NAMES = frozenset(
    {
        "end",
        "steady_state",
        "expectation",
        "pac_expectation",
        "diff",
        "adl",
        "nan",
        "inf",
    }
)
_NOSTRICT_FUNCTION_NAMES = _PARAM_EVAL_BUILTINS | frozenset(
    {
        "normpdf",
        "normcdf",
        "norminv",
        "logncdf",
        "erf",
        "erfc",
    }
)


def _has_nostrict_option(text: str) -> bool:
    return bool(_NOSTRICT_OPTION_RE.search(text))


def _looks_like_function_call(source: str, end_offset: int) -> bool:
    rest = source[end_offset:].lstrip()
    if not rest.startswith("("):
        return False
    return not bool(re.match(r"\(\s*[+-]?\s*\d+\s*\)", rest))


def _extract_nostrict_implicit_names(source: str) -> List[str]:
    cleaned = _strip_equation_tags_from_text(source)
    cleaned = _mask_string_literals(cleaned)
    names: List[str] = []
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", cleaned):
        name = match.group(1)
        lowered = name.lower()
        if lowered in _NOSTRICT_REFERENCE_SKIP_NAMES:
            continue
        if lowered in _NOSTRICT_FUNCTION_NAMES and _looks_like_function_call(
            cleaned,
            match.end(),
        ):
            continue
        names.append(name)
    return names


def _parse_equations(
    body: str, body_offset: int, text: str, filter_commands: bool = False
) -> List[Equation]:
    """Split a block body on ``;`` into individual equations.

    If *filter_commands* is True, skip lines that look like Dynare runtime
    commands (steady, check, stoch_simul …) which can appear after the model
    block when partial parsing is used (missing end;).
    """
    equations: List[Equation] = []
    for raw, rel_start in _split_equation_statements(body):
        eq_text = re.sub(r"\s+", " ", raw).strip()
        if not eq_text:
            continue
        # Skip purely whitespace/punctuation fragments — equations need
        # either a letter or a literal-equals-literal shape (e.g. ``0 = 1``
        # which is a contradictory equation worth flagging downstream).
        if not re.search(r"[A-Za-z]", eq_text) and not re.match(
            r"^\s*[\d.+\-*/() ]+=[\d.+\-*/() ]+\s*$", eq_text
        ):
            continue

        # Skip Dynare commands when parsing a partial (missing end;) block
        if filter_commands:
            first_word = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", eq_text)
            if first_word and first_word.group(1).lower() in _DYNARE_COMMANDS:
                continue

        # Check for equation tags.  Dynare tags can carry multiple
        # attributes (e.g. [name='eq', mcp='y > 0']); remove the whole
        # bracketed tag from equation text so lhs/rhs parsing sees only
        # the model equation.
        name = ""
        mcp_constraints: List[str] = []
        tags: List[str] = []
        for tag_start, tag_end in _iter_equation_tag_spans(eq_text):
            tag_text = eq_text[tag_start:tag_end]
            name_match = _EQ_NAME_PATTERN.search(tag_text)
            if name_match and not name:
                name = name_match.group(1)
            mcp_constraints.extend(_tag_mcp_constraints(tag_text))
            tags.extend(_tag_names(tag_text))
        tag_spans = _iter_equation_tag_spans(raw)
        on_the_fly_declarations: List[Tuple[str, str, SourceRange]] = []
        for tag_start, tag_end in tag_spans:
            tag_text = raw[tag_start:tag_end]
            tag_offset = body_offset + rel_start + tag_start
            # Use distinct loop names: ``name`` already holds the
            # [name='...'] tag value and must not be clobbered by the
            # on-the-fly [endogenous='...'] declaration scan.
            for decl_name, decl_rng in _tag_endogenous_declarations(
                tag_text,
                tag_offset,
                text,
            ):
                on_the_fly_declarations.append((decl_name, "e", decl_rng))
        for decl_match in _ON_THE_FLY_DECL_PATTERN.finditer(raw):
            if any(start <= decl_match.start() < end for start, end in tag_spans):
                continue
            abs_start = body_offset + rel_start + decl_match.start(1)
            abs_end = body_offset + rel_start + decl_match.end(1)
            on_the_fly_declarations.append(
                (
                    decl_match.group(1),
                    decl_match.group(2).lower(),
                    SourceRange(
                        _offset_to_position(text, abs_start),
                        _offset_to_position(text, abs_end),
                    ),
                )
            )
        eq_text = _strip_equation_tags_from_text(eq_text)
        eq_text = _ON_THE_FLY_DECL_PATTERN.sub(r"\1", eq_text)

        # Detect hashtag model-local variable definitions (e.g. #var = expr)
        # These are not real equations but should be parsed
        lhs, rhs = "", ""
        if "=" in eq_text:
            eq_parts = eq_text.split("=", 1)
            lhs = eq_parts[0].strip()
            rhs = eq_parts[1].strip()

        abs_start = body_offset + rel_start
        # ``raw`` comes from the comment/macro-stripped source while preserving
        # offsets.  Trim against it so blanked comments/directives before an
        # equation are not included in the original-source range.
        leading = len(raw) - len(raw.lstrip(" \t\n\r"))
        content_start = abs_start + leading
        content_end = abs_start + len(raw.rstrip(" \t\n\r"))

        if content_end <= content_start:
            content_end = content_start + 1

        equations.append(
            Equation(
                text=eq_text,
                name=name,
                range=SourceRange(
                    _offset_to_position(text, content_start),
                    _offset_to_position(text, min(content_end, len(text))),
                ),
                lhs=lhs,
                rhs=rhs,
                mcp_constraints=mcp_constraints,
                on_the_fly_declarations=on_the_fly_declarations,
                tags=tags,
            )
        )

    return equations


_MODEL_REMOVE_PATTERN = re.compile(
    r"(?<!\w)model_remove\s*\((.*?)\)\s*;",
    re.IGNORECASE | re.DOTALL,
)
_VAR_REMOVE_PATTERN = re.compile(
    r"(?<!\w)var_remove\s+([^;]+);",
    re.IGNORECASE | re.DOTALL,
)
_TAG_NAME_SELECTOR_PATTERN = re.compile(
    r"""\bname\s*=\s*(['"])(.*?)\1""",
    re.IGNORECASE | re.DOTALL,
)
_QUOTED_SELECTOR_PATTERN = re.compile(r"""(['"])(.*?)\1""", re.DOTALL)


def _tag_selection_names(selection: str) -> List[str]:
    """Return equation names selected by ``model_remove``/``model_replace``."""
    names: List[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if name and name not in seen:
            seen.add(name)
            names.append(name)

    for match in _TAG_NAME_SELECTOR_PATTERN.finditer(selection):
        add(match.group(2))

    for match in _QUOTED_SELECTOR_PATTERN.finditer(selection):
        prefix = selection[: match.start()].rstrip()
        if re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*=$", prefix):
            continue
        add(match.group(2))

    return names


def _parse_model_remove_names(stripped: str) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()
    for match in _MODEL_REMOVE_PATTERN.finditer(stripped):
        for name in _tag_selection_names(match.group(1)):
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _parse_var_removed_names(stripped: str) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()
    for match in _VAR_REMOVE_PATTERN.finditer(stripped):
        for name in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", match.group(1)):
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _parse_model_replacements(stripped: str, text: str) -> List[ModelReplacement]:
    replacements: List[ModelReplacement] = []
    for match in _find_all_blocks(stripped, "model_replace"):
        names = _tag_selection_names((match.group(1) or "").strip("() \t\r\n"))
        equations = _parse_equations(match.group(2), match.start(2), text)
        replacements.append(ModelReplacement(names=names, equations=equations))
    return replacements


def _sort_declarations(declarations: List[VarDeclaration]) -> None:
    declarations.sort(
        key=lambda v: (
            v.range.start.line,
            v.range.start.character,
            v.range.end.line,
            v.range.end.character,
        ),
    )


def _apply_on_the_fly_declarations(model: ParsedModel) -> None:
    """Promote Dynare ``name|e/x/p`` model syntax into declarations."""
    seen = model.all_declared_names()
    target_for_kind = {
        "e": model.endogenous,
        "x": model.exogenous,
        "p": model.parameters,
    }
    for equation in model.model_equations:
        for name, kind, rng in equation.on_the_fly_declarations:
            target = target_for_kind.get(kind)
            if target is None or name in seen:
                continue
            target.append(VarDeclaration(name=name, range=rng))
            seen.add(name)

    _sort_declarations(model.endogenous)
    _sort_declarations(model.exogenous)
    _sort_declarations(model.parameters)

    parameter_names = model.parameter_names()
    if parameter_names:
        promoted = [
            assignment
            for assignment in model.helper_assignments
            if assignment.name in parameter_names
        ]
        if promoted:
            model.param_assignments.extend(promoted)
            model.helper_assignments = [
                assignment
                for assignment in model.helper_assignments
                if assignment.name not in parameter_names
            ]


def _apply_nostrict_implicit_exogenous(model: ParsedModel) -> None:
    """Promote undeclared equation names to exogenous in Dynare nostrict mode."""
    if not model.nostrict:
        return

    seen = model.all_declared_names()
    local_vars: set[str] = set()
    for equation in model.model_equations:
        match = re.match(r"#\s*([A-Za-z_][A-Za-z0-9_]*)", equation.text.strip())
        if match:
            local_vars.add(match.group(1))

    for equation in model.model_equations:
        sources = [equation.text, *equation.mcp_constraints]
        for source in sources:
            for name in _extract_nostrict_implicit_names(source):
                if name in seen or name in local_vars:
                    continue
                model.exogenous.append(VarDeclaration(name=name, range=equation.range))
                seen.add(name)

    _sort_declarations(model.exogenous)


# ---------------------------------------------------------------------------
# initval / endval parsing
# ---------------------------------------------------------------------------


def _parse_initval_block(
    body: str, body_offset: int, text: str, known: Dict[str, float]
) -> List[InitvalEntry]:
    entries: List[InitvalEntry] = []
    pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);")
    for m in pattern.finditer(body):
        name = m.group(1)
        expr = m.group(2).strip()
        val = _safe_eval(expr, known)
        abs_start = body_offset + m.start()
        abs_end = body_offset + m.end()
        entries.append(
            InitvalEntry(
                name=name,
                expression=expr,
                value=val,
                range=SourceRange(
                    _offset_to_position(text, abs_start),
                    _offset_to_position(text, abs_end),
                ),
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Estimation blocks: varobs / estimated_params / observation_trends
# ---------------------------------------------------------------------------

# Dynare prior-shape keywords used to recognise the prior-shape field of an
# estimated_params entry.
_PRIOR_SHAPES = {
    "beta_pdf",
    "gamma_pdf",
    "normal_pdf",
    "inv_gamma_pdf",
    "inv_gamma1_pdf",
    "inv_gamma2_pdf",
    "uniform_pdf",
    "weibull_pdf",
}


def _parse_float_token(tok: str) -> Optional[float]:
    """Parse a numeric estimated_params field, allowing Dynare's inf/-inf."""
    tok = tok.strip()
    try:
        return float(tok)
    except ValueError:
        low = tok.lower()
        if low in ("inf", "+inf"):
            return float("inf")
        if low == "-inf":
            return float("-inf")
        return None


def _parse_estimated_param_entry(
    entry: str,
    rng: Optional[SourceRange],
) -> Optional[EstimatedParam]:
    """Parse one ``estimated_params`` entry (the text before its ``;``)."""
    fields = [f.strip() for f in entry.split(",")]
    if not fields or not fields[0]:
        return None
    first = fields[0]
    kind = "param"
    name = first
    corr_with = ""
    rest_start = 1

    std_match = re.match(r"(?i)^stderr\s+([A-Za-z_]\w*)$", first)
    corr_match = re.match(r"(?i)^corr\s+([A-Za-z_]\w*)$", first)
    if std_match:
        kind, name = "stderr", std_match.group(1)
    elif corr_match:
        kind, name = "corr", corr_match.group(1)
        if len(fields) >= 2:
            corr_with = fields[1]
        rest_start = 2

    # Collect leading numeric fields (name, init, lower, upper) up to the
    # prior-shape token; anything past the shape is prior mean/std.
    nums: List[float] = []
    shape = ""
    for fld in fields[rest_start:]:
        if fld.lower() in _PRIOR_SHAPES:
            shape = fld.lower()
            break
        val = _parse_float_token(fld)
        if val is None:
            break
        nums.append(val)

    if not re.fullmatch(r"[A-Za-z_]\w*", name):
        return None
    return EstimatedParam(
        name=name,
        kind=kind,
        corr_with=corr_with,
        init=nums[0] if len(nums) >= 1 else None,
        lower=nums[1] if len(nums) >= 3 else None,
        upper=nums[2] if len(nums) >= 3 else None,
        prior_shape=shape,
        range=rng,
    )


def _parse_estimation_blocks(stripped: str, text: str, model: "ParsedModel") -> None:
    """Populate varobs / estimated_params / observation_trends on *model*.

    Offsets in ``stripped`` align with ``text`` (comments are blanked to the
    same length), so spans computed on ``stripped`` map directly onto ``text``.
    """
    # varobs / varexobs statements (these are statements, not blocks).
    # Names are kept in declaration order *with* repeats so the duplicate
    # check (W091) can see them; counts dedup where needed.
    scan = _mask_string_literals(stripped)
    for match in re.finditer(r"(?<![\w.])varobs\b([^;]*);", scan, re.IGNORECASE):
        model.varobs_vars.extend(re.findall(r"[A-Za-z_]\w*", match.group(1)))
        if model.varobs_range is None:
            model.varobs_range = _range_from_match(text, match)
    for match in re.finditer(r"(?<![\w.])varexobs\b([^;]*);", scan, re.IGNORECASE):
        model.varexobs_vars.extend(re.findall(r"[A-Za-z_]\w*", match.group(1)))
        if model.varexobs_range is None:
            model.varexobs_range = _range_from_match(text, match)

    # estimated_params block.
    ep_match = _find_block(stripped, "estimated_params")
    if ep_match:
        model.estimated_params_range = _range_from_match(text, ep_match)
        body = ep_match.group(2)
        body_offset = ep_match.start(2)
        for entry_match in re.finditer(r"[^;]+;", body):
            raw = entry_match.group(0)[:-1]
            if not raw.strip():
                continue
            abs_start = body_offset + entry_match.start()
            abs_end = body_offset + entry_match.end()
            entry_rng = SourceRange(
                _offset_to_position(text, abs_start),
                _offset_to_position(text, abs_end),
            )
            entry = _parse_estimated_param_entry(raw, entry_rng)
            if entry is not None:
                model.estimated_params.append(entry)

    # observation_trends block: leading variable name of each line.
    ot_match = _find_block(stripped, "observation_trends")
    if ot_match:
        for line_match in re.finditer(r"([A-Za-z_]\w*)\s*[\(=]", ot_match.group(2)):
            name = line_match.group(1)
            if name not in model.observation_trends_vars:
                model.observation_trends_vars.append(name)
                abs_start = ot_match.start(2) + line_match.start(1)
                abs_end = ot_match.start(2) + line_match.end(1)
                model.observation_trends_ranges[name] = SourceRange(
                    _offset_to_position(text, abs_start),
                    _offset_to_position(text, abs_end),
                )


def _find_policy_command_options(
    stripped: str,
    command: str,
) -> Optional[Tuple[int, int, str]]:
    match = re.search(rf"(?<!\w){command}\b", stripped, re.IGNORECASE)
    if not match:
        return None
    idx = match.end()
    while idx < len(stripped) and stripped[idx].isspace():
        idx += 1
    if idx >= len(stripped) or stripped[idx] != "(":
        return match.start(), match.end(), ""

    depth = 1
    quote: Optional[str] = None
    option_start = idx + 1
    idx = option_start
    while idx < len(stripped):
        ch = stripped[idx]
        if quote is not None:
            if ch == quote:
                quote = None
            idx += 1
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return match.start(), idx + 1, stripped[option_start:idx]
        idx += 1
    return match.start(), match.end(), ""


def _option_assignment_value(options: str, option_name: str) -> Optional[str]:
    match = re.search(
        rf"(?<!\w){re.escape(option_name)}\s*=",
        options,
        re.IGNORECASE,
    )
    if not match:
        return None

    idx = match.end()
    start = idx
    depth = 0
    quote: Optional[str] = None
    while idx < len(options):
        ch = options[idx]
        if quote is not None:
            if ch == quote:
                quote = None
            idx += 1
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            if depth > 0:
                depth -= 1
        elif ch == "," and depth == 0:
            break
        idx += 1
    value = options[start:idx].strip()
    return value or None


def _parse_policy_constructs(stripped: str, text: str, model: "ParsedModel") -> None:
    """Populate optimal-policy constructs (planner_objective, ramsey/osr, etc.)."""
    # ``stripped`` preserves string-literal contents, so a word like ``osr``
    # inside a ``long_name='...'`` would otherwise register a policy command.
    # Masking is length-preserving, so ranges computed against *text* hold.
    masked = _mask_string_literals(stripped)
    objective = re.search(r"(?<!\w)planner_objective\b[^;]*;", masked, re.IGNORECASE)
    if objective:
        model.planner_objective_range = _range_from_match(text, objective)

    for command in ("ramsey_model", "ramsey_policy", "discretionary_policy", "osr"):
        command_options = _find_policy_command_options(masked, command)
        if command_options is None:
            continue
        command_start, command_end, options = command_options
        model.policy_commands.append(command)
        if model.policy_command_range is None:
            model.policy_command_range = SourceRange(
                _offset_to_position(text, command_start),
                _offset_to_position(text, command_end),
            )
        instruments = _option_assignment_value(options, "instruments")
        if instruments:
            if instruments.startswith("(") and instruments.endswith(")"):
                instruments = instruments[1:-1]
            for name in re.findall(r"[A-Za-z_]\w*", instruments):
                if name not in model.instruments:
                    model.instruments.append(name)
        discount = _option_assignment_value(options, "planner_discount")
        if discount and model.planner_discount is None:
            model.planner_discount = _safe_eval(
                discount,
                model.param_values(),
            )

    for match in re.finditer(r"(?<!\w)osr_params\b([^;]*);", masked, re.IGNORECASE):
        model.osr_params.extend(re.findall(r"[A-Za-z_]\w*", match.group(1)))

    if re.search(r"(?<!\w)optim_weights\s*;", masked, re.IGNORECASE):
        model.has_optim_weights = True

    if re.search(r"(?<!\w)occbin_(?:constraints|setup)\b", masked, re.IGNORECASE):
        model.has_occbin = True


# ---------------------------------------------------------------------------
# Unmatched block detection
# ---------------------------------------------------------------------------


def _detect_unmatched_blocks(stripped: str, text: str) -> List[Tuple]:
    """Detect block keywords (model, initval, …) without matching ``end;``."""
    errors: List[Tuple] = []
    block_keywords = ["model", "initval", "endval", "shocks", "steady_state_model"]

    for kw in block_keywords:
        # Find all keyword occurrences that look like block starts
        pat = re.compile(rf"(?<!\w){kw}\s*(\([^)]*\))?\s*;", re.IGNORECASE)
        for m in pat.finditer(stripped):
            remaining = stripped[m.end() :]
            remaining_scan = _mask_string_literals(remaining)
            is_missing = False

            end_match = re.search(r"\bend\s*;", remaining_scan, re.IGNORECASE)
            if not end_match:
                is_missing = True
            else:
                # Check if ANY block keyword (including a second copy
                # of the same one — ``model; ... model; ... end;`` is
                # not a single block, it's two openers with one closer)
                # appears before the ``end;``.
                other_block = re.search(
                    r"(?<!\w)(?:" + "|".join(block_keywords) + r")\s*(\([^)]*\))?\s*;",
                    remaining_scan[: end_match.start()],
                    re.IGNORECASE,
                )
                if other_block:
                    is_missing = True

            if is_missing:
                # Calculate where end; should be inserted
                # Find the next block keyword or end of file
                next_block = re.search(
                    r"(?<!\w)(?:" + "|".join(block_keywords) + r")\s*(\([^)]*\))?\s*;",
                    remaining_scan,
                    re.IGNORECASE,
                )
                if next_block:
                    insert_offset = m.end() + next_block.start()
                    next_block_name = re.match(r"\w+", next_block.group().strip())
                    next_block_label = (
                        next_block_name.group() if next_block_name else "next block"
                    )
                else:
                    insert_offset = len(stripped)
                    next_block_label = None
                insert_line = (
                    _offset_to_position(text, insert_offset).line + 1
                )  # 1-based

                # Find the last model equation (not a Dynare command) in the body
                # to tell the LLM where the block content actually ends
                body_before_next = (
                    remaining[: next_block.start()] if next_block else remaining
                )
                last_eq_line = None
                # Split on ; and find the last part that looks like an equation
                stmts = body_before_next.split(";")
                cumulative = 0
                for stmt in stmts:
                    stmt_stripped = stmt.strip()
                    cumulative += len(stmt) + 1  # +1 for ;
                    if not stmt_stripped or not re.search(r"[A-Za-z]", stmt_stripped):
                        continue
                    first_word_m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", stmt_stripped)
                    if (
                        first_word_m
                        and first_word_m.group(1).lower() in _DYNARE_COMMANDS
                    ):
                        continue
                    # This looks like a real equation/statement
                    semi_offset = m.end() + cumulative - 1  # position of ;
                    last_eq_line = _offset_to_position(text, semi_offset).line + 1

                # Also report which line the block starts
                start_line = _offset_to_position(text, m.start()).line + 1

                # Build a precise, actionable message
                if next_block_label and last_eq_line:
                    msg = (
                        f"Missing 'end;' for '{kw}' block (line {start_line}). "
                        f"Fix: add a new line containing only 'end;' between "
                        f"line {last_eq_line} and line {insert_line} (before '{next_block_label};')."
                    )
                elif next_block_label:
                    msg = (
                        f"Missing 'end;' for '{kw}' block (line {start_line}). "
                        f"Fix: add a new line containing only 'end;' before "
                        f"'{next_block_label};' on line {insert_line}."
                    )
                else:
                    msg = (
                        f"Missing 'end;' for '{kw}' block (line {start_line}). "
                        f"Fix: add 'end;' on its own line before line {insert_line}."
                    )

                # Compute auto-fix: insert "end;" on a new line
                # Use last_eq_line if available, otherwise insert_line - 1
                fix_line_0based = (
                    last_eq_line if last_eq_line else insert_line - 1
                ) - 1
                fix = {
                    "start_line": fix_line_0based,
                    "start_char": 999999,  # end of line
                    "end_line": fix_line_0based,
                    "end_char": 999999,
                    "new_text": "\nend;",
                }
                errors.append((msg, _range_from_match(text, m), fix))

    return errors


def _detect_missing_declaration_semicolons(
    stripped: str,
    text: str,
    precomputed_exclusions: "Optional[List[Tuple[int, int]]]" = None,
) -> List[Tuple[str, SourceRange]]:
    """Detect var/varexo/parameters declarations that are missing a semicolon.

    When a declaration like ``var x y z`` has no ``;``, the regex-based parser
    greedily matches past the intended end into the next declaration or block,
    silently merging declarations.  This check catches that case.
    """
    errors: List[Tuple] = []
    scan = _mask_string_literals(stripped)
    exclusions = (
        precomputed_exclusions
        if precomputed_exclusions is not None
        else _block_ranges(scan)
    )

    def _is_macro_directive_line(line: str) -> bool:
        return line.lstrip().startswith("@#")

    for kw in ("var", "varexo", "varexo_det", "parameters", "predetermined_variables"):
        # Find the keyword at the start of a declaration (not inside a block).
        # Tolerate Dynare's option syntax (``var(log) y``, ``var(deflator=A) y``)
        # by allowing an optional ``(...)`` group before the body whitespace.
        pat = re.compile(
            rf"(?<!\w){kw}\b(?:\s*\([^)]*\))?\s+",
            re.IGNORECASE,
        )
        for m in pat.finditer(scan):
            if _inside_block(m.start(), exclusions):
                continue
            # Check if there is a semicolon before the next block keyword or
            # another declaration keyword on the same logical line
            after = scan[m.end() :]
            # Find the next real declaration terminator, ignoring semicolons
            # inside metadata strings, metadata parens, and LaTeX spans.
            semi_abs = _find_declaration_terminator(scan, m.end())
            if semi_abs is None:
                # Find the last non-blank line in the file for fix
                text_lines = text.split("\n")
                fix_line = len(text_lines) - 1
                while fix_line > 0 and (
                    not text_lines[fix_line].strip()
                    or _is_macro_directive_line(text_lines[fix_line])
                ):
                    fix_line -= 1
                # Find end of the non-comment content on that line
                line_no_comment = re.sub(
                    r"(?://|%).*$", "", text_lines[fix_line]
                ).rstrip()
                fix = {
                    "start_line": fix_line,
                    "start_char": len(line_no_comment),
                    "end_line": fix_line,
                    "end_char": len(line_no_comment),
                    "new_text": ";",
                }
                errors.append(
                    (
                        f"Declaration '{kw}' is missing its terminating semicolon. "
                        f"Add ';' after the last variable name in this declaration.",
                        _range_from_match(text, m),
                        fix,
                    )
                )
                continue
            semi_pos = semi_abs - m.end()
            # Check if another declaration keyword, a block opener, OR an
            # assignment-style ``=`` appears before the semicolon — all
            # three indicate the semicolon belongs to a later statement
            # and the current declaration is missing its ``;``.  The
            # assignment case catches ``parameters a\na=.5;`` which the
            # earlier check missed (its first semicolon comes after the
            # ``a=.5`` assignment, so the declaration regex thinks ``a``
            # is being declared twice).
            #
            # Strip ``(long_name=...)`` metadata parens and ``$...$`` LaTeX
            # blobs first so an ``=`` inside legitimate declaration
            # metadata doesn't trip the assignment branch.  Replace with
            # same-length spaces so downstream offsets remain valid.
            between_raw = after[:semi_pos]

            def _blank_run(m: "re.Match") -> str:
                return " " * (m.end() - m.start())

            between = re.sub(r"\([^)]*\)", _blank_run, between_raw)
            between = re.sub(r"\$[^$]*\$", _blank_run, between)
            # NB: ``varexo_det`` must precede ``varexo`` in the
            # alternation so the regex engine matches the longer form
            # first; otherwise ``varexo_det`` would be split as
            # ``varexo`` + ``_det`` and the ``\b`` wouldn't bind.
            next_decl = re.search(
                r"(?P<decl>(?<!\w)(?:varexo_det|var|varexo|parameters|predetermined_variables|model|initval|endval|shocks|steady_state_model)\b)"
                r"|(?P<assign>(?<![<>=!])=(?![<>=]))",
                between,
                re.IGNORECASE,
            )
            if next_decl:
                # Find the line just before the next declaration keyword
                next_decl_abs = m.end() + next_decl.start()
                next_decl_line = text.count("\n", 0, next_decl_abs) + 1  # 1-based
                # Scan backwards from next_decl to find the last non-blank line
                text_lines = text.split("\n")
                insert_line = next_decl_line - 1
                last_var_content = ""
                while insert_line > 0:
                    line_text = text_lines[insert_line - 1].strip()
                    if _is_macro_directive_line(line_text):
                        insert_line -= 1
                        continue
                    if line_text:
                        # Strip inline comments before extracting identifier
                        line_no_comment = re.sub(r"(?://|%).*$", "", line_text).strip()
                        if not line_no_comment:
                            insert_line -= 1
                            continue
                        last_var_match = re.findall(
                            r"[A-Za-z_][A-Za-z0-9_]*", line_no_comment
                        )
                        if last_var_match:
                            last_var_content = last_var_match[-1]
                        break
                    insert_line -= 1

                next_label = (
                    f"'{next_decl.group().strip()}' declaration"
                    if next_decl.lastgroup == "decl"
                    else "assignment"
                )
                fix_msg = (
                    f"Declaration '{kw}' appears to be missing its terminating semicolon "
                    f"(the next {next_label} starts before a ';' is found). "
                )
                if last_var_content:
                    fix_msg += (
                        f"Fix: add ';' at the end of line {insert_line} "
                        f"(after '{last_var_content}')."
                    )
                else:
                    fix_msg += f"Add ';' after the last variable name in this {kw} declaration."
                # Compute fix: insert ";" at end of the line content (before comment).
                # ``insert_line`` is 1-based; convert to 0-based and clamp
                # to ``[0, len(text_lines)-1]`` so the same-line case
                # (``var y model;`` where ``insert_line`` may be 0)
                # doesn't produce ``start_line=-1`` and corrupt the
                # output via _apply_edits's clamp-to-EOF path.
                fix_line_0based = max(insert_line - 1, 0)
                if fix_line_0based >= len(text_lines):
                    fix_line_0based = len(text_lines) - 1
                fix_line_text = (
                    text_lines[fix_line_0based] if fix_line_0based >= 0 else ""
                )
                fix_line_no_comment = re.sub(r"(?://|%).*$", "", fix_line_text).rstrip()
                # If we're on the same line as the next-decl keyword,
                # insert the ``;`` right before that keyword instead of
                # at end-of-line (which would land past the rest of the
                # line's content).
                fix: Optional[dict] = None
                next_pos = _offset_to_position(text, next_decl_abs)
                if next_pos.line == fix_line_0based:
                    start_char = next_pos.character
                    if next_decl.lastgroup == "assign":
                        # The regex points at the "=".  Split before
                        # the assignment LHS when the line contains a
                        # prior declaration name, e.g.
                        # ``parameters beta beta = 0.9;`` ->
                        # ``parameters beta; beta = 0.9;``.  If the
                        # assignment is ambiguous (``parameters beta =
                        # 0.9;``), skip the auto-fix rather than
                        # corrupting the line.
                        k = start_char
                        while k > 0 and fix_line_text[k - 1] in " \t":
                            k -= 1
                        ident_end = k
                        while k > 0 and re.match(r"[A-Za-z0-9_]", fix_line_text[k - 1]):
                            k -= 1
                        candidate_lhs = fix_line_text[k:ident_end]
                        prefix = fix_line_text[:k]
                        prefix = re.sub(r"(?://|%).*$", "", prefix)
                        prefix = re.sub(r"\([^)]*\)", " ", prefix)
                        prefix = re.sub(r"\$[^$]*\$", " ", prefix)
                        prefix_ids = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", prefix)
                        if (
                            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate_lhs)
                            and len(prefix_ids) >= 2
                            and prefix_ids[0].lower() == kw
                        ):
                            start_char = k
                        else:
                            start_char = -1
                    if start_char >= 0:
                        while start_char > 0 and fix_line_text[start_char - 1] in " \t":
                            start_char -= 1
                        fix = {
                            "start_line": fix_line_0based,
                            "start_char": start_char,
                            "end_line": fix_line_0based,
                            "end_char": start_char,
                            "new_text": ";",
                        }
                else:
                    start_char = len(fix_line_no_comment)
                    fix = {
                        "start_line": fix_line_0based,
                        "start_char": start_char,
                        "end_line": fix_line_0based,
                        "end_char": start_char,
                        "new_text": ";",
                    }
                if fix is None:
                    errors.append((fix_msg, _range_from_match(text, m)))
                else:
                    errors.append((fix_msg, _range_from_match(text, m), fix))

    return errors


# ---------------------------------------------------------------------------
# Block keyword typo detection
# ---------------------------------------------------------------------------

_KEYWORD_TYPO_MAP = {
    # model typos
    "mdoel": "model",
    "modle": "model",
    "modl": "model",
    "modelo": "model",
    "mdel": "model",
    "moel": "model",
    "modeel": "model",
    "moedl": "model",
    "mmodel": "model",
    "modell": "model",
    # parameters typos
    "paramters": "parameters",
    "parametrs": "parameters",
    "paramaters": "parameters",
    "paremeters": "parameters",
    "parametres": "parameters",
    "paraemters": "parameters",
    "paramteres": "parameters",
    "parmaeters": "parameters",
    "prameters": "parameters",
    "parmeters": "parameters",
    "parametes": "parameters",
    "parametera": "parameters",
    "paramter": "parameters",
    "parametr": "parameters",
    # var typos (including LLM-style mistakes)
    "variable": "var",
    "vars": "var",
    "vasr": "var",
    # varexo typos
    "varexoo": "varexo",
    "varrexo": "varexo",
    "vaarexo": "varexo",
    "varxeo": "varexo",
    "vaxero": "varexo",
    "varexo0": "varexo",
    # shocks typos
    "shokcs": "shocks",
    "shcoks": "shocks",
    "shokc": "shocks",
    "schocks": "shocks",
    "shoks": "shocks",
    # initval typos
    "initvla": "initval",
    "inival": "initval",
    "intivals": "initval",
    "initvall": "initval",
    "initavl": "initval",
    # steady_state_model typos (just the first word)
    "staedy": "steady",
    "steday": "steady",
}


def _detect_keyword_typos(
    stripped: str,
    text: str,
    model: "ParsedModel",
    precomputed_exclusions: "Optional[List[Tuple[int, int]]]" = None,
) -> List[Tuple]:
    """Detect misspelled block keywords when the corresponding block was not found."""
    errors: List[Tuple] = []

    # Determine which blocks are missing
    has_model = bool(model.model_equations)
    has_params = bool(model.parameters)
    has_var = bool(model.endogenous)
    has_varexo = bool(model.exogenous)
    has_shocks = model.shocks_block_range is not None

    # Build set of keywords we should look for typos of
    check_keywords = set()
    if not has_model:
        check_keywords.add("model")
    if not has_var:
        check_keywords.add("var")
    if not has_params:
        check_keywords.add("parameters")
    if not has_varexo:
        check_keywords.add("varexo")
    # shocks and initval are optional — only check if they seem expected
    if not has_shocks and model.exogenous:
        check_keywords.add("shocks")

    if not check_keywords:
        return errors

    # Match words that look like block keywords: at line start, followed by
    # identifiers/options and eventually a semicolon (for model-like) or
    # just followed by identifiers (for declaration-like: var, varexo, parameters).
    #
    # IMPORTANT: skip matches that fall INSIDE an existing block.  Without
    # this guard, a model equation like ``variable = rho*variable(-1);``
    # would have its leading identifier ``variable`` flagged as a
    # misspelled ``var`` keyword and auto-fix would rewrite it to
    # ``var = rho*variable(-1);`` — actively corrupting the source.
    block_exclusions = (
        precomputed_exclusions
        if precomputed_exclusions is not None
        else _block_ranges(stripped)
    )
    for m in re.finditer(r"^\s*([A-Za-z]{3,12})\b", stripped, re.MULTILINE):
        if _inside_block(m.start(), block_exclusions):
            continue
        word = m.group(1).lower()
        correct = _KEYWORD_TYPO_MAP.get(word)
        if correct and correct in check_keywords:
            # Use m.start(1) for the keyword position (skip leading whitespace)
            line_pos = _offset_to_position(text, m.start(1))
            fix = {
                "start_line": line_pos.line,
                "start_char": line_pos.character,
                "end_line": line_pos.line,
                "end_char": line_pos.character + len(m.group(1)),
                "new_text": correct,
            }
            errors.append(
                (
                    f"Possible misspelling of '{correct}' keyword: '{m.group(1)}'. "
                    f"Fix: replace '{m.group(1)}' with '{correct}'.",
                    SourceRange(
                        _offset_to_position(text, m.start(1)),
                        _offset_to_position(text, m.end(1)),
                    ),
                    fix,
                )
            )

    return errors


# ---------------------------------------------------------------------------
# Missing parameter assignment semicolon detection
# ---------------------------------------------------------------------------

# Computation commands after which a .mod file's remaining ``name = expr``
# statements are trailing MATLAB (post-processing), not Dynare parameter
# assignments -- so missing-semicolon / merged-assignment checks must not fire
# on them.
_TERMINAL_COMMAND_RE = re.compile(
    r"(?<!\w)(?:stoch_simul|estimation|simul|perfect_foresight_solver|"
    r"ramsey_policy|discretionary_policy|osr|sensitivity|dynare_sensitivity|"
    r"send_endogenous_variables_to_workspace)\b",
    re.IGNORECASE,
)


def _trailing_code_line(stripped: str) -> Optional[int]:
    """Line of the first terminal computation command, or None.

    Statements on lines strictly greater than this are trailing MATLAB.
    """
    match = _TERMINAL_COMMAND_RE.search(stripped)
    if match is None:
        return None
    return stripped.count("\n", 0, match.start())


def _detect_missing_param_assignment_semicolons(
    stripped: str,
    text: str,
    param_names: set,
    block_exclusions: List[Tuple[int, int]],
) -> List[Tuple]:
    """Detect parameter assignments like ``name = expr`` missing trailing ``;``.

    When a parameter assignment has no semicolon, the parser merges it with
    the next line, corrupting both assignments.
    """
    errors: List[Tuple] = []
    stripped_lines = stripped.split("\n")

    # Precompute cumulative offsets to avoid O(n^2) recomputation
    line_offsets = []
    cumulative = 0
    for sline in stripped_lines:
        line_offsets.append(cumulative)
        cumulative += len(sline) + 1

    trailing_line = _trailing_code_line(stripped)

    for i, sline in enumerate(stripped_lines):
        # Statements after the first computation command are trailing MATLAB,
        # not Dynare parameter assignments.
        if trailing_line is not None and i > trailing_line:
            break
        # Check if this line is inside a block
        line_offset = line_offsets[i]
        if _inside_block(line_offset, block_exclusions):
            continue

        # Match lines that look like parameter assignments: name = value
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)", sline)
        if not m:
            continue
        name = m.group(1)
        # Accept both declared parameters AND top-level helper assignments.
        # A helper assignment that's missing its ``;`` would otherwise
        # surface only as a downstream W010/W012 about ``alpha`` or
        # ``AUX``, completely hiding the real fix (add ``;``).  Skip
        # only if the LHS looks like a keyword or block-opener (no
        # human-authored Dynare assigns to those).
        _BLOCK_LIKE = {
            "var",
            "varexo",
            "varexo_det",
            "parameters",
            "model",
            "predetermined_variables",
            "initval",
            "endval",
            "shocks",
            "steady_state_model",
            "end",
            "log",  # ``log`` appears inside ``var(log) y;`` options
        }
        if name.lower() in _BLOCK_LIKE:
            continue
        # Check if the line has a semicolon
        rhs = m.group(2).rstrip()
        if rhs.endswith(";"):
            continue
        # A .mod file may contain trailing MATLAB statements after the Dynare
        # commands (e.g. ``x = csolve('f', 0, [], 1e-8, M_, oo_, options_)``),
        # where omitting the semicolon is legal MATLAB.  A genuine Dynare
        # parameter assignment is scalar math: it never contains string
        # literals, matrix/cell brackets, or the runtime structs
        # ``M_``/``oo_``/``options_``.  Skip such MATLAB code so it is not
        # mistaken for a missing-semicolon parameter assignment.
        if (
            "'" in rhs
            or '"' in rhs
            or "[" in rhs
            or "]" in rhs
            or "{" in rhs
            or "}" in rhs
            or re.search(r"\b(?:M_|oo_|options_)\b", rhs)
        ):
            continue
        # Check the next non-empty stripped line (no fixed line cap —
        # blank lines and stripped comments between this assignment
        # and the next statement should not hide a real missing-;).
        for j in range(i + 1, len(stripped_lines)):
            next_sline = stripped_lines[j].strip()
            if not next_sline:
                continue
            # Next line is another assignment or a keyword
            if re.match(
                r"[A-Za-z_][A-Za-z0-9_]*\s*=", next_sline
            ) or _PARAM_ASSIGNMENT_FOLLOWER_RE.match(next_sline):
                # This line is missing semicolon — insert after the value,
                # before any inline comment
                line_content = sline.rstrip()
                fix_char = len(line_content)
                fix = {
                    "start_line": i,
                    "start_char": fix_char,
                    "end_line": i,
                    "end_char": fix_char,
                    "new_text": ";",
                }
                errors.append(
                    (
                        f"Parameter assignment '{name}' is missing its terminating semicolon. "
                        f"Fix: add ';' at the end of line {i + 1}.",
                        SourceRange(
                            _offset_to_position(text, line_offset + m.start()),
                            _offset_to_position(text, line_offset + m.end()),
                        ),
                        fix,
                    )
                )
            break  # Only check the first non-empty line after

    return errors


# ---------------------------------------------------------------------------
# Missing final statement semicolon before block end
# ---------------------------------------------------------------------------


def _detect_missing_final_block_semicolons(
    stripped: str,
    text: str,
) -> List[Tuple]:
    """Detect ``model``/value-block statements missing ``;`` before ``end;``."""
    errors: List[Tuple] = []

    for kw in ("model", "steady_state_model", "initval", "endval"):
        for block in _find_all_blocks(stripped, kw):
            body = block.group(2)
            body_start = block.start(2)
            body_code = body.rstrip()
            if not body_code:
                continue
            if body_code.endswith(";"):
                continue

            fix_offset = body_start + len(body_code)
            fix_pos = _offset_to_position(text, fix_offset)

            statement_start_rel = body_code.rfind(";") + 1
            while (
                statement_start_rel < len(body_code)
                and body_code[statement_start_rel].isspace()
            ):
                statement_start_rel += 1
            statement_start = body_start + statement_start_rel

            fix = {
                "start_line": fix_pos.line,
                "start_char": fix_pos.character,
                "end_line": fix_pos.line,
                "end_char": fix_pos.character,
                "new_text": ";",
            }
            errors.append(
                (
                    f"Statement in '{kw}' block is missing its terminating semicolon "
                    f"before 'end;'. Fix: add ';' at the end of line {fix_pos.line + 1}.",
                    SourceRange(
                        _offset_to_position(text, statement_start),
                        fix_pos,
                    ),
                    fix,
                )
            )

    return errors


# ---------------------------------------------------------------------------
# Missing shocks block semicolon detection
# ---------------------------------------------------------------------------


def _detect_missing_shocks_semicolons(
    stripped: str,
    text: str,
) -> List[Tuple]:
    """Detect missing semicolons inside shocks blocks."""
    errors: List[Tuple] = []
    for shocks_match in _find_all_blocks(stripped, "shocks"):
        errors.extend(
            _detect_missing_shocks_semicolons_in_block(
                shocks_match.group(2),
                shocks_match.start(2),
                text,
            )
        )
    return errors


def _detect_missing_shocks_semicolons_in_block(
    shocks_body: str,
    shocks_offset: int,
    text: str,
) -> List[Tuple]:
    """Detect missing semicolons inside one shocks block."""
    errors: List[Tuple] = []

    def _append_missing_before_keyword(
        statement_start_rel: int,
        keyword_abs: int,
        keyword: str,
    ) -> None:
        fix_start = keyword_abs
        while fix_start > 0 and text[fix_start - 1] in " \t":
            fix_start -= 1
        fix_start_pos = _offset_to_position(text, fix_start)
        fix_end_pos = _offset_to_position(text, keyword_abs)
        errors.append(
            (
                f"Missing semicolon in shocks block before '{keyword}'. "
                f"Fix: add ';' before '{keyword}' on line {fix_end_pos.line + 1}.",
                SourceRange(
                    _offset_to_position(text, shocks_offset + statement_start_rel),
                    fix_end_pos,
                ),
                {
                    "start_line": fix_start_pos.line,
                    "start_char": fix_start_pos.character,
                    "end_line": fix_end_pos.line,
                    "end_char": fix_end_pos.character,
                    "new_text": "; ",
                },
            )
        )

    # Find "var <name>" patterns not followed by ";"
    for m in re.finditer(r"\bvar\s+([A-Za-z_][A-Za-z0-9_]*)\b", shocks_body):
        # Check if a semicolon follows before the next newline or "var" keyword
        after = shocks_body[m.end() :]
        next_content = after.lstrip()
        # Find next semicolon or newline.  For ``var e = ...`` forms,
        # inspect the RHS first so ``stderr``/``corr`` on the same line
        # cannot be hidden by a later semicolon.
        semi = after.find(";")
        newline = after.find("\n")
        if (
            not next_content.startswith(("=", ","))
            and semi >= 0
            and (newline < 0 or semi < newline)
        ):
            same_statement_tail = after[:semi]
            next_statement = re.search(
                r"\b(?:var|stderr|corr)\b",
                same_statement_tail,
                re.IGNORECASE,
            )
            if next_statement is not None:
                abs_next = shocks_offset + m.end() + next_statement.start()
                _append_missing_before_keyword(
                    m.start(),
                    abs_next,
                    next_statement.group(0),
                )
                continue
            continue  # semicolon exists on same line, OK

        # Check if it's just whitespace before the next statement
        if next_content.startswith(";"):
            continue  # semicolon on next line, OK
        # ``var e = sigma^2`` and ``var e1, e2 = ...`` are valid Dynare
        # shock-declaration forms.  We're looking for ``var e`` alone,
        # not the ``=`` forms.  If the next non-whitespace token is
        # ``=`` or ``,``, the missing-; (if any) belongs at the end
        # of the RHS — find the END of the statement and check for ``;``
        # there.  Without this branch, the iter-20 skip would silently
        # accept ``shocks; var eps = sigma^2 end;`` (no terminator).
        if next_content.startswith(("=", ",")):
            # Walk to end-of-line; if no ``;`` before the newline AND
            # the next non-blank line is ``end;`` or another shock
            # statement, flag a missing ``;``.
            nl_pos = after.find("\n")
            same_line_tail = after if nl_pos < 0 else after[:nl_pos]
            next_statement = re.search(
                r"\b(?:var|stderr|corr)\b",
                same_line_tail,
                re.IGNORECASE,
            )
            if next_statement is not None:
                before_next = same_line_tail[: next_statement.start()]
                if ";" not in before_next:
                    abs_next = shocks_offset + m.end() + next_statement.start()
                    fix_pos = _offset_to_position(text, abs_next)
                    fix = {
                        "start_line": fix_pos.line,
                        "start_char": fix_pos.character,
                        "end_line": fix_pos.line,
                        "end_char": fix_pos.character,
                        "new_text": "; ",
                    }
                    errors.append(
                        (
                            f"Missing semicolon in shocks block before "
                            f"'{next_statement.group(0)}'. Fix: add ';' before "
                            f"'{next_statement.group(0)}' on line {fix_pos.line + 1}.",
                            SourceRange(
                                _offset_to_position(text, shocks_offset + m.start()),
                                _offset_to_position(text, abs_next),
                            ),
                            fix,
                        )
                    )
                    continue
            if ";" in same_line_tail:
                continue  # ``var e = sigma^2;`` — fine
            # Look at the rest after this line: if the next content
            # is ``end`` or another ``var``/``corr`` shock decl, this
            # form is missing its closing ``;``.
            rest = after if nl_pos < 0 else after[nl_pos:]
            rest_stripped = rest.lstrip()
            # Body ends right here (the shocks-block ``end;`` is
            # immediately after) — rest_stripped is empty.  That's the
            # ``shocks; var e = sigma^2 end;`` case and means the ``;``
            # is missing.
            if (
                nl_pos < 0
                or not rest_stripped
                or rest_stripped.startswith("end")
                or re.match(r"^(?:var|stderr|corr)\b", rest_stripped, re.IGNORECASE)
            ):
                abs_end_of_line = (
                    shocks_offset + m.end() + (nl_pos if nl_pos >= 0 else len(after))
                )
                fix_pos = _offset_to_position(text, abs_end_of_line)
                fix = {
                    "start_line": fix_pos.line,
                    "start_char": fix_pos.character,
                    "end_line": fix_pos.line,
                    "end_char": fix_pos.character,
                    "new_text": ";",
                }
                errors.append(
                    (
                        f"Missing semicolon at end of '{m.group(0)} ...' shock declaration. "
                        f"Fix: add ';' at the end of line {fix_pos.line + 1}.",
                        SourceRange(
                            _offset_to_position(text, shocks_offset + m.start()),
                            _offset_to_position(text, abs_end_of_line),
                        ),
                        fix,
                    )
                )
            continue

        # Missing semicolon
        abs_end = shocks_offset + m.end()
        fix_pos = _offset_to_position(text, abs_end)
        fix = {
            "start_line": fix_pos.line,
            "start_char": fix_pos.character,
            "end_line": fix_pos.line,
            "end_char": fix_pos.character,
            "new_text": ";",
        }
        errors.append(
            (
                f"Missing semicolon after 'var {m.group(1)}' in shocks block. "
                f"Fix: add ';' after '{m.group(1)}' on line {fix_pos.line + 1}.",
                SourceRange(
                    _offset_to_position(text, shocks_offset + m.start()),
                    _offset_to_position(text, abs_end),
                ),
                fix,
            )
        )

    # ``stderr`` and ``corr`` are also shocks-block statements and need
    # their own terminators.  The ``var`` scanner above cannot see these,
    # so catch the common Dynare typo:
    #
    #   shocks;
    #   var eps;
    #   stderr 0.01
    #   end;
    #
    # Be conservative around multiline continuations: if the next
    # non-empty line does not start another shocks statement, leave it
    # alone because the semicolon may appear on that continuation line.
    stmt_start_re = re.compile(r"(^|[;\n])([ \t]*)(stderr|corr)\b", re.IGNORECASE)
    for m in stmt_start_re.finditer(shocks_body):
        stmt = m.group(3).lower()
        stmt_start = m.start(3)
        after = shocks_body[m.end(3) :]
        semi = after.find(";")
        newline = after.find("\n")
        if semi >= 0 and (newline < 0 or semi < newline):
            same_statement_tail = after[:semi]
            next_statement = re.search(
                r"\b(?:var|stderr|corr)\b",
                same_statement_tail,
                re.IGNORECASE,
            )
            if next_statement is not None:
                abs_next = shocks_offset + m.end(3) + next_statement.start()
                _append_missing_before_keyword(
                    stmt_start,
                    abs_next,
                    next_statement.group(0),
                )
                continue
            continue

        if newline >= 0:
            line_end_rel = m.end(3) + newline
            rest = after[newline:]
        else:
            line_end_rel = len(shocks_body)
            rest = ""

        rest_stripped = rest.lstrip()
        if rest_stripped and not re.match(
            r"^(?:var|stderr|corr)\b",
            rest_stripped,
            re.IGNORECASE,
        ):
            continue

        line_start_rel = shocks_body.rfind("\n", 0, stmt_start) + 1
        line_prefix = shocks_body[line_start_rel:line_end_rel]
        abs_end = shocks_offset + line_start_rel + len(line_prefix.rstrip())
        fix_pos = _offset_to_position(text, abs_end)
        fix = {
            "start_line": fix_pos.line,
            "start_char": fix_pos.character,
            "end_line": fix_pos.line,
            "end_char": fix_pos.character,
            "new_text": ";",
        }
        errors.append(
            (
                f"Missing semicolon at end of '{stmt}' shock statement. "
                f"Fix: add ';' at the end of line {fix_pos.line + 1}.",
                SourceRange(
                    _offset_to_position(text, shocks_offset + stmt_start),
                    _offset_to_position(text, abs_end),
                ),
                fix,
            )
        )

    return errors


# ---------------------------------------------------------------------------
# @#include directive parsing
# ---------------------------------------------------------------------------

# Matches both the quoted and bare forms of @#include.  The filename is
# captured from one of two alternative groups so callers can pick whichever
# matched.  The directive extends to end-of-line; Dynare does not permit a
# line continuation inside the path.
_INCLUDE_PATTERN = re.compile(
    r"""
    ^\ufeff?[ \t]*
    @\#\s*include              # the @#include directive (whitespace tolerant)
    \s+
    (?:
        "([^"\n]+)"            # group 1: double-quoted filename
      | '([^'\n]+)'            # group 2: single-quoted filename
      | ([^\s"'][^\s\n]*)      # group 3: bare filename (no quotes)
    )
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)


# Matches any of the non-include macro directives Dynare's preprocessor
# accepts.  The kind is captured in group 1; everything after on the same
# line (the directive's argument, if any) is captured in group 2.  We do
# not interpret the argument here — that's a structural concern handled
# downstream (e.g. @#includepath strings registered with the workspace).
_MACRO_DIRECTIVE_PATTERN = re.compile(
    r"""
    ^\ufeff?[ \t]*
    @\#[ \t]*
    (define|includepath|elseif|endif|endfor|ifdef|ifndef|if|for|else|echo|error)
    \b
    (?:[ \t]+([^\n]*))?
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)


# Matches ``@#define NAME = value`` directives where *value* is either a
# bare identifier, a number, or a quoted string.  Arithmetic expressions,
# arrays (``["EU", "US"]``) and references to other macro variables are
# intentionally out of scope — those require a real macro evaluator,
# which would be its own subsystem.  The substitution we do here covers
# the common case (Ramsey wrappers and trivial variant labels).
_MACRO_DEFINE_PATTERN = re.compile(
    r"""
    ^\ufeff?[ \t]*
    @\#[ \t]*define[ \t]+
    ([A-Za-z_][A-Za-z0-9_]*)             # group 1: name
    (?:
        [ \t]*=[ \t]*
        (?:
            "([^"\n]*)"                  # group 2: double-quoted value
          | '([^'\n]*)'                  # group 3: single-quoted value
          | ([A-Za-z_][A-Za-z0-9_]*)     # group 4: bare identifier value
          | (-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?) # group 5: numeric value
          | (@\{[A-Za-z_][A-Za-z0-9_]*\}) # group 6: simple interpolation
          | ([^\n]*?\S)                  # group 7: raw expression/list/range
        )
    )
    ?
    [ \t]*                               # trailing horizontal whitespace
    (?=\n|$)                             # value must end the directive line
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

_SIMPLE_MACRO_FOR_PATTERN = re.compile(
    r"""
    ^\s*
    ([A-Za-z_][A-Za-z0-9_]*)             # loop variable
    \s+in\s+
    (.+?)                                 # literal list, range, or macro name
    \s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)

_SIMPLE_MACRO_FOR_LIST_PATTERN = re.compile(r"^\[(.*)\]$")
_SIMPLE_MACRO_FOR_RANGE_PATTERN = re.compile(r"^(-?\d+)\s*:\s*(-?\d+)$")
_SIMPLE_MACRO_FOR_BRACKETED_RANGE_PATTERN = re.compile(
    r"^\[\s*(-?\d+)\s*:\s*(-?\d+)\s*\]$"
)
_SIMPLE_MACRO_FOR_NAMED_RANGE_PATTERN = re.compile(
    r"^(\[?)\s*([A-Za-z_][A-Za-z0-9_]*|-?\d+)\s*:\s*"
    r"([A-Za-z_][A-Za-z0-9_]*|-?\d+)\s*(\]?)$",
)

_SIMPLE_MACRO_FOR_VALUE_PATTERN = re.compile(
    r"""
    \s*
    (?:
        "([^"\n]*)"
      | '([^'\n]*)'
      | ([A-Za-z_][A-Za-z0-9_]*|-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
    )
    \s*
    """,
    re.VERBOSE,
)


_MAX_MACRO_RANGE = (
    10000  # cap @#for range expansion to avoid OOM/hang on untrusted input
)


def _split_top_level_macro_plus(value: str) -> List[str]:
    """Split on ``+`` outside quotes/brackets (Dynare array concatenation)."""
    parts: List[str] = []
    depth = 0
    quote: Optional[str] = None
    start = 0
    for i, ch in enumerate(value):
        if quote is not None:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            if depth > 0:
                depth -= 1
        elif ch == "+" and depth == 0:
            parts.append(value[start:i].strip())
            start = i + 1
    parts.append(value[start:].strip())
    return parts


def _simple_macro_list_values(value: str) -> Optional[List[str]]:
    """Return values from a simple macro list/range literal."""
    raw = value.strip()
    # Dynare's macro language concatenates arrays with ``+``
    # (e.g. ``["US"] + ["EU"]``); resolve each operand and join.
    plus_parts = _split_top_level_macro_plus(raw)
    if len(plus_parts) > 1:
        combined: List[str] = []
        for part in plus_parts:
            # A bare (unbracketed) range operand like ``["a"] + 1:3`` is a
            # TYPE ERROR in Dynare (``:`` binds looser than ``+``) — do not
            # fold it; defer to the preprocessor.
            if _SIMPLE_MACRO_FOR_RANGE_PATTERN.match(part.strip()):
                return None
            part_values = _simple_macro_list_values(part)
            if part_values is None:
                return None
            combined.extend(part_values)
        return combined
    range_match = _SIMPLE_MACRO_FOR_RANGE_PATTERN.match(
        raw
    ) or _SIMPLE_MACRO_FOR_BRACKETED_RANGE_PATTERN.match(raw)
    if range_match is not None:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        # Guard against pathological ranges (e.g. [1:2000000]) that would
        # exhaust memory / hang the parser on untrusted input.
        if abs(end - start) > _MAX_MACRO_RANGE:
            return None
        step = 1 if start <= end else -1
        return [str(item) for item in range(start, end + step, step)]

    match = _SIMPLE_MACRO_FOR_LIST_PATTERN.match(raw)
    if match is None:
        return None
    body = match.group(1).strip()
    if not body:
        return []

    values: List[str] = []
    pos = 0
    while pos < len(body):
        value_match = _SIMPLE_MACRO_FOR_VALUE_PATTERN.match(body, pos)
        if value_match is None:
            return None
        value_item = (
            value_match.group(1) or value_match.group(2) or value_match.group(3)
        )
        if value_item is None:
            return None
        values.append(value_item)
        pos = value_match.end()
        if pos >= len(body):
            break
        if body[pos] != ",":
            return None
        pos += 1
    return values


def _resolve_simple_macro_range_bounds(
    value: str,
    defines: Dict[str, str],
) -> str:
    """Resolve bare macro names used as simple integer range endpoints."""
    match = _SIMPLE_MACRO_FOR_NAMED_RANGE_PATTERN.match(value.strip())
    if match is None:
        return value

    def endpoint(raw: str) -> Optional[str]:
        resolved = defines.get(raw, raw)
        return resolved if re.fullmatch(r"-?\d+", resolved.strip()) else None

    start = endpoint(match.group(2))
    end = endpoint(match.group(3))
    if start is None or end is None:
        return value
    body = f"{start}:{end}"
    return f"[{body}]" if match.group(1) or match.group(4) else body


def _is_simple_macro_list_literal(value: str) -> bool:
    return _simple_macro_list_values(value) is not None


def _is_supported_complex_macro_value(value: str) -> bool:
    """Return whether a raw define RHS is safe for parser-only expansion."""
    raw = value.strip()
    if not raw:
        return False
    if _is_simple_macro_list_literal(raw):
        return True
    if _SIMPLE_MACRO_FOR_RANGE_PATTERN.match(
        raw
    ) or _SIMPLE_MACRO_FOR_BRACKETED_RANGE_PATTERN.match(raw):
        return True
    return _safe_eval(raw, {}) is not None


def _macro_define_value(
    match: re.Match,
    allow_complex: bool = False,
) -> Optional[str]:
    value = (
        match.group(2)
        or match.group(3)
        or match.group(4)
        or match.group(5)
        or match.group(6)
    )
    if value is not None:
        return value
    raw = match.group(7)
    if raw is None:
        return "1"
    if not allow_complex:
        return None
    raw = raw.strip()
    return raw if _is_supported_complex_macro_value(raw) else None


def _resolve_plus_operand_defines(value: str, defines: Dict[str, str]) -> str:
    """Resolve bare macro names used as ``+`` concatenation operands.

    ``@#define B = A + ["EA"]`` (with ``A`` an array define) and
    ``@#for C in A + ["EA"]`` both need the bare ``A`` replaced by its
    array value before :func:`_simple_macro_list_values` can fold the
    concatenation.
    """
    if not defines:
        return value
    parts = _split_top_level_macro_plus(value)
    if len(parts) < 2:
        return value
    resolved_parts: List[str] = []
    for part in parts:
        seen: set = set()
        while (
            re.fullmatch(r"[A-Za-z_]\w*", part) and part in defines and part not in seen
        ):
            seen.add(part)
            part = defines[part].strip()
        resolved_parts.append(part)
    return " + ".join(resolved_parts)


def _simple_macro_for_values(
    argument: Optional[str],
    defines: Optional[Dict[str, str]] = None,
) -> Optional[Tuple[str, List[str]]]:
    """Return the loop variable and literal values for a simple ``@#for`` list."""
    if argument is None:
        return None
    match = _SIMPLE_MACRO_FOR_PATTERN.match(argument)
    if match is None:
        return None
    body = match.group(2).strip()
    if defines:
        body = _substitute_macro_arg(body, defines).strip()
        body = _resolve_plus_operand_defines(body, defines)
        body = _resolve_simple_macro_range_bounds(body, defines)

    values = _simple_macro_list_values(body)
    if values is None:
        return None
    return match.group(1), values


def _simple_macro_for_define(
    argument: Optional[str],
    defines: Optional[Dict[str, str]] = None,
) -> Optional[Tuple[str, str]]:
    """Return the loop variable/value for a singleton ``@#for`` list."""
    values = _simple_macro_for_values(argument, defines)
    if values is None:
        return None
    name, value_list = values
    if len(value_list) != 1:
        return None
    return name, value_list[0]


def _extract_macro_defines(
    text: str,
    allow_complex: bool = False,
) -> Dict[str, str]:
    """Return ``{name: value_string}`` for every ``@#define`` we can resolve.

    Only the four trivial RHS shapes are supported (see
    :data:`_MACRO_DEFINE_PATTERN`); anything else is left untouched and
    treated as out-of-scope by the LSP's macro substitution.
    """
    out: Dict[str, str] = {}
    for m in _MACRO_DEFINE_PATTERN.finditer(text):
        name = m.group(1)
        value = _macro_define_value(m, allow_complex)
        if value is None and allow_complex and out:
            # ``@#define B = A + ["EA"]`` — the raw RHS only becomes a
            # supported array literal after resolving bare define names
            # used as concatenation operands.
            raw = (m.group(7) or "").strip()
            if raw:
                rewritten = _resolve_plus_operand_defines(raw, out)
                if rewritten != raw and _is_supported_complex_macro_value(rewritten):
                    value = rewritten
        if value is None:
            continue
        if out:
            value = _substitute_macro_arg(value, out)
            value = _resolve_plus_operand_defines(value, out)
        out[name] = value
    return out


def _macro_truth_value(
    argument: Optional[str], defines: Dict[str, str]
) -> Optional[bool]:
    if argument is None:
        return None
    raw = argument.strip()
    unknown = object()

    def _scalar(expr: str, seen: Optional[set] = None) -> object:
        seen = seen or set()
        value = expr.strip()
        list_values = _simple_macro_list_values(value)
        if list_values is not None:
            out: List[object] = []
            for item in list_values:
                resolved = _scalar(item, seen)
                out.append(item if resolved is unknown else resolved)
            return out
        if value in defines and value not in seen:
            resolved = _scalar(defines[value], seen | {value})
            return defines[value].strip() if resolved is unknown else resolved
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            return value[1:-1].strip()
        lowered = value.lower()
        if lowered in {"", "0", "false", "no"}:
            return False
        if lowered in {"1", "true", "yes"}:
            return True
        try:
            return float(value)
        except ValueError:
            evaluated = _safe_eval(value, {})
            return unknown if evaluated is None else evaluated

    def _truth(value: object) -> Optional[bool]:
        if value is unknown:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() not in {"", "0", "false", "no"}
        return None

    direct = _scalar(raw)
    if direct is not unknown:
        return _truth(direct)

    expression = raw.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"(?<![=!<>])!(?!=)", " not ", expression).strip()
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None

    def _eval(node: ast.AST) -> object:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (bool, int, float, str)):
                return node.value
            return unknown
        if isinstance(node, ast.Name):
            return _scalar(node.id)
        if isinstance(node, ast.List):
            values = [_eval(elt) for elt in node.elts]
            return unknown if any(value is unknown for value in values) else values
        if isinstance(node, ast.Tuple):
            values = [_eval(elt) for elt in node.elts]
            return (
                unknown if any(value is unknown for value in values) else tuple(values)
            )
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if operand is unknown:
                return unknown
            if isinstance(node.op, ast.Not):
                truth = _truth(operand)
                return unknown if truth is None else not truth
            if isinstance(operand, (int, float)):
                if isinstance(node.op, ast.USub):
                    return -operand
                if isinstance(node.op, ast.UAdd):
                    return operand
            return unknown
        if isinstance(node, ast.BoolOp):
            values = [_truth(_eval(v)) for v in node.values]
            if any(v is None for v in values):
                return unknown
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
            return unknown
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if not isinstance(left, (int, float)) or not isinstance(
                right, (int, float)
            ):
                return unknown
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return unknown if right == 0 else left / right
            if isinstance(node.op, ast.Mod):
                return unknown if right == 0 else left % right
            return unknown

        def _loose_eq(a: object, b: object) -> bool:
            # ``_scalar`` eagerly floats numeric-looking define values, so a
            # loop variable bound to "2" arrives as 2.0 while the literal
            # stays the string "2" — compare across that coercion boundary.
            if isinstance(a, str) != isinstance(b, str):
                text_side, other = (a, b) if isinstance(a, str) else (b, a)
                try:
                    return float(text_side) == float(other)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    return False
            return a == b

        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            if left is unknown:
                return unknown
            for op, comparator in zip(node.ops, node.comparators):
                right = _eval(comparator)
                if right is unknown:
                    return unknown
                if isinstance(op, ast.In):
                    try:
                        ok = left in right  # type: ignore[operator]
                    except TypeError:
                        return unknown
                elif isinstance(op, ast.NotIn):
                    try:
                        ok = left not in right  # type: ignore[operator]
                    except TypeError:
                        return unknown
                elif isinstance(op, ast.Eq):
                    ok = _loose_eq(left, right)
                elif isinstance(op, ast.NotEq):
                    ok = not _loose_eq(left, right)
                elif isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                    try:
                        if isinstance(op, ast.Lt):
                            ok = left < right  # type: ignore[operator]
                        elif isinstance(op, ast.LtE):
                            ok = left <= right  # type: ignore[operator]
                        elif isinstance(op, ast.Gt):
                            ok = left > right  # type: ignore[operator]
                        else:
                            ok = left >= right  # type: ignore[operator]
                    except TypeError:
                        return unknown
                else:
                    return unknown
                if not ok:
                    return False
                left = right
            return True
        return unknown

    return _truth(_eval(tree))


def _macro_branch_state(
    text: str,
    initial_defines: Optional[Dict[str, str]] = None,
    line_macro_defines: Optional[Dict[int, Dict[str, str]]] = None,
) -> Tuple[Dict[str, str], List[bool], Dict[int, Dict[str, str]]]:
    """Return active simple defines plus per-line macro activity.

    This is deliberately small: it handles constant ``@#if``/
    ``@#ifdef``/``@#ifndef`` branches so known-inactive text does not
    leak into the LSP parser. Unknown conditions stay active because the
    parser cannot safely choose one branch without a full macro evaluator.
    """
    defines: Dict[str, str] = dict(initial_defines or {})
    directives = _parse_macro_directives(_strip_non_macro_comments(text))
    lines = text.splitlines(keepends=True)
    active_lines = [True] * len(lines)
    line_defines: Dict[int, Dict[str, str]] = {}
    by_line: Dict[int, List[MacroDirective]] = {}
    for directive in directives:
        by_line.setdefault(directive.range.start.line, []).append(directive)
    stack: List[dict] = []

    def _current_active() -> bool:
        return all(frame["active"] for frame in stack)

    def _current_certain_active() -> bool:
        return _current_active() and not any(frame["unknown"] for frame in stack)

    def _nearest_unknown_active_frame() -> Optional[dict]:
        for frame in reversed(stack):
            if frame.get("unknown") and frame.get("active"):
                return frame
        return None

    def _restore_branch_defines(frame: dict) -> None:
        restore = frame.get("define_restore", {})
        for name in list(frame.get("branch_defined", set())):
            had_previous, previous_value = restore.get(name, (False, None))
            if had_previous:
                defines[name] = "" if previous_value is None else previous_value
            else:
                defines.pop(name, None)
        frame["branch_defined"] = set()

    def _apply_define(argument: str) -> None:
        if not _current_active():
            return
        new_defines = _extract_macro_defines(
            f"@#define {argument}\n",
            allow_complex=True,
        )
        if not new_defines and defines:
            # ``@#define B = A + ["EA"]`` only becomes a supported array
            # value after resolving concat operands against the defines
            # accumulated so far (this helper extracts one directive at a
            # time, so _extract_macro_defines's own chaining can't see A).
            name_match = re.match(r"\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", argument)
            if name_match:
                rewritten = _resolve_plus_operand_defines(
                    name_match.group(2),
                    defines,
                )
                if rewritten != name_match.group(2):
                    new_defines = _extract_macro_defines(
                        f"@#define {name_match.group(1)} = {rewritten}\n",
                        allow_complex=True,
                    )
        if not new_defines:
            return
        for name, value in new_defines.items():
            value = _substitute_macro_arg(value, defines)
            frame = _nearest_unknown_active_frame()
            if frame is not None:
                restore = frame.setdefault("define_restore", {})
                if name not in restore:
                    restore[name] = (name in defines, defines.get(name))
                frame.setdefault("branch_defined", set()).add(name)
            defines[name] = value

    def _apply_directive(directive: MacroDirective) -> None:
        kind = directive.kind
        parent_active = _current_active()
        if kind == "define":
            if directive.argument:
                _apply_define(directive.argument)
            return
        if kind in {"if", "ifdef", "ifndef"}:
            if kind == "if":
                truth = _macro_truth_value(directive.argument, defines)
            else:
                name = (directive.argument or "").strip()
                truth = name in defines
                if kind == "ifndef":
                    truth = not truth
            active = parent_active if truth is None else parent_active and truth
            stack.append(
                {
                    "kind": "if",
                    "parent_active": parent_active,
                    "active": active,
                    "taken": bool(truth) if truth is not None else False,
                    "unknown": truth is None,
                }
            )
            return
        if kind == "elseif" and stack and stack[-1].get("kind") == "if":
            frame = stack[-1]
            if frame["unknown"]:
                _restore_branch_defines(frame)
                frame["active"] = frame["parent_active"]
            elif frame["taken"]:
                frame["active"] = False
            else:
                truth = _macro_truth_value(directive.argument, defines)
                frame["active"] = (
                    frame["parent_active"]
                    if truth is None
                    else frame["parent_active"] and truth
                )
                frame["taken"] = bool(truth) if truth is not None else False
                frame["unknown"] = truth is None
            return
        if kind == "else" and stack and stack[-1].get("kind") == "if":
            frame = stack[-1]
            if frame["unknown"]:
                _restore_branch_defines(frame)
                frame["active"] = frame["parent_active"]
            else:
                frame["active"] = frame["parent_active"] and not frame["taken"]
                frame["taken"] = True
            return
        if kind == "endif" and stack and stack[-1].get("kind") == "if":
            frame = stack.pop()
            _restore_branch_defines(frame)
            return
        if kind == "for":
            loop_values = _simple_macro_for_values(directive.argument, defines)
            if loop_values is not None:
                name, values = loop_values
                had_previous = name in defines
                previous_value = defines.get(name)
                active = parent_active and bool(values)
                if active:
                    value = (
                        defines[name]
                        if name in defines and defines[name] in values
                        else values[0]
                    )
                    defines[name] = value
                stack.append(
                    {
                        "kind": "for",
                        "parent_active": parent_active,
                        "active": active,
                        "taken": True,
                        "unknown": False,
                        "define_name": name if active else None,
                        "had_previous": had_previous,
                        "previous_value": previous_value,
                    }
                )
                return
            stack.append(
                {
                    "kind": "for",
                    "parent_active": parent_active,
                    "active": parent_active,
                    "taken": True,
                    "unknown": True,
                }
            )
            return
        if kind == "endfor" and stack and stack[-1].get("kind") == "for":
            frame = stack.pop()
            _restore_branch_defines(frame)
            name = frame.get("define_name")
            if name is not None:
                if frame.get("had_previous"):
                    defines[name] = frame.get("previous_value", "")
                else:
                    defines.pop(name, None)

    external_values: Dict[str, str] = {}
    for line_no in range(len(lines)):
        line_external_values = (line_macro_defines or {}).get(line_no, {})
        for name in set(external_values) - set(line_external_values):
            if defines.get(name) == external_values[name]:
                defines.pop(name, None)
            external_values.pop(name, None)
        for name, value in line_external_values.items():
            if name not in defines or external_values.get(name) != value:
                defines[name] = value
            external_values[name] = value
        active_lines[line_no] = _current_active()
        line_defines[line_no] = dict(defines)
        for directive in sorted(
            by_line.get(line_no, []),
            key=lambda d: d.range.start.character,
        ):
            _apply_directive(directive)
    return defines, active_lines, line_defines


def _extract_active_macro_defines(text: str) -> Dict[str, str]:
    """Return simple ``@#define`` values reached in active macro branches."""
    defines, _active_lines, _line_defines = _macro_branch_state(text)
    return defines


def _mask_inactive_macro_lines(text: str, active_lines: List[bool]) -> str:
    """Blank known-inactive macro lines while preserving line/char offsets."""
    if all(active_lines):
        return text

    def _blank(line: str) -> str:
        newline = ""
        body = line
        if line.endswith("\r\n"):
            body = line[:-2]
            newline = "\r\n"
        elif line.endswith("\n") or line.endswith("\r"):
            body = line[:-1]
            newline = line[-1]
        return (" " * len(body)) + newline

    out: List[str] = []
    for idx, line in enumerate(text.splitlines(keepends=True)):
        if idx < len(active_lines) and not active_lines[idx]:
            out.append(_blank(line))
        else:
            out.append(line)
    return "".join(out)


# ``@{NAME}`` interpolation — Dynare's macro language uses braces around
# the identifier to splice the value back into the surrounding text.
_MACRO_INTERP_PATTERN = re.compile(r"@\{([A-Za-z_][A-Za-z0-9_]*)\}")
# Dynare also allows an integer expression inside ``@{...}`` (e.g. the
# term-structure idiom ``p@{j-1}_fwrd``), which the bare-identifier pattern
# above cannot capture.  This broader pattern feeds the arithmetic resolver.
_MACRO_INTERP_EXPR_PATTERN = re.compile(r"@\{([^}\n]+)\}")


def _eval_macro_arith(expr: str, defines: Dict[str, str]) -> Optional[int]:
    """Evaluate a simple integer macro expression (e.g. ``j-1``) given defines.

    Define identifiers are substituted with their values, then the expression
    is parsed and evaluated allowing only integer arithmetic
    (``+ - * / // % **`` and parentheses).  Returns the integer result, or
    None if the expression is not a pure integer arithmetic expression.
    """

    def _sub(match: "re.Match") -> str:
        name = match.group(0)
        return f"({defines[name]})" if name in defines else name

    substituted = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", _sub, expr)
    try:
        tree = ast.parse(substituted, mode="eval")
    except SyntaxError:
        return None

    def _ev(node):
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = _ev(node.operand)
            if operand is None:
                return None
            return +operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp):
            left, right = _ev(node.left), _ev(node.right)
            if left is None or right is None:
                return None
            op = node.op
            try:
                if isinstance(op, ast.Add):
                    return left + right
                if isinstance(op, ast.Sub):
                    return left - right
                if isinstance(op, ast.Mult):
                    return left * right
                if isinstance(op, ast.Div):
                    return left / right if right else None
                if isinstance(op, ast.FloorDiv):
                    return left // right if right else None
                if isinstance(op, ast.Mod):
                    return left % right if right else None
                if isinstance(op, ast.Pow):
                    return left**right
            except (ZeroDivisionError, ValueError, OverflowError):
                return None
        return None

    try:
        value = _ev(tree)
    except Exception:
        return None
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    return value if isinstance(value, int) else None


def _resolve_macro_interp(expr: str, defines: Dict[str, str]) -> Optional[str]:
    """Resolve a single ``@{...}`` body to its replacement text, or None.

    A bare define name resolves to its value; otherwise the body is evaluated
    as an integer arithmetic expression (so ``@{j-1}`` works when ``j`` is a
    loop variable).  Returns None when neither applies (leave the text as-is).
    """
    name = expr.strip()
    if name in defines:
        return defines[name]
    value = _eval_macro_arith(name, defines)
    return str(value) if value is not None else None


def _substitute_macro_vars(text: str, defines: Dict[str, str]) -> str:
    """Replace ``@{NAME}`` with the value declared in ``@#define NAME = ...``.

    Substitutes the value directly without offset padding.  Padding with
    trailing spaces (the prior strategy) broke mid-identifier expansion:
    ``y_@{C}_t`` with ``C=US`` would become ``y_US  _t`` and the parser
    would see two tokens ``y_US`` and ``_t`` rather than the intended
    ``y_US_t``.

    Downstream offsets shift — declarations and references discovered
    in the substituted text have line/column positions that differ
    slightly from the original source.  That trade-off is documented in
    ``parser.parse()`` as the cost of the common-case macro expander
    landing without a full re-evaluator.
    """
    if not defines:
        return text

    def _replace(m: re.Match) -> str:
        resolved = _resolve_macro_interp(m.group(1), defines)
        return resolved if resolved is not None else m.group(0)

    return _MACRO_INTERP_EXPR_PATTERN.sub(_replace, text)


def _substitute_macro_arg(arg: str, defines: Dict[str, str]) -> str:
    """Resolve simple macro arguments used as include/include-path targets."""
    resolved = _substitute_macro_vars(arg, defines).strip()
    if resolved in defines:
        resolved = defines[resolved]
    return resolved


def _substitute_macro_vars_with_map(
    text: str,
    defines: Dict[str, str],
) -> Tuple[str, List[int]]:
    """Substitute macro variables and map substituted offsets to source offsets."""
    if not defines:
        return text, list(range(len(text) + 1))

    out: List[str] = []
    source_map: List[int] = [0]
    cursor = 0

    def _append_original(start: int, end: int) -> None:
        for idx in range(start, end):
            out.append(text[idx])
            source_map.append(idx + 1)

    def _append_replacement(value: str, start: int, end: int) -> None:
        if not value:
            source_map[-1] = end
            return
        span = max(end - start, 1)
        n_value = len(value)
        for i, ch in enumerate(value, start=1):
            out.append(ch)
            if i == n_value:
                source_map.append(end)
            else:
                source_map.append(start + round(i * span / n_value))

    for m in _MACRO_INTERP_EXPR_PATTERN.finditer(text):
        resolved = _resolve_macro_interp(m.group(1), defines)
        if resolved is None:
            continue
        _append_original(cursor, m.start())
        _append_replacement(resolved, m.start(), m.end())
        cursor = m.end()

    _append_original(cursor, len(text))
    return "".join(out), source_map


def _substitute_macro_vars_with_line_maps(
    text: str,
    line_defines: Dict[int, Dict[str, str]],
) -> Tuple[str, List[int]]:
    """Substitute macro variables using only definitions visible per line."""
    if not line_defines:
        return text, list(range(len(text) + 1))

    out: List[str] = []
    source_map: List[int] = [0]
    source_offset = 0
    for line_no, line in enumerate(text.splitlines(keepends=True)):
        substituted, local_map = _substitute_macro_vars_with_map(
            line,
            line_defines.get(line_no, {}),
        )
        out.append(substituted)
        for mapped in local_map[1:]:
            source_map.append(source_offset + mapped)
        source_offset += len(line)

    if not out:
        return text, list(range(len(text) + 1))
    return "".join(out), source_map


def _parse_macro_directives(text: str) -> List[MacroDirective]:
    """Find every non-``@#include`` macro directive in *text*.

    Runs against the raw source before comment-stripping blanks out
    macro lines.  Mirrors :func:`_parse_includes`: captures kind +
    argument + source range, performs no interpretation.
    """
    results: List[MacroDirective] = []
    for m in _MACRO_DIRECTIVE_PATTERN.finditer(text):
        kind = m.group(1).lower()
        argument = m.group(2)
        if argument is not None:
            argument = argument.strip()
            if argument == "":
                argument = None
        results.append(
            MacroDirective(
                kind=kind,
                argument=argument,
                range=_range_from_match(text, m),
            )
        )
    return results


# ``@#include`` with a string EXPRESSION argument (``@#include F`` or
# ``@#include P + "/x.mod"``) — valid Dynare; the quoted pattern misses it.
_EXPRESSION_INCLUDE_LINE_RE = re.compile(
    r"^[ \t]*@#[ \t]*include[ \t]+(?![\"'])(\S[^\n]*?)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


def _fold_include_expression(
    argument: str,
    defines: Dict[str, str],
) -> Optional[str]:
    """Fold a string-concat include argument to a literal path, or None."""
    parts = _split_top_level_macro_plus(argument.strip())
    if not parts:
        return None
    folded: List[str] = []
    for part in parts:
        part = part.strip()
        if len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'":
            folded.append(part[1:-1])
            continue
        if re.fullmatch(r"[A-Za-z_]\w*", part):
            value = defines.get(part)
            if value is None:
                return None
            folded.append(value)
            continue
        return None
    result = "".join(folded).strip()
    return result or None


def _has_unfoldable_expression_include(text: str) -> bool:
    """Whether an expression-form ``@#include`` resists constant folding."""
    defines: Optional[Dict[str, str]] = None
    for m in _EXPRESSION_INCLUDE_LINE_RE.finditer(text):
        if defines is None:
            defines = _extract_macro_defines(text, allow_complex=True)
        if _fold_include_expression(m.group(1), defines) is None:
            return True
    return False


def _parse_includes(text: str) -> List[IncludeDirective]:
    """Find every ``@#include`` directive in *text*.

    Runs against the raw source so the macro line is still present.  Each
    directive is recorded with the filename string exactly as written
    (no path resolution) and a :class:`SourceRange` spanning the entire
    directive.  Expression-form arguments built from string ``@#define``s
    and quoted literals are constant-folded to their literal path so they
    resolve like ordinary includes.
    """
    results: List[IncludeDirective] = []
    for m in _INCLUDE_PATTERN.finditer(text):
        filename = m.group(1) or m.group(2) or m.group(3)
        if filename is None:
            continue
        filename = filename.strip()
        if not filename:
            continue
        results.append(
            IncludeDirective(
                filename=filename,
                range=_range_from_match(text, m),
            )
        )
    defines: Optional[Dict[str, str]] = None
    for m in _EXPRESSION_INCLUDE_LINE_RE.finditer(text):
        if defines is None:
            defines = _extract_macro_defines(text, allow_complex=True)
        folded = _fold_include_expression(m.group(1), defines)
        if folded is None:
            continue
        results.append(
            IncludeDirective(
                filename=folded,
                range=_range_from_match(text, m),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------


def parse(
    text: str,
    initial_macro_defines: Optional[Dict[str, str]] = None,
    line_macro_defines: Optional[Dict[int, Dict[str, str]]] = None,
) -> ParsedModel:
    """Parse a Dynare .mod file and return a structured AST with positions."""
    model = ParsedModel(
        text=text, original_text=text, nostrict=_has_nostrict_option(text)
    )
    text = _mask_verbatim_blocks(text)
    model.text = text

    # Collect @#include and other macro directives.  Run against a copy
    # of the text with /* */, //, and % comments masked out so a
    # commented-out directive (a common idiom for leaving old includes
    # as documentation) is NOT treated as a real directive.  ``@#``
    # lines themselves stay intact in the masked text — only the
    # surrounding ``//`` / ``%`` / ``/* */`` are blanked.  No path
    # resolution here — that's workspace work.
    pre_masked = _strip_non_macro_comments(text)
    model.includes = _parse_includes(pre_masked)
    model.has_unfoldable_expression_include = _has_unfoldable_expression_include(
        pre_masked
    )
    model.macro_directives = _parse_macro_directives(pre_masked)
    original_includes = list(model.includes)
    original_macro_directives = list(model.macro_directives)
    macro_declaration_text = text

    # Common-case macro variable substitution.  Apply BEFORE comment
    # stripping, since _strip_comments blanks out ``@{...}`` (treating
    # it as a comment).  Only handles bare-value @#define forms — full
    # macro evaluation (arithmetic, arrays, @#for unrolling, @#if
    # conditional expansion) is intentionally out of scope.
    #
    # IMPORTANT: extract defines from the comment-masked text, not the
    # raw source, so a commented-out ``// @#define C = US`` does NOT
    # leak into substitution.  Same correctness rule that the include /
    # macro-directive scanners follow above.
    defines, active_macro_lines, line_defines = _macro_branch_state(
        pre_masked,
        initial_macro_defines,
        line_macro_defines,
    )
    if any(line_defines.values()):
        model.includes = [
            IncludeDirective(
                filename=_substitute_macro_arg(
                    directive.filename,
                    line_defines.get(directive.range.start.line, {}),
                ),
                range=directive.range,
            )
            for directive in original_includes
        ]
        model.macro_directives = [
            MacroDirective(
                kind=directive.kind,
                argument=(
                    directive.argument
                    if directive.kind in {"ifdef", "ifndef"}
                    else _substitute_macro_arg(
                        directive.argument,
                        line_defines.get(directive.range.start.line, {}),
                    )
                    if directive.argument is not None
                    else None
                ),
                range=directive.range,
            )
            for directive in original_macro_directives
        ]
        text, source_map = _substitute_macro_vars_with_line_maps(text, line_defines)
        # Keep model.text consistent with the text the rest of the
        # parser sees.  Downstream tooling (autofix lookup, hover
        # source-slice extraction, code-action computation) indexes
        # into model.text using ranges produced against the substituted
        # text — without this update, autofix can't find the identifier
        # it's being asked to rewrite when the source contained an
        # ``@{VAR}`` interpolation.
        model.text = text
        model.source_map = source_map

    macro_active_text = _mask_inactive_macro_lines(
        macro_declaration_text,
        active_macro_lines,
    )
    text = _mask_inactive_macro_lines(text, active_macro_lines)
    model.text = text
    # Cache the per-line activity vector so diagnostic checks can reuse it
    # without re-running _macro_branch_state on every keystroke.
    model._cached_active_macro_lines = active_macro_lines

    stripped = _strip_comments(text)

    # --- Block ranges and comment-masked variants (compute once each) ---
    # These 5 keyword regex scans + comment-stripping of the full text are expensive on
    # large models.  Compute once here and pass as precomputed_* to every function that
    # would otherwise recompute them independently.
    exclusions = _block_ranges(stripped)
    # macro_active_text may differ from text (inactive macro lines are blanked), but
    # in practice produces the same block structure; compute once.
    macro_stripped = _strip_comments(macro_active_text)
    macro_exclusions = (
        exclusions if macro_stripped == stripped else _block_ranges(macro_stripped)
    )
    # _strip_non_macro_comments on macro_active_text is called in multiple places;
    # compute once and reuse to avoid repeated O(N) passes over the full text.
    macro_pre_masked = _strip_non_macro_comments(macro_active_text)

    # --- Declarations ---
    model.endogenous = _parse_declarations(stripped, text, "var", exclusions)
    model.endogenous = _merge_declarations_by_name(
        model.endogenous,
        _parse_macro_for_declarations(
            macro_active_text, "var", line_defines, macro_exclusions, macro_pre_masked
        ),
    )
    model.endogenous = _merge_declarations_by_name(
        model.endogenous,
        _parse_statement_macro_for_declarations(
            macro_declaration_text,
            "var",
            line_defines,
            macro_exclusions,
            active_macro_lines,
        ),
    )
    model.exogenous = _parse_declarations(stripped, text, "varexo", exclusions)
    model.exogenous = _merge_declarations_by_name(
        model.exogenous,
        _parse_macro_for_declarations(
            macro_active_text,
            "varexo",
            line_defines,
            macro_exclusions,
            macro_pre_masked,
        ),
    )
    model.exogenous = _merge_declarations_by_name(
        model.exogenous,
        _parse_statement_macro_for_declarations(
            macro_declaration_text,
            "varexo",
            line_defines,
            macro_exclusions,
            active_macro_lines,
        ),
    )
    # Dynare also accepts ``varexo_det`` for deterministic exogenous
    # shocks (e.g. policy news, anticipated tax changes).  Treat those
    # as exogenous for the purposes of LSP analysis — semantically
    # equivalent for identifier-scope checks.  Keep duplicates visible:
    # ``varexo eps;`` plus ``varexo_det eps;`` is still a symbol conflict
    # that the duplicate-declaration diagnostic must report.
    det_exo = _parse_declarations(stripped, text, "varexo_det", exclusions)
    det_exo = _merge_declarations_by_name(
        det_exo,
        _parse_macro_for_declarations(
            macro_active_text,
            "varexo_det",
            line_defines,
            macro_exclusions,
            macro_pre_masked,
        ),
    )
    det_exo = _merge_declarations_by_name(
        det_exo,
        _parse_statement_macro_for_declarations(
            macro_declaration_text,
            "varexo_det",
            line_defines,
            macro_exclusions,
            active_macro_lines,
        ),
    )
    model.deterministic_exogenous = det_exo
    model.exogenous.extend(det_exo)
    model.exogenous.sort(
        key=lambda v: (
            v.range.start.line,
            v.range.start.character,
            v.range.end.line,
            v.range.end.character,
        )
    )
    model.parameters = _merge_declarations_by_name(
        _parse_declarations(stripped, text, "parameters", exclusions),
        _parse_macro_for_declarations(
            macro_active_text,
            "parameters",
            line_defines,
            macro_exclusions,
            macro_pre_masked,
        ),
    )
    model.parameters = _merge_declarations_by_name(
        model.parameters,
        _parse_statement_macro_for_declarations(
            macro_declaration_text,
            "parameters",
            line_defines,
            macro_exclusions,
            active_macro_lines,
        ),
    )
    model.predetermined_variables = _parse_declarations(
        stripped,
        text,
        "predetermined_variables",
        exclusions,
    )
    model.predetermined_variables = _merge_declarations_by_name(
        model.predetermined_variables,
        _parse_macro_for_declarations(
            macro_active_text,
            "predetermined_variables",
            line_defines,
            macro_exclusions,
            macro_pre_masked,
        ),
    )
    model.predetermined_variables = _merge_declarations_by_name(
        model.predetermined_variables,
        _parse_statement_macro_for_declarations(
            macro_declaration_text,
            "predetermined_variables",
            line_defines,
            macro_exclusions,
            active_macro_lines,
        ),
    )

    # --- Parameter assignments ---
    parsed_param_assignments, parsed_helper_assignments = _parse_param_assignments(
        stripped,
        text,
        model.parameter_names(),
        exclusions,
    )
    known_assignment_values = {
        assignment.name: assignment.value
        for assignment in parsed_param_assignments + parsed_helper_assignments
        if assignment.value is not None
    }
    macro_param_assignments, macro_helper_assignments = (
        _parse_macro_for_param_assignments(
            macro_active_text,
            model.parameter_names(),
            line_defines,
            known_assignment_values,
            macro_exclusions,
            macro_pre_masked,
        )
    )
    model.param_assignments = _merge_macro_param_assignments(
        parsed_param_assignments,
        macro_param_assignments,
    )
    model.helper_assignments = _merge_macro_param_assignments(
        parsed_helper_assignments,
        macro_helper_assignments,
    )

    # --- Model block(s) ---
    # Dynare concatenates multiple ``model; ... end;`` blocks in a single
    # file, so collect every complete block.  A block whose body contains
    # another block keyword is missing its own end; (the regex stole the
    # other block's end;) — drop it here and let the partial-parse path
    # below handle it.
    model_matches = [
        m
        for m in _find_all_blocks(stripped, "model")
        if not re.search(
            r"(?<!\w)(?:initval|endval|shocks|steady_state_model)\s*(\([^)]*\))?\s*;",
            _mask_string_literals(m.group(2)),
            re.IGNORECASE,
        )
    ]

    if model_matches:
        # ``stripped`` has macro interpolations substituted (length-changing),
        # so offsets do NOT align with ``macro_pre_masked`` — pair the
        # stripped/raw block lists by index, generalizing the historical
        # first-with-first pairing.
        raw_model_blocks = [
            m
            for m in _find_all_blocks(macro_pre_masked, "model")
            if not re.search(
                r"(?<!\w)(?:initval|endval|shocks|steady_state_model)\s*(\([^)]*\))?\s*;",
                _mask_string_literals(m.group(2)),
                re.IGNORECASE,
            )
        ]
        all_equations: List[Equation] = []
        for block_idx, block_match in enumerate(model_matches):
            options = block_match.group(1) or ""
            if "linear" in options.lower():
                model.is_linear = True
            body = block_match.group(2)
            body_offset = block_match.start(2)
            block_equations = _parse_equations(body, body_offset, text)
            raw_model_match = (
                raw_model_blocks[block_idx]
                if block_idx < len(raw_model_blocks)
                else None
            )
            macro_body = raw_model_match.group(2) if raw_model_match else body
            macro_body_offset = (
                raw_model_match.start(2) if raw_model_match else body_offset
            )
            macro_equations, macro_ranges = _parse_macro_for_equations(
                macro_body,
                macro_body_offset,
                macro_active_text,
                line_defines,
            )
            if macro_equations:
                block_equations = [
                    eq
                    for eq in block_equations
                    if not any(
                        start <= eq.range.start.line <= end
                        for start, end in macro_ranges
                    )
                ] + macro_equations
            all_equations.extend(block_equations)
        all_equations.sort(
            key=lambda eq: (
                eq.range.start.line,
                eq.range.start.character,
                eq.range.end.line,
                eq.range.end.character,
            ),
        )
        model.model_equations = all_equations
        model.model_block_range = _range_from_match(text, model_matches[0])
        model.model_block_ranges = [_range_from_match(text, m) for m in model_matches]
    else:
        # If there's a model keyword but no end, record error and still parse
        # equations from the partial block so downstream checks work
        kw_match = re.search(
            r"(?<!\w)model\s*(\([^)]*\))?\s*;", stripped, re.IGNORECASE
        )
        if kw_match:
            # Error will be reported by _detect_unmatched_blocks; don't duplicate
            options = kw_match.group(1) or ""
            model.is_linear = "linear" in options.lower()
            # Parse equations from after model; to next block keyword
            after_model = stripped[kw_match.end() :]
            next_block = re.search(
                r"(?<!\w)(?:initval|endval|shocks|steady_state_model)\s*(\([^)]*\))?\s*;",
                _mask_string_literals(after_model),
                re.IGNORECASE,
            )
            if next_block:
                body = after_model[: next_block.start()]
            else:
                body = after_model
            body_offset = kw_match.end()
            model.model_equations = _parse_equations(
                body, body_offset, text, filter_commands=True
            )
            raw_stripped = macro_pre_masked
            raw_kw_match = re.search(
                r"(?<!\w)model\s*(\([^)]*\))?\s*;",
                raw_stripped,
                re.IGNORECASE,
            )
            raw_body = body
            raw_body_offset = body_offset
            if raw_kw_match:
                raw_after_model = raw_stripped[raw_kw_match.end() :]
                raw_next_block = re.search(
                    r"(?<!\w)(?:initval|endval|shocks|steady_state_model)\s*(\([^)]*\))?\s*;",
                    _mask_string_literals(raw_after_model),
                    re.IGNORECASE,
                )
                raw_body = (
                    raw_after_model[: raw_next_block.start()]
                    if raw_next_block
                    else raw_after_model
                )
                raw_body_offset = raw_kw_match.end()
            macro_equations, macro_ranges = _parse_macro_for_equations(
                raw_body,
                raw_body_offset,
                macro_active_text,
                line_defines,
            )
            if macro_equations:
                model.model_equations = [
                    eq
                    for eq in model.model_equations
                    if not any(
                        start <= eq.range.start.line <= end
                        for start, end in macro_ranges
                    )
                ] + macro_equations
                model.model_equations.sort(
                    key=lambda eq: (
                        eq.range.start.line,
                        eq.range.start.character,
                        eq.range.end.line,
                        eq.range.end.character,
                    ),
                )
            model.model_block_range = SourceRange(
                _offset_to_position(text, kw_match.start()),
                _offset_to_position(text, kw_match.end() + len(body)),
            )
            model.model_block_ranges = [model.model_block_range]

    _apply_on_the_fly_declarations(model)
    _apply_nostrict_implicit_exogenous(model)
    model.model_remove_names = _parse_model_remove_names(stripped)
    model.model_replacements = _parse_model_replacements(stripped, text)
    model.var_removed_names = _parse_var_removed_names(stripped)

    # --- steady_state_model block ---
    ss_match = _find_block(stripped, "steady_state_model")
    if ss_match:
        body = ss_match.group(2)
        body_offset = ss_match.start(2)
        ss_equations = _parse_equations(body, body_offset, text)
        # Expand resolvable @#for loops inside the block the same way the
        # model block does — otherwise only the first iteration's
        # assignments materialize and later variables look unassigned
        # (false W040/W041/W042).
        raw_ss_match = _find_block(macro_pre_masked, "steady_state_model")
        ss_macro_body = raw_ss_match.group(2) if raw_ss_match else body
        ss_macro_offset = raw_ss_match.start(2) if raw_ss_match else body_offset
        ss_macro_equations, ss_macro_ranges = _parse_macro_for_equations(
            ss_macro_body,
            ss_macro_offset,
            macro_active_text,
            line_defines,
        )
        if ss_macro_equations:
            ss_equations = [
                eq
                for eq in ss_equations
                if not any(
                    start <= eq.range.start.line <= end
                    for start, end in ss_macro_ranges
                )
            ] + ss_macro_equations

            # steady_state_model statements are SEQUENTIAL; Dynare unrolls
            # a loop iteration-by-iteration at the loop's position.  Sort
            # by anchor line only (the loop's start line for expanded
            # equations) — the stable sort keeps each expansion's
            # iteration-major emission order, where a full positional sort
            # would regroup it statement-major and break cross-iteration
            # recursions (false W040/W041).
            def _ss_anchor(eq: Equation) -> int:
                line = eq.range.start.line
                for start, end in ss_macro_ranges:
                    if start <= line <= end:
                        return start
                return line

            ss_equations.sort(key=_ss_anchor)
        model.steady_state_equations = ss_equations
        model.steady_state_block_range = _range_from_match(text, ss_match)

    # --- initval block ---
    iv_match = _find_block(stripped, "initval")
    if iv_match:
        body = iv_match.group(2)
        body_offset = iv_match.start(2)
        model.initval_entries = _parse_initval_block(
            body,
            body_offset,
            text,
            model.param_values(),
        )
        model.initval_block_range = _range_from_match(text, iv_match)

    # --- endval block ---
    ev_match = _find_block(stripped, "endval")
    if ev_match:
        body = ev_match.group(2)
        body_offset = ev_match.start(2)
        model.endval_entries = _parse_initval_block(
            body,
            body_offset,
            text,
            model.param_values(),
        )
        model.endval_block_range = _range_from_match(text, ev_match)

    # --- shocks block(s) ---
    # Dynare allows multiple ``shocks; ... end;`` blocks in a single file
    # (e.g. one block per period, or added by @#include).  Use
    # _find_all_blocks so that every block contributes its variable names
    # to ``model.shocks_vars``.  ``model.shocks_block_range`` is set to
    # the first block for backward-compatibility with diagnostics that
    # anchor their range there.
    shocks_matches = _find_all_blocks(stripped, "shocks")
    if shocks_matches:
        # Extract variable names from every shock-declaration form
        # Dynare accepts inside a ``shocks`` block:
        #   var e;
        #   var e = sigma^2;
        #   var e1, e2 = sigma;
        #   var e1; stderr 0.01;
        #   corr e1, e2 = 0.5;
        _SHOCK_DECL = re.compile(
            r"\b(?:var|corr)\s+"  # var or corr keyword
            r"([A-Za-z_][A-Za-z0-9_]*"  # first name
            r"(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)"  # optional comma-list
            r"\s*(?:=|;)",  # ends with = expr or ;
            re.IGNORECASE,
        )
        macro_shock_decl = re.compile(
            r"\b(?:var|corr)\s+"
            r"("
            r"[A-Za-z_][A-Za-z0-9_]*(?:@\{[A-Za-z_][A-Za-z0-9_]*\}[A-Za-z0-9_]*)*"
            r"(?:\s*,\s*"
            r"[A-Za-z_][A-Za-z0-9_]*(?:@\{[A-Za-z_][A-Za-z0-9_]*\}[A-Za-z0-9_]*)*"
            r")*"
            r")"
            r"\s*(?:=|;)",
            re.IGNORECASE,
        )
        macro_shocks_source = macro_pre_masked
        raw_shocks_matches = _find_all_blocks(macro_shocks_source, "shocks")

        for idx, shocks_match in enumerate(shocks_matches):
            if idx == 0:
                model.shocks_block_range = _range_from_match(text, shocks_match)
            shocks_body = stripped[shocks_match.start(2) : shocks_match.end(2)]
            shocks_body_offset = shocks_match.start(2)
            for sv_match in _SHOCK_DECL.finditer(shocks_body):
                name_group = sv_match.group(1)
                group_start = shocks_body_offset + sv_match.start(1)
                for name_match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", name_group):
                    name = name_match.group(0)
                    abs_end = group_start + name_match.end()
                    if text[abs_end : abs_end + 2] == "@{":
                        continue
                    if name and name not in model.shocks_vars:
                        model.shocks_vars.append(name)

            # Process @#for-expanded shocks in the corresponding macro-source block
            raw_shocks_match = (
                raw_shocks_matches[idx] if idx < len(raw_shocks_matches) else None
            )
            if raw_shocks_match:
                macro_shocks_body = macro_shocks_source[
                    raw_shocks_match.start(2) : raw_shocks_match.end(2)
                ]
                macro_shocks_offset = raw_shocks_match.start(2)
            else:
                macro_shocks_body = shocks_body
                macro_shocks_offset = shocks_body_offset
            for (
                start,
                _end,
                body_start_rel,
                body_end_rel,
                argument,
            ) in _simple_for_blocks(
                macro_shocks_body,
            ):
                loop_line = macro_active_text.count(
                    "\n", 0, macro_shocks_offset + start
                )
                scoped_defines = dict(line_defines.get(loop_line, {}))
                loop_values = _simple_macro_for_values(argument, scoped_defines)
                if loop_values is None:
                    continue
                loop_name, values = loop_values
                loop_body = macro_shocks_body[body_start_rel:body_end_rel]
                for sv_match in macro_shock_decl.finditer(loop_body):
                    for raw in re.split(r"\s*,\s*", sv_match.group(1).strip()):
                        for value in values:
                            loop_defines = dict(scoped_defines)
                            loop_defines[loop_name] = value
                            expanded, _source_map = _substitute_macro_vars_with_map(
                                raw,
                                loop_defines,
                            )
                            if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", expanded):
                                continue
                            if expanded not in model.shocks_vars:
                                model.shocks_vars.append(expanded)

    # --- estimation blocks (varobs / estimated_params / observation_trends) ---
    _parse_estimation_blocks(stripped, text, model)

    # --- optimal-policy constructs (planner_objective / ramsey / osr) ---
    _parse_policy_constructs(stripped, text, model)

    # --- Detect structural errors ---
    model.errors.extend(_detect_unmatched_blocks(stripped, text))
    model.errors.extend(
        _detect_missing_declaration_semicolons(stripped, text, exclusions)
    )
    model.errors.extend(_detect_keyword_typos(stripped, text, model, exclusions))
    model.errors.extend(
        _detect_missing_param_assignment_semicolons(
            stripped,
            text,
            model.parameter_names(),
            exclusions,
        )
    )
    model.errors.extend(_detect_missing_final_block_semicolons(stripped, text))
    model.errors.extend(_detect_missing_shocks_semicolons(stripped, text))

    # --- Pre-compute diagnostic caches ---
    # _unresolved_macro_for_template_ranges re-runs _macro_branch_state (which
    # was already called above) and _strip_non_macro_comments.  Precomputing
    # here moves its ~8 ms cost from the diagnostics path to the parse path,
    # so the per-keystroke diagnostic call is faster.
    model._cached_macro_for_template_ranges = _unresolved_macro_for_template_ranges(
        text
    )

    return model
