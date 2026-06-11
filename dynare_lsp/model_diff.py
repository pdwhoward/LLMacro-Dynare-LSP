"""Structural diff between two parsed Dynare models.

Researchers iterate on DSGE model variants constantly — adding a variable
here, retuning a calibration there, dropping or rewriting an equation
elsewhere. Today they compare two ``.mod`` files by reading them
side-by-side; this module provides a first-class "compare two models"
capability that classifies the differences into the dimensions that matter
for model interpretation:

  - Added / removed / common declarations (``var``, ``varexo``,
    ``parameters``) compared by *name*.
  - Parameter calibration changes for parameters present in both files,
    compared numerically when both right-hand-sides parse as ``float`` and
    by raw text otherwise.
  - Added / removed equations (compared by normalized text) and *changed*
    equations — near-matches paired up conservatively so a small rewrite
    appears as one change rather than one add + one remove.

The module is pure-functional and side-effect free. It consumes
``ParsedModel`` instances produced by ``dynare_lsp.parser.parse`` and
emits a structured ``ModelDiff`` dataclass with both JSON-friendly
(``to_dict``) and human-readable (``to_markdown``) renderings.

Consumed by:
  - The LSP server's ``dynare/compareModels`` custom command.
  - The MCP server's ``dynare_compare_models`` tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from .parser import Equation, ParsedModel
from .solver import _effective_param_values
from .steady_state import (
    _compute_ss_values_from_block,
    _compute_ss_values_from_initval,
)

MetadataValue = Union[str, Tuple[str, ...]]


# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------

# Numerical equality tolerance for parameter values. Differences below this
# are considered noise (rounding from re-saving the file, etc.).
_VALUE_TOL = 1e-12

# Edit-distance threshold for pairing a removed and an added equation as a
# single "changed" equation. Ratio is ``levenshtein / max(len_a, len_b)``;
# a pair below this ratio is treated as a near-match. Set conservatively
# so that genuinely different equations stay classified as add + remove.
_EQ_CHANGE_RATIO = 0.4


# ---------------------------------------------------------------------------
# Change dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParameterChange:
    """A change to a parameter calibration that appears in both files.

    ``old_value`` / ``new_value`` are populated when the right-hand-side
    parses as a bare ``float``. Otherwise both are ``None`` and the change
    is reported via the raw text fields only.
    """
    name: str
    old_value: Optional[float]
    new_value: Optional[float]
    old_raw: str
    new_raw: str


@dataclass(frozen=True)
class SteadyStateValueChange:
    """A change to an endogenous steady-state assignment."""

    name: str
    old_value: Optional[float]
    new_value: Optional[float]
    old_raw: str
    new_raw: str


@dataclass(frozen=True)
class ShockCalibrationChange:
    """A change to a variance, stderr, or correlation inside ``shocks``."""

    target: str
    old_raw: str
    new_raw: str


@dataclass(frozen=True)
class EquationChange:
    """A near-match pairing of one removed and one added equation.

    ``old_text`` / ``new_text`` are the normalized equation texts as they
    are compared internally; ``line_old`` / ``line_new`` are 1-based source
    line numbers for human-facing display.
    """
    old_text: str
    new_text: str
    line_old: int
    line_new: int


@dataclass(frozen=True)
class EquationMetadataChange:
    """A semantic tag/MCP change on otherwise-identical equation text."""

    equation: str
    option: str
    old_value: MetadataValue
    new_value: MetadataValue
    line_old: int
    line_new: int


@dataclass(frozen=True)
class ExogenousKindChange:
    """A change between stochastic and deterministic exogenous declarations."""

    name: str
    old_kind: str
    new_kind: str


@dataclass(frozen=True)
class ModelOptionChange:
    """A semantic option change on the model block."""

    option: str
    old_value: bool
    new_value: bool


@dataclass(frozen=True)
class DeclarationOptionChange:
    """A semantic option change on a declaration."""

    name: str
    option: str
    old_value: Any
    new_value: Any


@dataclass(frozen=True)
class ModelDiff:
    """Structural diff between two parsed Dynare models.

    All list fields are alphabetised (or kept in source order for the
    changed-equations field) so the diff is deterministic and easy to
    diff-of-the-diff in regression tests.
    """

    # Declarations (names only; alphabetised)
    added_endogenous: List[str]
    removed_endogenous: List[str]
    common_endogenous: List[str]
    added_exogenous: List[str]
    removed_exogenous: List[str]
    common_exogenous: List[str]
    changed_exogenous_kinds: List[ExogenousKindChange]
    added_parameters: List[str]
    removed_parameters: List[str]
    common_parameters: List[str]
    added_predetermined_variables: List[str]
    removed_predetermined_variables: List[str]
    common_predetermined_variables: List[str]
    changed_model_options: List[ModelOptionChange]
    changed_endogenous_options: List[DeclarationOptionChange]

    # Parameter calibration changes (only for parameters present in BOTH models)
    changed_parameter_values: List[ParameterChange]
    changed_steady_state_values: List[SteadyStateValueChange]
    changed_shock_calibrations: List[ShockCalibrationChange]

    # Equations: compared by normalized text
    added_equations: List[str]
    removed_equations: List[str]
    common_equations: List[str]
    changed_equations: List[EquationChange]
    changed_equation_metadata: List[EquationMetadataChange]

    @property
    def has_structural_changes(self) -> bool:
        """True if any declaration or equation was added or removed.

        Pure calibration retuning (parameter values changed but the same
        variables and equations on both sides) returns False — useful for
        deciding whether downstream artifacts (steady state, IRFs) need a
        re-solve or only a re-simulation.
        """
        return bool(
            self.added_endogenous or self.removed_endogenous
            or self.added_exogenous or self.removed_exogenous
            or self.changed_exogenous_kinds
            or self.added_parameters or self.removed_parameters
            or self.added_predetermined_variables
            or self.removed_predetermined_variables
            or self.changed_model_options
            or self.changed_endogenous_options
            or self.added_equations or self.removed_equations
            or self.changed_equations
            or self.changed_equation_metadata
        )

    @property
    def has_value_changes(self) -> bool:
        """True if any common parameter has a different calibration."""
        return bool(
            self.changed_parameter_values
            or self.changed_steady_state_values
            or self.changed_shock_calibrations
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation.

        Every nested dataclass is unrolled into a plain dict, and the two
        quick-check flags are materialised as keys so downstream consumers
        don't need to evaluate the properties themselves.
        """
        return {
            "added_endogenous": list(self.added_endogenous),
            "removed_endogenous": list(self.removed_endogenous),
            "common_endogenous": list(self.common_endogenous),
            "added_exogenous": list(self.added_exogenous),
            "removed_exogenous": list(self.removed_exogenous),
            "common_exogenous": list(self.common_exogenous),
            "changed_exogenous_kinds": [
                {
                    "name": change.name,
                    "old_kind": change.old_kind,
                    "new_kind": change.new_kind,
                }
                for change in self.changed_exogenous_kinds
            ],
            "added_parameters": list(self.added_parameters),
            "removed_parameters": list(self.removed_parameters),
            "common_parameters": list(self.common_parameters),
            "added_predetermined_variables": list(
                self.added_predetermined_variables,
            ),
            "removed_predetermined_variables": list(
                self.removed_predetermined_variables,
            ),
            "common_predetermined_variables": list(
                self.common_predetermined_variables,
            ),
            "changed_model_options": [
                {
                    "option": change.option,
                    "old_value": change.old_value,
                    "new_value": change.new_value,
                }
                for change in self.changed_model_options
            ],
            "changed_endogenous_options": [
                {
                    "name": change.name,
                    "option": change.option,
                    "old_value": change.old_value,
                    "new_value": change.new_value,
                }
                for change in self.changed_endogenous_options
            ],
            "changed_parameter_values": [
                {
                    "name": p.name,
                    "old_value": p.old_value,
                    "new_value": p.new_value,
                    "old_raw": p.old_raw,
                    "new_raw": p.new_raw,
                }
                for p in self.changed_parameter_values
            ],
            "changed_steady_state_values": [
                {
                    "name": change.name,
                    "old_value": change.old_value,
                    "new_value": change.new_value,
                    "old_raw": change.old_raw,
                    "new_raw": change.new_raw,
                }
                for change in self.changed_steady_state_values
            ],
            "changed_shock_calibrations": [
                {
                    "target": change.target,
                    "old_raw": change.old_raw,
                    "new_raw": change.new_raw,
                }
                for change in self.changed_shock_calibrations
            ],
            "added_equations": list(self.added_equations),
            "removed_equations": list(self.removed_equations),
            "common_equations": list(self.common_equations),
            "changed_equations": [
                {
                    "old_text": e.old_text,
                    "new_text": e.new_text,
                    "line_old": e.line_old,
                    "line_new": e.line_new,
                }
                for e in self.changed_equations
            ],
            "changed_equation_metadata": [
                {
                    "equation": e.equation,
                    "option": e.option,
                    "old_value": _metadata_value_to_json(e.old_value),
                    "new_value": _metadata_value_to_json(e.new_value),
                    "line_old": e.line_old,
                    "line_new": e.line_new,
                }
                for e in self.changed_equation_metadata
            ],
            "has_structural_changes": self.has_structural_changes,
            "has_value_changes": self.has_value_changes,
        }

    def to_markdown(self) -> str:
        """Render a compact human-readable summary.

        Designed to be scanned in under 30 seconds when there are roughly
        ten changes: one section per change category, bullet items, no
        boilerplate. Sections with no changes are omitted entirely.
        """
        lines: List[str] = ["# Model diff"]

        def _section(title: str, items: List[str]) -> None:
            if not items:
                return
            lines.append("")
            lines.append(f"## {title}")
            for item in items:
                lines.append(f"- {item}")

        _section("Added endogenous", self.added_endogenous)
        _section("Removed endogenous", self.removed_endogenous)
        _section("Added exogenous", self.added_exogenous)
        _section("Removed exogenous", self.removed_exogenous)
        if self.changed_exogenous_kinds:
            lines.append("")
            lines.append("## Changed exogenous kinds")
            for change in self.changed_exogenous_kinds:
                lines.append(
                    f"- `{change.name}`: `{change.old_kind}` -> `{change.new_kind}`"
                )
        _section("Added parameters", self.added_parameters)
        _section("Removed parameters", self.removed_parameters)
        _section(
            "Added predetermined variables",
            self.added_predetermined_variables,
        )
        _section(
            "Removed predetermined variables",
            self.removed_predetermined_variables,
        )
        if self.changed_model_options:
            lines.append("")
            lines.append("## Changed model options")
            for change in self.changed_model_options:
                lines.append(
                    f"- `{change.option}`: {change.old_value} -> {change.new_value}"
                )
        if self.changed_endogenous_options:
            lines.append("")
            lines.append("## Changed endogenous options")
            for change in self.changed_endogenous_options:
                lines.append(
                    f"- `{change.name}` `{change.option}`: "
                    f"{change.old_value} -> {change.new_value}"
                )

        if self.changed_parameter_values:
            lines.append("")
            lines.append("## Changed parameter values")
            for p in self.changed_parameter_values:
                if p.old_value is not None and p.new_value is not None:
                    lines.append(
                        f"- `{p.name}`: {p.old_value:g} -> {p.new_value:g}"
                    )
                else:
                    lines.append(
                        f"- `{p.name}`: `{p.old_raw}` -> `{p.new_raw}`"
                    )

        if self.changed_steady_state_values:
            lines.append("")
            lines.append("## Changed steady-state values")
            for change in self.changed_steady_state_values:
                if change.old_value is not None and change.new_value is not None:
                    lines.append(
                        f"- `{change.name}`: "
                        f"{change.old_value:g} -> {change.new_value:g}"
                    )
                else:
                    lines.append(
                        f"- `{change.name}`: "
                        f"`{change.old_raw}` -> `{change.new_raw}`"
                    )

        if self.changed_shock_calibrations:
            lines.append("")
            lines.append("## Changed shock calibrations")
            for change in self.changed_shock_calibrations:
                lines.append(
                    f"- `{change.target}`: "
                    f"`{change.old_raw}` -> `{change.new_raw}`"
                )

        if self.changed_equations:
            lines.append("")
            lines.append("## Changed equations")
            for e in self.changed_equations:
                lines.append(
                    f"- L{e.line_old} -> L{e.line_new}: "
                    f"`{e.old_text}` -> `{e.new_text}`"
                )

        if self.changed_equation_metadata:
            lines.append("")
            lines.append("## Changed equation metadata")
            for e in self.changed_equation_metadata:
                lines.append(
                    f"- L{e.line_old} -> L{e.line_new}: `{e.equation}` "
                    f"`{e.option}` {_format_metadata_value(e.old_value)} "
                    f"-> {_format_metadata_value(e.new_value)}"
                )

        if self.added_equations:
            lines.append("")
            lines.append("## Added equations")
            for eq in self.added_equations:
                lines.append(f"- `{eq}`")

        if self.removed_equations:
            lines.append("")
            lines.append("## Removed equations")
            for eq in self.removed_equations:
                lines.append(f"- `{eq}`")

        if len(lines) == 1:
            lines.append("")
            lines.append("_No structural or calibration changes detected._")

        lines.append("")  # trailing newline
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _metadata_value_to_json(value: MetadataValue) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def _format_metadata_value(value: MetadataValue) -> str:
    if isinstance(value, tuple):
        return str(list(value))
    return value


# Strip ``//`` line comments from a single equation's text. We do not need
# to strip ``%`` or ``/* ... */`` because ``parser._strip_comments`` already
# blanks those out before the equation text is captured.
_LINE_COMMENT = re.compile(r"//[^\n]*")
_WHITESPACE = re.compile(r"\s+")
# Whitespace adjacent to an operator or punctuation token is purely
# cosmetic — ``y=c;`` and ``y = c ;`` describe the same equation.  This
# pattern is applied AFTER the run-collapse pass so the matched span is
# bounded to single spaces.
_OPERATOR_PADDING = re.compile(r"\s*([=+\-*/^,()\[\]<>])\s*")


def _normalize_equation(text: str) -> str:
    """Normalize an equation text for set-membership comparison.

    Removes ``//`` line comments, strips surrounding whitespace, collapses
    runs of internal whitespace to a single space, drops any trailing
    semicolon, AND removes whitespace adjacent to operators / punctuation.
    Two equations that differ only in formatting compare equal after this
    transformation: ``y=c;`` and ``y = c;`` and ``y=  c;`` all normalize
    to ``y=c``.
    """
    text = _LINE_COMMENT.sub("", text)
    text = _WHITESPACE.sub(" ", text).strip()
    text = _OPERATOR_PADDING.sub(r"\1", text)
    if text.endswith(";"):
        text = text[:-1].rstrip()
    return text


def _equation_line(eq: Equation) -> int:
    return eq.range.start.line + 1


def _metadata_tuple(values: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(str(value) for value in values))


def _equation_metadata_changes(
    equation_key: str,
    a_equations: List[Equation],
    b_equations: List[Equation],
    common_count: int,
) -> List[EquationMetadataChange]:
    changes: List[EquationMetadataChange] = []
    for eq_a, eq_b in zip(
        a_equations[:common_count],
        b_equations[:common_count],
    ):
        if eq_a.name != eq_b.name:
            changes.append(EquationMetadataChange(
                equation=equation_key,
                option="name",
                old_value=eq_a.name,
                new_value=eq_b.name,
                line_old=_equation_line(eq_a),
                line_new=_equation_line(eq_b),
            ))
        old_tags = _metadata_tuple(eq_a.tags)
        new_tags = _metadata_tuple(eq_b.tags)
        if old_tags != new_tags:
            changes.append(EquationMetadataChange(
                equation=equation_key,
                option="tags",
                old_value=old_tags,
                new_value=new_tags,
                line_old=_equation_line(eq_a),
                line_new=_equation_line(eq_b),
            ))
        old_mcp = _metadata_tuple(eq_a.mcp_constraints)
        new_mcp = _metadata_tuple(eq_b.mcp_constraints)
        if old_mcp != new_mcp:
            changes.append(EquationMetadataChange(
                equation=equation_key,
                option="mcp_constraints",
                old_value=old_mcp,
                new_value=new_mcp,
                line_old=_equation_line(eq_a),
                line_new=_equation_line(eq_b),
            ))
    return changes


def _try_float(raw: str) -> Optional[float]:
    """Try to parse ``raw`` as a bare float; return None on failure.

    Tolerates a trailing semicolon and surrounding whitespace, mirroring
    how the parser captures parameter right-hand-sides.
    """
    cleaned = raw.strip().rstrip(";").strip()
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def _exogenous_kind_map(model: ParsedModel) -> Dict[str, str]:
    """Return ``name -> varexo/varexo_det`` using declaration source context."""
    from .parser import _strip_non_macro_comments

    lines = _strip_non_macro_comments(model.original_text or model.text).split("\n")
    out: Dict[str, str] = {}
    deterministic_names = model.deterministic_exogenous_names()
    for decl in model.exogenous:
        if decl.name in deterministic_names:
            out[decl.name] = "varexo_det"
            continue
        line_no = decl.range.start.line
        for idx in range(line_no, max(-1, line_no - 50), -1):
            line = lines[idx] if 0 <= idx < len(lines) else ""
            if idx == line_no:
                line = line[:decl.range.start.character + len(decl.name)]
            matches = list(re.finditer(
                r"\b(varexo_det|varexo)\b",
                line,
                re.IGNORECASE,
            ))
            if matches:
                out[decl.name] = matches[-1].group(1).lower()
                break
        out.setdefault(decl.name, "varexo")
    return out


def _levenshtein(a: str, b: str) -> int:
    """Iterative two-row Levenshtein distance.

    Implemented inline (rather than depending on ``python-Levenshtein``)
    so the diff module stays stdlib-only. Small ASCII strings only — the
    equations we feed it are typically under 200 characters.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, ca in enumerate(a, start=1):
        curr[0] = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev
    return prev[-1]


def _pair_changed_equations(
    removed: List[Equation],
    added: List[Equation],
) -> Tuple[List[EquationChange], List[Equation], List[Equation]]:
    """Greedily pair removed and added equations whose texts are similar.

    Returns ``(changes, leftover_removed, leftover_added)`` where each
    pair classified as a change is removed from both leftover lists. The
    pairing is greedy by smallest edit-distance ratio and bounded by
    ``_EQ_CHANGE_RATIO`` so dissimilar equations stay separate.
    """
    if not removed or not added:
        return [], list(removed), list(added)

    # Compute every (removed_idx, added_idx, distance, ratio) candidate.
    candidates: List[Tuple[float, int, int]] = []
    for i, r in enumerate(removed):
        rt = _normalize_equation(r.text)
        for j, a in enumerate(added):
            at = _normalize_equation(a.text)
            max_len = max(len(rt), len(at))
            if max_len == 0:
                continue
            d = _levenshtein(rt, at)
            ratio = d / max_len
            if ratio < _EQ_CHANGE_RATIO:
                candidates.append((ratio, i, j))

    # Greedy assignment: lowest ratio first, skip if either side already used.
    candidates.sort()
    used_r: set = set()
    used_a: set = set()
    changes: List[EquationChange] = []
    for ratio, i, j in candidates:
        if i in used_r or j in used_a:
            continue
        used_r.add(i)
        used_a.add(j)
        r = removed[i]
        a = added[j]
        changes.append(EquationChange(
            old_text=_normalize_equation(r.text),
            new_text=_normalize_equation(a.text),
            line_old=r.range.start.line + 1,
            line_new=a.range.start.line + 1,
        ))

    # Keep deterministic ordering: by line_old then line_new.
    changes.sort(key=lambda c: (c.line_old, c.line_new))

    leftover_removed = [r for i, r in enumerate(removed) if i not in used_r]
    leftover_added = [a for j, a in enumerate(added) if j not in used_a]
    return changes, leftover_removed, leftover_added


def _latest_param_assignment(model: ParsedModel) -> Dict[str, str]:
    """Return the *last* raw RHS string for each declared parameter.

    When a parameter is assigned more than once, Dynare's runtime keeps
    the last value, so the diff should reflect that. Parameter
    assignments inside steady_state_model are also effective calibrations
    in this codebase's solver/diagnostics semantics. Helper assignments
    (to undeclared names) are ignored.
    """
    out: Dict[str, str] = {}
    declared = model.parameter_names()
    for a in model.param_assignments:
        if a.name in declared:
            out[a.name] = a.expression
    for eq in model.steady_state_equations:
        text = re.sub(r"\[[^\]]*\]", "", eq.text.strip()).strip()
        if not text:
            continue

        if text.startswith("#"):
            match = re.match(
                r"#\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)",
                text,
                flags=re.DOTALL,
            )
            if not match:
                continue
            name = match.group(1)
            expr = match.group(2).strip()
        elif "=" in text:
            lhs, rhs = text.split("=", 1)
            name = lhs.strip()
            expr = rhs.strip()
        else:
            continue

        if name in declared and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            out[name] = expr.rstrip(";").strip()
    return out


def _latest_steady_state_assignment(model: ParsedModel) -> Dict[str, str]:
    """Return the last raw RHS string for each endogenous steady-state value."""
    return {
        name: rhs
        for name, (_source, rhs) in _latest_steady_state_source_assignment(
            model,
        ).items()
    }


def _latest_steady_state_source_assignment(
    model: ParsedModel,
) -> Dict[str, Tuple[str, str]]:
    """Return ``name -> (source block, raw RHS)`` for steady-state values."""
    out: Dict[str, Tuple[str, str]] = {}
    declared = model.endogenous_names()
    if model.steady_state_equations:
        for eq in model.steady_state_equations:
            text = re.sub(r"\[[^\]]*\]", "", eq.text.strip()).strip()
            if not text or text.startswith("#") or "=" not in text:
                continue

            lhs, rhs = text.split("=", 1)
            name = lhs.strip()
            if name in declared and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                out[name] = ("steady_state_model", rhs.rstrip(";").strip())
        return out
    for entry in model.initval_entries:
        if entry.name in declared:
            out[entry.name] = ("initval", entry.expression.rstrip(";").strip())
    return out


def _source_qualified_steady_state_raw(
    source: Optional[str],
    raw: Optional[str],
) -> str:
    raw_text = raw if raw is not None else ""
    return f"{source}: {raw_text}" if source else raw_text


def _computed_steady_state_values(model: ParsedModel) -> Dict[str, float]:
    if not model.steady_state_equations and not model.initval_entries:
        return {}
    try:
        if model.steady_state_equations:
            values, _errors = _compute_ss_values_from_block(model)
        else:
            values, _errors = _compute_ss_values_from_initval(model)
    except Exception:
        return {}
    return values


_SHOCK_DECL_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _shock_calibration_map(model: ParsedModel) -> Dict[str, str]:
    """Return normalized shock calibration statements keyed by target."""
    from .parser import _strip_non_macro_comments

    source = _strip_non_macro_comments(model.original_text or model.text)
    match = re.search(
        r"(?<!\w)shocks\s*;(.*?)(?<!\w)end\s*;",
        source,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return {}

    calibrations: Dict[str, str] = {}
    current_vars: List[str] = []
    for statement in match.group(1).split(";"):
        normalized = _WHITESPACE.sub(" ", statement).strip()
        if not normalized:
            continue

        var_match = re.match(
            r"var\s+(.+?)(?:\s*=\s*(.+))?$",
            normalized,
            re.IGNORECASE,
        )
        if var_match:
            names = _SHOCK_DECL_IDENTIFIER.findall(var_match.group(1))
            current_vars = names
            rhs = var_match.group(2)
            if rhs is not None and names:
                target = "var " + ",".join(names)
                calibrations[target] = rhs.strip()
            continue

        stderr_match = re.match(r"stderr\s+(.+)$", normalized, re.IGNORECASE)
        if stderr_match:
            rhs = stderr_match.group(1).strip()
            for name in current_vars:
                calibrations[f"stderr {name}"] = rhs
            if not current_vars:
                calibrations["stderr"] = rhs
            continue

        corr_match = re.match(
            r"corr\s+(.+?)(?:\s*=\s*(.+))?$",
            normalized,
            re.IGNORECASE,
        )
        if corr_match:
            names = _SHOCK_DECL_IDENTIFIER.findall(corr_match.group(1))
            rhs = corr_match.group(2)
            if rhs is not None and names:
                target = "corr " + ",".join(names)
                calibrations[target] = rhs.strip()

    return calibrations


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compare_models(model_a: ParsedModel, model_b: ParsedModel) -> ModelDiff:
    """Compare two parsed Dynare models and return a structural diff.

    The diff is symmetric in semantics — ``model_a`` is the "before" /
    "old" side and ``model_b`` is the "after" / "new" side. ``added``
    means "present in B but not in A"; ``removed`` means "present in A
    but not in B"; ``common`` means "present in both".
    """
    # --- Declaration sets (compared by name) -------------------------------
    end_a = model_a.endogenous_names()
    end_b = model_b.endogenous_names()
    exo_a = model_a.exogenous_names()
    exo_b = model_b.exogenous_names()
    par_a = model_a.parameter_names()
    par_b = model_b.parameter_names()
    pred_a = {v.name for v in model_a.predetermined_variables}
    pred_b = {v.name for v in model_b.predetermined_variables}
    exo_kind_a = _exogenous_kind_map(model_a)
    exo_kind_b = _exogenous_kind_map(model_b)
    endo_log_a = {v.name: v.log_transform for v in model_a.endogenous}
    endo_log_b = {v.name: v.log_transform for v in model_b.endogenous}
    endo_options_a = {v.name: v.options.strip() for v in model_a.endogenous}
    endo_options_b = {v.name: v.options.strip() for v in model_b.endogenous}
    changed_exogenous_kinds = [
        ExogenousKindChange(name, exo_kind_a[name], exo_kind_b[name])
        for name in sorted(exo_a & exo_b)
        if exo_kind_a.get(name) != exo_kind_b.get(name)
    ]
    changed_model_options: List[ModelOptionChange] = []
    if model_a.is_linear != model_b.is_linear:
        changed_model_options.append(ModelOptionChange(
            option="linear",
            old_value=model_a.is_linear,
            new_value=model_b.is_linear,
        ))
    changed_endogenous_options = [
        DeclarationOptionChange(
            name=name,
            option="log_transform",
            old_value=endo_log_a.get(name, False),
            new_value=endo_log_b.get(name, False),
        )
        for name in sorted(end_a & end_b)
        if endo_log_a.get(name, False) != endo_log_b.get(name, False)
    ]
    changed_endogenous_options.extend(
        DeclarationOptionChange(
            name=name,
            option="declaration_options",
            old_value=endo_options_a.get(name, ""),
            new_value=endo_options_b.get(name, ""),
        )
        for name in sorted(end_a & end_b)
        if endo_options_a.get(name, "") != endo_options_b.get(name, "")
        and {endo_options_a.get(name, ""), endo_options_b.get(name, "")} != {"", "log"}
    )

    # --- Parameter calibration changes (intersection only) -----------------
    rhs_a = _latest_param_assignment(model_a)
    rhs_b = _latest_param_assignment(model_b)
    # Evaluated calibrations let us detect changes that the raw RHS
    # string misses — e.g. ``alpha = helper*2`` where ``helper`` itself
    # changed between the two models, so the raw RHS string is the
    # same on both sides but the realised value differs.
    eval_a = _effective_param_values(model_a)
    eval_b = _effective_param_values(model_b)
    common_params = par_a & par_b

    changed_params: List[ParameterChange] = []
    for name in sorted(common_params):
        ra = rhs_a.get(name)
        rb = rhs_b.get(name)
        if ra is None or rb is None:
            # Parameter declared in both but only assigned in one — that
            # is a calibration change in spirit, so report it.
            if ra != rb:
                changed_params.append(ParameterChange(
                    name=name,
                    old_value=eval_a.get(name) if eval_a.get(name) is not None else (_try_float(ra) if ra is not None else None),
                    new_value=eval_b.get(name) if eval_b.get(name) is not None else (_try_float(rb) if rb is not None else None),
                    old_raw=ra if ra is not None else "",
                    new_raw=rb if rb is not None else "",
                ))
            continue

        va = _try_float(ra)
        vb = _try_float(rb)
        # Prefer the fully-evaluated value from param_values() so changes
        # transmitted through helper variables (``helper = 0.5; alpha =
        # helper*2;`` vs ``helper = 0.6; alpha = helper*2;``) surface as
        # parameter changes even though ``alpha``'s raw RHS string is
        # identical on both sides.
        ev_a = eval_a.get(name)
        ev_b = eval_b.get(name)
        if ev_a is not None and ev_b is not None:
            if abs(ev_a - ev_b) > _VALUE_TOL:
                changed_params.append(ParameterChange(
                    name=name,
                    old_value=ev_a,
                    new_value=ev_b,
                    old_raw=ra,
                    new_raw=rb,
                ))
            continue
        if va is not None and vb is not None:
            if abs(va - vb) > _VALUE_TOL:
                changed_params.append(ParameterChange(
                    name=name,
                    old_value=va,
                    new_value=vb,
                    old_raw=ra,
                    new_raw=rb,
                ))
        else:
            # Fall back to raw-text comparison (normalised on whitespace).
            ca = _WHITESPACE.sub(" ", ra).strip().rstrip(";").strip()
            cb = _WHITESPACE.sub(" ", rb).strip().rstrip(";").strip()
            if ca != cb:
                changed_params.append(ParameterChange(
                    name=name,
                    old_value=va,
                    new_value=vb,
                    old_raw=ra,
                    new_raw=rb,
                ))

    # --- Endogenous steady-state value changes ----------------------------
    ss_source_rhs_a = _latest_steady_state_source_assignment(model_a)
    ss_source_rhs_b = _latest_steady_state_source_assignment(model_b)
    ss_rhs_a = {name: rhs for name, (_source, rhs) in ss_source_rhs_a.items()}
    ss_rhs_b = {name: rhs for name, (_source, rhs) in ss_source_rhs_b.items()}
    ss_eval_a = _computed_steady_state_values(model_a)
    ss_eval_b = _computed_steady_state_values(model_b)

    changed_ss_values: List[SteadyStateValueChange] = []
    for name in sorted(end_a & end_b):
        ra = ss_rhs_a.get(name)
        rb = ss_rhs_b.get(name)
        source_a = ss_source_rhs_a.get(name, ("", ""))[0]
        source_b = ss_source_rhs_b.get(name, ("", ""))[0]
        if ra is None and rb is None:
            continue
        ev_a = ss_eval_a.get(name)
        ev_b = ss_eval_b.get(name)
        if ev_a is not None and ev_b is not None:
            if abs(ev_a - ev_b) > _VALUE_TOL:
                changed_ss_values.append(SteadyStateValueChange(
                    name=name,
                    old_value=ev_a,
                    new_value=ev_b,
                    old_raw=ra if ra is not None else "",
                    new_raw=rb if rb is not None else "",
                ))
            elif source_a != source_b:
                changed_ss_values.append(SteadyStateValueChange(
                    name=name,
                    old_value=ev_a,
                    new_value=ev_b,
                    old_raw=_source_qualified_steady_state_raw(source_a, ra),
                    new_raw=_source_qualified_steady_state_raw(source_b, rb),
                ))
            continue
        ca = _WHITESPACE.sub(" ", ra or "").strip().rstrip(";").strip()
        cb = _WHITESPACE.sub(" ", rb or "").strip().rstrip(";").strip()
        if ca != cb or source_a != source_b:
            changed_ss_values.append(SteadyStateValueChange(
                name=name,
                old_value=ev_a,
                new_value=ev_b,
                old_raw=(
                    _source_qualified_steady_state_raw(source_a, ra)
                    if source_a != source_b else ra if ra is not None else ""
                ),
                new_raw=(
                    _source_qualified_steady_state_raw(source_b, rb)
                    if source_a != source_b else rb if rb is not None else ""
                ),
            ))

    # --- Shock calibration changes ---------------------------------------
    shock_a = _shock_calibration_map(model_a)
    shock_b = _shock_calibration_map(model_b)
    changed_shock_calibrations = [
        ShockCalibrationChange(
            target=target,
            old_raw=shock_a.get(target, ""),
            new_raw=shock_b.get(target, ""),
        )
        for target in sorted(set(shock_a) | set(shock_b))
        if shock_a.get(target) != shock_b.get(target)
    ]

    # --- Equation sets (compared by normalized text) -----------------------
    norm_a: Dict[str, List[Equation]] = {}
    for eq in model_a.model_equations:
        key = _normalize_equation(eq.text)
        if key:
            norm_a.setdefault(key, []).append(eq)
    norm_b: Dict[str, List[Equation]] = {}
    for eq in model_b.model_equations:
        key = _normalize_equation(eq.text)
        if key:
            norm_b.setdefault(key, []).append(eq)

    common_eq_keys = set(norm_a) & set(norm_b)

    removed_eqs: List[Equation] = []
    added_eqs: List[Equation] = []
    changed_eq_metadata: List[EquationMetadataChange] = []
    for key in sorted(set(norm_a) | set(norm_b)):
        a_equations = norm_a.get(key, [])
        b_equations = norm_b.get(key, [])
        common_count = min(len(a_equations), len(b_equations))
        changed_eq_metadata.extend(_equation_metadata_changes(
            key,
            a_equations,
            b_equations,
            common_count,
        ))
        removed_eqs.extend(a_equations[common_count:])
        added_eqs.extend(b_equations[common_count:])

    changed_eqs, leftover_removed, leftover_added = _pair_changed_equations(
        removed_eqs, added_eqs,
    )

    return ModelDiff(
        added_endogenous=sorted(end_b - end_a),
        removed_endogenous=sorted(end_a - end_b),
        common_endogenous=sorted(end_a & end_b),
        added_exogenous=sorted(exo_b - exo_a),
        removed_exogenous=sorted(exo_a - exo_b),
        common_exogenous=sorted(exo_a & exo_b),
        changed_exogenous_kinds=changed_exogenous_kinds,
        added_parameters=sorted(par_b - par_a),
        removed_parameters=sorted(par_a - par_b),
        common_parameters=sorted(common_params),
        added_predetermined_variables=sorted(pred_b - pred_a),
        removed_predetermined_variables=sorted(pred_a - pred_b),
        common_predetermined_variables=sorted(pred_a & pred_b),
        changed_model_options=changed_model_options,
        changed_endogenous_options=changed_endogenous_options,
        changed_parameter_values=changed_params,
        changed_steady_state_values=changed_ss_values,
        changed_shock_calibrations=changed_shock_calibrations,
        added_equations=sorted(_normalize_equation(e.text) for e in leftover_added),
        removed_equations=sorted(_normalize_equation(e.text) for e in leftover_removed),
        common_equations=sorted(common_eq_keys),
        changed_equations=changed_eqs,
        changed_equation_metadata=changed_eq_metadata,
    )
