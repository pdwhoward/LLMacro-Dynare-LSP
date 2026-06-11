"""Semantics-preserving formatter for Dynare ``.mod`` files.

The formatter only ever changes **whitespace** on code lines and leaves
comments and ``@#`` macro-directive lines verbatim.  Safety is guaranteed by a
token invariant: the input and the proposed output must be identical once
comments/macros are blanked and all whitespace is removed.  If that invariant
does not hold — or the file cannot be processed — the formatter declines and
returns ``None`` (no edits), so it can never change a model's meaning or "fix"
an error.

What it normalises:
  * block indentation by nesting depth (one indent unit per ``model;`` /
    ``initval;`` / ``shocks;`` / ... block; ``end;`` flush with its opener);
  * operator spacing on simple code lines (``y=a*c+e`` -> ``y = a * c + e``),
    leaving lead/lag and function-call parentheses tight (``c(-1)``, ``exp(x)``);
  * aligned ``=`` in a run of consecutive simple assignments;
  * trailing whitespace, collapsed blank-line runs, single final newline.

The indent unit is configurable; the default is a tab.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .parser import _strip_comments

# Blocks opened by a keyword and closed by ``end;`` (so their bodies indent).
_BLOCK_OPENERS = (
    "model",
    "steady_state_model",
    "initval",
    "endval",
    "histval",
    "shocks",
    "estimated_params_init",
    "estimated_params_bounds",
    "estimated_params",
    "observation_trends",
    "ramsey_constraints",
    "optim_weights",
    "homotopy_setup",
    "moment_calibration",
    "irf_calibration",
    "verbatim",
)
_OPENER_RE = re.compile(
    r"\s*(?:" + "|".join(_BLOCK_OPENERS) + r")\b\s*(?:\([^)]*\))?\s*;\s*",
    re.IGNORECASE,
)
_END_RE = re.compile(r"\s*end\s*;\s*", re.IGNORECASE)

# Tokens of a "simple" expression line (no strings/tags/macros).
_SIMPLE_TOKEN_RE = re.compile(
    r"\d+\.?\d*(?:[eE][+-]?\d+)?"  # 1, 1.5, 1e-3
    r"|\.\d+(?:[eE][+-]?\d+)?"  # .5
    r"|[A-Za-z_]\w*"  # identifier
    r"|<=|>=|==|!=|\*\*"  # multi-char operators
    r"|[-+*/^=(),;<>]"  # single-char operators / punctuation
)
# A line is only spaced if it contains none of these (strings, tags, macro
# interpolation, model-local defs, line continuations, ...).
_UNSAFE_CHARS = set("'\"[]{}@#%:\\!")

_BINARY_OPS = {"+", "-", "*", "/", "^", "=", "<", ">", "<=", ">=", "==", "!=", "**"}


# Tokens for the safety invariant: identifiers, numbers, and multi-char
# operators are kept whole so a token SPLIT (``==`` -> ``= =``) or JOIN
# (``y c`` -> ``yc``) changes the stream even though plain
# whitespace-stripping would hide it.
_CANONICAL_TOKEN_RE = re.compile(
    r"[A-Za-z_]\w*"
    r"|\d+\.?\d*(?:[eE][+-]?\d+)?"
    r"|\.\d+(?:[eE][+-]?\d+)?"
    r"|==|!=|<=|>=|&&|\|\||\*\*"
    r"|\S"
)

# Identifier glued to macro interpolation(s) (``e_xi@{index}``): whitespace
# between the parts changes the macro-expanded token, so the invariant must
# capture the glue even though ``_strip_comments`` blanks ``@{...}`` content.
_INTERP_ATOM_RE = re.compile(r"\w*(?:@\{[^}\n]*\}\w*)+")


def _canonical(text: str) -> str:
    """Comment/macro-blanked token stream (the whitespace-only invariant key),
    plus the glued interpolation atoms from the raw text."""
    tokens = _CANONICAL_TOKEN_RE.findall(_strip_comments(text))
    interp_atoms = _INTERP_ATOM_RE.findall(text)
    return "\x00".join(tokens) + "\x01" + "\x00".join(interp_atoms)


def format_text(text: str, indent_unit: str = "\t") -> Optional[str]:
    """Return *text* reformatted, or ``None`` to decline (leave unchanged)."""
    if not text.strip():
        return None
    try:
        formatted = _reformat(text, indent_unit)
    except Exception:
        return None
    if formatted == text:
        return None
    # Safety: only whitespace (and never comment/macro content) may have changed.
    if _canonical(formatted) != _canonical(text):
        return None
    return formatted


def _reformat(text: str, indent_unit: str) -> str:
    # Preserve the file's line-ending convention: the per-line strips below
    # drop every "\r", so a plain "\n" join would silently rewrite a CRLF
    # file wholesale (pure diff noise in the editor).
    eol = "\r\n" if "\r\n" in text else "\n"
    orig_lines = [line.rstrip("\r") for line in text.split("\n")]
    stripped_lines = [line.rstrip("\r") for line in _strip_comments(text).split("\n")]
    out = _format_lines(orig_lines, stripped_lines, indent_unit, 0)
    while out and out[-1] == "":
        out.pop()
    return eol.join(out) + eol


def _depth_before(stripped_lines: List[str], upto: int) -> int:
    """Block-nesting depth after the lines ``[0, upto)`` (openers +1, ``end;`` -1)."""
    depth = 0
    for stripped in stripped_lines[:upto]:
        if _END_RE.fullmatch(stripped):
            depth = max(0, depth - 1)
        elif _OPENER_RE.fullmatch(stripped):
            depth += 1
    return depth


def _format_lines(
    orig_lines: List[str],
    stripped_lines: List[str],
    indent_unit: str,
    start_depth: int,
) -> List[str]:
    out: List[str] = []
    depth = start_depth
    # Pending run of consecutive simple assignments for ``=`` alignment, stored
    # as (output_index, lhs, rhs_and_comment).
    assign_run: List[Tuple[int, str, str]] = []

    def _flush_run() -> None:
        if len(assign_run) > 1:
            width = max(len(lhs) for _idx, lhs, _rest in assign_run)
            for idx, lhs, rest in assign_run:
                out[idx] = (
                    out[idx][: out[idx].index(lhs)] + lhs.ljust(width) + " = " + rest
                )
        assign_run.clear()

    for orig, stripped in zip(orig_lines, stripped_lines):
        code = stripped.rstrip()

        if code.strip() == "":
            # No code on this line: blank, comment-only, or macro directive.
            _flush_run()
            if orig.strip() == "":
                if out and out[-1] == "":
                    continue  # collapse consecutive blank lines
                out.append("")
            else:
                out.append(orig.rstrip())  # comment / @# line, verbatim content
            continue

        if _END_RE.fullmatch(stripped):
            _flush_run()
            depth = max(0, depth - 1)
            out.append(indent_unit * depth + orig.strip())
            continue

        indent = indent_unit * depth
        code_end = len(code)
        body = orig[:code_end]
        trailing = orig[code_end:].strip()  # inline comment, verbatim

        if trailing and not trailing.startswith(("//", "%", "/*")):
            # The blanked tail is not a comment — it is macro content such as
            # a line-ending interpolation (``... e_xi@{index}``).  Re-gluing
            # it after a spaced body would inject whitespace into the
            # expanded token (``e_xi 7``), so keep the line verbatim.
            _flush_run()
            out.append(indent + orig.strip())
            if _OPENER_RE.fullmatch(stripped):
                depth += 1
            continue

        no_midline_comment = orig[:code_end] == stripped[:code_end]
        spaced = _space_line(body.strip()) if no_midline_comment else None

        line_body = spaced if spaced is not None else body.strip()
        line = indent + line_body
        if trailing:
            line += " " + trailing
        line_index = len(out)
        out.append(line)

        # Alignment bookkeeping: a plain ``lhs = rhs`` simple assignment, not a
        # block opener, with no inline comment, joins the current run.
        # ``(?!=)`` keeps ``a == b`` comparison lines (e.g. inside verbatim
        # blocks) out of the assignment run — re-splicing them as
        # ``lhs = rest`` would split the ``==`` token.
        match = (
            re.fullmatch(r"([A-Za-z_]\w*)\s*=(?!=)\s*(.*\S)", line_body)
            if spaced is not None and not trailing
            else None
        )
        if match and not _OPENER_RE.fullmatch(stripped):
            assign_run.append((line_index, match.group(1), match.group(2)))
        else:
            _flush_run()

        if _OPENER_RE.fullmatch(stripped):
            depth += 1

    _flush_run()
    return out


def format_range(
    text: str,
    start_line: int,
    end_line: int,
    indent_unit: str = "\t",
) -> Optional[Tuple[int, int, str]]:
    """Format only the line range ``[start_line, end_line]`` (inclusive).

    Returns ``(start_line, end_line, replacement_text)`` for a whole-line edit,
    or ``None`` to decline.  The starting indentation depth is computed from the
    blocks open before the range.
    """
    if not text.strip():
        return None
    # Match _reformat: emit the file's own line-ending convention, or a CRLF
    # document gets a mixed-EOL range after every selection format.
    eol = "\r\n" if "\r\n" in text else "\n"
    raw_lines = text.split("\n")
    orig_lines = [line.rstrip("\r") for line in raw_lines]
    stripped_lines = [line.rstrip("\r") for line in _strip_comments(text).split("\n")]
    start_line = max(0, start_line)
    end_line = min(end_line, len(orig_lines) - 1)
    if start_line > end_line:
        return None
    try:
        depth = _depth_before(stripped_lines, start_line)
        formatted = _format_lines(
            orig_lines[start_line : end_line + 1],
            stripped_lines[start_line : end_line + 1],
            indent_unit,
            depth,
        )
    except Exception:
        return None
    original_slice = "\n".join(raw_lines[start_line : end_line + 1])
    replacement = eol.join(formatted)
    if raw_lines[end_line].endswith("\r"):
        # The caller's whole-line edit span consumes the last line's "\r".
        replacement += "\r"
    if replacement == original_slice:
        return None
    if _canonical(replacement) != _canonical(original_slice):
        return None
    return start_line, end_line, replacement


def _space_line(code: str) -> Optional[str]:
    """Return *code* with normalised operator spacing, or ``None`` if not simple."""
    if any(ch in _UNSAFE_CHARS for ch in code):
        return None
    tokens = _SIMPLE_TOKEN_RE.findall(code)
    if "".join(tokens) != re.sub(r"\s+", "", code):
        return None  # unrecognised characters -> don't touch
    return _join_tokens(tokens)


def _join_tokens(tokens: List[str]) -> str:
    result = ""
    prev_kind: Optional[str] = None
    for i, tok in enumerate(tokens):
        kind = _classify(tok, prev_kind)
        result += _separator(prev_kind, kind) + tok
        prev_kind = kind
    return result


def _classify(tok: str, prev_kind: Optional[str]) -> str:
    if tok in ("+", "-"):
        # Unary when nothing meaningful precedes an operand: at the start, after
        # another operator, after ``(`` or ``,``.
        if prev_kind in (None, "binop", "unary", "lparen", "comma"):
            return "unary"
        return "binop"
    if tok in _BINARY_OPS:
        return "binop"
    if tok == "(":
        return "lparen"
    if tok == ")":
        return "rparen"
    if tok == ",":
        return "comma"
    if tok == ";":
        return "semi"
    return "operand"


def _separator(prev_kind: Optional[str], kind: str) -> str:
    if prev_kind is None:
        return ""
    if kind in ("rparen", "comma", "semi"):
        return ""
    if prev_kind in ("lparen", "unary"):
        return ""
    if prev_kind in ("comma", "semi"):
        return " "
    if kind == "lparen":
        # Function call / lead-lag: ``f(`` and ``)(`` stay tight; ``a * (`` spaces.
        return " " if prev_kind == "binop" else ""
    if kind == "binop" or prev_kind == "binop":
        return " "
    # operand/rparen followed by operand -> single space (e.g. ``var y c``).
    return " "
