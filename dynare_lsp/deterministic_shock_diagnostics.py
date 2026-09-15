"""Static validation for deterministic ``shocks`` schedules.

This module is intentionally separate from the stochastic variance/correlation
checks so deterministic-schedule validation composes cleanly with other shock
analysis features.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

from .diagnostics import Diagnostic, Severity
from .parser import (
    ParsedModel,
    SourceRange,
    _find_all_blocks,
    _offset_to_position,
    _strip_comments,
)


def _split_top_level_commas(text: str) -> List[str]:
    parts: List[str] = []
    start = 0
    depth = 0
    quote: Optional[str] = None
    for i, char in enumerate(text):
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _period_list(expr: str) -> tuple[Optional[List[int]], Optional[str]]:
    periods: List[int] = []
    for item in _split_top_level_commas(expr):
        if ":" in item:
            fields = [field.strip() for field in item.split(":")]
            if len(fields) != 2 or not all(
                re.fullmatch(r"[+-]?\d+", f) for f in fields
            ):
                return None, None
            start, end = (int(field) for field in fields)
            if start <= 0 or end <= 0 or end < start:
                return None, f"invalid deterministic-shock period range '{item}'"
            periods.extend(range(start, end + 1))
        else:
            if not re.fullmatch(r"[+-]?\d+", item):
                return None, None
            value = int(item)
            if value <= 0:
                return None, f"deterministic-shock period must be positive, got {value}"
            periods.append(value)
    return periods, None


def check_deterministic_shocks(models: List[ParsedModel]) -> List[Diagnostic]:
    """Return E070-E072/W114 findings for literal deterministic schedules."""
    diagnostics: List[Diagnostic] = []
    assigned: Set[Tuple[str, int]] = set()

    for model in models:
        stripped = _strip_comments(model.text)
        anchor_override = model.include_anchor_range
        for block in _find_all_blocks(stripped, "shocks"):
            body = block.group(2)
            body_start = block.start(2)
            current_name: Optional[str] = None
            current_start = 0
            periods_expr: Optional[str] = None
            values_expr: Optional[str] = None
            offset = 0

            def finalize() -> None:
                nonlocal current_name, periods_expr, values_expr
                if current_name is None or (
                    periods_expr is None and values_expr is None
                ):
                    return
                start = body_start + current_start
                rng = anchor_override or SourceRange(
                    _offset_to_position(model.text, start),
                    _offset_to_position(model.text, start + len(current_name)),
                )
                if periods_expr is None or values_expr is None:
                    missing = "periods" if periods_expr is None else "values"
                    diagnostics.append(
                        Diagnostic(
                            range=rng,
                            severity=Severity.ERROR,
                            message=(
                                f"Deterministic shock '{current_name}' has an incomplete "
                                f"schedule: missing {missing} statement."
                            ),
                            source="dynare",
                            code="E070",
                        )
                    )
                    return
                periods, error = _period_list(periods_expr)
                if error is not None:
                    diagnostics.append(
                        Diagnostic(
                            range=rng,
                            severity=Severity.ERROR,
                            message=error.capitalize() + ".",
                            source="dynare",
                            code="E072",
                        )
                    )
                    return
                if periods is None:
                    return
                period_groups = _split_top_level_commas(periods_expr)
                values = _split_top_level_commas(values_expr)
                valid_value_counts = {1, len(period_groups), len(periods)}
                if len(values) not in valid_value_counts:
                    diagnostics.append(
                        Diagnostic(
                            range=rng,
                            severity=Severity.ERROR,
                            message=(
                                f"Deterministic shock '{current_name}' expands to "
                                f"{len(period_groups)} schedule segment(s) but supplies "
                                f"{len(values)} "
                                "explicit value(s)."
                            ),
                            source="dynare",
                            code="E071",
                        )
                    )
                for period in periods:
                    key = (current_name, period)
                    if key in assigned:
                        diagnostics.append(
                            Diagnostic(
                                range=rng,
                                severity=Severity.WARNING,
                                message=(
                                    f"Deterministic shock '{current_name}' assigns period "
                                    f"{period} more than once; later assignments can "
                                    "override earlier values."
                                ),
                                source="dynare",
                                code="W114",
                            )
                        )
                    assigned.add(key)

            for raw in body.split(";"):
                text = raw.strip()
                raw_start = body.find(raw, offset)
                offset = raw_start + len(raw) + 1
                if not text:
                    continue
                lowered = text.lower()
                if lowered.startswith("var"):
                    finalize()
                    current_name = None
                    periods_expr = values_expr = None
                    match = re.match(r"(?i)var\s+([A-Za-z_]\w*)\s*$", text)
                    if match is not None:
                        current_name = match.group(1)
                        current_start = raw_start + match.start(1)
                    continue
                if current_name is None:
                    continue
                if lowered.startswith("stderr"):
                    current_name = None
                    periods_expr = values_expr = None
                    continue
                if lowered.startswith("periods"):
                    periods_expr = text[len("periods") :].strip()
                elif lowered.startswith("values"):
                    values_expr = text[len("values") :].strip()
            finalize()
    return diagnostics
