"""Value/consistency diagnostics for the Dynare ``shocks`` block.

Complements the existing "shock variable must be declared exogenous" check by
validating the *contents* of the block: a correlation must lie in [-1, 1], and
a given shock's variance/standard error (or a correlation pair) should not be
specified more than once.  All findings are warnings.

Codes:
  W110  shock correlation outside [-1, 1]
  W111  shock variance / correlation specified more than once
  W112  negative shock variance / standard error
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from .diagnostics import Diagnostic, Severity
from .parser import (
    ParsedModel,
    SourceRange,
    _find_all_blocks,
    _offset_to_position,
    _safe_eval,
    _strip_comments,
)


def check_shocks(
    model: ParsedModel,
    include_models: Optional[List[ParsedModel]] = None,
    param_known: Optional[Dict[str, float]] = None,
) -> List[Diagnostic]:
    """Validate the contents of active and include-visible ``shocks`` blocks."""
    known = param_known if param_known is not None else model.param_values()
    seen: Set[Tuple[str, object]] = set()
    diagnostics = _check_shocks_model(model, seen, known)
    for include_model in include_models or []:
        include_known = known if param_known is not None else include_model.param_values()
        diagnostics.extend(_check_shocks_model(
            include_model,
            seen,
            include_known,
            include_model.include_anchor_range,
        ))
    return diagnostics


def _check_shocks_model(
    model: ParsedModel,
    seen: Set[Tuple[str, object]],
    param_known: Dict[str, float],
    range_override: Optional[SourceRange] = None,
) -> List[Diagnostic]:
    """Validate one parsed model's own ``shocks`` block contents."""
    if model.shocks_block_range is None:
        return []

    blocks = _find_all_blocks(_strip_comments(model.text), "shocks")
    if not blocks:
        return []

    diagnostics: List[Diagnostic] = []

    for block in blocks:
        anchor = range_override or SourceRange(
            _offset_to_position(model.text, block.start()),
            _offset_to_position(model.text, block.end()),
        )
        for statement in block.group(2).split(";"):
            text = statement.strip()
            if not text:
                continue
            lowered = text.lower()

            if lowered.startswith("var"):
                names_part = text[3:].split("=")[0]
                names = re.findall(r"[A-Za-z_]\w*", names_part)

                if len(names) >= 2:
                    # ``var e1, e2 = X`` is a COVARIANCE between two shocks,
                    # not two separate variances; key on the unordered pair so
                    # a later ``var e1 = ...`` / ``var e2 = ...`` is not misread
                    # as a duplicate.  A covariance may legitimately be
                    # negative, so the W112 sign check does not apply here.
                    key = ("cov", frozenset(names))
                    if key in seen:
                        diagnostics.append(Diagnostic(
                            range=anchor,
                            severity=Severity.WARNING,
                            message=(
                                "Covariance between "
                                f"'{names[0]}' and '{names[1]}' is specified "
                                "more than once in the shocks block."
                            ),
                            source="dynare",
                            code="W111",
                        ))
                    seen.add(key)
                else:
                    for name in names:
                        key = ("var", name)
                        if key in seen:
                            diagnostics.append(Diagnostic(
                                range=anchor,
                                severity=Severity.WARNING,
                                message=(
                                    f"Shock '{name}' has its variance/standard "
                                    "error specified more than once in the "
                                    "shocks block."
                                ),
                                source="dynare",
                                code="W111",
                            ))
                        seen.add(key)

                    # W112 -- ``var e = <expr>;`` setting a negative variance.
                    # Only a single-variable variance assignment that folds to
                    # a constant < 0 is flagged (an unevaluable RHS is left
                    # alone).  The ``stderr`` form is NOT checked: Dynare
                    # squares the standard error, so a negative ``stderr`` still
                    # yields a valid variance.
                    if "=" in text and len(names) == 1:
                        value = _safe_eval(text.split("=", 1)[1].strip(), param_known)
                        if value is not None and value < 0:
                            diagnostics.append(Diagnostic(
                                range=anchor,
                                severity=Severity.WARNING,
                                message=(
                                    f"Shock '{names[0]}' is given a negative "
                                    f"variance ({value:g}). A variance must be "
                                    "non-negative."
                                ),
                                source="dynare",
                                code="W112",
                            ))

            elif lowered.startswith("corr"):
                match = re.match(
                    r"(?i)corr\s+([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*=\s*(.+)",
                    text,
                )
                if not match:
                    continue
                first, second, value = (
                    match.group(1),
                    match.group(2),
                    match.group(3).strip(),
                )
                key = ("corr", frozenset((first, second)))
                if key in seen:
                    diagnostics.append(Diagnostic(
                        range=anchor,
                        severity=Severity.WARNING,
                        message=(
                            f"Correlation between '{first}' and '{second}' is "
                            "specified more than once in the shocks block."
                        ),
                        source="dynare",
                        code="W111",
                    ))
                seen.add(key)
                numeric = _safe_eval(value, param_known)
                if numeric is None:
                    continue
                if abs(numeric) > 1.0:
                    diagnostics.append(Diagnostic(
                        range=anchor,
                        severity=Severity.WARNING,
                        message=(
                            f"Correlation between '{first}' and '{second}' is "
                            f"{numeric:g}, which is outside the valid range [-1, 1]."
                        ),
                        source="dynare",
                        code="W110",
                    ))

    return diagnostics
