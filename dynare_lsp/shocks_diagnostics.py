"""Value/consistency diagnostics for the Dynare ``shocks`` block.

Complements the existing "shock variable must be declared exogenous" check by
validating the *contents* of the block: a correlation must lie in [-1, 1], and
a given shock's variance/standard error (or a correlation pair) should not be
specified more than once.  All findings are warnings.

Codes:
  W110  shock correlation outside [-1, 1]
  W111  shock variance / correlation specified more than once
  W112  negative shock variance / standard error
  W113  jointly invalid shock covariance matrix
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Set, Tuple

from .diagnostics import Diagnostic, Severity

from .parser import (
    ParsedModel,
    Position,
    SourceRange,
    _find_all_blocks,
    _offset_to_position,
    _safe_eval,
    _strip_comments,
)

_FALLBACK_RANGE = SourceRange(Position(0, 0), Position(0, 1))


def check_shocks(
    model: ParsedModel,
    include_models: Optional[List[ParsedModel]] = None,
    param_known: Optional[Dict[str, float]] = None,
) -> List[Diagnostic]:
    """Validate the contents of active and include-visible ``shocks`` blocks."""
    known = param_known if param_known is not None else model.param_values()
    seen: Set[Tuple[str, object]] = set()
    diagnostics = _check_shocks_model(model, seen, known)
    include_models = include_models or []
    for include_model in include_models:
        include_known = (
            known if param_known is not None else include_model.param_values()
        )
        diagnostics.extend(
            _check_shocks_model(
                include_model,
                seen,
                include_known,
                include_model.include_anchor_range,
            )
        )
    diagnostics.extend(
        _check_covariance_matrix(
            [model, *include_models],
            known,
            model.shocks_block_range or _first_include_shocks_anchor(include_models),
        )
    )
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
                        diagnostics.append(
                            Diagnostic(
                                range=anchor,
                                severity=Severity.WARNING,
                                message=(
                                    "Covariance between "
                                    f"'{names[0]}' and '{names[1]}' is specified "
                                    "more than once in the shocks block."
                                ),
                                source="dynare",
                                code="W111",
                            )
                        )
                    seen.add(key)
                else:
                    for name in names:
                        key = ("var", name)
                        if key in seen:
                            diagnostics.append(
                                Diagnostic(
                                    range=anchor,
                                    severity=Severity.WARNING,
                                    message=(
                                        f"Shock '{name}' has its variance/standard "
                                        "error specified more than once in the "
                                        "shocks block."
                                    ),
                                    source="dynare",
                                    code="W111",
                                )
                            )
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
                            diagnostics.append(
                                Diagnostic(
                                    range=anchor,
                                    severity=Severity.WARNING,
                                    message=(
                                        f"Shock '{names[0]}' is given a negative "
                                        f"variance ({value:g}). A variance must be "
                                        "non-negative."
                                    ),
                                    source="dynare",
                                    code="W112",
                                )
                            )

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
                    diagnostics.append(
                        Diagnostic(
                            range=anchor,
                            severity=Severity.WARNING,
                            message=(
                                f"Correlation between '{first}' and '{second}' is "
                                "specified more than once in the shocks block."
                            ),
                            source="dynare",
                            code="W111",
                        )
                    )
                seen.add(key)
                numeric = _safe_eval(value, param_known)
                if numeric is None:
                    continue
                if abs(numeric) > 1.0:
                    diagnostics.append(
                        Diagnostic(
                            range=anchor,
                            severity=Severity.WARNING,
                            message=(
                                f"Correlation between '{first}' and '{second}' is "
                                f"{numeric:g}, which is outside the valid range [-1, 1]."
                            ),
                            source="dynare",
                            code="W110",
                        )
                    )

    return diagnostics


def _first_include_shocks_anchor(models: List[ParsedModel]) -> Optional[SourceRange]:
    for model in models:
        if model.shocks_block_range is not None:
            return model.include_anchor_range or model.shocks_block_range
    return None


def _ldl_psd_failure(matrix: List[List[float]], tol: float = 1e-10) -> Optional[float]:
    """Return a negative/invalid LDL pivot when a symmetric matrix is not PSD."""
    n = len(matrix)
    if n == 0:
        return None
    lower = [[0.0] * n for _ in range(n)]
    d = [0.0] * n
    scale = max(1.0, max(abs(value) for row in matrix for value in row))
    threshold = tol * scale
    for i in range(n):
        lower[i][i] = 1.0
        pivot = matrix[i][i] - sum(lower[i][k] * lower[i][k] * d[k] for k in range(i))
        if pivot < -threshold:
            return pivot
        if abs(pivot) <= threshold:
            d[i] = 0.0
            for j in range(i + 1, n):
                residual = matrix[j][i] - sum(
                    lower[j][k] * lower[i][k] * d[k] for k in range(i)
                )
                if abs(residual) > threshold:
                    return -abs(residual)
            continue
        d[i] = pivot
        for j in range(i + 1, n):
            residual = matrix[j][i] - sum(
                lower[j][k] * lower[i][k] * d[k] for k in range(i)
            )
            lower[j][i] = residual / pivot
    return None


def _check_covariance_matrix(
    models: List[ParsedModel],
    known: Dict[str, float],
    anchor: Optional[SourceRange],
) -> List[Diagnostic]:
    """W113 -- complete evaluable shock covariance matrix must be PSD."""
    exogenous: List[str] = []
    seen_exogenous: Set[str] = set()
    for model in models:
        for decl in model.exogenous:
            if decl.name not in seen_exogenous:
                seen_exogenous.add(decl.name)
                exogenous.append(decl.name)
    if len(exogenous) < 2:
        return []

    variances: Dict[str, float] = {name: 0.0 for name in exogenous}
    covariances: Dict[frozenset[str], float] = {}
    correlations: Dict[frozenset[str], float] = {}
    has_joint_spec = False

    for model in models:
        for block in _find_all_blocks(_strip_comments(model.text), "shocks"):
            pending_stderr_var: Optional[str] = None
            for raw_statement in block.group(2).split(";"):
                text = raw_statement.strip()
                if not text:
                    continue
                lowered = text.lower()
                if lowered.startswith("var"):
                    names_part = text[3:].split("=", 1)[0]
                    names = re.findall(r"[A-Za-z_]\w*", names_part)
                    pending_stderr_var = (
                        names[0] if len(names) == 1 and "=" not in text else None
                    )
                    if "=" not in text:
                        continue
                    value = _safe_eval(text.split("=", 1)[1].strip(), known)
                    if value is None:
                        continue
                    if len(names) == 1 and names[0] in variances:
                        variances[names[0]] = float(value)
                    elif (
                        len(names) >= 2
                        and names[0] in variances
                        and names[1] in variances
                    ):
                        covariances[frozenset((names[0], names[1]))] = float(value)
                        has_joint_spec = True
                    continue
                if lowered.startswith("stderr") and pending_stderr_var is not None:
                    expr = text[len("stderr") :].strip()
                    value = _safe_eval(expr, known)
                    if value is not None and pending_stderr_var in variances:
                        variances[pending_stderr_var] = float(value) ** 2
                    pending_stderr_var = None
                    continue
                pending_stderr_var = None
                if lowered.startswith("corr"):
                    match = re.match(
                        r"(?i)corr\s+([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*=\s*(.+)",
                        text,
                    )
                    if match is None:
                        continue
                    first, second = match.group(1), match.group(2)
                    value = _safe_eval(match.group(3).strip(), known)
                    if (
                        value is None
                        or first not in variances
                        or second not in variances
                    ):
                        continue
                    correlations[frozenset((first, second))] = float(value)
                    has_joint_spec = True

    if not has_joint_spec:
        return []
    if any(not math.isfinite(value) or value < 0 for value in variances.values()):
        # W112 (or an unevaluable upstream expression) owns diagonal problems.
        return []

    index = {name: i for i, name in enumerate(exogenous)}
    matrix = [[0.0] * len(exogenous) for _ in exogenous]
    for name, value in variances.items():
        matrix[index[name]][index[name]] = value
    for pair, value in covariances.items():
        first, second = tuple(pair)
        i, j = index[first], index[second]
        matrix[i][j] = matrix[j][i] = value
    for pair, rho in correlations.items():
        # Explicit covariance takes precedence if both forms are present; W111
        # already reports duplicate/conflicting specification surfaces.
        if pair in covariances:
            continue
        first, second = tuple(pair)
        value = rho * math.sqrt(variances[first] * variances[second])
        i, j = index[first], index[second]
        matrix[i][j] = matrix[j][i] = value

    failure = _ldl_psd_failure(matrix)
    if failure is None:
        return []

    evidence = f"LDL pivot {failure:.6g}"
    try:
        import numpy as np

        smallest = float(np.linalg.eigvalsh(np.asarray(matrix, dtype=float))[0])
        evidence = f"smallest eigenvalue {smallest:.6g}"
    except Exception:
        pass
    return [
        Diagnostic(
            range=anchor or _FALLBACK_RANGE,
            severity=Severity.WARNING,
            message=(
                "The jointly specified shock variance-covariance matrix is not "
                f"positive semidefinite ({evidence}). Pairwise variances and "
                "correlations can each be valid while their combination is not."
            ),
            source="dynare",
            code="W113",
        )
    ]
