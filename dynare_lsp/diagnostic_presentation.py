"""Richer presentation of global Dynare diagnostics for LSP clients.

The analysis engine deliberately stays transport-agnostic.  This adapter is
installed by the packaged server entrypoint and enriches already-computed
W071/W080 diagnostics with narrow primary ranges and LSP related locations.
"""

from __future__ import annotations

import re
from typing import Optional


def _matching_document(server_module, source_text: Optional[str]):
    if source_text is None:
        return None, None
    matches = []
    with server_module._state_lock:
        for uri, model in server_module._document_models.items():
            text = getattr(model, "original_text", "") or model.text
            if text == source_text:
                matches.append((uri, model))
    if len(matches) != 1:
        return None, None
    return matches[0]


def _model_keyword_range(server_module, model, source_text):
    from .parser import Position, SourceRange

    block = model.model_block_range
    if block is None:
        return None
    start = block.start
    return server_module._to_lsp_range_in_text(
        source_text,
        SourceRange(start, Position(start.line, start.character + len("model"))),
    )


def _related_for_bk(server_module, lsp, uri, model, source_text):
    with server_module._state_lock:
        bk = server_module._document_bk_results.get(uri)
    if bk is None:
        return []
    wanted = set(getattr(bk, "forward_variables", []) or [])
    out = []
    for declaration in model.endogenous:
        if declaration.name not in wanted:
            continue
        out.append(
            lsp.DiagnosticRelatedInformation(
                location=lsp.Location(
                    uri=uri,
                    range=server_module._to_lsp_range_in_text(
                        source_text, declaration.range
                    ),
                ),
                message=f"Forward-looking variable '{declaration.name}'",
            )
        )
    return out


def _related_for_jacobian(server_module, lsp, uri, model, source_text, message):
    out = []
    names = set()
    for match in re.finditer(r"Collinear variables[^:]*:\s*([^.]*)\.", message):
        names.update(name.strip() for name in match.group(1).split(",") if name.strip())
    for declaration in model.endogenous:
        if declaration.name in names:
            out.append(
                lsp.DiagnosticRelatedInformation(
                    location=lsp.Location(
                        uri=uri,
                        range=server_module._to_lsp_range_in_text(
                            source_text, declaration.range
                        ),
                    ),
                    message=f"Collinear variable '{declaration.name}'",
                )
            )

    equations = model.static_model_equations()
    ordinals = {
        int(value)
        for value in re.findall(r"\bequation\s+(\d+)\b", message, re.IGNORECASE)
    }
    for ordinal in sorted(ordinals):
        if not 1 <= ordinal <= len(equations):
            continue
        equation = equations[ordinal - 1]
        out.append(
            lsp.DiagnosticRelatedInformation(
                location=lsp.Location(
                    uri=uri,
                    range=server_module._to_lsp_range_in_text(
                        source_text, equation.range
                    ),
                ),
                message=(
                    f"Collinear equation {ordinal}"
                    + (f" ({equation.name})" if equation.name else "")
                ),
            )
        )
    return out


def install() -> None:
    """Install the presentation adapter once on the loaded LSP server module."""
    from lsprotocol import types as lsp

    from . import server as server_module

    if getattr(server_module, "_related_diagnostic_adapter_installed", False):
        return
    original = server_module._to_lsp_diagnostic

    def enriched(diagnostic, source_text=None):
        converted = original(diagnostic, source_text)
        if diagnostic.code not in {"W071", "W080"}:
            return converted
        uri, model = _matching_document(server_module, source_text)
        if uri is None or model is None or source_text is None:
            return converted

        narrow = _model_keyword_range(server_module, model, source_text)
        if narrow is not None:
            converted.range = narrow
        if diagnostic.code == "W071":
            related = _related_for_bk(server_module, lsp, uri, model, source_text)
        else:
            related = _related_for_jacobian(
                server_module, lsp, uri, model, source_text, diagnostic.message
            )
        if related:
            converted.related_information = related
        return converted

    server_module._to_lsp_diagnostic = enriched
    setattr(server_module, "_related_diagnostic_adapter_installed", True)
