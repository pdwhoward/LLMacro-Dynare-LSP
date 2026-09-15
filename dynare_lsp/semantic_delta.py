"""Stateful ``semanticTokens/full/delta`` support with bounded snapshots."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

from . import runtime_dispatch

_MAX_SNAPSHOTS_PER_DOCUMENT = 4
_lock = threading.Lock()
_snapshots: dict[str, "OrderedDict[str, list[int]]"] = {}


def _result_id(uri: str, data: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(uri.encode("utf-8", "surrogatepass"))
    digest.update(b"\0")
    digest.update(",".join(str(value) for value in data).encode("ascii"))
    return digest.hexdigest()[:24]


def remember(uri: str, data: list[int]) -> str:
    result_id = _result_id(uri, data)
    with _lock:
        entries = _snapshots.setdefault(uri, OrderedDict())
        entries[result_id] = list(data)
        entries.move_to_end(result_id)
        while len(entries) > _MAX_SNAPSHOTS_PER_DOCUMENT:
            entries.popitem(last=False)
    return result_id


def previous(uri: str, result_id: str):
    with _lock:
        value = _snapshots.get(uri, {}).get(result_id)
        return list(value) if value is not None else None


def clear(uri: str) -> None:
    with _lock:
        _snapshots.pop(uri, None)


def compute_delta(old: list[int], new: list[int]) -> tuple[int, int, list[int]]:
    """Return a single token-boundary-aligned replacement edit."""
    old_tokens = [old[index : index + 5] for index in range(0, len(old), 5)]
    new_tokens = [new[index : index + 5] for index in range(0, len(new), 5)]
    prefix = 0
    while (
        prefix < len(old_tokens)
        and prefix < len(new_tokens)
        and old_tokens[prefix] == new_tokens[prefix]
    ):
        prefix += 1
    suffix = 0
    while (
        suffix < len(old_tokens) - prefix
        and suffix < len(new_tokens) - prefix
        and old_tokens[-suffix - 1] == new_tokens[-suffix - 1]
    ):
        suffix += 1
    old_end = len(old_tokens) - suffix
    new_end = len(new_tokens) - suffix
    replacement = [value for token in new_tokens[prefix:new_end] for value in token]
    return prefix * 5, (old_end - prefix) * 5, replacement


def install(core) -> None:
    if getattr(core, "_semantic_delta_installed", False):
        return
    core._semantic_delta_installed = True
    lsp = core.lsp
    default_full = runtime_dispatch.default_semantic_tokens_full_handler
    if default_full is None:
        return

    def full(params):
        result = default_full(params)
        data = list(result.data)
        uri = params.text_document.uri
        result_id = remember(uri, data)
        return lsp.SemanticTokens(data=data, result_id=result_id)

    runtime_dispatch.semantic_tokens_full_handler = full
    method = getattr(
        lsp,
        "TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL_DELTA",
        "textDocument/semanticTokens/full/delta",
    )

    @core.server.feature(method)
    def semantic_tokens_delta(params):
        uri = params.text_document.uri
        current = default_full(params)
        data = list(current.data)
        result_id = remember(uri, data)
        old = previous(uri, params.previous_result_id)
        if old is None:
            return lsp.SemanticTokens(data=data, result_id=result_id)
        start, delete_count, replacement = compute_delta(old, data)
        edits = []
        if delete_count or replacement:
            edits.append(
                lsp.SemanticTokensEdit(
                    start=start,
                    delete_count=delete_count,
                    data=replacement or None,
                )
            )
        return lsp.SemanticTokensDelta(result_id=result_id, edits=edits)

    core.semantic_tokens_delta = semantic_tokens_delta
    runtime_dispatch.document_close_callbacks.append(clear)
