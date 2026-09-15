"""Incremental text synchronization and range-stable parse reuse."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Iterable

from .parser import ParsedModel, _strip_non_macro_comments
from . import runtime_dispatch


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
        camel = name.split("_")[0] + "".join(
            part.capitalize() for part in name.split("_")[1:]
        )
        return obj.get(camel, default)
    return getattr(obj, name, default)


def _encoding_name(position_encoding: Any) -> str:
    value = getattr(position_encoding, "value", position_encoding)
    return str(value or "utf-16").lower().replace("_", "-")


def _character_to_index(line: str, character: int, position_encoding: Any) -> int:
    character = max(0, int(character))
    encoding = _encoding_name(position_encoding)
    if encoding.endswith("utf-8") or encoding == "utf-8":
        units = 0
        for index, char in enumerate(line):
            width = len(char.encode("utf-8"))
            if units + width > character:
                return index
            if units + width == character:
                return index + 1
            units += width
        return len(line)
    if encoding.endswith("utf-32") or encoding == "utf-32":
        return min(character, len(line))
    units = 0
    for index, char in enumerate(line):
        width = 2 if ord(char) > 0xFFFF else 1
        if units + width > character:
            return index
        if units + width == character:
            return index + 1
        units += width
    return len(line)


def _position_to_offset(text: str, position: Any, position_encoding: Any) -> int:
    line_number = max(0, int(_value(position, "line", 0)))
    character = max(0, int(_value(position, "character", 0)))
    # LSP recognizes only LF, CRLF, and CR as line endings. Python's
    # splitlines() also splits Unicode separators that remain document content.
    lines = re.split(r"(?<=\n)|(?<=\r)(?!\n)", text)
    if not lines:
        return 0
    if line_number >= len(lines):
        return len(text)
    offset = sum(len(line) for line in lines[:line_number])
    logical_line = lines[line_number].rstrip("\r\n")
    return offset + _character_to_index(logical_line, character, position_encoding)


def apply_content_changes(
    text: str,
    changes: Iterable[Any],
    position_encoding: Any = "utf-16",
) -> str:
    """Apply LSP edits sequentially, including UTF-8/16/32 positions."""
    updated = text
    for change in changes:
        replacement = str(_value(change, "text", ""))
        change_range = _value(change, "range")
        if change_range is None:
            updated = replacement
            continue
        start = _position_to_offset(
            updated, _value(change_range, "start"), position_encoding
        )
        end = _position_to_offset(
            updated, _value(change_range, "end"), position_encoding
        )
        if end < start:
            start, end = end, start
        updated = updated[:start] + replacement + updated[end:]
    return updated


def can_reuse_semantic_model(old_text: str, new_text: str) -> bool:
    """True for same-offset edits confined to ordinary comments."""
    return (
        len(old_text) == len(new_text)
        and old_text.count("\n") == new_text.count("\n")
        and _strip_non_macro_comments(old_text)
        == _strip_non_macro_comments(new_text)
    )


def rebind_model_source(model: ParsedModel, source: str) -> ParsedModel:
    if model.source_map:
        return replace(model, original_text=source)
    return replace(model, text=source, original_text=source)


def _install_workspace_fast_update(core) -> None:
    cls = core.WorkspaceIndex
    if hasattr(cls, "update_document_model"):
        return

    def update_document_model(self, uri: str, source: str, model: ParsedModel) -> None:
        key = core._normalize_uri(uri)
        with self._lock:
            self._models[key] = model
            self._sources[key] = source
            self._file_signatures.pop(key, None)
            self._effective_models.pop(key, None)
        self._register_includepath_directives(key, model)

    cls.update_document_model = update_document_model


def reuse_semantically_identical_document(
    core,
    uri: str,
    old_text: str | None,
    new_text: str,
) -> bool:
    if old_text is None or not can_reuse_semantic_model(old_text, new_text):
        return False
    with core._state_lock:
        model = core._document_models.get(uri)
        if model is None:
            return False
        rebound = rebind_model_source(model, new_text)
        generation = core._document_generations.get(uri, 0) + 1
        core._document_generations[uri] = generation
        core._document_models[uri] = rebound
        core._document_sources[uri] = new_text
    try:
        core._workspace_index.update_document_model(uri, new_text, rebound)
    except Exception:
        return False
    core._publish_all_diagnostics(uri, generation)
    return True


def install(core) -> None:
    if getattr(core, "_incremental_extension_installed", False):
        return
    core._incremental_extension_installed = True
    core._document_sources = {}
    _install_workspace_fast_update(core)

    original_validate = core._validate_document

    def validate_document(uri: str, text: str):
        with core._state_lock:
            core._document_sources[uri] = text
        return original_validate(uri, text)

    core._validate_document = validate_document

    def did_change(params):
        uri = params.text_document.uri
        if not params.content_changes:
            return None
        with core._state_lock:
            previous = core._document_sources.get(uri)
            pending = core._pending_preprocess.pop(uri, None)
        if pending is not None:
            pending.cancel()
        try:
            text = (
                apply_content_changes(
                    previous,
                    params.content_changes,
                    core._position_encoding,
                )
                if previous is not None
                else core.server.workspace.get_text_document(uri).source
            )
        except Exception:
            text = core.server.workspace.get_text_document(uri).source
        if reuse_semantically_identical_document(core, uri, previous, text):
            return None
        core._validate_document(uri, text)
        core._schedule_solve(uri)
        core._revalidate_cached_documents(schedule_solve=True, exclude_uri=uri)
        return None

    runtime_dispatch.document_change_handler = did_change
    runtime_dispatch.document_close_callbacks.append(
        lambda uri: core._document_sources.pop(uri, None)
    )
