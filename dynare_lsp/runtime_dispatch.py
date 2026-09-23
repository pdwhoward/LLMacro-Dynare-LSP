"""Mutable protocol dispatch points populated by optional runtime extensions."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, Optional

Handler = Callable[..., Any]
default_document_change_handler: Optional[Handler] = None
document_change_handler: Optional[Handler] = None
default_semantic_tokens_full_handler: Optional[Handler] = None
semantic_tokens_full_handler: Optional[Handler] = None
default_watched_files_handler: Optional[Handler] = None
watched_files_handler: Optional[Handler] = None
document_close_callbacks: list[Callable[[str], None]] = []
# Installed after all analysis hooks; no-op during initial feature registration.
def document_lifecycle_gate(uri: str) -> Any:
    return nullcontext()
