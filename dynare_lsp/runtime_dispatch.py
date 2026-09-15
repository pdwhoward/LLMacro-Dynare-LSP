"""Mutable protocol dispatch points populated by optional runtime extensions."""

from __future__ import annotations

from typing import Any, Callable, Optional

Handler = Callable[..., Any]
default_document_change_handler: Optional[Handler] = None
document_change_handler: Optional[Handler] = None
default_semantic_tokens_full_handler: Optional[Handler] = None
semantic_tokens_full_handler: Optional[Handler] = None
default_watched_files_handler: Optional[Handler] = None
watched_files_handler: Optional[Handler] = None
document_close_callbacks: list[Callable[[str], None]] = []
