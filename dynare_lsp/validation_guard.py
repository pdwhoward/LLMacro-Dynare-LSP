"""Serialize same-document validation commits with editor lifecycle events.

The legacy validator mutates several caches and the workspace index while it
runs. Until it can commit an immutable snapshot, its whole mutation interval
must exclude didChange/didClose for that document. Numerical solves stay on
separate workers; no gate is held while revalidating dependent documents.
"""
from __future__ import annotations

import functools
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from weakref import WeakValueDictionary


def install(core) -> None:
    if getattr(core, "_validation_guard_installed", False):
        return
    core._validation_guard_installed = True
    from . import runtime_dispatch

    locks = WeakValueDictionary()
    active_documents: set[str] = set()
    registry_lock = threading.Lock()

    @contextmanager
    def gate(uri):
        # Keep a strong reference while waiting/holding; concurrent waiters
        # must never receive different locks for the same document.
        key = core._normalize_uri(uri)
        with registry_lock:
            lock = locks.get(key)
            if lock is None:
                lock = threading.RLock()
                locks[key] = lock
        with lock:
            yield

    original_validate = core._validate_document

    def live_document(uri):
        """Return an open editor document, or ``None`` for direct callers.

        pygls' ``get_text_document`` fabricates a disk-backed document for a
        URI that is not open (e.g. during didClose, after pygls dropped it).
        Treat that as "not open" so a closed URI is not re-registered as
        active and its final empty publish is not rejected.
        """
        try:
            workspace = core.server.workspace
            open_documents = getattr(workspace, "text_documents", None)
            if isinstance(open_documents, Mapping):
                document = open_documents.get(uri)
                if document is None:
                    return None
            else:
                document = workspace.get_text_document(uri)
        except Exception:
            return None
        with registry_lock:
            active_documents.add(core._normalize_uri(uri))
        return document

    @functools.wraps(original_validate)
    def validate(uri, text):
        with gate(uri):
            # Revalidation of dependent files can also arrive with text read
            # before waiting for this gate. Never accept that obsolete source.
            document = live_document(uri)
            if document is not None and document.source != text:
                return False
            return original_validate(uri, text)

    def validate_scheduled(uri, text, token):
        # This check is INSIDE the same gate used by didChange and didClose.
        # A worker that passed an earlier check cannot make old text current by
        # entering the legacy validator and allocating a newer generation.
        with gate(uri):
            with core._state_lock:
                if core._validation_tokens.get(uri, 0) != token:
                    return False
            document = live_document(uri)
            if document is None:
                return False
            if document.source != text:
                return False
            return core._validate_document(uri, text) is not False

    original_change = runtime_dispatch.document_change_handler
    if original_change is None:
        raise RuntimeError("Validation guard requires the document-change handler")

    @functools.wraps(original_change)
    def did_change(params):
        with gate(params.text_document.uri):
            return original_change(params)

    def publish(uri, diagnostics, generation, *, refresh_inlay_hints=False):
        from lsprotocol import types as lsp

        # pygls updates its document before calling didChange. Even while that
        # handler waits for the gate, never label an old analysis as new. The
        # version also lets the client discard a frame overtaken in transport.
        with core._client_send_lock:
            with core._state_lock:
                if core._document_generations.get(uri, 0) != generation:
                    return False
                source = core._document_sources.get(uri)
            document = live_document(uri)
            version = None
            if document is not None:
                version = getattr(document, "version", None)
                if source is None or document.source != source:
                    return False
                if getattr(document, "version", None) != version:
                    return False
            else:
                with registry_lock:
                    if core._normalize_uri(uri) in active_documents:
                        return False
            params = {"uri": uri, "diagnostics": diagnostics}
            if version is not None:
                params["version"] = version
            core.server.text_document_publish_diagnostics(
                lsp.PublishDiagnosticsParams(**params)
            )
            if refresh_inlay_hints:
                try:
                    core.server.workspace_inlay_hint_refresh(None)
                except Exception:
                    pass
        return True

    core._publish_diagnostics_if_current = publish

    def forget(uri: str) -> None:
        with registry_lock:
            active_documents.discard(core._normalize_uri(uri))

    # The loader's didClose handler holds this around callbacks AND cleanup.
    runtime_dispatch.document_close_callbacks.append(forget)
    runtime_dispatch.document_lifecycle_gate = gate
    runtime_dispatch.document_change_handler = did_change
    core._validate_document = validate
    core._validate_scheduled_document = validate_scheduled
