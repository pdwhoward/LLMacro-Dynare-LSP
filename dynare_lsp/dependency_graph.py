"""Persistent include graph and selective workspace invalidation."""

from __future__ import annotations

from . import runtime_dispatch


def _base_instance_key(key: str) -> str:
    base, marker, suffix = key.rpartition("#")
    return base if marker and suffix.isdigit() else key


def patch_workspace_index(workspace_module) -> None:
    cls = workspace_module.WorkspaceIndex
    if getattr(cls, "_dependency_graph_patched", False):
        return
    cls._dependency_graph_patched = True
    original_init = cls.__init__
    original_update = cls.update_document
    original_remove = cls.remove_document
    original_resolve = cls.resolve_all_includes

    def ensure(instance) -> None:
        if not hasattr(instance, "_dependency_edges"):
            instance._dependency_edges = {}
            instance._reverse_dependency_edges = {}

    def drop_root(instance, root: str) -> None:
        ensure(instance)
        for dependency in instance._dependency_edges.pop(root, set()):
            roots = instance._reverse_dependency_edges.get(dependency)
            if roots is None:
                continue
            roots.discard(root)
            if not roots:
                instance._reverse_dependency_edges.pop(dependency, None)

    def init(instance, *args, **kwargs):
        original_init(instance, *args, **kwargs)
        ensure(instance)

    def update(instance, uri, source):
        key = workspace_module._normalize_uri(uri)
        with instance._lock:
            drop_root(instance, key)
        return original_update(instance, uri, source)

    def remove(instance, uri):
        key = workspace_module._normalize_uri(uri)
        with instance._lock:
            drop_root(instance, key)
        return original_remove(instance, uri)

    def resolve(instance, uri):
        result = original_resolve(instance, uri)
        root = workspace_module._normalize_uri(uri)
        dependencies = {
            _base_instance_key(key) for key in result if _base_instance_key(key) != root
        }
        with instance._lock:
            drop_root(instance, root)
            instance._dependency_edges[root] = dependencies
            for dependency in dependencies:
                instance._reverse_dependency_edges.setdefault(dependency, set()).add(
                    root
                )
        return result

    def dependent_roots(instance, uri, include_self: bool = True):
        ensure(instance)
        key = workspace_module._normalize_uri(uri)
        with instance._lock:
            roots = set(instance._reverse_dependency_edges.get(key, set()))
            if include_self and key in instance._dependency_edges:
                roots.add(key)
        return roots

    def dependency_graph(instance):
        ensure(instance)
        with instance._lock:
            return {
                root: set(dependencies)
                for root, dependencies in instance._dependency_edges.items()
            }

    cls.__init__ = init
    cls.update_document = update
    cls.remove_document = remove
    cls.resolve_all_includes = resolve
    cls.dependent_roots = dependent_roots
    cls.dependency_graph = dependency_graph


def install(core) -> None:
    if getattr(core, "_dependency_invalidation_installed", False):
        return
    core._dependency_invalidation_installed = True
    import dynare_lsp.workspace as workspace_module

    patch_workspace_index(workspace_module)
    scopes = {}
    previous_change = runtime_dispatch.document_change_handler

    def remember_scope(uri: str) -> None:
        roots = core._workspace_index.dependent_roots(uri)
        scopes[core._normalize_uri(uri)] = roots

    def did_change(params):
        remember_scope(params.text_document.uri)
        if previous_change is not None:
            return previous_change(params)
        default = runtime_dispatch.default_document_change_handler
        return default(params) if default is not None else None

    original_revalidate = core._revalidate_cached_documents

    def revalidate(schedule_solve=True, exclude_uri=None):
        if exclude_uri is None:
            return original_revalidate(schedule_solve, exclude_uri)
        scope = scopes.pop(core._normalize_uri(exclude_uri), None)
        if not scope:
            return original_revalidate(schedule_solve, exclude_uri)
        with core._state_lock:
            uris = [
                uri
                for uri in core._document_models
                if uri != exclude_uri and core._normalize_uri(uri) in scope
            ]
        for uri in uris:
            try:
                source = core.server.workspace.get_text_document(uri).source
            except Exception:
                continue
            core._validate_document(uri, source)
            if schedule_solve:
                core._schedule_solve(uri)
        return None

    def watched_files(params):
        affected = set()
        for event in params.changes:
            affected.update(core._workspace_index.dependent_roots(event.uri))
            try:
                core._workspace_index.remove_document(event.uri)
            except Exception:
                pass
        with core._state_lock:
            uris = [
                uri
                for uri in core._document_models
                if core._normalize_uri(uri) in affected
            ]
        for uri in uris:
            try:
                source = core.server.workspace.get_text_document(uri).source
            except Exception:
                continue
            core._validate_document(uri, source)
            core._schedule_solve(uri)
        return None

    core._revalidate_cached_documents = revalidate
    runtime_dispatch.document_change_handler = did_change
    runtime_dispatch.watched_files_handler = watched_files
    runtime_dispatch.document_close_callbacks.append(remember_scope)
