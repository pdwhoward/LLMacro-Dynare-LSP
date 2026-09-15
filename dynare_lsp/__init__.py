"""Dynare Language Server Protocol implementation.

Provides IDE features for Dynare .mod files including syntax validation,
model-aware navigation, and deterministic analysis.  A small import hook loads
optional performance extensions around the established server/preprocessor
modules without duplicating their implementations.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules
from types import ModuleType
from typing import Optional, Sequence

__version__ = "0.4.0"

_TARGETS = {
    f"{__name__}.server",
    f"{__name__}.preprocessor",
    f"{__name__}.mcp_server",
}


class _ExtensionLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, original: importlib.abc.Loader) -> None:
        self.fullname = fullname
        self.original = original

    def create_module(self, spec):
        create = getattr(self.original, "create_module", None)
        return create(spec) if create is not None else None

    def get_code(self, fullname: str):
        """Preserve runpy / python -m support from the wrapped loader."""
        get_code = getattr(self.original, "get_code", None)
        return get_code(fullname) if get_code is not None else None

    def exec_module(self, module: ModuleType) -> None:
        if self.fullname.endswith(".server"):
            self._exec_server(module)
        else:
            self.original.exec_module(module)  # type: ignore[attr-defined]
        from .runtime_extensions import after_module_load

        after_module_load(self.fullname, module)

    def _exec_server(self, module: ModuleType) -> None:
        import functools
        import importlib

        from lsprotocol import types as lsp

        try:
            pygls_server = importlib.import_module("pygls.server")
            language_server = getattr(pygls_server, "LanguageServer")
        except (ImportError, AttributeError):
            pygls_server = importlib.import_module("pygls.lsp.server")
            language_server = getattr(pygls_server, "LanguageServer")

        from . import runtime_dispatch

        original_init = language_server.__init__
        original_feature = language_server.feature

        def incremental_init(instance, *args, **kwargs):
            if kwargs.get("text_document_sync_kind") == lsp.TextDocumentSyncKind.Full:
                kwargs["text_document_sync_kind"] = lsp.TextDocumentSyncKind.Incremental
            return original_init(instance, *args, **kwargs)

        def dispatching_feature(instance, feature_name, *args, **kwargs):
            register = original_feature(instance, feature_name, *args, **kwargs)

            def decorator(function):
                if feature_name == lsp.TEXT_DOCUMENT_DID_CHANGE:
                    runtime_dispatch.default_document_change_handler = function

                    @functools.wraps(function)
                    def did_change(*feature_args, **feature_kwargs):
                        handler = runtime_dispatch.document_change_handler
                        target = handler or function
                        return target(*feature_args, **feature_kwargs)

                    return register(did_change)

                if feature_name == lsp.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL:
                    runtime_dispatch.default_semantic_tokens_full_handler = function

                    @functools.wraps(function)
                    def semantic_tokens_full(*feature_args, **feature_kwargs):
                        handler = runtime_dispatch.semantic_tokens_full_handler
                        target = handler or function
                        return target(*feature_args, **feature_kwargs)

                    return register(semantic_tokens_full)

                if feature_name == lsp.WORKSPACE_DID_CHANGE_WATCHED_FILES:
                    runtime_dispatch.default_watched_files_handler = function

                    @functools.wraps(function)
                    def watched_files(*feature_args, **feature_kwargs):
                        handler = runtime_dispatch.watched_files_handler
                        target = handler or function
                        return target(*feature_args, **feature_kwargs)

                    return register(watched_files)

                if feature_name == lsp.TEXT_DOCUMENT_DID_CLOSE:

                    @functools.wraps(function)
                    def did_close(params, *feature_args, **feature_kwargs):
                        uri = params.text_document.uri
                        for callback in tuple(
                            runtime_dispatch.document_close_callbacks
                        ):
                            try:
                                callback(uri)
                            except Exception:
                                pass
                        return function(params, *feature_args, **feature_kwargs)

                    return register(did_close)

                return register(function)

            return decorator

        language_server.__init__ = incremental_init
        language_server.feature = dispatching_feature
        try:
            self.original.exec_module(module)  # type: ignore[attr-defined]
        finally:
            language_server.__init__ = original_init
            language_server.feature = original_feature


class _ExtensionFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]],
        target: Optional[ModuleType] = None,
    ):
        if fullname not in _TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if (
            spec is None
            or spec.loader is None
            or isinstance(spec.loader, _ExtensionLoader)
        ):
            return spec
        spec.loader = _ExtensionLoader(fullname, spec.loader)
        return spec


if not any(isinstance(finder, _ExtensionFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _ExtensionFinder())


def _issue_modules(prefix: str):
    package_dir = Path(__file__).parent
    return sorted(
        module.name
        for module in iter_modules([str(package_dir)])
        if module.name.startswith(prefix)
    )


def _load_diagnostic_explanation_extensions() -> None:
    for name in _issue_modules("explain_issue_"):
        import_module(f"{__name__}.{name}")


def _load_issue_installers() -> None:
    """Install composable feature hooks without turning core files into hotspots."""
    for name in _issue_modules("install_issue_"):
        module = import_module(f"{__name__}.{name}")
        install = getattr(module, "install", None)
        if callable(install):
            install()


_load_diagnostic_explanation_extensions()

# Apply Dynare-compatible QZ zero-threshold semantics to every import surface
# (LSP, CLI, and MCP) without duplicating the BK solver.
_install_bk_qz_compat = import_module(f"{__name__}.bk_qz_compat").install
_install_bk_qz_compat()
_load_issue_installers()
