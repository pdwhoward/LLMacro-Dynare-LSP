"""Install built-in runtime extensions after selected modules are imported."""

from __future__ import annotations

from types import ModuleType


def after_module_load(fullname: str, module: ModuleType) -> None:
    if fullname.endswith(".server"):
        from .dependency_graph import install as install_dependency_graph
        from .diagnostic_tiers import install as install_diagnostic_tiers
        from .equation_dependencies import install_server
        from .incremental import install as install_incremental
        from . import runtime_dispatch
        from .semantic_delta import install as install_semantic_delta
        from .semantic_index import install as install_semantic_index
        from .analysis_api import register_lsp

        install_incremental(module)
        install_diagnostic_tiers(module)
        install_dependency_graph(module)
        install_semantic_index(module)
        install_semantic_delta(module)
        install_server(module)
        register_lsp(module)
        if runtime_dispatch.document_change_handler is not None:
            setattr(module, "did_change", runtime_dispatch.document_change_handler)
    elif fullname.endswith(".preprocessor"):
        from .preprocessor_cache import install

        install(module)
    elif fullname.endswith(".mcp_server"):
        from .equation_dependencies import install_mcp
        from .analysis_api import install_mcp as install_analysis_api

        install_mcp(module)
        install_analysis_api(module)
