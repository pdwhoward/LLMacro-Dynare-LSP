"""One registration point for built-in, versioned analysis features.

Features are bundled Python modules, not user-supplied plugins. Their ordinary
functions are shared by MCP, LSP commands and the JSON CLI. Legacy tools remain
unchanged; feature discovery neither executes models nor performs analysis.
"""
from __future__ import annotations

import functools
import importlib
import inspect
import json
from pathlib import Path
from pkgutil import iter_modules
from typing import Any


def features():
    return [importlib.import_module(f"{__package__}.{name}") for name in sorted(
        m.name for m in iter_modules([str(Path(__file__).parent)]) if m.name.startswith("analysis_feature_"))]


def catalog() -> list[dict[str, Any]]:
    return [{"name": m.TOOL.__name__, "command": m.COMMAND, "cli": m.CLI,
             "description": inspect.getdoc(m.TOOL), "parameters": list(inspect.signature(m.TOOL).parameters), "example": getattr(m, "EXAMPLE", {})} for m in features()]


def register_mcp(mcp):
    for module in features():
        mcp.tool()(module.TOOL)


def install_mcp(module):
    if getattr(module, "_analysis_api_installed", False):
        return
    module._analysis_api_installed = True
    original = module.build_server

    @functools.wraps(original)
    def build_server(*args, **kwargs):
        mcp = original(*args, **kwargs)
        register_mcp(mcp)
        return mcp

    module.build_server = build_server


def _arguments(args) -> dict[str, Any]:
    while isinstance(args, (tuple, list)) and len(args) == 1:
        args = args[0]
    if not isinstance(args, dict):
        raise ValueError("Expected one JSON object")
    return dict(args)


def _buffers(core, uri: str) -> tuple[str, dict[str, str]]:
    with core._state_lock:
        buffers = dict(core._document_sources)
    if uri not in buffers:
        buffers[uri] = core.server.workspace.get_text_document(uri).source
    # Materialize the existing editor index before handing a frozen map to the
    # supplied-only service. The service itself never falls back to disk.
    from .workspace import _normalize_uri
    known = {_normalize_uri(name) for name in buffers}
    includes = core._workspace_index.resolve_all_includes(uri)
    for key in includes:
        source = core._workspace_index.get_source(key)
        if source is not None and _normalize_uri(key) not in known:
            buffers[key] = source
    return buffers[uri], buffers


def register_lsp(core):
    from .analysis_service import json_safe
    cache: dict[str, dict[str, Any]] = {}

    def invoke(function, args):
        values = _arguments(args)
        uri = values.pop("uri", None)
        if uri is None:
            return json_safe(function(**values))
        if not isinstance(uri, str):
            raise ValueError("uri must be a string")
        signature = inspect.signature(function).parameters
        if not {"file_content", "active_file", "files"} & set(signature):
            return json_safe(function(**values))
        content, buffers = _buffers(core, uri)
        if "file_content" in signature:
            values["file_content"] = content
        if "active_file" in signature:
            values["active_file"] = uri
        if "files" in signature:
            values["files"] = buffers
        # Compare all participating editor/index buffers after computation.
        result = function(**values)
        current_content, current_buffers = _buffers(core, uri)
        fresh = content == current_content and buffers == current_buffers
        if isinstance(result, dict):
            result["freshness"] = "current" if fresh else "stale"
            if not fresh:
                result["checks_passed"] = False
        if function.__name__ == "dynare_analysis_report" and fresh:
            cache[uri] = {"buffers": buffers, "report": result}
        return json_safe(result)

    def register(module):
        @core.server.command(module.COMMAND)
        @core.server.thread()
        def command(*args):
            try:
                return invoke(module.TOOL, args)
            except Exception as exc:
                return {"success": False, "error": type(exc).__name__, "message": str(exc)}
        return command

    for module in features():
        register(module)

    @core.server.command("dynare/analysisFeatures")
    def analysis_features(*args):
        return catalog()

    @core.server.command("dynare/cachedAnalysisReport")
    def cached_report(*args):
        request = _arguments(args)
        uri = request.get("uri")
        if not isinstance(uri, str):
            raise ValueError("uri must be a string")
        prior = cache.get(uri)
        if prior is None:
            return {"freshness": "not_run", "stages": {}}
        try:
            _content, buffers = _buffers(core, uri)
        except Exception:
            return {"freshness": "stale", "stages": {}}
        report = json.loads(json.dumps(prior["report"]))
        settings_changed = False
        if "config" in request:
            from .analysis_service import effective_config
            try:
                settings_changed = effective_config(request["config"]) != report.get("snapshot", {}).get("configuration")
            except (TypeError, ValueError):
                settings_changed = True
        if prior["buffers"] != buffers or settings_changed:
            report["freshness"] = "stale"
            report["checks_passed"] = False
        return report

    from . import runtime_dispatch
    def drop_cached_report(uri: str) -> None:
        cache.pop(uri, None)
    runtime_dispatch.document_close_callbacks.append(drop_cached_report)
