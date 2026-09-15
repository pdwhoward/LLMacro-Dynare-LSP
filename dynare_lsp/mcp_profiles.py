"""Explicit MCP launch profiles; analysis mode never registers MATLAB execution."""
from __future__ import annotations
import argparse
import functools
import inspect
import json
from typing import Any

# Closed allowlist, not a name-prefix inference. New capabilities require review
# before they become available in a restricted server.
ANALYSIS_TOOLS = frozenset({"dynare_analysis_report", "dynare_validate_patch", "dynare_get_context",
    "dynare_numerical_evidence", "dynare_environment", "dynare_project_analysis"})
_TEMP_WRITERS = frozenset({"dynare_analysis_report", "dynare_validate_patch", "dynare_project_analysis"})


def validate_limits(max_request_bytes: int, max_execution_seconds: int) -> None:
    for name, value, low, high in (("max_request_bytes", max_request_bytes, 1024, 64 * 1024 * 1024),
                                  ("max_execution_seconds", max_execution_seconds, 1, 600)):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def bounded_call(function, max_request_bytes: int):
    signature = inspect.signature(function)
    @functools.wraps(function)
    def call(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        data = json.dumps(bound.arguments, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        if len(data.encode("utf-8")) > max_request_bytes:
            raise ValueError("Request exceeds the server's configured byte limit")
        from .analysis_service import json_safe
        return json_safe(function(*args, **kwargs))
    return call


def execution_tool(max_execution_seconds: int):
    def dynare_execute_trusted_model(file_content: str, acknowledge_code_execution: bool = False,
                                     active_file: str | None = None, files: dict[str, str] | None = None,
                                     timeout: int = 300) -> dict[str, Any]:
        """EXECUTES arbitrary trusted model/MATLAB code; not a sandbox.

        Available only in an explicitly execution-enabled server. Acknowledgment
        is required on every call. Timeout is capped by the launch policy.
        Model code may access the account's filesystem/network; private staging
        and this policy do not provide OS isolation. Never call on untrusted text.
        """
        if acknowledge_code_execution is not True:
            raise PermissionError("Explicit code-execution acknowledgment is required")
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= max_execution_seconds:
            raise ValueError(f"timeout must be an integer in [1, {max_execution_seconds}]")
        if not isinstance(file_content, str):
            raise ValueError("file_content must be a string")
        if files:
            if not isinstance(active_file, str) or not active_file:
                raise ValueError("active_file is required for workspace execution")
            from .mcp_server import _rebase_relative_file_keys
            from .workspace import _normalize_uri
            supplied = _rebase_relative_file_keys(active_file, files) or files
            canonical = {}
            for name, text in supplied.items():
                if not isinstance(name, str) or not isinstance(text, str):
                    raise ValueError("files must map paths to source strings")
                key = _normalize_uri(name)
                if key in canonical and canonical[key] != text:
                    raise ValueError("Conflicting workspace path aliases")
                canonical[key] = text
            active_file = _normalize_uri(active_file)
            canonical[active_file] = file_content
            files = canonical
        from .matlab_runner import run_dynare_matlab
        return run_dynare_matlab(file_content, active_file=active_file, files=files, timeout=timeout)
    return dynare_execute_trusted_model


def register_profile(server, profile: str, modules, annotations_type, *,
                     max_request_bytes: int = 8 * 1024 * 1024, max_execution_seconds: int = 300) -> list[str]:
    """Register only allowed analysis functions; execution is a separate branch."""
    if profile not in {"analysis", "execution"}:
        raise ValueError("profile must be analysis or execution")
    validate_limits(max_request_bytes, max_execution_seconds)
    registered = []
    for module in modules:
        function = module.TOOL
        name = function.__name__
        if name not in ANALYSIS_TOOLS:
            continue
        annotations = annotations_type(readOnlyHint=name not in _TEMP_WRITERS,
            destructiveHint=False, idempotentHint=True, openWorldHint=False)
        server.tool(annotations=annotations)(bounded_call(function, max_request_bytes))
        registered.append(name)
    if profile == "execution":
        function = execution_tool(max_execution_seconds)
        annotations = annotations_type(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True)
        server.tool(annotations=annotations)(bounded_call(function, max_request_bytes))
        registered.append(function.__name__)
    return registered


def build_server(profile: str = "analysis", *, max_request_bytes: int = 8 * 1024 * 1024,
                 max_execution_seconds: int = 300):
    if profile not in {"analysis", "execution"}:
        raise ValueError("profile must be analysis or execution")
    validate_limits(max_request_bytes, max_execution_seconds)
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import ToolAnnotations
    except ImportError as exc:
        raise RuntimeError("Install the project's MCP extra with a FastMCP-compatible SDK supporting tool annotations.") from exc
    from .analysis_api import features
    server = FastMCP(f"dynare-{profile}")
    if "annotations" not in inspect.signature(server.tool).parameters:
        raise RuntimeError("This MCP SDK does not support tool annotations; upgrade the compatible MCP extra.")
    registered = register_profile(server, profile, features(), ToolAnnotations,
        max_request_bytes=max_request_bytes, max_execution_seconds=max_execution_seconds)
    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False))
    def dynare_profile_policy() -> dict[str, Any]:
        """Inspect this server's enforced catalog and configured input/execution limits."""
        return {"schema_version": "dynare-profile/1", "profile": profile, "tools": registered,
                "max_request_bytes": max_request_bytes, "max_execution_seconds": max_execution_seconds,
                "source_policy": "supplied-only versioned analysis; no legacy tool registration",
                "matlab_execution_enabled": profile == "execution", "os_sandbox": False}
    return server


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("analysis", "execution"), default="analysis")
    parser.add_argument("--max-request-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--max-execution-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    server = build_server(args.profile, max_request_bytes=args.max_request_bytes,
                          max_execution_seconds=args.max_execution_seconds)
    server.run()


if __name__ == "__main__":
    main()
