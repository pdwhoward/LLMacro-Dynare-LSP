"""Read-only environment discovery; never launch a runtime or inspect secrets."""
from __future__ import annotations
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

_DEPENDENCIES = (("numpy", "numpy"), ("scipy", "scipy"), ("sympy", "sympy"),
                 ("pygls", "pygls"), ("lsprotocol", "lsprotocol"), ("mcp", "mcp"))


def _dependency(module: str, distribution: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(module)
        discoverable = spec is not None
    except (ImportError, ValueError, AttributeError):
        discoverable = False
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"module": module, "distribution": distribution, "version": version,
            "status": "discoverable" if discoverable else "not_found", "import_test": "not_run"}


def _preprocessor() -> dict[str, Any]:
    try:
        from .preprocessor import find_preprocessor
        path = find_preprocessor()
        return {"status": "discoverable" if path else "not_found", "path": path, "launch_test": "not_run"}
    except Exception as exc:
        return {"status": "unavailable", "error_type": type(exc).__name__, "path": None, "launch_test": "not_run"}


def _is_executable(path: str) -> bool:
    candidate = Path(path)
    if not candidate.is_file():
        return False
    if os.name == "nt":
        return candidate.suffix.lower() in {".exe", ".com", ".bat", ".cmd"}
    return os.access(candidate, os.X_OK)


def dynare_environment(include_paths: bool = False) -> dict[str, Any]:
    """Discover interpreter, optional dependencies, and runtime configuration.

    This is metadata discovery only: no package import tests, subprocess,
    MATLAB launch, or model execution. Missing dependencies never mean that a
    model passed. Set include_paths=true to reveal selected runtime/interpreter
    paths; unrelated environment variables and credential files are not read.
    """
    if not isinstance(include_paths, bool):
        raise ValueError("include_paths must be boolean")
    from . import __version__
    dependencies = {module: _dependency(module, distribution) for module, distribution in _DEPENDENCIES}
    pp = _preprocessor()
    configured_matlab = os.environ.get("DYNARE_LSP_MATLAB")
    matlab = configured_matlab or shutil.which("matlab")
    matlab_found = bool(matlab and _is_executable(matlab))
    dynare = os.environ.get("DYNARE_LSP_DYNARE")
    dynare_found = bool(dynare and (Path(dynare) / "dynare.m").is_file())
    runtimes = {"preprocessor": pp,
                "matlab": {"status": "discoverable" if matlab_found else "configured_missing" if configured_matlab else "not_found",
                           "path": matlab, "source": "DYNARE_LSP_MATLAB" if configured_matlab else "PATH", "launch_test": "not_run"},
                "dynare_matlab_directory": {"status": "configured" if dynare_found else "configured_missing" if dynare else "not_configured",
                                            "path": dynare, "source": "DYNARE_LSP_DYNARE", "launch_test": "not_run"}}
    if not include_paths:
        for value in runtimes.values():
            value["path"] = None
    def found(name: str) -> bool:
        return dependencies[name]["status"] == "discoverable"
    capabilities = {
        "native_diagnostics": "bundled_not_exercised",
        "lsp_transport": "dependencies_discoverable" if found("pygls") and found("lsprotocol") else "missing_dependencies",
        "mcp_transport": "package_discoverable_api_not_tested" if found("mcp") else "missing_dependencies",
        "numerical_analysis": "dependencies_discoverable" if found("numpy") and found("scipy") else "missing_dependencies",
        "symbolic_analysis": "dependency_discoverable" if found("sympy") else "missing_dependencies",
        "dynare_execution": "runtime_paths_discoverable_not_launched" if matlab_found and dynare_found else "runtime_setup_incomplete",
    }
    return {"schema_version": "dynare-environment/1", "success": True,
            "engine_version": __version__, "python": {"version": platform.python_version(), "implementation": platform.python_implementation(),
                "executable": sys.executable if include_paths else None}, "platform": platform.system(),
            "dependencies": dependencies, "runtimes": runtimes, "capabilities": capabilities,
            "paths_redacted": not include_paths, "execution_performed": False,
            "disclaimer": "Discovery does not verify imports, API compatibility, licenses, or successful runtime/model execution."}


TOOL = dynare_environment
COMMAND = "dynare/environment"
CLI = "environment"
EXAMPLE = {"include_paths": True}
