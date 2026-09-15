"""Explicit project configuration shared across versioned analysis transports."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
from typing import Any
from urllib.parse import urlsplit
from urllib.request import url2pathname
from .analysis_service import analyze, effective_config, fingerprint

CONFIG_NAME = ".dynare-analysis.json"
_ALLOWED = {"schema_version", "entry_file", "macro_defines", "search_paths", "tolerance", "solve_budget", "numerical", "preprocessor"}


def _path(value: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Paths must be nonempty strings without NUL")
    if value.startswith("file:"):
        parsed = urlsplit(value)
        if parsed.query or parsed.fragment:
            raise ValueError("Encode literal query/fragment characters in file URIs")
        value = url2pathname(parsed.path)
        if parsed.netloc and parsed.netloc != "localhost":
            value = f"//{parsed.netloc}{value}"
    elif "://" in value:
        raise ValueError("Only local paths and file URIs are supported")
    return Path(value)


def _within(root: Path, value: str) -> Path:
    path = _path(value)
    target = (path if path.is_absolute() else root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("Configured paths must stay inside the explicit project root") from exc
    return target


def _relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Expected a nonempty relative project path")
    value = value.replace("\\", "/")
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute() or ":" in value:
        raise ValueError("Configuration paths must be relative to project_root")
    return value


def macro_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)) and math.isfinite(value):
        return str(value)
    if isinstance(value, str) and not any(ord(c) < 32 or c in "\\\"'" for c in value):
        return '"' + value + '"'
    raise ValueError("Macro values must be finite numbers, booleans, or strings without quotes, backslashes or control characters")


def validate_configuration(configuration: Any, root: Path) -> dict[str, Any]:
    if not isinstance(configuration, dict) or isinstance(configuration.get("schema_version"), bool) or configuration.get("schema_version") != 1:
        raise ValueError("configuration.schema_version must equal 1")
    unknown = set(configuration) - _ALLOWED
    if unknown:
        raise ValueError(f"Unknown project settings: {', '.join(sorted(unknown))}")
    entry = _within(root, _relative(configuration.get("entry_file")))
    settings = effective_config({k: v for k, v in configuration.items() if k not in {"schema_version", "entry_file", "macro_defines"}})
    settings["search_paths"] = [str(_within(root, _relative(p))) for p in settings["search_paths"]]
    defines = configuration.get("macro_defines", {})
    if not isinstance(defines, dict):
        raise ValueError("macro_defines must be an object")
    for name, value in defines.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("Macro names must be ASCII identifiers")
        macro_literal(value)
    return {"schema_version": 1, "project_root": str(root), "entry_file": str(entry),
            "settings": settings, "macro_defines": dict(sorted(defines.items()))}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def prepare_project(files: dict[str, str], configuration: dict[str, Any] | None, project_root: str) -> dict[str, Any]:
    """Resolve only supplied buffers. No config discovery or source reads on disk."""
    if not isinstance(files, dict):
        raise ValueError("files must be a filename-to-content object")
    root = _path(project_root).resolve()
    normalized, ignored = {}, []
    for name, text in files.items():
        if not isinstance(text, str):
            raise ValueError("Every supplied file must contain text")
        _path(name)  # Reject malformed names rather than silently ignoring them.
        try:
            path = _within(root, name)
        except ValueError:
            ignored.append(name)
            continue
        key = os.path.normcase(str(path))
        if key in normalized and normalized[key] != text:
            raise ValueError(f"Conflicting aliases for {name}")
        normalized[key] = text
    config_key = os.path.normcase(str(root / CONFIG_NAME))
    config_text = normalized.pop(config_key, None)
    source = "explicit_object"
    if configuration is None:
        if config_text is None:
            raise ValueError(f"Supply configuration or {CONFIG_NAME} in files")
        configuration = json.loads(config_text, object_pairs_hook=_unique_object)
        source = "supplied_configuration_file"
    resolved = validate_configuration(configuration, root)
    entry = os.path.normcase(resolved["entry_file"])
    if entry not in normalized:
        raise ValueError("The configured entry model must be present in supplied files")
    prefix = "".join(f"@#define {name} = {macro_literal(value)}\n" for name, value in resolved["macro_defines"].items())
    prepared = dict(normalized)
    original = normalized[entry]
    bom_moved = bool(prefix and original.startswith("\ufeff"))
    prepared[entry] = ("\ufeff" + prefix + original[1:]) if bom_moved else prefix + original
    return {"entry": entry, "raw_files": normalized, "prepared_files": prepared,
            "configuration": resolved, "configuration_source": source,
            "configuration_file_hash": hashlib.sha256(config_text.encode("utf-8")).hexdigest() if config_text is not None else None,
            "project_snapshot_id": fingerprint(normalized, resolved, entry), "ignored_files": ignored,
            "source_transform": {"file": entry, "prefix_lines": len(resolved["macro_defines"]), "bom_moved": bom_moved}}


def original_range(source_range: dict[str, Any], prefix_lines: int, bom_moved: bool) -> dict[str, Any] | None:
    if any(source_range[side]["line"] < prefix_lines for side in ("start", "end")):
        return None
    result = copy.deepcopy(source_range)
    for side in ("start", "end"):
        result[side]["line"] -= prefix_lines
        if bom_moved and result[side]["line"] == 0:
            result[side]["character"] += 1
    return result


def dynare_project_analysis(files: dict[str, str], configuration: dict[str, Any] | None = None,
                            project_root: str = ".") -> dict[str, Any]:
    """Analyze the explicitly configured entry model, even when editing an include.

    Supply configuration as an object or as .dynare-analysis.json in files.
    Config version 1 supports entry_file, search_paths, scalar macro_defines,
    tolerance, solve_budget, numerical and preprocessor. Paths are relative to
    the explicit project_root; only supplied in-root source buffers are used.
    No automatic config search, source writes, or MATLAB/model execution.
    """
    prepared = prepare_project(files, configuration, project_root)
    entry = prepared["entry"]
    report = analyze(prepared["prepared_files"][entry], entry, prepared["prepared_files"], prepared["configuration"]["settings"])
    transform = prepared["source_transform"]
    # Keep the nested analysis contract intact and add a separate original range.
    # Only root-native findings are mapped; other origins are not guessed.
    diagnostics = report["stages"]["diagnostics"]
    if diagnostics.get("source_file") == entry:
        for row in diagnostics.get("findings", []):
            if str(row.get("code", "")).startswith("P"):
                row["original_range"] = None
                row["original_file"] = None
                continue
            row["original_range"] = original_range(row["range"], transform["prefix_lines"], transform["bom_moved"])
            row["original_file"] = entry if row["original_range"] is not None else CONFIG_NAME
    return {"schema_version": "dynare-project/1", "checks_passed": report["checks_passed"],
            "project_snapshot_id": prepared["project_snapshot_id"], "configuration": prepared["configuration"],
            "configuration_source": prepared["configuration_source"], "configuration_file_hash": prepared["configuration_file_hash"],
            "raw_source_hashes": {name: hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest() for name, text in prepared["raw_files"].items()},
            "source_transform": transform, "ignored_files": prepared["ignored_files"], "analysis": report,
            "execution_performed": False, "disclaimer": "Project and transformed-analysis snapshot IDs are distinct. Passing checks does not establish economic fidelity."}


TOOL = dynare_project_analysis
COMMAND = "dynare/projectAnalysis"
CLI = "project"
EXAMPLE = {"project_root": "/path/to/project", "configuration": {"schema_version": 1, "entry_file": "main.mod"}}
