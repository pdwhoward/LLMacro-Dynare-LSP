"""MCP server exposing the LLMacro Dynare LSP as model-context-protocol tools.

This module wraps the same Python analysis backend used by ``dynare_lsp.server``
(the LSP) and re-publishes it as a Model Context Protocol server. The two
servers share the underlying parser/diagnostics/solver/explain modules so
behavior is identical; only the transport differs.

Why expose the same backend twice?

- The LSP transport requires an LSP-aware client (editor extension, Claude
  Code, Cursor, Cline). It carries stateful document lifecycle messages
  (``didOpen`` / ``didChange``) and is designed for the editing inner loop.
- The MCP transport is stateless tool calls — well-suited to AI clients
  that don't have a notion of an "open document," such as Claude desktop,
  the Anthropic Console, and the OpenAI Agent SDK. Each call accepts the
  file content directly and returns a structured result.

Usage::

    pip install dynare-lsp[mcp]
    dynare-mcp                          # stdio transport
    python -m dynare_lsp.mcp_server     # equivalent

Then point an MCP client at the resulting process. The tools registered
below appear in the client's tool catalog.

If the ``mcp`` package is not installed, importing this module is safe —
it provides a clear install message rather than failing on import — so the
broader ``dynare_lsp`` package stays usable for users who don't need MCP.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from .parser import _iter_equation_tag_spans

try:
    from mcp.server.fastmcp import FastMCP

    _MCP_AVAILABLE = True
except ImportError:
    FastMCP = None  # type: ignore[assignment]
    _MCP_AVAILABLE = False


_INSTALL_MESSAGE = (
    "The 'mcp' package is required to run the MCP server. "
    "Install with:  pip install dynare-lsp[mcp]"
)

string_literal = re.compile(r"\"[^\"\n]*\"|'[^'\n]*'")
_STRING_LITERAL_RE = string_literal
_MCP_TAG_VALUE_RE = re.compile(r"\bmcp\s*=\s*(['\"])(.*?)\1", re.IGNORECASE)
_TAG_ATTRIBUTE_KEY_RE = re.compile(r"\b(?:name|mcp)\b(?=\s*=)", re.IGNORECASE)
_ON_THE_FLY_KIND_MARKERS = frozenset({"e", "x", "p"})


def _is_on_the_fly_kind_marker(line: str, start: int, end: int) -> bool:
    return (
        start > 0
        and line[start - 1] == "|"
        and line[start:end].lower() in _ON_THE_FLY_KIND_MARKERS
    )


def _on_the_fly_marker_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        for match in re.finditer(r"\|([exp])\b", body, re.IGNORECASE):
            spans.append((offset + match.start(1), offset + match.end(1)))
        offset += len(line)
    return spans


def _inside_quoted_string(text: str, offset: int) -> bool:
    quote: Optional[str] = None
    for ch in text[:offset]:
        if quote is not None:
            if ch == quote:
                quote = None
            continue
        if ch in ('"', "'"):
            quote = ch
    return quote is not None


def _mcp_value_ranges(text: str) -> List[Tuple[int, int]]:
    return [
        (m.start(2), m.end(2))
        for m in _MCP_TAG_VALUE_RE.finditer(text)
        if not _inside_quoted_string(text, m.start())
    ]


def _non_mcp_string_spans(text: str) -> List[Tuple[int, int]]:
    """Return string spans to protect, leaving MCP expressions editable."""
    value_ranges = _mcp_value_ranges(text)
    spans: List[Tuple[int, int]] = []
    for match in _STRING_LITERAL_RE.finditer(text):
        covered = [
            (start, end)
            for start, end in value_ranges
            if match.start() <= start <= end <= match.end()
        ]
        if not covered:
            spans.append((match.start(), match.end()))
            continue
        cursor = match.start()
        for start, end in covered:
            if cursor < start:
                spans.append((cursor, start))
            cursor = end
        if cursor < match.end():
            spans.append((cursor, match.end()))
    return spans


def _mask_strings_preserving_mcp_values(text: str) -> str:
    chars = list(text)
    for start, end in _non_mcp_string_spans(text):
        for idx in range(start, end):
            if chars[idx] != "\n":
                chars[idx] = " "
    for tag_start, tag_end in _iter_equation_tag_spans(text):
        tag_text = text[tag_start:tag_end]
        for key in _TAG_ATTRIBUTE_KEY_RE.finditer(tag_text):
            start = tag_start + key.start()
            end = tag_start + key.end()
            for idx in range(start, end):
                chars[idx] = " "
    return "".join(chars)


def _tag_attribute_key_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for tag_start, tag_end in _iter_equation_tag_spans(text):
        tag_text = text[tag_start:tag_end]
        for key in _TAG_ATTRIBUTE_KEY_RE.finditer(tag_text):
            spans.append((tag_start + key.start(), tag_start + key.end()))
    return spans


def _workspace_scope_files(
    active_file: str,
    files: Dict[str, str],
    index,
) -> Dict[str, str]:
    """Return files related to *active_file* through the include graph.

    Scope mirrors the LSP reference/rename helper: active file, files
    transitively included by it, and supplied parent files whose own include
    graph reaches the active file.  Unrelated files in the caller's map are
    excluded so workspace rename cannot rewrite separate models that happen
    to use the same local identifiers.
    """
    from .workspace import _normalize_uri

    active_key = _normalize_uri(active_file)
    supplied_by_key = {
        _normalize_uri(fname): (fname, content) for fname, content in files.items()
    }
    scope: Dict[str, str] = {}
    seen: set = set()

    def _add_file(fname: str, content: Optional[str]) -> None:
        if content is None:
            return
        key = _normalize_uri(fname)
        if key in seen:
            return
        seen.add(key)
        scope[fname] = content

    def _add_path(path_key: str) -> None:
        if path_key in supplied_by_key:
            fname, content = supplied_by_key[path_key]
            _add_file(fname, content)
            return
        _add_file(str(path_key), index.get_source(path_key))

    _add_file(active_file, files.get(active_file))

    active_includes = index.resolve_all_includes(active_file)
    for path in active_includes.keys():
        _add_path(path)

    for fname, content in files.items():
        if _normalize_uri(fname) == active_key:
            continue
        included = index.resolve_all_includes(fname)
        if active_key not in included:
            continue
        _add_file(fname, content)
        for path in included.keys():
            _add_path(path)

    return scope


def _rebase_relative_file_keys(
    active_file: Optional[str],
    files: Optional[Dict[str, str]],
) -> Optional[Dict[str, str]]:
    """Anchor relative filenames in *files* to the active file's directory.

    Virtual relative keys ("params.mod") used to normalize against the
    process CWD, which made resolution cwd-dependent and let a same-named
    DISK sibling of the active file shadow the supplied content.  Anchoring
    them to the active file keeps the documented map-first contract.
    """
    import os
    from pathlib import PurePosixPath, PureWindowsPath

    from .workspace import _normalize_uri

    if not files or not active_file:
        return files
    # Only rebase when the ACTIVE file is absolute: a fully-relative
    # workspace map ("main.mod" + "sub/eqs.inc") already coheres against
    # one implicit root, and prefixing it with the active file's own
    # subdirectory would corrupt the keys.
    active_is_absolute = (
        active_file.startswith("file:")
        or PureWindowsPath(active_file).is_absolute()
        or PurePosixPath(active_file).is_absolute()
    )
    if not active_is_absolute:
        return files
    base = os.path.dirname(_normalize_uri(active_file))
    if not base:
        return files
    rebased: Dict[str, str] = {}
    for fname, content in files.items():
        is_absolute = (
            fname.startswith("file:")
            or PureWindowsPath(fname).is_absolute()
            or PurePosixPath(fname).is_absolute()
        )
        rebased[fname if is_absolute else os.path.join(base, fname)] = content
    return rebased


def _diagnostic_to_dict(d) -> Dict[str, Any]:
    """Convert a dynare_lsp.diagnostics.Diagnostic into a JSON-friendly dict."""
    return {
        "line": d.range.start.line + 1,
        "column": d.range.start.character + 1,
        "end_line": d.range.end.line + 1,
        "end_column": d.range.end.character + 1,
        "severity": d.severity.name,
        "code": d.code,
        "message": d.message,
    }


def build_server():
    """Construct and return the FastMCP server with all tools registered.

    Raises ``ImportError`` (with a clear install message) if ``mcp`` is not
    available. Otherwise returns a ready-to-run ``FastMCP`` instance.
    """
    if not _MCP_AVAILABLE:
        raise ImportError(_INSTALL_MESSAGE)
    assert FastMCP is not None

    # Import the underlying analysis surface only when actually building
    # the server. This keeps import-time side effects minimal and lets the
    # module load cleanly even when scipy / sympy / pygls are absent.
    from .parser import Position, SourceRange, parse
    from .diagnostics import (
        Diagnostic,
        Severity,
        auto_fix as _auto_fix,
        model_with_include_context,
        run_diagnostics,
        _reserved_identifier_reason,
        _with_model_editing_commands,
    )
    from . import explain as _explain_module
    from .workspace import WorkspaceIndex
    from .workspace import _normalize_uri as _workspace_normalize_uri
    from .server import (
        _collect_cross_file_references,
        _find_model_local_occurrences,
        _find_symbol_occurrences_in_model_source,
        _model_local_variables,
    )

    mcp = FastMCP("dynare-lsp")

    def _resolve_active_file_key(
        active_file: str,
        files: Dict[str, str],
        require_present: bool,
    ) -> str:
        if active_file in files:
            return active_file
        active_key = _workspace_normalize_uri(active_file)
        for fname in files:
            if _workspace_normalize_uri(fname) == active_key:
                return fname
        # Case-insensitive fallback: Windows paths are case-insensitive, and
        # virtual (non-existent) paths can't be canonicalized by resolve(),
        # so `C:\X.mod` vs `c:\x.mod` survive normalization as distinct keys.
        # os.path.normcase is the identity on POSIX, so this tier only ever
        # matches where the platform itself treats the paths as equal.
        active_fold = os.path.normcase(active_key)
        for fname in files:
            if os.path.normcase(_workspace_normalize_uri(fname)) == active_fold:
                return fname
        if require_present:
            raise ValueError(f"active_file {active_file!r} must appear in files keys")
        return active_file

    def _run_workspace_preprocessor(entry_file: str, files: Dict[str, str]):
        """Run the preprocessor against a temporary mirror of MCP files."""
        import os
        import shutil
        import tempfile
        from pathlib import Path

        from .preprocessor import (
            find_preprocessor,
            rewrite_supplied_absolute_includes,
            run_preprocessor,
        )

        pp_path = find_preprocessor()
        if not pp_path:
            return None

        # The caller's ``files`` map may omit includes that live on disk
        # next to the real active file — the workspace resolver's disk
        # fallback finds them, so mirror them into the temp tree too or
        # the preprocessor fails with a false "Could not open" on a model
        # the LSP itself accepts.
        index = WorkspaceIndex()
        for fname, content in files.items():
            index.update_document(fname, content)
        try:
            scope = _workspace_scope_files(entry_file, files, index)
        except Exception:
            scope = {}
        supplied_keys = {_workspace_normalize_uri(fname) for fname in files}
        expanded = dict(files)
        for fname, content in scope.items():
            if content is None:
                continue
            if _workspace_normalize_uri(fname) in supplied_keys:
                continue
            expanded[fname] = content
        files = expanded

        normalized = {fname: Path(_workspace_normalize_uri(fname)) for fname in files}
        parents = [str(path.parent) for path in normalized.values()]
        try:
            common_parent = Path(os.path.commonpath(parents))
        except ValueError:
            common_parent = normalized[entry_file].parent

        def _relative_path(fname: str) -> Path:
            path = normalized[fname]
            try:
                rel = path.relative_to(common_parent)
            except ValueError:
                rel = Path(path.name)
            if rel.is_absolute() or any(part == ".." for part in rel.parts):
                return Path(path.name)
            return rel

        tmp_root = Path(tempfile.mkdtemp(prefix="dynare_lsp_mcp_"))
        try:
            entry_parent = tmp_root / _relative_path(entry_file).parent
            entry_parent.mkdir(parents=True, exist_ok=True)
            # Plan all targets first so absolute @#include directives that
            # point at a SUPPLIED path can be rewritten to the mirror copy —
            # otherwise the preprocessor reads the real on-disk file and the
            # supplied (possibly edited) content is silently ignored.
            planned: List[Tuple[str, Path]] = []
            target_by_norm: Dict[str, str] = {}
            for fname in files:
                target = tmp_root / _relative_path(fname)
                planned.append((fname, target))
                target_by_norm[os.path.normcase(str(normalized[fname]))] = str(target)
            rewritten = {
                fname: rewrite_supplied_absolute_includes(files[fname], target_by_norm)
                for fname, _target in planned
            }

            basename_counts: Dict[str, int] = {}
            materialized: List[Tuple[str, Path]] = []
            for fname, target in planned:
                content = rewritten[fname]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                materialized.append((content, target))
                basename_counts[target.name] = basename_counts.get(target.name, 0) + 1

            # The workspace resolver accepts unique virtual basename matches
            # (for example, active "models/main.mod" plus supplied
            # "helper.mod"). Mirror those aliases so preprocessor
            # reconciliation sees the same include graph when possible.
            for content, target in materialized:
                if basename_counts.get(target.name) != 1:
                    continue
                alias = entry_parent / target.name
                if alias.exists():
                    continue
                alias.write_text(content, encoding="utf-8")

            return run_preprocessor(
                rewritten[entry_file],
                pp_path,
                source_dir=str(entry_parent),
            )
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    def _preprocessor_result_for_active_include(preproc_result, active_file: str):
        """Keep a parent preprocessor verdict scoped to the active include."""
        if preproc_result is None or preproc_result.success:
            return preproc_result

        from dataclasses import replace
        from .preprocessor import diagnostic_message_matches_file

        active_key = _workspace_normalize_uri(active_file)
        active_diagnostics = [
            diag
            for diag in preproc_result.diagnostics
            if diagnostic_message_matches_file(diag.message, active_key)
        ]
        if not active_diagnostics:
            return None
        return replace(preproc_result, diagnostics=active_diagnostics)

    def _reconcile_active_include_diagnostics(diags, preproc_result):
        """Reconcile an include fragment without losing helper-only lint."""
        if preproc_result is None:
            return diags
        if preproc_result.success:
            return [diag for diag in diags if diag.code != "E001"] + list(
                preproc_result.diagnostics
            )

        from .preprocessor import reconcile_diagnostics

        return reconcile_diagnostics(diags, preproc_result)

    def _suppress_active_include_false_e010(
        diags,
        parent_file: str,
        index: WorkspaceIndex,
    ):
        """Drop active-fragment E010 when the whole parent model balances."""
        if not any(getattr(diag, "code", None) == "E010" for diag in diags):
            return diags
        parent_model = index.get_effective_model(parent_file) or index.get_model(
            parent_file,
        )
        if parent_model is None:
            return diags
        parent_include_models = index.resolve_all_includes(parent_file)
        parent_diags = run_diagnostics(
            parent_model,
            include_symbols=index.collect_symbols(parent_file),
            include_models=list(parent_include_models.values()),
            include_cycles=index.find_circular_includes(parent_file),
            unresolved_includes=index.find_unresolved_includes(parent_file),
        )
        if any(getattr(diag, "code", None) == "E010" for diag in parent_diags):
            return diags
        return [diag for diag in diags if getattr(diag, "code", None) != "E010"]

    def _workspace_reference_ranges(
        active_file: str,
        symbol: str,
        files: Dict[str, str],
        index: WorkspaceIndex,
    ) -> List[Tuple[str, Any]]:
        active_model = index.get_effective_model(active_file) or index.get_model(
            active_file,
        )
        if active_model is not None and symbol in _model_local_variables(active_model):
            return [
                (active_file, rng)
                for rng in _find_model_local_occurrences(active_model, symbol)
            ]
        open_sources = {
            fname: content for fname, content in files.items() if fname != active_file
        }
        refs = _collect_cross_file_references(
            active_file,
            files[active_file],
            symbol,
            index,
            open_sources,
        )
        supplied_by_key = {_workspace_normalize_uri(fname): fname for fname in files}
        out: List[Tuple[str, Any]] = []
        for uri, rng in refs:
            fname = supplied_by_key.get(_workspace_normalize_uri(uri), uri)
            out.append((fname, rng))
        return out

    def _single_file_reference_ranges(file_content: str, symbol: str) -> List[Any]:
        model = parse(file_content)
        if symbol in _model_local_variables(model):
            return _find_model_local_occurrences(model, symbol)
        return _find_symbol_occurrences_in_model_source(
            file_content,
            symbol,
            model,
        )

    def _offset_for_position(text: str, pos) -> int:
        if pos.line <= 0:
            first = text.split("\n", 1)[0] if text else ""
            return min(max(pos.character, 0), len(first))
        lines = text.split("\n")
        if pos.line >= len(lines):
            return len(text)
        offset = sum(len(line) + 1 for line in lines[: pos.line])
        return min(offset + max(pos.character, 0), offset + len(lines[pos.line]))

    def _source_slice(text: str, rng) -> str:
        start = _offset_for_position(text, rng.start)
        end = _offset_for_position(text, rng.end)
        return text[start:end]

    def _rewrite_ranges(text: str, ranges: List[Any], new_text: str) -> str:
        out = text
        for rng in sorted(
            ranges,
            key=lambda r: (
                _offset_for_position(text, r.start),
                _offset_for_position(text, r.end),
            ),
            reverse=True,
        ):
            start = _offset_for_position(out, rng.start)
            end = _offset_for_position(out, rng.end)
            out = out[:start] + new_text + out[end:]
        return out

    def _parse_with_workspace_context(
        file_content: str,
        active_file: Optional[str] = None,
        files: Optional[Dict[str, str]] = None,
        *,
        whole_model: bool = False,
    ):
        if not active_file or not files:
            return _with_model_editing_commands(parse(file_content))
        files = _rebase_relative_file_keys(active_file, files) or files
        active_file = _resolve_active_file_key(
            active_file,
            files,
            require_present=False,
        )
        workspace_files = dict(files)
        workspace_files[active_file] = file_content
        index = WorkspaceIndex()
        for fname, content in workspace_files.items():
            index.update_document(fname, content)
        parent_context, _ambiguous_parents = _select_active_parent_context(
            active_file,
            workspace_files,
            index,
        )
        if parent_context is not None:
            model, parent_file, _parent_model, _included_model = parent_context
            if not whole_model:
                return _with_model_editing_commands(model)
            parent_file = (
                _outermost_parent_file(active_file, workspace_files, index)
                or parent_file
            )
            parent_model = index.get_effective_model(parent_file)
            if parent_model is None and parent_file in workspace_files:
                parent_model = parse(workspace_files[parent_file])
            if parent_model is None:
                return _with_model_editing_commands(model)
            include_models = index.resolve_all_includes(parent_file)
            return _with_model_editing_commands(
                model_with_include_context(parent_model, list(include_models.values())),
            )
        include_models = index.resolve_all_includes(active_file)
        model = index.get_effective_model(active_file) or parse(
            workspace_files[active_file]
        )
        return _with_model_editing_commands(
            model_with_include_context(model, list(include_models.values())),
        )

    def _steady_state_failure(message: str) -> Dict[str, Any]:
        return {
            "success": False,
            "timed_out": False,
            "message": message,
            "method_used": None,
            "residual_norm": None,
            "values": {},
            "n_symbolic": None,
            "n_numerical": None,
        }

    def _bk_failure(message: str) -> Dict[str, Any]:
        return {
            "status": None,
            "satisfied": None,
            "n_unstable": None,
            "n_forward": None,
            "message": message,
            "forward_variables": [],
            "predetermined_variables": [],
        }

    def _residuals_failure(
        message: str,
        values_used: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "message": message,
            "values_used": dict(values_used or {}),
            "max_abs_residual": None,
            "residuals": [],
        }

    def _identification_failure(message: str) -> Dict[str, Any]:
        return {
            "identified": None,
            "n_findings": 0,
            "findings": [],
            "message": message,
        }

    def _strip_include_instance_suffix(path_key: str) -> str:
        base, sep, suffix = path_key.rpartition("#")
        if sep and suffix.isdigit():
            return base
        return path_key

    def _outermost_parent_files(
        active_file: str,
        files: Dict[str, str],
        index: WorkspaceIndex,
    ) -> List[str]:
        active_key = _workspace_normalize_uri(active_file)
        include_keys_by_parent: Dict[str, set] = {}
        candidate_files: List[str] = []
        for parent_file in files:
            parent_key = _workspace_normalize_uri(parent_file)
            if parent_key == active_key:
                continue
            try:
                included = index.resolve_all_includes(parent_file)
            except Exception:
                continue
            include_keys = {
                _workspace_normalize_uri(_strip_include_instance_suffix(include_key))
                for include_key in included
            }
            include_keys_by_parent[parent_file] = include_keys
            if active_key in include_keys:
                candidate_files.append(parent_file)

        outermost: List[str] = []
        for parent_file in candidate_files:
            parent_key = _workspace_normalize_uri(parent_file)
            if not any(
                parent_key in include_keys_by_parent.get(other_file, set())
                for other_file in candidate_files
                if other_file != parent_file
            ):
                outermost.append(parent_file)
        return outermost

    def _outermost_parent_file(
        active_file: str,
        files: Dict[str, str],
        index: WorkspaceIndex,
    ) -> Optional[str]:
        candidates = _outermost_parent_files(active_file, files, index)
        return candidates[0] if len(candidates) == 1 else None

    def _active_parent_context_candidates(
        active_file: str,
        files: Dict[str, str],
        index: WorkspaceIndex,
    ):
        active_key = _workspace_normalize_uri(active_file)
        contexts = []
        for parent_file, _content in files.items():
            if _workspace_normalize_uri(parent_file) == active_key:
                continue
            try:
                included = index.resolve_all_includes(parent_file)
            except Exception:
                continue
            matched_key = None
            matched_model = None
            for include_key, included_model in included.items():
                real_key = _strip_include_instance_suffix(include_key)
                if _workspace_normalize_uri(real_key) == active_key:
                    matched_key = real_key
                    matched_model = included_model
                    break
            if matched_key is None or matched_model is None:
                continue
            parent_model = index.get_effective_model(parent_file) or index.get_model(
                parent_file,
            )
            if parent_model is None:
                continue
            context = None
            anchor_range = (
                matched_model.include_anchor_range or matched_model.model_block_range
            )
            if anchor_range is None:
                anchor_range = (
                    matched_model.steady_state_block_range
                    or matched_model.initval_block_range
                    or matched_model.endval_block_range
                    or matched_model.shocks_block_range
                )
            for (
                directive,
                resolved_key,
                include_context,
            ) in index.resolve_direct_includes(
                parent_file,
            ):
                if _workspace_normalize_uri(resolved_key) == active_key:
                    context = include_context
                    anchor_range = directive.range
                    break
            else:
                if matched_model.model_equations or matched_model.model_block_range:
                    context = "model"
                elif (
                    matched_model.steady_state_equations
                    or matched_model.steady_state_block_range
                ):
                    context = "steady_state_model"
                elif matched_model.initval_entries or matched_model.initval_block_range:
                    context = "initval"
                elif matched_model.endval_entries or matched_model.endval_block_range:
                    context = "endval"
                elif matched_model.shocks_block_range or matched_model.shocks_vars:
                    context = "shocks"
            if context is None:
                continue
            if anchor_range is None:
                anchor_range = next(
                    (
                        include_model.model_block_range
                        or include_model.steady_state_block_range
                        or include_model.initval_block_range
                        or include_model.endval_block_range
                        or include_model.shocks_block_range
                        or include_model.include_anchor_range
                        for include_model in included.values()
                        if (
                            include_model.model_block_range
                            or include_model.steady_state_block_range
                            or include_model.initval_block_range
                            or include_model.endval_block_range
                            or include_model.shocks_block_range
                            or include_model.include_anchor_range
                        )
                    ),
                    parent_model.model_block_range
                    or parent_model.steady_state_block_range
                    or parent_model.initval_block_range
                    or parent_model.endval_block_range
                    or parent_model.shocks_block_range,
                )
            if anchor_range is None:
                continue
            contextual = index._contextualize_include(  # noqa: SLF001
                parse(files[active_file]),
                files.get(active_file),
                context,
                anchor_range,
                preserve_ranges=True,
            )
            sibling_models = [
                include_model
                for include_key, include_model in included.items()
                if _workspace_normalize_uri(
                    _strip_include_instance_suffix(include_key),
                )
                != active_key
            ]
            declaration_context = model_with_include_context(
                contextual,
                sibling_models + [parent_model],
                include_model_equations=False,
            )
            contexts.append(
                (declaration_context, parent_file, parent_model, matched_model)
            )
        return contexts

    def _select_active_parent_context(
        active_file: str,
        files: Dict[str, str],
        index: WorkspaceIndex,
    ):
        contexts = _active_parent_context_candidates(active_file, files, index)
        if not contexts:
            return None, []
        if len(contexts) == 1:
            return contexts[0], []

        outermost = _outermost_parent_files(active_file, files, index)
        if len(outermost) == 1:
            selected_parent = outermost[0]
            for context in contexts:
                if context[1] == selected_parent:
                    return context, []

        return None, [context[1] for context in contexts]

    def _ambiguous_parent_diagnostic(parent_files: List[str]):
        parent_list = ", ".join(sorted(parent_files))
        return Diagnostic(
            range=SourceRange(Position(0, 0), Position(0, 1)),
            severity=Severity.WARNING,
            message=(
                "Active include is reachable from multiple parent files "
                f"({parent_list}); rerun with only the intended parent in "
                "files so include-scoped diagnostics use the right model context."
            ),
            source="dynare",
            code="W061",
        )

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    @mcp.tool()
    def dynare_diagnose(file_content: str) -> List[Dict[str, Any]]:
        """Run the full LLMacro diagnostic suite on a .mod file.

        Includes parser checks, equation-count verification, undeclared-
        identifier detection, duplicate-declaration detection, steady-state
        residual checks, and conventional-bounds checks.

        The result is reconciled against the bundled Dynare preprocessor
        (the authoritative front-end), so it matches what the LSP publishes
        on save: parser false positives are dropped when Dynare accepts the
        model, and the preprocessor's precise errors surface (as ``P###``
        codes) when it rejects. Falls back to the parser-only result if the
        bundled preprocessor binary is unavailable.

        Args:
            file_content: The full text of a Dynare .mod file.

        Returns:
            A list of diagnostic dicts. Each contains ``line``, ``column``,
            ``severity`` (``ERROR``, ``WARNING``, ``INFORMATION``, or
            ``HINT``), ``code`` (e.g. ``E010``, ``W041``, ``P001``), and
            ``message``. Returns an empty list if the file is clean.
        """
        model = parse(file_content)
        diagnostics = run_diagnostics(model)

        # Defer to the bundled Dynare preprocessor (the authoritative
        # front-end) so this tool's diagnostics match what the LSP
        # publishes on save and what ``python -m dynare_lsp --check``
        # prints: if Dynare accepts the model our hard parser errors are
        # dropped as false positives; if it rejects, its precise
        # parse/declaration errors supersede ours.  Degrades gracefully to
        # the parser-only result when the bundled binary is unavailable.
        try:
            from .preprocessor import (
                find_preprocessor,
                reconcile_diagnostics,
                run_preprocessor,
            )

            pp_path = find_preprocessor()
            if pp_path:
                preproc_result = run_preprocessor(file_content, pp_path)
                diagnostics = reconcile_diagnostics(diagnostics, preproc_result)
        except Exception:
            pass

        return [_diagnostic_to_dict(d) for d in diagnostics]

    @mcp.tool()
    def dynare_diagnose_workspace(
        active_file: str,
        files: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Run diagnostics with cross-file ``@#include`` awareness.

        Mirrors ``dynare_diagnose`` but takes a ``{filename: content}``
        dict so the engine can resolve ``@#include`` directives across
        files.  Names declared in transitively-included files are
        treated as visible (no spurious E020), circular include chains
        fire E060, and unresolvable include paths fire E061.

        Args:
            active_file: The filename of the document being analyzed.
                Must be a key in ``files``.  Paths may be relative or
                absolute — the engine canonicalises internally.
            files: Mapping of ``filename -> file content`` for every
                document the LLM has on hand.  Filenames that appear in
                ``@#include`` directives are resolved against this map
                first, then fall back to the active file's directory if
                the path happens to exist on disk.

        Returns:
            A list of diagnostic dicts, same shape as ``dynare_diagnose``.
        """
        files = _rebase_relative_file_keys(active_file, files) or files
        active_file = _resolve_active_file_key(
            active_file,
            files,
            require_present=True,
        )

        index = WorkspaceIndex()
        for fname, content in files.items():
            index.update_document(fname, content)

        parent_context, ambiguous_parent_files = _select_active_parent_context(
            active_file,
            files,
            index,
        )
        include_symbols = index.collect_symbols(active_file)
        include_models = index.resolve_all_includes(active_file)
        include_cycles = index.find_circular_includes(active_file)
        unresolved = index.find_unresolved_includes(active_file)

        model = index.get_effective_model(active_file) or parse(files[active_file])
        preprocessor_entry_file = active_file
        if parent_context is not None:
            model, parent_file, parent_model, _included_model = parent_context
            preprocessor_entry_file = parent_file
            include_symbols = {
                "endogenous": list(parent_model.endogenous),
                "exogenous": list(parent_model.exogenous),
                "parameters": list(parent_model.parameters),
            }
        diags = run_diagnostics(
            model,
            include_symbols=include_symbols,
            include_models=list(include_models.values()),
            include_cycles=include_cycles,
            unresolved_includes=unresolved,
        )
        try:
            from .preprocessor import reconcile_diagnostics

            preproc_result = _run_workspace_preprocessor(preprocessor_entry_file, files)
            if parent_context is not None:
                preproc_result = _preprocessor_result_for_active_include(
                    preproc_result,
                    active_file,
                )
            if parent_context is not None:
                diags = _reconcile_active_include_diagnostics(diags, preproc_result)
            elif preproc_result is not None:
                diags = reconcile_diagnostics(diags, preproc_result)
        except Exception:
            pass
        if parent_context is not None:
            _model, parent_file, _parent_model, _included_model = parent_context
            diags = _suppress_active_include_false_e010(diags, parent_file, index)
        if ambiguous_parent_files:
            diags.append(_ambiguous_parent_diagnostic(ambiguous_parent_files))
        return [_diagnostic_to_dict(d) for d in diags]

    @mcp.tool()
    def dynare_auto_fix(file_content: str) -> str:
        """Apply deterministic safe auto-fixes to a .mod file.

        Adds missing semicolons, inserts missing block closers, and applies
        any other rewrite the LSP knows is unambiguous. Does *not* attempt
        anything that requires guessing — that work belongs to the LLM
        consuming the diagnostics.

        Args:
            file_content: The full text of a Dynare .mod file.

        Returns:
            The patched file content. If no auto-fix applies, the original
            content is returned unchanged.
        """
        return _auto_fix(file_content)

    @mcp.tool()
    def dynare_parse_summary(file_content: str) -> Dict[str, Any]:
        """Parse a .mod file and return a structured outline.

        Useful for the LLM to inspect a model's structure without
        re-implementing the parser. Returns counts and identifier lists for
        each declaration class plus the number of equations in the model
        and steady-state blocks.

        Args:
            file_content: The full text of a Dynare .mod file.

        Returns:
            ``{
                "endogenous": [...],
                "exogenous": [...],
                "parameters": [...],
                "n_model_equations": int,
                "n_steady_state_equations": int,
                "n_initval_entries": int,
                "is_linear": bool,
                "has_model_block": bool,
                "has_steady_state_model_block": bool,
                "has_initval_block": bool,
                "has_shocks_block": bool,
            }``
        """
        model = parse(file_content)
        return {
            "endogenous": [d.name for d in model.endogenous],
            "exogenous": [d.name for d in model.exogenous],
            "parameters": [d.name for d in model.parameters],
            "n_model_equations": len(model.model_equations),
            "n_steady_state_equations": len(model.steady_state_equations),
            "n_initval_entries": len(model.initval_entries),
            "is_linear": model.is_linear,
            "has_model_block": model.model_block_range is not None,
            "has_steady_state_model_block": model.steady_state_block_range is not None,
            "has_initval_block": model.initval_block_range is not None,
            "has_shocks_block": model.shocks_block_range is not None,
        }

    # -----------------------------------------------------------------------
    # Solver
    # -----------------------------------------------------------------------

    @mcp.tool()
    def dynare_compute_steady_state(
        file_content: str,
        active_file: Optional[str] = None,
        files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Compute the deterministic steady state via the multi-strategy solver chain.

        Pipeline: Gauss-Seidel preconditioning → trust-region least squares
        → MINPACK / Levenberg-Marquardt / Broyden / Krylov root finders →
        homotopy continuation → random restarts with domain heuristics.

        Requires ``scipy`` (and ``sympy`` for the symbolic pre-pass) — install
        with ``pip install dynare-lsp[solver]``.

        Args:
            file_content: The full text of a Dynare .mod file.
            active_file: Optional active filename when *files* supplies a
                workspace with includes.
            files: Optional mapping of filename -> file content for
                include-aware solving.

        Returns:
            ``{
                "success": bool,
                "message": str,
                "method_used": str | None,
                "residual_norm": float | None,
                "values": {variable_name: value, ...},
                "n_symbolic": int | None,    # vars solved analytically
                "n_numerical": int | None,   # vars solved numerically
            }``
        """
        try:
            from .solver import compute_steady_state, default_solve_budget
        except ImportError:
            return _steady_state_failure(
                "scipy is required; install with pip install dynare-lsp[solver]",
            )

        model = _parse_with_workspace_context(
            file_content, active_file, files, whole_model=True
        )
        result = compute_steady_state(model, time_budget=default_solve_budget())
        return {
            "success": bool(result.success),
            "timed_out": bool(getattr(result, "timed_out", False)),
            "message": getattr(result, "message", ""),
            "method_used": getattr(result, "method_used", None),
            "residual_norm": getattr(result, "residual_norm", None),
            "values": dict(result.values) if getattr(result, "values", None) else {},
            "n_symbolic": getattr(result, "n_symbolic", None),
            "n_numerical": getattr(result, "n_numerical", None),
        }

    @mcp.tool()
    def dynare_check_blanchard_kahn(
        file_content: str,
        steady_state: Optional[Dict[str, float]] = None,
        active_file: Optional[str] = None,
        files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Run the Blanchard-Kahn determinacy check.

        If ``steady_state`` is omitted, the solver is invoked first to obtain
        it. The check numerically linearizes the model around the steady
        state via finite differences, forms the companion-matrix eigenvalue
        problem, and compares the number of explosive eigenvalues to the
        number of forward-looking variables.

        Args:
            file_content: The full text of a Dynare .mod file.
            steady_state: Optional dict mapping variable name to steady-state
                value. If omitted, the solver is invoked to compute one.
            active_file: Optional active filename when *files* supplies a
                workspace with includes.
            files: Optional mapping of filename -> file content for
                include-aware BK checks.

        Returns:
            ``{
                "status": "determinate" | "indeterminate" | "no_stable_solution" | None,
                "message": str,
                "forward_variables": [...],
                "predetermined_variables": [...],
            }``
        """
        try:
            from .bk_check import check_blanchard_kahn
        except ImportError:
            return _bk_failure(
                "scipy is required; install with pip install dynare-lsp[solver]",
            )

        model = _parse_with_workspace_context(
            file_content, active_file, files, whole_model=True
        )
        if steady_state is None:
            try:
                from .solver import compute_steady_state, default_solve_budget
            except ImportError:
                return _bk_failure(
                    "scipy is required to derive a steady state first",
                )
            ss_result = compute_steady_state(model, time_budget=default_solve_budget())
            if not ss_result.success:
                return _bk_failure(
                    f"Could not derive steady state: {ss_result.message}",
                )
            steady_state = dict(ss_result.values)

        bk = check_blanchard_kahn(model, steady_state)
        # BKResult exposes ``satisfied``/``n_unstable``/``n_forward`` but
        # no ``status`` string.  Translate to the documented enum here
        # so the tool surface matches its docstring.
        message = getattr(bk, "message", "") or ""
        skipped = message.lower().startswith("blanchard-kahn check skipped")
        if skipped:
            status = None
        elif getattr(bk, "satisfied", None) is True:
            status = "determinate"
        elif getattr(bk, "n_unstable", 0) > getattr(bk, "n_forward", 0):
            status = "no_stable_solution"
        elif getattr(bk, "n_unstable", 0) < getattr(bk, "n_forward", 0):
            status = "indeterminate"
        else:
            status = None
        return {
            "status": status,
            "satisfied": getattr(bk, "satisfied", None),
            "n_unstable": getattr(bk, "n_unstable", None),
            "n_forward": getattr(bk, "n_forward", None),
            "message": message,
            "forward_variables": list(getattr(bk, "forward_variables", []) or []),
            "predetermined_variables": list(
                getattr(bk, "predetermined_variables", []) or []
            ),
        }

    # -----------------------------------------------------------------------
    # Symbol queries (analog of LSP textDocument/references and rename)
    # -----------------------------------------------------------------------

    @mcp.tool()
    def dynare_find_references(
        file_content: str,
        symbol: str,
    ) -> List[Dict[str, Any]]:
        """Find every whole-word occurrence of a symbol in a .mod file.

        Comments are masked out, so a name inside ``//`` or ``/* */`` is not
        treated as a real use. Use this before renaming to verify the scope
        of the change.

        Args:
            file_content: The full text of a Dynare .mod file.
            symbol: The identifier to search for (whole-word match).

        Returns:
            List of ``{"line": int, "column": int, "end_column": int}``
            entries (1-based line and column numbers).
        """
        if not symbol:
            return []
        return [
            {
                "line": rng.start.line + 1,
                "column": rng.start.character + 1,
                "end_column": rng.end.character + 1,
            }
            for rng in _single_file_reference_ranges(file_content, symbol)
        ]

    @mcp.tool()
    def dynare_rename(
        file_content: str,
        old_name: str,
        new_name: str,
    ) -> str:
        """Rename a symbol throughout a .mod file.

        Replaces every whole-word occurrence of ``old_name`` with
        ``new_name``, skipping matches inside comments. Validates that
        ``new_name`` is a legal Dynare identifier before applying.

        Args:
            file_content: The full text of a Dynare .mod file.
            old_name: The identifier to rename.
            new_name: The replacement identifier.

        Returns:
            The rewritten file content. If ``new_name`` is invalid or
            ``old_name`` does not appear, the original content is returned.
        """
        import re as _re

        if not old_name or not new_name:
            return file_content
        # Dynare's NAME token allows a leading underscore.
        if not _re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", old_name):
            return file_content
        if not _re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", new_name):
            return file_content
        if _reserved_identifier_reason(old_name) is not None:
            return file_content
        if _reserved_identifier_reason(new_name) is not None:
            return file_content

        refs = _single_file_reference_ranges(file_content, old_name)
        if not refs:
            return file_content
        if any(_source_slice(file_content, rng) != old_name for rng in refs):
            return file_content
        return _rewrite_ranges(file_content, refs, new_name)

    @mcp.tool()
    def dynare_find_references_workspace(
        active_file: str,
        symbol: str,
        files: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Find every reference to *symbol* across an ``@#include`` graph.

        Mirrors ``dynare_find_references`` but takes a ``{filename:
        content}`` dict so references in transitively-included files and
        open parent files show up too.  Unrelated files in the mapping are
        skipped.  Comments are masked, names are matched as whole words.

        Args:
            active_file: The filename of the document where the cursor
                is.  Must appear in ``files``.
            symbol: The identifier to search for.
            files: ``filename -> content`` map for every file the
                caller has on hand.

        Returns:
            ``[{"file": str, "line": int, "column": int,
                "end_column": int}, ...]`` - 1-based line and column.
        """
        if not symbol:
            return []
        active_file = _resolve_active_file_key(
            active_file,
            files,
            require_present=True,
        )

        index = WorkspaceIndex()
        for fname, content in files.items():
            index.update_document(fname, content)

        results: List[Dict[str, Any]] = []
        for fname, rng in _workspace_reference_ranges(
            active_file,
            symbol,
            files,
            index,
        ):
            results.append(
                {
                    "file": fname,
                    "line": rng.start.line + 1,
                    "column": rng.start.character + 1,
                    "end_column": rng.end.character + 1,
                }
            )
        return results

    @mcp.tool()
    def dynare_rename_workspace(
        active_file: str,
        old_name: str,
        new_name: str,
        files: Dict[str, str],
    ) -> Dict[str, str]:
        """Rename *old_name* to *new_name* across an ``@#include`` graph.

        Returns a ``{filename: rewritten_content}`` mapping containing
        only the related files whose content actually changed.  Comments
        are preserved.  Unrelated files in the mapping are skipped.
        Returns ``{}`` if ``new_name`` is not a legal Dynare identifier
        or no occurrences are found.

        Args:
            active_file: The filename where the cursor is.
            old_name: Identifier to rename.
            new_name: Replacement identifier.
            files: ``filename -> content`` map for the workspace.
        """
        import re as _re

        if not old_name or not new_name:
            return {}
        # Dynare's NAME token allows a leading underscore.
        if not _re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", old_name):
            return {}
        if not _re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", new_name):
            return {}
        if _reserved_identifier_reason(old_name) is not None:
            return {}
        if _reserved_identifier_reason(new_name) is not None:
            return {}
        active_file = _resolve_active_file_key(
            active_file,
            files,
            require_present=True,
        )

        index = WorkspaceIndex()
        for fname, content in files.items():
            index.update_document(fname, content)

        refs = _workspace_reference_ranges(active_file, old_name, files, index)
        if not refs:
            return {}
        for fname, rng in refs:
            content = files.get(fname)
            if content is None:
                return {}
            lines = content.split("\n")
            if rng.start.line >= len(lines):
                return {}
            source_slice = lines[rng.start.line][
                rng.start.character : rng.end.character
            ]
            if source_slice != old_name:
                return {}

        out: Dict[str, str] = {}
        grouped_refs: Dict[str, List[Any]] = {}
        for fname, rng in refs:
            grouped_refs.setdefault(fname, []).append(rng)
        for fname, ranges in grouped_refs.items():
            content = files.get(fname)
            if content is None:
                return {}
            rewritten = _rewrite_ranges(content, ranges, new_name)
            if rewritten != content:
                out[fname] = rewritten
        return out

    # -----------------------------------------------------------------------
    # Documentation
    # -----------------------------------------------------------------------

    @mcp.tool()
    def dynare_explain(code: str) -> str:
        """Return markdown documentation for a diagnostic code.

        Args:
            code: A diagnostic code such as ``E010`` or ``W041`` (case-insensitive).

        Returns:
            A markdown explanation including title, causes, and fix recipe.
            If the code is unknown, returns a brief error with the list of
            documented codes.
        """
        rendered = _explain_module.render_markdown(code)
        if rendered is None:
            return (
                f"No documentation found for code '{code}'. "
                f"Known codes: {', '.join(_explain_module.known_codes())}"
            )
        return rendered

    @mcp.tool()
    def dynare_list_diagnostic_codes() -> List[Dict[str, str]]:
        """List every documented diagnostic code with its title.

        Useful for the LLM to discover what classes of errors LLMacro
        recognises before processing a specific file.

        Returns:
            A list of ``{"code": "E010", "title": "..."}`` dicts, sorted by
            code.
        """
        out: List[Dict[str, str]] = []
        for c in _explain_module.known_codes():
            entry = _explain_module.explain(c)
            out.append(
                {
                    "code": c,
                    "title": entry["title"] if entry else "",
                }
            )
        return out

    # -----------------------------------------------------------------------
    # Model diff
    # -----------------------------------------------------------------------

    @mcp.tool()
    def dynare_compare_models(
        file_content_a: str,
        file_content_b: str,
        active_file_a: Optional[str] = None,
        active_file_b: Optional[str] = None,
        files_a: Optional[Dict[str, str]] = None,
        files_b: Optional[Dict[str, str]] = None,
        files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Compare two Dynare model files structurally and return the diff.

        Useful for an LLM iterating on model variants — pass the prior and
        current file contents and receive a structured summary of what changed
        (added/removed variables, parameter calibration changes, added/removed
        equations).

        Args:
            file_content_a: The full text of the "before" .mod file.
            file_content_b: The full text of the "after" .mod file.
            active_file_a: Optional active filename for include-aware parsing.
            active_file_b: Optional active filename for include-aware parsing.
            files_a: Optional "before" workspace mapping for includes.
            files_b: Optional "after" workspace mapping for includes.
            files: Optional common workspace mapping used for both inputs.

        Returns:
            ModelDiff.to_dict() - JSON-friendly structural diff.
        """
        from .model_diff import compare_models as _compare_models

        def _context_model(content, active_file, file_map):
            # A workspace mapping without an active filename used to be
            # silently ignored (include context never engaged); synthesize
            # a unique virtual key so ``files`` works on its own.
            if file_map and not active_file:
                candidate = "__mcp_compare__.mod"
                suffix = 1
                while candidate in file_map:
                    suffix += 1
                    candidate = f"__mcp_compare_{suffix}__.mod"
                active_file = candidate
            return _parse_with_workspace_context(content, active_file, file_map)

        model_a = _context_model(file_content_a, active_file_a, files_a or files)
        model_b = _context_model(file_content_b, active_file_b, files_b or files)
        return _compare_models(model_a, model_b).to_dict()

    # -----------------------------------------------------------------------
    # Structure / analysis
    # -----------------------------------------------------------------------

    @mcp.tool()
    def dynare_model_info(
        file_content: str,
        active_file: Optional[str] = None,
        files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Summarise model structure (mirrors Dynare's ``model_info``).

        Classifies each endogenous variable by dynamic timing (static /
        predetermined / forward-looking / mixed), reports the Blanchard-Kahn
        state-space dimensions (state variables and jumpers), and describes the
        recursive block structure of the static model.  No solving required.

        Returns a dict with ``n_endogenous``/``endogenous`` (and the same for
        exogenous/parameters), ``n_equations``, the four timing lists with
        counts, ``n_state_variables``/``n_jumpers``, and ``blocks`` (number of
        block-triangular blocks, their sizes, and whether the model is
        recursive) or null when the structure could not be determined.
        """
        from .model_info import compute_model_info

        model = _parse_with_workspace_context(
            file_content, active_file, files, whole_model=True
        )
        return compute_model_info(model)

    @mcp.tool()
    def dynare_residuals(
        file_content: str,
        values: Optional[Dict[str, float]] = None,
        active_file: Optional[str] = None,
        files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate each steady-state equation's residual (Dynare's ``resid``).

        Shows which equations do not hold, and by how much, at the supplied
        values (or, if omitted, the ``initval`` values, with anything missing
        defaulting to zero).  Useful for debugging a candidate steady state.

        Args:
            file_content: The full text of a Dynare .mod file.
            values: Optional mapping of endogenous variable -> value to evaluate
                the residuals at.  Defaults to initval values (then 0).
            active_file / files: Optional include-aware workspace context.

        Returns ``{"success", "values_used", "max_abs_residual", "residuals":
        [{"equation", "text", "residual"}, ...]}``.
        """
        try:
            import math

            import numpy as np

            from .solver import _build_residual_function, _effective_param_values
        except ImportError:
            return _residuals_failure(
                "numpy is required; install with pip install dynare-lsp[solver]",
            )

        model = _parse_with_workspace_context(
            file_content, active_file, files, whole_model=True
        )
        var_names = [v.name for v in model.endogenous]
        params = _effective_param_values(model)
        exogenous = set(model.exogenous_names())

        resolved = dict(model.initval_values())
        if values:
            for name, value in values.items():
                try:
                    resolved[name] = float(value)
                except (TypeError, ValueError):
                    continue
        x = np.array(
            [float(resolved.get(name, 0.0)) for name in var_names], dtype=float
        )

        try:
            residual_fn = _build_residual_function(
                model, var_names, params, exogenous, {}
            )
            residual_vec = list(residual_fn(x))
        except Exception as exc:
            return _residuals_failure(
                f"residual evaluation failed: {exc}",
                {name: float(x[i]) for i, name in enumerate(var_names)},
            )

        equations = model.static_model_equations()
        aligned = len(equations) == len(residual_vec)
        residuals = []
        for i, value in enumerate(residual_vec):
            text = equations[i].text.strip()[:200] if aligned else f"equation {i + 1}"
            residuals.append(
                {
                    "equation": i + 1,
                    "text": text,
                    "residual": float(value),
                }
            )
        finite = [abs(r["residual"]) for r in residuals if math.isfinite(r["residual"])]
        return {
            "success": True,
            "values_used": {name: float(x[i]) for i, name in enumerate(var_names)},
            "max_abs_residual": max(finite) if finite else None,
            "residuals": residuals,
        }

    @mcp.tool()
    def dynare_check_identification(
        file_content: str,
        steady_state: Optional[Dict[str, float]] = None,
        active_file: Optional[str] = None,
        files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """First-order structural identification check (diagnostic code W081).

        Perturbs each calibrated parameter and compares its local effect on the
        steady-state residuals and first-order structure; flags parameters with
        no effect and groups whose effects are collinear.  This is a structural
        check, not a data/moment-based Iskrev test.

        If ``steady_state`` is omitted the solver computes one first.

        Returns ``{"identified": bool|None, "n_findings": int, "findings":
        [{"code", "message"}, ...]}``.
        """
        try:
            from .identification import check_identification
        except ImportError:
            return _identification_failure("numpy required")

        model = _parse_with_workspace_context(
            file_content, active_file, files, whole_model=True
        )
        if steady_state is None:
            try:
                from .solver import compute_steady_state, default_solve_budget

                ss_result = compute_steady_state(
                    model, time_budget=default_solve_budget()
                )
                if ss_result.success:
                    steady_state = dict(ss_result.values)
                else:
                    # Solver failed or timed out — evaluating identification at an
                    # invalid (or missing) steady state would produce unreliable results.
                    timed_out = getattr(ss_result, "timed_out", False)
                    reason = (
                        f"identification inconclusive: steady-state solve timed out "
                        f"({ss_result.message})"
                        if timed_out
                        else f"identification inconclusive: steady-state solve failed "
                        f"({ss_result.message})"
                    )
                    return {
                        "identified": None,
                        "n_findings": 0,
                        "findings": [],
                        "message": reason,
                    }
            except ImportError:
                return _identification_failure(
                    "identification inconclusive: steady-state solver unavailable "
                    "(install with pip install dynare-lsp[solver])",
                )

        diagnostics = check_identification(model, steady_state or {})
        findings = [{"code": d.code, "message": d.message} for d in diagnostics]
        return {
            "identified": len(findings) == 0,
            "n_findings": len(findings),
            "findings": findings,
        }

    @mcp.tool()
    def dynare_list_options(command: Optional[str] = None) -> Dict[str, Any]:
        """List the valid options for a Dynare command (anti-hallucination aid).

        The catalog is generated from the Dynare 7.1 grammar, so it is the set
        of options the bundled preprocessor actually accepts.

        Args:
            command: A command name (e.g. ``"stoch_simul"``).  If omitted, the
                list of known commands is returned instead.

        Returns ``{"command", "known": True, "n_options", "options":
        [{"name", "description"}, ...]}`` for a known command;
        ``{"command", "known": False, "message", "suggestions": [...]}``
        for an unknown one; or ``{"n_commands", "commands": [...]}`` when
        *command* is omitted.
        """
        from .dynare_commands import COMMAND_OPTIONS, command_options

        if command:
            key = command.lower()
            if key not in COMMAND_OPTIONS:
                # An unknown command must not look like a real zero-option
                # command, or the "anti-hallucination aid" quietly endorses
                # hallucinated command names.
                import difflib

                suggestions = difflib.get_close_matches(
                    key, COMMAND_OPTIONS.keys(), n=3, cutoff=0.6
                )
                return {
                    "command": key,
                    "known": False,
                    "message": (
                        f"'{key}' is not a Dynare command in the bundled "
                        "7.1 grammar."
                        + (
                            f" Did you mean: {', '.join(suggestions)}?"
                            if suggestions
                            else ""
                        )
                    ),
                    "suggestions": suggestions,
                }
            options = command_options(key)
            return {
                "command": key,
                "known": True,
                "n_options": len(options),
                "options": [
                    {"name": name, "description": desc} for name, desc in options
                ],
            }
        return {
            "n_commands": len(COMMAND_OPTIONS),
            "commands": sorted(COMMAND_OPTIONS.keys()),
        }

    # -----------------------------------------------------------------------
    # Preprocessor execution
    # -----------------------------------------------------------------------

    @mcp.tool()
    def dynare_run_preprocessor(
        file_content: str,
        active_file: Optional[str] = None,
        files: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute the bundled Dynare preprocessor and return its real output.

        Runs the pinned ``dynare-preprocessor`` (Dynare 7.1) shipped under
        ``dynare_lsp/bin/`` on the supplied ``.mod`` text in check mode and
        returns the preprocessor's own verdict — exit code, parsed
        diagnostics, and the raw stdout/stderr streams.  Unlike
        ``dynare_diagnose`` (which runs LLMacro's analysis suite), this is the
        authoritative Dynare front-end speaking for itself, so use it to
        confirm whether Dynare actually accepts a model.

        Args:
            file_content: The full text of a Dynare ``.mod`` file.
            active_file: Optional path or workspace key of the document being
                checked.  With ``files``, the tool materializes the workspace
                map in a private temp directory and overwrites the active file
                with ``file_content`` so unsaved edits are honored.
            files: Optional ``filename -> content`` map for related workspace
                files used by relative ``@#include`` directives.

        Returns:
            On a successful run (the binary executed)::

                {
                    "success": bool,        # True iff preprocessor exit code 0
                    "exit_code": int|None,
                    "diagnostics": [
                        {"line", "column", "severity", "message", "code"}, ...
                    ],
                    "raw_stdout": str,
                    "raw_stderr": str,
                }

            ``severity`` is one of ``ERROR`` / ``WARNING`` (preprocessor
            diagnostics are errors or warnings); ``line``/``column`` are
            1-based.  If the bundled preprocessor binary cannot be located the
            tool degrades gracefully to
            ``{"success": False, "message": "...", "diagnostics": []}``.
        """
        from .preprocessor import run_preprocessor_structured

        if active_file and files:
            files = _rebase_relative_file_keys(active_file, files) or files
            workspace_active = _resolve_active_file_key(
                active_file,
                files,
                require_present=False,
            )
            workspace_files = dict(files)
            workspace_files[workspace_active] = file_content
            result = _run_workspace_preprocessor(workspace_active, workspace_files)
            if result is not None:
                return {
                    "success": bool(result.success),
                    "exit_code": result.exit_code,
                    "diagnostics": [_diagnostic_to_dict(d) for d in result.diagnostics],
                    "raw_stdout": result.stdout,
                    "raw_stderr": result.stderr,
                }

        source_dir: Optional[str] = None
        if active_file:
            from .include_resolver import _uri_to_path

            active_path = _uri_to_path(active_file)
            candidate_path = active_path.parent if active_path.suffix else active_path
            try:
                candidate = str(candidate_path.resolve())
            except (OSError, RuntimeError):
                candidate = str(candidate_path.absolute())
            if candidate_path.is_dir():
                source_dir = candidate

        return run_preprocessor_structured(file_content, source_dir=source_dir)

    # -----------------------------------------------------------------------
    # Full model execution (MATLAB + Dynare)
    # -----------------------------------------------------------------------

    @mcp.tool()
    def dynare_run_dynare(
        file_content: str,
        active_file: Optional[str] = None,
        files: Optional[Dict[str, str]] = None,
        timeout: int = 300,
    ) -> Dict[str, Any]:
        """Execute a ``.mod`` model end-to-end with MATLAB + Dynare.

        WARNING: THIS TOOL EXECUTES CODE.  It launches a headless MATLAB
        session and runs the real Dynare toolbox on the supplied model.  A
        Dynare ``.mod`` file can carry trailing MATLAB statements after the
        model blocks, and those run inside that session — so only call this on
        model text you trust.  Unlike ``dynare_run_preprocessor`` (which only
        runs Dynare's parser/front-end) and ``dynare_compute_steady_state`` /
        ``dynare_check_blanchard_kahn`` (LLMacro's own clean-room numerics),
        this runs the *actual* Dynare solver and reports what Dynare itself
        computes: the steady state, the Blanchard-Kahn eigenvalue verdict, and
        Dynare's own error text when the model fails.

        MATLAB and Dynare are discovered explicitly and are overridable: set
        the ``DYNARE_LSP_MATLAB`` environment variable to a ``matlab``
        executable and ``DYNARE_LSP_DYNARE`` to the Dynare ``matlab/``
        directory (the one containing ``dynare.m``).  If neither can be
        located the tool degrades gracefully (it never raises) and returns
        ``matlab_available: False`` with a message explaining what to set.

        Args:
            file_content: The full text of a Dynare ``.mod`` file (executed).
            active_file: Optional filename for ``file_content``.  When supplied
                with ``files``, the runner materializes those workspace files in
                a private temp directory so relative ``@#include`` siblings can
                resolve while preserving the current unsaved active text.
            files: Optional ``filename -> content`` map for related workspace
                files used by ``@#include``.
            timeout: Seconds before the MATLAB run is abandoned (default 300).
                Heavy models (high-order perturbation, large systems) may need
                more.

        Returns:
            ``{
                "success": bool,            # True iff Dynare ran end-to-end
                "matlab_available": bool,   # False if no MATLAB was found
                "dynare_available": bool,   # False if MATLAB ran but no Dynare
                "status": str,              # success | dynare_error | timeout |
                                            #   matlab_crash | bad_json |
                                            #   no_matlab | no_dynare | ...
                "steady_state": {var: value},   # finite entries only
                "blanchard_kahn": {
                    "satisfied": bool | None,
                    "message": str,
                    "n_explosive": int | None,
                    "n_forward_looking": int | None,
                },
                "errors": [str],            # Dynare's own error text on failure
                "raw_log": str,             # full MATLAB stdout/stderr
            }``

            On the degraded / failure paths a ``message`` key is also present.
        """
        from .matlab_runner import run_dynare_matlab

        workspace_files = None
        workspace_active = active_file
        if active_file and files:
            files = _rebase_relative_file_keys(active_file, files) or files
            workspace_active = _resolve_active_file_key(
                active_file,
                files,
                require_present=False,
            )
            workspace_files = dict(files)
            workspace_files[workspace_active] = file_content

        return run_dynare_matlab(
            file_content,
            timeout=timeout,
            active_file=workspace_active,
            files=workspace_files,
        )

    return mcp


def main() -> None:
    """Entry point: build the server and run it over stdio."""
    if not _MCP_AVAILABLE:
        print(_INSTALL_MESSAGE, file=sys.stderr)
        sys.exit(1)
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
