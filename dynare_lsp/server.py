"""Dynare Language Server — LSP implementation using pygls.

Provides:
  - Real-time diagnostics on .mod files (errors, warnings)
  - Steady state equation validation
  - Variable reference checking
  - Equation count verification
  - Document symbols (outline of model structure)
  - Go to definition (jump to variable/parameter declarations)
  - Auto-completion (variables, parameters, keywords, built-in functions)
  - Code actions (quick fixes for common diagnostics)
  - Inlay hints (steady state values, parameter values inline)
  - Folding ranges (collapse blocks)
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

try:
    _pygls_server = importlib.import_module("pygls.server")
    LanguageServer = getattr(_pygls_server, "LanguageServer")  # noqa: N816
except (ImportError, AttributeError):
    # pygls < 2.0 exposed LanguageServer from this legacy module path.
    _pygls_server = importlib.import_module("pygls.lsp.server")
    LanguageServer = getattr(_pygls_server, "LanguageServer")  # noqa: N816
from lsprotocol import types as lsp

try:
    from pygls.capabilities import ServerCapabilitiesBuilder
except ImportError:  # pragma: no cover - pygls version compatibility
    ServerCapabilitiesBuilder = None  # type: ignore[assignment]

from .parser import (
    BlockRange,
    ParsedModel,
    VarDeclaration,
    parse,
    Position as DPos,
    SourceRange as DRange,
    _iter_equation_tag_spans,
    _macro_branch_state,
    _mask_inactive_macro_lines,
    _parse_equations,
    _parse_initval_block,
    _strip_comments,
)
from .diagnostics import (
    Diagnostic as DDiag,
    Severity,
    model_with_include_context,
    run_diagnostics,
    _reserved_identifier_reason,
)
from . import explain as _explain_module
from .workspace import WorkspaceIndex, _normalize_uri

logger = logging.getLogger(__name__)
_position_encoding = lsp.PositionEncodingKind.Utf16

if ServerCapabilitiesBuilder is not None:

    def _choose_position_encoding(cls, client_capabilities):
        global _position_encoding
        general = getattr(client_capabilities, "general", None)
        encodings = getattr(general, "position_encodings", None)
        if encodings is None and isinstance(general, dict):
            encodings = general.get("positionEncodings") or general.get(
                "position_encodings"
            )
        if encodings and lsp.PositionEncodingKind.Utf16 in encodings:
            _position_encoding = lsp.PositionEncodingKind.Utf16
            return lsp.PositionEncodingKind.Utf16
        if encodings and lsp.PositionEncodingKind.Utf8 in encodings:
            _position_encoding = lsp.PositionEncodingKind.Utf8
            return lsp.PositionEncodingKind.Utf8
        if encodings:
            return None
        _position_encoding = lsp.PositionEncodingKind.Utf16
        return lsp.PositionEncodingKind.Utf16

    cast(Any, ServerCapabilitiesBuilder).choose_position_encoding = classmethod(
        _choose_position_encoding,
    )

if TYPE_CHECKING:
    from .bk_check import BKResult
    from .preprocessor import PreprocessorResult
    from .solver import SolverResult
    from .steady_state import SteadyStateReport

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

server = LanguageServer(
    "dynare-language-server",
    "v0.3.1",
    text_document_sync_kind=lsp.TextDocumentSyncKind.Full,
)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _lsp_character_to_source_index(line: str, character: int) -> int:
    """Convert an LSP character offset to a Python string index."""
    character = max(0, character)
    units = 0
    for idx, ch in enumerate(line):
        width = (
            len(ch.encode("utf-8"))
            if _position_encoding == lsp.PositionEncodingKind.Utf8
            else 2
            if ord(ch) > 0xFFFF
            else 1
        )
        if units + width > character:
            return idx
        if units + width == character:
            return idx + 1
        units += width
    return len(line)


def _source_index_to_lsp_character(line: str, character: int) -> int:
    """Convert a Python string index to the negotiated LSP character offset."""
    character = max(0, min(character, len(line)))
    if _position_encoding == lsp.PositionEncodingKind.Utf8:
        return len(line[:character].encode("utf-8"))
    return sum(2 if ord(ch) > 0xFFFF else 1 for ch in line[:character])


def _lsp_position_to_source_position(
    lines: List[str],
    pos: lsp.Position,
) -> lsp.Position:
    if pos.line < 0 or pos.line >= len(lines):
        return pos
    return lsp.Position(
        line=pos.line,
        character=_lsp_character_to_source_index(
            lines[pos.line],
            pos.character,
        ),
    )


def _source_position_to_lsp_position(
    lines: List[str],
    pos: DPos,
) -> lsp.Position:
    if pos.line < 0 or pos.line >= len(lines):
        return lsp.Position(line=pos.line, character=pos.character)
    return lsp.Position(
        line=pos.line,
        character=_source_index_to_lsp_character(
            lines[pos.line],
            pos.character,
        ),
    )


def _valid_line_insert(
    lines: List[str],
    line_no: int,
    new_text: str,
) -> Tuple[lsp.Position, str]:
    """Return a valid insertion position, clamping EOF to the last character."""
    if line_no < len(lines):
        return lsp.Position(line=max(0, line_no), character=0), new_text
    eof_line = max(0, len(lines) - 1)
    source_line = lines[eof_line].rstrip("\r") if lines else ""
    eof_char = _source_index_to_lsp_character(source_line, len(source_line))
    if source_line and not new_text.startswith("\n"):
        new_text = "\n" + new_text
    return lsp.Position(line=eof_line, character=eof_char), new_text


def _to_lsp_range_in_text(text: str, r: DRange) -> lsp.Range:
    lines = text.split("\n")
    return lsp.Range(
        start=_source_position_to_lsp_position(lines, r.start),
        end=_source_position_to_lsp_position(lines, r.end),
    )


def _to_lsp_range(r: DRange) -> lsp.Range:
    return lsp.Range(
        start=lsp.Position(line=r.start.line, character=r.start.character),
        end=lsp.Position(line=r.end.line, character=r.end.character),
    )


def _position_to_offset_local(text: str, pos) -> int:
    """Convert a line/character position in *text* to a clamped offset."""
    lines = text.split("\n")
    if pos.line <= 0:
        first = lines[0] if lines else ""
        return min(max(pos.character, 0), len(first))
    if pos.line >= len(lines):
        return len(text)
    offset = sum(len(line) + 1 for line in lines[: pos.line])
    return min(offset + max(pos.character, 0), offset + len(lines[pos.line]))


def _offset_to_dpos(text: str, offset: int) -> DPos:
    offset = max(0, min(offset, len(text)))
    line = text.count("\n", 0, offset)
    last_nl = text.rfind("\n", 0, offset)
    character = offset if last_nl == -1 else offset - last_nl - 1
    return DPos(line, character)


def _map_range_to_original_source(model: ParsedModel, rng: DRange) -> DRange:
    source_map = getattr(model, "source_map", None)
    original_text = getattr(model, "original_text", "") or model.text
    if not source_map:
        return rng

    def _map_pos(pos: DPos) -> DPos:
        offset = _position_to_offset_local(model.text, pos)
        mapped = source_map[-1] if offset >= len(source_map) else source_map[offset]
        return _offset_to_dpos(original_text, mapped)

    return DRange(_map_pos(rng.start), _map_pos(rng.end))


def _to_lsp_range_for_model(model: Optional[ParsedModel], r: DRange) -> lsp.Range:
    source_text = None
    if model is not None:
        r = _map_range_to_original_source(model, r)
        source_text = getattr(model, "original_text", "") or model.text
    if source_text is not None:
        return _to_lsp_range_in_text(source_text, r)
    return _to_lsp_range(r)


def _source_position_to_model_position(
    model: ParsedModel,
    source_text: str,
    pos: lsp.Position,
) -> lsp.Position:
    source_map = getattr(model, "source_map", None)
    if not source_map:
        lines = source_text.split("\n")
        return _lsp_position_to_source_position(lines, pos)
    lines = source_text.split("\n")
    source_pos = _lsp_position_to_source_position(lines, pos)
    source_offset = _position_to_offset_local(source_text, source_pos)
    best_idx = min(
        range(len(source_map)),
        key=lambda idx: (abs(source_map[idx] - source_offset), idx),
    )
    mapped = _offset_to_dpos(model.text, best_idx)
    return lsp.Position(line=mapped.line, character=mapped.character)


def _document_model_for_uri(uri: str) -> Optional[ParsedModel]:
    with _state_lock:
        model = _document_models.get(uri)
        if model is not None:
            return model
        try:
            target_key = _normalize_uri(uri)
        except Exception:
            return None
        for known_uri, known_model in _document_models.items():
            try:
                if _normalize_uri(known_uri) == target_key:
                    return known_model
            except Exception:
                continue
    return None


def _model_for_uri(uri: str) -> Optional[ParsedModel]:
    model = _workspace_index.get_effective_model(uri)
    if model is not None:
        return model
    return _document_model_for_uri(uri)


def _to_lsp_severity(s: Severity) -> lsp.DiagnosticSeverity:
    return lsp.DiagnosticSeverity(int(s))


def _to_lsp_diagnostic(d: DDiag, source_text: Optional[str] = None) -> lsp.Diagnostic:
    return lsp.Diagnostic(
        range=(
            _to_lsp_range_in_text(source_text, d.range)
            if source_text is not None
            else _to_lsp_range(d.range)
        ),
        severity=_to_lsp_severity(d.severity),
        source=d.source,
        code=d.code,
        message=d.message,
        tags=[lsp.DiagnosticTag(t) for t in d.tags] if d.tags else None,
    )


def _assigned_values(
    model: ParsedModel,
    include_models: Optional[List[ParsedModel]] = None,
) -> dict[str, float]:
    """Return numeric assignments visible in a parsed file."""
    from .solver import _effective_param_values

    if include_models:
        model = model_with_include_context(
            model,
            include_models,
            include_model_equations=False,
        )
    values = _effective_param_values(model)
    for assignment in model.helper_assignments:
        if assignment.value is not None:
            values[assignment.name] = assignment.value
    return values


def _model_local_variables(model: ParsedModel) -> dict[str, str]:
    """Return model-local ``# name = expr`` definitions visible in a model block."""
    local_vars: dict[str, str] = {}
    for equation in model.model_equations:
        match = re.match(
            r"#\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+)$",
            equation.text.strip(),
        )
        if match:
            local_vars[match.group(1)] = match.group(2).strip()
    return local_vars


def _find_model_local_declaration(
    model: ParsedModel,
    name: str,
) -> Optional[VarDeclaration]:
    """Return the source range for a model-local ``#name = expr`` definition."""
    for equation in model.model_equations:
        start_offset = _position_to_offset_local(model.text, equation.range.start)
        end_offset = _position_to_offset_local(model.text, equation.range.end)
        raw_text = model.text[start_offset:end_offset]
        match = re.search(
            r"#\s*([A-Za-z][A-Za-z0-9_]*)\s*=",
            raw_text,
        )
        if match and match.group(1) == name:
            start = _offset_to_dpos(model.text, start_offset + match.start(1))
            end = _offset_to_dpos(model.text, start_offset + match.end(1))
            return VarDeclaration(name=name, range=DRange(start, end))
    return None


def _find_model_local_declaration_in_text(
    text: str,
    name: str,
) -> Optional[VarDeclaration]:
    pattern = re.compile(rf"^\s*#\s*({re.escape(name)})\s*=", re.MULTILINE)
    match = pattern.search(text)
    if match is None:
        return None
    start = _offset_to_dpos(text, match.start(1))
    end = _offset_to_dpos(text, match.end(1))
    return VarDeclaration(name=name, range=DRange(start, end))


def _find_non_model_local_declaration(
    model: ParsedModel,
    name: str,
) -> Optional[VarDeclaration]:
    for v in model.endogenous:
        if v.name == name:
            return v
    for v in model.exogenous:
        if v.name == name:
            return v
    for v in model.parameters:
        if v.name == name:
            return v
    return None


def _word_at_position(lines: List[str], pos: lsp.Position) -> Optional[re.Match]:
    """Find the identifier under the cursor."""
    if pos.line >= len(lines):
        return None
    line = lines[pos.line]
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", line):
        if m.start() <= pos.character < m.end():
            return m
    return None


def _word_at_lsp_position(lines: List[str], pos: lsp.Position) -> Optional[re.Match]:
    """Find the identifier under a UTF-16 LSP cursor position."""
    return _word_at_position(lines, _lsp_position_to_source_position(lines, pos))


def _identifier_at_source_position(
    model: ParsedModel,
    source_text: str,
    pos: lsp.Position,
) -> Tuple[Optional[str], Optional[re.Match]]:
    lines = source_text.split("\n")
    word_match = _word_at_lsp_position(lines, pos)
    name = word_match.group(1) if word_match is not None else None
    if getattr(model, "source_map", None):
        model_pos = _source_position_to_model_position(model, source_text, pos)
        model_word_match = _word_at_position(model.text.split("\n"), model_pos)
        if model_word_match is not None:
            name = model_word_match.group(1)
    return name, word_match


_MACRO_SOURCE_IDENTIFIER_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*(?:@\{[A-Za-z_][A-Za-z0-9_]*\}[A-Za-z0-9_]*)*",
)


def _macro_aware_identifier_span(
    model: ParsedModel,
    source_text: str,
    pos: lsp.Position,
    resolved_name: str,
    fallback: re.Match,
) -> Tuple[int, int]:
    """Return the raw-source span for an identifier that may contain ``@{...}``."""
    source_lines = source_text.split("\n")
    source_pos = _lsp_position_to_source_position(source_lines, pos)
    if source_pos.line >= len(source_lines):
        return fallback.start(), fallback.end()
    line_text = source_lines[source_pos.line]
    for match in _MACRO_SOURCE_IDENTIFIER_RE.finditer(line_text):
        if not (match.start() <= source_pos.character < match.end()):
            continue
        if "@{" not in match.group(0):
            return match.start(), match.end()
        lsp_char = _source_index_to_lsp_character(line_text, match.start())
        model_pos = _source_position_to_model_position(
            model,
            source_text,
            lsp.Position(line=source_pos.line, character=lsp_char),
        )
        model_match = _word_at_position(model.text.split("\n"), model_pos)
        if model_match is not None and model_match.group(1) == resolved_name:
            return match.start(), match.end()
    return fallback.start(), fallback.end()


# ---------------------------------------------------------------------------
# Document analysis
# ---------------------------------------------------------------------------

# Lock protecting all mutable state dicts below.  Keep critical sections
# short — just the dict read/write, never the computation itself.
_state_lock = threading.Lock()

# Serializes outbound JSON-RPC frames.  Validation runs on two independent
# single-worker executors (_solve_executor and _preprocess_executor), so two
# publishes (or a publish and an inlayHint/refresh) can otherwise interleave
# bytes on the shared transport and corrupt a frame.  Acquire only around the
# client send, never while holding _state_lock.
_client_send_lock = threading.Lock()

# Cache parsed models keyed by document URI
_document_models: dict[str, ParsedModel] = {}

# Cache diagnostics for code actions
_document_diagnostics: dict[str, List[DDiag]] = {}

# Cache solver results for display in hover/inlay hints
_document_solver_results: dict[str, "SolverResult"] = {}

# Last successful solver results used only as low-priority warm-start hints.
# This is separate from the display cache so validation can clear stale UI
# data without throwing away branch-continuity information for the next solve.
_document_warm_start_results: dict[str, "SolverResult"] = {}

# Cache BK results
_document_bk_results: dict[str, "BKResult"] = {}

# Cache model diagnostics from successful steady-state solves
_document_model_diagnostics: dict[str, List[DDiag]] = {}

# Cache first-order structural identification diagnostics
_document_identification: dict[str, List[DDiag]] = {}

# Cache preprocessor results
_document_preprocessor_results: dict[str, "PreprocessorResult"] = {}

# Diagnostics from an explicit editor-triggered run (preprocessor or
# MATLAB+Dynare).  Persist until the next edit so the run's findings stay
# visible alongside the static diagnostics; cleared in _validate_document.
_document_run_diagnostics: dict[str, List[DDiag]] = {}

# Cache steady state reports (avoids recomputing on every hover/inlay hint)
_document_ss_reports: dict[str, Optional["SteadyStateReport"]] = {}

# Background solver infrastructure
_solve_executor = ThreadPoolExecutor(max_workers=1)
_pending_solves: dict[str, threading.Timer] = {}  # uri -> debounce timer

# Background preprocessor infrastructure
_preprocess_executor = ThreadPoolExecutor(max_workers=1)
_pending_preprocess: dict[str, threading.Timer] = {}
_preprocessor_path: Optional[str] = None
_steady_state_tolerance: float = 1e-6
# Indent unit used by the document formatter.  Default is a tab; the
# ``dynare.formatIndent`` setting may override it ("tab" or a space count).
_format_indent_unit: str = "\t"

# Workspace-wide @#include index.  Refreshed in every did_open / did_change /
# did_save handler so cross-file diagnostics stay current.  See
# dynare_lsp.workspace for the resolution and cycle-detection logic.
_workspace_index = WorkspaceIndex()


def _declaration_only_model(model: ParsedModel) -> ParsedModel:
    """Return an include-context view that contributes declarations only."""
    return replace(
        model,
        model_equations=[],
        steady_state_equations=[],
        initval_entries=[],
        endval_entries=[],
        model_block_range=None,
        steady_state_block_range=None,
        initval_block_range=None,
        endval_block_range=None,
        shocks_block_range=None,
        blocks=[],
    )


def _include_symbol_map(models: List[ParsedModel]) -> dict:
    symbols = {"endogenous": [], "exogenous": [], "parameters": []}
    seen = {kind: set() for kind in symbols}
    for model in models:
        for kind, attr in (
            ("endogenous", "endogenous"),
            ("exogenous", "exogenous"),
            ("parameters", "parameters"),
        ):
            for decl in getattr(model, attr):
                if decl.name in seen[kind]:
                    continue
                seen[kind].add(decl.name)
                symbols[kind].append(decl)
    return symbols


def _context_models_for_open_include(
    active_uri: str,
    parent_uri: str,
    parent_model: ParsedModel,
    workspace_index: WorkspaceIndex,
) -> List[ParsedModel]:
    """Visible declaration providers for an opened include fragment."""
    active_key = _normalize_uri(active_uri)
    models = [parent_model]
    try:
        siblings = workspace_index.resolve_all_includes(parent_uri)
    except Exception:
        logger.exception("Open include sibling lookup failed for %s", parent_uri)
        siblings = {}
    for path_key, sibling_model in siblings.items():
        real_key = _normalize_uri(_strip_include_instance_suffix(path_key))
        if real_key == active_key:
            continue
        models.append(sibling_model)
    return models


def _contextual_open_include_model(
    local_model: ParsedModel,
    text: str,
    included_view: ParsedModel,
) -> ParsedModel:
    """Parse an open include fragment in its parent's block context.

    The parent-composed ``included_view`` tells us whether this file was
    included inside ``model``, ``steady_state_model``, ``initval``, or
    ``endval``.
    Re-parse the open file's unsaved text into that block while keeping
    all source ranges local to the open file.
    """
    view = replace(local_model)
    stripped = _strip_comments(text)
    context_body, _closes_parent = WorkspaceIndex._context_body_before_closing_end(
        stripped,
    )
    included_context = _context_from_included_view(included_view)
    if included_context == "model" and not view.model_equations:
        view.model_equations = _parse_equations(
            context_body,
            0,
            text,
            filter_commands=True,
        )
        view.param_assignments = []
        view.helper_assignments = []
    elif included_context == "steady_state_model" and not view.steady_state_equations:
        view.steady_state_equations = _parse_equations(
            context_body,
            0,
            text,
            filter_commands=True,
        )
        view.param_assignments = []
        view.helper_assignments = []
    elif included_context == "initval" and not view.initval_entries:
        known = local_model.param_values()
        view.initval_entries = _parse_initval_block(context_body, 0, text, known)
        view.param_assignments = []
        view.helper_assignments = []
    elif included_context == "endval" and not view.endval_entries:
        known = local_model.param_values()
        view.endval_entries = _parse_initval_block(context_body, 0, text, known)
        view.param_assignments = []
        view.helper_assignments = []
    return view


def _open_include_feature_context(
    uri: str,
    model: ParsedModel,
    parent_contexts: Optional[List[Tuple[str, ParsedModel, ParsedModel]]] = None,
    source_text: Optional[str] = None,
) -> Optional[Tuple[str, ParsedModel, List[ParsedModel]]]:
    """Return parent declarations and block context for an opened include."""
    if any(
        (
            model.model_block_range,
            model.steady_state_block_range,
            model.initval_block_range,
            model.endval_block_range,
            model.shocks_block_range,
        )
    ):
        return None
    parent_contexts = parent_contexts or _parent_include_contexts(
        uri,
        _workspace_index,
    )
    parent_context, _ambiguous_parents = _select_parent_include_context(
        uri,
        _workspace_index,
        parent_contexts,
    )
    if parent_context is None:
        return None
    parent_uri, parent_model, included_view = parent_context

    if model.model_equations or included_view.model_equations:
        context = "model"
    elif model.steady_state_equations or included_view.steady_state_equations:
        context = "steady_state_model"
    elif model.initval_entries or included_view.initval_entries:
        context = "initval"
    elif model.endval_entries or included_view.endval_entries:
        context = "endval"
    elif (included_context := _context_from_included_view(included_view)) is not None:
        context = included_context
    else:
        return None

    context_models = _context_models_for_open_include(
        uri,
        parent_uri,
        parent_model,
        _workspace_index,
    )
    declaration_models = [
        _declaration_only_model(context_model) for context_model in context_models
    ]
    local_model = (
        _contextual_open_include_model(model, source_text, included_view)
        if source_text is not None
        else model
    )
    feature_model = model_with_include_context(
        local_model,
        declaration_models,
        include_model_equations=False,
    )
    return context, feature_model, declaration_models


def _context_from_included_view(included_view: ParsedModel) -> Optional[str]:
    context = getattr(included_view, "context_closing_block", None) or getattr(
        included_view, "include_context", None
    )
    if context in {"model", "steady_state_model", "initval", "endval", "shocks"}:
        return context
    if included_view.model_equations or included_view.model_block_range:
        return "model"
    if included_view.steady_state_equations or included_view.steady_state_block_range:
        return "steady_state_model"
    if included_view.initval_entries or included_view.initval_block_range:
        return "initval"
    if included_view.endval_entries or included_view.endval_block_range:
        return "endval"
    if included_view.shocks_block_range or included_view.shocks_vars:
        return "shocks"
    return None


def _validate_document(uri: str, text: str) -> None:
    """Parse, diagnose, and publish diagnostics for a document."""
    # Derived results are tied to the exact parsed model.  Clear them as
    # soon as a fresh validation starts so pull diagnostics and inlay
    # hints cannot mix a new parse with stale solver/BK/preprocessor data.
    with _state_lock:
        _document_models.pop(uri, None)
        previous_solver_result = _document_solver_results.pop(uri, None)
        if previous_solver_result is not None and getattr(
            previous_solver_result, "success", False
        ):
            _document_warm_start_results[uri] = previous_solver_result
        _document_bk_results.pop(uri, None)
        _document_model_diagnostics.pop(uri, None)
        _document_identification.pop(uri, None)
        _document_preprocessor_results.pop(uri, None)
        # Run diagnostics describe a prior execution of now-stale text.
        _document_run_diagnostics.pop(uri, None)
        _document_ss_reports.pop(uri, None)
    _workspace_index.remove_document(uri)

    try:
        model = parse(text)

        # Populate model.blocks from already-detected block ranges (m1 fix)
        blocks: List[BlockRange] = []
        if model.model_block_range:
            blocks.append(BlockRange(keyword="model", range=model.model_block_range))
        if model.steady_state_block_range:
            blocks.append(
                BlockRange(
                    keyword="steady_state_model", range=model.steady_state_block_range
                )
            )
        if model.initval_block_range:
            blocks.append(
                BlockRange(keyword="initval", range=model.initval_block_range)
            )
        if model.endval_block_range:
            blocks.append(BlockRange(keyword="endval", range=model.endval_block_range))
        if model.shocks_block_range:
            blocks.append(BlockRange(keyword="shocks", range=model.shocks_block_range))
        model.blocks = blocks

        # Refresh the cross-file index for this URI and gather the
        # symbols visible through @#include directives.  Failures here
        # never block the single-file checks below.
        include_symbols = None
        include_models = None
        include_cycles: Optional[List[List[str]]] = None
        unresolved_includes = None
        open_include_contextualized = False
        workspace_context_warnings: List[DDiag] = []
        try:
            _workspace_index.update_document(uri, text)
            include_symbols = _workspace_index.collect_symbols(uri)
            include_models = list(_workspace_index.resolve_all_includes(uri).values())
            effective_model = _workspace_index.get_effective_model(uri)
            if effective_model is not None:
                model = effective_model
            include_cycles = _workspace_index.find_circular_includes(uri)
            unresolved_includes = _workspace_index.find_unresolved_includes(uri)
            parent_contexts = _parent_include_contexts(uri, _workspace_index)
            parent_context, ambiguous_parent_uris = _select_parent_include_context(
                uri,
                _workspace_index,
                parent_contexts,
            )
            if ambiguous_parent_uris:
                workspace_context_warnings.append(
                    _ambiguous_parent_context_diagnostic(ambiguous_parent_uris),
                )
            if parent_context is not None and not any(
                (
                    model.model_block_range,
                    model.steady_state_block_range,
                    model.initval_block_range,
                    model.endval_block_range,
                    model.shocks_block_range,
                )
            ):
                parent_uri, parent_model, included_view = parent_context
                model = _contextual_open_include_model(
                    model,
                    text,
                    included_view,
                )
                open_include_contextualized = True
                context_models = _context_models_for_open_include(
                    uri,
                    parent_uri,
                    parent_model,
                    _workspace_index,
                )
                include_symbols = _include_symbol_map(context_models)
                include_models = [
                    _declaration_only_model(context_model)
                    for context_model in context_models
                ]
        except Exception:
            logger.exception("Workspace include analysis failed for %s", uri)

        blocks = []
        if model.model_block_range:
            blocks.append(BlockRange(keyword="model", range=model.model_block_range))
        if model.steady_state_block_range:
            blocks.append(
                BlockRange(
                    keyword="steady_state_model", range=model.steady_state_block_range
                )
            )
        if model.initval_block_range:
            blocks.append(
                BlockRange(keyword="initval", range=model.initval_block_range)
            )
        if model.endval_block_range:
            blocks.append(BlockRange(keyword="endval", range=model.endval_block_range))
        if model.shocks_block_range:
            blocks.append(BlockRange(keyword="shocks", range=model.shocks_block_range))
        model.blocks = blocks

        diagnostics = run_diagnostics(
            model,
            include_symbols=include_symbols,
            include_models=include_models,
            include_cycles=include_cycles,
            unresolved_includes=unresolved_includes,
        )
        diagnostics.extend(workspace_context_warnings)
        if open_include_contextualized:
            diagnostics = [
                diagnostic for diagnostic in diagnostics if diagnostic.code != "E010"
            ]

        # Compute and cache the steady state report so hover/inlay hints
        # can read from the cache instead of recomputing each time.
        from .diagnostics import _with_model_editing_commands
        from .steady_state import validate_steady_state

        ss_model = _with_model_editing_commands(
            model_with_include_context(
                model,
                include_models,
                include_model_equations=False,
            )
        )
        ss_report = validate_steady_state(
            ss_model,
            tolerance=_steady_state_tolerance,
        )

        with _state_lock:
            _document_models[uri] = model
            _document_diagnostics[uri] = diagnostics
            _document_ss_reports[uri] = ss_report
    except Exception as e:
        logger.exception("Error analyzing document %s", uri)
        diagnostics = [
            DDiag(
                range=DRange(DPos(0, 0), DPos(0, 1)),
                severity=Severity.ERROR,
                message=f"Internal parser error: {e}",
                code="E999",
            )
        ]
        with _state_lock:
            _document_diagnostics[uri] = diagnostics
            _document_ss_reports[uri] = None

    lsp_diagnostics = [_to_lsp_diagnostic(d, text) for d in diagnostics]
    with _client_send_lock:
        server.text_document_publish_diagnostics(
            lsp.PublishDiagnosticsParams(uri=uri, diagnostics=lsp_diagnostics)
        )


def _dedupe_diagnostics(diags: List[DDiag]) -> List[DDiag]:
    """Drop exact-duplicate diagnostics, preserving first-seen order.

    Diagnostics are merged from several sources (the Python rule engine, the
    reconciled preprocessor output, Blanchard-Kahn, collinearity, and
    identification), and a few individual checks can emit the same finding more
    than once (e.g. a shock whose variance is specified three times).  Two
    diagnostics with identical severity, code, range, and message are the same
    finding to the user, so only one should be published.
    """
    seen: set = set()
    out: List[DDiag] = []
    for d in diags:
        key = (
            d.severity,
            d.code,
            d.range.start.line,
            d.range.start.character,
            d.range.end.line,
            d.range.end.character,
            d.message,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _publish_all_diagnostics(uri: str) -> None:
    """Merge base + BK + preprocessor diagnostics and publish."""
    with _state_lock:
        base_diags = _document_diagnostics.get(uri, [])
        bk_result = _document_bk_results.get(uri)
        model_diagnostics = _document_model_diagnostics.get(uri, [])
        identification_diagnostics = _document_identification.get(uri, [])
        model = _document_models.get(uri)
        preproc_result = _document_preprocessor_results.get(uri)
        run_diagnostics = list(_document_run_diagnostics.get(uri, []))
    source_text = (
        (getattr(model, "original_text", "") or model.text)
        if model is not None
        else None
    )

    from .preprocessor import reconcile_diagnostics

    all_diags = list(reconcile_diagnostics(base_diags, preproc_result))

    if bk_result is not None:
        try:
            from .bk_check import bk_to_diagnostics

            if model is not None:
                for diag in bk_to_diagnostics(bk_result, model):
                    if getattr(model, "source_map", None):
                        diag = replace(
                            diag,
                            range=_map_range_to_original_source(model, diag.range),
                        )
                    all_diags.append(diag)
        except Exception:
            pass

    if model_diagnostics:
        try:
            if model is not None:
                for diag in model_diagnostics:
                    if getattr(model, "source_map", None):
                        diag = replace(
                            diag,
                            range=_map_range_to_original_source(model, diag.range),
                        )
                    all_diags.append(diag)
        except Exception:
            pass

    if identification_diagnostics:
        try:
            if model is not None:
                for diag in identification_diagnostics:
                    if getattr(model, "source_map", None):
                        diag = replace(
                            diag,
                            range=_map_range_to_original_source(model, diag.range),
                        )
                    all_diags.append(diag)
        except Exception:
            pass

    all_diags.extend(run_diagnostics)
    all_diags = _dedupe_diagnostics(all_diags)
    lsp_diagnostics = [_to_lsp_diagnostic(d, source_text) for d in all_diags]
    with _client_send_lock:
        server.text_document_publish_diagnostics(
            lsp.PublishDiagnosticsParams(uri=uri, diagnostics=lsp_diagnostics)
        )
        try:
            server.workspace_inlay_hint_refresh(None)
        except Exception:
            pass


def _revalidate_cached_documents(
    schedule_solve: bool = True,
    exclude_uri: Optional[str] = None,
) -> None:
    """Re-run analysis for every open document tracked by the server."""
    with _state_lock:
        uris = list(_document_models.keys())

    for uri in uris:
        if uri == exclude_uri:
            continue
        try:
            doc = server.workspace.get_text_document(uri)
            source = doc.source
        except Exception:
            logger.debug("Could not fetch open document %s for revalidation", uri)
            continue
        _validate_document(uri, source)
        if schedule_solve:
            _schedule_solve(uri)


# ---------------------------------------------------------------------------
# Background steady state solver (auto-solve on edit)
# ---------------------------------------------------------------------------


def _schedule_solve(uri: str) -> None:
    """Schedule a background steady state solve, debounced by 1 second."""
    with _state_lock:
        timer = _pending_solves.pop(uri, None)
    if timer is not None:
        timer.cancel()

    def _do_solve():
        with _state_lock:
            _pending_solves.pop(uri, None)
        logger.info("Auto-solve: starting for %s", uri)
        # Capture the model AND a version token so a mid-flight edit
        # doesn't cause us to write stale solver / BK results back to
        # the cache.  The token is ``id(model)`` — every ``parse()``
        # produces a fresh ParsedModel object, so the identity changes
        # the instant ``_validate_document`` re-parses on a did_change.
        with _state_lock:
            model = _document_models.get(uri)
        if model is None:
            logger.info("Auto-solve: no model cached, skipping")
            return
        version_token = id(model)
        model_source = getattr(model, "original_text", "") or model.text
        solve_files = _workspace_files_for_execution(uri, model_source)

        include_models = None
        try:
            include_models = list(_workspace_index.resolve_all_includes(uri).values())
        except Exception:
            logger.exception("Auto-solve: include analysis failed for %s", uri)
        from .diagnostics import _with_model_editing_commands

        solve_model = _with_model_editing_commands(
            model_with_include_context(model, include_models)
        )

        var_names = [v.name for v in solve_model.endogenous]
        real_eqs = [
            eq
            for eq in solve_model.static_model_equations()
            if not eq.text.strip().startswith("#")
        ]
        if len(var_names) == 0 or len(var_names) != len(real_eqs):
            logger.info(
                "Auto-solve: equation/variable count mismatch (%d vars, %d eqs), skipping",
                len(var_names),
                len(real_eqs),
            )
            return

        try:
            from .solver import compute_steady_state, default_solve_budget
        except ImportError:
            logger.info("Auto-solve: scipy not available, skipping")
            return

        # Warm-start from previous solution if available
        with _state_lock:
            prev = _document_solver_results.get(uri)
            if prev is None or not prev.success:
                prev = _document_warm_start_results.get(uri)
        initial_guess = prev.values if prev and prev.success else None

        try:
            result = compute_steady_state(
                solve_model,
                warm_start_guess=initial_guess,
                time_budget=default_solve_budget(),
            )
        except Exception as e:
            logger.exception("Auto-solve: solver raised exception: %s", e)
            return

        snapshot_current = _execution_snapshot_is_current(solve_files)
        with _state_lock:
            current = _document_models.get(uri)
            if current is None or id(current) != version_token or not snapshot_current:
                # Document was edited (or closed) while we computed —
                # discard the stale result rather than overwrite the
                # cache with a value that no longer matches the model.
                logger.info("Auto-solve: discarding stale result for %s", uri)
                return
            _document_solver_results[uri] = result
            if result.success:
                _document_warm_start_results[uri] = result
        logger.info(
            "Auto-solve: success=%s, %d values, method=%s",
            result.success,
            len(result.values),
            result.method_used,
        )

        # Chain BK check after successful solve
        if result.success:
            try:
                from .bk_check import check_blanchard_kahn

                bk = check_blanchard_kahn(solve_model, result.values)
                snapshot_current = _execution_snapshot_is_current(solve_files)
                with _state_lock:
                    current = _document_models.get(uri)
                    if (
                        current is None
                        or id(current) != version_token
                        or not snapshot_current
                    ):
                        logger.info("Auto-solve: discarding stale BK for %s", uri)
                        return
                    _document_bk_results[uri] = bk
                logger.info("Auto-solve: BK check done, satisfied=%s", bk.satisfied)
            except Exception as e:
                logger.info("Auto-solve: BK check failed: %s", e)

            try:
                from .model_diagnostics import check_model_diagnostics

                model_diagnostics = check_model_diagnostics(
                    solve_model,
                    result.values,
                )
                snapshot_current = _execution_snapshot_is_current(solve_files)
                with _state_lock:
                    current = _document_models.get(uri)
                    if (
                        current is None
                        or id(current) != version_token
                        or not snapshot_current
                    ):
                        logger.info(
                            "Auto-solve: discarding stale model diagnostics for %s",
                            uri,
                        )
                        return
                    _document_model_diagnostics[uri] = model_diagnostics
                logger.info(
                    "Auto-solve: model diagnostics done, %d diagnostic(s)",
                    len(model_diagnostics),
                )
            except Exception as e:
                logger.info("Auto-solve: model diagnostics failed: %s", e)

            try:
                from .identification import check_identification

                identification_diagnostics = check_identification(
                    solve_model,
                    result.values,
                )
                snapshot_current = _execution_snapshot_is_current(solve_files)
                with _state_lock:
                    current = _document_models.get(uri)
                    if (
                        current is None
                        or id(current) != version_token
                        or not snapshot_current
                    ):
                        logger.info(
                            "Auto-solve: discarding stale identification for %s",
                            uri,
                        )
                        return
                    _document_identification[uri] = identification_diagnostics
                logger.info(
                    "Auto-solve: identification done, %d diagnostic(s)",
                    len(identification_diagnostics),
                )
            except Exception as e:
                logger.info("Auto-solve: identification failed: %s", e)

        _publish_all_diagnostics(uri)

    new_timer = threading.Timer(1.0, lambda: _solve_executor.submit(_do_solve))
    new_timer.daemon = True
    new_timer.start()
    with _state_lock:
        _pending_solves[uri] = new_timer


# ---------------------------------------------------------------------------
# LSP event handlers
# ---------------------------------------------------------------------------


@server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
def did_open(params: lsp.DidOpenTextDocumentParams) -> None:
    """Validate document on open."""
    doc = params.text_document
    _validate_document(doc.uri, doc.text)
    _schedule_solve(doc.uri)


@server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
def did_change(params: lsp.DidChangeTextDocumentParams) -> None:
    """Re-validate on every change (full sync)."""
    uri = params.text_document.uri
    if params.content_changes:
        # Read the pygls-synced document rather than content_changes[-1].text:
        # robust whether the client sends full or incremental changes.
        text = server.workspace.get_text_document(uri).source
        with _state_lock:
            pending_preprocess = _pending_preprocess.pop(uri, None)
        if pending_preprocess is not None:
            pending_preprocess.cancel()
        _validate_document(uri, text)
        _schedule_solve(uri)
        _revalidate_cached_documents(schedule_solve=True, exclude_uri=uri)


@server.feature(lsp.WORKSPACE_DID_CHANGE_WATCHED_FILES)
def did_change_watched_files(
    params: lsp.DidChangeWatchedFilesParams,
) -> None:
    """Invalidate cached disk-loaded includes when the client sees file changes."""
    for event in params.changes:
        try:
            _workspace_index.remove_document(event.uri)
        except Exception:
            logger.debug("Could not invalidate watched file %s", event.uri)
    _revalidate_cached_documents(schedule_solve=True)


# ---------------------------------------------------------------------------
# Background preprocessor (run on save only)
# ---------------------------------------------------------------------------


def _schedule_preprocess(uri: str) -> None:
    """Schedule a background preprocessor run, debounced by 2 seconds."""
    global _preprocessor_path

    with _state_lock:
        timer = _pending_preprocess.pop(uri, None)
    if timer is not None:
        timer.cancel()

    with _state_lock:
        scheduled_model = _document_models.get(uri)
    if scheduled_model is None:
        logger.info("Preprocessor: document not available, skipping %s", uri)
        return

    try:
        scheduled_doc = server.workspace.get_text_document(uri)
        scheduled_source = scheduled_doc.source
    except Exception:
        logger.info("Preprocessor: could not read document, skipping %s", uri)
        return

    def _do_preprocess():
        global _preprocessor_path
        # Compare by object identity against the scheduled model (which this
        # closure pins, so its id cannot be reused by a freshly-parsed model);
        # a bare id() token would be unreliable once the old model is GC'd.
        with _state_lock:
            _pending_preprocess.pop(uri, None)
            current = _document_models.get(uri)
            if current is None or current is not scheduled_model:
                logger.info("Preprocessor: discarding stale scheduled run for %s", uri)
                return

        # Auto-detect preprocessor on first call
        if _preprocessor_path is None:
            try:
                from .preprocessor import find_preprocessor

                _preprocessor_path = find_preprocessor() or ""
            except Exception:
                _preprocessor_path = ""

        if not _preprocessor_path:
            return  # silently skip

        try:
            (
                entry_uri,
                preprocessor_source,
                _source_dir,
                active_include_uri,
            ) = _preprocessor_scope_for_uri(uri, scheduled_source)
            preprocessor_files = _workspace_files_for_execution(
                entry_uri,
                preprocessor_source,
            )
            preprocessor_active_file = _execution_file_key(entry_uri)
            result = _run_preprocessor_with_snapshot(
                preprocessor_source,
                _preprocessor_path,
                preprocessor_active_file,
                preprocessor_files,
            )
            result = _scope_preprocessor_result_to_include(
                result,
                active_include_uri,
            )
            snapshot_current = _execution_snapshot_is_current(preprocessor_files)
            with _state_lock:
                current = _document_models.get(uri)
                if (
                    current is None
                    or current is not scheduled_model
                    or not snapshot_current
                ):
                    logger.info("Preprocessor: discarding stale result for %s", uri)
                    return
                _document_preprocessor_results[uri] = result
            logger.info(
                "Preprocessor: success=%s, %d diagnostics",
                result.success,
                len(result.diagnostics),
            )
        except Exception as e:
            logger.info("Preprocessor: failed: %s", e)
            return

        _publish_all_diagnostics(uri)

    new_timer = threading.Timer(
        2.0, lambda: _preprocess_executor.submit(_do_preprocess)
    )
    new_timer.daemon = True
    new_timer.start()
    with _state_lock:
        _pending_preprocess[uri] = new_timer


@server.feature(lsp.TEXT_DOCUMENT_DID_SAVE)
def did_save(params: lsp.DidSaveTextDocumentParams) -> None:
    """Re-validate on save and trigger preprocessor."""
    uri = params.text_document.uri
    doc = server.workspace.get_text_document(uri)
    _validate_document(uri, doc.source)
    _schedule_preprocess(uri)


@server.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
def did_close(params: lsp.DidCloseTextDocumentParams) -> None:
    """Clean up on close."""
    uri = params.text_document.uri
    with _state_lock:
        handle = _pending_solves.pop(uri, None)
    if handle is not None:
        handle.cancel()
    with _state_lock:
        handle = _pending_preprocess.pop(uri, None)
    if handle is not None:
        handle.cancel()
    with _state_lock:
        _document_models.pop(uri, None)
        _document_diagnostics.pop(uri, None)
        _document_solver_results.pop(uri, None)
        _document_warm_start_results.pop(uri, None)
        _document_bk_results.pop(uri, None)
        _document_model_diagnostics.pop(uri, None)
        _document_identification.pop(uri, None)
        _document_preprocessor_results.pop(uri, None)
        _document_run_diagnostics.pop(uri, None)
        _document_ss_reports.pop(uri, None)
    try:
        _workspace_index.remove_document(uri)
    except Exception:
        logger.exception("Workspace index cleanup failed for %s", uri)
    _revalidate_cached_documents(schedule_solve=True, exclude_uri=uri)
    with _client_send_lock:
        server.text_document_publish_diagnostics(
            lsp.PublishDiagnosticsParams(uri=uri, diagnostics=[])
        )


# ---------------------------------------------------------------------------
# Workspace configuration — honor dynare.searchPaths
# ---------------------------------------------------------------------------


def _apply_search_paths_from_config(raw_paths) -> None:
    """Replace ``dynare.searchPaths`` in the workspace index."""
    if raw_paths is None:
        return
    if not isinstance(raw_paths, list):
        return
    paths: List[Path] = []
    try:
        workspace_root = getattr(server.workspace, "root_path", None)
    except Exception:
        workspace_root = None
    for entry in raw_paths:
        if not isinstance(entry, str) or not entry.strip():
            continue
        try:
            candidate = Path(entry)
            # Resolve relative entries against the workspace root, not the
            # server's launch CWD (often the editor's install directory).
            if not candidate.is_absolute() and workspace_root:
                candidate = Path(workspace_root) / candidate
            paths.append(candidate.resolve())
        except Exception:
            logger.exception("Failed to normalize search path %r", entry)
    _workspace_index.set_search_paths(paths)


def _apply_search_paths_by_root_from_config(raw_mapping) -> None:
    """Replace root-scoped ``dynare.searchPaths`` in the workspace index."""
    if raw_mapping is None:
        return
    if not isinstance(raw_mapping, dict):
        return
    mapping: Dict[Path, List[Path]] = {}
    for root_entry, raw_paths in raw_mapping.items():
        if not isinstance(root_entry, str) or not root_entry.strip():
            continue
        if not isinstance(raw_paths, list):
            continue
        try:
            root = Path(root_entry).resolve()
        except Exception:
            logger.exception("Failed to normalize workspace root %r", root_entry)
            continue
        paths: List[Path] = []
        for entry in raw_paths:
            if not isinstance(entry, str) or not entry.strip():
                continue
            try:
                paths.append(Path(entry).resolve())
            except Exception:
                logger.exception("Failed to normalize search path %r", entry)
        mapping[root] = paths
    _workspace_index.set_root_search_paths(mapping)


def _apply_dynare_config(dynare_section: dict) -> None:
    """Apply supported ``dynare.*`` settings from the editor."""
    global _preprocessor_path, _steady_state_tolerance, _format_indent_unit

    if "formatIndent" in dynare_section:
        value = dynare_section.get("formatIndent")
        if isinstance(value, str) and value.strip().lower() == "tab":
            _format_indent_unit = "\t"
        elif isinstance(value, bool):
            pass
        elif (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and float(value).is_integer()
            and 1 <= int(value) <= 8
        ):
            _format_indent_unit = " " * int(value)
        elif isinstance(value, str) and value.strip().isdigit():
            count = int(value.strip())
            if 1 <= count <= 8:
                _format_indent_unit = " " * count

    if "searchPaths" in dynare_section:
        _apply_search_paths_from_config(dynare_section.get("searchPaths"))
    if "searchPathsByRoot" in dynare_section:
        _apply_search_paths_by_root_from_config(
            dynare_section.get("searchPathsByRoot"),
        )

    if "preprocessorPath" in dynare_section:
        value = dynare_section.get("preprocessorPath")
        if isinstance(value, str):
            _preprocessor_path = value.strip() or None

    if "steadyStateTolerance" in dynare_section:
        value = dynare_section.get("steadyStateTolerance")
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
        ):
            _steady_state_tolerance = float(value)


@server.feature(lsp.WORKSPACE_DID_CHANGE_CONFIGURATION)
def did_change_configuration(
    params: lsp.DidChangeConfigurationParams,
) -> None:
    """React to ``workspace/didChangeConfiguration`` updates.

    Honors ``dynare.searchPaths``, ``dynare.preprocessorPath``,
    ``dynare.steadyStateTolerance`` and ``dynare.formatIndent`` from clients
    such as VS Code.
    """
    settings = params.settings or {}
    dynare_section = settings.get("dynare", {}) if isinstance(settings, dict) else {}
    if isinstance(dynare_section, dict):
        _apply_dynare_config(dynare_section)
        _revalidate_cached_documents(schedule_solve=True)


# ---------------------------------------------------------------------------
# Hover support — show variable info on hover
# ---------------------------------------------------------------------------

_TIMING_LABEL = {
    "static": "static",
    "predetermined": "predetermined (state)",
    "forward_looking": "forward-looking (jumper)",
    "mixed": "mixed (state + jumper)",
}


def _format_time_offset(offset: int) -> str:
    """Render a timing offset for display: 0 -> ``t``, +1 -> ``t+1``, -1 -> ``t-1``."""
    return "t" if offset == 0 else f"t{offset:+d}"


def _format_timing_line(entry: dict) -> str:
    """One-line timing summary for a variable's hover from its classification."""
    label = _TIMING_LABEL.get(entry.get("class", "static"), "static")
    appears = ", ".join(_format_time_offset(o) for o in entry.get("offsets", [0]))
    return f"Timing: **{label}** · appears at {appears}"


def _aux_lead_lag_note(
    line_text: str,
    ident_end: int,
    is_exogenous: bool,
) -> Optional[str]:
    """Hover note for the auxiliary variables a ``|lead/lag| >= 2`` generates.

    *ident_end* is the column just past the hovered identifier on *line_text*.
    Dynare reduces every model to one-period leads/lags by introducing
    auxiliary variables, so e.g. ``x(+3)`` adds two ``AUX_ENDO_LEAD`` variables.
    This explains why a solved model reports more variables than were declared.
    """
    match = re.match(r"\s*\(\s*([+-]?\d+)\s*\)", line_text[ident_end:])
    if match is None:
        return None
    try:
        offset = int(match.group(1))
    except ValueError:
        return None
    if abs(offset) < 2:
        return None
    n_aux = abs(offset) - 1
    plural = "" if n_aux == 1 else "s"
    direction = "lead" if offset > 0 else "lag"
    if is_exogenous:
        family = "AUX_EXO_LEAD" if offset > 0 else "AUX_EXO_LAG"
    else:
        family = "AUX_ENDO_LEAD" if offset > 0 else "AUX_ENDO_LAG"
    return (
        f"📦 Auxiliary variables: this {direction} of {offset:+d} makes Dynare "
        f"add **{n_aux}** auxiliary variable{plural} (`{family}_*`) so the model "
        "keeps only one-period leads/lags — this is why a solved model can "
        "report more variables than are declared."
    )


@server.feature(lsp.TEXT_DOCUMENT_HOVER)
def hover(params: lsp.HoverParams) -> Optional[lsp.Hover]:
    """Show variable/parameter info on hover."""
    uri = params.text_document.uri
    with _state_lock:
        model = _document_models.get(uri)
    if model is None:
        return None

    pos = params.position
    doc = server.workspace.get_text_document(uri)
    lines = doc.source.split("\n")
    if _cursor_inside_comment_or_string(lines, pos) and not _position_inside_mcp_value(
        doc.source, pos
    ):
        return None

    word, word_match = _identifier_at_source_position(model, doc.source, pos)
    if word is None or word_match is None:
        return None
    if _is_on_the_fly_kind_marker(
        lines[pos.line],
        word_match.start(),
        word_match.end(),
    ):
        return None

    # Option hover: when the cursor sits on an option name inside a command's
    # parenthesised option list, describe that option from the manual catalog.
    option_command = _command_option_context(lines, pos)
    if option_command is not None:
        from .dynare_commands import command_options, option_doc

        if any(opt == word for opt, _ in command_options(option_command)):
            md = f"**`{option_command}` option**: `{word}`"
            description = option_doc(word)
            if description:
                md += f"\n\n{description}"
            return lsp.Hover(
                contents=lsp.MarkupContent(
                    kind=lsp.MarkupKind.Markdown,
                    value=md,
                ),
            )

    info_parts: List[str] = []

    # Check if it's an endogenous variable
    for v in model.endogenous:
        if v.name == word:
            info_parts.append(f"**Endogenous variable**: `{word}`")
            with _state_lock:
                report = _document_ss_reports.get(uri)
            if report and word in report.values:
                val = report.values[word]
                info_parts.append(f"Steady state value: `{val:.6g}`")
                if report.n_failed > 0:
                    total = len([r for r in report.results if not r.is_local_var])
                    info_parts.append(
                        f"⚠ Steady state: {report.n_satisfied}/{total} equations satisfied"
                    )
            elif report and word in report.missing_endogenous:
                info_parts.append("Steady state value: *not computed*")
            elif not report:
                # No SS block -- check solver cache
                with _state_lock:
                    solver_result = _document_solver_results.get(uri)
                if (
                    solver_result
                    and solver_result.success
                    and word in solver_result.values
                ):
                    val = solver_result.values[word]
                    info_parts.append(f"Computed steady state: `{val:.6g}`")
                    info_parts.append(f"(solver: {solver_result.method_used})")
            # Static timing classification (no solve required), so timing is
            # shown even for the broken models that never reach a steady state.
            from .model_info import classify_variable_timing

            timing_entry = classify_variable_timing(model).get(word)
            if timing_entry is not None:
                info_parts.append(_format_timing_line(timing_entry))
            break

    # Check if it's an exogenous variable
    for v in model.exogenous:
        if v.name == word:
            info_parts.append(f"**Exogenous variable**: `{word}`")
            info_parts.append("Steady state value: `0` (by definition)")
            break

    # Auxiliary-variable preview when the hovered variable carries a |lead/lag|
    # of 2 or more right here (e.g. ``x(+3)``).  word_match.end() is a column on
    # the cursor's line.
    if pos.line < len(lines) and (
        word in model.endogenous_names() or word in model.exogenous_names()
    ):
        aux_note = _aux_lead_lag_note(
            lines[pos.line],
            word_match.end(),
            is_exogenous=word in model.exogenous_names(),
        )
        if aux_note is not None:
            info_parts.append(aux_note)

    # Check if it's a parameter
    for v in model.parameters:
        if v.name == word:
            info_parts.append(f"**Parameter**: `{word}`")
            vals = _assigned_values(
                model,
                list(_workspace_index.resolve_all_includes(uri).values()),
            )
            if word in vals:
                info_parts.append(f"Value: `{vals[word]:.6g}`")
            else:
                info_parts.append("Value: *not assigned*")
            break

    local_vars = _model_local_variables(model)
    word_is_active_model_local = (
        word in local_vars
        and _is_model_local_symbol_at_position(model, doc.source, pos, word)
    )
    if not info_parts and word in local_vars and not word_is_active_model_local:
        return None
    if not info_parts and word_is_active_model_local:
        info_parts.append(f"**Model-local variable**: `{word}`")
        info_parts.append(f"Expression: `{local_vars[word]}`")

    # Cross-file fallback: nothing matched locally, look in transitively
    # included files so identifiers declared in @#include'd .mod files
    # still hover.  Cross-file hover is intentionally simpler than the
    # local one — no steady-state / BK / solver info, since those are
    # computed from the active model only.
    if not info_parts:
        cross = _find_identifier_in_includes(uri, word, _workspace_index)
        if cross is not None:
            class_label, _target_uri, source_filename, included_model = cross
            if class_label == "endogenous":
                info_parts.append(f"**Endogenous variable**: `{word}`")
            elif class_label == "exogenous":
                info_parts.append(f"**Exogenous variable**: `{word}`")
                info_parts.append("Steady state value: `0` (by definition)")
            elif class_label == "parameter":
                info_parts.append(f"**Parameter**: `{word}`")
                vals = {
                    **included_model.param_values(),
                    **_assigned_values(
                        model,
                        list(_workspace_index.resolve_all_includes(uri).values()),
                    ),
                }
                if word in vals:
                    info_parts.append(f"Value: `{vals[word]:.6g}`")
                else:
                    info_parts.append("Value: *not assigned*")
            elif class_label == "model-local":
                local_expr = _model_local_variables(included_model).get(word)
                info_parts.append(f"**Model-local variable**: `{word}`")
                if local_expr:
                    info_parts.append(f"Expression: `{local_expr}`")
            info_parts.append(f"Declared in `@#include`d file `{source_filename}`")

    if not info_parts:
        hit = _find_declaration_across_workspace_with_model(
            uri,
            word,
            _workspace_index,
        )
        if hit is not None:
            target_uri, decl, target_model = hit
            if target_model is not None:
                if any(v.name == decl.name for v in target_model.endogenous):
                    info_parts.append(f"**Endogenous variable**: `{word}`")
                elif any(v.name == decl.name for v in target_model.exogenous):
                    info_parts.append(f"**Exogenous variable**: `{word}`")
                    info_parts.append("Steady state value: `0` (by definition)")
                elif any(v.name == decl.name for v in target_model.parameters):
                    info_parts.append(f"**Parameter**: `{word}`")
                    vals = target_model.param_values()
                    if word in vals:
                        info_parts.append(f"Value: `{vals[word]:.6g}`")
                    else:
                        info_parts.append("Value: *not assigned*")
            source_filename = Path(_normalize_uri(target_uri)).name
            info_parts.append(f"Declared in `@#include`d file `{source_filename}`")

    if not info_parts:
        return None

    span_start, span_end = _macro_aware_identifier_span(
        model,
        doc.source,
        pos,
        word,
        word_match,
    )
    return lsp.Hover(
        contents=lsp.MarkupContent(
            kind=lsp.MarkupKind.Markdown,
            value="\n\n".join(info_parts),
        ),
        range=lsp.Range(
            start=_source_position_to_lsp_position(
                lines,
                DPos(pos.line, span_start),
            ),
            end=_source_position_to_lsp_position(
                lines,
                DPos(pos.line, span_end),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Document symbols — structured outline of the model
# ---------------------------------------------------------------------------


@server.feature(lsp.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
def document_symbol(
    params: lsp.DocumentSymbolParams,
) -> Optional[List[lsp.DocumentSymbol]]:
    """Return a hierarchical symbol outline of the .mod file.

    LLM-friendly: provides a structured view of all model components
    (endogenous vars, exogenous vars, parameters, equations, blocks).
    """
    uri = params.text_document.uri
    with _state_lock:
        model = _document_models.get(uri)
    if model is None:
        return None

    symbols: List[lsp.DocumentSymbol] = []

    # --- Endogenous variables ---
    if model.endogenous:
        endo_children = []
        for v in model.endogenous:
            r = _to_lsp_range_for_model(model, v.range)
            endo_children.append(
                lsp.DocumentSymbol(
                    name=v.name,
                    kind=lsp.SymbolKind.Variable,
                    range=r,
                    selection_range=r,
                )
            )
        # Parent symbol spanning all endogenous declarations
        first = _to_lsp_range_for_model(model, model.endogenous[0].range)
        last = _to_lsp_range_for_model(model, model.endogenous[-1].range)
        symbols.append(
            lsp.DocumentSymbol(
                name="var (endogenous)",
                detail=f"{len(model.endogenous)} variables",
                kind=lsp.SymbolKind.Namespace,
                range=lsp.Range(start=first.start, end=last.end),
                selection_range=first,
                children=endo_children,
            )
        )

    # --- Exogenous variables ---
    if model.exogenous:
        exo_children = []
        for v in model.exogenous:
            r = _to_lsp_range_for_model(model, v.range)
            exo_children.append(
                lsp.DocumentSymbol(
                    name=v.name,
                    kind=lsp.SymbolKind.Variable,
                    range=r,
                    selection_range=r,
                )
            )
        first = _to_lsp_range_for_model(model, model.exogenous[0].range)
        last = _to_lsp_range_for_model(model, model.exogenous[-1].range)
        symbols.append(
            lsp.DocumentSymbol(
                name="varexo (exogenous)",
                detail=f"{len(model.exogenous)} variables",
                kind=lsp.SymbolKind.Namespace,
                range=lsp.Range(start=first.start, end=last.end),
                selection_range=first,
                children=exo_children,
            )
        )

    # --- Parameters ---
    if model.parameters:
        param_children = []
        vals = _assigned_values(
            model,
            list(_workspace_index.resolve_all_includes(uri).values()),
        )
        for v in model.parameters:
            r = _to_lsp_range_for_model(model, v.range)
            detail = f"= {vals[v.name]:.6g}" if v.name in vals else "unassigned"
            param_children.append(
                lsp.DocumentSymbol(
                    name=v.name,
                    detail=detail,
                    kind=lsp.SymbolKind.Constant,
                    range=r,
                    selection_range=r,
                )
            )
        first = _to_lsp_range_for_model(model, model.parameters[0].range)
        last = _to_lsp_range_for_model(model, model.parameters[-1].range)
        symbols.append(
            lsp.DocumentSymbol(
                name="parameters",
                detail=f"{len(model.parameters)} parameters",
                kind=lsp.SymbolKind.Namespace,
                range=lsp.Range(start=first.start, end=last.end),
                selection_range=first,
                children=param_children,
            )
        )

    # --- Model block equations ---
    if model.model_equations and model.model_block_range:
        eq_children = []
        real_eq_count = 0
        for eq in model.model_equations:
            r = _to_lsp_range_for_model(model, eq.range)
            is_local = eq.text.strip().startswith("#")
            if is_local:
                label = eq.text.strip()[:60]
                kind = lsp.SymbolKind.Property
            else:
                real_eq_count += 1
                label = eq.name if eq.name else eq.text.strip()[:60]
                kind = lsp.SymbolKind.Function
            eq_children.append(
                lsp.DocumentSymbol(
                    name=label,
                    kind=kind,
                    range=r,
                    selection_range=r,
                )
            )
        block_range = _to_lsp_range_for_model(model, model.model_block_range)
        symbols.append(
            lsp.DocumentSymbol(
                name="model",
                detail=f"{real_eq_count} equations",
                kind=lsp.SymbolKind.Module,
                range=block_range,
                selection_range=block_range,
                children=eq_children,
            )
        )

    # --- Steady state model block ---
    if model.steady_state_equations and model.steady_state_block_range:
        ss_children = []
        for eq in model.steady_state_equations:
            r = _to_lsp_range_for_model(model, eq.range)
            label = eq.text.strip()[:60]
            ss_children.append(
                lsp.DocumentSymbol(
                    name=label,
                    kind=lsp.SymbolKind.Property,
                    range=r,
                    selection_range=r,
                )
            )
        block_range = _to_lsp_range_for_model(model, model.steady_state_block_range)
        symbols.append(
            lsp.DocumentSymbol(
                name="steady_state_model",
                detail=f"{len(model.steady_state_equations)} assignments",
                kind=lsp.SymbolKind.Module,
                range=block_range,
                selection_range=block_range,
                children=ss_children,
            )
        )

    # --- Macro preprocessor directives ---
    # Surface @#include / @#includepath / @#define entries so the outline
    # tree includes a navigable "macros" group.  @#if / @#for / etc are
    # control flow and intentionally not listed individually — they show
    # up as foldable regions in the editor instead.
    macro_children: List[lsp.DocumentSymbol] = []
    for inc in model.includes:
        r = _to_lsp_range_for_model(model, inc.range)
        macro_children.append(
            lsp.DocumentSymbol(
                name=f"@#include {inc.filename}",
                kind=lsp.SymbolKind.File,
                range=r,
                selection_range=r,
            )
        )
    for d in model.macro_directives:
        if d.kind not in ("includepath", "define"):
            continue
        r = _to_lsp_range_for_model(model, d.range)
        label = f"@#{d.kind} {d.argument}" if d.argument else f"@#{d.kind}"
        macro_children.append(
            lsp.DocumentSymbol(
                name=label[:80],
                kind=lsp.SymbolKind.Constant
                if d.kind == "define"
                else lsp.SymbolKind.File,
                range=r,
                selection_range=r,
            )
        )
    if macro_children:
        macro_children.sort(
            key=lambda child: (
                child.range.start.line,
                child.range.start.character,
                child.range.end.line,
                child.range.end.character,
            )
        )
        # Span from the first to the last directive so the parent range
        # is meaningful for editors that scroll into it.
        first_range = macro_children[0].range
        last_range = macro_children[-1].range
        symbols.append(
            lsp.DocumentSymbol(
                name="macros",
                detail=f"{len(macro_children)} directives",
                kind=lsp.SymbolKind.Namespace,
                range=lsp.Range(start=first_range.start, end=last_range.end),
                selection_range=first_range,
                children=macro_children,
            )
        )

    # --- Initval block ---
    if model.initval_entries and model.initval_block_range:
        iv_children = []
        for entry in model.initval_entries:
            r = _to_lsp_range_for_model(model, entry.range)
            detail = (
                f"= {entry.value:.6g}" if entry.value is not None else entry.expression
            )
            iv_children.append(
                lsp.DocumentSymbol(
                    name=entry.name,
                    detail=detail,
                    kind=lsp.SymbolKind.Property,
                    range=r,
                    selection_range=r,
                )
            )
        block_range = _to_lsp_range_for_model(model, model.initval_block_range)
        symbols.append(
            lsp.DocumentSymbol(
                name="initval",
                detail=f"{len(model.initval_entries)} entries",
                kind=lsp.SymbolKind.Module,
                range=block_range,
                selection_range=block_range,
                children=iv_children,
            )
        )

    # --- Endval block ---
    if model.endval_entries and model.endval_block_range:
        ev_children = []
        for entry in model.endval_entries:
            r = _to_lsp_range_for_model(model, entry.range)
            detail = (
                f"= {entry.value:.6g}" if entry.value is not None else entry.expression
            )
            ev_children.append(
                lsp.DocumentSymbol(
                    name=entry.name,
                    detail=detail,
                    kind=lsp.SymbolKind.Property,
                    range=r,
                    selection_range=r,
                )
            )
        block_range = _to_lsp_range_for_model(model, model.endval_block_range)
        symbols.append(
            lsp.DocumentSymbol(
                name="endval",
                detail=f"{len(model.endval_entries)} entries",
                kind=lsp.SymbolKind.Module,
                range=block_range,
                selection_range=block_range,
                children=ev_children,
            )
        )

    return symbols if symbols else None


# ---------------------------------------------------------------------------
# Go to definition — jump to variable/parameter declarations
# ---------------------------------------------------------------------------


def _find_declaration(model: ParsedModel, name: str) -> Optional[VarDeclaration]:
    """Find the declaration of a variable or parameter by name."""
    for v in model.endogenous:
        if v.name == name:
            return v
    for v in model.exogenous:
        if v.name == name:
            return v
    for v in model.parameters:
        if v.name == name:
            return v
    local_decl = _find_model_local_declaration(model, name)
    if local_decl is not None:
        return local_decl
    return None


def _find_identifier_in_includes(
    uri: str,
    name: str,
    workspace_index: WorkspaceIndex,
) -> Optional[Tuple[str, str, str, ParsedModel]]:
    """Look up *name*'s declaration class in transitively-included files.

    Returns ``(class_label, target_uri, source_filename, included_model)``
    for the first match.  ``class_label`` is one of ``"endogenous"``,
    ``"exogenous"``, ``"parameter"``; ``source_filename`` is the
    basename of the include file for human-readable hover content.  The
    active file is intentionally skipped — callers (hover, completion)
    check the local model first and only consult this helper to surface
    names that live in includes.  Returns ``None`` when the name is not
    declared anywhere reachable from *uri*.
    """
    try:
        included = workspace_index.resolve_all_includes(uri)
    except Exception:
        logger.exception("Cross-file include lookup failed for %s", uri)
        return None

    for path_key, included_model in included.items():
        target_uri = _path_to_uri(path_key)
        source_filename = _display_path_name(path_key)
        for v in included_model.endogenous:
            if v.name == name:
                return "endogenous", target_uri, source_filename, included_model
        for v in included_model.exogenous:
            if v.name == name:
                return "exogenous", target_uri, source_filename, included_model
        for v in included_model.parameters:
            if v.name == name:
                return "parameter", target_uri, source_filename, included_model
        if name in _model_local_variables(included_model):
            return "model-local", target_uri, source_filename, included_model

    return None


def _collect_included_declarations(
    uri: str,
    workspace_index: WorkspaceIndex,
) -> List[Tuple[str, VarDeclaration, str, str, Optional[float]]]:
    """Return declarations from transitively-included files of *uri*.

    Each entry is ``(class_label, declaration, target_uri,
    source_filename, value)`` where *value* is the parameter's numeric
    value when known and ``None`` otherwise.  Names already declared in
    the active file are excluded so the caller doesn't need to dedupe —
    local declarations win.  The helper is used by ``completion`` to
    offer cross-file identifiers as suggestions tagged with their
    source file.
    """
    active_model = workspace_index.get_model(uri)
    local_names: set = set()
    if active_model is not None:
        local_names = (
            {v.name for v in active_model.endogenous}
            | {v.name for v in active_model.exogenous}
            | {v.name for v in active_model.parameters}
        )

    try:
        included = workspace_index.resolve_all_includes(uri)
    except Exception:
        logger.exception("Cross-file include collection failed for %s", uri)
        return []

    visible_values = (
        _assigned_values(active_model, list(included.values()))
        if active_model is not None
        else {}
    )
    out: List[Tuple[str, VarDeclaration, str, str, Optional[float]]] = []
    for path_key, included_model in included.items():
        target_uri = _path_to_uri(path_key)
        source_filename = _display_path_name(path_key)
        for v in included_model.endogenous:
            if v.name in local_names:
                continue
            out.append(("endogenous", v, target_uri, source_filename, None))
        for v in included_model.exogenous:
            if v.name in local_names:
                continue
            out.append(("exogenous", v, target_uri, source_filename, None))
        for v in included_model.parameters:
            if v.name in local_names:
                continue
            out.append(
                (
                    "parameter",
                    v,
                    target_uri,
                    source_filename,
                    visible_values.get(v.name),
                )
            )
        for name in sorted(_model_local_variables(included_model)):
            if name in local_names:
                continue
            decl = _find_model_local_declaration_in_text(
                getattr(included_model, "original_text", "") or included_model.text,
                name,
            ) or _find_model_local_declaration(included_model, name)
            if decl is not None:
                out.append(("model-local", decl, target_uri, source_filename, None))
    return out


def _parent_include_contexts(
    uri: str,
    workspace_index: WorkspaceIndex,
) -> List[Tuple[str, ParsedModel, ParsedModel]]:
    """Return open/indexed parents whose include closure reaches *uri*."""
    active_key = _normalize_uri(uri)
    parents: List[Tuple[str, ParsedModel, ParsedModel]] = []
    for parent_key in workspace_index.all_uris():
        if _normalize_uri(parent_key) == active_key:
            continue
        try:
            included = workspace_index.resolve_all_includes(parent_key)
        except Exception:
            logger.exception("Parent include lookup failed for %s", parent_key)
            continue
        direct_contexts: dict[str, Tuple[Optional[str], Optional[DRange]]] = {}
        try:
            for (
                directive,
                resolved_key,
                context,
            ) in workspace_index.resolve_direct_includes(
                parent_key,
            ):
                direct_contexts[_normalize_uri(resolved_key)] = (
                    context,
                    directive.range,
                )
        except Exception:
            logger.exception("Direct include lookup failed for %s", parent_key)
        for include_key, included_model in included.items():
            real_key = _normalize_uri(_strip_include_instance_suffix(include_key))
            if real_key != active_key:
                continue
            context, anchor_range = direct_contexts.get(real_key, (None, None))
            if (
                context is not None
                and _context_from_included_view(included_model) is None
                and anchor_range is not None
            ):
                included_model = _included_view_with_context_marker(
                    included_model,
                    context,
                    anchor_range,
                )
            parent_model = workspace_index.get_effective_model(
                parent_key
            ) or workspace_index.get_model(parent_key)
            if parent_model is not None:
                parents.append((_path_to_uri(parent_key), parent_model, included_model))
            break
    return parents


def _select_parent_include_context(
    uri: str,
    workspace_index: WorkspaceIndex,
    parent_contexts: Optional[List[Tuple[str, ParsedModel, ParsedModel]]] = None,
) -> Tuple[Optional[Tuple[str, ParsedModel, ParsedModel]], List[str]]:
    parent_contexts = (
        parent_contexts
        if parent_contexts is not None
        else _parent_include_contexts(uri, workspace_index)
    )
    if not parent_contexts:
        return None, []
    if len(parent_contexts) == 1:
        return parent_contexts[0], []

    include_keys_by_parent: Dict[str, set] = {}
    for parent_uri, _parent_model, _included_model in parent_contexts:
        try:
            included = workspace_index.resolve_all_includes(parent_uri)
        except Exception:
            logger.exception("Parent disambiguation failed for %s", parent_uri)
            included = {}
        include_keys_by_parent[parent_uri] = {
            _normalize_uri(_strip_include_instance_suffix(include_key))
            for include_key in included
        }

    outermost = []
    for context in parent_contexts:
        parent_uri = context[0]
        parent_key = _normalize_uri(parent_uri)
        if not any(
            parent_key in include_keys_by_parent.get(other_uri, set())
            for other_uri, _other_model, _other_included in parent_contexts
            if other_uri != parent_uri
        ):
            outermost.append(context)

    if len(outermost) == 1:
        return outermost[0], []
    return None, [context[0] for context in parent_contexts]


def _display_parent_context_name(uri_or_path: str) -> str:
    try:
        if uri_or_path.startswith("file://"):
            from .include_resolver import _uri_to_path

            return _uri_to_path(uri_or_path).name
    except Exception:
        pass
    return Path(_strip_include_instance_suffix(uri_or_path)).name


def _ambiguous_parent_context_diagnostic(parent_uris: List[str]) -> DDiag:
    parent_names = ", ".join(
        sorted(_display_parent_context_name(parent_uri) for parent_uri in parent_uris)
    )
    return DDiag(
        range=DRange(DPos(0, 0), DPos(0, 1)),
        severity=Severity.WARNING,
        message=(
            "This include is reachable from multiple parent files "
            f"({parent_names}); open or run the intended parent model to get "
            "include-scoped diagnostics in the right context."
        ),
        source="dynare",
        code="W061",
    )


def _included_view_with_context_marker(
    included_model: ParsedModel,
    context: str,
    anchor_range: DRange,
) -> ParsedModel:
    if context == "model":
        return replace(included_model, model_block_range=anchor_range)
    if context == "steady_state_model":
        return replace(included_model, steady_state_block_range=anchor_range)
    if context == "initval":
        return replace(included_model, initval_block_range=anchor_range)
    if context == "endval":
        return replace(included_model, endval_block_range=anchor_range)
    if context == "shocks":
        return replace(included_model, shocks_block_range=anchor_range)
    return included_model


def _strip_include_instance_suffix(path_key: str) -> str:
    """Remove synthetic repeated-include suffixes such as ``#2``."""
    base, sep, suffix = path_key.rpartition("#")
    if sep and suffix.isdigit() and not Path(path_key).exists():
        return base
    return path_key


def _display_path_name(path_key: str) -> str:
    return Path(_strip_include_instance_suffix(path_key)).name


def _path_to_uri(path_key: str) -> str:
    """Convert a WorkspaceIndex absolute-path key to a ``file://`` URI.

    The index stores ``str(Path.resolve())`` keys; LSP consumers expect
    URIs.  Falls back to the raw key if ``Path.as_uri()`` rejects it
    (which only happens for relative paths — the index shouldn't store
    those, but better safe than crashing a handler).
    """
    try:
        return Path(_strip_include_instance_suffix(path_key)).as_uri()
    except ValueError:
        return _strip_include_instance_suffix(path_key)


def _find_declaration_across_workspace_with_model(
    uri: str,
    name: str,
    workspace_index: WorkspaceIndex,
) -> Optional[Tuple[str, VarDeclaration, Optional[ParsedModel]]]:
    """Look up *name*'s declaration in the active file or its includes.

    Returns ``(target_uri, VarDeclaration, ParsedModel)`` for the first match.  The
    active file is searched first so a local declaration always wins
    over one surfaced through ``@#include`` — that matches what Dynare's
    preprocessor would resolve to and keeps go-to-def deterministic even
    when E030 (duplicate declaration) is also being raised.  Returns
    ``None`` when the name is undeclared in the entire transitive graph.
    """
    try:
        included = workspace_index.resolve_all_includes(uri)
    except Exception:
        logger.exception("Cross-file definition lookup failed for %s", uri)
        return None

    model = workspace_index.get_effective_model(uri)
    if model is not None:
        decl = _find_declaration(model, name)
        if decl is not None:
            return uri, decl, model

    for path_key, included_model in included.items():
        if name in _model_local_variables(included_model):
            source_text = (
                getattr(included_model, "original_text", "") or included_model.text
            )
            local_decl = _find_model_local_declaration_in_text(source_text, name)
            if local_decl is not None:
                return _path_to_uri(path_key), local_decl, included_model
        decl = _find_declaration(included_model, name)
        if decl is not None:
            return _path_to_uri(path_key), decl, included_model

    for parent_uri, parent_model, _included_model in _parent_include_contexts(
        uri,
        workspace_index,
    ):
        active_key = _normalize_uri(uri)
        try:
            parent_included = workspace_index.resolve_all_includes(parent_uri)
        except Exception:
            logger.exception(
                "Parent sibling definition lookup failed for %s", parent_uri
            )
            parent_included = {}
        for path_key, sibling_model in parent_included.items():
            real_key = _normalize_uri(_strip_include_instance_suffix(path_key))
            if real_key == active_key:
                continue
            if name in _model_local_variables(sibling_model):
                source_text = (
                    getattr(sibling_model, "original_text", "") or sibling_model.text
                )
                local_decl = _find_model_local_declaration_in_text(source_text, name)
                if local_decl is not None:
                    return _path_to_uri(path_key), local_decl, sibling_model
            decl = _find_declaration(sibling_model, name)
            if decl is not None:
                return _path_to_uri(path_key), decl, sibling_model

        decl = _find_declaration(parent_model, name)
        if decl is not None:
            return parent_uri, decl, parent_model

    return None


def _find_declaration_across_workspace(
    uri: str,
    name: str,
    workspace_index: WorkspaceIndex,
) -> Optional[Tuple[str, VarDeclaration]]:
    hit = _find_declaration_across_workspace_with_model(
        uri,
        name,
        workspace_index,
    )
    if hit is None:
        return None
    target_uri, decl, _model = hit
    return target_uri, decl


@server.feature(lsp.TEXT_DOCUMENT_DEFINITION)
def definition(params: lsp.DefinitionParams) -> Optional[lsp.Location]:
    """Jump to the declaration of a variable or parameter.

    Searches the active file first, then walks ``@#include`` directives
    transitively via the workspace index so a click on an identifier
    declared in an included file lands in that file.
    """
    uri = params.text_document.uri
    with _state_lock:
        model = _document_models.get(uri)
    if model is None:
        return None

    doc = server.workspace.get_text_document(uri)
    lines = doc.source.split("\n")
    if _cursor_inside_comment_or_non_mcp_string(doc.source, lines, params.position):
        return None
    word_match = _word_at_lsp_position(lines, params.position)
    if word_match is None:
        return None
    if _is_on_the_fly_kind_marker(
        lines[params.position.line],
        word_match.start(),
        word_match.end(),
    ):
        return None

    name = word_match.group(1)
    if getattr(model, "source_map", None):
        model_pos = _source_position_to_model_position(
            model,
            doc.source,
            params.position,
        )
        model_word_match = _word_at_position(model.text.split("\n"), model_pos)
        if model_word_match is not None:
            name = model_word_match.group(1)
    if name in _model_local_variables(model) and not _is_model_local_symbol_at_position(
        model,
        doc.source,
        params.position,
        name,
    ):
        return None
    if name in _model_local_variables(model):
        local_decl = _find_model_local_declaration_in_text(
            doc.source, name
        ) or _find_model_local_declaration(model, name)
        if local_decl is not None:
            return lsp.Location(
                uri=uri,
                range=_to_lsp_range_in_text(doc.source, local_decl.range),
            )

    hit = _find_declaration_across_workspace_with_model(
        uri,
        name,
        _workspace_index,
    )
    if hit is None:
        return None

    target_uri, decl, target_model = hit
    return lsp.Location(
        uri=target_uri,
        range=_to_lsp_range_for_model(target_model, decl.range),
    )


# ---------------------------------------------------------------------------
# Auto-completion — variables, parameters, keywords, built-ins
# ---------------------------------------------------------------------------

# Dynare block keywords with descriptions
_DYNARE_KEYWORDS = [
    ("var", "Declare endogenous variables"),
    ("varexo", "Declare exogenous variables"),
    ("parameters", "Declare parameters"),
    ("model", "Begin model equation block"),
    ("end", "End a block"),
    ("steady_state_model", "Define steady state computation"),
    ("initval", "Set initial values for steady state computation"),
    ("endval", "Set terminal values"),
    ("shocks", "Define shock processes"),
    ("steady", "Compute the steady state"),
    ("check", "Check Blanchard-Kahn conditions"),
    ("stoch_simul", "Compute stochastic simulation"),
    ("simul", "Compute deterministic simulation"),
    ("resid", "Compute residuals of model equations"),
    ("estimated_params", "Define parameters to estimate"),
    ("estimation", "Run Bayesian or ML estimation"),
    ("varobs", "Declare observed variables"),
    ("calib_smoother", "Run the calibrated smoother"),
    ("forecast", "Compute forecasts"),
    ("osr", "Optimal simple rules"),
    ("ramsey_model", "Ramsey optimal policy model"),
    ("planner_objective", "Define planner's objective for Ramsey"),
    ("identification", "Run identification analysis"),
    ("sensitivity", "Run sensitivity analysis"),
]

# Built-in functions available in model equations
_BUILTIN_FUNCTIONS = [
    ("exp", "Exponential function exp(x)"),
    ("log", "Natural logarithm log(x)"),
    ("ln", "Natural logarithm (alias for log)"),
    ("log2", "Base-2 logarithm"),
    ("log10", "Base-10 logarithm"),
    ("sqrt", "Square root sqrt(x)"),
    ("abs", "Absolute value abs(x)"),
    ("sign", "Sign function: -1, 0, or 1"),
    ("sin", "Sine function"),
    ("cos", "Cosine function"),
    ("tan", "Tangent function"),
    ("asin", "Inverse sine"),
    ("acos", "Inverse cosine"),
    ("atan", "Inverse tangent"),
    ("min", "Minimum of two values"),
    ("max", "Maximum of two values"),
    ("erf", "Error function"),
    ("normpdf", "Normal PDF normpdf(x, mu, sigma)"),
    ("normcdf", "Normal CDF normcdf(x, mu, sigma)"),
    ("STEADY_STATE", "Reference steady state value: STEADY_STATE(x)"),
    ("EXPECTATION", "Expectation operator: EXPECTATION(t)(x)"),
]

# ---------------------------------------------------------------------------
# Snippet completion tables (block skeletons + command templates)
#
# These are offered only at top-level context.  Block/command labels match the
# ``.mod`` word pattern ([A-Za-z_]\w*), so plain ``insert_text`` is replaced
# correctly by the client's default word range.  Macro-pair labels show as
# ``@#for``/``@#if`` but filter on a plain word (``for``/``if``): the editor
# suppresses completion once ``@#`` is typed (it reads as a comment), so the
# snippet is reached by the bare word and the default word range is replaced
# cleanly (no leading-``@#`` duplication).  ``$0``/``${n:default}`` are LSP
# snippet tab stops.
# ---------------------------------------------------------------------------

# (label, snippet body, detail).  Indented body, cursor parked at $0.
_BLOCK_SNIPPETS: List[Tuple[str, str, str]] = [
    ("model", "model;\n    $0\nend;", "model equation block"),
    ("initval", "initval;\n    $0\nend;", "initial values for the steady state"),
    ("endval", "endval;\n    $0\nend;", "terminal values block"),
    ("histval", "histval;\n    $0\nend;", "historical values block"),
    ("shocks", "shocks;\n    $0\nend;", "shock specification block"),
    (
        "steady_state_model",
        "steady_state_model;\n    $0\nend;",
        "closed-form steady state block",
    ),
    (
        "estimated_params",
        "estimated_params;\n    $0\nend;",
        "estimated parameters block",
    ),
    (
        "observation_trends",
        "observation_trends;\n    $0\nend;",
        "observation trends block",
    ),
    (
        "ramsey_constraints",
        "ramsey_constraints;\n    $0\nend;",
        "Ramsey constraints block",
    ),
]

# Macro-pair skeletons.  (label, filter word, body, detail): the displayed
# label is ``@#for``/``@#if`` but filtering/replacement keys off the bare word.
_MACRO_SNIPPETS: List[Tuple[str, str, str, str]] = [
    ("@#if", "if", "@#if ${1:condition}\n    $0\n@#endif", "macro conditional"),
    ("@#for", "for", "@#for ${1:i} in ${2:1:N}\n    $0\n@#endfor", "macro loop"),
]

# (label, snippet body, detail, option names used).  ``options`` is validated
# against the catalog by the tests so a template never invents an option name.
_COMMAND_SNIPPETS: List[Tuple[str, str, str, Tuple[str, ...]]] = [
    (
        "stoch_simul",
        "stoch_simul(order = ${1:1}, irf = ${2:20})${3: var_list};",
        "stochastic simulation + IRFs",
        ("order", "irf"),
    ),
    (
        "estimation",
        "estimation(datafile = ${1:data}, mode_compute = ${2:4}, "
        "mh_replic = ${3:5000})${4: var_list};",
        "Bayesian / ML estimation",
        ("datafile", "mode_compute", "mh_replic"),
    ),
    (
        "forecast",
        "forecast(periods = ${1:40});",
        "unconditional forecast",
        ("periods",),
    ),
    (
        "ramsey_model",
        "ramsey_model(planner_discount = ${1:0.99}, instruments = (${2:r}));",
        "Ramsey optimal-policy model",
        ("planner_discount", "instruments"),
    ),
    (
        "perfect_foresight_setup",
        "perfect_foresight_setup(periods = ${1:100});",
        "deterministic simulation setup",
        ("periods",),
    ),
    (
        "perfect_foresight_solver",
        "perfect_foresight_solver;",
        "solve the deterministic model",
        (),
    ),
    ("steady", "steady;", "compute the steady state", ()),
    ("check", "check;", "check Blanchard-Kahn conditions", ()),
    ("varobs", "varobs ${1:obs_vars};", "declare observed variables", ()),
]


def _client_supports_snippets() -> bool:
    """Whether the connected client advertised ``completionItem.snippetSupport``.

    Defaults to ``True`` when the capability cannot be determined (mainstream
    clients support snippets, and direct in-process calls carry no
    capabilities object), and only suppresses snippets when a client has
    explicitly declared ``snippetSupport = false``.
    """
    caps = getattr(server, "client_capabilities", None)
    try:
        td = getattr(caps, "text_document", None)
        comp = getattr(td, "completion", None)
        item = getattr(comp, "completion_item", None)
        support = getattr(item, "snippet_support", None)
        if support is None and isinstance(item, dict):
            support = item.get("snippetSupport")
        return True if support is None else bool(support)
    except Exception:
        return True


def _snippet_completion_items() -> List[lsp.CompletionItem]:
    """Build block-skeleton, command-template, and macro-pair snippet items.

    Offered only at top level (the caller gates on context).  All use plain
    ``insert_text`` so the client's default word range is replaced on accept;
    macro pairs additionally set ``filter_text`` to a bare word.
    """
    items: List[lsp.CompletionItem] = []

    for label, body, detail in _BLOCK_SNIPPETS:
        items.append(
            lsp.CompletionItem(
                label=label,
                kind=lsp.CompletionItemKind.Snippet,
                detail=f"{detail} (snippet)",
                insert_text=body,
                insert_text_format=lsp.InsertTextFormat.Snippet,
                sort_text=f"7_{label}",
            )
        )

    for label, body, detail, _opts in _COMMAND_SNIPPETS:
        items.append(
            lsp.CompletionItem(
                label=label,
                kind=lsp.CompletionItemKind.Snippet,
                detail=f"{detail} (snippet)",
                insert_text=body,
                insert_text_format=lsp.InsertTextFormat.Snippet,
                sort_text=f"8_{label}",
            )
        )

    for label, filter_word, body, detail in _MACRO_SNIPPETS:
        items.append(
            lsp.CompletionItem(
                label=label,
                kind=lsp.CompletionItemKind.Snippet,
                detail=f"{detail} (snippet)",
                filter_text=filter_word,
                insert_text=body,
                insert_text_format=lsp.InsertTextFormat.Snippet,
                sort_text=f"9_{label}",
            )
        )

    return items


def _determine_context(lines: List[str], pos: lsp.Position) -> str:
    """Determine the Dynare block context at the cursor position.

    Returns one of: 'model', 'steady_state_model', 'initval', 'shocks',
    'parameters', or 'top_level'.
    """
    pos = _lsp_position_to_source_position(lines, pos)
    block_stack: List[str] = []
    block_keywords = {
        "model",
        "steady_state_model",
        "initval",
        "endval",
        "histval",
        "shocks",
    }

    token_re = re.compile(
        r"\b(model|steady_state_model|initval|endval|histval|shocks|end)\b\s*(?:\([^)]*\)\s*)?;",
        re.IGNORECASE,
    )

    prefix_lines = list(lines[: pos.line])
    if pos.line < len(lines):
        prefix_lines.append(lines[pos.line][: pos.character])
    prefix = "\n".join(prefix_lines)

    def _blank(match: re.Match) -> str:
        return re.sub(r"\S", " ", match.group(0))

    masked = _STRING_LITERAL_RE.sub(_blank, prefix)
    masked = _BLOCK_COMMENT_RE.sub(_blank, masked)
    masked = _LINE_COMMENT_RE.sub(_blank, masked)
    masked = re.sub(r"(?m)^[ \t]*@\#.*$", _blank, masked)

    for match in token_re.finditer(masked):
        token = match.group(1).lower()
        if token == "end":
            if block_stack:
                block_stack.pop()
        elif token in block_keywords:
            block_stack.append(token)

    return block_stack[-1] if block_stack else "top_level"


def _command_option_context(lines: List[str], pos: lsp.Position) -> Optional[str]:
    """Return the command name if the cursor sits inside its ``(option list)``.

    e.g. with the cursor inside ``stoch_simul(order=1, |)`` this returns
    ``"stoch_simul"`` so completion can offer that command's options.  Returns
    None when the innermost open parenthesis is not a known command call (or
    there is none), so normal context-based completion takes over.
    """
    from .dynare_commands import COMMAND_OPTIONS

    pos = _lsp_position_to_source_position(lines, pos)
    prefix_lines = list(lines[: pos.line])
    if pos.line < len(lines):
        prefix_lines.append(lines[pos.line][: pos.character])
    prefix = "\n".join(prefix_lines)

    def _blank(match: re.Match) -> str:
        return re.sub(r"\S", " ", match.group(0))

    masked = _STRING_LITERAL_RE.sub(_blank, prefix)
    masked = _BLOCK_COMMENT_RE.sub(_blank, masked)
    masked = _LINE_COMMENT_RE.sub(_blank, masked)
    masked = re.sub(r"(?m)^[ \t]*@\#.*$", _blank, masked)

    # Only consider the current (unterminated) statement.
    segment = masked[masked.rfind(";") + 1 :]

    owners: List[str] = []
    for i, ch in enumerate(segment):
        if ch == "(":
            j = i - 1
            while j >= 0 and segment[j].isspace():
                j -= 1
            k = j
            while k >= 0 and (segment[k].isalnum() or segment[k] == "_"):
                k -= 1
            owners.append(segment[k + 1 : j + 1])
        elif ch == ")" and owners:
            owners.pop()

    if owners and owners[-1].lower() in COMMAND_OPTIONS:
        return owners[-1].lower()
    return None


# Prior-distribution shapes valid inside an ``estimated_params`` block (the
# ``*_pdf`` forms Dynare's lexer accepts there — see DynareFlex.ll DYNARE_BLOCK).
_PRIOR_DISTRIBUTIONS = [
    ("beta_pdf", "Beta prior — bounded support, for ratios/probabilities"),
    ("gamma_pdf", "Gamma prior — positive support"),
    ("normal_pdf", "Normal (Gaussian) prior"),
    ("inv_gamma_pdf", "Inverse-gamma prior (type 1)"),
    ("inv_gamma1_pdf", "Inverse-gamma type-1 prior"),
    ("inv_gamma2_pdf", "Inverse-gamma type-2 prior"),
    ("uniform_pdf", "Uniform prior over [lower, upper]"),
    ("weibull_pdf", "Weibull prior — positive support"),
]


def _estimated_params_completion_items(
    model: ParsedModel,
) -> List[lsp.CompletionItem]:
    """Completion items offered inside an ``estimated_params`` block: prior
    distribution shapes, the ``stderr``/``corr`` keywords, and estimable
    symbols (parameters, shocks, observed variables)."""
    items: List[lsp.CompletionItem] = []
    for label, doc_str in _PRIOR_DISTRIBUTIONS:
        items.append(
            lsp.CompletionItem(
                label=label,
                kind=lsp.CompletionItemKind.EnumMember,
                detail="prior distribution",
                documentation=doc_str,
                sort_text=f"0_{label}",
            )
        )
    for kw, doc_str in (
        ("stderr", "Estimate the standard error of a shock or measurement error"),
        ("corr", "Estimate the correlation between two shocks"),
    ):
        items.append(
            lsp.CompletionItem(
                label=kw,
                kind=lsp.CompletionItemKind.Keyword,
                detail="estimated_params keyword",
                documentation=doc_str,
                sort_text=f"1_{kw}",
            )
        )
    for v in model.parameters:
        items.append(
            lsp.CompletionItem(
                label=v.name,
                kind=lsp.CompletionItemKind.Constant,
                detail="parameter",
                sort_text=f"2_{v.name}",
            )
        )
    for v in model.exogenous:
        items.append(
            lsp.CompletionItem(
                label=v.name,
                kind=lsp.CompletionItemKind.Variable,
                detail="exogenous variable (shock)",
                sort_text=f"3_{v.name}",
            )
        )
    for v in model.endogenous:
        items.append(
            lsp.CompletionItem(
                label=v.name,
                kind=lsp.CompletionItemKind.Variable,
                detail="endogenous variable (measurement error)",
                sort_text=f"4_{v.name}",
            )
        )
    return items


@server.feature(
    lsp.TEXT_DOCUMENT_COMPLETION,
    lsp.CompletionOptions(trigger_characters=[".", "_"]),
)
def completion(params: lsp.CompletionParams) -> Optional[lsp.CompletionList]:
    """Provide context-aware auto-completion.

    LLM-friendly: suggests only valid identifiers for the current context,
    preventing hallucinated variable names in generated Dynare code.
    """
    uri = params.text_document.uri
    with _state_lock:
        model = _document_models.get(uri)
    if model is None:
        return None

    doc = server.workspace.get_text_document(uri)
    lines = doc.source.split("\n")
    if _cursor_inside_comment_or_non_mcp_string(doc.source, lines, params.position):
        return None

    # Option-aware completion inside a command's parenthesised option list,
    # e.g. stoch_simul(order=1, <cursor>).  Validation of options is left to
    # the preprocessor; here we only offer the known option names.
    command = _command_option_context(lines, params.position)
    if command is not None:
        from .dynare_commands import command_options

        option_items = [
            lsp.CompletionItem(
                label=option,
                kind=lsp.CompletionItemKind.Property,
                detail=f"{command} option",
                documentation=doc_str,
                sort_text=f"0_{option}",
            )
            for option, doc_str in command_options(command)
        ]
        if option_items:
            return lsp.CompletionList(is_incomplete=False, items=option_items)

    context = _determine_context(lines, params.position)

    items: List[lsp.CompletionItem] = []
    included_decls = _collect_included_declarations(uri, _workspace_index)
    feature_model = model
    include_models_for_values = list(
        _workspace_index.resolve_all_includes(uri).values()
    )
    open_include_context = _open_include_feature_context(
        uri, model, source_text=doc.source
    )
    if open_include_context is not None:
        context, feature_model, include_models_for_values = open_include_context
        included_decls = []

    # Inside an estimated_params block, offer prior-distribution shapes plus the
    # estimable symbols (this vocabulary is not available anywhere else).
    ep_range = model.estimated_params_range
    if ep_range is not None:
        ep_lsp = _to_lsp_range_for_model(model, ep_range)
        if ep_lsp.start.line < params.position.line <= ep_lsp.end.line:
            return lsp.CompletionList(
                is_incomplete=False,
                items=_estimated_params_completion_items(feature_model),
            )

    if context in ("model", "steady_state_model"):
        # In model/SS blocks: offer all variables, parameters, and built-in functions
        vals = _assigned_values(
            feature_model,
            include_models_for_values,
        )

        for v in feature_model.endogenous:
            items.append(
                lsp.CompletionItem(
                    label=v.name,
                    kind=lsp.CompletionItemKind.Variable,
                    detail="endogenous variable",
                    documentation="Declared with var. Use time subscripts: "
                    f"{v.name}(-1) for lag, {v.name}(+1) for lead.",
                    sort_text=f"0_{v.name}",
                )
            )

        for v in feature_model.exogenous:
            items.append(
                lsp.CompletionItem(
                    label=v.name,
                    kind=lsp.CompletionItemKind.Variable,
                    detail="exogenous variable",
                    documentation="Declared with varexo. Value is 0 at steady state.",
                    sort_text=f"1_{v.name}",
                )
            )

        for v in feature_model.parameters:
            val_str = f" = {vals[v.name]:.6g}" if v.name in vals else ""
            items.append(
                lsp.CompletionItem(
                    label=v.name,
                    kind=lsp.CompletionItemKind.Constant,
                    detail=f"parameter{val_str}",
                    documentation="Declared with parameters.",
                    sort_text=f"2_{v.name}",
                )
            )

        for name in sorted(_model_local_variables(feature_model)):
            items.append(
                lsp.CompletionItem(
                    label=name,
                    kind=lsp.CompletionItemKind.Variable,
                    detail="model-local variable",
                    documentation="Defined with # inside the model block.",
                    sort_text=f"2_local_{name}",
                )
            )

        for name, doc_str in _BUILTIN_FUNCTIONS:
            items.append(
                lsp.CompletionItem(
                    label=name,
                    kind=lsp.CompletionItemKind.Function,
                    detail="built-in function",
                    documentation=doc_str,
                    sort_text=f"3_{name}",
                )
            )

        # Cross-file: include declarations from @#include'd files,
        # sorted after the local declarations and built-ins.
        for class_label, v, _t_uri, src_name, value in included_decls:
            if class_label == "endogenous":
                items.append(
                    lsp.CompletionItem(
                        label=v.name,
                        kind=lsp.CompletionItemKind.Variable,
                        detail=f"endogenous variable (from {src_name})",
                        sort_text=f"4_{v.name}",
                    )
                )
            elif class_label == "exogenous":
                items.append(
                    lsp.CompletionItem(
                        label=v.name,
                        kind=lsp.CompletionItemKind.Variable,
                        detail=f"exogenous variable (from {src_name})",
                        sort_text=f"5_{v.name}",
                    )
                )
            elif class_label == "parameter":
                val_str = f" = {value:.6g}" if value is not None else ""
                items.append(
                    lsp.CompletionItem(
                        label=v.name,
                        kind=lsp.CompletionItemKind.Constant,
                        detail=f"parameter{val_str} (from {src_name})",
                        sort_text=f"6_{v.name}",
                    )
                )
            elif class_label == "model-local":
                items.append(
                    lsp.CompletionItem(
                        label=v.name,
                        kind=lsp.CompletionItemKind.Variable,
                        detail=f"model-local variable (from {src_name})",
                        documentation="Defined with # inside an included model block.",
                        sort_text=f"4_local_{v.name}",
                    )
                )

    elif context in ("initval", "endval", "histval"):
        # In initval/endval/histval: offer endogenous and exogenous variables
        for v in feature_model.endogenous:
            items.append(
                lsp.CompletionItem(
                    label=v.name,
                    kind=lsp.CompletionItemKind.Variable,
                    detail="endogenous variable",
                    sort_text=f"0_{v.name}",
                )
            )
        for v in feature_model.exogenous:
            items.append(
                lsp.CompletionItem(
                    label=v.name,
                    kind=lsp.CompletionItemKind.Variable,
                    detail="exogenous variable",
                    sort_text=f"1_{v.name}",
                )
            )
        for class_label, v, _t_uri, src_name, _value in included_decls:
            if class_label == "endogenous":
                items.append(
                    lsp.CompletionItem(
                        label=v.name,
                        kind=lsp.CompletionItemKind.Variable,
                        detail=f"endogenous variable (from {src_name})",
                        sort_text=f"2_{v.name}",
                    )
                )
            elif class_label == "exogenous":
                items.append(
                    lsp.CompletionItem(
                        label=v.name,
                        kind=lsp.CompletionItemKind.Variable,
                        detail=f"exogenous variable (from {src_name})",
                        sort_text=f"3_{v.name}",
                    )
                )

    elif context == "shocks":
        # In shocks block: offer exogenous variables
        for v in feature_model.exogenous:
            items.append(
                lsp.CompletionItem(
                    label=v.name,
                    kind=lsp.CompletionItemKind.Variable,
                    detail="exogenous variable",
                    sort_text=f"0_{v.name}",
                )
            )
        for class_label, v, _t_uri, src_name, _value in included_decls:
            if class_label == "exogenous":
                items.append(
                    lsp.CompletionItem(
                        label=v.name,
                        kind=lsp.CompletionItemKind.Variable,
                        detail=f"exogenous variable (from {src_name})",
                        sort_text=f"1_{v.name}",
                    )
                )

    else:
        # Top level: offer keywords and all declared names
        for kw, doc_str in _DYNARE_KEYWORDS:
            items.append(
                lsp.CompletionItem(
                    label=kw,
                    kind=lsp.CompletionItemKind.Keyword,
                    detail="Dynare keyword",
                    documentation=doc_str,
                    sort_text=f"0_{kw}",
                )
            )
        for v in feature_model.endogenous:
            items.append(
                lsp.CompletionItem(
                    label=v.name,
                    kind=lsp.CompletionItemKind.Variable,
                    detail="endogenous variable",
                    sort_text=f"1_{v.name}",
                )
            )
        for v in feature_model.exogenous:
            items.append(
                lsp.CompletionItem(
                    label=v.name,
                    kind=lsp.CompletionItemKind.Variable,
                    detail="exogenous variable",
                    sort_text=f"2_{v.name}",
                )
            )
        for v in feature_model.parameters:
            items.append(
                lsp.CompletionItem(
                    label=v.name,
                    kind=lsp.CompletionItemKind.Constant,
                    detail="parameter",
                    sort_text=f"3_{v.name}",
                )
            )
        for class_label, v, _t_uri, src_name, _value in included_decls:
            if class_label == "endogenous":
                kind = lsp.CompletionItemKind.Variable
                detail = f"endogenous variable (from {src_name})"
                sort_prefix = "4"
            elif class_label == "exogenous":
                kind = lsp.CompletionItemKind.Variable
                detail = f"exogenous variable (from {src_name})"
                sort_prefix = "5"
            else:
                kind = lsp.CompletionItemKind.Constant
                detail = f"parameter (from {src_name})"
                sort_prefix = "6"
            items.append(
                lsp.CompletionItem(
                    label=v.name,
                    kind=kind,
                    detail=detail,
                    sort_text=f"{sort_prefix}_{v.name}",
                )
            )

        # Block-skeleton, macro, and command-template snippets at top level.
        if context == "top_level" and _client_supports_snippets():
            items.extend(_snippet_completion_items())

    return lsp.CompletionList(is_incomplete=False, items=items) if items else None


# ---------------------------------------------------------------------------
# Code actions — quick fixes for diagnostics
# ---------------------------------------------------------------------------


@server.feature(lsp.TEXT_DOCUMENT_CODE_ACTION)
def code_action(params: lsp.CodeActionParams) -> Optional[List[lsp.CodeAction]]:
    """Provide quick-fix code actions for diagnostics.

    LLM-friendly: returns structured edit suggestions that LLMs can
    directly apply to fix common Dynare errors.
    """
    uri = params.text_document.uri
    with _state_lock:
        model = _document_models.get(uri)
        diagnostics = _document_diagnostics.get(uri, [])
    if model is None:
        return None

    doc = server.workspace.get_text_document(uri)
    lines = doc.source.split("\n")
    actions: List[lsp.CodeAction] = []
    request_range = params.range
    requested_kinds = [str(kind) for kind in (params.context.only or [])]

    def _kind_allowed(kind: lsp.CodeActionKind) -> bool:
        if not requested_kinds:
            return True
        kind_str = str(kind)
        return any(
            kind_str == requested or kind_str.startswith(f"{requested}.")
            for requested in requested_kinds
        )

    def _range_overlaps(left: lsp.Range, right: lsp.Range) -> bool:
        left_start = (left.start.line, left.start.character)
        left_end = (left.end.line, left.end.character)
        right_start = (right.start.line, right.start.character)
        right_end = (right.end.line, right.end.character)
        if right_start == right_end:
            return left_start <= right_start <= left_end
        return left_start < right_end and right_start < left_end

    for diag in diagnostics:
        diag_range = _to_lsp_range_in_text(doc.source, diag.range)

        # Only return actions for diagnostics in the requested range
        if not _range_overlaps(diag_range, request_range):
            continue

        lsp_diag = _to_lsp_diagnostic(diag, doc.source)

        if diag.fix is not None and _kind_allowed(lsp.CodeActionKind.QuickFix):
            fix = diag.fix

            def _fix_pos(line_no: int, char_no: int) -> DPos:
                if line_no >= len(lines):
                    return DPos(len(lines) - 1, len(lines[-1].rstrip("\r")))
                line_no = max(0, min(line_no, len(lines) - 1))
                line_len = len(lines[line_no].rstrip("\r"))
                char_no = max(0, min(char_no, line_len))
                return DPos(line=line_no, character=char_no)

            fix_range = _to_lsp_range_in_text(
                doc.source,
                DRange(
                    start=_fix_pos(fix.start_line, fix.start_char),
                    end=_fix_pos(fix.end_line, fix.end_char),
                ),
            )
            if not _range_overlaps(fix_range, request_range):
                if diag.code == "E020":
                    continue
            title = f"Apply fix for {diag.code}"
            fix_match = re.search(r"Fix:\s*(.+)$", diag.message)
            if fix_match:
                title = fix_match.group(1).strip()
                if len(title) > 120:
                    title = title[:117].rstrip() + "..."
            actions.append(
                lsp.CodeAction(
                    title=title,
                    kind=lsp.CodeActionKind.QuickFix,
                    diagnostics=[lsp_diag],
                    is_preferred=True,
                    edit=lsp.WorkspaceEdit(
                        changes={
                            uri: [
                                lsp.TextEdit(
                                    range=fix_range,
                                    new_text=fix.new_text,
                                )
                            ]
                        }
                    ),
                )
            )
            if diag.code == "E020":
                continue

        if diag.fix is not None:
            pass
        elif diag.code == "E020" and _kind_allowed(lsp.CodeActionKind.QuickFix):
            if "likely a typo" in diag.message:
                continue
            # Undeclared identifier - offer to add to var, varexo, or parameters
            match = re.search(r"Undeclared identifier '(\w+)'", diag.message)
            if match:
                name = match.group(1)
                for decl_kw, decl_label in [
                    ("var", "endogenous variable"),
                    ("varexo", "exogenous variable"),
                    ("parameters", "parameter"),
                ]:
                    insert_text, insert_pos = _find_declaration_insert(
                        lines, decl_kw, name
                    )
                    if insert_text is not None:
                        insert_lsp_pos = _source_position_to_lsp_position(
                            lines,
                            DPos(insert_pos.line, insert_pos.character),
                        )
                        actions.append(
                            lsp.CodeAction(
                                title=f"Declare '{name}' as {decl_label} ({decl_kw})",
                                kind=lsp.CodeActionKind.QuickFix,
                                diagnostics=[lsp_diag],
                                edit=lsp.WorkspaceEdit(
                                    changes={
                                        uri: [
                                            lsp.TextEdit(
                                                range=lsp.Range(
                                                    start=insert_lsp_pos,
                                                    end=insert_lsp_pos,
                                                ),
                                                new_text=insert_text,
                                            )
                                        ]
                                    }
                                ),
                            )
                        )

        elif diag.code == "W010" and _kind_allowed(lsp.CodeActionKind.QuickFix):
            # Unassigned parameter - offer to add assignment
            match = re.search(r"Parameter '(\w+)'", diag.message)
            if match:
                name = match.group(1)
                # Insert after the last parameter assignment or after parameters declaration
                insert_line = _find_param_assign_insert_line(model, lines)
                insert_pos, insert_text = _valid_line_insert(
                    lines,
                    insert_line,
                    f"{name} = 0;\n",
                )
                actions.append(
                    lsp.CodeAction(
                        title=f"Add assignment: {name} = 0;",
                        kind=lsp.CodeActionKind.QuickFix,
                        diagnostics=[lsp_diag],
                        edit=lsp.WorkspaceEdit(
                            changes={
                                uri: [
                                    lsp.TextEdit(
                                        range=lsp.Range(
                                            start=insert_pos,
                                            end=insert_pos,
                                        ),
                                        new_text=insert_text,
                                    )
                                ]
                            }
                        ),
                    )
                )

        elif diag.code == "W042" and _kind_allowed(lsp.CodeActionKind.QuickFix):
            # Missing SS variable - offer to add to steady_state_model.
            # IMPORTANT: rebuild the missing-variable list from the model
            # itself rather than parsing the (potentially truncated)
            # diagnostic message.  The W042 message may render
            # ``y1, y2, ... (and 3 more)`` for long lists; previously
            # that ``(and 3 more)`` string got inserted as a literal
            # variable name and produced invalid Dynare.
            #
            # Equations expose ``lhs`` (the variable being assigned) and
            # ``name`` (the optional equation tag).  Use ``lhs`` - using
            # ``name`` would treat virtually every SS equation as
            # un-named and report all endogenous as missing, which would
            # then re-insert ``y = 0; c = 0; ...`` on top of the existing
            # SS assignments and corrupt them.
            ss_block_vars = {
                e.lhs.strip()
                for e in model.steady_state_equations
                if getattr(e, "lhs", "")
            }
            missing_vars = [
                v.name for v in model.endogenous if v.name not in ss_block_vars
            ]
            if missing_vars and model.steady_state_block_range:
                # Insert before 'end;' of steady_state_model
                end_line = model.steady_state_block_range.end.line
                insert = "\n".join(f"    {v} = 0;" for v in missing_vars[:5]) + "\n"
                insert_pos = lsp.Position(line=end_line, character=0)
                if model.steady_state_block_range.start.line == end_line:
                    line_text = lines[end_line] if end_line < len(lines) else ""
                    start_char = model.steady_state_block_range.start.character
                    end_match = re.search(
                        r"\bend\s*;",
                        line_text[start_char:],
                        re.IGNORECASE,
                    )
                    if end_match:
                        insert_pos = lsp.Position(
                            line=end_line,
                            character=start_char + end_match.start(),
                        )
                        insert = "\n" + insert
                insert_pos = _source_position_to_lsp_position(
                    lines,
                    DPos(insert_pos.line, insert_pos.character),
                )
                actions.append(
                    lsp.CodeAction(
                        title="Add missing variables to steady_state_model",
                        kind=lsp.CodeActionKind.QuickFix,
                        diagnostics=[lsp_diag],
                        edit=lsp.WorkspaceEdit(
                            changes={
                                uri: [
                                    lsp.TextEdit(
                                        range=lsp.Range(
                                            start=insert_pos,
                                            end=insert_pos,
                                        ),
                                        new_text=insert,
                                    )
                                ]
                            }
                        ),
                    )
                )

        elif diag.code == "W120" and _kind_allowed(lsp.CodeActionKind.QuickFix):
            # No exogenous for a stochastic command — insert a dummy varexo.
            insert_text, insert_pos = _find_declaration_insert(
                lines,
                "varexo",
                "dummy_e",
            )
            if insert_text is not None:
                insert_lsp = _source_position_to_lsp_position(
                    lines,
                    DPos(insert_pos.line, insert_pos.character),
                )
                actions.append(
                    lsp.CodeAction(
                        title="Declare a dummy exogenous variable (varexo dummy_e)",
                        kind=lsp.CodeActionKind.QuickFix,
                        diagnostics=[lsp_diag],
                        is_preferred=True,
                        edit=lsp.WorkspaceEdit(
                            changes={
                                uri: [
                                    lsp.TextEdit(
                                        range=lsp.Range(
                                            start=insert_lsp, end=insert_lsp
                                        ),
                                        new_text=insert_text,
                                    )
                                ]
                            }
                        ),
                    )
                )

        elif diag.code == "W121" and _kind_allowed(lsp.CodeActionKind.QuickFix):
            # Parameter used with a lead/lag — strip the time index.
            match = re.search(r"Parameter '(\w+)'", diag.message)
            if match:
                name = match.group(1)
                pattern = re.compile(
                    r"\b" + re.escape(name) + r"\s*\(\s*[+-]?\d+\s*\)",
                )
                last = min(diag_range.end.line, len(lines) - 1)
                for line_no in range(diag_range.start.line, last + 1):
                    hit = pattern.search(lines[line_no])
                    if hit is None:
                        continue
                    start_lsp = _source_position_to_lsp_position(
                        lines,
                        DPos(line_no, hit.start()),
                    )
                    end_lsp = _source_position_to_lsp_position(
                        lines,
                        DPos(line_no, hit.end()),
                    )
                    actions.append(
                        lsp.CodeAction(
                            title=f"Remove time index from parameter '{name}'",
                            kind=lsp.CodeActionKind.QuickFix,
                            diagnostics=[lsp_diag],
                            is_preferred=True,
                            edit=lsp.WorkspaceEdit(
                                changes={
                                    uri: [
                                        lsp.TextEdit(
                                            range=lsp.Range(
                                                start=start_lsp, end=end_lsp
                                            ),
                                            new_text=name,
                                        )
                                    ]
                                }
                            ),
                        )
                    )
                    break

        elif diag.code == "W122" and _kind_allowed(lsp.CodeActionKind.QuickFix):
            # Non-finite parameter — replace its value with 0.
            match = re.search(r"Parameter '(\w+)'", diag.message)
            if match:
                name = match.group(1)
                line_no = diag_range.start.line
                if line_no < len(lines):
                    line_text = lines[line_no]
                    assign = re.match(
                        r"(\s*" + re.escape(name) + r"\s*=\s*)([^;]*)(.*)",
                        line_text,
                    )
                    if assign:
                        end_char = _source_index_to_lsp_character(
                            line_text,
                            len(line_text),
                        )
                        actions.append(
                            lsp.CodeAction(
                                title=f"Set parameter '{name}' to 0",
                                kind=lsp.CodeActionKind.QuickFix,
                                diagnostics=[lsp_diag],
                                is_preferred=True,
                                edit=lsp.WorkspaceEdit(
                                    changes={
                                        uri: [
                                            lsp.TextEdit(
                                                range=lsp.Range(
                                                    start=lsp.Position(
                                                        line=line_no, character=0
                                                    ),
                                                    end=lsp.Position(
                                                        line=line_no, character=end_char
                                                    ),
                                                ),
                                                new_text=assign.group(1)
                                                + "0"
                                                + assign.group(3),
                                            )
                                        ]
                                    }
                                ),
                            )
                        )

        elif diag.code == "W112" and _kind_allowed(lsp.CodeActionKind.QuickFix):
            # Negative variance — make the right-hand side non-negative.
            match = re.search(r"Shock '(\w+)'", diag.message)
            if match:
                name = match.group(1)
                pattern = re.compile(
                    r"(\bvar\s+" + re.escape(name) + r"\s*=\s*)([^;]+)",
                    re.IGNORECASE,
                )
                last = min(diag_range.end.line, len(lines) - 1)
                for line_no in range(diag_range.start.line, last + 1):
                    hit = pattern.search(lines[line_no])
                    if hit is None:
                        continue
                    rhs = hit.group(2).strip()
                    fixed = rhs[1:].strip() if rhs.startswith("-") else f"abs({rhs})"
                    start_lsp = _source_position_to_lsp_position(
                        lines,
                        DPos(line_no, hit.start(2)),
                    )
                    end_lsp = _source_position_to_lsp_position(
                        lines,
                        DPos(line_no, hit.end(2)),
                    )
                    actions.append(
                        lsp.CodeAction(
                            title=f"Make the variance of '{name}' non-negative",
                            kind=lsp.CodeActionKind.QuickFix,
                            diagnostics=[lsp_diag],
                            is_preferred=True,
                            edit=lsp.WorkspaceEdit(
                                changes={
                                    uri: [
                                        lsp.TextEdit(
                                            range=lsp.Range(
                                                start=start_lsp, end=end_lsp
                                            ),
                                            new_text=fixed,
                                        )
                                    ]
                                }
                            ),
                        )
                    )
                    break

        elif diag.code == "E010" and _kind_allowed(lsp.CodeActionKind.QuickFix):
            # Equation count mismatch - informational action
            actions.append(
                lsp.CodeAction(
                    title="Equation count mismatch (informational)",
                    kind=lsp.CodeActionKind.QuickFix,
                    diagnostics=[lsp_diag],
                    is_preferred=False,
                )
            )

        elif diag.code == "I050" and _kind_allowed(lsp.CodeActionKind.QuickFix):
            # No steady state - offer to compute
            actions.append(
                lsp.CodeAction(
                    title="Compute steady state (solve numerically)",
                    kind=lsp.CodeActionKind.QuickFix,
                    diagnostics=[lsp_diag],
                    command=lsp.Command(
                        title="Compute Steady State",
                        command="dynare/computeSteadyState",
                        arguments=[uri],
                    ),
                )
            )

        elif diag.code == "E061" and _kind_allowed(lsp.CodeActionKind.QuickFix):
            # Unresolved @#include - offer to create the missing file as
            # an empty stub.  We previously dispatched to a server
            # command (``dynare/createIncludeTarget``) that was never
            # registered, so the action silently no-op'd.  Switch to a
            # CreateFile operation in the WorkspaceEdit so the client
            # actually performs the file creation; ``ignoreIfExists``
            # makes the action idempotent.
            match = re.search(r"'([^']+)'", diag.message)
            if match:
                missing = match.group(1)
                if "(included from" in missing or "@{" in missing or "}" in missing:
                    continue
                # Resolve target path relative to the including file.
                from .include_resolver import _uri_to_path

                if uri.startswith("file://"):
                    including_dir = _uri_to_path(uri)
                else:
                    including_dir = Path(uri)
                if including_dir.suffix or including_dir.is_file():
                    including_dir = including_dir.parent
                target = (including_dir / missing).resolve()
                try:
                    target_uri = target.as_uri()
                except ValueError:
                    target_uri = None
                if target_uri is not None:
                    actions.append(
                        lsp.CodeAction(
                            title=f"Create file: {missing}",
                            kind=lsp.CodeActionKind.QuickFix,
                            diagnostics=[lsp_diag],
                            edit=lsp.WorkspaceEdit(
                                document_changes=[
                                    lsp.CreateFile(
                                        uri=target_uri,
                                        options=lsp.CreateFileOptions(
                                            ignore_if_exists=True,
                                        ),
                                    ),
                                ],
                            ),
                        )
                    )

        elif diag.code == "E062" and _kind_allowed(lsp.CodeActionKind.QuickFix):
            # Unmatched @#if / @#for - offer to insert the closing directive.
            msg = diag.message
            if "@#if" in msg and "@#endif" in msg:
                closer = "@#endif"
            elif "@#for" in msg and "@#endfor" in msg:
                closer = "@#endfor"
            else:
                closer = None
            if closer is not None and msg.startswith("Unterminated"):
                # Insert the closer on a new line at end of file.
                if doc.source:
                    eof_lines = doc.source.split("\n")
                    eof_line = len(eof_lines) - 1
                    eof_char = (
                        0
                        if doc.source.endswith("\n")
                        else _source_index_to_lsp_character(
                            eof_lines[-1],
                            len(eof_lines[-1]),
                        )
                    )
                    new_text = (
                        f"{closer}\n" if doc.source.endswith("\n") else f"\n{closer}\n"
                    )
                else:
                    eof_line = 0
                    eof_char = 0
                    new_text = f"{closer}\n"
                actions.append(
                    lsp.CodeAction(
                        title=f"Insert {closer} at end of file",
                        kind=lsp.CodeActionKind.QuickFix,
                        diagnostics=[lsp_diag],
                        edit=lsp.WorkspaceEdit(
                            changes={
                                uri: [
                                    lsp.TextEdit(
                                        range=lsp.Range(
                                            start=lsp.Position(
                                                line=eof_line, character=eof_char
                                            ),
                                            end=lsp.Position(
                                                line=eof_line, character=eof_char
                                            ),
                                        ),
                                        new_text=new_text,
                                    )
                                ]
                            }
                        ),
                    )
                )

            continue

    # Source action: always available on model block
    if model and model.model_block_range and _kind_allowed(lsp.CodeActionKind.Source):
        model_range = _to_lsp_range_for_model(model, model.model_block_range)
        if (
            request_range.start.line <= model_range.end.line
            and request_range.end.line >= model_range.start.line
        ):
            actions.append(
                lsp.CodeAction(
                    title="Compute steady state",
                    kind=lsp.CodeActionKind.Source,
                    command=lsp.Command(
                        title="Compute Steady State",
                        command="dynare/computeSteadyState",
                        arguments=[uri],
                    ),
                )
            )

    # Update initval with solved values — available on initval block
    with _state_lock:
        solver_result = _document_solver_results.get(uri)
    if (
        _kind_allowed(lsp.CodeActionKind.QuickFix)
        and model
        and model.initval_block_range
        and model.initval_entries
        and solver_result
        and solver_result.success
    ):
        iv_range = _to_lsp_range_for_model(model, model.initval_block_range)
        if (
            request_range.start.line <= iv_range.end.line
            and request_range.end.line >= iv_range.start.line
        ):
            # Build text edits to replace each entry's value
            edits: List[lsp.TextEdit] = []
            for entry in model.initval_entries:
                if entry.name not in solver_result.values:
                    continue
                solved = solver_result.values[entry.name]
                if entry.value is not None and abs(entry.value - solved) < 1e-8:
                    continue  # already correct
                # Format the value
                if solved == int(solved) and abs(solved) < 1e6:
                    val_str = str(int(solved))
                else:
                    val_str = f"{solved:.10g}"
                new_line = f"{entry.name} = {val_str};"
                edits.append(
                    lsp.TextEdit(
                        range=_to_lsp_range_for_model(model, entry.range),
                        new_text=new_line,
                    )
                )
            if edits:
                actions.append(
                    lsp.CodeAction(
                        title=f"Update initval with solved steady state ({len(edits)} values)",
                        kind=lsp.CodeActionKind.QuickFix,
                        is_preferred=True,
                        edit=lsp.WorkspaceEdit(changes={uri: edits}),
                    )
                )

    return actions if actions else None


def _find_declaration_insert(lines: List[str], keyword: str, name: str) -> tuple:
    """Find where to insert a new variable into an existing declaration block.

    Returns (insert_text, insert_position) or (None, None) if the keyword block
    is not found.
    """

    def _mask_declaration_scan_text(text: str) -> str:
        def _blank(match: re.Match) -> str:
            return " " * (match.end() - match.start())

        masked = re.sub(r'"[^"\n]*"|\'[^\'\n]*\'', _blank, text)
        masked = re.sub(r"/\*.*?\*/", _blank, masked, flags=re.DOTALL)
        return re.sub(r"(?://|%).*$", "", masked, flags=re.MULTILINE)

    masked_lines = _mask_declaration_scan_text("\n".join(lines)).split("\n")
    block_depth = 0
    block_keywords = {"model", "steady_state_model", "initval", "endval", "shocks"}

    # Look for existing top-level keyword declaration.  ``var`` is also valid
    # inside a ``shocks`` block, where inserting a model variable would corrupt
    # the shock statement.
    for i, line in enumerate(masked_lines):
        stripped = line.strip().lower()
        if block_depth == 0 and re.match(rf"^{keyword}\b", stripped):
            # Find the semicolon that ends this declaration.  Mask
            # ``//`` and ``%`` comments and string literals so a ``;``
            # inside a same-line comment or quoted equation tag isn't
            # mistaken for the declaration's terminator (which would
            # cause the auto-fix to insert the new variable inside the
            # comment).
            for j in range(i, min(i + 50, len(lines))):
                masked = masked_lines[j]
                semi_pos = masked.find(";")
                if semi_pos >= 0:
                    # Insert the name before the semicolon
                    return (f" {name}", lsp.Position(line=j, character=semi_pos))
            break

        for token in re.finditer(
            r"\b(steady_state_model|model|initval|endval|shocks|end)\s*;",
            stripped,
        ):
            if token.group(1) == "end":
                block_depth = max(0, block_depth - 1)
            elif token.group(1) in block_keywords:
                block_depth += 1

    # Keyword block not found — offer to create it at the top
    return (f"{keyword} {name};\n", lsp.Position(line=0, character=0))


def _find_param_assign_insert_line(model: ParsedModel, lines: List[str]) -> int:
    """Find the line number where a new parameter assignment should be inserted."""
    # After the last parameter assignment
    if model.param_assignments:
        last = model.param_assignments[-1]
        return last.range.end.line + 1

    # After the parameters declaration
    for i, line in enumerate(lines):
        if re.match(r"^\s*parameters\b", line, re.IGNORECASE):
            for j in range(i, min(i + 50, len(lines))):
                # Mask comments/strings so a ``;`` inside them isn't
                # mistaken for the declaration's terminator.
                masked = re.sub(
                    r'"[^"\n]*"|\'[^\'\n]*\'',
                    lambda m: " " * (m.end() - m.start()),
                    lines[j],
                )
                masked = re.sub(r"(?://|%).*$", "", masked)
                if ";" in masked:
                    return j + 1
            return i + 1

    return 0


# ---------------------------------------------------------------------------
# Compute steady state command
# ---------------------------------------------------------------------------


def _format_solver_value(value: float) -> str:
    if value == int(value) and abs(value) < 1e6:
        return str(int(value))
    return f"{value:.10g}"


def _format_initval_block(values: dict, model: ParsedModel) -> str:
    """Format computed values as a Dynare initval block."""
    lines = ["\ninitval;"]
    for v in model.endogenous:
        if v.name in values:
            lines.append(f"    {v.name} = {_format_solver_value(values[v.name])};")
    lines.append("end;\n")
    return "\n".join(lines) + "\n"


def _build_initval_update_edits(
    values: dict,
    model: ParsedModel,
    value_model: Optional[ParsedModel] = None,
) -> List[lsp.TextEdit]:
    """Build edits that update an existing initval block with solver values."""
    if not model.initval_block_range:
        return []

    edits: List[lsp.TextEdit] = []
    seen_entries = set()
    for entry in model.initval_entries:
        seen_entries.add(entry.name)
        if entry.name not in values:
            continue
        solved = values[entry.name]
        if entry.value is not None and abs(entry.value - solved) < 1e-8:
            continue
        edits.append(
            lsp.TextEdit(
                range=_to_lsp_range_for_model(model, entry.range),
                new_text=f"{entry.name} = {_format_solver_value(solved)};",
            )
        )

    missing_lines = []
    for var in (value_model or model).endogenous:
        if var.name in values and var.name not in seen_entries:
            missing_lines.append(
                f"    {var.name} = {_format_solver_value(values[var.name])};\n"
            )
    if missing_lines:
        insert_line = model.initval_block_range.end.line
        insert_character = 0
        if model.initval_block_range.start.line == model.initval_block_range.end.line:
            source_text = getattr(model, "original_text", "") or model.text
            line = source_text.splitlines()[insert_line]
            search_until = min(model.initval_block_range.end.character, len(line))
            matches = list(
                re.finditer(r"\bend\s*;", line[:search_until], re.IGNORECASE)
            )
            if matches:
                insert_character = matches[-1].start()
                if insert_character > 0 and not missing_lines[0].startswith("\n"):
                    missing_lines.insert(0, "\n")
        insert_pos = _source_position_to_lsp_position(
            (getattr(model, "original_text", "") or model.text).split("\n"),
            DPos(insert_line, insert_character),
        )
        edits.append(
            lsp.TextEdit(
                range=lsp.Range(
                    start=insert_pos,
                    end=insert_pos,
                ),
                new_text="".join(missing_lines),
            )
        )

    return edits


def _find_ss_insert_line(model: ParsedModel, lines: List[str]) -> int:
    """Find the best line to insert an initval block."""
    if model.model_block_range:
        return model.model_block_range.end.line + 1
    return len(lines)


def _compute_ss_prepare(*args) -> Optional[dict]:
    """Prepare steady-state edit payload or return a terminal command result."""
    if len(args) == 1 and isinstance(args[0], list):
        args = tuple(args[0])
    if not args:
        return None
    # Accept either a bare URI string (legacy call sites) or a dict like
    # ``{"uri": ...}`` (the CodeLens emitter at server.py:~2374 packs the
    # URI inside an object).  Older code actions still pass the bare
    # string, and breaking either is observable.
    raw = args[0]
    if isinstance(raw, dict):
        uri = raw.get("uri")
    else:
        uri = raw
    if not isinstance(uri, str):
        return {
            "done": {"success": False, "message": "Missing or invalid URI argument"}
        }
    with _state_lock:
        model = _document_models.get(uri)
    if model is None:
        return {"done": {"success": False, "message": "No parsed model available"}}
    version_token = id(model)
    include_models = None
    try:
        include_models = list(_workspace_index.resolve_all_includes(uri).values())
    except Exception:
        logger.exception("Steady state command include analysis failed for %s", uri)
    from .diagnostics import _with_model_editing_commands

    # Apply model_remove / model_replace / var_remove like the diagnostics
    # and MCP paths do — solving the pre-edit model returns values for
    # variables Dynare has removed.
    solve_model = _with_model_editing_commands(
        model_with_include_context(model, include_models)
    )

    try:
        from .solver import compute_steady_state, default_solve_budget
    except ImportError:
        server.window_show_message(
            lsp.ShowMessageParams(
                type=lsp.MessageType.Warning,
                message="scipy is required for steady state computation. "
                "Install with: pip install dynare-lsp[solver]",
            )
        )
        return {"done": {"success": False, "message": "scipy not installed"}}

    result = compute_steady_state(solve_model, time_budget=default_solve_budget())
    with _state_lock:
        current = _document_models.get(uri)
        if current is None or id(current) != version_token:
            return {
                "done": {
                    "success": False,
                    "message": "Document changed before steady state could be applied",
                }
            }
        _document_solver_results[uri] = result
        if result.success:
            _document_warm_start_results[uri] = result

    if not result.success:
        server.window_show_message(
            lsp.ShowMessageParams(
                type=lsp.MessageType.Warning,
                message=f"Steady state computation failed: {result.message}",
            )
        )
        doc = server.workspace.get_text_document(uri)
        _validate_document(uri, doc.source)
        return {"done": {"success": False, "message": result.message}}

    # Build edits: update an existing initval block when present, otherwise
    # insert a fresh block after the model.
    doc = server.workspace.get_text_document(uri)
    lines = doc.source.split("\n")
    if model.initval_block_range is not None:
        edits = _build_initval_update_edits(result.values, model, solve_model)
        if edits:
            edit = lsp.WorkspaceEdit(changes={uri: edits})
            return {
                "done": None,
                "edit": edit,
                "result": result,
                "action": "Updated initval block",
            }
        return {
            "done": {
                "success": True,
                "method": result.method_used,
                "values": result.values,
                "message": "Existing initval block already matches solved values",
            }
        }

    initval_text = _format_initval_block(result.values, solve_model)
    insert_line = _find_ss_insert_line(model, lines)
    insert_pos, initval_text = _valid_line_insert(lines, insert_line, initval_text)

    with _state_lock:
        current = _document_models.get(uri)
        if current is None or id(current) != version_token:
            _document_solver_results.pop(uri, None)
            return {
                "done": {
                    "success": False,
                    "message": "Document changed before steady state could be applied",
                }
            }

    edit = lsp.WorkspaceEdit(
        changes={
            uri: [
                lsp.TextEdit(
                    range=lsp.Range(
                        start=insert_pos,
                        end=insert_pos,
                    ),
                    new_text=initval_text,
                )
            ]
        }
    )
    return {
        "done": None,
        "edit": edit,
        "result": result,
        "action": "Inserted initval block",
    }


def _compute_ss_finish(
    prepared: dict,
    apply_result: object,
) -> dict:
    """Finish a steady-state command after the client answers applyEdit."""
    result = prepared["result"]
    if apply_result is not None and getattr(apply_result, "applied", True) is False:
        reason = getattr(apply_result, "failure_reason", None) or "client rejected edit"
        message = f"Failed to apply steady state edit: {reason}"
        server.window_show_message(
            lsp.ShowMessageParams(
                type=lsp.MessageType.Warning,
                message=message,
            )
        )
        return {"success": False, "message": message}

    server.window_show_message(
        lsp.ShowMessageParams(
            type=lsp.MessageType.Info,
            message=f"Steady state computed ({result.method_used}). "
            f"{prepared.get('action', 'Applied initval edit')} "
            f"with {len(result.values)} values.",
        )
    )

    return {
        "success": True,
        "method": result.method_used,
        "values": result.values,
    }


@server.thread()
def _wait_for_apply_edit_result(apply_result):
    return apply_result.result(timeout=5)


def compute_ss_command(*args) -> Optional[dict]:
    """Execute steady state computation and insert results."""
    prepared = _compute_ss_prepare(*args)
    if prepared is None:
        return None
    if prepared.get("done") is not None:
        return prepared["done"]

    apply_result = server.workspace_apply_edit(
        lsp.ApplyWorkspaceEditParams(
            edit=prepared["edit"],
            label="Insert computed steady state",
        )
    )
    try:
        if hasattr(apply_result, "result"):
            apply_result = apply_result.result(timeout=5)
    except Exception as exc:
        message = f"Failed to apply steady state edit: {exc}"
        server.window_show_message(
            lsp.ShowMessageParams(
                type=lsp.MessageType.Warning,
                message=message,
            )
        )
        return {"success": False, "message": message}
    return _compute_ss_finish(prepared, apply_result)


@server.command("dynare/computeSteadyState")
def _compute_ss_command_protocol(*args):
    """Protocol-safe steady-state command handler."""
    prepared = _compute_ss_prepare(*args)
    if prepared is None:
        return None
    if prepared.get("done") is not None:
        return prepared["done"]

    apply_result = server.workspace_apply_edit(
        lsp.ApplyWorkspaceEditParams(
            edit=prepared["edit"],
            label="Insert computed steady state",
        )
    )
    try:
        if hasattr(apply_result, "result"):
            apply_result = yield _wait_for_apply_edit_result, (apply_result,), {}
    except Exception as exc:
        message = f"Failed to apply steady state edit: {exc}"
        server.window_show_message(
            lsp.ShowMessageParams(
                type=lsp.MessageType.Warning,
                message=message,
            )
        )
        return {"success": False, "message": message}
    return _compute_ss_finish(prepared, apply_result)


# ---------------------------------------------------------------------------
# Editor-integrated execution — run the preprocessor or MATLAB+Dynare and
# surface the results as diagnostics (the "Runnables" pattern).
# ---------------------------------------------------------------------------


def _uri_from_command_args(args) -> Optional[str]:
    """Extract a URI from a command's args (bare string or ``{"uri": ...}``)."""
    if len(args) == 1 and isinstance(args[0], list):
        args = tuple(args[0])
    if not args:
        return None
    raw = args[0]
    uri = raw.get("uri") if isinstance(raw, dict) else raw
    return uri if isinstance(uri, str) else None


def _command_document_text(uri: str) -> Optional[str]:
    """Best-effort current text of *uri* (live document, else cached model)."""
    try:
        return server.workspace.get_text_document(uri).source
    except Exception:
        with _state_lock:
            model = _document_models.get(uri)
        if model is not None:
            return getattr(model, "original_text", "") or model.text
        return None


def _source_dir_for_uri(uri: str) -> Optional[str]:
    try:
        if not uri.startswith("file://"):
            return None
        from .include_resolver import _uri_to_path

        parent = _uri_to_path(uri).parent
        return str(parent) if parent.is_dir() else None
    except Exception:
        return None


def _preprocessor_scope_for_uri(
    uri: str,
    text: str,
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """Return the model text/path context the preprocessor should check.

    Open include fragments are diagnosed in their parent block context.  Dynare's
    preprocessor needs the same entry file; running it on the fragment alone can
    reject valid equation snippets as syntax errors.
    """
    try:
        with _state_lock:
            model = _document_models.get(uri)
        model = model or parse(text)
        if not any(
            (
                model.model_block_range,
                model.steady_state_block_range,
                model.initval_block_range,
                model.endval_block_range,
                model.shocks_block_range,
            )
        ):
            parent_contexts = _parent_include_contexts(uri, _workspace_index)
            parent_context, _ambiguous_parents = _select_parent_include_context(
                uri,
                _workspace_index,
                parent_contexts,
            )
            if parent_context is not None:
                parent_uri, _parent_model, _included_model = parent_context
                parent_text = _command_document_text(parent_uri)
                if parent_text is not None:
                    return (
                        parent_uri,
                        parent_text,
                        _source_dir_for_uri(parent_uri),
                        uri,
                    )
    except Exception:
        logger.exception("Preprocessor parent scope lookup failed for %s", uri)
    return uri, text, _source_dir_for_uri(uri), None


def _scope_preprocessor_result_to_include(preproc_result, active_include_uri):
    if active_include_uri is None or preproc_result is None:
        return preproc_result
    if preproc_result.success:
        return preproc_result
    try:
        from .include_resolver import _uri_to_path

        active_file = str(_uri_to_path(active_include_uri))
    except Exception:
        active_file = active_include_uri
    from .preprocessor import diagnostic_message_matches_file

    diagnostics = [
        diag
        for diag in preproc_result.diagnostics
        if diagnostic_message_matches_file(diag.message, active_file)
    ]
    return replace(preproc_result, diagnostics=diagnostics)


def _execution_file_key(uri_or_path: str) -> str:
    try:
        if uri_or_path.startswith("file://"):
            from .include_resolver import _uri_to_path

            return str(_uri_to_path(uri_or_path))
    except Exception:
        pass
    return _strip_include_instance_suffix(uri_or_path)


def _workspace_files_for_execution(entry_uri: str, entry_text: str) -> Dict[str, str]:
    """Return current entry + included file contents for private temp execution."""
    files = {_execution_file_key(entry_uri): entry_text}
    try:
        included = _workspace_index.resolve_all_includes(entry_uri)
    except Exception:
        logger.exception("Execution include lookup failed for %s", entry_uri)
        return files

    for include_key, include_model in included.items():
        real_key = _strip_include_instance_suffix(include_key)
        include_uri = _path_to_uri(real_key)
        content = _command_document_text(include_uri)
        if content is None:
            content = getattr(include_model, "original_text", "") or include_model.text
        files[_execution_file_key(real_key)] = content
    return files


def _execution_file_current_text(path_key: str) -> Tuple[bool, Optional[str]]:
    """Return current text for an execution snapshot file key."""
    real_key = _strip_include_instance_suffix(path_key)
    uri = _path_to_uri(real_key)
    text = _command_document_text(uri)
    if text is not None:
        return True, text
    try:
        path = Path(real_key)
        if path.is_absolute() and not path.exists():
            return True, None
        if path.exists() and path.is_file():
            return True, path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        logger.exception("Execution file freshness check failed for %s", path_key)
    return False, None


def _execution_snapshot_is_current(files: Dict[str, str]) -> bool:
    """Return false when any file used for execution has changed."""
    for path_key, expected_text in files.items():
        available, current_text = _execution_file_current_text(path_key)
        if available and current_text != expected_text:
            return False
    return True


def _materialize_execution_workspace(
    entry_file: str,
    files: Dict[str, str],
    tmp_root: Path,
) -> Tuple[Path, Dict[str, str]]:
    """Mirror execution snapshot files under *tmp_root*.

    Returns ``(entry_dir, rewritten_files)`` where *rewritten_files* has
    absolute ``@#include`` directives pointing at SUPPLIED paths redirected
    to their mirror copies (otherwise the preprocessor would read the real
    on-disk file and ignore the supplied content).
    """
    from .preprocessor import rewrite_supplied_absolute_includes

    normalized = {fname: Path(_strip_include_instance_suffix(fname)) for fname in files}
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

    entry_parent = tmp_root / _relative_path(entry_file).parent
    entry_parent.mkdir(parents=True, exist_ok=True)
    planned: List[Tuple[str, Path]] = []
    target_by_norm: Dict[str, str] = {}
    for fname in files:
        target = tmp_root / _relative_path(fname)
        planned.append((fname, target))
        try:
            norm_key = os.path.normcase(os.path.abspath(str(normalized[fname])))
        except (OSError, ValueError):
            continue
        target_by_norm[norm_key] = str(target)
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

    for content, target in materialized:
        if basename_counts.get(target.name) != 1:
            continue
        alias = entry_parent / target.name
        if alias.exists():
            continue
        alias.write_text(content, encoding="utf-8")
    return entry_parent, rewritten


def _run_preprocessor_with_snapshot(
    mod_text: str,
    preprocessor_path: str,
    entry_file: str,
    files: Dict[str, str],
):
    from .preprocessor import (
        rewrite_supplied_absolute_includes,
        run_preprocessor,
    )

    tmp_root = Path(tempfile.mkdtemp(prefix="dynare_lsp_lsp_"))
    try:
        entry_parent, rewritten = _materialize_execution_workspace(
            entry_file,
            files,
            tmp_root,
        )
        entry_text = rewritten.get(entry_file)
        if entry_text is None:
            # Entry text supplied separately from the snapshot map — apply
            # the same absolute-include redirection to it.
            target_by_norm = {}
            for fname in files:
                key = _strip_include_instance_suffix(fname)
                try:
                    norm_key = os.path.normcase(os.path.abspath(key))
                except (OSError, ValueError):
                    continue
                target_by_norm[norm_key] = str(tmp_root / Path(key).name)
            entry_text = rewrite_supplied_absolute_includes(mod_text, target_by_norm)
        return run_preprocessor(
            entry_text,
            preprocessor_path,
            source_dir=str(entry_parent),
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _preprocessor_command_payload(result) -> dict:
    return {
        "success": result.success,
        "exit_code": getattr(result, "exit_code", None),
        "diagnostics": [
            {
                "line": d.range.start.line + 1,
                "column": d.range.start.character + 1,
                "severity": d.severity.name,
                "message": d.message,
                "code": d.code,
            }
            for d in result.diagnostics
        ],
        "raw_stdout": getattr(result, "stdout", ""),
        "raw_stderr": getattr(result, "stderr", ""),
    }


def _run_anchor_range(model) -> DRange:
    """A short range at the ``model;`` opener to anchor whole-model run errors."""
    if model is not None and model.model_block_range is not None:
        try:
            mapped = _map_range_to_original_source(model, model.model_block_range)
            start = mapped.start
            return DRange(start, DPos(start.line, start.character + 5))
        except Exception:
            pass
    return DRange(DPos(0, 0), DPos(0, 1))


def _set_run_diagnostics(uri: str, diags: List[DDiag]) -> None:
    with _state_lock:
        if diags:
            _document_run_diagnostics[uri] = diags
        else:
            _document_run_diagnostics.pop(uri, None)
    _publish_all_diagnostics(uri)


def _show_message(message_type: "lsp.MessageType", message: str) -> None:
    """Send a window message, serialized with other client sends.

    The run commands execute on a worker thread (``@server.thread()``); the
    send lock keeps their frames from interleaving with the background
    solver/preprocessor publishes.
    """
    with _client_send_lock:
        server.window_show_message(
            lsp.ShowMessageParams(type=message_type, message=message)
        )


@server.command("dynare/runPreprocessor")
@server.thread()
def run_preprocessor_command(*args):
    """Run the bundled Dynare preprocessor on demand and report the result.

    Structural diagnostics already surface on save via reconciliation; this
    command gives an explicit "validate now" with a summary and the full
    structured result (the same payload the ``dynare_run_preprocessor`` MCP
    tool returns).
    """
    uri = _uri_from_command_args(args)
    if uri is None:
        return {"success": False, "message": "Missing or invalid URI argument"}
    text = _command_document_text(uri)
    if text is None:
        return {"success": False, "message": "Document not available"}
    (
        entry_uri,
        preprocessor_source,
        _source_dir,
        active_include_uri,
    ) = _preprocessor_scope_for_uri(uri, text)
    preprocessor_files = _workspace_files_for_execution(entry_uri, preprocessor_source)
    preprocessor_active_file = _execution_file_key(entry_uri)

    from .preprocessor import find_preprocessor

    pp_path = _preprocessor_path or find_preprocessor()
    if not pp_path:
        _show_message(
            lsp.MessageType.Warning,
            "Dynare preprocessor binary not found (bundled under dynare_lsp/bin/).",
        )
        return {
            "success": False,
            "message": "preprocessor not found",
            "exit_code": None,
            "diagnostics": [],
            "raw_stdout": "",
            "raw_stderr": "",
        }

    result = _run_preprocessor_with_snapshot(
        preprocessor_source,
        pp_path,
        preprocessor_active_file,
        preprocessor_files,
    )
    result = _scope_preprocessor_result_to_include(
        result,
        active_include_uri,
    )
    # Route through the same cache the on-save path uses and re-publish, so the
    # findings actually appear in the Problems panel (via reconciliation) rather
    # than being computed and discarded.  Guard on the document still being open
    # and unchanged so a mid-run edit cannot leave a stale result cached.
    # (Read the current text outside the lock: _command_document_text may take
    # _state_lock itself.)
    snapshot_current = _execution_snapshot_is_current(preprocessor_files)
    with _state_lock:
        fresh = _document_models.get(uri) is not None and snapshot_current
        if fresh:
            _document_preprocessor_results[uri] = result
    if fresh:
        _publish_all_diagnostics(uri)
    else:
        _show_message(
            lsp.MessageType.Info,
            "Preprocessor run discarded: the document changed while it was running.",
        )
        return _preprocessor_command_payload(result)

    n = len(result.diagnostics)
    if result.success:
        message, kind = (
            "Preprocessor: model is structurally valid.",
            lsp.MessageType.Info,
        )
    elif n:
        message, kind = (
            f"Preprocessor found {n} issue(s); see the Problems panel.",
            lsp.MessageType.Warning,
        )
    else:
        message, kind = (
            "Preprocessor rejected the model (see the output log).",
            lsp.MessageType.Warning,
        )
    _show_message(kind, message)
    return _preprocessor_command_payload(result)


@server.command("dynare/runDynare")
@server.thread()
def run_dynare_command(*args):
    """Run the model end-to-end with MATLAB + Dynare and surface the result.

    Errors and a failed Blanchard-Kahn condition become diagnostics anchored
    at the model block; a successful run reports the computed steady state.
    Returns the full structured verdict (mirrors ``dynare_run_dynare``).
    """
    uri = _uri_from_command_args(args)
    if uri is None:
        return {"success": False, "message": "Missing or invalid URI argument"}
    text = _command_document_text(uri)
    if text is None:
        return {"success": False, "message": "Document not available"}
    (
        entry_uri,
        run_source,
        _source_dir,
        _active_include_uri,
    ) = _preprocessor_scope_for_uri(uri, text)
    run_files = _workspace_files_for_execution(entry_uri, run_source)
    run_active_file = _execution_file_key(entry_uri)

    from .matlab_runner import run_dynare_matlab

    result = run_dynare_matlab(
        run_source,
        active_file=run_active_file,
        files=run_files,
    )

    # The MATLAB run can take seconds; if the document changed meanwhile, the
    # result describes now-stale text -- discard it rather than publish stale
    # diagnostics on the new model.
    if not _execution_snapshot_is_current(run_files):
        _show_message(
            lsp.MessageType.Info,
            "Dynare run discarded: the document changed while it was running.",
        )
        return result

    with _state_lock:
        model = _document_models.get(uri)

    # An unavailable toolchain is an environment issue, not a model error:
    # report it but do not leave diagnostics on the file.
    if not result.get("matlab_available") or not result.get("dynare_available"):
        _set_run_diagnostics(uri, [])
        _show_message(
            lsp.MessageType.Info,
            f"Dynare run skipped: {result.get('message', 'toolchain unavailable')}",
        )
        return result

    anchor = _run_anchor_range(model)
    diags: List[DDiag] = []
    for err in result.get("errors") or []:
        diags.append(
            DDiag(
                range=anchor,
                severity=Severity.ERROR,
                message=f"Dynare: {err}",
                source="dynare-run",
                code="DYNR",
            )
        )
    bk = result.get("blanchard_kahn") or {}
    if bk.get("satisfied") is False:
        diags.append(
            DDiag(
                range=anchor,
                severity=Severity.WARNING,
                message=f"Blanchard-Kahn: {bk.get('message', 'conditions not satisfied')}",
                source="dynare-run",
                code="DYNR",
            )
        )
    _set_run_diagnostics(uri, diags)

    if result.get("success") and not diags:
        steady = result.get("steady_state") or {}
        preview = ", ".join(
            f"{name} = {value:.4g}" for name, value in list(steady.items())[:6]
        )
        more = "" if len(steady) <= 6 else f" (+{len(steady) - 6} more)"
        _show_message(
            lsp.MessageType.Info,
            "Dynare run succeeded."
            + (f" Steady state: {preview}{more}" if preview else ""),
        )
    else:
        _show_message(
            lsp.MessageType.Warning,
            f"Dynare run: {result.get('status', 'failed')} — "
            f"{len(diags)} issue(s); see the Problems panel.",
        )
    return result


# ---------------------------------------------------------------------------
# Inlay hints — show SS values and parameter values inline
# ---------------------------------------------------------------------------


@server.feature(lsp.TEXT_DOCUMENT_INLAY_HINT)
def inlay_hint(params: lsp.InlayHintParams) -> Optional[List[lsp.InlayHint]]:
    """Show parameter values and steady state values as inline hints.

    LLM-friendly: makes computed values visible without requiring hover,
    so LLMs processing the file can see all values at a glance.
    """
    uri = params.text_document.uri
    with _state_lock:
        model = _document_models.get(uri)
        report = _document_ss_reports.get(uri)
    if model is None:
        return None

    hints: List[lsp.InlayHint] = []
    start_line = params.range.start.line
    end_line = params.range.end.line

    def _hint_end(r: DRange) -> lsp.Position:
        return _to_lsp_range_for_model(model, r).end

    def _hint_statement_end(r: DRange) -> lsp.Position:
        lines = model.text.split("\n")
        end = r.end
        if 0 <= end.line < len(lines):
            line = lines[end.line]
            char = max(0, min(end.character, len(line)))
            while char < len(line) and line[char].isspace():
                char += 1
            if char < len(line) and line[char] == ";":
                char += 1
            end = DPos(end.line, char)
        return _to_lsp_range_for_model(model, DRange(r.start, end)).end

    # --- Parameter assignment value hints ---
    visible_values = _assigned_values(
        model,
        list(_workspace_index.resolve_all_includes(uri).values()),
    )
    for a in model.param_assignments:
        line = a.range.start.line
        value = a.value if a.value is not None else visible_values.get(a.name)
        if start_line <= line <= end_line and value is not None:
            a = replace(a, value=value)
            hints.append(
                lsp.InlayHint(
                    position=_hint_end(a.range),
                    label=f" → {a.value:.6g}",
                    kind=lsp.InlayHintKind.Type,
                    padding_left=True,
                )
            )

    # --- Steady state value hints in steady_state_model block ---
    if report:
        active_equation_ranges = {
            (
                eq.range.start.line,
                eq.range.start.character,
                eq.range.end.line,
                eq.range.end.character,
                eq.text,
            )
            for eq in model.model_equations
        }
        for eq in model.steady_state_equations:
            line = eq.range.start.line
            if start_line <= line <= end_line:
                text = eq.text.strip()
                if "=" in text and not text.startswith("#"):
                    name = text.split("=", 1)[0].strip()
                    if name in report.values:
                        val = report.values[name]
                        hints.append(
                            lsp.InlayHint(
                                position=_hint_statement_end(eq.range),
                                label=f" = {val:.6g}",
                                kind=lsp.InlayHintKind.Type,
                                padding_left=True,
                            )
                        )

        # --- Model equation residual hints ---
        for result in report.results:
            equation_key = (
                result.equation.range.start.line,
                result.equation.range.start.character,
                result.equation.range.end.line,
                result.equation.range.end.character,
                result.equation.text,
            )
            if equation_key not in active_equation_ranges:
                continue
            line = result.equation.range.start.line
            if start_line <= line <= end_line and not result.is_local_var:
                if result.is_satisfied:
                    hints.append(
                        lsp.InlayHint(
                            position=_hint_statement_end(result.equation.range),
                            label=" ✓ SS OK",
                            kind=lsp.InlayHintKind.Parameter,
                            padding_left=True,
                        )
                    )
                elif result.residual is not None:
                    hints.append(
                        lsp.InlayHint(
                            position=_hint_statement_end(result.equation.range),
                            label=f" ✗ residual={result.residual:.2e}",
                            kind=lsp.InlayHintKind.Parameter,
                            padding_left=True,
                        )
                    )
    # --- BK condition hint on model block ---
    with _state_lock:
        bk_result = _document_bk_results.get(uri)
    if bk_result is not None and model.model_block_range:
        model_line = model.model_block_range.start.line
        if start_line <= model_line <= end_line:
            if not bk_result.satisfied and bk_result.message.lower().startswith(
                "blanchard-kahn check skipped",
            ):
                bk_label = " BK skipped"
            elif bk_result.satisfied:
                bk_label = (
                    f" BK OK ({bk_result.n_unstable} unstable, "
                    f"{bk_result.n_forward} forward)"
                )
            else:
                if bk_result.n_unstable < bk_result.n_forward:
                    bk_label = (
                        f" BK FAIL: {bk_result.n_unstable} unstable, "
                        f"{bk_result.n_forward} forward (indeterminacy)"
                    )
                else:
                    bk_label = (
                        f" BK FAIL: {bk_result.n_unstable} unstable, "
                        f"{bk_result.n_forward} forward (explosive)"
                    )
            bk_pos = _to_lsp_range_for_model(
                model,
                DRange(
                    start=model.model_block_range.start,
                    end=DPos(
                        model_line,
                        model.model_block_range.start.character + 5,
                    ),
                ),
            ).end
            hints.append(
                lsp.InlayHint(
                    position=bk_pos,
                    label=bk_label,
                    kind=lsp.InlayHintKind.Parameter,
                    padding_left=True,
                )
            )

    # --- Solver hints (always shown when solver results exist) ---
    with _state_lock:
        solver_result = _document_solver_results.get(uri)
    if solver_result and solver_result.success:
        # Show solved value next to each endogenous var declaration
        for v in model.endogenous:
            line = v.range.start.line
            if start_line <= line <= end_line and v.name in solver_result.values:
                val = solver_result.values[v.name]
                hints.append(
                    lsp.InlayHint(
                        position=_hint_end(v.range),
                        label=f" SS={val:.6g}",
                        kind=lsp.InlayHintKind.Type,
                        padding_left=True,
                    )
                )

        # Show solved value next to initval entries (compare with current)
        for entry in model.initval_entries:
            line = entry.range.start.line
            if start_line <= line <= end_line and entry.name in solver_result.values:
                solved = solver_result.values[entry.name]
                if entry.value is not None and abs(entry.value - solved) < 1e-8:
                    label = " [OK]"
                else:
                    label = f" should be {solved:.6g}"
                hints.append(
                    lsp.InlayHint(
                        position=_hint_end(entry.range),
                        label=label,
                        kind=lsp.InlayHintKind.Type,
                        padding_left=True,
                    )
                )

        # Show residuals on model equations (only if no SS block already showing them)
        if not report:
            from .steady_state import validate_computed_steady_state

            computed_report = validate_computed_steady_state(
                model, solver_result.values
            )
            for eq_result in computed_report.results:
                line = eq_result.equation.range.start.line
                if start_line <= line <= end_line and not eq_result.is_local_var:
                    if eq_result.is_satisfied:
                        hints.append(
                            lsp.InlayHint(
                                position=_hint_statement_end(eq_result.equation.range),
                                label=" [solved] SS OK",
                                kind=lsp.InlayHintKind.Parameter,
                                padding_left=True,
                            )
                        )
                    elif eq_result.residual is not None:
                        hints.append(
                            lsp.InlayHint(
                                position=_hint_statement_end(eq_result.equation.range),
                                label=f" residual={eq_result.residual:.2e}",
                                kind=lsp.InlayHintKind.Parameter,
                                padding_left=True,
                            )
                        )

    return hints if hints else None


# ---------------------------------------------------------------------------
# Folding ranges — collapse blocks
# ---------------------------------------------------------------------------


@server.feature(lsp.TEXT_DOCUMENT_FOLDING_RANGE)
def folding_range(params: lsp.FoldingRangeParams) -> Optional[List[lsp.FoldingRange]]:
    """Provide folding ranges for Dynare blocks and macro control flow."""
    uri = params.text_document.uri
    with _state_lock:
        model = _document_models.get(uri)
    if model is None:
        return None

    ranges: List[lsp.FoldingRange] = []

    for block in model.blocks:
        r = block.range
        if r.start.line < r.end.line:
            ranges.append(
                lsp.FoldingRange(
                    start_line=r.start.line,
                    end_line=r.end.line,
                    kind=lsp.FoldingRangeKind.Region,
                    collapsed_text=f"{block.keyword} ... end;",
                )
            )

    # Also fold macro blocks.  Use one stack so crossed @#endif/@#endfor
    # pairs do not create ranges for mismatched structures.
    macro_stack: List = []
    conditional_openers = {"if", "ifdef", "ifndef"}
    for directive in model.macro_directives:
        if directive.kind in conditional_openers or directive.kind == "for":
            macro_stack.append(directive)
        elif (
            directive.kind == "endif"
            and macro_stack
            and macro_stack[-1].kind in conditional_openers
        ):
            opener = macro_stack.pop()
            if opener.range.start.line < directive.range.end.line:
                ranges.append(
                    lsp.FoldingRange(
                        start_line=opener.range.start.line,
                        end_line=directive.range.end.line,
                        kind=lsp.FoldingRangeKind.Region,
                        collapsed_text=f"@#{opener.kind} ... @#endif",
                    )
                )
        elif (
            directive.kind == "endfor" and macro_stack and macro_stack[-1].kind == "for"
        ):
            opener = macro_stack.pop()
            if opener.range.start.line < directive.range.end.line:
                ranges.append(
                    lsp.FoldingRange(
                        start_line=opener.range.start.line,
                        end_line=directive.range.end.line,
                        kind=lsp.FoldingRangeKind.Region,
                        collapsed_text="@#for ... @#endfor",
                    )
                )

    # Also fold block comments
    doc = server.workspace.get_text_document(uri)
    text = doc.source
    for m in re.finditer(r"/\*.*?\*/", text, re.DOTALL):
        start_pos = text.count("\n", 0, m.start())
        end_pos = text.count("\n", 0, m.end())
        if start_pos < end_pos:
            ranges.append(
                lsp.FoldingRange(
                    start_line=start_pos,
                    end_line=end_pos,
                    kind=lsp.FoldingRangeKind.Comment,
                )
            )

    return ranges if ranges else None


# ---------------------------------------------------------------------------
# Document Links — clickable @#include targets
# ---------------------------------------------------------------------------


def _gather_all_diagnostics(uri: str) -> List[lsp.Diagnostic]:
    """Collect base + BK + preprocessor diagnostics for *uri* (lock-safe)."""
    with _state_lock:
        base_diags = _document_diagnostics.get(uri, [])
        bk_result = _document_bk_results.get(uri)
        model_diagnostics = _document_model_diagnostics.get(uri, [])
        identification_diagnostics = _document_identification.get(uri, [])
        model = _document_models.get(uri)
        preproc_result = _document_preprocessor_results.get(uri)

    from .preprocessor import reconcile_diagnostics

    all_diags = list(reconcile_diagnostics(base_diags, preproc_result))
    if bk_result is not None:
        try:
            from .bk_check import bk_to_diagnostics

            if model is not None:
                for diag in bk_to_diagnostics(bk_result, model):
                    if getattr(model, "source_map", None):
                        diag = replace(
                            diag,
                            range=_map_range_to_original_source(model, diag.range),
                        )
                    all_diags.append(diag)
        except Exception:
            pass
    if model_diagnostics:
        try:
            if model is not None:
                for diag in model_diagnostics:
                    if getattr(model, "source_map", None):
                        diag = replace(
                            diag,
                            range=_map_range_to_original_source(model, diag.range),
                        )
                    all_diags.append(diag)
        except Exception:
            pass
    if identification_diagnostics:
        try:
            if model is not None:
                for diag in identification_diagnostics:
                    if getattr(model, "source_map", None):
                        diag = replace(
                            diag,
                            range=_map_range_to_original_source(model, diag.range),
                        )
                    all_diags.append(diag)
        except Exception:
            pass
    source_text = (
        (getattr(model, "original_text", "") or model.text)
        if model is not None
        else None
    )
    with _state_lock:
        all_diags.extend(_document_run_diagnostics.get(uri, []))
    all_diags = _dedupe_diagnostics(all_diags)
    return [_to_lsp_diagnostic(d, source_text) for d in all_diags]


@server.feature(
    lsp.TEXT_DOCUMENT_DIAGNOSTIC,
    lsp.DiagnosticOptions(
        inter_file_dependencies=True,
        workspace_diagnostics=True,
    ),
)
def diagnostic_pull(
    params: lsp.DocumentDiagnosticParams,
) -> lsp.RelatedFullDocumentDiagnosticReport:
    """LSP 3.17 pull-mode diagnostic provider.

    Returns the same diagnostics that the push-mode handler emits via
    ``textDocument/publishDiagnostics`` on edit.  Clients that prefer
    pull-mode (and CI / batch harnesses that want deterministic
    on-demand evaluation) call this; the push handler keeps firing for
    editors that haven't migrated.  Both paths read from the same cache,
    so they cannot disagree.
    """
    uri = params.text_document.uri
    return lsp.RelatedFullDocumentDiagnosticReport(
        items=_gather_all_diagnostics(uri),
    )


@server.feature(lsp.TEXT_DOCUMENT_DOCUMENT_LINK)
def document_link(
    params: lsp.DocumentLinkParams,
) -> Optional[List[lsp.DocumentLink]]:
    """Return clickable links for ``@#include`` directives.

    Each directive whose target resolves through the workspace's search
    paths becomes a ``DocumentLink`` whose target is the included file's
    ``file://`` URI.  Editors render these as ctrl-click links over the
    directive text.  Unresolved directives are still reported as E061
    diagnostics elsewhere; we just don't generate a link for them.
    """
    uri = params.text_document.uri
    with _state_lock:
        model = _document_models.get(uri)
    if model is None or not model.includes:
        return None
    source_text = getattr(model, "original_text", "") or model.text
    try:
        source_text = server.workspace.get_text_document(uri).source
    except Exception:
        pass

    links: List[lsp.DocumentLink] = []
    try:
        resolved_includes = _workspace_index.resolve_direct_includes(uri)
    except Exception:
        logger.exception("document links: include resolution failed for %s", uri)
        resolved_includes = []
    for directive, include_key, _context in resolved_includes:
        links.append(
            lsp.DocumentLink(
                range=_to_lsp_range_in_text(source_text, directive.range),
                target=_path_to_uri(str(_strip_include_instance_suffix(include_key))),
                tooltip=f"Open {directive.filename}",
            )
        )
    return links if links else None


# ---------------------------------------------------------------------------
# Find References / Document Highlight / Rename
# ---------------------------------------------------------------------------
#
# All three answer the same underlying question — "where does this symbol
# appear in the file?" — so they share one implementation. References return
# Locations; DocumentHighlight returns the same ranges with Read kind;
# Rename emits a WorkspaceEdit replacing each occurrence with the new name.

# Comment-stripping pattern shared across symbol-reference scans. We strip
# // line comments, % line comments (Matlab-style, which Dynare also
# accepts), and /* */ block comments before searching so an occurrence
# inside a comment is not treated as a real reference or renamed.
_LINE_COMMENT_RE = re.compile(r"(?://|%).*?$", re.MULTILINE)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

# Additional non-identifier zones that must be masked before reference /
# rename scans: ``"..."`` and ``'...'`` string literals (used by
# ``@#include "y.mod"`` and ``[name='y']`` equation tags).  Without
# this, ``find-references`` reports the string contents and ``rename``
# would rewrite filenames inside include directives or equation labels.
_STRING_LITERAL_RE = re.compile(r"\"[^\"\n]*\"|'[^'\n]*'")
_MCP_TAG_VALUE_RE = re.compile(r"\bmcp\s*=\s*(['\"])(.*?)\1", re.IGNORECASE)
_TAG_ATTRIBUTE_KEY_RE = re.compile(r"\b(?:name|mcp)\b(?=\s*=)", re.IGNORECASE)

# Dynare macro directives are line-oriented preprocessor statements, not
# model code.  Mask the whole line so bare include paths such as
# ``@#include helper.mod`` are not treated as symbol references.
_MACRO_DIRECTIVE_LINE_RE = re.compile(r"^[ \t]*@\#[^\n]*(?:\n|$)", re.MULTILINE)
_MACRO_INTERPOLATION_RE = re.compile(r"@\{[A-Za-z_][A-Za-z0-9_]*\}")
_ON_THE_FLY_KIND_MARKERS = frozenset({"e", "x", "p"})


def _is_on_the_fly_kind_marker(source_line: str, start: int, end: int) -> bool:
    return (
        start > 0
        and source_line[start - 1] == "|"
        and source_line[start:end].lower() in _ON_THE_FLY_KIND_MARKERS
    )


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


def _mask_strings_preserving_mcp_values(source: str) -> str:
    """Mask strings except the expression part of equation ``mcp=`` tags."""
    keep = [False] * len(source)
    for match in _MCP_TAG_VALUE_RE.finditer(source):
        if _inside_quoted_string(source, match.start()):
            continue
        for idx in range(match.start(2), match.end(2)):
            keep[idx] = True
    chars = list(source)
    for match in _STRING_LITERAL_RE.finditer(source):
        for idx in range(match.start(), match.end()):
            if not keep[idx] and chars[idx] != "\n":
                chars[idx] = " "
    for tag_start, tag_end in _iter_equation_tag_spans(source):
        tag_text = source[tag_start:tag_end]
        for key in _TAG_ATTRIBUTE_KEY_RE.finditer(tag_text):
            start = tag_start + key.start()
            end = tag_start + key.end()
            for idx in range(start, end):
                chars[idx] = " "
    return "".join(chars)


def _position_inside_mcp_value(source: str, pos: lsp.Position) -> bool:
    """Return True when an LSP position is inside an equation MCP expression."""
    lines = source.split("\n")
    if pos.line < 0 or pos.line >= len(lines):
        return False
    if _cursor_inside_comment_only(lines, pos):
        return False
    source_pos = _lsp_position_to_source_position(lines, pos)
    offset = _position_to_offset_local(source, source_pos)
    return any(
        match.start(2) <= offset < match.end(2)
        for match in _MCP_TAG_VALUE_RE.finditer(source)
        if not _inside_quoted_string(source, match.start())
    )


def _cursor_inside_comment_only(lines: List[str], pos: lsp.Position) -> bool:
    """Return True only for comments or macro directive/interpolation text."""
    if pos.line >= len(lines):
        return False
    pos = _lsp_position_to_source_position(lines, pos)
    line = lines[pos.line]
    macro_match = re.match(r"^[ \t]*@\#", line)
    if macro_match and pos.character >= macro_match.start():
        return True
    for macro_interp in _MACRO_INTERPOLATION_RE.finditer(line):
        if macro_interp.start() <= pos.character < macro_interp.end():
            return True

    in_block = False
    for prev_line in lines[: pos.line]:
        k = 0
        in_string_pq: Optional[str] = None
        while k < len(prev_line):
            if in_block:
                idx = prev_line.find("*/", k)
                if idx == -1:
                    break
                in_block = False
                k = idx + 2
                continue
            ch = prev_line[k]
            nxt = prev_line[k + 1] if k + 1 < len(prev_line) else ""
            if in_string_pq is not None:
                if ch == in_string_pq:
                    in_string_pq = None
                k += 1
                continue
            if ch in ('"', "'"):
                in_string_pq = ch
                k += 1
                continue
            if ch == "/" and nxt == "/":
                break
            if ch == "%":
                break
            if ch == "/" and nxt == "*":
                in_block = True
                k += 2
                continue
            k += 1

    in_string_q: Optional[str] = None
    i = 0
    limit = min(pos.character + 1, len(line))
    while i < limit:
        ch = line[i]
        nxt = line[i + 1] if i + 1 < len(line) else ""
        if in_block:
            close_idx = line.find("*/", i)
            if close_idx == -1 or pos.character < close_idx + 2:
                return True
            in_block = False
            i = close_idx + 2
            continue
        if in_string_q is not None:
            if ch == in_string_q:
                in_string_q = None
            i += 1
            continue
        if ch == "/" and nxt == "*":
            in_block = True
            i += 2
            if pos.character < i:
                return True
            continue
        if ch == "/" and nxt == "/":
            return i <= pos.character
        if ch == "%":
            return i <= pos.character
        if ch in ('"', "'"):
            in_string_q = ch
            i += 1
            continue
        i += 1
    return in_block


def _cursor_inside_comment_or_non_mcp_string(
    source: str,
    lines: List[str],
    pos: lsp.Position,
) -> bool:
    """Treat MCP tag expressions as model code, unlike ordinary strings."""
    return _cursor_inside_comment_or_string(
        lines, pos
    ) and not _position_inside_mcp_value(source, pos)


def _mask_inactive_macro_branches(source: str) -> str:
    """Blank known-inactive macro branch bodies while preserving positions."""
    try:
        _defines, active_lines, _line_defines = _macro_branch_state(source)
        return _mask_inactive_macro_lines(source, active_lines)
    except Exception:
        logger.exception("Could not mask inactive macro branches")
        return source


def _find_symbol_occurrences(
    source: str,
    name: str,
) -> List[DRange]:
    """Find every (line, col) range where ``name`` appears as a whole word.

    Comments are masked out (replaced with spaces of equal length to preserve
    column offsets) before the scan.
    """
    if not name:
        return []

    # Mask comments + string literals without disturbing line/column positions.
    # IMPORTANT: mask STRINGS FIRST.  Otherwise a ``//`` or ``%`` inside
    # a quoted equation tag (e.g. ``[name='a // b']``) gets caught by
    # the comment mask before the string mask can protect it, and the
    # rest of the line is blanked away.
    def _blank(match: re.Match) -> str:
        return re.sub(r"\S", " ", match.group(0))

    masked = _mask_inactive_macro_branches(source)
    masked = _mask_strings_preserving_mcp_values(masked)
    masked = _BLOCK_COMMENT_RE.sub(_blank, masked)
    masked = _MACRO_DIRECTIVE_LINE_RE.sub(_blank, masked)
    masked = _MACRO_INTERPOLATION_RE.sub(_blank, masked)
    masked = _LINE_COMMENT_RE.sub(_blank, masked)

    pattern = re.compile(rf"\b{re.escape(name)}\b")
    occurrences: List[DRange] = []

    # Build per-line start offsets: line_starts[k] is the absolute offset
    # of the first character on line k.  The previous implementation
    # accumulated character counts in a way that produced incorrect
    # offsets for line indices >= 2 (lengths past the second line
    # were derived from cumulative-running-total entries, not per-line
    # counts).  Bisect against this list to map a match offset to a
    # line index in O(log n).
    line_starts: List[int] = [0]
    for i, ch in enumerate(masked):
        if ch == "\n":
            line_starts.append(i + 1)

    import bisect as _bisect

    source_lines = source.split("\n")
    for m in pattern.finditer(masked):
        start = m.start()
        end = m.end()
        # bisect_right gives insertion point; subtract one to get the
        # line whose start is <= start.
        line_idx = _bisect.bisect_right(line_starts, start) - 1
        line_start = line_starts[line_idx]
        source_line = source_lines[line_idx] if line_idx < len(source_lines) else ""
        start_char = start - line_start
        end_char = end - line_start
        if _is_on_the_fly_kind_marker(source_line, start_char, end_char):
            continue
        occurrences.append(
            DRange(
                start=DPos(line=line_idx, character=start_char),
                end=DPos(line=line_idx, character=end_char),
            )
        )
    return occurrences


def _find_symbol_occurrences_in_model_source(
    source: str,
    name: str,
    model: Optional[ParsedModel],
) -> List[DRange]:
    if model is not None and getattr(model, "source_map", None):
        return [
            _map_range_to_original_source(model, rng)
            for rng in _find_symbol_occurrences(model.text, name)
        ]
    return _find_symbol_occurrences(source, name)


def _offset_equation_range(eq_range: DRange, local_range: DRange) -> DRange:
    def _add(pos: DPos) -> DPos:
        line = eq_range.start.line + pos.line
        character = (
            eq_range.start.character + pos.character if pos.line == 0 else pos.character
        )
        return DPos(line=line, character=character)

    return DRange(start=_add(local_range.start), end=_add(local_range.end))


def _range_contains_source_position(rng: DRange, pos: DPos) -> bool:
    start = (rng.start.line, rng.start.character)
    end = (rng.end.line, rng.end.character)
    here = (pos.line, pos.character)
    return start <= here < end


def _range_contains_range(outer: DRange, inner: DRange) -> bool:
    outer_start = (outer.start.line, outer.start.character)
    outer_end = (outer.end.line, outer.end.character)
    inner_start = (inner.start.line, inner.start.character)
    inner_end = (inner.end.line, inner.end.character)
    return outer_start <= inner_start and inner_end <= outer_end


def _find_model_local_occurrences(model: ParsedModel, name: str) -> List[DRange]:
    """Find occurrences of a model-local name within model equations only."""
    if not name or name not in _model_local_variables(model):
        return []
    occurrences: List[DRange] = []
    equation_ranges = [equation.range for equation in model.model_equations]
    for rng in _find_symbol_occurrences(model.text, name):
        if not any(
            _range_contains_range(eq_range, rng) for eq_range in equation_ranges
        ):
            continue
        if getattr(model, "source_map", None):
            rng = _map_range_to_original_source(model, rng)
        occurrences.append(rng)
    return occurrences


def _is_model_local_symbol_at_position(
    model: ParsedModel,
    source: str,
    pos: lsp.Position,
    name: str,
) -> bool:
    if name not in _model_local_variables(model):
        return False
    source_lines = source.split("\n")
    source_pos = _lsp_position_to_source_position(source_lines, pos)
    return any(
        _range_contains_source_position(
            rng, DPos(source_pos.line, source_pos.character)
        )
        for rng in _find_model_local_occurrences(model, name)
    )


def _position_is_in_model_equation(
    model: ParsedModel,
    source: str,
    pos: lsp.Position,
) -> bool:
    source_lines = source.split("\n")
    source_pos = _lsp_position_to_source_position(source_lines, pos)
    dpos = DPos(source_pos.line, source_pos.character)
    return any(
        _range_contains_source_position(equation.range, dpos)
        for equation in model.model_equations
    )


def _model_local_declaration_context(
    name: str,
    decl_hit: Optional[Tuple[str, VarDeclaration, Optional[ParsedModel]]],
    workspace_index: WorkspaceIndex,
    open_document_sources: dict,
) -> Optional[Tuple[str, str, ParsedModel]]:
    if decl_hit is None:
        return None
    decl_uri, _decl, decl_model = decl_hit
    if decl_model is None:
        decl_model = workspace_index.get_effective_model(decl_uri)
    if decl_model is None or name not in _model_local_variables(decl_model):
        return None
    source = (
        open_document_sources.get(decl_uri)
        or workspace_index.get_source(decl_uri)
        or getattr(decl_model, "original_text", "")
        or decl_model.text
    )
    return decl_uri, source, decl_model


def _build_model_local_rename_edit(
    uri: str,
    source: str,
    model: ParsedModel,
    old_name: str,
    new_name: str,
) -> Optional[lsp.WorkspaceEdit]:
    if not _VALID_IDENTIFIER_RE.match(new_name):
        return None
    if _reserved_identifier_reason(old_name) is not None:
        return None
    if _reserved_identifier_reason(new_name) is not None:
        return None
    edits = [
        lsp.TextEdit(
            range=_to_lsp_range_in_text(source, rng),
            new_text=new_name,
        )
        for rng in _find_model_local_occurrences(model, old_name)
    ]
    if not edits:
        return None
    return lsp.WorkspaceEdit(changes={uri: edits})


def _collect_model_local_references(
    active_uri: str,
    active_source: str,
    model: ParsedModel,
    name: str,
    workspace_index: WorkspaceIndex,
    open_document_sources: dict,
) -> List[Tuple[str, DRange]]:
    refs = [(active_uri, rng) for rng in _find_model_local_occurrences(model, name)]
    if (
        not refs
        and model.model_block_range is None
        and name in _model_local_variables(model)
    ):
        refs = [
            (active_uri, rng) for rng in _find_symbol_occurrences(active_source, name)
        ]
    active_key = _normalize_uri(active_uri)
    seen = {
        (
            _normalize_uri(uri),
            rng.start.line,
            rng.start.character,
            rng.end.line,
            rng.end.character,
        )
        for uri, rng in refs
    }
    for parent_uri, parent_model, _included_model in _parent_include_contexts(
        active_uri,
        workspace_index,
    ):
        if _normalize_uri(parent_uri) == active_key:
            continue
        parent_source = open_document_sources.get(parent_uri)
        if parent_source is None:
            parent_source = workspace_index.get_source(parent_uri)
        if not parent_source:
            continue
        parent_ranges = _find_model_local_occurrences(parent_model, name)
        if not parent_ranges:
            equation_ranges = [
                equation.range for equation in parent_model.model_equations
            ]
            parent_ranges = [
                rng
                for rng in _find_symbol_occurrences(parent_source, name)
                if any(
                    _range_contains_range(eq_range, rng) for eq_range in equation_ranges
                )
            ]
        for rng in parent_ranges:
            key = (
                _normalize_uri(parent_uri),
                rng.start.line,
                rng.start.character,
                rng.end.line,
                rng.end.character,
            )
            if key in seen:
                continue
            seen.add(key)
            refs.append((parent_uri, rng))
    return refs


def _build_model_local_context_rename_edit(
    uri: str,
    source: str,
    model: ParsedModel,
    old_name: str,
    new_name: str,
    workspace_index: WorkspaceIndex,
    open_document_sources: dict,
) -> Optional[lsp.WorkspaceEdit]:
    if not _VALID_IDENTIFIER_RE.match(new_name):
        return None
    if _reserved_identifier_reason(old_name) is not None:
        return None
    if _reserved_identifier_reason(new_name) is not None:
        return None
    refs = _collect_model_local_references(
        uri,
        source,
        model,
        old_name,
        workspace_index,
        open_document_sources,
    )
    if not refs:
        return None
    if not _reference_ranges_are_exact_source_slices(
        uri,
        source,
        old_name,
        refs,
        workspace_index,
        open_document_sources,
    ):
        return None
    grouped: dict = {}
    for target_uri, rng in refs:
        target_source = (
            source
            if target_uri == uri
            else open_document_sources.get(target_uri)
            or workspace_index.get_source(target_uri)
        )
        grouped.setdefault(target_uri, []).append(
            lsp.TextEdit(
                range=(
                    _to_lsp_range_in_text(target_source, rng)
                    if target_source is not None
                    else _to_lsp_range(rng)
                ),
                new_text=new_name,
            )
        )
    return lsp.WorkspaceEdit(changes=grouped)


def _collect_cross_file_references(
    active_uri: str,
    active_source: str,
    name: str,
    workspace_index: WorkspaceIndex,
    open_document_sources: dict,
) -> List[Tuple[str, DRange]]:
    """Find every reference to *name* across the active file and its workspace.

    Scope is the active file, files transitively included by it, and
    open parent documents whose own include graph reaches the active
    file.  Files are deduplicated on their normalised absolute path so
    a file that is both open in the editor and reached via an include
    doesn't get scanned twice.

    *open_document_sources* maps URI -> current source for every open
    document.  Sources for include-only files are pulled from the
    workspace index's cache.  Returns ``[(uri, range), ...]``.
    """
    if not name:
        return []

    results: List[Tuple[str, DRange]] = []
    seen_results: set = set()
    seen_paths: set = set()  # normalized absolute path of each scanned file
    active_key = _normalize_uri(active_uri)
    open_by_key = {
        _normalize_uri(uri): (uri, source)
        for uri, source in open_document_sources.items()
    }

    def _scan(
        uri: str,
        source: Optional[str],
        source_model: Optional[ParsedModel] = None,
        dedup_key: Optional[str] = None,
    ) -> None:
        if not source:
            return
        path_key = dedup_key or _normalize_uri(uri)
        if path_key in seen_paths:
            return
        seen_paths.add(path_key)
        if source_model is None:
            source_model = workspace_index.get_effective_model(uri)
        for r in _find_symbol_occurrences_in_model_source(
            source,
            name,
            source_model,
        ):
            result_key = (
                _normalize_uri(uri),
                r.start.line,
                r.start.character,
                r.end.line,
                r.end.character,
            )
            if result_key in seen_results:
                continue
            seen_results.add(result_key)
            results.append((uri, r))

    def _scan_path_key(
        path_key: str,
        included_model: Optional[ParsedModel] = None,
    ) -> None:
        real_key = _strip_include_instance_suffix(path_key)
        if real_key in open_by_key:
            uri, source = open_by_key[real_key]
            _scan(uri, source, included_model, path_key)
        else:
            _scan(
                _path_to_uri(real_key),
                workspace_index.get_source(real_key),
                included_model,
                path_key,
            )

    def _scan_include_closure(root_uri: str) -> dict:
        try:
            included = workspace_index.resolve_all_includes(root_uri)
        except Exception:
            logger.exception(
                "references: include resolution failed for %s",
                root_uri,
            )
            return {}
        for inc_path, inc_model in included.items():
            _scan_path_key(inc_path, inc_model)
        return included

    try:
        active_included = workspace_index.resolve_all_includes(active_uri)
    except Exception:
        logger.exception(
            "references: include resolution failed for %s",
            active_uri,
        )
        active_included = {}

    # 1) Active file first so its occurrences sort to the top.
    _scan(active_uri, active_source)

    # 2) Files included by the active document.  If an included file is
    # currently open, prefer its unsaved editor source over the disk cache.
    for inc_path, inc_model in active_included.items():
        _scan_path_key(inc_path, inc_model)

    # 3) Open parent documents whose include graph reaches the active file.
    # Unrelated open models are intentionally skipped; renaming ``y`` in one
    # model must not rewrite a separate open model that happens to use ``y``.
    for uri, source in open_document_sources.items():
        if _normalize_uri(uri) == active_key:
            continue
        try:
            included = workspace_index.resolve_all_includes(uri)
        except Exception:
            logger.exception(
                "references: parent include resolution failed for %s",
                uri,
            )
            continue
        if active_key not in included:
            continue
        _scan(uri, source)
        for inc_path, inc_model in included.items():
            _scan_path_key(inc_path, inc_model)

    return results


@server.feature(lsp.TEXT_DOCUMENT_REFERENCES)
def references(params: lsp.ReferenceParams) -> Optional[List[lsp.Location]]:
    """Find every reference to the symbol under the cursor.

    Scans the active file, included helper files, and open parent files
    that ``@#include`` the active file.  Unrelated open models with the
    same symbol name are excluded from the result.
    """
    uri = params.text_document.uri
    with _state_lock:
        model = _document_models.get(uri)
        open_uris = list(_document_models.keys())
    if model is None:
        return None

    doc = server.workspace.get_text_document(uri)
    lines = doc.source.split("\n")
    if _cursor_inside_comment_or_non_mcp_string(doc.source, lines, params.position):
        return None
    name, word_match = _identifier_at_source_position(
        model, doc.source, params.position
    )
    if name is None or word_match is None:
        return None
    if _reserved_identifier_reason(name) is not None:
        return None
    if name in _model_local_variables(model) and not _is_model_local_symbol_at_position(
        model,
        doc.source,
        params.position,
        name,
    ):
        return None

    # Snapshot the source of every open document so the helper doesn't
    # touch the pygls workspace API.
    open_document_sources: dict = {}
    for open_uri in open_uris:
        if open_uri == uri:
            continue
        try:
            other_doc = server.workspace.get_text_document(open_uri)
            open_document_sources[open_uri] = other_doc.source
        except Exception:
            logger.exception("references: could not read %s", open_uri)

    decl_hit = _find_declaration_across_workspace_with_model(
        uri,
        name,
        _workspace_index,
    )
    local_context = _model_local_declaration_context(
        name,
        decl_hit,
        _workspace_index,
        open_document_sources,
    )
    if _is_model_local_symbol_at_position(model, doc.source, params.position, name):
        refs = _collect_model_local_references(
            uri,
            doc.source,
            model,
            name,
            _workspace_index,
            open_document_sources,
        )
    elif local_context is not None:
        if not _position_is_in_model_equation(model, doc.source, params.position):
            return None
        context_uri, context_source, context_model = local_context
        context_open_sources = dict(open_document_sources)
        if context_uri != uri:
            context_open_sources[uri] = doc.source
        refs = _collect_model_local_references(
            context_uri,
            context_source,
            context_model,
            name,
            _workspace_index,
            context_open_sources,
        )
    else:
        refs = _collect_cross_file_references(
            uri,
            doc.source,
            name,
            _workspace_index,
            open_document_sources,
        )
    if not params.context.include_declaration:
        hit = decl_hit
        if hit is None and name in _model_local_variables(model):
            local_decl = _find_model_local_declaration_in_text(
                doc.source, name
            ) or _find_model_local_declaration(model, name)
            if local_decl is not None:
                hit = (uri, local_decl, None)
        if hit is not None:
            decl_uri, decl, decl_model = hit
            decl_range = (
                _map_range_to_original_source(decl_model, decl.range)
                if decl_model is not None
                else decl.range
            )
            refs = [
                (target_uri, rng)
                for target_uri, rng in refs
                if not (
                    _normalize_uri(target_uri) == _normalize_uri(decl_uri)
                    and rng == decl_range
                )
            ]
    if not refs:
        return None

    def _source_for_ref(target_uri: str) -> Optional[str]:
        if target_uri == uri:
            return doc.source
        if target_uri in open_document_sources:
            return open_document_sources[target_uri]
        return _workspace_index.get_source(target_uri)

    return [
        lsp.Location(
            uri=target_uri,
            range=(
                _to_lsp_range_in_text(source, r)
                if (source := _source_for_ref(target_uri)) is not None
                else _to_lsp_range(r)
            ),
        )
        for target_uri, r in refs
    ]


@server.feature(lsp.TEXT_DOCUMENT_DOCUMENT_HIGHLIGHT)
def document_highlight(
    params: lsp.DocumentHighlightParams,
) -> Optional[List[lsp.DocumentHighlight]]:
    """Highlight every occurrence of the symbol under the cursor in this file.

    Lighter-weight than references() because the result is per-file and the
    editor uses it for visual emphasis only.
    """
    uri = params.text_document.uri
    with _state_lock:
        model = _document_models.get(uri)
    if model is None:
        return None
    doc = server.workspace.get_text_document(uri)
    lines = doc.source.split("\n")
    if _cursor_inside_comment_or_non_mcp_string(doc.source, lines, params.position):
        return None
    name, word_match = _identifier_at_source_position(
        model, doc.source, params.position
    )
    if name is None or word_match is None:
        return None
    if _reserved_identifier_reason(name) is not None:
        return None
    if name in _model_local_variables(model) and not _is_model_local_symbol_at_position(
        model,
        doc.source,
        params.position,
        name,
    ):
        return None

    if _is_model_local_symbol_at_position(model, doc.source, params.position, name):
        occurrences = _find_model_local_occurrences(model, name)
    else:
        occurrences = _find_symbol_occurrences_in_model_source(
            doc.source,
            name,
            model,
        )
    if not occurrences:
        return None

    return [
        lsp.DocumentHighlight(
            range=_to_lsp_range_in_text(doc.source, r),
            kind=lsp.DocumentHighlightKind.Read,
        )
        for r in occurrences
    ]


# Dynare's NAME token allows a leading underscore.
_VALID_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _build_rename_edit(
    active_uri: str,
    active_source: str,
    old_name: str,
    new_name: str,
    workspace_index: WorkspaceIndex,
    open_document_sources: dict,
) -> Optional[lsp.WorkspaceEdit]:
    """Build a multi-file ``WorkspaceEdit`` for renaming *old_name*.

    Scope mirrors :func:`_collect_cross_file_references`: every file
    reachable from the active document via the open-document set and
    the transitive ``@#include`` graph.  Returns ``None`` if the new
    name is not a valid identifier or if no occurrences are found.
    """
    if not _VALID_IDENTIFIER_RE.match(new_name):
        return None
    if _reserved_identifier_reason(old_name) is not None:
        return None
    if _reserved_identifier_reason(new_name) is not None:
        return None

    refs = _collect_cross_file_references(
        active_uri,
        active_source,
        old_name,
        workspace_index,
        open_document_sources,
    )
    if not refs:
        return None

    grouped: dict = {}
    for target_uri, r in refs:
        if target_uri == active_uri:
            source = active_source
        elif target_uri in open_document_sources:
            source = open_document_sources[target_uri]
        else:
            source = workspace_index.get_source(target_uri)
        if source is not None:
            lines = source.split("\n")
            if r.start.line >= len(lines):
                return None
            source_slice = lines[r.start.line][r.start.character : r.end.character]
            if source_slice != old_name:
                return None
        grouped.setdefault(target_uri, []).append(
            lsp.TextEdit(
                range=(
                    _to_lsp_range_in_text(source, r)
                    if source is not None
                    else _to_lsp_range(r)
                ),
                new_text=new_name,
            )
        )
    return lsp.WorkspaceEdit(changes=grouped)


def _reference_ranges_are_exact_source_slices(
    active_uri: str,
    active_source: str,
    old_name: str,
    refs: List[Tuple[str, DRange]],
    workspace_index: WorkspaceIndex,
    open_document_sources: dict,
) -> bool:
    for target_uri, rng in refs:
        if target_uri == active_uri:
            source = active_source
        elif target_uri in open_document_sources:
            source = open_document_sources[target_uri]
        else:
            source = workspace_index.get_source(target_uri)
        if source is None:
            continue
        lines = source.split("\n")
        if rng.start.line >= len(lines):
            return False
        if lines[rng.start.line][rng.start.character : rng.end.character] != old_name:
            return False
    return True


def _cursor_inside_comment_or_string(lines: List[str], pos: lsp.Position) -> bool:
    """Return True if *pos* falls inside non-code text.

    Non-code includes ``//``, ``%``, ``/* */`` comments, ``"..."`` /
    ``'...'`` string literals, and Dynare ``@#`` macro directive lines.

    Used by ``prepare_rename`` / ``rename`` to refuse a rename whose
    cursor sits inside a non-identifier zone — without this check, the
    LSP would happily rename every occurrence of the identifier spelled
    inside a comment even though the user never meant to touch real code.
    """
    if pos.line >= len(lines):
        return False
    pos = _lsp_position_to_source_position(lines, pos)
    line = lines[pos.line]
    macro_match = re.match(r"^[ \t]*@\#", line)
    if macro_match and pos.character >= macro_match.start():
        return True
    for macro_interp in _MACRO_INTERPOLATION_RE.finditer(line):
        if macro_interp.start() <= pos.character < macro_interp.end():
            return True

    # First, determine whether the cursor's line started inside an
    # unmatched ``/*`` from a previous line.  We must respect strings
    # and line comments on those previous lines, otherwise a ``/*``
    # that lives inside a string literal (``[name='/* tag']``) or
    # after a ``//`` / ``%`` line comment would falsely mark the
    # cursor's line as inside a block comment.
    in_block = False
    for prev_line in lines[: pos.line]:
        k = 0
        in_string_pq: Optional[str] = None
        n_prev = len(prev_line)
        while k < n_prev:
            if in_block:
                idx = prev_line.find("*/", k)
                if idx == -1:
                    k = n_prev
                    break
                in_block = False
                k = idx + 2
                continue
            ch = prev_line[k]
            nxt = prev_line[k + 1] if k + 1 < n_prev else ""
            if in_string_pq is not None:
                if ch == in_string_pq:
                    in_string_pq = None
                k += 1
                continue
            if ch in ('"', "'"):
                in_string_pq = ch
                k += 1
                continue
            if ch == "/" and nxt == "/":
                break  # rest of line is a line comment — no block opener
            if ch == "%":
                break  # rest of line is a Matlab-style line comment
            if ch == "/" and nxt == "*":
                in_block = True
                k += 2
                continue
            k += 1
        # Line comments and unterminated single-line strings reset at EOL.
        in_string_pq = None

    # Now walk the cursor's line, tracking block-comment, line-comment,
    # and string-literal state.  This handles SAME-LINE block comments
    # like ``/* alpha */`` (which an earlier version missed) by checking
    # ``/*`` and ``*/`` tokens inline alongside the other states.
    in_string_q: Optional[str] = None
    i = 0
    limit = min(pos.character + 1, len(line))
    while i < limit:
        ch = line[i]
        nxt = line[i + 1] if i + 1 < len(line) else ""
        if in_block:
            # Look for the closing ``*/`` at or before the cursor.
            if ch == "*" and nxt == "/":
                in_block = False
                i += 2
                continue
            i += 1
            continue
        if in_string_q is not None:
            if ch == in_string_q:
                in_string_q = None
            i += 1
            continue
        if ch == "/" and nxt == "*":
            in_block = True
            i += 2
            continue
        if ch == "/" and nxt == "/":
            return i <= pos.character
        if ch == "%":
            return i <= pos.character
        if ch in ('"', "'"):
            in_string_q = ch
            i += 1
            continue
        i += 1

    if in_block or in_string_q is not None:
        return True
    return False


@server.feature(lsp.TEXT_DOCUMENT_PREPARE_RENAME)
def prepare_rename(
    params: lsp.PrepareRenameParams,
) -> Optional[lsp.Range]:
    """Validate the cursor position before the editor opens the rename UI.

    Returns the range of the identifier under the cursor when one is
    present, so the editor can highlight what will be renamed and
    reject (cleanly) attempts to rename whitespace, punctuation, or
    inside a comment / string literal.
    """
    uri = params.text_document.uri
    doc = server.workspace.get_text_document(uri)
    lines = doc.source.split("\n")
    if _cursor_inside_comment_or_non_mcp_string(doc.source, lines, params.position):
        return None
    with _state_lock:
        model = _document_models.get(uri)
        open_uris = list(_document_models.keys())
    if model is None:
        return None
    old_name, word_match = _identifier_at_source_position(
        model,
        doc.source,
        params.position,
    )
    if old_name is None or word_match is None:
        return None
    if _is_on_the_fly_kind_marker(
        lines[params.position.line],
        word_match.start(),
        word_match.end(),
    ):
        return None
    if (
        model is not None
        and old_name in _model_local_variables(model)
        and not _is_model_local_symbol_at_position(
            model,
            doc.source,
            params.position,
            old_name,
        )
    ):
        return None
    decl_hit = _find_declaration_across_workspace_with_model(
        uri,
        old_name,
        _workspace_index,
    )
    if decl_hit is None and old_name in _model_local_variables(model):
        local_decl = _find_model_local_declaration_in_text(
            doc.source, old_name
        ) or _find_model_local_declaration(model, old_name)
        if local_decl is not None:
            decl_hit = (uri, local_decl, model)
    if decl_hit is None:
        return None
    open_document_sources: dict = {}
    for open_uri in open_uris:
        if open_uri == uri:
            continue
        try:
            other_doc = server.workspace.get_text_document(open_uri)
            open_document_sources[open_uri] = other_doc.source
        except Exception:
            logger.exception("prepare rename: could not read %s", open_uri)
    local_context = _model_local_declaration_context(
        old_name,
        decl_hit,
        _workspace_index,
        open_document_sources,
    )
    if _is_model_local_symbol_at_position(model, doc.source, params.position, old_name):
        refs = _collect_model_local_references(
            uri,
            doc.source,
            model,
            old_name,
            _workspace_index,
            open_document_sources,
        )
    elif local_context is not None:
        if not _position_is_in_model_equation(model, doc.source, params.position):
            return None
        context_uri, context_source, context_model = local_context
        context_open_sources = dict(open_document_sources)
        if context_uri != uri:
            context_open_sources[uri] = doc.source
        refs = _collect_model_local_references(
            context_uri,
            context_source,
            context_model,
            old_name,
            _workspace_index,
            context_open_sources,
        )
    else:
        refs = _collect_cross_file_references(
            uri,
            doc.source,
            old_name,
            _workspace_index,
            open_document_sources,
        )
    if not _reference_ranges_are_exact_source_slices(
        uri,
        doc.source,
        old_name,
        refs,
        _workspace_index,
        open_document_sources,
    ):
        return None
    return lsp.Range(
        start=_source_position_to_lsp_position(
            lines,
            DPos(params.position.line, word_match.start()),
        ),
        end=_source_position_to_lsp_position(
            lines,
            DPos(params.position.line, word_match.end()),
        ),
    )


@server.feature(
    lsp.TEXT_DOCUMENT_RENAME,
    lsp.RenameOptions(prepare_provider=True),
)
def rename(params: lsp.RenameParams) -> Optional[lsp.WorkspaceEdit]:
    """Rename the symbol under the cursor across the whole include graph.

    Touches the active file, its included helper files, and open parent
    files that ``@#include`` the active file.  Unrelated open models are
    excluded so a rename cannot rewrite a separate model with the same
    local identifier.
    """
    uri = params.text_document.uri
    doc = server.workspace.get_text_document(uri)
    lines = doc.source.split("\n")
    # Reject the rename if the cursor sits inside a comment or string
    # literal.  Belt-and-braces with ``prepare_rename`` since clients
    # may dispatch rename without invoking the prepare step.
    if _cursor_inside_comment_or_non_mcp_string(doc.source, lines, params.position):
        return None
    with _state_lock:
        model = _document_models.get(uri)
    if model is None:
        return None
    old_name, word_match = _identifier_at_source_position(
        model,
        doc.source,
        params.position,
    )
    if old_name is None or word_match is None:
        return None
    if _is_on_the_fly_kind_marker(
        lines[params.position.line],
        word_match.start(),
        word_match.end(),
    ):
        return None
    if (
        model is not None
        and old_name in _model_local_variables(model)
        and not _is_model_local_symbol_at_position(
            model,
            doc.source,
            params.position,
            old_name,
        )
    ):
        return None
    decl_hit = _find_declaration_across_workspace_with_model(
        uri,
        old_name,
        _workspace_index,
    )
    if decl_hit is None and old_name in _model_local_variables(model):
        local_decl = _find_model_local_declaration_in_text(
            doc.source, old_name
        ) or _find_model_local_declaration(model, old_name)
        if local_decl is not None:
            decl_hit = (uri, local_decl, model)
    if decl_hit is None:
        return None

    with _state_lock:
        open_uris = list(_document_models.keys())
    open_document_sources: dict = {}
    for open_uri in open_uris:
        if open_uri == uri:
            continue
        try:
            other_doc = server.workspace.get_text_document(open_uri)
            open_document_sources[open_uri] = other_doc.source
        except Exception:
            logger.exception("rename: could not read %s", open_uri)

    local_context = _model_local_declaration_context(
        old_name,
        decl_hit,
        _workspace_index,
        open_document_sources,
    )
    if model is not None and _is_model_local_symbol_at_position(
        model,
        doc.source,
        params.position,
        old_name,
    ):
        return _build_model_local_context_rename_edit(
            uri,
            doc.source,
            model,
            old_name,
            params.new_name,
            _workspace_index,
            open_document_sources,
        )
    if local_context is not None:
        if not _position_is_in_model_equation(model, doc.source, params.position):
            return None
        context_uri, context_source, context_model = local_context
        context_open_sources = dict(open_document_sources)
        if context_uri != uri:
            context_open_sources[uri] = doc.source
        return _build_model_local_context_rename_edit(
            context_uri,
            context_source,
            context_model,
            old_name,
            params.new_name,
            _workspace_index,
            context_open_sources,
        )

    return _build_rename_edit(
        uri,
        doc.source,
        old_name,
        params.new_name,
        _workspace_index,
        open_document_sources,
    )


# ---------------------------------------------------------------------------
# Declaration / Type Definition (aliases over definition)
# ---------------------------------------------------------------------------


@server.feature(lsp.TEXT_DOCUMENT_DECLARATION)
def declaration(params: lsp.DeclarationParams) -> Optional[lsp.Location]:
    """Alias for go-to-definition; Dynare has no separate declaration concept."""
    return definition(
        lsp.DefinitionParams(
            text_document=params.text_document,
            position=params.position,
        )
    )


@server.feature(lsp.TEXT_DOCUMENT_TYPE_DEFINITION)
def type_definition(
    params: lsp.TypeDefinitionParams,
) -> Optional[lsp.Location]:
    """Alias for go-to-definition; Dynare identifiers don't have a distinct
    type-vs-declaration distinction."""
    return definition(
        lsp.DefinitionParams(
            text_document=params.text_document,
            position=params.position,
        )
    )


# ---------------------------------------------------------------------------
# Signature Help — parameter hints for built-in math functions
# ---------------------------------------------------------------------------

# (label, documentation, parameter labels). Kept in sync with
# ``_BUILTIN_FUNCTIONS`` in the completion section.
_FUNCTION_SIGNATURES = {
    "exp": ("exp(x)", "Exponential function", ["x"]),
    "log": ("log(x)", "Natural logarithm", ["x"]),
    "ln": ("ln(x)", "Natural logarithm (alias for log)", ["x"]),
    "log2": ("log2(x)", "Base-2 logarithm", ["x"]),
    "log10": ("log10(x)", "Base-10 logarithm", ["x"]),
    "sqrt": ("sqrt(x)", "Square root", ["x"]),
    "abs": ("abs(x)", "Absolute value", ["x"]),
    "sign": ("sign(x)", "Sign function: -1, 0, or 1", ["x"]),
    "sin": ("sin(x)", "Sine", ["x"]),
    "cos": ("cos(x)", "Cosine", ["x"]),
    "tan": ("tan(x)", "Tangent", ["x"]),
    "asin": ("asin(x)", "Inverse sine", ["x"]),
    "acos": ("acos(x)", "Inverse cosine", ["x"]),
    "atan": ("atan(x)", "Inverse tangent", ["x"]),
    "min": ("min(a, b)", "Minimum of two values", ["a", "b"]),
    "max": ("max(a, b)", "Maximum of two values", ["a", "b"]),
    "erf": ("erf(x)", "Error function", ["x"]),
    "normpdf": ("normpdf(x, mu, sigma)", "Normal PDF", ["x", "mu", "sigma"]),
    "normcdf": ("normcdf(x, mu, sigma)", "Normal CDF", ["x", "mu", "sigma"]),
    "STEADY_STATE": (
        "STEADY_STATE(x)",
        "Reference the steady-state value of a variable",
        ["x"],
    ),
    "EXPECTATION": (
        "EXPECTATION(t)(x)",
        "Expectation operator at horizon t",
        ["t", "x"],
    ),
}


@server.feature(
    lsp.TEXT_DOCUMENT_SIGNATURE_HELP,
    lsp.SignatureHelpOptions(trigger_characters=["(", ","]),
)
def signature_help(
    params: lsp.SignatureHelpParams,
) -> Optional[lsp.SignatureHelp]:
    """Show parameter hints for built-in math and Dynare functions."""
    uri = params.text_document.uri
    doc = server.workspace.get_text_document(uri)
    lines = doc.source.split("\n")
    if params.position.line >= len(lines):
        return None
    line = lines[params.position.line]
    source_pos = _lsp_position_to_source_position(lines, params.position)
    before = line[: source_pos.character]

    # Walk back to find the most recent unclosed `name(` and the args so far
    m = re.search(r"([A-Za-z_][A-Za-z_0-9]*)\s*\(([^()]*)$", before)
    if m is None:
        return None
    name = m.group(1)
    args_so_far = m.group(2)
    sig = _FUNCTION_SIGNATURES.get(name)
    if sig is None:
        return None
    label, doc_str, param_labels = sig

    active_param = min(args_so_far.count(","), max(0, len(param_labels) - 1))
    return lsp.SignatureHelp(
        signatures=[
            lsp.SignatureInformation(
                label=label,
                documentation=doc_str,
                parameters=[lsp.ParameterInformation(label=p) for p in param_labels],
            )
        ],
        active_signature=0,
        active_parameter=active_param,
    )


# ---------------------------------------------------------------------------
# Code Lens — inline action prompts on solver-relevant blocks
# ---------------------------------------------------------------------------


@server.feature(lsp.TEXT_DOCUMENT_CODE_LENS)
def code_lens(params: lsp.CodeLensParams) -> Optional[List[lsp.CodeLens]]:
    """Display inline "Compute Steady State" actions above relevant blocks."""
    uri = params.text_document.uri
    with _state_lock:
        model = _document_models.get(uri)
    if model is None:
        return None

    lenses: List[lsp.CodeLens] = []

    target_range: Optional[DRange] = None
    title = "▶ Compute steady state"
    if model.steady_state_block_range is not None:
        target_range = model.steady_state_block_range
        title = "▶ Compute steady state (insert initval block / update initval block)"
    elif model.initval_block_range is not None:
        target_range = model.initval_block_range
        title = "▶ Compute steady state (update initval block)"
    elif model.model_block_range is not None:
        target_range = model.model_block_range
        title = "▶ Compute steady state (no steady_state_model / initval block)"

    if target_range is not None:
        anchor_line = target_range.start.line
        lenses.append(
            lsp.CodeLens(
                range=lsp.Range(
                    start=lsp.Position(line=anchor_line, character=0),
                    end=lsp.Position(line=anchor_line, character=0),
                ),
                command=lsp.Command(
                    title=title,
                    command="dynare/computeSteadyState",
                    arguments=[{"uri": uri}],
                ),
            )
        )

    # Static model-structure summary above the model block (no solve required).
    if model.model_block_range is not None and model.endogenous:
        from .model_info import classify_variable_timing

        timing = classify_variable_timing(model)
        states = sum(
            1 for t in timing.values() if t["class"] in ("predetermined", "mixed")
        )
        jumpers = sum(
            1 for t in timing.values() if t["class"] in ("forward_looking", "mixed")
        )
        static_ct = sum(1 for t in timing.values() if t["class"] == "static")
        offsets = [o for t in timing.values() for o in t["offsets"]]
        max_lead = max([o for o in offsets if o > 0], default=0)
        max_lag = min([o for o in offsets if o < 0], default=0)
        summary = (
            f"ℹ {len(model.endogenous)} endogenous: "
            f"{states} state, {jumpers} jumper, {static_ct} static"
            f" · {len(model.exogenous)} varexo"
            f" · max lead {max_lead}, max lag {max_lag}"
        )
        model_line = model.model_block_range.start.line
        lenses.append(
            lsp.CodeLens(
                range=lsp.Range(
                    start=lsp.Position(line=model_line, character=0),
                    end=lsp.Position(line=model_line, character=0),
                ),
                # Empty command -> informational (non-clickable) lens.
                command=lsp.Command(title=summary, command=""),
            )
        )

    # Run actions above the model block: validate with the preprocessor, or
    # execute end-to-end with MATLAB + Dynare and surface the results.
    if model.model_block_range is not None:
        run_line = model.model_block_range.start.line
        run_anchor = lsp.Range(
            start=lsp.Position(line=run_line, character=0),
            end=lsp.Position(line=run_line, character=0),
        )
        for run_title, run_command in (
            ("▶ Run preprocessor (check)", "dynare/runPreprocessor"),
            ("▶ Run with Dynare (MATLAB)", "dynare/runDynare"),
        ):
            lenses.append(
                lsp.CodeLens(
                    range=run_anchor,
                    command=lsp.Command(
                        title=run_title,
                        command=run_command,
                        arguments=[{"uri": uri}],
                    ),
                )
            )

    return lenses if lenses else None


# ---------------------------------------------------------------------------
# Document formatting — semantics-preserving whitespace normalization
# ---------------------------------------------------------------------------


@server.feature(lsp.TEXT_DOCUMENT_FORMATTING)
def formatting(
    params: lsp.DocumentFormattingParams,
) -> Optional[List[lsp.TextEdit]]:
    """Reformat the whole document (tab indent, operator spacing, aligned ``=``).

    Returns no edits when the formatter declines (the change would not be a
    pure-whitespace, meaning-preserving edit, or the file is empty).
    """
    from .formatter import format_text

    uri = params.text_document.uri
    doc = server.workspace.get_text_document(uri)
    text = doc.source
    formatted = format_text(text, _format_indent_unit)
    if formatted is None:
        return None
    return [_full_document_edit(text, formatted)]


@server.feature(lsp.TEXT_DOCUMENT_RANGE_FORMATTING)
def range_formatting(
    params: lsp.DocumentRangeFormattingParams,
) -> Optional[List[lsp.TextEdit]]:
    """Reformat the full lines spanned by the selection."""
    from .formatter import format_range

    uri = params.text_document.uri
    doc = server.workspace.get_text_document(uri)
    text = doc.source
    # LSP ranges are half-open: a selection ending at column 0 of the next line
    # does not include that line, so don't reformat it.
    end_line = params.range.end.line
    if params.range.end.character == 0 and end_line > params.range.start.line:
        end_line -= 1
    result = format_range(
        text,
        params.range.start.line,
        end_line,
        _format_indent_unit,
    )
    if result is None:
        return None
    start_line, end_line, replacement = result
    lines = text.split("\n")
    end_char = _source_index_to_lsp_character(lines[end_line], len(lines[end_line]))
    return [
        lsp.TextEdit(
            range=lsp.Range(
                start=lsp.Position(line=start_line, character=0),
                end=lsp.Position(line=end_line, character=end_char),
            ),
            new_text=replacement,
        )
    ]


def _full_document_edit(old_text: str, new_text: str) -> lsp.TextEdit:
    """A single TextEdit replacing the whole document."""
    lines = old_text.split("\n")
    last = lines[-1]
    return lsp.TextEdit(
        range=lsp.Range(
            start=lsp.Position(line=0, character=0),
            end=lsp.Position(
                line=len(lines) - 1,
                character=_source_index_to_lsp_character(last, len(last)),
            ),
        ),
        new_text=new_text,
    )


# ---------------------------------------------------------------------------
# Selection range — smart expand/shrink: identifier -> equation -> block -> file
# ---------------------------------------------------------------------------


def _lsp_pos_le(a: lsp.Position, b: lsp.Position) -> bool:
    return (a.line, a.character) <= (b.line, b.character)


def _lsp_range_has_pos(rng: lsp.Range, pos: lsp.Position) -> bool:
    return _lsp_pos_le(rng.start, pos) and _lsp_pos_le(pos, rng.end)


def _lsp_strictly_contains(outer: lsp.Range, inner: lsp.Range) -> bool:
    if outer.start == inner.start and outer.end == inner.end:
        return False
    return _lsp_pos_le(outer.start, inner.start) and _lsp_pos_le(inner.end, outer.end)


def _selection_chain(
    model: ParsedModel,
    lines: List[str],
    pos: lsp.Position,
) -> List[lsp.Range]:
    """Candidate ranges around *pos*, innermost first (identifier..file)."""
    candidates: List[lsp.Range] = []

    src_pos = _lsp_position_to_source_position(lines, pos)
    word = _word_at_position(lines, src_pos)
    if word is not None and src_pos.line < len(lines):
        line_text = lines[src_pos.line]
        start_c = _source_index_to_lsp_character(line_text, word.start())
        end_c = _source_index_to_lsp_character(line_text, word.end())
        candidates.append(
            lsp.Range(
                start=lsp.Position(line=src_pos.line, character=start_c),
                end=lsp.Position(line=src_pos.line, character=end_c),
            )
        )

    for equation in (*model.model_equations, *model.steady_state_equations):
        rng = _to_lsp_range_for_model(model, equation.range)
        if _lsp_range_has_pos(rng, pos):
            candidates.append(rng)
            break

    for block in (
        model.model_block_range,
        model.steady_state_block_range,
        model.initval_block_range,
        model.endval_block_range,
        model.shocks_block_range,
    ):
        if block is None:
            continue
        rng = _to_lsp_range_for_model(model, block)
        if _lsp_range_has_pos(rng, pos):
            candidates.append(rng)

    last_line = max(len(lines) - 1, 0)
    candidates.append(
        lsp.Range(
            start=lsp.Position(line=0, character=0),
            end=lsp.Position(
                line=last_line,
                character=_source_index_to_lsp_character(
                    lines[last_line] if lines else "",
                    len(lines[last_line]) if lines else 0,
                ),
            ),
        )
    )

    # Innermost first: largest start, then smallest end.
    candidates.sort(
        key=lambda r: (-r.start.line, -r.start.character, r.end.line, r.end.character),
    )
    chain: List[lsp.Range] = []
    for rng in candidates:
        if not chain:
            chain.append(rng)
        elif _lsp_strictly_contains(rng, chain[-1]):
            chain.append(rng)
    return chain


@server.feature(lsp.TEXT_DOCUMENT_SELECTION_RANGE)
def selection_range(
    params: lsp.SelectionRangeParams,
) -> Optional[List[lsp.SelectionRange]]:
    """Smart expand-selection: identifier -> equation -> block -> file."""
    uri = params.text_document.uri
    with _state_lock:
        model = _document_models.get(uri)
    if model is None:
        return None
    doc = server.workspace.get_text_document(uri)
    lines = doc.source.split("\n")

    result: List[lsp.SelectionRange] = []
    for pos in params.positions:
        chain = _selection_chain(model, lines, pos)
        node: Optional[lsp.SelectionRange] = None
        for rng in reversed(chain):  # outermost first; build parent chain inward
            node = lsp.SelectionRange(range=rng, parent=node)
        if node is None:
            node = lsp.SelectionRange(
                range=lsp.Range(start=pos, end=pos),
            )
        result.append(node)
    return result


# ---------------------------------------------------------------------------
# Linked editing — live-rename a symbol's occurrences as you type
# ---------------------------------------------------------------------------


@server.feature(lsp.TEXT_DOCUMENT_LINKED_EDITING_RANGE)
def linked_editing_range(
    params: lsp.LinkedEditingRangeParams,
) -> Optional[lsp.LinkedEditingRanges]:
    """Group a declared symbol's occurrences so the editor edits them together."""
    uri = params.text_document.uri
    with _state_lock:
        model = _document_models.get(uri)
    if model is None:
        return None
    doc = server.workspace.get_text_document(uri)
    lines = doc.source.split("\n")
    if _cursor_inside_comment_or_string(lines, params.position):
        return None
    word = _word_at_lsp_position(lines, params.position)
    if word is None:
        return None
    name = word.group(1)
    # Restrict to file-global declared symbols (var / varexo / parameters);
    # model-local ``#`` variables have subtler scoping handled by rename.
    if name not in model.all_declared_names():
        return None
    occurrences = _find_symbol_occurrences(doc.source, name)
    if len(occurrences) < 2:
        return None
    return lsp.LinkedEditingRanges(
        ranges=[_to_lsp_range_in_text(doc.source, r) for r in occurrences],
    )


# ---------------------------------------------------------------------------
# Workspace pull diagnostics — every known document in one request
# ---------------------------------------------------------------------------


@server.feature(lsp.WORKSPACE_DIAGNOSTIC)
def workspace_diagnostic(
    params: lsp.WorkspaceDiagnosticParams,
) -> lsp.WorkspaceDiagnosticReport:
    """Report cached diagnostics for every document the server has validated."""
    with _state_lock:
        uris = list(_document_diagnostics.keys())
    items = [
        lsp.WorkspaceFullDocumentDiagnosticReport(
            uri=uri,
            version=None,
            items=_gather_all_diagnostics(uri),
        )
        for uri in uris
    ]
    return lsp.WorkspaceDiagnosticReport(items=items)


# ---------------------------------------------------------------------------
# Call hierarchy — variable <-> equation dependency navigation
#
# Repurposed: a *variable* item's "incoming calls" are the equations that
# reference it (with the referencing ranges); an *equation* item's "outgoing
# calls" are the variables it uses.  This gives "where is k used?" and "what
# does this equation depend on?" navigation in the editor's call-hierarchy view.
# ---------------------------------------------------------------------------


def _occ_within(occ: DRange, container: DRange) -> bool:
    start = (occ.start.line, occ.start.character)
    cs = (container.start.line, container.start.character)
    ce = (container.end.line, container.end.character)
    return cs <= start <= ce


def _equation_label(eq) -> str:
    if getattr(eq, "name", ""):
        return eq.name
    lhs = (getattr(eq, "lhs", "") or "").strip()
    if lhs:
        return f"{lhs} = ..."
    text = (eq.text or "").strip().splitlines()
    return text[0][:48] if text else "equation"


def _ch_variable_item(model: ParsedModel, name: str, uri: str) -> lsp.CallHierarchyItem:
    decl = _find_declaration(model, name)
    rng = (
        _to_lsp_range_for_model(model, decl.range)
        if decl is not None and decl.range is not None
        else lsp.Range(lsp.Position(0, 0), lsp.Position(0, 0))
    )
    return lsp.CallHierarchyItem(
        name=name,
        kind=lsp.SymbolKind.Variable,
        uri=uri,
        range=rng,
        selection_range=rng,
        detail="variable",
        data={"dynare": "variable", "name": name},
    )


def _ch_equation_item(model: ParsedModel, eq, uri: str) -> lsp.CallHierarchyItem:
    rng = _to_lsp_range_for_model(model, eq.range)
    return lsp.CallHierarchyItem(
        name=_equation_label(eq),
        kind=lsp.SymbolKind.Function,
        uri=uri,
        range=rng,
        selection_range=rng,
        detail="equation",
        data={
            "dynare": "equation",
            "line": eq.range.start.line,
            "char": eq.range.start.character,
        },
    )


def _equation_at_lsp_position(model: ParsedModel, pos: lsp.Position):
    for eq in model.model_equations:
        if eq.text.strip().startswith("#"):
            continue
        rng = _to_lsp_range_for_model(model, eq.range)
        if (
            (rng.start.line, rng.start.character)
            <= (pos.line, pos.character)
            <= (rng.end.line, rng.end.character)
        ):
            return eq
    return None


@server.feature(lsp.TEXT_DOCUMENT_PREPARE_CALL_HIERARCHY)
def prepare_call_hierarchy(
    params: lsp.CallHierarchyPrepareParams,
) -> Optional[List[lsp.CallHierarchyItem]]:
    """Offer a variable item and/or the enclosing equation item at the cursor."""
    uri = params.text_document.uri
    with _state_lock:
        model = _document_models.get(uri)
    if model is None:
        return None
    doc = server.workspace.get_text_document(uri)
    lines = doc.source.split("\n")
    if _cursor_inside_comment_or_string(lines, params.position):
        return None

    items: List[lsp.CallHierarchyItem] = []
    word = _word_at_lsp_position(lines, params.position)
    if word is not None and word.group(1) in model.all_declared_names():
        items.append(_ch_variable_item(model, word.group(1), uri))
    eq = _equation_at_lsp_position(model, params.position)
    if eq is not None:
        items.append(_ch_equation_item(model, eq, uri))
    return items or None


@server.feature(lsp.CALL_HIERARCHY_INCOMING_CALLS)
def call_hierarchy_incoming(
    params: lsp.CallHierarchyIncomingCallsParams,
) -> Optional[List[lsp.CallHierarchyIncomingCall]]:
    """For a variable item: the equations that reference it."""
    data = params.item.data if isinstance(params.item.data, dict) else {}
    if data.get("dynare") != "variable":
        return []
    uri = params.item.uri
    with _state_lock:
        model = _document_models.get(uri)
    if model is None:
        return None
    doc = server.workspace.get_text_document(uri)
    name = data.get("name")
    occurrences = _find_symbol_occurrences(doc.source, name) if name else []

    calls: List[lsp.CallHierarchyIncomingCall] = []
    for eq in model.model_equations:
        if eq.text.strip().startswith("#"):
            continue
        eq_src = _map_range_to_original_source(model, eq.range)
        in_eq = [o for o in occurrences if _occ_within(o, eq_src)]
        if in_eq:
            calls.append(
                lsp.CallHierarchyIncomingCall(
                    from_=_ch_equation_item(model, eq, uri),
                    from_ranges=[_to_lsp_range_in_text(doc.source, o) for o in in_eq],
                )
            )
    return calls


@server.feature(lsp.CALL_HIERARCHY_OUTGOING_CALLS)
def call_hierarchy_outgoing(
    params: lsp.CallHierarchyOutgoingCallsParams,
) -> Optional[List[lsp.CallHierarchyOutgoingCall]]:
    """For an equation item: the variables it references."""
    data = params.item.data if isinstance(params.item.data, dict) else {}
    if data.get("dynare") != "equation":
        return []
    uri = params.item.uri
    with _state_lock:
        model = _document_models.get(uri)
    if model is None:
        return None
    doc = server.workspace.get_text_document(uri)
    eq = next(
        (
            e
            for e in model.model_equations
            if e.range.start.line == data.get("line")
            and e.range.start.character == data.get("char", e.range.start.character)
        ),
        None,
    )
    if eq is None:
        return []
    eq_src = _map_range_to_original_source(model, eq.range)

    calls: List[lsp.CallHierarchyOutgoingCall] = []
    for var in (*model.endogenous, *model.exogenous):
        occurrences = [
            o
            for o in _find_symbol_occurrences(doc.source, var.name)
            if _occ_within(o, eq_src)
        ]
        if occurrences:
            calls.append(
                lsp.CallHierarchyOutgoingCall(
                    to=_ch_variable_item(model, var.name, uri),
                    from_ranges=[
                        _to_lsp_range_in_text(doc.source, o) for o in occurrences
                    ],
                )
            )
    return calls


# ---------------------------------------------------------------------------
# Workspace Symbols — search variables/parameters across all open documents
# ---------------------------------------------------------------------------


def _collect_workspace_symbols(
    query: str,
    document_models: dict,
    workspace_index: WorkspaceIndex,
) -> List[lsp.WorkspaceSymbol]:
    """Build the ``workspace/symbol`` response payload.

    Walks every open document plus the transitive include graph reachable
    from each opened URI.  Names are dedup'd on the normalised absolute
    path of the source file so the LSP-client URI form (e.g.
    ``file:///c%3A/foo.mod``) and the include's ``Path.as_uri()`` form
    of the same file collapse to a single emission.

    Pulled out of the ``WORKSPACE_SYMBOL`` handler so the cross-file
    behaviour can be tested directly without spinning up pygls.
    """
    query_lower = (query or "").lower()
    symbols: List[lsp.WorkspaceSymbol] = []
    seen: set = set()  # (normalized_path, name, kind_str)

    def _emit(
        uri: str,
        kind_str: str,
        decl: VarDeclaration,
        source_model: ParsedModel,
    ) -> None:
        if query_lower and query_lower not in decl.name.lower():
            return
        dedup_key = (_normalize_uri(uri), decl.name, kind_str)
        if dedup_key in seen:
            return
        seen.add(dedup_key)
        symbols.append(
            lsp.WorkspaceSymbol(
                name=decl.name,
                kind=(
                    lsp.SymbolKind.Constant
                    if kind_str == "parameters"
                    else lsp.SymbolKind.Variable
                ),
                location=lsp.Location(
                    uri=uri,
                    range=_to_lsp_range_for_model(source_model, decl.range),
                ),
                container_name=kind_str,
            )
        )

    for uri, model in document_models.items():
        for kind_str, decls in (
            ("var", model.endogenous),
            ("varexo", model.exogenous),
            ("parameters", model.parameters),
        ):
            for decl in decls:
                _emit(uri, kind_str, decl, model)

        # Also include declarations from transitively-included files.
        # Each included path is exposed as its own file:// URI so the
        # editor can jump straight to the declaration source.
        try:
            included = workspace_index.resolve_all_includes(uri)
        except Exception:
            logger.exception("workspace_symbol: include resolution failed for %s", uri)
            continue
        for inc_path, inc_model in included.items():
            inc_uri = _path_to_uri(inc_path)
            for kind_str, decls in (
                ("var", inc_model.endogenous),
                ("varexo", inc_model.exogenous),
                ("parameters", inc_model.parameters),
            ):
                for decl in decls:
                    _emit(inc_uri, kind_str, decl, inc_model)

    return symbols


@server.feature(lsp.WORKSPACE_SYMBOL)
def workspace_symbol(
    params: lsp.WorkspaceSymbolParams,
) -> Optional[List[lsp.WorkspaceSymbol]]:
    """Search declarations across every open .mod document and their includes.

    LLM-friendly: an agent working on a multi-file project can locate a
    declaration in any open file — or in any file reachable via an
    ``@#include`` chain from an open file — in a single round-trip.
    """
    with _state_lock:
        document_models = dict(_document_models)
    symbols = _collect_workspace_symbols(
        params.query or "",
        document_models,
        _workspace_index,
    )
    return symbols if symbols else None


# ---------------------------------------------------------------------------
# Semantic Tokens — LSP-driven syntax classification
# ---------------------------------------------------------------------------
#
# Distinguishes endogenous variables (`var`), exogenous variables (`varexo`),
# and parameters visually in the editor — three classes that the static
# TextMate grammar in the VS Code extension can't distinguish because they
# look identical at the lexical level. This pulls the LSP's symbol table
# through to the highlighter.

_SEMANTIC_TOKEN_TYPES = [
    "variable",  # 0 — endogenous (var)
    "type",  # 1 — exogenous (varexo) — repurposed for visual distinction
    "macro",  # 2 — parameter — repurposed for visual distinction
    "parameter",  # 3 — model-local (#) variable — repurposed; previously endo
]
# Appended (was empty): index 0 declaration, 1 forward-looking (jumper),
# 2 predetermined (state).  Indices are stable so the legend stays
# backward-compatible.
_SEMANTIC_TOKEN_MODIFIERS: List[str] = [
    "declaration",
    "forwardLooking",
    "predetermined",
]
_MOD_DECLARATION = 1 << 0
_MOD_FORWARD = 1 << 1
_MOD_PREDETERMINED = 1 << 2


def _encode_semantic_tokens(tokens: List[tuple]) -> List[int]:
    """Delta-encode a sorted list of (line, col, length, ttype, mods) tuples."""
    data: List[int] = []
    prev_line = 0
    prev_col = 0
    for line, col, length, ttype, mods in tokens:
        delta_line = line - prev_line
        delta_col = (col - prev_col) if delta_line == 0 else col
        data.extend([delta_line, delta_col, length, ttype, mods])
        prev_line, prev_col = line, col
    return data


def _semantic_token_tuples(uri: str) -> List[tuple]:
    """Compute sorted ``(line, col, length, type, modifiers)`` tuples for *uri*.

    Classifies each identifier as endogenous, exogenous, parameter, or
    model-local, and tags states/jumpers and declaration sites via modifiers.
    Shared by the full and range semantic-token requests.
    """
    with _state_lock:
        model = _document_models.get(uri)
    if model is None:
        return []

    doc = server.workspace.get_text_document(uri)
    source = doc.source

    feature_model = model
    open_include_kind = None
    try:
        include_models = list(_workspace_index.resolve_all_includes(uri).values())
        effective_model = _workspace_index.get_effective_model(uri)
        if effective_model is not None:
            feature_model = effective_model
        if include_models:
            feature_model = model_with_include_context(
                feature_model,
                include_models,
                include_model_equations=True,
            )
        open_include_context = _open_include_feature_context(
            uri, model, source_text=source
        )
        if open_include_context is not None:
            open_include_kind, feature_model, _declaration_models = open_include_context
    except Exception:
        logger.exception("semantic tokens: include resolution failed for %s", uri)

    endo_names = {d.name for d in feature_model.endogenous}
    exo_names = {d.name for d in feature_model.exogenous}
    param_names = {d.name for d in feature_model.parameters}
    local_names = set(_model_local_variables(feature_model))
    local_scope = (
        _map_range_to_original_source(feature_model, feature_model.model_block_range)
        if feature_model.model_block_range is not None
        else None
    )
    if local_scope is None and open_include_kind == "model":
        lines = source.split("\n")
        end_line = max(len(lines) - 1, 0)
        end_char = len(lines[end_line]) if lines else 0
        local_scope = DRange(DPos(0, 0), DPos(end_line, end_char))

    if not (endo_names or exo_names or param_names or local_names):
        return []

    # Mask comments AND string literals so identifier matches inside
    # them aren't classified.  Preserve the expression part of MCP tags,
    # which Dynare treats as model code even though it is quoted.
    # _mask_strings_preserving_mcp_values applies _STRING_LITERAL_RE.
    def _blank_match(m: re.Match) -> str:
        return re.sub(r"\S", " ", m.group(0))

    # Mask strings first so embedded ``//`` / ``%`` inside an equation
    # tag don't trigger comment-line blanking that wipes the rest of
    # the line.
    masked = _mask_inactive_macro_branches(source)
    masked = _mask_strings_preserving_mcp_values(masked)
    masked = _BLOCK_COMMENT_RE.sub(_blank_match, masked)
    masked = _MACRO_DIRECTIVE_LINE_RE.sub(_blank_match, masked)
    masked = _LINE_COMMENT_RE.sub(_blank_match, masked)

    type_endo = 0
    type_exo = 1
    type_param = 2
    type_local = 3

    # Timing (state/jumper) for the forward/predetermined modifiers, and the
    # source positions of declarations for the declaration modifier.
    from .model_info import classify_variable_timing

    timing = classify_variable_timing(feature_model)
    decl_starts: set = set()
    for decl in (
        *feature_model.endogenous,
        *feature_model.exogenous,
        *feature_model.parameters,
    ):
        try:
            mapped = _map_range_to_original_source(feature_model, decl.range)
        except Exception:
            mapped = None
        if mapped is not None:
            decl_starts.add((mapped.start.line, mapped.start.character))

    tokens: List[tuple] = []
    source_lines = source.split("\n")
    for line_no, line_text in enumerate(masked.split("\n")):
        source_line = (
            source_lines[line_no] if line_no < len(source_lines) else line_text
        )
        for m in _MACRO_SOURCE_IDENTIFIER_RE.finditer(line_text):
            if m.start() >= 2 and source_line[m.start() - 2 : m.start()] == "@{":
                continue
            if _is_on_the_fly_kind_marker(source_line, m.start(), m.end()):
                continue
            name = m.group(0)
            if "@{" in name:
                lsp_char = _source_index_to_lsp_character(source_line, m.start())
                model_pos = _source_position_to_model_position(
                    feature_model,
                    source,
                    lsp.Position(line=line_no, character=lsp_char),
                )
                model_match = _word_at_position(
                    feature_model.text.split("\n"), model_pos
                )
                if model_match is None:
                    continue
                name = model_match.group(1)
            if name in local_names:
                if local_scope is None or not _range_contains_source_position(
                    local_scope,
                    DPos(line_no, m.start()),
                ):
                    continue
                ttype = type_local
            elif name in endo_names:
                ttype = type_endo
            elif name in exo_names:
                ttype = type_exo
            elif name in param_names:
                ttype = type_param
            else:
                continue

            mods = 0
            if (line_no, m.start()) in decl_starts:
                mods |= _MOD_DECLARATION
            if ttype == type_endo:
                cls = timing.get(name, {}).get("class")
                if cls in ("forward_looking", "mixed"):
                    mods |= _MOD_FORWARD
                if cls in ("predetermined", "mixed"):
                    mods |= _MOD_PREDETERMINED

            start_col = _source_index_to_lsp_character(source_line, m.start())
            end_col = _source_index_to_lsp_character(source_line, m.end())
            tokens.append((line_no, start_col, end_col - start_col, ttype, mods))

    tokens.sort(key=lambda t: (t[0], t[1]))
    return tokens


@server.feature(
    lsp.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL,
    lsp.SemanticTokensLegend(
        token_types=_SEMANTIC_TOKEN_TYPES,
        token_modifiers=_SEMANTIC_TOKEN_MODIFIERS,
    ),
)
def semantic_tokens_full(
    params: lsp.SemanticTokensParams,
) -> lsp.SemanticTokens:
    """Whole-document semantic tokens (identifier semantic class + timing)."""
    tokens = _semantic_token_tuples(params.text_document.uri)
    return lsp.SemanticTokens(data=_encode_semantic_tokens(tokens))


@server.feature(lsp.TEXT_DOCUMENT_SEMANTIC_TOKENS_RANGE)
def semantic_tokens_range(
    params: lsp.SemanticTokensRangeParams,
) -> lsp.SemanticTokens:
    """Semantic tokens for a single range — avoids recolouring the whole file."""
    tokens = _semantic_token_tuples(params.text_document.uri)
    start = (params.range.start.line, params.range.start.character)
    end = (params.range.end.line, params.range.end.character)
    # LSP ranges are half-open: include a token iff its start is in [start, end).
    in_range = [t for t in tokens if start <= (t[0], t[1]) < end]
    return lsp.SemanticTokens(data=_encode_semantic_tokens(in_range))


# ---------------------------------------------------------------------------
# Explain Diagnostic — server-side command and CLI-shared helper
# ---------------------------------------------------------------------------


@server.command("dynare/explainDiagnostic")
def explain_diagnostic_command(args):
    """Return a markdown explanation for a diagnostic code.

    Called via ``workspace/executeCommand`` with ``{"code": "E010"}``.
    Returns the markdown body, or a fallback message if unknown.
    """
    if isinstance(args, list) and args:
        args = args[0]
    if not isinstance(args, dict):
        return None
    code = args.get("code")
    if not code:
        return None
    rendered = _explain_module.render_markdown(code)
    if rendered is None:
        return (
            f"Diagnostic `{code}` is not yet documented. "
            f"Known codes: {', '.join(_explain_module.known_codes())}"
        )
    return rendered


# ---------------------------------------------------------------------------
# Compare Models — server-side command sharing the model_diff backend
# ---------------------------------------------------------------------------


@server.command("dynare/compareModels")
def compare_models_command(*args) -> dict:
    """Compare two open .mod documents and return a structural diff.

    Args (positional or one-element list-wrapped dict):
        uri_a: str  - URI of the "before" document
        uri_b: str  - URI of the "after" document

    Returns a dict from ModelDiff.to_dict(), or
    {"error": "...", "code": "..."} if either URI is not in the index.
    """
    from .model_diff import compare_models as _compare_models

    # Accept both positional ``[uri_a, uri_b]`` and ``[{"uri_a": ..., "uri_b": ...}]``
    # invocation styles, mirroring ``dynare/explainDiagnostic``.
    uri_a: Optional[str] = None
    uri_b: Optional[str] = None
    values: tuple[object, ...]
    if len(args) == 1 and isinstance(args[0], list):
        values = tuple(args[0])
    else:
        values = args
    if len(values) >= 2 and isinstance(values[0], str) and isinstance(values[1], str):
        uri_a, uri_b = values[0], values[1]
    elif len(values) == 1 and isinstance(values[0], dict):
        uri_a = values[0].get("uri_a") or values[0].get("uriA")
        uri_b = values[0].get("uri_b") or values[0].get("uriB")

    if not uri_a or not uri_b:
        return {
            "error": "compareModels requires uri_a and uri_b arguments",
            "code": "BAD_ARGS",
        }

    model_a = _model_for_uri(uri_a)
    model_b = _model_for_uri(uri_b)
    if model_a is None:
        return {
            "error": f"No parsed model for uri_a: {uri_a}",
            "code": "URI_A_NOT_FOUND",
        }
    if model_b is None:
        return {
            "error": f"No parsed model for uri_b: {uri_b}",
            "code": "URI_B_NOT_FOUND",
        }

    model_a = model_with_include_context(
        model_a,
        list(_workspace_index.resolve_all_includes(uri_a).values()),
    )
    model_b = model_with_include_context(
        model_b,
        list(_workspace_index.resolve_all_includes(uri_b).values()),
    )

    return _compare_models(model_a, model_b).to_dict()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def start_server(host: str = "127.0.0.1", port: int = 0, stdio: bool = True) -> None:
    """Start the language server."""
    if stdio:
        server.start_io()
    else:
        server.start_tcp(host, port)
