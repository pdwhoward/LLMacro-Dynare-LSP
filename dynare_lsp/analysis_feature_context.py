"""Bounded source retrieval using equation-symbol incidence, not causal edges."""
from __future__ import annotations
import functools
import hashlib
import json
import re
from typing import Any
from .analysis_service import snapshot

_TOKEN = re.compile(r"(?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?)|(?P<name>[A-Za-z_][A-Za-z0-9_]*)")
_TIMING = re.compile(r"\s*\(\s*([+-]?\s*\d+)\s*\)")


MAX_OMITTED_RECORDS = 100
_NONCODE_START = re.compile(r"/\*|//|%|[\"']")
_EOL = re.compile(r"[\r\n]")
_NOT_EOL = re.compile(r"[^\r\n]")


def mask_noncode(text: str) -> str:
    """Offset-preserving masking for comments, strings, and leading equation tags."""
    pieces: list[str] = []
    cursor = i = 0
    while True:
        # Jump straight to the next comment or string opener (linear time).
        match = _NONCODE_START.search(text, i)
        if match is None:
            break
        start, token = match.start(), match.group()
        if token == "/*":
            end = text.find("*/", start + 2)
            i = len(text) if end < 0 else end + 2
        elif token in ("//", "%"):
            eol = _EOL.search(text, start)
            i = len(text) if eol is None else eol.start()
        else:
            i = start + 1
            while i < len(text):
                if text[i] == token:
                    i += 1
                    if i < len(text) and text[i] == token:
                        i += 1
                        continue
                    break
                i += 1
        pieces.append(text[cursor:start])
        pieces.append(_NOT_EOL.sub(" ", text[start:i]))
        cursor = i
    pieces.append(text[cursor:])
    scan = "".join(pieces)
    # Only leading [] groups are equation tags; interior brackets are not erased.
    chars: list[str] | None = None
    cursor = len(scan) - len(scan.lstrip())
    while cursor < len(scan) and scan[cursor] == "[":
        end = scan.find("]", cursor + 1)
        if end < 0:
            break
        chars = list(scan) if chars is None else chars
        for j in range(cursor, end + 1):
            if chars[j] not in "\r\n":
                chars[j] = " "
        cursor = end + 1
        while cursor < len(scan) and scan[cursor].isspace():
            cursor += 1
    return scan if chars is None else "".join(chars)


def incidence(text: str, declared: set[str]) -> dict[str, Any]:
    scan = mask_noncode(text)
    local = re.match(r"\s*#\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", scan)
    equals = re.search(r"(?<![<>=!])=(?!=)", scan)
    occurrences = []
    for token in _TOKEN.finditer(scan):
        name = token.group("name")
        if name is None or name not in declared:
            continue
        match = _TIMING.match(scan, token.end())
        offset = int(re.sub(r"\s+", "", match.group(1))) if match else 0
        occurrences.append({"symbol": name, "offset": offset, "start": token.start(), "end": token.end(),
                            "side": "residual" if equals is None else "left" if token.start() < equals.start() else "right"})
    definition = local.group(1) if local else None
    return {"kind": "local_definition" if local else "equilibrium_equation", "definition": definition,
            "symbols": sorted({o["symbol"] for o in occurrences}), "occurrences": occurrences,
            "local_dependencies": sorted({o["symbol"] for o in occurrences if o["symbol"] != definition}) if local else []}


@functools.lru_cache(maxsize=16)
def _line_starts(text: str) -> tuple[int, ...]:
    return (0, *(m.end() for m in re.finditer("\n", text)))


def _offset(text: str, pos) -> int:
    starts = _line_starts(text)
    if pos.line < 0 or pos.line >= len(starts):
        return len(text)
    line_end = starts[pos.line + 1] - 1 if pos.line + 1 < len(starts) else len(text)
    return starts[pos.line] + min(max(pos.character, 0), line_end - starts[pos.line])


def source_span(model, item, source: str) -> tuple[int, int] | None:
    """Return a verified original-source span, never a guessed expanded location."""
    original = getattr(model, "original_text", "") or model.text
    if original != source or getattr(item, "range", None) is None:
        return None
    start, end = _offset(model.text, item.range.start), _offset(model.text, item.range.end)
    mapping = getattr(model, "source_map", None)
    if mapping:
        start = mapping[min(start, len(mapping) - 1)]
        end = mapping[min(end, len(mapping) - 1)]
    if 0 <= start <= end <= len(source):
        return start, end
    return None


def _block(file: str, model, item, source: str, reason: str,
           masks: dict[str, str] | None = None) -> dict[str, Any]:
    span = source_span(model, item, source)
    if span is None:
        return {"file": file, "start": 0, "end": len(source), "text": source,
                "reason": reason, "origin": "whole_source_fallback"}
    # Mask each physical source once per request, not once per block.
    scan = None if masks is None else masks.get(file)
    if scan is None or len(scan) != len(source):
        scan = mask_noncode(source)
        if masks is not None:
            masks[file] = scan
    start, end = span
    start = scan.rfind(";", 0, start) + 1
    stop = scan.find(";", max(start, end - 1))
    end = len(source) if stop < 0 else stop + 1
    while start < end and source[start].isspace():
        start += 1
    return {"file": file, "start": start, "end": end, "text": source[start:end],
            "reason": reason, "origin": "original_source"}


def select_blocks(blocks: list[dict[str, Any]], max_bytes: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep complete source blocks only. Budget applies to their UTF-8 text."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 1 <= max_bytes <= 1_000_000:
        raise ValueError("max_bytes must be an integer in [1, 1000000]")
    kept, omitted, seen = [], [], set()
    used = 0
    for block in blocks:
        key = (block["file"], block["start"], block["end"])
        if key in seen:
            continue
        seen.add(key)
        size = len(block["text"].encode("utf-8", "surrogatepass"))
        if used + size <= max_bytes:
            kept.append(block)
            used += size
        else:
            omitted.append({k: v for k, v in block.items() if k != "text"} | {"bytes": size, "omission_reason": "budget"})
    return kept, omitted


def dynare_get_context(file_content: str, active_file: str = "model.mod",
                       files: dict[str, str] | None = None, symbol: str | None = None,
                       equation_id: str | None = None, line: int | None = None,
                       max_bytes: int = 12000, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Retrieve complete relevant equations and support statements from a snapshot.

    Select exactly one symbol, equation_id, or 1-based entry-file line. The
    incidence catalog records BOTH sides and timing offsets; equilibrium
    equations are not assignments or causal providers. Directed dependencies
    apply only to #local definitions. max_bytes bounds snippet text and,
    separately, the serialized incidence rows, which cover only the selected
    equations (incidence_total counts the whole model); at most 100 omitted
    block records are listed (omitted_total gives the count). Omitted blocks
    and source origins are explicit; nothing is truncated mid-equation. No
    solver or model execution.
    """
    if sum(x is not None for x in (symbol, equation_id, line)) != 1:
        raise ValueError("Select exactly one of symbol, equation_id, or line")
    if line is not None and (isinstance(line, bool) or not isinstance(line, int) or line < 1):
        raise ValueError("line must be a positive 1-based integer")
    snap = snapshot(file_content, active_file, files, config, preserve_include_ranges=True)
    models = [(snap.entry_file, snap.root_model)] + list(snap.include_models.items())
    declared = set(snap.model.all_declared_names())
    for _, model in models:
        # Helpers are not declared Dynare symbols, but their names must survive
        # incidence filtering so calibration dependencies can reach them.
        declared.update(item.name for item in model.helper_assignments)
        for eq in model.model_equations:
            local = re.match(r"\s*#\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", eq.text)
            if local:
                declared.add(local.group(1))
    masks: dict[str, str] = {}
    nodes: list[dict[str, Any]] = []
    objects: dict[str, tuple[str, Any, Any, str]] = {}
    for file, model in models:
        physical = file if file in snap.files else file.rsplit("#", 1)[0]
        source = snap.files.get(physical)
        if source is None:
            continue
        for ordinal, eq in enumerate(model.model_equations, 1):
            identifier = hashlib.sha256(file.encode("utf-8", "surrogatepass")).hexdigest()[:12] + f":{ordinal}"
            span = source_span(model, eq, source)
            node = {"id": identifier, "name": getattr(eq, "name", None), "file": physical,
                    "source_line": source.count("\n", 0, span[0]) + 1 if span else None,
                    "source_end_line": source.count("\n", 0, max(span[0], span[1] - 1)) + 1 if span else None,
                    **incidence(eq.text, declared)}
            nodes.append(node)
            objects[identifier] = (physical, model, eq, source)
    selected = [n for n in nodes if (symbol is not None and symbol in n["symbols"]) or
                (equation_id is not None and equation_id == n["id"]) or
                (line is not None and n["file"] == snap.entry_file and n["source_line"] is not None and n["source_line"] <= line <= n["source_end_line"])]
    needed = {s for n in selected for s in n["symbols"]}
    chosen = {n["id"] for n in selected}
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if node["definition"] in needed and node["id"] not in chosen:
                selected.append(node)
                chosen.add(node["id"])
                needed.update(node["symbols"])
                changed = True
    # Follow calibration/helper references as well as local equation definitions.
    # Use the actual original source slice, since assignment records need not
    # expose a common .text attribute.
    support = []
    for file, model in models:
        physical = file if file in snap.files else file.rsplit("#", 1)[0]
        source = snap.files.get(physical)
        if source is None:
            continue
        for attr in ("param_assignments", "helper_assignments", "steady_state_equations"):
            for item in getattr(model, attr, []) or []:
                block = _block(physical, model, item, source, attr, masks)
                row = incidence(block["text"], declared)
                targets = {getattr(item, "name", "")} | {o["symbol"] for o in row["occurrences"] if o["side"] == "left"}
                support.append((targets, set(row["symbols"]), block))
    changed = True
    while changed:
        before_needed = set(needed)
        for targets, dependencies, _block_value in support:
            if targets & needed:
                needed.update(dependencies)
        changed = needed != before_needed
    blocks = [_block(*objects[n["id"]], "selected equation or local definition", masks) for n in selected]
    blocks.extend(block for targets, _, block in support if targets & needed)
    for file, model in models:
        physical = file if file in snap.files else file.rsplit("#", 1)[0]
        source = snap.files.get(physical)
        if source is None:
            continue
        for attr in ("endogenous", "exogenous", "parameters", "param_assignments", "helper_assignments", "initval_entries", "steady_state_equations", "macro_directives", "includes"):
            for item in getattr(model, attr, []) or []:
                names = {getattr(item, "name", "")} | set(incidence(getattr(item, "text", ""), declared)["symbols"])
                if names & needed or attr in {"macro_directives", "includes"}:
                    blocks.append(_block(physical, model, item, source, attr, masks))
    kept, omitted = select_blocks(blocks, max_bytes)
    for block in kept:
        source = snap.files[block["file"]]
        block["start_line"] = source.count("\n", 0, block["start"]) + 1
        block["end_line"] = source.count("\n", 0, block["end"]) + 1
    # Incidence metadata covers only the selected equations and has its own
    # max_bytes budget (serialized JSON), so the response stays bounded.
    catalog, catalog_bytes = [], 0
    for node in selected:
        size = len(json.dumps(node, ensure_ascii=False).encode("utf-8", "surrogatepass"))
        if catalog_bytes + size > max_bytes:
            break
        catalog.append(node)
        catalog_bytes += size
    omitted_total = len(omitted)
    omitted = omitted[:MAX_OMITTED_RECORDS]
    return {"schema_version": "dynare-context/1", "snapshot": snap.manifest(), "success": bool(selected),
            "error": None if selected else "no_matching_equation", "snippets": kept, "omitted": omitted,
            "omitted_total": omitted_total, "omitted_truncated": omitted_total > len(omitted),
            "incidence": catalog, "incidence_total": len(nodes), "incidence_scope": "selected_equations",
            "incidence_truncated": len(catalog) < len(selected),
            "selected_equations": [n["id"] for n in selected],
            "budget": {"unit": "UTF-8 bytes of snippet text", "limit": max_bytes, "used": sum(len(b["text"].encode("utf-8", "surrogatepass")) for b in kept),
                       "incidence_unit": "UTF-8 bytes of serialized incidence JSON", "incidence_used": catalog_bytes},
            "tokenizer": None, "complete_context": bool(selected) and not omitted and not snap.unresolved and not snap.cycles,
            "disclaimer": "Syntactic incidence is not causality. Raw timing offsets are not a transformed Dynare state-space classification."}


TOOL = dynare_get_context
COMMAND = "dynare/getContext"
CLI = "context"
EXAMPLE = {"symbol": "y", "max_bytes": 12000}
