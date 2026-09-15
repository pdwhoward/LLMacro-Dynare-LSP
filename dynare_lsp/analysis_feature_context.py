"""Bounded source retrieval using equation-symbol incidence, not causal edges."""
from __future__ import annotations
import hashlib
import re
from typing import Any
from .analysis_service import snapshot

_TOKEN = re.compile(r"(?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?)|(?P<name>[A-Za-z_][A-Za-z0-9_]*)")
_TIMING = re.compile(r"\s*\(\s*([+-]?\s*\d+)\s*\)")


def mask_noncode(text: str) -> str:
    """Offset-preserving masking for comments, strings, and leading equation tags."""
    chars = list(text)
    i = 0
    while i < len(text):
        start = i
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = len(text) if end < 0 else end + 2
        elif text.startswith("//", i) or text[i] == "%":
            match = re.search(r"[\r\n]", text[i:])
            i = len(text) if match is None else i + match.start()
        elif text[i] in "\"'":
            quote = text[i]
            i += 1
            while i < len(text):
                if text[i] == quote:
                    i += 1
                    if i < len(text) and text[i] == quote:
                        i += 1
                        continue
                    break
                i += 1
        else:
            i += 1
            continue
        for j in range(start, i):
            if chars[j] not in "\r\n":
                chars[j] = " "
    # Only leading [] groups are equation tags; interior brackets are not erased.
    scan = "".join(chars)
    cursor = len(scan) - len(scan.lstrip())
    while cursor < len(scan) and scan[cursor] == "[":
        end = scan.find("]", cursor + 1)
        if end < 0:
            break
        for j in range(cursor, end + 1):
            if chars[j] not in "\r\n":
                chars[j] = " "
        cursor = end + 1
        while cursor < len(scan) and scan[cursor].isspace():
            cursor += 1
    return "".join(chars)


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
        offset = int(match.group(1).replace(" ", "")) if match else 0
        occurrences.append({"symbol": name, "offset": offset, "start": token.start(), "end": token.end(),
                            "side": "residual" if equals is None else "left" if token.start() < equals.start() else "right"})
    definition = local.group(1) if local else None
    return {"kind": "local_definition" if local else "equilibrium_equation", "definition": definition,
            "symbols": sorted({o["symbol"] for o in occurrences}), "occurrences": occurrences,
            "local_dependencies": sorted({o["symbol"] for o in occurrences if o["symbol"] != definition}) if local else []}


def _offset(text: str, pos) -> int:
    lines = text.split("\n")
    if pos.line < 0 or pos.line >= len(lines):
        return len(text)
    return sum(len(s) + 1 for s in lines[:pos.line]) + min(max(pos.character, 0), len(lines[pos.line]))


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


def _block(file: str, model, item, source: str, reason: str) -> dict[str, Any]:
    span = source_span(model, item, source)
    if span is None:
        return {"file": file, "start": 0, "end": len(source), "text": source,
                "reason": reason, "origin": "whole_source_fallback"}
    scan = mask_noncode(source)
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
    apply only to #local definitions. max_bytes bounds snippet text, not JSON
    metadata or model-specific tokens. Omitted blocks and source origins are
    explicit; nothing is truncated mid-equation. No solver or model execution.
    """
    if sum(x is not None for x in (symbol, equation_id, line)) != 1:
        raise ValueError("Select exactly one of symbol, equation_id, or line")
    if line is not None and (isinstance(line, bool) or not isinstance(line, int) or line < 1):
        raise ValueError("line must be a positive 1-based integer")
    snap = snapshot(file_content, active_file, files, config)
    models = [(snap.entry_file, snap.root_model)] + list(snap.include_models.items())
    declared = set(snap.model.all_declared_names())
    for _, model in models:
        for eq in model.model_equations:
            local = re.match(r"\s*#\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", eq.text)
            if local:
                declared.add(local.group(1))
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
                block = _block(physical, model, item, source, attr)
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
    blocks = [_block(*objects[n["id"]], "selected equation or local definition") for n in selected]
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
                    blocks.append(_block(physical, model, item, source, attr))
    kept, omitted = select_blocks(blocks, max_bytes)
    for block in kept:
        source = snap.files[block["file"]]
        block["start_line"] = source.count("\n", 0, block["start"]) + 1
        block["end_line"] = source.count("\n", 0, block["end"]) + 1
    return {"schema_version": "dynare-context/1", "snapshot": snap.manifest(), "success": bool(selected),
            "error": None if selected else "no_matching_equation", "snippets": kept, "omitted": omitted,
            "incidence": nodes, "selected_equations": [n["id"] for n in selected],
            "budget": {"unit": "UTF-8 bytes of snippet text", "limit": max_bytes, "used": sum(len(b["text"].encode("utf-8", "surrogatepass")) for b in kept)},
            "tokenizer": None, "complete_context": bool(selected) and not omitted and not snap.unresolved and not snap.cycles,
            "disclaimer": "Syntactic incidence is not causality. Raw timing offsets are not a transformed Dynare state-space classification."}


TOOL = dynare_get_context
COMMAND = "dynare/getContext"
CLI = "context"
EXAMPLE = {"symbol": "y", "max_bytes": 12000}
