"""Exact, in-memory unified-diff validation with separate repair evidence."""
from __future__ import annotations
import hashlib
import os
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from .analysis_service import analyze_snapshot, effective_config, fingerprint, limit_findings, bound_report, snapshot

_HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$")


def _lines(text: str) -> list[str]:
    return [part for part in re.split(r"(?<=\n)|(?<=\r)(?!\n)", text) if part]


def _body(line: str) -> str:
    return line.removesuffix("\n").removesuffix("\r")


def _path(header: str, prefix: str) -> str:
    name = header[4:].split("\t", 1)[0].replace("\\", "/")
    if name.startswith(prefix):
        name = name[len(prefix):]
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or ":" in name or "\x00" in name or str(path) == ".":
        raise ValueError("Patch filenames must be relative paths within the entry directory")
    return str(path)


def apply_unified_diff(files: dict[str, str], patch: str,
                       key: Callable[[str], str] | None = None) -> dict[str, str]:
    """Apply existing-file modifications exactly: no fuzzy offsets, writes, or renames.

    ``key`` maps a patch header path to a ``files`` key (e.g. case folding on
    case-insensitive filesystems); by default header paths must match exactly.
    """
    lines = _lines(patch)
    out_files = dict(files)
    seen: set[str] = set()
    i = 0
    while i < len(lines):
        line = _body(lines[i])
        if line.startswith(("diff --git ", "index ")) or (not line and seen):
            i += 1
            continue
        if not line.startswith("--- ") or i + 1 >= len(lines) or not _body(lines[i + 1]).startswith("+++ "):
            raise ValueError("Expected a standard unified-diff file header")
        old = _path(line, "a/")
        new = _path(_body(lines[i + 1]), "b/")
        if key is not None:
            old, new = key(old), key(new)
        if old != new or old not in files or old in seen:
            raise ValueError("Only one modification section per existing file is supported; no create/delete/rename")
        seen.add(old)
        source = _lines(files[old])
        eol = next((s[len(_body(s)):] for s in source if s != _body(s)), "\n")
        output: list[str] = []
        cursor = 0
        hunks = 0
        i += 2
        while i < len(lines) and _body(lines[i]).startswith("@@ "):
            match = _HUNK.fullmatch(_body(lines[i]))
            if not match:
                raise ValueError("Malformed hunk header")
            old_start, old_count, new_start, new_count = [int(x) for x in (
                match.group(1), match.group(2) or "1", match.group(3), match.group(4) or "1")]
            old_pos = old_start if old_count == 0 else old_start - 1
            new_pos = new_start if new_count == 0 else new_start - 1
            if old_pos < cursor or old_pos > len(source) or (old_count == new_count == 0):
                raise ValueError("Overlapping, out-of-range, or empty hunk")
            output.extend(source[cursor:old_pos])
            if new_pos != len(output):
                raise ValueError("New-file hunk coordinates are inconsistent")
            cursor = old_pos
            old_seen = new_seen = 0
            i += 1
            while old_seen < old_count or new_seen < new_count:
                if i >= len(lines) or (lines[i][:1] not in {" ", "-", "+"} and _body(lines[i])):
                    raise ValueError("Truncated hunk or invalid record")
                # A bare empty line is an empty context line whose leading
                # space was stripped (accepted by git apply and GNU patch).
                kind, content = (lines[i][0], _body(lines[i][1:])) if _body(lines[i]) else (" ", "")
                i += 1
                newline = True
                if i < len(lines) and _body(lines[i]) == "\\ No newline at end of file":
                    newline = False
                    i += 1
                if kind in {" ", "-"}:
                    if cursor >= len(source) or _body(source[cursor]) != content or (source[cursor] != _body(source[cursor])) != newline:
                        raise ValueError("Patch context does not match the exact source")
                    if kind == " ":
                        output.append(source[cursor])
                    cursor += 1
                    old_seen += 1
                if kind == "+":
                    output.append(content + (eol if newline else ""))
                if kind in {" ", "+"}:
                    new_seen += 1
                if old_seen > old_count or new_seen > new_count:
                    raise ValueError("Hunk record counts do not match its header")
            hunks += 1
        if not hunks:
            raise ValueError("File section has no hunks")
        output.extend(source[cursor:])
        if any(s == _body(s) for s in output[:-1]):
            raise ValueError("A no-newline record must be the final source line")
        out_files[old] = "".join(output)
    if not seen:
        raise ValueError("Patch is empty")
    return out_files


def diagnostic_delta(before: dict[str, Any], after: dict[str, Any], *, bounded: bool = False) -> dict[str, Any]:
    """Multiset comparison of diagnostics; ``bounded`` caps each list with counts."""
    def rows(report):
        return report["stages"]["diagnostics"].get("findings", [])
    def key(row):
        return (row.get("code"), row.get("severity"), row.get("message"))
    first, second = rows(before), rows(after)
    a, b = Counter(map(key, first)), Counter(map(key, second))
    def selected(records, counts):
        output = []
        for row in records:
            value = key(row)
            if counts[value] > 0:
                output.append(row)
                counts[value] -= 1
        return output
    delta: dict[str, Any] = {"resolved": selected(first, a - b), "introduced": selected(second, b - a),
                             "remaining": selected(second, a & b),
                             "matching": "code/severity/message multiset; not causal attribution"}
    if bounded:
        summary = {}
        for name in ("resolved", "introduced", "remaining"):
            delta[name], summary[name] = limit_findings(delta[name])
        delta["summary"] = summary
    return delta


# Rewritten buffers are returned inline only when small (or on request).
MAX_INLINE_FILE_BYTES = 32_000
# Structural-review lists longer than this are cut; truncated_lists has totals.
MAX_STRUCTURAL_ITEMS = 50


def _matching_config(before: Any, expected_snapshot_id: str) -> dict[str, Any] | None:
    """Find the report/preflight toggles that reproduce the expected snapshot ID."""
    for numerical in (False, True):
        for preprocessor in (False, True):
            candidate = effective_config({**before.config, "numerical": numerical, "preprocessor": preprocessor})
            if fingerprint(before.files, candidate, before.entry_file) == expected_snapshot_id:
                return candidate
    return None


def dynare_validate_patch(file_content: str, patch: str, expected_snapshot_id: str,
                          active_file: str = "model.mod", files: dict[str, str] | None = None,
                          config: dict[str, Any] | None = None,
                          include_file_text: bool | None = None) -> dict[str, Any]:
    """Preview an exact patch against a report snapshot; never write model files.

    Patch paths are relative to the entry model's directory and are matched
    case-insensitively where the filesystem is. The expected ID must come
    from dynare_analysis_report or dynare_preflight with the same source map.
    When config is omitted, the report/preflight check toggles that produced
    the ID are reused; an explicit config that differs returns
    error="config_mismatch" with expected_config. Returns applicability,
    before/after reports and a diagnostic delta with findings capped per
    severity (see findings_summary / summary), structural changes, and
    changed_files keyed by the caller's filenames. Rewritten text is inlined
    when include_file_text is true, or by default when all changed files
    total at most 32000 UTF-8 bytes; changed_file_summaries always gives
    sizes and SHA-256 hashes. structural_review lists are cut at 50 items
    (truncated_lists gives totals). It never certifies economic fidelity.
    No model execution.
    """
    before = snapshot(file_content, active_file, files, config)
    base = {"schema_version": "dynare-patch/1", "snapshot_id": before.snapshot_id,
            "source_files_written": False, "economic_fidelity": "not_assessed"}
    if not isinstance(expected_snapshot_id, str):
        return {**base, "success": False, "applies": False, "error": "stale_snapshot"}
    if before.snapshot_id != expected_snapshot_id:
        matched = _matching_config(before, expected_snapshot_id)
        if matched is None:
            return {**base, "success": False, "applies": False, "error": "stale_snapshot"}
        if config is not None:
            return {**base, "success": False, "applies": False, "error": "config_mismatch",
                    "expected_config": matched, "supplied_config": before.config,
                    "message": "Sources match the snapshot but the analysis configuration differs."}
        before.config = matched
        base["snapshot_id"] = before.snapshot_id
    root = Path(before.entry_file).parent
    by_relative = {}
    for name in before.files:
        try:
            by_relative[os.path.normcase(Path(name).relative_to(root).as_posix())] = name
        except ValueError:
            continue
    try:
        updated = apply_unified_diff({rel: before.files[name] for rel, name in by_relative.items()}, patch,
                                     key=os.path.normcase)
    except ValueError as exc:
        return {**base, "success": False, "applies": False, "error": "invalid_patch", "message": str(exc)}
    changes = {by_relative[rel]: text for rel, text in updated.items() if text != before.files[by_relative[rel]]}
    all_files = {**before.files, **changes}
    after = snapshot(all_files[before.entry_file], before.entry_file, all_files, before.config)
    prior_report, next_report = analyze_snapshot(before), analyze_snapshot(after)
    delta = diagnostic_delta(prior_report, next_report, bounded=True)
    try:
        from .model_diff import compare_models
        changes_dict = compare_models(before.model, after.model).to_dict()
        # Unchanged-symbol inventories dominate large models; keep lists bounded.
        truncated = {}
        for name, value in changes_dict.items():
            if isinstance(value, list) and len(value) > MAX_STRUCTURAL_ITEMS:
                truncated[name] = len(value)
                changes_dict[name] = value[:MAX_STRUCTURAL_ITEMS]
        comparison = {"status": "available", "changes": changes_dict, "truncated_lists": truncated}
    except Exception as exc:
        comparison = {"status": "unavailable", "message": str(exc)}
    summaries = {before.display_name(name): {
        "bytes_before": len(before.files[name].encode("utf-8", "surrogatepass")),
        "bytes_after": len(text.encode("utf-8", "surrogatepass")),
        "sha256_after": hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()}
        for name, text in changes.items()}
    inline = include_file_text
    if inline is None:
        inline = sum(item["bytes_after"] for item in summaries.values()) <= MAX_INLINE_FILE_BYTES
    return {**base, "success": True, "applies": True, "checks_passed": next_report["checks_passed"],
            "after_snapshot_id": after.snapshot_id,
            "changed_files": {before.display_name(name): text for name, text in changes.items()} if inline else {},
            "changed_files_omitted": not inline, "changed_file_summaries": summaries,
            "before": bound_report(prior_report), "after": bound_report(next_report),
            "diagnostic_delta": delta, "structural_review": comparison,
            "review_required": True, "disclaimer": "Fewer diagnostics do not establish that the intended economic repair was made."}


TOOL = dynare_validate_patch
COMMAND = "dynare/validatePatch"
CLI = "patch"
EXAMPLE = {"patch": "", "expected_snapshot_id": "copy ID from a report made with the same config"}
