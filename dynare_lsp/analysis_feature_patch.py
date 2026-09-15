"""Exact, in-memory unified-diff validation with separate repair evidence."""
from __future__ import annotations
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any
from .analysis_service import analyze_snapshot, snapshot

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


def apply_unified_diff(files: dict[str, str], patch: str) -> dict[str, str]:
    """Apply existing-file modifications exactly: no fuzzy offsets, writes, or renames."""
    lines = _lines(patch)
    out_files = dict(files)
    seen: set[str] = set()
    i = 0
    while i < len(lines):
        line = _body(lines[i])
        if line.startswith(("diff --git ", "index ")):
            i += 1
            continue
        if not line.startswith("--- ") or i + 1 >= len(lines) or not _body(lines[i + 1]).startswith("+++ "):
            raise ValueError("Expected a standard unified-diff file header")
        old = _path(line, "a/")
        new = _path(_body(lines[i + 1]), "b/")
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
                if i >= len(lines) or lines[i][:1] not in {" ", "-", "+"}:
                    raise ValueError("Truncated hunk or invalid record")
                kind, content = lines[i][0], _body(lines[i][1:])
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


def diagnostic_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
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
    return {"resolved": selected(first, a - b), "introduced": selected(second, b - a),
            "remaining": selected(second, a & b), "matching": "code/severity/message multiset; not causal attribution"}


def dynare_validate_patch(file_content: str, patch: str, expected_snapshot_id: str,
                          active_file: str = "model.mod", files: dict[str, str] | None = None,
                          config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Preview an exact patch against a report snapshot; never write model files.

    Patch paths are relative to the entry model's directory. The expected ID
    must come from dynare_analysis_report with the same source map and config.
    Returns applicability, before/after diagnostics, structural changes and
    rewritten buffers. It never certifies economic fidelity. No model execution.
    """
    before = snapshot(file_content, active_file, files, config)
    base = {"schema_version": "dynare-patch/1", "snapshot_id": before.snapshot_id,
            "source_files_written": False, "economic_fidelity": "not_assessed"}
    if not isinstance(expected_snapshot_id, str) or before.snapshot_id != expected_snapshot_id:
        return {**base, "success": False, "applies": False, "error": "stale_snapshot"}
    root = Path(before.entry_file).parent
    by_relative = {}
    for name in before.files:
        try:
            by_relative[Path(name).relative_to(root).as_posix()] = name
        except ValueError:
            continue
    try:
        updated = apply_unified_diff({rel: before.files[name] for rel, name in by_relative.items()}, patch)
    except ValueError as exc:
        return {**base, "success": False, "applies": False, "error": "invalid_patch", "message": str(exc)}
    changes = {by_relative[rel]: text for rel, text in updated.items() if text != before.files[by_relative[rel]]}
    all_files = {**before.files, **changes}
    after = snapshot(all_files[before.entry_file], before.entry_file, all_files, before.config)
    prior_report, next_report = analyze_snapshot(before), analyze_snapshot(after)
    try:
        from .model_diff import compare_models
        comparison = {"status": "available", "changes": compare_models(before.model, after.model).to_dict()}
    except Exception as exc:
        comparison = {"status": "unavailable", "message": str(exc)}
    return {**base, "success": True, "applies": True, "checks_passed": next_report["checks_passed"],
            "after_snapshot_id": after.snapshot_id, "changed_files": changes,
            "before": prior_report, "after": next_report,
            "diagnostic_delta": diagnostic_delta(prior_report, next_report), "structural_review": comparison,
            "review_required": True, "disclaimer": "Fewer diagnostics do not establish that the intended economic repair was made."}


TOOL = dynare_validate_patch
COMMAND = "dynare/validatePatch"
CLI = "patch"
EXAMPLE = {"patch": "", "expected_snapshot_id": "copy ID from a report made with the same config"}
