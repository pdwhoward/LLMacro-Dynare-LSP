"""Workspace-wide index for cross-file Dynare analysis.

Dynare production projects routinely split a model across many files using
the ``@#include`` macro directive.  When the LSP analyzes only the active
file, identifiers declared exclusively in included files trigger spurious
``E020 Undeclared identifier`` errors.  The :class:`WorkspaceIndex`
defined here keeps a thread-safe map of URI → :class:`ParsedModel` and
exposes a small number of cross-file queries used by the diagnostics
engine and the LSP server:

* :meth:`WorkspaceIndex.update_document` re-parses a single document and
  refreshes its entry; called from the ``did_open`` / ``did_change`` /
  ``did_save`` handlers.
* :meth:`WorkspaceIndex.resolve_all_includes` returns every transitively
  included file (reading from disk when an entry isn't already in the
  index).  Visited paths are tracked so circular includes don't recurse
  forever.
* :meth:`WorkspaceIndex.find_circular_includes` returns the cycles that
  touch a given document; the diagnostics engine emits ``E060`` for
  each cycle's first directive.
* :meth:`WorkspaceIndex.collect_symbols` concatenates every declaration
  reachable from a document, so the cross-file undeclared-identifier
  check can treat names declared in included files as visible.

Thread safety: the LSP server is multithreaded (handlers and the
background solver share the index), so a :class:`threading.Lock` guards
all reads and writes of the underlying dict.  Critical sections are kept
to dict access only — parsing and disk I/O run without the lock held.
"""

from __future__ import annotations

import ast
import logging
import re
import threading
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .include_resolver import _uri_to_path, resolve_include_path
from .parser import (
    IncludeDirective,
    MacroDirective,
    ParsedModel,
    Position,
    SourceRange,
    VarDeclaration,
    _extract_macro_defines,
    _mask_string_literals,
    _macro_branch_state,
    _parse_equations,
    _parse_includes,
    _parse_initval_block,
    _parse_macro_directives,
    _parse_macro_for_equations,
    _safe_eval,
    _simple_macro_list_values,
    _simple_macro_for_values,
    _strip_comments,
    _strip_non_macro_comments,
    _substitute_macro_arg,
    parse,
)

logger = logging.getLogger(__name__)


def _normalize_uri(uri_or_path: str) -> str:
    """Normalize a URI or path string to an absolute path string key.

    The LSP delivers ``file://`` URIs; tests and disk reads use plain
    paths.  Both forms must collapse to a single canonical key so the
    same physical file isn't tracked twice.
    """
    p = _uri_to_path(uri_or_path)
    try:
        return str(p.resolve())
    except (OSError, RuntimeError):
        # On some platforms resolve() may fail on a non-existent path;
        # fall back to a lexical absolute path.
        return str(p.absolute())


class WorkspaceIndex:
    """Thread-safe map of URI → :class:`ParsedModel` plus cross-file queries."""

    def __init__(self, search_paths: Optional[List[Path]] = None) -> None:
        self._models: Dict[str, ParsedModel] = {}
        self._sources: Dict[str, str] = {}
        # Search paths used when resolving include filenames that don't
        # exist relative to the including file.  May be extended at
        # runtime by the LSP server once workspace folders are known.
        self._search_paths: List[Path] = list(search_paths or [])
        self._root_search_paths: Dict[str, List[Path]] = {}
        self._file_signatures: Dict[str, Tuple[int, int]] = {}
        self._effective_models: Dict[str, ParsedModel] = {}
        # Search paths contributed by @#includepath directives, keyed by
        # the document that declared them.  These are replaced on each
        # update_document() call so removed directives do not leave stale
        # paths that suppress unresolved-include diagnostics.
        self._document_search_paths: Dict[str, List[Path]] = {}
        self._lock = threading.Lock()

    # -- mutation ----------------------------------------------------------

    def _register_includepath_directives(
        self,
        key: str,
        model: ParsedModel,
    ) -> None:
        """Register every ``@#includepath`` directive in *model* as a search path.

        Paths are resolved relative to the file at *key* so a
        Linux-authored ``@#includepath "../shared"`` works on Windows.
        Shared by :meth:`update_document` (editor-opened files) and
        :meth:`_read_and_parse_from_disk` (transitively-loaded includes),
        so a ``@#includepath`` declared inside an included file extends
        resolution for any further includes reachable from that file.
        """
        including_dir = Path(key).parent if Path(key).is_absolute() else None
        paths: List[Path] = []
        for directive in model.macro_directives:
            if directive.kind != "includepath" or not directive.argument:
                continue
            if not self._line_active(model, directive.range.start.line):
                continue
            for raw in self._split_includepath_argument(directive.argument):
                path = Path(raw.replace("\\", "/"))
                if not path.is_absolute() and including_dir is not None:
                    path = (including_dir / path).resolve()
                if path not in paths:
                    paths.append(path)
        with self._lock:
            self._document_search_paths[key] = paths

    @staticmethod
    def _split_includepath_argument(argument: str) -> List[str]:
        raw = argument.strip()
        if (raw.startswith('"') and raw.endswith('"')) or (
            raw.startswith("'") and raw.endswith("'")
        ):
            raw = raw[1:-1]
        if not raw:
            return []
        parts: List[str] = []
        start = 0
        for i, ch in enumerate(raw):
            if ch != ":":
                continue
            # Preserve Windows drive prefixes such as C:/models.
            if (
                i == start + 1
                and raw[start].isalpha()
                and i + 1 < len(raw)
                and raw[i + 1] in "/\\"
            ):
                continue
            parts.append(raw[start:i].strip())
            start = i + 1
        parts.append(raw[start:].strip())
        return [part for part in parts if part]

    @staticmethod
    def _macro_defines(model: ParsedModel) -> Dict[str, str]:
        return _extract_macro_defines(model.text)

    @staticmethod
    def _macro_event_model(model: ParsedModel) -> ParsedModel:
        source = getattr(model, "original_text", "") or model.text
        masked = _strip_non_macro_comments(source)
        return replace(
            model,
            text=source,
            original_text=source,
            includes=_parse_includes(masked),
            macro_directives=_parse_macro_directives(masked),
        )

    @staticmethod
    def _macro_truth_value(
        argument: Optional[str], defines: Dict[str, str]
    ) -> Optional[bool]:
        if argument is None:
            return None
        raw = argument.strip()
        unknown = object()

        def _scalar(expr: str, seen: Optional[set] = None) -> object:
            seen = seen or set()
            value = expr.strip()
            list_values = _simple_macro_list_values(value)
            if list_values is not None:
                out: List[object] = []
                for item in list_values:
                    resolved = _scalar(item, seen)
                    out.append(item if resolved is unknown else resolved)
                return out
            if value in defines and value not in seen:
                resolved = _scalar(defines[value], seen | {value})
                return defines[value].strip() if resolved is unknown else resolved
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                return value[1:-1].strip()
            lowered = value.lower()
            if lowered in {"", "0", "false", "no"}:
                return False
            if lowered in {"1", "true", "yes"}:
                return True
            try:
                return float(value)
            except ValueError:
                evaluated = _safe_eval(value, {})
                return unknown if evaluated is None else evaluated

        def _truth(value: object) -> Optional[bool]:
            if value is unknown:
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value != 0
            if isinstance(value, str):
                return value.strip().lower() not in {"", "0", "false", "no"}
            return None

        direct = _scalar(raw)
        if direct is not unknown:
            return _truth(direct)

        expression = raw.replace("&&", " and ").replace("||", " or ")
        expression = re.sub(r"(?<![=!<>])!(?!=)", " not ", expression).strip()
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            return None

        def _eval(node: ast.AST) -> object:
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (bool, int, float, str)):
                    return node.value
                return unknown
            if isinstance(node, ast.Name):
                return _scalar(node.id)
            if isinstance(node, ast.List):
                values = [_eval(elt) for elt in node.elts]
                return unknown if any(value is unknown for value in values) else values
            if isinstance(node, ast.Tuple):
                values = [_eval(elt) for elt in node.elts]
                return (
                    unknown
                    if any(value is unknown for value in values)
                    else tuple(values)
                )
            if isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand)
                if operand is unknown:
                    return unknown
                if isinstance(node.op, ast.Not):
                    truth = _truth(operand)
                    return unknown if truth is None else not truth
                if isinstance(operand, (int, float)):
                    if isinstance(node.op, ast.USub):
                        return -operand
                    if isinstance(node.op, ast.UAdd):
                        return operand
                return unknown
            if isinstance(node, ast.BoolOp):
                values = [_truth(_eval(v)) for v in node.values]
                if any(v is None for v in values):
                    return unknown
                if isinstance(node.op, ast.And):
                    return all(values)
                if isinstance(node.op, ast.Or):
                    return any(values)
                return unknown
            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                if not isinstance(left, (int, float)) or not isinstance(
                    right, (int, float)
                ):
                    return unknown
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return unknown if right == 0 else left / right
                if isinstance(node.op, ast.Mod):
                    return unknown if right == 0 else left % right
                return unknown
            if isinstance(node, ast.Compare):
                left = _eval(node.left)
                if left is unknown:
                    return unknown
                for op, comparator in zip(node.ops, node.comparators):
                    right = _eval(comparator)
                    if right is unknown:
                        return unknown
                    if isinstance(op, ast.In):
                        try:
                            ok = left in right  # type: ignore[operator]
                        except TypeError:
                            return unknown
                    elif isinstance(op, ast.NotIn):
                        try:
                            ok = left not in right  # type: ignore[operator]
                        except TypeError:
                            return unknown
                    elif isinstance(op, ast.Eq):
                        ok = left == right
                    elif isinstance(op, ast.NotEq):
                        ok = left != right
                    elif isinstance(op, ast.Lt):
                        ok = left < right  # type: ignore[operator]
                    elif isinstance(op, ast.LtE):
                        ok = left <= right  # type: ignore[operator]
                    elif isinstance(op, ast.Gt):
                        ok = left > right  # type: ignore[operator]
                    elif isinstance(op, ast.GtE):
                        ok = left >= right  # type: ignore[operator]
                    else:
                        return unknown
                    if not ok:
                        return False
                    left = right
                return True
            return unknown

        return _truth(_eval(tree))

    @staticmethod
    def _defines_for_macro_event(
        defines: Dict[str, str],
        directive: object,
    ) -> Dict[str, str]:
        if not isinstance(directive, IncludeDirective) or not directive.macro_defines:
            return defines
        out = dict(defines)
        out.update(directive.macro_defines)
        return out

    def _line_macro_define_variants(
        self,
        model: ParsedModel,
        line: int,
        defines: Dict[str, str],
        preset_defines: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        """Return loop variable bindings visible at *line* but not pre-expanded.

        ``_macro_events`` expands loops that are already resolvable when it is
        built.  A prior include can introduce the list only while the include
        walk is in progress, so body events from that loop arrive once and need
        a late expansion using the current side-effect defines.
        """
        preset = dict(preset_defines or {})
        event_model = self._macro_event_model(model)
        stack: List[MacroDirective] = []
        for directive in sorted(
            event_model.macro_directives,
            key=lambda d: (d.range.start.line, d.range.start.character),
        ):
            if directive.range.start.line >= line:
                break
            if directive.kind == "for":
                stack.append(directive)
            elif directive.kind == "endfor" and stack:
                stack.pop()

        variants: List[Dict[str, str]] = [{}]
        for directive in stack:
            local_defines = dict(defines)
            local_defines.update(preset)
            loop_values = _simple_macro_for_values(directive.argument, local_defines)
            if loop_values is None:
                continue
            name, values = loop_values
            if not values:
                return []
            if name in preset:
                continue
            next_variants: List[Dict[str, str]] = []
            for variant in variants:
                variant_defines = dict(local_defines)
                variant_defines.update(variant)
                resolved_values = _simple_macro_for_values(
                    directive.argument,
                    variant_defines,
                )
                if resolved_values is None:
                    next_variants.append(variant)
                    continue
                _resolved_name, resolved_items = resolved_values
                if not resolved_items:
                    return []
                for value in resolved_items:
                    expanded = dict(variant)
                    expanded[name] = value
                    next_variants.append(expanded)
            variants = next_variants
        return variants

    def _include_define_variants(
        self,
        model: ParsedModel,
        line: int,
        defines: Dict[str, str],
        directive: IncludeDirective,
    ) -> List[Dict[str, str]]:
        variants = self._line_macro_define_variants(
            model,
            line,
            defines,
            directive.macro_defines,
        )
        out: List[Dict[str, str]] = []
        for variant in variants:
            include_defines = dict(defines)
            include_defines.update(variant)
            include_defines.update(directive.macro_defines)
            out.append(include_defines)
        return out

    def _line_active(
        self,
        model: ParsedModel,
        line: int,
        initial_defines: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Return whether a source line is active under simple macro conditionals."""
        model = self._macro_event_model(model)
        defines: Dict[str, str] = dict(initial_defines or {})
        stack: List[dict] = []

        def _current_active() -> bool:
            return all(frame["active"] for frame in stack)

        def _current_certain_active() -> bool:
            return _current_active() and not any(frame["unknown"] for frame in stack)

        for directive in sorted(
            model.macro_directives,
            key=lambda d: (d.range.start.line, d.range.start.character),
        ):
            if directive.range.start.line >= line:
                break
            kind = directive.kind
            if kind == "define" and _current_certain_active() and directive.argument:
                defines.update(
                    _extract_macro_defines(
                        f"@#define {directive.argument}\n",
                        allow_complex=True,
                    ),
                )
                continue

            parent_active = _current_active()
            if kind in {"if", "ifdef", "ifndef"}:
                if kind == "if":
                    truth = self._macro_truth_value(directive.argument, defines)
                else:
                    name = (directive.argument or "").strip()
                    truth = name in defines
                    if kind == "ifndef":
                        truth = not truth
                active = parent_active if truth is None else parent_active and truth
                stack.append(
                    {
                        "parent_active": parent_active,
                        "active": active,
                        "taken": bool(truth) if truth is not None else False,
                        "unknown": truth is None,
                    }
                )
            elif kind == "elseif" and stack:
                frame = stack[-1]
                if frame["unknown"]:
                    frame["active"] = frame["parent_active"]
                elif frame["taken"]:
                    frame["active"] = False
                else:
                    truth = self._macro_truth_value(directive.argument, defines)
                    frame["active"] = (
                        frame["parent_active"]
                        if truth is None
                        else frame["parent_active"] and truth
                    )
                    frame["taken"] = bool(truth) if truth is not None else False
                    frame["unknown"] = truth is None
            elif kind == "else" and stack:
                frame = stack[-1]
                if frame["unknown"]:
                    frame["active"] = frame["parent_active"]
                else:
                    frame["active"] = frame["parent_active"] and not frame["taken"]
                    frame["taken"] = True
            elif kind in {"endif", "endfor"} and stack:
                frame = stack.pop()
                name = frame.get("define_name")
                if name is not None:
                    if frame.get("had_previous"):
                        defines[name] = frame.get("previous_value", "")
                    else:
                        defines.pop(name, None)
            elif kind == "for":
                loop_values = _simple_macro_for_values(directive.argument, defines)
                if loop_values is not None:
                    name, values = loop_values
                    had_previous = name in defines
                    previous_value = defines.get(name)
                    active = parent_active and bool(values)
                    if active:
                        value = (
                            defines[name]
                            if name in defines and defines[name] in values
                            else values[0]
                        )
                        defines[name] = value
                    stack.append(
                        {
                            "parent_active": parent_active,
                            "active": active,
                            "taken": True,
                            "unknown": False,
                            "define_name": name if active else None,
                            "had_previous": had_previous,
                            "previous_value": previous_value,
                        }
                    )
                    continue
                stack.append(
                    {
                        "parent_active": parent_active,
                        "active": parent_active,
                        "taken": True,
                        "unknown": True,
                    }
                )

        return _current_active()

    def _line_has_uncertain_macro_context(
        self,
        model: ParsedModel,
        line: int,
        initial_defines: Optional[Dict[str, str]] = None,
    ) -> bool:
        """True when *line* is active only because an enclosing branch is unknown."""
        model = self._macro_event_model(model)
        defines: Dict[str, str] = dict(initial_defines or {})
        stack: List[dict] = []

        def _current_active() -> bool:
            return all(frame["active"] for frame in stack)

        def _current_certain_active() -> bool:
            return _current_active() and not any(frame["unknown"] for frame in stack)

        for directive in sorted(
            model.macro_directives,
            key=lambda d: (d.range.start.line, d.range.start.character),
        ):
            if directive.range.start.line >= line:
                break
            kind = directive.kind
            if kind == "define" and _current_certain_active() and directive.argument:
                match = re.match(
                    r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$",
                    directive.argument.strip(),
                )
                if match:
                    defines[match.group(1)] = match.group(2).strip()
                continue

            parent_active = _current_active()
            if kind in {"if", "ifdef", "ifndef"}:
                if kind == "if":
                    truth = self._macro_truth_value(directive.argument, defines)
                else:
                    name = (directive.argument or "").strip()
                    truth = name in defines
                    if kind == "ifndef":
                        truth = not truth
                active = parent_active if truth is None else parent_active and truth
                stack.append(
                    {
                        "parent_active": parent_active,
                        "active": active,
                        "taken": bool(truth) if truth is not None else False,
                        "unknown": truth is None,
                    }
                )
            elif kind == "elseif" and stack:
                frame = stack[-1]
                if frame["unknown"]:
                    frame["active"] = frame["parent_active"]
                elif frame["taken"]:
                    frame["active"] = False
                else:
                    truth = self._macro_truth_value(directive.argument, defines)
                    frame["active"] = (
                        frame["parent_active"]
                        if truth is None
                        else frame["parent_active"] and truth
                    )
                    frame["taken"] = bool(truth) if truth is not None else False
                    frame["unknown"] = truth is None
            elif kind == "else" and stack:
                frame = stack[-1]
                if frame["unknown"]:
                    frame["active"] = frame["parent_active"]
                else:
                    frame["active"] = frame["parent_active"] and not frame["taken"]
                    frame["taken"] = True
            elif kind in {"endif", "endfor"} and stack:
                frame = stack.pop()
                name = frame.get("define_name")
                if name is not None:
                    if frame.get("had_previous"):
                        defines[name] = frame.get("previous_value", "")
                    else:
                        defines.pop(name, None)
            elif kind == "for":
                loop_values = _simple_macro_for_values(directive.argument, defines)
                if loop_values is not None:
                    name, values = loop_values
                    had_previous = name in defines
                    previous_value = defines.get(name)
                    active = parent_active and bool(values)
                    if active:
                        value = (
                            defines[name]
                            if name in defines and defines[name] in values
                            else values[0]
                        )
                        defines[name] = value
                    stack.append(
                        {
                            "parent_active": parent_active,
                            "active": active,
                            "taken": True,
                            "unknown": False,
                            "define_name": name if active else None,
                            "had_previous": had_previous,
                            "previous_value": previous_value,
                        }
                    )
                    continue
                stack.append(
                    {
                        "parent_active": parent_active,
                        "active": parent_active,
                        "taken": True,
                        "unknown": True,
                    }
                )

        return _current_active() and any(frame["unknown"] for frame in stack)

    @staticmethod
    def _macro_side_effect_scope_end(model: ParsedModel, line: int) -> int:
        """Return the line where an uncertain-branch side effect stops applying."""
        directives = sorted(
            model.macro_directives,
            key=lambda d: (d.range.start.line, d.range.start.character),
        )
        depth = 0
        target_depth = 0
        for directive in directives:
            dline = directive.range.start.line
            if dline >= line:
                break
            if directive.kind in {"if", "ifdef", "ifndef", "for"}:
                depth += 1
            elif directive.kind in {"endif", "endfor"}:
                depth = max(0, depth - 1)
        target_depth = depth

        scan_depth = target_depth
        for directive in directives:
            dline = directive.range.start.line
            if dline <= line:
                continue
            if directive.kind in {"if", "ifdef", "ifndef", "for"}:
                scan_depth += 1
                continue
            if directive.kind in {"else", "elseif"} and scan_depth == target_depth:
                return dline
            if directive.kind in {"endif", "endfor"}:
                if scan_depth == target_depth:
                    return dline
                scan_depth = max(0, scan_depth - 1)
        return 10**9

    @staticmethod
    def _drop_expired_macro_side_effects(
        line: int,
        side_defines: Dict[str, str],
        scoped_define_ends: Dict[str, int],
        side_paths: List[Path],
        scoped_path_ends: Dict[Path, int],
    ) -> None:
        for name, end_line in list(scoped_define_ends.items()):
            if line >= end_line:
                side_defines.pop(name, None)
                scoped_define_ends.pop(name, None)
        expired_paths = [
            path for path, end_line in scoped_path_ends.items() if line >= end_line
        ]
        for path in expired_paths:
            scoped_path_ends.pop(path, None)
            while path in side_paths:
                side_paths.remove(path)

    def _includepath_paths(self, key: str, directive: MacroDirective) -> List[Path]:
        including_dir = Path(key).parent if Path(key).is_absolute() else None
        paths: List[Path] = []
        if not directive.argument:
            return paths
        for raw in self._split_includepath_argument(directive.argument):
            path = Path(raw.replace("\\", "/"))
            if not path.is_absolute() and including_dir is not None:
                path = (including_dir / path).resolve()
            if path not in paths:
                paths.append(path)
        return paths

    def _active_include_events(
        self,
        key: str,
        model: ParsedModel,
        inherited_paths: List[Path],
        inherited_defines: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[IncludeDirective, List[Path], Dict[str, str]]]:
        """Return active includes with search paths visible at each line."""
        model = self._macro_event_model(model)
        events = self._macro_events(model)

        paths = list(inherited_paths)
        defines: Dict[str, str] = dict(inherited_defines or {})
        includes: List[Tuple[IncludeDirective, List[Path], Dict[str, str]]] = []
        scoped_define_ends: Dict[str, int] = {}
        scoped_path_ends: Dict[Path, int] = {}
        for line, _character, kind, directive in events:
            self._drop_expired_macro_side_effects(
                line,
                defines,
                scoped_define_ends,
                paths,
                scoped_path_ends,
            )
            active_defines = self._defines_for_macro_event(defines, directive)
            if not self._line_active(model, line, active_defines):
                continue
            if kind == "define":
                if not isinstance(directive, MacroDirective):
                    continue
                if directive.argument:
                    was_uncertain = self._line_has_uncertain_macro_context(
                        model,
                        line,
                        defines,
                    )
                    new_defines = _extract_macro_defines(
                        f"@#define {directive.argument}\n",
                        allow_complex=True,
                    )
                    new_defines = {
                        name: _substitute_macro_arg(value, defines)
                        for name, value in new_defines.items()
                    }
                    defines.update(new_defines)
                    if was_uncertain:
                        end_line = self._macro_side_effect_scope_end(model, line)
                        for name in new_defines:
                            scoped_define_ends[name] = end_line
            elif kind == "includepath":
                if not isinstance(directive, MacroDirective):
                    continue
                original_argument = directive.argument
                if original_argument is not None:
                    directive = replace(
                        directive,
                        argument=_substitute_macro_arg(original_argument, defines),
                    )
                new_paths = self._includepath_paths(key, directive)
                added_paths = [path for path in new_paths if path not in paths]
                paths = self._append_unique(paths, new_paths)
                if self._line_has_uncertain_macro_context(model, line, defines):
                    end_line = self._macro_side_effect_scope_end(model, line)
                    for path in added_paths:
                        scoped_path_ends[path] = end_line
            else:
                if not isinstance(directive, IncludeDirective):
                    continue
                for include_defines in self._include_define_variants(
                    model,
                    line,
                    defines,
                    directive,
                ):
                    include = replace(
                        directive,
                        filename=_substitute_macro_arg(
                            directive.filename,
                            include_defines,
                        ),
                    )
                    includes.append((include, list(paths), include_defines))
        return includes

    def _macro_events(self, model: ParsedModel) -> List[Tuple[int, int, str, object]]:
        """Return define/includepath/include events in source order."""
        model = self._macro_event_model(model)
        raw_events: List[Tuple[int, int, str, object]] = []
        for directive in model.macro_directives:
            if directive.kind in {"define", "includepath", "for", "endfor"}:
                raw_events.append(
                    (
                        directive.range.start.line,
                        directive.range.start.character,
                        directive.kind,
                        directive,
                    )
                )
        for directive in model.includes:
            raw_events.append(
                (
                    directive.range.start.line,
                    directive.range.start.character,
                    "include",
                    directive,
                )
            )
        raw_events.sort(key=lambda item: (item[0], item[1]))

        def _emit_event(
            line: int,
            character: int,
            kind: str,
            directive: object,
            _active_defines: Dict[str, str],
            substitution_defines: Dict[str, str],
            offset: int,
        ) -> Optional[Tuple[int, int, str, object]]:
            if kind not in {"define", "includepath", "include"}:
                return None
            if isinstance(directive, IncludeDirective):
                return (
                    line,
                    character + offset,
                    kind,
                    replace(directive, macro_defines=dict(substitution_defines)),
                )
            if isinstance(directive, MacroDirective):
                argument = directive.argument
                if argument is not None:
                    argument = _substitute_macro_arg(
                        argument,
                        substitution_defines,
                    )
                return (
                    line,
                    character + offset,
                    kind,
                    replace(directive, argument=argument),
                )
            return None

        def _expand_events(
            events_in: List[Tuple[int, int, str, object]],
            inherited_defines: Dict[str, str],
            inherited_substitution_defines: Optional[Dict[str, str]] = None,
        ) -> List[Tuple[int, int, str, object]]:
            out: List[Tuple[int, int, str, object]] = []
            current_defines = dict(inherited_defines)
            current_substitution_defines = dict(
                inherited_substitution_defines
                if inherited_substitution_defines is not None
                else inherited_defines
            )
            scoped_define_ends: Dict[str, int] = {}
            scoped_define_restores: Dict[
                str,
                Tuple[bool, Optional[str], bool, Optional[str]],
            ] = {}

            def _drop_expired_defines(line: int) -> None:
                for name, end_line in list(scoped_define_ends.items()):
                    if line < end_line:
                        continue
                    had_current, current_value, had_subst, subst_value = (
                        scoped_define_restores.get(name, (False, None, False, None))
                    )
                    if had_current:
                        current_defines[name] = current_value or ""
                    else:
                        current_defines.pop(name, None)
                    if had_subst:
                        current_substitution_defines[name] = subst_value or ""
                    else:
                        current_substitution_defines.pop(name, None)
                    scoped_define_ends.pop(name, None)
                    scoped_define_restores.pop(name, None)

            def _remember_scoped_define(name: str, end_line: int) -> None:
                if name not in scoped_define_restores:
                    scoped_define_restores[name] = (
                        name in current_defines,
                        current_defines.get(name),
                        name in current_substitution_defines,
                        current_substitution_defines.get(name),
                    )
                scoped_define_ends[name] = min(
                    scoped_define_ends.get(name, end_line),
                    end_line,
                )

            idx = 0
            while idx < len(events_in):
                line, character, kind, directive = events_in[idx]
                _drop_expired_defines(line)
                if kind == "for" and isinstance(directive, MacroDirective):
                    end_idx = None
                    depth = 0
                    for scan_idx in range(idx, len(events_in)):
                        scan_kind = events_in[scan_idx][2]
                        if scan_kind == "for":
                            depth += 1
                        elif scan_kind == "endfor":
                            depth -= 1
                            if depth == 0:
                                end_idx = scan_idx
                                break
                    if end_idx is None:
                        idx += 1
                        continue
                    body = events_in[idx + 1 : end_idx]
                    loop_values = _simple_macro_for_values(
                        directive.argument, current_defines
                    )
                    if loop_values is None:
                        out.extend(
                            _expand_events(
                                body,
                                dict(current_defines),
                                dict(current_substitution_defines),
                            ),
                        )
                    else:
                        name, values = loop_values
                        for offset, value in enumerate(values):
                            loop_defines = dict(current_defines)
                            loop_substitution_defines = dict(
                                current_substitution_defines,
                            )
                            loop_defines[name] = value
                            loop_substitution_defines[name] = value
                            expanded = _expand_events(
                                body,
                                loop_defines,
                                loop_substitution_defines,
                            )
                            out.extend(
                                (e_line, e_char + offset, e_kind, e_directive)
                                for e_line, e_char, e_kind, e_directive in expanded
                            )
                    idx = end_idx + 1
                    continue
                if kind == "endfor":
                    idx += 1
                    continue
                emitted = _emit_event(
                    line,
                    character,
                    kind,
                    directive,
                    current_defines,
                    current_substitution_defines,
                    0,
                )
                if emitted is not None:
                    out.append(emitted)
                    if kind == "define" and isinstance(emitted[3], MacroDirective):
                        if emitted[3].argument and self._line_active(
                            model,
                            line,
                            current_defines,
                        ):
                            was_uncertain = self._line_has_uncertain_macro_context(
                                model,
                                line,
                                current_defines,
                            )
                            all_defines = _extract_macro_defines(
                                f"@#define {emitted[3].argument}\n",
                                allow_complex=True,
                            )
                            safe_defines = _extract_macro_defines(
                                f"@#define {emitted[3].argument}\n",
                            )
                            if was_uncertain:
                                end_line = self._macro_side_effect_scope_end(
                                    model,
                                    line,
                                )
                                for name in all_defines:
                                    _remember_scoped_define(name, end_line)
                            current_defines.update(all_defines)
                            current_substitution_defines.update(safe_defines)
                idx += 1
            return out

        return _expand_events(raw_events, {})

    def update_document(self, uri: str, source: str) -> None:
        """Re-parse *source* and refresh the entry keyed by *uri*.

        Also registers any ``@#includepath`` directives found in *source*
        as workspace search paths, mimicking Dynare's preprocessor.
        """
        key = _normalize_uri(uri)
        model = parse(source)
        with self._lock:
            self._models[key] = model
            self._sources[key] = source
            self._file_signatures.pop(key, None)
            self._effective_models.pop(key, None)
        self._register_includepath_directives(key, model)

    def remove_document(self, uri: str) -> None:
        """Drop the entry keyed by *uri*."""
        key = _normalize_uri(uri)
        with self._lock:
            self._models.pop(key, None)
            self._sources.pop(key, None)
            self._document_search_paths.pop(key, None)
            self._file_signatures.pop(key, None)
            self._effective_models.pop(key, None)

    def add_search_path(self, path: Path) -> None:
        """Register an additional directory to consult for include resolution."""
        with self._lock:
            if path not in self._search_paths:
                self._search_paths.append(path)

    def set_search_paths(self, paths: List[Path]) -> None:
        """Replace configured workspace search paths, preserving order."""
        deduped: List[Path] = []
        for path in paths:
            if path not in deduped:
                deduped.append(path)
        with self._lock:
            self._search_paths = deduped

    def set_root_search_paths(self, paths_by_root: Dict[Path, List[Path]]) -> None:
        """Replace configured search paths scoped to workspace roots."""
        normalized: Dict[str, List[Path]] = {}
        for root, paths in paths_by_root.items():
            try:
                root_key = str(root.resolve())
            except (OSError, RuntimeError):
                root_key = str(root.absolute())
            deduped: List[Path] = []
            for path in paths:
                if path not in deduped:
                    deduped.append(path)
            normalized[root_key] = deduped
        with self._lock:
            self._root_search_paths = normalized

    @staticmethod
    def _append_unique(paths: List[Path], candidates: List[Path]) -> List[Path]:
        """Return *paths* plus candidates not already present, preserving order."""
        out = list(paths)
        for path in candidates:
            if path not in out:
                out.append(path)
        return out

    def _paths_for_document(self, key: str) -> List[Path]:
        """Snapshot ``@#includepath`` paths declared by one document."""
        with self._lock:
            return list(self._document_search_paths.get(key, []))

    def _configured_search_paths_for_key_unlocked(self, key: str) -> List[Path]:
        if not self._root_search_paths:
            return list(self._search_paths)
        try:
            including = Path(key).resolve()
        except (OSError, RuntimeError):
            including = Path(key).absolute()
        matches: List[Tuple[int, List[Path]]] = []
        for root_key, paths in self._root_search_paths.items():
            root = Path(root_key)
            try:
                including.relative_to(root)
            except ValueError:
                continue
            matches.append((len(root.parts), paths))
        if not matches:
            return list(self._search_paths)
        matches.sort(key=lambda item: item[0], reverse=True)
        return self._append_unique(matches[0][1], self._search_paths)

    # -- read-only accessors ----------------------------------------------

    def get_model(self, uri: str) -> Optional[ParsedModel]:
        """Return the cached :class:`ParsedModel` for *uri*, or ``None``."""
        key = _normalize_uri(uri)
        with self._lock:
            return self._models.get(key)

    def get_effective_model(self, uri: str) -> Optional[ParsedModel]:
        """Return the model parsed with include-created macro side effects."""
        key = _normalize_uri(uri)
        with self._lock:
            return self._effective_models.get(key) or self._models.get(key)

    def get_source(self, uri: str) -> Optional[str]:
        """Return the cached source text for *uri*, or ``None``.

        Mirrors :meth:`get_model` — used by callers that need the raw
        text (e.g. cross-file reference scanning) for files that may
        have been loaded from disk but never opened in the editor.
        """
        key = _normalize_uri(uri)
        with self._lock:
            return self._sources.get(key)

    def all_uris(self) -> List[str]:
        """Snapshot of every URI currently tracked."""
        with self._lock:
            return list(self._models.keys())

    # -- include resolution -----------------------------------------------

    def _read_and_parse_from_disk(self, path: Path) -> Optional[ParsedModel]:
        """Read *path* from disk, parse it, and cache the result.

        Returns ``None`` on read failure; callers treat that as "this
        include doesn't contribute symbols" rather than propagating the
        error, since the missing file might just be transient.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Legacy-encoded include (cp1252/latin-1 comments are common in
            # older Dynare projects, and Dynare itself is byte-tolerant).
            # latin-1 decodes any byte sequence and keeps offsets 1:1, so
            # cross-file features keep working instead of silently dropping
            # the file (which produced partial renames and false P001s).
            try:
                text = path.read_text(encoding="latin-1")
            except OSError as exc:
                logger.debug("Failed to read include %s: %s", path, exc)
                return None
        except OSError as exc:
            logger.debug("Failed to read include %s: %s", path, exc)
            return None
        model = parse(text)
        key = str(path.resolve())
        try:
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            signature = None
        with self._lock:
            self._models[key] = model
            self._sources[key] = text
            self._effective_models.pop(key, None)
            if signature is not None:
                self._file_signatures[key] = signature
        # Also register the loaded file's own @#includepath directives so
        # a transitive include chain whose deeper hops depend on a path
        # declared inside an included file still resolves.
        self._register_includepath_directives(key, model)
        return model

    def _load_include_walk_model(
        self,
        key: str,
        macro_defines: Optional[Dict[str, str]] = None,
    ) -> Tuple[Optional[ParsedModel], Optional[str]]:
        """Load a model/source pair for include graph walks."""
        with self._lock:
            model = self._models.get(key)
            source = self._sources.get(key)
        path = Path(key)
        if model is not None and self._disk_cache_stale(key, path):
            model = self._read_and_parse_from_disk(path)
            with self._lock:
                source = self._sources.get(key)
        if model is None:
            if not path.exists():
                return None, None
            model = self._read_and_parse_from_disk(path)
            if model is None:
                return None, None
            with self._lock:
                source = self._sources.get(key)
        if macro_defines and source is not None:
            return parse(source, initial_macro_defines=macro_defines), source
        return model, source

    def resolve_direct_includes(
        self,
        uri: str,
    ) -> List[Tuple[IncludeDirective, str, Optional[str]]]:
        """Return resolved include directives that appear directly in *uri*.

        The walk still descends into earlier includes to collect macro
        side effects such as ``@#define`` and ``@#includepath``.  Only the
        root file's own directives are emitted, which is what LSP document
        links and open-include parent lookups need.
        """
        root_key = _normalize_uri(uri)
        records: List[Tuple[IncludeDirective, str, Optional[str]]] = []
        stack_keys: set = {root_key}

        def _visit(
            current_key: str,
            active_search_paths: List[Path],
            inherited_context: Optional[str],
            inherited_defines: Optional[Dict[str, str]],
            emit_directives: bool,
        ) -> Tuple[List[Path], Dict[str, str]]:
            current_model, _current_source = self._load_include_walk_model(
                current_key,
                inherited_defines,
            )
            if current_model is None:
                return [], {}

            side_paths: List[Path] = []
            side_defines: Dict[str, str] = {}
            scoped_define_ends: Dict[str, int] = {}
            scoped_path_ends: Dict[Path, int] = {}
            for line, _character, kind, directive in self._macro_events(current_model):
                self._drop_expired_macro_side_effects(
                    line,
                    side_defines,
                    scoped_define_ends,
                    side_paths,
                    scoped_path_ends,
                )
                effective_paths = self._append_unique(active_search_paths, side_paths)
                effective_defines = dict(inherited_defines or {})
                effective_defines.update(side_defines)
                active_defines = self._defines_for_macro_event(
                    effective_defines,
                    directive,
                )
                if not self._line_active(current_model, line, active_defines):
                    continue
                if kind == "define":
                    if isinstance(directive, MacroDirective) and directive.argument:
                        was_uncertain = self._line_has_uncertain_macro_context(
                            current_model,
                            line,
                            effective_defines,
                        )
                        new_defines = _extract_macro_defines(
                            f"@#define {directive.argument}\n",
                            allow_complex=True,
                        )
                        new_defines = {
                            name: _substitute_macro_arg(value, effective_defines)
                            for name, value in new_defines.items()
                        }
                        side_defines.update(new_defines)
                        if was_uncertain:
                            end_line = self._macro_side_effect_scope_end(
                                current_model,
                                line,
                            )
                            for name in new_defines:
                                scoped_define_ends[name] = end_line
                    continue
                if kind == "includepath":
                    if not isinstance(directive, MacroDirective):
                        continue
                    was_uncertain = self._line_has_uncertain_macro_context(
                        current_model,
                        line,
                        effective_defines,
                    )
                    original_argument = directive.argument
                    if original_argument is not None:
                        directive = replace(
                            directive,
                            argument=_substitute_macro_arg(
                                original_argument,
                                effective_defines,
                            ),
                        )
                    new_paths = self._includepath_paths(current_key, directive)
                    added_paths = [path for path in new_paths if path not in side_paths]
                    side_paths = self._append_unique(side_paths, new_paths)
                    if was_uncertain:
                        end_line = self._macro_side_effect_scope_end(
                            current_model,
                            line,
                        )
                        for path in added_paths:
                            scoped_path_ends[path] = end_line
                    continue
                if not isinstance(directive, IncludeDirective):
                    continue

                for include_defines in self._include_define_variants(
                    current_model,
                    line,
                    effective_defines,
                    directive,
                ):
                    include_was_uncertain = self._line_has_uncertain_macro_context(
                        current_model,
                        line,
                        include_defines,
                    )
                    resolved_directive = replace(
                        directive,
                        filename=_substitute_macro_arg(
                            directive.filename,
                            include_defines,
                        ),
                    )
                    context = (
                        self._directive_context(current_model, resolved_directive)
                        or inherited_context
                    )
                    resolved = self._resolve_directive(
                        current_key,
                        resolved_directive.filename,
                        effective_paths,
                    )
                    if resolved is None:
                        continue
                    resolved_key = str(resolved)
                    if emit_directives:
                        records.append((resolved_directive, resolved_key, context))
                    if resolved_key in stack_keys:
                        continue
                    stack_keys.add(resolved_key)
                    nested_paths, nested_defines = _visit(
                        resolved_key,
                        effective_paths,
                        context,
                        include_defines,
                        False,
                    )
                    stack_keys.remove(resolved_key)
                    added_nested_paths = [
                        path for path in nested_paths if path not in side_paths
                    ]
                    side_paths = self._append_unique(side_paths, nested_paths)
                    side_defines.update(nested_defines)
                    if include_was_uncertain:
                        end_line = self._macro_side_effect_scope_end(
                            current_model,
                            line,
                        )
                        for path in added_nested_paths:
                            scoped_path_ends[path] = end_line
                        for name in nested_defines:
                            scoped_define_ends[name] = end_line
            return side_paths, side_defines

        _visit(root_key, [], None, None, True)
        return records

    def _disk_cache_stale(self, key: str, path: Path) -> bool:
        """True if a disk-loaded cached model no longer matches the file."""
        with self._lock:
            signature = self._file_signatures.get(key)
        if signature is None:
            return False
        try:
            stat = path.stat()
        except OSError:
            return True
        return signature != (stat.st_mtime_ns, stat.st_size)

    @staticmethod
    def _contains_range(outer: Optional[SourceRange], inner: SourceRange) -> bool:
        if outer is None:
            return False
        start = (inner.start.line, inner.start.character)
        end = (inner.end.line, inner.end.character)
        outer_start = (outer.start.line, outer.start.character)
        outer_end = (outer.end.line, outer.end.character)
        return outer_start <= start and end <= outer_end

    def _directive_context(
        self,
        model: ParsedModel,
        directive: IncludeDirective,
    ) -> Optional[str]:
        if self._contains_range(model.model_block_range, directive.range):
            return "model"
        if self._contains_range(model.steady_state_block_range, directive.range):
            return "steady_state_model"
        if self._contains_range(model.initval_block_range, directive.range):
            return "initval"
        if self._contains_range(model.endval_block_range, directive.range):
            return "endval"
        if self._contains_range(model.shocks_block_range, directive.range):
            return "shocks"
        return self._partial_directive_context(model, directive)

    @staticmethod
    def _position_to_offset(text: str, pos: Position) -> int:
        lines = text.splitlines(keepends=True)
        offset = 0
        for line in lines[: pos.line]:
            offset += len(line)
        if pos.line < len(lines):
            offset += min(pos.character, len(lines[pos.line]))
        return min(offset, len(text))

    @classmethod
    def _partial_directive_context(
        cls,
        model: ParsedModel,
        directive: IncludeDirective,
    ) -> Optional[str]:
        """Infer context for includes inside a block closed by that include."""
        source_text = getattr(model, "original_text", "") or model.text
        if not source_text:
            return None
        stripped = _strip_comments(source_text)
        directive_offset = cls._position_to_offset(stripped, directive.range.start)
        scan = _mask_string_literals(stripped[:directive_offset])
        block_re = re.compile(
            r"(?<!\w)(steady_state_model|model|initval|endval|shocks)"
            r"\s*(?:\([^)]*\))?\s*;|(?<!\w)end\s*;",
            re.IGNORECASE,
        )
        stack: List[str] = []
        for match in block_re.finditer(scan):
            opener = match.group(1)
            if opener:
                stack.append(opener.lower())
            elif stack:
                stack.pop()
        if not stack:
            return None
        context = stack[-1]
        return (
            context
            if context
            in {
                "model",
                "steady_state_model",
                "initval",
                "endval",
                "shocks",
            }
            else None
        )

    @staticmethod
    def _range_in_expansion(anchor_range: SourceRange, rng: SourceRange) -> SourceRange:
        """Approximate *rng* after textual insertion at *anchor_range*.

        Diagnostics should still anchor to the visible include directive, but
        assignment ordering needs a stable textual order across nested includes.
        The synthetic range preserves that order without pretending we know the
        included file's real location in the active document.
        """

        def shift(pos: Position) -> Position:
            line = anchor_range.start.line + pos.line
            character = (
                anchor_range.start.character + pos.character
                if pos.line == 0
                else pos.character
            )
            return Position(line, character)

        return SourceRange(shift(rng.start), shift(rng.end))

    @staticmethod
    def _assignment_range_in_expansion(
        anchor_range: SourceRange,
        rng: SourceRange,
    ) -> SourceRange:
        """Synthetic assignment range whose order matches textual expansion."""
        base_line = anchor_range.start.line
        base_char = anchor_range.start.character

        def shift(pos: Position) -> Position:
            return Position(base_line, base_char + pos.line * 10000 + pos.character)

        return SourceRange(shift(rng.start), shift(rng.end))

    def _contextualize_include(
        self,
        model: ParsedModel,
        source: Optional[str],
        context: Optional[str],
        anchor_range: SourceRange,
        macro_defines: Optional[Dict[str, str]] = None,
        preserve_ranges: bool = False,
    ) -> ParsedModel:
        """Parse raw include-body content in the block that included it.

        Dynare expands ``@#include`` textually.  A helper file included inside
        ``model; ... end;`` can therefore contain only equation statements
        rather than its own explicit ``model`` block.  The normal single-file
        parser cannot infer that context, so the workspace index supplies a
        shallow view with those raw statements parsed into the relevant block.
        """
        view = replace(model)
        view.include_anchor_range = anchor_range
        view.include_context = context

        def _map_range(rng: SourceRange) -> SourceRange:
            return (
                rng
                if preserve_ranges
                else self._assignment_range_in_expansion(
                    anchor_range,
                    rng,
                )
            )

        def _map_error(
            err: Tuple[str, SourceRange, dict],
        ) -> Tuple[str, SourceRange, dict]:
            return (err[0], _map_range(err[1]), err[2])

        view.model_equations = [
            replace(
                eq,
                range=_map_range(eq.range),
            )
            for eq in model.model_equations
        ]
        view.steady_state_equations = [
            replace(
                eq,
                range=_map_range(eq.range),
            )
            for eq in model.steady_state_equations
        ]
        view.initval_entries = [
            replace(
                entry,
                range=_map_range(entry.range),
            )
            for entry in model.initval_entries
        ]
        view.param_assignments = [
            replace(
                assignment,
                range=_map_range(assignment.range),
            )
            for assignment in model.param_assignments
        ]
        view.helper_assignments = [
            replace(
                assignment,
                range=_map_range(assignment.range),
            )
            for assignment in model.helper_assignments
        ]
        view.errors = []
        if context != "shocks":
            view.errors = [(err[0], _map_range(err[1])) for err in model.errors]
        if source is not None and context is None:
            missing_semicolon_error = self._include_missing_semicolon_error(
                source,
                context,
            )
            if missing_semicolon_error is not None:
                view.errors.append(_map_error(missing_semicolon_error))

        if context is None or source is None:
            return view

        context_source = model.text
        raw_context_source = source or getattr(model, "original_text", "") or model.text
        stripped_source = _strip_comments(context_source)
        raw_macro_source = _strip_non_macro_comments(raw_context_source)
        context_body, closes_parent = self._context_body_before_closing_end(
            stripped_source,
        )
        raw_context_body, _raw_closes_parent = self._context_body_before_closing_end(
            raw_macro_source,
        )
        missing_semicolon_error = self._include_missing_semicolon_error(
            context_body,
            context,
        )
        if missing_semicolon_error is not None:
            view.errors.append(_map_error(missing_semicolon_error))

        def _context_equations():
            base = _parse_equations(
                context_body,
                0,
                context_source,
                filter_commands=True,
            )
            _defines, _active_lines, context_line_defines = _macro_branch_state(
                raw_context_source,
                macro_defines,
            )
            macro_equations, macro_ranges = _parse_macro_for_equations(
                raw_context_body,
                0,
                raw_context_source,
                context_line_defines,
            )
            if not macro_equations:
                return base
            filtered_base = [
                eq
                for eq in base
                if not any(
                    start <= eq.range.start.line <= end for start, end in macro_ranges
                )
            ]
            equations = filtered_base + macro_equations

            # Anchor-only stable sort: expanded loop equations keep their
            # iteration-major emission order at the loop's position.  A
            # full positional sort would regroup them statement-major,
            # which breaks sequential steady_state_model recursions.
            def _anchor(eq) -> int:
                line = eq.range.start.line
                for start, end in macro_ranges:
                    if start <= line <= end:
                        return start
                return line

            equations.sort(key=_anchor)
            return equations

        if context == "model" and not view.model_equations:
            if closes_parent:
                view.context_closing_block = context
            view.model_equations = [
                replace(
                    eq,
                    range=_map_range(eq.range),
                )
                for eq in _context_equations()
            ]
            view.param_assignments = []
            view.helper_assignments = []
        elif context == "steady_state_model" and not view.steady_state_equations:
            if closes_parent:
                view.context_closing_block = context
            view.steady_state_equations = [
                replace(
                    eq,
                    range=_map_range(eq.range),
                )
                for eq in _context_equations()
            ]
            view.param_assignments = []
            view.helper_assignments = []
        elif context == "initval" and not view.initval_entries:
            if closes_parent:
                view.context_closing_block = context
            view.initval_entries = [
                replace(
                    entry,
                    range=_map_range(entry.range),
                )
                for entry in _parse_initval_block(
                    context_body,
                    0,
                    context_source,
                    view.param_values(),
                )
            ]
            view.param_assignments = []
            view.helper_assignments = []
        elif context == "endval" and not view.endval_entries:
            if closes_parent:
                view.context_closing_block = context
            view.endval_entries = [
                replace(
                    entry,
                    range=_map_range(entry.range),
                )
                for entry in _parse_initval_block(
                    context_body,
                    0,
                    context_source,
                    view.param_values(),
                )
            ]
            view.param_assignments = []
            view.helper_assignments = []
        elif context == "shocks":
            view.endogenous = []
            view.exogenous = []
            view.deterministic_exogenous = []
            view.parameters = []
            view.predetermined_variables = []
            view.model_equations = []
            view.steady_state_equations = []
            view.initval_entries = []
            view.param_assignments = []
            view.helper_assignments = []
            shock_names: List[str] = []
            shock_decl = re.compile(
                r"\b(?:var|corr)\s+"
                r"([A-Za-z][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z][A-Za-z0-9_]*)*)",
                re.IGNORECASE,
            )
            for match in shock_decl.finditer(stripped_source):
                for name in re.split(r"\s*,\s*", match.group(1).strip()):
                    if name and name not in shock_names:
                        shock_names.append(name)
            view.shocks_vars = shock_names
            view.shocks_block_range = anchor_range
        return view

    @staticmethod
    def _include_missing_semicolon_error(
        source: str,
        context: Optional[str],
    ) -> Optional[Tuple[str, SourceRange, dict]]:
        stripped = _strip_comments(source)
        lines = stripped.splitlines()
        for line_no in range(len(lines) - 1, -1, -1):
            line = lines[line_no]
            content = line.strip()
            if not content or content.startswith("@#"):
                continue
            if content.endswith(";"):
                return None
            if context is None and not re.match(
                r"[A-Za-z_][A-Za-z0-9_]*\s*=",
                content,
            ):
                return None
            if context not in {
                None,
                "model",
                "steady_state_model",
                "initval",
                "endval",
                "shocks",
            }:
                return None
            start_char = len(line) - len(line.lstrip())
            end_char = len(line.rstrip())
            if context in {
                "model",
                "steady_state_model",
                "initval",
                "endval",
                "shocks",
            }:
                message = (
                    f"Statement in included '{context}' fragment is missing "
                    "its terminating semicolon before the parent continues. "
                    f"Fix: add ';' at the end of line {line_no + 1}."
                )
            else:
                name = content.split("=", 1)[0].strip()
                message = (
                    f"Parameter assignment '{name}' is missing its terminating "
                    f"semicolon. Fix: add ';' at the end of line {line_no + 1}."
                )
            fix = {
                "start_line": line_no,
                "start_char": end_char,
                "end_line": line_no,
                "end_char": end_char,
                "new_text": ";",
            }
            return (
                message,
                SourceRange(
                    Position(line_no, start_char),
                    Position(line_no, end_char),
                ),
                fix,
            )
        return None

    @staticmethod
    def _context_body_before_closing_end(stripped_source: str) -> Tuple[str, bool]:
        # A contextual include may contain equation tags such as
        # ``[name='contains end; tag']``.  The embedded text is data, not the
        # parent block closer, so scan for ``end;`` after masking strings in
        # the same offset-preserving style used by parser block extraction.
        structural_source = _mask_string_literals(stripped_source)
        match = re.search(r"(?<!\w)end\s*;", structural_source, re.IGNORECASE)
        if match is None:
            return stripped_source, False
        return (
            stripped_source[: match.start()]
            + " " * (len(stripped_source) - match.start()),
            True,
        )

    def _resolve_directive(
        self,
        including_key: str,
        filename: str,
        active_search_paths: Optional[List[Path]] = None,
    ) -> Optional[Path]:
        """Resolve a directive filename against the index's search paths.

        ``active_search_paths`` contains ``@#includepath`` directives already
        reached on the current include chain.  Paths from unrelated open
        documents are intentionally excluded so they cannot hide unresolved
        include diagnostics in another model.
        """
        with self._lock:
            search_paths = self._configured_search_paths_for_key_unlocked(
                including_key,
            )
            model_keys = list(self._models.keys())
            disk_loaded_keys = set(self._file_signatures.keys())
        known_paths = {
            key
            for key in model_keys
            if key not in disk_loaded_keys or Path(key).is_file()
        }
        search_paths = self._append_unique(search_paths, active_search_paths or [])
        return resolve_include_path(
            filename,
            including_key,
            search_paths,
            known_paths=known_paths,
        )

    def resolve_all_includes(self, uri: str) -> Dict[str, ParsedModel]:
        """Return every transitively included file's :class:`ParsedModel`.

        Files already in the index are reused; missing ones are read from
        disk on demand.  A visited set prevents infinite recursion on
        circular includes — the root *uri* itself is excluded from the
        result so callers can distinguish "this file" from "an included
        file".
        """
        root_key = _normalize_uri(uri)
        result: Dict[str, ParsedModel] = {}
        stack_keys: set = {root_key}
        emitted_counts: Dict[str, int] = {}
        root_line_defines: Dict[int, Dict[str, str]] = {}

        def _record_line_defines_after(
            line_defines: Dict[int, Dict[str, str]],
            line_count: int,
            line: int,
            defines: Dict[str, str],
            local_define_lines: Dict[str, List[int]],
            stop_line: Optional[int] = None,
        ) -> None:
            if not defines or line_count == 0:
                return
            end_line = line_count if stop_line is None else min(line_count, stop_line)
            for line_no in range(line + 1, end_line):
                merged = dict(line_defines.get(line_no, {}))
                for name, value in defines.items():
                    if any(
                        line < define_line < line_no
                        for define_line in local_define_lines.get(name, [])
                    ):
                        continue
                    merged[name] = value
                line_defines[line_no] = merged

        def _load_model(
            current_key: str,
            macro_defines: Optional[Dict[str, str]] = None,
        ) -> Optional[ParsedModel]:
            with self._lock:
                current_model = self._models.get(current_key)
                current_source = self._sources.get(current_key)
            current_path = Path(current_key)
            if current_model is not None and self._disk_cache_stale(
                current_key,
                current_path,
            ):
                current_model = self._read_and_parse_from_disk(current_path)
                with self._lock:
                    current_source = self._sources.get(current_key)
            if current_model is None:
                # Lazy disk load for files that were never opened in the editor.
                current_path = Path(current_key)
                if not current_path.exists():
                    return None
                current_model = self._read_and_parse_from_disk(current_path)
                if current_model is None:
                    return None
                with self._lock:
                    current_source = self._sources.get(current_key)
            if macro_defines and current_source is not None:
                return parse(current_source, initial_macro_defines=macro_defines)
            return current_model

        def _visit(
            current_key: str,
            active_search_paths: List[Path],
            root_anchor: Optional[SourceRange],
            inherited_context: Optional[str],
            inherited_defines: Optional[Dict[str, str]],
        ) -> Tuple[List[Path], Dict[str, str], ParsedModel]:
            current_model = _load_model(current_key, inherited_defines)
            if current_model is None:
                return [], {}, parse("")
            with self._lock:
                current_source = self._sources.get(current_key)
            current_line_count = (
                len(current_source.splitlines()) if current_source is not None else 0
            )
            current_line_defines = root_line_defines if current_key == root_key else {}
            local_define_lines: Dict[str, List[int]] = {}
            for directive in current_model.macro_directives:
                if directive.kind != "define" or not directive.argument:
                    continue
                for name in _extract_macro_defines(
                    f"@#define {directive.argument}\n",
                    allow_complex=True,
                ):
                    local_define_lines.setdefault(name, []).append(
                        directive.range.start.line,
                    )

            side_paths: List[Path] = []
            side_defines: Dict[str, str] = {}
            scoped_define_ends: Dict[str, int] = {}
            scoped_path_ends: Dict[Path, int] = {}
            for line, _character, kind, directive in self._macro_events(current_model):
                self._drop_expired_macro_side_effects(
                    line,
                    side_defines,
                    scoped_define_ends,
                    side_paths,
                    scoped_path_ends,
                )
                effective_paths = self._append_unique(active_search_paths, side_paths)
                effective_defines = dict(inherited_defines or {})
                effective_defines.update(side_defines)
                active_defines = self._defines_for_macro_event(
                    effective_defines,
                    directive,
                )
                if not self._line_active(current_model, line, active_defines):
                    continue
                if kind == "define":
                    if isinstance(directive, MacroDirective) and directive.argument:
                        was_uncertain = self._line_has_uncertain_macro_context(
                            current_model,
                            line,
                            effective_defines,
                        )
                        new_defines = _extract_macro_defines(
                            f"@#define {directive.argument}\n",
                            allow_complex=True,
                        )
                        new_defines = {
                            name: _substitute_macro_arg(value, effective_defines)
                            for name, value in new_defines.items()
                        }
                        side_defines.update(new_defines)
                        if was_uncertain:
                            end_line = self._macro_side_effect_scope_end(
                                current_model,
                                line,
                            )
                            for name in new_defines:
                                scoped_define_ends[name] = end_line
                    continue
                if kind == "includepath":
                    if not isinstance(directive, MacroDirective):
                        continue
                    was_uncertain = self._line_has_uncertain_macro_context(
                        current_model,
                        line,
                        effective_defines,
                    )
                    original_argument = directive.argument
                    if original_argument is not None:
                        directive = replace(
                            directive,
                            argument=_substitute_macro_arg(
                                original_argument,
                                effective_defines,
                            ),
                        )
                    new_paths = self._includepath_paths(current_key, directive)
                    side_paths = self._append_unique(side_paths, new_paths)
                    if was_uncertain:
                        end_line = self._macro_side_effect_scope_end(
                            current_model,
                            line,
                        )
                        for path in new_paths:
                            scoped_path_ends[path] = end_line
                    continue
                if not isinstance(directive, IncludeDirective):
                    continue
                for include_defines in self._include_define_variants(
                    current_model,
                    line,
                    effective_defines,
                    directive,
                ):
                    include_was_uncertain = self._line_has_uncertain_macro_context(
                        current_model,
                        line,
                        include_defines,
                    )
                    resolved_directive = replace(
                        directive,
                        filename=_substitute_macro_arg(
                            directive.filename,
                            include_defines,
                        ),
                    )
                    local_context = self._directive_context(
                        current_model,
                        resolved_directive,
                    )
                    context = local_context or inherited_context
                    resolved = self._resolve_directive(
                        current_key,
                        resolved_directive.filename,
                        effective_paths,
                    )
                    if resolved is None:
                        continue
                    resolved_key = str(resolved)
                    if resolved_key in stack_keys:
                        continue

                    with self._lock:
                        included_model = self._models.get(resolved_key)
                    if included_model is not None and self._disk_cache_stale(
                        resolved_key,
                        resolved,
                    ):
                        included_model = self._read_and_parse_from_disk(resolved)
                    if included_model is None:
                        included_model = self._read_and_parse_from_disk(resolved)
                    if included_model is None:
                        continue

                    with self._lock:
                        included_source = self._sources.get(resolved_key)
                    if included_source is not None and include_defines:
                        included_model = parse(
                            included_source,
                            initial_macro_defines=include_defines,
                        )
                    anchor_range = (
                        self._assignment_range_in_expansion(
                            root_anchor,
                            resolved_directive.range,
                        )
                        if root_anchor is not None
                        else resolved_directive.range
                    )
                    nested_before_count = len(result)
                    stack_keys.add(resolved_key)
                    nested_paths, nested_defines, included_model = _visit(
                        resolved_key,
                        effective_paths,
                        anchor_range,
                        context,
                        include_defines,
                    )
                    stack_keys.remove(resolved_key)
                    added_nested_paths = [
                        path for path in nested_paths if path not in side_paths
                    ]
                    side_paths = self._append_unique(side_paths, nested_paths)
                    if include_was_uncertain:
                        end_line = self._macro_side_effect_scope_end(
                            current_model,
                            line,
                        )
                        for path in added_nested_paths:
                            scoped_path_ends[path] = end_line
                        for name in nested_defines:
                            scoped_define_ends[name] = end_line
                    side_defines.update(nested_defines)
                    record_stop_line = (
                        self._macro_side_effect_scope_end(current_model, line)
                        if include_was_uncertain
                        else None
                    )
                    _record_line_defines_after(
                        current_line_defines,
                        current_line_count,
                        resolved_directive.range.start.line,
                        side_defines,
                        local_define_lines,
                        record_stop_line,
                    )
                    view = self._contextualize_include(
                        included_model,
                        included_source,
                        context,
                        anchor_range,
                        include_defines,
                    )
                    result_key = resolved_key
                    if resolved_key in result:
                        emitted_counts[resolved_key] = (
                            emitted_counts.get(resolved_key, 1) + 1
                        )
                        result_key = f"{resolved_key}#{emitted_counts[resolved_key]}"
                    else:
                        emitted_counts[resolved_key] = 1
                    if len(result) == nested_before_count:
                        result[result_key] = view
                    else:
                        ordered_items = list(result.items())
                        prefix = ordered_items[:nested_before_count]
                        nested = ordered_items[nested_before_count:]
                        result.clear()
                        result.update(prefix)
                        result.update(nested)
                        result[result_key] = view

            if current_source is not None and current_line_defines:
                current_model = parse(
                    current_source,
                    initial_macro_defines=inherited_defines,
                    line_macro_defines=current_line_defines,
                )
            return side_paths, side_defines, current_model

        _visit(root_key, [], None, None, None)
        with self._lock:
            root_source = self._sources.get(root_key)
            if root_source is not None and root_line_defines:
                self._effective_models[root_key] = parse(
                    root_source,
                    line_macro_defines=root_line_defines,
                )
            else:
                self._effective_models.pop(root_key, None)

        return result

    def find_circular_includes(self, uri: str) -> List[List[str]]:
        """Return cycles in the include graph that touch *uri*.

        Each cycle is a list of absolute path strings; the first element
        of the returned cycle is the file whose include directive closes
        the loop.  Used by the diagnostics engine to emit ``E060`` at
        the first offending directive.
        """
        root_key = _normalize_uri(uri)
        cycles: List[List[str]] = []
        seen_cycles: set = set()

        def _load_cycle_model(
            node: str,
            macro_defines: Optional[Dict[str, str]] = None,
        ) -> Optional[ParsedModel]:
            with self._lock:
                model = self._models.get(node)
                source = self._sources.get(node)
            if model is None:
                node_path = Path(node)
                if not node_path.exists():
                    return None
                model = self._read_and_parse_from_disk(node_path)
                if model is None:
                    return None
                with self._lock:
                    source = self._sources.get(node)
            if macro_defines and source is not None:
                return parse(source, initial_macro_defines=macro_defines)
            return model

        def _dfs(
            node: str,
            include_stack: List[str],
            active_search_paths: List[Path],
            inherited_defines: Optional[Dict[str, str]],
        ) -> Tuple[List[Path], Dict[str, str]]:
            model = _load_cycle_model(node, inherited_defines)
            if model is None:
                return [], {}

            side_paths: List[Path] = []
            side_defines: Dict[str, str] = {}
            scoped_define_ends: Dict[str, int] = {}
            scoped_path_ends: Dict[Path, int] = {}
            for line, _character, kind, directive in self._macro_events(model):
                self._drop_expired_macro_side_effects(
                    line,
                    side_defines,
                    scoped_define_ends,
                    side_paths,
                    scoped_path_ends,
                )
                effective_paths = self._append_unique(active_search_paths, side_paths)
                effective_defines = dict(inherited_defines or {})
                effective_defines.update(side_defines)
                active_defines = self._defines_for_macro_event(
                    effective_defines,
                    directive,
                )
                if not self._line_active(model, line, active_defines):
                    continue
                if kind == "define":
                    if isinstance(directive, MacroDirective) and directive.argument:
                        was_uncertain = self._line_has_uncertain_macro_context(
                            model,
                            line,
                            effective_defines,
                        )
                        new_defines = _extract_macro_defines(
                            f"@#define {directive.argument}\n",
                            allow_complex=True,
                        )
                        new_defines = {
                            name: _substitute_macro_arg(value, effective_defines)
                            for name, value in new_defines.items()
                        }
                        side_defines.update(new_defines)
                        if was_uncertain:
                            end_line = self._macro_side_effect_scope_end(model, line)
                            for name in new_defines:
                                scoped_define_ends[name] = end_line
                    continue
                if kind == "includepath":
                    if not isinstance(directive, MacroDirective):
                        continue
                    was_uncertain = self._line_has_uncertain_macro_context(
                        model,
                        line,
                        effective_defines,
                    )
                    original_argument = directive.argument
                    if original_argument is not None:
                        directive = replace(
                            directive,
                            argument=_substitute_macro_arg(
                                original_argument,
                                effective_defines,
                            ),
                        )
                    new_paths = self._includepath_paths(node, directive)
                    side_paths = self._append_unique(side_paths, new_paths)
                    if was_uncertain:
                        end_line = self._macro_side_effect_scope_end(model, line)
                        for new_path in new_paths:
                            scoped_path_ends[new_path] = end_line
                    continue
                if not isinstance(directive, IncludeDirective):
                    continue
                for include_defines in self._include_define_variants(
                    model,
                    line,
                    effective_defines,
                    directive,
                ):
                    include_was_uncertain = self._line_has_uncertain_macro_context(
                        model,
                        line,
                        include_defines,
                    )
                    resolved_directive = replace(
                        directive,
                        filename=_substitute_macro_arg(
                            directive.filename,
                            include_defines,
                        ),
                    )
                    resolved = self._resolve_directive(
                        node,
                        resolved_directive.filename,
                        effective_paths,
                    )
                    if resolved is None:
                        continue
                    resolved_key = str(resolved)
                    if resolved_key in include_stack:
                        # Closed a cycle.  Record from the point of repetition.
                        idx = include_stack.index(resolved_key)
                        cycle = include_stack[idx:] + [resolved_key]
                        # Canonicalize the cycle's starting point so we don't
                        # emit the same cycle multiple times via different
                        # rotations.
                        rotation_key = tuple(cycle[:-1])
                        min_idx = rotation_key.index(min(rotation_key))
                        canonical = rotation_key[min_idx:] + rotation_key[:min_idx]
                        if canonical in seen_cycles:
                            continue
                        seen_cycles.add(canonical)
                        cycles.append(cycle)
                        continue
                    nested_paths, nested_defines = _dfs(
                        resolved_key,
                        include_stack + [resolved_key],
                        effective_paths,
                        include_defines,
                    )
                    added_nested_paths = [
                        nested_path
                        for nested_path in nested_paths
                        if nested_path not in side_paths
                    ]
                    side_paths = self._append_unique(side_paths, nested_paths)
                    if include_was_uncertain:
                        end_line = self._macro_side_effect_scope_end(model, line)
                        for nested_path in added_nested_paths:
                            scoped_path_ends[nested_path] = end_line
                        for name in nested_defines:
                            scoped_define_ends[name] = end_line
                    side_defines.update(nested_defines)

            return side_paths, side_defines

        _dfs(root_key, [root_key], [], None)
        return cycles

    def find_unresolved_includes(self, uri: str) -> List[IncludeDirective]:
        """Return ``@#include`` directives whose target file isn't on disk.

        Only inspects the directives in *uri*'s own model — unresolved
        includes deeper in the graph surface the next time that file is
        opened.  The returned objects carry both the literal filename and
        the source range, so the diagnostics engine can place an ``E061``
        squiggle exactly on the offending line.
        """
        root_key = _normalize_uri(uri)
        unresolved: List[IncludeDirective] = []

        def _load_model(
            current_key: str,
            macro_defines: Optional[Dict[str, str]] = None,
        ) -> Optional[ParsedModel]:
            with self._lock:
                model = self._models.get(current_key)
                source = self._sources.get(current_key)
            current_path = Path(current_key)
            if model is not None and self._disk_cache_stale(current_key, current_path):
                model = self._read_and_parse_from_disk(current_path)
                with self._lock:
                    source = self._sources.get(current_key)
            if model is None:
                if not current_path.exists():
                    return None
                model = self._read_and_parse_from_disk(current_path)
                if model is None:
                    return None
                with self._lock:
                    source = self._sources.get(current_key)
            if macro_defines and source is not None:
                return parse(source, initial_macro_defines=macro_defines)
            return model

        def _visit(
            current_key: str,
            active_search_paths: List[Path],
            root_directive: Optional[IncludeDirective],
            active_stack: set,
            inherited_defines: Optional[Dict[str, str]],
        ) -> Tuple[List[Path], Dict[str, str]]:
            model = _load_model(current_key, inherited_defines)
            if model is None:
                return [], {}

            side_paths: List[Path] = []
            side_defines: Dict[str, str] = {}
            scoped_define_ends: Dict[str, int] = {}
            scoped_path_ends: Dict[Path, int] = {}
            for line, _character, kind, directive in self._macro_events(model):
                self._drop_expired_macro_side_effects(
                    line,
                    side_defines,
                    scoped_define_ends,
                    side_paths,
                    scoped_path_ends,
                )
                effective_paths = self._append_unique(active_search_paths, side_paths)
                effective_defines = dict(inherited_defines or {})
                effective_defines.update(side_defines)
                active_defines = self._defines_for_macro_event(
                    effective_defines,
                    directive,
                )
                if not self._line_active(model, line, active_defines):
                    continue
                if kind == "define":
                    if isinstance(directive, MacroDirective) and directive.argument:
                        was_uncertain = self._line_has_uncertain_macro_context(
                            model,
                            line,
                            effective_defines,
                        )
                        new_defines = _extract_macro_defines(
                            f"@#define {directive.argument}\n",
                            allow_complex=True,
                        )
                        new_defines = {
                            name: _substitute_macro_arg(value, effective_defines)
                            for name, value in new_defines.items()
                        }
                        side_defines.update(new_defines)
                        if was_uncertain:
                            end_line = self._macro_side_effect_scope_end(model, line)
                            for name in new_defines:
                                scoped_define_ends[name] = end_line
                    continue
                if kind == "includepath":
                    if not isinstance(directive, MacroDirective):
                        continue
                    was_uncertain = self._line_has_uncertain_macro_context(
                        model,
                        line,
                        effective_defines,
                    )
                    original_argument = directive.argument
                    if original_argument is not None:
                        directive = replace(
                            directive,
                            argument=_substitute_macro_arg(
                                original_argument,
                                effective_defines,
                            ),
                        )
                    new_paths = self._includepath_paths(current_key, directive)
                    side_paths = self._append_unique(side_paths, new_paths)
                    if was_uncertain:
                        end_line = self._macro_side_effect_scope_end(model, line)
                        for path in new_paths:
                            scoped_path_ends[path] = end_line
                    continue
                if not isinstance(directive, IncludeDirective):
                    continue
                for include_defines in self._include_define_variants(
                    model,
                    line,
                    effective_defines,
                    directive,
                ):
                    include_was_uncertain = self._line_has_uncertain_macro_context(
                        model,
                        line,
                        include_defines,
                    )
                    resolved_directive = replace(
                        directive,
                        filename=_substitute_macro_arg(
                            directive.filename,
                            include_defines,
                        ),
                    )
                    resolved = self._resolve_directive(
                        current_key,
                        resolved_directive.filename,
                        effective_paths,
                    )
                    if resolved is None:
                        if current_key == root_key or root_directive is None:
                            unresolved.append(resolved_directive)
                        else:
                            unresolved.append(
                                IncludeDirective(
                                    filename=(
                                        f"{resolved_directive.filename} "
                                        f"(included from {Path(current_key).name})"
                                    ),
                                    range=root_directive.range,
                                )
                            )
                        continue
                    resolved_key = str(resolved)
                    if resolved_key in active_stack:
                        continue
                    nested_paths, nested_defines = _visit(
                        resolved_key,
                        effective_paths,
                        root_directive or resolved_directive,
                        active_stack | {resolved_key},
                        include_defines,
                    )
                    added_nested_paths = [
                        path for path in nested_paths if path not in side_paths
                    ]
                    side_paths = self._append_unique(side_paths, nested_paths)
                    if include_was_uncertain:
                        end_line = self._macro_side_effect_scope_end(model, line)
                        for path in added_nested_paths:
                            scoped_path_ends[path] = end_line
                        for name in nested_defines:
                            scoped_define_ends[name] = end_line
                    side_defines.update(nested_defines)

            return side_paths, side_defines

        _visit(root_key, [], None, {root_key}, None)
        return unresolved

    # -- symbol aggregation ----------------------------------------------

    def collect_symbols(self, uri: str) -> Dict[str, List[VarDeclaration]]:
        """Aggregate declarations across the file and its transitive includes.

        Returned dict has keys ``"endogenous"``, ``"exogenous"`` and
        ``"parameters"``.  Used by the diagnostics engine to silence
        ``E020``/``E030`` for names visible only through an include, and
        by the LSP ``workspace/symbol`` handler to surface declarations
        from non-open files.
        """
        symbols: Dict[str, List[VarDeclaration]] = {
            "endogenous": [],
            "exogenous": [],
            "parameters": [],
        }

        included = self.resolve_all_includes(uri)
        for model in included.values():
            symbols["endogenous"].extend(model.endogenous)
            symbols["exogenous"].extend(model.exogenous)
            symbols["parameters"].extend(model.parameters)

        return symbols
