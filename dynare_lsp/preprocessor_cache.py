"""Content-addressed cache for authoritative Dynare preprocessor results."""

from __future__ import annotations

import copy
import functools
import hashlib
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

_MAX_ENTRIES = 32
_cache: "OrderedDict[str, object]" = OrderedDict()
_lock = threading.Lock()
_hits = 0
_misses = 0


def clear_preprocessor_cache() -> None:
    global _hits, _misses
    with _lock:
        _cache.clear()
        _hits = 0
        _misses = 0


def preprocessor_cache_info() -> dict:
    with _lock:
        return {
            "entries": len(_cache),
            "max_entries": _MAX_ENTRIES,
            "hits": _hits,
            "misses": _misses,
        }


_INCLUDE_DIRECTIVE_LINE = re.compile(
    r"^[ \t]*@#[ \t]*include(?![A-Za-z0-9_])", re.IGNORECASE | re.MULTILINE
)


def _active_includepaths(
    module, text: str, working_dir: Optional[str]
) -> Optional[list[str]]:
    """Return literal search paths, or None when dependency resolution is unsafe.

    Dynare opens a relative ``@#includepath`` against its working directory
    (the entry model's folder), never against the declaring file.
    """
    scan = module._strip_non_macro_comments(text)
    _defines, active_lines, _line_defines = module._macro_branch_state(scan)
    scan = module._mask_inactive_macro_lines(scan, active_lines)
    paths = []
    for directive in module._parse_macro_directives(scan):
        if directive.kind != "includepath":
            continue
        argument = (directive.argument or "").strip()
        # Do not turn a macro variable, interpolation, or expression into a
        # literal directory (or silently drop it). Dynare may read dependencies
        # there that this fingerprint cannot see; bypass the cache instead.
        if (
            len(argument) < 2
            or argument[0] not in {"'", '"'}
            or argument[-1] != argument[0]
            or argument[0] in argument[1:-1]
            or "@{" in argument
        ):
            return None
        for raw in module._split_includepath_argument(argument):
            value = raw.strip().strip("'\"")
            if not value:
                continue
            if not os.path.isabs(value):
                if not working_dir:
                    # Relative to an ephemeral temp working directory.
                    return None
                value = os.path.join(working_dir, value)
            paths.append(os.path.abspath(value))
    return paths


def _dynare_candidates(
    include: str, working_dir: Optional[str], search_paths: list[str]
) -> list[str]:
    """Paths Dynare tries for ``@#include include``, in Dynare's order.

    ``macro/Directives.cc``: the name as given (relative to the working
    directory), then ``dir / name`` for each search directory.  The folder
    of the including file is never consulted.
    """
    if os.path.isabs(include):
        return [include]
    candidates = [os.path.join(working_dir, include)] if working_dir else []
    candidates.extend(os.path.join(path, include) for path in search_paths)
    return candidates


def _cache_key(module, text: str, executable: str, source_dir: Optional[str]):
    digest = hashlib.sha256()
    digest.update(b"dynare-lsp-preprocessor-cache-v2\0")
    try:
        stat = os.stat(executable)
        identity = (os.path.realpath(executable), stat.st_mtime_ns, stat.st_size)
    except OSError:
        identity = (os.path.abspath(executable), None, None)
    digest.update(repr(identity).encode("utf-8", "surrogatepass"))
    digest.update(repr(id(module._execute_preprocessor)).encode("ascii"))
    root = os.path.abspath(source_dir) if source_dir else None
    digest.update(repr(root).encode("utf-8", "surrogatepass"))
    digest.update(text.encode("utf-8", "surrogatepass"))
    seen = set()
    # Dynare keeps ONE search-path list for the whole run: an @#includepath
    # inside an included file also applies to every later include.
    search_paths: list[str] = []

    def walk(content: str) -> bool:
        # Branch state is shared across includes in Dynare. A per-file scan
        # cannot prove which conditional dependencies are active, so cache only
        # straight-line literal include graphs until that context is available.
        stripped = module._strip_non_macro_comments(content)
        directives = module._parse_macro_directives(stripped)
        if any(
            directive.kind in {"if", "ifdef", "ifndef", "elseif", "else", "for"}
            for directive in directives
        ):
            return False
        includes = list(module._active_include_paths(content))
        # An include the literal parser skipped (``@#include FILE`` or a
        # string expression) names a file this fingerprint cannot see.
        if len(_INCLUDE_DIRECTIVE_LINE.findall(stripped)) > len(includes):
            return False
        declared_paths = _active_includepaths(module, content, root)
        if declared_paths is None:
            return False
        for path in declared_paths:
            if path not in search_paths:
                search_paths.append(path)
            # Dynare rejects an @#includepath that is not a directory.
            digest.update(b"includepath\0")
            digest.update(repr((path, os.path.isdir(path))).encode("utf-8", "surrogatepass"))
        for raw in includes:
            include = raw.strip().strip("'\"")
            if not include:
                continue
            if "@{" in include:
                return False
            candidate = next(
                (
                    os.path.abspath(path)
                    for path in _dynare_candidates(include, root, search_paths)
                    if os.path.isfile(path)
                ),
                None,
            )
            if candidate is None:
                return False
            canonical = os.path.normcase(os.path.realpath(candidate))
            if canonical in seen:
                continue
            seen.add(canonical)
            try:
                data = Path(candidate).read_bytes()
            except OSError:
                return False
            digest.update(b"include\0")
            digest.update(canonical.encode("utf-8", "surrogatepass"))
            digest.update(hashlib.sha256(data).digest())
            nested = data.decode("utf-8-sig", errors="replace")
            if not walk(nested):
                return False
        return True

    if not walk(text):
        return None
    return digest.hexdigest()


def install(module) -> None:
    if getattr(module, "_content_cache_installed", False):
        return
    module._content_cache_installed = True
    original = module.run_preprocessor

    @functools.wraps(original)
    def cached_run(mod_text, preprocessor_path, timeout=30, source_dir=None, **kwargs):
        use_cache = kwargs.pop("use_cache", True)
        if kwargs:
            return original(
                mod_text,
                preprocessor_path,
                timeout=timeout,
                source_dir=source_dir,
                **kwargs,
            )
        key = (
            _cache_key(module, mod_text.lstrip("\ufeff"), preprocessor_path, source_dir)
            if use_cache
            else None
        )
        global _hits, _misses
        if key is not None:
            with _lock:
                result = _cache.get(key)
                if result is not None:
                    _cache.move_to_end(key)
                    _hits += 1
                    return copy.deepcopy(result)
                _misses += 1
        result = original(
            mod_text,
            preprocessor_path,
            timeout=timeout,
            source_dir=source_dir,
        )
        if key is not None and not any(
            getattr(item, "code", None) == "P000" for item in result.diagnostics
        ):
            with _lock:
                _cache[key] = copy.deepcopy(result)
                _cache.move_to_end(key)
                while len(_cache) > _MAX_ENTRIES:
                    _cache.popitem(last=False)
        return result

    module.run_preprocessor = cached_run
    module.clear_preprocessor_cache = clear_preprocessor_cache
    module.preprocessor_cache_info = preprocessor_cache_info
