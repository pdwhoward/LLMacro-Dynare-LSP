"""Path resolution for Dynare ``@#include`` directives.

The Dynare macro preprocessor expands ``@#include`` directives at build
time by reading another file from disk.  When the language server analyzes
a project across multiple files, it has to make the same resolution
decision the preprocessor would make so that identifiers declared in
included files are visible at the include site.

Two helpers are exposed here:

* :func:`resolve_include_path` — given the literal filename from inside
  the directive plus the URI of the including file, return the absolute
  :class:`pathlib.Path` of the file on disk, or ``None`` when no candidate
  matches.  Resolution mimics the Dynare preprocessor:

    1. relative to the directory of the including file
    2. each entry in *search_paths*, in order

* :func:`find_workspace_root` — walks up from a starting path until it
  finds a directory containing ``.git`` (or runs out of parents).  Used
  as the default workspace root when the LSP client did not supply one.

Both helpers are pure functions: they do not touch any global state and
are safe to call from multiple threads.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
from urllib.parse import unquote, urlparse


def _uri_to_path(uri_or_path: str) -> Path:
    """Coerce either a ``file://`` URI or a plain path string to :class:`Path`.

    Windows paths inside ``file://`` URIs use the form ``file:///C:/foo``
    where ``urlparse`` puts the drive letter in ``path`` with a leading
    slash, so we strip that.  Forward and backward slashes are normalized
    by :class:`Path` itself.
    """
    if uri_or_path.startswith("file://"):
        parsed = urlparse(uri_or_path)
        path_str = unquote(parsed.path)
        if parsed.netloc and parsed.netloc.lower() != "localhost":
            return Path(f"//{parsed.netloc}{path_str}")
        # On Windows, urlparse leaves a leading "/" before drive letter
        if len(path_str) >= 3 and path_str[0] == "/" and path_str[2] == ":":
            path_str = path_str[1:]
        return Path(path_str)
    return Path(uri_or_path)


def _normalize_separators(filename: str) -> str:
    """Normalize a directive filename's path separators for the host OS.

    Dynare model files written on Linux/macOS use forward slashes; files
    written on Windows may use backslashes.  :class:`Path` is happy with
    either separator on Windows but rejects backslashes on POSIX, so we
    rewrite backslashes to forward slashes — :class:`Path` then handles
    both uniformly.
    """
    return filename.replace("\\", "/")


def resolve_include_path(
    directive_filename: str,
    including_file_uri_or_path: str,
    search_paths: Optional[List[Path]] = None,
    known_paths: Optional[set] = None,
) -> Optional[Path]:
    """Resolve the absolute path of a ``@#include`` target.

    Tries the directory of *including_file_uri_or_path* first, then each
    entry in *search_paths*.  A candidate is considered resolved if it
    is a regular file on disk OR if its normalized absolute path appears
    in *known_paths* — the latter lets the workspace index pass virtual
    in-memory files (e.g. files supplied to the MCP server as a
    ``{filename: content}`` dict but not present on disk) through the
    same resolution flow.  Returns the absolute :class:`Path` of the
    first match, or ``None`` if no candidate matched.

    *directive_filename* is the string captured from inside the directive,
    exactly as written (with or without quotes already stripped by the
    parser).  Backslash separators are normalized so that Linux-authored
    paths work on Windows and vice versa.

    *known_paths* should contain strings produced by
    :func:`dynare_lsp.workspace._normalize_uri` so the same normalisation
    rule applies on both sides of the comparison.
    """
    if not directive_filename:
        return None

    known = known_paths or set()

    def _matches(p: Path) -> bool:
        # A directory with the right name is NOT a Dynare include target,
        # so reject it explicitly; ``Path.exists()`` would otherwise
        # accept a folder called e.g. ``helper.mod``.
        if p.is_file():
            return True
        try:
            return str(p.resolve()) in known
        except (OSError, RuntimeError):
            return False

    def _matches_known(p: Path) -> bool:
        try:
            return str(p.resolve()) in known
        except (OSError, RuntimeError):
            return str(p) in known

    def _parts_equal(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
        if os.name == "nt":
            return tuple(part.casefold() for part in left) == tuple(
                part.casefold() for part in right
            )
        return left == right

    def _resolve_known_path(p: Path) -> Path:
        try:
            return p.resolve()
        except (OSError, RuntimeError):
            return p.absolute()

    def _match_unique_known_relative_path(p: Path) -> Optional[Path]:
        target_parts = p.parts
        matches: List[Path] = []
        seen: set[str] = set()
        for entry in known:
            known_path = _uri_to_path(entry)
            if known_path.is_file():
                continue
            known_parts = known_path.parts
            if len(known_parts) < len(target_parts):
                continue
            if not _parts_equal(known_parts[-len(target_parts):], target_parts):
                continue
            resolved = _resolve_known_path(known_path)
            key = str(resolved)
            if key not in seen:
                matches.append(resolved)
                seen.add(key)
        if len(matches) == 1:
            return matches[0]
        return None

    normalized = _normalize_separators(directive_filename)
    candidate_rel = Path(normalized)

    # If the directive contains an absolute path, just check it directly.
    if candidate_rel.is_absolute():
        if _matches(candidate_rel):
            try:
                return candidate_rel.resolve()
            except (OSError, RuntimeError):
                return candidate_rel
        return None

    including_path = _uri_to_path(including_file_uri_or_path)
    # If the URI points to a file, use its parent; otherwise use it as-is.
    if including_path.is_file() or including_path.suffix:
        including_dir = including_path.parent
    else:
        including_dir = including_path

    # 1. Relative to the including file's directory.
    candidate = (including_dir / candidate_rel)
    if _matches(candidate):
        try:
            return candidate.resolve()
        except (OSError, RuntimeError):
            return candidate

    # 2. Each search path, in order.
    for sp in search_paths or []:
        candidate = sp / candidate_rel
        if _matches(candidate):
            try:
                return candidate.resolve()
            except (OSError, RuntimeError):
                return candidate

    # Virtual in-memory workspaces (notably MCP callers) may provide a
    # bare ``helper.mod`` file alongside a nested active file such as
    # ``models/main.mod``.  Keep this convenience fallback after the
    # explicit Dynare-style search paths so configured include directories
    # still win over unrelated virtual same-basename files.
    if _matches_known(candidate_rel):
        try:
            return candidate_rel.resolve()
        except (OSError, RuntimeError):
            return candidate_rel
    known_match = _match_unique_known_relative_path(candidate_rel)
    if known_match is not None:
        return known_match

    return None


def find_workspace_root(start_path: Path) -> Path:
    """Walk up from *start_path* to the nearest directory containing ``.git``.

    Used as the default workspace root when the LSP client has not
    supplied one explicitly.  If no ``.git`` directory is found up the
    chain, the filesystem root is returned (so callers always get a
    valid directory back).
    """
    current = start_path.resolve()
    if current.is_file():
        current = current.parent

    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            # Reached the filesystem root.
            return current
        current = parent
