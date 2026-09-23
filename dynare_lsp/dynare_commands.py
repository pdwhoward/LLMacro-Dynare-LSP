"""Catalog of Dynare commands and their options for option-aware completion.

The command -> option structure and the option descriptions are **generated**
from the Dynare 7.1 grammar (``DynareBison.yy``), lexer (``DynareFlex.ll``), and
reference manual by ``tools/generate_catalog.py`` into :mod:`dynare_catalog`.
This module is the stable public accessor used by the server for option-aware
completion and option hover; validation of options is still deferred to the
bundled preprocessor (the authoritative parser).

Regenerate after a Dynare version bump:

    cd dynare-lsp
    python -m tools.generate_catalog \\
        --dynare-src ../dynare-7.1/dynare-7.1-src \\
        --out dynare_lsp/dynare_catalog.py
"""

from __future__ import annotations

from typing import List, Tuple

from .dynare_catalog import COMMAND_OPTIONS, OPTION_DOCS

__all__ = ["COMMAND_OPTIONS", "OPTION_DOCS", "command_options", "option_doc"]


def command_options(command: str) -> List[Tuple[str, str]]:
    """Return the ``(option, description)`` list for *command* (empty if unknown)."""
    return COMMAND_OPTIONS.get(command.lower(), [])


def option_doc(option: str) -> str:
    """Return the one-line manual description for an option name (empty if none)."""
    return OPTION_DOCS.get(option, "")
