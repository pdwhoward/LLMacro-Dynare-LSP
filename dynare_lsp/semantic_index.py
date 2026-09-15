"""Shared, cached semantic tables for repeated LSP feature queries."""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import Mapping

from .parser import ParsedModel, Position, SourceRange, VarDeclaration

_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_BUILTINS = {
    "exp",
    "log",
    "ln",
    "sqrt",
    "abs",
    "sin",
    "cos",
    "tan",
    "max",
    "min",
    "normcdf",
    "normpdf",
    "steady_state",
    "expectation",
    "erf",
    "erfc",
}


@dataclass(frozen=True)
class SemanticIndex:
    declarations: Mapping[str, VarDeclaration]
    kinds: Mapping[str, str]
    endogenous: frozenset[str]
    exogenous: frozenset[str]
    parameters: frozenset[str]
    model_locals: Mapping[str, str]
    model_local_declarations: Mapping[str, VarDeclaration]
    equation_symbols: tuple[frozenset[str], ...]
    symbol_equations: Mapping[str, tuple[int, ...]]
    timing: Mapping[str, dict]


def _offset(text: str, position: Position) -> int:
    lines = text.splitlines(keepends=True)
    return sum(len(line) for line in lines[: position.line]) + position.character


def _position(text: str, offset: int) -> Position:
    line = text.count("\n", 0, offset)
    last = text.rfind("\n", 0, offset)
    return Position(line, offset if last < 0 else offset - last - 1)


def _local_declarations(model: ParsedModel):
    values = {}
    declarations = {}
    for equation in model.model_equations:
        match = re.match(
            r"\s*#\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$",
            equation.text,
            re.DOTALL,
        )
        if match is None:
            continue
        name = match.group(1)
        values[name] = match.group(2).strip()
        start_offset = _offset(model.text, equation.range.start)
        raw_end = _offset(model.text, equation.range.end)
        raw = model.text[start_offset:raw_end]
        raw_match = re.search(rf"#\s*({re.escape(name)})\s*=", raw)
        if raw_match is None:
            declarations[name] = VarDeclaration(name=name, range=equation.range)
            continue
        start = _position(model.text, start_offset + raw_match.start(1))
        end = _position(model.text, start_offset + raw_match.end(1))
        declarations[name] = VarDeclaration(
            name=name,
            range=SourceRange(start, end),
        )
    return values, declarations


def build_semantic_index(model: ParsedModel) -> SemanticIndex:
    cached = getattr(model, "_semantic_index_cache", None)
    if isinstance(cached, SemanticIndex):
        return cached

    declarations = {}
    kinds = {}
    groups = (
        ("endogenous", model.endogenous),
        ("exogenous", model.exogenous),
        ("parameter", model.parameters),
    )
    for kind, items in groups:
        for declaration in items:
            declarations.setdefault(declaration.name, declaration)
            kinds.setdefault(declaration.name, kind)
    locals_by_name, local_declarations = _local_declarations(model)
    for name, declaration in local_declarations.items():
        declarations.setdefault(name, declaration)
        kinds.setdefault(name, "model_local")

    declared = set(kinds)
    equation_symbols = []
    reverse = {}
    for index, equation in enumerate(model.model_equations):
        symbols = frozenset(
            name
            for name in _IDENTIFIER.findall(equation.text)
            if name in declared and name.lower() not in _BUILTINS
        )
        equation_symbols.append(symbols)
        for name in symbols:
            reverse.setdefault(name, []).append(index)

    from .model_info import classify_variable_timing

    result = SemanticIndex(
        declarations=declarations,
        kinds=kinds,
        endogenous=frozenset(item.name for item in model.endogenous),
        exogenous=frozenset(item.name for item in model.exogenous),
        parameters=frozenset(item.name for item in model.parameters),
        model_locals=locals_by_name,
        model_local_declarations=local_declarations,
        equation_symbols=tuple(equation_symbols),
        symbol_equations={name: tuple(indices) for name, indices in reverse.items()},
        timing=classify_variable_timing(model),
    )
    setattr(model, "_semantic_index_cache", result)
    return result


def install(core) -> None:
    if getattr(core, "_semantic_index_installed", False):
        return
    core._semantic_index_installed = True

    def model_local_variables(model):
        return dict(build_semantic_index(model).model_locals)

    def find_model_local_declaration(model, name):
        return build_semantic_index(model).model_local_declarations.get(name)

    def find_non_model_local_declaration(model, name):
        index = build_semantic_index(model)
        declaration = index.declarations.get(name)
        return declaration if index.kinds.get(name) != "model_local" else None

    core._model_local_variables = model_local_variables
    core._find_model_local_declaration = find_model_local_declaration
    core._find_non_model_local_declaration = find_non_model_local_declaration

    original_tokens = core._semantic_token_tuples

    @functools.wraps(original_tokens)
    def semantic_token_tuples(uri):
        with core._state_lock:
            model = core._document_models.get(uri)
        if model is not None:
            build_semantic_index(model)
        return original_tokens(uri)

    core._semantic_token_tuples = semantic_token_tuples
