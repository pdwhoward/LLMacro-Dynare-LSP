"""Syntactic equation and symbol dependency graph for Dynare models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .parser import ParsedModel

_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_SIMPLE_LHS = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*$")
_BUILTINS = {
    "exp", "log", "ln", "sqrt", "abs", "sin", "cos", "tan", "max", "min",
    "normcdf", "normpdf", "steady_state", "expectation", "erf", "erfc",
}


@dataclass(frozen=True)
class EquationNode:
    identifier: str
    line: int
    text: str
    defines: tuple[str, ...]
    uses: tuple[str, ...]


@dataclass(frozen=True)
class EquationDependencyGraph:
    equations: tuple[EquationNode, ...]
    edges: tuple[tuple[str, str, str], ...]
    symbol_equations: dict[str, tuple[str, ...]]
    unresolved_symbols: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "equations": [
                {
                    "id": node.identifier,
                    "line": node.line,
                    "text": node.text,
                    "defines": list(node.defines),
                    "uses": list(node.uses),
                }
                for node in self.equations
            ],
            "edges": [
                {"from": source, "to": target, "symbol": symbol}
                for source, target, symbol in self.edges
            ],
            "symbol_equations": {
                symbol: list(equations)
                for symbol, equations in self.symbol_equations.items()
            },
            "unresolved_symbols": list(self.unresolved_symbols),
        }

    def to_markdown(self) -> str:
        lines = ["# Equation dependency graph", ""]
        lines.append(
            f"{len(self.equations)} equations · {len(self.edges)} syntactic edges"
        )
        for node in self.equations:
            label = f"L{node.line} `{node.identifier}`"
            defines = ", ".join(node.defines) or "none"
            uses = ", ".join(node.uses) or "none"
            lines.extend(["", f"## {label}", f"- Defines: {defines}", f"- Uses: {uses}"])
        if self.edges:
            lines.extend(["", "## Edges"])
            lines.extend(
                f"- `{source}` → `{target}` via `{symbol}`"
                for source, target, symbol in self.edges
            )
        if self.unresolved_symbols:
            lines.extend(
                [
                    "",
                    "## Referenced without a syntactic provider",
                    "- " + ", ".join(self.unresolved_symbols),
                ]
            )
        lines.append("")
        return "\n".join(lines)


def build_equation_dependency_graph(model: ParsedModel) -> EquationDependencyGraph:
    declared = model.all_declared_names()
    nodes = []
    providers = {}
    symbol_equations = {}

    for equation in model.model_equations:
        stripped = equation.text.strip()
        local = re.match(
            r"#\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", stripped, re.DOTALL
        )
        if local:
            declared.add(local.group(1))

    for equation_number, equation in enumerate(model.model_equations, start=1):
        stripped = equation.text.strip()
        local = re.match(
            r"#\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", stripped, re.DOTALL
        )
        name = equation.name.strip() if equation.name else ""
        identifier = name or (
            f"local_{local.group(1)}" if local else f"equation_{equation_number}"
        )
        defines = []
        expression = stripped
        if local:
            defines = [local.group(1)]
            expression = local.group(2)
        elif "=" in stripped:
            lhs, rhs = stripped.split("=", 1)
            lhs_match = _SIMPLE_LHS.match(lhs)
            if lhs_match and lhs_match.group(1) in declared:
                defines = [lhs_match.group(1)]
            expression = rhs
        uses = sorted(
            {
                symbol
                for symbol in _IDENTIFIER.findall(expression)
                if symbol in declared
                and symbol not in defines
                and symbol.lower() not in _BUILTINS
            }
        )
        node = EquationNode(
            identifier=identifier,
            line=equation.range.start.line + 1,
            text=stripped,
            defines=tuple(defines),
            uses=tuple(uses),
        )
        nodes.append(node)
        for symbol in defines:
            providers.setdefault(symbol, identifier)
        for symbol in set(defines) | set(uses):
            symbol_equations.setdefault(symbol, []).append(identifier)

    edges = set()
    unresolved = set()
    for node in nodes:
        for symbol in node.uses:
            provider = providers.get(symbol)
            if provider is None:
                unresolved.add(symbol)
            elif provider != node.identifier:
                edges.add((provider, node.identifier, symbol))

    return EquationDependencyGraph(
        equations=tuple(nodes),
        edges=tuple(sorted(edges)),
        symbol_equations={
            symbol: tuple(equations)
            for symbol, equations in sorted(symbol_equations.items())
        },
        unresolved_symbols=tuple(sorted(unresolved)),
    )


def install_server(core) -> None:
    if getattr(core, "_equation_dependencies_installed", False):
        return
    core._equation_dependencies_installed = True

    @core.server.command("dynare/equationDependencies")
    def equation_dependencies_command(*args):
        values = args[0] if len(args) == 1 and isinstance(args[0], list) else args
        if len(values) == 1 and isinstance(values[0], dict):
            uri = values[0].get("uri")
        else:
            uri = values[0] if values else None
        if not isinstance(uri, str):
            return {"error": "A document URI is required", "code": "BAD_ARGS"}
        model = core._model_for_uri(uri)
        if model is None:
            return {"error": f"No parsed model for {uri}", "code": "NOT_FOUND"}
        try:
            include_models = list(
                core._workspace_index.resolve_all_includes(uri).values()
            )
            model = core.model_with_include_context(model, include_models)
        except Exception:
            pass
        return build_equation_dependency_graph(model).to_dict()

    core.equation_dependencies_command = equation_dependencies_command


def install_mcp(module) -> None:
    if getattr(module, "_equation_dependencies_installed", False):
        return
    module._equation_dependencies_installed = True
    original_build = module.build_server

    def build_server():
        mcp = original_build()

        @mcp.tool()
        def dynare_equation_dependencies(
            file_content: str,
            active_file: str | None = None,
            files: dict[str, str] | None = None,
        ) -> dict:
            from .diagnostics import model_with_include_context
            from .parser import parse
            from .workspace import WorkspaceIndex

            if not files:
                return build_equation_dependency_graph(parse(file_content)).to_dict()
            active = active_file or next(iter(files), "model.mod")
            workspace_files = dict(files)
            workspace_files.setdefault(active, file_content)
            index = WorkspaceIndex()
            for filename, content in workspace_files.items():
                index.update_document(filename, content)
            model = index.get_effective_model(active) or index.get_model(active)
            if model is None:
                model = parse(file_content)
            includes = list(index.resolve_all_includes(active).values())
            model = index.get_effective_model(active) or model
            return build_equation_dependency_graph(
                model_with_include_context(model, includes)
            ).to_dict()

        return mcp

    module.build_server = build_server
