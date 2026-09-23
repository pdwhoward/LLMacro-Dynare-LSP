# Dynare Language Support for VS Code

Language support for Dynare `.mod` and `.inc` files, backed by the
[Dynare LSP](https://github.com/pdwhoward/LLMacro-Dynare-LSP) Python language
server: diagnostics from the bundled Dynare 7.1 preprocessor, steady-state and
Blanchard-Kahn checks, hover, navigation, formatting, and a Dynare MCP server
definition for VS Code agents.

## Requirements

The extension launches a Python language server. Install it, with its `all`
extra, into a Python 3.11+ environment:

```bash
pip install "dynare-lsp[all] @ git+https://github.com/pdwhoward/LLMacro-Dynare-LSP.git"
# or, from a clone of the repository:
pip install -e ".[all]"
```

The `all` extra adds the steady-state solver dependencies (`numpy`, `scipy`,
`sympy`) and the `mcp` package used by the bundled Dynare MCP server
definition.

## Settings

- `dynare.pythonPath` (default `python`): the interpreter that has
  `dynare-lsp` installed. On macOS and Linux this is often `python3` or the
  full path to a virtual-environment interpreter (for example
  `/home/me/.venvs/dynare/bin/python`). The setting is machine-overridable, so
  a remote machine or a trusted workspace can set its own interpreter.
- `dynare.preprocessorPath`: explicit path to a `dynare-preprocessor` binary.
  Leave empty for auto-detection.
- `dynare.searchPaths`: additional `@#include` search directories.
- `dynare.steadyStateTolerance`, `dynare.formatIndent`.

If the server cannot start, the extension reports whether the interpreter was
missing or could not import `dynare_lsp`, with a shortcut to the
`dynare.pythonPath` setting. Server logs appear in the **Dynare Language
Server** output channel.

## Workspace trust

The extension runs a Python interpreter and the Dynare preprocessor on
workspace files, so it is disabled in Restricted Mode. Trust the workspace to
enable it.

## License

GPL-3.0-or-later. See the repository for the bundled Dynare preprocessor
notice.
