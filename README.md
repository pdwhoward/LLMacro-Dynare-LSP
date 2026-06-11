# Dynare LSP

A Language Server Protocol implementation and Model Context Protocol (MCP)
server for the [Dynare](https://www.dynare.org/) modeling language. It gives
you live diagnostics, steady-state solving,
Blanchard–Kahn checks, and model intelligence in your editor or in an AI
coding agent.

Structural parsing and validation defer to a bundled Dynare **7.1
preprocessor**, so what the language server accepts matches what Dynare itself
accepts. On top of that it adds a clean-room Blanchard–Kahn rank check, an
automatic steady-state solver, per-equation residuals, and a large family of
model, estimation, policy, shocks, and usage diagnostics.

You can use it three ways:

- **In VS Code**, via the bundled extension (`.vsix`).
- **In Claude Code** (or any MCP/LSP-capable agent), via the bundled plugin.
- **From the command line**, for one-shot `--check` runs.

---

## Working Paper

This repository accompanies the working paper **"LLMacro: A Language Server for
Dynare - Structured Context for AI-Assisted Macroeconomic Modeling"**.

Authors:

- Anthony Diercks, Federal Reserve Board
- Philip Howard, Wake Forest University
- Mehrdad Samadi, Rutgers University

Suggested citation:

> Diercks, Anthony, Philip Howard, and Mehrdad Samadi. 2026. "LLMacro: A
> Language Server for Dynare." Working paper.

## Requirements

- **Python 3.8+** for the core language server (**3.10+ for the MCP
  server** — every release of the `mcp` package requires Python 3.10)
- Optional, for the steady-state solver and identification checks:
  `numpy`, `scipy`, `sympy`
- Optional, for the MCP server: the `mcp` package
- Optional, for the "run model in MATLAB" tool: a local **MATLAB + Dynare**
  installation

A Windows Dynare 7.1 preprocessor (`dynare-preprocessor.exe`) is bundled under
`dynare_lsp/bin/`. On other platforms, install Dynare locally and the server
will discover its preprocessor.

## Install

Clone the repository and install the package (editable is convenient):

```bash
git clone https://github.com/pdwhoward/LLMacro-Dynare-LSP.git
cd LLMacro-Dynare-LSP
pip install -e ".[all]"
```

`".[all]"` pulls in the solver dependencies and the MCP server (and therefore
needs Python 3.10+). On Python 3.8/3.9 use `pip install -e ".[solver]"` for
the solver without MCP, or a plain `pip install -e .` for just the core
language server.

Two console scripts are installed:

- `dynare-lsp` — the language server / CLI (equivalent to `python -m dynare_lsp`)
- `dynare-mcp` — the MCP server (equivalent to `python -m dynare_lsp.mcp_server`)

## Command-line use

```bash
# Diagnostics on a single file
python -m dynare_lsp --check model.mod

# Diagnostics plus a computed steady state and Blanchard-Kahn check
python -m dynare_lsp --check --solve model.mod

# Documentation for a diagnostic code
python -m dynare_lsp --explain W071
python -m dynare_lsp --explain --list
```

`--check` exits non-zero when the model has errors, so it composes with CI.

## VS Code

Install the bundled extension from `vscode-dynare/`:

1. In VS Code, open the Command Palette → **Extensions: Install from VSIX…**
2. Select `vscode-dynare/dynare-lsp-0.3.1.vsix`.

The extension launches the Python language server, so make sure the
`dynare_lsp` package is installed in the Python environment VS Code uses
(`pip install -e .` as above). Open any `.mod` or `.inc` file to get
diagnostics, hover, and navigation.

## Claude Code

The repository ships a Claude Code plugin under `claude-code-plugin/` that
registers both the language server and the MCP server. Its
`plugins/dynare-lsp/.claude-plugin/plugin.json` wires up:

```jsonc
{
  "lspServers": {
    "dynare": { "command": "python", "args": ["-m", "dynare_lsp"],
                "extensionToLanguage": { ".mod": "dynare", ".inc": "dynare" } }
  },
  "mcpServers": {
    "dynare": { "command": "python", "args": ["-m", "dynare_lsp.mcp_server"] }
  }
}
```

Add `claude-code-plugin/` as a local plugin marketplace in Claude Code, then
install the `dynare-lsp` plugin. Install the Python package with the MCP extra
first so both `python -m dynare_lsp` and
`python -m dynare_lsp.mcp_server` resolve in your environment.

## MCP server (standalone)

To use the MCP server with any MCP client, run:

```bash
python -m dynare_lsp.mcp_server
```

It exposes tools for diagnostics, preprocessor checks, steady-state solving,
and — when MATLAB + Dynare are available — running a model end to end.

## License

Provided as-is for research and educational use. The bundled Dynare
preprocessor is distributed under Dynare's own license; see
[dynare.org](https://www.dynare.org/).
