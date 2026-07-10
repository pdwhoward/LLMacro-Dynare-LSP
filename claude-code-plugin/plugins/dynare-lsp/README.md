# dynare-lsp (Claude Code plugin)

Registers the LLMacro Dynare language server with Claude Code so `.mod` and `.inc`
files get diagnostics and code intelligence.

## Prerequisite

The `python` on your PATH must be able to import the language server and MCP
runtime. Run pip from the directory that contains the package's `setup.py` —
the repository root in the public `LLMacro-Dynare-LSP` repo, or `dynare-lsp/`
in the LLMacro development monorepo:

```bash
pip install -e ".[mcp]"                       # requires Python 3.11+
python -m dynare_lsp --check path/to/model.mod   # "No issues found" on a valid model
```

## Install

Inside Claude Code, add the marketplace directory that contains this plugin —
`./claude-code-plugin` in the public repo, or `./dynare-lsp/claude-code-plugin`
in the development monorepo:

```text
/plugin marketplace add ./claude-code-plugin
/plugin install dynare-lsp@llmacro-local
```

Then restart Claude Code so the language server attaches.

## Troubleshooting

- Run `python -c "import dynare_lsp, dynare_lsp.mcp_server"` in the same shell
  that launches Claude Code.
- Confirm the plugin is enabled with `/plugin` and restart Claude Code after
  installing or updating it.
- Confirm `.mod` and `.inc` files are recognized as Dynare files. The plugin
  maps both extensions to the `dynare` language server.
- If multiple Python installations are present, launch Claude Code from the
  environment where `dynare-lsp[mcp]` is installed.
