# dynare-lsp (Claude Code plugin)

Registers the LLMacro Dynare language server with Claude Code so `.mod` and `.inc`
files get diagnostics and code intelligence, and registers the analysis-only
Dynare MCP server.

## Prerequisite

The `python` on your PATH must be able to import the language server and MCP
runtime. Run pip from the directory that contains the package's `setup.py`
(the root of the `LLMacro-Dynare-LSP` repository):

```bash
pip install -e ".[mcp]"                          # requires Python 3.11+; ".[all]" adds the solver
python -m dynare_lsp --check path/to/model.mod   # exits 0 when the model has no errors
```

`--check` prints each diagnostic as `file:line:col: SEVERITY [CODE] message`
and a summary line, and exits non-zero only when there are errors. A model
with no diagnostics at all prints `No issues found in <file>`. Informational
notes can still appear on a valid model; for example, the bundled
`dynare_lsp/tests/fixtures/public_stable.mod` prints:

```text
dynare_lsp/tests/fixtures/public_stable.mod:3:1: INFO [I050] No steady state values provided. Use the 'Compute Steady State' code action to solve automatically.

1 issue(s): 0 error(s), 0 warning(s)
```

## macOS and Linux: `python` vs `python3`

The plugin manifest (`.claude-plugin/plugin.json`) launches both servers with
the command `python` on every platform (the manifest does not select a
command per operating system), so on systems where only `python3` exists (common on
macOS and Linux), use one of:

- Launch Claude Code from an activated virtual environment where
  `dynare-lsp[mcp]` is installed; inside a venv, `python` is the venv
  interpreter.
- Edit both `"command": "python"` entries in `.claude-plugin/plugin.json` to
  `python3` or an absolute interpreter path (for example
  `/home/me/.venvs/dynare/bin/python`), then reinstall or update the plugin.

## Install

Inside Claude Code, from the repository root, add the marketplace directory
that contains this plugin:

```text
/plugin marketplace add ./claude-code-plugin
/plugin install dynare-lsp@llmacro-local
```

Then restart Claude Code so the language server attaches.

## Troubleshooting

- Run `python -c "import dynare_lsp, dynare_lsp.mcp_preflight_server"` (or
  `python3 -c ...`) in the same shell that launches Claude Code.
- Confirm the plugin is enabled with `/plugin` and restart Claude Code after
  installing or updating it.
- Confirm `.mod` and `.inc` files are recognized as Dynare files. The plugin
  maps both extensions to the `dynare` language server.
- If multiple Python installations are present, launch Claude Code from the
  environment where `dynare-lsp[mcp]` is installed.
