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

- **Python 3.11+**
- **pygls 2.x** and **lsprotocol 2025+**, installed automatically. pygls 1.x is not supported.
- **VS Code 1.101+** when using the bundled extension
- Optional, for the steady-state solver and identification checks:
  `numpy`, `scipy`, `sympy`
- Optional, for the MCP server: the `mcp` package
- Optional, for the "run model in MATLAB" tool: a local **MATLAB + Dynare**
  installation

A Windows Dynare 7.1 preprocessor (`dynare-preprocessor.exe`) is bundled under
`dynare_lsp/bin/`. On other platforms, install Dynare locally and the server
will discover its preprocessor. See [Bundled Dynare preprocessor](#bundled-dynare-preprocessor)
for its license and source.

On macOS and Linux the interpreter is usually `python3`, and `python` may be
missing or point to another installation. Substitute `python3` (or the full
path to a virtual-environment interpreter) wherever this README, the VS Code
extension, or the Claude Code plugin says `python`.

## Install

Clone the repository and install the package (editable is convenient):

```bash
git clone https://github.com/pdwhoward/LLMacro-Dynare-LSP.git
cd LLMacro-Dynare-LSP
pip install -e ".[all]"
```

`".[all]"` pulls in the solver dependencies and the MCP server. Use
`pip install -e ".[solver]"` for the solver without MCP, or a plain
`pip install -e .` for just the core language server.

Two console scripts are installed:

- `dynare-lsp` — the language server / CLI (equivalent to `python -m dynare_lsp`)
- `dynare-mcp` — the analysis-only MCP server with consolidated preflight (equivalent to
  `python -m dynare_lsp.mcp_preflight_server`)

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
With `--solve`, a failed steady-state solve, failed/skipped/unavailable
Blanchard-Kahn check, reported numerical error, or failed residual check also
exits non-zero. Ordinary warning diagnostics remain non-blocking.

## VS Code

Install the bundled extension from `vscode-dynare/`:

1. In VS Code, open the Command Palette → **Extensions: Install from VSIX…**
2. Select `vscode-dynare/dynare-lsp-0.5.0.vsix`.

The extension launches the Python language server and registers the Dynare
MCP server for VS Code agents, so install the package with the `all` extra
(solver plus MCP) in the Python environment VS Code uses:

```bash
pip install -e ".[all]"
```

Then point the `dynare.pythonPath` setting at that interpreter if it is not
the `python` on `PATH` (on macOS/Linux typically `python3` or a venv path
such as `/home/me/.venvs/dynare/bin/python`). If the server cannot start, the
extension reports whether the interpreter was missing or could not import
`dynare_lsp`. Open any `.mod` or `.inc` file to get diagnostics, hover, and
navigation. The extension is disabled in untrusted (Restricted Mode)
workspaces because it runs Python and the Dynare preprocessor on workspace
files.

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
    "dynare": { "command": "python", "args": ["-m", "dynare_lsp.mcp_preflight_server"] }
  }
}
```

Add `claude-code-plugin/` as a local plugin marketplace in Claude Code, then
install the `dynare-lsp` plugin. Install the Python package with the MCP extra
first so both `python -m dynare_lsp` and
`python -m dynare_lsp.mcp_preflight_server` resolve in your environment.

The plugin manifest launches the command `python` on every platform (the
manifest does not select a command per operating system). On macOS/Linux, either launch
Claude Code from an activated virtual environment (where `python` is the venv
interpreter), or edit both `"command": "python"` entries in
`claude-code-plugin/plugins/dynare-lsp/.claude-plugin/plugin.json` to
`python3` or an absolute interpreter path, then reinstall the plugin.

## Codex

Codex can use the analysis engine and consolidated preflight through the default
MCP server. After installing the package with the MCP extra, add this to
`~/.codex/config.toml` (or to a trusted project's `.codex/config.toml`):

```toml
[mcp_servers.dynare]
command = "python"
args = ["-m", "dynare_lsp.mcp_preflight_server"]
```

Alternatively, register it from the command line:

```bash
codex mcp add dynare -- python -m dynare_lsp.mcp_preflight_server
codex mcp list
```

Pin `command` to the full interpreter path when `python` on `PATH` is not the
environment where `dynare-lsp[mcp]` is installed.

## MCP execution policy and migration

The default console script, VS Code provider, and Claude Code plugin launch the
bounded **analysis** profile. It includes `dynare_preflight`, the versioned
analysis tools, and `dynare_profile_policy`; it does not register MATLAB
execution. Requests are limited to 8 MiB by default.

Preflight now uses the versioned snapshot analysis service. Supply every
included file in `files`; relative keys are anchored to an absolute
`active_file`'s directory, and missing buffers do not fall back to disk.
Inspect `requested_checks`, individual stage statuses, and the snapshot
manifest. If no preprocessor is installed, its stage is `unavailable` and it
is not included in `requested_checks`. If one is detected, it must pass;
failed, unsupported, and unavailable requested stages cannot produce a pass.
No numerical result guarantees a full Dynare execution or economic fidelity.

Full MATLAB execution requires an explicit launch configuration:

```bash
dynare-mcp --profile execution --max-execution-seconds 300
# Equivalent:
python -m dynare_lsp.mcp_preflight_server --profile execution
```

Use `dynare_execute_trusted_model` with
`acknowledge_code_execution=true` on **each call**, and a timeout no greater
than the configured limit. This executes arbitrary trusted model/MATLAB code
with the account's permissions; neither temporary staging nor the profile is
an operating-system sandbox. Do not use it on untrusted models.

The unrestricted legacy granular catalog remains explicitly available through
`python -m dynare_lsp.mcp_server` for compatibility. It includes
`dynare_run_dynare` without the profile's acknowledgment and request limits;
it is **not** the default and should not be treated as an analysis-only server.

## Public regression tests

The release includes a curated, self-contained regression suite and small
fixture, not the private oracle corpus or benchmark data:

```bash
python -m pip install -e ".[dev]"
python -B -m pytest dynare_lsp/tests -q -p no:cacheprovider
```

Tests exercise validation races, numerical failure states, supplied-versus-disk
include precedence, CLI exits, MCP policy, and an LSP initialize/open/diagnostic
exchange. They do not run MATLAB, benchmarks, or result-generation scripts.

`.github/workflows/public-ci.yml` tests the minimum and current supported
transport dependencies. **CI is opt-in, not pre-provisioned:** a maintainer
must register a dedicated runner to this public repository with labels
`self-hosted`, `Windows`, `X64`, and `llmacro-ci`, review the execution boundary,
and only then set repository variable `DYNARE_PUBLIC_CI_ENABLED=true`.
Without that opt-in, the workflow skips rather than claiming a passing test run.
Fork pull requests are always skipped on the self-hosted runner. The workflow
uses a clean disposable checkout and isolated virtual environment; it is not
an OS sandbox and never publishes results or merges changes.

## Changelog

See `CHANGELOG.md` for release notes, including the breaking MCP and
dependency changes in 0.5.0.

## Bundled Dynare preprocessor

`dynare_lsp/bin/dynare-preprocessor.exe` is the Windows preprocessor binary
from the official **Dynare 7.1** release, redistributed **unmodified** (its
SHA-256 matches the binary shipped in the Dynare 7.1 Windows distribution).
It is free software licensed under the **GNU General Public License, version 3
or later (GPL-3.0-or-later)**; the license text and Dynare's copyright
inventory are included as `dynare_lsp/bin/DYNARE_COPYING` and
`dynare_lsp/bin/DYNARE_LICENSE.txt`.

Corresponding source code:

- Dynare 7.1 source, tag `7.1`:
  <https://git.dynare.org/Dynare/dynare/-/tree/7.1>
- Preprocessor source at the exact commit pinned by the `preprocessor`
  submodule of the Dynare `7.1` tag (the preprocessor repository has no
  separate `7.1` tag):
  <https://git.dynare.org/Dynare/preprocessor/-/tree/9c61fb6e8cb7845a63e0e30e9fa7d96f9647d6c9>
- Complete Dynare 7.1 source tarball (includes the preprocessor):
  <https://www.dynare.org/release/source/dynare-7.1.tar.xz>

## License

LLMacro Dynare LSP is licensed under the GNU General Public License v3.0 or
later (`GPL-3.0-or-later`), matching the copyleft license inherited from Dynare.
See `LICENSE` for the full terms. The bundled Dynare preprocessor remains
subject to its upstream notices; the exact Dynare 7.1 copyright inventory and
GPL text are also included as `dynare_lsp/bin/DYNARE_LICENSE.txt` and
`dynare_lsp/bin/DYNARE_COPYING`. See [dynare.org](https://www.dynare.org/).
