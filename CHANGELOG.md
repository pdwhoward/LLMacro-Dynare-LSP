# Changelog

All notable user-facing changes to the Dynare LSP (Python package, VS Code
extension, and Claude Code plugin) are recorded here. Versions follow the
shared package/extension/plugin version.

## 0.5.1 - 2026-10

Diagnostics now point at the cause of an error and report every error that
can be located reliably. No configuration or API changes.

### Changed

- **Syntax errors are reported at their cause.** When Dynare's preprocessor
  rejects a file, it reports the line where its parser stops, which can be far
  below the mistake: a missing `end;` surfaces at the next block, a misspelled
  `model` keyword at the first equation, a missing `;` at the next statement.
  The server now keeps its own diagnosis at the line of the cause (for
  example "Missing 'end;' for 'model' block ... add 'end;' before 'shocks;'")
  alongside Dynare's message. A variable declared twice is reported on the
  duplicate declaration, with a fix to remove it.
- **Independent errors are reported past a syntax error.** Previously any
  syntax error hid every other check. Errors in blocks the syntax error cannot
  affect, such as an invalid shock correlation, a negative shock variance, or
  a non-finite parameter, are now reported at the same time. Checks that read
  the broken block (equation counts, undeclared references, steady state,
  Blanchard-Kahn) still wait until the syntax error is fixed; a missing `end;`
  or a misspelled block keyword withholds every finding after it.
- **Shock warnings sit on their own statement.** W110, W111, and W112 now
  point at the offending `var`, `stderr`, or `corr` statement rather than the
  first line of the `shocks` block.
- **E010 no longer recommends deleting a variable.** When an endogenous
  variable appears in no model equation, the message offers both repairs: add
  the equation that determines it, or remove it from `var` if it is not part
  of the model. The automatic quick fix that deleted the variable is gone.

### Added

- **W043 steady-state hint.** When at least two equations fail at the steady
  state and only one or two `steady_state_model` assignments appear in all of
  them, the server adds an informational hint on those assignment lines.

### Fixed

- A valid top-level assignment such as `paramters = 1;` is no longer
  reported as a misspelled `parameters` keyword.
- A semicolon inside a quoted string in a `shocks` block no longer splits the
  statement.

## 0.5.0 - 2026-09

This release contains breaking changes to the MCP server surface and the
minimum dependency versions. Review "Breaking changes" before upgrading.

### Breaking changes

- **Default MCP server is now analysis-only.** `dynare-mcp` /
  `python -m dynare_lsp.mcp_preflight_server` (used by the VS Code MCP
  provider and the Claude Code plugin) now launches a bounded *analysis*
  profile with 8 tools instead of the previous 29-tool catalog:
  `dynare_preflight`, the versioned analysis tools (`dynare_analysis_report`,
  `dynare_validate_patch`, `dynare_get_context`, `dynare_numerical_evidence`,
  `dynare_environment`, `dynare_project_analysis`), and
  `dynare_profile_policy`. Requests are limited to 8 MiB by default.
- **MATLAB execution moved behind an explicit profile.** Running a model in
  MATLAB + Dynare now requires `dynare-mcp --profile execution` and the
  `dynare_execute_trusted_model` tool with `acknowledge_code_execution=true`
  on every call and a timeout within the configured
  `--max-execution-seconds` limit. The unrestricted legacy catalog is still
  available as `python -m dynare_lsp.mcp_server`, but it is not the default.
- **pygls 2.x is required** (`pygls>=2.0.0,<3`, `lsprotocol>=2025.0.0`).
  pygls 1.x is no longer supported.
- **MCP support is an optional extra.** The core install does not pull in the
  `mcp` package; install `".[mcp]"` or `".[all]"` for the MCP server, the VS
  Code MCP provider, and the Claude Code plugin. The `dev` extra now also
  includes `mcp`.

### Changed

- `dynare_preflight` uses the versioned snapshot analysis service. Supply
  every included file in `files`; relative keys are anchored to the directory
  of an absolute `active_file`, and missing buffers no longer fall back to
  disk. Results report `requested_checks`, per-stage statuses, and a snapshot
  manifest; a failed, unsupported, or unavailable requested stage can no
  longer produce a pass.
- `python -m dynare_lsp --check --solve` now exits non-zero when the
  steady-state solve fails, the Blanchard-Kahn check fails or is skipped or
  unavailable, a numerical error diagnostic is reported, or the residual
  check fails. Ordinary warnings remain non-blocking.
- The VS Code extension (0.5.0) shows an actionable error when the language
  server cannot start, naming whether the configured interpreter is missing
  or cannot import `dynare_lsp`, with a shortcut to the `dynare.pythonPath`
  setting and the install command.
- `dynare.pythonPath` is now a machine-overridable setting, and the extension
  declares that it does not run in untrusted (Restricted Mode) workspaces.
- The VS Code extension package now includes a README.

### Fixed

- Document validation is guarded against concurrent edits, so diagnostics
  from an older buffer version are no longer published over newer text.
- Snapshot analysis preprocesses only the entry file's include closure, so
  other supplied buffers that the entry file does not include no longer
  affect preprocessor results.
- Calibration-support context keeps helper parameter names in the dependency
  closure and preserves include-local source ranges.
- Timing offsets containing tabs or other whitespace (not only spaces) are
  normalized before integer conversion in model context output.
- VS Code: restarting the language server (for example after changing
  `dynare.pythonPath`) no longer leaks language clients, file watchers, or
  output channels.
- **Blanchard-Kahn rank condition now matches Dynare.** The check tested
  the unstable instead of the stable subspace, so some determinate models
  were reported as indeterminate and some models without a stable solution
  passed. It now uses Dynare's stable-first QZ ordering and its
  `rcond(Z22) < 1e-9` test (Dynare info=5), and honors an explicit
  `qz_criterium`. The W098 stochastic-singularity impact matrix uses the
  same corrected ordering.
- **Steady states Dynare accepts are no longer rejected.** A rank-deficient
  static Jacobian is now an informational note instead of an
  "underdetermined" failure; a complete `steady_state_model` is
  authoritative; the default residual tolerance is Dynare's `solve_tolf`
  (eps^(1/3)); and `initval` values are treated as solver guesses whenever a
  steady-state-computing command follows.
- The steady-state solve respects its time budget during symbolic
  reduction, identification checks are time-bounded and memory-bounded, and
  huge integer powers such as `10^10^8` no longer hang the parser or solver.
- **Include resolution matches Dynare.** Relative `@#include` and
  `@#includepath` resolve against the entry model's folder, not the including
  file's folder, so the preprocessor gate no longer accepts models Dynare
  rejects. The preprocessor cache hashes the files Dynare actually reads,
  non-ASCII paths no longer crash the bundled preprocessor, and preprocessor
  columns are converted from UTF-8 bytes to characters.
- Preprocessor errors without a file location keep Dynare's message instead
  of a generic "exited with code 1" diagnostic; a NUL byte is reported
  immediately instead of hanging the preprocessor.
- Dynare is found at its default macOS location
  (`/Applications/Dynare/<version>`) and in `/usr/lib64/dynare`.
- Editor: Format Document can no longer revert edits in files containing
  form feeds or other Unicode line separators; declaration quick fixes
  insert on the correct line after a multi-line header comment; rename and
  references no longer skip code after a `/*` inside a line comment or
  rewrite command option names; saving keeps solver, Blanchard-Kahn, and
  preprocessor results; closing a file clears its diagnostics; unsaved
  include buffers are used for preprocessing; creating a missing include
  clears E061; and diagnostics are no longer shown twice in VS Code (the
  server now uses push diagnostics only).
- Diagnostics: a regex that could take exponential time on an unfinished
  block is now linear; false positives were removed for `1.E-3`-style
  numbers, comparison operators, duplicate `var` declarations, command
  options such as `external_function(name=...)`, multi-line parameter
  assignments, `inf` inverse-gamma prior standard deviations, `histval`
  lag requirements, and re-specified or `learnt_in` shocks.
- The formatter no longer removes non-ASCII whitespace that Dynare rejects,
  `dynare-lsp --list` without `--explain` exits with a usage error instead of
  hanging, and model comparison handles every `shocks` block and is much
  faster on large models.
- MCP: `dynare_validate_patch` accepts file names with uppercase letters and
  blank context lines; `@#define`/`@#if`/`@#for` no longer block
  preprocessing; include errors no longer leak temporary paths; outputs for
  large models are bounded with truncation summaries; `dynare_get_context`
  is linear-time and respects `max_bytes`; tools run in worker threads with
  a per-call deadline (`--max-call-seconds`) so one slow call cannot stall
  the server; and preflight snapshot IDs can be reused by
  `dynare_validate_patch`.

### Packaging

- The public repository now ships this changelog, a `.gitattributes` file
  that normalizes text files to LF, and a curated public regression suite
  (`dynare_lsp/tests`) with opt-in self-hosted CI.
- The README documents the bundled, unmodified Dynare 7.1 preprocessor
  binary, its GPL-3.0-or-later license, and where to obtain the
  corresponding source.
- The Claude Code plugin and README document using `python3` (or a full
  interpreter path) on macOS and Linux.
