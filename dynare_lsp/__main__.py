"""Entry point for the Dynare Language Server.

Usage:
    python -m dynare_lsp                       # Start in stdio mode (for editors)
    python -m dynare_lsp --tcp                 # Start in TCP mode (for debugging)
    python -m dynare_lsp --check FILE.mod      # One-shot diagnostics
    python -m dynare_lsp --check --solve FILE.mod  # Diagnostics + solve SS
    python -m dynare_lsp --explain E010        # Print docs for a diagnostic code
    python -m dynare_lsp --explain --list      # List all documented codes
"""

import argparse
import sys

from .diagnostics import analyze_text
from . import explain as _explain_module

_MISSING_CHECK_FILE = "__DYNARE_LSP_MISSING_CHECK_FILE__"


def main() -> None:
    # The --check path echoes .mod source lines into diagnostic messages; on a
    # Windows cp1252 console that crashes (UnicodeEncodeError) for models using
    # non-Latin-1 characters (e.g. the LCP perpendicular operator U+27C2). Force
    # UTF-8 output with replacement so any source character prints safely. The
    # LSP server path (UTF-8 JSON-RPC) is unaffected; this only hardens the CLI.
    for _stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(_stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except ValueError:  # non-reconfigurable stream
            pass
    parser = argparse.ArgumentParser(
        description="Dynare Language Server",
    )
    parser.add_argument(
        "--tcp",
        action="store_true",
        help="Start in TCP mode instead of stdio",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="TCP host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2087,
        help="TCP port (default: 2087)",
    )
    parser.add_argument(
        "--check",
        metavar="FILE",
        nargs="?",
        const=_MISSING_CHECK_FILE,
        help="Run diagnostics on a .mod file and exit (no server)",
    )
    parser.add_argument(
        "--solve",
        action="store_true",
        help="Compute the deterministic steady state (requires scipy). "
        "Use with --check.",
    )
    parser.add_argument(
        "--explain",
        metavar="CODE",
        nargs="?",
        const="--list",
        help="Print markdown documentation for a diagnostic code "
        "(e.g. 'E010', 'W041'). Pass --explain --list to list all "
        "documented codes.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="With --explain, list all documented diagnostic codes.",
    )
    args, unknown = parser.parse_known_args()

    if args.check == _MISSING_CHECK_FILE:
        if unknown:
            args.check = unknown[0]
            unknown = unknown[1:]
        else:
            parser.error("argument --check: expected one argument")
    elif args.check is None and args.solve and args.explain is None:
        if unknown:
            args.check = unknown[0]
            unknown = unknown[1:]
        else:
            parser.error("argument --check: expected one argument")
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    if args.explain:
        _run_explain(args.explain, list_only=args.list or args.explain == "--list")
        return

    if args.check:
        _run_check(args.check, solve=args.solve)
        return

    import logging
    import os

    _log_path = os.path.join(os.path.expanduser("~"), "dynare_lsp.log")
    logging.basicConfig(
        filename=_log_path,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    from .server import start_server

    start_server(host=args.host, port=args.port, stdio=not args.tcp)


def _run_check(filepath: str, solve: bool = False) -> None:
    """Run diagnostics on a single file and print results.

    Builds a one-document ``WorkspaceIndex`` so ``@#include`` directives
    resolve against the file's own directory.  Without that, a CLI
    invocation on a main file that uses ``@#include "helper.mod"`` would
    misreport every helper-declared identifier as undeclared (E020).
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    except OSError as exc:
        # e.g. a directory path (PermissionError on Windows,
        # IsADirectoryError on POSIX) or an unreadable file.
        print(f"Error: Cannot read {filepath}: {exc}", file=sys.stderr)
        sys.exit(1)

    # Cross-file aware: feed the file through a WorkspaceIndex so
    # include symbols, circular includes, and unresolved-include
    # diagnostics all fire the same way they do under the LSP server.
    solve_model = None
    try:
        from .workspace import WorkspaceIndex
        from .diagnostics import model_with_include_context, run_diagnostics
        from .parser import parse as _parse
        import os

        abs_path = os.path.abspath(filepath)
        index = WorkspaceIndex()
        index.update_document(abs_path, text)
        include_symbols = index.collect_symbols(abs_path)
        include_models = index.resolve_all_includes(abs_path)
        include_cycles = index.find_circular_includes(abs_path)
        unresolved = index.find_unresolved_includes(abs_path)
        model = index.get_effective_model(abs_path) or _parse(text)
        solve_model = model_with_include_context(
            model,
            list(include_models.values()),
        )
        diagnostics = run_diagnostics(
            model,
            include_symbols=include_symbols,
            include_models=list(include_models.values()),
            include_cycles=include_cycles,
            unresolved_includes=unresolved,
        )
    except Exception:
        # Fall back to single-file analysis if workspace setup fails.
        diagnostics = analyze_text(text)

    # Defer to the Dynare preprocessor (authoritative parser) when available:
    # if Dynare accepts the model, our hard errors are false positives.
    try:
        import os as _os
        from .preprocessor import (
            find_preprocessor,
            run_preprocessor,
            reconcile_diagnostics,
        )

        pp_path = find_preprocessor()
        if pp_path:
            preproc_result = run_preprocessor(
                text,
                pp_path,
                source_dir=_os.path.dirname(_os.path.abspath(filepath)),
            )
            diagnostics = reconcile_diagnostics(diagnostics, preproc_result)
    except Exception:
        pass

    has_errors = any(d.severity == 1 for d in diagnostics)

    if not diagnostics:
        print(f"No issues found in {filepath}")
    else:
        errors = 0
        warnings = 0
        for d in diagnostics:
            severity_str = {1: "ERROR", 2: "WARNING", 3: "INFO", 4: "HINT"}.get(
                d.severity, "UNKNOWN"
            )
            line = d.range.start.line + 1
            col = d.range.start.character + 1
            prefix = f"{filepath}:{line}:{col}"
            print(f"{prefix}: {severity_str} [{d.code}] {d.message}")
            if d.severity == 1:
                errors += 1
            elif d.severity == 2:
                warnings += 1

        print(
            f"\n{len(diagnostics)} issue(s): {errors} error(s), {warnings} warning(s)"
        )

    if solve:
        _run_solve(text, filepath, model=solve_model)

    # Exit with error if diagnostics had errors, even when --solve also runs.
    if has_errors:
        sys.exit(1)


def _run_solve(text: str, filepath: str, model=None) -> None:
    """Compute the steady state and print results."""
    from .parser import parse

    try:
        from .solver import compute_steady_state
    except ImportError:
        print(
            "\nSolver requires scipy. Install with: pip install dynare-lsp[solver]",
            file=sys.stderr,
        )
        sys.exit(1)

    if model is None:
        model = parse(text)
    print(f"\nSolving steady state for {filepath}...")
    result = compute_steady_state(model)

    if result.success:
        print(f"Converged: {result.message}")
        print(f"Max residual: {result.residual_norm:.2e}")
        if result.n_symbolic > 0:
            print(
                f"\nSymbolic reduction: {result.n_symbolic} variables solved analytically"
            )
            if result.n_numerical > 0:
                print(
                    f"Numerical solver: {result.n_numerical} variables solved numerically"
                )
            for step in result.symbolic_steps:
                print(f"  {step}")
        print("\nSteady state values:")
        for name in sorted(result.values):
            val = result.values[name]
            print(f"  {name:20s} = {val:.10g}")

        # Blanchard-Kahn check
        try:
            from .bk_check import check_blanchard_kahn

            bk = check_blanchard_kahn(model, result.values)
            print(f"\n{bk.message}")
            if bk.forward_variables:
                print(f"  Forward-looking: {', '.join(bk.forward_variables)}")
            if bk.predetermined_variables:
                print(f"  Predetermined:   {', '.join(bk.predetermined_variables)}")
        except ImportError:
            pass
        except Exception as e:
            print(f"\nBK check failed: {e}", file=sys.stderr)

        try:
            from .model_diagnostics import check_model_diagnostics

            model_diagnostics = check_model_diagnostics(model, result.values)
            for diagnostic in model_diagnostics:
                severity_str = {
                    1: "ERROR",
                    2: "WARNING",
                    3: "INFO",
                    4: "HINT",
                }.get(diagnostic.severity, "UNKNOWN")
                line = diagnostic.range.start.line + 1
                col = diagnostic.range.start.character + 1
                prefix = f"{filepath}:{line}:{col}"
                print(
                    f"{prefix}: {severity_str} "
                    f"[{diagnostic.code}] {diagnostic.message}",
                )
        except ImportError:
            pass
        except Exception as e:
            print(f"\nModel diagnostics failed: {e}", file=sys.stderr)

        # Per-equation steady-state residuals (resid)
        try:
            from .steady_state import validate_computed_steady_state

            ss_report = validate_computed_steady_state(model, result.values)
            eq_results = [r for r in ss_report.results if not r.is_local_var]
            if eq_results:
                print("\nSteady-state residuals (resid):")
                for r in eq_results:
                    label = (
                        r.equation.name
                        or r.equation.text.strip().replace("\n", " ")[:60]
                    )
                    if r.residual is None:
                        print(f"  {'n/a':>12}  {label}")
                    else:
                        flag = "" if r.is_satisfied else "  <-- nonzero"
                        print(f"  {r.residual:12.3e}  {label}{flag}")
        except Exception:
            pass

        try:
            from .identification import check_identification

            identification_diagnostics = check_identification(model, result.values)
            for diagnostic in identification_diagnostics:
                severity_str = {
                    1: "ERROR",
                    2: "WARNING",
                    3: "INFO",
                    4: "HINT",
                }.get(diagnostic.severity, "UNKNOWN")
                line = diagnostic.range.start.line + 1
                col = diagnostic.range.start.character + 1
                prefix = f"{filepath}:{line}:{col}"
                print(
                    f"{prefix}: {severity_str} "
                    f"[{diagnostic.code}] {diagnostic.message}",
                )
        except ImportError:
            pass
        except Exception as e:
            print(f"\nIdentification check failed: {e}", file=sys.stderr)
    else:
        print(f"Failed: {result.message}", file=sys.stderr)
        sys.exit(1)


def _run_explain(code: str, list_only: bool = False) -> None:
    """Print documentation for a diagnostic code, or list all known codes."""
    if list_only or code == "--list":
        codes = _explain_module.known_codes()
        print("Documented diagnostic codes:")
        for c in codes:
            entry = _explain_module.explain(c)
            title = entry["title"] if entry else ""
            print(f"  {c:6s}  {title}")
        print(
            f"\n{len(codes)} codes. Run "
            f"`python -m dynare_lsp --explain <CODE>` for details."
        )
        return

    rendered = _explain_module.render_markdown(code)
    if rendered is None:
        print(
            f"No documentation found for diagnostic code '{code}'. "
            f"Run `python -m dynare_lsp --explain --list` to see known codes.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(rendered)


if __name__ == "__main__":
    main()
