"""Immediate syntax diagnostics plus debounced workspace validation."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from . import runtime_dispatch
from .incremental import apply_content_changes, reuse_semantically_identical_document


def run_fast_diagnostics(model):
    """Return conservative diagnostics that never require include context."""
    from . import diagnostics as d

    findings = []
    findings.extend(d._check_unmatched_macro_blocks(model))
    macro_errors = d._check_macro_error_directives(model)
    findings.extend(macro_errors)
    parse_errors = d._check_parse_errors(model)
    parse_errors.extend(d._check_invalid_identifier_declarations(model))
    findings.extend(parse_errors)
    merged_equations = d._check_merged_equations(model)
    merged_assignments = d._check_merged_assignments(model)
    unbalanced = d._check_unbalanced_parens(model)
    findings.extend(merged_equations)
    findings.extend(merged_assignments)
    findings.extend(unbalanced)
    if not (
        macro_errors
        or parse_errors
        or merged_equations
        or merged_assignments
        or unbalanced
    ):
        findings.extend(d._check_duplicate_declarations(model, None))
    return findings


def install(core) -> None:
    if getattr(core, "_diagnostic_tiers_installed", False):
        return
    core._diagnostic_tiers_installed = True
    core._validation_executor = ThreadPoolExecutor(max_workers=1)
    core._pending_validations = {}
    core._validation_tokens = {}

    def cancel(uri: str) -> None:
        with core._state_lock:
            timer = core._pending_validations.pop(uri, None)
            core._validation_tokens[uri] = core._validation_tokens.get(uri, 0) + 1
        if timer is not None:
            timer.cancel()

    def validate_fast(uri: str, text: str) -> None:
        with core._state_lock:
            generation = core._document_generations.get(uri, 0) + 1
            core._document_generations[uri] = generation
            core._solve_generations[uri] = core._solve_generations.get(uri, 0) + 1
            previous_solver_result = core._document_solver_results.pop(uri, None)
            if previous_solver_result is not None and previous_solver_result.success:
                core._document_warm_start_results[uri] = previous_solver_result
            core._document_bk_results.pop(uri, None)
            core._document_model_diagnostics.pop(uri, None)
            core._document_identification.pop(uri, None)
            core._document_preprocessor_results.pop(uri, None)
            core._document_run_diagnostics.pop(uri, None)
            core._document_ss_reports.pop(uri, None)
        try:
            model = core.parse(text)
            findings = run_fast_diagnostics(model)
        except Exception as exc:
            model = None
            findings = [
                core.DDiag(
                    range=core.DRange(core.DPos(0, 0), core.DPos(0, 1)),
                    severity=core.Severity.ERROR,
                    message=f"Internal parser error: {exc}",
                    code="E999",
                )
            ]
        with core._state_lock:
            if core._document_generations.get(uri, 0) != generation:
                return
            core._document_sources[uri] = text
            if model is not None:
                core._document_models[uri] = model
            core._document_diagnostics[uri] = findings
        core._publish_diagnostics_if_current(
            uri,
            [core._to_lsp_diagnostic(item, text) for item in findings],
            generation,
        )

    def schedule_full(uri: str, text: str, delay: float = 0.25) -> None:
        with core._state_lock:
            prior = core._pending_validations.pop(uri, None)
            token = core._validation_tokens.get(uri, 0) + 1
            core._validation_tokens[uri] = token
        if prior is not None:
            prior.cancel()

        def run_if_current() -> None:
            with core._state_lock:
                if core._validation_tokens.get(uri, 0) != token:
                    return
            try:
                current = core.server.workspace.get_text_document(uri).source
            except Exception:
                return
            if current != text:
                return
            core._validate_document(uri, text)
            core._schedule_solve(uri)
            core._revalidate_cached_documents(schedule_solve=True, exclude_uri=uri)

        def submit() -> None:
            with core._state_lock:
                if (
                    core._pending_validations.get(uri) is not timer
                    or core._validation_tokens.get(uri, 0) != token
                ):
                    return
                core._pending_validations.pop(uri, None)
            core._validation_executor.submit(run_if_current)

        timer = threading.Timer(delay, submit)
        timer.daemon = True
        with core._state_lock:
            core._pending_validations[uri] = timer
        timer.start()

    def did_change(params):
        uri = params.text_document.uri
        if not params.content_changes:
            return None
        with core._state_lock:
            previous = core._document_sources.get(uri)
            pending = core._pending_preprocess.pop(uri, None)
        if pending is not None:
            pending.cancel()
        try:
            text = (
                apply_content_changes(
                    previous, params.content_changes, core._position_encoding
                )
                if previous is not None
                else core.server.workspace.get_text_document(uri).source
            )
        except Exception:
            text = core.server.workspace.get_text_document(uri).source
        if reuse_semantically_identical_document(core, uri, previous, text):
            # Reusable parsing does not prove full validation has completed.
            # Reschedule even if the old timer already submitted its worker.
            schedule_full(uri, text)
            return None
        validate_fast(uri, text)
        schedule_full(uri, text)
        return None

    core._cancel_pending_validation = cancel
    core._schedule_full_validation = schedule_full
    runtime_dispatch.document_change_handler = did_change
    runtime_dispatch.document_close_callbacks.append(cancel)
