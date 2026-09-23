"""Self-contained public regressions: no oracle corpus, MATLAB, or private data.

Published verbatim by publish/publish_lsp.ps1. Install .[dev] before running.
"""
from __future__ import annotations

import asyncio
import ast
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

SOURCE = "var y; model(linear); y = 0.5*y(-1); end;"


@pytest.fixture
def guarded_core(monkeypatch):
    from dynare_lsp import runtime_dispatch, validation_guard

    docs = {"model.mod": NS(source="A", version=1), "other.mod": NS(source="other", version=1)}
    core = NS(
        _state_lock=threading.Lock(),
        _client_send_lock=threading.Lock(),
        _document_generations={"model.mod": 1, "other.mod": 1},
        _normalize_uri=lambda uri: uri,
        _validation_tokens={"model.mod": 1, "other.mod": 1},
        _document_sources={"model.mod": "A", "other.mod": "other"},
        server=NS(workspace=NS(get_text_document=lambda uri: docs[uri])),
        validated=[],
    )

    def write(uri, text):
        core.validated.append((uri, text))
        core._document_sources[uri] = text

    core.perform_validation = write
    core._validate_document = lambda uri, text: core.perform_validation(uri, text)

    def change(params):
        uri = params.text_document.uri
        with core._state_lock:
            core._validation_tokens[uri] += 1
            core._document_sources[uri] = params.text

    monkeypatch.setattr(runtime_dispatch, "document_change_handler", change)
    monkeypatch.setattr(runtime_dispatch, "document_close_callbacks", [])
    # Register restoration before install changes the shared dispatch point.
    monkeypatch.setattr(runtime_dispatch, "document_lifecycle_gate", runtime_dispatch.document_lifecycle_gate)
    validation_guard.install(core)
    return core, docs, runtime_dispatch


def test_validation_rechecks_token_before_mutation(guarded_core):
    core, docs, dispatch = guarded_core
    docs["model.mod"].source = "B"
    dispatch.document_change_handler(NS(text_document=NS(uri="model.mod"), text="B"))
    assert core._validate_scheduled_document("model.mod", "A", 1) is False
    assert core._document_sources["model.mod"] == "B"
    assert core.validated == []


def test_validation_rechecks_live_editor_source(guarded_core):
    core, docs, _ = guarded_core
    # pygls has updated the document but didChange has not reached our handler.
    docs["model.mod"].source = "B"
    assert core._validate_scheduled_document("model.mod", "A", 1) is False
    assert core.validated == []


def test_validation_and_edit_cannot_overwrite_each_other(guarded_core):
    core, docs, dispatch = guarded_core
    entered, release, attempted, edited = (threading.Event() for _ in range(4))

    def slow_write(uri, text):
        entered.set()
        assert release.wait(5), "validation test worker was not released"
        core._document_sources[uri] = text

    def edit():
        docs["model.mod"].source = "B"
        attempted.set()
        dispatch.document_change_handler(NS(text_document=NS(uri="model.mod"), text="B"))
        edited.set()

    core.perform_validation = slow_write
    with ThreadPoolExecutor(max_workers=2) as pool:
        old = pool.submit(core._validate_scheduled_document, "model.mod", "A", 1)
        try:
            assert entered.wait(5)
            newer = pool.submit(edit)
            assert attempted.wait(5)
            assert not edited.wait(0.05), "edit interleaved with a cache commit"
        finally:
            release.set()
        assert old.result(timeout=5) is True
        newer.result(timeout=5)
    assert core._document_sources["model.mod"] == "B"
    assert core._validation_tokens["model.mod"] == 2


def test_validation_close_cancels_already_queued_work(guarded_core):
    core, _, dispatch = guarded_core
    with dispatch.document_lifecycle_gate("model.mod"):
        core._validation_tokens["model.mod"] += 1
        core._document_sources.pop("model.mod")
    assert core._validate_scheduled_document("model.mod", "A", 1) is False
    assert "model.mod" not in core._document_sources


def test_validation_different_documents_have_independent_gates(guarded_core):
    core, _, dispatch = guarded_core
    with dispatch.document_lifecycle_gate("model.mod"):
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(core._validate_scheduled_document, "other.mod", "other", 1)
            assert result.result(timeout=5) is True


def test_validation_gate_is_reentrant_and_install_is_idempotent(guarded_core):
    from dynare_lsp.validation_guard import install

    core, _, dispatch = guarded_core
    before = core._validate_document
    install(core)
    assert core._validate_document is before
    with dispatch.document_lifecycle_gate("model.mod"):
        assert core._validate_scheduled_document("model.mod", "A", 1) is True


def test_diagnostic_publish_rejects_overtaken_source(guarded_core):
    core, docs, _ = guarded_core
    sent = []
    core.server.text_document_publish_diagnostics = sent.append
    docs["model.mod"].source = "B"
    docs["model.mod"].version = 2
    assert core._publish_diagnostics_if_current("model.mod", [], 1) is False
    assert not sent


def test_diagnostic_publish_rejects_document_removed_before_close(guarded_core):
    core, docs, _ = guarded_core
    sent = []
    core.server.text_document_publish_diagnostics = sent.append
    assert core._validate_scheduled_document("model.mod", "A", 1) is True
    docs.pop("model.mod")
    assert core._publish_diagnostics_if_current("model.mod", [], 1) is False
    assert not sent


def test_diagnostic_publish_includes_document_version(guarded_core):
    core, _, _ = guarded_core
    sent = []
    core.server.text_document_publish_diagnostics = sent.append
    assert core._publish_diagnostics_if_current("model.mod", [], 1) is True
    assert sent[0].version == 1


@pytest.fixture
def native_preflight(monkeypatch):
    from dynare_lsp import preprocessor
    from dynare_lsp.mcp_preflight_server import dynare_preflight

    # Never execute an external binary in fault-injection tests.
    monkeypatch.setattr(preprocessor, "find_preprocessor", lambda: None)
    return dynare_preflight


def test_preflight_jacobian_exception_is_not_a_pass(native_preflight, monkeypatch):
    from dynare_lsp import bk_check

    def fail(*args, **kwargs):
        raise RuntimeError("injected Jacobian failure")

    monkeypatch.setattr(bk_check, "_compute_jacobian", fail)
    result = native_preflight(SOURCE)
    assert result["stages"]["jacobian"]["status"] == "unavailable", result
    assert result["blocking_stage"] == "jacobian"
    assert result["preflight_passed"] is False


@pytest.mark.parametrize("kind,status", [("nonfinite", "unavailable"), ("shape", "unsupported"), ("singular", "failed")])
def test_preflight_requires_valid_jacobian_evidence(native_preflight, monkeypatch, kind, status):
    import numpy as np
    from dynare_lsp import bk_check

    shape = (2, 1) if kind == "shape" else (1, 1)
    matrix = np.full(shape, np.nan if kind == "nonfinite" else 0.0)
    monkeypatch.setattr(bk_check, "_compute_jacobian", lambda *a, **kw: (matrix, matrix, matrix))
    result = native_preflight(SOURCE)
    assert result["stages"]["jacobian"]["status"] == status, result
    assert result["preflight_passed"] is False


def test_preflight_empty_residual_report_is_not_a_pass(native_preflight, monkeypatch):
    from dynare_lsp import steady_state

    monkeypatch.setattr(steady_state, "validate_computed_steady_state", lambda *a, **kw: NS(results=[]))
    result = native_preflight(SOURCE)
    assert result["blocking_stage"] == "residuals", result
    assert result["preflight_passed"] is False


def test_preflight_failed_preprocessor_is_blocking(native_preflight, monkeypatch):
    from dynare_lsp import analysis_service, preprocessor

    monkeypatch.setattr(preprocessor, "find_preprocessor", lambda: "injected-preprocessor")
    monkeypatch.setattr(analysis_service, "_preprocess", lambda snap: ({"status": "failed", "findings": []}, None))
    result = native_preflight(SOURCE)
    assert result["blocking_stage"] == "preprocessor", result
    assert result["preflight_passed"] is False
    assert result["stages"]["steady_state"]["status"] == "not_run"


@pytest.mark.parametrize("cwd_name", ["caller-a", "caller-b"])
def test_preflight_supplied_include_beats_disk(native_preflight, tmp_path, monkeypatch, cwd_name):
    project = tmp_path / "project"
    project.mkdir()
    (project / "params.inc").write_text("parameters a; a=0.9;", encoding="utf-8")
    cwd = tmp_path / cwd_name
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    source = 'var y;\n@#include "params.inc"\nmodel; y=a*y(-1)+1; end;\n'
    result = native_preflight(source, active_file=str(project / "main.mod"), files={"params.inc": "parameters a; a=0.5;"})
    assert result["preflight_passed"], result
    assert result["stages"]["steady_state"]["values"]["y"] == pytest.approx(2.0)
    assert result["snapshot"]["file_policy"] == "supplied_only"


def test_preflight_missing_include_does_not_read_disk(native_preflight, tmp_path):
    (tmp_path / "params.inc").write_text("parameters a; a=0.5;", encoding="utf-8")
    source = 'var y;\n@#include "params.inc"\nmodel; y=a*y(-1); end;\n'
    result = native_preflight(source, active_file=str(tmp_path / "main.mod"))
    assert result["preflight_passed"] is False
    assert result["blocking_stage"] == "diagnostics"


def test_preflight_legacy_locations_and_json_are_preserved(native_preflight):
    result = native_preflight("var y; model; y=missing; end;")
    findings = result["stages"]["diagnostics"]["findings"]
    assert findings
    assert all(row["line"] == row["range"]["start"]["line"] + 1 for row in findings)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("behavior", ["unsatisfied", "skipped", "exception", "import_error"])
def test_cli_failed_bk_exits_nonzero(monkeypatch, behavior):
    from dynare_lsp import __main__ as cli, bk_check

    def verdict(*args, **kwargs):
        if behavior == "exception":
            raise RuntimeError("injected failure")
        if behavior == "import_error":
            raise ImportError("injected missing dependency")
        return NS(satisfied=behavior == "skipped", message="Blanchard-Kahn check skipped" if behavior == "skipped" else "BK conditions fail", forward_variables=[], predetermined_variables=[])

    monkeypatch.setattr(bk_check, "check_blanchard_kahn", verdict)
    with pytest.raises(SystemExit) as exc:
        cli._run_solve(SOURCE, "model.mod")
    assert exc.value.code == 1


def test_cli_successful_numerics_remain_successful():
    from dynare_lsp import __main__ as cli

    assert cli._run_solve(SOURCE, "model.mod") is None


def test_default_mcp_catalog_is_analysis_only():
    from dynare_lsp.mcp_preflight_server import build_server
    from dynare_lsp.mcp_profiles import ANALYSIS_TOOLS

    tools = asyncio.run(build_server().list_tools())
    assert {tool.name for tool in tools} == ANALYSIS_TOOLS | {"dynare_profile_policy"}
    preflight = next(tool for tool in tools if tool.name == "dynare_preflight")
    annotations = preflight.annotations
    assert annotations is not None
    assert annotations.destructiveHint is False
    assert annotations.openWorldHint is False


def test_execution_profile_is_explicit_and_annotated():
    from dynare_lsp.mcp_preflight_server import build_server

    tools = asyncio.run(build_server("execution").list_tools())
    assert "dynare_run_dynare" not in {tool.name for tool in tools}
    execute = next(tool for tool in tools if tool.name == "dynare_execute_trusted_model")
    annotations = execute.annotations
    assert annotations is not None
    assert annotations.destructiveHint is True
    assert annotations.openWorldHint is True


def test_execution_requires_acknowledgment_and_obeys_timeout(monkeypatch):
    from dynare_lsp import matlab_runner
    from dynare_lsp.mcp_profiles import execution_tool

    calls = []
    monkeypatch.setattr(matlab_runner, "run_dynare_matlab", lambda *a, **kw: calls.append((a, kw)) or {"success": True})
    execute = execution_tool(10)
    with pytest.raises(PermissionError):
        execute(SOURCE, timeout=5)
    with pytest.raises(ValueError):
        execute(SOURCE, acknowledge_code_execution=True, timeout=11)
    assert calls == []
    assert execute(SOURCE, acknowledge_code_execution=True, timeout=5)["success"]
    assert calls[0][1]["timeout"] == 5


def test_request_limit_runs_before_the_tool():
    from dynare_lsp.mcp_profiles import bounded_call

    calls = []

    def operation(file_content: str):
        calls.append(file_content)
        return {"success": True}

    bounded = bounded_call(operation, 64)
    with pytest.raises(ValueError, match="byte limit"):
        bounded("x" * 100)
    assert calls == []
    assert bounded("ok") == {"success": True}


def test_package_requires_supported_transport():
    from packaging.requirements import Requirement

    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "setup.py").read_text(encoding="utf-8"))
    setup_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    keywords = {item.arg: ast.literal_eval(item.value) for item in setup_call.keywords if item.arg in {"install_requires", "extras_require"}}
    requirements = {r.name: r for r in map(Requirement, keywords["install_requires"])}
    assert "1.3.1" not in requirements["pygls"].specifier
    assert "2.0.0" in requirements["pygls"].specifier
    assert "3.0.0" not in requirements["pygls"].specifier
    assert "2025.0.0" in requirements["lsprotocol"].specifier
    assert any(r.startswith("mcp") for r in keywords["extras_require"]["dev"])


def test_public_assets_are_self_contained():
    root = Path(__file__).resolve().parents[2]
    assert (Path(__file__).parent / "fixtures" / "public_stable.mod").is_file()
    workflow = root / ".github" / "workflows" / "public-ci.yml"
    if not workflow.is_file():
        workflow = root.parent / "publish" / "public-ci.yml"
    text = workflow.read_text(encoding="utf-8")
    assert "DYNARE_PUBLIC_CI_ENABLED == 'true'" in text
    assert "head.repo.full_name == github.repository" in text
    assert "runs-on: [self-hosted, Windows, X64, llmacro-ci]" in text
    assert "persist-credentials: false" in text
    assert "pygls==2.0.0" in text


def _frame(message):
    payload = json.dumps({"jsonrpc": "2.0", **message}).encode("utf-8")
    return f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii") + payload


def _messages(output):
    while output:
        headers, separator, body = output.partition(b"\r\n\r\n")
        assert separator, output
        length = next(int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n") if line.lower().startswith(b"content-length:"))
        yield json.loads(body[:length])
        output = body[length:]


def test_lsp_initialize_open_publish_smoke(tmp_path):
    path = tmp_path / "invalid.mod"
    path.write_text("var y; model; y=missing; end;", encoding="utf-8")
    messages = [
        {"id": 1, "method": "initialize", "params": {"processId": None, "rootUri": tmp_path.as_uri(), "capabilities": {}}},
        {"method": "initialized", "params": {}},
        {"method": "textDocument/didOpen", "params": {"textDocument": {"uri": path.as_uri(), "languageId": "dynare", "version": 1, "text": path.read_text(encoding="utf-8")}}},
        {"id": 2, "method": "shutdown", "params": None},
        {"method": "exit", "params": None},
    ]
    env = dict(os.environ, HOME=str(tmp_path), USERPROFILE=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-m", "dynare_lsp"], input=b"".join(map(_frame, messages)), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=30)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
    replies = list(_messages(proc.stdout))
    assert any(r.get("id") == 1 and "result" in r for r in replies), replies
    diagnostics = [r for r in replies if r.get("method") == "textDocument/publishDiagnostics"]
    assert diagnostics, replies
    assert any(r["params"]["diagnostics"] for r in diagnostics), replies
