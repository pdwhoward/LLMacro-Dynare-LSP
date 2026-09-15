import * as vscode from "vscode";
import * as path from "path";
import { LanguageClient } from "vscode-languageclient/node";

type GetClient = () => Promise<LanguageClient | undefined>;
type Report = { freshness?: string; stages?: Record<string, { status: string }> };

export function stageTitle(report: Report): string {
  if (report.freshness === "stale") return "Dynare analysis: stale — rerun";
  if (report.freshness !== "current") return "Dynare analysis: not run";
  return Object.entries(report.stages ?? {})
    .map(([name, value]) => `${name.replace(/_/g, " ")}: ${value.status}`)
    .join(" | ");
}

function editorSettings(uri: vscode.Uri): Record<string, unknown> {
  const config = vscode.workspace.getConfiguration("dynare", uri);
  const globalPaths = vscode.workspace.getConfiguration("dynare").get<string[]>("searchPaths", []).filter((value) => path.isAbsolute(value));
  const base = vscode.workspace.getWorkspaceFolder(uri)?.uri.fsPath ?? path.dirname(uri.fsPath);
  const resourcePaths = config.get<string[]>("searchPaths", []).filter((value) => value.trim().length > 0)
    .map((value) => path.isAbsolute(value.trim()) ? value.trim() : path.resolve(base, value.trim()));
  return { tolerance: config.get<number>("steadyStateTolerance", 1e-6), search_paths: Array.from(new Set([...globalPaths, ...resourcePaths])) };
}

function reportConfiguration(uri: vscode.Uri): Record<string, unknown> {
  return { ...editorSettings(uri), numerical: true, preprocessor: true };
}

export function registerAnalysisReport(context: vscode.ExtensionContext, getClient: GetClient): void {
  const changed = new vscode.EventEmitter<void>();
  const documents = new Map<string, string>();
  let sequence = 0;
  const provider: vscode.TextDocumentContentProvider = {
    provideTextDocumentContent: (uri) => documents.get(uri.toString()) ?? "Report expired; run analysis again.",
  };

  async function execute(command: string, uri: vscode.Uri | undefined, extra: Record<string, unknown> = {}) {
    const client = await getClient();
    if (!client?.isRunning()) throw new Error("Dynare language server is not running.");
    return client.sendRequest<any>("workspace/executeCommand", {
      command, arguments: [{ ...extra, ...(uri ? { uri: uri.toString() } : {}) }],
    });
  }

  async function display(command: string, uri: vscode.Uri | undefined, extra: Record<string, unknown>) {
    try {
      const settingsBefore = uri && extra.config !== undefined ? JSON.stringify(editorSettings(uri)) : undefined;
      const result = await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: "Dynare analysis" },
        () => execute(command, uri, extra)
      );
      if (uri && settingsBefore !== undefined && settingsBefore !== JSON.stringify(editorSettings(uri))) {
        result.freshness = "stale";
        result.checks_passed = false;
      }
      // Read-only virtual document; no model edits or workspace file writes.
      const reportUri = vscode.Uri.parse(`dynare-analysis:/report-${++sequence}.json`);
      documents.set(reportUri.toString(), JSON.stringify(result, null, 2));
      while (documents.size > 20) documents.delete(documents.keys().next().value!);
      await vscode.window.showTextDocument(await vscode.workspace.openTextDocument(reportUri), { preview: true, viewColumn: vscode.ViewColumn.Beside });
      changed.fire();
    } catch (error) {
      void vscode.window.showErrorMessage(`Dynare analysis failed: ${String(error)}`);
    }
  }

  context.subscriptions.push(
    changed,
    vscode.workspace.registerTextDocumentContentProvider("dynare-analysis", provider),
    vscode.commands.registerCommand("dynare.showAnalysisReport", async (uri?: vscode.Uri) => {
      const target = uri ?? vscode.window.activeTextEditor?.document.uri;
      if (!target) return;
      await display("dynare/analysisReport", target, { config: reportConfiguration(target) });
    }),
    vscode.commands.registerCommand("dynare.analysisTools", async () => {
      try {
        const client = await getClient();
        if (!client?.isRunning()) throw new Error("Dynare language server is not running.");
        const tools = await client.sendRequest<any[]>("workspace/executeCommand", { command: "dynare/analysisFeatures", arguments: [] });
        const selected = await vscode.window.showQuickPick(tools.map((tool) => ({ label: tool.name as string, description: tool.command as string, tool })));
        if (!selected) return;
        const input = await vscode.window.showInputBox({ prompt: "Analysis arguments as JSON (active source buffers are supplied automatically)", value: JSON.stringify(selected.tool.example ?? {}), validateInput: (text) => {
          try { const value = JSON.parse(text); return value && typeof value === "object" && !Array.isArray(value) ? undefined : "Enter a JSON object."; } catch { return "Enter valid JSON."; }
        } });
        if (input === undefined) return;
        const target = vscode.window.activeTextEditor?.document.uri;
        const values = JSON.parse(input);
        if (target && selected.tool.parameters.includes("config")) {
          values.config = { ...editorSettings(target), ...(values.config ?? {}) };
        }
        await display(selected.tool.command, target, values);
      } catch (error) { void vscode.window.showErrorMessage(String(error)); }
    }),
    vscode.languages.registerCodeLensProvider({ scheme: "file", language: "dynare" }, {
      onDidChangeCodeLenses: changed.event,
      async provideCodeLenses(document, token) {
        const version = document.version;
        try {
          // Cache lookup only: never start a solve just to paint a CodeLens.
          const result: Report = await execute("dynare/cachedAnalysisReport", document.uri, { config: reportConfiguration(document.uri) });
          if (token.isCancellationRequested || document.version !== version) return [];
          return [new vscode.CodeLens(new vscode.Range(0, 0, 0, 0), { title: stageTitle(result), command: "dynare.showAnalysisReport", arguments: [document.uri] })];
        } catch { return []; }
      },
    }),
    vscode.workspace.onDidChangeTextDocument(() => changed.fire()),
    vscode.workspace.onDidCloseTextDocument(() => changed.fire()),
    vscode.languages.onDidChangeDiagnostics(() => changed.fire()),
    vscode.workspace.onDidChangeConfiguration((event) => { if (event.affectsConfiguration("dynare")) changed.fire(); })
  );
}
