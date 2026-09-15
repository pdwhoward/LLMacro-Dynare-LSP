"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.stageTitle = stageTitle;
exports.registerAnalysisReport = registerAnalysisReport;
const vscode = __importStar(require("vscode"));
const path = __importStar(require("path"));
function stageTitle(report) {
    if (report.freshness === "stale")
        return "Dynare analysis: stale — rerun";
    if (report.freshness !== "current")
        return "Dynare analysis: not run";
    return Object.entries(report.stages ?? {})
        .map(([name, value]) => `${name.replace(/_/g, " ")}: ${value.status}`)
        .join(" | ");
}
function editorSettings(uri) {
    const config = vscode.workspace.getConfiguration("dynare", uri);
    const globalPaths = vscode.workspace.getConfiguration("dynare").get("searchPaths", []).filter((value) => path.isAbsolute(value));
    const base = vscode.workspace.getWorkspaceFolder(uri)?.uri.fsPath ?? path.dirname(uri.fsPath);
    const resourcePaths = config.get("searchPaths", []).filter((value) => value.trim().length > 0)
        .map((value) => path.isAbsolute(value.trim()) ? value.trim() : path.resolve(base, value.trim()));
    return { tolerance: config.get("steadyStateTolerance", 1e-6), search_paths: Array.from(new Set([...globalPaths, ...resourcePaths])) };
}
function reportConfiguration(uri) {
    return { ...editorSettings(uri), numerical: true, preprocessor: true };
}
function registerAnalysisReport(context, getClient) {
    const changed = new vscode.EventEmitter();
    const documents = new Map();
    let sequence = 0;
    const provider = {
        provideTextDocumentContent: (uri) => documents.get(uri.toString()) ?? "Report expired; run analysis again.",
    };
    async function execute(command, uri, extra = {}) {
        const client = await getClient();
        if (!client?.isRunning())
            throw new Error("Dynare language server is not running.");
        return client.sendRequest("workspace/executeCommand", {
            command, arguments: [{ ...extra, ...(uri ? { uri: uri.toString() } : {}) }],
        });
    }
    async function display(command, uri, extra) {
        try {
            const settingsBefore = uri && extra.config !== undefined ? JSON.stringify(editorSettings(uri)) : undefined;
            const result = await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: "Dynare analysis" }, () => execute(command, uri, extra));
            if (uri && settingsBefore !== undefined && settingsBefore !== JSON.stringify(editorSettings(uri))) {
                result.freshness = "stale";
                result.checks_passed = false;
            }
            // Read-only virtual document; no model edits or workspace file writes.
            const reportUri = vscode.Uri.parse(`dynare-analysis:/report-${++sequence}.json`);
            documents.set(reportUri.toString(), JSON.stringify(result, null, 2));
            while (documents.size > 20)
                documents.delete(documents.keys().next().value);
            await vscode.window.showTextDocument(await vscode.workspace.openTextDocument(reportUri), { preview: true, viewColumn: vscode.ViewColumn.Beside });
            changed.fire();
        }
        catch (error) {
            void vscode.window.showErrorMessage(`Dynare analysis failed: ${String(error)}`);
        }
    }
    context.subscriptions.push(changed, vscode.workspace.registerTextDocumentContentProvider("dynare-analysis", provider), vscode.commands.registerCommand("dynare.showAnalysisReport", async (uri) => {
        const target = uri ?? vscode.window.activeTextEditor?.document.uri;
        if (!target)
            return;
        await display("dynare/analysisReport", target, { config: reportConfiguration(target) });
    }), vscode.commands.registerCommand("dynare.analysisTools", async () => {
        try {
            const client = await getClient();
            if (!client?.isRunning())
                throw new Error("Dynare language server is not running.");
            const tools = await client.sendRequest("workspace/executeCommand", { command: "dynare/analysisFeatures", arguments: [] });
            const selected = await vscode.window.showQuickPick(tools.map((tool) => ({ label: tool.name, description: tool.command, tool })));
            if (!selected)
                return;
            const input = await vscode.window.showInputBox({ prompt: "Analysis arguments as JSON (active source buffers are supplied automatically)", value: JSON.stringify(selected.tool.example ?? {}), validateInput: (text) => {
                    try {
                        const value = JSON.parse(text);
                        return value && typeof value === "object" && !Array.isArray(value) ? undefined : "Enter a JSON object.";
                    }
                    catch {
                        return "Enter valid JSON.";
                    }
                } });
            if (input === undefined)
                return;
            const target = vscode.window.activeTextEditor?.document.uri;
            const values = JSON.parse(input);
            if (target && selected.tool.parameters.includes("config")) {
                values.config = { ...editorSettings(target), ...(values.config ?? {}) };
            }
            await display(selected.tool.command, target, values);
        }
        catch (error) {
            void vscode.window.showErrorMessage(String(error));
        }
    }), vscode.languages.registerCodeLensProvider({ scheme: "file", language: "dynare" }, {
        onDidChangeCodeLenses: changed.event,
        async provideCodeLenses(document, token) {
            const version = document.version;
            try {
                // Cache lookup only: never start a solve just to paint a CodeLens.
                const result = await execute("dynare/cachedAnalysisReport", document.uri, { config: reportConfiguration(document.uri) });
                if (token.isCancellationRequested || document.version !== version)
                    return [];
                return [new vscode.CodeLens(new vscode.Range(0, 0, 0, 0), { title: stageTitle(result), command: "dynare.showAnalysisReport", arguments: [document.uri] })];
            }
            catch {
                return [];
            }
        },
    }), vscode.workspace.onDidChangeTextDocument(() => changed.fire()), vscode.workspace.onDidCloseTextDocument(() => changed.fire()), vscode.languages.onDidChangeDiagnostics(() => changed.fire()), vscode.workspace.onDidChangeConfiguration((event) => { if (event.affectsConfiguration("dynare"))
        changed.fire(); }));
}
//# sourceMappingURL=analysisReport.js.map