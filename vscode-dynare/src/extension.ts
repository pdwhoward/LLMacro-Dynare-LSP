import * as path from "path";
import * as vscode from "vscode";
import { workspace, ExtensionContext, WorkspaceFolder } from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
} from "vscode-languageclient/node";
import { registerAnalysisReport } from "./analysisReport";

let client: LanguageClient | undefined;
let clientStartPromise: Promise<void> | undefined;
let clientGeneration = 0;
let currentPythonPath = "python";

function configuredPythonPath(): string {
  const config = workspace.getConfiguration("dynare");
  return config.get<string>("pythonPath", "python");
}

function resolveSearchPath(entry: string, workspaceFolder?: WorkspaceFolder): string {
  const trimmed = entry.trim();
  if (!trimmed || path.isAbsolute(trimmed)) {
    return trimmed;
  }
  if (!workspaceFolder) {
    return trimmed;
  }
  return path.join(workspaceFolder.uri.fsPath, trimmed);
}

function configuredSearchPaths(): string[] {
  const config = workspace.getConfiguration("dynare");
  // Relative entries belong only in searchPathsByRoot, where each workspace
  // folder resolves them against itself instead of leaking them across roots.
  const absolute = config
    .get<string[]>("searchPaths", [])
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0 && path.isAbsolute(entry));
  return Array.from(new Set(absolute));
}

function configuredSearchPathsByRoot(): Record<string, string[]> {
  const workspaceFolders = workspace.workspaceFolders;
  const pathsByRoot: Record<string, string[]> = {};
  if (!workspaceFolders || workspaceFolders.length === 0) {
    return pathsByRoot;
  }
  for (const workspaceFolder of workspaceFolders) {
    const config = workspace.getConfiguration("dynare", workspaceFolder.uri);
    const resolved: string[] = [];
    for (const entry of config.get<string[]>("searchPaths", [])) {
      resolved.push(resolveSearchPath(entry, workspaceFolder));
    }
    pathsByRoot[workspaceFolder.uri.fsPath] = Array.from(new Set(resolved));
  }
  return pathsByRoot;
}

function dynareConfigurationPayload() {
  const config = workspace.getConfiguration("dynare");
  return {
    dynare: {
      pythonPath: config.get<string>("pythonPath", "python"),
      steadyStateTolerance: config.get<number>("steadyStateTolerance", 1e-6),
      preprocessorPath: config.get<string>("preprocessorPath", ""),
      formatIndent: config.get<string | number>("formatIndent", "tab"),
      searchPaths: configuredSearchPaths(),
      searchPathsByRoot: configuredSearchPathsByRoot(),
    },
  };
}

async function pushDynareConfiguration(
  targetClient: LanguageClient | undefined = client
) {
  if (!targetClient || targetClient !== client || !targetClient.isRunning()) {
    return;
  }
  try {
    await targetClient.sendNotification("workspace/didChangeConfiguration", {
      settings: dynareConfigurationPayload(),
    });
  } catch (err) {
    if (targetClient === client) {
      console.error("Failed to send Dynare language server configuration", err);
    }
  }
}

function createClient(pythonPath: string): LanguageClient {
  const serverOptions: ServerOptions = {
    command: pythonPath,
    args: ["-m", "dynare_lsp"],
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file", language: "dynare" }],
    synchronize: {
      fileEvents: [
        workspace.createFileSystemWatcher("**/*.mod"),
        workspace.createFileSystemWatcher("**/*.inc"),
      ],
    },
  };

  return new LanguageClient(
    "dynareLSP",
    "Dynare Language Server",
    serverOptions,
    clientOptions
  );
}

async function stopClient(targetClient: LanguageClient): Promise<void> {
  if (!targetClient.isRunning()) {
    return;
  }
  try {
    await targetClient.stop();
  } catch (err) {
    console.warn("Failed to stop Dynare language client", err);
  }
}

function startClient(context: ExtensionContext, pythonPath: string): void {
  const generation = ++clientGeneration;
  currentPythonPath = pythonPath;
  const previousClient = client;
  const previousStart = clientStartPromise;

  // LanguageClient cannot stop while Starting. Let an older generation settle
  // and clean itself up before the newest generation creates another client.
  const lifecycle = (async () => {
    if (previousStart) {
      await previousStart;
    }
    if (generation !== clientGeneration) {
      return;
    }

    if (previousClient && client === previousClient) {
      await stopClient(previousClient);
      if (client === previousClient) {
        client = undefined;
      }
    }
    if (generation !== clientGeneration) {
      return;
    }

    const nextClient = createClient(pythonPath);
    client = nextClient;
    context.subscriptions.push(nextClient);

    try {
      await nextClient.start();
    } catch (err) {
      if (client === nextClient) {
        client = undefined;
      }
      console.error("Dynare language server failed to start", err);
      return;
    }

    if (generation !== clientGeneration || client !== nextClient) {
      await stopClient(nextClient);
      if (client === nextClient) {
        client = undefined;
      }
      return;
    }

    await pushDynareConfiguration(nextClient);
  })();

  let trackedStart: Promise<void>;
  trackedStart = lifecycle
    .catch((err) => {
      console.error("Dynare language client lifecycle failed", err);
    })
    .finally(() => {
      if (clientStartPromise === trackedStart) {
        clientStartPromise = undefined;
      }
    });
  clientStartPromise = trackedStart;
}

function maybeStartClientForDocument(
  context: ExtensionContext,
  document?: vscode.TextDocument
): void {
  if (
    client ||
    clientStartPromise ||
    !document ||
    document.languageId !== "dynare"
  ) {
    return;
  }
  startClient(context, configuredPythonPath());
}

function maybeStartClientForOpenDynareDocument(context: ExtensionContext): void {
  const document = workspace.textDocuments.find((doc) => doc.languageId === "dynare");
  maybeStartClientForDocument(context, document);
}

async function restartClient(context: ExtensionContext): Promise<boolean> {
  const nextPythonPath = configuredPythonPath();
  if (
    nextPythonPath === currentPythonPath &&
    (client !== undefined || clientStartPromise !== undefined)
  ) {
    return false;
  }

  startClient(context, nextPythonPath);
  return true;
}

function renderIncludeTrace(result: any): string {
  if (!result || result.success === false) {
    return `Include trace failed: ${result?.message ?? "unknown error"}`;
  }
  const counts = result.counts ?? {};
  const lines: string[] = [
    "Dynare macro/include resolution trace",
    `Root: ${result.root ?? ""}`,
    `Events: ${result.n_events ?? 0}; resolved ${counts.resolved ?? 0}; ` +
      `unresolved ${counts.unresolved ?? 0}; inactive ${counts.inactive ?? 0}; ` +
      `cycles ${counts.cycle ?? 0}`,
    "",
  ];
  for (const event of result.events ?? []) {
    const indent = "  ".repeat(Math.max(0, Number(event.depth ?? 0)));
    lines.push(
      `${event.sequence}. ${indent}` +
        `[${String(event.status ?? "unknown").toUpperCase()}] ` +
        `${event.including_file}:${event.line}:${event.column}`,
      `${indent}   ${event.directive ?? `@#include ${event.filename ?? ""}`}`,
      `${indent}   expanded: ${event.expanded_filename ?? event.filename ?? ""}`,
      `${indent}   selected: ${event.resolved_path ?? "none"}` +
        `${event.resolution_kind ? ` (${event.resolution_kind})` : ""}`,
    );
    const defines = Object.entries(event.visible_defines ?? {});
    if (defines.length > 0) {
      lines.push(
        `${indent}   defines: ${defines
          .map(([name, value]) => `${name}=${String(value)}`)
          .join(", ")}`
      );
    }
    for (const attempt of event.attempts ?? []) {
      const marker = attempt.selected ? "*" : "-";
      const state = attempt.on_disk
        ? "disk"
        : attempt.known_virtual
        ? "virtual"
        : "missing";
      lines.push(
        `${indent}   ${marker} ${attempt.kind}: ${attempt.path} [${state}]` +
          `${attempt.detail ? ` — ${attempt.detail}` : ""}`
      );
    }
    lines.push("");
  }
  return lines.join("\n");
}

function renderAnalysisProfile(result: any): string {
  if (!result || result.success === false) {
    return `Analysis profile failed: ${result?.message ?? "unknown error"}`;
  }
  const lines: string[] = [
    "Dynare deterministic analysis profile",
    `Active file: ${result.active_file ?? ""}`,
    `Supplied files: ${result.n_files_supplied ?? 0}; ` +
      `included files: ${result.n_files_included ?? 0}`,
    `Diagnostics: ${result.n_diagnostics ?? 0}; ` +
      `total operation units: ${result.total_operation_units ?? 0}`,
    `Scope: ${result.scope ?? ""}`,
    "",
    "Work units:",
  ];
  const units = Object.entries(result.work_units ?? {}).sort(([a], [b]) =>
    a.localeCompare(b)
  );
  for (const [name, value] of units) {
    lines.push(`- ${name}: ${value}`);
  }
  return lines.join("\n");
}

function registerAnalysisCommands(context: ExtensionContext): void {
  const output = vscode.window.createOutputChannel("Dynare Analysis");

  async function run(command: string, title: string, render: (value: any) => string) {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== "dynare") {
      void vscode.window.showInformationMessage(
        `${title} requires an active Dynare .mod or .inc document.`
      );
      return;
    }
    try {
      maybeStartClientForDocument(context, editor.document);
      if (clientStartPromise) {
        await clientStartPromise;
      }
      const targetClient = client;
      if (!targetClient || !targetClient.isRunning()) {
        void vscode.window.showErrorMessage(
          "Dynare language server is not running. Check dynare.pythonPath."
        );
        return;
      }
      const result = await targetClient.sendRequest<any>("workspace/executeCommand", {
        command,
        arguments: [{ uri: editor.document.uri.toString() }],
      });
      output.clear();
      output.appendLine(render(result));
      output.show(true);
    } catch (err) {
      output.clear();
      output.appendLine(`${title} failed: ${String(err)}`);
      output.show(true);
      void vscode.window.showErrorMessage(`${title} failed; see Dynare Analysis output.`);
    }
  }

  context.subscriptions.push(
    output,
    vscode.commands.registerCommand("dynare.traceIncludes", () =>
      run("dynare/traceIncludes", "Trace Dynare Includes", renderIncludeTrace)
    ),
    vscode.commands.registerCommand("dynare.profileAnalysis", () =>
      run("dynare/profileAnalysis", "Profile Dynare Analysis", renderAnalysisProfile)
    )
  );
}

function registerMcpProvider(context: ExtensionContext): void {
  // Register the bundled MCP server only when the host supports the stable API.
  const lm: any = (vscode as any).lm;
  const Def: any = (vscode as any).McpStdioServerDefinition;
  if (!lm || typeof lm.registerMcpServerDefinitionProvider !== "function" || typeof Def !== "function") return;
  const didChange = new vscode.EventEmitter<void>();
  context.subscriptions.push(
    didChange,
    lm.registerMcpServerDefinitionProvider("dynare.mcpServerProvider", {
      onDidChangeMcpServerDefinitions: didChange.event,
      provideMcpServerDefinitions: () => {
        const command = configuredPythonPath();
        const args = ["-m", "dynare_lsp.mcp_preflight_server"];
        let def: any = new Def("Dynare MCP", command, args);
        if (!def || def.command !== command) def = new Def({ label: "Dynare MCP", command, args });
        return [def];
      },
    }),
    workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("dynare.pythonPath")) didChange.fire();
    })
  );
}

export function activate(context: ExtensionContext) {
  registerMcpProvider(context);
  registerAnalysisReport(context, async () => {
    maybeStartClientForOpenDynareDocument(context);
    if (clientStartPromise) await clientStartPromise;
    return client;
  });
  // Preserve the old command ID, but route it to actual versioned evidence.
  context.subscriptions.push(vscode.commands.registerCommand("dynare.showPreflightStatus", (uri?: vscode.Uri) =>
    vscode.commands.executeCommand("dynare.showAnalysisReport", uri)));
  registerAnalysisCommands(context);
  maybeStartClientForOpenDynareDocument(context);
  context.subscriptions.push(
    workspace.onDidOpenTextDocument((document) => {
      maybeStartClientForDocument(context, document);
    }),
    workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("dynare.pythonPath")) {
        if (!client && !clientStartPromise) {
          maybeStartClientForOpenDynareDocument(context);
          return;
        }
        void (async () => {
          const restarted = await restartClient(context);
          if (!restarted && event.affectsConfiguration("dynare")) {
            await pushDynareConfiguration();
          }
        })();
        return;
      }
      if (event.affectsConfiguration("dynare")) {
        void pushDynareConfiguration();
      }
    }),
    workspace.onDidChangeWorkspaceFolders(() => {
      void pushDynareConfiguration();
    })
  );
}

export function deactivate(): Thenable<void> | undefined {
  const activeClient = client;
  const pendingStart = clientStartPromise;
  clientGeneration += 1;
  client = undefined;
  clientStartPromise = undefined;
  if (!activeClient && !pendingStart) return undefined;
  return (async () => {
    if (pendingStart) await pendingStart;
    if (activeClient) await stopClient(activeClient);
  })();
}
