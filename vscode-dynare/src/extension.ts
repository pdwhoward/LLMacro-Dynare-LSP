import * as path from "path";
import * as vscode from "vscode";
import { workspace, ExtensionContext, WorkspaceFolder } from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
} from "vscode-languageclient/node";

let client: LanguageClient | undefined;
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
  const resolved = config.get<string[]>("searchPaths", []).map((entry) => resolveSearchPath(entry));
  return Array.from(new Set(resolved));
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

async function pushDynareConfiguration() {
  if (!client || !client.isRunning()) {
    return;
  }
  await client.sendNotification("workspace/didChangeConfiguration", {
    settings: dynareConfigurationPayload(),
  });
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

function startClient(context: ExtensionContext, pythonPath: string) {
  currentPythonPath = pythonPath;
  client = createClient(pythonPath);
  context.subscriptions.push(client);
  void client.start().then(
    () => pushDynareConfiguration(),
    (err) => {
      // A failed start (e.g. wrong dynare.pythonPath) leaves the client in
      // the startFailed state; the LanguageClient surfaces its own error UI,
      // so just log here instead of leaking an unhandled rejection.
      console.error("Dynare language server failed to start", err);
    }
  );
}

function maybeStartClientForDocument(
  context: ExtensionContext,
  document?: vscode.TextDocument
): void {
  if (client || !document || document.languageId !== "dynare") {
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
  if (nextPythonPath === currentPythonPath) {
    return false;
  }

  if (client) {
    try {
      await client.stop();
    } catch (err) {
      // stop() throws for any state other than Running (notably
      // startFailed — exactly the state a corrected dynare.pythonPath is
      // meant to recover from). Discard the old client and start fresh.
      console.warn("Discarding Dynare language client that was not running", err);
    }
  }
  startClient(context, nextPythonPath);
  return true;
}

function registerMcpProvider(context: ExtensionContext): void {
  // Register the bundled `dynare-mcp` server with VS Code's MCP support (agent
  // mode in Chat) so installing this extension exposes BOTH the language server
  // and the MCP analysis tools — one engine behind two transports. The MCP
  // server-definition API arrived in VS Code 1.101; feature-detect it (and cast
  // through `any`) so the extension still loads on older VS Code and compiles
  // against @types/vscode that predate the API — the MCP server is just skipped.
  const lm: any = (vscode as any).lm;
  const Def: any = (vscode as any).McpStdioServerDefinition;
  if (
    !lm ||
    typeof lm.registerMcpServerDefinitionProvider !== "function" ||
    typeof Def !== "function"
  ) {
    return;
  }

  const didChange = new vscode.EventEmitter<void>();

  context.subscriptions.push(
    lm.registerMcpServerDefinitionProvider("dynare.mcpServerProvider", {
      onDidChangeMcpServerDefinitions: didChange.event,
      provideMcpServerDefinitions: () => {
        const command = configuredPythonPath();
        const args = ["-m", "dynare_lsp.mcp_server"];
        // The stable API uses a positional constructor
        // (label, command, args, env?, version?). Build positionally, then
        // fall back to an options-object form if a build expects that instead,
        // so this does not depend on a single constructor shape.
        let def: any = new Def("Dynare MCP", command, args);
        if (!def || def.command !== command) {
          def = new Def({ label: "Dynare MCP", command, args });
        }
        return [def];
      },
    }),
    workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("dynare.pythonPath")) {
        didChange.fire();
      }
    })
  );
}

export function activate(context: ExtensionContext) {
  registerMcpProvider(context);
  maybeStartClientForOpenDynareDocument(context);
  context.subscriptions.push(
    workspace.onDidOpenTextDocument((document) => {
      maybeStartClientForDocument(context, document);
    }),
    workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("dynare.pythonPath")) {
        if (!client) {
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
  if (!client) {
    return undefined;
  }
  // stop() throws (or rejects) for any state other than Running — e.g.
  // startFailed after a bad dynare.pythonPath. Swallow it so extension-host
  // shutdown never sees an unhandled rejection from us.
  try {
    return client.stop().then(undefined, () => undefined);
  } catch {
    return undefined;
  }
}
