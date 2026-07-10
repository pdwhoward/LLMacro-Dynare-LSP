import * as path from "path";
import * as vscode from "vscode";
import { workspace, ExtensionContext, WorkspaceFolder } from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
} from "vscode-languageclient/node";

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

  if (!activeClient && !pendingStart) {
    return undefined;
  }
  return (async () => {
    if (pendingStart) {
      await pendingStart;
    }
    if (activeClient) {
      await stopClient(activeClient);
    }
  })();
}
