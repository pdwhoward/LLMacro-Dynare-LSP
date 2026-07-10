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
exports.activate = activate;
exports.deactivate = deactivate;
const path = __importStar(require("path"));
const vscode = __importStar(require("vscode"));
const vscode_1 = require("vscode");
const node_1 = require("vscode-languageclient/node");
let client;
let clientStartPromise;
let clientGeneration = 0;
let currentPythonPath = "python";
function configuredPythonPath() {
    const config = vscode_1.workspace.getConfiguration("dynare");
    return config.get("pythonPath", "python");
}
function resolveSearchPath(entry, workspaceFolder) {
    const trimmed = entry.trim();
    if (!trimmed || path.isAbsolute(trimmed)) {
        return trimmed;
    }
    if (!workspaceFolder) {
        return trimmed;
    }
    return path.join(workspaceFolder.uri.fsPath, trimmed);
}
function configuredSearchPaths() {
    const config = vscode_1.workspace.getConfiguration("dynare");
    // Relative entries belong only in searchPathsByRoot, where each workspace
    // folder resolves them against itself instead of leaking them across roots.
    const absolute = config
        .get("searchPaths", [])
        .map((entry) => entry.trim())
        .filter((entry) => entry.length > 0 && path.isAbsolute(entry));
    return Array.from(new Set(absolute));
}
function configuredSearchPathsByRoot() {
    const workspaceFolders = vscode_1.workspace.workspaceFolders;
    const pathsByRoot = {};
    if (!workspaceFolders || workspaceFolders.length === 0) {
        return pathsByRoot;
    }
    for (const workspaceFolder of workspaceFolders) {
        const config = vscode_1.workspace.getConfiguration("dynare", workspaceFolder.uri);
        const resolved = [];
        for (const entry of config.get("searchPaths", [])) {
            resolved.push(resolveSearchPath(entry, workspaceFolder));
        }
        pathsByRoot[workspaceFolder.uri.fsPath] = Array.from(new Set(resolved));
    }
    return pathsByRoot;
}
function dynareConfigurationPayload() {
    const config = vscode_1.workspace.getConfiguration("dynare");
    return {
        dynare: {
            pythonPath: config.get("pythonPath", "python"),
            steadyStateTolerance: config.get("steadyStateTolerance", 1e-6),
            preprocessorPath: config.get("preprocessorPath", ""),
            formatIndent: config.get("formatIndent", "tab"),
            searchPaths: configuredSearchPaths(),
            searchPathsByRoot: configuredSearchPathsByRoot(),
        },
    };
}
async function pushDynareConfiguration(targetClient = client) {
    if (!targetClient || targetClient !== client || !targetClient.isRunning()) {
        return;
    }
    try {
        await targetClient.sendNotification("workspace/didChangeConfiguration", {
            settings: dynareConfigurationPayload(),
        });
    }
    catch (err) {
        if (targetClient === client) {
            console.error("Failed to send Dynare language server configuration", err);
        }
    }
}
function createClient(pythonPath) {
    const serverOptions = {
        command: pythonPath,
        args: ["-m", "dynare_lsp"],
    };
    const clientOptions = {
        documentSelector: [{ scheme: "file", language: "dynare" }],
        synchronize: {
            fileEvents: [
                vscode_1.workspace.createFileSystemWatcher("**/*.mod"),
                vscode_1.workspace.createFileSystemWatcher("**/*.inc"),
            ],
        },
    };
    return new node_1.LanguageClient("dynareLSP", "Dynare Language Server", serverOptions, clientOptions);
}
async function stopClient(targetClient) {
    if (!targetClient.isRunning()) {
        return;
    }
    try {
        await targetClient.stop();
    }
    catch (err) {
        console.warn("Failed to stop Dynare language client", err);
    }
}
function startClient(context, pythonPath) {
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
        }
        catch (err) {
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
    let trackedStart;
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
function maybeStartClientForDocument(context, document) {
    if (client ||
        clientStartPromise ||
        !document ||
        document.languageId !== "dynare") {
        return;
    }
    startClient(context, configuredPythonPath());
}
function maybeStartClientForOpenDynareDocument(context) {
    const document = vscode_1.workspace.textDocuments.find((doc) => doc.languageId === "dynare");
    maybeStartClientForDocument(context, document);
}
async function restartClient(context) {
    const nextPythonPath = configuredPythonPath();
    if (nextPythonPath === currentPythonPath &&
        (client !== undefined || clientStartPromise !== undefined)) {
        return false;
    }
    startClient(context, nextPythonPath);
    return true;
}
function registerMcpProvider(context) {
    // Register the bundled `dynare-mcp` server with VS Code's MCP support (agent
    // mode in Chat) so installing this extension exposes BOTH the language server
    // and the MCP analysis tools — one engine behind two transports. The MCP
    // server-definition API arrived in VS Code 1.101; feature-detect it (and cast
    // through `any`) so the extension still loads on older VS Code and compiles
    // against @types/vscode that predate the API — the MCP server is just skipped.
    const lm = vscode.lm;
    const Def = vscode.McpStdioServerDefinition;
    if (!lm ||
        typeof lm.registerMcpServerDefinitionProvider !== "function" ||
        typeof Def !== "function") {
        return;
    }
    const didChange = new vscode.EventEmitter();
    context.subscriptions.push(lm.registerMcpServerDefinitionProvider("dynare.mcpServerProvider", {
        onDidChangeMcpServerDefinitions: didChange.event,
        provideMcpServerDefinitions: () => {
            const command = configuredPythonPath();
            const args = ["-m", "dynare_lsp.mcp_server"];
            // The stable API uses a positional constructor
            // (label, command, args, env?, version?). Build positionally, then
            // fall back to an options-object form if a build expects that instead,
            // so this does not depend on a single constructor shape.
            let def = new Def("Dynare MCP", command, args);
            if (!def || def.command !== command) {
                def = new Def({ label: "Dynare MCP", command, args });
            }
            return [def];
        },
    }), vscode_1.workspace.onDidChangeConfiguration((event) => {
        if (event.affectsConfiguration("dynare.pythonPath")) {
            didChange.fire();
        }
    }));
}
function activate(context) {
    registerMcpProvider(context);
    maybeStartClientForOpenDynareDocument(context);
    context.subscriptions.push(vscode_1.workspace.onDidOpenTextDocument((document) => {
        maybeStartClientForDocument(context, document);
    }), vscode_1.workspace.onDidChangeConfiguration((event) => {
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
    }), vscode_1.workspace.onDidChangeWorkspaceFolders(() => {
        void pushDynareConfiguration();
    }));
}
function deactivate() {
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
//# sourceMappingURL=extension.js.map