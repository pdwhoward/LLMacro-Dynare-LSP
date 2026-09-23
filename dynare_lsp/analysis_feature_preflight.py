"""Expose the consolidated preflight through the bounded analysis catalog."""
from .mcp_preflight_server import dynare_preflight

TOOL = dynare_preflight
COMMAND = "dynare/preflight"
CLI = "preflight"
EXAMPLE = {"file_content": "var y; model(linear); y = 0.5*y(-1); end;"}
