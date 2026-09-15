"""JSON CLI for the same versioned analysis functions used by LSP and MCP."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .analysis_api import features
from .analysis_service import json_safe


def main(argv=None) -> int:
    choices = {m.CLI: m.TOOL for m in features()}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature", choices=sorted(choices))
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--request", type=Path, help="JSON arguments file; default is stdin")
    inputs.add_argument("--file", type=Path, help="Explicit single source file; includes are not implicitly read")
    args = parser.parse_args(argv)
    try:
        if args.file:
            with args.file.open(encoding="utf-8", newline="") as stream:
                request = {"file_content": stream.read(), "active_file": str(args.file.resolve())}
        else:
            request = json.loads(args.request.read_text(encoding="utf-8-sig") if args.request else sys.stdin.read())
        if not isinstance(request, dict):
            raise ValueError("Request must be a JSON object")
        result = json_safe(choices[args.feature](**request))
        print(json.dumps(result, ensure_ascii=True, allow_nan=False, indent=2))
        return 0 if result.get("checks_passed", result.get("success", True)) else 1
    except Exception as exc:
        print(json.dumps({"success": False, "error": type(exc).__name__, "message": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
