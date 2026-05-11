#!/usr/bin/env bash
# One-command end-to-end demo:
#   query -> input guardrail -> 4 agents -> output guardrail -> judge -> export
#
# Usage:
#   ./run_demo.sh                          # uses the default HCI query
#   ./run_demo.sh "your custom query"      # custom query
#
# Outputs (under outputs/):
#   session_<ts>.json          full transcript + judge scores
#   judge_raw_<ts>.jsonl       raw judge prompts + LLM responses (one line per call)

set -euo pipefail

QUERY="${1:-What are the key principles of accessible user interface design?}"

# Pick the Python interpreter: prefer the local venv if it exists.
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  PY="python"
fi

echo "========================================"
echo "  Multi-Agent Research Assistant — demo"
echo "========================================"
echo "Interpreter: $PY"
echo "Query:       $QUERY"
echo

if [[ ! -f ".env" ]]; then
  echo "WARNING: .env not found. Copy .env.example and fill in API keys first."
fi

"$PY" main.py --mode demo --query "$QUERY"

echo
echo "----------------------------------------"
echo "Latest artefacts:"
ls -lt outputs/session_*.json outputs/judge_raw_*.jsonl 2>/dev/null | head -4
