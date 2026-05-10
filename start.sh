#!/usr/bin/env bash
# 一键启动：创建 venv、装依赖、起 uvicorn。
# 设置 OPENAI_API_KEY 即走真实 LLM；不设置则使用本地 mock 也能完整跑通。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r backend/requirements.txt

mkdir -p data
echo
echo "拾忆 · Shiyi  →  http://localhost:8000"
echo "  - 浏览器打开 / 即可看到聊天界面"
echo "  - 未设置 OPENAI_API_KEY 时走本地 mock，仍能完整演示链路"
echo
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
