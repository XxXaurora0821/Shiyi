# 拾忆 · Shiyi

> 拾起对话里散落的记忆碎片，拼成一个会成长的用户画像。

一个从零实现的简化版 AI Runtime：多轮对话 + 分层用户记忆 + 动态特征扩展 + 检索。
架构与 trade-off 见 [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md)。

## 快速启动

```bash
./start.sh                 
# 自动建 venv、装依赖、起 8000 端口
# 然后浏览器打开 http://localhost:8000
```

或手动：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

> **可选**：启用真实 OpenAI 链路（chat + embeddings）有两种方式：
> 1. 复制 `.env.example` 为 `.env` 并填入 `OPENAI_API_KEY=sk-...`（启动时自动加载）；
> 2. 或直接 `export OPENAI_API_KEY=sk-...` 后再启动。
>
> **未设置**时全链路自动降级到本地 mock + 哈希嵌入，抽取/合并/检索/拼装仍可离线跑通。

## 跑测试

```bash
python3 backend/tests/test_smoke.py
```

涵盖：核心档案抽取、L1 合并、检索召回、容量淘汰、否定语义。

## 体验脚本

打开页面后依次发：

1. `你好我叫小红，今年22岁，是女生，喜欢摇滚乐和摄影，常熬夜爱点外卖。`
   — 看右侧 L0 档案 + 4 条 L1 特征即时出现。
2. `周末又约朋友去摄影了，超开心。`
   — `摄影` 特征 hit_count 变 2，置信度上调，未新建。
3. `推荐点周末活动？`
   — debug 区显示检索召回了"摄影 / 摇滚乐"等条目，被注入到 system prompt。

## 仓库结构

```
backend/        # FastAPI 后端 + 记忆/抽取/检索
  config.py     # 全部可调旋钮
  db.py         # SQLite schema
  embeddings.py # 嵌入（OpenAI / 哈希降级 + 缓存）
  llm.py        # LLM（OpenAI / mock）+ 速率限制
  extractor.py  # 规则抽 L0 + LLM 抽 L1
  features.py   # 合并 / 容量 / 打分
  memory.py     # 外观：用户/会话/抽取/摘要
  retrieval.py  # 特征 + 片段检索打分
  chat.py       # prompt 拼装 + 编排
  main.py       # FastAPI 入口
  tests/test_smoke.py
frontend/index.html   # 单页聊天界面
docs/                 # 架构 / 技术 / API 文档
start.sh              # 一键启动
```

## 文档

- [架构设计](./docs/ARCHITECTURE.md) — 模块、数据流、Memory 设计、Feature 扩展、成本策略
- [技术说明](./docs/TECH.md) — 选型理由、数据结构、检索打分、局限
- [API 文档](./docs/API.md) — 三个 HTTP 接口与字段说明
