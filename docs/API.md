# 拾忆 · API 文档

服务端口：默认 `8000`（`uvicorn backend.main:app --port 8000`）。
所有接口返回 JSON，错误以 HTTP 4xx/5xx + `{"detail": "..."}` 表达。

---

## `POST /api/chat`

发起一轮对话。会同步执行：消息入库 → 抽取（L0 规则 + L1 LLM/mock）→ 检索 → 拼 prompt → 调 LLM → 回复入库 → 可能落片段。

### Request body
```json
{
  "user_id": "demo_user",
  "session_id": "（可选；缺省自动新建）",
  "message": "你好我叫小明，今年25岁，喜欢打篮球。"
}
```

### Response
```json
{
  "session_id": "f3c7…",
  "reply": "（LLM 生成的回复）",
  "debug": {
    "profile": { "name": "小明", "age": 25, "gender": null },
    "retrieved_features": [
      { "id": 12, "category": "兴趣", "key": "篮球", "score": 0.412 }
    ],
    "retrieved_episodes": [],
    "history_turns": 4,
    "system_prompt_chars": 287
  }
}
```

`debug` 字段是前端的可观测面（展示本次抽取/检索结果），生产环境可关闭。

### 错误
- `400 message is empty`：消息为空白。

---

## `GET /api/memory/{user_id}`

取出用户的核心档案与全部 L1 特征（按 `last_seen` 倒序）。

### Response
```json
{
  "profile": { "name": "小明", "age": 25, "gender": "男" },
  "features": [
    {
      "feature_id": 12,
      "user_id": "demo_user",
      "category": "兴趣",
      "key": "篮球",
      "value": "对篮球有兴趣",
      "confidence": 0.75,
      "hit_count": 3,
      "created_at": 1714211200.1,
      "last_seen": 1714214400.4,
      "source_msg_id": 42
    }
  ]
}
```

---

## `GET /api/sessions/{session_id}/messages?limit=50`

按时间顺序取一个会话的最近 N 条消息。

### Response
```json
{
  "messages": [
    { "role": "user",      "content": "你好", "ts": 1714211200.0 },
    { "role": "assistant", "content": "你好，", "ts": 1714211200.4 }
  ]
}
```

---

## 调用流程（建议前端做法）

```
打开页面
  └─► refresh()  → GET /api/memory/{uid}        // 渲染右侧档案 + 特征
循环：用户输入消息
  ├─► POST /api/chat                            // 收 reply 与 debug
  ├─► 把 user / assistant 两条消息追加进聊天区
  └─► refresh()                                 // 同步右侧档案/特征/debug
新会话按钮
  └─► sessionId = null；刷新
```

---

## 环境变量

| 变量 | 说明 | 默认 |
|------|------|------|
| `OPENAI_API_KEY` | 设置即启用 OpenAI 真实链路；未设置走本地 mock | 空 |
| `OPENAI_CHAT_MODEL` | OpenAI Chat 模型 | `gpt-4o-mini` |
| `OPENAI_EMBED_MODEL` | OpenAI Embedding 模型 | `text-embedding-3-small` |
| `DB_PATH` | SQLite 文件位置 | `./data/runtime.db` |
