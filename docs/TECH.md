# 拾忆 · 技术说明

## 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| 后端框架 | FastAPI + uvicorn | 异步、Pydantic 验证、自动 OpenAPI；轻量到不需要框架级抽象 |
| 数据库 | SQLite (WAL) | 单文件零运维；支持索引、事务；接口与 Postgres 兼容，迁移代价低 |
| 向量存储 | 同库的 embeddings 表（BLOB） | 单用户特征量级在百级，全表余弦扫描即可；引入向量库是过度设计 |
| 嵌入 | OpenAI text-embedding-3-small / 哈希降级 | 真实路径质量好；降级路径无网络/无密钥也能跑通 |
| LLM | OpenAI Chat (gpt-4o-mini) / mock 降级 | 同上 |
| 前端 | 单 HTML + 原生 JS | 不引入构建链路；浏览器直接打开即可 |
| 测试 | 标准库 + FastAPI TestClient | 不引入 pytest 依赖；`python backend/tests/test_smoke.py` 直跑 |

## 数据结构（重点）

### `core_profile` (L0)
固定 schema，每用户一行：
```
user_id PK, name, age, gender, updated_at
```
不放置任何动态字段，是为了让"基础档案"始终是 O(1) 查询、零检索成本的输入。

### `features` (L1)
```
feature_id PK, user_id, category, key, value,
confidence REAL, hit_count INTEGER, created_at, last_seen, source_msg_id
```
- `key`：规范化短语（≤8 字），用于嵌入与展示；
- `value`：详细说明（≤30 字），多次合并时取较长版本；
- `confidence`：每次合并 +0.05 封顶 0.99；
- `hit_count` + `last_seen`：用于打分与淘汰；
- `category`：白名单 6 选 1，便于分类容量控制。

索引：
- `(user_id, category)` —— 合并时按类别遍历；
- `(user_id, last_seen)` —— 调试/展示按时间排序。

### `episodes` (L2)
对话片段摘要 + 时间区间。每 N 轮压一条，量级远小于 messages，但保留语义可检索。

### `embeddings`
独立表，主键 `(ref_type, ref_id)`。把"一个 feature 是否有向量"和 features 表本身解耦——重建嵌入或换 backend 时只动这张表。

### `embed_cache`
按 `(text_hash, backend)` 做 KV，避免同一短语反复打嵌入。带 `backend` 字段，避免 OpenAI/哈希两套向量混入同一缓存。

## 抽取策略（规则 + 模型混合）

```
user_text
   │
   ├─► 正则规则（永远开启）─► 命中 L0 字段，直接 UPDATE core_profile
   │
   └─► LLM/mock 抽取（按门槛）
        - len(text) ≥ 16 字
        - turn % N == 0
        - 系统 prompt 强约束 JSON 输出
        ─► 候选 [CandidateFeature] ─► 逐条 upsert_feature
                                         │
                                         ├─ embed(category:key)
                                         ├─ 同类别已有特征余弦最大值
                                         ├─ ≥ 0.82 合并 / 否则新建
                                         └─ 超容量按 score 淘汰
```

混合的理由：纯规则在开放语义上漏抽率高；纯模型在固定 schema 上过贵且不稳。两者结合：低成本、高准确、又能扩展。

## 检索策略（如何快且准）

- **快**：单用户百级特征，向量预先持久化为 BLOB，加载 + 余弦内存内一次完成（<5ms 在本机实测）；
- **准**：相似度只占 70%，剩 30% 给新近度（指数衰减）和置信度。这样"一个月前提过一次"不会压过"昨天反复说过"。
- **可控**：top-K 截断 → 注入 prompt 的特征条数固定 → token 预算可预测。

## 成本控制策略（如何减少 token）

1. **缓存层**：相同短语的嵌入只算一次（embed_cache）；
2. **门槛**：短消息不进 LLM 抽取；
3. **截断**：top-K 限定注入 prompt 的记忆条数；
4. **压缩**：history 用 episodes 摘要替代旧轮次；
5. **降级**：上游失败一律走本地 mock，不丢消息也不爆 token；
6. **结构化输出**：抽取用 JSON mode，避免重复尝试。

## 局限与未来工作

- 抽取是同步的，关键路径会等 LLM。生产应改为消息队列异步抽取；
- mock 抽取覆盖窄，仅作为离线/降级场景的兜底；
- 相似度合并阈值是全局的，理想做法是按类别分别校准；
- 没有用户级的隐私撤回流（GDPR 删除）——补一条接口删 user_id 全部数据即可。
