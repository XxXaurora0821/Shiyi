# 拾忆 · 架构设计文档

## 1. 总体架构

```
┌────────────────────────────────────────────────────────────────────┐
│ Frontend (单 HTML+JS)                                              │
│  - 聊天 UI                                                          │
│  - 实时显示 L0 档案 / L1 特征 / 检索 debug                          │
└──────────────────────────┬─────────────────────────────────────────┘
                           │ HTTP/JSON
┌──────────────────────────▼─────────────────────────────────────────┐
│ FastAPI Backend                                                    │
│                                                                    │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │ Chat Runtime  (chat.py / main.py)                        │     │
│   │   - 消息入库                                              │     │
│   │   - 调度抽取 → 检索 → prompt 拼装                         │     │
│   │   - 调用 LLM（带速率限制 + 失败降级）                     │     │
│   └─────┬───────────────────────────┬─────────────────────────┘     │
│         │                           │                               │
│   ┌─────▼──────────────┐    ┌───────▼───────────────────┐           │
│   │ Memory Service     │    │ Retrieval                 │           │
│   │   extractor.py     │    │   retrieval.py            │           │
│   │   features.py      │    │   - top-K 特征 + 片段      │           │
│   │   memory.py        │    │   - 余弦 + 新近度 + 置信度 │           │
│   └─────┬──────────────┘    └───────┬───────────────────┘           │
│         │                           │                               │
│   ┌─────▼───────────────────────────▼─────────────────────┐         │
│   │ Storage (SQLite, 单文件)                              │         │
│   │   users / sessions / messages                         │         │
│   │   core_profile (L0)                                   │         │
│   │   features      (L1)                                  │         │
│   │   episodes      (L2)                                  │         │
│   │   embeddings    (向量, BLOB)                          │         │
│   │   embed_cache   (按文本 hash 去重的嵌入缓存)          │         │
│   └───────────────────────────────────────────────────────┘         │
│                                                                    │
│   ┌────────────────────────────────────────────────────────┐       │
│   │ LLM / Embeddings 抽象 (llm.py / embeddings.py)         │       │
│   │   - 真实路径：OpenAI Chat + Embedding                  │       │
│   │   - 降级路径：本地 mock chat + char-ngram 哈希嵌入     │       │
│   │   - 二者同接口，运行时按 OPENAI_API_KEY 选择           │       │
│   └────────────────────────────────────────────────────────┘       │
└────────────────────────────────────────────────────────────────────┘
```

## 2. 数据流：从用户输入到回复

```
[POST /api/chat]
   │
   ▼
1) ensure_user / ensure_session                       memory.py
2) 用户消息入库                                       memory.append_message
3) 抽取 L0：正则 → 直接更新 core_profile              extractor.update_core_profile_from_rules
4) 抽取 L1（mock 或 LLM）→ 候选特征                   extractor.extract_l1_candidates
5) 每条候选 upsert_feature：嵌入 → 同类相似度合并     features.upsert_feature
   - sim ≥ 阈值: hit_count+1, last_seen=now, conf+0.05
   - 否则: 新建 + 写入 embeddings 表
   - 超出类别上限按分数淘汰
6) 拼装 prompt:                                       chat.assemble_prompt
   - SYSTEM_HEADER
   - 【用户档案】L0 一行
   - 【相关记忆】retrieve_features(query) top-K
   - 【过往片段】retrieve_episodes(query) top-K
   - 最近 N 轮 history（去重当前用户消息）
   - 当前用户消息
7) 调 LLM 生成回复                                    llm.chat
8) 回复入库；每 N 个用户回合落一条情景摘要 + 向量    memory.maybe_summarize_episode
9) 返回 reply + debug
```

## 3. Memory 设计
三层记忆：

| 层级 | 表          | 用途                | 写入时机                          | 读取时机           |
|------|-------------|---------------------|-----------------------------------|--------------------|
| L0   | core_profile | 固定结构 {name,age,gender} | 规则命中即写        | 每次拼 prompt      |
| L1   | features     | 开放结构、横向扩展  | LLM/mock 抽取后 upsert_with_merge | 每次拼 prompt（top-K） |
| L2   | episodes     | 对话片段摘要（向量化） | 每 N 用户回合 1 次               | 每次拼 prompt（top-K） |

**为什么这样切层？**
- L0 字段少且高价值（每次 prompt 都注入），固定 schema 让规则路径即可保证准确；
- L1 是开放空间——必须靠模型/关键词抽取，且每用户量级有限（百级）：直接全表余弦扫即可，<5ms；
- L2 把多轮对话压成一条摘要向量，避免 history 无限增长；命中再展开，节流 token。

**candidate → feature 合并**：
1. embed(`category: key`) 得到候选向量；
2. 在该用户该类别下取所有现有特征向量做余弦最大值匹配；
3. 若 max_sim ≥ `sim_merge_threshold`（默认 0.82）→ 合并到已有项；
4. 否则新建；
5. `value` 字段以"更长的为准"——多次提及自然倾向于更详细的描述会留下。

**生命周期**：
- 每个 (user, category) 维护一个上限 `feature_cap_per_category`（默认 50）；
- 超限时按 `score = log(1+hit_count) * exp(-Δdays/30) * confidence` 升序淘汰；
- 命中即抬升 confidence（封顶 0.99），衰减仅在打分时计算，不需要后台 job。

## 4. Feature 扩展策略

围绕四个核心问题展开：

### Q1：如何从非结构化对话中抽取新特征？
**两层抽取**：
- **规则路径**（永远开启，sub-ms）：处理 L0 三个固定字段以及一组高频关键词（兴趣/习惯/消费/关系）。规则的好处是"绝对可控"——不会幻觉、不会乱归类。
- **LLM 路径**（按字符长度、回合数节流）：用一段定死格式的 system prompt 让模型只输出 JSON。无 API key 时降级到本地 mock，仍能跑通。

### Q2：如何避免特征爆炸？
- **类别封顶**：每个类别最多 N 条，超出按 `log(hit) × recency × confidence` 升序淘汰；
- **合并优先**：相似度合并永远先于新建，本质上把同义短语压成一条；
- **节流 LLM 抽取**：太短的消息根本不进入 LLM 抽取（避免噪声特征）；
- **类别白名单**：只允许 `兴趣 / 习惯 / 消费 / 关系 / 人口 / 其他`，越界统一归为"其他"。

### Q3：如何归类与合并？
- 类别由 LLM/规则在抽取阶段直接给出，限定在白名单内；
- 同类内合并基于 `embed(key)` 的余弦相似度：例如 `embed("兴趣: 打篮球")` 与 `embed("兴趣: 篮球爱好者")` 高度相似，自然合并。
- 阈值 0.82 是 trade-off 旋钮：调低则更激进合并（可能误并），调高则更碎片（可能爆炸）。

### Q4：数据结构如何支持快速检索？
- 索引：`(user_id, category)`、`(user_id, last_seen)`；
- 向量分表：embeddings 表按 (ref_type, ref_id) 主键，避免与业务字段混在一起；
- 查询路径：先按 user_id 筛特征 → 加载向量 → 内存内余弦排序。单用户百级特征下，全表扫足够。

## 5. 检索系统

`retrieve_features(user, query, top_k)`：
1. embed(query)；
2. 该 user 下所有特征向量做余弦；
3. 综合打分 `0.7 × sim + 0.2 × recency + 0.1 × confidence`；
4. 取 top-K（默认 8）。

`retrieve_episodes` 类似，仅按相似度排序。

**为什么不直接用专业向量库？** 单用户量级在百级；开 ANN（hnsw 等）的常数开销远大于全表余弦。真到百万用户时，按 `user_id` 水平分片即可——每片仍是百级特征，扫全表毫秒级。这是水平扩展友好的设计。

## 6. 成本优化策略（必须考虑）

| 决策 | 节省的是什么 |
|------|--------------|
| 规则路径优先处理 L0 | L0 字段不再需要任何 LLM 调用 |
| 长度+回合数双门槛节流 LLM 抽取 | 短闲聊不触发抽取 |
| Embedding 走 SQLite cache（按文本 hash） | 重复短语零调用 |
| Episode 每 N 轮压缩 1 条 | history 不会无限增长，prompt token 受控 |
| Top-K 特征/片段截断 | system prompt 长度上限可预测 |
| LLM/Embedding 失败时静默降级到本地 mock | 单点失败不导致整条链路挂掉 |
| feature `value` "更长者覆盖" | 不需要多余 LLM 调用做摘要更新 |

**总开销近似**：
- 每条用户消息：≤1 次 chat call + ≤1 次 extract call + 至多 N（候选数，通常 0-3）次 embed cache miss。
- 重复说"我喜欢篮球"第二次：0 次 LLM 调用，仅 embedding 缓存命中 + DB 写入。

## 7. 高可用与可扩展性

- **单点失败防御**：LLM/Embedding 调用都被 try/except 包住，外部失败一律降级到本地 mock，业务不中断。
- **持久化**：SQLite WAL 模式，同步写入；进程崩溃后下次启动可继续。
- **水平扩展路径**（设计中预留，未实现）：
  - SQLite → Postgres / 分库；
  - embeddings 表 → Milvus / pgvector；
  - 抽取从同步改成消息队列异步（chat 关键路径不被阻塞）；
  - 关键路径无 cross-user 操作，按 user_id 哈希分片即可线性扩展。

## 8. Trade-off 摘要

| 选择 | 替代方案 | 为什么这么选 |
|------|----------|--------------|
| SQLite 单文件 | Postgres / Redis / Milvus | 单机零运维；接口设计兼容未来迁移 |
| 哈希嵌入降级 | 必须依赖外部模型 | 无 API key 时仍可离线跑通整条链路 |
| 同步抽取 | 异步队列 | 实现复杂度大幅增加；对单机毫秒级链路无收益 |
| 余弦阈值合并 | LLM 仲裁是否同义 | 节省成本一个量级；阈值可调 |
| 规则 + 模型混合抽取 | 纯模型 / 纯规则 | 纯规则覆盖差，纯模型贵且不稳定 |
| 类别白名单 | 自由生成 | 控制特征空间，便于检索与展示 |
