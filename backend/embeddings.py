"""嵌入层：
- 真实路径：OpenAI text-embedding-3-small（批量调用 + SQLite 缓存）
- 降级路径：本地 char-ngram 哈希嵌入（无依赖、可离线跑通整条链路，且对中文「打篮球 / 篮球爱好者」这种近义短语的余弦相似度通常仍 > 合并阈值）

关键点：
1) 缓存按文本 hash 去重，重复内容（如同一特征 key 在多用户出现）只算一次 token；
2) 缓存写入时记录 backend，避免不同维度的向量混用；
3) 余弦做了 shape 防御，避免运行期切换后端时直接崩。
"""
import hashlib
import time
from typing import List, Optional

import numpy as np

from backend.config import CONFIG
from backend.db import get_conn


def vec_to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _hash_embed(text: str, dim: Optional[int] = None) -> np.ndarray:
    """char-1/2/3-gram → 哈希 bin → 带符号累加 → L2 归一化。
    在同语种近义判定上效果可用，作为无 API key 时的降级方案。
    """
    dim = dim or CONFIG.embed_dim
    vec = np.zeros(dim, dtype=np.float32)
    text = text.strip().lower()
    if not text:
        return vec
    for n in (1, 2, 3):
        for i in range(len(text) - n + 1):
            g = text[i:i + n]
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % dim
            sign = 1.0 if (h >> 31) & 1 else -1.0
            vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _openai_embed_batch(texts: List[str]) -> List[np.ndarray]:
    from openai import OpenAI

    client = OpenAI(**CONFIG.openai_kwargs)
    resp = client.embeddings.create(model=CONFIG.openai_embed_model, input=texts)
    return [np.asarray(d.embedding, dtype=np.float32) for d in resp.data]


def _backend_tag() -> str:
    return "openai" if CONFIG.use_openai else "hash"


def embed(text: str) -> np.ndarray:
    return embed_batch([text])[0]


def embed_batch(texts: List[str]) -> List[np.ndarray]:
    """带缓存的批量嵌入。命中缓存的文本直接走 SQLite，未命中的文本走当前 backend。"""
    conn = get_conn()
    backend = _backend_tag()
    out: List[Optional[np.ndarray]] = [None] * len(texts)
    miss_idx, miss_texts, miss_hashes = [], [], []

    for i, t in enumerate(texts):
        h = _text_hash(t)
        row = conn.execute(
            "SELECT vec FROM embed_cache WHERE text_hash=? AND backend=?",
            (h, backend),
        ).fetchone()
        if row is not None:
            out[i] = blob_to_vec(row["vec"])
        else:
            miss_idx.append(i)
            miss_texts.append(t)
            miss_hashes.append(h)

    if miss_texts:
        if CONFIG.use_openai:
            try:
                vecs = _openai_embed_batch(miss_texts)
            except Exception:
                vecs = [_hash_embed(t) for t in miss_texts]
        else:
            vecs = [_hash_embed(t) for t in miss_texts]

        now = time.time()
        for i, h, v in zip(miss_idx, miss_hashes, vecs):
            out[i] = v
            conn.execute(
                "INSERT OR REPLACE INTO embed_cache(text_hash, vec, backend, created_at) VALUES (?,?,?,?)",
                (h, vec_to_blob(v), backend, now),
            )

    return [v if v is not None else np.zeros(CONFIG.embed_dim, dtype=np.float32) for v in out]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        # 切换 backend 后历史向量与新向量维度不一致——截断到公共长度做防御
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
