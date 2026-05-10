"""特征 / 情景检索。

实现：把当前用户消息嵌入为查询向量，对已有特征/片段做余弦匹配，
特征再叠加新近度与置信度。返回 top-K，由调用方决定如何注入 prompt。

为什么不用专门向量库：
- 单用户特征量级在百级，全表扫足够（<5ms），引入向量库反而是过度设计；
- 真要扩到百万用户，按 user_id 分片 + 在每片内做 ANN 即可——不会改变这里的接口。
"""
from typing import List, Tuple

from backend.config import CONFIG
from backend.db import get_conn
from backend.embeddings import blob_to_vec, cosine, embed
from backend.features import feature_score


def retrieve_features(user_id: str, query: str, top_k: int = None) -> List[Tuple[dict, float]]:
    top_k = top_k or CONFIG.retrieval_top_k_features
    conn = get_conn()
    rows = conn.execute("SELECT * FROM features WHERE user_id=?", (user_id,)).fetchall()
    if not rows:
        return []
    qvec = embed(query)
    scored: List[Tuple[dict, float]] = []
    for r in rows:
        f = dict(r)
        ev = conn.execute(
            "SELECT vec FROM embeddings WHERE ref_type='feature' AND ref_id=?",
            (f["feature_id"],),
        ).fetchone()
        if ev is None:
            continue
        sim = cosine(qvec, blob_to_vec(ev["vec"]))
        scored.append((f, feature_score(f, sim)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def retrieve_episodes(user_id: str, query: str, top_k: int = None) -> List[Tuple[dict, float]]:
    top_k = top_k or CONFIG.retrieval_top_k_episodes
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM episodes WHERE user_id=? ORDER BY end_ts DESC LIMIT 50",
        (user_id,),
    ).fetchall()
    if not rows:
        return []
    qvec = embed(query)
    scored: List[Tuple[dict, float]] = []
    for r in rows:
        ep = dict(r)
        ev = conn.execute(
            "SELECT vec FROM embeddings WHERE ref_type='episode' AND ref_id=?",
            (ep["episode_id"],),
        ).fetchone()
        if ev is None:
            continue
        sim = cosine(qvec, blob_to_vec(ev["vec"]))
        scored.append((ep, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
