"""L1 特征：插入即合并、容量管控、检索打分。

合并策略：候选特征嵌入与同类别下所有已有特征比余弦相似度，最大值 ≥ 阈值则合并；
否则新建。合并时累加 hit_count、刷新 last_seen、轻微抬升 confidence。

容量策略：每个 (user, category) 维持一个上限。超限时按 score 升序淘汰，
score = log(1+hit_count) * exp(-Δdays/30) * confidence。
"""
import math
import time
from typing import List, Optional

import numpy as np

from backend.config import CONFIG
from backend.db import get_conn
from backend.embeddings import blob_to_vec, cosine, embed, vec_to_blob
from backend.llm import CandidateFeature


def _now() -> float:
    return time.time()


def _store_embedding(ref_type: str, ref_id: int, vec: np.ndarray) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO embeddings(ref_type, ref_id, vec) VALUES (?,?,?)",
        (ref_type, ref_id, vec_to_blob(vec)),
    )


def _load_embedding(ref_type: str, ref_id: int) -> Optional[np.ndarray]:
    conn = get_conn()
    row = conn.execute(
        "SELECT vec FROM embeddings WHERE ref_type=? AND ref_id=?",
        (ref_type, ref_id),
    ).fetchone()
    return blob_to_vec(row["vec"]) if row else None


def _features_in_category(user_id: str, category: str) -> List[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM features WHERE user_id=? AND category=?",
        (user_id, category),
    ).fetchall()
    return [dict(r) for r in rows]


def upsert_feature(user_id: str, cand: CandidateFeature, source_msg_id: Optional[int]) -> dict:
    """合并优先：相似度过阈值则更新已有；否则新建。返回最终持久化的行。"""
    conn = get_conn()
    now = _now()
    cand_text = f"{cand.category}: {cand.key}"
    cand_vec = embed(cand_text)

    best_id, best_sim = None, -1.0
    for f in _features_in_category(user_id, cand.category):
        ev = _load_embedding("feature", f["feature_id"])
        if ev is None:
            continue
        s = cosine(cand_vec, ev)
        if s > best_sim:
            best_sim, best_id = s, f["feature_id"]

    if best_id is not None and best_sim >= CONFIG.sim_merge_threshold:
        conn.execute(
            "UPDATE features SET hit_count=hit_count+1, last_seen=?, "
            "confidence=MIN(0.99, confidence+0.05), "
            "value=CASE WHEN length(?)>length(value) THEN ? ELSE value END "
            "WHERE feature_id=?",
            (now, cand.value, cand.value, best_id),
        )
        return dict(conn.execute("SELECT * FROM features WHERE feature_id=?", (best_id,)).fetchone())

    cur = conn.execute(
        "INSERT INTO features(user_id, category, key, value, confidence, hit_count, "
        "created_at, last_seen, source_msg_id) VALUES (?,?,?,?,?,?,?,?,?)",
        (user_id, cand.category, cand.key, cand.value, cand.confidence, 1, now, now, source_msg_id),
    )
    fid = cur.lastrowid
    _store_embedding("feature", fid, cand_vec)
    _enforce_category_cap(user_id, cand.category)
    row = conn.execute("SELECT * FROM features WHERE feature_id=?", (fid,)).fetchone()
    return dict(row)


def _enforce_category_cap(user_id: str, category: str) -> None:
    rows = _features_in_category(user_id, category)
    if len(rows) <= CONFIG.feature_cap_per_category:
        return
    now = _now()

    def score(f: dict) -> float:
        days = max(0.0, (now - f["last_seen"]) / 86400.0)
        return math.log1p(f["hit_count"]) * math.exp(-days / 30.0) * (f.get("confidence") or 0.5)

    drop = sorted(rows, key=score)[: len(rows) - CONFIG.feature_cap_per_category]
    conn = get_conn()
    for f in drop:
        conn.execute("DELETE FROM features WHERE feature_id=?", (f["feature_id"],))
        conn.execute("DELETE FROM embeddings WHERE ref_type='feature' AND ref_id=?", (f["feature_id"],))


def feature_score(f: dict, sim: float) -> float:
    """检索打分：相似度为主，叠加新近度与置信度。"""
    days = max(0.0, (_now() - f["last_seen"]) / 86400.0)
    recency = math.exp(-days / 30.0)
    return 0.7 * sim + 0.2 * recency + 0.1 * (f.get("confidence") or 0.5)


def list_all_features(user_id: str) -> List[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM features WHERE user_id=? ORDER BY last_seen DESC",
        (user_id,),
    ).fetchall()
    return [dict(r) for r in rows]
