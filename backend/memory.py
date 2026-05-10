"""记忆系统外观：用户/会话/消息生命周期 + 触发抽取与情景摘要。

L0 核心档案在抽取时立即更新（同一回合就能影响本次 prompt）；
L1 候选特征在抽取时合并入库；
L2 情景摘要按每 N 个用户回合落一片段，向量入库供后续检索。
"""
import time
import uuid
from typing import Dict, List, Optional

from backend.config import CONFIG
from backend.db import get_conn
from backend.embeddings import embed, vec_to_blob
from backend.extractor import (
    extract_l1_candidates,
    should_run_l1_extract,
    update_core_profile_from_rules,
)
from backend.features import upsert_feature


def ensure_user(user_id: str) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO users(user_id, created_at) VALUES (?, ?)",
        (user_id, time.time()),
    )


def ensure_session(user_id: str, session_id: Optional[str]) -> str:
    conn = get_conn()
    now = time.time()
    if session_id:
        row = conn.execute(
            "SELECT session_id FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE sessions SET updated_at=? WHERE session_id=?", (now, session_id)
            )
            return session_id
    sid = session_id or str(uuid.uuid4())
    conn.execute(
        "INSERT INTO sessions(session_id, user_id, created_at, updated_at) VALUES (?,?,?,?)",
        (sid, user_id, now, now),
    )
    return sid


def append_message(session_id: str, role: str, content: str) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO messages(session_id, role, content, ts) VALUES (?,?,?,?)",
        (session_id, role, content, time.time()),
    )
    return cur.lastrowid


def recent_messages(session_id: str, limit: int) -> List[Dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT role, content, ts FROM messages WHERE session_id=? ORDER BY msg_id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    return list(reversed([dict(r) for r in rows]))


def get_core_profile(user_id: str) -> Dict:
    conn = get_conn()
    row = conn.execute("SELECT * FROM core_profile WHERE user_id=?", (user_id,)).fetchone()
    if not row:
        return {"name": None, "age": None, "gender": None}
    return {"name": row["name"], "age": row["age"], "gender": row["gender"]}


def user_turn_count(session_id: str) -> int:
    conn = get_conn()
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM messages WHERE session_id=? AND role='user'",
        (session_id,),
    ).fetchone()
    return row["c"] if row else 0


def ingest_user_message(user_id: str, session_id: str, text: str, msg_id: int) -> None:
    """对一条用户消息跑完整抽取链。务必在 prompt 拼装之前调用——
    这样同一回合中说出的 "我叫小明" 就能立刻进入本次回复的档案。"""
    update_core_profile_from_rules(user_id, text)
    if should_run_l1_extract(text, user_turn_count(session_id)):
        for c in extract_l1_candidates(text):
            upsert_feature(user_id, c, source_msg_id=msg_id)


def maybe_summarize_episode(user_id: str, session_id: str) -> None:
    """每 N 个用户回合，把最近若干条消息摘要为一片段并向量化。"""
    n = CONFIG.episode_every_n_turns
    cnt = user_turn_count(session_id)
    if cnt == 0 or cnt % n != 0:
        return
    conn = get_conn()
    rows = conn.execute(
        "SELECT role, content, ts FROM messages WHERE session_id=? ORDER BY msg_id DESC LIMIT ?",
        (session_id, n * 2),
    ).fetchall()
    if not rows:
        return
    rows = list(reversed([dict(r) for r in rows]))
    summary = " | ".join(f"{r['role']}: {r['content'][:60]}" for r in rows)[:500]
    cur = conn.execute(
        "INSERT INTO episodes(user_id, session_id, summary, start_ts, end_ts) VALUES (?,?,?,?,?)",
        (user_id, session_id, summary, rows[0]["ts"], rows[-1]["ts"]),
    )
    eid = cur.lastrowid
    vec = embed(summary)
    conn.execute(
        "INSERT OR REPLACE INTO embeddings(ref_type, ref_id, vec) VALUES (?,?,?)",
        ("episode", eid, vec_to_blob(vec)),
    )
