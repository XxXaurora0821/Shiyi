"""对话编排：消息入库 → 抽取 → 检索 → 拼装 prompt → 调 LLM → 写回。

prompt 结构（注入 token 受控）：
  [system]
    指令
    【用户档案】L0 一行
    【相关记忆】top-K L1 条目（带置信度/新近度/命中次数）
    【过往片段】top-K L2 摘要
  [history] 最近 K 轮逐字
  [user]    当前消息
"""
import time
from typing import Dict, List, Optional, Tuple

from backend import memory as mem
from backend.config import CONFIG
from backend.llm import chat as llm_chat
from backend.retrieval import retrieve_episodes, retrieve_features

SYSTEM_HEADER = (
    "你是一个有持续记忆的对话助手。结合下方「用户档案」与「相关记忆」自然回应用户，"
    "不要罗列记忆条目，不要主动确认隐私字段。回答简洁，不超过200字。"
)


def _humanize_seconds(sec: float) -> str:
    if sec < 60:
        return "刚刚"
    if sec < 3600:
        return f"{int(sec / 60)}分钟前"
    if sec < 86400:
        return f"{int(sec / 3600)}小时前"
    return f"{int(sec / 86400)}天前"


def assemble_prompt(user_id: str, session_id: str, user_text: str) -> Tuple[List[Dict], dict]:
    profile = mem.get_core_profile(user_id)
    feats = retrieve_features(user_id, user_text)
    eps = retrieve_episodes(user_id, user_text)

    lines = [SYSTEM_HEADER, "", "【用户档案】"]
    age_disp = "未知" if profile.get("age") in (None, "", 0) else f"{profile['age']}岁"
    lines.append(
        f"姓名：{profile.get('name') or '未知'}  "
        f"年龄：{age_disp}  "
        f"性别：{profile.get('gender') or '未知'}"
    )
    if feats:
        lines += ["", "【相关记忆】（按相关度排序，仅供你参考）"]
        now = time.time()
        for f, _ in feats:
            ago = _humanize_seconds(now - f["last_seen"])
            lines.append(
                f"- [{f['category']}] {f['key']} "
                f"(置信度{(f.get('confidence') or 0):.2f}, 上次提及：{ago}, 命中×{f['hit_count']})"
            )
    if eps:
        lines += ["", "【过往片段】"]
        for ep, _ in eps:
            lines.append(f"- {ep['summary'][:120]}")

    sys_msg = {"role": "system", "content": "\n".join(lines)}
    history = mem.recent_messages(session_id, CONFIG.history_window * 2)
    # 历史里最后一条已经是当前用户消息（在调用本函数之前已经 append 过），需排除避免重复
    history_pruned = [h for h in history if not (h["role"] == "user" and h["content"] == user_text)]
    messages = [sys_msg] + [{"role": h["role"], "content": h["content"]} for h in history_pruned]
    messages.append({"role": "user", "content": user_text})

    debug = {
        "profile": profile,
        "retrieved_features": [
            {"id": f["feature_id"], "category": f["category"], "key": f["key"], "score": round(s, 3)}
            for f, s in feats
        ],
        "retrieved_episodes": [{"id": e["episode_id"], "score": round(s, 3)} for e, s in eps],
        "history_turns": len(history_pruned),
        "system_prompt_chars": len(sys_msg["content"]),
    }
    return messages, debug


def handle_chat(user_id: str, session_id: Optional[str], user_text: str) -> Dict:
    mem.ensure_user(user_id)
    sid = mem.ensure_session(user_id, session_id)
    user_msg_id = mem.append_message(sid, "user", user_text)

    # 关键：先抽取再拼装。这样同一回合的 "我叫小明" 立刻能影响档案展示。
    mem.ingest_user_message(user_id, sid, user_text, user_msg_id)

    messages, debug = assemble_prompt(user_id, sid, user_text)
    reply = llm_chat(messages)

    mem.append_message(sid, "assistant", reply)
    mem.maybe_summarize_episode(user_id, sid)

    return {"session_id": sid, "reply": reply, "debug": debug}
