"""两层抽取：
1) 规则路径：低成本正则，命中 L0 核心档案 {name, age, gender}，sub-ms。
2) LLM 路径（或 mock）：从消息中抽取 L1 候选特征。

把规则与 LLM 解耦，是因为：
- L0 是固定结构、字段有限，正则足以覆盖且永远不能错；
- L1 是开放空间，规则注定漏抽，必须靠模型。
"""
import re
import time
from typing import List

from backend.config import CONFIG
from backend.db import get_conn
from backend.llm import CandidateFeature, extract_features_llm

_NAME_PATTERNS = [
    re.compile(r"我叫([一-龥A-Za-z·]{2,10})"),
    re.compile(r"我的名字(?:是|叫)([一-龥A-Za-z·]{2,10})"),
    re.compile(r"(?:你好|hi|hello)[，,]\s*我是([一-龥A-Za-z·]{2,10})"),
]
_AGE_PATTERNS = [
    re.compile(r"(?:我|今年)\s*(\d{1,3})\s*岁"),
    re.compile(r"今年(\d{1,3})(?![\d年])"),
]
_GENDER_PATTERNS = [
    (re.compile(r"(?:我是|是)男(?:生|性|的|孩)|男(?:生|性)的"), "男"),
    (re.compile(r"(?:我是|是)女(?:生|性|的|孩)|女(?:生|性)的"), "女"),
]


def update_core_profile_from_rules(user_id: str, text: str) -> dict:
    """规则抽取 L0。返回本次新写入/更新的字段。"""
    conn = get_conn()
    now = time.time()
    fields: dict = {}

    for p in _NAME_PATTERNS:
        m = p.search(text)
        if m:
            fields["name"] = m.group(1)
            break
    for p in _AGE_PATTERNS:
        m = p.search(text)
        if m:
            try:
                age = int(m.group(1))
                if 1 <= age <= 120:
                    fields["age"] = age
            except ValueError:
                pass
            break
    for p, val in _GENDER_PATTERNS:
        if p.search(text):
            fields["gender"] = val
            break

    if not fields:
        return {}

    row = conn.execute("SELECT * FROM core_profile WHERE user_id=?", (user_id,)).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO core_profile(user_id, name, age, gender, updated_at) VALUES (?,?,?,?,?)",
            (user_id, fields.get("name"), fields.get("age"), fields.get("gender"), now),
        )
    else:
        sets, vals = [], []
        for k in ("name", "age", "gender"):
            if k in fields:
                sets.append(f"{k}=?")
                vals.append(fields[k])
        sets.append("updated_at=?")
        vals.append(now)
        vals.append(user_id)
        conn.execute(f"UPDATE core_profile SET {', '.join(sets)} WHERE user_id=?", vals)
    return fields


def should_run_l1_extract(text: str, user_turn_count: int) -> bool:
    """L1 抽取节流。mock 模式下完全开启（无成本）；OpenAI 模式下用字数和回合数双重门槛。"""
    if not CONFIG.use_openai:
        return True
    if len(text) < CONFIG.llm_extract_min_chars:
        return False
    if user_turn_count % CONFIG.extract_every_n_turns != 0:
        return False
    return True


def extract_l1_candidates(text: str) -> List[CandidateFeature]:
    return extract_features_llm(text)
