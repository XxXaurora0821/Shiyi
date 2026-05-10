"""LLM 抽象层。
- chat(messages) -> str：对话生成
- extract_features_llm(text) -> [CandidateFeature]：从单条用户消息抽取候选特征

设置 OPENAI_API_KEY 即走 OpenAI；否则走本地 mock：
  * mock chat：基于 prompt 中的姓名/特征拼一段占位回复，便于离线跑通链路。
  * mock extract：用一组关键词/正则对照表近似抽取，对应「规则+模型混合」中的规则上限。

含一个进程内简单速率限制器（min interval token bucket），防止突发把上游打挂。
"""
import json
import re
import time
from dataclasses import dataclass
from typing import Dict, List

from backend.config import CONFIG


@dataclass
class CandidateFeature:
    category: str
    key: str
    value: str
    confidence: float = 0.7


ALLOWED_CATEGORIES = {"兴趣", "习惯", "消费", "关系", "人口", "其他"}


class _RateLimiter:
    def __init__(self, rps: float = 5.0) -> None:
        self.min_interval = 1.0 / rps
        self.last = 0.0

    def wait(self) -> None:
        now = time.time()
        delta = now - self.last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self.last = time.time()


_rl = _RateLimiter(rps=5.0)


# --------------------------- mock 实现（无网络） ---------------------------

_INTEREST_KEYWORDS = [
    ("篮球", "篮球"), ("足球", "足球"), ("羽毛球", "羽毛球"), ("乒乓", "乒乓球"),
    ("游戏", "电子游戏"), ("音乐", "音乐"), ("摇滚", "摇滚乐"), ("民谣", "民谣"),
    ("电影", "电影"), ("看书", "阅读"), ("读书", "阅读"), ("写作", "写作"),
    ("旅游", "旅游"), ("摄影", "摄影"), ("健身", "健身"), ("跑步", "跑步"),
    ("瑜伽", "瑜伽"), ("做饭", "烹饪"), ("烹饪", "烹饪"),
]

_HABIT_PATTERNS = [
    (r"熬夜|晚睡|凌晨.{0,4}睡", "熬夜"),
    (r"早起", "早起"),
    (r"点外卖|叫外卖|外卖", "常点外卖"),
    (r"喝咖啡|咖啡", "喝咖啡"),
    (r"奶茶", "爱喝奶茶"),
    (r"加班", "经常加班"),
]

_SHOP_PATTERNS = [
    (r"电子产品|数码|手机|电脑|平板|耳机|相机", "电子产品"),
    (r"主机|switch|ps5|steam", "游戏类消费"),
    (r"潮牌|球鞋|衣服|包包|奢侈品", "时尚消费"),
    (r"图书|书籍", "图书消费"),
]

_REL_PATTERNS = [
    (r"我女朋友|我老婆|我妻子|我男朋友|我老公|我丈夫", "有伴侣"),
    (r"我儿子|我女儿|我孩子|我家娃", "已为人父母"),
    (r"我爸|我妈|父母", "与父母紧密联系"),
]


def _mock_extract(text: str) -> List[CandidateFeature]:
    out: List[CandidateFeature] = []
    seen = set()

    def add(cat: str, key: str, value: str, conf: float = 0.6) -> None:
        sig = (cat, key)
        if sig in seen:
            return
        seen.add(sig)
        out.append(CandidateFeature(cat, key, value, conf))

    for kw, canon in _INTEREST_KEYWORDS:
        if kw in text:
            if re.search(r"不(?:喜欢|爱|想).{0,4}" + re.escape(kw), text):
                continue
            add("兴趣", canon, f"对{canon}有兴趣", 0.7)
    for pat, canon in _HABIT_PATTERNS:
        if re.search(pat, text):
            add("习惯", canon, canon, 0.65)
    for pat, canon in _SHOP_PATTERNS:
        if re.search(pat, text):
            add("消费", canon, f"{canon}消费偏好", 0.6)
    for pat, canon in _REL_PATTERNS:
        if re.search(pat, text):
            add("关系", canon, canon, 0.7)
    return out


def _mock_chat(messages: List[Dict[str, str]]) -> str:
    user_msg = ""
    for m in reversed(messages):
        if m["role"] == "user":
            user_msg = m["content"]
            break
    sysmsg = next((m["content"] for m in messages if m["role"] == "system"), "")
    name_match = re.search(r"姓名：(\S+?)\s", sysmsg)
    feats = re.findall(r"- \[(\S+?)\] (.+?)\s*\(", sysmsg)
    pieces = []
    if name_match and name_match.group(1) != "未知":
        pieces.append(f"{name_match.group(1)}你好。")
    else:
        pieces.append("你好。")
    if feats:
        sample = "、".join(k for _, k in feats[:3])
        pieces.append(f"我记得你提到过 {sample}。")
    pieces.append(f"关于「{user_msg[:40]}」我会继续帮你想想。")
    return "（本地占位回复）" + "".join(pieces)


# --------------------------- OpenAI 实现 ---------------------------

EXTRACT_SYSTEM = """你是一个用户特征抽取器。从用户消息中抽取 0~N 条新增特征。
仅返回严格 JSON 对象 {"items": [...]}，每个元素形如：
{"category":"...","key":"...","value":"...","confidence":0.0~1.0}
category 仅能取：兴趣 / 习惯 / 消费 / 关系 / 人口 / 其他。
key 必须是简短规范的名词短语（≤8字）。value 是详细说明（≤30字）。
若无可抽取项，返回 {"items": []}。不要解释，不要 markdown。"""


def _openai_chat(messages: List[Dict[str, str]]) -> str:
    from openai import OpenAI

    _rl.wait()
    client = OpenAI(**CONFIG.openai_kwargs)
    resp = client.chat.completions.create(
        model=CONFIG.openai_chat_model,
        messages=messages,
        temperature=0.6,
    )
    return resp.choices[0].message.content or ""


def _openai_extract(text: str) -> List[CandidateFeature]:
    from openai import OpenAI

    _rl.wait()
    client = OpenAI(**CONFIG.openai_kwargs)
    resp = client.chat.completions.create(
        model=CONFIG.openai_chat_model,
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
        items = data.get("items", []) if isinstance(data, dict) else data
        out: List[CandidateFeature] = []
        for d in items:
            cat = d.get("category", "其他")
            if cat not in ALLOWED_CATEGORIES:
                cat = "其他"
            key = str(d.get("key", "")).strip()[:32]
            if not key:
                continue
            out.append(CandidateFeature(
                category=cat,
                key=key,
                value=str(d.get("value", "")).strip()[:120],
                confidence=float(d.get("confidence", 0.6)),
            ))
        return out
    except Exception:
        return []


# --------------------------- 对外接口 ---------------------------

def chat(messages: List[Dict[str, str]]) -> str:
    if CONFIG.use_openai:
        try:
            return _openai_chat(messages)
        except Exception as e:
            return f"（LLM 调用失败，已降级）{e}"
    return _mock_chat(messages)


def extract_features_llm(text: str) -> List[CandidateFeature]:
    if CONFIG.use_openai:
        try:
            return _openai_extract(text)
        except Exception:
            return _mock_extract(text)
    return _mock_extract(text)
