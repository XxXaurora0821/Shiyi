"""端到端烟雾测试。
- 强制使用 mock LLM 路径（pop OPENAI_API_KEY）；
- 每个测试用独立 DB，避免互相污染；
- 覆盖：核心档案抽取、L1 合并、检索召回、特征容量上限。
"""
import os
import tempfile
import sys

# 必须在 import backend 之前设置 DB_PATH 与清掉 API key
_TMP = tempfile.mkdtemp(prefix="ai_runtime_test_")
os.environ["DB_PATH"] = os.path.join(_TMP, "test.db")
os.environ.pop("OPENAI_API_KEY", None)

# 把项目根加到 sys.path 便于 `python backend/tests/test_smoke.py` 直跑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend import db as dbmod  # noqa: E402
from backend import memory as mem  # noqa: E402
from backend.chat import handle_chat  # noqa: E402
from backend.config import CONFIG  # noqa: E402
from backend.features import list_all_features, upsert_feature  # noqa: E402
from backend.llm import CandidateFeature  # noqa: E402


def _reset():
    dbmod.reset_db()


def test_extract_core_profile():
    _reset()
    handle_chat("u1", None, "你好，我叫小明，今年25岁，我是男生，今天想聊聊运动。")
    p = mem.get_core_profile("u1")
    assert p["name"] == "小明", p
    assert p["age"] == 25, p
    assert p["gender"] == "男", p


def test_l1_extract_and_merge():
    _reset()
    handle_chat("u2", None, "我特别喜欢打篮球，下班还经常去打球。")
    handle_chat("u2", None, "周末又约了朋友打篮球，状态不错。")
    feats = list_all_features("u2")
    bb = [f for f in feats if "篮球" in f["key"]]
    assert len(bb) == 1, f"应当合并到一条篮球特征，实得：{bb}"
    assert bb[0]["hit_count"] >= 2, bb


def test_retrieval_surfaces_relevant_feature():
    _reset()
    handle_chat("u3", None, "我超爱听摇滚乐，最近在循环 Linkin Park。")
    handle_chat("u3", None, "我也常喝奶茶，每天一杯停不下来。")
    out = handle_chat("u3", None, "推荐点周末活动？")
    keys = [f["key"] for f in out["debug"]["retrieved_features"]]
    assert any("摇滚" in k or "音乐" in k for k in keys), keys


def test_category_cap_evicts_lowest_score():
    _reset()
    cap = CONFIG.feature_cap_per_category
    # 灌入 cap+5 条互不相似的兴趣特征
    for i in range(cap + 5):
        upsert_feature(
            "u4",
            CandidateFeature(category="兴趣", key=f"兴趣项{i:03d}", value=f"value{i}", confidence=0.6),
            source_msg_id=None,
        )
    feats = [f for f in list_all_features("u4") if f["category"] == "兴趣"]
    assert len(feats) <= cap, len(feats)


def test_negation_does_not_extract_interest():
    _reset()
    handle_chat("u5", None, "我其实不喜欢打篮球，太累了。")
    feats = [f for f in list_all_features("u5") if f["category"] == "兴趣" and "篮球" in f["key"]]
    assert feats == [], f"否定句不应抽出篮球兴趣：{feats}"


if __name__ == "__main__":
    fns = [
        test_extract_core_profile,
        test_l1_extract_and_merge,
        test_retrieval_surfaces_relevant_feature,
        test_category_cap_evicts_lowest_score,
        test_negation_does_not_extract_interest,
    ]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
