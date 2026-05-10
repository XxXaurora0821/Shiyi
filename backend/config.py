"""集中配置。所有可调参数集中在此，便于一眼定位 trade-off 旋钮。"""
import os
from dataclasses import dataclass


@dataclass
class Config:
    db_path: str = os.environ.get("DB_PATH", "./data/runtime.db")

    # 嵌入：本地哈希降级时的维度（OpenAI 模式不使用此值）
    embed_dim: int = 256

    # 对话窗口：最近 K 轮逐字注入 prompt
    history_window: int = 6

    # 单一类别下的特征上限，超过即按分数淘汰
    feature_cap_per_category: int = 50

    # 检索注入 prompt 的 top-K
    retrieval_top_k_features: int = 8
    retrieval_top_k_episodes: int = 3

    # L1 抽取节流：>= n 字才走 LLM 抽取（规则路径不受限）
    llm_extract_min_chars: int = 16
    extract_every_n_turns: int = 1

    # 合并阈值：余弦相似度 ≥ 该值则视为同一特征
    sim_merge_threshold: float = 0.82

    # 情景摘要：每 N 个用户回合落一片段
    episode_every_n_turns: int = 8

    # OpenAI（可选；未设置 key 走本地 mock 链路）
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    openai_chat_model: str = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_embed_model: str = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    # 项目级 sk-proj-xxx key 必须配套传 project ID，否则报 mismatched_project
    openai_project: str = os.environ.get("OPENAI_PROJECT", "")
    openai_org: str = os.environ.get("OPENAI_ORG", "")

    @property
    def use_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def openai_kwargs(self) -> dict:
        kw = {"api_key": self.openai_api_key}
        if self.openai_project:
            kw["project"] = self.openai_project
        if self.openai_org:
            kw["organization"] = self.openai_org
        return kw


CONFIG = Config()
