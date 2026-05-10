"""SQLite 存储层。单文件、单连接、check_same_thread=False。
分层记忆：
  L0 core_profile  —— 固定结构 {name, age, gender}
  L1 features      —— 动态扩展特征（带类别、置信度、命中次数、时间）
  L2 episodes      —— 对话片段摘要
  附：embeddings 表（L1/L2 的向量，分离表便于检索）+ embed_cache（按文本 hash 去重）
"""
import os
import sqlite3
import threading
from typing import Optional

from backend.config import CONFIG

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  user_id TEXT PRIMARY KEY,
  created_at REAL
);
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  user_id TEXT,
  created_at REAL,
  updated_at REAL
);
CREATE TABLE IF NOT EXISTS messages (
  msg_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  role TEXT,
  content TEXT,
  ts REAL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, msg_id);

CREATE TABLE IF NOT EXISTS core_profile (
  user_id TEXT PRIMARY KEY,
  name TEXT,
  age INTEGER,
  gender TEXT,
  updated_at REAL
);

CREATE TABLE IF NOT EXISTS features (
  feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT,
  category TEXT,
  key TEXT,
  value TEXT,
  confidence REAL,
  hit_count INTEGER DEFAULT 1,
  created_at REAL,
  last_seen REAL,
  source_msg_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_features_user_cat ON features(user_id, category);
CREATE INDEX IF NOT EXISTS idx_features_user_seen ON features(user_id, last_seen);

CREATE TABLE IF NOT EXISTS episodes (
  episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT,
  session_id TEXT,
  summary TEXT,
  start_ts REAL,
  end_ts REAL
);
CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id, end_ts);

CREATE TABLE IF NOT EXISTS embeddings (
  ref_type TEXT,
  ref_id INTEGER,
  vec BLOB,
  PRIMARY KEY (ref_type, ref_id)
);

CREATE TABLE IF NOT EXISTS embed_cache (
  text_hash TEXT PRIMARY KEY,
  vec BLOB,
  backend TEXT,
  created_at REAL
);
"""


def get_conn() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            os.makedirs(os.path.dirname(os.path.abspath(CONFIG.db_path)) or ".", exist_ok=True)
            _conn = sqlite3.connect(CONFIG.db_path, check_same_thread=False, isolation_level=None)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.executescript(SCHEMA)
        return _conn


def reset_db() -> None:
    """测试辅助：关闭并删除当前数据库。"""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None
        for suffix in ("", "-wal", "-shm"):
            p = CONFIG.db_path + suffix
            if os.path.exists(p):
                os.remove(p)
