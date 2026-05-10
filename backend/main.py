"""FastAPI 入口。挂载 /api/* 与前端静态页。"""
import os
from typing import Optional

# 在 import backend.config 之前加载 .env，否则 CONFIG 在 import 时已读完 os.environ 了。
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from backend import memory as mem  # noqa: E402
from backend.chat import handle_chat  # noqa: E402
from backend.features import list_all_features  # noqa: E402

app = FastAPI(title="拾忆 · Shiyi", description="分层用户记忆 + 动态特征扩展的 AI Runtime")


class ChatReq(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    message: str


class ChatResp(BaseModel):
    session_id: str
    reply: str
    debug: dict


@app.post("/api/chat", response_model=ChatResp)
def post_chat(req: ChatReq):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message is empty")
    return handle_chat(req.user_id, req.session_id, req.message)


@app.get("/api/memory/{user_id}")
def get_memory(user_id: str):
    return {
        "profile": mem.get_core_profile(user_id),
        "features": list_all_features(user_id),
    }


@app.get("/api/sessions/{session_id}/messages")
def get_messages(session_id: str, limit: int = 50):
    return {"messages": mem.recent_messages(session_id, limit)}


_FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"
)
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
