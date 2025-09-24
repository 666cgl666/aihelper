import os
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment (prefer .env.local)
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env.local"), override=False)
load_dotenv(override=False)

from app.chain import invoke_text, invoke_vision  # noqa: E402


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    text: str


class VisionChatRequest(BaseModel):
    prompt: str
    image_urls: Optional[List[str]] = None


app = FastAPI(title="AIHelper API", version="0.1.0")

# CORS
_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
origins = [o.strip() for o in _cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    text = await run_in_threadpool(invoke_text, req.message)
    return ChatResponse(text=text)


@app.post("/chat-vision", response_model=ChatResponse)
async def chat_vision(req: VisionChatRequest):
    text = await run_in_threadpool(invoke_vision, req.prompt, req.image_urls or [])
    return ChatResponse(text=text)

