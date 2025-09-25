import os
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from pydantic import Field
from dotenv import load_dotenv

# Load environment (prefer .env.local)
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env.local"), override=False)
load_dotenv(override=False)

from app.chain import invoke_text, invoke_vision, invoke_structured  # noqa: E402
from app import rag as rag  # noqa: E402


class ChatRequest(BaseModel):
    message: str
    memoryid: Optional[str] = None
    use_rag: Optional[bool] = None


class ChatResponse(BaseModel):
    text: str


class StructuredChatRequest(BaseModel):
    message: str
    schema: dict
    name: Optional[str] = None
    memoryid: Optional[str] = None


class StructuredChatResponse(BaseModel):
    data: dict


# RAG models
class RAGReindexResponse(BaseModel):
    docs: int = Field(..., description="Number of .md files indexed")
    chunks: int = Field(..., description="Number of chunks embedded")
    store_path: str


class RAGQueryRequest(BaseModel):
    question: str
    memoryid: Optional[str] = None


class RAGContext(BaseModel):
    score: float
    text: str
    metadata: dict


class RAGQueryResponse(BaseModel):
    text: str
    contexts: List[RAGContext]


class RAGStatusResponse(BaseModel):
    has_api_key: bool
    healthy: bool
    enabled_for_chat: bool
    embed_model: str


class VisionChatRequest(BaseModel):
    prompt: str
    image_urls: Optional[List[str]] = None
    memoryid: Optional[str] = None


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

# Static frontend (Scheme A): serve ./frontend at /ui
app.mount("/ui", StaticFiles(directory="frontend", html=True), name="ui")

# Optional: redirect root to /ui/
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    text = await run_in_threadpool(invoke_text, req.message, req.memoryid, req.use_rag)
    return ChatResponse(text=text)


@app.post("/chat-vision", response_model=ChatResponse)
async def chat_vision(req: VisionChatRequest):
    text = await run_in_threadpool(invoke_vision, req.prompt, req.image_urls or [], req.memoryid)
    return ChatResponse(text=text)


@app.post("/chat-structured", response_model=StructuredChatResponse)
async def chat_structured(req: StructuredChatRequest):
    data = await run_in_threadpool(invoke_structured, req.message, req.schema, req.name, req.memoryid)
    return StructuredChatResponse(data=data)


@app.post("/rag/reindex", response_model=RAGReindexResponse)
async def rag_reindex():
    result = await run_in_threadpool(rag.reindex_docs)
    return RAGReindexResponse(**result)


@app.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(req: RAGQueryRequest):
    result = await run_in_threadpool(rag.query_and_answer, req.question, req.memoryid)
    return RAGQueryResponse(text=result["text"], contexts=result["contexts"])


@app.get("/rag/status", response_model=RAGStatusResponse)
async def rag_status():
    status = await run_in_threadpool(rag.get_status)
    return RAGStatusResponse(**status)

