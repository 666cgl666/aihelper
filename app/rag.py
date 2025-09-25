import os
import json
import math
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from urllib import request as urlrequest

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Config
DOCS_DIR = os.getenv("RAG_DOCS_DIR", "local_docs")
STORE_PATH = os.getenv("RAG_STORE_PATH", os.path.join(DOCS_DIR, "index.json"))
EMBED_MODEL = os.getenv("ARK_EMBEDDING_MODEL", "doubao-embedding-vision-250615")
BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
API_KEY = os.getenv("ARK_API_KEY")
_RAG_HEALTHY: bool = bool(API_KEY)

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.75"))


# ---------- Utility: cosine similarity ----------
def _cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# ---------- Embedding client for Ark ----------
class ArkEmbeddingClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.endpoint = base_url.rstrip("/") + "/embeddings/multimodal"
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        if not self.api_key:
            raise RuntimeError("Missing ARK_API_KEY for embeddings")
        payload = {
            "model": self.model,
            "input": [{"type": "text", "text": text}]
        }
        data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(self.endpoint, data=data)
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        t0 = time.time()
        try:
            with urlrequest.urlopen(req, timeout=60) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            _RAG_HEALTHY = True
        except Exception as e:
            _RAG_HEALTHY = False
            raise
        dt_ms = int((time.time() - t0) * 1000)
        print(f"[RAG] embed_text model={self.model} chars={len(text)} took={dt_ms}ms")
        # Try to extract embedding vector
        try:
            return out["data"][0]["embedding"]
        except Exception:
            if "embedding" in out:
                return out["embedding"]
            raise RuntimeError(f"Unexpected embeddings response format: {out}")


# ---------- Vector store (JSON file) ----------
class SimpleVectorStore:
    def __init__(self, path: str):
        self.path = path
        self.items: List[Dict[str, Any]] = []
        self.model: str = EMBED_MODEL

    def load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.model = data.get("model", EMBED_MODEL)
            self.items = data.get("items", [])
        else:
            self.items = []

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"model": self.model, "items": self.items}, f, ensure_ascii=False)

    def add(self, vector: List[float], text: str, metadata: Dict[str, Any]) -> None:
        vid = hashlib.md5((metadata.get("doc_name", "") + str(metadata.get("chunk_index", "")) + text).encode("utf-8")).hexdigest()
        self.items.append({"id": vid, "vector": vector, "text": text, "metadata": metadata})

    def search(self, query_vec: List[float], top_k: int = TOP_K) -> List[Tuple[float, Dict[str, Any]]]:
        scored = []
        for it in self.items:
            score = _cosine(query_vec, it["vector"]) if it.get("vector") else 0.0
            scored.append((score, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]


# ---------- Document loading & chunking ----------
def _iter_markdown_files(root: str) -> List[str]:
    paths = []
    if not os.path.isdir(root):
        return paths
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isfile(p) and name.lower().endswith(".md"):
            paths.append(p)
    return paths


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


# ---------- Indexing ----------
def reindex_docs() -> Dict[str, Any]:
    """Load .md docs, split, embed, and store vectors."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    files = _iter_markdown_files(DOCS_DIR)
    store = SimpleVectorStore(STORE_PATH)
    # reset store to avoid duplicate appends on reindex
    store.items = []

    print(f"[RAG] reindex start docs_dir={DOCS_DIR} files={len(files)} model={EMBED_MODEL}")
    embedder = ArkEmbeddingClient(API_KEY, BASE_URL, EMBED_MODEL)

    total_chunks = 0
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        doc_name = os.path.basename(path)
        chunks = _split_text(content)
        print(f"[RAG] indexing file={doc_name} chunks={len(chunks)}")
        for idx, ch in enumerate(chunks):
            vec = embedder.embed_text(ch)
            store.add(vec, ch, {"doc_name": doc_name, "chunk_index": idx})
            total_chunks += 1

    store.save()
    print(f"[RAG] reindex done files={len(files)} chunks={total_chunks} store={STORE_PATH}")
    return {"docs": len(files), "chunks": total_chunks, "store_path": STORE_PATH}


def get_status() -> Dict[str, Any]:
    return {
        "has_api_key": bool(API_KEY),
        "healthy": bool(_RAG_HEALTHY),
        "enabled_for_chat": os.getenv("RAG_ENABLE_FOR_CHAT", "true").strip().lower() in ("1","true","yes","on"),
        "embed_model": EMBED_MODEL,
    }


# ---------- Query & RAG answer ----------
def _load_store() -> SimpleVectorStore:
    store = SimpleVectorStore(STORE_PATH)
    store.load()
    return store


def retrieve(query: str, top_k: int = TOP_K, min_score: float = MIN_SCORE) -> List[Dict[str, Any]]:
    t0 = time.time()
    embedder = ArkEmbeddingClient(API_KEY, BASE_URL, EMBED_MODEL)
    q_vec = embedder.embed_text(query)
    store = _load_store()
    results = []
    for score, it in store.search(q_vec, top_k=top_k):
        if score < min_score:
            continue
        results.append({
            "score": float(round(score, 4)),
            "text": it["text"],
            "metadata": it.get("metadata", {})
        })
    dt_ms = int((time.time() - t0) * 1000)
    print(f"[RAG] retrieve query_chars={len(query)} kept={len(results)}/{top_k} took={dt_ms}ms")
    return results


def answer_with_context(question: str, contexts: List[Dict[str, Any]], memory_id: Optional[str] = None) -> str:
    """Use Chat LLM to answer with retrieved contexts."""
    system = (
        "你是一个检索增强问答助手。请严格依据提供的文档片段回答；"
        "若无法从文档中找到答案，请明确说明不确定，不要编造。"
    )

    parts = []
    for i, c in enumerate(contexts, 1):
        meta = c.get("metadata", {})
        src = meta.get("doc_name", "unknown")
        parts.append(f"[{i}] (source: {src})\n{c.get('text','').strip()}")
    context_block = "\n\n".join(parts) if parts else "(无匹配片段)"

    prompt = (
        f"已检索到以下文档片段：\n\n{context_block}\n\n"
        f"问题：{question}\n"
        f"请给出基于文档的答案，并在需要时引用片段编号。"
    )

    llm = ChatOpenAI(model=os.getenv("ARK_MODEL", "doubao-seed-1-6-vision-250815"),
                     base_url=BASE_URL, api_key=API_KEY, temperature=0)
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def query_and_answer(question: str, memory_id: Optional[str] = None,
                     top_k: int = TOP_K, min_score: float = MIN_SCORE) -> Dict[str, Any]:
    ctxs = retrieve(question, top_k=top_k, min_score=min_score)
    answer = answer_with_context(question, ctxs, memory_id=memory_id)
    return {"text": answer, "contexts": ctxs}

