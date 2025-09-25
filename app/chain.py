import os, json
from typing import List, Union, Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from app import rag as rag

# In-memory chat histories keyed by memory_id (per-user/session isolation)
_MESSAGE_STORES: Dict[str, InMemoryChatMessageHistory] = {}


def _get_history(memory_id: Optional[str]) -> InMemoryChatMessageHistory:
    key = memory_id or "default"
    hist = _MESSAGE_STORES.get(key)
    if hist is None:
        hist = InMemoryChatMessageHistory()
        # Inject system prompt once when history is created
        system_prompt = os.getenv("SYSTEM_PROMPT", "").strip()
        if system_prompt:
            hist.add_message(SystemMessage(content=system_prompt))
        _MESSAGE_STORES[key] = hist
    return hist


def _build_llm() -> ChatOpenAI:
    """Create a ChatOpenAI client pointing to Ark (Doubao) via OpenAI-compatible API."""
    model = os.getenv("ARK_MODEL", "doubao-seed-1-6-vision-250815")
    base_url = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ARK_API_KEY. Please set it in .env.local or environment.")
    return ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0)


def _env_true(name: str, default: bool = True) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _build_rag_context_block(question: str) -> Optional[str]:
    try:
        ctxs = rag.retrieve(question)
    except Exception as e:
        # 静默失败，不影响普通对话
        print(f"[CHAT+RAG] retrieve failed: {e}")
        ctxs = []
    if not ctxs:
        return None
    parts = []
    for i, c in enumerate(ctxs, 1):
        src = (c.get("metadata") or {}).get("doc_name", "unknown")
        parts.append(f"[{i}] (source: {src})\n{(c.get('text') or '').strip()}")
    context_block = "\n\n".join(parts)
    instruction = (
        "你是一个检索增强问答助手。请严格依据提供的文档片段回答；"
        "若无法从文档中找到答案，请明确说明不确定，不要编造。"
    )
    return f"{instruction}\n\n已检索到以下文档片段：\n\n{context_block}\n\n当回答用户问题时，请优先依据上述片段。"


def invoke_text(message: str, memory_id: Optional[str] = None, rag_override: Optional[bool] = None) -> str:
    """Synchronous text chat with per-memory_id chat history (LangChain InMemoryChatMessageHistory).

    集成 RAG：若 RAG_ENABLE_FOR_CHAT=true（默认），则在每次对话时检索相关片段，
    将片段作为当轮的 SystemMessage 注入，仅参与本次生成，不写入会话记忆。
    """
    llm = _build_llm()
    hist = _get_history(memory_id)

    # 构造本轮消息：历史 + （可选）RAG 上下文 + 用户消息
    messages = list(hist.messages)
    enabled = _env_true("RAG_ENABLE_FOR_CHAT", True) if rag_override is None else bool(rag_override)
    if enabled:
        ctx_block = _build_rag_context_block(message)
        if ctx_block:
            messages.append(SystemMessage(content=ctx_block))
            print("[CHAT+RAG] context injected")

    messages.append(HumanMessage(content=message))
    resp = llm.invoke(messages)

    # 将用户与模型回复写入历史（不写入 RAG 上下文，避免污染记忆）
    try:
        hist.add_message(HumanMessage(content=message))
        hist.add_message(resp)
    except Exception:
        pass

    return resp.content if isinstance(resp.content, str) else str(resp.content)


def invoke_vision(prompt: str, image_urls: Union[str, List[str]], memory_id: Optional[str] = None) -> str:
    """Synchronous vision chat using image URLs with per-memory_id chat history.

    The message content follows the OpenAI-compatible multimodal format.
    """
    if isinstance(image_urls, str):
        image_list = [image_urls]
    else:
        image_list = image_urls or []

    content: List[dict] = []
    if prompt:
        content.append({"type": "text", "text": prompt})
    for url in image_list:
        content.append({"type": "image_url", "image_url": {"url": url}})

    llm = _build_llm()
    hist = _get_history(memory_id)

    # Add user multimodal message, then predict using full history
    hist.add_message(HumanMessage(content=content))
    resp = llm.invoke(hist.messages)

    try:
        hist.add_message(resp)
    except Exception:
        pass

    return resp.content if isinstance(resp.content, str) else str(resp.content)




def invoke_structured(message: str, schema: Dict[str, Any], name: Optional[str] = None, memory_id: Optional[str] = None) -> Dict[str, Any]:
    """Structured output using JSON Schema via LangChain.

    Accepts a JSON Schema (either a bare JSON Schema dict or {"name":..., "schema":{...}} payload)
    and returns a Python dict parsed from the model's structured output.
    """
    # Normalize payload for with_structured_output
    if "schema" in schema and "name" in schema:
        payload = schema  # already in {name, schema}
    else:
        payload = {"name": name or "output", "schema": schema}

    llm = _build_llm()
    structured_llm = llm.with_structured_output(payload)

    hist = _get_history(memory_id)
    # Add user message to history for context
    hist.add_message(HumanMessage(content=message))

    # Invoke with full history to preserve context
    result: Dict[str, Any] = structured_llm.invoke(hist.messages)

    # Persist assistant structured result as JSON string into history for continuity
    try:
        hist.add_message(AIMessage(content=json.dumps(result, ensure_ascii=False)))
    except Exception:
        pass

    return result
