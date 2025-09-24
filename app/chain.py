import os
from typing import List, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def _build_llm() -> ChatOpenAI:
    """Create a ChatOpenAI client pointing to Ark (Doubao) via OpenAI-compatible API."""
    model = os.getenv("ARK_MODEL", "doubao-seed-1-6-vision-250815")
    base_url = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ARK_API_KEY. Please set it in .env.local or environment.")
    return ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0)


def invoke_text(message: str) -> str:
    """Synchronous text generation using LangChain ChatOpenAI."""
    llm = _build_llm()
    resp = llm.invoke([HumanMessage(content=message)])
    # resp is an AIMessage; its .content may be str or list depending on API. Prefer .content -> str
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def invoke_vision(prompt: str, image_urls: Union[str, List[str]]) -> str:
    """Synchronous vision chat using image URLs with LangChain ChatOpenAI.

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
    resp = llm.invoke([HumanMessage(content=content)])
    return resp.content if isinstance(resp.content, str) else str(resp.content)

