// 同源部署（方案 A）：使用相对路径，避免端口硬编码
const API_BASE = "";
async function getJSON(path){ const res = await fetch(`${API_BASE}${path}`); if(!res.ok) throw new Error(await res.text()); return res.json(); }

// 初始化 RAG 开关：根据后端状态锁定/预设
(async function initRagToggle(){
  const toggle = document.getElementById("rag-toggle");
  const label = document.getElementById("rag-toggle-label");
  if(!toggle) return;
  try {
    const st = await getJSON("/rag/status");
    toggle.checked = !!st.enabled_for_chat;
    const lock = !st.has_api_key || !st.healthy;
    toggle.disabled = lock;
    if (label){
      label.textContent = lock ? "RAG 不可用（缺少或无效的密钥）" : "让 RAG 参与普通对话";
    }
  } catch (e) {
    // 后端无该接口时，不影响使用
    console.warn("initRagToggle failed", e);
  }
})();


async function postJSON(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// 文本
const textBtn = document.getElementById("send-text");
textBtn.addEventListener("click", async () => {
  const input = document.getElementById("text-input").value.trim();
  const out = document.getElementById("text-output");
  const toggle = document.getElementById("rag-toggle");
  out.textContent = "发送中...";
  try {
    const body = { message: input };
    if (toggle) body.use_rag = !!toggle.checked;
    const data = await postJSON("/chat", body);
    out.textContent = data.text ?? JSON.stringify(data, null, 2);
  } catch (e) {
    out.textContent = `错误：${e}`;
  }
});

// 视觉
const visionBtn = document.getElementById("send-vision");
visionBtn.addEventListener("click", async () => {
  const prompt = document.getElementById("vision-prompt").value.trim();
  const imageUrl = document.getElementById("vision-image").value.trim();
  const out = document.getElementById("vision-output");
  out.textContent = "发送中...";
  try {
    const data = await postJSON("/chat-vision", { prompt, image_urls: imageUrl ? [imageUrl] : [] });
    out.textContent = data.text ?? JSON.stringify(data, null, 2);
  } catch (e) {
    out.textContent = `错误：${e}`;
  }
});

// RAG: 重新索引
const reindexBtn = document.getElementById("rag-reindex");
if (reindexBtn) {
  reindexBtn.addEventListener("click", async () => {
    const out = document.getElementById("rag-reindex-out");
    out.textContent = "索引中...";
    try {
      const data = await postJSON("/rag/reindex", {});
      out.textContent = `docs=${data.docs} chunks=${data.chunks} store=${data.store_path}`;
    } catch (e) {
      out.textContent = `错误：${e}`;
    }
  });
}

// RAG: 检索问答
const askBtn = document.getElementById("rag-ask");
if (askBtn) {
  askBtn.addEventListener("click", async () => {
    const q = document.getElementById("rag-question").value.trim();
    const mid = document.getElementById("rag-memoryid").value.trim();
    const out = document.getElementById("rag-answer");
    if (!q) {
      out.textContent = "请先输入问题";
      return;
    }
    out.textContent = "检索中...";
    try {
      const body = { question: q };
      if (mid) body.memoryid = mid;
      const data = await postJSON("/rag/query", body);
      const ctx = Array.isArray(data.contexts) ? data.contexts : [];
      const ctxLines = ctx.map((c, i) => `#${i+1} score=${c.score} doc=${c.metadata?.doc_name}\n${(c.text||"").slice(0,200)}...`).join("\n\n");
      out.textContent = `Answer:\n${data.text}\n\nContexts:\n${ctxLines}`;
    } catch (e) {
      out.textContent = `错误：${e}`;
    }
  });
}

