const API_BASE = "http://127.0.0.1:8000";

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
  out.textContent = "发送中...";
  try {
    const data = await postJSON("/chat", { message: input });
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

