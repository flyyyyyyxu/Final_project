# rag_qianfan.py
import os
from openai import OpenAI
from typing import List
from rag_retrieval import search
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 从 .env 读取 QIANFAN_API_KEY（或设置环境变量）
API_KEY = os.getenv("QIANFAN_API_KEY")
if not API_KEY:
    raise EnvironmentError("Please set QIANFAN_API_KEY in your environment or .env file")

# OpenAI-compatible client pointing to 千帆 V2
client = OpenAI(api_key=API_KEY, base_url="https://qianfan.baidubce.com/v2")

# 可调整：千帆模型名（你在控制台可用的模型）
DEFAULT_MODEL = "ernie-speed-8k"   # 或 "ernie-speed-128k", "ernie-4.0-8k" 等

def build_context(retrieved: List[dict], max_chars_each=800):
    parts = []
    for i, r in enumerate(retrieved):
        meta = r.get("metadata", {})
        title = meta.get("title") or meta.get("file") or meta.get("source", "unknown")
        url = meta.get("url", "")
        chunk_text = r.get("chunk", "")
        # 限制长度，避免 prompt 过长
        snippet = chunk_text[:max_chars_each]
        parts.append(f"[{i+1}] 来源: {title}\n链接: {url}\n片段: {snippet}")
    return "\n\n".join(parts)

def generate_answer(user_question: str, top_k: int = 5, model: str = DEFAULT_MODEL, temperature: float = 0.2):
    # 1) Retrieve
    retrieved = search(user_question, top_k=top_k)

    # 2) Build context string
    context = build_context(retrieved)

    # 3) Prompt template
    prompt = f"""
你是一个专业的旅行规划助手。请基于下面的检索到的真实游记片段回答用户问题，给出务实、可执行的建议，并在回答末尾列出引用来源（编号与链接）。
用户问题：{user_question}

检索到的内容如下（请务必仅基于这些内容回答）：
{context}

请给出清晰的建议/行程/注意事项。如果检索信息不足，请说明哪些信息缺失，并给出合理的通用建议。
"""

    # 4) Call Qianfan (OpenAI-compatible)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是资深旅行规划顾问，回答要务实、礼貌并明确列出来源。"},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=800
    )

    # 5) Extract text (兼容不同返回格式)
    # 对于 OpenAI-compatible 返回，通常是 resp.choices[0].message.content 或 resp.choices[0].text
    try:
        content = resp.choices[0].message["content"]
    except Exception:
        try:
            content = resp.choices[0].text
        except Exception:
            content = str(resp)

    return content, retrieved