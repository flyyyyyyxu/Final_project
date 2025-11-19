import os
from typing import List, Dict
import json
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI

from rag_retrieval import search


# ======================
# 加载密钥 & 初始化客户端
# ======================
load_dotenv()

API_KEY = os.getenv("QIANFAN_API_KEY")
if not API_KEY:
    raise RuntimeError("QIANFAN_API_KEY 未设置，请在 .env 中配置。")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://qianfan.baidubce.com/v2",
)

DEFAULT_MODEL = "ernie-speed-8k"


# ======================
# 加载城市氛围关键词（来自 ingest）
# ======================
CITY_VIBES_PATH = "./vector_store/city_vibes.json"

try:
    with open(CITY_VIBES_PATH, "r", encoding="utf-8") as f:
        CITY_VIBES = json.load(f)
except Exception:
    CITY_VIBES = {}
    print("⚠️ 未找到 city_vibes.json，城市氛围关键词将为空。")


# ======================
# 构建检索片段上下文
# ======================
def build_context(retrieved: List[Dict], max_chars_each: int = 600) -> str:
    parts = []
    for i, r in enumerate(retrieved):
        md = r.get("metadata", {}) or {}

        title = (
            md.get("title")
            or md.get("file")
            or md.get("source")
            or f"片段 {i+1}"
        )
        url = md.get("url", "")
        chunk = r.get("chunk", "")
        snippet = chunk[:max_chars_each]

        parts.append(
            f"[{i+1}] 标题：{title}\n链接：{url}\n内容片段：{snippet}"
        )
    return "\n\n".join(parts)


# ======================
# 基于检索内容的 vibes 聚合
# ======================
def collect_vibes(retrieved: List[Dict], top_k: int = 8) -> list[str]:
    counter = Counter()
    for r in retrieved:
        md = r.get("metadata", {}) or {}
        vibes = md.get("vibes", [])
        if isinstance(vibes, list):
            for v in vibes:
                v = str(v).strip()
                if v:
                    counter[v] += 1

    if not counter:
        return []

    return [w for w, _ in counter.most_common(top_k)]


# ======================
# 生成最终回答（主函数）
# ======================
def generate_answer(
    user_question: str,
    days: int,
    top_k: int = 5,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    city: str | None = None,
):
    """根据用户问题 + 天数 + 检索结果，生成结构化行程"""

    days = max(1, min(days, 7))  # 限制天数范围

    # --------------------
    # STEP 1 检索
    # --------------------
    retrieved = search(user_question, top_k=top_k)

    # --------------------
    # STEP 2 按城市过滤
    # --------------------
    if city:
        city_l = city.strip().lower()
        filtered = []
        for r in retrieved:
            md = r.get("metadata", {}) or {}
            md_city = str(md.get("city", "")).lower()
            if md_city and md_city == city_l:
                filtered.append(r)

        # 若过滤后没有结果，则 fallback 使用非过滤检索
        if filtered:
            retrieved = filtered

    # --------------------
    # STEP 3：上下文
    # --------------------
    context = build_context(retrieved)

    # --------------------
    # STEP 4：提炼 vibes
    # --------------------
    vibes_from_data = collect_vibes(retrieved)

    # STEP 4.1 同时加载全城市 vibes.json（更稳）
    city_vibes = []
    if city:
        key = city.strip().lower()
        if key in CITY_VIBES:
            city_vibes = CITY_VIBES[key].get("vibes", [])

    # STEP 4.2 合并两种 vibes（检索 + 全局）
    merged = list(dict.fromkeys(vibes_from_data + city_vibes))  # 去重保持顺序
    if len(merged) == 0:
        vibe_str = "（暂无关键词）"
    else:
        vibe_str = "、".join(merged[:12])  # 最多取 12 个，更自然

    # --------------------
    # STEP 5：构造 Prompt
    # --------------------
    prompt = f"""
你是一名中文旅行规划顾问。现在你要根据用户的需求和真实游记片段，设计一份结构清晰、易执行的旅行计划。

【用户需求】
{user_question}

【从城市游记中提炼出的氛围关键词】
{vibe_str}

【检索到的真实游记片段】
{context}

【输出格式要求 —— 请严格遵守】

1. 开场总结（1–2 句）
   - 自然融入 2–4 个关键词（不要生硬罗列）；
   - 概括行程整体节奏。

2. 行程安排（共 {days} 天）：
   - 每一天必须用英文 Day 开头：
       Day 1 ｜ 一句话概括当天主题
         - 上午：...
         - 下午：...
         - 晚上：...

   - Day 数量必须为 1～{days} 天，不能多不能少；
   - 行程内容优先参考游记片段中的地点、路线、体验；
   - 不够时可使用常见景点（英文名称），但禁止编造不存在景点。

3. 注意事项（至少 3 条）
   - 包括天气、穿衣、预算、交通、节奏、预定等。

4. 参考来源（必须列出来源 URL）
   - 列出模型引用的游记片段：
     [1] 标题：xxx 链接：xxx
     [2] ...

请严格按上述结构输出，不要加入额外解释。
"""

    # --------------------
    # STEP 6：调用模型
    # --------------------
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是一名严谨、专业的中文旅行规划顾问。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=1500,
    )

    # --------------------
    # STEP 7：兼容解析输出
    # --------------------
    content = ""
    try:
        content = resp.choices[0].message["content"]
    except:
        try:
            content = resp.choices[0].message.content
        except:
            content = str(resp)

    return content, retrieved