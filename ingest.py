import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from collections import Counter

load_dotenv()
QIANFAN_API_KEY = os.getenv("QIANFAN_API_KEY")
qianfan_client = OpenAI(
    api_key=QIANFAN_API_KEY,
    base_url="https://qianfan.baidubce.com/v2",
)

TAG_MODEL = "ernie-speed-8k"  # 你可根据账号情况换成更稳的模型，如 ernie-4.0-8k

DATA_DIR = "./data"
VECTOR_DIR = "./vector_store"
os.makedirs(VECTOR_DIR, exist_ok=True)

# -----------------------
# 抽取城市氛围关键词（vibes）
# -----------------------


def extract_city_vibes(text: str, city: str) -> list[str]:
    """
    用千帆模型从一条游记（标题+正文）中抽取城市氛围/特点关键词。
    返回一个字符串列表，例如 ["浪漫", "适合步行", "夜景好看"]。
    """
    # 截断一下，避免太长
    short_text = text[:800]

    prompt = f"""
下面是一段关于城市「{city}」的旅行游记内容，请你用 3-7 个中文关键词概括这座城市在这篇游记中呈现出来的氛围和特点。
关键词可以是：情绪（例如“放松”、“浪漫”、“刺激”）、节奏（例如“步行友好”、“节奏很快”）、消费感受（例如“物价便宜”、“比较贵”）、适合人群（例如“适合情侣”、“适合亲子”）、环境特点（例如“夜景好看”、“街区很文艺”等）。

【游记内容】
{short_text}

【输出要求】
1. 只输出 JSON 数组，不要输出任何解释性文字。
2. JSON 数组元素是简短的中文短语，例如：
   ["浪漫", "适合步行", "美食丰富", "夜景好看", "物价略贵"]
"""

    try:
        resp = qianfan_client.chat.completions.create(
            model=TAG_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个擅长提炼城市旅行氛围关键词的助手。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=200,
        )

        content = resp.choices[0].message["content"]
        # 尝试按 JSON 解析
        vibes = json.loads(content)
        if isinstance(vibes, list):
            cleaned = [v.strip() for v in vibes if isinstance(v, str) and v.strip()]
            return cleaned[:10]
        return []
    except Exception as e:
        print("extract_city_vibes error:", e)
        return []


# -----------------------
# Step 1: Load local embedding model
# -----------------------
print("Loading SentenceTransformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")


# -----------------------
# Step 2: Chunk function
# -----------------------
def chunk_text(text, max_tokens=200):
    words = str(text).split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunks.append(" ".join(words[i : i + max_tokens]))
    return chunks


# -----------------------
# Step 3: Collect all CSVs
# -----------------------
def load_all_csv():
    medium_dir = os.path.join(DATA_DIR, "medium")
    reddit_dir = os.path.join(DATA_DIR, "reddit")

    medium_files = (
        [os.path.join(medium_dir, f) for f in os.listdir(medium_dir) if f.endswith(".csv")]
        if os.path.exists(medium_dir)
        else []
    )
    reddit_files = (
        [os.path.join(reddit_dir, f) for f in os.listdir(reddit_dir) if f.endswith(".csv")]
        if os.path.exists(reddit_dir)
        else []
    )

    print(f"Found Medium CSVs: {len(medium_files)}")
    print(f"Found Reddit CSVs: {len(reddit_files)}")

    return medium_files + reddit_files


# -----------------------
# Step 4: Chunk the CSV content
# -----------------------
def infer_city_from_path(csv_path: str) -> str:
    # 示例：data/medium/paris_medium_posts.csv -> paris
    base = os.path.basename(csv_path).lower()
    # 你可以按需要自己增减城市名
    for token in ["paris", "budapest", "rome", "london", "tokyo", "kyoto"]:
        if token in base:
            return token
    return ""


def build_chunks(csv_files):
    chunks = []
    metadata = []

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[ERROR] Cannot read {csv_path}: {e}")
            continue

        # 猜字段名
        title_col = None
        url_col = None
        text_col = None

        for c in df.columns:
            lc = c.lower()
            if lc == "title":
                title_col = c
            if lc == "url":
                url_col = c
            if lc == "content":  # Medium
                text_col = c
            if lc == "selftext":  # Reddit
                text_col = c

        # 再兜底：取第 2 列当正文
        if text_col is None and len(df.columns) >= 2:
            text_col = df.columns[1]

        city = infer_city_from_path(csv_path)

        print(f"Processing {csv_path} (city={city or '未知'}) rows={len(df)}")

        for i, row in df.iterrows():
            raw_text = str(row.get(text_col, ""))
            if not raw_text.strip():
                continue

            title = str(row.get(title_col, "")) if title_col else ""
            url = str(row.get(url_col, "")) if url_col else ""

            # ⚠️ 用“标题 + 正文开头”作为标签输入
            tag_input = (title + "\n" + raw_text).strip()
            vibes = extract_city_vibes(tag_input, city or "这座城市")

            # 对正文做分块，每个 chunk 共用同一份 metadata（包括 vibes）
            for c in chunk_text(raw_text):
                chunks.append(c)
                metadata.append(
                    {
                        "source": csv_path,
                        "row": int(i),
                        "content": c,
                        "title": title,
                        "url": url,
                        "city": city,
                        "vibes": vibes,  # 👈 把氛围标签写进 metadata
                    }
                )

    return chunks, metadata


# -----------------------
# Step 5: Vectorize using local model
# -----------------------
def vectorize_chunks(chunks):
    embeddings = []

    for c in tqdm(chunks, desc="Embedding chunks"):
        vec = embedder.encode(c)
        embeddings.append(vec)

    return np.array(embeddings).astype("float32")


# -----------------------
# Step 6: Save vector store
# -----------------------
def save_vector_store(embeddings, metadata, chunks):
    if embeddings.shape[0] == 0:
        print("❌ ERROR: No embeddings generated. Cannot save vector store.")
        return

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, f"{VECTOR_DIR}/index.faiss")

    with open(f"{VECTOR_DIR}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(f"{VECTOR_DIR}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Vector store saved!")


# -----------------------
# Step 7: 聚合所有城市的关键词，写入 city_vibes.json
# -----------------------
def build_city_vibes(metadata):
    """
    从所有 metadata 中聚合每个城市的 vibes 关键词，统计频次，写入一个文件：
    VECTOR_DIR/city_vibes.json

    结构示例：
    {
      "paris": {
        "vibes": ["浪漫", "适合步行", "美食丰富"],
        "counts": {"浪漫": 12, "适合步行": 8, ...}
      },
      "budapest": {
        "vibes": [...],
        "counts": {...}
      }
    }
    """
    city_counters = {}

    for md in metadata:
        city = (md.get("city") or "").strip().lower()
        if not city:
            continue

        vibes = md.get("vibes", [])
        if not isinstance(vibes, list):
            continue

        if city not in city_counters:
            city_counters[city] = Counter()

        for v in vibes:
            v = str(v).strip()
            if v:
                city_counters[city][v] += 1

    summary = {}
    for city, counter in city_counters.items():
        top_vibes = [w for w, _ in counter.most_common(15)]
        summary[city] = {
            "vibes": top_vibes,
            "counts": dict(counter),
        }

    out_path = os.path.join(VECTOR_DIR, "city_vibes.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✅ city_vibes.json saved to {out_path}")


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    csv_files = load_all_csv()
    chunks, metadata = build_chunks(csv_files)

    # 先构建城市关键词总表
    build_city_vibes(metadata)

    # 再向量化并保存向量库
    embeddings = vectorize_chunks(chunks)
    print("Embeddings shape:", embeddings.shape)
    save_vector_store(embeddings, metadata, chunks)