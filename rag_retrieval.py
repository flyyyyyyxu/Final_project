# rag_retrieval.py
import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

VECTOR_DIR = "./vector_store"   # 如果你的路径不同，改这里

# 加载本地 embedding 模型（与 ingest 时一致）
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

# 加载 FAISS index + metadata + chunks
_index = None
_metadata = None
_chunks = None
def load_vector_store(vector_dir=VECTOR_DIR):
    global _index, _metadata, _chunks
    if _index is not None:
        return _index, _metadata, _chunks

    idx_path = os.path.join(vector_dir, "index.faiss")
    meta_path = os.path.join(vector_dir, "metadata.json")
    chunks_path = os.path.join(vector_dir, "chunks.pkl")

    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"FAISS index not found: {idx_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json not found: {meta_path}")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"chunks.pkl not found: {chunks_path}")

    _index = faiss.read_index(idx_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        _metadata = json.load(f)
    with open(chunks_path, "rb") as f:
        _chunks = pickle.load(f)

    return _index, _metadata, _chunks

def embed_query(query: str):
    embedder = get_embedder()
    vec = embedder.encode(query)
    return np.array(vec).astype("float32")

def search(query: str, top_k: int = 5):
    """
    返回 list of dicts: [{ 'score': float, 'chunk': str, 'metadata': {...} }, ...]
    """
    index, metadata, chunks = load_vector_store()
    qvec = embed_query(query)
    if qvec.ndim == 1:
        qvec = qvec.reshape(1, -1)
    D, I = index.search(qvec, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        # faiss IndexFlatL2 返回欧式距离（越小越相似）
        entry = {
            "score": float(dist),
            "chunk": chunks[idx] if idx < len(chunks) else metadata[idx].get("content", ""),
            "metadata": metadata[idx] if idx < len(metadata) else {}
        }
        results.append(entry)
    return results