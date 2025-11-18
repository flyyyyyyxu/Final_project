import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer


DATA_DIR = "./data"
VECTOR_DIR = "./vector_store"
os.makedirs(VECTOR_DIR, exist_ok=True)

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
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunks.append(" ".join(words[i:i + max_tokens]))
    return chunks


# -----------------------
# Step 3: Collect all CSVs
# -----------------------
def load_all_csv():
    medium_dir = os.path.join(DATA_DIR, "medium")
    reddit_dir = os.path.join(DATA_DIR, "reddit")

    medium_files = [os.path.join(medium_dir, f) for f in os.listdir(medium_dir) if f.endswith(".csv")]
    reddit_files = [os.path.join(reddit_dir, f) for f in os.listdir(reddit_dir) if f.endswith(".csv")]

    print(f"Found Medium CSVs: {len(medium_files)}")
    print(f"Found Reddit CSVs: {len(reddit_files)}")

    return medium_files + reddit_files


# -----------------------
# Step 4: Chunk the CSV content
# -----------------------
def build_chunks(csv_files):
    chunks = []
    metadata = []

    for csv_path in tqdm(csv_files, desc="Chunking CSVs"):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[ERROR] Cannot read {csv_path}: {e}")
            continue

        # Automatically guess text column
        text_col = df.columns[1]

        for i, row in df.iterrows():
            text = str(row[text_col])
            for c in chunk_text(text):
                chunks.append(c)
                metadata.append({
                    "source": csv_path,
                    "row": i,
                    "content": c
                })

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

    with open(f"{VECTOR_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(f"{VECTOR_DIR}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Vector store saved!")


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    csv_files = load_all_csv()
    chunks, metadata = build_chunks(csv_files)
    embeddings = vectorize_chunks(chunks)
    print("Embeddings shape:", embeddings.shape)
    save_vector_store(embeddings, metadata, chunks)