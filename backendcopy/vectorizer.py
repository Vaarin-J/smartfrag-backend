from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import json
from tqdm import tqdm  # <-- global progress bar

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH  = "backendcopy/data/perfumes1.faiss"
ID_MAP_PATH = "backendcopy/data/id_map1.json"

def load_model():
    """Load the SentenceTransformer model."""
    return SentenceTransformer(MODEL_NAME)

def generate_embeddings_batched(texts, model, batch_size=64):
    """Efficiently generate embeddings in batches with one global progress bar."""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)  # Turn off per-batch bars
        embeddings.extend(batch_embeddings)
    return np.array(embeddings).astype("float32")

def build_faiss_index(df: pd.DataFrame, model):
    """Build a FAISS index from fragrance descriptions."""
    
    # Combine main accords and description into structured text
    embedding_input = [
        f"Accords: {', '.join(row['main_accords'])}. {row['description']}"
        for _, row in df.iterrows()
    ]

    # Generate embeddings with unified progress bar
    embeddings = generate_embeddings_batched(embedding_input, model)

    # Build and save FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)

    # Build and save ID map for lookup
    id_map = df[[
        "name", "gender", "main_accords", "rating_value", "rating_count", "rating_score", "url"
    ]].to_dict(orient="records")

    with open(ID_MAP_PATH, "w") as f:
        json.dump(id_map, f)

    print(f"✅ FAISS index saved to: {INDEX_PATH}")
    print(f"✅ ID map saved to: {ID_MAP_PATH}")