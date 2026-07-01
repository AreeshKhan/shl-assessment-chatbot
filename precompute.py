import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def precompute():
    print("Loading catalog...")
    catalog_path = os.path.join("data", "shl_product_catalog.json")
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f, strict=False)
    
    # We must format the text exactly as the agent does
    texts = []
    for item in catalog:
        text = f"{item.get('product_name', '')} {item.get('product_family', '')} {item.get('description', '')}"
        texts.append(text)
    
    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print(f"Generating {len(texts)} embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    
    out_path = os.path.join("data", "embeddings.npy")
    np.save(out_path, embeddings)
    print(f"Saved {embeddings.shape} to {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")

if __name__ == "__main__":
    precompute()
