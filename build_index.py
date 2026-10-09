import pickle
from pathlib import Path
from PIL import Image
import numpy as np
import faiss

from backend.encoder import MultimodalEncoder
import backend.config as config

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def main():
    print("1. Loading OpenCLIP model into memory...")
    encoder = MultimodalEncoder()

    # Collect images from data folder
    image_paths = [p for p in config.DATA_DIR.glob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    
    if not image_paths:
        print(f"⚠️ No images found in '{config.DATA_DIR}'. Please add image files first.")
        return

    print(f"2. Processing {len(image_paths)} image(s)...")
    
    embeddings = []
    metadata_list = []

    for idx, path in enumerate(image_paths):
        try:
            image = Image.open(path).convert("RGB")
            vector = encoder.encode_image(image)
            
            embeddings.append(vector)
            metadata_list.append({
                "id": idx,
                "file_path": str(path),
                "path": str(path),
                "filename": path.name
            })
            print(f"  [{idx + 1}/{len(image_paths)}] Encoded: {path.name}")
        except Exception as e:
            print(f"  ❌ Failed to process {path.name}: {e}")

    # Convert to NumPy array
    embedding_matrix = np.array(embeddings, dtype=np.float32)

    print("3. Creating FAISS Index (Inner Product for Cosine Similarity)...")
    index = faiss.IndexFlatIP(config.VECTOR_DIMENSION)
    index.add(embedding_matrix)

    print("4. Saving index and metadata files to disk...")
    faiss.write_index(index, config.INDEX_FILE)
    with open(config.METADATA_FILE, "wb") as f:
        pickle.dump(metadata_list, f)

    print(f"✅ Indexing complete! Saved to {config.INDEX_FILE} and {config.METADATA_FILE}")

if __name__ == "__main__":
    main()