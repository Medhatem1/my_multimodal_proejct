import json
import pickle
import numpy as np
import torch
import faiss
import open_clip

import backend.config as config

# 1. Load Ground Truth
with open("eval_queries.json", "r") as f:
    eval_data = json.load(f)

first_test = eval_data[0]
query_text = first_test["query"]
expected_files = first_test["relevant_files"]

print(f"--- DEBUGGING QUERY: '{query_text}' ---")
print(f"Expected in eval_queries.json: {expected_files}\n")

# 2. Load FAISS & Metadata
index = faiss.read_index(config.INDEX_FILE)
with open(config.METADATA_FILE, "rb") as f:
    metadata_list = pickle.load(f)

print(f"Total items in FAISS index: {index.ntotal}")
print(f"Sample item #0 in metadata.pkl: {metadata_list[0]}\n")

# 3. Encode Query
model, _, _ = open_clip.create_model_and_transforms(config.MODEL_NAME, pretrained=config.PRETRAINED_WEIGHTS)
tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)
model.to(config.DEVICE).eval()

with torch.no_grad():
    tokens = tokenizer([query_text]).to(config.DEVICE)
    text_emb = model.encode_text(tokens).cpu().numpy()

# Normalize query vector
norm = np.linalg.norm(text_emb, axis=1, keepdims=True)
text_emb = text_emb / np.maximum(norm, 1e-10)

# 4. Search Top 5
distances, indices = index.search(text_emb.astype("float32"), 5)

print("--- TOP 5 RETRIEVED FROM FAISS ---")
for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
    meta = metadata_list[idx] if idx != -1 else "INVALID INDEX"
    print(f"Rank {rank} | Score/Dist: {dist:.4f} | Metadata: {meta}")