import os
import json
import pickle
from pathlib import Path
import numpy as np
import torch
import faiss
import open_clip

import backend.config as config

def load_resources():
    print("Loading OpenCLIP model...")
    model, _, _ = open_clip.create_model_and_transforms(
        config.MODEL_NAME, 
        pretrained=config.PRETRAINED_WEIGHTS
    )
    tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)
    model.to(config.DEVICE)
    model.eval()

    print("Loading FAISS index...")
    index = faiss.read_index(config.INDEX_FILE)
    with open(config.METADATA_FILE, "rb") as f:
        metadata_list = pickle.load(f)

    # Extract filenames directly from metadata
    filenames = []
    for item in metadata_list:
        if isinstance(item, dict) and "filename" in item:
            filenames.append(item["filename"])
        else:
            p = item.get("file_path") or item.get("path") if isinstance(item, dict) else str(item)
            filenames.append(os.path.basename(p))

    return model, tokenizer, index, filenames

def search_text(query_str, model, tokenizer, index, filenames, top_k=10):
    with torch.no_grad():
        tokens = tokenizer([query_str]).to(config.DEVICE)
        text_features = model.encode_text(tokens).cpu().numpy()
        
    norm = np.linalg.norm(text_features, axis=1, keepdims=True)
    text_features = text_features / np.maximum(norm, 1e-10)

    distances, indices = index.search(text_features.astype("float32"), min(top_k, index.ntotal))
    
    retrieved_results = []
    for idx, score in zip(indices[0], distances[0]):
        if idx != -1 and idx < len(filenames):
            retrieved_results.append((filenames[idx], float(score)))
            
    return retrieved_results

def calculate_metrics(eval_dataset, model, tokenizer, index, filenames, top_k=5, verbose=False):
    recalls = []
    mrrs = []
    aps = []

    for item in eval_dataset:
        query = item["query"]
        relevant_set = {os.path.basename(p) for p in item["relevant_files"]}
        
        search_hits = search_text(query, model, tokenizer, index, filenames, top_k=top_k)
        retrieved = [fname for fname, _ in search_hits]
        
        # 1. Recall@K
        retrieved_set = set(retrieved)
        hits = len(relevant_set.intersection(retrieved_set))
        recall = hits / len(relevant_set) if len(relevant_set) > 0 else 0.0
        recalls.append(recall)

        # 2. Reciprocal Rank (MRR)
        rr = 0.0
        for rank, filename in enumerate(retrieved, start=1):
            if filename in relevant_set:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

        # 3. Average Precision (AP@K)
        score = 0.0
        num_hits = 0.0
        for i, filename in enumerate(retrieved):
            if filename in relevant_set:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        ap = score / min(len(relevant_set), top_k) if len(relevant_set) > 0 else 0.0
        aps.append(ap)

        # Optional Diagnostic Output on missing top-1 match
        if verbose and rr == 0.0:
            print(f"\n⚠️  Query: '{query}' | Expected: {list(relevant_set)}")
            print("   Top retrieved items:")
            for rank, (fname, dist) in enumerate(search_hits[:3], start=1):
                print(f"     Rank {rank}: {fname} (Cosine Similarity: {dist:.4f})")

    return {
        f"Recall@{top_k}": np.mean(recalls),
        "MRR": np.mean(mrrs),
        f"MAP@{top_k}": np.mean(aps)
    }

if __name__ == "__main__":
    if not os.path.exists("eval_queries.json"):
        print("❌ 'eval_queries.json' not found. Please create it first.")
        exit(1)

    with open("eval_queries.json", "r") as f:
        eval_data = json.load(f)

    model, tokenizer, index, filenames = load_resources()

    print("\n================ EVALUATION RESULTS ================")
    for k in [1, 3, 5, 10]:
        # Diagnostic verbose mode enabled for top-1 check
        verbose_flag = (k == 5)
        metrics = calculate_metrics(eval_data, model, tokenizer, index, filenames, top_k=k, verbose=verbose_flag)
        
        print(f"\n--- Top-K = {k} ---")
        for metric_name, val in metrics.items():
            print(f"  {metric_name}: {val:.4f}")