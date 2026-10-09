import os
import pickle
import io
import torch
import numpy as np
import faiss
import open_clip
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import backend.config as config

# ---------------------------------------------------------
# 1. INITIALIZE FASTAPI APP & CORS
# ---------------------------------------------------------
app = FastAPI(
    title="Multimodal Audiovisual Search API",
    description="Backend API powering OpenCLIP + FAISS vector search with score filtering."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 2. STATIC FILE MOUNTING
# ---------------------------------------------------------
os.makedirs(config.DATA_DIR, exist_ok=True)
app.mount("/data", StaticFiles(directory=str(config.DATA_DIR)), name="data")

# ---------------------------------------------------------
# 3. LOAD OPENCLIP MODEL
# ---------------------------------------------------------
print(f"Loading OpenCLIP model on device: {config.DEVICE}...")

model, _, preprocess = open_clip.create_model_and_transforms(
    config.MODEL_NAME, 
    pretrained=config.PRETRAINED_WEIGHTS
)
tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)
model.to(config.DEVICE)
model.eval()

# ---------------------------------------------------------
# 4. LOAD FAISS INDEX 
# ---------------------------------------------------------
index = None
metadata_list = []

def load_vector_store():
    global index, metadata_list
    if os.path.exists(config.INDEX_FILE) and os.path.exists(config.METADATA_FILE):
        try:
            index = faiss.read_index(config.INDEX_FILE)
            with open(config.METADATA_FILE, "rb") as f:
                metadata_list = pickle.load(f)
            print(f"✅ Loaded FAISS index with {index.ntotal} vectors.")
        except Exception as e:
            print(f"❌ Error loading index/metadata: {e}")
            index = None
            metadata_list = []
    else:
        print("⚠️ Index files not found. Please run build_index.py first.")

load_vector_store()

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10
    return vec / norm

def normalize_score_to_percentage(raw_score: float) -> float:
    """
    Maps standard OpenCLIP cosine similarity bounds [0.15, 0.35] 
    to a human-readable 0% - 100% scale.
    """
    min_val = 0.15
    max_val = 0.35
    
    scaled = (raw_score - min_val) / (max_val - min_val)
    scaled = max(0.0, min(1.0, scaled))
    return round(scaled * 100, 1)

def search_faiss(query_vec: np.ndarray, top_k: int = 6, min_score: float = 0.0):
    if index is None or index.ntotal == 0:
        return []
    
    query_vec = normalize_vector(query_vec).astype("float32")
    distances, indices = index.search(query_vec, min(top_k, index.ntotal))
    
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx != -1 and idx < len(metadata_list):
            item = metadata_list[idx]
            
            if isinstance(item, str):
                meta = {"path": item, "file_path": item}
            elif isinstance(item, dict):
                meta = item
            else:
                meta = {"path": str(item), "file_path": str(item)}
            
            raw_cosine = float(score)
            percentage_score = normalize_score_to_percentage(raw_cosine)
            
            # Apply threshold filtering
            if percentage_score >= min_score:
                results.append({
                    "raw_score": raw_cosine,
                    "score": percentage_score,
                    "metadata": meta,
                    "file_path": meta.get("file_path") or meta.get("path")
                })
            
    return results

# ---------------------------------------------------------
# 5. API ENDPOINTS
# ---------------------------------------------------------

@app.get("/")
def read_root():
    return {
        "status": "online",
        "total_vectors": index.ntotal if index else 0,
        "device": config.DEVICE
    }

@app.post("/api/search/text")
async def search_by_text(
    query: str = Form(...), 
    top_k: int = Form(6),
    min_score: float = Form(0.0)
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
        
    with torch.no_grad():
        tokens = tokenizer([query]).to(config.DEVICE)
        text_features = model.encode_text(tokens)
        text_vec = text_features.cpu().numpy()
        
    results = search_faiss(text_vec, top_k=top_k, min_score=min_score)
    return {"query": query, "min_score": min_score, "results": results}

@app.post("/api/search/image")
async def search_by_image(
    file: UploadFile = File(...), 
    top_k: int = Form(6),
    min_score: float = Form(0.0)
):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")
        
    with torch.no_grad():
        img_tensor = preprocess(image).unsqueeze(0).to(config.DEVICE)
        image_features = model.encode_image(img_tensor)
        img_vec = image_features.cpu().numpy()
        
    results = search_faiss(img_vec, top_k=top_k, min_score=min_score)
    return {"filename": file.filename, "min_score": min_score, "results": results}

@app.post("/api/search/hybrid")
async def search_hybrid(
    file: UploadFile = File(...),
    text_query: str = Form(...),
    alpha: float = Form(0.5),
    top_k: int = Form(6),
    min_score: float = Form(0.0)
):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")
        
    with torch.no_grad():
        img_tensor = preprocess(image).unsqueeze(0).to(config.DEVICE)
        img_features = model.encode_image(img_tensor).cpu().numpy()
        
        tokens = tokenizer([text_query]).to(config.DEVICE)
        txt_features = model.encode_text(tokens).cpu().numpy()

    norm_img = normalize_vector(img_features)
    norm_txt = normalize_vector(txt_features)
    
    fused_vec = (alpha * norm_txt) + ((1.0 - alpha) * norm_img)
    
    results = search_faiss(fused_vec, top_k=top_k, min_score=min_score)
    return {
        "text_query": text_query,
        "alpha": alpha,
        "min_score": min_score,
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)