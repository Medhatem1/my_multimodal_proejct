import torch
from pathlib import Path

# Base Paths (Points to audiovisual_search root)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Vector Database File Paths
INDEX_FILE = str(BASE_DIR / "faiss_index.bin")
METADATA_FILE = str(BASE_DIR / "metadata.pkl")

# OpenCLIP Settings
MODEL_NAME = "ViT-B-32"
PRETRAINED_WEIGHTS = "laion2b_s34b_b79k"
VECTOR_DIMENSION = 512

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"