import os
import pickle
import faiss
import numpy as np
import config as config

class FAISSIndexer:
    def __init__(self, vector_dim: int = config.VECTOR_DIMENSION):
        self.vector_dim = vector_dim
        # Inner Product (IP) on L2-normalized vectors computes Cosine Similarity
        self.index = faiss.IndexFlatIP(self.vector_dim)
        self.metadata = []

    def add(self, vector: np.ndarray, meta: dict):
        """Add a single vector and its metadata (e.g., file path)."""
        vector = np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)
        self.index.add(vector)
        self.metadata.append(meta)

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        """Search the FAISS index for top_k nearest neighbors."""
        query_vector = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append({
                    "metadata": self.metadata[idx],
                    "score": float(dist)
                })
        return results

    def save(self, index_file: str = config.INDEX_FILE, metadata_file: str = config.METADATA_FILE):
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, index_file)
        with open(metadata_file, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, index_file: str = config.INDEX_FILE, metadata_file: str = config.METADATA_FILE):
        """Load FAISS index and metadata from disk."""
        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            raise FileNotFoundError("Index or metadata file not found. Run build_index.py first.")
        self.index = faiss.read_index(index_file)
        with open(metadata_file, "rb") as f:
            self.metadata = pickle.load(f)