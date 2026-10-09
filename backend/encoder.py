import torch
import open_clip
from PIL import Image
import numpy as np
import backend.config as config  # ✅ Correct import

class MultimodalEncoder:
    def __init__(self):
        # 1. Load CLIP model, image preprocessing pipeline, and tokenizer
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.MODEL_NAME, 
            pretrained=config.PRETRAINED_WEIGHTS
        )
        self.tokenizer = open_clip.get_tokenizer(config.MODEL_NAME)
        
        # 2. Move model to GPU/CPU and set to evaluation mode
        self.model.to(config.DEVICE)
        self.model.eval()

    def encode_text(self, text: str) -> np.ndarray:
        """Convert a text string into a normalized vector embedding."""
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(config.DEVICE)
            vector = self.model.encode_text(tokens)
            vector /= vector.norm(dim=-1, keepdim=True)  # L2 Normalization for Cosine Similarity
            return vector.cpu().numpy()[0]

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Convert a PIL Image into a normalized vector embedding."""
        with torch.no_grad():
            tensor = self.preprocess(image).unsqueeze(0).to(config.DEVICE)
            vector = self.model.encode_image(tensor)
            vector /= vector.norm(dim=-1, keepdim=True)  # L2 Normalization for Cosine Similarity
            return vector.cpu().numpy()[0]