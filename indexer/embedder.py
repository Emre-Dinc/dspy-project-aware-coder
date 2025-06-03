import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())        
_EMB_MODEL = os.environ["_EMB_MODEL"]
_encoder = SentenceTransformer(_EMB_MODEL, device="cpu")


def embed(texts: list[str]) -> list[np.ndarray]:
    """Return L2-normalised 768-d float32 vectors."""
    vecs = _encoder.encode(texts, normalize_embeddings=True, batch_size=16)
    return [v.astype("float32") for v in vecs]
