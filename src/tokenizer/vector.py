"""
Vector Engine - Generates embeddings and manages USearch index.
Uses FastEmbed (ONNX) for fast CPU inference.
"""
import os
from typing import List, Tuple
import numpy as np
from fastembed import TextEmbedding
from usearch.index import Index


# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
INDEX_PATH = os.path.join(DATA_DIR, 'vectors.usearch')


class VectorEngine:
    """Generates embeddings and manages vector index."""

    # Model config - FastEmbed model name
    MODEL_NAME = 'intfloat/multilingual-e5-large'
    DIMENSIONS = 1024

    # USearch config (from article optimization)
    INDEX_CONFIG = {
        'ndim': 1024,
        'metric': 'cos',
        'dtype': 'f16',           # Half precision (2x memory savings)
        'connectivity': 32,       # Balanced quality
        'expansion_add': 200,     # Better graph construction
        'expansion_search': 100,  # Better recall
    }

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.model = None
        self.index = None

    def load_model(self):
        """Load the FastEmbed model."""
        if self.model is None:
            self.model = TextEmbedding(self.MODEL_NAME)

    def create_index(self):
        """Create a new USearch index."""
        self.index = Index(**self.INDEX_CONFIG)

    def load_index(self) -> bool:
        """Load existing index from disk. Returns False if not found."""
        if os.path.exists(INDEX_PATH):
            self.index = Index.restore(INDEX_PATH)
            return True
        return False

    def save_index(self):
        """Save index to disk."""
        if self.index:
            os.makedirs(DATA_DIR, exist_ok=True)
            self.index.save(INDEX_PATH)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        self.load_model()
        # FastEmbed returns generator, convert to numpy
        embeddings = list(self.model.embed(texts))
        return np.array(embeddings)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings in batches."""
        self.load_model()
        # FastEmbed handles batching internally
        embeddings = list(self.model.embed(texts))
        return np.array(embeddings)

    def add_documents(self, doc_ids: List[int], texts: List[str]):
        """
        Add documents to the index.

        Args:
            doc_ids: List of document IDs (keys for the index)
            texts: List of text content to embed
        """
        if not texts:
            return

        # Generate embeddings
        embeddings = self.encode_batch(texts)

        # Add to index
        if self.index is None:
            self.create_index()

        # Convert to numpy arrays
        keys = np.array(doc_ids, dtype=np.int64)
        vectors = embeddings.astype(np.float16)  # Match f16 dtype

        self.index.add(keys, vectors)

    def search(self, query: str, limit: int = 10) -> List[Tuple[int, float]]:
        """
        Search for similar documents.

        Args:
            query: Search query text
            limit: Number of results to return

        Returns:
            List of (doc_id, score) tuples
        """
        if self.index is None or len(self.index) == 0:
            return []

        # Encode query
        query_embedding = self.encode([query])[0].astype(np.float16)

        # Search
        matches = self.index.search(query_embedding, limit)

        # Convert to list of tuples
        results = []
        for key, distance in zip(matches.keys, matches.distances):
            # USearch returns distance, convert to similarity for cosine
            # For cosine metric, distance = 1 - similarity
            similarity = 1.0 - float(distance)
            results.append((int(key), similarity))

        return results

    def get_count(self) -> int:
        """Get number of vectors in index."""
        return len(self.index) if self.index else 0
