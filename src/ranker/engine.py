"""
Ranker Engine - Hybrid RRF Search (BM25 + Vector).
The "Judge" that merges lexical and semantic results.
"""
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from usearch.index import Index
from sentence_transformers import SentenceTransformer

from db import Documents, Vocab
from db.connection import get_db
from cleaner.parser import Parser
from tokenizer.bm25 import BM25Tokenizer
from tokenizer.vector import VectorEngine
from ranker.bm25_numpy import NumPyBM25Engine

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
INDEX_PATH = os.path.join(DATA_DIR, 'vectors.usearch')


class Ranker:
    """Hybrid search engine using RRF fusion."""

    # RRF constant (industry standard)
    RRF_K = 60

    # BM25 parameters
    BM25_K1 = 1.2
    BM25_B = 0.75

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.vector_model = None
        self.vector_index = None
        self.parser = Parser()
        self.tokenizer = BM25Tokenizer()

        # Initialize NumPy BM25 Engine (in-memory matrix)
        self.bm25_engine = NumPyBM25Engine(
            index_path=os.path.join(DATA_DIR, 'bm25_matrix.pkl')
        )
        if not self.bm25_engine.load():
            self._log("[yellow]BM25 matrix index not found. Run: python src/scripts/build_index.py[/yellow]")

        # Pre-load stats (for legacy compatibility)
        self.total_docs = 0
        self.avgdl = 0.0
        self._load_stats()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _load_stats(self):
        """Pre-load global stats for BM25."""
        self.total_docs = Documents.get_total_count()
        self.avgdl = Documents.get_avg_doc_len()
        if self.total_docs == 0:
            self._log("Warning: Index is empty.")

    def _load_vector_engine(self):
        """Lazy load vector model and index."""
        if self.vector_model is None:
            self._log("Loading vector model...")
            self.vector_model = SentenceTransformer(VectorEngine.MODEL_NAME)

        if self.vector_index is None and os.path.exists(INDEX_PATH):
            self._log("Loading vector index...")
            self.vector_index = Index.restore(INDEX_PATH, view=True)

    def search(self, query: str, k: int = 10) -> List[dict]:
        """
        Execute hybrid search (BM25 + Vector) and merge via RRF.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of result dicts with score, title, url
        """
        # Normalize query (same as indexing)
        normalized_query = self.parser._normalize(query)

        # 1. Execute both searches in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # BM25 is I/O bound (SQLite)
            future_bm25 = executor.submit(self._bm25_search, normalized_query, 50)
            # Vector is CPU bound (matrix math)
            future_vector = executor.submit(self._vector_search, query, 50)

            bm25_results = future_bm25.result()
            vector_results = future_vector.result()

        # 2. RRF Merge
        merged_scores: Dict[int, float] = {}

        # Add BM25 RRF scores
        for rank, doc_id in enumerate(bm25_results):
            if doc_id not in merged_scores:
                merged_scores[doc_id] = 0.0
            merged_scores[doc_id] += 1.0 / (self.RRF_K + rank + 1)

        # Add Vector RRF scores
        for rank, doc_id in enumerate(vector_results):
            if doc_id not in merged_scores:
                merged_scores[doc_id] = 0.0
            merged_scores[doc_id] += 1.0 / (self.RRF_K + rank + 1)

        # 3. Sort by merged score
        top_doc_ids = sorted(merged_scores, key=merged_scores.get, reverse=True)[:k]

        # 4. Fetch full results
        return self._fetch_results(top_doc_ids, merged_scores)

    def _bm25_search(self, query: str, limit: int = 50) -> List[int]:
        """
        High-performance in-memory BM25 search using NumPy sparse matrices.
        Replaces SQL CTE with vectorized linear algebra.

        Performance: ~2-3ms per query (vs 100-800ms with SQL)
        """
        # Check if BM25 engine loaded successfully
        if self.bm25_engine.term_matrix is None:
            self._log("[yellow]BM25 index not loaded. Using vector search fallback.[/yellow]")
            return []

        # Tokenize query (same as indexing)
        tokens = self.tokenizer.tokenize(query)

        if not tokens:
            return []

        # Convert tokens to vocab IDs
        token_ids = []
        for token in tokens:
            vocab = Vocab.get_by_token(token)
            if vocab:
                token_ids.append(vocab['id'])

        if not token_ids:
            return []

        # Fast NumPy BM25 search (no SQL joins)
        results = self.bm25_engine.search(token_ids, k=limit)

        # Extract just document IDs (scores discarded, merged with vector via RRF)
        return [doc_id for doc_id, score in results]

    def _vector_search(self, query: str, limit: int = 50) -> List[int]:
        """Semantic search via USearch."""
        self._load_vector_engine()

        if self.vector_index is None or self.vector_model is None:
            return []

        if len(self.vector_index) == 0:
            return []

        # Encode query
        embedding = self.vector_model.encode(query).astype(np.float16)

        # Search
        matches = self.vector_index.search(embedding, limit)

        return [int(key) for key in matches.keys]

    def _fetch_results(self, doc_ids: List[int], scores: Dict[int, float]) -> List[dict]:
        """Fetch full document info for results."""
        results = []

        for doc_id in doc_ids:
            doc = Documents.get_by_id(doc_id)
            if doc:
                results.append({
                    'doc_id': doc_id,
                    'score': scores[doc_id],
                    'title': doc['title'],
                    'url': doc['url'],
                    'clean_text': doc.get('clean_text', ''),
                })

        return results
