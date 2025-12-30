"""
Ranker Engine - Hybrid RRF Search (BM25 + Vector).
The "Judge" that merges lexical and semantic results.
"""
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict
import numpy as np
from usearch.index import Index
from sentence_transformers import SentenceTransformer

from db import Documents, Vocab
from db.connection import get_db
from cleaner.parser import Parser
from tokenizer.bm25 import BM25Tokenizer

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

        # Pre-load stats
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
            self.vector_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

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

        # 1. Execute both searches
        bm25_results = self._bm25_search(normalized_query, limit=50)
        vector_results = self._vector_search(query, limit=50)  # Use original for embeddings

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
        High-performance SQL-based BM25 search.
        Uses CTE to push scoring into SQLite engine.
        """
        # Tokenize query (same as indexing)
        tokens = self.tokenizer.tokenize(query)

        if not tokens:
            return []

        # Get vocab IDs and calculate IDFs
        query_terms = []  # List of (vocab_id, idf)

        for token in tokens:
            vocab = Vocab.get_by_token(token)
            if vocab:
                n = vocab['doc_count']
                # IDF formula: log((N - n + 0.5) / (n + 0.5) + 1)
                idf = math.log((self.total_docs - n + 0.5) / (n + 0.5) + 1)
                query_terms.append((vocab['id'], idf))

        if not query_terms:
            return []

        # Build the SQL query with CTE
        placeholders = ', '.join(['(?, ?)'] * len(query_terms))
        flat_params = [item for pair in query_terms for item in pair]
        flat_params.append(self.avgdl)

        sql = f"""
        WITH query_terms(vocab_id, idf) AS (
            VALUES {placeholders}
        )
        SELECT
            ii.doc_id,
            SUM(
                qt.idf * (
                    (ii.frequency * {self.BM25_K1 + 1}) /
                    (ii.frequency + {self.BM25_K1} * (1 - {self.BM25_B} + {self.BM25_B} * (d.doc_len / ?)))
                )
            ) as score
        FROM inverted_index ii
        JOIN documents d ON d.id = ii.doc_id
        JOIN query_terms qt ON qt.vocab_id = ii.vocab_id
        GROUP BY ii.doc_id
        ORDER BY score DESC
        LIMIT {limit}
        """

        with get_db() as conn:
            cursor = conn.execute(sql, flat_params)
            return [row['doc_id'] for row in cursor.fetchall()]

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
                })

        return results
