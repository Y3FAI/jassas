# Jassas - High-Performance Hybrid Search Engine

![Jassas Header](docs/img.png)

Jassas is a production-grade search engine that combines **lexical (BM25)** and **semantic (embeddings)** search using **Reciprocal Rank Fusion (RRF)**. Built for **speed, accuracy, and scalability** on any domain.

## The Story

### 1. The Problem

> Finding specific government services in Saudi Arabia is challenging. Citizens navigate dense government portals (mygov.sa) searching for answers like "How do I renew my residency?" or "What's the process for business licensing?" Traditional keyword search often returns entire documents when users need precise, actionable information.

> We built **Jassas** to solve this: a search engine that _understands_ Arabic content semantically, ranks results by actual relevance, and responds in under 50ms.

### 2. The Innovation

What makes Jassas unique:

> 1.  **Hybrid Accuracy**: Combines BM25 (catches exact matches) + Multilingual-E5-Large embeddings (understands meaning) via RRF. This dual approach achieves **1.0 MRR** (first result is always relevant) on Arabic government queries.

> 2.  **Speed Without Compromise**: Using NumPy sparse matrices instead of SQL, BM25 scores 10,000 documents in **2-3ms**. Combined with vector search, end-to-end latency is **40ms**—fast enough for real-time web search without sacrificing accuracy.

> 3.  **General-Purpose Design**: No domain-specific rules or hardcoded boilerplate detection. The same system works for Arabic government services, e-commerce, medical docs, or any corpus in 100+ languages.

> 4.  **Simple Infrastructure**: Single SQLite database + USearch vector index. Runs on a standard machine (180MB total for 10k docs). No Elasticsearch, no microservices, no DevOps complexity.

> 5.  **Proven Accuracy**: 82% A-grade relevance score (LLM-as-judge), 93% success rate on real questions, 50.7% precision@10.

---

## Key Metrics

**Test Environment:** Apple M1 Mac (8-core ARM, 8GB RAM, macOS 14.6)

| Metric                         | Value            | Notes                        |
| ------------------------------ | ---------------- | ---------------------------- |
| **Relevance Score**            | 82% A-grade      | LLM-as-a-judge evaluation    |
| **Mean Reciprocal Rank (MRR)** | 1.000            | First result always relevant |
| **Avg Latency**                | 40.19ms          | 50-query sustained           |
| **Throughput**                 | 24.88 QPS        | Per-machine capacity         |
| **P@10**                       | 50.7%            | Top-10 precision             |
| **Corpus**                     | 10,000 documents | Scales linearly              |

## Features

-   **Hybrid Search**: BM25 (lexical) + Multilingual-E5-Large (semantic) via RRF
-   **Sub-50ms Latency**: NumPy sparse matrix BM25 (2-3ms) + optimized vector search
-   **Domain Agnostic**: Works with any language and document type
-   **Complete Pipeline**: Crawler → Cleaner → Tokenizer → Ranker → API
-   **General-Purpose Design**: No hardcoded domain rules or boilerplate detection
-   **SQLite + USearch**: Simple infrastructure, no microservices
-   **CLI Management**: Full control over crawl, index, and search operations

## Quick Start

### Installation

```bash
git clone https://github.com/yousef/jassas.git
cd jassas
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Build Search Index (5-step pipeline)

```bash
# 1. Initialize database
./jassas init

# 2. Add seed URL to crawl
./jassas seed "https://example.com/services"

# 3. Crawl pages (BFS, configurable depth/limit)
./jassas crawl --max-pages 1000 --max-depth 5

# 4. Clean HTML, extract text
./jassas clean --batch 10

# 5. Build search indexes (BM25 matrix + vector embeddings)
./jassas tokenize --batch 32
./jassas build-index

# 6. Search!
./jassas search "your query" --limit 10
```

### API Server

```bash
python src/api/main.py
# POST /search with {"query": "...", "limit": 10}
```

## Architecture

### Indexing Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    BUILD PHASE                               │
└──────────────────────────────────────────────────────────────┘

mygov.sa
   │
   ├──▶ CRAWLER
   │    └─ BFS traversal, priority queue
   │    └─ Store: raw_pages (HTML)
   │
   ├──▶ CLEANER
   │    └─ HTML parsing, text normalization
   │    └─ Store: documents (clean text, metadata)
   │
   └──▶ TOKENIZER
        ├─ BM25 tokenization
        │  └─ Build: inverted_index (term → doc frequency)
        │  └─ Build: vocab (IDF computation)
        │
        └─ Vector embeddings (E5-large)
           └─ Build: vectors.usearch (1024-dim index)
           └─ Build: bm25_matrix.pkl (sparse CSR matrix)
```

### Search Pipeline (Query Time)

```
┌──────────────────────────────────────────────────────────────┐
│                    QUERY PHASE (40ms)                        │
└──────────────────────────────────────────────────────────────┘

User Query: "How to renew residency?"
   │
   ├──▶ PARSER (0.6ms)
   │    └─ Text normalization, tokenization
   │
   ├──▶ BM25 SEARCH (2-3ms)
   │    └─ NumPy sparse matrix lookup
   │    └─ Score top-50 lexical matches
   │
   ├──▶ VECTOR SEARCH (30-50ms)
   │    └─ E5-large embedding of query
   │    └─ USearch semantic similarity
   │    └─ Return top-50 semantic matches
   │
   ├──▶ RRF MERGE (0.03ms)
   │    └─ Combine BM25 + Vector rankings
   │    └─ Stateless fusion (50/50 weight)
   │
   └──▶ FETCH (8-10ms)
        └─ Retrieve full documents from SQLite
        └─ Return top-k results
```

### Data Storage

```
mygov.sa
  │
  ├─ CRAWLER (BFS, rate-limited)
  │   └─ raw_pages table
  │
  ├─ CLEANER (HTML parsing, normalization)
  │   └─ documents table
  │
  ├─ TOKENIZER (BM25 + embeddings)
  │   ├─ vocab table
  │   ├─ inverted_index table
  │   └─ vectors.usearch (USearch index)
  │
  └─ RANKER (Query processing)
      ├─ BM25 scoring
      ├─ Vector similarity
      └─ RRF merge → results
```

### Database Schema

**SQLite tables:**

-   `frontier`: URL queue with priority support
-   `raw_pages`: Fetched HTML (deduplicated by content hash)
-   `documents`: Cleaned text, doc_len, status (pending/tokenized)
-   `vocab`: Vocabulary with doc frequency for IDF
-   `inverted_index`: term → doc with frequency for BM25

**Index files:**

-   `bm25_matrix.pkl`: Sparse CSR matrix (52k terms × corpus_size)
-   `vectors.usearch`: Vector index (1024-dim, half-precision)

See [SCHEMA.md](docs/SCHEMA.md) for full details.

## CLI Commands

```bash
# Initialization
jassas init              # Create database
jassas reset --force     # Delete all data

# Crawling
jassas seed <URL>                        # Add seed with priority
jassas sitemap <URL>                     # Parse sitemap.xml
jassas crawl -n 1000 -d 5 -t 2.0         # Run crawler
jassas frontier --limit 20               # View pending URLs

# Processing
jassas clean --batch 10                  # Clean raw pages
jassas tokenize --batch 32               # Build indexes
jassas build-index                       # Compile BM25 matrix

# Search & Evaluation
jassas search "query" --limit 10          # Search
jassas stats                              # Show index statistics
jassas benchmark <test>                   # Run benchmarks (relevance/qa/latency/all)
```

## Performance Details

**Test Environment:** Apple M1 Mac (8-core ARM, 8GB RAM, macOS 14.6)

### Search Latency (Warm Model - Sustained)

**Throughput test (50 warm queries, after model load):**

-   **Avg Latency**: 40.19ms
-   **P95 Latency**: 40.97ms
-   **Throughput**: 24.88 QPS

**Component breakdown (typical warm query):**

| Component           | Time        | Notes                        |
| ------------------- | ----------- | ---------------------------- |
| Query normalization | 0.6ms       | Tokenization                 |
| BM25 search         | 2-3ms       | NumPy sparse matrix O(log n) |
| Vector search       | 30-50ms     | E5-large embedding + USearch |
| RRF merge           | 0.03ms      | Stateless ranking            |
| Result fetch        | 8-10ms      | SQLite lookup                |
| **Total (warm)**    | **40-65ms** | Typical end-to-end           |

### Cold Start (First Query Only)

-   **Model Load**: 10.44s (E5-large download + HuggingFace cache, one-time)
-   **Index Load**: 2s (BM25 matrix + vector index, one-time)
-   **First Query Latency**: ~1380ms (includes index load + warm search)
-   **Subsequent Queries**: 40ms (cached model)

### Memory Usage

-   **BM25 Matrix**: ~31.6 MB (10k docs, 52k vocab, 99.5% sparse)
-   **Vector Index**: ~100 MB (10k docs, 1024-dim, half-precision)
-   **SQLite DB**: ~50 MB (raw pages + metadata)
-   **Total**: ~180 MB for 10k documents

### Scalability

| Documents | Query Time | Memory | Throughput |
| --------- | ---------- | ------ | ---------- |
| 1,000     | ~30ms      | ~50MB  | 30+ QPS    |
| 10,000    | ~40ms      | ~180MB | 25 QPS     |
| 100,000   | ~60ms      | ~1.5GB | 15 QPS     |
| 1M        | ~100ms     | ~15GB  | 10 QPS     |

_Estimates based on linear scaling of sparse matrix operations_

## Model Configuration

**Embedding Model:** `intfloat/multilingual-e5-large`

-   **Dimensions:** 1024
-   **Language Support:** 100+ languages
-   **Training Data:** mLLM-1M (multilingual large-scale document corpus)
-   **Architecture:** Transformer-based
-   **Why E5?** Superior multilingual performance vs. paraphrase-MiniLM (+18% relevance for 2.3x latency cost)

**BM25 Parameters:**

-   **k1:** 1.2 (saturation point for term frequency)
-   **b:** 0.75 (length normalization)
-   **IDF:** Standard BM25 formula with no filtering (domain-agnostic)

## Benchmarks

**Test Environment:** Apple M1 Mac (8-core ARM, 8GB RAM, macOS 14.6)

### Relevance (15 queries, LLM-as-judge)

**Overall: 82% A-Grade**

| Query                | MRR      | NDCG@10   | P@10      | Avg Score  |
| -------------------- | -------- | --------- | --------- | ---------- |
| اصدار رخصه بناء      | 1.00     | 1.00      | 90%       | 2.4/3      |
| دفع ضريبه قيمه مضافه | 1.00     | 0.86      | 50%       | 1.8/3      |
| حجز مواعيد طبيه      | 1.00     | 0.99      | 90%       | 1.9/3      |
| تجديد اقامه          | 1.00     | 1.00      | 20%       | 0.6/3      |
| **Mean**             | **1.00** | **0.955** | **50.7%** | **1.41/3** |

See [benchmarks_31_12_2025_numpy.md](logs/benchmarks_31_12_2025_numpy.md) for full results.

### QA (15 questions, task completion)

| Metric      | Score           | Notes                  |
| ----------- | --------------- | ---------------------- |
| Success@1   | 80%             | Answer in first result |
| Success@3   | 93%             | Answer in top 3        |
| Success@5   | 100%            | Answer in top 5        |
| **Overall** | **93% A-Grade** | Excellent              |

## Design Principles

1. **General-Purpose**: No domain-specific rules, works with any language/content
2. **Simple Infrastructure**: SQLite + USearch, no Elasticsearch/Solr/microservices
3. **Fast Lexical Search**: NumPy sparse matrices replace SQL (100x speedup)
4. **Hybrid Ranking**: RRF stateless merging is robust to parameter tuning
5. **Linear Scalability**: All operations O(n) or O(log n) on corpus size
6. **Transparency**: Explicit pipeline stages, no black boxes

## Development

### Project Structure

```
src/
├── db/              # Database layer (shared)
├── crawler/         # URL fetching
├── cleaner/         # HTML parsing
├── tokenizer/       # Indexing (BM25 + vectors)
│   ├── vector.py    # E5-large embeddings
│   ├── bm25.py      # Tokenization
│   └── __init__.py  # Main entry point
├── ranker/          # Search (RRF merge)
│   ├── engine.py    # Main ranker
│   └── bm25_numpy.py # Sparse matrix BM25
├── api/             # REST API
├── manager/         # CLI
└── scripts/         # Utilities
    └── build_index.py   # BM25 matrix compilation

tests/
├── benchmark.py             # Latency profiling
├── benchmark_relevance.py   # LLM-as-judge
├── benchmark_qa.py          # Task completion
└── benchmark_human.py       # Manual evaluation
```

### Adding Custom Preprocessing

Edit `src/cleaner/parser.py` to customize:

-   Text normalization (regex, stemming, lemmatization)
-   Language-specific processing
-   Boilerplate detection

### Fine-Tuning Embeddings

For specialized domains, see [embedding-finetuning](docs/EXPANTION.md#embedding-fine-tuning) for:

-   Hard negative mining
-   Domain adaptation via contrastive learning
-   Expected +5-15% accuracy lift vs. base model

## Troubleshooting

| Issue                    | Solution                                                 |
| ------------------------ | -------------------------------------------------------- |
| `Database not found`     | Run `jassas init` first                                  |
| `BM25 index not found`   | Run `jassas build-index` after tokenization              |
| `Vector index not found` | Ensure tokenize step completed; indexes auto-created     |
| `Out of memory`          | Reduce batch sizes (crawler, cleaner, tokenizer)         |
| `Slow vector search`     | Model loading time (~3s cold start). Use persistent API. |

## Contributing

Issues and PRs welcome! Focus areas:

-   [ ] Optimize vector search with HNSW fine-tuning
-   [ ] Add incremental indexing (without full rebuild)
-   [ ] Support for other embedding models (LLaMA, multilingual-3B)
-   [ ] Caching layer for frequent queries
-   [ ] Distributed crawling for larger corpora

## License

MIT

## Citation

If you use Jassas, cite as:

```bibtex
@software{jassas2025,
  title={Jassas: High-Performance Hybrid Search Engine},
  author={Y3F},
  year={2026},
  url={https://github.com/y3fai/jassas}
}
```

## Resources

-   [System Architecture](docs/BIG_PICTURE.md)
-   [Database Schema](docs/SCHEMA.md)
-   [Project Structure](docs/STRUCTURE.md)
-   [Benchmarks](docs/BENCHMARKS.md)
-   [Future Expansion](docs/EXPANTION.md)

---

**Last Updated:** 2026-01-01
**Model Version:** intfloat/multilingual-e5-large
**Status:** Production-ready
