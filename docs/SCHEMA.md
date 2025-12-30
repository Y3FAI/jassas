# Jassas Database Schema

> SQLite + USearch vector index

## Overview

```
┌───────────────┐     ┌───────────────┐
│   frontier    │     │   raw_pages   │
│   (Queue)     │     │   (Archive)   │
└───────────────┘     └───────┬───────┘
                              │
                              ▼
                      ┌───────────────┐
                      │   documents   │
                      │  (+ doc_len)  │
                      └───────┬───────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      ┌───────────┐   ┌───────────────┐   ┌─────────┐
      │   vocab   │◀──│inverted_index │   │.usearch │
      │(dictionary)│   │ (integers)   │   │(vector) │
      └───────────┘   └───────────────┘   └─────────┘
```

## Table Ownership

| Table             | Service   | Purpose                    |
| ----------------- | --------- | -------------------------- |
| frontier          | Crawler   | URL queue (BFS)            |
| raw_pages         | Crawler   | Raw HTML archive           |
| documents         | Cleaner   | Clean text + metadata      |
| vocab             | Tokenizer | Word dictionary            |
| inverted_index    | Tokenizer | BM25 integer-based index   |
| .usearch file     | Tokenizer | Vector index for semantic  |

---

## Crawler Layer

### frontier

Tracks URLs to visit (The Queue).

```sql
CREATE TABLE frontier (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT UNIQUE NOT NULL,
    status          TEXT DEFAULT 'pending',
    depth           INTEGER DEFAULT 0,
    error_message   TEXT,
    discovered_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_frontier_status ON frontier(status);
CREATE INDEX idx_frontier_depth ON frontier(depth);
```

| Column        | Type     | Notes                                      |
| ------------- | -------- | ------------------------------------------ |
| id            | INTEGER  | PK, auto                                   |
| url           | TEXT     | UNIQUE                                     |
| status        | TEXT     | pending / in_progress / crawled / error    |
| depth         | INTEGER  | BFS priority (0 = seed URL)                |
| error_message | TEXT     | NULL or failure reason                     |
| discovered_at | DATETIME | When URL was found                         |
| updated_at    | DATETIME | Last status change                         |

---

### raw_pages

Stores fetched HTML (The Archive).

```sql
CREATE TABLE raw_pages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT UNIQUE NOT NULL,
    html_content    BLOB,
    content_hash    TEXT,
    http_status     INTEGER,
    crawled_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_raw_pages_hash ON raw_pages(content_hash);
```

| Column       | Type     | Notes                          |
| ------------ | -------- | ------------------------------ |
| id           | INTEGER  | PK, auto                       |
| url          | TEXT     | UNIQUE                         |
| html_content | BLOB     | Raw HTML (can compress later)  |
| content_hash | TEXT     | SHA256 for deduplication       |
| http_status  | INTEGER  | 200, 404, 500, etc.            |
| crawled_at   | DATETIME | When fetched                   |

---

## Cleaner Layer

### documents

Stores cleaned text with metadata for ranking.

```sql
CREATE TABLE documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_page_id     INTEGER NOT NULL,
    url             TEXT NOT NULL,
    title           TEXT,
    clean_text      TEXT,
    doc_len         INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'pending',
    cleaned_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(raw_page_id) REFERENCES raw_pages(id)
);

CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_raw_page ON documents(raw_page_id);
```

| Column      | Type     | Notes                              |
| ----------- | -------- | ---------------------------------- |
| id          | INTEGER  | PK, auto                           |
| raw_page_id | INTEGER  | FK → raw_pages                     |
| url         | TEXT     | Denormalized for fast access       |
| title       | TEXT     | Extracted page title               |
| clean_text  | TEXT     | Normalized content                 |
| doc_len     | INTEGER  | Word count (CRITICAL for BM25)     |
| status      | TEXT     | pending / tokenized                |
| cleaned_at  | DATETIME | When processed                     |

---

## Tokenizer Layer

### vocab

The dictionary - stores each unique word once.

```sql
CREATE TABLE vocab (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    token           TEXT UNIQUE NOT NULL,
    doc_count       INTEGER DEFAULT 0
);

CREATE INDEX idx_vocab_token ON vocab(token);
```

| Column    | Type    | Notes                                  |
| --------- | ------- | -------------------------------------- |
| id        | INTEGER | PK, auto                               |
| token     | TEXT    | UNIQUE stemmed word                    |
| doc_count | INTEGER | How many docs contain this (for IDF)   |

---

### inverted_index

The search index - integers only for speed.

```sql
CREATE TABLE inverted_index (
    vocab_id    INTEGER NOT NULL,
    doc_id      INTEGER NOT NULL,
    frequency   INTEGER NOT NULL,
    PRIMARY KEY (vocab_id, doc_id),
    FOREIGN KEY(vocab_id) REFERENCES vocab(id),
    FOREIGN KEY(doc_id) REFERENCES documents(id)
);

CREATE INDEX idx_inverted_vocab ON inverted_index(vocab_id);
CREATE INDEX idx_inverted_doc ON inverted_index(doc_id);
```

| Column    | Type    | Notes                          |
| --------- | ------- | ------------------------------ |
| vocab_id  | INTEGER | FK → vocab                     |
| doc_id    | INTEGER | FK → documents                 |
| frequency | INTEGER | Term frequency (TF for BM25)   |

---

## Vector Index

### jassas.usearch

USearch index file for semantic search.

- **Format:** Binary file (`.usearch`)
- **Key:** `doc_id` from documents table
- **Value:** 384-dimensional vector (MiniLM embeddings)
- **Location:** `/data/jassas.usearch`

---

## BM25 Requirements

All components needed for BM25 scoring:

| Component | Source | Formula Use |
| --------- | ------ | ----------- |
| TF (term frequency) | inverted_index.frequency | How often word appears in doc |
| IDF (inverse doc freq) | vocab.doc_count | How rare the word is |
| doc_len | documents.doc_len | Document length normalization |
| avgdl | AVG(documents.doc_len) | Average doc length |

```
BM25 Score = Σ IDF(qi) * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (doc_len / avgdl)))

Where: k1 = 1.2, b = 0.75
```

---

## Design Decisions

1. **Integer-based index:** Store word IDs, not strings (100x smaller)
2. **Denormalized URL:** Stored in documents for fast result display
3. **doc_len stored:** Pre-calculated for BM25 performance
4. **Composite PK:** (vocab_id, doc_id) for fast lookups
5. **Content hash:** Detect duplicate pages
6. **Status columns:** Enable stop/resume for all services

---

## Migration Path

When scaling to PostgreSQL:

1. Replace AUTOINCREMENT with SERIAL
2. Replace BLOB with BYTEA
3. Add connection pooling
4. Consider partitioning for inverted_index
