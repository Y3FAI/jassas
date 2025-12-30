# Jassas Folder Structure

> Flat services architecture for MVP

## Overview

```
jassas/
├── docs/                   # Documentation
│   ├── BIG_PICTURE.md      # System architecture
│   ├── SCHEMA.md           # Database schema
│   └── STRUCTURE.md        # This file
├── data/                   # Runtime data (gitignored)
│   ├── jassas.db           # SQLite database
│   └── jassas.usearch      # Vector index
├── src/                    # Source code
│   ├── db/                 # Shared database layer
│   ├── crawler/            # URL fetching service
│   ├── cleaner/            # HTML parsing service
│   ├── tokenizer/          # BM25 + Vector indexing
│   ├── ranker/             # Search & scoring
│   ├── api/                # HTTP endpoints
│   └── manager/            # CLI control
├── tests/                  # Test files
├── requirements.txt        # Python dependencies
└── README.md               # Project readme
```

---

## Module Details

### db/ (Shared)

Database connection and models shared by all services.

```
db/
├── __init__.py
├── connection.py       # SQLite connection manager
└── models.py           # Table operations (CRUD)
```

---

### crawler/

Fetches pages from mygov.sa using BFS.

```
crawler/
├── __init__.py
├── fetcher.py          # HTTP requests, rate limiting
└── extractor.py        # URL extraction from HTML
```

---

### cleaner/

Strips HTML and normalizes text.

```
cleaner/
├── __init__.py
└── parser.py           # HTML to clean text
```

---

### tokenizer/

Dual-path indexing: BM25 + Vector.

```
tokenizer/
├── __init__.py
├── bm25.py             # Tokenize, stem, build inverted index
└── vector.py           # Embed text, store in USearch
```

---

### ranker/

Queries indexes and merges results.

```
ranker/
├── __init__.py
└── rrf.py              # RRF merge algorithm
```

---

### api/

HTTP server for search queries.

```
api/
├── __init__.py
└── server.py           # FastAPI or Flask endpoints
```

---

### manager/

CLI to control all services.

```
manager/
├── __init__.py
└── cli.py              # Start, stop, status, stats
```

---

## Import Pattern

```python
# Any service can import db
from src.db.connection import get_db
from src.db.models import Frontier, Documents

# Manager can import any service
from src.crawler import start_crawler
from src.cleaner import start_cleaner
```

---

## Data Directory

The `data/` folder is gitignored and contains runtime files:

| File | Description |
|------|-------------|
| jassas.db | SQLite database |
| jassas.usearch | USearch vector index |

---

## Design Decisions

1. **Flat structure:** Simple imports, no package complexity
2. **Shared db/:** Single connection manager for all services
3. **One file per concern:** Easy to find and modify
4. **tests/ at root:** Mirrors src/ structure
5. **data/ gitignored:** No runtime files in repo
