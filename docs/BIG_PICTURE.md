# Jassas Search Engine - Big Picture

> High performance search engine.

## System Architecture

```
                        ┌─────────────┐
                        │   MANAGER   │
                        │    (CLI)    │
                        └──────┬──────┘
                               │ controls
       ┌───────────┬───────────┼───────────┬───────────┐
       ▼           ▼           ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────┐
│ CRAWLER  │ │ CLEANER  │ │ TOKENIZER │ │  RANKER  │ │   API    │
└────┬─────┘ └────┬─────┘ └─────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │             │            │            │
     └────────────┴──────┬──────┴────────────┴────────────┘
                         ▼
                  ┌─────────────┐
                  │  DATABASE   │
                  │   SQLite    │
                  │      +      │
                  │   .usearch  │
                  └─────────────┘
```

## Data Flow

```
mygov.sa ──▶ CRAWLER ──▶ CLEANER ──▶ TOKENIZER ──▶ DATABASE
                                                       │
USER QUERY ──▶ RANKER ◀────────────────────────────────┘
                  │
                  ▼
             API RESPONSE
```

## Pipeline Summary

| Service       | Role                            |
| ------------- | ------------------------------- |
| **Manager**   | CLI to control all services     |
| **Crawler**   | Fetch HTML from mygov.sa        |
| **Cleaner**   | Strip HTML, normalize text      |
| **Tokenizer** | BM25 tokens + Vector embeddings |
| **Ranker**    | Query & score results (RRF)     |
| **API**       | Serve results to users          |

## Core Principles

-   Each service runs independently
-   Shared DB is the only coupling
-   Start with SQLite, design for migration
-   Stop/Resume capability for all services

---

## Manager

Central control point for all services.

```
┌─────────────────────────────────────────────────────┐
│                     MANAGER                         │
├─────────────────────────────────────────────────────┤
│                                                     │
│   Commands:                                         │
│   ├── start <service>                              │
│   ├── stop <service>                               │
│   ├── status                                       │
│   ├── stats                                        │
│   └── logs <service>                               │
│                                                     │
│   Future: Web dashboard                            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Responsibilities:

-   Start/stop individual services
-   Monitor service health
-   View statistics (pages crawled, indexed, etc.)
-   CLI first, web later

---

## Crawler

Fetches pages from mygov.sa using Breadth-First Search.

```
┌─────────────────────────────────────────────────────┐
│                     CRAWLER                         │
├─────────────────────────────────────────────────────┤
│                                                     │
│   1. GET next PENDING url from DB                  │
│              │                                      │
│              ▼                                      │
│   2. FETCH html                                    │
│              │                                      │
│              ▼                                      │
│   3. STORE html in pages table                     │
│              │                                      │
│              ▼                                      │
│   4. EXTRACT new urls                              │
│              │                                      │
│              ▼                                      │
│   5. INSERT new urls as PENDING (if not exist)     │
│              │                                      │
│              ▼                                      │
│   6. MARK current url as COMPLETED                 │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### DB Schema:

```sql
urls table (The Queue):
├── id
├── url (unique)
├── status (PENDING / IN_PROGRESS / COMPLETED / FAILED)
├── depth (for BFS priority)
└── discovered_at

pages table (The Archive):
├── id
├── url_id (FK)
├── raw_html
└── crawled_at
```

### Rules (Politeness):

-   Respect robots.txt
-   Rate limit requests
-   Track visited URLs (no duplicates)
-   Handle errors gracefully

### Key Features:

-   Queue lives in DB (stop/resume works)
-   BFS via depth column (process shallow first)
-   IN_PROGRESS status (handles crashes mid-fetch)

---

## Cleaner

Processes raw HTML into clean, normalized text.

```
┌─────────────────────────────────────────────────────┐
│                     CLEANER                         │
├─────────────────────────────────────────────────────┤
│                                                     │
│   1. GET next uncleaned page from DB               │
│              │                                      │
│              ▼                                      │
│   2. PARSE html → strip tags, scripts, styles      │
│              │                                      │
│              ▼                                      │
│   3. NORMALIZE → clean whitespace, fix encoding    │
│              │                                      │
│              ▼                                      │
│   4. STORE clean content                           │
│              │                                      │
│              ▼                                      │
│   5. MARK as CLEANED                               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### DB Schema:

```sql
documents table:
├── id
├── page_id (FK)
├── title
├── clean_text
├── cleaned_at
└── status
```

### Cleaner Does:

-   Strip HTML noise
-   Extract readable text
-   Normalize encoding/whitespace

### Cleaner Does NOT:

-   Tokenization
-   Index building
-   Ranking logic

---

## Tokenizer

Dual-brain processing: BM25 (lexical) + Vector (semantic).

```
┌─────────────────────────────────────────────────────┐
│                    TOKENIZER                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│   GET clean document from DB                       │
│              │                                      │
│              ├─────────────┬─────────────┐         │
│              ▼             ▼             │         │
│         PATH A         PATH B            │         │
│         (BM25)        (Vector)           │         │
│           │              │               │         │
│       Tokenize       Embed text          │         │
│       Stem           (MiniLM)            │         │
│       Stop words         │               │         │
│           │              │               │         │
│           ▼              ▼               │         │
│       SQLite         .usearch            │         │
│    (inverted idx)    (vector idx)        │         │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Path A: BM25 Track (Lexical)

-   **Action:** Tokenize, Stem, Remove Stopwords
-   **Output:** `["renew", "passport", "riyadh"]`
-   **Destination:** SQLite (Inverted Index Table)
-   **Purpose:** Exact keyword matches

### Path B: Vector Track (Semantic)

-   **Action:** Pass text to Embedding Model (all-MiniLM-L6-v2)
-   **Output:** `[0.12, -0.45, 0.88, ...]` (384-dimensional vector)
-   **Destination:** USearch Index (.usearch file)
-   **Purpose:** Concept matches (semantic similarity)

### DB Schema:

```sql
tokens table (Inverted Index):
├── word
├── document_id (FK)
└── frequency

Vector Index:
└── .usearch file (USearch library)
```

---

## Ranker

Queries both indexes and merges results using RRF.

```
┌─────────────────────────────────────────────────────┐
│                      RANKER                         │
├─────────────────────────────────────────────────────┤
│                                                     │
│   USER QUERY: "How to renew passport"              │
│              │                                      │
│              ├─────────────┬─────────────┐         │
│              ▼             ▼             │         │
│         PATH A         PATH B            │         │
│         (BM25)        (Vector)           │         │
│           │              │               │         │
│       Search         Embed query         │         │
│       SQLite         Search .usearch     │         │
│           │              │               │         │
│           ▼              ▼               │         │
│       Top 10          Top 10             │         │
│       results         results            │         │
│           │              │               │         │
│           └──────┬───────┘               │         │
│                  ▼                                  │
│              MERGE (RRF)                           │
│                  │                                  │
│                  ▼                                  │
│           FINAL TOP K RESULTS                      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Merge Strategy: RRF (Reciprocal Rank Fusion)

```
score = Σ 1/(k + rank)
```

Where:

-   `k = 60` (constant)
-   `rank` = position in each result list

### Why RRF:

-   Simple formula
-   No tuning required
-   Industry standard for hybrid search
-   Combines best of both worlds

---

## API

Serves search results to users.

```
┌─────────────────────────────────────────────────────┐
│                       API                           │
├─────────────────────────────────────────────────────┤
│                                                     │
│   Endpoints:                                        │
│   ├── GET /search?q=<query>                        │
│   ├── GET /document/<id>                           │
│   └── GET /health                                  │
│                                                     │
│   Future: Web UI                                   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Responsibilities:

-   Accept search queries
-   Return ranked results
-   Serve document details
-   Health check endpoint

---

## Database

Shared storage for all services.

```
┌─────────────────────────────────────────────────────┐
│                    DATABASE                         │
├─────────────────────────────────────────────────────┤
│                                                     │
│   SQLite (jassas.db):                              │
│   ├── urls          (crawler queue)                │
│   ├── pages         (raw HTML archive)             │
│   ├── documents     (clean text)                   │
│   └── tokens        (inverted index)               │
│                                                     │
│   USearch (jassas.usearch):                        │
│   └── Vector index for semantic search             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Design Principles:

-   Start with SQLite (zero cost, portable)
-   Design for PostgreSQL migration
-   USearch for vector storage (lightweight, fast)

---

## Target

**Website:** mygov.sa (Saudi Government Services Portal)

**Value:** Official government documentation for services, costs, requirements.

**Example Use Cases:**

-   "How to renew passport"
-   "Cost of business license"
-   "Requirements for work visa"

---

## Philosophy

1. **Start Small** - MVP first, features later
2. **Stay Cheap** - SQLite + local files
3. **Scale Later** - Design allows growth
4. **Independent Services** - Each component runs alone
5. **Stop/Resume** - All services can pause and continue
