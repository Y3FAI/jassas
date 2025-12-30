"""
Database initialization script for Jassas Search Engine.
Creates all tables and indexes as per approved schema.
"""
import sqlite3
import os

# Define paths - all modules use these
DB_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
DB_PATH = os.path.join(DB_FOLDER, 'jassas.db')


def init_db():
    """Initialize the database with all tables and indexes."""

    # 1. Ensure data folder exists
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)
        print(f"Created data folder at: {DB_FOLDER}")

    # 2. Connect (creates file if missing)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("Building database schema...")

    # ========== CRAWLER LAYER ==========

    # frontier - URL queue for BFS crawling
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS frontier (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        url             TEXT UNIQUE NOT NULL,
        status          TEXT DEFAULT 'pending',
        depth           INTEGER DEFAULT 0,
        error_message   TEXT,
        discovered_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_frontier_status ON frontier(status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_frontier_depth ON frontier(depth)')

    # raw_pages - HTML archive
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS raw_pages (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        url             TEXT UNIQUE NOT NULL,
        html_content    BLOB,
        content_hash    TEXT,
        http_status     INTEGER,
        crawled_at      DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_pages_hash ON raw_pages(content_hash)')

    # ========== CLEANER LAYER ==========

    # documents - cleaned text with metadata
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        raw_page_id     INTEGER NOT NULL,
        url             TEXT NOT NULL,
        title           TEXT,
        clean_text      TEXT,
        doc_len         INTEGER DEFAULT 0,
        status          TEXT DEFAULT 'pending',
        cleaned_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(raw_page_id) REFERENCES raw_pages(id)
    )''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_raw_page ON documents(raw_page_id)')

    # ========== TOKENIZER LAYER ==========

    # vocab - word dictionary (store each word once)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS vocab (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        token           TEXT UNIQUE NOT NULL,
        doc_count       INTEGER DEFAULT 0
    )''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_vocab_token ON vocab(token)')

    # inverted_index - integer-based search index
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS inverted_index (
        vocab_id    INTEGER NOT NULL,
        doc_id      INTEGER NOT NULL,
        frequency   INTEGER NOT NULL,
        PRIMARY KEY (vocab_id, doc_id),
        FOREIGN KEY(vocab_id) REFERENCES vocab(id),
        FOREIGN KEY(doc_id) REFERENCES documents(id)
    )''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_inverted_vocab ON inverted_index(vocab_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_inverted_doc ON inverted_index(doc_id)')

    conn.commit()
    conn.close()

    print(f"Database initialized at: {DB_PATH}")


def get_stats():
    """Get database statistics."""
    if not os.path.exists(DB_PATH):
        print("Database not found. Run init_db() first.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    tables = ['frontier', 'raw_pages', 'documents', 'vocab', 'inverted_index']

    print("\nDatabase Statistics:")
    print("-" * 40)

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{table:20} {count:>10} rows")

    conn.close()


if __name__ == "__main__":
    init_db()
    get_stats()
