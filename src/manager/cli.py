"""
Jassas Manager CLI - Control center for all services.
"""
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from db import init_db, db_exists, Frontier, RawPages, Documents, Vocab, InvertedIndex
from db.connection import get_db

app = typer.Typer(
    name="jassas",
    help="Jassas Search Engine - Manager CLI",
    add_completion=False
)
console = Console()


@app.command()
def init():
    """Initialize the database."""
    if db_exists():
        console.print("[yellow]Database already exists.[/yellow]")
        if not typer.confirm("Reinitialize?"):
            raise typer.Abort()

    init_db()
    console.print("[green]Database initialized successfully.[/green]")


@app.command()
def seed(url: str = typer.Argument(..., help="Starting URL to crawl")):
    """Add a seed URL to the frontier with high priority."""
    if not db_exists():
        console.print("[red]Database not found. Run 'jassas init' first.[/red]")
        raise typer.Exit(1)

    # Arabic URLs get highest priority
    priority = 100 if '/ar' in url else 50
    added = Frontier.add_url(url, depth=0, priority=priority)
    if added:
        console.print(f"[green]Added seed URL (priority={priority}):[/green] {url}")
    else:
        console.print(f"[yellow]URL already exists:[/yellow] {url}")


@app.command()
def sitemap(url: str = typer.Argument(..., help="Sitemap URL to parse")):
    """Parse sitemap.xml and add URLs to frontier."""
    if not db_exists():
        console.print("[red]Database not found. Run 'jassas init' first.[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Parsing Sitemap[/bold cyan]\n")

    from crawler.sitemap import SitemapParser
    from crawler.fetcher import Fetcher

    fetcher = Fetcher()
    parser = SitemapParser(fetcher, verbose=True)

    try:
        urls = parser.parse(url)

        if not urls:
            console.print("[yellow]No URLs found in sitemap.[/yellow]")
            return

        # Add to frontier with depth=1 (sitemap discovered)
        urls_with_depth = [(u, 1, p) for u, p in urls]
        added = Frontier.add_urls(urls_with_depth)

        console.print(f"\n[green]Added {added} URLs to frontier[/green]")

        # Show priority breakdown
        high = sum(1 for _, p in urls if p >= 50)
        med = sum(1 for _, p in urls if 0 < p < 50)
        low = sum(1 for _, p in urls if p <= 0)
        console.print(f"  High priority (≥50): {high}")
        console.print(f"  Medium priority (1-49): {med}")
        console.print(f"  Low priority (≤0): {low}")

    finally:
        fetcher.close()


@app.command()
def stats():
    """Show database statistics."""
    if not db_exists():
        console.print("[red]Database not found. Run 'jassas init' first.[/red]")
        raise typer.Exit(1)

    # Frontier stats
    frontier_stats = Frontier.get_stats()
    total_urls = sum(frontier_stats.values())

    # Document stats
    doc_count = Documents.get_total_count()
    avg_doc_len = Documents.get_avg_doc_len()

    # Vocab stats
    with get_db() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM vocab")
        vocab_count = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM inverted_index")
        index_count = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM raw_pages")
        pages_count = cursor.fetchone()[0]

    # Build table
    table = Table(title="Jassas Statistics", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    # Frontier section
    table.add_row("─── Frontier ───", "")
    table.add_row("Total URLs", str(total_urls))
    for status, count in frontier_stats.items():
        table.add_row(f"  {status}", str(count))

    # Crawler section
    table.add_row("─── Crawler ───", "")
    table.add_row("Raw Pages", str(pages_count))

    # Cleaner section
    tokenized_count = Documents.get_tokenized_count()
    table.add_row("─── Cleaner ───", "")
    table.add_row("Documents (total)", str(doc_count))
    table.add_row("Tokenized (searchable)", str(tokenized_count))
    table.add_row("Avg Doc Length", f"{avg_doc_len:.1f}")

    # Tokenizer section
    table.add_row("─── Tokenizer ───", "")
    table.add_row("Vocabulary Size", str(vocab_count))
    table.add_row("Index Entries", str(index_count))

    console.print(table)


@app.command()
def frontier(limit: int = typer.Option(10, help="Number of URLs to show")):
    """Show pending URLs in the frontier."""
    if not db_exists():
        console.print("[red]Database not found. Run 'jassas init' first.[/red]")
        raise typer.Exit(1)

    pending = Frontier.get_next_pending(limit=limit)

    if not pending:
        console.print("[yellow]No pending URLs in frontier.[/yellow]")
        return

    table = Table(title=f"Frontier (Top {limit} Pending)", box=box.ROUNDED)
    table.add_column("ID", style="dim")
    table.add_column("Depth", justify="center")
    table.add_column("URL", style="cyan")

    for row in pending:
        table.add_row(str(row['id']), str(row['depth']), row['url'])

    console.print(table)


@app.command()
def crawl(
    max_pages: int = typer.Option(100, "--max-pages", "-n", help="Maximum pages to crawl"),
    max_depth: int = typer.Option(5, "--max-depth", "-d", help="Maximum BFS depth"),
    delay: float = typer.Option(2.0, "--delay", "-t", help="Delay between requests (seconds)"),
):
    """Run the crawler."""
    if not db_exists():
        console.print("[red]Database not found. Run 'jassas init' first.[/red]")
        raise typer.Exit(1)

    # Check if frontier has URLs
    pending = Frontier.get_next_pending(limit=1)
    if not pending:
        console.print("[yellow]No URLs in frontier. Run 'jassas seed <url>' first.[/yellow]")
        raise typer.Exit(1)

    from crawler import start
    start(max_pages=max_pages, max_depth=max_depth, delay=delay)


@app.command()
def clean(
    batch_size: int = typer.Option(10, "--batch", "-b", help="Batch size for processing"),
):
    """Run the cleaner to process raw pages."""
    if not db_exists():
        console.print("[red]Database not found. Run 'jassas init' first.[/red]")
        raise typer.Exit(1)

    # Check if there are raw pages
    with get_db() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM raw_pages")
        if cursor.fetchone()[0] == 0:
            console.print("[yellow]No raw pages found. Run 'jassas crawl' first.[/yellow]")
            raise typer.Exit(1)

    from cleaner import start
    start(batch_size=batch_size)


@app.command()
def tokenize(
    batch_size: int = typer.Option(32, "--batch", "-b", help="Batch size for processing"),
):
    """Run the tokenizer to build search indexes."""
    if not db_exists():
        console.print("[red]Database not found. Run 'jassas init' first.[/red]")
        raise typer.Exit(1)

    # Check if there are documents
    doc_count = Documents.get_total_count()
    if doc_count == 0:
        console.print("[yellow]No documents found. Run 'jassas clean' first.[/yellow]")
        raise typer.Exit(1)

    # Check for pending documents
    pending = Documents.get_pending(limit=1)
    if not pending:
        console.print("[yellow]No pending documents to tokenize.[/yellow]")
        return

    from tokenizer import start
    start(batch_size=batch_size)


@app.command()
def build_index():
    """Build NumPy BM25 matrix index from tokenized documents."""
    if not db_exists():
        console.print("[red]Database not found. Run 'jassas init' first.[/red]")
        raise typer.Exit(1)

    # Check if there are tokenized documents
    with get_db() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM documents WHERE status = 'tokenized'")
        tokenized_count = cursor.fetchone()[0]

    if tokenized_count == 0:
        console.print("[yellow]No tokenized documents found. Run 'jassas tokenize' first.[/yellow]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Building NumPy BM25 Index[/bold cyan]\n")

    from scripts.build_index import build_index as build_bm25_index
    try:
        build_bm25_index()
    except Exception as e:
        console.print(f"[red]Error building index: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results"),
):
    """Search using hybrid RRF (NumPy BM25 + Vector Embeddings)."""
    if not db_exists():
        console.print("[red]Database not found. Run 'jassas init' first.[/red]")
        raise typer.Exit(1)

    # Check if we have documents
    doc_count = Documents.get_total_count()
    if doc_count == 0:
        console.print("[yellow]No documents indexed. Run the pipeline first.[/yellow]")
        raise typer.Exit(1)

    # Check if BM25 index exists
    import os
    bm25_index_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'bm25_matrix.pkl')
    if not os.path.exists(bm25_index_path):
        console.print("[yellow]BM25 index not found. Run 'jassas build-index' first.[/yellow]")
        raise typer.Exit(1)

    console.print(f"\n[cyan]Searching:[/cyan] {query}\n")

    # Initialize ranker and search
    from ranker import Ranker
    ranker = Ranker(verbose=True)
    results = ranker.search(query, k=limit)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    # Display results
    table = Table(title=f"Results ({len(results)})", box=box.ROUNDED)
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", style="magenta", width=8)
    table.add_column("Title", style="bold white", max_width=50)
    table.add_column("URL", style="blue")

    for i, res in enumerate(results):
        table.add_row(
            str(i + 1),
            f"{res['score']:.4f}",
            (res['title'][:47] + "...") if len(res['title']) > 50 else res['title'],
            res['url']
        )

    console.print(table)


@app.command()
def benchmark(
    test: str = typer.Argument("relevance", help="Test: relevance, qa, human, latency, or all"),
):
    """Run benchmarks: accuracy (requires OPENROUTER_API_KEY) or latency."""
    if not db_exists():
        console.print("[red]Database not found. Run 'jassas init' first.[/red]")
        raise typer.Exit(1)

    # Check document count
    doc_count = Documents.get_total_count()
    if doc_count == 0:
        console.print("[yellow]No documents indexed. Run the pipeline first.[/yellow]")
        raise typer.Exit(1)

    import subprocess
    import sys

    test_lower = test.lower()

    # Latency benchmark (no API key needed)
    if test_lower == "latency":
        console.print(f"\n[bold cyan]Running latency benchmark...[/bold cyan]\n")
        result = subprocess.run([sys.executable, "tests/benchmark.py"])
        raise typer.Exit(result.returncode)

    # Accuracy benchmarks (require API key)
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENROUTER_API_KEY not found in .env[/red]")
        console.print("[dim]Create .env file from .env.example and add your API key[/dim]")
        raise typer.Exit(1)

    test_files = {
        "relevance": "tests/benchmark_relevance.py",
        "qa": "tests/benchmark_qa.py",
        "human": "tests/benchmark_human.py",
    }

    if test_lower == "all":
        tests = ["relevance", "qa", "human"]
    elif test_lower in test_files:
        tests = [test_lower]
    else:
        console.print(f"[red]Invalid test type: {test}[/red]")
        console.print(f"[dim]Choose: {', '.join(list(test_files.keys()) + ['latency', 'all'])}[/dim]")
        raise typer.Exit(1)

    env = {**os.environ, "OPENROUTER_API_KEY": api_key}

    for test_name in tests:
        test_file = test_files[test_name]
        console.print(f"\n[bold cyan]Running {test_name} benchmark...[/bold cyan]\n")
        result = subprocess.run([sys.executable, test_file], env=env)
        if result.returncode != 0:
            console.print(f"[red]FAIL {test_name} benchmark failed[/red]")
            raise typer.Exit(result.returncode)

    if len(tests) > 1:
        console.print(f"\n[bold green]OK All benchmarks completed[/bold green]")


@app.command()
def reset(force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation")):
    """Reset the database (delete all data)."""
    if not db_exists():
        console.print("[yellow]Database not found.[/yellow]")
        return

    if not force:
        console.print("[red]This will delete ALL data![/red]")
        if not typer.confirm("Are you sure?"):
            raise typer.Abort()

    from db.init_db import DB_PATH
    os.remove(DB_PATH)
    console.print("[green]Database deleted.[/green]")

    init_db()
    console.print("[green]Database reinitialized.[/green]")


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
