"""
Jassas Benchmarking Suite
Measures search latency breakdown by component.
"""
import time
import sys
import os
import statistics
from rich.console import Console
from rich.table import Table
from rich import box

# Ensure we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ranker.engine import Ranker

console = Console()


class ProfiledRanker(Ranker):
    """
    Subclass of Ranker that instruments the search pipeline
    to measure execution time of each component.
    """
    def search_profiled(self, query: str, k: int = 10):
        metrics = {}

        # 1. Normalization
        t0 = time.perf_counter()
        normalized_query = self.parser._normalize(query)
        metrics['normalization'] = (time.perf_counter() - t0) * 1000

        # 2. BM25 Search (SQL)
        t0 = time.perf_counter()
        bm25_results = self._bm25_search(normalized_query, limit=50)
        metrics['bm25_search'] = (time.perf_counter() - t0) * 1000

        # 3. Vector Search (USearch)
        t0 = time.perf_counter()
        vector_results = self._vector_search(query, limit=50)
        metrics['vector_search'] = (time.perf_counter() - t0) * 1000

        # 4. RRF Merge (CPU)
        t0 = time.perf_counter()
        merged_scores = {}

        # Add BM25 scores
        for rank, doc_id in enumerate(bm25_results):
            if doc_id not in merged_scores:
                merged_scores[doc_id] = 0.0
            merged_scores[doc_id] += 1.0 / (self.RRF_K + rank + 1)

        # Add Vector scores
        for rank, doc_id in enumerate(vector_results):
            if doc_id not in merged_scores:
                merged_scores[doc_id] = 0.0
            merged_scores[doc_id] += 1.0 / (self.RRF_K + rank + 1)

        # Sort
        top_doc_ids = sorted(merged_scores, key=merged_scores.get, reverse=True)[:k]
        metrics['rrf_merge'] = (time.perf_counter() - t0) * 1000

        # 5. Result Fetch (DB)
        t0 = time.perf_counter()
        results = self._fetch_results(top_doc_ids, merged_scores)
        metrics['result_fetch'] = (time.perf_counter() - t0) * 1000

        metrics['total'] = sum(metrics.values())
        return results, metrics


def run_benchmarks():
    console.print("\n[bold cyan]Jassas Benchmarking Suite[/bold cyan]\n")

    # --- Test 1: Cold Start ---
    console.print("[bold yellow]1. Measuring Cold Start...[/bold yellow]")
    t_start = time.perf_counter()
    ranker = ProfiledRanker(verbose=False)
    # Force load of lazy components
    ranker._load_vector_engine()
    t_load = (time.perf_counter() - t_start)
    console.print(f"   Model & Index Load Time: [bold red]{t_load:.2f}s[/bold red]\n")

    # --- Test 2: Latency Breakdown ---
    queries = [
        "تجديد جواز السفر",
        "رخصة القيادة",
        "المخالفات المرورية",
        "الأمن السيبراني",
        "وزارة الصحة",
    ]

    console.print(f"[bold yellow]2. Profiling Search Latency ({len(queries)} queries)...[/bold yellow]")

    table = Table(title="Latency Breakdown (ms)", box=box.ROUNDED)
    table.add_column("Query", style="cyan", max_width=25)
    table.add_column("Norm", justify="right", style="dim")
    table.add_column("BM25", justify="right")
    table.add_column("Vector", justify="right")
    table.add_column("RRF", justify="right", style="dim")
    table.add_column("Fetch", justify="right")
    table.add_column("Total", justify="right", style="bold green")

    all_metrics = []

    for q in queries:
        _, metrics = ranker.search_profiled(q)
        all_metrics.append(metrics)

        table.add_row(
            q[:25],
            f"{metrics['normalization']:.2f}",
            f"{metrics['bm25_search']:.2f}",
            f"{metrics['vector_search']:.2f}",
            f"{metrics['rrf_merge']:.2f}",
            f"{metrics['result_fetch']:.2f}",
            f"{metrics['total']:.2f}"
        )

    console.print(table)

    # --- Component Averages ---
    console.print("\n[bold yellow]Component Averages:[/bold yellow]")
    avg_table = Table(box=box.SIMPLE)
    avg_table.add_column("Component", style="cyan")
    avg_table.add_column("Avg (ms)", justify="right")
    avg_table.add_column("% of Total", justify="right")

    components = ['normalization', 'bm25_search', 'vector_search', 'rrf_merge', 'result_fetch']
    avg_total = statistics.mean([m['total'] for m in all_metrics])

    for comp in components:
        avg = statistics.mean([m[comp] for m in all_metrics])
        pct = (avg / avg_total) * 100
        avg_table.add_row(comp, f"{avg:.2f}", f"{pct:.1f}%")

    console.print(avg_table)

    # --- Test 3: Throughput ---
    iterations = 50
    console.print(f"\n[bold yellow]3. Measuring Throughput ({iterations} iterations)...[/bold yellow]")

    latencies = []
    start_time = time.perf_counter()
    for i in range(iterations):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        ranker.search(q, k=10)
        latencies.append((time.perf_counter() - t0) * 1000)
    end_time = time.perf_counter()

    duration = end_time - start_time
    qps = iterations / duration

    console.print(f"   Total Time: {duration:.2f}s")
    console.print(f"   Throughput: [bold green]{qps:.2f} QPS[/bold green]")
    console.print(f"   Avg Latency: {statistics.mean(latencies):.2f}ms")
    console.print(f"   Min Latency: {min(latencies):.2f}ms")
    console.print(f"   Max Latency: {max(latencies):.2f}ms")
    if len(latencies) >= 20:
        console.print(f"   P95 Latency: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}ms")


if __name__ == "__main__":
    try:
        run_benchmarks()
    except Exception as e:
        console.print(f"[red]Error running benchmarks: {e}[/red]")
        import traceback
        traceback.print_exc()
        console.print("\n[dim]Make sure you have run 'jassas init', 'crawl', 'clean', and 'tokenize' first.[/dim]")
