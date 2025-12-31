"""
Jassas API Benchmarking Suite
Measures end-to-end API response times.
"""
import time
import statistics
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

API_URL = "http://localhost:8000"


def single_request(query: str) -> dict:
    """Make a single search request and return timing info."""
    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{API_URL}/api/v1/search",
            json={"query": query, "limit": 10},
            timeout=30
        )
        elapsed = (time.perf_counter() - start) * 1000
        data = resp.json()
        return {
            "success": resp.status_code == 200,
            "latency_ms": elapsed,
            "server_time_ms": data.get("execution_time_ms", 0),
            "count": data.get("count", 0)
        }
    except Exception as e:
        return {
            "success": False,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "server_time_ms": 0,
            "count": 0,
            "error": str(e)
        }


def run_api_benchmarks():
    console.print("\n[bold cyan]Jassas API Benchmarking Suite[/bold cyan]\n")

    # Check if API is running
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code != 200:
            raise Exception("API not healthy")
        console.print(f"[green]OK API is running at {API_URL}[/green]\n")
    except Exception as e:
        console.print(f"[red]FAIL API not reachable at {API_URL}[/red]")
        console.print("[dim]Start the API with: uvicorn src.api.main:app --port 8000[/dim]")
        return

    queries = [
        "تجديد جواز السفر",
        "رخصة القيادة",
        "المخالفات المرورية",
        "الأمن السيبراني",
        "وزارة الصحة",
        "الهوية الوطنية",
        "العنوان الوطني",
        "government services",
    ]

    # --- Test 1: Sequential Latency ---
    console.print("[bold yellow]1. Sequential Request Latency[/bold yellow]")

    table = Table(box=box.ROUNDED)
    table.add_column("Query", style="cyan", max_width=25)
    table.add_column("Total (ms)", justify="right")
    table.add_column("Server (ms)", justify="right")
    table.add_column("Network (ms)", justify="right", style="dim")
    table.add_column("Results", justify="right")

    latencies = []
    server_times = []

    for q in queries:
        result = single_request(q)
        if result["success"]:
            network = result["latency_ms"] - result["server_time_ms"]
            latencies.append(result["latency_ms"])
            server_times.append(result["server_time_ms"])
            table.add_row(
                q[:25],
                f"{result['latency_ms']:.1f}",
                f"{result['server_time_ms']:.1f}",
                f"{network:.1f}",
                str(result["count"])
            )
        else:
            table.add_row(q[:25], "ERROR", "-", "-", "-")

    console.print(table)

    if latencies:
        console.print(f"\n   Avg Total Latency: [bold]{statistics.mean(latencies):.1f}ms[/bold]")
        console.print(f"   Avg Server Time:   [bold]{statistics.mean(server_times):.1f}ms[/bold]")
        console.print(f"   Avg Network/HTTP:  [bold]{statistics.mean(latencies) - statistics.mean(server_times):.1f}ms[/bold]")

    # --- Test 2: Concurrent Load ---
    console.print(f"\n[bold yellow]2. Concurrent Load Test[/bold yellow]")

    concurrent_levels = [1, 5, 10, 20]
    iterations_per_level = 50

    load_table = Table(box=box.ROUNDED)
    load_table.add_column("Concurrency", justify="center")
    load_table.add_column("Total Reqs", justify="right")
    load_table.add_column("Duration (s)", justify="right")
    load_table.add_column("RPS", justify="right", style="green")
    load_table.add_column("Avg Latency", justify="right")
    load_table.add_column("P95 Latency", justify="right")

    for concurrency in concurrent_levels:
        results = []
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for i in range(iterations_per_level):
                q = queries[i % len(queries)]
                futures.append(executor.submit(single_request, q))

            for future in as_completed(futures):
                results.append(future.result())

        duration = time.perf_counter() - start_time
        successful = [r for r in results if r["success"]]
        latency_list = [r["latency_ms"] for r in successful]

        if latency_list:
            rps = len(successful) / duration
            avg_lat = statistics.mean(latency_list)
            p95_lat = sorted(latency_list)[int(len(latency_list) * 0.95)]

            load_table.add_row(
                str(concurrency),
                str(len(successful)),
                f"{duration:.2f}",
                f"{rps:.1f}",
                f"{avg_lat:.1f}ms",
                f"{p95_lat:.1f}ms"
            )

    console.print(load_table)

    # --- Test 3: Cold vs Warm ---
    console.print(f"\n[bold yellow]3. Cold vs Warm Comparison[/bold yellow]")
    console.print("   [dim](First request after API start vs subsequent)[/dim]")

    # We can't easily test cold start without restarting API
    # So we just show warm performance
    warm_results = [single_request(q) for q in queries[:3]]
    warm_latencies = [r["latency_ms"] for r in warm_results if r["success"]]

    if warm_latencies:
        console.print(f"   Warm Avg: [bold]{statistics.mean(warm_latencies):.1f}ms[/bold]")
        console.print(f"   [dim]Note: Cold start adds ~6s for model loading[/dim]")


if __name__ == "__main__":
    run_api_benchmarks()
