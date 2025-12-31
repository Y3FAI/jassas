"""
Jassas Relevance Benchmark - LLM-as-a-Judge
Uses OpenRouter free model to evaluate search quality.
"""
import os
import sys
import json
import math
import requests
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich import box
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ranker.engine import Ranker

console = Console()

# Load environment variables
load_dotenv()

# OpenRouter config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Dual judges for more robust evaluation
JUDGES = [
    "mistralai/devstral-2512:free",
    "xiaomi/mimo-v2-flash:free",
]

# Test queries - exact service names from DB
QUERIES = [
    "اصدار رخصه بناء",
    "دفع ضريبه قيمه مضافه",
    "حجز مواعيد طبيه",
    "تجديد اقامه",
    "اصدار تاشيرات عمل",
    "حجز اسم تجاري",
    "تسجيل تصرف عقاري",
    "اصدار هويه وطنيه",
    "اصدار تاشيره خروج والعوده",
    "اصدار رخصه سير",
    "تجديد رخص عمل",
    "اصدار شهاده اشتراك",
    "تسجيل في جامعات",
    "اصدار رخصه حرفيه",
    "طلب ابتعاث خارجي",
]


def format_results_for_llm(results: List[dict]) -> str:
    """Format search results for LLM evaluation."""
    formatted = []
    for i, r in enumerate(results[:10], 1):
        title = r.get('title', 'No Title')[:100]
        snippet = r.get('clean_text', '')[:200]
        formatted.append(f"{i}. Title: {title}\n   Snippet: {snippet}")
    return "\n\n".join(formatted)


def call_single_judge(model: str, prompt: str, num_results: int) -> List[int]:
    """Call a single LLM judge and return scores."""
    try:
        resp = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 100,
            },
            timeout=60
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        # Parse JSON array from response
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        # Find JSON array in response
        import re
        match = re.search(r'\[[\d,\s]+\]', content)
        if match:
            scores = json.loads(match.group())
            return scores[:num_results]

        scores = json.loads(content)
        return scores[:num_results]
    except Exception as e:
        return None  # Return None to indicate failure


def call_llm_judge(query: str, results: List[dict]) -> dict:
    """
    Send results to multiple LLM judges and return detailed scores.
    Returns dict with averaged scores, individual judge scores, and doc titles.
    """
    if not OPENROUTER_API_KEY:
        console.print("[red]Error: OPENROUTER_API_KEY not set[/red]")
        return None

    num_results = min(len(results), 10)

    prompt = f"""You are a Search Relevance Evaluator for Saudi government services.

Query: "{query}"

Rate each document's relevance to the query on a scale of 0-3:
- 0: Irrelevant (wrong topic entirely)
- 1: Tangentially related (mentions topic but not useful)
- 2: Relevant (answers the query partially)
- 3: Perfect (exact match, official service page)

Documents:
{format_results_for_llm(results)}

Return ONLY a JSON array of {num_results} integers (scores), nothing else.
Example: [3, 2, 1, 0, 2, 1, 0, 0, 1, 2]"""

    all_scores = []
    judge_scores_by_model = {}

    for model in JUDGES:
        scores = call_single_judge(model, prompt, num_results)
        if scores and len(scores) == num_results:
            all_scores.append(scores)
            model_name = model.split('/')[1].split(':')[0]
            judge_scores_by_model[model_name] = scores

    if not all_scores:
        console.print(f"[red]All judges failed[/red]")
        return None

    # Average scores from all judges
    averaged = []
    for i in range(num_results):
        avg = sum(s[i] for s in all_scores) / len(all_scores)
        averaged.append(round(avg))

    return {
        "averaged_scores": averaged,
        "judge_scores": judge_scores_by_model,
        "num_judges": len(all_scores),
        "doc_titles": [r.get('title', 'No Title')[:80] for r in results[:num_results]]
    }


def calculate_mrr(scores: List[int]) -> float:
    """
    Mean Reciprocal Rank - position of first relevant result.
    Relevant = score >= 2
    """
    for i, score in enumerate(scores):
        if score >= 2:
            return 1.0 / (i + 1)
    return 0.0


def calculate_ndcg(scores: List[int], k: int = 10) -> float:
    """
    Normalized Discounted Cumulative Gain.
    Measures ranking quality - are best results at the top?
    """
    scores = scores[:k]
    if not scores or max(scores) == 0:
        return 0.0

    # DCG
    dcg = scores[0]
    for i in range(1, len(scores)):
        dcg += scores[i] / math.log2(i + 2)

    # Ideal DCG (perfect ranking)
    ideal = sorted(scores, reverse=True)
    idcg = ideal[0]
    for i in range(1, len(ideal)):
        idcg += ideal[i] / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def calculate_precision_at_k(scores: List[int], k: int = 10) -> float:
    """Precision@K - fraction of top-k results that are relevant."""
    scores = scores[:k]
    if not scores:
        return 0.0
    relevant = sum(1 for s in scores if s >= 2)
    return relevant / len(scores)


def run_relevance_benchmark():
    console.print("\n[bold cyan]Jassas Relevance Benchmark (LLM-as-a-Judge)[/bold cyan]\n")

    if not OPENROUTER_API_KEY:
        console.print("[red]FAIL Set OPENROUTER_API_KEY environment variable[/red]")
        console.print("[dim]export OPENROUTER_API_KEY='your-key-here'[/dim]")
        return

    console.print(f"[dim]Judges: {', '.join(j.split('/')[1].split(':')[0] for j in JUDGES)}[/dim]")
    console.print(f"[dim]Queries: {len(QUERIES)}[/dim]\n")

    # Load ranker
    console.print("[yellow]Loading search engine...[/yellow]")
    ranker = Ranker(verbose=False)
    ranker._load_vector_engine()
    console.print("[green]OK Engine ready[/green]\n")

    # Results table
    table = Table(title="Relevance Scores by Query", box=box.ROUNDED)
    table.add_column("Query", style="cyan", max_width=30)
    table.add_column("MRR", justify="right")
    table.add_column("NDCG@10", justify="right")
    table.add_column("P@10", justify="right")
    table.add_column("Avg Score", justify="right")
    table.add_column("Scores", style="dim", max_width=30)

    all_mrr = []
    all_ndcg = []
    all_precision = []
    all_avg_scores = []

    for query in QUERIES:
        console.print(f"[dim]Evaluating: {query[:40]}...[/dim]")

        # Search
        results = ranker.search(query, k=10)

        if not results:
            table.add_row(query[:30], "-", "-", "-", "-", "No results")
            continue

        # Get LLM scores
        judgment = call_llm_judge(query, results)
        if not judgment:
            table.add_row(query[:30], "-", "-", "-", "-", "Judge failed")
            continue

        scores = judgment["averaged_scores"]

        # Calculate metrics
        mrr = calculate_mrr(scores)
        ndcg = calculate_ndcg(scores)
        precision = calculate_precision_at_k(scores)
        avg_score = sum(scores) / len(scores) if scores else 0

        all_mrr.append(mrr)
        all_ndcg.append(ndcg)
        all_precision.append(precision)
        all_avg_scores.append(avg_score)

        # Display detailed results
        console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")
        console.print(f"[dim]Judges: {judgment['num_judges']} ({', '.join(judgment['judge_scores'].keys())})[/dim]")

        # Document details table
        doc_table = Table(title="Scored Documents", box=box.ROUNDED)
        doc_table.add_column("#", style="dim", width=3)
        doc_table.add_column("Title", style="cyan", max_width=60)
        doc_table.add_column("Avg", justify="center", style="bold")
        for judge_name in judgment['judge_scores'].keys():
            doc_table.add_column(judge_name[:8], justify="center")

        for i, (title, avg_score_val) in enumerate(zip(judgment['doc_titles'], scores)):
            row = [str(i+1), title, str(avg_score_val)]
            for judge_name, judge_scores in judgment['judge_scores'].items():
                row.append(str(judge_scores[i]))
            doc_table.add_row(*row)

        console.print(doc_table)

        # Summary for this query
        console.print(f"[yellow]Metrics:[/yellow] MRR={mrr:.2f} | NDCG@10={ndcg:.2f} | P@10={precision:.0%} | Avg={avg_score:.1f}/3\n")

        # Also add to summary table
        score_str = str(scores)
        table.add_row(
            query[:30],
            f"{mrr:.2f}",
            f"{ndcg:.2f}",
            f"{precision:.0%}",
            f"{avg_score:.1f}/3",
            score_str[:30]
        )

    console.print(table)

    # Summary
    if all_mrr:
        console.print("\n[bold yellow]Summary Metrics:[/bold yellow]")

        summary = Table(box=box.SIMPLE)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Score", justify="right", style="bold green")
        summary.add_column("Interpretation", style="dim")

        avg_mrr = sum(all_mrr) / len(all_mrr)
        avg_ndcg = sum(all_ndcg) / len(all_ndcg)
        avg_precision = sum(all_precision) / len(all_precision)
        avg_score = sum(all_avg_scores) / len(all_avg_scores)

        summary.add_row("Mean MRR", f"{avg_mrr:.3f}", "1.0 = first result always relevant")
        summary.add_row("Mean NDCG@10", f"{avg_ndcg:.3f}", "1.0 = perfect ranking")
        summary.add_row("Mean P@10", f"{avg_precision:.1%}", "% of top 10 that are relevant")
        summary.add_row("Avg Relevance", f"{avg_score:.2f}/3", "0=bad, 3=perfect")

        console.print(summary)

        # Overall grade
        overall = (avg_mrr + avg_ndcg + avg_precision) / 3 * 100

        if overall >= 80:
            grade = "[bold green]A - Excellent[/bold green]"
        elif overall >= 60:
            grade = "[bold yellow]B - Good[/bold yellow]"
        elif overall >= 40:
            grade = "[bold orange]C - Needs Work[/bold orange]"
        else:
            grade = "[bold red]D - Poor[/bold red]"

        console.print(f"\n[bold]Overall Quality Score: {overall:.0f}% - {grade}[/bold]")


if __name__ == "__main__":
    run_relevance_benchmark()
