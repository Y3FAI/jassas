"""
Jassas QA Benchmark - Task Completion Metric
Measures: Can the user's question be answered using search results?

Unlike P@10/NDCG which assume many relevant docs, this measures
whether the search helps users complete their task.
"""
import os
import sys
import json
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

# Dual judges
JUDGES = [
    "mistralai/devstral-2512:free",
    "xiaomi/mimo-v2-flash:free",
]

# Test questions - specific queries matching actual DB content
QUESTIONS = [
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


def format_results(results: List[dict], k: int) -> str:
    """Format top-k results for judge."""
    formatted = []
    for i, r in enumerate(results[:k], 1):
        title = r.get('title', 'No Title')[:100]
        snippet = r.get('clean_text', '')[:300]
        formatted.append(f"[{i}] {title}\n{snippet}")
    return "\n\n".join(formatted)


def judge_answerable(model: str, question: str, results: List[dict], k: int) -> bool:
    """
    Ask judge: Can this question be answered using top-k results?
    Returns True/False
    """
    if not results:
        return False

    prompt = f"""You are evaluating a search engine for Saudi government services.

User Question: "{question}"

Search Results (Top {k}):
{format_results(results, k)}

Do the search results help the user find what they're looking for?
- YES if: Results contain the relevant service page or direct link to answer
- YES if: User would click on a result and find their answer
- NO if: Results are completely unrelated to the question
- NO if: No result points to the right service/information

This is a SERVICE DIRECTORY - results point to services, not full answers.

Reply with ONLY: YES or NO"""

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
                "max_tokens": 10,
            },
            timeout=60
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip().upper()
        return "YES" in answer
    except Exception as e:
        return None


def evaluate_question(question: str, results: List[dict]) -> Dict:
    """
    Evaluate if question is answerable at different K values.
    Uses multiple judges and takes majority vote.
    Returns detailed results with individual judge votes.
    """
    k_values = [1, 3, 5, 10]
    scores = {}
    judge_votes = {}  # Track individual judge votes

    for k in k_values:
        votes = []
        judge_votes[f"top_{k}"] = {}

        for model in JUDGES:
            result = judge_answerable(model, question, results, k)
            model_name = model.split('/')[1].split(':')[0]
            judge_votes[f"top_{k}"][model_name] = result

            if result is not None:
                votes.append(result)

        # Majority vote (or True if tie with at least one True)
        if votes:
            scores[f"top_{k}"] = sum(votes) >= len(votes) / 2
        else:
            scores[f"top_{k}"] = False

    return {
        "k_scores": scores,
        "judge_votes": judge_votes,
        "doc_titles": [r.get('title', 'No Title')[:80] for r in results[:10]]
    }


def run_qa_benchmark():
    console.print("\n[bold cyan]Jassas QA Benchmark (Task Completion)[/bold cyan]\n")

    if not OPENROUTER_API_KEY:
        console.print("[red]FAIL Set OPENROUTER_API_KEY environment variable[/red]")
        return

    console.print(f"[dim]Judges: {', '.join(j.split('/')[1].split(':')[0] for j in JUDGES)}[/dim]")
    console.print(f"[dim]Questions: {len(QUESTIONS)}[/dim]")
    console.print("[dim]Metric: Can user answer their question with Top-K results?[/dim]\n")

    # Load ranker
    console.print("[yellow]Loading search engine...[/yellow]")
    ranker = Ranker(verbose=False)
    ranker._load_vector_engine()
    console.print("[green]OK Engine ready[/green]\n")

    # Results table
    table = Table(title="Question Answerability", box=box.ROUNDED)
    table.add_column("Question", style="cyan", max_width=35)
    table.add_column("Top 1", justify="center")
    table.add_column("Top 3", justify="center")
    table.add_column("Top 5", justify="center")
    table.add_column("Top 10", justify="center")

    # Aggregate scores
    totals = {"top_1": 0, "top_3": 0, "top_5": 0, "top_10": 0}

    for question in QUESTIONS:
        console.print(f"[dim]Evaluating: {question[:40]}...[/dim]")

        # Search
        results = ranker.search(question, k=10)

        # Evaluate at each K
        evaluation = evaluate_question(question, results)
        k_scores = evaluation["k_scores"]
        judge_votes = evaluation["judge_votes"]
        doc_titles = evaluation["doc_titles"]

        # Update totals
        for key in totals:
            if k_scores.get(key, False):
                totals[key] += 1

        # Display detailed results
        console.print(f"\n[bold cyan]Question:[/bold cyan] {question}")
        console.print(f"[dim]Top results found:[/dim]")
        for i, title in enumerate(doc_titles[:5], 1):
            console.print(f"  {i}. {title}")

        # Judge votes breakdown
        vote_table = Table(title="Judge Votes by K", box=box.ROUNDED)
        vote_table.add_column("K", justify="center", style="dim")
        vote_table.add_column("Decision", justify="center", style="bold")
        for judge_name in JUDGES:
            judge_short = judge_name.split('/')[1].split(':')[0][:8]
            vote_table.add_column(judge_short, justify="center")

        for k in [1, 3, 5, 10]:
            k_key = f"top_{k}"
            decision = "[green]OK[/green]" if k_scores[k_key] else "[red]FAIL[/red]"
            row = [str(k), decision]

            for judge_name in JUDGES:
                judge_short = judge_name.split('/')[1].split(':')[0][:8]
                judge_key = judge_short.split('-')[0] if '-' in judge_short else judge_short
                # Find the judge in the votes
                vote_result = None
                for key, votes_dict in judge_votes.items():
                    if key == k_key:
                        for vname, vresult in votes_dict.items():
                            if judge_short.startswith(vname[:3]) or vname.startswith(judge_short[:3]):
                                vote_result = vresult
                                break
                vote_text = "[green]Y[/green]" if vote_result else "[red]N[/red]" if vote_result is False else "?"
                row.append(vote_text)

            vote_table.add_row(*row)

        console.print(vote_table)
        console.print()

        # Format row for summary table
        def fmt(k):
            return "[green]OK[/green]" if k_scores.get(k, False) else "[red]FAIL[/red]"

        table.add_row(
            question[:35],
            fmt("top_1"),
            fmt("top_3"),
            fmt("top_5"),
            fmt("top_10")
        )

    console.print(table)

    # Summary
    n = len(QUESTIONS)
    console.print("\n[bold yellow]Success Rate (% of questions answerable):[/bold yellow]")

    summary = Table(box=box.SIMPLE)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Score", justify="right", style="bold")
    summary.add_column("Interpretation", style="dim")

    s1 = totals["top_1"] / n * 100
    s3 = totals["top_3"] / n * 100
    s5 = totals["top_5"] / n * 100
    s10 = totals["top_10"] / n * 100

    summary.add_row("Success@1", f"{s1:.0f}%", "Answer in first result")
    summary.add_row("Success@3", f"{s3:.0f}%", "Answer in top 3")
    summary.add_row("Success@5", f"{s5:.0f}%", "Answer in top 5")
    summary.add_row("Success@10", f"{s10:.0f}%", "Answer in top 10")

    console.print(summary)

    # Overall grade
    avg_success = (s1 + s3 + s5 + s10) / 4

    if avg_success >= 70:
        grade = "[bold green]A - Excellent[/bold green]"
    elif avg_success >= 50:
        grade = "[bold yellow]B - Good[/bold yellow]"
    elif avg_success >= 30:
        grade = "[bold orange3]C - Needs Work[/bold orange3]"
    else:
        grade = "[bold red]D - Poor[/bold red]"

    console.print(f"\n[bold]Overall Score: {avg_success:.0f}% - {grade}[/bold]")

    # Insight
    console.print("\n[dim]Insight: Success@1 is the 'I'm feeling lucky' metric.[/dim]")
    console.print("[dim]Success@10 shows if the answer exists in your corpus at all.[/dim]")


if __name__ == "__main__":
    run_qa_benchmark()
