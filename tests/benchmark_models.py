"""
Jassas Embedding Model Benchmark
Tests FastEmbed models for speed and accuracy on Arabic text.

Usage:
    python tests/benchmark_models.py

Modify MODELS array to test different models.
"""
import sys
import os
import time
import statistics
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rich.console import Console
from rich.table import Table
from rich import box
from fastembed import TextEmbedding

# ============================================================================
# MODELS TO TEST - Modify this array to add/remove models
# ============================================================================
MODELS = [
    # Multilingual E5 family (best for Arabic)
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",

    # BGE multilingual
    "BAAI/bge-small-en-v1.5",  # English baseline for comparison

    # Add more models here:
    # "model-name/model-id",
]

# ============================================================================
# TEST DATA - Arabic queries and documents
# ============================================================================
ARABIC_QUERIES = [
    "كيف أجدد رخصة القيادة",
    "استخراج جواز سفر جديد",
    "تسجيل مركبة جديدة",
    "الاستعلام عن المخالفات المرورية",
    "حجز موعد في الأحوال المدنية",
    "تحديث بيانات الهوية الوطنية",
    "نقل ملكية سيارة",
    "إصدار شهادة ميلاد",
    "تجديد الإقامة للمقيمين",
    "الاستعلام عن تأشيرة خروج وعودة",
]

# Sample documents (simulating indexed content)
ARABIC_DOCUMENTS = [
    "خدمة تجديد رخصة القيادة إلكترونياً عبر منصة أبشر. يمكنك تجديد الرخصة قبل انتهائها بـ 180 يوم.",
    "إصدار جواز السفر السعودي للمواطنين. المتطلبات: صورة شخصية، الهوية الوطنية، دفع الرسوم.",
    "تسجيل المركبات الجديدة في المملكة العربية السعودية. يتطلب فحص فني ودفع رسوم التسجيل.",
    "الاستعلام عن المخالفات المرورية برقم الهوية أو رقم اللوحة عبر منصة أبشر.",
    "حجز موعد في الأحوال المدنية لإصدار أو تجديد الهوية الوطنية.",
    "تحديث البيانات الشخصية في الهوية الوطنية عبر أبشر أو زيارة مكتب الأحوال.",
    "نقل ملكية السيارة من مالك لآخر. يتطلب حضور الطرفين أو وكالة شرعية.",
    "إصدار شهادة ميلاد للمواليد الجدد خلال 30 يوم من تاريخ الولادة.",
    "تجديد الإقامة للعمالة الوافدة. يجب التجديد قبل انتهاء الصلاحية بـ 90 يوم.",
    "خدمة إصدار تأشيرة خروج وعودة للمقيمين عبر منصة أبشر أعمال.",
    "خدمات وزارة الداخلية الإلكترونية للمواطنين والمقيمين في المملكة.",
    "منصة أبشر للخدمات الحكومية الإلكترونية. تسجيل الدخول برقم الهوية.",
    "دفع رسوم الخدمات الحكومية عبر نظام سداد للمدفوعات.",
    "الاستعلام عن صلاحية التأمين على المركبة برقم الهوية.",
    "خدمة تصديق الوثائق الرسمية من وزارة الخارجية السعودية.",
]

# Ground truth: query index -> relevant doc indices
RELEVANCE_MAP = {
    0: [0],      # رخصة القيادة
    1: [1],      # جواز سفر
    2: [2],      # تسجيل مركبة
    3: [3],      # مخالفات
    4: [4],      # أحوال مدنية
    5: [5],      # هوية
    6: [6],      # نقل ملكية
    7: [7],      # شهادة ميلاد
    8: [8],      # إقامة
    9: [9],      # تأشيرة
}

console = Console()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def benchmark_model(model_name: str) -> dict:
    """Benchmark a single model for speed and accuracy."""
    results = {
        "model": model_name,
        "load_time_ms": 0,
        "encode_query_ms": [],
        "encode_doc_ms": 0,
        "mrr": 0,
        "recall_at_1": 0,
        "recall_at_3": 0,
        "recall_at_5": 0,
        "error": None,
    }

    try:
        # 1. Load model
        console.print(f"  Loading model...", style="dim")
        start = time.perf_counter()
        model = TextEmbedding(model_name)
        results["load_time_ms"] = (time.perf_counter() - start) * 1000

        # 2. Encode documents (batch)
        console.print(f"  Encoding {len(ARABIC_DOCUMENTS)} documents...", style="dim")
        start = time.perf_counter()
        doc_embeddings = list(model.embed(ARABIC_DOCUMENTS))
        doc_embeddings = np.array(doc_embeddings)
        results["encode_doc_ms"] = (time.perf_counter() - start) * 1000

        # 3. Encode queries and measure per-query speed
        console.print(f"  Encoding {len(ARABIC_QUERIES)} queries...", style="dim")
        query_embeddings = []
        for query in ARABIC_QUERIES:
            start = time.perf_counter()
            emb = list(model.embed([query]))[0]
            results["encode_query_ms"].append((time.perf_counter() - start) * 1000)
            query_embeddings.append(emb)
        query_embeddings = np.array(query_embeddings)

        # 4. Calculate accuracy metrics
        console.print(f"  Calculating accuracy...", style="dim")
        mrr_scores = []
        recall_1 = 0
        recall_3 = 0
        recall_5 = 0

        for q_idx, q_emb in enumerate(query_embeddings):
            # Calculate similarities to all docs
            similarities = [cosine_similarity(q_emb, d_emb) for d_emb in doc_embeddings]

            # Rank documents by similarity
            ranked_docs = np.argsort(similarities)[::-1]

            # Get relevant docs for this query
            relevant_docs = set(RELEVANCE_MAP.get(q_idx, []))

            if relevant_docs:
                # MRR: Find rank of first relevant doc
                for rank, doc_idx in enumerate(ranked_docs):
                    if doc_idx in relevant_docs:
                        mrr_scores.append(1.0 / (rank + 1))
                        break
                else:
                    mrr_scores.append(0)

                # Recall@K
                top_1 = set(ranked_docs[:1])
                top_3 = set(ranked_docs[:3])
                top_5 = set(ranked_docs[:5])

                if relevant_docs & top_1:
                    recall_1 += 1
                if relevant_docs & top_3:
                    recall_3 += 1
                if relevant_docs & top_5:
                    recall_5 += 1

        num_queries = len(RELEVANCE_MAP)
        results["mrr"] = statistics.mean(mrr_scores) if mrr_scores else 0
        results["recall_at_1"] = recall_1 / num_queries
        results["recall_at_3"] = recall_3 / num_queries
        results["recall_at_5"] = recall_5 / num_queries

    except Exception as e:
        results["error"] = str(e)

    return results


def run_benchmarks():
    """Run benchmarks on all models."""
    console.print("\n[bold cyan]Jassas Embedding Model Benchmark[/bold cyan]")
    console.print(f"Testing {len(MODELS)} models on Arabic text\n")
    console.print("=" * 60)

    all_results = []

    for i, model_name in enumerate(MODELS):
        console.print(f"\n[bold][{i+1}/{len(MODELS)}] {model_name}[/bold]")

        results = benchmark_model(model_name)
        all_results.append(results)

        if results["error"]:
            console.print(f"  [red]Error: {results['error']}[/red]")
        else:
            avg_query_ms = statistics.mean(results["encode_query_ms"])
            console.print(f"  [green]Done[/green] - Query: {avg_query_ms:.1f}ms, MRR: {results['mrr']:.3f}")

    # Print results table
    console.print("\n" + "=" * 60)
    console.print("\n[bold cyan]SPEED RESULTS[/bold cyan]\n")

    speed_table = Table(box=box.ROUNDED)
    speed_table.add_column("Model", style="cyan")
    speed_table.add_column("Load (ms)", justify="right")
    speed_table.add_column("Query Avg (ms)", justify="right")
    speed_table.add_column("Query Min (ms)", justify="right")
    speed_table.add_column("Query Max (ms)", justify="right")
    speed_table.add_column("Docs (ms)", justify="right")

    for r in all_results:
        if r["error"]:
            speed_table.add_row(r["model"], "[red]ERROR[/red]", "", "", "", "")
        else:
            avg_q = statistics.mean(r["encode_query_ms"])
            min_q = min(r["encode_query_ms"])
            max_q = max(r["encode_query_ms"])

            # Color coding for speed
            avg_color = "green" if avg_q < 50 else "yellow" if avg_q < 150 else "red"

            speed_table.add_row(
                r["model"].split("/")[-1],
                f"{r['load_time_ms']:.0f}",
                f"[{avg_color}]{avg_q:.1f}[/{avg_color}]",
                f"{min_q:.1f}",
                f"{max_q:.1f}",
                f"{r['encode_doc_ms']:.0f}",
            )

    console.print(speed_table)

    # Accuracy table
    console.print("\n[bold cyan]ACCURACY RESULTS[/bold cyan]\n")

    acc_table = Table(box=box.ROUNDED)
    acc_table.add_column("Model", style="cyan")
    acc_table.add_column("MRR", justify="right")
    acc_table.add_column("Recall@1", justify="right")
    acc_table.add_column("Recall@3", justify="right")
    acc_table.add_column("Recall@5", justify="right")

    for r in all_results:
        if r["error"]:
            acc_table.add_row(r["model"], "[red]ERROR[/red]", "", "", "")
        else:
            # Color coding for accuracy
            mrr_color = "green" if r["mrr"] > 0.8 else "yellow" if r["mrr"] > 0.5 else "red"
            r1_color = "green" if r["recall_at_1"] > 0.8 else "yellow" if r["recall_at_1"] > 0.5 else "red"

            acc_table.add_row(
                r["model"].split("/")[-1],
                f"[{mrr_color}]{r['mrr']:.3f}[/{mrr_color}]",
                f"[{r1_color}]{r['recall_at_1']:.1%}[/{r1_color}]",
                f"{r['recall_at_3']:.1%}",
                f"{r['recall_at_5']:.1%}",
            )

    console.print(acc_table)

    # Recommendation
    console.print("\n[bold cyan]RECOMMENDATION[/bold cyan]\n")

    # Find best balance (exclude errors)
    valid_results = [r for r in all_results if not r["error"]]

    if valid_results:
        # Score = MRR * 0.6 + Speed_Score * 0.4
        # Speed_Score = 1 - (avg_query_ms / max_query_ms)
        max_query_time = max(statistics.mean(r["encode_query_ms"]) for r in valid_results)

        scored = []
        for r in valid_results:
            avg_q = statistics.mean(r["encode_query_ms"])
            speed_score = 1 - (avg_q / max_query_time) if max_query_time > 0 else 1
            combined = r["mrr"] * 0.6 + speed_score * 0.4
            scored.append((r["model"], combined, avg_q, r["mrr"]))

        scored.sort(key=lambda x: x[1], reverse=True)

        best = scored[0]
        console.print(f"  Best overall: [bold green]{best[0]}[/bold green]")
        console.print(f"    Speed: {best[2]:.1f}ms | MRR: {best[3]:.3f}")

        # Fastest
        fastest = min(valid_results, key=lambda r: statistics.mean(r["encode_query_ms"]))
        console.print(f"\n  Fastest: [bold yellow]{fastest['model']}[/bold yellow]")
        console.print(f"    Speed: {statistics.mean(fastest['encode_query_ms']):.1f}ms | MRR: {fastest['mrr']:.3f}")

        # Most accurate
        most_accurate = max(valid_results, key=lambda r: r["mrr"])
        console.print(f"\n  Most accurate: [bold blue]{most_accurate['model']}[/bold blue]")
        console.print(f"    Speed: {statistics.mean(most_accurate['encode_query_ms']):.1f}ms | MRR: {most_accurate['mrr']:.3f}")


if __name__ == "__main__":
    run_benchmarks()
