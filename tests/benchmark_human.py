"""
Jassas Human Evaluation Benchmark
Interactive CLI for manual grading of search results.
"""
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Load environment variables
load_dotenv()

from ranker.engine import Ranker

# Results file
RESULTS_FILE = Path(__file__).parent / "data" / "human_eval_results.json"

# Test questions - specific queries matching actual DB content
QUESTIONS = [
    # Exact service matches
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


def clear_screen():
    # Disabled to avoid terminal issues
    print("\n" * 2)


def load_previous_results():
    """Load previous evaluation results if they exist."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {"evaluations": [], "summary": {}}


def save_results(results):
    """Save evaluation results."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def print_header():
    print("\n" + "="*60)
    print("       JASSAS HUMAN EVALUATION BENCHMARK")
    print("="*60)


def print_results(question: str, results: list, q_num: int, total: int):
    """Display question and results for grading."""
    clear_screen()
    print_header()
    print(f"\n[Question {q_num}/{total}]")
    print(f"\nQuery: {question}")
    print("\n" + "-"*60)
    print("Search Results:")
    print("-"*60)

    for i, r in enumerate(results[:10], 1):
        title = r.get('title', 'No Title')
        # Truncate long titles
        if len(title) > 70:
            title = title[:67] + "..."
        print(f"\n  [{i}] {title}")

    print("\n" + "-"*60)


def get_grade():
    """Get grade from user."""
    print("\nGrade the search results:")
    print("  [3] Perfect - First result is exactly what user needs")
    print("  [2] Good    - Answer in top 3 results")
    print("  [1] OK      - Answer somewhere in top 10")
    print("  [0] Fail    - No relevant results found")
    print("  [s] Skip    - Skip this question")
    print("  [q] Quit    - Save and exit")

    while True:
        choice = input("\nYour grade (0-3, s, q): ").strip().lower()
        if choice in ['0', '1', '2', '3']:
            return int(choice)
        elif choice == 's':
            return None
        elif choice == 'q':
            return 'quit'
        else:
            print("Invalid input. Enter 0-3, s, or q.")


def calculate_summary(evaluations):
    """Calculate summary statistics."""
    graded = [e for e in evaluations if e['grade'] is not None]

    if not graded:
        return {}

    grades = [e['grade'] for e in graded]

    # Success rates
    success_1 = sum(1 for g in grades if g == 3) / len(grades) * 100
    success_3 = sum(1 for g in grades if g >= 2) / len(grades) * 100
    success_10 = sum(1 for g in grades if g >= 1) / len(grades) * 100

    avg_grade = sum(grades) / len(grades)

    return {
        "total_graded": len(graded),
        "average_grade": round(avg_grade, 2),
        "success_at_1": round(success_1, 1),
        "success_at_3": round(success_3, 1),
        "success_at_10": round(success_10, 1),
        "grade_distribution": {
            "perfect_3": sum(1 for g in grades if g == 3),
            "good_2": sum(1 for g in grades if g == 2),
            "ok_1": sum(1 for g in grades if g == 1),
            "fail_0": sum(1 for g in grades if g == 0),
        }
    }


def print_summary(summary):
    """Print evaluation summary."""
    clear_screen()
    print_header()
    print("\nEVALUATION SUMMARY")
    print("="*60)

    if not summary:
        print("\nNo evaluations yet.")
        return

    print(f"\nTotal Questions Graded: {summary['total_graded']}")
    print(f"Average Grade: {summary['average_grade']}/3")

    print("\n┌─────────────┬────────┐")
    print("│ Metric      │ Score  │")
    print("├─────────────┼────────┤")
    print(f"│ Success@1   │ {summary['success_at_1']:5.1f}% │")
    print(f"│ Success@3   │ {summary['success_at_3']:5.1f}% │")
    print(f"│ Success@10  │ {summary['success_at_10']:5.1f}% │")
    print("└─────────────┴────────┘")

    print("\nGrade Distribution:")
    dist = summary['grade_distribution']
    print(f"  [3] Perfect: {dist['perfect_3']}")
    print(f"  [2] Good:    {dist['good_2']}")
    print(f"  [1] OK:      {dist['ok_1']}")
    print(f"  [0] Fail:    {dist['fail_0']}")

    # Overall grade
    avg = summary['average_grade']
    if avg >= 2.5:
        grade = "A - Excellent"
    elif avg >= 2.0:
        grade = "B - Good"
    elif avg >= 1.5:
        grade = "C - Needs Work"
    else:
        grade = "D - Poor"

    print(f"\n🏆 Overall: {grade}")


def run_evaluation(ranker):
    """Main evaluation loop."""

    # Load previous results
    data = load_previous_results()
    evaluated_questions = {e['question'] for e in data['evaluations']}

    # Filter to unevaluated questions
    remaining = [q for q in QUESTIONS if q not in evaluated_questions]

    if not remaining:
        print("All questions have been evaluated!")
        print_summary(data['summary'])

        response = input("\nReset and start over? (y/n): ").strip().lower()
        if response == 'y':
            data = {"evaluations": [], "summary": {}}
            remaining = QUESTIONS.copy()
        else:
            return

    print(f"\n{len(remaining)} questions remaining to evaluate.")
    input("Press Enter to start...")

    for i, question in enumerate(remaining, 1):
        # Search
        results = ranker.search(question, k=10)

        # Display
        total_remaining = len(remaining)
        print_results(question, results, i, total_remaining)

        # Get grade
        grade = get_grade()

        if grade == 'quit':
            break

        if grade is not None:
            data['evaluations'].append({
                "question": question,
                "grade": grade,
                "timestamp": datetime.now().isoformat(),
                "top_result": results[0]['title'] if results else None,
            })

    # Calculate and save summary
    data['summary'] = calculate_summary(data['evaluations'])
    save_results(data)

    # Show summary
    print_summary(data['summary'])
    print(f"\nResults saved to: {RESULTS_FILE}")


def main():
    print_header()
    print("\nLoading search engine...")
    ranker = Ranker(verbose=False)
    ranker._load_vector_engine()
    print("OK Ready")

    while True:
        print("\n" + "-"*40)
        print("Options:")
        print("  [1] Start/Continue Evaluation")
        print("  [2] View Summary")
        print("  [3] Reset All Results")
        print("  [q] Quit")

        choice = input("\nChoice: ").strip().lower()

        if choice == '1':
            run_evaluation(ranker)
        elif choice == '2':
            data = load_previous_results()
            print_summary(data.get('summary', {}))
        elif choice == '3':
            confirm = input("Delete all results? (yes/no): ").strip().lower()
            if confirm == 'yes':
                if RESULTS_FILE.exists():
                    RESULTS_FILE.unlink()
                print("Results cleared.")
        elif choice == 'q':
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
