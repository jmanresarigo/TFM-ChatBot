"""Score retrieval: does the right page come back for each question?

For every answerable question in evals/questions.jsonl, the question is sent
to app/retrieval.py's search() and the top K chunks are checked. A question is
a HIT at rank r if the r-th chunk comes from an accepted document and shares
at least one page with that document's accepted pages. The main answer is
"source"/"pages"; generic questions may list other documents that also answer
them in "also_accept": [{"source": ..., "pages": [...]}, ...].

Metrics:
    hit@1, hit@3, hit@5  share of questions with a correct chunk in the top 1/3/5
    MRR                  mean reciprocal rank: 1/rank of the first correct chunk
                         (1.0 = always first, 0.5 = always second, 0 = never)

Usage:
    python evals/run_retrieval_eval.py                 # hybrid search
    python evals/run_retrieval_eval.py --mode all      # compare hybrid/vector/keyword
    python evals/run_retrieval_eval.py --save          # also write evals/results/*.json

Unanswerable questions are skipped here: retrieval always returns something.
They are scored later, when the LLM must decide to say "I don't know".
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))  # so "import retrieval" finds app/retrieval.py

import retrieval  # noqa: E402

QUESTIONS_FILE = ROOT / "evals" / "questions.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"
K = 5


def normalize(name: str) -> str:
    """'Review - SNN Models.pdf' and 'Review_-_SNN_Models.pdf' compare equal."""
    return re.sub(r"[\s_\-]+", "", name.lower())


def load_questions() -> list[dict]:
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_sources(questions: list[dict]) -> None:
    """Warn about expected documents that are not in the database: those
    questions could never be hits, which would silently lower the score."""
    with retrieval.connect() as conn:
        stored = {normalize(r[0]) for r in conn.execute("SELECT DISTINCT source FROM chunks")}
    expected = {src for q in questions for src, _ in accepted_answers(q)}
    missing = sorted(name for name in expected if normalize(name) not in stored)
    for name in missing:
        print(f"WARNING: '{name}' is not in the database; its questions will all miss.")


def accepted_answers(question: dict) -> list[tuple[str, set[str]]]:
    """The main (source, pages) plus any alternatives in also_accept."""
    answers = [(question["source"], set(question["pages"]))]
    for alt in question.get("also_accept", []):
        answers.append((alt["source"], set(alt["pages"])))
    return answers


def first_hit_rank(question: dict, results: list[dict]) -> int | None:
    accepted = [(normalize(src), pages) for src, pages in accepted_answers(question)]
    for rank, chunk in enumerate(results, start=1):
        source, pages = normalize(chunk["source"]), set(chunk["pages"])
        if any(source == src and pages & accepted_pages for src, accepted_pages in accepted):
            return rank
    return None


def evaluate(questions: list[dict], mode: str) -> dict:
    per_question = []
    for q in questions:
        results = retrieval.search(q["question"], top_k=K, mode=mode)
        rank = first_hit_rank(q, results)
        top = results[0] if results else None
        per_question.append({
            "id": q["id"],
            "type": q["type"],
            "rank": rank,
            "top1": f"{top['source']} p.{','.join(top['pages'])}" if top else "-",
        })

    def summarize(rows: list[dict]) -> dict:
        n = len(rows)
        ranks = [r["rank"] for r in rows]
        return {
            "n": n,
            "hit@1": sum(r is not None and r <= 1 for r in ranks) / n,
            "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
            "hit@5": sum(r is not None for r in ranks) / n,
            "mrr": sum(1 / r for r in ranks if r) / n,
        }

    by_type = defaultdict(list)
    for row in per_question:
        by_type[row["type"]].append(row)
    return {
        "mode": mode,
        "overall": summarize(per_question),
        "by_type": {t: summarize(rows) for t, rows in sorted(by_type.items())},
        "questions": per_question,
    }


def print_report(report: dict, questions: dict[str, dict], verbose: bool) -> None:
    def line(label: str, s: dict) -> str:
        return (f"  {label:<12} n={s['n']:<3} hit@1 {s['hit@1']:5.0%}   hit@3 {s['hit@3']:5.0%}"
                f"   hit@5 {s['hit@5']:5.0%}   MRR {s['mrr']:.2f}")

    print(f"\n=== mode: {report['mode']} ===")
    print(line("overall", report["overall"]))
    for qtype, s in report["by_type"].items():
        print(line(qtype, s))

    misses = [r for r in report["questions"] if r["rank"] is None]
    if misses:
        print(f"\n  Misses ({len(misses)}): expected -> got as top result")
        for r in misses:
            q = questions[r["id"]]
            print(f"  {r['id']} {q['question'][:60]:<60}")
            expected = " or ".join(f"{src} p.{','.join(sorted(pg))}" for src, pg in accepted_answers(q))
            print(f"       expected {expected}")
            print(f"       got      {r['top1']}")
    if verbose:
        print("\n  Per question (rank of first correct chunk):")
        for r in report["questions"]:
            print(f"  {r['id']} {r['type']:<10} {r['rank'] or 'miss'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score retrieval against evals/questions.jsonl")
    parser.add_argument("--mode", choices=["hybrid", "vector", "keyword", "all"], default="hybrid")
    parser.add_argument("--save", action="store_true", help="write results to evals/results/")
    parser.add_argument("--verbose", action="store_true", help="list every question's rank")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    all_questions = load_questions()
    questions = [q for q in all_questions if q["type"] != "unanswerable"]
    skipped = len(all_questions) - len(questions)
    print(f"{len(questions)} answerable questions ({skipped} unanswerable skipped), top {K}")
    check_sources(questions)

    by_id = {q["id"]: q for q in questions}
    modes = ["hybrid", "vector", "keyword"] if args.mode == "all" else [args.mode]
    for mode in modes:
        start = time.perf_counter()
        report = evaluate(questions, mode)
        report["seconds"] = round(time.perf_counter() - start, 1)
        report["embedding_model"] = retrieval.EMBEDDING_MODEL
        print_report(report, by_id, args.verbose)
        if args.save:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            out = RESULTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{mode}.json"
            out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\n  Saved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()