"""Evaluate generated answers in two steps.

Step 1 - generate (Layer 1: deterministic checks)
    python evals/run_answer_eval.py generate
    Runs all questions through app/llm.py's ask() and checks with plain code:
      - unanswerable questions: was the refusal sentence used?
      - answerable questions: false refusals, invalid citations ([7] with 5
        excerpts), uncited answers, and whether a cited chunk is an expected page
      - attribution: was the right page even retrieved? (retrieval vs generation)

Step 2 - judge (Layer 2: LLM as judge, a DIFFERENT model family)
    python evals/run_answer_eval.py judge          # judges the most recent run
    python evals/run_answer_eval.py judge data/eval_runs/<run>.jsonl
    A second local model (JUDGE_MODEL in .env) grades each answered question:
      - correctness 0/1/2 against the key facts in questions.jsonl
      - faithfulness: is every claim supported by the excerpts?

Where results go:
    data/eval_runs/<run>.jsonl          full records, INCLUDING excerpt text
                                        -> under data/, so git-ignored (thesis text)
    evals/results/<run>_summary.json    scores only, no document text -> commit

Re-print a summary:  python evals/run_answer_eval.py report [data/eval_runs/<run>.jsonl]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))  # so "import llm" finds app/llm.py

import httpx  # noqa: E402

import llm  # noqa: E402
import retrieval  # noqa: E402

QUESTIONS_FILE = ROOT / "evals" / "questions.jsonl"
RUNS_DIR = ROOT / "data" / "eval_runs"
RESULTS_DIR = ROOT / "evals" / "results"

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "")
JUDGE_PROMPT_FILE = ROOT / os.getenv("JUDGE_PROMPT", "prompts/judge_v1.md")

# Ollama forces the judge's output to match this JSON schema. "reasoning" comes
# first on purpose: the model writes its justification before the grades.
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "correctness": {"type": "integer", "enum": [0, 1, 2]},
        "faithful": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reasoning", "correctness", "faithful", "unsupported_claims"],
}


# --- Helpers ------------------------------------------------------------------


def normalize(name: str) -> str:
    """'Review - SNN Models.pdf' and 'Review_-_SNN_Models.pdf' compare equal."""
    return re.sub(r"[\s_\-]+", "", name.lower())


def accepted_answers(q: dict) -> list[tuple[str, set[str]]]:
    if not q.get("source"):
        return []
    answers = [(normalize(q["source"]), set(q["pages"]))]
    for alt in q.get("also_accept", []):
        answers.append((normalize(alt["source"]), set(alt["pages"])))
    return answers


def matches(chunk: dict, accepted: list[tuple[str, set[str]]]) -> bool:
    source, pages = normalize(chunk["source"]), set(chunk["pages"])
    return any(source == src and pages & pg for src, pg in accepted)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", text).strip("-")


# --- Step 1: generate + deterministic checks -------------------------------------


def layer1_checks(q: dict, result: dict) -> dict:
    """Everything that plain code can decide, without judgement."""
    if q["type"] == "unanswerable":
        return {"correct_refusal": result["refused"]}

    accepted = accepted_answers(q)
    context_hit = any(matches(c, accepted) for c in result["chunks"])
    citation_hit = any(matches(c, accepted) for c in result["cited"])
    # "uncited" = no [n] at all; an answer citing only [7] is "invalid", not uncited
    uncited = (
        not result["refused"]
        and not result["cited"]
        and not result["invalid_citations"]
    )

    if not context_hit:
        failure = "retrieval"  # the right page never reached the model
    elif (
        result["refused"] or uncited or result["invalid_citations"] or not citation_hit
    ):
        failure = "generation"  # it was there, the model didn't use it well
    else:
        failure = None
    return {
        "false_refusal": result["refused"],
        "invalid_citations": result["invalid_citations"],
        "uncited": uncited,
        "context_hit": context_hit,
        "citation_hit": citation_hit,
        "layer1_failure": failure,
    }


def generate() -> Path:
    questions = load_jsonl(QUESTIONS_FILE)
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{safe_name(llm.LLM_MODEL)}"
    print(f"Run {run_id}: {len(questions)} questions, model {llm.LLM_MODEL}\n")

    rows = []
    for q in questions:
        start = time.perf_counter()
        result = llm.ask(q["question"])
        checks = layer1_checks(q, result)
        rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "key_facts": q["answer"],
                "answer": result["answer"],
                "refused": result["refused"],
                "cited": [
                    {"n": c["n"], "source": c["source"], "pages": c["pages"]}
                    for c in result["cited"]
                ],
                "excerpts": [
                    {
                        "n": i,
                        "source": c["source"],
                        "pages": c["pages"],
                        "content": c["content"],
                    }
                    for i, c in enumerate(result["chunks"], start=1)
                ],
                "checks": checks,
                "seconds": round(time.perf_counter() - start, 1),
                "stats": result["stats"],
            }
        )
        print(
            f"  {q['id']} {q['type']:<12} {describe(rows[-1]):<40} {rows[-1]['seconds']:5.1f} s"
        )

    run_file = RUNS_DIR / f"{run_id}.jsonl"
    save_jsonl(run_file, rows)
    config = {
        "llm_model": llm.LLM_MODEL,
        "system_prompt": llm.SYSTEM_PROMPT_FILE.name,
        "embedding_model": retrieval.EMBEDDING_MODEL,
        "top_k": llm.TOP_K,
    }
    report(run_file, config)
    return run_file


def describe(row: dict) -> str:
    c = row["checks"]
    if row["type"] == "unanswerable":
        return (
            "refused (correct)" if c["correct_refusal"] else "ANSWERED (should refuse)"
        )
    if c["false_refusal"]:
        return "REFUSED (should answer)"
    flags = []
    if not c["context_hit"]:
        flags.append("right page not retrieved")
    if c["uncited"]:
        flags.append("no citations")
    if c["invalid_citations"]:
        flags.append(f"invalid {c['invalid_citations']}")
    if row["cited"] and not c["citation_hit"] and c["context_hit"]:
        flags.append("cites wrong page")
    return "ok" if not flags else ", ".join(flags)


# --- Step 2: LLM as judge ------------------------------------------------------


def judge_one(row: dict, system_prompt: str) -> dict:
    excerpts = "\n\n".join(
        f"[{e['n']}] ({e['source']}, p. {', '.join(e['pages'])})\n{e['content']}"
        for e in row["excerpts"]
    )
    user = (
        f"QUESTION:\n{row['question']}\n\nKEY FACTS:\n{row['key_facts']}\n\n"
        f"EXCERPTS:\n{excerpts}\n\nANSWER:\n{row['answer']}"
    )
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "format": JUDGE_SCHEMA,  # forces valid JSON with exactly these fields
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    response = httpx.post(
        f"{llm.LLM_BASE_URL}/api/chat",
        json=payload,
        timeout=httpx.Timeout(10, read=600),
    )
    if response.status_code == 404:
        raise SystemExit(
            f"Judge model '{JUDGE_MODEL}' not found. Run: ollama pull {JUDGE_MODEL}"
        )
    response.raise_for_status()
    return json.loads(response.json()["message"]["content"])


def judge(run_file: Path) -> None:
    if not JUDGE_MODEL:
        raise SystemExit(
            "Set JUDGE_MODEL in .env (a different model family than LLM_MODEL)."
        )
    if JUDGE_MODEL.split(":")[0] == llm.LLM_MODEL.split(":")[0]:
        print(
            f"Warning: judge '{JUDGE_MODEL}' looks like the same family as '{llm.LLM_MODEL}'."
        )

    rows = load_jsonl(run_file)
    system_prompt = JUDGE_PROMPT_FILE.read_text(encoding="utf-8").strip()
    print(f"Judging {run_file.name} with {JUDGE_MODEL}\n")
    for row in rows:
        # Only answered, answerable questions need a judge: refusals and
        # unanswerable questions are already decided by Layer 1.
        if row["type"] == "unanswerable" or row["refused"]:
            row.pop("judge", None)
            continue
        start = time.perf_counter()
        verdict = judge_one(row, system_prompt)
        row["judge"] = verdict
        mark = {2: "correct", 1: "partial", 0: "WRONG"}[verdict["correctness"]]
        faithful = "faithful" if verdict["faithful"] else "UNFAITHFUL"
        print(
            f"  {row['id']} {mark:<8} {faithful:<11} {time.perf_counter() - start:5.1f} s"
        )

    save_jsonl(run_file, rows)
    report(
        run_file, {"judge_model": JUDGE_MODEL, "judge_prompt": JUDGE_PROMPT_FILE.name}
    )


# --- Report --------------------------------------------------------------------


def pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "-"


def report(run_file: Path, extra_config: dict | None = None) -> None:
    rows = load_jsonl(run_file)
    unans = [r for r in rows if r["type"] == "unanswerable"]
    ans = [r for r in rows if r["type"] != "unanswerable"]
    answered = [r for r in ans if not r["refused"]]
    judged = [r for r in ans if "judge" in r]

    s = {
        "correct_refusals": sum(r["checks"]["correct_refusal"] for r in unans),
        "false_refusals": sum(r["checks"]["false_refusal"] for r in ans),
        "invalid_citation_answers": sum(
            bool(r["checks"]["invalid_citations"]) for r in ans
        ),
        "uncited": sum(r["checks"]["uncited"] for r in ans),
        "citation_hits": sum(r["checks"]["citation_hit"] for r in ans),
        "context_hits": sum(r["checks"]["context_hit"] for r in ans),
        "retrieval_failures": sum(
            r["checks"]["layer1_failure"] == "retrieval" for r in ans
        ),
        "generation_failures": sum(
            r["checks"]["layer1_failure"] == "generation" for r in ans
        ),
    }
    print("\n=== Layer 1: deterministic checks ===")
    print(
        f"  Unanswerable correctly refused   {pct(s['correct_refusals'], len(unans))}"
    )
    print(f"  Answerable falsely refused       {pct(s['false_refusals'], len(ans))}")
    print(
        f"  Answers with invalid citations   {pct(s['invalid_citation_answers'], len(ans))}"
    )
    print(f"  Answers without citations        {pct(s['uncited'], len(answered))}")
    print(f"  Right page in the 5 excerpts     {pct(s['context_hits'], len(ans))}")
    print(f"  Cites an expected page           {pct(s['citation_hits'], len(ans))}")
    print(
        f"  Failures: retrieval {s['retrieval_failures']}  |  generation {s['generation_failures']}"
    )

    if judged:
        grades = [r["judge"]["correctness"] for r in judged]
        s.update(
            {
                "judged": len(judged),
                "correct": grades.count(2),
                "partial": grades.count(1),
                "wrong": grades.count(0),
                "faithful": sum(r["judge"]["faithful"] for r in judged),
            }
        )
        print("\n=== Layer 2: LLM judge ===")
        print(
            f"  Correct / partial / wrong        {s['correct']} / {s['partial']} / {s['wrong']}  (of {len(judged)} answered)"
        )
        print(f"  Faithful to the excerpts         {pct(s['faithful'], len(judged))}")
        flagged = [
            r
            for r in judged
            if r["judge"]["correctness"] < 2 or not r["judge"]["faithful"]
        ]
        if flagged:
            print("\n  To review:")
            for r in flagged:
                print(
                    f"  {r['id']} correctness {r['judge']['correctness']}, faithful {r['judge']['faithful']}: "
                    f"{r['judge']['reasoning'][:110]}"
                )

    # A summary without any document text: safe to commit
    summary_file = RESULTS_DIR / f"{run_file.stem}_summary.json"
    old = (
        json.loads(summary_file.read_text(encoding="utf-8"))
        if summary_file.exists()
        else {}
    )
    config = {**old.get("config", {}), **(extra_config or {})}
    per_question = [
        {
            "id": r["id"],
            "type": r["type"],
            **r["checks"],
            **(
                {
                    "correctness": r["judge"]["correctness"],
                    "faithful": r["judge"]["faithful"],
                }
                if "judge" in r
                else {}
            ),
        }
        for r in rows
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(
        json.dumps(
            {"config": config, "scores": s, "questions": per_question}, indent=2
        ),
        encoding="utf-8",
    )
    print(
        f"\nFull records: {run_file.relative_to(ROOT)}  (git-ignored, contains excerpts)"
    )
    print(
        f"Summary:      {summary_file.relative_to(ROOT)}  (no document text, commit it)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated answers.")
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("generate", help="run all questions and apply deterministic checks")
    for name in ("judge", "report"):
        p = sub.add_parser(name)
        p.add_argument(
            "run_file", type=Path, nargs="?", help="default: the most recent run"
        )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.step == "generate":
        generate()
        return
    if args.run_file is None:
        runs = sorted(RUNS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not runs:
            raise SystemExit(
                "No runs found in data/eval_runs/. Run: python evals/run_answer_eval.py generate"
            )
        run_file = runs[-1]
        print(f"Using the most recent run: {run_file.name}")
    else:
        run_file = (
            args.run_file.resolve()
        )  # accept relative paths like data/eval_runs/x.jsonl
    if not run_file.exists():
        available = (
            "\n  ".join(p.name for p in sorted(RUNS_DIR.glob("*.jsonl"))) or "(none)"
        )
        raise SystemExit(
            f"Run file not found: {args.run_file}\nAvailable runs in data/eval_runs/:\n  {available}"
        )
    if args.step == "judge":
        judge(run_file)
    else:
        report(run_file)


if __name__ == "__main__":
    try:
        main()
    except (llm.LLMError, retrieval.RetrievalError) as e:
        raise SystemExit(str(e)) from None
