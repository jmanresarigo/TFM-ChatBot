"""Answer a question with the local LLM, grounded in retrieved chunks: the "G" in RAG.

Flow for one question:
    1. search()            -> the 5 best chunks (app/retrieval.py)
    2. build_user_message  -> numbered excerpts [1]..[5] + the question
    3. stream_chat()       -> sends system prompt + user message to Ollama,
                              yields the answer piece by piece as it is generated
    4. check_citations     -> maps [n] in the answer back to document and pages,
                              and flags citations to excerpts that don't exist

Usage:
    python app/llm.py "What is the membrane decay factor in the LIF model?"
"""

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx

import retrieval  # also loads .env

ROOT = Path(__file__).resolve().parent.parent
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4:12b")
SYSTEM_PROMPT_FILE = ROOT / os.getenv("SYSTEM_PROMPT", "prompts/system_v1.md")

REFUSAL = "I don't know based on the provided documents."
TOP_K = 5
OPTIONS = {
    # Low temperature: stick to the most likely wording instead of being creative.
    "temperature": 0.1,
    # Context window in tokens. Ollama's default can be smaller than system
    # prompt + 5 chunks + question, and it truncates SILENTLY when exceeded.
    "num_ctx": 8192,
}


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()


def build_user_message(question: str, chunks: list[dict]) -> str:
    """Number the chunks so the model can cite them as [1], [2], ..."""
    parts = ["Excerpts:"]
    for i, chunk in enumerate(chunks, start=1):
        pages = ", ".join(chunk["pages"])
        parts.append(f"[{i}] ({chunk['source']}, p. {pages})\n{chunk['content']}")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def stream_chat(messages: list[dict], stats: dict) -> Iterator[str]:
    """Send messages to Ollama's /api/chat and yield the answer text as it arrives.

    With "stream": true, Ollama answers with one JSON object per line, each
    holding the next few characters; the last one ("done": true) carries the
    statistics stored in `stats` (see read_stats).
    """
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": True,
        "think": False,  # no visible reasoning step: faster, and keeps it out of the answer
        "options": OPTIONS,
    }
    url = f"{LLM_BASE_URL}/api/chat"
    try:
        with httpx.stream(
            "POST", url, json=payload, timeout=httpx.Timeout(10, read=300)
        ) as response:
            if response.status_code != 200:
                response.read()
                detail = response.text
                if response.status_code == 404:
                    raise SystemExit(
                        f"Model '{LLM_MODEL}' not found. Run: ollama pull {LLM_MODEL}"
                    )
                raise SystemExit(f"Ollama returned {response.status_code}: {detail}")
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise SystemExit(f"Ollama error: {data['error']}")
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if data.get("done"):
                    stats.update(read_stats(data))
    except httpx.ConnectError:
        raise SystemExit(
            f"Cannot reach Ollama at {LLM_BASE_URL}. Is it running? "
            f"Check with: curl.exe {LLM_BASE_URL}"
        ) from None


def read_stats(data: dict) -> dict:
    """Token counts and timings from Ollama's final message.

    Ollama's own names: prompt_eval_* = reading the input, eval_* = writing the
    output. Durations are in nanoseconds.
      input tokens   system prompt + excerpts + question, read in one parallel
                     pass ("prefill"); this is what delays the first word
      output tokens  the answer, generated one token at a time ("decode");
                     this sets the speed at which text appears
    """

    def seconds(key: str) -> float:
        return data.get(key, 0) / 1e9

    return {
        "input_tokens": data.get("prompt_eval_count", 0),
        "input_seconds": seconds("prompt_eval_duration"),
        "output_tokens": data.get("eval_count", 0),
        "output_seconds": seconds("eval_duration"),
        "load_seconds": seconds("load_duration"),  # loading the model into VRAM
    }


def speed(tokens: int, secs: float) -> float:
    return tokens / secs if secs > 0 else 0.0


def check_citations(answer: str, chunks: list[dict]) -> tuple[list[dict], list[int]]:
    """Return the cited chunks (in order of first citation) and invalid numbers."""
    cited, invalid = [], []
    for number in dict.fromkeys(int(n) for n in re.findall(r"\[(\d+)\]", answer)):
        if 1 <= number <= len(chunks):
            cited.append({"n": number, **chunks[number - 1]})
        else:
            invalid.append(number)
    return cited, invalid


def ask(question: str, on_token=None) -> dict:
    """Full RAG round trip. on_token(piece) is called for each streamed piece."""
    timings = {}
    t = time.perf_counter()
    chunks = retrieval.search(question, top_k=TOP_K)
    timings["retrieval"] = time.perf_counter() - t

    messages = [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user", "content": build_user_message(question, chunks)},
    ]
    stats, pieces = {}, []
    t = time.perf_counter()
    for piece in stream_chat(messages, stats):
        if not pieces:
            timings["first_token"] = (
                time.perf_counter() - t
            )  # what the user feels as "lag"
        pieces.append(piece)
        if on_token:
            on_token(piece)
    timings["generation"] = time.perf_counter() - t

    answer = "".join(pieces).strip()
    cited, invalid = check_citations(answer, chunks)
    return {
        "question": question,
        "answer": answer,
        "refused": REFUSAL.lower() in answer.lower(),
        "cited": cited,
        "invalid_citations": invalid,
        "chunks": chunks,
        "timings": timings,
        "stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question about the thesis corpus."
    )
    parser.add_argument("question")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"\nQ: {args.question}\n")
    result = ask(args.question, on_token=lambda piece: print(piece, end="", flush=True))
    print("\n")

    if result["cited"]:
        print("Sources:")
        for c in result["cited"]:
            print(f"  [{c['n']}] {c['source']}, p. {', '.join(c['pages'])}")
    elif not result["refused"]:
        print("Warning: the answer cites no excerpts.")
    if result["invalid_citations"]:
        print(
            f"Warning: cites excerpts that do not exist: {result['invalid_citations']}"
        )

    t, s = result["timings"], result["stats"]
    print(f"\n[{LLM_MODEL}]")
    print(f"  retrieval        {t['retrieval']:6.2f} s")
    if s.get("load_seconds", 0) > 0.5:
        print(
            f"  model load       {s['load_seconds']:6.2f} s  (first request after a pause)"
        )
    print(
        f"  input tokens     {s.get('input_tokens', 0):6d}    read in "
        f"{s.get('input_seconds', 0):5.2f} s  ({speed(s.get('input_tokens', 0), s.get('input_seconds', 0)):,.0f} tokens/s)"
    )
    print(
        f"  output tokens    {s.get('output_tokens', 0):6d}    written in "
        f"{s.get('output_seconds', 0):5.2f} s  ({speed(s.get('output_tokens', 0), s.get('output_seconds', 0)):,.0f} tokens/s)"
    )
    print(
        f"  first token      {t.get('first_token', 0):6.2f} s"
    )
    print(f"  total            {t['retrieval'] + t['generation']:6.2f} s")


if __name__ == "__main__":
    main()