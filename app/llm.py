"""Answer a question with the local LLM, grounded in retrieved chunks: the "G" in RAG.

Flow for one question (history = earlier user/assistant turns, may be empty):
    1. route_question()    -> with history, one short LLM call decides whether
                              the question needs the documents at all ("what did
                              I ask first?" doesn't) and rewrites follow-ups
                              ("and its value?") as a standalone search query
    2. search()            -> the 5 best chunks for that query (app/retrieval.py),
                              skipped when no search is needed
    3. build_messages()    -> system prompt + history (oldest turns dropped if
                              they don't fit) + numbered excerpts [1]..[5] + question
    4. stream_chat()       -> sends the messages to Ollama, yields the answer
                              piece by piece as it is generated
    5. check_citations     -> maps [n] in the answer back to document and pages,
                              and flags citations to excerpts that don't exist

Usage:
    python app/llm.py "What is the membrane decay factor in the LIF model?"
    python app/llm.py --chat        # a conversation in the terminal
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
SYSTEM_PROMPT_FILE = ROOT / os.getenv("SYSTEM_PROMPT", "prompts/system_v3.md")
ROUTE_PROMPT_FILE = ROOT / "prompts" / "route_v1.md"

REFUSAL = "I don't know based on the provided documents."
TOP_K = 5

# Context window in tokens: everything the model sees at once (system prompt,
# history, excerpts, question) plus the answer it writes. Ollama's default can
# be smaller, and when the input doesn't fit it drops text SILENTLY.
NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))
OPTIONS = {
    "temperature": 0.1,  # stick to the most likely wording instead of being creative
    "num_ctx": NUM_CTX,
}
# Tokens kept free for the system prompt (~600), 5 excerpts (~2000), the
# question and the answer. Conversation history gets whatever is left.
RESERVED_TOKENS = 4096
# Rough token estimate for trimming history before sending it. Real text is
# ~3.5-4 characters per token; 3 overestimates on purpose, to stay on the safe side.
CHARS_PER_TOKEN = 3
CITATION = re.compile(r"\s*\[\d+\]")


class LLMError(Exception):
    """Ollama unreachable, model missing, or an error from the model server.
    Raised instead of SystemExit so the web server keeps running."""


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()


def build_user_message(question: str, chunks: list[dict]) -> str:
    """Number the chunks so the model can cite them as [1], [2], ..."""
    if not chunks:
        return f"No excerpts: this question is about the conversation itself.\n\nQuestion: {question}"
    parts = ["Excerpts:"]
    for i, chunk in enumerate(chunks, start=1):
        pages = ", ".join(chunk["pages"])
        parts.append(f"[{i}] ({chunk['source']}, p. {pages})\n{chunk['content']}")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- conversation


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN + 1


def clean_history(history: list[dict]) -> list[dict]:
    """Keep only role and text, and remove [n] markers from earlier answers:
    they pointed to that turn's excerpts, which the model no longer sees, and
    would clash with the new excerpts' numbers."""
    cleaned = []
    for m in history:
        content = m["content"]
        if m["role"] == "assistant":
            content = CITATION.sub("", content)
        cleaned.append({"role": m["role"], "content": content})
    return cleaned


def trim_history(history: list[dict], budget: int) -> tuple[list[dict], int]:
    """Drop the oldest messages until the history fits in `budget` tokens.
    Returns (kept messages, number of messages dropped). Messages are dropped
    in question+answer pairs, so the history always starts with a question."""
    kept = list(history)
    while kept and sum(estimate_tokens(m["content"]) for m in kept) > budget:
        kept = kept[2:] if len(kept) >= 2 else []
    return kept, len(history) - len(kept)


def build_messages(
    question: str, chunks: list[dict], history: list[dict] | None = None
) -> tuple[list[dict], int]:
    """The full message list for Ollama, and how many history messages were
    dropped to fit the context window."""
    kept, dropped = trim_history(clean_history(history or []), NUM_CTX - RESERVED_TOKENS)
    messages = [
        {"role": "system", "content": load_system_prompt()},
        *kept,
        {"role": "user", "content": build_user_message(question, chunks)},
    ]
    return messages, dropped


ROUTE_SCHEMA = {
    "type": "object",
    "properties": {"search": {"type": "boolean"}, "query": {"type": "string"}},
    "required": ["search", "query"],
}


def route_question(question: str, history: list[dict] | None) -> dict:
    """Decide whether the question needs a document search, and with what query.

    Returns {"search": bool, "query": str}. Two jobs in one short LLM call:
    - "what did I ask first?" needs no search: the answer is in the history,
      and 5 retrieved excerpts would only be noise.
    - retrieval only sees the text it is given: "what about its time step?"
      finds nothing useful on its own, so follow-ups are rewritten into a
      standalone query, e.g. "LIF neuron time step".
    Without history the question is always searched as is: a first question
    can't refer to a conversation. If the answer can't be parsed, search anyway:
    an unneeded search costs a second, a skipped one costs a wrong answer.
    """
    default = {"search": True, "query": question}
    if not history:
        return default
    lines = []
    for m in clean_history(history[-6:]):  # the last 3 exchanges are enough
        speaker = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {m['content'][:1000]}")
    messages = [
        {"role": "system", "content": ROUTE_PROMPT_FILE.read_text(encoding="utf-8").strip()},
        {
            "role": "user",
            "content": "Conversation:\n" + "\n".join(lines) + f"\n\nLatest message: {question}",
        },
    ]
    options = {"temperature": 0, "num_ctx": NUM_CTX, "num_predict": 150}
    raw = "".join(stream_chat(messages, {}, options, schema=ROUTE_SCHEMA))
    try:
        decision = json.loads(raw)
        search, query = bool(decision["search"]), str(decision["query"]).strip()
    except (json.JSONDecodeError, KeyError, TypeError):
        return default
    if not search:
        return {"search": False, "query": ""}
    return {"search": True, "query": query or question}


def context_usage(stats: dict) -> dict:
    """How full the context window was in this turn: input + output tokens.
    The next turn is about the same size plus this question and answer, so this
    is a good measure of how much room the conversation has left."""
    used = stats.get("input_tokens", 0) + stats.get("output_tokens", 0)
    return {"used": used, "limit": NUM_CTX, "percent": round(100 * used / NUM_CTX, 1)}


# ---------------------------------------------------------------- model server


def stream_chat(
    messages: list[dict], stats: dict, options: dict | None = None, schema: dict | None = None
) -> Iterator[str]:
    """Send messages to Ollama's /api/chat and yield the answer text as it arrives.

    With "stream": true, Ollama answers with one JSON object per line, each
    holding the next few characters; the last one ("done": true) carries the
    statistics stored in `stats` (see read_stats). With a JSON `schema`, Ollama
    forces the output to be JSON matching it (used by route_question).
    """
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": True,
        "think": False,  # no visible reasoning step: faster, and keeps it out of the answer
        "options": options or OPTIONS,
    }
    if schema:
        payload["format"] = schema
    url = f"{LLM_BASE_URL}/api/chat"
    try:
        with httpx.stream(
            "POST", url, json=payload, timeout=httpx.Timeout(10, read=300)
        ) as response:
            if response.status_code != 200:
                response.read()
                detail = response.text
                if response.status_code == 404:
                    raise LLMError(
                        f"Model '{LLM_MODEL}' not found. Run: ollama pull {LLM_MODEL}"
                    )
                raise LLMError(f"Ollama returned {response.status_code}: {detail}")
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise LLMError(f"Ollama error: {data['error']}")
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if data.get("done"):
                    stats.update(read_stats(data))
    except httpx.ConnectError:
        raise LLMError(
            f"Cannot reach Ollama at {LLM_BASE_URL}. Is it running? "
            f"Check with: curl.exe {LLM_BASE_URL}"
        ) from None


def read_stats(data: dict) -> dict:
    """Token counts and timings from Ollama's final message.

    Ollama's own names: prompt_eval_* = reading the input, eval_* = writing the
    output. Durations are in nanoseconds.
      input tokens   system prompt + history + excerpts + question, read in one
                     parallel pass ("prefill"); this is what delays the first word
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


def ask(question: str, on_token=None, history: list[dict] | None = None) -> dict:
    """Full RAG round trip. on_token(piece) is called for each streamed piece.
    history: earlier turns as [{"role": "user"|"assistant", "content": ...}]."""
    timings = {}
    t = time.perf_counter()
    route = route_question(question, history)
    timings["route"] = time.perf_counter() - t

    t = time.perf_counter()
    chunks = retrieval.search(route["query"], top_k=TOP_K) if route["search"] else []
    timings["retrieval"] = time.perf_counter() - t

    messages, dropped = build_messages(question, chunks, history)
    stats, pieces = {}, []
    t = time.perf_counter()
    for piece in stream_chat(messages, stats):
        if not pieces:
            timings["first_token"] = time.perf_counter() - t  # what the user feels as "lag"
        pieces.append(piece)
        if on_token:
            on_token(piece)
    timings["generation"] = time.perf_counter() - t

    answer = "".join(pieces).strip()
    cited, invalid = check_citations(answer, chunks)
    return {
        "question": question,
        "searched": route["search"],
        "search_query": route["query"],
        "answer": answer,
        "refused": REFUSAL.lower() in answer.lower(),
        "cited": cited,
        "invalid_citations": invalid,
        "chunks": chunks,
        "history_dropped": dropped,
        "context": context_usage(stats),
        "timings": timings,
        "stats": stats,
    }


# ---------------------------------------------------------------- command line


def main() -> None:
    try:
        cli()
    except (LLMError, retrieval.RetrievalError) as e:
        raise SystemExit(str(e)) from None


def print_answer(question: str, history: list[dict]) -> dict:
    result = ask(question, on_token=lambda piece: print(piece, end="", flush=True), history=history)
    print("\n")
    if not result["searched"]:
        print("(no document search: question about the conversation)")
    elif result["search_query"] != question:
        print(f"Searched for: {result['search_query']}")
    if result["cited"]:
        print("Sources:")
        for c in result["cited"]:
            print(f"  [{c['n']}] {c['source']}, p. {', '.join(c['pages'])}")
    elif result["searched"] and not result["refused"]:
        print("(no excerpts cited)")
    if result["invalid_citations"]:
        print(f"Warning: cites excerpts that do not exist: {result['invalid_citations']}")
    if result["history_dropped"]:
        print(f"Note: the {result['history_dropped']} oldest messages no longer fit and were left out.")
    return result


def print_report(result: dict) -> None:
    t, s, c = result["timings"], result["stats"], result["context"]
    print(f"\n[{LLM_MODEL}]")
    if t["route"] > 0.01:
        print(f"  route + rewrite  {t['route']:6.2f} s")
    if result["searched"]:
        print(f"  retrieval        {t['retrieval']:6.2f} s")
    if s.get("load_seconds", 0) > 0.5:
        print(f"  model load       {s['load_seconds']:6.2f} s  (first request after a pause)")
    print(
        f"  input tokens     {s.get('input_tokens', 0):6d}    read in "
        f"{s.get('input_seconds', 0):5.2f} s  ({speed(s.get('input_tokens', 0), s.get('input_seconds', 0)):,.0f} tokens/s)"
    )
    print(
        f"  output tokens    {s.get('output_tokens', 0):6d}    written in "
        f"{s.get('output_seconds', 0):5.2f} s  ({speed(s.get('output_tokens', 0), s.get('output_seconds', 0)):,.0f} tokens/s)"
    )
    print(f"  context window   {c['used']:6d} / {c['limit']} tokens ({c['percent']}%)")
    print(f"  first token      {t.get('first_token', 0):6.2f} s  (what the user waits before text appears)")
    print(f"  total            {t['route'] + t['retrieval'] + t['generation']:6.2f} s")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about the thesis corpus.")
    parser.add_argument("question", nargs="?", help="one question (omit with --chat)")
    parser.add_argument("--chat", action="store_true", help="a conversation; type 'reset' to start over, 'exit' to quit")
    args = parser.parse_args()
    if not args.chat and not args.question:
        parser.error("give a question, or use --chat")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not args.chat:
        print(f"\nQ: {args.question}\n")
        print_report(print_answer(args.question, []))
        return

    history: list[dict] = []
    print("Conversation mode. 'reset' starts over, 'exit' quits.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "reset":
            history = []
            print("(conversation reset)")
            continue
        if not question:
            continue
        print()
        result = print_answer(question, history)
        c = result["context"]
        print(f"[context {c['used']} / {c['limit']} tokens, {c['percent']}%]")
        history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": result["answer"]},
        ]


if __name__ == "__main__":
    main()
