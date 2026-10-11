"""Answer a question with an LLM, grounded in retrieved chunks: the "G" in RAG.

The LLM is reached through the OpenAI-compatible chat API, the format most
providers speak: Groq (default), OpenAI, OpenRouter, ... and also a local Ollama
(http://127.0.0.1:11434/v1). Switching provider = changing LLM_BASE_URL,
LLM_API_KEY and LLM_MODEL in .env; no code changes.

Flow for one question (history = earlier user/assistant turns, may be empty):
    1. route_question()    -> with history, one short LLM call decides whether
                              the question needs the documents at all ("what did
                              I ask first?" doesn't) and rewrites follow-ups
                              ("and its value?") as a standalone search query
    2. search()            -> the 5 best chunks for that query (app/retrieval.py),
                              skipped when no search is needed
    3. build_messages()    -> system prompt + history (oldest turns dropped if
                              they don't fit) + numbered excerpts [1]..[5] + question
    4. stream_chat()       -> sends the messages to the LLM API, yields the answer
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
from dataclasses import dataclass
from pathlib import Path

import httpx

import retrieval  # also loads .env

ROOT = Path(__file__).resolve().parent.parent
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # empty for a local Ollama
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
# Thinking before answering: "none" turns it off for Qwen (faster, and all
# tokens go to the answer). gpt-oss models can't turn it off: use "low".
# Empty = don't send the setting (e.g. for a local Ollama).
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "none")
# The router (search or not?) may use a different model, so its calls don't
# use up the main model's quota. Defaults to the main model.
ROUTER_MODEL = os.getenv("ROUTER_MODEL", LLM_MODEL)
ROUTER_REASONING_EFFORT = os.getenv("ROUTER_REASONING_EFFORT", LLM_REASONING_EFFORT)
SYSTEM_PROMPT_FILE = ROOT / os.getenv("SYSTEM_PROMPT", "prompts/system_v3.md")
ROUTE_PROMPT_FILE = ROOT / "prompts" / "route_v1.md"

REFUSAL = "I don't know based on the provided documents."
TOP_K = 5

# Token budget per request: everything the model sees (system prompt, history,
# excerpts, question) plus the answer. The model itself accepts more, but
# Groq's free tier allows 8,000 tokens per minute, so one request must stay
# below that anyway. History is trimmed to fit (see trim_history).
NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))
TEMPERATURE = 0.1  # stick to the most likely wording instead of being creative
MAX_ANSWER_TOKENS = 1024  # caps the answer length, and so the quota one answer can use
# On "too many requests" (HTTP 429) wait and retry, up to this many seconds in
# total. Short for the web app (a user is waiting); the eval script raises it.
MAX_RETRY_WAIT = float(os.getenv("LLM_MAX_RETRY_WAIT", "20"))
# Tokens kept free for the system prompt (~600), 5 excerpts (~2000), the
# question and the answer. Conversation history gets whatever is left.
RESERVED_TOKENS = 4096
# Rough token estimate for trimming history before sending it. Real text is
# ~3.5-4 characters per token; 3 overestimates on purpose, to stay on the safe side.
CHARS_PER_TOKEN = 3
CITATION = re.compile(r"\s*\[\d+\]")


class LLMError(Exception):
    """API unreachable, key rejected, model missing, rate limit exhausted, or
    another error from the provider. Raised instead of SystemExit so the web
    server keeps running."""


@dataclass(frozen=True)
class Endpoint:
    """Where and how to call a model: the provider's URL, key, model name."""

    base_url: str
    api_key: str
    model: str
    reasoning_effort: str = ""


MAIN = Endpoint(LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_REASONING_EFFORT)
ROUTER = Endpoint(LLM_BASE_URL, LLM_API_KEY, ROUTER_MODEL, ROUTER_REASONING_EFFORT)


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
    """The full message list for the LLM, and how many history messages were
    dropped to fit the context window."""
    kept, dropped = trim_history(clean_history(history or []), NUM_CTX - RESERVED_TOKENS)
    messages = [
        {"role": "system", "content": load_system_prompt()},
        *kept,
        {"role": "user", "content": build_user_message(question, chunks)},
    ]
    return messages, dropped


# Strict schemas: the provider guarantees the reply is JSON with exactly these
# fields ("strict" requires every field listed in "required" and no extras).
ROUTE_SCHEMA = {
    "type": "object",
    "properties": {"search": {"type": "boolean"}, "query": {"type": "string"}},
    "required": ["search", "query"],
    "additionalProperties": False,
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
    raw = chat_json(messages, ROUTE_SCHEMA, endpoint=ROUTER, max_tokens=150)
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

_client = httpx.Client(timeout=httpx.Timeout(10, read=300))  # reused: keeps connections open


def _payload(endpoint: Endpoint, messages: list[dict], **fields) -> dict:
    payload = {"model": endpoint.model, "messages": messages, **fields}
    if endpoint.reasoning_effort:
        payload["reasoning_effort"] = endpoint.reasoning_effort
    return payload


def _headers(endpoint: Endpoint) -> dict:
    return {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}


def _error(endpoint: Endpoint, response: httpx.Response) -> LLMError:
    """A readable error for a failed request (the body has already been read)."""
    try:
        detail = response.json().get("error", {}).get("message", response.text)
    except ValueError:
        detail = response.text
    if response.status_code == 401:
        return LLMError(f"The API key was rejected by {endpoint.base_url}. Check LLM_API_KEY in .env.")
    if response.status_code == 404:
        return LLMError(f"Model '{endpoint.model}' not found at {endpoint.base_url}: {detail}")
    if response.status_code == 429:
        return LLMError(f"Rate limit reached for '{endpoint.model}' (free tier). Try again in a minute. {detail}")
    return LLMError(f"LLM API returned {response.status_code}: {detail}")


def _retry_delay(response: httpx.Response) -> float:
    """Seconds to wait before retrying a 429/503, from the retry-after header."""
    try:
        return max(float(response.headers.get("retry-after", "2")), 0.5)
    except ValueError:
        return 2.0


def _send(endpoint: Endpoint, payload: dict, stream: bool):
    """Open the request, waiting and retrying while the provider says
    "too many requests" (429) or "busy" (503), within MAX_RETRY_WAIT seconds.
    Returns an open response (the caller must close it)."""
    waited = 0.0
    while True:
        try:
            request = _client.build_request(
                "POST", f"{endpoint.base_url}/chat/completions", json=payload, headers=_headers(endpoint)
            )
            response = _client.send(request, stream=stream)
        except httpx.ConnectError:
            raise LLMError(f"Cannot reach the LLM API at {endpoint.base_url}. Check LLM_BASE_URL and your connection.") from None
        except httpx.TimeoutException:
            raise LLMError(f"The LLM API at {endpoint.base_url} did not answer in time.") from None
        if response.status_code == 200:
            return response
        response.read()
        if response.status_code in (429, 503):
            delay = _retry_delay(response)
            if waited + delay <= MAX_RETRY_WAIT:
                print(f"  (rate limited by the LLM API, waiting {delay:.0f} s)", file=sys.stderr)
                response.close()
                time.sleep(delay)
                waited += delay
                continue
        response.close()
        raise _error(endpoint, response)


def stream_chat(
    messages: list[dict],
    stats: dict,
    endpoint: Endpoint = MAIN,
    max_tokens: int = MAX_ANSWER_TOKENS,
) -> Iterator[str]:
    """Send messages to the chat API and yield the answer text as it arrives.

    With "stream": true the API answers with Server-Sent Events: lines
    "data: {json}", each holding the next few characters in
    choices[0].delta.content, and finally "data: [DONE]". With
    stream_options.include_usage, the last chunk carries token counts, stored
    in `stats` (see read_stats).
    """
    payload = _payload(
        endpoint,
        messages,
        stream=True,
        stream_options={"include_usage": True},
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
    )
    response = _send(endpoint, payload, stream=True)
    try:
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue  # blank lines and SSE comments
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise LLMError(f"LLM API error: {chunk['error']}")
            for choice in chunk.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    yield piece
            usage = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage")
            if usage:
                stats.update(read_stats(usage))
    finally:
        response.close()


def chat_json(
    messages: list[dict], schema: dict, endpoint: Endpoint = MAIN, max_tokens: int = 300
) -> str:
    """One non-streamed call whose reply must be JSON matching `schema`
    (strict structured output). Returns the raw JSON text; callers parse it."""
    payload = _payload(
        endpoint,
        messages,
        temperature=0,
        max_tokens=max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "reply", "strict": True, "schema": schema},
        },
    )
    response = _send(endpoint, payload, stream=False)
    try:
        return response.json()["choices"][0]["message"]["content"] or ""
    finally:
        response.close()


def list_models(endpoint: Endpoint = MAIN) -> list[str]:
    """Model names the provider offers with this key (GET /models)."""
    try:
        response = _client.get(f"{endpoint.base_url}/models", headers=_headers(endpoint), timeout=5)
    except httpx.HTTPError as e:
        raise LLMError(f"Cannot reach the LLM API at {endpoint.base_url}: {e}") from None
    if response.status_code != 200:
        raise _error(endpoint, response)
    return [m["id"] for m in response.json().get("data", [])]


def read_stats(usage: dict) -> dict:
    """Token counts (and timings, when the provider reports them) for one call.

      input tokens   system prompt + history + excerpts + question
      output tokens  the answer (plus hidden reasoning, for models that think)
    Groq also reports how long it spent reading the input and writing the output.
    """
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "input_seconds": usage.get("prompt_time", 0.0),
        "output_seconds": usage.get("completion_time", 0.0),
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
