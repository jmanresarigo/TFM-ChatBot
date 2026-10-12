"""Prompt building, citations, history, routing and the API client (app/llm.py).
No real API is ever called: model calls are replaced by fakes, and the HTTP
client tests use httpx.MockTransport, a fake server inside the test."""

import json

import httpx
import pytest

import llm

CHUNKS = [
    {"source": "thesis.pdf", "pages": ["3"], "content": "Made-up excerpt one."},
    {"source": "paper.pdf", "pages": ["7", "8"], "content": "Made-up excerpt two."},
]


def fake_json_model(reply):
    """A stand-in for chat_json that 'generates' a fixed reply."""

    def chat_json(messages, schema, endpoint=None, max_tokens=300):
        return reply

    return chat_json


# --- citations -------------------------------------------------------------------


def test_check_citations_maps_numbers_and_flags_invalid_ones():
    cited, invalid = llm.check_citations("A [2]. B [1][2]. C [7].", CHUNKS)
    assert [c["n"] for c in cited] == [2, 1]  # order of first citation, no duplicates
    assert cited[0]["source"] == "paper.pdf"
    assert invalid == [7]


def test_check_citations_without_excerpts_flags_everything():
    assert llm.check_citations("See [1].", []) == ([], [1])


# --- prompt ----------------------------------------------------------------------


def test_user_message_numbers_excerpts_with_sources():
    message = llm.build_user_message("What is it?", CHUNKS)
    assert "[1] (thesis.pdf, p. 3)\nMade-up excerpt one." in message
    assert "[2] (paper.pdf, p. 7, 8)" in message
    assert message.endswith("Question: What is it?")


def test_user_message_without_excerpts_says_so():
    message = llm.build_user_message("What did I ask?", [])
    assert message.startswith("No excerpts")
    assert "[1]" not in message


# --- history -----------------------------------------------------------------------


def test_clean_history_strips_citations_from_answers_only():
    history = [
        {"role": "user", "content": "Is [1] right?"},
        {"role": "assistant", "content": "Yes [1], it is [2][3].", "extra": "x"},
    ]
    assert llm.clean_history(history) == [
        {"role": "user", "content": "Is [1] right?"},
        {"role": "assistant", "content": "Yes, it is."},
    ]


def test_trim_history_drops_oldest_pairs_first():
    long = "word " * 300  # 1,500 characters, about 500 estimated tokens
    history = []
    for i in range(4):
        history += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": long}]
    kept, dropped = llm.trim_history(history, budget=1100)
    assert dropped == 4
    assert [m["content"] for m in kept if m["role"] == "user"] == ["q2", "q3"]
    assert kept[0]["role"] == "user"  # still starts with a question


def test_trim_history_keeps_everything_that_fits():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert llm.trim_history(history, budget=1000) == (history, 0)


def test_build_messages_order():
    history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1 [1]"}]
    messages, dropped = llm.build_messages("q2", CHUNKS, history)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == "a1"
    assert messages[-1]["content"].endswith("Question: q2")
    assert dropped == 0


# --- routing ------------------------------------------------------------------------

HISTORY = [{"role": "user", "content": "What is the decay factor?"}, {"role": "assistant", "content": "It is beta."}]


def test_first_question_is_searched_without_calling_the_model(monkeypatch):
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("the model should not be called without history")

    monkeypatch.setattr(llm, "chat_json", must_not_be_called)
    assert llm.route_question("What is beta?", []) == {"search": True, "query": "What is beta?"}


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"search": true, "query": "decay factor time step"}', {"search": True, "query": "decay factor time step"}),
        ('{"search": false, "query": "ignored"}', {"search": False, "query": ""}),
        ('{"search": true, "query": ""}', {"search": True, "query": "and its time step?"}),
        ("not json at all", {"search": True, "query": "and its time step?"}),  # unreadable: search anyway
    ],
)
def test_route_question_reads_the_model_decision(monkeypatch, reply, expected):
    monkeypatch.setattr(llm, "chat_json", fake_json_model(reply))
    assert llm.route_question("and its time step?", HISTORY) == expected


# --- statistics ------------------------------------------------------------------------


def test_read_stats_from_usage():
    stats = llm.read_stats({"prompt_tokens": 100, "completion_tokens": 50, "prompt_time": 0.2})
    assert stats == {"input_tokens": 100, "output_tokens": 50, "input_seconds": 0.2, "output_seconds": 0.0}


def test_context_usage_percent():
    usage = llm.context_usage({"input_tokens": 3000, "output_tokens": 1096})
    assert usage == {"used": 4096, "limit": llm.NUM_CTX, "percent": round(100 * 4096 / llm.NUM_CTX, 1)}


# --- HTTP client (fake server) ------------------------------------------------------

ENDPOINT = llm.Endpoint("https://api.example.com/v1", "secret-key", "some-model", "none")


def sse_body(*pieces, usage=None):
    """A streamed chat response in the OpenAI format."""
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}" for p in pieces]
    if usage:
        lines.append(f"data: {json.dumps({'choices': [], 'usage': usage})}")
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


@pytest.fixture
def fake_server(monkeypatch):
    """Replace the HTTP client with one whose requests go to `handler`."""

    def install(handler):
        monkeypatch.setattr(llm, "_client", httpx.Client(transport=httpx.MockTransport(handler)))

    return install


def test_stream_chat_yields_text_and_records_usage(fake_server):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, text=sse_body("The answer ", "is beta [1].", usage={"prompt_tokens": 900, "completion_tokens": 12}))

    fake_server(handler)
    stats = {}
    text = "".join(llm.stream_chat([{"role": "user", "content": "q"}], stats, endpoint=ENDPOINT))
    assert text == "The answer is beta [1]."
    assert stats["input_tokens"] == 900 and stats["output_tokens"] == 12
    assert seen["auth"] == "Bearer secret-key"
    assert seen["body"]["model"] == "some-model"
    assert seen["body"]["stream"] is True
    assert seen["body"]["reasoning_effort"] == "none"


def test_rate_limit_is_retried_after_waiting(fake_server, monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)  # don't actually wait
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "3"}, json={"error": {"message": "slow down"}})
        return httpx.Response(200, text=sse_body("ok"))

    fake_server(handler)
    assert "".join(llm.stream_chat([], {}, endpoint=ENDPOINT)) == "ok"
    assert len(calls) == 2


def test_rate_limit_gives_up_beyond_the_wait_budget(fake_server, monkeypatch):
    monkeypatch.setattr(llm, "MAX_RETRY_WAIT", 5)
    fake_server(lambda request: httpx.Response(429, headers={"retry-after": "60"}, json={"error": {"message": "slow down"}}))
    with pytest.raises(llm.LLMError, match="usage limit") as error:
        list(llm.stream_chat([], {}, endpoint=ENDPOINT))
    assert error.value.usage_limit and error.value.retry_after == 60  # from the retry-after header


GROQ_DAILY = (
    "Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_01abc` service tier `on_demand` "
    "on tokens per day (TPD): Limit 500000, Used 499100, Requested 1800. Please try again in 2h31m12s. "
    "Need more tokens? Upgrade to Dev Tier today."
)


def test_rate_limit_message_names_the_limit_and_the_wait(fake_server, monkeypatch):
    monkeypatch.setattr(llm, "MAX_RETRY_WAIT", 0)
    fake_server(lambda request: httpx.Response(429, headers={"retry-after": "9072"}, json={"error": {"message": GROQ_DAILY}}))
    with pytest.raises(llm.LLMError) as error:
        list(llm.stream_chat([], {}, endpoint=ENDPOINT))
    message = str(error.value)
    assert "tokens per day" in message
    assert "about 2 h 32 min" in message
    assert "org_01abc" not in message  # the account ID stays in the server log
    assert error.value.retry_after == 9072
    assert error.value.detail == GROQ_DAILY


@pytest.mark.parametrize(
    "text, seconds",
    [("Please try again in 7.66s.", 7.66), ("try again in 1m26.4s", 86.4), ("try again in 2h31m12s", 9072), ("try again in 560ms", 0.56), ("slow down", None)],
)
def test_parse_wait_reads_groq_durations(text, seconds):
    assert llm.parse_wait(text) == pytest.approx(seconds) if seconds else llm.parse_wait(text) is None


def test_rejected_key_gives_a_clear_error(fake_server):
    fake_server(lambda request: httpx.Response(401, json={"error": {"message": "Invalid API Key"}}))
    with pytest.raises(llm.LLMError, match="LLM_API_KEY"):
        list(llm.stream_chat([], {}, endpoint=ENDPOINT))


def test_chat_json_requests_strict_schema(fake_server):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"search": false, "query": ""}'}}]})

    fake_server(handler)
    reply = llm.chat_json([], llm.ROUTE_SCHEMA, endpoint=ENDPOINT)
    assert json.loads(reply) == {"search": False, "query": ""}
    fmt = seen["body"]["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert "stream" not in seen["body"]  # one complete reply, not streamed


def test_no_key_sends_no_authorization_header(fake_server):
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": [{"id": "local-model"}]})

    fake_server(handler)
    assert llm.list_models(llm.Endpoint("http://127.0.0.1:11434/v1", "", "local-model")) == ["local-model"]
    assert seen["auth"] is None
