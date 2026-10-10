"""Prompt building, citations, history and routing (app/llm.py). The model
server is never called: stream_chat is replaced by a fake where needed."""

import pytest

import llm

CHUNKS = [
    {"source": "thesis.pdf", "pages": ["3"], "content": "Made-up excerpt one."},
    {"source": "paper.pdf", "pages": ["7", "8"], "content": "Made-up excerpt two."},
]


def fake_model(reply):
    """A stand-in for stream_chat that 'generates' a fixed reply."""

    def stream_chat(messages, stats, options=None, schema=None):
        yield reply

    return stream_chat


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

    monkeypatch.setattr(llm, "stream_chat", must_not_be_called)
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
    monkeypatch.setattr(llm, "stream_chat", fake_model(reply))
    assert llm.route_question("and its time step?", HISTORY) == expected


# --- statistics ------------------------------------------------------------------------


def test_read_stats_converts_nanoseconds():
    stats = llm.read_stats({"prompt_eval_count": 100, "prompt_eval_duration": 2e9, "eval_count": 50, "eval_duration": 5e8})
    assert stats["input_tokens"] == 100
    assert stats["input_seconds"] == 2.0
    assert stats["output_seconds"] == 0.5


def test_context_usage_percent():
    usage = llm.context_usage({"input_tokens": 3000, "output_tokens": 1096})
    assert usage == {"used": 4096, "limit": llm.NUM_CTX, "percent": round(100 * 4096 / llm.NUM_CTX, 1)}
