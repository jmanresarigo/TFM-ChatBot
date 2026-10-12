"""Retrieval (app/retrieval.py): Reciprocal Rank Fusion with the two searches
faked, and the embeddings client against a fake server. No database, no model,
no real API needed."""

import json

import httpx
import numpy as np
import pytest

import retrieval


def hits(*ids):
    return [{"id": i, "source": "doc.pdf", "pages": ["1"], "content": f"text {i}"} for i in ids]


def test_hybrid_search_rewards_chunks_found_by_both_searches(monkeypatch):
    monkeypatch.setattr(retrieval, "vector_search", lambda conn, q, n: hits("a", "b", "c"))
    monkeypatch.setattr(retrieval, "keyword_search", lambda conn, q, n: hits("c", "d"))
    results = retrieval.hybrid_search(None, "question", top_k=3)
    # "c" is only 3rd and 1st, but appearing in both lists beats "a" (1st in one)
    assert [r["id"] for r in results] == ["c", "a", "b"]
    assert results[0]["vector_rank"] == 3
    assert results[0]["keyword_rank"] == 1
    assert "keyword_rank" not in results[1]


def test_keyword_only_hit_counts_less_than_vector_only_hit(monkeypatch):
    # Same rank in each list: the vector hit wins, the keyword hit gets KEYWORD_WEIGHT as much
    monkeypatch.setattr(retrieval, "vector_search", lambda conn, q, n: hits("v"))
    monkeypatch.setattr(retrieval, "keyword_search", lambda conn, q, n: hits("k"))
    results = retrieval.hybrid_search(None, "q", top_k=2)
    assert [r["id"] for r in results] == ["v", "k"]
    assert results[1]["score"] == pytest.approx(retrieval.KEYWORD_WEIGHT * results[0]["score"])


def test_hybrid_search_works_when_one_search_finds_nothing(monkeypatch):
    monkeypatch.setattr(retrieval, "vector_search", lambda conn, q, n: hits("a", "b"))
    monkeypatch.setattr(retrieval, "keyword_search", lambda conn, q, n: [])
    assert [r["id"] for r in retrieval.hybrid_search(None, "q", top_k=5)] == ["a", "b"]


# --- embeddings client (fake server) -------------------------------------------------


@pytest.fixture
def fake_embeddings_api(monkeypatch):
    monkeypatch.setattr(retrieval, "EMBEDDING_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setattr(retrieval, "EMBEDDING_API_KEY", "cf-token")

    def install(handler):
        monkeypatch.setattr(retrieval, "_client", httpx.Client(transport=httpx.MockTransport(handler)))

    return install


def test_embed_query_adds_instruction_and_normalizes(fake_embeddings_api):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"embedding": [3.0, 4.0]}]})

    fake_embeddings_api(handler)
    vector = retrieval.embed_query("What is beta?")
    assert np.allclose(vector, [0.6, 0.8])  # scaled to length 1
    assert seen["url"] == "https://api.example.com/v1/embeddings"
    assert seen["auth"] == "Bearer cf-token"
    text = seen["body"]["input"][0]
    assert text.startswith("Instruct: ") and text.endswith("\nQuery:What is beta?")


def test_embed_query_retries_once_on_rate_limit(fake_embeddings_api, monkeypatch):
    monkeypatch.setattr(retrieval.time, "sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "1"})
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0]}]})

    fake_embeddings_api(handler)
    assert np.allclose(retrieval.embed_query("q"), [1.0, 0.0])
    assert len(calls) == 2


def test_embed_query_usage_limit_says_how_long_to_wait(fake_embeddings_api, monkeypatch):
    monkeypatch.setattr(retrieval.time, "sleep", lambda s: None)
    fake_embeddings_api(lambda request: httpx.Response(429, headers={"retry-after": "90"}, json={"errors": ["limit"]}))
    with pytest.raises(retrieval.RetrievalError, match="about 90 seconds") as error:
        retrieval.embed_query("q")
    assert error.value.usage_limit and error.value.retry_after == 90


def test_embed_query_usage_limit_without_a_wait_mentions_the_daily_reset(fake_embeddings_api, monkeypatch):
    monkeypatch.setattr(retrieval.time, "sleep", lambda s: None)
    fake_embeddings_api(lambda request: httpx.Response(429, json={"errors": ["limit"]}))
    with pytest.raises(retrieval.RetrievalError, match="00:00 UTC") as error:
        retrieval.embed_query("q")
    assert error.value.usage_limit and error.value.retry_after is None


def test_embed_query_rejected_key(fake_embeddings_api):
    fake_embeddings_api(lambda request: httpx.Response(401, json={"errors": ["bad token"]}))
    with pytest.raises(retrieval.RetrievalError, match="EMBEDDING_API_KEY"):
        retrieval.embed_query("q")


def test_embed_query_needs_a_base_url(monkeypatch):
    monkeypatch.setattr(retrieval, "EMBEDDING_BASE_URL", "")
    with pytest.raises(retrieval.RetrievalError, match="EMBEDDING_BASE_URL"):
        retrieval.embed_query("q")
