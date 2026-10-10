"""Reciprocal Rank Fusion (app/retrieval.py), with the two searches faked:
no database and no embedding model needed."""

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


def test_hybrid_search_works_when_one_search_finds_nothing(monkeypatch):
    monkeypatch.setattr(retrieval, "vector_search", lambda conn, q, n: hits("a", "b"))
    monkeypatch.setattr(retrieval, "keyword_search", lambda conn, q, n: [])
    assert [r["id"] for r in retrieval.hybrid_search(None, "q", top_k=5)] == ["a", "b"]
