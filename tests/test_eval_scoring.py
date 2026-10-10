"""Scoring logic of the retrieval evaluation (evals/run_retrieval_eval.py)."""

from run_retrieval_eval import first_hit_rank, normalize

QUESTION = {
    "source": "Review - SNN Models.pdf",
    "pages": ["18"],
    "also_accept": [{"source": "thesis.pdf", "pages": ["21", "23"]}],
}


def chunk(source, *pages):
    return {"source": source, "pages": list(pages)}


def test_normalize_ignores_spaces_dashes_underscores_and_case():
    assert normalize("Review - SNN Models.pdf") == normalize("review_-_snn_models.PDF")


def test_first_hit_rank_finds_main_answer():
    results = [chunk("other.pdf", "18"), chunk("Review_-_SNN_Models.pdf", "17", "18")]
    assert first_hit_rank(QUESTION, results) == 2


def test_first_hit_rank_accepts_alternative_sources():
    assert first_hit_rank(QUESTION, [chunk("thesis.pdf", "23")]) == 1


def test_right_document_wrong_page_is_a_miss():
    assert first_hit_rank(QUESTION, [chunk("Review - SNN Models.pdf", "19")]) is None
