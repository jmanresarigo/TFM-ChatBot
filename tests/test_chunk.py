"""Chunking (ingestion/chunk.py) on made-up text: sizes, overlap, headings."""

from chunk import MAX_WORDS, OVERLAP_WORDS, chunk_pages, is_heading


def sentences(n, word="neuron"):
    """n distinct ten-word sentences."""
    return " ".join(f"Sentence {i} talks about the {word} model in simple terms." for i in range(n))


def test_short_document_is_one_chunk():
    chunks = chunk_pages([{"label": "1", "text": sentences(5)}], "doc.pdf")
    assert len(chunks) == 1
    assert chunks[0]["id"] == "doc-0000"
    assert chunks[0]["pages"] == ["1"]
    assert chunks[0]["source"] == "doc.pdf"


def test_long_document_is_split_into_bounded_chunks_with_ids():
    pages = [{"label": str(p), "text": sentences(20)} for p in range(1, 6)]  # 1,000 words
    chunks = chunk_pages(pages, "doc.pdf")
    assert len(chunks) > 3
    assert [c["id"] for c in chunks] == [f"doc-{i:04d}" for i in range(len(chunks))]
    for c in chunks:
        assert c["words"] <= MAX_WORDS + OVERLAP_WORDS


def test_consecutive_chunks_overlap():
    pages = [{"label": "1", "text": p} for p in [sentences(15, "spike"), sentences(15, "gust")]]
    first, second = chunk_pages(pages, "doc.pdf")[:2]
    last_sentence = first["text"].split(". ")[-1]
    assert last_sentence.rstrip(".") in second["text"]


def test_every_page_is_tracked():
    pages = [{"label": str(p), "text": sentences(20)} for p in range(1, 6)]
    covered = {p for c in chunk_pages(pages, "doc.pdf") for p in c["pages"]}
    assert covered == {"1", "2", "3", "4", "5"}


def test_heading_moves_to_the_next_chunk():
    # 140 words + heading + 140 words: too much for one chunk, so it splits at the heading
    text = "\n\n".join([sentences(14), "2.4. Sparse Sensing", sentences(14)])
    chunks = chunk_pages([{"label": "1", "text": text}], "doc.pdf")
    assert not chunks[0]["text"].endswith("2.4. Sparse Sensing")
    assert chunks[1]["text"].startswith("2.4. Sparse Sensing")


def test_tiny_last_piece_is_merged_into_previous_chunk():
    pages = [{"label": "1", "text": sentences(25)}, {"label": "2", "text": "Short end."}]
    chunks = chunk_pages(pages, "doc.pdf")
    assert "Short end." in chunks[-1]["text"]
    assert chunks[-1]["words"] > 10


def test_is_heading():
    assert is_heading({"text": "3. Results and Discussion"})
    assert not is_heading({"text": "This sentence ends with a period."})
    assert not is_heading({"text": sentences(2)})
