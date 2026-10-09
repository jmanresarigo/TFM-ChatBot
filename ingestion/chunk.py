"""Cut cleaned PDF text into chunks: the pieces that will be embedded and searched.

How it works:
    1. Take the cleaned pages from parse.py. Each page is a list of paragraphs
       and keeps its printed page label ("15").
    2. Paragraphs longer than MAX_WORDS are split at sentence boundaries.
    3. Paragraphs are packed into a chunk until adding the next one would go
       over MAX_WORDS. Chunks never cut a paragraph in half unless it is too long.
    4. Each new chunk starts with the last sentence(s) of the previous one
       (OVERLAP_WORDS), so an idea that straddles a boundary isn't lost.
    5. A heading is never left at the end of a chunk: it is moved to the start
       of the next one, together with the text it introduces. A chunk that
       starts a new section gets no overlap, since the previous section's
       last sentences are unrelated to it.
    6. Every chunk records its source file and the page labels it came from,
       which is what citations will use later.

Words are used as a stand-in for tokens (1 word is roughly 1.3 tokens), which
avoids depending on a specific model's tokenizer at this stage.

Usage:
    python ingestion/chunk.py data/thesis.pdf    # one file
    python ingestion/chunk.py                    # every PDF in data/
"""

import json
import random
import re
import statistics
import sys
from pathlib import Path

from parse import parse_pdf

# --- Settings (we will tune these later using the evals) ---------------------

MAX_WORDS = 250  # upper size of a chunk (~330 tokens)
OVERLAP_WORDS = 40  # roughly how much of the previous chunk is repeated
MIN_WORDS = 40  # a final chunk smaller than this is merged into the previous one
HEADING_MAX_WORDS = 12  # a heading is a short paragraph...

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
# ...that does not end like a sentence ("2.4. Sparse and Event-Based Sensing")
ENDS_LIKE_SENTENCE = re.compile(r"[.!?:;,)\]]$")


def word_count(text: str) -> int:
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_END.split(text) if s.strip()]


def split_long_paragraph(paragraph: str) -> list[str]:
    """Split a paragraph over MAX_WORDS into pieces at sentence boundaries."""
    pieces, current = [], []
    for sentence in split_sentences(paragraph):
        # A single sentence longer than MAX_WORDS (rare, e.g. a garbled
        # table) is cut by words as a last resort.
        words = sentence.split()
        while len(words) > MAX_WORDS:
            pieces.append(" ".join(words[:MAX_WORDS]))
            words = words[MAX_WORDS:]
        sentence = " ".join(words)
        if current and word_count(" ".join(current + [sentence])) > MAX_WORDS:
            pieces.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def to_units(pages: list[dict]) -> list[dict]:
    """Flatten pages into paragraph-sized units: {"text", "page"}."""
    units = []
    for page in pages:
        for paragraph in page["text"].split("\n\n"):
            if word_count(paragraph) > MAX_WORDS:
                pieces = split_long_paragraph(paragraph)
            else:
                pieces = [paragraph]
            units.extend({"text": p, "page": page["label"]} for p in pieces)
    return units


def is_heading(unit: dict) -> bool:
    text = unit["text"].strip()
    return word_count(text) <= HEADING_MAX_WORDS and not ENDS_LIKE_SENTENCE.search(text)


def split_trailing_headings(units: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separate headings at the end of a chunk from the content before them.

    [text, text, "2.4. Sparse ..."] -> ([text, text], ["2.4. Sparse ..."])
    If the chunk is only headings, nothing is moved.
    """
    end = len(units)
    while end > 0 and is_heading(units[end - 1]):
        end -= 1
    if end == 0:
        return units, []
    return units[:end], units[end:]


def overlap_from(units: list[dict]) -> list[dict]:
    """The last sentences of a chunk, up to about OVERLAP_WORDS words."""
    last = units[-1]
    taken = []
    for sentence in reversed(split_sentences(last["text"])):
        if taken and word_count(" ".join([sentence] + taken)) > OVERLAP_WORDS:
            break
        taken.insert(0, sentence)
    text = " ".join(taken)
    # Don't overlap if it would repeat almost a whole chunk
    if word_count(text) > OVERLAP_WORDS * 2:
        return []
    return [{"text": text, "page": last["page"]}]


def make_chunk(units: list[dict], source: str, index: int) -> dict:
    text = "\n\n".join(u["text"] for u in units)
    pages = list(dict.fromkeys(u["page"] for u in units))  # unique, in order
    return {
        "id": f"{Path(source).stem}-{index:04d}",
        "source": source,
        "pages": pages,
        "words": word_count(text),
        "text": text,
    }


def chunk_pages(pages: list[dict], source: str) -> list[dict]:
    chunks, current = [], []
    for unit in to_units(pages):
        current_words = sum(word_count(u["text"]) for u in current)
        if current and current_words + word_count(unit["text"]) > MAX_WORDS:
            content, headings = split_trailing_headings(current)
            chunks.append(make_chunk(content, source, len(chunks)))
            # New section: start with its heading. Same section: start with
            # the overlap from the previous chunk.
            current = headings if headings else overlap_from(content)
        current.append(unit)

    if current:
        last = make_chunk(current, source, len(chunks))
        if chunks and last["words"] < MIN_WORDS:
            # Too small to stand alone: append it to the previous chunk
            merged = chunks[-1]["text"] + "\n\n" + last["text"]
            chunks[-1]["text"] = merged
            chunks[-1]["words"] = word_count(merged)
            chunks[-1]["pages"] = list(
                dict.fromkeys(chunks[-1]["pages"] + last["pages"])
            )
        else:
            chunks.append(last)
    return chunks


def save_chunks(chunks: list[dict], out_file: Path) -> None:
    """One JSON object per line (JSONL): easy to read back and to inspect."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks)


def main() -> None:
    if len(sys.argv) > 2:
        print("Usage: python ingestion/chunk.py [data/yourfile.pdf]")
        sys.exit(1)
    if len(sys.argv) == 2:
        pdfs = [Path(sys.argv[1])]
    else:
        pdfs = sorted(Path("data").glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in data/")
        sys.exit(1)

    for pdf_path in pdfs:
        pages, _ = parse_pdf(pdf_path)
        chunks = chunk_pages(pages, pdf_path.name)
        out_file = Path("data/chunks") / f"{pdf_path.stem}.jsonl"
        save_chunks(chunks, out_file)

        sizes = [c["words"] for c in chunks]
        print(f"\n{pdf_path.name}")
        print(f"  Chunks:      {len(chunks)}")
        print(
            f"  Words/chunk: min {min(sizes)}, "
            f"median {statistics.median(sizes):.0f}, max {max(sizes)}"
        )
        print(f"  Saved to:    {out_file}")

        # Show two random chunks: reading them is the real quality check
        for chunk in random.Random(0).sample(chunks, k=min(2, len(chunks))):
            print(
                f"\n  --- {chunk['id']} | pages {chunk['pages']} | {chunk['words']} words"
            )
            print("  " + chunk["text"][:600].replace("\n\n", "\n  ") + " ...")


if __name__ == "__main__":
    main()
