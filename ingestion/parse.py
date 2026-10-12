"""Extract and clean text from PDFs, page by page.

Pipeline for each PDF:
    1. Extract   - text blocks (roughly paragraphs) from every page, plus the
                   printed page label ("4", "ii") so citations match the paper.
    2. Fix chars - ligatures (ﬁ -> fi) and broken accents (´n -> ń).
    3. Drop      - repeated headers/footers, lines without letters,
                   table-of-contents pages, and the references section.
    4. Join      - lines inside a block become one paragraph, undoing
                   hyphenation and the "one word per line" problem.

Known, accepted limitations: equations and tables come out fragmented, and
citation numbers can stay glued to words ("vortices24").

Usage:
    python ingestion/parse.py data/thesis.pdf     # one file
    python ingestion/parse.py                     # every PDF in data/
"""

import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

# --- Settings (tune these by looking at the output, not by guessing) --------

# A line is treated as a header/footer if it appears on at least this
# fraction of pages (digits are ignored, so "Page 3" and "Page 4" match).
# 0.3 also catches headers that alternate between odd and even pages.
REPEATED_LINE_FRACTION = 0.3
# A page is treated as a table of contents / list of figures if at least this
# fraction of its lines contain dot leaders (". . . .").
TOC_LINE_FRACTION = 0.3
# Headings that start the references section, and headings that end it.
# IEEE journals letter-space the heading ("R E F E R E N C E S"); Annual
# Reviews calls it "Literature cited".
REFERENCES_HEADING = re.compile(
    r"^(" + r"\s?".join("references") + r"|bibliography|literature cited)$", re.IGNORECASE
)
# Some journals (e.g. Physical Review) print no heading: the list just starts
# with "[1] A. R. Jones, ...". Only a first entry with initials counts, so a
# sentence that happens to start with "[1]" doesn't end the document.
FIRST_REFERENCE = re.compile(r"^\[1\]\s+(?:[A-Z]\.\s?)+(?:-?[A-Z]\.\s?)*[A-Z][a-z]")
AFTER_REFERENCES_HEADING = re.compile(
    r"^(appendix|annex)\b|^[A-Z]\.\s+[A-Z][A-Z ]+$", re.IGNORECASE
)

LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}
# Spacing accents that some PDFs place *before* the letter: "´n" means "ń".
SPACING_ACCENTS = {"´": "́", "¨": "̈", "˜": "̃", "ˆ": "̂", "ˇ": "̌"}
BROKEN_ACCENT = re.compile("([" + "".join(SPACING_ACCENTS) + r"])\s?([A-Za-z])")
# Invisible control characters some PDFs contain, e.g. NUL (0x00), which
# Postgres refuses to store in text columns. Tabs and newlines are kept.
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
DOT_LEADERS = re.compile(r"(\. ){3,}|\.{4,}")
LIST_HEADING = re.compile(
    r"^(contents|table of contents|list of (figures|tables))$", re.IGNORECASE
)
ROMAN_PAGE_NUMBER = re.compile(r"^[ivxlc]+$", re.IGNORECASE)


# Some PDFs store page labels still encoded as hex UTF-16, e.g. "<FEFF0038>"
# for "8". PyMuPDF passes these through as-is, so they are decoded here.
ENCODED_LABEL = re.compile(r"<FEFF((?:[0-9A-Fa-f]{4})+)>")


def clean_label(label: str, number: int) -> str:
    """Printed page label, decoded if needed; the page position as fallback."""
    match = ENCODED_LABEL.fullmatch(label)
    if match:
        label = bytes.fromhex(match.group(1)).decode("utf-16-be")
    return label.strip() or str(number)


# --- 1. Extract --------------------------------------------------------------


def extract_pages(pdf_path: Path) -> list[dict]:
    """Return one entry per page with its blocks of text lines.

    Each entry: {"page": 20, "label": "4", "blocks": [["line", "line"], ...]}
    "page" is the position in the PDF; "label" is the number printed on it.
    """
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for number, page in enumerate(doc, start=1):
            blocks = []
            # Each block is (x0, y0, x1, y1, text, block_no, block_type);
            # block_type 0 is text, 1 is an image.
            for *_, text, _, block_type in page.get_text("blocks", sort=False):
                if block_type == 0:
                    blocks.append(text.splitlines())
            pages.append(
                {
                    "page": number,
                    "label": clean_label(page.get_label(), number),
                    "blocks": blocks,
                }
            )
    return pages


# --- 2. Fix characters -------------------------------------------------------


def fix_characters(line: str) -> str:
    line = CONTROL_CHARS.sub("", line)
    for ligature, letters in LIGATURES.items():
        line = line.replace(ligature, letters)
    # "´n" -> "n" + combining accent, then NFC merges them into "ń"
    line = BROKEN_ACCENT.sub(lambda m: m.group(2) + SPACING_ACCENTS[m.group(1)], line)
    return unicodedata.normalize("NFC", line)


# --- 3. Drop noise -----------------------------------------------------------


def line_key(line: str) -> str:
    """Normalize a line for header/footer detection (digits don't matter).

    Characters are fixed first, so the key is the same before and after
    cleaning ("Morzy´nski" and "Morzyński" give the same key)."""
    return re.sub(r"\d+", "#", fix_characters(line).strip().lower())


def find_repeated_lines(pages: list[dict]) -> set[str]:
    """Lines that appear on many pages are headers/footers, not content."""
    if len(pages) < 3:
        return set()
    counts = Counter()
    for page in pages:
        keys = {line_key(line) for block in page["blocks"] for line in block}
        counts.update(k for k in keys if k)
    min_pages = max(3, REPEATED_LINE_FRACTION * len(pages))
    # Only lines with letters: number-only lines such as equation numbers
    # "(2.5)" repeat across pages too, but they are content. Page numbers
    # are removed later by the block filter instead.
    return {
        key
        for key, count in counts.items()
        if count >= min_pages and any(ch.isalpha() for ch in key)
    }


def is_toc_page(page: dict) -> bool:
    """Table of contents or list of figures/tables: no answers, many keywords."""
    lines = [line for block in page["blocks"] for line in block if line.strip()]
    if not lines:
        return False
    if LIST_HEADING.match(lines[0].strip()):
        return True
    dotted = sum(1 for line in lines if DOT_LEADERS.search(line))
    return dotted / len(lines) >= TOC_LINE_FRACTION


def is_content_block(paragraph: str) -> bool:
    """False for blocks with no letters (axis ticks, page numbers, table
    values) and for roman page numbers ("xii").

    This is checked per block, not per line: inside a paragraph, a line that
    is only a number ("0012" in "NACA 0012 airfoil") is real content."""
    has_letters = any(ch.isalpha() for ch in paragraph)
    return has_letters and not ROMAN_PAGE_NUMBER.match(paragraph)


# --- 4. Join lines into paragraphs ---------------------------------------------


def join_lines(lines: list[str]) -> str:
    """Join a block's lines into one paragraph, undoing end-of-line hyphens.

    "bifurca-" + "tions" -> "bifurcations". This also merges a few real
    hyphenated words split across lines ("mean-" + "field" -> "meanfield").
    """
    text = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if text.endswith("-") and line[0].islower():
            text = text[:-1] + line
        else:
            text = f"{text} {line}" if text else line
    return re.sub(r"\s+", " ", text)


# --- Putting it together --------------------------------------------------------


def parse_pdf(pdf_path: Path) -> tuple[list[dict], dict]:
    """Return cleaned pages [{"page", "label", "text"}] and cleaning stats."""
    pages = extract_pages(pdf_path)
    raw_text = "".join(line for p in pages for b in p["blocks"] for line in b)
    stats = {
        "pages": len(pages),
        "chars_before": len(raw_text),
        "ligatures_before": sum(raw_text.count(lig) for lig in LIGATURES),
        "header_lines_removed": 0,
        "toc_pages_removed": [],
        "references_removed_from_page": None,
    }

    repeated = find_repeated_lines(pages)
    in_references = False
    cleaned = []

    for page in pages:
        if is_toc_page(page):
            stats["toc_pages_removed"].append(page["label"])
            continue

        paragraphs = []
        for block in page["blocks"]:
            kept = []
            for line in block:
                line = fix_characters(line)
                stripped = line.strip()
                if REFERENCES_HEADING.match(stripped) or (
                    not in_references and FIRST_REFERENCE.match(stripped)
                ):
                    in_references = True
                    stats["references_removed_from_page"] = page["label"]
                elif in_references and AFTER_REFERENCES_HEADING.match(stripped):
                    in_references = False
                if in_references:
                    continue
                if line_key(line) in repeated:
                    stats["header_lines_removed"] += 1
                    continue
                kept.append(line)
            paragraph = join_lines(kept)
            if is_content_block(paragraph):
                paragraphs.append(paragraph)

        text = "\n\n".join(paragraphs)
        if text:
            cleaned.append({"page": page["page"], "label": page["label"], "text": text})

    clean_text = "".join(p["text"] for p in cleaned)
    stats["chars_after"] = len(clean_text)
    stats["ligatures_after"] = sum(clean_text.count(lig) for lig in LIGATURES)
    return cleaned, stats


def save_pages(pages: list[dict], out_file: Path) -> None:
    """Write a readable dump so the result can be inspected by hand."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for p in pages:
            f.write(f"\n===== PAGE {p['page']} (printed: {p['label']}) =====\n")
            f.write(p["text"] + "\n")


def main() -> None:
    if len(sys.argv) > 2:
        print("Usage: python ingestion/parse.py [data/yourfile.pdf]")
        sys.exit(1)
    pdfs = (
        [Path(sys.argv[1])]
        if len(sys.argv) == 2
        else sorted(Path("data").glob("*.pdf"))
    )
    if not pdfs:
        print("No PDFs found in data/")
        sys.exit(1)

    for pdf_path in pdfs:
        pages, stats = parse_pdf(pdf_path)
        out_file = Path("data/parsed") / f"{pdf_path.stem}.txt"
        save_pages(pages, out_file)

        print(f"\n{pdf_path.name}")
        print(f"  Pages:                 {stats['pages']} ({len(pages)} kept)")
        print(
            f"  Characters:            {stats['chars_before']} -> {stats['chars_after']}"
        )
        print(
            f"  Ligatures:             {stats['ligatures_before']} -> {stats['ligatures_after']}"
        )
        print(f"  Header/footer lines:   {stats['header_lines_removed']} removed")
        print(f"  TOC/list pages:        {stats['toc_pages_removed'] or 'none'}")
        print(
            f"  References removed from printed page: "
            f"{stats['references_removed_from_page'] or 'not found'}"
        )
        print(f"  Saved to:              {out_file}")


if __name__ == "__main__":
    main()
