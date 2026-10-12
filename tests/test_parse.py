"""PDF cleaning helpers (ingestion/parse.py). No PDF needed: the functions
work on strings and on pages as {"blocks": [[line, ...], ...]} dicts."""

from parse import (
    FIRST_REFERENCE,
    REFERENCES_HEADING,
    clean_label,
    find_repeated_lines,
    fix_characters,
    is_content_block,
    is_toc_page,
    join_lines,
    line_key,
)


def page(*blocks):
    return {"blocks": [list(b) for b in blocks]}


# --- page labels ---------------------------------------------------------------


def test_clean_label_keeps_plain_labels():
    assert clean_label("12", 20) == "12"
    assert clean_label("xii", 3) == "xii"


def test_clean_label_decodes_hex_utf16():
    # Some PDFs store "8" as "<FEFF0038>" (UTF-16 with a byte-order mark)
    assert clean_label("<FEFF0038>", 20) == "8"
    assert clean_label("<FEFF00310032>", 20) == "12"


def test_clean_label_falls_back_to_page_position():
    assert clean_label("", 7) == "7"
    assert clean_label("   ", 7) == "7"


# --- characters ----------------------------------------------------------------


def test_fix_characters_expands_ligatures():
    assert fix_characters("eﬃcient ﬁeld ﬂow") == "efficient field flow"


def test_fix_characters_joins_spacing_accents():
    assert fix_characters("Morzy´nski") == "Morzyński"


def test_fix_characters_removes_control_characters():
    # NUL bytes made Postgres refuse to store a chunk
    assert fix_characters("spike\x00 train\x07") == "spike train"


def test_line_key_ignores_digits_case_and_broken_accents():
    assert line_key("  Page 12 of 80 ") == line_key("page 13 of 80")
    assert line_key("Morzy´nski et al.") == line_key("Morzyński et al.")


# --- noise detection -------------------------------------------------------------


def test_repeated_header_is_detected_but_equation_numbers_are_not():
    pages = [
        page(["Journal of Made-Up Studies"], [f"Body text number {i}.", "(2.5)"])
        for i in range(10)
    ]
    repeated = find_repeated_lines(pages)
    assert line_key("Journal of Made-Up Studies") in repeated
    assert line_key("(2.5)") not in repeated  # no letters: content, not header


def test_too_few_pages_never_have_headers():
    assert find_repeated_lines([page(["Same line"]), page(["Same line"])]) == set()


def test_toc_page_by_heading_or_dot_leaders():
    assert is_toc_page(page(["Contents"], ["1 Introduction 1"]))
    dotted = page(["1 Introduction . . . . . . 1", "2 Method . . . . . . 5", "3 Results 9"])
    assert is_toc_page(dotted)
    assert not is_toc_page(page(["A normal paragraph of text."]))


def test_content_blocks_need_letters():
    assert is_content_block("The NACA 0012 airfoil")
    assert not is_content_block("0.0 0.5 1.0")  # axis ticks
    assert not is_content_block("12")  # page number
    assert not is_content_block("xii")  # roman page number


def test_references_heading_variants():
    for heading in ["References", "BIBLIOGRAPHY", "R E F E R E N C E S", "LITERATURE CITED"]:
        assert REFERENCES_HEADING.match(heading), heading
    assert not REFERENCES_HEADING.match("References to earlier work show that")


def test_unheaded_reference_list_starts_at_first_entry():
    # Physical Review papers print no heading before "[1] A. R. Jones, ..."
    assert FIRST_REFERENCE.match("[1] A. R. Jones, O. Cetiner, and M. J. Smith, Physics")
    assert FIRST_REFERENCE.match("[1] I.-A. Lungu and Y. Hu, Conversion")
    assert not FIRST_REFERENCE.match("[1] shows that the lift peak depends on circulation")
    assert not FIRST_REFERENCE.match("[2] A. R. Jones, a later entry")


# --- joining lines -----------------------------------------------------------------


def test_join_lines_undoes_end_of_line_hyphens():
    assert join_lines(["the bifurca-", "tions of the", "  flow"]) == "the bifurcations of the flow"


def test_join_lines_keeps_hyphen_before_capital():
    # "Navier-" + "Stokes" is a real hyphenated name, not a line break
    assert join_lines(["Navier-", "Stokes equations"]) == "Navier- Stokes equations"
