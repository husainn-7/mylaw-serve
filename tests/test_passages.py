from mlserve.passages import MAX_CHARS, doc_text, load_passages, split_text


def squash(text):
    return "".join(text.split())


def test_short_text_stays_in_one_piece():
    assert split_text("A short section.") == ["A short section."]


def test_long_text_is_split_under_the_limit_without_losing_words():
    text = "\n".join(f"({i}) " + "word " * 60 for i in range(1, 30))
    pieces = split_text(text)
    assert len(pieces) > 1
    assert all(len(p) <= MAX_CHARS for p in pieces)
    assert squash("".join(pieces)) == squash(text)


def test_one_enormous_line_is_split_at_spaces():
    pieces = split_text("lorem " * 1000)
    assert all(len(p) <= MAX_CHARS for p in pieces)


def test_load_passages_keeps_section_metadata(corpus):
    passages = load_passages(corpus["chunks"])
    assert len({p["id"] for p in passages}) == len(passages)
    assert all(len(p["text"]) <= MAX_CHARS for p in passages)
    ids = [p["id"] for p in passages]
    assert "employment_act-60E-p0" in ids
    assert "employment_act-2-p1" in ids  # the long section was split into several passages


def test_doc_text_format():
    p = {"act": "Contracts Act 1950", "section": "15", "heading": "Coercion", "text": "Coercion is..."}
    assert doc_text(p) == "Contracts Act 1950, section 15: Coercion\nCoercion is..."
