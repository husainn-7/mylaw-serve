import json
from pathlib import Path

MAX_CHARS = 1500


def split_text(text, max_chars=MAX_CHARS):
    pieces, cur = [], ""
    for unit in text.split("\n"):
        while len(unit) > max_chars:
            cut = unit.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(unit[:cut])
            unit = unit[cut:].strip()
        if cur and len(cur) + len(unit) + 1 > max_chars:
            pieces.append(cur)
            cur = unit
        else:
            cur = f"{cur}\n{unit}" if cur else unit
    if cur:
        pieces.append(cur)
    return pieces


def load_passages(chunks_path="data/chunks.jsonl"):
    """Sections -> passages of at most 1,500 characters, same rule as MyLaw-RAG."""
    rows = [json.loads(line) for line in open(Path(chunks_path), encoding="utf-8")]
    passages = []
    for r in rows:
        for k, piece in enumerate(split_text(r["text"])):
            passages.append({
                "id": f"{r['id']}-p{k}",
                "act": r["act"],
                "section": r["section"],
                "heading": r["heading"],
                "text": piece,
            })
    return passages


def doc_text(p):
    """The text that is embedded and reranked: section label + heading + passage."""
    return f"{p['act']}, section {p['section']}: {p['heading']}\n{p['text']}"
