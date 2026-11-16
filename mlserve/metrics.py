TOP_SECTIONS = 10


def section_id(passage_id):
    return passage_id.rsplit("-p", 1)[0]


def collapse(passage_ids, k=TOP_SECTIONS):
    """Several passages can come from one section. Keep the first (best) per section."""
    seen, ranked = set(), []
    for pid in passage_ids:
        sid = section_id(pid)
        if sid not in seen:
            seen.add(sid)
            ranked.append(sid)
        if len(ranked) == k:
            break
    return ranked


def first_gold_rank(ranked, gold):
    for i, sid in enumerate(ranked, start=1):
        if sid in gold:
            return i
    return None


def metrics_for(ranks):
    """ranks: list of int (1-based) or None for each question."""
    n = len(ranks)
    if n == 0:
        return None

    def hit(k):
        return sum(1 for r in ranks if r and r <= k) / n

    return {
        "n": n,
        "hit@1": round(hit(1), 3),
        "hit@3": round(hit(3), 3),
        "hit@5": round(hit(5), 3),
        "hit@10": round(hit(10), 3),
        "mrr@10": round(sum(1 / r for r in ranks if r) / n, 3),
    }
