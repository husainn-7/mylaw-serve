import json

import pytest

LONG = " ".join(f"({i}) 'term{i}' means a thing that is defined in this Act for the purposes of interpretation." for i in range(1, 60))

SECTIONS = [
    ("employment_act", "Employment Act 1955", "60E", "Annual leave",
     "60E. (1) An employee shall be entitled to paid annual leave of eight days for every twelve months of continuous service."),
    ("employment_act", "Employment Act 1955", "37", "Maternity leave",
     "37. (1) Every female employee shall be entitled to maternity leave of ninety days and a maternity allowance for each confinement."),
    ("employment_act", "Employment Act 1955", "2", "Interpretation", "2. In this Act:\n" + "\n".join(LONG.split(" (")[:60])),
    ("pdpa", "Personal Data Protection Act 2010", "8", "Disclosure Principle",
     "8. No personal data shall be disclosed without the consent of the data subject to any third party."),
    ("pdpa", "Personal Data Protection Act 2010", "129", "Transfer of personal data to places outside Malaysia",
     "129. A data user shall not transfer any personal data to a place outside Malaysia unless specified by the Minister."),
    ("contracts_act", "Contracts Act 1950", "15", "Coercion",
     "15. Coercion is the committing or threatening to commit any act forbidden by the Penal Code to make someone sign an agreement."),
    ("contracts_act", "Contracts Act 1950", "74", "Compensation for breach of contract",
     "74. When a contract has been broken the party who suffers by the breach is entitled to compensation for loss or damage."),
]

QUESTIONS = [
    ("q001", "easy", "How many days of annual leave does an employee get?", ["employment_act-60E"]),
    ("q002", "easy", "How long is maternity leave for a female employee?", ["employment_act-37"]),
    ("q003", "easy", "Can personal data be disclosed without the consent of the data subject?", ["pdpa-8"]),
    ("h001", "hard", "Can a data user transfer personal data to a place outside Malaysia?", ["pdpa-129"]),
    ("h002", "hard", "Someone is threatening to commit an act forbidden by the Penal Code unless I sign. Is that coercion?", ["contracts_act-15"]),
    ("h003", "hard", "What compensation can I claim for loss or damage when a contract has been broken?", ["contracts_act-74"]),
]


@pytest.fixture
def corpus(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    with open(chunks, "w", encoding="utf-8") as f:
        for stem, act, number, heading, text in SECTIONS:
            f.write(json.dumps({"id": f"{stem}-{number}", "act": act, "section": number, "heading": heading, "page": 1, "text": text}) + "\n")
    questions = tmp_path / "questions.jsonl"
    with open(questions, "w", encoding="utf-8") as f:
        for qid, split, q, gold in QUESTIONS:
            f.write(json.dumps({"id": qid, "split": split, "question": q, "gold": gold, "answerable": True}) + "\n")
        f.write(json.dumps({"id": "q900", "split": "easy", "question": "What is the minimum wage?", "gold": [], "answerable": False}) + "\n")
    return {"chunks": str(chunks), "questions": str(questions), "dir": tmp_path}
