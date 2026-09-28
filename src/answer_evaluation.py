"""
Answer evaluation.

Three things changed from the first version, all in the direction of
"the metric must be able to fail":

1. Keyword matching used `any()`. With `expected_keywords = ["IHSG"]` on a
   question that already contains the word IHSG, the model only had to
   echo the question to score a point. Now every expected keyword must be
   present, and numbers are compared numerically instead of as substrings.

2. Citation checking only tested that the `[source hal.N]` pattern
   existed. Now the cited (source, page) pair must actually appear among
   the chunks that were sent to the model, which is what catches an
   invented page number.

3. Infrastructure failures were scored as wrong answers. A row is now
   left out of the metrics only when its status is neither "ok" nor
   "empty_retrieval"; those rows are reported separately, and a run with
   too many of them is marked invalid. "empty_retrieval" deliberately
   stays in: retrieval coming back with nothing is a real miss by the
   system, not the database being down. So it is scored like any other
   row - it counts in answer_rate, citation_rate and false_refusal_rate,
   and never toward `failed_ratio` or `run_is_valid`.
"""

import json
import re

from src import config
from src.rag_chain import ask_question


# ============================================================
# DATASET
# ============================================================

def load_answer_evaluation_dataset(file_path=None):
    """
    Load the answer evaluation dataset.
    """
    path = config.EVAL_QUESTIONS_PATH if file_path is None else file_path

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# KEYWORD MATCHING
# ============================================================

NUMBER_PATTERN = re.compile(
    r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?"
)


def extract_numbers(text):
    """
    Parse Indonesian-formatted numbers into floats.

    'Rp3.400'  -> 3400.0
    '13,75%'   -> 13.75
    '9.134,70' -> 9134.70
    """
    numbers = []

    for match in NUMBER_PATTERN.finditer(str(text or "")):
        raw = match.group()

        if "." in raw and "," in raw:
            raw = raw.replace(".", "").replace(",", ".")
        elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", raw):
            raw = raw.replace(".", "")
        else:
            raw = raw.replace(",", ".")

        try:
            numbers.append(float(raw))
        except ValueError:
            continue

    return numbers


def keyword_matches(answer, keyword, tolerance=0.005):
    """
    Match one expected keyword against an answer.

    Numeric keywords are compared numerically so that "Rp 3.400,00",
    "3400" and "3.400" all count, while "1,0" no longer matches "31,08".
    """
    targets = extract_numbers(keyword)

    if not targets:
        return str(keyword).lower().strip() in str(answer or "").lower()

    candidates = extract_numbers(answer)

    return all(
        any(
            abs(candidate - target) <= max(abs(target) * tolerance, 1e-9)
            for candidate in candidates
        )
        for target in targets
    )


def matched_keywords(answer, expected_keywords):
    """
    Return the subset of expected keywords found in the answer.
    """
    return [
        keyword
        for keyword in (expected_keywords or [])
        if keyword_matches(answer, keyword)
    ]


def contains_expected_keyword(answer, expected_keywords):
    """
    True only when EVERY expected keyword is present.
    """
    if not expected_keywords:
        return False

    return len(matched_keywords(answer, expected_keywords)) == len(
        expected_keywords
    )


# ============================================================
# CITATION
# ============================================================

CITATION_PATTERN = re.compile(r"\[([^\[\]]+?)\s+hal\.(\d+)\]")


def contains_citation(answer):
    """
    Whether the answer carries at least one [source hal.N] citation.
    """
    return bool(CITATION_PATTERN.search(str(answer or "")))


def citation_accuracy(answer, documents):
    """
    Fraction of citations in the answer that point at a (source, page)
    pair actually present in the retrieved context.

    Returns None when the answer carries no citation at all.
    """
    citations = CITATION_PATTERN.findall(str(answer or ""))

    if not citations:
        return None

    valid_pairs = {
        (
            str(document.metadata.get("source", "")).strip(),
            str(document.metadata.get("page", "")).strip(),
        )
        for document in documents
    }

    correct = sum(
        1
        for source, page in citations
        if (source.strip(), page.strip()) in valid_pairs
    )

    return round(correct / len(citations), 4)


# ============================================================
# REFUSAL
# ============================================================

def _pola_dari_pesan_sistem():
    """
    The refusal sentences the system itself speaks, built from the
    constants that hold them.

    This is not tidiness, it is load-bearing. This detector decides
    `out_of_scope_refusal_rate`. If the list were written out by hand,
    every reword of a refusal sentence would drop the metric without
    anything changing in the system's behavior. That is exactly what
    happened on the first end-to-end run: the system refused correctly
    on 5 of 5 out-of-scope questions, but only 1 was counted, because
    the wording was not in the list.

    Deriving the patterns from the constants means the two can no longer
    drift apart - change the sentence and the pattern changes with it.
    """
    from src import prompts

    pesan = [
        config.REFUSAL_MESSAGE,
        prompts.PESAN_DI_LUAR_CAKUPAN,
        prompts.PESAN_TICKER_TIDAK_DIKENAL,
    ]

    return [
        re.compile(re.escape(teks.strip()), re.IGNORECASE)
        for teks in pesan
        if teks and teks.strip()
    ]


# Paraphrase patterns. The system does not always use the exact
# sentences above - the LLM may refuse in its own words.
POLA_PARAFRASA_PENOLAKAN = [
    re.compile(r"tidak\s+ada\s+di\s+dokumen", re.IGNORECASE),
    re.compile(
        r"tidak\s+(?:di)?te(?:mu|rsedia)\w*\s+(?:di|dalam)\s+"
        r"(?:dokumen|konteks)",
        re.IGNORECASE,
    ),
    re.compile(r"tidak\s+dapat\s+menjawab", re.IGNORECASE),
    re.compile(r"di\s+luar\s+cakupan", re.IGNORECASE),
    re.compile(r"tidak\s+bisa\s+memastikan\s+saham", re.IGNORECASE),
    re.compile(r"hanya\s+melayani\s+data\s+saham", re.IGNORECASE),
    re.compile(
        r"tidak\s+(?:bisa|dapat)\s+(?:saya\s+)?(?:bantu|layani)",
        re.IGNORECASE,
    ),
]

REFUSAL_PATTERNS = _pola_dari_pesan_sistem() + POLA_PARAFRASA_PENOLAKAN


def is_refusal(answer):
    """
    Whether the answer is a refusal, tolerant of paraphrase.
    """
    return any(
        pattern.search(str(answer or ""))
        for pattern in REFUSAL_PATTERNS
    )


# ============================================================
# RUNNER
# ============================================================

def evaluate_answers(dataset, ask=None, verbose=True):
    """
    Evaluate generated RAG answers.
    """
    ask = ask or ask_question

    results = []

    for item in dataset:
        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])

        if verbose:
            print(f"Menguji: {question}")

        response = ask(question)

        answer = response.get("answer")
        status = response.get("status", "ok")
        documents = response.get("documents", [])

        row = {
            "id": item["id"],
            "question": question,
            "type": item["type"],
            "status": status,
            "error": response.get("error"),
            "latency_seconds": response.get("latency_seconds"),
            "answer": answer,
            "retrieved_chunk_ids": response.get("chunk_ids", []),
            "expected_keywords": expected_keywords,
            # Empty in RAG-only mode, filled in end-to-end mode. Carried
            # so that "why was this one not answered" can be settled by
            # the intent the router picked, instead of guessed from the
            # text of the answer.
            "intent": response.get("intent"),
            "trace": response.get("trace"),
        }

        if status not in {"ok", "empty_retrieval"}:
            # Infrastructure failure: not scored, reported separately.
            row.update(
                {
                    "matched_keywords": None,
                    "keyword_match": None,
                    "citation": None,
                    "citation_accuracy": None,
                    "refusal": None,
                }
            )
            results.append(row)
            continue

        found = matched_keywords(answer, expected_keywords)

        row.update(
            {
                "matched_keywords": found,
                "keyword_match": bool(
                    expected_keywords
                    and len(found) == len(expected_keywords)
                ),
                "citation": contains_citation(answer),
                "citation_accuracy": citation_accuracy(answer, documents),
                "refusal": is_refusal(answer),
            }
        )

        results.append(row)

    return summarise_answers(results)


def summarise_answers(results):
    """
    Aggregate per-question rows into the report.
    """
    total = len(results)

    failed = [
        row
        for row in results
        if row["status"] not in {"ok", "empty_retrieval"}
    ]

    scored = [
        row
        for row in results
        if row["status"] in {"ok", "empty_retrieval"}
    ]

    in_scope = [row for row in scored if row["type"] == "in_scope"]
    out_of_scope = [row for row in scored if row["type"] == "out_of_scope"]

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    citation_scores = [
        row["citation_accuracy"]
        for row in in_scope
        if row["citation_accuracy"] is not None
    ]

    failed_ratio = len(failed) / total if total else 0.0

    # What one run cost, for the paths that report it. The claim that
    # the router saves LLM calls cannot be checked without this number.
    jejak = [row.get("trace") for row in results if row.get("trace")]

    llm_calls = {
        "router": sum(t.get("router_llm_calls", 0) for t in jejak),
        "resolution": sum(t.get("resolution_llm_calls", 0) for t in jejak),
        "answer": sum(t.get("answer_llm_calls", 0) for t in jejak),
    }
    llm_calls["total"] = sum(llm_calls.values())

    intents = {}

    for row in results:
        if row.get("intent"):
            intents[row["intent"]] = intents.get(row["intent"], 0) + 1

    return {
        "config": config.describe(),
        "total_questions": total,
        "failed_rows": len(failed),
        "failed_ratio": round(failed_ratio, 4),
        "run_is_valid": failed_ratio <= config.MAX_FAILED_ROW_RATIO,
        "in_scope_questions": len(in_scope),
        "out_of_scope_questions": len(out_of_scope),
        "answer_rate": mean(
            [1.0 if row["keyword_match"] else 0.0 for row in in_scope]
        ),
        "citation_rate": mean(
            [1.0 if row["citation"] else 0.0 for row in in_scope]
        ),
        "citation_accuracy": mean(citation_scores),
        "citation_accuracy_scored": len(citation_scores),
        "false_refusal_rate": mean(
            [1.0 if row["refusal"] else 0.0 for row in in_scope]
        ),
        "out_of_scope_refusal_rate": mean(
            [1.0 if row["refusal"] else 0.0 for row in out_of_scope]
        ),
        "average_latency_seconds": round(
            mean(
                [
                    row["latency_seconds"] or 0.0
                    for row in scored
                ]
            ),
            3,
        ),
        "llm_calls": llm_calls if jejak else None,
        "intents": intents or None,
        "details": results,
    }
