"""
Gold chunk annotation.

Measuring retrieval at source level cannot fail. With four documents and
k=8 at least one chunk from the right document is almost always
returned, which is why the first evaluation reported HitRate, Recall and
MRR of 1.0000.

To measure anything useful the benchmark has to know which CHUNK holds
the answer. This module derives that from the expected keywords, writes
it to a JSON file that can be corrected by hand, and reports how
reliable the derivation was.
"""

import json

from src import config
from src.preprocessing import load_and_split_documents


GOLD_MODE_STRICT = "strict"      # one chunk contains every keyword
GOLD_MODE_RELAXED = "relaxed"    # no single chunk has all, union used
GOLD_MODE_NONE = "none"          # no chunk matched at all

# Above this many gold chunks, Recall@k stops discriminating: almost any
# retrieval will hit one of them. Surfaced as a warning rather than
# hidden, because it is a property of the dataset, not of the system.
SATURATION_THRESHOLD = 12


def _matches(chunk, keyword):
    return keyword.lower().strip() in chunk.page_content.lower()


def derive_gold_for_item(chunks, item, keywords):
    """
    Find the gold chunks for one benchmark item.

    Candidates are the chunks that come from one of the item's expected
    sources. An item that lists no expected sources at all is not
    filtered: every chunk in the knowledge base stays a candidate.

    Strict mode then requires a single chunk to contain every expected
    keyword. If nothing matches that way, relaxed mode falls back to the
    union of the per-keyword matches.
    """
    expected_sources = set(item.get("expected_sources") or [])

    candidates = [
        chunk
        for chunk in chunks
        if not expected_sources
        or chunk.metadata.get("source") in expected_sources
    ]

    if not keywords:
        return {
            "gold_chunk_ids": [],
            "gold_mode": GOLD_MODE_NONE,
            "keywords": [],
        }

    strict = [
        chunk
        for chunk in candidates
        if all(_matches(chunk, keyword) for keyword in keywords)
    ]

    if strict:
        return {
            "gold_chunk_ids": sorted(
                chunk.metadata["chunk_id"] for chunk in strict
            ),
            "gold_mode": GOLD_MODE_STRICT,
            "keywords": list(keywords),
        }

    relaxed = [
        chunk
        for chunk in candidates
        if any(_matches(chunk, keyword) for keyword in keywords)
    ]

    if relaxed:
        return {
            "gold_chunk_ids": sorted(
                chunk.metadata["chunk_id"] for chunk in relaxed
            ),
            "gold_mode": GOLD_MODE_RELAXED,
            "keywords": list(keywords),
        }

    return {
        "gold_chunk_ids": [],
        "gold_mode": GOLD_MODE_NONE,
        "keywords": list(keywords),
    }


def build_gold_chunks(dataset, keywords_by_id, chunks=None):
    """
    Derive gold chunks for every in-scope item in the dataset.
    """
    if chunks is None:
        chunks = load_and_split_documents()

    gold = {}

    for item in dataset:
        if item.get("type") != "in_scope":
            continue

        gold[item["id"]] = derive_gold_for_item(
            chunks,
            item,
            keywords_by_id.get(item["id"], []),
        )

    return gold


def save_gold_chunks(gold, file_path=None):
    """
    Persist the gold annotation so it can be reviewed and corrected.
    """
    path = config.GOLD_CHUNKS_PATH if file_path is None else file_path
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {"version": 1, "items": gold},
            file,
            ensure_ascii=False,
            indent=2,
        )

    return path


def load_gold_chunks(file_path=None):
    """
    Read the gold annotation. Returns an empty dict when absent.
    """
    path = config.GOLD_CHUNKS_PATH if file_path is None else file_path

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict) and "items" in payload:
        return payload["items"]

    return payload


def audit_gold_chunks(gold):
    """
    Report how trustworthy the derived annotation is.
    """
    total = len(gold)

    by_mode = {
        GOLD_MODE_STRICT: 0,
        GOLD_MODE_RELAXED: 0,
        GOLD_MODE_NONE: 0,
    }

    sizes = []
    saturated = []

    for question_id, entry in gold.items():
        by_mode[entry["gold_mode"]] = by_mode.get(entry["gold_mode"], 0) + 1

        size = len(entry["gold_chunk_ids"])
        sizes.append(size)

        if size > SATURATION_THRESHOLD:
            saturated.append((question_id, size))

    return {
        "total_items": total,
        "strict": by_mode[GOLD_MODE_STRICT],
        "relaxed": by_mode[GOLD_MODE_RELAXED],
        "without_gold": by_mode[GOLD_MODE_NONE],
        "average_gold_chunks": (
            round(sum(sizes) / len(sizes), 2) if sizes else 0.0
        ),
        "max_gold_chunks": max(sizes) if sizes else 0,
        "saturated_items": sorted(
            saturated,
            key=lambda pair: pair[1],
            reverse=True,
        ),
    }
