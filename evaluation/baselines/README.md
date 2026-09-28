# Frozen evaluation snapshots

| Directory | Meaning | Retrieval fingerprint |
|---|---|---|
| `historical_4pdf/` | A: historical 4 PDFs / 308 chunks / 25 questions, restored from the commit recorded in its manifest | `d2768255a0ff` |
| `current_legacy_25/` | B: 5 PDFs / 489 chunks / original 25 questions, saved before the 27 September 2026 improvement pass | `27195dcc002f` |
| `full_corpus_33/` | C: 5 PDFs / 489 chunks / 33 questions, including eight document-5 questions; saved before the retrieval experiment | `eeae64b5f63f` |

These are reference snapshots, not evaluator inputs. Active inputs remain
`evaluation/dataset/eval_questions.json` and `gold_chunks.json`. Scripts write
new measurements to `evaluation/results/`; do not overwrite these snapshots.

Different fingerprints prohibit attributing aggregate differences to a
retrieval change. Compare the C snapshot with current C results to measure
the candidate/final retrieval. The adjacent-chunk candidate was rejected;
`scripts/compare_retrieval.py` reproduces it without changing application code.

Router results use a separate dataset and must not be treated as retrieval or
answer results. The older answer reports lack a fingerprint field: their
question/gold snapshots preserve their benchmark identity. New answer runs
record both the retrieval fingerprint and a SHA-256 covering all question and
expected-keyword data.

For manual gold evidence, ranks, methodology and limitations, see
[`docs/EVALUATION_AUDIT_2026-09-27.md`](../../docs/EVALUATION_AUDIT_2026-09-27.md).
