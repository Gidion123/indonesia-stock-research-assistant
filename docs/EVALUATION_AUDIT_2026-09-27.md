# Five-document evaluation audit — 27 September 2026

## Scope and status

This pass inspected preprocessing, ingestion/vector storage, both retrievers,
RAG generation, routing, entity resolution, market-data/EODHD handling,
live price/compare, deterministic P/L, evaluation scripts/datasets, and their
tests. Production deployment and secrets were not changed.

Retrieval, full regression including real LLM tests, both live answer modes,
and router/entity evaluation are complete. External evaluations ran after
explicit user consent to transmit benchmark questions and research excerpts
to DeepSeek. Both answer runs have zero failed rows and `run_is_valid=true`.

## Benchmark versions

| Version | Corpus | Questions | Retrieval fingerprint | Status |
|---|---|---:|---|---|
| A: historical | 4 PDFs / 308 chunks | 25 (20 in scope) | `d2768255a0ff` | Archived results from Git HEAD; not rerun |
| B: current corpus, legacy questions | 5 PDFs / 489 chunks | 25 (20 in scope) | `27195dcc002f` | Saved before this pass |
| C: current full-corpus coverage | 5 PDFs / 489 chunks | 33 (28 in scope) | `eeae64b5f63f` | Retrieval rerun during this pass |

Snapshots live in [`evaluation/baselines/`](../evaluation/baselines/).
A, B and C have different fingerprints: their aggregate scores must not be
presented as before/after improvements. C preserves all 25 B questions,
keywords and gold annotations, and adds eight dedicated document-5 questions.
“Full-corpus coverage” means every PDF is represented, not exhaustive coverage
of every company or claim. These are development benchmarks, not an unseen
held-out test set.

## Manual audit of document-5 questions and gold

The repository parser was used to read the PDF, retained pages and actual
chunks. Page numbers below are one-based PDF pages. Evidence is what the
research document says; financial claims were not independently verified.
All eight new questions use strict gold matching. All 11 derived gold chunks
were inspected, including surrounding pages for continuation passages.
No annotations or keywords were loosened after observing retrieval scores.

| ID / category | Evidence and expected keywords | Audited gold chunk IDs / pages |
|---|---|---|
| `dc_01` / investment thesis | Industrial-estate data-center land premium 30–50%; gross margin 65–70%. Keywords `30%`, `50%`, `65%`. | `cb56ea0c4598`, p1 |
| `dc_02` / operational fact | POWR contracted data-center capacity 274 MW in H1 2026, up 47 MW from end-2025; 12% of industrial-customer consumption. | `b5f2d13c15ee`, p33 |
| `dc_03` / catalyst | Gas normalization from April 2026; projected contracts 303 MW in 2026 and 385 MW in 2027. These are forecasts in the report. | `267d9a5c0a10`, p33; `8a3e6fbd05c8`, p9 |
| `dc_04` / valuation | POWR P/E 12.3x, PBV 1.2x (`12,3`, `1,2`). | `d02ae09bd238`, p16; `e968c25f8d42`, p34 |
| `dc_05` / financial fact | DMAS H1 2026 revenue Rp1.80tn (+194% YoY), net income Rp1.19tn (+175%). Keywords `1,80`, `1,19`, `194%`, `175%`. | `2b21d280ec23`, p35 |
| `dc_06` / risk | PGEO drilling risk and PPA/PLN contract exposure; COD delays also discussed. Keywords `pengeboran`, `PPA`, `PLN`. | `1118eacbe6bd`, p37 |
| `dc_07` / valuation risk | DCII valuation warning: P/E 422x and PBV above 40x despite operational quality. | `2780cdeba491`, p30; `ddcd7213c907`, p39 |
| `dc_08` / strategic partnership | PGEO partner Masdar, 15% ownership, access to green financing. | `e43d7852c396`, p36 |

The p9 contract table is a continuation whose surrounding page identifies
POWR. The p34 POWR valuation continues its p33 section. The p35 DMAS
financial passage continues the section starting on p34. PGEO's section
starts on p35–36 and continues to p37. Some standalone chunks therefore
lack their company ticker, although their gold attribution is correct in
document context. This matters to the UI's ticker-priority ranking.

The full rebuilt gold set has **80 question–chunk associations**, 27 strict
questions, one relaxed question, zero questions without gold, average 2.86
gold chunks/question and maximum 12. The original 69 associations remain
unchanged; document 5 adds 11. Gold is keyword-derived and is not necessarily
an exhaustive annotation of all semantically useful passages.

## Root cause: `ihsg_02`

Question: “Apa faktor utama yang memengaruhi IHSG pada 2026?”

Both gold annotations are valid in `Analisis IHSG 2026 & Saham Unggulan.pdf`:

- `ef73a1de1dec`, p1, **rank 19**: MSCI/FTSE removal sentiment and rupiah pressure.
- `e72a2a356c9c`, p6, **rank 79**: MSCI rebalancing and rupiah scenario factors.

The actual first three results are `aca1c70bb161` (p5 scenario outlook),
`892e04401aa4` (p1 bullish target/macro discussion), and `e4d2774f14f1`
(p15 conclusion). The top eight omit both gold passages. Full top-eight
texts, distances and gold ranks are saved in
[`retrieval_diagnosis.json`](../evaluation/results/retrieval_diagnosis.json).

The evidence survived parsing; this is not lost-document ingestion. Dense
similarity favors broad IHSG outlook content for this broad question. The
500-character chunks also separate the topical introduction from the causal
list. A diagnostic paraphrase, “Apa penyebab tekanan terhadap IHSG selama
2026?”, moves the first gold to rank 7. This demonstrates wording sensitivity;
the benchmark question was **not** replaced by that more favorable wording.
The evidence supports ranking sensitivity with context fragmentation, not a
claim that chunking alone explains the failure.

## Root cause: `equity_03`

Question: “Bagaimana kondisi dan risiko saham BUMI?”

All four gold passages in `Indonesian Equity Trading Research.pdf` remain
valid under the existing explicit `gold_keywords=["stripping ratio"]` rule:

| Gold ID | Page | Similarity rank | Evidence |
|---|---:|---:|---|
| `03b2bc1f0847` | 19 | 50 | BUMI invalidation threshold: stripping ratio >9.0 |
| `52d6dfc384a8` | 22 | 71 | Stripping ratio >9.0 and Rp172 stop loss; fullest combined answer |
| `a402b237110b` | 15 | 181 | BUMI bull/base operating assumptions, stripping ratio 7.5 |
| `042ff8eda32c` | 9 | 247 | Operating table: stripping ratio 7.5 versus 8.1 |

The top eight mix general BUMI valuation and other companies' risk text.
The first three are `d01b1264e80f` (IHSG p12), `84747ddfdb29`
(Equity p21), and `a9983b76f681` (Equity p9). The third is immediately
after the p9 gold table. See the diagnostic JSON for the authoritative full
IDs/texts and ranking.

The broad “condition and risk” query favors general narrative over specific
operating thresholds and stop-loss tables. A diagnostic invalidation-parameter
query moves the first gold to rank 10 but still misses top eight. Query
sensitivity and separated table context both contribute. The answer criterion
still requires **both** `stripping ratio` and `Rp172`; a gold hit on a
stripping-ratio-only chunk does not establish full answer coverage.

## Preprocessing false positive

`with a` → `witha` was a bug. The original match was in the reference title
“DCI is developing the Largest Local Data Center in Indonesia with a” on
PDF5 p46. A one-letter English article/pronoun must not be treated as a broken
word suffix. The minimal guard preserves `a` and `i`, while retaining the
existing Indonesian repair such as `perbanka n` → `perbankan`.

A regression test covers both legitimate English text and the retained
Indonesian repair. Corpus parsing/chunk generation was rerun with the old
and new rules: **all 489 retained chunk texts and metadata are identical**.
The affected bibliography page is excluded. The repair count changes 1 → 0;
PDF/page/chunk counts and embedding dimension remain unchanged. No database
reingestion or production reindex is needed for this correction.

## Measured retrieval experiment — rejected

One dependency-free experiment keeps four similarity seeds and adds an
adjacent same-page chunk per seed, returning at most eight chunks. It uses
neither expected answer keywords nor gold labels when selecting candidates.
Reproduce with `python -m scripts.compare_retrieval`. It changes no app code.

All rows below use benchmark C, fingerprint `eeae64b5f63f`:

| Retriever | Hit@8 | Recall@8 | Precision@8 | MRR (top 8) |
|---|---:|---:|---:|---:|
| Similarity baseline | 0.8571 | 0.5693 | 0.1473 | 0.4712 |
| Neighbor candidate | 0.7500 | 0.4473 | 0.1295 | 0.4477 |
| Final similarity, unchanged | 0.8571 | 0.5693 | 0.1473 | 0.4712 |
| Existing UI session retriever, empty owner | 0.7500 | 0.4432 | 0.1250 | 0.5381 |

The candidate gains `equity_03` but loses `ihsg_01`, `ihsg_05`, `dc_04`,
and `dc_08`. **Rejected: runtime retrieval remains unchanged.** Similarity
still misses `ihsg_02`, `equity_03`, `dc_01` (first gold rank 28), and
`dc_05` (rank 9) at k=8. The new-document subset gets 6/8 hits.

The UI row measures an existing retriever, not a proposed replacement.
Its hard entity priority can favor ticker-bearing passages over continuation
paragraphs. CLI RAG/e2e use plain similarity by default; CLI e2e additionally
exercises router/resolver but does not simulate session uploads or memory.
These retrieval modes must be labeled separately. Existing session-isolation
tests passed in the regression suite; this benchmark does not replace them.

## Validation and current answer results

- Gold rebuild: 5 PDFs, 117 total pages, 105 retained, 12 bibliography pages
  removed, 5 partially cut, 202 citation markers removed, 489 chunks, 384 dimensions.
- Full pytest invocation with cached embeddings, local PostgreSQL, and real
  LLM calls: **383 passed, 4 skipped in 57.91s**. The skips are three opt-in
  live market tests and one optional BIPI/additional-corpus test. An earlier
  offline run passed 379 tests with eight skips while consent was pending.
- Retrieval evaluation: complete; C metrics and fingerprint saved.
- `evaluate_answers` and `evaluate_answers --end-to-end`: completed on all
  33 questions; matching fingerprint `eeae64b5f63f` and question/keyword digest
  `6e15da6066a034adbad7434cdc11a4e2158709a1a3b14ebc8c1bbdffa87be39c`.
- `evaluate_router --llm --semantik`: rerun on its separate 25-question dataset;
  intent 25/25, entity 14/14, zero false refusals, five LLM calls, 24 pattern
  routes and one LLM route. This is separate from the answer dataset.
- Future answer reports now include retrieval fingerprint, a SHA-256 of the
  complete question/keyword dataset, and `retrieval_mode=similarity`. Two
  regression tests ensure keyword-only changes alter the scoring digest.

| Metric | C: RAG only | C: CLI end-to-end |
|---|---:|---:|
| Answer Rate (all expected keywords) | 0.8571 (24/28) | 0.8571 (24/28) |
| Citation Rate | 1.0000 | 1.0000 |
| Citation source/page accuracy | 1.0000 (28 scored) | 1.0000 (28 scored) |
| False Refusal Rate | 0.0000 | 0.0000 |
| Out-of-Scope Refusal | 0.8000 (4/5) | 1.0000 (5/5) |
| Failed API/evaluation rows | 0 | 0 |

In both modes, `ihsg_02` misses MSCI/rupiah and `equity_03` misses stripping
ratio/Rp172. `dc_01` includes the 30–50% premium but misses the 65% margin;
`dc_05` includes revenue/growth but misses net income/growth. These are the
same four top-eight retrieval misses. The legacy subset still scores 18/20
for both retrieval hits and answer keywords; document 5 scores 6/8. This
does not mean every statement in those 24 passing answers is factually valid.

The end-to-end run made 44 LLM calls (router 14, resolver 1, answers 29),
zero Hugging Face requests and zero Yahoo requests; embeddings came from
local cache. Observed intents were RAG 29, OUT_OF_SCOPE 3, LIVE_PRICE 1.
The out-of-scope gold-price query was rejected before a market request.
The router's separate 25/25 benchmark must not be confused with this intent
distribution. RAG-only OOS refusal bypasses routing and does not describe
production refusal behavior. Citation matching verifies source/page pairs,
not claim-level factual grounding. Latencies saved in the raw files were
measured while evaluation processes shared the machine, not in a controlled
production load test, so no performance improvement is claimed.

Reproduce the final runs with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q --tb=short
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.build_gold_chunks
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.evaluate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.evaluate_answers
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.evaluate_answers --end-to-end
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.evaluate_router --llm --semantik
```

Use the configured environment and existing API keys for live evaluation;
it sends questions/retrieved excerpts to the selected LLM provider and incurs
API usage. Do not paste credentials into documentation or terminal arguments.
Inspect `run_is_valid` and failed rows before citing any future run.

Recommendation: stop this pass here.
A later, separate small experiment could soften hard entity priority while
preserving SQL session/expiry filtering. It needs its own measured gate;
there is no justification here for a new retrieval service or architecture.

Evidence: [command summaries](../evaluation/results/validation_2026-09-27.txt),
[machine-readable validation](../evaluation/results/validation_2026-09-27.json),
and [Git / pass-only diff statistics](EVALUATION_DIFF_2026-09-27.txt).
