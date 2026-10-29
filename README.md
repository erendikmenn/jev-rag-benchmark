# jev-rag-benchmark

A small, auditable Turkish RAG experiment: does reranking improve passage selection,
and how do four answer models perform on identical retrieved evidence?

**Current evaluation: 29 October 2026 — 1,044 XQuAD-TR test questions, 240 passages.**
This is a short-answer benchmark on a small corpus, not a general-purpose RAG leaderboard.

## What is held constant?

- The candidate pool is the historical hybrid BM25 + BGE-M3 baseline's top 20, frozen once.
- Jev, Cohere, Clef Flash and Luna Decisions receive identical candidate **texts and order** for every question.
- All rerankers return five passages. The baseline retains the original hybrid order.
- Clef Flash is reported separately for 20-question batch and one-passage-per-request
  scoring; Jev 1.13 is the original batch run, Jev 1.14 and Luna Decisions have both modes, and Cohere uses its native rerank endpoint.
- Gemini, DeepSeek, Haiku and Luna receive the **same Jev 1.13 top-five context and exact prompt**, capped at
  6,000 context characters. They must return a short answer span with a source citation.
- All use nominal medium reasoning and a 4,096 output-token limit. Gemini/DeepSeek use
  temperature 0 and seed 20260919. Temperature is unsupported for Luna/Haiku; only Luna
  supports the seed. Equal effort labels do not imply equal reasoning compute.
- Generation errors and truncated outputs receive zero EM/F1; they are not dropped.

The published run contains 1,044 unique, matched questions and **zero candidate mismatches**.
The corpus SHA-256 matches the historical corpus. The frozen pool, per-question outputs,
prompt hashes and model identities are committed under [`results/fair-2026-10-29/`](results/fair-2026-10-29/).

## Passage selection

Generation is excluded from this comparison. All metrics use the final **five** passages.

| Method | Recall@5 ↑ | Gold in top-5 | nDCG@5 ↑ | MRR@5 ↑ | Reranking cost |
|---|---:|---:|---:|---:|---:|
| Hybrid baseline | 97.605% | 1,019 / 1,044 | 94.314% | 93.194% | $0.000000 |
| Jev 1.13 — batch | 99.425% | 1,038 / 1,044 | 98.151% | 97.712% | $0.410866 |
| Jev 1.14 — batch | 99.425% | 1,038 / 1,044 | 98.067% | 97.601% | $0.410866 |
| Jev 1.14 — pointwise | 99.234% | 1,036 / 1,044 | 97.728% | 97.213% | $0.643918 |
| Cohere Rerank 3.5 — native API | 99.425% | 1,038 / 1,044 | 98.626% | 98.348% | $1.044000 |
| Clef Flash — batch, DeepInfra FP8 | 94.444% | 986 / 1,044 | 92.567% | 91.935% | $0.168440 |
| Clef Flash — pointwise, DeepInfra FP8 | 99.521% | 1,039 / 1,044 | 98.270% | 97.840% | $0.195843 |
| Luna Decisions — batch | 98.659% | 1,030 / 1,044 | 97.503% | 97.106% | $1.041619 |
| Luna Decisions — pointwise | 99.234% | 1,036 / 1,044 | 98.268% | 97.933% | $1.077098 |


Jev 1.14 is pinned to `typesafe/jev-1.14-20261014` through TypeSafe on OpenRouter.
Its batch run finds **1,038 / 1,044** gold passages in the top five, versus
**1,038** for Jev 1.13 at the same batch prompt. Its pointwise run finds **1,036**,
versus **1,039** for Clef Flash with the same one-passage relevance question.

The candidate pool contains the annotated gold passage for only 1,039 questions; that is
the recall ceiling for every method here. Pointwise methods issue 20 calls per query.
Compare Jev versions within batch, or Jev 1.14 and Flash within pointwise, to avoid conflating
model and strategy changes. Small differences on this near-ceiling, 240-passage dataset do
not establish general superiority. This is a reused evaluation set, not an independent
model-selection test.

![Reranker recall, nDCG at five, and cost](assets/benchmark/reranker-comparison.png)

![Questions missing the gold passage in the top five](assets/benchmark/retrieval-errors.png)


Luna Decisions improves from 1,030 batch hits to 1,036 pointwise hits. Its pointwise
nDCG@5 (98.268%) is nearly equal to Flash pointwise (98.270%), but it loses three more gold
passages from the top five and costs about 5.5 times as much. Cohere has higher nDCG and
two more hits at a slightly lower cost, though its native scoring strategy differs. No general model winner is implied by these small-set observations.

### Luna Decisions + Haiku/Luna extension

All four new configurations completed the same 1,044 questions: **24,012 completed API responses**.
Luna Decisions used the unchanged batch and pointwise relevance prompts, candidate order
and tie-break rule. The first ten-query technical pilot is retained in the full results.
The batch prompt retains the target-reference ambiguity described in the Clef diagnostics;
compare models within the same scoring strategy.

Haiku 5.5 and GPT-6 Luna generated answers using **the original Jev 1.13 top-five contexts**,
not the new Luna Decisions rankings. Every prompt hash matches the original generation run.
These are separate retrieval and generation comparisons, not end-to-end measurements of
all possible model combinations. Terra was not run.

OpenRouter routing was pinned to OpenAI / Anthropic with fallbacks disabled. Endpoint metadata
advertised dated versions, but generator responses returned undated aliases; the snapshot
is provenance, not proof of an immutable resolved version. Both generators use nominal
medium reasoning and 4,096 output tokens. Neither supports temperature; Haiku does not support
seed. Luna uses seed 20260919, standard service tier and explicit cache mode without breakpoints.
No search or tools were used. Equal reasoning labels do not imply equal compute across models.

There were **16 failed transport attempts**, retained in the raw journal;
only transport failures were retried. Empty/truncated generator answers remain zero-scored.
Retained model costs are shown in the tables. Haiku returned two empty content-filter
refusals without usage telemetry: its $0.397719 includes a $0.008 reserve; observed billed
usage totals $0.389719. Token totals cover only the 1,042 responses with usage.
The 16 unsuccessful transport attempts were HTTP 504 timeouts, reserved at $0.0015 each.
Including uncertain-call reserves, the new extension accounts for **$2.728969**. Conservative cumulative project
accounting is **$9.294570**, within the approved **$11** cap.
These are telemetry/reserve figures, not a reconciled invoice. New model differences are
descriptive; no new confidence interval or controlled speed claim is made.

[Raw responses, rankings, answers, protocol, endpoint snapshot and costs](results/luna-haiku-2026-10-29/)
are committed. Recompute without API calls:

```bash
uv run python scripts/analyze_luna_haiku.py
```

### Jev 1.14 extension

Both modes ran on all 1,044 frozen queries: 1,044 batch calls and 20,880 pointwise calls.
The model version and TypeSafe provider were pinned, with provider fallbacks disabled.
Batch retained the original 1.13 prompt (including its known target-reference ambiguity)
to isolate the version change. Pointwise used the exact Flash prompt. No pilot observations
were reused as full-run results.

| Fixed-strategy comparison | nDCG@5 change | Recall@5 change |
|---|---:|---:|
| Jev 1.14 batch minus Jev 1.13 batch | -0.084 percentage points | +0.000 percentage points |
| Jev 1.14 pointwise minus Flash pointwise | -0.541 percentage points | -0.287 percentage points |

The 63-call preflight cost $0.002970. Full-run successful calls cost
$1.054783. Including preflight and any recorded/reserved failed attempts,
this extension accounts for **$1.057754**. The approved total project
cap was increased to **$7**; conservative cumulative accounting is
**$6.565602**.

All 21,924 required inferences eventually returned valid scores. The 140 unsuccessful
attempts were HTTP 429 rate-limit rejections. Initial OpenRouter rate
limits stopped the run after 4,092 successful calls; adding 25-request/second pacing and
shared cooldown resumed missing work without redoing successful calls. All attempts are
journaled; retries were transport-related, never selected on answer quality. These runtime
changes preclude a controlled latency comparison. Billing is telemetry/reserve accounting,
not an independently reconciled invoice.

[Raw calls, both-mode rankings, protocol and costs](results/jev114-2026-10-29/) are committed.
Recompute without API calls:

```bash
uv run python scripts/analyze_jev114.py
```

### Clef Flash extension

Clef Flash was evaluated on the same frozen 1,044 questions, with exactly the Jev batch
state and relevance questions, selecting five passages by descending probability and
breaking ties by original candidate order. OpenRouter routing was pinned to **DeepInfra
(FP8)** with provider fallbacks disabled. This is a provider-specific result, not a claim
about every Clef Flash deployment.

Clef Flash achieved **94.444% Recall@5** and **92.567% nDCG@5**,
below both Jev and the hybrid baseline. Its retained-run cost was **$0.168440**,
59.0% below Jev, at the observed $0.019/million input-token rate.
Its lower cost came with worse retrieval quality in this **20-question batch configuration**.
Follow-up diagnostics below show strong configuration sensitivity; this result should not be
read as an intrinsic Turkish-language capability ranking. No answer generation was rerun for Clef Flash.

A longest-character-input preflight correctly read a sentinel at the end of the state;
this is a truncation smoke check, not exhaustive proof that every passage was attended to.
All measured requests stayed below the provider's 24,576-token window
(maximum observed: 10,815). Ten pilot queries passed technical checks and were
retained in the full run. All 1,044 requests completed without errors or retries.
The extension cost **$0.168647 including preflight**, under its separate $0.60 cap.
The original $6 run accounting below is unchanged.

[Raw Clef Flash responses and provenance](results/clef-flash-2026-10-29/) include all
20 candidate probabilities, provider-reported usage, request hashes and paired hit counts.
The dry-run command below makes no API calls; adding `--run` starts paid calls:

```bash
uv run python scripts/run_clef_flash.py
```

### Clef Flash failure diagnosis

Of 58 misses, five had no annotated gold passage in the frozen top-20 pool; 53 were
ranking failures. Clef Flash lost 41 baseline hits and recovered eight baseline misses.
Some failures were obvious: for the Marlee Matlin sign-language question, the answer-bearing
passage fell from candidate rank 1 to rank 12 while an unrelated Tibetan Buddhism passage
ranked first.

We selected six failures with gold present plus six successful controls (selection IDs/seed
committed), then varied one aspect of the request at a time:

| Diagnostic | Original failures recovered (of 6) | Controls retained (of 6) |
|---|---:|---:|
| Repeat original batch request | 0 | 6 |
| Explicit target path in relevance question | 2 | 6 |
| Reverse question order, same candidate state | 4 | 6 |
| Reverse candidate order, same question template | 1 | 1 |
| Original batch on Makora BF16 instead of DeepInfra FP8 | 0 | 6 |
| Score every passage separately, then sort all 20 scores | 6 | 6 |

The batch prompt starts with `candidate.text` although the state contains `candidates[]`,
then appends `Evaluate only candidates[i]`. Explicit paths remove this ambiguity, but that
change alone did not resolve four of the six selected failures. Candidate/question-order
sensitivity and isolated-passage recovery are stronger evidence of an interaction between
this model and the batch formulation. The BF16 check does not support blaming FP8 alone;
it cannot isolate every provider/runtime difference.

Pointwise scoring put the gold passage first for 11 of these 12 queries and third for one.
This is **a diagnostic case-control sample selected partly from failures, not a 100% recall
benchmark**. The full 1,044-query result above remains the original batch result. No prompt
was tuned and then relabeled as an independent test. A fresh, prespecified held-out comparison
of batch and pointwise scoring is needed before changing the model recommendation.

The 312 diagnostic calls cost **$0.013130**, bringing Clef Flash usage at that stage, including the earlier
preflight/full run, to **$0.181777**, within the $0.60 limit. On these 12 queries, 240 pointwise
calls cost $0.002241 versus $0.001937 for 12 repeated batch calls; request count and latency
also matter, and these costs are not a full-run estimate.

[All 58 misses, exact diagnostic requests/responses and per-query ranks](results/clef-flash-diagnostics-2026-10-29/)
are committed. Recompute the diagnostic table without API calls:

```bash
uv run python scripts/analyze_clef_diagnostics.py
```

### Full-set Flash pointwise rerun and Jev 1.13/Cohere audit

The pointwise rerun scored all 20 passages separately for each of the 1,044 questions:
**20,880 successful calls**. No diagnostic scores were reused. The same relevance question,
provider and tie-breaking rule were fixed before this run. Five requests timed out and were
retried for transport failure only; every attempt is retained. Successful-call cost was
**$0.195843**, plus a conservative $0.002500 reserve for uncertain timed-out calls,
giving **$0.198343** run accounting.

The 12 previously probed questions scored 12/12; the other 1,032 scored 1,027/1,032
(99.516% Recall@5). This split is descriptive: revisiting the same evaluation dataset after
diagnosis is not equivalent to a new independent test set.

This audit concerns Jev **1.13**, not the later 1.14 extension.
We checked **all 1,044 original Jev/Cohere records**: all 20 candidate scores were present,
and score-to-document mapping plus top-five selection matched the saved outputs.
There was no indexing or sorting bug. Cohere receives a query and document list through
its native rerank API; the Jev relevance prompt is not sent to Cohere.

Live audit: 16 seeded random gold-present queries, excluding the earlier 12 diagnostics,
plus each model's one gold-present miss (18 distinct queries). In the 16 random cases,
Jev retained all gold hits under explicit paths, reversed questions, reversed candidates
and pointwise scoring. It is **not invariant**, however: reversing candidates changed its
top-five set in 18/18 audit cases and gold rank in 3/18. Even repeated original requests
changed its top-five set in 12/18, so not every difference can be attributed to order.

For Jev's targeted miss, gold rank was 7 originally, 8 on repeat, 6 with explicit paths,
4 with reversed candidates and 1 pointwise. The shared batch formulation can therefore
hurt Jev too, although the observed effect on gold retrieval was much smaller here.

Cohere's repeat and reversed-order runs preserved both the top-five sets and gold ranks
in all 18 cases. Its targeted miss remained rank 7. This is evidence against a mapping/order
problem in this sample, not a proof that Cohere is universally order-invariant.

[Pointwise requests, ranks and accounting](results/clef-pointwise-2026-10-29/) and
[the Jev/Cohere audit](results/reranker-audit-2026-10-29/) are available for inspection.
The audit cost $0.075348. Cumulative Clef Flash accounting is $0.380120, below its $0.60 cap.
At that stage, total project accounting including earlier uncertainty reserves and this audit
was $5.507848, below the original $6 cap. The later Jev 1.14 extension uses the approved $7 cap. These amounts are telemetry/reserve accounting, not reconciled invoices.

Recompute without API calls:

```bash
uv run python scripts/analyze_clef_pointwise.py
uv run python scripts/analyze_reranker_audit.py
```

## Short-answer generation on identical Jev contexts

Citations are removed before text scoring. Token F1 measures word overlap with extractive
references; exact match measures normalized text equality. Neither is a human factuality score.

| Model (1,044 questions each) | Token F1 ↑ | Exact match ↑ | F1 ≥ 0.5 | Empty/truncated | Generation cost |
|---|---:|---:|---:|---:|---:|
| Gemini 3.8 Flash | 77.058% | 57.567% | 83.716% | 0 | $2.539025 |
| DeepSeek V4.1 Flash | 70.634% | 44.732% | 78.927% | 6 | $0.459019 |
| GPT-6 Luna | 59.402% | 29.215% | 66.762% | 0 | $0.188533 |
| Haiku 5.5 | 71.185% | 49.138% | 78.927% | 2 | $0.397719 |

Haiku scores 71.185% F1, 0.551 percentage points above DeepSeek and 5.873 below Gemini.
Its $0.397719 accounted cost (including $0.008 for two missing-usage refusals) is 13.4%
below DeepSeek. Luna is cheapest at $0.188533, but its 59.402% F1 and 29.215% exact match
are lowest under this short-answer protocol. These lexical scores do not establish factuality.
Add $0.410866 for the original Jev 1.13 reranking stage if calculating Jev-plus-generation
cost; preprocessing and earlier operational overhead remain excluded.

For the original Gemini/DeepSeek pair only, Gemini's paired mean-F1 advantage is **6.424 percentage points**; the question-level paired
bootstrap 95% interval is **4.806–7.987 points** (2,000 resamples, seed 20261029).
Questions sharing passages are not independent; this interval does not model passage/article
clustering, repeated runs or provider variability. DeepSeek has lower observed API cost.
The six failed DeepSeek responses are included with zero scores, not silently retried away.

![Short-answer F1, exact match, and generation cost](assets/benchmark/generator-tradeoffs.png)

Observed generation medians were 3.641 s (Gemini) and 1.676 s (DeepSeek), but **this is not a
controlled speed ranking**: concurrency changed from four to sixteen during the run.
Provider routes and time of measurement can also affect latency.

## Budget and operational record

Current approved cumulative cap: **$11**. After all extensions, conservative accounting is
**$9.294570**; see the extension records above. The following is the original run history.

The original run cap was $6. Recorded API charges across pilots, partial attempts and the full
run were **$4.552380**. An additional **$0.50 uncertainty reserve** was retained for interrupted
in-flight requests: conservative budget accounting was **$5.052380**, below the cap.
This is telemetry-based accounting, not a reconciled invoice. Method costs above describe
retained full-run observations and exclude pilot/restart overhead and transport reserves.
Haiku is the explicit exception: its table cost includes $0.008 reserved for two responses
that lack usage telemetry.

The first generation pilot rejected disabled reasoning. The successful pilot and full run
used medium reasoning. Execution was restarted to increase concurrency and later after an
empty generation. Completed observations were retained; incomplete queries were rerun.
After the error-policy correction, empty/truncated outputs were retained as failures.
No retries were selected on answer F1. These operational changes limit claims of a perfectly
preregistered experiment; a clean repeated run is a useful future validation.

No new embedding calls were needed: this run reused the frozen historical candidate pool.
Offline pool construction is excluded from reported online costs and timings.

## Audit and regenerate — no API calls

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen --extra dev
uv run pytest
uv run ruff check .
uv run --with matplotlib==3.10.8 python scripts/publish_results.py
```

The publishing script validates coverage, candidate order, prompt reconstruction and answer
scores against the frozen artifacts before regenerating all six SVG/PNG figures. The figures
retain the monochrome presentation and use zero-based axes. Raw observations are available in
[`results.jsonl`](results/fair-2026-10-29/results.jsonl); aggregate values and provenance are in
[`summary.json`](results/fair-2026-10-29/summary.json) and
[`manifest.json`](results/fair-2026-10-29/manifest.json).

## Running a new experiment

```bash
uv run jev-rag data prepare
# Free fixture smoke; does not measure actual model quality.
uv run jev-rag benchmark smoke --dataset scifact --branches A,D --fixture-jev --skip-generation
```

Paid runs require `OPENROUTER_API_KEY` and an explicit positive budget in a copied config;
never commit credentials. Default budget is zero. Review current provider pricing first.
`retrieval.frozen_pool_path` selects the immutable snapshot path. Use a different path for a
new dataset, sample or retrieval configuration. A mismatch fails instead of overwriting the
pool. Use the same path/config for all shards of one experiment.

`merge-results` and the report generator reject missing/incompatible protocol IDs, duplicate
observations, differing candidates and unequal branch coverage. Legacy result files cannot
be fed into the new reporting path. Replay rejects changed source/configuration on resume.
The general-purpose API clients have estimated-cost checks, not provider-side prepaid caps;
for strict spending limits also configure a provider key limit. The operational $6 run used
an additional shared reservation gate and conservative reserve.

## Historical results and corrections

The September 20 reports remain under `reports/generated/` and `docs/benchmark-2026-09-20.md`
for auditability and are explicitly marked **legacy**. They are not current leaderboard data.

1. Cohere candidate order differed on 124 questions, with different sets on 25. The old
   claim that all rerankers saw exactly the same candidates was incorrect.
2. Legacy `nDCG@10`/`MRR@10` were computed over only five returned contexts. Their proper
   interpretation is truncated-at-five scoring, not a full top-ten ranking evaluation.
3. Sentence-form answers were compared with short extractive references. For example,
   “Panthers savunması ligde sadece 308 sayı bırakmıştır” scored F1 0.25 against `308` despite
   answering the question. Legacy “wrong answer” and “cost per successful answer” claims
   must not be interpreted as factual correctness or used for model recommendations.
4. The old DeepSeek quality recommendation is superseded. Under the new short-answer
   protocol Gemini has higher lexical quality; DeepSeek has lower generation cost.
5. Older strategy ablations and self-verification claims were not rerun. They are excluded
   from current recommendations, rather than combined with the new full-run scores.

Old and new F1 values must not be read as model improvement over time: the protocol changed
and provider/model execution was repeated. The new results establish a comparison only
within this fixed dataset, prompt and model configuration.

## Scope and licenses

XQuAD-TR is primarily extractive QA with one annotated gold passage per question and only
240 corpus passages. Results do not establish robustness on large enterprise corpora,
unanswerable questions, adversarial documents, conversational tasks or other languages.
Citation IDs alone do not prove source support. Human-reviewed factuality evaluation and
article-level uncertainty estimates remain future work.

Source code: MIT. XQuAD: CC BY-SA 4.0; frozen corpus excerpts retain that license and originate
from [Google DeepMind XQuAD](https://github.com/google-deepmind/xquad). SciFact annotations:
CC BY 4.0; abstracts: ODC-By 1.0. Upstream code attribution is in [`NOTICE`](NOTICE).
