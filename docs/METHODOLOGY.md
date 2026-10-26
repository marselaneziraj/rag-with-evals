# Evaluation methodology

This document explains *why* each metric exists, what it catches, what it
cannot catch, and how to answer the questions a reviewer will ask. If you only
read one file in this repository before defending the work, read this one.

The implementations live in `src/ragevals/evaluation/metrics.py`; every
function there is pure, deterministic and unit-tested in `tests/test_metrics.py`.

---

## 1. The premise: one number cannot tell you what to fix

A RAG system is a pipeline, and a pipeline fails in component-shaped ways. A
single "accuracy" score collapses every failure into one scalar and tells you
nothing about where to spend the next week. The metric set here is chosen so
that each metric isolates one component:

| Question | Metric |
|---|---|
| Did retrieval find the evidence at all? | **context recall** |
| Did retrieval put the evidence where the model would use it? | **context precision@k** |
| Did the generator invent anything? | **faithfulness** |
| Did the generator actually answer? | **answer correctness** |
| Does the system know when to stay silent? | **refusal accuracy**, over-refusal rate |
| Can you audit any individual answer? | **citation validity / coverage** |
| Can you ship it? | **p95 latency**, **cost per query** |

The diagnostic that matters most is the *pair*: high context recall with low
answer correctness means the generator is wasting good evidence, and you fix
the prompt or the model. Low context recall with high faithfulness means the
system is honestly grounded in the wrong passages, and you fix the retriever.
Neither conclusion is available from a single score.

---

## 2. Faithfulness

**Definition.** Of the claims the answer makes, what share does the retrieved
context support?

    faithfulness = supported_claims / total_claims

**Implementation.** The answer is split into sentences (one sentence = one
claim, citations stripped). A claim counts as supported when its content words
have at least 55% token recall against some retrieved chunk, or against the
concatenated context.

**Why it is the most important metric.** It is the only one in the set that
needs no reference answer. That means you can compute it on live production
traffic, not just on a gold set — which makes it the natural candidate for a
monitoring alarm. Everything else here is an offline metric.

**What it is deliberately blind to.** Correctness. An answer can be perfectly
faithful to a retrieved passage that does not answer the question, and it will
score 1.0. That is not a bug: separating *grounding* from *correctness* is the
point, because the two have different fixes. If you conflate them you cannot
tell a retrieval problem from a hallucination problem.

**Known weakness.** Lexical support cannot distinguish a faithful paraphrase
from an unfaithful one that happens to reuse the context's vocabulary, and it
will mark a correct inference ("therefore the deadline has passed") as
unsupported because the words are not in the source. Sentence-level claims are
also coarser than the proposition-level decomposition RAGAS performs with an
LLM: a compound sentence counts as supported only if all of it is, which biases
the score *downwards*. When a judge is configured, faithfulness can be scored
against the rubric in `configs/judge_rubric.yaml` instead.

**Expect it to be near 1.0 with an extractive generator** — the offline backend
copies sentences verbatim, so it cannot be unfaithful. A faithfulness of 1.00
in the shipped results is a statement about the backend, not a victory. The
metric only becomes informative with a real generative model.

---

## 3. Context precision@k

**Definition.** The rank-weighted share of retrieved chunks that are relevant —
average precision over the retrieved list:

    context_precision = (1/R) * sum over relevant positions i of precision@i

where R is the number of relevant chunks retrieved.

**Why rank weighting, and not plain precision.** Two configurations can each
retrieve 2 relevant chunks out of 5 and be far apart in quality. The one that
puts them at ranks 1–2 feeds the generator a prompt whose first words are the
answer; the one that puts them at ranks 4–5 buries the answer under three
irrelevant passages that dilute attention and consume context budget. Plain
precision scores both 0.4. Average precision scores them 1.00 and 0.45. The
metric must see what the generator sees.

**How relevance is decided.** Two modes, and the difference matters:

* **Annotated** (preferred): the gold item lists the `(source, page)` pairs that
  contain the answer, and a retrieved chunk is relevant if it comes from one of
  them. This is a measurement.
* **Lexical fallback**: a chunk is relevant if it covers at least 30% of the
  reference answer's content words. This is a proxy, and it is noisy in both
  directions — a chunk sharing boilerplate vocabulary scores as relevant, and a
  chunk that answers in different words does not.

Annotating `gold_sources` is the single highest-leverage hour of manual work in
the whole project. `python -m ragevals.cli propose` pre-fills the annotation
from the chunk each candidate question was drawn from, so curation is a reading
task rather than a writing task. All 60 answerable items in the shipped gold
set are annotated.

---

## 4. Context recall

**Definition.** The share of the ground-truth answer that the retrieved context
can support. With annotations it is the share of gold chunks that were
retrieved; without them, the share of reference-answer claims attributable to
the retrieved context.

**Why it is the ceiling.** A claim that was never retrieved cannot be generated
faithfully. Context recall is therefore an upper bound on everything
downstream, and it is the first number to look at when correctness is bad: if
recall is low the generator is innocent.

**Reading it with top_k in mind.** Recall rises monotonically with `top_k` and
precision usually falls. Reporting recall without the `top_k` that produced it
is meaningless — which is why every row in `results/comparison.csv` carries the
config fingerprint, and the full config is stored in each run's `summary.json`.
In the shipped sweep, going from `top_k=5` to `top_k=8` moves recall 0.983 →
0.992 and precision 0.800 → 0.718: nine points of precision bought for one
point of recall, which is a bad trade for a model with a finite attention
budget.

---

## 5. Answer correctness

**Definition.** Agreement with a human-written reference answer, scored 0–4 by
an LLM judge against a fixed rubric at temperature 0, normalised to [0, 1].

**Why a judge at all.** Lexical overlap cannot see that "the controller must
notify within 72 hours" and "notification is due no later than three days"
agree. Only a model (or a human) can.

**What buying that capability costs, and how this repo pays it back.**

| Cost | Mitigation here |
|---|---|
| Non-determinism | temperature 0, fixed seed, integer 0–4 scale |
| Unreproducible scores | the rubric is a file, hashed into every score; changing it is a new experiment |
| Cost and latency | disk cache keyed by (metric, rubric hash, model, question, answer) |
| Crashes mid-sweep | strict JSON parse with a lexical fallback; a judge failure never aborts a run |
| No key, no evaluation | heuristic fallback: `0.6 * token F1 + 0.4 * key-fact recall` |

**The key-fact term is not decoration.** In this domain the facts that matter
are numbers, deadlines and article references, and a bag-of-words F1 scores
"72 hours" and "27 hours" almost identically. Key-fact recall extracts numbers,
legal references and quoted phrases from the reference and checks for them
literally, so a wrong number is penalised the way a reviewer would penalise it.

**Judge bias — say this before a reviewer does.** LLM judges favour verbose,
confident answers, and a judge from the same family as the generator flatters
it. Three rules follow: hold the judge model fixed across any configurations
you intend to compare; record which model judged (the results do, in
`summary.json → judge`); and hand-check 10–20 items before you trust a ranking.
`evaluation.judge.backend: same-as-generator` is convenient and is the setting
most likely to produce a number you cannot defend.

**On unanswerable items** correctness is defined as the refusal decision itself
(1.0 for a correct refusal, 0.0 otherwise). Scoring the prose of a refusal
against an empty reference would be meaningless.

---

## 6. Refusal accuracy, and why 25% of the gold set is unanswerable

**Definition.** The share of items where the refusal decision was right, across
the whole gold set. Reported with two components that matter more than the
headline:

* **over-refusal rate** — refusing a question the corpus does answer. Cost: an
  unhelpful product.
* **hallucination rate on unanswerable items** — answering a question the
  corpus does not answer. Cost: a confident falsehood, which in a regulatory
  domain is the failure that ends the project.

**Why negatives are mandatory.** Without unanswerable items, a system that
never refuses looks perfect. Refusal accuracy is undefined, over-refusal is
zero, and every other metric is computed only over questions the corpus
answers. The gold-set validator in `goldset.py` therefore *rejects* a set with
no negatives outright.

**What makes a good negative.** It must be plausible — it should look like
something the corpus would cover. "What is the retention period for CCTV
footage in the office?" is a fair test against a data-governance corpus.
"What is the capital of Peru?" is a straw man that every system passes.

**Refusal is a token, not a tone.** The model must emit the literal string
`INSUFFICIENT_CONTEXT`. A free-text apology is unparseable; a fixed marker
makes refusal a measurable binary event.

**There is no setting that wins both error types.** The abstention threshold
sweeps as follows on the shipped corpus (`generation.coverage_threshold`):

| threshold | refusal accuracy | answered-when-unanswerable | over-refusal | correctness (answerable) |
|---|---|---|---|---|
| 0.34 (default) | 0.850 | 0.45 | 0.05 | 0.527 |
| 0.45 | 0.838 | 0.20 | 0.15 | 0.501 |
| 0.55 | 0.738 | 0.10 | 0.32 | 0.464 |
| 0.65 | 0.662 | 0.00 | 0.45 | 0.446 |

Refusal *accuracy* peaks at the most permissive setting while the dangerous
error — answering something the corpus cannot support — is at its worst there.
That is exactly why the headline number is never reported alone. Pick the
operating point from the cost of each error in your domain, not from the column
with the biggest number.

---

## 7. Citation validity and coverage

**Coverage** is the share of non-refused answers carrying at least one
citation. **Validity** is the share of emitted citations that resolve to a
`(source, page)` that was actually retrieved.

These catch a failure the other metrics miss: a *fabricated reference*. A model
can state a true fact and attribute it to a page that does not exist, and
faithfulness will not notice, because the fact is supported somewhere in the
context. Citation validity is also the cheapest effective defence against
prompt injection — an injected payload is normally emitted bare, with no
citation at all.

---

## 8. Latency and cost

**p95, not the mean.** RAG latency distributions are right-skewed: a cache miss
plus a long generation is a different animal from the median query, and users
feel the tail. The mean hides it.

**Cost per query is half of every deployment decision.** A configuration that
wins faithfulness by two points and costs four times as much is not obviously
the winner, and you cannot have that argument without the number. Prices are
config values, not constants, because vendor pricing moves.

With the offline backend both columns are trivially small and cost is exactly
zero. They exist to be populated by `configs/openai.yaml` and
`configs/ollama.yaml`.

---

## 9. How much difference is real? (the resolution of an 80-item gold set)

This is the question a good reviewer asks about the comparison table, and the
honest answer is: **less than the table appears to show.**

`scripts/significance.py` computes Wilson intervals for proportions and
percentile bootstrap intervals (20 000 resamples) for means and for *paired*
differences. On the shipped results:

    refusal_accuracy      68/80 = 0.850   95% CI [0.756, 0.912]     width 15.6 pp
    context_precision     n=60  0.800     95% CI [0.742, 0.855]
    answer_correctness    n=80  0.532     95% CI [0.471, 0.593]

    hybrid-k5 - dense-only, answer_correctness, 80 paired items:
        mean difference +0.0134   95% CI [-0.0023, +0.0326]
        -> the interval spans zero: this gold set cannot tell these two apart

So the correct reading of the sweep is: the chunker and retriever variants
tested here are *not* separated at n=80. Report the ranking as a hypothesis,
not a result, and either grow the gold set or stop claiming the difference.
Pairing matters — both configurations answer the same questions, so the paired
interval is much tighter than two independent intervals would be, and it is
still not tight enough.

Rules of thumb for this scale: a proportion measured on 80 items carries roughly
±10 percentage points; detecting a 5-point difference in a mean metric needs
several hundred items. Spend your annotation budget accordingly.

---

## 10. RAGAS, and why it is optional

RAGAS is the reference implementation of faithfulness, context precision and
context recall, and its LLM-based claim decomposition is finer-grained than the
sentence-level decomposition here. When a key is available, `evaluation.use_ragas:
true` runs it.

It is optional because it pulls a large dependency tree, needs a key and a
network, and is the most common reason a RAG repository cannot be reproduced
six months later. The built-in metrics are the default so that the evaluation
suite itself is never the flaky part of CI.

**Never average a RAGAS score with a built-in score.** They are different
quantities. Every run records which engine produced its numbers in
`summary.json → metrics_engine`.

---

## 11. Threats to validity (the list to volunteer, not wait to be asked)

1. **The demo corpus is synthetic.** `data/samples/*.md` are paraphrases
   written for this repository, not the official texts. Numbers computed on it
   describe the system, not the law. `scripts/fetch_corpus.py` pulls the real
   documents; the shipped gold annotations will not transfer.
2. **The default generator is not a language model.** The offline backend is a
   scripted extractive stub. It makes faithfulness trivially 1.0 and caps
   correctness around 0.5. Treat the shipped numbers as a *floor* and a proof
   that the harness works, not as a model evaluation.
3. **The injection numbers are partly simulated.** See README, "Honest
   limitations".
4. **The heuristic judge is lexical.** It under-scores correct paraphrases.
   Configure a real judge before comparing generative models.
5. **Relevance is defined by annotation.** A gold annotation that names one
   page when two pages support the answer will understate context precision.
6. **n = 80.** See section 9.
7. **Single-turn only.** No conversational context, no query rewriting, no
   multi-hop retrieval. Questions requiring two documents are out of scope and
   are not in the gold set.
