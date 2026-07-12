# Probing Procedural Ordering Knowledge in LLM Hidden States

Linear probes trained on Mistral-7B hidden states can recover temporal ordering
relationships between procedural steps — and on most conditions tested, they do
so better than the model's own output predictions.

This repository contains the experimental pipeline for two probe tasks on the
[ProScript v1a](https://github.com/collinskatie/proscript) dataset of procedural
plans:

- **Probe 1 (ordering direction):** given two steps, is A required to come
  before B?
- **Probe 2 (ordering necessity):** given two steps, is the ordering between
  them strictly necessary, or could they happen in either order?

---

## Key results

### Probes learn genuine signal (permutation tests)

Real cross-validated AUC vs. a label-permuted null, computed via 3-fold
GroupKFold CV on a training-set subsample (≤5,000 rows), 30 permutations:

| Probe | Layer | Real CV AUC | Permuted mean | p-value |
|---|---|---|---|---|
| Probe 1 (direction) | 16 | 0.836 | 0.499 | 0.032 |
| Probe 2 (necessity) | 15 | 0.841 | 0.500 | 0.032 |

p = 0.032 is the minimum reportable value at 30 permutations (0/30 permuted
runs reached the real AUC). This is a different, independent computation from
the dev-set layer-sweep AUC below — see the researcher guide for why the two
numbers aren't interchangeable.

### Held-out test-set evaluation

Thresholds are locked on the dev set (best-F1 threshold, swept over
0.30–0.75), then applied fixed to the test set. No synthetic corruption is
used — these are the original test DAGs.

**Probe 1** — real edge vs. reversed edge / incomparable pair, evaluated on
all 2,077 test plans (31,680 pairs, 43.8% positive):

| System | F1 | Precision | Recall | Threshold |
|---|---|---|---|---|
| Probe | **0.776** | 0.752 | 0.802 | 0.50 |
| LLM output | 0.606 | 0.437 | 0.984 | 0.30 |

Margin: **+0.171 F1**.

**Probe 2** — real edge vs. spurious (same-depth / cross-branch incomparable
pair), evaluated on the 817 multi-ordering test plans (10,095 pairs, 60.9%
positive):

| Condition | Probe F1 | LLM F1 | Margin |
|---|---|---|---|
| Combined | **0.789** | 0.757 | +0.032 |
| Same-depth (in-distribution) | **0.826** | 0.816 | +0.010 |
| Cross-branch (out-of-distribution) | 0.843 | **0.911** | **−0.067** |

The LLM's F1 "win" on cross-branch is not genuine discrimination — see
[Known limitations](docs/researcher_guide.md#known-limitations) for why raw F1
is misleading here and what the specificity numbers actually show.

---

## Quick start

### 1. Install dependencies

```bash
pip install transformers accelerate scikit-learn tqdm matplotlib
```

### 2. Download ProScript v1a

The ProScript dataset is not included due to redistribution constraints.
Download from the [ProScript repository](https://github.com/collinskatie/proscript).

**Notebook 1** (`probe_training_and_permutation_v2.ipynb`) expects
`train.jsonl` and `dev.jsonl` uploaded directly to `/content/` in Colab.
`test.jsonl` is **not** required for this notebook.

**Notebook 2** (`probe_classification_v2.ipynb`) expects `dev.jsonl`,
`test.jsonl`, `probe_manifest.json`, and both probe `.pkl` files placed in a
Google Drive folder at `MyDrive/dag-probing/` — it mounts Drive and copies
them locally. Update `DRIVE_DIR` in the setup cell if you use a different
folder name.

### 3. Run notebooks in order

| Notebook | What it does |
|---|---|
| `probe_training_and_permutation_v2.ipynb` | Layer sweep (layers 15–18), probe training, permutation tests |
| `probe_classification_v2.ipynb` | Dev-threshold locking, held-out test-set evaluation |

Notebook 2 requires the probe `.pkl` files and `probe_manifest.json` saved by
Notebook 1. *(The copies currently in `notebooks/` still carry a `(1)`
download-artifact suffix in their filenames — rename before committing.)*

No verified runtime figures are available yet for either notebook; budget GPU
time for roughly 40k forward passes for Probe 1's training set alone.

---

## Methodology

### Probe architecture

Both probes are `StandardScaler → LogisticRegression(C=1.0, max_iter=2000,
class_weight='balanced')` pipelines fitted on Mistral-7B-Instruct-v0.1 hidden
states, using an `[INST] ... [/INST]` chat-template wrapper. Each probe uses
its own dedicated prompt:

- **Probe 1** (`tp1_direction`): *"Must Action A happen before Action B?
  Answer yes or no."*
- **Probe 2** (`tp2_necessity`): *"Is it strictly necessary for A to happen
  before B, or could they happen in either order without affecting the task?
  Answer yes (necessary) or no (flexible)."*

Layer selection sweeps layers 15–18, fitting on training features and scoring
on held-out dev AUC. **Probe 1 selected layer 16** (dev AUC 0.900, macro
0.912). **Probe 2 selected layer 15** (dev AUC 0.855, macro 0.857) — the
committed `probe2_layer15.pkl` and `probe_manifest.json` both confirm this;
layer 15 outperforms layer 17 in the actual sweep data.

### Training data

**Probe 1:** confirmed edges (y=1) vs. reversed edges + incomparable pairs
from multi-ordering plans (y=0). Balanced to 41,126 rows.

**Probe 2:** confirmed edges (y=1) vs. same-depth incomparable pairs (y=0).
Balanced to 6,672 rows.

All training uses the official ProScript v1a training split. Dev plan names
are excluded from the Probe 1/2 training set via an explicit leakage guard.

### Evaluation design

**Threshold locking:** for each probe (and the LLM baseline), the F1-optimal
threshold is found by sweeping {0.30, 0.35, ..., 0.75} on the dev set, then
held fixed for the test-set evaluation. Probe 2 uses one combined threshold
for both the same-depth and cross-branch subtypes — there are no separate
per-subtype thresholds.

**Probe 1 pairs:** for each real edge (a, b), the reversed pair (b, a) is
included as a negative, plus every incomparable pair (in both directions) for
plans with genuine parallel structure. Evaluated on **all** test plans, since
every plan contributes reversed-edge negatives even without incomparable
pairs.

**Probe 2 pairs:** for each real edge, both same-depth and cross-branch
incomparable pairs (in both directions) are candidate spurious edges.
Same-depth pairs are in-distribution for Probe 2's training; cross-branch
pairs test generalization to a topology the probe never saw negatives from.
Evaluated only on the 817 multi-ordering test plans, since single-ordering
plans have no incomparable pairs to draw from.

### Metrics

- **Edge F1 / precision / recall** — the primary reported metrics.
- **Specificity** (true-negative rate) — not in the saved CSVs by default,
  but derivable from the tp/fp/fn/tn columns and important to check
  alongside F1: with a large positive-class majority (e.g. 6,147 real edges
  vs. 1,182 cross-branch spurious pairs), F1 is dominated by recall, and a
  system that predicts "yes" almost universally can post a high F1 while
  discriminating almost nothing. See the researcher guide for the actual
  specificity numbers.

A DAG-level reconstruction evaluation (transitive-closure agreement, graph
edit distance, perfect-reconstruction rate, under synthetic edge
deletion/insertion at varying severity) is **not yet implemented** in this
repository — see "Recommended next steps" in the researcher guide.

---

## Repository structure

```
dag-probing/
├── notebooks/
│   ├── probe_training_and_permutation_v2.ipynb
│   └── probe_classification_v2.ipynb
├── results/
│   ├── probes/           probe PKLs + manifest
│   ├── layer_sweep/      per-layer dev AUC CSVs + figure
│   ├── permutation/      permuted AUC CSVs + figure
│   └── enrichment/       test-set classification results CSV + score dumps + figure
├── scripts/
│   └── check_proscript_splits.py
├── docs/
│   └── researcher_guide.md
├── .gitignore
└── README.md
```

---

## Data splits

The notebook pipeline actually loaded **3,252 train / 1,085 dev / 2,077 test
plans** (total 6,414 — matching Sakaguchi et al.'s stated dataset size).
`scripts/check_proscript_splits.py`, run separately, reports different totals
(3,099 / 1,031 / 1,966, total 6,096) — this likely reflects a different local
copy or download of the ProScript v1a release and should be reconciled before
citing split sizes in the paper.

The multi-ordering subset counts used for Probe 2 evaluation are internally
consistent regardless: **361 dev plans (33.3%)** and **817 test plans
(39.3%)** have at least one genuinely incomparable step pair.

| Split | Plans (notebook-loaded) | Multi-ordering | Role |
|---|---|---|---|
| Train | 3,252 | — | Probe training |
| Dev | 1,085 | 361 (33.3%) | Layer selection + threshold locking |
| Test | 2,077 | 817 (39.3%) | Final evaluation (reported results) |

---

## Citation

```bibtex
@inproceedings{,
  title     = {Probing Procedural Ordering Knowledge in LLM Hidden States},
  author    = {},
  booktitle = {LM4Plan Workshop @ ICAPS 2026},
  year      = {2026}
}
```

## Acknowledgements

Extends [Orgad et al., ICLR 2025](https://arxiv.org/abs/2410.02707) to
procedural temporal reasoning. ProScript dataset: [Sakaguchi et al.,
2021](https://arxiv.org/abs/2104.08251).