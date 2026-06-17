# Probing Procedural Ordering Knowledge in LLM Hidden States

Linear probes trained on Mistral-7B hidden states can recover temporal ordering relationships between procedural steps — and they do so significantly better than the model's own output predictions.

This repository contains the full experimental pipeline for two probe tasks on the [ProScript v1a](https://github.com/collinskatie/proscript) dataset of procedural plans:

- **Mode 1 (ordering direction recovery):** given a DAG with missing edges, identify which direction the ordering should go between steps that lost reachability
- **Mode 2 (spurious edge detection):** given a DAG with inserted false edges, identify which edges don't belong

---

## Key results

### Probes learn genuine signal (permutation tests)

| Probe | Best layer | Real CV AUC | Permuted mean | p-value |
|---|---|---|---|---|
| Probe 1 (Mode 1) | 16 | 0.824 | 0.501 | < 0.01 |
| Probe 2 (Mode 2) | 17 | 0.881 | 0.502 | < 0.01 |

### Mode 1 — ordering direction recovery

The probe identifies which step should come first among pairs that lost ordering due to edge deletion. Evaluated on 817 multi-ordering test plans.

| Hard edges deleted | Probe F1 | LLM F1 | Probe TCA | LLM TCA |
|---|---|---|---|---|
| 20% | **0.861** | 0.658 | **0.812** | 0.337 |
| 40% | **0.854** | 0.658 | **0.741** | 0.295 |
| 60% | **0.848** | 0.659 | **0.705** | 0.292 |

### Mode 2 — spurious edge detection

The probe identifies inserted false edges. Two spurious-edge types tested: same-depth parallel (M2a, in-distribution for probe training) and cross-branch (M2b, out-of-distribution).

| Condition | Probe F1 | LLM F1 | Probe TCA | LLM TCA |
|---|---|---|---|---|
| M2a n=2 | **0.511** | 0.282 | **0.861** | 0.773 |
| M2b n=2 | **0.470** | 0.310 | **0.845** | 0.770 |

---

## Quick start

### 1. Clone and install dependencies

```bash
git clone https://github.com/markmartirosian1/LinearProbingProceduralPlans.git
pip install transformers accelerate scikit-learn tqdm matplotlib
```

### 2. Download ProScript v1a

The ProScript dataset is not included due to redistribution constraints. Download from the [ProScript repository](https://github.com/collinskatie/proscript) and place the JSONL files at:

```
/content/train.jsonl
/content/dev.jsonl
/content/test.jsonl
```

The dataset contains 6,096 procedural plans: 3,099 training, 1,031 development, 1,966 test. The test set includes plans from ROCStories, DeScript, and VirtualHome.

### 3. Run notebooks in order

Both notebooks are designed for Google Colab with GPU (A100 recommended).

| Notebook | Runtime (A100) | What it does |
|---|---|---|
| `01_probe_training_and_permutation.ipynb` | ~3 hours | Layer sweep, probe training, permutation tests |
| `02_enrichment_scaling.ipynb` | ~2 hours | Dev threshold selection, test evaluation, scaling curves |

Notebook 2 requires the probe PKLs and manifest saved by Notebook 1.

---

## Methodology

### Probe architecture

Both probes are `StandardScaler → LogisticRegression(C=1.0, class_weight='balanced')` pipelines fitted on Mistral-7B-Instruct-v0.1 hidden states. The prompt asks: *"Must Action A happen before Action B? Answer yes or no."*

Layer selection uses held-out dev features (fit on train, evaluate on dev AUC). Probe 1 selected layer 16; Probe 2 selected layer 17 — consistent with the 15–18 consensus zone observed across multiple prior experiments.

### Training data

**Probe 1:** confirmed edges (y=1) vs reversed edges + incomparable pairs from multi-ordering plans (y=0). Balanced to ~40k rows.

**Probe 2:** confirmed edges (y=1) vs same-depth incomparable pairs (y=0). Balanced to ~6.5k rows.

All training uses the official ProScript v1a training split. Dev and test plan names are excluded from training via an explicit leakage guard.

### Evaluation design

**Threshold locking:** thresholds are selected on the dev set at a single reference severity (40% deletion for Mode 1, n=2 for Mode 2) and held fixed across all test conditions. Scaling curves reflect genuine robustness, not threshold retuning.

**Mode 1 — direction-labeled candidates:** when edges are deleted from a DAG, multiple step pairs lose reachability (not just the directly deleted pair). For each newly-incomparable directed pair (a, b):
- y=1 if a could reach b in the original DAG (correct ordering direction)
- y=0 if b could reach a instead (reversed direction)

This gives ~50% positive rate and tests what the probe actually learns: temporal direction. Both probe and LLM baseline are evaluated on identical candidate pairs with identical labels.

**Mode 2a/2b:** spurious edges are drawn from incomparable pairs in multi-ordering plans. Mode 2a uses same-depth pairs (in-distribution for Probe 2's training). Mode 2b uses cross-branch pairs at different depths (out-of-distribution — tests generalization).

### Metrics

- **Edge F1** — precision/recall on individual pair predictions. Primary metric for Mode 1 and Mode 2.
- **TCA (transitive closure agreement)** — fraction of all step-pair reachability relationships that agree between the repaired and original DAGs. Captures ordering preservation even when the probe adds non-minimal direct edges.
- **PRR (perfect reconstruction rate)** — fraction of plans with exact edge-set recovery. Very strict; near zero for Mode 1 because the probe adds correct-direction transitive shortcuts.
- **GEDR (graph edit distance reduction)** — measures whether the repair moves the DAG closer to or further from the original. Negative when the probe adds correct-but-redundant edges, which is expected behavior for a probe that learns reachability rather than minimal graph structure.

**Which metrics to prioritize:** F1 and TCA. F1 measures discrimination quality; TCA measures practical ordering preservation. GEDR and PRR are reported for completeness but penalize correct-direction transitive edges, which is a property of the evaluation metric rather than a probe failure.

---

## Repository structure

```
dag-probing/
├── notebooks/
│   ├── 01_probe_training_and_permutation.ipynb
│   └── 02_enrichment_scaling.ipynb
├── results/
│   ├── probes/           probe PKLs + manifest
│   ├── layer_sweep/      per-layer dev AUC CSVs + figure
│   ├── permutation/      permuted AUC CSVs + figure
│   └── enrichment/       test results CSV + locked thresholds + figure
├── scripts/
│   └── check_proscript_splits.py
├── docs/
│   └── researcher_guide.md
├── .gitignore
└── README.md
```

---

## Data splits

The ProScript v1a dataset (6,096 plans) uses official train/dev/test splits:

| Split | Plans | Multi-ordering | Role |
|---|---|---|---|
| Train | 3,099 | 1,001 (32.3%) | Probe training |
| Dev | 1,031 | 361 (35.0%) | Layer selection + threshold locking |
| Test | 1,966 | 817 (41.6%) | Final evaluation (reported results) |

The test set's higher multi-ordering rate (41.6% vs 32.3%) reflects its inclusion of DeScript and VirtualHome plans alongside ROCStories, providing a genuine cross-domain generalization test.

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

Extends [Orgad et al., ICLR 2025](https://arxiv.org/abs/2410.02707) to procedural temporal reasoning. ProScript dataset: [Sakaguchi et al., 2021](https://arxiv.org/abs/2104.08251).