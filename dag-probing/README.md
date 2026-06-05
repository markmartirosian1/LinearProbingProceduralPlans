# Probing Procedural Knowledge in LLM Hidden States
### DAG Enrichment via Mistral-7B Linear Probes

Research code for *"LLMs Know More Than They Show"*-style probing applied to procedural temporal reasoning. 

The project investigates whether procedural ordering knowledge is linearly encoded in Mistral-7B's hidden states at layers 16–18, using two tasks on ProScript and CaptainCook4D procedural DAGs:

- **Mode 1 (enrichment):** given a partially-specified DAG, identify which incomparable step pairs should receive ordering edges
- **Mode 2 (correction):** given an over-constrained DAG, identify which existing edges are spurious and should be removed

---

## Repository structure

```
.
├── notebooks/
│   ├── 02_mode2_probe2_v2.ipynb       # Full 4-system sweep: Sys1–4 × all prompts × PS + CC
│                                      # Binary Probe 2 + LLM baselines (primary Mode 2 eval)
│   ├── 03_validation_notebook.ipynb               # Probe training, Mode 1 test set, permutation tests
│   └── 04_enrichment_scaling.ipynb       # Deletion sweep + spurious insertion scaling experiment
│
├── data/
│   ├── train/
│   │   ├── proscript_train_edges_v2.csv  # Group B: confirmed (label=1) + reversed (label=0) edges
│   │   └── proscript_train_edges_v3.csv  # Group B: truly_parallel edge type (multi-ordering plans)
│   ├── eval/
│   │   ├── proscript_pipeline_eval_final.csv   # Group C: dag_edges, GT new/removed edges
│   │   └── captaincook_pipeline_eval.csv       # CaptainCook zero-shot eval
│   └── annotation/
│       └── DAGAnnotationFinal.xlsx       # 46 hand-annotated plans: 15 MULTI + 31 SINGLE
│                                         # Column G = input DOT, Column I = ground-truth DOT
│
├── results/
│   ├── mode1/
│   │   ├── mode1_test_set.csv            # 11 positive + 30 negative pairs (annotation DOT diff)
│   │   ├── mode1_test_results.csv        # Threshold sweep — annotation test set
│   │   ├── mode1_synthetic_val.csv       # Threshold sweep — synthetic validation set
│   │   ├── mode1_eval_diagnostics.csv    # Per-pair scores, pred_class, calibration
│   │   ├── mode1_antisymmetry.csv        # Directional confidence analysis
│   │   └── mode1_balanced_sweep.csv      # Balanced threshold sweep
        ├── permutation_test_mode1.png        # Mode 1: tp1 + tp4 permutation panels
│       ├── probe1_calibration.png            # Reliability diagram + directional confidence
│   ├── mode2/
│   │   ├── mode2_probe2v2_sweep.csv      # Binary Probe 2: corpus/enrichable F1, LLM 
baselines 
        ├── permutation_test.png          # Mode 2 permutation test
│   ├── enrichment/
│   │   ├── enrichment_mode1.csv          # Mode 1 deletion sweep (20/40/60%)
│   │   ├── enrichment_mode2.csv          # Mode 2a/2b spurious insertion (n=1/2/3)
│   │   ├── combined_m1_probe.csv         # Combined experiment — Mode 1 probe
│   │   ├── combined_m1_llm.csv           # Combined experiment — Mode 1 LLM baseline
│   │   ├── combined_m2_probe.csv         # Combined experiment — Mode 2 probe
│   │   └── combined_m2_llm.csv          # Combined experiment — Mode 2 LLM low-confidence
│       ├── mode1_scaling.png                 # F1 vs deletion rate, precision/recall ]
│       ├── mode2_scaling.png                # Mode 2a/2b F1 vs spurious level
│   └── permutation/
│       ├── permutation_aucs_mode2.csv    # 100 shuffled-label CV AUCs — Probe 2
│       ├── permutation_aucs_mode1_tp1_ordering.csv
│       └── permutation_aucs_mode1_tp4_flexibility.csv
│
│
├── docs/
│   └── results.md              # Diagnostic guide: all results, validity analysis, next steps
│   └── FILES_README.md                  # A guide on every single notebook and data file. 
├── .gitignore
└── README.md
```

---

## Quick start

### 1. Get the data

ProScript JSON files are required for all notebooks. Download from the [ProScript dataset repository](https://github.com/collinskatie/proscript) and extract to `/content/proScript_data/` (for Colab) or update `DATA_DIR` in each notebook config cell.

CaptainCook4D task graphs are available from the [CaptainCook4D dataset](https://github.com/CaptainCook4D). The pipeline eval CSV (`data/eval/captaincook_pipeline_eval.csv`) is pre-processed and included here.

### 2. Run notebooks in order

All notebooks are designed for Google Colab with A100 GPU. Upload files to `/content/` before running.

| Notebook | Uploads required | Saves |
|---|---|---|
| `02_mode2_probe2_v2.ipynb` | ZIP + v2 + v3 + PS eval + CC eval | `mode2_probe2v2_sweep.csv`, probe PKLs |
| `02_mode2_probe2_v2.ipynb` | ZIP + v2 + v3 + PS eval + CC eval | `mode2_probe2v2_sweep.csv`, probe PKLs |
| `03_validation_notebook.ipynb` | ZIP + v2 + v3 + PS eval + CC eval + `DAGAnnotationFinal.xlsx` | All `results/mode1/`, permutation CSVs, figures |
| `04_enrichment_scaling.ipynb` | ZIP + v2 + v3 + PS eval (+ probe PKLs if `retrain=False`) | All `results/enrichment/`, figures |

Notebooks 03 and 04 can load pre-trained probes from PKL files saved by 02 (`retrain=False` in config) to skip the ~2h training step.

### 3. Dependencies

```bash
pip install transformers accelerate scikit-learn tqdm openpyxl matplotlib
```

Model: `mistralai/Mistral-7B-Instruct-v0.1` (loaded via HuggingFace in each notebook).

---

## Key results

### Mode 2 — Probe 2 vs. LLM baseline (annotation evaluation)

| Dataset | Probe 2 corpus F1 | LLM direct F1 | Margin |
|---|---|---|---|
| ProScript | 0.404 | 0.217 | +0.187 (+86%) |
| CaptainCook (zero-shot) | 0.549 | 0.397 | +0.152 (+38%) |

### Mode 2 — Permutation test

Real CV AUC = **0.902** · Permuted mean = 0.502 ± 0.028 · Max permuted = 0.560 · **p < 0.01** (0/100 runs ≥ real AUC) · Gap = 0.400

### Mode 1 — Annotation test (11 positives, 15 MULTI plans)

Best F1 = **0.667** (tp1_ordering, t=0.60) · TP=7, FP=3, FN=4

### Enrichment scaling experiment

| | 20% deletion | 40% deletion | 60% deletion |
|---|---|---|---|
| Probe 1 F1 | 0.795 | 0.752 | 0.737 |
| LLM direct F1 | 0.673 | 0.665 | 0.673 |

| | 2a same-depth (n=2) | 2b cross-branch (n=2) |
|---|---|---|
| Probe 2 F1 | 0.772 | 0.839 |
| LLM low-confidence F1 | 0.583 | 0.519 |

---

## Probe architecture

- **Probe 1 (Mode 1):** binary logistic regression on Mistral-7B hidden states at layer 17. Training: `confirmed_keep` edges (label=1) vs. `incomparable_pairs` + `reversed` edges (label=0) from Group B plans. Applied to incomparable pairs at inference.
- **Probe 2 (Mode 2):** binary logistic regression at layer 17. Training: `confirmed_keep` edges (label=1) vs. same-depth incomparable pairs (label=0, `synthetic_spurious_parallel`). Applied to existing DAG edges at inference.
- Both trained with `class_weight='balanced'`, 5-fold plan-level `GroupKFold` CV.

---

## Data splits

ProScript 622 plans are split into three groups:
- **Group A:** held out
- **Group B:** probe training (confirmed/reversed/flexible examples)
- **Group C:** evaluation (`proscript_pipeline_eval_final.csv`, 91 plans)

`DAGAnnotationFinal.xlsx` covers 46 Group C plans with human-annotated ground-truth DAGs. Green edges in Column I = Mode 1 positives; dashed orange edges = Mode 2 positives.

---

## Probe weights

Pre-trained probe PKL files are not committed (regenerate via `03_validation_notebook.ipynb` or `02_mode2_probe2_v2.ipynb`). If you need them without retraining, open an issue.

---

## Citation

```bibtex
@inproceedings{,
  title     = {Probing Procedural Knowledge in LLM Hidden States for DAG Enrichment},
  author    = {},
  booktitle = {LM4Plan Workshop @ ICAPS 2026},
  year      = {2026}
}
```

---

## Acknowledgements

Extends [Orgad et al., ICLR 2025](https://arxiv.org/abs/2410.02707) — *"LLMs Know More Than They Show"* — to procedural temporal reasoning. ProScript dataset: [Singh et al., EMNLP 2022](https://aclanthology.org/2022.emnlp-main.701/). CaptainCook4D: [Peddi et al., 2023](https://arxiv.org/abs/2312.14556).
