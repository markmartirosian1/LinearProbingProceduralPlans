# File Reference — DAG Probing Research

Descriptions for the 8 files in this package.

---

## Notebooks

### `mode2_probe2_v2__2_.ipynb`
**Mode 2 — Binary Probe 2 with LLM baselines**

Evaluates Mode 2 (spurious edge removal) using a binary logistic probe trained on Mistral-7B hidden states at layer 17. The probe distinguishes `confirmed_keep` edges (real ordering constraints, `label=1` from v2) from `synthetic_spurious_parallel` edges (same-depth incomparable pairs, used as flexible negatives).

Runs a full threshold sweep (t = 0.35–0.70) across all four training prompt variants (`tp1_ordering`, `tp2_parallel`, `tp3_dependency`, `tp4_flexibility`) on both ProScript and CaptainCook. Includes two LLM baselines: **direct** (`p_yes < threshold` → propose removal) and **inverted** (`p_yes > threshold` → propose removal, to account for yes-bias). p_yes values are cached during the probe pass — no extra inference cost.

Reports three F1 metrics per configuration: `macro_f1` (all plans), `enrichable_f1` (plans with ≥1 GT edge to remove), and `corpus_f1` (micro-averaged across all edges).

**Requires:** `proscript_train_edges_v2.csv`, `proscript_train_edges_v3.csv`, `proscript_pipeline_eval_final.csv`, `captaincook_pipeline_eval.csv`, ProScript JSON ZIP.

**Saves:** `mode2_probe2v2_sweep.csv`, probe PKL files.

---

### `validation_notebook__1_.ipynb`
**Mode 1 & Mode 2 — Validation suite with permutation tests**

Three experiments in one notebook:

**Mode 1 test set** — parses `DAGAnnotationFinal.xlsx` (not in this package) using GraphViz DOT diffs to extract ground-truth positives (green edges = missing ordering constraints). Builds a 22-pair balanced test set (11 positives + 17 negatives) from the 15 MULTI plans and evaluates Probe 1 across threshold sweep. Also builds a synthetic validation set from Group C single-ordering plans (hard edges removed = synthetic positives).

**Mode 1 permutation test** — reuses the Probe 1 training hidden states (already extracted) and runs 100 label-shuffled CV trials to confirm the probe is learning genuine signal, not overfitting to training set statistics.

**Mode 2 permutation test** — same approach for Probe 2: 100 shuffled-label CV trials on the binary `confirmed_keep` vs `synthetic_spurious_parallel` training set. Produces the key validity result: real AUC = 0.902, permuted mean ≈ 0.502, p < 0.01.

**Requires:** `proscript_train_edges_v2.csv`, `proscript_train_edges_v3.csv`, `proscript_pipeline_eval_final.csv`, `captaincook_pipeline_eval.csv`, `DAGAnnotationFinal.xlsx` (annotation file, not in this package), ProScript JSON ZIP.

**Saves:** `mode1_test_set.csv`, `mode1_test_results.csv`, `mode1_synthetic_val.csv`, `mode1_eval_diagnostics.csv`, `mode1_antisymmetry.csv`, `permutation_aucs_mode2.csv`, `permutation_aucs_mode1_tp1_ordering.csv`, `permutation_aucs_mode1_tp4_flexibility.csv`, `permutation_test.png`, `permutation_test_mode1.png`, `probe1_calibration.png`.

---

### `enrichment_scaling_final.ipynb`
**DAG enrichment scaling experiment**

Tests both probe modes across a range of corruption levels on single-ordering and multi-ordering Group C plans. Three experiments:

**Mode 1 deletion sweep** — removes 20%, 40%, and 60% of load-bearing edges (edges with no transitive alternative path) from single-ordering plans, then measures how well Probe 1 recovers them. Negatives come from genuine incomparable pairs in multi-ordering plans. LLM direct baseline included.

**Mode 2 spurious insertion** — inserts 1, 2, or 3 spurious edges per plan across two subconditions: **Mode 2a** (same-depth incomparable pairs added as false ordering constraints, matching the probe's training distribution) and **Mode 2b** (cross-branch incomparable pairs). Negatives are real existing edges. LLM inverted baseline included.

**Combined** — 40% deletion + 2 spurious edges on the same plan, both probes run simultaneously on the corrupted DAG.

**Requires:** `proscript_train_edges_v2.csv`, `proscript_train_edges_v3.csv`, `proscript_pipeline_eval_final.csv`, ProScript JSON ZIP. Optionally loads pre-trained probe PKL files (`retrain=False` in config) to skip ~2h training.

**Saves:** `enrichment_mode1.csv`, `enrichment_mode2.csv`, `combined_m1_probe.csv`, `combined_m1_llm.csv`, `combined_m2_probe.csv`, `combined_m2_llm.csv`, `mode1_scaling.png`, `mode2_scaling.png`.

---

## Data files

**Data splits — Groups A, B, and C**
The 622 ProScript plans are divided into three non-overlapping groups. Group B is the probe training split: all confirmed, reversed, and flexible edge examples in proscript_train_edges_v2.csv and v3.csv come from Group B plans. Group C is the evaluation split: the 91 plans in proscript_pipeline_eval_final.csv are all Group C plans, and this is the only population used for Mode 1 and Mode 2 pipeline evaluation, the annotation-based test set, and the enrichment scaling experiment. Group A is a held-out split not used in any notebook in this repository. The split ensures that no plan used to train a probe ever appears in evaluation — all leakage checks in the notebooks assert this explicitly. CaptainCook4D plans are entirely separate from the ProScript split and were never seen during training, making CaptainCook evaluation a zero-shot transfer test.

### `proscript_train_edges_v2.csv`
**Probe training data — Group B, three-class, 2,154 rows**

| Column | Description |
|---|---|
| `goal` | Plan name (e.g. "eat a quick dinner") |
| `a` | First step text |
| `b` | Second step text |
| `label` | 0 = reversed, 1 = confirmed, 2 = flexible |
| `edge_type` | `confirmed`, `reversed`, or `flexible` |
| `source` | How the row was generated |

Perfectly balanced: 718 rows per class. All plans are from Group B (training split) — no overlap with the 91 Group C eval plans.

**How notebooks use it:**
- Probe 1 training: `label==1` (confirmed) as positives + `label==0` (reversed) + incomparable pairs from multi-ordering plans as negatives.
- Probe 2 training: `label==1` (confirmed_keep) as positives + same-depth incomparable pairs (generated from JSON files) as negatives. `label==2` rows are not used by either probe directly.

---

### `proscript_train_edges_v3.csv`
**Probe training data — Group B with multi-ordering tag, 1,380 rows**

Same structure as v2 with an additional `system_tag` column. The key difference: includes a `truly_parallel` edge type identifying which Group B plans have genuine incomparable pairs at the same topological depth.

| Column | Description |
|---|---|
| `goal` | Plan name |
| `a`, `b` | Step pair |
| `label` | 0 = reversed, 1 = confirmed, 2 = truly parallel |
| `edge_type` | `confirmed`, `reversed`, or `truly_parallel` |
| `source` | Row provenance |
| `system_tag` | Experiment version tag |

Balanced at 460 rows per class. Used by all three notebooks to identify which Group B plans are multi-ordering (`edge_type == 'truly_parallel'`), which determines which plans contribute flexible/spurious training examples.

---

### `proscript_pipeline_eval_final.csv`
**ProScript Group C evaluation set — 91 plans**

The held-out evaluation set. All 91 plans are from Group C (never seen during probe training).

| Column | Description |
|---|---|
| `goal` | Plan name |
| `enrichable` | 1 if the plan has at least one GT change (add or remove), else 0 |
| `label` | `single_ordering` (62 plans) or `multi_ordering` (29 plans) |
| `steps` | Pipe-separated step names |
| `dag_edges` | `A→B \| C→D` format — original ProScript DAG |
| `incomparable_pairs` | `A↔B \| C↔D` format — NaN for single-ordering plans |
| `n_incomparable` | Count of incomparable pairs (0 for single-ordering) |
| `n_dag_edges` | Count of existing directed edges |
| `ground_truth_new_edges` | Edges to ADD (Mode 1 GT) — NaN if plan is correctly constrained |
| `ground_truth_removed_edges` | Edges to REMOVE (Mode 2 GT) — NaN if plan is not over-constrained |
| `ground_truth_full_dag` | The complete corrected DAG |
| `n_added`, `n_removed` | Counts of GT changes |

NULLs in GT columns are expected: 46 plans have no new edges to add, 52 have no edges to remove. Only 45 plans are enrichable (have at least one GT change).

---

### `captaincook_pipeline_eval.csv`
**CaptainCook zero-shot evaluation set — 21 recipes**

Used exclusively for zero-shot transfer evaluation — no CaptainCook data appears in probe training. Structure mirrors the ProScript eval file.

| Column | Description |
|---|---|
| `goal` | Recipe name (e.g. "blenderbananapancakes") |
| `enrichable` | 1 if plan has at least one GT change |
| `n_steps` | Number of steps in the recipe |
| `steps` | Pipe-separated step names (note: steps use the raw CaptainCook annotation text including action prefixes) |
| `dag_edges` | Original CaptainCook task graph edges |
| `incomparable_pairs` | Parallel step pairs |
| `n_incomparable`, `n_dag_edges` | Counts |
| `ground_truth_new_edges` | Edges to ADD — NaN for 8 plans |
| `ground_truth_removed_edges` | Edges to REMOVE — NaN for 7 plans |
| `n_added`, `n_removed` | Counts of GT changes |

14 of 21 plans are enrichable. GT edges were derived from the official CaptainCook4D ground-truth task graphs (`edges_completed` field in the dataset JSON).

---
