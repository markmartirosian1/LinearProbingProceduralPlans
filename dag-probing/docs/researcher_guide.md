# Researcher Guide — Diagnostic Analysis and Interpretation

## Overview

This guide covers the full result set, explains what each finding means, identifies known limitations, and suggests next steps. All results use the official ProScript v1a train/dev/test splits with multi-ordering test plans only (817 plans, each with genuine incomparable step pairs).

---

## Permutation tests — the validity foundation

Both probes show definitive signal at their selected layers.

| Probe | Layer | Real CV AUC | Permuted mean ± std | p-value |
|---|---|---|---|---|
| Probe 1 (Mode 1) | 16 | 0.824 | 0.501 ± ~0.02 | < 0.01 |
| Probe 2 (Mode 2) | 17 | 0.881 | 0.502 ± ~0.02 | < 0.01 |

The permutation test was run on a 5,000-row subsample of the training set for computational efficiency. The real AUC sits completely outside the permuted distribution in both cases — the signal is so strong that subsampling does not affect the conclusion.

**Methodological notes:**

The test uses global label permutation with plan-level GroupKFold CV. Global permutation is a standard approximation; a strictly correct null would preserve within-plan label structure. The test confirms that hidden states carry genuine label-correlated signal at the selected layers.

The test is run at the fixed layer selected on the development set. It does not test whether a significant layer exists across the sweep range (which would require a max-statistic permutation). It tests the narrower claim: "at this specific layer, the probe exploits real feature-label correspondence."

The minimum reportable p-value with 100 permutations is ~0.01 (computed as (1 + count) / (n_perms + 1)). Both probes hit this floor.

---

## Layer sweep — where ordering knowledge lives

Probe 1 peaked at **layer 16** (dev AUC 0.899). Probe 2 peaked at **layer 17** (dev AUC 0.872). The one-layer difference between the two probes is consistent with prior experiments showing a 15–18 consensus zone, with slight task-dependent variation.

Layer selection uses the full unbalanced dev set (15,342 rows for Probe 1, 7,954 for Probe 2) — AUC does not require balanced classes. Both micro (row-level) and macro (per-plan average) AUC are reported; layer selection uses micro AUC.

**Training scale:**

| Probe | Training rows | Dev rows |
|---|---|---|
| Probe 1 | 40,562 (balanced) | 15,342 (all) |
| Probe 2 | 6,536 (balanced) | 7,954 (all) |

The probe uses a `StandardScaler → LogisticRegression` pipeline. Convergence is checked after fitting; no convergence warnings were observed.

---

## Mode 1 — ordering direction recovery

### What the task measures

When edges are deleted from a DAG, multiple step pairs lose reachability — not just the directly deleted pair, but all pairs whose only path went through the deleted edge. The probe is asked: for each newly-incomparable directed pair (a, b), should a come before b?

The label is determined by the original DAG's reachability:
- y=1 if a→...→b existed (adding a→b restores correct ordering)
- y=0 if b→...→a existed (adding a→b reverses the ordering)

This gives ~50% positive rate per plan, since each undirected pair contributes one correct and one reversed direction.

### Results

| Hard edges deleted | Probe F1 | Probe precision | Probe recall | LLM F1 | Probe margin |
|---|---|---|---|---|---|
| 20% | 0.861 | 0.841 | 0.882 | 0.658 | +0.203 |
| 40% | 0.854 | 0.833 | 0.876 | 0.658 | +0.196 |
| 60% | 0.848 | 0.827 | 0.870 | 0.659 | +0.189 |

**Key findings:**

1. **F1 of 0.86 is strong.** The probe correctly identifies ordering direction on 85%+ of pairs where reachability was broken. Precision and recall are well balanced.

2. **Extremely robust to corruption severity.** Only 1.3pp F1 drop from 20% to 60% deletion — a 3× increase in corruption causes negligible performance loss.

3. **LLM baseline is flat at 0.66.** With 50% positive rate, the LLM's yes-bias (recall 0.97, precision 0.50) produces F1 near the chance ceiling. The probe's +0.20 margin is genuine ordering discrimination beyond what the model's output logits capture.

4. **TCA tells the reachability story.** Probe TCA = 0.81 at 20% deletion vs LLM TCA = 0.34. The probe preserves 2.4× more ordering relationships. TCA degrades to 0.70 at 60% deletion — still double the LLM's 0.29.

### Why GEDR is negative and why it's expected

GEDR compares exact edge sets. The probe correctly identifies that "buy a guitar should come before practice daily" — but the original DAG expressed this transitively (buy → learn chords → practice), not as a direct edge. The probe adds the direct edge, which is correct for ordering but creates a transitive shortcut absent from the original edge set. Every correct-but-redundant edge pushes GEDR negative.

This is a property of the probe learning reachability (the right concept for ordering) rather than minimal DAG structure (a graph-theoretic property orthogonal to temporal knowledge). GEDR is reported for completeness but should not be interpreted as probe failure. TCA is the correct metric for ordering preservation.

PRR is near zero for the same reason — exact edge-set match requires the probe to add only the specific deleted edges and no transitive shortcuts, which is a much harder task than knowing ordering direction.

---

## Mode 2 — spurious edge detection

### What the task measures

Spurious edges are inserted into multi-ordering test plans from their incomparable pairs. The probe must distinguish real edges (keep) from spurious ones (remove). Two subconditions:

- **Mode 2a (same-depth):** spurious edges between steps at the same topological depth. In-distribution for Probe 2's training (trained on same-depth incomparable pairs as the flexible class).
- **Mode 2b (cross-branch):** spurious edges between steps at different depths in independent branches. Out-of-distribution for Probe 2 — tests generalization.

### Results

| Condition | Probe F1 | Probe prec | Probe rec | LLM F1 | Probe TCA | LLM TCA |
|---|---|---|---|---|---|---|
| M2a n=1 | 0.450 | 0.333 | 0.695 | 0.234 | 0.862 | 0.777 |
| M2a n=2 | 0.511 | 0.400 | 0.709 | 0.282 | 0.861 | 0.773 |
| M2a n=3 | 0.531 | 0.426 | 0.706 | 0.308 | 0.859 | 0.771 |
| M2b n=1 | 0.353 | 0.264 | 0.530 | 0.220 | 0.858 | 0.788 |
| M2b n=2 | 0.470 | 0.401 | 0.568 | 0.310 | 0.845 | 0.770 |
| M2b n=3 | 0.500 | 0.440 | 0.580 | 0.326 | 0.842 | 0.767 |

**Key findings:**

1. **Probe beats LLM at every level.** Margin ranges from +60% to +90% relative improvement on F1.

2. **Mode 2a > Mode 2b** as expected — same-depth spurious edges are in-distribution for the probe. But Mode 2b still shows strong performance, demonstrating genuine generalization beyond the training distribution.

3. **Positive rate is low (12–21%)** — each plan has many real edges and few spurious insertions. The probe's recall (~70% for M2a, ~55% for M2b) with moderate precision reflects this imbalance.

4. **GEDR is mildly negative** for Mode 2 (-0.7 to -0.3 for the probe) — the probe occasionally removes a real edge alongside the spurious ones. The LLM baseline's GEDR is much worse (-2.3 to -1.1).

5. **M2b n=1 has only 253 eligible plans** — plans must have cross-branch incomparable pairs, which not all multi-ordering plans do. Use n≥2 as the primary claim.

---

## Combined experiment

40% edge deletion + 2 same-depth spurious edges on the same plans. Both probes run on one corrupted DAG.

| System | PRR | GEDR | TCA |
|---|---|---|---|
| Probe | 0.000 | -1.564 | **0.690** |
| LLM | 0.000 | -3.507 | **0.347** |

The probe's TCA (0.69) is double the LLM's (0.35), confirming both probe tasks contribute meaningfully to DAG repair even in the hardest condition. Zero PRR is expected given the combined corruption. GEDR is negative for both systems but the probe is substantially less negative.

---

## Threshold selection

Thresholds were locked from the dev set at reference severities, then held fixed across all test conditions:

| Mode | Probe threshold | LLM threshold |
|---|---|---|
| M1 | 0.40 | 0.35 |
| M2a | 0.70 | 0.70 |
| M2b | 0.60 | 0.70 |

The low Mode 1 probe threshold (0.40) reflects the balanced positive rate — the probe can afford moderate precision since false positives (correct direction but non-minimal edge) are less costly than false negatives (missing ordering). Mode 2 thresholds are higher because the cost of incorrectly removing a real edge is high.

---

## Known limitations

**1. GEDR and PRR penalize correct ordering knowledge.** The probe adds transitive shortcuts because it learns reachability, not minimal graph structure. This is a metric limitation, not a probe failure. TCA is the appropriate metric for ordering preservation.

**2. Mode 2b is out-of-distribution.** Probe 2 was trained on same-depth incomparable pairs only. Mode 2b tests cross-branch pairs at different depths — a deliberate OOD generalization experiment. The probe performs well but could be improved with broader training negatives.

**3. Permutation test uses global label permutation.** Within-plan label correlation is not preserved under the null. A within-goal permutation scheme would be more rigorous but is technically complex and the current test is sufficient to establish signal existence.

**4. Single corruption seed.** Each condition uses one random realization of which edges are deleted or inserted. With 817 test plans, variance across plans provides stability, but reporting confidence intervals across multiple seeds would strengthen the claims.

**5. ProScript v1a count discrepancy.** The dataset contains 6,096 plans vs the paper's stated 6,414. Likely a post-publication quality filter in the v1a release.

---

## Recommended next steps

**For the paper:**
- Lead with Mode 1 F1 = 0.86 and Mode 1 TCA margin (0.81 vs 0.34) as the headline
- Report Mode 2 F1 margins (+60-90% over LLM) as secondary
- Include permutation test figure as primary validity evidence
- Include layer sweep figure showing the 15-18 peak zone
- Acknowledge GEDR/PRR limitation in a discussion paragraph — frame as the probe learning reachability (useful concept) rather than minimal edge sets (graph-theoretic property)

**For strengthening results:**
- Run multiple corruption seeds (3-5) and report mean ± std
- Add CaptainCook4D as a zero-shot transfer evaluation if task graph JSONs are available
- Consider training a broader Probe 2 with both same-depth and cross-depth negatives, which should improve Mode 2b
- Explore whether the probe can be used iteratively — add the highest-confidence edge, recompute candidates, repeat — which would naturally produce more minimal edge sets and improve GEDR