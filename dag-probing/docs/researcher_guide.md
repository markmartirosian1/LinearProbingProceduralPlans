# Researcher Guide — Diagnostic Analysis and Interpretation

## Overview

This guide covers the full result set from the two committed notebooks
(`probe_training_and_permutation_v2.ipynb`, `probe_classification_v2.ipynb`),
explains what each finding means, identifies known limitations, and suggests
next steps. All numbers below are taken directly from the notebooks' saved
execution outputs and the committed CSV/JSON/PKL artifacts, cross-checked
against each other for consistency.

**A note on scope:** an earlier version of this guide described a DAG-level
repair evaluation (synthetic edge deletion at 20/40/60% severity, spurious
edge insertion at n=1/2/3, and TCA/GEDR/PRR reconstruction metrics). That
methodology does not exist in either notebook — there is no corruption code,
no severity sweep, and no TCA/GEDR/PRR computation anywhere in this
repository. This guide describes only what was actually run: a single
dev-threshold-locked, pairwise classification evaluation on the unmodified
test DAGs. If the DAG-repair evaluation is still a goal for the paper, it
needs to be built — see "Recommended next steps."

---

## Permutation tests — the validity foundation

Both probes show signal well above chance at their selected layers.

| Probe | Layer | Real CV AUC | Permuted mean | p-value |
|---|---|---|---|---|
| Probe 1 (direction) | 16 | 0.836 | 0.499 | 0.032 |
| Probe 2 (necessity) | 15 | 0.841 | 0.500 | 0.032 |

**Important — this is a different number from the layer-sweep dev AUC.**
The permutation test's "real AUC" (0.836 / 0.841) is computed via **3-fold
GroupKFold cross-validation on a subsample (≤5,000 rows) of the training
set**. The layer-sweep AUC reported in the next section (0.900 / 0.855) is
computed by **fitting on the full training set and scoring on the held-out
dev set**. These measure related but distinct things — training-set CV
generalization vs. train→dev transfer — and shouldn't be quoted
interchangeably. (The previous version of this guide reported 0.824/0.881,
which matches neither number and appears to be from an earlier or unrelated
run.)

**Methodological notes:**

- **30 permutations, not 100.** `cfg['n_perms'] = 30` in the actual config,
  confirmed by both `permutation_aucs_probe1.csv` and
  `permutation_aucs_probe2.csv` each containing exactly 30 rows. The minimum
  reportable p-value at 30 permutations is 1/31 ≈ 0.032 (computed as
  `(1 + count) / (n_perms + 1)`), not <0.01. Both probes hit this floor: 0 of
  30 permuted runs reached the real AUC.
- The test uses **global label permutation** with plan-level GroupKFold CV
  (3 folds). Global permutation is a standard approximation; a strictly
  correct null would preserve within-plan label structure. The test confirms
  hidden states carry genuine label-correlated signal at the selected
  layers — it does not by itself validate the layer-sweep dev AUC or the
  test-set F1 numbers reported elsewhere in this guide.
- The test is run only at the fixed layer selected on the dev set. It does
  not test whether a significant layer exists across the full sweep range
  (which would require a max-statistic permutation test).

---

## Layer sweep — where ordering knowledge lives

Full sweep results (dev AUC, layers 15–18):

| Layer | Probe 1 AUC | Probe 1 macro | Probe 2 AUC | Probe 2 macro |
|---|---|---|---|---|
| 15 | 0.8941 | 0.9086 | **0.8548** | **0.8567** |
| 16 | **0.9000** | **0.9116** | 0.8541 | 0.8546 |
| 17 | 0.8984 | 0.9111 | 0.8537 | 0.8456 |
| 18 | 0.8973 | 0.9094 | 0.8513 | 0.8425 |

**Probe 1 selects layer 16** (probe AUC 0.9000, margin +0.3445 over the LLM
baseline AUC of 0.5555). **Probe 2 selects layer 15** (probe AUC 0.8548,
margin +0.2546 over LLM baseline AUC 0.6002) — layer 15 is the actual argmax;
layer 17 is third-best. Both probes converge in the 15–16 range rather than
the 15–18 range broadly — worth softening the "consensus zone" framing
somewhat, since the AUC differences across layers 15–18 are small (≤0.6pp for
Probe 1, ≤3.5pp for Probe 2) but layer 16→15 is a genuine, if modest,
displacement from what was previously documented.

**Training scale (from actual notebook output, not the previously documented
numbers):**

| Probe | Training rows (balanced) | Dev rows (all, layer-selection) |
|---|---|---|
| Probe 1 | 41,126 | 15,342 |
| Probe 2 | 6,672 | 7,954 |

(Previously documented as 40,562 / 6,536 — close but not exact; the current
numbers are read directly from the executed cell output and the underlying
CSVs, so they supersede the earlier figures.)

The probe pipeline is `StandardScaler → LogisticRegression`. A convergence
check runs after fitting each final probe; no convergence warnings appear in
the saved output for either probe.

---

## Held-out test-set evaluation

### What the task measures

No synthetic corruption is applied. Both probes are evaluated directly on
the original test DAGs, using dev-threshold-locked predictions, with the
**same dedicated prompt each probe was trained on** (so scoring requires two
separate forward passes per candidate pair — one per probe).

**Probe 1:** y=1 if (a, b) is a real edge; y=0 if (a, b) is the reverse of a
real edge or an incomparable pair. Evaluated on all 2,077 test plans (every
plan contributes reversed-edge negatives, even single-ordering ones).

**Probe 2:** y=1 (keep) if (a, b) is a real edge; y=0 (spurious) if (a, b) is
a same-depth or cross-branch incomparable pair, in either direction.
Evaluated only on the 817 multi-ordering test plans.

### Results

**Probe 1** (31,680 test pairs, 43.8% positive; threshold 0.50 probe / 0.30
LLM):

| System | F1 | Precision | Recall | Specificity |
|---|---|---|---|---|
| Probe | **0.776** | 0.752 | 0.802 | **79.4%** |
| LLM output | 0.606 | 0.437 | 0.984 | 1.5% |

**Probe 2** (10,095 test pairs, 60.9% positive; threshold 0.35 probe / 0.30
LLM):

| Condition | System | F1 | Precision | Recall | Specificity |
|---|---|---|---|---|---|
| Combined | Probe | **0.789** | 0.783 | 0.795 | **65.7%** |
| Combined | LLM | 0.757 | 0.610 | 0.996 | 0.9% |
| Same-depth (in-dist.) | Probe | **0.826** | 0.859 | 0.795 | **71.1%** |
| Same-depth (in-dist.) | LLM | 0.816 | 0.691 | 0.996 | 1.1% |
| Cross-branch (OOD) | Probe | 0.843 | 0.898 | 0.795 | **53.1%** |
| Cross-branch (OOD) | LLM | **0.911** | 0.839 | 0.996 | 0.5% |

**Key findings:**

1. **Probe 1 shows a clean, large margin** (+0.171 F1) driven mostly by
   specificity: the probe correctly rejects 79.4% of reversed/incomparable
   pairs, while the LLM baseline rejects only 1.5% — it is answering "yes"
   to nearly every direction question regardless of correctness, consistent
   with the yes-bias documented elsewhere in this project.

2. **Probe 2's combined and same-depth margins are real but modest**
   (+0.032 and +0.010 F1). The probe's specificity advantage is still large
   (65.7% vs 0.9% combined) but recall is capped at 0.795 for both subtypes,
   while the LLM's near-universal "yes" answers push its recall to 0.996 —
   this recall gap is what compresses the F1 margin relative to Probe 1.

3. **Cross-branch is the one condition where the LLM's F1 is higher**
   (0.911 vs 0.843, margin **−0.067**). This is not evidence the LLM
   generalizes better out-of-distribution — its specificity there is 0.5%,
   the worst of any condition tested (only 6 of 1,182 cross-branch spurious
   pairs correctly flagged). Because cross-branch has a large class
   imbalance (6,147 positive edges vs. 1,182 negative pairs), a
   near-blanket-"yes" system's F1 is dominated by its 0.996 recall on the
   majority class, inflating F1 despite negligible actual discrimination.
   The probe's specificity (53.1%) is markedly lower than on same-depth
   pairs (71.1%) — a genuine generalization gap, since Probe 2 was trained
   only on same-depth negatives — but it is still two orders of magnitude
   better than the LLM's. **F1 alone is a misleading metric on this
   condition; specificity/precision tell the more accurate story.**

4. **Recall is identical across subtypes within each system** (0.795 for the
   probe, 0.996 for the LLM, in both same-depth and cross-branch rows). This
   is expected: the positive-class (real-edge) predictions are unchanged
   across the two evaluations — only the negative-pool composition differs
   — so all the between-subtype variation in the table comes from precision
   and specificity, not recall.

---

## Threshold selection

Locked from the dev set via best-F1 sweep over {0.30, 0.35, ..., 0.75}, held
fixed for test evaluation:

| Probe | Probe threshold | LLM threshold |
|---|---|---|
| Probe 1 | 0.50 | 0.30 |
| Probe 2 (both subtypes) | 0.35 | 0.30 |

Probe 2 uses one combined threshold for both same-depth and cross-branch
evaluation — there is no separate per-subtype threshold in the current
pipeline.

---

## Known limitations

**1. F1 is a misleading headline metric on the cross-branch condition.** As
detailed above, the LLM's higher F1 there reflects extreme class imbalance
and near-universal positive predictions, not genuine out-of-distribution
generalization. Any paper claim about Probe 2's performance should report
specificity or precision alongside F1, particularly for this cell, and
should not claim "the probe beats the LLM in every condition" — it doesn't,
on raw F1.

**2. The LLM baseline shows severe yes-bias across every condition tested**
(specificity ranges 0.5%–1.5%). This makes the probe's margin over the LLM
partly a story about the LLM's near-degenerate behavior rather than purely
about the probe's competence. Both framings are defensible for the paper,
but they support different claims and shouldn't be conflated.

**3. Permutation test uses global label permutation.** Within-plan label
correlation is not preserved under the null. A within-goal permutation
scheme would be more rigorous but is technically complex; the current test
is sufficient to establish signal existence, not to validate downstream
test-set metrics.

**4. No variance estimates on test-set metrics.** Every F1/precision/recall
number above is a single point estimate from one run on one fixed test set.
The permutation test establishes that the *training-time* signal is real,
but there are no bootstrap CIs or repeated-seed estimates for the *test-set*
classification numbers reported in this section.

**5. Dataset split-count discrepancy.** `check_proscript_splits.py` reports
3,099/1,031/1,966 plans (train/dev/test, total 6,096); the actual notebook
run loaded 3,252/1,085/2,077 (total 6,414 — matching Sakaguchi et al.'s
stated dataset size exactly). These are likely two different local copies of
the ProScript v1a release. The multi-ordering subset counts (361 dev, 817
test) are consistent with what both notebooks actually used, but the total
counts should be reconciled before citing split sizes in the paper.

**6. Two different dev sets are used for two different purposes.** Notebook
1 builds a 7,954-row Probe 2 dev set for layer selection (from `build_rows`,
applied across all dev plans); notebook 2 builds a separate 4,210-row Probe 2
dev set for threshold locking (from `build_probe2_pairs`, restricted to
plans with incomparable pairs). Both are legitimate for their respective
purposes, but conflating them in write-ups would be inaccurate.

**7. No DAG-level reconstruction evaluation exists yet.** All current
metrics are pairwise (edge-level F1/precision/recall). There is no code that
reconstructs a repaired DAG from probe predictions or measures
transitive-closure agreement, graph edit distance, or perfect-reconstruction
rate — despite this being described in earlier planning material. If this
remains part of the paper's contribution, it needs to be built and run
before it can be reported.

---

## Recommended next steps

**For the paper:**
- Lead with Probe 1's F1 margin (+0.171) and specificity gap (79.4% vs 1.5%)
  as the cleanest headline result.
- Report Probe 2's combined and same-depth margins as secondary, positive
  results, and report the cross-branch cell honestly (probe loses on F1,
  wins substantially on specificity) with the class-imbalance explanation —
  this is a more defensible and more interesting finding than omitting it.
- Include the permutation test figure as validity evidence, with the
  corrected 30-permutation, p=0.032 framing.
- Include the layer sweep figure, and verify its annotation actually
  highlights layer 15 for Probe 2 (not 17) before using it.

**For strengthening results:**
- Reconcile the two dataset-count sources (script vs. notebook) before
  finalizing any split-size claims.
- Add bootstrap or repeated-sample confidence intervals for the test-set F1
  numbers.
- If the DAG-repair/TCA/GEDR/PRR evaluation is still intended, build it as a
  new notebook rather than continuing to describe it as already-completed
  work.
- Add CaptainCook4D as a zero-shot transfer evaluation if task graph JSONs
  are available — not yet present in this repository.
- Consider training a broader Probe 2 with both same-depth and cross-branch
  negatives, which should directly address the specificity drop observed
  out-of-distribution.