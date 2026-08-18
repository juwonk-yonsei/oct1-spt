# I-track — “don’t” is shown; lock the “instead”

Locked: **2026-08-18**, before any surface-normalized wet-lab readout.
**I1 primary is computational I1C** (`met_prereg_instead_dry.md`). Wet-lab I1W is optional.
Does **not** retune SPT 10%/30%. Does **not** add Ridge/ANM features. Not C7/C8.
Does **not** reopen P2. Does **not** treat SM73_1 as a second substrate.

MS1 (TPJ draft) may keep the **don’t** results and the DMS **class recovery** (P1/P3/P4).
I1C, if it passes, may add a **DMS-proxy** instead (GFP ≠ surface). It may **not** claim
that cell-surface or trafficking assays have been run.

This file is the confirmatory plan that would make the second sentence a result.

---

## The gap (why a new track)

Shown (don’t / DMS classes):

| Don’t | Evidence |
|-------|----------|
| AM as first-line at EXPOSED | P4 |
| “Trust AM if buried” | P2 FAIL |
| AM as uptake score | AM–SM73 ρ ≈ 0 |
| Point-mutant AF2 RMSD as collapse | design set < 3.284 Å |
| Arbitrary/outward AF2 as SPT reference | AF2 vs 8ET6 FAIL; vs 8SC1 81.7% |

Shown (positive on the **same DMS GFP/ΔΔG**, not on the next assay):

- CORE is an abundance/stability **class** (P1, P3).
- EXPOSED is an AM-blind abundance-loss **class** (P4).

**Not shown:** doing the Table 3 assays (surface, trafficking, thermal, uptake/surface)
recovers those classes. Yee GFP is total protein. R5 is locked but **not run**.

Forbidden “instead” (already computed or already failed; do not rebrand):

- ΔΔG ranks GFP better in CORE than EXPOSED — `wp3_ddg_dms.json` Δ\|ρ\| CI includes 0.
- ΔΔG out-ranks AM inside CORE — same file, CORE Δ\|ρ\|(ΔΔG−AM) CI includes 0 / AM wins.
- Residual ρ toward 0.3, ANM-conditioned panels (S6 superseded), GREY as a trust-AM bin.

The instead that can still be true is **not a better scalar scorer**.
It is: **SPT predicts the (surface, uptake/surface) phenotype type; AM does not.**

---

## Phenotype types (gold, wet-lab only)

Same wells, vs WT biological-replicate mean ± 2 SD (not vs DMS GFP).

| Type | Surface S | Uptake/surface U | Meaning |
|------|-----------|------------------|---------|
| **Stab** | low | not low | delivery/abundance; per-site transport OK |
| **Trans** | not low | low | transport-specific |
| **Mixed** | low | low | both |
| **WT** | not low | not low | assay-negative |

Assay (collaborator; lock before first clone is scored):

1. Surface OCT1: ECD antibody or a surface-exposed tag. Not total GFP alone.
2. Organic-cation uptake on the **same** wells (MPP+ or metformin preferred; SM73 analog allowed if metformin unavailable).
3. U = uptake / surface. Always compute U; never interpret raw uptake as transport.

Optional later (I3 only): thermal/CHX for Stab; biotinylation/microscopy for Stab at EXPOSED.
I1/I2 do **not** require those.

---

## Predicted type (SPT policy) — locked before wet-lab

Use the **R5 panel_class**, not a new inclusion rule.

| panel_class | SPT instead-prediction |
|-------------|------------------------|
| `abundance_loss_resid_ok` | **Stab** |
| `exposed_am_benign_gfp_loss` | **Stab** |
| `dms_resid_loss_gfp_ok` | **Trans** |
| `near_wt_control` | **WT** |
| `literature_exposed_loss` | report only; not in pass counts |

AM stop-rule (current practice, comparator):

- AM benign (`am_pathogenicity` < 0.34) → call **WT** (do not assay / treat as fine).
- AM pathogenic → call **damaging** with **no type** (Stab vs Trans unspecified).

AM cannot predict Stab vs Trans. That is the instead: SPT returns a type; AM returns a scalar.

---

## I1 — R5 pilot (already named; do not rename)

Panel: `challenge/r_residual/r5_experiment_panel.tsv` (16 discovery + 3 literature).
Cuts and names stay as R5. Do not drop a variant after seeing S or U.

Pass (same as R5, restated as types):

| ID | Class | Pass |
|----|-------|------|
| I1.1 | `dms_resid_loss_gfp_ok` | ≥3/4 **Trans** (not Stab, not WT) |
| I1.2 | `abundance_loss_resid_ok` | ≥3/4 **Stab** |
| I1.3 | `exposed_am_benign_gfp_loss` | ≥3/4 **Stab** (surface low; U not required to be low) |
| I1.4 | `near_wt_control` | ≥3/4 **WT** |

I1.3 is **not** a new discovery that AM misses EXPOSED loss (P4 already). It tests that
Yee GFP-loss at those sites is a **real surface** defect, not a total-GFP artifact.

Go / stop:

- I1.4 fail → assay invalid; stop I-track (do not expand).
- I1.2 and I1.3 pass → CORE-Stab and EXPOSED-surface **instead** is provisionally true → I2 allowed for those claims.
- I1.1 fail, I1.2+I1.3 pass → drop transport-specific instead; keep Stab instead; I2 omits residual class.
- I1.2 or I1.3 fail → that instead-claim is false on this panel; do not rewrite Table 3 as if it passed.

I1 is a pilot (n=4/class). A paper titled “instead this works” needs I2 or an explicit
pilot limitation.

---

## I2 — disjoint replication (lock rules now; do **not** pick names until I1 go)

Start I2 only after I1 go above. Same inclusion rules as R5. Positions **not** in R5.
Same unique-`pos` / cluster cap. Do not retune GFP −0.814 or AM 0.34.

Pool remaining after R5 (train_ok, before new uniqueness): residual 104−4, abundance 58−4,
EXPOSED AM-benign GFP-loss **13−4 = 9** (bottleneck), near-WT 195−4.

| Class | I2 n | Note |
|-------|------|------|
| `abundance_loss_resid_ok` | 8 | enough pool |
| `dms_resid_loss_gfp_ok` | 8 | skip if I1.1 failed |
| `exposed_am_benign_gfp_loss` | **all remaining after R5 uniqueness** (target 8, expect ≤9) | do not relax AM/GFP cuts to fill |
| `near_wt_control` | 8 | |

If EXPOSED remaining < 6 after uniqueness, I2.3 is **underpowered**; report as replication
of I1.3 with whatever n remains, not a new cut.

Pass: same ≥75% type-match per class (n=8 → ≥6/8; n=k → ≥ ceil(0.75k)).
Holm within I2 class tests. Literature still excluded from counts.

Primary I2 (instead vs AM, non-type): among gold-damaging (Stab or Trans or Mixed),
AM stop-rule false-negative rate (AM benign) is reported. For `exposed_am_benign_gfp_loss`
this FN is **1 by construction** if they are gold-damaging — do not sell that as news.
The news is the gold type (Stab vs GFP artifact).

Primary I2 (instead that is not circular): **type accuracy** of the SPT table above
vs two nulls, on I1+I2 discovery variants pooled (pre-specify):

1. Null always-Stab.
2. Null always-WT.

McNemar: SPT type vs always-Stab, two-sided α=0.05. Must beat always-Stab **and**
always-WT on the pooled discovery set. If the panel is mostly Stab classes, always-Stab
is a hard null — that is intended. Trans class is what makes SPT beat always-Stab.

---

## I3 — mechanism (only if the matching I1/I2 class passed)

Not required for the instead headline.

- Stab (CORE `abundance_loss`): CHX chase or thermal/protease sensitivity vs near-WT.
  Pass: ≥75% more labile or lower surface than near-WT on the same run.
- Stab (EXPOSED AM-benign): surface biotinylation and/or ER vs plasma-membrane marker.
  Pass: ≥75% surface-low with intracellular retention or lower biotinylation.
- Do **not** add PTM as the EXPOSED instead. Uddin Y240F/Y361F/Y376F are CORE/GREY.
  A PTM panel would be a separate lock (those three + matched WT), not I3.

---

## I4 — metformin (independent substrate, not SM73_1)

Same clones as I1 (or I2 if I1 substrate was not metformin).
Score S and U with **metformin** uptake.

Pass: for each I1 class that passed, ≥75% keep the **same type** as on the I1 substrate.
Fail: types flip → instead is substrate-specific; do not claim OCT1-PGx transport in general.

SM73_1 remains banned as replication (anti-GFP / orthogonal to SM73_0).

---

## What a second paper may claim if gates pass

If I1.2+I1.3 (+ I2 for those classes) pass:

**Don’t** use AM/AF2 as a uniform OCT1 missense score (MS1).
**Instead** measure surface and uptake/surface, with SPT choosing the expected type:
CORE abundance-loss variants are Stab; EXPOSED AM-benign GFP-loss variants are Stab
(AM would have stopped); GFP-ok residual-loss variants are Trans (only if I1.1/I2.1 pass).

If I4 passes, the type is not an SM73-cytotoxicity artifact.

Still not a clinical guideline. Still not an uptake predictor with ρ ≈ 0.3.

---

## Out of this track

C7 transcript module, C8 UKB, ProteinGym SOTA, TPT, Multimer, SLC map,
retuning SPT, renaming R5 after seeing S/U, S6 ANM panel, GREY trust-AM policy.
