# Residual ceiling track (R1–R4) — prereg lock

Locked: **2026-08-18**, before running `met_r_residual.py`.

Not C7. Does not retune SPT 10%/30%. Does not change P1–P6 / C3 verdicts.
Goal: can GFP-residual SM73_0 Spearman be raised toward 0.20 (gate) / 0.30 (stretch)
without leaking helix identity or cherry-picking TM4.

## Frozen inputs

- C3 LOPO table: `$MET_HDD/challenge/c3_ensemble/oct1_ens_lopo.tsv`
- Yee scores: `$MET_DMS/oct1_combined_scores.csv` (primary) and `oct1_scores.csv` (version check)
- Structures: 8SC1, 8ET6 as in C3
- `train_ok` / design-position exclusion unchanged

## R1 — residual SNR / Spearman ceiling

On missense with finite GFP and SM73_0:

1. Spearman(GFP, SM73_0), Spearman(GFP, SM73_1), Spearman(SM73_0, SM73_1).
2. OLS `SM73_0 ~ GFP` on missense; residual SD on missense vs synonymous (`mutation_type==S`).
3. SE attenuation: `rel_y = 1 - mean(SM73_0_SE²) / var(SM73_0)`.
   Residual error `SE_r² ≈ SM73_0_SE² + b² GFP_SE²`; `rel_resid = 1 - mean(SE_r²)/var(resid)`.
4. Approximate Spearman ceiling on residual ≈ `sqrt(max(rel_resid, 0))`.

Interpret (locked):

- If ceiling **< 0.20** on all missense: 0.30 on the full set is not a model problem; stop promising it.
- If ceiling **< 0.30** but ≥ 0.20: 0.30 is a stretch; 0.20 is the honest gate.
- Synonymous residual SD is the noise floor, not a second biological replicate (SM73_1 is not a replicate).

## R2 — Yee column identity

Report mutation_type counts, WT/syn/missense means, and whether `oct1_scores.csv` GFP matches `combined`.
Do **not** treat SM73_1 as independent-substrate replication (already failed that test).

## R3 — locked subset, no helix cherry-pick

Structure-only gate (computed on unique TM positions, not on DMS ρ):

- `gate_disp` = 8SC1 vs 8ET6 Cα displacement after superposition (existing column).
- **GATE** positions: `topology == Transmembrane` and `gate_disp >= q75` among unique TM positions with finite `gate_disp`.

Abundance windows from synonymous GFP in `combined` (same rule as P1, not retuned):

- **not_loss:** `GFP_score >= mean(syn GFP) − 2 SD(syn GFP)`  (P1 loss cutoff inverted)
- **near_wt:** `|GFP_score − median(syn GFP)| <= SD(syn GFP)`

If combined GFP does not match C3 `GFP_score` (median |Δ| > 0.05 on overlapping missense), compute syn mean/sd from C3-table synonymous join; if synonymous cannot be joined, fall back to **not_loss = GFP_score >= −0.814** (already published P1 cutoff) and **near_wt = |GFP_score| <= 0.407** (half of 0.814). Write which branch was used.

Primary evaluation (no retrain): existing `ens_resid` vs `SM73_resid_fold` on `train_ok ∩ GATE ∩ not_loss`.

Secondary: same on `train_ok ∩ GATE ∩ near_wt`.

Report n, Spearman, residue-bootstrap 95% CI vs AM on the same rows.

**R3.1 gate (promising):** primary ρ ≥ 0.20 and Δρ vs AM CI_lo > 0.  
**R3.2 stretch:** primary ρ ≥ 0.30.  
Do not declare success from a single helix (e.g. TM4).

## R4 — mutation-specific features (run even if R3 fails; skip only if R1 ceiling < 0.05)

Add to C3 feature bundle (still no AM):

- `gate_x_dvol = gate_disp * d_volume`
- `gate_x_absdvol = gate_disp * |d_volume|`
- `gate_x_dcharge = gate_disp * abs_d_charge`
- `msf_x_dvol = anm_msf_8sc1 * d_volume`
- `msf_x_absdcharge = anm_msf_8sc1 * abs_d_charge`
- `lambda_steric = gate_disp * max(d_volume, 0)`

Helix-LOPO Ridge α=1, fold-wise SM73~GFP residual, `train_ok` only — same as C3.

**R4.1:** full-set ρ > C3 `rho_ens_resid` (0.0767) and Δρ vs C3 `ens_resid` CI_lo > 0.  
**R4.2:** primary GATE∩not_loss ρ ≥ 0.20.

If R4.1 fails, stop this computational track; do not add more features in the same run.
