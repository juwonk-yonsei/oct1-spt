#!/usr/bin/env python3
"""WP3: SPT class vs OCT1 DMS + AlphaMissense (pre-registered P1, P2, P4).

Gold standard: Yee et al. Mol Cell 2024, odcambc/OCT1_DMS oct1_combined_scores.csv
  GFP_score   ~ abundance / surface expression
  SM73_*_score ~ substrate uptake (survivability)

Design-set positions 61, 88, 401, 420, 465 are EXCLUDED (rule-development only).

P3 (ΔΔG) is not run here — no stability tool installed yet.

    $MET_PY met_dms.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MET_HDD = Path(os.environ.get("MET_HDD", str(Path(__file__).resolve().parent / "data")))
MET_DMS = Path(os.environ.get("MET_DMS", str(MET_HDD / "dms")))
MET_AM = Path(os.environ.get("MET_AM", str(MET_HDD / "alphamissense")))
MET_SPT = Path(os.environ.get("MET_SPT", str(MET_HDD / "spt")))

DESIGN_POS = {61, 88, 401, 420, 465}
N_BOOT = 10_000
RNG = np.random.default_rng(20260812)
HGVS_RE = re.compile(r"p\.\(([A-Z])(\d+)([A-Z])\)")


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    prev = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, p * (m - i)))
        out[k] = adj
        prev = adj
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    r, _ = stats.spearmanr(x, y)
    return float(r)


def residue_bootstrap_rho(df: pd.DataFrame, xcol: str, ycol: str, n=N_BOOT):
    """Spearman ρ with residue-clustered bootstrap CI (numpy, not pandas concat)."""
    sub = df[["pos", xcol, ycol]].dropna()
    if sub["pos"].nunique() < 5:
        return None
    pos = sub["pos"].to_numpy()
    x = sub[xcol].to_numpy(dtype=float)
    y = sub[ycol].to_numpy(dtype=float)
    order = np.argsort(pos, kind="mergesort")
    pos, x, y = pos[order], x[order], y[order]
    starts = np.r_[0, np.flatnonzero(pos[1:] != pos[:-1]) + 1]
    ends = np.r_[starts[1:], pos.size]
    gx = [x[s:e] for s, e in zip(starts, ends)]
    gy = [y[s:e] for s, e in zip(starts, ends)]
    n_res = len(gx)
    n_var = int(pos.size)

    def rho_idx(idx):
        return _spearman(np.concatenate([gx[i] for i in idx]),
                         np.concatenate([gy[i] for i in idx]))

    all_idx = np.arange(n_res)
    point = rho_idx(all_idx)
    draws = RNG.integers(0, n_res, size=(n, n_res))
    boots = np.array([rho_idx(d) for d in draws], dtype=float)
    boots = boots[np.isfinite(boots)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"rho": point, "ci_lo": float(lo), "ci_hi": float(hi), "n_res": n_res,
            "n_var": n_var, "n_boot": len(boots), "boots": boots}


def main():
    spt = pd.read_csv(MET_SPT / "oct1_af2_rank1_spt.tsv", sep="\t")
    spt["pos"] = spt["pos"].astype(int)
    class_of = dict(zip(spt["pos"], spt["class"]))

    dms = pd.read_csv(MET_DMS / "oct1_combined_scores.csv")
    am = pd.read_csv(MET_AM / "by_target/SLC22A1_O15245.tsv", sep="\t")
    am["protein_variant"] = am["protein_variant"].astype(str)

    # --- QC: score scale from synonymous (not used for class comparison) ---
    syn = dms.loc[dms["mutation_type"] == "S", "GFP_score"].dropna()
    print("=== QC: synonymous GFP_score (null for abundance) ===")
    print(f"  n={len(syn)}  median={syn.median():.3f}  mean={syn.mean():.3f}  sd={syn.std():.3f}")
    loss_cut = float(syn.mean() - 2 * syn.std())
    print(f"  loss threshold (mean_syn - 2 sd) = {loss_cut:.3f}")

    miss = dms[dms["mutation_type"] == "M"].copy()
    miss = miss[miss["is.wt"] == False]
    miss["pos"] = miss["pos"].astype(int)
    miss = miss[miss["pos"].between(1, 554)]
    miss["wt_aa"] = miss["wt_pos"].astype(str)
    miss["mut_aa"] = miss["variants"].astype(str)
    # drop rows where letters are not single AA codes
    miss = miss[miss["wt_aa"].str.fullmatch(r"[A-Z]") & miss["mut_aa"].str.fullmatch(r"[A-Z]")]
    miss["hgvs_short"] = miss["wt_aa"] + miss["pos"].astype(str) + miss["mut_aa"]
    miss["class"] = miss["pos"].map(class_of)
    miss = miss.dropna(subset=["class", "GFP_score"])
    miss = miss.merge(
        am.rename(columns={"protein_variant": "hgvs_short"}),
        on="hgvs_short", how="left",
    )

    n_design = miss["pos"].isin(DESIGN_POS).sum()
    val = miss[~miss["pos"].isin(DESIGN_POS)].copy()
    print(f"\nmissense with GFP: {len(miss)}  (design-pos rows excluded: {n_design})")
    print(f"validation missense: {len(val)}")
    print("  by class:", val["class"].value_counts().to_dict())
    print(f"  AM annotated: {val['am_pathogenicity'].notna().sum()}")

    # ---------- P1: CORE abundance lower than EXPOSED ----------
    # primary: per-residue median GFP
    res_med = (val.groupby(["pos", "class"], as_index=False)["GFP_score"]
               .median().rename(columns={"GFP_score": "gfp_median"}))
    core_r = res_med.loc[res_med["class"] == "CORE", "gfp_median"]
    exp_r = res_med.loc[res_med["class"] == "EXPOSED", "gfp_median"]
    u_stat, p1_res = stats.mannwhitneyu(core_r, exp_r, alternative="two-sided")
    # also variant-level
    core_v = val.loc[val["class"] == "CORE", "GFP_score"]
    exp_v = val.loc[val["class"] == "EXPOSED", "GFP_score"]
    _, p1_var = stats.mannwhitneyu(core_v, exp_v, alternative="two-sided")
    p1_dir_ok = core_r.median() < exp_r.median()

    print("\n=== P1  CORE vs EXPOSED DMS abundance (GFP) ===")
    print(f"  residue-median GFP: CORE n={len(core_r)} median={core_r.median():.3f}  "
          f"EXPOSED n={len(exp_r)} median={exp_r.median():.3f}")
    print(f"  Mann-Whitney (residue) U={u_stat:.1f}  p={p1_res:.4g}  "
          f"direction CORE<EXPOSED: {p1_dir_ok}")
    print(f"  variant-level GFP: CORE n={len(core_v)} median={core_v.median():.3f}  "
          f"EXPOSED n={len(exp_v)} median={exp_v.median():.3f}  p={p1_var:.4g}")
    p1_pass = bool(p1_dir_ok and p1_res < 0.05)

    # ---------- P2: |Spearman(AM, GFP)| larger in CORE ----------
    def class_rho(label):
        sub = val.loc[val["class"] == label, ["pos", "am_pathogenicity", "GFP_score"]].dropna()
        return residue_bootstrap_rho(sub, "am_pathogenicity", "GFP_score")

    print("\n=== P2  Spearman AM pathogenicity vs GFP abundance ===")
    rhos = {}
    for lab in ("CORE", "EXPOSED", "GREY"):
        rhos[lab] = class_rho(lab)
        r = rhos[lab]
        if r is None:
            print(f"  {lab}: insufficient data")
            continue
        print(f"  {lab:8s} ρ={r['rho']:+.3f}  95% CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  "
              f"n_res={r['n_res']} n_var={r['n_var']}")

    rho_c = rhos["CORE"]["rho"] if rhos["CORE"] else np.nan
    rho_e = rhos["EXPOSED"]["rho"] if rhos["EXPOSED"] else np.nan
    dabs_point = abs(rho_c) - abs(rho_e)
    # independent class-wise bootstrap → Δ|ρ| distribution (pair by index; classes resampled separately)
    b_core = rhos["CORE"]["boots"] if rhos["CORE"] else np.array([])
    b_exp = rhos["EXPOSED"]["boots"] if rhos["EXPOSED"] else np.array([])
    n_pair = min(len(b_core), len(b_exp))
    dabs = np.abs(b_core[:n_pair]) - np.abs(b_exp[:n_pair])
    dabs = dabs[np.isfinite(dabs)]
    dabs_lo, dabs_hi = np.quantile(dabs, [0.025, 0.975]) if len(dabs) else (np.nan, np.nan)
    p2_sign_ok = rho_c < 0
    p2_mag_ok = bool(dabs_lo > 0)
    print(f"  Δ|ρ| = |ρ_CORE|-|ρ_EXPOSED| = {dabs_point:+.3f}  "
          f"95% CI [{dabs_lo:+.3f}, {dabs_hi:+.3f}]")
    print(f"  ρ_CORE < 0: {p2_sign_ok}   Δ|ρ| CI entirely > 0: {p2_mag_ok}")
    p2_pass = bool(p2_sign_ok and p2_mag_ok)

    # uptake (secondary, not a registered primary)
    print("\n=== secondary: Spearman AM vs SM73_0 uptake ===")
    for lab in ("CORE", "EXPOSED"):
        sub = val.loc[val["class"] == lab, ["pos", "am_pathogenicity", "SM73_0_score"]].dropna()
        r = residue_bootstrap_rho(sub, "am_pathogenicity", "SM73_0_score")
        if r:
            print(f"  {lab:8s} ρ={r['rho']:+.3f}  95% CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  "
                  f"n_res={r['n_res']}")

    # ---------- P4: AM-benign functional-loss variants enriched in EXPOSED ----------
    val["am_benign"] = val["am_class"] == "benign"
    val["dms_loss"] = val["GFP_score"] < loss_cut
    # literature loss (held-out list; design variants may appear but we still exclude by pos)
    lit = pd.read_csv(MET_DMS / "literature_variants.csv")
    lit["hgvs_short"] = lit["hgvs"].map(
        lambda s: (lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else None)(
            HGVS_RE.search(str(s))
        )
    )
    lit_loss = set(
        lit.loc[lit["literature_impact_function"].astype(str).str.contains("loss", case=False, na=False),
                "hgvs_short"].dropna()
    )
    val["lit_loss"] = val["hgvs_short"].isin(lit_loss)
    val["func_loss"] = val["dms_loss"] | val["lit_loss"]

    bg = val.dropna(subset=["am_class"])
    hit = bg[bg["am_benign"] & bg["func_loss"]]
    # enrichment of EXPOSED among AM-benign∩loss vs all missense background
    def frac_exposed(df):
        return float((df["class"] == "EXPOSED").mean()) if len(df) else np.nan

    # Fisher's exact: (benign∩loss, rest) × (EXPOSED, not EXPOSED)
    table = np.array([
        [((hit["class"] == "EXPOSED")).sum(), ((hit["class"] != "EXPOSED")).sum()],
        [((bg["class"] == "EXPOSED") & ~bg.index.isin(hit.index)).sum(),
         ((bg["class"] != "EXPOSED") & ~bg.index.isin(hit.index)).sum()],
    ], dtype=int)
    # simpler 2x2 on hit vs bg for EXPOSED membership
    table2 = np.array([
        [(hit["class"] == "EXPOSED").sum(), (hit["class"] != "EXPOSED").sum()],
        [(bg["class"] == "EXPOSED").sum(), (bg["class"] != "EXPOSED").sum()],
    ], dtype=int)
    # Use hit vs (bg minus hit) to avoid double-counting
    rest = bg.loc[~bg.index.isin(hit.index)]
    fisher_table = np.array([
        [(hit["class"] == "EXPOSED").sum(), (hit["class"] != "EXPOSED").sum()],
        [(rest["class"] == "EXPOSED").sum(), (rest["class"] != "EXPOSED").sum()],
    ], dtype=int)
    oddsr, p4 = stats.fisher_exact(fisher_table, alternative="greater")  # EXPOSED enrichment
    print("\n=== P4  AM-benign ∩ functional-loss  EXPOSED enrichment ===")
    print(f"  loss cutoff GFP < {loss_cut:.3f}  OR literature loss")
    print(f"  n AM-benign ∩ loss = {len(hit)}   frac EXPOSED = {frac_exposed(hit):.3f}")
    print(f"  background missense n={len(bg)}   frac EXPOSED = {frac_exposed(bg):.3f}")
    print(f"  class breakdown of hits:", hit["class"].value_counts().to_dict())
    print(f"  Fisher exact (greater EXPOSED in hits vs rest): OR={oddsr:.3f}  p={p4:.4g}")
    p4_pass = bool(p4 < 0.05 and frac_exposed(hit) > frac_exposed(bg))

    # Holm on P1 residue p, P2 (use one-sided bootstrap p for Δ|ρ|), P4
    # bootstrap p for P2: proportion of Δ|ρ| <= 0
    p2_boot_p = float((dabs <= 0).mean()) if len(dabs) else 1.0
    raw_p = {"P1": float(p1_res), "P2": p2_boot_p, "P4": float(p4)}
    adj = holm(raw_p)
    print("\n=== Holm-adjusted p (P1, P2, P4; P3 skipped) ===")
    for k in ("P1", "P2", "P4"):
        print(f"  {k}: raw={raw_p[k]:.4g}  Holm={adj[k]:.4g}")

    verdict = {
        "P1": {"pass": p1_pass, "p_raw": raw_p["P1"], "p_holm": adj["P1"],
               "core_median_gfp": float(core_r.median()),
               "exposed_median_gfp": float(exp_r.median())},
        "P2": {"pass": p2_pass, "p_raw": raw_p["P2"], "p_holm": adj["P2"],
               "rho_CORE": rho_c, "rho_EXPOSED": rho_e,
               "delta_abs_rho": dabs_point, "delta_abs_ci": [dabs_lo, dabs_hi]},
        "P3": {"pass": None, "note": "not run — ΔΔG tool not installed"},
        "P4": {"pass": p4_pass, "p_raw": raw_p["P4"], "p_holm": adj["P4"],
               "n_hit": int(len(hit)), "frac_exposed_hit": frac_exposed(hit),
               "frac_exposed_bg": frac_exposed(bg), "odds_ratio": float(oddsr)},
        "n_validation_missense": int(len(val)),
        "loss_cutoff_gfp": loss_cut,
        "design_pos_excluded": sorted(DESIGN_POS),
    }
    print("\n=== pre-registered verdict ===")
    for k in ("P1", "P2", "P3", "P4"):
        v = verdict[k]["pass"]
        flag = "PASS" if v is True else ("SKIP" if v is None else "FAIL")
        print(f"  {k}: {flag}")

    MET_SPT.mkdir(parents=True, exist_ok=True)
    (MET_SPT / "wp3_p1_p2_p4.json").write_text(json.dumps(verdict, indent=2) + "\n")

    # save per-variant table for figures (validation set only)
    out_cols = ["hgvs_short", "pos", "wt_aa", "mut_aa", "class",
                "GFP_score", "SM73_0_score", "SM73_1_score",
                "am_pathogenicity", "am_class", "dms_loss", "lit_loss", "func_loss"]
    val[out_cols].to_csv(MET_SPT / "wp3_validation_missense.tsv", sep="\t", index=False)
    res_med.to_csv(MET_SPT / "wp3_residue_median_gfp.tsv", sep="\t", index=False)
    print(f"\noutputs -> {MET_SPT}")

    if not (p1_pass and p2_pass):
        # P2 is the paper's core claim; exit 0 still so the log is saved.
        print("NOTE: one or more primary hypotheses did not pass. See verdict JSON.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
