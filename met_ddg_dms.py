#!/usr/bin/env python3
"""WP3 completion: ThermoMPNN ΔΔG ↔ DMS abundance/uptake by SPT class.

Planned in met_af_plan.md §13 WP3 ("군별 AM↔DMS 상관, ΔΔG↔DMS 상관").
Not a new P-number — P1–P4 stay locked. GREY is exploratory.

Expect ρ(ΔΔG, GFP) < 0 (destabilizing → lower abundance).
Compare |ρ| CORE vs EXPOSED with residue-clustered bootstrap (same as P2).

    $MET_PY met_ddg_dms.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from met_dms import residue_bootstrap_rho

MET_HDD = Path(os.environ.get("MET_HDD", str(Path(__file__).resolve().parent / "data")))
MET_SPT = Path(os.environ.get("MET_SPT", str(MET_HDD / "spt")))


def fmt_rho(label, r):
    if r is None:
        print(f"  {label:8s} insufficient data")
        return
    print(f"  {label:8s} ρ={r['rho']:+.3f}  95% CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  "
          f"n_res={r['n_res']} n_var={r['n_var']}")


def delta_abs(a, b):
    if a is None or b is None:
        return None
    point = abs(a["rho"]) - abs(b["rho"])
    n = min(len(a["boots"]), len(b["boots"]))
    d = np.abs(a["boots"][:n]) - np.abs(b["boots"][:n])
    d = d[np.isfinite(d)]
    lo, hi = np.quantile(d, [0.025, 0.975]) if len(d) else (np.nan, np.nan)
    return {"point": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_le0": float((d <= 0).mean()) if len(d) else 1.0}


def main():
    dms = pd.read_csv(MET_SPT / "wp3_validation_missense.tsv", sep="\t")
    ddg = pd.read_csv(MET_SPT / "wp3_p3_thermompnn_variants.tsv", sep="\t")
    ddg["hgvs_short"] = ddg["wt"].astype(str) + ddg["pos"].astype(int).astype(str) + ddg["mut"].astype(str)
    df = dms.merge(ddg[["hgvs_short", "ddg"]], on="hgvs_short", how="inner")
    print(f"validation missense ∩ ThermoMPNN: {len(df)}  "
          f"(DMS {len(dms)}, ΔΔG {ddg['hgvs_short'].nunique()})")
    print("  by class:", df["class"].value_counts().to_dict())
    print(f"  overall median ΔΔG={df['ddg'].median():+.3f}  GFP={df['GFP_score'].median():+.3f}")

    print("\n=== Spearman ΔΔG vs GFP abundance ===")
    rhos_gfp = {}
    for lab in ("CORE", "EXPOSED", "GREY"):
        sub = df.loc[df["class"] == lab, ["pos", "ddg", "GFP_score"]].dropna()
        rhos_gfp[lab] = residue_bootstrap_rho(sub, "ddg", "GFP_score")
        fmt_rho(lab, rhos_gfp[lab])
    d_ce = delta_abs(rhos_gfp["CORE"], rhos_gfp["EXPOSED"])
    d_ge = delta_abs(rhos_gfp["GREY"], rhos_gfp["EXPOSED"])
    d_gc = delta_abs(rhos_gfp["GREY"], rhos_gfp["CORE"])
    print(f"  Δ|ρ| CORE−EXPOSED = {d_ce['point']:+.3f}  CI [{d_ce['ci_lo']:+.3f}, {d_ce['ci_hi']:+.3f}]  "
          f"P(Δ≤0)={d_ce['p_le0']:.3f}")
    print(f"  Δ|ρ| GREY−EXPOSED = {d_ge['point']:+.3f}  CI [{d_ge['ci_lo']:+.3f}, {d_ge['ci_hi']:+.3f}]  "
          f"(exploratory)")
    print(f"  Δ|ρ| GREY−CORE    = {d_gc['point']:+.3f}  CI [{d_gc['ci_lo']:+.3f}, {d_gc['ci_hi']:+.3f}]  "
          f"(exploratory)")

    print("\n=== Spearman AM vs GFP (same rows, for comparison) ===")
    rhos_am = {}
    for lab in ("CORE", "EXPOSED", "GREY"):
        sub = df.loc[df["class"] == lab, ["pos", "am_pathogenicity", "GFP_score"]].dropna()
        rhos_am[lab] = residue_bootstrap_rho(sub, "am_pathogenicity", "GFP_score")
        fmt_rho(lab, rhos_am[lab])

    print("\n=== |ρ_ΔΔG| vs |ρ_AM| within class ===")
    within = {}
    for lab in ("CORE", "EXPOSED", "GREY"):
        g, a = rhos_gfp[lab], rhos_am[lab]
        if g is None or a is None:
            continue
        diff = abs(g["rho"]) - abs(a["rho"])
        n = min(len(g["boots"]), len(a["boots"]))
        db = np.abs(g["boots"][:n]) - np.abs(a["boots"][:n])
        db = db[np.isfinite(db)]
        lo, hi = np.quantile(db, [0.025, 0.975]) if len(db) else (np.nan, np.nan)
        within[lab] = {"delta_abs": float(diff), "ci_lo": float(lo), "ci_hi": float(hi)}
        print(f"  {lab:8s} |ρ_ΔΔG|−|ρ_AM| = {diff:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]")

    print("\n=== secondary: Spearman ΔΔG vs SM73_0 uptake ===")
    rhos_up = {}
    for lab in ("CORE", "EXPOSED", "GREY"):
        sub = df.loc[df["class"] == lab, ["pos", "ddg", "SM73_0_score"]].dropna()
        rhos_up[lab] = residue_bootstrap_rho(sub, "ddg", "SM73_0_score")
        fmt_rho(lab, rhos_up[lab])

    def pack(r):
        if r is None:
            return None
        return {k: (None if k == "boots" else r[k]) for k in r}

    out = {
        "n_merged": int(len(df)),
        "rho_ddg_gfp": {k: pack(v) for k, v in rhos_gfp.items()},
        "rho_am_gfp": {k: pack(v) for k, v in rhos_am.items()},
        "rho_ddg_sm73": {k: pack(v) for k, v in rhos_up.items()},
        "delta_abs_ddg_gfp_CORE_minus_EXPOSED": d_ce,
        "delta_abs_ddg_gfp_GREY_minus_EXPOSED": d_ge,
        "delta_abs_ddg_gfp_GREY_minus_CORE": d_gc,
        "within_class_ddg_minus_am": within,
        "note": "WP3 planned ΔΔG↔DMS; not a new pre-registered P. GREY exploratory. Thresholds unchanged.",
    }
    MET_SPT.mkdir(parents=True, exist_ok=True)
    df.to_csv(MET_SPT / "wp3_ddg_dms_merged.tsv", sep="\t", index=False)
    (MET_SPT / "wp3_ddg_dms.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\noutputs -> {MET_SPT}")


if __name__ == "__main__":
    main()
