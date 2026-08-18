#!/usr/bin/env python3
"""P3: ThermoMPNN ΔΔG vs SPT class (pre-registered in met_prereg.md).

Expect: destabilizing ΔΔG median CORE > EXPOSED (EXPOSED ≈ 0).
Design-set positions 61/88/401/420/465 excluded.

    $MET_PY met_p3.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MET_HDD = Path(os.environ.get("MET_HDD", str(Path(__file__).resolve().parent / "data")))
MET_SPT = Path(os.environ.get("MET_SPT", str(MET_HDD / "spt")))
MET_DDG = Path(os.environ.get("MET_DDG", str(MET_HDD / "ddg")))

DESIGN_POS = {61, 88, 401, 420, 465}
# ThermoMPNN single-mutant strings are WT + chain + pos + mut (e.g. MA1C);
# after some renumber paths they drop the chain (M1C). Accept both.
MUT_RE = re.compile(r"^([A-Z])[A-Z]?(\d+)([A-Z])$")


def load_ddg():
    hits = sorted(MET_DDG.glob("oct1_af2_wt_thermompnn*.csv")) + \
           sorted(MET_DDG.glob("*thermompnn*.csv")) + \
           sorted(MET_SPT.glob("*thermompnn*.csv"))
    if not hits:
        raise SystemExit(f"no ThermoMPNN csv under {MET_DDG} or {MET_SPT}")
    path = hits[0]
    print(f"ThermoMPNN file: {path}")
    df = pd.read_csv(path)
    # column names vary: "ddG (kcal/mol)" + "Mutation" or similar
    ddg_col = next((c for c in df.columns if "ddg" in c.lower() or "Δ" in c or "kcal" in c.lower()), None)
    mut_col = next((c for c in df.columns if "mut" in c.lower()), None)
    if ddg_col is None or mut_col is None:
        raise SystemExit(f"unexpected columns: {list(df.columns)}")
    df = df.rename(columns={ddg_col: "ddg", mut_col: "mutation"})
    parsed = df["mutation"].astype(str).map(lambda s: MUT_RE.match(s.strip()))
    df = df[parsed.notna()].copy()
    df["wt"] = [m.group(1) for m in parsed.dropna()]
    df["pos"] = [int(m.group(2)) for m in parsed.dropna()]
    df["mut"] = [m.group(3) for m in parsed.dropna()]
    df = df[df["wt"] != df["mut"]]  # drop self
    return df


def main():
    ddg = load_ddg()
    spt = pd.read_csv(MET_SPT / "oct1_af2_rank1_spt.tsv", sep="\t")
    spt["pos"] = spt["pos"].astype(int)
    df = ddg.merge(spt[["pos", "aa", "topology", "rel_sasa", "class"]], on="pos", how="left")
    # sanity: WT letter vs SPT aa
    mismatch = df.dropna(subset=["aa"])
    mismatch = mismatch[mismatch["wt"] != mismatch["aa"]]
    if len(mismatch):
        print(f"WARNING: {mismatch['pos'].nunique()} positions have WT letter ≠ SPT aa "
              f"(numbering offset?). e.g.\n", mismatch.head(8).to_string(index=False))

    val = df[~df["pos"].isin(DESIGN_POS)].dropna(subset=["class", "ddg"])
    print(f"missense ΔΔG: {len(val)}  (design-pos excluded)")
    print("  by class:", val["class"].value_counts().to_dict())
    print(f"  ThermoMPNN sign check: median ΔΔG overall = {val['ddg'].median():.3f} "
          f"(>0 expected destabilizing)")

    # residue-median primary (same spirit as P1)
    res = val.groupby(["pos", "class"], as_index=False)["ddg"].median()
    core = res.loc[res["class"] == "CORE", "ddg"]
    exp = res.loc[res["class"] == "EXPOSED", "ddg"]
    grey = res.loc[res["class"] == "GREY", "ddg"]
    u, p = stats.mannwhitneyu(core, exp, alternative="two-sided")
    p3_dir = core.median() > exp.median()
    # EXPOSED ≈ 0: |median| small vs CORE
    print("\n=== P3  ThermoMPNN ΔΔG (kcal/mol), residue median ===")
    print(f"  CORE    n={len(core)}  median={core.median():+.3f}")
    print(f"  EXPOSED n={len(exp)}  median={exp.median():+.3f}")
    print(f"  GREY    n={len(grey)}  median={grey.median():+.3f}")
    print(f"  MWU CORE vs EXPOSED  U={u:.1f}  p={p:.4g}  CORE>EXPOSED: {p3_dir}")
    # variant-level
    vc = val.loc[val["class"] == "CORE", "ddg"]
    ve = val.loc[val["class"] == "EXPOSED", "ddg"]
    _, p_var = stats.mannwhitneyu(vc, ve, alternative="two-sided")
    print(f"  variant-level medians CORE={vc.median():+.3f} EXPOSED={ve.median():+.3f} p={p_var:.4g}")

    p3_pass = bool(p3_dir and p < 0.05)
    print(f"\n=== P3 verdict: {'PASS' if p3_pass else 'FAIL'} ===")

    out = {
        "pass": p3_pass,
        "p_residue": float(p),
        "p_variant": float(p_var),
        "core_median": float(core.median()),
        "exposed_median": float(exp.median()),
        "grey_median": float(grey.median()),
        "n_core_res": int(len(core)),
        "n_exposed_res": int(len(exp)),
        "n_grey_res": int(len(grey)),
        "n_variants": int(len(val)),
        "overall_median_ddg": float(val["ddg"].median()),
    }
    MET_SPT.mkdir(parents=True, exist_ok=True)
    val.to_csv(MET_SPT / "wp3_p3_thermompnn_variants.tsv", sep="\t", index=False)
    res.to_csv(MET_SPT / "wp3_p3_thermompnn_residue_median.tsv", sep="\t", index=False)
    (MET_SPT / "wp3_p3_verdict.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"outputs -> {MET_SPT}")


if __name__ == "__main__":
    main()
