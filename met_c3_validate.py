#!/usr/bin/env python3
"""C3x — ablation / null checks for locked C3 (c3x_prereg_lock.md).

    $MET_PY met_c3_validate.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from met_c3_ensemble import (  # noqa: E402
    ENS_NUM,
    N_BOOT,
    RNG,
    fit_predict,
    residue_delta_rho,
    sm73_residual,
    spearman,
)

MET_HDD = Path(os.environ.get("MET_HDD", str(Path(__file__).resolve().parent / "data")))
OUT = MET_HDD / "challenge" / "c3_ensemble"
PRED = OUT / "oct1_ens_lopo.tsv"

ANM_COLS = ["anm_msf_8sc1", "anm_msf_8et6", "anm_msf_ratio"]
TOPO = ["topo_Transmembrane", "topo_Extracellular", "topo_Cytoplasmic"]


def full_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in ENS_NUM + TOPO if c in df.columns]


def lopo_resid(df: pd.DataFrame, cols: list[str], pred_name: str, y_resid_col: str | None = None):
    """Helix-LOPO residual target. If y_resid_col set, use that column as y (global resid)."""
    out = df.copy()
    out[pred_name] = np.nan
    fold_col = f"{pred_name}_fold_y"
    if y_resid_col is None:
        out[fold_col] = np.nan
    clusters = sorted(c for c in out.loc[out["train_ok"], "cluster"].dropna().unique())
    for cl in clusters:
        tr_m = out["train_ok"] & (out["cluster"] != cl)
        te_m = out["train_ok"] & (out["cluster"] == cl)
        if int(te_m.sum()) < 10 or int(tr_m.sum()) < 50:
            continue
        tr, te = out.loc[tr_m], out.loc[te_m]
        tr2, te2 = tr.copy(), te.copy()
        if y_resid_col is None:
            r_tr, r_te = sm73_residual(tr, te)
            out.loc[te_m, fold_col] = r_te
            tr2["_y"], te2["_y"] = r_tr, r_te
        else:
            tr2["_y"] = tr[y_resid_col]
            te2["_y"] = te[y_resid_col]
        out.loc[te_m, pred_name] = fit_predict(tr2, te2, cols, "_y")
    return out


def eval_frame(df, pred, ycol="SM73_resid_fold"):
    return df[df["train_ok"] & df[pred].notna() & df[ycol].notna()].copy()


def main() -> None:
    df = pd.read_csv(PRED, sep="\t")
    cols = full_cols(df)
    print("full cols", cols)

    # --- C3x.1 no ANM ---
    no_anm = [c for c in cols if c not in ANM_COLS]
    print("\nLOPO no-ANM")
    df = lopo_resid(df, no_anm, "pred_no_anm")
    d1 = df[df["train_ok"] & df["ens_resid"].notna() & df["pred_no_anm"].notna() & df["SM73_resid_fold"].notna()]
    rho_full = spearman(d1["ens_resid"].to_numpy(), d1["SM73_resid_fold"].to_numpy())
    rho_noanm = spearman(d1["pred_no_anm"].to_numpy(), d1["SM73_resid_fold"].to_numpy())
    d_anm = residue_delta_rho(d1, "ens_resid", "pred_no_anm", "SM73_resid_fold")
    c3x1 = bool(d_anm and d_anm["ci_lo"] > 0)

    # --- C3x.2 no dphi ---
    no_phi = [c for c in cols if c != "dphi_8sc1"]
    print("LOPO no-dphi")
    df = lopo_resid(df, no_phi, "pred_no_dphi")
    d2 = df[df["train_ok"] & df["ens_resid"].notna() & df["pred_no_dphi"].notna() & df["SM73_resid_fold"].notna()]
    rho_nophi = spearman(d2["pred_no_dphi"].to_numpy(), d2["SM73_resid_fold"].to_numpy())
    d_phi = residue_delta_rho(d2, "ens_resid", "pred_no_dphi", "SM73_resid_fold")
    c3x2 = bool(d_phi and d_phi["ci_lo"] > 0)

    # --- C3x.3 permute ANM by residue ---
    print("LOPO ANM-permuted")
    rng = np.random.default_rng(20260813)
    pos = df["pos"].to_numpy()
    uniq = np.unique(pos)
    perm = uniq.copy()
    rng.shuffle(perm)
    pmap = dict(zip(uniq.tolist(), perm.tolist()))
    src = df.drop_duplicates("pos").set_index("pos")
    dfp = df.copy()
    for c in ANM_COLS:
        dfp[c] = dfp["pos"].map(lambda p, c=c: src.loc[pmap[int(p)], c] if int(p) in pmap else np.nan)
    dfp = lopo_resid(dfp, cols, "pred_anm_perm")
    d3 = df.merge(dfp[["hgvs_short", "pred_anm_perm"]], on="hgvs_short", how="left")
    d3 = d3[d3["train_ok"] & d3["ens_resid"].notna() & d3["pred_anm_perm"].notna() & d3["SM73_resid_fold"].notna()]
    rho_perm = spearman(d3["pred_anm_perm"].to_numpy(), d3["SM73_resid_fold"].to_numpy())
    d_perm = residue_delta_rho(d3, "ens_resid", "pred_anm_perm", "SM73_resid_fold")
    c3x3 = bool(d_perm and d_perm["ci_lo"] > 0)

    # --- C3x.8 ANM+topo only vs AM ---
    print("LOPO ANM-only")
    anm_only = [c for c in ANM_COLS + TOPO if c in df.columns]
    df = lopo_resid(df, anm_only, "pred_anm_only")
    d8 = df[df["train_ok"] & df["pred_anm_only"].notna() & df["SM73_resid_fold"].notna() & df["am_fitness"].notna()]
    rho_anmonly = spearman(d8["pred_anm_only"].to_numpy(), d8["SM73_resid_fold"].to_numpy())
    rho_am = spearman(d8["am_fitness"].to_numpy(), d8["SM73_resid_fold"].to_numpy())
    d_am_only = residue_delta_rho(d8, "pred_anm_only", "am_fitness", "SM73_resid_fold")
    c3x8 = bool(d_am_only and d_am_only["ci_lo"] > 0 and rho_anmonly > rho_am)

    # --- C3x.4 helix-wise ---
    helix = []
    for cl, g in d1.groupby("cluster"):
        if len(g) < 50:
            continue
        re = spearman(g["ens_resid"].to_numpy(), g["SM73_resid_fold"].to_numpy())
        ra = spearman(g["am_fitness"].to_numpy(), g["SM73_resid_fold"].to_numpy()) if "am_fitness" in g else float("nan")
        helix.append({"cluster": cl, "n": int(len(g)), "rho_ens": re, "rho_am": ra, "ens_gt_am": bool(re > ra)})
    n_h = len(helix)
    n_ok = sum(1 for h in helix if h["ens_gt_am"])
    c3x4 = bool(n_h >= 4 and (n_ok / n_h) >= 0.5)

    # --- C3x.5 leakage to GFP ---
    rho_gfp = spearman(d1["ens_resid"].to_numpy(), d1["GFP_score"].to_numpy())
    c3x5 = bool(abs(rho_gfp) < 0.15)

    # --- C3x.6 global residual ---
    print("LOPO global-resid")
    ok = df["train_ok"] & df["GFP_score"].notna() & df["SM73_0_score"].notna()
    g = df.loc[ok, "GFP_score"].to_numpy(float)
    y = df.loc[ok, "SM73_0_score"].to_numpy(float)
    X = np.c_[np.ones(len(g)), g]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    glob = np.full(len(df), np.nan)
    m = df["GFP_score"].notna() & df["SM73_0_score"].notna()
    glob[m.to_numpy()] = df.loc[m, "SM73_0_score"].to_numpy(float) - (
        coef[0] + coef[1] * df.loc[m, "GFP_score"].to_numpy(float)
    )
    df["SM73_resid_global"] = glob
    df = lopo_resid(df, cols, "pred_global", y_resid_col="SM73_resid_global")
    d6 = df[df["train_ok"] & df["pred_global"].notna() & df["SM73_resid_global"].notna() & df["am_fitness"].notna()]
    rho_glob = spearman(d6["pred_global"].to_numpy(), d6["SM73_resid_global"].to_numpy())
    rho_am_g = spearman(d6["am_fitness"].to_numpy(), d6["SM73_resid_global"].to_numpy())
    d_glob = residue_delta_rho(d6, "pred_global", "am_fitness", "SM73_resid_global")
    c3x6 = bool(d_glob and d_glob["ci_lo"] > 0 and rho_glob > rho_am_g)

    motion = bool(c3x1 and c3x3 and c3x8)
    verdict = {
        "C3x_1_anm_ablation": {"pass": c3x1, "rho_full": rho_full, "rho_no_anm": rho_noanm, "delta": d_anm},
        "C3x_2_dphi_ablation": {"pass": c3x2, "rho_no_dphi": rho_nophi, "delta": d_phi},
        "C3x_3_anm_permute": {"pass": c3x3, "rho_perm": rho_perm, "delta": d_perm},
        "C3x_8_anm_only_vs_AM": {
            "pass": c3x8,
            "rho_anm_only": rho_anmonly,
            "rho_am": rho_am,
            "delta": d_am_only,
        },
        "C3x_4_helix": {"pass": c3x4, "n_clusters": n_h, "n_ens_gt_am": n_ok, "rows": helix},
        "C3x_5_gfp_leak": {"pass": c3x5, "rho_ens_gfp": rho_gfp},
        "C3x_6_global_resid": {
            "pass": c3x6,
            "rho_ens": rho_glob,
            "rho_am": rho_am_g,
            "delta": d_glob,
        },
        "motion_interpretation_ok": motion,
        "note": (
            "Keep C3 PASS as bundle if C3x.5/6 hold; "
            "withdraw ANM causal claim unless motion_interpretation_ok."
        ),
    }
    path = OUT / "c3x_verdict.json"
    path.write_text(json.dumps(verdict, indent=2, default=str) + "\n")
    pd.DataFrame(helix).to_csv(OUT / "c3x_helix.tsv", sep="\t", index=False)

    def pf(x):
        return "PASS" if x else "FAIL"

    print("\n=== C3x ===")
    print(f"C3x.1 ANM ablation   {pf(c3x1)}  full {rho_full:+.4f}  noANM {rho_noanm:+.4f}  {d_anm}")
    print(f"C3x.2 dphi ablation  {pf(c3x2)}  noφ {rho_nophi:+.4f}  {d_phi}")
    print(f"C3x.3 ANM permute    {pf(c3x3)}  perm {rho_perm:+.4f}  {d_perm}")
    print(f"C3x.8 ANM-only vs AM {pf(c3x8)}  ANM {rho_anmonly:+.4f}  AM {rho_am:+.4f}  {d_am_only}")
    print(f"C3x.4 helix          {pf(c3x4)}  {n_ok}/{n_h} clusters ens>AM")
    print(f"C3x.5 GFP leak       {pf(c3x5)}  ρ(ens,GFP)={rho_gfp:+.4f}")
    print(f"C3x.6 global resid   {pf(c3x6)}  {rho_glob:+.4f} vs AM {rho_am_g:+.4f}  {d_glob}")
    print(f"ANM/motion claim: {'KEEP' if motion else 'WITHDRAW'}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
