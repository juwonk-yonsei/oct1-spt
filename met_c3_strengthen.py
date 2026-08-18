#!/usr/bin/env python3
"""C3s S1–S5: strengthen C3 interpretation (c3s_prereg_lock.md).

    $MET_PY met_c3_strengthen.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import Superimposer
from Bio.PDB.Atom import Atom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from met_c3_ensemble import RC, N_MODES, residue_delta_rho, spearman  # noqa: E402
from met_c3_validate import ANM_COLS, TOPO, full_cols, lopo_resid  # noqa: E402
from met_wp5 import load_ca  # noqa: E402

MET_HDD = Path(os.environ.get("MET_HDD", str(Path(__file__).resolve().parent / "data")))
MET_PDB = Path(os.environ.get("MET_PDB", str(MET_HDD / "pdb")))
MET_STRUCT = Path(os.environ.get("MET_STRUCT", str(MET_HDD / "structures")))
MET_DMS = Path(os.environ.get("MET_DMS", str(MET_HDD / "dms")))
MET_SPT = Path(os.environ.get("MET_SPT", str(MET_HDD / "spt")))
OUT = MET_HDD / "challenge" / "c3_ensemble"
PRED = OUT / "oct1_ens_lopo.tsv"
TPT = MET_SPT / "tpt" / "oct1_tpt_variants.tsv"

HGVS_RE = re.compile(r"p\.\(([A-Z])(\d+)([A-Z]|del)\)", re.I)


def anm_hessian_modes(ca: dict, cutoff: float = RC, n_modes: int = N_MODES):
    pos = sorted(ca)
    xyz = np.array([np.asarray(ca[p].coord, dtype=float) for p in pos])
    n = len(pos)
    h = np.zeros((3 * n, 3 * n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dvec = xyz[j] - xyz[i]
            d = float(np.linalg.norm(dvec))
            if d < 1e-3 or d > cutoff:
                continue
            outer = np.outer(dvec, dvec) / (d * d)
            bi, bj = 3 * i, 3 * j
            h[bi : bi + 3, bj : bj + 3] -= outer
            h[bj : bj + 3, bi : bi + 3] -= outer
            h[bi : bi + 3, bi : bi + 3] += outer
            h[bj : bj + 3, bj : bj + 3] += outer
    w, v = np.linalg.eigh(h)
    nz = np.where(w > 1e-8)[0]
    use = nz[:n_modes]
    msf = np.zeros(n)
    modes = []
    for k in use:
        mode = v[:, k].reshape(n, 3)
        msf += np.sum(mode**2, axis=1) / float(w[k])
        modes.append(v[:, k].copy())
    return pos, xyz, msf, modes, w[use]


def s1(df: pd.DataFrame) -> dict:
    tpt = pd.read_csv(TPT, sep="\t")[["hgvs_short", "SM73_1_score"]]
    d = df.drop(columns=[c for c in ("SM73_1_score",) if c in df.columns]).merge(
        tpt, on="hgvs_short", how="left"
    )
    d["_sm73_save"] = d["SM73_0_score"]
    d["SM73_0_score"] = d["SM73_1_score"]  # sm73_residual reads SM73_0_score
    cols = full_cols(d)
    print("S1 LOPO ENS on SM73_1 residual")
    d = lopo_resid(d, cols, "ens_resid_s1")
    ycol = "ens_resid_s1_fold_y"
    sub = d[d["train_ok"] & d["ens_resid_s1"].notna() & d[ycol].notna() & d["am_fitness"].notna()]
    rho_ens = spearman(sub["ens_resid_s1"].to_numpy(), sub[ycol].to_numpy())
    rho_am = spearman(sub["am_fitness"].to_numpy(), sub[ycol].to_numpy())
    delta = residue_delta_rho(sub, "ens_resid_s1", "am_fitness", ycol)
    s1_1 = bool(delta and delta["ci_lo"] > 0 and rho_ens > rho_am)

    anm_only = [c for c in ANM_COLS + TOPO if c in d.columns]
    print("S1 LOPO ANM-only on SM73_1 residual")
    d = lopo_resid(d, anm_only, "anm_s1")
    sub2 = d[d["train_ok"] & d["anm_s1"].notna() & d[ycol].notna() & d["am_fitness"].notna()]
    # use same fold y from first LOPO when available
    rho_anm = spearman(sub2["anm_s1"].to_numpy(), sub2[ycol].to_numpy())
    delta_anm = residue_delta_rho(sub2, "anm_s1", "am_fitness", ycol)
    s1_2 = bool(delta_anm and delta_anm["ci_lo"] > 0 and rho_anm > spearman(sub2["am_fitness"].to_numpy(), sub2[ycol].to_numpy()))

    r0 = df["SM73_resid_fold"] if "SM73_resid_fold" in df.columns else None
    rho_01 = float("nan")
    if r0 is not None:
        m = d[ycol].notna() & df["SM73_resid_fold"].notna() & d["train_ok"]
        rho_01 = spearman(df.loc[m, "SM73_resid_fold"].to_numpy(), d.loc[m, ycol].to_numpy())

    return {
        "S1_1_pass": s1_1,
        "S1_2_pass": s1_2,
        "rho_ens_sm73_1": rho_ens,
        "rho_am_sm73_1": rho_am,
        "delta_vs_AM": delta,
        "rho_anm_sm73_1": rho_anm,
        "delta_anm_vs_AM": delta_anm,
        "rho_resid_0_vs_1": rho_01,
        "n": int(len(sub)),
    }


def s2() -> dict:
    print("S2 ANM vs 8SC1→8ET6 displacement")
    _, ca1, aa1 = load_ca(MET_PDB / "8SC1.pdb")
    _, ca6, aa6 = load_ca(MET_PDB / "8ET6.pdb")
    common = sorted(
        p
        for p in set(ca1) & set(ca6)
        if aa1.get(p) == aa6.get(p) and aa1.get(p) not in (None, "X")
    )
    A = np.array([np.asarray(ca1[p].coord, dtype=float) for p in common])
    B = np.array([np.asarray(ca6[p].coord, dtype=float) for p in common])
    fixed = [Atom("CA", A[i].copy(), 0, 1, " ", " CA ", i + 1, element="C") for i in range(len(common))]
    moving = [Atom("CA", B[i].copy(), 0, 1, " ", " CA ", i + 1, element="C") for i in range(len(common))]
    sup = Superimposer()
    sup.set_atoms(fixed, moving)
    R, t = np.asarray(sup.rotran[0]), np.asarray(sup.rotran[1])
    Bal = B @ R + t
    dvec = (Bal - A).reshape(-1)
    ca_sub = {p: ca1[p] for p in common}
    pos, xyz, msf, modes, evals = anm_hessian_modes(ca_sub)
    overlaps = []
    nd = float(np.dot(dvec, dvec))
    for i, mode in enumerate(modes):
        ov = float(np.dot(mode, dvec) ** 2 / (np.dot(mode, mode) * nd))
        overlaps.append({"mode": i + 1, "overlap": ov, "eval": float(evals[i])})
    best = max(overlaps, key=lambda x: x["overlap"]) if overlaps else {"overlap": 0.0, "mode": None}
    s2_1 = bool(best["overlap"] >= 0.20)
    return {
        "S2_1_pass": s2_1,
        "n_common": len(common),
        "fit_rms": float(sup.rms),
        "best_mode": best,
        "top5": sorted(overlaps, key=lambda x: -x["overlap"])[:5],
    }


def s3(df: pd.DataFrame) -> dict:
    sub = df[df["train_ok"] & df["ens_resid"].notna() & df["SM73_resid_fold"].notna()].copy()
    pk = sub[sub["pocket"] == 1]
    npk = sub[sub["pocket"] != 1]
    rho_e_p = spearman(pk["ens_resid"].to_numpy(), pk["SM73_resid_fold"].to_numpy())
    rho_a_p = spearman(pk["am_fitness"].to_numpy(), pk["SM73_resid_fold"].to_numpy())
    rho_e_n = spearman(npk["ens_resid"].to_numpy(), npk["SM73_resid_fold"].to_numpy())
    s3_1 = bool(np.isfinite(rho_e_p) and np.isfinite(rho_a_p) and rho_e_p > rho_a_p)
    s3_2 = bool(np.isfinite(rho_e_p) and np.isfinite(rho_e_n) and rho_e_p > rho_e_n)
    return {
        "S3_1_pass": s3_1,
        "S3_2_pass": s3_2,
        "n_pocket": int(len(pk)),
        "n_nonpocket": int(len(npk)),
        "rho_ens_pocket": rho_e_p,
        "rho_am_pocket": rho_a_p,
        "rho_ens_nonpocket": rho_e_n,
    }


def s4(df: pd.DataFrame) -> dict:
    rows = []
    sub = df[df["train_ok"] & df["ens_resid"].notna() & df["SM73_resid_fold"].notna()]
    for cl, g in sub.groupby("cluster"):
        if len(g) < 50:
            continue
        re = spearman(g["ens_resid"].to_numpy(), g["SM73_resid_fold"].to_numpy())
        ra = spearman(g["am_fitness"].to_numpy(), g["SM73_resid_fold"].to_numpy())
        rows.append(
            {
                "cluster": cl,
                "n": int(len(g)),
                "rho_ens": re,
                "rho_am": ra,
                "fail_ens_le_am": bool(not (re > ra)),
            }
        )
    fail = [r["cluster"] for r in rows if r["fail_ens_le_am"]]
    pd.DataFrame(rows).to_csv(OUT / "c3s_helix_map.tsv", sep="\t", index=False)
    return {"S4_1_pass": True, "n_clusters": len(rows), "fail_clusters": fail, "rows": rows}


def s5() -> dict:
    print("S5 OCT2 ANM transfer")
    from met_c3_ensemble import anm_msf

    pdb = sorted((MET_STRUCT / "oct2_wt_20260812_092657").glob("SLC22A2_WT_unrelaxed_rank_001_*.pdb"))
    if not pdb:
        return {"S5_1_pass": False, "error": "no OCT2 WT pdb"}
    _, ca, _ = load_ca(pdb[0])
    msf = anm_msf(ca)
    vals = np.array(list(msf.values()), dtype=float)
    cut = float(np.quantile(vals, 0.75))
    lit = pd.read_csv(MET_DMS / "oct2_literature_variants.csv")
    loss_pos = []
    for _, r in lit.iterrows():
        m = HGVS_RE.search(str(r["hgvs"]))
        if not m or m.group(3).lower() == "del":
            continue
        impact = str(r["literature_impact_function"]).lower()
        if impact != "loss":
            continue
        pos = int(m.group(2))
        loss_pos.append(
            {
                "hgvs": r["hgvs"],
                "pos": pos,
                "msf": msf.get(pos),
                "top25": bool(msf.get(pos) is not None and msf[pos] >= cut),
            }
        )
    n = len(loss_pos)
    n_hit = sum(1 for x in loss_pos if x["top25"])
    s5_1 = bool(n >= 3 and (n_hit / n) >= 0.5)
    return {
        "S5_1_pass": s5_1,
        "anm_q75": cut,
        "n_loss": n,
        "n_top25": n_hit,
        "variants": loss_pos,
        "pdb": str(pdb[0]),
    }


def s6(df: pd.DataFrame) -> None:
    """Lock a 12-variant experimental panel from C3 scores (no wet lab)."""
    d = df[df["train_ok"]].copy()
    d = d.dropna(subset=["GFP_score", "SM73_resid_fold", "anm_msf_8sc1", "ens_resid"])
    rows = []

    def take(sub, n, tag):
        out = sub.head(n).copy()
        out["panel_class"] = tag
        return out

    resid_lo = d["SM73_resid_fold"] <= d["SM73_resid_fold"].quantile(0.10)
    gfp_ok = d["GFP_score"] >= -0.30
    anm_hi = d["anm_msf_8sc1"] >= d["anm_msf_8sc1"].quantile(0.75)
    a = d[resid_lo & gfp_ok & anm_hi].sort_values("SM73_resid_fold")
    rows.append(take(a, 4, "resid_loss_gfp_ok_anm_hi"))

    gfp_lo = d["GFP_score"] <= -0.80
    resid_ok = d["SM73_resid_fold"].abs() <= d["SM73_resid_fold"].abs().quantile(0.25)
    b = d[gfp_lo & resid_ok].sort_values("GFP_score")
    rows.append(take(b, 4, "abundance_loss_resid_ok"))

    gfp_mid = d["GFP_score"].abs() <= 0.20
    resid_mid = d["SM73_resid_fold"].abs() <= 0.05
    c = d[gfp_mid & resid_mid].copy()
    c["anm_dev"] = (c["anm_msf_8sc1"] - c["anm_msf_8sc1"].median()).abs()
    c = c.sort_values("anm_dev")
    rows.append(take(c, 4, "both_near_wt_control"))

    pan = pd.concat(rows, ignore_index=True)
    keep = [
        "panel_class",
        "hgvs_short",
        "pos",
        "wt_aa",
        "mut_aa",
        "cluster",
        "spt_class",
        "GFP_score",
        "SM73_0_score",
        "SM73_resid_fold",
        "ens_resid",
        "anm_msf_8sc1",
        "pocket",
        "am_fitness",
    ]
    pan[[c for c in keep if c in pan.columns]].to_csv(OUT / "c3s_experiment_panel.tsv", sep="\t", index=False)
    print(f"S6 wrote {OUT / 'c3s_experiment_panel.tsv'} n={len(pan)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PRED, sep="\t")
    v1 = s1(df)
    v2 = s2()
    v3 = s3(df)
    v4 = s4(df)
    v5 = s5()
    s6(df)
    verdict = {
        "S1": v1,
        "S2": v2,
        "S3": v3,
        "S4": v4,
        "S5": v5,
        "S6_panel": str(OUT / "c3s_experiment_panel.tsv"),
        "S1_pass": bool(v1.get("S1_1_pass") and v1.get("S1_2_pass")),
        "S2_pass": bool(v2.get("S2_1_pass")),
        "S3_pass": bool(v3.get("S3_1_pass")),
        "S5_pass": bool(v5.get("S5_1_pass")),
    }
    (OUT / "c3s_verdict.json").write_text(json.dumps(verdict, indent=2, default=str) + "\n")

    def pf(x):
        return "PASS" if x else "FAIL"

    print("\n=== C3s ===")
    print(f"S1.1 {pf(v1['S1_1_pass'])} ENS {v1['rho_ens_sm73_1']:+.4f} AM {v1['rho_am_sm73_1']:+.4f}")
    print(f"S1.2 {pf(v1['S1_2_pass'])} ANM {v1['rho_anm_sm73_1']:+.4f}")
    print(f"S2   {pf(v2['S2_1_pass'])} best overlap={v2['best_mode'].get('overlap')}")
    print(f"S3.1 {pf(v3['S3_1_pass'])} pocket ENS {v3['rho_ens_pocket']:+.4f} AM {v3['rho_am_pocket']:+.4f}")
    print(f"S3.2 {pf(v3['S3_2_pass'])} nonpocket ENS {v3['rho_ens_nonpocket']:+.4f}")
    print(f"S4   fail clusters: {v4['fail_clusters']}")
    print(f"S5   {pf(v5['S5_1_pass'])} {v5.get('n_top25')}/{v5.get('n_loss')} loss* in ANM top25%")
    print(f"wrote {OUT / 'c3s_verdict.json'}")


if __name__ == "__main__":
    main()
