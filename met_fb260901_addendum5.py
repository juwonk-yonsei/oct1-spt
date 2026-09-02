#!/usr/bin/env python3
"""Addendum-5: SERT OR saturation, cutoff sweep, myc offset, CORE benign.

    source /SSD1T/PhD/AlphaFold/met_env.sh
    $MET_PY met_fb260901_addendum5.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from met_fb260901 import (  # noqa: E402
    AM_BENIGN,
    GFP_CUT,
    MS1,
    N_BOOT,
    OUT,
    clustered_bootstrap,
    dump,
    grouped_indices,
    logit_fit,
    or_from_counts,
)

MET_HDD = Path(os.environ.get("MET_HDD", "/HDD8T1/WORK/Metformin_HDD"))
MET_SPT = Path(os.environ.get("MET_SPT", str(MET_HDD / "spt")))
ADD4 = OUT / "addendum4"
ADD5 = OUT / "addendum5"
ADD5.mkdir(parents=True, exist_ok=True)

# Andersen / Kristensen / Jorgensen S1 pocket (UniProt P31645 numbering)
SERT_POCKET = [
    {"res": "Y95", "pos": 95, "role": "TM1 S1 subsite A; SSRI aromatic", "expected_topo": "Transmembrane"},
    {"res": "D98", "pos": 98, "role": "TM1 amine salt bridge; substrate/SSRI", "expected_topo": "Transmembrane"},
    {"res": "I172", "pos": 172, "role": "TM3 S1; antidepressant affinity", "expected_topo": "Transmembrane"},
    {"res": "N177", "pos": 177, "role": "TM3 S1; (S)-citalopram", "expected_topo": "Transmembrane"},
    {"res": "F335", "pos": 335, "role": "TM6 S1; after myc insert (construct 359)", "expected_topo": "Transmembrane"},
    {"res": "F341", "pos": 341, "role": "TM6 S1; after myc insert (construct 365)", "expected_topo": "Transmembrane"},
    {"res": "S438", "pos": 438, "role": "TM8 S1; after myc insert (construct 462)", "expected_topo": "Transmembrane"},
]


def cells_from(g: pd.DataFrame, class_col="class"):
    core = g[g[class_col] == "CORE"]
    exp = g[g[class_col] == "EXPOSED"]
    a = int((core["am_class"] == "pathogenic").sum())
    b = int((core["am_class"] != "pathogenic").sum())
    c = int((exp["am_class"] == "pathogenic").sum())
    d = int((exp["am_class"] != "pathogenic").sum())
    n_c, n_e = a + b, c + d
    rec_c = a / n_c if n_c else np.nan
    rec_e = c / n_e if n_e else np.nan
    rd = rec_c - rec_e if np.isfinite(rec_c) and np.isfinite(rec_e) else np.nan
    rr = rec_c / rec_e if rec_e not in (0, np.nan) else np.nan
    or_raw = or_from_counts(a, b, c, d)
    or_ha = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    return {
        "a_core_path": a,
        "b_core_nonpath": b,
        "c_exp_path": c,
        "d_exp_nonpath": d,
        "n_core": n_c,
        "n_exposed": n_e,
        "core_fn_frac": b / n_c if n_c else np.nan,
        "recall_core": rec_c,
        "recall_exposed": rec_e,
        "rd": rd,
        "rr": rr,
        "or_raw": or_raw,
        "or_haldane": or_ha,
    }


def clustered_rd_rr(g: pd.DataFrame, class_col="class"):
    g = g.reset_index(drop=True)
    groups = grouped_indices(g["pos"].to_numpy())
    am_path = (g["am_class"] == "pathogenic").to_numpy()
    is_core = (g[class_col] == "CORE").to_numpy()
    is_exp = (g[class_col] == "EXPOSED").to_numpy()

    def rec(idx, mask):
        m = mask[idx]
        if m.sum() == 0:
            return np.nan
        return float(am_path[idx][m].mean())

    def rd_fn(idx):
        rc, re = rec(idx, is_core), rec(idx, is_exp)
        return rc - re if np.isfinite(rc) and np.isfinite(re) else np.nan

    def rr_fn(idx):
        rc, re = rec(idx, is_core), rec(idx, is_exp)
        return rc / re if np.isfinite(rc) and np.isfinite(re) and re > 0 else np.nan

    return {
        "rd": clustered_bootstrap(groups, rd_fn),
        "rr": clustered_bootstrap(groups, rr_fn),
    }


def firth_logit(X, y, niter=50):
    """Firth-penalized logistic. X includes intercept. Returns beta."""
    beta = np.zeros(X.shape[1], dtype=float)
    n = X.shape[0]
    for _ in range(niter):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        p = np.clip(p, 1e-8, 1 - 1e-8)
        w = p * (1 - p)
        xtw = X.T * w
        hmat = xtw @ X
        try:
            inv = np.linalg.inv(hmat)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(hmat)
        # diagonal of hat matrix H = W^{1/2} X (X'WX)^{-1} X' W^{1/2}
        # h_i = w_i * x_i (X'WX)^{-1} x_i'
        xi_iwi = np.einsum("ij,jk,ik->i", X, inv, X) * w
        adj = xi_iwi * (0.5 - p)
        score = X.T @ (y - p + adj)
        try:
            step = inv @ score
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hmat, score, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def exposed_or_models(g: pd.DataFrame, class_col="class"):
    g = g.dropna(subset=["am_class", class_col]).copy()
    g["y"] = (g["am_class"] == "pathogenic").astype(float)
    g["exposed"] = (g[class_col] == "EXPOSED").astype(float)
    X = np.column_stack([np.ones(len(g)), g["exposed"].to_numpy()])
    y = g["y"].to_numpy()
    mle = logit_fit(X, y)
    firth = firth_logit(X, y)
    cells = cells_from(g, class_col)
    return {
        "mle_exposed_or": float(np.exp(mle[1])),
        "mle_core_vs_exp_or": float(np.exp(-mle[1])),
        "firth_exposed_or": float(np.exp(firth[1])),
        "firth_core_vs_exp_or": float(np.exp(-firth[1])),
        "haldane_core_vs_exp_or": cells["or_haldane"],
        "n": int(len(g)),
    }


def cutoff_sweep(miss: pd.DataFrame, lo=-3.0, hi=-1.0, step=0.1):
    cuts = np.round(np.arange(lo, hi + 1e-9, step), 1)
    rows = []
    for cut in cuts:
        g = miss[miss["myc"] <= cut].copy()
        g = g[g["class"].isin(["CORE", "EXPOSED"])].dropna(subset=["am_class"])
        rec = cells_from(g)
        rec["cut"] = float(cut)
        rec["n_loss_ce"] = rec["n_core"] + rec["n_exposed"]
        rec["core_gt_exposed"] = bool(rec["recall_core"] > rec["recall_exposed"])
        rec["rd_positive"] = bool(rec["rd"] > 0)
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_sweep(sw: pd.DataFrame, dest: Path):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    ax = axes[0]
    ax.plot(sw["cut"], 100 * sw["recall_core"], color="#1f4e79", lw=2, label="CORE recall")
    ax.plot(sw["cut"], 100 * sw["recall_exposed"], color="#c45911", lw=2, label="EXPOSED recall")
    ax.axvline(-1.907, color="#595959", ls="--", lw=1, label="tech-SD −1.907")
    ax.axvline(-2.164, color="#7f7f7f", ls=":", lw=1, label="nonsense median −2.164")
    ax.set_ylabel("AM-pathogenic recall (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax.set_title("SERT myc-loss: CORE vs EXPOSED recall by abundance cutoff")
    ax = axes[1]
    ax.plot(sw["cut"], 100 * sw["rd"], color="#2e7d32", lw=2, label="Risk difference (CORE − EXPOSED)")
    ax.axhline(36.6, color="#1f4e79", ls="--", lw=1, label="OCT1 GFP RD 36.6%p")
    ax.axvline(-1.907, color="#595959", ls="--", lw=1)
    ax.axvline(-2.164, color="#7f7f7f", ls=":", lw=1)
    ax.set_xlabel("myc enrichment cutoff (loss if ≤ cut)")
    ax.set_ylabel("Risk difference (%p)")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(dest, dpi=160)
    fig.savefig(dest.with_suffix(".pdf"))
    plt.close(fig)


def verify_offset(spt: pd.DataFrame, raw: pd.DataFrame, uni_seq: str):
    spt = spt.copy()
    spt["pos"] = spt["pos"].astype(int)
    by_spt = spt.set_index("pos")
    wt = raw.drop_duplicates("pos_construct")
    wt_map = dict(zip(wt["pos_construct"].astype(int), wt["wt_aa"]))
    mapped = raw[raw["pos"].notna()].drop_duplicates("pos")
    mapped_aa = dict(zip(mapped["pos"].astype(int), mapped["wt_aa"]))
    rows = []
    for rec in SERT_POCKET:
        p = rec["pos"]
        construct = p if p < 217 else p + 24
        s = by_spt.loc[p] if p in by_spt.index else None
        uni_aa = uni_seq[p - 1]
        young_mapped = mapped_aa.get(p)
        young_construct = wt_map.get(construct)
        young_unshifted = wt_map.get(p)  # what we'd see if we forgot −24
        ok_aa = (young_mapped == uni_aa == rec["res"][0])
        rows.append({
            **rec,
            "uniprot_aa": uni_aa,
            "spt_aa": None if s is None else s["aa"],
            "topology": None if s is None else s["topology"],
            "rel_sasa": None if s is None else float(s["rel_sasa"]),
            "class": None if s is None else s["class"],
            "construct_pos": construct,
            "young_aa_mapped": young_mapped,
            "young_aa_at_construct": young_construct,
            "young_aa_if_no_shift": young_unshifted,
            "aa_match": bool(ok_aa and (s is not None) and s["aa"] == uni_aa),
            "topo_is_TM": bool(s is not None and s["topology"] == "Transmembrane"),
            "not_exposed": bool(s is not None and s["class"] != "EXPOSED"),
        })
    return rows


def oct1_core_benign(val: pd.DataFrame):
    g = val[(val["dms_loss"]) & (val["class"] == "CORE")].dropna(subset=["am_class"])
    n = len(g)
    n_ben = int((g["am_class"] == "benign").sum())
    n_amb = int((g["am_class"] == "ambiguous").sum())
    n_path = int((g["am_class"] == "pathogenic").sum())
    return {
        "n_gfp_loss_core": n,
        "n_benign": n_ben,
        "n_ambiguous": n_amb,
        "n_pathogenic": n_path,
        "frac_benign": n_ben / n if n else np.nan,
        "frac_nonpathogenic": (n_ben + n_amb) / n if n else np.nan,
        "note": "Validation missense; design 61/88/401/420/465 already excluded in this table.",
    }


def main():
    miss = pd.read_csv(ADD4 / "SLC6A4_Young2021_missense_locked.tsv", sep="\t")
    raw = pd.read_csv(ADD4 / "SLC6A4_Young2021_raw.tsv", sep="\t")
    spt = pd.read_csv(ADD4 / "SLC6A4_AFDB_v6_spt_oct1lock.tsv", sep="\t")
    val = pd.read_csv(MET_SPT / "wp3_validation_missense.tsv", sep="\t")
    val["dms_loss"] = val["GFP_score"] <= GFP_CUT
    uni = json.loads(
        Path("/HDD8T1/WORK/Metformin_HDD/challenge/c5_slcmap/cache/uniprot/P31645.json").read_text()
    )["sequence"]["value"]

    oct1 = val[(val["dms_loss"]) & val["class"].isin(["CORE", "EXPOSED"])].dropna(subset=["am_class"])
    sert = miss[(miss["dms_loss"]) & miss["class"].isin(["CORE", "EXPOSED"])].dropna(subset=["am_class"])

    oct1_cells = cells_from(oct1)
    sert_cells = cells_from(sert)
    oct1_cl = clustered_rd_rr(oct1)
    sert_cl = clustered_rd_rr(sert)
    oct1_mod = exposed_or_models(oct1)
    sert_mod = exposed_or_models(sert)

    sw = cutoff_sweep(miss)
    sw.to_csv(ADD5 / "SLC6A4_cutoff_sweep.tsv", sep="\t", index=False)
    plot_sweep(sw, ADD5 / "SLC6A4_cutoff_sweep.png")
    n_dir = int(sw["core_gt_exposed"].sum())
    n_rd = int(sw["rd_positive"].sum())

    offset = verify_offset(spt, raw, uni)
    pd.DataFrame(offset).to_csv(ADD5 / "SLC6A4_offset_check.tsv", sep="\t", index=False)
    offset_ok = all(r["aa_match"] and r["topo_is_TM"] and r["not_exposed"] for r in offset)

    # Wrong map: join SPT on construct coordinates (no −24)
    wrong = miss.copy()
    wrong["class"] = wrong["pos_construct"].map(dict(zip(spt["pos"].astype(int), spt["class"])))
    wrong_loss = wrong[(wrong["dms_loss"]) & wrong["class"].isin(["CORE", "EXPOSED"])].dropna(subset=["am_class"])
    wrong_cells = cells_from(wrong_loss) if len(wrong_loss) else {"error": "empty"}

    benign = oct1_core_benign(val)

    payload = {
        "headline": (
            "Do not lead with SERT OR 33.4. CORE false-negative rate is 2.9% (29/1003); "
            "the OR inflates from cell saturation. RD 46.1%p and RR 1.90 match OCT1 "
            "RD 36.6%p / RR 1.88."
        ),
        "oct1": {"cells": oct1_cells, "clustered": oct1_cl, "models": oct1_mod},
        "sert": {"cells": sert_cells, "clustered": sert_cl, "models": sert_mod},
        "cutoff_sweep": {
            "n_cuts": int(len(sw)),
            "n_core_gt_exposed": n_dir,
            "n_rd_positive": n_rd,
            "all_cuts_same_direction": bool(n_dir == len(sw) and n_rd == len(sw)),
            "rd_min": float(sw["rd"].min()),
            "rd_max": float(sw["rd"].max()),
            "rd_at_techSD": float(sw.loc[np.isclose(sw["cut"], -1.9), "rd"].iloc[0]) if any(np.isclose(sw["cut"], -1.9)) else None,
            "plot": str(ADD5 / "SLC6A4_cutoff_sweep.png"),
        },
        "myc_offset": {
            "residues": offset,
            "all_aa_TM_not_exposed": offset_ok,
            "wrong_map_no_minus24": wrong_cells,
            "rule": "UniProt pos = construct pos if <217 else construct−24; insert 217–240 dropped.",
        },
        "oct1_core_gfp_loss_am_benign": benign,
        "locks": {"am_benign": AM_BENIGN, "gfp_cut": GFP_CUT, "n_boot": N_BOOT},
        "wording": {
            "not_identical_lock": (
                "Same SPT 10%/30% Extracellular/Cytoplasmic rule and AM>0.564; "
                "loss cutoff adjusted because Young 2021 has no synonymous-codon GFP."
            ),
            "promotion": (
                "SERT uses the same SPT rule and a single surface-expression readout; "
                "the ProteinGym panel uses a different SASA dictionary and mixed assay types."
            ),
            "noise_floor_methods": (
                "The RMSD noise floor (3.284 Å) is the maximum Cα RMSD among five "
                "ColabFold wild-type models of OCT1 generated under a single protocol. "
                "It does not include pipeline-to-pipeline differences. ColabFold rank-1 "
                "versus AFDB v6 (3.74 Å; TM-score 0.951) is reported in SI and is outside "
                "that floor by construction."
            ),
        },
    }
    dump(payload, ADD5 / "ms1_feedback2_addendum5.json")
    dump(payload, MS1 / "ms1_feedback2_addendum5.json")
    # also copy plot next to the manuscript
    import shutil
    shutil.copy(ADD5 / "SLC6A4_cutoff_sweep.png", MS1 / "SLC6A4_cutoff_sweep.png")
    shutil.copy(ADD5 / "SLC6A4_cutoff_sweep.pdf", MS1 / "SLC6A4_cutoff_sweep.pdf")
    print("OCT1", {k: oct1_cells[k] for k in ("a_core_path", "b_core_nonpath", "recall_core", "recall_exposed", "rd", "rr", "or_raw")})
    print("SERT", {k: sert_cells[k] for k in ("a_core_path", "b_core_nonpath", "recall_core", "recall_exposed", "rd", "rr", "or_raw")})
    print("SERT clustered RD", sert_cl["rd"])
    print("SERT clustered RR", sert_cl["rr"])
    print("SERT Firth core:exp OR", sert_mod["firth_core_vs_exp_or"], "Haldane", sert_mod["haldane_core_vs_exp_or"])
    print("sweep same direction", n_dir, "/", len(sw), "RD range", sw["rd"].min(), sw["rd"].max())
    print("offset ok", offset_ok)
    for r in offset:
        print(f"  {r['res']:5s} topo={r['topology']:16s} class={r['class']:8s} "
              f"SASA={r['rel_sasa']:.1f} mapped={r['young_aa_mapped']} "
              f"noshift={r['young_aa_if_no_shift']} match={r['aa_match']}")
    print("CORE GFP-loss AM-benign", benign)
    print("wrote", ADD5)


if __name__ == "__main__":
    main()
