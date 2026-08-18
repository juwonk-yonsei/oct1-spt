#!/usr/bin/env python3
"""R1–R4 residual ceiling track (met_prereg_residual_r.md).

Sequential: R1 SNR → R2 Yee columns → R3 locked subset → R4 mutant×gate features.

    source met_env.sh && $MET_PY met_r_residual.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from met_c3_ensemble import residue_delta_rho, sm73_residual, spearman  # noqa: E402
from met_c3_validate import TOPO, eval_frame, full_cols, lopo_resid  # noqa: E402

MET_HDD = Path(os.environ.get("MET_HDD", str(Path(__file__).resolve().parent / "data")))
MET_DMS = Path(os.environ.get("MET_DMS", str(MET_HDD / "dms")))
OUT = MET_HDD / "challenge" / "r_residual"
PRED = MET_HDD / "challenge" / "c3_ensemble" / "oct1_ens_lopo.tsv"
C3_VERDICT = MET_HDD / "challenge" / "c3_ensemble" / "c3_verdict.json"

HGVS_RE = re.compile(r"p\.\(([A-Z])(\d+)([A-Z])\)")
P1_LOSS_CUT = -0.814  # published; fallback only
C3_RHO = 0.07668773882339218

R4_EXTRA = [
    "gate_x_dvol",
    "gate_x_absdvol",
    "gate_x_dcharge",
    "msf_x_dvol",
    "msf_x_absdcharge",
    "lambda_steric",
]


def _ols_resid(y: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(y) & np.isfinite(g)
    coef = np.array([np.nan, np.nan])
    resid = np.full(len(y), np.nan)
    if ok.sum() < 30:
        return resid, coef
    X = np.c_[np.ones(ok.sum()), g[ok]]
    coef, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    resid[ok] = y[ok] - (coef[0] + coef[1] * g[ok])
    return resid, coef


def _parse_hgvs(s: str):
    m = HGVS_RE.search(str(s))
    if not m:
        return None
    return m.group(1) + m.group(2) + m.group(3)


def r1_r2(df: pd.DataFrame, combined: pd.DataFrame, scores: pd.DataFrame) -> dict:
    out: dict = {"step": "R1_R2"}

    def mut_summary(dms: pd.DataFrame, label: str) -> dict:
        vc = dms["mutation_type"].astype(str).value_counts().to_dict()
        rows = {}
        for lab, mask in (
            ("wt", dms["is.wt"] == True),  # noqa: E712
            ("syn", dms["mutation_type"].astype(str) == "S"),
            ("missense", dms["mutation_type"].astype(str) == "M"),
        ):
            sub = dms.loc[mask]
            rows[lab] = {
                "n": int(len(sub)),
                "GFP_mean": float(sub["GFP_score"].mean()) if len(sub) else None,
                "GFP_sd": float(sub["GFP_score"].std()) if len(sub) else None,
                "SM73_0_mean": float(sub["SM73_0_score"].mean()) if len(sub) else None,
                "SM73_0_sd": float(sub["SM73_0_score"].std()) if len(sub) else None,
            }
        return {"mutation_type_counts": {str(k): int(v) for k, v in vc.items()}, "groups": rows, "file": label}

    out["combined"] = mut_summary(combined, "oct1_combined_scores.csv")
    out["scores_file"] = mut_summary(scores, "oct1_scores.csv")

    # GFP version check
    c = combined.copy()
    c["hgvs_short"] = c["hgvs"].map(_parse_hgvs)
    s = scores.copy()
    s["hgvs_short"] = s["hgvs"].map(_parse_hgvs)
    both = c[["hgvs_short", "GFP_score", "SM73_0_score"]].dropna().merge(
        s[["hgvs_short", "GFP_score", "SM73_0_score"]].dropna(),
        on="hgvs_short",
        suffixes=("_combined", "_scores"),
    )
    both = both[both["hgvs_short"].notna()]
    gfp_delta = (both["GFP_score_combined"] - both["GFP_score_scores"]).abs()
    out["file_match"] = {
        "n_overlap": int(len(both)),
        "median_abs_GFP_delta": float(gfp_delta.median()) if len(both) else None,
        "median_abs_SM73_0_delta": float(
            (both["SM73_0_score_combined"] - both["SM73_0_score_scores"]).abs().median()
        )
        if len(both)
        else None,
        "combined_matches_c3_gfp": None,
    }

    miss_c = c[c["mutation_type"].astype(str) == "M"].copy()
    syn_c = c[c["mutation_type"].astype(str) == "S"].copy()
    out["spearman_combined_missense"] = {
        "GFP_vs_SM73_0": spearman(miss_c["GFP_score"].to_numpy(float), miss_c["SM73_0_score"].to_numpy(float)),
        "GFP_vs_SM73_1": spearman(miss_c["GFP_score"].to_numpy(float), miss_c["SM73_1_score"].to_numpy(float)),
        "SM73_0_vs_SM73_1": spearman(miss_c["SM73_0_score"].to_numpy(float), miss_c["SM73_1_score"].to_numpy(float)),
        "n": int(len(miss_c)),
    }
    out["spearman_combined_syn"] = {
        "GFP_vs_SM73_0": spearman(syn_c["GFP_score"].to_numpy(float), syn_c["SM73_0_score"].to_numpy(float)),
        "n": int(len(syn_c)),
    }

    # C3 table vs combined GFP
    m = df.merge(
        c[["hgvs_short", "GFP_score", "SM73_0_SE", "GFP_SE", "SM73_0_score"]].rename(
            columns={"GFP_score": "GFP_combined", "SM73_0_score": "SM73_combined"}
        ),
        on="hgvs_short",
        how="left",
    )
    dlt = (m["GFP_score"] - m["GFP_combined"]).abs()
    out["file_match"]["combined_matches_c3_gfp"] = bool(dlt.dropna().median() <= 0.05) if dlt.notna().any() else False
    out["file_match"]["median_abs_C3_vs_combined_GFP"] = float(dlt.dropna().median()) if dlt.notna().any() else None

    te = df[df["train_ok"] & df["GFP_score"].notna() & df["SM73_0_score"].notna()].copy()
    y = te["SM73_0_score"].to_numpy(float)
    g = te["GFP_score"].to_numpy(float)
    resid, coef = _ols_resid(y, g)
    te = te.copy()
    te["_resid"] = resid

    syn_join = syn_c.merge(
        df[["hgvs_short", "GFP_score", "SM73_0_score"]],
        on="hgvs_short",
        how="inner",
        suffixes=("_dms", "_c3"),
    )
    # synonymous are usually absent from C3 missense table; residualize with missense coef
    syn_y = syn_c["SM73_0_score"].to_numpy(float)
    syn_g = syn_c["GFP_score"].to_numpy(float)
    syn_resid = syn_y - (coef[0] + coef[1] * syn_g)
    syn_ok = np.isfinite(syn_resid) & np.isfinite(syn_g) & np.isfinite(syn_y)

    miss_resid = resid[np.isfinite(resid)]
    var_miss = float(np.var(miss_resid, ddof=1)) if len(miss_resid) > 5 else float("nan")
    var_syn = float(np.var(syn_resid[syn_ok], ddof=1)) if syn_ok.sum() > 5 else float("nan")
    rel_syn = float(1.0 - var_syn / var_miss) if np.isfinite(var_syn) and var_miss > 0 else float("nan")

    # SE attenuation on C3 missense joined to combined SE
    se = m.loc[m["train_ok"] & m["SM73_0_score"].notna() & m["GFP_score"].notna()].copy()
    se_r2 = se["SM73_0_SE"].to_numpy(float) ** 2 + (coef[1] ** 2) * se["GFP_SE"].to_numpy(float) ** 2
    yv = se["SM73_0_score"].to_numpy(float)
    gv = se["GFP_score"].to_numpy(float)
    rv, _ = _ols_resid(yv, gv)
    ok_se = np.isfinite(se_r2) & np.isfinite(rv)
    rel_se = float("nan")
    rel_raw = float("nan")
    if ok_se.sum() > 50 and np.var(rv[ok_se], ddof=1) > 0:
        rel_se = float(1.0 - np.nanmean(se_r2[ok_se]) / np.var(rv[ok_se], ddof=1))
    if se["SM73_0_SE"].notna().sum() > 50 and np.var(yv[np.isfinite(yv)], ddof=1) > 0:
        rel_raw = float(
            1.0
            - np.nanmean(se["SM73_0_SE"].to_numpy(float) ** 2)
            / np.var(yv[np.isfinite(yv)], ddof=1)
        )

    ceiling = float(np.sqrt(max(rel_se, 0.0))) if np.isfinite(rel_se) else float("nan")
    ceiling_syn = float(np.sqrt(max(rel_syn, 0.0))) if np.isfinite(rel_syn) else float("nan")

    out["R1"] = {
        "n_train_ok": int(len(te)),
        "spearman_GFP_SM73_0_c3": spearman(g, y),
        "ols_intercept": float(coef[0]),
        "ols_slope": float(coef[1]),
        "resid_sd_missense": float(np.std(miss_resid, ddof=1)) if len(miss_resid) > 5 else None,
        "resid_sd_synonymous": float(np.std(syn_resid[syn_ok], ddof=1)) if syn_ok.sum() > 5 else None,
        "n_synonymous": int(syn_ok.sum()),
        "reliability_syn_as_noise": rel_syn,
        "reliability_raw_SM73_SE": rel_raw,
        "reliability_residual_SE": rel_se,
        "spearman_ceiling_from_SE": ceiling,
        "spearman_ceiling_from_syn": ceiling_syn,
        "full_set_0p30_possible_SE": bool(np.isfinite(ceiling) and ceiling >= 0.30),
        "full_set_0p20_possible_SE": bool(np.isfinite(ceiling) and ceiling >= 0.20),
        "skip_r4_ceiling_below_0p05": bool(np.isfinite(ceiling) and ceiling < 0.05),
        "n_syn_in_c3_table": int(len(syn_join)),
    }
    out["R1_note"] = (
        "Ceiling is an upper bound on Spearman if the model predicted the noiseless residual. "
        "SM73_1 is not a replicate."
    )
    return out


def abundance_rules(df: pd.DataFrame, combined: pd.DataFrame, r12: dict) -> dict:
    syn = combined[combined["mutation_type"].astype(str) == "S"]["GFP_score"].dropna()
    syn_mean, syn_sd, syn_med = float(syn.mean()), float(syn.std(ddof=1)), float(syn.median())
    match = bool(r12["file_match"].get("combined_matches_c3_gfp"))
    if match and syn_sd > 0:
        branch = "combined_syn_on_matching_GFP"
        not_loss = syn_mean - 2 * syn_sd
        near_lo, near_hi = syn_med - syn_sd, syn_med + syn_sd
    else:
        branch = "published_P1_fallback"
        not_loss = P1_LOSS_CUT
        half = abs(P1_LOSS_CUT) / 2.0
        near_lo, near_hi = -half, half
        syn_mean, syn_sd, syn_med = None, None, None
    return {
        "branch": branch,
        "syn_GFP_mean": syn_mean,
        "syn_GFP_sd": syn_sd,
        "syn_GFP_median": syn_med,
        "not_loss_cut": float(not_loss),
        "near_wt_lo": float(near_lo),
        "near_wt_hi": float(near_hi),
    }


def gate_positions(df: pd.DataFrame) -> dict:
    pos = (
        df.loc[df["topology"] == "Transmembrane", ["pos", "gate_disp"]]
        .dropna()
        .drop_duplicates("pos")
    )
    q75 = float(pos["gate_disp"].quantile(0.75))
    gate = set(pos.loc[pos["gate_disp"] >= q75, "pos"].astype(int))
    return {
        "n_tm_positions": int(len(pos)),
        "gate_disp_q75": q75,
        "n_gate_positions": int(len(gate)),
        "gate_pos": sorted(int(p) for p in gate),
    }


def subset_eval(df: pd.DataFrame, mask: pd.Series, pred_col: str, ycol: str, label: str) -> dict:
    sub = df.loc[mask & df[pred_col].notna() & df[ycol].notna()].copy()
    n = int(len(sub))
    if n < 50 or sub["pos"].nunique() < 8:
        return {"label": label, "n": n, "n_pos": int(sub["pos"].nunique()), "rho": None, "underpowered": True}
    rho = spearman(sub[pred_col].to_numpy(float), sub[ycol].to_numpy(float))
    d_am = None
    if "am_fitness" in sub.columns:
        d_am = residue_delta_rho(sub, pred_col, "am_fitness", ycol)
        d_am = {k: v for k, v in d_am.items()} if d_am else None
    rho_am = (
        spearman(sub["am_fitness"].to_numpy(float), sub[ycol].to_numpy(float))
        if "am_fitness" in sub.columns
        else float("nan")
    )
    return {
        "label": label,
        "n": n,
        "n_pos": int(sub["pos"].nunique()),
        "rho": rho,
        "rho_am": rho_am,
        "delta_vs_AM": d_am,
        "underpowered": False,
    }


def add_r4_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    gd = d["gate_disp"].to_numpy(float)
    dv = d["d_volume"].to_numpy(float)
    msf = d["anm_msf_8sc1"].to_numpy(float)
    adc = d["abs_d_charge"].to_numpy(float)
    d["gate_x_dvol"] = gd * dv
    d["gate_x_absdvol"] = gd * np.abs(dv)
    d["gate_x_dcharge"] = gd * adc
    d["msf_x_dvol"] = msf * dv
    d["msf_x_absdcharge"] = msf * adc
    d["lambda_steric"] = gd * np.clip(dv, 0, None)
    return d


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PRED, sep="\t")
    combined = pd.read_csv(MET_DMS / "oct1_combined_scores.csv")
    scores = pd.read_csv(MET_DMS / "oct1_scores.csv")

    print("=== R1 / R2 ceiling and Yee columns ===")
    r12 = r1_r2(df, combined, scores)
    r1 = r12["R1"]
    print(f"  GFP vs SM73_0 (C3 train_ok) ρ={r1['spearman_GFP_SM73_0_c3']:+.3f}")
    print(f"  combined missense GFP vs SM73_0 ρ={r12['spearman_combined_missense']['GFP_vs_SM73_0']:+.3f}")
    print(f"  GFP vs SM73_1 ρ={r12['spearman_combined_missense']['GFP_vs_SM73_1']:+.3f}")
    print(f"  SM73_0 vs SM73_1 ρ={r12['spearman_combined_missense']['SM73_0_vs_SM73_1']:+.3f}")
    print(f"  OLS slope b={r1['ols_slope']:+.4f}  resid SD miss={r1['resid_sd_missense']}  syn={r1['resid_sd_synonymous']}")
    print(f"  reliability residual(SE)={r1['reliability_residual_SE']}  ceiling={r1['spearman_ceiling_from_SE']}")
    print(f"  C3 GFP matches combined? {r12['file_match']['combined_matches_c3_gfp']}  "
          f"median|Δ|={r12['file_match']['median_abs_C3_vs_combined_GFP']}")

    rules = abundance_rules(df, combined, r12)
    gate = gate_positions(df)
    print("\n=== R3 locked subset ===")
    print(f"  abundance branch={rules['branch']}  not_loss>={rules['not_loss_cut']:.3f}  "
          f"near_wt=[{rules['near_wt_lo']:.3f},{rules['near_wt_hi']:.3f}]")
    print(f"  GATE TM q75 disp={gate['gate_disp_q75']:.3f} Å  n_pos={gate['n_gate_positions']}")

    gpos = set(gate["gate_pos"])
    not_loss = df["train_ok"] & (df["GFP_score"] >= rules["not_loss_cut"]) & df["pos"].isin(gpos)
    near_wt = (
        df["train_ok"]
        & df["GFP_score"].between(rules["near_wt_lo"], rules["near_wt_hi"])
        & df["pos"].isin(gpos)
    )

    r3_primary = subset_eval(df, not_loss, "ens_resid", "SM73_resid_fold", "GATE∩not_loss")
    r3_near = subset_eval(df, near_wt, "ens_resid", "SM73_resid_fold", "GATE∩near_wt")
    r3_gate_only = subset_eval(df, df["train_ok"] & df["pos"].isin(gpos), "ens_resid", "SM73_resid_fold", "GATE_all_GFP")
    print(f"  primary {r3_primary}")
    print(f"  near_wt {r3_near}")
    print(f"  gate_all {r3_gate_only}")

    rho_p = r3_primary.get("rho")
    d_am = r3_primary.get("delta_vs_AM") or {}
    r3_1 = bool(
        rho_p is not None
        and rho_p >= 0.20
        and d_am.get("ci_lo") is not None
        and d_am["ci_lo"] > 0
    )
    r3_2 = bool(rho_p is not None and rho_p >= 0.30)
    print(f"  R3.1 {'PASS' if r3_1 else 'FAIL'}  R3.2 stretch {'PASS' if r3_2 else 'FAIL'}")

    r4 = {"skipped": False}
    skip_r4 = bool(r1.get("skip_r4_ceiling_below_0p05"))
    if skip_r4:
        print("\n=== R4 skipped (R1 ceiling < 0.05) ===")
        r4 = {"skipped": True, "reason": "R1 ceiling < 0.05"}
    else:
        print("\n=== R4 mutation-specific features, helix-LOPO ===")
        d4 = add_r4_features(df)
        cols = full_cols(d4) + [c for c in R4_EXTRA if c in d4.columns]
        print("  cols", cols)
        d4 = lopo_resid(d4, cols, "r4_pred")
        ev = eval_frame(d4, "r4_pred")
        rho_r4 = spearman(ev["r4_pred"].to_numpy(), ev["SM73_resid_fold"].to_numpy())
        rho_c3 = spearman(ev["ens_resid"].to_numpy(), ev["SM73_resid_fold"].to_numpy())
        d_vs_c3 = residue_delta_rho(ev, "r4_pred", "ens_resid", "SM73_resid_fold")
        r4_primary = subset_eval(d4, not_loss, "r4_pred", "SM73_resid_fold", "R4 GATE∩not_loss")
        r4_1 = bool(
            np.isfinite(rho_r4)
            and rho_r4 > C3_RHO
            and d_vs_c3
            and d_vs_c3["ci_lo"] > 0
        )
        rho_sub = r4_primary.get("rho")
        r4_2 = bool(rho_sub is not None and rho_sub >= 0.20)
        r4 = {
            "skipped": False,
            "n_eval": int(len(ev)),
            "rho_r4_full": rho_r4,
            "rho_c3_same_rows": rho_c3,
            "delta_vs_C3": d_vs_c3,
            "subset_primary": r4_primary,
            "R4_1_pass": r4_1,
            "R4_2_pass": r4_2,
            "features_added": R4_EXTRA,
        }
        print(f"  full R4 ρ={rho_r4:+.4f}  C3 same rows ρ={rho_c3:+.4f}  Δ vs C3 {d_vs_c3}")
        print(f"  subset {r4_primary}")
        print(f"  R4.1 {'PASS' if r4_1 else 'FAIL'}  R4.2 {'PASS' if r4_2 else 'FAIL'}")
        d4.to_csv(OUT / "oct1_r4_lopo.tsv", sep="\t", index=False)

    verdict = {
        "track": "residual_R",
        "prereg": "met_prereg_residual_r.md",
        "R1_R2": r12,
        "abundance_rules": rules,
        "gate": {k: v for k, v in gate.items() if k != "gate_pos"} | {"gate_pos": gate["gate_pos"]},
        "R3": {
            "primary": r3_primary,
            "near_wt": r3_near,
            "gate_all_GFP": r3_gate_only,
            "R3_1_pass": r3_1,
            "R3_2_pass": r3_2,
        },
        "R4": r4,
        "stop_more_features": bool(not r4.get("skipped") and not r4.get("R4_1_pass", False)),
    }
    (OUT / "r_verdict.json").write_text(json.dumps(verdict, indent=2, default=str) + "\n")
    pd.Series(gate["gate_pos"], name="pos").to_csv(OUT / "gate_positions.tsv", sep="\t", index=False)
    print(f"\nwrote {OUT / 'r_verdict.json'}")


if __name__ == "__main__":
    main()
