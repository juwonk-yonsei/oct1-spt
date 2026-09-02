#!/usr/bin/env python3
"""USM helix-LOPO U1–U3 (met_prereg_uptake.md).

    $MET_PY met_uptake_lopo.py
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

MET_HDD = Path(os.environ.get("MET_HDD", str(Path(__file__).resolve().parent / "data")))
MET_SPT = Path(os.environ.get("MET_SPT", str(MET_HDD / "spt")))
OUT = MET_SPT / "uptake"
FEAT = OUT / "oct1_usm_variants.tsv"

N_BOOT = 10_000
RNG = np.random.default_rng(20260812)
RIDGE_ALPHA = 1.0

USM_NUM = [
    "gate_disp", "dist_metformin", "pocket", "pocket_x_charge", "pocket_x_volume",
    "d_charge", "d_volume", "d_hydro", "abs_d_charge",
    "delta_sasa_io", "tm_interface", "gate_x_tm",
    "rel_sasa_8sc1", "plddt",
]


def ridge():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", Ridge(alpha=RIDGE_ALPHA)),
    ])


def add_topo(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for t in ("Transmembrane", "Extracellular", "Cytoplasmic"):
        out[f"topo_{t}"] = (out["topology"] == t).astype(float)
    return out


def usm_cols(df: pd.DataFrame) -> list[str]:
    dummy = [c for c in df.columns if c.startswith("topo_")]
    return [c for c in USM_NUM + dummy if c in df.columns]


def spearman(x, y) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20:
        return float("nan")
    r, _ = spearmanr(x[m], y[m])
    return float(r) if np.isfinite(r) else float("nan")


def residue_delta_rho(df, cola, colb, ycol, n=N_BOOT):
    sub = df[["pos", cola, colb, ycol]].dropna()
    if sub["pos"].nunique() < 5:
        return None
    pos = sub["pos"].to_numpy()
    a = sub[cola].to_numpy(float)
    b = sub[colb].to_numpy(float)
    y = sub[ycol].to_numpy(float)
    order = np.argsort(pos, kind="mergesort")
    pos, a, b, y = pos[order], a[order], b[order], y[order]
    starts = np.r_[0, np.flatnonzero(pos[1:] != pos[:-1]) + 1]
    ends = np.r_[starts[1:], pos.size]
    ga = [a[s:e] for s, e in zip(starts, ends)]
    gb = [b[s:e] for s, e in zip(starts, ends)]
    gy = [y[s:e] for s, e in zip(starts, ends)]
    n_res = len(ga)

    def delta(idx):
        return spearman(np.concatenate([ga[i] for i in idx]),
                        np.concatenate([gy[i] for i in idx])) - \
               spearman(np.concatenate([gb[i] for i in idx]),
                        np.concatenate([gy[i] for i in idx]))

    point = delta(np.arange(n_res))
    draws = RNG.integers(0, n_res, size=(n, n_res))
    boots = np.array([delta(d) for d in draws], dtype=float)
    boots = boots[np.isfinite(boots)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"delta": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
            "n_res": n_res, "n_boot": len(boots)}


def fit_predict(tr, te, cols, y):
    yy = tr[y].to_numpy(float)
    ok = np.isfinite(yy)
    if ok.sum() < 30:
        return np.full(len(te), np.nan)
    m = ridge()
    m.fit(tr.loc[ok, cols], yy[ok])
    return m.predict(te[cols])


def sm73_residual(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fit SM73 ~ GFP on train; return residuals for train and test."""
    y = tr["SM73_0_score"].to_numpy(float)
    g = tr["GFP_score"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(g)
    if ok.sum() < 30:
        return np.full(len(tr), np.nan), np.full(len(te), np.nan)
    X = np.c_[np.ones(ok.sum()), g[ok]]
    coef, *_ = np.linalg.lstsq(X, y[ok], rcond=None)

    def resid(df):
        yy = df["SM73_0_score"].to_numpy(float)
        gg = df["GFP_score"].to_numpy(float)
        out = np.full(len(df), np.nan)
        m = np.isfinite(yy) & np.isfinite(gg)
        out[m] = yy[m] - (coef[0] + coef[1] * gg[m])
        return out

    return resid(tr), resid(te)


def helix_lopo(df, cols, ycol, pred_name, residual_target=False):
    out = df.copy()
    out[pred_name] = np.nan
    if residual_target:
        out["SM73_resid"] = np.nan
    clusters = sorted(c for c in out.loc[out["train_ok"], "cluster"].dropna().unique())
    for cl in clusters:
        tr_m = out["train_ok"] & (out["cluster"] != cl)
        te_m = out["train_ok"] & (out["cluster"] == cl)
        if int(te_m.sum()) < 10 or int(tr_m.sum()) < 50:
            continue
        tr, te = out.loc[tr_m], out.loc[te_m]
        if residual_target:
            r_tr, r_te = sm73_residual(tr, te)
            out.loc[tr_m, "SM73_resid"] = r_tr
            out.loc[te_m, "SM73_resid"] = r_te
            # refit using residual as y — need temp frames
            tr2 = tr.copy()
            te2 = te.copy()
            tr2["_y"] = r_tr
            te2["_y"] = r_te
            pred = fit_predict(tr2, te2, cols, "_y")
        else:
            pred = fit_predict(tr, te, cols, ycol)
        out.loc[te_m, pred_name] = pred
        print(f"  {pred_name:10s} hold {cl:12s} n_te={int(te_m.sum()):4d}")
    return out


def ci_ok(d):
    return bool(d and d["delta"] > 0 and d["ci_lo"] > 0)


def main():
    if not FEAT.exists():
        raise SystemExit(f"missing {FEAT}")
    df = add_topo(pd.read_csv(FEAT, sep="\t"))
    cols = usm_cols(df)
    print("USM features:", cols)
    print("train_ok", int(df["train_ok"].sum()))

    print("\n--- USM LOPO SM73_0 ---")
    df = helix_lopo(df, cols, "SM73_0_score", "usm")
    print("\n--- USM LOPO SM73_resid ---")
    df = helix_lopo(df, cols, "SM73_0_score", "usm_resid", residual_target=True)

    # HGBR sensitivity on SM73_0
    print("\n--- HGBR sensitivity SM73_0 ---")
    df["usm_hgbr"] = np.nan
    clusters = sorted(c for c in df.loc[df["train_ok"], "cluster"].dropna().unique())
    for cl in clusters:
        tr_m = df["train_ok"] & (df["cluster"] != cl)
        te_m = df["train_ok"] & (df["cluster"] == cl)
        if int(te_m.sum()) < 10:
            continue
        tr, te = df.loc[tr_m], df.loc[te_m]
        y = tr["SM73_0_score"].to_numpy(float)
        ok = np.isfinite(y)
        model = HistGradientBoostingRegressor(
            max_depth=3, max_iter=100, learning_rate=0.05,
            min_samples_leaf=50, random_state=0)
        # impute for HGBR via median fill
        Xtr = tr[cols].astype(float)
        Xte = te[cols].astype(float)
        med = Xtr.median()
        Xtr = Xtr.fillna(med)
        Xte = Xte.fillna(med)
        model.fit(Xtr.loc[ok], y[ok])
        df.loc[te_m, "usm_hgbr"] = model.predict(Xte)

    te = df[df["train_ok"] & df["usm"].notna()].copy()
    rho_usm = spearman(te["usm"].to_numpy(), te["SM73_0_score"].to_numpy())
    rho_am = spearman(te["am_fitness"].to_numpy(), te["SM73_0_score"].to_numpy())
    rho_ddg = spearman(te["ddg_fitness"].to_numpy(), te["SM73_0_score"].to_numpy())
    rho_old = spearman(te["u_head"].to_numpy(), te["SM73_0_score"].to_numpy()) if "u_head" in te else float("nan")
    rho_hgbr = spearman(
        df.loc[df["train_ok"] & df["usm_hgbr"].notna(), "usm_hgbr"].to_numpy(),
        df.loc[df["train_ok"] & df["usm_hgbr"].notna(), "SM73_0_score"].to_numpy())

    d_am = residue_delta_rho(te, "usm", "am_fitness", "SM73_0_score")
    d_ddg = residue_delta_rho(te.dropna(subset=["ddg_fitness"]), "usm", "ddg_fitness", "SM73_0_score")
    d_old = residue_delta_rho(te.dropna(subset=["u_head"]), "usm", "u_head", "SM73_0_score") if "u_head" in te else None

    u1 = bool(np.isfinite(rho_usm) and rho_usm > rho_am and rho_usm > rho_ddg
              and (not np.isfinite(rho_old) or rho_usm > rho_old)
              and ci_ok(d_am) and ci_ok(d_ddg) and (d_old is None or ci_ok(d_old)))

    # U2 residual
    te2 = df[df["train_ok"] & df["usm_resid"].notna() & df["SM73_resid"].notna()].copy()
    rho_ur = spearman(te2["usm_resid"].to_numpy(), te2["SM73_resid"].to_numpy())
    rho_am_r = spearman(te2["am_fitness"].to_numpy(), te2["SM73_resid"].to_numpy())
    rho_ddg_r = spearman(te2["ddg_fitness"].to_numpy(), te2["SM73_resid"].to_numpy())
    d_am_r = residue_delta_rho(te2, "usm_resid", "am_fitness", "SM73_resid")
    d_ddg_r = residue_delta_rho(te2.dropna(subset=["ddg_fitness"]), "usm_resid", "ddg_fitness", "SM73_resid")
    u2 = bool(np.isfinite(rho_ur) and rho_ur > rho_am_r and rho_ur > rho_ddg_r
              and ci_ok(d_am_r) and ci_ok(d_ddg_r))

    # U3 pocket only
    pk = te[te["pocket"] == 1]
    rho_usm_p = spearman(pk["usm"].to_numpy(), pk["SM73_0_score"].to_numpy())
    rho_am_p = spearman(pk["am_fitness"].to_numpy(), pk["SM73_0_score"].to_numpy())
    u3 = bool(len(pk) >= 50 and np.isfinite(rho_usm_p) and np.isfinite(rho_am_p) and rho_usm_p > rho_am_p)

    print("\n=== U1 SM73_0 ===")
    print(f"  USM {rho_usm:+.4f}  AM {rho_am:+.4f}  ΔΔG {rho_ddg:+.4f}  TPT-U {rho_old:+.4f}  HGBR {rho_hgbr:+.4f}")
    print(f"  Δ vs AM {d_am}\n  Δ vs ΔΔG {d_ddg}\n  Δ vs TPT-U {d_old}")
    print(f"  U1 {'PASS' if u1 else 'FAIL'}")
    print("\n=== U2 SM73_resid ===")
    print(f"  USM {rho_ur:+.4f}  AM {rho_am_r:+.4f}  ΔΔG {rho_ddg_r:+.4f}")
    print(f"  U2 {'PASS' if u2 else 'FAIL'}")
    print(f"\n=== U3 pocket n={len(pk)} ===")
    print(f"  USM {rho_usm_p:+.4f}  AM {rho_am_p:+.4f}  U3 {'PASS' if u3 else 'FAIL'}")

    df.to_csv(OUT / "oct1_usm_lopo_preds.tsv", sep="\t", index=False)
    summary = {
        "U1_pass": u1, "U2_pass": u2, "U3_pass": u3,
        "rho_usm_sm73": rho_usm, "rho_am_sm73": rho_am, "rho_ddg_sm73": rho_ddg,
        "rho_tpt_u_sm73": rho_old, "rho_hgbr_sm73": rho_hgbr,
        "delta_vs_AM": d_am, "delta_vs_ddG": d_ddg, "delta_vs_tpt_u": d_old,
        "rho_usm_resid": rho_ur, "rho_am_resid": rho_am_r, "rho_ddg_resid": rho_ddg_r,
        "delta_resid_vs_AM": d_am_r, "delta_resid_vs_ddG": d_ddg_r,
        "n_pocket": int(len(pk)), "rho_usm_pocket": rho_usm_p, "rho_am_pocket": rho_am_p,
        "features": cols, "n_eval": int(len(te)),
    }
    (OUT / "usm_verdict.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\nwrote {OUT}/usm_verdict.json")


if __name__ == "__main__":
    main()
