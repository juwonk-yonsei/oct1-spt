#!/usr/bin/env python3
"""C3 — CA-ANM ensemble-gated residual uptake (c3 prereg_lock).

    $MET_PY met_c3_ensemble.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from met_wp5 import load_ca  # noqa: E402

MET_HDD = Path(os.environ.get("MET_HDD", str(Path(__file__).resolve().parent / "data")))
MET_PDB = Path(os.environ.get("MET_PDB", str(MET_HDD / "pdb")))
OUT = MET_HDD / "challenge" / "c3_ensemble"
VDMS = MET_HDD / "challenge" / "c1_vdms" / "oct1_vdms_matrix.tsv"
DPHI = MET_HDD / "challenge" / "c2_electro" / "oct1_dphi.tsv"
USM_PRED = MET_HDD / "spt" / "uptake" / "oct1_usm_lopo_preds.tsv"

RC = 13.0
N_MODES = 20  # after 6 rigid-body
N_BOOT = 10_000
RNG = np.random.default_rng(20260813)
RIDGE_ALPHA = 1.0

ENS_NUM = [
    "anm_msf_8sc1", "anm_msf_8et6", "anm_msf_ratio",
    "dphi_8sc1", "gate_disp", "dist_metformin", "pocket",
    "d_charge", "d_volume", "d_hydro", "abs_d_charge",
    "delta_sasa_io", "rel_sasa_8sc1", "plddt",
]


def anm_msf(ca: dict, cutoff: float = RC, n_modes: int = N_MODES) -> dict[int, float]:
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
    # skip near-zero rigid modes
    nz = np.where(w > 1e-8)[0]
    use = nz[:n_modes]
    msf = np.zeros(n)
    for k in use:
        mode = v[:, k].reshape(n, 3)
        msf += np.sum(mode**2, axis=1) / float(w[k])
    return {pos[i]: float(msf[i]) for i in range(n)}


def ridge():
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


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
        return spearman(
            np.concatenate([ga[i] for i in idx]), np.concatenate([gy[i] for i in idx])
        ) - spearman(
            np.concatenate([gb[i] for i in idx]), np.concatenate([gy[i] for i in idx])
        )

    point = delta(np.arange(n_res))
    draws = RNG.integers(0, n_res, size=(n, n_res))
    boots = np.array([delta(d) for d in draws], dtype=float)
    boots = boots[np.isfinite(boots)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "delta": float(point),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_res": int(n_res),
        "n_boot": int(len(boots)),
    }


def sm73_residual(tr, te):
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


def fit_predict(tr, te, cols, y):
    yy = tr[y].to_numpy(float)
    ok = np.isfinite(yy)
    if ok.sum() < 30:
        return np.full(len(te), np.nan)
    m = ridge()
    m.fit(tr.loc[ok, cols], yy[ok])
    return m.predict(te[cols])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("ANM 8SC1...")
    _, ca1, _ = load_ca(MET_PDB / "8SC1.pdb")
    msf1 = anm_msf(ca1)
    print(f"  residues {len(msf1)}")
    print("ANM 8ET6...")
    _, ca6, _ = load_ca(MET_PDB / "8ET6.pdb")
    msf6 = anm_msf(ca6)
    print(f"  residues {len(msf6)}")

    df = pd.read_csv(VDMS, sep="\t")
    df["anm_msf_8sc1"] = df["pos"].map(msf1)
    df["anm_msf_8et6"] = df["pos"].map(msf6)
    df["anm_msf_ratio"] = df["anm_msf_8et6"] / df["anm_msf_8sc1"]
    dphi = pd.read_csv(DPHI, sep="\t")[["hgvs_short", "dphi_8sc1"]]
    if "dphi_8sc1" in df.columns:
        df = df.drop(columns=["dphi_8sc1"])
    df = df.merge(dphi, on="hgvs_short", how="left")
    usm = pd.read_csv(USM_PRED, sep="\t")
    u_cols = [c for c in ("usm_resid", "SM73_resid", "am_fitness") if c in usm.columns]
    df = df.drop(columns=[c for c in u_cols if c in df.columns], errors="ignore")
    df = df.merge(usm[["hgvs_short"] + u_cols], on="hgvs_short", how="left")

    for t in ("Transmembrane", "Extracellular", "Cytoplasmic"):
        df[f"topo_{t}"] = (df["topology"] == t).astype(float)
    cols = [c for c in ENS_NUM + [f"topo_{t}" for t in ("Transmembrane", "Extracellular", "Cytoplasmic")] if c in df.columns]
    print("ENS features:", cols)

    df["ens_resid"] = np.nan
    df["SM73_resid_fold"] = np.nan
    clusters = sorted(c for c in df.loc[df["train_ok"], "cluster"].dropna().unique())
    print("\n--- ENS LOPO residual ---")
    for cl in clusters:
        tr_m = df["train_ok"] & (df["cluster"] != cl)
        te_m = df["train_ok"] & (df["cluster"] == cl)
        if int(te_m.sum()) < 10 or int(tr_m.sum()) < 50:
            continue
        tr, te = df.loc[tr_m], df.loc[te_m]
        r_tr, r_te = sm73_residual(tr, te)
        df.loc[te_m, "SM73_resid_fold"] = r_te
        tr2, te2 = tr.copy(), te.copy()
        tr2["_y"], te2["_y"] = r_tr, r_te
        df.loc[te_m, "ens_resid"] = fit_predict(tr2, te2, cols, "_y")
        print(f"  hold {cl:12s} n_te={int(te_m.sum()):4d}")

    te = df[df["train_ok"] & df["ens_resid"].notna() & df["SM73_resid_fold"].notna()].copy()
    rho_ens = spearman(te["ens_resid"].to_numpy(), te["SM73_resid_fold"].to_numpy())
    rho_am = spearman(te["am_fitness"].to_numpy(), te["SM73_resid_fold"].to_numpy()) if "am_fitness" in te else float("nan")
    d_am = residue_delta_rho(te, "ens_resid", "am_fitness", "SM73_resid_fold") if "am_fitness" in te else None
    c3_1 = bool(
        np.isfinite(rho_ens)
        and np.isfinite(rho_am)
        and rho_ens > rho_am
        and d_am
        and d_am["delta"] > 0
        and d_am["ci_lo"] > 0
    )

    te2 = te.dropna(subset=["usm_resid"]) if "usm_resid" in te.columns else te.iloc[0:0]
    rho_usm = spearman(te2["usm_resid"].to_numpy(), te2["SM73_resid_fold"].to_numpy()) if len(te2) else float("nan")
    d_usm = residue_delta_rho(te2, "ens_resid", "usm_resid", "SM73_resid_fold") if len(te2) else None
    c3_2 = bool(
        np.isfinite(rho_ens)
        and np.isfinite(rho_usm)
        and rho_ens > rho_usm
        and d_usm
        and d_usm["delta"] > 0
        and d_usm["ci_lo"] > 0
    )

    pred_path = OUT / "oct1_ens_lopo.tsv"
    df.to_csv(pred_path, sep="\t", index=False)
    verdict = {
        "C3_1_pass": c3_1,
        "C3_2_pass": c3_2,
        "pass": bool(c3_1 and c3_2),
        "degraded_ok_for_next": True,
        "method": "ca_anm",
        "rc": RC,
        "n_modes": N_MODES,
        "rho_ens_resid": rho_ens,
        "rho_am_resid": rho_am,
        "rho_usm_resid": rho_usm,
        "delta_vs_AM": d_am,
        "delta_vs_USM": d_usm,
        "features": cols,
        "n_eval": int(len(te)),
        "artifacts": {"preds": str(pred_path)},
    }
    (OUT / "c3_verdict.json").write_text(json.dumps(verdict, indent=2, default=str) + "\n")
    print("\n=== C3.1 vs AM ===")
    print(f"  ENS {rho_ens:+.4f}  AM {rho_am:+.4f}  {d_am}")
    print(f"  C3.1 {'PASS' if c3_1 else 'FAIL'}")
    print("=== C3.2 vs USM ===")
    print(f"  ENS {rho_ens:+.4f}  USM {rho_usm:+.4f}  {d_usm}")
    print(f"  C3.2 {'PASS' if c3_2 else 'FAIL'}")
    print(f"C3={'PASS' if c3_1 and c3_2 else 'DEGRADED'}")


if __name__ == "__main__":
    main()
