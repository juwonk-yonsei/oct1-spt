#!/usr/bin/env python3
"""Feedback-260901 reanalysis (A1–A3, B1–B2, C).

Does not retune SPT 10%/30%. Design positions 61/88/401/420/465 stay excluded.
Locked GFP-loss cutoff −0.814; AM pathogenic >0.564; AM benign <0.34.
P4 uses dms_loss only (never func_loss n=493).
Residue-clustered bootstrap: 10 000 draws, seed 20260812.

    source /SSD1T/PhD/AlphaFold/met_env.sh
    $MET_PY met_fb260901.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MET_HDD = Path(os.environ.get("MET_HDD", "/HDD8T1/WORK/Metformin_HDD"))
MET_SPT = Path(os.environ.get("MET_SPT", str(MET_HDD / "spt")))
MET_AM = Path(os.environ.get("MET_AM", str(MET_HDD / "alphamissense")))
MET_STRUCT = Path(os.environ.get("MET_STRUCT", str(MET_HDD / "structures")))
OUT = MET_SPT / "fb260901"
MS1 = Path("/SSD1T/PhD/AlphaFold/_manuscript_1")

DESIGN_POS = {61, 88, 401, 420, 465}
GFP_CUT = -0.814
AM_PATH = 0.564
AM_BENIGN = 0.34
N_BOOT = int(os.environ.get("FB_N_BOOT", "10000"))
SEED = 20260812
AA20 = list("ARNDCQEGHILKMFPSTWYV")
AA_SET = set(AA20)

# Robinson–Robinson frequencies (Capra & Singh 2007 background)
AA_BG = np.array(
    [0.078, 0.051, 0.041, 0.052, 0.024, 0.034, 0.059, 0.083,
     0.025, 0.062, 0.092, 0.056, 0.024, 0.044, 0.043, 0.059,
     0.055, 0.014, 0.034, 0.072],
    dtype=float,
)
AA_BG = AA_BG / AA_BG.sum()

# Grantham 1974 distances
# Grantham 1974; rows/cols in AA20 order. Letter must be space-separated from numbers.
_GRANTHAM_ROWS = """
A 0 112 111 126 195 91 107 60 86 94 96 106 84 113 27 99 58 148 112 64
R 112 0 86 96 180 43 54 125 29 97 102 26 91 97 103 110 71 101 77 96
N 111 86 0 23 139 46 42 80 68 149 153 94 142 158 91 46 65 174 143 133
D 126 96 23 0 154 61 45 94 81 168 172 101 160 177 108 65 85 181 160 152
C 195 180 139 154 0 154 170 159 174 198 198 202 196 205 169 112 149 215 194 192
Q 91 43 46 61 154 0 29 87 24 109 113 53 101 116 76 68 42 130 99 96
E 107 54 42 45 170 29 0 98 40 134 138 56 126 140 93 80 65 152 122 121
G 60 125 80 94 159 87 98 0 98 135 138 127 127 153 42 56 59 184 147 109
H 86 29 68 81 174 24 40 98 0 94 99 32 87 100 77 89 47 115 83 84
I 94 97 149 168 198 109 134 135 94 0 5 102 10 21 95 142 89 61 33 29
L 96 102 153 172 198 113 138 138 99 5 0 107 15 22 98 145 92 61 36 32
K 106 26 94 101 202 53 56 127 32 102 107 0 95 102 103 121 78 110 85 97
M 84 91 142 160 196 101 126 127 87 10 15 95 0 28 87 135 81 67 36 21
F 113 97 158 177 205 116 140 153 100 21 22 102 28 0 114 155 103 40 22 50
P 27 103 91 108 169 76 93 42 77 95 98 103 87 114 0 74 38 147 110 68
S 99 110 46 65 112 68 80 56 89 142 145 121 135 155 74 0 58 177 144 124
T 58 71 65 85 149 42 65 59 47 89 92 78 81 103 38 58 0 128 92 69
W 148 101 174 181 215 130 152 184 115 61 61 110 67 40 147 177 128 0 37 88
Y 112 77 143 160 194 99 122 147 83 33 36 85 36 22 110 144 92 37 0 55
V 64 96 133 152 192 96 121 109 84 29 32 97 21 50 68 124 69 88 55 0
""".strip().splitlines()
_GRANTHAM = {}
for line in _GRANTHAM_ROWS:
    parts = line.split()
    a = parts[0]
    if len(a) != 1 or len(parts) != 21:
        raise ValueError(f"Grantham row parse failed: {parts[:3]} n={len(parts)}")
    for b, v in zip(AA20, parts[1:]):
        _GRANTHAM[(a, b)] = int(v)
if len(_GRANTHAM) != 400:
    raise ValueError(f"Grantham incomplete: {len(_GRANTHAM)}")


def grantham(a: str, b: str) -> float:
    return float(_GRANTHAM.get((a, b), _GRANTHAM.get((b, a), np.nan)))


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def dump(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=json_default) + "\n")


def fisher_or_table(a, b, c, d):
    tab = np.array([[a, b], [c, d]], dtype=int)
    if tab.min() < 0 or tab.sum() == 0:
        return {"or": np.nan, "p": np.nan, "a": int(a), "b": int(b), "c": int(c), "d": int(d)}
    oddsr, p = stats.fisher_exact(tab)
    lo, hi = woolf_ci(a, b, c, d)
    return {"or": float(oddsr), "p": float(p), "a": int(a), "b": int(b), "c": int(c),
            "d": int(d), "ci95": [lo, hi]}


def woolf_ci(a, b, c, d, z=1.96):
    a, b, c, d = [max(0.5, float(x)) for x in (a, b, c, d)]
    logor = np.log((a * d) / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return float(np.exp(logor - z * se)), float(np.exp(logor + z * se))


def or_from_counts(a, b, c, d):
    if min(a, b, c, d) <= 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return (a * d) / (b * c)


def grouped_indices(pos: np.ndarray):
    order = np.argsort(pos, kind="mergesort")
    pos = pos[order]
    starts = np.r_[0, np.flatnonzero(pos[1:] != pos[:-1]) + 1]
    ends = np.r_[starts[1:], pos.size]
    groups = [order[s:e] for s, e in zip(starts, ends)]
    return groups


def clustered_bootstrap(groups, stat_fn, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n_res = len(groups)
    point = stat_fn(np.concatenate(groups))
    draws = rng.integers(0, n_res, size=(n, n_res))
    boots = []
    for d in draws:
        idx = np.concatenate([groups[i] for i in d])
        v = stat_fn(idx)
        if np.isfinite(v):
            boots.append(v)
    boots = np.asarray(boots, dtype=float)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "point": float(point) if np.isfinite(point) else np.nan,
        "median": float(np.median(boots)) if boots.size else np.nan,
        "ci95": [float(lo), float(hi)] if boots.size else [np.nan, np.nan],
        "n_boot": int(boots.size),
        "n_residues": n_res,
        "frac_le_1": float((boots <= 1).mean()) if boots.size else np.nan,
        "frac_le_0": float((boots <= 0).mean()) if boots.size else np.nan,
    }


def mannwhitney_auroc(scores, labels):
    """AUROC of scores predicting label=1. Ties handled by Mann–Whitney."""
    s1 = scores[labels == 1]
    s0 = scores[labels == 0]
    if s1.size < 1 or s0.size < 1:
        return np.nan
    u, _ = stats.mannwhitneyu(s1, s0, alternative="greater")
    return float(u / (s1.size * s0.size))


def logit_fit(X, y, niter=30):
    """Newton–Raphson logistic regression. X includes intercept. Returns beta."""
    beta = np.zeros(X.shape[1], dtype=float)
    for _ in range(niter):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        p = np.clip(p, 1e-8, 1 - 1e-8)
        w = p * (1 - p)
        xtw = X.T * w
        h = xtw @ X
        g = X.T @ (y - p)
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(h + 1e-6 * np.eye(h.shape[0]), g, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def parse_a3m(path: Path):
    seqs = []
    name, buf = None, []
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs.append((name, "".join(buf)))
                name, buf = line[1:], []
            else:
                buf.append(line)
        if name is not None:
            seqs.append((name, "".join(buf)))
    return seqs


def match_states(seq: str) -> str:
    return "".join(c.upper() if c != "-" else "-" for c in seq if not c.islower())


def conservation_from_a3m(path: Path, query_len: int = 554) -> pd.DataFrame:
    seqs = parse_a3m(path)
    if not seqs:
        raise SystemExit(f"empty a3m {path}")
    query = match_states(seqs[0][1]).replace("-", "")
    if len(query) != query_len:
        # keep ungapped query length
        query_len = len(query)
    cols = np.zeros((query_len, 20), dtype=float)
    n_gap = np.zeros(query_len, dtype=float)
    n_obs = np.zeros(query_len, dtype=float)
    n_used = 0
    for _, raw in seqs:
        ms = match_states(raw)
        if len(ms) != query_len:
            continue
        n_used += 1
        for i, c in enumerate(ms):
            if c in AA_SET:
                cols[i, AA20.index(c)] += 1.0
                n_obs[i] += 1.0
            else:
                n_gap[i] += 1.0
    js = np.full(query_len, np.nan)
    shan = np.full(query_len, np.nan)
    n_eff = np.zeros(query_len)
    for i in range(query_len):
        if n_obs[i] < 10:
            continue
        p = cols[i] / cols[i].sum()
        n_eff[i] = n_obs[i]
        # JS vs background (higher = more conserved)
        m = 0.5 * (p + AA_BG)
        def H(v):
            v = v[v > 0]
            return float(-(v * np.log2(v)).sum())
        js[i] = H(m) - 0.5 * H(p) - 0.5 * H(AA_BG)
        h = H(p)
        shan[i] = 1.0 - h / np.log2(20.0)
    return pd.DataFrame({
        "pos": np.arange(1, query_len + 1),
        "wt_aa_msa": list(query),
        "cons_js": js,
        "cons_shannon": shan,
        "msa_n_obs": n_obs,
        "msa_occupancy": n_obs / np.maximum(n_used, 1),
        "msa_n_seqs_used": n_used,
    })


def class_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {"n": int(y_true.size), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sens": sens, "spec": spec, "ppv": ppv, "npv": npv}


def youden_threshold(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(scores)
    s = scores[order]
    y = labels[order]
    # threshold = score; predict 1 if score > t
    uniq = np.unique(s)
    best_j, best_t = -np.inf, np.nan
    P = y.sum()
    N = y.size - P
    # cumulative positives below or equal each point, scan unique
    # evaluate t as midpoints plus min-eps
    cands = np.r_[s.min() - 1e-6, (uniq[:-1] + uniq[1:]) / 2.0, s.max()]
    for t in cands:
        pred = scores > t
        tp = int((pred & (labels == 1)).sum())
        tn = int((~pred & (labels == 0)).sum())
        sens = tp / P if P else 0
        spec = tn / N if N else 0
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t, float(best_j)


def precision_threshold(scores, labels, target=0.90):
    """Lowest threshold with PPV >= target (pathogenic call precision)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    uniq = np.unique(scores)
    best_t, best_sens = np.nan, -1.0
    P = labels.sum()
    for t in uniq:
        pred = scores > t
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        if tp + fp == 0:
            continue
        ppv = tp / (tp + fp)
        sens = tp / P if P else 0
        if ppv >= target and sens > best_sens:
            best_sens = sens
            best_t = float(t)
    return best_t, float(best_sens)


def recall_match_threshold(scores, labels, target_recall):
    """Smallest threshold such that recall among positives >= target (more calls).
    For EXPOSED matching CORE recall we want the *highest* t that still achieves
    the target recall (stricter), or the t that matches recall most closely.
    Use closest recall to target.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    if pos.size == 0:
        return np.nan
    uniq = np.unique(scores)
    best_t, best_err = np.nan, np.inf
    P = pos.size
    for t in uniq:
        rec = float((pos > t).mean())
        err = abs(rec - target_recall)
        if err < best_err:
            best_err, best_t = err, float(t)
    return best_t


def mantel_haenszel(tables):
    """tables: list of (a,b,c,d) 2x2. Returns common OR and Robins–Greenland CI."""
    num = 0.0
    den = 0.0
    var_num = 0.0
    for a, b, c, d in tables:
        a, b, c, d = map(float, (a, b, c, d))
        n = a + b + c + d
        if n == 0:
            continue
        num += a * d / n
        den += b * c / n
        var_num += ((a + d) / n) * (a * d / n) + ((b + c) / n) * (b * c / n)
        # Robins, Greenland, Breslow SE of log MH OR
    if den <= 0 or num <= 0:
        return {"or": np.nan, "ci95": [np.nan, np.nan], "n_strata": len(tables)}
    or_mh = num / den
    # Robins-Breslow-Greenland variance
    r = num
    s = den
    p1 = p2 = q1 = q2 = 0.0
    for a, b, c, d in tables:
        a, b, c, d = map(float, (a, b, c, d))
        n = a + b + c + d
        if n == 0:
            continue
        p1 += (a * d / n) * ((a + d) / n)
        p2 += (a * d / n) * ((b + c) / n)
        q1 += (b * c / n) * ((a + d) / n)
        q2 += (b * c / n) * ((b + c) / n)
    if r <= 0 or s <= 0:
        se = np.nan
    else:
        se = np.sqrt(p1 / (2 * r * r) + (p2 + q1) / (2 * r * s) + q2 / (2 * s * s))
    lo, hi = np.exp(np.log(or_mh) - 1.96 * se), np.exp(np.log(or_mh) + 1.96 * se)
    return {"or": float(or_mh), "ci95": [float(lo), float(hi)], "n_strata": int(len(tables)), "se_log": float(se)}


def p1_stats(val, class_col):
    res = (val.dropna(subset=[class_col, "GFP_score"])
           .groupby(["pos", class_col], as_index=False)["GFP_score"].median())
    core = res.loc[res[class_col] == "CORE", "GFP_score"]
    exp = res.loc[res[class_col] == "EXPOSED", "GFP_score"]
    if len(core) < 3 or len(exp) < 3:
        return {"n_core": int(len(core)), "n_exposed": int(len(exp)),
                "median_core": np.nan, "median_exposed": np.nan, "p": np.nan}
    u, p = stats.mannwhitneyu(core, exp, alternative="less")
    return {
        "n_core": int(len(core)), "n_exposed": int(len(exp)),
        "median_core": float(core.median()), "median_exposed": float(exp.median()),
        "p": float(p), "U": float(u),
    }


def p3_stats(ddg, class_col):
    sub = ddg.dropna(subset=[class_col, "ddg"])
    core = sub.loc[sub[class_col] == "CORE", "ddg"]
    exp = sub.loc[sub[class_col] == "EXPOSED", "ddg"]
    if len(core) < 3 or len(exp) < 3:
        return {"n_core": int(len(core)), "n_exposed": int(len(exp)),
                "median_core": np.nan, "median_exposed": np.nan, "p": np.nan}
    u, p = stats.mannwhitneyu(core, exp, alternative="greater")
    return {
        "n_core": int(len(core)), "n_exposed": int(len(exp)),
        "median_core": float(core.median()), "median_exposed": float(exp.median()),
        "p": float(p), "U": float(u),
    }


def recall_gap(val, class_col):
    g = val[(val["dms_loss"]) & (val[class_col].isin(["CORE", "EXPOSED"]))].copy()
    g = g.dropna(subset=["am_class", class_col])
    def rec(lab):
        s = g[g[class_col] == lab]
        n = len(s)
        n_path = int((s["am_class"] == "pathogenic").sum())
        return n, n_path, n_path / n if n else np.nan
    n_c, np_c, r_c = rec("CORE")
    n_e, np_e, r_e = rec("EXPOSED")
    a, b, c, d = np_c, n_c - np_c, np_e, n_e - np_e
    fish = fisher_or_table(a, b, c, d)
    # clustered bootstrap of OR
    g = g.reset_index(drop=True)
    groups = grouped_indices(g["pos"].to_numpy())
    am_path = (g["am_class"] == "pathogenic").to_numpy()
    is_core = (g[class_col] == "CORE").to_numpy()

    def stat(idx):
        ic = is_core[idx]
        ip = am_path[idx]
        a_ = int((ic & ip).sum())
        b_ = int((ic & ~ip).sum())
        c_ = int((~ic & ip).sum())
        d_ = int((~ic & ~ip).sum())
        return or_from_counts(a_, b_, c_, d_)

    boot = clustered_bootstrap(groups, stat)
    return {
        "n_loss_core": n_c, "n_path_core": np_c, "recall_core": r_c,
        "n_loss_exposed": n_e, "n_path_exposed": np_e, "recall_exposed": r_e,
        "or_variant": fish,
        "or_clustered": boot,
    }


def a1_model(val, class_col, cons_col, n_boot=N_BOOT):
    """GFP-loss CORE+EXPOSED; outcome AM-pathogenic; EXPOSED + conservation + Grantham."""
    g = val[(val["dms_loss"]) & (val[class_col].isin(["CORE", "EXPOSED"]))].copy()
    g = g.dropna(subset=["am_class", class_col, cons_col, "grantham"])
    g["y"] = (g["am_class"] == "pathogenic").astype(int)
    g["exposed"] = (g[class_col] == "EXPOSED").astype(float)
    g["cons"] = g[cons_col].astype(float)
    g["gran"] = g["grantham"].astype(float)
    n = len(g)
    n_res = int(g["pos"].nunique())
    a = int(((g[class_col] == "CORE") & (g["am_class"] == "pathogenic")).sum())
    b = int(((g[class_col] == "CORE") & (g["am_class"] != "pathogenic")).sum())
    c = int(((g[class_col] == "EXPOSED") & (g["am_class"] == "pathogenic")).sum())
    d = int(((g[class_col] == "EXPOSED") & (g["am_class"] != "pathogenic")).sum())
    unadj = fisher_or_table(a, b, c, d)
    X = np.column_stack([
        np.ones(n),
        g["exposed"].to_numpy(),
        g["cons"].to_numpy(),
        g["gran"].to_numpy(),
    ])
    y = g["y"].to_numpy(dtype=float)
    beta = logit_fit(X, y)
    or_pt = float(np.exp(beta[1]))
    # clustered bootstrap of adjusted OR
    g = g.reset_index(drop=True)
    groups = grouped_indices(g["pos"].to_numpy())
    Xg = X
    yg = y

    def stat(idx):
        b = logit_fit(Xg[idx], yg[idx])
        return float(np.exp(b[1]))

    boot = clustered_bootstrap(groups, stat, n=n_boot)
    boot["or_core_vs_exposed_median"] = (1.0 / boot["median"]) if boot["median"] else np.nan
    boot["or_core_vs_exposed_ci95"] = (
        [1.0 / boot["ci95"][1], 1.0 / boot["ci95"][0]]
        if boot["ci95"] and np.isfinite(boot["ci95"][0]) and boot["ci95"][0] != 0
        else [np.nan, np.nan]
    )
    # GEE
    gee = {"ok": False}
    try:
        import statsmodels.api as sm
        from statsmodels.genmod.cov_struct import Exchangeable, Independence
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.generalized_estimating_equations import GEE
        Xx = sm.add_constant(g[["exposed", "cons", "gran"]])
        for cov_name, cov in (("exchangeable", Exchangeable()), ("independence", Independence())):
            try:
                md = GEE(g["y"], Xx, groups=g["pos"], family=Binomial(), cov_struct=cov)
                rs = md.fit()
                ci = rs.conf_int().loc["exposed"]
                gee = {
                    "ok": True, "cov_struct": cov_name,
                    "or": float(np.exp(rs.params["exposed"])),
                    "ci95": [float(np.exp(ci.iloc[0])), float(np.exp(ci.iloc[1]))],
                    "p": float(rs.pvalues["exposed"]),
                    "beta": {k: float(v) for k, v in rs.params.items()},
                }
                break
            except Exception as e:
                gee = {"ok": False, "cov_struct": cov_name, "error": str(e)}
    except Exception as e:
        gee = {"ok": False, "error": str(e)}

    # residue-level conservation vs class (descriptive, one row per residue)
    res = (g.groupby("pos")
           .agg(exposed=("exposed", "first"), cons=("cons", "first"),
                cls=(class_col, "first"))
           .reset_index())
    core_c = res.loc[res["cls"] == "CORE", "cons"]
    exp_c = res.loc[res["cls"] == "EXPOSED", "cons"]
    if len(core_c) and len(exp_c):
        u, p_cons = stats.mannwhitneyu(core_c, exp_c, alternative="greater")
        cons_desc = {
            "median_core": float(core_c.median()),
            "median_exposed": float(exp_c.median()),
            "n_core": int(len(core_c)), "n_exposed": int(len(exp_c)),
            "p_core_greater": float(p_cons),
            "spearman_exposed_vs_cons": float(stats.spearmanr(res["exposed"], res["cons"]).statistic),
        }
    else:
        cons_desc = {}

    # quintiles on residue conservation, then map back
    res = res.dropna(subset=["cons"]).copy()
    try:
        res["q"] = pd.qcut(res["cons"], 5, labels=False, duplicates="drop")
    except ValueError:
        res["q"] = 0
    qmap = dict(zip(res["pos"], res["q"]))
    g["q"] = g["pos"].map(qmap)
    strata = []
    tables = []
    for q in sorted(g["q"].dropna().unique()):
        sq = g[g["q"] == q]
        def rec(lab):
            s = sq[sq[class_col] == lab]
            nloc = len(s)
            npath = int((s["am_class"] == "pathogenic").sum())
            return nloc, npath, npath / nloc if nloc else np.nan
        nc, npc, rc = rec("CORE")
        ne, npe, re_ = rec("EXPOSED")
        a, b, c, d = npc, nc - npc, npe, ne - npe
        tables.append((a, b, c, d))
        strata.append({
            "quintile": int(q) + 1,
            "cons_min": float(sq["cons"].min()),
            "cons_max": float(sq["cons"].max()),
            "n_core": nc, "recall_core": rc,
            "n_exposed": ne, "recall_exposed": re_,
            "or": fisher_or_table(a, b, c, d),
        })
    mh = mantel_haenszel(tables)
    return {
        "n_variants": n, "n_residues": n_res,
        "unadjusted_or": unadj,
        "logit_or_point": or_pt,
        "logit_or_clustered": boot,
        "gee": gee,
        "conservation_by_class": cons_desc,
        "quintiles": strata,
        "mantel_haenszel": mh,
        "cons_col": cons_col,
        "class_col": class_col,
    }


def p4_clustered(val, class_col):
    """AM-benign ∩ dms_loss EXPOSED enrichment vs rest. dms_loss only."""
    bg = val.dropna(subset=["am_class", class_col]).copy()
    bg["hit"] = (bg["am_class"] == "benign") & bg["dms_loss"]
    hit = bg[bg["hit"]]
    rest = bg[~bg["hit"]]
    a = int((hit[class_col] == "EXPOSED").sum())
    b = int((hit[class_col] != "EXPOSED").sum())
    c = int((rest[class_col] == "EXPOSED").sum())
    d = int((rest[class_col] != "EXPOSED").sum())
    fish = fisher_or_table(a, b, c, d)
    bg = bg.reset_index(drop=True)
    groups = grouped_indices(bg["pos"].to_numpy())
    hit_v = bg["hit"].to_numpy()
    exp_v = (bg[class_col] == "EXPOSED").to_numpy()

    def stat(idx):
        h = hit_v[idx]
        e = exp_v[idx]
        aa = int((h & e).sum())
        bb = int((h & ~e).sum())
        cc = int((~h & e).sum())
        dd = int((~h & ~e).sum())
        return or_from_counts(aa, bb, cc, dd)

    boot = clustered_bootstrap(groups, stat)
    return {
        "n_hit": int(hit_v.sum()),
        "n_exposed_hit": a,
        "frac_exposed_hit": a / (a + b) if (a + b) else np.nan,
        "frac_exposed_bg": float((bg[class_col] == "EXPOSED").mean()),
        "or_variant": fish,
        "or_clustered": boot,
        "loss_definition": "dms_loss (GFP <= -0.814); NOT func_loss",
    }


def auroc_clustered(val, class_name, class_col):
    sub = val[val[class_col] == class_name].dropna(subset=["am_pathogenicity", "dms_loss"])
    if sub.empty:
        return {}
    sub = sub.reset_index(drop=True)
    scores = sub["am_pathogenicity"].to_numpy(dtype=float)
    labels = sub["dms_loss"].astype(int).to_numpy()
    groups = grouped_indices(sub["pos"].to_numpy())

    def stat(idx):
        return mannwhitney_auroc(scores[idx], labels[idx])

    boot = clustered_bootstrap(groups, stat)
    # Hanley–McNeil for comparison
    point = mannwhitney_auroc(scores, labels)
    n1 = int(labels.sum())
    n0 = int((1 - labels).sum())
    q1 = point ** 2 / (2 - point) if 0 < point < 1 else np.nan
    q2 = 2 * point ** 2 / (1 + point) if 0 < point < 1 else np.nan
    inside = (point * (1 - point) + (n1 - 1) * (q1 - point ** 2) + (n0 - 1) * (q2 - point ** 2)) / (n1 * n0) if n1 * n0 else np.nan
    se = np.sqrt(inside) if np.isfinite(inside) and inside >= 0 else np.nan
    return {
        "n": int(len(sub)), "n_loss": n1, "auroc": point,
        "hanley_mcneil_ci95": [float(point - 1.96 * se), float(point + 1.96 * se)] if np.isfinite(se) else [np.nan, np.nan],
        "clustered": boot,
    }


def cutoff_performance(val, t, class_col=None):
    out = {}
    labs = ["ALL"]
    if class_col:
        labs += ["CORE", "EXPOSED", "GREY"]
    for lab in labs:
        sub = val if lab == "ALL" else val[val[class_col] == lab]
        sub = sub.dropna(subset=["am_pathogenicity", "dms_loss"])
        y = sub["dms_loss"].astype(int).to_numpy()
        pred = (sub["am_pathogenicity"] > t).astype(int).to_numpy()
        m = class_metrics(y, pred)
        m["class"] = lab
        m["threshold"] = float(t)
        out[lab] = m
    return out


def helix_lopo_cutoffs(val, clusters):
    """Leave-one-cluster-out cutoff estimation. Never train and test on same residues."""
    df = val.dropna(subset=["am_pathogenicity", "dms_loss", "cluster"]).copy()
    folds = []
    for cl in sorted(df["cluster"].unique()):
        tr = df[df["cluster"] != cl]
        te = df[df["cluster"] == cl]
        if len(te) < 20 or len(tr) < 100:
            continue
        ytr = tr["dms_loss"].astype(int).to_numpy()
        str_ = tr["am_pathogenicity"].to_numpy(dtype=float)
        t_youden, j = youden_threshold(str_, ytr)
        core_loss = tr[(tr["class"] == "CORE") & tr["dms_loss"]]
        target_rec = float((core_loss["am_class"] == "pathogenic").mean()) if len(core_loss) else 0.783
        exp_tr = tr[(tr["class"] == "EXPOSED") & tr["dms_loss"]]
        t_match = recall_match_threshold(
            exp_tr["am_pathogenicity"].to_numpy(), np.ones(len(exp_tr), dtype=int), target_rec
        ) if len(exp_tr) >= 5 else np.nan
        t_prec, _ = precision_threshold(str_, ytr, 0.90)
        # conservation-adjusted: logit P(loss) ~ AM + cons_js
        tr2 = tr.dropna(subset=["cons_js"])
        te2 = te.dropna(subset=["cons_js"])
        t_cons = np.nan
        cons_held = {}
        if len(tr2) > 50 and tr2["dms_loss"].nunique() == 2:
            X = np.column_stack([
                np.ones(len(tr2)),
                tr2["am_pathogenicity"].to_numpy(dtype=float),
                tr2["cons_js"].to_numpy(dtype=float),
            ])
            beta = logit_fit(X, tr2["dms_loss"].astype(float).to_numpy())
            eta = np.clip(X @ beta, -30, 30)
            p_tr = 1 / (1 + np.exp(-eta))
            t_cons, _ = youden_threshold(p_tr, tr2["dms_loss"].astype(int).to_numpy())
            if len(te2):
                Xe = np.column_stack([
                    np.ones(len(te2)),
                    te2["am_pathogenicity"].to_numpy(dtype=float),
                    te2["cons_js"].to_numpy(dtype=float),
                ])
                pe = 1 / (1 + np.exp(-np.clip(Xe @ beta, -30, 30)))
                cons_held = cutoff_performance(
                    te2.assign(am_pathogenicity=pe, dms_loss=te2["dms_loss"]),
                    t_cons, class_col="class",
                )
        fold = {
            "held_out": str(cl),
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "t_youden": t_youden, "youden_j_train": j,
            "t_exposed_match_core_recall": t_match,
            "train_core_recall_target": target_rec,
            "t_precision90": t_prec,
            "t_cons_prob_youden": t_cons,
            "heldout_youden": cutoff_performance(te, t_youden, "class"),
            "heldout_cheng": cutoff_performance(te, AM_PATH, "class"),
            "heldout_prec90": cutoff_performance(te, t_prec, "class") if np.isfinite(t_prec) else {},
            "heldout_match": cutoff_performance(te, t_match, "class") if np.isfinite(t_match) else {},
            "heldout_cons_adj": cons_held,
        }
        folds.append(fold)
    # summarise numeric cutoffs
    def med(key):
        xs = [f[key] for f in folds if np.isfinite(f.get(key, np.nan))]
        return {"median": float(np.median(xs)), "mean": float(np.mean(xs)),
                "min": float(np.min(xs)), "max": float(np.max(xs)), "n_folds": len(xs)} if xs else {}

    def mean_sens(block_key, lab="EXPOSED"):
        xs = []
        for f in folds:
            b = f.get(block_key) or {}
            m = b.get(lab) or {}
            if m.get("sens") is not None and np.isfinite(m["sens"]):
                xs.append(m["sens"])
        return {"mean_sens": float(np.mean(xs)), "n": len(xs)} if xs else {}

    return {
        "n_folds": len(folds),
        "cutoff_youden": med("t_youden"),
        "cutoff_match_core_recall": med("t_exposed_match_core_recall"),
        "cutoff_precision90": med("t_precision90"),
        "cutoff_cons_prob": med("t_cons_prob_youden"),
        "heldout_EXPOSED_sens_youden": mean_sens("heldout_youden"),
        "heldout_EXPOSED_sens_cheng": mean_sens("heldout_cheng"),
        "heldout_EXPOSED_sens_match": mean_sens("heldout_match"),
        "heldout_EXPOSED_sens_prec90": mean_sens("heldout_prec90"),
        "heldout_CORE_sens_youden": mean_sens("heldout_youden", "CORE"),
        "heldout_CORE_sens_cheng": mean_sens("heldout_cheng", "CORE"),
        "folds": folds,
    }


def parse_hgvsp(hgvsp: str | None):
    if not hgvsp:
        return None
    s = hgvsp.replace("%3D", "=")
    s = s.replace("p.", "")
    m = re.search(r"([A-Za-z]{3})(\d+)([A-Za-z]{3}|=)", s)
    if not m:
        return None
    three = {
        "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
        "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
        "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
        "Tyr": "Y", "Val": "V", "Ter": "*", "Sec": "U",
    }
    wt, pos, mut = m.group(1), m.group(2), m.group(3)
    if wt not in three or mut not in three:
        return None
    if mut == "*" or wt == "*":
        return None
    return f"{three[wt]}{pos}{three[mut]}"


def fetch_gnomad():
    query = """
    query {
      gene(gene_symbol: "SLC22A1", reference_genome: GRCh38) {
        gene_id
        gnomad_constraint {
          oe_mis
          oe_mis_lower
          oe_mis_upper
          mis_z
        }
        variants(dataset: gnomad_r4) {
          variant_id
          hgvsp
          consequence
          rsids
          joint {
            ac
            an
            populations { id ac an }
          }
        }
      }
    }
    """
    req = urllib.request.Request(
        "https://gnomad.broadinstitute.org/api/",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "oct1-spt-research/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        payload = json.loads(r.read().decode())
    if payload.get("errors") and not payload.get("data"):
        raise RuntimeError(payload["errors"])
    return payload


# Legacy OCT1 star-allele cores (Seitz 2015 / Shu 2007). SLC22A1 is not in PharmVar.
# Haplotypes are NOT single missense — listed separately.
PHARM_MISSENSE = [
    {"hgvs": "R61C", "rsid": "rs12208357", "legacy_allele": "*3", "function": "reduced",
     "note": "design-set position; haplotype-defining missense"},
    {"hgvs": "C88R", "rsid": "rs55918055", "legacy_allele": "*4", "function": "reduced",
     "note": "design-set position; haplotype-defining missense"},
    {"hgvs": "G401S", "rsid": "rs34130495", "legacy_allele": "*5", "function": "reduced",
     "note": "design-set position; haplotype-defining missense"},
    {"hgvs": "G465R", "rsid": "rs34059508", "legacy_allele": "*6", "function": "reduced",
     "note": "design-set position; haplotype-defining missense"},
    {"hgvs": "G220V", "rsid": "rs36103218", "legacy_allele": None, "function": "reduced",
     "note": "Shu 2003/2007; not a star-allele core in all nomenclatures"},
    {"hgvs": "P341L", "rsid": "rs2282143", "legacy_allele": None, "function": "substrate-dependent/often intact",
     "note": "common EAS; Seitz 2015 often no loss across screened substrates"},
    {"hgvs": "R488M", "rsid": "rs35270274", "legacy_allele": None, "function": "reduced/substrate-dependent",
     "note": "Shu 2007"},
    {"hgvs": "S14F", "rsid": "rs34447885", "legacy_allele": None, "function": "reduced/substrate-dependent",
     "note": "Shu 2003"},
    {"hgvs": "R206C", "rsid": "rs4646278", "legacy_allele": None, "function": "reported reduced",
     "note": "Shu 2003/literature missense; rsid may vary by build"},
    {"hgvs": "L160F", "rsid": "rs683369", "legacy_allele": None, "function": "uncertain/often intact",
     "note": "common; not a canonical LoF allele"},
]
PHARM_HAPLOTYPES = [
    {"legacy_allele": "*2", "protein": "M420del", "function": "reduced",
     "note": "in-frame deletion, not a missense; AM missense scores do not apply to the deletion itself"},
    {"legacy_allele": "*7", "protein": "C88R + M420del", "function": "reduced",
     "note": "haplotype; do not treat as single missense"},
    {"legacy_allele": "*8", "protein": "G401S + M420del? / R61C+C88R nomenclatures vary", "function": "reduced",
     "note": "nomenclature is literature-legacy; SLC22A1 is not in PharmVar"},
    {"legacy_allele": "*10", "protein": "R61C + M420del", "function": "reduced",
     "note": "haplotype"},
]
DRUGS = [
    {"drug": "metformin", "role": "hepatic uptake / PK and glycemic response (mixed clinical evidence)"},
    {"drug": "tramadol", "role": "OCT1 uptake of O-desmethyltramadol; reduced-function alleles raise plasma exposure / alter PK"},
    {"drug": "morphine", "role": "hepatic uptake of morphine; reduced-function alleles associated with higher plasma morphine"},
    {"drug": "ondansetron", "role": "OCT1 substrate; reduced-function alleles associated with higher exposure / greater antiemetic effect"},
    {"drug": "tropisetron", "role": "OCT1 substrate; similar PGx signal to ondansetron"},
]


def classify_am_score(s, t_path=AM_PATH, t_benign=AM_BENIGN):
    try:
        s = float(s)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(s):
        return None
    if s > t_path:
        return "pathogenic"
    if s < t_benign:
        return "benign"
    return "ambiguous"


def try_gnomad(am_map, t_recal):
    out = {"ok": False}
    try:
        payload = fetch_gnomad()
        gene = (payload.get("data") or {}).get("gene") or {}
        variants = gene.get("variants") or []
        rows = []
        pop_ids = ("nfe", "eas", "sas", "afr", "amr", "fin", "asj", "mid", "remaining")
        pop_alias = {"nfe": "EUR", "eas": "EAS", "sas": "SAS", "afr": "AFR", "amr": "AMR"}
        for v in variants:
            consq = (v.get("consequence") or "").lower()
            hgvs = parse_hgvsp(v.get("hgvsp"))
            if "missense" not in consq and hgvs is None:
                continue
            if "missense" not in consq:
                continue
            joint = v.get("joint") or {}
            ac, an = joint.get("ac"), joint.get("an")
            pops = {p["id"]: p for p in (joint.get("populations") or []) if p.get("id")}
            rec = {
                "variant_id": v.get("variant_id"),
                "hgvs_short": hgvs,
                "consequence": v.get("consequence"),
                "rsids": ",".join(v.get("rsids") or []),
                "ac": ac, "an": an,
                "af": (ac / an) if ac is not None and an else np.nan,
            }
            for pid in pop_ids:
                p = pops.get(pid) or {}
                rec[f"ac_{pid}"] = p.get("ac")
                rec[f"an_{pid}"] = p.get("an")
                rec[f"af_{pid}"] = (p["ac"] / p["an"]) if p.get("ac") is not None and p.get("an") else np.nan
            rec["am"] = float(am_map[hgvs]) if hgvs and hgvs in am_map else np.nan
            rec["am_class_cheng"] = classify_am_score(rec["am"], AM_PATH, AM_BENIGN)
            rec["am_class_recal"] = classify_am_score(rec["am"], t_recal, AM_BENIGN)
            rows.append(rec)
        df = pd.DataFrame(rows)
        out["ok"] = True
        out["constraint"] = gene.get("gnomad_constraint")
        out["n_variants_returned"] = len(variants)
        out["n_missense"] = int(len(df))
        if df.empty:
            out["note"] = "GraphQL returned no missense rows"
            return out, df
        def cum_af(mask, pop):
            col = f"af_{pop}"
            if col not in df.columns:
                return np.nan
            return float(df.loc[mask, col].fillna(0).sum())

        summary = {}
        for label, tname in (("cheng_0.564", "am_class_cheng"), ("recal", "am_class_recal")):
            summary[label] = {}
            for cls in ("pathogenic", "benign", "ambiguous"):
                m = df[tname] == cls
                item = {"n": int(m.sum())}
                for pid, alias in pop_alias.items():
                    item[f"sum_af_{alias}"] = cum_af(m, pid)
                summary[label][cls] = item
        # reclassified: Cheng benign/amb -> recal pathogenic
        rec_mask = (df["am_class_cheng"] != "pathogenic") & (df["am_class_recal"] == "pathogenic")
        summary["newly_pathogenic_vs_cheng"] = {
            "n": int(rec_mask.sum()),
            **{f"sum_af_{alias}": cum_af(rec_mask, pid) for pid, alias in pop_alias.items()},
        }
        lost_mask = (df["am_class_cheng"] == "pathogenic") & (df["am_class_recal"] != "pathogenic")
        summary["no_longer_pathogenic_vs_cheng"] = {
            "n": int(lost_mask.sum()),
            **{f"sum_af_{alias}": cum_af(lost_mask, pid) for pid, alias in pop_alias.items()},
        }
        out["summary"] = summary
        out["n_with_am"] = int(df["am"].notna().sum())
        return out, df
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out, pd.DataFrame()


def protein_gym_recall():
    feat = MET_HDD / "proteingym" / "features" / "primary_variants.tsv"
    assays = MET_HDD / "proteingym" / "membrane_assays.tsv"
    if not feat.exists():
        return {"ok": False, "error": "missing primary_variants.tsv"}
    df = pd.read_csv(feat, sep="\t")
    meta = pd.read_csv(assays, sep="\t") if assays.exists() else pd.DataFrame()
    df = df[df["am_pathogenicity"].notna()].copy()
    if df.empty:
        return {"ok": False, "error": "no AM-annotated ProteinGym variants"}
    df["am_path"] = df["am_pathogenicity"] > AM_PATH
    # ProteinGym bin 0 = lower function (deleterious) in their binarization
    df["loss"] = df["DMS_score_bin"].astype(str).isin(["0", "0.0"])
    rows = []
    for dms_id, sub in df.groupby("DMS_id"):
        ce = sub[sub["class"].isin(["CORE", "EXPOSED"])]
        loss = ce[ce["loss"]]
        if len(loss) < 30:
            continue
        def rec(lab):
            s = loss[loss["class"] == lab]
            n = len(s)
            k = int(s["am_path"].sum())
            return n, k, k / n if n else np.nan
        nc, kc, rc = rec("CORE")
        ne, ke, re_ = rec("EXPOSED")
        if nc < 10 or ne < 10:
            continue
        uid = str(sub["UniProt_ID"].iloc[0])
        meta_row = meta[meta["DMS_id"] == dms_id]
        molecule = str(meta_row["molecule_name"].iloc[0]) if len(meta_row) else uid
        taxon = str(meta_row["taxon"].iloc[0]) if len(meta_row) else ""
        sel = str(meta_row["coarse_selection_type"].iloc[0]) if len(meta_row) else ""
        ntm = int(meta_row["n_tm"].iloc[0]) if len(meta_row) and pd.notna(meta_row["n_tm"].iloc[0]) else None
        rows.append({
            "DMS_id": dms_id, "UniProt_ID": uid, "molecule": molecule,
            "taxon": taxon, "selection": sel, "n_tm": ntm,
            "is_s22a1": uid.startswith("S22A1"),
            "n_loss_core": nc, "recall_core": rc,
            "n_loss_exposed": ne, "recall_exposed": re_,
            "delta_recall": (rc - re_) if np.isfinite(rc) and np.isfinite(re_) else np.nan,
            "or": fisher_or_table(kc, nc - kc, ke, ne - ke),
        })
    slc_like = [r for r in rows if ("SLC" in r["UniProt_ID"] or "S22" in r["UniProt_ID"]
                                    or "SC6" in r["UniProt_ID"] or "transporter" in r["molecule"].lower()
                                    or "Oct1" in r["molecule"] or "OCT" in r["molecule"])]
    return {
        "ok": True,
        "n_assays_with_gap": len(rows),
        "assays": rows,
        "slc_or_transporter_like": slc_like,
        "slc22_family": [r for r in rows if "S22" in r["UniProt_ID"]],
        "note": "AM pathogenic >0.564; loss = ProteinGym DMS_score_bin==0. SPT classes from AF2-based proteingym SPT, not OCT1 10/30 lock re-derived. Not a paper result.",
    }


def confusion_af2_8sc1(cmp_df):
    labs = ["CORE", "EXPOSED", "GREY"]
    mat = []
    for a in labs:
        row = []
        for b in labs:
            row.append(int(((cmp_df["af2_class"] == a) & (cmp_df["exp_class"] == b)).sum()))
        mat.append(row)
    n = int(len(cmp_df))
    agree = int((cmp_df["af2_class"] == cmp_df["exp_class"]).sum())
    return {
        "labels": labs,
        "counts": mat,
        "n": n,
        "n_agree": agree,
        "frac_agree": agree / n if n else np.nan,
        "n_disagree": n - agree,
        "frac_disagree": (n - agree) / n if n else np.nan,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng_note = {"n_boot": N_BOOT, "seed": SEED}

    # --- Section 6 lock (written into freeze before A1 numbers are interpreted) ---
    section6 = {
        "locked_before_A1": True,
        "criterion": (
            "If the conservation-adjusted EXPOSED odds-ratio 95% CI excludes 1, "
            "the claim that the AM recall gap is not solely conservation stands. "
            "If the CI includes 1, withdraw or rewrite: AM misses LoF at low-conservation "
            "sites; in transporters those sites are systematically solvent-exposed; "
            "SPT is a practical proxy for conservation. Then target a Brief/computational journal."
        ),
        "primary_conservation": "Jensen–Shannon divergence vs Robinson–Robinson background (Capra & Singh 2007), from ColabFold a3m",
        "sensitivity_conservation": "Shannon conservation 1-H/log2(20)",
        "rate4site": "not installed; not computed",
        "independent_msa": "jackhmmer/UniRef90 and UniClust30 not available locally; reused ColabFold SLC22A1_WT.a3m (n≈10k). Caveat: same MSA family as AM may over-control.",
        "adjusted_or_primary": "residue-clustered bootstrap of logistic EXPOSED coefficient (GFP-loss CORE+EXPOSED; outcome AM-pathogenic; covariates JS conservation + Grantham); GEE reported in parallel",
    }

    val = pd.read_csv(MET_SPT / "wp3_validation_missense.tsv", sep="\t")
    val["pos"] = val["pos"].astype(int)
    val["dms_loss"] = val["GFP_score"] <= GFP_CUT
    val["grantham"] = [grantham(w, m) for w, m in zip(val["wt_aa"], val["mut_aa"])]
    val["am_pathogenic"] = val["am_class"] == "pathogenic"
    assert not val["pos"].isin(DESIGN_POS).any(), "design positions leaked into validation table"

    af2 = pd.read_csv(MET_SPT / "oct1_af2_rank1_spt.tsv", sep="\t")
    sc1 = pd.read_csv(MET_SPT / "oct1_8sc1_spt.tsv", sep="\t")
    cmp = pd.read_csv(MET_SPT / "oct1_af2_vs_8sc1.tsv", sep="\t")
    ddg = pd.read_csv(MET_SPT / "wp3_p3_thermompnn_residue_median.tsv", sep="\t")
    tpt = pd.read_csv(MET_SPT / "tpt" / "oct1_tpt_residues.tsv", sep="\t")

    af2_map = dict(zip(af2["pos"].astype(int), af2["class"]))
    sc1_map = dict(zip(sc1["pos"].astype(int), sc1["class"]))
    agree_pos = set(cmp.loc[cmp["af2_class"] == cmp["exp_class"], "pos"].astype(int))
    cluster_map = dict(zip(tpt["pos"].astype(int), tpt["cluster"]))

    val["class_af2"] = val["pos"].map(af2_map)
    val["class_8sc1"] = val["pos"].map(sc1_map)
    val["class_agree"] = np.where(val["pos"].isin(agree_pos), val["class_af2"], np.nan)
    # sanity: AF2 class column in freeze table
    assert (val["class"] == val["class_af2"]).all()
    val["cluster"] = val["pos"].map(cluster_map)

    ddg["class_af2"] = ddg["pos"].map(af2_map)
    ddg["class_8sc1"] = ddg["pos"].map(sc1_map)
    ddg["class_agree"] = np.where(ddg["pos"].isin(agree_pos), ddg["class_af2"], np.nan)

    a3m = MET_STRUCT / "oct1_variants_20260811_204134" / "SLC22A1_WT.a3m"
    cons = conservation_from_a3m(a3m)
    cons.to_csv(OUT / "oct1_conservation_a3m.tsv", sep="\t", index=False)
    val = val.merge(cons[["pos", "cons_js", "cons_shannon", "msa_n_obs", "msa_occupancy"]], on="pos", how="left")

    # residue-level conservation by AF2 class (all residues, not only GFP-loss)
    res_cons = cons.merge(af2[["pos", "class"]], on="pos", how="left")
    desc_all = {}
    for col in ("cons_js", "cons_shannon"):
        desc_all[col] = {}
        for lab in ("CORE", "EXPOSED", "GREY"):
            s = res_cons.loc[res_cons["class"] == lab, col].dropna()
            desc_all[col][lab] = {"n": int(len(s)), "median": float(s.median()) if len(s) else np.nan}
        c = res_cons.loc[res_cons["class"] == "CORE", col].dropna()
        e = res_cons.loc[res_cons["class"] == "EXPOSED", col].dropna()
        _, p = stats.mannwhitneyu(c, e, alternative="greater")
        desc_all[col]["mwu_core_greater_p"] = float(p)

    labels = {
        "AF2": "class_af2",
        "8SC1": "class_8sc1",
        "agree": "class_agree",
    }

    three_col = {}
    a1_by_label = {}
    for name, col in labels.items():
        three_col[name] = {
            "P1": p1_stats(val, col),
            "P3": p3_stats(ddg, col),
            "recall_gap": recall_gap(val, col),
            "P4": p4_clustered(val, col),
            "auroc": {lab: auroc_clustered(val, lab, col) for lab in ("CORE", "EXPOSED", "GREY")},
        }
        a1_by_label[name] = {
            "js": a1_model(val, col, "cons_js", n_boot=N_BOOT),
            "shannon": a1_model(val, col, "cons_shannon", n_boot=min(2000, N_BOOT)),
        }
        print(f"[{name}] P1 med {three_col[name]['P1'].get('median_core')} vs "
              f"{three_col[name]['P1'].get('median_exposed')}  "
              f"recall {three_col[name]['recall_gap']['recall_core']:.3f} vs "
              f"{three_col[name]['recall_gap']['recall_exposed']:.3f}  "
              f"A1 JS clustered OR {a1_by_label[name]['js']['logit_or_clustered']['median']:.3f} "
              f"CI {a1_by_label[name]['js']['logit_or_clustered']['ci95']}")

    # A1 primary = AF2 labels, JS conservation (preregistered structure labels)
    a1_primary = a1_by_label["AF2"]["js"]
    adj_ci = a1_primary["logit_or_clustered"]["ci95"]
    gee_ci = (a1_primary["gee"].get("ci95") if a1_primary["gee"].get("ok") else None)
    # verdict uses clustered-bootstrap CI of adjusted logit OR; GEE as corroboration
    ci_lo, ci_hi = adj_ci
    # EXPOSED dummy OR is <1 (lower AM-pathogenic odds). Claim stands iff CI excludes 1.
    claim_stands = bool(np.isfinite(ci_lo) and np.isfinite(ci_hi) and (ci_hi < 1 or ci_lo > 1))
    gee_stands = None
    if gee_ci:
        gee_stands = bool(np.isfinite(gee_ci[0]) and np.isfinite(gee_ci[1]) and (gee_ci[1] < 1 or gee_ci[0] > 1))

    section6_verdict = {
        "claim_stands_clustered_logit": claim_stands,
        "coding": (
            "Logistic outcome is AM-pathogenic; EXPOSED dummy OR <1 means GFP-loss at EXPOSED "
            "is still less often AM-pathogenic after conservation+Grantham. The 5.05-style "
            "CORE:EXPOSED odds ratio is the reciprocal. Section 6 tests whether the CI excludes 1."
        ),
        "adjusted_EXPOSED_or_median": a1_primary["logit_or_clustered"]["median"],
        "adjusted_EXPOSED_or_ci95": adj_ci,
        "adjusted_CORE_vs_EXPOSED_or_median": a1_primary["logit_or_clustered"].get("or_core_vs_exposed_median"),
        "adjusted_CORE_vs_EXPOSED_or_ci95": a1_primary["logit_or_clustered"].get("or_core_vs_exposed_ci95"),
        "unadjusted_recall_or": a1_primary["unadjusted_or"],
        "gee_stands": gee_stands,
        "gee": a1_primary["gee"],
        "path": "proceed" if claim_stands else "rewrite_brief_computational",
        "applied_rule": section6["criterion"],
    }

    conf = confusion_af2_8sc1(cmp)

    # B1 cutoffs on full data (leaky — report only as descriptive; LOPO is inferential)
    y_all = val["dms_loss"].astype(int).to_numpy()
    s_all = val["am_pathogenicity"].to_numpy(dtype=float)
    t_youden, j_all = youden_threshold(s_all, y_all)
    core_loss = val[(val["class"] == "CORE") & val["dms_loss"]]
    core_rec = float((core_loss["am_class"] == "pathogenic").mean())
    exp_loss = val[(val["class"] == "EXPOSED") & val["dms_loss"]]
    t_match = recall_match_threshold(
        exp_loss["am_pathogenicity"].to_numpy(),
        np.ones(len(exp_loss), dtype=int),
        core_rec,
    )
    t_prec, prec_sens = precision_threshold(s_all, y_all, 0.90)
    # class-specific Youden (descriptive)
    class_youden = {}
    for lab in ("CORE", "EXPOSED", "GREY"):
        sub = val[val["class"] == lab]
        t, j = youden_threshold(sub["am_pathogenicity"].to_numpy(), sub["dms_loss"].astype(int).to_numpy())
        class_youden[lab] = {"t": t, "J": j}

    b1_full = {
        "note": "Full-data cutoffs are descriptive only. Helix LOPO is the reported inference.",
        "core_pathogenic_recall_cheng": core_rec,
        "t_youden": t_youden, "youden_J": j_all,
        "t_exposed_to_match_core_recall": t_match,
        "t_precision90": t_prec, "precision90_sens_at_t": prec_sens,
        "class_specific_youden": class_youden,
        "performance_cheng": cutoff_performance(val, AM_PATH, "class"),
        "performance_youden": cutoff_performance(val, t_youden, "class"),
        "performance_match": cutoff_performance(val, t_match, "class"),
        "performance_prec90": cutoff_performance(val, t_prec, "class") if np.isfinite(t_prec) else {},
    }
    print("B1 full-data Youden", t_youden, "match", t_match, "prec90", t_prec)
    b1_lopo = helix_lopo_cutoffs(val, val["cluster"])
    print("B1 LOPO Youden median", b1_lopo["cutoff_youden"])

    # conservation-adjusted class Youden on AM residual (descriptive)
    ok = val.dropna(subset=["cons_js", "am_pathogenicity"])
    lr = np.polyfit(ok["cons_js"].to_numpy(), ok["am_pathogenicity"].to_numpy(), 1)
    val = val.copy()
    val["am_resid_cons"] = val["am_pathogenicity"] - (lr[0] * val["cons_js"] + lr[1])
    resid_youden = {}
    for lab in ("CORE", "EXPOSED", "GREY"):
        sub = val[val["class"] == lab].dropna(subset=["am_resid_cons"])
        t, j = youden_threshold(sub["am_resid_cons"].to_numpy(), sub["dms_loss"].astype(int).to_numpy())
        resid_youden[lab] = {"t": t, "J": j, "n": int(len(sub))}
    b1_full["am_residualized_on_js_youden_by_class"] = resid_youden
    b1_full["am_vs_js_linreg_slope"] = float(lr[0])

    # B2
    am_tbl = pd.read_csv(MET_AM / "by_target" / "SLC22A1_O15245.tsv", sep="\t")
    am_map = dict(zip(am_tbl["protein_variant"].astype(str), am_tbl["am_pathogenicity"].astype(float)))
    t_recal = b1_lopo["cutoff_youden"].get("median", t_youden)
    gnomad_meta, gnomad_df = try_gnomad(am_map, t_recal)
    if not gnomad_df.empty:
        gnomad_df.to_csv(OUT / "gnomad_slc22a1_missense.tsv", sep="\t", index=False)

    pharm_rows = []
    for item in PHARM_MISSENSE:
        s = am_map.get(item["hgvs"], np.nan)
        pharm_rows.append({
            **item,
            "am": s,
            "am_class_cheng": classify_am_score(s) if np.isfinite(s) else None,
            "am_class_recal_lopo_youden": classify_am_score(s, t_recal) if np.isfinite(s) else None,
            "design_pos": int(re.search(r"\d+", item["hgvs"]).group()) in DESIGN_POS,
            "in_validation_table": item["hgvs"] in set(val["hgvs_short"]),
        })
    b2 = {
        "pharmvar": "SLC22A1 is not a PharmVar gene; star alleles are literature-legacy haplotypes (Seitz 2015 / Shu 2007). Do not treat haplotypes as single missense.",
        "pharmgkb": "ClinPGx/PharmGKB VIP (former Tier 2) lists SLC22A1; reduced-function evidence is allele- and substrate-specific.",
        "missense_alleles": pharm_rows,
        "haplotypes": PHARM_HAPLOTYPES,
        "drugs": DRUGS,
        "gnomad": gnomad_meta,
        "recal_threshold_used": t_recal,
    }

    c_out = protein_gym_recall()
    if c_out.get("assays"):
        pd.DataFrame(c_out["assays"]).to_csv(OUT / "proteingym_recall_gap.tsv", sep="\t", index=False)

    # A4 failed-bundle (locked numbers + this run)
    failed_bundle = {
        "P2": "FAIL: AM ranking not better in CORE (Δ|ρ| CI includes 0; Holm p=0.103). Do not rewrite as a pass.",
        "P4": "preregistered P4 did not survive clustered inference (dms_loss n=485; clustered OR CI includes 1).",
        "P5": "held-out AF2 omitted; do not advertise 7/33.",
        "H4.1_H4.3": "uptake secondary hypotheses failed as previously frozen.",
        "I1C": "I1C family failed; I1C.5 is GFP-confounded — not an instead-clause.",
        "literature": "n=34 is overlapping illustrative, not held-out (32/34 substitutions and 34/34 positions overlap DMS). Remove n=3 EXPOSED loss* from abstract.",
    }

    freeze = {
        "title": "Feedback 260901 freeze",
        "n_boot": N_BOOT,
        "seed": SEED,
        "locks": {
            "gfp_cut": GFP_CUT,
            "am_pathogenic": AM_PATH,
            "am_benign": AM_BENIGN,
            "design_pos": sorted(DESIGN_POS),
            "spt_not_retuned": True,
            "p4_endpoint": "dms_loss n from this run; never func_loss 493",
        },
        "section6_prespecified": section6,
        "section6_verdict": section6_verdict,
        "conservation_descriptive_all_residues": desc_all,
        "msa": {
            "path": str(a3m),
            "n_seqs_used": int(cons["msa_n_seqs_used"].iloc[0]),
            "n_pos_with_js": int(cons["cons_js"].notna().sum()),
        },
        "A1": a1_by_label,
        "A2_confusion_AF2_x_8SC1": conf,
        "A2_A3_three_label": three_col,
        "A4_failed_bundle": failed_bundle,
        "B1_full_data_descriptive": b1_full,
        "B1_helix_LOPO": {k: v for k, v in b1_lopo.items() if k != "folds"},
        "B1_helix_LOPO_folds": b1_lopo["folds"],
        "B2": b2,
        "C_proteingym": c_out,
        "D_hygiene_checklist": {
            "remove_code_availability_freeze_memo": True,
            "define_dms_loss_vs_func_loss_in_methods": True,
            "strip_P1P6_jargon_from_main_text": True,
            "osf_prereg_retrospective_with_commit_hash": True,
            "result_statement_title": True,
            "abstract_drugs_and_AM_benign_not_clearance": True,
            "llm_disclosure_narrow": True,
            "drop_80pct_passfail_language": True,
            "literature_not_heldout": True,
            "af2_remains_preregistered_primary_8sc1_robustness_unless_promoted_explicitly": True,
        },
    }
    dump(freeze, OUT / "ms1_feedback2_freeze.json")
    dump(freeze, MS1 / "ms1_feedback2_freeze.json")

    # compact tables
    rows = []
    for name in ("AF2", "8SC1", "agree"):
        b = three_col[name]
        a1 = a1_by_label[name]["js"]
        rows.append({
            "label": name,
            "P1_median_CORE": b["P1"]["median_core"],
            "P1_median_EXPOSED": b["P1"]["median_exposed"],
            "P1_p": b["P1"]["p"],
            "P3_median_CORE": b["P3"]["median_core"],
            "P3_median_EXPOSED": b["P3"]["median_exposed"],
            "P3_p": b["P3"]["p"],
            "recall_CORE": b["recall_gap"]["recall_core"],
            "recall_EXPOSED": b["recall_gap"]["recall_exposed"],
            "recall_OR_variant": b["recall_gap"]["or_variant"]["or"],
            "recall_OR_clustered_median": b["recall_gap"]["or_clustered"]["median"],
            "recall_OR_clustered_lo": b["recall_gap"]["or_clustered"]["ci95"][0],
            "recall_OR_clustered_hi": b["recall_gap"]["or_clustered"]["ci95"][1],
            "A1_unadj_OR": a1["unadjusted_or"]["or"],
            "A1_adj_EXPOSED_OR_median": a1["logit_or_clustered"]["median"],
            "A1_adj_EXPOSED_OR_lo": a1["logit_or_clustered"]["ci95"][0],
            "A1_adj_EXPOSED_OR_hi": a1["logit_or_clustered"]["ci95"][1],
            "A1_adj_CORE_vs_EXPOSED_OR_median": a1["logit_or_clustered"].get("or_core_vs_exposed_median"),
            "A1_adj_CORE_vs_EXPOSED_OR_lo": (a1["logit_or_clustered"].get("or_core_vs_exposed_ci95") or [None, None])[0],
            "A1_adj_CORE_vs_EXPOSED_OR_hi": (a1["logit_or_clustered"].get("or_core_vs_exposed_ci95") or [None, None])[1],
            "P4_OR_variant": b["P4"]["or_variant"]["or"],
            "P4_OR_clustered_median": b["P4"]["or_clustered"]["median"],
            "P4_OR_clustered_lo": b["P4"]["or_clustered"]["ci95"][0],
            "P4_OR_clustered_hi": b["P4"]["or_clustered"]["ci95"][1],
            "P4_n_hit": b["P4"]["n_hit"],
        })
    pd.DataFrame(rows).to_csv(OUT / "three_label_comparison.tsv", sep="\t", index=False)
    pd.DataFrame(a1_primary["quintiles"]).to_csv(OUT / "a1_quintile_recall_AF2_js.tsv", sep="\t", index=False)

    print("\n=== SECTION 6 VERDICT ===")
    print("claim_stands", claim_stands, "adj OR", a1_primary["logit_or_clustered"])
    print("outputs ->", OUT)


if __name__ == "__main__":
    main()
