#!/usr/bin/env python3
"""I1 scoring: surface + uptake/surface types vs locked R5 predictions.

    source met_env.sh && $MET_PY met_i1_score.py
    $MET_PY met_i1_score.py --results /path/to/i1_results.tsv

Requires filled numeric `surface` and `uptake` columns. Background-subtract
EMPTY (if present) before z-scoring vs WT. Does not retune SPT or rename clones.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MET_HDD = Path(os.environ.get("MET_HDD", str(Path(__file__).resolve().parent / "data")))
OUT = MET_HDD / "challenge" / "i_instead"
Z = 2.0
PASS_N = {4: 3}  # n_class -> min n matching type

PRED = {
    "abundance_loss_resid_ok": "Stab",
    "exposed_am_benign_gfp_loss": "Stab",
    "dms_resid_loss_gfp_ok": "Trans",
    "near_wt_control": "WT",
}


def _mean(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").dropna()
    return float(v.mean()) if len(v) else float("nan")


def _sd(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").dropna()
    if len(v) < 2:
        return float("nan")
    return float(v.std(ddof=1))


def call_type(s_low: bool, u_low: bool) -> str:
    if s_low and u_low:
        return "Mixed"
    if s_low:
        return "Stab"
    if u_low:
        return "Trans"
    return "WT"


def match_pred(pred: str, got: str) -> bool:
    if pred == "Stab":
        return got == "Stab"
    if pred == "Trans":
        return got == "Trans"
    if pred == "WT":
        return got == "WT"
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=OUT / "i1_results.tsv")
    args = ap.parse_args()

    key_p = OUT / "i1_clone_key.tsv"
    if not key_p.exists():
        raise SystemExit("run met_i1_prep.py first")
    key = pd.read_csv(key_p, sep="\t")

    if not args.results.exists():
        print(f"awaiting wet-lab: no {args.results}")
        print("copy i1_results_template.tsv -> i1_results.tsv and fill surface/uptake")
        return

    raw = pd.read_csv(args.results, sep="\t")
    need = {"clone_id", "replicate", "surface", "uptake"}
    if not need <= set(raw.columns):
        raise SystemExit(f"results need columns {sorted(need)}")
    raw["surface"] = pd.to_numeric(raw["surface"], errors="coerce")
    raw["uptake"] = pd.to_numeric(raw["uptake"], errors="coerce")
    scored = raw.dropna(subset=["surface", "uptake"])
    n_empty_cells = int(raw["surface"].isna().sum() + raw["uptake"].isna().sum())
    if scored["clone_id"].nunique() < 5:
        print(f"awaiting wet-lab: {args.results} has too few numeric rows")
        (OUT / "i1_verdict.json").write_text(json.dumps({
            "status": "awaiting_wetlab",
            "pass": None,
            "n_numeric_rows": int(len(scored)),
            "n_empty_cells": n_empty_cells,
        }, indent=2) + "\n")
        return

    bg_s = _mean(scored.loc[scored["clone_id"] == "EMPTY", "surface"])
    bg_u = _mean(scored.loc[scored["clone_id"] == "EMPTY", "uptake"])
    if not np.isfinite(bg_s):
        bg_s = 0.0
    if not np.isfinite(bg_u):
        bg_u = 0.0
    scored = scored.copy()
    scored["S"] = scored["surface"] - bg_s
    scored["raw_u"] = scored["uptake"] - bg_u
    # per-well U = uptake / surface after background; guard non-positive S
    scored["U"] = np.where(scored["S"] > 0, scored["raw_u"] / scored["S"], np.nan)

    wt = scored[scored["clone_id"] == "WT"]
    if len(wt) < 4:
        raise SystemExit("need ≥4 numeric WT replicates for ±2 SD")
    wt_s_m, wt_s_sd = _mean(wt["S"]), _sd(wt["S"])
    wt_u = wt["U"].dropna()
    if len(wt_u) < 4:
        raise SystemExit("need ≥4 WT wells with S>0 to define U")
    wt_u_m, wt_u_sd = float(wt_u.mean()), float(wt_u.std(ddof=1))
    s_cut = wt_s_m - Z * wt_s_sd
    u_cut = wt_u_m - Z * wt_u_sd

    mut = scored[scored["clone_id"].astype(str).str.startswith("I1-")]
    agg = mut.groupby("clone_id").agg(S=("S", "mean"), U=("U", "mean"), n=("S", "size")).reset_index()
    agg = agg.merge(key, on="clone_id", how="left")
    agg["S_low"] = agg["S"] < s_cut
    agg["U_low"] = agg["U"] < u_cut
    agg["observed_type"] = [call_type(s, u) for s, u in zip(agg["S_low"], agg["U_low"])]
    agg["type_match"] = [
        match_pred(p, g) if p in PRED.values() else None
        for p, g in zip(agg["predicted_type"], agg["observed_type"])
    ]
    agg.to_csv(OUT / "i1_scored_clones.tsv", sep="\t", index=False)

    gates = {}
    class_map = {
        "I1.1": ("dms_resid_loss_gfp_ok", "Trans"),
        "I1.2": ("abundance_loss_resid_ok", "Stab"),
        "I1.3": ("exposed_am_benign_gfp_loss", "Stab"),
        "I1.4": ("near_wt_control", "WT"),
    }
    for gid, (cls, _pred) in class_map.items():
        sub = agg[agg["panel_class"] == cls]
        n = int(len(sub))
        n_hit = int(sub["type_match"].fillna(False).sum())
        need_n = int(np.ceil(0.75 * n)) if n not in PASS_N else PASS_N[n]
        if n == 4:
            need_n = 3
        gates[gid] = {
            "class": cls,
            "n": n,
            "n_match": n_hit,
            "need": need_n,
            "pass": bool(n >= 1 and n_hit >= need_n),
            "hgvs": sub["hgvs_short"].tolist(),
            "observed": sub["observed_type"].tolist(),
        }

    i1_4 = gates["I1.4"]["pass"]
    i1_2 = gates["I1.2"]["pass"]
    i1_3 = gates["I1.3"]["pass"]
    i1_1 = gates["I1.1"]["pass"]

    if not i1_4:
        go = "stop_assay_invalid"
    elif i1_2 and i1_3:
        go = "I2_allowed_stab" + ("_and_trans" if i1_1 else "_drop_trans")
    else:
        go = "stop_instead_class_fail"

    overall = bool(i1_4 and i1_2 and i1_3)
    verdict = {
        "status": "scored",
        "pass_stab_instead": overall,
        "pass_trans_instead": bool(i1_1 and i1_4),
        "go": go,
        "gates": gates,
        "wt": {
            "n": int(len(wt)),
            "S_mean": wt_s_m,
            "S_sd": wt_s_sd,
            "U_mean": wt_u_m,
            "U_sd": wt_u_sd,
            "S_cut": s_cut,
            "U_cut": u_cut,
        },
        "background": {"surface": bg_s, "uptake": bg_u},
        "i2_names": "not_picked" if "I2_allowed" not in go else "allowed_run_picker_after_this_go",
        "scored_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "do_not_drop_clones": True,
    }
    (OUT / "i1_verdict.json").write_text(json.dumps(verdict, indent=2, default=str) + "\n")

    st = json.loads((OUT / "STATUS.json").read_text())
    st["status"] = go
    st["wetlab"] = "scored"
    st["updated_at"] = verdict["scored_at"]
    (OUT / "STATUS.json").write_text(json.dumps(st, indent=2) + "\n")

    print(json.dumps({k: gates[k]["pass"] for k in gates}, indent=2))
    print("go:", go)
    print(f"wrote {OUT / 'i1_verdict.json'}")


if __name__ == "__main__":
    main()
