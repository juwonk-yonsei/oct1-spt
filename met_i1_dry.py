#!/usr/bin/env python3
"""I1C computational instead (met_prereg_instead_dry.md).

    source met_env.sh && $MET_PY met_i1_dry.py

Does not score R5 names on the labels that selected them.
Does not retune SPT. Does not start C8.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import roc_auc_score

MET_HDD = Path(os.environ.get("MET_HDD", str(Path(__file__).resolve().parent / "data")))
MET_DMS = Path(os.environ.get("MET_DMS", str(MET_HDD / "dms")))
PRED = MET_HDD / "challenge" / "c3_ensemble" / "oct1_ens_lopo.tsv"
R5 = MET_HDD / "challenge" / "r_residual" / "r5_verdict.json"
R5_PANEL = MET_HDD / "challenge" / "r_residual" / "r5_experiment_panel.tsv"
OUT = MET_HDD / "challenge" / "i_instead"

P1_LOSS = -0.814
AM_BENIGN = 0.34
RESID_Q10 = -0.2882529540825411
ABS_Q25 = 0.06608502985979253
NEAR_LO = -0.4136934113086963
NEAR_HI = 0.43836465250467216
GFP_SE_MED = 0.2893084872802353
SM_SE_MED = 0.1531337654802551
HGVS_RE = re.compile(r"p\.\(([A-Z])(\d+)([A-Z])\)")


def holm(ps: dict[str, float]) -> dict[str, float]:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    acc = 0.0
    for i, (k, p) in enumerate(items, 1):
        adj = min(1.0, (m - i + 1) * p)
        acc = max(acc, adj)
        out[k] = acc
    return out


def fisher_or(a_yes: int, a_no: int, b_yes: int, b_no: int) -> dict:
    tab = [[a_yes, a_no], [b_yes, b_no]]
    or_, p = fisher_exact(tab, alternative="two-sided")
    return {"table": tab, "or": float(or_), "p": float(p)}


def parse_hgvs(s: str):
    m = HGVS_RE.search(str(s))
    return m.group(1) + m.group(2) + m.group(3) if m else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PRED, sep="\t")
    dms = pd.read_csv(MET_DMS / "oct1_combined_scores.csv")
    dms["hgvs_short"] = dms["hgvs"].map(parse_hgvs)
    se = dms[["hgvs_short", "GFP_SE", "SM73_0_SE"]].dropna(subset=["hgvs_short"])
    df = df.merge(se, on="hgvs_short", how="left")

    r5 = json.loads(R5.read_text())
    pan = pd.read_csv(R5_PANEL, sep="\t")
    r5_pos = set(int(x) for x in pan["pos"].tolist())

    ok = (
        df["train_ok"].astype(bool)
        & df["GFP_score"].notna()
        & df["SM73_resid_fold"].notna()
        & df["ddg"].notna()
        & df["am_pathogenicity"].notna()
        & (df["design_pos"].fillna(0) == 0)
        & ~df["pos"].isin(r5_pos)
    )
    se_ok = (df["GFP_SE"] <= GFP_SE_MED) & (df["SM73_0_SE"] <= SM_SE_MED)
    base = df.loc[ok & se_ok].copy()
    gfp = base["GFP_score"]
    resid = base["SM73_resid_fold"]

    stab = (gfp <= P1_LOSS) & (resid.abs() <= ABS_Q25)
    trans = (gfp >= P1_LOSS) & (resid <= RESID_Q10)
    wt = gfp.between(NEAR_LO, NEAR_HI) & (resid.abs() <= 0.05)
    # disjoint
    trans = trans & ~stab
    wt = wt & ~stab & ~trans

    def pack(mask: pd.Series, name: str) -> pd.DataFrame:
        sub = base.loc[mask].copy()
        sub["gold_type"] = name
        return sub

    typed = pd.concat(
        [pack(stab, "Stab*"), pack(trans, "Trans*"), pack(wt, "WT*")],
        ignore_index=True,
    )
    typed.to_csv(OUT / "i1c_typed_variants.tsv", sep="\t", index=False)

    s = typed[typed["gold_type"] == "Stab*"]
    t = typed[typed["gold_type"] == "Trans*"]
    w = typed[typed["gold_type"] == "WT*"]

    i1c1 = fisher_or(
        int((s["spt_class"] == "CORE").sum()),
        int((s["spt_class"] != "CORE").sum()),
        int((t["spt_class"] == "CORE").sum()),
        int((t["spt_class"] != "CORE").sum()),
    )
    i1c1["pass"] = bool(i1c1["or"] > 1 and i1c1["p"] < 0.05)

    ds = s["ddg"].to_numpy(float)
    dt = t["ddg"].to_numpy(float)
    mw = mannwhitneyu(ds, dt, alternative="two-sided")
    i1c2 = {
        "n_stab": int(len(ds)),
        "n_trans": int(len(dt)),
        "median_stab": float(np.median(ds)),
        "median_trans": float(np.median(dt)),
        "p": float(mw.pvalue),
        "pass": bool(np.median(ds) > np.median(dt) and mw.pvalue < 0.05),
    }

    i1c5 = fisher_or(
        int((t["am_pathogenicity"] < AM_BENIGN).sum()),
        int((t["am_pathogenicity"] >= AM_BENIGN).sum()),
        int((s["am_pathogenicity"] < AM_BENIGN).sum()),
        int((s["am_pathogenicity"] >= AM_BENIGN).sum()),
    )
    i1c5["pass"] = bool(i1c5["or"] > 1 and i1c5["p"] < 0.05)

    i1c4 = fisher_or(
        int((s["spt_class"] == "CORE").sum()),
        int((s["spt_class"] != "CORE").sum()),
        int((w["spt_class"] == "CORE").sum()),
        int((w["spt_class"] != "CORE").sum()),
    )
    i1c4["pass"] = bool(i1c4["or"] > 1 and i1c4["p"] < 0.05)

    y = np.concatenate([np.ones(len(s)), np.zeros(len(t))])
    i1c3 = {"auroc_ddg": None, "auroc_am": None}
    if len(s) >= 10 and len(t) >= 10:
        i1c3["auroc_ddg"] = float(roc_auc_score(y, np.concatenate([ds, dt])))
        i1c3["auroc_am"] = float(
            roc_auc_score(y, np.concatenate([s["am_pathogenicity"], t["am_pathogenicity"]]))
        )

    holm_p = holm({"I1C.1": i1c1["p"], "I1C.2": i1c2["p"], "I1C.5": i1c5["p"]})
    i1c1["p_holm"] = holm_p["I1C.1"]
    i1c2["p_holm"] = holm_p["I1C.2"]
    i1c5["p_holm"] = holm_p["I1C.5"]
    i1c1["pass_holm"] = bool(i1c1["or"] > 1 and holm_p["I1C.1"] < 0.05)
    i1c2["pass_holm"] = bool(np.median(ds) > np.median(dt) and holm_p["I1C.2"] < 0.05)
    i1c5["pass_holm"] = bool(i1c5["or"] > 1 and holm_p["I1C.5"] < 0.05)

    primary = bool(i1c1["pass_holm"] and i1c2["pass_holm"] and i1c5["pass_holm"])
    counts = typed.groupby(["gold_type", "spt_class"]).size().unstack(fill_value=0)

    verdict = {
        "prereg": "met_prereg_instead_dry.md",
        "status": "pass" if primary else "fail",
        "pass_primary": primary,
        "n": {
            "universe": int(len(base)),
            "Stab*": int(len(s)),
            "Trans*": int(len(t)),
            "WT*": int(len(w)),
            "r5_pos_excluded": sorted(r5_pos),
        },
        "cuts": {
            "P1_LOSS": P1_LOSS,
            "AM_BENIGN": AM_BENIGN,
            "resid_q10": RESID_Q10,
            "abs_q25": ABS_Q25,
        },
        "I1C.1": i1c1,
        "I1C.2": i1c2,
        "I1C.5": i1c5,
        "I1C.4": i1c4,
        "I1C.3": i1c3,
        "class_by_type": counts.to_dict(),
        "claim_if_pass": (
            "Don't use AM to type or as a stop-rule. Instead (DMS proxies): "
            "CORE is Stab* (abundance/stability, high ΔΔG); Trans* is GFP-ok residual-loss that AM misses. "
            "Not surface-normalized; not a clinical guideline."
        ),
        "scored_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    (OUT / "i1c_verdict.json").write_text(json.dumps(verdict, indent=2, default=str) + "\n")

    st_p = OUT / "STATUS.json"
    st = json.loads(st_p.read_text()) if st_p.exists() else {}
    st.update({
        "track": "instead",
        "current": "I1C",
        "status": verdict["status"],
        "wetlab": "optional_I1W_not_required",
        "i1c_pass": primary,
        "updated_at": verdict["scored_at"],
        "i2_names": "not_picked",
    })
    st_p.write_text(json.dumps(st, indent=2) + "\n")

    print("n Stab*/Trans*/WT*", len(s), len(t), len(w))
    print("I1C.1 CORE Stab vs Trans OR", round(i1c1["or"], 3), "p", i1c1["p"], "holm", i1c1["p_holm"], "pass", i1c1["pass_holm"])
    print("I1C.2 ddg med", round(i1c2["median_stab"], 3), "vs", round(i1c2["median_trans"], 3), "p", i1c2["p"], "pass", i1c2["pass_holm"])
    print("I1C.5 AM-benign Trans vs Stab OR", round(i1c5["or"], 3), "p", i1c5["p"], "pass", i1c5["pass_holm"])
    print("I1C.4 CORE Stab vs WT OR", round(i1c4["or"], 3), "p", i1c4["p"])
    print("I1C.3 AUROC ddg/AM", i1c3)
    print("PRIMARY", primary, "->", OUT / "i1c_verdict.json")


if __name__ == "__main__":
    main()
