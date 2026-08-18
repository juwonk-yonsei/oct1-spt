#!/usr/bin/env python3
"""WP4: held-out OCT1 literature variants vs SPT class + AlphaMissense.

Pre-registered in met_prereg_grey.md (H4.1–H4.3). Design-set positions excluded.

    $MET_PY met_wp4.py
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
MET_DMS = Path(os.environ.get("MET_DMS", str(MET_HDD / "dms")))
MET_AM = Path(os.environ.get("MET_AM", str(MET_HDD / "alphamissense")))
MET_SPT = Path(os.environ.get("MET_SPT", str(MET_HDD / "spt")))

DESIGN_POS = {61, 88, 401, 420, 465}
HGVS_RE = re.compile(r"p\.\(([A-Z])(\d+)(del|[A-Z])\)", re.I)


def parse_hgvs(s: str):
    m = HGVS_RE.search(str(s))
    if not m:
        return None
    wt, pos, mut = m.group(1).upper(), int(m.group(2)), m.group(3)
    if mut.lower() == "del":
        return wt, pos, "del", f"{wt}{pos}del"
    return wt, pos, mut.upper(), f"{wt}{pos}{mut.upper()}"


def fisher_greater(a_yes, a_no, b_yes, b_no):
    table = np.array([[a_yes, a_no], [b_yes, b_no]], dtype=int)
    if table.min() < 0 or table.sum() == 0:
        return np.nan, np.nan, table
    oddsr, p = stats.fisher_exact(table, alternative="greater")
    return float(oddsr), float(p), table


def main():
    spt = pd.read_csv(MET_SPT / "oct1_af2_rank1_spt.tsv", sep="\t")
    class_of = dict(zip(spt["pos"].astype(int), spt["class"]))
    topo_of = dict(zip(spt["pos"].astype(int), spt["topology"]))
    sasa_of = dict(zip(spt["pos"].astype(int), spt["rel_sasa"]))

    am = pd.read_csv(MET_AM / "by_target/SLC22A1_O15245.tsv", sep="\t")
    am_map = am.set_index("protein_variant")[["am_pathogenicity", "am_class"]].to_dict("index")

    lit = pd.read_csv(MET_DMS / "literature_variants.csv")
    rows = []
    for rec in lit.to_dict("records"):
        parsed = parse_hgvs(rec["hgvs"])
        if parsed is None:
            rec.update({"parse_ok": False, "excluded_design": False})
            rows.append(rec)
            continue
        wt, pos, mut, short = parsed
        rec.update({
            "parse_ok": True,
            "wt_aa": wt, "res_pos": pos, "mut_aa": mut, "hgvs_short": short,
            "excluded_design": pos in DESIGN_POS,
            "is_deletion": mut == "del",
            "uncertain_note": ("?" in str(rec.get("note", "") or "")),
            "spt_class": class_of.get(pos),
            "topology": topo_of.get(pos),
            "rel_sasa": sasa_of.get(pos),
        })
        am_hit = am_map.get(short)
        if am_hit:
            rec["am_pathogenicity"] = am_hit["am_pathogenicity"]
            rec["am_class"] = am_hit["am_class"]
        func = str(rec.get("literature_impact_function") or "").strip().lower()
        if func in ("loss", "partial_loss"):
            rec["func_bundle"] = "loss*"
        elif func == "neutral":
            rec["func_bundle"] = "neutral"
        elif func == "gain":
            rec["func_bundle"] = "gain"
        else:
            rec["func_bundle"] = func or "unknown"
        rows.append(rec)

    df = pd.DataFrame(rows)
    held = df[(df["parse_ok"] == True) & (~df["excluded_design"]) & (~df["is_deletion"])].copy()
    print("=== WP4 held-out literature missense (design 5 excluded) ===")
    print(f"  n parsed held-out missense: {len(held)}")
    print(f"  excluded design: {(df['excluded_design']==True).sum()}")
    print("  func_bundle:", held["func_bundle"].value_counts().to_dict())
    print("  spt_class:", held["spt_class"].value_counts().to_dict())

    def show_subset(label, sub):
        print(f"\n--- {label} n={len(sub)} ---")
        cols = ["hgvs_short", "func_bundle", "spt_class", "rel_sasa", "am_class",
                "am_pathogenicity", "literature_impact_trafficking"]
        with pd.option_context("display.max_rows", 80, "display.width", 140):
            print(sub.sort_values(["func_bundle", "spt_class", "res_pos"])[cols].to_string(index=False))

    show_subset("all held-out missense", held)

    loss = held[held["func_bundle"] == "loss*"]
    neu = held[held["func_bundle"] == "neutral"]
    loss_c = loss[loss["spt_class"] == "CORE"]
    loss_e = loss[loss["spt_class"] == "EXPOSED"]
    loss_g = loss[loss["spt_class"] == "GREY"]

    def patho_frac(sub):
        sub = sub.dropna(subset=["am_class"])
        if len(sub) == 0:
            return np.nan, 0, 0
        n_p = int((sub["am_class"] == "pathogenic").sum())
        return n_p / len(sub), n_p, len(sub)

    f_c, n_pc, n_c = patho_frac(loss_c)
    f_e, n_pe, n_e = patho_frac(loss_e)
    f_g, n_pg, n_g = patho_frac(loss_g)
    print("\n=== H4.1  AM pathogenic fraction among loss* ===")
    print(f"  CORE    {n_pc}/{n_c} = {f_c if n_c else float('nan'):.3f}")
    print(f"  EXPOSED {n_pe}/{n_e} = {f_e if n_e else float('nan'):.3f}")
    print(f"  GREY    {n_pg}/{n_g} = {f_g if n_g else float('nan'):.3f}")
    or41, p41, t41 = fisher_greater(n_pc, n_c - n_pc, n_pe, n_e - n_pe)
    print(f"  Fisher CORE>EXPOSED pathogenic: OR={or41:.3f}  p={p41:.4g}  table={t41.tolist()}")
    h41_pass = bool(p41 < 0.05 and (n_c and n_e) and f_c > f_e)

    print("\n=== H4.2  EXPOSED loss*: AM pathogenic fraction ≤ 0.5 ===")
    print(f"  pathogenic fraction = {f_e if n_e else float('nan'):.3f}  (n={n_e})")
    h42_pass = bool(n_e > 0 and f_e <= 0.5)

    # H4.3 neutral depleted in CORE vs loss*
    neu_core = int((neu["spt_class"] == "CORE").sum())
    neu_not = len(neu) - neu_core
    loss_core = int((loss["spt_class"] == "CORE").sum())
    loss_not = len(loss) - loss_core
    or43, p43, t43 = fisher_greater(loss_core, loss_not, neu_core, neu_not)
    print("\n=== H4.3  CORE enrichment: loss* vs neutral ===")
    print(f"  loss*   CORE {loss_core}/{len(loss)}")
    print(f"  neutral CORE {neu_core}/{len(neu)}")
    print(f"  Fisher loss* more CORE than neutral: OR={or43:.3f}  p={p43:.4g}  table={t43.tolist()}")
    h43_pass = bool(p43 < 0.05 and loss_core / max(len(loss), 1) > neu_core / max(len(neu), 1))

    # sensitivity: drop uncertain notes
    held_s = held[~held["uncertain_note"]]
    print(f"\n=== sensitivity: drop uncertain-note rows, n={len(held_s)} "
          f"(dropped {(held['uncertain_note']==True).sum()}) ===")
    loss_s = held_s[held_s["func_bundle"] == "loss*"]
    neu_s = held_s[held_s["func_bundle"] == "neutral"]
    print("  loss* class:", loss_s["spt_class"].value_counts().to_dict())
    print("  neutral class:", neu_s["spt_class"].value_counts().to_dict())

    # trafficking-annotated losses
    traf = loss[loss["literature_impact_trafficking"].astype(str).str.contains("loss", case=False, na=False)]
    print(f"\n=== trafficking-loss subset n={len(traf)} ===")
    if len(traf):
        print(traf[["hgvs_short", "spt_class", "am_class", "literature_impact_trafficking"]].to_string(index=False))

    verdict = {
        "n_heldout_missense": int(len(held)),
        "H4.1": {"pass": h41_pass, "p": p41, "OR": or41,
                 "CORE_pathogenic": [n_pc, n_c], "EXPOSED_pathogenic": [n_pe, n_e]},
        "H4.2": {"pass": h42_pass, "exposed_pathogenic_frac": f_e, "n_exposed_loss": n_e},
        "H4.3": {"pass": h43_pass, "p": p43, "OR": or43,
                 "loss_CORE": [loss_core, len(loss)], "neutral_CORE": [neu_core, len(neu)]},
        "GREY_loss_pathogenic": [n_pg, n_g],
    }
    print("\n=== pre-registered WP4 verdict ===")
    for k in ("H4.1", "H4.2", "H4.3"):
        print(f"  {k}: {'PASS' if verdict[k]['pass'] else 'FAIL'}")

    MET_SPT.mkdir(parents=True, exist_ok=True)
    held.to_csv(MET_SPT / "wp4_heldout_literature.tsv", sep="\t", index=False)
    (MET_SPT / "wp4_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(f"\noutputs -> {MET_SPT}")


if __name__ == "__main__":
    main()
