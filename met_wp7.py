#!/usr/bin/env python3
"""WP7 failure modes + 8SC1 vs 8SC4 SPT sanity.

Documents where AM / ΔΔG / AF2 RMSD are structurally the wrong tool:
  disulfide Cys, PTM Tyr, deletion, trafficking literature, design-set ΔΔG.
Also: does metformin-bound inward 8SC4 flip SPT vs apo 8SC1?

    $MET_PY met_wp7.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from met_classify import (  # noqa: E402
    MET_PDB, MET_SEQ, MET_SPT,
    classify_structure, fetch_uniprot_json, load_topology, print_counts, write_tsv,
)
from met_wp6 import compare_classes, parse_missing_pdb  # noqa: E402

MET_HDD = Path(os.environ.get("MET_HDD", str(Path(__file__).resolve().parent / "data")))
MET_DMS = Path(os.environ.get("MET_DMS", str(MET_HDD / "dms")))
MET_AM = Path(os.environ.get("MET_AM", str(MET_HDD / "alphamissense")))
MET_DDG = Path(os.environ.get("MET_DDG", str(MET_HDD / "ddg")))

DISULFIDES = [(50, 121), (62, 102), (88, 142)]
PTM = {"Y240F": 240, "Y361F": 361, "Y376F": 376}
DESIGN_MISSENSE = {"R61C": 61, "C88R": 88, "G401S": 401, "G465R": 465}
MUT_RE = re.compile(r"^([A-Z])[A-Z]?(\d+)([A-Z])$")


def load_raw_ddg():
    path = MET_DDG / "oct1_af2_wt_thermompnn.csv"
    df = pd.read_csv(path)
    ddg_col = next(c for c in df.columns if "ddg" in c.lower() or "kcal" in c.lower())
    mut_col = next(c for c in df.columns if "mut" in c.lower())
    df = df.rename(columns={ddg_col: "ddg", mut_col: "mutation"})
    parsed = df["mutation"].astype(str).map(lambda s: MUT_RE.match(s.strip()))
    df = df[parsed.notna()].copy()
    df["wt"] = [m.group(1) for m in parsed.dropna()]
    df["pos"] = [int(m.group(2)) for m in parsed.dropna()]
    df["mut"] = [m.group(3) for m in parsed.dropna()]
    df = df[df["wt"] != df["mut"]]
    df["hgvs_short"] = df["wt"] + df["pos"].astype(str) + df["mut"]
    return df


def lookup_am(am, short):
    hit = am.loc[am["protein_variant"] == short]
    if hit.empty:
        return None, None
    return float(hit.iloc[0]["am_pathogenicity"]), str(hit.iloc[0]["am_class"])


def main():
    spt = pd.read_csv(MET_SPT / "oct1_af2_rank1_spt.tsv", sep="\t")
    class_of = dict(zip(spt["pos"].astype(int), spt["class"]))
    sasa_of = dict(zip(spt["pos"].astype(int), spt["rel_sasa"]))
    topo_of = dict(zip(spt["pos"].astype(int), spt["topology"]))
    am = pd.read_csv(MET_AM / "by_target/SLC22A1_O15245.tsv", sep="\t")
    ddg = load_raw_ddg()
    dms = pd.read_csv(MET_DMS / "oct1_combined_scores.csv")
    dms_m = dms[(dms["mutation_type"] == "M") & (dms["is.wt"] == False)].copy()
    dms_m["hgvs_short"] = dms_m["wt_pos"].astype(str) + dms_m["pos"].astype(int).astype(str) + dms_m["variants"].astype(str)
    gfp_of = dms_m.dropna(subset=["GFP_score"]).set_index("hgvs_short")["GFP_score"].to_dict()

    rows = []

    print("=== design missense: AM / ΔΔG / DMS GFP ===")
    print(f"{'var':8s} {'class':8s} {'sasa':>6s} {'ΔΔG':>8s} {'AM':>8s} {'AMcls':12s} {'GFP':>8s}")
    for name, pos in DESIGN_MISSENSE.items():
        rec = ddg.loc[ddg["hgvs_short"] == name]
        ddg_v = float(rec["ddg"].iloc[0]) if len(rec) else np.nan
        am_s, am_c = lookup_am(am, name)
        gfp = gfp_of.get(name)
        print(f"{name:8s} {class_of.get(pos,'?'):8s} {sasa_of.get(pos, np.nan):6.1f} "
              f"{ddg_v:8.3f} {am_s if am_s is not None else float('nan'):8.3f} "
              f"{(am_c or ''):12s} {gfp if gfp is not None else float('nan'):8.3f}")
        rows.append({
            "mode": "design_missense", "variant": name, "pos": pos,
            "spt_class": class_of.get(pos), "rel_sasa": sasa_of.get(pos),
            "topology": topo_of.get(pos), "ddg": ddg_v,
            "am_pathogenicity": am_s, "am_class": am_c, "GFP_score": gfp,
            "note": "R61C expected ΔΔG≈0 (EXPOSED); CORE expected high ΔΔG",
        })

    print("\n=== disulfide Cys: median ΔΔG of non-self substitutions ===")
    cys_pos = sorted({p for pair in DISULFIDES for p in pair})
    print(f"{'pos':>5s} {'pair':>10s} {'class':8s} {'sasa':>6s} {'medΔΔG':>8s} {'C→R':>8s} {'C→S':>8s} {'C→A':>8s}")
    for pos in cys_pos:
        pair = next((f"C{a}–C{b}" for a, b in DISULFIDES if pos in (a, b)), "")
        sub = ddg[ddg["pos"] == pos]
        med = float(sub["ddg"].median()) if len(sub) else np.nan
        def one(mt):
            h = sub[sub["mut"] == mt]
            return float(h["ddg"].iloc[0]) if len(h) else np.nan
        print(f"{pos:5d} {pair:>10s} {class_of.get(pos,'?'):8s} {sasa_of.get(pos, np.nan):6.1f} "
              f"{med:8.3f} {one('R'):8.3f} {one('S'):8.3f} {one('A'):8.3f}")
        rows.append({
            "mode": "disulfide_cys", "variant": f"C{pos}*", "pos": pos,
            "spt_class": class_of.get(pos), "rel_sasa": sasa_of.get(pos),
            "topology": topo_of.get(pos), "ddg": med,
            "am_pathogenicity": None, "am_class": None, "GFP_score": None,
            "note": f"pair {pair}; median over 19 substitutions",
        })

    print("\n=== PTM Tyr→Phe (literature trafficking loss) ===")
    print(f"{'var':8s} {'class':8s} {'sasa':>6s} {'topo':16s} {'ΔΔG':>8s} {'AM':>8s} {'AMcls':12s} {'GFP':>8s}")
    for name, pos in PTM.items():
        rec = ddg.loc[ddg["hgvs_short"] == name]
        ddg_v = float(rec["ddg"].iloc[0]) if len(rec) else np.nan
        am_s, am_c = lookup_am(am, name)
        gfp = gfp_of.get(name)
        print(f"{name:8s} {class_of.get(pos,'?'):8s} {sasa_of.get(pos, np.nan):6.1f} "
              f"{str(topo_of.get(pos,'')):16s} {ddg_v:8.3f} "
              f"{am_s if am_s is not None else float('nan'):8.3f} {(am_c or ''):12s} "
              f"{gfp if gfp is not None else float('nan'):8.3f}")
        rows.append({
            "mode": "ptm_yf", "variant": name, "pos": pos,
            "spt_class": class_of.get(pos), "rel_sasa": sasa_of.get(pos),
            "topology": topo_of.get(pos), "ddg": ddg_v,
            "am_pathogenicity": am_s, "am_class": am_c, "GFP_score": gfp,
            "note": "phosphorylation site; AM/ΔΔG expected uninformative",
        })

    print("\n=== deletion M420del ===")
    print("  AM unscorable (indel). ThermoMPNN single-missense N/A. SPT class CORE (rel.SASA 5.5%).")
    rows.append({
        "mode": "deletion", "variant": "M420del", "pos": 420,
        "spt_class": class_of.get(420), "rel_sasa": sasa_of.get(420),
        "topology": topo_of.get(420), "ddg": None,
        "am_pathogenicity": None, "am_class": None, "GFP_score": gfp_of.get("M420del"),
        "note": "AM indel gap; substrate-specific loss with near-normal abundance in some reports",
    })

    lit = pd.read_csv(MET_DMS / "literature_variants.csv")
    traf = lit[lit["literature_impact_trafficking"].astype(str).str.lower().eq("loss")]
    print(f"\n=== literature trafficking-loss (n={len(traf)}) ===")
    print(f"{'hgvs':16s} {'func':12s} {'class':8s} {'AM':12s}")
    for rec in traf.to_dict("records"):
        m = re.search(r"p\.\(([A-Z])(\d+)(del|[A-Z])\)", str(rec["hgvs"]), re.I)
        if not m:
            continue
        short = f"{m.group(1).upper()}{m.group(2)}{m.group(3)}"
        pos = int(m.group(2))
        am_s, am_c = (None, "unscorable") if m.group(3).lower() == "del" else lookup_am(am, short)
        print(f"{rec['hgvs']:16s} {str(rec['literature_impact_function']):12s} "
              f"{str(class_of.get(pos,'?')):8s} {str(am_c or ''):12s}")

    # 8SC1 vs 8SC4 SPT
    acc = "O15245"
    topo = load_topology(fetch_uniprot_json(acc, MET_SEQ / f"{acc}_uniprot.json"))
    sc1, sc4 = MET_PDB / "8SC1.pdb", MET_PDB / "8SC4.pdb"
    _, r1 = classify_structure(sc1, topo, chain_id="A")
    _, r4 = classify_structure(sc4, topo, chain_id="A")
    print_counts("8SC1 inward apo", r1)
    print_counts("8SC4 inward + metformin", r4)
    exclude = parse_missing_pdb(sc1) | parse_missing_pdb(sc4)
    cmp_rows, summary = compare_classes(r1, r4, exclude, "8SC1 vs 8SC4 SPT")
    write_tsv(MET_SPT / "oct1_8sc1_vs_8sc4_spt.tsv", cmp_rows,
              ["pos", "aa", "topology", "af2_sasa", "exp_sasa", "af2_class", "exp_class", "agree"])

    pd.DataFrame(rows).to_csv(MET_SPT / "wp7_failure_modes.tsv", sep="\t", index=False)
    out = {
        "disulfide_pairs": [list(p) for p in DISULFIDES],
        "ptm": list(PTM),
        "design_ddg": {r["variant"]: r["ddg"] for r in rows if r["mode"] == "design_missense"},
        "spt_8sc1_vs_8sc4": summary,
        "note": "WP7; thresholds unchanged. 8SC4 SPT is ligand sanity not a new G3.",
    }
    (MET_SPT / "wp7_verdict.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\noutputs -> {MET_SPT}")


if __name__ == "__main__":
    main()
