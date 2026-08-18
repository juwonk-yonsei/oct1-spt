#!/usr/bin/env python3
"""WP6: SPT on OCT2 (and optional G3 on OCT1 AF2 vs 8ET6 outward).

G1 skipped unless an independent DMS exists.
G2: AM-benign ∩ literature-loss enriched in EXPOSED (OCT2).
G3: AF2 vs experimental SPT agreement ≥80%.

    $MET_PY met_wp6.py g3-oct1          # AF2 OCT1 vs 8ET6 (no new AF2 needed)
    $MET_PY met_wp6.py oct2             # SPT OCT2 AF2 + 8ET9 + G2 literature
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Reuse locked SPT implementation
sys.path.insert(0, str(Path(__file__).resolve().parent))
from met_classify import (  # noqa: E402
    MET_PDB, MET_SEQ, MET_SPT, MET_STRUCT,
    classify_structure, fetch_uniprot_json, load_topology, print_counts,
    write_tsv,
)

HGVS_RE = re.compile(r"p\.\(([A-Z])(\d+)(del|[A-Z])\)", re.I)
MET_DMS = Path(os.environ.get("MET_DMS", str(Path(__file__).resolve().parent / "data" / "dms")))
MET_HDD = Path(os.environ.get("MET_HDD", str(Path(__file__).resolve().parent / "data")))


def parse_missing_pdb(path: Path) -> set[int]:
    """REMARK 465 missing residue numbers + ±2 neighbours."""
    missing: set[int] = set()
    in_block = False
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("REMARK 465") and "SSSEQ" in line:
            in_block = True
            continue
        if in_block:
            if not line.startswith("REMARK 465"):
                break
            parts = line.split()
            # REMARK 465 MET A 1   or REMARK 465 M RES C SSSEQI header
            if len(parts) >= 5 and parts[-1].isdigit():
                try:
                    missing.add(int(parts[-1]))
                except ValueError:
                    pass
    bad = set(missing)
    for m in list(missing):
        bad.update(range(max(1, m - 2), m + 3))
    return bad


def compare_classes(af2_rows, exp_rows, exclude: set[int], label: str):
    af2 = {r["pos"]: r for r in af2_rows}
    exp = {r["pos"]: r for r in exp_rows}
    common = sorted(set(af2) & set(exp) - exclude)
    rows, agree = [], 0
    for pos in common:
        a, e = af2[pos], exp[pos]
        match = a["class"] == e["class"]
        agree += int(match)
        rows.append({
            "pos": pos, "aa": a["aa"], "topology": a["topology"],
            "af2_sasa": a["rel_sasa"], "exp_sasa": e["rel_sasa"],
            "af2_class": a["class"], "exp_class": e["class"], "agree": int(match),
        })
    n = len(common)
    frac = agree / n if n else None
    print(f"\n=== G3 {label} ===")
    print(f"  common (excl. missing±2): {n}   agreement: {agree}/{n}"
          + (f"  ({100*frac:.1f}%)" if frac is not None else ""))
    if n:
        by = defaultdict(lambda: [0, 0])
        for r in rows:
            by[r["exp_class"]][1] += 1
            by[r["exp_class"]][0] += r["agree"]
        print(f"  {'exp class':10s} {'agree':>8s} {'n':>6s} {'pct':>7s}")
        for k in ("CORE", "EXPOSED", "GREY"):
            ok, tot = by[k]
            print(f"  {k:10s} {ok:8d} {tot:6d} {100*ok/tot if tot else float('nan'):6.1f}%")
    passed = bool(frac is not None and frac >= 0.80)
    print(f"  G3 gate ≥80%: {'PASS' if passed else 'FAIL'}")
    return rows, {"n": n, "agree": agree, "frac": frac, "pass": passed}


def cmd_g3_oct1():
    """OCT1 AF2 rank-1 vs 8ET6 outward (available now)."""
    acc = "O15245"
    topo = load_topology(fetch_uniprot_json(acc, MET_SEQ / f"{acc}_uniprot.json"))
    af2_dir = sorted(MET_STRUCT.glob("oct1_variants_*"))[-1]
    af2_pdb = sorted(af2_dir.glob("SLC22A1_WT_unrelaxed_rank_001_*.pdb"))[0]
    exp = MET_PDB / "8ET6.pdb"
    if not exp.exists():
        raise SystemExit("8ET6 missing — ./met_download.sh pdb 8ET6")
    _, af2_rows = classify_structure(af2_pdb, topo)
    _, exp_rows = classify_structure(exp, topo, chain_id="A")
    print_counts("OCT1 AF2 WT rank-1", af2_rows)
    print_counts("PDB 8ET6 outward OCT1CS", exp_rows)
    exclude = parse_missing_pdb(exp)
    rows, summary = compare_classes(af2_rows, exp_rows, exclude, "OCT1 AF2 vs 8ET6")
    write_tsv(MET_SPT / "oct1_af2_vs_8et6.tsv", rows,
              ["pos", "aa", "topology", "af2_sasa", "exp_sasa", "af2_class", "exp_class", "agree"])
    (MET_SPT / "g3_oct1_8et6.json").write_text(json.dumps(summary, indent=2) + "\n")


def latest_oct2_dir() -> Path | None:
    hits = sorted(MET_STRUCT.glob("oct2*")) + sorted(MET_STRUCT.glob("*SLC22A2*"))
    return hits[-1] if hits else None


def find_oct2_rank1(outdir: Path) -> Path:
    hits = sorted(outdir.glob("*unrelaxed_rank_001_*.pdb")) + \
           sorted(outdir.glob("SLC22A2*_unrelaxed_rank_001_*.pdb"))
    if not hits:
        raise SystemExit(f"no rank_001 PDB in {outdir}")
    return hits[0]


def cmd_oct2():
    acc = "O15244"
    topo = load_topology(fetch_uniprot_json(acc, MET_SEQ / f"{acc}_uniprot.json"))
    print(f"UniProt {acc} topology: {len(topo)} residues")

    outdir = latest_oct2_dir()
    if outdir is None:
        raise SystemExit("OCT2 AF2 not found. Run: ./met_predict.sh Metformin_HDD/sequences/SLC22A2_WT.fasta full oct2_wt")

    af2_pdb = find_oct2_rank1(outdir)
    print(f"AF2: {af2_pdb}")
    models = sorted(outdir.glob("*unrelaxed_rank_*.pdb"))
    _, af2_rows = classify_structure(af2_pdb, topo)
    print_counts("OCT2 AF2 rank-1", af2_rows)
    write_tsv(MET_SPT / "oct2_af2_rank1_spt.tsv", af2_rows,
              ["pos", "aa", "resname", "topology", "rel_sasa", "plddt", "class", "source"])

    # reproducibility if 5 models
    if len(models) >= 2:
        per = []
        for p in models:
            _, rows = classify_structure(p, topo)
            per.append({r["pos"]: r["class"] for r in rows})
        n_flip = sum(1 for pos in per[0] if len({d.get(pos) for d in per}) > 1)
        print(f"  reproducibility: {len(per[0]) - n_flip}/{len(per[0])} unanimous "
              f"({100*(len(per[0])-n_flip)/len(per[0]):.1f}%)  n_models={len(models)}")

    # G3 vs 8ET9
    exp_path = MET_PDB / "8ET9.pdb"
    g3 = {"pass": None, "note": "8ET9 missing"}
    if exp_path.exists():
        _, exp_rows = classify_structure(exp_path, topo, chain_id="A")
        print_counts("PDB 8ET9 OCT2CS", exp_rows)
        write_tsv(MET_SPT / "oct2_8et9_spt.tsv", exp_rows,
                  ["pos", "aa", "resname", "topology", "rel_sasa", "plddt", "class", "source"])
        exclude = parse_missing_pdb(exp_path)
        cmp_rows, g3 = compare_classes(af2_rows, exp_rows, exclude, "OCT2 AF2 vs 8ET9")
        write_tsv(MET_SPT / "oct2_af2_vs_8et9.tsv", cmp_rows,
                  ["pos", "aa", "topology", "af2_sasa", "exp_sasa", "af2_class", "exp_class", "agree"])
    else:
        print("8ET9 not downloaded — ./met_download.sh pdb 8ET9")

    # G2 literature
    lit_path = MET_DMS / "oct2_literature_variants.csv"
    am_path = Path(os.environ.get("MET_AM", str(MET_HDD / "alphamissense"))) / "by_target/SLC22A2_O15244.tsv"
    class_of = {r["pos"]: r["class"] for r in af2_rows}
    sasa_of = {r["pos"]: r["rel_sasa"] for r in af2_rows}
    am = pd.read_csv(am_path, sep="\t") if am_path.exists() else pd.DataFrame()
    am_map = {}
    if len(am):
        am_map = am.set_index("protein_variant")[["am_pathogenicity", "am_class"]].to_dict("index")

    lit = pd.read_csv(lit_path)
    recs = []
    for row in lit.to_dict("records"):
        m = HGVS_RE.search(str(row["hgvs"]))
        if not m:
            continue
        wt, pos, mut = m.group(1).upper(), int(m.group(2)), m.group(3)
        if mut.lower() == "del":
            continue
        short = f"{wt}{pos}{mut.upper()}"
        func = str(row.get("literature_impact_function") or "").lower()
        bundle = "loss*" if func in ("loss", "partial_loss") else func
        amh = am_map.get(short, {})
        recs.append({
            **row, "hgvs_short": short, "res_pos": pos, "spt_class": class_of.get(pos),
            "rel_sasa": sasa_of.get(pos), "am_class": amh.get("am_class"),
            "am_pathogenicity": amh.get("am_pathogenicity"), "func_bundle": bundle,
        })
    hdf = pd.DataFrame(recs)
    print("\n=== G2 OCT2 literature variants ===")
    cols = ["hgvs_short", "func_bundle", "spt_class", "rel_sasa", "am_class", "am_pathogenicity"]
    print(hdf[cols].to_string(index=False))

    loss = hdf[hdf["func_bundle"] == "loss*"].dropna(subset=["am_class", "spt_class"])
    hit = loss[(loss["am_class"] == "benign")]
    rest_n = len(hdf.dropna(subset=["am_class", "spt_class"]))
    # enrichment of EXPOSED among AM-benign ∩ loss vs all annotated missense in this small set
    # For tiny n, also compare hit EXPOSED frac vs background of this table
    bg = hdf.dropna(subset=["spt_class"])
    def frac_e(d):
        return float((d["spt_class"] == "EXPOSED").mean()) if len(d) else np.nan
    fisher_p, oddsr = np.nan, np.nan
    if len(hit) and len(bg):
        rest = bg.loc[~bg.index.isin(hit.index)]
        table = np.array([
            [(hit["spt_class"] == "EXPOSED").sum(), (hit["spt_class"] != "EXPOSED").sum()],
            [(rest["spt_class"] == "EXPOSED").sum(), (rest["spt_class"] != "EXPOSED").sum()],
        ], dtype=int)
        if table.min() >= 0 and table.sum() > 0:
            oddsr, fisher_p = stats.fisher_exact(table, alternative="greater")
    print(f"\n  AM-benign ∩ loss* n={len(hit)}  frac EXPOSED={frac_e(hit):.3f}")
    print(f"  background n={len(bg)}  frac EXPOSED={frac_e(bg):.3f}")
    print(f"  Fisher greater EXPOSED in hits: OR={oddsr}  p={fisher_p}")
    g2_pass = bool(pd.notna(fisher_p) and fisher_p < 0.05 and frac_e(hit) > frac_e(bg))
    print(f"  G2: {'PASS' if g2_pass else 'FAIL'}  (n is small — report counts, do not overclaim)")

    hdf.to_csv(MET_SPT / "wp6_oct2_literature.tsv", sep="\t", index=False)
    summary = {
        "G1": {"pass": None, "note": "no independent OCT2 DMS — not run"},
        "G2": {"pass": g2_pass, "p": None if pd.isna(fisher_p) else float(fisher_p),
               "OR": None if (pd.isna(oddsr) or not np.isfinite(oddsr)) else float(oddsr),
               "n_hit": int(len(hit)), "frac_exposed_hit": frac_e(hit),
               "frac_exposed_bg": frac_e(bg), "n_literature": int(len(hdf))},
        "G3": g3,
        "af2_counts": dict(Counter(r["class"] for r in af2_rows)),
        "af2_dir": str(outdir),
    }
    (MET_SPT / "wp6_verdict.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\noutputs -> {MET_SPT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["g3-oct1", "oct2"])
    args = ap.parse_args()
    MET_SPT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "g3-oct1":
        cmd_g3_oct1()
    else:
        cmd_oct2()


if __name__ == "__main__":
    main()
