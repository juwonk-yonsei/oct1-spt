#!/usr/bin/env python3
"""WP5 / P6: experimental-structure RMSD controls for the AF2 noise floor.

Pre-registered in met_prereg_grey.md (before this script was run):

  P6 primary   8SC1 (OCT1 inward WT) vs 8ET6 (OCT1CS outward)  → expect > 3.284 Å
  P6 sanity    8SC1 vs 8SC4 (both inward; 8SC4 metformin-bound) → expect ≤ 3.284 Å

Only residues with CA in both structures AND identical amino acid are used
(OCT1CS engineered substitutions are dropped).

    $MET_PY met_wp5.py
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, Superimposer

warnings.filterwarnings("ignore")

MET_HDD = Path(os.environ.get("MET_HDD", str(Path(__file__).resolve().parent / "data")))
MET_PDB = Path(os.environ.get("MET_PDB", str(MET_HDD / "pdb")))
MET_SEQ = Path(os.environ.get("MET_SEQ", str(MET_HDD / "sequences")))
MET_SPT = Path(os.environ.get("MET_SPT", str(MET_HDD / "spt")))

NOISE_MAX = 3.284  # Å, AF2 WT 5-model max CA RMSD
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

# UniProt O15245 TM helices (same as classifier)
TM = [(22, 42), (150, 170), (177, 197), (207, 229), (236, 256), (263, 283),
      (348, 368), (377, 397), (403, 423), (432, 452), (465, 485), (493, 513)]


def load_ca(pdb_path: Path, chain_id: str | None = None):
    struct = PDBParser(QUIET=True).get_structure(pdb_path.stem, str(pdb_path))
    model = next(struct.get_models())
    chain = model[chain_id] if chain_id else next(iter(model))
    ca, aa = {}, {}
    for res in chain:
        het, seq, icode = res.id
        if het != " " or icode != " " or "CA" not in res:
            continue
        ca[int(seq)] = res["CA"]
        aa[int(seq)] = THREE_TO_ONE.get(res.get_resname(), "X")
    return chain.id, ca, aa


def rmsd_pair(ca_a, aa_a, ca_b, aa_b, positions):
    common = [p for p in positions if p in ca_a and p in ca_b]
    if len(common) < 10:
        return None
    sup = Superimposer()
    sup.set_atoms([ca_a[p] for p in common], [ca_b[p] for p in common])
    return float(sup.rms), len(common)


def compare(label, pdb_a: Path, pdb_b: Path, chain_a=None, chain_b=None):
    id_a, ca_a, aa_a = load_ca(pdb_a, chain_a)
    id_b, ca_b, aa_b = load_ca(pdb_b, chain_b)
    identical = sorted(p for p in set(ca_a) & set(ca_b) if aa_a.get(p) == aa_b.get(p) and aa_a.get(p) != "X")
    numbered = sorted(set(ca_a) & set(ca_b))
    tm_id = [p for p in identical if any(a <= p <= b for a, b in TM)]

    r_id, n_id = rmsd_pair(ca_a, aa_a, ca_b, aa_b, identical) or (None, 0)
    r_all, n_all = rmsd_pair(ca_a, aa_a, ca_b, aa_b, numbered) or (None, 0)
    r_tm, n_tm = rmsd_pair(ca_a, aa_a, ca_b, aa_b, tm_id) or (None, 0)

    print(f"\n=== {label} ===")
    print(f"  {pdb_a.name} chain {id_a}  nCA={len(ca_a)}   vs   "
          f"{pdb_b.name} chain {id_b}  nCA={len(ca_b)}")
    print(f"  identical-AA CA RMSD : {r_id:.3f} Å   n={n_id}" if r_id is not None else "  identical-AA: n/a")
    print(f"  all-common  CA RMSD  : {r_all:.3f} Å   n={n_all}" if r_all is not None else "  all-common: n/a")
    print(f"  TM identical CA RMSD : {r_tm:.3f} Å   n={n_tm}" if r_tm is not None else "  TM identical: n/a")
    print(f"  noise floor          : {NOISE_MAX:.3f} Å")
    if r_id is not None:
        print(f"  vs noise             : {'ABOVE (P6-like)' if r_id > NOISE_MAX else 'within / below'}")
    aa_mismatch = sorted(p for p in numbered if aa_a.get(p) != aa_b.get(p))
    if aa_mismatch:
        print(f"  AA mismatches dropped: {len(aa_mismatch)}  e.g. {aa_mismatch[:12]}")
    return {
        "label": label,
        "pdb_a": pdb_a.name, "pdb_b": pdb_b.name,
        "rmsd_identical": r_id, "n_identical": n_id,
        "rmsd_all_common": r_all, "n_all_common": n_all,
        "rmsd_tm_identical": r_tm, "n_tm": n_tm,
        "n_aa_mismatch": len(aa_mismatch),
        "above_noise_identical": None if r_id is None else bool(r_id > NOISE_MAX),
    }


def main():
    sc1 = MET_PDB / "8SC1.pdb"
    sc4 = MET_PDB / "8SC4.pdb"
    et6 = MET_PDB / "8ET6.pdb"
    zh0 = MET_PDB / "7ZH0.pdb"
    for p in (sc1, sc4, et6):
        if not p.exists():
            raise SystemExit(f"missing {p} — run: ./met_download.sh pdb 8SC1 8SC4 8ET6")

    results = []
    results.append(compare("P6 primary: inward WT vs outward OCT1CS", sc1, et6, "A", "A"))
    results.append(compare("P6 sanity: inward apo vs inward metformin", sc1, sc4, "A", "A"))
    if zh0.exists():
        results.append(compare("secondary: OCT1 inward vs OCT3 outward", sc1, zh0, "A", "A"))
    else:
        print("\n[skip] 7ZH0 not downloaded (optional secondary)")

    p6 = results[0]
    sanity = results[1]
    p6_pass = bool(p6["above_noise_identical"])
    sanity_pass = bool(sanity["rmsd_identical"] is not None and sanity["rmsd_identical"] <= NOISE_MAX)
    print("\n=== pre-registered P6 verdict ===")
    print(f"  P6 primary (8SC1 vs 8ET6 > {NOISE_MAX} Å): {'PASS' if p6_pass else 'FAIL'}  "
          f"RMSD={p6['rmsd_identical']:.3f}")
    print(f"  P6 sanity  (8SC1 vs 8SC4 ≤ {NOISE_MAX} Å): {'PASS' if sanity_pass else 'FAIL'}  "
          f"RMSD={sanity['rmsd_identical']:.3f}")

    out = {
        "noise_max_A": NOISE_MAX,
        "P6_primary_pass": p6_pass,
        "P6_sanity_pass": sanity_pass,
        "pairs": results,
    }
    MET_SPT.mkdir(parents=True, exist_ok=True)
    (MET_SPT / "wp5_p6_rmsd.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\noutputs -> {MET_SPT / 'wp5_p6_rmsd.json'}")


if __name__ == "__main__":
    main()
