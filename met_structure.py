#!/usr/bin/env python
"""Structural analysis of the OCT1 wild-type and variant AlphaFold2 models.

Two questions:

1. Do the mutant structures actually differ from wild type? (AF2 is known to be
   largely blind to point mutations, so a near-zero RMSD is the expected result
   and is worth quantifying rather than assuming.)
2. Where do the variant residues sit? OCT1 has the MFS fold: two six-helix
   bundles enclosing a central substrate cavity. A residue's burial and its
   radial distance from the pseudo-symmetry axis say whether it lines that
   cavity, and that context is what AlphaMissense's score cannot provide.

    $MET_PY met_structure.py
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.SASA import ShrakeRupley

warnings.filterwarnings("ignore")

STRUCT = Path(os.environ.get("MET_STRUCT", str(Path(__file__).resolve().parent / "data" / "structures")))
WT = "SLC22A1_WT"
VARIANTS = {"R61C": 61, "C88R": 88, "G401S": 401, "M420del": 420, "G465R": 465}

# Tien et al. 2013, theoretical maximum solvent accessibility (A^2)
MAX_ASA = {"ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167, "GLN": 225,
           "GLU": 223, "GLY": 104, "HIS": 224, "ILE": 197, "LEU": 201, "LYS": 236,
           "MET": 224, "PHE": 240, "PRO": 159, "SER": 155, "THR": 172, "TRP": 285,
           "TYR": 263, "VAL": 174}


def rank1(outdir: Path, name: str) -> Path:
    hits = sorted(outdir.glob(f"{name}_unrelaxed_rank_001_*.pdb"))
    if not hits:
        raise FileNotFoundError(f"no rank_001 PDB for {name}")
    return hits[0]


def load(path: Path):
    return PDBParser(QUIET=True).get_structure("s", str(path))[0]["A"]


def ca_map(chain):
    return {r.id[1]: r["CA"] for r in chain if "CA" in r}


def main():
    outdir = sorted(STRUCT.glob("oct1_variants_*"))[-1]
    print(f"models: {outdir.name}\n")

    wt = load(rank1(outdir, WT))
    wt_ca = ca_map(wt)

    # ---- 0. noise floor: how much do models of the SAME sequence differ? ----
    # Without this any wild-type-vs-mutant RMSD is uninterpretable, because AF2's
    # own model-to-model spread can be as large as a real mutational effect.
    wt_models = sorted(outdir.glob(f"{WT}_unrelaxed_rank_*.pdb"))
    spread = []
    for i in range(len(wt_models)):
        for j in range(i + 1, len(wt_models)):
            a, b = load(wt_models[i]), load(wt_models[j])
            aca, bca = ca_map(a), ca_map(b)
            common = sorted(set(aca) & set(bca))
            sup = Superimposer()
            sup.set_atoms([aca[k] for k in common], [bca[k] for k in common])
            spread.append(sup.rms)
    print("=== noise floor: wild type vs itself (5 AF2 models, 10 pairs) ===")
    print(f"  CA RMSD  min {min(spread):.3f}A   median {np.median(spread):.3f}A   "
          f"max {max(spread):.3f}A")
    print("  A wild-type-vs-mutant RMSD inside this range means nothing.\n")
    noise_max = max(spread)

    # ---- 1. does the structure change at all? ----
    print("=== wild type vs variant (rank-1 models, CA superposition) ===")
    print(f"{'variant':10s} {'global RMSD':>12s} {'local RMSD':>11s}   (local = residues within 10A of the site)")
    for var, pos in VARIANTS.items():
        mut = load(rank1(outdir, f"SLC22A1_{var}"))
        mut_ca = ca_map(mut)

        # M420del removes one residue, so mutant i>=420 corresponds to WT i+1
        if var.endswith("del"):
            pairs = [(i, i) for i in mut_ca if i < pos] + \
                    [(i + 1, i) for i in mut_ca if i >= pos]
        else:
            pairs = [(i, i) for i in mut_ca if i in wt_ca]
        pairs = [(w, m) for w, m in pairs if w in wt_ca and m in mut_ca]

        fixed = [wt_ca[w] for w, _ in pairs]
        moving = [mut_ca[m] for _, m in pairs]
        sup = Superimposer()
        sup.set_atoms(fixed, moving)
        sup.apply([a for a in mut.get_atoms()])
        glob_rmsd = sup.rms

        centre = wt_ca[pos].coord if pos in wt_ca else wt_ca[min(wt_ca)].coord
        near = [(w, m) for w, m in pairs
                if np.linalg.norm(wt_ca[w].coord - centre) <= 10.0]
        if near:
            d = [np.linalg.norm(wt_ca[w].coord - mut_ca[m].coord) for w, m in near]
            local = float(np.sqrt(np.mean(np.square(d))))
        else:
            local = float("nan")
        print(f"{var:10s} {glob_rmsd:11.3f}A {local:10.3f}A   ({len(near)} residues)"
              f"{'' if glob_rmsd > noise_max else '   within noise'}")

    # ---- 2. structural context of each site in the wild type ----
    ShrakeRupley().compute(wt, level="R")
    coords = np.array([a.coord for a in wt_ca.values()])
    centroid = coords.mean(axis=0)
    # longest principal axis approximates the membrane normal for a TM bundle
    axis = np.linalg.svd(coords - centroid)[2][0]

    def radial(xyz):
        v = xyz - centroid
        return float(np.linalg.norm(v - np.dot(v, axis) * axis))

    radii = np.array([radial(a.coord) for a in wt_ca.values()])

    print("\n=== where each variant sits in the wild-type fold ===")
    print(f"{'site':8s} {'residue':8s} {'pLDDT':>7s} {'rel.SASA':>9s} {'burial':>10s} "
          f"{'radial':>8s} {'axis pctile':>12s}")
    for var, pos in VARIANTS.items():
        res = wt[pos]
        rsa = res.sasa / MAX_ASA.get(res.get_resname(), 200) * 100
        plddt = res["CA"].get_bfactor()
        r = radial(res["CA"].coord)
        pct = (radii < r).mean() * 100
        state = "buried" if rsa < 10 else ("partial" if rsa < 35 else "exposed")
        print(f"{var:8s} {res.get_resname():8s} {plddt:7.1f} {rsa:8.1f}% {state:>10s} "
              f"{r:7.1f}A {pct:11.1f}%")

    print(f"\n  radial = distance from the bundle's long axis; a low percentile means the")
    print(f"  residue points towards the central cavity rather than the lipid-facing rim.")

    # ---- 3. per-residue confidence around each site ----
    print("\n=== local model confidence (mean pLDDT +/-5 residues) ===")
    for var, pos in VARIANTS.items():
        win = [wt_ca[i].get_bfactor() for i in range(pos - 5, pos + 6) if i in wt_ca]
        print(f"  {var:10s} {np.mean(win):5.1f}")


if __name__ == "__main__":
    main()
