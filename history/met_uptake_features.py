#!/usr/bin/env python3
"""USM features: substrate/gate physics for OCT1 uptake (met_prereg_uptake.md).

    $MET_PY met_uptake_features.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from met_classify import THREE_TO_ONE  # noqa: E402
from met_wp5 import load_ca  # noqa: E402

MET_HDD = Path(os.environ.get("MET_HDD", str(Path(__file__).resolve().parent / "data")))
MET_PDB = Path(os.environ.get("MET_PDB", str(MET_HDD / "pdb")))
MET_SPT = Path(os.environ.get("MET_SPT", str(MET_HDD / "spt")))
TPT = MET_SPT / "tpt" / "oct1_tpt_variants.tsv"
OUT = MET_SPT / "uptake"

# Kyte–Doolittle / side-chain volume (Å³, approximate) / charge
HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4,
    "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9,
    "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}
CHARGE = {
    "D": -1, "E": -1, "K": 1, "R": 1, "H": 0.1,
}


def gate_ca_disp() -> dict[int, float]:
    """Per-residue CA displacement after superposition, identical AA only."""
    _, ca_a, aa_a = load_ca(MET_PDB / "8SC1.pdb")
    _, ca_b, aa_b = load_ca(MET_PDB / "8ET6.pdb")
    common = sorted(
        p for p in set(ca_a) & set(ca_b)
        if aa_a.get(p) == aa_b.get(p) and aa_a.get(p) not in (None, "X")
    )
    if len(common) < 50:
        return {}
    A = np.array([np.asarray(ca_a[p].coord, dtype=float) for p in common])
    B = np.array([np.asarray(ca_b[p].coord, dtype=float) for p in common])
    # Superimposer only to get RMS + rotran; use coord copies so PDBs stay intact
    from Bio.PDB.Atom import Atom
    fixed = [Atom("CA", A[i].copy(), 0, 1, " ", " CA ", i + 1, element="C") for i in range(len(common))]
    moving = [Atom("CA", B[i].copy(), 0, 1, " ", " CA ", i + 1, element="C") for i in range(len(common))]
    sup = Superimposer()
    sup.set_atoms(fixed, moving)
    R, t = np.asarray(sup.rotran[0]), np.asarray(sup.rotran[1])
    Bal = B @ R + t  # Bio.PDB: coord = coord @ rot + tran
    disp = np.linalg.norm(Bal - A, axis=1)
    print(f"  gate fit RMS={sup.rms:.3f} Å  median disp={float(np.median(disp)):.3f}")
    return {common[i]: float(disp[i]) for i in range(len(common))}



def main():
    if not TPT.exists():
        raise SystemExit(f"missing {TPT} — run met_tpt_features.py first")
    OUT.mkdir(parents=True, exist_ok=True)
    gate = gate_ca_disp()
    print(f"gate CA disp residues: {len(gate)}  median={np.median(list(gate.values())):.2f} Å")

    df = pd.read_csv(TPT, sep="\t")
    df["gate_disp"] = df["pos"].map(gate)
    df["d_charge"] = df.apply(
        lambda r: CHARGE.get(str(r["mut_aa"]), 0) - CHARGE.get(str(r["wt_aa"]), 0), axis=1)
    df["d_volume"] = df.apply(
        lambda r: VOLUME.get(str(r["mut_aa"]), np.nan) - VOLUME.get(str(r["wt_aa"]), np.nan), axis=1)
    df["d_hydro"] = df.apply(
        lambda r: HYDRO.get(str(r["mut_aa"]), np.nan) - HYDRO.get(str(r["wt_aa"]), np.nan), axis=1)
    df["abs_d_charge"] = df["d_charge"].abs()
    df["pocket_x_charge"] = df["pocket"].astype(float) * df["abs_d_charge"]
    df["gate_x_tm"] = df["gate_disp"].astype(float) * df["tm_interface"].astype(float)
    df["pocket_x_volume"] = df["pocket"].astype(float) * df["d_volume"].abs()

    # pull TPT U-head if available for U1 comparison
    lopo = MET_SPT / "tpt" / "oct1_tpt_lopo_preds.tsv"
    if lopo.exists():
        old = pd.read_csv(lopo, sep="\t")[["hgvs_short", "u_head", "t_head"]]
        df = df.drop(columns=[c for c in ("u_head", "t_head") if c in df.columns], errors="ignore")
        df = df.merge(old, on="hgvs_short", how="left")

    keep = [
        "hgvs_short", "pos", "wt_aa", "mut_aa", "cluster", "topology", "spt_class",
        "GFP_score", "SM73_0_score", "train_ok",
        "am_pathogenicity", "am_fitness", "ddg", "ddg_fitness",
        "rel_sasa_af2", "rel_sasa_8sc1", "rel_sasa_8et6", "delta_sasa_io",
        "plddt", "dist_metformin", "pocket", "tm_interface",
        "gate_disp", "d_charge", "d_volume", "d_hydro", "abs_d_charge",
        "pocket_x_charge", "gate_x_tm", "pocket_x_volume",
        "u_head", "t_head",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df[keep].to_csv(OUT / "oct1_usm_variants.tsv", sep="\t", index=False)
    print(f"variants {len(df)}  train_ok {int(df['train_ok'].sum())}  "
          f"pocket {int((df['pocket']==1).sum())}  gate_annot {df['gate_disp'].notna().sum()}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
