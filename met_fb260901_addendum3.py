#!/usr/bin/env python3
"""Addendum-3: paper-path RMSD+TM, Cheng-miss split, phyloP on 8SC1/AFDB.

    source /SSD1T/PhD/AlphaFold/met_env.sh
    $MET_PY met_fb260901_addendum3.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import Superimposer

sys.path.insert(0, "/SSD1T/PhD/AlphaFold")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from met_fb260901 import (  # noqa: E402
    GFP_CUT,
    MS1,
    N_BOOT,
    OUT,
    a1_model,
    dump,
    grantham,
)
from met_wp5 import TM, load_ca  # noqa: E402

MET_HDD = Path(os.environ.get("MET_HDD", "/HDD8T1/WORK/Metformin_HDD"))
MET_SPT = Path(os.environ.get("MET_SPT", str(MET_HDD / "spt")))
MET_PGYM = MET_HDD / "proteingym"
ADD = OUT / "addendum"
ADD2 = OUT / "addendum2"
ADD3 = OUT / "addendum3"

PGYM = {
    "ADRB2_HUMAN": "P07550",
    "CCR5_HUMAN": "P51681",
    "HMDH_HUMAN": "P04035",
    "NPC1_HUMAN": "O15118",
    "SC6A4_HUMAN": "P31645",
    "VKOR1_HUMAN": "Q9BQB6",
    "S22A1_HUMAN": "O15245",
}
POPS = [
    ("EUR", "af_nfe"),
    ("EAS", "af_eas"),
    ("SAS", "af_sas"),
    ("AFR", "af_afr"),
    ("AMR", "af_amr"),
]


def numpy_kabsch(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    return float(np.sqrt(((a @ r - b) ** 2).sum(1).mean()))


def tm_d0(L: int) -> float:
    L = max(int(L), 21)
    return 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8


def tmscore(d: np.ndarray, L: int) -> float:
    d0 = tm_d0(L)
    return float(np.sum(1.0 / (1.0 + (d / d0) ** 2)) / L)


def paper_overlay(pdb_a: Path, pdb_b: Path, positions=None, identical_aa=False) -> dict:
    """Manuscript P6 path: Bio.PDB Superimposer on CA, then Zhang TM-score."""
    _ida, ca_a, aa_a = load_ca(pdb_a)
    _idb, ca_b, aa_b = load_ca(pdb_b)
    common = sorted(set(ca_a) & set(ca_b))
    if positions is not None:
        common = [p for p in common if p in positions]
    n_mismatch = sum(1 for p in common if aa_a.get(p) != aa_b.get(p))
    if identical_aa:
        common = [p for p in common if aa_a.get(p) == aa_b.get(p) and aa_a.get(p) != "X"]
    if len(common) < 10:
        return {"n": len(common), "ok": False}
    fixed = [ca_a[p] for p in common]
    moving = [ca_b[p] for p in common]
    sup = Superimposer()
    sup.set_atoms(fixed, moving)
    rot, tran = sup.rotran
    a = np.array([ca_a[p].coord for p in common], dtype=float)
    b = np.array([ca_b[p].coord for p in common], dtype=float)
    b_sup = np.dot(b, rot) + tran
    d = np.linalg.norm(a - b_sup, axis=1)
    raw = float(np.sqrt(((a - b) ** 2).sum(1).mean()))
    return {
        "ok": True,
        "n": len(common),
        "n_aa_mismatch_before_filter": int(n_mismatch),
        "rmsd_superimposer": float(sup.rms),
        "rmsd_numpy_kabsch": numpy_kabsch(a, b),
        "rmsd_unaligned": raw,
        "tmscore_Laligned": tmscore(d, len(common)),
        "tmscore_Lref": tmscore(d, len(ca_a)),
        "n_ref": len(ca_a),
        "code_path": "Bio.PDB.Superimposer.set_atoms (met_wp5 / met_structure / manuscript Fig. 4)",
    }


def plddt_ca(path: Path) -> dict[int, float]:
    out = {}
    with path.open() as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                out[int(line[22:26])] = float(line[60:66])
    return out


def oct1_rmsd() -> dict:
    cf = Path(
        "/HDD8T1/WORK/Metformin_HDD/structures/oct1_variants_20260811_204134/"
        "SLC22A1_WT_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb"
    )
    afdb = ADD / "afdb" / "AF-O15245-F1-model_v6.pdb"
    pgym = MET_PGYM / "structures" / "ProteinGym_AF2_structures" / "S22A1_HUMAN.pdb"
    _, _, _ = load_ca(afdb)
    tm_pos = {p for a, b in TM for p in range(a, b + 1)}
    plddt = plddt_ca(cf)
    plddt_b = plddt_ca(afdb)
    hi = {p for p in plddt if plddt[p] >= 70 and plddt_b.get(p, 0) >= 70}
    out = {
        "colabfold": str(cf),
        "afdb_v6": str(afdb),
        "all_ca": paper_overlay(cf, afdb),
        "identical_aa": paper_overlay(cf, afdb, identical_aa=True),
        "plddt70": paper_overlay(cf, afdb, positions=hi),
        "transmembrane": paper_overlay(cf, afdb, positions=tm_pos, identical_aa=True),
        "tm_plddt70": paper_overlay(cf, afdb, positions=hi & tm_pos, identical_aa=True),
    }
    if pgym.exists():
        out["colabfold_vs_pgym_S22A1"] = paper_overlay(cf, pgym, identical_aa=True)
        out["pgym_S22A1_vs_afdb_v6"] = paper_overlay(pgym, afdb, identical_aa=True)
    prev = 35.192728551963555
    now = out["all_ca"]["rmsd_superimposer"]
    out["vs_previous_numpy_kabsch_35A"] = {
        "previous": prev,
        "now": now,
        "delta": now - prev,
        "dropped": now < prev - 5.0,
        "same_fold_tm_ge_0.5": bool(out["all_ca"]["tmscore_Laligned"] >= 0.5),
        "withdraw_protocol_claim": bool(now < 5.0 or out["all_ca"]["tmscore_Laligned"] >= 0.90),
    }
    return out


def pgym_rmsd() -> list[dict]:
    struct = MET_PGYM / "structures" / "ProteinGym_AF2_structures"
    rows = []
    for uid, acc in PGYM.items():
        pg = struct / f"{uid}.pdb"
        af = ADD / "afdb" / f"AF-{acc}-F1-model_v6.pdb"
        rec = {"UniProt_ID": uid, "uniprot_acc": acc}
        if not pg.exists() or not af.exists():
            rec["ok"] = False
            rec["reason"] = "missing pdb"
            rows.append(rec)
            continue
        rec.update(paper_overlay(pg, af, identical_aa=True))
        rec["UniProt_ID"] = uid
        rec["uniprot_acc"] = acc
        plddt_a, plddt_b = plddt_ca(pg), plddt_ca(af)
        hi = {p for p in plddt_a if plddt_a[p] >= 70 and plddt_b.get(p, 0) >= 70}
        rec["plddt70"] = paper_overlay(pg, af, positions=hi, identical_aa=True)
        rec["previous_reported_v6"] = {
            "ADRB2_HUMAN": 19.4, "CCR5_HUMAN": 8.47, "HMDH_HUMAN": 24.7,
            "NPC1_HUMAN": 37.1, "SC6A4_HUMAN": 30.9, "VKOR1_HUMAN": 19.1,
            "S22A1_HUMAN": None,
        }.get(uid)
        rows.append(rec)
        print(uid, "RMSD", rec.get("rmsd_superimposer"), "TM", rec.get("tmscore_Laligned"))
    pd.DataFrame([
        {
            "UniProt_ID": r["UniProt_ID"],
            "n": r.get("n"),
            "rmsd_superimposer": r.get("rmsd_superimposer"),
            "rmsd_numpy_kabsch": r.get("rmsd_numpy_kabsch"),
            "rmsd_unaligned": r.get("rmsd_unaligned"),
            "tmscore_Laligned": r.get("tmscore_Laligned"),
            "tmscore_Lref": r.get("tmscore_Lref"),
            "rmsd_plddt70": (r.get("plddt70") or {}).get("rmsd_superimposer"),
            "tm_plddt70": (r.get("plddt70") or {}).get("tmscore_Laligned"),
            "previous_v6": r.get("previous_reported_v6"),
        }
        for r in rows
    ]).to_csv(ADD3 / "rmsd_tm_proteingym_afdb_v6.tsv", sep="\t", index=False)
    return rows


def cheng_miss_split() -> dict:
    df = pd.read_csv(ADD / "gnomad_gfp_loss.tsv", sep="\t")
    miss = df[df["cheng_miss"] == True].copy()  # noqa: E712
    assert len(miss) == 130, len(miss)

    def block(sub: pd.DataFrame) -> dict:
        out = {
            "n": int(len(sub)),
            "sum_af": {pop: float(sub[col].fillna(0).sum()) for pop, col in POPS},
        }
        by = {}
        for lab in ("CORE", "EXPOSED", "GREY"):
            s = sub[sub["class"] == lab]
            by[lab] = {
                "n": int(len(s)),
                "sum_af": {pop: float(s[col].fillna(0).sum()) for pop, col in POPS},
            }
        out["by_class"] = by
        return out

    benign = miss[miss["am_class_cheng"] == "benign"]
    amb = miss[miss["am_class_cheng"] == "ambiguous"]
    payload = {
        "n_gfp_loss": int(len(df)),
        "n_miss": int(len(miss)),
        "benign_lt_0.34": block(benign),
        "ambiguous": block(amb),
        "by_class_then_cheng": {},
        "top_benign_by_eas": (
            miss[miss["am_class_cheng"] == "benign"]
            .sort_values("af_eas", ascending=False)[["hgvs_short", "class", "am", "am_class_cheng", "af_nfe", "af_eas", "af_sas"]]
            .head(8).to_dict(orient="records")
        ),
        "top_ambiguous_by_sas": (
            miss[miss["am_class_cheng"] == "ambiguous"]
            .sort_values("af_sas", ascending=False)[["hgvs_short", "class", "am", "am_class_cheng", "af_nfe", "af_eas", "af_sas"]]
            .head(8).to_dict(orient="records")
        ),
        "P283L": miss[miss["hgvs_short"] == "P283L"].to_dict(orient="records"),
        "R287W": miss[miss["hgvs_short"] == "R287W"].to_dict(orient="records"),
    }
    for lab in ("CORE", "EXPOSED", "GREY", "ALL"):
        s = miss if lab == "ALL" else miss[miss["class"] == lab]
        payload["by_class_then_cheng"][lab] = {
            "n_miss": int(len(s)),
            "n_benign": int((s["am_class_cheng"] == "benign").sum()),
            "n_ambiguous": int((s["am_class_cheng"] == "ambiguous").sum()),
            "benign_sum_af": {pop: float(s.loc[s["am_class_cheng"] == "benign", col].fillna(0).sum()) for pop, col in POPS},
            "ambiguous_sum_af": {pop: float(s.loc[s["am_class_cheng"] == "ambiguous", col].fillna(0).sum()) for pop, col in POPS},
        }
    rows = []
    for split, name in ((benign, "benign_<0.34"), (amb, "ambiguous")):
        for lab in ("CORE", "EXPOSED", "GREY"):
            s = split[split["class"] == lab]
            rec = {"split": name, "class": lab, "n": int(len(s))}
            for pop, col in POPS:
                rec[f"sumAF_{pop}"] = float(s[col].fillna(0).sum())
            rows.append(rec)
    pd.DataFrame(rows).to_csv(ADD3 / "cheng_miss_130_benign_ambiguous.tsv", sep="\t", index=False)
    return payload


def phylop_three_plus_afdb() -> dict:
    val = pd.read_csv(MET_SPT / "wp3_validation_missense.tsv", sep="\t")
    val["pos"] = val["pos"].astype(int)
    val["dms_loss"] = val["GFP_score"] <= GFP_CUT
    val["grantham"] = [grantham(w, m) for w, m in zip(val["wt_aa"], val["mut_aa"])]
    af2 = pd.read_csv(MET_SPT / "oct1_af2_rank1_spt.tsv", sep="\t")
    sc1 = pd.read_csv(MET_SPT / "oct1_8sc1_spt.tsv", sep="\t")
    cmp = pd.read_csv(MET_SPT / "oct1_af2_vs_8sc1.tsv", sep="\t")
    afdb = pd.read_csv(ADD2 / "oct1_afdb_v6_spt.tsv", sep="\t")
    cons = pd.read_csv(ADD2 / "oct1_phylop_phastcons.tsv", sep="\t")
    val = val.merge(cons[["pos", "cons_phylop", "cons_phastcons"]], on="pos", how="left")
    val["class_af2"] = val["pos"].map(dict(zip(af2["pos"].astype(int), af2["class"])))
    val["class_8sc1"] = val["pos"].map(dict(zip(sc1["pos"].astype(int), sc1["class"])))
    val["class_afdb"] = val["pos"].map(dict(zip(afdb["pos"].astype(int), afdb["class"])))
    agree_pos = set(cmp.loc[cmp["af2_class"] == cmp["exp_class"], "pos"].astype(int))
    val["class_agree"] = np.where(val["pos"].isin(agree_pos), val["class_af2"], np.nan)
    labels = {
        "AF2": "class_af2",
        "8SC1": "class_8sc1",
        "agree": "class_agree",
        "AFDB_v6": "class_afdb",
    }
    out = {}
    rows = []
    for name, col in labels.items():
        phy = a1_model(val, col, "cons_phylop", n_boot=N_BOOT)
        pha = a1_model(val, col, "cons_phastcons", n_boot=N_BOOT)
        out[name] = {"phylop": phy, "phastcons": pha}
        boot = phy["logit_or_clustered"]
        rows.append({
            "label": name,
            "n_variants": phy["n_variants"],
            "n_residues": phy["n_residues"],
            "EXPOSED_OR_median": boot["median"],
            "EXPOSED_OR_lo": boot["ci95"][0],
            "EXPOSED_OR_hi": boot["ci95"][1],
            "CORE_EXPOSED_OR_median": boot["or_core_vs_exposed_median"],
            "CORE_EXPOSED_OR_lo": boot["or_core_vs_exposed_ci95"][0],
            "CORE_EXPOSED_OR_hi": boot["or_core_vs_exposed_ci95"][1],
            "excludes_1": bool(boot["ci95"][1] < 1 or boot["ci95"][0] > 1),
            "GEE_OR": phy["gee"].get("or"),
            "GEE_lo": (phy["gee"].get("ci95") or [None, None])[0],
            "GEE_hi": (phy["gee"].get("ci95") or [None, None])[1],
            "MH_OR": phy["mantel_haenszel"].get("or"),
            "MH_lo": (phy["mantel_haenszel"].get("ci95") or [None, None])[0],
            "MH_hi": (phy["mantel_haenszel"].get("ci95") or [None, None])[1],
        })
        print(name, "phyloP EXPOSED OR", boot["median"], boot["ci95"], "n", phy["n_variants"], phy["n_residues"])
    pd.DataFrame(rows).to_csv(ADD3 / "a1_phylop_by_label.tsv", sep="\t", index=False)
    return out


def main():
    ADD3.mkdir(parents=True, exist_ok=True)
    print("=== paper-path RMSD + TM-score ===")
    oct1 = oct1_rmsd()
    print(json.dumps({k: oct1[k] for k in oct1 if k != "colabfold"}, indent=2, default=str)[:2500])
    print("=== ProteinGym vs AFDB v6 ===")
    pg = pgym_rmsd()
    print("=== Cheng miss 130 split ===")
    miss = cheng_miss_split()
    print("benign n", miss["benign_lt_0.34"]["n"], "ambiguous n", miss["ambiguous"]["n"])
    print("=== phyloP A1 on 8SC1 / agree / AFDB ===")
    phy = phylop_three_plus_afdb()
    payload = {
        "oct1_colabfold_vs_afdb_v6": oct1,
        "proteingym_vs_afdb_v6": pg,
        "cheng_miss_split": miss,
        "A1_phylop_by_label": {
            k: {
                "n_variants": v["phylop"]["n_variants"],
                "n_residues": v["phylop"]["n_residues"],
                "logit_or_clustered": v["phylop"]["logit_or_clustered"],
                "gee": v["phylop"]["gee"],
                "mantel_haenszel": v["phylop"]["mantel_haenszel"],
            }
            for k, v in phy.items()
        },
        "section22_promotion": {
            "promote_to_results": False,
            "reason": (
                "C-group lock: not a paper result. SPT is classify_sota (multi-taxon soluble "
                "prefixes), not OCT1 Extracellular/Cytoplasmic. Loss is ProteinGym DMS_score_bin==0, "
                "not GFP<=-0.814. Assays mix activity/binding/abundance. SLC22 family still OCT1 only. "
                "AFDB v6 rerun is SI robustness; if later promoted, footnote those two definition gaps."
            ),
        },
        "P283L_citation_draft": {
            "design_safe": True,
            "design_positions": [61, 88, 401, 420, 465],
            "gfp": -1.195048211719825,
            "am": 0.3286,
            "sentence_en": (
                "P283L is not a design-set residue (locked design positions are 61, 88, 401, 420 and 465) "
                "and remains in the validation table. In HEK293 cells P283L abolished TEA uptake while "
                "the protein was still detected at the plasma membrane (Takeuchi et al., Drug Metab "
                "Pharmacokinet 2003;18:409). Subsequent cellular assays reported preserved metformin "
                "uptake for this substitution (reviewed in Arimany-Nardi et al., 2013); we therefore do "
                "not interpret P283L as a metformin loss-of-function allele. Independently, P283L is "
                "GFP-loss (score −1.20; cutoff −0.814) and AlphaMissense-benign (0.329) in the abundance "
                "scan, so an AM-benign call is not evidence of preserved transporter function."
            ),
        },
    }
    dump(payload, ADD3 / "ms1_feedback2_addendum3.json")
    dump(payload, MS1 / "ms1_feedback2_addendum3.json")
    print("wrote", ADD3)


if __name__ == "__main__":
    main()
