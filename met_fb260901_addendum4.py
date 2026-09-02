#!/usr/bin/env python3
"""Addendum-4: second-transporter replication (locked OCT1 rules) + reporting table.

Priority-1 of collaborator 260901-final: reanalyse one other SLC22 or transporter
DMS with the same SPT 10%/30% lock, UniProt Extracellular/Cytoplasmic soluble
domains, abundance-loss = synonymous/WT mean − 2 SD, AM pathogenic > 0.564.

ProteinGym DMS_score_bin is not used as the loss definition.

    source /SSD1T/PhD/AlphaFold/met_env.sh
    $MET_PY met_fb260901_addendum4.py
"""
from __future__ import annotations

import json
import os
import ssl
import subprocess
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/SSD1T/PhD/AlphaFold")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from met_classify import classify_structure, load_topology  # noqa: E402
from met_fb260901 import (  # noqa: E402
    AM_BENIGN,
    AM_PATH,
    MS1,
    N_BOOT,
    OUT,
    a1_model,
    classify_am_score,
    dump,
    grantham,
    recall_gap,
)
from met_fb260901_addendum2 import residue_genomic_map, ucsc_track  # noqa: E402

MET_HDD = Path(os.environ.get("MET_HDD", "/HDD8T1/WORK/Metformin_HDD"))
MET_AM = Path(os.environ.get("MET_AM", str(MET_HDD / "alphamissense")))
MET_SPT = Path(os.environ.get("MET_SPT", str(MET_HDD / "spt")))
ADD4 = OUT / "addendum4"
ADD4.mkdir(parents=True, exist_ok=True)

CTX = ssl._create_unverified_context()
AA20 = set("ARNDCQEGHILKMFPSTWYV")
MIN_READS = 10  # Young 2021 missing-data rule
SERT_ACC = "P31645"
SERT_TL = "ENSP00000261707"
SERT_LEN = 630
SERT_XLSX = ADD4 / "GSE109499_SERT_Publication.xlsx"
SERT_AM = ADD4 / "SLC6A4_P31645.tsv"
UNIPROT_JSON = Path("/HDD8T1/WORK/Metformin_HDD/challenge/c5_slcmap/cache/uniprot/P31645.json")
AFDB_PDB = Path("/HDD8T1/WORK/Metformin_HDD/spt/fb260901/addendum/afdb/AF-P31645-F1-model_v6.pdb")
AM_GZ = MET_AM / "AlphaMissense_aa_substitutions.tsv.gz"
# Young 2021 numbers a 24-aa myc+GS insertion after UniProt D216 as construct
# positions 217–240 (EQKLISEEDL plus linkers). Map those out before AM/SPT join.
YOUNG_INSERT = (217, 240)  # inclusive construct coordinates


def young_to_uniprot(pos: int) -> int | None:
    p = int(pos)
    lo, hi = YOUNG_INSERT
    if p < lo:
        return p
    if p <= hi:
        return None
    return p - (hi - lo + 1)


def mavedb_post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        "https://api.mavedb.org" + path,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "oct1-spt/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
        return json.loads(r.read().decode())


def search_mavedb() -> dict:
    cached = ADD4 / "mavedb_search.json"
    if cached.exists() and cached.stat().st_size > 100:
        return json.loads(cached.read_text())
    queries = [
        "SLC22A1", "OCT1", "SLC22A2", "OCT2", "SLC22", "SLCO1B1", "OATP1B1",
        "SLC6A4", "SERT", "SLC47", "MATE", "ABCB1", "CFTR", "NPC1",
        "transporter", "solute carrier", "CYP2C9", "Kir2.1",
    ]
    hits = {}
    for q in queries:
        try:
            res = mavedb_post("/api/v1/score-sets/search", {"text": q})
        except Exception as e:
            hits[q] = {"ok": False, "error": str(e)}
            continue
        items = res.get("scoreSets") or []
        hits[q] = {
            "ok": True,
            "n": int(res.get("numScoreSets") or len(items)),
            "urns": [
                {
                    "urn": s.get("urn"),
                    "title": s.get("title"),
                    "n": s.get("numVariants"),
                    "targets": [
                        t.get("name") for t in (s.get("targetGenes") or [])
                    ],
                }
                for s in items[:8]
            ],
        }
    return {
        "queries": hits,
        "human_slc_drug_transporter_score_sets": 0,
        "note": (
            "No MaveDB score-set for SLC22A1/OCT1, SLC22A2, SLCO1B1, SLC6A4, "
            "SLC47, ABCB1 or CFTR. NPC1 00001232 is LysoTracker essentiality "
            "(Erwood 2022), not surface-abundance GFP. CYP2C9 is VAMP-seq of "
            "an enzyme. mKir2.1 is an ion-channel surface-expression scan."
        ),
    }


def extract_am_p31645() -> Path:
    if SERT_AM.exists() and SERT_AM.stat().st_size > 1000:
        return SERT_AM
    if not AM_GZ.exists():
        raise FileNotFoundError(AM_GZ)
    print("extracting AlphaMissense P31645 from", AM_GZ.name)
    with SERT_AM.open("w") as out:
        out.write("uniprot_id\tprotein_variant\tam_pathogenicity\tam_class\n")
        proc = subprocess.Popen(["zcat", str(AM_GZ)], stdout=subprocess.PIPE, text=True)
        n = 0
        seen = 0
        for line in proc.stdout:
            if line.startswith(("#", "uniprot_id")):
                continue
            seen += 1
            if line.startswith("P31645\t"):
                out.write(line)
                n += 1
            if seen % 20_000_000 == 0:
                print(f"  scanned {seen:,} rows, kept {n:,}")
        proc.stdout.close()
        proc.wait()
    print(f"wrote {SERT_AM} n={n}")
    return SERT_AM


def parse_sert_xlsx(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Publication Data", header=None)
    # row0: NT headers; row1: # Seq AA Reads MYC1 MYC2 APP1 APP2
    pos = None
    wt = None
    rows = []
    for i in range(2, len(raw)):
        rec = raw.iloc[i]
        pos_cell, seq_cell, aa_cell, reads = rec[0], rec[1], rec[2], rec[3]
        if pd.notna(pos_cell) and str(pos_cell).strip() not in ("", "#"):
            try:
                pos = int(float(pos_cell))
            except (TypeError, ValueError):
                continue
        if pd.notna(seq_cell) and str(seq_cell).strip():
            wt = str(seq_cell).strip().upper()
        aa = None if pd.isna(aa_cell) else str(aa_cell).strip().upper()
        if pos is None or wt is None or not aa:
            continue
        is_wt_row = str(reads).strip().upper() == "WT" or aa == wt
        try:
            n_reads = np.nan if str(reads).strip().upper() == "WT" else float(reads)
        except (TypeError, ValueError):
            n_reads = np.nan

        def f(x):
            try:
                v = float(x)
                return v if np.isfinite(v) else np.nan
            except (TypeError, ValueError):
                return np.nan

        myc1, myc2, app1, app2 = f(rec[4]), f(rec[5]), f(rec[6]), f(rec[7])
        upos = young_to_uniprot(pos)
        rows.append({
            "pos_construct": pos,
            "pos": upos,
            "wt_aa": wt,
            "mut_aa": aa,
            "n_reads": n_reads,
            "is_wt_row": bool(is_wt_row and aa == wt),
            "is_stop": aa == "*",
            "is_insert": upos is None,
            "myc1": myc1,
            "myc2": myc2,
            "app1": app1,
            "app2": app2,
            "myc": np.nanmean([myc1, myc2]),
            "app": np.nanmean([app1, app2]),
        })
    df = pd.DataFrame(rows)
    df["variant"] = [
        f"{w}{int(p)}{m}" if pd.notna(p) else f"{w}{c}{m}"
        for w, p, m, c in zip(df["wt_aa"], df["pos"], df["mut_aa"], df["pos_construct"])
    ]
    return df


def loss_cutoffs(df: pd.DataFrame) -> dict:
    """Pre-specify cutoffs before recall. OCT1 formula: mean(syn) − 2 SD(syn)."""
    myc = df["myc"].to_numpy(dtype=float)
    both = df[["myc1", "myc2"]].to_numpy(dtype=float)
    miss = (
        (~df["is_wt_row"])
        & (~df["is_stop"])
        & df["mut_aa"].isin(AA20)
        & (df["n_reads"] >= MIN_READS)
        & np.isfinite(myc)
    )
    stops = df["is_stop"] & (df["n_reads"] >= MIN_READS) & np.isfinite(myc)
    wt = df["is_wt_row"] & np.isfinite(myc)

    syn_scores = myc[wt.to_numpy()]
    syn_unique = np.unique(np.round(syn_scores, 6)) if syn_scores.size else np.array([])
    degenerate = syn_scores.size == 0 or (syn_unique.size <= 1 and abs(float(syn_unique[0]) if syn_unique.size else 0) < 1e-12)

    # Technical SD of the two MYC replicates on scored missense (measurement-error analog
    # of synonymous SD when the table zeros WT rather than scanning synonymous codons).
    both_ok = miss.to_numpy() & np.isfinite(both[:, 0]) & np.isfinite(both[:, 1])
    diffs = (both[both_ok, 0] - both[both_ok, 1]) / np.sqrt(2.0)
    sd_tech = float(np.std(diffs, ddof=1)) if diffs.size >= 3 else np.nan
    cut_tech = 0.0 - 2.0 * sd_tech if np.isfinite(sd_tech) else np.nan

    stop_s = myc[stops.to_numpy()]
    cut_stop = float(np.median(stop_s)) if stop_s.size else np.nan

    return {
        "n_wt_rows": int(wt.sum()),
        "wt_mean": float(np.mean(syn_scores)) if syn_scores.size else np.nan,
        "wt_sd": float(np.std(syn_scores, ddof=1)) if syn_scores.size > 1 else 0.0,
        "synonymous_formula_degenerate": bool(degenerate),
        "n_missense_scored": int(miss.sum()),
        "n_stop_scored": int(stops.sum()),
        "n_tech_pairs": int(both_ok.sum()),
        "sd_tech": sd_tech,
        "cut_tech": cut_tech,
        "cut_stop_median": cut_stop,
        "stop_mean": float(np.mean(stop_s)) if stop_s.size else np.nan,
        "min_reads": MIN_READS,
        "primary_cutoff": "cut_tech",
        "primary_rule": (
            "Young 2021 zeros WT (enrichment 0) and does not report synonymous-codon "
            "scores. Identical arithmetic to OCT1 GFP (mean − 2 SD) with WT mean = 0 "
            "and SD = technical replicate SD of MYC1/MYC2."
        ),
    }


def classify_sert_spt() -> pd.DataFrame:
    dest = ADD4 / "SLC6A4_AFDB_v6_spt_oct1lock.tsv"
    if dest.exists() and dest.stat().st_size > 100:
        spt = pd.read_csv(dest, sep="\t")
        print("SPT cache", dict(Counter(spt["class"])))
        return spt
    topo = load_topology(UNIPROT_JSON)
    _, rows = classify_structure(AFDB_PDB, topo)
    spt = pd.DataFrame(rows)
    spt.to_csv(ADD4 / "SLC6A4_AFDB_v6_spt_oct1lock.tsv", sep="\t", index=False)
    print("SPT counts", dict(Counter(spt["class"])))
    return spt


def phylop_sert() -> pd.DataFrame:
    cached = ADD4 / "SLC6A4_phylop.tsv"
    if cached.exists() and cached.stat().st_size > 100:
        cons = pd.read_csv(cached, sep="\t")
        print("phyloP cache", cached, "coverage", float(cons["cons_phylop"].notna().mean()))
        return cons
    print("=== SERT codon map + phyloP100way ===")
    cmap = residue_genomic_map(SERT_TL, SERT_LEN)
    cmap.to_csv(ADD4 / "SLC6A4_codon_map.tsv", sep="\t", index=False)
    chrom = str(cmap["chrom"].iloc[0])
    g0 = int(cmap["g_start"].min())
    g1 = int(cmap["g_end"].max())
    phylo = ucsc_track("phyloP100way", chrom, g0, g1)
    if not phylo:
        phylo = ucsc_track("phyloP100wayAll", chrom, g0, g1)
    rows = []
    for rec in cmap.to_dict("records"):
        vals = [phylo[p] for p in rec["codon_nt"] if p in phylo]
        rows.append({
            "pos": int(rec["pos"]),
            "cons_phylop": float(np.mean(vals)) if vals else np.nan,
            "n_nt_phylop": len(vals),
        })
    cons = pd.DataFrame(rows)
    cons.to_csv(ADD4 / "SLC6A4_phylop.tsv", sep="\t", index=False)
    print("phyloP coverage", float(cons["cons_phylop"].notna().mean()))
    return cons


def pack_recall(rg: dict) -> dict:
    return {
        "recall_core": rg["recall_core"],
        "recall_exposed": rg["recall_exposed"],
        "n_loss_core": rg["n_loss_core"],
        "n_loss_exposed": rg["n_loss_exposed"],
        "or_variant": rg["or_variant"],
        "or_clustered": rg["or_clustered"],
        "ci_excludes_1": bool(
            rg["or_clustered"]["ci95"][0] > 1 or rg["or_clustered"]["ci95"][1] < 1
        ),
    }


def reporting_table() -> list[dict]:
    """Operational PGx reporting rules. Cost 0. Not a star-allele catalog."""
    return [
        {
            "spt": "EXPOSED",
            "am": "benign (<0.34)",
            "grade": "functional validation recommended",
            "action": (
                "Do not treat AM-benign as evidence of preserved transporter function. "
                "Recommend HEK (or equivalent) uptake/surface assay before using the "
                "variant as a functionally conserved allele in PGx reporting."
            ),
            "anchor": "OCT1 GFP-loss EXPOSED AM-benign set (n=17 in the manuscript); R61C is the clinical example (AM 0.268, still benign after Youden 0.479).",
        },
        {
            "spt": "EXPOSED",
            "am": "ambiguous (0.34–0.564)",
            "grade": "functional validation recommended",
            "action": "AM is uninformative. Prioritise functional assay if the variant is observed or is a candidate PGx allele.",
            "anchor": "Cheng miss includes 11 EXPOSED ambiguous; R287W is SAS-common and AM-ambiguous with no dedicated uptake paper.",
        },
        {
            "spt": "EXPOSED",
            "am": "pathogenic (>0.564)",
            "grade": "reduced-function candidate; confirm if a clinical decision depends on it",
            "action": "AM and SPT are directionally concordant with abundance loss more often than in EXPOSED-benign, but AM is still not a substitute for a star allele.",
            "anchor": "EXPOSED GFP-loss AM-pathogenic recall remains below CORE (AF2 41.7% vs 78.3%).",
        },
        {
            "spt": "CORE",
            "am": "pathogenic (>0.564)",
            "grade": "concordant abundance-loss support",
            "action": "AM pathogenic at a buried site is the setting where ClinVar-calibrated AM tracks OCT1 GFP-loss. Still not a PharmVar allele.",
            "anchor": "CORE GFP-loss recall 78.3% (AF2).",
        },
        {
            "spt": "CORE",
            "am": "benign (<0.34)",
            "grade": "discordant; inspect assay and haplotype",
            "action": "Uncommon among GFP-loss CORE. Do not override a known reduced-function haplotype with an AM-benign singleton call.",
            "anchor": "Haplotype context (e.g. R61C+M420del) is outside AM's variant-level score.",
        },
        {
            "spt": "GREY",
            "am": "any",
            "grade": "no SPT-informed AM override",
            "action": "Do not apply the CORE/EXPOSED recall correction. Report AM as-is and recommend function if the allele is PGx-actionable.",
            "anchor": "P283L is GREY, GFP-loss, AM-benign; TEA LoF with membrane expression and preserved metformin uptake in later assays.",
        },
    ]


def journal_rec() -> dict:
    return {
        "identity": (
            "This manuscript is a variant-effect predictor calibration paper "
            "(AM vs transporter abundance DMS, SPT as a structural prior), "
            "not a PGx discovery paper."
        ),
        "submit_to": ["PLOS Computational Biology", "Human Genomics"],
        "do_not_insist_on": "The Pharmacogenomics Journal / CPT / similar PGx venues as first target after a 2-week second-transporter attempt",
        "rationale": (
            "PGx journals will keep asking for wet-lab uptake of the EXPOSED AM-benign 17. "
            "That experiment changes the paper class and takes months. Computational Biology "
            "/ Human Genomics match the current evidence. Citations will come from the VEP "
            "and rare-variant interpretation community."
        ),
        "next_paper": (
            "The SPT lock and validation pipeline are the platform. Adding a second "
            "transporter (SERT if this addendum holds, or a future SLC22 DMS) is a "
            "separate, stronger manuscript. Do not delay the current paper for that expansion."
        ),
        "wetlab_not_this_cycle": (
            "Priority-2 HEK293 uptake of EXPOSED benign 17 is months, not this revision. "
            "It is the expansion-paper experiment, not a requirement for PLOS CB / Human Genomics."
        ),
    }


def main():
    print("=== MaveDB ===")
    mv = search_mavedb()
    dump(mv, ADD4 / "mavedb_search.json")
    slc_empty = [
        q for q in ("SLC22A1", "OCT1", "SLC22A2", "SLCO1B1", "SLC6A4", "SLC47", "ABCB1", "CFTR")
        if (mv["queries"].get(q) or {}).get("n", -1) == 0
    ]
    print("empty SLC/drug-transporter queries", slc_empty)

    print("=== AM P31645 ===")
    extract_am_p31645()
    am = pd.read_csv(SERT_AM, sep="\t")
    am["pos"] = am["protein_variant"].str.extract(r"^[A-Z](\d+)[A-Z]$").astype(float)
    am_map = dict(zip(am["protein_variant"], am["am_pathogenicity"]))

    print("=== SPT OCT1 lock on AFDB v6 ===")
    spt = classify_sert_spt()
    class_map = dict(zip(spt["pos"].astype(int), spt["class"]))

    print("=== Young 2021 GEO expression table ===")
    if not SERT_XLSX.exists():
        raise FileNotFoundError(SERT_XLSX)
    raw = parse_sert_xlsx(SERT_XLSX)
    raw.to_csv(ADD4 / "SLC6A4_Young2021_raw.tsv", sep="\t", index=False)
    cuts = loss_cutoffs(raw)
    print("loss cutoffs", {k: cuts[k] for k in (
        "synonymous_formula_degenerate", "n_missense_scored", "sd_tech",
        "cut_tech", "cut_stop_median", "n_stop_scored",
    )})

    uni_seq = json.loads(UNIPROT_JSON.read_text())["sequence"]["value"]
    mapped = raw[raw["pos"].notna()].copy()
    mapped["uni_aa"] = [uni_seq[int(p) - 1] for p in mapped["pos"]]
    n_aa_mismatch = int((mapped["wt_aa"] != mapped["uni_aa"]).sum())
    print("construct->UniProt AA mismatches (non-insert rows)", n_aa_mismatch)

    miss = raw[
        (~raw["is_wt_row"])
        & (~raw["is_stop"])
        & (~raw["is_insert"])
        & raw["pos"].notna()
        & raw["mut_aa"].isin(AA20)
        & (raw["n_reads"] >= MIN_READS)
        & np.isfinite(raw["myc"])
    ].copy()
    miss["pos"] = miss["pos"].astype(int)
    miss["am"] = miss["variant"].map(am_map)
    miss["am_class"] = [classify_am_score(s) for s in miss["am"]]
    miss["class"] = miss["pos"].map(class_map)
    miss["grantham"] = [grantham(w, m) for w, m in zip(miss["wt_aa"], miss["mut_aa"])]
    miss["dms_loss"] = miss["myc"] <= cuts["cut_tech"]
    miss["dms_loss_stop"] = miss["myc"] <= cuts["cut_stop_median"]
    miss.to_csv(ADD4 / "SLC6A4_Young2021_missense_locked.tsv", sep="\t", index=False)

    print("=== recall gap (MYC surface, tech-SD cutoff, OCT1 SPT, AM>0.564) ===")
    val = miss.dropna(subset=["am_class", "class"]).copy()
    rg = recall_gap(val, "class")
    print("recall CORE/EXPOSED", rg["recall_core"], rg["recall_exposed"])
    print("clustered OR", rg["or_clustered"])

    print("=== recall gap sensitivity: nonsense-median cutoff ===")
    val_s = val.copy()
    val_s["dms_loss"] = val_s["dms_loss_stop"]
    rg_stop = recall_gap(val_s, "class")
    print("stop-cut recall", rg_stop["recall_core"], rg_stop["recall_exposed"])

    print("=== phyloP A1 on SERT ===")
    a1 = None
    a1_err = None
    try:
        cons = phylop_sert()
        val2 = val.merge(cons, on="pos", how="left")
        a1 = a1_model(val2, "class", "cons_phylop", n_boot=N_BOOT)
        print("A1 phyloP clustered", a1["logit_or_clustered"])
        print("A1 GEE", a1["gee"])
    except Exception as e:
        a1_err = str(e)
        print("phyloP A1 failed:", e)

    n_ce = int(((val["dms_loss"]) & val["class"].isin(["CORE", "EXPOSED"])).sum())
    n_res = int(val.loc[val["dms_loss"] & val["class"].isin(["CORE", "EXPOSED"]), "pos"].nunique())

    success = bool(
        rg["or_clustered"]["ci95"][0] > 1
        and np.isfinite(rg["recall_core"])
        and np.isfinite(rg["recall_exposed"])
        and rg["recall_core"] > rg["recall_exposed"]
    )
    verdict = {
        "second_transporter": "SLC6A4 / SERT (Young 2021 GEO GSE109499)",
        "assay": "anti-myc surface expression (MYC1/MYC2), not ProteinGym APP+ activity bin",
        "spt": "OCT1 lock classify() on AFDB v6 + UniProt P31645 Extracellular/Cytoplasmic",
        "loss": cuts,
        "n_myc_loss_core_exposed": n_ce,
        "n_residues_core_exposed": n_res,
        "recall_techSD": pack_recall(rg),
        "recall_nonsense_median": pack_recall(rg_stop),
        "A1_phylop": a1,
        "A1_error": a1_err,
        "direction_matches_OCT1": bool(
            np.isfinite(rg["recall_core"]) and rg["recall_core"] > rg.get("recall_exposed", 0)
        ),
        "clustered_OR_excludes_1": pack_recall(rg)["ci_excludes_1"],
        "success_for_paper": success,
        "caveat": (
            "Not SLC22. Young zeros most WT rows instead of synonymous-codon GFP. "
            "Technical-SD cutoff is the same arithmetic as OCT1 mean−2SD with WT mean ≈ 0. "
            "Nonsense-median cutoff is a sensitivity, not the lock. Construct positions "
            "217–240 are a myc+GS insertion after UniProt D216 and were excluded; "
            "positions ≥241 were shifted −24 onto P31645 before AM/SPT join."
        ),
        "numbering": {
            "young_insert_inclusive": list(YOUNG_INSERT),
            "n_insert_rows": int(raw["is_insert"].sum()),
            "aa_mismatches_after_map": n_aa_mismatch,
            "am_join_rate": float(miss["am"].notna().mean()),
            "spt_join_rate": float(miss["class"].notna().mean()),
        },
    }

    payload = {
        "mavedb": mv,
        "empty_human_slc_drug_transporter_queries": slc_empty,
        "rejected": {
            "Zhang_SLCO1B1": {
                "why": "Landing-pad GFP but only 137 ExAC missense. Underpowered for locked CORE/EXPOSED recall + residue-clustered CI.",
                "pmid": "PMC8042483",
            },
            "NPC1_Erwood_MaveDB_00001232": {
                "why": "LysoTracker essentiality after saturation prime editing, not surface-abundance GFP.",
            },
            "CYP2C9_VAMPseq": {
                "why": "Abundance MAVE of a PGx enzyme, not a transporter. SPT Extracellular/Cytoplasmic lock does not apply to the same biology.",
            },
            "mKir2.1_DIMPLE": {
                "why": "Mouse Kir2.1 ion-channel surface expression, not a solute carrier.",
            },
            "ProteinGym_SC6A4_Young_2021": {
                "why": "DMS_score_bin is APP+ transport activity, not myc surface expression. Forbidden as the loss definition.",
            },
            "Yee_OCT1": {
                "why": "Only SLC22 saturation abundance+uptake DMS in the literature; not deposited in MaveDB; already the discovery set.",
            },
        },
        "sert_replication": verdict,
        "spt_counts": dict(Counter(spt["class"])),
        "reporting_table": reporting_table(),
        "journal": journal_rec(),
        "locks_held": {
            "spt_10_30": True,
            "soluble_Extracellular_Cytoplasmic": True,
            "am_pathogenic": AM_PATH,
            "am_benign": AM_BENIGN,
            "n_boot": N_BOOT,
            "seed": 20260812,
            "proteingym_bin_not_used": True,
        },
    }
    dump(payload, ADD4 / "ms1_feedback2_addendum4.json")
    dump(payload, MS1 / "ms1_feedback2_addendum4.json")
    rows = [{
        "cutoff": "techSD",
        "cut_value": cuts["cut_tech"],
        "recall_CORE": rg["recall_core"],
        "recall_EXPOSED": rg["recall_exposed"],
        "n_CORE": rg["n_loss_core"],
        "n_EXPOSED": rg["n_loss_exposed"],
        "OR_clustered_median": rg["or_clustered"]["median"],
        "OR_lo": rg["or_clustered"]["ci95"][0],
        "OR_hi": rg["or_clustered"]["ci95"][1],
        "excludes_1": pack_recall(rg)["ci_excludes_1"],
    }, {
        "cutoff": "nonsense_median",
        "cut_value": cuts["cut_stop_median"],
        "recall_CORE": rg_stop["recall_core"],
        "recall_EXPOSED": rg_stop["recall_exposed"],
        "n_CORE": rg_stop["n_loss_core"],
        "n_EXPOSED": rg_stop["n_loss_exposed"],
        "OR_clustered_median": rg_stop["or_clustered"]["median"],
        "OR_lo": rg_stop["or_clustered"]["ci95"][0],
        "OR_hi": rg_stop["or_clustered"]["ci95"][1],
        "excludes_1": pack_recall(rg_stop)["ci_excludes_1"],
    }]
    pd.DataFrame(rows).to_csv(ADD4 / "SLC6A4_recall_gap.tsv", sep="\t", index=False)
    print("wrote", ADD4)
    print("SUCCESS_FOR_PAPER", success)


if __name__ == "__main__":
    main()
