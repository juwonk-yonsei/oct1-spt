#!/usr/bin/env python3
"""I1 prep: R5 panel → FASTA, blind clone sheet, empty results template.

Does not pick I2 names. Does not start C8. Does not retune SPT.

    source met_env.sh && $MET_PY met_i1_prep.py
"""
from __future__ import annotations

import json
import os
import random
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MET_HDD = Path(os.environ.get("MET_HDD", str(Path(__file__).resolve().parent / "data")))
MET_SEQ = Path(os.environ.get("MET_SEQ", str(MET_HDD / "sequences")))
R5 = MET_HDD / "challenge" / "r_residual" / "r5_experiment_panel.tsv"
OUT = MET_HDD / "challenge" / "i_instead"
FASTA_DIR = MET_SEQ / "oct1_i1"
CDS_PATH = OUT / "SLC22A1_ENST00000366963_cds.fa"
SEED = 20260818

PRED = {
    "abundance_loss_resid_ok": "Stab",
    "exposed_am_benign_gfp_loss": "Stab",
    "dms_resid_loss_gfp_ok": "Trans",
    "near_wt_control": "WT",
    "literature_exposed_loss": "report_only",
}

# Human-preferred codons (first = most used). Designer minimizes mismatches vs WT codon.
AA_CODONS = {
    "A": ["GCC", "GCT", "GCA", "GCG"],
    "C": ["TGC", "TGT"],
    "D": ["GAC", "GAT"],
    "E": ["GAG", "GAA"],
    "F": ["TTC", "TTT"],
    "G": ["GGC", "GGA", "GGG", "GGT"],
    "H": ["CAC", "CAT"],
    "I": ["ATC", "ATT", "ATA"],
    "K": ["AAG", "AAA"],
    "L": ["CTG", "CTC", "TTG", "CTT", "CTA", "TTA"],
    "M": ["ATG"],
    "N": ["AAC", "AAT"],
    "P": ["CCC", "CCT", "CCA", "CCG"],
    "Q": ["CAG", "CAA"],
    "R": ["CGG", "AGA", "CGA", "CGC", "CGT", "AGG"],
    "S": ["AGC", "TCC", "TCT", "TCA", "AGT", "TCG"],
    "T": ["ACC", "ACA", "ACT", "ACG"],
    "V": ["GTG", "GTC", "GTT", "GTA"],
    "W": ["TGG"],
    "Y": ["TAC", "TAT"],
}

CODON = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "TCT": "S", "TCC": "S",
    "TCA": "S", "TCG": "S", "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CTT": "L", "CTC": "L",
    "CTA": "L", "CTG": "L", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "CGT": "R", "CGC": "R",
    "CGA": "R", "CGG": "R", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "GCT": "A", "GCC": "A",
    "GCA": "A", "GCG": "A", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def read_fasta(path: Path) -> str:
    return "".join(l.strip() for l in path.read_text().splitlines() if not l.startswith(">"))


def write_fasta(path: Path, header: str, seq: str, width: int = 60) -> None:
    body = "\n".join(seq[i : i + width] for i in range(0, len(seq), width))
    path.write_text(f">{header}\n{body}\n")


def tm_quik(n: int, gc_frac: float, mismatch_frac: float) -> float:
    return 81.5 + 0.41 * (gc_frac * 100) - 675 / n - mismatch_frac * 100


def pick_mut_codon(wt_codon: str, mut_aa: str) -> str:
    best, best_d = None, 99
    for c in AA_CODONS[mut_aa]:
        d = sum(a != b for a, b in zip(wt_codon, c))
        if d < best_d:
            best, best_d = c, d
    return best


def primer_for(cds: str, pos: int, mut_aa: str) -> dict:
    i = (pos - 1) * 3
    wt_codon = cds[i : i + 3]
    mut_codon = pick_mut_codon(wt_codon, mut_aa)
    # grow flanks until Tm >= 78 or max 45 nt
    for flank in range(12, 22):
        start = max(0, i - flank)
        end = min(len(cds), i + 3 + flank)
        sense = cds[start:i] + mut_codon + cds[i + 3 : end]
        n = len(sense)
        mm = sum(a != b for a, b in zip(sense, cds[start:end])) / n
        gc = sum(b in "GC" for b in sense) / n
        tm = tm_quik(n, gc, mm)
        if tm >= 78 and n >= 25:
            break
    anti = "".join({"A": "T", "T": "A", "G": "C", "C": "G"}[b] for b in sense[::-1])
    return {
        "wt_codon": wt_codon,
        "mut_codon": mut_codon,
        "n_mismatch": int(sum(a != b for a, b in zip(wt_codon, mut_codon))),
        "sense": sense,
        "antisense": anti,
        "len": len(sense),
        "tm_est": round(tm, 1),
        "gc": round(gc, 3),
    }


def fetch_cds(wt_aa: str) -> str:
    url = "https://rest.ensembl.org/sequence/id/ENST00000366963?type=cds"
    req = urllib.request.Request(url, headers={"Content-Type": "text/x-fasta"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    CDS_PATH.write_text(text)
    cds = "".join(l.strip() for l in text.splitlines() if not l.startswith(">"))
    cds = cds.upper().replace("U", "T")
    if len(cds) % 3:
        raise SystemExit(f"CDS length {len(cds)} not divisible by 3")
    aa = "".join(CODON[cds[i : i + 3]] for i in range(0, len(cds), 3)).rstrip("*")
    if aa != wt_aa:
        raise SystemExit("Ensembl CDS does not translate to locked OCT1 WT")
    return cds


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FASTA_DIR.mkdir(parents=True, exist_ok=True)

    pan = pd.read_csv(R5, sep="\t")
    wt = read_fasta(MET_SEQ / "SLC22A1_O15245.fasta")
    cds = fetch_cds(wt)

    write_fasta(FASTA_DIR / "SLC22A1_WT.fasta", "SLC22A1_WT", wt)

    primer_rows = []
    fasta_ok = []
    for rec in pan.to_dict("records"):
        hgvs = rec["hgvs_short"]
        pos, wt_aa, mut_aa = int(rec["pos"]), rec["wt_aa"], rec["mut_aa"]
        if not (1 <= pos <= len(wt)):
            raise SystemExit(f"{hgvs}: pos out of range")
        if wt[pos - 1] != wt_aa:
            raise SystemExit(f"{hgvs}: FASTA has {wt[pos - 1]} not {wt_aa}")
        mut_seq = wt[: pos - 1] + mut_aa + wt[pos:]
        write_fasta(FASTA_DIR / f"SLC22A1_{hgvs}.fasta", f"SLC22A1_{hgvs}", mut_seq)
        fasta_ok.append(hgvs)
        row = {
            "hgvs_short": hgvs,
            "pos": pos,
            "wt_aa": wt_aa,
            "mut_aa": mut_aa,
        }
        if cds and len(cds) >= pos * 3:
            row.update(primer_for(cds, pos, mut_aa))
        primer_rows.append(row)

    rng = random.Random(SEED)
    order = list(pan["hgvs_short"])
    rng.shuffle(order)
    key_rows = []
    for i, hgvs in enumerate(order, 1):
        rec = pan.loc[pan["hgvs_short"] == hgvs].iloc[0]
        cls = rec["panel_class"]
        key_rows.append({
            "clone_id": f"I1-{i:02d}",
            "hgvs_short": hgvs,
            "panel_class": cls,
            "predicted_type": PRED[cls],
            "spt_class": rec["spt_class"],
            "am_pathogenicity": rec["am_pathogenicity"],
            "am_stop_calls_WT": bool(rec["am_pathogenicity"] < 0.34),
            "in_pass_counts": cls != "literature_exposed_loss",
        })
    key = pd.DataFrame(key_rows)
    key.to_csv(OUT / "i1_clone_key.tsv", sep="\t", index=False)

    blind = key[["clone_id"]].copy()
    blind["construct"] = "OCT1_missense"
    blind["notes"] = "assay identical to WT; do not unblind until scoring"
    blind.to_csv(OUT / "i1_plate_blind.tsv", sep="\t", index=False)

    tmpl = []
    for cid, nrep in [("WT", 6), ("EMPTY", 3)]:
        for r in range(1, nrep + 1):
            tmpl.append({"clone_id": cid, "replicate": r, "surface": "", "uptake": "", "notes": ""})
    for cid in key["clone_id"]:
        for r in range(1, 4):
            tmpl.append({"clone_id": cid, "replicate": r, "surface": "", "uptake": "", "notes": ""})
    pd.DataFrame(tmpl).to_csv(OUT / "i1_results_template.tsv", sep="\t", index=False)

    primers = pd.DataFrame(primer_rows)
    primers.to_csv(OUT / "i1_primers.tsv", sep="\t", index=False)

    status = {
        "track": "instead",
        "prereg": "met_prereg_instead.md",
        "current": "I1",
        "status": "awaiting_wetlab",
        "seed_blind": SEED,
        "n_panel": int(len(pan)),
        "n_discovery": int((pan["panel_class"] != "literature_exposed_loss").sum()),
        "fasta_dir": str(FASTA_DIR),
        "hgvs": fasta_ok,
        "predicted_type": PRED,
        "i2_names": "not_picked_until_I1_go",
        "started_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "wetlab": "not_run",
        "note": "Fill i1_results.tsv (copy of template) then run met_i1_score.py. Do not drop clones.",
    }
    (OUT / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    (OUT / "i1_verdict.json").write_text(json.dumps({
        "status": "awaiting_wetlab",
        "pass": None,
        "gates": {k: None for k in ["I1.1", "I1.2", "I1.3", "I1.4"]},
        "n_scored": 0,
    }, indent=2) + "\n")

    print(f"wrote {len(fasta_ok)} mutant FASTAs + WT -> {FASTA_DIR}")
    print(f"blind key seed={SEED} -> {OUT / 'i1_clone_key.tsv'}")
    print("send to lab: i1_plate_blind.tsv + i1_results_template.tsv + collaborator_protocol.md")
    print("do not send: i1_clone_key.tsv (predicted types)")
    print("I2 names not picked")


if __name__ == "__main__":
    main()
