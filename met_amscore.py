#!/usr/bin/env python3
"""Extract AlphaMissense pathogenicity scores for the project's target proteins.

AlphaMissense_aa_substitutions.tsv.gz is ~1.2 GB and keyed by UniProt accession,
so this does a SINGLE streaming pass and splits out every target at once rather
than re-scanning the file per gene.

    ./met_amscore.py extract              # one pass -> alphamissense/by_target/<GENE>_<ACC>.tsv
    ./met_amscore.py report SLC22A1 R61C C88R G401S M420del G465R
    ./met_amscore.py report SLC22A1       # top pathogenic variants for the gene

Note: AlphaMissense scores a variant's *pathogenicity*; it does not model the
structural change. Pair it with the AF2 WT/mutant structures and a ddG method.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

MET_AM = Path(os.environ.get("MET_AM", str(Path(__file__).resolve().parent / "data" / "alphamissense")))
MET_SEQ = Path(os.environ.get("MET_SEQ", str(Path(__file__).resolve().parent / "data" / "sequences")))
AA_FILE = MET_AM / "AlphaMissense_aa_substitutions.tsv.gz"
BY_TARGET = MET_AM / "by_target"


def load_targets():
    """gene -> accession, from the table met_targets.sh wrote."""
    table = MET_SEQ / "targets.tsv"
    if not table.exists():
        sys.exit(f"error: {table} not found - run ./met_targets.sh first")
    targets = {}
    with table.open() as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                targets[parts[0]] = parts[2]
    return targets


def cmd_extract():
    if not AA_FILE.exists():
        sys.exit(f"error: {AA_FILE} not found - run ./met_download.sh alphamissense")
    targets = load_targets()
    acc2gene = {acc: gene for gene, acc in targets.items()}
    BY_TARGET.mkdir(parents=True, exist_ok=True)

    handles = {}
    for acc, gene in acc2gene.items():
        fh = (BY_TARGET / f"{gene}_{acc}.tsv").open("w")
        fh.write("uniprot_id\tprotein_variant\tam_pathogenicity\tam_class\n")
        handles[acc] = fh

    counts = {acc: 0 for acc in acc2gene}
    print(f"scanning {AA_FILE.name} for {len(acc2gene)} accessions ...")

    # zcat is markedly faster than Python's gzip for a file this size.
    proc = subprocess.Popen(["zcat", str(AA_FILE)], stdout=subprocess.PIPE, text=True)
    seen = 0
    for line in proc.stdout:
        if line.startswith(("#", "uniprot_id")):
            continue
        acc = line[:line.index("\t")] if "\t" in line else ""
        fh = handles.get(acc)
        if fh is not None:
            fh.write(line)
            counts[acc] += 1
        seen += 1
        if seen % 20_000_000 == 0:
            print(f"  {seen:,} rows ...")
    proc.stdout.close()
    proc.wait()

    for fh in handles.values():
        fh.close()

    print(f"\ndone - {seen:,} rows scanned\n")
    print(f"{'gene':10s} {'accession':11s} {'variants':>9s}")
    for acc, gene in sorted(acc2gene.items(), key=lambda kv: kv[1]):
        flag = "" if counts[acc] else "   <- NOT FOUND"
        print(f"{gene:10s} {acc:11s} {counts[acc]:9,d}{flag}")
    print(f"\nper-target files -> {BY_TARGET}")


def cmd_report(gene, variants):
    targets = load_targets()
    if gene not in targets:
        sys.exit(f"error: {gene} is not in targets.tsv")
    acc = targets[gene]
    path = BY_TARGET / f"{gene}_{acc}.tsv"
    if not path.exists():
        sys.exit(f"error: {path} not found - run './met_amscore.py extract' first")

    rows = {}
    with path.open() as fh:
        next(fh)
        for line in fh:
            _, var, score, cls = line.rstrip("\n").split("\t")
            rows[var] = (float(score), cls)

    print(f"{gene} ({acc}) - {len(rows):,} scored substitutions\n")
    if variants:
        print(f"{'variant':10s} {'score':>7s}  class")
        for v in variants:
            if v in rows:
                s, c = rows[v]
                print(f"{v:10s} {s:7.4f}  {c}")
            else:
                # deletions and frameshifts are out of AlphaMissense's scope
                print(f"{v:10s} {'-':>7s}  not scored (AlphaMissense covers substitutions only)")
    else:
        top = sorted(rows.items(), key=lambda kv: -kv[1][0])[:20]
        print(f"top 20 most pathogenic:\n{'variant':10s} {'score':>7s}  class")
        for v, (s, c) in top:
            print(f"{v:10s} {s:7.4f}  {c}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract")
    rep = sub.add_parser("report")
    rep.add_argument("gene")
    rep.add_argument("variants", nargs="*")
    args = ap.parse_args()

    if args.cmd == "extract":
        cmd_extract()
    else:
        cmd_report(args.gene, args.variants)


if __name__ == "__main__":
    main()
