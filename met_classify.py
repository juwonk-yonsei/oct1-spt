#!/usr/bin/env python3
"""Structure-Position Triage (SPT) classifier.

Locked rule (met_prereg.md, 2026-08-12) — do not change thresholds after the fact:

    CORE     rel.SASA < 10%
    EXPOSED  rel.SASA > 30% AND topology in {Extracellular, Cytoplasmic}
    GREY     otherwise

    $MET_PY met_classify.py              # AF2 WT + 5-model reproducibility + 8SC1 compare
    $MET_PY met_classify.py --no-exp     # AF2 only (if PDB not downloaded yet)

Design-set variants (R61C, C88R, G401S, M420del, G465R) are rule-development
examples and must be excluded from WP3/WP4 validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

warnings.filterwarnings("ignore")

MET_HDD = Path(os.environ.get("MET_HDD", str(Path(__file__).resolve().parent / "data")))
MET_STRUCT = Path(os.environ.get("MET_STRUCT", str(MET_HDD / "structures")))
MET_SEQ = Path(os.environ.get("MET_SEQ", str(MET_HDD / "sequences")))
MET_PDB = Path(os.environ.get("MET_PDB", str(MET_HDD / "pdb")))
MET_SPT = Path(os.environ.get("MET_SPT", str(MET_HDD / "spt")))

CORE_CUTOFF = 10.0      # rel.SASA %  — LOCKED
EXPOSED_CUTOFF = 30.0   # rel.SASA %  — LOCKED
SOLUBLE = {"Extracellular", "Cytoplasmic"}

# Tien et al. 2013 theoretical maximum ASA (A^2)
MAX_ASA = {
    "ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167, "GLN": 225,
    "GLU": 223, "GLY": 104, "HIS": 224, "ILE": 197, "LEU": 201, "LYS": 236,
    "MET": 224, "PHE": 240, "PRO": 159, "SER": 155, "THR": 172, "TRP": 285,
    "TYR": 263, "VAL": 174,
}
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

DESIGN = {
    "R61C": 61, "C88R": 88, "G401S": 401, "M420del": 420, "G465R": 465,
}
DESIGN_EXPECTED = {
    61: "EXPOSED", 88: "GREY", 401: "CORE", 420: "CORE", 465: "CORE",
}

# 8SC1 missing stretches (REMARK 465) + ±2 neighbours excluded from AF2-vs-exp SASA compare
EXP_MISSING = [(1, 18), (280, 330), (516, 554)]
DISULFIDES_8SC1 = [(50, 121), (62, 102), (88, 142)]  # SSBOND records

WT_NAME = "SLC22A1_WT"
UNIPROT_ACC = "O15245"


def classify(rel_sasa: float, topology: str) -> str:
    if rel_sasa < CORE_CUTOFF:
        return "CORE"
    if rel_sasa > EXPOSED_CUTOFF and topology in SOLUBLE:
        return "EXPOSED"
    return "GREY"


def load_topology(path: Path) -> dict[int, str]:
    """pos -> 'Transmembrane' | 'Extracellular' | 'Cytoplasmic' | 'Unknown'."""
    data = json.loads(path.read_text())
    topo: dict[int, str] = {}
    for feat in data.get("features", []):
        ftype = feat.get("type")
        loc = feat.get("location", {})
        start = loc.get("start", {}).get("value")
        end = loc.get("end", {}).get("value")
        if start is None or end is None:
            continue
        if ftype == "Transmembrane":
            label = "Transmembrane"
        elif ftype == "Topological domain":
            desc = (feat.get("description") or "").strip()
            if desc.startswith("Extracellular"):
                label = "Extracellular"
            elif desc.startswith("Cytoplasmic"):
                label = "Cytoplasmic"
            else:
                label = desc or "Unknown"
        else:
            continue
        for i in range(int(start), int(end) + 1):
            topo[i] = label
    return topo


def fetch_uniprot_json(acc: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    import urllib.request
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
    print(f"fetching UniProt {acc} ...")
    urllib.request.urlretrieve(url, dest)
    return dest


def load_chain(pdb_path: Path, chain_id: str | None = None):
    struct = PDBParser(QUIET=True).get_structure(pdb_path.stem, str(pdb_path))
    model = next(struct.get_models())
    if chain_id is None:
        chains = list(model)
        if not chains:
            raise ValueError(f"no chains in {pdb_path}")
        return chains[0]
    return model[chain_id]


def residue_map(chain) -> dict[int, object]:
    out = {}
    for res in chain:
        het, seq, icode = res.id
        if het != " " or icode != " ":
            continue
        if "CA" not in res:
            continue
        out[int(seq)] = res
    return out


def rel_sasa_map(chain) -> dict[int, float]:
    ShrakeRupley().compute(chain, level="R")
    out = {}
    for pos, res in residue_map(chain).items():
        max_asa = MAX_ASA.get(res.get_resname(), 200)
        out[pos] = float(res.sasa) / max_asa * 100.0
    return out


def plddt_map(chain) -> dict[int, float]:
    # AF2 stores pLDDT in the CA B-factor column.
    return {pos: float(res["CA"].get_bfactor()) for pos, res in residue_map(chain).items()}


def sg_distance(chain, a: int, b: int) -> float | None:
    rm = residue_map(chain)
    if a not in rm or b not in rm:
        return None
    ra, rb = rm[a], rm[b]
    if "SG" not in ra or "SG" not in rb:
        return None
    return float(np.linalg.norm(ra["SG"].coord - rb["SG"].coord))


def exp_exclude_set() -> set[int]:
    """Residues missing in 8SC1, plus ±2 neighbours (SASA artefact at chain breaks)."""
    bad: set[int] = set()
    for a, b in EXP_MISSING:
        bad.update(range(a, b + 1))
        bad.update(range(max(1, a - 2), a))
        bad.update(range(b + 1, min(554, b + 2) + 1))
    return bad


def latest_oct1_dir() -> Path:
    hits = sorted(MET_STRUCT.glob("oct1_variants_*"))
    if not hits:
        sys.exit(f"error: no oct1_variants_* under {MET_STRUCT}")
    return hits[-1]


def wt_models(outdir: Path) -> list[Path]:
    hits = sorted(outdir.glob(f"{WT_NAME}_unrelaxed_rank_*.pdb"))
    if not hits:
        sys.exit(f"error: no {WT_NAME} unrelaxed PDBs in {outdir}")
    return hits


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore", delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def classify_structure(pdb_path: Path, topology: dict[int, str], chain_id: str | None = None):
    chain = load_chain(pdb_path, chain_id)
    sasa = rel_sasa_map(chain)
    plddt = plddt_map(chain)
    aa = residue_map(chain)
    rows = []
    for pos in sorted(sasa):
        resname = aa[pos].get_resname()
        top = topology.get(pos, "Unknown")
        rsa = sasa[pos]
        klass = classify(rsa, top)
        rows.append({
            "pos": pos,
            "aa": THREE_TO_ONE.get(resname, "X"),
            "resname": resname,
            "topology": top,
            "rel_sasa": round(rsa, 3),
            "plddt": round(plddt.get(pos, float("nan")), 2),
            "class": klass,
            "source": pdb_path.stem,
        })
    return chain, rows


def print_counts(label: str, rows: list[dict]) -> Counter:
    c = Counter(r["class"] for r in rows)
    n = sum(c.values())
    print(f"\n=== {label}  n={n} ===")
    for k in ("CORE", "EXPOSED", "GREY"):
        print(f"  {k:8s} {c[k]:4d}  ({100 * c[k] / n:5.1f}%)")
    return c


def design_check(rows: list[dict], label: str) -> bool:
    by_pos = {r["pos"]: r for r in rows}
    print(f"\n=== design-set check ({label}) ===")
    print(f"{'site':10s} {'topo':16s} {'rel.SASA':>8s} {'class':>8s} {'expected':>10s}  ok?")
    ok = True
    for name, pos in DESIGN.items():
        r = by_pos.get(pos)
        if r is None:
            print(f"{name:10s} {'MISSING':16s}")
            ok = False
            continue
        exp = DESIGN_EXPECTED[pos]
        match = r["class"] == exp
        ok = ok and match
        print(f"{name:10s} {r['topology']:16s} {r['rel_sasa']:7.1f}% {r['class']:>8s} "
              f"{exp:>10s}  {'OK' if match else 'MISMATCH'}")
    return ok


def reproducibility(model_paths: list[Path], topology: dict[int, str]) -> tuple[list[dict], dict]:
    """Classify every WT AF2 model; report per-residue flip rate."""
    per_model = []
    for p in model_paths:
        _, rows = classify_structure(p, topology)
        per_model.append((p.name, {r["pos"]: r["class"] for r in rows}))

    all_pos = sorted(set().union(*(d.keys() for _, d in per_model)))
    summary = []
    n_flip = 0
    for pos in all_pos:
        labels = [d.get(pos) for _, d in per_model]
        labels_ok = [x for x in labels if x]
        votes = Counter(labels_ok)
        majority, maj_n = votes.most_common(1)[0] if votes else ("NA", 0)
        unanimous = len(votes) == 1
        if not unanimous:
            n_flip += 1
        summary.append({
            "pos": pos,
            "topology": topology.get(pos, "Unknown"),
            "majority": majority,
            "n_models": len(labels_ok),
            "n_majority": maj_n,
            "unanimous": int(unanimous),
            "classes": ",".join(f"{k}:{v}" for k, v in votes.most_common()),
        })

    n = len(all_pos)
    print("\n=== reproducibility: AF2 WT models ===")
    print(f"  models: {len(model_paths)}")
    print(f"  residues classified in all/any: {n}")
    print(f"  unanimous: {n - n_flip}/{n}  ({100 * (n - n_flip) / n:.1f}%)")
    print(f"  flips:     {n_flip}/{n}  ({100 * n_flip / n:.1f}%)")

    print("\n  design sites across models:")
    print(f"  {'site':10s} " + " ".join(f"{'r'+str(i+1):>6s}" for i in range(len(model_paths)))
          + f"  {'maj':>8s}  unan?")
    for name, pos in DESIGN.items():
        labs = [d.get(pos, "?") for _, d in per_model]
        votes = Counter(l for l in labs if l != "?")
        maj = votes.most_common(1)[0][0] if votes else "?"
        unan = len(votes) == 1
        print(f"  {name:10s} " + " ".join(f"{l:>6s}" for l in labs)
              + f"  {maj:>8s}  {'yes' if unan else 'NO'}")

    return summary, {"n": n, "n_flip": n_flip, "n_models": len(model_paths)}


def compare_af2_exp(af2_rows: list[dict], exp_rows: list[dict]) -> list[dict]:
    exclude = exp_exclude_set()
    af2 = {r["pos"]: r for r in af2_rows}
    exp = {r["pos"]: r for r in exp_rows}
    common = sorted(set(af2) & set(exp) - exclude)
    rows = []
    agree = 0
    for pos in common:
        a, e = af2[pos], exp[pos]
        match = a["class"] == e["class"]
        agree += int(match)
        rows.append({
            "pos": pos,
            "aa": a["aa"],
            "topology": a["topology"],
            "af2_sasa": a["rel_sasa"],
            "exp_sasa": e["rel_sasa"],
            "af2_class": a["class"],
            "exp_class": e["class"],
            "agree": int(match),
        })
    n = len(common)
    print("\n=== AF2 rank-1 vs PDB 8SC1 ===")
    print(f"  common residues (excl. missing ±2): {n}")
    if n:
        print(f"  agreement: {agree}/{n}  ({100 * agree / n:.1f}%)")
        # per-class recall of AF2 against experimental labels
        by_exp = defaultdict(lambda: [0, 0])
        for r in rows:
            by_exp[r["exp_class"]][1] += 1
            by_exp[r["exp_class"]][0] += r["agree"]
        print(f"  {'exp class':10s} {'agree':>8s} {'n':>6s} {'pct':>7s}")
        for k in ("CORE", "EXPOSED", "GREY"):
            ok, tot = by_exp[k]
            pct = 100 * ok / tot if tot else float("nan")
            print(f"  {k:10s} {ok:8d} {tot:6d} {pct:6.1f}%")
    print(f"  8SC1 coverage vs 554: {len(exp)} residues present, "
          f"{len(exclude)} excluded (missing+neighbours)")

    print("\n  design sites:")
    print(f"  {'site':10s} {'AF2':>8s} {'8SC1':>8s}  agree?")
    for name, pos in DESIGN.items():
        if pos in exclude:
            print(f"  {name:10s} {'n/a':>8s} {'EXCLUDED':>8s}")
            continue
        a = af2.get(pos)
        e = exp.get(pos)
        if a is None or e is None:
            print(f"  {name:10s} {str(a['class'] if a else 'missing'):>8s} "
                  f"{str(e['class'] if e else 'missing'):>8s}")
            continue
        print(f"  {name:10s} {a['class']:>8s} {e['class']:>8s}  "
              f"{'yes' if a['class'] == e['class'] else 'NO'}")
    return rows


def disulfide_report(af2_chain, exp_chain) -> list[dict]:
    print("\n=== ECD disulfides (SG–SG, Å) ===")
    print(f"  {'pair':12s} {'AF2':>7s} {'8SC1':>7s}  (SSBOND ~2.03 Å)")
    rows = []
    for a, b in DISULFIDES_8SC1:
        d_af = sg_distance(af2_chain, a, b)
        d_ex = sg_distance(exp_chain, a, b) if exp_chain is not None else None
        print(f"  C{a}-C{b:<4d} {d_af if d_af is not None else float('nan'):7.2f} "
              f"{d_ex if d_ex is not None else float('nan'):7.2f}")
        rows.append({
            "cys_a": a, "cys_b": b,
            "af2_sg_sg": None if d_af is None else round(d_af, 3),
            "exp_sg_sg": None if d_ex is None else round(d_ex, 3),
        })
    return rows


def sensitivity(rows: list[dict]) -> list[dict]:
    """How many residues flip class if cutoffs move? Locked values stay 10/30."""
    print("\n=== threshold sensitivity (LOCKED remains 10 / 30) ===")
    base = {r["pos"]: r["class"] for r in rows}
    out = []
    cores = [5, 8, 10, 12, 15]
    expos = [25, 30, 35, 40]
    print(f"  {'core':>5s} {'exp':>5s} {'n_diff_vs_locked':>18s}  CORE EXPOSED GREY")
    for c in cores:
        for e in expos:
            n_diff = 0
            counts = Counter()
            for r in rows:
                rsa, top = r["rel_sasa"], r["topology"]
                if rsa < c:
                    k = "CORE"
                elif rsa > e and top in SOLUBLE:
                    k = "EXPOSED"
                else:
                    k = "GREY"
                counts[k] += 1
                if k != base[r["pos"]]:
                    n_diff += 1
            print(f"  {c:5.0f} {e:5.0f} {n_diff:18d}  "
                  f"{counts['CORE']:4d} {counts['EXPOSED']:7d} {counts['GREY']:4d}")
            out.append({
                "core_cutoff": c, "exposed_cutoff": e,
                "n_diff_vs_locked": n_diff,
                "n_CORE": counts["CORE"], "n_EXPOSED": counts["EXPOSED"],
                "n_GREY": counts["GREY"],
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-exp", action="store_true", help="skip 8SC1 comparison")
    ap.add_argument("--exp", default=str(MET_PDB / "8SC1.pdb"))
    args = ap.parse_args()

    MET_SPT.mkdir(parents=True, exist_ok=True)
    topo_json = fetch_uniprot_json(UNIPROT_ACC, MET_SEQ / f"{UNIPROT_ACC}_uniprot.json")
    topology = load_topology(topo_json)
    print(f"UniProt {UNIPROT_ACC} topology: {len(topology)} residues annotated")
    print(f"  CORE cutoff={CORE_CUTOFF}%   EXPOSED cutoff={EXPOSED_CUTOFF}%  (LOCKED)")

    outdir = latest_oct1_dir()
    models = wt_models(outdir)
    print(f"AF2 dir: {outdir.name}  ({len(models)} WT models)")

    rank1 = [p for p in models if "_rank_001_" in p.name]
    if not rank1:
        sys.exit("error: no rank_001 WT PDB")
    af2_chain, af2_rows = classify_structure(rank1[0], topology)
    print_counts("AF2 WT rank-1", af2_rows)
    design_ok = design_check(af2_rows, "AF2 rank-1")

    write_tsv(
        MET_SPT / "oct1_af2_rank1_spt.tsv",
        af2_rows,
        ["pos", "aa", "resname", "topology", "rel_sasa", "plddt", "class", "source"],
    )

    repro_rows, repro_stats = reproducibility(models, topology)
    write_tsv(
        MET_SPT / "oct1_af2_reproducibility.tsv",
        repro_rows,
        ["pos", "topology", "majority", "n_models", "n_majority", "unanimous", "classes"],
    )

    sens_rows = sensitivity(af2_rows)
    write_tsv(
        MET_SPT / "oct1_threshold_sensitivity.tsv",
        sens_rows,
        ["core_cutoff", "exposed_cutoff", "n_diff_vs_locked", "n_CORE", "n_EXPOSED", "n_GREY"],
    )

    exp_chain = None
    exp_rows = []
    compare_rows = []
    exp_path = Path(args.exp)
    if not args.no_exp and exp_path.exists():
        exp_chain, exp_rows = classify_structure(exp_path, topology, chain_id="A")
        print_counts("PDB 8SC1", exp_rows)
        design_check(exp_rows, "8SC1")
        write_tsv(
            MET_SPT / "oct1_8sc1_spt.tsv",
            exp_rows,
            ["pos", "aa", "resname", "topology", "rel_sasa", "plddt", "class", "source"],
        )
        compare_rows = compare_af2_exp(af2_rows, exp_rows)
        write_tsv(
            MET_SPT / "oct1_af2_vs_8sc1.tsv",
            compare_rows,
            ["pos", "aa", "topology", "af2_sasa", "exp_sasa", "af2_class", "exp_class", "agree"],
        )
    elif not args.no_exp:
        print(f"\n[skip] experimental PDB not found: {exp_path}")
        print("       run: ./met_download.sh pdb 8SC1 8SC4")

    ss_rows = disulfide_report(af2_chain, exp_chain)
    write_tsv(
        MET_SPT / "oct1_disulfides.tsv",
        ss_rows,
        ["cys_a", "cys_b", "af2_sg_sg", "exp_sg_sg"],
    )

    # compact summary json for the plan log
    agree_n = sum(r["agree"] for r in compare_rows) if compare_rows else None
    summary = {
        "core_cutoff": CORE_CUTOFF,
        "exposed_cutoff": EXPOSED_CUTOFF,
        "af2_dir": outdir.name,
        "n_af2_residues": len(af2_rows),
        "af2_counts": dict(Counter(r["class"] for r in af2_rows)),
        "design_rank1_ok": design_ok,
        "repro_unanimous_frac": (repro_stats["n"] - repro_stats["n_flip"]) / repro_stats["n"],
        "repro_n_flip": repro_stats["n_flip"],
        "n_8sc1_residues": len(exp_rows),
        "n_compared": len(compare_rows),
        "af2_8sc1_agree": agree_n,
        "af2_8sc1_agree_frac": (agree_n / len(compare_rows)) if compare_rows else None,
        "disulfides": ss_rows,
    }
    (MET_SPT / "oct1_spt_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\noutputs -> {MET_SPT}")
    if not design_ok:
        print("WARNING: design-set classification did not match pre-registered labels.",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
