#!/usr/bin/env python3
"""Addendum-2: phyloP A1, Youden pooled, AFDB v6 SPT, ProteinGym AFDB, gnomAD OR.

    source /SSD1T/PhD/AlphaFold/met_env.sh
    $MET_PY met_fb260901_addendum2.py
"""
from __future__ import annotations

import json
import os
import ssl
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "/SSD1T/PhD/AlphaFold")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from met_classify import classify_structure, load_topology, write_tsv  # noqa: E402
from met_fb260901 import (  # noqa: E402
    AM_PATH,
    GFP_CUT,
    MS1,
    N_BOOT,
    OUT,
    a1_model,
    clustered_bootstrap,
    dump,
    fisher_or_table,
    grantham,
    grouped_indices,
    or_from_counts,
    p1_stats,
    p3_stats,
    p4_clustered,
    recall_gap,
    youden_threshold,
)
from met_pgym_spt import classify_sota, load_topology_sota  # noqa: E402

MET_HDD = Path(os.environ.get("MET_HDD", "/HDD8T1/WORK/Metformin_HDD"))
MET_SPT = Path(os.environ.get("MET_SPT", str(MET_HDD / "spt")))
MET_SEQ = Path(os.environ.get("MET_SEQ", str(MET_HDD / "sequences")))
MET_PGYM = MET_HDD / "proteingym"
ADD = OUT / "addendum"
ADD2 = OUT / "addendum2"
CTX = ssl._create_unverified_context()

PGYM7 = {
    "ADRB2_HUMAN": "P07550",
    "CCR5_HUMAN": "P51681",
    "HMDH_HUMAN": "P04035",
    "NPC1_HUMAN": "O15118",
    "SC6A4_HUMAN": "P31645",
    "VKOR1_HUMAN": "Q9BQB6",
}
PGYM_DMS = [
    "ADRB2_HUMAN_Jones_2020",
    "CCR5_HUMAN_Gill_2023",
    "HMDH_HUMAN_Jiang_2019",
    "NPC1_HUMAN_Erwood_2022_HEK293T",
    "SC6A4_HUMAN_Young_2021",
    "VKOR1_HUMAN_Chiasson_2020_abundance",
    "VKOR1_HUMAN_Chiasson_2020_activity",
]


def http_json(url: str, timeout=180):
    req = urllib.request.Request(
        url,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "oct1-spt-research/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return json.loads(r.read().decode())


def residue_genomic_map(tl_id="ENSP00000355930", length=554) -> pd.DataFrame:
    """Flatten exon-split codons: concatenate 1-based genomic nts, then groups of 3."""
    mp = http_json(
        f"https://rest.ensembl.org/map/translation/{tl_id}/1..{length}?content-type=application/json"
    )
    nts = []
    for seg in mp.get("mappings") or []:
        chrom = str(seg["seq_region_name"])
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"
        strand = int(seg["strand"])
        start, end = int(seg["start"]), int(seg["end"])
        if strand < 0:
            coords = list(range(end, start - 1, -1))
        else:
            coords = list(range(start, end + 1))
        for gpos in coords:
            nts.append((chrom, int(gpos), strand))
    if len(nts) != length * 3:
        print("WARNING nt count", len(nts), "expected", length * 3)
    rows = []
    n_aa = min(length, len(nts) // 3)
    for aa in range(n_aa):
        codon = nts[aa * 3:(aa + 1) * 3]
        gpos = [c[1] for c in codon]
        rows.append({
            "pos": aa + 1,
            "chrom": codon[0][0],
            "codon_nt": gpos,
            "strand": codon[0][2],
            "g_start": min(gpos),
            "g_end": max(gpos) + 1,
        })
    return pd.DataFrame(rows)


def ucsc_track(track: str, chrom: str, start: int, end: int, chunk=8000) -> dict[int, float]:
    out = {}
    for s in range(start, end, chunk):
        e = min(s + chunk, end)
        url = (
            f"https://api.genome.ucsc.edu/getData/track?genome=hg38"
            f"&track={track}&chrom={chrom}&start={s}&end={e}"
        )
        data = http_json(url, timeout=180)
        arr = data.get(track)
        if arr is None:
            for k, v in data.items():
                if k in ("dataTime", "dataTime_utc", "start", "end", "chrom", "genome", "track"):
                    continue
                arr = v
                break
        if isinstance(arr, dict):
            arr = arr.get(chrom) or arr.get("chr6") or []
        arr = arr or []
        for rec in arr:
            out[int(rec["start"])] = float(rec["value"])
        print(f"  {track} {s}-{e} n={len(arr)}")
    return out


def consurf_submit() -> dict:
    """Submit 8SC1 to consurf.tau.ac.il today. Do not wait for grades."""
    try:
        import requests
        import urllib3
        urllib3.disable_warnings()
    except ImportError:
        return {"ok": False, "error": "requests not installed"}
    pdb = Path(os.environ.get("MET_PDB", "/HDD8T1/WORK/Metformin_HDD/pdb")) / "8SC1.pdb"
    email = "junjeong@yonsei.ac.kr"
    s = requests.Session()
    s.verify = False
    try:
        home = s.get("https://consurf.tau.ac.il/", timeout=60)
        csrf = s.cookies.get("csrftoken") or ""
        files = {}
        data = {
            "MSAprogram": "MAFFT",
            "proteins_DB": "UNIREF90",
            "MAX_REDUNDANCY": "95",
            "MIN_IDENTITY": "35",
            "MAX_NUM_HOMOL": "150",
            "best_uniform_sequences": "sample",
            "ITERATIONS": "1",
            "E_VALUE": "0.0001",
            "SUB_MATRIX": "BEST",
            "Homolog_search_algorithm": "HMMER",
            "ALGORITHM": "Bayes",
            "user_select_seq": "no",
            "Run_Number": "",
            "DNA_AA": "AA",
            "pdb_ID": "8SC1",
            "JOB_TITLE": "OCT1_O15245_8SC1_ms1_A1",
            "user_email": email,
            "csrfmiddlewaretoken": csrf,
        }
        fh = None
        if pdb.exists():
            fh = pdb.open("rb")
            files["pdb_FILE"] = (pdb.name, fh, "chemical/x-pdb")
        r = s.post(
            "https://consurf.tau.ac.il/consurf/",
            data=data,
            files=files or None,
            headers={"X-CSRFToken": csrf, "Referer": "https://consurf.tau.ac.il/"},
            timeout=120,
        )
        if fh is not None:
            fh.close()
        info = {
            "ok": r.status_code in (200, 201, 202, 302),
            "status": r.status_code,
            "csrf_present": bool(csrf),
            "home_status": home.status_code,
            "email": email,
            "pdb_uploaded": pdb.exists(),
            "content_type": r.headers.get("Content-Type", ""),
        }
        try:
            js = r.json()
            info["json"] = js
            info["Run_Number"] = js.get("Run_Number")
            info["error"] = js.get("error")
            info["ok"] = bool(js.get("Run_Number")) and not js.get("error")
            if js.get("Run_Number"):
                info["progress_url"] = f"https://consurf.tau.ac.il/progress/?number={js.get('Run_Number')}"
        except Exception:
            info["text_head"] = r.text[:800]
        ADD2.mkdir(parents=True, exist_ok=True)
        (ADD2 / "consurf_submit.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
        print("ConSurf submit", info.get("Run_Number"), info.get("status"), info.get("error"))
        return info
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


LIT_P283_R287 = {
    "P283L": {
        "gnomad": "6-160136228-C-T",
        "hgvs": "P283L",
        "rsid": "rs4646277",
        "cDNA": "c.848C>T",
        "assay": True,
        "key_paper": "Takeuchi et al. DMPK 2003;18:409",
        "doi": "10.2133/dmpk.18.409",
        "finding": (
            "HEK293: TEA uptake abolished; protein still at plasma membrane "
            "(not a trafficking failure)."
        ),
        "metformin_caveat": (
            "Arimany-Nardi 2013 (PMC3747481): P283L reduces MPP+/TEA but does not "
            "affect metformin uptake in the cited cellular assays — substrate-dependent."
        ),
        "class_af2": "GREY",
        "am_cheng": "benign (0.3286)",
        "af_eas": 0.00608,
    },
    "R287W": {
        "gnomad": "6-160136239-C-T",
        "hgvs": "R287W",
        "rsid": "rs4646278 (C>T allele; literature R287G is C>G at the same codon)",
        "cDNA": "c.859C>T",
        "assay": False,
        "key_paper": None,
        "finding": (
            "No published OCT1 uptake assay for Arg287Trp. Codon-287 literature is Arg287Gly "
            "(Takeuchi 2003; rs4646278 C>G): TEA uptake abolished, membrane localization retained. "
            "Do not cite R287G as evidence for R287W."
        ),
        "class_af2": "GREY",
        "am_cheng": "ambiguous (0.4912)",
        "af_sas": 0.00841,
    },
}


def residue_conservation(cmap: pd.DataFrame, phylo: dict, phast: dict) -> pd.DataFrame:
    rows = []
    for _, r in cmap.iterrows():
        vals_p, vals_h = [], []
        for nt in r["codon_nt"]:
            key = int(nt) - 1
            if key in phylo:
                vals_p.append(phylo[key])
            if key in phast:
                vals_h.append(phast[key])
        rows.append({
            "pos": int(r["pos"]),
            "cons_phylop": float(np.mean(vals_p)) if vals_p else np.nan,
            "cons_phastcons": float(np.mean(vals_h)) if vals_h else np.nan,
            "n_nt_phylop": len(vals_p),
            "n_nt_phast": len(vals_h),
        })
    return pd.DataFrame(rows)


def pdb_ca(path: Path):
    xyz, bf = {}, {}
    with path.open() as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resi = int(line[22:26])
                xyz[resi] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                bf[resi] = float(line[60:66])
    return xyz, bf


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    return float(np.sqrt(((a @ r - b) ** 2).sum(1).mean()))


def rmsd_pair(p1: Path, p2: Path, keys=None, plddt_min=None):
    A, Ab = pdb_ca(p1)
    B, Bb = pdb_ca(p2)
    kk = sorted(set(A) & set(B))
    if keys is not None:
        kk = [k for k in kk if k in keys]
    if plddt_min is not None:
        kk = [k for k in kk if Ab.get(k, 0) >= plddt_min and Bb.get(k, 0) >= plddt_min]
    if len(kk) < 10:
        return {"n": len(kk), "rmsd": np.nan}
    a = np.stack([A[k] for k in kk])
    b = np.stack([B[k] for k in kk])
    return {"n": len(kk), "rmsd": kabsch_rmsd(a, b)}


def youden_pooled(val: pd.DataFrame):
    df = val.dropna(subset=["am_pathogenicity", "dms_loss", "cluster"]).copy()
    clusters = sorted(c for c in df["cluster"].dropna().unique())
    te_parts = []
    fold_t = []
    for cl in clusters:
        tr = df[df["cluster"] != cl]
        te = df[df["cluster"] == cl]
        if len(te) < 20 or len(tr) < 100:
            continue
        t, _j = youden_threshold(
            tr["am_pathogenicity"].to_numpy(dtype=float),
            tr["dms_loss"].astype(int).to_numpy(),
        )
        te = te.copy()
        te["yhat_youden"] = te["am_pathogenicity"] > t
        te["fold_t"] = t
        te_parts.append(te)
        fold_t.append(t)
    pooled = pd.concat(te_parts, ignore_index=True)
    out = {
        "n_folds": len(te_parts),
        "t_youden_median": float(np.median(fold_t)),
        "n_pooled": int(len(pooled)),
    }
    for lab in ("ALL", "CORE", "EXPOSED", "GREY"):
        sub = pooled if lab == "ALL" else pooled[pooled["class"] == lab]
        y = sub["dms_loss"].astype(int)
        pred = sub["yhat_youden"].astype(int)
        tp = int(((y == 1) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        out[lab] = {
            "n": int(len(sub)),
            "n_loss": int(y.sum()),
            "sens": tp / (tp + fn) if tp + fn else np.nan,
            "spec": tn / (tn + fp) if tn + fp else np.nan,
            "ppv": tp / (tp + fp) if tp + fp else np.nan,
        }
    xs = []
    for te in te_parts:
        sl = te[(te["class"] == "EXPOSED") & te["dms_loss"]]
        if len(sl):
            xs.append(float(sl["yhat_youden"].mean()))
    out["unweighted_fold_mean_EXPOSED_sens"] = float(np.mean(xs)) if xs else np.nan
    out["n_folds_with_exposed_loss"] = len(xs)
    return out


def gnomad_recall_or():
    loss = pd.read_csv(ADD / "gnomad_gfp_loss.tsv", sep="\t")
    g = loss[loss["class"].isin(["CORE", "EXPOSED"])].copy()
    g = g.dropna(subset=["am_class_cheng", "class", "pos"])
    g["path"] = g["am_class_cheng"] == "pathogenic"

    def rec(lab):
        s = g[g["class"] == lab]
        n = len(s)
        k = int(s["path"].sum())
        return n, k, k / n if n else np.nan

    nc, kc, rc = rec("CORE")
    ne, ke, re_ = rec("EXPOSED")
    a, b, c, d = kc, nc - kc, ke, ne - ke
    fish = fisher_or_table(a, b, c, d)
    g = g.reset_index(drop=True)
    groups = grouped_indices(g["pos"].to_numpy())
    is_core = (g["class"] == "CORE").to_numpy()
    ip = g["path"].to_numpy()

    def stat(idx):
        ic = is_core[idx]
        p = ip[idx]
        return or_from_counts(
            int((ic & p).sum()), int((ic & ~p).sum()),
            int((~ic & p).sum()), int((~ic & ~p).sum()),
        )

    boot = clustered_bootstrap(groups, stat)
    hand = (kc / b) / (ke / d) if min(b, ke, d) else np.nan
    return {
        "n_core": nc, "n_path_core": kc, "recall_core": rc,
        "n_exposed": ne, "n_path_exposed": ke, "recall_exposed": re_,
        "table": {"a": int(a), "b": int(b), "c": int(c), "d": int(d)},
        "or_variant_woolf": fish,
        "or_clustered": boot,
        "hand_or": float(hand),
        "user_quoted": "3.65 [1.6, 8.4]",
    }


def classify_afdb_oct1():
    pdb = ADD / "afdb" / "AF-O15245-F1-model_v6.pdb"
    topo = load_topology(MET_SEQ / "O15245_uniprot.json")
    _, rows = classify_structure(pdb, topo)
    write_tsv(
        ADD2 / "oct1_afdb_v6_spt.tsv",
        rows,
        ["pos", "aa", "resname", "topology", "rel_sasa", "plddt", "class", "source"],
    )
    return pd.DataFrame(rows), Counter(r["class"] for r in rows)


def proteingym_afdb():
    feat = pd.read_csv(MET_PGYM / "features" / "primary_variants.tsv", sep="\t")
    assays = pd.read_csv(MET_PGYM / "membrane_assays.tsv", sep="\t")
    feat = feat[feat["am_pathogenicity"].notna()].copy()
    feat["am_path"] = feat["am_pathogenicity"] > AM_PATH
    feat["loss"] = feat["DMS_score_bin"].astype(str).isin(["0", "0.0"])
    rows_out = []
    spt_dir = ADD2 / "pgym_afdb_spt"
    spt_dir.mkdir(exist_ok=True)
    for uid, acc in PGYM7.items():
        pdb = ADD / "afdb" / f"AF-{acc}-F1-model_v6.pdb"
        uj = MET_PGYM / "uniprot_json" / f"{uid}.json"
        if not pdb.exists() or not uj.exists():
            rows_out.append({"UniProt_ID": uid, "ok": False, "reason": "missing pdb/json"})
            continue
        data = json.loads(uj.read_text())
        topo = load_topology_sota(data)
        _, spt_rows = classify_structure(pdb, topo)
        for r in spt_rows:
            r["class"] = classify_sota(float(r["rel_sasa"]), str(r["topology"]))
        write_tsv(
            spt_dir / f"{uid}_spt.tsv",
            spt_rows,
            ["pos", "aa", "resname", "topology", "rel_sasa", "plddt", "class", "source"],
        )
        cmap = {int(r["pos"]): r["class"] for r in spt_rows}
        sub = feat[(feat["UniProt_ID"] == uid) & (feat["DMS_id"].isin(PGYM_DMS))].copy()
        for dms_id in sorted(sub["DMS_id"].unique()):
            ss = sub[sub["DMS_id"] == dms_id].copy()
            ss["class_afdb"] = ss["pos"].map(cmap)
            loss = ss[ss["loss"] & ss["class_afdb"].isin(["CORE", "EXPOSED"])].copy()

            def rec(lab):
                s = loss[loss["class_afdb"] == lab]
                n = len(s)
                k = int(s["am_path"].sum())
                return n, k, k / n if n else np.nan

            nc, kc, rc = rec("CORE")
            ne, ke, re_ = rec("EXPOSED")
            meta = assays[assays["DMS_id"] == dms_id]
            mol = str(meta["molecule_name"].iloc[0]) if len(meta) else uid
            item = {
                "DMS_id": dms_id, "UniProt_ID": uid, "molecule": mol, "ok": True,
                "n_core_res": int(sum(v == "CORE" for v in cmap.values())),
                "n_exposed_res": int(sum(v == "EXPOSED" for v in cmap.values())),
                "n_loss_core": nc, "recall_core": rc,
                "n_loss_exposed": ne, "recall_exposed": re_,
                "delta_recall": (rc - re_) if np.isfinite(rc) and np.isfinite(re_) else np.nan,
                "or": fisher_or_table(kc, nc - kc, ke, ne - ke) if min(nc, ne) >= 1 else {},
            }
            rows_out.append(item)
            print(uid, dms_id, "C", rc, "E", re_, "nC", nc, "nE", ne)
    pd.DataFrame(rows_out).to_csv(ADD2 / "proteingym_afdb_v6_recall.tsv", sep="\t", index=False)
    return rows_out


def main():
    ADD2.mkdir(parents=True, exist_ok=True)
    print("=== ConSurf submit (do not wait) ===")
    consurf = consurf_submit()
    val = pd.read_csv(MET_SPT / "wp3_validation_missense.tsv", sep="\t")
    val["pos"] = val["pos"].astype(int)
    val["dms_loss"] = val["GFP_score"] <= GFP_CUT
    val["grantham"] = [grantham(w, m) for w, m in zip(val["wt_aa"], val["mut_aa"])]
    val["class_af2"] = val["class"]
    tpt = pd.read_csv(MET_SPT / "tpt" / "oct1_tpt_residues.tsv", sep="\t")
    val["cluster"] = val["pos"].map(dict(zip(tpt["pos"].astype(int), tpt["cluster"])))
    ddg = pd.read_csv(MET_SPT / "wp3_p3_thermompnn_residue_median.tsv", sep="\t")

    print("=== phyloP / phastCons ===")
    cmap = residue_genomic_map()
    cmap.to_csv(ADD2 / "oct1_codon_map.tsv", sep="\t", index=False)
    chrom = str(cmap["chrom"].iloc[0])
    g0, g1 = int(cmap["g_start"].min()) - 2, int(cmap["g_end"].max()) + 2
    print("span", chrom, g0, g1, "n_pos", len(cmap))
    phylo = ucsc_track("phyloP100way", chrom, g0, g1)
    if len(phylo) < 100:
        print("phyloP100way empty, trying phyloP100wayAll")
        phylo = ucsc_track("phyloP100wayAll", chrom, g0, g1)
    phast = ucsc_track("phastCons100way", chrom, g0, g1)
    if len(phast) < 100:
        print("phastCons100way empty, trying phastCons100wayAll")
        phast = ucsc_track("phastCons100wayAll", chrom, g0, g1)
    cons = residue_conservation(cmap, phylo, phast)
    cons.to_csv(ADD2 / "oct1_phylop_phastcons.tsv", sep="\t", index=False)
    print("phylop cov", float(cons["cons_phylop"].notna().mean()),
          "phast cov", float(cons["cons_phastcons"].notna().mean()))

    val = val.merge(cons, on="pos", how="left")
    js = pd.read_csv(OUT / "oct1_conservation_a3m.tsv", sep="\t")
    val = val.merge(js[["pos", "cons_js"]], on="pos", how="left")

    res = (
        val.groupby("pos")
        .agg(cls=("class_af2", "first"), phylop=("cons_phylop", "first"),
             phast=("cons_phastcons", "first"), js=("cons_js", "first"))
        .reset_index()
    )
    desc = {}
    for col in ("phylop", "phast"):
        c = res.loc[res["cls"] == "CORE", col].dropna()
        e = res.loc[res["cls"] == "EXPOSED", col].dropna()
        _, p = stats.mannwhitneyu(c, e, alternative="greater")
        desc[col] = {
            "median_core": float(c.median()), "median_exposed": float(e.median()),
            "n_core": int(len(c)), "n_exposed": int(len(e)), "p": float(p),
            "spearman_vs_js": float(stats.spearmanr(res[col], res["js"], nan_policy="omit").statistic),
        }
        print(col, desc[col])

    print("=== A1 phyloP / phastCons ===")
    a1_phy = a1_model(val, "class_af2", "cons_phylop", n_boot=N_BOOT)
    a1_pha = a1_model(val, "class_af2", "cons_phastcons", n_boot=N_BOOT)
    print("phyloP clustered", a1_phy["logit_or_clustered"])
    print("phyloP GEE", a1_phy["gee"])
    print("phastCons clustered", a1_pha["logit_or_clustered"])
    print("MH phyloP", a1_phy["mantel_haenszel"])

    print("=== Youden pooled ===")
    youden = youden_pooled(val)
    print(json.dumps(youden, indent=2, default=str)[:2000])

    print("=== AFDB v6 OCT1 SPT ===")
    afdb_spt, counts = classify_afdb_oct1()
    print("counts", dict(counts))
    val["class_afdb"] = val["pos"].map(dict(zip(afdb_spt["pos"].astype(int), afdb_spt["class"])))
    ddg["class_afdb"] = ddg["pos"].map(dict(zip(afdb_spt["pos"].astype(int), afdb_spt["class"])))
    four = {
        "counts": dict(counts),
        "P1": p1_stats(val, "class_afdb"),
        "P3": p3_stats(ddg, "class_afdb"),
        "recall_gap": recall_gap(val, "class_afdb"),
        "P4": p4_clustered(val, "class_afdb"),
    }
    four["A1_js"] = a1_model(val, "class_afdb", "cons_js", n_boot=N_BOOT)
    four["A1_phylop"] = a1_model(val, "class_afdb", "cons_phylop", n_boot=N_BOOT)
    print("AFDB P1", four["P1"])
    print("AFDB recall", four["recall_gap"]["recall_core"], four["recall_gap"]["recall_exposed"])
    print("AFDB A1 JS", four["A1_js"]["logit_or_clustered"])

    m = afdb_spt.merge(
        pd.read_csv(MET_SPT / "oct1_af2_rank1_spt.tsv", sep="\t")[["pos", "class"]].rename(columns={"class": "af2"}),
        on="pos",
    )
    labs = ["CORE", "EXPOSED", "GREY"]
    conf = [[int(((m["af2"] == a) & (m["class"] == b)).sum()) for b in labs] for a in labs]
    four["confusion_AF2_x_AFDBv6"] = {
        "labels": labs, "counts": conf,
        "frac_agree": float((m["af2"] == m["class"]).mean()),
    }

    print("=== RMSD ColabFold vs AFDB v6 ===")
    cf = Path(
        "/HDD8T1/WORK/Metformin_HDD/structures/oct1_variants_20260811_204134/"
        "SLC22A1_WT_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb"
    )
    afdb_pdb = ADD / "afdb" / "AF-O15245-F1-model_v6.pdb"
    tm_pos = set(int(p) for p, t in zip(afdb_spt["pos"], afdb_spt["topology"]) if t == "Transmembrane")
    rmsd = {
        "colabfold_rank1": str(cf),
        "afdb_v6": str(afdb_pdb),
        "all_ca": rmsd_pair(cf, afdb_pdb),
        "plddt70": rmsd_pair(cf, afdb_pdb, plddt_min=70),
        "transmembrane_ca": rmsd_pair(cf, afdb_pdb, keys=tm_pos),
        "transmembrane_plddt70": rmsd_pair(cf, afdb_pdb, keys=tm_pos, plddt_min=70),
    }
    print(rmsd)

    print("=== ProteinGym AFDB v6 ===")
    pg = proteingym_afdb()

    print("=== gnomAD recall OR ===")
    gor = gnomad_recall_or()
    print(json.dumps(gor, indent=2, default=str)[:2500])

    payload = {
        "conservation_desc": desc,
        "A1_phylop": a1_phy,
        "A1_phastcons": a1_pha,
        "youden_pooled": youden,
        "AFDB_v6": four,
        "rmsd_colabfold_vs_afdb_v6": rmsd,
        "proteingym_afdb_v6": pg,
        "gnomad_recall_or": gor,
        "consurf_submit": consurf,
        "literature_P283L_R287W": LIT_P283_R287,
        "section6_phylop": {
            "ci": a1_phy["logit_or_clustered"]["ci95"],
            "excludes_1": bool(
                a1_phy["logit_or_clustered"]["ci95"][1] < 1
                or a1_phy["logit_or_clustered"]["ci95"][0] > 1
            ),
        },
    }
    dump(payload, ADD2 / "ms1_feedback2_addendum2.json")
    dump(payload, MS1 / "ms1_feedback2_addendum2.json")
    b = four
    pd.DataFrame([{
        "label": "AFDB_v6",
        "P1_median_CORE": b["P1"]["median_core"],
        "P1_median_EXPOSED": b["P1"]["median_exposed"],
        "P1_p": b["P1"]["p"],
        "recall_CORE": b["recall_gap"]["recall_core"],
        "recall_EXPOSED": b["recall_gap"]["recall_exposed"],
        "recall_OR_clustered_median": b["recall_gap"]["or_clustered"]["median"],
        "recall_OR_clustered_lo": b["recall_gap"]["or_clustered"]["ci95"][0],
        "recall_OR_clustered_hi": b["recall_gap"]["or_clustered"]["ci95"][1],
        "A1_js_EXPOSED_OR_median": b["A1_js"]["logit_or_clustered"]["median"],
        "A1_js_EXPOSED_OR_lo": b["A1_js"]["logit_or_clustered"]["ci95"][0],
        "A1_js_EXPOSED_OR_hi": b["A1_js"]["logit_or_clustered"]["ci95"][1],
        "P4_OR_clustered_lo": b["P4"]["or_clustered"]["ci95"][0],
        "P4_OR_clustered_hi": b["P4"]["or_clustered"]["ci95"][1],
        "AF2_AFDB_agree": four["confusion_AF2_x_AFDBv6"]["frac_agree"],
    }]).to_csv(ADD2 / "afdb_v6_fourth_column.tsv", sep="\t", index=False)
    print("wrote", ADD2)


if __name__ == "__main__":
    main()
