#!/usr/bin/env python3
"""Addendum to feedback 260901: ConSurf-DB A1 rerun, gnomAD×DMS, pooled Cheng, ProteinGym protocol.

    source /SSD1T/PhD/AlphaFold/met_env.sh
    $MET_PY met_fb260901_addendum.py
"""
from __future__ import annotations

import json
import os
import re
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from met_fb260901 import (  # noqa: E402
    AM_PATH,
    DESIGN_POS,
    GFP_CUT,
    MS1,
    N_BOOT,
    OUT,
    SEED,
    a1_model,
    dump,
    grantham,
    json_default,
)

MET_HDD = Path(os.environ.get("MET_HDD", "/HDD8T1/WORK/Metformin_HDD"))
MET_SPT = Path(os.environ.get("MET_SPT", str(MET_HDD / "spt")))
MET_PGYM = MET_HDD / "proteingym"
ADD_OUT = OUT / "addendum"
CTX = ssl._create_unverified_context()


def get(url: str, timeout=60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "oct1-spt-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.read()


def consurfdb_fetch(pdb: str, chain: str = "A") -> dict:
    import requests
    s = requests.Session()
    r = s.get(
        "https://consurfdb.tau.ac.il/scripts/chain_selection.php",
        params={"pdb_ID": pdb.upper()},
        verify=False,
        timeout=60,
    )
    r.raise_for_status()
    mapping = dict(re.findall(r'option value="(\w) (\w{5})"', r.text))
    info = {"pdb": pdb, "chain": chain, "html_ok": True, "n_chains": len(mapping),
            "mapping": mapping, "no_chains": "No chains found for" in r.text}
    if info["no_chains"] or chain not in mapping and len(mapping) != 1:
        info["ok"] = False
        info["snippet"] = r.text[:500]
        return info
    final = list(mapping.values())[0] if len(mapping) == 1 else mapping[chain]
    url = f"https://consurfdb.tau.ac.il/DB/{final}/consurf_summary.txt"
    g = s.get(url, verify=False, timeout=60)
    info["final"] = final
    info["grades_url"] = url
    info["grades_status"] = g.status_code
    info["ok"] = g.status_code == 200 and "POS" in g.text
    if info["ok"]:
        info["grades_text"] = g.text
        info["n_chars"] = len(g.text)
    else:
        info["snippet"] = g.text[:500]
    return info


def parse_consurf_grades(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = [p.strip() for p in line.split("\t") if p.strip() != ""]
        if len(parts) < 5:
            parts = [p for p in re.split(r"\s{2,}|\t", line) if p]
        if len(parts) < 5:
            continue
        try:
            pos = int(parts[0])
            seq = parts[1][0] if parts[1] else "?"
            atom = parts[2]
            score = float(str(parts[3]).replace("*", ""))
            color = int(str(parts[4]).replace("*", "")[0])
        except (ValueError, IndexError):
            continue
        m = re.search(r"(\d+)", atom)
        atom_res = int(m.group(1)) if m and atom != "-" else np.nan
        rows.append({"consurf_pos": pos, "aa": seq, "atom": atom, "atom_resno": atom_res,
                     "score": score, "grade": color})
    return pd.DataFrame(rows)


def pdb_ca(path: Path) -> dict[int, np.ndarray]:
    out = {}
    with path.open() as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resi = int(line[22:26])
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                out[resi] = xyz
    return out


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(0)
    b = b - b.mean(0)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    return float(np.sqrt(((a @ r - b) ** 2).sum(axis=1).mean()))


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        dest.write_bytes(get(url, timeout=90))
        return dest.stat().st_size > 1000
    except Exception as e:
        print("download fail", url, e)
        return False


def gnomad_dms_join():
    g = pd.read_csv(OUT / "gnomad_slc22a1_missense.tsv", sep="\t")
    dms = pd.read_csv(MET_SPT / "wp3_validation_missense.tsv", sep="\t")
    dms["dms_loss"] = dms["GFP_score"] <= GFP_CUT
    dms["pos"] = dms["pos"].astype(int)
    # design positions are absent from validation; join AM/GFP from full DMS if needed
    j = g.merge(dms[["hgvs_short", "pos", "class", "GFP_score", "dms_loss", "am_class"]],
                on="hgvs_short", how="left")
    j["in_dms_validation"] = j["GFP_score"].notna()
    j["design_pos"] = j["hgvs_short"].str.extract(r"(\d+)").astype(float).isin(DESIGN_POS)
    matched = j[j["in_dms_validation"]].copy()
    loss = matched[matched["dms_loss"]].copy()
    # Cheng miss = not pathogenic (benign or ambiguous)
    loss["cheng_miss"] = loss["am_class_cheng"] != "pathogenic"
    pops = {"EUR": "af_nfe", "EAS": "af_eas", "SAS": "af_sas", "AFR": "af_afr", "AMR": "af_amr"}

    def pack(df, mask=None):
        sub = df if mask is None else df[mask]
        out = {"n": int(len(sub))}
        for alias, col in pops.items():
            out[f"sum_af_{alias}"] = float(sub[col].fillna(0).sum()) if col in sub.columns else np.nan
        return out

    by_class = {}
    for lab in ("CORE", "EXPOSED", "GREY"):
        sl = loss[loss["class"] == lab]
        by_class[lab] = {
            "n_gfp_loss": int(len(sl)),
            "n_cheng_path": int((sl["am_class_cheng"] == "pathogenic").sum()),
            "n_cheng_miss": int(sl["cheng_miss"].sum()),
            "frac_cheng_miss": float(sl["cheng_miss"].mean()) if len(sl) else np.nan,
            "sum_af_miss": {k: float(sl.loc[sl["cheng_miss"], c].fillna(0).sum())
                            for k, c in pops.items()},
            "sum_af_all_loss": {k: float(sl[c].fillna(0).sum()) for k, c in pops.items()},
        }
    summary = {
        "n_gnomad_missense_rows": int(len(g)),
        "n_with_am": int(g["am"].notna().sum()) if "am" in g.columns else None,
        "n_joined_dms_validation": int(matched.shape[0]),
        "n_gnomad_not_in_validation": int((~j["in_dms_validation"]).sum()),
        "n_gfp_loss": int(len(loss)),
        "n_cheng_miss_among_gfp_loss": int(loss["cheng_miss"].sum()),
        "frac_cheng_miss_among_gfp_loss": float(loss["cheng_miss"].mean()) if len(loss) else np.nan,
        "gfp_loss_all": pack(loss),
        "gfp_loss_cheng_miss": pack(loss, loss["cheng_miss"]),
        "gfp_loss_cheng_path": pack(loss, ~loss["cheng_miss"]),
        "by_spt_class": by_class,
        "note": (
            "Join key = hgvs_short on validation missense (design 61/88/401/420/465 excluded). "
            "Cheng miss = am_class_cheng != pathogenic (benign+ambiguous). "
            "sum_af is the sum of variant AFs, not unique-person prevalence."
        ),
    }
    j.to_csv(ADD_OUT / "gnomad_x_dms.tsv", sep="\t", index=False)
    loss.to_csv(ADD_OUT / "gnomad_gfp_loss.tsv", sep="\t", index=False)
    return summary, j, loss


def cheng_pooled_vs_fold():
    freeze = json.loads((OUT / "ms1_feedback2_freeze.json").read_text())
    folds = freeze["B1_helix_LOPO_folds"]
    val = pd.read_csv(MET_SPT / "wp3_validation_missense.tsv", sep="\t")
    val["dms_loss"] = val["GFP_score"] <= GFP_CUT
    tpt = pd.read_csv(MET_SPT / "tpt" / "oct1_tpt_residues.tsv", sep="\t")
    cmap = dict(zip(tpt["pos"].astype(int), tpt["cluster"]))
    val["cluster"] = val["pos"].map(cmap)
    tested = [f["held_out"] for f in folds]
    te = val[val["cluster"].isin(tested)].dropna(subset=["am_pathogenicity", "dms_loss"])

    def metrics(df, t=AM_PATH):
        y = df["dms_loss"].astype(int).to_numpy()
        pred = (df["am_pathogenicity"] > t).astype(int).to_numpy()
        tp = int(((y == 1) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        return {
            "n": int(len(df)), "n_loss": int(y.sum()),
            "sens": tp / (tp + fn) if (tp + fn) else np.nan,
            "spec": tn / (tn + fp) if (tn + fp) else np.nan,
            "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
            "npv": tn / (tn + fn) if (tn + fn) else np.nan,
        }

    pooled = {"ALL": metrics(te)}
    for lab in ("CORE", "EXPOSED", "GREY"):
        pooled[lab] = metrics(te[te["class"] == lab])

    def fold_mean(lab, key="sens"):
        xs = []
        for f in folds:
            m = (f.get("heldout_cheng") or {}).get(lab) or {}
            if m.get(key) is not None and np.isfinite(m[key]):
                xs.append(m[key])
        return {"mean": float(np.mean(xs)), "n_folds": len(xs)} if xs else {}

    full = {"ALL": metrics(val.dropna(subset=["am_pathogenicity"]))}
    for lab in ("CORE", "EXPOSED", "GREY"):
        full[lab] = metrics(val[val["class"] == lab])

    return {
        "why_pooled": (
            "Cheng 0.564 is a locked external cutoff, not estimated inside each fold. "
            "The inferential quantity is therefore the pooled held-out table "
            "(every residue appears in exactly one test fold), not the unweighted mean of fold rates."
        ),
        "n_folds": len(folds),
        "n_pooled_variants": int(len(te)),
        "pooled_heldout_cheng": pooled,
        "unweighted_fold_mean_cheng_sens": {lab: fold_mean(lab) for lab in ("ALL", "CORE", "EXPOSED", "GREY")},
        "full_data_cheng": full,
        "youden_fold_mean_kept_as_cv": freeze["B1_helix_LOPO"]["heldout_EXPOSED_sens_youden"],
    }


def proteingym_protocol():
    struct = MET_PGYM / "structures" / "ProteinGym_AF2_structures"
    acc = {
        "ADRB2_HUMAN": "P07550",
        "CCR5_HUMAN": "P51681",
        "HMDH_HUMAN": "P04035",
        "NPC1_HUMAN": "O15118",
        "SC6A4_HUMAN": "P31645",
        "VKOR1_HUMAN": "Q9BQB6",
    }
    cache = ADD_OUT / "afdb"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for uid, uniprot in acc.items():
        pg = struct / f"{uid}.pdb"
        header_title = None
        is_af_monomer_v2 = False
        has_hydrogens = False
        with pg.open() as f:
            for i, line in enumerate(f):
                if line.startswith("TITLE") and header_title is None:
                    header_title = line[10:].strip()
                if "ALPHAFOLD MONOMER V2.0" in line:
                    is_af_monomer_v2 = True
                if line.startswith("ATOM") and line[12:16].strip() in {"H", "H2", "H3", "HA"}:
                    has_hydrogens = True
                if i > 40 and line.startswith("ATOM"):
                    break
        pg_ca = pdb_ca(pg)
        rec = {
            "UniProt_ID": uid, "uniprot_acc": uniprot,
            "pgym_title": header_title,
            "pgym_title_says_af_monomer_v2": is_af_monomer_v2,
            "pgym_has_hydrogens": has_hydrogens,
            "n_ca": len(pg_ca),
        }
        for ver, tag in (("v2", "model_v2"), ("v4", "model_v4")):
            url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-{tag}.pdb"
            dest = cache / f"AF-{uniprot}-F1-{tag}.pdb"
            ok = download(url, dest)
            rec[f"afdb_{ver}_downloaded"] = ok
            if not ok:
                rec[f"rmsd_pgym_vs_afdb_{ver}"] = np.nan
                continue
            af_ca = pdb_ca(dest)
            keys = sorted(set(pg_ca) & set(af_ca))
            if len(keys) < 20:
                rec[f"rmsd_pgym_vs_afdb_{ver}"] = np.nan
                rec[f"n_ca_overlap_{ver}"] = len(keys)
                continue
            a = np.stack([pg_ca[k] for k in keys])
            b = np.stack([af_ca[k] for k in keys])
            rec[f"rmsd_pgym_vs_afdb_{ver}"] = kabsch_rmsd(a, b)
            rec[f"n_ca_overlap_{ver}"] = len(keys)
            rec[f"max_ca_absdiff_{ver}"] = float(np.linalg.norm(a - b, axis=1).max())
        rec["verdict"] = (
            "AFDB AlphaFold-Monomer v2 (DeepMind/EBI), not ColabFold"
            if (rec.get("rmsd_pgym_vs_afdb_v2") is not None
                and np.isfinite(rec.get("rmsd_pgym_vs_afdb_v2", np.nan))
                and rec["rmsd_pgym_vs_afdb_v2"] < 0.15)
            else (
                "AFDB v2 header present"
                if is_af_monomer_v2
                else "compare RMSD"
            )
        )
        rows.append(rec)
        print(uid, rec.get("rmsd_pgym_vs_afdb_v2"), rec.get("rmsd_pgym_vs_afdb_v4"), rec["verdict"])
    df = pd.DataFrame(rows)
    df.to_csv(ADD_OUT / "proteingym_structure_protocol.tsv", sep="\t", index=False)
    return {
        "source_zip": "ProteinGym v1.3 ProteinGym_AF2_structures.zip (Zenodo 15293562)",
        "not_colabfold": (
            "These PDBs are the ProteinGym-distributed AlphaFold2 models. "
            "CCR5 carries the AFDB 'ALPHAFOLD MONOMER V2.0' header and DeepMind author REMARK. "
            "Cα RMSD vs EBI AFDB v2 models near 0 ⇒ same coordinates as AFDB v2, not a local ColabFold run."
        ),
        "proteins": rows,
    }


def consurf_a1(val: pd.DataFrame):
    attempts = []
    grades_df = None
    used = None
    for pdb in ("8SC1", "8ET6", "8SC4", "8ET8"):
        print("ConSurf-DB", pdb)
        try:
            info = consurfdb_fetch(pdb, "A")
        except Exception as e:
            info = {"pdb": pdb, "ok": False, "error": f"{type(e).__name__}: {e}"}
        keep = {k: v for k, v in info.items() if k != "grades_text"}
        attempts.append(keep)
        if info.get("ok"):
            (ADD_OUT / f"consurf_{pdb}_summary.txt").write_text(info["grades_text"])
            grades_df = parse_consurf_grades(info["grades_text"])
            used = pdb
            break
    if grades_df is None or grades_df.empty:
        return {"ok": False, "attempts": attempts, "a1": None}

    # Map to UniProt via ATOM residue number when present, else consurf POS
    grades_df["pos"] = grades_df["atom_resno"].fillna(grades_df["consurf_pos"]).astype(int)
    grades_df["cons_r4s"] = -grades_df["score"]  # higher = more conserved
    grades_df["cons_grade"] = grades_df["grade"].astype(float)
    grades_df.to_csv(ADD_OUT / f"consurf_{used}_parsed.tsv", sep="\t", index=False)

    v = val.merge(grades_df[["pos", "cons_r4s", "cons_grade", "score", "aa"]], on="pos", how="left")
    n_cov = int(v["cons_r4s"].notna().mean() * v["pos"].nunique()) if False else int(v.dropna(subset=["cons_r4s"])["pos"].nunique())
    a1_grade = a1_model(v, "class_af2", "cons_grade", n_boot=N_BOOT)
    a1_r4s = a1_model(v, "class_af2", "cons_r4s", n_boot=N_BOOT)
    # descriptive
    res = (v.dropna(subset=["cons_grade"])
           .groupby("pos")
           .agg(cls=("class_af2", "first"), grade=("cons_grade", "first"), r4s=("cons_r4s", "first"))
           .reset_index())
    from scipy import stats
    desc = {}
    for col in ("grade", "r4s"):
        c = res.loc[res["cls"] == "CORE", col]
        e = res.loc[res["cls"] == "EXPOSED", col]
        _, p = stats.mannwhitneyu(c, e, alternative="greater")
        desc[col] = {"median_core": float(c.median()), "median_exposed": float(e.median()),
                     "n_core": int(len(c)), "n_exposed": int(len(e)), "p_core_greater": float(p)}
    return {
        "ok": True,
        "source": f"ConSurf-DB PDB {used} chain A (HMMER homologues + MAFFT + Rate4Site; same engine as ConSurf web)",
        "not_interactive_job": True,
        "attempts": attempts,
        "n_residues_scored": int(len(grades_df)),
        "n_validation_residues_with_score": n_cov,
        "descriptive": desc,
        "a1_grade_1to9": a1_grade,
        "a1_neg_score": a1_r4s,
        "spearman_vs_js": None,
    }


def three_estimator_note():
    freeze = json.loads((OUT / "ms1_feedback2_freeze.json").read_text())
    a1 = freeze["A1"]["AF2"]["js"]
    gee_or = a1["gee"]["or"]
    return {
        "mh": a1["mantel_haenszel"]["or"],
        "mh_ci": a1["mantel_haenszel"]["ci95"],
        "logit_core_vs_exposed": a1["logit_or_clustered"]["or_core_vs_exposed_median"],
        "logit_ci": a1["logit_or_clustered"]["or_core_vs_exposed_ci95"],
        "gee_exposed_or": gee_or,
        "gee_core_vs_exposed": 1.0 / gee_or,
        "gee_ci_inverted": [1.0 / a1["gee"]["ci95"][1], 1.0 / a1["gee"]["ci95"][0]],
    }


def main():
    ADD_OUT.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(MET_SPT / "wp3_validation_missense.tsv", sep="\t")
    val["pos"] = val["pos"].astype(int)
    val["dms_loss"] = val["GFP_score"] <= GFP_CUT
    val["grantham"] = [grantham(w, m) for w, m in zip(val["wt_aa"], val["mut_aa"])]
    val["class_af2"] = val["class"]
    cons = pd.read_csv(OUT / "oct1_conservation_a3m.tsv", sep="\t")
    val = val.merge(cons[["pos", "cons_js"]], on="pos", how="left")

    print("=== ConSurf-DB A1 ===")
    consurf = consurf_a1(val)
    if consurf.get("ok"):
        # Spearman vs JS at residue level
        g = pd.read_csv(ADD_OUT / f"consurf_{[a for a in consurf['attempts'] if a.get('ok')][0]['pdb']}_parsed.tsv", sep="\t")
        m = cons.merge(g, on="pos", how="inner")
        from scipy import stats
        consurf["spearman_vs_js"] = {
            "n": int(len(m.dropna(subset=["cons_js", "cons_r4s"]))),
            "rho_js_vs_negscore": float(stats.spearmanr(m["cons_js"], m["cons_r4s"], nan_policy="omit").statistic),
            "rho_js_vs_grade": float(stats.spearmanr(m["cons_js"], m["cons_grade"], nan_policy="omit").statistic),
        }
        print("A1 grade clustered", consurf["a1_grade_1to9"]["logit_or_clustered"])
        print("A1 -score clustered", consurf["a1_neg_score"]["logit_or_clustered"])
        print("GEE grade", consurf["a1_grade_1to9"]["gee"])
    else:
        print("ConSurf failed", consurf.get("attempts"))

    print("=== gnomAD x DMS ===")
    gsum, _, _ = gnomad_dms_join()
    print({k: gsum[k] for k in gsum if k not in ("by_spt_class", "gfp_loss_all",
                                                 "gfp_loss_cheng_miss", "gfp_loss_cheng_path", "note")})
    print("by class", json.dumps(gsum["by_spt_class"], default=str)[:1500])
    print("miss AF", gsum["gfp_loss_cheng_miss"])

    print("=== Cheng pooled ===")
    cheng = cheng_pooled_vs_fold()
    print(json.dumps({k: cheng[k] for k in cheng if k != "why_pooled"}, indent=2, default=str)[:2500])

    print("=== ProteinGym protocol ===")
    pg = proteingym_protocol()

    est = three_estimator_note()
    payload = {
        "consurf": {k: v for k, v in consurf.items() if k not in ("a1_grade_1to9", "a1_neg_score")}
        | {
            "a1_grade_1to9": consurf.get("a1_grade_1to9"),
            "a1_neg_score": consurf.get("a1_neg_score"),
        },
        "gnomad_x_dms": gsum,
        "cheng_pooled": cheng,
        "proteingym_protocol": pg,
        "three_estimators": est,
    }
    # strip huge nested boots? a1 already stores clustered summary only
    dump(payload, ADD_OUT / "ms1_feedback2_addendum.json")
    dump(payload, MS1 / "ms1_feedback2_addendum.json")
    print("wrote", ADD_OUT)


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings()
    main()
