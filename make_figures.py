#!/usr/bin/env python3
"""Build TPJ display items for manuscript 1 from frozen SPT files.

Does not retune SPT cuts. Does not invent statistics: point estimates come from
frozen JSON/TSV. Residue-bootstrap CIs for Fig. 2a are computed with the same
clustered method as P2 (seed 20260812, 10 000 resamples) and cached.

Outputs (PDF vector + TIFF 300 dpi) under figures/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator
from scipy import stats

ROOT = Path(__file__).resolve().parent
HDD = Path(os.environ.get("MET_HDD", str(ROOT / "data")))
SPT = HDD / "spt"
OUT = ROOT / "figures"
CACHE = SPT / "ms1_figure_stats.json"

# Design-set AF2 mutant vs WT global Cα RMSD (met_af_plan.md §12-A.12).
DESIGN_RMSD = {
    "R61C": 0.995,
    "C88R": 1.423,
    "G401S": 1.405,
    "M420del": 1.445,
    "G465R": 0.944,
}

# Okabe–Ito (colourblind-safe). Distinct when used as identifiers.
C_CORE = "#0072B2"
C_EXPOSED = "#E69F00"
C_GREY = "#7A7A7A"
C_PATH = "#D55E00"
C_BENIGN = "#009E73"
C_AMBIG = "#CC79A7"
C_NEUT = "#56B4E9"
C_GAIN = "#009E73"
C_LOSS = "#D55E00"
C_LINE = "#222222"
C_MUTED = "#4D4D4D"

ORDER = ["CORE", "EXPOSED", "GREY"]
CLASS_COLOR = {"CORE": C_CORE, "EXPOSED": C_EXPOSED, "GREY": C_GREY}
N_BOOT = 10_000
GFP_CUT = -0.814


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _panel(ax, letter: str):
    ax.text(
        -0.12, 1.08, letter, transform=ax.transAxes, fontsize=11,
        fontweight="bold", va="top", ha="left",
    )


def _save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / f"{stem}.pdf"
    tiff = OUT / f"{stem}.tiff"
    png = OUT / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(tiff, dpi=300, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {pdf.name}, {tiff.name}")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    r, _ = stats.spearmanr(x, y)
    return float(r)


def residue_bootstrap_rho(df: pd.DataFrame, xcol: str, ycol: str, seed: int):
    """Same clustered bootstrap as met_dms.residue_bootstrap_rho (independent RNG per class)."""
    rng = np.random.default_rng(seed)
    sub = df[["pos", xcol, ycol]].dropna()
    pos = sub["pos"].to_numpy()
    x = sub[xcol].to_numpy(dtype=float)
    y = sub[ycol].to_numpy(dtype=float)
    order = np.argsort(pos, kind="mergesort")
    pos, x, y = pos[order], x[order], y[order]
    starts = np.r_[0, np.flatnonzero(pos[1:] != pos[:-1]) + 1]
    ends = np.r_[starts[1:], pos.size]
    gx = [x[s:e] for s, e in zip(starts, ends)]
    gy = [y[s:e] for s, e in zip(starts, ends)]
    n_res = len(gx)

    def rho_idx(idx):
        return _spearman(
            np.concatenate([gx[i] for i in idx]),
            np.concatenate([gy[i] for i in idx]),
        )

    all_idx = np.arange(n_res)
    point = rho_idx(all_idx)
    draws = rng.integers(0, n_res, size=(N_BOOT, n_res))
    boots = np.array([rho_idx(d) for d in draws], dtype=float)
    boots = boots[np.isfinite(boots)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"rho": float(point), "ci_lo": float(lo), "ci_hi": float(hi), "n_res": n_res}


def load_p2_cis(val: pd.DataFrame) -> dict:
    if CACHE.exists():
        cached = json.loads(CACHE.read_text())
        if cached.get("n_boot") == N_BOOT and "P2_rho_by_class" in cached:
            return cached["P2_rho_by_class"]
    seeds = {"CORE": 20260812, "EXPOSED": 20260813, "GREY": 20260814}
    out = {}
    for cl, seed in seeds.items():
        sub = val[val["class"] == cl]
        out[cl] = residue_bootstrap_rho(sub, "am_pathogenicity", "GFP_score", seed)
        print(f"  P2 bootstrap {cl}: ρ={out[cl]['rho']:.3f} [{out[cl]['ci_lo']:.3f}, {out[cl]['ci_hi']:.3f}]")
    payload = {}
    if CACHE.exists():
        payload = json.loads(CACHE.read_text())
    payload["n_boot"] = N_BOOT
    payload["P2_rho_by_class"] = out
    payload["note"] = (
        "Per-class CIs: independent residue-clustered bootstrap, 10 000 resamples. "
        "P2 delta CI remains the preregistered value in wp3_p1_p2_p4.json."
    )
    CACHE.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def strip_box(ax, data, colors, ylabel, hline=None, hlabel=None):
    rng = np.random.default_rng(1)
    for i, (lab, vals) in enumerate(data):
        vals = np.asarray(vals, dtype=float)
        x = i + 1
        bp = ax.boxplot(
            [vals], positions=[x], widths=0.55, patch_artist=True, showfliers=False,
            medianprops={"color": C_LINE, "linewidth": 1.2},
            whiskerprops={"color": C_LINE, "linewidth": 1.0},
            capprops={"color": C_LINE, "linewidth": 1.0},
            boxprops={"facecolor": "none", "edgecolor": colors[i], "linewidth": 1.4},
        )
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(
            np.full(len(vals), x) + jitter, vals,
            s=6, c=colors[i], alpha=0.28, linewidths=0, zorder=2, rasterized=True,
        )
        ax.scatter([x], [np.median(vals)], s=18, c=C_LINE, zorder=4, linewidths=0)
    ax.set_xlim(0.4, len(data) + 0.6)
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels([d[0] for d in data])
    ax.set_ylabel(ylabel)
    if hline is not None:
        ax.axhline(hline, color=C_MUTED, lw=1.0, ls="--", zorder=0)
        if hlabel:
            ax.text(len(data) + 0.55, hline, hlabel, va="center", ha="left", fontsize=7, color=C_MUTED)


def fig1():
    gfp = pd.read_csv(SPT / "wp3_residue_median_gfp.tsv", sep="\t")
    ddg = pd.read_csv(SPT / "wp3_p3_thermompnn_residue_median.tsv", sep="\t")
    p3 = json.loads((SPT / "wp3_p3_verdict.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    gfp_data = [(cl, gfp.loc[gfp["class"] == cl, "gfp_median"].to_numpy()) for cl in ORDER]
    ddg_data = [(cl, ddg.loc[ddg["class"] == cl, "ddg"].to_numpy()) for cl in ORDER]
    cols = [CLASS_COLOR[c] for c in ORDER]
    strip_box(axes[0], gfp_data, cols, "Residue-median GFP score", GFP_CUT, "loss cutoff")
    y0 = max(gfp.gfp_median.max() + 0.15, 1.35)
    axes[0].plot([1, 1, 2, 2], [y0 - 0.18, y0, y0, y0 - 0.18], color=C_LINE, lw=1.0)
    axes[0].text(1.5, y0 + 0.06, "P1 Holm p = 1.3 × 10⁻²²", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylim(min(-3.4, axes[0].get_ylim()[0]), y0 + 0.55)
    _panel(axes[0], "a")
    strip_box(axes[1], ddg_data, cols, r"Residue-median $\Delta\Delta G$ (kcal mol$^{-1}$)")
    axes[1].axhline(0, color=C_MUTED, lw=1.0, zorder=0)
    axes[1].set_ylim(bottom=0)
    axes[1].text(
        0.98, 0.02, f"P3  n = {p3['n_variants']:,} missense".replace(",", " "),
        transform=axes[1].transAxes, ha="right", va="bottom", fontsize=7, color=C_MUTED,
    )
    _panel(axes[1], "b")
    fig.tight_layout(w_pad=2.0)
    _save(fig, "Fig1")


def fig2(val: pd.DataFrame, freeze: dict, p2_cis: dict, p2_json: dict):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    ax = axes[0]
    xs = np.arange(len(ORDER))
    rhos = [p2_cis[cl]["rho"] for cl in ORDER]
    yerr = np.array([
        [p2_cis[cl]["rho"] - p2_cis[cl]["ci_lo"] for cl in ORDER],
        [p2_cis[cl]["ci_hi"] - p2_cis[cl]["rho"] for cl in ORDER],
    ])
    bars = ax.bar(
        xs, rhos, color=[CLASS_COLOR[c] for c in ORDER], width=0.62,
        edgecolor=C_LINE, linewidth=1.0, zorder=2,
    )
    ax.errorbar(xs, rhos, yerr=yerr, fmt="none", ecolor=C_LINE, elinewidth=1.0, capsize=3, capthick=1.0, zorder=3)
    ax.axhline(0, color=C_LINE, lw=1.0, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(ORDER)
    ax.set_ylabel(r"Spearman $\rho$ (AM vs GFP)")
    dlo, dhi = p2_json["delta_abs_ci"]
    ax.text(
        0.02, 0.02,
        f"P2  Δ|ρ| CORE−EXPOSED 95% CI [{dlo:.3f}, {dhi:.3f}] includes 0  (fail)",
        transform=ax.transAxes, fontsize=6.5, va="bottom", ha="left", color=C_MUTED,
    )
    ax.set_ylim(-0.85, 0.05)
    _panel(ax, "a")

    ax = axes[1]
    rec = freeze["am_pathogenic_recall_among_gfp_loss"]
    bottom = np.zeros(3)
    layers = [
        ("pathogenic", C_PATH, [rec[cl]["n_pathogenic"] / rec[cl]["n_loss"] for cl in ORDER]),
        ("ambiguous", C_AMBIG, [rec[cl]["n_ambiguous"] / rec[cl]["n_loss"] for cl in ORDER]),
        ("benign", C_BENIGN, [rec[cl]["n_benign"] / rec[cl]["n_loss"] for cl in ORDER]),
    ]
    for name, col, fracs in layers:
        ax.bar(xs, fracs, bottom=bottom, color=col, width=0.62, edgecolor=C_LINE, linewidth=1.0, label=name)
        bottom += np.array(fracs)
    ax.set_xticks(xs)
    ax.set_xticklabels(ORDER)
    ax.set_ylabel("Fraction of GFP-loss missense")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    for i, cl in enumerate(ORDER):
        r = rec[cl]["recall_pathogenic"]
        ax.text(i, r / 2, f"{100*r:.0f}%", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    ax.legend(frameon=False, loc="upper right", fontsize=7, title="AM class")
    ax.text(
        0.02, 0.02, "pathogenic recall  (secondary to P2/P4)",
        transform=ax.transAxes, fontsize=6.5, va="bottom", color=C_MUTED,
    )
    _panel(ax, "b")
    fig.tight_layout(w_pad=2.2)
    _save(fig, "Fig2")


def fig3():
    af2 = pd.read_csv(SPT / "oct1_af2_vs_8sc1.tsv", sep="\t")
    frac_8sc1 = float(af2["agree"].mean())
    g3 = json.loads((SPT / "g3_oct1_8et6.json").read_text())
    wp6 = json.loads((SPT / "wp6_verdict.json").read_text())
    wp7 = json.loads((SPT / "wp7_verdict.json").read_text())
    labels = [
        "AF2 vs 8SC1\n(inward WT)",
        "AF2 vs 8ET6\n(outward OCT1CS)",
        "OCT2 AF2 vs 8ET9\n(outward OCT2CS)",
        "8SC1 vs 8SC4\n(inward ± metformin)",
    ]
    vals = [
        100 * frac_8sc1,
        100 * g3["frac"],
        100 * wp6["G3"]["frac"],
        100 * wp7["spt_8sc1_vs_8sc4"]["frac"],
    ]
    passed = [v >= 80 for v in vals]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    xs = np.arange(len(vals))
    for i, (v, ok) in enumerate(zip(vals, passed)):
        ax.bar(
            i, v, width=0.62, facecolor=C_CORE if ok else "white",
            edgecolor=C_CORE if ok else C_PATH, linewidth=1.4,
            hatch="" if ok else "///", zorder=2,
        )
        ax.text(i, v + 1.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.axhline(80, color=C_LINE, ls="--", lw=1.2, zorder=1)
    ax.text(3.55, 80, "80% bar", va="center", ha="left", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("SPT class agreement (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.6, 3.7)
    ax.legend(
        handles=[
            Patch(facecolor=C_CORE, edgecolor=C_CORE, label="≥80% (pass)"),
            Patch(facecolor="white", edgecolor=C_PATH, hatch="///", label="<80% (fail)"),
        ],
        frameon=False, loc="lower right", fontsize=7,
    )
    fig.tight_layout()
    _save(fig, "Fig3")


def fig4():
    p6 = json.loads((SPT / "wp5_p6_rmsd.json").read_text())
    noise = p6["noise_max_A"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    ax = axes[0]
    names = list(DESIGN_RMSD)
    vals = [DESIGN_RMSD[k] for k in names]
    xs = np.arange(len(names))
    ax.bar(xs, vals, width=0.62, facecolor="white", edgecolor=C_CORE, linewidth=1.4, zorder=2)
    ax.scatter(xs, vals, s=18, c=C_CORE, zorder=3, linewidths=0)
    ax.axhline(noise, color=C_PATH, ls="--", lw=1.2, zorder=1)
    ax.text(len(names) - 0.45, noise + 0.08, f"noise {noise:.3f} Å", color=C_PATH, fontsize=7, ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(r"AF2 mutant vs WT global C$\alpha$ RMSD (Å)")
    ax.set_ylim(0, 4.0)
    _panel(ax, "a")

    ax = axes[1]
    pair_et6 = p6["pairs"][0]
    pair_sc4 = p6["pairs"][1]
    labs = ["8SC1 vs 8ET6\n(OCT1CS)", "8SC1 vs 8SC4\n(inward ± met)"]
    glob = [pair_et6["rmsd_identical"], pair_sc4["rmsd_identical"]]
    tm = [pair_et6["rmsd_tm_identical"], pair_sc4["rmsd_tm_identical"]]
    x = np.arange(2)
    w = 0.32
    ax.bar(x - w / 2, glob, width=w, facecolor=C_CORE, edgecolor=C_LINE, linewidth=1.0, label="identical-residue")
    ax.bar(x + w / 2, tm, width=w, facecolor="white", edgecolor=C_LINE, linewidth=1.0, hatch="///", label="TM-only")
    ax.axhline(noise, color=C_PATH, ls="--", lw=1.2)
    ax.text(1.48, noise + 0.08, f"noise {noise:.3f} Å", color=C_PATH, fontsize=7, ha="right")
    for i, (g, t) in enumerate(zip(glob, tm)):
        ax.text(i - w / 2, g + 0.08, f"{g:.3f}", ha="center", fontsize=7)
        ax.text(i + w / 2, t + 0.08, f"{t:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel(r"Experimental C$\alpha$ RMSD (Å)")
    ax.set_ylim(0, 5.0)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    _panel(ax, "b")
    fig.tight_layout(w_pad=2.0)
    _save(fig, "Fig4")


def fig5():
    lit = pd.read_csv(SPT / "wp4_heldout_literature.tsv", sep="\t")
    lit = lit[(~lit.excluded_design) & (~lit.is_deletion)].copy()
    assert len(lit) == 34
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    ax = axes[0]
    funcs = ["loss*", "neutral", "gain"]
    fcol = {"loss*": C_LOSS, "neutral": C_NEUT, "gain": C_GAIN}
    x = np.arange(len(ORDER))
    w = 0.24
    for j, fn in enumerate(funcs):
        counts = [int(((lit.spt_class == cl) & (lit.func_bundle == fn)).sum()) for cl in ORDER]
        ax.bar(x + (j - 1) * w, counts, width=w, color=fcol[fn], edgecolor=C_LINE, linewidth=1.0, label=fn)
        for xi, c in zip(x + (j - 1) * w, counts):
            if c:
                ax.text(xi, c + 0.12, str(c), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER)
    ax.set_ylabel("Literature missense alleles")
    ax.set_ylim(0, 12)
    ax.legend(frameon=False, fontsize=7, title="reported function")
    ax.text(0.02, 0.98, r"$n$ = 34 after design exclusion", transform=ax.transAxes, va="top", fontsize=7, color=C_MUTED)
    _panel(ax, "a")

    ax = axes[1]
    loss = lit[lit.func_bundle == "loss*"].copy()
    am_levels = ["pathogenic", "ambiguous", "benign"]
    am_col = {"pathogenic": C_PATH, "ambiguous": C_AMBIG, "benign": C_BENIGN}
    bottom = np.zeros(3)
    for lab in am_levels:
        counts = np.array([
            int(((loss.spt_class == cl) & (loss.am_class.fillna("none") == lab)).sum())
            for cl in ORDER
        ])
        ax.bar(x, counts, bottom=bottom, color=am_col[lab], edgecolor=C_LINE, linewidth=1.0, label=lab)
        bottom += counts
    names = ", ".join(sorted(loss.loc[loss.spt_class == "EXPOSED", "hgvs_short"].tolist()))
    ax.text(
        1.0, 3.35, f"H4.2: {names}\nall AM-benign (n = 3)",
        ha="center", va="bottom", fontsize=7, color=C_LINE,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER)
    ax.set_ylabel("Literature loss* alleles")
    ax.set_ylim(0, 12)
    ax.legend(frameon=False, fontsize=7, title="AM class", loc="upper left")
    ax.text(0.98, 0.02, "not a P4 replicate", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=C_MUTED)
    _panel(ax, "b")
    fig.tight_layout(w_pad=1.8)
    _save(fig, "Fig5")


def fig_s1(val: pd.DataFrame):
    c3 = json.loads((HDD / "challenge/c3_ensemble/c3_verdict.json").read_text())
    c3x = json.loads((HDD / "challenge/c3_ensemble/c3x_verdict.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [1.05, 1.15, 0.95]})

    ax = axes[0]
    rhos = []
    for cl in ORDER:
        sub = val.loc[val["class"] == cl, ["am_pathogenicity", "SM73_0_score"]].dropna()
        rhos.append(_spearman(sub.am_pathogenicity.to_numpy(), sub.SM73_0_score.to_numpy()))
    xs = np.arange(3)
    ax.bar(xs, rhos, color=[CLASS_COLOR[c] for c in ORDER], edgecolor=C_LINE, linewidth=1.0, width=0.62)
    ax.axhline(0, color=C_LINE, lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(ORDER)
    ax.set_ylabel(r"Spearman $\rho$ (AM vs raw SM73)")
    ax.set_ylim(-0.15, 0.15)
    for i, r in enumerate(rhos):
        ax.text(i, r + (0.012 if r >= 0 else -0.012), f"{r:.3f}", ha="center", va="bottom" if r >= 0 else "top", fontsize=7)
    _panel(ax, "a")

    ax = axes[1]
    am = c3["rho_am_resid"]
    anm = c3x["C3x_8_anm_only_vs_AM"]["rho_anm_only"]
    ens = c3["rho_ens_resid"]
    # Error bars: published Δρ vs AM CIs (AM ρ ≈ 0 so Δρ ≈ ρ)
    am_ci = c3x["C3x_8_anm_only_vs_AM"]["delta"]
    ens_ci = c3["delta_vs_AM"]
    names = ["AM", "ANM-only", "Full ensemble"]
    pts = [am, anm, ens]
    lo = [am, am + am_ci["ci_lo"], am + ens_ci["ci_lo"]]
    hi = [am, am + am_ci["ci_hi"], am + ens_ci["ci_hi"]]
    cols = [C_GREY, C_NEUT, C_CORE]
    xs = np.arange(3)
    ax.bar(xs, pts, color=cols, edgecolor=C_LINE, linewidth=1.0, width=0.62, zorder=2)
    yerr = np.array([np.array(pts) - np.array(lo), np.array(hi) - np.array(pts)])
    yerr[0, 0] = 0
    yerr[1, 0] = 0
    ax.errorbar(xs, pts, yerr=yerr, fmt="none", ecolor=C_LINE, elinewidth=1.0, capsize=3, capthick=1.0, zorder=3)
    ax.axhline(0, color=C_LINE, lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel(r"Spearman $\rho$ vs GFP-adjusted SM73")
    ax.set_ylim(0, 0.13)
    ax.text(0.02, 0.98, r"error bars: bootstrap $\Delta\rho$ vs AM", transform=ax.transAxes, va="top", fontsize=6.5, color=C_MUTED)
    _panel(ax, "b")

    ax = axes[2]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_autoscale_on(False)
    tr = ax.transAxes
    # Vertical flow so labels stay inside boxes in a narrow panel.
    boxes = [
        (0.08, 0.70, 0.84, 0.26, "GFP abundance", "explains most SM73"),
        (0.08, 0.38, 0.84, 0.26, "Residual SM73", "after GFP adjustment"),
        (0.08, 0.06, 0.84, 0.26, "Geometry + ANM", "ensemble ρ = 0.077\nweak; not a predictor"),
    ]
    for x, y, w, h, title, sub in boxes:
        ax.add_patch(Rectangle(
            (x, y), w, h, transform=tr, facecolor="white",
            edgecolor=C_LINE, linewidth=1.2, clip_on=False,
        ))
        ax.text(x + w / 2, y + h - 0.035, title, transform=tr,
                ha="center", va="top", fontsize=8, fontweight="bold", clip_on=False)
        ax.text(x + w / 2, y + 0.04, sub, transform=tr,
                ha="center", va="bottom", fontsize=6.5, color=C_MUTED, clip_on=False,
                linespacing=1.25)
    # Arrows in axes coordinates (between boxes).
    for y0, y1 in ((0.70, 0.64), (0.38, 0.32)):
        ax.annotate(
            "", xy=(0.50, y1), xytext=(0.50, y0),
            xycoords=tr, textcoords=tr,
            arrowprops={"arrowstyle": "-|>", "color": C_LINE, "lw": 1.2,
                        "mutation_scale": 8, "shrinkA": 0, "shrinkB": 0},
            annotation_clip=False,
        )
    _panel(ax, "c")
    fig.tight_layout(w_pad=1.2)
    _save(fig, "FigS1")


def supp_table_s1():
    v = json.loads((HDD / "challenge/i_instead/i1c_verdict.json").read_text())
    rows = [
        ["Universe after R5 exclusion", v["n"]["universe"], "", ""],
        ["Stab*", v["n"]["Stab*"], "GFP-loss, near-zero residual", "not surface gold"],
        ["Trans*", v["n"]["Trans*"], "GFP intact, residual-loss", "not surface gold"],
        ["WT*", v["n"]["WT*"], "GFP intact, residual ok", ""],
        ["I1C.1 CORE in Stab* vs Trans*", f"OR {v['I1C.1']['or']:.2f}", f"Holm p = {v['I1C.1']['p_holm']:.2f}", "FAIL"],
        ["I1C.2 ΔΔG Stab* vs Trans*", f"{v['I1C.2']['median_stab']:.3f} vs {v['I1C.2']['median_trans']:.3f}", f"Holm p = {v['I1C.2']['p_holm']:.3f}", "PASS (GFP-redundant)"],
        ["I1C.5 AM-benign Trans* vs Stab*", f"OR {v['I1C.5']['or']:.2f}", f"Holm p = {v['I1C.5']['p_holm']:.4f}", "GFP-confounded; not an instead-rule"],
    ]
    df = pd.DataFrame(rows, columns=["Item", "Statistic", "Detail", "Verdict"])
    csv = OUT / "TableS1_I1C.csv"
    df.to_csv(csv, index=False)
    try:
        xlsx = OUT / "TableS1_I1C.xlsx"
        df.to_excel(xlsx, index=False)
        print(f"wrote {csv.name}, {xlsx.name}")
    except Exception:
        print(f"wrote {csv.name} (xlsx skipped)")


def main():
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(SPT / "wp3_validation_missense.tsv", sep="\t")
    freeze = json.loads((SPT / "ms1_feedback1_freeze.json").read_text())
    p2_json = json.loads((SPT / "wp3_p1_p2_p4.json").read_text())["P2"]
    print("computing / loading Fig. 2a bootstrap CIs…")
    p2_cis = load_p2_cis(val)
    fig1()
    fig2(val, freeze, p2_cis, p2_json)
    fig3()
    fig4()
    fig5()
    fig_s1(val)
    supp_table_s1()
    print("done", OUT)


if __name__ == "__main__":
    main()
