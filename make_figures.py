#!/usr/bin/env python3
"""Build PLOS CB display items from frozen SPT files.

Does not retune SPT cuts. Does not recompute clustered CIs: those are
locked in ms1_feedback2_*.json. Point estimates in the plots match the
manuscript.

Reads data/spt/ unless MET_HDD is set. Outputs PDF + PNG + TIFF under figures/.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
HDD = Path(os.environ.get("MET_HDD", str(ROOT / "data")))
SPT = HDD / "spt"
OUT = ROOT / "figures"

C_CORE = "#0072B2"
C_EXPOSED = "#E69F00"
C_GREY = "#7A7A7A"
C_LINE = "#222222"
C_MUTED = "#4D4D4D"
CLASS_COLOR = {"CORE": C_CORE, "EXPOSED": C_EXPOSED, "GREY": C_GREY}


def _roc_curve(y: np.ndarray, scores: np.ndarray):
    """Sensitivity vs 1-specificity; no sklearn."""
    y = y.astype(bool)
    order = np.argsort(-scores, kind="mergesort")
    y = y[order]
    tps = np.cumsum(y)
    fps = np.cumsum(~y)
    n_pos = tps[-1] if tps.size else 0
    n_neg = fps[-1] if fps.size else 0
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    tpr = tps / n_pos
    fpr = fps / n_neg
    fpr = np.r_[0.0, fpr, 1.0]
    tpr = np.r_[0.0, tpr, 1.0]
    return fpr, tpr
ORDER = ["CORE", "EXPOSED", "GREY"]
GFP_CUT = -0.814
AM_PATH = 0.564
AM_BENIGN = 0.34


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
        -0.14, 1.08, letter, transform=ax.transAxes, fontsize=11,
        fontweight="bold", va="top", ha="left",
    )


def _save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(
        OUT / f"{stem}.tiff", dpi=300, bbox_inches="tight", facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    print(f"wrote {stem}")


def fig1():
    gfp = pd.read_csv(SPT / "wp3_residue_median_gfp.tsv", sep="\t")
    ddg = pd.read_csv(SPT / "wp3_p3_thermompnn_residue_median.tsv", sep="\t")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3))
    for ax, df, col, ylab, hline in (
        (axes[0], gfp, "gfp_median", "Residue-median GFP", GFP_CUT),
        (axes[1], ddg, "ddg", "Residue-median ΔΔG (kcal mol⁻¹)", None),
    ):
        data = [df.loc[df["class"] == cl, col].dropna().to_numpy() for cl in ORDER]
        ns = [d.size for d in data]
        labels = [f"{name}\n(n = {n})" for name, n in zip(["Buried", "Exposed", "Grey"], ns)]
        bp = ax.boxplot(
            data, tick_labels=labels,
            patch_artist=True, widths=0.62, showfliers=False,
            medianprops={"color": C_LINE, "linewidth": 1.2},
            whiskerprops={"color": C_LINE}, capprops={"color": C_LINE},
            boxprops={"linewidth": 0.8},
        )
        for patch, cl in zip(bp["boxes"], ORDER):
            patch.set_facecolor(CLASS_COLOR[cl])
            patch.set_alpha(0.85)
        if hline is not None:
            ax.axhline(hline, color=C_MUTED, ls="--", lw=0.8, zorder=0)
            lo, hi = ax.get_ylim()
            ticks = [float(t) for t in ax.get_yticks() if lo <= t <= hi]
            if not any(abs(t - hline) < 1e-9 for t in ticks):
                ticks = sorted(ticks + [hline])
            ax.set_yticks(ticks)
            labs = []
            for t in ticks:
                if abs(t - hline) < 1e-9:
                    labs.append("−0.814")
                elif abs(t) < 1e-12:
                    labs.append("0")
                elif abs(t - round(t)) < 1e-9:
                    v = int(round(t))
                    labs.append(f"−{abs(v)}" if v < 0 else str(v))
                else:
                    labs.append(f"{t:g}")
            ax.set_yticklabels(labs)
            for tick, lab in zip(ax.get_yticks(), ax.get_yticklabels()):
                if abs(tick - hline) < 1e-9:
                    lab.set_color(C_MUTED)
        ax.set_ylabel(ylab)
        ax.set_xlim(0.4, 3.6)
    _panel(axes[0], "A")
    _panel(axes[1], "B")
    fig.tight_layout()
    _save(fig, "Fig1")


def fig2():
    # Locked recall and phyloP quintiles (addendum2 A1_phylop).
    quint = [
        (1, 0.6575, 0.2353),
        (2, 0.6883, 0.2422),
        (3, 0.7537, 0.3889),
        (4, 0.8349, 0.4211),
        (5, 0.8510, 0.7478),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [0.88, 1.5]})

    ax = axes[0]
    rec = [0.783, 0.417]
    x = np.array([0, 1])
    ax.bar(x, rec, color=[C_CORE, C_EXPOSED], width=0.62, edgecolor=C_LINE, linewidth=0.6)
    ax.set_xticks(x, ["Buried", "Exposed"])
    ax.set_ylabel("AM pathogenic recall among GFP-loss")
    ax.set_ylim(0, 1.12)
    for i, (r, n) in enumerate(((0.783, "1,378/1,760"), (0.417, "245/588"))):
        ax.text(i, r + 0.03, f"{100 * r:.1f}%\n{n}", ha="center", va="bottom", fontsize=7)
    ax.annotate(
        "", xy=(0.5, 0.417), xytext=(0.5, 0.783),
        arrowprops=dict(arrowstyle="<->", color=C_LINE, lw=0.8),
    )
    ax.text(0.58, 0.60, "RD 36.6 pp", ha="left", va="center", fontsize=7)
    ax.set_xlim(-0.42, 1.36)
    _panel(ax, "A")

    ax = axes[1]
    # phyloP GFP-loss n from addendum2 A1_phylop quintiles.
    quint_n = [(219, 68), (231, 128), (337, 144), (436, 133), (537, 115)]
    q = np.arange(1, 6)
    w = 0.36
    ax.bar(q - w / 2, [t[1] for t in quint], width=w, color=C_CORE, edgecolor=C_LINE, lw=0.5, label="Buried")
    ax.bar(q + w / 2, [t[2] for t in quint], width=w, color=C_EXPOSED, edgecolor=C_LINE, lw=0.5, label="Exposed")
    ax.set_xticks(q, [f"Q{i}\n{nc}/{ne}" for i, (nc, ne) in enumerate(quint_n, start=1)])
    ax.set_ylabel("AM pathogenic recall among GFP-loss")
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("phyloP quintile (buried n / exposed n)")
    ax.legend(frameon=False, loc="upper left")
    ax.annotate(
        "Q5 OR 1.93", xy=(5, 0.88), xytext=(5, 1.00),
        ha="center", va="bottom", fontsize=7, color=C_MUTED,
        arrowprops=dict(arrowstyle="-", color=C_MUTED, lw=0.6),
    )
    _panel(ax, "B")
    fig.tight_layout()
    _save(fig, "Fig2")


def fig3():
    cm = np.array([
        [117, 3, 48],
        [0, 85, 9],
        [7, 13, 156],
    ], dtype=float)
    labels = ["Buried", "Exposed", "Grey"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.05, 1.2]})

    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list("seq", ["#FFFFFF", C_CORE])
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=160)
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("8SC1 (inward-open)")
    ax.set_ylabel("AF2 (preregistered)")
    for i in range(3):
        for j in range(3):
            val = int(cm[i, j])
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > 80 else C_LINE, fontsize=9)
    ax.set_title("358/438 agree (81.7%)", fontsize=8, pad=4)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    _panel(ax, "A")

    ax = axes[1]
    rows = [
        ("AF2", 5.11, 3.42, 7.64, True),
        ("8SC1", 3.60, 2.20, 5.98, False),
        ("Agreement", 5.17, 2.91, 9.43, False),
        ("AFDB v6", 5.18, 3.50, 7.84, False),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, row in zip(y, rows):
        name, pt, lo, hi, primary = row
        ax.plot([lo, hi], [yi, yi], color=C_CORE if primary else C_MUTED, lw=1.6)
        ax.plot(pt, yi, "o", color=C_CORE if primary else C_MUTED, ms=6)
        ax.text(hi + 0.15, yi, f"{pt:.2f}", va="center", fontsize=7)
    ax.axvline(1, color=C_LINE, lw=0.8, ls="--")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Clustered buried:exposed recall odds ratio (log scale)")
    ax.set_xlim(0.5, 11)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    _panel(ax, "B")
    fig.tight_layout()
    _save(fig, "Fig3")


def fig4():
    val = pd.read_csv(SPT / "wp3_validation_missense.tsv", sep="\t")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))

    ax = axes[0]
    for cl in ORDER:
        sub = val.loc[val["class"] == cl, ["am_pathogenicity", "dms_loss"]].dropna()
        y = sub["dms_loss"].astype(int).to_numpy()
        s = sub["am_pathogenicity"].to_numpy()
        fpr, tpr = _roc_curve(y, s)
        ax.plot(fpr, tpr, color=CLASS_COLOR[cl], lw=1.4,
                label={"CORE": "Buried 0.746", "EXPOSED": "Exposed 0.813", "GREY": "Grey 0.840"}[cl])
    ax.plot([0, 1], [0, 1], color=C_MUTED, lw=0.7, ls="--")
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity (GFP-loss)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right", title="AUROC")
    _panel(ax, "A")

    ax = axes[1]
    rules = ["Cheng 0.564\n(pooled held-out)", "Youden ~0.479\n(pooled held-out)"]
    core_s = [0.783, 0.839]
    exp_s = [0.417, 0.490]
    x = np.arange(2)
    w = 0.36
    ax.bar(x - w / 2, core_s, width=w, color=C_CORE, edgecolor=C_LINE, lw=0.5, label="Buried")
    ax.bar(x + w / 2, exp_s, width=w, color=C_EXPOSED, edgecolor=C_LINE, lw=0.5, label="Exposed")
    ax.set_xticks(x, rules)
    ax.set_ylabel("Pathogenic sensitivity among GFP-loss")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, loc="upper right")
    ax.annotate("", xy=(0.18, 0.417), xytext=(0.18, 0.783),
                arrowprops=dict(arrowstyle="<->", color=C_LINE, lw=0.8))
    ax.text(0.22, 0.60, "36.6 pp", fontsize=7, va="center")
    ax.annotate("", xy=(1.18, 0.490), xytext=(1.18, 0.839),
                arrowprops=dict(arrowstyle="<->", color=C_LINE, lw=0.8))
    ax.text(1.22, 0.66, "34.9 pp", fontsize=7, va="center")
    _panel(ax, "B")
    fig.tight_layout()
    _save(fig, "Fig4")


def fig5():
    sweep = pd.read_csv(SPT / "fb260901/addendum5/SLC6A4_cutoff_sweep.tsv", sep="\t")
    fig = plt.figure(figsize=(7.4, 3.8))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.15, 1.25], height_ratios=[1.55, 1.05],
        hspace=0.62, wspace=0.38, left=0.18, right=0.98, top=0.90, bottom=0.12,
    )

    ax = fig.add_subplot(gs[0, 0])
    rows = [
        ("OCT1\n(total-cell GFP)", 1.88, 1.56, 2.37),
        ("SERT\n(anti-myc surface)", 1.90, 1.53, 2.52),
    ]
    y = np.array([1.0, 0.0])
    for yi, row in zip(y, rows):
        _name, pt, lo, hi = row
        ax.plot([lo, hi], [yi, yi], color=C_CORE, lw=1.8)
        ax.plot(pt, yi, "o", color=C_CORE, ms=7, zorder=3)
        ax.text(pt, yi + 0.18, f"{pt:.2f} [{lo:.2f}–{hi:.2f}]", ha="center", va="bottom", fontsize=7)
    ax.axvline(1, color=C_LINE, lw=0.8, ls="--")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Clustered buried:exposed risk ratio (log scale)")
    ax.set_xscale("log")
    ax.set_xlim(0.95, 3.1)
    ax.set_xticks([1, 1.5, 2, 2.5], ["1", "1.5", "2", "2.5"])
    ax.set_ylim(-0.45, 1.45)
    _panel(ax, "A")

    rec = fig.add_subplot(gs[1, 0])
    proteins = ["OCT1", "SERT"]
    core = [0.783, 0.971]
    exp = [0.417, 0.510]
    xi = np.arange(2)
    w = 0.36
    rec.bar(xi - w / 2, core, width=w, color=C_CORE, edgecolor=C_LINE, lw=0.5)
    rec.bar(xi + w / 2, exp, width=w, color=C_EXPOSED, edgecolor=C_LINE, lw=0.5)
    rec.set_xticks(xi, proteins)
    rec.set_ylim(0, 1.15)
    rec.set_ylabel("Recall")
    rec.set_title("Underlying recall", fontsize=7, pad=2)

    ax = fig.add_subplot(gs[:, 1])
    ax.plot(sweep["cut"], 100 * sweep["rd"], color=C_CORE, lw=1.4, marker="o", ms=3.5)
    ax.axhline(36.6, color=C_EXPOSED, ls="--", lw=1.0, label="OCT1 RD 36.6 pp")
    ax.axvline(-1.907, color=C_MUTED, ls=":", lw=0.9)
    ax.text(-1.88, 52.2, "−1.907", fontsize=7, color=C_MUTED)
    ax.set_xlabel("SERT surface-expression loss cutoff")
    ax.set_ylabel("Risk difference (percentage points)")
    ax.set_ylim(35, 55)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("21/21 cutoffs, RD 42.8–51.2 pp", fontsize=8, pad=4)
    _panel(ax, "B")
    _save(fig, "Fig5")


def fig6():
    g = pd.read_csv(SPT / "fb260901/addendum/gnomad_gfp_loss.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    rng = np.random.default_rng(20260812)
    recall = {"CORE": "61/113", "EXPOSED": "9/37", "GREY": "58/108"}
    for i, cl in enumerate(ORDER):
        sub = g.loc[g["class"] == cl, "am"].dropna().to_numpy()
        jitter = rng.uniform(-0.18, 0.18, size=sub.size)
        ax.scatter(np.full(sub.size, i) + jitter, sub, s=10, alpha=0.7,
                   c=CLASS_COLOR[cl], edgecolors="none", zorder=2)
        ax.text(i, 1.02, recall[cl], ha="center", va="bottom", fontsize=7)
    ax.axhline(AM_PATH, color=C_LINE, ls="--", lw=0.8)
    ax.axhline(AM_BENIGN, color=C_MUTED, ls=":", lw=0.8)
    ax.annotate(
        "0.564",
        xy=(1.0, AM_PATH),
        xycoords=("axes fraction", "data"),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=7,
        annotation_clip=False,
    )
    ax.annotate(
        "0.34",
        xy=(1.0, AM_BENIGN),
        xycoords=("axes fraction", "data"),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=7,
        color=C_MUTED,
        annotation_clip=False,
    )
    ax.set_xticks(range(3), ["Buried", "Exposed", "Grey"])
    ax.set_ylabel("AlphaMissense score")
    ax.set_ylim(-0.02, 1.12)
    ax.set_xlim(-0.5, 2.6)
    fig.tight_layout()
    _save(fig, "Fig6")


def main():
    _style()
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print(f"figures → {OUT}")


if __name__ == "__main__":
    main()
