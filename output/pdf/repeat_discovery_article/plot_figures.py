"""Rebuild article figures from exported JSON, using NumPy and Matplotlib only."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/genomml-article-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "data/article_data.json").read_text())
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)
COLORS = ["#2166ac", "#d6604d", "#22836b", "#8c62aa", "#b08332"]


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def workflow():
    fig, ax = plt.subplots(figsize=(6.6, 2.65))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def box(x, y, text, color="#eef4fa", height=0.21):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                0.28,
                height,
                boxstyle="round,pad=.008,rounding_size=.018",
                facecolor=color,
                edgecolor="#55718a",
                linewidth=0.9,
            )
        )
        ax.text(
            x + 0.14,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=8.5,
            linespacing=1.4,
        )

    def arrow(a, b, dashed=False):
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1,
                color="#4b5d6c",
                linestyle="--" if dashed else "-",
            )
        )

    box(0.02, 0.68, "Genome sequence\n(no labels or ancestor)")
    box(0.36, 0.68, "Canonical 9-mer index\nUp to 32 sites per word")
    box(0.70, 0.68, "Oriented triplets\nSeed columns excluded")
    arrow((0.31, 0.785), (0.35, 0.785))
    arrow((0.65, 0.785), (0.69, 0.785))
    box(0.70, 0.29, "Genome-wide scan\nBoth DNA strands")
    box(0.36, 0.29, "Bootstrap and refine\nFive update iterations")
    box(0.02, 0.29, "Families and loci\nReject duplicate families")
    arrow((0.84, 0.67), (0.84, 0.51))
    arrow((0.69, 0.395), (0.65, 0.395))
    arrow((0.35, 0.395), (0.31, 0.395))
    box(0.70, 0.015, "Independent DNA\nCalibrate each profile", "#fbf1df", height=0.14)
    arrow((0.84, 0.165), (0.84, 0.28), True)
    ax.text(
        0.02,
        0.08,
        "Blind inference uses DNA only.\nTruth is reserved for evaluation.",
        fontsize=8,
        color="#475564",
        va="center",
    )
    save(fig, "workflow")


def final_performance():
    rows = DATA["final"]
    x = [r["x"] for r in rows]
    fig, axes = plt.subplots(
        1, 3, figsize=(6.6, 2.55), sharey=True, layout="constrained"
    )
    for ax, key, title, color in zip(
        axes,
        ["locus_precision", "locus_recall", "best_family_f1"],
        ["(a) Locus precision", "(b) Locus recall", "(c) Best-family F1"],
        COLORS,
    ):
        ax.axvspan(0, 1.5, color="#edf6f1", zorder=0)
        ax.axhline(0.9, color="#7d878e", ls="--", lw=0.8)
        ax.plot(x, [r[key] for r in rows], "-o", color=color, lw=1.8, ms=4.5)
        ax.set(
            title=title,
            xlabel="Mutation parameter x",
            xlim=(-0.08, 2.08),
            ylim=(-0.035, 1.055),
            xticks=x,
        )
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_ylabel(r"Score (IoU $\geq 0.5$)")
    save(fig, "final_performance")


def divergence_boundaries():
    rows = DATA["divergence"]
    x = np.array([r["x"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.85), layout="constrained")
    a, b = axes
    a.errorbar(
        x,
        [r["ancestor_identity"] for r in rows],
        yerr=[r["changed_sd"] for r in rows],
        fmt="o",
        color=COLORS[0],
        capsize=2.5,
        ms=4,
        label=r"To ancestor: mean $\pm$ SD",
    )
    a.plot(
        x,
        [r["theory_ancestor"] for r in rows],
        color=COLORS[0],
        lw=1.2,
        label="Ancestor: event-model expectation",
    )
    a.plot(
        x,
        [r["pair_identity"] for r in rows],
        "s",
        color=COLORS[1],
        ms=4,
        label="Between copies: mean",
    )
    a.plot(
        x,
        [r["theory_pair"] for r in rows],
        color=COLORS[1],
        ls="--",
        lw=1.2,
        label="Pair: event-model expectation",
    )
    a.axhline(0.25, color="gray", ls=":", lw=1)
    a.set(
        title="(a) Event budget versus sequence identity",
        xlabel="Mutation parameter x",
        ylabel="Aligned nucleotide identity",
        xticks=x,
        ylim=(0.2, 1.055),
    )
    a.legend(loc="upper right", fontsize=6.6, frameon=False, labelspacing=0.55)
    for row, color in zip(DATA["boundaries"], COLORS):
        if not row["iou"]:
            continue
        values = np.sort(row["iou"])
        unique, counts = np.unique(values, return_counts=True)
        b.step(
            np.r_[0.85, unique, 1.005],
            np.r_[0, np.cumsum(counts) / len(values), 1],
            where="post",
            lw=1.8,
            color=color,
            label=f"x = {row['x']:g}",
        )
    b.set(
        title="(b) Boundary accuracy of matched loci",
        xlabel="Intersection over union",
        ylabel="Empirical cumulative fraction",
        xlim=(0.85, 1.005),
        ylim=(0, 1.04),
    )
    b.legend(loc="upper left", frameon=False, fontsize=7.5)
    save(fig, "divergence_boundaries")


def diagnostic_comparison():
    labels = list(dict.fromkeys(r["variant"] for r in DATA["comparison"]))
    short = [
        "SW, 200 bp",
        "SW, adaptive",
        "Ungapped",
        "Ungapped + NMS",
        "Triplet + profile",
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(6.6, 3.1), sharey=True, layout="constrained"
    )
    for ax, key, title in zip(
        axes, ["locus_f1", "best_family_f1"], ["(a) Locus F1", "(b) Best-family F1"]
    ):
        ys = np.arange(len(labels))
        for offset, x, color in [(-0.17, 0, COLORS[0]), (0.17, 0.5, COLORS[1])]:
            vals = [
                next(
                    r[key]
                    for r in DATA["comparison"]
                    if r["variant"] == label and r["x"] == x
                )
                for label in labels
            ]
            ax.barh(ys + offset, vals, height=0.30, color=color, label=f"x = {x:g}")
        ax.set(
            xlim=(0, 1.04),
            yticks=ys,
            yticklabels=short,
            title=title,
            xlabel="Score (validation seed 44)",
        )
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
    axes[1].tick_params(labelleft=False)
    axes[0].legend(loc="lower right", frameon=False, ncol=2, fontsize=7)
    save(fig, "diagnostic_comparison")


def profile_dynamics():
    evolution = DATA["profile_evolution"]
    x = np.arange(len(evolution))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), layout="constrained")
    a, b = axes
    a.plot(
        x,
        [r["hits"] for r in evolution],
        "-o",
        color=COLORS[0],
        ms=4,
        label="Accepted genomic hits",
    )
    a.plot(
        x,
        [r["matched"] for r in evolution],
        "--s",
        color=COLORS[1],
        ms=3.5,
        label="Matched true loci (post hoc)",
    )
    a.axhline(150, color="gray", ls=":", lw=0.9)
    a.set(
        title="(a) Bootstrap at x = 1.5",
        xlabel="Profile iteration",
        ylabel="Number of loci",
        xticks=x,
        xticklabels=[str(r["iteration"]) for r in evolution],
    )
    a.legend(loc="lower right", frameon=False, fontsize=7)
    values = np.array(DATA["profile_information"])
    smooth = np.convolve(values, np.ones(21) / 21, mode="same")
    b.plot(
        np.arange(len(values)),
        values,
        color="#a9bfd2",
        lw=0.7,
        label="Per-column information",
    )
    b.plot(
        np.arange(len(values)),
        smooth,
        color=COLORS[0],
        lw=1.5,
        label="21-column moving mean",
    )
    b.set(
        title="(b) Learned profile information",
        xlabel="Position in final profile (bp)",
        ylabel="Information relative to IID (nats)",
    )
    b.legend(loc="upper right", frameon=False, fontsize=6.8)
    save(fig, "profile_dynamics")


if __name__ == "__main__":
    workflow()
    final_performance()
    divergence_boundaries()
    diagnostic_comparison()
    profile_dynamics()
    print("Wrote five vector PDF figures and five 300 dpi PNG figures.")
