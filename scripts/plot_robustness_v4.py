"""Plot saved robustness assays with system Python/Matplotlib, not inference."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/genomml-robustness-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/robustness_v4"
DATA = json.loads((OUT / "report_data.json").read_text())
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "svg.fonttype": "none",
    }
)
BLUE, ORANGE = "#2166ac", "#d6604d"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=250, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def indels():
    rows = DATA["holdout"]
    pos = np.arange(len(rows))
    labels = [
        f"x={r['case']['x']:g}\nD={100 * r['case']['deletion_rate']:g}%\nI={100 * r['case']['insertion_rate']:g}%"
        for r in rows
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    for ax, key, title in zip(
        axes,
        ["locus_recall", "mean_matched_iou"],
        ["(a) Locus recall (IoU >= 0.5)", "(b) Boundary IoU of matched loci"],
    ):
        for offset, method, color, label in [
            (-0.18, "ungapped", BLUE, "Frozen v3"),
            (0.18, "gapped", ORANGE, "Gap-aware refinement"),
        ]:
            values = [r[method][key] if r[method]["locus_tp"] else np.nan for r in rows]
            ax.bar(pos + offset, values, width=0.34, color=color, label=label)
        for i, r in enumerate(rows):
            if r["ungapped"]["candidate_families"] == 0:
                ax.text(
                    i,
                    0.08,
                    "no initial\nprofile",
                    ha="center",
                    fontsize=7,
                    color="#59636c",
                )
        ax.set(title=title, xticks=pos, xticklabels=labels, ylim=(0, 1.08))
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Fraction")
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Indel holdout: seed 72, 150 copies in 500 kb", fontsize=11)
    fig.subplots_adjust(top=0.79, bottom=0.24, wspace=0.27)
    fig.text(
        0.5,
        0.02,
        "D/I: deleted/inserted bases as a fraction of ancestral length; single-base events. One design, no confidence intervals.",
        ha="center",
        fontsize=7,
    )
    save(fig, "indel_holdout")


def real():
    rows = DATA["real"]
    pos = np.arange(len(rows))
    labels = [
        "22.1-22.6\nv3",
        "24.1-24.6\nv3",
        "24.1-24.6\nraw gaps",
        "24.1-24.6\n+ final NMS*",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    known = np.array([r["overlapping_interspersed_repeatmasker"] for r in rows])
    other = np.array([r["overlapping_any_repeatmasker"] for r in rows]) - known
    unmatched = np.array(
        [r["predicted_loci"] - r["overlapping_any_repeatmasker"] for r in rows]
    )
    a, b = axes
    a.bar(pos, known, color=BLUE, label="Interspersed-repeat annotation")
    a.bar(pos, other, bottom=known, color="#bc8c37", label="Other repeat annotation")
    a.bar(
        pos,
        unmatched,
        bottom=known + other,
        color="#b9c0c6",
        label="No assigned annotation",
    )
    for i, r in enumerate(rows):
        a.text(
            i,
            r["predicted_loci"] + 8,
            str(r["predicted_loci"]),
            ha="center",
            fontsize=8,
        )
    a.set(
        title="(a) Predicted intervals and annotation overlap",
        ylabel="Number of intervals",
        xticks=pos,
        xticklabels=labels,
        ylim=(0, 620),
    )
    a.legend(loc="upper left", frameon=False, fontsize=7)
    for offset, key, color, label in [
        (
            -0.18,
            "short_annotation_row_recall_iou0.5",
            BLUE,
            "All 100-1000 bp annotation rows",
        ),
        (0.18, "near_complete_row_recall_iou0.5", ORANGE, "Near-complete subset"),
    ]:
        values = [r[key] for r in rows]
        b.bar(pos + offset, values, width=0.34, color=color, label=label)
        for i, value in enumerate(values):
            b.text(i + offset, value + 0.025, f"{value:.3f}", ha="center", fontsize=7)
    b.set(
        title="(b) Annotation-row recall at IoU >= 0.5",
        ylabel="Fraction of annotation rows",
        xticks=pos,
        xticklabels=labels,
        ylim=(0, 1.08),
    )
    b.legend(loc="upper left", frameon=False, fontsize=7)
    for ax in axes:
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
    fig.subplots_adjust(bottom=0.24, top=0.88, wspace=0.28)
    fig.text(
        0.5,
        0.02,
        "hg38 chr22: annotation overlap is NOT precision. Several predicted intervals can map to one RepeatMasker row.",
        ha="center",
        fontsize=8,
        color="#4d565e",
    )
    fig.text(
        0.5,
        0.065,
        "* Post hoc curation on the same real interval; not an additional independent test.",
        ha="center",
        fontsize=7,
        color="#4d565e",
    )
    save(fig, "real_concordance")


if __name__ == "__main__":
    indels()
    real()
    print("Saved two PNG/SVG figures")
