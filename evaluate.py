"""
evaluate.py
===========
Evaluates the VRU pipeline by comparing its predictions to GT labels.
Supports multiple (pipeline_json, gt_json) pairs and reports both
per-video and unified statistics.

Outputs:
  - Console: per-video + global metrics
  - EPS file: normalized confusion matrix heatmap

Reads  : one or more (tracks_pipeline.json, tracks_gt.json) pairs
Matching: direct track_id lookup — no IoU, no re-runs.
Skipped tracks (null in GT) are excluded from all metrics.
"""

from pathlib import Path
from collections import defaultdict
import json

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# List of (pipeline_json, gt_json) pairs — one entry per video
VIDEO_PAIRS = [
    (r"tracks_pipeline_video1.json", r"tracks_gt_video1.json"),
    (r"tracks_pipeline_video2.json", r"tracks_gt_video2.json"),
    (r"tracks_pipeline_video3.json", r"tracks_gt_video3.json"),
]

# Output EPS path
CONFUSION_MATRIX_EPS = r"confusion_matrix.eps"

# Set to True to also print per-video breakdowns before the global summary
PRINT_PER_VIDEO = True


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CLASSES = ["pedestrian", "cyclist", "motorcyclist", "pmd"]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load(pipeline_path: str, gt_path: str):
    with open(pipeline_path, "r") as f:
        pipeline = json.load(f)
    with open(gt_path, "r") as f:
        gt = json.load(f)
    return pipeline["tracks"], gt


# ─────────────────────────────────────────────────────────────────────────────
# CORE EVALUATION — returns raw result records for one video
# ─────────────────────────────────────────────────────────────────────────────

def collect_results(pipeline_tracks: dict, gt: dict) -> list[dict]:
    results = []
    for tid, gt_label in gt.items():
        if gt_label is None:
            continue
        pred_info = pipeline_tracks.get(tid)
        pred_cls  = pred_info["pipeline_class"] if pred_info else "unknown"
        pred_conf = pred_info["confidence"]      if pred_info else 0.0
        results.append({
            "tid":     tid,
            "gt":      gt_label,
            "pred":    pred_cls,
            "conf":    pred_conf,
            "correct": pred_cls == gt_label,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE — compute TP/FP/FN + confusion from a list of results
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(results: list[dict]):
    tp        = defaultdict(int)
    fp        = defaultdict(int)
    fn        = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))

    for r in results:
        gt_c = r["gt"]
        pr_c = r["pred"]
        confusion[gt_c][pr_c] += 1
        if pr_c == gt_c:
            tp[gt_c] += 1
        else:
            fn[gt_c] += 1
            fp[pr_c] += 1

    return tp, fp, fn, confusion


# ─────────────────────────────────────────────────────────────────────────────
# PRINT REPORT — shared by per-video and global sections
# ─────────────────────────────────────────────────────────────────────────────

def print_report(label: str, results: list[dict], gt_total: int, skipped: int):
    total   = len(results)
    correct = sum(r["correct"] for r in results)
    wrong   = total - correct

    if total == 0:
        print(f"\n[{label}] No annotated tracks to evaluate.\n")
        return

    tp, fp, fn, confusion = aggregate(results)

    print()
    print("=" * 65)
    print(f"  {label}")
    print("=" * 65)
    print(f"  Tracks in GT file           : {gt_total}")
    print(f"  Skipped (excluded)          : {skipped}")
    print(f"  Evaluated                   : {total}")
    print(f"  Correct                     : {correct}")
    print(f"  Wrong                       : {wrong}")
    print("─" * 65)
    print(f"  Overall accuracy            : {correct / total * 100:.2f}%")
    print("=" * 65)

    print(f"\n  {'Class':<16} {'TP':>5} {'FP':>5} {'FN':>5}"
          f" {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("  " + "─" * 55)
    for cls in CLASSES:
        t   = tp[cls]
        f_p = fp[cls]
        f_n = fn[cls]
        prec = t / (t + f_p) * 100 if (t + f_p) > 0 else 0.0
        rec  = t / (t + f_n) * 100 if (t + f_n) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"  {cls:<16} {t:>5} {f_p:>5} {f_n:>5}"
              f" {prec:>7.2f}% {rec:>7.2f}% {f1:>7.2f}%")

    print()
    print("  Per-class accuracy (TP / GT instances for that class):")
    for cls in CLASSES:
        gt_count = tp[cls] + fn[cls]
        acc      = tp[cls] / gt_count * 100 if gt_count > 0 else 0.0
        print(f"    {cls:<16}  {tp[cls]:>3} / {gt_count:<3}  ({acc:.1f}%)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# EPS CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix_eps(confusion: defaultdict, output_path: str):
    """
    Saves a recall-normalized confusion matrix as a publication-ready EPS file.
    Each row sums to 100 % (i.e. percentage of GT instances per class).
    Raw counts are printed inside each cell alongside the percentage.
    """
    n = len(CLASSES)

    # Build raw count matrix
    counts = np.zeros((n, n), dtype=int)
    for i, gt_cls in enumerate(CLASSES):
        for j, pr_cls in enumerate(CLASSES):
            counts[i, j] = confusion[gt_cls].get(pr_cls, 0)

    # Row-normalize to percentages
    row_sums = counts.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1   # avoid division by zero
    norm = counts / row_sums * 100.0

    # ── Figure setup ─────────────────────────────────────────────────────────
    matplotlib.rcParams.update({
        "font.family":      "serif",
        "font.size":        11,
        "axes.linewidth":   0.8,
        "pdf.fonttype":     42,   # embed fonts (also respected by EPS backend)
        "ps.fonttype":      42,
    })

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (%)", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))

    # Tick labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(CLASSES, fontsize=10)

    ax.set_xlabel("Predicted", fontsize=11, labelpad=8)
    ax.set_ylabel("Ground Truth", fontsize=11, labelpad=8)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)

    # Cell annotations: percentage + raw count
    thresh = 50.0   # switch text colour for readability
    for i in range(n):
        for j in range(n):
            pct = norm[i, j]
            cnt = counts[i, j]
            color = "white" if pct > thresh else "black"
            ax.text(
                j, i,
                f"{pct:.1f}%\n({cnt})",
                ha="center", va="center",
                fontsize=9, color=color,
                fontweight="bold" if i == j else "normal",
            )

    fig.tight_layout()
    fig.savefig(output_path, format="eps", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved → {output_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    all_results = []   # accumulates records from every video

    for idx, (pipeline_path, gt_path) in enumerate(VIDEO_PAIRS, start=1):
        pipeline_tracks, gt = load(pipeline_path, gt_path)
        results = collect_results(pipeline_tracks, gt)
        skipped = sum(1 for v in gt.values() if v is None)

        all_results.extend(results)

        if PRINT_PER_VIDEO:
            label = f"Video {idx}  —  {Path(gt_path).name}"
            print_report(label, results, len(gt), skipped)

    # ── Global report ─────────────────────────────────────────────────────────
    total_gt      = 0
    total_skipped = 0
    for _, gt_path in VIDEO_PAIRS:
        with open(gt_path, "r") as f:
            gt = json.load(f)
        total_gt      += len(gt)
        total_skipped += sum(1 for v in gt.values() if v is None)

    print_report(
        f"GLOBAL  —  {len(VIDEO_PAIRS)} videos combined",
        all_results,
        total_gt,
        total_skipped,
    )

    # ── EPS confusion matrix from global data ─────────────────────────────────
    _, _, _, global_confusion = aggregate(all_results)
    save_confusion_matrix_eps(global_confusion, CONFUSION_MATRIX_EPS)


if __name__ == "__main__":
    main()
