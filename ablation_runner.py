"""
ablation_runner.py
==================
Runs the accident-detection pipeline in 4 ablation configurations,
collects metrics from each run, and produces a side-by-side comparison
(text table + PNG plots + JSON summary).

Usage:
    python ablation_runner.py                     # uses Config.VIDEO_PATH
    python ablation_runner.py --video path/to.mp4
    python ablation_runner.py --modes full kinematic   # subset of modes

Configurations
--------------
    baseline   — YOLOv8 + DeepSORT + Euclidean distance trigger (80 px)
    kinematic  — YOLOv8 + DeepSORT + TTC + Cosine Similarity filter
    lstm_only  — YOLOv8 + DeepSORT + MotionLSTM predictor
    full       — YOLOv8 + DeepSORT + LSTM + TTC + ResNet50 (default)

Output (under logs/ablation/<run_timestamp>/)
---------------------------------------------
    results.json            — raw per-mode metrics
    comparison_table.txt    — printable metric table
    quality_comparison.png  — Precision / Recall / F1 / AUC bars
    cost_comparison.png     — CNN inferences / FPS / memory bars
    incident_comparison.png — incidents detected / false-alarm rate
    ablation_summary.png    — combined 3×2 dashboard
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ── matplotlib — set non-interactive backend before any other mpl import ──────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ═════════════════════════════════════════════════════════════════════════════
#  Mode definitions
# ═════════════════════════════════════════════════════════════════════════════

ABLATION_MODES: list[tuple[str, str]] = [
    ("baseline",  "Baseline\n(Distance Only)"),
    ("kinematic", "Kinematic Filter\n(TTC + Cosine)"),
    ("lstm_only", "AI Predictive\n(LSTM Only)"),
    ("full",      "Full Hybrid\nArchitecture"),
]

MODE_COLORS = {
    "baseline":  "#6e7681",   # grey
    "kinematic": "#58a6ff",   # blue
    "lstm_only": "#d2a8ff",   # purple
    "full":      "#3fb950",   # green
}

DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
GRID_CLR  = "#21262d"
TEXT_CLR  = "#c9d1d9"
RED       = "#f78166"
ORANGE    = "#ffa657"


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _fresh_deepsort():
    """Instantiate a clean DeepSort tracker (no stale state between runs)."""
    from deep_sort_realtime.deepsort_tracker import DeepSort
    return DeepSort(
        max_age=45, n_init=4,
        max_iou_distance=0.7, max_cosine_distance=0.3,
        nn_budget=30, embedder="mobilenet",
        half=True, nms_max_overlap=0.4,
    )


def _safe_get(d: dict, *keys, default=0.0) -> Any:
    """Nested dict access with fallback default."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d if d is not None else default


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ═════════════════════════════════════════════════════════════════════════════
#  Single-mode runner
# ═════════════════════════════════════════════════════════════════════════════

def _run_mode(mode_id: str, mode_label: str, console_mod, Config, output_root: str) -> dict:
    """
    Configure, execute, and collect results for one ablation mode.
    Returns a dict with raw metrics and computed summary values.
    """
    mode_dir = os.path.join(output_root, mode_id)
    os.makedirs(mode_dir, exist_ok=True)
    for sub in ("output", "output/clip", "artifacts", "resnet_crops"):
        os.makedirs(os.path.join(mode_dir, sub), exist_ok=True)

    # ── Override Config for this mode ────────────────────────────────────────
    Config.ABLATION_MODE    = mode_id
    Config.METRICS_FILE     = os.path.join(mode_dir, "metrics.json")
    Config.OUTPUT_DIR       = os.path.join(mode_dir, "output")
    Config.OUTPUT_DIR_CLIP  = os.path.join(mode_dir, "output", "clip")
    Config.ARTIFACTS_PATH   = os.path.join(mode_dir, "artifacts")
    Config.RESNET_CROPS_DIR = os.path.join(mode_dir, "resnet_crops")

    # Clean up ResNet crops dir (module-level init already ran once at import)
    if os.path.exists(Config.RESNET_CROPS_DIR):
        shutil.rmtree(Config.RESNET_CROPS_DIR)
    os.makedirs(Config.RESNET_CROPS_DIR, exist_ok=True)

    # Fresh DeepSort — crucial: resets all track state between runs
    console_mod.deepsort = _fresh_deepsort()

    # ── Run ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        console_mod.accident_detection(Config.VIDEO_PATH)
        elapsed = time.time() - t0
        status = "ok"
    except Exception as exc:
        elapsed = time.time() - t0
        status = f"ERROR: {exc}"
        traceback.print_exc()
        print(f"  ⚠️  Mode '{mode_id}' failed: {exc}")

    # ── Read results ─────────────────────────────────────────────────────────
    metrics   = _load_json(Config.METRICS_FILE)
    video_name = os.path.splitext(os.path.basename(Config.VIDEO_PATH))[0]
    eval_path  = os.path.join(mode_dir, "eval", video_name, "eval_summary.json")
    eval_data  = _load_json(eval_path)

    return {
        "mode_id":    mode_id,
        "mode_label": mode_label,
        "status":     status,
        "wall_time_s": round(elapsed, 1),
        "metrics":    metrics,
        "eval":       eval_data,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Extract comparison values
# ═════════════════════════════════════════════════════════════════════════════

def _extract(run: dict) -> dict:
    """Pull a flat dict of comparison-ready floats from a run result."""
    m = run["metrics"]
    e = run["eval"]
    proxy = _safe_get(e, "proxy_metrics", default={})

    return {
        # Detection quality
        "precision":    _safe_get(m, "evaluation", "precision"),
        "recall":       _safe_get(m, "evaluation", "recall"),
        "f1":           _safe_get(m, "evaluation", "f1_score"),
        "accuracy":     _safe_get(m, "evaluation", "accuracy"),
        "roc_auc":      _safe_get(m, "roc_data", "auc"),
        "pr_auc":       _safe_get(m, "precision_recall_data", "average_precision"),
        "mota":         _safe_get(m, "tracking_evaluation", "MOTA"),
        "idf1":         _safe_get(m, "tracking_evaluation", "IDF1"),
        # Incident sensitivity
        "unique_incidents":  _safe_get(m, "accidents", "unique_incidents"),
        "collision_warnings": _safe_get(m, "accidents", "collision_warnings"),
        "sudden_stops":      _safe_get(m, "accidents", "sudden_stops"),
        # Computational cost
        "cnn_inferences":  _safe_get(m, "detection", "cnn", "total_inferences"),
        "avg_fps":         _safe_get(m, "processing", "avg_fps"),
        "memory_mb":       _safe_get(m, "performance", "memory_usage_mb"),
        "p95_latency_ms":  _safe_get(m, "latency", "p95_ms"),
        # False alarm
        "false_alarm_rate": _safe_get(proxy, "system", "false_alarm_rate_proxy"),
        "id_switches":      _safe_get(proxy, "deepsort", "id_switches"),
        # LSTM-specific
        "lstm_avg_risk":    _safe_get(proxy, "lstm", "avg_risk_score"),
        "lstm_high_risk_pct": _safe_get(proxy, "lstm", "high_risk_pct_07"),
        # Wall time
        "wall_time_s": run["wall_time_s"],
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Text comparison table
# ═════════════════════════════════════════════════════════════════════════════

def _print_table(runs: list[dict], extracted: list[dict], out_path: str):
    headers = ["Metric"] + [r["mode_id"] for r in runs]
    col_w = 22

    ROWS = [
        ("--- Detection Quality ---", None),
        ("Precision",             "precision"),
        ("Recall",                "recall"),
        ("F1-Score",              "f1"),
        ("Accuracy",              "accuracy"),
        ("ROC-AUC",               "roc_auc"),
        ("PR-AUC",                "pr_auc"),
        ("--- Tracking ---", None),
        ("MOTA",                  "mota"),
        ("IDF1",                  "idf1"),
        ("ID Switches",           "id_switches"),
        ("--- Incidents ---", None),
        ("Unique Incidents",      "unique_incidents"),
        ("Collision Warnings",    "collision_warnings"),
        ("False Alarm Rate",      "false_alarm_rate"),
        ("--- Performance ---", None),
        ("CNN Inferences",        "cnn_inferences"),
        ("Avg FPS",               "avg_fps"),
        ("P95 Latency (ms)",      "p95_latency_ms"),
        ("Memory (MB)",           "memory_mb"),
        ("Wall Time (s)",         "wall_time_s"),
        ("--- LSTM (if on) ---", None),
        ("LSTM Avg Risk",         "lstm_avg_risk"),
        ("LSTM High-Risk %",      "lstm_high_risk_pct"),
    ]

    sep = "=" * (col_w + len(runs) * (col_w + 1))
    lines = []
    lines.append(sep)
    lines.append("  ABLATION STUDY — COMPARISON TABLE")
    lines.append(sep)
    lines.append("")

    # Header
    hdr = f"{'Metric':<{col_w}}" + "".join(f"{h:^{col_w}}" for h in headers[1:])
    lines.append(hdr)
    lines.append("-" * len(hdr))

    def _fmt(v, key):
        if key is None:
            return ""
        if isinstance(v, float):
            if key in ("cnn_inferences", "unique_incidents", "collision_warnings",
                       "id_switches", "sudden_stops"):
                return f"{int(v):>10}"
            if key in ("false_alarm_rate", "lstm_high_risk_pct"):
                return f"{v*100:>9.3f}%"
            if key == "avg_fps":
                return f"{v:>9.1f}"
            if key in ("wall_time_s", "p95_latency_ms", "memory_mb"):
                return f"{v:>9.1f}"
            return f"{v:>9.4f}"
        return str(v)

    for label, key in ROWS:
        if key is None:
            lines.append("")
            lines.append(f"  {label}")
            continue
        row = f"{label:<{col_w}}"
        vals = [e[key] for e in extracted]
        # Highlight best value
        fmt_vals = [_fmt(v, key) for v in vals]
        row += "".join(f"{fv:^{col_w}}" for fv in fmt_vals)
        lines.append(row)

    lines.append("")
    lines.append(sep)

    text = "\n".join(lines)
    print(text)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n  📋 Table saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ═════════════════════════════════════════════════════════════════════════════

def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_CLR)
    ax.grid(axis="y", color=GRID_CLR, lw=0.6, alpha=0.7)
    ax.title.set_color(TEXT_CLR)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)


def _bar_group(ax, categories: list[str], groups: dict[str, list[float]],
               mode_ids: list[str], title: str, ylabel: str,
               ylim: tuple | None = None, pct: bool = False):
    """Grouped bar chart: each category has one bar per mode."""
    n_cat   = len(categories)
    n_modes = len(mode_ids)
    x       = np.arange(n_cat)
    width   = 0.8 / n_modes
    offsets = np.linspace(-(n_modes - 1) * width / 2,
                          (n_modes - 1) * width / 2, n_modes)

    for offset, mode_id in zip(offsets, mode_ids):
        vals = groups[mode_id]
        display = [v * 100 if pct else v for v in vals]
        bars = ax.bar(x + offset, display, width * 0.9,
                      color=MODE_COLORS[mode_id], alpha=0.85, label=mode_id)
        for bar, dv in zip(bars, display):
            if dv > 0:
                label = f"{dv:.1f}{'%' if pct else ''}"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.002 if pct else dv * 0.02),
                        label, ha="center", va="bottom",
                        color=TEXT_CLR, fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8, color=TEXT_CLR)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=7, framealpha=0, labelcolor=TEXT_CLR,
              loc="upper right")


def _save_quality_plot(extracted: list[dict], mode_ids: list[str], out_dir: str):
    """Plot 1 — Detection quality metrics."""
    fig = plt.figure(figsize=(16, 6), facecolor=DARK_BG)
    fig.suptitle("Detection Quality Comparison", color="white",
                 fontsize=14, fontweight="bold", y=1.0)
    gs = gridspec.GridSpec(1, 3, wspace=0.35)

    # 1a. Precision / Recall / F1
    ax = fig.add_subplot(gs[0, 0])
    _style_ax(ax)
    cats = ["Precision", "Recall", "F1-Score"]
    groups = {
        mid: [e["precision"], e["recall"], e["f1"]]
        for mid, e in zip(mode_ids, extracted)
    }
    _bar_group(ax, cats, groups, mode_ids, "Precision / Recall / F1", "Score", (0, 1.15))

    # 1b. ROC-AUC / PR-AUC
    ax = fig.add_subplot(gs[0, 1])
    _style_ax(ax)
    cats = ["ROC-AUC", "PR-AUC"]
    groups = {
        mid: [e["roc_auc"], e["pr_auc"]]
        for mid, e in zip(mode_ids, extracted)
    }
    _bar_group(ax, cats, groups, mode_ids, "ROC-AUC / PR-AUC", "Score", (0, 1.15))

    # 1c. MOTA / IDF1
    ax = fig.add_subplot(gs[0, 2])
    _style_ax(ax)
    cats = ["MOTA", "IDF1"]
    groups = {
        mid: [e["mota"], e["idf1"]]
        for mid, e in zip(mode_ids, extracted)
    }
    _bar_group(ax, cats, groups, mode_ids, "Tracking Quality (MOTA / IDF1)", "Score")

    plt.tight_layout()
    p = os.path.join(out_dir, "quality_comparison.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  📊 quality_comparison.png saved")


def _save_cost_plot(extracted: list[dict], mode_ids: list[str], out_dir: str):
    """Plot 2 — Computational cost."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK_BG)
    fig.suptitle("Computational Cost Comparison", color="white",
                 fontsize=14, fontweight="bold")

    mode_labels = [m.replace("\n", " ") for m in mode_ids]
    colors = [MODE_COLORS[m] for m in mode_ids]

    # CNN inferences
    ax = axes[0]; _style_ax(ax)
    vals = [e["cnn_inferences"] for e in extracted]
    bars = ax.bar(mode_labels, vals, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{int(v):,}", ha="center", color=TEXT_CLR, fontsize=8)
    ax.set_title("CNN Inferences Total", fontsize=10)
    ax.set_ylabel("Count")

    # Avg FPS
    ax = axes[1]; _style_ax(ax)
    vals = [e["avg_fps"] for e in extracted]
    bars = ax.bar(mode_labels, vals, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{v:.1f}", ha="center", color=TEXT_CLR, fontsize=8)
    ax.set_title("Average Processing FPS", fontsize=10)
    ax.set_ylabel("FPS ↑ better")

    # P95 Latency
    ax = axes[2]; _style_ax(ax)
    vals = [e["p95_latency_ms"] for e in extracted]
    bars = ax.bar(mode_labels, vals, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{v:.0f} ms", ha="center", color=TEXT_CLR, fontsize=8)
    ax.set_title("P95 Frame Latency", fontsize=10)
    ax.set_ylabel("ms ↓ better")

    plt.tight_layout()
    p = os.path.join(out_dir, "cost_comparison.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  📊 cost_comparison.png saved")


def _save_incident_plot(extracted: list[dict], mode_ids: list[str], out_dir: str):
    """Plot 3 — Incident sensitivity and false alarm rate."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=DARK_BG)
    fig.suptitle("Incident Detection & False Alarm Analysis", color="white",
                 fontsize=14, fontweight="bold")

    mode_labels = [m for m in mode_ids]
    colors = [MODE_COLORS[m] for m in mode_ids]

    # Unique incidents detected
    ax = axes[0]; _style_ax(ax)
    vals = [e["unique_incidents"] for e in extracted]
    bars = ax.bar(mode_labels, vals, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{int(v)}", ha="center", color=TEXT_CLR, fontsize=10, fontweight="bold")
    ax.set_title("Unique Incidents Detected", fontsize=10)
    ax.set_ylabel("Count ↑ better")

    # False alarm rate
    ax = axes[1]; _style_ax(ax)
    vals = [e["false_alarm_rate"] * 100 for e in extracted]
    bars = ax.bar(mode_labels, vals, color=[RED if v > 5 else ORANGE if v > 2 else MODE_COLORS[m]
                                            for m, v in zip(mode_ids, vals)],
                  alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}%", ha="center", color=TEXT_CLR, fontsize=8)
    ax.set_title("False Alarm Rate (proxy)", fontsize=10)
    ax.set_ylabel("% of frames ↓ better")

    # Collision warnings (kinematic signal)
    ax = axes[2]; _style_ax(ax)
    vals = [e["collision_warnings"] for e in extracted]
    bars = ax.bar(mode_labels, vals, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{int(v)}", ha="center", color=TEXT_CLR, fontsize=8)
    ax.set_title("Kinematic Collision Warnings", fontsize=10)
    ax.set_ylabel("Count")

    plt.tight_layout()
    p = os.path.join(out_dir, "incident_comparison.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  📊 incident_comparison.png saved")


def _save_dashboard(extracted: list[dict], mode_ids: list[str], out_dir: str):
    """Plot 4 — Combined 3×2 summary dashboard."""
    fig = plt.figure(figsize=(20, 12), facecolor=DARK_BG)
    fig.suptitle("Ablation Study — Full Dashboard",
                 color="white", fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, hspace=0.50, wspace=0.38)

    mode_labels_short = [m.split("\n")[0] for m in mode_ids]
    colors = [MODE_COLORS[m] for m in mode_ids]

    def _simple_bar(ax, vals, title, ylabel, color_override=None, fmt=".3f", suffix=""):
        c = color_override or colors
        bars = ax.bar(mode_labels_short, vals, color=c, alpha=0.85, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01 if max(vals) > 0 else 0.01,
                    f"{v:{fmt}}{suffix}", ha="center", color=TEXT_CLR, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        _style_ax(ax)

    # R1C1 — F1 Score
    ax = fig.add_subplot(gs[0, 0])
    _simple_bar(ax, [e["f1"] for e in extracted],
                "F1-Score (↑ better)", "Score", fmt=".4f")
    ax.set_ylim(0, 1.15)

    # R1C2 — ROC-AUC
    ax = fig.add_subplot(gs[0, 1])
    _simple_bar(ax, [e["roc_auc"] for e in extracted],
                "ROC-AUC (↑ better)", "AUC", fmt=".4f")
    ax.set_ylim(0, 1.15)

    # R1C3 — False Alarm Rate
    ax = fig.add_subplot(gs[0, 2])
    fa_vals = [e["false_alarm_rate"] * 100 for e in extracted]
    fa_colors = [RED if v > 5 else ORANGE if v > 2 else MODE_COLORS[m]
                 for m, v in zip(mode_ids, fa_vals)]
    _simple_bar(ax, fa_vals, "False Alarm Rate % (↓ better)", "% of frames",
                color_override=fa_colors, fmt=".3f", suffix="%")

    # R2C1 — CNN inferences
    ax = fig.add_subplot(gs[1, 0])
    cnn_vals = [e["cnn_inferences"] for e in extracted]
    _simple_bar(ax, cnn_vals, "CNN Inferences (↓ = cheaper)", "Count", fmt=".0f")

    # R2C2 — Avg FPS
    ax = fig.add_subplot(gs[1, 1])
    _simple_bar(ax, [e["avg_fps"] for e in extracted],
                "Processing Speed (↑ better)", "FPS", fmt=".1f")

    # R2C3 — Unique incidents
    ax = fig.add_subplot(gs[1, 2])
    _simple_bar(ax, [e["unique_incidents"] for e in extracted],
                "Unique Incidents Detected", "Count", fmt=".0f")

    # Legend
    from matplotlib.patches import Patch
    patches = [Patch(facecolor=MODE_COLORS[m], label=m, alpha=0.85) for m in mode_ids]
    fig.legend(handles=patches, loc="lower center", ncol=len(mode_ids),
               fontsize=10, framealpha=0, labelcolor=TEXT_CLR,
               bbox_to_anchor=(0.5, -0.02))

    plt.savefig(os.path.join(out_dir, "ablation_summary.png"),
                dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  📊 ablation_summary.png saved")


# ═════════════════════════════════════════════════════════════════════════════
#  Main orchestrator
# ═════════════════════════════════════════════════════════════════════════════

def run_ablations(selected_modes: list[str] | None = None, video_path: str | None = None):
    """
    Execute all (or selected) ablation modes and generate comparison artefacts.

    Parameters
    ----------
    selected_modes : list of mode IDs to run; None = all four modes
    video_path     : override Config.VIDEO_PATH; None = use config default
    """
    # ── Late import so models load after Config might be patched by CLI ───────
    from modules.config.config import Config
    import console  # triggers module-level model loading

    if video_path:
        Config.VIDEO_PATH = video_path

    # ── Prepare output root ───────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join(Config.LOG_DIR, "ablation", ts)
    os.makedirs(output_root, exist_ok=True)
    print(f"\n  Ablation output root: {output_root}")

    # Save original config values to restore after all runs
    _orig = {k: getattr(Config, k) for k in (
        "ABLATION_MODE", "METRICS_FILE", "OUTPUT_DIR", "OUTPUT_DIR_CLIP",
        "ARTIFACTS_PATH", "RESNET_CROPS_DIR",
    )}

    modes_to_run = [
        (mid, label) for mid, label in ABLATION_MODES
        if selected_modes is None or mid in selected_modes
    ]

    all_runs: list[dict] = []
    mode_ids: list[str] = []

    # ════════════════════════════════════════════════════════════════
    for idx, (mode_id, mode_label) in enumerate(modes_to_run, 1):
        print(f"\n{'═'*65}")
        print(f"  [{idx}/{len(modes_to_run)}] MODE: {mode_label.replace(chr(10), ' ')}")
        print(f"{'═'*65}")

        run_result = _run_mode(mode_id, mode_label, console, Config, output_root)
        all_runs.append(run_result)
        mode_ids.append(mode_id)

        print(f"  ✓ Completed in {run_result['wall_time_s']:.1f}s  |  status={run_result['status']}")
    # ════════════════════════════════════════════════════════════════

    # Restore original Config
    for k, v in _orig.items():
        setattr(Config, k, v)

    if not all_runs:
        print("No runs completed.")
        return {}

    # ── Extract comparison values ─────────────────────────────────────────────
    extracted = [_extract(r) for r in all_runs]

    # ── Save raw results JSON ─────────────────────────────────────────────────
    results_path = os.path.join(output_root, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "video": Config.VIDEO_PATH,
                    "timestamp": ts,
                    "modes": mode_ids,
                    "descriptions": {
                        mid: Config.ABLATION_CONFIGS.get(mid, mid)
                        for mid in mode_ids
                    },
                },
                "runs": {
                    r["mode_id"]: {
                        "status": r["status"],
                        "wall_time_s": r["wall_time_s"],
                        "summary": extracted[i],
                    }
                    for i, r in enumerate(all_runs)
                },
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"\n  💾 Raw results: {results_path}")

    # ── Text table ────────────────────────────────────────────────────────────
    table_path = os.path.join(output_root, "comparison_table.txt")
    _print_table(all_runs, extracted, table_path)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generating plots…")
    _save_quality_plot(extracted, mode_ids, output_root)
    _save_cost_plot(extracted, mode_ids, output_root)
    _save_incident_plot(extracted, mode_ids, output_root)
    _save_dashboard(extracted, mode_ids, output_root)

    print(f"\n{'═'*65}")
    print(f"  ABLATION STUDY COMPLETE")
    print(f"  Output: {output_root}")
    print(f"{'═'*65}\n")

    return {mid: extracted[i] for i, mid in enumerate(mode_ids)}


# ═════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description="Run ablation study across 4 detection configurations")
    p.add_argument("--video",  default=None, help="Override video path (default: Config.VIDEO_PATH)")
    p.add_argument("--modes",  nargs="+",
                   choices=["baseline", "kinematic", "lstm_only", "full"],
                   default=None,
                   help="Subset of modes to run (default: all 4)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        run_ablations(selected_modes=args.modes, video_path=args.video)
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[FATAL] {exc}")
        traceback.print_exc()
        sys.exit(1)
