"""
results_aggregator.py
=====================
Scans ALL ablation run directories and per-video evaluation data produced by
ablation_runner.py + console.py, merges them into one dataset, and writes:

    evaluation_report.xlsx   — 5-sheet workbook (raw data, per-video,
                               per-mode, correlation heatmap, metadata)
    evaluation_report.png    — 4-panel summary dashboard (PNG)
    evaluation_summary.txt   — human-readable text report

Usage
-----
    python results_aggregator.py                          # auto-discover log dir
    python results_aggregator.py --logs path/to/logs     # explicit log root
    python results_aggregator.py --out  path/to/report   # override output dir

Directory layout expected
-------------------------
    <logs_root>/
        ablation/
            <timestamp1>/
                results.json
                <mode_id>/
                    metrics.json
                    eval/<video_name>/eval_summary.json
            <timestamp2>/
                ...
        evaluation/                  (runs from plain console.py, no ablation)
            <video_name>/
                eval_summary.json
        metrics_summary.json         (last plain run)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from openpyxl import Workbook
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side, GradientFill
)
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule, DataBarRule
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.series import SeriesLabel


# ═══════════════════════════════════════════════════════════════════════════
#  Palette & constants
# ═══════════════════════════════════════════════════════════════════════════

DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
GRID_CLR  = "#21262d"
TEXT_CLR  = "#c9d1d9"

MODE_COLORS_HEX = {
    "baseline":  "#6e7681",
    "kinematic": "#58a6ff",
    "lstm_only": "#d2a8ff",
    "full":      "#3fb950",
}
MODE_COLORS_MPL = {k: v for k, v in MODE_COLORS_HEX.items()}

MODE_LABELS = {
    "baseline":  "Baseline (Distance)",
    "kinematic": "Kinematic (TTC+Cos)",
    "lstm_only": "AI Predictive (LSTM)",
    "full":      "Full Hybrid",
}

# Excel palette
XL_HEADER_FILL = "1F3864"   # dark navy
XL_SUBHDR_FILL = "2E4B8B"   # mid navy
XL_ALT_ROW     = "EEF2FF"   # light lavender
XL_GREEN        = "C6EFCE"
XL_RED          = "FFC7CE"
XL_ORANGE       = "FFEB9C"
XL_WHITE        = "FFFFFF"

THIN = Side(style="thin", color="BFBFBF")
MEDIUM = Side(style="medium", color="4472C4")
THIN_BORDER  = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
MEDIUM_BORDER = Border(left=MEDIUM, right=MEDIUM, top=MEDIUM, bottom=MEDIUM)


# ═══════════════════════════════════════════════════════════════════════════
#  Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════

def _safe(d: Any, *keys, default=0.0) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d if d is not None else default


def _load(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _fmt_pct(v: float) -> str:
    return f"{v*100:.2f}%"


# ═══════════════════════════════════════════════════════════════════════════
#  Discovery
# ═══════════════════════════════════════════════════════════════════════════

def discover_runs(logs_root: str) -> list[dict]:
    """
    Walk the logs directory tree and return a flat list of run dicts, one per
    (video, mode) pair. Each dict contains all extractable metrics.
    """
    runs: list[dict] = []
    logs = Path(logs_root)

    # ── A: ablation/<timestamp>/<mode_id>/metrics.json ────────────────────
    ablation_root = logs / "ablation"
    if ablation_root.exists():
        for ts_dir in sorted(ablation_root.iterdir()):
            if not ts_dir.is_dir():
                continue
            meta = _load(str(ts_dir / "results.json"))
            video = _safe(meta, "meta", "video", default="unknown")
            video_name = Path(video).stem if video != "unknown" else "unknown"

            for mode_dir in sorted(ts_dir.iterdir()):
                if not mode_dir.is_dir():
                    continue
                mode_id = mode_dir.name
                if mode_id not in ("baseline", "kinematic", "lstm_only", "full"):
                    continue

                metrics  = _load(str(mode_dir / "metrics.json"))
                eval_dir = mode_dir / "eval" / video_name
                eval_sum = _load(str(eval_dir / "eval_summary.json"))

                run_meta = _safe(meta, "runs", mode_id, default={})
                runs.append(_build_run(
                    ablation_ts  = ts_dir.name,
                    video        = video,
                    video_name   = video_name,
                    mode_id      = mode_id,
                    metrics      = metrics,
                    eval_summary = eval_sum,
                    wall_time    = _safe(run_meta, "wall_time_s"),
                    status       = _safe(run_meta, "status", default="ok"),
                ))

    # ── B: evaluation/<video_name>/eval_summary.json  (plain console runs) ─
    eval_root = logs / "evaluation"
    if eval_root.exists():
        for vid_dir in sorted(eval_root.iterdir()):
            if not vid_dir.is_dir():
                continue
            eval_sum = _load(str(vid_dir / "eval_summary.json"))
            if not eval_sum:
                continue
            video = _safe(eval_sum, "meta", "video_path", default=vid_dir.name)
            video_name = vid_dir.name
            metrics = _load(str(logs / "metrics_summary.json"))
            runs.append(_build_run(
                ablation_ts  = "plain_run",
                video        = video,
                video_name   = video_name,
                mode_id      = "full",
                metrics      = metrics,
                eval_summary = eval_sum,
                wall_time    = 0.0,
                status       = "ok",
            ))

    return runs


def _build_run(ablation_ts, video, video_name, mode_id,
               metrics, eval_summary, wall_time, status) -> dict:
    """Flatten one (video, mode) run into a single dict of metrics."""
    proxy = _safe(eval_summary, "proxy_metrics", default={})
    ev    = _safe(metrics, "evaluation", default={})
    te    = _safe(metrics, "tracking_evaluation", default={})
    acc   = _safe(metrics, "accidents", default={})
    cnn   = _safe(metrics, "detection", "cnn", default={})
    yolo  = _safe(metrics, "detection", "yolo", default={})
    lat   = _safe(metrics, "latency", default={})
    perf  = _safe(metrics, "performance", default={})
    lstm  = _safe(metrics, "lstm", default={})
    roc   = _safe(metrics, "roc_data", default={})
    pr    = _safe(metrics, "precision_recall_data", default={})
    proc  = _safe(metrics, "processing", default={})

    ds   = _safe(proxy, "deepsort", default={})
    yp   = _safe(proxy, "yolo",     default={})
    rn   = _safe(proxy, "resnet50", default={})
    lp   = _safe(proxy, "lstm",     default={})
    sp   = _safe(proxy, "system",   default={})

    return {
        # Identification
        "ablation_ts":  ablation_ts,
        "video":        video,
        "video_name":   video_name,
        "mode_id":      mode_id,
        "mode_label":   MODE_LABELS.get(mode_id, mode_id),
        "status":       str(status),
        "wall_time_s":  float(wall_time),

        # ── Detection quality ────────────────────────────────────────────────
        "precision":    float(_safe(ev, "precision")),
        "recall":       float(_safe(ev, "recall")),
        "f1_score":     float(_safe(ev, "f1_score")),
        "accuracy":     float(_safe(ev, "accuracy")),
        "roc_auc":      float(_safe(roc, "auc")),
        "pr_auc":       float(_safe(pr, "average_precision")),
        "map_50":       float(_safe(ev, "mAP_50")),
        "map_75":       float(_safe(ev, "mAP_75")),
        "map":          float(_safe(ev, "mAP")),

        # ── Tracking quality ─────────────────────────────────────────────────
        "mota":         float(_safe(te, "MOTA")),
        "idf1":         float(_safe(te, "IDF1")),
        "id_switches":  int(_safe(te, "id_switches")),
        "fp_tracking":  int(_safe(te, "false_positives")),
        "fn_tracking":  int(_safe(te, "false_negatives")),

        # ── Incident counts ──────────────────────────────────────────────────
        "unique_incidents":    int(_safe(acc, "unique_incidents")),
        "collision_warnings":  int(_safe(acc, "collision_warnings")),
        "sudden_stops":        int(_safe(acc, "sudden_stops")),
        "false_pos_filtered":  int(_safe(acc, "false_positives_filtered")),

        # ── YOLO ─────────────────────────────────────────────────────────────
        "yolo_detections":    int(_safe(yolo, "total_detections")),
        "yolo_avg_conf":      float(_safe(yolo, "avg_confidence")),
        "yolo_avg_conf_p":    float(_safe(yp, "avg_confidence")),
        "yolo_low_conf_pct":  float(_safe(yp, "low_conf_pct_06")),

        # ── CNN / ResNet50 ───────────────────────────────────────────────────
        "cnn_inferences":     int(_safe(cnn, "total_inferences")),
        "cnn_avg_time_ms":    float(_safe(cnn, "avg_inference_time_ms")),
        "cnn_p95_ms":         float(_safe(rn, "p95_inference_time_ms")),
        "cnn_high_conf":      int(_safe(cnn, "high_confidence_predictions")),
        "cnn_acc_triggers":   int(_safe(cnn, "accident_trigger_inferences")),
        "cnn_avg_score":      float(_safe(rn, "avg_score_all")),
        "cnn_score_sep":      float(_safe(rn, "score_separation") or 0),

        # ── DeepSort proxy ───────────────────────────────────────────────────
        "ds_total_tracks":    int(_safe(ds, "total_tracks")),
        "ds_id_switch_rate":  float(_safe(ds, "id_switch_rate")),
        "ds_avg_len":         float(_safe(ds, "avg_track_length_frames")),
        "ds_short_pct":       float(_safe(ds, "short_tracks_pct")),
        "ds_long_pct":        float(_safe(ds, "long_tracks_pct")),

        # ── LSTM ─────────────────────────────────────────────────────────────
        "lstm_enabled":       bool(_safe(lstm, "enabled")),
        "lstm_avg_risk":      float(_safe(lp, "avg_risk_score")),
        "lstm_high_risk_pct": float(_safe(lp, "high_risk_pct_07")),
        "lstm_kin_agree":     float(_safe(lp, "lstm_kinematic_agreement") or 0),

        # ── Performance / Latency ────────────────────────────────────────────
        "avg_fps":            float(_safe(proc, "avg_fps")),
        "total_frames":       int(_safe(proc, "total_frames")),
        "proc_time_s":        float(_safe(proc, "processing_time_seconds")),
        "memory_mb":          float(_safe(perf, "memory_usage_mb")),
        "p50_ms":             float(_safe(lat, "p50_ms")),
        "p95_ms":             float(_safe(lat, "p95_ms")),
        "p99_ms":             float(_safe(lat, "p99_ms")),

        # ── System proxy ─────────────────────────────────────────────────────
        "false_alarm_rate":   float(_safe(sp, "false_alarm_rate_proxy")),
        "accident_rate":      float(_safe(sp, "accident_rate")),
        "accident_frames":    int(_safe(sp, "accident_frames_count")),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Excel helpers
# ═══════════════════════════════════════════════════════════════════════════

def _hdr_cell(ws, row, col, value, level=1):
    c = ws.cell(row=row, column=col, value=value)
    if level == 1:
        c.fill = PatternFill("solid", fgColor=XL_HEADER_FILL)
        c.font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
    elif level == 2:
        c.fill = PatternFill("solid", fgColor=XL_SUBHDR_FILL)
        c.font = Font(bold=True, color="FFFFFF", name="Arial", size=9)
    else:
        c.fill = PatternFill("solid", fgColor="D9E1F2")
        c.font = Font(bold=True, color="1F3864", name="Arial", size=9)
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    c.border = THIN_BORDER
    return c


def _data_cell(ws, row, col, value, fmt=None, alt=False):
    c = ws.cell(row=row, column=col, value=value)
    c.font = Font(name="Arial", size=9)
    c.border = THIN_BORDER
    c.alignment = Alignment(horizontal="center", vertical="center")
    if alt:
        c.fill = PatternFill("solid", fgColor=XL_ALT_ROW)
    if fmt:
        c.number_format = fmt
    return c


def _set_col_widths(ws, widths: list[int]):
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _color_scale(ws, col_letter, start_row, end_row,
                 low="F8696B", mid="FFEB84", high="63BE7B"):
    rule = ColorScaleRule(
        start_type="min", start_color=low,
        mid_type="percentile", mid_value=50, mid_color=mid,
        end_type="max", end_color=high,
    )
    ws.conditional_formatting.add(
        f"{col_letter}{start_row}:{col_letter}{end_row}", rule
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Sheet 1 — Raw Data
# ═══════════════════════════════════════════════════════════════════════════

RAW_COLUMNS = [
    # (header, field_key, excel_fmt, col_width)
    ("Run",           "ablation_ts",    "@",        18),
    ("Video",         "video_name",     "@",        22),
    ("Mode",          "mode_label",     "@",        22),
    ("Status",        "status",         "@",         8),
    # Detection quality
    ("Precision",     "precision",      "0.0000",   11),
    ("Recall",        "recall",         "0.0000",   11),
    ("F1-Score",      "f1_score",       "0.0000",   11),
    ("Accuracy",      "accuracy",       "0.0000",   11),
    ("ROC-AUC",       "roc_auc",        "0.0000",   11),
    ("PR-AUC",        "pr_auc",         "0.0000",   11),
    ("mAP",           "map",            "0.0000",   10),
    ("mAP@0.5",       "map_50",         "0.0000",   10),
    ("mAP@0.75",      "map_75",         "0.0000",   10),
    # Tracking
    ("MOTA",          "mota",           "0.0000",   10),
    ("IDF1",          "idf1",           "0.0000",   10),
    ("ID Switches",   "id_switches",    "#,##0",    12),
    ("FP Tracking",   "fp_tracking",    "#,##0",    12),
    ("FN Tracking",   "fn_tracking",    "#,##0",    12),
    # Incidents
    ("Incidents",     "unique_incidents","#,##0",   10),
    ("Col. Warnings", "collision_warnings","#,##0", 14),
    ("Sudden Stops",  "sudden_stops",   "#,##0",    12),
    ("False Al. Rate","false_alarm_rate","0.000%",  14),
    # YOLO
    ("YOLO Dets",     "yolo_detections","#,##0",    11),
    ("YOLO Conf",     "yolo_avg_conf",  "0.0000",   11),
    # CNN
    ("CNN Infer.",    "cnn_inferences", "#,##0",    11),
    ("CNN P95 ms",    "cnn_p95_ms",     "0.0",      11),
    ("CNN Score Sep", "cnn_score_sep",  "0.0000",   13),
    # LSTM
    ("LSTM Avg Risk", "lstm_avg_risk",  "0.0000",   13),
    ("LSTM High%",    "lstm_high_risk_pct","0.0%",  12),
    ("LSTM↔Kin Agr",  "lstm_kin_agree", "0.0%",    13),
    # Performance
    ("Avg FPS",       "avg_fps",        "0.0",      10),
    ("P50 ms",        "p50_ms",         "0.0",      10),
    ("P95 ms",        "p95_ms",         "0.0",      10),
    ("Memory MB",     "memory_mb",      "#,##0",    11),
    ("Frames",        "total_frames",   "#,##0",    10),
]


def _sheet_raw(wb: Workbook, runs: list[dict]):
    ws = wb.create_sheet("Raw Data")
    ws.freeze_panes = "E2"

    # Header row
    for col, (hdr, _, _, _) in enumerate(RAW_COLUMNS, 1):
        _hdr_cell(ws, 1, col, hdr, level=1)

    ws.row_dimensions[1].height = 32

    for r, run in enumerate(runs, 2):
        alt = (r % 2 == 0)
        for col, (_, key, fmt, _) in enumerate(RAW_COLUMNS, 1):
            val = run.get(key, "")
            _data_cell(ws, r, col, val, fmt=fmt if fmt != "@" else None, alt=alt)

    # Conditional formatting on quality columns (E–M)
    n = len(runs) + 1
    for col_letter in ["E", "F", "G", "H", "I", "J", "K"]:
        _color_scale(ws, col_letter, 2, n)
    # Inverse scale for false alarm rate (lower = better)
    _color_scale(ws, "V", 2, n, low="63BE7B", high="F8696B")

    _set_col_widths(ws, [c[3] for c in RAW_COLUMNS])
    ws.auto_filter.ref = f"A1:{get_column_letter(len(RAW_COLUMNS))}{n}"
    return ws


# ═══════════════════════════════════════════════════════════════════════════
#  Sheet 2 — Per-Video Summary
# ═══════════════════════════════════════════════════════════════════════════

VIDEO_METRICS = [
    ("Precision",      "precision",     "0.0000", True),
    ("Recall",         "recall",        "0.0000", True),
    ("F1-Score",       "f1_score",      "0.0000", True),
    ("ROC-AUC",        "roc_auc",       "0.0000", True),
    ("PR-AUC",         "pr_auc",        "0.0000", True),
    ("MOTA",           "mota",          "0.0000", True),
    ("IDF1",           "idf1",          "0.0000", True),
    ("Incidents",      "unique_incidents","#,##0", True),
    ("Fa. Alarm %",    "false_alarm_rate","0.000%",False),
    ("Avg FPS",        "avg_fps",       "0.0",    True),
    ("CNN Infer.",     "cnn_inferences","#,##0",  False),
    ("Memory MB",      "memory_mb",     "#,##0",  False),
]


def _sheet_per_video(wb: Workbook, runs: list[dict]):
    ws = wb.create_sheet("Per-Video Summary")

    videos = sorted({r["video_name"] for r in runs})
    modes  = [m for m in ("baseline", "kinematic", "lstm_only", "full")
              if any(r["mode_id"] == m for r in runs)]

    # Title
    total_cols = 1 + len(modes) * len(VIDEO_METRICS)
    ws.merge_cells(start_row=1, start_column=1,
                   end_row=1, end_column=total_cols)
    t = ws.cell(1, 1, "Per-Video Metrics Summary — All Modes")
    t.font = Font(bold=True, color="FFFFFF", name="Arial", size=12)
    t.fill = PatternFill("solid", fgColor="1F3864")
    t.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 24

    # Mode sub-headers (row 2)
    col = 2
    ws.cell(2, 1, "Video").font = Font(bold=True, name="Arial", size=9)
    for mode in modes:
        label = MODE_LABELS.get(mode, mode)
        ws.merge_cells(start_row=2, start_column=col,
                       end_row=2, end_column=col + len(VIDEO_METRICS) - 1)
        c = ws.cell(2, col, label)
        c.fill = PatternFill("solid", fgColor=XL_SUBHDR_FILL)
        c.font = Font(bold=True, color="FFFFFF", name="Arial", size=9)
        c.alignment = Alignment(horizontal="center", vertical="center")
        col += len(VIDEO_METRICS)

    # Metric sub-headers (row 3)
    ws.cell(3, 1, "Video")
    _hdr_cell(ws, 3, 1, "Video", level=3)
    col = 2
    for _ in modes:
        for hdr, _, _, _ in VIDEO_METRICS:
            _hdr_cell(ws, 3, col, hdr, level=3)
            col += 1

    ws.row_dimensions[3].height = 28
    ws.freeze_panes = "B4"

    # Data rows
    run_map: dict[tuple, dict] = {
        (r["video_name"], r["mode_id"]): r for r in runs
    }

    for ri, video in enumerate(videos, 4):
        alt = (ri % 2 == 0)
        ws.cell(ri, 1, video)
        c = ws.cell(ri, 1)
        c.font = Font(bold=True, name="Arial", size=9)
        c.border = THIN_BORDER
        if alt:
            c.fill = PatternFill("solid", fgColor=XL_ALT_ROW)

        col = 2
        for mode in modes:
            run = run_map.get((video, mode), {})
            for _, key, fmt, higher_better in VIDEO_METRICS:
                val = run.get(key, None)
                _data_cell(ws, ri, col, val,
                           fmt=fmt if val is not None else None,
                           alt=alt)
                col += 1

    # Auto-fit columns
    widths = [22] + [11] * (len(modes) * len(VIDEO_METRICS))
    _set_col_widths(ws, widths)
    return ws


# ═══════════════════════════════════════════════════════════════════════════
#  Sheet 3 — Per-Mode Aggregate
# ═══════════════════════════════════════════════════════════════════════════

AGGREGATE_GROUPS = [
    ("Detection Quality", [
        ("Precision",          "precision",        "0.0000", True),
        ("Recall",             "recall",            "0.0000", True),
        ("F1-Score",           "f1_score",          "0.0000", True),
        ("Accuracy",           "accuracy",          "0.0000", True),
        ("ROC-AUC",            "roc_auc",           "0.0000", True),
        ("PR-AUC (AP)",        "pr_auc",            "0.0000", True),
        ("mAP",                "map",               "0.0000", True),
        ("mAP@0.5",            "map_50",            "0.0000", True),
        ("mAP@0.75",           "map_75",            "0.0000", True),
    ]),
    ("Tracking Quality", [
        ("MOTA",               "mota",              "0.0000", True),
        ("IDF1",               "idf1",              "0.0000", True),
        ("Avg ID Switches",    "id_switches",       "#,##0",  False),
        ("FP (Tracking)",      "fp_tracking",       "#,##0",  False),
        ("FN (Tracking)",      "fn_tracking",       "#,##0",  False),
        ("DS Avg Track Len",   "ds_avg_len",        "0.0",    True),
        ("DS Short Tracks %",  "ds_short_pct",      "0.0%",   False),
        ("DS Long Tracks %",   "ds_long_pct",       "0.0%",   True),
    ]),
    ("Incident Detection", [
        ("Avg Unique Incidents","unique_incidents", "#,##0",  True),
        ("Collision Warnings", "collision_warnings","#,##0",  True),
        ("Sudden Stops",       "sudden_stops",      "#,##0",  True),
        ("False Alarm Rate",   "false_alarm_rate",  "0.000%", False),
        ("Accident Rate",      "accident_rate",     "0.000%", True),
    ]),
    ("YOLO Performance", [
        ("Total Detections",   "yolo_detections",  "#,##0",  True),
        ("Avg Confidence",     "yolo_avg_conf",     "0.0000", True),
        ("Low-Conf Det. %",    "yolo_low_conf_pct", "0.0%",   False),
    ]),
    ("CNN / ResNet50", [
        ("Total Inferences",   "cnn_inferences",   "#,##0",  False),
        ("Avg Inference ms",   "cnn_avg_time_ms",  "0.0",    False),
        ("P95 Inference ms",   "cnn_p95_ms",       "0.0",    False),
        ("Avg CNN Score",      "cnn_avg_score",     "0.0000", False),
        ("Score Separation",   "cnn_score_sep",     "0.0000", True),
        ("Accident Triggers",  "cnn_acc_triggers",  "#,##0",  True),
    ]),
    ("LSTM (Predictive)", [
        ("Avg Risk Score",     "lstm_avg_risk",     "0.0000", False),
        ("High-Risk Tracks %", "lstm_high_risk_pct","0.0%",   False),
        ("LSTM↔Kin Agreement", "lstm_kin_agree",    "0.0%",   True),
    ]),
    ("System Performance", [
        ("Avg FPS",            "avg_fps",           "0.0",    True),
        ("P50 Latency ms",     "p50_ms",            "0.0",    False),
        ("P95 Latency ms",     "p95_ms",            "0.0",    False),
        ("P99 Latency ms",     "p99_ms",            "0.0",    False),
        ("Memory Usage MB",    "memory_mb",         "#,##0",  False),
        ("Wall Time s",        "wall_time_s",       "0.0",    False),
    ]),
]


def _sheet_per_mode(wb: Workbook, runs: list[dict]):
    ws = wb.create_sheet("Per-Mode Aggregate")

    modes = [m for m in ("baseline", "kinematic", "lstm_only", "full")
             if any(r["mode_id"] == m for r in runs)]

    # Group runs by mode
    by_mode: dict[str, list[dict]] = {m: [] for m in modes}
    for r in runs:
        if r["mode_id"] in by_mode:
            by_mode[r["mode_id"]].append(r)

    def _agg(vals: list[float]) -> tuple[float, float, float]:
        a = np.array([v for v in vals if v is not None], dtype=float)
        if not len(a):
            return (0.0, 0.0, 0.0)
        return float(np.mean(a)), float(np.min(a)), float(np.max(a))

    # Header row 1: title
    total_cols = 2 + len(modes) * 3
    ws.merge_cells(start_row=1, start_column=1,
                   end_row=1, end_column=total_cols)
    t = ws.cell(1, 1, "Per-Mode Aggregate — Mean / Min / Max across all videos")
    t.font = Font(bold=True, color="FFFFFF", name="Arial", size=12)
    t.fill = PatternFill("solid", fgColor="1F3864")
    t.alignment = Alignment(horizontal="center")
    ws.row_dimensions[1].height = 24

    # Header row 2: mode names
    ws.cell(2, 1, "Group")
    ws.cell(2, 2, "Metric")
    col = 3
    for mode in modes:
        label = MODE_LABELS.get(mode, mode)
        ws.merge_cells(start_row=2, start_column=col,
                       end_row=2, end_column=col + 2)
        c = ws.cell(2, col, label)
        c.fill = PatternFill("solid", fgColor=XL_SUBHDR_FILL)
        c.font = Font(bold=True, color="FFFFFF", name="Arial", size=9)
        c.alignment = Alignment(horizontal="center")
        col += 3

    # Header row 3: mean/min/max
    _hdr_cell(ws, 3, 1, "Group", 3)
    _hdr_cell(ws, 3, 2, "Metric", 3)
    col = 3
    for _ in modes:
        _hdr_cell(ws, 3, col,   "Mean",  3)
        _hdr_cell(ws, 3, col+1, "Min",   3)
        _hdr_cell(ws, 3, col+2, "Max",   3)
        col += 3

    ws.freeze_panes = "C4"

    row = 4
    for group_name, metrics_list in AGGREGATE_GROUPS:
        # Group label (spanning all rows of this group)
        group_start = row
        for mi, (m_label, m_key, m_fmt, higher_better) in enumerate(metrics_list):
            alt = (row % 2 == 0)
            # Group name only on first metric of group
            g = ws.cell(row, 1, group_name if mi == 0 else "")
            g.font = Font(bold=True, name="Arial", size=9, color="1F3864")
            g.fill = PatternFill("solid", fgColor="D9E1F2")
            g.border = THIN_BORDER
            g.alignment = Alignment(vertical="center", wrap_text=True)

            _data_cell(ws, row, 2, m_label, alt=alt)

            col = 3
            mode_means = []
            for mode in modes:
                vals = [r.get(m_key) for r in by_mode[mode] if r.get(m_key) is not None]
                mean, mn, mx = _agg(vals)
                mode_means.append(mean)
                for c_off, (v, lbl) in enumerate([(mean, "Mean"), (mn, "Min"), (mx, "Max")]):
                    cell = _data_cell(ws, row, col + c_off, round(v, 4), fmt=m_fmt, alt=alt)
                col += 3

            # Highlight best mean
            if mode_means and any(v > 0 for v in mode_means):
                best_idx = (np.argmax(mode_means) if higher_better
                            else np.argmin(mode_means))
                best_col = 3 + best_idx * 3
                best_cell = ws.cell(row, best_col)
                best_cell.fill = PatternFill("solid", fgColor=XL_GREEN)
                best_cell.font = Font(bold=True, name="Arial", size=9, color="375623")

            row += 1

        # Merge group cell vertically
        if len(metrics_list) > 1:
            ws.merge_cells(start_row=group_start, start_column=1,
                           end_row=row - 1, end_column=1)

    _set_col_widths(ws, [20, 24] + [13, 11, 11] * len(modes))
    return ws


# ═══════════════════════════════════════════════════════════════════════════
#  Sheet 4 — Head-to-Head Matrix
# ═══════════════════════════════════════════════════════════════════════════

KEY_METRICS_H2H = [
    ("F1-Score",        "f1_score",       True,  "0.0000"),
    ("ROC-AUC",         "roc_auc",        True,  "0.0000"),
    ("Recall",          "recall",         True,  "0.0000"),
    ("Precision",       "precision",      True,  "0.0000"),
    ("MOTA",            "mota",           True,  "0.0000"),
    ("IDF1",            "idf1",           True,  "0.0000"),
    ("Incidents Det.",  "unique_incidents",True, "#,##0"),
    ("False Alarm %",   "false_alarm_rate",False,"0.000%"),
    ("CNN Inferences",  "cnn_inferences", False, "#,##0"),
    ("Avg FPS",         "avg_fps",        True,  "0.0"),
    ("P95 Latency ms",  "p95_ms",         False, "0.0"),
    ("Memory MB",       "memory_mb",      False, "#,##0"),
    ("LSTM Avg Risk",   "lstm_avg_risk",  False, "0.0000"),
    ("Score Sep.",      "cnn_score_sep",  True,  "0.0000"),
    ("ID Switches",     "id_switches",    False, "#,##0"),
]


def _sheet_h2h(wb: Workbook, runs: list[dict]):
    ws = wb.create_sheet("Head-to-Head Matrix")

    modes = [m for m in ("baseline", "kinematic", "lstm_only", "full")
             if any(r["mode_id"] == m for r in runs)]
    by_mode: dict[str, list[dict]] = {m: [r for r in runs if r["mode_id"] == m]
                                       for m in modes}

    # Title
    total_cols = 2 + len(modes) + 2
    ws.merge_cells(start_row=1, start_column=1,
                   end_row=1, end_column=total_cols)
    t = ws.cell(1, 1, "Head-to-Head Metric Comparison Matrix  (mean across all videos)")
    t.font = Font(bold=True, color="FFFFFF", name="Arial", size=12)
    t.fill = PatternFill("solid", fgColor="1F3864")
    t.alignment = Alignment(horizontal="center")
    ws.row_dimensions[1].height = 24

    # Column headers
    _hdr_cell(ws, 2, 1, "Metric", 1)
    _hdr_cell(ws, 2, 2, "Higher=Better?", 2)
    for ci, mode in enumerate(modes, 3):
        _hdr_cell(ws, 2, ci, MODE_LABELS.get(mode, mode), 1)
    _hdr_cell(ws, 2, len(modes) + 3, "Best Mode", 1)
    _hdr_cell(ws, 2, len(modes) + 4, "Δ Best-Worst", 1)
    ws.row_dimensions[2].height = 30
    ws.freeze_panes = "C3"

    for ri, (m_label, m_key, higher_better, m_fmt) in enumerate(KEY_METRICS_H2H, 3):
        alt = (ri % 2 == 0)
        _data_cell(ws, ri, 1, m_label, alt=alt)
        _data_cell(ws, ri, 2, "↑" if higher_better else "↓", alt=alt)

        means = []
        for ci, mode in enumerate(modes, 3):
            vals = [r.get(m_key) for r in by_mode[mode] if r.get(m_key) is not None]
            mean = float(np.mean(vals)) if vals else 0.0
            means.append(mean)
            _data_cell(ws, ri, ci, round(mean, 4), fmt=m_fmt, alt=alt)

        if means:
            best_idx = (np.argmax(means) if higher_better else np.argmin(means))
            worst_idx = (np.argmin(means) if higher_better else np.argmax(means))
            best_mode = MODE_LABELS.get(modes[best_idx], modes[best_idx])
            delta = abs(means[best_idx] - means[worst_idx])

            # highlight best
            best_col = 3 + best_idx
            bc = ws.cell(ri, best_col)
            bc.fill = PatternFill("solid", fgColor=XL_GREEN)
            bc.font = Font(bold=True, name="Arial", size=9)

            _data_cell(ws, ri, len(modes) + 3, best_mode, alt=alt)
            _data_cell(ws, ri, len(modes) + 4, round(delta, 4), fmt=m_fmt, alt=alt)

    _set_col_widths(ws, [24, 14] + [18] * len(modes) + [20, 14])
    return ws


# ═══════════════════════════════════════════════════════════════════════════
#  Sheet 5 — Metadata & Config
# ═══════════════════════════════════════════════════════════════════════════

def _sheet_meta(wb: Workbook, runs: list[dict], logs_root: str, out_path: str):
    ws = wb.create_sheet("Metadata")

    rows = [
        ("EVALUATION REPORT METADATA", ""),
        ("Generated at",   datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Logs root",      logs_root),
        ("Output file",    out_path),
        ("Total runs",     len(runs)),
        ("Unique videos",  len({r["video_name"] for r in runs})),
        ("Unique modes",   len({r["mode_id"]    for r in runs})),
        ("Ablation runs",  len({r["ablation_ts"] for r in runs
                                if r["ablation_ts"] != "plain_run"})),
        ("", ""),
        ("ABLATION MODES", ""),
        ("baseline",  "YOLOv8 + DeepSORT + Euclidean distance trigger (80 px)"),
        ("kinematic", "YOLOv8 + DeepSORT + TTC + Cosine Similarity filter"),
        ("lstm_only", "YOLOv8 + DeepSORT + MotionLSTM predictor"),
        ("full",      "YOLOv8 + DeepSORT + LSTM + TTC + ResNet50 cascade"),
        ("", ""),
        ("VIDEOS PROCESSED", ""),
    ]
    for video in sorted({r["video_name"] for r in runs}):
        rows.append((video, ""))

    for ri, (k, v) in enumerate(rows, 1):
        ka = ws.cell(ri, 1, k)
        va = ws.cell(ri, 2, v)
        ka.font = Font(bold=True if k.isupper() or k in (
            "baseline","kinematic","lstm_only","full") else False,
            name="Arial", size=9)
        va.font = Font(name="Arial", size=9)
        if k.isupper() and k:
            ka.fill = PatternFill("solid", fgColor=XL_HEADER_FILL)
            ka.font = Font(bold=True, color="FFFFFF", name="Arial", size=9)
            ws.merge_cells(start_row=ri, start_column=1, end_row=ri, end_column=2)
            ka.alignment = Alignment(horizontal="center")

    _set_col_widths(ws, [24, 70])
    return ws


# ═══════════════════════════════════════════════════════════════════════════
#  PNG Dashboard
# ═══════════════════════════════════════════════════════════════════════════

def _build_dashboard(runs: list[dict], out_path: str):
    modes = [m for m in ("baseline", "kinematic", "lstm_only", "full")
             if any(r["mode_id"] == m for r in runs)]
    by_mode = {m: [r for r in runs if r["mode_id"] == m] for m in modes}

    def _mean(mode, key):
        vals = [r.get(key, 0.0) for r in by_mode[mode]]
        return float(np.mean(vals)) if vals else 0.0

    colors = [MODE_COLORS_MPL[m] for m in modes]
    labels_short = [MODE_LABELS[m].split(" ")[0] for m in modes]

    fig = plt.figure(figsize=(22, 14), facecolor=DARK_BG)
    fig.suptitle("System Evaluation Dashboard — All Videos, All Modes",
                 color="white", fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 4, hspace=0.50, wspace=0.40,
                           left=0.06, right=0.97, top=0.92, bottom=0.10)

    def _style(ax):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_CLR, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_CLR)
        ax.grid(axis="y", color=GRID_CLR, lw=0.5, alpha=0.7)
        ax.title.set_color(TEXT_CLR)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)

    def _bar(ax, keys, title, ylabel, ylim=None, pct=False, inv=False):
        n = len(keys)
        x = np.arange(n)
        w = 0.8 / len(modes)
        offsets = np.linspace(-(len(modes)-1)*w/2, (len(modes)-1)*w/2, len(modes))
        for offset, mode, clr in zip(offsets, modes, colors):
            vals = [_mean(mode, k) for k in keys]
            if pct:
                vals = [v * 100 for v in vals]
            ax.bar(x + offset, vals, w * 0.9, color=clr, alpha=0.85,
                   label=MODE_LABELS[mode])
        ax.set_xticks(x)
        ax.set_xticklabels(keys if n > 1 else [""], fontsize=8, color=TEXT_CLR)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        if ylim:
            ax.set_ylim(*ylim)
        _style(ax)

    # R1C1 — Precision / Recall / F1
    ax = fig.add_subplot(gs[0, 0])
    _bar(ax, ["precision", "recall", "f1_score"],
         "Precision / Recall / F1", "Score", ylim=(0, 1.12))
    ax.set_xticklabels(["Precision", "Recall", "F1"], fontsize=8, color=TEXT_CLR)
    ax.legend(fontsize=7, framealpha=0, labelcolor=TEXT_CLR, loc="lower right")

    # R1C2 — ROC-AUC / PR-AUC
    ax = fig.add_subplot(gs[0, 1])
    _bar(ax, ["roc_auc", "pr_auc"], "ROC-AUC / PR-AUC", "Score", ylim=(0, 1.12))
    ax.set_xticklabels(["ROC-AUC", "PR-AUC"], fontsize=8, color=TEXT_CLR)

    # R1C3 — MOTA / IDF1
    ax = fig.add_subplot(gs[0, 2])
    _bar(ax, ["mota", "idf1"], "Tracking: MOTA / IDF1", "Score")
    ax.set_xticklabels(["MOTA", "IDF1"], fontsize=8, color=TEXT_CLR)

    # R1C4 — False Alarm Rate (lower=better)
    ax = fig.add_subplot(gs[0, 3])
    fa_vals = [_mean(m, "false_alarm_rate") * 100 for m in modes]
    bars = ax.bar(labels_short, fa_vals, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, fa_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f"{v:.3f}%", ha="center", color=TEXT_CLR, fontsize=8)
    ax.set_title("False Alarm Rate (↓ better)", fontsize=10)
    ax.set_ylabel("% of frames")
    _style(ax)

    # R2C1 — Incidents detected
    ax = fig.add_subplot(gs[1, 0])
    inc_vals = [_mean(m, "unique_incidents") for m in modes]
    bars = ax.bar(labels_short, inc_vals, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, inc_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.05,
                f"{v:.1f}", ha="center", color=TEXT_CLR, fontsize=9, fontweight="bold")
    ax.set_title("Avg Incidents Detected", fontsize=10)
    ax.set_ylabel("Count (↑ better)")
    _style(ax)

    # R2C2 — CNN inferences
    ax = fig.add_subplot(gs[1, 1])
    cnn_vals = [_mean(m, "cnn_inferences") for m in modes]
    bars = ax.bar(labels_short, cnn_vals, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, cnn_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.01,
                f"{int(v):,}", ha="center", color=TEXT_CLR, fontsize=8)
    ax.set_title("Avg CNN Inferences (↓ = cheaper)", fontsize=10)
    ax.set_ylabel("Count")
    _style(ax)

    # R2C3 — FPS
    ax = fig.add_subplot(gs[1, 2])
    fps_vals = [_mean(m, "avg_fps") for m in modes]
    bars = ax.bar(labels_short, fps_vals, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, fps_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.01,
                f"{v:.1f}", ha="center", color=TEXT_CLR, fontsize=8)
    ax.set_title("Avg Processing FPS (↑ better)", fontsize=10)
    ax.set_ylabel("FPS")
    _style(ax)

    # R2C4 — Radar / spider (quality summary)
    ax = fig.add_subplot(gs[1, 3], polar=True)
    radar_keys  = ["f1_score", "roc_auc", "recall", "mota", "idf1"]
    radar_lbls  = ["F1", "ROC-AUC", "Recall", "MOTA", "IDF1"]
    n_r = len(radar_keys)
    angles = np.linspace(0, 2 * np.pi, n_r, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_facecolor(PANEL_BG)
    ax.spines["polar"].set_color(GRID_CLR)
    ax.tick_params(colors=TEXT_CLR, labelsize=7)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_lbls, color=TEXT_CLR, fontsize=8)
    ax.set_ylim(0, 1)
    ax.yaxis.set_tick_params(labelcolor=GRID_CLR)
    ax.grid(color=GRID_CLR, lw=0.5)

    for mode, clr in zip(modes, colors):
        vals = [_mean(mode, k) for k in radar_keys]
        vals += vals[:1]
        ax.plot(angles, vals, color=clr, lw=2)
        ax.fill(angles, vals, color=clr, alpha=0.10)

    ax.set_title("Quality Radar", color=TEXT_CLR, fontsize=10, pad=14)

    # Legend at bottom
    patches = [Patch(facecolor=MODE_COLORS_MPL[m], label=MODE_LABELS[m], alpha=0.85)
               for m in modes]
    fig.legend(handles=patches, loc="lower center", ncol=len(modes),
               fontsize=10, framealpha=0, labelcolor=TEXT_CLR,
               bbox_to_anchor=(0.5, 0.01))

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  📊 Dashboard: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Text summary
# ═══════════════════════════════════════════════════════════════════════════

def _build_text_summary(runs: list[dict], logs_root: str) -> str:
    modes = [m for m in ("baseline", "kinematic", "lstm_only", "full")
             if any(r["mode_id"] == m for r in runs)]
    by_mode = {m: [r for r in runs if r["mode_id"] == m] for m in modes}
    videos  = sorted({r["video_name"] for r in runs})

    def _mean(lst, key):
        v = [r.get(key, 0.0) for r in lst if r.get(key) is not None]
        return float(np.mean(v)) if v else 0.0

    sep  = "=" * 80
    sep2 = "-" * 80
    lines = [
        "", sep,
        "  ACCIDENT DETECTION SYSTEM — COMPREHENSIVE EVALUATION REPORT",
        sep,
        f"  Generated:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Log root:        {logs_root}",
        f"  Total runs:      {len(runs)}",
        f"  Videos tested:   {len(videos)}",
        f"  Modes compared:  {len(modes)}",
        "",
        "  Videos: " + ", ".join(videos),
        "",
    ]

    # ── Per-mode overview ─────────────────────────────────────────────────
    lines += [sep2, "  AGGREGATE QUALITY METRICS (mean across all tested videos)", sep2]
    col_w = 24

    hdr  = f"  {'Metric':<{col_w}}"
    hdr += "".join(f"{MODE_LABELS.get(m, m):^18}" for m in modes)
    lines.append(hdr)
    lines.append("  " + "-" * (col_w + 18 * len(modes)))

    TABLE_ROWS = [
        ("Precision",         "precision"),
        ("Recall",            "recall"),
        ("F1-Score",          "f1_score"),
        ("Accuracy",          "accuracy"),
        ("ROC-AUC",           "roc_auc"),
        ("PR-AUC",            "pr_auc"),
        ("mAP@0.5",           "map_50"),
        ("---", None),
        ("MOTA",              "mota"),
        ("IDF1",              "idf1"),
        ("Avg ID Switches",   "id_switches"),
        ("---", None),
        ("Unique Incidents",  "unique_incidents"),
        ("False Alarm Rate",  "false_alarm_rate"),
        ("Collision Warnings","collision_warnings"),
        ("---", None),
        ("CNN Inferences",    "cnn_inferences"),
        ("Avg FPS",           "avg_fps"),
        ("P95 Latency ms",    "p95_ms"),
        ("Memory MB",         "memory_mb"),
    ]

    for label, key in TABLE_ROWS:
        if key is None:
            lines.append("")
            continue
        row = f"  {label:<{col_w}}"
        means = [_mean(by_mode[m], key) for m in modes]
        for v in means:
            if key in ("cnn_inferences", "unique_incidents", "collision_warnings",
                       "id_switches", "memory_mb"):
                row += f"{int(v):^18,}"
            elif key == "false_alarm_rate":
                row += f"{v*100:^17.3f}%"
            elif key in ("avg_fps", "p95_ms"):
                row += f"{v:^17.1f} "
            else:
                row += f"{v:^18.4f}"
        lines.append(row)

    # ── Per-video breakdown ───────────────────────────────────────────────
    lines += ["", sep2, "  PER-VIDEO BREAKDOWN — F1 / Recall / False-Alarm Rate", sep2]

    run_map = {(r["video_name"], r["mode_id"]): r for r in runs}
    for video in videos:
        lines.append(f"\n  [{video}]")
        hdr2 = f"    {'Mode':<26}" + "  F1      Recall   ROC-AUC  FA-Rate  Incidents"
        lines.append(hdr2)
        lines.append("    " + "-" * 70)
        for mode in modes:
            r = run_map.get((video, mode))
            if not r:
                lines.append(f"    {MODE_LABELS.get(mode, mode):<26}  —")
                continue
            lines.append(
                f"    {MODE_LABELS.get(mode, mode):<26}"
                f"  {r.get('f1_score', 0):.4f}  "
                f"{r.get('recall', 0):.4f}   "
                f"{r.get('roc_auc', 0):.4f}   "
                f"{r.get('false_alarm_rate', 0)*100:.3f}%   "
                f"{int(r.get('unique_incidents', 0))}"
            )

    # ── Ranking ───────────────────────────────────────────────────────────
    lines += ["", sep2, "  MODE RANKING (by mean F1-Score ↓ best first)", sep2]
    ranked = sorted(
        [(m, _mean(by_mode[m], "f1_score")) for m in modes],
        key=lambda x: -x[1],
    )
    for rank, (mode, val) in enumerate(ranked, 1):
        lines.append(f"    #{rank}  {MODE_LABELS.get(mode, mode):<26}  F1 = {val:.4f}")

    lines += ["", sep, "  END OF REPORT", sep, ""]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def aggregate(logs_root: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  Scanning: {logs_root}")
    runs = discover_runs(logs_root)

    if not runs:
        print("  ⚠️  No runs found. Run ablation_runner.py first.")
        return

    print(f"  Found {len(runs)} run(s) across "
          f"{len({r['video_name'] for r in runs})} video(s) "
          f"and {len({r['mode_id'] for r in runs})} mode(s).\n")

    xlsx_path = os.path.join(out_dir, "evaluation_report.xlsx")
    png_path  = os.path.join(out_dir, "evaluation_report.png")
    txt_path  = os.path.join(out_dir, "evaluation_summary.txt")

    # ── Excel workbook ────────────────────────────────────────────────────
    print("  Building Excel workbook…")
    wb = Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    _sheet_raw(wb, runs)
    _sheet_per_video(wb, runs)
    _sheet_per_mode(wb, runs)
    _sheet_h2h(wb, runs)
    _sheet_meta(wb, runs, logs_root, xlsx_path)

    # Reorder sheets
    order = ["Raw Data", "Per-Video Summary", "Per-Mode Aggregate",
             "Head-to-Head Matrix", "Metadata"]
    for i, name in enumerate(order):
        if name in wb.sheetnames:
            wb.move_sheet(name, offset=i - wb.sheetnames.index(name))

    wb.save(xlsx_path)
    print(f"  💾 Workbook: {xlsx_path}")

    # Recalculate via LibreOffice if available
    recalc_script = Path(__file__).parent / "scripts" / "recalc.py"
    if recalc_script.exists():
        import subprocess
        result = subprocess.run(
            [sys.executable, str(recalc_script), xlsx_path],
            capture_output=True, text=True
        )
        try:
            info = json.loads(result.stdout)
            if info.get("status") == "errors_found":
                print(f"  ⚠️  Formula errors: {info.get('total_errors')}")
            else:
                print(f"  ✅ Formulas recalculated OK ({info.get('total_formulas', 0)} formulas)")
        except Exception:
            pass

    # ── Dashboard PNG ─────────────────────────────────────────────────────
    print("  Rendering dashboard…")
    _build_dashboard(runs, png_path)

    # ── Text summary ──────────────────────────────────────────────────────
    print("  Writing text summary…")
    summary = _build_text_summary(runs, logs_root)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"  📄 Summary: {txt_path}")

    print(summary)

    print(f"\n  ✅ All outputs written to: {out_dir}\n")
    return {"xlsx": xlsx_path, "png": png_path, "txt": txt_path, "runs": len(runs)}


def _parse():
    p = argparse.ArgumentParser(description="Aggregate all ablation results into one report")
    p.add_argument("--logs", default=None,
                   help="Path to logs root (default: auto-detect from Config)")
    p.add_argument("--out",  default=None,
                   help="Output directory (default: <logs>/evaluation_report)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    logs_root = args.logs
    if not logs_root:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from modules.config.config import Config
            logs_root = Config.LOG_DIR
        except Exception:
            logs_root = os.path.join(os.path.dirname(__file__), "logs")

    out_dir = args.out or os.path.join(logs_root, "evaluation_report")
    aggregate(logs_root, out_dir)