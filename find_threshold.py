"""
find_threshold.py
=================
Аналізує cnn_scores.jsonl + ground truth розмітку і знаходить оптимальні
пороги для Precision, Recall та F1-score окремо, а також при constrained-умовах.

Використання:
    python find_threshold.py \\
        --scores  logs/evaluation/accident_video_v8/cnn_scores.jsonl \\
        --gt      logs/gt/gt_accident_video_v8.json \\
        --target_recall    0.80 \\
        --target_precision 0.85 \\
        --target_f1        0.80 \\
        --output_dir       logs/threshold_analysis

         python find_threshold.py --scores  logs/evaluation/accident_video_v8/cnn_scores.jsonl --gt logs/gt/gt_accident_video_v8.json --target_recall 0.80 --target_precision 0.85 --target_f1 0.80 --output_dir logs/threshold_analysis

Ground truth формат (JSON):
    {"frame_annotations": {"150": {"accident": true}, ...}}
    АБО
    {"accident_ranges": [[150, 320], [540, 610]]}
"""

import argparse
import json
import os
import sys
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
#  Структура результату для одного оптимуму
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThresholdResult:
    label:       str
    threshold:   float
    precision:   float
    recall:      float
    f1:          float
    tp:          int
    fp:          int
    fn:          int
    tn:          int
    accuracy:    float
    fpr:         float   # false positive rate = fp / (fp + tn)
    constraint:  Optional[str] = None   # наприклад "recall >= 0.80"


# ─────────────────────────────────────────────────────────────────────────────
#  1. Завантаження даних
# ─────────────────────────────────────────────────────────────────────────────

def load_cnn_scores(path: str) -> dict:
    """
    Читає cnn_scores.jsonl → {frame: max_eff_score}.
    Якщо eff_score відсутній — fallback на raw_score.
    """
    frame_scores: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec   = json.loads(line)
            fn    = int(rec["frame"])
            score = float(rec.get("eff_score", rec.get("raw_score", 0.0)))
            if fn not in frame_scores or score > frame_scores[fn]:
                frame_scores[fn] = score
    return frame_scores


def load_ground_truth(path: str) -> set:
    """
    Повертає set кадрів де є аварія.
    Підтримує два формати: frame_annotations і accident_ranges.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    accident_frames: set = set()

    if "accident_ranges" in data:
        for start, end in data["accident_ranges"]:
            accident_frames.update(range(int(start), int(end) + 1))
    elif "frame_annotations" in data:
        for frame_str, ann in data["frame_annotations"].items():
            if ann.get("accident", False):
                accident_frames.add(int(frame_str))

    return accident_frames


def build_arrays(frame_scores: dict, accident_frames: set):
    """
    Вирівнює scores і labels по всіх кадрах.
    Кадри без CNN-інференсу отримують score=0.0.
    """
    all_frames = sorted(set(frame_scores.keys()) | accident_frames)
    scores = np.array([frame_scores.get(f, 0.0) for f in all_frames], dtype=np.float32)
    labels = np.array([1 if f in accident_frames else 0 for f in all_frames], dtype=np.int32)
    return scores, labels, np.array(all_frames)


# ─────────────────────────────────────────────────────────────────────────────
#  2. Розрахунок confusion matrix для одного порогу
# ─────────────────────────────────────────────────────────────────────────────

def _confusion(scores: np.ndarray, labels: np.ndarray, thr: float):
    pred = (scores >= thr).astype(int)
    tp   = int(((pred == 1) & (labels == 1)).sum())
    fp   = int(((pred == 1) & (labels == 0)).sum())
    fn   = int(((pred == 0) & (labels == 1)).sum())
    tn   = int(((pred == 0) & (labels == 0)).sum())
    return tp, fp, fn, tn


def _metrics_from_confusion(tp, fp, fn, tn) -> tuple:
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy  = (tp + tn) / (tp + fp + fn + tn + 1e-9)
    fpr       = fp / (fp + tn + 1e-9)
    return precision, recall, f1, accuracy, fpr


# ─────────────────────────────────────────────────────────────────────────────
#  3. Побудова кривих по всіх порогах
# ─────────────────────────────────────────────────────────────────────────────

def build_curves(scores: np.ndarray, labels: np.ndarray, n_steps: int = 2001):
    """
    Повертає масиви thresholds, precisions, recalls, f1s, accuracies, fprs.
    n_steps=2001 дає крок 0.0005 — достатньо для плавних кривих.
    """
    thresholds  = np.linspace(0.0, 1.0, n_steps)
    precisions  = np.zeros(n_steps)
    recalls     = np.zeros(n_steps)
    f1s         = np.zeros(n_steps)
    accuracies  = np.zeros(n_steps)
    fprs        = np.zeros(n_steps)

    for i, thr in enumerate(thresholds):
        tp, fp, fn, tn = _confusion(scores, labels, thr)
        p, r, f, a, fpr = _metrics_from_confusion(tp, fp, fn, tn)
        precisions[i] = p
        recalls[i]    = r
        f1s[i]        = f
        accuracies[i] = a
        fprs[i]       = fpr

    return thresholds, precisions, recalls, f1s, accuracies, fprs


# ─────────────────────────────────────────────────────────────────────────────
#  4. Пошук оптимальних порогів
# ─────────────────────────────────────────────────────────────────────────────

def _make_result(label, thr, scores, labels, constraint=None) -> ThresholdResult:
    tp, fp, fn, tn = _confusion(scores, labels, thr)
    p, r, f, a, fpr = _metrics_from_confusion(tp, fp, fn, tn)
    return ThresholdResult(
        label=label, threshold=float(thr),
        precision=p, recall=r, f1=f,
        tp=tp, fp=fp, fn=fn, tn=tn,
        accuracy=a, fpr=fpr,
        constraint=constraint,
    )


def find_optimal_thresholds(
    thresholds:        np.ndarray,
    precisions:        np.ndarray,
    recalls:           np.ndarray,
    f1s:               np.ndarray,
    scores:            np.ndarray,
    labels:            np.ndarray,
    target_recall:     float = 0.80,
    target_precision:  float = 0.80,
    target_f1:         float = 0.80,
    current_threshold: float = 0.90,
) -> dict:
    """
    Знаходить оптимальні пороги за шістьма критеріями:

      max_precision   — максимальна Precision (ціна — падіння Recall)
      max_recall      — максимальний Recall (ціна — падіння Precision)
      max_f1          — максимальний F1
      constrained_recall     — max Precision при Recall >= target_recall
      constrained_precision  — max Recall при Precision >= target_precision
      constrained_f1         — поріг при F1 >= target_f1 з max Precision
      current                — поточний поріг config.py
    """
    results = {}

    # ── Поточний поріг ────────────────────────────────────────────────────────
    idx_current = int(np.argmin(np.abs(thresholds - current_threshold)))
    results["current"] = _make_result(
        f"Поточний (thr={current_threshold:.2f})",
        thresholds[idx_current], scores, labels,
    )

    # ── Max Precision ─────────────────────────────────────────────────────────
    # Серед порогів де є хоч одна позитивна предикція (precision не дорівнює 1 по вакууму)
    valid_mask = (thresholds <= 1.0) & (recalls > 0.0)   # хоч один TP
    if valid_mask.any():
        idx_maxp = int(np.argmax(precisions * valid_mask + (1 - valid_mask) * -1))
    else:
        idx_maxp = int(np.argmax(precisions))
    results["max_precision"] = _make_result(
        "Max Precision", thresholds[idx_maxp], scores, labels,
    )

    # ── Max Recall ────────────────────────────────────────────────────────────
    idx_maxr = int(np.argmax(recalls))
    results["max_recall"] = _make_result(
        "Max Recall", thresholds[idx_maxr], scores, labels,
    )

    # ── Max F1 ────────────────────────────────────────────────────────────────
    idx_maxf1 = int(np.argmax(f1s))
    results["max_f1"] = _make_result(
        "Max F1", thresholds[idx_maxf1], scores, labels,
    )

    # ── Constrained: max Precision при Recall >= target_recall ───────────────
    mask_r = recalls >= target_recall
    if mask_r.any():
        best_p_in_mask = precisions[mask_r].max()
        candidates     = np.where(mask_r & (precisions >= best_p_in_mask * 0.99))[0]
        idx_cpr        = candidates[np.argmax(thresholds[candidates])]
        results["constrained_recall"] = _make_result(
            f"Max Precision при Recall≥{target_recall:.2f}",
            thresholds[idx_cpr], scores, labels,
            constraint=f"recall >= {target_recall:.2f}",
        )
    else:
        # Ціль недосяжна — беремо поріг з максимальним recall
        results["constrained_recall"] = _make_result(
            f"⚠ Recall≥{target_recall:.2f} недосяжний → Max Recall",
            thresholds[idx_maxr], scores, labels,
            constraint=f"recall >= {target_recall:.2f} [NOT MET]",
        )

    # ── Constrained: max Recall при Precision >= target_precision ────────────
    mask_p = precisions >= target_precision
    if mask_p.any():
        best_r_in_mask = recalls[mask_p].max()
        candidates     = np.where(mask_p & (recalls >= best_r_in_mask * 0.99))[0]
        idx_cre        = candidates[np.argmin(thresholds[candidates])]  # нижчий поріг = більше recall
        results["constrained_precision"] = _make_result(
            f"Max Recall при Precision≥{target_precision:.2f}",
            thresholds[idx_cre], scores, labels,
            constraint=f"precision >= {target_precision:.2f}",
        )
    else:
        results["constrained_precision"] = _make_result(
            f"⚠ Precision≥{target_precision:.2f} недосяжний → Max Precision",
            thresholds[idx_maxp], scores, labels,
            constraint=f"precision >= {target_precision:.2f} [NOT MET]",
        )

    # ── Constrained: max Precision при F1 >= target_f1 ───────────────────────
    mask_f1 = f1s >= target_f1
    if mask_f1.any():
        best_p_in_f1_mask = precisions[mask_f1].max()
        candidates        = np.where(mask_f1 & (precisions >= best_p_in_f1_mask * 0.99))[0]
        idx_cf1           = candidates[np.argmax(thresholds[candidates])]
        results["constrained_f1"] = _make_result(
            f"Max Precision при F1≥{target_f1:.2f}",
            thresholds[idx_cf1], scores, labels,
            constraint=f"f1 >= {target_f1:.2f}",
        )
    else:
        results["constrained_f1"] = _make_result(
            f"⚠ F1≥{target_f1:.2f} недосяжний → Max F1",
            thresholds[idx_maxf1], scores, labels,
            constraint=f"f1 >= {target_f1:.2f} [NOT MET]",
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  5. Виведення у консоль
# ─────────────────────────────────────────────────────────────────────────────

def _check(val, tgt, higher_better=True):
    if higher_better:
        return "✅" if val >= tgt else "❌"
    return "✅" if val <= tgt else "❌"


def print_results(results: dict, dataset_info: dict,
                  target_recall: float, target_precision: float, target_f1: float):

    sep   = "=" * 72
    sep2  = "-" * 72
    W_KEY = 40

    def row(k, v, suffix=""):
        print(f"    {k:<{W_KEY}} {v}{suffix}")

    print(f"\n{sep}")
    print("  THRESHOLD OPTIMIZATION — ПОВНИЙ ЗВІТ")
    print(sep)

    d = dataset_info
    print(f"\n  Датасет:")
    print(f"    Кадрів всього:            {d['total_frames']}")
    print(f"    Аварійних (positive):     {d['accident_frames']}  "
          f"({d['positive_rate']*100:.1f}%)")
    print(f"    Нормальних (negative):    {d['normal_frames']}")

    print(f"\n{sep2}")
    print("  ОПТИМАЛЬНІ ПОРОГИ ПО МЕТРИКАХ")
    print(sep2)

    # Порядок виведення і цілі для ✅/❌
    order = [
        ("current",               target_recall, target_precision, target_f1),
        ("max_precision",         target_recall, target_precision, target_f1),
        ("max_recall",            target_recall, target_precision, target_f1),
        ("max_f1",                target_recall, target_precision, target_f1),
        ("constrained_recall",    target_recall, target_precision, target_f1),
        ("constrained_precision", target_recall, target_precision, target_f1),
        ("constrained_f1",        target_recall, target_precision, target_f1),
    ]

    for key, tr, tp_t, tf1 in order:
        r: ThresholdResult = results[key]
        print(f"\n  ▶  {r.label}")
        if r.constraint:
            print(f"     Обмеження: {r.constraint}")
        print(f"     threshold  = {r.threshold:.4f}")
        print(f"     precision  = {r.precision:.4f}  "
              f"{_check(r.precision, tp_t)}")
        print(f"     recall     = {r.recall:.4f}  "
              f"{_check(r.recall, tr)}")
        print(f"     f1-score   = {r.f1:.4f}  "
              f"{_check(r.f1, tf1)}")
        print(f"     accuracy   = {r.accuracy:.4f}")
        print(f"     TP={r.tp}  FP={r.fp}  FN={r.fn}  TN={r.tn}  "
              f"FPR={r.fpr:.4f}")


def print_config_recommendations(results: dict, target_recall: float):
    sep = "=" * 72
    print(f"\n{sep}")
    print("  РЕКОМЕНДОВАНІ ЗМІНИ В config.py")
    print(sep)

    scenarios = [
        ("max_f1",               "F1-оптимум (збалансований)"),
        ("constrained_recall",   f"Пріоритет Recall≥{target_recall:.2f} (мінімум пропусків)"),
        ("constrained_precision","Пріоритет Precision (мінімум хибних тривог)"),
    ]

    for key, desc in scenarios:
        r: ThresholdResult = results[key]
        thr = r.threshold
        low = round(max(0.55, thr - 0.08), 2)
        avg_t = round(max(0.70, thr - 0.02), 2)
        min_t = round(max(0.60, thr - 0.10), 2)

        print(f"\n  # ── {desc} ──")
        print(f"  # Precision={r.precision:.3f}  Recall={r.recall:.3f}  F1={r.f1:.3f}")
        print(f"  CONF_ACCIDENT_HIGH = {thr:.2f}   # поріг підтвердження")
        print(f"  CONF_ACCIDENT_LOW  = {low:.2f}   # поріг WARNING")
        print(f"  # accidenttracker.py → should_confirm_accident():")
        print(f"  #   avg_thresh = {avg_t}   min_thresh = {min_t}")

    print(f"\n{sep}")
    print("  ПОРІВНЯЛЬНА ТАБЛИЦЯ ВСІХ ОПТИМУМІВ")
    print(sep)
    header = f"  {'Варіант':<32} {'Thr':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Acc':>6}"
    print(header)
    print(f"  {'-'*68}")
    for key in ("current", "max_precision", "max_recall", "max_f1",
                "constrained_recall", "constrained_precision", "constrained_f1"):
        r: ThresholdResult = results[key]
        name = r.label[:32]
        print(f"  {name:<32} {r.threshold:>6.3f}  {r.precision:>6.3f}  "
              f"{r.recall:>6.3f}  {r.f1:>6.3f}  {r.accuracy:>6.3f}")


# ─────────────────────────────────────────────────────────────────────────────
#  6. Візуалізація
# ─────────────────────────────────────────────────────────────────────────────

def save_plots(
    thresholds:   np.ndarray,
    precisions:   np.ndarray,
    recalls:      np.ndarray,
    f1s:          np.ndarray,
    accuracies:   np.ndarray,
    results:      dict,
    target_recall:    float,
    target_precision: float,
    target_f1:        float,
    output_dir:   str,
):
    if not HAS_MPL:
        print("  matplotlib не встановлено — графіки пропущено")
        return

    DARK   = "#0d1117"
    PANEL  = "#0d1117"
    GRID   = "#21262d"
    TEXT   = "#c9d1d9"
    BLUE   = "#58a6ff"
    GREEN  = "#3fb950"
    RED    = "#f78166"
    ORANGE = "#ffa657"
    PURPLE = "#d2a8ff"
    TEAL   = "#39d353"
    WHITE  = "#ffffff"

    def _style(ax):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(color=GRID, lw=0.5, alpha=0.6)
        ax.title.set_color(TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)

    # ══════════════════════════════════════════════════════════════════════
    #  Графік 1 — 2×2: метрики vs поріг + PR-крива + Confusion heatmap
    # ══════════════════════════════════════════════════════════════════════

    fig = plt.figure(figsize=(16, 10), facecolor=DARK)
    fig.suptitle("Threshold Optimization — Full Analysis",
                 color=WHITE, fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.38, figure=fig)

    # ── 1a. Precision vs Threshold ───────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    _style(ax)
    ax.plot(thresholds, precisions, color=BLUE, lw=2, label="Precision")
    _mark = results["max_precision"]
    ax.axvline(_mark.threshold, color=BLUE, lw=1.5, ls="--",
               label=f"Max P thr={_mark.threshold:.3f}")
    _con = results["constrained_precision"]
    ax.axvline(_con.threshold, color=ORANGE, lw=1.2, ls=":",
               label=f"MaxR|P≥{target_precision:.2f} thr={_con.threshold:.3f}")
    ax.axhline(target_precision, color=ORANGE, lw=0.8, ls=":", alpha=0.5)
    _cur = results["current"]
    ax.axvline(_cur.threshold, color=WHITE, lw=1.0, ls=":",
               label=f"Current thr={_cur.threshold:.3f}")
    ax.set_title("Precision vs Threshold")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, framealpha=0, labelcolor=TEXT)

    # ── 1b. Recall vs Threshold ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    _style(ax)
    ax.plot(thresholds, recalls, color=GREEN, lw=2, label="Recall")
    _mark = results["max_recall"]
    ax.axvline(_mark.threshold, color=GREEN, lw=1.5, ls="--",
               label=f"Max R thr={_mark.threshold:.3f}")
    _con = results["constrained_recall"]
    ax.axvline(_con.threshold, color=TEAL, lw=1.2, ls=":",
               label=f"MaxP|R≥{target_recall:.2f} thr={_con.threshold:.3f}")
    ax.axhline(target_recall, color=TEAL, lw=0.8, ls=":", alpha=0.5)
    ax.axvline(_cur.threshold, color=WHITE, lw=1.0, ls=":",
               label=f"Current thr={_cur.threshold:.3f}")
    ax.set_title("Recall vs Threshold")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Recall")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, framealpha=0, labelcolor=TEXT)

    # ── 1c. F1 vs Threshold ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    _style(ax)
    ax.plot(thresholds, f1s, color=ORANGE, lw=2, label="F1-score")
    _mark = results["max_f1"]
    ax.axvline(_mark.threshold, color=ORANGE, lw=1.5, ls="--",
               label=f"Max F1 thr={_mark.threshold:.3f}")
    _con = results["constrained_f1"]
    ax.axvline(_con.threshold, color=PURPLE, lw=1.2, ls=":",
               label=f"MaxP|F1≥{target_f1:.2f} thr={_con.threshold:.3f}")
    ax.axhline(target_f1, color=PURPLE, lw=0.8, ls=":", alpha=0.5)
    ax.axvline(_cur.threshold, color=WHITE, lw=1.0, ls=":",
               label=f"Current thr={_cur.threshold:.3f}")
    ax.set_title("F1-score vs Threshold")
    ax.set_xlabel("Threshold"); ax.set_ylabel("F1")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, framealpha=0, labelcolor=TEXT)

    # ── 1d. Всі метрики разом ────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    _style(ax)
    ax.plot(thresholds, precisions, color=BLUE,   lw=1.6, label="Precision")
    ax.plot(thresholds, recalls,    color=GREEN,  lw=1.6, label="Recall")
    ax.plot(thresholds, f1s,        color=ORANGE, lw=1.6, label="F1")
    ax.plot(thresholds, accuracies, color=PURPLE, lw=1.2, ls="--", label="Accuracy")
    for key, clr in [("max_precision", BLUE), ("max_recall", GREEN),
                     ("max_f1", ORANGE), ("current", WHITE)]:
        r = results[key]
        ax.axvline(r.threshold, color=clr, lw=1.0, ls=":",
                   alpha=0.7, label=f"{key.replace('_', ' ')} ({r.threshold:.3f})")
    ax.set_title("Всі метрики vs Threshold")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6.5, framealpha=0, labelcolor=TEXT, ncol=2)

    # ── 1e. Precision-Recall крива ───────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    _style(ax)
    ax.plot(recalls, precisions, color=BLUE, lw=2, label="PR curve")

    points = [
        ("current",               WHITE,  "o", 60, f"Current ({_cur.threshold:.2f})"),
        ("max_precision",         BLUE,   "^", 80, "Max Precision"),
        ("max_recall",            GREEN,  "v", 80, "Max Recall"),
        ("max_f1",                ORANGE, "D", 80, "Max F1"),
        ("constrained_recall",    TEAL,   "s", 70, f"MaxP|R≥{target_recall:.2f}"),
        ("constrained_precision", ORANGE, "P", 70, f"MaxR|P≥{target_precision:.2f}"),
        ("constrained_f1",        PURPLE, "*", 90, f"MaxP|F1≥{target_f1:.2f}"),
    ]
    for key, clr, marker, sz, lbl in points:
        r = results[key]
        ax.scatter([r.recall], [r.precision], color=clr, marker=marker,
                   s=sz, zorder=5, label=lbl)

    ax.axhline(target_precision, color=ORANGE, lw=0.7, ls=":", alpha=0.5)
    ax.axvline(target_recall,    color=TEAL,   lw=0.7, ls=":", alpha=0.5)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6.5, framealpha=0, labelcolor=TEXT)

    # ── 1f. Bar chart — порівняння всіх варіантів ───────────────────────
    ax = fig.add_subplot(gs[1, 2])
    _style(ax)

    bar_keys   = ["current", "max_precision", "max_recall",
                  "max_f1", "constrained_recall",
                  "constrained_precision", "constrained_f1"]
    bar_labels = ["Current", "MaxPrec", "MaxRec",
                  "MaxF1", f"R≥{target_recall:.2f}",
                  f"P≥{target_precision:.2f}", f"F1≥{target_f1:.2f}"]
    x       = np.arange(len(bar_keys))
    width   = 0.25
    prec_v  = [results[k].precision for k in bar_keys]
    rec_v   = [results[k].recall    for k in bar_keys]
    f1_v    = [results[k].f1        for k in bar_keys]

    ax.bar(x - width, prec_v, width, color=BLUE,   alpha=0.85, label="Precision")
    ax.bar(x,         rec_v,  width, color=GREEN,  alpha=0.85, label="Recall")
    ax.bar(x + width, f1_v,   width, color=ORANGE, alpha=0.85, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_title("Порівняння варіантів")
    ax.set_ylabel("Score")
    ax.legend(fontsize=7, framealpha=0, labelcolor=TEXT)

    plot_path = os.path.join(output_dir, "threshold_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"\n  📊 Збережено: {plot_path}")

    # ══════════════════════════════════════════════════════════════════════
    #  Графік 2 — Confusion Matrix Heatmap для 3-х ключових порогів
    # ══════════════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor=DARK)
    fig.suptitle("Confusion Matrix: Current vs Max F1 vs Constrained",
                 color=WHITE, fontsize=12, fontweight="bold")

    cm_keys   = ["current", "max_f1", "constrained_recall"]
    cm_titles = [f"Current (thr={results['current'].threshold:.3f})",
                 f"Max F1 (thr={results['max_f1'].threshold:.3f})",
                 f"MaxP|R≥{target_recall:.2f} (thr={results['constrained_recall'].threshold:.3f})"]

    for ax, key, title in zip(axes, cm_keys, cm_titles):
        r = results[key]
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        im = ax.imshow(cm, cmap="Blues", aspect="auto")

        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: NEG", "Pred: POS"], color=TEXT, fontsize=9)
        ax.set_yticklabels(["True: NEG", "True: POS"], color=TEXT, fontsize=9)
        ax.set_title(title, color=TEXT, fontsize=9, pad=8)

        labels_cm = [[f"TN\n{r.tn}", f"FP\n{r.fp}"],
                     [f"FN\n{r.fn}", f"TP\n{r.tp}"]]
        colors_cm = [["#3fb950", "#f78166"], ["#ffa657", "#58a6ff"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, labels_cm[i][j], ha="center", va="center",
                        color=colors_cm[i][j], fontsize=12, fontweight="bold")

        sub = (f"P={r.precision:.3f}  R={r.recall:.3f}  "
               f"F1={r.f1:.3f}  Acc={r.accuracy:.3f}")
        ax.set_xlabel(sub, color=TEXT, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    cm_path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  📊 Збережено: {cm_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  7. Головна функція аналізу
# ─────────────────────────────────────────────────────────────────────────────

def analyse(
    scores:            np.ndarray,
    labels:            np.ndarray,
    target_recall:     float = 0.80,
    target_precision:  float = 0.80,
    target_f1:         float = 0.80,
    current_threshold: float = 0.90,
    output_dir:        str   = ".",
) -> dict:

    total_pos = int(labels.sum())
    total_neg = int((labels == 0).sum())
    if total_pos == 0:
        print("ПОМИЛКА: у ground truth немає жодного позитивного кадру!")
        sys.exit(1)

    dataset_info = {
        "total_frames":  len(labels),
        "accident_frames": total_pos,
        "normal_frames":   total_neg,
        "positive_rate":   float(total_pos / len(labels)),
    }

    # Будуємо криві
    thresholds, precisions, recalls, f1s, accuracies, fprs = build_curves(
        scores, labels, n_steps=2001
    )

    # Знаходимо оптимуми
    results = find_optimal_thresholds(
        thresholds, precisions, recalls, f1s,
        scores, labels,
        target_recall=target_recall,
        target_precision=target_precision,
        target_f1=target_f1,
        current_threshold=current_threshold,
    )

    # Виводимо в консоль
    print_results(results, dataset_info, target_recall, target_precision, target_f1)
    print_config_recommendations(results, target_recall)

    # Графіки
    save_plots(
        thresholds, precisions, recalls, f1s, accuracies,
        results,
        target_recall, target_precision, target_f1,
        output_dir,
    )

    # JSON
    json_results = {
        "dataset":    dataset_info,
        "thresholds": {
            k: asdict(v) for k, v in results.items()
        },
        "curves": {
            "thresholds":  thresholds.tolist()[::10],   # decimated для компактності
            "precisions":  precisions.tolist()[::10],
            "recalls":     recalls.tolist()[::10],
            "f1s":         f1s.tolist()[::10],
            "accuracies":  accuracies.tolist()[::10],
            "fprs":        fprs.tolist()[::10],
        },
        "targets": {
            "recall":    target_recall,
            "precision": target_precision,
            "f1":        target_f1,
        },
    }

    result_path = os.path.join(output_dir, "threshold_analysis.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 JSON збережено: {result_path}\n")

    return json_results


# ─────────────────────────────────────────────────────────────────────────────
#  8. CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Знаходить оптимальні пороги для CNN класифікатора "
                    "(max Precision / Recall / F1 + constrained варіанти)"
    )
    parser.add_argument("--scores",            required=True,
                        help="Шлях до cnn_scores.jsonl")
    parser.add_argument("--gt",                required=True,
                        help="Шлях до ground truth JSON")
    parser.add_argument("--target_recall",     type=float, default=0.80,
                        help="Цільовий Recall для constrained оптимізації (default: 0.80)")
    parser.add_argument("--target_precision",  type=float, default=0.80,
                        help="Цільова Precision для constrained оптимізації (default: 0.80)")
    parser.add_argument("--target_f1",         type=float, default=0.80,
                        help="Цільовий F1 для constrained оптимізації (default: 0.80)")
    parser.add_argument("--current_threshold", type=float, default=0.90,
                        help="Поточний поріг з config.py для порівняння (default: 0.90)")
    parser.add_argument("--output_dir",        default=".",
                        help="Директорія для збереження результатів")
    args = parser.parse_args()

    for path, name in [(args.scores, "scores"), (args.gt, "ground truth")]:
        if not os.path.exists(path):
            print(f"ПОМИЛКА: {name} файл не знайдено: {path}")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n  Scores:            {args.scores}")
    print(f"  Ground truth:      {args.gt}")
    print(f"  Target recall:     {args.target_recall}")
    print(f"  Target precision:  {args.target_precision}")
    print(f"  Target f1:         {args.target_f1}")
    print(f"  Current threshold: {args.current_threshold}")

    frame_scores    = load_cnn_scores(args.scores)
    accident_frames = load_ground_truth(args.gt)
    scores, labels, _ = build_arrays(frame_scores, accident_frames)

    analyse(
        scores, labels,
        target_recall=args.target_recall,
        target_precision=args.target_precision,
        target_f1=args.target_f1,
        current_threshold=args.current_threshold,
        output_dir=args.output_dir,
    )