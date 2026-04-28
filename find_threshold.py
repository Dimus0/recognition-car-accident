"""
find_threshold.py
=================
Аналізує cnn_scores.jsonl + ground truth розмітку (межі аварій по кадрах)
і знаходить оптимальний поріг для досягнення Recall >= 0.80.

Використання:
    python find_threshold.py \
        --scores  logs/evaluation/accident_video_v8/cnn_scores.jsonl \
        --gt      logs/gt/gt_accident_video_v8.json \
        --target_recall 0.80

Ground truth формат (JSON, ваш поточний):
    {
      "frame_annotations": {
        "150": {"accident": true},
        "151": {"accident": true},
        ...
      }
    }

    АБО простіший формат — список діапазонів:
    {
      "accident_ranges": [[150, 320], [540, 610]]
    }
"""

import argparse
import json
import os
import sys
import numpy as np

# matplotlib опціональний — якщо немає, друкуємо лише текст
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
#  1. Завантаження даних
# ─────────────────────────────────────────────────────────────────────────────

def load_cnn_scores(path: str) -> dict:
    """
    Читає cnn_scores.jsonl і повертає {frame: max_score_in_frame}.
    Кожен рядок у jsonl — один CNN інференс (один трек в одному кадрі).
    Беремо максимальний eff_score по кадру як представника кадру.
    """
    frame_scores: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fn  = int(rec["frame"])
            # eff_score є — використовуємо його; fallback на raw_score
            score = float(rec.get("eff_score", rec.get("raw_score", 0.0)))
            if fn not in frame_scores or score > frame_scores[fn]:
                frame_scores[fn] = score
    return frame_scores


def load_ground_truth(path: str) -> set:
    """
    Повертає set кадрів де є аварія (label=1).
    Підтримує обидва формати:
      • {"frame_annotations": {"150": {"accident": true}, ...}}
      • {"accident_ranges": [[150, 320], [540, 610]]}
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


# ─────────────────────────────────────────────────────────────────────────────
#  2. Побудова масивів labels / scores по кадрах
# ─────────────────────────────────────────────────────────────────────────────

def build_arrays(frame_scores: dict, accident_frames: set):
    """
    Вирівнює scores і labels по спільних кадрах.
    Кадри без score (CNN не запускався) — отримують score=0.0 і правильний label.
    """
    # Всі кадри що є хоча б в одному джерелі
    all_frames = sorted(set(frame_scores.keys()) | accident_frames)

    scores = np.array([frame_scores.get(f, 0.0) for f in all_frames], dtype=np.float32)
    labels = np.array([1 if f in accident_frames else 0 for f in all_frames], dtype=np.int32)

    return scores, labels, np.array(all_frames)


# ─────────────────────────────────────────────────────────────────────────────
#  3. Основний аналіз
# ─────────────────────────────────────────────────────────────────────────────

def analyse(scores: np.ndarray, labels: np.ndarray,
            target_recall: float = 0.80,
            output_dir: str = "."):

    # ── Precision-Recall curve вручну (без sklearn) ───────────────────────────
    thresholds = np.linspace(0.0, 1.0, 1001)
    precisions, recalls, f1s = [], [], []

    total_pos = int(labels.sum())
    if total_pos == 0:
        print("ПОМИЛКА: у ground truth немає жодного позитивного кадру!")
        sys.exit(1)

    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        tp   = int(((pred == 1) & (labels == 1)).sum())
        fp   = int(((pred == 1) & (labels == 0)).sum())
        fn   = int(((pred == 0) & (labels == 1)).sum())

        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    precisions = np.array(precisions)
    recalls    = np.array(recalls)
    f1s        = np.array(f1s)

    # ── Знаходимо оптимальний поріг ──────────────────────────────────────────
    # 1. Поріг де Recall >= target і Precision максимальний
    mask_recall = recalls >= target_recall
    if not mask_recall.any():
        # Recall недосяжний навіть при threshold=0 (дуже мало позитивних передбачень)
        idx_best = np.argmax(recalls)
        print(f"⚠️  Recall={target_recall:.2f} недосяжний! Максимальний: {recalls.max():.4f}")
    else:
        # Серед усіх порогів що дають потрібний Recall — беремо з найвищим Precision
        best_precision_among_recall = precisions[mask_recall].max()
        candidates = np.where(mask_recall & (precisions >= best_precision_among_recall * 0.99))[0]
        idx_best = candidates[np.argmax(thresholds[candidates])]  # вищий поріг при рівному прецизії

    # 2. Поріг F1-max
    idx_f1 = np.argmax(f1s)

    # 3. Поточний поріг (0.90 з config — CONF_ACCIDENT_HIGH)
    idx_current = int(round(0.90 * 1000))

    results = {
        "optimal_for_recall": {
            "threshold":  float(thresholds[idx_best]),
            "precision":  float(precisions[idx_best]),
            "recall":     float(recalls[idx_best]),
            "f1":         float(f1s[idx_best]),
        },
        "optimal_f1": {
            "threshold":  float(thresholds[idx_f1]),
            "precision":  float(precisions[idx_f1]),
            "recall":     float(recalls[idx_f1]),
            "f1":         float(f1s[idx_f1]),
        },
        "current_090": {
            "threshold":  0.90,
            "precision":  float(precisions[idx_current]),
            "recall":     float(recalls[idx_current]),
            "f1":         float(f1s[idx_current]),
        },
        "dataset": {
            "total_frames":    len(labels),
            "accident_frames": total_pos,
            "normal_frames":   int((labels == 0).sum()),
            "positive_rate":   float(total_pos / len(labels)),
        }
    }

    # ── Друкуємо ─────────────────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}")
    print("  THRESHOLD ANALYSIS")
    print(sep)

    d = results["dataset"]
    print(f"\n  Кадрів всього:    {d['total_frames']}")
    print(f"  Аварійних:        {d['accident_frames']}  ({d['positive_rate']*100:.1f}%)")
    print(f"  Нормальних:       {d['normal_frames']}")

    def _row(label, r):
        print(f"\n  {label}")
        print(f"    threshold  = {r['threshold']:.3f}")
        print(f"    precision  = {r['precision']:.4f}")
        print(f"    recall     = {r['recall']:.4f}  {'✅' if r['recall'] >= target_recall else '❌'}")
        print(f"    f1-score   = {r['f1']:.4f}")

    _row("▶  ПОТОЧНИЙ (threshold=0.90):",        results["current_090"])
    _row(f"▶  ОПТИМАЛЬНИЙ для Recall≥{target_recall}:", results["optimal_for_recall"])
    _row("▶  ОПТИМАЛЬНИЙ по F1:",                results["optimal_f1"])

    # ── Рекомендовані зміни в config.py ──────────────────────────────────────
    thr_rec = results["optimal_for_recall"]["threshold"]
    thr_f1  = results["optimal_f1"]["threshold"]

    print(f"\n{sep}")
    print("  РЕКОМЕНДОВАНІ ЗМІНИ в config.py")
    print(sep)
    print(f"""
  # Варіант 1 — пріоритет Recall≥{target_recall}  (більше виявлень, менше пропусків)
  CONF_ACCIDENT_HIGH = {thr_rec:.2f}   # було 0.90
  CONF_ACCIDENT_LOW  = {max(0.60, thr_rec - 0.08):.2f}   # було 0.85  (WARNING поріг)

  # Варіант 2 — баланс Precision/Recall  (F1-оптимум)
  CONF_ACCIDENT_HIGH = {thr_f1:.2f}   # було 0.90
  CONF_ACCIDENT_LOW  = {max(0.60, thr_f1 - 0.08):.2f}   # було 0.85
""")

    # ── Рекомендовані зміни в AccidentStateTracker ────────────────────────────
    print(f"{sep}")
    print("  РЕКОМЕНДОВАНІ ЗМІНИ в accidenttracker.py")
    print(sep)

    # Формула: якщо новий поріг < 0.85, знижуємо avg_thresh у should_confirm_accident
    new_avg_thresh    = round(max(0.75, thr_rec - 0.02), 2)
    new_min_thresh    = round(max(0.65, thr_rec - 0.10), 2)
    new_avg_thresh_f1 = round(max(0.75, thr_f1 - 0.02), 2)
    new_min_thresh_f1 = round(max(0.65, thr_f1 - 0.10), 2)

    print(f"""
  У методі should_confirm_accident():

  # Варіант 1 — пріоритет Recall
  avg_thresh = {new_avg_thresh}   # було 0.85
  min_thresh = {new_min_thresh}   # було 0.75

  # Варіант 2 — F1-баланс
  avg_thresh = {new_avg_thresh_f1}   # було 0.85
  min_thresh = {new_min_thresh_f1}   # було 0.75

  Також розгляньте:
    confirmation_threshold = 2   # було 3  (менше кадрів для підтвердження)
    HEARTBEAT_RATE         = 10  # було 15 (частіша перевірка CNN)
""")

    # ── Графік ────────────────────────────────────────────────────────────────
    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1117")
        DARK = "#0d1117"; TEXT = "#c9d1d9"; GRID = "#21262d"
        BLUE = "#58a6ff"; GREEN = "#3fb950"; RED = "#f78166"; ORANGE = "#ffa657"

        for ax in axes:
            ax.set_facecolor(DARK)
            ax.tick_params(colors=TEXT)
            for sp in ax.spines.values(): sp.set_edgecolor(GRID)
            ax.grid(color=GRID, lw=0.6, alpha=0.7)
            ax.title.set_color(TEXT); ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)

        # ── Лівий: Precision-Recall по threshold ──────────────────────────────
        ax = axes[0]
        ax.plot(thresholds, precisions, color=BLUE,   lw=1.8, label="Precision")
        ax.plot(thresholds, recalls,    color=GREEN,  lw=1.8, label="Recall")
        ax.plot(thresholds, f1s,        color=ORANGE, lw=1.4, label="F1", ls="--")

        ax.axvline(0.90, color="white", lw=1.2, ls=":", label="Current (0.90)")
        ax.axvline(thr_rec, color=GREEN, lw=1.8, ls="--",
                   label=f"Recall≥{target_recall} ({thr_rec:.2f})")
        ax.axvline(thr_f1,  color=ORANGE, lw=1.4, ls="--",
                   label=f"F1-max ({thr_f1:.2f})")
        ax.axhline(target_recall, color=GREEN, lw=0.8, ls=":", alpha=0.6)

        ax.set_title("Precision / Recall / F1 vs Threshold")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, framealpha=0, labelcolor=TEXT)

        # ── Правий: PR крива ──────────────────────────────────────────────────
        ax = axes[1]
        ax.plot(recalls, precisions, color=BLUE, lw=2)
        ax.scatter([results["current_090"]["recall"]],
                   [results["current_090"]["precision"]],
                   color="white", s=60, zorder=5, label="Current (0.90)")
        ax.scatter([results["optimal_for_recall"]["recall"]],
                   [results["optimal_for_recall"]["precision"]],
                   color=GREEN, s=80, zorder=5, label=f"Recall≥{target_recall}")
        ax.scatter([results["optimal_f1"]["recall"]],
                   [results["optimal_f1"]["precision"]],
                   color=ORANGE, s=80, zorder=5, label="F1-max")
        ax.axvline(target_recall, color=GREEN, lw=0.8, ls=":", alpha=0.6)

        ax.set_title("Precision-Recall Curve")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, framealpha=0, labelcolor=TEXT)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "threshold_analysis.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=DARK)
        plt.close()
        print(f"\n  Графік збережено: {plot_path}")

    # ── Зберігаємо JSON результат ─────────────────────────────────────────────
    result_path = os.path.join(output_dir, "threshold_analysis.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  JSON збережено:  {result_path}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Знаходить оптимальний поріг для CNN класифікатора")
    parser.add_argument("--scores",        required=True,  help="Шлях до cnn_scores.jsonl")
    parser.add_argument("--gt",            required=True,  help="Шлях до ground truth JSON")
    parser.add_argument("--target_recall", type=float, default=0.80, help="Цільовий Recall (default: 0.80)")
    parser.add_argument("--output_dir",    default=".",    help="Директорія для збереження графіків")
    args = parser.parse_args()

    if not os.path.exists(args.scores):
        print(f"ПОМИЛКА: файл scores не знайдено: {args.scores}"); sys.exit(1)
    if not os.path.exists(args.gt):
        print(f"ПОМИЛКА: GT файл не знайдено: {args.gt}"); sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"  Scores:   {args.scores}")
    print(f"  GT:       {args.gt}")
    print(f"  Target recall: {args.target_recall}")

    frame_scores    = load_cnn_scores(args.scores)
    accident_frames = load_ground_truth(args.gt)
    scores, labels, frames = build_arrays(frame_scores, accident_frames)

    analyse(scores, labels,
            target_recall=args.target_recall,
            output_dir=args.output_dir)
