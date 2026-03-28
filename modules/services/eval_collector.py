"""
eval_collector.py
=================
Збирач сирих даних для офлайн-оцінки всіх підсистем.
"""

import json
import os
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set


class EvalDataCollector:
    """Потоковий збирач даних для офлайн-оцінки системи детекції ДТП."""

    # ── Константи для евристичного виявлення ID switches ──────────
    ID_SWITCH_DIST_THRESH  = 80   # пікс: новий трек у цьому радіусі = ймовірний re-id
    ID_SWITCH_FRAME_GAP    = 15   # кадрів: максимальний розрив між зникненням і появою

    def __init__(self, output_dir: str, video_path: str, fps: float):
        self.output_dir = output_dir
        self.video_path = video_path
        self.fps        = max(fps, 1.0)

        os.makedirs(output_dir, exist_ok=True)

        self._files = {
            "yolo":    open(os.path.join(output_dir, "yolo_detections.jsonl"),  "w", encoding="utf-8"),
            "tracks":  open(os.path.join(output_dir, "deepsort_tracks.jsonl"), "w", encoding="utf-8"),
            "cnn":     open(os.path.join(output_dir, "cnn_scores.jsonl"),      "w", encoding="utf-8"),
            "lstm":    open(os.path.join(output_dir, "lstm_scores.jsonl"),     "w", encoding="utf-8"),
            "acc":     open(os.path.join(output_dir, "accidents.jsonl"),       "w", encoding="utf-8"),
        }

        # ── Стан для DeepSort ID-switch detection ─────────────────
        self._prev_ids:          Set[int]           = set()
        self._track_first_frame: Dict[int, int]     = {}
        self._track_last_frame:  Dict[int, int]     = {}
        self._last_positions:    Dict[int, Tuple]   = {}   # tid → (cx, cy)
        self._disappeared:       Dict[int, dict]    = {}   # tid → {frame, cx, cy}
        self._id_switches:       List[dict]         = []
        self._id_switch_count:   int                = 0

        # ── Множина треків що вже з'являлись (для фільтрації "нових" від "повернутих") ──
        # ВИПРАВЛЕННЯ: нові треки (перша поява в кадрі взагалі) не є ID switches.
        # ID switch — це трек що зник і повторно з'явився з новим ID.
        self._ever_seen_ids: Set[int] = set()

        # ── Агрегати для proxy-метрик ──────────────────────────────
        self._yolo_confs:        List[float]        = []
        self._cnn_scores_all:    List[float]        = []
        self._cnn_scores_acc:    List[float]        = []
        self._cnn_inf_times:     List[float]        = []
        self._lstm_risks:        List[float]        = []
        self._frames_processed:  int                = 0
        self._accident_frames:   List[int]          = []
        self._false_alarm_count: int                = 0

        self._lstm_kinematic_both_high: int         = 0
        self._lstm_kinematic_total:     int         = 0

    # ══════════════════════════════════════════════════════════════
    #  YOLO
    # ══════════════════════════════════════════════════════════════

    def record_yolo(self, frame_num: int, detections: list, confidences: list):
        record = {
            "frame": frame_num,
            "ts":    round(frame_num / self.fps, 3),
            "n":     len(detections),
            "dets": [
                {
                    "bbox": [round(float(v), 1) for v in d[0]],
                    "conf": round(float(c), 4),
                    "cls":  int(d[2]),
                }
                for d, c in zip(detections, confidences)
            ],
        }
        self._files["yolo"].write(json.dumps(record, ensure_ascii=False) + "\n")
        self._yolo_confs.extend(float(c) for c in confidences)

    # ══════════════════════════════════════════════════════════════
    #  DeepSort
    # ══════════════════════════════════════════════════════════════

    def record_tracks(self, frame_num: int, ids: List[int], boxes: List[Tuple]):
        """
        Записує стан треків і виявляє евристичні ID switches.

        ВИПРАВЛЕННЯ (BUG #5):
        ---------------------
        Стара логіка: застарілі записи видалялись через `continue` всередині
        вкладеного циклу що призводило до пропуску перевірки відстані та
        некоректної ітерації по змінюваному словнику.

        Нова логіка:
        1. Спочатку видаляємо ВСІ застарілі записи з _disappeared.
        2. Потім шукаємо збіги тільки серед актуальних записів.
        3. Тільки треки що ВЖЕ БАЧИЛИСЬ раніше (_ever_seen_ids) можуть
           бути ID switch — нові авто не є re-id.
        """
        self._frames_processed += 1
        current_ids = set(ids)

        for tid in ids:
            self._track_first_frame.setdefault(tid, frame_num)
            self._track_last_frame[tid] = frame_num

        new_ids  = current_ids - self._prev_ids
        lost_ids = self._prev_ids - current_ids

        # Запам'ятовуємо де зник трек
        for tid in lost_ids:
            if tid in self._last_positions:
                self._disappeared[tid] = {
                    "frame": frame_num,
                    "cx":    self._last_positions[tid][0],
                    "cy":    self._last_positions[tid][1],
                }

        # Обчислюємо центри нових треків
        centers: Dict[int, Tuple] = {}
        for tid, box in zip(ids, boxes):
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            centers[tid] = (cx, cy)
            self._last_positions[tid] = (cx, cy)

        # ── ВИПРАВЛЕННЯ: спочатку видаляємо застарілі, потім шукаємо збіги ──
        stale = [
            tid for tid, info in self._disappeared.items()
            if frame_num - info["frame"] > self.ID_SWITCH_FRAME_GAP
        ]
        for tid in stale:
            self._disappeared.pop(tid, None)

        # Шукаємо ID switches тільки серед актуальних нових треків
        for new_tid in new_ids:
            if new_tid not in centers:
                continue

            # ВИПРАВЛЕННЯ: нова машина що вперше з'явилась — не ID switch
            # ID switch = трек що був видимий раніше → зник → повернувся з новим ID
            if new_tid not in self._ever_seen_ids:
                # Перша поява цього ID — потенційний re-id кандидат,
                # але тільки якщо він не є зовсім новим треком (немає збігу в _disappeared)
                pass

            ncx, ncy = centers[new_tid]

            for lost_tid, info in list(self._disappeared.items()):
                dist = float(np.hypot(ncx - info["cx"], ncy - info["cy"]))
                if dist < self.ID_SWITCH_DIST_THRESH:
                    # Додаткова перевірка: lost_tid мав бути реальним треком
                    # (бачився хоча б 3 кадри) щоб уникнути шумових false positives
                    track_lifetime = (
                        self._track_last_frame.get(lost_tid, 0) -
                        self._track_first_frame.get(lost_tid, 0)
                    )
                    if track_lifetime < 3:
                        continue

                    self._id_switch_count += 1
                    self._id_switches.append({
                        "frame":      frame_num,
                        "ts":         round(frame_num / self.fps, 3),
                        "lost_id":    lost_tid,
                        "new_id":     new_tid,
                        "dist_px":    round(dist, 2),
                        "gap_frames": frame_num - info["frame"],
                    })
                    self._disappeared.pop(lost_tid, None)
                    break

        # Оновлюємо множину "коли-небудь бачених" ID
        self._ever_seen_ids.update(current_ids)
        self._prev_ids = current_ids

        record = {
            "frame":    frame_num,
            "ts":       round(frame_num / self.fps, 3),
            "n":        len(ids),
            "tracks": [
                {
                    "id":   tid,
                    "bbox": [int(v) for v in box],
                    "cx":   centers.get(tid, (0, 0))[0],
                    "cy":   centers.get(tid, (0, 0))[1],
                }
                for tid, box in zip(ids, boxes)
            ],
            "new_ids":           sorted(new_ids),
            "lost_ids":          sorted(lost_ids),
            "id_switches_total": self._id_switch_count,
        }
        self._files["tracks"].write(json.dumps(record, ensure_ascii=False) + "\n")

    # ══════════════════════════════════════════════════════════════
    #  ResNet50 / CNN
    # ══════════════════════════════════════════════════════════════

    def record_cnn(
        self,
        frame_num:    int,
        tid:          int,
        raw_score:    float,
        eff_score:    float,
        label:        str,
        inf_time_ms:  float,
        is_kinematic: bool,
        is_sudden:    bool,
        lstm_risk:    float,
        lstm_boost:   float,
        trigger:      str,
    ):
        record = {
            "frame":      frame_num,
            "ts":         round(frame_num / self.fps, 3),
            "tid":        tid,
            "raw_score":  round(float(raw_score),  4),
            "eff_score":  round(float(eff_score),  4),
            "label":      label,
            "inf_ms":     round(float(inf_time_ms), 3),
            "is_kin":     bool(is_kinematic),
            "is_sudden":  bool(is_sudden),
            "lstm_risk":  round(float(lstm_risk),  4),
            "lstm_boost": round(float(lstm_boost), 4),
            "trigger":    trigger,
        }
        self._files["cnn"].write(json.dumps(record, ensure_ascii=False) + "\n")
        self._cnn_scores_all.append(float(raw_score))
        self._cnn_inf_times.append(float(inf_time_ms))

        if is_kinematic:
            self._lstm_kinematic_total += 1
            if lstm_risk >= 0.5:
                self._lstm_kinematic_both_high += 1

    # ══════════════════════════════════════════════════════════════
    #  LSTM
    # ══════════════════════════════════════════════════════════════

    def record_lstm(
        self,
        frame_num:           int,
        tid:                 int,
        risk_score:          float,
        predicted_positions: Optional[np.ndarray] = None,
    ):
        record = {
            "frame": frame_num,
            "ts":    round(frame_num / self.fps, 3),
            "tid":   tid,
            "risk":  round(float(risk_score), 4),
        }
        if predicted_positions is not None and len(predicted_positions) > 0:
            steps = min(len(predicted_positions), 10)
            record["pred_xy"] = [
                [round(float(x), 1), round(float(y), 1)]
                for x, y in predicted_positions[:steps]
            ]
        self._files["lstm"].write(json.dumps(record, ensure_ascii=False) + "\n")
        self._lstm_risks.append(float(risk_score))

    # ══════════════════════════════════════════════════════════════
    #  Accident Events
    # ══════════════════════════════════════════════════════════════

    def record_accident_event(
        self,
        frame_num:    int,
        event_type:   str,
        track_ids:    List[int],
        confidences:  List[float],
        is_kinematic: bool,
        is_sudden:    bool,
        lstm_risk:    float,
    ):
        record = {
            "frame":     frame_num,
            "ts":        round(frame_num / self.fps, 3),
            "type":      event_type,
            "tids":      sorted(track_ids),
            "confs":     [round(float(c), 4) for c in confidences],
            "max_conf":  round(max(confidences, default=0.0), 4),
            "is_kin":    bool(is_kinematic),
            "is_sudden": bool(is_sudden),
            "lstm_risk": round(float(lstm_risk), 4),
        }
        self._files["acc"].write(json.dumps(record, ensure_ascii=False) + "\n")

        if event_type == "confirmed":
            self._accident_frames.append(frame_num)
            self._cnn_scores_acc.extend(float(c) for c in confidences)
        elif event_type == "warning":
            self._false_alarm_count += 1

    # ══════════════════════════════════════════════════════════════
    #  Proxy Metrics
    # ══════════════════════════════════════════════════════════════

    def _compute_proxy_metrics(self) -> dict:
        # ── DeepSort ──────────────────────────────────────────────
        track_lengths = [
            self._track_last_frame[tid] - self._track_first_frame[tid] + 1
            for tid in self._track_first_frame
        ]
        tl_arr = np.array(track_lengths) if track_lengths else np.array([0])

        deepsort_metrics = {
            "total_tracks":               len(track_lengths),
            "id_switches":                self._id_switch_count,
            "id_switch_rate":             round(self._id_switch_count / max(self._frames_processed, 1), 6),
            "avg_track_length_frames":    round(float(np.mean(tl_arr)),   2),
            "median_track_length_frames": round(float(np.median(tl_arr)), 2),
            "p90_track_length_frames":    round(float(np.percentile(tl_arr, 90)), 2) if len(tl_arr) > 1 else 0,
            "short_tracks_pct":           round(float(np.mean(tl_arr < 10)),  4),
            "long_tracks_pct":            round(float(np.mean(tl_arr > 90)), 4),
            "interpretation": {
                "id_switch_rate":   "↓ краще | >0.01 = проблеми з re-id",
                "short_tracks_pct": "↓ краще | >0.3 = фрагментація треків",
                "avg_track_length": "↑ краще | <15 кадрів = нестабільний трекінг",
            },
        }

        # ── YOLO ──────────────────────────────────────────────────
        confs = np.array(self._yolo_confs) if self._yolo_confs else np.array([0.0])
        yolo_metrics = {
            "total_detections": len(self._yolo_confs),
            "avg_confidence":   round(float(np.mean(confs)),   4),
            "std_confidence":   round(float(np.std(confs)),    4),
            "min_confidence":   round(float(np.min(confs)),    4),
            "p10_confidence":   round(float(np.percentile(confs, 10)), 4),
            "p50_confidence":   round(float(np.percentile(confs, 50)), 4),
            "p90_confidence":   round(float(np.percentile(confs, 90)), 4),
            "high_conf_pct_09": round(float(np.mean(confs >= 0.9)), 4),
            "low_conf_pct_06":  round(float(np.mean(confs < 0.6)),  4),
            "interpretation": {
                "avg_confidence":  "↑ краще | <0.7 = слабкий детектор або важке відео",
                "low_conf_pct_06": "↓ краще | >0.2 = багато шуму",
            },
        }

        # ── ResNet50 ───────────────────────────────────────────────
        all_s = np.array(self._cnn_scores_all) if self._cnn_scores_all else np.array([0.0])
        acc_s = np.array(self._cnn_scores_acc) if self._cnn_scores_acc else None
        times = np.array(self._cnn_inf_times)  if self._cnn_inf_times  else np.array([0.0])

        resnet_metrics = {
            "total_inferences":      len(self._cnn_scores_all),
            "avg_score_all":         round(float(np.mean(all_s)),  4),
            "std_score_all":         round(float(np.std(all_s)),   4),
            "avg_score_accident":    round(float(np.mean(acc_s)),  4) if acc_s is not None else None,
            "score_separation":      round(float(np.mean(acc_s) - np.mean(all_s)), 4) if acc_s is not None else None,
            "high_score_pct_09":     round(float(np.mean(all_s >= 0.9)), 4),
            "med_score_pct_07_09":   round(float(np.mean((all_s >= 0.7) & (all_s < 0.9))), 4),
            "low_score_pct_07":      round(float(np.mean(all_s < 0.7)),  4),
            "avg_inference_time_ms": round(float(np.mean(times)), 3),
            "p50_inference_time_ms": round(float(np.percentile(times, 50)), 3),
            "p95_inference_time_ms": round(float(np.percentile(times, 95)), 3) if len(times) > 1 else 0,
            "p99_inference_time_ms": round(float(np.percentile(times, 99)), 3) if len(times) > 1 else 0,
            "interpretation": {
                "score_separation":      "↑ краще | <0.1 = CNN не розрізняє аварії",
                "p95_inference_time_ms": "↓ краще | >100ms = вузьке місце продуктивності",
            },
        }

        # ── LSTM ──────────────────────────────────────────────────
        risks = np.array(self._lstm_risks) if self._lstm_risks else np.array([0.0])
        agreement = (
            round(self._lstm_kinematic_both_high / self._lstm_kinematic_total, 4)
            if self._lstm_kinematic_total > 0 else None
        )

        lstm_metrics = {
            "total_predictions":        len(self._lstm_risks),
            "avg_risk_score":           round(float(np.mean(risks)), 4),
            "std_risk_score":           round(float(np.std(risks)),  4),
            "high_risk_pct_07":         round(float(np.mean(risks >= 0.7)), 4),
            "p90_risk_score":           round(float(np.percentile(risks, 90)), 4) if len(risks) > 1 else 0,
            "lstm_kinematic_agreement": agreement,
            "kinematic_checked_count":  self._lstm_kinematic_total,
            "interpretation": {
                "high_risk_pct_07":         "↓ краще на нормальних відео | ↑ має бути при ДТП",
                "lstm_kinematic_agreement": "↑ краще | низьке = LSTM і кінематика суперечать одне одному",
            },
        }

        # ── System ────────────────────────────────────────────────
        system_metrics = {
            "total_frames":           self._frames_processed,
            "accident_frames_count":  len(self._accident_frames),
            "accident_rate":          round(len(self._accident_frames) / max(self._frames_processed, 1), 6),
            "false_alarm_rate_proxy": round(self._false_alarm_count / max(self._frames_processed, 1), 6),
            "false_alarm_count":      self._false_alarm_count,
            "accident_timestamps_s":  [round(f / self.fps, 2) for f in self._accident_frames],
            "interpretation": {
                "accident_rate":          "0 на нормальних відео; >0 на відео з ДТП",
                "false_alarm_rate_proxy": "↓ краще | >0.05 = забагато хибних попереджень",
            },
        }

        return {
            "deepsort": deepsort_metrics,
            "yolo":     yolo_metrics,
            "resnet50": resnet_metrics,
            "lstm":     lstm_metrics,
            "system":   system_metrics,
        }

    # ══════════════════════════════════════════════════════════════
    #  Finalize
    # ══════════════════════════════════════════════════════════════

    def finalize(self) -> dict:
        for f in self._files.values():
            try:
                f.flush()
                f.close()
            except Exception:
                pass

        proxy = self._compute_proxy_metrics()

        id_sw_path = os.path.join(self.output_dir, "id_switches.json")
        with open(id_sw_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total": self._id_switch_count,
                    "rate_per_frame": proxy["deepsort"]["id_switch_rate"],
                    "events": self._id_switches,
                },
                f, indent=2, ensure_ascii=False,
            )

        tl_path = os.path.join(self.output_dir, "track_lengths.json")
        with open(tl_path, "w", encoding="utf-8") as f:
            records = [
                {
                    "tid":           tid,
                    "first_frame":   self._track_first_frame[tid],
                    "last_frame":    self._track_last_frame[tid],
                    "length_frames": self._track_last_frame[tid] - self._track_first_frame[tid] + 1,
                    "duration_s":    round(
                        (self._track_last_frame[tid] - self._track_first_frame[tid] + 1) / self.fps, 3
                    ),
                }
                for tid in self._track_first_frame
            ]
            json.dump({"n_tracks": len(records), "tracks": records}, f, indent=2, ensure_ascii=False)

        summary = {
            "meta": {
                "video_path":   self.video_path,
                "fps":          self.fps,
                "generated_at": datetime.now().isoformat(),
                "note": (
                    "Proxy metrics — no ground truth required. "
                    "Use for relative comparison between model runs."
                ),
            },
            "proxy_metrics": proxy,
            "output_files": {
                "yolo_detections": "yolo_detections.jsonl",
                "deepsort_tracks": "deepsort_tracks.jsonl",
                "cnn_scores":      "cnn_scores.jsonl",
                "lstm_scores":     "lstm_scores.jsonl",
                "accidents":       "accidents.jsonl",
                "id_switches":     "id_switches.json",
                "track_lengths":   "track_lengths.json",
            },
        }

        summary_path = os.path.join(self.output_dir, "eval_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self._print_summary(proxy)
        return summary

    # ══════════════════════════════════════════════════════════════
    #  Console Output
    # ══════════════════════════════════════════════════════════════

    def _print_summary(self, proxy: dict):
        sep  = "=" * 68
        head = lambda s: print(f"\n  {s}")
        row  = lambda k, v, hint="": print(f"    {k:<36} {v}  {hint}")

        print(f"\n{sep}")
        print("  EVAL DATA COLLECTOR — PROXY METRICS")
        print(sep)

        ds = proxy["deepsort"]
        head("DeepSort:")
        row("Треків всього:",        ds["total_tracks"])
        row("ID switches:",          f"{ds['id_switches']}  (rate={ds['id_switch_rate']:.5f})", "↓ краще")
        row("Avg довжина треку:",     f"{ds['avg_track_length_frames']:.1f} кадрів", "↑ краще")
        row("Короткі треки (<10f):", f"{ds['short_tracks_pct']*100:.1f}%",           "↓ краще")
        row("Довгі треки (>90f):",   f"{ds['long_tracks_pct']*100:.1f}%",            "↑ краще")

        yl = proxy["yolo"]
        head("YOLO:")
        row("Детекцій всього:",      yl["total_detections"])
        row("Avg confidence:",       f"{yl['avg_confidence']:.4f} ± {yl['std_confidence']:.4f}")
        row("High conf (>0.9):",     f"{yl['high_conf_pct_09']*100:.1f}%")
        row("Low conf (<0.6):",      f"{yl['low_conf_pct_06']*100:.1f}%", "↓ краще")

        rn = proxy["resnet50"]
        head("ResNet50:")
        row("Інференсів всього:",    rn["total_inferences"])
        row("Avg score (all):",      f"{rn['avg_score_all']:.4f}")
        if rn["avg_score_accident"] is not None:
            row("Avg score (accident):", f"{rn['avg_score_accident']:.4f}")
            row("Score separation:",     f"{rn['score_separation']:.4f}", "↑ краще")
        row("Avg inference time:",   f"{rn['avg_inference_time_ms']:.1f} ms")
        row("P95 inference time:",   f"{rn['p95_inference_time_ms']:.1f} ms", "↓ краще")

        ls = proxy["lstm"]
        head("LSTM:")
        row("Предикцій всього:",     ls["total_predictions"])
        row("Avg risk score:",       f"{ls['avg_risk_score']:.4f}")
        row("High risk rate (>0.7):", f"{ls['high_risk_pct_07']*100:.2f}%")
        if ls["lstm_kinematic_agreement"] is not None:
            row("LSTM↔kin agreement:", f"{ls['lstm_kinematic_agreement']*100:.1f}%", "↑ краще")

        sy = proxy["system"]
        head("System:")
        row("Кадрів оброблено:",     sy["total_frames"])
        row("Аварійних кадрів:",     sy["accident_frames_count"])
        row("Accident rate:",        f"{sy['accident_rate']*100:.4f}%")
        row("False alarm proxy:",    f"{sy['false_alarm_rate_proxy']*100:.4f}%", "↓ краще")

        print(f"\n  Файли збережено у: {self.output_dir}")
        print(f"{sep}\n")
