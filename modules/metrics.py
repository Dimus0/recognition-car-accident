import cv2
import numpy as np
import torch
from torchvision import transforms
import logging
import os
import json
import psutil
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from modules.config.config import Config


class MetricsTracker:

    def __init__(self, ground_truth_path: Optional[str] = None,
                 auto_generate_gt: bool = True):
        self.ground_truth_path = ground_truth_path
        self.auto_generate_gt  = auto_generate_gt
        self.ground_truth      = self._load_ground_truth() if ground_truth_path else None
        self.reset()

    # ----------------------------------------------------------
    #  Ground Truth helpers
    # ----------------------------------------------------------

    def add_ground_truth_frame(self, frame_num: int, vehicles: list,
                               bboxes: list, accident: bool):
        if self.ground_truth is None:
            self.ground_truth = {"frame_annotations": {}}

        self.ground_truth["frame_annotations"][str(frame_num)] = {
            "vehicles": vehicles,
            "bboxes":   bboxes,
            "accident": accident
        }
        if self.ground_truth_path:
            with open(self.ground_truth_path, "w", encoding="utf-8") as f:
                json.dump(self.ground_truth, f, indent=2, ensure_ascii=False)

    def _load_ground_truth(self) -> Optional[Dict]:
        if not os.path.exists(self.ground_truth_path):
            print(f"⚠️ GT не знайдено: {self.ground_truth_path}")
            # BUG FIX #1a: НЕ створюємо порожній файл на диску —
            # це призводило до того, що _auto_generate_ground_truth() потім
            # перезаписував порожній файл своїм pseudo-GT, який наступний
            # запуск читав як "реальний GT" → ідентичні метрики для всіх конфігурацій.
            return None

        if os.path.getsize(self.ground_truth_path) == 0:
            return None

        with open(self.ground_truth_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Ground truth некоректний JSON, пропускаю")
                return None

        # BUG FIX #1b: Якщо файл є auto-generated pseudo-GT з попереднього запуску —
        # відкидаємо його як "реальний GT".  Без цієї перевірки кожен наступний
        # запуск читав pseudo-GT і оцінював модель проти власних передбачень.
        if data.get("auto_generated"):
            print("ℹ️  Знайдено pseudo-GT (auto_generated=true) — ігнорується як real GT")
            return None

        if not data.get("frame_annotations"):
            return None

        print(f"✅ Реальний GT завантажено: {self.ground_truth_path} "
              f"({len(data['frame_annotations'])} кадрів)")
        return data

    # ----------------------------------------------------------
    #  Reset / Init
    # ----------------------------------------------------------

    def reset(self):
        self.metrics = {
            "video_info": {},
            "processing": {
                "total_frames":            0,
                "processing_time_seconds": 0,
                "avg_fps":                 0,
                "frames_with_detections":  0,
            },
            "detection": {
                "yolo": {
                    "total_detections":    0,
                    "avg_confidence":      [],
                    "detections_per_frame": [],
                },
                "cnn": {
                    "total_inferences":              0,
                    "avg_inference_time_ms":         [],
                    # Всі інференси (включаючи heartbeat на звичайних кадрах):
                    "high_confidence_predictions":   0,   # score > 0.9
                    "medium_confidence_predictions": 0,   # score 0.7-0.9
                    "low_confidence_predictions":    0,   # score < 0.7
                    # Тільки інференси що спрацювали НА АВАРІЮ (score > CONF_ACCIDENT_HIGH):
                    "accident_trigger_inferences":   0,
                    "scores_history":                [],
                },
            },
            "tracking": {
                "unique_vehicles":          set(),
                "avg_track_length":         [],
                "max_simultaneous_tracks":  0,
                "tracks_per_frame":         [],   # NEW: для графіку
            },
            "accidents": {
                # unique_incidents — кількість УНІКАЛЬНИХ ДТП-подій (не кадрів!).
                # total_accidents_detected — тепер алiас до unique_incidents (для сумісності).
                "unique_incidents":             0,
                "total_accidents_detected":     0,   # = unique_incidents, оновлюється в finalize
                "false_positives_filtered":     0,
                "collision_warnings":           0,
                "sudden_stops":                 0,
                "accidents_by_frame":           [],
                # Тепер: кількість УНІКАЛЬНИХ авто в кожному інциденті (не per-frame):
                "average_vehicles_per_accident": [],
                "accident_confidences":         [],
                "accident_labels":              [],
                "all_frame_labels":             [],
                "all_frame_scores":             [],
            },
            "performance": {
                "roi_processing_ratio": 0.0,
                "cnn_skip_ratio":       0.0,
                "memory_usage_mb":      0.0,
                "frame_times_ms":       [],   # NEW: сирі значення для перцентилів
            },
            "evaluation": {
                "mAP":    0.0,
                "mAP_50": 0.0,
                "mAP_75": 0.0,
                # NEW: метрики класифікації аварій
                "precision":        0.0,
                "recall":           0.0,
                "f1_score":         0.0,
                "accuracy":         0.0,
                "false_positive_rate": 0.0,
                "true_positive_rate":  0.0,
            },
            "tracking_evaluation": {
                "MOTA":           0.0,
                "MOTP":           0.0,
                "IDF1":           0.0,
                "mostly_tracked": 0,
                "mostly_lost":    0,
                "id_switches":    0,
                "fragmentations": 0,
                "false_positives": 0,
                "false_negatives": 0,
            },
            "roc_data": {
                "fpr":        [],
                "tpr":        [],
                "thresholds": [],
                "auc":        0.0,
            },
            "precision_recall_data": {
                "precision":         [],
                "recall":            [],
                "thresholds":        [],
                "average_precision": 0.0,
            },
            # NEW: LSTM метрики
            "lstm": {
                "enabled":                False,
                "avg_accident_prob":      [],   # середня ймовірність по кадрам
                "max_accident_prob":      [],   # макс ймовірність в кадрі
                "high_risk_tracks_count": [],   # к-сть треків >0.7
                "predictions_timeline":   [],   # (frame, tid, prob)
            },
            # NEW: Latency percentiles
            "latency": {
                "p50_ms":  0.0,
                "p90_ms":  0.0,
                "p95_ms":  0.0,
                "p99_ms":  0.0,
                "min_ms":  0.0,
                "max_ms":  0.0,
            },
            # NEW: швидкісні метрики авто
            "speed_metrics": {
                "avg_speed_px_per_frame": 0.0,
                "max_speed_px_per_frame": 0.0,
                "speed_histogram":        [],
            },
        }

        self.start_time  = None
        self.frame_times = deque(maxlen=100)

        self._current_frame_max_score: float = 0.0

        # ── Лічильники для ROI / CNN-skip ratio ─────────────────────────────
        self._roi_frames_count:     int = 0   # кадрів де активно ROI-зона
        self._cnn_skipped_count:    int = 0   # кадрів де CNN-інференс пропущено
        # ── Зразки пам'яті (МБ) для усереднення ─────────────────────────────
        self._memory_samples: list = []
        self._process = psutil.Process(os.getpid())

        # ── Реєстр інцидентів ────────────────────────────────────────────────
        # Кожна запис = один реальний ДТП-інцидент (не кадр, не трек).
        # {incident_id: {start_frame, last_frame, track_ids: set, max_conf, severity}}
        # Два підтвердження одного треку або сусідніх треків у вікні
        # INCIDENT_MERGE_FRAMES → один і той самий інцидент.
        self._incidents: dict = {}          # incident_id → dict
        self._track_to_incident: dict = {}  # track_id → incident_id
        self._next_incident_id: int = 0
        self.INCIDENT_MERGE_FRAMES = 90     # 3 сек при 30fps — вікно злиття

        # Для MOTA/IDF1
        self.tracking_data = {
            "predictions":             defaultdict(list),
            "ground_truth":            defaultdict(list),
            "id_mappings":             {},
            "id_switches":             0,
            "false_positives_tracking": 0,
            "false_negatives_tracking": 0,
            "matches":                 0,
        }

        # NEW: записи треків по кадрам (для швидкостей)
        self._track_records: dict = defaultdict(list)   # {tid: [(frame, cx, cy)]}
        self._accident_records: list = []               # list of dicts

    # ----------------------------------------------------------
    #  Базові update методи
    # ----------------------------------------------------------

    def set_video_info(self, width, height, fps, total_frames, path):
        self.metrics["video_info"] = {
            "path":             path,
            "resolution":       f"{width}x{height}",
            "fps":              fps,
            "total_frames":     total_frames,
            "duration_seconds": total_frames / fps if fps > 0 else 0,
        }

    def start_processing(self):
        self.start_time = cv2.getTickCount()

    def update_frame_metrics(self, frame_time_ms: float):
        self.frame_times.append(frame_time_ms)
        self.metrics["processing"]["total_frames"] += 1
        self.metrics["performance"]["frame_times_ms"].append(frame_time_ms)
        # Знімаємо зразок пам'яті кожні 30 кадрів (щоб не навантажувати)
        tf = self.metrics["processing"]["total_frames"]
        if tf % 30 == 0:
            self.update_memory_usage()

    def update_memory_usage(self):
        """Фіксує поточне RSS-споживання пам'яті процесу (в МБ)."""
        try:
            mem_mb = self._process.memory_info().rss / (1024 * 1024)
            self._memory_samples.append(mem_mb)
        except Exception:
            pass

    def record_roi_frame(self):
        """Викликається коли кадр оброблявся через ROI-зону."""
        self._roi_frames_count += 1

    def record_cnn_skip(self):
        """Викликається коли CNN-інференс пропущено для кадру."""
        self._cnn_skipped_count += 1

    def update_yolo_metrics(self, detections, confidences):
        self.metrics["detection"]["yolo"]["total_detections"] += len(detections)
        if confidences:
            self.metrics["detection"]["yolo"]["avg_confidence"].extend(confidences)
        self.metrics["detection"]["yolo"]["detections_per_frame"].append(len(detections))
        if detections:
            self.metrics["processing"]["frames_with_detections"] += 1

    def update_cnn_metrics(self, num_inferences: int, inference_time_ms: float, scores: list):
        self.metrics["detection"]["cnn"]["total_inferences"] += num_inferences
        self.metrics["detection"]["cnn"]["avg_inference_time_ms"].append(inference_time_ms)
        self.metrics["detection"]["cnn"]["scores_history"].extend(scores)
        for score in scores:
            if score > 0.9:
                self.metrics["detection"]["cnn"]["high_confidence_predictions"] += 1
            elif score > 0.7:
                self.metrics["detection"]["cnn"]["medium_confidence_predictions"] += 1
            else:
                self.metrics["detection"]["cnn"]["low_confidence_predictions"] += 1
            # Рахуємо тільки ті що реально тригернули аварійну перевірку
            if score >= 0.86:
                self.metrics["detection"]["cnn"]["accident_trigger_inferences"] += 1
        if scores:
            self._current_frame_max_score = max(self._current_frame_max_score, max(scores))

    def update_tracking_metrics(self, active_ids: list, track_history: dict):
        self.metrics["tracking"]["unique_vehicles"].update(active_ids)
        n = len(active_ids)
        self.metrics["tracking"]["tracks_per_frame"].append(n)
        if n > self.metrics["tracking"]["max_simultaneous_tracks"]:
            self.metrics["tracking"]["max_simultaneous_tracks"] = n
        for tid in active_ids:
            if tid in track_history:
                self.metrics["tracking"]["avg_track_length"].append(len(track_history[tid]))

    def update_accident_metrics(self, accidents_this_frame: bool,
                                num_vehicles: int,
                                collision_warnings: int,
                                sudden_stops: int):
        frame_num = self.metrics["processing"]["total_frames"]
        label = 1 if accidents_this_frame else 0
        self.metrics["accidents"]["all_frame_labels"].append(label)
        self.metrics["accidents"]["all_frame_scores"].append(self._current_frame_max_score)
        self._current_frame_max_score = 0.0

        if accidents_this_frame:
            # НЕ інкрементуємо total_accidents_detected тут!
            # Підрахунок унікальних інцидентів відбувається в record_accident()
            # через реєстр _incidents. Це виключає повторний рахунок одного ДТП
            # на кожному кадрі поки воно активне (ACCIDENT_LIFETIME = 90 кадрів).
            self.metrics["accidents"]["accidents_by_frame"].append(frame_num)
        self.metrics["accidents"]["collision_warnings"] += collision_warnings
        self.metrics["accidents"]["sudden_stops"]       += sudden_stops

    # ----------------------------------------------------------
    #  NEW: record_tracks — зберігає позиції треків по кадрам
    # ----------------------------------------------------------
    def record_tracks(self, frame_num: int, boxes: list, ids: list):
        """Зберігає позиції бок-сів для кожного треку (для швидкостей та MOTP)."""
        for tid, (x1, y1, x2, y2) in zip(ids, boxes):
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            self._track_records[tid].append((frame_num, cx, cy))

    # ----------------------------------------------------------
    #  NEW: record_accident
    # ----------------------------------------------------------
    def record_accident(self, frame_num: int, accident_type: str,
                        confidence: float, involved_tracks: list, severity: str):
        """
        Детальний запис аварійної події + дедуплікація інцидентів.

        Логіка підрахунку унікальних ДТП:
        ----------------------------------
        Проблема: один ДТП активний 90 кадрів (ACCIDENT_LIFETIME).
        Без дедуплікації → 90 записів → «50 аварій» замість 1-2.

        Рішення — реєстр _incidents:
        1. При першому підтвердженні треку (type='collision') → новий інцидент.
        2. Якщо цей же трек знову підтверджується в межах INCIDENT_MERGE_FRAMES
           → той самий інцидент (не новий).
        3. Якщо РІЗНІ треки підтверджуються в межах INCIDENT_MERGE_FRAMES
           → перевіряємо чи не перетинаються з існуючим інцидентом.
           Якщо так — merge (id=6 вдарився в id=5 після id=4+id=5 → один інцидент).
        4. Кількість авто = len(incident['track_ids']) — унікальні, не сума по кадрах.
        """
        self._accident_records.append({
            "frame":            frame_num,
            "type":             accident_type,
            "confidence":       confidence,
            "involved_tracks":  involved_tracks,
            "severity":         severity,
            "timestamp":        datetime.now().isoformat(),
        })
        self.metrics["accidents"]["accident_confidences"].append(confidence)

        # Тільки перше підтвердження треку рахується як потенційно новий інцидент
        if accident_type not in ('collision', 'confirmed'):
            return

        for track_id in involved_tracks:
            # Перевіряємо чи цей трек вже прив'язаний до активного інциденту
            existing_incident_id = self._track_to_incident.get(track_id)
            if existing_incident_id is not None:
                inc = self._incidents.get(existing_incident_id)
                if inc and (frame_num - inc['last_frame']) <= self.INCIDENT_MERGE_FRAMES:
                    # Той самий інцидент — оновлюємо метадані
                    inc['last_frame'] = frame_num
                    inc['max_conf']   = max(inc['max_conf'], confidence)
                    inc['track_ids'].add(track_id)
                    continue
                # Інцидент застарів → трек може стати частиною нового

            # Шукаємо активний інцидент до якого можна приєднати цей трек
            # (наприклад: id=5 вже в інциденті 1, тепер id=5 б'є id=6 → merge)
            merged = False
            for iid, inc in self._incidents.items():
                if (frame_num - inc['last_frame']) <= self.INCIDENT_MERGE_FRAMES:
                    if track_id in inc['track_ids']:
                        # Цей трек вже в активному інциденті → merge нових треків
                        for new_tid in involved_tracks:
                            inc['track_ids'].add(new_tid)
                            self._track_to_incident[new_tid] = iid
                        inc['last_frame'] = frame_num
                        inc['max_conf']   = max(inc['max_conf'], confidence)
                        merged = True
                        break

            if not merged:
                # Новий інцидент
                iid = self._next_incident_id
                self._next_incident_id += 1
                self._incidents[iid] = {
                    'start_frame': frame_num,
                    'last_frame':  frame_num,
                    'track_ids':   set(involved_tracks),
                    'max_conf':    confidence,
                    'severity':    severity,
                }
                for new_tid in involved_tracks:
                    self._track_to_incident[new_tid] = iid

    # ----------------------------------------------------------
    #  NEW: update_tracking_for_evaluation
    # ----------------------------------------------------------
    def update_tracking_for_evaluation(self, frame_num: int,
                                       predicted_tracks: list,
                                       iou_threshold: float = 0.5):
        """
        Зберігає predicted треки для frame_num і одразу
        порівнює з ground truth якщо є.
        predicted_tracks: list of (tid, bbox, conf)

        ВАЖЛИВО: predictions зберігаються ЗАВЖДИ — навіть якщо ground_truth
        ще не завантажено. Це необхідно для auto_generate_gt режиму, де
        _auto_generate_ground_truth() будує pseudo-GT з цих predictions у finalize().
        """
        # Завжди зберігаємо predictions (потрібно для auto_generate_gt)
        self.tracking_data["predictions"][frame_num] = predicted_tracks

        # Порівнюємо з GT тільки якщо він вже є
        if self.ground_truth:
            gt_tracks = self._get_ground_truth_tracks(frame_num)
            self.tracking_data["ground_truth"][frame_num] = gt_tracks

            matches, fp, fn, id_sw = self._match_tracks(
                predicted_tracks, gt_tracks, iou_threshold
            )
            self.tracking_data["matches"]                  += matches
            self.tracking_data["false_positives_tracking"] += fp
            self.tracking_data["false_negatives_tracking"] += fn
            self.tracking_data["id_switches"]              += id_sw

    # ----------------------------------------------------------
    #  NEW: update_lstm_metrics
    # ----------------------------------------------------------
    def update_lstm_metrics(self, lstm_probs: dict):
        """
        lstm_probs: {tid: probability}
        """
        if not lstm_probs:
            return
        self.metrics["lstm"]["enabled"] = True
        probs = list(lstm_probs.values())
        self.metrics["lstm"]["avg_accident_prob"].append(float(np.mean(probs)))
        self.metrics["lstm"]["max_accident_prob"].append(float(np.max(probs)))
        high = sum(1 for p in probs if p >= 0.7)
        self.metrics["lstm"]["high_risk_tracks_count"].append(high)

    # ----------------------------------------------------------
    #  Ground Truth helpers
    # ----------------------------------------------------------

    def _is_accident_in_ground_truth(self, frame_num: int) -> bool:
        if not self.ground_truth or "frame_annotations" not in self.ground_truth:
            return False
        ann = self.ground_truth["frame_annotations"].get(str(frame_num), {})
        return ann.get("accident", False)

    def _get_ground_truth_tracks(self, frame_num: int) -> List[Tuple]:
        if not self.ground_truth or "frame_annotations" not in self.ground_truth:
            return []
        ann = self.ground_truth["frame_annotations"].get(str(frame_num), {})
        if "vehicles" in ann and "bboxes" in ann:
            return list(zip(ann["vehicles"], ann["bboxes"]))
        return []

    # ----------------------------------------------------------
    #  Matching / IoU
    # ----------------------------------------------------------

    def _match_tracks(self, predicted: list, ground_truth: list,
                      iou_threshold: float) -> Tuple[int, int, int, int]:
        if not predicted and not ground_truth:
            return 0, 0, 0, 0
        if not predicted:
            return 0, 0, len(ground_truth), 0
        if not ground_truth:
            return 0, len(predicted), 0, 0

        iou_matrix = np.zeros((len(predicted), len(ground_truth)))
        for i, (pred_id, pred_bbox, *_) in enumerate(predicted):
            for j, (gt_id, gt_bbox, *_) in enumerate(ground_truth):
                iou_matrix[i, j] = self._compute_iou(pred_bbox, gt_bbox)

        matches, id_switches = 0, 0
        matched_pred = set()
        matched_gt   = set()

        for idx in np.argsort(-iou_matrix.flatten()):
            i = idx // len(ground_truth)
            j = idx % len(ground_truth)
            if i in matched_pred or j in matched_gt:
                continue
            if iou_matrix[i, j] >= iou_threshold:
                matches += 1
                matched_pred.add(i)
                matched_gt.add(j)
                pred_id = predicted[i][0]
                gt_id   = ground_truth[j][0]
                if pred_id in self.tracking_data["id_mappings"]:
                    if self.tracking_data["id_mappings"][pred_id] != gt_id:
                        id_switches += 1
                self.tracking_data["id_mappings"][pred_id] = gt_id

        fp = len(predicted)  - matches
        fn = len(ground_truth) - matches
        return matches, fp, fn, id_switches

    def _compute_iou(self, box1, box2) -> float:
        x1a, y1a, x2a, y2a = box1
        x1b, y1b, x2b, y2b = box2
        xi1 = max(x1a, x1b);  yi1 = max(y1a, y1b)
        xi2 = min(x2a, x2b);  yi2 = min(y2a, y2b)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        a1 = (x2a - x1a) * (y2a - y1a)
        a2 = (x2b - x1b) * (y2b - y1b)
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    # ----------------------------------------------------------
    #  Compute mAP
    # ----------------------------------------------------------

    def compute_map(self, iou_thresholds: List[float] = [0.5, 0.75]):
        if not self.ground_truth:
            return
        all_preds, all_gts = [], []
        for fn in self.tracking_data["predictions"]:
            preds = self.tracking_data["predictions"][fn]
            gts   = self.tracking_data["ground_truth"].get(fn, [])
            all_preds.extend([(fn, *p) for p in preds])
            all_gts.extend([(fn, *g) for g in gts])

        aps = []
        for thr in iou_thresholds:
            ap = self._compute_average_precision(all_preds, all_gts, thr)
            aps.append(ap)
            if thr == 0.5:
                self.metrics["evaluation"]["mAP_50"] = round(ap, 4)
            elif thr == 0.75:
                self.metrics["evaluation"]["mAP_75"] = round(ap, 4)
        self.metrics["evaluation"]["mAP"] = round(float(np.mean(aps)), 4)

    def _compute_average_precision(self, predictions, ground_truths, iou_threshold):
        if not predictions or not ground_truths:
            return 0.0
        predictions = sorted(predictions, key=lambda x: x[3], reverse=True)
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        gt_matched = set()
        for i, pred in enumerate(predictions):
            frame, pred_id, pred_bbox, conf = pred
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(ground_truths):
                if gt[0] != frame:
                    continue
                iou = self._compute_iou(pred_bbox, gt[2])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold and best_j not in gt_matched:
                tp[i] = 1
                gt_matched.add(best_j)
            else:
                fp[i] = 1
        tpc = np.cumsum(tp)
        fpc = np.cumsum(fp)
        recalls    = tpc / max(len(ground_truths), 1)
        precisions = tpc / (tpc + fpc + 1e-9)
        ap = float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))
        return ap

    # ----------------------------------------------------------
    #  Compute MOTA / IDF1
    # ----------------------------------------------------------

    def compute_mota_idf1(self):
        total_gt = total_matches = total_fp = total_fn = total_id_sw = 0
        for frame_num in self.tracking_data["predictions"]:
            pred = self.tracking_data["predictions"][frame_num]
            gt   = self.tracking_data["ground_truth"].get(frame_num, [])
            m, fp, fn, id_sw = self._match_tracks(pred, gt, iou_threshold=0.3)
            total_matches += m
            total_fp      += fp
            total_fn      += fn
            total_id_sw   += id_sw
            total_gt      += len(gt)

        if total_gt > 0:
            mota = 1 - (total_fn + total_fp + total_id_sw) / total_gt
        else:
            mota = 0.0
        denom = 2 * total_matches + total_fp + total_fn
        idf1  = 2 * total_matches / denom if denom > 0 else 0.0

        self.metrics["tracking_evaluation"]["MOTA"]            = round(mota, 4)
        self.metrics["tracking_evaluation"]["IDF1"]            = round(idf1, 4)
        self.metrics["tracking_evaluation"]["id_switches"]     = total_id_sw
        self.metrics["tracking_evaluation"]["false_positives"] = total_fp
        self.metrics["tracking_evaluation"]["false_negatives"] = total_fn

    # ----------------------------------------------------------
    #  NEW: Compute classification metrics (Precision/Recall/F1/ROC)
    # ----------------------------------------------------------

    def compute_classification_metrics(self):
        """
        Обчислює Precision, Recall, F1, Accuracy, ROC-AUC, PR-AUC
        для детекції аварій на рівні кадрів.

        BUG FIX #3 (self-evaluation):
        --------------------------------
        Стара логіка: labels = all_frame_labels, де label=1 виставлявся коли
        МОДЕЛЬ детектувала аварію. Тобто мітки будувались із виходів моделі,
        а не з розмітки. ROC-AUC = 0.9945 для ВСІХ конфігурацій — не тому що
        модель хороша, а тому що вона оцінювала сама себе.

        Нова логіка:
        - Якщо є РЕАЛЬНИЙ GT (не auto_generated) → будуємо labels з нього
          побитово по frame_num. Це справжня зовнішня оцінка.
        - Якщо реального GT немає → fallback на model-generated labels,
          але попереджаємо що це pseudo-evaluation.
        """
        scores_raw = self.metrics["accidents"]["all_frame_scores"]
        if len(scores_raw) < 2:
            return

        total_frames = len(scores_raw)
        scores = np.array(scores_raw, dtype=float)

        is_real_gt = (
            self.ground_truth is not None
            and not self.ground_truth.get("auto_generated", False)
            and "frame_annotations" in self.ground_truth
        )

        if is_real_gt:
            # ── Реальна зовнішня оцінка ────────────────────────────────────────
            # GT розмітка по frame_num → вектор labels довжиною total_frames
            labels = np.zeros(total_frames, dtype=int)
            for fstr, ann in self.ground_truth["frame_annotations"].items():
                if ann.get("accident", False):
                    fn = int(fstr)
                    # frame_count в console.py починається з 1
                    if 1 <= fn <= total_frames:
                        labels[fn - 1] = 1
            print("ℹ️  compute_classification_metrics: використовується РЕАЛЬНИЙ GT")
        else:
            # ── Pseudo-evaluation (model vs model) ─────────────────────────────
            labels_raw = self.metrics["accidents"]["all_frame_labels"]
            n = min(total_frames, len(labels_raw))
            labels = np.array(labels_raw[:n], dtype=int)
            scores = scores[:n]
            print("⚠️  compute_classification_metrics: GT відсутній — pseudo-evaluation "
                  "(model vs model). Метрики оптимістичні.")

        n = min(len(scores), len(labels))
        scores, labels = scores[:n], labels[:n]

        if len(np.unique(labels)) < 2:
            print("⚠️  compute_classification_metrics: у labels тільки один клас — пропускаємо.")
            return

        # ROC
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        self.metrics["roc_data"]["fpr"]        = fpr.tolist()
        self.metrics["roc_data"]["tpr"]        = tpr.tolist()
        self.metrics["roc_data"]["thresholds"] = thresholds.tolist()
        self.metrics["roc_data"]["auc"]        = round(float(roc_auc), 4)

        # Precision-Recall
        precision_arr, recall_arr, pr_thresh = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        self.metrics["precision_recall_data"]["precision"]         = precision_arr.tolist()
        self.metrics["precision_recall_data"]["recall"]            = recall_arr.tolist()
        self.metrics["precision_recall_data"]["thresholds"]        = pr_thresh.tolist()
        self.metrics["precision_recall_data"]["average_precision"] = round(float(ap), 4)

        # При фіксованому порозі CONF_ACCIDENT_LOW
        predicted_labels = (scores >= Config.CONF_ACCIDENT_LOW).astype(int)
        tp = int(np.sum((predicted_labels == 1) & (labels == 1)))
        fp = int(np.sum((predicted_labels == 1) & (labels == 0)))
        fn = int(np.sum((predicted_labels == 0) & (labels == 1)))
        tn = int(np.sum((predicted_labels == 0) & (labels == 0)))

        precision  = tp / (tp + fp + 1e-9)
        recall     = tp / (tp + fn + 1e-9)
        f1         = 2 * precision * recall / (precision + recall + 1e-9)
        accuracy   = (tp + tn) / (tp + fp + fn + tn + 1e-9)
        fpr_val    = fp / (fp + tn + 1e-9)

        self.metrics["evaluation"]["precision"]           = round(float(precision), 4)
        self.metrics["evaluation"]["recall"]              = round(float(recall), 4)
        self.metrics["evaluation"]["f1_score"]            = round(float(f1), 4)
        self.metrics["evaluation"]["accuracy"]            = round(float(accuracy), 4)
        self.metrics["evaluation"]["false_positive_rate"] = round(float(fpr_val), 4)
        self.metrics["evaluation"]["true_positive_rate"]  = round(float(recall), 4)
        # Зберігаємо прапор щоб ablation runner знав чи це реальна оцінка
        self.metrics["evaluation"]["used_real_gt"]        = bool(is_real_gt)

    # ----------------------------------------------------------
    #  NEW: Compute latency percentiles
    # ----------------------------------------------------------

    def compute_latency_percentiles(self):
        times = self.metrics["performance"]["frame_times_ms"]
        if not times:
            return
        arr = np.array(times, dtype=float)
        self.metrics["latency"]["p50_ms"] = round(float(np.percentile(arr, 50)), 2)
        self.metrics["latency"]["p90_ms"] = round(float(np.percentile(arr, 90)), 2)
        self.metrics["latency"]["p95_ms"] = round(float(np.percentile(arr, 95)), 2)
        self.metrics["latency"]["p99_ms"] = round(float(np.percentile(arr, 99)), 2)
        self.metrics["latency"]["min_ms"] = round(float(np.min(arr)), 2)
        self.metrics["latency"]["max_ms"] = round(float(np.max(arr)), 2)

    def compute_speed_metrics(self):
        speeds = []
        for tid, records in self._track_records.items():
            for k in range(1, len(records)):
                dx = records[k][1] - records[k-1][1]
                dy = records[k][2] - records[k-1][2]
                speeds.append(np.sqrt(dx**2 + dy**2))
        if not speeds:
            return
        arr = np.array(speeds)
        self.metrics["speed_metrics"]["avg_speed_px_per_frame"] = round(float(np.mean(arr)), 2)
        self.metrics["speed_metrics"]["max_speed_px_per_frame"] = round(float(np.max(arr)), 2)
        hist, _ = np.histogram(arr, bins=20)
        self.metrics["speed_metrics"]["speed_histogram"] = hist.tolist()


    def _auto_generate_ground_truth(self):
        """
        Будує pseudo-GT із збережених predicted треків:
        якщо хоча б один трек у кадрі має accident_confidence >= 0.86,
        кадр позначається як аварійний.

        BUG FIX #2: Pseudo-GT зберігається у ОКРЕМИЙ файл з префіксом 'pseudo_',
        НІКОЛИ не перезаписує реальний ground_truth_path.
        Це запобігає ситуації, коли наступний запуск читає pseudo-GT як реальну
        розмітку і оцінює модель проти власних передбачень.
        """
        if not self.tracking_data["predictions"]:
            return

        annotations = {}
        accident_confidences = {
            r["frame"]: r["confidence"] for r in self._accident_records
        }

        for frame_num, preds in self.tracking_data["predictions"].items():
            conf = accident_confidences.get(frame_num, 0.0)
            bboxes   = [list(p[1]) for p in preds]
            vehicles = [p[0] for p in preds]
            annotations[str(frame_num)] = {
                "vehicles": vehicles,
                "bboxes":   bboxes,
                "accident": conf >= 0.86,
            }

        self.ground_truth = {
            "frame_annotations": annotations,
            "auto_generated": True,   # маркер — не читати як реальний GT
        }

        if self.ground_truth_path:
            try:
                gt_dir  = os.path.dirname(self.ground_truth_path)
                gt_base = os.path.splitext(os.path.basename(self.ground_truth_path))[0]
                # BUG FIX #2: окремий шлях —ніколи не перезаписуємо реальний GT
                pseudo_path = os.path.join(gt_dir, f"pseudo_{gt_base}.json")
                os.makedirs(gt_dir, exist_ok=True)
                with open(pseudo_path, "w", encoding="utf-8") as f:
                    json.dump(self.ground_truth, f, indent=2, ensure_ascii=False)
                print(f"✅ Pseudo-GT збережено (окремо): {pseudo_path}")
            except Exception as e:
                print(f"⚠️ Не вдалось зберегти pseudo-GT: {e}")

    def _populate_tracking_gt_from_gt(self):
        """
        BUG FIX #4 (MOTA/IDF1 = 0):
        --------------------------------
        Проблема: під час головного циклу processing `self.ground_truth` = None
        (GT генерується тільки в finalize() → _auto_generate_ground_truth()).
        Тому `update_tracking_for_evaluation()` ніколи не заповнювала
        `tracking_data["ground_truth"]` → у `compute_mota_idf1()` total_gt = 0
        → MOTA = 0, IDF1 = 0 для всіх конфігурацій.

        Виправлення: після того як GT стає доступним (реальний або pseudo),
        ретроактивно заповнюємо `tracking_data["ground_truth"]` з анотацій GT,
        потім викликаємо compute_mota_idf1() вже з коректними GT-треками.
        """
        if not self.ground_truth:
            return

        ann = self.ground_truth.get("frame_annotations", {})
        if not ann:
            return

        filled = 0
        for frame_str, frame_ann in ann.items():
            frame_num = int(frame_str)
            vehicles  = frame_ann.get("vehicles", [])
            bboxes    = frame_ann.get("bboxes", [])
            if not vehicles or not bboxes:
                continue
            # Формат: list of (tid, bbox, conf=1.0)
            gt_tracks = [
                (int(vid), tuple(map(int, bbox)), 1.0)
                for vid, bbox in zip(vehicles, bboxes)
                if len(bbox) == 4
            ]
            if gt_tracks:
                self.tracking_data["ground_truth"][frame_num] = gt_tracks
                filled += 1

        print(f"ℹ️  _populate_tracking_gt_from_gt: заповнено GT для {filled} кадрів "
              f"({'real' if not self.ground_truth.get('auto_generated') else 'pseudo'} GT)")

    # ----------------------------------------------------------
    #  Finalize
    # ----------------------------------------------------------

    def finalize(self) -> dict:
        end_time = cv2.getTickCount()
        if self.start_time:
            elapsed = (end_time - self.start_time) / cv2.getTickFrequency()
            self.metrics["processing"]["processing_time_seconds"] = round(elapsed, 2)
            total_frames = self.metrics["processing"]["total_frames"]
            if total_frames > 0:
                self.metrics["processing"]["avg_fps"] = round(total_frames / elapsed, 2)

        # Усереднення списків → скаляри
        def avg(lst): return round(float(np.mean(lst)), 3) if lst else 0.0

        yolo_conf = self.metrics["detection"]["yolo"]["avg_confidence"]
        if isinstance(yolo_conf, list):
            self.metrics["detection"]["yolo"]["avg_confidence"] = avg(yolo_conf)

        cnn_t = self.metrics["detection"]["cnn"]["avg_inference_time_ms"]
        if isinstance(cnn_t, list):
            self.metrics["detection"]["cnn"]["avg_inference_time_ms"] = avg(cnn_t)

        tl = self.metrics["tracking"]["avg_track_length"]
        if isinstance(tl, list):
            self.metrics["tracking"]["avg_track_length"] = avg(tl)

        # ── Рахуємо унікальні інциденти з реєстру ──────────────────────────────
        # Кожен інцидент = одна реальна ДТП-подія, незалежно від тривалості.
        unique_count = len(self._incidents)
        self.metrics["accidents"]["unique_incidents"]         = unique_count
        self.metrics["accidents"]["total_accidents_detected"] = unique_count

        # Середня кількість авто = унікальні track_ids на інцидент (не per-frame сума)
        if self._incidents:
            vehicles_per = [len(inc['track_ids']) for inc in self._incidents.values()]
            self.metrics["accidents"]["average_vehicles_per_accident"] = round(
                float(np.mean(vehicles_per)), 2
            )
        else:
            av = self.metrics["accidents"]["average_vehicles_per_accident"]
            if isinstance(av, list):
                self.metrics["accidents"]["average_vehicles_per_accident"] = avg(av)

        # set → int
        uniq = self.metrics["tracking"]["unique_vehicles"]
        self.metrics["tracking"]["unique_vehicles"] = len(uniq) if isinstance(uniq, set) else uniq

        # Detection rate
        tf = self.metrics["processing"]["total_frames"]
        if tf > 0:
            self.metrics["performance"]["detection_rate"] = round(
                self.metrics["processing"]["frames_with_detections"] / tf, 4
            )
            # ── ROI / CNN-skip ratios ────────────────────────────────────────
            self.metrics["performance"]["roi_processing_ratio"] = round(
                self._roi_frames_count / tf, 4
            )
            self.metrics["performance"]["cnn_skip_ratio"] = round(
                self._cnn_skipped_count / tf, 4
            )

        # ── Memory usage (average of samples, fallback to current) ──────────
        self.update_memory_usage()   # фінальний зразок
        if self._memory_samples:
            self.metrics["performance"]["memory_usage_mb"] = round(
                float(np.mean(self._memory_samples)), 2
            )
        else:
            try:
                self.metrics["performance"]["memory_usage_mb"] = round(
                    self._process.memory_info().rss / (1024 * 1024), 2
                )
            except Exception:
                self.metrics["performance"]["memory_usage_mb"] = 0.0

        # Розширені обчислення
        # auto_generate_gt — якщо GT не завантажено, будуємо pseudo-GT
        # з впевнених передбачень (score > 0.86). Дозволяє рахувати mAP/MOTA
        # без ручної розмітки. Результати будуть оптимістичними, але корисними
        # для відлагодження моделі.
        if self.ground_truth is None and self.auto_generate_gt:
            self._auto_generate_ground_truth()

        # BUG FIX #4: Ретроактивно заповнюємо GT-треки ДО compute_mota_idf1.
        # Раніше tracking_data["ground_truth"] залишався порожнім бо GT
        # ставав доступним тільки тут, після основного циклу.
        if self.ground_truth:
            self._populate_tracking_gt_from_gt()
            self.compute_map()
            self.compute_mota_idf1()

        self.compute_classification_metrics()
        self.compute_latency_percentiles()
        self.compute_speed_metrics()

        # Зберігаємо детальні записи аварій
        self.metrics["accident_records"] = self._accident_records

        return self.metrics

    # ----------------------------------------------------------
    #  Save / Print
    # ----------------------------------------------------------

    def save_to_file(self, filepath: str):
        """Зберігає метрики у JSON (очищаємо списки що не серіалізуються)."""
        def make_serializable(obj):
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
            if isinstance(obj, set):            return list(obj)
            return obj

        def recurse(d):
            if isinstance(d, dict):
                return {k: recurse(v) for k, v in d.items()}
            if isinstance(d, list):
                return [recurse(i) for i in d]
            return make_serializable(d)

        safe = recurse(self.metrics)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(safe, f, indent=2, ensure_ascii=False)

    def _build_summary_lines(self) -> list:
        """
        Будує список рядків зведення метрик.
        Використовується як print_summary() так і save_summary_to_file().
        Повертає list[str] — без символів нового рядка на кінці.
        """
        m    = self.metrics
        sep  = "=" * 70
        lines = []

        def ln(s=""): lines.append(s)

        ln(); ln(sep); ln("  METRICS SUMMARY"); ln(sep)

        ln()
        ln("VIDEO:")
        ln(f"  Path:                {m['video_info'].get('path', 'N/A')}")
        ln(f"  Resolution:          {m['video_info'].get('resolution', 'N/A')}")
        ln(f"  Video FPS:           {m['video_info'].get('fps', 'N/A')}")
        ln(f"  Duration:            {m['video_info'].get('duration_seconds', 0):.1f} sec")

        ln()
        ln("PERFORMANCE:")
        ln(f"  Frames processed:    {m['processing']['total_frames']}")
        ln(f"  Processing time:     {m['processing']['processing_time_seconds']} sec")
        ln(f"  Average FPS:         {m['processing']['avg_fps']}")
        lat = m.get("latency", {})
        ln(f"  Latency P50/P95/P99: "
           f"{lat.get('p50_ms',0)}/{lat.get('p95_ms',0)}/{lat.get('p99_ms',0)} ms")
        ln(f"  Memory usage:        {m['performance'].get('memory_usage_mb', 0):.1f} MB")

        ln()
        ln("YOLO:")
        ln(f"  Total detections:    {m['detection']['yolo']['total_detections']}")
        ln(f"  Avg confidence:      {m['detection']['yolo']['avg_confidence']}")
        det_pf = m['detection']['yolo']['detections_per_frame']
        ln(f"  Avg per frame:       {np.mean(det_pf):.1f}" if det_pf else
           f"  Avg per frame:       N/A")

        ln()
        cnn         = m['detection']['cnn']
        total_inf   = cnn['total_inferences']
        acc_trigger = cnn.get('accident_trigger_inferences', 0)
        heartbeat   = total_inf - acc_trigger
        ln("CNN:")
        ln(f"  Total inferences:    {total_inf}")
        ln(f"    heartbeat:         {heartbeat}  (scheduled checks)")
        ln(f"    accident triggers: {acc_trigger}  (score >= 0.86)")
        ln(f"  Avg inference time:  {cnn['avg_inference_time_ms']} ms")
        if acc_trigger > 0:
            ln(f"  High conf (>0.9):   {cnn['high_confidence_predictions']}")
            ln(f"  Med  conf (0.7-0.9):{cnn['medium_confidence_predictions']}")
        else:
            ln(f"  High/Med/Low:        "
               f"{cnn['high_confidence_predictions']}/"
               f"{cnn['medium_confidence_predictions']}/"
               f"{cnn['low_confidence_predictions']}  (mostly heartbeat)")

        ln()
        ln("TRACKING:")
        ln(f"  Unique vehicles:     {m['tracking']['unique_vehicles']}")
        ln(f"  Max simultaneous:    {m['tracking']['max_simultaneous_tracks']}")
        ln(f"  Avg track length:    {m['tracking']['avg_track_length']}")

        ln()
        ln("ACCIDENTS:")
        unique_inc = m['accidents'].get('unique_incidents', 0)
        ln(f"  Unique incidents:    {unique_inc}")
        for iid, inc in (self._incidents.items() if self._incidents else {}.items()):
            tids = sorted(inc['track_ids'])
            dur  = inc['last_frame'] - inc['start_frame']
            ln(f"    Incident #{iid+1}: frames {inc['start_frame']}-{inc['last_frame']} "
               f"({dur} frames) | vehicle IDs={tids} | "
               f"max_conf={inc['max_conf']:.3f} | {inc['severity']}")
        ln(f"  Avg vehicles/incident: {m['accidents']['average_vehicles_per_accident']}")
        cw = m['accidents']['collision_warnings']
        ss = m['accidents']['sudden_stops']
        if cw: ln(f"  TTC warnings:        {cw}")
        if ss: ln(f"  Sudden stops:        {ss}")

        ln()
        ev = m.get("evaluation", {})
        ln("DETECTION QUALITY:")
        ln(f"  Precision:           {ev.get('precision', 0):.4f}")
        ln(f"  Recall:              {ev.get('recall', 0):.4f}")
        ln(f"  F1-Score:            {ev.get('f1_score', 0):.4f}")
        ln(f"  Accuracy:            {ev.get('accuracy', 0):.4f}")
        ln(f"  ROC-AUC:             {m['roc_data'].get('auc', 0):.4f}")
        ln(f"  PR-AUC (AP):         {m['precision_recall_data'].get('average_precision', 0):.4f}")
        ln(f"  mAP:                 {ev.get('mAP', 0):.4f}")
        ln(f"  mAP@0.5:             {ev.get('mAP_50', 0):.4f}")
        ln(f"  mAP@0.75:            {ev.get('mAP_75', 0):.4f}")

        ln()
        te = m.get("tracking_evaluation", {})
        ln("TRACKING EVALUATION (MOTA/IDF1):")
        ln(f"  MOTA:                {te.get('MOTA', 0):.4f}")
        ln(f"  IDF1:                {te.get('IDF1', 0):.4f}")
        ln(f"  ID Switches:         {te.get('id_switches', 0)}")
        ln(f"  False Positives:     {te.get('false_positives', 0)}")
        ln(f"  False Negatives:     {te.get('false_negatives', 0)}")

        lstm = m.get("lstm", {})
        if lstm.get("enabled"):
            avg_probs = lstm.get("avg_accident_prob", [])
            ln()
            ln("LSTM PREDICTIONS:")
            ln(f"  Status:              enabled")
            ln((f"  Avg probability:     {np.mean(avg_probs):.4f}")
               if avg_probs else "  Data:                N/A")

        sm = m.get("speed_metrics", {})
        if sm.get("avg_speed_px_per_frame", 0) > 0:
            ln()
            ln("VEHICLE SPEED:")
            ln(f"  Average:             {sm['avg_speed_px_per_frame']} px/frame")
            ln(f"  Maximum:             {sm['max_speed_px_per_frame']} px/frame")

        ln(); ln(sep); ln()
        return lines

    def print_summary(self):
        """Виводить зведення метрик у консоль."""
        for line in self._build_summary_lines():
            print(line)

    def save_summary_to_file(self, filepath: str):
        """
        Зберігає текстове зведення метрик у файл.

        Parameters
        ----------
        filepath : str
            Повний шлях до файлу (наприклад, .../logs_video/metrics_summary.txt).
            Директорія створюється автоматично.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        lines = self._build_summary_lines()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  Metrics summary saved: {filepath}")

    # ----------------------------------------------------------
    #  NEW: save_all_plots — зберігає всі графіки
    # ----------------------------------------------------------

    def save_all_plots(self, output_dir: str):
        """Генерує і зберігає всі графіки у output_dir."""
        os.makedirs(output_dir, exist_ok=True)

        DARK_BG   = "#000000"
        PANEL_BG  = "#000000"
        GRID_CLR  = "#21262d"
        TEXT_CLR  = "#c9d1d9"
        BLUE      = "#58a6ff"
        GREEN     = "#3fb950"
        RED       = "#f78166"
        ORANGE    = "#ffa657"
        PURPLE    = "#d2a8ff"

        def _style_ax(ax):
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT_CLR, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID_CLR)
            ax.grid(color=GRID_CLR, lw=0.6, alpha=0.7)
            ax.title.set_color(TEXT_CLR)
            ax.xaxis.label.set_color(TEXT_CLR)
            ax.yaxis.label.set_color(TEXT_CLR)

        # ── 1. Performance Dashboard ──────────────────────────────

        fig = plt.figure(figsize=(18, 10), facecolor=DARK_BG)
        fig.suptitle("System Performance Dashboard", color="white",
                     fontsize=14, fontweight="bold", y=0.98)
        gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

        # 1a. Frame time
        ax = fig.add_subplot(gs[0, 0])
        _style_ax(ax)
        times = self.metrics["performance"]["frame_times_ms"]
        if times:
            ax.plot(times, color=BLUE, lw=0.8, alpha=0.7, label="Frame time")
            lat = self.metrics.get("latency", {})
            for pct, val, clr in [("P50", lat.get("p50_ms", 0), GREEN),
                                   ("P95", lat.get("p95_ms", 0), ORANGE),
                                   ("P99", lat.get("p99_ms", 0), RED)]:
                ax.axhline(val, color=clr, lw=1.2, ls="--", label=f"{pct}={val:.1f}ms")
            ax.legend(fontsize=7, framealpha=0, labelcolor=TEXT_CLR)
        ax.set_title("Frame Processing Time")
        ax.set_xlabel("Frame"); ax.set_ylabel("ms")

        # 1b. Detections per frame
        ax = fig.add_subplot(gs[0, 1])
        _style_ax(ax)
        dpf = self.metrics["detection"]["yolo"]["detections_per_frame"]
        if dpf:
            ax.plot(dpf, color=GREEN, lw=1.0, alpha=0.8)
            ax.fill_between(range(len(dpf)), dpf, alpha=0.2, color=GREEN)
        ax.set_title("YOLO Detections / Frame")
        ax.set_xlabel("Frame"); ax.set_ylabel("Count")

        # 1c. Tracks per frame
        ax = fig.add_subplot(gs[0, 2])
        _style_ax(ax)
        tpf = self.metrics["tracking"].get("tracks_per_frame", [])
        if tpf:
            ax.plot(tpf, color=PURPLE, lw=1.0)
            ax.fill_between(range(len(tpf)), tpf, alpha=0.2, color=PURPLE)
        ax.set_title("Active Tracks / Frame")
        ax.set_xlabel("Frame"); ax.set_ylabel("Count")

        # 1d. CNN score histogram
        ax = fig.add_subplot(gs[1, 0])
        _style_ax(ax)
        scores_hist = self.metrics["detection"]["cnn"]["scores_history"]
        if scores_hist:
            ax.hist(scores_hist, bins=40, color=BLUE, alpha=0.8, edgecolor=DARK_BG, lw=0.3)
            ax.axvline(0.5, color=ORANGE, lw=1.5, ls="--", label="0.5")
            ax.axvline(0.9, color=RED, lw=1.5, ls="--", label="0.9")
            ax.legend(fontsize=7, framealpha=0, labelcolor=TEXT_CLR)
        ax.set_title("CNN Score Distribution")
        ax.set_xlabel("Score"); ax.set_ylabel("Count")

        # 1e. Accident timeline
        ax = fig.add_subplot(gs[1, 1])
        _style_ax(ax)
        total_f = self.metrics["processing"]["total_frames"]
        acc_frames = self.metrics["accidents"]["accidents_by_frame"]
        if total_f > 0:
            timeline = np.zeros(total_f)
            for f in acc_frames:
                if 0 < f <= total_f:
                    timeline[f - 1] = 1
            ax.fill_between(range(total_f), timeline, color=RED, alpha=0.8, step="mid")
        ax.set_title("Accident Timeline")
        ax.set_xlabel("Frame"); ax.set_ylabel("Accident")

        # 1f. Speed histogram
        ax = fig.add_subplot(gs[1, 2])
        _style_ax(ax)
        speed_hist = self.metrics["speed_metrics"].get("speed_histogram", [])
        if speed_hist:
            ax.bar(range(len(speed_hist)), speed_hist, color=ORANGE, alpha=0.8, width=0.8)
        ax.set_title("Vehicle Speed Distribution")
        ax.set_xlabel("Speed bucket"); ax.set_ylabel("Count")

        plt.savefig(os.path.join(output_dir, "performance_dashboard.png"),
                    dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print("  📊 performance_dashboard.png збережено")

        # ── 2. ROC + Precision-Recall ─────────────────────────────

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
        fig.suptitle("Detection Quality — ROC & PR Curves", color="white",
                     fontsize=13, fontweight="bold")

        # ROC
        ax = axes[0]; _style_ax(ax)
        fpr_v = self.metrics["roc_data"]["fpr"]
        tpr_v = self.metrics["roc_data"]["tpr"]
        roc_a = self.metrics["roc_data"]["auc"]
        if fpr_v and tpr_v:
            ax.plot(fpr_v, tpr_v, color=BLUE, lw=2, label=f"ROC (AUC={roc_a:.3f})")
            ax.fill_between(fpr_v, tpr_v, alpha=0.15, color=BLUE)
        ax.plot([0, 1], [0, 1], color=GRID_CLR, lw=1, ls="--", label="Random")
        ax.legend(fontsize=9, framealpha=0, labelcolor=TEXT_CLR)
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")

        # PR
        ax = axes[1]; _style_ax(ax)
        pr_p = self.metrics["precision_recall_data"]["precision"]
        pr_r = self.metrics["precision_recall_data"]["recall"]
        pr_ap = self.metrics["precision_recall_data"]["average_precision"]
        if pr_p and pr_r:
            ax.plot(pr_r, pr_p, color=GREEN, lw=2, label=f"PR (AP={pr_ap:.3f})")
            ax.fill_between(pr_r, pr_p, alpha=0.15, color=GREEN)
        ax.legend(fontsize=9, framealpha=0, labelcolor=TEXT_CLR)
        ax.set_title("Precision-Recall Curve")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, "roc_pr_curves.png"),
                    dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print("  📊 roc_pr_curves.png збережено")

        # ── 3. LSTM Dashboard ─────────────────────────────────────

        lstm = self.metrics.get("lstm", {})
        if lstm.get("enabled"):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=DARK_BG)
            fig.suptitle("LSTM Accident Prediction", color="white",
                         fontsize=13, fontweight="bold")

            for ax, data, title, clr in [
                (axes[0], lstm.get("avg_accident_prob", []),  "Avg Accident Prob / Frame", ORANGE),
                (axes[1], lstm.get("max_accident_prob", []),  "Max Accident Prob / Frame", RED),
                (axes[2], lstm.get("high_risk_tracks_count", []), "High-Risk Tracks / Frame", PURPLE),
            ]:
                _style_ax(ax)
                if data:
                    ax.plot(data, color=clr, lw=1.2)
                    ax.fill_between(range(len(data)), data, alpha=0.2, color=clr)
                    if title != "High-Risk Tracks / Frame":
                        ax.axhline(0.7, color="white", lw=1, ls="--", alpha=0.6, label="Threshold 0.7")
                        ax.legend(fontsize=7, framealpha=0, labelcolor=TEXT_CLR)
                ax.set_title(title)
                ax.set_xlabel("Frame")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(output_dir, "lstm_predictions.png"),
                        dpi=150, bbox_inches="tight", facecolor=DARK_BG)
            plt.close(fig)
            print("  📊 lstm_predictions.png збережено")

        # ── 4. Latency Percentile Bar ─────────────────────────────

        lat = self.metrics.get("latency", {})
        if any(lat.get(k, 0) > 0 for k in ["p50_ms", "p90_ms", "p95_ms", "p99_ms"]):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=DARK_BG)
            fig.suptitle("Latency Analysis", color="white",
                         fontsize=13, fontweight="bold")

            # Percentile bars
            ax = axes[0]; _style_ax(ax)
            labels_l  = ["P50", "P90", "P95", "P99"]
            values_l  = [lat.get("p50_ms", 0), lat.get("p90_ms", 0),
                         lat.get("p95_ms", 0), lat.get("p99_ms", 0)]
            colors_l  = [GREEN, BLUE, ORANGE, RED]
            bars = ax.bar(labels_l, values_l, color=colors_l, alpha=0.85, width=0.6)
            for bar, val in zip(bars, values_l):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.1f}ms", ha="center", color=TEXT_CLR, fontsize=9)
            ax.set_title("Frame Time Percentiles")
            ax.set_ylabel("ms")

            # Rolling avg FPS
            ax = axes[1]; _style_ax(ax)
            times = self.metrics["performance"]["frame_times_ms"]
            if len(times) > 10:
                rolling = np.convolve(
                    [1000 / max(t, 1) for t in times],
                    np.ones(30) / 30, mode="valid"
                )
                ax.plot(rolling, color=BLUE, lw=1.2)
                ax.axhline(self.metrics["video_info"].get("fps", 25),
                           color=GREEN, lw=1, ls="--", label="Video FPS")
                ax.legend(fontsize=8, framealpha=0, labelcolor=TEXT_CLR)
            ax.set_title("Rolling Avg Processing FPS")
            ax.set_xlabel("Frame"); ax.set_ylabel("FPS")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(output_dir, "latency_analysis.png"),
                        dpi=150, bbox_inches="tight", facecolor=DARK_BG)
            plt.close(fig)
            print("  📊 latency_analysis.png збережено")

        # ── 5. Evaluation Summary (bar chart) ─────────────────────

        ev = self.metrics.get("evaluation", {})
        te = self.metrics.get("tracking_evaluation", {})

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
        fig.suptitle("Evaluation Metrics Summary", color="white",
                     fontsize=13, fontweight="bold")

        # Detection metrics
        ax = axes[0]; _style_ax(ax)
        det_names  = ["Precision", "Recall", "F1", "Accuracy",
                      "ROC-AUC", "PR-AUC", "mAP@0.5", "mAP@0.75"]
        det_vals   = [
            ev.get("precision", 0),  ev.get("recall", 0),
            ev.get("f1_score", 0),   ev.get("accuracy", 0),
            self.metrics["roc_data"].get("auc", 0),
            self.metrics["precision_recall_data"].get("average_precision", 0),
            ev.get("mAP_50", 0),     ev.get("mAP_75", 0),
        ]
        bar_colors = [GREEN if v >= 0.7 else (ORANGE if v >= 0.5 else RED)
                      for v in det_vals]
        bars = ax.barh(det_names, det_vals, color=bar_colors, alpha=0.85)
        for bar, val in zip(bars, det_vals):
            ax.text(max(val, 0.02), bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", color=TEXT_CLR, fontsize=8)
        ax.set_xlim(0, 1.15)
        ax.set_title("Detection & Classification")
        ax.set_xlabel("Score")

        # Tracking metrics (MOTA/IDF1)
        ax = axes[1]; _style_ax(ax)
        tr_names = ["MOTA", "IDF1"]
        tr_vals  = [te.get("MOTA", 0), te.get("IDF1", 0)]
        tr_clrs  = [GREEN if v >= 0.6 else (ORANGE if v >= 0.3 else RED)
                    for v in tr_vals]
        bars2 = ax.bar(tr_names, tr_vals, color=tr_clrs, alpha=0.85, width=0.4)
        for bar, val in zip(bars2, tr_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", color=TEXT_CLR, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_title("Tracking Quality")
        ax.set_ylabel("Score")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, "evaluation_summary.png"),
                    dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print("  📊 evaluation_summary.png збережено")

        print(f"\n  ✅ Всі графіки збережено у: {output_dir}")