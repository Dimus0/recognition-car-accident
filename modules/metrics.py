import cv2
import numpy as np
import torch
from torchvision import transforms
import logging
import os
import json
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


class MetricsTracker:
    """
    Розширений клас для збору та аналізу метрик роботи системи
    Включає: Precision, Recall, F1, mAP, MOTA, IDF1, ROC-AUC
    """
    
    def __init__(self, ground_truth_path: Optional[str] = None):
        """
        Args:
            ground_truth_path: Шлях до файлу з ground truth анотаціями
        """
        self.ground_truth_path = ground_truth_path
        self.ground_truth = self._load_ground_truth() if ground_truth_path else None
        self.reset()
    
    def _load_ground_truth(self) -> Dict:
        """
        Завантажує ground truth анотації
        
        Формат JSON:
        {
            "accidents": [
                {
                    "frame": 150,
                    "vehicles": [1, 2, 3],
                    "bbox": [[x1, y1, x2, y2], ...],
                    "type": "collision"
                }
            ],
            "frame_annotations": {
                "150": {
                    "accident": true,
                    "vehicles": [1, 2, 3]
                }
            }
        }
        """
        if not os.path.exists(self.ground_truth_path):
            print(f"⚠️ Ground truth файл не знайдено: {self.ground_truth_path}")
            return None
        
        with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def reset(self):
        self.metrics = {
            'video_info': {},
            'processing': {
                'total_frames': 0,
                'processing_time_seconds': 0,
                'avg_fps': 0,
                'frames_with_detections': 0
            },
            'detection': {
                'yolo': {
                    'total_detections': 0,
                    'avg_confidence': [],
                    'detections_per_frame': []
                },
                'cnn': {
                    'total_inferences': 0,
                    'avg_inference_time_ms': [],
                    'high_confidence_predictions': 0,
                    'medium_confidence_predictions': 0,
                    'low_confidence_predictions': 0
                }
            },
            'tracking': {
                'unique_vehicles': set(),
                'avg_track_length': [],
                'max_simultaneous_tracks': 0
            },
            'accidents': {
                'total_accidents_detected': 0,
                'false_positives_filtered': 0,
                'collision_warnings': 0,
                'sudden_stops': 0,
                'accidents_by_frame': [],
                'average_vehicles_per_accident': []
            },
            'performance': {
                'roi_processing_ratio': 0,
                'cnn_skip_ratio': 0,
                'memory_usage_mb': []
            },
            # ===== НОВІ МЕТРИКИ =====
            'evaluation': {
                'confusion_matrix': {
                    'true_positives': 0,
                    'false_positives': 0,
                    'true_negatives': 0,
                    'false_negatives': 0
                },
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'accuracy': 0.0,
                'mAP': 0.0,
                'mAP_50': 0.0,
                'mAP_75': 0.0
            },
            'tracking_evaluation': {
                'MOTA': 0.0,  # Multiple Object Tracking Accuracy
                'MOTP': 0.0,  # Multiple Object Tracking Precision
                'IDF1': 0.0,  # ID F1 Score
                'mostly_tracked': 0,
                'mostly_lost': 0,
                'id_switches': 0,
                'fragmentations': 0
            },
            'roc_data': {
                'fpr': [],  # False Positive Rate
                'tpr': [],  # True Positive Rate
                'thresholds': [],
                'auc': 0.0
            },
            'precision_recall_data': {
                'precision': [],
                'recall': [],
                'thresholds': [],
                'average_precision': 0.0
            }
        }
        
        self.start_time = None
        self.frame_times = deque(maxlen=100)
        
        # Для обчислення метрик
        self.predictions_per_frame = {}  # {frame_num: [predictions]}
        self.detection_scores = []  # Всі скори детекцій
        self.detection_labels = []  # Ground truth labels (0/1)
        
        # Для MOTA/IDF1
        self.tracking_data = {
            'predictions': defaultdict(list),  # {frame: [(id, bbox, conf)]}
            'ground_truth': defaultdict(list),  # {frame: [(id, bbox)]}
            'id_mappings': {},  # Відповідність predicted_id -> gt_id
            'id_switches': 0,
            'false_positives_tracking': 0,
            'false_negatives_tracking': 0,
            'matches': 0
        }
    
    def set_video_info(self, width, height, fps, total_frames, path):
        self.metrics['video_info'] = {
            'path': path,
            'resolution': f"{width}x{height}",
            'fps': fps,
            'total_frames': total_frames,
            'duration_seconds': total_frames / fps if fps > 0 else 0
        }
    
    def start_processing(self):
        self.start_time = cv2.getTickCount()
    
    def update_frame_metrics(self, frame_time_ms):
        self.frame_times.append(frame_time_ms)
        self.metrics['processing']['total_frames'] += 1
    
    def update_yolo_metrics(self, detections, confidences):
        self.metrics['detection']['yolo']['total_detections'] += len(detections)
        if confidences:
            self.metrics['detection']['yolo']['avg_confidence'].extend(confidences)
        self.metrics['detection']['yolo']['detections_per_frame'].append(len(detections))
        
        if len(detections) > 0:
            self.metrics['processing']['frames_with_detections'] += 1
    
    def update_cnn_metrics(self, num_inferences, inference_time_ms, scores):
        self.metrics['detection']['cnn']['total_inferences'] += num_inferences
        self.metrics['detection']['cnn']['avg_inference_time_ms'].append(inference_time_ms)
        
        for score in scores:
            if score > 0.9:
                self.metrics['detection']['cnn']['high_confidence_predictions'] += 1
            elif score > 0.7:
                self.metrics['detection']['cnn']['medium_confidence_predictions'] += 1
            else:
                self.metrics['detection']['cnn']['low_confidence_predictions'] += 1
    
    def update_tracking_metrics(self, active_ids, track_history):
        self.metrics['tracking']['unique_vehicles'].update(active_ids)
        current_tracks = len(active_ids)
        
        if current_tracks > self.metrics['tracking']['max_simultaneous_tracks']:
            self.metrics['tracking']['max_simultaneous_tracks'] = current_tracks
        
        for tid in active_ids:
            if tid in track_history:
                self.metrics['tracking']['avg_track_length'].append(len(track_history[tid]))
    
    def update_accident_metrics(self, accidents_this_frame, num_vehicles, collision_warnings, sudden_stops):
        if accidents_this_frame:
            self.metrics['accidents']['total_accidents_detected'] += 1
            self.metrics['accidents']['accidents_by_frame'].append(
                self.metrics['processing']['total_frames']
            )
            self.metrics['accidents']['average_vehicles_per_accident'].append(num_vehicles)
        
        self.metrics['accidents']['collision_warnings'] += collision_warnings
        self.metrics['accidents']['sudden_stops'] += sudden_stops
    
    # ========== НОВІ МЕТОДИ ДЛЯ РОЗШИРЕНИХ МЕТРИК ==========
    
    def update_detection_for_evaluation(self, frame_num: int, predicted_accident: bool, 
                                       confidence: float):
        """
        Оновлює дані для обчислення Precision, Recall, F1, ROC
        
        Args:
            frame_num: Номер кадру
            predicted_accident: Чи була передбачена аварія
            confidence: Впевненість передбачення (0-1)
        """
        if self.ground_truth is None:
            return
        
        # Перевіряємо ground truth
        gt_accident = self._is_accident_in_ground_truth(frame_num)
        
        # Зберігаємо для ROC
        self.detection_scores.append(confidence)
        self.detection_labels.append(1 if gt_accident else 0)
        
        # Оновлюємо confusion matrix
        if predicted_accident and gt_accident:
            self.metrics['evaluation']['confusion_matrix']['true_positives'] += 1
        elif predicted_accident and not gt_accident:
            self.metrics['evaluation']['confusion_matrix']['false_positives'] += 1
        elif not predicted_accident and gt_accident:
            self.metrics['evaluation']['confusion_matrix']['false_negatives'] += 1
        else:
            self.metrics['evaluation']['confusion_matrix']['true_negatives'] += 1
    
    def update_tracking_for_evaluation(self, frame_num: int, predicted_tracks: List[Tuple],
                                      iou_threshold: float = 0.5):
        """
        Оновлює дані для обчислення MOTA, IDF1
        
        Args:
            frame_num: Номер кадру
            predicted_tracks: [(track_id, bbox, confidence), ...]
            iou_threshold: Поріг IoU для matching
        """
        if self.ground_truth is None:
            return
        
        # Зберігаємо предикції
        self.tracking_data['predictions'][frame_num] = predicted_tracks
        
        # Отримуємо ground truth для цього кадру
        gt_tracks = self._get_ground_truth_tracks(frame_num)
        self.tracking_data['ground_truth'][frame_num] = gt_tracks
        
        # Matching predicted і ground truth треків
        matches, fp, fn, id_switches = self._match_tracks(
            predicted_tracks, gt_tracks, iou_threshold
        )
        
        self.tracking_data['matches'] += matches
        self.tracking_data['false_positives_tracking'] += fp
        self.tracking_data['false_negatives_tracking'] += fn
        self.tracking_data['id_switches'] += id_switches
    
    def _is_accident_in_ground_truth(self, frame_num: int) -> bool:
        """Перевіряє чи є аварія в ground truth для цього кадру"""
        if not self.ground_truth or 'frame_annotations' not in self.ground_truth:
            return False
        
        frame_key = str(frame_num)
        if frame_key in self.ground_truth['frame_annotations']:
            return self.ground_truth['frame_annotations'][frame_key].get('accident', False)
        
        return False
    
    def _get_ground_truth_tracks(self, frame_num: int) -> List[Tuple]:
        """Повертає ground truth треки для кадру"""
        if not self.ground_truth or 'frame_annotations' not in self.ground_truth:
            return []
        
        frame_key = str(frame_num)
        if frame_key in self.ground_truth['frame_annotations']:
            annotation = self.ground_truth['frame_annotations'][frame_key]
            if 'vehicles' in annotation and 'bboxes' in annotation:
                return list(zip(annotation['vehicles'], annotation['bboxes']))
        
        return []
    
    def _match_tracks(self, predicted: List[Tuple], ground_truth: List[Tuple],
                     iou_threshold: float) -> Tuple[int, int, int, int]:
        """
        Matching predicted і ground truth треків
        
        Returns:
            (matches, false_positives, false_negatives, id_switches)
        """
        if not predicted and not ground_truth:
            return 0, 0, 0, 0
        
        if not predicted:
            return 0, 0, len(ground_truth), 0
        
        if not ground_truth:
            return 0, len(predicted), 0, 0
        
        # Обчислюємо IoU матрицю
        iou_matrix = np.zeros((len(predicted), len(ground_truth)))
        
        for i, (pred_id, pred_bbox, _) in enumerate(predicted):
            for j, (gt_id, gt_bbox) in enumerate(ground_truth):
                iou_matrix[i, j] = self._compute_iou(pred_bbox, gt_bbox)
        
        # Жадібний matching
        matches = 0
        id_switches = 0
        matched_pred = set()
        matched_gt = set()
        
        # Сортуємо за IoU (найбільші спочатку)
        flat_indices = np.argsort(-iou_matrix.flatten())
        
        for idx in flat_indices:
            i = idx // len(ground_truth)
            j = idx % len(ground_truth)
            
            if i in matched_pred or j in matched_gt:
                continue
            
            if iou_matrix[i, j] >= iou_threshold:
                matches += 1
                matched_pred.add(i)
                matched_gt.add(j)
                
                # Перевірка ID switch
                pred_id = predicted[i][0]
                gt_id = ground_truth[j][0]
                
                if pred_id in self.tracking_data['id_mappings']:
                    if self.tracking_data['id_mappings'][pred_id] != gt_id:
                        id_switches += 1
                
                self.tracking_data['id_mappings'][pred_id] = gt_id
        
        false_positives = len(predicted) - matches
        false_negatives = len(ground_truth) - matches
        
        return matches, false_positives, false_negatives, id_switches
    
    def _compute_iou(self, box1: List, box2: List) -> float:
        """Обчислює IoU між двома bbox"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Перетин
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
        
        # Об'єднання
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def compute_precision_recall_f1(self):
        """Обчислює Precision, Recall, F1-score"""
        cm = self.metrics['evaluation']['confusion_matrix']
        tp = cm['true_positives']
        fp = cm['false_positives']
        fn = cm['false_negatives']
        tn = cm['true_negatives']
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1-Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        self.metrics['evaluation']['precision'] = round(precision, 4)
        self.metrics['evaluation']['recall'] = round(recall, 4)
        self.metrics['evaluation']['f1_score'] = round(f1, 4)
        self.metrics['evaluation']['accuracy'] = round(accuracy, 4)
    
    def compute_map(self, iou_thresholds: List[float] = [0.5, 0.75]):
        """
        Обчислює mAP (mean Average Precision)
        
        Args:
            iou_thresholds: Пороги IoU для обчислення (наприклад, [0.5, 0.75])
        """
        if not self.ground_truth:
            return
        
        # Збираємо всі детекції по кадрах
        all_predictions = []
        all_ground_truths = []
        
        for frame_num in self.tracking_data['predictions']:
            predictions = self.tracking_data['predictions'][frame_num]
            gt = self.tracking_data['ground_truth'].get(frame_num, [])
            
            all_predictions.extend([(frame_num, *p) for p in predictions])
            all_ground_truths.extend([(frame_num, *g) for g in gt])
        
        # Обчислюємо AP для кожного IoU threshold
        aps = []
        for iou_thresh in iou_thresholds:
            ap = self._compute_average_precision(all_predictions, all_ground_truths, iou_thresh)
            aps.append(ap)
            
            if iou_thresh == 0.5:
                self.metrics['evaluation']['mAP_50'] = round(ap, 4)
            elif iou_thresh == 0.75:
                self.metrics['evaluation']['mAP_75'] = round(ap, 4)
        
        # mAP - середнє по всіх IoU thresholds
        self.metrics['evaluation']['mAP'] = round(np.mean(aps), 4)
    
    def _compute_average_precision(self, predictions, ground_truths, iou_threshold):
        """Обчислює Average Precision для заданого IoU threshold"""
        if not predictions or not ground_truths:
            return 0.0
        
        # Сортуємо predictions по confidence (від найбільшої)
        predictions = sorted(predictions, key=lambda x: x[3], reverse=True)
        
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        
        gt_matched = set()
        
        for i, pred in enumerate(predictions):
            frame, pred_id, pred_bbox, conf = pred
            
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(ground_truths):
                gt_frame, gt_id, gt_bbox = gt
                
                if frame != gt_frame:
                    continue
                
                iou = self._compute_iou(pred_bbox, gt_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold and best_gt_idx not in gt_matched:
                tp[i] = 1
                gt_matched.add(best_gt_idx)
            else:
                fp[i] = 1
        
        # Обчислюємо precision і recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(ground_truths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Обчислюємо AP (площа під PR кривою)
        ap = 0
        for i in range(1, len(recalls)):
            ap += (recalls[i] - recalls[i-1]) * precisions[i]
        
        return ap
    
    def compute_mota_idf1(self):
        """
        Обчислює MOTA (Multiple Object Tracking Accuracy) та IDF1
        """
        if not self.ground_truth:
            return
        
        # MOTA = 1 - (FN + FP + ID_switches) / GT
        total_gt = sum(len(self.tracking_data['ground_truth'][f]) 
                      for f in self.tracking_data['ground_truth'])
        
        if total_gt == 0:
            return
        
        fn = self.tracking_data['false_negatives_tracking']
        fp = self.tracking_data['false_positives_tracking']
        id_sw = self.tracking_data['id_switches']
        
        mota = 1 - (fn + fp + id_sw) / total_gt
        self.metrics['tracking_evaluation']['MOTA'] = round(mota, 4)
        
        # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
        matches = self.tracking_data['matches']
        
        idf1 = 2 * matches / (2 * matches + fp + fn) if (2 * matches + fp + fn) > 0 else 0
        self.metrics['tracking_evaluation']['IDF1'] = round(idf1, 4)
        
        # Додаткові метрики
        self.metrics['tracking_evaluation']['id_switches'] = id_sw
        self.metrics['tracking_evaluation']['false_positives'] = fp
        self.metrics['tracking_evaluation']['false_negatives'] = fn
    
    def compute_roc_auc(self):
        """Обчислює ROC криву та AUC"""
        if len(self.detection_scores) == 0 or len(self.detection_labels) == 0:
            return
        
        # Обчислюємо ROC
        fpr, tpr, thresholds = roc_curve(self.detection_labels, self.detection_scores)
        roc_auc = auc(fpr, tpr)
        
        self.metrics['roc_data']['fpr'] = fpr.tolist()
        self.metrics['roc_data']['tpr'] = tpr.tolist()
        self.metrics['roc_data']['thresholds'] = thresholds.tolist()
        self.metrics['roc_data']['auc'] = round(roc_auc, 4)
    
    def compute_precision_recall_curve(self):
        """Обчислює Precision-Recall криву"""
        if len(self.detection_scores) == 0 or len(self.detection_labels) == 0:
            return
        
        precision, recall, thresholds = precision_recall_curve(
            self.detection_labels, self.detection_scores
        )
        
        # Average Precision
        from sklearn.metrics import average_precision_score
        avg_precision = average_precision_score(self.detection_labels, self.detection_scores)
        
        self.metrics['precision_recall_data']['precision'] = precision.tolist()
        self.metrics['precision_recall_data']['recall'] = recall.tolist()
        self.metrics['precision_recall_data']['thresholds'] = thresholds.tolist()
        self.metrics['precision_recall_data']['average_precision'] = round(avg_precision, 4)
    
    def plot_roc_curve(self, save_path: str):
        """Малює та зберігає ROC криву"""
        if not self.metrics['roc_data']['fpr']:
            print("⚠️ Немає даних для ROC кривої")
            return
        
        plt.figure(figsize=(10, 8))
        plt.plot(
            self.metrics['roc_data']['fpr'], 
            self.metrics['roc_data']['tpr'],
            color='darkorange',
            lw=2,
            label=f'ROC curve (AUC = {self.metrics["roc_data"]["auc"]:.4f})'
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ ROC крива збережена: {save_path}")
    
    def plot_precision_recall_curve(self, save_path: str):
        """Малює та зберігає Precision-Recall криву"""
        if not self.metrics['precision_recall_data']['precision']:
            print("⚠️ Немає даних для PR кривої")
            return
        
        plt.figure(figsize=(10, 8))
        plt.plot(
            self.metrics['precision_recall_data']['recall'],
            self.metrics['precision_recall_data']['precision'],
            color='blue',
            lw=2,
            label=f'PR curve (AP = {self.metrics["precision_recall_data"]["average_precision"]:.4f})'
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ PR крива збережена: {save_path}")
    
    def plot_confusion_matrix(self, save_path: str):
        """Малює та зберігає confusion matrix"""
        cm = self.metrics['evaluation']['confusion_matrix']
        
        matrix = np.array([
            [cm['true_negatives'], cm['false_positives']],
            [cm['false_negatives'], cm['true_positives']]
        ])
        
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix', fontsize=14)
        plt.colorbar()
        
        classes = ['No Accident', 'Accident']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, fontsize=12)
        plt.yticks(tick_marks, classes, fontsize=12)
        
        # Додаємо текст
        thresh = matrix.max() / 2.
        for i in range(2):
            for j in range(2):
                plt.text(j, i, format(matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if matrix[i, j] > thresh else "black",
                        fontsize=16)
        
        plt.ylabel('True label', fontsize=12)
        plt.xlabel('Predicted label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion matrix збережена: {save_path}")
    
    def finalize(self):
        """Фінальні обчислення після обробки всього відео"""
        end_time = cv2.getTickCount()

        if self.start_time:
            elapsed = (end_time - self.start_time) / cv2.getTickFrequency()
            self.metrics['processing']['processing_time_seconds'] = round(elapsed, 2)

            total_frames = self.metrics['processing']['total_frames']
            if total_frames > 0:
                self.metrics['processing']['avg_fps'] = round(total_frames / elapsed, 2)

        # Середні значення
        if self.metrics['detection']['yolo']['avg_confidence']:
            self.metrics['detection']['yolo']['avg_confidence'] = round(
                float(np.mean(self.metrics['detection']['yolo']['avg_confidence'])), 3
            )

        if self.metrics['detection']['cnn']['avg_inference_time_ms']:
            self.metrics['detection']['cnn']['avg_inference_time_ms'] = round(
                float(np.mean(self.metrics['detection']['cnn']['avg_inference_time_ms'])), 2
            )

        if self.metrics['tracking']['avg_track_length']:
            self.metrics['tracking']['avg_track_length'] = round(
                float(np.mean(self.metrics['tracking']['avg_track_length'])), 1
            )

        if self.metrics['accidents']['average_vehicles_per_accident']:
            self.metrics['accidents']['average_vehicles_per_accident'] = round(
                float(np.mean(self.metrics['accidents']['average_vehicles_per_accident'])), 1
            )

        # set → int (JSON-safe)
        self.metrics['tracking']['unique_vehicles'] = len(
            self.metrics['tracking']['unique_vehicles']
        )

        # Detection rate
        total_frames = self.metrics['processing']['total_frames']
        if total_frames > 0:
            self.metrics['performance']['detection_rate'] = round(
                self.metrics['processing']['frames_with_detections'] / total_frames, 4
            )

        # ===== РОЗШИРЕНІ МЕТРИКИ =====
        if self.ground_truth:
            self.compute_precision_recall_f1()
            self.compute_map()
            self.compute_mota_idf1()
            self.compute_roc_auc()
            self.compute_precision_recall_curve()

        return self.metrics


    def save_to_file(self, filepath):
        """Зберігає метрики у JSON файл"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)

    def save_all_plots(self, output_dir: str):
        """Зберігає всі графіки"""
        os.makedirs(output_dir, exist_ok=True)
        
        roc_path = os.path.join(output_dir, 'roc_curve.png')
        pr_path = os.path.join(output_dir, 'precision_recall_curve.png')
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        
        self.plot_roc_curve(roc_path)
        self.plot_precision_recall_curve(pr_path)
        self.plot_confusion_matrix(cm_path)

    def print_summary(self):
        """Виводить зведення метрик у консоль"""
        m = self.metrics
        
        print("\n" + "="*70)
        print(" "*20 + "МЕТРИКИ СИСТЕМИ")
        print("="*70)
        
        print("\n📹 ВІДЕО:")
        print(f"  Шлях: {m['video_info'].get('path', 'N/A')}")
        print(f"  Роздільна здатність: {m['video_info'].get('resolution', 'N/A')}")
        print(f"  FPS відео: {m['video_info'].get('fps', 'N/A')}")
        print(f"  Тривалість: {m['video_info'].get('duration_seconds', 0):.1f} сек")
        
        print("\n⚡ ПРОДУКТИВНІСТЬ:")
        print(f"  Оброблено кадрів: {m['processing']['total_frames']}")
        print(f"  Час обробки: {m['processing']['processing_time_seconds']} сек")
        print(f"  Середній FPS обробки: {m['processing']['avg_fps']}")
        print(f"  Кадрів з детекціями: {m['processing']['frames_with_detections']}")
        
        print("\n🎯 YOLO ДЕТЕКЦІЯ:")
        print(f"  Всього детекцій: {m['detection']['yolo']['total_detections']}")
        print(f"  Середня впевненість: {m['detection']['yolo']['avg_confidence']}")
        if m['detection']['yolo']['detections_per_frame']:
            avg_det = np.mean(m['detection']['yolo']['detections_per_frame'])
            print(f"  Середньо на кадр: {avg_det:.1f}")
        
        print("\n🧠 CNN КЛАСИФІКАЦІЯ:")
        print(f"  Всього інференсів: {m['detection']['cnn']['total_inferences']}")
        print(f"  Середній час: {m['detection']['cnn']['avg_inference_time_ms']} мс")
        print(f"  Висока впевненість (>0.9): {m['detection']['cnn']['high_confidence_predictions']}")
        print(f"  Середня впевненість (0.7-0.9): {m['detection']['cnn']['medium_confidence_predictions']}")
        print(f"  Низька впевненість (<0.7): {m['detection']['cnn']['low_confidence_predictions']}")
        
        print("\n🚗 ТРЕКІНГ:")
        print(f"  Унікальних авто: {m['tracking']['unique_vehicles']}")
        print(f"  Максимум одночасно: {m['tracking']['max_simultaneous_tracks']}")
        print(f"  Середня довжина треку: {m['tracking']['avg_track_length']}")
        
        print("\n🚨 АВАРІЇ:")
        print(f"  Виявлено аварій: {m['accidents']['total_accidents_detected']}")
        print(f"  Попереджень про зіткнення: {m['accidents']['collision_warnings']}")
        print(f"  Раптових зупинок: {m['accidents']['sudden_stops']}")
        print(f"  Відфільтровано хибних: {m['accidents']['false_positives_filtered']}")
        if m['accidents']['average_vehicles_per_accident']:
            print(f"  Середньо авто в аварії: {m['accidents']['average_vehicles_per_accident']}")
        
        # ===== РОЗШИРЕНІ МЕТРИКИ =====
        if m['evaluation']['precision'] > 0 or m['evaluation']['recall'] > 0:
            print("\n📊 ЯКІСТЬ ДЕТЕКЦІЇ:")
            print(f"  Precision: {m['evaluation']['precision']:.4f}")
            print(f"  Recall: {m['evaluation']['recall']:.4f}")
            print(f"  F1-Score: {m['evaluation']['f1_score']:.4f}")
            print(f"  Accuracy: {m['evaluation']['accuracy']:.4f}")
            print(f"  mAP: {m['evaluation']['mAP']:.4f}")
            print(f"  mAP@0.5: {m['evaluation']['mAP_50']:.4f}")
            print(f"  mAP@0.75: {m['evaluation']['mAP_75']:.4f}")
            
            print("\n  Confusion Matrix:")
            cm = m['evaluation']['confusion_matrix']
            print(f"    True Positives:  {cm['true_positives']}")
            print(f"    False Positives: {cm['false_positives']}")
            print(f"    True Negatives:  {cm['true_negatives']}")
            print(f"    False Negatives: {cm['false_negatives']}")
        
        if m['tracking_evaluation']['MOTA'] != 0:
            print("\n🎯 ЯКІСТЬ ТРЕКІНГУ:")
            print(f"  MOTA: {m['tracking_evaluation']['MOTA']:.4f}")
            print(f"  IDF1: {m['tracking_evaluation']['IDF1']:.4f}")
            print(f"  ID Switches: {m['tracking_evaluation']['id_switches']}")
            print(f"  False Positives: {m['tracking_evaluation'].get('false_positives', 0)}")
            print(f"  False Negatives: {m['tracking_evaluation'].get('false_negatives', 0)}")
        
        if m['roc_data']['auc'] > 0:
            print("\n📈 ROC-AUC:")
            print(f"  AUC Score: {m['roc_data']['auc']:.4f}")
            print(f"  Average Precision: {m['precision_recall_data']['average_precision']:.4f}")
        
        print("\n" + "="*70 + "\n")
              