import json
import motmetrics as mm
import numpy as np
import os

# ---------------------------------------------------------
# 1. ЗАВАНТАЖЕННЯ ДАНИХ
# ---------------------------------------------------------
def load_ground_truth_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    frames_dict = {}
    if "annotations" in raw_data:
        for frame_data in raw_data["annotations"]:
            frame_id = str(frame_data["frame_id"])
            frames_dict[frame_id] = frame_data.get("objects", [])
    return frames_dict

def load_predictions_jsonl(filepath):
    frames_dict = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            frame_id = str(data["frame"])
            
            objects = data.get("tracks", data.get("dets", []))
            frames_dict[frame_id] = objects
    return frames_dict

# ---------------------------------------------------------
# 2. УНІФІКАЦІЯ ФОРМАТІВ У [x_min, y_min, width, height]
# ---------------------------------------------------------
def format_gt_bboxes(objects):
    bboxes, ids = [], []
    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        bboxes.append([x1, y1, w, h])
        ids.append(str(obj['track_id'])) 
    return np.array(bboxes), np.array(ids)

def format_pred_bboxes(objects):
    bboxes, ids, confs = [], [], []
    for i, obj in enumerate(objects):
        # ВИПРАВЛЕНО: Тепер скрипт знає, що це [x1, y1, x2, y2]
        x1, y1, x2, y2 = obj['bbox']
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        bboxes.append([x1, y1, w, h])
        
        track_id = obj.get('track_id', obj.get('id', f"temp_{i}"))
        ids.append(str(track_id))
        confs.append(obj.get('conf', 1.0))
        
    return np.array(bboxes), np.array(ids), confs

# ---------------------------------------------------------
# 3. ОБЧИСЛЕННЯ mAP
# ---------------------------------------------------------
def calculate_iou(box1, box2):
    # На вхід приходять [x, y, w, h] з наших функцій format_...
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    b1_x1, b1_y1, b1_x2, b1_y2 = x1, y1, x1 + w1, y1 + h1
    b2_x1, b2_y1, b2_x2, b2_y2 = x2, y2, x2 + w2, y2 + h2
    
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0

def calculate_map(gt_dict, pred_dict, iou_threshold=0.5):
    all_detections = []
    total_gt_boxes = 0
    frames = set(gt_dict.keys()).union(set(pred_dict.keys()))
    
    formatted_gt = {}
    for frame_id in frames:
        gt_objs = gt_dict.get(frame_id, [])
        gt_bboxes, _ = format_gt_bboxes(gt_objs)
        formatted_gt[frame_id] = gt_bboxes.tolist() if len(gt_bboxes) > 0 else []
        total_gt_boxes += len(formatted_gt[frame_id])
        
        pred_objs = pred_dict.get(frame_id, [])
        pred_bboxes, _, pred_confs = format_pred_bboxes(pred_objs)
        
        for i, bbox in enumerate(pred_bboxes):
            all_detections.append({
                'frame': frame_id,
                'bbox': bbox,
                'conf': pred_confs[i]
            })

    if total_gt_boxes == 0 or len(all_detections) == 0:
        return 0.0

    all_detections.sort(key=lambda x: x['conf'], reverse=True)
    tp = np.zeros(len(all_detections))
    fp = np.zeros(len(all_detections))
    matched_gt = {frame: [] for frame in frames}
    
    for idx, det in enumerate(all_detections):
        frame_id = det['frame']
        pred_bbox = det['bbox']
        gt_boxes = formatted_gt[frame_id]
        
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_bbox in enumerate(gt_boxes):
            if gt_idx in matched_gt[frame_id]: continue
            iou = calculate_iou(pred_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
                
        if best_iou >= iou_threshold:
            tp[idx] = 1
            matched_gt[frame_id].append(best_gt_idx)
        else:
            fp[idx] = 1
            
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / total_gt_boxes
    precisions = cum_tp / (cum_tp + cum_fp)
    
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
        
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    return np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

# ---------------------------------------------------------
# 4. ГОЛОВНИЙ ПАЙПЛАЙН ОЦІНКИ
# ---------------------------------------------------------
def evaluate_pipeline(gt_path, pred_path, iou_threshold=0.5):
    print(f"Завантаження GT: {gt_path}")
    gt_data = load_ground_truth_json(gt_path)
    
    print(f"Завантаження Predictions: {pred_path}")
    pred_data = load_predictions_jsonl(pred_path)
    
    acc = mm.MOTAccumulator()
    all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())), key=int)
    
    for frame_id in all_frames:
        gt_objects = gt_data.get(str(frame_id), [])
        pred_objects = pred_data.get(str(frame_id), [])
        
        gt_bboxes, gt_ids = format_gt_bboxes(gt_objects)
        pred_bboxes, pred_ids, _ = format_pred_bboxes(pred_objects)
        
        if len(gt_bboxes) > 0 and len(pred_bboxes) > 0:
            dist_matrix = mm.distances.iou_matrix(gt_bboxes, pred_bboxes, max_iou=1.0 - iou_threshold)
        else:
            dist_matrix = np.empty((len(gt_ids), len(pred_ids)))
            
        acc.update(gt_ids, pred_ids, dist_matrix, frameid=int(frame_id))

    mh = mm.metrics.create()
    metrics_list = ['num_frames', 'mota', 'idf1', 'idp', 'idr', 'num_switches', 'num_false_positives', 'num_misses']
    summary = mh.compute(acc, metrics=metrics_list, name='Evaluation')
    
    strsummary = mm.io.render_summary(
        summary, 
        formatters={'mota': '{:.2%}'.format, 'idf1': '{:.2%}'.format, 'idp': '{:.2%}'.format, 'idr': '{:.2%}'.format},
        namemap={'num_frames': 'Кадри', 'mota': 'MOTA', 'idf1': 'IDF1', 'idp': 'Precision', 'idr': 'Recall', 
                 'num_switches': 'ID Sw.', 'num_false_positives': 'FP', 'num_misses': 'FN'}
    )
    
    map_score = calculate_map(gt_data, pred_data, iou_threshold)
    
    print("\n" + "="*65)
    print(" РЕЗУЛЬТАТИ ОЦІНКИ СИСТЕМИ (Трекінг + Детекція)")
    print("="*65)
    print(strsummary)
    print("-" * 65)
    print(f" mAP@0.5 (Середня точність детекції):   {map_score:.2%}")
    print("="*65)

if __name__ == "__main__":
    GT_FILE = r"E:\personalproject\bachelor\logs\gt_box\gt_noaccident_video_v1.json"
    PRED_FILE = r"E:\personalproject\bachelor\logs\evaluation\noaccident_video_v1\deepsort_tracks.jsonl"
    
    if os.path.exists(GT_FILE) and os.path.exists(PRED_FILE):
        evaluate_pipeline(GT_FILE, PRED_FILE)
    else:
        print("Помилка: Файли не знайдено. Перевір шляхи!")