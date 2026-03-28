import json
import numpy as np
import motmetrics as mm

# ==========================================
# 1. НАЛАШТУВАННЯ ШЛЯХІВ
# ==========================================
# Шлях до вашої ідеальної ручної розмітки (з боксами та ID)
GT_PATH = r"E:\personalproject\bachelor\logs\gt\gt_tracking_micro.json"

# Шлях до файлу передбачень системи для цього ж відео
PRED_PATH = r"E:\personalproject\bachelor\logs\ground_truh.json" 

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def main():
    print("⏳ Аналіз підсистеми трекінгу (DeepSORT + YOLOv8)...")
    
    with open(GT_PATH, 'r') as f:
        gt_data = json.load(f).get("frame_annotations", {})
    with open(PRED_PATH, 'r') as f:
        pred_data = json.load(f).get("frame_annotations", {})

    acc = mm.MOTAccumulator(auto_id=True)
    tp_det, fp_det, fn_det = 0, 0, 0

    all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())), key=int)

    for f_str in all_frames:
        gt_frame = gt_data.get(f_str, {})
        pred_frame = pred_data.get(f_str, {})

        gt_boxes, gt_ids = [], []
        pred_boxes, pred_ids = [], []

        # ==========================================
        # УНІВЕРСАЛЬНА ФУНКЦІЯ ПАРСИНГУ
        # ==========================================
        def parse_frame_data(frame_data):
            boxes = []
            ids = []
            
            vehicles_list = frame_data.get("vehicles", [])
            bboxes_list = frame_data.get("bboxes", [])

            if not vehicles_list:
                return boxes, ids

            # Формат 1: Власний анотатор (Список словників [{"id": 1, "bbox": [...]}, ...])
            if isinstance(vehicles_list[0], dict):
                for v in vehicles_list:
                    if "bbox" in v and "id" in v:
                        boxes.append(v["bbox"])
                        ids.append(int(v["id"]))
                        
            # Формат 2: Програма (Паралельні списки "vehicles": ["1"], "bboxes": [[...]])
            elif isinstance(vehicles_list[0], str) or isinstance(vehicles_list[0], int):
                # Перевіряємо, чи кількість ID збігається з кількістю боксів
                if len(vehicles_list) == len(bboxes_list):
                    for idx, v_id in enumerate(vehicles_list):
                        boxes.append(bboxes_list[idx])
                        ids.append(int(v_id))
            
            return boxes, ids

        # 1. Витягуємо дані еталону (GT)
        gt_boxes, gt_ids = parse_frame_data(gt_frame)

        # 2. Витягуємо дані передбачення (PRED)
        pred_boxes, pred_ids = parse_frame_data(pred_frame)

        # --- Розрахунок детекції (mAP@50 proxy) ---
        matched_gt = set()
        for p_box in pred_boxes:
            best_iou, best_gt_idx = 0, -1
            for i, g_box in enumerate(gt_boxes):
                if i in matched_gt: continue
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, i
            
            if best_iou >= 0.5:
                tp_det += 1
                matched_gt.add(best_gt_idx)
            else:
                fp_det += 1
        fn_det += len(gt_boxes) - len(matched_gt)

        # --- Матриця відстаней для трекінгу (MOTA/IDF1) ---
        distance_matrix = []
        for g_box in gt_boxes:
            row = []
            for p_box in pred_boxes:
                iou = compute_iou(g_box, p_box)
                row.append(1.0 - iou if iou >= 0.5 else np.nan)
            distance_matrix.append(row)
        
        if gt_ids or pred_ids:
            acc.update(gt_ids, pred_ids, distance_matrix)

    # Фінальні розрахунки
    det_prec = tp_det / (tp_det + fp_det) if (tp_det + fp_det) > 0 else 0.0
    det_rec  = tp_det / (tp_det + fn_det) if (tp_det + fn_det) > 0 else 0.0
    map_50 = (2 * det_prec * det_rec) / (det_prec + det_rec) if (det_prec + det_rec) > 0 else 0.0

    print("\n" + "="*50)
    print(" 📊 РЕЗУЛЬТАТИ ОЦІНКИ ТРЕКІНГУ (Мікро-датасет)")
    print("="*50)
    print(f"  Детекція (YOLOv8) mAP@50 proxy : {map_50:.4f}")
    print(f"  Precision боксів               : {det_prec:.4f}")
    print(f"  Recall боксів                  : {det_rec:.4f}")
    print("-" * 50)
    
    if len(acc.mot_events) > 0:
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=['mota', 'idf1', 'num_switches', 'num_false_positives', 'num_misses'])
        print(f"  Трекінг (DeepSORT) MOTA        : {summary.iloc[0]['mota']:.4f}")
        print(f"  Трекінг (DeepSORT) IDF1        : {summary.iloc[0]['idf1']:.4f}")
        print(f"  Втрати ID (ID Switches)        : {summary.iloc[0]['num_switches']}")
        print(f"  Пропуски об'єктів (Misses)     : {summary.iloc[0]['num_misses']}")
        print(f"  Хибні об'єкти (False Pos)      : {summary.iloc[0]['num_false_positives']}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()