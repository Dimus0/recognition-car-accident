import cv2
import numpy as np
import torch
from torchvision import transforms
import logging
import os
from collections import defaultdict, deque
from modules.utils import *
from modules.analyzer import TrafficAnalyzer
from modules.accidenttracker import AccidentStateTracker
from modules.accidentcapture import AccidentFrameCapture
from modules.metrics import MetricsTracker

import time
from datetime import datetime
import json

# ---------------------- CONFIG ----------------------

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_DIR = r'D:\project\bachelor\logs'
OUTPUT_DIR = os.path.join(LOG_DIR, "output_result")
METRICS_OUTPUT_DIR = os.path.join(LOG_DIR, "metrics_output")
YOLO_MODEL_PATH = r"D:\project\bachelor\model\weights\yolov8-fine-tuning-15.pt"
CNN_WEIGHTS_PATH = r"D:\project\bachelor\model\weights\accident_cnn_model.pth"
METRICS_FILE = os.path.join(LOG_DIR, "metrics_summary.json")
DETAILED_METRICS_FILE = os.path.join(LOG_DIR, "detailed_metrics.json")

# VIDEO_PATH = r"D:\project\highway.mp4"
VIDEO_PATH = r"D:\project\videoplayback.mp4"

GROUND_TRUTH = os.path.join(LOG_DIR, "ground_truth.json" )

CONF_YOLO = 0.6
CONF_ACCIDENT_HIGH = 0.9632  # Поріг точного ДТП
CONF_ACCIDENT_LOW = 0.843   # Поріг попередження
HEARTBEAT_RATE = 30        # Як часто перевіряти авто без підозр

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(METRICS_OUTPUT_DIR, exist_ok=True)

# ---------- ground_truth.json (FILE, not directory) ----------
if os.path.isdir(GROUND_TRUTH):
    raise RuntimeError(
        f"Помилка: {GROUND_TRUTH} існує як папка. Видали її вручну."
    )

if not os.path.exists(GROUND_TRUTH):
    with open(GROUND_TRUTH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "video_path": VIDEO_PATH,
                    "created_at": datetime.now().isoformat()
                },
                "frames": {},
                "tracking": {},
                "events": {}
            },
            f,
            indent=4,
            ensure_ascii=False
        )


log_file = os.path.join(LOG_DIR, "accident_detection.log")
if os.path.exists(log_file):
    os.remove(log_file)

logging.basicConfig(
    filename=log_file,
    encoding='utf-8',
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("accident_detector")



# ---------------------- INIT ----------------------

cnn_model, yolo_model,deepsort = load_models(DEVICE,CNN_WEIGHTS_PATH,YOLO_MODEL_PATH)
logger.info("Models loaded successfully.")

cnn_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ---------------------- MAIN PIPELINE ----------------------
def accident_detection(input_video):

    metrics = MetricsTracker(ground_truth_path=GROUND_TRUTH)

    cap = cv2.VideoCapture(input_video)

    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    metrics.set_video_info(width, height, fps, total_frames, input_video)
    metrics.start_processing()

    output_path = os.path.join(OUTPUT_DIR, "result.mp4")
    out = cv2.VideoWriter(output_path, 
                          cv2.VideoWriter_fourcc(*"mp4v"), 
                          fps, 
                          (width, height))

    # -------- Лінії після яких відбувається трекінг --------)

    '''
        Область для відео із шосе
    '''
    # ROI_POLYGON = np.array([
    #     (842, 580),
    #     (240, 829),
    #     (958, 930),
    #     (1023, 583)
    # ], dtype=np.int32)

    '''
        Область для відео із ДТП
    '''
    ROI_POLYGON = np.array([
        (9, 216),
        (19, 996),
        (1450, 791),
        (511, 230)
    ], dtype=np.int32)

    # -------- END --------

    analyzer = TrafficAnalyzer(
        collision_ttc_threshold=1.5,
        sudden_stop_threshold=0.3,
        min_speed_for_stop=5.0
    )
    accident_capture = AccidentFrameCapture(OUTPUT_DIR)
    accident_state = AccidentStateTracker()

    frame_count = 0
    cnn_results_cache = {}

    logger.info(f"Starting processing: {total_frames} frames")

    while cap.isOpened():

        frame_start = cv2.getTickCount()

        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start_time = time.time()
        frame_count += 1

        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")

        cv2.polylines(frame, [ROI_POLYGON], isClosed=True, color=(255, 0, 0), thickness=2)

        results = yolo_model(frame, conf=CONF_YOLO, verbose=False)

        detections =  []
        yolo_confidences = []

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if conf < 0.3:
                    continue
                
                center_x = int((x1 + x2) / 2)
                bottom_y = int(y2)


                if is_point_in_polygon((center_x, bottom_y), ROI_POLYGON):
                    detections.append((
                        [x1, y1, x2 - x1, y2 - y1], # XYWH format for DeepSort
                        conf,
                        cls
                    ))
                    yolo_confidences.append(conf)

        metrics.update_yolo_metrics(detections,yolo_confidences)
        
        tracks = deepsort.update_tracks(detections, frame=frame)

        boxes = []
        ids = []
        active_tracks_objects = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            l, t, r, b = map(int, track.to_ltrb())
            tid = track.track_id

            if not is_point_in_polygon((int((l+r)/2), int(b)), ROI_POLYGON): continue

            boxes.append((l, t, r, b))
            ids.append(tid)
            active_tracks_objects.append(track)
        
        
        analyzer.update_tracks(boxes, ids)
        analyzer.clean_old_tracks(ids)

        metrics.update_tracking_metrics(ids,analyzer.track_history)

        crops_to_process = []
        ids_to_process = []
        box_map_for_cnn = []

        collision_risky_ids = analyzer.predict_collision(boxes=boxes,ids=ids)

        accident_state.cleanup_old_accidents(frame_count)

        crops_to_process = []
        ids_to_process = []
        box_map_for_cnn = []
        cnn_skipped = 0

        for i, tid in enumerate(ids):
            x1, y1, x2, y2 = boxes[i]
            metrics.tracking_data['predictions'][frame_count].append(
                (tid, (x1,y1,x2,y2),cnn_results_cache.get(tid,(0,0,0))[2])
            )

            for frame_num_str, annotation in metrics.ground_truth['frame_annotations'].items():
                frame_num = int(frame_num_str)
                vehicles = annotation.get('vehicles', [])
                bboxes = annotation.get('bboxes', [])
                
                # Переконуємось, що довжина списків збігається
                frame_gt = []
                for vid, bbox in zip(vehicles, bboxes):
                    # bbox у форматі XYXY
                    frame_gt.append((vid, bbox))
                
                metrics.tracking_data['ground_truth'][frame_num] = frame_gt

            gt_ids = ids.copy()
            gt_bboxes = boxes.copy()
            metrics.add_ground_truth_frame(
                frame_num=frame_count,
                vehicles=gt_ids,
                bboxes=gt_bboxes,
                accident=accident_detected_this_frame
            )

            # Викликати matching для MOTA/IDF1
            matches, fp, fn, id_sw = metrics._match_tracks(
                predicted=metrics.tracking_data['predictions'][frame_count],
                ground_truth=metrics.tracking_data['ground_truth'][frame_count],
                iou_threshold=0.3
            )
            metrics.tracking_data['matches'] += matches
            metrics.tracking_data['false_positives_tracking'] += fp
            metrics.tracking_data['false_negatives_tracking'] += fn
            metrics.tracking_data['id_switches'] += id_sw
            
            should_run_cnn = False
            
            if accident_state.is_accident_active(tid, frame_count):
                should_run_cnn = True
            elif tid in collision_risky_ids:
                should_run_cnn = True
            elif analyzer.detect_sudden_stop(tid, boxes, ids):
                should_run_cnn = True
            elif frame_count % HEARTBEAT_RATE == 0:
                should_run_cnn = True
            else:
                cnn_skipped += 1
            
            if should_run_cnn:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops_to_process.append(crop)
                    ids_to_process.append(tid)
                    box_map_for_cnn.append((x1, y1, x2, y2))

            # Якщо треба запускати CNN
            if should_run_cnn:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops_to_process.append(crop)
                    ids_to_process.append(tid)
                    box_map_for_cnn.append((x1, y1, x2, y2))

        # ============= CNN INFERENCE =============
        accident_detected_this_frame = False
        accident_objects = []
        cnn_scores = []

        # ====== Логування детекції
        if crops_to_process:
            cnn_start_time = time.time()

            scores = process_cnn_batch(crops_to_process, cnn_transforms, cnn_model)
            cnn_time = ((cv2.getTickCount() - cnn_start_time) / cv2.getTickFrequency()) * 1000
            
            metrics.update_cnn_metrics(len(crops_to_process), cnn_time, scores)
            
            for (x1, y1, x2, y2), score, tid in zip(box_map_for_cnn, scores, ids_to_process):
                accident_state.update_score(tid, score, frame_count)
                is_already_confirmed = accident_state.is_confirmed_accident(tid)
                should_confirm = accident_state.should_confirm_accident(tid, score)
                
                if is_already_confirmed:
                    label = f"ACCIDENT {score:.2f}"
                    color = (0, 0, 255)
                    
                    accident_detected_this_frame = True
                    accident_objects.append({
                        'track_id': tid,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': score,
                        'type': 'primary'
                    })
                    
                    logger.info(f"[FRAME {frame_count}] CONFIRMED ACCIDENT | ID={tid} | SCORE={score:.2f}")
                
                elif should_confirm and score > CONF_ACCIDENT_HIGH:
                    accident_state.confirm_accident(tid, frame_count)
                    
                    label = f"ACCIDENT {score:.2f}"
                    color = (0, 0, 255)
                    
                    accident_detected_this_frame = True
                    accident_objects.append({
                        'track_id': tid,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': score,
                        'type': 'primary'
                    })
                    
                    logger.warning(f"[FRAME {frame_count}] NEW ACCIDENT | ID={tid} | SCORE={score:.2f}")
                
                elif tid in collision_risky_ids and score > CONF_ACCIDENT_LOW:
                    label = f"WARNING {score:.2f}"
                    color = (0, 255, 255)
                    
                    if accident_detected_this_frame:
                        accident_objects.append({
                            'track_id': tid,
                            'bbox': (x1, y1, x2, y2),
                            'confidence': score,
                            'type': 'secondary'
                        })
                    
                    logger.info(f"[FRAME {frame_count}] WARNING | ID={tid} | SCORE={score:.2f}")
                
                else:
                    label = f"NORMAL {score:.2f}"
                    color = (0, 255, 0)
                
                cnn_results_cache[tid] = (color, label, score)

        # ============= ДЕТЕКЦІЯ ДОДАТКОВИХ УЧАСНИКІВ АВАРІЇ =============
        if accident_detected_this_frame and accident_objects:
            primary_positions = [obj['bbox'] for obj in accident_objects 
                                 if obj['type'] == 'primaty']

            for i, tid in enumerate(ids):
                if tid not in [obj['track_id'] for obj in accident_objects]:
                    x1,y1,x2,y2 = boxes[i]

                    for px1,py1,px2,py2 in primary_positions:
                        distance = calculate_box_distance(
                            (x1,y1,x2,y2),
                            (px1,py1,px2,py2)
                        )

                        if distance < 199:
                            accident_objects.append({
                                'track_id': tid,
                                'bbox': (x1,x2,y1,y2),
                                'confidence': cnn_results_cache.get(tid, (None,None,0.0))[2],
                                'type': 'secondary'
                            })
                            break

        if accident_detected_this_frame and accident_objects:
            should_save = any(
                accident_capture.should_save_accident(obj['track_id'], frame_count)
                for obj in accident_objects if obj['type'] == 'primary'
            )

            if should_save:
                saved_path = accident_capture.save_accident_frame(
                    frame=frame,
                    frame_number=frame_count,
                    accident_objects=accident_objects,
                    video_path=input_video
                )

                logger.critical(
                    f"[ACCIDENT SAVED] Frame {frame_count} | "
                    f"Vehicles: {len(accident_objects)} | "
                    f"Saved to: {saved_path}"
                )

        collision_warnings = len(collision_risky_ids)
        sudden_stops = sum(1 for tid in ids if analyzer.detect_sudden_stop(tid, boxes, ids))
        
        metrics.update_accident_metrics(
            accident_detected_this_frame,
            len(accident_objects) if accident_objects else 0,
            collision_warnings,
            sudden_stops
        )

        # 7. Візуалізація
        vizualuzate_tracks(ids, boxes, frame, analyzer, cnn_results_cache)

        out.write(frame)

        frame_time = ((cv2.getTickCount() - frame_start) / cv2.getTickFrequency()) * 1000
        metrics.update_frame_metrics(frame_time)

    cap.release()
    out.release()

    for frame_str, annotation in metrics.ground_truth.get('frame_annotations', {}).items():
        frame_num = int(frame_str)
        vehicles = annotation.get('vehicles', [])
        bboxes = annotation.get('bboxes', [])
        
        # Конвертуємо ID в int (щоб співпадали з DeepSort track_id)
        try:
            vehicles = [int(v) for v in vehicles]
        except ValueError:
            # якщо ID вже int або некоректні рядки
            pass
        
        # Формуємо список ground truth треків у форматі (id, bbox)
        gt_tracks = list(zip(vehicles, bboxes))
        
        # Додаємо у tracking_data
        metrics.tracking_data['ground_truth'][frame_num] = gt_tracks

    metrics.compute_map()
    metrics.compute_mota_idf1()

    final_metrics = metrics.finalize()

    analyzer_metrics = analyzer.get_metrics_summary()
    final_metrics['analyzer'] = analyzer_metrics

    metrics.save_to_file(METRICS_FILE)

    summary = accident_capture.get_accident_summary()

    print("\n" + "="*70)
    print(" "*20 + "РЕЗУЛЬТАТИ ОБРОБКИ")
    print("="*70)
    metrics.print_summary()

    print(f"\n{'*'*70}")
    print(" "*20 + "ЗВЕДЕННЯ ПО АВАРІЯХ")
    print(f"{'*'*70}")
    print(f"Всього аварій: {summary['total_accidents']}")
    print(f"Всього авто задіяно: {summary.get('total_vehicles_involved', 0)}")
    print(f"Унікальних авто: {summary.get('unique_vehicles', 0)}")
    print(f"Збережено до: {accident_capture.accidents_dir}")
    print(f"Лог метаданих: {accident_capture.metadata_file}")
    print(f"{'*'*70}\n")
    
    print(f"Обробку завершено!")
    print(f"Відео збережено: {output_path}")
    print(f"Метрики збережено: {METRICS_FILE}")
    
    logger.info(f"Processing finished. Output: {output_path}")
    return output_path


# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    try:
        output_path = accident_detection(VIDEO_PATH)
        print(f"\nУспішно завершено!")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        print(f"\nПомилка: {e}")
        raise
