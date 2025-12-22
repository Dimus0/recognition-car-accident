import cv2
import numpy as np
import torch
from torchvision import transforms
import logging
import os
from ultralytics import YOLO
from collections import defaultdict, deque
from modules.utils import *
from modules.analyzer import TrafficAnalyzer
from modules.accidenttracker import AccidentStateTracker
from modules.accidentcapture import AccidentFrameCapture

from deep_sort_realtime.deepsort_tracker import DeepSort



# ---------------------- CONFIG ----------------------

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_DIR = r'D:\project\bachelor\logs'
OUTPUT_DIR = os.path.join(LOG_DIR, "output_result")
YOLO_MODEL_PATH = r"D:\project\bachelor\model\weights\yolov8-fine-tuning-15.pt"
CNN_WEIGHTS_PATH = r"D:\project\bachelor\model\weights\accident_cnn_model.pth"
SUMMARY_STATICTICS = r"D:\project\bachelor\logs\statictics.txt"


os.makedirs(OUTPUT_DIR, exist_ok=True)

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



# ---------------------- LOAD MODELS ----------------------

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
    cap = cv2.VideoCapture(input_video)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_path = os.path.join(OUTPUT_DIR, "result.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # -------- Лінії після яких відбувається трекінг --------)

    '''
        Область для відео із шосе
    '''
    ROI_POLYGON = np.array([
        (881, 582),
        (1018, 587),
        (912, 1004),
        (226, 892)
    ], dtype=np.int32)

    '''
        Область для відео із ДТП
    '''
    # ROI_POLYGON = np.array([
    #     (9, 216),
    #     (19, 996),
    #     (1450, 791),
    #     (511, 230)
    # ], dtype=np.int32)

    # -------- END --------
    analyzer = TrafficAnalyzer()

    accident_capture = AccidentFrameCapture(OUTPUT_DIR)
    accident_state = AccidentStateTracker()

    frame_count = 0
    last_results = []

    cnn_results_cache = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        cv2.polylines(frame, [ROI_POLYGON], isClosed=True, color=(255, 0, 0), thickness=2)

        results = yolo_model(frame, conf=0.4, verbose=False)

        detections =  []
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

        crops_to_process = []
        ids_to_process = []
        box_map_for_cnn = []

        collision_risky_ids = analyzer.predict_collision(ids)

        accident_state.cleanup_old_accidents(frame_count)

        for i, tid in enumerate(ids):
            x1, y1, x2, y2 = boxes[i]
            
            # Евристика для запуску CNN:
            should_run_cnn = False
            
            # A. Якщо аналізатор каже про ризик зіткнення
            if accident_state.is_accident_active(tid, frame_count):
                should_run_cnn = True
            
            # B. Кінематична аномалія (раптова зупинка)
            elif tid in collision_risky_ids:
                should_run_cnn = True
                
            # C. Heartbeat (запускаємо рідко для профілактики, наприклад, кожні 15 кадрів)
            elif check_kinematic_anomalies(analyzer.track_history, tid, boxes[i]):
                # Перевіряємо чи є інші авто поруч
                nearby_vehicles = count_nearby_vehicles(boxes, i, distance_threshold=150)
                
                # Запускаємо CNN тільки якщо є авто поруч
                if nearby_vehicles > 0:
                    should_run_cnn = True

            elif frame_count % 30 == 0:  # Знижена частота
                should_run_cnn = True
            
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

        accident_detected_this_frame = False
        accident_objects = []

        # 6. CNN Inference (тільки якщо є "підозрілі" кандидати)
        if crops_to_process:
            scores = process_cnn_batch(crops_to_process, cnn_transforms, cnn_model)
            
            for (x1, y1, x2, y2), score, tid in zip(box_map_for_cnn, scores, ids_to_process):
                # Оновлюємо кеш результатів

                accident_state.update_score(tid, score, frame_count)
                is_already_confirmed = accident_state.is_confirmed_accident(tid)
                should_confirm = accident_state.should_confirm_accident(tid, score)

                if is_already_confirmed:
                    # Аварія вже підтверджена - показуємо як аварію незалежно від поточного скору
                    label = f"ACCIDENT {score:.2f}"
                    color = (0, 0, 255)  # Red
                    
                    accident_detected_this_frame = True
                    accident_objects.append({
                        'track_id': tid,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': score,
                        'type': 'primary'
                    })
                    
                    logger.info(f"[FRAME {frame_count}] CONFIRMED ACCIDENT | ID={tid} | SCORE={score:.2f}")
                
                elif should_confirm and score > 0.85:
                    # НОВА АВАРІЯ - підтверджуємо
                    accident_state.confirm_accident(tid, frame_count)
                    
                    label = f"ACCIDENT {score:.2f}"
                    color = (0, 0, 255)  # Red
                    
                    accident_detected_this_frame = True
                    accident_objects.append({
                        'track_id': tid,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': score,
                        'type': 'primary'
                    })
                    
                    logger.warning(f"[FRAME {frame_count}] NEW ACCIDENT DETECTED | ID={tid} | SCORE={score:.2f}")
                
                elif tid in collision_risky_ids and score > 0.70:
                    # Ризик аварії - WARNING
                    label = f"WARNING {score:.2f}"
                    color = (0, 255, 255)  # Yellow
                    
                    # Додаємо як вторинний об'єкт якщо є основна аварія
                    if accident_detected_this_frame:
                        accident_objects.append({
                            'track_id': tid,
                            'bbox': (x1, y1, x2, y2),
                            'confidence': score,
                            'type': 'secondary'
                        })
                    
                    logger.info(f"[FRAME {frame_count}] WARNING | ID={tid} | SCORE={score:.2f}")
                
                else:
                    # Нормальна ситуація
                    label = f"NORMAL {score:.2f}"
                    color = (0, 255, 0)  # Green
                    # logger.warning(f"[FRAME {frame_count}] NORMAL | ID={tid} | SCORE={score:.2f}")
                
                cnn_results_cache[tid] = (color, label, score)

        if accident_detected_this_frame and accident_objects:
            primary_positions = [obj['bbox'] for obj in accident_objects if obj['type'] == 'primaty']

            for i, tid in enumerate(ids):
                if tid not in [obj['track_id'] for obj in accident_objects]:
                    x1,y1,x2,y2 = boxes[i]

                    for px1,py1,px2,py2 in primary_positions:
                        distance = calculate_box_distance(
                            (x1,y1,x2,y2),
                            (px1,py1,px2,py2)
                        )

                        if distance < 200:
                            accident_objects.append({
                                'track_id': tid,
                                'bbox': (x1,x2,y1,y2),
                                'confidence': cnn_results_cache.get(tid, (None,None,0.0))[2],
                                'type': 'secondary'
                            })
                            break
            should_save = any(
                accident_capture.should_save_accident(obj['track_id'],frame_count)
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
                    f"[ACCIDENT SAVED] Frame {frame_count} |"
                    f"Vehicles: {len(accident_objects)} |"
                    f"Saved to: {saved_path}"
                )


        # 7. Візуалізація (Draw Loop)
        for i, tid in enumerate(ids):
            x1, y1, x2, y2 = boxes[i]
            
            # Малюємо траєкторію
            points = analyzer.track_history.get(tid, [])
            if len(points) > 1:
                pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=(255, 0, 0), thickness=2)

            # Беремо дані з кешу (або дефолтні, якщо CNN ще не запускалась для цього ID)
            if tid in cnn_results_cache:
                color, label, score = cnn_results_cache[tid]
            else:
                color, label = (255, 255, 255), f"ID {tid}" # White defaults

            # Малюємо бокс і текст
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)

    cap.release()
    out.release()

    summary = accident_capture.get_accident_summary()

    print(f"\n {'*'*50}")
    print("ACCIDENT DETECTION SUMMARY")
    print(f"Total accidents detected: {summary['total_accidents']}")
    print(f"Total vehicles involved: {summary.get('total_vehicles_involved', 0)}")
    print(f"Unique vehicles: {summary.get('unique_vehicles', 0)}")
    print(f"Accidents saved to: {accident_capture.accidents_dir}")
    print(f"Metadata log: {accident_capture.metadata_file}")
    print(f"\n {'*'*50}")

    
    print(f"Processing finished. Output saved to: {output_path}")
    logger.info(f"Processing finished. Output saved to: {output_path}")

    return output_path


# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    video_path = r"D:\project\test_video.mp4"
    # video_path = r"D:\project\videoplayback.mp4"
    accident_detection(video_path)
