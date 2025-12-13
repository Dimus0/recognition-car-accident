import cv2
import numpy as np
import torch
from torchvision import transforms
import logging
import os
from ultralytics import YOLO
from collections import defaultdict, deque
from model.src.cnn import AccidentCNN
from modules.utils import load_models,TrafficAnalyzer,process_cnn_batch,is_inside_roi,compute_iou


from deep_sort_realtime.deepsort_tracker import DeepSort



# ---------------------- CONFIG ----------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_DIR = r'D:\project\bachelor\logs'
OUTPUT_DIR = os.path.join(LOG_DIR, "output_result")
YOLO_MODEL_PATH = r"D:\project\bachelor\model\weights\yolov8-fine-tuning-15.pt"
CNN_WEIGHTS_PATH = r"D:\project\bachelor\model\weights\accident_cnn_model.pt"

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
    transforms.ToTensor()
])


# ---------------------- MAIN PIPELINE ----------------------
def accident_detection(input_video):
    cap = cv2.VideoCapture(input_video)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_path = os.path.join(OUTPUT_DIR, "result.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # -------- Лінії після яких відбувається трекінг --------
    LINE_YELLOW = ((880, 582), (1021, 585))
    LINE_GREEN = ((1068, 577), (1184, 571))

    ROI_POLYGON = np.array([
        (881, 582),
        (1018, 587),
        (912, 1004),
        (226, 892)
    ], dtype=np.int32)

    lines = [LINE_YELLOW, LINE_GREEN]
    # -------- END --------

    analyzer = TrafficAnalyzer()
    frame_count = 0
    last_results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        # Малюємо лінії
        # cv2.line(frame, LINE_YELLOW[0], LINE_YELLOW[1], (0, 255, 255), 3)
        # cv2.line(frame, LINE_GREEN[0], LINE_GREEN[1], (0, 255, 0), 3)

        cv2.polylines(frame, [ROI_POLYGON], isClosed=True, color=(255, 0, 0), thickness=2)

        # Зона активації перед лінією
        for line, color in zip(lines, [(0,255,255),(0,255,0)]):
            cv2.line(frame, (line[0][0], line[0][1]-80), (line[1][0], line[1][1]-80), color, 1)

        # YOLO трекінг
        # results = yolo_model.track(frame, persist=True, verbose=False, conf=0.3, tracker="bytetrack.yaml")

        results = yolo_model(frame, conf=0.4, verbose=False)

        detections =  []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                if conf < 0.3:
                    continue

                detections.append((
                    [x1,y1, x2-x1, y2-y1],
                    conf,
                    cls
                ))
        
        tracks = deepsort.update_tracks(detections, frame=frame)

        boxes = []
        ids = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            l, t, r, b = map(int, track.to_ltrb())
            boxes.append((l, t, r, b))
            ids.append(track.track_id)
        
        analyzer.update_tracks(boxes, ids)
        analyzer.clean_old_tracks(ids)
        analyzer.activate_near_line(boxes, ids, lines, threshold=80)

        # ---------- DRAW TRAJECTORIES ----------
        for tid, points in analyzer.track_history.items():
            if tid not in analyzer.active_ids:
                continue

            if len(points) < 2:
                continue

            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

            cv2.polylines(
                frame,
                [pts],
                isClosed=False,
                color=(255, 0, 0),   # синя траєкторія
                thickness=2
            )

            # ID біля останньої точки
            cx, cy = points[-1]
            cv2.putText(
                frame,
                f"ID {tid}",
                (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

        
        crops, boxes_filtered, ids_filtered = [], [], []

        for (x1, y1, x2, y2), tid in zip(boxes, ids):

            if not is_inside_roi((x1, y1, x2, y2), ROI_POLYGON):
                continue

            if tid not in analyzer.active_ids:
                continue

            if not analyzer.is_moving_towards_camera(tid):
                continue

            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:
                crops.append(crop)
                boxes_filtered.append((x1, y1, x2, y2))
                ids_filtered.append(tid)

        collision_risky_ids = analyzer.predict_collision(ids_filtered)
        current_results = []

        if crops and frame_count % 3 == 0:
            scores = process_cnn_batch(crops,cnn_transforms,cnn_model)
            for (x1, y1, x2, y2), score, tid in zip(boxes_filtered, scores, ids_filtered):
                if score > 0.90:
                    color, label = (0, 0, 255), f"ACCIDENT {score:.2f}"
                    logger.warning(f"[FRAME {frame_count}] ACCIDENT | ID={tid} | SCORE={score:.2f}")
                elif tid in collision_risky_ids and score > 0.77:
                    color, label = (0, 255, 255), "WARNING"
                    logger.info(f"[FRAME {frame_count}] WARNING | ID={tid}")
                else:
                    color, label = (0, 255, 0), f"NORMAL {score:.2f}"
                current_results.append((x1, y1, x2, y2, color, label))
            last_results = current_results

        for (x1, y1, x2, y2, color, label) in last_results:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Processing finished. Output saved to: {output_path}")
    return output_path

# ---------------------- ENTRY POINT ----------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python run.py <video_path>")
        exit()
    accident_detection(sys.argv[1])
