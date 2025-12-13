import gradio as gr
import cv2
import numpy as np
import torch
from torchvision import transforms
import logging
import os
from ultralytics import YOLO
from collections import defaultdict, deque
from model.src.cnn import AccidentCNN

from modules.utils import TrafficAnalyzer,process_cnn_batch,is_crossing_line,point_to_line_distance


'''
    РОЗРОБКА UI. Запускати console.py для основного функціоналу.
'''









# ---------------------- CONFIG ----------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOG_DIR = r'D:\project\bachelor\logs'
OUTPUT_DIR = os.path.join(LOG_DIR, "output_result")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.remove(r"D:\project\bachelor\logs\accident_detection.log")

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "accident_detection.log"),
    encoding='utf-8',
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("accident_detector")

YOLO_MODEL_PATH = r"D:\project\bachelor\model\weights\yolov8-fine-tuning-15.pt"
CNN_WEIGHTS_PATH = r"D:\project\bachelor\model\weights\accident_cnn_model.pt"


# ---------------------- LOAD MODELS ----------------------
def load_models():
    # CNN
    cnn = AccidentCNN().to(DEVICE)
    cnn_weights = torch.load(CNN_WEIGHTS_PATH, map_location=DEVICE)
    cnn.load_state_dict(cnn_weights)
    cnn.eval()

    # YOLO
    yolo = YOLO(YOLO_MODEL_PATH)
    yolo.to(DEVICE)

    logger.info("Models loaded successfully.")
    return cnn, yolo


cnn_model, yolo_model = load_models()

cnn_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ---------------------- UTILS ----------------------
class TrafficAnalyzer:
    """Tracks movement history & detects prediction-based collision risks."""
    def __init__(self):
        self.track_history = defaultdict(lambda: deque(maxlen=20))

    def update_tracks(self, boxes, ids):
        if ids is None:
            return
        for (x1, y1, x2, y2), tid in zip(boxes, ids):
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            self.track_history[tid].append((cx, cy))

    def clean_old_tracks(self, active_ids):
        for tid in list(self.track_history.keys()):
            if tid not in active_ids:
                del self.track_history[tid]

    def predict_collision(self, ids):
        risky = set()
        ids = list(ids)

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]

                if len(self.track_history[a]) < 5 or len(self.track_history[b]) < 5:
                    continue

                p1 = np.array(self.track_history[a][-1])
                p2 = np.array(self.track_history[b][-1])

                dist = np.linalg.norm(p1 - p2)

                if dist < 80:  # threshold
                    risky.add(a)
                    risky.add(b)

        return risky

def process_cnn_batch(crops):
    if not crops:
        return []

    tensors = []
    for crop in crops:
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            t = cnn_transforms(rgb)
            tensors.append(t)
        except:
            tensors.append(torch.zeros(3, 224, 224))

    batch = torch.stack(tensors).to(DEVICE)

    with torch.no_grad():
        out = cnn_model(batch)

        if out.shape[1] == 1:  # BCE
            return torch.sigmoid(out).cpu().numpy().flatten().tolist()
        else:
            return torch.softmax(out, dim=1)[:, 1].cpu().numpy().tolist()

def point_to_line_distance(px, py, x1, y1, x2, y2):
    """Мінімальна відстань від точки до лінії"""
    A = px - x1
    B = py - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = dot / len_sq if len_sq != 0 else -1

    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = px - xx
    dy = py - yy
    return np.sqrt(dx * dx + dy * dy)


def is_crossing_line(bbox, line, threshold=5):
    """Перевірка чи центр бокса перетинає лінію"""
    (x1, y1, x2, y2) = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    (lx1, ly1), (lx2, ly2) = line

    dist = point_to_line_distance(cx, cy, lx1, ly1, lx2, ly2)
    return dist < threshold

# ---------------------- MAIN PIPELINE ----------------------
def accident_detection(input_video):
    if input_video is None:
        raise ValueError("No input video provided.")

    cap = cv2.VideoCapture(input_video)

    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = os.path.join(OUTPUT_DIR, "result.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # LINE_Y = int(height * 0.4)
    LINE_YELLOW = ((880, 582), (1021, 585))
    LINE_GREEN  = ((1068, 577), (1184, 571))
    analyzer = TrafficAnalyzer()

    frame_count = 0
    last_results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # cv2.line(frame, (880, 582), (1021, 585), (0, 255, 255), 3)
        # cv2.line(frame, (1068, 577), (1184, 571), (0, 255, 0), 3)

        frame_count += 1

        # ---- YOLO Tracking ----
        results = yolo_model.track(frame, persist=True, verbose=False, conf=0.3, tracker="bytetrack.yaml")

        if not results or results[0].boxes.id is None:
            # draw only previous boxes
            for (x1, y1, x2, y2, color, label) in last_results:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            out.write(frame)
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.int().cpu().numpy()

        analyzer.update_tracks(boxes, ids)
        analyzer.clean_old_tracks(ids)

        # ---- collect active objects that crossed line ----
        crops, boxes_filtered, ids_filtered = [], [], []

        for (x1, y1, x2, y2), tid in zip(boxes, ids):

            crossed = (
                is_crossing_line((x1, y1, x2, y2), LINE_YELLOW) or
                is_crossing_line((x1, y1, x2, y2), LINE_GREEN)
            )
            if crossed:
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)
                    boxes_filtered.append((x1, y1, x2, y2))
                    ids_filtered.append(tid)

        collision_risky_ids = analyzer.predict_collision(ids_filtered)

        # ---- CNN every 3 frames ----
        current_results = []

        if crops and frame_count % 3 == 0:
            scores = process_cnn_batch(crops)

            for (x1, y1, x2, y2), score, tid in zip(boxes_filtered, scores, ids_filtered):

                if score > 0.65:
                    color, label = (0, 0, 255), f"ACCIDENT {score:.2f}"

                    # --- Logging Accident ---
                    logger.warning(
                        f"[FRAME {frame_count}] ACCIDENT DETECTED | "
                        f"ID={tid} | SCORE={score:.2f} | BOX=({x1},{y1},{x2},{y2})"
                    )

                elif tid in collision_risky_ids:
                    color, label = (0, 255, 255), "WARNING"

                    # --- Logging Warning ---
                    logger.info(
                        f"[FRAME {frame_count}] WARNING: potential collision | "
                        f"ID={tid} | BOX=({x1},{y1},{x2},{y2})"
                    )

                else:
                    color, label = (0, 255, 0), f"NORMAL {score:.2f}"

                    # --- Logging Normal ---
                    logger.debug(
                        f"[FRAME {frame_count}] Normal object | "
                        f"ID={tid} | SCORE={score:.2f}"
                    )

                current_results.append((x1, y1, x2, y2, color, label))

            last_results = current_results  # update only when new data arrived

        # ---- Draw last known results ----
        for (x1, y1, x2, y2, color, label) in last_results:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)

    cap.release()
    out.release()

    return output_path


# ---------------------- UI ----------------------
iface = gr.Interface(
    fn=accident_detection,
    inputs=gr.Video(label="Upload video"),
    outputs=gr.Video(label="Processed video"),
    title="Accident Detection & Prediction System",
    description="YOLOv8 Tracking → Movement Prediction → CNN Verification"
)

if __name__ == "__main__":
    iface.launch()
