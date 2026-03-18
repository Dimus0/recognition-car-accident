import os
import shutil
import cv2
from collections import defaultdict, deque
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------

VIDEO_PATH = r"E:\personalproject\bachelor\data\video\highway_traffic.mp4"
YOLO_WEIGHTS = r"E:\personalproject\bachelor\model\weights\yolov8-fine-tuning.pt"      # або твої fine-tuned ваги
OUTPUT_DIR = r"E:\personalproject\bachelor\data\output"

MAX_TRAJ_LEN = 30

# -----------------------------
# Очистка папки результатів
# -----------------------------

def reset_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path, exist_ok=True)


# -----------------------------
# Малювання траєкторій
# -----------------------------

def draw_trajectory(frame, track_id, track_history):

    points = track_history[track_id]

    for i in range(1, len(points)):

        cv2.line(
            frame,
            points[i - 1],
            points[i],
            (0, 255, 255),
            2
        )


# -----------------------------
# MAIN
# -----------------------------

def main():

    reset_folder(OUTPUT_DIR)

    model = YOLO(YOLO_WEIGHTS)

    cap = cv2.VideoCapture(VIDEO_PATH)

    track_history = defaultdict(lambda: deque(maxlen=MAX_TRAJ_LEN))

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="botsort.yaml"
        )

        result = results[0]

        if result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, ids):

                x1, y1, x2, y2 = map(int, box)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                track_history[int(track_id)].append((cx, cy))

                # bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # ID
                cv2.putText(
                    frame,
                    f"ID {int(track_id)}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                draw_trajectory(frame, int(track_id), track_history)

        cv2.imshow("Tracking + Trajectory", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()