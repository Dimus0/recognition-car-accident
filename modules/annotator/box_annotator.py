import cv2
import json
import torch
import numpy as np
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------------------- CONFIG ----------------------
MODEL_PATH = r"E:\personalproject\bachelor\modules\annotator\yolo11l.pt" 
VIDEO_PATH = r"E:\personalproject\bachelor\data\video\noaccident_video_v3.mp4"
OUTPUT_GT_PATH = r"E:\personalproject\bachelor\logs\gt_box\gt_noaccident_video_v3.json"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ROI_POLYGON = np.array([
    [(3, 633), (633, 641), (633, 424), (474, 429), (3, 632)]
], dtype=np.int32)


# ---------------------- ROI ----------------------
def is_point_in_roi(point, polygon):
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


# ---------------------- MAIN ----------------------
def generate_annotations():
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)  # ✅ явно вказуємо device

    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = 0
    annotations = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ---------------- ROI VIS ----------------
        cv2.polylines(frame, [ROI_POLYGON], True, (255, 0, 0), 2)

        # ---------------- YOLO ----------------
        results = model(frame, conf=0.25, verbose=False)[0]

        print(f"[Frame {frame_count}] detections: {len(results.boxes)}")  # ✅ дебаг

        detections_in_roi = []

        for box in results.boxes:   # ✅ ВИПРАВЛЕНО (без results[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            original_cls = int(box.cls[0])

            # -------- DEBUG CLASSES --------
            print(f"Class: {original_cls}, Conf: {conf:.2f}")

            # ---------------- CLASS MAPPING ----------------
            if original_cls == 2:
                mapped_class = "car"
            elif original_cls in [5,7]:
                mapped_class = "bigcar"
            else:
                continue

            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            # ---------------- ROI FILTER ----------------
            if is_point_in_roi(center, ROI_POLYGON):
                detections_in_roi.append(
                    ([x1, y1, x2 - x1, y2 - y1], conf, mapped_class)
                )

        print("After ROI:", len(detections_in_roi))  # ✅ дебаг

        # ---------------- TRACKING ----------------
        tracks = tracker.update_tracks(detections_in_roi, frame=frame)

        frame_data = {"frame_id": frame_count, "objects": []}

        for track in tracks:

            ltrb = track.to_ltrb()

            track_class = track.det_class if track.det_class else "car"

            frame_data["objects"].append({
                "track_id": int(track.track_id),
                "bbox": [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])],
                "class": track_class
            })

            # ---------------- VISUALIZATION ----------------
            cv2.rectangle(frame,
                          (int(ltrb[0]), int(ltrb[1])),
                          (int(ltrb[2]), int(ltrb[3])),
                          (0, 255, 0), 2)

            cv2.putText(frame,
                        f"{track_class} ID:{track.track_id}",
                        (int(ltrb[0]), int(ltrb[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        annotations.append(frame_data)

        cv2.imshow("Annotation Process", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ---------------- SAVE ----------------
    output_data = {
        "video_source": os.path.basename(VIDEO_PATH),
        "total_frames": frame_count,
        "annotations": annotations
    }

    with open(OUTPUT_GT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

    print(f"Finished. Data saved to {OUTPUT_GT_PATH}")

    cap.release()
    cv2.destroyAllWindows()
# Проблема в записі

if __name__ == "__main__":
    generate_annotations()