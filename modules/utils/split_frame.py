import cv2
import os
import sys

# ---------------------- ПАРАМЕТРИ ----------------------

video_path = r"E:\personalproject\bachelor\data\video\Highway_traffic.mp4"
output_folder = r"E:\personalproject\bachelor\data\video\fragments"

os.makedirs(output_folder, exist_ok=True)

# ---------------------- ВІДКРИВАЄМО ВІДЕО ----------------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Зберігаємо кадр
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
    cv2.imwrite(frame_filename, frame)

    if frame_count == 3:
        break
    else:
        frame_count += 1

cap.release()
print(f"Finished! {frame_count} frames saved to {output_folder}")
