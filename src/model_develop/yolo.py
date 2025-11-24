from ultralytics import YOLO
import os
import cv2 

model = YOLO('yolov8m.pt') 

# --- Шлях до відеофайлу ---
source_path = r'D:\bachelor\video\videoplayback.mp4'
# source_path = "https://www.youtube.com/watch?v=qMYlpMsWsBE" 

print(f"Обробка джерела: {source_path}")

try:
    # --- Запуск трекінгу з візуалізацією ---
    results_generator = model.track(
        source=source_path,
        stream=True,     
        show=True,       
        save=False,      
        conf=0.5,
        classes=(2, 3, 5, 7), # <--- ВИПРАВЛЕНО: тепер це кортеж (tuple)
        persist=True     
    )

    frame_number = 0
    for result in results_generator:
        frame_number += 1
        print(f"\n========== КАДР {frame_number} ==========")
        
        boxes = result.boxes
        names = result.names
        
        if len(boxes) == 0:
            print("  На цьому кадрі нічого не знайдено.")
            continue

        for box in boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            class_name = names[class_id]
            
            track_id = None
            if box.id is not None:
                track_id = int(box.id[0].cpu().numpy())
            
            print(f"  ЗНАЙДЕНО: {class_name.upper()}")
            print(f"    Впевненість: {confidence*100:.1f}%")
            print(f"    Координати (xyxy): {coords}")
            if track_id is not None:
                print(f"    ID об'єкта: {track_id}")

except Exception as e:
    # Тепер виведення помилки буде більш детальним
    print(f"Сталася помилка під час обробки: {e}")
    import traceback
    traceback.print_exc() # Друкує повний стек виклику помилки

print("\nОбробку завершено.")
cv2.destroyAllWindows()