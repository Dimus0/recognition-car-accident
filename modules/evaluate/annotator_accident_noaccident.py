import cv2
import json
import os

def annotate_video(video_path, output_json):
    if not os.path.exists(video_path):
        print(f"❌ Відео не знайдено: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    accident_intervals = []
    start_frame = None

    frame_idx = 0
    paused = True

    cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotator", 1280, 720)

    print("\n" + "="*50)
    print(" 🎬 ІНСТРУКЦІЯ З РОЗМІТКИ:")
    print(" [Пробіл] - Відтворення / Пауза")
    print(" [ a ]    - Попередній кадр (в режимі паузи)")
    print(" [ d ]    - Наступний кадр (в режимі паузи)")
    print(" [ s ]    - Позначити ПОЧАТОК аварії (Start)")
    print(" [ e ]    - Позначити КІНЕЦЬ аварії (End)")
    print(" [ q ]    - Зберегти та вийти (Quit)")
    print("="*50 + "\n")

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        # Виведення інформації на екран
        status = "PAUSED" if paused else "PLAYING"
        color = (0, 0, 255) if paused else (0, 255, 0)
        cv2.putText(display_frame, f"Frame: {frame_idx}/{total_frames} | {status}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if start_frame is not None:
            cv2.putText(display_frame, f"⚠️ Accident started at frame: {start_frame}", 
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            # Малюємо червону рамку під час активної розмітки інциденту
            cv2.rectangle(display_frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 5)

        cv2.imshow("Annotator", display_frame)

        # Обробка клавіш
        delay = 0 if paused else int(1000 / fps)
        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('d') and paused:
            frame_idx = min(frame_idx + 1, total_frames - 1)
        elif key == ord('a') and paused:
            frame_idx = max(frame_idx - 1, 0)
        elif key == ord('s'):
            start_frame = frame_idx
            print(f"🟢 Початок ДТП зафіксовано на кадрі: {start_frame}")
        elif key == ord('e'):
            if start_frame is not None:
                accident_intervals.append({"start": start_frame, "end": frame_idx})
                print(f"🔴 Кінець ДТП зафіксовано. Збережено інтервал: {start_frame} - {frame_idx}")
                start_frame = None
            else:
                print("⚠️ Спочатку натисніть 's', щоб вказати початок!")
        
        # Рух вперед, якщо відео грає
        if not paused:
            frame_idx += 1

        if frame_idx >= total_frames:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Формування JSON у форматі для MetricsTracker
    ground_truth = {
        "meta": {"note": "Manual frame-level annotation"},
        "frame_annotations": {}
    }

    total_accident_frames = 0
    for interval in accident_intervals:
        for f in range(interval["start"], interval["end"] + 1):
            ground_truth["frame_annotations"][str(f)] = {
                "accident": True,
                "bboxes": [],   # Порожні, оскільки оцінюємо на рівні кадрів
                "vehicles": []
            }
            total_accident_frames += 1

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=4, ensure_ascii=False)
    
    print("\n✅ Розмітку завершено!")
    print(f"Збережено інцидентів: {len(accident_intervals)}")
    print(f"Всього аварійних кадрів: {total_accident_frames}")
    print(f"Файл збережено: {output_json}")

if __name__ == "__main__":
    # ТУТ ВКАЖИ ШЛЯХ ДО ВІДЕО ТА ШЛЯХ КУДИ ЗБЕРЕГТИ JSON
    VIDEO_FILE = "data/video/accident_video_v8.mp4" 
    OUTPUT_FILE = r"E:\personalproject\bachelor\logs\gt\gt_accident_video_v8.json"
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    annotate_video(VIDEO_FILE, OUTPUT_FILE)