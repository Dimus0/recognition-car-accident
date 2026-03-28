import cv2
import json
import os

# ==========================================
# 1. НАЛАШТУВАННЯ
# ==========================================
VIDEO_PATH = r"E:\personalproject\bachelor\data\system_test\Accident\accident_video_v3.mp4" # Ваше коротке відео (5-10 сек)
OUTPUT_JSON = r"E:\personalproject\bachelor\logs\gt\gt_tracking_micro.json"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Помилка: Не вдалося відкрити відео {VIDEO_PATH}")
        return

    # Завантажуємо існуючу розмітку (якщо хочете продовжити пізніше)
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, 'r') as f:
            data = json.load(f)
            print("✅ Знайдено існуючий файл розмітки, продовжуємо...")
    else:
        data = {"frame_annotations": {}}

    frame_idx = 0
    print("\n" + "="*50)
    print(" 🛠 ІНСТРУКЦІЯ З РОЗМІТКИ:")
    print(" - Натисніть 'a' (add), щоб виділити машину мишкою.")
    print(" - Після виділення натисніть ENTER або SPACE.")
    print(" - Потім у консолі введіть ID цієї машини (наприклад, 1 або 2).")
    print(" - Натисніть 'n' (next) або ПРОБІЛ, щоб перейти до наступного кадру.")
    print(" - Натисніть 'q' (quit), щоб зберегти і вийти.")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🎬 Відео завершено.")
            break

        frame_idx += 1
        frame_str = str(frame_idx)

        # Якщо кадр вже розмічений, беремо його дані
        if frame_str not in data["frame_annotations"]:
            data["frame_annotations"][frame_str] = {
                "accident": False, # Для цього тесту це не так важливо
                "vehicles": []
            }

        while True:
            display_frame = frame.copy()
            
            # Малюємо вже додані бокси на цьому кадрі
            for v in data["frame_annotations"][frame_str]["vehicles"]:
                box = v["bbox"]
                v_id = v["id"]
                cv2.rectangle(display_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(display_frame, f"ID: {v_id}", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(display_frame, f"Frame: {frame_idx} | 'a'=Add Box | 'n'=Next | 'q'=Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Annotator", display_frame)

            key = cv2.waitKey(0) & 0xFF

            if key == ord('n') or key == 32: # 'n' або Space - наступний кадр
                break
            
            elif key == ord('q'): # 'q' - вихід і збереження
                cap.release()
                cv2.destroyAllWindows()
                save_annotations(data, OUTPUT_JSON)
                return

            elif key == ord('a'): # 'a' - додати бокс
                # Відкриває вікно виділення OpenCV (тягніть мишкою, потім натисніть Enter)
                bbox = cv2.selectROI("Annotator", display_frame, fromCenter=False, showCrosshair=True)
                
                # cv2.selectROI повертає (x, y, w, h). Нам треба (x1, y1, x2, y2)
                x, y, w, h = bbox
                if w > 0 and h > 0:
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    
                    # Запитуємо ID у консолі (це важливо для трекінгу!)
                    print(f"👉 Кадр {frame_idx}. Введіть ID для цієї машини: ", end="")
                    try:
                        v_id = int(input())
                        data["frame_annotations"][frame_str]["vehicles"].append({
                            "id": v_id,
                            "bbox": [x1, y1, x2, y2]
                        })
                        print(f"   ✅ Збережено ID {v_id}")
                    except ValueError:
                        print("   ❌ Помилка: ID має бути числом. Бокс не збережено.")
                else:
                    print("   ⚠️ Виділення скасовано.")

    save_annotations(data, OUTPUT_JSON)
    cap.release()
    cv2.destroyAllWindows()

def save_annotations(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"\n💾 Розмітку успішно збережено у {filepath}")

if __name__ == "__main__":
    main()