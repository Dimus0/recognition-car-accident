import cv2
import os

class ROISelector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.points = []
        
        self.cap = cv2.VideoCapture(video_path)
        ret, self.frame = self.cap.read()
        if not ret:
            print(f"❌ Не вдалося прочитати відео: {video_path}")
            return
            
        self.clone = self.frame.copy()
        
        cv2.namedWindow("ROI Selector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ROI Selector", 1280, 720)
        cv2.setMouseCallback("ROI Selector", self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            # Малюємо точку
            cv2.circle(self.frame, (x, y), 5, (0, 0, 255), -1)
            # З'єднуємо лініями
            if len(self.points) > 1:
                cv2.line(self.frame, self.points[-2], self.points[-1], (0, 255, 0), 2)
            cv2.imshow("ROI Selector", self.frame)
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Видалення останньої точки (правий клік)
            if self.points:
                self.points.pop()
                self.frame = self.clone.copy()
                for i in range(len(self.points)):
                    cv2.circle(self.frame, self.points[i], 5, (0, 0, 255), -1)
                    if i > 0:
                        cv2.line(self.frame, self.points[i-1], self.points[i], (0, 255, 0), 2)
                cv2.imshow("ROI Selector", self.frame)

    def run(self):
        if not hasattr(self, 'frame'):
            return
            
        print("\n" + "="*50)
        print(f" 🎯 НАЛАШТУВАННЯ ROI ДЛЯ: {self.video_name}")
        print(" [Лівий клік]  - Поставити точку")
        print(" [Правий клік] - Скасувати останню точку")
        print(" [ C ]         - Очистити все")
        print(" [ Enter ]     - Зберегти та вивести координати")
        print(" [ Q ]         - Вийти без збереження")
        print("="*50 + "\n")

        cv2.imshow("ROI Selector", self.frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter
                if len(self.points) > 2:
                    # Замикаємо полігон для краси на екрані
                    cv2.line(self.frame, self.points[-1], self.points[0], (0, 255, 0), 2)
                    cv2.imshow("ROI Selector", self.frame)
                    cv2.waitKey(500)
                    
                    # Виводимо готовий рядок для config.py
                    print("\n✅ ГОТОВО! Скопіюйте цей рядок у ваш словник ROI_MAPPING в config.py:\n")
                    print(f'    "{self.video_name}": {self.points},')
                    print("\n")
                else:
                    print("⚠️ Потрібно мінімум 3 точки для створення зони (полігону)!")
                break
            elif key == ord('c'):
                self.points = []
                self.frame = self.clone.copy()
                cv2.imshow("ROI Selector", self.frame)
            elif key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Вкажіть шлях до нового відео тут:
    VIDEO_FILE = r"E:\personalproject\bachelor\data\video\accident_video_v7.mp4"
    
    selector = ROISelector(VIDEO_FILE)
    selector.run()