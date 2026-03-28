import os
import torch
import cv2
from torchvision import datasets
from torch.utils.data import DataLoader

# Імпортуємо ВАШ клас (замініть 'accident_classifier' на назву вашого файлу, якщо вона інша)
from resnet50 import AccidentClassifier

# ==========================================
# 1. НАЛАШТУВАННЯ
# ==========================================
WEIGHTS_PATH = r"E:\personalproject\bachelor\model\weights\resnet50_accident.pth"
TEST_DIR = r"E:\personalproject\bachelor\data\classifier_dataset" # Папка з тестовими фото (підпапки: Accident, Normal)
PLOT_DIR = r"E:\personalproject\bachelor\data\plot" # Куди зберегти графіки та звіти

os.makedirs(PLOT_DIR, exist_ok=True)

# ==========================================
# 2. ТЕСТУВАННЯ
# ==========================================
def main():
    print("⏳ Завантаження моделі...")
    clf = AccidentClassifier.load(WEIGHTS_PATH)
    print(f"✅ Модель готова. Класи: {clf.class_names}")

    # Створюємо папку для результатів, якщо її немає
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Допустимі формати зображень
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    processed_count = 0

    print(f"🔍 Скануємо папку {TEST_DIR}...")
    
    # Проходимося по всіх підпапках і файлах у тестовій директорії
    for root, dirs, files in os.walk(TEST_DIR):
        for file in files:
            if file.lower().endswith(valid_extensions):
                img_path = os.path.join(root, file)
                
                # Справжній клас беремо з назви папки, в якій лежить файл
                true_label = os.path.basename(root) 
                
                # 1. Робимо передбачення вашою моделлю
                pred_label, probs = clf.predict(img_path)
                
                # Знаходимо впевненість для передбаченого класу
                pred_idx = clf.class_names.index(pred_label)
                confidence = probs[pred_idx] * 100
                
                # 2. Відкриваємо картинку через OpenCV для малювання
                img_cv2 = cv2.imread(img_path)
                if img_cv2 is None:
                    continue
                
                # 3. Визначаємо колір тексту
                # Зелений, якщо вгадали. Червоний, якщо помилилися.
                is_correct = (pred_label.lower() == true_label.lower())
                color = (0, 255, 0) if is_correct else (0, 0, 255) # BGR формат
                
                # 4. Формуємо текст
                text_top = f"Pred: {pred_label} ({confidence:.1f}%)"
                text_bot = f"True: {true_label}"
                
                # Малюємо плашки і текст, щоб було добре видно на будь-якому фоні
                cv2.putText(img_cv2, text_top, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(img_cv2, text_bot, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # 5. Зберігаємо результат
                # Назва файлу буде: СправжнійКлас_ПередбаченийКлас_Ім'яФайлу.jpg
                status_prefix = "OK_" if is_correct else "FAIL_"
                new_filename = f"{status_prefix}True-{true_label}_Pred-{pred_label}_{file}"
                out_path = os.path.join(PLOT_DIR, new_filename)
                
                cv2.imwrite(out_path, img_cv2)
                processed_count += 1

    print(f"\n🎉 Готово! Оброблено {processed_count} зображень.")
    print(f"📁 Всі картинки з передбаченнями лежать у папці: {PLOT_DIR}")
# def main():
#     print("⏳ Завантаження моделі через AccidentClassifier.load()...")
#     # Ваш клас сам знає, як правильно відновити шари і завантажити ваги!
#     clf = AccidentClassifier.load(WEIGHTS_PATH)
#     print(f"✅ Модель завантажена. Класи: {clf.class_names}")

#     # Використовуємо трансформації, які вже прописані у вашому класі (_INFERENCE_TRANSFORM)
#     print(f"📂 Завантаження тестових даних з {TEST_DIR}...")
#     test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=clf._INFERENCE_TRANSFORM)
#     test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#     print("\n🚀 Запуск оцінки (evaluate)...")
#     # Цей єдиний виклик зробить усе: порахує метрики, виведе звіт і збереже матрицю помилок!
#     results = clf.evaluate(loader=test_loader, plot_dir=PLOT_DIR)

#     print(results)
#     print(f"\n📁 Звіти та матрицю помилок збережено у папку: {PLOT_DIR}")

if __name__ == "__main__":
    main()