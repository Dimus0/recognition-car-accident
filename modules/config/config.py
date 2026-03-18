import os
import torch
import numpy as np

class Config:
    """Глобальні налаштування проекту"""

    # ------------------ 1. BASE PATHS ------------------
    BASE_DIR = r"E:\personalproject\bachelor"
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    OUTPUT_DIR = os.path.join(LOG_DIR, "fragments")
    OUTPUT_DIR_CLIP = os.path.join(OUTPUT_DIR, "clip")
    METRICS_FILE = os.path.join(LOG_DIR, "metrics_summary.json")
    ARTIFACTS_PATH = os.path.join(LOG_DIR,"artifacts")
    
    GROUND_TRUTH_PATH = os.path.join(LOG_DIR, "ground_truth.json")


    # VIDEO_PATH                = r"E:\personalproject\bachelor\data\video\accident_video.mp4"
    VIDEO_PATH              = r"E:\personalproject\bachelor\data\video\highway_traffic.mp4"
    # VIDEO_PATH              = r"E:\personalproject\bachelor\data\video\traffic.mp4"

    # ------------------ 2. MODELS ------------------
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    YOLO_MODEL_PATH = os.path.join(BASE_DIR, "model", "weights", "yolov8-fine-tuning.pt")
    # YOLO_MODEL_PATH = os.path.join(BASE_DIR, "model", "weights", "yolov11.pt")
    # CNN_WEIGHTS_PATH = os.path.join(BASE_DIR, "model", "weights", "accident_cnn_model.pth")
    CLASSIFIER_WEIGHTS_PATH = os.path.join(BASE_DIR, "model", "weights", "resnet50_accident.pth")
    LSTM_MODEL_PATH = os.path.join(BASE_DIR, "model", "weights", "accident_lstm_model.pt")
    LSTM_SCALER_PATH = os.path.join(BASE_DIR, "model", "weights", "motion_lstm_scaler.pkl")
    # ------------------ 3. DETECTION & CNN ------------------
    CONF_YOLO               = 0.6
    CONF_ACCIDENT_HIGH      = 0.95   # Поріг точного ДТП
    CONF_ACCIDENT_LOW       = 0.88    # Поріг попередження
    HEARTBEAT_RATE          = 30         # Як часто перевіряти авто без підозр # було 10
    VIDEO_BUFFER_SECONDS    = 4.0  # Буфер до/після аварії               # було 2

    # ------------------ 4. MOTION LSTM ------------------
    LSTM_OBS_LEN            = 20
    LSTM_PRED_LEN           = 30
    LSTM_HIDDEN             = 128
    LSTM_LAYERS             = 2
    LSTM_DROPOUT            = 0.3
    LSTM_COORD_SCALE        = 10.0
    LSTM_COLLISION_PX       = 80
    LSTM_COLLISION_FRAMES   = 10
    LSTM_ACCIDENT_THRESH    = 0.50

    # --------------------- ResNet50 ------------------

    RESNET50_NUM_CLASSES  = 2
    RESNET50_DROPOUT      = 0.3
    RESNET50_ACCIDENT_IDX = 0
    CNN_CONTEXT_PADDING = 120
    CNN_SAVE_CROP_THRESH = 0.80
    RESNET_CROPS_DIR     = os.path.join(LOG_DIR, "resnet_crops")
    # ------------------ 5. TRAFFIC ANALYZER ------------------
    ANALYZER_MAX_HISTORY        = 25
    ANALYZER_COLLISION_DIST     = 80
    COLLISION_TTC_THRESHOLD     = 1.5
    # Кінематичні пороги (фінальні, без дублювань)
    SUDDEN_STOP_THRESHOLD       = 0.35
    MIN_SPEED_FOR_STOP          = 6.0


    # ------------------ 6. ANALYZER (кінематика) ------------------
    # Рекомендовані значення для мінімізації false-positives
    _DEFAULT_MIN_COSINE_CONVERGENCE   = 0.22   # нижче → ігноруємо як розбіжну пару
    _DEFAULT_SAME_DIRECTION_THRESHOLD = 0.82   # вище → попутні авто, не ризик
    _DEFAULT_MIN_RELATIVE_SPEED       = 1.2    # пікс/кадр; нижче → не рахуємо TTC
    _DEFAULT_TTC_EMA_ALPHA            = 0.35   # вага нового значення в EMA
    _DEFAULT_DENSITY_RADIUS           = 200    # пікс для підрахунку сусідів
    _DEFAULT_DENSITY_DIVISOR          = 0.30   # знижує поріг при щільному русі
    _DEFAULT_COMPOSITE_RISK_MIN       = 0.50   # трохи суворіший поріг ризику
    _DEFAULT_COMPOSITE_WEIGHTS        = (0.50, 0.30, 0.20)  # (ttc, cos, dist)
# ------------------ 7. ADAPTIVE HEARTBEAT ------------------
    HEARTBEAT_BASE          = 30   # базовий ритм (кадрів)
    HEARTBEAT_CALM          = 60   # ритм для спокійних треків (avg score < 0.15)
    HEARTBEAT_ALERT         = 10   # ритм для тривожних треків (avg score > 0.40)
    HEARTBEAT_CALM_THRESH   = 0.15
    HEARTBEAT_ALERT_THRESH  = 0.40
    HEARTBEAT_HISTORY_LEN   = 10   # скільки останніх score аналізуємо

    CNN_TEMPORAL_FRAMES     = 3
    # ------------------ 8. ACCIDENT STATE ------------------
    ACCIDENT_LIFETIME = 90 # 3 сек при 30 FPS
