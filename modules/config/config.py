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


    # VIDEO_PATH                = r"E:\personalproject\bachelor\data\video\videoplayback.mp4"
    VIDEO_PATH              = r"E:\personalproject\bachelor\data\video\highway_traffic.mp4"
    # VIDEO_PATH              = r"E:\personalproject\bachelor\data\video\traffic.mp4"

    # ------------------ 2. MODELS ------------------
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    YOLO_MODEL_PATH = os.path.join(BASE_DIR, "model", "weights", "yolov8-fine-tuning.pt")
    CNN_WEIGHTS_PATH = os.path.join(BASE_DIR, "model", "weights", "accident_cnn_model.pth")
    LSTM_MODEL_PATH = os.path.join(BASE_DIR, "model", "weights", "accident_lstm_model.pt")
    LSTM_SCALER_PATH = os.path.join(BASE_DIR, "model", "weights", "motion_lstm_scaler.pkl")

    # ------------------ 3. DETECTION & CNN ------------------
    CONF_YOLO = 0.55
    CONF_ACCIDENT_HIGH = 0.92   # Поріг точного ДТП
    CONF_ACCIDENT_LOW = 0.86    # Поріг попередження
    HEARTBEAT_RATE = 10         # Як часто перевіряти авто без підозр
    VIDEO_BUFFER_SECONDS = 2.0  # Буфер до/після аварії

    # ------------------ 4. MOTION LSTM ------------------
    LSTM_OBS_LEN = 20
    LSTM_PRED_LEN = 30
    LSTM_HIDDEN = 128
    LSTM_LAYERS = 2
    LSTM_DROPOUT = 0.3
    LSTM_COORD_SCALE = 10.0
    LSTM_COLLISION_PX = 80
    LSTM_COLLISION_FRAMES = 10
    LSTM_ACCIDENT_THRESH = 0.70

    # ------------------ 5. TRAFFIC ANALYZER ------------------
    ANALYZER_MAX_HISTORY = 25
    ANALYZER_COLLISION_DIST = 80
    COLLISION_TTC_THRESHOLD = 1.5
    SUDDEN_STOP_THRESHOLD = 0.19
    MIN_SPEED_FOR_STOP = 5.0


    # ------------------ 6. ANALYZER ------------------
    _DEFAULT_MIN_COSINE_CONVERGENCE   = 0.15   # нижче → ігноруємо як розбіжну пару
    _DEFAULT_SAME_DIRECTION_THRESHOLD = 0.85   # вище → попутні авто, не ризик
    _DEFAULT_MIN_RELATIVE_SPEED       = 0.8    # пікс/кадр; нижче → не рахуємо TTC
    _DEFAULT_TTC_EMA_ALPHA            = 0.40   # вага нового значення в EMA
    _DEFAULT_DENSITY_RADIUS           = 220    # пікс для підрахунку сусідів
    _DEFAULT_DENSITY_DIVISOR          = 0.25   # знижує поріг при щільному русі
    _DEFAULT_COMPOSITE_RISK_MIN       = 0.35   # мінімальний composite score
    _DEFAULT_COMPOSITE_WEIGHTS        = (0.50, 0.30, 0.20)  # (ttc, cos, dist)


    # ------------------ 6. ANALYZER ------------------
    ACCIDENT_LIFETIME = 90 # 3 сек при 30 FPS

    # ------------------ 6. REGION OF INTEREST (ROI) ------------------
    # Область для відео із шосе
    ROI_POLYGON_HIGHWAY = np.array([
        (931, 1074), 
        (120, 956), 
        (816, 591), 
        (1025, 597)
    ], dtype=np.int32)