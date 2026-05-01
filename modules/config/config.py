import os
import torch

class Config:
    """Глобальні налаштування проекту"""

    ABLATION_MODE: str = "full"

    ABLATION_CONFIGS: dict = {
        'baseline':  'YOLOv8 + DeepSORT + Distance trigger (80 px) — no TTC / LSTM',
        'kinematic': 'YOLOv8 + DeepSORT + TTC + Cosine Similarity — no LSTM',
        'lstm_only': 'YOLOv8 + DeepSORT + MotionLSTM predictor — no TTC kinematic',
        'full':      'YOLOv8 + DeepSORT + LSTM + TTC + ResNet50 cascade (default)',
    }


    # ------------------ 1. BASE PATHS ------------------
    BASE_DIR = r"E:\personalproject\bachelor"
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    OUTPUT_DIR = os.path.join(LOG_DIR, "fragments")
    OUTPUT_DIR_CLIP = os.path.join(OUTPUT_DIR, "clip")
    METRICS_FILE = os.path.join(LOG_DIR, "metrics_summary.json")
    ARTIFACTS_PATH = os.path.join(LOG_DIR,"artifacts")
    
    # GROUND_TRUTH_PATH         = r"E:\personalproject\bachelor\logs\gt\gt_accident_video_v1.json"
    # GROUND_TRUTH_PATH         = r"E:\personalproject\bachelor\logs\gt\gt_accident_video_v3.json" # FAVORITE
    # GROUND_TRUTH_PATH         = r"E:\personalproject\bachelor\logs\gt\gt_accident_video_v6.json"
    # GROUND_TRUTH_PATH         = r"E:\personalproject\bachelor\logs\gt\gt_accident_video_v7.json" # FAVORITE
    # GROUND_TRUTH_PATH         = r"E:\personalproject\bachelor\logs\gt\gt_accident_video_v8.json" # test
    GROUND_TRUTH_PATH         = r"E:\personalproject\bachelor\logs\gt\gt_accident_video_v10.json"
    # GROUND_TRUTH_PATH         = r"E:\personalproject\bachelor\logs\gt\gt_noaccident_video_v1.json"
    # GROUND_TRUTH_PATH         = r"E:\personalproject\bachelor\logs\gt\gt_noaccident_video_v2.json"
    # GROUND_TRUTH_PATH         = r"E:\personalproject\bachelor\logs\gt\gt_noaccident_video_v3.json"

    # VIDEO_PATH                = r"E:\personalproject\bachelor\data\video\accident_video_v1.mp4"
    # VIDEO_PATH                = r"E:\personalproject\bachelor\data\video\accident_video_v3.mp4" # FAVORITE
    # VIDEO_PATH                = r"E:\personalproject\bachelor\data\video\accident_video_v6.mp4"
    # VIDEO_PATH                = r"E:\personalproject\bachelor\data\video\accident_video_v7.mp4" # FAVORITE
    # VIDEO_PATH                = r"E:\personalproject\bachelor\data\video\accident_video_v8.mp4" # test
    VIDEO_PATH                = r"E:\personalproject\bachelor\data\video\accident_video_v10.mp4"
    # VIDEO_PATH                = r"E:\personalproject\bachelor\data\video\noaccident_video_v1.mp4"
    # VIDEO_PATH                = r"E:\personalproject\bachelor\data\video\noaccident_video_v2.mp4"
    # VIDEO_PATH                = r"E:\personalproject\bachelor\data\video\noaccident_video_v3.mp4"

    # ------------------ 2. MODELS ------------------
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    YOLO_MODEL_PATH = os.path.join(BASE_DIR, "model", "weights", "yolov8-50epochs.pt")
    CLASSIFIER_WEIGHTS_PATH = os.path.join(BASE_DIR, "model", "weights", "resnet50_accident.pth")
    LSTM_MODEL_PATH = os.path.join(BASE_DIR, "model", "weights", "accident_lstm_model.pt")
    LSTM_SCALER_PATH = os.path.join(BASE_DIR, "model", "weights", "motion_lstm_scaler.pkl")
    # ------------------ 3. DETECTION ------------------
    CONF_YOLO               = 0.45
    IOU_YOLO                = 0.40
    CONF_ACCIDENT_HIGH      = 0.85   # Поріг точного ДТП
    CONF_ACCIDENT_LOW       = 0.65    # Поріг попередження
    HEARTBEAT_RATE          = 7         # Як часто перевіряти авто без підозр # було 10
    VIDEO_BUFFER_SECONDS    = 4.0  # Буфер до/після аварії               # було 2

    # ------------------ 3. DeepSort ------------------

    DEEPSORT_MAX_AGE             = 60 
    DEEPSORT_N_INIT              = 2    
    DEEPSORT_NN_BUDGET           = 100
    DEEPSORT_MAX_IOU_DISTANCE    = 0.7 
    DEEPSORT_MAX_COSINE_DISTANCE = 0.4 

    # ------------------ 4. MOTION LSTM ------------------
    LSTM_OBS_LEN            = 20
    LSTM_PRED_LEN           = 30
    LSTM_HIDDEN             = 128
    LSTM_LAYERS             = 2
    LSTM_DROPOUT            = 0.3
    LSTM_COORD_SCALE        = 10.0
    LSTM_COLLISION_PX       = 80
    LSTM_COLLISION_FRAMES   = 10
    LSTM_ACCIDENT_THRESH    = 0.62 # 0.70
    LSTM_HIGH_CONFIDENCE_THRESH = 0.85
    LSTM_THRESHOLD_REDUCTION_K  = 0.60
    LSTM_MIN_CNN_THRESH         = 0.72
    LSTM_DIRECT_OVERRIDE        = 0.92
    LSTM_DIRECT_CNN_MIN         = 0.68


    TRIGGER_LSTM_THRESH     = 0.65   # LSTM поріг для тригера
    TRIGGER_KIN_LSTM_JOINT  = 0.30   # поріг LSTM якщо є кінематика
    CONFIRMATION_WINDOW_SEC = 3.0    # вікно підтвердження
    CONFIRMATION_CNN_AVG    = 0.88   # середній CNN-скор для підтвердження
    CONFIRMATION_MIN_CROPS  = 10     # мінімум вимірювань в вікні

    # --------------------- ResNet50 ------------------

    RESNET50_NUM_CLASSES  = 2
    RESNET50_DROPOUT      = 0.2
    RESNET50_ACCIDENT_IDX = 0
    CNN_CONTEXT_PADDING = 125
    CNN_SAVE_CROP_THRESH = 0.90    
    RESNET_CROPS_DIR     = os.path.join(LOG_DIR, "resnet_crops")
    # ------------------ 5. TRAFFIC ANALYZER ------------------
    ANALYZER_MAX_HISTORY        = 25
    ANALYZER_COLLISION_DIST     = 100 # 80
    COLLISION_TTC_THRESHOLD     = 2.0 # 1.5
    # Кінематичні пороги (фінальні, без дублювань)
    SUDDEN_STOP_THRESHOLD       = 0.4
    MIN_SPEED_FOR_STOP          = 12.0

    # ------------------ 6. ANALYZER (кінематика) ------------------
    # Рекомендовані значення для мінімізації false-positives
    _DEFAULT_MIN_COSINE_CONVERGENCE   = 0.15   # нижче → ігноруємо як розбіжну пару
    _DEFAULT_SAME_DIRECTION_THRESHOLD = 0.85   # вище → попутні авто, не ризик
    _DEFAULT_MIN_RELATIVE_SPEED       = 0.5    # пікс/кадр; нижче → не рахуємо TTC
    _DEFAULT_TTC_EMA_ALPHA            = 0.35   # вага нового значення в EMA
    _DEFAULT_DENSITY_RADIUS           = 200    # пікс для підрахунку сусідів
    _DEFAULT_DENSITY_DIVISOR          = 0.30   # знижує поріг при щільному русі
    _DEFAULT_COMPOSITE_RISK_MIN       = 0.28   # трохи суворіший поріг ризику
    _DEFAULT_COMPOSITE_WEIGHTS        = (0.45, 0.25, 0.30)  # (ttc, cos, dist)


    # ------------------ 7. ACCIDENT STATE ------------------
    ACCIDENT_LIFETIME = 90 # 3 сек при 30 FPS