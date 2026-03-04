import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "model" / "weights"

class Config:
    YOLO_PATH = MODELS_DIR / "yolov8-fine-tuning-15.pt"
    CNN_PATH = MODELS_DIR / "accident_cnn_model.pth"
    VIDEO_SOURCE = BASE_DIR / "data" / "video" / "videoplayback.mp4"
    
    # Thresholds
    CONF_YOLO = 0.6
    CONF_ACCIDENT_HIGH = 0.96
    
    # ROI (можна винести в окремий JSON якщо він змінюється від камери до камери)
    ROI_POLYGON = [...]
    
