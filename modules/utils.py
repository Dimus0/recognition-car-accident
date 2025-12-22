import cv2
import numpy as np
import torch
from torchvision import transforms
from collections import defaultdict, deque
from model.src.cnn import ImproveAccidentCNN
from modules.analyzer import TrafficAnalyzer
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)


def load_models(DEVICE,CNN_WEIGHTS_PATH,YOLO_MODEL_PATH):
    cnn = ImproveAccidentCNN().to(DEVICE)
    cnn.load_state_dict(torch.load(CNN_WEIGHTS_PATH, map_location=DEVICE))
    cnn.eval()

    yolo = YOLO(YOLO_MODEL_PATH)
    yolo.to(DEVICE)

    deepsort = DeepSort(
        max_age=15, # скількитрек живе без детекції
        n_init=2,
        max_iou_distance=0.4,
        max_cosine_distance=0.2,
        nn_budget=20,
    )

    # logger.info("Models loaded successfully.")
    return cnn, yolo,deepsort


def process_cnn_batch(crops,cnn_transforms,cnn_model):
    if not crops:
        return []
    tensors = []
    for crop in crops:
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            t = cnn_transforms(rgb)
            tensors.append(t)
        except:
            tensors.append(torch.zeros(3, 224, 224))
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        out = cnn_model(batch)
        if out.shape[1] == 1:
            return torch.sigmoid(out).cpu().numpy().flatten().tolist()
        else:
            return torch.softmax(out, dim=1)[:, 1].cpu().numpy().tolist()
        

def is_inside_roi(bbox, polygon):
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    return cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0



# add 19.12.2025
def get_box_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def is_point_in_polygon(point, polygon):
    # point має бути (x, y), polygon - np.array
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def check_kinematic_anomalies(track_history, tid, current_box):
    """
    Перевіряє, чи об'єкт різко зупинився або поводиться дивно.
    Повертає True, якщо поведінка підозріла.
    """
    history = track_history.get(tid, [])
    if len(history) < 5:
        return False
    
    # Останні 5 точок
    recent_points = [history[len(history) - i] for i in range(1, 6)]
    recent_points.reverse()
    
    # Розрахунок середнього переміщення (швидкості)
    displacements = []
    for i in range(len(recent_points) - 1):
        p1 = np.array(recent_points[i])
        p2 = np.array(recent_points[i+1])
        dist = np.linalg.norm(p2 - p1)
        displacements.append(dist)
    
    avg_speed = np.mean(displacements)
    
    # ЕВРИСТИКА: Якщо швидкість дуже мала (машина стала посеред дороги), але це не край кадру
    # Поріг швидкості треба підбирати під відео (тут умовно < 2 пікселів за кадр)
    if avg_speed < 2.0: 
        return True
        
    return False


def calculate_box_distance(box1, box2):
    """Обчислює відстань між центрами двох боксів"""
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    
    return np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)


def count_nearby_vehicles(boxes, current_idx, distance_threshold=150):
    """
    Підраховує кількість авто поблизу від поточного
    """
    current_box = boxes[current_idx]
    cx = (current_box[0] + current_box[2]) / 2
    cy = (current_box[1] + current_box[3]) / 2
    
    nearby_count = 0
    
    for i, box in enumerate(boxes):
        if i == current_idx:
            continue
        
        bx = (box[0] + box[2]) / 2
        by = (box[1] + box[3]) / 2
        
        distance = np.sqrt((cx - bx)**2 + (cy - by)**2)
        
        if distance < distance_threshold:
            nearby_count += 1
    
    return nearby_count