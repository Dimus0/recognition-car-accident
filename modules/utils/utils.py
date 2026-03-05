import cv2
import numpy as np
import torch
from model.src.cnn import ImproveAccidentCNN
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque, defaultdict
from sklearn.preprocessing import StandardScaler
from model.src.lstm import MotionLSTM
import pickle
import logging
import os

logger = logging.getLogger("accident_detector")

LSTM_OBS_LEN         = 20      # кадрів спостереження
LSTM_PRED_LEN        = 30      # кадрів прогнозу
LSTM_HIDDEN          = 128
LSTM_LAYERS          = 2
LSTM_DROPOUT         = 0.3
LSTM_COORD_SCALE     = 10.0     # ділення координат при тренуванні
LSTM_COLLISION_PX    = 80       # Поріг зближення траєкторій (в пікселях після зворотного масштабування)
LSTM_COLLISION_FRAMES= 10       # Скільки майбутніх кроків прогнозу перевіряти на зближення

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
        max_cosine_distance=0.3,
        nn_budget=20,
    )

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


def get_interaction_crop(box1, box2, frame, padding=20):
    x1 = min(box1[0], box2[0]) - padding
    y1 = min(box1[1], box2[1]) - padding
    x2 = max(box1[2], box2[2]) + padding
    y2 = max(box1[3], box2[3]) + padding
    
    # Clamping coordinates
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    return frame[y1:y2, x1:x2]


def calculate_ttc(pos_a, vel_a, pos_b, vel_b):
    """
    pos: np.array([x, y]) - центроїди
    vel: np.array([vx, vy]) - вектори швидкості
    """
    rel_pos = pos_b - pos_a
    rel_vel = vel_b - vel_a
    
    # Проекція відносної швидкості на вектор відстані
    speed_rel_towards = -np.dot(rel_pos, rel_vel) / np.linalg.norm(rel_pos)
    
    if speed_rel_towards <= 0:
        return float('inf') # Об'єкти віддаляються
        
    distance = np.linalg.norm(rel_pos)
    ttc = distance / speed_rel_towards
    return ttc

class TrajectoryPredictor:
    """
    Накопичує останні LSTM_OBS_LEN кадрів [x, y, vx, vy] для кожного
    треку, масштабує через той самий StandardScaler що при тренуванні,
    запускає MotionLSTM і повертає прогноз у пікселях відеокадру.

    Препроцесинг повторює ArgoverseDataset:
        1) obs[:, 0:2] /= coord_scale   (як tgt /= 10.0 при тренуванні)
        2) scaler.transform(...)         (StandardScaler fitted на train)
    Постпроцесинг:
        pred_unscaled = pred * std_xy + mean_xy
        pred_px       = pred_unscaled * coord_scale
    """

    def __init__(self, model: MotionLSTM, scaler: StandardScaler,
                 device: torch.device,
                 obs_len:     int   = LSTM_OBS_LEN,
                 pred_len:    int   = LSTM_PRED_LEN,
                 coord_scale: float = LSTM_COORD_SCALE):
        self.model       = model.to(device).eval()
        self.scaler      = scaler
        self.device      = device
        self.obs_len     = obs_len
        self.pred_len    = pred_len
        self.coord_scale = coord_scale

        # Кеш параметрів scaler для зворотного масштабування прогнозу
        self._mean_xy = np.array(scaler.mean_[:2],                dtype=np.float32)
        self._std_xy  = np.array(np.sqrt(scaler.var_[:2]),        dtype=np.float32)

        # {tid: deque([x_raw, y_raw, vx_raw, vy_raw])}
        self._buffers: dict = defaultdict(lambda: deque(maxlen=obs_len))
        # {tid: np.ndarray (pred_len, 2) у пікселях}
        self.predictions: dict = {}

    # ── Оновлення буфера ──────────────────────────────────────

    def update_from_history(self, tid: int, history: list):
        """
        Оновлює буфер треку через список (x,y) центрів з TrafficAnalyzer.
        Швидкість = різниця двох останніх позицій (пікселі/кадр).
        """
        if not history:
            return
        x,  y  = float(history[-1][0]), float(history[-1][1])
        if len(history) >= 2:
            vx = x - float(history[-2][0])
            vy = y - float(history[-2][1])
        else:
            vx, vy = 0.0, 0.0
        self._buffers[tid].append([x, y, vx, vy])

    # ── Батчевий інференс ─────────────────────────────────────

    @torch.no_grad()
    def run_batch(self, active_ids: list) -> dict:
        """
        Запускає MotionLSTM для треків із повним obs_len буфером.
        Повертає {tid: ndarray(pred_len, 2)} у пікселях.
        """
        ready_ids = [tid for tid in active_ids
                     if len(self._buffers[tid]) == self.obs_len]
        if not ready_ids:
            return {}

        raw_seqs = []
        for tid in ready_ids:
            seq = np.array(self._buffers[tid], dtype=np.float32)  # (obs_len, 4)

            # Повторюємо ArgoverseDataset pre-processing:
            # Центруємо відносно останньої спостережуваної позиції
            # (як x0, y0 = agent[obs_len-1] при тренуванні)
            x0, y0 = seq[-1, 0], seq[-1, 1]
            seq[:, 0] -= x0
            seq[:, 1] -= y0
            seq[:, 2] -= 0.0   # vx вже відносний
            seq[:, 3] -= 0.0

            # Масштабування /10 (як при тренуванні)
            seq /= self.coord_scale
            raw_seqs.append((seq, x0, y0))

        batch_np = np.stack([s[0] for s in raw_seqs], axis=0)   # (B, obs_len, 4)
        B, T, F  = batch_np.shape

        # StandardScaler: трансформуємо кожний кадр
        flat   = batch_np.reshape(-1, F)
        flat   = self.scaler.transform(flat)
        tensor = torch.tensor(flat.reshape(B, T, F),
                              dtype=torch.float32).to(self.device)

        pred_scaled = self.model(tensor, tf_ratio=0.)   # (B, pred_len, 2) in scaled space

        # Зворотне масштабування:
        mean_t = torch.tensor(self._mean_xy, device=self.device)
        std_t  = torch.tensor(self._std_xy,  device=self.device)
        pred_unscaled = (pred_scaled * std_t + mean_t).cpu().numpy()  # /10 space
        pred_abs      = pred_unscaled * self.coord_scale               # пікселі, відносно x0,y0

        result = {}
        for i, (tid, (_, x0, y0)) in enumerate(zip(ready_ids, raw_seqs)):
            # Переводимо з відносних координат назад у абсолютні пікселі
            pred_abs_tid          = pred_abs[i].copy()
            pred_abs_tid[:, 0]   += x0
            pred_abs_tid[:, 1]   += y0
            self.predictions[tid] = pred_abs_tid.astype(np.int32)
            result[tid]           = self.predictions[tid]

        return result

    # ── Аналіз зіткнень між прогнозованими траєкторіями ───────

    def get_collision_risk_pairs(
        self,
        active_ids:   list,
        threshold_px: float = LSTM_COLLISION_PX,
        check_steps:  int   = LSTM_COLLISION_FRAMES,
    ) -> dict:
        """
        Перевіряє всі пари треків на зближення у перших check_steps
        кроках прогнозу. Повертає {tid: risk ∈ [0,1]}.

        Логіка ризику:
            risk = 1.0 - min_dist / threshold_px
            При min_dist=0   → risk=1.0 (пряме зіткнення)
            При min_dist≥thr → risk=0.0 (безпечно)
        """
        risk: dict = {}
        tids_pred = [tid for tid in active_ids if tid in self.predictions]
        if len(tids_pred) < 2:
            return risk

        for i in range(len(tids_pred)):
            for j in range(i + 1, len(tids_pred)):
                tid_i = tids_pred[i]
                tid_j = tids_pred[j]
                traj_i = self.predictions[tid_i][:check_steps].astype(float)
                traj_j = self.predictions[tid_j][:check_steps].astype(float)
                dists  = np.linalg.norm(traj_i - traj_j, axis=1)
                min_d  = float(dists.min())
                if min_d < threshold_px:
                    score = max(0.0, 1.0 - min_d / threshold_px)
                    risk[tid_i] = max(risk.get(tid_i, 0.0), score)
                    risk[tid_j] = max(risk.get(tid_j, 0.0), score)
        return risk

    # ── Візуалізація прогнозів ─────────────────────────────────

    def draw_predictions(self, frame: np.ndarray, active_ids: list,
                         risk_ids: set, steps: int = 15) -> np.ndarray:
        """
        Малює прогнозовані траєкторії:
            • червоні лінії  — треки з ризиком зіткнення
            • жовті пунктири — безпечні треки
        """
        for tid in active_ids:
            if tid not in self.predictions:
                continue
            traj  = self.predictions[tid][:steps]
            color = (0, 0, 220) if tid in risk_ids else (0, 200, 255)
            lw    = 2           if tid in risk_ids else 1
            for k in range(1, len(traj)):
                cv2.line(frame, tuple(traj[k - 1]), tuple(traj[k]),
                         color, lw, cv2.LINE_AA)
            if len(traj):
                cv2.circle(frame, tuple(traj[-1]), 5, color, -1)
        return frame

    # ── Очистка мертвих треків ─────────────────────────────────

    def cleanup(self, active_ids: list):
        dead = [tid for tid in list(self._buffers) if tid not in active_ids]
        for tid in dead:
            del self._buffers[tid]
            self.predictions.pop(tid, None)


def load_motion_lstm(weights_path: str, scaler_path: str,
                     device: torch.device):
    """
    Завантажує MotionLSTM і StandardScaler.
    Якщо файли відсутні → повертає None (LSTM-модуль вимкнено).

    ── Як зберегти після тренування ──────────────────────────────
    # В ltsm.ipynb, після main():
    torch.save(model.state_dict(), r"...\\motion_lstm.pth")
    import pickle
    with open(r"...\\motion_lstm_scaler.pkl", "wb") as f:
        pickle.dump(ds.scaler, f)   # ds = ArgoverseDataset (з fit_scaler=True)
    """
    missing = [p for p in (weights_path, scaler_path) if not os.path.exists(p)]
    if missing:
        for p in missing:
            logger.warning(f"[LSTM] Файл не знайдено: {p}")
            print(f"  ⚠️  LSTM вимкнено — не знайдено: {p}")
        return None

    try:
        model = MotionLSTM(input_size=4, hidden=LSTM_HIDDEN,
                           layers=LSTM_LAYERS, pred_len=LSTM_PRED_LEN,
                           drop=LSTM_DROPOUT)
        state_dict = torch.load(weights_path, map_location=device)["model_state"]

        model.load_state_dict(state_dict)
        model.eval()

        with open(scaler_path, "rb") as f:
            scaler: StandardScaler = pickle.load(f)

        predictor = TrajectoryPredictor(
            model=model, scaler=scaler, device=device,
            obs_len=LSTM_OBS_LEN, pred_len=LSTM_PRED_LEN,
            coord_scale=LSTM_COORD_SCALE
        )
        logger.info(f"[LSTM] MotionLSTM завантажено | obs={LSTM_OBS_LEN} pred={LSTM_PRED_LEN}")
        print(f"  ✅ MotionLSTM завантажено (obs={LSTM_OBS_LEN}, pred={LSTM_PRED_LEN})")
        return predictor

    except Exception as e:
        logger.error(f"[LSTM] Помилка: {e}", exc_info=True)
        print(f"  ❌ LSTM помилка: {e}")
        return None