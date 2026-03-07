import cv2
import numpy as np
import torch
from model.src.cnn import ImproveAccidentCNN
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque, defaultdict
from sklearn.preprocessing import StandardScaler
from model.src.lstm import MotionLSTM
from modules.config.config import Config
import pickle
import logging
import os
from typing import Tuple, Optional

logger = logging.getLogger("accident_detector")


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
        max_age=45,              # ЗБІЛЬШЕНО: Чекаємо 45 кадрів (1.5 сек), якщо YOLO загубив авто, перш ніж вбити трек
        n_init=4,                # ЗБІЛЬШЕНО: Об'єкт має бути впевнено знайдений 4 кадри поспіль, щоб відсіяти "фантоми"
        max_iou_distance=0.7,    # Залишаємо стандарт
        max_cosine_distance=0.3, # Залишаємо (відповідає за порівняння візуальної схожості)
        nn_budget=20,            # Залишаємо (скільки попередніх кадрів авто пам'ятає мережа)
        embedder="mobilenet",    # ДОДАНО (Опціонально): Вказує DeepSort використовувати легку нейромережу для розрізнення машин за виглядом
        half=True                # ДОДАНО: Прискорює роботу на GPU
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
    batch = torch.stack(tensors).to(Config.DEVICE)
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
                 obs_len:     int   = Config.LSTM_OBS_LEN,
                 pred_len:    int   = Config.LSTM_PRED_LEN,
                 coord_scale: float = Config.LSTM_COORD_SCALE):
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
        threshold_px: float = Config.LSTM_COLLISION_PX,
        check_steps:  int   = Config.LSTM_COLLISION_FRAMES,
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
        model = MotionLSTM(input_size=4, hidden=Config.LSTM_HIDDEN,
                           layers=Config.LSTM_LAYERS, pred_len=Config.LSTM_PRED_LEN,
                           drop=Config.LSTM_DROPOUT)
        state_dict = torch.load(weights_path, map_location=device)["model_state"]

        model.load_state_dict(state_dict)
        model.eval()

        with open(scaler_path, "rb") as f:
            scaler: StandardScaler = pickle.load(f)

        predictor = TrajectoryPredictor(
            model=model, scaler=scaler, device=device,
            obs_len=Config.LSTM_OBS_LEN, pred_len=Config.LSTM_PRED_LEN,
            coord_scale=Config.LSTM_COORD_SCALE
        )
        logger.info(f"[LSTM] MotionLSTM завантажено | obs={Config.LSTM_OBS_LEN} pred={Config.LSTM_PRED_LEN}")
        print(f"  ✅ MotionLSTM завантажено (obs={Config.LSTM_OBS_LEN}, pred={Config.LSTM_PRED_LEN})")
        return predictor

    except Exception as e:
        logger.error(f"[LSTM] Помилка: {e}", exc_info=True)
        print(f"  ❌ LSTM помилка: {e}")
        return None
    


def calculate_ttc_advanced(
    pos_a: np.ndarray,
    vel_a: np.ndarray,
    pos_b: np.ndarray,
    vel_b: np.ndarray,
    min_relative_speed: float = 0.8,
    max_ttc: float = 30.0,
) -> Tuple[float, float]:
    """
    Просунутий розрахунок TTC (Time To Collision) з перевіркою зближення.

    Відрізняється від стандартного ``calculate_ttc`` тим що:
      1. Безпечно обробляє нульові швидкості (немає ділення на нуль).
      2. Повертає cosine convergence разом із TTC.
      3. Обмежує максимальне повернуте значення (max_ttc) — без нескінченностей.
      4. Фільтрує пари з надто малою відносною швидкістю (заторна ситуація).

    Алгоритм
    ---------
    1. rel_pos = pos_B - pos_A                       (вектор розділення)
    2. rel_vel = vel_A - vel_B                       (відносна швидкість)
    3. unit_sep = rel_pos / |rel_pos|                (одиничний вектор)
    4. approach_cos = dot(rel_vel, unit_sep) / |rel_vel|
       > 0 → зближуються, < 0 → розбігаються, 0 → перпендикулярно
    5. closing_speed = dot(rel_vel, unit_sep)        (проекція на вісь розділення)
    6. ttc = |rel_pos| / closing_speed               (тільки якщо > 0)

    Parameters
    ----------
    pos_a, pos_b        : np.array([x, y]) — центроїди боксів (пікселі).
    vel_a, vel_b        : np.array([vx, vy]) — вектори швидкості (пікс/кадр).
    min_relative_speed  : float — мінімальна |rel_vel| для розрахунку TTC.
                          Нижче = пара у затоостані, повертаємо (max_ttc, 0.0).
    max_ttc             : float — обмеження max TTC (замість inf).

    Returns
    -------
    (ttc, approach_cos) : tuple[float, float]
        ttc         — час до зіткнення (кадри). max_ttc якщо розбігаються.
        approach_cos — косинус між rel_vel і осю розділення [-1, 1].
                       Дозволяє зовнішньому коду вирішити чи TTC має сенс.
    """
    rel_pos = pos_b - pos_a
    rel_vel = vel_a - vel_b

    dist     = float(np.linalg.norm(rel_pos))
    rel_spd  = float(np.linalg.norm(rel_vel))

    # Запобігання нестабільному TTC у заторі
    if rel_spd < min_relative_speed:
        return max_ttc, 0.0

    if dist < 1e-9:
        return 0.0, 1.0  # Об'єкти вже збіглись

    unit_sep     = rel_pos / dist
    approach_cos = float(np.clip(np.dot(rel_vel, unit_sep) / rel_spd, -1.0, 1.0))
    closing_spd  = float(np.dot(rel_vel, unit_sep))  # проекція, може бути < 0

    if closing_spd <= 0.0:
        # Розбігаються або паралельний рух
        return max_ttc, approach_cos

    ttc = min(dist / closing_spd, max_ttc)
    return float(ttc), approach_cos

def cosine_approach_angle(
    vel_a: np.ndarray,
    vel_b: np.ndarray,
    pos_a: np.ndarray,
    pos_b: np.ndarray,
) -> float:
    """
    Повертає кут (в градусах) між відносною швидкістю та вектором розділення.

    Призначення
    -----------
    Зручна функція для логування та налагодження. Дає інтуїтивно зрозумілий
    числовий показник замість косинуса.

    Значення
    ---------
      0°   → пряме зіткнення лоб-в-лоб
     90°   → авто рухається перпендикулярно (ковзний удар)
    180°   → розбіжність (авто розбігаються)

    Parameters
    ----------
    vel_a, vel_b : np.array([vx, vy]) — вектори швидкостей.
    pos_a, pos_b : np.array([x, y]) — поточні позиції (для вектора розділення).

    Returns
    -------
    float
        Кут у градусах [0, 180].
    """
    rel_vel = vel_a - vel_b
    rel_pos = pos_b - pos_a

    nrv = float(np.linalg.norm(rel_vel))
    nrp = float(np.linalg.norm(rel_pos))

    if nrv < 1e-9 or nrp < 1e-9:
        return 90.0  # Невизначено → нейтральний кут

    cos_val = float(np.clip(np.dot(rel_vel, rel_pos) / (nrv * nrp), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_val)))

def get_velocity_smoothed(track_history: dict,tid: int,smooth_window: int = 3,) -> np.ndarray:
    """
    Обчислює вектор швидкості зі ковзним середнім по track_history.

    Стандартний підхід (остання позиція - передостання) чутливий до bbox-джиттеру.
    Ковзне середнє по ``smooth_window`` крокам дає стабільніший вектор.

    Формула
    -------
    vel = mean( pos[k] - pos[k-1] для k в останніх smooth_window кроках )

    Parameters
    ----------
    track_history  : dict {tid: deque([(x,y), ...])}
    tid            : track ID
    smooth_window  : кількість кроків для усереднення (default 3).

    Returns
    -------
    np.ndarray
        Вектор [vx, vy] у пікс/кадр. [0,0] якщо недостатньо точок.
    """
    pts = list(track_history.get(tid, []))
    if len(pts) < 2:
        return np.array([0.0, 0.0])

    n  = min(smooth_window, len(pts) - 1)
    deltas = [
        np.array(pts[-(k)]) - np.array(pts[-(k + 1)])
        for k in range(1, n + 1)
    ]
    return np.mean(deltas, axis=0).astype(np.float32)

def build_accident_description(accident_objects, frame_count, fps, camera_id="", camera_location=""):
    """
    Приймає сирі дані про ДТП і формує детальний словник та текст для диспетчера.
    """
    primary_objects  = [o for o in accident_objects if o['type'] == 'primary']
    secondary_objects = [o for o in accident_objects if o['type'] == 'secondary']
    all_confidences   = [o['confidence'] for o in accident_objects]
    
    max_confidence = max(all_confidences) if all_confidences else 0.0
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    
    # Визначаємо рівень небезпеки
    if max_confidence > 0.85:
        severity = "CRITICAL"
        severity_ua = "КРИТИЧНИЙ"
    elif max_confidence > 0.70:
        severity = "HIGH"
        severity_ua = "ВИСОКИЙ"
    else:
        severity = "MEDIUM"
        severity_ua = "СЕРЕДНІЙ"

    # Рахуємо час на відео
    minutes = int(frame_count / fps // 60)
    seconds = int(frame_count / fps % 60)
    timestamp_formatted = f"{minutes:02d}:{seconds:02d}"

    # Формуємо красивий текст для Telegram
    dispatcher_summary = (
        f"⚠️ <b>ЗАФІКСОВАНО ДТП</b>\n\n"
        f"🚗 ТЗ задіяно: {len(accident_objects)} "
        f"({len(primary_objects)} осн., {len(secondary_objects)} поруч)\n"
        f"🚨 Небезпека: <b>{severity_ua}</b>\n"
        f"🎯 Впевненість ШІ: {max_confidence:.0%}\n"
        f"⏱ Час на відео: {timestamp_formatted}\n"
        f"📷 Камера: {camera_id or 'Не вказано'}\n"
        f"📍 Локація: {camera_location or 'Не вказано'}"
    )

    # Збираємо все в один словник
    return {
        "vehicles_total": len(accident_objects),
        "vehicles_primary": len(primary_objects),
        "vehicles_secondary": len(secondary_objects),
        "involved_track_ids": [o['track_id'] for o in accident_objects],
        "max_confidence": round(max_confidence, 3),
        "avg_confidence": round(avg_confidence, 3),
        "severity": severity,
        "frame_number": frame_count,
        "timestamp_sec": round(frame_count / fps, 2),
        "timestamp_formatted": timestamp_formatted,
        "camera_id": camera_id,
        "camera_location": camera_location,
        "gps_coordinates": "",
        "dispatcher_summary": dispatcher_summary # Готовий текст для повідомлення
    }

