import cv2
import numpy as np
from collections import defaultdict, deque
from typing import List, Tuple, Set, Dict, Optional
import time


class TrafficAnalyzer:
    """
    Покращений аналізатор трафіку з детекцією ДТП та метриками
    """
    
    def __init__(
        self, 
        max_history: int = 25,
        collision_distance: int = 80,
        min_motion: int = 3,
        collision_ttc_threshold: float = 1.5,
        sudden_stop_threshold: float = 0.3,  # Відносна зміна швидкості
        min_speed_for_stop: float = 5.0  # Мінімальна швидкість перед зупинкою
    ):
        # Основні параметри
        self.max_history = max_history
        self.collision_distance = collision_distance
        self.min_motion = min_motion
        self.ttc_threshold = collision_ttc_threshold
        self.sudden_stop_threshold = sudden_stop_threshold
        self.min_speed_for_stop = min_speed_for_stop
        
        # Історія треків
        self.track_history = defaultdict(lambda: deque(maxlen=max_history))
        self.velocity_history = defaultdict(lambda: deque(maxlen=10))
        self.active_ids = set()
        
        # Стан об'єктів
        self.object_states = {}  # {track_id: {'speed', 'acceleration', 'stopped'}}
        
        # Історія зіткнень для уникнення дублювання
        self.collision_pairs_history = {}  # {(id1, id2): last_frame}
        self.collision_cooldown = 30  # кадрів
        
        # Метрики
        self.metrics = {
            'total_frames_processed': 0,
            'total_vehicles_tracked': set(),
            'collision_warnings': 0,
            'sudden_stops_detected': 0,
            'false_positive_stops': 0,  # Зупинки в заторі
            'processing_times': deque(maxlen=100),
            'average_vehicles_per_frame': deque(maxlen=100)
        }
    
    def update_tracks(self, boxes: List[Tuple], ids: List[int]):
        """
        Оновлює історію треків та обчислює швидкості
        """
        start_time = time.time()
        
        self.metrics['total_frames_processed'] += 1
        self.metrics['average_vehicles_per_frame'].append(len(ids))
        
        for (x1, y1, x2, y2), tid in zip(boxes, ids):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Оновлюємо історію позицій
            prev_pos = self.track_history[tid][-1] if self.track_history[tid] else None
            self.track_history[tid].append((cx, cy))
            
            # Обчислюємо швидкість
            if prev_pos is not None:
                velocity = np.array([cx - prev_pos[0], cy - prev_pos[1]])
                speed = np.linalg.norm(velocity)
                self.velocity_history[tid].append(speed)
            else:
                self.velocity_history[tid].append(0.0)
            
            # Оновлюємо стан об'єкта
            self._update_object_state(tid)
            
            # Додаємо до метрик
            self.metrics['total_vehicles_tracked'].add(tid)
        
        processing_time = (time.time() - start_time) * 1000  # мс
        self.metrics['processing_times'].append(processing_time)
    
    def _update_object_state(self, tid: int):
        """
        Оновлює стан об'єкта (швидкість, прискорення, зупинка)
        """
        if len(self.velocity_history[tid]) < 3:
            return
        
        velocities = list(self.velocity_history[tid])
        current_speed = velocities[-1]
        avg_speed = np.mean(velocities[-5:]) if len(velocities) >= 5 else current_speed
        
        # Обчислюємо прискорення (зміну швидкості)
        if len(velocities) >= 2:
            acceleration = velocities[-1] - velocities[-2]
        else:
            acceleration = 0.0
        
        # Визначаємо чи об'єкт зупинився
        is_stopped = current_speed < 1.0 and avg_speed < 2.0
        
        self.object_states[tid] = {
            'speed': current_speed,
            'avg_speed': avg_speed,
            'acceleration': acceleration,
            'stopped': is_stopped
        }
    
    def detect_sudden_stop(self, tid: int, boxes: List[Tuple], ids: List[int]) -> bool:
        """
        Детектує раптову зупинку що може вказувати на ДТП
        
        КРИТИЧНА ЛОГІКА: Перевіряємо чи це ДТП, а не затор
        """
        if tid not in self.velocity_history or len(self.velocity_history[tid]) < 5:
            return False
        
        velocities = list(self.velocity_history[tid])
        
        # Швидкість ДО зупинки
        prev_speed = np.mean(velocities[-5:-1]) if len(velocities) >= 5 else 0
        current_speed = velocities[-1]
        
        # 1. Перевірка: чи була достатня швидкість перед зупинкою
        if prev_speed < self.min_speed_for_stop:
            return False  # Авто і так їхало повільно
        
        # 2. Перевірка: чи була раптова зміна швидкості
        speed_drop = (prev_speed - current_speed) / (prev_speed + 1e-6)
        if speed_drop < self.sudden_stop_threshold:
            return False  # Зупинка занадто плавна
        
        # 3. КРИТИЧНО: Перевіряємо чи є інші авто ДУЖЕ близько (фізичний контакт)
        tid_idx = ids.index(tid) if tid in ids else -1
        if tid_idx == -1:
            return False
        
        tid_box = boxes[tid_idx]
        has_close_contact = False
        
        for i, other_id in enumerate(ids):
            if other_id == tid:
                continue
            
            other_box = boxes[i]
            distance = self._calculate_box_distance(tid_box, other_box)
            
            # Дуже близька відстань (менше 50 пікселів = можливий контакт)
            if distance < 50:
                # Перевіряємо чи інше авто теж різко зупинилося
                if other_id in self.object_states:
                    other_speed = self.object_states[other_id].get('avg_speed', 100)
                    if other_speed < 2.0:  # Інше авто теж зупинилося
                        has_close_contact = True
                        break
        
        # 4. Перевіряємо чи навколо є ЗАТОР (багато зупинених авто)
        if self._is_traffic_jam(tid, boxes, ids):
            self.metrics['false_positive_stops'] += 1
            return False  # Це затор, а не ДТП
        
        if has_close_contact:
            self.metrics['sudden_stops_detected'] += 1
            return True
        
        return False
    
    def _is_traffic_jam(self, tid: int, boxes: List[Tuple], ids: List[int], 
                       radius: int = 200, stopped_threshold: int = 3) -> bool:
        """
        Перевіряє чи об'єкт знаходиться в заторі
        
        Логіка: Якщо навколо об'єкта багато інших зупинених авто - це затор
        """
        if tid not in ids:
            return False
        
        tid_idx = ids.index(tid)
        tid_box = boxes[tid_idx]
        tid_pos = np.array([
            (tid_box[0] + tid_box[2]) / 2,
            (tid_box[1] + tid_box[3]) / 2
        ])
        
        stopped_nearby = 0
        
        for i, other_id in enumerate(ids):
            if other_id == tid:
                continue
            
            other_box = boxes[i]
            other_pos = np.array([
                (other_box[0] + other_box[2]) / 2,
                (other_box[1] + other_box[3]) / 2
            ])
            
            distance = np.linalg.norm(tid_pos - other_pos)
            
            if distance < radius:
                # Перевіряємо чи інше авто зупинилося
                if other_id in self.object_states:
                    if self.object_states[other_id].get('stopped', False):
                        stopped_nearby += 1
        
        # Якщо 3+ авто зупинилися поруч - це затор
        return stopped_nearby >= stopped_threshold
    
    def predict_collision(
        self, 
        boxes: List[Tuple] = None, 
        ids: List[int] = None
    ) -> Set[int]:
        """
        Передбачає можливі зіткнення на основі TTC (Time To Collision)
        
        Покращена логіка: враховує швидкість та напрямок руху
        """
        risky = set()
        current_frame = self.metrics['total_frames_processed']
        
        if ids is None:
            ids = list(self.active_ids)
        
        if boxes is None or len(boxes) != len(ids):
            return risky
        
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                
                # Перевіряємо чи не було недавно детекції для цієї пари
                pair_key = tuple(sorted([id_a, id_b]))
                if pair_key in self.collision_pairs_history:
                    last_frame = self.collision_pairs_history[pair_key]
                    if current_frame - last_frame < self.collision_cooldown:
                        continue  # Пропускаємо - недавно вже було
                
                # Перевіряємо наявність історії
                if (id_a not in self.track_history or 
                    id_b not in self.track_history or
                    len(self.track_history[id_a]) < 5 or 
                    len(self.track_history[id_b]) < 5):
                    continue
                
                # Поточні позиції та швидкості
                pos_a = np.array(self.track_history[id_a][-1])
                pos_b = np.array(self.track_history[id_b][-1])
                
                # Обчислюємо вектори швидкості
                vel_a = self._get_velocity_vector(id_a)
                vel_b = self._get_velocity_vector(id_b)
                
                # Відстань між об'єктами
                distance = np.linalg.norm(pos_a - pos_b)
                
                # Перевірка 1: Чи вони дуже близько (можливий контакт)
                if distance < self.collision_distance:
                    # Перевірка 2: Чи вони рухаються один до одного
                    rel_pos = pos_b - pos_a
                    rel_vel = vel_b - vel_a
                    
                    # Проекція швидкості на вектор відстані
                    if np.linalg.norm(rel_pos) > 0:
                        closing_speed = -np.dot(rel_pos, rel_vel) / np.linalg.norm(rel_pos)
                        
                        # Якщо зближуються (closing_speed > 0) або дуже близько
                        if closing_speed > 0 or distance < 60:
                            # Обчислюємо TTC
                            if closing_speed > 0.1:
                                ttc = distance / closing_speed
                                if ttc < self.ttc_threshold:
                                    risky.add(id_a)
                                    risky.add(id_b)
                                    self.collision_pairs_history[pair_key] = current_frame
                                    self.metrics['collision_warnings'] += 1
                            elif distance < 60:  # Дуже близько незалежно від швидкості
                                risky.add(id_a)
                                risky.add(id_b)
                                self.collision_pairs_history[pair_key] = current_frame
        
        return risky
    
    def _get_velocity_vector(self, tid: int) -> np.ndarray:
        """
        Обчислює вектор швидкості об'єкта
        """
        if tid not in self.track_history or len(self.track_history[tid]) < 2:
            return np.array([0.0, 0.0])
        
        points = list(self.track_history[tid])
        
        # Беремо останні 3 точки для згладжування
        if len(points) >= 3:
            velocity = np.array(points[-1]) - np.array(points[-3])
            velocity = velocity / 3.0  # Нормалізуємо на кількість кадрів
        else:
            velocity = np.array(points[-1]) - np.array(points[-2])
        
        return velocity
    
    def _calculate_box_distance(self, box1: Tuple, box2: Tuple) -> float:
        """
        Обчислює відстань між центрами боксів
        """
        center1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
        center2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
        return np.linalg.norm(center1 - center2)
    
    def clean_old_tracks(self, current_ids: List[int]):
        """
        Очищає старі треки
        """
        for tid in list(self.track_history.keys()):
            if tid not in current_ids:
                del self.track_history[tid]
                del self.velocity_history[tid]
                if tid in self.object_states:
                    del self.object_states[tid]
                self.active_ids.discard(tid)
    
    def get_metrics_summary(self) -> Dict:
        """
        Повертає зведення по метриках
        """
        avg_processing_time = (
            np.mean(self.metrics['processing_times']) 
            if self.metrics['processing_times'] else 0
        )
        
        avg_vehicles = (
            np.mean(self.metrics['average_vehicles_per_frame'])
            if self.metrics['average_vehicles_per_frame'] else 0
        )
        
        return {
            'total_frames': self.metrics['total_frames_processed'],
            'total_unique_vehicles': len(self.metrics['total_vehicles_tracked']),
            'collision_warnings': self.metrics['collision_warnings'],
            'sudden_stops_detected': self.metrics['sudden_stops_detected'],
            'false_positive_stops': self.metrics['false_positive_stops'],
            'avg_processing_time_ms': round(avg_processing_time, 2),
            'avg_vehicles_per_frame': round(avg_vehicles, 2),
            'fps': round(1000 / avg_processing_time, 2) if avg_processing_time > 0 else 0
        }
    
    def reset_metrics(self):
        """
        Скидає метрики (для нового відео)
        """
        self.metrics = {
            'total_frames_processed': 0,
            'total_vehicles_tracked': set(),
            'collision_warnings': 0,
            'sudden_stops_detected': 0,
            'false_positive_stops': 0,
            'processing_times': deque(maxlen=100),
            'average_vehicles_per_frame': deque(maxlen=100)
        }
        self.collision_pairs_history.clear()
    
    def point_to_line_distance(self, px, py, x1, y1, x2, y2):
        """
        Обчислює відстань від точки до лінії
        """
        A = px - x1
        B = py - y1
        C = x2 - x1
        D = y2 - y1
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        
        if len_sq == 0:
            return np.hypot(px - x1, py - y1)
        
        param = dot / len_sq
        
        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D
        
        return np.hypot(px - xx, py - yy)
    
    def is_moving_towards_camera(self, tid: int) -> bool:
        """
        Перевіряє чи об'єкт рухається до камери
        """
        pts = self.track_history.get(tid, [])
        if len(pts) < self.min_motion:
            return False
        
        y_start = pts[0][1]
        y_end = pts[-1][1]
        
        return (y_end - y_start) > 10