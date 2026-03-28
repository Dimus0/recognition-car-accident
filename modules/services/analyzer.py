import cv2
import numpy as np
from collections import defaultdict, deque
from typing import List, Tuple, Set, Dict, Optional
from modules.utils.utils import (
    calculate_ttc_advanced,
    get_velocity_smoothed,
    cosine_approach_angle
)
import time
from modules.config.config import Config

class TrafficAnalyzer:
    """
    Аналізатор трафіку з детекцією ДТП і просунутими фільтрами для щільного руху.

    Основні можливості
    ------------------
    - Трекінг швидкості та прискорення кожного об'єкта.
    - Детекція раптових зупинок з відрізненням від заторів.
    - Передбачення зіткнень через AdaptiveTTC + CosineConvergence + DirectionFilter.
    - EMA-згладжування TTC для усунення джиттера детектора.
    - Composite risk score для м'якого гейтування CNN-інференсу.

    Parameters
    ----------
    max_history : int
        Максимальна кількість збережених позицій треку.
    collision_distance : int
        Пікселі, нижче яких об'єкти вважаються дуже близькими.
    min_motion : int
        Мінімум точок в історії для аналізу.
    collision_ttc_threshold : float
        Базовий поріг TTC (кадри або секунди залежно від FPS).
    sudden_stop_threshold : float
        Відносна зміна швидкості для детекції раптової зупинки.
    min_speed_for_stop : float
        Мінімальна швидкість (пікс/кадр) перед зупинкою щоб вважати її раптовою.
    min_cosine_convergence : float
        Мінімальний cosine convergence score щоб позначити пару як небезпечну.
        Діапазон [-1, 1]. Рекомендовано 0.10–0.25 залежно від трафіку.
    same_direction_threshold : float
        Cosine similarity між векторами швидкостей. Вище порогу → попутні авто.
        Рекомендовано 0.80–0.90.
    min_relative_speed : float
        Мінімальна відносна швидкість (пікс/кадр). Запобігає нестабільним TTC.
    ttc_ema_alpha : float
        Коефіцієнт EMA: 0 = повністю старий, 1 = тільки новий. Рекомендовано 0.3–0.5.
    density_radius : int
        Радіус (пікс) зони підрахунку сусідів для оцінки щільності.
    density_divisor : float
        density * density_divisor — divisor для adaptive TTC threshold.
        Вище = більш агресивне зниження чутливості при щільному трафіку.
    composite_risk_min : float
        Мінімальний composite score [0, 1] для включення пари в risky set.
    composite_weights : tuple[float, float, float]
        Ваги (w_ttc, w_cos, w_dist) для composite score. Сума має = 1.0.
    """

    def __init__(
        self,
        max_history: int = 25,
        collision_distance: int = 80,
        min_motion: int = 3,
        collision_ttc_threshold: float = 1.5,
        sudden_stop_threshold: float = Config.SUDDEN_STOP_THRESHOLD,
        min_speed_for_stop: float = Config.MIN_SPEED_FOR_STOP,
        # ── Просунуті фільтри щільного трафіку ──────────────────────────────
        min_cosine_convergence:   float = Config._DEFAULT_MIN_COSINE_CONVERGENCE,
        same_direction_threshold: float = Config._DEFAULT_SAME_DIRECTION_THRESHOLD,
        min_relative_speed:       float = Config._DEFAULT_MIN_RELATIVE_SPEED,
        ttc_ema_alpha:            float = Config._DEFAULT_TTC_EMA_ALPHA,
        density_radius:           int   = Config._DEFAULT_DENSITY_RADIUS,
        density_divisor:          float = Config._DEFAULT_DENSITY_DIVISOR,
        composite_risk_min:       float = Config._DEFAULT_COMPOSITE_RISK_MIN,
        composite_weights:        tuple = Config._DEFAULT_COMPOSITE_WEIGHTS,
    ):
        # ── Базові параметри ─────────────────────────────────────────────────
        self.max_history           = max_history
        self.collision_distance    = collision_distance
        self.min_motion            = min_motion
        self.ttc_threshold         = collision_ttc_threshold
        self.sudden_stop_threshold = sudden_stop_threshold
        self.min_speed_for_stop    = min_speed_for_stop

        # ── Параметри просунутих фільтрів ────────────────────────────────────
        self.min_cosine_convergence   = min_cosine_convergence
        self.same_direction_threshold = same_direction_threshold
        self.min_relative_speed       = min_relative_speed
        self.ttc_ema_alpha            = ttc_ema_alpha
        self.density_radius           = density_radius
        self.density_divisor          = density_divisor
        self.composite_risk_min       = composite_risk_min

        assert abs(sum(composite_weights) - 1.0) < 1e-6, \
            "composite_weights мають давати суму 1.0"
        self.w_ttc, self.w_cos, self.w_dist = composite_weights

        # ── Стан треків ──────────────────────────────────────────────────────
        self.track_history    = defaultdict(lambda: deque(maxlen=max_history))
        self.velocity_history = defaultdict(lambda: deque(maxlen=10))
        self.active_ids       = set()
        self.object_states: Dict = {}

        # ── Кеш EMA для TTC: {pair_key: smoothed_ttc} ───────────────────────
        self._ttc_ema: Dict[tuple, float] = {}

        # ── Cooldown зіткнень ────────────────────────────────────────────────
        self.collision_pairs_history: Dict[tuple, int] = {}
        self.collision_cooldown = 30

        # ── LSTM predicted positions {tid: ndarray(steps, 2)} ──────────────
        # Оновлюється кожен кадр через update_predicted_positions()
        self._lstm_predicted: Dict[int, np.ndarray] = {}

        # ── Стаціонарні авта: лічильник кадрів з малою швидкістю ────────────
        # Дозволяє розрізняти:
        #   • Запарковане/стаціонарне авто (стоїть довго → завжди slow_frames = великий)
        #   • Авто на світлофорі (поступово загальмувало → prev_speed < min_speed_for_stop)
        #   • Аварійна зупинка (різка, після руху → speed_drop > threshold)
        self._slow_frames: Dict[int, int] = {}   # {tid: кадри поспіль зі швидкістю < 2 px/f}
        self.standing_threshold_frames: int = 60  # ~2 с при 30 fps → "стаціонарне авто"

        # ── Метрики ──────────────────────────────────────────────────────────
        self.metrics = {
            'total_frames_processed':    0,
            'total_vehicles_tracked':    set(),
            'collision_warnings':        0,
            'sudden_stops_detected':     0,
            'false_positive_stops':      0,
            'processing_times':          deque(maxlen=100),
            'average_vehicles_per_frame': deque(maxlen=100),
            # Нові лічильники просунутих фільтрів:
            'filtered_by_direction':     0,   # виключено як попутні авто
            'filtered_by_rel_speed':     0,   # виключено через малу відносну швидкість
            'filtered_by_cos_conv':      0,   # виключено через недостатнє зближення
            'filtered_by_composite':     0,   # не пройшли composite threshold
            'adaptive_ttc_adjustments':  0,   # скільки разів поріг знижувався через щільність
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════════

    def update_tracks(self, boxes: List[Tuple], ids: List[int]):
        """
        Оновлює історію треків та обчислює швидкості.

        Parameters
        ----------
        boxes : list of (x1, y1, x2, y2)
        ids   : list of track IDs відповідних боксів
        """
        start_time = time.time()

        self.metrics['total_frames_processed'] += 1
        self.metrics['average_vehicles_per_frame'].append(len(ids))

        for (x1, y1, x2, y2), tid in zip(boxes, ids):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            prev_pos = self.track_history[tid][-1] if self.track_history[tid] else None
            self.track_history[tid].append((cx, cy))

            if prev_pos is not None:
                speed = float(np.linalg.norm([cx - prev_pos[0], cy - prev_pos[1]]))
                self.velocity_history[tid].append(speed)
            else:
                self.velocity_history[tid].append(0.0)

            self._update_object_state(tid)
            self.metrics['total_vehicles_tracked'].add(tid)

        self.metrics['processing_times'].append((time.time() - start_time) * 1000)

    def update_predicted_positions(self, predicted: Dict[int, np.ndarray]):
        """
        Отримує передбачені позиції від LSTM (TrajectoryPredictor.get_predicted_positions).
        Зберігає для використання в predict_collision() як додатковий сигнал ризику.

        Parameters
        ----------
        predicted : {tid: ndarray(steps, 2)}  — майбутні позиції у пікселях
        """
        self._lstm_predicted = predicted if predicted else {}

    def _predict_collision_lstm(
        self,
        ids:          List[int],
        boxes:        List[Tuple],
        centers:      Dict[int, np.ndarray],
        existing_risky: Set[int],
    ) -> Set[int]:
        """
        Перевіряє передбачені траєкторії (з LSTM) на майбутні зіткнення.
        Пари що не пройшли стандартний TTC але мають небезпечні прогнози
        також додаються до risky set.

        Логіка: якщо у будь-якому з перших steps кроків прогнозована відстань
        між двома треками менша за collision_distance → ризик підтверджений.
        """
        lstm_risky: Set[int] = set()
        tids_with_pred = [tid for tid in ids if tid in self._lstm_predicted]

        if len(tids_with_pred) < 2:
            return lstm_risky

        for i in range(len(tids_with_pred)):
            for j in range(i + 1, len(tids_with_pred)):
                tid_a = tids_with_pred[i]
                tid_b = tids_with_pred[j]

                # Вже підтверджені — пропускаємо
                if tid_a in existing_risky and tid_b in existing_risky:
                    continue

                traj_a = self._lstm_predicted[tid_a].astype(float)
                traj_b = self._lstm_predicted[tid_b].astype(float)
                steps  = min(len(traj_a), len(traj_b))
                if steps == 0:
                    continue

                dists    = np.linalg.norm(traj_a[:steps] - traj_b[:steps], axis=1)
                min_dist = float(dists.min())

                # Поріг: 1.2x замість 1.5x — LSTM має бути точнішим.
                # Більший поріг = більше false positives від неточних траєкторій.
                # 1.2 означає авто мають зблизитись до 1.2 * collision_distance
                # у передбаченій траєкторії — досить суворо.
                lstm_thresh = self.collision_distance * 1.2

                if min_dist < lstm_thresh:
                    lstm_risky.add(tid_a)
                    lstm_risky.add(tid_b)
                    self.metrics['collision_warnings'] += 1

        return lstm_risky

    def predict_collision(
        self,
        boxes: List[Tuple] = None,
        ids:   List[int]   = None,
    ) -> Set[int]:
        risky: Set[int] = set()
        current_frame = self.metrics['total_frames_processed']

        if ids is None:
            ids = list(self.active_ids)
        if boxes is None or len(boxes) != len(ids):
            return risky

        # ── 1. Кешування: O(N) перед вкладеними циклами ──────────────────────────
        # Створюємо словники центрів, швидкостей та щільності один раз для всіх авто
        centers = {}
        for i, tid in enumerate(ids):
            box = boxes[i]
            centers[tid] = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

        velocities = {tid: self._get_velocity_vector(tid) for tid in ids}
        densities = {tid: self._compute_local_density(tid, centers) for tid in ids}

        # ── 2. Оцінка ризику: O(N²) з легкими перевірками ────────────────────────
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                pair_key = tuple(sorted([id_a, id_b]))

                # Cooldown
                if pair_key in self.collision_pairs_history:
                    if current_frame - self.collision_pairs_history[pair_key] < self.collision_cooldown:
                        continue

                # Обидва стаціонарні
                if self._is_parked_or_standing(id_a) and self._is_parked_or_standing(id_b):
                    continue

                # Мінімальна історія
                if len(self.track_history.get(id_a, [])) < 5 or len(self.track_history.get(id_b, [])) < 5:
                    continue

                # Беремо закешовані значення (уникаємо перерахунку O(N^3))
                pos_a, pos_b = centers[id_a], centers[id_b]
                vel_a, vel_b = velocities[id_a], velocities[id_b]

                distance = float(np.linalg.norm(pos_a - pos_b))

                # MinRelativeSpeed
                if self._relative_speed_too_low(vel_a, vel_b):
                    self.metrics['filtered_by_rel_speed'] += 1
                    continue

                # DirectionSimilarity
                if self._are_parallel_trajectories(vel_a, vel_b):
                    self.metrics['filtered_by_direction'] += 1
                    continue

                # Попередня перевірка відстані
                if distance >= self.collision_distance * 2.5:
                    continue

                # CosineConvergence + AdaptiveTTC
                raw_ttc, cos_conv = calculate_ttc_advanced(
                    pos_a, vel_a, pos_b, vel_b,
                    min_relative_speed=self.min_relative_speed,
                    max_ttc=self.ttc_threshold * 10,
                )

                if cos_conv < self.min_cosine_convergence and distance >= self.collision_distance:
                    self.metrics['filtered_by_cos_conv'] += 1
                    continue

                # Використовуємо закешовану щільність
                adaptive_thresh = self._adaptive_ttc_threshold(densities[id_a])
                smoothed_ttc = self._get_smoothed_ttc(pair_key, raw_ttc)

                # CompositeRiskScore
                ttc_ok  = smoothed_ttc < adaptive_thresh or distance < self.collision_distance
                dist_ok = distance < self.collision_distance * 1.8

                if not (ttc_ok or dist_ok):
                    continue

                composite = self._composite_risk_score(smoothed_ttc, adaptive_thresh, cos_conv, distance)

                if composite < self.composite_risk_min:
                    self.metrics['filtered_by_composite'] += 1
                    continue

                # Ризик підтверджено
                risky.add(id_a)
                risky.add(id_b)
                self.collision_pairs_history[pair_key] = current_frame
                self.metrics['collision_warnings'] += 1

        # ── 3. LSTM кінематичний фільтр ──────────────────────────────────────────
        if self._lstm_predicted:
            lstm_extra = self._predict_collision_lstm(ids, boxes, centers, risky)
            risky = risky | lstm_extra

        return risky

    def detect_sudden_stop(self, tid: int, boxes: List[Tuple], ids: List[int]) -> bool:
        if tid not in self.velocity_history or len(self.velocity_history[tid]) < 5:
            return False

        velocities = list(self.velocity_history[tid])
        prev_speed = np.mean(velocities[-5:-1]) if len(velocities) >= 5 else 0.0
        curr_speed = velocities[-1]

        # 1. Авто мало достатню швидкість
        if prev_speed < self.min_speed_for_stop:
            return False

        # 2. Різкість падіння швидкості
        speed_drop = (prev_speed - curr_speed) / (prev_speed + 1e-6)
        if speed_drop < self.sudden_stop_threshold:
            return False

        # Знаходимо індекс лише один раз (уникаємо багаторазового O(N))
        try:
            tid_idx = ids.index(tid)
        except ValueError:
            return False

        # Кешуємо центр цільового авто
        tid_box = boxes[tid_idx]
        tid_center = np.array([(tid_box[0] + tid_box[2]) / 2, (tid_box[1] + tid_box[3]) / 2])

        # 3 & 4. Об'єднана логіка: шукаємо сусідів та перевіряємо затор за один прохід
        JAM_RADIUS = 200
        CONTACT_RADIUS = 100
        stopped_nearby = 0
        has_nearby_vehicle = False

        for i, other_id in enumerate(ids):
            if other_id == tid:
                continue
                
            other_box = boxes[i]
            other_center = np.array([(other_box[0] + other_box[2]) / 2, (other_box[1] + other_box[3]) / 2])
            dist = float(np.linalg.norm(tid_center - other_center))

            if dist < CONTACT_RADIUS:
                has_nearby_vehicle = True

            if dist < JAM_RADIUS and self.object_states.get(other_id, {}).get('stopped', False):
                stopped_nearby += 1

        # Перевірка на затор (замінює виклик _is_traffic_jam)
        if stopped_nearby >= 3: 
            self.metrics['false_positive_stops'] += 1
            return False

        if has_nearby_vehicle:
            self.metrics['sudden_stops_detected'] += 1
            return True

        # 5. Надзвичайно різка зупинка без сусіда
        if speed_drop > 0.6:
            self.metrics['sudden_stops_detected'] += 1
            return True

        return False

    def clean_old_tracks(self, current_ids: List[int]):
        """
        Очищає стан для треків що вийшли з кадру.
        Також прибирає EMA-кеш для пар де один з учасників зник.
        """
        removed = set(self.track_history.keys()) - set(current_ids)
        for tid in removed:
            self.track_history.pop(tid, None)
            self.velocity_history.pop(tid, None)
            self.object_states.pop(tid, None)
            self.active_ids.discard(tid)
            self._slow_frames.pop(tid, None)   # очищаємо лічильник стаціонарності

        # Очистка EMA-кешу для зниклих треків
        dead_pairs = [
            pk for pk in list(self._ttc_ema.keys())
            if pk[0] not in current_ids or pk[1] not in current_ids
        ]
        for pk in dead_pairs:
            del self._ttc_ema[pk]

    def get_metrics_summary(self) -> Dict:
        """
        Повертає зведення по метриках аналізатора,
        включаючи лічильники просунутих фільтрів.
        """
        avg_proc = (
            float(np.mean(self.metrics['processing_times']))
            if self.metrics['processing_times'] else 0.0
        )
        avg_veh = (
            float(np.mean(self.metrics['average_vehicles_per_frame']))
            if self.metrics['average_vehicles_per_frame'] else 0.0
        )

        return {
            'total_frames':            self.metrics['total_frames_processed'],
            'total_unique_vehicles':   len(self.metrics['total_vehicles_tracked']),
            'collision_warnings':      self.metrics['collision_warnings'],
            'sudden_stops_detected':   self.metrics['sudden_stops_detected'],
            'false_positive_stops':    self.metrics['false_positive_stops'],
            'avg_processing_time_ms':  round(avg_proc, 2),
            'avg_vehicles_per_frame':  round(avg_veh, 2),
            'fps':                     round(1000 / avg_proc, 2) if avg_proc > 0 else 0,
            # Просунуті фільтри
            'filtered_by_direction':   self.metrics['filtered_by_direction'],
            'filtered_by_rel_speed':   self.metrics['filtered_by_rel_speed'],
            'filtered_by_cos_conv':    self.metrics['filtered_by_cos_conv'],
            'filtered_by_composite':   self.metrics['filtered_by_composite'],
            'adaptive_ttc_adjustments': self.metrics['adaptive_ttc_adjustments'],
        }

    def reset_metrics(self):
        """Скидає метрики (для нового відео)."""
        self.metrics = {
            'total_frames_processed':    0,
            'total_vehicles_tracked':    set(),
            'collision_warnings':        0,
            'sudden_stops_detected':     0,
            'false_positive_stops':      0,
            'processing_times':          deque(maxlen=100),
            'average_vehicles_per_frame': deque(maxlen=100),
            'filtered_by_direction':     0,
            'filtered_by_rel_speed':     0,
            'filtered_by_cos_conv':      0,
            'filtered_by_composite':     0,
            'adaptive_ttc_adjustments':  0,
        }
        self.collision_pairs_history.clear()
        self._ttc_ema.clear()
        self._slow_frames.clear()

    # ═══════════════════════════════════════════════════════════════════════════
    # ADVANCED FILTERS — PRIVATE METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    # _cosine_convergence було видалено:
    # тепер cosine convergence (approach_cos) повертає calculate_ttc_advanced()
    # з utils.py разом з TTC за один виклик -- без дублювання логіки.


    def _is_parked_or_standing(self, tid: int) -> bool:
        """
        True якщо авто стоїть без руху більш ніж standing_threshold_frames кадрів.

        Відрізняє три стани:
          • Запарковане авто     → стоїть сотні кадрів → True
          • Авто на світлофорі   → стоїть 30-90 кадрів → True (поки чекає)
          • Щойно зупинилось     → slow_frames = 0-5   → False (свіжа зупинка)

        Використання: виключати пари де ОБИДВА стаціонарні з predict_collision —
        вони не можуть самостійно зіткнутись без зовнішньої причини.
        Якщо одне авто рухається → пара залишається в аналізі.
        """
        return self._slow_frames.get(tid, 0) >= self.standing_threshold_frames

    def _direction_similarity(
        self,
        vel_a: np.ndarray,
        vel_b: np.ndarray,
    ) -> float:
        """
        Косинусна подібність між векторами швидкостей (напрямок руху).

        Описує наскільки схожі напрямки руху двох транспортних засобів.
        Не пов'язаний з позицією — тільки кут між векторами швидкостей.

        Returns
        -------
        float
            [-1, 1].  +1 = одна сторона, -1 = назустріч, 0 = перпендикулярно.
        """
        na = float(np.linalg.norm(vel_a))
        nb = float(np.linalg.norm(vel_b))
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.clip(np.dot(vel_a, vel_b) / (na * nb), -1.0, 1.0))

    def _are_parallel_trajectories(
        self,
        vel_a: np.ndarray,
        vel_b: np.ndarray,
    ) -> bool:
        """
        Перевіряє, чи два авто рухаються практично в один бік (паралельні смуги).

        Якщо cos(vel_A, vel_B) > same_direction_threshold, вони — «попутні сусіди»
        і не становлять прямого ризику зіткнення незалежно від відстані.

        Типові сценарії де спрацьовує:
          • Два авто в сусідніх смугах на магістралі.
          • Колона авто з однаковою швидкістю.
          • Затор де всі поступово рушають вперед.

        Returns
        -------
        bool
            True → попутні (виключити з ризику).
        """
        return self._direction_similarity(vel_a, vel_b) > self.same_direction_threshold

    def _relative_speed_too_low(
        self,
        vel_a: np.ndarray,
        vel_b: np.ndarray,
    ) -> bool:
        """
        Перевіряє, чи відносна швидкість між парою замала для розрахунку TTC.

        При надто малій відносній швидкості (<<1 піксель/кадр):
          - TTC → нескінченність або дуже нестабільний через шум bbox.
          - У заторі всі авто майже стоять → цей фільтр прибирає весь затор.

        Returns
        -------
        bool
            True → відносна швидкість замала, пару пропускаємо.
        """
        return float(np.linalg.norm(vel_a - vel_b)) < self.min_relative_speed

    def _compute_local_density(
        self,
        tid: int,
        centers: Dict[int, np.ndarray],
    ) -> int:
        """
        Підраховує кількість транспортних засобів у радіусі density_radius навколо tid.

        Використовується для адаптивного масштабування порогу TTC:
        висока щільність → менш агресивні попередження.

        Parameters
        ----------
        tid     : track ID цільового авто
        centers : {tid: np.array([cx, cy])} поточного кадру

        Returns
        -------
        int
            Кількість сусідів (без самого tid).
        """
        if tid not in centers:
            return 0
        pos  = centers[tid]
        return sum(
            1
            for other_tid, other_pos in centers.items()
            if other_tid != tid
            and float(np.linalg.norm(pos - other_pos)) < self.density_radius
        )

    def _adaptive_ttc_threshold(self, local_density: int) -> float:
        """
        Обчислює адаптивний поріг TTC залежно від локальної щільності трафіку.

        Логіка
        ------
        У щільному трафіку багато авто постійно близько одне до одного,
        і стандартний TTC постійно < порогу → хибні тривоги.
        Тому при зростанні щільності знижуємо поріг (підвищуємо планку).

        Формула:
            divisor        = max(1.0, density * density_divisor)
            adaptive_thresh = base_thresh / divisor

        При density=0  → adaptive_thresh = base_thresh (без змін)
        При density=4  → adaptive_thresh = base_thresh / (4*0.25) = base_thresh / 1.0
        При density=8  → adaptive_thresh = base_thresh / 2.0  (вдвічі суворіше)

        Parameters
        ----------
        local_density : int
            Кількість сусідів в зоні density_radius.

        Returns
        -------
        float
            Адаптивний поріг TTC.
        """
        divisor = max(1.0, local_density * self.density_divisor)
        if divisor > 1.0:
            self.metrics['adaptive_ttc_adjustments'] += 1
        return self.ttc_threshold / divisor

    def _get_smoothed_ttc(self, pair_key: tuple, raw_ttc: float) -> float:
        """
        Повертає EMA-згладжений TTC для пари.

        EMA (Exponential Moving Average):
            ttc_smooth[k] = α * ttc_raw[k] + (1-α) * ttc_smooth[k-1]

        де α = ttc_ema_alpha (default 0.4).

        Переваги
        --------
        - Усуває одиночні різкі стрибки TTC через bbox-джиттер.
        - Забезпечує темпоральну когерентність сигналу ризику.
        - При α=0.4: ~60% ваги на 2 останніх значення, решта — на минуле.

        Parameters
        ----------
        pair_key : tuple
            Відсортований tuple (id_a, id_b).
        raw_ttc  : float
            Щойно обчислений TTC (може бути inf).

        Returns
        -------
        float
            Згладжений TTC.
        """
        clipped_ttc = min(raw_ttc, self.ttc_threshold * 10)  # обрізаємо inf

        if pair_key not in self._ttc_ema:
            self._ttc_ema[pair_key] = clipped_ttc
        else:
            α = self.ttc_ema_alpha
            self._ttc_ema[pair_key] = α * clipped_ttc + (1 - α) * self._ttc_ema[pair_key]

        return self._ttc_ema[pair_key]

    def _composite_risk_score(
        self,
        smoothed_ttc:    float,
        adaptive_thresh: float,
        cos_conv:        float,
        distance:        float,
    ) -> float:
        """
        Обчислює composite risk score [0, 1] для пари транспортних засобів.

        Три компоненти (ваги задаються через composite_weights)
        --------------------------------------------------------
        1. TTC-компонент  (w_ttc):
               ttc_score = clip(1 - ttc / adaptive_thresh, 0, 1)
               При ttc=0 → 1.0 (найгірше), при ttc≥thresh → 0.0.

        2. Convergence-компонент  (w_cos):
               cos_score = clip(cos_conv, 0, 1)
               +1 = пряме зіткнення, 0 = перпендикулярне або розбіжне.

        3. Distance-компонент  (w_dist):
               dist_score = clip(1 - dist / collision_distance, 0, 1)
               При dist=0 → 1.0 (перекриття), при dist≥collision_dist → 0.0.

        Підсумок:
               composite = w_ttc * ttc_score + w_cos * cos_score + w_dist * dist_score

        Parameters
        ----------
        smoothed_ttc    : EMA-згладжений TTC.
        adaptive_thresh : адаптивний поріг TTC.
        cos_conv        : cosine convergence в [-1, 1].
        distance        : відстань між центроїдами боксів (пікс).

        Returns
        -------
        float
            Composite risk score в [0, 1]. Вище = більш небезпечно.
        """
        ttc_score  = float(np.clip(1.0 - smoothed_ttc / (adaptive_thresh + 1e-9), 0.0, 1.0))
        cos_score  = float(np.clip(cos_conv, 0.0, 1.0))
        dist_score = float(np.clip(1.0 - distance / (self.collision_distance + 1e-9), 0.0, 1.0))

        return self.w_ttc * ttc_score + self.w_cos * cos_score + self.w_dist * dist_score

    # ═══════════════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _update_object_state(self, tid: int):
        """Оновлює стан об'єкта (швидкість, прискорення, зупинка)."""
        if len(self.velocity_history[tid]) < 3:
            return

        velocities    = list(self.velocity_history[tid])
        current_speed = velocities[-1]
        avg_speed     = np.mean(velocities[-5:]) if len(velocities) >= 5 else current_speed
        acceleration  = velocities[-1] - velocities[-2] if len(velocities) >= 2 else 0.0
        is_stopped    = current_speed < 1.0 and avg_speed < 2.0

        # ── Лічильник стаціонарності ─────────────────────────────────────────
        # Збільшуємо якщо авто майже не рухається, скидаємо при русі.
        # Використовується для розрізнення «стоїть давно» vs «щойно зупинилось».
        if current_speed < 2.0:
            self._slow_frames[tid] = self._slow_frames.get(tid, 0) + 1
        else:
            self._slow_frames[tid] = 0

        self.object_states[tid] = {
            'speed':           current_speed,
            'avg_speed':       avg_speed,
            'acceleration':    acceleration,
            'stopped':         is_stopped,
            'standing_frames': self._slow_frames.get(tid, 0),
        }

    def _get_velocity_vector(self, tid: int, smooth_window: int = 3) -> np.ndarray:
        """
        Тонка обгортка навколо get_velocity_smoothed з utils.py.

        Делегує логіку утиліті — без дублювання коду.
        smooth_window=3 → ковзне середнє по 3 останнім крокам.
        """
        return get_velocity_smoothed(self.track_history, tid, smooth_window)

    def _calculate_box_distance(self, box1: Tuple, box2: Tuple) -> float:
        """Евклідова відстань між центрами боксів."""
        c1 = np.array([(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2])
        c2 = np.array([(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2])
        return float(np.linalg.norm(c1 - c2))

    def _is_traffic_jam(
        self,
        tid: int,
        boxes: List[Tuple],
        ids: List[int],
        radius: int = 200,
        stopped_threshold: int = 3,
    ) -> bool:
        """
        Перевіряє чи об'єкт знаходиться в заторі.

        Логіка: якщо ≥ stopped_threshold авто зупинені в радіусі radius → затор.
        """
        if tid not in ids:
            return False

        tid_idx = ids.index(tid)
        tid_pos = np.array([
            (boxes[tid_idx][0] + boxes[tid_idx][2]) / 2,
            (boxes[tid_idx][1] + boxes[tid_idx][3]) / 2,
        ])
        stopped_nearby = 0

        for i, other_id in enumerate(ids):
            if other_id == tid:
                continue
            other_pos = np.array([
                (boxes[i][0] + boxes[i][2]) / 2,
                (boxes[i][1] + boxes[i][3]) / 2,
            ])
            if (float(np.linalg.norm(tid_pos - other_pos)) < radius
                    and self.object_states.get(other_id, {}).get('stopped', False)):
                stopped_nearby += 1

        return stopped_nearby >= stopped_threshold