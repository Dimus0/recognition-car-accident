from collections import defaultdict, deque
from modules.config.config import Config


class AccidentStateTracker:
    """
    Клас для відстеження стану аварій та уникнення помилкових спрацювань.
    """

    def __init__(self):
        # Історія CNN скорів для кожного track_id
        self.cnn_score_history = defaultdict(lambda: deque(maxlen=10))

        # Підтверджені аварії (track_id -> frame_number коли виявлено)
        self.confirmed_accidents = {}

        # Час життя підтвердженої аварії (кількість кадрів)
        self.accident_lifetime = Config.ACCIDENT_LIFETIME

        # Мінімальна кількість високих скорів для підтвердження
        self.confirmation_threshold = 2

    def update_score(self, track_id: int, score: float, frame_number: int):
        """Оновлює історію скорів для track_id"""
        self.cnn_score_history[track_id].append(score)

    def is_confirmed_accident(self, track_id: int) -> bool:
        """Перевіряє чи є підтверджена аварія для цього track_id"""
        return track_id in self.confirmed_accidents

    def should_confirm_accident(
        self,
        track_id:      int,
        lstm_risk:     float = 0.0,
        is_kinematic:  bool  = False,
        is_sudden:     bool  = False,
    ) -> bool:
        """
        Вирішує чи підтверджувати аварію для track_id.
        """
        history = list(self.cnn_score_history[track_id])
        if not history: return False

        latest_score = history[-1]

        if lstm_risk >= 0.88 and latest_score > 0.74:
            return True
        
        if latest_score > 0.92 and (is_sudden or is_kinematic or lstm_risk > 0.5):
            return True

        # ── Визначаємо мінімальну кількість кадрів для підтвердження ─────────
        min_frames = 10 if (is_sudden and not is_kinematic) else self.confirmation_threshold
        if len(history) < min_frames:
            return False

        recent_scores = history[-min_frames:]
        avg_score = sum(recent_scores) / len(recent_scores)
        min_score = min(recent_scores)

        # ── Визначаємо базові пороги згідно з матрицею (виправлено значення) ──
        if is_sudden and not is_kinematic:
            avg_thresh = 0.93
            min_thresh = 0.90
        else:
            avg_thresh = 0.75
            min_thresh = 0.65

        # ── LSTM знижує поріг для БУДЬ-ЯКОГО сценарію (додано втрачену логіку) ──
        if lstm_risk >= 0.5:
            reduction = (lstm_risk - 0.5) * 0.30
            avg_thresh -= reduction
            min_thresh -= reduction

        return avg_score > avg_thresh and min_score > min_thresh

    def confirm_accident(self, track_id: int, frame_number: int):
        """Підтверджує аварію для track_id"""
        self.confirmed_accidents[track_id] = frame_number

    def is_accident_active(self, track_id: int, current_frame: int) -> bool:
        """Перевіряє чи аварія все ще активна"""
        if track_id not in self.confirmed_accidents:
            return False
        return (current_frame - self.confirmed_accidents[track_id]) < self.accident_lifetime

    def cleanup_old_accidents(self, current_frame: int):
        """Видаляє старі аварії"""
        # Створюємо список ключів для видалення, щоб не змінювати словник під час ітерації
        to_remove = [
            tid for tid, frame in self.confirmed_accidents.items()
            if (current_frame - frame) >= self.accident_lifetime
        ]
        for tid in to_remove:
            del self.confirmed_accidents[tid]