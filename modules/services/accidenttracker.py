import cv2
import json
import os
from datetime import datetime
from typing import List, Tuple, Dict,Union
from collections import defaultdict, deque
from modules.config.config import Config


class AccidentStateTracker:
    """
    Клас для відстеження стану аварій та уникнення помилкових спрацювань
    """
    def __init__(self):
        # Історія CNN скорів для кожного track_id
        self.cnn_score_history = defaultdict(lambda: deque(maxlen=10))
        
        # Підтверджені аварії (track_id -> frame_number коли виявлено)
        self.confirmed_accidents = {}
        
        # Час життя підтвердженої аварії (кількість кадрів)
        self.accident_lifetime = Config.ACCIDENT_LIFETIME  # ~3 секунди при 30 FPS
        
        # Мінімальна кількість високих скорів для підтвердження
        self.confirmation_threshold = 3
        
    def update_score(self, track_id: int, score: float, frame_number: int):
        """Оновлює історію скорів для track_id"""
        self.cnn_score_history[track_id].append(score)
        
    def is_confirmed_accident(self, track_id: int) -> bool:
        """Перевіряє чи є підтверджена аварія для цього track_id"""
        if track_id not in self.confirmed_accidents:
            return False
        return True
    
    def should_confirm_accident(
        self,
        track_id:      int,
        current_score: float,
        lstm_risk:     float = 0.0,
        is_kinematic:  bool  = False,
    ) -> bool:
        """
        Визначає чи потрібно підтвердити аварію на основі CNN-скор-історії.

        ПРИНЦИП: Кінематика (TTC/cosine/distance) — PRIMARY gate.
        LSTM — лише МОДИФІКАТОР порогів коли кінематика ВЖЕ погодилась.

        Логіка порогів:
        - is_kinematic=False: суворі пороги avg>0.88, min>0.78. LSTM не знижує.
        - is_kinematic=True, lstm_risk>=0.5: зниження до -0.12 (обидві моделі згодні)
        - is_kinematic=True, lstm_risk<0.5: avg>0.85, min>0.75 (тільки кінематика)
        """
        history = list(self.cnn_score_history[track_id])

        if len(history) < self.confirmation_threshold:
            return False

        recent_scores = history[-self.confirmation_threshold:]
        avg_score = sum(recent_scores) / len(recent_scores)
        min_score = min(recent_scores)

        if not is_kinematic:
            # Без кінематики CNN мусить бути впевнений самостійно.
            # LSTM не знижує пороги — без TTC/cosine підтвердження
            # LSTM alone не може підтвердити аварію.
            avg_thresh = 0.88
            min_thresh = 0.78
        elif lstm_risk >= 0.5:
            # Обидві моделі погодились: кінематика + LSTM
            reduction  = min(lstm_risk - 0.5, 0.5) * 0.24  # max -0.12
            avg_thresh = 0.88 - reduction
            min_thresh = 0.78 - reduction
        else:
            # Тільки кінематика, LSTM не впевнений
            avg_thresh = 0.85
            min_thresh = 0.75

        return avg_score > avg_thresh and min_score > min_thresh
    
    def confirm_accident(self, track_id: int, frame_number: int):
        """Підтверджує аварію для track_id"""
        self.confirmed_accidents[track_id] = frame_number
        
    def is_accident_active(self, track_id: int, current_frame: int) -> bool:
        """Перевіряє чи аварія все ще активна"""
        if track_id not in self.confirmed_accidents:
            return False
        
        accident_frame = self.confirmed_accidents[track_id]
        return (current_frame - accident_frame) < self.accident_lifetime
    
    def cleanup_old_accidents(self, current_frame: int):
        """Видаляє старі аварії"""
        to_remove = [
            tid for tid, frame in self.confirmed_accidents.items()
            if (current_frame - frame) >= self.accident_lifetime
        ]
        for tid in to_remove:
            del self.confirmed_accidents[tid]