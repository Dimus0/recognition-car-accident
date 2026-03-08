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
        track_id:    int,
        current_score: float,
        lstm_risk:   float = 0.0,
    ) -> bool:
        """
        Визначає чи потрібно підтвердити аварію на основі історії CNN-скорів.
        Якщо LSTM також вказує на ризик — пороги знижуються, бо дві незалежні
        моделі погоджуються між собою.

        Адаптивні пороги залежно від lstm_risk
        ----------------------------------------
        lstm_risk = 0.0  → avg > 0.85, min > 0.75  (базовий, суворий)
        lstm_risk = 0.5  → avg > 0.78, min > 0.68
        lstm_risk = 0.8  → avg > 0.72, min > 0.62
        lstm_risk ≥ 1.0  → avg > 0.68, min > 0.58  (максимальне зниження)
        """
        history = list(self.cnn_score_history[track_id])

        if len(history) < self.confirmation_threshold:
            return False

        recent_scores = history[-self.confirmation_threshold:]
        avg_score = sum(recent_scores) / len(recent_scores)
        min_score = min(recent_scores)

        # LSTM-aware пороги: кожен 0.1 ризику знижує поріг на 0.017 / 0.017
        # Максимальне зниження обмежено щоб уникнути хибних спрацьовувань
        reduction   = min(lstm_risk, 1.0) * 0.17
        avg_thresh  = 0.85 - reduction
        min_thresh  = 0.75 - reduction

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