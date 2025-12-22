import cv2
import json
import os
from datetime import datetime
from typing import List, Tuple, Dict
from collections import defaultdict, deque

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
        self.accident_lifetime = 90  # ~3 секунди при 30 FPS
        
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
    
    def should_confirm_accident(self, track_id: int, current_score: float) -> bool:
        """
        Визначає чи потрібно підтвердити аварію на основі історії скорів
        Логіка: не один високий скор, а стабільно високі значення
        """
        history = list(self.cnn_score_history[track_id])
        
        if len(history) < self.confirmation_threshold:
            return False
        
        # Беремо останні N скорів
        recent_scores = history[-self.confirmation_threshold:]
        
        # Підтверджуємо якщо:
        # 1. Середній скор > 0.85
        # 2. Мінімальний скор > 0.75
        avg_score = sum(recent_scores) / len(recent_scores)
        min_score = min(recent_scores)
        
        return avg_score > 0.85 and min_score > 0.75
    
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
