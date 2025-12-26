import cv2
import json
import os
from datetime import datetime
from typing import List, Tuple, Dict
import shutil


class AccidentFrameCapture:
    """
    Клас для збереження кадрів з аваріями з виділенням об'єктів та метаданими
    """
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Директорія для збереження кадрів з аваріями
        """
        self.output_dir = output_dir
        self.accidents_dir = os.path.join(output_dir, "accidents")
        self.metadata_file = os.path.join(self.accidents_dir, "accidents_log.json")
        
        if os.path.exists(self.accidents_dir):
            shutil.rmtree(self.accidents_dir)
        os.makedirs(self.accidents_dir,exist_ok=True)
        
        # Історія аварій (щоб не дублювати одну й ту саму аварію)
        self.accident_history = {}  # {track_id: last_accident_frame}
        self.cooldown_frames = 30  # Мінімальна відстань між збереженнями для одного ID
        
        # Лог всіх аварій
        self.accidents_log = []
        
    def should_save_accident(self, track_id: int, current_frame: int) -> bool:
        """
        Перевіряє чи потрібно зберігати аварію (щоб не дублювати)
        """
        if track_id not in self.accident_history:
            return True
        
        last_frame = self.accident_history[track_id]
        return (current_frame - last_frame) > self.cooldown_frames
    
    def save_accident_frame(
        self,
        frame: any,
        frame_number: int,
        accident_objects: List[Dict],
        video_path: str = None
    ) -> str:
        
        # Створюємо копію кадру для малювання
        annotated_frame = frame.copy()
        
        # Timestamp для унікальності
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Малюємо виділення навколо об'єктів аварії
        primary_ids = []
        secondary_ids = []
        
        for obj in accident_objects:
            track_id = obj['track_id']
            x1, y1, x2, y2 = obj['bbox']
            conf = obj.get('confidence', 0.0)
            obj_type = obj.get('type', 'primary')
            
            # Колір залежить від типу об'єкта
            if obj_type == 'primary':
                color = (0, 0, 255)  # Червоний для основних учасників
                thickness = 3
                primary_ids.append(track_id)
            else:
                color = (0, 165, 255)  # Помаранчевий для додаткових
                thickness = 2
                secondary_ids.append(track_id)
            
            # Малюємо прямокутник
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Додаємо текст з ID та впевненістю
            label = f"ID:{track_id} | {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Фон для тексту
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Текст
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Додаємо заголовок на кадр
        header = f"ACCIDENT DETECTED | Frame: {frame_number}"
        cv2.putText(
            annotated_frame,
            header,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )
        
        # Додаємо інформацію про кількість об'єктів
        info_text = f"Vehicles involved: {len(accident_objects)}"
        cv2.putText(
            annotated_frame,
            info_text,
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
        
        # Зберігаємо кадр
        filename = f"accident_{timestamp}_frame{frame_number}.jpg"
        filepath = os.path.join(self.accidents_dir, filename)
        cv2.imwrite(filepath, annotated_frame)
        
        # Також зберігаємо оригінальний кадр (без анотацій)
        original_filename = f"accident_{timestamp}_frame{frame_number}_original.jpg"
        original_filepath = os.path.join(self.accidents_dir, original_filename)
        cv2.imwrite(original_filepath, frame)
        
        # Оновлюємо історію для кожного ID
        for obj in accident_objects:
            self.accident_history[obj['track_id']] = frame_number
        
        # Додаємо метадані в лог
        accident_data = {
            'timestamp': timestamp,
            'frame_number': frame_number,
            'annotated_image': filename,
            'original_image': original_filename,
            'video_source': video_path,
            'total_vehicles': len(accident_objects),
            'primary_vehicles': primary_ids,
            'secondary_vehicles': secondary_ids,
            'objects': [
                {
                    'track_id': obj['track_id'],
                    'bbox': obj['bbox'],
                    'confidence': obj.get('confidence', 0.0),
                    'type': obj.get('type', 'primary')
                }
                for obj in accident_objects
            ]
        }
        
        self.accidents_log.append(accident_data)
        
        # Зберігаємо JSON лог
        self._save_metadata()
        
        return filepath
    
    def _save_metadata(self):
        """Зберігає метадані всіх аварій у JSON файл"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.accidents_log, f, indent=2, ensure_ascii=False)
    
    def get_accident_summary(self) -> Dict:
        """
        Повертає загальну статистику по всіх аваріях
        """
        if not self.accidents_log:
            return {'total_accidents': 0}
        
        total_vehicles = sum(a['total_vehicles'] for a in self.accidents_log)
        unique_vehicles = len(set(
            id for a in self.accidents_log 
            for id in a['primary_vehicles'] + a['secondary_vehicles']
        ))
        
        return {
            'total_accidents': len(self.accidents_log),
            'total_vehicles_involved': total_vehicles,
            'unique_vehicles': unique_vehicles,
            'accidents': self.accidents_log
        }