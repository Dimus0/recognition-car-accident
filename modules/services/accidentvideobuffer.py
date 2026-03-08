import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque, defaultdict
from typing import Optional
import logging
import os
from modules.utils import *
from modules.config.config import Config

logger = logging.getLogger("accident_detector")

class AccidentVideoBuffer:
    def __init__(self, output_dir: str, fps: float,
                 before_sec:   float = Config.VIDEO_BUFFER_SECONDS,
                 after_sec:    float = Config.VIDEO_BUFFER_SECONDS,
                 # Аліаси для зворотньої сумісності з console.py
                 pre_seconds:  float = None,
                 post_seconds: float = None):
        self.output_dir    = output_dir
        self.fps           = fps
        # pre_seconds/post_seconds мають пріоритет над before_sec/after_sec
        before = pre_seconds  if pre_seconds  is not None else before_sec
        after  = post_seconds if post_seconds is not None else after_sec
        self.before_frames = max(1, int(fps * before))
        self.after_frames  = max(1, int(fps * after))
        self.pre_buffer    = deque(maxlen=self.before_frames)
        self.post_buffer   = []
        self.recording     = False
        self.frames_left   = 0
        self.clip_index    = 0
        self.trigger_frame = None
        self._last_path    = None

    def push(self, frame: np.ndarray):
        """Додає кадр у rolling pre-буфер або post-буфер під час запису."""
        if not self.recording:
            self.pre_buffer.append(frame.copy())
        else:
            self.post_buffer.append(frame.copy())
            self.frames_left -= 1
            if self.frames_left <= 0:
                self._save_clip()

    # Аліас — console.py викликає add_frame()
    def add_frame(self, frame: np.ndarray, frame_count: int = None):
        self.push(frame)

    def trigger(self, frame_count: int) -> Optional[str]:
        """
        Запускає збереження кліпу.
        Повертає майбутній шлях файлу ОДРАЗУ (до фактичного запису),
        щоб schedule_notification міг передати шлях зі затримкою.
        Якщо вже записує — повертає None.
        """
        if self.recording:
            return None
        self.recording     = True
        self.frames_left   = self.after_frames
        self.trigger_frame = frame_count
        self.post_buffer   = []
        # Передбачуємо майбутній шлях — він буде готовий після after_frames кадрів
        future_path = os.path.join(
            self.output_dir,
            f"accident_clip_{self.clip_index + 1:04d}_f{frame_count}.mp4"
        )
        self._last_path = future_path
        logger.info(f"[VideoBuffer] Тригер на кадрі {frame_count} → {future_path}")
        return future_path

    def _save_clip(self):
        all_frames = list(self.pre_buffer) + self.post_buffer
        if not all_frames:
            self._reset(); return
        h, w = all_frames[0].shape[:2]
        self.clip_index += 1
        path = os.path.join(
            self.output_dir,
            f"accident_clip_{self.clip_index:04d}_f{self.trigger_frame}.mp4"
        )
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
        )
        for fr in all_frames:
            writer.write(fr)
        writer.release()
        self._last_path = path
        logger.critical(f"[VideoBuffer] Кліп збережено: {path} ({len(all_frames)} кадрів)")
        print(f"  🎥 Кліп збережено: {path}")
        self._reset()

    def _reset(self):
        self.recording     = False
        self.frames_left   = 0
        self.post_buffer   = []
        # Очищаємо pre_buffer після збереження кліпу —
        # щоб наступний кліп не містив кадрів із попереднього інциденту
        self.pre_buffer.clear()

    def flush(self):
        """
        Примусово зберігає кліп якщо запис активний але відео закінчилось
        (наприклад, аварія сталась наприкінці файлу і after_frames не добрались).
        Викликати після виходу з основного циклу.
        """
        if self.recording and (self.pre_buffer or self.post_buffer):
            logger.info("[VideoBuffer] flush() — примусове збереження незавершеного кліпу")
            self._save_clip()