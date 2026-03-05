import cv2
import numpy as np
import torch
import torch.nn as nn
import random
import pickle
from torchvision import transforms
from collections import deque, defaultdict
from sklearn.preprocessing import StandardScaler
import logging
import os
from modules.utils import *
from modules.services.analyzer import TrafficAnalyzer
from modules.services.accidenttracker import AccidentStateTracker
from modules.services.accidentcapture import AccidentFrameCapture
from modules.metrics import MetricsTracker

VIDEO_BUFFER_SECONDS    = 2.0        # Буфер до/після аварії (секунди)
logger = logging.getLogger("accident_detector")

class AccidentVideoBuffer:
    def __init__(self, output_dir: str, fps: float,
                 before_sec: float = VIDEO_BUFFER_SECONDS,
                 after_sec:  float = VIDEO_BUFFER_SECONDS):
        self.output_dir    = output_dir
        self.fps           = fps
        self.before_frames = max(1, int(fps * before_sec))
        self.after_frames  = max(1, int(fps * after_sec))
        self.pre_buffer    = deque(maxlen=self.before_frames)
        self.post_buffer   = []
        self.recording     = False
        self.frames_left   = 0
        self.clip_index    = 0
        self.trigger_frame = None
        self._last_path    = None

    def push(self, frame: np.ndarray):
        if not self.recording:
            self.pre_buffer.append(frame.copy())
        else:
            self.post_buffer.append(frame.copy())
            self.frames_left -= 1
            if self.frames_left <= 0:
                self._save_clip()

    def trigger(self, frame_count: int):
        if self.recording:
            return
        self.recording     = True
        self.frames_left   = self.after_frames
        self.trigger_frame = frame_count
        self.post_buffer   = []
        logger.info(f"[VideoBuffer] Тригер на кадрі {frame_count}")

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
        self.recording   = False
        self.frames_left = 0
        self.post_buffer = []