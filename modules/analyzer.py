import cv2
import numpy as np
import torch
from torchvision import transforms
from collections import defaultdict, deque

class TrafficAnalyzer:
    def __init__(self, max_history=25, collision_distance=80,min_motion=3):
        self.track_history = defaultdict(lambda: deque(maxlen=max_history))
        self.active_ids = set()
        self.collision_distance = collision_distance
        self.min_motion = min_motion

    def update_tracks(self, boxes, ids):
        for (x1, y1, x2, y2), tid in zip(boxes, ids):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            self.track_history[tid].append((cx, cy))

    def point_to_line_distance(self, px, py, x1, y1, x2, y2):
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

    def activate_near_line(self, boxes, ids, lines, threshold=80):
        for (x1, y1, x2, y2), tid in zip(boxes, ids):
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            for (lx1, ly1), (lx2, ly2) in lines:
                dist = self.point_to_line_distance(cx, cy, lx1, ly1, lx2, ly2)
                if dist < threshold:
                    self.active_ids.add(tid)

    def predict_collision(self, ids=None):
        risky = set()
        ids = list(ids) if ids is not None else list(self.active_ids)

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]

                if a not in self.track_history or b not in self.track_history:
                    continue

                if len(self.track_history[a]) < 5 or len(self.track_history[b]) < 5:
                    continue

                p1 = np.array(self.track_history[a][-1])
                p2 = np.array(self.track_history[b][-1])

                if np.linalg.norm(p1 - p2) < self.collision_distance:
                    risky.add(a)
                    risky.add(b)

        return risky

    def clean_old_tracks(self, current_ids):
        for tid in list(self.track_history.keys()):
            if tid not in current_ids:
                del self.track_history[tid]
                self.active_ids.discard(tid)

    def is_moving_towards_camera(self, tid):
        """
        True якщо обʼєкт рухається в камеру
        """
        pts = self.track_history.get(tid, [])
        if len(pts) < self.min_motion:
            return False

        y_start = pts[0][1]
        y_end = pts[-1][1]

        return (y_end - y_start) > 10  # поріг у пікселях