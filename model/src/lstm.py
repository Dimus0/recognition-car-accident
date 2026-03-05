import numpy as np
import torch
import torch.nn as nn
import random
from torchvision import transforms
from collections import deque, defaultdict
from sklearn.preprocessing import StandardScaler
from modules.utils import *

class MotionLSTM(nn.Module):
    """
    Encoder-Decoder LSTM (архітектура ідентична тренуванню).

    Вхід:  (B, obs_len=20, 4)  — x, y, vx, vy  (StandardScaler-scaled)
    Вихід: (B, pred_len=30, 2) — x, y           (scaled простір)
    """
    def __init__(self, input_size: int = 4, hidden: int = 128,
                 layers: int = 2, pred_len: int = 30, drop: float = 0.3):
        super().__init__()
        self.pred_len = pred_len
        self.encoder  = nn.LSTM(input_size, hidden, layers,
                                batch_first=True,
                                dropout=drop if layers > 1 else 0.)
        self.decoder  = nn.LSTM(2, hidden, layers,
                                batch_first=True,
                                dropout=drop if layers > 1 else 0.)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(64, 2)
        )

    def forward(self, x: torch.Tensor,
                target: torch.Tensor = None,
                tf_ratio: float = 0.0) -> torch.Tensor:
        # При інференсі tf_ratio=0 — авторегресивно без ground truth
        _, (h, c) = self.encoder(x)
        dec  = x[:, -1, :2].unsqueeze(1)   # остання спостережена (x,y) scaled
        outs = []
        for t in range(self.pred_len):
            out, (h, c) = self.decoder(dec, (h, c))
            pred = self.fc(out.squeeze(1))
            outs.append(pred.unsqueeze(1))
            dec = (target[:, t, :].unsqueeze(1)
                   if target is not None and random.random() < tf_ratio
                   else pred.unsqueeze(1))
        return torch.cat(outs, dim=1)   # (B, pred_len, 2)