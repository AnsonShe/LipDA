import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalDynamicStream(nn.Module):
    def __init__(self, input_size: int = 478 * 3, hidden_size: int = 128, num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, n, c = x.shape
        x = x.reshape(b, t, n * c).transpose(1, 2)
        x = self.conv1d(x).transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class AVSynchronizationStream(nn.Module):
    def __init__(self, mfcc_dim: int = 39, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.visual_encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(mfcc_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(10),
        )
        self.visual_projection = nn.Linear(256, 256)
        self.audio_projection = nn.Linear(256, 256)
        self.cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, lip_roi: torch.Tensor, mfcc: torch.Tensor):
        # lip_roi: [B, T, C, H, W]
        visual = self.visual_encoder(lip_roi.permute(0, 2, 1, 3, 4))
        t = visual.size(2)
        visual = F.adaptive_avg_pool3d(visual, (t, 1, 1)).squeeze(-1).squeeze(-1)
        visual = self.visual_projection(visual.transpose(1, 2))
        audio = self.audio_encoder(mfcc.transpose(1, 2)).transpose(1, 2)
        audio = self.audio_projection(audio)
        n = min(visual.size(1), audio.size(1))
        visual, audio = visual[:, :n], audio[:, :n]
        attn_out, attn_w = self.cross_attention(visual, audio, audio)
        fused = torch.cat([visual.mean(dim=1), attn_out.mean(dim=1)], dim=1)
        return self.fc(fused), attn_w


class AVTSTAN(nn.Module):
    """Shared AV backbone for binary detection (num_classes=2) and attribution."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.temporal_stream = TemporalDynamicStream(num_classes=num_classes)
        self.av_sync_stream = AVSynchronizationStream(num_classes=num_classes)
        self.fusion_fc = nn.Sequential(
            nn.Linear(num_classes * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, landmarks: torch.Tensor, lip_roi: torch.Tensor, mfcc: torch.Tensor):
        temporal_out = self.temporal_stream(landmarks)
        av_out, attn_w = self.av_sync_stream(lip_roi, mfcc)
        return self.fusion_fc(torch.cat([temporal_out, av_out], dim=1)), attn_w
