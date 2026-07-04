# models.py
import torch
import torch.nn as nn
import torchvision.models as tv
import numpy as np
import librosa
from torch.utils.data import Dataset
import json
import torch.nn.functional as F
# -------------------------
# helpers
# -------------------------
def l2_normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)

# -------------------------
# Lip Encoder (per-frame -> per-combo)
# -------------------------
class LipEncoder(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=False, per_frame_dim=256, proj_dim=128):
        super().__init__()
        if backbone == 'resnet18':
            net = tv.resnet18(pretrained=pretrained)
            modules = list(net.children())[:-2]
            self.backbone = nn.Sequential(*modules)
            feat_dim = 512
        else:
            raise NotImplementedError
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.frame_fc = nn.Sequential(
            nn.Linear(feat_dim, per_frame_dim),
            nn.ReLU(),
            nn.Linear(per_frame_dim, proj_dim)
        )

    def forward(self, combo_imgs):
        # combo_imgs shape: [B, T, C, H, W] 或 [B*T, C, H, W]
        if combo_imgs.dim() == 5:
            B, T, C, H, W = combo_imgs.shape
            x = combo_imgs.view(B*T, C, H, W)
        else:
            B_T, C, H, W = combo_imgs.shape
            x = combo_imgs
        
        featmap = self.backbone(x)
        pooled = self.pool(featmap).view(-1, featmap.size(1))
        per_frame = self.frame_fc(pooled)
        
        if combo_imgs.dim() == 5:
            per_frame = per_frame.view(B, T, -1)
            concat = per_frame.view(B, -1)
        else:
            concat = per_frame
        
        return concat, per_frame


class PoseEncoder(nn.Module):
    # === MODIFIED: 增加了 lstm_hidden_dim 参数 ===
    def __init__(self, per_frame_dim=128, proj_dim=128, lstm_hidden_dim=256, T=5):
        super().__init__()
        self.T = T
        in_dim = 68*2
        self.frame_mlp = nn.Sequential(
            nn.Linear(in_dim, per_frame_dim),
            nn.ReLU(),
            nn.Linear(per_frame_dim, proj_dim)
        )
        # +++ NEW: 增加LSTM层用于时序建模 +++
        self.lstm = nn.LSTM(
            input_size=proj_dim, 
            hidden_size=lstm_hidden_dim, 
            num_layers=2,            # 使用两层LSTM增加模型深度
            batch_first=True,        # 输入数据格式为 [B, T, D]
            bidirectional=True       # 使用双向LSTM捕捉前后文信息
        )
        # 因为是双向LSTM，所以输出维度是 hidden_dim * 2
        self.output_dim = lstm_hidden_dim * 2

    def forward(self, lms):
        B = lms.shape[0]
        T = lms.shape[1]
        x = lms.view(B*T, -1)
        
        per = self.frame_mlp(x)
        per = per.view(B, T, -1) # shape: [B, T, proj_dim]

        # +++ NEW: 将逐帧特征送入LSTM +++
        # self.lstm 返回 (所有时间步的输出, (最后一个时间步的hidden_state, 最后一个时间步的cell_state))
        lstm_out, _ = self.lstm(per)
        
        # 使用最后一个时间步的输出作为整个序列的特征表示
        # lstm_out shape: [B, T, hidden_dim * 2]
        final_feature = lstm_out[:, -1, :] # shape: [B, hidden_dim * 2]

        # === MODIFIED: 返回LSTM处理后的特征和逐帧特征 ===
        return final_feature, per


# -------------------------
# Projection to hypersphere unit vector
# -------------------------
class Projector(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x):
        v = self.fc(x)
        v = nn.functional.normalize(v, p=2, dim=1)
        return v

# -------------------------
# Detection Head (per-combo)
# -------------------------
class DetectionHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2, 2)
        )
    def forward(self, x):
        return self.classifier(x)
    

