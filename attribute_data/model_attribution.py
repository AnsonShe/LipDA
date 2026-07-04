import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalDynamicStream(nn.Module):
    """时序动态分支：处理landmark序列"""
    def __init__(self, input_size=478*3, hidden_size=128, num_layers=2, num_classes=3):
        super().__init__()
        
        # 1D卷积层提取局部时序特征
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        # Bi-LSTM捕捉长时序依赖
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 分类器
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        # x形状: (batch_size, seq_len, landmarks, 3) 需要先展平
        batch_size, seq_len, num_landmarks, coords = x.shape
        
        # 展平landmark坐标: (batch_size, seq_len, landmarks*3)
        x = x.reshape(batch_size, seq_len, -1)
        
        # 转换为(batch_size, features, seq_len) 用于1D卷积
        x = x.transpose(1, 2)  # (batch_size, landmarks*3, seq_len)
        
        # 1D卷积
        x = self.conv1d(x)  # (batch_size, 128, seq_len/4)
        
        # 转换回(batch_size, seq_len, features) 用于LSTM
        x = x.transpose(1, 2)  # (batch_size, seq_len/4, 128)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        out = lstm_out[:, -1, :]
        
        # 分类
        out = self.fc(out)
        
        return out

class AVSynchronizationStream(nn.Module):
    """音视同步分支：处理唇部ROI和MFCC特征"""
    def __init__(self, lip_roi_shape=(75, 3, 96, 96), mfcc_dim=39, hidden_dim=128, num_classes=3):
        super().__init__()
        
        # 视觉编码器 (3D CNN)
        self.visual_encoder = nn.Sequential(
            # 输入: (batch, channels, time, height, width)
            nn.Conv3d(3, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.AdaptiveAvgPool3d((None, 1, 1))  # 添加自适应池化来固定空间维度
        )
        
        # 音频编码器 (1D CNN)
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(mfcc_dim, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(10)  # 固定音频时间维度
        )
        
        # 视觉特征投影层
        self.visual_projection = nn.Linear(256, 256)  # 投影到256维
        
        # 音频特征投影层
        self.audio_projection = nn.Linear(256, 256)   # 投影到256维
        
        # 跨模态注意力融合
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256, 
            num_heads=8, 
            batch_first=True,
            dropout=0.1
        )
        
        # 分类器
        self.fc = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, lip_roi, mfcc):
        batch_size = lip_roi.size(0)
        
        # lip_roi形状: (batch_size, seq_len, channels, height, width)
        # 需要转换为: (batch_size, channels, seq_len, height, width) 用于3D CNN
        lip_roi = lip_roi.permute(0, 2, 1, 3, 4)
        
        # 视觉特征提取
        visual_features = self.visual_encoder(lip_roi)  # (batch, 256, time, 1, 1)
        
        # 移除空间维度
        visual_features = visual_features.squeeze(-1).squeeze(-1)  # (batch, 256, time)
        visual_features = visual_features.permute(0, 2, 1)  # (batch, time, 256)
        
        # 投影到注意力期望的维度
        visual_features = self.visual_projection(visual_features)  # (batch, time, 256)
        
        # 音频特征提取
        # mfcc形状: (batch_size, seq_len, mfcc_dim)
        mfcc = mfcc.transpose(1, 2)  # (batch_size, mfcc_dim, seq_len)
        audio_features = self.audio_encoder(mfcc)  # (batch_size, 256, 10)
        audio_features = audio_features.transpose(1, 2)  # (batch_size, 10, 256)
        
        # 投影到注意力期望的维度
        audio_features = self.audio_projection(audio_features)  # (batch_size, 10, 256)
        
        # 调整时间维度对齐
        min_time = min(visual_features.size(1), audio_features.size(1))
        visual_features = visual_features[:, :min_time, :]
        audio_features = audio_features[:, :min_time, :]
        
        # 跨模态注意力 (视觉作为Query，音频作为Key和Value)
        attn_output, attn_weights = self.cross_attention(
            visual_features, audio_features, audio_features
        )
        
        # 合并特征
        visual_pooled = torch.mean(visual_features, dim=1)
        attn_pooled = torch.mean(attn_output, dim=1)
        fused_features = torch.cat([visual_pooled, attn_pooled], dim=1)
        
        # 分类
        out = self.fc(fused_features)
        
        return out, attn_weights

class AVTSTAN(nn.Module):
    """视听双流时空归因网络"""
    def __init__(self, num_classes=3):
        super().__init__()
        self.temporal_stream = TemporalDynamicStream(num_classes=num_classes)
        self.av_sync_stream = AVSynchronizationStream(num_classes=num_classes)
        
        # 融合分类器
        self.fusion_fc = nn.Sequential(
            nn.Linear(num_classes * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, landmarks, lip_roi, mfcc):
        # 打印输入形状用于调试
        # print(f"landmarks shape: {landmarks.shape}")
        # print(f"lip_roi shape: {lip_roi.shape}")
        # print(f"mfcc shape: {mfcc.shape}")
        
        # 时序动态分支
        temporal_out = self.temporal_stream(landmarks)
        
        # 音视同步分支
        av_sync_out, attn_weights = self.av_sync_stream(lip_roi, mfcc)
        
        # 特征融合
        fused = torch.cat([temporal_out, av_sync_out], dim=1)
        out = self.fusion_fc(fused)
        
        return out, attn_weights