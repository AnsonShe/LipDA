"""Official LipDA(V) vision stack: ResNet18 lip (T=5 concat) + dlib 68 PoseEncoder.

Standalone copy of the published LipDA(V) modules so this repo can load
`weights/vision_deepfake_detector.pth` without depending on unpublished source trees.
"""
import torch
import torch.nn as nn
import torchvision.models as tv


def _resnet18(pretrained: bool = False) -> nn.Module:
    try:
        return tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    except TypeError:
        return tv.resnet18(pretrained=pretrained)


class LipEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = False, per_frame_dim: int = 256, proj_dim: int = 128):
        super().__init__()
        if backbone != "resnet18":
            raise NotImplementedError(backbone)
        net = _resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.frame_fc = nn.Sequential(
            nn.Linear(512, per_frame_dim),
            nn.ReLU(),
            nn.Linear(per_frame_dim, proj_dim),
        )

    def forward(self, combo_imgs: torch.Tensor):
        if combo_imgs.dim() == 5:
            b, t, c, h, w = combo_imgs.shape
            x = combo_imgs.reshape(b * t, c, h, w)
        else:
            x = combo_imgs
            b = None
            t = None
        featmap = self.backbone(x)
        pooled = self.pool(featmap).view(-1, featmap.size(1))
        per_frame = self.frame_fc(pooled)
        if combo_imgs.dim() == 5:
            per_frame = per_frame.view(b, t, -1)
            return per_frame.reshape(b, -1), per_frame
        return per_frame, per_frame


class PoseEncoder(nn.Module):
    def __init__(self, per_frame_dim: int = 128, proj_dim: int = 128, lstm_hidden_dim: int = 256, T: int = 5):
        super().__init__()
        self.T = T
        self.frame_mlp = nn.Sequential(
            nn.Linear(68 * 2, per_frame_dim),
            nn.ReLU(),
            nn.Linear(per_frame_dim, proj_dim),
        )
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.output_dim = lstm_hidden_dim * 2

    def forward(self, lms: torch.Tensor):
        b, t = lms.shape[:2]
        per = self.frame_mlp(lms.reshape(b * t, -1)).view(b, t, -1)
        lstm_out, _ = self.lstm(per)
        return lstm_out[:, -1, :], per


class DetectionHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class OfficialVisionModel(nn.Module):
    """Published LipDA(V): T=5 lip concat 640 + BiLSTM pose 512 → 2-class head."""

    def __init__(self, lstm_hidden_dim: int = 256, t: int = 5):
        super().__init__()
        self.t = int(t)
        self.lip_enc = LipEncoder(pretrained=False)
        self.pose_enc = PoseEncoder(lstm_hidden_dim=lstm_hidden_dim, T=self.t)
        self.det_head = DetectionHead(self.t * 128 + self.pose_enc.output_dim)

    def forward(self, lips: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        lip_feat, _ = self.lip_enc(lips)
        pose_feat, _ = self.pose_enc(landmarks)
        return self.det_head(torch.cat([lip_feat, pose_feat], dim=1))

    def load_official_checkpoint(self, path: str, map_location=None) -> dict:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        if "lip_enc" in ckpt and "pose_enc" in ckpt and "det_head" in ckpt:
            self.lip_enc.load_state_dict(ckpt["lip_enc"])
            self.pose_enc.load_state_dict(ckpt["pose_enc"])
            self.det_head.load_state_dict(ckpt["det_head"])
            return ckpt
        state = ckpt.get("model", ckpt)
        if state and next(iter(state)).startswith("module."):
            state = {k[len("module.") :]: v for k, v in state.items()}
        self.load_state_dict(state)
        return ckpt


class Projector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.fc(x), p=2, dim=1)


class FeatureAlignmentLoss(nn.Module):
    def __init__(self, initial_epsilon: float = 1.0):
        super().__init__()
        self.epsilon = nn.Parameter(torch.tensor(float(initial_epsilon)))

    def forward(self, lip_feat: torch.Tensor, pose_feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        distances = torch.norm(lip_feat - pose_feat, p=2, dim=1)
        real_mask = labels == 0
        fake_mask = labels == 1
        real_loss = (distances[real_mask] ** 2).mean() if real_mask.any() else distances.new_zeros(())
        fake_loss = (
            torch.clamp(self.epsilon - distances[fake_mask], min=0) ** 2
        ).mean() if fake_mask.any() else distances.new_zeros(())
        return real_loss + fake_loss


class OfficialVisionTrainModel(OfficialVisionModel):
    """Official LipDA(V) plus the alignment projectors from the original finetune recipe."""

    def __init__(self, lstm_hidden_dim: int = 256, t: int = 5):
        super().__init__(lstm_hidden_dim=lstm_hidden_dim, t=t)
        self.proj_lip = Projector(self.t * 128)
        self.proj_pose = Projector(self.pose_enc.output_dim)

    def forward(self, lips: torch.Tensor, landmarks: torch.Tensor):
        lip_feat, _ = self.lip_enc(lips)
        pose_feat, _ = self.pose_enc(landmarks)
        logits = self.det_head(torch.cat([lip_feat, pose_feat], dim=1))
        return logits, self.proj_lip(lip_feat), self.proj_pose(pose_feat)

    def load_train_checkpoint(self, path: str, map_location=None) -> dict:
        ckpt = self.load_official_checkpoint(path, map_location=map_location)
        if "proj_lip" in ckpt:
            self.proj_lip.load_state_dict(ckpt["proj_lip"])
        if "proj_pose" in ckpt:
            self.proj_pose.load_state_dict(ckpt["proj_pose"])
        return ckpt

    def export_official_state(self) -> dict:
        return {
            "lip_enc": self.lip_enc.state_dict(),
            "pose_enc": self.pose_enc.state_dict(),
            "proj_lip": self.proj_lip.state_dict(),
            "proj_pose": self.proj_pose.state_dict(),
            "det_head": self.det_head.state_dict(),
        }
