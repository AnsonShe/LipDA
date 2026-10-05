import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from common.utils import (
    FAMILY_DISPLAY_NAMES,
    GENERATOR_TO_ID,
    cache_path,
    generator_of,
    normalize_landmarks,
    pad_or_truncate,
    safe_video_name,
)


def _load_clip_arrays(path: str):
    with np.load(path, allow_pickle=False) as data:
        landmarks = data["landmarks"].astype(np.float32)
        lip_roi = data["lip_roi"]
        fail_mask = data["fail_mask"].astype(np.float32)
        mfcc = data["mfcc"].astype(np.float32) if "mfcc" in data.files else None
        has_audio = int(data["has_audio"]) if "has_audio" in data.files else 0
    return landmarks, lip_roi, fail_mask, mfcc, has_audio


class ClipAVDataset(Dataset):
    """Full-clip dataset for AV detection (binary) and attribution (K-way)."""

    def __init__(
        self,
        video_json: str,
        cache_dir: str,
        cfg: Dict[str, Any],
        task: str = "audiovisual",
        split: str = "train",
        require_audio: bool = True,
        max_frames: int = 75,
        verbose: bool = True,
    ):
        if task not in {"audiovisual", "attribution"}:
            raise ValueError(task)
        self.task = task
        self.split = split
        self.max_frames = max_frames
        mp_cfg = cfg["mediapipe"]
        self.nose_index = int(mp_cfg["nose_index"])
        self.left_eye_index = int(mp_cfg["left_eye_index"])
        self.right_eye_index = int(mp_cfg["right_eye_index"])
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        with open(video_json, "r") as f:
            items = json.load(f)

        self.samples: List[Dict[str, Any]] = []
        self.missing_cache = 0
        self.missing_audio = 0
        self.skipped_real = 0
        self.unknown_gen = 0
        for item in items:
            name = safe_video_name(item)
            path = cache_path(cache_dir, name)
            if not os.path.exists(path):
                self.missing_cache += 1
                continue
            try:
                with np.load(path, allow_pickle=False) as data:
                    has_audio = int(data["has_audio"]) if "has_audio" in data.files else 0
                    t = int(data["landmarks"].shape[0])
            except Exception:
                self.missing_cache += 1
                continue
            if t < 8:
                continue
            if require_audio and has_audio != 1:
                self.missing_audio += 1
                continue
            if task == "attribution":
                if int(item["label"]) == 0:
                    self.skipped_real += 1
                    continue
                gen = generator_of(item)
                if gen not in GENERATOR_TO_ID:
                    self.unknown_gen += 1
                    continue
                label = GENERATOR_TO_ID[gen]
            else:
                label = int(item["label"])
            self.samples.append({
                "cache": path,
                "label": label,
                "name": name,
                "video": item["video"],
                "has_audio": has_audio,
            })
        if verbose:
            print(
                f"[{split}/{task}] clips={len(self.samples)} missing_cache={self.missing_cache} "
                f"no_audio={self.missing_audio} skip_real={self.skipped_real} unknown_gen={self.unknown_gen}"
            )
            if task == "attribution":
                from collections import Counter
                hist = Counter(int(s["label"]) for s in self.samples)
                silent = sum(1 for s in self.samples if int(s["has_audio"]) != 1)
                parts = [f"{FAMILY_DISPLAY_NAMES[i]}={hist.get(i, 0)}" for i in range(len(FAMILY_DISPLAY_NAMES))]
                print(f"[{split}/{task}] families {' '.join(parts)} zero_mfcc={silent}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        landmarks, lip_roi, _, mfcc, _ = _load_clip_arrays(sample["cache"])
        if mfcc is None:
            mfcc = np.zeros((landmarks.shape[0], 39), dtype=np.float32)
        landmarks = normalize_landmarks(
            pad_or_truncate(landmarks, self.max_frames),
            self.nose_index, self.left_eye_index, self.right_eye_index,
        )
        lips = pad_or_truncate(lip_roi, self.max_frames)
        mfcc = pad_or_truncate(mfcc, self.max_frames)
        frames = []
        for frame in lips:
            img = frame if frame.dtype == np.uint8 else np.clip(frame, 0, 255).astype(np.uint8)
            frames.append(self.normalize(self.to_tensor(img)))
        lip_tensor = torch.stack(frames, dim=0)  # [T, 3, H, W]
        return {
            "landmarks": torch.from_numpy(landmarks),
            "lip_roi": lip_tensor,
            "mfcc": torch.from_numpy(mfcc.astype(np.float32)),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "name": sample["name"],
            "has_audio": int(sample.get("has_audio", 0)),
        }
