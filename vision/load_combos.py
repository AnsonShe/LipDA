"""Load official LipDA(V) dlib-68 / T=5 combo cache and aggregate video scores.

Official published protocol (paper LipDA(V)):
  video_score = max_combo P(fake), Acc threshold = 0.985

mean(combo) is an alternate aggregation; it is reported but not the release default.
The audio-visual detector does not aggregate combos — one clip, one score. Silent
clips fall back to this module's official **max** video score.
"""
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

from common.utils import anonymous_id
from vision.model import OfficialVisionModel

TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def cache_key(split: str, video_id: str, combo_name: str) -> str:
    return f"preprocessed_data/{split}/{video_id}/combos/{combo_name}"


def index_cache(cache: dict) -> Dict[str, List[str]]:
    lips = cache["lip_images"]
    by_video = defaultdict(list)
    for key in lips:
        parts = str(key).split("/")
        if len(parts) >= 5:
            orig = parts[2]
            by_video[orig].append(key)
            by_video[anonymous_id(orig)].append(key)
    return by_video


def load_combo_cache(path: str) -> Tuple[dict, Dict[str, List[str]]]:
    cache = np.load(path, allow_pickle=True).item()
    return cache, index_cache(cache)


def combo_tensors(cache: dict, keys: List[str]):
    lip_tensors, lm_tensors = [], []
    for key in keys:
        arr = cache["lip_images"].get(key)
        lm = cache["landmarks"].get(key)
        if arr is None or lm is None:
            continue
        arr = np.asarray(arr)
        lm = np.asarray(lm)
        if arr.ndim != 4 or arr.shape[0] != 5:
            continue
        lip_tensors.append(torch.stack([TRANSFORM(img) for img in arr]))
        lm_tensors.append(torch.tensor(lm, dtype=torch.float32))
    if not lip_tensors:
        return None, None
    return torch.stack(lip_tensors), torch.stack(lm_tensors)


@torch.no_grad()
def score_combos(model: OfficialVisionModel, lips: torch.Tensor, lms: torch.Tensor, device, batch_size: int = 64) -> np.ndarray:
    loader = DataLoader(TensorDataset(lips, lms), batch_size=batch_size, shuffle=False)
    probs = []
    for lip_x, lm_x in loader:
        logits = model(lip_x.to(device), lm_x.to(device))
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    return torch.cat(probs, dim=0).numpy().astype(np.float64)


def aggregate_video(combo_probs: np.ndarray, how: str) -> float:
    if how == "mean":
        return float(np.mean(combo_probs))
    if how != "max":
        raise ValueError(f"aggregate must be max or mean, got {how}")
    return float(np.max(combo_probs))


def infer_split(
    model: OfficialVisionModel,
    items: list,
    cache: dict,
    by_video: Dict[str, List[str]],
    device,
    batch_size: int = 64,
    aggregate: str = "max",
) -> dict:
    combo_y, combo_s = [], []
    rows = []
    skipped = []
    for item in tqdm(items, desc="official-vision"):
        name = item["name"]
        y = int(item["label"])
        keys = by_video.get(name, [])
        lips, lms = combo_tensors(cache, keys)
        if lips is None:
            skipped.append(name)
            continue
        probs = score_combos(model, lips, lms, device, batch_size=batch_size)
        combo_y.extend([y] * len(probs))
        combo_s.extend(probs.tolist())
        rows.append(
            {
                "name": name,
                "label": y,
                "score_max": float(np.max(probs)),
                "score_mean": float(np.mean(probs)),
                "n_combos": int(len(probs)),
            }
        )
    for row in rows:
        row["score"] = row["score_max"] if aggregate == "max" else row["score_mean"]
        row["aggregate"] = aggregate
    return {
        "video": rows,
        "combo_auc_labels": combo_y,
        "combo_auc_scores": combo_s,
        "skipped": skipped,
        "aggregate": aggregate,
    }


def build_official_model(checkpoint: str, device, lstm_hidden: int = 256) -> OfficialVisionModel:
    model = OfficialVisionModel(lstm_hidden_dim=lstm_hidden).to(device)
    model.load_official_checkpoint(checkpoint, map_location=device)
    model.eval()
    return model
