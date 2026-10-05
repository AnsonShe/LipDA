import hashlib
import os
import random
from typing import Any, Dict

import numpy as np
import yaml

try:
    import torch
except ImportError:  # preprocess-only hosts may not have torch
    torch = None

LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 320, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
]


def project_root() -> str:
    env = os.environ.get("LIPDA_ROOT")
    if env:
        return os.path.abspath(env)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def repo_file(*parts: str) -> str:
    return os.path.join(project_root(), *parts)


def resolve_path(path: str) -> str:
    """Expand ~/$VARS; join repo-relative paths to the project root."""
    if not path:
        return path
    path = os.path.expandvars(os.path.expanduser(str(path)))
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(project_root(), path))


def resolve_video_path(item: Dict[str, Any], video_root: str = "") -> str:
    """Join list JSON `video` with LIPDA_DATA_ROOT / config data.video_root."""
    video = os.path.expandvars(os.path.expanduser(str(item.get("video", ""))))
    if os.path.isabs(video):
        return video
    root = video_root or os.environ.get("LIPDA_DATA_ROOT", "")
    if root:
        return os.path.normpath(os.path.join(os.path.expanduser(root), video))
    return video


_REPO_PATH_KEYS = {
    ("data", "train_json"),
    ("data", "val_json"),
    ("data", "test_json"),
    ("data", "cache_dir"),
    ("vision", "checkpoint"),
    ("vision", "preprocessed_dir"),
    ("audiovisual", "save_dir"),
    ("attribution", "save_dir"),
    ("attribution", "generator_map"),
    ("attribution", "init_from_audiovisual"),
}


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["log_dir"] = resolve_path(cfg.get("log_dir") or "logs")
    data = cfg.setdefault("data", {})
    if not data.get("video_root"):
        data["video_root"] = os.environ.get("LIPDA_DATA_ROOT", "")
    for section, key in _REPO_PATH_KEYS:
        if section == "data":
            if key in data and data[key]:
                data[key] = resolve_path(data[key])
            continue
        block = cfg.get(section, {})
        if key in block and block[key]:
            block[key] = resolve_path(block[key])
    ov = cfg.get("vision", {})
    caches = ov.get("cache") or {}
    for split, cache_path_value in list(caches.items()):
        if cache_path_value:
            caches[split] = resolve_path(cache_path_value)
    dlib_env = os.environ.get("LIPDA_DLIB_CACHE", "")
    if dlib_env and not ov.get("preprocessed_dir"):
        ov["preprocessed_dir"] = os.path.abspath(os.path.expanduser(dlib_env))
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def anonymous_id(text: str) -> str:
    """Stable 16-hex id. Used in released splits so original clip names are not published."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def cache_path(cache_dir: str, name: str) -> str:
    return os.path.join(cache_dir, f"{anonymous_id(name)}.npz")


def normalize_landmarks(
    landmarks: np.ndarray,
    nose_index: int = 1,
    left_eye_index: int = 33,
    right_eye_index: int = 263,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize MediaPipe xyz per frame: origin=nose, scale=inter-ocular distance."""
    pts = landmarks.astype(np.float32).copy()
    if pts.ndim != 3 or pts.shape[-1] != 3:
        raise ValueError(f"expected [T, 478, 3], got {pts.shape}")

    origin = pts[:, nose_index : nose_index + 1, :]
    left_eye = pts[:, left_eye_index, :]
    right_eye = pts[:, right_eye_index, :]
    scale = np.linalg.norm(left_eye - right_eye, axis=1, keepdims=True)
    face_width = pts[:, :, 0].max(axis=1, keepdims=True) - pts[:, :, 0].min(axis=1, keepdims=True)
    scale = np.where(scale < eps, np.maximum(face_width, eps), scale)
    pts = (pts - origin) / scale[:, :, None]
    return pts


def best_accuracy_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Scan thresholds on [0, 1] and return the one with highest accuracy."""
    if len(scores) == 0:
        return 0.5
    candidates = np.unique(np.concatenate([[0.5], scores]))
    best_thr, best_acc = 0.5, -1.0
    for thr in candidates:
        pred = (scores >= thr).astype(np.int64)
        acc = float((pred == labels).mean())
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_thr


def safe_video_name(item: Dict[str, Any]) -> str:
    return item.get("name") or os.path.splitext(os.path.basename(item["video"]))[0]


# Paper Table 3 / Appendix B.2: five generator families (not per-model 10-way).
FAMILY_NAMES = [
    "transformer",
    "gan",
    "diffusion",
    "vae",
    "statistical",
]
FAMILY_DISPLAY_NAMES = [
    "Transformer-based",
    "GAN-based",
    "Diffusion-based",
    "VAE-based",
    "Statistical models",
]
FAMILY_TO_ID = {name: i for i, name in enumerate(FAMILY_NAMES)}
GENERATOR_TO_FAMILY = {
    "IP_LAP": "transformer",
    "makeittalk": "gan",
    "wav2lip": "gan",
    "dreamtalk": "diffusion",
    "sadtalker": "vae",
    "dinet": "statistical",
}
GENERATOR_TO_ID = {gen: FAMILY_TO_ID[fam] for gen, fam in GENERATOR_TO_FAMILY.items()}
GENERATOR_NAMES = FAMILY_DISPLAY_NAMES


def generator_of(item: Dict[str, Any]) -> str:
    if int(item.get("label", 1)) == 0:
        return "real"
    if item.get("generator"):
        return str(item["generator"])
    parts = os.path.normpath(str(item.get("orig_video") or item.get("video", ""))).split(os.sep)
    for key in ("fake", "1_fake"):
        if key in parts:
            idx = parts.index(key)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return "unknown"


def pad_or_truncate(array: np.ndarray, length: int) -> np.ndarray:
    if len(array) >= length:
        return array[:length]
    pad = [(0, length - len(array))] + [(0, 0)] * (array.ndim - 1)
    return np.pad(array, pad, mode="constant")


def align_mfcc_to_frames(mfcc: np.ndarray, num_frames: int) -> np.ndarray:
    """mfcc: [39, T_audio] -> [num_frames, 39]"""
    if mfcc.size == 0 or mfcc.shape[1] == 0:
        return np.zeros((num_frames, 39), dtype=np.float32)
    t_audio = mfcc.shape[1]
    out = np.zeros((num_frames, mfcc.shape[0]), dtype=np.float32)
    for i in range(num_frames):
        j = min(int(i * t_audio / max(num_frames, 1)), t_audio - 1)
        out[i] = mfcc[:, j]
    return out
