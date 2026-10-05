import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from common.utils import LIP_INDICES, cache_path, load_config, repo_file, resolve_video_path, safe_video_name


def extract_lip_roi(frame_rgb: np.ndarray, landmarks_xy: np.ndarray, expand_factor: float, lip_size: int) -> np.ndarray:
    lip = landmarks_xy[LIP_INDICES]
    x_min, y_min = lip.min(axis=0)
    x_max, y_max = lip.max(axis=0)
    width = max(x_max - x_min, 1e-6)
    height = max(y_max - y_min, 1e-6)
    x_min = max(0.0, x_min - width * expand_factor)
    x_max = min(1.0, x_max + width * expand_factor)
    y_min = max(0.0, y_min - height * expand_factor)
    y_max = min(1.0, y_max + height * expand_factor)

    h, w = frame_rgb.shape[:2]
    x1, x2 = int(x_min * w), int(x_max * w)
    y1, y2 = int(y_min * h), int(y_max * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))
    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((lip_size, lip_size, 3), dtype=np.uint8)
    return cv2.resize(crop, (lip_size, lip_size), interpolation=cv2.INTER_LINEAR)


def process_video(video_path: str, cfg: Dict) -> Optional[Dict[str, np.ndarray]]:
    mp_cfg = cfg["mediapipe"]
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=bool(mp_cfg["static_image_mode"]),
        max_num_faces=int(mp_cfg["max_num_faces"]),
        refine_landmarks=bool(mp_cfg["refine_landmarks"]),
        min_detection_confidence=float(mp_cfg["min_detection_confidence"]),
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        face_mesh.close()
        return None

    landmarks, lips, fail_mask = [], [], []
    prev_lm: Optional[np.ndarray] = None
    prev_lip: Optional[np.ndarray] = None
    consec_fail = 0
    expected = int(cfg["num_landmarks"])
    lip_size = int(cfg["lip_size"])
    expand = float(cfg["expand_factor"])

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(frame_rgb)
            failed = True
            lm = None
            if result.multi_face_landmarks:
                raw = result.multi_face_landmarks[0].landmark
                if len(raw) == expected:
                    lm = np.array([(p.x, p.y, p.z) for p in raw], dtype=np.float32)
                    failed = False
                    consec_fail = 0
            if failed:
                consec_fail += 1
                if prev_lm is not None:
                    lm = prev_lm.copy()
                    lip = prev_lip.copy() if prev_lip is not None else np.zeros((lip_size, lip_size, 3), dtype=np.uint8)
                else:
                    lm = np.zeros((expected, 3), dtype=np.float32)
                    lip = np.zeros((lip_size, lip_size, 3), dtype=np.uint8)
            else:
                lip = extract_lip_roi(frame_rgb, lm[:, :2], expand, lip_size)
                prev_lm, prev_lip = lm, lip

            landmarks.append(lm)
            lips.append(lip)
            fail_mask.append(1 if failed else 0)
    finally:
        cap.release()
        face_mesh.close()

    if not landmarks:
        return None
    return {
        "landmarks": np.stack(landmarks, axis=0).astype(np.float32),
        "lip_roi": np.stack(lips, axis=0).astype(np.uint8),
        "fail_mask": np.asarray(fail_mask, dtype=np.uint8),
    }


def load_split(json_path: str) -> List[Dict]:
    with open(json_path, "r") as f:
        return json.load(f)


def resolve_items(cfg: Dict, split: str) -> List[Tuple[str, List[Dict]]]:
    mapping = {
        "train": [("train", cfg["data"]["train_json"])],
        "val": [("val", cfg["data"]["val_json"])],
        "test": [("test", cfg["data"]["test_json"])],
        "all": [
            ("train", cfg["data"]["train_json"]),
            ("val", cfg["data"]["val_json"]),
            ("test", cfg["data"]["test_json"]),
        ],
    }
    if split not in mapping:
        raise ValueError(f"unknown split: {split}")
    return [(name, load_split(path)) for name, path in mapping[split]]


def save_features(out_path: str, feat: Dict[str, np.ndarray], item: Dict) -> None:
    np.savez_compressed(
        out_path,
        landmarks=feat["landmarks"],
        lip_roi=feat["lip_roi"],
        fail_mask=feat["fail_mask"],
        label=np.int64(item["label"]),
        name=np.asarray(safe_video_name(item)),
    )


def process_item(item: Dict, cfg: Dict, cache_dir: str, overwrite: bool) -> str:
    name = safe_video_name(item)
    out_path = cache_path(cache_dir, name)
    if os.path.exists(out_path) and not overwrite:
        return "skip"
    video = resolve_video_path(item, cfg.get("data", {}).get("video_root", ""))
    if not os.path.exists(video):
        return "fail"
    feat = process_video(video, cfg)
    if feat is None:
        return "fail"
    save_features(out_path, feat, item)
    return "ok"


_MP_CFG = None
_MP_CACHE = None
_MP_OVERWRITE = False


def _mp_init(cfg: Dict, cache_dir: str, overwrite: bool) -> None:
    global _MP_CFG, _MP_CACHE, _MP_OVERWRITE
    _MP_CFG = cfg
    _MP_CACHE = cache_dir
    _MP_OVERWRITE = overwrite


def _mp_run(item: Dict) -> str:
    return process_item(item, _MP_CFG, _MP_CACHE, _MP_OVERWRITE)


def run_items(split_name: str, items: List[Dict], cfg: Dict, cache_dir: str, overwrite: bool, workers: int) -> None:
    ok = skipped = failed = 0
    if workers <= 1:
        pbar = tqdm(items, desc=f"preprocess[{split_name}]")
        for item in pbar:
            status = process_item(item, cfg, cache_dir, overwrite)
            if status == "ok":
                ok += 1
            elif status == "skip":
                skipped += 1
            else:
                failed += 1
            pbar.set_postfix(ok=ok, skip=skipped, fail=failed)
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_mp_init, initargs=(cfg, cache_dir, overwrite)) as pool:
            pbar = tqdm(pool.imap_unordered(_mp_run, items, chunksize=1), total=len(items), desc=f"preprocess[{split_name}]")
            for status in pbar:
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skipped += 1
                else:
                    failed += 1
                pbar.set_postfix(ok=ok, skip=skipped, fail=failed)
    print(f"[{split_name}] saved={ok} skipped={skipped} failed={failed} total={len(items)}")


def main():
    parser = argparse.ArgumentParser(description="Precompute MediaPipe 478 landmarks and lip ROIs")
    parser.add_argument("--config", default=repo_file("configs", "default.yaml"))
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--list", default="", help="optional json list of {video,label,name}; overrides --split")
    parser.add_argument("--limit", type=int, default=0, help="process at most N videos per split (0=all)")
    parser.add_argument("--workers", type=int, default=1, help="CPU processes; MediaPipe does not use GPU")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache_dir", default="", help="override config data.cache_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache_dir = args.cache_dir or cfg["data"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    if args.list:
        groups = [("list", load_split(args.list))]
    else:
        groups = resolve_items(cfg, args.split)
    for split_name, items in groups:
        if args.limit > 0:
            items = items[: args.limit]
        run_items(split_name, items, cfg, cache_dir, args.overwrite, max(1, int(args.workers)))


if __name__ == "__main__":
    main()
