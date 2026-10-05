"""Visual distortions for robustness tests (DeeperForensics-style).

Types: CS color saturation, CC color contrast, BW block-wise, GNC Gaussian
noise in color, GB Gaussian blur, JPEG compression, PXL pixelation.
Levels 1–5 (1 mild, 5 strong).
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import random
import subprocess

import cv2
import numpy as np
from tqdm import tqdm

DIST_TYPES = ["CS", "CC", "BW", "GNC", "GB", "JPEG", "PXL"]


def bgr2ycbcr(img_bgr):
    img_bgr = img_bgr.astype(np.float32)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
    img_ycbcr = img_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0
    return img_ycbcr


def ycbcr2bgr(img_ycbcr):
    img_ycbcr = img_ycbcr.astype(np.float32)
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * 255.0 - 16) / (235 - 16)
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * 255.0 - 16) / (240 - 16)
    img_ycrcb = img_ycbcr[:, :, (0, 2, 1)].astype(np.float32)
    return cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2BGR)


def color_saturation(img, param):
    ycbcr = bgr2ycbcr(img)
    ycbcr[:, :, 1] = 0.5 + (ycbcr[:, :, 1] - 0.5) * param
    ycbcr[:, :, 2] = 0.5 + (ycbcr[:, :, 2] - 0.5) * param
    return ycbcr2bgr(ycbcr).astype(np.uint8)


def color_contrast(img, param):
    img = np.clip(img.astype(np.float32) * param, 0, 255)
    return img.astype(np.uint8)


def block_wise(img, param):
    width = 8
    block = np.ones((width, width, 3), dtype=np.uint8) * 128
    param = min(img.shape[0], img.shape[1]) // 256 * param
    out = img.copy()
    for _ in range(int(param)):
        r_w = random.randint(0, max(img.shape[1] - 1 - width, 0))
        r_h = random.randint(0, max(img.shape[0] - 1 - width, 0))
        out[r_h : r_h + width, r_w : r_w + width, :] = block
    return out


def gaussian_noise_color(img, param):
    ycbcr = bgr2ycbcr(img) / 255
    noise = np.random.randn(*ycbcr.shape)
    b = (ycbcr + math.sqrt(param) * noise) * 255
    return np.clip(ycbcr2bgr(b), 0, 255).astype(np.uint8)


def gaussian_blur(img, param):
    ksize = int(param)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def jpeg_compression(img, param):
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100 - int(param)])
    if not ok:
        return img
    return cv2.imdecode(enc, 1)


def pixelation(img, param):
    h, w = img.shape[:2]
    param = max(int(param), 1)
    small_w = max(w // param, 1)
    small_h = max(h // param, 1)
    temp = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


_PARAM = {
    "CS": [0.4, 0.3, 0.2, 0.1, 0.0],
    "CC": [0.85, 0.725, 0.6, 0.475, 0.35],
    "BW": [16, 32, 48, 64, 80],
    "GNC": [0.001, 0.002, 0.005, 0.01, 0.05],
    "GB": [7, 9, 13, 17, 21],
    "JPEG": [30, 32, 35, 38, 40],
    "PXL": [2, 3, 4, 5, 6],
}
_FUNC = {
    "CS": color_saturation,
    "CC": color_contrast,
    "BW": block_wise,
    "GNC": gaussian_noise_color,
    "GB": gaussian_blur,
    "JPEG": jpeg_compression,
    "PXL": pixelation,
}


def get_parameter(dist_type: str, level: int):
    return _PARAM[dist_type][int(level) - 1]


def apply_frame(img, dist_type: str, level: int):
    return _FUNC[dist_type](img, get_parameter(dist_type, level))


def pick_type_level(dist_type: str, level):
    if dist_type == "random":
        dist_type = random.choice(DIST_TYPES)
    if str(level) == "random":
        level = random.randint(1, 5)
    else:
        level = int(level)
    if dist_type not in _FUNC:
        raise ValueError(f"unknown distortion type {dist_type}")
    if level not in range(1, 6):
        raise ValueError(f"level must be 1–5, got {level}")
    return dist_type, level


def distort_video(vid_in: str, vid_out: str, dist_type="random", level="random", via_xvid=False):
    dist_type, dist_level = pick_type_level(dist_type, level)
    os.makedirs(os.path.dirname(os.path.abspath(vid_out)) or ".", exist_ok=True)

    cap = cv2.VideoCapture(vid_in)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    tmp_avi = f"{vid_out[:-4]}_tmp.avi" if via_xvid else None
    writer = (
        cv2.VideoWriter(tmp_avi, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
        if via_xvid
        else cv2.VideoWriter(vid_out, fourcc, fps, (w, h))
    )
    fn = _FUNC[dist_type]
    param = get_parameter(dist_type, dist_level)
    for frame in tqdm(frames, desc=f"{dist_type}:{dist_level}", leave=False):
        writer.write(fn(frame, param))
    writer.release()
    if via_xvid:
        subprocess.run(["ffmpeg", "-y", "-i", tmp_avi, vid_out], check=False, capture_output=True)
        if os.path.exists(tmp_avi):
            os.remove(tmp_avi)
    return dist_type, dist_level


def write_meta(meta_path, vid_in, vid_out, dist_type, dist_level):
    os.makedirs(os.path.dirname(os.path.abspath(meta_path)) or ".", exist_ok=True)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            for line in f.read().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                meta[parts[0]] = parts[1:]
    prev = copy.deepcopy(meta.get(vid_in, []))
    prev.append(f"{dist_type}:{dist_level}")
    meta[vid_out] = prev
    with open(meta_path, "w") as f:
        for path, tags in meta.items():
            f.write(" ".join([path] + tags) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Apply a visual distortion to one video")
    parser.add_argument("--vid_in", required=True)
    parser.add_argument("--vid_out", required=True)
    parser.add_argument("--type", default="random", help="CS|CC|BW|GNC|GB|JPEG|PXL|random")
    parser.add_argument("--level", default="random", help="1–5 or random")
    parser.add_argument("--meta_path", default=None)
    parser.add_argument("--via_xvid", action="store_true")
    args = parser.parse_args()
    if not os.path.isfile(args.vid_in):
        raise FileNotFoundError(args.vid_in)
    if os.path.abspath(args.vid_in) == os.path.abspath(args.vid_out):
        raise ValueError("vid_in and vid_out must differ")
    dist_type, dist_level = distort_video(args.vid_in, args.vid_out, args.type, args.level, args.via_xvid)
    print(f"wrote {args.vid_out}  {dist_type}:{dist_level}")
    if args.meta_path:
        write_meta(args.meta_path, args.vid_in, args.vid_out, dist_type, dist_level)


if __name__ == "__main__":
    main()
