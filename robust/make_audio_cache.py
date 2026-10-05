"""Rebuild npz caches with perturbed audio, keeping original landmarks / lip ROI."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import librosa
import numpy as np
from tqdm import tqdm

from common.utils import (
    cache_path,
    load_config,
    repo_file,
    resolve_video_path,
    safe_video_name,
)
from preprocess.extract_audio import audio_to_mfcc, extract_audio
from robust.audio_perturb import PAPER_PRESETS, AudioPerturbation


def main():
    keys = [k for k, _ in PAPER_PRESETS]
    parser = argparse.ArgumentParser(description="Write a new cache dir with perturbed MFCC")
    parser.add_argument("--config", default=repo_file("configs", "default.yaml"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--json", default="")
    parser.add_argument("--src_cache", default="", help="original MediaPipe+MFCC cache (default: config cache_dir)")
    parser.add_argument("--out_cache", required=True)
    parser.add_argument("--preset", default="noise_light", help="one of: " + ",".join(keys))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.preset not in keys:
        raise ValueError(f"preset must be one of {keys}")

    cfg = load_config(args.config)
    json_path = args.json or cfg["data"][{"train": "train_json", "val": "val_json", "test": "test_json"}[args.split]]
    src_cache = args.src_cache or cfg["data"]["cache_dir"]
    items = json.load(open(json_path))
    if args.limit:
        items = items[: args.limit]
    os.makedirs(args.out_cache, exist_ok=True)
    video_root = cfg.get("data", {}).get("video_root", "")
    preset_fn = dict(PAPER_PRESETS)[args.preset]
    ok = skip = fail = 0
    for item in tqdm(items, desc=f"audio {args.preset}"):
        name = safe_video_name(item)
        src = cache_path(src_cache, name)
        dst = cache_path(args.out_cache, name)
        if not os.path.isfile(src):
            skip += 1
            continue
        with np.load(src, allow_pickle=False) as data:
            payload = {k: data[k] for k in data.files}
        t = int(payload["landmarks"].shape[0])
        video = resolve_video_path(item, video_root)
        audio, sr = extract_audio(video)
        if audio is None:
            payload["mfcc"] = np.zeros((t, 39), dtype=np.float32)
            payload["has_audio"] = np.int64(0)
            np.savez_compressed(dst, **payload)
            fail += 1
            continue
        perturb = AudioPerturbation()
        perturb.load_array(audio, sr)
        wav, out_sr, _ = preset_fn(perturb)
        if out_sr != sr:
            wav = librosa.resample(wav, orig_sr=out_sr, target_sr=sr)
            out_sr = sr
        payload["mfcc"] = audio_to_mfcc(wav.astype(np.float32), int(out_sr), t).astype(np.float32)
        payload["has_audio"] = np.int64(1)
        np.savez_compressed(dst, **payload)
        ok += 1
    print(f"ok={ok} skip_missing_cache={skip} no_audio={fail} -> {args.out_cache}")
    print("next: python audiovisual/infer.py --cache_dir", args.out_cache)
    print("      python attribution/infer.py --cache_dir", args.out_cache)


if __name__ == "__main__":
    main()
