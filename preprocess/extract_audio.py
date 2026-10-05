"""Second-pass audio features. Does not interrupt the running vision preprocess.

For each existing cache npz, extract MFCC (13+delta+delta2=39) aligned to T frames
and write it back. Videos without usable audio get has_audio=0 and zero MFCC.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import os
import subprocess
import tempfile

import librosa
import numpy as np
from tqdm import tqdm

from common.utils import align_mfcc_to_frames, cache_path, load_config, repo_file, resolve_video_path, safe_video_name


def extract_audio(video_path: str, sr: int = 22050):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
            "-ar", str(sr), "-ac", "1", "-y", wav_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=45)
        if result.returncode != 0 or not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
            return None, sr
        audio, _ = librosa.load(wav_path, sr=sr)
        if audio is None or len(audio) < sr * 0.3:
            return None, sr
        return audio, sr
    except Exception:
        return None, sr
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def audio_to_mfcc(audio: np.ndarray, sr: int, num_frames: int) -> np.ndarray:
    feats = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=512)
    delta = librosa.feature.delta(feats)
    delta2 = librosa.feature.delta(feats, order=2)
    mfcc = np.vstack([feats, delta, delta2])
    return align_mfcc_to_frames(mfcc, num_frames)


def update_cache(npz_path: str, video_path: str, overwrite: bool = False) -> str:
    with np.load(npz_path, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files}
    if (not overwrite) and "mfcc" in payload and "has_audio" in payload:
        return "skip"
    t = int(payload["landmarks"].shape[0])
    audio, sr = extract_audio(video_path)
    if audio is None:
        payload["mfcc"] = np.zeros((t, 39), dtype=np.float32)
        payload["has_audio"] = np.int64(0)
        status = "no_audio"
    else:
        payload["mfcc"] = audio_to_mfcc(audio, sr, t).astype(np.float32)
        payload["has_audio"] = np.int64(1)
        status = "ok"
    tmp_path = npz_path[:-4] + ".tmp.npz" if npz_path.endswith(".npz") else npz_path + ".tmp.npz"
    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, npz_path)
    return status


_MP_OVERWRITE = False


def _mp_init(overwrite: bool) -> None:
    global _MP_OVERWRITE
    _MP_OVERWRITE = overwrite


def _mp_run(job):
    npz_path, video_path = job
    if not os.path.exists(npz_path):
        return "missing_cache"
    try:
        return update_cache(npz_path, video_path, overwrite=_MP_OVERWRITE)
    except Exception as e:
        print(f"FAIL {os.path.basename(npz_path)}: {type(e).__name__}: {e}", flush=True)
        return "fail"


def run_jobs(jobs, overwrite: bool, workers: int) -> None:
    global _MP_OVERWRITE
    _MP_OVERWRITE = overwrite
    counts = {"ok": 0, "no_audio": 0, "skip": 0, "missing_cache": 0, "fail": 0}
    if workers <= 1:
        iterator = (_mp_run(job) for job in jobs)
        pbar = tqdm(iterator, total=len(jobs), desc="audio")
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        pool = ctx.Pool(workers, initializer=_mp_init, initargs=(overwrite,))
        pbar = tqdm(pool.imap_unordered(_mp_run, jobs, chunksize=2), total=len(jobs), desc="audio")
    try:
        for status in pbar:
            counts[status] = counts.get(status, 0) + 1
            pbar.set_postfix(**{k: v for k, v in counts.items() if v})
    finally:
        if workers > 1:
            pool.close()
            pool.join()
    print(counts)


def main():
    parser = argparse.ArgumentParser(description="Add MFCC into existing vision caches")
    parser.add_argument("--config", default=repo_file("configs", "default.yaml"))
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--list", default="", help="optional json list of {video,label,name}")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache_dir", default="", help="override config data.cache_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache_dir = args.cache_dir or cfg["data"]["cache_dir"]
    if args.list:
        with open(args.list) as f:
            items = json.load(f)
    else:
        split_map = {
            "train": [cfg["data"]["train_json"]],
            "val": [cfg["data"]["val_json"]],
            "test": [cfg["data"]["test_json"]],
            "all": [cfg["data"]["train_json"], cfg["data"]["val_json"], cfg["data"]["test_json"]],
        }
        items = []
        for json_path in split_map[args.split]:
            with open(json_path) as f:
                items.extend(json.load(f))
    video_root = cfg.get("data", {}).get("video_root", "")
    jobs = []
    for item in items:
        name = safe_video_name(item)
        jobs.append((cache_path(cache_dir, name), resolve_video_path(item, video_root)))
    run_jobs(jobs, args.overwrite, max(1, int(args.workers)))


if __name__ == "__main__":
    main()
