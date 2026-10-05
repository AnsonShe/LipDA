"""Official LipDA(V) vision inference.

Published protocol (paper Acc):
  video_score = max_combo P(fake)
  decision threshold = 0.985

mean(combo) is also printed. The audio-visual detector is clip-level (no combo
max/mean); its silent-video fallback reads this file's official **max** `score`.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from vision.load_combos import build_official_model, infer_split, load_combo_cache
from common.utils import load_config, repo_file, set_seed


def report(labels, scores, threshold, title, unit="Videos"):
    pred = (np.asarray(scores) >= threshold).astype(np.int64)
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    tn, fp, fn, tp = (cm.ravel().tolist() + [0, 0, 0, 0])[:4]
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    print("=" * 40)
    print(title)
    print("=" * 40)
    print(f"{unit}:     {len(labels)}  real={(labels == 0).sum()}  fake={(labels == 1).sum()}")
    print(f"Threshold:  {threshold:.4f}")
    print(f"Accuracy:   {accuracy_score(labels, pred):.4f}")
    print(f"AUC:        {roc_auc_score(labels, scores):.4f}" if len(np.unique(labels)) == 2 else "AUC: n/a")
    print(f"AP:         {average_precision_score(labels, scores):.4f}" if len(np.unique(labels)) == 2 else "AP: n/a")
    print(f"F1:         {f1_score(labels, pred, zero_division=0):.4f}")
    print(f"Precision:  {precision_score(labels, pred, zero_division=0):.4f}")
    print(f"Recall:     {recall_score(labels, pred, zero_division=0):.4f}")
    print(f"FPR:        {fpr:.4f}")
    print(f"FNR:        {fnr:.4f}")
    print(f"Confusion:  TN={tn} FP={fp} FN={fn} TP={tp}")


def youden_thr(y, scores):
    fpr, tpr, thrs = roc_curve(y, scores)
    i = int(np.argmax(tpr - fpr))
    return float(thrs[i])


def resolve_cache(cfg, args, split: str) -> str:
    if args.cache_file:
        return args.cache_file
    ov = cfg.get("vision", {})
    caches = ov.get("cache", {})
    path = caches.get(split, "")
    if path:
        return path
    root = ov.get("preprocessed_dir") or os.environ.get("LIPDA_DLIB_CACHE", "")
    if not root:
        raise FileNotFoundError(
            f"set vision.preprocessed_dir or LIPDA_DLIB_CACHE to the dlib T=5 npy directory"
        )
    return os.path.join(root, f"{split}_cache.npy")


def main():
    parser = argparse.ArgumentParser(description="Official LipDA(V) vision eval (max default, also reports mean)")
    parser.add_argument("--config", default=repo_file("configs", "default.yaml"))
    parser.add_argument("--checkpoint", default=repo_file("weights", "vision_deepfake_detector.pth"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", default="paper", help="paper=0.985 | auto | 0.5 | float")
    parser.add_argument("--aggregate", default="", choices=["", "max", "mean"], help="default from config: max")
    parser.add_argument("--save_scores", default="")
    parser.add_argument("--json", default="", help="override split json")
    parser.add_argument("--cache_file", default="", help="official dlib T=5 npy cache")
    parser.add_argument("--cache_dir", default="", help="ignored for vision LipDA(V); use --cache_file")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    ov = cfg.get("vision", {})
    aggregate = args.aggregate or str(ov.get("aggregate", "max"))
    paper_thr = float(ov.get("threshold", 0.985))
    json_key = {"train": "train_json", "val": "val_json", "test": "test_json"}[args.split]
    json_path = args.json or cfg["data"][json_key]
    cache_file = resolve_cache(cfg, args, args.split)
    if not os.path.isfile(cache_file):
        raise FileNotFoundError(
            f"official vision cache missing: {cache_file}. "
            "Need dlib-68 T=5 npy (e.g. $LIPDA_DLIB_CACHE/test_cache.npy), not MediaPipe cache/."
        )

    gpu = str(args.gpus).split(",")[0].strip()
    if gpu not in ("", "cpu", "all") and torch.cuda.is_available():
        if gpu == "all":
            gpu = "0"
        device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ckpt_path = args.checkpoint or ov.get("checkpoint")
    print(f"official LipDA(V) device={device}")
    print(f"checkpoint={ckpt_path}")
    print(f"json={json_path}")
    print(f"cache={cache_file}")
    print(f"official aggregate={aggregate}  paper_thr={paper_thr}")
    print("arch1 video score = max(combo) [paper]; mean(combo) is alternate")
    print("arch2 is clip-level (no combo aggregate); silent fallback uses arch1 max")

    items = json.load(open(json_path))
    cache, by_video = load_combo_cache(cache_file)
    model = build_official_model(ckpt_path, device, lstm_hidden=int(ov.get("lstm_hidden", 256)))
    payload = infer_split(
        model, items, cache, by_video, device,
        batch_size=args.batch_size, aggregate=aggregate,
    )
    rows = payload["video"]
    if payload["skipped"]:
        print(f"skipped {len(payload['skipped'])} videos missing from official cache")

    combo_y = np.asarray(payload["combo_auc_labels"], dtype=np.int64)
    combo_s = np.asarray(payload["combo_auc_scores"], dtype=np.float64)
    vid_y = np.asarray([r["label"] for r in rows], dtype=np.int64)
    vid_max = np.asarray([r["score_max"] for r in rows], dtype=np.float64)
    vid_mean = np.asarray([r["score_mean"] for r in rows], dtype=np.float64)
    official = vid_max if aggregate == "max" else vid_mean

    report(combo_y, combo_s, 0.5, f"{args.split} combo @ 0.5", unit="Combos")
    report(vid_y, vid_max, 0.5, f"{args.split} video-max @ 0.5", unit="Videos")
    report(vid_y, vid_max, paper_thr, f"{args.split} video-max @ {paper_thr:.4f} (paper / official Acc)", unit="Videos")
    report(vid_y, vid_mean, 0.5, f"{args.split} video-mean @ 0.5 (alternate)", unit="Videos")
    y_max = youden_thr(vid_y, vid_max)
    y_mean = youden_thr(vid_y, vid_mean)
    report(vid_y, vid_max, y_max, f"{args.split} video-max @ Youden {y_max:.4f}", unit="Videos")
    report(vid_y, vid_mean, y_mean, f"{args.split} video-mean @ Youden {y_mean:.4f}", unit="Videos")

    if args.threshold not in ("paper", "auto"):
        thr = float(args.threshold)
        how = "max" if aggregate == "max" else "mean"
        report(vid_y, official, thr, f"{args.split} video-{how} @ {thr:.4f}", unit="Videos")

    if args.save_scores:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_scores)) or ".", exist_ok=True)
        with open(args.save_scores, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"saved scores to {args.save_scores} (score={aggregate})")


if __name__ == "__main__":
    main()
