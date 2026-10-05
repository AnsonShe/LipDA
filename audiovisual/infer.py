import argparse
import json
import os
import sys

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
)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data.distributed import DistributedSampler

from audiovisual.train_loop import av_collate
from common.clip_dataset import ClipAVDataset
from common.gpu import (
    cleanup,
    effective_batch_size,
    gather_list,
    init_distributed,
    is_main,
    load_state_dict,
    parse_gpu_ids,
    resolve_device,
    wrap_model,
)
from audiovisual.model import AVTSTAN
from common.utils import cache_path, load_config, repo_file, safe_video_name, set_seed


def _report(labels, scores, title, threshold=0.5):
    pred = (scores >= threshold).astype(np.int64)
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    tn, fp, fn, tp = (cm.ravel().tolist() + [0, 0, 0, 0])[:4]
    print("=" * 40)
    print(title)
    print("=" * 40)
    print(f"Videos:     {len(labels)}  real={(labels == 0).sum()}  fake={(labels == 1).sum()}")
    print(f"Threshold:  {threshold:.4f}")
    print(f"Accuracy:   {accuracy_score(labels, pred):.4f}")
    print(f"AUC:        {roc_auc_score(labels, scores):.4f}" if len(np.unique(labels)) == 2 else "AUC: n/a")
    print(f"AP:         {average_precision_score(labels, scores):.4f}" if len(np.unique(labels)) == 2 else "AP: n/a")
    print(f"F1:         {f1_score(labels, pred, zero_division=0):.4f}")
    print(f"Precision:  {precision_score(labels, pred, zero_division=0):.4f}")
    print(f"Recall:     {recall_score(labels, pred, zero_division=0):.4f}")
    print(f"Confusion:  TN={tn} FP={fp} FN={fn} TP={tp}")


def _load_vision_scores(path):
    with open(path) as f:
        payload = json.load(f)
    rows = payload["video"] if isinstance(payload, dict) and "video" in payload else payload
    return {row["name"]: float(row["score"]) for row in rows}


def _silent_test_rows(json_path, cache_dir):
    with open(json_path) as f:
        items = json.load(f)
    rows = []
    for item in items:
        name = safe_video_name(item)
        path = cache_path(cache_dir, name)
        if not os.path.exists(path):
            continue
        with np.load(path, allow_pickle=False) as data:
            has_audio = int(data["has_audio"]) if "has_audio" in data.files else 0
            t = int(data["landmarks"].shape[0])
        if t < 8:
            continue
        if has_audio != 1:
            rows.append((name, int(item["label"])))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate AV detector; silent clips fall back to vision scores")
    parser.add_argument("--config", default=repo_file("configs", "default.yaml"))
    parser.add_argument("--checkpoint", default=repo_file("weights", "audiovisual_deepfake_detector.pth"))
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument(
        "--vision_scores",
        default="",
        help="task-1 video-MAX JSON from vision/infer.py; silent clips use these scores",
    )
    parser.add_argument("--json", default="", help="override split json")
    parser.add_argument("--cache_dir", default="", help="override cache dir")
    parser.add_argument("--gpus", default="all")
    args = parser.parse_args()
    distributed, local_rank, _, _ = init_distributed()
    gpu_ids = parse_gpu_ids(args.gpus)
    device = resolve_device(distributed, local_rank, gpu_ids)
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    json_path = args.json or (cfg["data"]["val_json"] if args.split == "val" else cfg["data"]["test_json"])
    cache_dir = args.cache_dir or cfg["data"]["cache_dir"]
    dataset = ClipAVDataset(
        json_path, cache_dir, cfg, task="audiovisual", split=args.split,
        require_audio=True, max_frames=int(cfg["audiovisual"]["max_frames"]),
        verbose=is_main(),
    )
    per_gpu = int(cfg["audiovisual"]["batch_size"])
    batch_size = effective_batch_size(per_gpu, gpu_ids, distributed)
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, collate_fn=av_collate)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = wrap_model(AVTSTAN(num_classes=2), device, gpu_ids, distributed, local_rank)
    load_state_dict(model, ckpt["model"])
    model.eval()
    labels, scores, names = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="infer", disable=not is_main()):
            logits, _ = model(
                batch["landmarks"].to(device),
                batch["lip_roi"].to(device),
                batch["mfcc"].to(device),
            )
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            scores.extend(prob.tolist())
            labels.extend(batch["label"].numpy().tolist())
            names.extend(batch["name"])
    names = gather_list(names)
    labels = gather_list(labels)
    scores = gather_list(scores)
    seen = set()
    audio_rows = []
    for name, y, s in zip(names, labels, scores):
        if name in seen:
            continue
        seen.add(name)
        audio_rows.append((name, int(y), float(s)))
    if is_main():
        y = np.array([t[1] for t in audio_rows], dtype=np.int64)
        s = np.array([t[2] for t in audio_rows], dtype=np.float64)
        _report(y, s, f"{args.split} AV-only (has_audio=1, no silent clips)")
        if args.vision_scores:
            vision = _load_vision_scores(args.vision_scores)
            silent = _silent_test_rows(json_path, cache_dir)
            merged_y, merged_s = list(y), list(s)
            used, missing = 0, 0
            for name, lab in silent:
                if name not in vision:
                    missing += 1
                    continue
                merged_y.append(lab)
                merged_s.append(vision[name])
                used += 1
            print(f"vision fallback: silent={len(silent)} used={used} missing_scores={missing}")
            _report(
                np.asarray(merged_y, dtype=np.int64),
                np.asarray(merged_s, dtype=np.float64),
                f"{args.split} full-set = AV(audio) + vision(silent)",
            )
    cleanup()


if __name__ == "__main__":
    main()
