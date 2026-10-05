import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
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
from common.utils import GENERATOR_NAMES, load_config, repo_file, set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate 5-family attribution")
    parser.add_argument("--config", default=repo_file("configs", "default.yaml"))
    parser.add_argument("--checkpoint", default=repo_file("weights", "generator_attribution.pth"))
    parser.add_argument("--split", default="test", choices=["val", "test"])
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
        json_path, cache_dir, cfg, task="attribution", split=args.split,
        require_audio=bool(cfg["attribution"].get("require_audio", False)),
        max_frames=int(cfg["attribution"]["max_frames"]),
    )
    per_gpu = int(cfg["attribution"]["batch_size"])
    batch_size = effective_batch_size(per_gpu, gpu_ids, distributed)
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, collate_fn=av_collate)
    ckpt = torch.load(args.checkpoint, map_location=device)
    num_classes = int(ckpt.get("num_classes", cfg["attribution"]["num_classes"]))
    model = wrap_model(AVTSTAN(num_classes=num_classes), device, gpu_ids, distributed, local_rank)
    load_state_dict(model, ckpt["model"])
    model.eval()
    labels, preds, names = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="infer", disable=not is_main()):
            logits, _ = model(
                batch["landmarks"].to(device),
                batch["lip_roi"].to(device),
                batch["mfcc"].to(device),
            )
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            labels.extend(batch["label"].numpy().tolist())
            names.extend(batch["name"])
    names = gather_list(names)
    labels = gather_list(labels)
    preds = gather_list(preds)
    seen = set()
    uniq = []
    for name, y, p in zip(names, labels, preds):
        if name in seen:
            continue
        seen.add(name)
        uniq.append((y, p))
    y = np.array([t[0] for t in uniq], dtype=np.int64)
    p = np.array([t[1] for t in uniq], dtype=np.int64)
    if is_main():
        zero_mfcc = sum(1 for s in dataset.samples if int(s.get("has_audio", 1)) != 1)
        print(
            f"clips={len(y)} zero_mfcc={zero_mfcc} "
            f"Acc={accuracy_score(y,p):.4f} macroF1={f1_score(y,p,average='macro',zero_division=0):.4f}"
        )
        print(classification_report(y, p, target_names=GENERATOR_NAMES[:num_classes], zero_division=0))
    cleanup()


if __name__ == "__main__":
    main()
