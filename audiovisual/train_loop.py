import json
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from audiovisual.model import AVTSTAN
from common.clip_dataset import ClipAVDataset
from common.gpu import (
    all_reduce_sum,
    cleanup,
    effective_batch_size,
    gather_numpy,
    init_distributed,
    is_main,
    load_state_dict,
    parse_gpu_ids,
    resolve_device,
    state_dict_to_save,
    unwrap_model,
    wrap_model,
)
from common.utils import GENERATOR_NAMES, load_config, repo_file, set_seed


def setup_logger(log_dir: str, name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"), mode="a")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def append_metrics(path: str, record: dict) -> None:
    clean = {}
    for key, value in record.items():
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        clean[key] = value
    with open(path, "a") as f:
        f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def av_collate(batch):
    return {
        "landmarks": torch.stack([x["landmarks"] for x in batch], dim=0),
        "lip_roi": torch.stack([x["lip_roi"] for x in batch], dim=0),
        "mfcc": torch.stack([x["mfcc"] for x in batch], dim=0),
        "label": torch.stack([x["label"] for x in batch], dim=0),
        "name": [x["name"] for x in batch],
    }


def class_balanced_weights(labels, num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def load_matching_weights(model, src_state: dict) -> tuple:
    if src_state and next(iter(src_state)).startswith("module."):
        src_state = {k[len("module.") :]: v for k, v in src_state.items()}
    dst = unwrap_model(model)
    cur = dst.state_dict()
    matched = {k: v for k, v in src_state.items() if k in cur and tuple(cur[k].shape) == tuple(v.shape)}
    cur.update(matched)
    dst.load_state_dict(cur)
    return len(matched), len(cur)


def run_av_epoch(
    model,
    loader,
    device,
    optimizer=None,
    class_weight=None,
    label_smoothing: float = 0.0,
    max_grad_norm: float = 0.0,
):
    train_mode = optimizer is not None
    model.train(train_mode)
    crit = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=float(label_smoothing))
    total_loss = correct = total = 0
    all_labels, all_probs, all_preds = [], [], []
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="train" if train_mode else "eval", leave=False, disable=not is_main()):
            landmarks = batch["landmarks"].to(device, non_blocking=True)
            lip_roi = batch["lip_roi"].to(device, non_blocking=True)
            mfcc = batch["mfcc"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits, _ = model(landmarks, lip_roi, mfcc)
            loss = crit(logits, labels)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if max_grad_norm and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            total_loss += float(loss.item())
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.detach().cpu().numpy())
            all_preds.append(pred.detach().cpu().numpy())
    labels = np.concatenate(all_labels) if all_labels else np.zeros((0,), dtype=np.int64)
    preds = np.concatenate(all_preds) if all_preds else np.zeros((0,), dtype=np.int64)
    n_cls = all_probs[0].shape[1] if all_probs else 2
    probs = np.concatenate(all_probs) if all_probs else np.zeros((0, n_cls), dtype=np.float32)
    labels = gather_numpy(labels)
    preds = gather_numpy(preds)
    probs = gather_numpy(probs)
    correct = all_reduce_sum(correct, device)
    total = all_reduce_sum(total, device)
    total_loss = all_reduce_sum(total_loss, device)
    num_batches = all_reduce_sum(len(loader), device)
    metrics = {
        "loss": total_loss / max(num_batches, 1),
        "acc": correct / max(total, 1),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)) if len(labels) else 0.0,
    }
    if probs.shape[1] == 2 and len(np.unique(labels)) == 2:
        metrics["auc"] = float(roc_auc_score(labels, probs[:, 1]))
        metrics["ap"] = float(average_precision_score(labels, probs[:, 1]))
    elif probs.shape[1] > 2 and len(labels):
        try:
            metrics["auc"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
        except ValueError:
            metrics["auc"] = 0.0
    else:
        metrics["auc"] = 0.0
        metrics["ap"] = 0.0
    return metrics


def train_task(task: str, args):
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    section = cfg["audiovisual"] if task == "audiovisual" else cfg["attribution"]
    num_classes = 2 if task == "audiovisual" else int(section["num_classes"])
    epochs = args.epochs or int(section["epochs"])
    per_gpu = args.batch_size or int(section["batch_size"])
    lr = args.lr or float(section["lr"])
    save_dir = args.save_dir or section["save_dir"]
    log_name = getattr(args, "log_name", "") or task
    log_dir = os.path.join(cfg.get("log_dir", repo_file("logs")), log_name)
    require_audio = bool(section.get("require_audio", True))
    gpus = getattr(args, "gpus", "all")
    distributed, local_rank, world_size, _ = init_distributed()
    gpu_ids = parse_gpu_ids(gpus)
    device = resolve_device(distributed, local_rank, gpu_ids)
    batch_size = effective_batch_size(per_gpu, gpu_ids, distributed)
    logger = setup_logger(log_dir, f"detect_{task}") if is_main() else None
    metrics_path = os.path.join(log_dir, "metrics.jsonl")
    if is_main():
        os.makedirs(save_dir, exist_ok=True)
        logger.info(
            f"{task} device={device} gpus={gpu_ids} ddp={distributed} world={world_size} "
            f"batch={batch_size} epochs={epochs} lr={lr} require_audio={require_audio} "
            f"select={section.get('select', 'auc' if task == 'audiovisual' else 'macro_f1')} "
            f"label_smoothing={section.get('label_smoothing', 0.0)} "
            f"class_weight={section.get('class_weight', False)}"
        )

    train_set = ClipAVDataset(
        cfg["data"]["train_json"], cfg["data"]["cache_dir"], cfg,
        task=task, split="train",
        require_audio=require_audio,
        max_frames=int(section["max_frames"]),
        verbose=is_main(),
    )
    val_set = ClipAVDataset(
        cfg["data"]["val_json"], cfg["data"]["cache_dir"], cfg,
        task=task, split="val",
        require_audio=require_audio,
        max_frames=int(section["max_frames"]),
        verbose=is_main(),
    )
    if len(train_set) == 0 or len(val_set) == 0:
        raise RuntimeError("empty split: run preprocess/extract_mesh.py and preprocess/extract_audio.py first")
    if is_main() and task == "attribution":
        present = {int(s["label"]) for s in train_set.samples}
        missing = [GENERATOR_NAMES[i] for i in range(num_classes) if i not in present]
        if missing:
            logger.warning(f"attribute train missing classes: {missing}")

    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if distributed else None
    drop_last = bool(section.get("drop_last", task != "attribution"))
    if distributed:
        drop_last = True
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=train_sampler is None, sampler=train_sampler,
        num_workers=int(cfg["train"]["num_workers"]), pin_memory=True, collate_fn=av_collate,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, sampler=val_sampler,
        num_workers=max(1, int(cfg["train"]["num_workers"]) // 2), pin_memory=True, collate_fn=av_collate,
    )
    model = wrap_model(AVTSTAN(num_classes=num_classes), device, gpu_ids, distributed, local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(cfg["train"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    writer = SummaryWriter(os.path.join(log_dir, "tb", datetime.now().strftime("%Y%m%d_%H%M%S"))) if is_main() else None
    best = 0.0
    patience = int(section.get("patience", 4))
    stale = 0
    ckpt_name = "best_model.pth"
    select_key = str(section.get("select", "auc" if task == "audiovisual" else "macro_f1"))
    label_smoothing = float(section.get("label_smoothing", 0.0))
    class_weight = None
    if task == "attribution" and bool(section.get("class_weight", True)):
        class_weight = class_balanced_weights(
            [int(s["label"]) for s in train_set.samples], num_classes
        ).to(device)
    resume = getattr(args, "resume", "") or ""
    init_from = getattr(args, "init_from_audiovisual", "") or section.get("init_from_audiovisual", "") or ""
    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location=device)
        load_state_dict(model, ckpt["model"])
        prev = ckpt.get("val") or {}
        best = float(prev.get(select_key, prev.get("auc" if task == "audiovisual" else "macro_f1", 0.0)))
        if is_main():
            logger.info(
                f"resumed weights from {resume} prev_best={best:.4f} select={select_key}; "
                f"fresh cosine T_max={epochs} lr={lr} (optimizer not reused)"
            )
    elif task == "attribution" and init_from and os.path.isfile(init_from):
        av_ckpt = torch.load(init_from, map_location=device)
        n_ok, n_all = load_matching_weights(model, av_ckpt.get("model") or av_ckpt)
        if is_main():
            logger.info(f"init attribution from audiovisual {init_from}: loaded {n_ok}/{n_all} tensors (skip mismatched heads)")
    if is_main():
        steps = max(len(train_loader), 1)
        extra = ""
        if class_weight is not None:
            extra = " class_w=" + ",".join(f"{x:.3f}" for x in class_weight.detach().cpu().tolist())
        logger.info(
            f"train_clips={len(train_set)} val_clips={len(val_set)} "
            f"steps/epoch={steps} drop_last={drop_last}{extra} tensorboard={writer.log_dir}"
        )
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_m = run_av_epoch(
            model, train_loader, device, optimizer,
            class_weight=class_weight, label_smoothing=label_smoothing,
        )
        val_m = run_av_epoch(model, val_loader, device, optimizer=None, class_weight=class_weight)
        scheduler.step()
        score = float(val_m.get(select_key, val_m.get("auc" if task == "audiovisual" else "macro_f1", 0.0)))
        if is_main():
            extra = {"lr": float(optimizer.param_groups[0]["lr"])}
            logger.info(
                f"{task} epoch {epoch}/{epochs - 1} | "
                f"train loss {train_m['loss']:.4f} acc {train_m['acc']:.4f} f1 {train_m['macro_f1']:.4f} "
                f"auc {train_m.get('auc', 0):.4f} | "
                f"val loss {val_m['loss']:.4f} acc {val_m['acc']:.4f} f1 {val_m['macro_f1']:.4f} "
                f"auc {val_m.get('auc', 0):.4f} lr={extra['lr']:.6g} select={select_key}={score:.4f}"
            )
            append_metrics(metrics_path, {"epoch": epoch, "split": "train", **train_m, **extra})
            append_metrics(metrics_path, {"epoch": epoch, "split": "val", **val_m, **extra})
            for k, v in train_m.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_m.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            writer.add_scalar("params/lr", extra["lr"], epoch)
            payload = {"model": state_dict_to_save(model), "epoch": epoch, "val": val_m, "num_classes": num_classes, "config": cfg}
            torch.save(payload, os.path.join(save_dir, "last_model.pth"))
            if score >= best:
                best = score
                stale = 0
                torch.save(payload, os.path.join(save_dir, ckpt_name))
                logger.info(f"saved {ckpt_name} score={best:.4f}")
            else:
                stale += 1
                logger.info(f"no val score gain ({stale}/{patience})")
        should_stop = patience > 0 and stale >= patience
        if distributed:
            stop = torch.tensor([int(should_stop)], device=device)
            torch.distributed.broadcast(stop, src=0)
            should_stop = bool(stop.item())
        if should_stop:
            if is_main():
                logger.info(f"early stop at epoch {epoch}")
            break
    if writer is not None:
        writer.close()
    if is_main():
        logger.info(f"{task} done. best={best:.4f}")
    cleanup()
