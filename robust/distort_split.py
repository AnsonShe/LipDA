"""Apply a visual distortion to every video in an ours split and write a new JSON list."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm

from common.utils import generator_of, load_config, repo_file, resolve_video_path, safe_video_name
from robust.video_distort import DIST_TYPES, distort_video, write_meta


def main():
    parser = argparse.ArgumentParser(description="Distort all videos in a split")
    parser.add_argument("--config", default=repo_file("configs", "default.yaml"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--json", default="", help="override split json")
    parser.add_argument("--type", required=True, help="CS|CC|BW|GNC|GB|JPEG|PXL|random")
    parser.add_argument("--level", default="3", help="1–5 or random")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    json_path = args.json or cfg["data"][{"train": "train_json", "val": "val_json", "test": "test_json"}[args.split]]
    items = json.load(open(json_path))
    if args.limit:
        items = items[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, "meta.txt")
    out_items = []
    video_root = cfg.get("data", {}).get("video_root", "")
    for item in tqdm(items, desc=f"distort {args.type}:{args.level}"):
        src = resolve_video_path(item, video_root)
        name = safe_video_name(item)
        dst = os.path.join(args.out_dir, f"{name}.mp4")
        if not os.path.isfile(src):
            print(f"missing {src}")
            continue
        dist_type, dist_level = distort_video(src, dst, args.type, args.level)
        write_meta(meta_path, src, dst, dist_type, dist_level)
        rec = dict(item)
        rec["orig_video"] = item["video"]
        rec["video"] = os.path.abspath(dst)
        rec["name"] = name
        rec["label"] = int(item["label"])
        rec["generator"] = generator_of(item)
        rec["distortion"] = f"{dist_type}:{dist_level}"
        out_items.append(rec)
    out_json = os.path.join(args.out_dir, "split.json")
    with open(out_json, "w") as f:
        json.dump(out_items, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(out_items)} videos and {out_json}")
    print("next: python preprocess/extract_mesh.py --list", out_json, "--cache_dir <new_cache>")
    print("      python preprocess/extract_audio.py --list", out_json, "--cache_dir <new_cache>")


if __name__ == "__main__":
    main()
