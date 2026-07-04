# 包含训练集、验证集、测试集  第二步
import os
import cv2
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ===== 修改部分：增加测试集划分 =====
def balanced_split_train_val_test(items, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    保持 real / fake 比例一致地划分训练集、验证集和测试集
    """
    # 确保比例总和小于1
    assert train_ratio + val_ratio < 1, "The sum of train_ratio and val_ratio must be less than 1."
    
    by_label = {0: [], 1: []}
    for it in items:
        # 确保标签存在于字典中
        if it['label'] not in by_label:
            by_label[it['label']] = []
        by_label[it['label']].append(it)

    train_items, val_items, test_items = [], [], []
    random.seed(seed)
    
    for label in by_label:
        lst = by_label[label]
        random.shuffle(lst)
        
        n_total = len(lst)
        train_idx = int(n_total * train_ratio)
        val_idx = int(n_total * (train_ratio + val_ratio))
        
        train_items.extend(lst[:train_idx])
        val_items.extend(lst[train_idx:val_idx])
        test_items.extend(lst[val_idx:]) # 剩余部分作为测试集

    # 最后再整体 shuffle，防止 real/fake 样本顺序扎堆
    random.shuffle(train_items)
    random.shuffle(val_items)
    random.shuffle(test_items)
    
    return train_items, val_items, test_items


# ===== 以下函数与原代码相同，无需修改 =====
def extract_frames(video_path, out_dir, fps=25):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        idx += 1
    cap.release()
    return idx

def build_combos(frame_dir, combo_len=5, save_dir=None):
    frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    combos = []
    for i in range(len(frames) - combo_len + 1):
        combo_frames = frames[i:i+combo_len]
        if save_dir:
            combo_out_dir = os.path.join(save_dir, f"combo_{i:06d}")
            os.makedirs(combo_out_dir, exist_ok=True)
            for cf in combo_frames:
                src = os.path.join(frame_dir, cf)
                dst = os.path.join(combo_out_dir, cf)
                if not os.path.exists(dst):
                    import shutil
                    shutil.copy(src, dst)
        combos.append(combo_frames)
    return combos


def process_split(items, split_name, out_root, combo_len=5):
    """
    对训练集、验证集或测试集做统一处理
    """
    combo_index = []
    for item in tqdm(items, desc=f"Processing {split_name}"):
        video_path = item['video']
        vid_name = Path(video_path).stem
        vid_dir = os.path.join(out_root, split_name, vid_name)
        frame_dir = os.path.join(vid_dir, 'frames')

        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir, exist_ok=True)
            extract_frames(video_path, frame_dir)

        combo_dir = os.path.join(vid_dir, 'combos')
        combos = build_combos(frame_dir, combo_len=combo_len, save_dir=combo_dir)

        for idx, _ in enumerate(combos):
            combo_path = os.path.join(combo_dir, f"combo_{idx:06d}")
            combo_index.append({
                "combo_path": combo_path,
                "label": item['label'],
                "video": vid_name,
                "start_frame": idx
            })

    index_path = os.path.join(out_root, f"{split_name}_combo_index.json")
    with open(index_path, 'w') as f:
        json.dump(combo_index, f, indent=2)
    print(f"Saved {split_name} combo index ({len(combo_index)} combos) to {index_path}")
    return index_path


# ===== 修改部分：主流程增加测试集处理 =====
def preprocess_dataset(list_json, out_root, train_ratio=0.8, val_ratio=0.1, combo_len=5):
    with open(list_json, 'r') as f:
        all_items = json.load(f)

    # 调用新的三向划分函数
    train_items, val_items, test_items = balanced_split_train_val_test(
        all_items, train_ratio=train_ratio, val_ratio=val_ratio, seed=42
    )

    os.makedirs(out_root, exist_ok=True)

    # 分别处理训练集、验证集和测试集
    process_split(train_items, "train", out_root, combo_len)
    process_split(val_items,   "val",   out_root, combo_len)
    process_split(test_items,  "test",  out_root, combo_len) # 新增对测试集的处理

    # 额外保存三份纯视频列表，便于调试
    with open(os.path.join(out_root, "train_videos.json"), 'w') as f:
        json.dump(train_items, f, indent=2)
    with open(os.path.join(out_root, "val_videos.json"), 'w') as f:
        json.dump(val_items, f, indent=2)
    with open(os.path.join(out_root, "test_videos.json"), 'w') as f: # 新增保存测试集列表
        json.dump(test_items, f, indent=2)

    # 更新打印信息以包含测试集
    print("\n" + "="*40)
    print(f"Dataset splitting finished!")
    print(f"Total videos: {len(all_items)}")
    print("-" * 40)
    print(f"Train videos: {len(train_items):>5} "
          f"(real={sum(it['label']==0 for it in train_items)}, "
          f"fake={sum(it['label']==1 for it in train_items)})")
    print(f"Val videos:   {len(val_items):>5} "
          f"(real={sum(it['label']==0 for it in val_items)}, "
          f"fake={sum(it['label']==1 for it in val_items)})")
    print(f"Test videos:  {len(test_items):>5} "
          f"(real={sum(it['label']==0 for it in test_items)}, "
          f"fake={sum(it['label']==1 for it in test_items)})")
    print("="*40)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess videos into frame combos')
    parser.add_argument('--list_json', type=str, default='talkheadbench/talkheadbench_lists.json',
                        help='Video list JSON generated by dataset.py')
    parser.add_argument('--out_root', type=str, default='talkheadbench',
                        help='Output directory for preprocessed data')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--combo_len', type=int, default=5)
    args = parser.parse_args()

    set_seed(42)
    preprocess_dataset(
        args.list_json, args.out_root,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, combo_len=args.combo_len
    )



