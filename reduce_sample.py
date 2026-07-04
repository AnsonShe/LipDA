# 每个视频挑选N个组合帧，减少样本的数量，以供训练， 第三步
import os
import json
import random
from pathlib import Path
from tqdm import tqdm

def set_seed(seed=42):
    """设置所有相关的随机种子保证可复现性"""
    random.seed(seed)

def create_reduced_index(original_index_path, output_path, samples_per_video=1, seed=42):
    """
    从原始组合帧索引中，每个视频随机选取指定数量的组合帧，创建精简索引
    
    参数:
        original_index_path: 原始组合帧索引JSON文件路径
        output_path: 新生成的精简索引保存路径
        samples_per_video: 每个视频选取的组合帧数量 (默认1)
        seed: 随机种子 (默认42)
    """
    # 设置随机种子
    set_seed(seed)
    
    with open(original_index_path, 'r') as f:
        original_index = json.load(f)
    
    # 按视频分组所有组合帧
    video_to_combos = {}
    for combo in original_index:
        vid_name = combo['video']
        if vid_name not in video_to_combos:
            video_to_combos[vid_name] = []
        video_to_combos[vid_name].append(combo)
    
    # 从每个视频中随机选取指定数量的组合帧
    reduced_index = []
    for vid_name, combos in video_to_combos.items():
        # 先对每个视频的组合帧列表进行固定随机排序
        random.Random(seed + hash(vid_name) % 1000).shuffle(combos)
        selected_combos = combos[:min(samples_per_video, len(combos))]
        reduced_index.extend(selected_combos)
    
    # 保存精简后的索引
    with open(output_path, 'w') as f:
        json.dump(reduced_index, f, indent=2)
    
    print(f"Reduced index saved to {output_path} (from {len(original_index)} to {len(reduced_index)} combos)")

def reduce_all_indexes(original_data_dir, reduced_data_dir, samples_per_video=1, seed=42):
    """
    处理原始预处理数据目录，为训练集和验证集都创建精简索引
    
    参数:
        original_data_dir: 原始预处理数据目录(包含train_combo_index.json等)
        reduced_data_dir: 精简索引保存目录
        samples_per_video: 每个视频选取的组合帧数量 (默认1)
        seed: 随机种子 (默认42)
    """
    os.makedirs(reduced_data_dir, exist_ok=True)
    
    # 设置全局随机种子
    set_seed(seed)
    
    # 处理训练集和验证集
    for split in ['train', 'val']:
        original_path = os.path.join(original_data_dir, f"{split}_combo_index.json")
        reduced_path = os.path.join(reduced_data_dir, f"{split}_combo_index_reduced.json")
        
        if os.path.exists(original_path):
            create_reduced_index(
                original_index_path=original_path,
                output_path=reduced_path,
                samples_per_video=samples_per_video,
                seed=seed  # 传递相同的种子
            )
        else:
            print(f"Warning: Original index file not found - {original_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Reduce combo index by sampling N combos per video')
    parser.add_argument('--original_data_dir', type=str, default='avlip',
                        help='Directory containing train/val combo index JSON files')
    parser.add_argument('--reduced_data_dir', type=str, default='reduced_data',
                        help='Output directory for reduced index files')
    parser.add_argument('--samples_per_video', type=int, default=4,
                        help='Number of combos to sample per video')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    reduce_all_indexes(
        original_data_dir=args.original_data_dir,
        reduced_data_dir=args.reduced_data_dir,
        samples_per_video=args.samples_per_video,
        seed=args.seed,
    )