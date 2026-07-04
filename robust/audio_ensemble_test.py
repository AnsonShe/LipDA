import sys
import os

_LIPDA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _LIPDA_ROOT not in sys.path:
    sys.path.insert(0, _LIPDA_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import pickle
import argparse
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score)

# ----------------- 模型定义导入 (与之前相同) -----------------
from models import LipEncoder, PoseEncoder, DetectionHead
from attribute_data.model_attribution import AVTSTAN
# 'import distortion' 已被移除，不再需要

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# +++ 核心评估函数 (已修改) +++
def evaluate_audio_robustness(args, audio_distortion_type):
    """
    评估模型在特定音频扰动下的性能。
    
    Args:
        args: 命令行参数。
        audio_distortion_type (str): 当前要评估的音频扰动类型 (例如 'baseline', 'noise_light')。
    """
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 1. 加载阶段一 (Vision-Only) 模型 (无变化) ---
    s1_lip_enc = LipEncoder().to(device)
    s1_pose_enc = PoseEncoder(lstm_hidden_dim=args.s1_lstm_hidden).to(device)
    s1_det_head_in_dim = (5 * 128) + s1_pose_enc.output_dim
    s1_det_head = DetectionHead(s1_det_head_in_dim).to(device)
    checkpoint_s1 = torch.load(args.s1_checkpoint, map_location=device, weights_only=False)
    s1_lip_enc.load_state_dict(checkpoint_s1['lip_enc'])
    s1_pose_enc.load_state_dict(checkpoint_s1['pose_enc'])
    s1_det_head.load_state_dict(checkpoint_s1['det_head'])
    s1_lip_enc.eval(); s1_pose_enc.eval(); s1_det_head.eval()

    # --- 2. 加载阶段二 (Multi-Modal Robust) 模型 (无变化) ---
    model_s2 = AVTSTAN(num_classes=2).to(device)
    model_s2.load_state_dict(torch.load(args.s2_checkpoint, map_location=device, weights_only=False))
    model_s2.eval()

    # --- 3. 加载数据集定义和特征缓存 (无变化) ---
    with open(args.test_json, 'r') as f:
        test_video_list = json.load(f)
    
    print(f"正在从 {args.s1_cache_path} 加载阶段一特征...")
    s1_cached_features = np.load(args.s1_cache_path, allow_pickle=True).item()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # --- 4. 视觉扰动逻辑已被完全移除 ---
    
    # --- 5. 循环评估 ---
    ground_truths, final_predictions, final_scores = [], [], []

    for video_info in tqdm(test_video_list, desc=f"Audio Distortion: {audio_distortion_type}"):
        video_name = video_info['name']
        true_label = video_info['label']

        # --- 5.1 阶段一模型推理 (无变化，永远使用干净的视觉特征) ---
        prob_s1 = 0.5 # 默认概率
        try:
            video_combo_base_path = os.path.join(args.s1_preprocessed_dir, 'test', video_name, 'combos')
            available_combos = sorted(os.listdir(video_combo_base_path))
            num_to_sample = min(len(available_combos), 30)
            sampled_combo_names = np.random.choice(available_combos, num_to_sample, replace=False)
            
            combo_probs_s1 = []
            for combo_name in sampled_combo_names:
                combo_key = os.path.join(os.path.basename(args.s1_preprocessed_dir), 'test', video_name, 'combos', combo_name)
                lip_images_s1_np, landmarks_s1_np = s1_cached_features['lip_images'][combo_key], s1_cached_features['landmarks'][combo_key]
                
                # 视觉扰动代码块已被移除
                lip_tensor = torch.stack([transform(img) for img in lip_images_s1_np]).to(device)
                lm_tensor = torch.tensor(landmarks_s1_np, dtype=torch.float32).to(device)

                with torch.no_grad():
                    lip_features, _ = s1_lip_enc(lip_tensor.unsqueeze(0))
                    pose_features, _ = s1_pose_enc(lm_tensor.unsqueeze(0))
                    fused = torch.cat([lip_features, pose_features], dim=1)
                    logits = s1_det_head(fused)
                    probs = F.softmax(logits, dim=1)[:, 1]
                    combo_probs_s1.extend(probs.cpu().numpy())
            
            if combo_probs_s1:
                prob_s1 = np.mean(combo_probs_s1)
        except Exception as e:
            pass 

        # --- 5.2 阶段二模型推理 (核心修改点) ---
        final_score = prob_s1 
        
        # 动态构建指向特定音频扰动特征的路径
        s2_feature_path = os.path.join(args.s2_cache_dir_base, audio_distortion_type, f"{video_name}.pkl")

        if os.path.exists(s2_feature_path):
            with open(s2_feature_path, 'rb') as f:
                s2_features = pickle.load(f)

            mfcc_s2 = s2_features['mfcc']
            if mfcc_s2 is not None and np.any(mfcc_s2):
                lip_images_s2_np = s2_features['lip_roi']
                landmarks_s2_np = s2_features['landmarks']

                # 视觉扰动代码块已被移除
                lip_s2 = torch.FloatTensor(lip_images_s2_np.transpose(0, 3, 1, 2)).unsqueeze(0).to(device) / 255.0
                landmarks_s2 = torch.FloatTensor(landmarks_s2_np).unsqueeze(0).to(device)
                mfcc_s2 = torch.FloatTensor(mfcc_s2).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits_s2, _ = model_s2(landmarks_s2, lip_s2, mfcc_s2)
                    prob_s2 = F.softmax(logits_s2, dim=1)[:, 1].item()
                
                # 执行集成 (无变化)
                final_score = args.w1 * prob_s1 + args.w2 * prob_s2
            
        ground_truths.append(true_label)
        final_scores.append(final_score)
        final_predictions.append(1 if final_score >= 0.5 else 0)

    # --- 6. 计算并打印性能指标 (修改了打印内容) ---
    acc = accuracy_score(ground_truths, final_predictions)
    f1 = f1_score(ground_truths, final_predictions, zero_division=0)
    auc = roc_auc_score(ground_truths, final_scores) if len(np.unique(ground_truths)) > 1 else 0.0
    
    print("-" * 50)
    print(f"音频扰动类型: {audio_distortion_type}")
    print(f"处理视频数: {len(ground_truths)}")
    print(f"融合权重: w1(S1)={args.w1}, w2(S2)={args.w2}")
    print(f"Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | AUC: {auc:.4f}")
    print("-" * 50)


# +++ 主执行逻辑 (已修改) +++
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble Audio Robustness Evaluation Script.')
    # 路径参数
    parser.add_argument('--s1_checkpoint', type=str, default='Ours/AVLip_checkpoints/finetuned_best_model.pth', help='Path to Stage 1 model checkpoint')
    parser.add_argument('--s2_checkpoint', type=str, default='attribute_data/best_2class.pth', help='Path to Stage 2 robust detector checkpoint')
    parser.add_argument('--test_json', type=str, default='avlip/test_videos.json', help='Path to the test set JSON file')
    parser.add_argument('--s1_cache_path', type=str, default='avlip/test_cache.npy', help='Path to Stage 1 features cache (.npy)')
    parser.add_argument('--s1_preprocessed_dir', type=str, default='avlip', help='Path to Stage 1 preprocessed combo directories')
    parser.add_argument('--s2_cache_dir_base', type=str, default='attribute_data/features_cache_audio_distorted', help='Base directory containing S2 feature caches, with subdirs for each audio distortion type')
    # 模型参数
    parser.add_argument('--s1_lstm_hidden', type=int, default=256, help='Hidden dim for Stage 1 PoseEncoder LSTM')
    # 融合参数
    parser.add_argument('--w1', type=float, default=0.4, help='Weight for Stage 1 model')
    parser.add_argument('--w2', type=float, default=0.6, help='Weight for Stage 2 model')
    args = parser.parse_args()

    # 定义要测试的音频扰动 (与特征生成脚本的输出目录名对应)
    AUDIO_DISTORTION_TYPES = [
        'baseline',
        'noise_heavy',
        'noise_light',
        'pitch_shift_down3',
        'pitch_shift_up3',
        'resample_16k',
        'resample_44k',
        'rir_large_room',
        'rir_medium_room',
        'rir_small_room',
        'silence_random',
        'time_shift_backward',
        'time_shift_forward'
    ]
    
    print("="*60)
    print("         STARTING ENSEMBLE AUDIO ROBUSTNESS EVALUATION")
    print("="*60)

    # 循环遍历所有音频扰动
    for dist_type in AUDIO_DISTORTION_TYPES:
        evaluate_audio_robustness(args, audio_distortion_type=dist_type)
            
    print("\n" + "="*60)
    print("            AUDIO ROBUSTNESS EVALUATION COMPLETE")
    print("="*60)