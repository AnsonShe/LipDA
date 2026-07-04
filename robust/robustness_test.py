# robustness_test.py
import sys
import os

_LIPDA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_ROBUST_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_LIPDA_ROOT, _ROBUST_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import dlib
import argparse
import torch.nn.functional as F
import random
from collections import Counter
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             confusion_matrix, accuracy_score, f1_score,
                             precision_score, recall_score)
from models import LipEncoder, PoseEncoder, DetectionHead

# 导入您提供的扰动模块
import distortion

# ... (set_seed 和 FeatureExtractor 类的定义保持不变) ...
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")

class FeatureExtractor:
    def __init__(self, dlib_predictor_path):
        print("初始化即时特征提取器 (dlib)...")
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor(dlib_predictor_path)
        except RuntimeError as e:
            print(f"错误: 无法加载dlib模型 '{dlib_predictor_path}'。")
            print("请确保文件存在且路径正确。")
            exit()

    def extract_features_from_combo(self, combo_path):
        frame_files = sorted(os.listdir(combo_path))
        if len(frame_files) == 0: return None, None
        lip_images, landmarks = [], []
        for frame_file in frame_files:
            frame_path = os.path.join(combo_path, frame_file)
            img = cv2.imread(frame_path)
            if img is None: return None, None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dets = self.detector(img, 1)
            lip_img, lms = None, None
            if len(dets) > 0:
                d = dets[0]
                shape = self.predictor(img, d)
                lms = np.array([[p.x, p.y] for p in shape.parts()])
                face_height = d.bottom() - d.top()
                crop_height = int(face_height * 0.45)
                y1 = max(0, d.bottom() - crop_height)
                y2 = d.bottom()
                x_center = (d.left() + d.right()) // 2
                crop_width = int(face_height * 0.45 * 1.5)
                x1 = max(0, x_center - crop_width // 2)
                x2 = min(img.shape[1], x_center + crop_width // 2)
                lip_img_crop = img[y1:y2, x1:x2]
                if lip_img_crop.size > 0: lip_img = cv2.resize(lip_img_crop, (96, 96))
                else: lip_img = np.zeros((96, 96, 3), dtype=np.uint8)
            else:
                h, w = img.shape[:2]
                y_start = int(h * 0.55)
                lip_img_crop = img[y_start:h, 0:w]
                if lip_img_crop.size > 0: lip_img = cv2.resize(lip_img_crop, (96, 96))
                else: lip_img = np.zeros((96, 96, 3), dtype=np.uint8)
                lms = np.zeros((68, 2))
            lip_images.append(lip_img)
            landmarks.append(lms)
        return lip_images, np.array(landmarks)

# +++ MODIFIED: `inference` 函数现在接受扰动参数 +++
def inference(args, distortion_type=None, distortion_level=None):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # --- 模型加载 ---
    lip_enc = LipEncoder().to(device)
    pose_enc = PoseEncoder(lstm_hidden_dim=args.lstm_hidden).to(device)
    det_head_in_dim = (5 * 128) + pose_enc.output_dim
    det_head = DetectionHead(det_head_in_dim).to(device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        lip_enc.load_state_dict(checkpoint['lip_enc'])
        pose_enc.load_state_dict(checkpoint['pose_enc'])
        det_head.load_state_dict(checkpoint['det_head'])
    except Exception as e:
        print(f"错误: 加载模型检查点失败: {e}")
        return
    lip_enc.eval(); pose_enc.eval(); det_head.eval()
    
    # --- 特征加载/准备 ---
    feature_extractor = None
    cached_data = {'lip_images': {}, 'landmarks': {}}
    cache_exists = os.path.exists(args.cache_file)
    if cache_exists:
        cached_data = np.load(args.cache_file, allow_pickle=True).item()
    else:
        feature_extractor = FeatureExtractor(dlib_predictor_path=args.dlib_predictor)

    with open(args.test_videos_json, 'r') as f:
        test_video_list = json.load(f)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    video_gts, video_preds, video_scores = [], [], []

    # 如果指定了扰动，获取扰动函数和参数
    dist_func, dist_param = None, None
    if distortion_type and distortion_level:
        print(f"应用扰动: {distortion_type}, 等级: {distortion_level}")
        dist_func = distortion.get_distortion_function(distortion_type)
        dist_param = distortion.get_distortion_parameter(distortion_type, distortion_level)

    # 主循环 (为了减少重复输出，移除了内部的 tqdm)
    for video_info in test_video_list:
        video_name = video_info['name']
        true_label = video_info['label']
        video_combo_base_path = os.path.join(args.preprocessed_dir, 'test', video_name, 'combos')
        
        if not os.path.exists(video_combo_base_path) or not os.listdir(video_combo_base_path):
            continue
            
        available_combos = sorted(os.listdir(video_combo_base_path))
        num_to_sample = min(len(available_combos), args.num_combos_per_video)
        sampled_combo_names = random.sample(available_combos, num_to_sample)
        
        lip_tensors, lm_tensors = [], []
        for combo_name in sampled_combo_names:
            relative_base = os.path.basename(args.preprocessed_dir)
            combo_key = os.path.join(relative_base, 'test', video_name, 'combos', combo_name)
            lip_images_np, landmarks_np = None, None
            
            if combo_key in cached_data['lip_images']:
                lip_images_np, landmarks_np = cached_data['lip_images'][combo_key], cached_data['landmarks'][combo_key]
            elif feature_extractor:
                full_combo_path = os.path.join(video_combo_base_path, combo_name)
                lip_images_np, landmarks_np = feature_extractor.extract_features_from_combo(full_combo_path)
            
            if lip_images_np is not None:
                # +++ NEW: 在此应用指定的扰动 +++
                if dist_func and dist_param:
                    distorted_images = []
                    for img_rgb in lip_images_np:
                        # 转换到 BGR 以匹配 distortion 函数的输入
                        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                        # 应用扰动
                        distorted_bgr = dist_func(img_bgr, dist_param)
                        # 转换回 RGB 以进行后续处理
                        distorted_rgb = cv2.cvtColor(distorted_bgr, cv2.COLOR_BGR2RGB)
                        distorted_images.append(distorted_rgb)
                    lip_images_np = np.array(distorted_images)

                lip_tensor = torch.stack([transform(img) for img in lip_images_np])
                lm_tensor = torch.tensor(landmarks_np, dtype=torch.float32)
                lip_tensors.append(lip_tensor)
                lm_tensors.append(lm_tensor)
        
        if not lip_tensors:
            continue
            
        lip_batch_tensor, lm_batch_tensor = torch.stack(lip_tensors).to(device), torch.stack(lm_tensors).to(device)
        combo_dataset = TensorDataset(lip_batch_tensor, lm_batch_tensor)
        combo_loader = DataLoader(combo_dataset, batch_size=args.batch_size, shuffle=False)
        
        combo_preds, combo_probs = [], []
        with torch.no_grad():
            for lip_batch, lm_batch in combo_loader:
                lip_features_cat, _ = lip_enc(lip_batch)
                pose_features_lstm, _ = pose_enc(lm_batch)
                fused = torch.cat([lip_features_cat, pose_features_lstm], dim=1)
                logits = det_head(fused)
                probs = F.softmax(logits, dim=1)[:, 1]
                _, predicted = torch.max(logits, 1)
                combo_preds.extend(predicted.cpu().numpy())
                combo_probs.extend(probs.cpu().numpy())

        fake_ratio = np.sum(combo_preds) / len(combo_preds)
        video_prediction = 1 if fake_ratio >= args.threshold else 0
        video_score = np.mean(combo_probs)
        
        video_gts.append(true_label)
        video_preds.append(video_prediction)
        video_scores.append(video_score)

    # --- 结果评估 ---
    print("\n" + "="*30 + "\n  Video-Level Performance Results\n" + "="*30)
    if not video_gts:
        print("未能处理任何视频，无法计算指标。")
        return

    final_label_counts = Counter(video_gts)
    num_unique_labels = len(final_label_counts)

    acc = accuracy_score(video_gts, video_preds)
    precision = precision_score(video_gts, video_preds, zero_division=0)
    recall = recall_score(video_gts, video_preds, zero_division=0)
    f1 = f1_score(video_gts, video_preds, zero_division=0)
    
    if num_unique_labels > 1:
        auc = roc_auc_score(video_gts, video_scores)
        ap = average_precision_score(video_gts, video_scores)
    else:
        auc = float('nan')
        ap = float('nan')

    tn, fp, fn, tp = confusion_matrix(video_gts, video_preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    print(f"Total Videos Processed: {len(video_gts)}")
    print(f"Accuracy:  {acc:.4f}, AUC:       {auc:.4f}, AP:        {ap:.4f}")
    print(f"F1-Score:  {f1:.4f}, Precision: {precision:.4f}, Recall:    {recall:.4f}")
    print(f"FPR:       {fpr:.4f}, FNR:       {fnr:.4f}")
    print("="*30 + "\n")


# +++ NEW: 主执行逻辑用于鲁棒性测试 +++
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Robustness evaluation script.')
    # 保留所有原始参数
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--checkpoint', type=str, default='Ours/AVLip_checkpoints/finetuned_best_model.pth', help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--test_videos_json', type=str, default='avlip/test_videos.json', help='Path to the test set JSON file')
    parser.add_argument('--preprocessed_dir', type=str, default='avlip', help='Path to the root of preprocessed data')
    parser.add_argument('--cache_file', type=str, default='avlip/test_cache.npy', help='Path to read/write the .npy cache file')
    parser.add_argument('--dlib_predictor', type=str, default='shape_predictor_68_face_landmarks.dat', help='Path to dlib`s facial landmark predictor')
    parser.add_argument('--num_combos_per_video', type=int, default=30, help='Number of combos to sample from each video')
    parser.add_argument('--threshold', type=float, default=0.5, help='Voting threshold for video-level prediction')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for model inference')
    parser.add_argument('--lstm_hidden', type=int, default=256, help='Hidden dim for PoseEncoder LSTM (must match training)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference ("cuda" or "cpu")')
    # 注意: --save_cache 在这里意义不大，因为我们不想缓存受扰动的图像
    
    args = parser.parse_args()

    # 定义要测试的扰动类型和级别
    # 注意: 'VC' (视频压缩) 作用于文件，不适用于此处的帧级实时处理，故排除
    DISTORTION_TYPES = ['CS', 'CC', 'BW', 'GN', 'GB', 'JPEG', 'PXL']
    DISTORTION_LEVELS = [1, 2, 3, 4, 5]

    print("="*50)
    print("           STARTING ROBUSTNESS EVALUATION")
    print("="*50)

    # 首先，运行一次无扰动的基线评估 (baseline)
    print("\n" + "*"*20 + " RUNNING BASELINE (NO DISTORTION) " + "*"*20)
    inference(args, distortion_type=None, distortion_level=None)

    # 循环遍历所有扰动类型和级别
    for dist_type in DISTORTION_TYPES:
        for level in DISTORTION_LEVELS:
            header = f" EVALUATING: {dist_type} | Level: {level} "
            print("\n" + "*"*20 + header + "*"*20)
            
            # 调用 inference 函数并传入当前扰动参数
            inference(args, distortion_type=dist_type, distortion_level=level)
            
    print("\n" + "="*50)
    print("           ROBUSTNESS EVALUATION COMPLETE")
    print("="*50)