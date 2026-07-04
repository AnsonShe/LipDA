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
    # --- NO CHANGE ---
    # (根据您的要求，此类保持原样，完全不变)
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
            
            ####  self.detector(img, 0)   采样从1改为0
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

def inference(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 模型加载 (无变化) ---
    lip_enc = LipEncoder().to(device)
    pose_enc = PoseEncoder(lstm_hidden_dim=args.lstm_hidden).to(device)
    det_head_in_dim = (5 * 128) + pose_enc.output_dim
    det_head = DetectionHead(det_head_in_dim).to(device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        lip_enc.load_state_dict(checkpoint['lip_enc'])
        pose_enc.load_state_dict(checkpoint['pose_enc'])
        det_head.load_state_dict(checkpoint['det_head'])
        print(f"模型权重从 '{args.checkpoint}' 加载成功。")
    except Exception as e:
        print(f"错误: 加载模型检查点失败: {e}")
        return
    lip_enc.eval(); pose_enc.eval(); det_head.eval()
    
    # --- 特征加载/准备 (无变化) ---
    feature_extractor = None
    cached_data = {'lip_images': {}, 'landmarks': {}}
    cache_exists = os.path.exists(args.cache_file)
    if cache_exists:
        print(f"发现缓存文件，正在从 '{args.cache_file}' 加载...")
        cached_data = np.load(args.cache_file, allow_pickle=True).item()
        print("缓存加载成功。")
    else:
        print(f"警告: 未找到缓存文件 '{args.cache_file}'。")
        print("将进行即时特征提取，这会比较慢。")
        feature_extractor = FeatureExtractor(dlib_predictor_path=args.dlib_predictor)

    with open(args.test_videos_json, 'r') as f:
        test_video_list = json.load(f)

    # +++ 诊断 1 (无变化) +++
    initial_labels = [item['label'] for item in test_video_list]
    print(f"\n诊断信息: 原始JSON文件包含 {len(initial_labels)} 个视频。")
    print(f"标签分布: {Counter(initial_labels)}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    video_gts, video_preds, video_scores = [], [], []

    for video_info in tqdm(test_video_list, desc="Testing Videos"):
        video_name = video_info['name']
        true_label = video_info['label']
        video_combo_base_path = os.path.join(args.preprocessed_dir, 'test', video_name, 'combos')
        
        # +++ 诊断 2 (无变化) +++
        if not os.path.exists(video_combo_base_path):
            print(f"\n[跳过] 视频 '{video_name}' (标签: {true_label}): 预处理目录不存在。")
            continue
        available_combos = sorted(os.listdir(video_combo_base_path))
        if not available_combos:
            print(f"\n[跳过] 视频 '{video_name}' (标签: {true_label}): 组合帧目录为空。")
            continue
        
        # --- MODIFIED: 移除随机采样，处理所有 Combos ---
        num_to_sample = min(len(available_combos), args.num_combos_per_video)
        sampled_combo_names = random.sample(available_combos, num_to_sample)
        combos_to_process = available_combos # 直接处理所有可用的 combos
        
        lip_tensors, lm_tensors = [], []
        # --- MODIFIED: 循环变量名变更 ---
        # for combo_name in combos_to_process:
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
                    cached_data['lip_images'][combo_key] = lip_images_np
                    cached_data['landmarks'][combo_key] = landmarks_np
            
            if lip_images_np is not None:
                lip_tensor = torch.stack([transform(img) for img in lip_images_np])
                lm_tensor = torch.tensor(landmarks_np, dtype=torch.float32)
                lip_tensors.append(lip_tensor)
                lm_tensors.append(lm_tensor)
        
        # +++ 诊断 3 (无变化) +++
        if not lip_tensors:
            print(f"\n[跳过] 视频 '{video_name}' (标签: {true_label}): 未能从 {len(combos_to_process)} 个组合帧中成功提取任何特征。")
            continue
        
        lip_batch_tensor, lm_batch_tensor = torch.stack(lip_tensors).to(device), torch.stack(lm_tensors).to(device)
        combo_dataset = TensorDataset(lip_batch_tensor, lm_batch_tensor)
        combo_loader = DataLoader(combo_dataset, batch_size=args.batch_size, shuffle=False)
        
        # --- MODIFIED: 简化推理，只收集概率 ---
        # 移除了 combo_preds = []
        combo_probs = [] 
        
        with torch.no_grad():
            for lip_batch, lm_batch in combo_loader:
                lip_features_cat, _ = lip_enc(lip_batch)
                pose_features_lstm, _ = pose_enc(lm_batch)
                fused = torch.cat([lip_features_cat, pose_features_lstm], dim=1)
                logits = det_head(fused)
                
                # 只计算和保存概率 (用于 AUC/AP 和 最终决策)
                probs = F.softmax(logits, dim=1)[:, 1] 
                
                # --- MODIFIED: 移除不必要的硬预测 ---
                # 移除了 _, predicted = torch.max(logits, 1)
                # 移除了 combo_preds.extend(predicted.cpu().numpy())
                
                combo_probs.extend(probs.cpu().numpy())

        # --- MODIFIED: 统一的视频级聚合策略 ---
        # 基于 "Combo 平均概率" 进行所有决策
        
        if not combo_probs:
            # 这是一个边缘情况，以防万一
            video_score = 0.0
            video_prediction = 0
        else:
            # 1. 计算视频的统一分数 (用于 AUC/AP)
            video_score = np.max(combo_probs)   ####             mean  或者max
            
            
            # 2. 基于统一分数和阈值进行预测 (用于 F1/ACC/FPR/FNR)
            video_prediction = 1 if video_score >= args.threshold else 0

        # --- MODIFIED: 移除旧的、不一致的投票逻辑 ---
        # 移除了 num_fake_combos = np.sum(combo_preds)
        # 移除了 fake_ratio = num_fake_combos / len(combo_preds)
        # 移除了旧的 video_prediction = 1 if fake_ratio >= args.threshold else 0
        
        video_gts.append(true_label)
        video_preds.append(video_prediction)
        video_scores.append(video_score)

    # --- 缓存保存逻辑 (无变化) ---
    if not cache_exists and feature_extractor and args.save_cache:
        print(f"\n正在将新提取的特征保存到缓存文件: '{args.cache_file}' ...")
        cache_dir = os.path.dirname(args.cache_file)
        if not os.path.exists(cache_dir): os.makedirs(cache_dir)
        np.save(args.cache_file, cached_data)
        print("缓存保存成功。")

    print("\n" + "="*30 + "\n     Video-Level Performance Results\n" + "="*30)
    if not video_gts:
        print("未能处理任何视频，无法计算指标。")
        return
        
    # +++ 诊断 4 (无变化) +++
    final_label_counts = Counter(video_gts)
    print(f"诊断信息: 共有 {len(video_gts)} 个视频被成功处理并用于最终评估。")
    print(f"最终标签分布: {final_label_counts}")

    # === 指标计算 (无变化, 因为输入 video_preds 和 video_scores 已经统一) ===
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
        print("\n警告: 最终评估数据中只存在一种标签，无法计算AUC和AP分数。")

    tn, fp, fn, tp = confusion_matrix(video_gts, video_preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    print(f"\nTotal Videos Processed: {len(video_gts)}")
    print(f"Threshold used: {args.threshold:.4f}") # +++ 新增: 打印使用的阈值 +++
    print(f"Accuracy:  {acc:.4f}, AUC:       {auc:.4f}, AP:        {ap:.4f}")
    print(f"F1-Score:  {f1:.4f}, Precision: {precision:.4f}, Recall:    {recall:.4f}")
    print(f"FPR:       {fpr:.4f}, FNR:       {fnr:.4f}")
    print("="*30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproducible inference with on-the-fly feature extraction.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--checkpoint', type=str, default='Ours/TalkHead_checkpoints/best_model.pth', help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--test_videos_json', type=str, default='talkheadbench/test_videos.json', help='Path to the test set JSON file')
    parser.add_argument('--preprocessed_dir', type=str, default='talkheadbench', help='Path to the root of preprocessed data')
    parser.add_argument('--cache_file', type=str, default='talkheadbench/test_cache.npy', help='Path to read/write the .npy cache file')
    parser.add_argument('--dlib_predictor', type=str, default='shape_predictor_68_face_landmarks.dat', help='Path to dlib`s facial landmark predictor')
    
    # --- MODIFIED: 移除不必要的参数 ---
    parser.add_argument('--num_combos_per_video', type=int, default=30, help='Number of combos to sample from each video')
    
    parser.add_argument('--threshold', type=float, default=0.5, help='Voting threshold for video-level prediction')
    parser.add_argument('--save_cache', action='store_true', help='If specified, save the extracted features to cache_file if it does not exist.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for model inference')
    parser.add_argument('--lstm_hidden', type=int, default=256, help='Hidden dim for PoseEncoder LSTM (must match training)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference ("cuda" or "cpu")')
    args = parser.parse_args()
    inference(args)