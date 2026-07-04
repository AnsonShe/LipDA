# 从精简后的 json 索引中提取lip 裁剪和landmark ，第四步

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import dlib  # 用于人脸检测和关键点
import argparse
import librosa


# ======================
# 数据集类
# ======================
class ComboDataset(Dataset):
    def __init__(self, index_json, combo_len=5, transform=None):
        """
        index_json: 组合索引文件路径 (如 train_combo_index.json)
        combo_len: 组合帧长度 (默认5)
        transform: 图像变换
        """
        with open(index_json, 'r') as f:
            self.combos = json.load(f)
        self.combo_len = combo_len
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # 初始化人脸检测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # 缓存预处理结果
        self.cache_dir = os.path.join(os.path.dirname(index_json), "precomputed")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, 
                                     f"{os.path.basename(index_json).replace('.json', '_cache.npy')}")
        
        # 尝试加载缓存
        if os.path.exists(self.cache_file):
            print(f"Loading precomputed data from {self.cache_file}")
            self.cached_data = np.load(self.cache_file, allow_pickle=True).item()
        else:
            self.cached_data = {'lip_images': {}, 'landmarks': {}}
            self.precompute_all()
    
    def precompute_all(self):
        print("Precomputing lip crops and landmarks...")
        for idx in tqdm(range(len(self.combos))):
            item = self.combos[idx]
            combo_path = item['combo_path']
            
            if combo_path in self.cached_data['lip_images']:
                continue
                
            frame_files = sorted(os.listdir(combo_path))
            lip_images = []
            landmarks = []
            
            for frame_file in frame_files:
                frame_path = os.path.join(combo_path, frame_file)
                img = cv2.imread(frame_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 人脸检测
                dets = self.detector(img, 1)
                if len(dets) == 0:
                    # 尝试放大图像进行检测
                    scaled_img = cv2.resize(img, (0,0), fx=1.5, fy=1.5)
                    dets = self.detector(scaled_img, 1)
                    if len(dets) == 0:
                        # 使用整个图像的下方45%作为后备方案
                        h, w = img.shape[:2]
                        y_start = int(h * 0.55)  # 从55%高度开始
                        lip_img = img[y_start:h, 0:w]
                        if lip_img.size == 0:
                            lip_img = np.zeros((96, 96, 3), dtype=np.uint8)
                        else:
                            lip_img = cv2.resize(lip_img, (96, 96))
                        lms = np.zeros((68, 2))
                    else:
                        d = dets[0]
                        # 缩放回原始尺寸
                        d = dlib.rectangle(
                            int(d.left()/1.5), int(d.top()/1.5),
                            int(d.right()/1.5), int(d.bottom()/1.5)
                        )
                        shape = self.predictor(img, d)
                        lms = np.array([[p.x, p.y] for p in shape.parts()])
                        # 获取面部区域并计算下方45%
                        face_height = d.bottom() - d.top()
                        crop_height = int(face_height * 0.45)
                        y1 = max(0, d.bottom() - crop_height)
                        y2 = d.bottom()
                        x_center = (d.left() + d.right()) // 2
                        crop_width = int(face_height * 0.45 * 1.5)  # 宽高比1.5:1
                        x1 = max(0, x_center - crop_width//2)
                        x2 = min(img.shape[1], x_center + crop_width//2)
                        lip_img = img[y1:y2, x1:x2]
                        if lip_img.size == 0:
                            lip_img = np.zeros((96, 96, 3), dtype=np.uint8)
                        else:
                            lip_img = cv2.resize(lip_img, (96, 96))
                else:
                    d = dets[0]
                    shape = self.predictor(img, d)
                    lms = np.array([[p.x, p.y] for p in shape.parts()])
                    # 获取面部区域并计算下方45%
                    face_height = d.bottom() - d.top()
                    crop_height = int(face_height * 0.45)
                    y1 = max(0, d.bottom() - crop_height)
                    y2 = d.bottom()
                    x_center = (d.left() + d.right()) // 2
                    crop_width = int(face_height * 0.45 * 1.5)  # 宽高比1.5:1
                    x1 = max(0, x_center - crop_width//2)
                    x2 = min(img.shape[1], x_center + crop_width//2)
                    lip_img = img[y1:y2, x1:x2]
                    if lip_img.size == 0:
                        lip_img = np.zeros((96, 96, 3), dtype=np.uint8)
                    else:
                        lip_img = cv2.resize(lip_img, (96, 96))
                
                lip_images.append(lip_img)
                landmarks.append(lms)
            
            self.cached_data['lip_images'][combo_path] = lip_images
            self.cached_data['landmarks'][combo_path] = np.array(landmarks)
        
        np.save(self.cache_file, self.cached_data)
    
    def __len__(self):               
        return len(self.combos)
    
    def __getitem__(self, idx):
        item = self.combos[idx]
        combo_path = item['combo_path']
        label = item['label']
        
        # 从缓存获取数据
        lip_images = self.cached_data['lip_images'][combo_path]
        landmarks = self.cached_data['landmarks'][combo_path]
        
        # 应用变换
        transformed_imgs = []
        for img in lip_images:
            if self.transform:
                img = self.transform(img)
            transformed_imgs.append(img)
        
        # 组合成张量 [T, C, H, W]
        combo_imgs = torch.stack(transformed_imgs)
        
        # 转换为张量 [T, 68, 2]
        landmarks = torch.tensor(landmarks, dtype=torch.float32)
        
        return {
            'combo_imgs': combo_imgs,  # [5, C, H, W]
            'landmarks': landmarks,     # [5, 68, 2]
            'label': label
        }
   
   

     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lip-forensics Stage-1 Training')
    parser.add_argument('--train_index', type=str, default='Ours/reduced_data/train_combo_index_reduced.json',
                        help='Path to train combo index JSON')
    parser.add_argument('--val_index', type=str, default='Ours/reduced_data/val_combo_index_reduced.json',
                        help='Path to validation combo index JSON')
    
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    args = parser.parse_args()
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据集和数据加载器
    train_dataset = ComboDataset(args.train_index)
    val_dataset = ComboDataset(args.val_index)
    
    train_loader = DataLoader(train_dataset,  batch_size=args.batch_size,  shuffle=True, num_workers=4, pin_memory=True )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=2,pin_memory=True)