import os
import glob
import pickle
import numpy as np
import cv2
import librosa
import mediapipe as mp
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import subprocess
import tempfile

def extract_audio_from_video(video_path):
        """
        专门从视频文件中提取音频
        """
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            temp_audio_path = tmp_audio.name
        
        try:
            # 使用ffmpeg提取音频
            cmd = [
                'ffmpeg', '-i', video_path, 
                '-vn',              # 禁用视频
                '-acodec', 'pcm_s16le',  # PCM编码
                '-ar', '22050',     # 采样率
                '-ac', '1',         # 单声道
                '-y',               # 覆盖输出文件
                temp_audio_path
            ]
            
            # 运行ffmpeg命令
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=30  # 30秒超时
            )
            
            if result.returncode != 0:
                print(f"FFmpeg错误: {result.stderr}")
                # 返回空音频
                return np.zeros(22050 * 5), 22050  # 5秒静音
            
            # 使用librosa读取提取的音频
            audio, sr = librosa.load(temp_audio_path, sr=None)
            return audio, sr
            
        except subprocess.TimeoutExpired:
            print(f"音频提取超时: {video_path}")
            return np.zeros(22050 * 5), 22050
        except Exception as e:
            print(f"音频提取失败 {video_path}: {e}")
            return np.zeros(22050 * 5), 22050
        finally:
            # 清理临时文件
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

class LipSyncDataset(Dataset):
    def __init__(self, transformer_dir, gan_dir, diffusion_dir, vae_dir, cnn_dir, max_frames=75, preprocess=True, cache_dir="./cache"):
        """
        初始化LipSync数据集
        
        参数:
            transformer_dir: Transformer生成视频的目录
            gan_dir: GAN生成视频的目录
            diffusion_dir: Diffusion生成视频的目录
            vae_dir: VAE生成视频的目录
            cnn_dir: CNN生成视频的目录
            max_frames: 最大帧数(统一长度)
            preprocess: 是否进行预处理
            cache_dir: 缓存目录
        """
        self.max_frames = max_frames
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 收集文件路径和标签
        self.file_paths = []
        self.labels = []
        
        # Transformer视频
        transformer_files = glob.glob(os.path.join(transformer_dir, "*.mp4"))
        self.file_paths.extend(transformer_files)
        self.labels.extend([0] * len(transformer_files))
        
        # GAN视频
        gan_files = glob.glob(os.path.join(gan_dir, "*.mp4"))
        self.file_paths.extend(gan_files)
        self.labels.extend([1] * len(gan_files))
        
        # Diffusion视频
        diffusion_files = glob.glob(os.path.join(diffusion_dir, "*.mp4"))
        self.file_paths.extend(diffusion_files)
        self.labels.extend([2] * len(diffusion_files))
        
        # VAE视频 (新增)
        vae_files = glob.glob(os.path.join(vae_dir, "*.mp4"))
        self.file_paths.extend(vae_files)
        self.labels.extend([3] * len(vae_files))

        # CNN视频 (新增)
        cnn_files = glob.glob(os.path.join(cnn_dir, "*.mp4"))
        self.file_paths.extend(cnn_files)
        self.labels.extend([4] * len(cnn_files))
            
        # 预处理所有视频(如果启用)
        if preprocess:
            self.preprocess_all()
            
        # 加载预处理后的特征
        self.load_preprocessed_features()
    
    def preprocess_all(self):
        """预处理所有视频并保存特征"""
        print("开始预处理所有视频...")
        for i, video_path in enumerate(self.file_paths):
            print(f"处理视频 {i+1}/{len(self.file_paths)}: {video_path}")
            
            # 生成缓存文件名
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            cache_file = os.path.join(self.cache_dir, f"{base_name}.pkl")
            
            # 如果已经处理过，跳过
            if os.path.exists(cache_file):
                continue
                
            # 提取特征
            features = self.extract_features(video_path)
            
            # 保存到缓存
            with open(cache_file, 'wb') as f:
                pickle.dump(features, f)
    




    def extract_features(self, video_path):
        """
        从单个视频提取特征
        
        返回:
            features: 包含landmarks, lip_roi, mfcc特征的字典
        """
        # 初始化MediaPipe面部网格
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5)
        
        # 读取视频
        cap = cv2.VideoCapture(video_path)
        frames = []
        landmarks_list = []
        
        # 使用专门的音频提取函数
        audio, sr = extract_audio_from_video(video_path)
        
        # 提取MFCC特征
        if len(audio) > 1000:  # 确保有足够的音频数据
            try:
                mfcc_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=512)
                delta_mfcc = librosa.feature.delta(mfcc_features)
                delta2_mfcc = librosa.feature.delta(mfcc_features, order=2)
                mfcc = np.vstack([mfcc_features, delta_mfcc, delta2_mfcc])
            except Exception as e:
                print(f"MFCC提取失败 {video_path}: {e}")
                mfcc = np.zeros((39, 100))
        else:
            print(f"音频数据不足 {video_path}, 长度: {len(audio)}")
            mfcc = np.zeros((39, 100))
        
        # 计算MFCC与视频帧的对应关系
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25  # 默认值
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # 如果无法获取帧数，手动计算
            total_frames = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                total_frames += 1
            cap = cv2.VideoCapture(video_path)  # 重新打开视频
        
        mfcc_per_frame = mfcc.shape[1] / max(total_frames, 1)
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 转换为RGB (MediaPipe需要RGB格式)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            # 面部landmark检测
            results = face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
                landmarks_list.append(landmarks)
            else:
                # 如果没有检测到面部，使用前一帧的landmarks或零填充
                if landmarks_list:
                    landmarks_list.append(landmarks_list[-1])
                else:
                    landmarks_list.append(np.zeros((478, 3)))
            
            frame_count += 1
            if frame_count >= self.max_frames:
                break
                
        cap.release()
        
        # 检查是否成功读取到帧
        if len(frames) == 0:
            print(f"警告: 无法从视频读取帧 {video_path}")
            # 创建空的特征
            landmarks_array = np.zeros((self.max_frames, 478, 3))
            lip_roi_array = np.zeros((self.max_frames, 96, 96, 3))
            mfcc_array = np.zeros((self.max_frames, 39))
            return {
                'landmarks': landmarks_array,
                'lip_roi': lip_roi_array,
                'mfcc': mfcc_array
            }
        
        # 对齐帧数和MFCC特征
        aligned_mfcc = []
        for i in range(min(self.max_frames, len(frames))):
            mfcc_idx = min(int(i * mfcc_per_frame), mfcc.shape[1] - 1)
            aligned_mfcc.append(mfcc[:, mfcc_idx])
        
        # 裁剪唇部ROI
        lip_roi_list = self.extract_lip_roi(frames, landmarks_list)
        
        # 转换为numpy数组并统一长度
        landmarks_array = self.pad_or_truncate(np.array(landmarks_list), self.max_frames)
        lip_roi_array = self.pad_or_truncate(np.array(lip_roi_list), self.max_frames)
        mfcc_array = self.pad_or_truncate(np.array(aligned_mfcc), self.max_frames)
        
        return {
            'landmarks': landmarks_array,
            'lip_roi': lip_roi_array,
            'mfcc': mfcc_array
        }
    
    def extract_lip_roi(self, frames, landmarks_list):
        """提取唇部ROI区域"""
        # 唇部landmark索引 (MediaPipe定义)
        lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 320, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        
        lip_roi_list = []
        for i, (frame, landmarks) in enumerate(zip(frames, landmarks_list)):
            # 获取唇部landmarks
            lip_landmarks = landmarks[lip_indices]
            
            # 计算唇部边界框
            x_min, y_min = np.min(lip_landmarks[:, :2], axis=0)
            x_max, y_max = np.max(lip_landmarks[:, :2], axis=0)
            
            # 扩展边界框
            expand_factor = 0.2
            width = x_max - x_min
            height = y_max - y_min
            x_min = max(0, x_min - width * expand_factor)
            x_max = min(1, x_max + width * expand_factor)
            y_min = max(0, y_min - height * expand_factor)
            y_max = min(1, y_max + height * expand_factor)
            
            # 转换为像素坐标
            h, w = frame.shape[:2]
            x_min_px, x_max_px = int(x_min * w), int(x_max * w)
            y_min_px, y_max_px = int(y_min * h), int(y_max * h)
            
            # 裁剪ROI区域
            lip_roi = frame[y_min_px:y_max_px, x_min_px:x_max_px]
            
            # 调整大小为统一尺寸
            if lip_roi.size > 0:
                lip_roi = cv2.resize(lip_roi, (96, 96))
            else:
                lip_roi = np.zeros((96, 96, 3), dtype=np.uint8)
                
            lip_roi_list.append(lip_roi)
            
        return lip_roi_list
    
    def pad_or_truncate(self, array, target_length):
        """填充或截断数组到目标长度"""
        if len(array) > target_length:
            return array[:target_length]
        elif len(array) < target_length:
            pad_shape = [(0, target_length - len(array))] + [(0, 0)] * (array.ndim - 1)
            return np.pad(array, pad_shape, mode='constant')
        return array
    
    def load_preprocessed_features(self):
        """加载所有预处理后的特征"""
        self.features = []
        self.valid_indices = []
        
        for i, video_path in enumerate(self.file_paths):
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            cache_file = os.path.join(self.cache_dir, f"{base_name}.pkl")
            
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    self.features.append(pickle.load(f))
                self.valid_indices.append(i)
        
        # 更新文件路径和标签为有效索引
        self.file_paths = [self.file_paths[i] for i in self.valid_indices]
        self.labels = [self.labels[i] for i in self.valid_indices]
        
        print(f"成功加载 {len(self.features)} 个视频的特征")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        # 转换为PyTorch张量
        landmarks = torch.FloatTensor(feature['landmarks'])
        lip_roi = torch.FloatTensor(feature['lip_roi'].transpose(0, 3, 1, 2)) / 255.0  # 转换为CxHxW并归一化
        mfcc = torch.FloatTensor(feature['mfcc'])
        
        return {
            'landmarks': landmarks,
            'lip_roi': lip_roi,
            'mfcc': mfcc,
            'label': torch.LongTensor([label])
        }

# 创建数据集
def create_datasets(transformer_dir, gan_dir, diffusion_dir, vae_dir, cnn_dir, test_size=0.2, val_size=0.1, batch_size=8):
    """创建训练、验证和测试数据集"""
    full_dataset = LipSyncDataset(transformer_dir, gan_dir, diffusion_dir, vae_dir, cnn_dir)
    
    # 划分训练集和测试集
    train_idx, test_idx = train_test_split(
        range(len(full_dataset)), 
        test_size=test_size, 
        stratify=full_dataset.labels,
        random_state=42
    )
    
    # 从训练集中划分验证集
    train_idx, val_idx = train_test_split(
        train_idx, 
        test_size=val_size/(1-test_size), 
        stratify=[full_dataset.labels[i] for i in train_idx],
        random_state=42
    )
    
    # 创建子数据集
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader