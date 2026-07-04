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
import torch.optim as optim
import torch.nn.functional as F
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
    
from models import LipEncoder, PoseEncoder, Projector, DetectionHead
from extract_feature import ComboDataset

# ======================
# 损失函数
# ======================
# class FeatureAlignmentLoss(nn.Module):
#     """特征对齐损失函数，根据样本类型调整lip和pose特征距离"""
#     def __init__(self, margin=1.0, epsilon=0.5):
#         super().__init__()
#         self.margin = margin  # 用于fake样本的margin
#         self.epsilon = nn.Parameter(torch.tensor(epsilon))  # 可学习的阈值
    
#     def forward(self, lip_feat, pose_feat, labels):
#         """
#         lip_feat: lip特征 [B, D]
#         pose_feat: pose特征 [B, D]
#         labels: 真实/伪造标签 [B], 0=real, 1=fake
#         """
#         # 计算lip和pose特征间的距离
#         distances = torch.norm(lip_feat - pose_feat, p=2, dim=1)  # [B]
        
#         # real样本: 最小化距离
#         real_loss = torch.mean(distances[labels == 0] ** 2)
        
#         # fake样本: 最大化距离，至少margin
#         fake_loss = torch.mean(torch.clamp(self.margin - distances[labels == 1], min=0) ** 2)
        
#         # 总损失
#         total_loss = real_loss + fake_loss
        
#         # 返回总损失和当前epsilon值
#         return total_loss, self.epsilon.item()
class FeatureAlignmentLoss(nn.Module):
    # === MODIFIED: Epsilon现在是可学习的margin ===
    def __init__(self, initial_epsilon=1.0):
        super().__init__()
        # epsilon被定义为可学习参数，作为伪造样本的距离边界
        self.epsilon = nn.Parameter(torch.tensor(initial_epsilon))

    def forward(self, lip_feat, pose_feat, labels):
        distances = torch.norm(lip_feat - pose_feat, p=2, dim=1)
        
        # real样本: 最小化距离
        real_loss = (distances[labels == 0] ** 2).mean() if (labels == 0).sum() > 0 else 0.0
        
        # fake样本: 距离应大于可学习的epsilon
        # 使用 clamp(epsilon - distance, min=0) 作为损失
        fake_loss = (torch.clamp(self.epsilon - distances[labels == 1], min=0) ** 2).mean() if (labels == 1).sum() > 0 else 0.0
        
        total_loss = real_loss + fake_loss
        
        # 只返回损失，epsilon的值在主循环中记录
        return total_loss



# ======================
# 训练函数
# ======================
# def train_epoch(model_components, dataloader, optimizer, device, epoch, lambda_align=1.0, lambda_det=1.0):
#     """
#     训练一个epoch
#     """
#     lip_enc, pose_enc, proj_lip, proj_pose, det_head = model_components
#     lip_enc.train(); pose_enc.train(); proj_lip.train(); proj_pose.train(); det_head.train()
    
#     align_loss_fn = FeatureAlignmentLoss(margin=1.0)
#     det_loss_fn = nn.CrossEntropyLoss()
    
#     total_loss = 0.0
#     total_align_loss = 0.0
#     total_det_loss = 0.0
#     correct = 0
#     total = 0
#     epsilon_values = []
    
#     for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
#         # 准备数据
#         imgs = batch['combo_imgs'].to(device)   # [B, 5, C, H, W]
#         lms = batch['landmarks'].to(device)     # [B, 5, 68, 2]
#         labels = batch['label'].to(device)       # [B]
        
#         # 前向传播
#         lip_features, _ = lip_enc(imgs)         # [B, 5*128]
#         pose_features, _ = pose_enc(lms)        # [B, 5*128]
        
#         # 投影到超球面
#         lip_proj = proj_lip(lip_features)       # [B, 128]
#         pose_proj = proj_pose(pose_features)    # [B, 128]
        
#         # 计算特征对齐损失
#         align_loss, epsilon = align_loss_fn(lip_proj, pose_proj, labels)
#         epsilon_values.append(epsilon)
        
#         # 融合特征用于检测
#         fused = torch.cat([lip_features, pose_features], dim=1)  # [B, 10*128]
#         logits = det_head(fused)                # [B, 2]
        
#         # 计算检测损失
#         det_loss = det_loss_fn(logits, labels)
        
#         # 总损失
#         loss = lambda_det * det_loss + lambda_align * align_loss
        
#         # 反向传播
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         # 统计信息
#         total_loss += loss.item()
#         total_align_loss += align_loss.item()
#         total_det_loss += det_loss.item()
        
#         # 计算准确率
#         _, predicted = torch.max(logits, 1)
#         correct += (predicted == labels).sum().item()
#         total += labels.size(0)
    
#     # 计算指标
#     avg_loss = total_loss / len(dataloader)
#     avg_align_loss = total_align_loss / len(dataloader)
#     avg_det_loss = total_det_loss / len(dataloader)
#     avg_epsilon = np.mean(epsilon_values)
#     accuracy = correct / total
    
#     print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Align: {avg_align_loss:.4f} | "
#           f"Det: {avg_det_loss:.4f} | Acc: {accuracy:.4f} | Epsilon: {avg_epsilon:.4f}")
    
#     return avg_loss, accuracy, avg_epsilon
def train_epoch(model_components, dataloader, optimizer, device, epoch, align_loss_fn, lambda_align=1.0, lambda_det=1.0):
    lip_enc, pose_enc, proj_lip, proj_pose, det_head = model_components
    lip_enc.train(); pose_enc.train(); proj_lip.train(); proj_pose.train(); det_head.train()
    
    det_loss_fn = nn.CrossEntropyLoss()
    
    total_loss, total_align_loss, total_det_loss = 0.0, 0.0, 0.0
    correct, total = 0, 0
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch} [Train]"):
        imgs = batch['combo_imgs'].to(device)
        lms = batch['landmarks'].to(device)
        labels = batch['label'].to(device)
        
        # === MODIFIED: PoseEncoder现在返回不同的特征维度 ===
        lip_features_cat, _ = lip_enc(imgs)
        pose_features_lstm, _ = pose_enc(lms)
        
        lip_proj = proj_lip(lip_features_cat)
        # 注意：现在需要为LSTM的输出创建一个新的Projector
        # 或者为了简单起见，我们继续使用拼接的特征进行对齐，这里我们假设为前者
        pose_proj = proj_pose(pose_features_lstm)
        
        align_loss = align_loss_fn(lip_proj, pose_proj, labels)
        
        # 融合特征用于检测
        fused = torch.cat([lip_features_cat, pose_features_lstm], dim=1)
        logits = det_head(fused)
        
        det_loss = det_loss_fn(logits, labels)
        
        loss = lambda_det * det_loss + lambda_align * align_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_align_loss += align_loss.item()
        total_det_loss += det_loss.item()
        
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return {
        'loss': total_loss / len(dataloader),
        'align_loss': total_align_loss / len(dataloader),
        'det_loss': total_det_loss / len(dataloader),
        'accuracy': correct / total
    }

def evaluate_epoch(model_components, dataloader, device, align_loss_fn):
    lip_enc, pose_enc, proj_lip, proj_pose, det_head = model_components
    lip_enc.eval(); pose_enc.eval(); proj_lip.eval(); proj_pose.eval(); det_head.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            imgs = batch['combo_imgs'].to(device)
            lms = batch['landmarks'].to(device)
            labels = batch['label'].to(device)

            lip_features_cat, _ = lip_enc(imgs)
            pose_features_lstm, _ = pose_enc(lms)
            
            fused = torch.cat([lip_features_cat, pose_features_lstm], dim=1)
            logits = det_head(fused)
            
            probs = F.softmax(logits, dim=1)[:, 1] # 获取标签为1的概率
            _, predicted = torch.max(logits, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算各项指标
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    accuracy = (all_labels == all_preds).mean()
    auc = roc_auc_score(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)

    # 计算混淆矩阵获取FPR, FNR
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'auc': auc,
        'ap': ap,
        'fpr': fpr,
        'fnr': fnr
    }





# ======================
# 主函数
# ======================
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Lip-forensics Stage-1 Training')
    
    parser.add_argument('--train_index', type=str, default='avlip/reduced_data/train_combo_index_reduced.json',
                        help='Path to train combo index JSON')
    parser.add_argument('--val_index', type=str, default='avlip/reduced_data/val_combo_index_reduced.json',
                        help='Path to validation combo index JSON')

    parser.add_argument('--save_dir', type=str, default='AVLip_checkpoints',
                        help='Directory to save checkpoints')
    
    
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lambda_align', type=float, default=1.0,
                        help='Weight for alignment loss')
    parser.add_argument('--lambda_det', type=float, default=1.0,
                        help='Weight for detection loss')
    
    parser.add_argument('--lstm_hidden', type=int, default=256, help='Hidden dim for PoseEncoder LSTM')
    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # TensorBoard日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=f'runs/exp_{timestamp}')
    
    # 数据集和数据加载器
    train_dataset = ComboDataset(args.train_index)
    val_dataset = ComboDataset(args.val_index)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # 模型初始化
    lip_enc = LipEncoder().to(device)
    # pose_enc = PoseEncoder().to(device)
    pose_enc = PoseEncoder(lstm_hidden_dim=args.lstm_hidden).to(device)
    proj_lip = Projector(5*128).to(device)
    # proj_pose = Projector(5*128).to(device)
    # Pose Projector 输入是 LSTM 的输出维度
    proj_pose = Projector(pose_enc.output_dim).to(device) 
    # det_head = DetectionHead(10*128).to(device)
    det_head_in_dim = (5 * 128) + pose_enc.output_dim
    det_head = DetectionHead(det_head_in_dim).to(device)
    
    # 优化器
    # model_params = list(lip_enc.parameters()) + list(pose_enc.parameters()) + \
    #               list(proj_lip.parameters()) + list(proj_pose.parameters()) + \
    #               list(det_head.parameters())
    model_components = (lip_enc, pose_enc, proj_lip, proj_pose, det_head)
    
    align_loss_fn = FeatureAlignmentLoss(initial_epsilon=1.0).to(device)
    model_params = list(lip_enc.parameters()) + list(pose_enc.parameters()) + \
                   list(proj_lip.parameters()) + list(proj_pose.parameters()) + \
                   list(det_head.parameters()) + list(align_loss_fn.parameters()) # 加入align_loss的参数
    
    
    optimizer = optim.Adam(model_params, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # # 训练循环
    # best_val_acc = 0.0
    # best_epsilon = 0.5
    # for epoch in range(args.epochs):
    #     # 训练
    #     train_loss, train_acc, train_epsilon = train_epoch(
    #         (lip_enc, pose_enc, proj_lip, proj_pose, det_head),
    #         train_loader,
    #         optimizer,
    #         device,
    #         epoch,
    #         lambda_align=args.lambda_align,
    #         lambda_det=args.lambda_det
    #     )
        
    #     # 验证
    #     lip_enc.eval(); pose_enc.eval(); proj_lip.eval(); proj_pose.eval(); det_head.eval()
    #     val_correct = 0
    #     val_total = 0
    #     val_distances = []
        
    #     with torch.no_grad():
    #         for batch in val_loader:
    #             imgs = batch['combo_imgs'].to(device)
    #             lms = batch['landmarks'].to(device)
    #             labels = batch['label'].to(device)
                
    #             lip_features, _ = lip_enc(imgs)
    #             pose_features, _ = pose_enc(lms)
    #             lip_proj = proj_lip(lip_features)
    #             pose_proj = proj_pose(pose_features)
                
    #             # 计算特征距离
    #             distances = torch.norm(lip_proj - pose_proj, p=2, dim=1)
    #             val_distances.extend(distances.cpu().numpy())
                
    #             # 检测任务
    #             fused = torch.cat([lip_features, pose_features], dim=1)
    #             logits = det_head(fused)
                
    #             _, predicted = torch.max(logits, 1)
    #             val_correct += (predicted == labels).sum().item()
    #             val_total += labels.size(0)
        
    #     val_acc = val_correct / val_total
    #     avg_val_distance = np.mean(val_distances)
    #     print(f"Validation | Acc: {val_acc:.4f} | Avg Distance: {avg_val_distance:.4f} | "
    #           f"Train Epsilon: {train_epsilon:.4f}")
        
    #     # 保存最佳模型
    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         best_epsilon = train_epsilon
    #         torch.save({
    #             'lip_enc': lip_enc.state_dict(),
    #             'pose_enc': pose_enc.state_dict(),
    #             'proj_lip': proj_lip.state_dict(),
    #             'proj_pose': proj_pose.state_dict(),
    #             'det_head': det_head.state_dict(),
    #             'epoch': epoch,
    #             'val_acc': val_acc,
    #             'epsilon': train_epsilon
    #         }, os.path.join(args.save_dir, 'best_model.pth'))
    #         print(f"Saved best model with val acc {val_acc:.4f} and epsilon {train_epsilon:.4f}")
        
    #     # 定期保存
    #     if epoch % 5 == 0:
    #         torch.save({
    #             'lip_enc': lip_enc.state_dict(),
    #             'pose_enc': pose_enc.state_dict(),
    #             'proj_lip': proj_lip.state_dict(),
    #             'proj_pose': proj_pose.state_dict(),
    #             'det_head': det_head.state_dict(),
    #             'epoch': epoch,
    #             'val_acc': val_acc,
    #             'epsilon': train_epsilon
    #         }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    # print(f"Training complete. Best validation accuracy: {best_val_acc:.4f} with epsilon: {best_epsilon:.4f}")
    
    
    best_val_auc = 0.0
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model_components, train_loader, optimizer, device, epoch, align_loss_fn,
                                    lambda_align=args.lambda_align, lambda_det=args.lambda_det)
        
        val_metrics = evaluate_epoch(model_components, val_loader, device, align_loss_fn)
        
        # 更新学习率
        scheduler.step()

        # --- 日志打印 ---
        print(f"Epoch {epoch} | Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Validation | Acc: {val_metrics['accuracy']:.4f} | AUC: {val_metrics['auc']:.4f} | AP: {val_metrics['ap']:.4f}")
        print(f"             FPR: {val_metrics['fpr']:.4f} | FNR: {val_metrics['fnr']:.4f} | Epsilon: {align_loss_fn.epsilon.item():.4f}")
        
        # --- TensorBoard 记录 ---
        writer.add_scalar('Loss/train_total', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/train_align', train_metrics['align_loss'], epoch)
        writer.add_scalar('Loss/train_det', train_metrics['det_loss'], epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        
        writer.add_scalar('Metrics/Val_Accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('Metrics/Val_AUC', val_metrics['auc'], epoch)
        writer.add_scalar('Metrics/Val_AP', val_metrics['ap'], epoch)
        writer.add_scalar('Metrics/Val_FPR', val_metrics['fpr'], epoch)
        writer.add_scalar('Metrics/Val_FNR', val_metrics['fnr'], epoch)
        
        writer.add_scalar('Params/Epsilon', align_loss_fn.epsilon.item(), epoch)
        writer.add_scalar('Params/LearningRate', optimizer.param_groups[0]['lr'], epoch)

        # === MODIFIED: 使用AUC作为保存最佳模型的标准 ===
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save({
                'lip_enc': lip_enc.state_dict(), 'pose_enc': pose_enc.state_dict(),
                'proj_lip': proj_lip.state_dict(), 'proj_pose': proj_pose.state_dict(),
                'det_head': det_head.state_dict(), 'align_loss_fn': align_loss_fn.state_dict(),
                'epoch': epoch, 'val_auc': best_val_auc
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Saved best model with val AUC {best_val_auc:.4f}")
    
    # +++ NEW: 训练结束后记录超参数和最终结果 +++
    hparams = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float))}
    final_metrics = {'hparam/best_val_auc': best_val_auc}
    writer.add_hparams(hparams, final_metrics)
    
    writer.close()
    print(f"Training complete. Best validation AUC: {best_val_auc:.4f}")