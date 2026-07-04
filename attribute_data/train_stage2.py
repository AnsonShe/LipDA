import time
import pickle
import os
import argparse
import torch
import torch.nn as nn
from model_attribution import AVTSTAN
from attri_dataloader import LipSyncDataset, create_datasets
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']   # 思源黑体
plt.rcParams['axes.unicode_minus'] = False               # 解决负号显示问题


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, device='cuda'):
    """训练模型"""
    # 创建TensorBoard写入器
    writer = SummaryWriter()
    
    # 将模型移动到设备
    model = model.to(device)
    
    # 训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = 'best_model.pth'
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        start_time = time.time()
        
        for i, batch in enumerate(train_loader):
            # 获取输入和标签
            landmarks = batch['landmarks'].to(device)
            lip_roi = batch['lip_roi'].to(device)
            mfcc = batch['mfcc'].to(device)
            # labels = batch['label'].squeeze().to(device)
            labels = batch['label'].view(-1).to(device)
            
            # print(f"Batch shapes:")
            # print(f"  landmarks: {landmarks.shape}")  # 应该是 (batch, seq_len, landmarks, 3)
            # print(f"  lip_roi: {lip_roi.shape}")      # 应该是 (batch, seq_len, C, H, W)
            # print(f"  mfcc: {mfcc.shape}")            # 应该是 (batch, seq_len, mfcc_dim)
            # print(f"  labels: {labels.shape}")
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs, attn_weights = model(landmarks, lip_roi, mfcc)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item() * landmarks.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            
            # 每10个batch打印一次
            if i % 10 == 0:
                print(f'Batch {i}, Loss: {loss.item():.4f}')
        
        # 计算epoch损失和准确率
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.cpu().numpy())
        
        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for batch in val_loader:
                landmarks = batch['landmarks'].to(device)
                lip_roi = batch['lip_roi'].to(device)
                mfcc = batch['mfcc'].to(device)
                # labels = batch['label'].squeeze().to(device)
                labels = batch['label'].view(-1).to(device)
                
                outputs, _ = model(landmarks, lip_roi, mfcc)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * landmarks.size(0)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/val', val_epoch_loss, epoch)
        writer.add_scalar('Accuracy/val', val_epoch_acc, epoch)
        
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.cpu().numpy())
        
        # 更新学习率
        scheduler.step(val_epoch_loss)
        
        # 打印统计信息
        epoch_time = time.time() - start_time
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        print(f'Time: {epoch_time:.0f}s')
        
        # 保存最佳模型
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'保存最佳模型，准确率: {best_val_acc:.4f}')
    
    # 关闭TensorBoard写入器
    writer.close()
    
    print(f'最佳验证准确率: {best_val_acc:.4f}')
    
    # 保存训练历史
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    return history




def test_model(model, test_loader, device='cuda'):
    """测试模型"""
    model.eval()
    test_corrects = 0
    all_preds = []
    all_labels = []
    
    # <<< MODIFICATION START >>>
    all_probs = []  # 新增一个列表来存储概率
    # <<< MODIFICATION END >>>
    
    with torch.no_grad():
        for batch in test_loader:
            landmarks = batch['landmarks'].to(device)
            lip_roi = batch['lip_roi'].to(device)
            mfcc = batch['mfcc'].to(device)
            labels = batch['label'].view(-1).to(device)
            
            outputs, _ = model(landmarks, lip_roi, mfcc)
            
            # <<< MODIFICATION START >>>
            # 将模型的logits输出转换为概率
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            # <<< MODIFICATION END >>>

            _, preds = torch.max(outputs, 1)
            
            test_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # <<< MODIFICATION START >>>
    # 将批次的概率列表拼接成一个大的numpy数组
    all_probs = np.concatenate(all_probs, axis=0)
    # <<< MODIFICATION END >>>

    test_acc = test_corrects.double() / len(test_loader.dataset)
    print(f'测试准确率: {test_acc:.4f}')
    
    # 保存预测结果
    results = {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,  # 新增概率结果
        'accuracy': test_acc.cpu().numpy()
    }
    
    with open('test_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results
# def test_model(model, test_loader, device='cuda'):
#     """测试模型"""
#     model.eval()
#     test_corrects = 0
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for batch in test_loader:
#             landmarks = batch['landmarks'].to(device)
#             lip_roi = batch['lip_roi'].to(device)
#             mfcc = batch['mfcc'].to(device)
#             # labels = batch['label'].squeeze().to(device)
#             labels = batch['label'].view(-1).to(device)
            
#             outputs, _ = model(landmarks, lip_roi, mfcc)
#             _, preds = torch.max(outputs, 1)
            
#             test_corrects += torch.sum(preds == labels.data)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
    
#     test_acc = test_corrects.double() / len(test_loader.dataset)
#     print(f'测试准确率: {test_acc:.4f}')
    
#     # 保存预测结果
#     results = {
#         'predictions': all_preds,
#         'labels': all_labels,
#         'accuracy': test_acc.cpu().numpy()
#     }
    
#     with open('test_results.pkl', 'wb') as f:
#         pickle.dump(results, f)
    
#     return results

# 主函数
def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建数据集
    train_loader, val_loader, test_loader = create_datasets(
        args.transformer_dir, args.gan_dir, args.diffusion_dir,
        args.vae_dir, args.cnn_dir,
        batch_size=args.batch_size
    )
    
    # 创建模型
    model = AVTSTAN(num_classes=5)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练模型
    history = train_model(
        model, train_loader, val_loader, 
        criterion, optimizer, scheduler,
        num_epochs=args.epochs, device=device
    )
    
    # 加载最佳模型并测试
    model.load_state_dict(torch.load(args.checkpoint))
    test_results = test_model(model, test_loader, device=device)
    
    # 可视化结果
    visualize_results(history, test_results)



# def visualize_results(history, test_results):
#     """可视化训练和测试结果"""
#     import matplotlib.pyplot as plt
#     from sklearn.metrics import confusion_matrix, classification_report
#     import seaborn as sns
    
#     # 绘制训练和验证损失
#     plt.figure(figsize=(12, 5))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(history['train_loss'], label='Training Loss')
#     plt.plot(history['val_loss'], label='Validation Loss')
#     plt.title('Model Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
    
#     # 绘制训练和验证准确率
#     plt.subplot(1, 2, 2)
#     plt.plot(history['train_acc'], label='Training Accuracy')
#     plt.plot(history['val_acc'], label='Validation Accuracy')
#     plt.title('Model Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.savefig('training_history.png')
#     plt.show()
    
#     target_names_5_classes = ['Transformer', 'GAN', 'Diffusion', 'VAE', 'CNN']
#     # 绘制混淆矩阵
#     cm = confusion_matrix(test_results['labels'], test_results['predictions'])
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=target_names_5_classes,
#                 yticklabels=target_names_5_classes)
#     plt.title('混淆矩阵')
#     plt.ylabel('真实标签')
#     plt.xlabel('预测标签')
#     plt.savefig('confusion_matrix.png')
#     plt.show()
    
#     # 打印分类报告
#     print("分类报告:")
#     print(classification_report(
#         test_results['labels'], 
#         test_results['predictions'],
#         target_names=target_names_5_classes
#     ))
def visualize_results(history, test_results):
    """可视化训练和测试结果"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import label_binarize
    import numpy as np
    import seaborn as sns
    
    # 绘制训练和验证损失
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制训练和验证准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    target_names_5_classes = ['Transformer', 'GAN', 'Diffusion', 'VAE', 'CNN']
    
        # 绘制混淆矩阵
    cm = confusion_matrix(test_results['labels'], test_results['predictions'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names_5_classes,
                yticklabels=target_names_5_classes)
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # 打印分类报告
    print("="*50)
    print("分类报告 (Classification Report):")
    print("="*50)
    print(classification_report(
        test_results['labels'], 
        test_results['predictions'],
        target_names=target_names_5_classes
    ))
    
    # <<< MODIFICATION START >>>
    # 计算并打印 AP 和 AUC
    print("="*50)
    print("AP 和 AUC 报告:")
    print("="*50)

    y_true = test_results['labels']
    y_prob = test_results['probabilities']
    n_classes = len(target_names_5_classes)

    # 需要将真实标签进行 one-hot 编码 (binarize)
    y_true_binarized = label_binarize(y_true, classes=range(n_classes))

    # 计算每个类别的 AP 和 AUC
    for i in range(n_classes):
        class_name = target_names_5_classes[i]
        
        # 计算 AP (Average Precision)
        ap = average_precision_score(y_true_binarized[:, i], y_prob[:, i])
        
        # 计算 AUC
        # 确保该类别在测试集中至少有一个正样本和一个负样本
        if len(np.unique(y_true_binarized[:, i])) == 2:
            auc = roc_auc_score(y_true_binarized[:, i], y_prob[:, i])
            print(f"类别: {class_name: <12} | AUC: {auc:.4f} | AP (mAP): {ap:.4f}")
        else:
            print(f"类别: {class_name: <12} | AUC: N/A (单一类别) | AP (mAP): {ap:.4f}")

    # 计算宏平均 (Macro Average) AUC 和 AP
    # 对于AUC，可以直接使用 scikit-learn 的多分类支持
    macro_auc = roc_auc_score(y_true_binarized, y_prob, multi_class='ovr', average='macro')
    # 对于AP，需要手动计算平均值
    macro_ap = average_precision_score(y_true_binarized, y_prob, average='macro')

    print("-" * 50)
    print(f"宏平均 (Macro Avg) | AUC: {macro_auc:.4f} | AP (mAP): {macro_ap:.4f}")
    print("="*50)
    # <<< MODIFICATION END >>>


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage-2 generator attribution training (AVTSTAN)')
    parser.add_argument('--transformer_dir', type=str, default='data/transformer')
    parser.add_argument('--gan_dir', type=str, default='data/GAN')
    parser.add_argument('--diffusion_dir', type=str, default='data/diffusion')
    parser.add_argument('--vae_dir', type=str, default='data/VAE')
    parser.add_argument('--cnn_dir', type=str, default='data/CNN_makeittalk')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--checkpoint', type=str, default='best_model.pth')
    main(parser.parse_args())