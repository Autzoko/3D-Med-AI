"""
SegMamba + Mask2Former 测试/推理脚本

功能：
1. 加载训练好的模型
2. 在验证集/测试集上评估
3. 可视化预测结果
4. 保存结果图像
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入模型和数据
try:
    from model.model import segmamba_mask2former_tiny, segmamba_mask2former_small, segmamba_mask2former_base
    from data.dataloader import NPYSegmentationDataset
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure model.py and dataloader.py are in the correct paths")
    exit(1)


@torch.no_grad()
def calculate_metrics(pred_mask, gt_mask):
    """
    计算评估指标
    
    Args:
        pred_mask: (H, W) binary prediction
        gt_mask: (H, W) ground truth
        
    Returns:
        dict with metrics
    """
    pred = pred_mask.flatten()
    gt = gt_mask.flatten()
    
    # Dice Score
    intersection = (pred * gt).sum()
    dice = (2 * intersection + 1e-6) / (pred.sum() + gt.sum() + 1e-6)
    
    # IoU
    union = (pred + gt).clamp(0, 1).sum()
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Precision & Recall
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
    }


def predict_single_image(model, image, device, threshold=0.5):
    """
    对单张图像进行预测
    
    Args:
        model: 模型
        image: (1, C, H, W) or (C, H, W)
        device: 设备
        threshold: 二值化阈值
        
    Returns:
        pred_mask: (H, W) numpy array
        confidence: float
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    image = image.to(device)
    
    # 前向传播
    outputs = model(image)
    
    # 提取预测
    pred_logits = outputs['pred_logits']  # (1, Q, C+1)
    pred_masks = outputs['pred_masks']    # (1, Q, H/4, W/4)
    
    # 选择confidence最高的query
    foreground_probs = pred_logits.softmax(dim=-1)[..., :-1]  # 排除background
    confidences = foreground_probs.max(dim=-1)[0]  # (1, Q)
    best_query = confidences.argmax(dim=1)  # (1,)
    
    # 提取最佳mask
    final_mask = pred_masks[0, best_query[0]]  # (H/4, W/4)
    
    # 上采样到原图尺寸
    original_size = image.shape[-2:]
    final_mask = F.interpolate(
        final_mask.unsqueeze(0).unsqueeze(0),
        size=original_size,
        mode='bilinear',
        align_corners=False
    ).squeeze()  # (H, W)
    
    # 二值化
    pred_binary = (final_mask.sigmoid() > threshold).float()
    
    # 返回numpy
    pred_mask = pred_binary.cpu().numpy()
    confidence = confidences[0, best_query[0]].item()
    
    return pred_mask, confidence


def visualize_prediction(image, gt_mask, pred_mask, metrics, save_path=None):
    """
    可视化预测结果
    
    Args:
        image: (C, H, W) or (H, W) 原始图像
        gt_mask: (H, W) GT mask
        pred_mask: (H, W) 预测mask
        metrics: dict 评估指标
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 原始图像
    if image.dim() == 3 and image.shape[0] == 1:
        img_show = image[0].cpu().numpy()
    else:
        img_show = image.cpu().numpy() if torch.is_tensor(image) else image
    
    axes[0].imshow(img_show, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # GT mask
    axes[1].imshow(gt_mask, cmap='jet', alpha=0.6)
    axes[1].imshow(img_show, cmap='gray', alpha=0.4)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # 预测mask
    axes[2].imshow(pred_mask, cmap='jet', alpha=0.6)
    axes[2].imshow(img_show, cmap='gray', alpha=0.4)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Overlay对比
    overlay = np.zeros((*gt_mask.shape, 3))
    overlay[gt_mask > 0] = [0, 1, 0]      # GT: 绿色
    overlay[pred_mask > 0] = [1, 0, 0]    # Pred: 红色
    overlap = (gt_mask > 0) & (pred_mask > 0)
    overlay[overlap] = [1, 1, 0]          # Overlap: 黄色
    
    axes[3].imshow(overlay)
    axes[3].set_title(f"Overlay (Dice: {metrics['dice']:.4f})\nGreen=GT, Red=Pred, Yellow=Overlap")
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_dataset(model, dataloader, device, output_dir=None, num_visualize=10):
    """
    在整个数据集上评估
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        output_dir: 可视化保存目录（可选）
        num_visualize: 可视化样本数量
        
    Returns:
        metrics: 平均指标
    """
    model.eval()
    
    all_metrics = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': [],
    }
    
    if output_dir:
        vis_dir = Path(output_dir) / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nEvaluating...")
    pbar = tqdm(dataloader, desc='Testing')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].cpu().numpy()
        
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            # 预测
            pred_mask, confidence = predict_single_image(
                model, images[i], device
            )
            
            gt_mask = masks[i, 0] if masks.ndim == 4 else masks[i]
            
            # 计算指标
            metrics = calculate_metrics(
                torch.from_numpy(pred_mask),
                torch.from_numpy(gt_mask)
            )
            
            # 累积
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
            
            # 可视化前N个样本
            if output_dir and batch_idx * batch_size + i < num_visualize:
                save_path = vis_dir / f'sample_{batch_idx * batch_size + i:04d}.png'
                visualize_prediction(
                    images[i],
                    gt_mask,
                    pred_mask,
                    metrics,
                    save_path
                )
            
            # 更新进度条
            avg_dice = np.mean(all_metrics['dice'])
            pbar.set_postfix({'Dice': f'{avg_dice:.4f}'})
    
    # 计算平均值
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    return avg_metrics


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    # ===== 1. 加载数据 =====
    print("\n[1] Loading dataset...")
    
    test_dataset = NPYSegmentationDataset(
        root_dir=args.data_root,
        split=args.split,
        image_size=args.image_size,
        transforms=None,  # 测试时不用数据增强
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"  {args.split.capitalize()} set: {len(test_dataset)} samples")
    
    # ===== 2. 创建模型 =====
    print("\n[2] Loading model...")
    
    if args.model_size == 'tiny':
        model = segmamba_mask2former_tiny(
            num_classes=args.num_classes,
            num_queries=args.num_queries,
        )
    elif args.model_size == 'small':
        model = segmamba_mask2former_small(
            num_classes=args.num_classes,
            num_queries=args.num_queries,
        )
    elif args.model_size == 'base':
        model = segmamba_mask2former_base(
            num_classes=args.num_classes,
            num_queries=args.num_queries,
        )
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
    
    # 加载权重
    if args.checkpoint:
        print(f"  Loading checkpoint: {args.checkpoint}")
        if args.checkpoint.endswith('.pth') and 'checkpoint' in args.checkpoint:
            # 加载完整checkpoint
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'val_dice_score' in checkpoint:
                print(f"  Checkpoint Dice Score: {checkpoint['val_dice_score']:.4f}")
            print(f"  Checkpoint Epoch: {checkpoint.get('epoch', 'unknown')}")
        else:
            # 加载state_dict
            state_dict = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(state_dict)
    else:
        print("  WARNING: No checkpoint provided, using random weights!")
    
    model = model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {args.model_size}")
    print(f"  Parameters: {num_params / 1e6:.2f}M")
    
    # ===== 3. 评估 =====
    print("\n[3] Evaluating on dataset...")
    
    metrics = evaluate_dataset(
        model, 
        test_loader, 
        device,
        output_dir=output_dir,
        num_visualize=args.num_visualize
    )
    
    # ===== 4. 打印结果 =====
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Dataset: {args.split}")
    print(f"Samples: {len(test_dataset)}")
    print(f"\nMetrics:")
    print(f"  Dice Score:  {metrics['dice']:.4f}")
    print(f"  IoU:         {metrics['iou']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print("=" * 80)
    
    # 保存结果
    if output_dir:
        results_file = output_dir / 'results.txt'
        with open(results_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EVALUATION RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Dataset: {args.split}\n")
            f.write(f"Samples: {len(test_dataset)}\n\n")
            f.write("Metrics:\n")
            f.write(f"  Dice Score:  {metrics['dice']:.4f}\n")
            f.write(f"  IoU:         {metrics['iou']:.4f}\n")
            f.write(f"  Precision:   {metrics['precision']:.4f}\n")
            f.write(f"  Recall:      {metrics['recall']:.4f}\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n✓ Results saved to: {results_file}")
        print(f"✓ Visualizations saved to: {output_dir / 'visualizations'}")
    
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SegMamba + Mask2Former Testing')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to processed dataset')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Which split to evaluate')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    
    # 模型参数
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['tiny', 'small', 'base'],
                       help='Model size')
    parser.add_argument('--num_classes', type=int, default=1,
                       help='Number of classes')
    parser.add_argument('--num_queries', type=int, default=20,
                       help='Number of queries')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint (.pth file)')
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--num_visualize', type=int, default=10,
                       help='Number of samples to visualize')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./test_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    main(args)