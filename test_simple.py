"""
SegMamba Simple - 测试、推理和可视化脚本
支持：
- 单张图像推理
- 批量测试评估
- 可视化结果
- 保存预测mask
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# 导入模型
try:
    from model.model_simple import segmamba2d_simple_tiny, segmamba2d_simple_small, segmamba2d_simple_base
    from data.dataloader import NPYSegmentationDataset, get_default_transforms
except ImportError as e:
    print(f"Error: {e}")
    print("Please ensure files are in correct locations")
    exit(1)


# ============================================================================
# Metrics计算
# ============================================================================

def calculate_metrics(pred_binary, gt_binary):
    """
    计算Dice, IoU, Precision, Recall
    
    Args:
        pred_binary: (H, W) binary prediction
        gt_binary: (H, W) binary ground truth
    
    Returns:
        dict with metrics
    """
    pred_binary = pred_binary.astype(bool)
    gt_binary = gt_binary.astype(bool)
    
    # Intersection and Union
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum()
    pred_sum = pred_binary.sum()
    gt_sum = gt_binary.sum()
    
    # Dice Score
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    
    # IoU
    iou = (intersection + 1e-6) / (pred_sum + gt_sum - intersection + 1e-6)
    
    # Precision and Recall
    precision = (intersection + 1e-6) / (pred_sum + 1e-6)
    recall = (intersection + 1e-6) / (gt_sum + 1e-6)
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
    }


# ============================================================================
# 推理函数
# ============================================================================

@torch.no_grad()
def predict_single_image(model, image, device, threshold=0.5):
    """
    对单张图像进行推理
    
    Args:
        model: 模型
        image: (1, H, W) or (H, W) numpy array
        device: torch device
        threshold: 二值化阈值
    
    Returns:
        pred_mask: (H, W) binary mask
        pred_prob: (H, W) probability map
    """
    model.eval()
    
    # 预处理
    if image.ndim == 2:
        image = image[np.newaxis, ...]  # (1, H, W)
    
    # 归一化到[0, 1]
    if image.max() > 1.0:
        image = image / 255.0
    
    # 转为tensor
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)  # (1, 1, H, W)
    
    # 推理
    output = model(image_tensor)  # (1, 1, H, W)
    
    # 处理deep supervision输出
    if isinstance(output, dict):
        output = output['out']
    
    # Sigmoid
    pred_prob = output.sigmoid().squeeze().cpu().numpy()  # (H, W)
    
    # 二值化
    pred_mask = (pred_prob > threshold).astype(np.uint8)
    
    return pred_mask, pred_prob


# ============================================================================
# 可视化函数
# ============================================================================

def visualize_prediction(image, gt_mask, pred_mask, save_path=None, show_prob=None):
    """
    可视化预测结果
    
    Args:
        image: (H, W) 原图
        gt_mask: (H, W) GT mask
        pred_mask: (H, W) 预测mask
        save_path: 保存路径
        show_prob: (H, W) 概率图（可选）
    """
    # 归一化image到[0, 1]
    if image.max() > 1.0:
        image = image / 255.0
    
    # 创建子图
    n_cols = 5 if show_prob is not None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4))
    
    # 1. 原图
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. GT Mask
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # 3. 预测Mask
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Prediction', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # 4. Overlay
    overlay = np.zeros((*image.shape, 3))
    overlay[..., 0] = image  # R channel: 原图
    overlay[..., 1] = image  # G channel: 原图
    overlay[..., 2] = image  # B channel: 原图
    
    # GT: 绿色
    overlay[gt_mask > 0, 1] = 0.8
    # 预测: 红色
    overlay[pred_mask > 0, 0] = 0.8
    # 重叠: 黄色 (红+绿)
    
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (GT=Green, Pred=Red)', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    # 5. 概率图（可选）
    if show_prob is not None:
        im = axes[4].imshow(show_prob, cmap='jet', vmin=0, vmax=1)
        axes[4].set_title('Probability Map', fontsize=12, fontweight='bold')
        axes[4].axis('off')
        plt.colorbar(im, ax=axes[4], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# 批量测试
# ============================================================================

@torch.no_grad()
def evaluate_dataset(model, dataloader, device, threshold=0.5, save_dir=None, visualize=False, num_vis=None):
    """
    在整个数据集上评估
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: torch device
        threshold: 二值化阈值
        save_dir: 保存目录
        visualize: 是否可视化
        num_vis: 可视化数量限制（None=全部）
    
    Returns:
        dict with average metrics
    """
    model.eval()
    
    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []
    
    # 收集所有数据用于批量可视化
    all_samples = []
    
    if visualize and save_dir:
        vis_dir = Path(save_dir) / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nEvaluating on dataset...")
    pbar = tqdm(dataloader, desc='Evaluating')
    
    sample_count = 0
    
    for idx, batch in enumerate(pbar):
        images = batch['image'].to(device)  # (B, 1, H, W)
        masks = batch['mask'].to(device)    # (B, H, W)
        
        # 推理
        outputs = model(images)
        
        # 处理deep supervision
        if isinstance(outputs, dict):
            outputs = outputs['out']
        
        # Sigmoid和二值化
        pred_probs = outputs.sigmoid()
        pred_binary = (pred_probs > threshold).float()
        
        # 转为numpy
        pred_binary_np = pred_binary.squeeze(1).cpu().numpy()  # (B, H, W)
        masks_np = masks.cpu().numpy()  # (B, H, W)
        images_np = images.squeeze(1).cpu().numpy()  # (B, H, W)
        pred_probs_np = pred_probs.squeeze(1).cpu().numpy()  # (B, H, W)
        
        # 计算每个样本的metrics
        batch_size = images.shape[0]
        for i in range(batch_size):
            metrics = calculate_metrics(pred_binary_np[i], masks_np[i])
            
            all_dice.append(metrics['dice'])
            all_iou.append(metrics['iou'])
            all_precision.append(metrics['precision'])
            all_recall.append(metrics['recall'])
            
            # 收集数据用于批量可视化
            if visualize and (num_vis is None or sample_count < num_vis):
                all_samples.append({
                    'image': images_np[i],
                    'gt_mask': masks_np[i],
                    'pred_mask': pred_binary_np[i],
                    'pred_prob': pred_probs_np[i],
                    'dice': metrics['dice'],
                    'idx': sample_count,
                })
                sample_count += 1
        
        # 更新进度条
        pbar.set_postfix({
            'Dice': f'{np.mean(all_dice):.4f}',
            'IoU': f'{np.mean(all_iou):.4f}'
        })
    
    # 批量可视化
    if visualize and save_dir and all_samples:
        print(f"\nCreating visualizations for {len(all_samples)} samples...")
        batch_visualize(all_samples, vis_dir)
    
    # 计算平均metrics
    results = {
        'dice': np.mean(all_dice),
        'iou': np.mean(all_iou),
        'precision': np.mean(all_precision),
        'recall': np.mean(all_recall),
        'dice_std': np.std(all_dice),
        'iou_std': np.std(all_iou),
        'num_samples': len(all_dice),
    }
    
    return results


def batch_visualize(samples, save_dir, ncols=4):
    """
    批量创建可视化图像
    
    Args:
        samples: list of dicts，每个dict包含image, gt_mask, pred_mask, pred_prob, dice, idx
        save_dir: 保存目录
        ncols: 每行显示多少个样本
    """
    from matplotlib.gridspec import GridSpec
    
    print("  Creating batch visualizations...")
    
    # 计算需要多少个大图
    samples_per_fig = ncols * 3  # 每个大图显示3行
    num_figs = (len(samples) + samples_per_fig - 1) // samples_per_fig
    
    for fig_idx in range(num_figs):
        start_idx = fig_idx * samples_per_fig
        end_idx = min(start_idx + samples_per_fig, len(samples))
        fig_samples = samples[start_idx:end_idx]
        
        # 计算行数
        nrows = (len(fig_samples) + ncols - 1) // ncols
        
        # 创建大图
        fig = plt.figure(figsize=(ncols * 5, nrows * 5))
        gs = GridSpec(nrows, ncols, figure=fig, hspace=0.3, wspace=0.2)
        
        for i, sample in enumerate(fig_samples):
            row = i // ncols
            col = i % ncols
            
            # 创建子图的子图 (5个小图)
            ax_main = fig.add_subplot(gs[row, col])
            ax_main.axis('off')
            
            # 创建5个更小的子图
            gs_inner = GridSpec(2, 3, hspace=0.3, wspace=0.2,
                               left=col/ncols + 0.02, right=(col+1)/ncols - 0.02,
                               bottom=1-(row+1)/nrows + 0.02, top=1-row/nrows - 0.02)
            
            image = sample['image']
            gt_mask = sample['gt_mask']
            pred_mask = sample['pred_mask']
            pred_prob = sample['pred_prob']
            dice = sample['dice']
            
            # 归一化
            if image.max() > 1.0:
                image = image / 255.0
            
            # 1. 原图
            ax1 = fig.add_subplot(gs_inner[0, 0])
            ax1.imshow(image, cmap='gray')
            ax1.set_title('Image', fontsize=8)
            ax1.axis('off')
            
            # 2. GT
            ax2 = fig.add_subplot(gs_inner[0, 1])
            ax2.imshow(gt_mask, cmap='gray')
            ax2.set_title('GT', fontsize=8)
            ax2.axis('off')
            
            # 3. 预测
            ax3 = fig.add_subplot(gs_inner[0, 2])
            ax3.imshow(pred_mask, cmap='gray')
            ax3.set_title('Pred', fontsize=8)
            ax3.axis('off')
            
            # 4. Overlay
            ax4 = fig.add_subplot(gs_inner[1, 0])
            overlay = np.zeros((*image.shape, 3))
            overlay[..., 0] = image
            overlay[..., 1] = image
            overlay[..., 2] = image
            overlay[gt_mask > 0, 1] = 0.8
            overlay[pred_mask > 0, 0] = 0.8
            ax4.imshow(overlay)
            ax4.set_title('Overlay', fontsize=8)
            ax4.axis('off')
            
            # 5. 概率图
            ax5 = fig.add_subplot(gs_inner[1, 1])
            im = ax5.imshow(pred_prob, cmap='jet', vmin=0, vmax=1)
            ax5.set_title('Prob', fontsize=8)
            ax5.axis('off')
            
            # 6. Dice文本
            ax6 = fig.add_subplot(gs_inner[1, 2])
            ax6.text(0.5, 0.5, f'Dice\n{dice:.3f}',
                    ha='center', va='center', fontsize=12, fontweight='bold')
            ax6.axis('off')
        
        # 保存
        save_path = save_dir / f'batch_{fig_idx:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {save_path.name} ({len(fig_samples)} samples)")
    
    print(f"  ✓ Created {num_figs} batch visualizations")


# ============================================================================
# Main
# ============================================================================

def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # 创建输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    # ===== 1. 加载模型 =====
    print("[1] Loading model...")
    
    if args.model_size == 'tiny':
        model = segmamba2d_simple_tiny(num_classes=args.num_classes)
    elif args.model_size == 'small':
        model = segmamba2d_simple_small(num_classes=args.num_classes)
    elif args.model_size == 'base':
        model = segmamba2d_simple_base(num_classes=args.num_classes)
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
    
    # 加载权重
    if args.checkpoint:
        print(f"  Loading checkpoint: {args.checkpoint}")
        
        # 尝试加载checkpoint或state_dict
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 完整的checkpoint
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'val_metrics' in checkpoint:
                print(f"  Checkpoint Dice: {checkpoint['val_metrics'].get('dice_score', 'N/A')}")
        else:
            # 只有state_dict
            model.load_state_dict(checkpoint)
        
        print("  ✓ Model loaded successfully!")
    else:
        print("  Warning: No checkpoint provided, using random weights")
    
    model = model.to(device)
    model.eval()
    
    # ===== 2. 加载数据 =====
    print("\n[2] Loading data...")
    
    dataset = NPYSegmentationDataset(
        root_dir=args.data_root,
        split=args.split,
        image_size=args.image_size,
        transforms=get_default_transforms(args.image_size, 'val'),
        num_classes=args.num_classes,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    print(f"  Dataset: {len(dataset)} samples")
    print(f"  Split: {args.split}")
    
    # ===== 3. 评估 =====
    print("\n[3] Evaluating...")
    
    results = evaluate_dataset(
        model,
        dataloader,
        device,
        threshold=args.threshold,
        save_dir=output_dir,
        visualize=args.visualize,
        num_vis=args.num_vis,
    )
    
    # ===== 4. 打印和保存结果 =====
    print("\n" + "=" * 80)
    print("Evaluation Results:")
    print("=" * 80)
    print(f"  Dice Score:     {results['dice']:.4f} ± {results['dice_std']:.4f}")
    print(f"  IoU:            {results['iou']:.4f} ± {results['iou_std']:.4f}")
    print(f"  Precision:      {results['precision']:.4f}")
    print(f"  Recall:         {results['recall']:.4f}")
    print(f"  Num Samples:    {results['num_samples']}")
    print("=" * 80)
    
    # 保存结果
    if output_dir:
        results_file = output_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {results_file}")
        
        # 保存文本格式
        results_txt = output_dir / 'results.txt'
        with open(results_txt, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Evaluation Results\n")
            f.write("=" * 80 + "\n")
            f.write(f"Model: {args.model_size}\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Dataset: {args.data_root}\n")
            f.write(f"Split: {args.split}\n")
            f.write(f"Threshold: {args.threshold}\n")
            f.write("\nMetrics:\n")
            f.write(f"  Dice Score:     {results['dice']:.4f} ± {results['dice_std']:.4f}\n")
            f.write(f"  IoU:            {results['iou']:.4f} ± {results['iou_std']:.4f}\n")
            f.write(f"  Precision:      {results['precision']:.4f}\n")
            f.write(f"  Recall:         {results['recall']:.4f}\n")
            f.write(f"  Num Samples:    {results['num_samples']}\n")
            f.write("=" * 80 + "\n")
        print(f"✓ Results saved to {results_txt}")
        
        if args.visualize:
            print(f"✓ Visualizations saved to {output_dir / 'visualizations'}")
    
    print("\n✓ Evaluation completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SegMamba Simple - Test and Visualization')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to processed dataset')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--num_classes', type=int, default=1,
                       help='Number of classes')
    
    # 模型参数
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['tiny', 'small', 'base'],
                       help='Model size')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # 推理参数
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary prediction')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./test_results',
                       help='Output directory')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize predictions')
    parser.add_argument('--num_vis', type=int, default=None,
                       help='Number of samples to visualize (None=all)')
    
    args = parser.parse_args()
    
    main(args)