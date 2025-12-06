"""
SegMamba Simple - 单张图像推理和可视化
快速测试模型效果
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 导入模型
try:
    from model.model_simple import segmamba2d_simple_tiny, segmamba2d_simple_small, segmamba2d_simple_base
except ImportError:
    print("Error: Cannot import model. Please check file locations.")
    exit(1)


def load_image(image_path):
    """加载.npy图像"""
    if image_path.endswith('.npy'):
        image = np.load(image_path)
    else:
        # 如果是png/jpg，用opencv
        import cv2
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    return image


@torch.no_grad()
def predict(model, image, device, threshold=0.5):
    """推理"""
    model.eval()
    
    # 预处理
    if image.ndim == 2:
        image = image[np.newaxis, ...]  # (1, H, W)
    
    # 归一化
    if image.max() > 1.0:
        image = image / 255.0
    
    # 转tensor
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)  # (1, 1, H, W)
    
    # 推理
    output = model(image_tensor)
    
    # 处理deep supervision
    if isinstance(output, dict):
        output = output['out']
    
    # Sigmoid
    pred_prob = output.sigmoid().squeeze().cpu().numpy()
    
    # 二值化
    pred_mask = (pred_prob > threshold).astype(np.uint8)
    
    return pred_mask, pred_prob


def visualize(image, pred_mask, pred_prob, gt_mask=None, save_path=None):
    """可视化"""
    if image.max() > 1.0:
        image = image / 255.0
    
    # 创建子图
    if gt_mask is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 预测mask
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 概率图
    im = axes[2].imshow(pred_prob, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title('Probability Map', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    if gt_mask is not None:
        # GT
        axes[3].imshow(gt_mask, cmap='gray')
        axes[3].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[3].axis('off')
        
        # Overlay
        overlay = np.zeros((*image.shape, 3))
        overlay[..., 0] = image
        overlay[..., 1] = image
        overlay[..., 2] = image
        overlay[gt_mask > 0, 1] = 0.8  # GT: 绿色
        overlay[pred_mask > 0, 0] = 0.8  # Pred: 红色
        
        axes[4].imshow(overlay)
        axes[4].set_title('Overlay (GT=Green, Pred=Red)', fontsize=14, fontweight='bold')
        axes[4].axis('off')
        
        # Dice计算
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum()
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        
        axes[5].text(0.5, 0.5, f'Dice Score:\n{dice:.4f}', 
                    ha='center', va='center', fontsize=24, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[5].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # 加载模型
    print("[1] Loading model...")
    
    if args.model_size == 'tiny':
        model = segmamba2d_simple_tiny(num_classes=1)
    elif args.model_size == 'small':
        model = segmamba2d_simple_small(num_classes=1)
    elif args.model_size == 'base':
        model = segmamba2d_simple_base(num_classes=1)
    
    # 加载权重
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print("  ✓ Model loaded\n")
    
    # 加载图像
    print("[2] Loading image...")
    image = load_image(args.image)
    print(f"  Image shape: {image.shape}")
    
    if args.gt_mask:
        gt_mask = load_image(args.gt_mask)
        print(f"  GT mask shape: {gt_mask.shape}")
    else:
        gt_mask = None
    
    # 推理
    print("\n[3] Predicting...")
    pred_mask, pred_prob = predict(model, image, device, args.threshold)
    print(f"  Prediction shape: {pred_mask.shape}")
    print(f"  Foreground ratio: {pred_mask.sum() / pred_mask.size:.2%}")
    
    if gt_mask is not None:
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum()
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        print(f"  Dice Score: {dice:.4f}")
    
    # 可视化
    print("\n[4] Visualizing...")
    save_path = args.output if args.output else None
    visualize(image, pred_mask, pred_prob, gt_mask, save_path)
    
    print("\n✓ Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Image Inference')
    
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image (.npy or .png)')
    parser.add_argument('--gt_mask', type=str, default=None,
                       help='Path to ground truth mask (optional)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['tiny', 'small', 'base'],
                       help='Model size')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Binary threshold')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for visualization')
    
    args = parser.parse_args()
    
    main(args)