"""
BUSI数据集预处理脚本 - 修复版
修复：确保Image和Mask尺寸匹配，统一resize

关键改进：
1. ✅ 确保每个Image和Mask尺寸完全一致
2. ✅ 统一resize到目标尺寸（默认256x256）
3. ✅ 正确的插值方法（图像用LINEAR，mask用NEAREST）
4. ✅ 保存为uint8格式节省空间
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import argparse


def preprocess_busi_dataset(
    source_dir: str,
    output_dir: str,
    target_size: int = 256,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    预处理BUSI数据集 - 修复版
    
    Args:
        source_dir: BUSI原始数据集目录
        output_dir: 输出目录
        target_size: 统一resize的目标尺寸
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    print("="*80)
    print("BUSI Dataset Preprocessing - FIXED VERSION")
    print("="*80)
    print(f"Target size: {target_size}x{target_size}")
    
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    image_dir = output_dir / 'images'
    mask_dir = output_dir / 'masks'
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有样本
    print("\n[Step 1] Collecting and processing samples...")
    samples = []
    sample_id = 0
    skipped = 0
    
    categories = ['benign', 'malignant', 'normal']
    category_mapping = {
        'benign': 1,
        'malignant': 1,  # 合并为前景类
        'normal': 0,     # 背景类
    }
    
    for category in categories:
        category_dir = source_dir / category
        if not category_dir.exists():
            print(f"Warning: {category} directory not found, skipping...")
            continue
        
        # 获取所有图像文件（不含mask）
        image_files = [
            f for f in category_dir.iterdir()
            if f.suffix == '.png' and '_mask' not in f.name
        ]
        
        print(f"\nProcessing {category}: {len(image_files)} images")
        
        for img_file in tqdm(image_files, desc=f"  {category}"):
            try:
                # 读取图像
                image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Failed to load: {img_file}")
                    skipped += 1
                    continue
                
                # 读取mask
                mask_file = category_dir / (img_file.stem + '_mask.png')
                
                if mask_file.exists():
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        print(f"Failed to load mask: {mask_file}")
                        skipped += 1
                        continue
                else:
                    # normal类或缺失mask，创建全0 mask
                    mask = np.zeros_like(image, dtype=np.uint8)
                
                # ===== 关键修复1：确保尺寸匹配 =====
                if image.shape != mask.shape:
                    print(f"  Fixing size mismatch: img {image.shape} vs mask {mask.shape}")
                    # 将mask resize到与image相同尺寸
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
                
                # ===== 关键修复2：统一resize =====
                # 图像用双线性插值
                image_resized = cv2.resize(image, (target_size, target_size), 
                                         interpolation=cv2.INTER_LINEAR)
                # Mask用最近邻插值（保持标签不变）
                mask_resized = cv2.resize(mask, (target_size, target_size), 
                                        interpolation=cv2.INTER_NEAREST)
                
                # ===== 关键修复3：正确的二值化和类别映射 =====
                # 二值化mask
                mask_binary = (mask_resized > 127).astype(np.uint8)
                # 应用类别映射
                mask_final = mask_binary * category_mapping[category]
                
                # 验证
                assert image_resized.shape == mask_final.shape, \
                    f"Shape mismatch after resize: {image_resized.shape} vs {mask_final.shape}"
                assert image_resized.shape == (target_size, target_size), \
                    f"Wrong size: {image_resized.shape}"
                
                # 保存为npy（uint8格式节省空间）
                file_id = f"{sample_id:04d}"
                np.save(image_dir / f"{file_id}.npy", image_resized.astype(np.uint8))
                np.save(mask_dir / f"{file_id}.npy", mask_final.astype(np.uint8))
                
                samples.append({
                    'id': file_id,
                    'category': category,
                    'category_id': category_mapping[category],
                    'original_file': str(img_file.name),
                    'original_shape': image.shape,
                    'resized_shape': image_resized.shape,
                })
                
                sample_id += 1
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                skipped += 1
                continue
    
    print(f"\nTotal samples processed: {len(samples)}")
    if skipped > 0:
        print(f"Skipped: {skipped} samples")
    
    # 划分数据集
    print(f"\n[Step 2] Splitting dataset...")
    print(f"  Train: {train_ratio:.1%}")
    print(f"  Val: {val_ratio:.1%}")
    print(f"  Test: {test_ratio:.1%}")
    
    np.random.seed(seed)
    indices = np.random.permutation(len(samples))
    
    train_size = int(len(samples) * train_ratio)
    val_size = int(len(samples) * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    splits = {
        'train': [samples[i]['id'] for i in train_indices],
        'val': [samples[i]['id'] for i in val_indices],
        'test': [samples[i]['id'] for i in test_indices],
    }
    
    # 保存split.json
    split_file = output_dir / 'split.json'
    with open(split_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nSplit saved to: {split_file}")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val: {len(splits['val'])} samples")
    print(f"  Test: {len(splits['test'])} samples")
    
    # 保存元数据
    metadata = {
        'source': str(source_dir),
        'target_size': target_size,
        'total_samples': len(samples),
        'categories': category_mapping,
        'splits': {k: len(v) for k, v in splits.items()},
        'samples': samples,
    }
    
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_file}")
    
    # 统计信息
    print(f"\n[Step 3] Statistics")
    print("-" * 40)
    
    category_counts = {}
    for sample in samples:
        cat = sample['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("Category distribution:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count} ({count/len(samples)*100:.1f}%)")
    
    # 验证数据
    print(f"\n[Step 4] Verification")
    print("-" * 40)
    
    all_match = True
    for i in range(min(5, len(samples))):
        file_id = samples[i]['id']
        img_path = image_dir / f"{file_id}.npy"
        mask_path = mask_dir / f"{file_id}.npy"
        
        img = np.load(img_path)
        mask = np.load(mask_path)
        
        match = "✓" if img.shape == mask.shape else "✗"
        if img.shape != mask.shape:
            all_match = False
        
        print(f"\nSample {file_id}: {match}")
        print(f"  Image: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")
        print(f"  Mask:  shape={mask.shape}, dtype={mask.dtype}, unique={np.unique(mask)}")
    
    if all_match:
        print("\n✓ All samples verified! Image and Mask shapes match!")
    else:
        print("\n✗ WARNING: Some samples have mismatched shapes!")
        return False
    
    print("\n" + "="*80)
    print("Preprocessing completed successfully! ✓")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"  images/: {len(list(image_dir.glob('*.npy')))} files")
    print(f"  masks/: {len(list(mask_dir.glob('*.npy')))} files")
    print(f"  split.json: train/val/test splits")
    print(f"  metadata.json: dataset information")
    print(f"\nAll images and masks are {target_size}x{target_size}")
    print("\nYou can now train with:")
    print(f"  python train_fixed.py --data_root {output_dir}")
    
    return True


def create_visualization(output_dir: str, num_samples: int = 5):
    """创建可视化验证预处理结果"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return
    
    output_dir = Path(output_dir)
    image_dir = output_dir / 'images'
    mask_dir = output_dir / 'masks'
    
    # 加载split
    with open(output_dir / 'split.json', 'r') as f:
        splits = json.load(f)
    
    # 可视化训练集的前几个样本
    train_ids = splits['train'][:num_samples]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, file_id in enumerate(train_ids):
        # 加载数据
        image = np.load(image_dir / f"{file_id}.npy")
        mask = np.load(mask_dir / f"{file_id}.npy")
        
        # 显示
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title(f'Image {file_id}\nShape: {image.shape}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f'Mask\nUnique: {np.unique(mask)}')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(image, cmap='gray')
        axes[i, 2].imshow(mask, alpha=0.5, cmap='Reds')
        axes[i, 2].set_title('Overlay')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    vis_file = output_dir / 'preprocessing_visualization.png'
    plt.savefig(vis_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {vis_file}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess BUSI dataset - FIXED VERSION')
    parser.add_argument('--source_dir', type=str, required=True,
                       help='BUSI source directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for NPY files')
    parser.add_argument('--target_size', type=int, default=256,
                       help='Target image size (default: 256)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization')
    
    args = parser.parse_args()
    
    # 预处理
    success = preprocess_busi_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        target_size=args.target_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    
    # 可视化
    if success and args.visualize:
        try:
            create_visualization(args.output_dir, num_samples=5)
        except Exception as e:
            print(f"\nVisualization failed: {e}")
            print("But preprocessing was successful!")