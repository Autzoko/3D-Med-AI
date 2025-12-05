"""
通用分割数据集DataLoader - 基于.npy格式
所有数据集预先转换为标准npy格式，然后使用统一的DataLoader

标准格式:
1. 数据组织: dataset_root/
   ├── images/
   │   ├── 0001.npy  # shape: (H, W) or (H, W, C)
   │   ├── 0002.npy
   │   └── ...
   ├── masks/
   │   ├── 0001.npy  # shape: (H, W), values: 0, 1, 2, ...
   │   ├── 0002.npy
   │   └── ...
   └── split.json (optional)
       {
           "train": ["0001", "0002", ...],
           "val": ["0100", "0101", ...],
           "test": ["0200", "0201", ...]
       }

2. 数据规范:
   - 图像: (H, W) 灰度图 或 (H, W, C) 彩色图，值范围[0, 255] uint8
   - Mask: (H, W) 整数，0=背景，1,2,3...=不同类别

Author: For universal medical image segmentation
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional, Union, Callable
import cv2
from pathlib import Path


class NPYSegmentationDataset(Dataset):
    """
    基于npy格式的通用分割数据集
    
    使用方法:
        # 方式1: 自动匹配images和masks目录
        dataset = NPYSegmentationDataset(
            root_dir='/data/my_dataset',
            split='train',  # 如果有split.json
        )
        
        # 方式2: 手动指定
        dataset = NPYSegmentationDataset(
            image_dir='/data/my_dataset/images',
            mask_dir='/data/my_dataset/masks',
        )
        
        # 方式3: 提供文件列表
        dataset = NPYSegmentationDataset(
            image_dir='/data/my_dataset/images',
            mask_dir='/data/my_dataset/masks',
            file_list=['0001', '0002', '0003'],  # 不带扩展名
        )
    """
    
    def __init__(
        self,
        # 数据路径
        root_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
        mask_dir: Optional[str] = None,
        
        # 数据划分
        split: Optional[str] = None,  # 'train', 'val', 'test'
        split_file: Optional[str] = None,  # split.json路径
        file_list: Optional[List[str]] = None,  # 手动指定文件列表
        
        # 数据处理
        image_size: int = 256,
        transforms: Optional[A.Compose] = None,
        normalize: bool = True,  # 是否归一化到[0,1]
        
        # Mask处理
        num_classes: int = 1,  # 分类数（不含背景）
        mask_values: Optional[List[int]] = None,  # mask中的类别值，例如[0,1,2]
        binary_mask: bool = False,  # 是否转为二值mask（前景/背景）
        
        # Instance分割（用于Mask2Former）
        return_instance: bool = False,
        max_instances: int = 20,
        
        # 其他
        return_path: bool = False,
        grayscale: bool = True,  # 图像是否为灰度
    ):
        """
        Args:
            root_dir: 数据集根目录（包含images/和masks/子目录）
            image_dir: 图像目录（.npy文件）
            mask_dir: mask目录（.npy文件）
            split: 数据划分 ('train', 'val', 'test')
            split_file: 划分文件路径（JSON格式）
            file_list: 手动指定的文件列表（不含扩展名）
            image_size: 目标图像大小
            transforms: 数据增强（albumentations）
            normalize: 是否归一化图像到[0,1]
            num_classes: 类别数（不含背景）
            mask_values: mask中有效的类别值
            binary_mask: 是否转为二值mask（所有前景类合并）
            return_instance: 是否返回instance格式
            max_instances: 最多保留的实例数
            return_path: 是否返回文件路径
            grayscale: 图像是否为灰度
        """
        super().__init__()
        
        # 确定图像和mask目录
        if root_dir is not None:
            self.image_dir = Path(root_dir) / 'images'
            self.mask_dir = Path(root_dir) / 'masks'
            self.root_dir = Path(root_dir)
        else:
            if image_dir is None or mask_dir is None:
                raise ValueError("Must provide either root_dir or (image_dir and mask_dir)")
            self.image_dir = Path(image_dir)
            self.mask_dir = Path(mask_dir)
            self.root_dir = self.image_dir.parent
        
        self.image_size = image_size
        self.transforms = transforms
        self.normalize = normalize
        self.num_classes = num_classes
        self.mask_values = mask_values
        self.binary_mask = binary_mask
        self.return_instance = return_instance
        self.max_instances = max_instances
        self.return_path = return_path
        self.grayscale = grayscale
        
        # 加载文件列表
        if file_list is not None:
            # 使用提供的文件列表
            self.file_list = file_list
        elif split is not None:
            # 从split.json加载
            split_path = split_file if split_file else self.root_dir / 'split.json'
            self.file_list = self._load_split(split_path, split)
        else:
            # 自动扫描所有文件
            self.file_list = self._scan_files()
        
        print(f"\n[NPYDataset] Loaded dataset")
        print(f"  Image dir: {self.image_dir}")
        print(f"  Mask dir: {self.mask_dir}")
        print(f"  Split: {split if split else 'all'}")
        print(f"  Samples: {len(self.file_list)}")
        print(f"  Image size: {image_size}x{image_size}")
        print(f"  Num classes: {num_classes}")
        print(f"  Return instance: {return_instance}")
    
    def _load_split(self, split_file: Path, split: str) -> List[str]:
        """从JSON文件加载数据划分"""
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            splits = json.load(f)
        
        if split not in splits:
            raise ValueError(f"Split '{split}' not found in {split_file}")
        
        return splits[split]
    
    def _scan_files(self) -> List[str]:
        """自动扫描所有.npy文件"""
        npy_files = list(self.image_dir.glob('*.npy'))
        # 去掉扩展名
        file_list = [f.stem for f in npy_files]
        file_list.sort()
        return file_list
    
    def __len__(self):
        return len(self.file_list)
    
    def _load_image(self, file_id: str) -> np.ndarray:
        """加载图像npy文件"""
        image_path = self.image_dir / f"{file_id}.npy"
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = np.load(image_path)
        
        # 确保是uint8类型
        if image.dtype != np.uint8:
            # 如果是float且在[0,1]范围，转换为[0,255]
            if image.dtype in [np.float32, np.float64]:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # 处理不同的shape
        if len(image.shape) == 2:
            # (H, W) - 已经是灰度图
            pass
        elif len(image.shape) == 3:
            if self.grayscale:
                # (H, W, C) -> (H, W) 转为灰度
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                elif image.shape[2] == 1:
                    image = image[:, :, 0]
            else:
                # 保持彩色
                pass
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        return image
    
    def _load_mask(self, file_id: str) -> np.ndarray:
        """加载mask npy文件"""
        mask_path = self.mask_dir / f"{file_id}.npy"
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        mask = np.load(mask_path)
        
        # 确保是2D
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # 处理mask值
        if self.mask_values is not None:
            # 只保留指定的类别值
            new_mask = np.zeros_like(mask)
            for new_id, old_id in enumerate(self.mask_values):
                new_mask[mask == old_id] = new_id
            mask = new_mask
        
        if self.binary_mask:
            # 转为二值：0=背景，1=前景
            mask = (mask > 0).astype(np.uint8)
        
        return mask
    
    def _extract_instances(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        从semantic mask提取instances
        
        Returns:
            masks: (N, H, W) bool array
            labels: (N,) int array
        """
        # 对每个类别分别提取连通区域
        unique_classes = np.unique(mask)
        unique_classes = unique_classes[unique_classes > 0]  # 排除背景
        
        instance_masks = []
        instance_labels = []
        
        for class_id in unique_classes:
            # 提取该类别的mask
            class_mask = (mask == class_id).astype(np.uint8)
            
            # 连通组件分析
            num_labels, labels_im = cv2.connectedComponents(class_mask)
            
            # 跳过背景（label=0）
            for i in range(1, num_labels):
                instance_mask = (labels_im == i)
                
                # 过滤太小的区域
                if instance_mask.sum() < 50:
                    continue
                
                instance_masks.append(instance_mask)
                instance_labels.append(class_id)
                
                if len(instance_masks) >= self.max_instances:
                    break
            
            if len(instance_masks) >= self.max_instances:
                break
        
        # 如果没有实例，返回空mask
        if len(instance_masks) == 0:
            h, w = mask.shape
            instance_masks = [np.zeros((h, w), dtype=bool)]
            instance_labels = [0]
        
        masks = np.stack(instance_masks, axis=0)  # (N, H, W)
        labels = np.array(instance_labels, dtype=np.int64)  # (N,)
        
        return masks, labels
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        获取一个样本
        
        Returns:
            如果return_instance=False:
                {
                    'image': (C, H, W) tensor,
                    'mask': (H, W) tensor,
                    'path': str (如果return_path=True)
                }
            如果return_instance=True:
                {
                    'image': (C, H, W) tensor,
                    'masks': (N, H, W) bool tensor,
                    'labels': (N,) long tensor,
                    'path': str (如果return_path=True)
                }
        """
        file_id = self.file_list[idx]
        
        # 加载数据
        image = self._load_image(file_id)
        mask = self._load_mask(file_id)
        
        # 数据增强
        if self.transforms is not None:
            if len(image.shape) == 2:
                # 灰度图需要添加一个维度用于albumentations
                transformed = self.transforms(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            else:
                # 彩色图
                transformed = self.transforms(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
        else:
            # 默认处理
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size), 
                            interpolation=cv2.INTER_NEAREST)
            
            # 归一化
            if self.normalize:
                image = image.astype(np.float32) / 255.0
            
            # 转tensor
            if len(image.shape) == 2:
                image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
            else:
                image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
            
            mask = torch.from_numpy(mask).long()
        
        # 返回结果
        result = {'image': image}
        
        if self.return_instance:
            # Instance格式
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            
            inst_masks, inst_labels = self._extract_instances(mask)
            result['masks'] = torch.from_numpy(inst_masks)
            result['labels'] = torch.from_numpy(inst_labels)
        else:
            # Semantic格式
            result['mask'] = mask
        
        if self.return_path:
            result['path'] = file_id
        
        return result
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        print("Calculating dataset statistics...")
        
        class_counts = {}
        total_pixels = 0
        foreground_pixels = 0
        
        for i in range(min(len(self), 100)):  # 只统计前100个样本
            file_id = self.file_list[i]
            mask = self._load_mask(file_id)
            
            unique, counts = np.unique(mask, return_counts=True)
            for cls, cnt in zip(unique, counts):
                class_counts[int(cls)] = class_counts.get(int(cls), 0) + cnt
            
            total_pixels += mask.size
            foreground_pixels += (mask > 0).sum()
        
        stats = {
            'total_samples': len(self),
            'sampled_for_stats': min(len(self), 100),
            'class_distribution': class_counts,
            'foreground_ratio': foreground_pixels / total_pixels if total_pixels > 0 else 0,
        }
        
        return stats


def get_default_transforms(image_size: int = 256, mode: str = 'train') -> A.Compose:
    """
    获取默认的数据增强
    
    Args:
        image_size: 目标大小
        mode: 'train' or 'val'
    """
    if mode == 'train':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.7
            ),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.3
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
    else:  # val/test
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])


def create_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 4,
    use_split_file: bool = True,
    val_split: float = 0.2,
    seed: int = 42,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证DataLoader
    
    Args:
        root_dir: 数据集根目录
        batch_size: batch大小
        image_size: 图像大小
        num_workers: 数据加载线程数
        use_split_file: 是否使用split.json文件
        val_split: 如果没有split.json，验证集比例
        seed: 随机种子
        **dataset_kwargs: 传递给NPYSegmentationDataset的其他参数
        
    Returns:
        train_loader, val_loader
    """
    if use_split_file:
        # 使用split.json
        train_dataset = NPYSegmentationDataset(
            root_dir=root_dir,
            split='train',
            image_size=image_size,
            transforms=get_default_transforms(image_size, 'train'),
            **dataset_kwargs
        )
        
        val_dataset = NPYSegmentationDataset(
            root_dir=root_dir,
            split='val',
            image_size=image_size,
            transforms=get_default_transforms(image_size, 'val'),
            **dataset_kwargs
        )
    else:
        # 自动划分
        full_dataset = NPYSegmentationDataset(
            root_dir=root_dir,
            image_size=image_size,
            **dataset_kwargs
        )
        
        total_size = len(full_dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        
        torch.manual_seed(seed)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # 设置不同的transforms
        train_dataset.dataset.transforms = get_default_transforms(image_size, 'train')
        val_dataset.dataset.transforms = get_default_transforms(image_size, 'val')
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    print(f"\n[DataLoaders Created]")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader


# Mask2Former专用collate函数
def collate_fn_instance(batch: List[Dict]) -> Dict:
    """
    Instance格式的collate函数（用于Mask2Former）
    """
    images = torch.stack([item['image'] for item in batch])
    masks = [item['masks'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    result = {
        'images': images,
        'masks': masks,
        'labels': labels,
    }
    
    if 'path' in batch[0]:
        result['paths'] = [item['path'] for item in batch]
    
    return result


def create_mask2former_dataloaders(
    root_dir: str,
    batch_size: int = 4,
    image_size: int = 256,
    num_workers: int = 4,
    use_split_file: bool = True,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    创建Mask2Former专用的DataLoader（instance格式）
    """
    if use_split_file:
        train_dataset = NPYSegmentationDataset(
            root_dir=root_dir,
            split='train',
            image_size=image_size,
            transforms=get_default_transforms(image_size, 'train'),
            return_instance=True,
            **dataset_kwargs
        )
        
        val_dataset = NPYSegmentationDataset(
            root_dir=root_dir,
            split='val',
            image_size=image_size,
            transforms=get_default_transforms(image_size, 'val'),
            return_instance=True,
            **dataset_kwargs
        )
    else:
        full_dataset = NPYSegmentationDataset(
            root_dir=root_dir,
            image_size=image_size,
            return_instance=True,
            **dataset_kwargs
        )
        
        val_size = int(len(full_dataset) * 0.2)
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        train_dataset.dataset.transforms = get_default_transforms(image_size, 'train')
        val_dataset.dataset.transforms = get_default_transforms(image_size, 'val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_instance,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_instance,
    )
    
    print(f"\n[Mask2Former DataLoaders Created]")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test NPY DataLoader')
    parser.add_argument('--root_dir', type=str, required=True,
                       help='Dataset root directory')
    parser.add_argument('--use_split', action='store_true',
                       help='Use split.json file')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("NPY DataLoader Testing")
    print("="*80)
    
    # 测试1: 基础Dataset
    print("\n[Test 1] Basic NPYDataset")
    print("-" * 40)
    
    dataset = NPYSegmentationDataset(
        root_dir=args.root_dir,
        split='train' if args.use_split else None,
        image_size=256,
        transforms=get_default_transforms(256, 'train'),
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 获取样本
    sample = dataset[0]
    print(f"\nSample:")
    print(f"  image shape: {sample['image'].shape}")
    print(f"  mask shape: {sample['mask'].shape}")
    print(f"  mask unique: {torch.unique(sample['mask']).tolist()}")
    
    # 测试2: DataLoader
    print("\n[Test 2] DataLoader")
    print("-" * 40)
    
    train_loader, val_loader = create_dataloaders(
        root_dir=args.root_dir,
        batch_size=4,
        image_size=256,
        use_split_file=args.use_split,
    )
    
    batch = next(iter(train_loader))
    print(f"Batch:")
    print(f"  images: {batch['image'].shape}")
    print(f"  masks: {batch['mask'].shape}")
    
    # 测试3: 统计信息
    print("\n[Test 3] Statistics")
    print("-" * 40)
    
    stats = dataset.get_statistics()
    print(f"Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Foreground ratio: {stats['foreground_ratio']:.4f}")
    print(f"  Class distribution: {stats['class_distribution']}")
    
    # 测试4: Instance格式
    print("\n[Test 4] Instance Format")
    print("-" * 40)
    
    inst_dataset = NPYSegmentationDataset(
        root_dir=args.root_dir,
        split='train' if args.use_split else None,
        image_size=256,
        return_instance=True,
    )
    
    sample = inst_dataset[0]
    print(f"Instance sample:")
    print(f"  image: {sample['image'].shape}")
    print(f"  masks: {sample['masks'].shape}")
    print(f"  labels: {sample['labels'].tolist()}")
    
    print("\n" + "="*80)
    print("All tests passed! ✓")
    print("="*80)