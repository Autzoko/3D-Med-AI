"""
训练前检查脚本
验证数据、模型、依赖是否准备就绪
"""

import sys
import torch
import numpy as np
from pathlib import Path


def check_dependencies():
    """检查依赖"""
    print("\n[1] 检查依赖...")
    
    required = {
        'torch': torch,
        'numpy': np,
        'scipy': None,
        'albumentations': None,
        'cv2': None,
        'tqdm': None,
    }
    
    missing = []
    
    for name, module in required.items():
        if module is None:
            try:
                __import__(name)
                print(f"  ✓ {name}")
            except ImportError:
                print(f"  ✗ {name} - MISSING")
                missing.append(name)
        else:
            print(f"  ✓ {name}")
    
    if missing:
        print(f"\n  ❌ 缺少依赖: {', '.join(missing)}")
        print(f"  安装命令: pip install {' '.join(missing)}")
        return False
    
    print("  ✓ 所有依赖已安装")
    return True


def check_gpu():
    """检查GPU"""
    print("\n[2] 检查GPU...")
    
    if not torch.cuda.is_available():
        print("  ⚠️  未检测到GPU，将使用CPU训练（会很慢）")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"  ✓ 检测到 {gpu_count} 个GPU")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    return True


def check_data(data_root):
    """检查数据"""
    print("\n[3] 检查数据...")
    
    data_root = Path(data_root)
    
    # 检查目录结构
    required_dirs = ['images', 'masks']
    required_files = ['split.json']
    
    for dir_name in required_dirs:
        dir_path = data_root / dir_name
        if not dir_path.exists():
            print(f"  ✗ 目录不存在: {dir_path}")
            return False
        
        npy_files = list(dir_path.glob('*.npy'))
        print(f"  ✓ {dir_name}/: {len(npy_files)} 个.npy文件")
    
    for file_name in required_files:
        file_path = data_root / file_name
        if not file_path.exists():
            print(f"  ✗ 文件不存在: {file_path}")
            return False
        print(f"  ✓ {file_name}")
    
    # 检查数据划分
    import json
    with open(data_root / 'split.json', 'r') as f:
        splits = json.load(f)
    
    print(f"  数据划分:")
    for split_name, split_list in splits.items():
        print(f"    {split_name}: {len(split_list)} 样本")
    
    # 检查单个样本
    sample_id = splits['train'][0]
    image_path = data_root / 'images' / f'{sample_id}.npy'
    mask_path = data_root / 'masks' / f'{sample_id}.npy'
    
    image = np.load(image_path)
    mask = np.load(mask_path)
    
    print(f"  样本检查 ({sample_id}):")
    print(f"    Image: shape={image.shape}, dtype={image.dtype}, range=[{image.min()}, {image.max()}]")
    print(f"    Mask: shape={mask.shape}, dtype={mask.dtype}, unique={np.unique(mask).tolist()}")
    
    # 验证
    if image.shape[:2] != mask.shape[:2]:
        print(f"  ✗ 图像和mask尺寸不匹配!")
        return False
    
    if image.dtype != np.uint8:
        print(f"  ⚠️  Image dtype不是uint8，可能有问题")
    
    if mask.dtype != np.uint8:
        print(f"  ⚠️  Mask dtype不是uint8，可能有问题")
    
    print("  ✓ 数据格式正确")
    return True


def check_model():
    """检查模型"""
    print("\n[4] 检查模型...")
    
    try:
        from model import segmamba_mask2former_small
        
        model = segmamba_mask2former_small(num_classes=1, num_queries=20, in_chans=1)
        
        # 测试前向传播
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            output = model(x)
        
        print(f"  ✓ 模型加载成功")
        print(f"  输出:")
        print(f"    pred_logits: {output['pred_logits'].shape}")
        print(f"    pred_masks: {output['pred_masks'].shape}")
        print(f"    aux_outputs: {len(output['aux_outputs'])} 层")
        
        # 参数量
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {num_params / 1e6:.2f}M")
        
        return True
    
    except Exception as e:
        print(f"  ✗ 模型检查失败: {e}")
        return False


def check_dataloader(data_root):
    """检查DataLoader"""
    print("\n[5] 检查DataLoader...")
    
    try:
        from universal_dataloader import NPYSegmentationDataset, get_default_transforms
        from torch.utils.data import DataLoader
        
        dataset = NPYSegmentationDataset(
            root_dir=data_root,
            split='train',
            image_size=256,
            transforms=get_default_transforms(256, 'train'),
            num_classes=1,
        )
        
        print(f"  ✓ Dataset加载成功: {len(dataset)} 样本")
        
        # 获取一个样本
        sample = dataset[0]
        print(f"  样本:")
        print(f"    image: {sample['image'].shape}")
        print(f"    mask: {sample['mask'].shape}")
        
        # 测试DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        
        print(f"  Batch:")
        print(f"    image: {batch['image'].shape}")
        print(f"    mask: {batch['mask'].shape}")
        
        print("  ✓ DataLoader工作正常")
        return True
    
    except Exception as e:
        print(f"  ✗ DataLoader检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_training_script():
    """检查训练脚本"""
    print("\n[6] 检查训练脚本...")
    
    train_script = Path('train.py')
    
    if not train_script.exists():
        print(f"  ✗ train.py 不存在")
        return False
    
    print(f"  ✓ train.py 存在")
    
    # 尝试导入关键组件
    try:
        from train import HungarianMatcher, Mask2FormerLoss
        print(f"  ✓ HungarianMatcher 可导入")
        print(f"  ✓ Mask2FormerLoss 可导入")
        return True
    except Exception as e:
        print(f"  ✗ 导入失败: {e}")
        return False


def estimate_memory(batch_size=4, image_size=256):
    """估算显存需求"""
    print(f"\n[7] 估算显存需求...")
    
    # 粗略估算
    # Model: ~18M params * 4 bytes = 72 MB
    # Activations: batch_size * image_size^2 * channels * layers * 4 bytes
    
    model_memory = 18 * 4  # MB
    activation_memory = batch_size * image_size * image_size * 256 * 10 * 4 / (1024**2)  # MB
    
    total_memory = model_memory + activation_memory
    
    print(f"  模型参数: ~{model_memory:.0f} MB")
    print(f"  激活值(batch={batch_size}): ~{activation_memory:.0f} MB")
    print(f"  总计(估算): ~{total_memory:.0f} MB ({total_memory/1024:.1f} GB)")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU显存: {gpu_memory:.1f} GB")
        
        if total_memory / 1024 < gpu_memory * 0.8:
            print(f"  ✓ 显存充足")
            return True
        else:
            print(f"  ⚠️  显存可能不足，建议:")
            print(f"    - 减小batch_size")
            print(f"    - 减小image_size")
            print(f"    - 使用model_size=tiny")
            return False
    else:
        print(f"  ⚠️  未检测到GPU")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='训练前检查')
    parser.add_argument('--data_root', type=str, required=True,
                       help='NPY数据集根目录')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='计划使用的batch size')
    
    args = parser.parse_args()
    
    print("="*80)
    print("训练前检查")
    print("="*80)
    
    checks = []
    
    # 1. 依赖
    checks.append(('依赖', check_dependencies()))
    
    # 2. GPU
    checks.append(('GPU', check_gpu()))
    
    # 3. 数据
    checks.append(('数据', check_data(args.data_root)))
    
    # 4. 模型
    checks.append(('模型', check_model()))
    
    # 5. DataLoader
    checks.append(('DataLoader', check_dataloader(args.data_root)))
    
    # 6. 训练脚本
    checks.append(('训练脚本', check_training_script()))
    
    # 7. 显存
    checks.append(('显存', estimate_memory(args.batch_size)))
    
    # 总结
    print("\n" + "="*80)
    print("检查总结")
    print("="*80)
    
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
    
    all_passed = all(passed for _, passed in checks)
    
    if all_passed:
        print("\n✅ 所有检查通过！可以开始训练了！")
        print("\n运行训练:")
        print(f"  python train.py --data_root {args.data_root} --output_dir ./outputs/experiment")
    else:
        print("\n❌ 部分检查未通过，请先解决问题")
        sys.exit(1)
    
    print("="*80)


if __name__ == "__main__":
    main()