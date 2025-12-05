"""
训练前检查脚本
验证所有组件是否正常工作

检查项目:
1. 依赖安装
2. GPU可用性
3. 数据格式
4. 模型加载
5. DataLoader
6. 前向传播
7. 显存估算
"""

import sys
import torch
import numpy as np
from pathlib import Path
import json


def check_dependencies():
    """检查依赖"""
    print("\n[1] 检查依赖...")
    print("-" * 40)
    
    required = {
        'torch': None,
        'numpy': None,
        'scipy': None,
        'albumentations': None,
        'cv2': None,
        'tqdm': None,
        'mamba_ssm': None,  # SegMamba需要
    }
    
    missing = []
    versions = {}
    
    for name in required.keys():
        try:
            if name == 'cv2':
                import cv2
                versions[name] = cv2.__version__
            else:
                module = __import__(name)
                versions[name] = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {name:20s} {versions[name]}")
        except ImportError:
            print(f"  ✗ {name:20s} - MISSING")
            missing.append(name)
    
    if missing:
        print(f"\n  ❌ 缺少依赖: {', '.join(missing)}")
        print(f"\n  安装命令:")
        if 'mamba_ssm' in missing:
            print(f"    pip install mamba-ssm")
        if any(x in missing for x in ['scipy', 'albumentations', 'cv2', 'tqdm']):
            deps = [x for x in missing if x not in ['mamba_ssm']]
            if 'cv2' in deps:
                deps.remove('cv2')
                deps.append('opencv-python')
            print(f"    pip install {' '.join(deps)}")
        return False
    
    print("  ✓ 所有依赖已安装")
    return True


def check_gpu():
    """检查GPU"""
    print("\n[2] 检查GPU...")
    print("-" * 40)
    
    if not torch.cuda.is_available():
        print("  ⚠️  未检测到CUDA，将使用CPU训练（会很慢）")
        print("  建议: 使用GPU进行训练")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"  ✓ 检测到 {gpu_count} 个GPU")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"    GPU {i}: {gpu_name}")
        print(f"           显存: {gpu_memory:.1f} GB")
    
    # 测试CUDA
    try:
        x = torch.randn(100, 100).cuda()
        y = x @ x
        print(f"  ✓ CUDA功能正常")
        return True
    except Exception as e:
        print(f"  ✗ CUDA测试失败: {e}")
        return False


def check_data(data_root):
    """检查数据"""
    print("\n[3] 检查数据...")
    print("-" * 40)
    
    data_root = Path(data_root)
    
    if not data_root.exists():
        print(f"  ✗ 数据目录不存在: {data_root}")
        return False
    
    # 检查目录结构
    required_dirs = ['images', 'masks']
    for dir_name in required_dirs:
        dir_path = data_root / dir_name
        if not dir_path.exists():
            print(f"  ✗ 目录不存在: {dir_path}")
            print(f"\n  请运行预处理脚本:")
            print(f"    python preprocess_busi.py --source_dir <原始BUSI> --output_dir {data_root}")
            return False
        
        npy_files = list(dir_path.glob('*.npy'))
        print(f"  ✓ {dir_name}/: {len(npy_files)} 个.npy文件")
        
        if len(npy_files) == 0:
            print(f"  ✗ {dir_name}/ 目录为空")
            return False
    
    # 检查split.json
    split_file = data_root / 'split.json'
    if not split_file.exists():
        print(f"  ✗ 文件不存在: split.json")
        print(f"\n  请运行预处理脚本生成split.json")
        return False
    
    with open(split_file, 'r') as f:
        splits = json.load(f)
    
    print(f"  ✓ split.json")
    print(f"    数据划分:")
    for split_name, split_list in splits.items():
        print(f"      {split_name}: {len(split_list)} 样本")
    
    # 检查单个样本
    if 'train' in splits and len(splits['train']) > 0:
        sample_id = splits['train'][0]
        image_path = data_root / 'images' / f'{sample_id}.npy'
        mask_path = data_root / 'masks' / f'{sample_id}.npy'
        
        try:
            image = np.load(image_path)
            mask = np.load(mask_path)
            
            print(f"\n  样本检查 ({sample_id}):")
            print(f"    Image: shape={image.shape}, dtype={image.dtype}, range=[{image.min()}, {image.max()}]")
            print(f"    Mask: shape={mask.shape}, dtype={mask.dtype}, unique={np.unique(mask).tolist()}")
            
            # 验证
            if image.shape[:2] != mask.shape[:2]:
                print(f"  ✗ 图像和mask尺寸不匹配!")
                return False
            
            if image.dtype != np.uint8:
                print(f"  ⚠️  Image dtype={image.dtype}，建议为uint8")
            
            if mask.dtype != np.uint8:
                print(f"  ⚠️  Mask dtype={mask.dtype}，建议为uint8")
            
            print("  ✓ 数据格式正确")
            return True
            
        except Exception as e:
            print(f"  ✗ 加载样本失败: {e}")
            return False
    else:
        print(f"  ⚠️  无法找到训练样本进行检查")
        return True


def check_model_files():
    """检查模型文件"""
    print("\n[4] 检查模型文件...")
    print("-" * 40)
    
    required_files = {
        'model.py': 'SegMambaMask2Former完整模型',
        'segmamba_backbone_2d.py': 'SegMamba backbone',
        'pixel_decoder.py': 'Pixel decoder (FPN)',
        'mask2former_decoder.py': 'Mask2Former transformer decoder',
        'dataloader.py': 'NPY数据加载器',
    }
    
    missing = []
    for file_name, description in required_files.items():
        if Path(file_name).exists():
            print(f"  ✓ {file_name:30s} ({description})")
        else:
            print(f"  ✗ {file_name:30s} - 缺失")
            missing.append(file_name)
    
    if missing:
        print(f"\n  ❌ 缺少文件: {', '.join(missing)}")
        print(f"  请确保这些文件在当前目录或Python path中")
        return False
    
    print("  ✓ 所有模型文件存在")
    return True


def check_model_import():
    """检查模型导入"""
    print("\n[5] 检查模型导入...")
    print("-" * 40)
    
    try:
        from model.model import segmamba_mask2former_small
        print("  ✓ model.segmamba_mask2former_small")
    except ImportError as e:
        print(f"  ✗ 无法导入 segmamba_mask2former_small")
        print(f"    错误: {e}")
        return False
    
    try:
        from data.dataloader import NPYSegmentationDataset, get_default_transforms
        print("  ✓ dataloader.NPYSegmentationDataset")
        print("  ✓ dataloader.get_default_transforms")
    except ImportError as e:
        print(f"  ✗ 无法导入 dataloader")
        print(f"    错误: {e}")
        return False
    
    print("  ✓ 所有模块导入成功")
    return True


def check_model_creation():
    """检查模型创建"""
    print("\n[6] 检查模型创建...")
    print("-" * 40)
    
    try:
        from model.model import segmamba_mask2former_small
        
        model = segmamba_mask2former_small(num_classes=1, num_queries=20, in_chans=1)
        
        print(f"  ✓ 模型创建成功")
        
        # 参数量
        num_params = sum(p.numel() for p in model.parameters())
        print(f"    总参数: {num_params / 1e6:.2f}M")
        
        # 测试前向传播
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            output = model(x)
        
        print(f"  ✓ 前向传播成功")
        print(f"    pred_logits: {output['pred_logits'].shape}")
        print(f"    pred_masks: {output['pred_masks'].shape}")
        print(f"    aux_outputs: {len(output['aux_outputs'])} 层")
        
        # 检查输出格式
        B, Q, C = output['pred_logits'].shape
        _, _, H, W = output['pred_masks'].shape
        print(f"    验证: Q={Q}, C={C} (应该=num_classes+1={1+1})")
        
        if C != 2:  # num_classes=1, 所以C应该=2 (1类+背景)
            print(f"  ⚠️  pred_logits的类别数不对: 期望2，实际{C}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 模型创建/前向传播失败")
        print(f"    错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dataloader(data_root):
    """检查DataLoader"""
    print("\n[7] 检查DataLoader...")
    print("-" * 40)
    
    try:
        from data.dataloader import NPYSegmentationDataset, get_default_transforms
        from torch.utils.data import DataLoader
        
        # 创建dataset
        dataset = NPYSegmentationDataset(
            root_dir=data_root,
            split='train',
            image_size=256,
            transforms=get_default_transforms(256, 'train'),
            num_classes=1,
        )
        
        print(f"  ✓ Dataset创建成功")
        print(f"    样本数: {len(dataset)}")
        
        # 获取一个样本
        sample = dataset[0]
        print(f"\n  样本:")
        print(f"    image: {sample['image'].shape}, dtype={sample['image'].dtype}")
        print(f"    mask: {sample['mask'].shape}, dtype={sample['mask'].dtype}")
        print(f"    mask unique: {torch.unique(sample['mask']).tolist()}")
        
        # 测试DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        
        print(f"\n  Batch (batch_size=2):")
        print(f"    image: {batch['image'].shape}")
        print(f"    mask: {batch['mask'].shape}")
        
        print("  ✓ DataLoader工作正常")
        return True
        
    except Exception as e:
        print(f"  ✗ DataLoader测试失败")
        print(f"    错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_training_components():
    """检查训练组件"""
    print("\n[8] 检查训练组件...")
    print("-" * 40)
    
    train_script = Path('train.py')
    
    if not train_script.exists():
        print(f"  ✗ train.py 不存在")
        return False
    
    print(f"  ✓ train.py 存在")
    
    # 尝试导入关键组件
    try:
        import sys
        sys.path.insert(0, '.')
        from train import HungarianMatcher, SetCriterion, prepare_targets
        
        print(f"  ✓ HungarianMatcher 可导入")
        print(f"  ✓ SetCriterion 可导入")
        print(f"  ✓ prepare_targets 可导入")
        
        # 测试prepare_targets
        dummy_batch = {
            'mask': torch.randint(0, 2, (2, 64, 64))
        }
        targets = prepare_targets(dummy_batch)
        print(f"\n  测试 prepare_targets:")
        print(f"    输入: {dummy_batch['mask'].shape}")
        print(f"    输出: labels={len(targets['labels'])} 个tensor, masks={len(targets['masks'])} 个tensor")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 导入训练组件失败")
        print(f"    错误: {e}")
        return False


def estimate_memory(batch_size=4, image_size=256):
    """估算显存需求"""
    print(f"\n[9] 估算显存需求 (batch_size={batch_size}, image_size={image_size})...")
    print("-" * 40)
    
    try:
        from model.model import segmamba_mask2former_small
        
        model = segmamba_mask2former_small(num_classes=1, num_queries=20)
        
        # 模型参数
        model_params = sum(p.numel() for p in model.parameters())
        model_memory = model_params * 4 / (1024**2)  # MB (float32)
        
        # 粗略估算激活值
        # 一个batch的图像: batch_size * C * H * W * 4 bytes
        # backbone特征: 多个尺度
        # transformer: queries等
        activation_memory = batch_size * image_size * image_size * 256 * 4 / (1024**2)  # MB
        
        total_memory = (model_memory + activation_memory) / 1024  # GB
        
        print(f"  模型参数: ~{model_memory:.0f} MB")
        print(f"  激活值(估算): ~{activation_memory:.0f} MB")
        print(f"  总计(估算): ~{total_memory:.1f} GB")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU显存: {gpu_memory:.1f} GB")
            
            if total_memory < gpu_memory * 0.8:
                print(f"  ✓ 显存充足")
                return True
            else:
                print(f"  ⚠️  显存可能不足")
                print(f"  建议:")
                print(f"    - 减小 batch_size (当前{batch_size})")
                print(f"    - 减小 image_size (当前{image_size})")
                print(f"    - 使用 model_size=tiny")
                return False
        else:
            print(f"  ⚠️  未检测到GPU")
            return False
            
    except Exception as e:
        print(f"  ✗ 显存估算失败: {e}")
        return False


def check_full_pipeline(data_root):
    """完整流程测试"""
    print(f"\n[10] 完整流程测试...")
    print("-" * 40)
    
    try:
        from model.model import segmamba_mask2former_small
        from data.dataloader import NPYSegmentationDataset, get_default_transforms
        from torch.utils.data import DataLoader
        from train import SetCriterion, HungarianMatcher, prepare_targets
        
        # 1. 创建模型
        model = segmamba_mask2former_small(num_classes=1, num_queries=20)
        print("  ✓ 模型创建")
        
        # 2. 创建DataLoader
        dataset = NPYSegmentationDataset(
            root_dir=data_root,
            split='train',
            image_size=256,
            transforms=get_default_transforms(256, 'train'),
            num_classes=1,
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        print("  ✓ DataLoader创建")
        
        # 3. 获取一个batch
        batch = next(iter(loader))
        images = batch['image']
        masks = batch['mask']
        print(f"  ✓ 获取batch: images={images.shape}, masks={masks.shape}")
        
        # 4. 前向传播
        with torch.no_grad():
            outputs = model(images)
        print(f"  ✓ 前向传播: pred_logits={outputs['pred_logits'].shape}")
        
        # 5. 准备targets
        targets = prepare_targets({'mask': masks})
        print(f"  ✓ 准备targets: {len(targets['labels'])} samples")
        
        # 6. 创建criterion
        matcher = HungarianMatcher(
            cost_class=2.0,
            cost_mask=5.0,
            cost_dice=5.0,
            num_points=12544,
        )
        
        weight_dict = {
            "loss_ce": 2.0,
            "loss_mask": 5.0,
            "loss_dice": 5.0,
        }
        # Deep supervision
        for i in range(len(outputs['aux_outputs'])):
            weight_dict.update({
                f"loss_ce_{i}": 2.0,
                f"loss_mask_{i}": 5.0,
                f"loss_dice_{i}": 5.0,
            })
        
        criterion = SetCriterion(
            num_classes=1,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=0.1,
        )
        print("  ✓ Criterion创建")
        
        # 7. 计算loss
        with torch.no_grad():
            loss_dict = criterion(outputs, targets)
            total_loss = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )
        
        print(f"  ✓ Loss计算成功")
        print(f"    loss_ce: {loss_dict.get('loss_ce', 0):.4f}")
        print(f"    loss_mask: {loss_dict.get('loss_mask', 0):.4f}")
        print(f"    loss_dice: {loss_dict.get('loss_dice', 0):.4f}")
        print(f"    total: {total_loss:.4f}")
        
        print("\n  ✓ 完整流程测试通过！")
        return True
        
    except Exception as e:
        print(f"  ✗ 完整流程测试失败")
        print(f"    错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='训练前检查')
    parser.add_argument('--data_root', type=str, required=True,
                       help='NPY数据集根目录')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='计划使用的batch size')
    parser.add_argument('--image_size', type=int, default=256,
                       help='图像大小')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SegMamba + Mask2Former 训练前检查")
    print("="*80)
    
    checks = []
    
    # 1. 依赖
    checks.append(('依赖', check_dependencies()))
    
    # 2. GPU
    checks.append(('GPU', check_gpu()))
    
    # 3. 模型文件
    checks.append(('模型文件', check_model_files()))
    
    # 4. 模型导入
    checks.append(('模型导入', check_model_import()))
    
    # 5. 模型创建
    checks.append(('模型创建', check_model_creation()))
    
    # 6. 数据
    checks.append(('数据', check_data(args.data_root)))
    
    # 7. DataLoader
    checks.append(('DataLoader', check_dataloader(args.data_root)))
    
    # 8. 训练组件
    checks.append(('训练组件', check_training_components()))
    
    # 9. 显存
    checks.append(('显存估算', estimate_memory(args.batch_size, args.image_size)))
    
    # 10. 完整流程
    checks.append(('完整流程', check_full_pipeline(args.data_root)))
    
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
        print("\n开始训练:")
        print(f"  python train.py \\")
        print(f"      --data_root {args.data_root} \\")
        print(f"      --output_dir ./outputs/experiment \\")
        print(f"      --model_size small \\")
        print(f"      --batch_size {args.batch_size} \\")
        print(f"      --epochs 50")
    else:
        print("\n❌ 部分检查未通过，请先解决问题")
        sys.exit(1)
    
    print("="*80)


if __name__ == "__main__":
    main()