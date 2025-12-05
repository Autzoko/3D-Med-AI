import os
import glob
import random

import nibabel as nib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function

from model_segmamba.segmamba_adda import SegMamba


# ======================= 基本配置 =======================
device = "cuda:0" if torch.cuda.is_available() else "cpu"

max_epoch   = 50
batch_size  = 2
num_workers = 4

busbra_root = "/vast/xl5874/SegMamba/data/BUSBRA_fullres/train"
busi_root   = "/vast/xl5874/SegMamba/data/BUSI_fullres/train"

ckpt_path   = "/vast/xl5874/SegMamba/logs/segmamba_busbra/model/best_0.8424.pt"

save_dir    = "./logs_disentangle_align"
os.makedirs(save_dir, exist_ok=True)

lambda_rec   = 1.0
lambda_dom_c = 0.1
lambda_dom_s = 0.1


# ======================= 工具函数 =======================
def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_volume(vol):
    vol = vol.astype(np.float32)
    m = np.mean(vol)
    s = np.std(vol)
    if s < 1e-6:
        return vol - m
    return (vol - m) / s


def dice_loss_with_logits(logits, targets, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    if probs.shape[1] < 2:
        pred = probs[:, 0:1, ...]
        targets_fg = (targets > 0.5).float()
        return F.mse_loss(pred, targets_fg)

    fg = probs[:, 1:2, ...]
    targets_fg = (targets > 0.5).float()

    inter = torch.sum(fg * targets_fg)
    union = torch.sum(fg) + torch.sum(targets_fg) + eps
    dice = (2.0 * inter + eps) / union
    return 1.0 - dice


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


# ======================= 数据集 =======================
def collect_pairs(root):
    data_dir   = os.path.join(root, "data")
    labels_dir = os.path.join(root, "labels")

    img_paths_all = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
    lbl_paths_all = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))

    lbl_dict = {}
    for p in lbl_paths_all:
        name = os.path.basename(p)
        stem = name.replace(".nii.gz", "")
        lbl_dict[stem] = p

    img_list, lbl_list = [], []
    for img_p in img_paths_all:
        name = os.path.basename(img_p)
        stem = name.replace(".nii.gz", "")
        if "_" in stem:
            base = stem.rsplit("_", 1)[0]
        else:
            base = stem
        if base in lbl_dict:
            img_list.append(img_p)
            lbl_list.append(lbl_dict[base])

    return img_list, lbl_list


class NiftiDataset3D(Dataset):
    def __init__(self, root, with_label=True):
        self.root = root
        self.with_label = with_label

        if with_label:
            self.img_paths, self.lbl_paths = collect_pairs(root)
            print("[NiftiDataset3D] root =", root,
                  "with_label=True, n =", len(self.img_paths))
        else:
            data_dir = os.path.join(root, "data")
            self.img_paths = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
            print("[NiftiDataset3D] root =", root,
                  "with_label=False, n =", len(self.img_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_nii = nib.load(img_path)
        img = img_nii.get_fdata()
        img = normalize_volume(img)
        img = img[None, ...]

        if self.with_label:
            lbl_path = self.lbl_paths[idx]
            lbl_nii = nib.load(lbl_path)
            lbl = lbl_nii.get_fdata()
            lbl = (lbl > 0).astype(np.float32)
            lbl = lbl[None, ...]
            return torch.from_numpy(img), torch.from_numpy(lbl), os.path.basename(img_path)
        else:
            return torch.from_numpy(img), os.path.basename(img_path)


# ======================= Disentangle 模块 =======================
class StyleEncoder(nn.Module):
    def __init__(self, in_ch=1, feat_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 16, 3, padding=1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(32, feat_ch, 3, stride=2, padding=1),
            nn.InstanceNorm3d(feat_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ContentEncoderFromLogits(nn.Module):
    def __init__(self, in_ch=2, feat_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 64, 1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, feat_ch, 3, padding=1),
            nn.InstanceNorm3d(feat_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ReconDecoder(nn.Module):
    """
    不改变空间尺寸，只做卷积重建，保证输出和输入 (128,128,16) 一致。
    """
    def __init__(self, in_ch, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 64, 3, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 32, 3, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(32, 16, 3, padding=1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(16, out_ch, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class DomainClassifier3D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(16, 2)

    def forward(self, feat):
        x = self.conv(feat)
        x = torch.mean(x, dim=[2, 3, 4])
        logit = self.fc(x)
        return logit


class DisentangleAlignNet(nn.Module):
    def __init__(self, segmamba, feat_c=64, feat_s=64):
        super().__init__()
        self.backbone = segmamba

        dev = next(self.backbone.parameters()).device
        dummy = torch.randn(1, 1, 128, 128, 16, device=dev)
        with torch.no_grad():
            dummy_out = self.backbone(dummy)
        out_ch = dummy_out.shape[1]
        print("[DisentangleAlignNet] SegMamba 输出通道数 =", out_ch)

        self.enc_c = ContentEncoderFromLogits(in_ch=out_ch, feat_ch=feat_c)
        self.enc_s = StyleEncoder(in_ch=1, feat_ch=feat_s)
        self.recon = ReconDecoder(in_ch=feat_c + feat_s, out_ch=1)

        self.dom_c = DomainClassifier3D(in_ch=feat_c)
        self.dom_s = DomainClassifier3D(in_ch=feat_s)

    def forward(self, img, lambd=0.0):
        logits = self.backbone(img)

        c_feat = self.enc_c(logits)
        s_feat = self.enc_s(img)

        # 统一空间尺寸
        if s_feat.shape[2:] != c_feat.shape[2:]:
            s_feat = F.interpolate(
                s_feat,
                size=c_feat.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        rec_in = torch.cat([c_feat, s_feat], dim=1)
        rec = self.recon(rec_in)

        c_rev = grad_reverse(c_feat, lambd)
        dom_c_logit = self.dom_c(c_rev)
        dom_s_logit = self.dom_s(s_feat)

        return logits, c_feat, s_feat, rec, dom_c_logit, dom_s_logit


# ======================= SegMamba 初始化 =======================
def build_segmamba_from_ckpt():
    print("[build_segmamba_from_ckpt] ckpt_path =", ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt 不存在: {ckpt_path}")

    model = SegMamba()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "net" in ckpt:
        state = ckpt["net"]
    else:
        state = ckpt

    model_dict = model.state_dict()
    new_state = model_dict.copy()

    matched, skipped = 0, 0
    for k, v in state.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            new_state[k] = v
            matched += 1
        else:
            skipped += 1
    model.load_state_dict(new_state)
    print(f"[build_segmamba_from_ckpt] matched {matched} keys, skipped {skipped} keys")

    return model


# ======================= 训练循环 =======================
def main():
    set_seed(2025)

    src_ds = NiftiDataset3D(busbra_root, with_label=True)
    tgt_ds = NiftiDataset3D(busi_root,   with_label=False)

    src_loader = DataLoader(
        src_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    tgt_loader = DataLoader(
        tgt_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    seg_backbone = build_segmamba_from_ckpt()
    seg_backbone.to(device)

    net = DisentangleAlignNet(seg_backbone, feat_c=64, feat_s=64)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)

    ce_loss = nn.CrossEntropyLoss()
    l1_loss = nn.L1Loss()

    best_dice = 0.0

    for epoch in range(max_epoch):
        net.train()
        running_seg = running_rec = running_dom_c = running_dom_s = 0.0

        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)
        num_steps = min(len(src_loader), len(tgt_loader))

        for step in range(num_steps):
            try:
                img_s, lbl_s, _ = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                img_s, lbl_s, _ = next(src_iter)

            try:
                img_t, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                img_t, _ = next(tgt_iter)

            img_s = img_s.to(device, non_blocking=True)
            lbl_s = lbl_s.to(device, non_blocking=True)
            img_t = img_t.to(device, non_blocking=True)

            lambd = float(epoch) / float(max_epoch)

            logits_s, c_s, s_s, rec_s, dom_c_s, dom_s_s = net(img_s, lambd=lambd)
            logits_t, c_t, s_t, rec_t, dom_c_t, dom_s_t = net(img_t, lambd=lambd)

            seg_ce   = ce_loss(logits_s, lbl_s.squeeze(1).long())
            seg_dice = dice_loss_with_logits(logits_s, lbl_s)
            loss_seg = seg_ce + seg_dice

            loss_rec_s = l1_loss(rec_s, img_s)
            loss_rec_t = l1_loss(rec_t, img_t)
            loss_rec   = loss_rec_s + loss_rec_t

            dom_label_s = torch.zeros(dom_c_s.size(0), dtype=torch.long, device=device)
            dom_label_t = torch.ones(dom_c_t.size(0), dtype=torch.long, device=device)

            dom_c_all = torch.cat([dom_c_s, dom_c_t], dim=0)
            dom_c_lbl = torch.cat([dom_label_s, dom_label_t], dim=0)
            loss_dom_c = ce_loss(dom_c_all, dom_c_lbl)

            dom_s_all = torch.cat([dom_s_s, dom_s_t], dim=0)
            dom_s_lbl = torch.cat([dom_label_s, dom_label_t], dim=0)
            loss_dom_s = ce_loss(dom_s_all, dom_s_lbl)

            loss = loss_seg + lambda_rec * loss_rec + lambda_dom_c * loss_dom_c + lambda_dom_s * loss_dom_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_seg   += loss_seg.item()
            running_rec   += loss_rec.item()
            running_dom_c += loss_dom_c.item()
            running_dom_s += loss_dom_s.item()

            if (step + 1) % 20 == 0:
                print(f"[Ep {epoch:03d} | Step {step+1:04d}/{num_steps:04d}] "
                      f"Seg={running_seg/(step+1):.4f}  "
                      f"Rec={running_rec/(step+1):.4f}  "
                      f"DomC={running_dom_c/(step+1):.4f}  "
                      f"DomS={running_dom_s/(step+1):.4f}")

        # ====== epoch 结束，在 BUSBRA 上估 Dice ======
        net.eval()
        dices = []
        with torch.no_grad():
            for img_s, lbl_s, _ in src_loader:
                img_s = img_s.to(device, non_blocking=True)
                lbl_s = lbl_s.to(device, non_blocking=True)
                logits_s, _, _, _, _, _ = net(img_s, lambd=0.0)
                probs = F.softmax(logits_s, dim=1)
                if probs.shape[1] >= 2:
                    fg = probs[:, 1:2, ...]
                else:
                    fg = probs[:, 0:1, ...]
                pred = (fg > 0.5).float()
                gt = (lbl_s > 0.5).float()
                inter = torch.sum(pred * gt).item()
                union = torch.sum(pred).item() + torch.sum(gt).item() + 1e-6
                d = (2.0 * inter + 1e-6) / union
                dices.append(d)

        mean_dice = float(np.mean(dices))
        print(f"[Ep {epoch:03d}] BUSBRA mean Dice = {mean_dice:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            save_path = os.path.join(save_dir, "disentangle_align_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "best_dice": best_dice,
                },
                save_path,
            )
            print("  >> 更新 best dice, 保存到", save_path)


if __name__ == "__main__":
    main()
