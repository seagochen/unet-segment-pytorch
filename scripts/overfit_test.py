#!/usr/bin/env python3
"""
过拟合测试脚本

在少量样本上进行过拟合测试，验证：
1. 数据加载是否正确
2. 模型是否能学习特征
3. 损失函数是否正常工作

如果模型无法在少量样本上过拟合，说明存在根本性问题。
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unet.models import UNet, AttentionUNet
from unet.data.dataset import LungTumorDataset
from unet.utils.loss import DiceLoss, DiceBCELoss, DeepSupervisionLoss


def visualize_samples(dataset, indices, save_path="overfit_samples.png"):
    """可视化选中的样本"""
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    if n == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        img, mask = dataset[idx]

        # Convert to numpy for visualization
        img_np = img.squeeze().numpy()
        mask_np = mask.numpy()

        # Image
        axes[i, 0].imshow(img_np, cmap='gray')
        axes[i, 0].set_title(f'Sample {idx}: Image')
        axes[i, 0].axis('off')

        # Mask
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title(f'Mask (tumor pixels: {mask_np.sum()})')
        axes[i, 1].axis('off')

        # Overlay
        axes[i, 2].imshow(img_np, cmap='gray')
        axes[i, 2].imshow(mask_np, cmap='Reds', alpha=0.5)
        axes[i, 2].set_title('Overlay')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"样本可视化已保存: {save_path}")


def overfit_test(
    data_root: str = "./liver_tumor_segmentation",
    num_samples: int = 4,
    num_epochs: int = 200,
    lr: float = 0.001,
    loss_type: str = "dice_bce",
    model_type: str = "unet",
    img_size: int = 256,
):
    """
    过拟合测试

    Args:
        data_root: 数据集路径
        num_samples: 使用的样本数量
        num_epochs: 训练轮数
        lr: 学习率
        loss_type: 损失函数类型 ('focal_tversky', 'dice', 'ce', 'balanced_focal_tversky')
        model_type: 模型类型 ('unet', 'attention_unet')
        img_size: 输入图像尺寸
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载数据集
    print(f"\n加载数据集: {data_root}")
    dataset = LungTumorDataset(
        root=data_root,
        split='train',
        transform=None,
        img_size=img_size
    )

    # 找出有肿瘤的样本
    print("查找含肿瘤的样本...")
    tumor_indices = []
    for i in range(len(dataset)):
        _, mask = dataset[i]
        tumor_pixels = mask.sum().item()
        if tumor_pixels > 100:  # 至少100个肿瘤像素
            tumor_indices.append((i, tumor_pixels))

    print(f"找到 {len(tumor_indices)} 个含肿瘤样本")

    if len(tumor_indices) < num_samples:
        print(f"警告: 含肿瘤样本不足 {num_samples} 个!")
        num_samples = len(tumor_indices)

    # 选择肿瘤像素最多的样本
    tumor_indices.sort(key=lambda x: x[1], reverse=True)
    selected_indices = [idx for idx, _ in tumor_indices[:num_samples]]

    print(f"\n选中的样本:")
    for idx, pixels in tumor_indices[:num_samples]:
        print(f"  样本 {idx}: {pixels} 肿瘤像素")

    # 可视化样本
    visualize_samples(dataset, selected_indices)

    # 创建小数据集
    small_dataset = Subset(dataset, selected_indices)
    dataloader = DataLoader(small_dataset, batch_size=num_samples, shuffle=True)

    # 创建模型
    use_deep_supervision = (model_type == 'attention_unet')
    if model_type == 'attention_unet':
        model = AttentionUNet(n_channels=1, n_classes=2, bilinear=True, deep_supervision=True).to(device)
        print(f"使用 AttentionUNet + Deep Supervision")
    else:
        model = UNet(n_channels=1, n_classes=2, bilinear=True).to(device)
        print(f"使用 UNet")

    # 创建损失函数
    if loss_type == 'dice_bce':
        base_criterion = DiceBCELoss()
    elif loss_type == 'dice':
        base_criterion = DiceLoss(ignore_background=True)
    else:
        base_criterion = torch.nn.CrossEntropyLoss()

    # 如果使用深监督，包装损失函数
    if use_deep_supervision:
        criterion = DeepSupervisionLoss(base_criterion, weights=[1.0, 0.4, 0.2, 0.1])
        print(f"损失函数: {loss_type} + DeepSupervision")
    else:
        criterion = base_criterion
        print(f"损失函数: {loss_type}")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练
    print(f"\n开始过拟合测试 ({num_epochs} epochs)...")
    print("=" * 60)

    losses = []
    dice_scores = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 计算 Dice
        model.eval()
        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                # eval mode returns single tensor even with deep supervision
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                probs = F.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)

                # 计算肿瘤 Dice
                pred_tumor = (preds == 1).float()
                true_tumor = (masks == 1).float()

                intersection = (pred_tumor * true_tumor).sum()
                union = pred_tumor.sum() + true_tumor.sum()

                if union > 0:
                    dice = (2 * intersection / union).item()
                else:
                    dice = 0.0

        losses.append(epoch_loss)
        dice_scores.append(dice)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Loss={epoch_loss:.4f}, Tumor Dice={dice:.4f}")

    print("=" * 60)

    # 最终结果
    print(f"\n最终结果:")
    print(f"  Loss: {losses[-1]:.4f}")
    print(f"  Tumor Dice: {dice_scores[-1]:.4f}")

    # 可视化预测结果
    print("\n生成预测可视化...")
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(dataloader))
        images = images.to(device)
        outputs = model(images)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        probs = F.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1).cpu()

    # 绘制结果
    n = min(4, num_samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))

    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        img_np = images[i].cpu().squeeze().numpy()
        mask_np = masks[i].numpy()
        pred_np = preds[i].numpy()
        prob_np = probs[i, 1].cpu().numpy()  # 肿瘤概率

        axes[i, 0].imshow(img_np, cmap='gray')
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title(f'Ground Truth ({mask_np.sum()} px)')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_np, cmap='gray')
        axes[i, 2].set_title(f'Prediction ({pred_np.sum()} px)')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(prob_np, cmap='hot', vmin=0, vmax=1)
        axes[i, 3].set_title('Tumor Probability')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig('overfit_predictions.png', dpi=150)
    plt.close()
    print("预测可视化已保存: overfit_predictions.png")

    # 绘制训练曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)

    ax2.plot(dice_scores)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Tumor Dice Score')
    ax2.grid(True)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('overfit_curves.png', dpi=150)
    plt.close()
    print("训练曲线已保存: overfit_curves.png")

    # 判断是否成功过拟合
    if dice_scores[-1] > 0.8:
        print("\n✓ 过拟合测试通过! 模型可以学习肿瘤特征。")
        return True
    else:
        print("\n✗ 过拟合测试失败! 模型无法学习肿瘤特征。")
        print("  可能的原因:")
        print("  1. 数据加载问题")
        print("  2. 损失函数问题")
        print("  3. 模型架构问题")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="过拟合测试")
    parser.add_argument("--data", type=str, default="./liver_tumor_segmentation",
                        help="数据集路径")
    parser.add_argument("--samples", type=int, default=4,
                        help="样本数量")
    parser.add_argument("--epochs", type=int, default=200,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率")
    parser.add_argument("--loss", type=str, default="dice_bce",
                        choices=["dice_bce", "dice", "ce"],
                        help="损失函数类型")
    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "attention_unet"],
                        help="模型类型")
    parser.add_argument("--img-size", type=int, default=256,
                        help="输入图像尺寸")

    args = parser.parse_args()

    overfit_test(
        data_root=args.data,
        num_samples=args.samples,
        num_epochs=args.epochs,
        lr=args.lr,
        loss_type=args.loss,
        model_type=args.model,
        img_size=args.img_size,
    )
