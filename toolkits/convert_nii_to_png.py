#!/usr/bin/env python3
"""
NIfTI 转 PNG 格式转换脚本

将 Medical Decathlon Lung Tumor Segmentation 数据集从 NIfTI 格式转换为 PNG 切片
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

try:
    import nibabel as nib
except ImportError:
    print("错误: 请先安装 nibabel")
    print("  pip install nibabel")
    exit(1)

try:
    from PIL import Image
except ImportError:
    print("错误: 请先安装 Pillow")
    print("  pip install Pillow")
    exit(1)


def normalize_image(
    data: np.ndarray,
    window_center: int = -600,
    window_width: int = 1500,
    auto_detect: bool = True
) -> np.ndarray:
    """
    图像归一化到 [0, 255]

    支持两种模式：
    1. 自动检测：如果数据范围在 [0, 1]，直接缩放到 [0, 255]
    2. CT 窗宽窗位：对 HU 值进行窗宽窗位处理

    常用窗位参考：
    - 肺窗: window_center=-600, window_width=1500
    - 纵隔窗: window_center=40, window_width=400
    - 骨窗: window_center=400, window_width=1800

    Args:
        data: 图像数据
        window_center: 窗位 (Window Level)
        window_width: 窗宽 (Window Width)
        auto_detect: 是否自动检测数据范围

    Returns:
        归一化到 [0, 255] 的 uint8 数据
    """
    data_min, data_max = data.min(), data.max()

    # 自动检测：如果数据已经在 [0, 1] 范围内，直接缩放
    if auto_detect and data_min >= 0 and data_max <= 1.0:
        return (data * 255).astype(np.uint8)

    # 自动检测：如果数据已经在 [0, 255] 范围内，直接转换
    if auto_detect and data_min >= 0 and data_max <= 255:
        return data.astype(np.uint8)

    # CT 窗宽窗位处理
    min_val = window_center - window_width // 2
    max_val = window_center + window_width // 2
    data = np.clip(data, min_val, max_val)
    data = (data - min_val) / (max_val - min_val) * 255
    return data.astype(np.uint8)


def find_nii_pairs(src_path: Path) -> List[Tuple[Path, Path, str]]:
    """
    查找所有图像和标签的 NIfTI 文件对

    支持两种数据结构：
    1. Kaggle 版本: 0/data/0_data.nii, 0/label/0_mask.nii
    2. Medical Decathlon 标准版本: imagesTr/xxx.nii.gz, labelsTr/xxx.nii.gz

    Returns:
        List of (image_path, label_path, case_id) tuples
    """
    src_path = Path(src_path)
    pairs = []

    # 检查是否是 Kaggle 版本（数字目录结构）
    subdirs = [d for d in src_path.iterdir() if d.is_dir() and d.name.isdigit()]

    if subdirs:
        # Kaggle 版本: 0/data/xxx.nii, 0/label/xxx.nii
        print("检测到 Kaggle 版本数据结构")
        for subdir in sorted(subdirs, key=lambda x: int(x.name)):
            data_dir = subdir / "data"
            label_dir = subdir / "label"

            if not data_dir.exists() or not label_dir.exists():
                continue

            # 查找 NIfTI 文件
            data_files = list(data_dir.glob("*.nii*"))
            label_files = list(label_dir.glob("*.nii*"))

            if data_files and label_files:
                case_id = subdir.name
                pairs.append((data_files[0], label_files[0], case_id))

        return pairs

    # 检查是否是 Medical Decathlon 标准版本
    images_dir = src_path / "imagesTr"
    labels_dir = src_path / "labelsTr"

    # 也检查子目录
    if not images_dir.exists():
        for subdir in src_path.iterdir():
            if subdir.is_dir():
                if (subdir / "imagesTr").exists():
                    images_dir = subdir / "imagesTr"
                    labels_dir = subdir / "labelsTr"
                    break

    if images_dir.exists() and labels_dir.exists():
        print("检测到 Medical Decathlon 标准数据结构")
        for img_file in sorted(images_dir.glob("*.nii*")):
            # Medical Decathlon 命名规则: lung_xxx_0000.nii.gz -> lung_xxx.nii.gz
            label_name = img_file.name.replace("_0000", "")
            label_file = labels_dir / label_name

            if not label_file.exists():
                label_file = labels_dir / img_file.name

            if label_file.exists():
                case_id = img_file.stem.replace(".nii", "").replace("_0000", "")
                pairs.append((img_file, label_file, case_id))

        return pairs

    raise FileNotFoundError(
        f"无法识别数据结构\n"
        f"支持的格式:\n"
        f"  1. Kaggle: 0/data/xxx.nii, 0/label/xxx.nii\n"
        f"  2. Medical Decathlon: imagesTr/xxx.nii, labelsTr/xxx.nii\n"
        f"请检查: {src_path}"
    )


def convert_nii_to_png(
    src_path: Path,
    dst_path: Path,
    window_center: int = -600,
    window_width: int = 1500,
    empty_slice_ratio: float = 0.1,
    tumor_only: bool = False,
    min_tumor_pixels: int = 0,
    seed: int = 42
) -> dict:
    """
    将 NIfTI 数据转换为 PNG 切片

    Args:
        src_path: 原始数据目录
        dst_path: 输出目录
        window_center: CT 窗位
        window_width: CT 窗宽
        empty_slice_ratio: 保留的空白切片比例 (0-1)
        tumor_only: 如果为 True，只保存有肿瘤的切片
        min_tumor_pixels: 最小肿瘤像素数（过滤太小的标注）
        seed: 随机种子（用于选择空白切片）

    Returns:
        数据集统计信息
    """
    random.seed(seed)
    np.random.seed(seed)

    src_path = Path(src_path)
    dst_path = Path(dst_path)

    # 创建输出目录
    images_out = dst_path / "images"
    labels_out = dst_path / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # 查找所有图像-标签对
    nii_pairs = find_nii_pairs(src_path)
    print(f"找到 {len(nii_pairs)} 个 NIfTI 文件对")

    if len(nii_pairs) == 0:
        raise FileNotFoundError(f"在 {src_path} 中没有找到有效的 NIfTI 文件对")

    # 如果 tumor_only 为 True，强制设置 empty_slice_ratio 为 0
    if tumor_only:
        empty_slice_ratio = 0.0
        print("模式: 仅保存含肿瘤的切片")
    else:
        print(f"模式: 保留 {empty_slice_ratio*100:.0f}% 的空白切片")

    if min_tumor_pixels > 0:
        print(f"过滤: 肿瘤像素数 < {min_tumor_pixels} 的切片将被视为空白")

    # 统计信息
    stats = {
        "total_volumes": len(nii_pairs),
        "total_slices": 0,
        "saved_slices": 0,
        "slices_with_tumor": 0,
        "slices_filtered_small_tumor": 0,
        "empty_slices_saved": 0,
        "empty_slices_skipped": 0,
        "window_center": window_center,
        "window_width": window_width,
        "empty_slice_ratio": empty_slice_ratio,
        "tumor_only": tumor_only,
        "min_tumor_pixels": min_tumor_pixels,
        "files": []
    }

    # 处理每个 NIfTI 文件对
    for img_file, label_file, case_id in tqdm(nii_pairs, desc="转换中"):
        # 加载数据
        img_nii = nib.load(str(img_file))
        label_nii = nib.load(str(label_file))

        img_data = img_nii.get_fdata()
        label_data = label_nii.get_fdata()

        # 处理每个切片
        num_slices = img_data.shape[2]
        stats["total_slices"] += num_slices

        # 收集空白切片索引，后续随机选择保留一部分
        empty_slice_indices = []

        for slice_idx in range(num_slices):
            img_slice = img_data[:, :, slice_idx]
            label_slice = label_data[:, :, slice_idx]

            # 计算肿瘤像素数
            tumor_pixels = np.sum(label_slice > 0)
            has_valid_tumor = tumor_pixels >= min_tumor_pixels and tumor_pixels > 0

            if has_valid_tumor:
                # 有效肿瘤切片全部保留
                stats["slices_with_tumor"] += 1
            elif tumor_pixels > 0:
                # 有肿瘤但太小，被过滤
                stats["slices_filtered_small_tumor"] += 1
                empty_slice_indices.append(slice_idx)
                continue
            else:
                # 空白切片先收集，后续随机选择
                empty_slice_indices.append(slice_idx)
                continue

            # 保存有肿瘤的切片
            _save_slice(
                img_slice, label_slice,
                case_id, slice_idx,
                images_out, labels_out,
                window_center, window_width,
                stats
            )

        # 随机选择部分空白切片保留
        num_empty_to_keep = int(len(empty_slice_indices) * empty_slice_ratio)
        if num_empty_to_keep > 0:
            selected_empty = random.sample(empty_slice_indices, num_empty_to_keep)
            for slice_idx in selected_empty:
                img_slice = img_data[:, :, slice_idx]
                label_slice = label_data[:, :, slice_idx]
                _save_slice(
                    img_slice, label_slice,
                    case_id, slice_idx,
                    images_out, labels_out,
                    window_center, window_width,
                    stats
                )
                stats["empty_slices_saved"] += 1

        stats["empty_slices_skipped"] += len(empty_slice_indices) - num_empty_to_keep

    # 保存统计信息
    stats_summary = {k: v for k, v in stats.items() if k != "files"}
    stats_file = dst_path / "dataset_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_summary, f, indent=2, ensure_ascii=False)

    # 保存文件列表
    file_list = dst_path / "file_list.txt"
    with open(file_list, 'w') as f:
        f.write('\n'.join(stats["files"]))

    # 打印统计信息
    print("\n" + "=" * 50)
    print("转换完成!")
    print("=" * 50)
    print(f"总体积数:        {stats['total_volumes']}")
    print(f"总切片数:        {stats['total_slices']}")
    print(f"保存的切片数:    {stats['saved_slices']}")
    print(f"  - 含肿瘤切片:  {stats['slices_with_tumor']}")
    print(f"  - 空白切片:    {stats['empty_slices_saved']}")
    if stats['slices_filtered_small_tumor'] > 0:
        print(f"过滤的小肿瘤:   {stats['slices_filtered_small_tumor']}")
    print(f"跳过的空白切片:  {stats['empty_slices_skipped']}")
    print(f"输出目录:        {dst_path}")
    print("=" * 50)

    return stats


def _save_slice(
    img_slice: np.ndarray,
    label_slice: np.ndarray,
    case_id: str,
    slice_idx: int,
    images_out: Path,
    labels_out: Path,
    window_center: int,
    window_width: int,
    stats: dict
) -> None:
    """保存单个切片为 PNG"""
    # 归一化 CT 图像
    img_normalized = normalize_image(img_slice, window_center, window_width)

    # 标签二值化
    label_binary = ((label_slice > 0) * 255).astype(np.uint8)

    # 生成文件名
    slice_name = f"{case_id}_slice_{slice_idx:04d}.png"

    # 保存
    Image.fromarray(img_normalized).save(images_out / slice_name)
    Image.fromarray(label_binary).save(labels_out / slice_name)

    stats["saved_slices"] += 1
    stats["files"].append(slice_name)


def main():
    parser = argparse.ArgumentParser(
        description="将 NIfTI 格式数据转换为 PNG 切片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python convert_nii_to_png.py --input ./data/raw --output ./data/png

  # 只导出有肿瘤的切片（推荐用于训练）
  python convert_nii_to_png.py --input ./data/raw --output ./data/png --tumor-only

  # 过滤太小的肿瘤标注（至少100个像素）
  python convert_nii_to_png.py --input ./data/raw --output ./data/png \\
      --tumor-only --min-tumor-pixels 100

  # 自定义窗宽窗位（纵隔窗）
  python convert_nii_to_png.py --input ./data/raw --output ./data/png \\
      --window-center 40 --window-width 400

  # 调整空白切片保留比例
  python convert_nii_to_png.py --input ./data/raw --output ./data/png \\
      --empty-ratio 0.2

CT 窗宽窗位参考:
  肺窗:   --window-center -600 --window-width 1500 (默认)
  纵隔窗: --window-center 40   --window-width 400
  骨窗:   --window-center 400  --window-width 1800
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="原始 NIfTI 数据目录"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="PNG 输出目录"
    )
    parser.add_argument(
        "--window-center", "-wc",
        type=int,
        default=-600,
        help="CT 窗位 (默认: -600，肺窗)"
    )
    parser.add_argument(
        "--window-width", "-ww",
        type=int,
        default=1500,
        help="CT 窗宽 (默认: 1500，肺窗)"
    )
    parser.add_argument(
        "--empty-ratio", "-e",
        type=float,
        default=0.1,
        help="保留的空白切片比例 (默认: 0.1，即保留 10%%)"
    )
    parser.add_argument(
        "--tumor-only", "-t",
        action="store_true",
        help="只保存含肿瘤的切片 (等同于 --empty-ratio 0)"
    )
    parser.add_argument(
        "--min-tumor-pixels", "-m",
        type=int,
        default=0,
        help="最小肿瘤像素数，小于此值的切片被视为空白 (默认: 0，不过滤)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )

    args = parser.parse_args()

    # 验证参数
    if not 0 <= args.empty_ratio <= 1:
        print("错误: --empty-ratio 必须在 0 到 1 之间")
        exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入目录不存在: {input_path}")
        exit(1)

    output_path = Path(args.output)

    # 执行转换
    convert_nii_to_png(
        src_path=input_path,
        dst_path=output_path,
        window_center=args.window_center,
        window_width=args.window_width,
        empty_slice_ratio=args.empty_ratio,
        tumor_only=args.tumor_only,
        min_tumor_pixels=args.min_tumor_pixels,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
