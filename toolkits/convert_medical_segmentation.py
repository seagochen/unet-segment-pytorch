#!/usr/bin/env python3
"""
Medical Image Segmentation 数据转换脚本

将 modaresimr/medical-image-segmentation 数据集转换为 PNG 切片格式
输出格式与 convert_nii_to_png.py 相同，便于统一训练

数据集结构:
  medical_segmentation/
  └── Task00X_*/
      ├── CT.zip              # CT 图像 (NIfTI 格式，需解压)
      ├── GroundTruth/        # 标签 (pickle 格式，稀疏存储)
      │   ├── 101.pkl
      │   └── ...
      ├── Predictions/        # 预测结果 (不需要)
      └── metadata.json       # 元信息
"""

import argparse
import json
import pickle
import random
import sys
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

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


# ============================================================================
# Fake classes for loading pickle files without evalseg module
# ============================================================================

class FakeClass:
    """Fake class to load pickle objects without original module."""
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


class FakeModule:
    """Fake module that returns FakeClass for any attribute."""
    def __getattr__(self, name):
        return FakeClass


def setup_fake_modules():
    """Setup fake modules to load evalseg pickle files."""
    for mod_name in ['evalseg', 'evalseg.io', 'evalseg.io.segment_array',
                     'evalseg.io.single_segment', 'evalseg.metrics']:
        sys.modules[mod_name] = FakeModule()


# ============================================================================
# Image processing utilities
# ============================================================================

def normalize_image(
    data: np.ndarray,
    window_center: Optional[int] = None,
    window_width: Optional[int] = None,
    percentile_clip: Tuple[float, float] = (0.5, 99.5)
) -> np.ndarray:
    """
    图像归一化到 [0, 255]

    Args:
        data: 图像数据
        window_center: 窗位 (如果为 None，使用自动模式)
        window_width: 窗宽 (如果为 None，使用自动模式)
        percentile_clip: 百分位数裁剪范围（自动模式）

    Returns:
        归一化到 [0, 255] 的 uint8 数据
    """
    data = data.astype(np.float32)
    data_min, data_max = data.min(), data.max()

    # 如果数据已经在 [0, 1] 范围内，直接缩放
    if data_min >= 0 and data_max <= 1.0:
        return (data * 255).astype(np.uint8)

    # 如果数据已经在 [0, 255] 范围内，直接转换
    if data_min >= 0 and data_max <= 255:
        return data.astype(np.uint8)

    # 使用窗宽窗位（CT 数据）
    if window_center is not None and window_width is not None:
        min_val = window_center - window_width // 2
        max_val = window_center + window_width // 2
        data = np.clip(data, min_val, max_val)
        data = (data - min_val) / (max_val - min_val) * 255
        return data.astype(np.uint8)

    # 自动模式：使用百分位数裁剪
    low = np.percentile(data, percentile_clip[0])
    high = np.percentile(data, percentile_clip[1])

    if high - low < 1e-6:
        return np.zeros_like(data, dtype=np.uint8)

    data = np.clip(data, low, high)
    data = (data - low) / (high - low) * 255
    return data.astype(np.uint8)


def load_ground_truth_pkl(pkl_path: Path) -> np.ndarray:
    """
    加载 GroundTruth pickle 文件并重建完整的标签体积

    Args:
        pkl_path: pickle 文件路径

    Returns:
        3D numpy array (H, W, D) 的标签体积
    """
    setup_fake_modules()

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 创建空白标签体积
    shape = data.shape  # (512, 512, num_slices)
    label_volume = np.zeros(shape, dtype=np.uint8)

    # 将每个 segment 的数据填入对应位置
    for seg in data.segments:
        if hasattr(seg, 'data') and hasattr(seg, 'roi'):
            roi = seg.roi
            seg_data = seg.data
            # roi 是 tuple of slices: (slice_x, slice_y, slice_z)
            label_volume[roi] = np.maximum(label_volume[roi], seg_data)

    return label_volume


# ============================================================================
# Dataset exploration
# ============================================================================

def explore_dataset(src_path: Path) -> Dict[str, Any]:
    """
    探索数据集结构

    Returns:
        数据集信息字典
    """
    src_path = Path(src_path)
    info = {
        "tasks": [],
        "total_volumes": 0,
    }

    # 查找所有 Task 目录
    task_dirs = sorted([d for d in src_path.iterdir()
                       if d.is_dir() and d.name.startswith("Task")])

    for task_dir in task_dirs:
        ct_zip = task_dir / "CT.zip"
        gt_dir = task_dir / "GroundTruth"

        if not ct_zip.exists() or not gt_dir.exists():
            continue

        # 读取 metadata.json
        task_info = {
            "name": task_dir.name,
            "path": str(task_dir),
            "ct_zip": str(ct_zip),
            "gt_dir": str(gt_dir),
            "num_labeled": 0,
            "num_total_ct": 0,
            "labels": {},
            "modality": "CT"
        }

        metadata_file = task_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    meta = json.load(f)
                task_info["labels"] = meta.get("labels", {})
                task_info["description"] = meta.get("description", "")
            except (json.JSONDecodeError, KeyError):
                pass

        # 统计标注文件数量
        gt_files = list(gt_dir.glob("*.pkl"))
        task_info["num_labeled"] = len(gt_files)

        # 统计 CT 文件数量
        try:
            with zipfile.ZipFile(ct_zip, 'r') as zf:
                nii_files = [n for n in zf.namelist() if n.endswith('.nii') or n.endswith('.nii.gz')]
                task_info["num_total_ct"] = len(nii_files)
        except zipfile.BadZipFile:
            pass

        info["total_volumes"] += task_info["num_labeled"]
        info["tasks"].append(task_info)

    return info


# ============================================================================
# Conversion functions
# ============================================================================

def find_labeled_pairs(task_info: Dict[str, Any]) -> List[Tuple[str, Path]]:
    """
    查找所有有标签的 CT-Label 对

    Returns:
        List of (case_id, gt_pkl_path) tuples
    """
    gt_dir = Path(task_info["gt_dir"])
    pairs = []

    for pkl_file in sorted(gt_dir.glob("*.pkl")):
        case_id = pkl_file.stem  # e.g., "101"
        pairs.append((case_id, pkl_file))

    return pairs


def convert_task_to_png(
    task_info: Dict[str, Any],
    dst_path: Path,
    window_center: Optional[int] = None,
    window_width: Optional[int] = None,
    empty_slice_ratio: float = 0.1,
    tumor_only: bool = False,
    min_tumor_pixels: int = 0,
    seed: int = 42
) -> dict:
    """
    将一个任务的数据转换为 PNG 切片

    Args:
        task_info: 任务信息字典
        dst_path: 输出目录
        window_center: CT 窗位
        window_width: CT 窗宽
        empty_slice_ratio: 保留的空白切片比例
        tumor_only: 只保存有标注的切片
        min_tumor_pixels: 最小标注像素数
        seed: 随机种子

    Returns:
        统计信息
    """
    random.seed(seed)
    np.random.seed(seed)

    task_name = task_info.get("name", "unknown").replace(" ", "_")
    ct_zip_path = Path(task_info["ct_zip"])

    # 创建输出目录
    images_out = dst_path / "images"
    labels_out = dst_path / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # 查找所有有标签的数据对
    pairs = find_labeled_pairs(task_info)
    print(f"找到 {len(pairs)} 个有标签的 CT 体积")

    if len(pairs) == 0:
        print(f"警告: 任务 {task_name} 没有找到有效的数据对")
        return {}

    # 调整参数
    if tumor_only:
        empty_slice_ratio = 0.0
        print("模式: 仅保存含标注的切片")
    else:
        print(f"模式: 保留 {empty_slice_ratio*100:.0f}% 的空白切片")

    # 统计信息
    stats = {
        "task_name": task_name,
        "total_volumes": len(pairs),
        "total_slices": 0,
        "saved_slices": 0,
        "slices_with_label": 0,
        "slices_filtered_small": 0,
        "empty_slices_saved": 0,
        "empty_slices_skipped": 0,
        "files": []
    }

    # 打开 CT.zip
    with zipfile.ZipFile(ct_zip_path, 'r') as zf:
        # 处理每个有标签的 CT 体积
        for case_id, gt_pkl_path in tqdm(pairs, desc=f"转换 {task_name}"):
            # 查找对应的 CT 文件
            ct_filename = f"{case_id}.nii"
            ct_filename_gz = f"{case_id}.nii.gz"

            if ct_filename in zf.namelist():
                ct_file = ct_filename
            elif ct_filename_gz in zf.namelist():
                ct_file = ct_filename_gz
            else:
                print(f"警告: 找不到 CT 文件: {case_id}")
                continue

            # 从 zip 中读取 CT 数据
            with zf.open(ct_file) as f:
                # nibabel 需要文件路径，所以我们用 BytesIO
                import io
                import tempfile

                # 将数据写入临时文件
                with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name

                try:
                    img_nii = nib.load(tmp_path)
                    img_data = img_nii.get_fdata()
                finally:
                    Path(tmp_path).unlink()

            # 加载标签数据
            label_data = load_ground_truth_pkl(gt_pkl_path)

            # 处理 4D 图像（多模态）
            if img_data.ndim == 4:
                img_data = img_data[:, :, :, 0]

            # 确保形状匹配
            if img_data.shape != label_data.shape:
                print(f"警告: 形状不匹配 CT={img_data.shape} vs Label={label_data.shape}")
                # 尝试调整
                min_shape = tuple(min(a, b) for a, b in zip(img_data.shape, label_data.shape))
                img_data = img_data[:min_shape[0], :min_shape[1], :min_shape[2]]
                label_data = label_data[:min_shape[0], :min_shape[1], :min_shape[2]]

            # 处理每个切片
            num_slices = img_data.shape[2]
            stats["total_slices"] += num_slices

            empty_slice_indices = []

            for slice_idx in range(num_slices):
                img_slice = img_data[:, :, slice_idx]
                label_slice = label_data[:, :, slice_idx]

                # 计算标注像素数
                label_pixels = np.sum(label_slice > 0)
                has_valid_label = label_pixels >= min_tumor_pixels and label_pixels > 0

                if has_valid_label:
                    stats["slices_with_label"] += 1
                elif label_pixels > 0:
                    stats["slices_filtered_small"] += 1
                    empty_slice_indices.append(slice_idx)
                    continue
                else:
                    empty_slice_indices.append(slice_idx)
                    continue

                # 保存有标注的切片
                _save_slice(
                    img_slice, label_slice,
                    case_id, slice_idx, task_name,
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
                        case_id, slice_idx, task_name,
                        images_out, labels_out,
                        window_center, window_width,
                        stats
                    )
                    stats["empty_slices_saved"] += 1

            stats["empty_slices_skipped"] += len(empty_slice_indices) - num_empty_to_keep

    return stats


def _save_slice(
    img_slice: np.ndarray,
    label_slice: np.ndarray,
    case_id: str,
    slice_idx: int,
    task_name: str,
    images_out: Path,
    labels_out: Path,
    window_center: Optional[int],
    window_width: Optional[int],
    stats: dict
) -> None:
    """保存单个切片为 PNG"""
    # 归一化图像
    img_normalized = normalize_image(img_slice, window_center, window_width)

    # 标签二值化 (确保是 0 或 255)
    label_binary = ((label_slice > 0) * 255).astype(np.uint8)

    # 生成文件名
    slice_name = f"{task_name}_{case_id}_slice_{slice_idx:04d}.png"

    # 保存
    Image.fromarray(img_normalized).save(images_out / slice_name)
    Image.fromarray(label_binary).save(labels_out / slice_name)

    stats["saved_slices"] += 1
    stats["files"].append(slice_name)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="将 Medical Image Segmentation 数据集转换为 PNG 切片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 探索数据集结构
  python convert_medical_segmentation.py --input ~/datasets/medical_segmentation --explore

  # 转换肝脏肿瘤任务（使用肝窗）
  python convert_medical_segmentation.py \\
      --input ~/datasets/medical_segmentation \\
      --output ./liver_tumor_segmentation \\
      --task Task001_LiverTumor \\
      --tumor-only \\
      --window-center 40 --window-width 400

  # 转换肺部（使用肺窗）
  python convert_medical_segmentation.py \\
      --input ~/datasets/medical_segmentation \\
      --output ./lung_segmentation \\
      --task Task003 \\
      --window-center -600 --window-width 1500

CT 窗宽窗位参考:
  肺窗:   --window-center -600 --window-width 1500
  肝窗:   --window-center 40   --window-width 400
  骨窗:   --window-center 400  --window-width 1800
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="数据集根目录"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="PNG 输出目录（探索模式可不指定）"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default=None,
        help="只转换指定任务（如 Task001_LiverTumor）"
    )
    parser.add_argument(
        "--explore", "-e",
        action="store_true",
        help="探索数据集结构"
    )
    parser.add_argument(
        "--window-center", "-wc",
        type=int,
        default=None,
        help="CT 窗位（不指定则使用自动归一化）"
    )
    parser.add_argument(
        "--window-width", "-ww",
        type=int,
        default=None,
        help="CT 窗宽（不指定则使用自动归一化）"
    )
    parser.add_argument(
        "--empty-ratio",
        type=float,
        default=0.1,
        help="保留的空白切片比例 (默认: 0.1)"
    )
    parser.add_argument(
        "--tumor-only",
        action="store_true",
        help="只保存含标注的切片"
    )
    parser.add_argument(
        "--min-pixels", "-m",
        type=int,
        default=0,
        help="最小标注像素数（过滤太小的标注）"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入目录不存在: {input_path}")
        exit(1)

    # 探索数据集
    print(f"正在分析数据集: {input_path}")
    dataset_info = explore_dataset(input_path)

    if not dataset_info["tasks"]:
        print("错误: 未找到有效的任务数据")
        exit(1)

    # 显示数据集信息
    print("\n" + "=" * 60)
    print("数据集信息")
    print("=" * 60)
    print(f"任务数量:        {len(dataset_info['tasks'])}")
    print(f"有标签的体积数:  {dataset_info['total_volumes']}")
    print("\n可用任务:")
    for task in dataset_info["tasks"]:
        print(f"  - {task['name']}:")
        print(f"      有标签: {task['num_labeled']} 个体积")
        print(f"      总 CT:  {task['num_total_ct']} 个体积")
        if task.get("labels"):
            labels_str = ", ".join(f"{k}={v}" for k, v in task["labels"].items())
            print(f"      标签:   {labels_str}")
    print("=" * 60)

    # 如果只是探索模式，到此结束
    if args.explore:
        return

    # 检查输出目录
    if not args.output:
        print("\n错误: 请指定输出目录 (--output)")
        exit(1)

    output_path = Path(args.output)

    # 筛选要转换的任务
    tasks_to_convert = dataset_info["tasks"]
    if args.task:
        tasks_to_convert = [t for t in tasks_to_convert
                           if args.task.lower() in t["name"].lower()]
        if not tasks_to_convert:
            print(f"错误: 未找到匹配的任务: {args.task}")
            print("可用任务:", [t["name"] for t in dataset_info["tasks"]])
            exit(1)

    print(f"\n将转换 {len(tasks_to_convert)} 个任务到: {output_path}")

    # 转换每个任务
    all_stats = []
    for task_info in tasks_to_convert:
        print(f"\n处理任务: {task_info['name']}")
        stats = convert_task_to_png(
            task_info=task_info,
            dst_path=output_path,
            window_center=args.window_center,
            window_width=args.window_width,
            empty_slice_ratio=args.empty_ratio,
            tumor_only=args.tumor_only,
            min_tumor_pixels=args.min_pixels,
            seed=args.seed
        )
        if stats:
            all_stats.append(stats)

    # 汇总统计
    if all_stats:
        total_stats = {
            "tasks": [s["task_name"] for s in all_stats],
            "total_volumes": sum(s["total_volumes"] for s in all_stats),
            "total_slices": sum(s["total_slices"] for s in all_stats),
            "saved_slices": sum(s["saved_slices"] for s in all_stats),
            "slices_with_label": sum(s["slices_with_label"] for s in all_stats),
            "empty_slices_saved": sum(s["empty_slices_saved"] for s in all_stats),
            "empty_slices_skipped": sum(s["empty_slices_skipped"] for s in all_stats),
        }

        # 保存统计信息
        stats_file = output_path / "dataset_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(total_stats, f, indent=2, ensure_ascii=False)

        # 保存文件列表
        all_files = []
        for s in all_stats:
            all_files.extend(s.get("files", []))
        file_list = output_path / "file_list.txt"
        with open(file_list, 'w') as f:
            f.write('\n'.join(all_files))

        # 打印统计
        print("\n" + "=" * 60)
        print("转换完成!")
        print("=" * 60)
        print(f"转换的任务:      {', '.join(total_stats['tasks'])}")
        print(f"总体积数:        {total_stats['total_volumes']}")
        print(f"总切片数:        {total_stats['total_slices']}")
        print(f"保存的切片数:    {total_stats['saved_slices']}")
        print(f"  - 含标注切片:  {total_stats['slices_with_label']}")
        print(f"  - 空白切片:    {total_stats['empty_slices_saved']}")
        print(f"跳过的空白切片:  {total_stats['empty_slices_skipped']}")
        print(f"输出目录:        {output_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
