#!/usr/bin/env python3
"""
Medical Image Segmentation 数据下载脚本

从 Kaggle 下载 modaresimr/medical-image-segmentation 数据集
这是一个 40GB 的大型医学图像分割数据集
"""

import argparse
import shutil
from pathlib import Path


def download_dataset() -> Path:
    """从 Kaggle 下载数据集"""
    try:
        import kagglehub
    except ImportError:
        print("错误: 请先安装 kagglehub")
        print("  pip install kagglehub")
        print("\n并配置 Kaggle API:")
        print("  https://www.kaggle.com/docs/api")
        exit(1)

    print("正在从 Kaggle 下载数据集...")
    print("注意: 这是一个 40GB 的大型数据集，下载可能需要较长时间")
    path = kagglehub.dataset_download("modaresimr/medical-image-segmentation")
    print(f"下载完成: {path}")
    return Path(path)


def explore_dataset(path: Path) -> None:
    """探索数据集结构"""
    print("\n数据集结构:")
    print("=" * 50)

    def print_tree(directory: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0):
        if current_depth >= max_depth:
            return

        items = sorted(directory.iterdir())
        dirs = [item for item in items if item.is_dir()]
        files = [item for item in items if item.is_file()]

        # 显示文件数量
        if files:
            extensions = {}
            for f in files:
                ext = f.suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1

            for ext, count in sorted(extensions.items()):
                print(f"{prefix}[{count} {ext or 'no-ext'} 文件]")

        # 递归显示目录
        for i, d in enumerate(dirs[:10]):  # 最多显示10个目录
            is_last = (i == len(dirs) - 1) or (i == 9)
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}{d.name}/")

            new_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(d, new_prefix, max_depth, current_depth + 1)

        if len(dirs) > 10:
            print(f"{prefix}... 还有 {len(dirs) - 10} 个目录")

    print_tree(path)
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="下载 Medical Image Segmentation 数据集 (40GB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载到指定目录
  python download_medical_segmentation.py --output ./data/medical_segmentation

  # 仅显示下载路径，不复制
  python download_medical_segmentation.py

  # 下载并探索数据集结构
  python download_medical_segmentation.py --explore

注意:
  - 数据集大小约 40GB，请确保有足够的磁盘空间
  - 下载需要配置 Kaggle API 密钥
        """
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="保存数据集的目标目录（不指定则仅下载到 kagglehub 缓存）"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="如果目标目录已存在，强制覆盖"
    )
    parser.add_argument(
        "--explore", "-e",
        action="store_true",
        help="下载后探索数据集结构"
    )

    args = parser.parse_args()

    # 下载数据集
    download_path = download_dataset()

    # 探索数据集结构
    if args.explore:
        explore_dataset(download_path)

    # 如果指定了输出目录，复制数据
    if args.output:
        output_path = Path(args.output)

        if output_path.exists():
            if args.force:
                print(f"目标目录已存在，正在删除: {output_path}")
                shutil.rmtree(output_path)
            else:
                print(f"错误: 目标目录已存在: {output_path}")
                print("使用 --force 强制覆盖")
                exit(1)

        print(f"正在复制数据到: {output_path}")
        print("注意: 复制 40GB 数据可能需要较长时间...")
        shutil.copytree(download_path, output_path)
        print("复制完成!")
        print(f"\n数据集路径: {output_path}")
    else:
        print(f"\n数据集路径: {download_path}")
        print("提示: 使用 --output 参数可将数据复制到指定目录")
        print("提示: 使用 --explore 参数可查看数据集结构")


if __name__ == "__main__":
    main()
