#!/usr/bin/env python3
"""
Medical Decathlon Lung Tumor Segmentation 数据下载脚本

从 Kaggle 下载数据集并保存到指定目录
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
    path = kagglehub.dataset_download("jadhavrajveer/medical-decathlon-lung-tumor-segmentation")
    print(f"下载完成: {path}")
    return Path(path)


def main():
    parser = argparse.ArgumentParser(
        description="下载 Medical Decathlon Lung Tumor Segmentation 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载到指定目录
  python download_lung_tumor.py --output ./data/raw

  # 仅显示下载路径，不复制
  python download_lung_tumor.py
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

    args = parser.parse_args()

    # 下载数据集
    download_path = download_dataset()

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
        shutil.copytree(download_path, output_path)
        print("复制完成!")
        print(f"\n数据集路径: {output_path}")
    else:
        print(f"\n数据集路径: {download_path}")
        print("提示: 使用 --output 参数可将数据复制到指定目录")


if __name__ == "__main__":
    main()
