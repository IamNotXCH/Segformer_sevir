import sys
sys.path.append(".")

import argparse
from sevir_torch_wrap import download_SEVIR


def get_parser():
    parser = argparse.ArgumentParser(description="Download SEVIR dataset.")
    parser.add_argument("--dataset", default="sevir", type=str, help="Dataset name, should be 'sevir'.")
    parser.add_argument("--save", default="../../data", type=str, help="Directory to save the dataset.")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # 判断是否为 'sevir' 数据集并下载
    if args.dataset == "sevir":
        download_SEVIR(save_dir=args.save)  # 调用 download_SEVIR 函数
    else:
        raise ValueError(f"Wrong dataset name {args.dataset}! Must be 'sevir'.")

if __name__ == "__main__":
    main()
