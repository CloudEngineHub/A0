# brew install poppler
# pip install pdf2image
# python scripts/pdf2png.py --src ICCV2025-Author-Kit-Feb --des a0_latex # --force

from pathlib import Path
import shutil
import argparse
from pdf2image import convert_from_path

def convert_files(source_root: Path, target_root: Path):
    """
    遍历源文件夹 source_root 中所有文件，
    在目标文件夹 target_root 中构造相同的目录结构，
    对于 PDF 文件转换为 PNG（只转换第一页），
    其他图像文件直接复制。
    """
    # 使用 rglob 遍历所有子文件
    for file in source_root.rglob("*"):
        if file.is_file():
            # 计算相对路径
            relative_path = file.relative_to(source_root)
            # 如果是 PDF 文件，则转换为 PNG（扩展名改为 .png）
            if file.suffix.lower() == ".pdf":
                target_file = target_root / relative_path.with_suffix(".png")
                target_file.parent.mkdir(parents=True, exist_ok=True)
                try:
                    # 转换 PDF 为图像，只转换第一页
                    images = convert_from_path(str(file), first_page=1, last_page=1)
                    images[0].save(target_file, "PNG")
                    print(f"已转换：{file} -> {target_file}")
                except Exception as e:
                    print(f"转换文件 {file} 时出错：{e}")
            # 如果是常见图像文件，则直接复制
            elif file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
                target_file = target_root / relative_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(file, target_file)
                    print(f"已复制：{file} -> {target_file}")
                except Exception as e:
                    print(f"复制文件 {file} 时出错：{e}")

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(
        description="根据源文件夹路径 --src，将所有 PDF 文件转换为 PNG，并复制其他图像文件到目标文件夹 --des 中，保持原有文件结构。"
    )
    parser.add_argument("--src", required=True, help="源文件夹路径")
    parser.add_argument("--des", required=True, help="目标文件夹路径")
    parser.add_argument("--force", "-f", action="store_true", help="若目标文件夹存在，则先删除后创建")
    args = parser.parse_args()

    source_root = Path(args.src)
    target_root = Path(args.des)

    # 判断目标文件夹是否存在
    if target_root.exists():
        if args.force:
            print(f"目标文件夹 {target_root} 已存在，将进行强制覆盖，删除原文件夹...")
            shutil.rmtree(target_root)
            target_root.mkdir(parents=True, exist_ok=True)
        else:
            raise FileExistsError(f"目标文件夹 {target_root} 已存在。若要覆盖，请使用 --force 参数。")
    else:
        target_root.mkdir(parents=True, exist_ok=True)

    # 开始处理文件
    convert_files(source_root, target_root)

if __name__ == '__main__':
    main()