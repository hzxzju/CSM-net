"""递归修正 new_images 文件名末尾的 lightdirection。

预览：python tools/fix_lightdirection.py
执行：python tools/fix_lightdirection.py --apply
指定目录：python tools/fix_lightdirection.py --root "其他目录" --apply
"""

import argparse
import re
from pathlib import Path


DIRECTIONS = {"03": "high_right", "04": "high_left", "05": "hightop"}

# 从文件名末尾识别 __imageid_lightdirection，并保留前缀和扩展名。
# 示例：class__lotid_processid_class__caseid__03_wrong.png
FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>.+__)(?P<imageid>03|04|05)_"
    r"(?P<direction>[^.]+)(?P<extension>(?:\.[^.]+)*)$"
)


def corrected_name(filename: str) -> str:
    """只有 imageid 为 03、04、05 时才替换方向，其他部分保持原样。"""
    match = FILENAME_PATTERN.fullmatch(filename)
    if match is None:
        return filename
    return (
        f"{match['prefix']}{match['imageid']}_"
        f"{DIRECTIONS[match['imageid']]}{match['extension']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "new_images",
        help="待处理目录，默认是项目中的 new_images",
    )
    parser.add_argument("--apply", action="store_true", help="执行改名；默认只预览")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"目录不存在或不是文件夹：{root}")

    changes = []
    targets = set()
    for source in sorted(root.rglob("*")):
        if source.is_symlink() or not source.is_file():
            continue
        new_name = corrected_name(source.name)
        if new_name == source.name:
            continue
        target = source.with_name(new_name)
        # 先检查全部计划，发现冲突时不会开始改名。
        if target.exists() or target.is_symlink() or target in targets:
            parser.error(f"目标文件重名，已停止：{target}")
        targets.add(target)
        changes.append((source, target))

    for source, target in changes:
        print(f"{source.relative_to(root)} -> {target.relative_to(root)}")

    if not args.apply:
        print(f"预览完成：{len(changes)} 个文件需要改名。添加 --apply 执行。")
        return 0

    completed = 0
    try:
        for source, target in changes:
            # 再次检查，避免预览后出现的新文件被覆盖。
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"目标文件已存在：{target}")
            source.rename(target)
            completed += 1
    except OSError as exc:
        print(f"改名失败：{exc}；已完成 {completed}/{len(changes)} 个文件。")
        return 1
    print(f"完成：已修正 {completed} 个文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
