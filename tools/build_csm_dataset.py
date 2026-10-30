"""按 case 原样复制 block_etch 的 01–05 图像，并生成 CSM_Dataset 的信息模板。

运行：python3 tools/build_csm_dataset.py
预览：python3 tools/build_csm_dataset.py --dry-run
如需额外包含光学图：python3 tools/build_csm_dataset.py --include-optical
可选划分：python3 tools/build_csm_dataset.py --split-ratios 0.7 0.15 0.15

依赖 Pillow。保留源文件；已有 JSON、标注和非空划分文件不会被覆盖。
"""

import argparse
import hashlib
import json
import math
import random
import re
import shutil
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILENAME_PATTERN = re.compile(
    r"^block_etch__(?P<sample_id>(?P<lot_id>[^_]+)_(?P<process_id>.+?)__"
    r"(?P<case_id>case[0-9]{4}))__(?P<view>0[1-6])_"
    r"(?P<label>.+)\.(?:jpg|jpeg|png|tif|tiff)$",
    re.IGNORECASE,
)
VIEW_STEMS = {
    "01": "low_mag/01",
    "02": "normal_reference/02",
    "03": "high_mag/angle_03",
    "04": "high_mag/angle_04",
    "05": "high_mag/angle_05",
    "06": "optical/06",
}
REQUIRED_VIEWS = {"01", "02", "03", "04", "05"}
SPLIT_NAMES = ("train", "val", "test")


def write_text_if_missing(path, text):
    """使用独占创建，保留人工填写或已有的内容。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    except FileExistsError:
        pass


def write_json_if_missing(path, content):
    write_text_if_missing(
        path, json.dumps(content, ensure_ascii=False, indent=2) + "\n"
    )


def writing_json(sample_dir, view_paths, *, dry_run=False):
    """每个 case 记录一份高倍率缺陷 mask，以及各视角的低倍率视野 bbox。

    路径均相对于 case 目录。dry_run=True 只返回模板，不创建目录或文件。
    实际写入时保留已有 JSON，避免覆盖人工标注。
    """
    reference = view_paths.get("01")
    templates = {
        "registration.json": {
            "schema_version": "2.0",
            "reference_image": reference,
            "coordinate_system": {
                "unit": "pixel",
                "origin": "top_left",
                "axis_order": ["x", "y"],
                "x_direction": "right",
                "y_direction": "down",
                "bbox_format": "[xmin, ymin, xmax, ymax]",
                "bbox_max_exclusive": True,
            },
            "high_mag": {
                view: {
                    "image": view_paths.get(view),
                    "bbox_in_low_mag_xyxy": None,
                    "status": (
                        "missing_image" if view not in view_paths else
                        "missing_reference" if reference is None else
                        "not_annotated"
                    ),
                }
                for view in ("03", "04", "05")
            },
        },
        "annotations/instances.json": {
            "schema_version": "2.0",
            "status": (
                "not_annotated" if any(view in view_paths for view in ("03", "04", "05"))
                else "missing_image"
            ),
            "coordinate_image": None,
            "coordinate_system": {
                "unit": "pixel",
                "origin": "top_left",
                "axis_order": ["x", "y"],
                "x_direction": "right",
                "y_direction": "down",
            },
            "background_value": 0,
            "instance_mask": None,
            "instances": None,
        },
    }
    if not dry_run:
        for name, content in templates.items():
            write_json_if_missing(sample_dir / name, content)
    return templates


def collect_samples(source, include_optical=False):
    samples = {}
    for path in sorted(source.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        match = FILENAME_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"无法识别文件名：{path.name}")
        if match["view"] == "06" and not include_optical:
            continue
        sample = samples.setdefault(match["sample_id"], {})
        if match["view"] in sample:
            raise ValueError(f"同一 case 的视角编号重复：{path.name}")
        sample[match["view"]] = (path, match.groupdict())
    return samples


def copy_image(source, target):
    """直接复制原文件；不解码重存、不重新压缩，保留原始格式与内容。"""
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with Image.open(source) as image:
        source_format = image.format
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != source_hash:
                raise ValueError(f"目标文件已存在且内容不同：{target}")
            created = False
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            created = True
        return {
            "path": target.name,
            "width_px": image.width,
            "height_px": image.height,
            "mode": image.mode,
            "source_filename": source.name,
            "source_format": source_format,
            "source_sha256": source_hash,
            "magnification": None,
            "angle_deg": None,
            "pixel_size": {"x": None, "y": None, "unit": "um"},
        }, created


def build_sample(dataset_root, sample_id, views):
    sample_dir = dataset_root / "block_etch" / sample_id
    for folder in ("high_mag", "low_mag", "normal_reference", "annotations"):
        (sample_dir / folder).mkdir(parents=True, exist_ok=True)

    images = {}
    view_paths = {
        view: VIEW_STEMS[view] + source.suffix
        for view, (source, _) in views.items()
    }
    created = 0
    for view, (source, fields) in sorted(views.items()):
        relative_path = view_paths[view]
        info, is_new = copy_image(source, sample_dir / relative_path)
        info.update({"path": relative_path, "filename_label": fields["label"]})
        images[view] = info
        created += is_new

    fields = next(iter(views.values()))[1]
    missing = sorted(REQUIRED_VIEWS - views.keys())
    write_json_if_missing(sample_dir / "metadata.json", {
        "schema_version": "1.0",
        "sample_id": sample_id,
        "defect_class": "block_etch",
        "filename_fields": {
            "lot_id": fields["lot_id"],
            "case_id": fields["case_id"],
            "process_tokens": ["ct", "et"],
        },
        "physical_sample_id": None,
        "completeness": {
            "required_view_codes": sorted(REQUIRED_VIEWS),
            "missing_view_codes": missing,
            "complete": not missing,
        },
        "images": images,
        "acquisition_conditions": {
            "instrument": None,
            "accelerating_voltage_kv": None,
            "working_distance_mm": None,
            "detector": None,
            "acquired_at": None,
        },
        "notes": "倍率、真实角度、像素尺寸和采集条件需根据原始记录补充。",
    })

    writing_json(sample_dir, view_paths)
    return created


def prepare_splits(samples, ratios, seed):
    if ratios is None:
        return {name: "" for name in SPLIT_NAMES}
    eligible = sorted(
        sample_id for sample_id, views in samples.items()
        if REQUIRED_VIEWS <= views.keys()
    )
    random.Random(seed).shuffle(eligible)
    total_ratio = sum(ratios)
    expected = [len(eligible) * ratio / total_ratio for ratio in ratios]
    counts = [math.floor(value) for value in expected]
    remainder = len(eligible) - sum(counts)
    order = sorted(range(3), key=lambda i: (-(expected[i] - counts[i]), i))
    for index in order[:remainder]:
        counts[index] += 1
    result = {}
    start = 0
    for name, count in zip(SPLIT_NAMES, counts):
        result[name] = "".join(
            f"block_etch/{sample_id}\n" for sample_id in eligible[start:start + count]
        )
        start += count
    return result


README = """# CSM_Dataset

图像来自 new_images/images/block_etch，按文件名中的 case 分组，原文件保留。
默认仅整理 01–05，跳过 06 光学图；仅包含 06 的 case 不建立目录。
图像直接复制，保留原始格式、扩展名和文件内容，不转换、不重新压缩。
当前源文件为 JPG，因此以下示例使用 .jpg。脚本不缩放、不裁剪；
文件名中的 03/04/05 是视角编号，
不代表已经知道真实拍摄角度。

## 目录与图像对应关系

```text
CSM_Dataset/
├── block_etch/
│   └── at00561_ct_et_block_etch__case0008/
│       ├── high_mag/
│       │   ├── angle_03.jpg
│       │   ├── angle_04.jpg
│       │   └── angle_05.jpg
│       ├── low_mag/01.jpg
│       ├── normal_reference/02.jpg
│       ├── annotations/
│       │   ├── instances.json
│       │   └── instance_mask.png  # 实际标注完成后生成，每个 case 仅一份
│       ├── registration.json
│       └── metadata.json
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

每个 case 只保存一份高倍率缺陷 mask：annotations/instance_mask.png。
annotations/instances.json 不再按 03、04、05 分别保存 mask 记录。
coordinate_image 初始为 null，标注时填写实际使用的高倍率参考图，
例如 high_mag/angle_03.jpg；mask 使用该参考图的像素坐标。
初始 status 为 not_annotated，instances 和 instance_mask 为 null。
尚未标注时不生成空白掩码。完成标注后将 status 改为 annotated，
instance_mask 填为 annotations/instance_mask.png，instances 改为实例对象列表。
mask 与所选高倍率参考图的宽高一致，左上角为 (0, 0)，x 向右、
y 向下；0 表示背景，正整数为实例 ID，建议保存为 16 位 PNG。
不需要额外手工标注高倍率缺陷 bbox。实例对象示例：

```json
{"instance_id": 1, "category": "block_etch"}
```

## 低倍率视野框与元数据

registration.json 的 reference_image 指向 low_mag/01.jpg。每个高倍率
视角分别记录 image、bbox_in_low_mag_xyxy 和 status。
bbox 初始为 null，手工在低倍率图上框出该高倍率视角对应的视野后填写。
框使用低倍率原图的像素坐标，左上角为 (0, 0)，x 向右、y 向下，
格式为 [xmin, ymin, xmax, ymax]，最大边界不包含在框内。
完成视野框标注后将对应视角的 status 改为 annotated。
这里的 bbox 表示整个高倍率视野在低倍率图中的范围。

当前模板不记录变换矩阵、变换模型或配准方法。
metadata.json 保存各图尺寸、原始文件名、SHA256、视角标签，以及待补充的
倍率、真实角度、像素尺寸（um）、物理样本编号和采集条件。
从文件名读到的 lot_id/case_id 仅放在 filename_fields 中。
包含部分 01–05 的 case 会保留已有图像，并在 completeness 中记录缺失编号。

## 数据划分与重复运行

默认仅创建空的 train.txt/val.txt/test.txt，不擅自分配样本。
各行使用相对数据集根目录的路径，例如 block_etch/at00561_ct_et_block_etch__case0008。
可手工填写，或在这些文件仍为空时运行：

```bash
python3 tools/build_csm_dataset.py --split-ratios 0.7 0.15 0.15 --seed 42
```

自动划分只使用 01–05 完整的 case；按 case 整体划分，固定随机种子。
比例采用最大余数法取整。若多个 case 来自同一物理样本，正式训练前还需
根据 physical_sample_id 调整分组，避免同一物理样本跨集合。

重复运行时验证已有图像与源文件的 SHA256 一致，并保留已有 JSON 和标注文件。
已有非空划分文件与新划分不同时会报错，不会覆盖。
标注模板 schema_version 为 2.0，已有旧版 JSON 不会被自动迁移或覆盖。
--dry-run 会打印首个 case 的标注模板，供 review；不会写入文件。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "new_images/images/block_etch")
    parser.add_argument("--dest", type=Path, default=PROJECT_ROOT / "CSM_Dataset")
    optical = parser.add_mutually_exclusive_group()
    optical.add_argument("--include-optical", dest="include_optical", action="store_true",
                         help="额外包含 06 光学图；默认跳过")
    optical.add_argument("--exclude-optical", dest="include_optical", action="store_false",
                         help="仅包含 01–05（默认行为）")
    parser.set_defaults(include_optical=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split-ratios", nargs=3, type=float, metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error(f"源目录不存在：{args.source}")
    if args.split_ratios is not None and (
        any(not math.isfinite(r) or r < 0 for r in args.split_ratios)
        or not math.isfinite(sum(args.split_ratios))
        or sum(args.split_ratios) <= 0
    ):
        parser.error("划分比例必须为有限非负数，且总和大于零。")
    try:
        samples = collect_samples(args.source, args.include_optical)
        if not samples:
            parser.error("没有匹配到图像。")
        splits = prepare_splits(samples, args.split_ratios, args.seed)
        if args.split_ratios is not None:
            for name, content in splits.items():
                path = args.dest / "splits" / f"{name}.txt"
                if path.exists():
                    existing = path.read_text(encoding="utf-8")
                    if existing.strip() and existing != content:
                        parser.error(f"已有非空划分与请求不同，请使用其他目标目录：{path}")

        total = sum(len(views) for views in samples.values())
        print(f"共 {len(samples)} 个 case、{total} 张图像。")
        for sample_id, views in sorted(samples.items()):
            missing = sorted(REQUIRED_VIEWS - views.keys())
            print(f"{sample_id}: {','.join(sorted(views))}"
                  + (f"；缺少 {','.join(missing)}" if missing else ""))
        if args.dry_run:
            sample_id = sorted(samples)[0]
            view_paths = {
                view: VIEW_STEMS[view] + source.suffix
                for view, (source, _) in samples[sample_id].items()
            }
            sample_dir = args.dest / "block_etch" / sample_id
            templates = writing_json(sample_dir, view_paths, dry_run=True)
            print(f"\n首个 case 的标注模板预览：{sample_id}")
            for name, content in templates.items():
                print(f"\n{name}：")
                print(json.dumps(content, ensure_ascii=False, indent=2))
                if (sample_dir / name).exists():
                    print("该文件已存在；实际运行也会保留原文件，不自动迁移。")
            print("预览完成，未写入文件。")
            return

        created = sum(
            build_sample(args.dest, sample_id, views)
            for sample_id, views in sorted(samples.items())
        )
        for name, content in splits.items():
            path = args.dest / "splits" / f"{name}.txt"
            if args.split_ratios is not None and path.exists() and not path.read_text(encoding="utf-8").strip():
                path.write_text(content, encoding="utf-8")
            else:
                write_text_if_missing(path, content)
        write_text_if_missing(args.dest / "README.md", README)
        print(f"完成：原样复制 {created} 张图像，已有 {total - created} 张。")
        print(f"数据集：{args.dest.resolve()}")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"整理失败：{exc}\n")


if __name__ == "__main__":
    main()
