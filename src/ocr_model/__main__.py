"""Usage: PYTHONPATH=src python -m ocr_model region.png [--roi X Y W H]."""

import argparse
from dataclasses import asdict
import json
import sys

from .fov import TesseractFOVBaseline


def main() -> int:
    parser = argparse.ArgumentParser(description="Tesseract baseline：识别 SEM 右上角裁剪区域中的 FOV")
    parser.add_argument("image", help="图像路径；默认将整个输入当作已裁剪区域")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--scale", type=float, default=3.0)
    args = parser.parse_args()
    try:
        model = TesseractFOVBaseline(scale=args.scale)
        result = model.predict(
            args.image, roi=tuple(args.roi) if args.roi else None
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if result is None:
        print(json.dumps({"model": model.model_name, "fov": None,
                          "message": "未识别到明确的 FOV"}, ensure_ascii=False))
        return 1
    print(json.dumps({"model": model.model_name, **asdict(result),
                      "value_um": result.value_um}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
