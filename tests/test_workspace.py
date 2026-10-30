import ast
from collections import Counter
from pathlib import Path

from PIL import Image


def test_all_workspace_images_are_expected_format():
    files = list(Path("image").glob("*.jpg"))
    assert len(files) == 1475
    sizes = Counter()
    modes = Counter()
    for path in files:
        with Image.open(path) as image:
            sizes[image.size] += 1
            modes[image.mode] += 1
    assert sizes == {(480, 480): 1475}
    assert modes == {"RGB": 1475}


def test_python_sources_parse_without_optional_ml_dependencies():
    for path in Path("src").rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

