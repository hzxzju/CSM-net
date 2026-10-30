from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from lazystrike_cls.data import SEM_CLASS_NAMES, SEMClassificationDataset, class_name_from_filename


def test_workspace_image_format():
    image_dir = Path("image")
    if not image_dir.exists():
        pytest.skip("workspace image directory is unavailable")
    dataset = SEMClassificationDataset(image_dir)
    assert len(dataset) == 1475
    assert tuple(dataset.class_to_idx) == SEM_CLASS_NAMES
    sample = dataset[0]
    assert sample["pixel_values"].shape == (3, 480, 480)
    assert sample["pixel_values"].dtype == torch.float32
    assert isinstance(sample["class_label"], int)
    assert sample["class_name"] in SEM_CLASS_NAMES


@pytest.mark.parametrize(
    ("filename", "expected"),
    [("hole12.jpg", "hole"), ("pattern_deform8.jpg", "pattern_deform"), ("metal_3.jpg", "metal")],
)
def test_class_name_from_filename(filename, expected):
    assert class_name_from_filename(filename) == expected
