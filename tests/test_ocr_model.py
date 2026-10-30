from pathlib import Path
import shutil

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from ocr_model import TesseractFOVBaseline, parse_fov


@pytest.mark.parametrize(
    "text,value,unit,value_um",
    [
        ("Location: (22393, 20472)\nFOV: 4.00um\nImage Rotation: 0.00 deg", 4, "um", 4),
        ("FOV = 250 nm", 250, "nm", 0.25),
        ("F0V: 2.5 µm", 2.5, "um", 2.5),
        ("fov: .75 μm", 0.75, "um", 0.75),
        ("FOV 1,250.50 um", 1250.5, "um", 1250.5),
        ("FOV: 0.2 mm", 0.2, "mm", 200),
        ("FOV: 4.00", 4, None, None),
    ],
)
def test_parse_annotation(text, value, unit, value_um):
    result = parse_fov(text)
    assert result is not None
    assert (result.value, result.unit, result.value_um) == (value, unit, value_um)
    assert result.raw_text == text


@pytest.mark.parametrize("text", [
    "Location: (12, 34)\nImage Rotation: 0.00 deg", "FOV: 0um",
    "FOV: -4um", "FOV: 4.2.3um", "FOV: 4abc", "FOV: 1,23um",
    "FOV: 4um\nFOV: 5um", "HFOV: 4um", "FOV:",
])
def test_reject_missing_malformed_or_ambiguous_value(text):
    assert parse_fov(text) is None


@pytest.mark.parametrize("image", [
    np.empty((0, 0), dtype=np.uint8), np.zeros((20, 20), dtype=np.float32),
    np.zeros((20, 20, 2), dtype=np.uint8),
])
def test_invalid_images(image):
    with pytest.raises(ValueError):
        TesseractFOVBaseline().predict(image)


@pytest.mark.parametrize("roi", [(-1, 0, 10, 10), (0, 0, 21, 10), (0, 0, 0, 10)])
def test_invalid_roi(roi):
    with pytest.raises(ValueError):
        TesseractFOVBaseline().predict(np.zeros((20, 20), dtype=np.uint8), roi=roi)


def test_real_tesseract_on_synthetic_crop(tmp_path):
    pytest.importorskip("pytesseract")
    if shutil.which("tesseract") is None:
        pytest.skip("Tesseract executable is not installed")
    crop = np.full((100, 420, 3), 255, dtype=np.uint8)
    cv2.putText(crop, "FOV: 4.00 um", (20, 62), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (0, 0, 0), 2, cv2.LINE_AA)
    path = tmp_path / "右上角.png"
    path.write_bytes(cv2.imencode(".png", crop)[1].tobytes())
    result = TesseractFOVBaseline().predict(path)
    assert result is not None
    assert (result.value, result.unit) == (4.0, "um")


@pytest.mark.parametrize("filename,expected", [("hole0.jpg", 4), ("hole100.jpg", 1), ("hole104.jpg", 1)])
def test_workspace_sem_annotations(filename, expected):
    pytest.importorskip("pytesseract")
    path = Path("image") / filename
    if not path.exists() or shutil.which("tesseract") is None:
        pytest.skip("Local SEM sample or Tesseract is unavailable")
    result = TesseractFOVBaseline().predict(path, roi=(285, 0, 195, 80))
    assert result is not None
    assert (result.value, result.unit) == (expected, "um")
