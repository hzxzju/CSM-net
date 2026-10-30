"""Tesseract baseline for FOV recognition from cropped SEM annotations.

OpenCV handles preprocessing; Tesseract (via pytesseract) reads the text.
NumPy inputs follow OpenCV's BGR/BGRA channel order.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
import math
import re

import cv2
import numpy as np


_FOV_PATTERN = re.compile(
    r"\bF[O0]V[ \t]*[:=]?[ \t]*"
    r"(?P<value>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"[ \t]*(?P<unit>[uµμ]m|nm|mm|m)?"
    r"(?=$|[\s;|])",
    re.IGNORECASE,
)
_UNIT_TO_UM = {"nm": 0.001, "um": 1.0, "mm": 1000.0, "m": 1_000_000.0}


@dataclass(frozen=True)
class FOVResult:
    """Parsed value; a missing unit is preserved rather than assumed."""

    value: float
    unit: str | None
    raw_text: str
    preprocessing: str | None = None

    @property
    def value_um(self) -> float | None:
        return None if self.unit is None else self.value * _UNIT_TO_UM[self.unit]


def parse_fov(text: str) -> FOVResult | None:
    """Read a positive FOV value, without treating other numeric fields as FOV.

    Supports decimal points, grouped thousands, and nm/um/µm/μm/mm/m units.
    F0V is accepted as a common OCR spelling. Conflicting FOV values fail closed.
    """
    matches: dict[tuple[float, str | None], FOVResult] = {}
    for match in _FOV_PATTERN.finditer(text):
        value = float(match["value"].replace(",", ""))
        if not math.isfinite(value) or value <= 0:
            continue
        unit = match["unit"]
        if unit:
            unit = unit.lower().replace("µ", "u").replace("μ", "u")
        matches[(value, unit)] = FOVResult(value, unit, text)
    return next(iter(matches.values())) if len(matches) == 1 else None


class TesseractFOVBaseline:
    """The current FOV OCR baseline, using pretrained English Tesseract.

    By default, the whole input is the manually cropped top-right region.
    ``roi=(x, y, width, height)`` can instead select a region from a full image.
    OCR failures return None; missing dependencies and invalid inputs raise errors.
    """

    model_name = "tesseract_baseline"

    def __init__(self, *, scale: float = 3.0, timeout: float = 10.0) -> None:
        if not math.isfinite(scale) or scale < 1:
            raise ValueError("scale must be finite and >= 1")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and > 0")
        self.scale = scale
        self.timeout = timeout

    @staticmethod
    def _load(image: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(image, (str, Path)):
            # imdecode supports Chinese paths on platforms where imread may not.
            path = Path(image)
            data = np.fromfile(path, dtype=np.uint8)
            if not data.size:
                raise ValueError(f"Empty image file: {path}")
            image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Cannot decode image: {path}")
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a path or an OpenCV NumPy array")
        if image.size == 0 or image.dtype != np.uint8:
            raise ValueError("image must be a nonempty uint8 array")
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] in (3, 4):
            return image
        raise ValueError("image must be grayscale, BGR, or BGRA")

    def _variants(self, image: np.ndarray) -> Iterator[tuple[str, np.ndarray]]:
        if image.ndim == 3:
            code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            gray = cv2.cvtColor(image, code)
        else:
            gray = image
        enlarged = cv2.resize(
            gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC
        )
        _, binary = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # SEM annotations have bright letters with dark outlines on varied backgrounds.
        _, bright = cv2.threshold(enlarged, 200, 255, cv2.THRESH_BINARY)
        variants = [
            ("gray", enlarged), ("otsu", binary), ("bright", bright)
        ]
        if image.ndim == 3:
            # Filter white glyph fill before interpolation. Grayscale alone also
            # retains bright pink/yellow SEM background and can bury the text.
            white = cv2.inRange(image[:, :, :3], (190, 190, 190), (255, 255, 255))
            variants.append(("white_text", cv2.resize(
                white, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC
            )))
        for name, variant in variants:
            if float(np.median(variant)) < 127:
                variant = cv2.bitwise_not(variant)
            yield name, cv2.copyMakeBorder(
                variant, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255
            )

    def predict(
        self,
        image: str | Path | np.ndarray,
        *,
        roi: tuple[int, int, int, int] | None = None,
    ) -> FOVResult | None:
        """Return FOV with its unit, or None when no unambiguous FOV is read.

        Each preprocessing variant is read as a text block and as sparse text.
        Successful variants must agree on the value and any recognized unit.
        ``timeout`` is a per-Tesseract-call timeout in seconds.
        """
        region = self._load(image)
        if roi is not None:
            if len(roi) != 4 or any(type(v) is not int for v in roi):
                raise ValueError("roi must contain four integers: x, y, width, height")
            x, y, width, height = roi
            if (x < 0 or y < 0 or width <= 0 or height <= 0
                    or x + width > region.shape[1] or y + height > region.shape[0]):
                raise ValueError("roi must be a nonempty rectangle inside the image")
            region = region[y:y + height, x:x + width]
        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError("Install OCR dependencies: pip install -e '.[ocr]'") from exc

        results: list[FOVResult] = []
        for name, variant in self._variants(region):
            for psm in (6, 11):
                try:
                    text = pytesseract.image_to_string(
                        variant, lang="eng", config=f"--psm {psm}", timeout=self.timeout
                    )
                except pytesseract.TesseractNotFoundError as exc:
                    raise RuntimeError(
                        "Tesseract executable is missing. Install tesseract-ocr and "
                        "its English language data (see src/ocr_model/README.md)."
                    ) from exc
                result = parse_fov(text)
                if result is not None:
                    results.append(replace(result, preprocessing=f"{name}/psm{psm}"))
        if not results or len({r.value for r in results}) != 1:
            return None
        units = {r.unit for r in results if r.unit is not None}
        if len(units) > 1:
            return None
        return next((r for r in results if r.unit is not None), results[0])


# Keep existing callers working while exposing the baseline's explicit name.
FOVRecognizer = TesseractFOVBaseline
