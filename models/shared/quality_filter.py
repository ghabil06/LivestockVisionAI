"""
Image Quality and Rejection Filter.
Implements non-negotiable production quality screening to enforce:
'Unable to estimate reliably' for degraded, blurred, or occluded captures.
"""

from typing import Union, Tuple, Optional
import io
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

from LivestockVisionAI.data.schemas.capture_metadata import (
    QualityCheckResult,
    QualityStatus,
)


class QualityFilter:
    """Enforces rigorous quality gates on livestock image captures."""

    def __init__(
        self,
        min_laplacian_variance: float = 100.0,
        min_brightness_mean: float = 40.0,
        max_brightness_mean: float = 225.0,
        min_bbox_area_ratio: float = 0.15,
        max_bbox_area_ratio: float = 0.88,
    ):
        self.min_laplacian_variance = float(min_laplacian_variance)
        self.min_brightness_mean = float(min_brightness_mean)
        self.max_brightness_mean = float(max_brightness_mean)
        self.min_bbox_area_ratio = float(min_bbox_area_ratio)
        self.max_bbox_area_ratio = float(max_bbox_area_ratio)

    def _load_as_gray_and_rgb(
        self,
        image_input: Union[bytes, str, Path, np.ndarray, Image.Image],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert any supported image input into (grayscale_uint8, rgb_uint8)."""
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            bgr = cv2.imread(path_str)
            if bgr is None:
                raise ValueError(f"Unable to read image from path: {path_str}")
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return gray, rgb

        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
            rgb = np.array(pil_img)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            return gray, rgb

        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            return gray, rgb

        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                gray = image_input.astype(np.uint8)
                rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                return gray, rgb
            elif image_input.ndim == 3:
                # Assume RGB if 3 channels
                rgb = image_input.astype(np.uint8)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                return gray, rgb
            else:
                raise ValueError(f"Unsupported numpy array dimensions: {image_input.shape}")

        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def compute_laplacian_variance(self, gray: np.ndarray) -> float:
        """Compute the variance of the Laplacian filter as a focus/sharpness metric."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def compute_brightness_mean(self, gray: np.ndarray) -> float:
        """Compute average grayscale luminance across all pixels."""
        return float(np.mean(gray))

    def evaluate(
        self,
        image_input: Union[bytes, str, Path, np.ndarray, Image.Image],
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> QualityCheckResult:
        """
        Evaluate image against all quality thresholds.
        
        Args:
            image_input: The input image (bytes, path, PIL Image, or numpy array).
            bbox: Optional bounding box (x1, y1, x2, y2) of the detected animal.
            
        Returns:
            QualityCheckResult with pass/rejection status and metrics.
        """
        gray, rgb = self._load_as_gray_and_rgb(image_input)
        h, w = gray.shape[:2]

        laplacian_var = self.compute_laplacian_variance(gray)
        brightness = self.compute_brightness_mean(gray)

        # Check Bounding Box coverage if provided
        bbox_ratio: Optional[float] = None
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)
            bbox_area = float(bw * bh)
            total_area = float(w * h)
            bbox_ratio = bbox_area / total_area if total_area > 0 else 0.0

        # Quality Gate 1: Underexposure (Too Dark)
        if brightness < self.min_brightness_mean:
            return QualityCheckResult(
                passed=False,
                status=QualityStatus.REJECTED_UNDEREXPOSED,
                laplacian_variance=round(laplacian_var, 2),
                brightness_mean=round(brightness, 2),
                bbox_area_ratio=round(bbox_ratio, 3) if bbox_ratio is not None else None,
                rejection_reason=(
                    f"Unable to estimate reliably: Subject underexposed "
                    f"(mean brightness {brightness:.1f} < {self.min_brightness_mean:.1f})."
                ),
                recommendation="Move the animal into natural daylight or provide auxiliary lighting.",
            )

        # Quality Gate 2: Overexposure (Glare / Washed out)
        if brightness > self.max_brightness_mean:
            return QualityCheckResult(
                passed=False,
                status=QualityStatus.REJECTED_OVEREXPOSED,
                laplacian_variance=round(laplacian_var, 2),
                brightness_mean=round(brightness, 2),
                bbox_area_ratio=round(bbox_ratio, 3) if bbox_ratio is not None else None,
                rejection_reason=(
                    f"Unable to estimate reliably: Subject overexposed / harsh glare "
                    f"(mean brightness {brightness:.1f} > {self.max_brightness_mean:.1f})."
                ),
                recommendation="Avoid direct sunlight into the lens; position the animal under diffuse open shade.",
            )

        # Quality Gate 3: Animal scale / framing check
        if bbox_ratio is not None:
            if bbox_ratio < self.min_bbox_area_ratio:
                return QualityCheckResult(
                    passed=False,
                    status=QualityStatus.REJECTED_TOO_DISTANT,
                    laplacian_variance=round(laplacian_var, 2),
                    brightness_mean=round(brightness, 2),
                    bbox_area_ratio=round(bbox_ratio, 3),
                    rejection_reason=(
                        f"Unable to estimate reliably: Animal too distant "
                        f"(occupies {bbox_ratio * 100:.1f}% < {self.min_bbox_area_ratio * 100:.1f}% of frame)."
                    ),
                    recommendation="Step closer to the animal so it occupies at least 20% to 80% of the frame.",
                )
            if bbox_ratio > self.max_bbox_area_ratio:
                return QualityCheckResult(
                    passed=False,
                    status=QualityStatus.REJECTED_TOO_CLOSE,
                    laplacian_variance=round(laplacian_var, 2),
                    brightness_mean=round(brightness, 2),
                    bbox_area_ratio=round(bbox_ratio, 3),
                    rejection_reason=(
                        f"Unable to estimate reliably: Animal truncated or too close "
                        f"(occupies {bbox_ratio * 100:.1f}% > {self.max_bbox_area_ratio * 100:.1f}% of frame)."
                    ),
                    recommendation="Step back to ensure the entire animal (including hooves and withers) is in frame.",
                )

        # Quality Gate 4: Blur Check
        if laplacian_var < self.min_laplacian_variance:
            return QualityCheckResult(
                passed=False,
                status=QualityStatus.REJECTED_BLUR,
                laplacian_variance=round(laplacian_var, 2),
                brightness_mean=round(brightness, 2),
                bbox_area_ratio=round(bbox_ratio, 3) if bbox_ratio is not None else None,
                rejection_reason=(
                    f"Unable to estimate reliably: Image blur detected "
                    f"(variance {laplacian_var:.1f} < {self.min_laplacian_variance:.1f})."
                ),
                recommendation="Ensure the animal is stationary and hold the camera steady in good lighting.",
            )

        # Passed all quality checks
        return QualityCheckResult(
            passed=True,
            status=QualityStatus.PASSED,
            laplacian_variance=round(laplacian_var, 2),
            brightness_mean=round(brightness, 2),
            bbox_area_ratio=round(bbox_ratio, 3) if bbox_ratio is not None else None,
            rejection_reason=None,
            recommendation="Image passed all quality gates. Proceeding to inference.",
        )
