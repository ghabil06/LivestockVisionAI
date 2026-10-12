"""
Unit tests for Image Quality and Rejection Filter.
Verifies blur detection, exposure gates, and framing bounds.
"""

import numpy as np
import cv2
import pytest
from PIL import Image

from LivestockVisionAI.models.shared.quality_filter import QualityFilter
from LivestockVisionAI.data.schemas.capture_metadata import QualityStatus


@pytest.fixture
def filter_instance():
    return QualityFilter(
        min_laplacian_variance=100.0,
        min_brightness_mean=40.0,
        max_brightness_mean=225.0,
        min_bbox_area_ratio=0.15,
        max_bbox_area_ratio=0.88,
    )


def test_sharp_image_passes(filter_instance):
    """High-contrast 10px checkerboard should have high Laplacian variance."""
    board = ((np.indices((300, 300)).sum(axis=0) // 10) % 2) * 160 + 50
    board = board.astype(np.uint8)

    res = filter_instance.evaluate(board)
    assert res.passed
    assert res.status == QualityStatus.PASSED
    assert res.laplacian_variance > 100.0


def test_blurred_image_rejected(filter_instance):
    """Smoothed image with low variance should be rejected as REJECTED_BLUR."""
    # Flat with tiny gradient
    arr = np.ones((300, 300), dtype=np.uint8) * 128
    # Extremely blurred
    blurred = cv2.GaussianBlur(arr, (21, 21), 10.0)

    res = filter_instance.evaluate(blurred)
    assert not res.passed
    assert res.status == QualityStatus.REJECTED_BLUR
    assert "Unable to estimate reliably" in res.rejection_reason
    assert "blur detected" in res.rejection_reason.lower()


def test_underexposed_image_rejected(filter_instance):
    """Very dark image should be rejected as REJECTED_UNDEREXPOSED."""
    dark_arr = np.ones((200, 200), dtype=np.uint8) * 15  # Brightness 15 < 40

    res = filter_instance.evaluate(dark_arr)
    assert not res.passed
    assert res.status == QualityStatus.REJECTED_UNDEREXPOSED
    assert "underexposed" in res.rejection_reason.lower()


def test_overexposed_image_rejected(filter_instance):
    """Washed out image should be rejected as REJECTED_OVEREXPOSED."""
    bright_arr = np.ones((200, 200), dtype=np.uint8) * 245  # Brightness 245 > 225

    res = filter_instance.evaluate(bright_arr)
    assert not res.passed
    assert res.status == QualityStatus.REJECTED_OVEREXPOSED
    assert "overexposed" in res.rejection_reason.lower()


def test_bbox_framing_rejections(filter_instance):
    # Base sharp image using checkerboard
    board = ((np.indices((400, 400)).sum(axis=0) // 10) % 2) * 160 + 50
    board = board.astype(np.uint8)

    # 1. BBox occupying only 5% of area (too distant)
    small_bbox = (10, 10, 80, 80)  # Area 4900 / 160000 = ~3%
    res_small = filter_instance.evaluate(board, bbox=small_bbox)
    assert not res_small.passed
    assert res_small.status == QualityStatus.REJECTED_TOO_DISTANT

    # 2. BBox occupying 95% of area (truncated / too close)
    large_bbox = (5, 5, 395, 395)  # Area 152100 / 160000 = ~95%
    res_large = filter_instance.evaluate(board, bbox=large_bbox)
    assert not res_large.passed
    assert res_large.status == QualityStatus.REJECTED_TOO_CLOSE

    # 3. Proper BBox occupying ~50%
    good_bbox = (50, 50, 330, 330)  # Area 78400 / 160000 = ~49%
    res_good = filter_instance.evaluate(board, bbox=good_bbox)
    assert res_good.passed
    assert res_good.status == QualityStatus.PASSED
