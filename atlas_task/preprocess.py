from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .constants import LETTERBOX_COLOR
from .types import LetterboxInfo


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR image, got shape {image.shape}")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image must be non-empty")


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = LETTERBOX_COLOR,
    *,
    scale_up: bool = True,
) -> tuple[np.ndarray, LetterboxInfo]:
    _validate_image(image)
    if len(new_shape) != 2 or min(new_shape) <= 0:
        raise ValueError(f"invalid target shape: {new_shape}")

    import cv2

    original_h, original_w = image.shape[:2]
    target_h, target_w = int(new_shape[0]), int(new_shape[1])
    ratio = min(target_h / original_h, target_w / original_w)
    if not scale_up:
        ratio = min(ratio, 1.0)
    resized_w = int(round(original_w * ratio))
    resized_h = int(round(original_h * ratio))
    pad_w = (target_w - resized_w) / 2.0
    pad_h = (target_h - resized_h) / 2.0

    if (resized_w, resized_h) != (original_w, original_h):
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    if image.shape[:2] != (target_h, target_w):
        raise RuntimeError(
            f"letterbox produced {image.shape[:2]}, expected {(target_h, target_w)}"
        )
    return image, LetterboxInfo(
        original_shape=(original_h, original_w),
        input_shape=(target_h, target_w),
        ratio=(ratio, ratio),
        pad=(pad_w, pad_h),
    )


def preprocess_bgr(
    frame: np.ndarray,
    input_shape: tuple[int, int] = (640, 640),
    *,
    dtype: np.dtype | type[np.floating] = np.float16,
) -> tuple[np.ndarray, LetterboxInfo]:
    output_dtype = np.dtype(dtype)
    if output_dtype not in {np.dtype(np.float16), np.dtype(np.float32)}:
        raise ValueError(f"only float16/float32 model input is supported, got {output_dtype}")
    padded, info = letterbox(frame, input_shape)
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))
    tensor = tensor.astype(output_dtype, copy=False) / np.asarray(255.0, dtype=output_dtype)
    return np.ascontiguousarray(tensor[None, ...]), info


def restore_boxes(
    boxes: np.ndarray | Sequence[Sequence[float]], info: LetterboxInfo
) -> np.ndarray:
    result = np.asarray(boxes, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != 4:
        raise ValueError(f"boxes must have shape Nx4, got {result.shape}")
    if result.size == 0:
        return result
    if info.ratio[0] <= 0.0 or info.ratio[1] <= 0.0:
        raise ValueError("letterbox ratios must be positive")
    result[:, [0, 2]] = (result[:, [0, 2]] - info.pad[0]) / info.ratio[0]
    result[:, [1, 3]] = (result[:, [1, 3]] - info.pad[1]) / info.ratio[1]
    original_h, original_w = info.original_shape
    result[:, [0, 2]] = result[:, [0, 2]].clip(0, original_w)
    result[:, [1, 3]] = result[:, [1, 3]].clip(0, original_h)
    return result
