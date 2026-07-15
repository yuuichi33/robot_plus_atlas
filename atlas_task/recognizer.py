from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np

from .constants import (
    CLASS_NAMES,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    MAX_DETECTIONS,
    MODEL_INPUT_SHAPE,
    RecognitionCode,
)
from .interfaces import FrameSource, InferenceBackend
from .postprocess import postprocess_yolov5
from .preprocess import preprocess_bgr
from .types import Detection


def select_result_code(detections: Sequence[Detection]) -> int:
    """Map one frame of detections to the fixed 0..5 peripheral ABI."""
    task_detections = [detection for detection in detections if 1 <= detection.class_id <= 4]
    if task_detections:
        winner = min(task_detections, key=lambda item: (-item.confidence, item.class_id))
        return winner.class_id
    if any(detection.class_id == 0 for detection in detections):
        return int(RecognitionCode.NO_TASK)
    return int(RecognitionCode.NO_DETECTION)


class AtlasTaskRecognizer:
    def __init__(
        self,
        backend: InferenceBackend,
        *,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU,
        max_det: int = MAX_DETECTIONS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if tuple(backend.input_shape) != MODEL_INPUT_SHAPE:
            raise ValueError(f"unsupported backend input shape: {backend.input_shape}")
        if np.dtype(backend.input_dtype) not in {np.dtype(np.float16), np.dtype(np.float32)}:
            raise ValueError(f"unsupported backend input dtype: {backend.input_dtype}")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        self.backend = backend
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_det = int(max_det)
        self.clock = clock

    def detect(self, frame: np.ndarray) -> list[Detection]:
        tensor, info = preprocess_bgr(
            frame,
            (self.backend.input_shape[2], self.backend.input_shape[3]),
            dtype=self.backend.input_dtype,
        )
        prediction = self.backend.run(tensor)
        return postprocess_yolov5(
            prediction,
            info,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            max_det=self.max_det,
            timestamp=self.clock(),
        )

    def recognize_frame(self, frame: np.ndarray) -> int:
        return select_result_code(self.detect(frame))

    def recognize_once(self, source: FrameSource, *, timeout_s: float = 2.0) -> int:
        try:
            source.open()
            frame = source.read(timeout_s=timeout_s)
            if frame is None:
                raise RuntimeError(f"timed out reading one frame from {source.description}")
            return self.recognize_frame(frame)
        finally:
            source.close()

    def close(self) -> None:
        self.backend.close()

    def __enter__(self) -> "AtlasTaskRecognizer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
