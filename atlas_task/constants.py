from __future__ import annotations

from enum import IntEnum

CLASS_NAMES: tuple[str, ...] = ("NoTask", "Task1", "Task2", "Task3", "Task4")
MODEL_INPUT_SHAPE = (1, 3, 640, 640)
MODEL_OUTPUT_SHAPE = (1, 25200, 10)
LETTERBOX_COLOR = (114, 114, 114)
DEFAULT_CONFIDENCE = 0.60
DEFAULT_IOU = 0.45
MAX_DETECTIONS = 20


class RecognitionCode(IntEnum):
    NO_DETECTION = 0
    TASK1 = 1
    TASK2 = 2
    TASK3 = 3
    TASK4 = 4
    NO_TASK = 5
