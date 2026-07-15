from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    timestamp: float


@dataclass(frozen=True)
class LetterboxInfo:
    original_shape: tuple[int, int]
    input_shape: tuple[int, int]
    ratio: tuple[float, float]
    pad: tuple[float, float]
