from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np


class CameraError(RuntimeError):
    pass


def parse_camera(value: str | int) -> str | int:
    if isinstance(value, int):
        return value
    stripped = value.strip()
    if not stripped:
        raise ValueError("camera must not be empty")
    if stripped.isdecimal():
        return int(stripped)
    return stripped


def _default_capture_factory(source: str | int) -> Any:
    import cv2

    if hasattr(cv2, "CAP_V4L2"):
        return cv2.VideoCapture(source, cv2.CAP_V4L2)
    return cv2.VideoCapture(source)


class OpenCVCameraSource:
    """One-frame V4L2 source compatible with integer IDs and stable device paths."""

    def __init__(
        self,
        camera: str | int = "/dev/video0",
        *,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        open_retries: int = 10,
        retry_delay_s: float = 2.0,
        read_retry_delay_s: float = 0.05,
        capture_factory: Callable[[str | int], Any] = _default_capture_factory,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("width, height and fps must be positive")
        if open_retries < 1:
            raise ValueError("open_retries must be at least 1")
        self.camera = parse_camera(camera)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.open_retries = int(open_retries)
        self.retry_delay_s = max(0.0, float(retry_delay_s))
        self.read_retry_delay_s = max(0.0, float(read_retry_delay_s))
        self.capture_factory = capture_factory
        self.clock = clock
        self.sleeper = sleeper
        self._capture: Any | None = None
        self._actual_settings: dict[str, float] = {}

    @property
    def description(self) -> str:
        return f"camera:{self.camera}"

    @property
    def actual_settings(self) -> dict[str, float]:
        return dict(self._actual_settings)

    @staticmethod
    def _release(capture: Any | None) -> None:
        if capture is None:
            return
        try:
            capture.release()
        except Exception:
            pass

    def open(self) -> None:
        if self._capture is not None:
            return
        import cv2

        last_error = "device did not open"
        for attempt in range(self.open_retries):
            capture = None
            try:
                capture = self.capture_factory(self.camera)
                if capture is not None and capture.isOpened():
                    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    capture.set(cv2.CAP_PROP_FPS, self.fps)
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
                    self._actual_settings = {
                        "width": float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
                    }
                    self._capture = capture
                    return
                last_error = "device did not open"
            except Exception as exc:
                last_error = str(exc)
            self._release(capture)
            if attempt + 1 < self.open_retries:
                self.sleeper(self.retry_delay_s)
        raise CameraError(
            f"unable to open {self.description} after {self.open_retries} attempts: {last_error}"
        )

    def read(self, timeout_s: float = 2.0) -> np.ndarray | None:
        if self._capture is None:
            raise CameraError("camera source is not open")
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        deadline = self.clock() + timeout_s
        while True:
            try:
                ok, frame = self._capture.read()
            except Exception as exc:
                raise CameraError(f"failed to read {self.description}: {exc}") from exc
            if ok and isinstance(frame, np.ndarray) and frame.size > 0:
                return frame
            if self.clock() >= deadline:
                return None
            self.sleeper(self.read_retry_delay_s)

    def close(self) -> None:
        capture = self._capture
        self._capture = None
        self._release(capture)

    def __enter__(self) -> "OpenCVCameraSource":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
