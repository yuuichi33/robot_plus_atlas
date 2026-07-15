from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FrameSource(Protocol):
    def open(self) -> None: ...

    def read(self, timeout_s: float = 2.0) -> np.ndarray | None: ...

    def close(self) -> None: ...

    @property
    def description(self) -> str: ...


@runtime_checkable
class InferenceBackend(Protocol):
    @property
    def input_shape(self) -> tuple[int, int, int, int]: ...

    @property
    def input_dtype(self) -> np.dtype: ...

    def run(self, tensor: np.ndarray) -> np.ndarray: ...

    def close(self) -> None: ...
