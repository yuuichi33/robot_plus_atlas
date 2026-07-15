from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .constants import MODEL_INPUT_SHAPE, MODEL_OUTPUT_SHAPE


class AclError(RuntimeError):
    pass


def _check_ret(operation: str, ret: Any) -> None:
    if ret not in (None, 0):
        raise AclError(f"{operation} failed with ACL error {ret}")


def _value_and_ret(operation: str, result: Any) -> Any:
    if not isinstance(result, tuple) or len(result) != 2:
        raise AclError(f"{operation} returned an unexpected value: {result!r}")
    value, ret = result
    _check_ret(operation, ret)
    return value


def _created_value(operation: str, result: Any) -> Any:
    if isinstance(result, tuple) and len(result) == 2:
        return _value_and_ret(operation, result)
    if result is None:
        raise AclError(f"{operation} returned no resource")
    return result


class AclBackend:
    """Strict synchronous pyACL backend for the static YOLOv5 OM contract."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device_id: int = 0,
        acl_module: ModuleType | Any | None = None,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file() or self.model_path.stat().st_size == 0:
            raise FileNotFoundError(f"OM model not found or empty: {self.model_path}")
        if device_id < 0:
            raise ValueError("device_id must be non-negative")
        if acl_module is None:
            try:
                acl_module = importlib.import_module("acl")
            except ImportError as exc:
                raise AclError(
                    "pyACL module 'acl' is unavailable; source the matching CANN set_env.sh"
                ) from exc
        self._acl = acl_module
        self.device_id = int(device_id)
        self._input_shape = MODEL_INPUT_SHAPE
        self._output_shape = MODEL_OUTPUT_SHAPE
        self._input_dtype = np.dtype(np.float16)
        self._output_dtype = np.dtype(np.float32)
        self._input_size = 0
        self._output_size = 0

        self._acl_initialized = False
        self._device_set = False
        self._context: Any | None = None
        self._model_id: Any | None = None
        self._model_desc: Any | None = None
        self._input_device: Any | None = None
        self._output_device: Any | None = None
        self._output_host: Any | None = None
        self._input_dataset: Any | None = None
        self._output_dataset: Any | None = None
        self._input_data_buffer: Any | None = None
        self._output_data_buffer: Any | None = None
        self._closed = False
        try:
            self._open()
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return self._input_shape

    @property
    def input_dtype(self) -> np.dtype:
        return self._input_dtype

    @staticmethod
    def _dims(operation: str, result: Any) -> tuple[str, tuple[int, ...]]:
        payload = _value_and_ret(operation, result)
        if not isinstance(payload, dict) or "dims" not in payload:
            raise AclError(f"{operation} returned invalid dimensions: {payload!r}")
        dims = tuple(int(value) for value in payload["dims"])
        return str(payload.get("name", "")), dims

    def _numpy_dtype(self, code: Any, *, field: str) -> np.dtype:
        mapping = {
            getattr(self._acl, "ACL_FLOAT", 0): np.dtype(np.float32),
            getattr(self._acl, "ACL_FLOAT16", 1): np.dtype(np.float16),
        }
        if code not in mapping:
            raise AclError(f"unsupported {field} ACL dtype code: {code}")
        return mapping[code]

    def _open(self) -> None:
        acl = self._acl
        _check_ret("acl.init", acl.init())
        self._acl_initialized = True
        _check_ret("acl.rt.set_device", acl.rt.set_device(self.device_id))
        self._device_set = True
        self._context = _value_and_ret(
            "acl.rt.create_context", acl.rt.create_context(self.device_id)
        )
        self._model_id = _value_and_ret(
            "acl.mdl.load_from_file", acl.mdl.load_from_file(str(self.model_path))
        )
        self._model_desc = _created_value("acl.mdl.create_desc", acl.mdl.create_desc())
        _check_ret("acl.mdl.get_desc", acl.mdl.get_desc(self._model_desc, self._model_id))

        if acl.mdl.get_num_inputs(self._model_desc) != 1:
            raise AclError("OM model must have exactly one input")
        if acl.mdl.get_num_outputs(self._model_desc) != 1:
            raise AclError("OM model must have exactly one output")
        input_name, input_shape = self._dims(
            "acl.mdl.get_input_dims", acl.mdl.get_input_dims(self._model_desc, 0)
        )
        output_name, output_shape = self._dims(
            "acl.mdl.get_output_dims", acl.mdl.get_output_dims(self._model_desc, 0)
        )
        if input_name != "images" or input_shape != MODEL_INPUT_SHAPE:
            raise AclError(
                f"unexpected OM input contract: {input_name!r} {input_shape}, "
                f"expected 'images' {MODEL_INPUT_SHAPE}"
            )
        if output_name != "output0" or output_shape != MODEL_OUTPUT_SHAPE:
            raise AclError(
                f"unexpected OM output contract: {output_name!r} {output_shape}, "
                f"expected 'output0' {MODEL_OUTPUT_SHAPE}"
            )
        self._input_dtype = self._numpy_dtype(
            acl.mdl.get_input_data_type(self._model_desc, 0), field="input"
        )
        self._output_dtype = self._numpy_dtype(
            acl.mdl.get_output_data_type(self._model_desc, 0), field="output"
        )
        self._input_size = int(acl.mdl.get_input_size_by_index(self._model_desc, 0))
        self._output_size = int(acl.mdl.get_output_size_by_index(self._model_desc, 0))
        expected_input_size = int(np.prod(MODEL_INPUT_SHAPE)) * self._input_dtype.itemsize
        expected_output_size = int(np.prod(MODEL_OUTPUT_SHAPE)) * self._output_dtype.itemsize
        if self._input_size != expected_input_size:
            raise AclError(
                f"OM input byte size mismatch: {self._input_size}, expected {expected_input_size}"
            )
        if self._output_size != expected_output_size:
            raise AclError(
                f"OM output byte size mismatch: {self._output_size}, expected {expected_output_size}"
            )

        policy = getattr(acl, "ACL_MEM_MALLOC_HUGE_FIRST", 0)
        self._input_device = _value_and_ret(
            "acl.rt.malloc(input)", acl.rt.malloc(self._input_size, policy)
        )
        self._output_device = _value_and_ret(
            "acl.rt.malloc(output)", acl.rt.malloc(self._output_size, policy)
        )
        self._output_host = _value_and_ret(
            "acl.rt.malloc_host(output)", acl.rt.malloc_host(self._output_size)
        )

        self._input_dataset = _created_value(
            "acl.mdl.create_dataset(input)", acl.mdl.create_dataset()
        )
        self._output_dataset = _created_value(
            "acl.mdl.create_dataset(output)", acl.mdl.create_dataset()
        )
        self._input_data_buffer = _created_value(
            "acl.create_data_buffer(input)",
            acl.create_data_buffer(self._input_device, self._input_size),
        )
        self._output_data_buffer = _created_value(
            "acl.create_data_buffer(output)",
            acl.create_data_buffer(self._output_device, self._output_size),
        )
        self._input_dataset = _value_and_ret(
            "acl.mdl.add_dataset_buffer(input)",
            acl.mdl.add_dataset_buffer(self._input_dataset, self._input_data_buffer),
        )
        self._output_dataset = _value_and_ret(
            "acl.mdl.add_dataset_buffer(output)",
            acl.mdl.add_dataset_buffer(self._output_dataset, self._output_data_buffer),
        )

    def run(self, tensor: np.ndarray) -> np.ndarray:
        if self._closed or self._model_id is None:
            raise AclError("ACL backend is closed")
        value = np.asarray(tensor)
        if value.shape != self._input_shape or value.dtype != self._input_dtype:
            raise ValueError(
                f"input must be {self._input_shape}/{self._input_dtype}, "
                f"got {value.shape}/{value.dtype}"
            )
        if not value.flags.c_contiguous:
            raise ValueError("input tensor must be C-contiguous")
        if value.nbytes != self._input_size or not np.isfinite(value).all():
            raise ValueError("input tensor has an invalid byte size or contains NaN/Inf")

        acl = self._acl
        if hasattr(acl.util, "numpy_to_ptr"):
            input_ptr = acl.util.numpy_to_ptr(value)
            input_owner: Any = value
        elif hasattr(acl.util, "bytes_to_ptr"):
            input_owner = value.tobytes()
            input_ptr = acl.util.bytes_to_ptr(input_owner)
        else:
            raise AclError("acl.util exposes neither numpy_to_ptr nor bytes_to_ptr")
        _check_ret(
            "acl.rt.memcpy(H2D)",
            acl.rt.memcpy(
                self._input_device,
                self._input_size,
                input_ptr,
                self._input_size,
                getattr(acl, "ACL_MEMCPY_HOST_TO_DEVICE", 1),
            ),
        )
        del input_owner
        _check_ret(
            "acl.mdl.execute",
            acl.mdl.execute(self._model_id, self._input_dataset, self._output_dataset),
        )
        _check_ret(
            "acl.rt.memcpy(D2H)",
            acl.rt.memcpy(
                self._output_host,
                self._output_size,
                self._output_device,
                self._output_size,
                getattr(acl, "ACL_MEMCPY_DEVICE_TO_HOST", 2),
            ),
        )
        output_bytes = acl.util.ptr_to_bytes(self._output_host, self._output_size)
        if len(output_bytes) != self._output_size:
            raise AclError(
                f"ACL output copy returned {len(output_bytes)} bytes, expected {self._output_size}"
            )
        output = np.frombuffer(output_bytes, dtype=self._output_dtype).copy()
        output = output.reshape(self._output_shape)
        if not np.isfinite(output).all():
            raise AclError("ACL model output contains NaN/Inf")
        return output

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        acl = self._acl
        errors: list[str] = []

        def cleanup(name: str, func: Any, *args: Any) -> None:
            try:
                ret = func(*args)
                if ret not in (None, 0):
                    errors.append(f"{name}={ret}")
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        for attribute in ("_output_data_buffer", "_input_data_buffer"):
            resource = getattr(self, attribute)
            setattr(self, attribute, None)
            if resource is not None:
                cleanup("acl.destroy_data_buffer", acl.destroy_data_buffer, resource)
        for attribute in ("_output_dataset", "_input_dataset"):
            resource = getattr(self, attribute)
            setattr(self, attribute, None)
            if resource is not None:
                cleanup("acl.mdl.destroy_dataset", acl.mdl.destroy_dataset, resource)
        host = self._output_host
        self._output_host = None
        if host is not None:
            cleanup("acl.rt.free_host", acl.rt.free_host, host)
        for attribute in ("_output_device", "_input_device"):
            resource = getattr(self, attribute)
            setattr(self, attribute, None)
            if resource is not None:
                cleanup("acl.rt.free", acl.rt.free, resource)
        model_id = self._model_id
        self._model_id = None
        if model_id is not None:
            cleanup("acl.mdl.unload", acl.mdl.unload, model_id)
        desc = self._model_desc
        self._model_desc = None
        if desc is not None:
            cleanup("acl.mdl.destroy_desc", acl.mdl.destroy_desc, desc)
        context = self._context
        self._context = None
        if context is not None:
            cleanup("acl.rt.destroy_context", acl.rt.destroy_context, context)
        if self._device_set:
            self._device_set = False
            cleanup("acl.rt.reset_device", acl.rt.reset_device, self.device_id)
        if self._acl_initialized:
            self._acl_initialized = False
            cleanup("acl.finalize", acl.finalize)
        if errors:
            raise AclError("ACL cleanup failed: " + "; ".join(errors))

    def __enter__(self) -> "AclBackend":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
