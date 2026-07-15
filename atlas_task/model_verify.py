from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_sidecar(model_path: str | Path) -> str:
    model = Path(model_path).resolve()
    if not model.is_file() or model.stat().st_size == 0:
        raise FileNotFoundError(f"OM model not found or empty: {model}")
    sidecar = Path(f"{model}.sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"OM SHA-256 sidecar is required: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if not fields or len(fields[0]) != 64:
        raise ValueError(f"invalid SHA-256 sidecar: {sidecar}")
    expected = fields[0].lower()
    try:
        int(expected, 16)
    except ValueError as exc:
        raise ValueError(f"invalid SHA-256 value in {sidecar}") from exc
    actual = sha256_file(model)
    if actual != expected:
        raise ValueError(
            f"OM SHA-256 mismatch: expected {expected}, got {actual} ({model})"
        )
    return actual
