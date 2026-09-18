from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

DEPTH_UINT16_MM_MAX_METERS = 65.535
RGB_JPEG_QUALITY = 75


def encode_rgb_image(rgb: np.ndarray) -> str:
    array = np.asarray(rgb, dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=RGB_JPEG_QUALITY)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_rgb_image(payload: str) -> np.ndarray:
    raw = base64.b64decode(payload.encode("utf-8"))
    image = Image.open(BytesIO(raw)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def encode_array(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array)
    buffer = BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return {
        "encoding": "npy_base64",
        "data": base64.b64encode(buffer.getvalue()).decode("utf-8"),
    }


def decode_array(payload: dict[str, Any]) -> np.ndarray:
    if payload.get("encoding") != "npy_base64":
        raise ValueError(f"Unsupported array encoding: {payload.get('encoding')}")
    raw = base64.b64decode(payload["data"].encode("utf-8"))
    return np.load(BytesIO(raw), allow_pickle=False)


def encode_depth_meters(depth: np.ndarray) -> dict[str, Any]:
    depth_m = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth_m)
    clipped = np.where(
        finite,
        np.clip(depth_m, 0.0, DEPTH_UINT16_MM_MAX_METERS),
        0.0,
    )
    depth_mm = np.rint(clipped * 1000.0).astype(np.uint16)
    payload = encode_array(depth_mm)
    payload["encoding"] = "depth_uint16_mm_npy_base64"
    payload["invalid_value_mm"] = 0
    payload["unit"] = "millimeter"
    return payload


def decode_depth_meters(payload: dict[str, Any]) -> np.ndarray:
    encoding = payload.get("encoding")
    if encoding == "depth_uint16_mm_npy_base64":
        raw_payload = dict(payload)
        raw_payload["encoding"] = "npy_base64"
        depth_mm = decode_array(raw_payload).astype(np.float32)
        return depth_mm / 1000.0
    return decode_array(payload).astype(np.float32)
