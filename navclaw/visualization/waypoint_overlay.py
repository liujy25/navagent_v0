from __future__ import annotations

from typing import Any

import numpy as np

from navclaw.visualization.rendering import _normalize_rgb


def normalized_point_to_pixel(
    point_2d: tuple[float, float] | list[float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float]:
    if int(image_width) <= 0 or int(image_height) <= 0:
        raise ValueError(f"image size must be positive, got {image_width}x{image_height}")
    if len(point_2d) != 2:
        raise ValueError(f"point_2d must have length 2, got {point_2d!r}")
    x = float(point_2d[0])
    y = float(point_2d[1])
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError(f"point_2d must be in [0,1], got {point_2d!r}")
    return (
        float(x * float(max(0, int(image_width) - 1))),
        float(y * float(max(0, int(image_height) - 1))),
    )


def draw_waypoint_overlay_rgb(
    image: Any,
    *,
    point_pixel: tuple[float, float],
    label: str = "",
) -> np.ndarray:
    _ = label
    canvas = _normalize_rgb(image).copy()
    height, width = int(canvas.shape[0]), int(canvas.shape[1])
    cx = int(round(float(point_pixel[0])))
    cy = int(round(float(point_pixel[1])))
    cx = max(0, min(width - 1, cx))
    cy = max(0, min(height - 1, cy))
    radius = max(5, int(round(float(min(width, height)) * 0.018)))
    outline_radius = radius + 2
    yy, xx = np.ogrid[:height, :width]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    canvas[dist2 <= outline_radius**2] = np.array([0, 0, 0], dtype=np.uint8)
    canvas[dist2 <= radius**2] = np.array([255, 0, 0], dtype=np.uint8)
    return np.asarray(canvas, dtype=np.uint8)
