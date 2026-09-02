#!/usr/bin/env python3
"""Depth edge filter to remove flying pixels at depth discontinuities.

For each valid pixel, if the depth difference from its local neighborhood
min or max exceeds depth_threshold * depth, it is masked out.
"""

from __future__ import annotations

import time

import cv2
import numpy as np


def filter_depth_edges(
    depth: np.ndarray,
    kernel_size: int = 5,
    depth_threshold: float = 0.03,
) -> tuple[np.ndarray, float]:
    """Remove flying pixels at depth edges using local consistency check.

    Args:
        depth: (H, W) float32 depth image in meters, 0 = invalid.
        kernel_size: Morphological kernel size for neighborhood search.
        depth_threshold: Relative threshold — a pixel whose depth differs
                         from the local min/max by more than this fraction
                         of its own depth is masked out.

    Returns:
        (filtered_depth, elapsed_seconds)
    """
    t0 = time.perf_counter()
    valid = depth > 0
    if not np.any(valid):
        return depth.copy(), time.perf_counter() - t0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    min_depth = cv2.erode(depth, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1e9)
    max_depth = cv2.dilate(depth, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    relative_threshold = depth_threshold * depth
    edge_mask = (depth - min_depth > relative_threshold) | (max_depth - depth > relative_threshold)

    filtered = depth.copy()
    filtered[edge_mask] = 0.0
    elapsed = time.perf_counter() - t0
    return filtered, elapsed
