from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from navclaw.mapping.exploration.bev_visuals import (
    BEV_FRONTIER_FONT_SIZE_PX,
    BEV_FRONTIER_PADDING_X_PX,
    BEV_FRONTIER_RECT_HEIGHT_PX,
    clamp_numbered_circle_marker_center,
    draw_frontier_marker,
    draw_numbered_circle_marker,
    load_visual_font,
    marker_bbox,
    numbered_circle_marker_bbox,
    numbered_circle_marker_style,
    place_node_marker_style,
)


RGB_TRAJECTORY_COLOR = (255, 0, 0)
RGB_TRAJECTORY_WIDTH = 6
RGB_FRONTIER_BADGE_RECT_HEIGHT_PX = BEV_FRONTIER_RECT_HEIGHT_PX * 2
RGB_FRONTIER_BADGE_PADDING_X_PX = BEV_FRONTIER_PADDING_X_PX * 2
RGB_FRONTIER_BADGE_FONT_SIZE_PX = BEV_FRONTIER_FONT_SIZE_PX * 2
RGB_FRONTIER_BADGE_RECT_HEIGHT_IMAGE_FRACTION = 0.035
RGB_FRONTIER_BADGE_RECT_HEIGHT_MIN_PX = 14
RGB_FRONTIER_BADGE_RECT_HEIGHT_MAX_PX = 22
RGB_FRONTIER_BADGE_FONT_SIZE_MIN_PX = 10
RGB_FRONTIER_BADGE_FONT_SIZE_MAX_PX = 14
RGB_FRONTIER_BADGE_PADDING_X_MIN_PX = 3
RGB_FRONTIER_BADGE_PADDING_X_MAX_PX = 5
RGB_FRONTIER_BADGE_BACKOFF_M = 0.05
RGB_PLACE_NODE_CIRCLE_FILL = (20, 100, 255)
RGB_PLACE_NODE_CIRCLE_OUTLINE = (255, 255, 255)
RGB_PLACE_NODE_CIRCLE_LABEL = (255, 255, 255)
RGB_PLACE_NODE_CIRCLE_RADIUS_PX = 15
RGB_FRONTIER_CIRCLE_FILL = (255, 190, 30)
RGB_FRONTIER_CIRCLE_OUTLINE = (255, 255, 255)
RGB_FRONTIER_CIRCLE_LABEL = (0, 0, 0)
RGB_FRONTIER_CIRCLE_LABEL_STROKE = (255, 255, 255)
RGB_FRONTIER_CIRCLE_RADIUS_IMAGE_FRACTION = 0.011
RGB_FRONTIER_CIRCLE_RADIUS_MIN_PX = 4
RGB_FRONTIER_CIRCLE_RADIUS_MAX_PX = 7
RGB_FRONTIER_CIRCLE_FONT_SIZE_IMAGE_FRACTION = 0.026
RGB_FRONTIER_CIRCLE_FONT_SIZE_MIN_PX = 10
RGB_FRONTIER_CIRCLE_FONT_SIZE_MAX_PX = 15
FRONTIER_BADGE_FONT = ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    size=RGB_FRONTIER_BADGE_FONT_SIZE_PX,
)
FRONTIER_SCORE_FONT = ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    size=18,
)
ROOM_CATEGORY_FONT = ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    size=18,
)
BEV_SELECTION_CROP_MARGIN_PX = 120


def _frontier_label_text(frontier_id: str) -> str:
    if frontier_id.startswith("f") and frontier_id[1:].isdigit():
        return frontier_id[1:]
    return frontier_id


def _clamp_int(value: float, min_value: int, max_value: int) -> int:
    return max(int(min_value), min(int(max_value), int(round(float(value)))))


@lru_cache(maxsize=8)
def _frontier_badge_font(size_px: int) -> ImageFont.ImageFont:
    return ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        size=int(size_px),
    )


def _rgb_frontier_badge_style(
    image_width: int,
    image_height: int,
) -> tuple[ImageFont.ImageFont, int, int]:
    base_px = max(1, min(int(image_width), int(image_height)))
    rect_height_px = _clamp_int(
        float(base_px) * RGB_FRONTIER_BADGE_RECT_HEIGHT_IMAGE_FRACTION,
        RGB_FRONTIER_BADGE_RECT_HEIGHT_MIN_PX,
        RGB_FRONTIER_BADGE_RECT_HEIGHT_MAX_PX,
    )
    font_size_px = _clamp_int(
        float(rect_height_px) * 0.65,
        RGB_FRONTIER_BADGE_FONT_SIZE_MIN_PX,
        RGB_FRONTIER_BADGE_FONT_SIZE_MAX_PX,
    )
    padding_x_px = _clamp_int(
        float(rect_height_px) * 0.18,
        RGB_FRONTIER_BADGE_PADDING_X_MIN_PX,
        RGB_FRONTIER_BADGE_PADDING_X_MAX_PX,
    )
    return _frontier_badge_font(font_size_px), rect_height_px, padding_x_px


def _clamp_frontier_badge_center(
    *,
    draw: ImageDraw.ImageDraw,
    center_xy: tuple[float, float],
    label: str,
    font: ImageFont.ImageFont,
    rect_height_px: int,
    padding_x_px: int,
    image_size: tuple[int, int],
) -> tuple[float, float]:
    bbox = marker_bbox(
        draw=draw,
        center_xy=center_xy,
        label=str(label),
        font=font,
        rect_height_px=int(rect_height_px),
        padding_x_px=int(padding_x_px),
    )
    dx = 0.0
    dy = 0.0
    if bbox.left < 0.0:
        dx = -float(bbox.left)
    elif bbox.right >= float(image_size[0]):
        dx = float(image_size[0] - 1) - float(bbox.right)
    if bbox.top < 0.0:
        dy = -float(bbox.top)
    elif bbox.bottom >= float(image_size[1]):
        dy = float(image_size[1] - 1) - float(bbox.bottom)
    return (float(center_xy[0]) + dx, float(center_xy[1]) + dy)


def _draw_frontier_badge(
    draw: ImageDraw.ImageDraw,
    center_xy: tuple[float, float],
    frontier_id: str,
    label_text: str | None = None,
    font: ImageFont.ImageFont | None = None,
    rect_height_px: int = RGB_FRONTIER_BADGE_RECT_HEIGHT_PX,
    padding_x_px: int = RGB_FRONTIER_BADGE_PADDING_X_PX,
    image: Image.Image | None = None,
    mode: str = "bev",
) -> None:
    label = _frontier_label_text(frontier_id) if label_text is None else str(label_text)
    if image is not None:
        draw_numbered_circle_marker(
            image,
            center_xy,
            label,
            style=numbered_circle_marker_style(
                image_width=int(image.size[0]),
                image_height=int(image.size[1]),
                mode=str(mode),
            ),
        )
        return
    draw_frontier_marker(
        draw,
        center_xy,
        font=FRONTIER_BADGE_FONT if font is None else font,
        label=label,
        rect_height_px=int(rect_height_px),
        padding_x_px=int(padding_x_px),
    )


def _rgb_frontier_circle_style(
    image_width: int,
    image_height: int,
) -> tuple[ImageFont.ImageFont, int]:
    style = numbered_circle_marker_style(
        image_width=int(image_width),
        image_height=int(image_height),
        mode="rgb",
    )
    return load_visual_font(style.font_size_px), int(style.radius_px)


def _frontier_circle_marker_bbox(
    *,
    draw: ImageDraw.ImageDraw,
    center_xy: tuple[float, float],
    label: str,
    font: ImageFont.ImageFont,
    radius_px: int,
) -> tuple[float, float, float, float]:
    style = numbered_circle_marker_style(
        image_width=1,
        image_height=1,
        mode="rgb",
    )
    style = style.__class__(
        radius_px=int(radius_px),
        font_size_px=getattr(font, "size", style.font_size_px),
        fill=style.fill,
        outline=style.outline,
        text_fill=style.text_fill,
        outline_width_px=style.outline_width_px,
        supersample=style.supersample,
    )
    bbox = numbered_circle_marker_bbox(center_xy=center_xy, style=style)
    return bbox.as_tuple()


def _clamp_frontier_circle_marker_center(
    *,
    draw: ImageDraw.ImageDraw,
    center_xy: tuple[float, float],
    label: str,
    font: ImageFont.ImageFont,
    radius_px: int,
    image_size: tuple[int, int],
) -> tuple[float, float]:
    style = numbered_circle_marker_style(
        image_width=int(image_size[0]),
        image_height=int(image_size[1]),
        mode="rgb",
    )
    return clamp_numbered_circle_marker_center(
        center_xy=center_xy,
        style=style,
        image_size=image_size,
    )


def _draw_frontier_circle_marker(
    *,
    draw: ImageDraw.ImageDraw,
    center_xy: tuple[float, float],
    label: str,
    font: ImageFont.ImageFont,
    radius_px: int,
    image: Image.Image | None = None,
) -> None:
    if image is not None:
        draw_numbered_circle_marker(
            image,
            center_xy,
            str(label),
            style=numbered_circle_marker_style(
                image_width=int(image.size[0]),
                image_height=int(image.size[1]),
                mode="rgb",
            ),
        )
        return
    radius = float(radius_px)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=(255, 255, 255, 145),
        outline=(35, 35, 35, 72),
        width=1,
    )
    draw.text(
        (cx, cy),
        str(label),
        fill=(18, 18, 18, 245),
        font=font,
        anchor="mm",
    )


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    fill: tuple[int, int, int],
    width: int,
    dash_length: float = 12.0,
    gap_length: float = 8.0,
) -> None:
    x0, y0 = start_xy
    x1, y1 = end_xy
    total_length = float(np.hypot(x1 - x0, y1 - y0))
    if total_length <= 1e-6:
        return
    direction_x = (x1 - x0) / total_length
    direction_y = (y1 - y0) / total_length
    cursor = 0.0
    while cursor < total_length:
        dash_end = min(total_length, cursor + float(dash_length))
        segment_start = (
            x0 + direction_x * cursor,
            y0 + direction_y * cursor,
        )
        segment_end = (
            x0 + direction_x * dash_end,
            y0 + direction_y * dash_end,
        )
        draw.line([segment_start, segment_end], fill=fill, width=int(width))
        cursor += float(dash_length) + float(gap_length)


def _draw_dashed_rectangle_outline(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[float, float, float, float],
    fill: tuple[int, int, int],
    width: int,
) -> None:
    left, top, right, bottom = bbox
    _draw_dashed_line(draw, (left, top), (right, top), fill=fill, width=width)
    _draw_dashed_line(draw, (right, top), (right, bottom), fill=fill, width=width)
    _draw_dashed_line(draw, (right, bottom), (left, bottom), fill=fill, width=width)
    _draw_dashed_line(draw, (left, bottom), (left, top), fill=fill, width=width)


def _crop_bev_overlay(
    image: np.ndarray,
    explored_mask: np.ndarray | None,
    extra_points_px: list[tuple[float, float]] | None = None,
    margin_px: int = BEV_SELECTION_CROP_MARGIN_PX,
) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    crop_points: list[np.ndarray] = []
    if explored_mask is not None:
        explored = np.asarray(explored_mask, dtype=bool)
        if explored.shape != (image_height, image_width):
            raise ValueError("crop explored_mask shape must match image shape")
        if np.any(explored):
            explored_y, explored_x = np.nonzero(explored)
            crop_points.append(np.stack([explored_x, explored_y], axis=1).astype(np.float64))
    if extra_points_px is not None and extra_points_px != []:
        crop_points.append(np.asarray(extra_points_px, dtype=np.float64).reshape(-1, 2))
    if crop_points == []:
        return image
    stacked = np.concatenate(crop_points, axis=0)
    min_x = max(0, int(np.floor(np.min(stacked[:, 0]))) - int(margin_px))
    max_x = min(image.shape[1], int(np.ceil(np.max(stacked[:, 0]))) + int(margin_px) + 1)
    min_y = max(0, int(np.floor(np.min(stacked[:, 1]))) - int(margin_px))
    max_y = min(image.shape[0], int(np.ceil(np.max(stacked[:, 1]))) + int(margin_px) + 1)
    return image[min_y:max_y, min_x:max_x]


def _draw_place_node_circle(
    *,
    draw: ImageDraw.ImageDraw,
    center_xy: tuple[float, float],
    label: str,
    font: ImageFont.ImageFont,
    image_size: tuple[int, int],
    image: Image.Image | None = None,
) -> None:
    display_label = _place_node_display_label(label)
    if image is not None:
        draw_numbered_circle_marker(
            image,
            center_xy,
            display_label,
            style=place_node_marker_style(
                image_width=int(image.size[0]),
                image_height=int(image.size[1]),
                mode="rgb",
            ),
        )
        return
    radius = int(RGB_PLACE_NODE_CIRCLE_RADIUS_PX)
    x = int(round(float(center_xy[0])))
    y = int(round(float(center_xy[1])))
    x = max(radius, min(int(image_size[0]) - radius - 1, x))
    y = max(radius, min(int(image_size[1]) - radius - 1, y))
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=RGB_PLACE_NODE_CIRCLE_FILL,
        outline=RGB_PLACE_NODE_CIRCLE_OUTLINE,
        width=2,
    )
    bbox = draw.textbbox((0, 0), display_label, font=font)
    text_x = x - (float(bbox[0]) + float(bbox[2])) / 2.0
    text_y = y - (float(bbox[1]) + float(bbox[3])) / 2.0
    draw.text((text_x, text_y), display_label, fill=RGB_PLACE_NODE_CIRCLE_LABEL, font=font)


def _place_node_display_label(label: object) -> str:
    text = str(label).strip()
    if len(text) >= 2 and text[0].lower() == "n" and text[1:].isdigit():
        return text[1:]
    return text
