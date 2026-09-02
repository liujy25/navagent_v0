from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BEV_UNKNOWN_BACKGROUND_COLOR = (92, 92, 92)
BEV_EXPLORED_FREE_COLOR = (190, 248, 200)
BEV_DILATED_OBSTACLE_COLOR = (220, 38, 38)
BEV_OBSTACLE_COLOR = (220, 38, 38)

BEV_PLACE_NODE_FILL = (26, 112, 220, 255)
BEV_PLACE_NODE_TEXT = (255, 255, 255, 255)
PLACE_NODE_MARKER_FILL = (26, 112, 220, 230)
PLACE_NODE_MARKER_OUTLINE = (255, 255, 255, 255)
PLACE_NODE_MARKER_TEXT = (255, 255, 255, 255)
BEV_FRONTIER_MARKER_FILL = (0, 0, 0, 255)
BEV_FRONTIER_MARKER_TEXT = (255, 255, 255, 255)
BEV_GRAPH_EDGE_COLOR = (20, 20, 20, 210)
BEV_DIRECTION_BORDER_COLOR = (198, 198, 198)
BEV_DIRECTION_BORDER_TEXT_COLOR = (20, 20, 20)
BEV_DIRECTION_BORDER_LINE_COLOR = (110, 110, 110)

BEV_VIEW_SCALE = 2.0
BEV_PLACE_NODE_RADIUS_PX = 10
BEV_PLACE_NODE_FONT_SIZE_PX = 14
BEV_FRONTIER_RECT_HEIGHT_PX = 18
BEV_FRONTIER_FONT_SIZE_PX = 14
BEV_FRONTIER_PADDING_X_PX = 4
BEV_OBS_FRONTIER_ICON_SIZE_PX = 30
BEV_OBS_LABEL_RECT_HEIGHT_PX = 28
BEV_OBS_LABEL_FONT_SIZE_PX = 20
BEV_OBS_LABEL_PADDING_X_PX = 5
BEV_LANDMARK_MARKER_SIZE_PX = 50
BEV_ROBOT_MARKER_SIZE_M = 0.5
BEV_ROBOT_MARKER_MIN_SIZE_PX = 18
BEV_ROBOT_MARKER_MAX_SIZE_PX = 26
BEV_DIRECTION_BORDER_WIDTH_PX = 30
BEV_DIRECTION_BORDER_FONT_SIZE_PX = 20
BEV_MAP_DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
_VECTOR_SECTOR_DIRECTIONS = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")


def load_visual_font(size_px: int) -> ImageFont.ImageFont:
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, size=int(size_px))
        except OSError:
            continue
    return ImageFont.load_default()


def map_direction_label_from_vector_xy(vector_xy: object) -> str:
    """Convert a map-frame XY direction vector to one of N/NE/E/SE/S/SW/W/NW."""
    try:
        values = list(vector_xy)
        vx = float(values[0])
        vy = float(values[1])
    except (TypeError, ValueError, IndexError):
        return ""
    if math.hypot(vx, vy) <= 1e-6:
        return ""
    angle_deg = math.degrees(math.atan2(vy, vx))
    sector = int(round(angle_deg / 45.0)) % 8
    return _VECTOR_SECTOR_DIRECTIONS[sector]


def add_direction_border(
    image: Image.Image | np.ndarray,
    *,
    cardinal_only: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Add a fixed gray BEV direction border and return image plus metadata."""
    pil_image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image, dtype=np.uint8))
    if pil_image.mode not in {"RGB", "RGBA"}:
        pil_image = pil_image.convert("RGB")
    width, height = int(pil_image.size[0]), int(pil_image.size[1])
    border = int(BEV_DIRECTION_BORDER_WIDTH_PX)
    mode = pil_image.mode
    border_fill = _with_alpha(BEV_DIRECTION_BORDER_COLOR, mode)
    text_fill = _with_alpha(BEV_DIRECTION_BORDER_TEXT_COLOR, mode)
    line_fill = _with_alpha(BEV_DIRECTION_BORDER_LINE_COLOR, mode)
    canvas = Image.new(mode, (width + border * 2, height + border * 2), border_fill)
    canvas.paste(pil_image, (border, border))
    draw = ImageDraw.Draw(canvas)
    font = load_visual_font(BEV_DIRECTION_BORDER_FONT_SIZE_PX)
    label_positions = {
        "N": (border + width / 2.0, border / 2.0),
        "NE": (border + width + border / 2.0, border / 2.0),
        "E": (border + width + border / 2.0, border + height / 2.0),
        "SE": (border + width + border / 2.0, border + height + border / 2.0),
        "S": (border + width / 2.0, border + height + border / 2.0),
        "SW": (border / 2.0, border + height + border / 2.0),
        "W": (border / 2.0, border + height / 2.0),
        "NW": (border / 2.0, border / 2.0),
    }
    directions = ("N", "E", "S", "W") if bool(cardinal_only) else BEV_MAP_DIRECTIONS
    for label, position in label_positions.items():
        if label not in directions:
            continue
        draw.text(position, label, fill=text_fill, font=font, anchor="mm")
    draw.rectangle(
        (border - 1, border - 1, border + width, border + height),
        outline=line_fill,
        width=1,
    )
    return np.asarray(canvas, dtype=np.uint8), {
        "border_width_px": border,
        "directions": list(directions),
        "map_offset_px": [border, border],
        "map_size_px": [width, height],
        "map_frame": {
            "N": "+Y / image up",
            "E": "+X / image right",
            "S": "-Y / image down",
            "W": "-X / image left",
        },
    }


def add_cardinal_direction_border(
    image: Image.Image | np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    return add_direction_border(image, cardinal_only=True)


def _with_alpha(color: tuple[int, int, int], mode: str) -> tuple[int, ...]:
    if mode == "RGBA":
        return (int(color[0]), int(color[1]), int(color[2]), 255)
    return (int(color[0]), int(color[1]), int(color[2]))


def bev_source_mask(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8)
    background = np.asarray(BEV_UNKNOWN_BACKGROUND_COLOR, dtype=np.uint8)
    return ~np.all(arr == background, axis=2)


def strict_source_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    mask = bev_source_mask(image)
    if not np.any(mask):
        return (0, 0, int(image.shape[1]), int(image.shape[0]))
    ys, xs = np.nonzero(mask)
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


@dataclass(frozen=True)
class MarkerBBox:
    left: float
    top: float
    right: float
    bottom: float

    def expanded(self, gap_px: float) -> "MarkerBBox":
        return MarkerBBox(
            self.left - float(gap_px),
            self.top - float(gap_px),
            self.right + float(gap_px),
            self.bottom + float(gap_px),
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.left, self.top, self.right, self.bottom)


@dataclass(frozen=True)
class NumberedCircleMarkerStyle:
    radius_px: int
    font_size_px: int
    fill: tuple[int, int, int, int] = (255, 255, 255, 145)
    outline: tuple[int, int, int, int] = (35, 35, 35, 72)
    text_fill: tuple[int, int, int, int] = (18, 18, 18, 245)
    outline_width_px: int = 1
    supersample: int = 4


def _clamp_marker_int(value: float, min_value: int, max_value: int) -> int:
    return max(int(min_value), min(int(max_value), int(round(float(value)))))


def numbered_circle_marker_style(
    *,
    image_width: int,
    image_height: int,
    mode: str = "rgb",
) -> NumberedCircleMarkerStyle:
    short_side = max(1, min(int(image_width), int(image_height)))
    if str(mode).strip().lower() == "bev":
        radius_px = _clamp_marker_int(float(short_side) * 0.018, 8, 11)
        font_size_px = int(round(float(radius_px) * 1.28))
    else:
        radius_px = _clamp_marker_int(float(short_side) * 0.024, 9, 12)
        font_size_px = int(round(float(radius_px) * 1.18))
    return NumberedCircleMarkerStyle(
        radius_px=int(radius_px),
        font_size_px=int(font_size_px),
    )


def place_node_marker_style(
    *,
    image_width: int,
    image_height: int,
    mode: str = "rgb",
) -> NumberedCircleMarkerStyle:
    base_style = numbered_circle_marker_style(
        image_width=int(image_width),
        image_height=int(image_height),
        mode=str(mode),
    )
    return NumberedCircleMarkerStyle(
        radius_px=int(base_style.radius_px),
        font_size_px=int(base_style.font_size_px),
        fill=PLACE_NODE_MARKER_FILL,
        outline=PLACE_NODE_MARKER_OUTLINE,
        text_fill=PLACE_NODE_MARKER_TEXT,
        outline_width_px=max(2, int(base_style.outline_width_px)),
        supersample=int(base_style.supersample),
    )


def numbered_circle_marker_bbox(
    *,
    center_xy: tuple[float, float],
    style: NumberedCircleMarkerStyle,
) -> MarkerBBox:
    radius = float(style.radius_px) + float(style.outline_width_px)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    return MarkerBBox(cx - radius, cy - radius, cx + radius, cy + radius)


def clamp_numbered_circle_marker_center(
    *,
    center_xy: tuple[float, float],
    style: NumberedCircleMarkerStyle,
    image_size: tuple[int, int],
) -> tuple[float, float]:
    radius = float(style.radius_px) + float(style.outline_width_px)
    width, height = int(image_size[0]), int(image_size[1])
    x = min(max(float(center_xy[0]), radius), max(radius, float(width) - radius - 1.0))
    y = min(max(float(center_xy[1]), radius), max(radius, float(height) - radius - 1.0))
    return (float(x), float(y))


def _numbered_circle_text_y_offset(font: ImageFont.ImageFont) -> float:
    probe = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(probe)
    bbox = draw.textbbox((0.0, 0.0), "88", font=font, anchor="mm")
    return -((float(bbox[1]) + float(bbox[3])) / 2.0)


def draw_numbered_circle_marker(
    image: Image.Image,
    center_xy: tuple[float, float],
    label: str,
    *,
    style: NumberedCircleMarkerStyle,
) -> None:
    if image.mode != "RGBA":
        raise ValueError("draw_numbered_circle_marker requires an RGBA image")
    scale = max(1, int(style.supersample))
    radius = float(style.radius_px)
    patch_margin = int(math.ceil(float(style.outline_width_px) + 3.0))
    patch_radius = int(math.ceil(radius + float(patch_margin)))
    patch_size = max(1, patch_radius * 2 + 1)
    width, height = int(image.size[0]), int(image.size[1])
    cx = min(
        max(float(center_xy[0]), float(patch_radius)),
        max(float(patch_radius), float(width) - float(patch_radius) - 1.0),
    )
    cy = min(
        max(float(center_xy[1]), float(patch_radius)),
        max(float(patch_radius), float(height) - float(patch_radius) - 1.0),
    )
    high_size = int(patch_size * scale)
    high_radius = float(radius) * float(scale)
    high_center = float(high_size) / 2.0
    marker_patch = Image.new("RGBA", (high_size, high_size), (0, 0, 0, 0))
    marker_draw = ImageDraw.Draw(marker_patch, "RGBA")
    marker_draw.ellipse(
        (
            high_center - high_radius,
            high_center - high_radius,
            high_center + high_radius,
            high_center + high_radius,
        ),
        fill=style.fill,
        outline=style.outline,
        width=max(1, int(style.outline_width_px) * scale),
    )
    marker_patch = marker_patch.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
    left = int(round(cx - float(patch_size) / 2.0))
    top = int(round(cy - float(patch_size) / 2.0))
    image.alpha_composite(marker_patch, (left, top))

    text = str(label).strip()
    if text == "":
        return
    font = load_visual_font(int(style.font_size_px))
    text_draw = ImageDraw.Draw(image, "RGBA")
    text_draw.text(
        (cx, cy + _numbered_circle_text_y_offset(font)),
        text,
        fill=style.text_fill,
        font=font,
        anchor="mm",
    )


def marker_bbox(
    *,
    draw: ImageDraw.ImageDraw,
    center_xy: tuple[float, float],
    label: str,
    font: ImageFont.ImageFont,
    rect_height_px: int,
    padding_x_px: int,
) -> MarkerBBox:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    radius = max(float(rect_height_px) / 2.0, float(BEV_FRONTIER_RECT_HEIGHT_PX) / 2.0)
    return MarkerBBox(
        cx - radius,
        cy - radius,
        cx + radius,
        cy + radius,
    )


def draw_frontier_marker(
    draw: ImageDraw.ImageDraw,
    center_xy: tuple[float, float],
    label: str,
    *,
    font: ImageFont.ImageFont | None = None,
    rect_height_px: int = BEV_FRONTIER_RECT_HEIGHT_PX,
    padding_x_px: int = BEV_FRONTIER_PADDING_X_PX,
) -> None:
    marker_font = load_visual_font(BEV_FRONTIER_FONT_SIZE_PX) if font is None else font
    radius = max(float(rect_height_px) / 2.0, float(BEV_FRONTIER_RECT_HEIGHT_PX) / 2.0)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=(255, 255, 255, 145),
        outline=(35, 35, 35, 72),
        width=1,
    )
    draw.text(
        (float(center_xy[0]), float(center_xy[1])),
        str(label),
        fill=(18, 18, 18, 245),
        font=marker_font,
        anchor="mm",
    )
