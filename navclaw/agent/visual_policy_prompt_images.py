from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from navclaw.agent.visual_action_context import angle_convention_text
from navclaw.agent.visual_action_context import ordered_visual_views_left_to_right
from navclaw.agent.visual_action_context import panorama_angle_order_text
from navclaw.agent.visual_action_context import VisualActionContext, VisualViewContext
from navclaw.llm.image_preprocessing import compose_llm_image_tile_sheet
from navclaw.llm.image_preprocessing import encode_rgb_to_jpeg_data_url
from navclaw.llm.image_preprocessing import LLM_CAMERA_IMAGE_MAX_SIZE
from navclaw.llm.image_preprocessing import LLM_IMAGE_TILE_SIZE
from navclaw.llm.image_preprocessing import normalize_rgb_array
from navclaw.llm.image_preprocessing import resize_rgb_to_fit
from navclaw.visualization.waypoint_overlay import draw_waypoint_overlay_rgb
from navclaw.visualization.waypoint_overlay import normalized_point_to_pixel

if TYPE_CHECKING:
    from navclaw.runtime.cache import RuntimeCache


PANORAMA_STRIP_SEPARATOR_WIDTH_PX = 10


def current_panorama_strip_prompt_text(
    visual_context: VisualActionContext,
    *,
    include_visited_nodes: bool = True,
    planning_reference: bool = False,
    separate_reference_heading: bool = False,
) -> str:
    angles = visual_context.available_angles
    if planning_reference:
        node_role = "Current planning node"
        heading_reference = (
            f"the stored heading of planning node {visual_context.current_node_id}"
        )
    else:
        node_role = "Current graph node"
        heading_reference = "the current robot heading"
    visited_nodes_section = (
        _visible_visited_nodes_section(visual_context)
        if include_visited_nodes
        else ""
    )
    if separate_reference_heading:
        angle_text = angle_convention_text(
            angles,
            heading_reference="each panorama's reference heading",
        )
        reference_heading_text = (
            f"- The reference heading for this current panorama is {heading_reference}."
        )
    else:
        angle_text = angle_convention_text(
            angles,
            heading_reference=heading_reference,
        )
        reference_heading_text = ""
    reference_heading_line = (
        f"{reference_heading_text}\n" if reference_heading_text != "" else ""
    )
    return f"""
Current panorama strip:
- {node_role}: {visual_context.current_node_id}.
- Views are ordered left-to-right as {panorama_angle_order_text(angles)}.
- {angle_text}
{reference_heading_line}- Angle labels are view labels, not scene objects.
{visited_nodes_section}
""".strip()


def vertical_transition_panorama_strip_prompt_text(visual_context: VisualActionContext) -> str:
    angles = visual_context.available_angles
    return f"""
Current panorama:
- Views are ordered left-to-right as {panorama_angle_order_text(angles)}.
- {angle_convention_text(angles)}
- Angle labels are labels, not scene objects.
""".strip()


def selected_view_prompt_text(view: VisualViewContext) -> str:
    visited_nodes_section = _visible_visited_nodes_for_view_section(view)
    return f"""
Selected RGB image for angle_{int(view.angle_deg)}:
- The attached image is the selected view for marking a reachable local waypoint.
{visited_nodes_section}
""".strip()


def image_content_for_current_panorama_strip(
    *,
    cache: "RuntimeCache",
    views: list[VisualViewContext],
    include_visited_nodes: bool = True,
) -> dict[str, object]:
    ordered_views = ordered_visual_views_left_to_right(views)
    images = [
        (
            _image_array_for_view(cache=cache, view=view)
            if include_visited_nodes
            else _image_array_for_obs(cache=cache, obs_id=view.obs_id)
        )
        for view in ordered_views
    ]
    labels = [f"angle_{int(view.angle_deg)}" for view in ordered_views]
    strip = _compose_labeled_image_strip(images=images, labels=labels)
    return image_content_for_array(strip)


def image_content_for_vertical_transition_panorama_strip(
    *,
    cache: "RuntimeCache",
    views: list[VisualViewContext],
) -> dict[str, object]:
    ordered_views = ordered_visual_views_left_to_right(views)
    images = [_image_array_for_obs(cache=cache, obs_id=view.obs_id) for view in ordered_views]
    labels = [f"angle_{int(view.angle_deg)}" for view in ordered_views]
    strip = _compose_labeled_image_strip(images=images, labels=labels)
    return image_content_for_array(strip)


def image_content_for_view(*, cache: "RuntimeCache", view: VisualViewContext) -> dict[str, object]:
    return image_content_for_camera_array(_image_array_for_view(cache=cache, view=view))


def image_content_for_array(image: np.ndarray) -> dict[str, object]:
    return {
        "type": "image_url",
        "image_url": {"url": encode_rgb_to_jpeg_data_url(image, quality=90)},
    }


def image_content_for_camera_array(image: np.ndarray) -> dict[str, object]:
    resized = resize_rgb_to_fit(image, max_size=LLM_CAMERA_IMAGE_MAX_SIZE).image
    return image_content_for_array(resized)


def image_content_for_stop_waypoint_overlay(
    *,
    cache: "RuntimeCache",
    pending_stop: dict[str, object],
) -> dict[str, object] | None:
    obs_id = str(pending_stop.get("selected_obs_id") or pending_stop.get("waypoint_obs_id") or "").strip()
    point = _point_2d_from_pending_stop(pending_stop)
    if obs_id == "" or point is None:
        return None
    observation = cache.get_observation(obs_id).observation
    rgb = np.asarray(observation.rgb)
    resized = resize_rgb_to_fit(rgb, max_size=LLM_CAMERA_IMAGE_MAX_SIZE).image
    height, width = int(resized.shape[0]), int(resized.shape[1])
    point_pixel = normalized_point_to_pixel(point, image_width=width, image_height=height)
    overlay = draw_waypoint_overlay_rgb(resized, point_pixel=point_pixel, label="waypoint")
    return image_content_for_array(overlay)


def image_content_for_movement_history_sheet(
    *,
    cache: "RuntimeCache",
    obs_ids: list[str],
) -> dict[str, object] | None:
    normalized_obs_ids = [str(obs_id).strip() for obs_id in obs_ids if str(obs_id).strip() != ""]
    if normalized_obs_ids == []:
        return None
    images = [
        np.asarray(cache.get_observation(obs_id).observation.rgb)
        for obs_id in normalized_obs_ids
    ]
    first_image = normalize_rgb_array(images[0])
    source_height, source_width = int(first_image.shape[0]), int(first_image.shape[1])
    max_tile_width, max_tile_height = LLM_IMAGE_TILE_SIZE
    tile_scale = min(
        float(max_tile_width) / float(source_width),
        float(max_tile_height) / float(source_height),
    )
    tile_size = (
        max(1, int(round(float(source_width) * tile_scale))),
        max(1, int(round(float(source_height) * tile_scale))),
    )
    sheet = compose_llm_image_tile_sheet(
        images,
        tile_size=tile_size,
        max_columns=4,
    ).image
    return image_content_for_array(sheet)


def _point_2d_from_pending_stop(pending_stop: dict[str, object]) -> list[float] | None:
    for key in ("point_2d", "waypoint_point_2d", "visual_action_point_2d"):
        value = pending_stop.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return [float(value[0]), float(value[1])]
    return None


def _visible_visited_nodes_text(visual_context: VisualActionContext) -> str:
    lines: list[str] = []
    for view in visual_context.views:
        node_items = [_visited_node_label_mapping(node_id) for node_id in view.visible_visited_nodes]
        if node_items == []:
            continue
        lines.append(f"angle_{int(view.angle_deg)}: {', '.join(node_items)}")
    return "\n".join(lines)


def _visible_visited_nodes_for_view_text(view: VisualViewContext) -> str:
    node_items = [_visited_node_label_mapping(node_id) for node_id in view.visible_visited_nodes]
    return ", ".join(node_items)


def _visible_visited_nodes_section(visual_context: VisualActionContext) -> str:
    visible_nodes_text = _visible_visited_nodes_text(visual_context)
    if visible_nodes_text == "":
        return ""
    return f"""

Visited node overlay in this panorama:
- Blue numbered circles like "3" or "12" mark previously visited nodes projected into the current view.
- These node circles and labels are drawn overlays, not physical objects.
- Use these node overlays as spatial references and visited-place evidence.
- The visible_visited_nodes list maps those numeric overlay labels back to graph node ids.

visible_visited_nodes:
{visible_nodes_text}
""".rstrip()


def _visible_visited_nodes_for_view_section(view: VisualViewContext) -> str:
    visible_nodes_text = _visible_visited_nodes_for_view_text(view)
    if visible_nodes_text == "":
        return ""
    return f"""
- Blue numbered circles like "3" or "12" mark previously visited nodes.
- These node circles and labels are drawn overlays, not physical objects.
- Use these node overlays as spatial references from LA.
- The visible_visited_nodes list maps those numeric overlay labels back to graph node ids.

visible_visited_nodes:
{visible_nodes_text}
""".rstrip()


def _visited_node_label_mapping(node_id: object) -> str:
    text = str(node_id).strip()
    if len(text) >= 2 and text[0].lower() == "n" and text[1:].isdigit():
        return f"{text[1:]} -> {text}"
    return text


def _image_array_for_view(*, cache: "RuntimeCache", view: VisualViewContext) -> np.ndarray:
    image_id = str(view.node_overlay_image_id).strip()
    if image_id != "":
        return np.asarray(cache.get_image(image_id).image)
    return _image_array_for_obs(cache=cache, obs_id=view.obs_id)


def _image_array_for_obs(*, cache: "RuntimeCache", obs_id: str) -> np.ndarray:
    observation = cache.get_observation(str(obs_id)).observation
    return np.asarray(observation.rgb)


def _compose_labeled_image_strip(
    *,
    images: list[np.ndarray],
    labels: list[str],
) -> np.ndarray:
    if images == []:
        raise ValueError("panorama strip requires at least one image")
    if len(images) != len(labels):
        raise ValueError("panorama strip images and labels must have the same length")

    resized_images = [
        resize_rgb_to_fit(image, max_size=LLM_CAMERA_IMAGE_MAX_SIZE).image
        for image in images
    ]
    first = normalize_rgb_array(resized_images[0])
    tile_height = int(first.shape[0])
    tile_width = int(first.shape[1])
    label_height = max(40, int(round(float(tile_height) * 0.09)))
    tile_count = int(len(images))
    separator_width = int(PANORAMA_STRIP_SEPARATOR_WIDTH_PX)
    canvas = np.full(
        (
            tile_height + label_height,
            tile_width * tile_count + separator_width * (tile_count - 1),
            3,
        ),
        255,
        dtype=np.uint8,
    )
    font = _strip_label_font(tile_height)
    for index, (raw_image, label) in enumerate(zip(resized_images, labels)):
        rgb = normalize_rgb_array(raw_image)
        if int(rgb.shape[0]) != tile_height or int(rgb.shape[1]) != tile_width:
            rgb = np.asarray(
                Image.fromarray(rgb).resize(
                    (tile_width, tile_height),
                    Image.Resampling.LANCZOS,
                ),
                dtype=np.uint8,
            )
        x0 = index * (tile_width + separator_width)
        x1 = x0 + tile_width
        canvas[:tile_height, x0:x1] = rgb
        if index < tile_count - 1:
            sep_x0 = x1
            sep_x1 = min(sep_x0 + separator_width, canvas.shape[1])
            canvas[:, sep_x0:sep_x1] = np.asarray([0, 0, 0], dtype=np.uint8)
        _draw_centered_label(
            canvas=canvas,
            label=str(label),
            x0=x0,
            x1=x1,
            y0=tile_height,
            y1=tile_height + label_height,
            font=font,
        )
    return canvas


def _strip_label_font(tile_height: int) -> ImageFont.ImageFont:
    font_size = max(18, int(round(float(tile_height) * 0.045)))
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
    except OSError:
        return ImageFont.load_default()


def _draw_centered_label(
    *,
    canvas: np.ndarray,
    label: str,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    font: ImageFont.ImageFont,
) -> None:
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), str(label), font=font)
    text_width = int(bbox[2] - bbox[0])
    text_height = int(bbox[3] - bbox[1])
    text_x = int(x0 + max(0, (int(x1) - int(x0) - text_width) // 2))
    text_y = int(y0 + max(0, (int(y1) - int(y0) - text_height) // 2))
    draw.text((text_x, text_y), str(label), fill=(0, 0, 0), font=font)
    canvas[:, :, :] = np.asarray(image, dtype=np.uint8)
