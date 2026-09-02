from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from navclaw.agent.visual_action_context import direction_for_angle
from navclaw.agent.visual_action_context import ordered_visual_views
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


def current_panorama_prompt_text(
    visual_context: VisualActionContext,
    *,
    include_visited_nodes: bool = True,
    planning_reference: bool = False,
) -> str:
    if planning_reference:
        node_role = "Current planning node"
    else:
        node_role = "Current graph node"
    visited_nodes_section = (
        _visible_visited_nodes_section(visual_context)
        if include_visited_nodes
        else ""
    )
    return f"""
{node_role}: {visual_context.current_node_id}.
{visited_nodes_section}
""".strip()


def selected_view_prompt_text(view: VisualViewContext) -> str:
    visited_nodes_section = _visible_visited_nodes_for_view_section(view)
    return f"""
Selected RGB image for {direction_for_angle(view.angle_deg)}:
- The attached image is the selected view for marking a reachable local waypoint.
{visited_nodes_section}
""".strip()


def image_content_for_current_panorama_views(
    *,
    cache: "RuntimeCache",
    views: list[VisualViewContext],
    include_visited_nodes: bool = True,
    image_overrides_by_angle: dict[int, np.ndarray] | None = None,
    planning_reference: bool = False,
    label_prefix: str | None = None,
) -> list[dict[str, object]]:
    content: list[dict[str, object]] = []
    overrides = {
        int(angle) % 360: np.asarray(image, dtype=np.uint8)
        for angle, image in (image_overrides_by_angle or {}).items()
    }
    for view in ordered_visual_views(views):
        angle = int(view.angle_deg) % 360
        image = (
            overrides[angle]
            if angle in overrides
            else (
                _image_array_for_view(cache=cache, view=view)
                if include_visited_nodes
                else _image_array_for_obs(cache=cache, obs_id=view.obs_id)
            )
        )
        direction = direction_for_angle(angle)
        if label_prefix is None:
            view_label = (
                f"Current planner {direction} view:"
                if planning_reference
                else f"Current {direction} view:"
            )
        else:
            view_label = f"{label_prefix}: {direction}."
        content.append({"type": "text", "text": view_label})
        content.append(image_content_for_camera_array(image))
    return content


def image_content_for_vertical_transition_panorama_views(
    *,
    cache: "RuntimeCache",
    views: list[VisualViewContext],
) -> list[dict[str, object]]:
    return image_content_for_current_panorama_views(
        cache=cache,
        views=views,
        include_visited_nodes=False,
        label_prefix="Current vertical-transition panorama view",
    )


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
    for view in ordered_visual_views(visual_context.views):
        node_items = [_visited_node_label_mapping(node_id) for node_id in view.visible_visited_nodes]
        if node_items == []:
            continue
        lines.append(
            f"{direction_for_angle(view.angle_deg)}: {', '.join(node_items)}"
        )
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
