from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from navprobe.agent.visual_action_context import direction_for_angle
from navprobe.agent.visual_action_context import ordered_visual_views
from navprobe.agent.visual_action_context import VisualActionContext, VisualViewContext
from navprobe.agent.visual_policy_decisions import NodeFrontierOverlayPromptImage
from navprobe.llm.image_preprocessing import compose_llm_image_tile_sheet
from navprobe.llm.image_preprocessing import encode_rgb_to_jpeg_data_url
from navprobe.llm.image_preprocessing import LLM_CAMERA_IMAGE_MAX_SIZE
from navprobe.llm.image_preprocessing import LLM_IMAGE_TILE_SIZE
from navprobe.llm.image_preprocessing import normalize_rgb_array
from navprobe.llm.image_preprocessing import resize_rgb_to_fit
from navprobe.visualization.waypoint_overlay import draw_waypoint_overlay_rgb
from navprobe.visualization.waypoint_overlay import normalized_point_to_pixel

if TYPE_CHECKING:
    from navprobe.memory.task_progress import TaskProgressMemory
    from navprobe.runtime.cache import RuntimeCache


PANORAMA_STRIP_SEPARATOR_WIDTH_PX = 10


def inline_int_list(items: list[int]) -> str:
    normalized = [int(item) for item in items]
    return ", ".join(str(item) for item in normalized) if normalized != [] else "none"


def current_panorama_prompt_text(
    visual_context: VisualActionContext,
    *,
    include_visited_nodes: bool = True,
    planning_reference: bool = False,
) -> str:
    if planning_reference:
        node_role = "Planning reference node"
    else:
        node_role = "Current graph node"
    visited_nodes_section = (
        _visible_visited_nodes_section(visual_context)
        if include_visited_nodes
        else ""
    )
    arrival_section = ""
    if (
        include_visited_nodes
        and not planning_reference
        and visual_context.arrival_edge_id
        and any(view.visible_arrival_edge_ids for view in visual_context.views)
    ):
        arrival_section = (
            f"- The blue curve shows the last executed edge {visual_context.arrival_edge_id}, "
            f"from {visual_context.arrival_src_node_id} to {visual_context.current_node_id}. "
            "Arrows show the direction already traveled toward the current position.\n"
            "- The curve is smoothed for display and shares its endpoints with the node markers. "
            "Only visible path segments are drawn."
        )
    return f"""
{node_role}: {visual_context.current_node_id}.
{visited_nodes_section}
{arrival_section}
""".strip()


def selected_view_prompt_text(view: VisualViewContext) -> str:
    visited_nodes_section = _visible_visited_nodes_for_view_section(view)
    return f"""
Selected RGB image for {direction_for_angle(view.angle_deg)}:
- The attached image is the selected view for marking a reachable local waypoint.
{visited_nodes_section}
""".strip()


def compact_bev_overview_prompt_text() -> str:
    return """
Compact BEV overview image:
- The attached BEV overview shows the current robot position, blue visited place-node markers, and active frontier markers.
- It intentionally does not draw move edges or visited trajectory.
- Frontier markers only indicate that active global frontiers exist and where they roughly are.
""".strip()


def explore_bev_frontier_overlay_prompt_text(
    *,
    include_graph_context: bool = True,
) -> str:
    if not include_graph_context:
        return """
Global BEV waypoint overlay image:
- The robot icon marks the current position.
- Numbered frontier circles mark active waypoint candidates.
- Numbers inside circles are the candidate labels.
""".strip()
    return """
Global BEV frontier overlay image:
- The robot icon marks the current position/current node.
- Blue numbered place-node circles mark visited nodes.
- Numbered frontier circles mark active global frontiers.
- Numbers inside circles are the candidate labels.
- Edges are intentionally not drawn; use graph memory text for connectivity.
""".strip()


def node_frontier_overlay_prompt_text(
    overlay: NodeFrontierOverlayPromptImage,
    *,
    include_graph_context: bool = True,
) -> str:
    direction_text = (
        ""
        if overlay.angle_deg is None
        else f" for {direction_for_angle(overlay.angle_deg)}"
    )
    if not include_graph_context:
        return (
            f"Frontier RGB candidate source view{direction_text}:\n"
            "- Numbered frontier circles mark visible active waypoint candidates.\n"
            "- Numbers inside circles are the same labels used in the candidate BEV.\n"
            f"- Visible frontier labels: {inline_int_list(overlay.frontier_labels)}."
        )
    return (
        f"Frontier RGB overlay image for {overlay.node_reference}:\n"
        "- Numbered frontier circles mark visible active global frontiers in this source-node RGB overlay.\n"
        "- Numbers inside circles are the same frontier labels used in the global BEV frontier overlay.\n"
        f"- Visible frontier labels: {inline_int_list(overlay.frontier_labels)}."
    )


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
                f"Planning reference {direction} view:"
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


def append_task_progress_memory_images(
    *,
    content: list[dict[str, object]],
    cache: "RuntimeCache",
    task_progress: "TaskProgressMemory",
) -> None:
    for item in task_progress.items:
        if item.kind != "verify_candidate" or item.status != "active":
            continue
        candidate = item.candidate
        source_view = candidate.get("source_view")
        if not isinstance(source_view, dict):
            continue
        candidate_id = str(candidate.get("candidate_id", "")).strip()
        if candidate_id == "":
            continue
        content.append({"type": "text", "text": f"Task progress source view: {candidate_id}"})
        _append_source_view_image(
            content=content,
            cache=cache,
            source_view=source_view,
        )


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


def _append_source_view_image(
    *,
    content: list[dict[str, object]],
    cache: "RuntimeCache",
    source_view: dict[str, object],
) -> None:
    image_id = str(source_view.get("image_id", "")).strip()
    if image_id != "":
        image = np.asarray(cache.get_image(image_id).image)
        content.append(image_content_for_camera_array(image))
        return
    obs_id = str(source_view.get("obs_id", "")).strip()
    if obs_id != "":
        content.append(_image_content_for_obs(cache=cache, obs_id=obs_id))


def _image_content_for_obs(*, cache: "RuntimeCache", obs_id: str) -> dict[str, object]:
    return image_content_for_camera_array(_image_array_for_obs(cache=cache, obs_id=obs_id))


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
