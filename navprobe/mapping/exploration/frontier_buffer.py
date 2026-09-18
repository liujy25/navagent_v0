from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FrontierEntry:
    id: str
    xy: tuple[float, float]
    goal_xy: tuple[float, float] | None = None
    goal_yaw_deg: float | None = None
    evidence_obs_id: str | None = None
    evidence_frontier_xy: tuple[float, float] | None = None
    evidence_end_xy: tuple[float, float] | None = None
    evidence_center_distance_sq: float | None = None
    score_sum: float = 0.0
    score_count: int = 0
    score_mean: float = 0.0
    blacklisted: bool = False
    blacklist_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "frontier_id": self.id,
            "xy": [float(self.xy[0]), float(self.xy[1])],
            "score_sum": float(self.score_sum),
            "score_count": int(self.score_count),
            "score_mean": float(self.score_mean),
        }
        if self.goal_xy is not None and self.goal_yaw_deg is not None:
            payload["goal_pose"] = {
                "x": float(self.goal_xy[0]),
                "y": float(self.goal_xy[1]),
                "yaw": float(self.goal_yaw_deg),
            }
        if (
            self.evidence_obs_id is not None
            and self.evidence_frontier_xy is not None
            and self.evidence_end_xy is not None
            and self.evidence_center_distance_sq is not None
        ):
            payload["visual_evidence"] = {
                "obs_id": str(self.evidence_obs_id),
                "frontier_xy": [
                    float(self.evidence_frontier_xy[0]),
                    float(self.evidence_frontier_xy[1]),
                ],
                "end_xy": [
                    float(self.evidence_end_xy[0]),
                    float(self.evidence_end_xy[1]),
                ],
                "center_distance_sq": float(self.evidence_center_distance_sq),
            }
        if self.blacklisted:
            payload["blacklisted"] = True
            payload["blacklist_reason"] = self.blacklist_reason
        return payload


class FrontierBuffer:
    def __init__(self, match_distance_m: float = 0.75) -> None:
        self.match_distance_m = float(match_distance_m)
        self.entries: list[FrontierEntry] = []
        self._frontier_index = 0

    def clear(self) -> None:
        self.entries = []
        self._frontier_index = 0

    def selectable_entries(self) -> list[FrontierEntry]:
        return [entry for entry in self.entries if not bool(entry.blacklisted)]

    def blacklist(self, frontier_id: str, reason: str) -> dict[str, object]:
        frontier_id_str = str(frontier_id)
        for entry in self.entries:
            if entry.id != frontier_id_str:
                continue
            entry.blacklisted = True
            entry.blacklist_reason = str(reason)
            return {
                "frontier_id": frontier_id_str,
                "blacklisted": True,
                "reason": str(reason),
            }
        raise ValueError(f"Unknown frontier_id for blacklist: {frontier_id_str}")

    def update(self, frontiers_xy: np.ndarray) -> list[FrontierEntry]:
        previous_entries = list(self.entries)
        matched_previous: set[int] = set()
        updated_entries: list[FrontierEntry] = []

        for frontier_xy in frontiers_xy:
            frontier_xy = np.asarray(frontier_xy, dtype=np.float64)

            matched_index = None
            matched_distance = None
            for index, entry in enumerate(previous_entries):
                if index in matched_previous:
                    continue
                entry_xy = np.asarray(entry.xy, dtype=np.float64)

                distance = float(np.linalg.norm(frontier_xy - entry_xy))
                if distance > self.match_distance_m:
                    continue

                if matched_distance is None or distance < matched_distance:
                    matched_index = index
                    matched_distance = distance

            if matched_index is None:
                frontier_id = f"f{self._frontier_index}"
                self._frontier_index += 1
                score_sum = 0.0
                score_count = 0
                score_mean = 0.0
                blacklisted = False
                blacklist_reason = None
            else:
                previous_entry = previous_entries[matched_index]
                frontier_id = previous_entry.id
                evidence_obs_id = previous_entry.evidence_obs_id
                evidence_frontier_xy = previous_entry.evidence_frontier_xy
                evidence_end_xy = previous_entry.evidence_end_xy
                evidence_center_distance_sq = previous_entry.evidence_center_distance_sq
                score_sum = float(previous_entry.score_sum)
                score_count = int(previous_entry.score_count)
                score_mean = float(previous_entry.score_mean)
                blacklisted = bool(previous_entry.blacklisted)
                blacklist_reason = previous_entry.blacklist_reason
                matched_previous.add(matched_index)
            if matched_index is None:
                evidence_obs_id = None
                evidence_frontier_xy = None
                evidence_end_xy = None
                evidence_center_distance_sq = None

            updated_entries.append(
                FrontierEntry(
                    id=frontier_id,
                    xy=(float(frontier_xy[0]), float(frontier_xy[1])),
                    goal_xy=None,
                    goal_yaw_deg=None,
                    evidence_obs_id=evidence_obs_id,
                    evidence_frontier_xy=evidence_frontier_xy,
                    evidence_end_xy=evidence_end_xy,
                    evidence_center_distance_sq=evidence_center_distance_sq,
                    score_sum=score_sum,
                    score_count=score_count,
                    score_mean=score_mean,
                    blacklisted=blacklisted,
                    blacklist_reason=blacklist_reason,
                )
            )

        self.entries = updated_entries
        return list(self.entries)

    def update_navigation_goals(self, candidates: list[object]) -> None:
        candidate_by_id: dict[str, object] = {}
        for candidate in candidates:
            frontier_id = getattr(candidate, "frontier_id", None)
            if frontier_id is None:
                continue
            candidate_by_id[str(frontier_id)] = candidate

        for entry in self.entries:
            candidate = candidate_by_id.get(str(entry.id))
            if candidate is None:
                entry.goal_xy = None
                entry.goal_yaw_deg = None
                continue
            goal_xy = getattr(candidate, "goal_xy", None)
            goal_yaw = getattr(candidate, "goal_yaw_degrees", None)
            if goal_xy is None or not callable(goal_yaw):
                entry.goal_xy = None
                entry.goal_yaw_deg = None
                continue
            goal_xy_arr = np.asarray(goal_xy, dtype=np.float64).reshape(2)
            entry.goal_xy = (float(goal_xy_arr[0]), float(goal_xy_arr[1]))
            entry.goal_yaw_deg = float(goal_yaw())

    def apply_ranking(self, frontier_ids: list[str]) -> list[dict[str, object]]:
        unique_frontier_ids: list[str] = []
        for frontier_id in frontier_ids:
            frontier_id_str = str(frontier_id)
            if frontier_id_str not in unique_frontier_ids:
                unique_frontier_ids.append(frontier_id_str)

        if unique_frontier_ids == []:
            return []

        entry_by_id = {entry.id: entry for entry in self.entries}
        valid_frontier_ids = [frontier_id for frontier_id in unique_frontier_ids if frontier_id in entry_by_id]
        if valid_frontier_ids == []:
            return []

        ranking_size = len(valid_frontier_ids)
        updates: list[dict[str, object]] = []
        for rank_index, frontier_id in enumerate(valid_frontier_ids):
            entry = entry_by_id[frontier_id]
            score = float(ranking_size - rank_index) / float(ranking_size)
            entry.score_sum = float(entry.score_sum) + score
            entry.score_count = int(entry.score_count) + 1
            entry.score_mean = float(entry.score_sum) / float(entry.score_count)
            updates.append(
                {
                    "frontier_id": frontier_id,
                    "score": score,
                    "score_sum": float(entry.score_sum),
                    "score_count": int(entry.score_count),
                    "score_mean": float(entry.score_mean),
                }
            )
        return updates
