"""Explicit scan, candidate selection, approach, and recovery state machine."""

from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mc_rl.candidates import CandidateMap, ResourceCandidate, ResourceDetection
from mc_rl.navigation import wrap_degrees
from mc_rl.progress import VisualProgressMonitor
from mc_rl.resource_adapters import ResourceAdapter
from mc_rl.telemetry import (
    AgentTelemetry,
    SENSOR_PROFILE_F3,
    SENSOR_PROFILES,
    bearing_and_distance_to,
)


class SearchState(str, Enum):
    SCAN = "SCAN"
    BUILD_CANDIDATE_MAP = "BUILD_CANDIDATE_MAP"
    SELECT_TARGET = "SELECT_TARGET"
    ALIGN = "ALIGN"
    APPROACH = "APPROACH"
    LOCAL_REACQUIRE = "LOCAL_REACQUIRE"
    RECOVER = "RECOVER"
    REPLAN = "REPLAN"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


@dataclass
class SearchConfig:
    camera_delta: float = 10.0
    full_scan_degrees: float = 360.0
    align_threshold_degrees: float = 6.0
    selected_match_degrees: float = 28.0
    scan_detection_half_width: float = 14.0
    local_reacquire_turns: int = 3
    recovery_backward_steps: int = 5
    recovery_turn_steps: int = 3
    cooldown_steps: int = 45
    max_recovery_attempts: int = 1
    noop_action: int = 0
    forward_action: int = 1
    forward_jump_action: int = 2
    left_action: int = 3
    right_action: int = 4
    backward_action: int = 7
    # Non-zero values are only for labelled fault-injection diagnostics. The
    # deployment default always chooses score rank zero.
    initial_selection_rank: int = 0
    rescan_after_replan: bool = True
    sensor_profile: str = "pov_only"
    world_arrival_radius: float = 2.5


class CandidateSearchPolicy:
    """Inspectable candidate policy for POV-only or F3-assisted deployment.

    F3 mode reads only the player's own pose/biome telemetry. Evaluation-only
    oracle and block-grid fields can never influence selection or actions.
    """

    def __init__(
        self,
        adapter: ResourceAdapter,
        config: Optional[SearchConfig] = None,
        candidate_map: Optional[CandidateMap] = None,
        progress_monitor: Optional[VisualProgressMonitor] = None,
    ):
        self.adapter = adapter
        self.config = config or SearchConfig()
        if self.config.sensor_profile not in SENSOR_PROFILES:
            raise ValueError(
                "unknown sensor profile: {}".format(self.config.sensor_profile)
            )
        self.candidate_map = candidate_map or CandidateMap()
        self.progress = progress_monitor or VisualProgressMonitor()
        self.reset()

    def reset(self, episode: int = 0) -> None:
        self.episode = int(episode)
        self.state = SearchState.SCAN
        self.step = 0
        self.heading_yaw = 0.0
        self.scan_yaw = 0.0
        self.scan_cycles = 0
        self.selected_candidate: Optional[ResourceCandidate] = None
        self.initial_selected_candidate_id: Optional[int] = None
        self.transition_log: List[Dict[str, Any]] = []
        self.action_counts: Counter = Counter()
        self.replan_count = 0
        self.recovery_count = 0
        self.stalled_count = 0
        self._previous_pov: Optional[np.ndarray] = None
        self._last_action: Optional[int] = None
        self._last_action_state = self.state
        self._last_progress_step = -1
        self._local_actions: Deque[int] = deque()
        self._recovery_actions: Deque[int] = deque()
        self._recoveries_by_candidate: Counter = Counter()
        self._local_trigger = ""
        self._finished = False
        self.current_telemetry: Optional[AgentTelemetry] = None
        # A policy object is reusable across episodes, but candidate identities
        # are deliberately episode-local.
        self.candidate_map = CandidateMap(
            merge_yaw_degrees=self.candidate_map.merge_yaw_degrees,
            merge_log_size_tolerance=self.candidate_map.merge_log_size_tolerance,
            world_merge_min_distance=self.candidate_map.world_merge_min_distance,
            score_config=self.candidate_map.score_config,
        )
        self.progress.reset()

    @staticmethod
    def _pov(observation: Dict[str, Any]) -> np.ndarray:
        # Deliberately do not iterate the observation mapping: a guarded test
        # can prove that an ``oracle`` sibling is never touched.
        return np.asarray(observation["pov"])

    def _telemetry(
        self, observation: Dict[str, Any]
    ) -> Optional[AgentTelemetry]:
        if self.config.sensor_profile != SENSOR_PROFILE_F3:
            return None
        # Access the explicitly declared telemetry sibling directly. Never
        # iterate the mapping, which also keeps oracle isolation testable.
        return AgentTelemetry.from_observation(observation["telemetry"])

    def _transition(self, new_state: SearchState, reason: str) -> None:
        if new_state == self.state:
            return
        old_state = self.state
        self.state = new_state
        self.transition_log.append(
            {
                "episode": self.episode,
                "step": self.step,
                "old_state": old_state.value,
                "new_state": new_state.value,
                "reason": reason,
                "selected_candidate_id": (
                    "" if self.selected_candidate is None else self.selected_candidate.candidate_id
                ),
            }
        )

    def _ingest_scan_detections(
        self,
        detections: Sequence[ResourceDetection],
        telemetry: Optional[AgentTelemetry],
    ) -> None:
        for detection in detections:
            if abs(detection.horizontal_yaw) > self.config.scan_detection_half_width:
                continue
            range_estimate = (
                None if telemetry is None else self.adapter.estimate_range(detection)
            )
            self.candidate_map.add_detection(
                detection,
                observer_yaw=self.heading_yaw,
                step=self.step,
                telemetry=(telemetry if range_estimate is not None else None),
                range_estimate=range_estimate,
            )

    def _selected_detection(
        self, detections: Sequence[ResourceDetection]
    ) -> Optional[ResourceDetection]:
        if self.selected_candidate is None:
            return None
        matches = []
        for detection in detections:
            world_yaw = wrap_degrees(self.heading_yaw + detection.horizontal_yaw)
            error = self.candidate_map.angular_distance(
                world_yaw, self.selected_candidate.relative_yaw
            )
            if error <= self.config.selected_match_degrees:
                matches.append((error, -detection.apparent_size, detection))
        return min(matches)[-1] if matches else None

    def _update_selected_bearing(
        self,
        detection: ResourceDetection,
        telemetry: Optional[AgentTelemetry],
    ) -> None:
        if self.selected_candidate is None:
            return
        range_estimate = (
            None if telemetry is None else self.adapter.estimate_range(detection)
        )
        if telemetry is not None and range_estimate is not None:
            self.candidate_map.update_candidate_position(
                self.selected_candidate,
                detection,
                telemetry,
                range_estimate,
                self.step,
            )
        else:
            observed_yaw = wrap_degrees(
                self.heading_yaw + detection.horizontal_yaw
            )
            # A moderate update follows apparent bearing changes while
            # retaining enough memory to bridge a few missed frames.
            delta = wrap_degrees(
                observed_yaw - self.selected_candidate.relative_yaw
            )
            self.selected_candidate.relative_yaw = wrap_degrees(
                self.selected_candidate.relative_yaw + 0.45 * delta
            )
        self.selected_candidate.confidence = max(
            self.selected_candidate.confidence, detection.confidence
        )
        self.selected_candidate.apparent_size = max(
            self.selected_candidate.apparent_size, detection.apparent_size
        )
        self.selected_candidate.last_seen_step = self.step
        self.selected_candidate.observation_count += 1

    def _world_route(self) -> Optional[Tuple[float, float]]:
        if (
            self.current_telemetry is None
            or self.selected_candidate is None
            or not self.selected_candidate.has_world_position
        ):
            return None
        bearing, distance = bearing_and_distance_to(
            self.current_telemetry,
            float(self.selected_candidate.estimated_world_x),
            float(self.selected_candidate.estimated_world_z),
        )
        return wrap_degrees(bearing - self.current_telemetry.yaw), distance

    def _record_progress(
        self, pov: np.ndarray, selected_detection: Optional[ResourceDetection]
    ) -> None:
        if self.state != SearchState.APPROACH or self._last_progress_step == self.step:
            return
        forward_actions = {
            self.config.forward_action,
            self.config.forward_jump_action,
            self.adapter.interaction_action(),
        }
        self.progress.add(
            forward=self._last_action in forward_actions,
            apparent_size=(
                None if selected_detection is None else selected_detection.apparent_size
            ),
            alignment_error=(
                None if selected_detection is None else selected_detection.horizontal_yaw
            ),
            frame_change=self.progress.frame_change(self._previous_pov, pov),
            visible=selected_detection is not None,
        )
        self._last_progress_step = self.step

    def _start_local_reacquire(self, reason: str) -> None:
        self._local_trigger = reason
        turns = self.config.local_reacquire_turns
        self._local_actions = deque(
            [self.config.left_action] * turns
            + [self.config.right_action] * (2 * turns)
        )
        self._transition(SearchState.LOCAL_REACQUIRE, reason)

    def _start_recovery(self) -> None:
        self.recovery_count += 1
        self._recovery_actions = deque(
            [self.config.backward_action] * self.config.recovery_backward_steps
            + [self.config.right_action] * self.config.recovery_turn_steps
        )
        self._transition(SearchState.RECOVER, "local reacquire exhausted")

    def _finish_recovery(self) -> None:
        if self.selected_candidate is None:
            self._transition(SearchState.REPLAN, "recovery lost candidate identity")
            return
        candidate_id = self.selected_candidate.candidate_id
        self._recoveries_by_candidate[candidate_id] += 1
        self.selected_candidate.approach_attempts += 1
        self.progress.reset()
        self._transition(SearchState.ALIGN, "backoff and offset turn complete")

    def _mark_selected_for_replan(self) -> None:
        if self.selected_candidate is not None:
            self.selected_candidate.stalled_steps += self.progress.window_size
            self.candidate_map.mark_cooldown(
                self.selected_candidate, self.step, self.config.cooldown_steps
            )
        self.replan_count += 1
        self.selected_candidate = None
        self.progress.reset()

    def _return_action(self, action: int) -> int:
        self._last_action = int(action)
        self._last_action_state = self.state
        self.action_counts[int(action)] += 1
        return int(action)

    def act(self, observation: Dict[str, Any]) -> int:
        pov = self._pov(observation)
        telemetry = self._telemetry(observation)
        if telemetry is not None:
            self.current_telemetry = telemetry
            self.heading_yaw = telemetry.yaw
            self.candidate_map.refresh_world_bearings(telemetry)
        detections = self.adapter.detect(pov)
        selected_detection = self._selected_detection(detections)
        if selected_detection is not None:
            self._update_selected_bearing(selected_detection, telemetry)
        self._record_progress(pov, selected_detection)

        # BUILD/SELECT/REPLAN are explicit loggable states but do not consume a
        # Minecraft tick. The bound makes malformed transitions fail closed.
        for _ in range(16):
            if self.state == SearchState.SCAN:
                if self.scan_yaw + 1e-6 >= self.config.full_scan_degrees:
                    self.scan_cycles += 1
                    self._transition(
                        SearchState.BUILD_CANDIDATE_MAP, "full 360-degree scan complete"
                    )
                    continue
                self._ingest_scan_detections(detections, telemetry)
                self._previous_pov = pov.copy()
                return self._return_action(self.config.right_action)

            if self.state == SearchState.BUILD_CANDIDATE_MAP:
                consolidated = self.candidate_map.consolidate()
                self._transition(
                    SearchState.SELECT_TARGET,
                    "candidate map contains {} candidates after {} consolidation merges".format(
                        len(self.candidate_map.candidates), consolidated
                    ),
                )
                continue

            if self.state == SearchState.SELECT_TARGET:
                selection_rank = (
                    self.config.initial_selection_rank
                    if self.initial_selected_candidate_id is None else 0
                )
                selected = self.candidate_map.select(
                    self.heading_yaw, self.step, rank=selection_rank
                )
                if selected is None:
                    self.scan_yaw = 0.0
                    self._transition(
                        SearchState.SCAN,
                        "no available candidate; rescan while cooldowns advance",
                    )
                    continue
                self.selected_candidate = selected
                if self.initial_selected_candidate_id is None:
                    self.initial_selected_candidate_id = selected.candidate_id
                self.progress.reset()
                reason = (
                    "highest deployment score"
                    if selection_rank == 0
                    else "diagnostic forced score rank {}".format(selection_rank)
                )
                self._transition(SearchState.ALIGN, reason)
                continue

            if self.state == SearchState.ALIGN:
                if self.selected_candidate is None:
                    self._transition(SearchState.REPLAN, "missing selected candidate")
                    continue
                if selected_detection is not None:
                    yaw_error = selected_detection.horizontal_yaw
                else:
                    yaw_error = wrap_degrees(
                        self.selected_candidate.relative_yaw - self.heading_yaw
                    )
                if abs(yaw_error) <= self.config.align_threshold_degrees:
                    self.progress.reset()
                    self._transition(SearchState.APPROACH, "candidate aligned")
                    continue
                self._previous_pov = pov.copy()
                action = (
                    self.config.right_action if yaw_error > 0 else self.config.left_action
                )
                return self._return_action(action)

            if self.state == SearchState.APPROACH:
                world_route = self._world_route()
                if self.progress.is_stalled() and (
                    selected_detection is not None or world_route is None
                ):
                    self.stalled_count += 1
                    self._start_local_reacquire("visual progress window stalled")
                    continue
                if self.progress.target_lost() and world_route is None:
                    self._start_local_reacquire("candidate lost for consecutive frames")
                    continue
                if selected_detection is None:
                    if world_route is not None:
                        yaw_error, distance = world_route
                        uncertainty = float(
                            self.selected_candidate.position_uncertainty or 0.0
                        )
                        arrival_radius = max(
                            self.config.world_arrival_radius, uncertainty
                        )
                        if distance <= arrival_radius:
                            self._start_local_reacquire(
                                "predicted world coordinate reached without visual contact"
                            )
                            continue
                        if abs(yaw_error) > self.config.align_threshold_degrees:
                            action = (
                                self.config.right_action
                                if yaw_error > 0
                                else self.config.left_action
                            )
                        else:
                            action = self.config.forward_action
                        self._previous_pov = pov.copy()
                        return self._return_action(action)
                    self._previous_pov = pov.copy()
                    return self._return_action(self.config.right_action)
                if abs(selected_detection.horizontal_yaw) > self.config.align_threshold_degrees:
                    action = (
                        self.config.right_action
                        if selected_detection.horizontal_yaw > 0
                        else self.config.left_action
                    )
                elif self.adapter.ready_to_interact(selected_detection):
                    action = self.adapter.interaction_action()
                else:
                    action = self.config.forward_action
                self._previous_pov = pov.copy()
                return self._return_action(action)

            if self.state == SearchState.LOCAL_REACQUIRE:
                # Loss recovery may return as soon as the remembered object is
                # visible. A stall recovery deliberately completes the +/-30
                # sweep so a nearby false commitment cannot trap the policy.
                if selected_detection is not None and (
                    self._local_trigger.startswith("candidate lost")
                    or self._local_trigger.startswith("predicted world coordinate")
                ):
                    self.progress.reset()
                    self._transition(SearchState.ALIGN, "candidate locally reacquired")
                    continue
                if self._local_actions:
                    self._previous_pov = pov.copy()
                    return self._return_action(self._local_actions.popleft())
                candidate_id = (
                    None if self.selected_candidate is None else self.selected_candidate.candidate_id
                )
                if (
                    candidate_id is not None
                    and self._recoveries_by_candidate[candidate_id]
                    < self.config.max_recovery_attempts
                ):
                    self._start_recovery()
                else:
                    self._transition(SearchState.REPLAN, "candidate retry budget exhausted")
                continue

            if self.state == SearchState.RECOVER:
                if self._recovery_actions:
                    self._previous_pov = pov.copy()
                    return self._return_action(self._recovery_actions.popleft())
                self._finish_recovery()
                continue

            if self.state == SearchState.REPLAN:
                self._mark_selected_for_replan()
                available_world_candidates = any(
                    candidate.status == "available" and candidate.has_world_position
                    for candidate in self.candidate_map.candidates
                )
                if (
                    self.config.sensor_profile == SENSOR_PROFILE_F3
                    and available_world_candidates
                ):
                    self._transition(
                        SearchState.SELECT_TARGET,
                        "failed candidate cooling down; use remembered world coordinates",
                    )
                elif self.config.rescan_after_replan:
                    # Bearings remembered at reset become stale after walking
                    # to a distractor. Preserve failed/cooldown identities, but
                    # rebuild every untried bearing from the current location.
                    self.candidate_map.candidates = [
                        candidate
                        for candidate in self.candidate_map.candidates
                        if candidate.status in ("cooldown", "failed", "completed")
                    ]
                    self.scan_yaw = 0.0
                    self._transition(
                        SearchState.SCAN,
                        "failed candidate cooling down; rebuild map after translation",
                    )
                else:
                    self._transition(
                        SearchState.SELECT_TARGET, "failed candidate cooling down"
                    )
                continue

            if self.state in (SearchState.SUCCESS, SearchState.FAILED):
                self._previous_pov = pov.copy()
                return self._return_action(self.config.noop_action)

        self._transition(SearchState.FAILED, "internal transition bound exceeded")
        return self._return_action(self.config.noop_action)

    def observe_transition(
        self,
        action: int,
        observation: Dict[str, Any],
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ) -> None:
        """Update scan accounting, fallback yaw dead reckoning, and terminal state."""

        self.step += 1
        if int(action) == self.config.right_action:
            if self.config.sensor_profile != SENSOR_PROFILE_F3:
                self.heading_yaw = wrap_degrees(
                    self.heading_yaw + self.config.camera_delta
                )
            if self._last_action_state == SearchState.SCAN:
                self.scan_yaw += self.config.camera_delta
        elif int(action) == self.config.left_action:
            if self.config.sensor_profile != SENSOR_PROFILE_F3:
                self.heading_yaw = wrap_degrees(
                    self.heading_yaw - self.config.camera_delta
                )

        if self.adapter.success(observation, reward, info):
            if self.selected_candidate is not None:
                self.selected_candidate.status = "completed"
            self._transition(SearchState.SUCCESS, "resource success signal")
            self._finished = True
        elif done:
            self._transition(SearchState.FAILED, "environment terminated without success")
            self._finished = True

    @property
    def terminal(self) -> bool:
        return self.state in (SearchState.SUCCESS, SearchState.FAILED)
