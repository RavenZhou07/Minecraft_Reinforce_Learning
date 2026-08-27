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
    RaycastHit,
    SENSOR_PROFILE_RAYCAST,
    SENSOR_PROFILES,
    bearing_and_distance_to,
    detection_world_position,
    sensor_uses_telemetry,
)
from mc_rl.trunk_contact import (
    CONTACT_PROFILES,
    CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
    CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
    CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
    CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
    CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
    CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
    CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
    CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10,
    CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11,
    CONTACT_PROFILE_V6_1,
    TrunkContactConfig,
    TrunkContactController,
    TrunkContactState,
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
    fine_left_action: int = 10
    fine_right_action: int = 11
    fine_look_up_action: int = 12
    fine_look_down_action: int = 13
    contact_attack_action: int = 7
    # Non-zero values are only for labelled fault-injection diagnostics. The
    # deployment default always chooses score rank zero.
    initial_selection_rank: int = 0
    rescan_after_replan: bool = True
    sensor_profile: str = "pov_only"
    world_arrival_radius: float = 2.5
    position_update_interval: int = 5
    # Once inside this radius of the estimated coordinate, the coarse world
    # route hands over to the vision-only trunk contact controller.
    enable_trunk_contact: bool = True
    contact_region_radius: float = 6.0
    route_obstacle_jump_steps: int = 3
    route_obstacle_turn_steps: int = 2
    max_route_obstacle_recoveries: int = 2
    # v6.1 remains executable as a frozen A/B baseline. New experiments opt
    # into the bounded canopy-clearing profile explicitly.
    contact_profile: str = CONTACT_PROFILE_V6_1
    handoff_uncertainty_cap: float = 2.0
    handoff_visual_alignment_degrees: float = 12.0
    handoff_raycast_max_distance: float = 14.0
    episode_max_steps: int = 300
    handoff_relocalization_reserve_steps: int = 56
    # v9.8 privileged teacher: a visual trunk may guide navigation but only
    # an exact raycast log can transfer control to the contact controller.
    require_raycast_handoff_confirmation: bool = False
    # v9.9: once the privileged scan has both a visual candidate and an exact
    # log coordinate, further panorama actions only spend episode budget.
    enable_early_exact_log_scan_exit: bool = False
    # v9.6 terrain route recovery: coarse world routes across staircased
    # terrain receive bounded forward-jump assistance that is verified by
    # real route-distance reduction, and previously route-blocked physical
    # regions are remembered so the same walk is not replayed forever.
    enable_terrain_route_recovery: bool = False
    terrain_route_progress_window: int = 10
    terrain_route_progress_minimum: float = 0.5
    terrain_route_climb_steps: int = 6
    terrain_route_recovery_max_attempts: int = 6
    terrain_route_recovery_failure_limit: int = 2
    terrain_route_recovery_success_progress: float = 0.5
    terrain_route_blocked_region_radius: float = 6.0


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
        if self.config.contact_profile not in CONTACT_PROFILES:
            raise ValueError(
                "unknown contact profile: {}".format(
                    self.config.contact_profile
                )
            )
        self._candidate_handoff_guard = bool(
            self.config.contact_profile
            in (
                CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
                CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
                CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
                CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
                CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
                CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
                CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
                CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10,
                CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11,
            )
        )
        self._contact_ownership_guard = bool(
            self.config.contact_profile
            in (
                CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
                CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
                CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
                CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
                CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
                CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10,
                CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11,
            )
        )
        self._terrain_route_recovery = bool(
            self.config.enable_terrain_route_recovery
            or self.config.contact_profile
            in (
                CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
                CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
                CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
                CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
                CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10,
                CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11,
            )
        )
        self._raycast_owned_handoff = bool(
            self.config.require_raycast_handoff_confirmation
            or self.config.contact_profile
            in (
                CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
                CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
                CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10,
                CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11,
            )
        )
        self._early_exact_log_scan_exit = bool(
            self.config.enable_early_exact_log_scan_exit
            or self.config.contact_profile
            in (
                CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
                CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10,
                CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11,
            )
        )
        self._dynamic_exact_log_route = bool(
            self.config.contact_profile
            in (
                CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
                CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10,
                CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11,
            )
        )
        self._terrain_route_failure_limit = int(
            1
            if self.config.contact_profile
            in (
                CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
                CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10,
                CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11,
            )
            else self.config.terrain_route_recovery_failure_limit
        )
        self.candidate_map = candidate_map or CandidateMap()
        if self._candidate_handoff_guard:
            self.candidate_map.cooldown_yaw_only_merge = False
        self.progress = progress_monitor or VisualProgressMonitor()
        self._contact: Optional[TrunkContactController] = None
        if self.config.enable_trunk_contact and hasattr(adapter, "trunk_view"):
            contact_config = TrunkContactConfig.for_profile(
                self.config.contact_profile,
                noop_action=self.config.noop_action,
                forward_action=self.config.forward_action,
                forward_jump_action=self.config.forward_jump_action,
                left_action=self.config.left_action,
                right_action=self.config.right_action,
                fine_left_action=self.config.fine_left_action,
                fine_right_action=self.config.fine_right_action,
                fine_look_up_action=self.config.fine_look_up_action,
                fine_look_down_action=self.config.fine_look_down_action,
                attack_action=self.config.contact_attack_action,
                clear_occlusion_action=self.adapter.interaction_action(),
                backward_action=self.config.backward_action,
            )
            self._contact = TrunkContactController(adapter, contact_config)
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
        self.obstacle_recovery_count = 0
        self.handoff_guard_checks = 0
        self.handoff_guard_rejections = 0
        self.handoff_visual_confirmations = 0
        self.handoff_raycast_confirmations = 0
        self.handoff_raycast_memory_confirmations = 0
        self.raycast_memory_route_selections = 0
        self.exact_log_early_scan_exits = 0
        self.dynamic_exact_route_updates = 0
        self.handoff_spatial_rejections = 0
        self.handoff_relocalization_scans = 0
        self.handoff_relocalization_skipped_late = 0
        self.suppressed_contact_position_updates = 0
        self.contact_owner_lock_steps = 0
        self.contact_owner_mismatches = 0
        self.last_action_source = "global"
        self.terrain_route_recovery_attempts = 0
        self.terrain_route_recovery_steps = 0
        self.terrain_route_recovery_successes = 0
        self.terrain_route_recovery_failures = 0
        self.repeated_physical_region_route_rejections = 0
        self.terrain_route_recovery_records: List[Dict[str, Any]] = []
        self._route_blocked_regions: List[Tuple[float, float]] = []
        self._previous_pov: Optional[np.ndarray] = None
        self._last_action: Optional[int] = None
        self._last_action_state = self.state
        self._last_progress_step = -1
        self._local_actions: Deque[int] = deque()
        self._recovery_actions: Deque[int] = deque()
        self._recoveries_by_candidate: Counter = Counter()
        self._local_trigger = ""
        self._route_distances: Deque[float] = deque(maxlen=24)
        self._obstacle_actions: Deque[int] = deque()
        self._obstacle_attempts_by_candidate: Counter = Counter()
        self._route_target_candidate_id: Optional[int] = None
        self._route_target_x: Optional[float] = None
        self._route_target_z: Optional[float] = None
        self._route_target_uncertainty: Optional[float] = None
        self._terrain_climb_actions: Deque[int] = deque()
        self._terrain_progress_distances: Deque[float] = deque(
            maxlen=self.config.terrain_route_progress_window
        )
        self._terrain_climb_bursts = 0
        self._terrain_climb_failures = 0
        self._terrain_climb_disabled = False
        self._terrain_climb_start: Optional[Dict[str, float]] = None
        self._force_handoff_relocalization = False
        self._finished = False
        self.current_telemetry: Optional[AgentTelemetry] = None
        # A policy object is reusable across episodes, but candidate identities
        # are deliberately episode-local.
        self.candidate_map = CandidateMap(
            merge_yaw_degrees=self.candidate_map.merge_yaw_degrees,
            merge_log_size_tolerance=self.candidate_map.merge_log_size_tolerance,
            world_merge_min_distance=self.candidate_map.world_merge_min_distance,
            position_uncertainty_floor=(
                self.candidate_map.position_uncertainty_floor
            ),
            cooldown_yaw_only_merge=(
                False
                if self._candidate_handoff_guard
                else self.candidate_map.cooldown_yaw_only_merge
            ),
            score_config=self.candidate_map.score_config,
        )
        if self._contact is not None:
            self._contact.start()
        self.progress.reset()

    @staticmethod
    def _pov(observation: Dict[str, Any]) -> np.ndarray:
        # Deliberately do not iterate the observation mapping: a guarded test
        # can prove that an ``oracle`` sibling is never touched.
        return np.asarray(observation["pov"])

    def _telemetry(
        self, observation: Dict[str, Any]
    ) -> Optional[AgentTelemetry]:
        if not sensor_uses_telemetry(self.config.sensor_profile):
            return None
        # Access the explicitly declared telemetry sibling directly. Never
        # iterate the mapping, which also keeps oracle isolation testable.
        return AgentTelemetry.from_observation(observation["telemetry"])

    def _raycast(
        self, observation: Dict[str, Any]
    ) -> Optional[RaycastHit]:
        if self.config.sensor_profile != SENSOR_PROFILE_RAYCAST:
            return None
        # This is the only privileged deployment path and is selected by an
        # explicit diagnostic sensor profile. It never exists in POV/F3 runs.
        return RaycastHit.from_observation(observation["raycast"])

    def _transition(self, new_state: SearchState, reason: str) -> None:
        if new_state == self.state:
            return
        old_state = self.state
        self.state = new_state
        selected = self.selected_candidate
        self.transition_log.append(
            {
                "episode": self.episode,
                "step": self.step,
                "old_state": old_state.value,
                "new_state": new_state.value,
                "reason": reason,
                "selected_candidate_id": (
                    "" if selected is None else selected.candidate_id
                ),
                "estimated_world_x": (
                    ""
                    if selected is None or not selected.has_world_position
                    else round(float(selected.estimated_world_x), 3)
                ),
                "estimated_world_z": (
                    ""
                    if selected is None or not selected.has_world_position
                    else round(float(selected.estimated_world_z), 3)
                ),
                "position_uncertainty": (
                    ""
                    if selected is None or not selected.has_world_position
                    else round(float(selected.position_uncertainty), 3)
                ),
                "route_target_x": (
                    ""
                    if self._route_target_x is None
                    else round(float(self._route_target_x), 3)
                ),
                "route_target_z": (
                    ""
                    if self._route_target_z is None
                    else round(float(self._route_target_z), 3)
                ),
                "route_target_uncertainty": (
                    ""
                    if self._route_target_uncertainty is None
                    else round(float(self._route_target_uncertainty), 3)
                ),
                "heading_yaw": round(float(self.heading_yaw), 2),
                "trunk_contact_state": self.contact_state or "",
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
        self,
        detections: Sequence[ResourceDetection],
        telemetry: Optional[AgentTelemetry],
    ) -> Optional[ResourceDetection]:
        if self.selected_candidate is None:
            return None
        matches = []
        for detection_index, detection in enumerate(detections):
            world_yaw = wrap_degrees(self.heading_yaw + detection.horizontal_yaw)
            error = self.candidate_map.angular_distance(
                world_yaw, self.selected_candidate.relative_yaw
            )
            if error > self.config.selected_match_degrees:
                continue
            position_error = 0.0
            if telemetry is not None and self.selected_candidate.has_world_position:
                range_estimate = self.adapter.estimate_range(detection)
                if range_estimate is not None:
                    observed_position = detection_world_position(
                        telemetry, detection.horizontal_yaw, range_estimate
                    )
                    position_error = float(
                        np.hypot(
                            float(self.selected_candidate.estimated_world_x)
                            - observed_position[0],
                            float(self.selected_candidate.estimated_world_z)
                            - observed_position[2],
                        )
                    )
                    position_gate = max(
                        3.0,
                        float(self.selected_candidate.position_uncertainty)
                        + range_estimate.uncertainty
                        + 1.0,
                    )
                    if position_error > position_gate:
                        continue
            matches.append(
                (
                    error,
                    position_error,
                    -detection.apparent_size,
                    detection_index,
                    detection,
                )
            )
        return min(matches)[-1] if matches else None

    def _update_selected_bearing(
        self,
        detection: ResourceDetection,
        telemetry: Optional[AgentTelemetry],
    ) -> None:
        if self.selected_candidate is None:
            return
        if (
            self._candidate_handoff_guard
            and self._contact is not None
            and self._contact.engaged
        ):
            # Contact camera motion can expose unrelated brown components.
            # Never let those frames drag the selected world coordinate.
            self.suppressed_contact_position_updates += 1
            return
        range_estimate = (
            None if telemetry is None else self.adapter.estimate_range(detection)
        )
        position_update_due = bool(
            self.selected_candidate.last_position_update_step < 0
            or self.step - self.selected_candidate.last_position_update_step
            >= self.config.position_update_interval
        )
        if (
            telemetry is not None
            and range_estimate is not None
            and position_update_due
        ):
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

    def _capture_route_target(self) -> None:
        candidate = self.selected_candidate
        if candidate is None or not candidate.has_world_position:
            self._route_target_candidate_id = None
            self._route_target_x = None
            self._route_target_z = None
            self._route_target_uncertainty = None
            return
        self._route_target_candidate_id = int(candidate.candidate_id)
        self._route_target_x = float(candidate.estimated_world_x)
        self._route_target_z = float(candidate.estimated_world_z)
        self._route_target_uncertainty = float(
            candidate.position_uncertainty or 0.0
        )

    def _clear_route_target(self) -> None:
        self._route_target_candidate_id = None
        self._route_target_x = None
        self._route_target_z = None
        self._route_target_uncertainty = None

    def _capture_raycast_memory_route(self) -> bool:
        """Let a scan-verified exact log own the coarse v9.8 route."""

        if (
            not self._raycast_owned_handoff
            or self._contact is None
            or self.current_telemetry is None
            or self.selected_candidate is None
        ):
            return False
        route = self._contact.nearest_remembered_log_route(
            self.current_telemetry, self.step
        )
        if route is None:
            return False
        changed = bool(
            self._route_target_x != route[0]
            or self._route_target_z != route[1]
            or self._route_target_uncertainty != 0.0
        )
        self._route_target_candidate_id = self.selected_candidate.candidate_id
        self._route_target_x = route[0]
        self._route_target_z = route[1]
        self._route_target_uncertainty = 0.0
        if changed:
            self.raycast_memory_route_selections += 1
        return True

    def _refresh_dynamic_exact_log_route(self) -> None:
        """Adopt a log first observed after the initial panorama."""

        if (
            not self._dynamic_exact_log_route
            or self.selected_candidate is None
            or self.current_telemetry is None
            or self.state
            not in (
                SearchState.ALIGN,
                SearchState.APPROACH,
                SearchState.LOCAL_REACQUIRE,
            )
        ):
            return
        previous = (
            self._route_target_x,
            self._route_target_z,
            self._route_target_uncertainty,
        )
        if not self._capture_raycast_memory_route():
            return
        current = (
            self._route_target_x,
            self._route_target_z,
            self._route_target_uncertainty,
        )
        if current != previous:
            self.dynamic_exact_route_updates += 1
            self._route_distances.clear()
            if self._terrain_route_recovery:
                self._reset_terrain_climb_state()

    def _inside_route_blocked_region(self, x: float, z: float) -> bool:
        if not self._route_blocked_regions:
            return False
        radius = self.config.terrain_route_blocked_region_radius
        return any(
            float(np.hypot(x - region_x, z - region_z)) <= radius
            for region_x, region_z in self._route_blocked_regions
        )

    def _record_route_blocked_region(self) -> None:
        if self._route_target_x is None or self._route_target_z is None:
            return
        x = float(self._route_target_x)
        z = float(self._route_target_z)
        if self._inside_route_blocked_region(x, z):
            return
        self._route_blocked_regions.append((x, z))

    def _reset_terrain_climb_state(self) -> None:
        self._terrain_climb_actions.clear()
        self._terrain_progress_distances.clear()
        self._terrain_climb_bursts = 0
        self._terrain_climb_failures = 0
        self._terrain_climb_disabled = False
        self._terrain_climb_start = None

    def _verify_terrain_climb_burst(
        self, world_route: Tuple[float, float]
    ) -> None:
        """Grade the last completed climb burst by real route progress."""

        if self._terrain_climb_start is None or self._terrain_climb_actions:
            return
        start = self._terrain_climb_start
        self._terrain_climb_start = None
        self._terrain_progress_distances.clear()
        telemetry = self.current_telemetry
        route_distance = float(world_route[1])
        progress = float(start["route_distance"]) - route_distance
        horizontal = None
        vertical = None
        end_x = end_y = end_z = None
        if telemetry is not None and start["x"] is not None:
            end_x = float(telemetry.x)
            end_y = float(telemetry.y)
            end_z = float(telemetry.z)
            horizontal = float(
                np.hypot(end_x - float(start["x"]), end_z - float(start["z"]))
            )
            vertical = end_y - float(start["y"])
        candidate_id = (
            None
            if self.selected_candidate is None
            else self.selected_candidate.candidate_id
        )
        outcome = (
            "success"
            if progress >= self.config.terrain_route_recovery_success_progress
            else "failure"
        )
        self.terrain_route_recovery_records.append(
            {
                "step": int(start["step"]),
                "candidate_id": "" if candidate_id is None else candidate_id,
                "start_x": start["x"],
                "start_y": start["y"],
                "start_z": start["z"],
                "end_x": end_x,
                "end_y": end_y,
                "end_z": end_z,
                "horizontal_displacement": horizontal,
                "vertical_displacement": vertical,
                "route_distance_start": float(start["route_distance"]),
                "route_distance_end": route_distance,
                "outcome": outcome,
            }
        )
        if outcome == "success":
            self.terrain_route_recovery_successes += 1
            self._terrain_climb_failures = 0
        else:
            self.terrain_route_recovery_failures += 1
            self._terrain_climb_failures += 1
            if (
                self._terrain_climb_failures
                >= self._terrain_route_failure_limit
            ):
                self._terrain_climb_disabled = True

    def _terrain_route_climb_action(
        self, world_route: Tuple[float, float]
    ) -> Optional[int]:
        """Return one bounded forward-jump when the route advances too slowly."""

        if not self._terrain_route_recovery or self._terrain_climb_disabled:
            return None
        if self._terrain_climb_actions:
            self.terrain_route_recovery_steps += 1
            return self._terrain_climb_actions.popleft()
        if (
            self._terrain_climb_bursts
            >= self.config.terrain_route_recovery_max_attempts
        ):
            return None
        self._terrain_progress_distances.append(float(world_route[1]))
        if (
            len(self._terrain_progress_distances)
            < self.config.terrain_route_progress_window
        ):
            return None
        window_progress = (
            self._terrain_progress_distances[0]
            - min(self._terrain_progress_distances)
        )
        if window_progress >= self.config.terrain_route_progress_minimum:
            return None
        # Plain forward barely reduced the route distance: the direct walk
        # is meeting staircased terrain. Arm a short jump burst and verify it
        # by the next real route-distance reading instead of replaying the
        # same blocked walk until the stall detector fires.
        self._terrain_progress_distances.clear()
        self._terrain_climb_bursts += 1
        self.terrain_route_recovery_attempts += 1
        telemetry = self.current_telemetry
        self._terrain_climb_start = {
            "step": self.step,
            "route_distance": float(world_route[1]),
            "x": None if telemetry is None else float(telemetry.x),
            "y": None if telemetry is None else float(telemetry.y),
            "z": None if telemetry is None else float(telemetry.z),
        }
        self._terrain_climb_actions.extend(
            [self.config.forward_jump_action]
            * self.config.terrain_route_climb_steps
        )
        self.terrain_route_recovery_steps += 1
        return self._terrain_climb_actions.popleft()

    def _world_route(self) -> Optional[Tuple[float, float]]:
        if (
            self.current_telemetry is None
            or self.selected_candidate is None
            or not self.selected_candidate.has_world_position
        ):
            return None
        if (
            self._candidate_handoff_guard
            and self._route_target_candidate_id
            != self.selected_candidate.candidate_id
        ):
            self._capture_route_target()
        target_x = float(self.selected_candidate.estimated_world_x)
        target_z = float(self.selected_candidate.estimated_world_z)
        if (
            self._candidate_handoff_guard
            and self._route_target_x is not None
            and self._route_target_z is not None
        ):
            target_x = self._route_target_x
            target_z = self._route_target_z
        bearing, distance = bearing_and_distance_to(
            self.current_telemetry, target_x, target_z
        )
        return wrap_degrees(bearing - self.current_telemetry.yaw), distance

    def _contact_region_limit(self) -> float:
        uncertainty = float(
            0.0
            if self.selected_candidate is None
            else self.selected_candidate.position_uncertainty or 0.0
        )
        if self._candidate_handoff_guard:
            if self._route_target_uncertainty is not None:
                uncertainty = self._route_target_uncertainty
            uncertainty = min(uncertainty, self.config.handoff_uncertainty_cap)
        return self.config.contact_region_radius + uncertainty

    def _local_handoff_evidence(
        self,
        selected_detection: Optional[ResourceDetection],
        raycast: Optional[RaycastHit],
    ) -> Optional[str]:
        if (
            raycast is not None
            and raycast.is_log
            and raycast.distance <= self.config.handoff_raycast_max_distance
        ):
            return "raycast"
        if self._raycast_owned_handoff:
            return None
        if (
            selected_detection is not None
            and bool(getattr(selected_detection, "sees_trunk", False))
            and abs(float(selected_detection.horizontal_yaw))
            <= self.config.handoff_visual_alignment_degrees
        ):
            return "visual_trunk"
        return None

    def _note_contact_replan(self) -> None:
        if not self._candidate_handoff_guard or self._contact is None:
            return
        attempts = self._contact.diagnostics().get("attempt_results", [])
        reason = "" if not attempts else str(attempts[-1].get("reason", ""))
        bad_handoff_reasons = (
            "exact log rescan",
            "no remembered 3-D log target",
            "no eligible local log",
            "post-recovery translation budget exhausted without progress",
            "all remembered 3-D log targets are cooling down",
        )
        if any(token in reason for token in bad_handoff_reasons):
            self._force_handoff_relocalization = True

    def _contact_step(
        self,
        pov: np.ndarray,
        telemetry: Optional[AgentTelemetry],
        world_route: Optional[Tuple[float, float]],
        selected_detection: Optional[ResourceDetection],
        raycast: Optional[RaycastHit] = None,
        route_stalled: bool = False,
    ) -> Tuple[Optional[int], bool]:
        """Delegate the terminal approach to the trunk contact controller.

        Returns ``(action, engaged)``; the caller must handle a ``replan``
        result through the normal REPLAN bookkeeping.
        """

        if self._contact is None or self.selected_candidate is None:
            return None, False
        within_region = bool(
            world_route is not None
            and world_route[1] <= self._contact_region_limit()
        )
        visual_close = bool(
            selected_detection is not None
            and self.adapter.ready_to_interact(selected_detection)
        )
        evidence = None
        if self._candidate_handoff_guard and not self._contact.active:
            self.handoff_guard_checks += 1
            evidence = self._local_handoff_evidence(selected_detection, raycast)
            if (
                evidence is None
                and self._raycast_owned_handoff
                and self._contact.has_reachable_remembered_log_target(
                    telemetry, self.step
                )
            ):
                evidence = "raycast_memory"
            if self._contact_ownership_guard:
                candidate_local = bool(
                    within_region
                    or evidence in ("raycast", "raycast_memory")
                )
                if not candidate_local:
                    self.handoff_spatial_rejections += 1
                    self.handoff_guard_rejections += 1
                    return None, False
            if evidence is None:
                self.handoff_guard_rejections += 1
                return None, False
            if evidence == "raycast":
                self.handoff_raycast_confirmations += 1
            elif evidence == "raycast_memory":
                self.handoff_raycast_memory_confirmations += 1
            else:
                self.handoff_visual_confirmations += 1
        candidate_local = bool(
            self._contact.active
            or within_region
            or (visual_close and not self._contact_ownership_guard)
            or (
                self._contact_ownership_guard
                and evidence in ("raycast", "raycast_memory")
            )
            or (route_stalled and not self._contact_ownership_guard)
        )
        if not candidate_local:
            return None, False
        if not self._contact.active:
            target_hint = None
            if self.selected_candidate.has_world_position:
                target_hint = (
                    float(
                        self._route_target_x
                        if self._candidate_handoff_guard
                        and self._route_target_x is not None
                        else self.selected_candidate.estimated_world_x
                    ),
                    float(
                        self._route_target_z
                        if self._candidate_handoff_guard
                        and self._route_target_z is not None
                        else self.selected_candidate.estimated_world_z
                    ),
                )
            self._contact.engage(
                telemetry=telemetry,
                find_first=route_stalled and not within_region,
                candidate_id=self.selected_candidate.candidate_id,
                global_step=self.step,
                target_hint=target_hint,
            )
        action = self._contact.act(
            pov,
            telemetry,
            world_route,
            global_step=self.step,
            raycast=raycast,
        )
        return int(action), True

    @property
    def contact_state(self) -> Optional[str]:
        if self._contact is None or not self._contact.active:
            return None
        return self._contact.state.value

    def set_external_attack_permission(
        self, permission: Optional[bool]
    ) -> None:
        """Provide one-step visual attack permission to the contact owner."""

        if self._contact is not None:
            self._contact.set_external_attack_permission(permission)

    def contact_diagnostics(self) -> Dict[str, Any]:
        if self._contact is None:
            return {}
        return self._contact.diagnostics()

    def handoff_diagnostics(self) -> Dict[str, Any]:
        """Expose upstream handoff state without leaking evaluation truth."""

        return {
            "enabled": self._candidate_handoff_guard,
            "route_target_candidate_id": self._route_target_candidate_id,
            "route_target_x": self._route_target_x,
            "route_target_z": self._route_target_z,
            "route_target_uncertainty": self._route_target_uncertainty,
            "contact_region_limit": self._contact_region_limit(),
            "checks": self.handoff_guard_checks,
            "rejections": self.handoff_guard_rejections,
            "visual_confirmations": self.handoff_visual_confirmations,
            "raycast_confirmations": self.handoff_raycast_confirmations,
            "raycast_memory_confirmations": (
                self.handoff_raycast_memory_confirmations
            ),
            "raycast_memory_route_selections": (
                self.raycast_memory_route_selections
            ),
            "exact_log_early_scan_exits": self.exact_log_early_scan_exits,
            "dynamic_exact_route_updates": self.dynamic_exact_route_updates,
            "spatial_rejections": self.handoff_spatial_rejections,
            "relocalization_scans": self.handoff_relocalization_scans,
            "relocalization_skipped_late": (
                self.handoff_relocalization_skipped_late
            ),
            "suppressed_contact_position_updates": (
                self.suppressed_contact_position_updates
            ),
            "contact_owner_lock_steps": self.contact_owner_lock_steps,
            "contact_owner_mismatches": self.contact_owner_mismatches,
            "terrain_route_recovery_enabled": self._terrain_route_recovery,
            "terrain_route_failure_limit": self._terrain_route_failure_limit,
            "terrain_route_recovery_attempts": (
                self.terrain_route_recovery_attempts
            ),
            "terrain_route_recovery_steps": (
                self.terrain_route_recovery_steps
            ),
            "terrain_route_recovery_successes": (
                self.terrain_route_recovery_successes
            ),
            "terrain_route_recovery_failures": (
                self.terrain_route_recovery_failures
            ),
            "repeated_physical_region_route_rejections": (
                self.repeated_physical_region_route_rejections
            ),
            "route_blocked_regions": list(self._route_blocked_regions),
        }

    def search_diagnostics(self) -> Dict[str, Any]:
        """Expose deployment-side search timing without evaluation oracles."""

        selected = self.selected_candidate
        route_yaw_error = None
        route_distance = None
        target_x = self._route_target_x
        target_z = self._route_target_z
        if (
            (target_x is None or target_z is None)
            and selected is not None
            and selected.has_world_position
        ):
            target_x = float(selected.estimated_world_x)
            target_z = float(selected.estimated_world_z)
        if (
            self.current_telemetry is not None
            and target_x is not None
            and target_z is not None
        ):
            bearing, route_distance = bearing_and_distance_to(
                self.current_telemetry, float(target_x), float(target_z)
            )
            route_yaw_error = wrap_degrees(
                bearing - self.current_telemetry.yaw
            )
        candidates = []
        for candidate in self.candidate_map.candidates:
            candidates.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "status": candidate.status,
                    "score": candidate.score,
                    "relative_yaw": candidate.relative_yaw,
                    "apparent_size": candidate.apparent_size,
                    "trunk_observations": candidate.trunk_observations,
                    "canopy_observations": candidate.canopy_observations,
                    "approach_attempts": candidate.approach_attempts,
                    "stalled_steps": candidate.stalled_steps,
                    "failed_until_step": candidate.failed_until_step,
                    "estimated_world_x": candidate.estimated_world_x,
                    "estimated_world_z": candidate.estimated_world_z,
                    "position_uncertainty": candidate.position_uncertainty,
                    "range_estimate": candidate.range_estimate,
                }
            )
        return {
            "state": self.state.value,
            "step": self.step,
            "remaining_steps": max(0, self.config.episode_max_steps - self.step),
            "last_action_source": self.last_action_source,
            "selected_candidate_id": (
                None if selected is None else selected.candidate_id
            ),
            "selected_candidate_status": (
                None if selected is None else selected.status
            ),
            "route_target_x": target_x,
            "route_target_z": target_z,
            "route_yaw_error": route_yaw_error,
            "route_distance": route_distance,
            "contact_region_limit": self._contact_region_limit(),
            "scan_yaw": self.scan_yaw,
            "scan_cycles": self.scan_cycles,
            "exact_log_early_scan_exits": self.exact_log_early_scan_exits,
            "dynamic_exact_route_updates": self.dynamic_exact_route_updates,
            "candidate_count": len(self.candidate_map.candidates),
            "replan_count": self.replan_count,
            "recovery_count": self.recovery_count,
            "stalled_count": self.stalled_count,
            "obstacle_recovery_count": self.obstacle_recovery_count,
            "local_reacquire_actions_remaining": len(self._local_actions),
            "recovery_actions_remaining": len(self._recovery_actions),
            "obstacle_actions_remaining": len(self._obstacle_actions),
            "terrain_climb_actions_remaining": len(self._terrain_climb_actions),
            "handoff": self.handoff_diagnostics(),
            "candidates": candidates,
            "transitions": list(self.transition_log),
        }

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
        self._clear_route_target()
        self.progress.reset()

    def _return_action(self, action: int) -> int:
        self._last_action = int(action)
        self._last_action_state = self.state
        self.action_counts[int(action)] += 1
        return int(action)

    def act(self, observation: Dict[str, Any]) -> int:
        # Purely observational: records which controller produced this step's
        # action. It never influences the action itself, so every frozen
        # profile's behaviour is unchanged; behaviour cloning collectors use
        # it to delimit contact-owner trajectories.
        self.last_action_source = "global"
        pov = self._pov(observation)
        telemetry = self._telemetry(observation)
        raycast = self._raycast(observation)
        if telemetry is not None:
            self.current_telemetry = telemetry
            self.heading_yaw = telemetry.yaw
            self.candidate_map.refresh_world_bearings(telemetry)
        if self._contact is not None:
            self._contact.observe_raycast_target(raycast, telemetry, self.step)
        self._refresh_dynamic_exact_log_route()
        detections = self.adapter.detect(pov)
        selected_detection = self._selected_detection(detections, telemetry)
        if selected_detection is not None:
            self._update_selected_bearing(selected_detection, telemetry)
        self._record_progress(pov, selected_detection)

        # BUILD/SELECT/REPLAN are explicit loggable states but do not consume a
        # Minecraft tick. The bound makes malformed transitions fail closed.
        for _ in range(16):
            if (
                self._contact_ownership_guard
                and self._contact is not None
                and self._contact.active
            ):
                selected_id = (
                    None
                    if self.selected_candidate is None
                    else self.selected_candidate.candidate_id
                )
                if selected_id != self._contact.candidate_id:
                    self.contact_owner_mismatches += 1
                    self._contact.cancel(
                        "contact owner mismatch between local and global policy"
                    )
                    self._note_contact_replan()
                    self._transition(
                        SearchState.REPLAN,
                        "contact owner invariant violated; cancel and replan",
                    )
                    continue
                self.contact_owner_lock_steps += 1
                contact_action, contact_engaged = self._contact_step(
                    pov,
                    telemetry,
                    self._world_route(),
                    selected_detection,
                    raycast=raycast,
                )
                if contact_engaged:
                    if self._contact.result == "replan":
                        self._note_contact_replan()
                        self._transition(
                            SearchState.REPLAN,
                            "contact owner released after local replan",
                        )
                        continue
                    self._previous_pov = pov.copy()
                    self.last_action_source = "contact"
                    return self._return_action(contact_action)

            if self.state == SearchState.SCAN:
                if self._early_exact_log_scan_exit:
                    self._ingest_scan_detections(detections, telemetry)
                    exact_route = (
                        None
                        if self._contact is None or telemetry is None
                        else self._contact.nearest_remembered_log_route(
                            telemetry, self.step
                        )
                    )
                    if self.candidate_map.candidates and exact_route is not None:
                        self.exact_log_early_scan_exits += 1
                        self._transition(
                            SearchState.BUILD_CANDIDATE_MAP,
                            "visual candidate and exact log route discovered; end scan early",
                        )
                        continue
                if self.scan_yaw + 1e-6 >= self.config.full_scan_degrees:
                    self.scan_cycles += 1
                    self._transition(
                        SearchState.BUILD_CANDIDATE_MAP, "full 360-degree scan complete"
                    )
                    continue
                if not self._early_exact_log_scan_exit:
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
                if self._terrain_route_recovery and self._route_blocked_regions:
                    for candidate in self.candidate_map.candidates:
                        if (
                            candidate.status == "available"
                            and candidate.has_world_position
                            and self._inside_route_blocked_region(
                                float(candidate.estimated_world_x),
                                float(candidate.estimated_world_z),
                            )
                        ):
                            # The world route into this physical region was
                            # already blocked twice without climb progress:
                            # cool it down instead of replaying the same walk.
                            candidate.status = "cooldown"
                            candidate.failed_until_step = max(
                                candidate.failed_until_step,
                                self.step + self.config.cooldown_steps,
                            )
                            self.repeated_physical_region_route_rejections += 1
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
                self._capture_route_target()
                self._capture_raycast_memory_route()
                self._route_distances.clear()
                if self._terrain_route_recovery:
                    self._reset_terrain_climb_state()
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
                world_route = self._world_route()
                if world_route is not None:
                    # A remembered world coordinate is stable across frames;
                    # a newly associated RGB component is not.  Preferring
                    # the latter caused a 10-degree left/right limit cycle
                    # when neighbouring trunk/leaf components alternated.
                    yaw_error = world_route[0]
                elif selected_detection is not None:
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
                if self._obstacle_actions:
                    self._previous_pov = pov.copy()
                    return self._return_action(self._obstacle_actions.popleft())
                route_stalled = False
                if world_route is not None:
                    if self._last_action in (
                        self.config.forward_action,
                        self.config.forward_jump_action,
                    ):
                        self._route_distances.append(float(world_route[1]))
                    if (
                        len(self._route_distances)
                        == self._route_distances.maxlen
                        and min(list(self._route_distances)[:12])
                        - min(list(self._route_distances)[12:])
                        < 0.5
                    ):
                        route_stalled = True
                if (
                    route_stalled
                    and world_route is not None
                    and self.selected_candidate is not None
                ):
                    uncertainty = float(
                        self.selected_candidate.position_uncertainty or 0.0
                    )
                    outside_contact_region = bool(
                        world_route[1]
                        > (
                            self._contact_region_limit()
                            if self._candidate_handoff_guard
                            else self.config.contact_region_radius + uncertainty
                        )
                    )
                    candidate_id = self.selected_candidate.candidate_id
                    attempts = self._obstacle_attempts_by_candidate[candidate_id]
                    if (
                        outside_contact_region
                        and attempts
                        < self.config.max_route_obstacle_recoveries
                    ):
                        # A direct coordinate route can meet a one-block ledge
                        # or trunk. Try one short jump-and-offset manoeuvre
                        # before concluding that the candidate is unreachable.
                        turn_action = (
                            self.config.right_action
                            if attempts % 2 == 0
                            else self.config.left_action
                        )
                        self._obstacle_actions.extend(
                            [self.config.forward_jump_action]
                            * self.config.route_obstacle_jump_steps
                            + [turn_action]
                            * self.config.route_obstacle_turn_steps
                            + [self.config.forward_jump_action]
                            * self.config.route_obstacle_jump_steps
                        )
                        self._obstacle_attempts_by_candidate[candidate_id] += 1
                        self.obstacle_recovery_count += 1
                        self._route_distances.clear()
                        self._previous_pov = pov.copy()
                        return self._return_action(
                            self._obstacle_actions.popleft()
                        )
                    if outside_contact_region:
                        if self._terrain_route_recovery:
                            self._record_route_blocked_region()
                        self._transition(
                            SearchState.REPLAN,
                            "world route blocked outside contact region",
                        )
                        continue
                if world_route is None:
                    self._route_distances.clear()
                contact_action, contact_engaged = self._contact_step(
                    pov,
                    telemetry,
                    world_route,
                    selected_detection,
                    raycast=raycast,
                    route_stalled=route_stalled,
                )
                if contact_engaged:
                    if (
                        self._contact is not None
                        and self._contact.result == "replan"
                    ):
                        self._note_contact_replan()
                        self._transition(
                            SearchState.REPLAN,
                            "trunk contact attempts exhausted",
                        )
                        continue
                    self._previous_pov = pov.copy()
                    self.last_action_source = "contact"
                    return self._return_action(contact_action)
                if (
                    self._contact_ownership_guard
                    and route_stalled
                    and world_route is not None
                    and world_route[1] <= self._contact_region_limit()
                ):
                    # The coarse route has reached its local handoff envelope,
                    # but neither the raster nor the current frame supplies a
                    # trunk/log owner. Rebuilding from this translated pose is
                    # safer than replaying forward forever against terrain.
                    self._force_handoff_relocalization = True
                    self._transition(
                        SearchState.REPLAN,
                        "world route stalled inside contact region without local evidence",
                    )
                    continue
                if (
                    self._contact_ownership_guard
                    and world_route is not None
                    and world_route[1] > self._contact_region_limit()
                ):
                    # Outside the local contact region, the frozen world
                    # coordinate owns coarse navigation. A distant RGB trunk
                    # component can flicker between neighbouring trees and
                    # otherwise induce an endless left/right alignment loop.
                    if self._terrain_route_recovery:
                        self._verify_terrain_climb_burst(world_route)
                        if (
                            self._terrain_climb_failures
                            >= self._terrain_route_failure_limit
                            and not self._terrain_climb_actions
                        ):
                            # Consecutive climb bursts produced no real route
                            # progress: the terrain between here and this
                            # candidate is genuinely blocked. Remember the
                            # physical region and replan instead of paying
                            # more recoveries against the same obstacle.
                            self._record_route_blocked_region()
                            self._transition(
                                SearchState.REPLAN,
                                "terrain route climb recovery failed without progress",
                            )
                            continue
                    yaw_error = float(world_route[0])
                    if abs(yaw_error) > self.config.align_threshold_degrees:
                        action = (
                            self.config.right_action
                            if yaw_error > 0
                            else self.config.left_action
                        )
                    else:
                        climb_action = self._terrain_route_climb_action(
                            world_route
                        )
                        action = (
                            self.config.forward_action
                            if climb_action is None
                            else climb_action
                        )
                    self._previous_pov = pov.copy()
                    return self._return_action(action)
                if self.progress.is_stalled() and (
                    selected_detection is not None or world_route is None
                ):
                    self.stalled_count += 1
                    if self._candidate_handoff_guard:
                        self._start_local_reacquire(
                            "visual progress window stalled without verified handoff"
                        )
                        continue
                    if (
                        self._contact is not None
                        and self.selected_candidate is not None
                    ):
                        # The world-route walk is stuck; hand over to the
                        # contact controller's local trunk search instead of
                        # replaying the v4 turn-sweep against an obstacle.
                        if not self._contact.active:
                            target_hint = None
                            if self.selected_candidate.has_world_position:
                                target_hint = (
                                    float(self.selected_candidate.estimated_world_x),
                                    float(self.selected_candidate.estimated_world_z),
                                )
                            self._contact.engage(
                                telemetry=telemetry,
                                candidate_id=self.selected_candidate.candidate_id,
                                global_step=self.step,
                                target_hint=target_hint,
                            )
                        action = self._contact.act(
                            pov,
                            telemetry,
                            world_route,
                            global_step=self.step,
                            raycast=raycast,
                        )
                        if self._contact.result == "replan":
                            self._note_contact_replan()
                            self._transition(
                                SearchState.REPLAN,
                                "trunk contact attempts exhausted",
                            )
                            continue
                        self._previous_pov = pov.copy()
                        self.last_action_source = "contact"
                        return self._return_action(action)
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
                        if self._candidate_handoff_guard:
                            uncertainty = min(
                                float(
                                    self._route_target_uncertainty
                                    if self._route_target_uncertainty is not None
                                    else uncertainty
                                ),
                                self.config.handoff_uncertainty_cap,
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
                elif (
                    self._candidate_handoff_guard
                    and world_route is not None
                    and world_route[1]
                    <= max(
                        self.config.world_arrival_radius,
                        min(
                            float(
                                self._route_target_uncertainty
                                if self._route_target_uncertainty is not None
                                else self.selected_candidate.position_uncertainty or 0.0
                            ),
                            self.config.handoff_uncertainty_cap,
                        ),
                    )
                    and self._local_handoff_evidence(
                        selected_detection, raycast
                    )
                    is None
                ):
                    self._start_local_reacquire(
                        "predicted world coordinate reached without local trunk evidence"
                    )
                    continue
                elif self.adapter.ready_to_interact(selected_detection):
                    if self._candidate_handoff_guard:
                        self._start_local_reacquire(
                            "predicted world coordinate reached with interaction "
                            "geometry but without local trunk evidence"
                        )
                        continue
                    action = self.adapter.interaction_action()
                else:
                    action = self.config.forward_action
                self._previous_pov = pov.copy()
                return self._return_action(action)

            if self.state == SearchState.LOCAL_REACQUIRE:
                if self._candidate_handoff_guard:
                    contact_action, contact_engaged = self._contact_step(
                        pov,
                        telemetry,
                        self._world_route(),
                        selected_detection,
                        raycast=raycast,
                        route_stalled=True,
                    )
                    if contact_engaged:
                        if (
                            self._contact is not None
                            and self._contact.result == "replan"
                        ):
                            self._note_contact_replan()
                            self._transition(
                                SearchState.REPLAN,
                                "verified handoff contact attempts exhausted",
                            )
                            continue
                        self._previous_pov = pov.copy()
                        self.last_action_source = "contact"
                        return self._return_action(contact_action)
                # Loss recovery may return as soon as the remembered object is
                # visible. A stall recovery deliberately completes the +/-30
                # sweep so a nearby false commitment cannot trap the policy.
                if (
                    selected_detection is not None
                    and self._local_trigger.startswith("candidate lost")
                ):
                    self.progress.reset()
                    self._transition(SearchState.ALIGN, "candidate locally reacquired")
                    continue
                if self._local_actions:
                    self._previous_pov = pov.copy()
                    return self._return_action(self._local_actions.popleft())
                if self._local_trigger.startswith("predicted world coordinate"):
                    if self._candidate_handoff_guard:
                        self._force_handoff_relocalization = True
                    self._transition(
                        SearchState.REPLAN,
                        "predicted coordinate inspected without resource success",
                    )
                    continue
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
                if self._force_handoff_relocalization:
                    self._force_handoff_relocalization = False
                    latest_step = (
                        self.config.episode_max_steps
                        - self.config.handoff_relocalization_reserve_steps
                    )
                    if self.step <= latest_step:
                        self.candidate_map.candidates = [
                            candidate
                            for candidate in self.candidate_map.candidates
                            if candidate.status
                            in ("cooldown", "failed", "completed")
                        ]
                        for candidate in self.candidate_map.candidates:
                            if candidate.status == "cooldown":
                                # A full scan takes 36 actions. Do not let an
                                # old route hypothesis become selectable in
                                # the middle of rebuilding from a new pose.
                                candidate.failed_until_step = max(
                                    candidate.failed_until_step,
                                    self.config.episode_max_steps
                                    + self.config.cooldown_steps,
                                )
                        self.scan_yaw = 0.0
                        self.handoff_relocalization_scans += 1
                        self._transition(
                            SearchState.SCAN,
                            "bad local handoff; rebuild candidate map from current pose",
                        )
                        continue
                    self.handoff_relocalization_skipped_late += 1
                available_world_candidates = any(
                    candidate.status == "available" and candidate.has_world_position
                    for candidate in self.candidate_map.candidates
                )
                if (
                    sensor_uses_telemetry(self.config.sensor_profile)
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
            if not sensor_uses_telemetry(self.config.sensor_profile):
                self.heading_yaw = wrap_degrees(
                    self.heading_yaw + self.config.camera_delta
                )
            if self._last_action_state == SearchState.SCAN:
                self.scan_yaw += self.config.camera_delta
        elif int(action) == self.config.left_action:
            if not sensor_uses_telemetry(self.config.sensor_profile):
                self.heading_yaw = wrap_degrees(
                    self.heading_yaw - self.config.camera_delta
                )
        elif int(action) == self.config.fine_right_action:
            if not sensor_uses_telemetry(self.config.sensor_profile):
                self.heading_yaw = wrap_degrees(
                    self.heading_yaw + 0.5 * self.config.camera_delta
                )
        elif int(action) == self.config.fine_left_action:
            if not sensor_uses_telemetry(self.config.sensor_profile):
                self.heading_yaw = wrap_degrees(
                    self.heading_yaw - 0.5 * self.config.camera_delta
                )

        if self._contact is not None and self._contact.engaged:
            self._contact.observe(action, reward, done, info)

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
