"""Bounded world-coordinate search for a recently broken block drop.

Minecraft drops are entities rather than blocks, so the planner visits the
broken block centre first and then a small eight-direction ring around it.
It uses only the player's own F3-like pose and the internally remembered
contact coordinate.  A waypoint that does not get closer after consecutive
forward commands is treated as blocked and skipped.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Optional, Tuple

from mc_rl.navigation import wrap_degrees
from mc_rl.telemetry import AgentTelemetry, bearing_and_distance_to


@dataclass(frozen=True)
class DropRecoveryConfig:
    radius: float = 0.85
    arrival_radius: float = 0.35
    yaw_tolerance_degrees: float = 12.0
    blocked_forward_steps: int = 3
    minimum_progress: float = 0.04
    max_steps: int = 20
    enable_obstacle_recovery: bool = False
    waypoint_max_steps: int = 7
    centre_waypoint_max_steps: int = 7
    jump_budget: int = 1
    offset_budget: int = 1
    offset_distance: float = 0.35
    ordered_ring: bool = False
    # v9.7 starts the ordered ring on the side nearest the player at the
    # moment recovery begins. Earlier profiles retain the fixed north-first
    # traversal exactly.
    orient_ring_to_start: bool = False
    # v9.6 elevated pickup: when the broken block centre is vertically above
    # the player, the item can only enter the pickup box during a jump apex
    # next to the block column. ``None`` keeps the frozen 2-D behaviour of
    # every earlier profile.
    elevated_vertical_gap: Optional[float] = None
    elevated_arrival_radius: Optional[float] = None
    elevated_jump_steps: Optional[int] = None
    # A broken item falls after the supporting log disappears. v9.7 therefore
    # reserves the vertical probe for the remembered block centre and walks
    # the surrounding ring normally instead of repeatedly jumping at every
    # waypoint.
    elevated_jump_centre_only: bool = False
    # v9.9: after the direct horizontal projection, traverse an outer ring,
    # revisit the projection, then sweep the original inner pickup ring.
    ground_sweep: bool = False
    ground_sweep_outer_radius: float = 1.6

    def __post_init__(self) -> None:
        if self.radius <= 0 or self.arrival_radius <= 0:
            raise ValueError("drop recovery radii must be positive")
        if self.blocked_forward_steps <= 0 or self.max_steps <= 0:
            raise ValueError("drop recovery budgets must be positive")
        if self.waypoint_max_steps <= 0 or self.centre_waypoint_max_steps <= 0:
            raise ValueError("drop waypoint step budget must be positive")
        if self.jump_budget < 0 or self.offset_budget < 0:
            raise ValueError("drop obstacle recovery budgets cannot be negative")
        if self.offset_distance <= 0:
            raise ValueError("drop obstacle offset must be positive")
        if self.ground_sweep and self.ground_sweep_outer_radius <= self.radius:
            raise ValueError("ground sweep outer radius must exceed inner radius")
        if (
            self.elevated_vertical_gap is not None
            and self.elevated_vertical_gap <= 0
        ):
            raise ValueError("elevated vertical gap must be positive when set")
        if (
            self.elevated_arrival_radius is not None
            and self.elevated_arrival_radius <= 0
        ):
            raise ValueError("elevated arrival radius must be positive when set")
        if self.elevated_jump_steps is not None and self.elevated_jump_steps <= 0:
            raise ValueError("elevated jump budget must be positive when set")


class DropRecoveryPlanner:
    """Drive toward the contact point and a bounded eight-direction ring."""

    def __init__(
        self,
        centre_x: float,
        centre_z: float,
        config: Optional[DropRecoveryConfig] = None,
        centre_y: Optional[float] = None,
    ) -> None:
        self.config = config or DropRecoveryConfig()
        self.centre_y = centre_y
        radius = self.config.radius
        diagonal = radius / sqrt(2.0)
        self._ring_slices: List[Tuple[int, int]] = []
        if self.config.ordered_ring:
            # Adjacent waypoints differ by 45 degrees. The earlier N/E/S/W
            # ordering spent most of a seven-step waypoint budget turning
            # across the centre instead of translating through the ring.
            ring_offsets = [
                (0.0, radius),
                (diagonal, diagonal),
                (radius, 0.0),
                (diagonal, -diagonal),
                (0.0, -radius),
                (-radius, 0.0),
                (-diagonal, -diagonal),
                (-diagonal, diagonal),
            ]
            if self.config.ground_sweep:
                outer_radius = self.config.ground_sweep_outer_radius
                outer_diagonal = outer_radius / sqrt(2.0)
                outer_ring_offsets = [
                    (0.0, outer_radius),
                    (outer_diagonal, outer_diagonal),
                    (outer_radius, 0.0),
                    (outer_diagonal, -outer_diagonal),
                    (0.0, -outer_radius),
                    (-outer_radius, 0.0),
                    (-outer_diagonal, -outer_diagonal),
                    (-outer_diagonal, outer_diagonal),
                ]
                offsets = (
                    [(0.0, 0.0)]
                    + outer_ring_offsets
                    + [(0.0, 0.0)]
                    + ring_offsets
                )
                self._ring_slices = [(1, 9), (10, 18)]
            else:
                offsets = [(0.0, 0.0)] + ring_offsets
                self._ring_slices = [(1, 9)]
        else:
            offsets = (
                (0.0, 0.0),
                (0.0, radius),
                (radius, 0.0),
                (0.0, -radius),
                (-radius, 0.0),
                (diagonal, diagonal),
                (diagonal, -diagonal),
                (-diagonal, -diagonal),
                (-diagonal, diagonal),
            )
        self.waypoints: List[Tuple[float, float]] = [
            (float(centre_x) + dx, float(centre_z) + dz)
            for dx, dz in offsets
        ]
        self.index = 0
        self.steps = 0
        self.reached_waypoints = 0
        self.blocked_waypoints = 0
        self.elevated_jump_steps = 0
        self.consecutive_blocked_waypoints = 0
        self.max_consecutive_blocked_waypoints = 0
        self._elevated_jumps_this_waypoint = 0
        self._last_action: Optional[int] = None
        self._last_distance: Optional[float] = None
        self._no_progress_steps = 0
        self._waypoint_steps = 0
        self._jump_attempts = 0
        self._offset_attempts = 0
        self._detour: Optional[Tuple[float, float]] = None
        self._current_record: Optional[Dict[str, object]] = None
        self.waypoint_records: List[Dict[str, object]] = []

    def orient_ring_to_start(self, start_x: float, start_z: float) -> None:
        """Rotate each ordered sweep ring to start nearest the player."""

        if not self.config.orient_ring_to_start or len(self.waypoints) <= 2:
            return
        for start, end in self._ring_slices:
            ring = self.waypoints[start:end]
            nearest = min(
                range(len(ring)),
                key=lambda index: (ring[index][0] - float(start_x)) ** 2
                + (ring[index][1] - float(start_z)) ** 2,
            )
            self.waypoints[start:end] = ring[nearest:] + ring[:nearest]

    @property
    def complete(self) -> bool:
        return bool(
            self.steps >= self.config.max_steps
            or self.index >= len(self.waypoints)
        )

    @property
    def current_waypoint(self) -> Optional[Tuple[float, float]]:
        if self.index >= len(self.waypoints):
            return None
        return self.waypoints[self.index]

    def _advance(self, blocked: bool) -> None:
        if blocked:
            self.blocked_waypoints += 1
        else:
            self.reached_waypoints += 1
        self.index += 1
        self._last_action = None
        self._last_distance = None
        self._no_progress_steps = 0

    @staticmethod
    def _distance_to(
        telemetry: AgentTelemetry, waypoint: Tuple[float, float]
    ) -> float:
        return sqrt(
            (float(waypoint[0]) - telemetry.x) ** 2
            + (float(waypoint[1]) - telemetry.z) ** 2
        )

    def _ensure_record(
        self, telemetry: AgentTelemetry, waypoint: Tuple[float, float]
    ) -> None:
        if self._current_record is not None:
            return
        distance = self._distance_to(telemetry, waypoint)
        self._current_record = {
            "waypoint_index": self.index,
            "target_x": waypoint[0],
            "target_z": waypoint[1],
            "start_distance": distance,
            "minimum_distance": distance,
            "end_distance": distance,
            "steps": 0,
            "jump_attempts": 0,
            "offset_attempts": 0,
            "elevated_jumps": 0,
            "vertical_gap": (
                None
                if self.centre_y is None
                else round(float(self.centre_y) - float(telemetry.y), 3)
            ),
            "end_reason": "",
            "reward": 0.0,
        }

    def _update_record(
        self, telemetry: AgentTelemetry, waypoint: Tuple[float, float]
    ) -> None:
        self._ensure_record(telemetry, waypoint)
        distance = self._distance_to(telemetry, waypoint)
        if self._current_record is not None:
            self._current_record["minimum_distance"] = min(
                float(self._current_record["minimum_distance"]), distance
            )
            self._current_record["end_distance"] = distance

    def _finalize_record(self, reason: str, reward: float = 0.0) -> None:
        if self._current_record is None:
            return
        self._current_record["steps"] = self._waypoint_steps
        self._current_record["jump_attempts"] = self._jump_attempts
        self._current_record["offset_attempts"] = self._offset_attempts
        self._current_record["elevated_jumps"] = self._elevated_jumps_this_waypoint
        self._current_record["end_reason"] = reason
        self._current_record["reward"] = float(reward)
        self.waypoint_records.append(dict(self._current_record))
        self._current_record = None

    def _advance_enhanced(self, reason: str, blocked: bool) -> None:
        self._finalize_record(reason)
        if blocked:
            self.blocked_waypoints += 1
            self.consecutive_blocked_waypoints += 1
        else:
            self.reached_waypoints += 1
            self.consecutive_blocked_waypoints = 0
        self.max_consecutive_blocked_waypoints = max(
            self.max_consecutive_blocked_waypoints,
            self.consecutive_blocked_waypoints,
        )
        self.index += 1
        self._last_action = None
        self._last_distance = None
        self._no_progress_steps = 0
        self._waypoint_steps = 0
        self._jump_attempts = 0
        self._offset_attempts = 0
        self._elevated_jumps_this_waypoint = 0
        self._detour = None

    def record_reward(self, reward: float) -> None:
        if float(reward) <= 0:
            return
        self._finalize_record("reward", float(reward))

    def finish(self, reason: str = "search_complete") -> None:
        self._finalize_record(reason)

    def next_action(
        self,
        telemetry: AgentTelemetry,
        forward_action: int,
        left_action: int,
        right_action: int,
        noop_action: int,
        forward_jump_action: Optional[int] = None,
    ) -> Optional[int]:
        """Return one movement action, or ``None`` when the budget ends."""

        if self.config.enable_obstacle_recovery:
            return self._next_enhanced_action(
                telemetry,
                forward_action,
                left_action,
                right_action,
                noop_action,
                forward_jump_action,
            )

        while not self.complete:
            waypoint = self.current_waypoint
            if waypoint is None:
                return None
            bearing, distance = bearing_and_distance_to(
                telemetry, waypoint[0], waypoint[1]
            )
            if self._last_action == forward_action:
                progress = (
                    0.0
                    if self._last_distance is None
                    else self._last_distance - distance
                )
                if progress < self.config.minimum_progress:
                    self._no_progress_steps += 1
                else:
                    self._no_progress_steps = 0
                if self._no_progress_steps >= self.config.blocked_forward_steps:
                    self._advance(blocked=True)
                    continue
            if distance <= self.config.arrival_radius:
                self._advance(blocked=False)
                if self.complete:
                    return None
                continue

            yaw_error = wrap_degrees(bearing - telemetry.yaw)
            if abs(yaw_error) > self.config.yaw_tolerance_degrees:
                action = right_action if yaw_error > 0 else left_action
            else:
                action = forward_action
            self.steps += 1
            self._last_action = action
            self._last_distance = float(distance)
            return action

        return None

    def _elevated_pickup_active(self, telemetry: AgentTelemetry) -> bool:
        """True when the drop sits vertically above the player's reach."""

        if (
            self.centre_y is None
            or self.config.elevated_vertical_gap is None
            or self.config.elevated_arrival_radius is None
            or self.config.elevated_jump_steps is None
        ):
            return False
        if self.config.elevated_jump_centre_only and self.index != 0:
            return False
        return bool(
            float(self.centre_y) - float(telemetry.y)
            > float(self.config.elevated_vertical_gap)
        )

    def _next_enhanced_action(
        self,
        telemetry: AgentTelemetry,
        forward_action: int,
        left_action: int,
        right_action: int,
        noop_action: int,
        forward_jump_action: Optional[int],
    ) -> Optional[int]:
        """Navigate with one bounded jump and one small lateral detour."""

        while not self.complete:
            waypoint = self.current_waypoint
            if waypoint is None:
                return None
            self._update_record(telemetry, waypoint)
            waypoint_step_budget = (
                self.config.centre_waypoint_max_steps
                if self.index == 0
                else self.config.waypoint_max_steps
            )
            if self._waypoint_steps >= waypoint_step_budget:
                self._advance_enhanced("waypoint_step_budget", blocked=True)
                continue

            if self._elevated_pickup_active(telemetry):
                # The item rests on a lower trunk block above the player, so
                # walking the horizontal ring can never enter the pickup box.
                # Within the elevated arrival radius, only repeated jump
                # apexes next to the block column can collect it.
                waypoint_distance = self._distance_to(telemetry, waypoint)
                if waypoint_distance <= float(
                    self.config.elevated_arrival_radius
                ):
                    if (
                        forward_jump_action is not None
                        and self._elevated_jumps_this_waypoint
                        < int(self.config.elevated_jump_steps)
                    ):
                        bearing, _distance = bearing_and_distance_to(
                            telemetry, waypoint[0], waypoint[1]
                        )
                        yaw_error = wrap_degrees(bearing - telemetry.yaw)
                        if (
                            abs(yaw_error)
                            > self.config.yaw_tolerance_degrees
                        ):
                            action = (
                                right_action if yaw_error > 0 else left_action
                            )
                        else:
                            action = forward_jump_action
                            self._elevated_jumps_this_waypoint += 1
                            self.elevated_jump_steps += 1
                        self.steps += 1
                        self._waypoint_steps += 1
                        self._last_action = action
                        self._last_distance = float(waypoint_distance)
                        return action
                    self._advance_enhanced(
                        "elevated_jump_budget", blocked=True
                    )
                    continue
                # Still outside the elevated arrival radius: navigate toward
                # the waypoint with the ordinary bounded obstacle machinery.

            navigation_target = self._detour or waypoint
            bearing, distance = bearing_and_distance_to(
                telemetry, navigation_target[0], navigation_target[1]
            )
            was_translation = self._last_action == forward_action or (
                forward_jump_action is not None
                and self._last_action == forward_jump_action
            )
            if was_translation:
                progress = (
                    0.0
                    if self._last_distance is None
                    else self._last_distance - distance
                )
                if progress < self.config.minimum_progress:
                    self._no_progress_steps += 1
                else:
                    self._no_progress_steps = 0
                if self._no_progress_steps >= self.config.blocked_forward_steps:
                    if (
                        forward_jump_action is not None
                        and self._jump_attempts < self.config.jump_budget
                    ):
                        self._jump_attempts += 1
                        self._no_progress_steps = 0
                        action = forward_jump_action
                        self.steps += 1
                        self._waypoint_steps += 1
                        self._last_action = action
                        self._last_distance = float(distance)
                        return action
                    if self._offset_attempts < self.config.offset_budget:
                        dx = waypoint[0] - telemetry.x
                        dz = waypoint[1] - telemetry.z
                        length = max(sqrt(dx * dx + dz * dz), 1e-6)
                        self._detour = (
                            telemetry.x
                            - dz / length * self.config.offset_distance,
                            telemetry.z
                            + dx / length * self.config.offset_distance,
                        )
                        self._offset_attempts += 1
                        self._last_action = None
                        self._last_distance = None
                        self._no_progress_steps = 0
                        continue
                    self._advance_enhanced(
                        "blocked_after_obstacle_recovery", blocked=True
                    )
                    continue

            if distance <= self.config.arrival_radius:
                if self._detour is not None:
                    self._detour = None
                    self._last_action = None
                    self._last_distance = None
                    self._no_progress_steps = 0
                    continue
                self._advance_enhanced("reached", blocked=False)
                continue

            yaw_error = wrap_degrees(bearing - telemetry.yaw)
            if abs(yaw_error) > self.config.yaw_tolerance_degrees:
                action = right_action if yaw_error > 0 else left_action
            else:
                action = forward_action
            self.steps += 1
            self._waypoint_steps += 1
            self._last_action = action
            self._last_distance = float(distance)
            return action

        self._finalize_record("global_step_budget")
        return None

    def diagnostics(self) -> Dict[str, object]:
        waypoint = self.current_waypoint
        return {
            "steps": self.steps,
            "waypoint_index": self.index,
            "waypoint_count": len(self.waypoints),
            "reached_waypoints": self.reached_waypoints,
            "blocked_waypoints": self.blocked_waypoints,
            "target_x": None if waypoint is None else waypoint[0],
            "target_z": None if waypoint is None else waypoint[1],
            "complete": self.complete,
            "waypoint_steps": self._waypoint_steps,
            "jump_attempts": self._jump_attempts,
            "offset_attempts": self._offset_attempts,
            "detour_x": None if self._detour is None else self._detour[0],
            "detour_z": None if self._detour is None else self._detour[1],
            "centre_y": self.centre_y,
            "elevated_jump_steps": self.elevated_jump_steps,
            "consecutive_blocked_waypoints": (
                self.consecutive_blocked_waypoints
            ),
            "max_consecutive_blocked_waypoints": (
                self.max_consecutive_blocked_waypoints
            ),
            "waypoint_records": list(self.waypoint_records),
        }
