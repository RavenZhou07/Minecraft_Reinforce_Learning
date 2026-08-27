"""Privileged 3-D log target memory and coordinate aiming geometry.

The memory is used only by the explicit ``f3_raycast`` teacher profile.  It
records log coordinates that were actually under the crosshair during the
360-degree scan.  The aiming geometry itself is reusable with future
POV-derived coordinates: every step recomputes yaw and pitch from the current
self pose instead of treating an old camera command as ground truth.
"""

from collections import deque
from dataclasses import dataclass, field
from math import atan2, degrees, hypot, log1p, sqrt
from typing import Dict, List, Optional, Tuple

from mc_rl.navigation import target_bearing_degrees, wrap_degrees
from mc_rl.telemetry import AgentTelemetry, RaycastHit


TARGET_AVAILABLE = "available"
TARGET_SELECTED = "selected"
TARGET_COOLDOWN = "cooldown"
TARGET_COMPLETED = "completed"


@dataclass
class TrunkBlockTarget:
    target_id: int
    x: float
    y: float
    z: float
    last_seen_step: int
    observation_count: int = 1
    approach_attempts: int = 0
    recovery_attempts: int = 0
    status: str = TARGET_AVAILABLE
    failed_until_step: int = 0
    score: float = float("-inf")
    eligible: bool = True
    ineligibility_reason: str = ""
    score_terms: Dict[str, float] = field(default_factory=dict, repr=False)

    def distance_to(self, x: float, y: float, z: float) -> float:
        return sqrt((self.x - x) ** 2 + (self.y - y) ** 2 + (self.z - z) ** 2)


@dataclass(frozen=True)
class CoordinateAimError:
    target_yaw: float
    target_pitch: float
    yaw_error: float
    pitch_error: float
    horizontal_distance: float
    distance: float


@dataclass(frozen=True)
class TrunkTargetScoreConfig:
    """Explainable costs for choosing a locally reachable log face."""

    hint_distance_weight: float = 2.0
    horizontal_distance_weight: float = 0.12
    vertical_distance_weight: float = 0.9
    reach_excess_weight: float = 0.35
    failure_weight: float = 2.0
    recovery_weight: float = 0.0
    observation_weight: float = 0.05
    attack_reach: float = 4.0
    maximum_horizontal_distance: Optional[float] = None


class CoordinateProgressMonitor:
    """Track target-distance progress over translation commands only."""

    def __init__(self, window_size: int = 12, minimum_progress: float = 0.35):
        if window_size < 2:
            raise ValueError("coordinate progress window must contain at least two samples")
        if minimum_progress <= 0:
            raise ValueError("coordinate minimum progress must be positive")
        self.window_size = int(window_size)
        self.minimum_progress = float(minimum_progress)
        self.distances = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self.distances.clear()

    def add(self, distance: float) -> None:
        self.distances.append(float(distance))

    @property
    def progress(self) -> Optional[float]:
        if len(self.distances) < 2:
            return None
        return float(self.distances[0] - min(self.distances))

    def is_stalled(self) -> bool:
        progress = self.progress
        return bool(
            len(self.distances) == self.window_size
            and progress is not None
            and progress < self.minimum_progress
        )

    def diagnostics(self) -> Dict[str, object]:
        return {
            "window_size": self.window_size,
            "sample_count": len(self.distances),
            "minimum_progress": self.minimum_progress,
            "observed_progress": self.progress,
            "stalled": self.is_stalled(),
        }


def coordinate_aim_error(
    telemetry: AgentTelemetry,
    target: TrunkBlockTarget,
    eye_height: float = 1.62,
) -> CoordinateAimError:
    """Return Minecraft-convention yaw/pitch errors to a 3-D target point."""

    dx = float(target.x) - telemetry.x
    dz = float(target.z) - telemetry.z
    horizontal = hypot(dx, dz)
    eye_y = telemetry.y + float(eye_height)
    dy = float(target.y) - eye_y
    target_yaw = wrap_degrees(target_bearing_degrees(dx, dz))
    # Minecraft pitch increases while looking down.
    target_pitch = -degrees(atan2(dy, max(horizontal, 1e-6)))
    return CoordinateAimError(
        target_yaw=target_yaw,
        target_pitch=target_pitch,
        yaw_error=wrap_degrees(target_yaw - telemetry.yaw),
        pitch_error=target_pitch - telemetry.pitch,
        horizontal_distance=horizontal,
        distance=hypot(horizontal, dy),
    )


class TrunkTargetMemory:
    """Episode-local object memory for raycast-observed log points."""

    def __init__(self, merge_distance: float = 0.7) -> None:
        if merge_distance <= 0:
            raise ValueError("target merge distance must be positive")
        self.merge_distance = float(merge_distance)
        self.targets: List[TrunkBlockTarget] = []
        self._next_id = 1
        self.last_selection_reason = "not_selected"

    def reset(self) -> None:
        self.targets = []
        self._next_id = 1
        self.last_selection_reason = "not_selected"

    def observe(
        self, hit: RaycastHit, step: int
    ) -> Optional[TrunkBlockTarget]:
        if not hit.is_log:
            return None
        matches = [
            target
            for target in self.targets
            if target.status != TARGET_COMPLETED
            and target.distance_to(hit.x, hit.y, hit.z) <= self.merge_distance
        ]
        if matches:
            target = min(
                matches, key=lambda item: item.distance_to(hit.x, hit.y, hit.z)
            )
            count = target.observation_count
            target.x = (count * target.x + hit.x) / (count + 1)
            target.y = (count * target.y + hit.y) / (count + 1)
            target.z = (count * target.z + hit.z) / (count + 1)
            target.observation_count += 1
            target.last_seen_step = int(step)
            return target
        target = TrunkBlockTarget(
            target_id=self._next_id,
            x=float(hit.x),
            y=float(hit.y),
            z=float(hit.z),
            last_seen_step=int(step),
        )
        self._next_id += 1
        self.targets.append(target)
        return target

    def select(
        self,
        step: int,
        telemetry: Optional[AgentTelemetry] = None,
        candidate_hint: Optional[Tuple[float, float]] = None,
        score_config: Optional[TrunkTargetScoreConfig] = None,
    ) -> Optional[TrunkBlockTarget]:
        available = []
        had_available_status = False
        rejected_by_reachability = False
        for target in self.targets:
            if target.status == TARGET_COOLDOWN and step >= target.failed_until_step:
                target.status = TARGET_AVAILABLE
            if target.status not in (TARGET_AVAILABLE, TARGET_SELECTED):
                continue
            had_available_status = True
            hint_distance = (
                0.0
                if candidate_hint is None
                else hypot(target.x - candidate_hint[0], target.z - candidate_hint[1])
            )
            player_distance = (
                0.0
                if telemetry is None
                else hypot(target.x - telemetry.x, target.z - telemetry.z)
            )
            if score_config is None:
                # Frozen v8 ordering: candidate identity first, then player
                # distance and observation support.
                available.append(
                    (
                        hint_distance,
                        player_distance,
                        -target.observation_count,
                        target.target_id,
                        target,
                    )
                )
                continue
            eye_y = target.y if telemetry is None else telemetry.y + 1.62
            vertical_distance = abs(target.y - eye_y)
            spatial_distance = hypot(player_distance, vertical_distance)
            reach_excess = max(0.0, spatial_distance - score_config.attack_reach)
            terms = {
                "candidate_hint_distance_cost": -score_config.hint_distance_weight
                * hint_distance,
                "horizontal_distance_cost": -score_config.horizontal_distance_weight
                * player_distance,
                "vertical_distance_cost": -score_config.vertical_distance_weight
                * vertical_distance,
                "reach_excess_cost": -score_config.reach_excess_weight
                * reach_excess,
                "failure_cost": -score_config.failure_weight
                * target.approach_attempts,
                "recovery_cost": -score_config.recovery_weight
                * target.recovery_attempts,
                "observation_support": score_config.observation_weight
                * log1p(target.observation_count),
            }
            target.score_terms = terms
            target.score = float(sum(terms.values()))
            target.eligible = bool(
                score_config.maximum_horizontal_distance is None
                or player_distance <= score_config.maximum_horizontal_distance
            )
            target.ineligibility_reason = (
                ""
                if target.eligible
                else "horizontal distance exceeds local selection limit"
            )
            if not target.eligible:
                rejected_by_reachability = True
                continue
            # Maximise score, then observation support, then stable id.
            available.append(
                (-target.score, -target.observation_count, target.target_id, target)
            )
        if not available:
            self.last_selection_reason = (
                "no_eligible_targets"
                if had_available_status and rejected_by_reachability
                else "no_available_targets"
            )
            return None
        target = min(available)[-1]
        self.last_selection_reason = "selected"
        for other in self.targets:
            if other is not target and other.status == TARGET_SELECTED:
                other.status = TARGET_AVAILABLE
        target.status = TARGET_SELECTED
        return target

    def mark_failed(
        self, target: Optional[TrunkBlockTarget], step: int, cooldown_steps: int
    ) -> None:
        if target is None:
            return
        target.approach_attempts += 1
        target.failed_until_step = int(step) + int(cooldown_steps)
        target.status = TARGET_COOLDOWN

    @staticmethod
    def mark_completed(target: Optional[TrunkBlockTarget]) -> None:
        if target is not None:
            target.status = TARGET_COMPLETED

    def rows(self) -> List[Dict[str, object]]:
        rows = []
        for target in self.targets:
            row = {
                "target_id": target.target_id,
                "x": target.x,
                "y": target.y,
                "z": target.z,
                "last_seen_step": target.last_seen_step,
                "observation_count": target.observation_count,
                "approach_attempts": target.approach_attempts,
                "recovery_attempts": target.recovery_attempts,
                "status": target.status,
                "failed_until_step": target.failed_until_step,
                "score": target.score,
                "eligible": target.eligible,
                "ineligibility_reason": target.ineligibility_reason,
                "score_terms": dict(target.score_terms),
            }
            # Keep the complete mapping for human inspection and also flatten
            # every term into a stable CSV column for direct analysis.
            row.update(target.score_terms)
            rows.append(row)
        return rows
