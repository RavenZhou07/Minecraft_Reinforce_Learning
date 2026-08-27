"""Object-centric resource detections and episode-scale candidate memory."""

from dataclasses import asdict, dataclass, field
from math import atan2, cos, hypot, log, radians, sin
from typing import Dict, List, Optional, Sequence, Tuple

from mc_rl.navigation import wrap_degrees
from mc_rl.telemetry import (
    AgentTelemetry,
    VisualRangeEstimate,
    bearing_and_distance_to,
    bearing_ray_intersection,
    detection_world_position,
)


CANDIDATE_STATUSES = frozenset(
    ("available", "selected", "cooldown", "failed", "completed")
)


@dataclass(frozen=True)
class ResourceDetection:
    """One POV-only resource observation in camera-relative coordinates."""

    resource_type: str
    horizontal_yaw: float
    confidence: float
    apparent_size: float
    center_x: float = 0.5
    geometry_size: Optional[float] = None


@dataclass
class ResourceCandidate:
    candidate_id: int
    resource_type: str
    relative_yaw: float
    confidence: float
    apparent_size: float
    last_seen_step: int
    observation_count: int = 1
    approach_attempts: int = 0
    stalled_steps: int = 0
    status: str = "available"
    failed_until_step: int = 0
    best_center_offset: float = 180.0
    estimated_world_x: Optional[float] = None
    estimated_world_y: Optional[float] = None
    estimated_world_z: Optional[float] = None
    position_uncertainty: Optional[float] = None
    last_position_update_step: int = -1
    position_observation_count: int = 0
    position_weight: float = field(default=0.0, repr=False)
    position_conflicts: int = 0
    last_observer_x: Optional[float] = None
    last_observer_z: Optional[float] = None
    last_bearing_world: Optional[float] = None
    trunk_observations: int = 0
    canopy_observations: int = 0
    range_basis: Optional[str] = None
    overlap_count: int = 0
    range_estimate: Optional[float] = None
    range_uncertainty: Optional[float] = None
    score: float = float("-inf")
    score_terms: Dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.status not in CANDIDATE_STATUSES:
            raise ValueError("invalid candidate status: {}".format(self.status))
        self.relative_yaw = wrap_degrees(self.relative_yaw)

    @property
    def has_world_position(self) -> bool:
        return (
            self.estimated_world_x is not None
            and self.estimated_world_z is not None
            and self.position_uncertainty is not None
        )


@dataclass
class CandidateScoreConfig:
    confidence_weight: float = 1.0
    size_weight: float = 2.0
    turn_weight: float = 0.15
    failure_weight: float = 1.0
    age_weight: float = 0.002
    # Trunk sightings locate a tree far better than canopy patches, and
    # surviving near-duplicate hypotheses should not outrank distinct trees.
    trunk_support_weight: float = 0.3
    overlap_weight: float = 0.35
    epsilon: float = 1e-3


class CandidateMap:
    """Merge nearby temporal detections and rank reusable candidates."""

    def __init__(
        self,
        merge_yaw_degrees: float = 45.0,
        merge_log_size_tolerance: float = 1.6,
        world_merge_min_distance: float = 2.5,
        position_uncertainty_floor: float = 1.0,
        cooldown_yaw_only_merge: bool = True,
        score_config: Optional[CandidateScoreConfig] = None,
    ):
        if (
            merge_yaw_degrees <= 0
            or merge_log_size_tolerance < 0
            or world_merge_min_distance <= 0
            or position_uncertainty_floor <= 0
        ):
            raise ValueError("candidate merge tolerances must be non-negative")
        self.merge_yaw_degrees = float(merge_yaw_degrees)
        self.merge_log_size_tolerance = float(merge_log_size_tolerance)
        self.world_merge_min_distance = float(world_merge_min_distance)
        self.position_uncertainty_floor = float(position_uncertainty_floor)
        self.cooldown_yaw_only_merge = bool(cooldown_yaw_only_merge)
        self.score_config = score_config or CandidateScoreConfig()
        self.candidates: List[ResourceCandidate] = []
        self.duplicate_candidate_count = 0
        self.split_candidate_count = 0
        self._next_candidate_id = 1

    @staticmethod
    def angular_distance(first: float, second: float) -> float:
        return abs(wrap_degrees(float(first) - float(second)))

    @staticmethod
    def _circular_average(
        first: float, first_weight: float, second: float, second_weight: float
    ) -> float:
        x = first_weight * cos(radians(first)) + second_weight * cos(radians(second))
        y = first_weight * sin(radians(first)) + second_weight * sin(radians(second))
        return wrap_degrees(atan2(y, x) * 180.0 / 3.141592653589793)

    def _merge_match(
        self,
        detection: ResourceDetection,
        world_yaw: float,
        observed_position: Optional[Tuple[float, float, float]] = None,
        observed_uncertainty: Optional[float] = None,
    ) -> Optional[ResourceCandidate]:
        matches = []
        for candidate in self.candidates:
            if candidate.resource_type != detection.resource_type:
                continue
            yaw_error = self.angular_distance(candidate.relative_yaw, world_yaw)
            size_error = abs(
                log(max(candidate.apparent_size, 1e-3))
                - log(max(detection.apparent_size, 1e-3))
            )
            position_error = None
            position_match = False
            if candidate.has_world_position and observed_position is not None:
                position_error = hypot(
                    float(candidate.estimated_world_x) - observed_position[0],
                    float(candidate.estimated_world_z) - observed_position[2],
                )
                position_gate = max(
                    self.world_merge_min_distance,
                    float(candidate.position_uncertainty)
                    + float(observed_uncertainty or 0.0),
                )
                position_match = position_error <= position_gate
            visual_match = bool(
                yaw_error <= self.merge_yaw_degrees
                and size_error <= self.merge_log_size_tolerance
            )
            # A trunk component and its own canopy can differ in apparent
            # size by far more than the log tolerance, which split one tree
            # into several candidates in v4. Trunk sightings of an already
            # trunk-supported hypothesis, or a trunk landing inside a
            # canopy hypothesis's position gate, reunify the same tree.
            sees_trunk = bool(getattr(detection, "sees_trunk", False))
            trunk_identity_match = bool(
                sees_trunk
                and candidate.trunk_observations > 0
                and yaw_error <= self.merge_yaw_degrees
            )
            if (
                candidate.status == "cooldown"
                and not self.cooldown_yaw_only_merge
                and candidate.has_world_position
                and observed_position is not None
                and not position_match
            ):
                visual_match = False
                trunk_identity_match = False
            trunk_into_canopy_match = bool(sees_trunk and position_match)
            cooldown_identity_match = bool(
                candidate.status == "cooldown"
                and (
                    position_match
                    or (
                        self.cooldown_yaw_only_merge
                        and yaw_error <= max(self.merge_yaw_degrees, 60.0)
                    )
                )
            )
            # During a scan, repeated views from nearly the same origin should
            # retain the proven yaw/scale association rule even if an edge-on
            # visual range estimate is noisy. After translation, agreement in
            # estimated world position supplies the alternative association.
            if (
                cooldown_identity_match
                or visual_match
                or position_match
                or trunk_identity_match
                or trunk_into_canopy_match
            ):
                matches.append(
                    (
                        float("inf") if position_error is None else position_error,
                        yaw_error,
                        size_error,
                        candidate.candidate_id,
                        candidate,
                    )
                )
        return min(matches)[-1] if matches else None

    def _set_initial_position(
        self,
        candidate: ResourceCandidate,
        position: Tuple[float, float, float],
        range_estimate: VisualRangeEstimate,
        step: int,
        confidence: float,
    ) -> None:
        weight = max(float(confidence), 0.1) / (range_estimate.uncertainty ** 2)
        candidate.estimated_world_x = float(position[0])
        candidate.estimated_world_y = float(position[1])
        candidate.estimated_world_z = float(position[2])
        candidate.position_uncertainty = max(
            self.position_uncertainty_floor,
            float(range_estimate.uncertainty),
        )
        candidate.last_position_update_step = int(step)
        candidate.position_observation_count = 1
        candidate.position_weight = weight
        candidate.range_estimate = float(range_estimate.distance)
        candidate.range_uncertainty = float(range_estimate.uncertainty)

    def _fuse_position(
        self,
        candidate: ResourceCandidate,
        position: Tuple[float, float, float],
        range_estimate: VisualRangeEstimate,
        step: int,
        confidence: float,
        independent_measurement: bool = True,
    ) -> None:
        if not candidate.has_world_position or candidate.position_weight <= 0:
            self._set_initial_position(
                candidate, position, range_estimate, step, confidence
            )
            return
        new_weight = max(float(confidence), 0.1) / (range_estimate.uncertainty ** 2)
        if not independent_measurement:
            # Consecutive frames from one block position are correlated. They
            # may smooth the point slightly, but must not masquerade as new
            # triangulation evidence and collapse uncertainty toward zero.
            proposed_fraction = new_weight / (candidate.position_weight + new_weight)
            new_fraction = min(0.2, proposed_fraction)
            old_fraction = 1.0 - new_fraction
            candidate.estimated_world_x = (
                old_fraction * float(candidate.estimated_world_x)
                + new_fraction * position[0]
            )
            candidate.estimated_world_y = (
                old_fraction * float(candidate.estimated_world_y)
                + new_fraction * position[1]
            )
            candidate.estimated_world_z = (
                old_fraction * float(candidate.estimated_world_z)
                + new_fraction * position[2]
            )
            candidate.position_uncertainty = max(
                self.position_uncertainty_floor,
                min(
                    float(candidate.position_uncertainty),
                    float(range_estimate.uncertainty),
                ),
            )
            candidate.last_position_update_step = int(step)
            candidate.position_observation_count += 1
            candidate.range_estimate = float(range_estimate.distance)
            candidate.range_uncertainty = float(range_estimate.uncertainty)
            return
        total_weight = candidate.position_weight + new_weight
        old_fraction = candidate.position_weight / total_weight
        new_fraction = new_weight / total_weight
        candidate.estimated_world_x = (
            old_fraction * float(candidate.estimated_world_x)
            + new_fraction * position[0]
        )
        candidate.estimated_world_y = (
            old_fraction * float(candidate.estimated_world_y)
            + new_fraction * position[1]
        )
        candidate.estimated_world_z = (
            old_fraction * float(candidate.estimated_world_z)
            + new_fraction * position[2]
        )
        candidate.position_weight = total_weight
        candidate.position_uncertainty = max(
            self.position_uncertainty_floor, total_weight ** -0.5
        )
        candidate.last_position_update_step = int(step)
        candidate.position_observation_count += 1
        candidate.range_estimate = float(range_estimate.distance)
        candidate.range_uncertainty = float(range_estimate.uncertainty)

    def add_detection(
        self,
        detection: ResourceDetection,
        observer_yaw: float,
        step: int,
        telemetry: Optional[AgentTelemetry] = None,
        range_estimate: Optional[VisualRangeEstimate] = None,
    ) -> Tuple[ResourceCandidate, bool]:
        """Insert or merge one detection; returns ``(candidate, was_merge)``."""

        if not 0.0 <= detection.confidence <= 1.0:
            raise ValueError("detection confidence must be in [0, 1]")
        if detection.apparent_size <= 0:
            raise ValueError("detection apparent_size must be positive")
        if (telemetry is None) != (range_estimate is None):
            raise ValueError("telemetry and range_estimate must be supplied together")
        world_yaw = wrap_degrees(observer_yaw + detection.horizontal_yaw)
        observed_position = (
            None
            if telemetry is None or range_estimate is None
            else detection_world_position(
                telemetry, detection.horizontal_yaw, range_estimate
            )
        )
        match = self._merge_match(
            detection,
            world_yaw,
            observed_position=observed_position,
            observed_uncertainty=(
                None if range_estimate is None else range_estimate.uncertainty
            ),
        )
        if match is None:
            candidate = ResourceCandidate(
                candidate_id=self._next_candidate_id,
                resource_type=detection.resource_type,
                relative_yaw=world_yaw,
                confidence=float(detection.confidence),
                apparent_size=float(detection.apparent_size),
                last_seen_step=int(step),
                best_center_offset=abs(float(detection.horizontal_yaw)),
            )
            self._next_candidate_id += 1
            self.candidates.append(candidate)
            self._record_view_provenance(
                candidate, detection, telemetry, world_yaw
            )
            if observed_position is not None and range_estimate is not None:
                self._set_initial_position(
                    candidate,
                    observed_position,
                    range_estimate,
                    step,
                    detection.confidence,
                )
                candidate.range_basis = range_estimate.basis
            return candidate, False

        old_weight = float(match.observation_count)
        new_weight = max(float(detection.confidence), 0.1)
        independent_position_view = bool(
            telemetry is not None
            and match.last_observer_x is not None
            and match.last_observer_z is not None
            and hypot(
                telemetry.x - float(match.last_observer_x),
                telemetry.z - float(match.last_observer_z),
            )
            >= 1.5
        )
        match.relative_yaw = self._circular_average(
            match.relative_yaw, old_weight, world_yaw, new_weight
        )
        # Scale is taken from the most centred view. Minecraft's perspective
        # projection stretches objects near the horizontal image edges.
        center_offset = abs(float(detection.horizontal_yaw))
        if center_offset < match.best_center_offset:
            match.apparent_size = float(detection.apparent_size)
            match.best_center_offset = center_offset
        match.confidence = max(match.confidence, float(detection.confidence))
        match.last_seen_step = int(step)
        match.observation_count += 1
        self._record_view_provenance(match, detection, telemetry, world_yaw)
        if observed_position is not None and range_estimate is not None:
            self._fuse_position(
                match,
                observed_position,
                range_estimate,
                step,
                detection.confidence,
                independent_measurement=independent_position_view,
            )
            match.range_basis = range_estimate.basis
        self.duplicate_candidate_count += 1
        return match, True

    @staticmethod
    def _record_view_provenance(
        candidate: ResourceCandidate,
        detection: ResourceDetection,
        telemetry: Optional[AgentTelemetry],
        world_yaw: float,
    ) -> None:
        """Track which visual layer produced a view and where it came from."""

        if bool(getattr(detection, "sees_trunk", False)):
            candidate.trunk_observations += 1
        else:
            candidate.canopy_observations += 1
        if telemetry is not None:
            candidate.last_observer_x = float(telemetry.x)
            candidate.last_observer_z = float(telemetry.z)
            candidate.last_bearing_world = float(world_yaw)

    def refresh_world_bearings(self, telemetry: AgentTelemetry) -> None:
        """Recompute remembered bearings after the player translates."""

        for candidate in self.candidates:
            if not candidate.has_world_position:
                continue
            candidate.relative_yaw, _distance = bearing_and_distance_to(
                telemetry,
                float(candidate.estimated_world_x),
                float(candidate.estimated_world_z),
            )

    def update_candidate_position(
        self,
        candidate: ResourceCandidate,
        detection: ResourceDetection,
        telemetry: AgentTelemetry,
        range_estimate: VisualRangeEstimate,
        step: int,
    ) -> None:
        """Fuse a confirmed candidate view into its remembered world point.

        Gross disagreement between the remembered point and a fresh view is
        treated as a conflict: confidence drops, and after repeated conflicts
        the hypothesis is re-seeded from the newest observation (counted as a
        split) instead of silently averaging two different trees together.
        When the player has translated since the last position fix, the two
        bearing rays are intersected as a range-free consistency check.
        """

        position = detection_world_position(
            telemetry, detection.horizontal_yaw, range_estimate
        )
        observed_world_yaw = wrap_degrees(
            telemetry.yaw + detection.horizontal_yaw
        )
        independent_position_view = False
        if candidate.has_world_position:
            remembered_distance = hypot(
                float(candidate.estimated_world_x) - telemetry.x,
                float(candidate.estimated_world_z) - telemetry.z,
            )
            innovation = hypot(
                float(candidate.estimated_world_x) - position[0],
                float(candidate.estimated_world_z) - position[2],
            )
            innovation_gate = max(
                3.0,
                float(candidate.position_uncertainty)
                + range_estimate.uncertainty
                + 1.0,
            )
            # The range calibration is reliable at curriculum scan ranges.
            # Very close canopies clip out of frame; a small residual patch
            # must not make a stationary tree drift away from the player.
            clipped_close_view = bool(
                remembered_distance <= 4.0
                and range_estimate.distance > remembered_distance + 1.5
            )
            triangulation_conflict = False
            if (
                candidate.last_observer_x is not None
                and candidate.last_observer_z is not None
                and candidate.last_bearing_world is not None
            ):
                baseline = hypot(
                    telemetry.x - float(candidate.last_observer_x),
                    telemetry.z - float(candidate.last_observer_z),
                )
                independent_position_view = baseline >= 1.5
                if baseline >= 2.0:
                    previous = AgentTelemetry(
                        x=float(candidate.last_observer_x),
                        y=telemetry.y,
                        z=float(candidate.last_observer_z),
                        yaw=0.0,
                        pitch=0.0,
                    )
                    crossing = bearing_ray_intersection(
                        previous,
                        float(candidate.last_bearing_world),
                        telemetry,
                        observed_world_yaw,
                    )
                    if crossing is not None:
                        crossing_error = hypot(
                            float(candidate.estimated_world_x) - crossing[0],
                            float(candidate.estimated_world_z) - crossing[1],
                        )
                        if crossing_error > innovation_gate:
                            triangulation_conflict = True
                        else:
                            # The two bearings agree on a point; pull the
                            # estimate toward it with the range uncertainty
                            # as weight so consistent views sharpen the fix.
                            self._fuse_position(
                                candidate,
                                (crossing[0], position[1], crossing[1]),
                                VisualRangeEstimate(
                                    distance=max(
                                        hypot(
                                            crossing[0] - telemetry.x,
                                            crossing[1] - telemetry.z,
                                        ),
                                        0.5,
                                    ),
                                    uncertainty=max(
                                        range_estimate.uncertainty, 1.0
                                    ),
                                    basis="bearing_crossing",
                                ),
                                step,
                                detection.confidence,
                                independent_measurement=True,
                            )
            if (
                innovation > innovation_gate
                or clipped_close_view
                or triangulation_conflict
            ):
                candidate.position_conflicts += 1
                candidate.confidence = max(0.05, candidate.confidence * 0.6)
                if candidate.position_conflicts >= 2:
                    self._set_initial_position(
                        candidate,
                        position,
                        range_estimate,
                        step,
                        detection.confidence,
                    )
                    candidate.position_uncertainty = max(
                        1.5, range_estimate.uncertainty * 1.5
                    )
                    self.split_candidate_count += 1
                return
        self._fuse_position(
            candidate,
            position,
            range_estimate,
            step,
            detection.confidence,
            independent_measurement=independent_position_view,
        )
        candidate.range_basis = range_estimate.basis
        candidate.last_observer_x = float(telemetry.x)
        candidate.last_observer_z = float(telemetry.z)
        candidate.last_bearing_world = float(observed_world_yaw)
        candidate.relative_yaw, _distance = bearing_and_distance_to(
            telemetry,
            float(candidate.estimated_world_x),
            float(candidate.estimated_world_z),
        )

    def refresh_cooldowns(self, step: int) -> None:
        for candidate in self.candidates:
            if candidate.status == "cooldown" and step >= candidate.failed_until_step:
                candidate.status = "available"

    def consolidate(self, yaw_tolerance: float = 55.0) -> int:
        """Merge split scan tracks after a full circle.

        The live association gate stays tighter. This second pass handles the
        same object appearing on both sides of the 0/360 scan boundary or
        after one missed trunk frame.
        """

        merged_count = 0
        changed = True
        while changed:
            changed = False
            for first_index, first in enumerate(self.candidates):
                for second_index in range(first_index + 1, len(self.candidates)):
                    second = self.candidates[second_index]
                    position_error = None
                    position_match = False
                    if first.has_world_position and second.has_world_position:
                        position_error = hypot(
                            float(first.estimated_world_x)
                            - float(second.estimated_world_x),
                            float(first.estimated_world_z)
                            - float(second.estimated_world_z),
                        )
                        position_gate = max(
                            self.world_merge_min_distance,
                            float(first.position_uncertainty)
                            + float(second.position_uncertainty),
                        )
                        position_match = position_error <= position_gate
                    size_error = abs(
                        log(max(first.apparent_size, 1e-3))
                        - log(max(second.apparent_size, 1e-3))
                    )
                    cooldown_identity_match = bool(
                        (first.status == "cooldown" or second.status == "cooldown")
                        and (
                            position_match
                            or (
                                self.cooldown_yaw_only_merge
                                and self.angular_distance(
                                    first.relative_yaw, second.relative_yaw
                                )
                                <= max(yaw_tolerance, 60.0)
                            )
                        )
                    )
                    allowed_yaw = (
                        max(yaw_tolerance, 60.0)
                        if cooldown_identity_match else yaw_tolerance
                    )
                    visual_match = bool(
                        self.angular_distance(
                            first.relative_yaw, second.relative_yaw
                        )
                        <= allowed_yaw
                        and size_error <= self.merge_log_size_tolerance
                    )
                    if (
                        not self.cooldown_yaw_only_merge
                        and (first.status == "cooldown" or second.status == "cooldown")
                        and first.has_world_position
                        and second.has_world_position
                        and not position_match
                    ):
                        visual_match = False
                    if first.resource_type != second.resource_type or not (
                        cooldown_identity_match or position_match or visual_match
                    ):
                        continue
                    first.relative_yaw = self._circular_average(
                        first.relative_yaw,
                        max(first.observation_count, 1),
                        second.relative_yaw,
                        max(second.observation_count, 1),
                    )
                    if second.has_world_position:
                        if not first.has_world_position or first.position_weight <= 0:
                            first.estimated_world_x = second.estimated_world_x
                            first.estimated_world_y = second.estimated_world_y
                            first.estimated_world_z = second.estimated_world_z
                            first.position_uncertainty = second.position_uncertainty
                            first.position_weight = second.position_weight
                            first.position_observation_count = (
                                second.position_observation_count
                            )
                        elif second.position_weight > 0:
                            total_weight = first.position_weight + second.position_weight
                            for axis in ("x", "y", "z"):
                                field_name = "estimated_world_{}".format(axis)
                                setattr(
                                    first,
                                    field_name,
                                    (
                                        first.position_weight
                                        * float(getattr(first, field_name))
                                        + second.position_weight
                                        * float(getattr(second, field_name))
                                    )
                                    / total_weight,
                                )
                            first.position_weight = total_weight
                            first.position_uncertainty = max(
                                self.position_uncertainty_floor,
                                min(
                                    float(first.position_uncertainty),
                                    float(second.position_uncertainty),
                                ),
                            )
                            first.position_observation_count += (
                                second.position_observation_count
                            )
                        first.last_position_update_step = max(
                            first.last_position_update_step,
                            second.last_position_update_step,
                        )
                    if second.best_center_offset < first.best_center_offset:
                        first.apparent_size = second.apparent_size
                        first.best_center_offset = second.best_center_offset
                    first.confidence = max(first.confidence, second.confidence)
                    first.last_seen_step = max(
                        first.last_seen_step, second.last_seen_step
                    )
                    first.observation_count += second.observation_count
                    first.trunk_observations += second.trunk_observations
                    first.canopy_observations += second.canopy_observations
                    first.approach_attempts += second.approach_attempts
                    first.stalled_steps += second.stalled_steps
                    first.failed_until_step = max(
                        first.failed_until_step, second.failed_until_step
                    )
                    if second.status == "cooldown":
                        first.status = "cooldown"
                    self.candidates.pop(second_index)
                    self.duplicate_candidate_count += second.observation_count
                    merged_count += 1
                    changed = True
                    break
                if changed:
                    break
        return merged_count

    def score_candidate(
        self,
        candidate: ResourceCandidate,
        current_yaw: float,
        step: int,
        overlap_count: int = 0,
    ) -> Tuple[float, Dict[str, float]]:
        config = self.score_config
        age = max(0, int(step) - candidate.last_seen_step)
        turn_fraction = self.angular_distance(candidate.relative_yaw, current_yaw) / 180.0
        terms = {
            "confidence": config.confidence_weight * candidate.confidence,
            "log_size": config.size_weight
            * log(candidate.apparent_size + config.epsilon),
            "turn": -config.turn_weight * turn_fraction,
            "failures": -config.failure_weight * candidate.approach_attempts,
            "age": -config.age_weight * age,
            # Trunk-supported hypotheses localize an attackable trunk;
            # canopy-only support keeps a weaker, range-uncertain guess.
            "trunk_support": config.trunk_support_weight
            * min(candidate.trunk_observations, 3),
            # Surviving near-duplicates of one tree must not crowd out
            # distinct alternatives in the ranking.
            "overlap": -config.overlap_weight * overlap_count,
        }
        return float(sum(terms.values())), terms

    def _overlap_counts(self) -> Dict[int, int]:
        """Count highly overlapping world-position hypotheses per candidate."""

        counts: Dict[int, int] = {
            candidate.candidate_id: 0 for candidate in self.candidates
        }
        for index, first in enumerate(self.candidates):
            for second in self.candidates[index + 1:]:
                overlapping = False
                if first.has_world_position and second.has_world_position:
                    separation = hypot(
                        float(first.estimated_world_x)
                        - float(second.estimated_world_x),
                        float(first.estimated_world_z)
                        - float(second.estimated_world_z),
                    )
                    gate = 0.6 * max(
                        self.world_merge_min_distance,
                        float(first.position_uncertainty)
                        + float(second.position_uncertainty),
                    )
                    overlapping = separation <= gate
                else:
                    size_error = abs(
                        log(max(first.apparent_size, 1e-3))
                        - log(max(second.apparent_size, 1e-3))
                    )
                    overlapping = bool(
                        self.angular_distance(
                            first.relative_yaw, second.relative_yaw
                        )
                        <= 12.0
                        and size_error
                        <= 0.5 * self.merge_log_size_tolerance
                    )
                if overlapping:
                    counts[first.candidate_id] += 1
                    counts[second.candidate_id] += 1
        return counts

    def ranked(self, current_yaw: float, step: int) -> List[ResourceCandidate]:
        self.refresh_cooldowns(step)
        overlap_counts = self._overlap_counts()
        ranked = []
        for candidate in self.candidates:
            if candidate.status not in ("available", "selected"):
                continue
            candidate.overlap_count = overlap_counts.get(
                candidate.candidate_id, 0
            )
            candidate.score, candidate.score_terms = self.score_candidate(
                candidate, current_yaw, step, candidate.overlap_count
            )
            ranked.append(candidate)
        return sorted(ranked, key=lambda item: (-item.score, item.candidate_id))

    def select(
        self, current_yaw: float, step: int, rank: int = 0
    ) -> Optional[ResourceCandidate]:
        ranked = [item for item in self.ranked(current_yaw, step) if item.status == "available"]
        if not ranked:
            return None
        selected = ranked[min(max(int(rank), 0), len(ranked) - 1)]
        selected.status = "selected"
        selected.approach_attempts += 1
        return selected

    def mark_cooldown(
        self, candidate: ResourceCandidate, step: int, cooldown_steps: int
    ) -> None:
        if cooldown_steps <= 0:
            raise ValueError("cooldown_steps must be positive")
        candidate.status = "cooldown"
        candidate.failed_until_step = int(step) + int(cooldown_steps)

    def rows(self, current_yaw: float, step: int) -> List[Dict[str, object]]:
        rows = []
        for candidate in self.candidates:
            candidate.score, candidate.score_terms = self.score_candidate(
                candidate, current_yaw, step
            )
            row = asdict(candidate)
            row.pop("score_terms", None)
            row.update(
                {"score_{}".format(key): value for key, value in candidate.score_terms.items()}
            )
            rows.append(row)
        return rows
