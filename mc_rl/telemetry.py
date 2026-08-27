"""F3-like self telemetry and visual-to-world coordinate geometry.

Only the agent's own pose and biome are represented here. Resource positions
remain estimates derived from POV detections; no block grid or oracle target
position is accepted by this module.
"""

from dataclasses import dataclass
from math import cos, hypot, radians, sin
from typing import Any, Dict, Optional, Tuple

from mc_rl.navigation import target_bearing_degrees, wrap_degrees


SENSOR_PROFILE_POV_ONLY = "pov_only"
SENSOR_PROFILE_F3 = "f3_telemetry"
SENSOR_PROFILE_RAYCAST = "f3_raycast"
SENSOR_PROFILES = frozenset(
    (SENSOR_PROFILE_POV_ONLY, SENSOR_PROFILE_F3, SENSOR_PROFILE_RAYCAST)
)


def sensor_uses_telemetry(profile: str) -> bool:
    return profile in (SENSOR_PROFILE_F3, SENSOR_PROFILE_RAYCAST)


@dataclass(frozen=True)
class AgentTelemetry:
    """Player-visible self state analogous to Minecraft's F3 overlay."""

    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    biome_id: Optional[int] = None
    biome_temperature: Optional[float] = None
    biome_rainfall: Optional[float] = None

    @classmethod
    def from_observation(cls, value: Dict[str, Any]) -> "AgentTelemetry":
        return cls(
            x=float(value["x"]),
            y=float(value["y"]),
            z=float(value["z"]),
            yaw=wrap_degrees(float(value["yaw"])),
            pitch=float(value["pitch"]),
            biome_id=(
                None if "biome_id" not in value else int(value["biome_id"])
            ),
            biome_temperature=(
                None
                if "biome_temperature" not in value
                else float(value["biome_temperature"])
            ),
            biome_rainfall=(
                None
                if "biome_rainfall" not in value
                else float(value["biome_rainfall"])
            ),
        )


@dataclass(frozen=True)
class RaycastHit:
    """Privileged diagnostic description of the block under the crosshair."""

    has_block: bool
    is_log: bool
    is_leaves: bool
    in_range: bool
    distance: float
    x: float
    y: float
    z: float

    @classmethod
    def from_observation(cls, value: Dict[str, Any]) -> "RaycastHit":
        return cls(
            has_block=bool(float(value.get("has_block", 0.0))),
            is_log=bool(float(value.get("is_log", 0.0))),
            is_leaves=bool(float(value.get("is_leaves", 0.0))),
            in_range=bool(float(value.get("in_range", 0.0))),
            distance=float(value.get("distance", 50.0)),
            x=float(value.get("x", 0.0)),
            y=float(value.get("y", 0.0)),
            z=float(value.get("z", 0.0)),
        )


@dataclass(frozen=True)
class VisualRangeEstimate:
    distance: float
    uncertainty: float
    basis: str = "calibration"

    def __post_init__(self) -> None:
        if self.distance <= 0 or self.uncertainty <= 0:
            raise ValueError("visual range and uncertainty must be positive")


def detection_world_position(
    telemetry: AgentTelemetry,
    horizontal_yaw: float,
    range_estimate: VisualRangeEstimate,
) -> Tuple[float, float, float]:
    """Project a camera-relative detection onto the horizontal world plane."""

    bearing = wrap_degrees(telemetry.yaw + float(horizontal_yaw))
    angle = radians(bearing)
    return (
        telemetry.x - range_estimate.distance * sin(angle),
        telemetry.y,
        telemetry.z + range_estimate.distance * cos(angle),
    )


def bearing_and_distance_to(
    telemetry: AgentTelemetry, world_x: float, world_z: float
) -> Tuple[float, float]:
    dx = float(world_x) - telemetry.x
    dz = float(world_z) - telemetry.z
    return wrap_degrees(target_bearing_degrees(dx, dz)), hypot(dx, dz)


def bearing_ray_intersection(
    first: AgentTelemetry,
    first_world_yaw: float,
    second: AgentTelemetry,
    second_world_yaw: float,
) -> Optional[Tuple[float, float]]:
    """Intersect two world-frame bearing rays observed from two positions.

    Returns ``None`` when the rays are (nearly) parallel or intersect behind
    either observer, in which case the pair carries no triangulation signal.
    """

    def direction(world_yaw: float) -> Tuple[float, float]:
        angle = radians(float(world_yaw))
        return (-sin(angle), cos(angle))

    dx = second.x - first.x
    dz = second.z - first.z
    d1x, d1z = direction(first_world_yaw)
    d2x, d2z = direction(second_world_yaw)
    denominator = d1x * (-d2z) - d1z * (-d2x)
    if abs(denominator) < 1e-6:
        return None
    t1 = (dx * (-d2z) - dz * (-d2x)) / denominator
    t2 = (dx * (-d1z) - dz * (-d1x)) / denominator
    if t1 <= 0.0 or t2 <= 0.0:
        return None
    return (first.x + t1 * d1x, first.z + t1 * d1z)
