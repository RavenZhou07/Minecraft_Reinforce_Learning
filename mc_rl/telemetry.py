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
SENSOR_PROFILES = frozenset((SENSOR_PROFILE_POV_ONLY, SENSOR_PROFILE_F3))


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
class VisualRangeEstimate:
    distance: float
    uncertainty: float

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
