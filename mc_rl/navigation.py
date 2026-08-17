"""Pure navigation math shared by the FindTree environment and tests."""

from dataclasses import dataclass
from math import atan2, cos, degrees, floor, hypot, radians, sin
from typing import Optional, Sequence, Tuple

import numpy as np


LOG_BLOCKS = frozenset(("log", "log2"))


def wrap_degrees(angle: float) -> float:
    """Wrap an angle to Minecraft's conventional ``[-180, 180)`` range."""

    return (float(angle) + 180.0) % 360.0 - 180.0


def target_bearing_degrees(dx: float, dz: float) -> float:
    """Return Minecraft yaw for a world-space target displacement.

    Minecraft yaw 0 points toward +z, +90 toward -x, and -90 toward +x.
    """

    return degrees(atan2(-float(dx), float(dz)))


def relative_target_state(
    pose: Sequence[float], target_xyz: Optional[Sequence[float]]
) -> np.ndarray:
    """Build ``[available, distance, sin(yaw_error), cos(yaw_error), dy]``."""

    if target_xyz is None:
        return np.zeros(5, dtype=np.float32)
    x, y, z, yaw, _pitch = (float(value) for value in pose)
    tx, ty, tz = (float(value) for value in target_xyz)
    dx, dy, dz = tx - x, ty - y, tz - z
    distance = hypot(dx, dz)
    relative_yaw = wrap_degrees(target_bearing_degrees(dx, dz) - yaw)
    angle = radians(relative_yaw)
    return np.array(
        [1.0, distance, sin(angle), cos(angle), dy], dtype=np.float32
    )


def relative_yaw_degrees(oracle: Sequence[float]) -> float:
    """Recover yaw error from a relative target state vector."""

    if float(oracle[0]) <= 0:
        return 0.0
    return degrees(atan2(float(oracle[2]), float(oracle[3])))


def nearest_log_from_grid(
    log_grid: np.ndarray,
    pose: Sequence[float],
    x_min: int,
    y_min: int,
    z_min: int,
) -> Optional[Tuple[float, float, float]]:
    """Return the nearest log block centre in world coordinates.

    Malmo serializes a grid with x varying fastest, then z, then y. The
    handler reshapes that flat list to ``(y, z, x)``, which this function
    receives. Player-relative grid offsets are based on the floored player
    block position.
    """

    grid = np.asarray(log_grid)
    candidates = np.argwhere(grid > 0)
    if candidates.size == 0:
        return None

    px, py, pz = (floor(float(pose[index])) for index in range(3))
    best = None
    best_key = None
    for y_index, z_index, x_index in candidates:
        dx = int(x_min + x_index)
        dy = int(y_min + y_index)
        dz = int(z_min + z_index)
        # Prefer the closest trunk column, then the lowest block in it.
        key = (dx * dx + dz * dz, abs(dy), dy)
        if best_key is None or key < best_key:
            best_key = key
            best = (px + dx + 0.5, py + dy + 0.5, pz + dz + 0.5)
    return best


def progress_reward(
    previous_distance: Optional[float],
    current_distance: Optional[float],
    progress_scale: float = 1.0,
    step_cost: float = 0.01,
) -> float:
    """Dense reward for reducing horizontal distance to the locked target."""

    reward = -float(step_cost)
    if previous_distance is not None and current_distance is not None:
        reward += float(progress_scale) * (
            float(previous_distance) - float(current_distance)
        )
    return reward


@dataclass
class OracleNavigator:
    """Small rule controller used to validate the privileged task signal."""

    turn_threshold_degrees: float = 8.0
    stuck_tolerance: float = 0.03
    stuck_steps_before_jump: int = 8
    search_clockwise_outside_fov: bool = False
    visual_half_fov_degrees: float = 40.0

    def reset(self) -> None:
        self._previous_distance = None
        self._stuck_steps = 0

    def act(self, oracle: Sequence[float]) -> int:
        if float(oracle[0]) <= 0:
            return 4  # Search clockwise if no target was detected.

        yaw_error = relative_yaw_degrees(oracle)
        distance = float(oracle[1])
        if (
            self.search_clockwise_outside_fov
            and abs(yaw_error) > self.visual_half_fov_degrees
        ):
            # When the target cannot be seen, shortest-path left/right is not
            # inferable from pixels. A fixed scan direction gives the student
            # a realizable teacher action until the tree enters the camera.
            action = 4
        elif yaw_error > self.turn_threshold_degrees:
            action = 4  # turn_right
        elif yaw_error < -self.turn_threshold_degrees:
            action = 3  # turn_left
        else:
            if (
                self._previous_distance is not None
                and self._previous_distance - distance < self.stuck_tolerance
            ):
                self._stuck_steps += 1
            else:
                self._stuck_steps = 0
            action = 2 if self._stuck_steps >= self.stuck_steps_before_jump else 1

        self._previous_distance = distance
        return action


def target_position_from_yaw(distance: float, yaw_degrees: float) -> Tuple[int, int]:
    """Convert a Minecraft yaw/distance pair to integer x/z offsets."""

    angle = radians(float(yaw_degrees))
    dx = int(round(-sin(angle) * float(distance)))
    dz = int(round(cos(angle) * float(distance)))
    return dx, dz
