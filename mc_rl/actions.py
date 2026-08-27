"""Convert a small integer action set into MineRL action dictionaries.

MineRL 0.4.4 defines ``camera`` as ``[delta_pitch, delta_yaw]`` in
degrees. Minecraft adds positive pitch when looking down and positive yaw
when turning right. Therefore up/left use negative values below.
"""

from numbers import Integral
from typing import Any, Dict

import numpy as np


ACTION_NAMES = (
    "noop",
    "forward",
    "forward_jump",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
    "attack",
    "forward_attack",
    "backward",
    "fine_turn_left",
    "fine_turn_right",
    "fine_look_up",
    "fine_look_down",
)

DEFAULT_CAMERA_DELTA = 10.0


def discrete_action_to_minerl(
    action_id: int, env: Any, camera_delta: float = DEFAULT_CAMERA_DELTA
) -> Dict[str, Any]:
    """Return a valid MineRL action dictionary for one discrete action.

    ``env.action_space.noop()`` is deliberately used as the starting point.
    A MineRL environment can contain task-specific entries (for example
    ``place`` in Navigate), and hand-building a dictionary could omit them.
    """

    if isinstance(action_id, bool) or not isinstance(action_id, Integral):
        raise TypeError("action_id must be an integer")
    action_id = int(action_id)
    if not 0 <= action_id < len(ACTION_NAMES):
        raise ValueError(
            "action_id must be between 0 and {} (received {})".format(
                len(ACTION_NAMES) - 1, action_id
            )
        )
    if camera_delta <= 0:
        raise ValueError("camera_delta must be positive")

    # The converter is normally given the raw MineRL env. Accepting a wrapped
    # env as well makes the public function less surprising to use.
    action_space = env.action_space
    if not hasattr(action_space, "noop") and hasattr(env, "unwrapped"):
        action_space = env.unwrapped.action_space
    if not hasattr(action_space, "noop"):
        raise TypeError("env must expose a MineRL action space with noop()")

    action = action_space.noop()

    if action_id == 1:
        action["forward"] = 1
    elif action_id == 2:
        action["forward"] = 1
        action["jump"] = 1
    elif action_id == 3:
        action["camera"] = np.array([0.0, -camera_delta], dtype=np.float32)
    elif action_id == 4:
        action["camera"] = np.array([0.0, camera_delta], dtype=np.float32)
    elif action_id == 5:
        action["camera"] = np.array([-camera_delta, 0.0], dtype=np.float32)
    elif action_id == 6:
        action["camera"] = np.array([camera_delta, 0.0], dtype=np.float32)
    elif action_id == 7:
        action["attack"] = 1
    elif action_id == 8:
        action["forward"] = 1
        action["attack"] = 1
    elif action_id == 9:
        action["back"] = 1
    elif action_id == 10:
        action["camera"] = np.array(
            [0.0, -0.5 * camera_delta], dtype=np.float32
        )
    elif action_id == 11:
        action["camera"] = np.array(
            [0.0, 0.5 * camera_delta], dtype=np.float32
        )
    elif action_id == 12:
        action["camera"] = np.array(
            [-0.5 * camera_delta, 0.0], dtype=np.float32
        )
    elif action_id == 13:
        action["camera"] = np.array(
            [0.5 * camera_delta, 0.0], dtype=np.float32
        )

    if not action_space.contains(action):
        raise ValueError("converted action is not accepted by MineRL action_space")
    return action
