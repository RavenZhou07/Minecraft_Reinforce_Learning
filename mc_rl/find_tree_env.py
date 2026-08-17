"""Isolated privileged FindTree curriculum built on MineRL 0.4.4/Malmo.

The actor-facing observation is still the 64x64 POV. A compact ``oracle``
vector is exposed for diagnostics, teacher supervision, or a privileged
critic. The environment never modifies the installed MineRL package.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import gym
import numpy as np
import psutil

from mc_rl.actions import ACTION_NAMES, discrete_action_to_minerl
from mc_rl.envs import configure_minerl_runtime
from mc_rl.navigation import (
    nearest_log_from_grid,
    progress_reward,
    relative_target_state,
    target_position_from_yaw,
    wrap_degrees,
)
from mc_rl.telemetry import SENSOR_PROFILE_F3, SENSOR_PROFILES


ENV_ID = "MineRLFindTree-v0"
NAVIGATION_ACTION_NAMES = ACTION_NAMES[:7]
CANDIDATE_NAVIGATION_GLOBAL_ACTIONS = (0, 1, 2, 3, 4, 5, 6, 9)
CANDIDATE_NAVIGATION_ACTION_NAMES = NAVIGATION_ACTION_NAMES + ("backward",)
GRID_NAME = "find_tree_log_grid"
GRID_BOUNDS = (-16, 16, -1, 4, -16, 16)  # x_min/max, y_min/max, z_min/max
POSE_KEYS = ("xpos", "ypos", "zpos", "yaw", "pitch")
BIOME_KEYS = ("biome_id", "biome_temperature", "biome_rainfall")


def _minerl_types():
    """Import old MineRL classes only when this experimental task is used."""

    from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
    from minerl.herobraine.env_specs.treechop_specs import Treechop
    from minerl.herobraine.hero import handlers, spaces
    from minerl.herobraine.hero.handlers.translation import (
        KeymapTranslationHandler,
        TranslationHandler,
        TranslationHandlerGroup,
    )

    return (
        SimpleEmbodimentEnvSpec,
        Treechop,
        handlers,
        spaces,
        KeymapTranslationHandler,
        TranslationHandler,
        TranslationHandlerGroup,
    )


def build_find_tree_spec_class():
    """Build the EnvSpec class after MineRL registration/import is configured."""

    (
        SimpleEmbodimentEnvSpec,
        Treechop,
        handlers,
        spaces,
        KeymapTranslationHandler,
        TranslationHandler,
        TranslationHandlerGroup,
    ) = _minerl_types()

    class LogGridObservation(TranslationHandler):
        def __init__(self):
            x_min, x_max, y_min, y_max, z_min, z_max = GRID_BOUNDS
            self.shape = (
                y_max - y_min + 1,
                z_max - z_min + 1,
                x_max - x_min + 1,
            )
            super().__init__(
                spaces.Box(low=0, high=1, shape=self.shape, dtype=np.uint8)
            )

        def to_string(self) -> str:
            return "log_grid"

        def xml_template(self) -> str:
            x_min, x_max, y_min, y_max, z_min, z_max = GRID_BOUNDS
            return (
                '<ObservationFromGrid><Grid name="{}">'
                '<min x="{}" y="{}" z="{}"/>'
                '<max x="{}" y="{}" z="{}"/>'
                "</Grid></ObservationFromGrid>"
            ).format(GRID_NAME, x_min, y_min, z_min, x_max, y_max, z_max)

        def from_hero(self, hero_dict: Dict[str, Any]) -> np.ndarray:
            cells = hero_dict.get(GRID_NAME)
            if cells is None:
                raise KeyError("Malmo observation is missing {}".format(GRID_NAME))
            if len(cells) != int(np.prod(self.shape)):
                raise ValueError(
                    "{} has {} cells; expected {}".format(
                        GRID_NAME, len(cells), int(np.prod(self.shape))
                    )
                )
            mask = np.fromiter(
                (1 if str(cell) in ("log", "log2") else 0 for cell in cells),
                dtype=np.uint8,
                count=len(cells),
            )
            # Malmo loops y, then z, then x (x is the fastest axis).
            return mask.reshape(self.shape)

        def from_universal(self, universal_dict: Dict[str, Any]) -> np.ndarray:
            return np.asarray(universal_dict["log_grid"], dtype=np.uint8)

        def to_hero(self, _value):
            raise NotImplementedError("observations are not sent to Malmo")

    class _PoseScalar(KeymapTranslationHandler):
        def __init__(self, key: str, low: float, high: float):
            super().__init__(
                hero_keys=[key],
                univ_keys=["privileged_pose", key],
                to_string=key,
                space=spaces.Box(
                    low=low, high=high, shape=(), dtype=np.float32
                ),
            )

    class PrivilegedPoseObservation(TranslationHandlerGroup):
        def __init__(self):
            super().__init__(
                handlers=[
                    _PoseScalar("xpos", -640000.0, 640000.0),
                    _PoseScalar("ypos", -640000.0, 640000.0),
                    _PoseScalar("zpos", -640000.0, 640000.0),
                    _PoseScalar("yaw", -180.0, 180.0),
                    _PoseScalar("pitch", -180.0, 180.0),
                    _PoseScalar("biome_id", -1.0, 255.0),
                    _PoseScalar("biome_temperature", -10.0, 10.0),
                    _PoseScalar("biome_rainfall", 0.0, 10.0),
                ]
            )

        def to_string(self) -> str:
            return "privileged_pose"

        def xml_template(self) -> str:
            return "<ObservationFromFullStats/>"

    class FindTreeEnvSpec(Treechop):
        def __init__(
            self,
            max_episode_steps: int = 2000,
            yaw_noise_degrees: float = 30.0,
            target_distance_min: int = 5,
            target_distance_max: int = 8,
            distractor_tree_count: int = 0,
        ):
            self.find_tree_max_steps = int(max_episode_steps)
            if not 0 <= yaw_noise_degrees <= 180:
                raise ValueError("yaw_noise_degrees must be between 0 and 180")
            self.yaw_noise_degrees = float(yaw_noise_degrees)
            if not 2 <= target_distance_min <= target_distance_max <= 10:
                raise ValueError(
                    "target distance range must satisfy 2 <= min <= max <= 10"
                )
            self.target_distance_min = int(target_distance_min)
            self.target_distance_max = int(target_distance_max)
            if not 0 <= distractor_tree_count <= 4:
                raise ValueError("distractor_tree_count must be between 0 and 4")
            self.distractor_tree_count = int(distractor_tree_count)
            self.episode_seed = 0
            self.target_block = (0, 4, 6)
            self.tree_blocks = [self.target_block]
            self.agent_yaw = 0.0
            SimpleEmbodimentEnvSpec.__init__(
                self,
                name=ENV_ID,
                max_episode_steps=self.find_tree_max_steps,
                reward_threshold=None,
            )

        def set_episode_seed(self, seed: int) -> None:
            self.episode_seed = int(seed)

        def reset(self):
            rng = np.random.RandomState(self.episode_seed)
            target_yaw = float(rng.uniform(-180.0, 180.0))
            distance = int(
                rng.randint(self.target_distance_min, self.target_distance_max + 1)
            )
            target_x, target_z = target_position_from_yaw(distance, target_yaw)
            self.target_block = (target_x, 4, target_z)
            self.tree_blocks = [self.target_block]
            # Distractors are identical trees placed at least two nominal
            # blocks farther away. The nearest-tree target rule is therefore
            # well-defined from pixels through relative apparent scale, while
            # still requiring the student to compare multiple candidates.
            for index in range(self.distractor_tree_count):
                base_angle = target_yaw + (index + 1) * (
                    360.0 / (self.distractor_tree_count + 1)
                )
                distractor_yaw = wrap_degrees(base_angle + rng.uniform(-20.0, 20.0))
                minimum_distance = min(14, distance + 2)
                distractor_distance = int(rng.randint(minimum_distance, 15))
                dx, dz = target_position_from_yaw(
                    distractor_distance, distractor_yaw
                )
                self.tree_blocks.append((dx, 4, dz))
            # The first curriculum keeps the target inside or near the initial
            # camera frustum. Later stages can widen this noise range.
            self.agent_yaw = wrap_degrees(
                target_yaw
                + rng.uniform(-self.yaw_noise_degrees, self.yaw_noise_degrees)
            )
            super().reset()

        def create_observables(self) -> List[Any]:
            return super().create_observables() + [
                PrivilegedPoseObservation(),
                LogGridObservation(),
            ]

        def create_rewardables(self) -> List[Any]:
            # Dense navigation reward is calculated in the project wrapper so
            # its formula remains readable and unit-testable in Python.
            return []

        def create_agent_start(self) -> List[Any]:
            return [
                handlers.AgentStartPlacement(
                    x=0.5, y=4.0, z=0.5, yaw=self.agent_yaw, pitch=0.0
                )
            ]

        def create_agent_handlers(self) -> List[Any]:
            return []

        def create_server_world_generators(self) -> List[Any]:
            return [
                handlers.FlatWorldGenerator(
                    force_reset=True, generatorString="1;7,2x3,2;1"
                )
            ]

        def create_server_decorators(self) -> List[Any]:
            drawing_parts = []
            for x, y, z in self.tree_blocks:
                drawing_parts.append(
                    (
                        '<DrawCuboid x1="{x}" y1="{y}" z1="{z}" '
                        'x2="{x}" y2="{top}" z2="{z}" type="log"/>'
                        '<DrawSphere x="{x}" y="{leaf_y}" z="{z}" '
                        'radius="2" type="leaves"/>'
                    ).format(x=x, y=y, z=z, top=y + 2, leaf_y=y + 4)
                )
            drawing = "".join(drawing_parts)
            return [handlers.DrawingDecorator(drawing)]

        def create_server_quit_producers(self) -> List[Any]:
            from minerl.herobraine.hero.mc import MS_PER_STEP

            return [
                handlers.ServerQuitFromTimeUp(
                    self.find_tree_max_steps * MS_PER_STEP
                ),
                handlers.ServerQuitWhenAnyAgentFinishes(),
            ]

        def determine_success_from_rewards(self, _rewards: list) -> bool:
            return False

        def is_from_folder(self, _folder: str) -> bool:
            return False

        def get_docstring(self):
            return "Find a single log target using POV; oracle state is training-only."

    return FindTreeEnvSpec


class NavigationDiscreteActionWrapper(gym.ActionWrapper):
    """Expose only the seven non-combat actions for navigation."""

    def __init__(self, env: gym.Env, camera_delta: float = 10.0):
        super().__init__(env)
        self.camera_delta = float(camera_delta)
        self.action_space = gym.spaces.Discrete(len(NAVIGATION_ACTION_NAMES))

    def action(self, action: int) -> Dict[str, Any]:
        return discrete_action_to_minerl(action, self.env, self.camera_delta)


class CandidateNavigationActionWrapper(gym.ActionWrapper):
    """Add backward recovery without changing the legacy seven action IDs."""

    def __init__(self, env: gym.Env, camera_delta: float = 10.0):
        super().__init__(env)
        self.camera_delta = float(camera_delta)
        self.action_space = gym.spaces.Discrete(
            len(CANDIDATE_NAVIGATION_GLOBAL_ACTIONS)
        )

    def action(self, action: int) -> Dict[str, Any]:
        action = int(action)
        if not 0 <= action < len(CANDIDATE_NAVIGATION_GLOBAL_ACTIONS):
            raise ValueError("invalid candidate-navigation action: {}".format(action))
        global_action = CANDIDATE_NAVIGATION_GLOBAL_ACTIONS[action]
        return discrete_action_to_minerl(global_action, self.env, self.camera_delta)


class FindTreeTaskWrapper(gym.Wrapper):
    """Lock one observed log target and compute dense navigation progress."""

    def __init__(
        self,
        env: gym.Env,
        max_episode_steps: int = 250,
        success_distance: float = 1.8,
        success_bonus: float = 5.0,
        progress_scale: float = 1.0,
        step_cost: float = 0.01,
        sensor_profile: str = "pov_only",
    ):
        super().__init__(env)
        if max_episode_steps <= 0 or success_distance <= 0:
            raise ValueError("max_episode_steps and success_distance must be positive")
        self.max_episode_steps = int(max_episode_steps)
        self.success_distance = float(success_distance)
        self.success_bonus = float(success_bonus)
        self.progress_scale = float(progress_scale)
        self.step_cost = float(step_cost)
        if sensor_profile not in SENSOR_PROFILES:
            raise ValueError("unknown sensor profile: {}".format(sensor_profile))
        self.sensor_profile = sensor_profile
        observation_spaces = {
            "pov": env.observation_space.spaces["pov"],
            "oracle": gym.spaces.Box(
                low=np.array([0.0, 0.0, -1.0, -1.0, -10.0]),
                high=np.array([1.0, 50.0, 1.0, 1.0, 10.0]),
                dtype=np.float32,
            ),
        }
        if self.sensor_profile == SENSOR_PROFILE_F3:
            observation_spaces["telemetry"] = gym.spaces.Dict(
                {
                    "x": gym.spaces.Box(-640000.0, 640000.0, shape=(), dtype=np.float32),
                    "y": gym.spaces.Box(-640000.0, 640000.0, shape=(), dtype=np.float32),
                    "z": gym.spaces.Box(-640000.0, 640000.0, shape=(), dtype=np.float32),
                    "yaw": gym.spaces.Box(-180.0, 180.0, shape=(), dtype=np.float32),
                    "pitch": gym.spaces.Box(-180.0, 180.0, shape=(), dtype=np.float32),
                    "biome_id": gym.spaces.Box(-1, 255, shape=(), dtype=np.int32),
                    "biome_temperature": gym.spaces.Box(-10.0, 10.0, shape=(), dtype=np.float32),
                    "biome_rainfall": gym.spaces.Box(0.0, 10.0, shape=(), dtype=np.float32),
                }
            )
        self.observation_space = gym.spaces.Dict(observation_spaces)
        self.elapsed_steps = 0
        self.target_xyz = None
        self.previous_distance = None
        self.grid_target_valid = False

    @staticmethod
    def _pose(raw_observation: Dict[str, Any]) -> np.ndarray:
        pose = raw_observation["privileged_pose"]
        return np.array([pose[key] for key in POSE_KEYS], dtype=np.float32)

    def _find_target(self, raw_observation: Dict[str, Any]):
        x_min, _x_max, y_min, _y_max, z_min, _z_max = GRID_BOUNDS
        return nearest_log_from_grid(
            raw_observation["log_grid"],
            self._pose(raw_observation),
            x_min=x_min,
            y_min=y_min,
            z_min=z_min,
        )

    def _observation(self, raw_observation: Dict[str, Any]) -> Dict[str, Any]:
        oracle = relative_target_state(self._pose(raw_observation), self.target_xyz)
        observation = {"pov": raw_observation["pov"], "oracle": oracle}
        if self.sensor_profile == SENSOR_PROFILE_F3:
            pose = raw_observation["privileged_pose"]
            observation["telemetry"] = {
                "x": np.float32(pose["xpos"]),
                "y": np.float32(pose["ypos"]),
                "z": np.float32(pose["zpos"]),
                "yaw": np.float32(wrap_degrees(float(pose["yaw"]))),
                "pitch": np.float32(pose["pitch"]),
                "biome_id": np.int32(pose["biome_id"]),
                "biome_temperature": np.float32(pose["biome_temperature"]),
                "biome_rainfall": np.float32(pose["biome_rainfall"]),
            }
        return observation

    def seed(self, seed=None, **kwargs):
        if seed is not None:
            self.env.unwrapped.task.set_episode_seed(int(seed))
        return self.env.seed(seed, **kwargs)

    def reset(self, **kwargs):
        self.elapsed_steps = 0
        raw_observation = self.env.reset(**kwargs)
        self.target_xyz = self._find_target(raw_observation)
        self.grid_target_valid = self.target_xyz is not None
        transformed = self._observation(raw_observation)
        self.previous_distance = (
            float(transformed["oracle"][1]) if self.grid_target_valid else None
        )
        return transformed

    def step(self, action: int):
        raw_observation, raw_reward, raw_done, info = self.env.step(action)
        self.elapsed_steps += 1

        if self.target_xyz is None:
            self.target_xyz = self._find_target(raw_observation)
            self.grid_target_valid = self.target_xyz is not None

        transformed = self._observation(raw_observation)
        current_distance = (
            float(transformed["oracle"][1]) if self.target_xyz is not None else None
        )
        reward = progress_reward(
            self.previous_distance,
            current_distance,
            progress_scale=self.progress_scale,
            step_cost=self.step_cost,
        )
        success = current_distance is not None and current_distance <= self.success_distance
        if success:
            reward += self.success_bonus

        done = bool(raw_done or success or self.elapsed_steps >= self.max_episode_steps)
        info = dict(info)
        if self.elapsed_steps >= self.max_episode_steps and not raw_done and not success:
            info["TimeLimit.truncated"] = True
        info.update(
            {
                "success": bool(success),
                "target_grid_valid": bool(self.grid_target_valid),
                "target_distance": current_distance,
                "oracle": transformed["oracle"].copy(),
                "raw_minerl_reward": float(raw_reward),
            }
        )
        self.previous_distance = current_distance
        return transformed, float(reward), done, info


def make_find_tree_env(
    seed: int = 42,
    max_episode_steps: int = 250,
    yaw_noise_degrees: float = 30.0,
    target_distance_min: int = 5,
    target_distance_max: int = 8,
    distractor_tree_count: int = 0,
    candidate_actions: bool = False,
    sensor_profile: str = "pov_only",
) -> gym.Env:
    """Create the isolated custom task without registering global Gym state."""

    configure_minerl_runtime()
    import minerl  # noqa: F401 - initializes the MineRL runtime

    FindTreeEnvSpec = build_find_tree_spec_class()
    spec = FindTreeEnvSpec(
        max_episode_steps=max(max_episode_steps, 1000),
        yaw_noise_degrees=yaw_noise_degrees,
        target_distance_min=target_distance_min,
        target_distance_max=target_distance_max,
        distractor_tree_count=distractor_tree_count,
    )
    raw_env = spec.make()
    raw_env._is_fault_tolerant = False
    if candidate_actions:
        env = CandidateNavigationActionWrapper(raw_env)
    else:
        env = NavigationDiscreteActionWrapper(raw_env)
    env = FindTreeTaskWrapper(
        env,
        max_episode_steps=max_episode_steps,
        sensor_profile=sensor_profile,
    )
    env.seed(int(seed))
    return env


def close_find_tree_env(env: gym.Env) -> None:
    """Close while reporting MineRL's known already-exited Windows PID race."""

    try:
        env.close()
    except psutil.NoSuchProcess as error:
        print("WARNING: Minecraft already exited during close: {}".format(error))
