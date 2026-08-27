"""Gym wrappers used by the first MineRL learning milestone."""

import csv
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gym

from mc_rl.actions import ACTION_NAMES, discrete_action_to_minerl


LOG_ITEM_KEYS = ("log", "log2")


def inventory_log_count(observation: Dict[str, Any]) -> Optional[int]:
    """Return the observable log count, or ``None`` if inventory is absent."""

    inventory = observation.get("inventory")
    if inventory is None:
        return None
    return int(sum(int(inventory.get(key, 0)) for key in LOG_ITEM_KEYS))


class DiscreteActionWrapper(gym.ActionWrapper):
    """Expose readable integer actions instead of MineRL's Dict space.

    The final four actions are half-step camera motions used only for precise
    resource contact. Existing action ids remain unchanged for checkpoints
    and earlier curriculum logs.
    """

    def __init__(self, env: gym.Env, camera_delta: float = 10.0):
        super().__init__(env)
        self.camera_delta = camera_delta
        self.action_space = gym.spaces.Discrete(len(ACTION_NAMES))

    def action(self, action: int) -> Dict[str, Any]:
        return discrete_action_to_minerl(action, self.env, self.camera_delta)


class OneLogTreechopWrapper(gym.Wrapper):
    """End Treechop on the first positive log reward or at a step limit.

    MineRLTreechop gives +1 whenever a log is collected. This wrapper turns
    that first piece of wood into a compact binary milestone while preserving
    MineRL's original observation and reward.
    """

    def __init__(
        self,
        env: gym.Env,
        max_episode_steps: int = 1000,
        require_inventory_confirmation: bool = False,
    ):
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.require_inventory_confirmation = bool(require_inventory_confirmation)
        self.elapsed_steps = 0
        self.initial_log_count: Optional[int] = None

    def reset(self, **kwargs: Any) -> Any:
        self.elapsed_steps = 0
        observation = self.env.reset(**kwargs)
        self.initial_log_count = inventory_log_count(observation)
        if self.require_inventory_confirmation and self.initial_log_count is None:
            raise KeyError(
                "inventory-confirmed Treechop requires an inventory observation"
            )
        return observation

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        observation, reward, done, info = self.env.step(action)
        self.elapsed_steps += 1

        info = dict(info)
        current_log_count = inventory_log_count(observation)
        inventory_delta = (
            None
            if current_log_count is None or self.initial_log_count is None
            else current_log_count - self.initial_log_count
        )
        inventory_success = bool(inventory_delta is not None and inventory_delta >= 1)
        success = (
            inventory_success
            if self.require_inventory_confirmation
            else bool(inventory_success or reward > 0)
        )
        if success:
            done = True
        if self.elapsed_steps >= self.max_episode_steps and not done:
            done = True
            info["TimeLimit.truncated"] = True
        info["success"] = success
        info["success_source"] = (
            "inventory" if inventory_success else ("reward" if success else "none")
        )
        info["inventory_log_count"] = current_log_count
        info["inventory_log_delta"] = inventory_delta
        info["reward_inventory_mismatch"] = bool(
            reward > 0 and current_log_count is not None and not inventory_success
        )
        return observation, reward, done, info


class EpisodeCSVLogger(gym.Wrapper):
    """Append one compact row of episode statistics when an episode ends."""

    FIELDNAMES = (
        "episode",
        "steps",
        "cumulative_reward",
        "success",
        "duration_seconds",
    )

    def __init__(self, env: gym.Env, csv_path: str):
        super().__init__(env)
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.episode = self._existing_episode_count()
        self._active = False
        self._steps = 0
        self._reward = 0.0
        self._started_at = 0.0

    def _existing_episode_count(self) -> int:
        if not self.csv_path.exists():
            return 0
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            return max(sum(1 for _ in handle) - 1, 0)

    def reset(self, **kwargs: Any) -> Any:
        observation = self.env.reset(**kwargs)
        self._active = True
        self._steps = 0
        self._reward = 0.0
        self._started_at = time.perf_counter()
        return observation

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        observation, reward, done, info = self.env.step(action)
        self._steps += 1
        self._reward += float(reward)
        if done:
            self._write_episode(bool(info.get("success", False)))
        return observation, reward, done, info

    def _write_episode(self, success: bool) -> None:
        if not self._active:
            return
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        self.episode += 1
        duration = time.perf_counter() - self._started_at
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "episode": self.episode,
                    "steps": self._steps,
                    "cumulative_reward": self._reward,
                    "success": success,
                    "duration_seconds": "{:.3f}".format(duration),
                }
            )
        self._active = False
