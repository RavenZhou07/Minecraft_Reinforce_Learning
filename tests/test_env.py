import os
from collections import Counter

import gym
import numpy as np
import pytest

from mc_rl.wrappers import OneLogTreechopWrapper
from scripts.treechop_random_baseline import summarize, wilson_interval


class RewardSequenceEnv(gym.Env):
    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Dict(
        {"pov": gym.spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8)}
    )

    def __init__(self, rewards, log_counts=None):
        self.rewards = list(rewards)
        self.log_counts = list(log_counts or [0] * len(rewards))
        self.index = 0

    def reset(self):
        self.index = 0
        return self.reset_observation(0)

    def step(self, action):
        reward = self.rewards[self.index]
        self.index += 1
        return self.reset_observation(self.log_counts[self.index - 1]), reward, False, {"source": "fake"}

    @staticmethod
    def reset_observation(log_count=0):
        return {
            "pov": np.zeros((64, 64, 3), dtype=np.uint8),
            "inventory": {"log": int(log_count), "log2": 0},
        }


def test_one_log_wrapper_terminates_on_positive_reward():
    env = OneLogTreechopWrapper(RewardSequenceEnv([0.0, 1.0]), max_episode_steps=10)
    observation = env.reset()
    assert observation["pov"].shape == (64, 64, 3)
    _, _, done, info = env.step(0)
    assert not done
    assert info["success"] is False
    _, reward, done, info = env.step(0)
    assert reward == 1.0
    assert done
    assert info["success"] is True


def test_one_log_wrapper_enforces_step_limit():
    env = OneLogTreechopWrapper(RewardSequenceEnv([0.0, 0.0]), max_episode_steps=2)
    env.reset()
    env.step(0)
    _, _, done, info = env.step(0)
    assert done
    assert info["success"] is False
    assert info["TimeLimit.truncated"] is True


def test_one_log_wrapper_requires_inventory_when_configured():
    env = OneLogTreechopWrapper(
        RewardSequenceEnv([0.0, 1.0], log_counts=[0, 1]),
        max_episode_steps=10,
        require_inventory_confirmation=True,
    )
    env.reset()
    _, _, done, info = env.step(0)
    assert not done and info["success_source"] == "none"
    _, _, done, info = env.step(0)
    assert done and info["success"] is True
    assert info["success_source"] == "inventory"
    assert info["inventory_log_delta"] == 1


def test_positive_reward_does_not_bypass_required_inventory_confirmation():
    env = OneLogTreechopWrapper(
        RewardSequenceEnv([1.0], log_counts=[0]),
        max_episode_steps=1,
        require_inventory_confirmation=True,
    )
    env.reset()
    _, _, done, info = env.step(0)
    assert done and info["success"] is False
    assert info["reward_inventory_mismatch"] is True


def test_random_baseline_summary():
    rows = [
        {
            "success": False,
            "cumulative_reward": 0.0,
            "steps": 10,
            "reset_seconds": 2.0,
            "rollout_seconds": 1.0,
        },
        {
            "success": True,
            "cumulative_reward": 1.0,
            "steps": 5,
            "reset_seconds": 4.0,
            "rollout_seconds": 0.5,
        },
    ]
    summary = summarize(rows, Counter({0: 3, 7: 2}))
    assert summary["success_rate"] == 0.5
    assert summary["mean_reward"] == 0.5
    assert summary["mean_steps"] == 7.5
    assert summary["action_counts"]["noop"] == 3
    assert summary["action_counts"]["attack"] == 2
    lower, upper = wilson_interval(1, 2)
    assert lower < 0.5 < upper


RUN_INTEGRATION = os.environ.get("RUN_MINERL_INTEGRATION") == "1"


@pytest.mark.integration
@pytest.mark.skipif(not RUN_INTEGRATION, reason="set RUN_MINERL_INTEGRATION=1")
@pytest.mark.parametrize("env_id", ["MineRLNavigateDense-v0", "MineRLTreechop-v0"])
def test_real_minerl_reset_step_observation_and_close(env_id):
    # Each environment is started only once, stepped briefly, and closed in a
    # finally block. This checks real Malmo/Minecraft without spawning a fleet.
    import gym
    import minerl  # noqa: F401
    from mc_rl.envs import configure_minerl_runtime

    configure_minerl_runtime()
    env = gym.make(env_id)
    # Integration tests should expose the first launch failure instead of
    # MineRL's normal fault-tolerant launcher restarting Minecraft forever.
    env.unwrapped._is_fault_tolerant = False
    try:
        observation = env.reset()
        pov = observation["pov"]
        assert pov.shape == (64, 64, 3)
        assert pov.dtype == np.uint8
        action = env.action_space.noop()
        observation, reward, done, info = env.step(action)
        assert observation["pov"].shape == (64, 64, 3)
        assert isinstance(float(reward), float)
        assert isinstance(done, (bool, np.bool_))
        assert isinstance(info, dict)
    finally:
        env.close()
