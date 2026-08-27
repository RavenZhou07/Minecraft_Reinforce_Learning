import gym
import numpy as np
import pytest

from mc_rl.actions import ACTION_NAMES, discrete_action_to_minerl


class NoopDictSpace(gym.spaces.Dict):
    def noop(self):
        return {
            "forward": 0,
            "back": 0,
            "jump": 0,
            "attack": 0,
            "camera": np.zeros(2, dtype=np.float32),
        }


class FakeMineRLEnv:
    action_space = NoopDictSpace(
        {
            "forward": gym.spaces.Discrete(2),
            "back": gym.spaces.Discrete(2),
            "jump": gym.spaces.Discrete(2),
            "attack": gym.spaces.Discrete(2),
            "camera": gym.spaces.Box(-180.0, 180.0, shape=(2,), dtype=np.float32),
        }
    )


@pytest.mark.parametrize("action_id", range(len(ACTION_NAMES)))
def test_every_discrete_action_is_accepted(action_id):
    env = FakeMineRLEnv()
    action = discrete_action_to_minerl(action_id, env)
    assert env.action_space.contains(action)


def test_camera_order_and_directions_are_pitch_then_yaw():
    env = FakeMineRLEnv()
    np.testing.assert_array_equal(discrete_action_to_minerl(3, env)["camera"], [0, -10])
    np.testing.assert_array_equal(discrete_action_to_minerl(4, env)["camera"], [0, 10])
    np.testing.assert_array_equal(discrete_action_to_minerl(5, env)["camera"], [-10, 0])
    np.testing.assert_array_equal(discrete_action_to_minerl(6, env)["camera"], [10, 0])
    np.testing.assert_array_equal(discrete_action_to_minerl(10, env)["camera"], [0, -5])
    np.testing.assert_array_equal(discrete_action_to_minerl(11, env)["camera"], [0, 5])
    np.testing.assert_array_equal(discrete_action_to_minerl(12, env)["camera"], [-5, 0])
    np.testing.assert_array_equal(discrete_action_to_minerl(13, env)["camera"], [5, 0])


def test_invalid_discrete_action_is_rejected():
    with pytest.raises(ValueError):
        discrete_action_to_minerl(len(ACTION_NAMES), FakeMineRLEnv())


@pytest.mark.integration
@pytest.mark.parametrize("env_id", ["MineRLNavigateDense-v0", "MineRLTreechop-v0"])
def test_actions_are_accepted_by_real_minerl_spaces(env_id):
    """Validate dictionaries against MineRL itself without starting Minecraft."""

    import minerl  # noqa: F401

    env = gym.make(env_id)
    try:
        for action_id in range(len(ACTION_NAMES)):
            action = discrete_action_to_minerl(action_id, env)
            assert env.action_space.contains(action)
    finally:
        env.close()
