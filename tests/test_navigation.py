import numpy as np

from mc_rl.find_tree_env import GRID_BOUNDS, build_find_tree_spec_class
from mc_rl.navigation import (
    OracleNavigator,
    nearest_log_from_grid,
    progress_reward,
    relative_target_state,
    relative_yaw_degrees,
    target_bearing_degrees,
)


def test_minecraft_target_bearing_signs():
    assert target_bearing_degrees(0, 1) == 0
    assert target_bearing_degrees(1, 0) == -90
    assert target_bearing_degrees(-1, 0) == 90


def test_nearest_log_grid_uses_y_z_x_order():
    grid = np.zeros((3, 5, 5), dtype=np.uint8)
    # y offset 0, z offset +1, x offset -2 for minima (-1, -2, -2).
    grid[1, 3, 0] = 1
    target = nearest_log_from_grid(
        grid, pose=[10.7, 4.0, -3.2, 0.0, 0.0], x_min=-2, y_min=-1, z_min=-2
    )
    assert target == (8.5, 4.5, -2.5)


def test_oracle_state_and_rule_controller_turn_then_move():
    pose = [0.5, 4.0, 0.5, 0.0, 0.0]
    east_target = [5.5, 4.5, 0.5]
    oracle = relative_target_state(pose, east_target)
    assert np.isclose(oracle[1], 5.0)
    assert np.isclose(relative_yaw_degrees(oracle), -90.0)
    controller = OracleNavigator()
    controller.reset()
    assert controller.act(oracle) == 3
    straight = relative_target_state(pose, [0.5, 4.5, 5.5])
    assert controller.act(straight) == 1

    student_teacher = OracleNavigator(search_clockwise_outside_fov=True)
    student_teacher.reset()
    # A target far to the left is outside the camera; the realizable teacher
    # uses the same clockwise search action as it would for a hidden right target.
    assert student_teacher.act(oracle) == 4


def test_progress_reward_is_positive_when_distance_decreases():
    assert progress_reward(5.0, 4.5) > 0
    assert progress_reward(4.5, 5.0) < 0


def test_find_tree_spec_contains_privileged_handlers_without_patch():
    FindTreeEnvSpec = build_find_tree_spec_class()
    spec = FindTreeEnvSpec(max_episode_steps=1000)
    spec.set_episode_seed(42)
    spec.reset()
    xml = spec.to_xml()
    assert "<ObservationFromGrid>" in xml
    assert "<ObservationFromFullStats/>" in xml
    assert 'type="log"' in xml
    assert {handler.to_string() for handler in spec.observables} == {
        "pov",
        "privileged_pose",
        "log_grid",
    }

    grid_handler = next(
        handler for handler in spec.observables if handler.to_string() == "log_grid"
    )
    cells = ["air"] * int(np.prod(grid_handler.shape))
    cells[0] = "log"
    mask = grid_handler.from_hero({"find_tree_log_grid": cells})
    assert mask.shape == grid_handler.shape
    assert int(mask.sum()) == 1
    assert GRID_BOUNDS == (-16, 16, -1, 4, -16, 16)

    full_stats_handler = next(
        handler
        for handler in spec.observables
        if handler.to_string() == "privileged_pose"
    )
    translated = full_stats_handler.from_hero(
        {
            "xpos": 1.0,
            "ypos": 4.0,
            "zpos": 2.0,
            "yaw": 10.0,
            "pitch": 0.0,
            "biome_id": 1,
            "biome_temperature": 0.8,
            "biome_rainfall": 0.4,
        }
    )
    assert int(translated["biome_id"]) == 1
    assert np.isclose(translated["biome_temperature"], 0.8)


def test_find_tree_yaw_curriculum_is_seeded_and_configurable():
    FindTreeEnvSpec = build_find_tree_spec_class()
    narrow = FindTreeEnvSpec(yaw_noise_degrees=30)
    wide = FindTreeEnvSpec(yaw_noise_degrees=90)
    narrow.set_episode_seed(99)
    wide.set_episode_seed(99)
    narrow.reset()
    wide.reset()
    assert narrow.target_block == wide.target_block
    assert narrow.agent_yaw != wide.agent_yaw


def test_find_tree_distance_curriculum_is_seeded_and_bounded():
    FindTreeEnvSpec = build_find_tree_spec_class()
    spec = FindTreeEnvSpec(target_distance_min=3, target_distance_max=10)
    distances = []
    for seed in range(20):
        spec.set_episode_seed(seed)
        spec.reset()
        x, _y, z = spec.target_block
        distances.append(float(np.hypot(x, z)))
    assert min(distances) >= 2.5
    assert max(distances) <= 10.7
    assert len(set(round(value, 1) for value in distances)) >= 5


def test_find_tree_distractors_are_deterministic_and_farther_than_target():
    FindTreeEnvSpec = build_find_tree_spec_class()
    spec = FindTreeEnvSpec(
        target_distance_min=3,
        target_distance_max=10,
        distractor_tree_count=2,
    )
    spec.set_episode_seed(123)
    spec.reset()
    first_layout = list(spec.tree_blocks)
    spec.reset()
    assert spec.tree_blocks == first_layout
    assert len(first_layout) == 3
    radii = [float(np.hypot(x, z)) for x, _y, z in first_layout]
    assert all(radius > radii[0] for radius in radii[1:])
    assert spec.to_xml().count('type="log"') == 3
