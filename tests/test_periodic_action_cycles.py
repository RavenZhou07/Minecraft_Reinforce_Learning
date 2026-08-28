from mc_rl.runtime_observability import (
    max_periodic_cycle_streak,
    periodic_cycle_diagnostics,
)


def test_constant_actions_are_period_one_not_false_higher_period_cycle():
    actions = [3] * 500
    result = periodic_cycle_diagnostics(actions)
    assert result["max_period_1_streak"] == 500
    assert result["max_period_2_cycle_streak"] == 0
    assert result["max_period_3_cycle_streak"] == 0
    assert result["max_period_4_cycle_streak"] == 0
    assert result["dominant_period_1_to_4"] == 1
    assert result["fraction_of_episode_in_dominant_period_1_to_4_cycle"] == 1.0
    assert result["pure_single_action_fixed_point"]
    assert result["time_to_first_action_transition"] is None


def test_alternation_is_detected_as_primitive_period_two():
    actions = [3, 4] * 100
    result = periodic_cycle_diagnostics(actions)
    assert result["max_period_1_streak"] == 0
    assert result["max_period_2_cycle_streak"] == 200
    assert result["max_period_4_cycle_streak"] == 0
    assert result["dominant_period_1_to_4"] == 2
    assert result["dominant_period_2_to_4_cycle_fraction"] == 1.0
    assert result["time_to_first_action_transition"] == 1


def test_period_three_and_four_are_kept_separate():
    assert max_periodic_cycle_streak([1, 2, 3] * 20, 3) == 60
    assert max_periodic_cycle_streak([1, 2, 3, 4] * 20, 4) == 80
    assert max_periodic_cycle_streak([1, 2, 1, 2] * 20, 4) == 0


def test_short_nonperiodic_sequence_has_no_cycle():
    result = periodic_cycle_diagnostics([0, 1, 2, 3, 4])
    assert result["fraction_of_episode_in_dominant_period_1_to_4_cycle"] == 0.0
    assert not result["pure_single_action_fixed_point"]
