from scripts.evaluate_natural_treechop_student import (
    action_dynamics,
    classify_failure,
    wilson_interval,
)


def test_failure_taxonomy_is_task_progress_ordered():
    assert classify_failure(False, False, False, None, None) == "search_timeout"
    assert classify_failure(False, True, False, None, None) == "approach_timeout"
    assert classify_failure(False, True, True, None, None) == "contact_without_valid_attack"
    assert classify_failure(False, True, True, 10, None) == "attack_without_observed_break"
    assert classify_failure(False, True, True, 10, 20) == "break_without_inventory_pickup"
    assert classify_failure(True, True, True, 10, 20) == "success"


def test_wilson_interval_contains_observed_rate():
    lower, upper = wilson_interval(3, 5)
    assert lower < 0.6 < upper


def test_action_dynamics_reports_fixed_points_without_intervention():
    metrics = action_dynamics([3, 3, 3, 1, 1, 3])
    assert metrics["dominant_action"] == "turn_left"
    assert metrics["dominant_fraction"] == 4 / 6
    assert metrics["max_same_action_streak"] == 3
    assert metrics["action_transitions"] == 2
    assert metrics["action_entropy_nats"] > 0
