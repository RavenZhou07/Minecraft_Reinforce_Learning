from scripts.evaluate_natural_treechop_student import classify_failure, wilson_interval


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
