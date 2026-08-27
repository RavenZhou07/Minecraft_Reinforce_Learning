import numpy as np

from scripts.evaluate_natural_treechop_attack_gate_v2a import (
    classify_gate_outcome,
    compact_contact_trace,
    compact_search_trace,
    diagnostic_sample_selected,
    parse_seed_list,
)
from scripts.train_natural_treechop_attack_gate_v2b import (
    split_targeted_by_seed,
    targeted_training_indices,
    temporal_predictions,
)


def test_temporal_replay_requires_episode_steps_from_base_loader(tmp_path):
    from scripts.train_natural_treechop_attack_gate_v2a import load_attack_gate_split

    path = tmp_path / "base.npz"
    np.savez_compressed(
        path,
        pov=np.zeros((2, 4, 4, 4, 3), dtype=np.uint8),
        action=np.asarray([0, 7]),
        previous_action=np.asarray([0, 0]),
        episode_success=np.asarray([1, 1]),
        episode_seed=np.asarray([10, 10]),
        episode_step=np.asarray([4, 5]),
        audit_contact_state=np.asarray(["CENTER_TRUNK", "CENTER_TRUNK"]),
    )
    loaded = load_attack_gate_split(path)
    assert loaded["episode_step"].tolist() == [4, 5]


def test_diagnostic_outcomes_and_selection_keep_errors_and_margin_cases():
    assert classify_gate_outcome(1, 1) == "true_positive"
    assert classify_gate_outcome(0, 1) == "false_positive"
    assert classify_gate_outcome(0, 0) == "true_negative"
    assert classify_gate_outcome(1, 0) == "false_negative"
    assert diagnostic_sample_selected("false_positive", 0.99, 0.6, 0.01)
    assert diagnostic_sample_selected("true_negative", 0.62, 0.6, 0.03)
    assert not diagnostic_sample_selected("true_negative", 0.20, 0.6, 0.03)


def test_exact_seed_list_and_compact_contact_trace_are_auditable():
    assert parse_seed_list("17404, 17407,17412") == [17404, 17407, 17412]
    try:
        parse_seed_list("17404,17404")
    except ValueError as error:
        assert "duplicates" in str(error)
    else:
        raise AssertionError("duplicate diagnostic seeds must fail")
    trace = compact_contact_trace(
        {
            "state": "ATTACK_TRUNK",
            "attempt_id": 3,
            "transition_records": [{"reason": "trunk centred"}],
            "drop_recovery": {"phase": "approach", "steps": 4},
        },
        {"attack_steps": 12, "block_disappearances": 1},
    )
    assert trace["last_transition_reason"] == "trunk centred"
    assert trace["attack_steps"] == 12
    assert trace["block_disappearances"] == 1
    assert trace["drop_phase"] == "approach"

    search_trace = compact_search_trace(
        {
            "state": "APPROACH",
            "remaining_steps": 44,
            "route_distance": 8.5,
            "transitions": [{"reason": "candidate aligned"}],
            "handoff": {"rejections": 3},
        }
    )
    assert search_trace["search_state"] == "APPROACH"
    assert search_trace["search_remaining_steps"] == 44
    assert search_trace["search_route_distance"] == 8.5
    assert search_trace["search_last_transition_reason"] == "candidate aligned"
    assert search_trace["handoff_rejections"] == 3


def test_targeted_false_positive_repeat_uses_student_audit_only_for_indices():
    split = {
        "label": np.asarray([0, 0, 1, 0]),
        "student_gate_decision": np.asarray([1, 0, 1, 1]),
    }
    indices, report = targeted_training_indices(split, false_positive_repeat=3)
    assert indices.tolist() == [0, 1, 2, 3, 0, 3, 0, 3]
    assert report["false_positive_samples"] == 2
    assert report["added_false_positive_samples"] == 4


def test_targeted_seed_split_is_disjoint_and_rejects_autonomous_holdout():
    split = {
        "episode_seed": np.asarray([17100, 17100, 17101, 17101, 17102]),
        "label": np.asarray([0, 1, 0, 1, 0]),
    }
    train, calibration, report = split_targeted_by_seed(split, 1)
    assert set(train["episode_seed"]) == {17100, 17101}
    assert set(calibration["episode_seed"]) == {17102}
    assert report["calibration_seeds"] == [17102]
    split["episode_seed"][-1] = 17200
    try:
        split_targeted_by_seed(split, 1)
    except ValueError as error:
        assert "autonomous holdout" in str(error)
    else:
        raise AssertionError("17200 must never enter targeted training")


def test_temporal_confirmation_resets_on_hold_gap_state_and_seed():
    probabilities = np.asarray([0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9])
    seeds = np.asarray([1, 1, 1, 1, 1, 1, 2])
    steps = np.asarray([5, 6, 7, 8, 10, 11, 1])
    states = np.asarray(["A", "A", "A", "A", "A", "B", "B"])
    predictions = temporal_predictions(
        probabilities, 0.5, seeds, steps, states, confirmation_frames=2
    )
    assert predictions.tolist() == [0, 1, 0, 0, 0, 0, 0]
