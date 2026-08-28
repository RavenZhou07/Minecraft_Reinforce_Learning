import json
from pathlib import Path

from scripts.run_treechop_no_action_pipeline import seed29_decision


def rollout_summary(pure_fixed, transitions, dominant, below_cycle, period_2_to_4=0, **progression):
    counts = {
        "meaningful_interaction": 0,
        "approach": 0,
        "contact": 0,
        "valid_attack": 0,
        "block_break": 0,
        "pickup": 0,
        "inventory_acquisition": 0,
    }
    counts.update(progression)
    return {
        "pure_500_step_single_action_fixed_point_episode_count": pure_fixed,
        "median_action_transitions": transitions,
        "median_dominant_action_fraction": dominant,
        "episodes_below_0_80_dominant_period_1_to_4_cycle": below_cycle,
        "episodes_at_or_above_0_80_dominant_period_2_to_4_cycle": period_2_to_4,
        "progression_counts": counts,
    }


def test_seed29_strong_gate_is_exactly_predeclared():
    decision = seed29_decision(rollout_summary(1, 10, 0.949, 3))
    assert decision["replication_eligible"]
    assert decision["classification"] == "replication_gate_passed"
    assert not seed29_decision(rollout_summary(2, 10, 0.949, 3))["replication_eligible"]
    assert not seed29_decision(rollout_summary(1, 9, 0.949, 3))["replication_eligible"]
    assert not seed29_decision(rollout_summary(1, 10, 0.95, 3))["replication_eligible"]
    assert not seed29_decision(rollout_summary(1, 10, 0.949, 2))["replication_eligible"]


def test_deep_progression_is_the_only_alternative_replication_trigger():
    break_decision = seed29_decision(
        rollout_summary(4, 0, 1.0, 0, block_break=1)
    )
    assert break_decision["replication_eligible"]
    assert break_decision["alternative_deep_progression_trigger"]
    assert not seed29_decision(
        rollout_summary(4, 0, 1.0, 0, valid_attack=1)
    )["replication_eligible"]


def test_low_period_replacement_requires_period_2_to_4_not_period_1_dominance():
    period_one_dominant = seed29_decision(
        rollout_summary(0, 1.5, 0.978, 0, period_2_to_4=0)
    )
    assert period_one_dominant["classification"] == "previous_action_removal_partially_changes_dynamics"
    low_period = seed29_decision(
        rollout_summary(0, 20, 0.6, 0, period_2_to_4=3)
    )
    assert low_period["classification"] == "period_1_collapse_replaced_by_low_period_cycle"


def test_frozen_configs_encode_one_capacity_attempt_and_three_formal_seeds():
    capacity = json.loads(
        Path("configs/learning/no_action_multi_capacity_exp15.json").read_text(encoding="utf-8")
    )
    formal = json.loads(
        Path("configs/learning/no_action_formal_seed29_exp16.json").read_text(encoding="utf-8")
    )
    replication = json.loads(
        Path("configs/learning/no_action_replication_exp17.json").read_text(encoding="utf-8")
    )
    assert capacity["bounded_training"] == {
        "maximum_epochs": 1000,
        "checkpoint_selection": "first_epoch_meeting_all_capacity_thresholds",
        "minimum_accuracy": 0.95,
        "minimum_balanced_accuracy": 0.90,
        "required_zero_slot_max_abs": 0.0,
        "attempts": 1,
        "extensions": 0,
    }
    assert formal["training"]["training_seed"] == 29
    assert formal["training"]["minimum_recorded_horizon"] == 60
    assert formal["training"]["maximum_epochs"] == 180
    assert formal["training"]["validation_ce_patience_after_minimum_horizon"] == 30
    assert replication["training_seeds"] == [11, 47]
    assert capacity["protected_splits"] == formal["protected_splits"] == replication["protected_splits"] == ["student_holdout", "final_test"]
