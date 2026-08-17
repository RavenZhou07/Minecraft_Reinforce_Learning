import numpy as np

from mc_rl.candidates import CandidateMap, CandidateScoreConfig, ResourceDetection
from mc_rl.progress import VisualProgressMonitor


def detection(yaw, size=20.0, confidence=0.8):
    return ResourceDetection("tree", yaw, confidence, size)


def test_temporal_detections_merge_across_adjacent_scan_angles():
    memory = CandidateMap(merge_yaw_degrees=12.0)
    first, merged = memory.add_detection(detection(10, 20), observer_yaw=0, step=0)
    assert not merged
    second, merged = memory.add_detection(detection(1, 22), observer_yaw=10, step=1)
    assert merged
    assert second is first
    assert len(memory.candidates) == 1
    assert first.observation_count == 2
    assert memory.duplicate_candidate_count == 1


def test_different_bearings_do_not_merge():
    memory = CandidateMap(merge_yaw_degrees=15.0)
    memory.add_detection(detection(0), observer_yaw=-80, step=0)
    memory.add_detection(detection(0), observer_yaw=80, step=1)
    assert len(memory.candidates) == 2


def test_default_scan_merge_absorbs_delayed_views_but_not_another_tree():
    memory = CandidateMap()
    memory.add_detection(detection(-25, size=5), observer_yaw=0, step=0)
    memory.add_detection(detection(5, size=20), observer_yaw=0, step=1)
    memory.add_detection(detection(80, size=18), observer_yaw=0, step=2)
    assert len(memory.candidates) == 2
    assert memory.duplicate_candidate_count == 1


def test_candidate_score_prefers_larger_nearer_looking_target():
    memory = CandidateMap(
        score_config=CandidateScoreConfig(
            confidence_weight=1.0,
            size_weight=2.0,
            turn_weight=0.1,
            failure_weight=1.0,
            age_weight=0.0,
        )
    )
    smaller, _ = memory.add_detection(detection(0, size=10), 0, 0)
    larger, _ = memory.add_detection(detection(100, size=80), 0, 0)
    ranked = memory.ranked(current_yaw=0, step=0)
    assert ranked[0] is larger
    assert ranked[0].score_terms["log_size"] > smaller.score_terms["log_size"]


def test_candidate_cooldown_excludes_then_restores_candidate():
    memory = CandidateMap()
    candidate, _ = memory.add_detection(detection(0), 0, 0)
    memory.mark_cooldown(candidate, step=5, cooldown_steps=10)
    assert memory.select(0, 14) is None
    assert memory.select(0, 15) is candidate


def test_cooldown_identity_merges_large_approach_scale_change():
    memory = CandidateMap()
    candidate, _ = memory.add_detection(detection(5, size=20), 0, 0)
    memory.mark_cooldown(candidate, step=10, cooldown_steps=50)
    same, merged = memory.add_detection(detection(8, size=500), 0, 20)
    assert merged
    assert same is candidate
    assert same.status == "cooldown"
    assert len(memory.candidates) == 1


def test_stalled_requires_forward_without_size_or_alignment_progress():
    monitor = VisualProgressMonitor(window_size=15, minimum_forward_steps=10)
    for _ in range(15):
        monitor.add(
            forward=True,
            apparent_size=30.0,
            alignment_error=1.0,
            frame_change=0.001,
            visible=True,
        )
    assert monitor.is_stalled()
    assert monitor.last_diagnostics["size_growth"] == 0.0


def test_growing_candidate_is_not_stalled():
    monitor = VisualProgressMonitor(window_size=15, minimum_forward_steps=10)
    for index in range(15):
        monitor.add(True, 10.0 + 2 * index, 1.0, 0.02, True)
    assert not monitor.is_stalled()
