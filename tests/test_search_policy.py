from typing import Any, Dict, List

import numpy as np

from mc_rl.candidates import ResourceDetection
from mc_rl.resource_adapters import ResourceAdapter
from mc_rl.search_policy import CandidateSearchPolicy, SearchConfig, SearchState
from mc_rl.telemetry import VisualRangeEstimate


class FakeAdapter(ResourceAdapter):
    resource_type = "tree"

    def detect(self, pov: np.ndarray) -> List[ResourceDetection]:
        code = int(pov[0, 0, 0])
        if code == 0:
            return []
        return [ResourceDetection("tree", 0.0, 0.9, float(code))]

    def interaction_action(self) -> int:
        return 8

    def success(
        self, observation: Dict[str, Any], reward: float, info: Dict[str, Any]
    ) -> bool:
        return bool(info.get("success", False))

    def estimate_range(self, detection):
        return VisualRangeEstimate(float(detection.apparent_size), 1.0)


def observation(code=0):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[0, 0, 0] = code
    return {"pov": frame}


def telemetry_observation(code=0, x=0.0, z=0.0, yaw=0.0):
    result = observation(code)
    result["telemetry"] = {
        "x": x,
        "y": 4.0,
        "z": z,
        "yaw": yaw,
        "pitch": 0.0,
        "biome_id": 1,
        "biome_temperature": 0.8,
        "biome_rainfall": 0.4,
    }
    return result


def advance(policy, obs, count):
    for _ in range(count):
        action = policy.act(obs)
        policy.observe_transition(action, obs, 0.0, False, {})


def test_complete_scan_accumulates_commanded_camera_angle():
    policy = CandidateSearchPolicy(
        FakeAdapter(), SearchConfig(full_scan_degrees=30, camera_delta=10)
    )
    advance(policy, observation(), 3)
    assert policy.scan_yaw == 30
    policy.act(observation())
    assert policy.scan_cycles == 1
    assert any(
        row["old_state"] == "SCAN" and row["new_state"] == "BUILD_CANDIDATE_MAP"
        for row in policy.transition_log
    )


def test_failed_selected_candidate_switches_to_second_candidate():
    policy = CandidateSearchPolicy(
        FakeAdapter(), SearchConfig(rescan_after_replan=False)
    )
    first, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 0, 0.9, 100), 0, 0
    )
    second, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 90, 0.9, 50), 0, 0
    )
    first.status = "selected"
    first.approach_attempts = 2
    policy.selected_candidate = first
    policy.state = SearchState.REPLAN
    policy.act(observation())
    assert first.status == "cooldown"
    assert policy.selected_candidate is second
    assert second.status == "selected"


def test_all_candidates_cooling_down_starts_a_new_full_scan():
    policy = CandidateSearchPolicy(FakeAdapter())
    only, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 0, 0.9, 30), 0, 0
    )
    only.status = "selected"
    policy.selected_candidate = only
    policy.state = SearchState.REPLAN
    action = policy.act(observation())
    assert action == policy.config.right_action
    assert policy.state == SearchState.SCAN
    assert only.status == "cooldown"


def test_diagnostic_rank_one_forces_wrong_candidate_then_replan_uses_best():
    policy = CandidateSearchPolicy(
        FakeAdapter(),
        SearchConfig(initial_selection_rank=1, rescan_after_replan=False),
    )
    best, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 0, 0.9, 100), 0, 0
    )
    forced, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 90, 0.9, 20), 0, 0
    )
    policy.state = SearchState.SELECT_TARGET
    policy.act(observation())
    assert policy.selected_candidate is forced
    policy.state = SearchState.REPLAN
    policy.act(observation())
    assert forced.status == "cooldown"
    assert policy.selected_candidate is best


def test_default_replan_discards_stale_bearings_and_rescans():
    policy = CandidateSearchPolicy(FakeAdapter())
    failed, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 0, 0.9, 100), 0, 0
    )
    policy.candidate_map.add_detection(
        ResourceDetection("tree", 100, 0.9, 40), 0, 0
    )
    failed.status = "selected"
    policy.selected_candidate = failed
    policy.state = SearchState.REPLAN
    action = policy.act(observation())
    assert action == policy.config.right_action
    assert policy.state == SearchState.SCAN
    assert policy.candidate_map.candidates == [failed]
    assert failed.status == "cooldown"


def test_empty_scene_state_machine_does_not_internal_loop():
    policy = CandidateSearchPolicy(
        FakeAdapter(), SearchConfig(full_scan_degrees=40, camera_delta=10)
    )
    advance(policy, observation(), 100)
    assert policy.state == SearchState.SCAN
    assert policy.scan_cycles >= 20
    assert not any(
        row["reason"] == "internal transition bound exceeded"
        for row in policy.transition_log
    )


class GuardedObservation(dict):
    def __getitem__(self, key):
        if key != "pov":
            raise AssertionError("deployment policy accessed privileged field {}".format(key))
        return super().__getitem__(key)

    def __iter__(self):
        raise AssertionError("deployment policy iterated privileged observation fields")


def test_deployment_action_path_does_not_access_oracle():
    policy = CandidateSearchPolicy(FakeAdapter())
    guarded = GuardedObservation(pov=observation()["pov"], oracle="forbidden")
    assert policy.act(guarded) == policy.config.right_action


def test_dense_positive_reward_does_not_end_arena_policy():
    from mc_rl.resource_adapters import TreeResourceAdapter

    adapter = TreeResourceAdapter(reward_is_success=False)
    assert not adapter.success(observation(), 0.5, {})
    assert adapter.success(observation(), 0.0, {"success": True})


def test_f3_replan_switches_to_remembered_second_candidate_without_rescan():
    policy = CandidateSearchPolicy(
        FakeAdapter(), SearchConfig(sensor_profile="f3_telemetry")
    )
    first, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 0, 0.9, 5),
        0,
        0,
        telemetry=policy._telemetry(telemetry_observation()),
        range_estimate=VisualRangeEstimate(5.0, 1.0),
    )
    second, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", -90, 0.9, 10),
        0,
        0,
        telemetry=policy._telemetry(telemetry_observation()),
        range_estimate=VisualRangeEstimate(10.0, 1.0),
    )
    first.status = "selected"
    policy.selected_candidate = first
    policy.state = SearchState.REPLAN

    action = policy.act(telemetry_observation())
    assert first.status == "cooldown"
    assert policy.selected_candidate is second
    assert policy.state == SearchState.ALIGN
    assert action == policy.config.left_action
    assert any(
        "remembered world coordinates" in row["reason"]
        for row in policy.transition_log
    )


class F3GuardedObservation(dict):
    def __getitem__(self, key):
        if key not in ("pov", "telemetry"):
            raise AssertionError("deployment policy accessed {}".format(key))
        return super().__getitem__(key)

    def __iter__(self):
        raise AssertionError("deployment policy iterated observation fields")


def test_f3_deployment_path_reads_self_telemetry_but_not_oracle_or_grid():
    policy = CandidateSearchPolicy(
        FakeAdapter(), SearchConfig(sensor_profile="f3_telemetry")
    )
    allowed = telemetry_observation()
    guarded = F3GuardedObservation(
        pov=allowed["pov"],
        telemetry=allowed["telemetry"],
        oracle="forbidden",
        log_grid="forbidden",
    )
    assert policy.act(guarded) == policy.config.right_action
