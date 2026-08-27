from typing import Any, Dict, List

import numpy as np

from mc_rl.candidates import CandidateMap, ResourceDetection
from mc_rl.resource_adapters import ResourceAdapter, TreeDetection, TrunkView
from mc_rl.search_policy import CandidateSearchPolicy, SearchConfig, SearchState
from mc_rl.telemetry import RaycastHit, VisualRangeEstimate
from mc_rl.trunk_contact import (
    CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
    CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
    CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
    CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2,
)


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


class HandoffAdapter(FakeAdapter):
    def ready_to_interact(self, detection):
        return detection is not None

    def trunk_view(self, pov):
        return TrunkView(
            present=False,
            center_x=0.5,
            center_y=0.5,
            bottom_y=1.0,
            width_px=0.0,
            height_px=0.0,
            area_px=0.0,
            crosshair_trunk_fraction=0.0,
            horizontal_yaw=0.0,
            vertical_offset_deg=0.0,
            clipped_vertical=False,
        )


class NotReadyHandoffAdapter(HandoffAdapter):
    def ready_to_interact(self, detection):
        return False


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


def test_f3_align_prefers_stable_world_route_over_noisy_visual_bearing():
    class NoisyBearingAdapter(FakeAdapter):
        def detect(self, pov):
            return [ResourceDetection("tree", 7.0, 0.9, 10.0)]

    policy = CandidateSearchPolicy(
        NoisyBearingAdapter(), SearchConfig(sensor_profile="f3_telemetry")
    )
    candidate, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 0.0, 0.9, 10.0), 0.0, 0
    )
    candidate.estimated_world_x = 0.0
    candidate.estimated_world_y = 4.0
    candidate.estimated_world_z = 10.0
    candidate.position_uncertainty = 1.0
    candidate.last_position_update_step = 0
    candidate.status = "selected"
    policy.selected_candidate = candidate
    policy.state = SearchState.ALIGN

    policy.act(telemetry_observation(code=10, yaw=0.0))

    assert policy.state == SearchState.APPROACH
    assert any(
        row["old_state"] == "ALIGN" and row["new_state"] == "APPROACH"
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


def test_f3_predicted_coordinate_without_success_replans_to_next_candidate():
    policy = CandidateSearchPolicy(
        FakeAdapter(), SearchConfig(sensor_profile="f3_telemetry")
    )
    telemetry = policy._telemetry(telemetry_observation())
    first, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 0, 0.9, 2),
        0,
        0,
        telemetry=telemetry,
        range_estimate=VisualRangeEstimate(2.0, 1.0),
    )
    second, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", -90, 0.9, 10),
        0,
        0,
        telemetry=telemetry,
        range_estimate=VisualRangeEstimate(10.0, 1.0),
    )
    first.status = "selected"
    policy.selected_candidate = first
    policy.state = SearchState.LOCAL_REACQUIRE
    policy._local_trigger = "predicted world coordinate reached without visual contact"
    policy._local_actions.clear()

    policy.act(telemetry_observation())
    assert first.status == "cooldown"
    assert policy.selected_candidate is second
    assert policy.replan_count == 1


def test_selected_detection_ties_use_stable_input_order():
    policy = CandidateSearchPolicy(FakeAdapter())
    candidate, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 0, 0.9, 10), 0, 0
    )
    candidate.status = "selected"
    policy.selected_candidate = candidate
    first = ResourceDetection("tree", 0, 0.9, 10)
    second = ResourceDetection("tree", 0, 0.9, 10)
    assert policy._selected_detection([first, second], None) is first


def _selected_world_candidate(policy, x=0.0, z=5.0, uncertainty=1.0):
    telemetry = policy._telemetry(telemetry_observation())
    candidate, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 0.0, 0.9, max(z, 1.0)),
        0.0,
        0,
        telemetry=telemetry,
        range_estimate=VisualRangeEstimate(max(z, 1.0), uncertainty),
    )
    candidate.estimated_world_x = x
    candidate.estimated_world_z = z
    candidate.position_uncertainty = uncertainty
    candidate.status = "selected"
    policy.selected_candidate = candidate
    policy.current_telemetry = telemetry
    return candidate, telemetry


def test_v9_3_route_snapshot_does_not_follow_later_rgb_position_drift():
    policy = CandidateSearchPolicy(
        HandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
        ),
    )
    candidate, _telemetry = _selected_world_candidate(policy, z=10.0)
    assert policy._world_route()[1] == 10.0
    candidate.estimated_world_z = 30.0
    assert policy._world_route()[1] == 10.0


def test_v9_4_inherits_v9_3_route_snapshot_and_strict_candidate_merge():
    policy = CandidateSearchPolicy(
        HandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
        ),
    )
    candidate, _telemetry = _selected_world_candidate(policy, z=10.0)
    assert policy._world_route()[1] == 10.0
    candidate.estimated_world_z = 30.0
    assert policy._world_route()[1] == 10.0
    assert policy.candidate_map.cooldown_yaw_only_merge is False


def test_v9_5_inherits_frozen_route_snapshot_and_strict_candidate_merge():
    policy = CandidateSearchPolicy(
        HandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
        ),
    )
    candidate, _telemetry = _selected_world_candidate(policy, z=10.0)
    assert policy._world_route()[1] == 10.0
    candidate.estimated_world_z = 30.0
    assert policy._world_route()[1] == 10.0
    assert policy.candidate_map.cooldown_yaw_only_merge is False


def test_v9_5_active_contact_keeps_action_ownership_across_global_replan_state():
    policy = CandidateSearchPolicy(
        HandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
        ),
    )
    candidate, telemetry = _selected_world_candidate(policy, z=3.0)
    policy._contact.engage(
        telemetry=telemetry,
        candidate_id=candidate.candidate_id,
        global_step=10,
        target_hint=(0.0, 3.0),
    )
    policy.state = SearchState.REPLAN

    action = policy.act(telemetry_observation())

    assert action in set(policy._contact._exact_log_rescan_sequence())
    assert policy.selected_candidate is candidate
    assert policy._contact.active
    assert policy._contact.candidate_id == candidate.candidate_id
    assert policy.contact_owner_lock_steps == 1
    assert policy.replan_count == 0


def test_v9_5_contact_owner_mismatch_cancels_local_controller_before_replan():
    policy = CandidateSearchPolicy(
        HandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
        ),
    )
    candidate, telemetry = _selected_world_candidate(policy, z=3.0)
    policy._contact.engage(
        telemetry=telemetry,
        candidate_id=candidate.candidate_id + 1,
        global_step=10,
        target_hint=(0.0, 3.0),
    )
    policy.state = SearchState.REPLAN

    policy.act(telemetry_observation())

    assert policy.contact_owner_mismatches == 1
    assert not policy._contact.active
    assert policy._contact.result == "replan"


def test_v9_5_route_stall_and_far_visual_trunk_cannot_trigger_handoff():
    policy = CandidateSearchPolicy(
        HandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
        ),
    )
    _candidate, telemetry = _selected_world_candidate(policy, z=17.0)
    detection = TreeDetection(
        "tree", 0.0, 0.9, 10.0, sees_trunk=True
    )

    action, engaged = policy._contact_step(
        observation()["pov"],
        telemetry,
        (0.0, 17.0),
        detection,
        route_stalled=True,
    )

    assert action is None and engaged is False
    assert policy.handoff_spatial_rejections == 1
    assert policy._contact is not None and not policy._contact.engaged


def test_v9_4_frozen_route_stall_can_still_trigger_visual_handoff():
    policy = CandidateSearchPolicy(
        NotReadyHandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
        ),
    )
    _candidate, telemetry = _selected_world_candidate(policy, z=17.0)
    detection = TreeDetection(
        "tree", 0.0, 0.9, 10.0, sees_trunk=True
    )

    _action, engaged = policy._contact_step(
        observation()["pov"],
        telemetry,
        (0.0, 17.0),
        detection,
        route_stalled=True,
    )

    assert engaged is True
    assert policy._contact is not None and policy._contact.engaged


def test_v9_5_nearby_log_raycast_can_override_visual_region_guard():
    policy = CandidateSearchPolicy(
        NotReadyHandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
        ),
    )
    _candidate, telemetry = _selected_world_candidate(policy, z=17.0)
    raycast = RaycastHit(True, True, False, False, 10.0, 0.0, 5.0, 10.0)

    _action, engaged = policy._contact_step(
        observation()["pov"],
        telemetry,
        (0.0, 17.0),
        None,
        raycast=raycast,
        route_stalled=True,
    )

    assert engaged is True
    assert policy.handoff_raycast_confirmations == 1
    assert policy._contact is not None and policy._contact.engaged


def test_v9_5_world_route_owns_coarse_navigation_outside_contact_region():
    policy = CandidateSearchPolicy(
        NotReadyHandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
        ),
    )
    _selected_world_candidate(policy, z=17.0)
    policy.state = SearchState.APPROACH

    action = policy.act(telemetry_observation(code=3, yaw=0.0))

    assert action == policy.config.forward_action
    assert policy.state == SearchState.APPROACH
    assert policy._contact is not None and not policy._contact.engaged


def test_v9_5_stalled_route_inside_region_relocalizes_without_visual_evidence():
    policy = CandidateSearchPolicy(
        NotReadyHandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
        ),
    )
    candidate, _telemetry = _selected_world_candidate(policy, z=7.0)
    policy.state = SearchState.APPROACH
    policy._last_action = policy.config.forward_action
    policy._route_distances.extend([7.0] * 23)

    action = policy.act(telemetry_observation())

    assert action == policy.config.right_action
    assert candidate.status == "cooldown"
    assert policy.state == SearchState.SCAN
    assert policy.handoff_relocalization_scans == 1
    assert any(
        row["reason"]
        == "world route stalled inside contact region without local evidence"
        for row in policy.transition_log
    )


def test_v9_2_route_still_tracks_live_candidate_position():
    policy = CandidateSearchPolicy(
        HandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2,
        ),
    )
    candidate, _telemetry = _selected_world_candidate(policy, z=10.0)
    assert policy._world_route()[1] == 10.0
    candidate.estimated_world_z = 30.0
    assert policy._world_route()[1] == 30.0


def test_v9_3_handoff_rejects_near_coordinate_without_local_evidence():
    policy = CandidateSearchPolicy(
        HandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
        ),
    )
    _candidate, telemetry = _selected_world_candidate(
        policy, z=3.0, uncertainty=8.0
    )
    action, engaged = policy._contact_step(
        observation()["pov"], telemetry, (0.0, 3.0), None
    )
    assert action is None and engaged is False
    assert policy.handoff_guard_rejections == 1
    assert policy._contact is not None and not policy._contact.engaged
    assert policy._contact_region_limit() == 8.0


def test_v9_3_ready_geometry_cannot_bypass_handoff_evidence_guard():
    policy = CandidateSearchPolicy(
        HandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
        ),
    )
    _selected_world_candidate(policy, z=3.0)
    policy.state = SearchState.APPROACH
    action = policy.act(telemetry_observation(code=3))
    assert action != policy.adapter.interaction_action()
    assert policy.state == SearchState.LOCAL_REACQUIRE
    assert policy._contact is not None and not policy._contact.engaged
    assert policy.handoff_guard_rejections >= 1


def test_v9_3_handoff_accepts_aligned_local_trunk_evidence():
    policy = CandidateSearchPolicy(
        HandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
        ),
    )
    _candidate, telemetry = _selected_world_candidate(policy, z=3.0)
    trunk = TreeDetection(
        resource_type="tree",
        horizontal_yaw=4.0,
        confidence=0.9,
        apparent_size=200.0,
        sees_trunk=True,
    )
    action, engaged = policy._contact_step(
        observation()["pov"], telemetry, (0.0, 3.0), trunk
    )
    assert engaged is True
    assert action is not None
    assert policy.handoff_visual_confirmations == 1


def test_v9_2_near_coordinate_handoff_remains_unchanged_without_evidence():
    policy = CandidateSearchPolicy(
        HandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2,
        ),
    )
    _candidate, telemetry = _selected_world_candidate(policy, z=3.0)
    action, engaged = policy._contact_step(
        observation()["pov"], telemetry, (0.0, 3.0), None
    )
    assert engaged is True
    assert action is not None


def test_v9_3_bad_handoff_forces_fresh_scan_and_discards_stale_available():
    policy = CandidateSearchPolicy(
        HandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
        ),
    )
    failed, _telemetry = _selected_world_candidate(policy, z=3.0)
    stale, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 90.0, 0.9, 12.0), 0.0, 0
    )
    policy._contact._attempt_results = [
        {"reason": "bounded exact log rescan found no eligible local log"}
    ]
    policy._note_contact_replan()
    policy.state = SearchState.REPLAN
    action = policy.act(telemetry_observation())
    assert action == policy.config.right_action
    assert policy.state == SearchState.SCAN
    assert failed in policy.candidate_map.candidates
    assert stale not in policy.candidate_map.candidates
    assert policy.handoff_relocalization_scans == 1
    assert failed.failed_until_step > policy.config.episode_max_steps


def test_v9_3_failed_coordinate_recovery_triggers_upstream_relocalization():
    policy = CandidateSearchPolicy(
        HandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
        ),
    )
    _selected_world_candidate(policy, z=3.0)
    policy._contact._attempt_results = [
        {
            "reason": (
                "post-recovery translation budget exhausted without progress"
            )
        }
    ]
    policy._note_contact_replan()
    assert policy._force_handoff_relocalization is True


def test_v9_3_predicted_coordinate_miss_rebuilds_from_current_pose():
    policy = CandidateSearchPolicy(
        HandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
        ),
    )
    failed, _telemetry = _selected_world_candidate(policy, z=3.0)
    stale, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 90.0, 0.9, 12.0), 0.0, 0
    )
    policy.state = SearchState.LOCAL_REACQUIRE
    policy._local_trigger = (
        "predicted world coordinate reached without local trunk evidence"
    )
    policy._local_actions.clear()
    action = policy.act(telemetry_observation())
    assert action == policy.config.right_action
    assert policy.state == SearchState.SCAN
    assert failed in policy.candidate_map.candidates
    assert stale not in policy.candidate_map.candidates
    assert policy.handoff_relocalization_scans == 1


def test_v9_3_late_bad_handoff_skips_full_rescan_to_preserve_episode_budget():
    policy = CandidateSearchPolicy(
        HandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
            episode_max_steps=300,
            handoff_relocalization_reserve_steps=56,
        ),
    )
    _failed, _telemetry = _selected_world_candidate(policy, z=3.0)
    stale, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 90.0, 0.9, 12.0), 0.0, 0
    )
    stale.estimated_world_x = 10.0
    stale.estimated_world_z = 0.0
    stale.position_uncertainty = 1.0
    policy.step = 245
    policy._force_handoff_relocalization = True
    policy.state = SearchState.REPLAN
    policy.act(telemetry_observation())
    assert policy.handoff_relocalization_scans == 0
    assert policy.handoff_relocalization_skipped_late == 1
    assert policy.selected_candidate is stale


def test_v9_3_handoff_diagnostics_expose_frozen_route_snapshot():
    policy = CandidateSearchPolicy(
        HandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
        ),
    )
    candidate, _telemetry = _selected_world_candidate(
        policy, x=2.0, z=10.0, uncertainty=5.0
    )
    policy._world_route()
    candidate.estimated_world_x = 20.0
    diagnostics = policy.handoff_diagnostics()
    assert diagnostics["route_target_x"] == 2.0
    assert diagnostics["route_target_z"] == 10.0
    assert diagnostics["contact_region_limit"] == 8.0


def test_v9_3_cooldown_candidate_needs_world_position_match_after_translation():
    memory = CandidateMap(cooldown_yaw_only_merge=False)
    first, _ = memory.add_detection(
        ResourceDetection("tree", 0.0, 0.9, 10.0),
        0.0,
        0,
        telemetry=CandidateSearchPolicy(
            FakeAdapter(), SearchConfig(sensor_profile="f3_telemetry")
        )._telemetry(telemetry_observation()),
        range_estimate=VisualRangeEstimate(10.0, 1.0),
    )
    first.status = "cooldown"
    moved_policy = CandidateSearchPolicy(
        FakeAdapter(), SearchConfig(sensor_profile="f3_telemetry")
    )
    moved = moved_policy._telemetry(telemetry_observation(z=20.0))
    second, merged = memory.add_detection(
        ResourceDetection("tree", 0.0, 0.9, 10.0),
        0.0,
        10,
        telemetry=moved,
        range_estimate=VisualRangeEstimate(10.0, 1.0),
    )
    assert merged is False
    assert second is not first
    assert len(memory.candidates) == 2


def test_default_candidate_map_keeps_frozen_yaw_only_cooldown_merge():
    memory = CandidateMap()
    first, _ = memory.add_detection(
        ResourceDetection("tree", 0.0, 0.9, 10.0), 0.0, 0
    )
    first.status = "cooldown"
    second, merged = memory.add_detection(
        ResourceDetection("tree", 1.0, 0.9, 10.0), 0.0, 10
    )
    assert merged is True
    assert second is first
    assert len(memory.candidates) == 1
