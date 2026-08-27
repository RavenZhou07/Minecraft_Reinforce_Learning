"""Fast tests for natural-tree candidates and the trunk contact controller."""

from math import hypot
from typing import Any, Dict, List

import numpy as np

from mc_rl.candidates import CandidateMap, ResourceCandidate, ResourceDetection
from mc_rl.resource_adapters import TreeDetection, TreeResourceAdapter, TrunkView
from mc_rl.search_policy import CandidateSearchPolicy, SearchConfig, SearchState
from mc_rl.telemetry import AgentTelemetry, RaycastHit, VisualRangeEstimate
from mc_rl.trunk_contact import (
    CONTACT_PROFILE_CLEAR_OCCLUSION,
    CONTACT_PROFILE_DROP_RECOVERY,
    CONTACT_PROFILE_COORDINATE_RECOVERY_V9_1,
    CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2,
    CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
    CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
    CONTACT_PROFILE_V6_1,
    TrunkContactController,
    TrunkContactConfig,
    TrunkContactState,
)


POV = np.zeros((64, 64, 3), dtype=np.uint8)
POV[:] = (90, 120, 60)  # plain background, never trunk-coloured


def telemetry(x=0.0, z=0.0, yaw=0.0, pitch=0.0) -> AgentTelemetry:
    return AgentTelemetry(x=x, y=64.0, z=z, yaw=yaw, pitch=pitch)


def telemetry_dict(x=0.0, z=0.0, yaw=0.0, pitch=0.0) -> Dict[str, Any]:
    return {
        "x": x,
        "y": 64.0,
        "z": z,
        "yaw": yaw,
        "pitch": pitch,
        "biome_id": 4,
        "biome_temperature": 0.7,
        "biome_rainfall": 0.8,
    }


def observation(pov=POV, **telemetry_kwargs) -> Dict[str, Any]:
    return {"pov": pov, "telemetry": telemetry_dict(**telemetry_kwargs)}


def trunk_view(
    center_x=0.5,
    center_y=0.5,
    area=200.0,
    height_px=30.0,
    width_px=8.0,
    crosshair=0.5,
    material="oak",
) -> TrunkView:
    return TrunkView(
        present=True,
        center_x=center_x,
        center_y=center_y,
        bottom_y=min(1.0, center_y + height_px / 126.0),
        width_px=width_px,
        height_px=height_px,
        area_px=area,
        crosshair_trunk_fraction=crosshair,
        horizontal_yaw=(center_x - 0.5) * 70.0,
        vertical_offset_deg=(center_y - 0.5) * 70.0,
        clipped_vertical=False,
        material=material,
    )


ABSENT_VIEW = TrunkView(
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


class ScriptedTrunkAdapter(TreeResourceAdapter):
    """Tree adapter whose visual layer is scripted by the test."""

    def __init__(self):
        super().__init__(interaction_size=None, reward_is_success=True)
        self.views: List[TrunkView] = []
        self.cursor = 0
        self.occlusion_fraction = 0.0

    def set_views(self, *views: TrunkView) -> None:
        self.views = list(views)
        self.cursor = 0

    def trunk_view(self, pov: np.ndarray) -> TrunkView:
        if not self.views:
            return ABSENT_VIEW
        view = self.views[min(self.cursor, len(self.views) - 1)]
        self.cursor += 1
        return view

    def detect(self, pov: np.ndarray) -> List[ResourceDetection]:
        return []

    def leaf_occlusion_fraction(self, pov: np.ndarray) -> float:
        return self.occlusion_fraction


def make_detection(
    yaw_deg: float,
    apparent_size: float,
    sees_trunk: bool,
    confidence: float = 0.8,
) -> TreeDetection:
    """Build a detection whose visual range projects to a wanted world point."""

    return TreeDetection(
        resource_type="tree",
        horizontal_yaw=yaw_deg,
        confidence=confidence,
        apparent_size=apparent_size,
        geometry_size=apparent_size,
        sees_trunk=sees_trunk,
        angular_height_deg=20.0 if sees_trunk else None,
        angular_width_deg=6.0 if sees_trunk else None,
    )


def add_view(
    memory: CandidateMap,
    det: ResourceDetection,
    observer: AgentTelemetry,
    range_estimate: VisualRangeEstimate,
    step: int,
):
    return memory.add_detection(
        det,
        observer_yaw=observer.yaw,
        step=step,
        telemetry=observer,
        range_estimate=range_estimate,
    )


# ---------------------------------------------------------------------------
# Candidate layer: trunk/leaves hypotheses, merging, and range fusion
# ---------------------------------------------------------------------------


def test_same_tree_trunk_and_canopy_views_merge():
    memory = CandidateMap()
    observer = telemetry()
    canopy = make_detection(0.0, 300.0, False)
    trunk = make_detection(2.0, 40.0, True)
    first, merged_first = add_view(
        memory, canopy, observer, VisualRangeEstimate(8.0, 3.0, "canopy_size"), 0
    )
    second, merged_second = add_view(
        memory, trunk, observer, VisualRangeEstimate(8.4, 1.4, "trunk_height"), 1
    )
    assert merged_first is False
    assert merged_second is True
    assert second is first
    assert first.trunk_observations == 1
    assert first.canopy_observations == 1
    assert len(memory.candidates) == 1


def test_trunk_first_then_canopy_still_one_hypothesis():
    memory = CandidateMap()
    observer = telemetry()
    trunk = make_detection(-1.0, 45.0, True)
    canopy = make_detection(1.0, 280.0, False)
    first, _ = add_view(
        memory, trunk, observer, VisualRangeEstimate(8.0, 1.4, "trunk_height"), 0
    )
    second, merged = add_view(
        memory, canopy, observer, VisualRangeEstimate(8.6, 3.2, "canopy_size"), 1
    )
    assert merged
    assert second is first
    assert first.trunk_observations == 1
    assert first.canopy_observations == 1


def test_different_trees_are_not_merged():
    memory = CandidateMap()
    observer = telemetry()
    left = make_detection(-25.0, 60.0, True)
    right = make_detection(25.0, 55.0, True)
    add_view(
        memory, left, observer, VisualRangeEstimate(5.0, 1.4, "trunk_height"), 0
    )
    add_view(
        memory, right, observer, VisualRangeEstimate(8.0, 1.6, "trunk_height"), 1
    )
    assert len(memory.candidates) == 2


def test_low_confidence_canopy_range_does_not_override_trunk_fix():
    memory = CandidateMap()
    observer = telemetry()
    trunk = make_detection(0.0, 50.0, True)
    candidate, _ = add_view(
        memory, trunk, observer, VisualRangeEstimate(8.0, 1.2, "trunk_height"), 0
    )
    trunk_point = (candidate.estimated_world_x, candidate.estimated_world_z)
    # Canopy view landing ~3.4 blocks away with 4-block uncertainty.
    canopy = make_detection(-17.0, 260.0, False)
    candidate, merged = add_view(
        memory, canopy, observer, VisualRangeEstimate(11.0, 4.0, "canopy_size"), 1
    )
    assert merged
    drift = hypot(
        float(candidate.estimated_world_x) - float(trunk_point[0]),
        float(candidate.estimated_world_z) - float(trunk_point[1]),
    )
    assert drift <= 1.2


def test_correlated_position_views_smooth_without_fake_uncertainty_shrink():
    memory = CandidateMap()
    observer = telemetry()
    first = make_detection(0.0, 50.0, True)
    second = make_detection(4.0, 52.0, True)
    candidate, _ = add_view(
        memory, first, observer, VisualRangeEstimate(8.0, 1.2, "trunk_height"), 0
    )
    initial_uncertainty = float(candidate.position_uncertainty)
    candidate, merged = add_view(
        memory, second, observer, VisualRangeEstimate(8.8, 1.2, "trunk_height"), 1
    )
    assert merged
    assert float(candidate.position_uncertainty) == initial_uncertainty
    assert candidate.position_observation_count == 2
    midpoint_z = 0.5 * (8.0 + 8.8)
    assert abs(float(candidate.estimated_world_z) - midpoint_z) < 0.5


def test_translated_position_view_can_shrink_uncertainty():
    memory = CandidateMap()
    first_observer = telemetry()
    moved_observer = telemetry(x=2.0)
    candidate, _ = add_view(
        memory,
        make_detection(0.0, 50.0, True),
        first_observer,
        VisualRangeEstimate(8.0, 1.4, "trunk_height"),
        0,
    )
    initial_uncertainty = float(candidate.position_uncertainty)
    candidate, merged = add_view(
        memory,
        make_detection(14.0, 52.0, True),
        moved_observer,
        VisualRangeEstimate(8.25, 1.4, "trunk_height"),
        5,
    )
    assert merged
    assert float(candidate.position_uncertainty) < initial_uncertainty


def test_conflicting_updates_lower_confidence_then_split():
    memory = CandidateMap()
    observer = telemetry()
    first = make_detection(0.0, 50.0, True)
    candidate, _ = add_view(
        memory, first, observer, VisualRangeEstimate(8.0, 1.2, "trunk_height"), 0
    )
    original_confidence = candidate.confidence
    far = make_detection(40.0, 48.0, True)
    memory.update_candidate_position(
        candidate, far, observer, VisualRangeEstimate(9.0, 1.2, "trunk_height"), 5
    )
    assert candidate.position_conflicts == 1
    assert candidate.confidence < original_confidence
    assert memory.split_candidate_count == 0
    memory.update_candidate_position(
        candidate, far, observer, VisualRangeEstimate(9.0, 1.2, "trunk_height"), 6
    )
    assert candidate.position_conflicts == 2
    assert memory.split_candidate_count == 1


def test_overlapping_duplicates_are_penalised_in_ranking():
    memory = CandidateMap()
    # Two hypotheses that survived merging at nearly the same world point.
    memory.candidates.append(
        ResourceCandidate(
            candidate_id=1,
            resource_type="tree",
            relative_yaw=0.0,
            confidence=0.8,
            apparent_size=50.0,
            last_seen_step=0,
            estimated_world_x=0.0,
            estimated_world_y=64.0,
            estimated_world_z=8.0,
            position_uncertainty=1.0,
        )
    )
    memory.candidates.append(
        ResourceCandidate(
            candidate_id=2,
            resource_type="tree",
            relative_yaw=2.0,
            confidence=0.8,
            apparent_size=52.0,
            last_seen_step=0,
            estimated_world_x=0.3,
            estimated_world_y=64.0,
            estimated_world_z=8.4,
            position_uncertainty=1.0,
        )
    )
    ranked = memory.ranked(current_yaw=0.0, step=0)
    assert len(ranked) == 2
    assert all(candidate.overlap_count == 1 for candidate in ranked)
    assert ranked[0].score_terms["overlap"] < 0.0


# ---------------------------------------------------------------------------
# Trunk contact controller
# ---------------------------------------------------------------------------


def make_controller(adapter: ScriptedTrunkAdapter, **config_overrides):
    config = TrunkContactConfig(**config_overrides)
    controller = TrunkContactController(adapter, config)
    controller.engage()
    return controller


def test_route_stall_outside_contact_region_tries_obstacle_recovery_first():
    adapter = ScriptedTrunkAdapter()
    policy = CandidateSearchPolicy(
        adapter, SearchConfig(sensor_profile="f3_telemetry")
    )
    candidate, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 0.0, 0.9, 30.0),
        observer_yaw=0.0,
        step=0,
        telemetry=telemetry(pitch=10.0),
        range_estimate=VisualRangeEstimate(13.5, 1.0),
    )
    candidate.status = "selected"
    policy.selected_candidate = candidate
    policy.state = SearchState.APPROACH
    # Simulate 24 forward steps during which the route distance froze.
    policy._route_distances.extend([13.5] * 24)
    action = policy.act(observation(pitch=10.0))
    assert action == policy.config.forward_jump_action
    assert policy._contact is not None and not policy._contact.engaged
    assert policy.obstacle_recovery_count == 1


def test_entering_region_switches_to_find_trunk():
    adapter = ScriptedTrunkAdapter()
    policy = CandidateSearchPolicy(
        adapter, SearchConfig(sensor_profile="f3_telemetry")
    )
    candidate, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 0.0, 0.9, 30.0),
        observer_yaw=0.0,
        step=0,
        telemetry=telemetry(pitch=10.0),
        range_estimate=VisualRangeEstimate(1.5, 1.0),
    )
    candidate.status = "selected"
    policy.selected_candidate = candidate
    policy.state = SearchState.APPROACH
    action = policy.act(observation(pitch=10.0))
    assert policy.contact_state == "FIND_TRUNK"
    assert action in (policy.config.left_action, policy.config.right_action)


def test_v6_1_profile_is_frozen_without_occlusion_clearing():
    config = TrunkContactConfig.for_profile(CONTACT_PROFILE_V6_1)
    assert config.enable_clear_occlusion is False
    assert config.enable_drop_recovery is False


def test_drop_recovery_profile_keeps_older_profiles_frozen():
    clear = TrunkContactConfig.for_profile(CONTACT_PROFILE_CLEAR_OCCLUSION)
    recovery = TrunkContactConfig.for_profile(CONTACT_PROFILE_DROP_RECOVERY)
    assert clear.enable_clear_occlusion is True
    assert clear.enable_drop_recovery is False
    assert recovery.enable_clear_occlusion is True
    assert recovery.enable_drop_recovery is True
    assert recovery.region_max_steps == 100


def test_persistent_canopy_triggers_bounded_clear_profile():
    adapter = ScriptedTrunkAdapter()
    adapter.occlusion_fraction = 0.9
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_CLEAR_OCCLUSION,
        leaf_occlusion_consecutive_frames=2,
        occlusion_clear_steps=3,
    )
    controller = TrunkContactController(adapter, config)
    controller.engage(find_first=True)

    controller.act(POV, telemetry())
    action = controller.act(POV, telemetry())
    assert controller.state == TrunkContactState.CLEAR_OCCLUSION
    assert action == controller.config.clear_occlusion_action

    actions = [action]
    for _ in range(5):
        actions.append(controller.act(POV, telemetry()))
        if controller.state == TrunkContactState.FIND_TRUNK:
            break
    assert controller.state == TrunkContactState.FIND_TRUNK
    assert controller.counters.occlusion_clears == 1
    assert controller.counters.occlusion_clear_steps == 3
    assert actions.count(controller.config.clear_occlusion_action) == 3


def test_trunk_exposed_during_clear_immediately_stops_leaf_attack():
    adapter = ScriptedTrunkAdapter()
    adapter.occlusion_fraction = 0.9
    adapter.set_views(ABSENT_VIEW, ABSENT_VIEW, trunk_view(center_x=0.5))
    controller = TrunkContactController(
        adapter,
        TrunkContactConfig.for_profile(
            CONTACT_PROFILE_CLEAR_OCCLUSION,
            leaf_occlusion_consecutive_frames=2,
        ),
    )
    controller.engage(find_first=True)
    controller.act(POV, telemetry())
    assert controller.act(POV, telemetry()) == controller.config.clear_occlusion_action
    action = controller.act(POV, telemetry())
    assert controller.state in (
        TrunkContactState.CENTER_TRUNK,
        TrunkContactState.ADJUST_PITCH,
        TrunkContactState.ATTACK_TRUNK,
    )
    assert action != controller.config.clear_occlusion_action


def test_privileged_raycast_log_in_reach_attacks_without_rgb_trunk():
    adapter = ScriptedTrunkAdapter()
    controller = make_controller(adapter)
    controller.state = TrunkContactState.FIND_TRUNK
    hit = RaycastHit(True, True, False, True, 3.0, 1.0, 65.0, 2.0)
    action = controller.act(POV, telemetry(), raycast=hit)
    assert action == controller.config.attack_action
    assert controller.state == TrunkContactState.ATTACK_TRUNK
    assert controller.counters.raycast_log_actions >= 1


def test_privileged_raycast_log_out_of_reach_moves_forward():
    adapter = ScriptedTrunkAdapter()
    controller = make_controller(adapter)
    controller.state = TrunkContactState.FIND_TRUNK
    hit = RaycastHit(True, True, False, False, 7.0, 1.0, 65.0, 6.0)
    assert (
        controller.act(POV, telemetry(), raycast=hit)
        == controller.config.forward_action
    )


def test_v9_2_rgb_trunk_cannot_start_unconfirmed_out_of_range_attack():
    adapter = ScriptedTrunkAdapter()
    adapter.set_views(trunk_view(center_x=0.5, center_y=0.5, area=300.0))
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2
    )
    controller = TrunkContactController(adapter, config)
    controller.start()
    controller.engage(
        telemetry=telemetry(), candidate_id=12, global_step=10
    )
    controller.state = TrunkContactState.ADJUST_PITCH
    far_log = RaycastHit(True, True, False, False, 8.0, 29.0, 65.0, 0.0)
    action = controller.act(POV, telemetry(), raycast=far_log)
    assert action == controller.config.noop_action
    assert action != controller.config.attack_action
    assert controller.state == TrunkContactState.CENTER_TRUNK
    assert controller.counters.attack_steps == 0
    assert controller.counters.prevented_unconfirmed_attacks == 1


def test_v9_2_attack_state_stops_before_emitting_unconfirmed_attack():
    adapter = ScriptedTrunkAdapter()
    adapter.set_views(trunk_view(center_x=0.5, center_y=0.5, area=300.0))
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2
    )
    controller = TrunkContactController(adapter, config)
    controller.start()
    controller.engage(
        telemetry=telemetry(), candidate_id=13, global_step=10
    )
    controller.state = TrunkContactState.ATTACK_TRUNK
    empty = RaycastHit(False, False, False, False, 0.0, 0.0, 0.0, 0.0)
    action = controller.act(POV, telemetry(), raycast=empty)
    assert action == controller.config.noop_action
    assert controller.state == TrunkContactState.CENTER_TRUNK
    assert controller.counters.attack_steps == 0
    assert controller.counters.attack_out_of_range_loops == 1


def test_privileged_raycast_leaf_uses_bounded_clear_action():
    adapter = ScriptedTrunkAdapter()
    controller = make_controller(adapter)
    controller.state = TrunkContactState.FIND_TRUNK
    hit = RaycastHit(True, False, True, True, 2.0, 1.0, 65.0, 1.0)
    assert (
        controller.act(POV, telemetry(), raycast=hit)
        == controller.config.clear_occlusion_action
    )
    assert controller.counters.raycast_leaf_actions == 1


def test_raycast_log_disappearance_after_sustained_attack_probes_drop():
    adapter = ScriptedTrunkAdapter()
    controller = make_controller(adapter, attack_burst_steps=24)
    controller.state = TrunkContactState.ATTACK_TRUNK
    log_hit = RaycastHit(True, True, False, True, 2.0, 1.0, 65.0, 1.0)
    empty_hit = RaycastHit(False, False, False, False, 0.0, 0.0, 0.0, 0.0)

    for _ in range(5):
        assert (
            controller.act(POV, telemetry(), raycast=log_hit)
            == controller.config.attack_action
        )
    action = controller.act(POV, telemetry(), raycast=empty_hit)

    assert action == controller.config.forward_action
    assert controller.state == TrunkContactState.COLLECT_DROP
    reasons = [row[3] for row in controller.counters.transitions]
    assert "raycast log disappeared after sustained attack" in reasons


def test_drop_profile_searches_last_raycast_coordinate_then_collects_reward():
    adapter = ScriptedTrunkAdapter()
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_DROP_RECOVERY,
        attack_burst_steps=24,
        drop_recovery_max_steps=8,
    )
    controller = TrunkContactController(adapter, config)
    controller.engage()
    controller.state = TrunkContactState.ATTACK_TRUNK
    log_hit = RaycastHit(True, True, False, True, 2.0, 0.0, 64.0, 2.0)
    empty_hit = RaycastHit(False, False, False, False, 50.0, 0.0, 0.0, 0.0)

    for _ in range(5):
        controller.act(POV, telemetry(), raycast=log_hit)
    action = controller.act(POV, telemetry(), raycast=empty_hit)

    assert action == controller.config.forward_action
    assert controller.state == TrunkContactState.DROP_RECOVERY
    assert controller.counters.block_disappearances == 1
    assert controller.diagnostics()["last_contact_z"] == 2.0
    controller.observe(action, 1.0, False, {})
    assert controller.result == "success"
    assert controller.counters.pickup_after_disappearance == 1


def test_exhausted_drop_search_reacquires_same_trunk_without_backoff():
    adapter = ScriptedTrunkAdapter()
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_DROP_RECOVERY,
        drop_recovery_max_steps=1,
        max_attack_rounds=2,
        region_max_steps=20,
    )
    controller = TrunkContactController(adapter, config)
    controller.engage()
    controller._last_contact_point = (0.0, 64.0, 2.0)
    controller._last_contact_yaw = 0.0
    controller._start_drop_recovery(
        "test disappearance", telemetry(), block_disappeared=True
    )

    controller.act(POV, telemetry())
    controller.act(POV, telemetry(z=0.2))

    assert controller.state in (
        TrunkContactState.REACQUIRE_SAME_TRUNK,
        TrunkContactState.FIND_TRUNK,
    )
    assert controller.counters.same_trunk_reacquires == 1
    assert controller.counters.backoffs == 0
    assert any(
        row[2] == TrunkContactState.REACQUIRE_SAME_TRUNK.value
        for row in controller.counters.transitions
    )


def test_drop_recovery_freezes_broken_block_coordinate():
    adapter = ScriptedTrunkAdapter()
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_DROP_RECOVERY,
        drop_recovery_max_steps=4,
        region_max_steps=20,
    )
    controller = TrunkContactController(adapter, config)
    controller.engage()
    controller._last_contact_point = (1.0, 64.0, 2.0)
    controller._start_drop_recovery(
        "test disappearance", telemetry(), block_disappeared=True
    )
    adjacent_log = RaycastHit(
        True, True, False, False, 6.0, 5.0, 64.0, 6.0
    )

    controller.act(POV, telemetry(), raycast=adjacent_log)

    assert controller.diagnostics()["last_contact_x"] == 1.0
    assert controller.diagnostics()["last_contact_z"] == 2.0


def test_v9_1_drop_recovery_keeps_broken_coordinate_and_is_bounded():
    adapter = ScriptedTrunkAdapter()
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_COORDINATE_RECOVERY_V9_1,
        drop_recovery_max_steps=6,
        drop_recovery_waypoint_max_steps=3,
        region_max_steps=20,
    )
    controller = TrunkContactController(adapter, config)
    controller.engage()
    controller._last_contact_point = (1.0, 64.0, 2.0)
    controller._start_drop_recovery(
        "v9.1 disappearance", telemetry(), block_disappeared=True
    )
    adjacent_log = RaycastHit(
        True, True, False, False, 6.0, 9.0, 70.0, 9.0
    )

    steps = 0
    while controller.state == TrunkContactState.DROP_RECOVERY and steps < 20:
        controller.act(POV, telemetry(), raycast=adjacent_log)
        steps += 1

    diagnostics = controller.diagnostics()
    assert diagnostics["last_contact_x"] == 1.0
    assert diagnostics["last_contact_z"] == 2.0
    assert steps <= config.drop_recovery_max_steps + 1
    assert controller.state != TrunkContactState.DROP_RECOVERY


def test_v9_4_drop_recovery_targets_block_centre_not_hit_face():
    adapter = ScriptedTrunkAdapter()
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
        drop_recovery_max_steps=6,
    )
    controller = TrunkContactController(adapter, config)
    controller.engage()
    controller._last_contact_point = (-218.0, 65.413, 213.3446)
    player = telemetry(x=-221.57, z=215.41)

    controller._start_drop_recovery(
        "v9.4 disappearance", player, block_disappeared=True
    )

    diagnostics = controller.diagnostics()
    assert diagnostics["last_contact_x"] == -218.0
    assert diagnostics["drop_target_x"] == -217.5
    assert diagnostics["drop_target_y"] == 65.5
    assert diagnostics["drop_target_z"] == 213.5
    assert diagnostics["drop_recovery"]["target_x"] == -217.5
    assert controller.counters.drop_block_center_normalizations == 1


def test_v9_3_drop_recovery_keeps_hit_face_target_frozen():
    adapter = ScriptedTrunkAdapter()
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
        drop_recovery_max_steps=6,
    )
    controller = TrunkContactController(adapter, config)
    controller.engage()
    controller._last_contact_point = (-218.0, 65.413, 213.3446)

    controller._start_drop_recovery(
        "v9.3 disappearance",
        telemetry(x=-221.57, z=215.41),
        block_disappeared=True,
    )

    diagnostics = controller.diagnostics()
    assert diagnostics["drop_target_x"] == -218.0
    assert diagnostics["drop_target_z"] == 213.345
    assert controller.counters.drop_block_center_normalizations == 0


def test_v9_1_drop_search_has_bounded_budget_independent_of_contact_timeout():
    adapter = ScriptedTrunkAdapter()
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_COORDINATE_RECOVERY_V9_1,
        region_max_steps=1,
        drop_recovery_max_steps=4,
        drop_recovery_contact_extension_steps=4,
    )
    controller = TrunkContactController(adapter, config)
    controller.engage()
    controller._last_contact_point = (0.0, 64.0, 3.0)
    controller._start_drop_recovery(
        "bounded independent drop test", telemetry(), block_disappeared=True
    )

    controller.act(POV, telemetry())
    controller.act(POV, telemetry())

    assert controller.result is None
    assert controller.state == TrunkContactState.DROP_RECOVERY
    assert controller.diagnostics()["drop_timeout_extension"] == 4


def test_second_failed_drop_search_replans_candidate():
    adapter = ScriptedTrunkAdapter()
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_DROP_RECOVERY,
        drop_recovery_max_steps=1,
        max_attack_rounds=2,
        region_max_steps=20,
    )
    controller = TrunkContactController(adapter, config)
    controller.engage()
    controller._failed_attack_rounds = 1
    controller._last_contact_point = (0.0, 64.0, 2.0)
    controller._start_drop_recovery(
        "second test disappearance", telemetry(), block_disappeared=True
    )

    controller.act(POV, telemetry())
    controller.act(POV, telemetry(z=0.2))

    assert controller.result == "replan"
    assert controller.state == TrunkContactState.REPLAN
    assert controller.counters.backoffs == 0


def test_raycast_policy_profile_passes_only_declared_hit_to_contact():
    adapter = ScriptedTrunkAdapter()
    policy = CandidateSearchPolicy(
        adapter,
        SearchConfig(sensor_profile="f3_raycast"),
    )
    candidate, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 0.0, 0.9, 30.0),
        observer_yaw=0.0,
        step=0,
        telemetry=telemetry(),
        range_estimate=VisualRangeEstimate(1.5, 1.0),
    )
    candidate.status = "selected"
    policy.selected_candidate = candidate
    policy.state = SearchState.APPROACH
    obs = observation()
    obs["raycast"] = {
        "has_block": 1.0,
        "is_log": 1.0,
        "is_leaves": 0.0,
        "in_range": 1.0,
        "distance": 3.0,
        "x": 0.0,
        "y": 65.0,
        "z": 1.0,
    }
    assert policy.act(obs) == policy.config.contact_attack_action
    assert policy.contact_state == "ATTACK_TRUNK"


def test_trunk_horizontal_offset_produces_correct_yaw_action():
    adapter = ScriptedTrunkAdapter()
    adapter.set_views(trunk_view(center_x=0.62))
    controller = make_controller(adapter)
    controller.state = TrunkContactState.CENTER_TRUNK
    assert controller.act(POV, telemetry()) == controller.config.fine_right_action

    adapter.set_views(trunk_view(center_x=0.38))
    controller = make_controller(adapter)
    controller.state = TrunkContactState.CENTER_TRUNK
    assert controller.act(POV, telemetry()) == controller.config.fine_left_action


def test_trunk_vertical_offset_produces_correct_pitch_action():
    adapter = ScriptedTrunkAdapter()
    adapter.set_views(trunk_view(center_y=0.66))
    controller = make_controller(adapter)
    controller.state = TrunkContactState.ADJUST_PITCH
    assert controller.act(POV, telemetry(pitch=0.0)) == controller.config.fine_look_down_action

    adapter.set_views(trunk_view(center_y=0.34))
    controller = make_controller(adapter)
    controller.state = TrunkContactState.ADJUST_PITCH
    assert controller.act(POV, telemetry(pitch=0.0)) == controller.config.fine_look_up_action


def test_pitch_bounds_prevent_looking_past_limits():
    adapter = ScriptedTrunkAdapter()
    adapter.set_views(trunk_view(center_y=0.9))
    controller = make_controller(adapter)
    controller.state = TrunkContactState.ADJUST_PITCH
    # Already at the down limit with the trunk below: attack instead of
    # staring past the bound.
    assert controller.act(POV, telemetry(pitch=60.0)) == controller.config.attack_action


def test_pitch_direction_flip_brackets_and_attacks():
    # Regression for the 10-degree quantum limit cycle seen in the first
    # natural smoke run: error above the deadband at pitch 40, below at 50.
    adapter = ScriptedTrunkAdapter()
    adapter.set_views(trunk_view(center_y=0.66))
    controller = make_controller(adapter)
    controller.state = TrunkContactState.ADJUST_PITCH
    assert controller.act(POV, telemetry(pitch=40.0)) == controller.config.fine_look_down_action
    adapter.set_views(trunk_view(center_y=0.34))
    assert controller.act(POV, telemetry(pitch=50.0)) == controller.config.attack_action


def test_yaw_direction_flip_brackets_and_advances():
    adapter = ScriptedTrunkAdapter()
    adapter.set_views(trunk_view(center_x=0.62))
    controller = make_controller(adapter)
    controller.state = TrunkContactState.CENTER_TRUNK
    assert controller.act(POV, telemetry()) == controller.config.fine_right_action
    adapter.set_views(trunk_view(center_x=0.38))
    action = controller.act(POV, telemetry())
    # Direction flip means the trunk sits between two quantized headings;
    # the same act() call may continue into pitch alignment or attacking.
    assert controller.state in (
        TrunkContactState.CENTER_TRUNK,
        TrunkContactState.ADJUST_PITCH,
        TrunkContactState.ATTACK_TRUNK,
    )
    assert action in (
        controller.config.fine_look_up_action,
        controller.config.fine_look_down_action,
        controller.config.attack_action,
        controller.config.noop_action,
    )


def test_small_distant_trunk_keeps_closing_instead_of_attacking():
    adapter = ScriptedTrunkAdapter()
    adapter.set_views(trunk_view(area=75.0, width_px=3.0))
    controller = make_controller(adapter)
    controller.state = TrunkContactState.ADJUST_PITCH
    assert controller.act(POV, telemetry()) == controller.config.forward_action


def test_short_wide_dirt_patch_does_not_trigger_attack():
    adapter = ScriptedTrunkAdapter()
    adapter.set_views(
        trunk_view(area=200.0, width_px=10.0, height_px=9.0)
    )
    controller = make_controller(adapter)
    controller.state = TrunkContactState.ADJUST_PITCH
    assert controller.act(POV, telemetry()) == controller.config.forward_action


class MultiComponentAdapter(ScriptedTrunkAdapter):
    def __init__(self, sequences):
        super().__init__()
        self.sequences = sequences
        self.sequence_cursor = 0

    def trunk_views(self, pov: np.ndarray) -> List[TrunkView]:
        result = self.sequences[min(self.sequence_cursor, len(self.sequences) - 1)]
        self.sequence_cursor += 1
        return result


def test_contact_tracking_keeps_same_component_when_ranking_flips():
    tracked = trunk_view(center_x=0.42, material="oak")
    nearby = trunk_view(center_x=0.44, material="oak")
    distractor = trunk_view(center_x=0.78, area=500.0, material="oak")
    adapter = MultiComponentAdapter([[tracked], [distractor, nearby]])
    controller = make_controller(adapter)
    controller.state = TrunkContactState.CENTER_TRUNK
    controller.act(POV, telemetry())
    controller.act(POV, telemetry())
    assert controller._last_view is nearby


class AlternatingTrunkAdapter(ScriptedTrunkAdapter):
    """Alternate between two trunk views every frame."""

    def __init__(self, view_a: TrunkView, view_b: TrunkView):
        super().__init__()
        self.view_a = view_a
        self.view_b = view_b
        self.toggle = False

    def trunk_view(self, pov: np.ndarray) -> TrunkView:
        self.toggle = not self.toggle
        return self.view_a if self.toggle else self.view_b


def test_center_adjust_limit_cycle_escapes_via_orbit():
    # Regression for seed 16002: two brown components ~10 degrees apart
    # made centring and pitch adjustment ping-pong forever.
    adapter = AlternatingTrunkAdapter(
        trunk_view(center_x=0.66), trunk_view(center_x=0.34)
    )
    controller = make_controller(adapter)
    controller.state = TrunkContactState.CENTER_TRUNK
    reasons = []
    for _ in range(60):
        if controller.result is not None:
            break
        controller.act(POV, telemetry())
        reasons.extend(
            reason
            for _step, _old, _new, reason in controller.counters.transitions
        )
    assert "yaw centring cycle; orbiting tree" in reasons
    assert controller.counters.orbits >= 1


def test_attack_without_progress_backs_off_orbits_then_replans():
    adapter = ScriptedTrunkAdapter()
    adapter.set_views(trunk_view(crosshair=0.4))
    controller = make_controller(adapter, region_max_steps=600)
    actions = []
    for _ in range(400):
        if controller.result is not None:
            break
        action = controller.act(POV, telemetry())
        controller.observe(action, 0.0, False, {})
        actions.append(action)
    assert controller.result == "replan"
    assert controller.counters.backoffs >= 1
    assert controller.counters.orbits >= 1
    assert controller.counters.attack_steps >= 16
    assert controller.config.backward_action in actions
    assert len(actions) < 400  # terminates well inside the safety loop


def test_contact_attack_is_pure_attack_and_pickup_probe_is_forward():
    adapter = ScriptedTrunkAdapter()
    adapter.set_views(trunk_view(crosshair=0.4))
    controller = make_controller(adapter, attack_burst_steps=2)
    controller.state = TrunkContactState.ATTACK_TRUNK
    assert controller.config.attack_action == 7
    assert controller.act(POV, telemetry()) == 7
    assert controller.act(POV, telemetry()) == 7
    assert controller.state == TrunkContactState.COLLECT_DROP
    assert controller.act(POV, telemetry()) == controller.config.forward_action


def test_contact_metrics_accumulate_across_candidate_attempts():
    adapter = ScriptedTrunkAdapter()
    controller = TrunkContactController(
        adapter, TrunkContactConfig(region_max_steps=1)
    )
    controller.engage(candidate_id=11, global_step=20)
    controller.act(POV, telemetry(), global_step=20)
    controller.act(POV, telemetry(), global_step=21)
    assert controller.result == "replan"
    controller.engage(candidate_id=12, global_step=30)
    controller.act(POV, telemetry(), global_step=30)
    diagnostics = controller.diagnostics()
    assert diagnostics["counters"]["attempts"] == 2
    assert diagnostics["counters"]["steps"] == 3
    assert {row["candidate_id"] for row in diagnostics["transition_records"]} == {
        11,
        12,
    }


def test_oak_and_birch_trunks_are_detected_but_wide_dirt_is_rejected():
    adapter = TreeResourceAdapter()
    oak = POV.copy()
    oak[8:60, 29:35] = (125, 72, 38)
    assert adapter.trunk_view(oak).present

    birch = POV.copy()
    birch[6:60, 28:36] = (190, 184, 170)
    for row in range(10, 58, 8):
        birch[row:row + 2, 28:36] = (55, 52, 48)
    assert adapter.trunk_view(birch).present

    dirt = POV.copy()
    dirt[32:64, :] = (120, 68, 35)
    assert adapter.detect(dirt) == []


def test_leaf_occlusion_cue_ignores_lower_half_grass():
    adapter = TreeResourceAdapter()
    green = (30, 90, 25)
    grass_only = np.full_like(POV, (100, 150, 220))
    grass_only[32:64, :] = green
    canopy = np.full_like(POV, (100, 150, 220))
    canopy[6:32, 18:46] = green

    assert adapter.leaf_occlusion_fraction(grass_only) == 0.0
    assert adapter.leaf_occlusion_fraction(canopy) > 0.9


def test_canopy_support_requires_vertical_contact_with_trunk_top():
    adapter = TreeResourceAdapter()
    frame = POV.copy()
    # One broad skyline canopy overlaps both components in x.  Only the left
    # component reaches it; the lower-right brown edge represents a dirt bank.
    frame[0:10, :] = (30, 90, 25)
    frame[8:45, 10:15] = (125, 72, 38)
    frame[42:64, 48:54] = (125, 72, 38)

    detections = adapter.detect(frame)

    assert len(detections) == 2
    real_trunk = min(detections, key=lambda item: item.center_x)
    dirt_edge = max(detections, key=lambda item: item.center_x)
    assert real_trunk.apparent_size > real_trunk.geometry_size
    assert dirt_edge.apparent_size == dirt_edge.geometry_size
    assert detections[0] is real_trunk


def test_attempts_exhausted_returns_global_replan():
    adapter = ScriptedTrunkAdapter()  # never sees a trunk
    policy = CandidateSearchPolicy(
        adapter, SearchConfig(sensor_profile="f3_telemetry")
    )
    candidate, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 0.0, 0.9, 30.0),
        observer_yaw=0.0,
        step=0,
        telemetry=telemetry(),
        range_estimate=VisualRangeEstimate(1.5, 1.0),
    )
    candidate.status = "selected"
    policy.selected_candidate = candidate
    policy.state = SearchState.APPROACH
    obs = observation()
    for _ in range(200):
        if any(
            row["reason"] == "trunk contact attempts exhausted"
            for row in policy.transition_log
        ):
            break
        action = policy.act(obs)
        policy.observe_transition(action, obs, 0.0, False, {})
    assert any(
        row["reason"] == "trunk contact attempts exhausted"
        for row in policy.transition_log
    )
    assert policy.replan_count == 1
    assert policy.selected_candidate is None


def test_log_reward_switches_policy_and_controller_to_success():
    adapter = ScriptedTrunkAdapter()
    adapter.set_views(trunk_view(crosshair=0.4))
    policy = CandidateSearchPolicy(
        adapter, SearchConfig(sensor_profile="f3_telemetry")
    )
    candidate, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 0.0, 0.9, 30.0),
        observer_yaw=0.0,
        step=0,
        telemetry=telemetry(),
        range_estimate=VisualRangeEstimate(1.5, 1.0),
    )
    candidate.status = "selected"
    policy.selected_candidate = candidate
    policy.state = SearchState.APPROACH
    obs = observation()
    action = policy.act(obs)
    assert policy._contact is not None and policy._contact.engaged
    policy.observe_transition(action, obs, 1.0, True, {"success": True})
    assert policy.state == SearchState.SUCCESS
    assert policy._contact.result == "success"


def test_controller_state_machine_is_finite_without_trunk():
    adapter = ScriptedTrunkAdapter()
    controller = make_controller(adapter)
    steps = 0
    while controller.result is None and steps < 500:
        action = controller.act(POV, telemetry())
        controller.observe(action, 0.0, False, {})
        steps += 1
    assert controller.result == "replan"
    assert steps <= controller.config.region_max_steps + 1


class F3GuardedObservation(dict):
    def __getitem__(self, key):
        if key not in ("pov", "telemetry"):
            raise AssertionError("contact path accessed privileged field {}".format(key))
        return super().__getitem__(key)

    def __iter__(self):
        raise AssertionError("contact path iterated privileged observation fields")


def test_contact_action_path_does_not_access_oracle_or_grid():
    adapter = ScriptedTrunkAdapter()
    policy = CandidateSearchPolicy(
        adapter, SearchConfig(sensor_profile="f3_telemetry")
    )
    candidate, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 0.0, 0.9, 30.0),
        observer_yaw=0.0,
        step=0,
        telemetry=telemetry(),
        range_estimate=VisualRangeEstimate(4.0, 1.0),
    )
    candidate.status = "selected"
    policy.selected_candidate = candidate
    policy.state = SearchState.APPROACH
    allowed = observation()
    guarded = F3GuardedObservation(
        pov=allowed["pov"],
        telemetry=allowed["telemetry"],
        oracle="forbidden",
        log_grid="forbidden",
    )
    for _ in range(20):
        action = policy.act(guarded)
        policy.observe_transition(action, guarded, 0.0, False, {})
    assert policy._contact is not None and policy._contact.engaged
