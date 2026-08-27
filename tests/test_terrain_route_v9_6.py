"""v9.6 terrain-route, reachable-contact, and drop-completion regression tests.

Every test here is derived from a specific v9.5 gate-v2 failure trace:
16702 (coarse terrain routing), 16714 (reachable local contact), and
16716/16719 (post-disappearance pickup), plus the frozen-boundary checks
that keep v9.5 and older profiles byte-identical in behaviour.
"""

import inspect
from dataclasses import fields as dataclass_fields
from typing import Any, Dict, List

import numpy as np

from mc_rl.candidates import ResourceDetection
from mc_rl.drop_recovery import DropRecoveryConfig, DropRecoveryPlanner
from mc_rl.resource_adapters import ResourceAdapter, TrunkView
from mc_rl.search_policy import CandidateSearchPolicy, SearchConfig, SearchState
from mc_rl.telemetry import AgentTelemetry, RaycastHit, VisualRangeEstimate
from mc_rl.trunk_contact import (
    CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
    CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
    TrunkContactConfig,
    TrunkContactController,
    TrunkContactState,
)


POV = np.zeros((64, 64, 3), dtype=np.uint8)

V9_6_ONLY_FIELDS = frozenset(
    (
        "reset_loop_budgets_on_rescan_success",
        "enable_coordinate_climb_assist",
        "drop_recovery_elevated_pickup",
    )
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


def telemetry(x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0):
    return AgentTelemetry(x=x, y=y, z=z, yaw=yaw, pitch=pitch)


def hit(x, y, z, is_log=True, is_leaves=False, in_range=False):
    return RaycastHit(
        has_block=True,
        is_log=is_log,
        is_leaves=is_leaves,
        in_range=in_range,
        distance=2.0,
        x=x,
        y=y,
        z=z,
    )


def v9_6_controller(**overrides):
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6, **overrides
    )
    controller = TrunkContactController(HandoffAdapter(), config)
    controller.start()
    return controller


def v9_5_controller(**overrides):
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5, **overrides
    )
    controller = TrunkContactController(HandoffAdapter(), config)
    controller.start()
    return controller


def v9_6_policy(**overrides):
    return CandidateSearchPolicy(
        HandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
            **overrides
        ),
    )


def v9_5_policy():
    return CandidateSearchPolicy(
        HandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
        ),
    )


def _selected_world_candidate(policy, x=0.0, z=10.0, uncertainty=1.0):
    tel = policy._telemetry(telemetry_observation())
    candidate, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 0.0, 0.9, max(z, 1.0)),
        0.0,
        0,
        telemetry=tel,
        range_estimate=VisualRangeEstimate(max(z, 1.0), uncertainty),
    )
    candidate.estimated_world_x = x
    candidate.estimated_world_z = z
    candidate.position_uncertainty = uncertainty
    candidate.status = "selected"
    policy.selected_candidate = candidate
    policy.current_telemetry = tel
    policy.state = SearchState.APPROACH
    return candidate, tel


def _drive(policy, positions, yaw=0.0):
    actions = []
    for x, z in positions:
        obs = telemetry_observation(x=x, z=z, yaw=yaw)
        action = policy.act(obs)
        policy.observe_transition(action, obs, 0.0, False, {})
        actions.append(action)
    return actions


# ---------------------------------------------------------------------------
# 1/2/12. Frozen boundaries and inheritance.
# ---------------------------------------------------------------------------


def test_v9_5_profile_defaults_remain_frozen():
    v9_5 = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5
    )
    assert v9_5.reset_loop_budgets_on_rescan_success is False
    assert v9_5.enable_coordinate_climb_assist is False
    assert v9_5.drop_recovery_elevated_pickup is False
    assert v9_5.enable_spatial_exact_log_rescan_cooldown is True
    assert v9_5.exact_log_rescan_spatial_radius == 8.0
    assert v9_5.exact_log_rescan_max_attempts_per_candidate == 1
    assert v9_5.exact_log_rescan_budget == 40
    assert v9_5.require_raycast_attack_confirmation is True
    assert v9_5.drop_recovery_max_steps == 72
    assert v9_5.drop_recovery_normalize_block_centre is True
    assert v9_5.drop_recovery_ordered_ring is True
    assert v9_5.coordinate_maximum_horizontal_selection_distance == 14.0
    assert v9_5.coordinate_recovery_jump_steps == 8
    assert v9_5.drop_recovery_waypoint_max_steps == 10
    assert v9_5.drop_recovery_centre_max_steps == 28
    # The search-side mechanism also defaults to disabled for every profile.
    assert SearchConfig().enable_terrain_route_recovery is False


def test_v9_6_inherits_v9_5_ownership_visual_boundary_and_spatial_cooldown():
    v9_6 = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6
    )
    v9_5 = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5
    )
    for field in dataclass_fields(TrunkContactConfig):
        if field.name in V9_6_ONLY_FIELDS:
            continue
        assert getattr(v9_6, field.name) == getattr(v9_5, field.name), field.name
    assert v9_6.reset_loop_budgets_on_rescan_success is True
    assert v9_6.enable_coordinate_climb_assist is True
    assert v9_6.drop_recovery_elevated_pickup is True

    policy = v9_6_policy()
    assert policy._contact_ownership_guard
    assert policy._candidate_handoff_guard
    assert policy._terrain_route_recovery
    assert policy.config.handoff_raycast_max_distance == 14.0
    frozen = v9_5_policy()
    assert frozen._contact_ownership_guard
    assert not frozen._terrain_route_recovery


def test_v9_6_active_contact_keeps_ownership_across_global_replan_state():
    policy = v9_6_policy()
    candidate, tel = _selected_world_candidate(policy, z=3.0)
    policy._contact.engage(
        telemetry=tel,
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


def test_v9_6_contact_owner_mismatch_cancels_and_records():
    policy = v9_6_policy()
    candidate, tel = _selected_world_candidate(policy, z=3.0)
    policy._contact.engage(
        telemetry=tel,
        candidate_id=candidate.candidate_id + 1,
        global_step=10,
        target_hint=(0.0, 3.0),
    )
    policy.state = SearchState.REPLAN

    policy.act(telemetry_observation())

    assert policy.contact_owner_mismatches == 1
    assert not policy._contact.active
    cancel_reasons = [
        row["reason"] for row in policy._contact.diagnostics()["attempt_results"]
    ]
    assert "contact owner mismatch between local and global policy" in cancel_reasons


# ---------------------------------------------------------------------------
# 5/6/7. Terrain route recovery (16702).
# ---------------------------------------------------------------------------


def test_v9_6_real_route_progress_is_not_treated_as_stall():
    policy = v9_6_policy()
    _selected_world_candidate(policy, z=10.0, uncertainty=1.0)
    positions = [(0.0, 0.2 * index) for index in range(1, 25)]

    actions = _drive(policy, positions)

    assert policy.state == SearchState.APPROACH
    assert set(actions) == {policy.config.forward_action}
    assert policy.terrain_route_recovery_attempts == 0
    assert policy.terrain_route_recovery_failures == 0
    assert policy.obstacle_recovery_count == 0
    assert policy.replan_count == 0


def test_v9_6_no_displacement_replans_within_bounded_steps():
    policy = v9_6_policy()
    _selected_world_candidate(policy, z=10.0, uncertainty=1.0)

    actions = _drive(policy, [(0.0, 0.0)] * 80)

    reasons = [row["reason"] for row in policy.transition_log]
    assert policy.state != SearchState.APPROACH or policy.replan_count > 0
    assert any(
        reason in reasons
        for reason in (
            "terrain route climb recovery failed without progress",
            "world route blocked outside contact region",
        )
    )
    # Bounded recovery, not an infinite obstacle loop.
    assert policy.terrain_route_recovery_attempts <= (
        policy.config.terrain_route_recovery_max_attempts
    )
    assert policy.obstacle_recovery_count <= (
        policy.config.max_route_obstacle_recoveries
    )
    assert len(actions) == 80


def test_v9_6_climb_recovery_is_verified_and_recorded():
    policy = v9_6_policy(
        terrain_route_progress_window=4,
        terrain_route_progress_minimum=0.5,
        terrain_route_climb_steps=3,
    )
    _selected_world_candidate(policy, z=10.0, uncertainty=1.0)

    # Four forwards with no progress arm the first burst; the arming call
    # itself consumes the first jump.
    arming = _drive(policy, [(0.0, 0.0)] * 4)
    assert policy.terrain_route_recovery_attempts == 1
    assert arming[-1] == policy.config.forward_jump_action
    burst = _drive(policy, [(0.0, 0.0)] * 2)
    assert set(burst) == {policy.config.forward_jump_action}

    # The burst must now be graded against the real route distance.
    _drive(policy, [(0.0, 0.0)])
    assert policy.terrain_route_recovery_failures == 1
    records = policy.terrain_route_recovery_records
    assert len(records) == 1
    assert records[0]["outcome"] == "failure"
    assert records[0]["route_distance_start"] == 10.0
    assert records[0]["route_distance_end"] == 10.0
    assert records[0]["horizontal_displacement"] == 0.0


def test_v9_6_repeated_physical_region_route_rejection():
    policy = v9_6_policy(
        terrain_route_progress_window=4,
        terrain_route_progress_minimum=0.5,
        terrain_route_climb_steps=2,
        terrain_route_recovery_failure_limit=2,
    )
    _selected_world_candidate(policy, z=10.0, uncertainty=1.0)

    _drive(policy, [(0.0, 0.0)] * 40)
    assert policy._route_blocked_regions == [(0.0, 10.0)]

    # A fresh available candidate inside the blocked region must be cooled
    # down at selection time instead of replaying the same blocked walk.
    tel = policy._telemetry(telemetry_observation())
    candidate, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 0.0, 0.9, 10.0),
        0.0,
        0,
        telemetry=tel,
        range_estimate=VisualRangeEstimate(10.0, 1.0),
    )
    candidate.estimated_world_x = 0.5
    candidate.estimated_world_z = 10.5
    candidate.position_uncertainty = 1.0
    candidate.status = "available"
    policy.selected_candidate = None
    policy.state = SearchState.SELECT_TARGET

    policy.act(telemetry_observation())

    assert policy.repeated_physical_region_route_rejections == 1
    assert candidate.status == "cooldown"
    assert candidate.failed_until_step > 0


# ---------------------------------------------------------------------------
# 8/9. Reachable local contact (16714) and the 16716 rescan defect.
# ---------------------------------------------------------------------------


def test_v9_6_rescan_success_resets_stale_loop_budgets():
    controller = v9_6_controller(exact_log_rescan_budget=3)
    pose = telemetry()
    controller.engage(telemetry=pose, candidate_id=1, global_step=10)
    assert controller.state == TrunkContactState.EXACT_LOG_RESCAN
    # 16716 precondition: centring loops accumulated before the raster.
    controller._attempt_center_adjust_loops = (
        controller.config.exact_log_rescan_loop_budget
    )
    for _ in range(2):
        controller.act(POV, pose, raycast=None, global_step=10)

    controller.observe_raycast_target(
        hit(0.0, 65.62, 5.0, in_range=True), pose, 12
    )

    assert controller.state == TrunkContactState.COORDINATE_AIM
    assert controller.counters.rescan_success_loop_resets == 1
    action = controller.act(
        POV, pose, raycast=hit(0.0, 65.62, 5.0, in_range=True), global_step=13
    )
    assert controller.state == TrunkContactState.ATTACK_TRUNK
    assert controller.result is None
    assert action == controller.config.attack_action


def test_v9_5_keeps_the_stale_loop_replan_after_rescan_success():
    controller = v9_5_controller(exact_log_rescan_budget=3)
    pose = telemetry()
    controller.engage(telemetry=pose, candidate_id=1, global_step=10)
    controller._attempt_center_adjust_loops = (
        controller.config.exact_log_rescan_loop_budget
    )
    for _ in range(2):
        controller.act(POV, pose, raycast=None, global_step=10)

    controller.observe_raycast_target(
        hit(0.0, 65.62, 5.0, in_range=True), pose, 12
    )

    assert controller.state == TrunkContactState.COORDINATE_AIM
    assert controller.counters.rescan_success_loop_resets == 0
    controller.act(POV, pose, raycast=None, global_step=13)
    assert controller.result == "replan"
    reasons = [row["reason"] for row in controller.diagnostics()["attempt_results"]]
    assert "exact log rescan already used for candidate" in reasons


def test_v9_6_coordinate_climb_assist_arms_and_disables_after_failures():
    controller = v9_6_controller(
        coordinate_climb_window=3,
        coordinate_climb_minimum_progress=0.5,
        coordinate_climb_burst_steps=2,
        coordinate_climb_failure_limit=2,
        coordinate_climb_success_progress=0.4,
    )
    pose = telemetry()
    controller.observe_raycast_target(hit(0.0, 65.62, 8.0), pose, 1)
    controller.observe_raycast_target(hit(0.0, 65.62, 12.0), pose, 2)
    controller.engage(telemetry=pose, global_step=2, target_hint=(0.0, 8.0))
    assert controller.state == TrunkContactState.COORDINATE_AIM

    # Slow 0.05-blocks-per-step progress: below the climb minimum, so two
    # verified bursts fire and both fail, disabling the assist for this
    # target before the frozen-pose stall path is exercised.
    actions = []
    for index in range(1, 11):
        actions.append(
            controller.act(POV, telemetry(z=0.05 * index), raycast=None)
        )
    assert controller.counters.coordinate_climb_bursts == 2
    assert controller.config.forward_jump_action in actions
    assert controller.counters.coordinate_climb_failures == 2
    assert controller._coordinate_climb_disabled

    # Freeze the pose: the ordinary stall/recovery/switch path must take
    # over and hand the target off (switch or replan) in bounded steps.
    for _ in range(120):
        if controller.result is not None or (
            controller.counters.coordinate_target_switches > 0
        ):
            break
        controller.act(POV, pose, raycast=None)
    assert (
        controller.counters.coordinate_target_switches > 0
        or controller.result == "replan"
    )


def test_v9_6_strict_raycast_attack_confirmation_is_not_bypassed():
    controller = v9_6_controller(
        coordinate_climb_window=3,
        coordinate_climb_minimum_progress=0.5,
        coordinate_climb_burst_steps=2,
    )
    pose = telemetry()
    controller.observe_raycast_target(hit(0.0, 65.62, 8.0), pose, 1)
    controller.engage(telemetry=pose, global_step=2, target_hint=(0.0, 8.0))

    for index in range(1, 11):
        action = controller.act(
            POV, telemetry(z=0.05 * index), raycast=None
        )
        assert action != controller.config.attack_action
        assert controller.state != TrunkContactState.ATTACK_TRUNK
    assert controller.config.require_raycast_attack_confirmation is True


# ---------------------------------------------------------------------------
# 10/11. Post-disappearance pickup (16716/16719).
# ---------------------------------------------------------------------------


def test_v9_6_elevated_drop_uses_bounded_jump_pickup():
    config = DropRecoveryConfig(
        enable_obstacle_recovery=True,
        ordered_ring=True,
        elevated_vertical_gap=1.2,
        elevated_arrival_radius=1.8,
        elevated_jump_steps=3,
    )
    planner = DropRecoveryPlanner(0.0, 0.0, config, centre_y=67.5)
    pose = telemetry(x=0.0, y=64.0, z=0.0)

    actions = [planner.next_action(pose, 1, 3, 4, 0, 2) for _ in range(3)]
    assert actions == [2, 2, 2]
    assert planner.elevated_jump_steps == 3

    # The jump budget ends and the waypoint is skipped, not retried forever.
    action = planner.next_action(pose, 1, 3, 4, 0, 2)
    assert planner.index == 1
    assert planner.blocked_waypoints == 1
    records = planner.waypoint_records
    assert records[0]["end_reason"] == "elevated_jump_budget"
    assert records[0]["vertical_gap"] == 3.5
    assert records[0]["elevated_jumps"] == 3


def test_v9_6_elevated_drop_navigates_before_jumping():
    config = DropRecoveryConfig(
        enable_obstacle_recovery=True,
        ordered_ring=True,
        elevated_vertical_gap=1.2,
        elevated_arrival_radius=1.8,
        elevated_jump_steps=3,
    )
    planner = DropRecoveryPlanner(0.0, 0.0, config, centre_y=67.5)
    # Outside the elevated arrival radius: ordinary steering, no jumps yet.
    action = planner.next_action(telemetry(x=0.0, y=64.0, z=3.0), 1, 3, 4, 0, 2)
    assert action in (3, 4)
    assert planner.elevated_jump_steps == 0


def test_frozen_profiles_keep_two_dimensional_drop_behaviour():
    config = DropRecoveryConfig(
        enable_obstacle_recovery=True,
        ordered_ring=True,
    )
    planner = DropRecoveryPlanner(0.0, 0.0, config)
    pose = telemetry(x=0.0, y=64.0, z=0.0)

    planner.next_action(pose, 1, 3, 4, 0, 2)

    assert planner.index == 1
    assert planner.reached_waypoints == 1
    assert planner.elevated_jump_steps == 0
    v9_5 = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5
    )
    assert v9_5.drop_recovery_elevated_pickup is False
    elevated = DropRecoveryConfig(
        elevated_vertical_gap=1.2,
        elevated_arrival_radius=1.8,
        elevated_jump_steps=8,
    )
    assert elevated.elevated_vertical_gap == 1.2


def test_v9_6_drop_recovery_reads_no_item_entity_oracle_or_log_grid():
    planner_params = set(
        inspect.signature(DropRecoveryPlanner.__init__).parameters
    )
    assert planner_params == {"self", "centre_x", "centre_z", "config", "centre_y"}
    action_params = set(
        inspect.signature(DropRecoveryPlanner.next_action).parameters
    )
    assert action_params == {
        "self",
        "telemetry",
        "forward_action",
        "left_action",
        "right_action",
        "noop_action",
        "forward_jump_action",
    }
    # The only spatial input is the player's own F3-style pose plus the
    # internally remembered contact point; no observation dict is accepted.
    controller = v9_6_controller()
    drop_config = controller._drop_recovery_config()
    assert drop_config.elevated_vertical_gap == 1.2
    frozen_controller = v9_5_controller()
    assert frozen_controller._drop_recovery_config().elevated_vertical_gap is None
