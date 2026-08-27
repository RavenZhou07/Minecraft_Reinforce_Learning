"""Fast tests for privileged 3-D trunk target memory and servo geometry."""

import numpy as np

from mc_rl.coordinate_aim import (
    CoordinateProgressMonitor,
    TARGET_COOLDOWN,
    TrunkBlockTarget,
    TrunkTargetScoreConfig,
    TrunkTargetMemory,
    coordinate_aim_error,
)
from mc_rl.resource_adapters import TreeResourceAdapter
from mc_rl.telemetry import AgentTelemetry, RaycastHit
from mc_rl.trunk_contact import (
    CONTACT_PROFILE_COORDINATE_AIM,
    CONTACT_PROFILE_COORDINATE_RECOVERY,
    CONTACT_PROFILE_COORDINATE_RECOVERY_V9_1,
    CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2,
    CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
    CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
    CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
    TrunkContactConfig,
    TrunkContactController,
    TrunkContactState,
)


POV = np.zeros((64, 64, 3), dtype=np.uint8)


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


def coordinate_controller(**overrides):
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_COORDINATE_AIM, **overrides
    )
    controller = TrunkContactController(TreeResourceAdapter(), config)
    controller.start()
    return controller


def recovery_controller(**overrides):
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_COORDINATE_RECOVERY, **overrides
    )
    controller = TrunkContactController(TreeResourceAdapter(), config)
    controller.start()
    return controller


def v9_1_controller(**overrides):
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_COORDINATE_RECOVERY_V9_1, **overrides
    )
    controller = TrunkContactController(TreeResourceAdapter(), config)
    controller.start()
    return controller


def v9_2_controller(**overrides):
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2, **overrides
    )
    controller = TrunkContactController(TreeResourceAdapter(), config)
    controller.start()
    return controller


def v9_4_controller(**overrides):
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4, **overrides
    )
    controller = TrunkContactController(TreeResourceAdapter(), config)
    controller.start()
    return controller


def v9_5_controller(**overrides):
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5, **overrides
    )
    controller = TrunkContactController(TreeResourceAdapter(), config)
    controller.start()
    return controller


def test_coordinate_aim_uses_minecraft_yaw_and_pitch_signs():
    eye_y = 65.62
    forward = coordinate_aim_error(
        telemetry(), TrunkBlockTarget(1, 0.0, eye_y, 10.0, 0)
    )
    east = coordinate_aim_error(
        telemetry(), TrunkBlockTarget(2, 10.0, eye_y, 0.0, 0)
    )
    below = coordinate_aim_error(
        telemetry(), TrunkBlockTarget(3, 0.0, 64.0, 10.0, 0)
    )
    assert abs(forward.yaw_error) < 1e-6
    assert abs(forward.pitch_error) < 1e-6
    assert east.yaw_error == -90.0
    assert below.pitch_error > 0.0


def test_target_memory_merges_near_hits_but_keeps_different_blocks():
    memory = TrunkTargetMemory(merge_distance=0.7)
    first = memory.observe(hit(1.0, 65.0, 2.0), 1)
    merged = memory.observe(hit(1.2, 65.1, 2.1), 2)
    separate = memory.observe(hit(2.0, 65.0, 2.0), 3)
    assert merged is first
    assert first.observation_count == 2
    assert separate is not first
    assert len(memory.targets) == 2


def test_target_selection_prefers_candidate_world_hint_and_honours_cooldown():
    memory = TrunkTargetMemory()
    near_player = memory.observe(hit(0.0, 65.0, 2.0), 1)
    near_hint = memory.observe(hit(9.0, 65.0, 0.0), 2)
    selected = memory.select(3, telemetry(), candidate_hint=(10.0, 0.0))
    assert selected is near_hint
    memory.mark_failed(selected, 3, cooldown_steps=10)
    assert selected.status == TARGET_COOLDOWN
    assert memory.select(4, telemetry(), candidate_hint=(10.0, 0.0)) is near_player
    assert memory.select(14, telemetry(), candidate_hint=(10.0, 0.0)) is near_hint
    assert near_player.status == "available"


def test_reachability_score_prefers_lower_log_face_and_logs_terms():
    memory = TrunkTargetMemory()
    lower = memory.observe(hit(0.0, 65.62, 4.0), 1)
    upper = memory.observe(hit(0.0, 70.0, 4.0), 2)
    selected = memory.select(
        3,
        telemetry(),
        candidate_hint=(0.0, 4.0),
        score_config=TrunkTargetScoreConfig(),
    )
    assert selected is lower
    assert lower.score > upper.score
    assert upper.score_terms["vertical_distance_cost"] < lower.score_terms[
        "vertical_distance_cost"
    ]


def test_natural_one_log_score_prefers_near_low_reachable_log_over_hint():
    memory = TrunkTargetMemory()
    reachable = memory.observe(hit(0.0, 65.62, 3.0), 1)
    hinted_high = memory.observe(hit(15.0, 72.0, 0.0), 2)
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_COORDINATE_RECOVERY_V9_1
    )
    selected = memory.select(
        3,
        telemetry(),
        candidate_hint=(15.0, 0.0),
        score_config=TrunkTargetScoreConfig(
            hint_distance_weight=config.coordinate_hint_distance_weight,
            horizontal_distance_weight=config.coordinate_horizontal_distance_weight,
            vertical_distance_weight=config.coordinate_vertical_distance_weight,
            reach_excess_weight=config.coordinate_reach_excess_weight,
            failure_weight=config.coordinate_failure_weight,
            recovery_weight=config.coordinate_recovery_weight,
            observation_weight=config.coordinate_observation_weight,
            attack_reach=config.coordinate_attack_reach,
        ),
    )
    assert selected is reachable
    assert reachable.score > hinted_high.score
    assert abs(reachable.score_terms["candidate_hint_distance_cost"]) < 1.0
    assert "recovery_cost" in reachable.score_terms


def test_legacy_arena_identity_ordering_is_unchanged_by_v9_1_scoring():
    memory = TrunkTargetMemory()
    near_player = memory.observe(hit(0.0, 65.62, 2.0), 1)
    near_initial_arena_target = memory.observe(hit(10.0, 70.0, 0.0), 2)
    selected = memory.select(
        3, telemetry(), candidate_hint=(10.0, 0.0), score_config=None
    )
    assert selected is near_initial_arena_target
    assert selected is not near_player


def test_v9_1_does_not_force_a_lone_far_exact_target_into_local_contact():
    memory = TrunkTargetMemory()
    far = memory.observe(hit(29.0, 65.62, 0.0), 1)
    selected = memory.select(
        2,
        telemetry(),
        candidate_hint=(29.0, 0.0),
        score_config=TrunkTargetScoreConfig(
            hint_distance_weight=0.05,
            horizontal_distance_weight=1.0,
            vertical_distance_weight=1.25,
            reach_excess_weight=2.0,
            maximum_horizontal_distance=14.0,
        ),
    )
    assert selected is None
    assert memory.last_selection_reason == "no_eligible_targets"
    assert far.eligible is False
    assert far.score < -70.0
    assert far.score_terms["horizontal_distance_cost"] == -29.0


def test_coordinate_progress_window_detects_no_translation():
    monitor = CoordinateProgressMonitor(window_size=4, minimum_progress=0.3)
    for distance in (8.0, 7.95, 7.9, 7.85):
        monitor.add(distance)
    assert monitor.is_stalled()
    assert monitor.diagnostics()["observed_progress"] < 0.3
    monitor.reset()
    for distance in (8.0, 7.8, 7.5, 7.2):
        monitor.add(distance)
    assert not monitor.is_stalled()


def test_controller_recomputes_3d_aim_and_attacks_only_after_confirmation():
    controller = coordinate_controller()
    pose = telemetry()
    log = hit(0.0, 65.62, 3.0, in_range=False)
    controller.observe_raycast_target(log, pose, global_step=1)
    controller.engage(
        telemetry=pose,
        candidate_id=4,
        global_step=2,
        target_hint=(0.0, 3.0),
    )
    assert controller.state == TrunkContactState.COORDINATE_AIM
    assert controller.act(POV, pose, raycast=log) == controller.config.forward_action

    moved_pose = telemetry(z=1.2)
    in_range = hit(0.0, 65.62, 3.0, in_range=True)
    action = controller.act(POV, moved_pose, raycast=in_range)
    assert action == controller.config.attack_action
    assert controller.state == TrunkContactState.ATTACK_TRUNK
    assert controller.diagnostics()["coordinate_horizontal_distance"] < 3.0


def test_controller_turns_and_changes_pitch_before_moving():
    controller = coordinate_controller()
    pose = telemetry()
    target_hit = hit(4.0, 68.0, 0.0)
    controller.observe_raycast_target(target_hit, pose, 1)
    controller.engage(telemetry=pose, global_step=2, target_hint=(4.0, 0.0))
    assert controller.act(POV, pose, raycast=None) == controller.config.left_action

    facing_east = telemetry(yaw=-90.0)
    assert controller.act(POV, facing_east, raycast=None) == controller.config.look_up_action


def test_coordinate_profile_falls_back_instead_of_looping_forever():
    controller = coordinate_controller(
        coordinate_miss_budget=2,
        coordinate_forward_stop_distance=10.0,
    )
    pose = telemetry()
    target_hit = hit(0.0, 65.62, 1.0)
    controller.observe_raycast_target(target_hit, pose, 1)
    controller.engage(telemetry=pose, global_step=2, target_hint=(0.0, 1.0))
    controller.act(POV, pose, raycast=None)
    controller.act(POV, pose, raycast=None)
    assert controller.state == TrunkContactState.FIND_TRUNK
    assert controller.counters.coordinate_aim_fallbacks == 1


def test_late_local_raycast_bootstraps_coordinate_aim():
    controller = coordinate_controller()
    pose = telemetry()
    controller.engage(telemetry=pose, global_step=1, target_hint=(0.0, 2.0))
    assert controller.state == TrunkContactState.APPROACH_REGION
    controller.observe_raycast_target(
        hit(0.0, 65.62, 2.0), pose, global_step=2
    )
    assert controller.state == TrunkContactState.COORDINATE_AIM
    assert controller.diagnostics()["coordinate_target_id"] == 1


def test_v9_1_far_local_raycast_stays_in_rgb_contact_instead_of_hijacking():
    controller = v9_1_controller()
    pose_value = telemetry()
    controller.engage(
        telemetry=pose_value, global_step=1, target_hint=(29.0, 0.0)
    )
    controller.observe_raycast_target(
        hit(29.0, 65.62, 0.0), pose_value, global_step=2
    )
    assert controller.state == TrunkContactState.APPROACH_REGION
    assert controller.diagnostics()["coordinate_target_id"] is None
    assert controller.counters.coordinate_no_eligible_targets == 1


def test_v9_2_no_exact_target_starts_one_bounded_local_rescan():
    controller = v9_2_controller(exact_log_rescan_budget=40)
    pose_value = telemetry()
    controller.engage(
        telemetry=pose_value,
        candidate_id=7,
        global_step=20,
        target_hint=(0.0, 3.0),
    )
    assert controller.state == TrunkContactState.EXACT_LOG_RESCAN
    actions = []
    while controller.result is None:
        actions.append(controller.act(POV, pose_value, raycast=None))
        assert len(actions) <= 41
    assert len(actions) == 41  # 40 camera samples plus the terminal noop.
    assert controller.result == "replan"
    assert controller.counters.exact_log_rescan_attempts == 1
    assert controller.counters.exact_log_rescan_steps == 40
    assert controller.counters.exact_log_rescan_failures == 1
    assert set(actions[:-1]).issubset(
        {
            controller.config.fine_left_action,
            controller.config.fine_right_action,
            controller.config.look_up_action,
            controller.config.look_down_action,
        }
    )
    # The yaw and pitch command deltas both sum to zero, so a failed scan does
    # not hand global replanning a rotated camera.
    assert actions[:-1].count(controller.config.fine_left_action) == 15
    assert actions[:-1].count(controller.config.fine_right_action) == 15
    assert actions[:-1].count(controller.config.look_up_action) == 5
    assert actions[:-1].count(controller.config.look_down_action) == 5


def test_v9_2_rescan_adopts_near_log_and_returns_to_coordinate_aim():
    controller = v9_2_controller()
    pose_value = telemetry()
    controller.engage(
        telemetry=pose_value,
        candidate_id=8,
        global_step=10,
        target_hint=(0.0, 3.0),
    )
    controller.act(POV, pose_value, raycast=None)
    controller.observe_raycast_target(
        hit(0.0, 65.62, 3.0, in_range=True),
        pose_value,
        global_step=11,
    )
    assert controller.state == TrunkContactState.COORDINATE_AIM
    assert controller.counters.exact_log_rescan_successes == 1
    assert controller.diagnostics()["coordinate_target_id"] == 1
    assert (
        controller.act(
            POV,
            pose_value,
            raycast=hit(0.0, 65.62, 3.0, in_range=True),
        )
        == controller.config.attack_action
    )


def test_v9_2_rescan_rejects_far_exact_point_and_cannot_repeat_candidate():
    controller = v9_2_controller(exact_log_rescan_budget=2)
    pose_value = telemetry()
    controller.engage(
        telemetry=pose_value,
        candidate_id=9,
        global_step=10,
        target_hint=(29.0, 0.0),
    )
    controller.observe_raycast_target(
        hit(29.0, 65.62, 0.0), pose_value, global_step=11
    )
    assert controller.state == TrunkContactState.EXACT_LOG_RESCAN
    assert controller.diagnostics()["coordinate_target_id"] is None
    while controller.result is None:
        controller.act(POV, pose_value, raycast=None)
    controller.engage(
        telemetry=pose_value,
        candidate_id=9,
        global_step=20,
        target_hint=(29.0, 0.0),
    )
    assert controller.result == "replan"
    assert controller.counters.exact_log_rescan_attempts == 1


def test_v9_2_contact_loop_budget_enters_scan_then_fails_finitely():
    controller = v9_2_controller(
        exact_log_rescan_budget=3,
        exact_log_rescan_loop_budget=2,
    )
    pose_value = telemetry()
    controller.observe_raycast_target(
        hit(0.0, 65.62, 3.0), pose_value, global_step=1
    )
    controller.engage(
        telemetry=pose_value,
        candidate_id=10,
        global_step=2,
        target_hint=(0.0, 3.0),
    )
    # Exercise the loop counter directly through real state transitions. The
    # next act must route into the one bounded scan instead of looping again.
    for _ in range(2):
        controller.state = TrunkContactState.ADJUST_PITCH
        controller._transition(
            TrunkContactState.CENTER_TRUNK, "trunk lost during pitch"
        )
    controller.act(POV, pose_value, raycast=None)
    assert controller.state == TrunkContactState.EXACT_LOG_RESCAN
    while controller.result is None:
        controller.act(POV, pose_value, raycast=None)
    assert controller.result == "replan"
    assert controller.counters.center_adjust_loop_cycles == 2
    assert controller.counters.exact_log_rescan_attempts == 1
    assert controller.counters.exact_log_rescan_failures == 1


def _drain_coordinate_recovery(controller, pose):
    steps = 0
    while controller.state == TrunkContactState.COORDINATE_RECOVER:
        controller.act(POV, pose, raycast=None)
        steps += 1
        assert steps < 30


def test_stall_recovers_once_then_switches_to_second_target():
    controller = recovery_controller(
        coordinate_progress_window=3,
        coordinate_minimum_progress=0.1,
        coordinate_recovery_backward_steps=1,
        coordinate_recovery_turn_steps=1,
        coordinate_recovery_jump_steps=1,
    )
    pose = telemetry()
    controller.observe_raycast_target(hit(0.0, 65.62, 8.0), pose, 1)
    controller.observe_raycast_target(hit(0.0, 65.62, 10.0), pose, 2)
    controller.engage(telemetry=pose, global_step=3, target_hint=(0.0, 8.0))
    first_id = controller.diagnostics()["coordinate_target_id"]

    for _ in range(3):
        action = controller.act(POV, pose, raycast=None)
    assert action == controller.config.backward_action
    assert controller.state == TrunkContactState.COORDINATE_RECOVER
    _drain_coordinate_recovery(controller, pose)

    for _ in range(3):
        controller.act(POV, pose, raycast=None)
    assert controller.state == TrunkContactState.COORDINATE_AIM
    assert controller.diagnostics()["coordinate_target_id"] != first_id
    assert controller.counters.coordinate_recoveries == 1
    assert controller.counters.coordinate_target_switches == 1


def test_all_coordinate_targets_cooldown_returns_global_replan():
    controller = recovery_controller(
        coordinate_progress_window=2,
        coordinate_minimum_progress=0.1,
        coordinate_recovery_backward_steps=0,
        coordinate_recovery_turn_steps=0,
        coordinate_recovery_jump_steps=0,
    )
    pose = telemetry()
    controller.observe_raycast_target(hit(0.0, 65.62, 8.0), pose, 1)
    controller.engage(telemetry=pose, global_step=2, target_hint=(0.0, 8.0))
    # First stall consumes the one recovery; the next stalls and cools down
    # the only target, which must return control to global REPLAN.
    for _ in range(2):
        controller.act(POV, pose, raycast=None)
    _drain_coordinate_recovery(controller, pose)
    for _ in range(2):
        controller.act(POV, pose, raycast=None)
    assert controller.result == "replan"
    assert controller.counters.coordinate_all_targets_cooldown == 1


def _enter_v9_1_post_recovery(controller, pose_value):
    for _ in range(controller.config.coordinate_progress_window):
        controller.act(POV, pose_value, raycast=None)
    assert controller.state == TrunkContactState.COORDINATE_RECOVER
    while controller.state == TrunkContactState.COORDINATE_RECOVER:
        controller.act(POV, pose_value, raycast=None)
    assert controller.state == TrunkContactState.POST_RECOVERY_VERIFY


def test_post_recovery_verification_has_independent_translation_budget():
    controller = v9_1_controller(
        region_max_steps=2,
        coordinate_progress_window=2,
        coordinate_minimum_progress=0.1,
        coordinate_recovery_backward_steps=0,
        coordinate_recovery_turn_steps=0,
        coordinate_recovery_jump_steps=0,
        coordinate_post_recovery_translation_budget=4,
        coordinate_post_recovery_max_steps=10,
    )
    pose_value = telemetry()
    controller.observe_raycast_target(hit(0.0, 65.62, 8.0), pose_value, 1)
    controller.observe_raycast_target(hit(0.0, 65.62, 10.0), pose_value, 2)
    controller.engage(telemetry=pose_value, global_step=3, target_hint=(0.0, 8.0))
    _enter_v9_1_post_recovery(controller, pose_value)
    while controller.state == TrunkContactState.POST_RECOVERY_VERIFY:
        controller.act(POV, pose_value, raycast=None)
    assert controller.counters.coordinate_post_recovery_no_progress == 1
    assert controller.counters.coordinate_target_switches == 1
    assert controller.result is None


def test_post_recovery_progress_returns_to_coordinate_aim_without_switch():
    controller = v9_1_controller(
        coordinate_progress_window=2,
        coordinate_minimum_progress=0.1,
        coordinate_recovery_backward_steps=0,
        coordinate_recovery_turn_steps=0,
        coordinate_recovery_jump_steps=0,
        coordinate_post_recovery_translation_budget=4,
        coordinate_post_recovery_minimum_progress=0.2,
    )
    start = telemetry()
    controller.observe_raycast_target(hit(0.0, 65.62, 8.0), start, 1)
    controller.engage(telemetry=start, global_step=2, target_hint=(0.0, 8.0))
    _enter_v9_1_post_recovery(controller, start)
    controller.act(POV, telemetry(z=0.4), raycast=None)
    assert controller.state == TrunkContactState.COORDINATE_AIM
    assert controller.counters.coordinate_post_recovery_progress == 1
    assert controller.counters.coordinate_target_switches == 0


def test_post_recovery_camera_actions_do_not_consume_translation_samples():
    controller = v9_1_controller(coordinate_post_recovery_max_steps=20)
    pose_value = telemetry()
    controller.observe_raycast_target(hit(8.0, 65.62, 0.0), pose_value, 1)
    controller.engage(telemetry=pose_value, global_step=2, target_hint=(8.0, 0.0))
    controller._start_post_recovery_verification(pose_value)
    for _ in range(5):
        assert controller.act(POV, pose_value, raycast=None) in (
            controller.config.left_action,
            controller.config.fine_left_action,
        )
    post = controller.diagnostics()["coordinate_post_recovery"]
    assert post["translation_samples"] == 0


def test_v9_1_all_targets_cooling_returns_global_replan():
    controller = v9_1_controller(
        coordinate_post_recovery_translation_budget=2,
        coordinate_post_recovery_max_steps=5,
    )
    pose_value = telemetry()
    controller.observe_raycast_target(hit(0.0, 65.62, 8.0), pose_value, 1)
    controller.engage(telemetry=pose_value, global_step=2, target_hint=(0.0, 8.0))
    controller._coordinate_target.recovery_attempts = 1
    controller._start_post_recovery_verification(pose_value)
    controller.act(POV, pose_value, raycast=None)
    controller.act(POV, pose_value, raycast=None)
    assert controller.result == "replan"
    assert controller.counters.coordinate_all_targets_cooldown == 1


def test_late_episode_does_not_start_unverifiable_coordinate_recovery():
    controller = v9_1_controller(
        coordinate_progress_window=2,
        coordinate_minimum_progress=0.1,
    )
    pose_value = telemetry()
    controller.observe_raycast_target(hit(0.0, 65.62, 8.0), pose_value, 294)
    controller.engage(
        telemetry=pose_value, global_step=295, target_hint=(0.0, 8.0)
    )
    controller.act(POV, pose_value, raycast=None)
    controller.act(POV, pose_value, raycast=None)
    assert controller.counters.coordinate_recoveries == 0
    assert controller.result == "replan"
    assert any(
        "insufficient episode budget" in row[3]
        for row in controller.counters.transitions
    )


def test_v9_4_coordinate_recovery_has_one_extended_jump_segment():
    controller = v9_4_controller(
        coordinate_progress_window=2,
        coordinate_minimum_progress=0.1,
        coordinate_recovery_backward_steps=0,
        coordinate_recovery_turn_steps=0,
    )
    pose_value = telemetry()
    controller.observe_raycast_target(hit(0.0, 65.62, 8.0), pose_value, 1)
    controller.engage(telemetry=pose_value, global_step=2, target_hint=(0.0, 8.0))

    controller.act(POV, pose_value, raycast=None)
    controller.act(POV, pose_value, raycast=None)

    assert controller.state == TrunkContactState.COORDINATE_RECOVER
    queued = list(controller._coordinate_recovery_actions)
    assert queued.count(controller.config.forward_jump_action) == 7
    assert controller.counters.coordinate_recoveries == 1
    assert controller._coordinate_target.recovery_attempts == 1


def test_frozen_profiles_do_not_enable_coordinate_teacher():
    for profile in ("v6_1_baseline", "clear_occlusion", "drop_recovery"):
        assert not TrunkContactConfig.for_profile(profile).enable_coordinate_aim
    assert not TrunkContactConfig.for_profile(
        CONTACT_PROFILE_COORDINATE_AIM
    ).enable_coordinate_recovery
    v9 = TrunkContactConfig.for_profile(CONTACT_PROFILE_COORDINATE_RECOVERY)
    v9_1 = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_COORDINATE_RECOVERY_V9_1
    )
    v9_2 = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2
    )
    v9_3 = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3
    )
    v9_4 = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4
    )
    v9_5 = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5
    )
    assert v9.coordinate_hint_distance_weight == 2.0
    assert v9.enable_post_recovery_verification is False
    assert v9.enable_enhanced_drop_recovery is False
    assert v9.drop_recovery_max_steps == 20
    assert v9.drop_recovery_contact_extension_steps == 0
    assert v9.coordinate_maximum_horizontal_selection_distance is None
    assert v9_1.coordinate_hint_distance_weight == 0.05
    assert v9_1.enable_post_recovery_verification is True
    assert v9_1.enable_enhanced_drop_recovery is True
    assert v9_1.drop_recovery_centre_max_steps == 24
    assert v9_1.drop_recovery_max_steps == 72
    assert v9_1.drop_recovery_contact_extension_steps == 72
    assert v9_1.coordinate_maximum_horizontal_selection_distance == 14.0
    assert v9_1.enable_exact_log_rescan is False
    assert v9_1.require_raycast_attack_confirmation is False
    assert v9_2.coordinate_hint_distance_weight == 0.05
    assert v9_2.coordinate_maximum_horizontal_selection_distance == 14.0
    assert v9_2.enable_post_recovery_verification is True
    assert v9_2.enable_enhanced_drop_recovery is True
    assert v9_2.enable_exact_log_rescan is True
    assert v9_2.require_raycast_attack_confirmation is True
    assert v9_3.coordinate_hint_distance_weight == 0.05
    assert v9_3.coordinate_maximum_horizontal_selection_distance == 14.0
    assert v9_3.enable_exact_log_rescan is True
    assert v9_3.require_raycast_attack_confirmation is True
    assert v9_3.coordinate_recovery_jump_steps == 4
    assert v9_3.drop_recovery_centre_max_steps == 24
    assert v9_3.drop_recovery_waypoint_max_steps == 7
    assert v9_3.drop_recovery_normalize_block_centre is False
    assert v9_3.drop_recovery_ordered_ring is False
    assert v9_4.enable_exact_log_rescan is True
    assert v9_4.require_raycast_attack_confirmation is True
    assert v9_4.coordinate_recovery_jump_steps == 8
    assert v9_4.drop_recovery_centre_max_steps == 28
    assert v9_4.drop_recovery_waypoint_max_steps == 10
    assert v9_4.drop_recovery_max_steps == 72
    assert v9_4.drop_recovery_normalize_block_centre is True
    assert v9_4.drop_recovery_ordered_ring is True
    assert v9_4.enable_spatial_exact_log_rescan_cooldown is False
    assert v9_5.enable_exact_log_rescan is True
    assert v9_5.require_raycast_attack_confirmation is True
    assert v9_5.coordinate_recovery_jump_steps == 8
    assert v9_5.drop_recovery_centre_max_steps == 28
    assert v9_5.drop_recovery_waypoint_max_steps == 10
    assert v9_5.drop_recovery_normalize_block_centre is True
    assert v9_5.drop_recovery_ordered_ring is True
    assert v9_5.enable_spatial_exact_log_rescan_cooldown is True
    assert v9_5.exact_log_rescan_spatial_radius == 8.0


def test_v9_5_failed_exact_scan_blocks_new_candidate_in_same_physical_region():
    controller = v9_5_controller(exact_log_rescan_budget=2)
    first_pose = telemetry(x=0.0, z=0.0)
    controller.engage(
        telemetry=first_pose,
        candidate_id=1,
        global_step=10,
    )
    assert controller.state == TrunkContactState.EXACT_LOG_RESCAN
    controller.act(POV, first_pose, raycast=None, global_step=10)
    controller.act(POV, first_pose, raycast=None, global_step=11)
    controller.act(POV, first_pose, raycast=None, global_step=12)
    assert controller.result == "replan"
    assert controller.diagnostics()["exact_log_rescan"]["failed_regions"] == [
        (0.0, 0.0)
    ]

    nearby_pose = telemetry(x=3.0, z=0.0)
    controller.engage(
        telemetry=nearby_pose,
        candidate_id=2,
        global_step=20,
    )
    assert controller.result == "replan"
    assert controller.counters.spatial_exact_log_rescan_rejections == 1
    assert controller.counters.exact_log_rescan_attempts == 1

    outside_pose = telemetry(x=8.01, z=0.0)
    controller.engage(
        telemetry=outside_pose,
        candidate_id=3,
        global_step=30,
    )
    assert controller.result is None
    assert controller.state == TrunkContactState.EXACT_LOG_RESCAN
    assert controller.counters.exact_log_rescan_attempts == 2
