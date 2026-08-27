"""v9.10 online coordinate-target preemption and frozen-boundary tests."""

from dataclasses import fields as dataclass_fields

from mc_rl.resource_adapters import ResourceAdapter, TrunkView
from mc_rl.search_policy import CandidateSearchPolicy, SearchConfig
from mc_rl.telemetry import AgentTelemetry, RaycastHit
from mc_rl.trunk_contact import (
    CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10,
    CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
    TrunkContactConfig,
    TrunkContactController,
    TrunkContactState,
)


class Adapter(ResourceAdapter):
    resource_type = "tree"

    def detect(self, pov):
        return []

    def interaction_action(self):
        return 8

    def success(self, observation, reward, info):
        return False

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


def pose():
    return AgentTelemetry(x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0)


def log_hit(x, y, z):
    return RaycastHit(
        has_block=True,
        is_log=True,
        is_leaves=False,
        in_range=False,
        distance=(x * x + z * z) ** 0.5,
        x=float(x),
        y=float(y),
        z=float(z),
    )


def controller(**overrides):
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10,
        **overrides,
    )
    return TrunkContactController(Adapter(), config)


def test_v9_10_changes_only_preemption_switch_from_frozen_v9_9():
    v9_9 = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9
    )
    v9_10 = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10
    )

    changed = {
        field.name
        for field in dataclass_fields(TrunkContactConfig)
        if getattr(v9_9, field.name) != getattr(v9_10, field.name)
    }
    assert changed == {"enable_coordinate_target_preemption"}
    assert not v9_9.enable_coordinate_target_preemption
    assert v9_10.enable_coordinate_target_preemption


def test_v9_10_search_inherits_v9_9_upstream_behaviour():
    policy = CandidateSearchPolicy(
        Adapter(),
        SearchConfig(
            sensor_profile="f3_raycast",
            contact_profile=CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10,
        ),
    )

    assert policy._raycast_owned_handoff
    assert policy._early_exact_log_scan_exit
    assert policy._dynamic_exact_log_route
    assert policy._terrain_route_failure_limit == 1


def test_v9_10_preempts_stale_target_after_support_margin_and_hold():
    subject = controller()
    subject.observe_raycast_target(log_hit(0.0, 70.0, 10.0), pose(), 0)
    subject.engage(
        telemetry=pose(),
        candidate_id=1,
        global_step=0,
        target_hint=(0.0, 10.0),
    )
    stale = subject._coordinate_target
    subject._coordinate_progress.add(10.0)
    subject._coordinate_progress.add(9.9)
    subject._coordinate_climb_actions.extend([2, 2])

    subject.observe_raycast_target(log_hit(0.0, 65.5, 6.0), pose(), 9)
    subject.observe_raycast_target(log_hit(0.0, 65.5, 6.0), pose(), 10)
    assert subject._coordinate_target is stale

    subject.observe_raycast_target(log_hit(0.0, 65.5, 6.0), pose(), 11)

    assert subject._coordinate_target is not stale
    assert subject._coordinate_target.observation_count == 3
    assert subject.counters.coordinate_target_preemptions == 1
    assert subject.counters.coordinate_target_switches == 1
    assert subject.counters.coordinate_target_preemption_support_rejections == 2
    assert not subject._coordinate_progress.distances
    assert not subject._coordinate_climb_actions
    assert "preempted current target" in subject._coordinate_selection_records[-1][
        "reason"
    ]


def test_v9_10_rejects_better_target_before_minimum_hold_period():
    subject = controller(
        coordinate_target_preemption_min_hold_steps=20,
        coordinate_target_preemption_min_observations=1,
    )
    subject.observe_raycast_target(log_hit(0.0, 70.0, 10.0), pose(), 0)
    subject.engage(telemetry=pose(), candidate_id=1, global_step=0)
    stale = subject._coordinate_target

    subject.observe_raycast_target(log_hit(0.0, 65.5, 5.0), pose(), 9)

    assert subject._coordinate_target is stale
    assert subject.counters.coordinate_target_preemptions == 0
    assert subject.counters.coordinate_target_preemption_hold_rejections == 1


def test_v9_10_restores_selected_status_when_score_margin_is_not_met():
    subject = controller(
        coordinate_target_preemption_score_margin=100.0,
        coordinate_target_preemption_min_observations=1,
    )
    subject.observe_raycast_target(log_hit(0.0, 66.0, 7.0), pose(), 0)
    subject.engage(telemetry=pose(), candidate_id=1, global_step=0)
    stale = subject._coordinate_target

    subject.observe_raycast_target(log_hit(0.0, 65.5, 5.0), pose(), 9)

    assert subject._coordinate_target is stale
    assert stale.status == "selected"
    alternatives = [
        target for target in subject._target_memory.targets if target is not stale
    ]
    assert alternatives and alternatives[0].status == "available"
    assert subject.counters.coordinate_target_preemptions == 0
    assert subject.counters.coordinate_target_preemption_margin_rejections == 1


def test_v9_10_can_end_failed_recovery_verification_with_better_target():
    subject = controller(
        coordinate_target_preemption_min_observations=1,
    )
    subject.observe_raycast_target(log_hit(0.0, 70.0, 10.0), pose(), 0)
    subject.engage(telemetry=pose(), candidate_id=1, global_step=0)
    stale = subject._coordinate_target
    stale.recovery_attempts = 1
    subject.state = TrunkContactState.POST_RECOVERY_VERIFY
    subject._coordinate_post_recovery_samples = 12
    subject._coordinate_post_recovery_state_steps = 13

    subject.observe_raycast_target(log_hit(0.0, 65.5, 5.0), pose(), 9)

    assert subject._coordinate_target is not stale
    assert stale.status == "cooldown"
    assert subject.state == TrunkContactState.COORDINATE_AIM
    assert subject._coordinate_post_recovery_samples == 0
    assert subject._coordinate_post_recovery_state_steps == 0
    assert subject.counters.coordinate_target_preemptions == 1
