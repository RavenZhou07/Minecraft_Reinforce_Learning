"""v9.11 state-conditioned emergency target-preemption tests."""

from dataclasses import fields as dataclass_fields

from mc_rl.resource_adapters import ResourceAdapter, TrunkView
from mc_rl.search_policy import CandidateSearchPolicy, SearchConfig
from mc_rl.telemetry import AgentTelemetry, RaycastHit
from mc_rl.trunk_contact import (
    CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10,
    CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11,
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
    return TrunkContactController(
        Adapter(),
        TrunkContactConfig.for_profile(
            CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11,
            **overrides,
        ),
    )


def engaged_subject(current_hit):
    subject = controller()
    subject.observe_raycast_target(current_hit, pose(), 0)
    subject.engage(telemetry=pose(), candidate_id=1, global_step=0)
    return subject


def test_v9_11_changes_only_emergency_switch_from_frozen_v9_10():
    v9_10 = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10
    )
    v9_11 = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11
    )

    changed = {
        field.name
        for field in dataclass_fields(TrunkContactConfig)
        if getattr(v9_10, field.name) != getattr(v9_11, field.name)
    }
    assert changed == {"enable_post_recovery_emergency_preemption"}
    assert not v9_10.enable_post_recovery_emergency_preemption
    assert v9_11.enable_post_recovery_emergency_preemption


def test_v9_11_search_inherits_v9_10_upstream_behaviour():
    policy = CandidateSearchPolicy(
        Adapter(),
        SearchConfig(
            sensor_profile="f3_raycast",
            contact_profile=CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11,
        ),
    )

    assert policy._raycast_owned_handoff
    assert policy._early_exact_log_scan_exit
    assert policy._dynamic_exact_log_route
    assert policy._terrain_route_failure_limit == 1
    assert policy._contact.config.enable_coordinate_target_preemption
    assert policy._contact.config.enable_post_recovery_emergency_preemption


def test_v9_11_keeps_three_observation_rule_during_ordinary_aim():
    subject = engaged_subject(log_hit(0.0, 70.0, 10.0))
    stale = subject._coordinate_target

    subject.observe_raycast_target(log_hit(0.0, 65.5, 5.0), pose(), 9)

    assert subject._coordinate_target is stale
    assert subject.counters.coordinate_target_preemptions == 0
    assert subject.counters.coordinate_target_preemption_support_rejections == 1
    assert subject.counters.coordinate_emergency_preemption_checks == 0


def test_v9_11_accepts_one_lower_better_observation_after_failed_recovery():
    subject = engaged_subject(log_hit(0.0, 70.0, 10.0))
    stale = subject._coordinate_target
    stale.recovery_attempts = 1
    subject.state = TrunkContactState.POST_RECOVERY_VERIFY
    subject._coordinate_post_recovery_samples = 8

    subject.observe_raycast_target(log_hit(0.0, 65.5, 5.0), pose(), 9)

    assert subject._coordinate_target is not stale
    assert subject._coordinate_target.observation_count == 1
    assert stale.status == "cooldown"
    assert subject.state == TrunkContactState.COORDINATE_AIM
    assert subject._coordinate_post_recovery_samples == 0
    assert subject.counters.coordinate_target_preemptions == 1
    assert subject.counters.coordinate_emergency_preemptions == 1
    assert "failed recovery yielded" in subject._coordinate_selection_records[-1][
        "reason"
    ]


def test_v9_11_emergency_rejects_target_that_is_not_lower():
    subject = engaged_subject(log_hit(0.0, 66.0, 10.0))
    stale = subject._coordinate_target
    stale.recovery_attempts = 1
    subject.state = TrunkContactState.POST_RECOVERY_VERIFY

    subject.observe_raycast_target(log_hit(0.0, 67.0, 4.0), pose(), 9)

    assert subject._coordinate_target is stale
    assert stale.status == "selected"
    assert subject.counters.coordinate_emergency_preemptions == 0
    assert subject.counters.coordinate_emergency_preemption_vertical_rejections == 1


def test_v9_11_emergency_requires_a_completed_recovery_attempt():
    subject = engaged_subject(log_hit(0.0, 70.0, 10.0))
    stale = subject._coordinate_target
    subject.state = TrunkContactState.POST_RECOVERY_VERIFY

    subject.observe_raycast_target(log_hit(0.0, 65.5, 5.0), pose(), 9)

    assert subject._coordinate_target is stale
    assert subject.counters.coordinate_emergency_preemptions == 0
    assert subject.counters.coordinate_emergency_preemption_recovery_rejections == 1
