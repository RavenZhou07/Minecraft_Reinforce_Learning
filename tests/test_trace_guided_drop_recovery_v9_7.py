"""Trace-derived v9.7 drop-recovery regression tests."""

from dataclasses import fields as dataclass_fields

import pytest

from mc_rl.drop_recovery import DropRecoveryConfig, DropRecoveryPlanner
from mc_rl.resource_adapters import ResourceAdapter
from mc_rl.search_policy import CandidateSearchPolicy, SearchConfig
from mc_rl.telemetry import AgentTelemetry
from mc_rl.trunk_contact import (
    CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
    CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
    TrunkContactConfig,
)


V9_7_ONLY_FIELDS = frozenset(
    (
        "drop_recovery_centre_max_steps",
        "drop_recovery_yaw_tolerance_degrees",
        "drop_recovery_orient_ring_to_start",
        "drop_recovery_elevated_jump_centre_only",
    )
)


class Adapter(ResourceAdapter):
    resource_type = "tree"

    def detect(self, pov):
        return []

    def interaction_action(self):
        return 8

    def success(self, observation, reward, info):
        return False


def telemetry(x=0.0, y=64.0, z=0.0, yaw=0.0):
    return AgentTelemetry(x=x, y=y, z=z, yaw=yaw, pitch=0.0)


def test_v9_6_remains_frozen_when_v9_7_fields_are_added():
    v9_6 = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6
    )
    assert v9_6.drop_recovery_centre_max_steps == 28
    assert v9_6.drop_recovery_yaw_tolerance_degrees == 12.0
    assert v9_6.drop_recovery_orient_ring_to_start is False
    assert v9_6.drop_recovery_elevated_jump_centre_only is False


def test_v9_7_inherits_v9_6_except_trace_derived_drop_fields():
    v9_6 = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6
    )
    v9_7 = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7
    )
    for field in dataclass_fields(TrunkContactConfig):
        if field.name in V9_7_ONLY_FIELDS:
            continue
        assert getattr(v9_7, field.name) == getattr(v9_6, field.name), field.name
    assert v9_7.drop_recovery_centre_max_steps == 48
    assert v9_7.drop_recovery_yaw_tolerance_degrees == 18.0
    assert v9_7.drop_recovery_orient_ring_to_start is True
    assert v9_7.drop_recovery_elevated_jump_centre_only is True


def test_v9_7_keeps_all_search_side_v9_6_guards_enabled():
    policy = CandidateSearchPolicy(
        Adapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
        ),
    )
    assert policy._candidate_handoff_guard
    assert policy._contact_ownership_guard
    assert policy._terrain_route_recovery


def test_oriented_ring_starts_on_side_nearest_recovery_pose():
    planner = DropRecoveryPlanner(
        10.5,
        20.5,
        DropRecoveryConfig(ordered_ring=True, orient_ring_to_start=True),
    )
    planner.orient_ring_to_start(11.4, 20.5)
    assert planner.waypoints[0] == (10.5, 20.5)
    assert planner.waypoints[1] == pytest.approx((11.35, 20.5))


def test_v9_7_centre_budget_does_not_switch_to_ring_at_old_limit():
    planner = DropRecoveryPlanner(
        0.0,
        0.0,
        DropRecoveryConfig(
            enable_obstacle_recovery=True,
            max_steps=72,
            centre_waypoint_max_steps=48,
            waypoint_max_steps=10,
        ),
    )
    pose = telemetry(x=4.8, z=0.0, yaw=0.0)
    actions = [
        planner.next_action(pose, 1, 3, 4, 0, 2) for _ in range(29)
    ]
    assert all(action == 4 for action in actions)
    assert planner.index == 0


def test_centre_only_elevated_probe_walks_ring_normally_after_centre():
    planner = DropRecoveryPlanner(
        0.0,
        0.0,
        DropRecoveryConfig(
            ordered_ring=True,
            elevated_vertical_gap=1.2,
            elevated_arrival_radius=1.8,
            elevated_jump_steps=8,
            elevated_jump_centre_only=True,
        ),
        centre_y=66.5,
    )
    pose = telemetry(y=64.0)
    assert planner._elevated_pickup_active(pose)
    planner.index = 1
    assert not planner._elevated_pickup_active(pose)
