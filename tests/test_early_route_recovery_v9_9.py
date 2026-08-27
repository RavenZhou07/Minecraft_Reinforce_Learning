"""v9.9 early exact routing, alternating detours, and ground sweep tests."""

from dataclasses import fields as dataclass_fields

import numpy as np
import pytest

from mc_rl.candidates import ResourceDetection
from mc_rl.coordinate_aim import TrunkBlockTarget
from mc_rl.drop_recovery import DropRecoveryConfig, DropRecoveryPlanner
from mc_rl.resource_adapters import ResourceAdapter, TrunkView
from mc_rl.search_policy import CandidateSearchPolicy, SearchConfig, SearchState
from mc_rl.telemetry import AgentTelemetry, RaycastHit, VisualRangeEstimate
from mc_rl.trunk_contact import (
    CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
    CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
    TrunkContactConfig,
    TrunkContactController,
)


class Adapter(ResourceAdapter):
    resource_type = "tree"

    def detect(self, pov):
        return []

    def interaction_action(self):
        return 8

    def success(self, observation, reward, info):
        return False

    def estimate_range(self, detection):
        return VisualRangeEstimate(float(detection.apparent_size), 1.0)

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


def pose(x=0.0, z=0.0, yaw=0.0):
    return AgentTelemetry(x=x, y=64.0, z=z, yaw=yaw, pitch=0.0)


def observation():
    return {
        "pov": np.zeros((8, 8, 3), dtype=np.uint8),
        "raycast": {},
        "telemetry": {
            "x": 0.0,
            "y": 64.0,
            "z": 0.0,
            "yaw": 0.0,
            "pitch": 0.0,
            "biome_id": 1,
            "biome_temperature": 0.8,
            "biome_rainfall": 0.4,
        },
    }


def raycast_observation(x=0.0, z=18.0):
    result = observation()
    result["raycast"] = {
        "has_block": 1.0,
        "is_log": 1.0,
        "in_range": 0.0,
        "distance": 18.0,
        "x": x,
        "y": 65.0,
        "z": z,
    }
    return result


def test_v9_8_remains_frozen_when_v9_9_fields_are_added():
    baseline = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8
    )
    v9_9 = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9
    )

    changed = {
        field.name
        for field in dataclass_fields(TrunkContactConfig)
        if getattr(baseline, field.name) != getattr(v9_9, field.name)
    }
    assert changed == {
        "coordinate_max_recoveries_per_target",
        "enable_coordinate_side_detour",
        "drop_recovery_centre_max_steps",
        "drop_recovery_ground_sweep",
    }
    assert baseline.drop_recovery_centre_max_steps == 48
    assert v9_9.drop_recovery_centre_max_steps == 24
    assert CandidateSearchPolicy(
        Adapter(),
        SearchConfig(
            contact_profile=CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8
        ),
    )._terrain_route_failure_limit == 2
    assert CandidateSearchPolicy(
        Adapter(),
        SearchConfig(
            contact_profile=CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9
        ),
    )._terrain_route_failure_limit == 1


def test_v9_9_ends_scan_when_visual_candidate_and_exact_log_are_known():
    policy = CandidateSearchPolicy(
        Adapter(),
        SearchConfig(
            sensor_profile="f3_raycast",
            contact_profile=CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
        ),
    )
    policy.candidate_map.add_detection(
        ResourceDetection("tree", 0.0, 0.9, 9.0),
        0.0,
        0,
        telemetry=pose(),
        range_estimate=VisualRangeEstimate(9.0, 1.0),
    )
    policy._contact.observe_raycast_target(
        RaycastHit(
            has_block=True,
            is_log=True,
            is_leaves=False,
            in_range=False,
            distance=9.0,
            x=0.0,
            y=65.0,
            z=9.0,
        ),
        pose(),
        0,
    )

    policy.act(observation())

    assert policy.exact_log_early_scan_exits == 1
    assert policy.scan_yaw == 0.0
    assert policy.state is not SearchState.SCAN
    assert any(
        row["old_state"] == "SCAN"
        and "end scan early" in row["reason"]
        for row in policy.transition_log
    )


def test_v9_9_coordinate_retries_mirror_right_then_left():
    controller = TrunkContactController(
        Adapter(),
        TrunkContactConfig.for_profile(
            CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
            coordinate_recovery_backward_steps=1,
            coordinate_side_detour_turn_steps=2,
            coordinate_side_detour_translation_steps=3,
        ),
    )
    controller._coordinate_target = TrunkBlockTarget(1, 0.0, 65.0, 5.0, 0)

    first = [controller._start_coordinate_recovery()]
    first.extend(controller._coordinate_recovery_actions)
    controller._coordinate_recovery_actions.clear()
    second = [controller._start_coordinate_recovery()]
    second.extend(controller._coordinate_recovery_actions)

    assert first == [9, 4, 4, 2, 2, 2, 3, 3]
    assert second == [9, 3, 3, 2, 2, 2, 4, 4]
    assert controller.counters.coordinate_right_detours == 1
    assert controller.counters.coordinate_left_detours == 1
    assert controller._coordinate_recovery_step_budget() == 8


def test_v9_9_adopts_exact_log_first_seen_during_approach():
    policy = CandidateSearchPolicy(
        Adapter(),
        SearchConfig(
            sensor_profile="f3_raycast",
            contact_profile=CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
        ),
    )
    candidate, _ = policy.candidate_map.add_detection(
        ResourceDetection("tree", 0.0, 0.9, 24.0),
        0.0,
        0,
        telemetry=pose(),
        range_estimate=VisualRangeEstimate(24.0, 1.0),
    )
    candidate.status = "selected"
    policy.selected_candidate = candidate
    policy.current_telemetry = pose()
    policy._capture_route_target()
    policy.state = SearchState.APPROACH

    policy.act(raycast_observation(x=4.0, z=17.0))

    assert policy._route_target_x == 4.0
    assert policy._route_target_z == 17.0
    assert policy._route_target_uncertainty == 0.0
    assert policy.dynamic_exact_route_updates == 1
    assert policy.raycast_memory_route_selections == 1


def test_v9_9_ground_sweep_has_projection_outer_revisit_and_inner_ring():
    planner = DropRecoveryPlanner(
        2.0,
        3.0,
        DropRecoveryConfig(
            radius=0.85,
            ordered_ring=True,
            ground_sweep=True,
            ground_sweep_outer_radius=1.6,
        ),
    )

    assert len(planner.waypoints) == 18
    assert planner.waypoints[0] == (2.0, 3.0)
    assert planner.waypoints[9] == (2.0, 3.0)
    assert max(
        ((x - 2.0) ** 2 + (z - 3.0) ** 2) ** 0.5
        for x, z in planner.waypoints[1:9]
    ) == pytest.approx(1.6)
    assert max(
        ((x - 2.0) ** 2 + (z - 3.0) ** 2) ** 0.5
        for x, z in planner.waypoints[10:18]
    ) == pytest.approx(0.85)


def test_v9_9_orients_outer_and_inner_sweeps_independently():
    planner = DropRecoveryPlanner(
        0.0,
        0.0,
        DropRecoveryConfig(
            ordered_ring=True,
            orient_ring_to_start=True,
            ground_sweep=True,
        ),
    )
    planner.orient_ring_to_start(2.0, 0.0)

    assert planner.waypoints[1] == (1.6, 0.0)
    assert planner.waypoints[10] == (0.85, 0.0)
