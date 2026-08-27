"""v9.8 raycast-owned handoff and frozen-boundary regressions."""

from dataclasses import fields as dataclass_fields

import numpy as np

from mc_rl.candidates import ResourceDetection
from mc_rl.resource_adapters import ResourceAdapter, TreeDetection, TrunkView
from mc_rl.search_policy import CandidateSearchPolicy, SearchConfig, SearchState
from mc_rl.telemetry import AgentTelemetry, RaycastHit, VisualRangeEstimate
from mc_rl.trunk_contact import (
    CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
    CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
    TrunkContactConfig,
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


def telemetry():
    return AgentTelemetry(x=0.0, y=64.0, z=0.0, yaw=0.0, pitch=0.0)


def policy(profile):
    return CandidateSearchPolicy(
        Adapter(),
        SearchConfig(sensor_profile="f3_raycast", contact_profile=profile),
    )


def select_candidate(subject, z=7.0):
    candidate, _merged = subject.candidate_map.add_detection(
        ResourceDetection("tree", 0.0, 0.9, z),
        0.0,
        0,
        telemetry=telemetry(),
        range_estimate=VisualRangeEstimate(z, 1.0),
    )
    candidate.estimated_world_x = 0.0
    candidate.estimated_world_z = z
    candidate.position_uncertainty = 1.0
    candidate.status = "selected"
    subject.selected_candidate = candidate
    subject.current_telemetry = telemetry()
    subject._capture_route_target()
    subject.state = SearchState.APPROACH
    return candidate


def test_v9_8_contact_profile_is_identical_to_frozen_v9_7():
    v9_7 = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7
    )
    v9_8 = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8
    )
    for field in dataclass_fields(TrunkContactConfig):
        assert getattr(v9_8, field.name) == getattr(v9_7, field.name), field.name
    assert v9_8.drop_recovery_centre_max_steps == 48


def test_v9_8_enables_raycast_owned_handoff_without_changing_default():
    assert SearchConfig().require_raycast_handoff_confirmation is False
    assert policy(CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8)._raycast_owned_handoff
    assert not policy(
        CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7
    )._raycast_owned_handoff


def test_v9_8_visual_trunk_guides_route_but_cannot_engage_contact():
    subject = policy(CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8)
    select_candidate(subject, z=7.0)
    trunk = TreeDetection(
        "tree", 0.0, 0.9, 100.0, sees_trunk=True
    )

    action, engaged = subject._contact_step(
        np.zeros((8, 8, 3), dtype=np.uint8),
        telemetry(),
        (0.0, 7.0),
        trunk,
    )

    assert action is None
    assert engaged is False
    assert subject.handoff_visual_confirmations == 0
    assert subject.handoff_guard_rejections == 1
    assert not subject._contact.active


def test_v9_8_exact_log_raycast_can_engage_contact():
    subject = policy(CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8)
    candidate = select_candidate(subject, z=7.0)
    raycast = RaycastHit(
        has_block=True,
        is_log=True,
        is_leaves=False,
        in_range=False,
        distance=7.0,
        x=0.0,
        y=65.0,
        z=7.0,
    )
    subject._contact.observe_raycast_target(raycast, telemetry(), 10)

    action, engaged = subject._contact_step(
        np.zeros((8, 8, 3), dtype=np.uint8),
        telemetry(),
        (0.0, 7.0),
        None,
        raycast=raycast,
    )

    assert engaged is True
    assert action is not None
    assert subject.handoff_raycast_confirmations == 1
    assert subject._contact.candidate_id == candidate.candidate_id


def test_v9_8_scan_memory_can_engage_after_crosshair_moves_off_log():
    subject = policy(CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8)
    candidate = select_candidate(subject, z=12.0)
    remembered = RaycastHit(
        has_block=True,
        is_log=True,
        is_leaves=False,
        in_range=False,
        distance=10.0,
        x=1.0,
        y=65.0,
        z=10.0,
    )
    subject._contact.observe_raycast_target(remembered, telemetry(), 20)

    action, engaged = subject._contact_step(
        np.zeros((8, 8, 3), dtype=np.uint8),
        telemetry(),
        (0.0, 12.0),
        None,
        raycast=None,
    )

    assert engaged is True
    assert action is not None
    assert subject.handoff_raycast_memory_confirmations == 1
    assert subject.handoff_raycast_confirmations == 0
    assert subject._contact.candidate_id == candidate.candidate_id


def test_v9_8_exact_scan_memory_overrides_noisy_visual_route():
    subject = policy(CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8)
    select_candidate(subject, z=24.0)
    remembered = RaycastHit(
        has_block=True,
        is_log=True,
        is_leaves=False,
        in_range=False,
        distance=18.0,
        x=4.0,
        y=65.0,
        z=17.0,
    )
    subject._contact.observe_raycast_target(remembered, telemetry(), 20)

    assert subject._capture_raycast_memory_route()
    assert subject._route_target_x == 4.0
    assert subject._route_target_z == 17.0
    assert subject._route_target_uncertainty == 0.0
    assert subject.raycast_memory_route_selections == 1
    yaw_error, distance = subject._world_route()
    assert distance < 18.0
    assert yaw_error != 0.0
