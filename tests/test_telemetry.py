import numpy as np

from mc_rl.candidates import CandidateMap, ResourceDetection
from mc_rl.telemetry import (
    AgentTelemetry,
    RaycastHit,
    SENSOR_PROFILE_F3,
    SENSOR_PROFILE_POV_ONLY,
    SENSOR_PROFILE_RAYCAST,
    VisualRangeEstimate,
    bearing_and_distance_to,
    detection_world_position,
    sensor_uses_telemetry,
)


def pose(x=0.0, z=0.0, yaw=0.0):
    return AgentTelemetry(x=x, y=4.0, z=z, yaw=yaw, pitch=0.0)


def test_detection_projects_with_minecraft_yaw_convention():
    distance = VisualRangeEstimate(10.0, 1.0)
    assert np.allclose(detection_world_position(pose(), 0.0, distance), (0, 4, 10))
    assert np.allclose(detection_world_position(pose(), 90.0, distance), (-10, 4, 0))
    assert np.allclose(detection_world_position(pose(), -90.0, distance), (10, 4, 0))


def test_raycast_profile_and_hit_are_explicitly_privileged():
    assert not sensor_uses_telemetry(SENSOR_PROFILE_POV_ONLY)
    assert sensor_uses_telemetry(SENSOR_PROFILE_F3)
    assert sensor_uses_telemetry(SENSOR_PROFILE_RAYCAST)
    hit = RaycastHit.from_observation(
        {
            "has_block": 1,
            "is_log": 1,
            "is_leaves": 0,
            "in_range": 1,
            "distance": 3.0,
            "x": 1.5,
            "y": 65.0,
            "z": -2.5,
        }
    )
    assert hit.has_block and hit.is_log and hit.in_range
    assert not hit.is_leaves


def test_same_candidate_fuses_observations_after_translation():
    memory = CandidateMap(world_merge_min_distance=2.5)
    first_detection = ResourceDetection("tree", 0.0, 0.9, 20.0)
    first, merged = memory.add_detection(
        first_detection,
        observer_yaw=0.0,
        step=0,
        telemetry=pose(),
        range_estimate=VisualRangeEstimate(10.0, 1.5),
    )
    assert not merged

    second_pose = pose(x=2.0)
    bearing, distance = bearing_and_distance_to(second_pose, 0.0, 10.0)
    second, merged = memory.add_detection(
        ResourceDetection("tree", bearing, 0.9, 22.0),
        observer_yaw=0.0,
        step=10,
        telemetry=second_pose,
        range_estimate=VisualRangeEstimate(distance, 1.5),
    )
    assert merged
    assert second is first
    assert first.position_observation_count == 2
    assert np.allclose(
        [first.estimated_world_x, first.estimated_world_z], [0.0, 10.0]
    )
    assert first.position_uncertainty < 1.5


def test_distinct_world_positions_do_not_merge_despite_similar_bearing():
    memory = CandidateMap(world_merge_min_distance=2.0)
    detection = ResourceDetection("tree", 0.0, 0.9, 20.0)
    memory.add_detection(
        ResourceDetection("tree", 0.0, 0.9, 2.0),
        0.0,
        0,
        telemetry=pose(),
        range_estimate=VisualRangeEstimate(5.0, 1.0),
    )
    memory.add_detection(
        detection,
        0.0,
        1,
        telemetry=pose(),
        range_estimate=VisualRangeEstimate(12.0, 1.0),
    )
    assert len(memory.candidates) == 2


def test_remembered_world_point_updates_bearing_after_translation():
    memory = CandidateMap()
    candidate, _ = memory.add_detection(
        ResourceDetection("tree", 0.0, 0.9, 20.0),
        0.0,
        0,
        telemetry=pose(),
        range_estimate=VisualRangeEstimate(10.0, 1.0),
    )
    memory.refresh_world_bearings(pose(x=10.0))
    assert np.isclose(candidate.relative_yaw, 45.0)


def test_selected_candidate_rejects_large_range_outlier():
    memory = CandidateMap()
    candidate, _ = memory.add_detection(
        ResourceDetection("tree", 0.0, 0.9, 20.0),
        0.0,
        0,
        telemetry=pose(),
        range_estimate=VisualRangeEstimate(10.0, 1.0),
    )
    memory.update_candidate_position(
        candidate,
        ResourceDetection("tree", 0.0, 0.9, 3.0),
        pose(),
        VisualRangeEstimate(20.0, 1.0),
        step=10,
    )
    assert np.isclose(candidate.estimated_world_z, 10.0)
    assert candidate.position_observation_count == 1
