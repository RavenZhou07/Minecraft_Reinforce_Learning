"""Fast tests for the bounded drop waypoint planner."""

from mc_rl.drop_recovery import DropRecoveryConfig, DropRecoveryPlanner
from mc_rl.telemetry import AgentTelemetry


def pose(x=0.0, z=0.0, yaw=0.0):
    return AgentTelemetry(x=x, y=64.0, z=z, yaw=yaw, pitch=0.0)


def test_drop_planner_moves_to_broken_block_centre_first():
    planner = DropRecoveryPlanner(
        0.0, 2.0, DropRecoveryConfig(max_steps=8)
    )

    action = planner.next_action(pose(), 1, 3, 4, 0)

    assert action == 1
    assert planner.current_waypoint == (0.0, 2.0)
    assert planner.diagnostics()["waypoint_index"] == 0


def test_drop_planner_visits_eight_direction_ring_after_centre():
    planner = DropRecoveryPlanner(
        1.0,
        1.0,
        DropRecoveryConfig(arrival_radius=0.4, max_steps=8),
    )

    planner.next_action(pose(x=1.0, z=1.0), 1, 3, 4, 0)

    assert len(planner.waypoints) == 9
    assert planner.reached_waypoints == 1
    assert planner.current_waypoint != (1.0, 1.0)


def test_v9_4_ordered_ring_visits_adjacent_45_degree_waypoints():
    planner = DropRecoveryPlanner(
        1.0,
        1.0,
        DropRecoveryConfig(radius=1.0, ordered_ring=True),
    )

    assert planner.waypoints[:4] == [
        (1.0, 1.0),
        (1.0, 2.0),
        (1.0 + 1.0 / 2.0 ** 0.5, 1.0 + 1.0 / 2.0 ** 0.5),
        (2.0, 1.0),
    ]


def test_default_ring_order_remains_frozen():
    planner = DropRecoveryPlanner(1.0, 1.0, DropRecoveryConfig(radius=1.0))

    assert planner.waypoints[1:5] == [
        (1.0, 2.0),
        (2.0, 1.0),
        (1.0, 0.0),
        (0.0, 1.0),
    ]


def test_drop_planner_skips_waypoint_after_no_forward_progress():
    planner = DropRecoveryPlanner(
        0.0,
        2.0,
        DropRecoveryConfig(blocked_forward_steps=2, max_steps=10),
    )
    stuck = pose()

    planner.next_action(stuck, 1, 3, 4, 0)
    planner.next_action(stuck, 1, 3, 4, 0)
    planner.next_action(stuck, 1, 3, 4, 0)

    assert planner.blocked_waypoints == 1
    assert planner.index == 1


def test_drop_planner_stops_at_global_step_budget():
    planner = DropRecoveryPlanner(
        0.0, 4.0, DropRecoveryConfig(max_steps=2)
    )

    assert planner.next_action(pose(), 1, 3, 4, 0) is not None
    assert planner.next_action(pose(z=0.2), 1, 3, 4, 0) is not None
    assert planner.next_action(pose(z=0.4), 1, 3, 4, 0) is None
    assert planner.complete


def test_enhanced_drop_waypoint_uses_one_jump_then_terminates_blocked():
    planner = DropRecoveryPlanner(
        0.0,
        2.0,
        DropRecoveryConfig(
            enable_obstacle_recovery=True,
            blocked_forward_steps=1,
            waypoint_max_steps=5,
            jump_budget=1,
            offset_budget=0,
            max_steps=10,
        ),
    )
    stuck = pose()
    assert planner.next_action(stuck, 1, 3, 4, 0, 2) == 1
    assert planner.next_action(stuck, 1, 3, 4, 0, 2) == 2
    planner.next_action(stuck, 1, 3, 4, 0, 2)

    assert planner.blocked_waypoints == 1
    first = planner.waypoint_records[0]
    assert first["jump_attempts"] == 1
    assert first["offset_attempts"] == 0
    assert first["end_reason"] == "blocked_after_obstacle_recovery"
    assert first["steps"] <= 5


def test_enhanced_drop_waypoint_offset_and_step_budget_are_finite():
    planner = DropRecoveryPlanner(
        0.0,
        2.0,
        DropRecoveryConfig(
            enable_obstacle_recovery=True,
            blocked_forward_steps=1,
            waypoint_max_steps=3,
            jump_budget=0,
            offset_budget=1,
            max_steps=8,
        ),
    )
    stuck = pose()
    for _ in range(5):
        planner.next_action(stuck, 1, 3, 4, 0, 2)
        if planner.index > 0:
            break

    assert planner.index > 0
    first = planner.waypoint_records[0]
    assert first["offset_attempts"] == 1
    assert first["steps"] <= 3
    assert first["end_reason"] in (
        "blocked_after_obstacle_recovery",
        "waypoint_step_budget",
    )


def test_drop_waypoint_record_includes_reward_and_distance_diagnostics():
    planner = DropRecoveryPlanner(
        0.0,
        2.0,
        DropRecoveryConfig(enable_obstacle_recovery=True, max_steps=8),
    )
    planner.next_action(pose(), 1, 3, 4, 0, 2)
    planner.record_reward(1.0)

    record = planner.waypoint_records[-1]
    assert record["start_distance"] == 2.0
    assert record["minimum_distance"] <= record["start_distance"]
    assert record["end_reason"] == "reward"
    assert record["reward"] == 1.0
