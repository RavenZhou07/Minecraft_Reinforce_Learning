"""Vision-guided trunk contact controller for natural Treechop.

The coarse candidate coordinate only has to bring the agent into the tree's
neighbourhood. This controller then owns the final approach purely from POV:
find a vertical trunk, centre it in yaw, aim the camera at it in pitch, and
chop until a log is collected. MineRL 0.4.4 offers no raycast observation,
so crosshair contact is approximated by the trunk-coloured fraction of the
image-centre patch, which is the same signal a player reads from the screen.

No oracle distance or log grid is accepted here. Privileged raycast block
metadata is accepted only through the explicitly selected diagnostic profile;
POV/F3 deployment uses the estimated contact point instead.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from math import floor, hypot
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from mc_rl.resource_adapters import ResourceAdapter, TrunkView
from mc_rl.telemetry import (
    AgentTelemetry,
    RaycastHit,
    bearing_and_distance_to,
)
from mc_rl.navigation import wrap_degrees
from mc_rl.drop_recovery import DropRecoveryConfig, DropRecoveryPlanner
from mc_rl.coordinate_aim import (
    CoordinateProgressMonitor,
    CoordinateAimError,
    TrunkBlockTarget,
    TrunkTargetScoreConfig,
    TrunkTargetMemory,
    coordinate_aim_error,
)


CONTACT_PROFILE_V6_1 = "v6_1_baseline"
CONTACT_PROFILE_CLEAR_OCCLUSION = "clear_occlusion"
CONTACT_PROFILE_DROP_RECOVERY = "drop_recovery"
CONTACT_PROFILE_COORDINATE_AIM = "coordinate_aim"
CONTACT_PROFILE_COORDINATE_RECOVERY = "coordinate_recovery"
CONTACT_PROFILE_COORDINATE_RECOVERY_V9_1 = "coordinate_recovery_v9_1"
CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2 = "coordinate_contact_guard_v9_2"
CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3 = "candidate_handoff_guard_v9_3"
CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4 = "contact_drop_completion_v9_4"
CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5 = (
    "contact_ownership_spatial_guard_v9_5"
)
CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6 = (
    "terrain_route_drop_completion_v9_6"
)
CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7 = (
    "trace_guided_drop_recovery_v9_7"
)
CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8 = "raycast_owned_handoff_v9_8"
CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9 = "early_route_recovery_v9_9"
CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10 = (
    "coordinate_target_preemption_v9_10"
)
CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11 = (
    "emergency_target_preemption_v9_11"
)
CONTACT_PROFILES = frozenset(
    (
        CONTACT_PROFILE_V6_1,
        CONTACT_PROFILE_CLEAR_OCCLUSION,
        CONTACT_PROFILE_DROP_RECOVERY,
        CONTACT_PROFILE_COORDINATE_AIM,
        CONTACT_PROFILE_COORDINATE_RECOVERY,
        CONTACT_PROFILE_COORDINATE_RECOVERY_V9_1,
        CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2,
        CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
        CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
        CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
        CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
        CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
        CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
        CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
        CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10,
        CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11,
    )
)


class TrunkContactState(str, Enum):
    APPROACH_REGION = "APPROACH_REGION"
    COORDINATE_AIM = "COORDINATE_AIM"
    COORDINATE_RECOVER = "COORDINATE_RECOVER"
    POST_RECOVERY_VERIFY = "POST_RECOVERY_VERIFY"
    COORDINATE_REPLAN = "COORDINATE_REPLAN"
    EXACT_LOG_RESCAN = "EXACT_LOG_RESCAN"
    FIND_TRUNK = "FIND_TRUNK"
    CLEAR_OCCLUSION = "CLEAR_OCCLUSION"
    CENTER_TRUNK = "CENTER_TRUNK"
    ADJUST_PITCH = "ADJUST_PITCH"
    ATTACK_TRUNK = "ATTACK_TRUNK"
    BLOCK_DISAPPEARED = "BLOCK_DISAPPEARED"
    DROP_RECOVERY = "DROP_RECOVERY"
    REACQUIRE_SAME_TRUNK = "REACQUIRE_SAME_TRUNK"
    COLLECT_DROP = "COLLECT_DROP"
    VERIFY_PROGRESS = "VERIFY_PROGRESS"
    BACKOFF = "BACKOFF"
    ORBIT_REACQUIRE = "ORBIT_REACQUIRE"
    SUCCESS = "SUCCESS"
    REPLAN = "REPLAN"


@dataclass
class TrunkContactConfig:
    # Contact uses five-degree camera commands while coarse search remains at
    # ten degrees. The smaller deadband avoids the v5 left/right limit cycle.
    yaw_deadband_degrees: float = 3.0
    pitch_deadband_degrees: float = 3.5
    pitch_min_degrees: float = -40.0
    pitch_max_degrees: float = 60.0
    region_stop_radius: float = 2.0
    region_steer_threshold_degrees: float = 12.0
    trunk_present_min_area_px: float = 12.0
    trunk_present_min_height_px: float = 4.0
    # A one-block trunk must fill roughly this much of the frame before the
    # attack can physically reach it (Minecraft reach is ~4 blocks).
    attack_min_area_px: float = 150.0
    attack_min_width_px: float = 4.5
    find_trunk_sweep_actions: int = 4
    find_trunk_budget: int = 18
    # Disabled in the frozen v6.1 profile.  The v6.2 profile enables one
    # bounded forward+attack foliage-clearing pass when an upper-centre leaf
    # cue persists and no usable trunk is visible.
    enable_clear_occlusion: bool = False
    leaf_occlusion_fraction: float = 0.55
    leaf_occlusion_consecutive_frames: int = 2
    occlusion_clear_steps: int = 10
    max_occlusion_clears: int = 1
    # Consecutive CENTER<->ADJUST round trips tolerated before treating
    # the heading as unrecoverable (two brown components ~10 degrees apart
    # can otherwise ping-pong forever across the yaw deadband).
    max_center_cycles: int = 4
    attack_burst_steps: int = 24
    # Optional per-step visual attack permission. It is inert unless an
    # external runner explicitly supplies True/False before ``act``.
    external_attack_gate_recenter_rejections: int = 3
    pickup_probe_steps: int = 2
    # v7 replaces the two-step forward probe with a bounded coordinate search
    # around the last block contact. Older profiles keep their exact behaviour.
    enable_drop_recovery: bool = False
    # v8 is a privileged scripted-teacher experiment. Raycast-observed log
    # points are retained episode-wide and current F3 pose closes a 3-D
    # yaw/pitch loop. The same geometry can later consume POV-estimated points.
    enable_coordinate_aim: bool = False
    coordinate_eye_height: float = 1.62
    coordinate_coarse_threshold_degrees: float = 14.0
    coordinate_target_merge_distance: float = 0.7
    coordinate_target_cooldown_steps: int = 45
    coordinate_miss_budget: int = 8
    coordinate_forward_stop_distance: float = 1.7
    # v9 closes the translation loop and ranks lower/reachable log faces.
    # These remain disabled in the frozen v8 coordinate_aim profile.
    enable_coordinate_recovery: bool = False
    enable_post_recovery_verification: bool = False
    coordinate_progress_window: int = 12
    coordinate_minimum_progress: float = 0.35
    coordinate_max_recoveries_per_target: int = 1
    coordinate_recovery_backward_steps: int = 3
    coordinate_recovery_turn_steps: int = 2
    coordinate_recovery_jump_steps: int = 4
    # v9.9 alternates diagonal obstacle detours for the same exact target.
    # Frozen profiles retain the original one-sided right/jump manoeuvre.
    enable_coordinate_side_detour: bool = False
    coordinate_side_detour_turn_steps: int = 4
    coordinate_side_detour_translation_steps: int = 6
    coordinate_post_recovery_translation_budget: int = 24
    coordinate_post_recovery_max_steps: int = 60
    coordinate_post_recovery_minimum_progress: float = 0.25
    coordinate_recovery_episode_extension_steps: int = 120
    episode_max_steps: int = 300
    coordinate_hint_distance_weight: float = 2.0
    coordinate_horizontal_distance_weight: float = 0.12
    coordinate_vertical_distance_weight: float = 0.9
    coordinate_reach_excess_weight: float = 0.35
    coordinate_failure_weight: float = 2.0
    coordinate_recovery_weight: float = 0.0
    coordinate_observation_weight: float = 0.05
    coordinate_attack_reach: float = 4.0
    coordinate_maximum_horizontal_selection_distance: Optional[float] = None
    # v9.10 may replace a stale coordinate target when a repeatedly observed
    # reachable log has a materially better score. Margin, support, and hold
    # constraints prevent per-frame target oscillation.
    enable_coordinate_target_preemption: bool = False
    coordinate_target_preemption_score_margin: float = 3.0
    coordinate_target_preemption_min_observations: int = 3
    coordinate_target_preemption_min_hold_steps: int = 8
    # v9.11 keeps the ordinary three-observation rule, but a target that has
    # already failed obstacle recovery may yield during verification to one
    # lower, reachable, score-improving exact observation.
    enable_post_recovery_emergency_preemption: bool = False
    coordinate_emergency_preemption_min_observations: int = 1
    coordinate_emergency_preemption_min_vertical_drop: float = 0.75
    coordinate_emergency_preemption_score_margin: float = 0.0
    # v9.2 adds a privileged-teacher contact guard. A bounded local camera
    # raster must acquire an eligible exact log point before any attack.
    enable_exact_log_rescan: bool = False
    require_raycast_attack_confirmation: bool = False
    exact_log_rescan_budget: int = 40
    exact_log_rescan_max_attempts_per_candidate: int = 1
    exact_log_rescan_loop_budget: int = 3
    # v9.5 prevents a new candidate identity from buying another identical
    # 40-step privileged raster at effectively the same physical location.
    enable_spatial_exact_log_rescan_cooldown: bool = False
    exact_log_rescan_spatial_radius: float = 8.0
    # v9.6 repairs the three remaining failure classes without touching any
    # frozen profile default. A successful exact rescan resets the per-attempt
    # centring loop budgets, slow 3-D approaches receive bounded jump-climb
    # assistance verified by real distance progress, and drop pickup learns
    # the vertical gap between the player and the broken block centre.
    reset_loop_budgets_on_rescan_success: bool = False
    enable_coordinate_climb_assist: bool = False
    coordinate_climb_window: int = 8
    coordinate_climb_minimum_progress: float = 0.5
    coordinate_climb_burst_steps: int = 6
    coordinate_climb_max_bursts_per_target: int = 4
    coordinate_climb_failure_limit: int = 2
    coordinate_climb_success_progress: float = 0.4
    drop_recovery_elevated_pickup: bool = False
    drop_recovery_elevated_vertical_gap: float = 1.2
    drop_recovery_elevated_arrival_radius: float = 1.8
    drop_recovery_elevated_jump_steps: int = 8
    drop_recovery_radius: float = 0.85
    drop_recovery_arrival_radius: float = 0.35
    drop_recovery_yaw_tolerance_degrees: float = 12.0
    drop_recovery_blocked_forward_steps: int = 3
    drop_recovery_minimum_progress: float = 0.04
    drop_recovery_max_steps: int = 20
    enable_enhanced_drop_recovery: bool = False
    drop_recovery_waypoint_max_steps: int = 7
    drop_recovery_centre_max_steps: int = 7
    drop_recovery_jump_budget: int = 1
    drop_recovery_offset_budget: int = 1
    drop_recovery_offset_distance: float = 0.35
    drop_recovery_normalize_block_centre: bool = False
    drop_recovery_ordered_ring: bool = False
    drop_recovery_orient_ring_to_start: bool = False
    drop_recovery_elevated_jump_centre_only: bool = False
    # v9.9 separates reaching the block's horizontal projection from a
    # two-radius ground sweep that can route around an intervening obstacle.
    drop_recovery_ground_sweep: bool = False
    drop_recovery_ground_sweep_outer_radius: float = 1.6
    drop_recovery_contact_extension_steps: int = 0
    same_trunk_reacquire_steps: int = 8
    attack_lost_tolerance: int = 4
    attack_lost_crosshair_fraction: float = 0.10
    attack_area_progress_fraction: float = 0.08
    attack_frame_progress: float = 0.01
    max_attack_rounds: int = 2
    max_backoffs: int = 2
    max_orbits: int = 1
    orbit_turn_steps: int = 9
    orbit_forward_steps: int = 5
    backoff_backward_steps: int = 4
    backoff_turn_steps: int = 2
    region_max_steps: int = 80
    # Discrete action ids (see mc_rl.actions.ACTION_NAMES).
    noop_action: int = 0
    forward_action: int = 1
    forward_jump_action: int = 2
    left_action: int = 3
    right_action: int = 4
    look_up_action: int = 5
    look_down_action: int = 6
    fine_left_action: int = 10
    fine_right_action: int = 11
    fine_look_up_action: int = 12
    fine_look_down_action: int = 13
    # Pure attack is deliberate. Forward+attack belongs to approach, not to
    # the centred contact phase where movement destroys crosshair alignment.
    attack_action: int = 7
    clear_occlusion_action: int = 8
    backward_action: int = 9

    @classmethod
    def for_profile(cls, profile: str, **kwargs):
        """Build an immutable experiment profile without parameter drift."""

        if profile not in CONTACT_PROFILES:
            raise ValueError("unknown trunk contact profile: {}".format(profile))
        requested_profile = profile
        if profile in (
            CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10,
            CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11,
        ):
            # v9.10 inherits the exact v9.9 configuration and changes only
            # the explicitly declared online target-preemption switch.
            profile = CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9
        kwargs["enable_clear_occlusion"] = bool(
            profile
            in (
                CONTACT_PROFILE_CLEAR_OCCLUSION,
                CONTACT_PROFILE_DROP_RECOVERY,
                CONTACT_PROFILE_COORDINATE_AIM,
                CONTACT_PROFILE_COORDINATE_RECOVERY,
                CONTACT_PROFILE_COORDINATE_RECOVERY_V9_1,
                CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2,
                CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
                CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
                CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
                CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
                CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
                CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
                CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
            )
        )
        kwargs["enable_drop_recovery"] = bool(
            profile
            in (
                CONTACT_PROFILE_DROP_RECOVERY,
                CONTACT_PROFILE_COORDINATE_AIM,
                CONTACT_PROFILE_COORDINATE_RECOVERY,
                CONTACT_PROFILE_COORDINATE_RECOVERY_V9_1,
                CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2,
                CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
                CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
                CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
                CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
                CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
                CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
                CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
            )
        )
        kwargs["enable_coordinate_aim"] = bool(
            profile
            in (
                CONTACT_PROFILE_COORDINATE_AIM,
                CONTACT_PROFILE_COORDINATE_RECOVERY,
                CONTACT_PROFILE_COORDINATE_RECOVERY_V9_1,
                CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2,
                CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
                CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
                CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
                CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
                CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
                CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
                CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
            )
        )
        kwargs["enable_coordinate_recovery"] = bool(
            profile
            in (
                CONTACT_PROFILE_COORDINATE_RECOVERY,
                CONTACT_PROFILE_COORDINATE_RECOVERY_V9_1,
                CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2,
                CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
                CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
                CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
                CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
                CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
                CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
                CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
            )
        )
        kwargs["enable_post_recovery_verification"] = bool(
            profile
            in (
                CONTACT_PROFILE_COORDINATE_RECOVERY_V9_1,
                CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2,
                CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
                CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
                CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
                CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
                CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
                CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
                CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
            )
        )
        kwargs["enable_enhanced_drop_recovery"] = bool(
            profile
            in (
                CONTACT_PROFILE_COORDINATE_RECOVERY_V9_1,
                CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2,
                CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
                CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
                CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
                CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
                CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
                CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
                CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
            )
        )
        kwargs["enable_exact_log_rescan"] = bool(
            profile
            in (
                CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2,
                CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
                CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
                CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
                CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
                CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
                CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
                CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
            )
        )
        kwargs["require_raycast_attack_confirmation"] = bool(
            profile
            in (
                CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2,
                CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
                CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
                CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
                CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
                CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
                CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
                CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
            )
        )
        if profile == CONTACT_PROFILE_DROP_RECOVERY:
            kwargs.setdefault("region_max_steps", 100)
        if profile == CONTACT_PROFILE_COORDINATE_AIM:
            kwargs.setdefault("region_max_steps", 120)
        if profile == CONTACT_PROFILE_COORDINATE_RECOVERY:
            kwargs.setdefault("region_max_steps", 140)
        if profile in (
            CONTACT_PROFILE_COORDINATE_RECOVERY_V9_1,
            CONTACT_PROFILE_COORDINATE_CONTACT_GUARD_V9_2,
            CONTACT_PROFILE_CANDIDATE_HANDOFF_GUARD_V9_3,
            CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
            CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
            CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
            CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
            CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
            CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
        ):
            kwargs.setdefault("region_max_steps", 140)
            # Natural one-log utility: exact current reachability dominates;
            # the coarse candidate identity is only a weak association tie-break.
            kwargs.setdefault("coordinate_hint_distance_weight", 0.05)
            kwargs.setdefault("coordinate_horizontal_distance_weight", 1.0)
            kwargs.setdefault("coordinate_vertical_distance_weight", 1.25)
            kwargs.setdefault("coordinate_reach_excess_weight", 2.0)
            kwargs.setdefault("coordinate_failure_weight", 2.0)
            kwargs.setdefault("coordinate_recovery_weight", 3.0)
            kwargs.setdefault("coordinate_observation_weight", 0.10)
            kwargs.setdefault(
                "coordinate_maximum_horizontal_selection_distance", 14.0
            )
            kwargs.setdefault("drop_recovery_max_steps", 72)
            kwargs.setdefault(
                "drop_recovery_centre_max_steps",
                48
                if profile
                in (
                    CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
                    CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
                )
                else 28
                if profile
                in (
                    CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
                    CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
                    CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
                    CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
                    CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
                )
                else 24,
            )
            kwargs.setdefault("drop_recovery_contact_extension_steps", 72)
        if profile in (
            CONTACT_PROFILE_CONTACT_DROP_COMPLETION_V9_4,
            CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
            CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
            CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
            CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
            CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
        ):
            kwargs.setdefault("coordinate_recovery_jump_steps", 8)
            kwargs.setdefault("drop_recovery_waypoint_max_steps", 10)
            kwargs.setdefault("drop_recovery_normalize_block_centre", True)
            kwargs.setdefault("drop_recovery_ordered_ring", True)
        if profile in (
            CONTACT_PROFILE_CONTACT_OWNERSHIP_SPATIAL_GUARD_V9_5,
            CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
            CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
            CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
            CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
        ):
            kwargs.setdefault("enable_spatial_exact_log_rescan_cooldown", True)
            kwargs.setdefault("exact_log_rescan_spatial_radius", 8.0)
        if profile in (
            CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
            CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
            CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
            CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
        ):
            # Trace-derived v9.6 repairs: a successful exact rescan must not
            # be discarded by a stale centring-loop budget, slow uphill 3-D
            # approaches get bounded jump assistance, and drop pickup learns
            # the vertical gap to the broken block centre.
            kwargs.setdefault("reset_loop_budgets_on_rescan_success", True)
            kwargs.setdefault("enable_coordinate_climb_assist", True)
            kwargs.setdefault("drop_recovery_elevated_pickup", True)
        if profile in (
            CONTACT_PROFILE_TRACE_GUIDED_DROP_RECOVERY_V9_7,
            CONTACT_PROFILE_RAYCAST_OWNED_HANDOFF_V9_8,
            CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9,
        ):
            # Failure traces 17407 and 17413 show two independent sources of
            # wasted recovery steps: abandoning a still-distant centre after
            # 28 steps, and treating every ring waypoint as an elevated jump
            # target. Keep the episode boundary frozen and spend those steps
            # on direct translation through the likely pickup region.
            kwargs.setdefault("drop_recovery_yaw_tolerance_degrees", 18.0)
            kwargs.setdefault("drop_recovery_orient_ring_to_start", True)
            kwargs.setdefault("drop_recovery_elevated_jump_centre_only", True)
        if profile == CONTACT_PROFILE_EARLY_ROUTE_RECOVERY_V9_9:
            # A failed first diagonal is remembered on the target through its
            # recovery count; the second attempt mirrors the direction.
            kwargs.setdefault("enable_coordinate_side_detour", True)
            kwargs.setdefault("coordinate_max_recoveries_per_target", 2)
            kwargs.setdefault("drop_recovery_ground_sweep", True)
            kwargs.setdefault("drop_recovery_centre_max_steps", 24)
        if requested_profile in (
            CONTACT_PROFILE_COORDINATE_TARGET_PREEMPTION_V9_10,
            CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11,
        ):
            kwargs.setdefault("enable_coordinate_target_preemption", True)
        if requested_profile == CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11:
            kwargs.setdefault("enable_post_recovery_emergency_preemption", True)
        return cls(**kwargs)


@dataclass
class TrunkContactCounters:
    steps: int = 0
    attack_steps: int = 0
    attack_rounds: int = 0
    backoffs: int = 0
    orbits: int = 0
    trunk_reacquires: int = 0
    occlusion_clears: int = 0
    occlusion_clear_steps: int = 0
    raycast_log_actions: int = 0
    raycast_leaf_actions: int = 0
    raycast_in_range_attack_steps: int = 0
    block_disappearances: int = 0
    drop_recovery_attempts: int = 0
    drop_recovery_steps: int = 0
    drop_waypoints_reached: int = 0
    drop_blocked_waypoints: int = 0
    drop_block_center_normalizations: int = 0
    pickup_after_disappearance: int = 0
    same_trunk_reacquires: int = 0
    coordinate_targets_observed: int = 0
    coordinate_target_selections: int = 0
    coordinate_aim_steps: int = 0
    coordinate_aim_fallbacks: int = 0
    coordinate_leaf_clears: int = 0
    coordinate_attacks: int = 0
    coordinate_progress_stalls: int = 0
    coordinate_recoveries: int = 0
    coordinate_right_detours: int = 0
    coordinate_left_detours: int = 0
    coordinate_post_recovery_verifications: int = 0
    coordinate_post_recovery_progress: int = 0
    coordinate_post_recovery_no_progress: int = 0
    coordinate_target_switches: int = 0
    coordinate_target_preemption_checks: int = 0
    coordinate_target_preemptions: int = 0
    coordinate_target_preemption_hold_rejections: int = 0
    coordinate_target_preemption_support_rejections: int = 0
    coordinate_target_preemption_margin_rejections: int = 0
    coordinate_emergency_preemption_checks: int = 0
    coordinate_emergency_preemptions: int = 0
    coordinate_emergency_preemption_vertical_rejections: int = 0
    coordinate_emergency_preemption_recovery_rejections: int = 0
    coordinate_all_targets_cooldown: int = 0
    coordinate_no_eligible_targets: int = 0
    exact_log_rescan_attempts: int = 0
    exact_log_rescan_steps: int = 0
    exact_log_rescan_successes: int = 0
    exact_log_rescan_failures: int = 0
    spatial_exact_log_rescan_rejections: int = 0
    rescan_success_loop_resets: int = 0
    coordinate_climb_bursts: int = 0
    coordinate_climb_successes: int = 0
    coordinate_climb_failures: int = 0
    coordinate_climb_steps: int = 0
    drop_elevated_pickup_attempts: int = 0
    drop_elevated_jump_steps: int = 0
    drop_elevated_vertical_gap_total: float = 0.0
    prevented_unconfirmed_attacks: int = 0
    external_attack_gate_checks: int = 0
    external_attack_gate_allows: int = 0
    external_attack_gate_rejections: int = 0
    external_attack_gate_recenters: int = 0
    center_adjust_loop_cycles: int = 0
    center_find_loop_cycles: int = 0
    attack_out_of_range_loops: int = 0
    attempts: int = 0
    transitions: Deque[Tuple[int, str, str, str]] = field(
        default_factory=deque
    )

    def as_dict(self) -> Dict[str, float]:
        return {
            "steps": self.steps,
            "attack_steps": self.attack_steps,
            "attack_rounds": self.attack_rounds,
            "backoffs": self.backoffs,
            "orbits": self.orbits,
            "trunk_reacquires": self.trunk_reacquires,
            "occlusion_clears": self.occlusion_clears,
            "occlusion_clear_steps": self.occlusion_clear_steps,
            "raycast_log_actions": self.raycast_log_actions,
            "raycast_leaf_actions": self.raycast_leaf_actions,
            "raycast_in_range_attack_steps": (
                self.raycast_in_range_attack_steps
            ),
            "block_disappearances": self.block_disappearances,
            "drop_recovery_attempts": self.drop_recovery_attempts,
            "drop_recovery_steps": self.drop_recovery_steps,
            "drop_waypoints_reached": self.drop_waypoints_reached,
            "drop_blocked_waypoints": self.drop_blocked_waypoints,
            "drop_block_center_normalizations": (
                self.drop_block_center_normalizations
            ),
            "pickup_after_disappearance": self.pickup_after_disappearance,
            "same_trunk_reacquires": self.same_trunk_reacquires,
            "coordinate_targets_observed": self.coordinate_targets_observed,
            "coordinate_target_selections": self.coordinate_target_selections,
            "coordinate_aim_steps": self.coordinate_aim_steps,
            "coordinate_aim_fallbacks": self.coordinate_aim_fallbacks,
            "coordinate_leaf_clears": self.coordinate_leaf_clears,
            "coordinate_attacks": self.coordinate_attacks,
            "coordinate_progress_stalls": self.coordinate_progress_stalls,
            "coordinate_recoveries": self.coordinate_recoveries,
            "coordinate_right_detours": self.coordinate_right_detours,
            "coordinate_left_detours": self.coordinate_left_detours,
            "coordinate_post_recovery_verifications": (
                self.coordinate_post_recovery_verifications
            ),
            "coordinate_post_recovery_progress": (
                self.coordinate_post_recovery_progress
            ),
            "coordinate_post_recovery_no_progress": (
                self.coordinate_post_recovery_no_progress
            ),
            "coordinate_target_switches": self.coordinate_target_switches,
            "coordinate_target_preemption_checks": (
                self.coordinate_target_preemption_checks
            ),
            "coordinate_target_preemptions": self.coordinate_target_preemptions,
            "coordinate_target_preemption_hold_rejections": (
                self.coordinate_target_preemption_hold_rejections
            ),
            "coordinate_target_preemption_support_rejections": (
                self.coordinate_target_preemption_support_rejections
            ),
            "coordinate_target_preemption_margin_rejections": (
                self.coordinate_target_preemption_margin_rejections
            ),
            "coordinate_emergency_preemption_checks": (
                self.coordinate_emergency_preemption_checks
            ),
            "coordinate_emergency_preemptions": (
                self.coordinate_emergency_preemptions
            ),
            "coordinate_emergency_preemption_vertical_rejections": (
                self.coordinate_emergency_preemption_vertical_rejections
            ),
            "coordinate_emergency_preemption_recovery_rejections": (
                self.coordinate_emergency_preemption_recovery_rejections
            ),
            "coordinate_all_targets_cooldown": self.coordinate_all_targets_cooldown,
            "coordinate_no_eligible_targets": self.coordinate_no_eligible_targets,
            "exact_log_rescan_attempts": self.exact_log_rescan_attempts,
            "exact_log_rescan_steps": self.exact_log_rescan_steps,
            "exact_log_rescan_successes": self.exact_log_rescan_successes,
            "exact_log_rescan_failures": self.exact_log_rescan_failures,
            "spatial_exact_log_rescan_rejections": (
                self.spatial_exact_log_rescan_rejections
            ),
            "rescan_success_loop_resets": self.rescan_success_loop_resets,
            "coordinate_climb_bursts": self.coordinate_climb_bursts,
            "coordinate_climb_successes": self.coordinate_climb_successes,
            "coordinate_climb_failures": self.coordinate_climb_failures,
            "coordinate_climb_steps": self.coordinate_climb_steps,
            "drop_elevated_pickup_attempts": (
                self.drop_elevated_pickup_attempts
            ),
            "drop_elevated_jump_steps": self.drop_elevated_jump_steps,
            "drop_elevated_vertical_gap_total": (
                self.drop_elevated_vertical_gap_total
            ),
            "prevented_unconfirmed_attacks": self.prevented_unconfirmed_attacks,
            "external_attack_gate_checks": self.external_attack_gate_checks,
            "external_attack_gate_allows": self.external_attack_gate_allows,
            "external_attack_gate_rejections": (
                self.external_attack_gate_rejections
            ),
            "external_attack_gate_recenters": self.external_attack_gate_recenters,
            "center_adjust_loop_cycles": self.center_adjust_loop_cycles,
            "center_find_loop_cycles": self.center_find_loop_cycles,
            "attack_out_of_range_loops": self.attack_out_of_range_loops,
            "attempts": self.attempts,
        }


class TrunkContactController:
    """Own the terminal approach once the agent is near the target tree."""

    def __init__(
        self,
        adapter: ResourceAdapter,
        config: Optional[TrunkContactConfig] = None,
    ):
        if not hasattr(adapter, "trunk_view"):
            raise TypeError("adapter must provide trunk_view(pov)")
        self.adapter = adapter
        self.config = config or TrunkContactConfig()
        self.counters = TrunkContactCounters()
        self.state = TrunkContactState.APPROACH_REGION
        self.result: Optional[str] = None
        self.engaged = False
        self._queue: Deque[int] = deque()
        self._scan_steps = 0
        self._scan_direction = 1
        self._misses = 0
        self._burst_steps = 0
        self._burst_reward = 0.0
        self._lost_steps = 0
        self._burst_start_area: Optional[float] = None
        self._burst_start_pov: Optional[np.ndarray] = None
        self._last_view: Optional[TrunkView] = None
        self._attempt_steps = 0
        self._attempt_id = 0
        self._candidate_id: Optional[int] = None
        self._global_step = 0
        self._transition_records: List[Dict[str, Any]] = []
        self._attempt_results: List[Dict[str, Any]] = []
        self._coordinate_selection_records: List[Dict[str, Any]] = []
        self._leaf_occlusion_history: Deque[float] = deque(maxlen=3)
        self._last_leaf_occlusion = 0.0
        self._last_raycast: Optional[RaycastHit] = None
        self._last_log_raycast: Optional[RaycastHit] = None
        self._raycast_log_attack_steps = 0
        self._external_attack_permission: Optional[bool] = None
        self._external_attack_rejection_streak = 0
        self._last_contact_point: Optional[Tuple[float, float, float]] = None
        self._drop_target_point: Optional[Tuple[float, float, float]] = None
        self._last_contact_yaw: Optional[float] = None
        self._drop_planner: Optional[DropRecoveryPlanner] = None
        self._drop_fallback_actions: Deque[int] = deque()
        self._drop_recovery_had_disappearance = False
        self._awaiting_drop_reward = False
        self._same_trunk_steps = 0
        self._drop_recorded_reached = 0
        self._drop_recorded_blocked = 0
        self._drop_recorded_elevated_jumps = 0
        self._drop_waypoint_records: List[Dict[str, object]] = []
        self._target_memory = TrunkTargetMemory(
            merge_distance=self.config.coordinate_target_merge_distance
        )
        self._coordinate_target: Optional[TrunkBlockTarget] = None
        self._coordinate_target_selected_step: Optional[int] = None
        self._coordinate_error: Optional[CoordinateAimError] = None
        self._coordinate_misses = 0
        self._coordinate_target_hint: Optional[Tuple[float, float]] = None
        self._coordinate_progress = CoordinateProgressMonitor(
            window_size=self.config.coordinate_progress_window,
            minimum_progress=self.config.coordinate_minimum_progress,
        )
        self._coordinate_recovery_actions: Deque[int] = deque()
        self._coordinate_post_recovery_samples = 0
        self._coordinate_post_recovery_state_steps = 0
        self._coordinate_post_recovery_initial_distance: Optional[float] = None
        self._coordinate_post_recovery_minimum_distance: Optional[float] = None
        self._coordinate_timeout_extension = 0
        self._drop_timeout_extension = 0
        self._exact_log_rescan_actions: Deque[int] = deque()
        self._exact_log_rescan_reason: Optional[str] = None
        self._exact_log_rescans_by_candidate: Dict[Tuple[str, int], int] = {}
        self._exact_log_rescan_origin: Optional[Tuple[float, float]] = None
        self._failed_exact_log_scan_regions: List[Tuple[float, float]] = []
        self._center_to_find_pending = False
        self._attempt_center_adjust_loops = 0
        self._attempt_center_find_loops = 0
        self._attempt_attack_out_of_range_loops = 0
        # v9.6 climb assist state: a bounded forward-jump burst that is only
        # armed when real target-distance progress stays below the configured
        # minimum, and only kept when each burst verifiably reduces distance.
        self._coordinate_climb_actions: Deque[int] = deque()
        self._coordinate_climb_distances: Deque[float] = deque()
        self._coordinate_climb_bursts = 0
        self._coordinate_climb_failures = 0
        self._coordinate_climb_disabled = False
        self._coordinate_climb_burst_start_distance: Optional[float] = None
        self._coordinate_score_config = TrunkTargetScoreConfig(
            hint_distance_weight=self.config.coordinate_hint_distance_weight,
            horizontal_distance_weight=(
                self.config.coordinate_horizontal_distance_weight
            ),
            vertical_distance_weight=self.config.coordinate_vertical_distance_weight,
            reach_excess_weight=self.config.coordinate_reach_excess_weight,
            failure_weight=self.config.coordinate_failure_weight,
            recovery_weight=self.config.coordinate_recovery_weight,
            observation_weight=self.config.coordinate_observation_weight,
            attack_reach=self.config.coordinate_attack_reach,
            maximum_horizontal_distance=(
                self.config.coordinate_maximum_horizontal_selection_distance
            ),
        )

    @property
    def active(self) -> bool:
        return bool(self.engaged and self.result is None)

    @property
    def candidate_id(self) -> Optional[int]:
        """Candidate whose terminal approach currently owns action selection."""

        return self._candidate_id

    def cancel(self, reason: str) -> None:
        """End an active approach through the ordinary replan audit path."""

        if self.active:
            self._finish("replan", reason)

    def start(self) -> None:
        """Reset the episode-level controller and all accumulated metrics."""

        self.counters = TrunkContactCounters()
        self._attempt_id = 0
        self._candidate_id = None
        self._global_step = 0
        self._transition_records = []
        self._attempt_results = []
        self._coordinate_selection_records = []
        self._drop_waypoint_records = []
        self._exact_log_rescans_by_candidate = {}
        self._failed_exact_log_scan_regions = []
        self._target_memory.reset()
        self.state = TrunkContactState.APPROACH_REGION
        self.result = None
        self.engaged = False
        self._external_attack_permission = None
        self._external_attack_rejection_streak = 0
        self._reset_approach()

    def set_external_attack_permission(
        self, permission: Optional[bool]
    ) -> None:
        """Set one-step visual permission without changing frozen defaults."""

        self._external_attack_permission = (
            None if permission is None else bool(permission)
        )

    def _external_attack_allowed(self) -> bool:
        """Audit and apply an optional state-safe visual attack veto."""

        permission = self._external_attack_permission
        if permission is None:
            return True
        self.counters.external_attack_gate_checks += 1
        if permission:
            self.counters.external_attack_gate_allows += 1
            self._external_attack_rejection_streak = 0
            return True
        self.counters.external_attack_gate_rejections += 1
        self._external_attack_rejection_streak += 1
        limit = max(1, self.config.external_attack_gate_recenter_rejections)
        if self._external_attack_rejection_streak >= limit:
            self._external_attack_rejection_streak = 0
            self.counters.external_attack_gate_recenters += 1
            self._transition(
                TrunkContactState.CENTER_TRUNK,
                "visual attack gate rejected attack; recenter",
            )
        return False

    def engage(
        self,
        telemetry: Optional[AgentTelemetry] = None,
        find_first: bool = False,
        candidate_id: Optional[int] = None,
        global_step: int = 0,
        target_hint: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Arm the controller for one approach into the target region.

        ``find_first`` skips the region walk and starts with the local
        trunk sweep, for engagements caused by a stalled world route.
        """

        self.state = TrunkContactState.APPROACH_REGION
        self.result = None
        self.engaged = False
        self._external_attack_permission = None
        self._external_attack_rejection_streak = 0
        self._reset_approach()
        self._attempt_id += 1
        self.counters.attempts += 1
        self._candidate_id = candidate_id
        self._global_step = int(global_step)
        self.engaged = True
        self._coordinate_target_hint = target_hint
        if self.config.enable_coordinate_aim:
            self._coordinate_target = self._target_memory.select(
                step=self._global_step,
                telemetry=telemetry,
                candidate_hint=target_hint,
                score_config=(
                    self._coordinate_score_config
                    if self.config.enable_coordinate_recovery
                    else None
                ),
            )
            if self._coordinate_target is not None and telemetry is not None:
                self._coordinate_target_selected_step = self._global_step
                self.counters.coordinate_target_selections += 1
                self._record_coordinate_selection(
                    None,
                    self._coordinate_target,
                    "selected remembered 3-D log target",
                    telemetry,
                )
                self._transition(
                    TrunkContactState.COORDINATE_AIM,
                    "selected remembered 3-D log target",
                )
                return
            if (
                self._target_memory.last_selection_reason
                == "no_eligible_targets"
            ):
                self.counters.coordinate_no_eligible_targets += 1
            if self.config.enable_exact_log_rescan:
                self._start_exact_log_rescan(
                    "no eligible exact local log target at contact handoff",
                    telemetry,
                )
                return
        if find_first:
            self._queue_find_trunk_sweep(telemetry)
            self._transition(
                TrunkContactState.FIND_TRUNK,
                "engaged after route stall",
            )

    def _reset_approach(self) -> None:
        self._queue.clear()
        self._scan_steps = 0
        self._scan_direction = 1
        self._misses = 0
        self._burst_steps = 0
        self._burst_reward = 0.0
        self._lost_steps = 0
        self._burst_start_area = None
        self._burst_start_pov = None
        self._pickup_steps = 0
        self._failed_attack_rounds = 0
        self._last_view = None
        self._last_yaw_command = 0
        self._last_pitch_command = 0
        self._reach_miss_steps = 0
        self._center_cycles = 0
        self._attempt_steps = 0
        self._route_distances: Deque[float] = deque(maxlen=24)
        self._leaf_occlusion_history = deque(maxlen=3)
        self._last_leaf_occlusion = 0.0
        self._last_raycast = None
        self._last_log_raycast = None
        self._raycast_log_attack_steps = 0
        self._last_contact_point = None
        self._drop_target_point = None
        self._last_contact_yaw = None
        self._drop_planner = None
        self._drop_fallback_actions = deque()
        self._drop_recovery_had_disappearance = False
        self._awaiting_drop_reward = False
        self._same_trunk_steps = 0
        self._drop_recorded_reached = 0
        self._drop_recorded_blocked = 0
        self._drop_recorded_elevated_jumps = 0
        self._occlusion_clear_steps = 0
        self._attempt_occlusion_clears = 0
        self._coordinate_target = None
        self._coordinate_target_selected_step = None
        self._coordinate_error = None
        self._coordinate_misses = 0
        self._coordinate_target_hint = None
        self._coordinate_progress.reset()
        self._coordinate_recovery_actions = deque()
        self._coordinate_post_recovery_samples = 0
        self._coordinate_post_recovery_state_steps = 0
        self._coordinate_post_recovery_initial_distance = None
        self._coordinate_post_recovery_minimum_distance = None
        self._coordinate_timeout_extension = 0
        self._drop_timeout_extension = 0
        self._exact_log_rescan_actions = deque()
        self._exact_log_rescan_reason = None
        self._exact_log_rescan_origin = None
        self._center_to_find_pending = False
        self._attempt_center_adjust_loops = 0
        self._attempt_center_find_loops = 0
        self._attempt_attack_out_of_range_loops = 0
        self._coordinate_climb_actions = deque()
        self._coordinate_climb_distances = deque()
        self._coordinate_climb_bursts = 0
        self._coordinate_climb_failures = 0
        self._coordinate_climb_disabled = False
        self._coordinate_climb_burst_start_distance = None

    def observe_raycast_target(
        self,
        raycast: Optional[RaycastHit],
        telemetry: Optional[AgentTelemetry],
        global_step: int,
    ) -> Optional[TrunkBlockTarget]:
        """Remember a visible log point for the privileged coordinate teacher."""

        if (
            not self.config.enable_coordinate_aim
            or raycast is None
            or telemetry is None
            or not raycast.is_log
        ):
            return None
        before = len(self._target_memory.targets)
        target = self._target_memory.observe(raycast, int(global_step))
        if len(self._target_memory.targets) > before:
            self.counters.coordinate_targets_observed += 1
        if (
            self.active
            and self.state
            in (
                TrunkContactState.COORDINATE_AIM,
                TrunkContactState.POST_RECOVERY_VERIFY,
            )
            and target is not None
        ):
            self._maybe_preempt_coordinate_target(telemetry, int(global_step))
        # A coarse candidate may reach the contact controller before its
        # exact block point crosses the centre ray. Adopt a later ray hit
        # immediately, so RGB local reacquisition can bootstrap the 3-D loop.
        reacquire_states = (
            TrunkContactState.APPROACH_REGION,
            TrunkContactState.EXACT_LOG_RESCAN,
            TrunkContactState.FIND_TRUNK,
            TrunkContactState.CLEAR_OCCLUSION,
            TrunkContactState.CENTER_TRUNK,
            TrunkContactState.ADJUST_PITCH,
        )
        if self.active and self.state in reacquire_states and target is not None:
            was_exact_rescan = self.state == TrunkContactState.EXACT_LOG_RESCAN
            previous_target = self._coordinate_target
            if self.config.enable_coordinate_recovery:
                selected = self._target_memory.select(
                    step=int(global_step),
                    telemetry=telemetry,
                    candidate_hint=self._coordinate_target_hint,
                    score_config=self._coordinate_score_config,
                )
                if (
                    selected is None
                    and self.config.coordinate_maximum_horizontal_selection_distance
                    is not None
                ):
                    # Keep the scored point in memory, but do not let a lone
                    # far ray hit hijack local RGB contact.
                    self.counters.coordinate_no_eligible_targets += 1
                    return target
                self._coordinate_target = selected or target
            else:
                self._coordinate_target = target
                target.status = "selected"
            self._coordinate_misses = 0
            self._coordinate_progress.reset()
            self._coordinate_target_selected_step = int(global_step)
            self.counters.coordinate_target_selections += 1
            self._record_coordinate_selection(
                previous_target,
                self._coordinate_target,
                "local raycast supplied a 3-D log target",
                telemetry,
            )
            self._transition(
                TrunkContactState.COORDINATE_AIM,
                "local raycast supplied a 3-D log target",
            )
            if was_exact_rescan:
                self.counters.exact_log_rescan_successes += 1
                self._exact_log_rescan_actions.clear()
                if self.config.reset_loop_budgets_on_rescan_success:
                    # A successful raster replaces the centring evidence, so
                    # stale per-attempt loop budgets must not immediately
                    # demand another raster (and then replan because the
                    # candidate already spent its single attempt).
                    self._attempt_center_adjust_loops = 0
                    self._attempt_center_find_loops = 0
                    self._attempt_attack_out_of_range_loops = 0
                    self._center_cycles = 0
                    self._center_to_find_pending = False
                    self.counters.rescan_success_loop_resets += 1
        return target

    def _maybe_preempt_coordinate_target(
        self, telemetry: AgentTelemetry, global_step: int
    ) -> bool:
        """Switch to a substantially better exact log without waiting to stall."""

        if not self.config.enable_coordinate_target_preemption:
            return False
        current = self._coordinate_target
        if current is None:
            return False
        self.counters.coordinate_target_preemption_checks += 1
        emergency = bool(
            self.config.enable_post_recovery_emergency_preemption
            and self.state == TrunkContactState.POST_RECOVERY_VERIFY
        )
        if emergency:
            self.counters.coordinate_emergency_preemption_checks += 1
            if current.recovery_attempts <= 0:
                self.counters.coordinate_emergency_preemption_recovery_rejections += 1
                return False
        held_steps = (
            int(global_step)
            if self._coordinate_target_selected_step is None
            else int(global_step) - self._coordinate_target_selected_step
        )
        if held_steps < self.config.coordinate_target_preemption_min_hold_steps:
            self.counters.coordinate_target_preemption_hold_rejections += 1
            return False

        statuses = {
            item.target_id: item.status for item in self._target_memory.targets
        }
        candidate = self._target_memory.select(
            step=int(global_step),
            telemetry=telemetry,
            candidate_hint=self._coordinate_target_hint,
            score_config=self._coordinate_score_config,
        )
        if candidate is None or candidate is current:
            return False

        accepted = True
        required_observations = (
            self.config.coordinate_emergency_preemption_min_observations
            if emergency
            else self.config.coordinate_target_preemption_min_observations
        )
        if (
            candidate.observation_count
            < required_observations
        ):
            self.counters.coordinate_target_preemption_support_rejections += 1
            accepted = False
        score_gain = float(candidate.score - current.score)
        required_score_gain = (
            self.config.coordinate_emergency_preemption_score_margin
            if emergency
            else self.config.coordinate_target_preemption_score_margin
        )
        if score_gain < required_score_gain:
            self.counters.coordinate_target_preemption_margin_rejections += 1
            accepted = False
        vertical_drop = float(current.y - candidate.y)
        if (
            emergency
            and vertical_drop
            < self.config.coordinate_emergency_preemption_min_vertical_drop
        ):
            self.counters.coordinate_emergency_preemption_vertical_rejections += 1
            accepted = False
        if not accepted:
            for item in self._target_memory.targets:
                item.status = statuses[item.target_id]
            return False

        previous = current
        was_verifying_failed_recovery = bool(
            self.state == TrunkContactState.POST_RECOVERY_VERIFY
        )
        if was_verifying_failed_recovery:
            self._target_memory.mark_failed(
                previous,
                int(global_step),
                self.config.coordinate_target_cooldown_steps,
            )
        self._coordinate_target = candidate
        self._coordinate_target_selected_step = int(global_step)
        self._coordinate_progress.reset()
        self._reset_coordinate_climb_state()
        self._coordinate_misses = 0
        self._coordinate_error = None
        self._coordinate_post_recovery_samples = 0
        self._coordinate_post_recovery_state_steps = 0
        self._coordinate_post_recovery_initial_distance = None
        self._coordinate_post_recovery_minimum_distance = None
        self.counters.coordinate_target_selections += 1
        self.counters.coordinate_target_switches += 1
        self.counters.coordinate_target_preemptions += 1
        if emergency:
            self.counters.coordinate_emergency_preemptions += 1
        selection_reason = (
            "failed recovery yielded to lower exact log "
            "(score gain {:.2f}, vertical drop {:.2f})".format(
                score_gain, vertical_drop
            )
            if emergency
            else "better observed 3-D log preempted current target "
            "(score gain {:.2f})".format(score_gain)
        )
        self._record_coordinate_selection(
            previous,
            candidate,
            selection_reason,
            telemetry,
        )
        if was_verifying_failed_recovery:
            self._transition(
                TrunkContactState.COORDINATE_AIM,
                "better observed 3-D log ended stale recovery verification",
            )
        return True

    def has_reachable_remembered_log_target(
        self,
        telemetry: Optional[AgentTelemetry],
        global_step: int,
    ) -> bool:
        """Return whether exact episode memory has an eligible local log.

        This is a privileged-teacher ownership signal, not a visual estimate.
        Selection applies the same reachability and cooldown rules that
        ``engage`` will immediately reuse.
        """

        if not self.config.enable_coordinate_aim or telemetry is None:
            return False
        target = self._target_memory.select(
            step=int(global_step),
            telemetry=telemetry,
            candidate_hint=None,
            score_config=(
                self._coordinate_score_config
                if self.config.enable_coordinate_recovery
                else None
            ),
        )
        return target is not None

    def nearest_remembered_log_route(
        self,
        telemetry: Optional[AgentTelemetry],
        global_step: int,
    ) -> Optional[Tuple[float, float, float, int]]:
        """Return the nearest exact log route observed by this episode's scan."""

        if not self.config.enable_coordinate_aim or telemetry is None:
            return None
        routes = []
        for target in self._target_memory.targets:
            if target.status == "cooldown" and int(global_step) >= int(
                target.failed_until_step
            ):
                target.status = "available"
            if target.status not in ("available", "selected"):
                continue
            distance = hypot(
                float(target.x) - float(telemetry.x),
                float(target.z) - float(telemetry.z),
            )
            routes.append(
                (
                    distance,
                    -int(target.observation_count),
                    int(target.target_id),
                    target,
                )
            )
        if not routes:
            return None
        distance, _support, target_id, target = min(routes)
        return float(target.x), float(target.z), float(distance), target_id

    def _transition(self, new_state: TrunkContactState, reason: str) -> None:
        if new_state == self.state:
            return
        old_state = self.state
        if (
            old_state == TrunkContactState.ADJUST_PITCH
            and new_state == TrunkContactState.CENTER_TRUNK
        ):
            self.counters.center_adjust_loop_cycles += 1
            self._attempt_center_adjust_loops += 1
        if (
            old_state == TrunkContactState.CENTER_TRUNK
            and new_state == TrunkContactState.FIND_TRUNK
        ):
            self._center_to_find_pending = True
        elif (
            old_state == TrunkContactState.FIND_TRUNK
            and new_state == TrunkContactState.CENTER_TRUNK
            and self._center_to_find_pending
        ):
            self.counters.center_find_loop_cycles += 1
            self._attempt_center_find_loops += 1
            self._center_to_find_pending = False
        if (
            old_state == TrunkContactState.ATTACK_TRUNK
            and new_state == TrunkContactState.CENTER_TRUNK
            and "out of reach" in reason
        ):
            self.counters.attack_out_of_range_loops += 1
            self._attempt_attack_out_of_range_loops += 1
        self.counters.transitions.append(
            (self.counters.steps, self.state.value, new_state.value, reason)
        )
        self._transition_records.append(
            {
                "global_step": self._global_step,
                "contact_step": self.counters.steps,
                "attempt_step": self._attempt_steps,
                "attempt_id": self._attempt_id,
                "candidate_id": (
                    "" if self._candidate_id is None else self._candidate_id
                ),
                "old_state": self.state.value,
                "new_state": new_state.value,
                "reason": reason,
            }
        )
        self.state = new_state

    def _exact_log_rescan_key(self) -> Tuple[str, int]:
        if self._candidate_id is not None:
            return ("candidate", int(self._candidate_id))
        return ("attempt", int(self._attempt_id))

    def _exact_log_rescan_sequence(self) -> Deque[int]:
        """Return a 40-step local raster, restoring its starting camera pose."""

        yaw_raster = (
            [self.config.fine_right_action] * 2
            + [self.config.fine_left_action] * 4
            + [self.config.fine_right_action] * 2
        )
        sequence = (
            # Candidate handoff already supplies a coarse yaw. Scan four
            # vertical bands, including logs high above a lower-terrain player.
            [self.config.look_down_action]
            + yaw_raster  # pitch +10, yaw +/-10
            + [self.config.look_up_action] * 2
            + yaw_raster  # pitch -10
            + [self.config.look_up_action] * 2
            + yaw_raster  # pitch -30
            + [self.config.look_up_action]
            + [self.config.fine_right_action]
            + [self.config.fine_left_action] * 2
            + [self.config.fine_right_action] * 2
            + [self.config.fine_left_action]  # pitch -40, yaw +/-5
            + [self.config.look_down_action] * 4  # restore original pitch
        )
        budget = max(0, int(self.config.exact_log_rescan_budget))
        return deque(sequence[:budget])

    @staticmethod
    def _horizontal_pose(
        telemetry: Optional[AgentTelemetry],
    ) -> Optional[Tuple[float, float]]:
        if telemetry is None:
            return None
        return (float(telemetry.x), float(telemetry.z))

    def _inside_failed_exact_log_scan_region(
        self, pose: Optional[Tuple[float, float]]
    ) -> bool:
        if pose is None:
            return False
        radius = max(0.0, float(self.config.exact_log_rescan_spatial_radius))
        return any(
            hypot(pose[0] - centre[0], pose[1] - centre[1]) <= radius
            for centre in self._failed_exact_log_scan_regions
        )

    def _record_failed_exact_log_scan_region(self) -> None:
        origin = self._exact_log_rescan_origin
        if origin is None or self._inside_failed_exact_log_scan_region(origin):
            return
        self._failed_exact_log_scan_regions.append(origin)

    def _start_exact_log_rescan(
        self,
        reason: str,
        telemetry: Optional[AgentTelemetry] = None,
    ) -> bool:
        """Start at most one bounded privileged local scan per candidate."""

        if not self.config.enable_exact_log_rescan:
            return False
        pose = self._horizontal_pose(telemetry)
        if (
            self.config.enable_spatial_exact_log_rescan_cooldown
            and self._inside_failed_exact_log_scan_region(pose)
        ):
            self.counters.spatial_exact_log_rescan_rejections += 1
            self._finish(
                "replan",
                "exact log rescan blocked by failed spatial region",
            )
            return False
        key = self._exact_log_rescan_key()
        used = self._exact_log_rescans_by_candidate.get(key, 0)
        if used >= self.config.exact_log_rescan_max_attempts_per_candidate:
            self._finish("replan", "exact log rescan already used for candidate")
            return False
        remaining = self.config.episode_max_steps - self._global_step
        if remaining < self.config.exact_log_rescan_budget:
            self._finish("replan", "insufficient episode budget for exact log rescan")
            return False
        contact_remaining = (
            self.config.region_max_steps
            + self._coordinate_timeout_extension
            + self._drop_timeout_extension
            - self._attempt_steps
        )
        if contact_remaining < self.config.exact_log_rescan_budget:
            self._finish("replan", "insufficient contact budget for exact log rescan")
            return False
        self._exact_log_rescans_by_candidate[key] = used + 1
        self._exact_log_rescan_actions = self._exact_log_rescan_sequence()
        self._exact_log_rescan_reason = reason
        self._exact_log_rescan_origin = pose
        self._queue.clear()
        self.counters.exact_log_rescan_attempts += 1
        self._transition(TrunkContactState.EXACT_LOG_RESCAN, reason)
        return True

    def _exact_log_loop_reason(self) -> Optional[str]:
        budget = self.config.exact_log_rescan_loop_budget
        if self._attempt_center_adjust_loops >= budget:
            return "CENTER-ADJUST loop budget exhausted"
        if self._attempt_center_find_loops >= budget:
            return "CENTER-FIND loop budget exhausted"
        if self._attempt_attack_out_of_range_loops >= budget:
            return "ATTACK-CENTER out-of-range loop budget exhausted"
        return None

    @staticmethod
    def _raycast_confirms_attack(raycast: Optional[RaycastHit]) -> bool:
        return bool(
            raycast is not None and raycast.is_log and raycast.in_range
        )

    def _record_coordinate_selection(
        self,
        old_target: Optional[TrunkBlockTarget],
        new_target: Optional[TrunkBlockTarget],
        reason: str,
        telemetry: Optional[AgentTelemetry],
    ) -> None:
        """Record an explainable target choice at the pose where it occurred."""

        self._coordinate_selection_records.append(
            {
                "global_step": self._global_step,
                "contact_step": self.counters.steps,
                "attempt_id": self._attempt_id,
                "candidate_id": "" if self._candidate_id is None else self._candidate_id,
                "old_target_id": "" if old_target is None else old_target.target_id,
                "old_target_score": "" if old_target is None else old_target.score,
                "new_target_id": "" if new_target is None else new_target.target_id,
                "new_target_score": "" if new_target is None else new_target.score,
                "reason": reason,
                "player_x": "" if telemetry is None else telemetry.x,
                "player_y": "" if telemetry is None else telemetry.y,
                "player_z": "" if telemetry is None else telemetry.z,
            }
        )

    def _finish(self, result: str, reason: str) -> None:
        if self._drop_planner is not None and result != "success":
            self._sync_drop_planner_counters()
            self._drop_planner.finish(reason)
            self._drop_waypoint_records.extend(
                self._drop_planner.waypoint_records
            )
            self._drop_planner = None
        if self.config.enable_coordinate_aim:
            if result == "success":
                self._target_memory.mark_completed(self._coordinate_target)
            else:
                self._target_memory.mark_failed(
                    self._coordinate_target,
                    self._global_step,
                    self.config.coordinate_target_cooldown_steps,
                )
        self.result = result
        self._transition(
            TrunkContactState.SUCCESS
            if result == "success"
            else TrunkContactState.REPLAN,
            reason,
        )
        self._attempt_results.append(
            {
                "attempt_id": self._attempt_id,
                "candidate_id": (
                    "" if self._candidate_id is None else self._candidate_id
                ),
                "global_step": self._global_step,
                "attempt_steps": self._attempt_steps,
                "result": result,
                "reason": reason,
            }
        )

    def _select_next_coordinate_target(
        self, telemetry: Optional[AgentTelemetry], reason: str
    ) -> None:
        """Cooldown the stalled point and select another remembered face."""

        previous = self._coordinate_target
        if previous is not None:
            self._target_memory.mark_failed(
                previous,
                self._global_step,
                self.config.coordinate_target_cooldown_steps,
            )
        self._transition(TrunkContactState.COORDINATE_REPLAN, reason)
        self._coordinate_target = self._target_memory.select(
            step=self._global_step,
            telemetry=telemetry,
            candidate_hint=self._coordinate_target_hint,
            score_config=self._coordinate_score_config,
        )
        self._coordinate_progress.reset()
        self._reset_coordinate_climb_state()
        self._coordinate_misses = 0
        if self._coordinate_target is None:
            self._coordinate_target_selected_step = None
            no_eligible = bool(
                self._target_memory.last_selection_reason
                == "no_eligible_targets"
            )
            if no_eligible:
                self.counters.coordinate_no_eligible_targets += 1
            else:
                self.counters.coordinate_all_targets_cooldown += 1
            self._record_coordinate_selection(previous, None, reason, telemetry)
            self._finish(
                "replan",
                (
                    "no remembered 3-D log target is currently reachable"
                    if no_eligible
                    else "all remembered 3-D log targets are cooling down"
                ),
            )
            return
        self.counters.coordinate_target_selections += 1
        self.counters.coordinate_target_switches += 1
        self._coordinate_target_selected_step = self._global_step
        self._record_coordinate_selection(
            previous, self._coordinate_target, reason, telemetry
        )
        self._transition(
            TrunkContactState.COORDINATE_AIM,
            "selected next scored 3-D log target",
        )

    def _start_coordinate_recovery(self) -> int:
        """Run one bounded backoff/offset/jump manoeuvre for this target."""

        target = self._coordinate_target
        if target is None:
            self._finish("replan", "coordinate recovery lost target identity")
            return self.config.noop_action
        target.recovery_attempts += 1
        self.counters.coordinate_recoveries += 1
        self._coordinate_progress.reset()
        if self.config.enable_coordinate_side_detour:
            # recovery_attempts is target-local memory. Odd attempts take the
            # historical right side; an unsuccessful retry mirrors left.
            turn_right = bool(target.recovery_attempts % 2 == 1)
            outward_action = (
                self.config.right_action
                if turn_right
                else self.config.left_action
            )
            restore_action = (
                self.config.left_action
                if turn_right
                else self.config.right_action
            )
            if turn_right:
                self.counters.coordinate_right_detours += 1
            else:
                self.counters.coordinate_left_detours += 1
            self._coordinate_recovery_actions = deque(
                [self.config.backward_action]
                * self.config.coordinate_recovery_backward_steps
                + [outward_action]
                * self.config.coordinate_side_detour_turn_steps
                + [self.config.forward_jump_action]
                * self.config.coordinate_side_detour_translation_steps
                + [restore_action]
                * self.config.coordinate_side_detour_turn_steps
            )
        else:
            self._coordinate_recovery_actions = deque(
                [self.config.backward_action]
                * self.config.coordinate_recovery_backward_steps
                + [self.config.right_action]
                * self.config.coordinate_recovery_turn_steps
                + [self.config.forward_jump_action]
                * self.config.coordinate_recovery_jump_steps
                + [self.config.left_action]
                * self.config.coordinate_recovery_turn_steps
            )
        if self.config.enable_post_recovery_verification:
            grant = len(self._coordinate_recovery_actions) + int(
                self.config.coordinate_post_recovery_max_steps
            )
            self._coordinate_timeout_extension = min(
                self.config.coordinate_recovery_episode_extension_steps,
                self._coordinate_timeout_extension + grant,
            )
        self._transition(
            TrunkContactState.COORDINATE_RECOVER,
            "3-D target distance stalled; bounded obstacle recovery",
        )
        return (
            self._coordinate_recovery_actions.popleft()
            if self._coordinate_recovery_actions
            else self.config.noop_action
        )

    def _start_post_recovery_verification(
        self, telemetry: Optional[AgentTelemetry]
    ) -> None:
        """Start a movement-sample budget independent of contact timeout."""

        distance = None
        if telemetry is not None and self._coordinate_target is not None:
            distance = coordinate_aim_error(
                telemetry,
                self._coordinate_target,
                eye_height=self.config.coordinate_eye_height,
            ).horizontal_distance
        self._coordinate_post_recovery_samples = 0
        self._coordinate_post_recovery_state_steps = 0
        self._coordinate_post_recovery_initial_distance = distance
        self._coordinate_post_recovery_minimum_distance = distance
        self.counters.coordinate_post_recovery_verifications += 1
        self._transition(
            TrunkContactState.POST_RECOVERY_VERIFY,
            "bounded coordinate recovery complete; verify translation progress",
        )

    def _post_recovery_translation_action(
        self,
        action: int,
        error: CoordinateAimError,
        telemetry: Optional[AgentTelemetry],
    ) -> int:
        """Consume one translation sample; camera-only actions never call here."""

        distance = float(error.horizontal_distance)
        if self._coordinate_post_recovery_initial_distance is None:
            self._coordinate_post_recovery_initial_distance = distance
            self._coordinate_post_recovery_minimum_distance = distance
        else:
            current_minimum = self._coordinate_post_recovery_minimum_distance
            self._coordinate_post_recovery_minimum_distance = min(
                distance,
                distance if current_minimum is None else current_minimum,
            )
        self._coordinate_post_recovery_samples += 1
        progress = float(
            self._coordinate_post_recovery_initial_distance
            - self._coordinate_post_recovery_minimum_distance
        )
        if progress >= self.config.coordinate_post_recovery_minimum_progress:
            self.counters.coordinate_post_recovery_progress += 1
            self._coordinate_progress.reset()
            self._transition(
                TrunkContactState.COORDINATE_AIM,
                "post-recovery target distance decreased",
            )
            return action
        if (
            self._coordinate_post_recovery_samples
            >= self.config.coordinate_post_recovery_translation_budget
        ):
            self.counters.coordinate_post_recovery_no_progress += 1
            self._select_next_coordinate_target(
                telemetry,
                "post-recovery translation budget exhausted without progress",
            )
            return self.config.noop_action
        return action

    def _coordinate_climb_translation_action(
        self,
        action: int,
        error: CoordinateAimError,
    ) -> Optional[int]:
        """Substitute bounded forward-jumps when forward progress is slow.

        The burst is armed only after the configured window of plain forward
        translations fails to reduce the 3-D target distance, and each burst
        must itself reduce the distance by ``coordinate_climb_success_progress``
        or it counts as a failure. Repeated failures disable the assist for
        the current target so the ordinary stall/recovery/switch path applies.
        """

        if not self.config.enable_coordinate_climb_assist:
            return None
        if (
            self._coordinate_climb_burst_start_distance is not None
            and not self._coordinate_climb_actions
        ):
            burst_progress = (
                self._coordinate_climb_burst_start_distance
                - float(error.horizontal_distance)
            )
            self._coordinate_climb_burst_start_distance = None
            self._coordinate_climb_distances.clear()
            if burst_progress >= self.config.coordinate_climb_success_progress:
                self.counters.coordinate_climb_successes += 1
                self._coordinate_climb_failures = 0
            else:
                self._coordinate_climb_failures += 1
                self.counters.coordinate_climb_failures += 1
                if (
                    self._coordinate_climb_failures
                    >= self.config.coordinate_climb_failure_limit
                ):
                    self._coordinate_climb_disabled = True
        if self._coordinate_climb_actions:
            self.counters.coordinate_climb_steps += 1
            return self._coordinate_climb_actions.popleft()
        if self._coordinate_climb_disabled:
            return None
        if (
            self._coordinate_climb_bursts
            >= self.config.coordinate_climb_max_bursts_per_target
        ):
            return None
        if action != self.config.forward_action:
            return None
        self._coordinate_climb_distances.append(float(error.horizontal_distance))
        if (
            len(self._coordinate_climb_distances)
            < self.config.coordinate_climb_window
        ):
            return None
        window_progress = (
            self._coordinate_climb_distances[0]
            - min(self._coordinate_climb_distances)
        )
        if window_progress >= self.config.coordinate_climb_minimum_progress:
            return None
        self._coordinate_climb_distances.clear()
        self._coordinate_climb_bursts += 1
        self.counters.coordinate_climb_bursts += 1
        self._coordinate_climb_burst_start_distance = float(
            error.horizontal_distance
        )
        self._coordinate_climb_actions.extend(
            [self.config.forward_jump_action]
            * self.config.coordinate_climb_burst_steps
        )
        self.counters.coordinate_climb_steps += 1
        return self._coordinate_climb_actions.popleft()

    def _reset_coordinate_climb_state(self) -> None:
        self._coordinate_climb_actions.clear()
        self._coordinate_climb_distances.clear()
        self._coordinate_climb_bursts = 0
        self._coordinate_climb_failures = 0
        self._coordinate_climb_disabled = False
        self._coordinate_climb_burst_start_distance = None

    def _coordinate_translation_action(
        self,
        action: int,
        error: CoordinateAimError,
        telemetry: Optional[AgentTelemetry],
    ) -> int:
        """Record intended translation and recover/switch when it stalls."""

        if not self.config.enable_coordinate_recovery:
            return action
        if self.state == TrunkContactState.POST_RECOVERY_VERIFY:
            return self._post_recovery_translation_action(
                action, error, telemetry
            )
        climb_action = self._coordinate_climb_translation_action(action, error)
        if climb_action is not None:
            return climb_action
        self._coordinate_progress.add(error.horizontal_distance)
        if not self._coordinate_progress.is_stalled():
            return action
        self.counters.coordinate_progress_stalls += 1
        target = self._coordinate_target
        if (
            target is not None
            and target.recovery_attempts
            < self.config.coordinate_max_recoveries_per_target
        ):
            if self.config.enable_post_recovery_verification:
                recovery_steps = self._coordinate_recovery_step_budget()
                required_remaining = (
                    recovery_steps
                    + self.config.coordinate_post_recovery_translation_budget
                )
                remaining = self.config.episode_max_steps - self._global_step
                if remaining < required_remaining:
                    self._select_next_coordinate_target(
                        telemetry,
                        "insufficient episode budget for recovery verification",
                    )
                    return self.config.noop_action
            return self._start_coordinate_recovery()
        self._select_next_coordinate_target(
            telemetry, "3-D target stalled after recovery budget"
        )
        return self.config.noop_action

    def _coordinate_recovery_step_budget(self) -> int:
        if self.config.enable_coordinate_side_detour:
            return int(
                self.config.coordinate_recovery_backward_steps
                + 2 * self.config.coordinate_side_detour_turn_steps
                + self.config.coordinate_side_detour_translation_steps
            )
        return int(
            self.config.coordinate_recovery_backward_steps
            + 2 * self.config.coordinate_recovery_turn_steps
            + self.config.coordinate_recovery_jump_steps
        )

    def _trunk_usable(self, view: TrunkView) -> bool:
        return bool(
            view.present
            and view.area_px >= self.config.trunk_present_min_area_px
            and view.height_px >= self.config.trunk_present_min_height_px
        )

    def _within_attack_reach(self, view: TrunkView) -> bool:
        """Reject small dirt-like patches that only satisfy one size cue."""

        return bool(
            self._trunk_usable(view)
            and view.area_px >= self.config.attack_min_area_px
            and view.width_px >= self.config.attack_min_width_px
            and view.height_px >= 12.0
        )

    def _tracked_trunk_view(self, pov: np.ndarray) -> TrunkView:
        """Keep contact on one component instead of switching every frame."""

        if not hasattr(self.adapter, "trunk_views"):
            return self.adapter.trunk_view(pov)
        views = self.adapter.trunk_views(pov)
        if not views:
            return self.adapter.trunk_view(pov)
        previous = self._last_view
        lock_states = (
            TrunkContactState.CENTER_TRUNK,
            TrunkContactState.ADJUST_PITCH,
            TrunkContactState.ATTACK_TRUNK,
            TrunkContactState.COLLECT_DROP,
            TrunkContactState.VERIFY_PROGRESS,
        )
        if previous is None or not previous.present or self.state not in lock_states:
            return views[0]
        nearby = [
            view
            for view in views
            if view.material == previous.material
            and abs(view.center_x - previous.center_x) <= 0.22
        ]
        if not nearby:
            return views[0]
        return min(
            nearby,
            key=lambda view: (
                abs(view.center_x - previous.center_x)
                + 0.5 * abs(view.center_y - previous.center_y)
                + 0.002 * abs(view.area_px - previous.area_px)
            ),
        )

    def _queue_find_trunk_sweep(self, telemetry: Optional[AgentTelemetry]):
        self._scan_steps = 0
        self._queue.clear()
        self._last_yaw_command = 0
        self._last_pitch_command = 0
        pitch = None if telemetry is None else telemetry.pitch
        if pitch is not None and pitch < 5.0:
            # Trunks meet the ground below the horizon; a level camera at
            # region entry often only shows canopy.
            self._queue.append(self.config.look_down_action)

    def _enter_find_trunk(
        self, reason: str, telemetry: Optional[AgentTelemetry], reacquire: bool
    ) -> None:
        if reacquire:
            self.counters.trunk_reacquires += 1
        self._center_cycles = 0
        self._queue_find_trunk_sweep(telemetry)
        self._last_view = None
        self._transition(TrunkContactState.FIND_TRUNK, reason)

    def _leaf_occlusion_fraction(self, pov: np.ndarray) -> float:
        detector = getattr(self.adapter, "leaf_occlusion_fraction", None)
        if detector is None:
            return 0.0
        return float(detector(pov))

    def _can_clear_occlusion(self, view: TrunkView) -> bool:
        required = max(1, self.config.leaf_occlusion_consecutive_frames)
        recent = list(self._leaf_occlusion_history)[-required:]
        return bool(
            self.config.enable_clear_occlusion
            and not self._trunk_usable(view)
            and self._attempt_occlusion_clears
            < self.config.max_occlusion_clears
            and len(recent) == required
            and min(recent) >= self.config.leaf_occlusion_fraction
        )

    def _start_occlusion_clear(self, reason: str) -> None:
        self._queue.clear()
        self._occlusion_clear_steps = 0
        self._attempt_occlusion_clears += 1
        self.counters.occlusion_clears += 1
        self._transition(TrunkContactState.CLEAR_OCCLUSION, reason)

    def _remember_contact_point(
        self,
        telemetry: Optional[AgentTelemetry],
        raycast: Optional[RaycastHit],
    ) -> None:
        if raycast is not None and raycast.is_log:
            self._last_log_raycast = raycast
            self._last_contact_point = (raycast.x, raycast.y, raycast.z)
        elif telemetry is not None and self._last_contact_point is None:
            # POV/F3 fallback: an attack-ready trunk is close, so project a
            # conservative point 1.5 blocks along the current view heading.
            angle = np.radians(float(telemetry.yaw))
            self._last_contact_point = (
                telemetry.x - 1.5 * float(np.sin(angle)),
                telemetry.y,
                telemetry.z + 1.5 * float(np.cos(angle)),
            )
        if telemetry is not None:
            self._last_contact_yaw = float(telemetry.yaw)

    def _start_attack_burst(
        self,
        view: TrunkView,
        pov: np.ndarray,
        telemetry: Optional[AgentTelemetry] = None,
        raycast: Optional[RaycastHit] = None,
    ) -> None:
        self._remember_contact_point(telemetry, raycast)
        self._burst_steps = 0
        self._burst_reward = 0.0
        self._lost_steps = 0
        self._reach_miss_steps = 0
        self._center_cycles = 0
        self._last_yaw_command = 0
        self._last_pitch_command = 0
        self._burst_start_area = float(view.area_px)
        self._burst_start_pov = np.asarray(pov, dtype=np.float32).copy()
        self._pickup_steps = self.config.pickup_probe_steps
        self._raycast_log_attack_steps = 0
        self._transition(TrunkContactState.ATTACK_TRUNK, "trunk centred")

    def _drop_recovery_config(self) -> DropRecoveryConfig:
        return DropRecoveryConfig(
            radius=self.config.drop_recovery_radius,
            arrival_radius=self.config.drop_recovery_arrival_radius,
            yaw_tolerance_degrees=(
                self.config.drop_recovery_yaw_tolerance_degrees
            ),
            blocked_forward_steps=(
                self.config.drop_recovery_blocked_forward_steps
            ),
            minimum_progress=self.config.drop_recovery_minimum_progress,
            max_steps=self.config.drop_recovery_max_steps,
            enable_obstacle_recovery=self.config.enable_enhanced_drop_recovery,
            waypoint_max_steps=self.config.drop_recovery_waypoint_max_steps,
            centre_waypoint_max_steps=self.config.drop_recovery_centre_max_steps,
            jump_budget=self.config.drop_recovery_jump_budget,
            offset_budget=self.config.drop_recovery_offset_budget,
            offset_distance=self.config.drop_recovery_offset_distance,
            ordered_ring=self.config.drop_recovery_ordered_ring,
            orient_ring_to_start=(
                self.config.drop_recovery_orient_ring_to_start
            ),
            elevated_vertical_gap=(
                self.config.drop_recovery_elevated_vertical_gap
                if self.config.drop_recovery_elevated_pickup
                else None
            ),
            elevated_arrival_radius=(
                self.config.drop_recovery_elevated_arrival_radius
                if self.config.drop_recovery_elevated_pickup
                else None
            ),
            elevated_jump_steps=(
                self.config.drop_recovery_elevated_jump_steps
                if self.config.drop_recovery_elevated_pickup
                else None
            ),
            elevated_jump_centre_only=(
                self.config.drop_recovery_elevated_jump_centre_only
            ),
            ground_sweep=self.config.drop_recovery_ground_sweep,
            ground_sweep_outer_radius=(
                self.config.drop_recovery_ground_sweep_outer_radius
            ),
        )

    @staticmethod
    def _block_centre_from_hit(
        point: Tuple[float, float, float],
        telemetry: Optional[AgentTelemetry],
    ) -> Tuple[float, float, float]:
        """Convert a ray hit-face coordinate into its containing block centre.

        A tiny nudge from the eye through the hit resolves exact integer block
        boundaries, including negative Minecraft coordinates.
        """

        x, y, z = (float(value) for value in point)
        if telemetry is not None:
            dx = x - float(telemetry.x)
            dz = z - float(telemetry.z)
            length = max(float(np.hypot(dx, dz)), 1e-6)
            x += 1e-3 * dx / length
            z += 1e-3 * dz / length
        return floor(x) + 0.5, floor(y) + 0.5, floor(z) + 0.5

    def _start_drop_recovery(
        self,
        reason: str,
        telemetry: Optional[AgentTelemetry],
        block_disappeared: bool,
    ) -> None:
        self._drop_planner = None
        self._drop_fallback_actions.clear()
        self._drop_recovery_had_disappearance = bool(block_disappeared)
        self._awaiting_drop_reward = True
        self._same_trunk_steps = 0
        self._drop_recorded_reached = 0
        self._drop_recorded_blocked = 0
        self._drop_recorded_elevated_jumps = 0
        self._drop_target_point = self._last_contact_point
        self.counters.drop_recovery_attempts += 1
        if self.config.enable_enhanced_drop_recovery:
            self._drop_timeout_extension = max(
                self._drop_timeout_extension,
                self.config.drop_recovery_contact_extension_steps,
            )
        if block_disappeared:
            self.counters.block_disappearances += 1
            if (
                self.config.drop_recovery_normalize_block_centre
                and self._last_contact_point is not None
            ):
                self._drop_target_point = self._block_centre_from_hit(
                    self._last_contact_point, telemetry
                )
                self.counters.drop_block_center_normalizations += 1
            self._transition(TrunkContactState.BLOCK_DISAPPEARED, reason)
        if telemetry is not None and self._drop_target_point is not None:
            self._drop_planner = DropRecoveryPlanner(
                self._drop_target_point[0],
                self._drop_target_point[2],
                self._drop_recovery_config(),
                centre_y=(
                    self._drop_target_point[1]
                    if self.config.drop_recovery_elevated_pickup
                    else None
                ),
            )
            self._drop_planner.orient_ring_to_start(
                telemetry.x, telemetry.z
            )
            if (
                self.config.drop_recovery_elevated_pickup
                and self._drop_target_point is not None
            ):
                gap = float(self._drop_target_point[1]) - float(telemetry.y)
                if gap > self.config.drop_recovery_elevated_vertical_gap:
                    self.counters.drop_elevated_pickup_attempts += 1
                    self.counters.drop_elevated_vertical_gap_total += gap
        else:
            # POV-only fallback keeps the same bounded semantics with dead
            # reckoning when no self coordinate is available.
            self._drop_fallback_actions.extend(
                [self.config.forward_action] * 6
                + [self.config.left_action, self.config.forward_action] * 2
                + [self.config.right_action] * 2
                + [self.config.forward_action] * 2
                + [self.config.right_action] * 2
                + [self.config.forward_action] * 2
            )
        self._transition(TrunkContactState.DROP_RECOVERY, reason)

    def _sync_drop_planner_counters(self) -> None:
        if self._drop_planner is None:
            return
        reached = self._drop_planner.reached_waypoints
        blocked = self._drop_planner.blocked_waypoints
        self.counters.drop_waypoints_reached += max(
            0, reached - self._drop_recorded_reached
        )
        self.counters.drop_blocked_waypoints += max(
            0, blocked - self._drop_recorded_blocked
        )
        self.counters.drop_elevated_jump_steps += max(
            0,
            self._drop_planner.elevated_jump_steps
            - self._drop_recorded_elevated_jumps,
        )
        self._drop_recorded_reached = reached
        self._drop_recorded_blocked = blocked
        self._drop_recorded_elevated_jumps = (
            self._drop_planner.elevated_jump_steps
        )

    def _finish_drop_recovery_without_reward(
        self, telemetry: Optional[AgentTelemetry]
    ) -> None:
        self._sync_drop_planner_counters()
        if self._drop_planner is not None:
            self._drop_planner.finish("search_exhausted_without_reward")
            self._drop_waypoint_records.extend(
                self._drop_planner.waypoint_records
            )
        self._drop_planner = None
        self._drop_fallback_actions.clear()
        self.counters.attack_rounds += 1
        self._failed_attack_rounds += 1
        if self._failed_attack_rounds < self.config.max_attack_rounds:
            self.counters.same_trunk_reacquires += 1
            self._same_trunk_steps = 0
            self._transition(
                TrunkContactState.REACQUIRE_SAME_TRUNK,
                "drop search exhausted; reacquire same trunk",
            )
        else:
            self._finish(
                "replan",
                "same trunk retry exhausted without a log",
            )

    def _escape_centring_cycle(self) -> None:
        """Recover from a CENTER<->ADJUST limit cycle by moving instead."""

        self._center_cycles = 0
        self._last_yaw_command = 0
        self._last_pitch_command = 0
        if self.counters.orbits < self.config.max_orbits:
            self._queue.extend(
                [self.config.right_action] * self.config.orbit_turn_steps
                + [self.config.forward_action] * self.config.orbit_forward_steps
            )
            self.counters.orbits += 1
            self._transition(
                TrunkContactState.ORBIT_REACQUIRE,
                "yaw centring cycle; orbiting tree",
            )
        else:
            self._finish("replan", "yaw centring cycle exhausted")

    def _burst_progress(self, view: TrunkView, pov: np.ndarray) -> bool:
        # Area and whole-frame changes were false progress in v5: walking,
        # leaves, and camera motion reset the retry counter without a log.
        # Reward is the only deployment-safe evidence that chopping completed.
        return bool(self._burst_reward > 0)

    def act(
        self,
        pov: np.ndarray,
        telemetry: Optional[AgentTelemetry] = None,
        world_route: Optional[Tuple[float, float]] = None,
        global_step: Optional[int] = None,
        raycast: Optional[RaycastHit] = None,
    ) -> int:
        """Choose one discrete action from the current POV and self pose."""

        if not self.active:
            return self.config.noop_action
        config = self.config
        if global_step is not None:
            self._global_step = int(global_step)
        self.counters.steps += 1
        self._attempt_steps += 1
        if self._attempt_steps > (
            config.region_max_steps
            + self._coordinate_timeout_extension
            + self._drop_timeout_extension
        ):
            self._finish("replan", "trunk contact step budget exhausted")
            return config.noop_action
        if self.state == TrunkContactState.POST_RECOVERY_VERIFY:
            self._coordinate_post_recovery_state_steps += 1
            if (
                self._coordinate_post_recovery_state_steps
                > config.coordinate_post_recovery_max_steps
            ):
                self.counters.coordinate_post_recovery_no_progress += 1
                self._select_next_coordinate_target(
                    telemetry,
                    "post-recovery safety step budget exhausted without progress",
                )
                return config.noop_action

        view = self._tracked_trunk_view(pov)
        self._last_view = view
        self._last_leaf_occlusion = self._leaf_occlusion_fraction(pov)
        self._leaf_occlusion_history.append(self._last_leaf_occlusion)
        self._last_raycast = raycast
        if (
            raycast is not None
            and raycast.is_log
            and self.state
            not in (
                TrunkContactState.BLOCK_DISAPPEARED,
                TrunkContactState.DROP_RECOVERY,
                TrunkContactState.REACQUIRE_SAME_TRUNK,
            )
        ):
            self._remember_contact_point(telemetry, raycast)

        if self.state == TrunkContactState.EXACT_LOG_RESCAN:
            if self._exact_log_rescan_actions:
                self.counters.exact_log_rescan_steps += 1
                return self._exact_log_rescan_actions.popleft()
            self.counters.exact_log_rescan_failures += 1
            if self.config.enable_spatial_exact_log_rescan_cooldown:
                self._record_failed_exact_log_scan_region()
            self._finish(
                "replan", "bounded exact log rescan found no eligible local log"
            )
            return config.noop_action

        if config.enable_exact_log_rescan:
            loop_reason = self._exact_log_loop_reason()
            if loop_reason is not None:
                self._start_exact_log_rescan(loop_reason, telemetry)
                return config.noop_action

        if (
            config.enable_coordinate_recovery
            and self.state == TrunkContactState.COORDINATE_RECOVER
        ):
            if self._coordinate_recovery_actions:
                return self._coordinate_recovery_actions.popleft()
            self._coordinate_progress.reset()
            if config.enable_post_recovery_verification:
                self._start_post_recovery_verification(telemetry)
            else:
                self._transition(
                    TrunkContactState.COORDINATE_AIM,
                    "bounded coordinate obstacle recovery complete",
                )

        if (
            config.enable_coordinate_aim
            and self.state
            in (
                TrunkContactState.COORDINATE_AIM,
                TrunkContactState.POST_RECOVERY_VERIFY,
            )
        ):
            self.counters.coordinate_aim_steps += 1
            if telemetry is None or self._coordinate_target is None:
                self.counters.coordinate_aim_fallbacks += 1
                self._enter_find_trunk(
                    "3-D target or self pose unavailable", telemetry, reacquire=True
                )
            else:
                error = coordinate_aim_error(
                    telemetry,
                    self._coordinate_target,
                    eye_height=config.coordinate_eye_height,
                )
                self._coordinate_error = error
                if abs(error.yaw_error) > config.yaw_deadband_degrees:
                    coarse = (
                        abs(error.yaw_error)
                        > config.coordinate_coarse_threshold_degrees
                    )
                    if error.yaw_error > 0:
                        return (
                            config.right_action if coarse else config.fine_right_action
                        )
                    return config.left_action if coarse else config.fine_left_action
                if abs(error.pitch_error) > config.pitch_deadband_degrees:
                    coarse = (
                        abs(error.pitch_error)
                        > config.coordinate_coarse_threshold_degrees
                    )
                    if error.pitch_error > 0:
                        return (
                            config.look_down_action
                            if coarse
                            else config.fine_look_down_action
                        )
                    return (
                        config.look_up_action
                        if coarse
                        else config.fine_look_up_action
                    )

                if raycast is not None and raycast.is_log:
                    self._coordinate_misses = 0
                    self.counters.raycast_log_actions += 1
                    if raycast.in_range:
                        if not self._external_attack_allowed():
                            return config.noop_action
                        self.counters.coordinate_attacks += 1
                        self._start_attack_burst(
                            view, pov, telemetry, raycast
                        )
                    else:
                        return self._coordinate_translation_action(
                            config.forward_action, error, telemetry
                        )
                elif raycast is not None and raycast.is_leaves:
                    self._coordinate_misses = 0
                    self.counters.raycast_leaf_actions += 1
                    self.counters.coordinate_leaf_clears += 1
                    return self._coordinate_translation_action(
                        config.clear_occlusion_action, error, telemetry
                    )
                else:
                    self._coordinate_misses += 1
                    if (
                        error.horizontal_distance
                        > config.coordinate_forward_stop_distance
                    ):
                        return self._coordinate_translation_action(
                            config.forward_action, error, telemetry
                        )
                    if self._coordinate_misses >= config.coordinate_miss_budget:
                        self.counters.coordinate_aim_fallbacks += 1
                        self._enter_find_trunk(
                            "3-D aim not confirmed by raycast",
                            telemetry,
                            reacquire=True,
                        )
                    else:
                        return config.noop_action

        # Privileged diagnostic upper bound: the ray only identifies the
        # block already under the crosshair. Candidate search and coarse
        # navigation remain unchanged. This path is unavailable to POV/F3.
        raycast_aim_states = (
            TrunkContactState.APPROACH_REGION,
            TrunkContactState.FIND_TRUNK,
            TrunkContactState.CLEAR_OCCLUSION,
            TrunkContactState.CENTER_TRUNK,
            TrunkContactState.ADJUST_PITCH,
        )
        if (
            not config.enable_coordinate_aim
            and
            raycast is not None
            and raycast.is_log
            and self.state in raycast_aim_states
        ):
            self.counters.raycast_log_actions += 1
            if raycast.in_range:
                if not self._external_attack_allowed():
                    return config.noop_action
                self._start_attack_burst(view, pov, telemetry, raycast)
            else:
                return config.forward_action
        elif (
            not config.enable_coordinate_aim
            and
            raycast is not None
            and raycast.is_leaves
            and self.state
            in (
                TrunkContactState.APPROACH_REGION,
                TrunkContactState.FIND_TRUNK,
                TrunkContactState.CLEAR_OCCLUSION,
            )
        ):
            self.counters.raycast_leaf_actions += 1
            return config.clear_occlusion_action

        if self.state == TrunkContactState.APPROACH_REGION:
            if self._trunk_usable(view) and view.area_px >= 40.0:
                self._transition(
                    TrunkContactState.CENTER_TRUNK, "trunk visible in region"
                )
            elif self._can_clear_occlusion(view):
                self._start_occlusion_clear(
                    "persistent canopy blocks region approach"
                )
            elif world_route is not None:
                yaw_error, distance = world_route
                self._route_distances.append(float(distance))
                if distance <= config.region_stop_radius:
                    self._enter_find_trunk(
                        "region coordinate reached", telemetry, reacquire=False
                    )
                elif (
                    len(self._route_distances) == self._route_distances.maxlen
                    and min(list(self._route_distances)[:12])
                    - min(list(self._route_distances)[12:])
                    < 0.5
                ):
                    # The remembered coordinate is not getting closer;
                    # search visually instead of walking into the drift.
                    self._enter_find_trunk(
                        "route distance stalled near estimate",
                        telemetry,
                        reacquire=True,
                    )
                elif abs(yaw_error) > config.region_steer_threshold_degrees:
                    return (
                        config.right_action
                        if yaw_error > 0
                        else config.left_action
                    )
                else:
                    return config.forward_action
            else:
                self._enter_find_trunk(
                    "no usable route inside region",
                    telemetry,
                    reacquire=False,
                )

        if self.state == TrunkContactState.FIND_TRUNK:
            if self._trunk_usable(view):
                self._transition(
                    TrunkContactState.CENTER_TRUNK, "trunk found during sweep"
                )
            elif self._can_clear_occlusion(view):
                self._start_occlusion_clear(
                    "persistent canopy blocks trunk sweep"
                )
            else:
                if not self._queue:
                    if self._scan_steps >= config.find_trunk_budget:
                        if self.counters.orbits < config.max_orbits:
                            self._queue.extend(
                                [config.right_action] * config.orbit_turn_steps
                                + [config.forward_action]
                                * config.orbit_forward_steps
                            )
                            self.counters.orbits += 1
                            self._scan_steps = 0
                            self._transition(
                                TrunkContactState.ORBIT_REACQUIRE,
                                "trunk not found; orbiting tree",
                            )
                        else:
                            self._finish(
                                "replan",
                                "trunk not found after sweeps and orbits",
                            )
                            return config.noop_action
                    else:
                        self._queue.extend(
                            [config.right_action
                             if self._scan_direction > 0
                             else config.left_action]
                            * config.find_trunk_sweep_actions
                        )
                        self._scan_direction *= -1
                if self._queue:
                    self._scan_steps += 1
                    return self._queue.popleft()

        if self.state == TrunkContactState.CLEAR_OCCLUSION:
            if self._trunk_usable(view):
                self._transition(
                    TrunkContactState.CENTER_TRUNK,
                    "trunk exposed while clearing canopy",
                )
            elif (
                self._occlusion_clear_steps >= config.occlusion_clear_steps
                or (
                    self._occlusion_clear_steps >= 3
                    and self._last_leaf_occlusion
                    < 0.65 * config.leaf_occlusion_fraction
                )
            ):
                self._enter_find_trunk(
                    "bounded canopy clear complete",
                    telemetry,
                    reacquire=True,
                )
            else:
                self._occlusion_clear_steps += 1
                self.counters.occlusion_clear_steps += 1
                return config.clear_occlusion_action

        if self.state == TrunkContactState.ORBIT_REACQUIRE:
            if self._queue:
                return self._queue.popleft()
            self._enter_find_trunk(
                "orbit step complete", telemetry, reacquire=False
            )

        if self.state == TrunkContactState.BACKOFF:
            if self._queue:
                return self._queue.popleft()
            self._enter_find_trunk(
                "backoff complete", telemetry, reacquire=True
            )

        if self.state == TrunkContactState.DROP_RECOVERY:
            action = None
            if self._drop_planner is not None and telemetry is not None:
                action = self._drop_planner.next_action(
                    telemetry,
                    config.forward_action,
                    config.left_action,
                    config.right_action,
                    config.noop_action,
                    config.forward_jump_action,
                )
                self._sync_drop_planner_counters()
            elif self._drop_fallback_actions:
                action = self._drop_fallback_actions.popleft()
            if action is not None:
                self.counters.drop_recovery_steps += 1
                return action
            self._finish_drop_recovery_without_reward(telemetry)
            if self.result is not None:
                return config.noop_action

        if self.state == TrunkContactState.REACQUIRE_SAME_TRUNK:
            if (
                telemetry is None
                or self._same_trunk_steps >= config.same_trunk_reacquire_steps
            ):
                self._enter_find_trunk(
                    "same trunk bearing reacquired", telemetry, reacquire=True
                )
            else:
                target_yaw = self._last_contact_yaw
                if self._last_contact_point is not None:
                    bearing, distance = bearing_and_distance_to(
                        telemetry,
                        self._last_contact_point[0],
                        self._last_contact_point[2],
                    )
                    if distance > config.drop_recovery_arrival_radius:
                        target_yaw = bearing
                if target_yaw is None:
                    self._enter_find_trunk(
                        "same trunk direction unavailable",
                        telemetry,
                        reacquire=True,
                    )
                else:
                    yaw_error = wrap_degrees(target_yaw - telemetry.yaw)
                    if abs(yaw_error) <= config.yaw_deadband_degrees + 4.0:
                        self._enter_find_trunk(
                            "same trunk bearing reacquired",
                            telemetry,
                            reacquire=True,
                        )
                    else:
                        self._same_trunk_steps += 1
                        return (
                            config.fine_right_action
                            if yaw_error > 0
                            else config.fine_left_action
                        )

        if self.state == TrunkContactState.CENTER_TRUNK:
            if not self._trunk_usable(view):
                self._misses += 1
                if self._misses >= 3:
                    self._misses = 0
                    self._enter_find_trunk(
                        "trunk lost while centring",
                        telemetry,
                        reacquire=True,
                    )
                return config.noop_action
            self._misses = 0
            yaw_error = view.horizontal_yaw
            if abs(yaw_error) > config.yaw_deadband_degrees:
                direction = 1 if yaw_error > 0 else -1
                # Camera commands are 10-degree quanta: if the required
                # direction flipped since the last command, the target sits
                # between the two quantized headings and is centred enough.
                if direction == -self._last_yaw_command:
                    self._last_yaw_command = 0
                    self._transition(
                        TrunkContactState.ADJUST_PITCH,
                        "yaw bracketed by adjacent commands",
                    )
                else:
                    self._last_yaw_command = direction
                    return (
                        config.fine_right_action
                        if direction > 0
                        else config.fine_left_action
                    )
            self._last_yaw_command = 0
            self._transition(
                TrunkContactState.ADJUST_PITCH, "yaw aligned to trunk"
            )

        if self.state == TrunkContactState.ADJUST_PITCH:
            if not self._trunk_usable(view):
                self._transition(
                    TrunkContactState.CENTER_TRUNK, "trunk lost during pitch"
                )
                return config.noop_action
            yaw_error = view.horizontal_yaw
            if abs(yaw_error) > config.yaw_deadband_degrees + 4.0:
                self._center_cycles += 1
                if self._center_cycles >= config.max_center_cycles:
                    self._escape_centring_cycle()
                    return config.noop_action
                self._transition(
                    TrunkContactState.CENTER_TRUNK, "yaw drifted during pitch"
                )
                return config.noop_action
            pitch_error = view.vertical_offset_deg
            look_down = pitch_error > 0
            direction = 1 if look_down else -1
            pitch = None if telemetry is None else telemetry.pitch
            blocked = False
            if pitch is not None:
                if look_down and pitch >= config.pitch_max_degrees:
                    blocked = True
                if (not look_down) and pitch <= config.pitch_min_degrees:
                    blocked = True
            bracketed = bool(
                direction == -self._last_pitch_command
                and self._last_pitch_command != 0
            )
            if (
                blocked
                or bracketed
                or abs(pitch_error) <= config.pitch_deadband_degrees
            ):
                # At a pitch bound, between two quantized pitches, or inside
                # the deadband: aim is settled. Attack only when the trunk is
                # close enough to reach; otherwise keep closing distance.
                if self._within_attack_reach(view):
                    if (
                        config.require_raycast_attack_confirmation
                        and not self._raycast_confirms_attack(raycast)
                    ):
                        self.counters.prevented_unconfirmed_attacks += 1
                        self._transition(
                            TrunkContactState.CENTER_TRUNK,
                            "teacher attack not confirmed by in-range log raycast",
                        )
                        return config.noop_action
                    if not self._external_attack_allowed():
                        return config.noop_action
                    self._start_attack_burst(view, pov, telemetry, raycast)
                else:
                    return config.forward_action
            else:
                self._last_pitch_command = direction
                return (
                    config.fine_look_down_action
                    if look_down
                    else config.fine_look_up_action
                )

        if self.state == TrunkContactState.ATTACK_TRUNK:
            raycast_confirms_attack = self._raycast_confirms_attack(raycast)
            raycast_confirms_disappearance = bool(
                raycast is not None
                and not raycast.is_log
                and self._raycast_log_attack_steps >= 5
            )
            if (
                config.require_raycast_attack_confirmation
                and not raycast_confirms_attack
                and not raycast_confirms_disappearance
            ):
                self.counters.prevented_unconfirmed_attacks += 1
                self._transition(
                    TrunkContactState.CENTER_TRUNK,
                    "target out of reach; teacher attack confirmation lost",
                )
                return config.noop_action
            # The visual student is a permission veto, not an attack source.
            # Consult it only after the frozen teacher has independently
            # established a real attack opportunity. Otherwise ordinary
            # raycast-loss HOLD frames would incorrectly accumulate external
            # rejection streaks and trigger student-caused recentering.
            if not self._external_attack_allowed():
                return config.noop_action
            self.counters.attack_steps += 1
            self._burst_steps += 1
            if raycast is not None and raycast.is_log and raycast.in_range:
                self._raycast_log_attack_steps += 1
                self.counters.raycast_in_range_attack_steps += 1
                self._reach_miss_steps = 0
                self._lost_steps = 0
                self.counters.raycast_log_actions += 1
            elif (
                raycast is not None
                and self._raycast_log_attack_steps >= 5
            ):
                # With a fixed camera and pure attack, an in-range log that
                # disappears after sustained hits was almost certainly
                # broken. Reward arrives only after the dropped item is
                # collected, so probe forward instead of treating this as a
                # centring failure.
                if config.enable_drop_recovery:
                    self._start_drop_recovery(
                        "raycast log disappeared after sustained attack",
                        telemetry,
                        block_disappeared=True,
                    )
                    if (
                        config.enable_enhanced_drop_recovery
                        and self._drop_planner is not None
                        and telemetry is not None
                    ):
                        action = self._drop_planner.next_action(
                            telemetry,
                            config.forward_action,
                            config.left_action,
                            config.right_action,
                            config.noop_action,
                            config.forward_jump_action,
                        )
                        self._sync_drop_planner_counters()
                        self.counters.drop_recovery_steps += 1
                        return (
                            config.noop_action if action is None else action
                        )
                    # The next call routes from the new player pose; take one
                    # immediate step toward the remembered block meanwhile.
                    self.counters.drop_recovery_steps += 1
                    return config.forward_action
                self._pickup_steps = max(0, config.pickup_probe_steps - 1)
                self._transition(
                    TrunkContactState.COLLECT_DROP,
                    "raycast log disappeared after sustained attack",
                )
                return config.forward_action
            elif raycast is not None:
                self._reach_miss_steps += 1
                self._lost_steps += 1
            elif view.present and not self._within_attack_reach(view):
                # The centred trunk became a thin distant strip (view
                # switched components mid-burst): the attack cannot reach
                # it, so stop chopping and close distance again.
                self._reach_miss_steps += 1
            else:
                self._reach_miss_steps = 0
            if self._reach_miss_steps >= 3:
                self._reach_miss_steps = 0
                self._transition(
                    TrunkContactState.CENTER_TRUNK, "target out of reach"
                )
                return config.noop_action
            if raycast is None and (
                view.present
                and view.crosshair_trunk_fraction
                < config.attack_lost_crosshair_fraction
            ):
                self._lost_steps += 1
            else:
                self._lost_steps = 0
            if self._lost_steps >= config.attack_lost_tolerance:
                self._transition(
                    TrunkContactState.CENTER_TRUNK,
                    "crosshair lost trunk contact",
                )
                return config.noop_action
            if self._burst_steps >= config.attack_burst_steps:
                if config.enable_drop_recovery:
                    self._start_drop_recovery(
                        "attack burst complete; search expected drop",
                        telemetry,
                        block_disappeared=False,
                    )
                    return config.attack_action
                self._pickup_steps = config.pickup_probe_steps
                self._transition(
                    TrunkContactState.COLLECT_DROP, "attack burst complete"
                )
                return config.attack_action
            return config.attack_action

        if self.state == TrunkContactState.COLLECT_DROP:
            if self._pickup_steps > 0:
                self._pickup_steps -= 1
                return config.forward_action
            self._transition(
                TrunkContactState.VERIFY_PROGRESS, "drop pickup probe complete"
            )

        if self.state == TrunkContactState.VERIFY_PROGRESS:
            progress = self._burst_progress(view, pov)
            if progress:
                self._failed_attack_rounds = 0
                if self._trunk_usable(view):
                    if (
                        config.require_raycast_attack_confirmation
                        and not self._raycast_confirms_attack(raycast)
                    ):
                        self.counters.prevented_unconfirmed_attacks += 1
                        self._start_exact_log_rescan(
                            "progress retry requires exact in-range log confirmation",
                            telemetry,
                        )
                        return config.noop_action
                    self._start_attack_burst(view, pov, telemetry, raycast)
                else:
                    self._enter_find_trunk(
                        "trunk consumed; locating next trunk face",
                        telemetry,
                        reacquire=True,
                    )
            else:
                self.counters.attack_rounds += 1
                self._failed_attack_rounds += 1
                if self._failed_attack_rounds < config.max_attack_rounds:
                    self._queue.extend(
                        [config.backward_action] * config.backoff_backward_steps
                        + [config.right_action] * config.backoff_turn_steps
                    )
                    self.counters.backoffs += 1
                    self._transition(
                        TrunkContactState.BACKOFF,
                        "attack burst without progress",
                    )
                elif self.counters.orbits < config.max_orbits:
                    self._queue.extend(
                        [config.right_action] * config.orbit_turn_steps
                        + [config.forward_action] * config.orbit_forward_steps
                    )
                    self.counters.orbits += 1
                    self._failed_attack_rounds = 0
                    self._transition(
                        TrunkContactState.ORBIT_REACQUIRE,
                        "repeated bursts without progress; orbiting",
                    )
                else:
                    self._finish(
                        "replan",
                        "attack attempts exhausted without a log",
                    )
                    return config.noop_action

        if self.state == TrunkContactState.SUCCESS:
            return config.noop_action
        if self.state == TrunkContactState.REPLAN:
            return config.noop_action
        return config.noop_action

    def observe(
        self, action: int, reward: float, done: bool, info: Dict[str, Any]
    ) -> None:
        """Consume step outcomes; only reward counts as log evidence."""

        if not self.engaged:
            return
        self._burst_reward += float(reward)
        if float(reward) > 0 and self.engaged and self.result != "success":
            if self._drop_planner is not None:
                self._drop_planner.record_reward(float(reward))
            if (
                self._awaiting_drop_reward
                and self._drop_recovery_had_disappearance
            ):
                self.counters.pickup_after_disappearance += 1
            self._awaiting_drop_reward = False
            # MineRL can report the log reward during the short pickup probe,
            # or one step after a contact attempt requested global replanning.
            self._finish("success", "log collected")

    def diagnostics(self) -> Dict[str, Any]:
        view = self._last_view
        drop_diagnostics = (
            {} if self._drop_planner is None else self._drop_planner.diagnostics()
        )
        drop_waypoint_records = list(self._drop_waypoint_records)
        if self._drop_planner is not None:
            drop_waypoint_records.extend(self._drop_planner.waypoint_records)
        return {
            "state": self.state.value,
            "result": self.result,
            "engaged": self.engaged,
            "active": self.active,
            "attempt_id": self._attempt_id,
            "candidate_id": self._candidate_id,
            "attempt_step": self._attempt_steps,
            "crosshair_trunk_fraction": (
                None if view is None else round(view.crosshair_trunk_fraction, 3)
            ),
            "trunk_area_px": (
                None if view is None else round(view.area_px, 1)
            ),
            "trunk_material": (
                None if view is None else view.material
            ),
            "leaf_occlusion_fraction": round(
                self._last_leaf_occlusion, 3
            ),
            "raycast_has_block": (
                None
                if self._last_raycast is None
                else self._last_raycast.has_block
            ),
            "raycast_is_log": (
                None
                if self._last_raycast is None
                else self._last_raycast.is_log
            ),
            "raycast_is_leaves": (
                None
                if self._last_raycast is None
                else self._last_raycast.is_leaves
            ),
            "raycast_in_range": (
                None
                if self._last_raycast is None
                else self._last_raycast.in_range
            ),
            "raycast_distance": (
                None
                if self._last_raycast is None
                else round(self._last_raycast.distance, 3)
            ),
            "last_contact_x": (
                None
                if self._last_contact_point is None
                else round(self._last_contact_point[0], 3)
            ),
            "last_contact_y": (
                None
                if self._last_contact_point is None
                else round(self._last_contact_point[1], 3)
            ),
            "last_contact_z": (
                None
                if self._last_contact_point is None
                else round(self._last_contact_point[2], 3)
            ),
            "drop_target_x": (
                None
                if self._drop_target_point is None
                else round(self._drop_target_point[0], 3)
            ),
            "drop_target_y": (
                None
                if self._drop_target_point is None
                else round(self._drop_target_point[1], 3)
            ),
            "drop_target_z": (
                None
                if self._drop_target_point is None
                else round(self._drop_target_point[2], 3)
            ),
            "drop_recovery": drop_diagnostics,
            "drop_waypoint_records": drop_waypoint_records,
            "coordinate_target_id": (
                None
                if self._coordinate_target is None
                else self._coordinate_target.target_id
            ),
            "coordinate_target_x": (
                None if self._coordinate_target is None else round(self._coordinate_target.x, 3)
            ),
            "coordinate_target_y": (
                None if self._coordinate_target is None else round(self._coordinate_target.y, 3)
            ),
            "coordinate_target_z": (
                None if self._coordinate_target is None else round(self._coordinate_target.z, 3)
            ),
            "coordinate_target_count": len(self._target_memory.targets),
            "coordinate_target_rows": self._target_memory.rows(),
            "coordinate_target_score": (
                None
                if self._coordinate_target is None
                else round(self._coordinate_target.score, 4)
            ),
            "coordinate_target_score_terms": (
                {}
                if self._coordinate_target is None
                else dict(self._coordinate_target.score_terms)
            ),
            "coordinate_target_yaw": (
                None if self._coordinate_error is None else round(self._coordinate_error.target_yaw, 3)
            ),
            "coordinate_target_pitch": (
                None if self._coordinate_error is None else round(self._coordinate_error.target_pitch, 3)
            ),
            "coordinate_yaw_error": (
                None if self._coordinate_error is None else round(self._coordinate_error.yaw_error, 3)
            ),
            "coordinate_pitch_error": (
                None if self._coordinate_error is None else round(self._coordinate_error.pitch_error, 3)
            ),
            "coordinate_horizontal_distance": (
                None if self._coordinate_error is None else round(self._coordinate_error.horizontal_distance, 3)
            ),
            "coordinate_distance": (
                None if self._coordinate_error is None else round(self._coordinate_error.distance, 3)
            ),
            "coordinate_progress": self._coordinate_progress.diagnostics(),
            "coordinate_post_recovery": {
                "translation_budget": (
                    self.config.coordinate_post_recovery_translation_budget
                ),
                "translation_samples": self._coordinate_post_recovery_samples,
                "state_step_budget": self.config.coordinate_post_recovery_max_steps,
                "state_steps": self._coordinate_post_recovery_state_steps,
                "initial_distance": self._coordinate_post_recovery_initial_distance,
                "minimum_distance": self._coordinate_post_recovery_minimum_distance,
                "timeout_extension": self._coordinate_timeout_extension,
            },
            "exact_log_rescan": {
                "budget": self.config.exact_log_rescan_budget,
                "remaining_actions": len(self._exact_log_rescan_actions),
                "reason": self._exact_log_rescan_reason,
                "candidate_attempts": self._exact_log_rescans_by_candidate.get(
                    self._exact_log_rescan_key(), 0
                ),
                "spatial_cooldown_enabled": (
                    self.config.enable_spatial_exact_log_rescan_cooldown
                ),
                "spatial_radius": self.config.exact_log_rescan_spatial_radius,
                "origin": self._exact_log_rescan_origin,
                "failed_regions": list(self._failed_exact_log_scan_regions),
            },
            "drop_timeout_extension": self._drop_timeout_extension,
            "coordinate_selection_records": list(
                self._coordinate_selection_records
            ),
            "counters": self.counters.as_dict(),
            "transitions": list(self.counters.transitions),
            "transition_records": list(self._transition_records),
            "attempt_results": list(self._attempt_results),
        }
