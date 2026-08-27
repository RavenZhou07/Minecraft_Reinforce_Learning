"""Shared construction for the bootstrap teacher and learned-policy runtime."""

from mc_rl.resource_adapters import TreeResourceAdapter
from mc_rl.search_policy import CandidateSearchPolicy, SearchConfig
from mc_rl.trunk_contact import CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11


def make_bootstrap_teacher(
    max_episode_steps: int,
    contact_profile: str = CONTACT_PROFILE_EMERGENCY_TARGET_PREEMPTION_V9_11,
) -> CandidateSearchPolicy:
    adapter = TreeResourceAdapter(
        interaction_action_id=8,
        interaction_size=45.0,
        interaction_uses_geometry=True,
        interaction_min_apparent_size=12.0,
        range_size_cap=120.0,
        reward_is_success=True,
    )
    return CandidateSearchPolicy(
        adapter,
        SearchConfig(
            backward_action=9,
            sensor_profile="f3_raycast",
            align_threshold_degrees=12.0,
            enable_trunk_contact=True,
            contact_profile=contact_profile,
            episode_max_steps=int(max_episode_steps),
        ),
    )
