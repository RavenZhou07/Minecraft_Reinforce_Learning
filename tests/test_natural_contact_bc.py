"""Tests for the natural Treechop contact behaviour-cloning pipeline.

Covers the teacher/student information boundary, contact-owner action
replacement, seed isolation, frame-stack causality, mirror augmentation,
training numerics, and checkpoint integrity for the v9.6 BC student.
"""

from typing import Any, Dict, List

import numpy as np
import pytest

from mc_rl.candidates import ResourceDetection
from mc_rl.natural_bc_runner import NaturalContactRunner
from mc_rl.natural_contact_bc import (
    ACTION_CLASSES,
    BANNED_SEED_RANGE,
    ContactFrameHistory,
    MIRROR_ACTION_MAP,
    NaturalContactBCPolicy,
    STUDENT_INPUT_MANIFEST,
    StudentContactAgent,
    StudentObservation,
    assert_seed_isolation,
    mirror_actions,
    mirror_pov_frames,
    previous_action_one_hot,
    student_observation,
)
from mc_rl.resource_adapters import ResourceAdapter, TrunkView
from mc_rl.search_policy import CandidateSearchPolicy, SearchConfig, SearchState
from mc_rl.telemetry import VisualRangeEstimate
from mc_rl.trunk_contact import (
    CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
    TrunkContactConfig,
)


RNG = np.random.RandomState(7)


def random_frames(count: int, height: int = 8, width: int = 8) -> np.ndarray:
    return RNG.randint(
        0, 255, size=(count, height, width, 3), dtype=np.uint8
    )


def observation(pov: np.ndarray) -> Dict[str, Any]:
    return {
        "pov": pov,
        "telemetry": {
            "x": 0.0,
            "y": 4.0,
            "z": 0.0,
            "yaw": 0.0,
            "pitch": 0.0,
            "biome_id": 1,
            "biome_temperature": 0.8,
            "biome_rainfall": 0.4,
        },
        "raycast": {
            "has_block": 1.0,
            "is_log": 1.0,
            "is_leaves": 0.0,
            "in_range": 1.0,
            "distance": 2.0,
            "x": 0.0,
            "y": 5.0,
            "z": 1.0,
        },
    }


class ScriptedTeacherPolicy:
    """Fake teacher with a fixed contact-ownership schedule."""

    def __init__(
        self,
        contact_steps=(3, 4, 5),
        contact_action=7,
        global_action=1,
    ):
        self.contact_steps = set(contact_steps)
        self.contact_action = int(contact_action)
        self.global_action = int(global_action)
        self.step_index = 0
        self.last_action_source = "global"

    def act(self, obs: Dict[str, Any]) -> int:
        if self.step_index in self.contact_steps:
            self.last_action_source = "contact"
            action = self.contact_action
        else:
            self.last_action_source = "global"
            action = self.global_action
        self.step_index += 1
        return action

    def contact_diagnostics(self) -> Dict[str, Any]:
        return {}

    def observe_transition(self, *args) -> None:
        pass


def tiny_trained_policy(
    samples: int = 24, frame_stack: int = 2, feature_size: int = 4
) -> NaturalContactBCPolicy:
    """A deterministic student with fitted normalization and zero weights."""

    policy = NaturalContactBCPolicy(
        feature_size=feature_size, frame_stack=frame_stack
    )
    base = random_frames(1, 8, 8)[0]
    pov = np.repeat(base[None, None, ...], samples, axis=0)
    pov = np.repeat(pov, frame_stack, axis=1)
    features = policy.build_features(pov, [0] * samples, fit_normalization=True)
    policy.weights = np.zeros(
        (features.shape[1], len(ACTION_CLASSES)), dtype=np.float32
    )
    policy.best_epoch = 0
    return policy


class FixedActionAgent(StudentContactAgent):
    """Student agent stub that always answers with one action."""

    def __init__(self, policy: NaturalContactBCPolicy, action: int):
        super().__init__(policy)
        self.fixed_action = int(action)

    def act(self, observation: StudentObservation) -> int:
        pov = observation["pov"]
        self.history.push(pov)
        return self.fixed_action


class PrivilegedAgent(StudentContactAgent):
    """Deliberately malicious agent that tries to read the raycast."""

    def act(self, observation: StudentObservation) -> int:
        _ = observation["raycast"]  # must raise through the guarded view
        return 0


# ---------------------------------------------------------------------------
# 1. Teacher freeze and runner transparency.
# ---------------------------------------------------------------------------


def test_runner_teacher_mode_executes_the_frozen_v9_6_actions_exactly():
    frames = random_frames(12)
    teacher = ScriptedTeacherPolicy()
    runner = NaturalContactRunner(teacher, None, "teacher", 4)
    executed_actions = []
    for frame in frames:
        executed, record = runner.act(observation(frame))
        executed_actions.append(executed)
        assert record["executed_action"] == record["teacher_action"]
        assert record["student_action"] is None
    assert executed_actions == [
        teacher.global_action,
        teacher.global_action,
        teacher.global_action,
        teacher.contact_action,
        teacher.contact_action,
        teacher.contact_action,
    ] + [teacher.global_action] * 6
    assert runner.privileged_student_input_accesses == 0


def test_v9_6_teacher_profile_defaults_are_unchanged_for_bc():
    config = TrunkContactConfig.for_profile(
        CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6
    )
    assert config.require_raycast_attack_confirmation is True
    assert config.enable_spatial_exact_log_rescan_cooldown is True
    assert config.exact_log_rescan_spatial_radius == 8.0
    assert config.reset_loop_budgets_on_rescan_success is True
    assert config.enable_coordinate_climb_assist is True
    assert config.drop_recovery_elevated_pickup is True
    assert SearchConfig().enable_terrain_route_recovery is False


# ---------------------------------------------------------------------------
# 2/3. Frame-stack causality.
# ---------------------------------------------------------------------------


def test_frame_stack_never_crosses_episode_boundary():
    history = ContactFrameHistory(4)
    history.reset_episode()
    first = random_frames(6)
    for frame in first:
        history.push(frame)
    stack_before = history.current_stack()
    history.reset_episode()
    second = random_frames(2)
    for frame in second:
        history.push(frame)
    stack_after = history.current_stack()
    assert stack_after.shape == (4, 8, 8, 3)
    # After the reset the stack can only contain the two new frames
    # (padded by repeating the episode's first frame).
    unique = {
        frame.tobytes() for frame in stack_after
    }
    assert unique == {second[0].tobytes(), second[1].tobytes()}
    assert stack_before[-1].tobytes() not in unique


def test_frame_stack_does_not_span_contact_attempts():
    history = ContactFrameHistory(4)
    attempt_a = random_frames(6)
    for frame in attempt_a:
        history.push(frame)
    attempt_b = random_frames(5)
    for index, frame in enumerate(attempt_b):
        history.push(frame)
        stack = history.current_stack()
        if index >= 3:
            # Four steps into attempt B the stack lies entirely inside B.
            expected = attempt_b[index - 3 : index + 1]
        else:
            # Early attempt-B steps may use at most three causal pre-roll
            # frames (the tail of attempt A), never frames before its end.
            pre_roll = attempt_a[max(0, 3 - index) :]
            expected = np.concatenate(
                (pre_roll, attempt_b[: index + 1]), axis=0
            )[-4:]
        assert [f.tobytes() for f in stack] == [f.tobytes() for f in expected]


# ---------------------------------------------------------------------------
# 4/5. Seed isolation.
# ---------------------------------------------------------------------------


def test_seed_overlap_between_train_and_validation_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        assert_seed_isolation([16900, 16901], [16901, 17000])


def test_banned_gate_seeds_are_rejected_in_training_data():
    with pytest.raises(ValueError, match="banned"):
        assert_seed_isolation([16799, 16900], [17000])
    with pytest.raises(ValueError, match="banned"):
        assert_seed_isolation([16900], [16500])
    assert_seed_isolation([16900, 16979], [17000, 17019])
    low, high = BANNED_SEED_RANGE
    assert (low, high) == (16500, 16819)


# ---------------------------------------------------------------------------
# 6/7. Student input manifest and privileged access.
# ---------------------------------------------------------------------------


def test_student_input_manifest_excludes_all_privileged_fields():
    forbidden = (
        "telemetry",
        "raycast",
        "oracle",
        "log_grid",
        "grid",
        "coordinate",
        "target",
        "item",
        "contact_state",
        "teacher",
    )
    for entry in STUDENT_INPUT_MANIFEST:
        for token in forbidden:
            assert token not in entry


def test_guarded_student_observation_fails_on_privileged_access():
    guarded = student_observation(random_frames(1)[0])
    assert "pov" in guarded
    with pytest.raises(KeyError, match="raycast"):
        guarded["raycast"]
    with pytest.raises(KeyError, match="telemetry"):
        guarded["telemetry"]


def test_runner_counts_and_raises_on_privileged_student_access():
    policy = tiny_trained_policy()
    agent = PrivilegedAgent(policy)
    runner = NaturalContactRunner(
        ScriptedTeacherPolicy(contact_steps=(0,)), agent, "shadow", 4
    )
    with pytest.raises(KeyError):
        runner.act(observation(random_frames(1)[0]))
    assert runner.privileged_student_input_accesses == 1


# ---------------------------------------------------------------------------
# 8/9. Action classes and previous-action encoding.
# ---------------------------------------------------------------------------


def test_all_fourteen_discrete_actions_remain_classes():
    assert ACTION_CLASSES.tolist() == list(range(14))
    policy = tiny_trained_policy()
    labels = [0, 2, 5, 7, 9, 13]
    encoded = policy._validate_labels(labels)
    assert encoded.tolist() == labels
    with pytest.raises(ValueError):
        policy._validate_labels([14])
    with pytest.raises(ValueError):
        policy._validate_labels([-1])


def test_previous_action_one_hot_encoding_is_correct():
    one_hot = previous_action_one_hot([0, 3, 13])
    assert one_hot.shape == (3, 14)
    assert one_hot.sum(axis=1).tolist() == [1.0, 1.0, 1.0]
    assert one_hot[0, 0] == 1.0
    assert one_hot[1, 3] == 1.0
    assert one_hot[2, 13] == 1.0
    with pytest.raises(ValueError):
        previous_action_one_hot([99])


# ---------------------------------------------------------------------------
# 10/11. Mirror augmentation.
# ---------------------------------------------------------------------------


def test_horizontal_mirror_swaps_yaw_action_pairs_only():
    swapped = mirror_actions([3, 4, 10, 11])
    assert swapped.tolist() == [4, 3, 11, 10]
    unchanged = [0, 1, 2, 5, 6, 7, 8, 9, 12, 13]
    assert mirror_actions(unchanged).tolist() == unchanged
    for action in range(14):
        assert MIRROR_ACTION_MAP[MIRROR_ACTION_MAP[action]] == action


def test_mirror_keeps_attack_pitch_forward_and_jump_fixed():
    for action in (1, 2, 7, 12, 13):
        assert mirror_actions([action]).tolist() == [action]


def test_mirrored_pov_frames_round_trip():
    frames = random_frames(5)
    mirrored = mirror_pov_frames(frames)
    assert np.array_equal(mirror_pov_frames(mirrored), frames)
    assert not np.array_equal(mirrored[0], frames[0])


# ---------------------------------------------------------------------------
# 12/13/14. Training numerics, early stopping, checkpointing.
# ---------------------------------------------------------------------------


def _synthetic_split(samples: int = 40):
    frames = random_frames(samples)
    pov = np.repeat(frames[:, None, ...], 4, axis=1)
    actions = RNG.randint(0, 14, size=samples)
    previous = RNG.randint(0, 14, size=samples)
    return pov, actions, previous


def test_class_balanced_training_stays_finite():
    train = _synthetic_split()
    validation = _synthetic_split(16)
    policy = NaturalContactBCPolicy(feature_size=4, frame_stack=4)
    history = policy.fit(
        *train, *validation, epochs=5, learning_rate=0.01, patience=None
    )
    assert len(history) == 6
    for row in history:
        assert np.isfinite(row["train_loss"])
        assert np.isfinite(row["validation_loss"])
    assert np.all(np.isfinite(policy.weights))


def test_early_stopping_restores_minimum_validation_loss_checkpoint():
    train = _synthetic_split(60)
    validation = _synthetic_split(20)
    policy = NaturalContactBCPolicy(feature_size=4, frame_stack=4)
    history = policy.fit(
        *train,
        *validation,
        epochs=200,
        learning_rate=0.5,
        patience=3,
    )
    losses = [row["validation_loss"] for row in history]
    assert policy.stopped_early
    assert len(history) < 201
    assert policy.best_validation_loss == pytest.approx(min(losses))
    assert policy.best_epoch == int(np.argmin(losses))


def test_checkpoint_save_load_reproduces_predictions(tmp_path):
    train = _synthetic_split(40)
    validation = _synthetic_split(12)
    policy = NaturalContactBCPolicy(feature_size=4, frame_stack=4)
    policy.fit(*train, *validation, epochs=3, patience=None)
    path = tmp_path / "student.npz"
    policy.save(
        str(path),
        dataset_hashes={"train": "abc123"},
        seed_ranges={"train": [16900, 16979]},
    )
    loaded = NaturalContactBCPolicy.load(str(path))
    probe = _synthetic_split(6)
    probabilities = policy.predict_proba_from_features(
        policy.build_features(probe[0], probe[2])
    )
    reloaded_probabilities = loaded.predict_proba_from_features(
        loaded.build_features(probe[0], probe[2])
    )
    assert np.allclose(probabilities, reloaded_probabilities, atol=1e-6)
    assert loaded.student_input_manifest == STUDENT_INPUT_MANIFEST
    assert loaded.dataset_hashes == {"train": "abc123"}
    assert loaded.seed_ranges == {"train": "[16900, 16979]"}
    assert loaded.model_version == "natural_treechop_contact_bc_v1"
    assert "raycast" not in loaded.dataset_hashes


# ---------------------------------------------------------------------------
# 15/16/17/18/19. Runner ownership semantics.
# ---------------------------------------------------------------------------


def test_student_cannot_control_while_contact_owner_is_inactive():
    policy = tiny_trained_policy()
    agent = FixedActionAgent(policy, action=9)
    runner = NaturalContactRunner(
        ScriptedTeacherPolicy(contact_steps=(3,)), agent, "autonomous", 4
    )
    frames = random_frames(5)
    executed = []
    for frame in frames:
        action, record = runner.act(observation(frame))
        executed.append(action)
        if record["action_source"] == "global":
            assert record["executed_action"] == record["teacher_action"]
    assert executed[:3] == [1, 1, 1]  # teacher global action
    assert executed[3] == 9  # student only inside the contact phase
    assert executed[4] == 1
    assert runner.student_actions_executed == 1


def test_shadow_mode_executes_teacher_action_not_student_action():
    policy = tiny_trained_policy()
    agent = FixedActionAgent(policy, action=9)
    runner = NaturalContactRunner(
        ScriptedTeacherPolicy(contact_steps=(1, 2)), agent, "shadow", 4
    )
    frames = random_frames(3)
    for frame in frames:
        action, record = runner.act(observation(frame))
        if record["action_source"] == "contact":
            assert record["executed_action"] == record["teacher_action"] == 7
            assert record["student_action"] == 9
            assert action == 7
    assert runner.student_actions_executed == 0


def test_autonomous_mode_executes_student_action_on_every_contact_step():
    policy = tiny_trained_policy()
    agent = FixedActionAgent(policy, action=4)
    runner = NaturalContactRunner(
        ScriptedTeacherPolicy(contact_steps=(0, 1, 2)), agent, "autonomous", 4
    )
    frames = random_frames(3)
    executed = [runner.act(observation(frame))[0] for frame in frames]
    assert executed == [4, 4, 4]
    assert runner.student_actions_executed == runner.contact_steps == 3
    assert runner.teacher_actions_in_contact == 0


def test_control_returns_to_upstream_after_contact_ends():
    policy = tiny_trained_policy()
    agent = FixedActionAgent(policy, action=9)
    runner = NaturalContactRunner(
        ScriptedTeacherPolicy(contact_steps=(1,)), agent, "autonomous", 4
    )
    frames = random_frames(4)
    executed = [runner.act(observation(frame))[0] for frame in frames]
    assert executed == [1, 9, 1, 1]
    assert not runner.in_contact


def test_runner_preserves_v9_6_owner_mismatch_cancellation():
    class HandoffAdapter(ResourceAdapter):
        resource_type = "tree"

        def detect(self, pov):
            return []

        def interaction_action(self):
            return 8

        def success(self, observation, reward, info):
            return bool(info.get("success", False))

        def estimate_range(self, detection):
            return VisualRangeEstimate(5.0, 1.0)

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

    def telemetry_observation(pov):
        result = {"pov": pov}
        result["telemetry"] = {
            "x": 0.0,
            "y": 4.0,
            "z": 0.0,
            "yaw": 0.0,
            "pitch": 0.0,
            "biome_id": 1,
            "biome_temperature": 0.8,
            "biome_rainfall": 0.4,
        }
        return result

    search_policy = CandidateSearchPolicy(
        HandoffAdapter(),
        SearchConfig(
            sensor_profile="f3_telemetry",
            contact_profile=CONTACT_PROFILE_TERRAIN_ROUTE_DROP_COMPLETION_V9_6,
        ),
    )
    telemetry = search_policy._telemetry(telemetry_observation(np.zeros((8, 8, 3), dtype=np.uint8)))
    candidate, _ = search_policy.candidate_map.add_detection(
        ResourceDetection("tree", 0.0, 0.9, 5.0),
        0.0,
        0,
        telemetry=telemetry,
        range_estimate=VisualRangeEstimate(5.0, 1.0),
    )
    candidate.estimated_world_x = 0.0
    candidate.estimated_world_z = 5.0
    candidate.position_uncertainty = 1.0
    candidate.status = "selected"
    search_policy.selected_candidate = candidate
    search_policy.current_telemetry = telemetry
    search_policy.state = SearchState.APPROACH
    search_policy._contact.engage(
        telemetry=telemetry,
        candidate_id=candidate.candidate_id + 1,
        global_step=10,
        target_hint=(0.0, 5.0),
    )
    runner = NaturalContactRunner(search_policy, None, "teacher", 4)
    runner.act(telemetry_observation(np.zeros((8, 8, 3), dtype=np.uint8)))
    assert search_policy.contact_owner_mismatches == 1
    assert not search_policy._contact.active
    assert runner.privileged_student_input_accesses == 0


def test_student_agent_rejects_untrained_policy():
    policy = NaturalContactBCPolicy(feature_size=4, frame_stack=2)
    with pytest.raises(RuntimeError, match="not been trained"):
        StudentContactAgent(policy)
