import numpy as np

from mc_rl.natural_treechop_bc import (
    ACTION_CLASSES,
    NaturalTreechopBCPolicy,
    action_history_one_hot,
    build_causal_action_history,
    coarse_teacher_phase,
)


def test_action_history_is_causal_and_never_crosses_episodes():
    histories = build_causal_action_history(
        [0, 1, 2, 0, 7], [1, 1, 1, 2, 2], history_length=3
    )
    assert histories.tolist() == [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 2],
        [0, 0, 0],
        [0, 0, 7],
    ]
    assert action_history_one_hot(histories).shape == (5, 3 * len(ACTION_CLASSES))


def test_coarse_phase_routes_pickup_and_recovery_before_generic_contact():
    assert coarse_teacher_phase("APPROACH", "DROP_RECOVERY", "contact") == "pickup"
    assert coarse_teacher_phase("APPROACH", "COORDINATE_RECOVER", "contact") == "recovery"
    assert coarse_teacher_phase("APPROACH", "ATTACK_TRUNK", "contact") == "contact"
    assert coarse_teacher_phase("APPROACH", "", "global") == "approach"
    assert coarse_teacher_phase("SCAN", "", "global") == "search"


def test_end_to_end_policy_trains_and_reloads_without_privileged_inputs(tmp_path):
    rng = np.random.RandomState(4)
    train_count = 24
    validation_count = 10
    train_pov = rng.randint(0, 255, (train_count, 2, 8, 8, 3), dtype=np.uint8)
    validation_pov = rng.randint(0, 255, (validation_count, 2, 8, 8, 3), dtype=np.uint8)
    train_vectors = rng.randn(train_count, 16).astype(np.float32)
    validation_vectors = rng.randn(validation_count, 16).astype(np.float32)
    train_histories = rng.randint(0, 14, (train_count, 3), dtype=np.int64)
    validation_histories = rng.randint(0, 14, (validation_count, 3), dtype=np.int64)
    train_actions = np.asarray(([1, 4, 7] * 8), dtype=np.int64)
    validation_actions = np.asarray(([1, 4, 7, 1, 4] * 2), dtype=np.int64)
    train_phases = np.asarray((["search", "approach", "contact"] * 8))
    validation_phases = np.asarray((["search", "approach", "contact", "search", "approach"] * 2))
    policy = NaturalTreechopBCPolicy(
        feature_size=3, frame_stack=2, action_history=3, use_phase_head=True
    )
    history = policy.fit(
        train_pov,
        train_vectors,
        train_histories,
        train_actions,
        train_phases,
        validation_pov,
        validation_vectors,
        validation_histories,
        validation_actions,
        validation_phases,
        epochs=3,
        patience=3,
    )
    assert history["phase"] and history["action"]
    prediction = policy.predict(
        validation_pov[0], validation_vectors[0], validation_histories[0]
    )
    checkpoint = tmp_path / "student.npz"
    policy.save(str(checkpoint), {"train": "abc"}, "manifest.json")
    loaded = NaturalTreechopBCPolicy.load(str(checkpoint))
    assert loaded.predict(
        validation_pov[0], validation_vectors[0], validation_histories[0]
    ) == prediction
    assert all("raycast" not in item for item in loaded.student_input_manifest)
