import numpy as np

from mc_rl.vision import (
    LinearVisualPolicy,
    build_frame_stacks,
    clockwise_search_action,
    trend_summary,
)


def _synthetic_tree_frames(samples_per_class=20):
    rng = np.random.RandomState(7)
    frames = []
    actions = []
    for action, centre_x in ((3, 12), (1, 32), (4, 52)):
        for _ in range(samples_per_class):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[..., 1] = 110
            left = centre_x - 5 + rng.randint(-2, 3)
            frame[15:58, left : left + 10] = np.array([120, 75, 35])
            noise = rng.randint(0, 8, size=frame.shape, dtype=np.uint8)
            frames.append(np.clip(frame + noise, 0, 255))
            actions.append(action)
    return np.asarray(frames), np.asarray(actions)


def test_linear_visual_policy_learns_spatial_target_direction(tmp_path):
    frames, actions = _synthetic_tree_frames()
    training_indices = np.concatenate(
        (np.arange(0, 15), np.arange(20, 35), np.arange(40, 55))
    )
    validation_indices = np.setdiff1d(np.arange(len(frames)), training_indices)
    policy = LinearVisualPolicy(feature_size=8)
    history = policy.fit(
        frames[training_indices],
        actions[training_indices],
        frames[validation_indices],
        actions[validation_indices],
        epochs=80,
        learning_rate=0.05,
    )
    summary = trend_summary(history)
    assert summary["relative_validation_loss_improvement"] > 0.5
    assert summary["final_validation_accuracy"] > 0.9

    calibration = policy.calibrate_forward_bias(
        frames[validation_indices], actions[validation_indices]
    )
    assert calibration["calibrated_balanced_accuracy"] > 0.9

    model_path = tmp_path / "visual_policy.npz"
    policy.save(str(model_path))
    loaded = LinearVisualPolicy.load(str(model_path))
    assert np.array_equal(
        loaded.predict(frames[validation_indices]), actions[validation_indices]
    )


def test_frame_stacks_are_causal_and_do_not_cross_episodes():
    frames = np.arange(5, dtype=np.uint8)[:, None, None, None]
    stacked = build_frame_stacks(frames, [0, 0, 0, 1, 1], frame_stack=3)
    assert stacked[:, :, 0, 0, 0].tolist() == [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 2],
        [3, 3, 3],
        [3, 3, 4],
    ]


def test_clockwise_search_keeps_forward_and_remaps_both_turns():
    assert clockwise_search_action(1) == 1
    assert clockwise_search_action(3) == 4
    assert clockwise_search_action(4) == 4
