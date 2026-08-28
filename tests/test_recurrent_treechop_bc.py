import numpy as np
import pytest
import torch

from mc_rl.recurrent_treechop_bc import (
    ACTION_COUNT,
    RECURRENT_DATASET_FIELDS,
    RECURRENT_STUDENT_INPUT_MANIFEST,
    START_ACTION_TOKEN,
    EpisodeSequence,
    RecurrentArchitecture,
    RecurrentTreechopActor,
    RecurrentTreechopPolicy,
    RecurrentTreechopStudentAgent,
    collate_episode_sequences,
    episode_sequences_from_arrays,
    masked_cross_entropy,
)


class AuditedArrays(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accessed = []

    def __contains__(self, key):
        self.accessed.append(key)
        return super().__contains__(key)

    def __getitem__(self, key):
        self.accessed.append(key)
        return super().__getitem__(key)


def arrays_for_two_episodes():
    actions = np.asarray([3, 1, 7, 4, 8], dtype=np.int32)
    return AuditedArrays(
        pov=np.zeros((5, 64, 64, 3), dtype=np.uint8),
        legal_vector=np.zeros((5, 16), dtype=np.float32),
        action=actions,
        previous_action=np.asarray([0, 3, 1, 0, 4], dtype=np.int32),
        episode=np.asarray([1, 1, 1, 2, 2], dtype=np.int32),
        episode_seed=np.asarray([10, 10, 10, 11, 11], dtype=np.int32),
        episode_step=np.asarray([0, 1, 2, 0, 1], dtype=np.int32),
        episode_success=np.ones(5, dtype=np.int8),
        audit_coarse_phase=np.asarray(["search"] * 5),
        raycast=np.ones(5),
        teacher_target=np.ones((5, 3)),
    )


def raw_observation():
    return {
        "pov": np.zeros((64, 64, 3), dtype=np.uint8),
        "telemetry": {
            "x": 0.0,
            "y": 64.0,
            "z": 0.0,
            "yaw": 0.0,
            "pitch": 0.0,
            "biome_id": 4,
            "biome_temperature": 0.7,
            "biome_rainfall": 0.8,
        },
        "inventory": {"log": 0, "log2": 0},
        "raycast": {"is_log": True},
        "teacher_target": [1.0, 2.0, 3.0],
    }


def test_sequence_alignment_uses_start_then_only_prior_actions():
    arrays = arrays_for_two_episodes()
    episodes = episode_sequences_from_arrays(arrays)
    assert episodes[0].previous_action_token.tolist() == [START_ACTION_TOKEN, 3, 1]
    assert episodes[1].previous_action_token.tolist() == [START_ACTION_TOKEN, 4]
    assert episodes[0].action.tolist() == [3, 1, 7]
    assert episodes[0].previous_action_token[0] != episodes[0].action[0]


def test_sequence_loader_reads_no_privileged_actor_arrays():
    arrays = arrays_for_two_episodes()
    episode_sequences_from_arrays(arrays)
    assert set(arrays.accessed).issubset(set(RECURRENT_DATASET_FIELDS))
    assert "audit_coarse_phase" not in arrays.accessed
    assert "raycast" not in arrays.accessed
    assert all("phase" not in name and "raycast" not in name for name in RECURRENT_STUDENT_INPUT_MANIFEST)


def test_sequence_loader_rejects_action_off_by_one():
    arrays = arrays_for_two_episodes()
    arrays["previous_action"][2] = 3
    with pytest.raises(ValueError, match="alignment"):
        episode_sequences_from_arrays(arrays)


def test_padding_mask_excludes_padding_loss_and_preserves_episode_boundary():
    episodes = episode_sequences_from_arrays(arrays_for_two_episodes())
    batch = collate_episode_sequences(episodes)
    assert batch.mask.tolist() == [[True, True, True], [True, True, False]]
    assert batch.previous_action_token[:, 0].tolist() == [START_ACTION_TOKEN] * 2
    logits = torch.zeros((2, 3, ACTION_COUNT), dtype=torch.float32)
    logits[1, 2, 0] = -1e6
    loss = masked_cross_entropy(logits, batch.action, batch.mask)
    assert torch.isclose(loss, torch.tensor(np.log(ACTION_COUNT), dtype=torch.float32))


def test_batched_gru_rows_do_not_share_hidden_state():
    torch.manual_seed(2)
    actor = RecurrentTreechopActor(RecurrentArchitecture())
    actor.eval()
    episodes = episode_sequences_from_arrays(arrays_for_two_episodes())
    repeated = EpisodeSequence(
        episode_id=3,
        seed=12,
        pov=episodes[0].pov.copy(),
        legal_vector=episodes[0].legal_vector.copy(),
        previous_action_token=episodes[0].previous_action_token.copy(),
        action=episodes[0].action.copy(),
    )
    batch = collate_episode_sequences([episodes[0], repeated])
    with torch.no_grad():
        logits, _ = actor(batch.pov, batch.legal_vector, batch.previous_action_token)
    assert torch.allclose(logits[0], logits[1])


def test_agent_environment_reset_zeros_hidden_and_restores_start_token():
    torch.manual_seed(3)
    policy = RecurrentTreechopPolicy()
    agent = RecurrentTreechopStudentAgent(policy, max_episode_steps=20)
    agent.act(raw_observation(), 0)
    assert agent.hidden is not None
    agent.observe_transition(7)
    assert agent.previous_action_token == 7
    agent.reset_episode()
    assert agent.hidden is None
    assert agent.previous_action_token == START_ACTION_TOKEN
    assert not agent.started


def test_recurrent_checkpoint_reload_preserves_legal_manifest(tmp_path):
    torch.manual_seed(4)
    policy = RecurrentTreechopPolicy()
    checkpoint = tmp_path / "recurrent.pt"
    policy.save(str(checkpoint), {"train": "abc"}, "manifest.json")
    loaded = RecurrentTreechopPolicy.load(str(checkpoint))
    assert loaded.student_input_manifest == RECURRENT_STUDENT_INPUT_MANIFEST
    assert loaded.dataset_hashes == {"train": "abc"}
    for key in policy.model.state_dict():
        assert torch.equal(policy.model.state_dict()[key], loaded.model.state_dict()[key])
