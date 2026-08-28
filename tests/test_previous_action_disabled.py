import numpy as np
import torch

from mc_rl.recurrent_treechop_bc import (
    ACTION_COUNT,
    PREVIOUS_ACTION_DISABLED_ZERO,
    START_ACTION_TOKEN,
    RecurrentArchitecture,
    RecurrentTreechopActor,
    RecurrentTreechopPolicy,
    masked_cross_entropy,
    paired_disabled_zero_policy,
)


def disabled_actor():
    torch.manual_seed(29)
    actor = RecurrentTreechopActor(
        RecurrentArchitecture(previous_action_mode=PREVIOUS_ACTION_DISABLED_ZERO)
    )
    actor.eval()
    return actor


def fixed_inputs(batch=2, timesteps=5):
    generator = torch.Generator().manual_seed(123)
    pov = torch.randint(
        0, 256, (batch, timesteps, 64, 64, 3), dtype=torch.uint8, generator=generator
    )
    vector = torch.randn((batch, timesteps, 16), generator=generator)
    hidden = torch.randn((1, batch, 128), generator=generator)
    return pov, vector, hidden


def test_previous_action_mutation_is_output_invariant():
    actor = disabled_actor()
    pov, vector, hidden = fixed_inputs(batch=1, timesteps=1)
    reference = None
    with torch.no_grad():
        for token in range(START_ACTION_TOKEN + 1):
            outputs = actor.forward_with_diagnostics(
                pov, vector, torch.tensor([[token]]), hidden.clone()
            )
            logits, next_hidden, diagnostics = outputs
            probabilities = torch.softmax(logits, dim=-1)
            current = {
                "combined": diagnostics["combined_embedding"],
                "hidden": next_hidden,
                "logits": logits,
                "probabilities": probabilities,
                "argmax": torch.argmax(probabilities, dim=-1),
            }
            assert diagnostics["action_embedding"].abs().max().item() == 0.0
            if reference is None:
                reference = {key: value.clone() for key, value in current.items()}
            else:
                for key in current:
                    assert torch.equal(current[key], reference[key]), key


def test_sequence_mutation_preserves_logits_loss_and_shared_gradients():
    actor = disabled_actor()
    actor.train()
    pov, vector, hidden = fixed_inputs()
    actions = torch.tensor([[0, 1, 7, 8, 3], [4, 4, 1, 2, 10]], dtype=torch.long)
    mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
    teacher_tokens = torch.tensor(
        [[START_ACTION_TOKEN, 0, 1, 7, 8], [START_ACTION_TOKEN, 4, 4, 1, 2]],
        dtype=torch.long,
    )
    random_tokens = torch.randint(
        0, ACTION_COUNT + 1, teacher_tokens.shape, generator=torch.Generator().manual_seed(9)
    )

    def run(tokens):
        actor.zero_grad(set_to_none=True)
        logits, _, _ = actor.forward_with_diagnostics(pov, vector, tokens, hidden.clone())
        loss = masked_cross_entropy(logits, actions, mask)
        loss.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in actor.named_parameters()
        }
        return logits.detach().clone(), loss.detach().clone(), gradients

    logits_a, loss_a, gradients_a = run(teacher_tokens)
    logits_b, loss_b, gradients_b = run(random_tokens)
    assert torch.equal(logits_a, logits_b)
    assert torch.equal(loss_a, loss_b)
    assert gradients_a.keys() == gradients_b.keys()
    assert all(torch.equal(gradients_a[key], gradients_b[key]) for key in gradients_a)


def test_disabled_actor_has_no_action_embedding_parameter_or_checkpoint_dependency(tmp_path):
    policy, audit = paired_disabled_zero_policy(seed=29)
    assert audit["shared_tensors_exactly_equal"]
    assert audit["gru_input_width"] == 144
    assert not audit["disabled_state_has_action_embedding"]
    assert not audit["disabled_trainable_action_embedding_parameters"]
    assert all(
        not name.startswith("previous_action_embedding.")
        for name in policy.model.state_dict()
    )
    checkpoint = tmp_path / "disabled.pt"
    policy.save(str(checkpoint), {"train": "abc"}, "manifest")
    reloaded = RecurrentTreechopPolicy.load(str(checkpoint))
    assert reloaded.architecture.previous_action_mode == PREVIOUS_ACTION_DISABLED_ZERO
    for name, value in policy.model.state_dict().items():
        assert torch.equal(value, reloaded.model.state_dict()[name])


def test_numpy_one_step_api_ignores_arbitrary_previous_action():
    policy, _ = paired_disabled_zero_policy(seed=29)
    pov = np.zeros((64, 64, 3), dtype=np.uint8)
    vector = np.zeros(16, dtype=np.float32)
    outputs = [policy.predict_step_with_diagnostics(pov, vector, token, None) for token in range(15)]
    for current in outputs[1:]:
        assert current[0] == outputs[0][0]
        assert np.array_equal(current[1], outputs[0][1])
        assert torch.equal(current[2], outputs[0][2])
        for key in ("combined_embedding", "logits", "action_embedding"):
            assert np.array_equal(current[3][key], outputs[0][3][key])

