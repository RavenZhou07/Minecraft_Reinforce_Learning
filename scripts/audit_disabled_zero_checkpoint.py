"""Audit a trained disabled-zero checkpoint for exact token invariance."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc_rl.experiments import file_sha256
from mc_rl.recurrent_treechop_bc import (
    ACTION_COUNT,
    START_ACTION_TOKEN,
    RecurrentTreechopPolicy,
    collate_episode_sequences,
    load_episode_sequences,
    masked_cross_entropy,
)
from scripts.train_recurrent_treechop_bc import atomic_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def forward(policy, batch):
    logits, next_hidden, diagnostics = policy.model.forward_with_diagnostics(
        batch.pov, batch.legal_vector, batch.previous_action_token, None
    )
    probabilities = torch.softmax(logits, dim=-1)
    return {
        "combined": diagnostics["combined_embedding"],
        "recurrent": diagnostics["recurrent_output"],
        "next_hidden": next_hidden,
        "logits": logits,
        "probabilities": probabilities,
        "argmax": torch.argmax(probabilities, dim=-1),
        "zero_slot": diagnostics["action_embedding"],
    }


def gradients(policy, batch):
    policy.model.zero_grad(set_to_none=True)
    logits, _, _ = policy.model.forward_with_diagnostics(
        batch.pov, batch.legal_vector, batch.previous_action_token, None
    )
    loss = masked_cross_entropy(logits, batch.action, batch.mask)
    loss.backward()
    return loss.detach().clone(), {
        name: parameter.grad.detach().clone()
        for name, parameter in policy.model.named_parameters()
    }


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    dataset_hash = file_sha256(dataset_path)
    if dataset_hash != args.expected_dataset_sha256.lower():
        raise RuntimeError("dataset hash mismatch")
    policy = RecurrentTreechopPolicy.load(args.checkpoint)
    if policy.architecture.previous_action_mode != "disabled_zero":
        raise ValueError("checkpoint is not disabled_zero")
    action_embedding_names = [
        name for name in policy.model.state_dict()
        if name.startswith("previous_action_embedding.")
    ]
    episodes = load_episode_sequences(dataset_path, False, args.seeds)
    batch = collate_episode_sequences(episodes).to(policy.device)
    generator = torch.Generator().manual_seed(1517)
    mutated_tokens = torch.randint(
        0, ACTION_COUNT + 1, batch.previous_action_token.shape, generator=generator
    ).to(policy.device)
    mutated = replace(batch, previous_action_token=mutated_tokens)
    policy.model.eval()
    with torch.no_grad():
        original_outputs = forward(policy, batch)
        mutated_outputs = forward(policy, mutated)
        token_outputs = []
        first_episode = episodes[0]
        pov = torch.from_numpy(first_episode.pov[:1])[None].to(policy.device)
        vector = torch.from_numpy(first_episode.legal_vector[:1])[None].to(policy.device)
        hidden = torch.zeros((1, 1, policy.architecture.hidden_size), device=policy.device)
        for token in range(START_ACTION_TOKEN + 1):
            previous = torch.tensor([[token]], dtype=torch.long, device=policy.device)
            token_outputs.append(forward(policy, replace(batch, pov=pov, legal_vector=vector, previous_action_token=previous, action=batch.action[:1, :1], mask=batch.mask[:1, :1], seeds=(episodes[0].seed,))))
    policy.model.train()
    original_loss, original_gradients = gradients(policy, batch)
    mutated_loss, mutated_gradients = gradients(policy, mutated)
    policy.model.eval()
    sequence_equal = {
        key: bool(torch.equal(original_outputs[key], mutated_outputs[key]))
        for key in ("combined", "recurrent", "next_hidden", "logits", "probabilities", "argmax")
    }
    token_equal = {
        key: all(torch.equal(candidate[key], token_outputs[0][key]) for candidate in token_outputs[1:])
        for key in ("combined", "recurrent", "next_hidden", "logits", "probabilities", "argmax")
    }
    gradient_equal = all(
        torch.equal(original_gradients[name], mutated_gradients[name])
        for name in original_gradients
    )
    zero_max = max(
        float(original_outputs["zero_slot"].abs().max().item()),
        float(mutated_outputs["zero_slot"].abs().max().item()),
        max(float(value["zero_slot"].abs().max().item()) for value in token_outputs),
    )
    passed = bool(
        all(sequence_equal.values())
        and all(token_equal.values())
        and torch.equal(original_loss, mutated_loss)
        and gradient_equal
        and zero_max == 0.0
        and not action_embedding_names
    )
    payload = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(Path(args.checkpoint)),
        "dataset": args.dataset,
        "dataset_sha256": dataset_hash,
        "seeds": args.seeds,
        "start_plus_14_token_outputs_exact": token_equal,
        "complete_sequence_outputs_exact": sequence_equal,
        "masked_loss_exact": bool(torch.equal(original_loss, mutated_loss)),
        "shared_gradients_exact": gradient_equal,
        "max_abs_disabled_action_channel": zero_max,
        "serialized_action_embedding_names": action_embedding_names,
        "passed": passed,
    }
    atomic_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
