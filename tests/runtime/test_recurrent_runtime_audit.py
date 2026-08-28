import numpy as np
import torch

from mc_rl.learning_observation import LegalObservationAdapter
from mc_rl.recurrent_treechop_bc import (
    START_ACTION_TOKEN,
    RecurrentTreechopPolicy,
    RecurrentTreechopStudentAgent,
)
from mc_rl.runtime_observability import (
    RuntimeTraceRecorder,
    atomic_save_trace,
    load_trace,
    standalone_replay,
    validate_trace_integrity,
)


def raw_observation(step=0):
    pov = np.zeros((64, 64, 3), dtype=np.uint8)
    pov[:, :, step % 3] = step * 7
    return {
        "pov": pov,
        "telemetry": {
            "x": 0.1 * step,
            "y": 64.0,
            "z": 0.0,
            "yaw": 5.0 * step,
            "pitch": 0.0,
            "biome_id": 4,
            "biome_temperature": 0.7,
            "biome_rainfall": 0.8,
        },
        "inventory": {"log": 0, "log2": 0},
        "raycast": {"is_log": bool(step % 2), "in_range": False, "distance": 8.0},
    }


def make_trace(policy, checkpoint):
    adapter = LegalObservationAdapter(max_episode_steps=10)
    recorder = RuntimeTraceRecorder(str(checkpoint), 29, 18500, 10)
    hidden = None
    previous = START_ACTION_TOKEN
    for step in range(3):
        observation = raw_observation(step)
        legal = adapter.reset(observation) if step == 0 else adapter.adapt(observation, step)
        action, probabilities, hidden, diagnostics = policy.predict_step_with_diagnostics(
            legal.pov, legal.vector, previous, hidden
        )
        recorder.append(
            step,
            observation,
            legal.pov,
            legal.vector,
            previous,
            probabilities,
            diagnostics,
            hidden,
            action,
            action,
        )
        previous = action
    recorder.finalize(False, {"success": False, "inventory_log_delta": 0})
    return recorder.arrays()


def test_live_and_standalone_replay_are_identical(tmp_path):
    torch.manual_seed(12)
    policy = RecurrentTreechopPolicy()
    checkpoint = tmp_path / "actor.pt"
    policy.save(str(checkpoint), {"train": "diagnostic"}, "manifest")
    trace = make_trace(policy, checkpoint)
    parity = standalone_replay(str(checkpoint), trace)
    assert parity["passed"]
    assert parity["action_logits_max_abs_error"] <= 1e-6


def test_selected_executed_action_mapping_and_previous_token_are_causal(tmp_path):
    torch.manual_seed(13)
    policy = RecurrentTreechopPolicy()
    checkpoint = tmp_path / "actor.pt"
    policy.save(str(checkpoint), {}, "manifest")
    trace = make_trace(policy, checkpoint)
    assert validate_trace_integrity(trace)["passed"]
    corrupted = dict(trace)
    corrupted["executed_action_id"] = trace["executed_action_id"].copy()
    corrupted["executed_action_id"][1] = (int(corrupted["executed_action_id"][1]) + 1) % 14
    assert not validate_trace_integrity(corrupted)["selected_executed_match"]


def test_agent_advances_hidden_exactly_once_per_decision():
    torch.manual_seed(14)
    policy = RecurrentTreechopPolicy()
    calls = []
    handle = policy.model.gru.register_forward_hook(lambda *args: calls.append(1))
    try:
        agent = RecurrentTreechopStudentAgent(policy, max_episode_steps=10)
        action, _ = agent.act(raw_observation(0), 0)
        assert len(calls) == 1
        hidden_after_decision = agent.hidden.clone()
        agent.observe_transition(action)
        assert len(calls) == 1
        assert torch.equal(agent.hidden, hidden_after_decision)
    finally:
        handle.remove()


def test_trace_serialization_round_trip(tmp_path):
    torch.manual_seed(15)
    policy = RecurrentTreechopPolicy()
    checkpoint = tmp_path / "actor.pt"
    policy.save(str(checkpoint), {}, "manifest")
    trace = make_trace(policy, checkpoint)
    path = tmp_path / "trace.npz"
    atomic_save_trace(path, trace)
    loaded = load_trace(path)
    assert set(loaded) == set(trace)
    for key in trace:
        assert np.array_equal(loaded[key], trace[key])
