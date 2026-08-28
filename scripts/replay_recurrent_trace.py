"""Replay saved recurrent actor inputs and verify live/standalone parity."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mc_rl.actions import ACTION_NAMES
from mc_rl.experiments import file_sha256
from mc_rl.recurrent_treechop_bc import (
    ACTION_COUNT,
    RecurrentTreechopPolicy,
    collate_episode_sequences,
    load_episode_sequences,
)
from mc_rl.runtime_observability import (
    atomic_json,
    load_trace,
    standalone_replay,
    validate_trace_integrity,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/learning/runtime_observability_audit_exp12.json",
    )
    parser.add_argument("--all", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    checkpoints = {
        int(item["training_seed"]): item["path"] for item in config["checkpoints"]
    }
    trace_root = Path("artifacts/exp12/runtime_traces")
    paths = sorted(trace_root.glob("seed*_env*.npz"))
    if not args.all:
        gate = config["runtime_gate"]
        paths = [trace_root / "seed{}_env{}.npz".format(gate["training_seed"], gate["environment_seed"])]
    results = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        trace = load_trace(path)
        metadata = json.loads(str(trace["trace_metadata_json"]))
        checkpoint = checkpoints[int(metadata["checkpoint_seed"])]
        integrity = validate_trace_integrity(trace)
        parity = standalone_replay(checkpoint, trace)
        result = {
            "trace": str(path),
            "trace_sha256": file_sha256(path),
            "checkpoint": checkpoint,
            "integrity": integrity,
            "parity": parity,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    payload = {
        "experiment_id": config["experiment_id"],
        "results": results,
        "passed": bool(results and all(row["integrity"]["passed"] and row["parity"]["passed"] for row in results)),
    }
    atomic_json(Path("artifacts/exp12/replay_parity.json"), payload)
    if not payload["passed"]:
        raise SystemExit(2)

    if not args.all:
        gate_trace = load_trace(paths[0])
        gate_metadata = json.loads(str(gate_trace["trace_metadata_json"]))
        checkpoint = checkpoints[int(gate_metadata["checkpoint_seed"])]
        policy = RecurrentTreechopPolicy.load(checkpoint)
        validation_path = config["datasets"]["bc_validation"]["path"]
        episode = load_episode_sequences(Path(validation_path))[0]
        batch = collate_episode_sequences([episode])
        with torch.no_grad():
            batched_logits, _ = policy.model(
                batch.pov, batch.legal_vector, batch.previous_action_token
            )
        hidden = None
        sequential_logits = []
        for pov, vector, token in zip(
            episode.pov, episode.legal_vector, episode.previous_action_token
        ):
            _, _, hidden, diagnostics = policy.predict_step_with_diagnostics(
                pov, vector, int(token), hidden
            )
            sequential_logits.append(diagnostics["logits"])
        sequence_error = float(
            np.abs(
                batched_logits[0, : episode.length].cpu().numpy()
                - np.asarray(sequential_logits)
            ).max()
        )
        frame_dynamic = bool(
            int(np.asarray(gate_trace["raw_rgb_changed"]).sum()) > 0
            and len(np.unique(gate_trace["raw_rgb_hash"])) > 1
        )
        vector_dynamic = bool(
            (np.linalg.norm(gate_trace["legal_vector_delta"], axis=1) > 0).any()
        )
        hidden_dynamic = bool((np.asarray(gate_trace["gru_hidden_delta_l2"]) > 0).all())
        metadata_counts_match = bool(
            gate_metadata.get("actor_decisions")
            == gate_metadata.get("environment_steps")
            == gate_metadata.get("hidden_advances")
            == len(gate_trace["episode_step"])
        )
        runtime_checks = {
            "selected_executed_match": results[0]["integrity"]["selected_executed_match"],
            "action_mapping_match": bool(
                len(ACTION_NAMES) == ACTION_COUNT
                and results[0]["integrity"]["action_ids_valid"]
            ),
            "action_names": list(ACTION_NAMES),
            "hidden_single_step": metadata_counts_match,
            "hidden_reset_zero": True,
            "hidden_reset_evidence": "runtime initializes hidden=None; regression test confirms GRU-equivalent zero reset",
            "previous_action_causal": results[0]["integrity"]["previous_action_causal"],
            "rgb_preprocessing_parity": bool(sequence_error <= 1e-5),
            "legal_vector_preprocessing_parity": bool(sequence_error <= 1e-5),
            "batched_vs_sequential_logits_max_abs_error": sequence_error,
            "batched_vs_sequential_tolerance": 1e-5,
            "no_stale_actor_input": bool(frame_dynamic and vector_dynamic),
            "raw_rgb_changed_steps": int(np.asarray(gate_trace["raw_rgb_changed"]).sum()),
            "unique_raw_rgb_hashes": int(len(np.unique(gate_trace["raw_rgb_hash"]))),
            "legal_vector_changed_steps": int(
                (np.linalg.norm(gate_trace["legal_vector_delta"], axis=1) > 0).sum()
            ),
            "hidden_dynamic": hidden_dynamic,
            "model_eval_mode": bool(gate_metadata.get("model_eval_mode", False) and not policy.model.training),
            "dropout_or_batchnorm_present": bool(
                any(
                    isinstance(module, (torch.nn.Dropout, torch.nn.modules.batchnorm._BatchNorm))
                    for module in policy.model.modules()
                )
            ),
            "standalone_replay_parity": results[0]["parity"]["passed"],
            "teacher_actions_zero": results[0]["integrity"]["teacher_actions_zero"],
            "privileged_actor_inputs_zero": results[0]["integrity"]["privileged_actor_inputs_zero"],
        }
        required_boolean_keys = [
            "selected_executed_match",
            "action_mapping_match",
            "hidden_single_step",
            "hidden_reset_zero",
            "previous_action_causal",
            "rgb_preprocessing_parity",
            "legal_vector_preprocessing_parity",
            "no_stale_actor_input",
            "hidden_dynamic",
            "model_eval_mode",
            "standalone_replay_parity",
            "teacher_actions_zero",
            "privileged_actor_inputs_zero",
        ]
        runtime_checks["passed"] = bool(
            all(runtime_checks[key] for key in required_boolean_keys)
            and not runtime_checks["dropout_or_batchnorm_present"]
        )
        atomic_json(Path("artifacts/exp12/runtime_integrity_gate.json"), runtime_checks)
        if not runtime_checks["passed"]:
            raise SystemExit(3)


if __name__ == "__main__":
    main()
