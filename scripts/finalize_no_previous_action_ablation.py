"""Register the completed exp13 decision after a gate stop or full evaluation."""

import json
from pathlib import Path

from mc_rl.experiments import append_experiment, file_sha256


EXPERIMENT_ID = "exp13_no_previous_action_ablation_decision"


def artifact(path: str):
    return {"path": path, "sha256": file_sha256(Path(path))}


def main() -> None:
    registry = Path("experiments/registry.jsonl")
    existing_ids = {
        json.loads(line)["experiment_id"]
        for line in registry.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if EXPERIMENT_ID in existing_ids:
        raise FileExistsError("experiment decision is already registered")
    decision = json.loads(
        Path("artifacts/exp13/decision_summary.json").read_text(encoding="utf-8")
    )
    append_experiment(
        {
            "experiment_id": EXPERIMENT_ID,
            "config": artifact("configs/learning/no_previous_action_ablation_exp13.json"),
            "dataset_hashes": {
                "bc_train": "4a0f66617689f8c510cd7fbac7c8803f85a8165daecf47de0c3035e41f0282d3",
                "bc_validation": "51921c5cd6c9523a11294896d6991d3bd75a05fdc397fb0ee5a3929eecdef368",
            },
            "artifacts": {
                "dataset_hash_audit": artifact("artifacts/exp13/dataset_hash_audit.json"),
                "initialization_pairing": artifact("artifacts/exp13/initialization_pairing_audit.json"),
                "zero_channel_invariance": artifact("artifacts/exp13/zero_channel_invariance.json"),
                "sanity_single": artifact("artifacts/exp13/sanity_single_trajectory.json"),
                "decision": artifact("artifacts/exp13/decision_summary.json"),
                "report": artifact("reports/no_previous_action_ablation.md"),
            },
            "metrics": decision["single_trajectory_gate"],
            "correctness_audit": decision["correctness_audit"],
            "downstream_execution": decision["downstream_execution"],
            "conclusion": decision["executive_conclusion"],
            "causal_decision": decision["causal_decision"],
            "exactly_one_recommendation": decision["exactly_one_recommendation"],
            "promotion": "none",
            "status": decision["status"],
        }
    )


if __name__ == "__main__":
    main()
