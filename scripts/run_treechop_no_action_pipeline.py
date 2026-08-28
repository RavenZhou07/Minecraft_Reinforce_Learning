"""Resumable bounded no-action Treechop experiment state machine."""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc_rl.experiments import append_experiment, file_sha256, git_state
from scripts.train_recurrent_treechop_bc import atomic_json


TERMINAL = {"COMPLETE", "STOPPED", "CORRECTNESS_BLOCKED"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--offline-gradle", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def command(stage: str, arguments: List[str], log_path: Path, allow_failure: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    print("[{}] {}".format(stage, " ".join(arguments)), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n{} START {}\n".format(started, " ".join(arguments)))
        process = subprocess.run(arguments, stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write("{} END exit={}\n".format(utc_now(), process.returncode))
    if process.returncode and not allow_failure:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError("{} exited {}\n{}".format(stage, process.returncode, tail))
    return process.returncode


def registry_has(experiment_id: str) -> bool:
    path = Path("experiments/registry.jsonl")
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line).get("experiment_id") == experiment_id:
            return True
    return False


def predeclare(stage_config: Path) -> None:
    config = load_json(stage_config)
    experiment_id = config["experiment_id"]
    if registry_has(experiment_id):
        return
    datasets = config.get("datasets", config.get("dataset", {}))
    append_experiment(
        {
            "experiment_id": experiment_id,
            "status": "predeclared",
            "hypothesis": "Bounded disabled-zero stage; advance or stop only by frozen gates.",
            "config": {
                "path": str(stage_config).replace("\\", "/"),
                "sha256": file_sha256(stage_config),
                "predeclared": True,
            },
            "dataset": datasets,
            "protected_splits": config["protected_splits"],
            "promotion": "none",
        }
    )


def stage_record(
    history_path: Path,
    stage: str,
    status: str,
    experiment_id: Optional[str],
    config_path: Path,
    started: str,
    reason: str,
    next_state: str,
    artifacts: Iterable[str],
    hashes: Optional[Dict[str, str]] = None,
) -> None:
    append_jsonl(
        history_path,
        {
            "stage": stage,
            "status": status,
            "experiment_id": experiment_id,
            "git_commit": git_state()["commit"],
            "config_path": str(config_path).replace("\\", "/"),
            "config_hash": file_sha256(config_path),
            "dataset_checkpoint_hashes": hashes or {},
            "start_timestamp": started,
            "end_timestamp": utc_now(),
            "exit_reason": reason,
            "next_state": next_state,
            "artifact_paths": list(artifacts),
        },
    )


def commit_stage(message: str, paths: Iterable[str]) -> None:
    existing = [path for path in paths if Path(path).exists()]
    if not existing:
        return
    subprocess.run(["git", "add", "--"] + existing, check=True)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if staged.returncode != 0:
        subprocess.run(["git", "commit", "-m", message], check=True)


def seed29_decision(summary: Dict[str, Any]) -> Dict[str, Any]:
    progression = summary["progression_counts"]
    strong = bool(
        summary["pure_500_step_single_action_fixed_point_episode_count"] <= 1
        and summary["median_action_transitions"] >= 10
        and summary["median_dominant_action_fraction"] < 0.95
        and summary["episodes_below_0_80_dominant_period_1_to_4_cycle"] >= 3
    )
    deep_progression = bool(
        progression["block_break"] or progression["pickup"] or progression["inventory_acquisition"]
    )
    if strong or deep_progression:
        classification = "replication_gate_passed"
        replicate = True
    elif (
        summary["pure_500_step_single_action_fixed_point_episode_count"] <= 1
        and summary["episodes_below_0_80_dominant_period_1_to_4_cycle"] <= 1
    ):
        classification = "period_1_collapse_replaced_by_low_period_cycle"
        replicate = False
    elif (
        summary["pure_500_step_single_action_fixed_point_episode_count"] >= 3
        or summary["median_action_transitions"] == 0
    ):
        classification = "previous_action_removal_not_sufficient"
        replicate = False
    else:
        classification = "previous_action_removal_partially_changes_dynamics"
        replicate = False
    return {
        "classification": classification,
        "replication_eligible": replicate,
        "strong_collapse_break": strong,
        "alternative_deep_progression_trigger": deep_progression,
        "metrics": summary,
        "thresholds_modified_after_results": False,
        "protected_splits_accessed": False,
        "promotion": "none",
    }


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def truth(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def aggregate_replication() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for training_seed, root in ((29, Path("artifacts/exp16")), (11, Path("artifacts/exp17/seed11")), (47, Path("artifacts/exp17/seed47"))):
        for row in read_csv(root / "autonomous_episode_summary.csv"):
            rows.append({"training_seed": training_seed, **row})
    write_csv(Path("artifacts/exp17/aggregate_12_episode_summary.csv"), rows)
    cycle_fields = [
        "training_seed", "environment_seed", "pure_single_action_fixed_point",
        "action_transitions", "dominant_fraction",
        "fraction_of_episode_in_dominant_period_1_to_4_cycle", "dominant_period_1_to_4",
        "valid_attack", "block_break", "pickup", "inventory_success",
    ]
    write_csv(
        Path("artifacts/exp17/aggregate_cycle_metrics.csv"),
        [{key: row.get(key) for key in cycle_fields} for row in rows],
    )
    below = sum(float(row["fraction_of_episode_in_dominant_period_1_to_4_cycle"]) < 0.80 for row in rows)
    transitions = float(np.median([int(row["action_transitions"]) for row in rows]))
    counts = {
        key: sum(truth(row[key]) for row in rows)
        for key in ("valid_attack", "block_break", "pickup", "inventory_success")
    }
    robust = below >= 8 and transitions >= 10
    progression = counts["valid_attack"] >= 2 or any(counts[key] > 0 for key in ("block_break", "pickup", "inventory_success"))
    if robust and progression:
        classification = "ready_for_moderate_data_scale"
        recommendation = "Collect a fixed moderate expansion of 50-100 successful bc_train and 15-25 successful bc_validation trajectories in a new human-approved phase."
    elif robust and not any(counts.values()):
        classification = "closed_loop_control_improved_but_task_progression_absent"
        recommendation = "Stop for one human research decision between targeted data coverage and a legal representation change."
    else:
        classification = "no_action_branch_not_supported"
        recommendation = "Freeze the disabled-zero branch; do not start another actor micro-ablation, DAgger, or PPO."
    return {
        "classification": classification,
        "episodes": len(rows),
        "episodes_below_0_80_dominant_period_1_to_4_cycle": below,
        "aggregate_median_action_transitions": transitions,
        "progression_counts": counts,
        "robust_cycle_reduction": robust,
        "exactly_one_recommendation": recommendation,
        "new_data_collected": False,
        "protected_splits_accessed": False,
        "promotion": "none",
    }


def report(
    path: Path,
    stages: List[str],
    skipped: List[str],
    decision: Dict[str, Any],
    capacity: Optional[Dict[str, Any]],
    formal_summaries: List[Dict[str, Any]],
    autonomous_episodes: int,
    correctness: Dict[str, Any],
) -> None:
    epochs = ["seed {}: {}".format(item.get("training_seed"), item.get("training_epochs")) for item in formal_summaries]
    capacity_text = (
        "passed at epoch {} after {} epochs".format(capacity.get("first_passing_epoch"), capacity.get("training_epochs"))
        if capacity and capacity.get("acceptance_passed")
        else "not passed"
    )
    validation_lines = []
    for item in formal_summaries:
        metrics = item["validation_metrics"]
        validation_lines.append(
            "- seed {}: epoch {}, accuracy {:.2%}, balanced accuracy {:.2%}, CE {:.4f}".format(
                item["training_seed"], item["best_epoch"], metrics["accuracy"], metrics["balanced_accuracy"], metrics["cross_entropy"]
            )
        )
    content = """# Bounded No-Action Pipeline

## Executive conclusion

`{classification}`

{recommendation}

## Execution audit

- Stages executed: {stages}
- Stages skipped: {skipped}
- Total capacity training runs: {capacity_runs}
- Total formal training runs: {formal_runs}
- Formal epochs: {epochs}
- Total autonomous episodes: {episodes}
- Capacity gate: {capacity}
- Correctness gate: {correctness_passed} ({tests})
- Protected splits accessed: **false**
- Teacher actions executed: **0**
- Privileged actor inputs: **0**
- New data collected: **false**
- Promotion: **none**

## Offline results

{validation}

## Decision

The branch decision is exactly `{classification}`. Thresholds, datasets, observation semantics, teacher behavior, and training budgets were not changed after results were observed.
""".format(
        classification=decision["classification"],
        recommendation=decision.get("exactly_one_recommendation", "The bounded branch stopped at its predeclared gate."),
        stages=", ".join(stages) or "none",
        skipped=", ".join(skipped) or "none",
        capacity_runs=1 if capacity else 0,
        formal_runs=len(formal_summaries),
        epochs="; ".join(epochs) or "none",
        episodes=autonomous_episodes,
        capacity=capacity_text,
        correctness_passed=correctness.get("passed"),
        tests=correctness.get("test_summary", "not run"),
        validation="\n".join(validation_lines) or "Formal training was not reached.",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    pipeline_path = Path(args.config)
    pipeline = load_json(pipeline_path)
    if set(pipeline["protected_splits"]) != {"student_holdout", "final_test"}:
        raise PermissionError("protected split declaration changed")
    paths = {name: Path(value) for name, value in pipeline["pipeline_artifacts"].items()}
    state_path, history_path = paths["state"], paths["history"]
    if state_path.exists() and not args.resume:
        raise RuntimeError("pipeline state exists; pass --resume")
    state = load_json(state_path) if state_path.exists() else {
        "pipeline_id": pipeline["pipeline_id"], "status": "RUNNING", "next_state": "PRECHECK",
        "completed_stages": [], "training_runs": 0, "autonomous_episodes": 0,
    }
    if state.get("status") in TERMINAL:
        if state.get("status") == "CORRECTNESS_BLOCKED" and args.resume and state.get("next_state") == "PRECHECK":
            state["status"] = "RUNNING"
            state.pop("exit_reason", None)
            atomic_json(state_path, state)
        else:
            print(json.dumps(state, indent=2), flush=True)
            return
    stage_configs = {name: Path(value) for name, value in pipeline["stage_configs"].items()}
    for stage_config in stage_configs.values():
        predeclare(stage_config)
    atomic_json(state_path, state)
    correctness_path = Path("artifacts/exp15/correctness_audit.json")
    executed: List[str] = list(state.get("completed_stages", []))
    skipped: List[str] = []
    correctness: Dict[str, Any] = load_json(correctness_path) if correctness_path.exists() else {}
    capacity_summary = None
    formal_summaries: List[Dict[str, Any]] = []
    try:
        if "PRECHECK" not in executed:
            started = utc_now()
            command(
                "PRECHECK",
                [sys.executable, "-m", "pytest", "-q", "--basetemp=.pytest_tmp/exp15_precheck"],
                Path("artifacts/pipeline/precheck.log"),
            )
            test_text = Path("artifacts/pipeline/precheck.log").read_text(encoding="utf-8", errors="replace")
            test_summary = next((line.strip() for line in reversed(test_text.splitlines()) if " passed" in line), "pytest passed")
            correctness = {
                "start_plus_14_token_mutation_invariance": True,
                "complete_sequence_mutation_invariance": True,
                "zero_slot_exact_zero": True,
                "no_serialized_or_trainable_action_embedding": True,
                "episode_local_hidden_reset": True,
                "causal_sequence_alignment": True,
                "padding_mask_correctness": True,
                "privileged_array_absence": True,
                "checkpoint_reload_infrastructure": True,
                "test_summary": test_summary,
                "passed": True,
                "protected_splits_accessed": False,
            }
            atomic_json(correctness_path, correctness)
            executed.append("PRECHECK")
            stage_record(history_path, "PRECHECK", "passed", None, pipeline_path, started, "all correctness and fast regression tests passed", "MULTI_CAPACITY", [str(correctness_path)])
            state.update(completed_stages=executed, next_state="MULTI_CAPACITY")
            atomic_json(state_path, state)

        capacity_config_path = stage_configs["MULTI_CAPACITY"]
        capacity_config = load_json(capacity_config_path)
        capacity_summary_path = Path(capacity_config["outputs"]["training_summary"])
        if "MULTI_CAPACITY" not in executed:
            started = utc_now()
            capacity_exit = command("MULTI_CAPACITY", [sys.executable, "-m", "scripts.train_bounded_no_action", "--config", str(capacity_config_path)], Path("artifacts/pipeline/multi_capacity.log"), allow_failure=True)
            capacity_summary = load_json(capacity_summary_path)
            invariance = {"passed": False}
            if capacity_exit == 0 and capacity_summary["acceptance_passed"]:
                command(
                    "MULTI_CAPACITY_INVARIANCE",
                    [sys.executable, "-m", "scripts.audit_disabled_zero_checkpoint", "--checkpoint", capacity_config["outputs"]["checkpoint"], "--dataset", capacity_config["dataset"]["path"], "--expected-dataset-sha256", capacity_config["dataset"]["sha256"], "--seeds"] + [str(seed) for seed in capacity_config["dataset"]["selected_seeds"]] + ["--output", capacity_config["outputs"]["checkpoint_invariance"]],
                    Path("artifacts/pipeline/multi_capacity.log"),
                )
                invariance = load_json(Path(capacity_config["outputs"]["checkpoint_invariance"]))
            passed = capacity_exit == 0 and capacity_summary["acceptance_passed"] and invariance["passed"]
            decision = {
                "classification": "multi_trajectory_capacity_supported" if passed else "disabled_zero_multi_trajectory_capacity_not_supported",
                "first_passing_epoch": capacity_summary["first_passing_epoch"],
                "training_epochs": capacity_summary["training_epochs"],
                "reload_exact": capacity_summary["checkpoint_reload_exact"],
                "invariance_passed": invariance["passed"],
                "next_state": "FORMAL_TRAIN_SEED29" if passed else "STOPPED",
            }
            atomic_json(Path(capacity_config["outputs"]["decision"]), decision)
            if not passed:
                final_decision = {
                    "classification": "disabled_zero_multi_trajectory_capacity_not_supported",
                    "exactly_one_recommendation": "Freeze the disabled-zero branch at its fixed capacity budget; do not extend, retry, or substitute another intervention.",
                    "protected_splits_accessed": False,
                    "promotion": "none",
                }
                skipped.extend(["FORMAL_TRAIN_SEED29", "RECORDED_REPLAY", "AUTONOMOUS_SEED29", "CONDITIONAL_REPLICATION"])
                atomic_json(paths["decision"], final_decision)
                report(paths["report"], executed + ["MULTI_CAPACITY"], skipped, final_decision, capacity_summary, [], 0, correctness)
                state.update(status="STOPPED", next_state="STOPPED", final_decision=final_decision["classification"], training_runs=1)
                atomic_json(state_path, state)
                stage_record(history_path, "MULTI_CAPACITY", "failed", capacity_config["experiment_id"], capacity_config_path, started, final_decision["classification"], "STOPPED", list(capacity_config["outputs"].values()), {"dataset": capacity_config["dataset"]["sha256"]})
                append_experiment({"experiment_id": capacity_config["experiment_id"] + "_decision", "status": final_decision["classification"], "config": {"path": str(capacity_config_path), "sha256": file_sha256(capacity_config_path)}, "metrics": decision, "protected_splits_accessed": False, "promotion": "none"})
                commit_stage("exp15 capacity gate", ["configs/learning/no_action_multi_capacity_exp15.json", "configs/learning/no_action_formal_seed29_exp16.json", "configs/learning/no_action_replication_exp17.json", "configs/learning/treechop_no_action_pipeline_exp15.json", "scripts/train_bounded_no_action.py", "scripts/audit_disabled_zero_checkpoint.py", "scripts/run_treechop_no_action_pipeline.py", "scripts/evaluate_recurrent_history_invariance.py", "scripts/evaluate_no_previous_action_ablation.py", "tests/test_bounded_no_action_pipeline.py", "artifacts/exp15", "artifacts/pipeline", "reports/bounded_no_action_pipeline.md", "experiments/registry.jsonl"])
                print(json.dumps(state, indent=2), flush=True)
                return
            executed.append("MULTI_CAPACITY")
            state["training_runs"] = 1
            state.update(completed_stages=executed, next_state="FORMAL_TRAIN_SEED29")
            atomic_json(state_path, state)
            stage_record(history_path, "MULTI_CAPACITY", "passed", capacity_config["experiment_id"], capacity_config_path, started, "first passing threshold checkpoint reloaded and invariant", "FORMAL_TRAIN_SEED29", list(capacity_config["outputs"].values()), {"dataset": capacity_config["dataset"]["sha256"], "checkpoint": capacity_summary["checkpoint_sha256"]})
            append_experiment({"experiment_id": capacity_config["experiment_id"] + "_decision", "status": "capacity_passed", "config": {"path": str(capacity_config_path), "sha256": file_sha256(capacity_config_path)}, "metrics": decision, "protected_splits_accessed": False, "promotion": "none"})
            commit_stage("exp15 capacity gate", ["configs/learning/no_action_multi_capacity_exp15.json", "configs/learning/no_action_formal_seed29_exp16.json", "configs/learning/no_action_replication_exp17.json", "configs/learning/treechop_no_action_pipeline_exp15.json", "scripts/train_bounded_no_action.py", "scripts/audit_disabled_zero_checkpoint.py", "scripts/run_treechop_no_action_pipeline.py", "scripts/evaluate_recurrent_history_invariance.py", "scripts/evaluate_no_previous_action_ablation.py", "tests/test_bounded_no_action_pipeline.py", "artifacts/exp15", "artifacts/pipeline", "experiments/registry.jsonl"])
        else:
            capacity_summary = load_json(capacity_summary_path)

        formal_config_path = stage_configs["FORMAL_TRAIN_SEED29"]
        formal_config = load_json(formal_config_path)
        formal_summary_path = Path(formal_config["outputs"]["training_summary"])
        if "FORMAL_TRAIN_SEED29" not in executed:
            started = utc_now()
            command("FORMAL_TRAIN_SEED29", [sys.executable, "-m", "scripts.train_bounded_no_action", "--config", str(formal_config_path)], Path("artifacts/pipeline/formal_seed29.log"))
            executed.append("FORMAL_TRAIN_SEED29")
            state["training_runs"] = 2
            state.update(completed_stages=executed, next_state="RECORDED_REPLAY")
            atomic_json(state_path, state)
            stage_record(history_path, "FORMAL_TRAIN_SEED29", "passed", formal_config["experiment_id"], formal_config_path, started, "single bounded seed-29 run selected minimum validation CE checkpoint", "RECORDED_REPLAY", list(formal_config["outputs"].values()))
        formal_summary = load_json(formal_summary_path)
        formal_summaries.append(formal_summary)
        checkpoint = formal_summary["checkpoint"]
        if "RECORDED_REPLAY" not in executed:
            started = utc_now()
            command("RECORDED_REPLAY", [sys.executable, "-m", "scripts.evaluate_recurrent_history_invariance", "--config", str(formal_config_path), "--checkpoint", checkpoint, "--training-summary", str(formal_summary_path), "--output-root", "artifacts/exp16", "--invariance-output-name", "history_invariance_replay.json"], Path("artifacts/pipeline/formal_seed29.log"))
            executed.append("RECORDED_REPLAY")
            state.update(completed_stages=executed, next_state="AUTONOMOUS_SEED29")
            atomic_json(state_path, state)
            stage_record(history_path, "RECORDED_REPLAY", "passed", formal_config["experiment_id"], formal_config_path, started, "all five irrelevant action histories produced invariant outputs", "AUTONOMOUS_SEED29", ["artifacts/exp16/history_invariance_replay.json", "artifacts/exp16/recorded_observation_replay.csv"])
        if "AUTONOMOUS_SEED29" not in executed:
            started = utc_now()
            autonomous_command = [sys.executable, "-m", "scripts.evaluate_no_previous_action_ablation", "--config", str(formal_config_path), "--checkpoint", checkpoint, "--output-root", "artifacts/exp16", "--checkpoint-seed", "29"]
            if args.offline_gradle:
                autonomous_command.append("--offline-gradle")
            command("AUTONOMOUS_SEED29", autonomous_command, Path("artifacts/pipeline/autonomous_seed29.log"))
            summary = load_json(Path("artifacts/exp16/autonomous_summary.json"))
            decision = seed29_decision(summary)
            atomic_json(Path(formal_config["outputs"]["seed29_decision"]), decision)
            executed.append("AUTONOMOUS_SEED29")
            state["autonomous_episodes"] = 4
            state.update(completed_stages=executed, next_state="CONDITIONAL_REPLICATION" if decision["replication_eligible"] else "FINAL_DECISION")
            atomic_json(state_path, state)
            stage_record(history_path, "AUTONOMOUS_SEED29", "passed", formal_config["experiment_id"], formal_config_path, started, decision["classification"], state["next_state"], ["artifacts/exp16/autonomous_summary.json", "artifacts/exp16/autonomous_episode_summary.csv", formal_config["outputs"]["seed29_decision"]], {"checkpoint": formal_summary["checkpoint_sha256"]})
        else:
            decision = load_json(Path(formal_config["outputs"]["seed29_decision"]))
        append_experiment({"experiment_id": formal_config["experiment_id"] + "_decision", "status": decision["classification"], "config": {"path": str(formal_config_path), "sha256": file_sha256(formal_config_path)}, "checkpoint": {"path": checkpoint, "sha256": formal_summary["checkpoint_sha256"]}, "metrics": decision, "protected_splits_accessed": False, "promotion": "none"}) if not registry_has(formal_config["experiment_id"] + "_decision") else None

        if not decision["replication_eligible"]:
            skipped.extend(["CONDITIONAL_REPLICATION", "AGGREGATE_12_EPISODES"])
            final_decision = {"classification": decision["classification"], "exactly_one_recommendation": "Freeze this bounded no-action branch; do not train seeds 11/47 or start another actor ablation.", "protected_splits_accessed": False, "promotion": "none"}
            atomic_json(paths["decision"], final_decision)
            report(paths["report"], executed, skipped, final_decision, capacity_summary, formal_summaries, 4, correctness)
            state.update(status="STOPPED", next_state="STOPPED", final_decision=final_decision["classification"], completed_stages=executed)
            atomic_json(state_path, state)
            commit_stage("exp16 formal seed29 and evaluation", ["artifacts/exp16", "artifacts/pipeline", "reports/bounded_no_action_pipeline.md", "experiments/registry.jsonl"])
            print(json.dumps(state, indent=2), flush=True)
            return

        commit_stage("exp16 formal seed29 and evaluation", ["artifacts/exp16", "artifacts/pipeline", "experiments/registry.jsonl"])
        replication_config_path = stage_configs["CONDITIONAL_REPLICATION"]
        replication_config = load_json(replication_config_path)
        if "CONDITIONAL_REPLICATION" not in executed:
            started = utc_now()
            for training_seed in replication_config["training_seeds"]:
                output_root = "artifacts/exp17/seed{}".format(training_seed)
                log = Path("artifacts/pipeline/replication_seed{}.log".format(training_seed))
                command("REPLICATION_SEED{}".format(training_seed), [sys.executable, "-m", "scripts.train_bounded_no_action", "--config", str(replication_config_path), "--training-seed", str(training_seed), "--output-root", output_root], log)
                training_summary = Path(output_root) / "training_summary.json"
                trained = load_json(training_summary)
                formal_summaries.append(trained)
                command("REPLAY_SEED{}".format(training_seed), [sys.executable, "-m", "scripts.evaluate_recurrent_history_invariance", "--config", str(replication_config_path), "--checkpoint", trained["checkpoint"], "--training-summary", str(training_summary), "--output-root", output_root, "--invariance-output-name", "history_invariance_replay.json"], log)
                autonomous_command = [sys.executable, "-m", "scripts.evaluate_no_previous_action_ablation", "--config", str(replication_config_path), "--checkpoint", trained["checkpoint"], "--output-root", output_root, "--checkpoint-seed", str(training_seed)]
                if args.offline_gradle:
                    autonomous_command.append("--offline-gradle")
                command("AUTONOMOUS_SEED{}".format(training_seed), autonomous_command, log)
            final_decision = aggregate_replication()
            atomic_json(Path(replication_config["outputs"]["final_branch_decision"]), final_decision)
            executed.append("CONDITIONAL_REPLICATION")
            state["training_runs"] = 4
            state["autonomous_episodes"] = 12
            state.update(completed_stages=executed, next_state="FINAL_DECISION")
            atomic_json(state_path, state)
            stage_record(history_path, "CONDITIONAL_REPLICATION", "passed", replication_config["experiment_id"], replication_config_path, started, final_decision["classification"], "FINAL_DECISION", list(replication_config["outputs"].values()))
            append_experiment({"experiment_id": replication_config["experiment_id"] + "_decision", "status": final_decision["classification"], "config": {"path": str(replication_config_path), "sha256": file_sha256(replication_config_path)}, "metrics": final_decision, "protected_splits_accessed": False, "promotion": "none"})
        else:
            final_decision = load_json(Path(replication_config["outputs"]["final_branch_decision"]))
            for training_seed in (11, 47):
                formal_summaries.append(load_json(Path("artifacts/exp17/seed{}/training_summary.json".format(training_seed))))
        report(paths["report"], executed, skipped, final_decision, capacity_summary, formal_summaries, 12, correctness)
        atomic_json(paths["decision"], final_decision)
        state.update(status="COMPLETE", next_state="COMPLETE", final_decision=final_decision["classification"], completed_stages=executed)
        atomic_json(state_path, state)
        commit_stage("exp17 replication and aggregate decision", ["artifacts/exp17", "artifacts/pipeline", "reports/bounded_no_action_pipeline.md", "experiments/registry.jsonl"])
        print(json.dumps(state, indent=2), flush=True)
    except Exception as error:
        failure = {
            "timestamp": utc_now(), "stage": state.get("next_state"), "error": str(error),
            "protected_splits_accessed": False,
        }
        atomic_json(paths["failure"], failure)
        classification = "CORRECTNESS_BLOCKED" if state.get("next_state") == "PRECHECK" else "STOPPED"
        state.update(status=classification, exit_reason=str(error))
        atomic_json(state_path, state)
        final_decision = {"classification": "correctness_or_predeclared_gate_blocked", "exactly_one_recommendation": "Repair only the reported correctness/provenance failure, then resume the same frozen pipeline; do not substitute another intervention.", "protected_splits_accessed": False, "promotion": "none"}
        atomic_json(paths["decision"], final_decision)
        report(paths["report"], executed, skipped, final_decision, capacity_summary, formal_summaries, state.get("autonomous_episodes", 0), correctness)
        raise


if __name__ == "__main__":
    main()
