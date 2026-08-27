"""Seed-manifest and append-only experiment-registry utilities."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_SEED_MANIFEST = "configs/seeds/natural_treechop_v1.json"
DEFAULT_REGISTRY = "experiments/registry.jsonl"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_seed_manifest(path: str = DEFAULT_SEED_MANIFEST) -> Dict[str, Any]:
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "splits" not in manifest or "final_test" not in manifest["splits"]:
        raise ValueError("seed manifest is missing required splits")
    return manifest


def seeds_for_split(
    split: str,
    manifest_path: str = DEFAULT_SEED_MANIFEST,
    limit: Optional[int] = None,
    allow_final_test: bool = False,
) -> List[int]:
    manifest = load_seed_manifest(manifest_path)
    if split not in manifest["splits"]:
        raise KeyError("unknown seed split: {}".format(split))
    specification = manifest["splits"][split]
    if specification.get("protected", False) and not allow_final_test:
        raise PermissionError(
            "protected final_test requires explicit project-owner approval"
        )
    start = int(specification["seed_start"])
    count = int(specification["count"])
    seeds = list(range(start, start + count))
    if limit is not None:
        if int(limit) <= 0 or int(limit) > len(seeds):
            raise ValueError("limit must be within the declared split")
        seeds = seeds[: int(limit)]
    return seeds


def git_state() -> Dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git"] + list(args), text=True, encoding="utf-8"
        ).strip()

    commit = run("rev-parse", "HEAD")
    dirty = bool(run("status", "--porcelain"))
    return {"commit": commit, "dirty": dirty}


def append_experiment(
    record: Dict[str, Any], registry_path: str = DEFAULT_REGISTRY
) -> Dict[str, Any]:
    output = Path(registry_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
    payload.setdefault("git", git_state())
    encoded = json.dumps(payload, sort_keys=True)
    existing = output.read_text(encoding="utf-8") if output.exists() else ""
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(existing + encoded + "\n", encoding="utf-8")
    temporary.replace(output)
    return payload
