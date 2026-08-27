import json

import pytest

from mc_rl.experiments import seeds_for_split


def test_seed_manifest_splits_are_disjoint_and_expected_sizes():
    names = (
        "teacher_dev",
        "teacher_holdout",
        "bc_train",
        "bc_validation",
        "dagger_rollout",
        "student_dev",
        "student_holdout",
    )
    splits = {name: set(seeds_for_split(name)) for name in names}
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            assert not splits[first] & splits[second]
    assert len(splits["bc_train"]) == 64
    assert len(splits["student_holdout"]) == 64


def test_final_test_is_protected_by_default():
    with pytest.raises(PermissionError):
        seeds_for_split("final_test")
    assert len(seeds_for_split("final_test", allow_final_test=True)) == 100
