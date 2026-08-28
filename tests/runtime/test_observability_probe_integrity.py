import numpy as np
import pytest

from mc_rl.observability_audit import (
    TrainOnlyPCA,
    TrainOnlyStandardizer,
    assert_disjoint_episode_splits,
)


def test_probe_split_rejects_episode_seed_leakage():
    assert_disjoint_episode_splits([18200, 18201], [18300, 18301])
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint_episode_splits([18200, 18201], [18201, 18300])


def test_scaler_and_pca_can_only_fit_bc_train():
    values = np.arange(60, dtype=np.float32).reshape(12, 5)
    with pytest.raises(ValueError, match="bc_train"):
        TrainOnlyStandardizer().fit(values, "bc_validation")
    with pytest.raises(ValueError, match="bc_train"):
        TrainOnlyPCA(3).fit(values, "student_dev")

    scaler = TrainOnlyStandardizer().fit(values, "bc_train")
    pca = TrainOnlyPCA(3).fit(scaler.transform(values), "bc_train")
    transformed = pca.transform(scaler.transform(values[:2]))
    assert scaler.fit_split == "bc_train"
    assert pca.fit_split == "bc_train"
    assert transformed.shape == (2, 3)
