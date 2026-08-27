import numpy as np

from mc_rl.learning_observation import (
    LEGAL_RAW_KEYS,
    STUDENT_VECTOR_NAMES,
    TRAIN_ONLY_PRIVILEGED_KEYS,
    LegalObservationAdapter,
)


class AuditedObservation(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accessed = []

    def __getitem__(self, key):
        self.accessed.append(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.accessed.append(key)
        return super().get(key, default)


def observation(x=10.0, yaw=90.0, logs=0):
    return AuditedObservation(
        pov=np.zeros((64, 64, 3), dtype=np.uint8),
        telemetry={
            "x": x,
            "y": 64.0,
            "z": -5.0,
            "yaw": yaw,
            "pitch": 10.0,
            "biome_id": 4,
            "biome_temperature": 0.7,
            "biome_rainfall": 0.8,
        },
        inventory={"log": logs, "log2": 0},
        raycast={"is_log": 1, "x": 99.0},
        nearest_tree_xyz=np.array([99.0, 64.0, 99.0]),
    )


def test_legal_adapter_emits_fixed_schema_without_privileged_access():
    raw = observation()
    adapted = LegalObservationAdapter(500).reset(raw)
    assert adapted.pov.shape == (64, 64, 3)
    assert adapted.vector.shape == (len(STUDENT_VECTOR_NAMES),)
    assert set(raw.accessed).issubset(set(LEGAL_RAW_KEYS))
    assert not set(raw.accessed) & set(TRAIN_ONLY_PRIVILEGED_KEYS)


def test_legal_adapter_uses_relative_pose_and_inventory_only():
    adapter = LegalObservationAdapter(500)
    adapter.reset(observation())
    second = adapter.adapt(observation(x=12.0, yaw=135.0, logs=1), 25)
    assert np.isclose(second.vector[0], 2.0 / 16.0)
    assert np.isclose(second.vector[3], 2.0 / 2.0)
    assert np.isclose(second.vector[9], 1.0)
    assert np.isclose(second.vector[14], 0.25)
    assert np.isclose(second.vector[15], 0.05)
    assert second.inventory_log_count == 1
