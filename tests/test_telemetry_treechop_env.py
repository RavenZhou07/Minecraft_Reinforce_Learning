import numpy as np

from mc_rl.telemetry_treechop_env import (
    ENV_ID,
    build_telemetry_treechop_spec_class,
)


def test_local_treechop_spec_adds_full_stats_without_third_party_patch():
    spec = build_telemetry_treechop_spec_class()()
    xml = spec.to_xml()
    assert spec.name == ENV_ID
    assert "<ObservationFromFullStats/>" in xml
    assert {handler.to_string() for handler in spec.observables} >= {
        "pov",
        "telemetry",
    }

    handler = next(
        observable
        for observable in spec.observables
        if observable.to_string() == "telemetry"
    )
    translated = handler.from_hero(
        {
            "xpos": 1.5,
            "ypos": 64.0,
            "zpos": -2.5,
            "yaw": 370.0,
            "pitch": 5.0,
            "biome_id": 4,
            "biome_temperature": 0.7,
            "biome_rainfall": 0.8,
        }
    )
    assert np.isclose(translated["x"], 1.5)
    assert int(translated["biome_id"]) == 4
