import numpy as np

from mc_rl.telemetry_treechop_env import (
    ENV_ID,
    RAYCAST_ENV_ID,
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


def test_diagnostic_treechop_spec_translates_line_of_sight_without_grid():
    spec = build_telemetry_treechop_spec_class(include_raycast=True)()
    xml = spec.to_xml()
    assert spec.name == RAYCAST_ENV_ID
    assert "<ObservationFromRay/>" in xml
    assert "ObservationFromGrid" not in xml

    handler = next(
        observable
        for observable in spec.observables
        if observable.to_string() == "raycast"
    )
    translated = handler.from_hero(
        {
            "LineOfSight": {
                "hitType": "block",
                "type": "log",
                "inRange": True,
                "distance": 3.25,
                "x": 10.5,
                "y": 65.2,
                "z": -4.5,
            }
        }
    )
    assert float(translated["has_block"]) == 1.0
    assert float(translated["is_log"]) == 1.0
    assert float(translated["is_leaves"]) == 0.0
    assert float(translated["in_range"]) == 1.0
    assert np.isclose(translated["distance"], 3.25)
