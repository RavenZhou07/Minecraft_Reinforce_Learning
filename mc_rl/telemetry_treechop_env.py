"""Local F3-enabled Treechop EnvSpec without patching MineRL sources."""

from typing import Any, Dict, List

import numpy as np

from mc_rl.envs import configure_minerl_runtime
from mc_rl.wrappers import DiscreteActionWrapper, OneLogTreechopWrapper


ENV_ID = "MineRLTreechopF3Local-v0"


def build_telemetry_treechop_spec_class():
    """Create a Treechop subclass that exposes only F3-like self telemetry."""

    from minerl.herobraine.env_specs.treechop_specs import Treechop
    from minerl.herobraine.hero import spaces
    from minerl.herobraine.hero.handlers.translation import (
        KeymapTranslationHandler,
        TranslationHandlerGroup,
    )

    class _TelemetryScalar(KeymapTranslationHandler):
        def __init__(
            self, hero_key: str, actor_key: str, low: float, high: float
        ):
            super().__init__(
                hero_keys=[hero_key],
                univ_keys=["telemetry", actor_key],
                to_string=actor_key,
                space=spaces.Box(
                    low=low, high=high, shape=(), dtype=np.float32
                ),
            )

    class F3SelfTelemetryObservation(TranslationHandlerGroup):
        def __init__(self):
            super().__init__(
                handlers=[
                    _TelemetryScalar("xpos", "x", -640000.0, 640000.0),
                    _TelemetryScalar("ypos", "y", -640000.0, 640000.0),
                    _TelemetryScalar("zpos", "z", -640000.0, 640000.0),
                    _TelemetryScalar("yaw", "yaw", -100000.0, 100000.0),
                    _TelemetryScalar("pitch", "pitch", -180.0, 180.0),
                    _TelemetryScalar("biome_id", "biome_id", -1.0, 255.0),
                    _TelemetryScalar(
                        "biome_temperature", "biome_temperature", -10.0, 10.0
                    ),
                    _TelemetryScalar(
                        "biome_rainfall", "biome_rainfall", 0.0, 10.0
                    ),
                ]
            )

        def to_string(self) -> str:
            return "telemetry"

        def xml_template(self) -> str:
            return "<ObservationFromFullStats/>"

    class TelemetryTreechop(Treechop):
        def __init__(self):
            super().__init__(name=ENV_ID)

        def create_observables(self) -> List[Any]:
            return super().create_observables() + [
                F3SelfTelemetryObservation()
            ]

        def is_from_folder(self, _folder: str) -> bool:
            return False

    return TelemetryTreechop


def make_telemetry_treechop_env(
    seed: int = 42, max_episode_steps: int = 300
):
    """Construct natural Treechop with discrete actions and one-log success."""

    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    configure_minerl_runtime()
    import minerl  # noqa: F401 - initializes the MineRL runtime

    spec_class = build_telemetry_treechop_spec_class()
    raw_env = spec_class().make()
    raw_env._is_fault_tolerant = False
    env = DiscreteActionWrapper(raw_env)
    env = OneLogTreechopWrapper(env, max_episode_steps=max_episode_steps)
    env.seed(int(seed))
    return env
