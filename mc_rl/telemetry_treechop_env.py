"""Local F3-enabled Treechop EnvSpec without patching MineRL sources."""

from typing import Any, Dict, List

import numpy as np

from mc_rl.envs import configure_minerl_runtime
from mc_rl.wrappers import DiscreteActionWrapper, OneLogTreechopWrapper


ENV_ID = "MineRLTreechopF3Local-v0"
RAYCAST_ENV_ID = "MineRLTreechopF3RaycastLocal-v0"


def build_telemetry_treechop_spec_class(include_raycast: bool = False):
    """Create local Treechop with F3 telemetry and optional diagnostic ray."""

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

    class _RayScalar(KeymapTranslationHandler):
        def __init__(
            self,
            hero_key: str,
            actor_key: str,
            low: float,
            high: float,
            default: float,
        ):
            super().__init__(
                hero_keys=["LineOfSight", hero_key],
                univ_keys=["raycast", actor_key],
                to_string=actor_key,
                default_if_missing=default,
                ignore_missing=True,
                space=spaces.Box(
                    low=low, high=high, shape=(), dtype=np.float32
                ),
            )

    class _RayPredicate(KeymapTranslationHandler):
        def __init__(self, hero_key: str, actor_key: str, accepted):
            self.accepted = frozenset(accepted)
            super().__init__(
                hero_keys=["LineOfSight", hero_key],
                univ_keys=["raycast", actor_key],
                to_string=actor_key,
                default_if_missing="",
                ignore_missing=True,
                space=spaces.Box(
                    low=0.0, high=1.0, shape=(), dtype=np.float32
                ),
            )

        def from_hero(self, hero_dict):
            raw = self.walk_dict(hero_dict, self.hero_keys)
            value = raw.item() if getattr(raw, "shape", ()) == () else raw
            return np.float32(str(value) in self.accepted)

    class DiagnosticRaycastObservation(TranslationHandlerGroup):
        def __init__(self):
            super().__init__(
                handlers=[
                    _RayPredicate("hitType", "has_block", ("block",)),
                    _RayPredicate("type", "is_log", ("log", "log2")),
                    _RayPredicate(
                        "type", "is_leaves", ("leaves", "leaves2")
                    ),
                    _RayScalar("inRange", "in_range", 0.0, 1.0, 0.0),
                    _RayScalar("distance", "distance", 0.0, 50.0, 50.0),
                    _RayScalar("x", "x", -640000.0, 640000.0, 0.0),
                    _RayScalar("y", "y", -640000.0, 640000.0, 0.0),
                    _RayScalar("z", "z", -640000.0, 640000.0, 0.0),
                ]
            )

        def to_string(self) -> str:
            return "raycast"

        def xml_template(self) -> str:
            return "<ObservationFromRay/>"

    class TelemetryTreechop(Treechop):
        def __init__(self):
            super().__init__(
                name=RAYCAST_ENV_ID if include_raycast else ENV_ID
            )

        def create_observables(self) -> List[Any]:
            observables = super().create_observables() + [
                F3SelfTelemetryObservation()
            ]
            if include_raycast:
                observables.append(DiagnosticRaycastObservation())
            return observables

        def is_from_folder(self, _folder: str) -> bool:
            return False

    return TelemetryTreechop


def make_telemetry_treechop_env(
    seed: int = 42,
    max_episode_steps: int = 300,
    include_raycast: bool = False,
):
    """Construct natural Treechop with discrete actions and one-log success."""

    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")
    configure_minerl_runtime()
    import minerl  # noqa: F401 - initializes the MineRL runtime

    spec_class = build_telemetry_treechop_spec_class(
        include_raycast=include_raycast
    )
    raw_env = spec_class().make()
    raw_env._is_fault_tolerant = False
    env = DiscreteActionWrapper(raw_env)
    env = OneLogTreechopWrapper(env, max_episode_steps=max_episode_steps)
    env.seed(int(seed))
    return env
