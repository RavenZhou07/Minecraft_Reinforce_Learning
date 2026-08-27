"""Data-closed-loop visual attack permission gate for natural Treechop BC v2b."""

from mc_rl.natural_attack_gate_bc import NaturalAttackGatePolicy


MODEL_VERSION = "natural_treechop_attack_gate_bc_v2b"


class NaturalAttackGateV2Policy(NaturalAttackGatePolicy):
    """The v2a visual model with an auditable temporal confirmation contract."""

    def __init__(
        self,
        feature_size: int = 10,
        frame_stack: int = 4,
        include_centre_pixels: bool = False,
        decision_threshold: float = 0.5,
        attack_confirmation_frames: int = 2,
    ):
        super().__init__(
            feature_size=feature_size,
            frame_stack=frame_stack,
            include_centre_pixels=include_centre_pixels,
            decision_threshold=decision_threshold,
        )
        self.model_version = MODEL_VERSION
        self.attack_confirmation_frames = int(attack_confirmation_frames)
        if self.attack_confirmation_frames < 1:
            raise ValueError("attack confirmation frames must be at least one")
