"""Visual progress and stall diagnostics for resource approaches."""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np


@dataclass
class ProgressSample:
    forward: bool
    apparent_size: Optional[float]
    alignment_error: Optional[float]
    frame_change: float
    visible: bool


class VisualProgressMonitor:
    def __init__(
        self,
        window_size: int = 20,
        minimum_forward_steps: int = 12,
        minimum_size_growth: float = 0.08,
        minimum_alignment_improvement: float = 2.0,
        low_frame_change: float = 0.025,
        lost_steps: int = 6,
    ):
        if window_size < 3 or not 1 <= minimum_forward_steps <= window_size:
            raise ValueError("invalid progress window configuration")
        self.window_size = int(window_size)
        self.minimum_forward_steps = int(minimum_forward_steps)
        self.minimum_size_growth = float(minimum_size_growth)
        self.minimum_alignment_improvement = float(minimum_alignment_improvement)
        self.low_frame_change = float(low_frame_change)
        self.lost_steps = int(lost_steps)
        self.samples: Deque[ProgressSample] = deque(maxlen=self.window_size)
        self.consecutive_lost = 0
        self.last_diagnostics: Dict[str, float] = {}

    def reset(self) -> None:
        self.samples.clear()
        self.consecutive_lost = 0
        self.last_diagnostics = {}

    @staticmethod
    def frame_change(previous: Optional[np.ndarray], current: np.ndarray) -> float:
        if previous is None:
            return 1.0
        first = np.asarray(previous, dtype=np.float32)
        second = np.asarray(current, dtype=np.float32)
        if first.shape != second.shape:
            raise ValueError("progress frames must share a shape")
        return float(np.mean(np.abs(first - second)) / 255.0)

    def add(
        self,
        forward: bool,
        apparent_size: Optional[float],
        alignment_error: Optional[float],
        frame_change: float,
        visible: bool,
    ) -> None:
        self.samples.append(
            ProgressSample(
                bool(forward), apparent_size, alignment_error, float(frame_change), bool(visible)
            )
        )
        self.consecutive_lost = 0 if visible else self.consecutive_lost + 1

    @staticmethod
    def _split_medians(values):
        if len(values) < 2:
            return 0.0, 0.0
        split = max(1, len(values) // 3)
        return float(np.median(values[:split])), float(np.median(values[-split:]))

    def is_stalled(self) -> bool:
        if len(self.samples) < self.window_size:
            return False
        samples = list(self.samples)
        forward_samples = [sample for sample in samples if sample.forward]
        sizes = [sample.apparent_size for sample in samples if sample.apparent_size is not None]
        alignments = [
            abs(sample.alignment_error)
            for sample in samples
            if sample.alignment_error is not None
        ]
        first_size, last_size = self._split_medians(sizes)
        size_growth = (
            (last_size - first_size) / max(first_size, 1e-6) if sizes else -1.0
        )
        first_alignment, last_alignment = self._split_medians(alignments)
        alignment_improvement = first_alignment - last_alignment
        median_change = float(
            np.median([sample.frame_change for sample in forward_samples])
        ) if forward_samples else 1.0
        visible_fraction = float(np.mean([sample.visible for sample in samples]))
        self.last_diagnostics = {
            "forward_steps": float(len(forward_samples)),
            "size_growth": float(size_growth),
            "alignment_improvement": float(alignment_improvement),
            "median_frame_change": median_change,
            "visible_fraction": visible_fraction,
        }
        return bool(
            len(forward_samples) >= self.minimum_forward_steps
            and size_growth < self.minimum_size_growth
            and alignment_improvement < self.minimum_alignment_improvement
            and (
                median_change < self.low_frame_change
                or visible_fraction < 0.5
                or size_growth <= 0.0
            )
        )

    def target_lost(self) -> bool:
        return self.consecutive_lost >= self.lost_steps

