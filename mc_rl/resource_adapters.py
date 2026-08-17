"""POV-only resource adapters used by the generic candidate search policy."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from mc_rl.candidates import ResourceCandidate, ResourceDetection
from mc_rl.telemetry import VisualRangeEstimate


class ResourceAdapter(ABC):
    resource_type = "resource"

    @abstractmethod
    def detect(self, pov: np.ndarray) -> List[ResourceDetection]:
        pass

    def candidate_score(self, candidate: ResourceCandidate) -> float:
        return 0.0

    @abstractmethod
    def interaction_action(self) -> int:
        pass

    @abstractmethod
    def success(
        self, observation: Dict[str, Any], reward: float, info: Dict[str, Any]
    ) -> bool:
        pass

    def ready_to_interact(self, detection: ResourceDetection) -> bool:
        return False

    def estimate_range(
        self, detection: ResourceDetection
    ) -> Optional[VisualRangeEstimate]:
        """Return a POV-derived range estimate, if this resource supports it."""

        return None


class TreeResourceAdapter(ResourceAdapter):
    """Detect Minecraft trunks with a small explainable RGB component mask.

    The thresholds were checked against the existing distance-3--10 dataset.
    Trunk component area correlates strongly with inverse oracle distance, but
    only RGB pixels are read at deployment time.
    """

    resource_type = "tree"

    def __init__(
        self,
        horizontal_fov_degrees: float = 70.0,
        minimum_component_pixels: int = 3,
        interaction_action_id: int = 8,
        interaction_size: Optional[float] = 150.0,
        reward_is_success: bool = True,
        range_scale: float = 41.55,
        range_exponent: float = -0.395,
        range_size_cap: Optional[float] = None,
        interaction_uses_geometry: bool = False,
        interaction_min_apparent_size: float = 0.0,
    ):
        self.horizontal_fov_degrees = float(horizontal_fov_degrees)
        self.minimum_component_pixels = int(minimum_component_pixels)
        self.interaction_action_id = int(interaction_action_id)
        self.interaction_size = interaction_size
        self.reward_is_success = bool(reward_is_success)
        self.range_scale = float(range_scale)
        self.range_exponent = float(range_exponent)
        self.range_size_cap = range_size_cap
        self.interaction_uses_geometry = bool(interaction_uses_geometry)
        self.interaction_min_apparent_size = float(
            interaction_min_apparent_size
        )
        if self.range_scale <= 0 or self.range_exponent >= 0:
            raise ValueError("tree range calibration must have positive scale and negative exponent")
        if self.range_size_cap is not None and self.range_size_cap <= 0:
            raise ValueError("range_size_cap must be positive")
        if self.interaction_min_apparent_size < 0:
            raise ValueError("interaction_min_apparent_size cannot be negative")

    @staticmethod
    def trunk_mask(pov: np.ndarray) -> np.ndarray:
        frame = np.asarray(pov)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("POV must have shape (height, width, 3)")
        rgb = frame.astype(np.int16)
        mask = (
            (rgb[:, :, 0] > rgb[:, :, 1] * 1.15)
            & (rgb[:, :, 1] > rgb[:, :, 2] * 1.10)
            & (rgb[:, :, 0] > 45)
            & (rgb[:, :, 1] > 25)
            & (rgb[:, :, 2] < 110)
        )
        return mask.astype(np.uint8)

    def detect(self, pov: np.ndarray) -> List[ResourceDetection]:
        mask = self.trunk_mask(pov)
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        height, width = mask.shape
        rgb = np.asarray(pov).astype(np.int16)
        red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        leaf_mask = (
            (green > red * 1.05)
            & (green > blue * 1.15)
            & (green < 150)
            & (red < 110)
        ).astype(np.uint8)
        # Grass occupies the lower half of the flat curriculum. Restricting
        # scale support to the skyline isolates leaf canopies and is strongly
        # correlated with inverse distance in the existing 3--10 dataset.
        leaf_mask[min(30, height) :, :] = 0
        leaf_count, _leaf_labels, leaf_stats, leaf_centroids = (
            cv2.connectedComponentsWithStats(leaf_mask, connectivity=8)
        )
        leaf_components = [
            index
            for index in range(1, leaf_count)
            if int(leaf_stats[index, cv2.CC_STAT_AREA]) >= 2
        ]
        detections = []
        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            component_height = int(stats[component, cv2.CC_STAT_HEIGHT])
            if area < self.minimum_component_pixels or component_height < 2:
                continue
            center_x = float(centroids[component, 0])
            normalized_x = center_x / max(width - 1, 1)
            horizontal_yaw = (
                normalized_x - 0.5
            ) * self.horizontal_fov_degrees
            # Confidence expresses component support, while area remains a
            # separate score feature. It saturates quickly so size dominates
            # nearest-target selection rather than being counted twice.
            nearest_leaf = min(
                leaf_components,
                key=lambda index: abs(float(leaf_centroids[index, 0]) - center_x),
                default=None,
            )
            leaf_area = (
                float(leaf_stats[nearest_leaf, cv2.CC_STAT_AREA])
                if nearest_leaf is not None
                and abs(float(leaf_centroids[nearest_leaf, 0]) - center_x) <= 18.0
                else float(area)
            )
            confidence = min(1.0, 0.25 + max(area, leaf_area / 8.0) / 30.0)
            detections.append(
                ResourceDetection(
                    resource_type=self.resource_type,
                    horizontal_yaw=horizontal_yaw,
                    confidence=confidence,
                    apparent_size=leaf_area,
                    center_x=normalized_x,
                    geometry_size=float(area),
                )
            )
        return sorted(detections, key=lambda item: -item.apparent_size)

    def interaction_action(self) -> int:
        return self.interaction_action_id

    def success(
        self, observation: Dict[str, Any], reward: float, info: Dict[str, Any]
    ) -> bool:
        return bool(
            info.get("success", False) or (self.reward_is_success and reward > 0)
        )

    def ready_to_interact(self, detection: ResourceDetection) -> bool:
        interaction_size = (
            detection.geometry_size
            if self.interaction_uses_geometry
            and detection.geometry_size is not None
            else detection.apparent_size
        )
        return (
            self.interaction_size is not None
            and interaction_size >= self.interaction_size
            and detection.apparent_size >= self.interaction_min_apparent_size
        )

    def estimate_range(
        self, detection: ResourceDetection
    ) -> Optional[VisualRangeEstimate]:
        """Estimate horizontal tree range from canopy support.

        The power law was fit once on the existing distance-3--10 arena data.
        A conservative uncertainty floor prevents the map from treating this
        noisy visual measurement as an exact block coordinate.
        """

        range_size = float(detection.apparent_size)
        if self.range_size_cap is not None:
            range_size = min(range_size, float(self.range_size_cap))
        distance = self.range_scale * range_size ** self.range_exponent
        edge_fraction = min(1.0, abs(float(detection.horizontal_yaw)) / 35.0)
        uncertainty = max(1.25, 0.12 * distance + 0.75 * edge_fraction)
        return VisualRangeEstimate(distance=distance, uncertainty=uncertainty)
