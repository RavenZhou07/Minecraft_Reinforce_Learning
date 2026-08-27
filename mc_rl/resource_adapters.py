"""POV-only resource adapters used by the generic candidate search policy."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import radians, tan
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from mc_rl.candidates import ResourceCandidate, ResourceDetection
from mc_rl.telemetry import VisualRangeEstimate


@dataclass(frozen=True)
class TreeDetection(ResourceDetection):
    """Trunk-component detection carrying its visual geometry.

    ``apparent_size`` keeps the canopy-support convention used by candidate
    merging, while the extra fields describe the trunk itself: bounding box,
    centroid, bottom-center, angular extents, and how much of the support is
    trunk versus leaves. ``sees_trunk`` distinguishes these from canopy-only
    observations.
    """

    center_y: float = 0.5
    bbox_left: Optional[float] = None
    bbox_right: Optional[float] = None
    bbox_top: Optional[float] = None
    bbox_bottom: Optional[float] = None
    bottom_center_x: Optional[float] = None
    angular_height_deg: Optional[float] = None
    angular_width_deg: Optional[float] = None
    trunk_fraction: Optional[float] = None
    sees_trunk: bool = False


@dataclass(frozen=True)
class TrunkView:
    """Single-frame trunk observation used by the contact controller."""

    present: bool
    center_x: float
    center_y: float
    bottom_y: float
    width_px: float
    height_px: float
    area_px: float
    crosshair_trunk_fraction: float
    horizontal_yaw: float
    vertical_offset_deg: float
    clipped_vertical: bool
    material: str = "unknown"


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
        vertical_fov_degrees: float = 70.0,
        minimum_component_pixels: int = 3,
        interaction_action_id: int = 8,
        interaction_size: Optional[float] = 150.0,
        reward_is_success: bool = True,
        range_scale: float = 41.55,
        range_exponent: float = -0.395,
        range_size_cap: Optional[float] = None,
        interaction_uses_geometry: bool = False,
        interaction_min_apparent_size: float = 0.0,
        assumed_trunk_height_blocks: float = 4.5,
        assumed_trunk_width_blocks: float = 1.0,
        trunk_view_min_height_px: int = 3,
    ):
        self.horizontal_fov_degrees = float(horizontal_fov_degrees)
        self.vertical_fov_degrees = float(vertical_fov_degrees)
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
        self.assumed_trunk_height_blocks = float(assumed_trunk_height_blocks)
        self.assumed_trunk_width_blocks = float(assumed_trunk_width_blocks)
        self.trunk_view_min_height_px = int(trunk_view_min_height_px)
        if self.range_scale <= 0 or self.range_exponent >= 0:
            raise ValueError("tree range calibration must have positive scale and negative exponent")
        if self.range_size_cap is not None and self.range_size_cap <= 0:
            raise ValueError("range_size_cap must be positive")
        if self.interaction_min_apparent_size < 0:
            raise ValueError("interaction_min_apparent_size cannot be negative")
        if self.assumed_trunk_height_blocks <= 0 or self.assumed_trunk_width_blocks <= 0:
            raise ValueError("assumed trunk dimensions must be positive")

    @staticmethod
    def oak_trunk_mask(pov: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def birch_trunk_mask(pov: np.ndarray) -> np.ndarray:
        """Return light birch bark that is locally supported by dark flecks.

        A plain low-saturation threshold also selects sky, stone, and clouds.
        Birch's dark bark marks provide a compact texture cue available in
        POV without block metadata. The small closing kernel reconnects the
        supported light pixels into the vertical component used downstream.
        """

        frame = np.asarray(pov)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("POV must have shape (height, width, 3)")
        rgb = frame.astype(np.int16)
        red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        light_bark = (
            (red >= 115)
            & (green >= 105)
            & (blue >= 85)
            & (red <= 235)
            & (green <= 225)
            & (blue <= 210)
            & ((np.maximum(np.maximum(red, green), blue)
                - np.minimum(np.minimum(red, green), blue)) <= 48)
        ).astype(np.uint8)
        dark_fleck = (
            (red < 105) & (green < 105) & (blue < 105)
        ).astype(np.uint8)
        nearby_fleck = cv2.dilate(
            dark_fleck, np.ones((5, 5), dtype=np.uint8), iterations=1
        )
        supported = light_bark & nearby_fleck
        return cv2.morphologyEx(
            supported,
            cv2.MORPH_CLOSE,
            np.ones((5, 3), dtype=np.uint8),
        ).astype(np.uint8)

    @classmethod
    def trunk_mask(cls, pov: np.ndarray) -> np.ndarray:
        return np.maximum(
            cls.oak_trunk_mask(pov), cls.birch_trunk_mask(pov)
        ).astype(np.uint8)

    @staticmethod
    def leaf_mask(pov: np.ndarray) -> np.ndarray:
        """Return the explainable green-leaf cue over the full POV.

        Candidate scale uses only the skyline subset below, but terminal
        contact needs the full mask to tell when the player has walked into a
        canopy and should deliberately clear a bounded amount of foliage.
        """

        frame = np.asarray(pov)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("POV must have shape (height, width, 3)")
        rgb = frame.astype(np.int16)
        red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        return (
            (green > red * 1.05)
            & (green > blue * 1.15)
            & (green < 150)
            & (red < 110)
        ).astype(np.uint8)

    def leaf_occlusion_fraction(self, pov: np.ndarray) -> float:
        """Leaf-like fraction immediately above the crosshair.

        Grass commonly fills the lower half of natural frames.  Measuring the
        24x24 patch ending at the vertical centre makes this an occluding-
        canopy cue rather than a generic green-pixel cue.
        """

        mask = self.leaf_mask(pov)
        height, width = mask.shape
        patch_size = max(8, min(height, width, 24))
        half_width = patch_size // 2
        center_y, center_x = height // 2, width // 2
        patch = mask[
            max(0, center_y - patch_size): center_y,
            max(0, center_x - half_width): min(
                width, center_x + half_width
            ),
        ]
        return 0.0 if patch.size == 0 else float(patch.mean())

    def detect(self, pov: np.ndarray) -> List[ResourceDetection]:
        mask = self.trunk_mask(pov)
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        height, width_axis = mask.shape
        leaf_mask = self.leaf_mask(pov)
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
            left = int(stats[component, cv2.CC_STAT_LEFT])
            top = int(stats[component, cv2.CC_STAT_TOP])
            width = int(stats[component, cv2.CC_STAT_WIDTH])
            # Candidate search should remember vertical trunk hypotheses, not
            # broad dirt or grass-block side faces. Contact applies a slightly
            # stricter form of the same geometric check below.
            if component_height < 0.65 * max(width, 1):
                continue
            center_x = float(centroids[component, 0])
            center_y = float(centroids[component, 1])
            normalized_x = center_x / max(width_axis - 1, 1)
            normalized_y = center_y / max(height - 1, 1)
            horizontal_yaw = (
                normalized_x - 0.5
            ) * self.horizontal_fov_degrees
            # Confidence expresses component support, while area remains a
            # separate score feature. It saturates quickly so size dominates
            # nearest-target selection rather than being counted twice.
            # A horizontal-only match is unsafe in natural terrain: a broad
            # skyline canopy can share x with an unrelated dirt-bank edge at
            # the bottom of the image.  Require the leaf component to meet
            # the *top* of this particular trunk component as it would for a
            # real tree silhouette.  Interval distance is used instead of
            # centroid distance because a broad canopy legitimately spans a
            # narrow trunk whose centre is far from the canopy centroid.
            trunk_right = left + width - 1

            def canopy_relation(index: int):
                leaf_left = int(leaf_stats[index, cv2.CC_STAT_LEFT])
                leaf_width = int(leaf_stats[index, cv2.CC_STAT_WIDTH])
                leaf_right = leaf_left + leaf_width - 1
                leaf_top = int(leaf_stats[index, cv2.CC_STAT_TOP])
                leaf_height = int(leaf_stats[index, cv2.CC_STAT_HEIGHT])
                leaf_bottom = leaf_top + leaf_height - 1
                horizontal_gap = max(
                    0, leaf_left - trunk_right, left - leaf_right
                )
                vertical_gap = max(0, top - leaf_bottom - 1)
                allowed_vertical_gap = max(
                    4, int(round(0.25 * component_height))
                )
                related = bool(
                    horizontal_gap <= 8
                    and vertical_gap <= allowed_vertical_gap
                    and leaf_top <= top + component_height - 1
                )
                return related, horizontal_gap + vertical_gap

            related_leaves = [
                index
                for index in leaf_components
                if canopy_relation(index)[0]
            ]
            nearest_leaf = min(
                related_leaves,
                key=lambda index: canopy_relation(index)[1],
                default=None,
            )
            leaf_area = (
                float(leaf_stats[nearest_leaf, cv2.CC_STAT_AREA])
                if nearest_leaf is not None
                else float(area)
            )
            has_canopy_support = nearest_leaf is not None
            confidence = min(1.0, 0.25 + max(area, leaf_area / 8.0) / 30.0)
            detections.append(
                TreeDetection(
                    resource_type=self.resource_type,
                    horizontal_yaw=horizontal_yaw,
                    confidence=confidence,
                    apparent_size=leaf_area,
                    center_x=normalized_x,
                    geometry_size=float(area),
                    center_y=normalized_y,
                    bbox_left=left / max(width_axis - 1, 1),
                    bbox_right=(left + width - 1) / max(width_axis - 1, 1),
                    bbox_top=top / max(height - 1, 1),
                    bbox_bottom=(top + component_height - 1) / max(height - 1, 1),
                    bottom_center_x=(left + (width - 1) / 2.0)
                    / max(width_axis - 1, 1),
                    angular_height_deg=(
                        component_height / max(height, 1)
                    ) * self.vertical_fov_degrees,
                    angular_width_deg=(
                        width / max(width_axis, 1)
                    ) * self.horizontal_fov_degrees,
                    trunk_fraction=(
                        area / (area + leaf_area)
                        if has_canopy_support
                        else 1.0
                    ),
                    sees_trunk=True,
                )
            )
        return sorted(detections, key=lambda item: -item.apparent_size)

    def _views_from_mask(
        self, mask: np.ndarray, material: str
    ) -> List[TrunkView]:
        """Extract vertical components without joining different materials."""

        height, width_axis = mask.shape
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        patch = mask[
            max(0, height // 2 - 4): height // 2 + 4,
            max(0, width_axis // 2 - 4): width_axis // 2 + 4,
        ]
        views = []
        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            component_height = int(stats[component, cv2.CC_STAT_HEIGHT])
            width = int(stats[component, cv2.CC_STAT_WIDTH])
            if (
                area < max(self.minimum_component_pixels, 4)
                or component_height < self.trunk_view_min_height_px
                or component_height < 0.8 * max(width, 1)
            ):
                continue
            left = int(stats[component, cv2.CC_STAT_LEFT])
            top = int(stats[component, cv2.CC_STAT_TOP])
            center_x = float(centroids[component, 0]) / max(
                width_axis - 1, 1
            )
            center_y = float(centroids[component, 1]) / max(height - 1, 1)
            views.append(
                TrunkView(
                    present=True,
                    center_x=center_x,
                    center_y=center_y,
                    bottom_y=(top + component_height - 1)
                    / max(height - 1, 1),
                    width_px=float(width),
                    height_px=float(component_height),
                    area_px=float(area),
                    crosshair_trunk_fraction=float(patch.mean()),
                    horizontal_yaw=(center_x - 0.5)
                    * self.horizontal_fov_degrees,
                    vertical_offset_deg=(center_y - 0.5)
                    * self.vertical_fov_degrees,
                    clipped_vertical=bool(
                        top <= 0 or top + component_height >= height
                    ),
                    material=material,
                )
            )
        return views

    def trunk_views(self, pov: np.ndarray) -> List[TrunkView]:
        """Return all plausible trunks, ranked by support and centre distance."""

        views = self._views_from_mask(self.oak_trunk_mask(pov), "oak")
        views.extend(self._views_from_mask(self.birch_trunk_mask(pov), "birch"))
        return sorted(
            views,
            key=lambda view: -view.area_px
            / (1.0 + 6.0 * abs(view.center_x - 0.5)),
        )

    def trunk_view(self, pov: np.ndarray) -> TrunkView:
        """Describe the most probable in-view trunk for the contact stage."""

        views = self.trunk_views(pov)
        if views:
            return views[0]
        return TrunkView(
            present=False,
            center_x=0.5,
            center_y=0.5,
            bottom_y=1.0,
            width_px=0.0,
            height_px=0.0,
            area_px=0.0,
            crosshair_trunk_fraction=0.0,
            horizontal_yaw=0.0,
            vertical_offset_deg=0.0,
            clipped_vertical=False,
        )

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
        """Estimate horizontal tree range, preferring trunk geometry.

        Trunk sightings measure the visible trunk's angular height (or width
        when the height is clipped by the frame) against an assumed block
        size, which survives canopy shape variety far better than the
        canopy-size power law. Canopy-only views keep the calibrated power
        law but carry a larger, explicitly low-confidence uncertainty so
        they can never outvote a trunk-based fix inside position fusion.
        """

        if isinstance(detection, TreeDetection) and detection.sees_trunk:
            angular_height = detection.angular_height_deg or 0.0
            angular_width = detection.angular_width_deg or 0.0
            trunk_fraction = (
                1.0
                if detection.trunk_fraction is None
                else float(detection.trunk_fraction)
            )
            clipped = bool(
                detection.bbox_top is not None
                and detection.bbox_bottom is not None
                and (
                    float(detection.bbox_top) <= 0.02
                    or float(detection.bbox_bottom) >= 0.98
                )
            )
            if not clipped and angular_height >= 8.0:
                distance = self.assumed_trunk_height_blocks / (
                    2.0 * tan(radians(angular_height) / 2.0)
                )
                occlusion_penalty = 0.15 * distance * (1.0 - trunk_fraction)
                uncertainty = max(
                    1.2, 0.16 * distance + 0.6 + occlusion_penalty
                )
                return VisualRangeEstimate(
                    distance=min(max(distance, 1.2), 24.0),
                    uncertainty=min(max(uncertainty, 1.2), 6.0),
                    basis="trunk_height",
                )
            if angular_width >= 2.5:
                distance = self.assumed_trunk_width_blocks / (
                    2.0 * tan(radians(angular_width) / 2.0)
                )
                uncertainty = max(1.8, 0.25 * distance + 1.0)
                return VisualRangeEstimate(
                    distance=min(max(distance, 1.2), 24.0),
                    uncertainty=min(max(uncertainty, 1.8), 7.0),
                    basis="trunk_width",
                )
        range_size = float(detection.apparent_size)
        if self.range_size_cap is not None:
            range_size = min(range_size, float(self.range_size_cap))
        distance = self.range_scale * range_size ** self.range_exponent
        edge_fraction = min(1.0, abs(float(detection.horizontal_yaw)) / 35.0)
        uncertainty = max(2.2, 0.22 * distance + 1.0 + 0.75 * edge_fraction)
        return VisualRangeEstimate(
            distance=distance,
            uncertainty=min(max(uncertainty, 2.2), 8.0),
            basis="canopy_size",
        )
