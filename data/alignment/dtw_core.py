# -*- coding: utf-8 -*-
"""DTW-based alignment core for OCR-GT sequence matching.

Spec Reference: L2W1-DE-001 (Data Engine - Alignment Core)

This module implements Dynamic Time Warping (DTW) based forced alignment
to solve the length mismatch between OCR predictions and Ground Truth,
with Gap Interpolation for handling deletion errors.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastdtw import fastdtw


class AlignmentType(Enum):
    """Alignment result types as defined in L2W1-DE-001."""

    MATCH = "MATCH"
    SUBSTITUTION = "SUBSTITUTION"
    DELETION_RECOVERY = "DELETION_RECOVERY"  # Gap interpolated
    INSERTION = "INSERTION"  # OCR hallucinated extra char
    IGNORE = "IGNORE"  # Low confidence, excluded from training


@dataclass
class AlignedPair:
    """A single aligned character pair with bounding box.

    Attributes:
        gt_char: Ground truth character.
        pred_char: OCR predicted character (empty string if deletion).
        box: Bounding box [x1, y1, x2, y2], may be interpolated.
        conf: OCR confidence score (0.0 for interpolated boxes).
        align_type: Type of alignment result.
    """

    gt_char: str
    pred_char: str
    box: List[float]
    conf: float
    align_type: AlignmentType


class DTWAligner:
    """DTW-based aligner for OCR results and Ground Truth text.

    Implements the core alignment algorithm defined in L2W1-DE-001:
    1. Build cost matrix using character similarity
    2. Find optimal path via fastdtw
    3. Interpolate bounding boxes for deletion errors (gap interpolation)

    Attributes:
        min_conf_threshold: Minimum confidence to trust OCR result (Gate 2).
        min_box_width: Minimum valid box width in pixels (Gate 3).

    Example:
        >>> aligner = DTWAligner()
        >>> ocr_results = [
        ...     {'char': '阿', 'box': [0, 0, 20, 30], 'conf': 0.99},
        ...     {'char': '西', 'box': [40, 0, 60, 30], 'conf': 0.95},
        ... ]
        >>> gt = "阿莫西"
        >>> aligned = aligner.align(ocr_results, gt)
    """

    def __init__(
        self,
        min_conf_threshold: float = 0.5,
        min_box_width: float = 8.0,
    ) -> None:
        """Initialize DTWAligner.

        Args:
            min_conf_threshold: OCR results with conf < threshold and wrong char
                will be marked as IGNORE (L2W1-DE-001 Gate 2).
            min_box_width: Interpolated boxes with width < this value will be
                discarded (L2W1-DE-001 Gate 3).
        """
        self.min_conf_threshold = min_conf_threshold
        self.min_box_width = min_box_width

    def align(
        self,
        ocr_results: List[Dict],
        ground_truth: str,
    ) -> List[AlignedPair]:
        """Perform DTW alignment between OCR results and ground truth.

        Args:
            ocr_results: List of OCR detections, each containing:
                - 'char': Detected character (str)
                - 'box': Bounding box [x1, y1, x2, y2] (List[float])
                - 'conf': Confidence score (float, 0-1)
            ground_truth: The ground truth text string.

        Returns:
            List of AlignedPair objects, one for each GT character.

        Raises:
            ValueError: If ocr_results is empty but ground_truth is not,
                or if data format is invalid.
        """
        # === Validation ===
        if not ground_truth:
            return []

        if not ocr_results:
            raise ValueError(
                f"OCR results empty but GT has {len(ground_truth)} chars. "
                "Cannot perform alignment without any OCR detections."
            )

        self._validate_ocr_format(ocr_results)

        # === Step 1 & 2: DTW Path Finding ===
        ocr_chars = [r["char"] for r in ocr_results]
        path = self._compute_dtw_path(ocr_chars, ground_truth)

        # === Step 3: Build alignment with Gap Interpolation ===
        aligned_pairs = self._build_alignment(
            ocr_results=ocr_results,
            ground_truth=ground_truth,
            path=path,
        )

        return aligned_pairs

    def _validate_ocr_format(self, ocr_results: List[Dict]) -> None:
        """Validate OCR result format.

        Args:
            ocr_results: List of OCR detection dictionaries.

        Raises:
            ValueError: If required keys are missing or format is invalid.
        """
        required_keys = {"char", "box", "conf"}
        for i, result in enumerate(ocr_results):
            missing = required_keys - set(result.keys())
            if missing:
                raise ValueError(
                    f"OCR result at index {i} missing keys: {missing}. Got: {result}"
                )
            if len(result["box"]) != 4:
                raise ValueError(
                    f"OCR result at index {i} has invalid box format. "
                    f"Expected [x1, y1, x2, y2], got: {result['box']}"
                )

    def _char_distance(self, char_a: str, char_b: str) -> float:
        """Compute distance between two characters.

        As per L2W1-DE-001 Step 1:
        Cost(i, j) = 1 - Similarity(char_ocr[i], char_gt[j])

        Args:
            char_a: First character.
            char_b: Second character.

        Returns:
            Distance value: 0.0 if identical, 1.0 otherwise.

        Note:
            Future enhancement: Could use glyph-based similarity for
            visually similar characters (e.g., '己' vs '已').
        """
        # Simple exact match for now; can be extended with glyph similarity
        return 0.0 if char_a == char_b else 1.0

    def _compute_dtw_path(
        self,
        ocr_chars: List[str],
        gt_chars: str,
    ) -> List[Tuple[int, int]]:
        """Compute optimal DTW alignment path.

        Args:
            ocr_chars: List of OCR detected characters. Shape: [N]
            gt_chars: Ground truth string. Shape: [M]

        Returns:
            List of (ocr_idx, gt_idx) tuples representing the alignment path.
        """
        # Convert to numpy arrays for fastdtw
        # fastdtw expects sequences that can compute element-wise distance
        # We use indices and provide custom distance function
        ocr_indices = np.arange(len(ocr_chars)).reshape(-1, 1)  # [N, 1]
        gt_indices = np.arange(len(gt_chars)).reshape(-1, 1)  # [M, 1]

        def index_distance(idx_a: np.ndarray, idx_b: np.ndarray) -> float:
            """Distance function using character indices."""
            i, j = int(idx_a[0]), int(idx_b[0])
            return self._char_distance(ocr_chars[i], gt_chars[j])

        # fastdtw returns (distance, path)
        _, path = fastdtw(ocr_indices, gt_indices, dist=index_distance)

        return path

    def _build_alignment(
        self,
        ocr_results: List[Dict],
        ground_truth: str,
        path: List[Tuple[int, int]],
    ) -> List[AlignedPair]:
        """Build final alignment with gap interpolation.

        This implements L2W1-DE-001 Step 3: Gap Interpolation for deletions.

        Args:
            ocr_results: Original OCR detection list.
            ground_truth: Ground truth string.
            path: DTW alignment path as (ocr_idx, gt_idx) tuples.

        Returns:
            List of AlignedPair, one per GT character.
        """
        n_gt = len(ground_truth)

        # Initialize result array - one entry per GT character
        # We'll fill this based on DTW path
        gt_to_ocr_mapping: List[Optional[int]] = [None] * n_gt

        # Build mapping from GT index to OCR index
        # DTW path may have multiple OCR chars mapping to one GT char (or vice versa)
        # We take the first valid mapping for each GT index
        for ocr_idx, gt_idx in path:
            if gt_to_ocr_mapping[gt_idx] is None:
                gt_to_ocr_mapping[gt_idx] = ocr_idx

        # Find indices of matched (non-None) GT characters for interpolation anchors
        matched_indices = [
            i for i, ocr_idx in enumerate(gt_to_ocr_mapping) if ocr_idx is not None
        ]

        # Build final aligned pairs
        aligned_pairs: List[AlignedPair] = []

        for gt_idx in range(n_gt):
            gt_char = ground_truth[gt_idx]
            ocr_idx = gt_to_ocr_mapping[gt_idx]

            if ocr_idx is not None:
                # === Matched or Substituted ===
                ocr_result = ocr_results[ocr_idx]
                pred_char = ocr_result["char"]
                conf = ocr_result["conf"]
                box = list(ocr_result["box"])  # Copy to avoid mutation

                # Determine alignment type
                if pred_char == gt_char:
                    align_type = AlignmentType.MATCH
                else:
                    # Gate 2: Low confidence wrong char -> IGNORE
                    if conf < self.min_conf_threshold:
                        align_type = AlignmentType.IGNORE
                    else:
                        align_type = AlignmentType.SUBSTITUTION

                aligned_pairs.append(
                    AlignedPair(
                        gt_char=gt_char,
                        pred_char=pred_char,
                        box=box,
                        conf=conf,
                        align_type=align_type,
                    )
                )
            else:
                # === Deletion: Need Gap Interpolation ===
                interpolated_box = self._interpolate_gap_box(
                    gt_idx=gt_idx,
                    gt_to_ocr_mapping=gt_to_ocr_mapping,
                    matched_indices=matched_indices,
                    ocr_results=ocr_results,
                    n_gt=n_gt,
                )

                if interpolated_box is None:
                    # Gate 3 failed or no anchors available
                    aligned_pairs.append(
                        AlignedPair(
                            gt_char=gt_char,
                            pred_char="",
                            box=[0.0, 0.0, 0.0, 0.0],
                            conf=0.0,
                            align_type=AlignmentType.IGNORE,
                        )
                    )
                else:
                    aligned_pairs.append(
                        AlignedPair(
                            gt_char=gt_char,
                            pred_char="",
                            box=interpolated_box,
                            conf=0.0,
                            align_type=AlignmentType.DELETION_RECOVERY,
                        )
                    )

        return aligned_pairs

    def _interpolate_gap_box(
        self,
        gt_idx: int,
        gt_to_ocr_mapping: List[Optional[int]],
        matched_indices: List[int],
        ocr_results: List[Dict],
        n_gt: int,
    ) -> Optional[List[float]]:
        """Interpolate bounding box for a deleted character.

        Implements L2W1-DE-001 Gap Interpolation:
        - Find C_prev (previous matched char) and C_next (next matched char)
        - Linearly interpolate box based on relative position

        Args:
            gt_idx: Index of the deleted GT character.
            gt_to_ocr_mapping: Mapping from GT indices to OCR indices.
            matched_indices: Sorted list of GT indices that have OCR matches.
            ocr_results: Original OCR results.
            n_gt: Total number of GT characters.

        Returns:
            Interpolated box [x1, y1, x2, y2] or None if interpolation fails.
        """
        # Find previous and next matched anchors
        prev_matched_idx: Optional[int] = None
        next_matched_idx: Optional[int] = None

        for idx in matched_indices:
            if idx < gt_idx:
                prev_matched_idx = idx
            elif idx > gt_idx and next_matched_idx is None:
                next_matched_idx = idx
                break

        # Edge cases: no anchors on one or both sides
        if prev_matched_idx is None and next_matched_idx is None:
            # No anchors at all - cannot interpolate
            return None

        if prev_matched_idx is None:
            # Only next anchor available - use it directly
            ocr_idx = gt_to_ocr_mapping[next_matched_idx]
            return list(ocr_results[ocr_idx]["box"])

        if next_matched_idx is None:
            # Only prev anchor available - use it directly
            ocr_idx = gt_to_ocr_mapping[prev_matched_idx]
            return list(ocr_results[ocr_idx]["box"])

        # Both anchors available - perform linear interpolation
        prev_ocr_idx = gt_to_ocr_mapping[prev_matched_idx]
        next_ocr_idx = gt_to_ocr_mapping[next_matched_idx]

        box_prev = np.array(ocr_results[prev_ocr_idx]["box"], dtype=np.float32)
        box_next = np.array(ocr_results[next_ocr_idx]["box"], dtype=np.float32)

        # Calculate relative position t in [0, 1]
        # t = (gt_idx - prev_idx) / (next_idx - prev_idx)
        gap_span = next_matched_idx - prev_matched_idx
        t = (gt_idx - prev_matched_idx) / gap_span

        # Linear interpolation: B_gap = B_prev + t * (B_next - B_prev)
        box_gap = box_prev + t * (box_next - box_prev)
        box_gap = box_gap.tolist()

        # Gate 3: Check minimum box width
        box_width = box_gap[2] - box_gap[0]  # x2 - x1
        if box_width < self.min_box_width:
            return None

        return box_gap

    def get_hard_negatives(
        self,
        aligned_pairs: List[AlignedPair],
    ) -> List[AlignedPair]:
        """Extract hard negative samples for Agent B training.

        Hard negatives are samples where OCR made errors that Agent B
        should learn to correct:
        - DELETION_RECOVERY: OCR missed a character
        - SUBSTITUTION: OCR confused characters

        Args:
            aligned_pairs: Output from align() method.

        Returns:
            Filtered list containing only hard negative samples.
        """
        hard_types = {
            AlignmentType.DELETION_RECOVERY,
            AlignmentType.SUBSTITUTION,
        }
        return [p for p in aligned_pairs if p.align_type in hard_types]

    def to_dict_list(
        self,
        aligned_pairs: List[AlignedPair],
    ) -> List[Dict]:
        """Convert AlignedPair objects to dictionary format.

        This matches the output format specified in L2W1-DE-001.

        Args:
            aligned_pairs: List of AlignedPair objects.

        Returns:
            List of dictionaries with keys: gt_char, pred_char, box, type.
        """
        return [
            {
                "gt_char": p.gt_char,
                "pred_char": p.pred_char,
                "box": p.box,
                "conf": p.conf,
                "type": p.align_type.value,
            }
            for p in aligned_pairs
        ]
