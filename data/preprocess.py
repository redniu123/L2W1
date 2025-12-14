# -*- coding: utf-8 -*-
"""Data preprocessing pipeline for L2W1.

Spec Reference: 
- L2W1-DE-002 (Data Pipeline & Cropping)
- L2W1-MOD-002 (Refactor Preprocess for Entropy)
- L2W1-MOD-004 (Feature Injection)

This module implements the data processing pipeline that:
1. Runs OCR inference via Agent A (with entropy calculation)
2. Computes semantic perplexity via Router
3. Performs DTW alignment
4. Crops character regions with context expansion
5. Normalizes to 336x336 square with padding
6. Generates training metadata with entropy and PPL values
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# L2W1-MOD-002: Replace PaddleOCR with AgentA (Dependency Injection)
# L2W1-MOD-004: Add Router for PPL calculation
from core.agent_a import AgentA
from core.router import Router
from data.alignment.dtw_core import AlignedPair, AlignmentType, DTWAligner

logger = logging.getLogger(__name__)


class SampleType:
    """Sample type constants matching L2W1-DE-002 Schema."""

    POSITIVE_ANCHOR = "POSITIVE_ANCHOR"
    HARD_NEGATIVE_SUB = "HARD_NEGATIVE_SUB"
    HARD_NEGATIVE_DEL = "HARD_NEGATIVE_DEL"
    IGNORED = "IGNORED"


class DataProcessor:
    """Data processing pipeline for L2W1 training data generation.

    Implements the full pipeline defined in L2W1-DE-002, L2W1-MOD-002, and L2W1-MOD-004:
    - OCR inference using Agent A (with entropy calculation)
    - Semantic perplexity calculation via Router
    - DTW-based alignment
    - Context-expanded cropping
    - Square padding normalization
    - Entropy and PPL metadata for routing decisions

    Attributes:
        agent_a: AgentA instance (The Scout) for OCR with entropy.
        router: Router instance (The Gatekeeper) for PPL calculation.
        aligner: DTWAligner instance for sequence alignment.
        target_size: Target canvas size for normalized output (default 336).
        context_alpha: Context expansion ratio (default 0.3).
        output_dir: Directory to save cropped images.

    Example:
        >>> processor = DataProcessor(output_dir="./crops")
        >>> results = processor.process_single_image(
        ...     image_path="./page_001.jpg",
        ...     ground_truth_text="阿莫西林胶囊"
        ... )
        >>> # Results now include ocr_entropy and ppl_score fields!
        >>> print(f"Entropy: {results[0]['ocr_entropy']:.3f}, PPL: {results[0]['ppl_score']:.2f}")
    """

    def __init__(
        self,
        output_dir: str | Path,
        target_size: int = 336,
        context_alpha: float = 0.3,
        use_gpu: bool = True,
        lang: str = "ch",
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
    ) -> None:
        """Initialize DataProcessor.

        Args:
            output_dir: Directory to save cropped character images.
            target_size: Target canvas size in pixels (L2W1-DE-002: 336).
            context_alpha: Context expansion ratio (L2W1-DE-002: 0.3).
            use_gpu: Whether to use GPU for OCR inference.
            lang: OCR language, "ch" for Chinese.
            det_model_dir: Custom detection model path (optional).
            rec_model_dir: Custom recognition model path (optional).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_size = target_size
        self.context_alpha = context_alpha

        # L2W1-MOD-002: Initialize Agent A instead of raw PaddleOCR
        logger.info("Initializing Agent A (The Scout)...")
        try:
            self.agent_a = AgentA(
                use_gpu=use_gpu,
                lang=lang,
                det_model_dir=det_model_dir,
                rec_model_dir=rec_model_dir,
            )
            logger.info("Agent A initialized successfully.")
        except Exception as e:
            logger.warning(f"Failed to init AgentA with custom params: {e}")
            logger.warning("Trying with default parameters...")
            self.agent_a = AgentA(use_gpu=False)

        # L2W1-MOD-004: Initialize Router for PPL calculation
        logger.info("Initializing Router (The Gatekeeper)...")
        try:
            # Router uses device_map="auto" to efficiently share GPU with Agent A
            self.router = Router()
            logger.info("Router initialized successfully.")
        except Exception as e:
            logger.warning(f"Failed to init Router: {e}")
            logger.warning("Router will be disabled. PPL scores will be set to 0.0")
            self.router = None

        # Initialize DTWAligner
        self.aligner = DTWAligner()

    def process_single_image(
        self,
        image_path: str | Path,
        ground_truth_text: str,
        image_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Process a single image through the full pipeline.

        Pipeline steps (L2W1-DE-002 + L2W1-MOD-002 + L2W1-MOD-004):
        1. OCR Inference via Agent A -> ocr_results with entropy
        2. Compute PPL via Router -> ppl_score for full text line
        3. DTW Alignment -> aligned_pairs
        4. Context Expansion & Cropping
        5. Square Padding (336x336)
        6. Save crops & generate metadata (including ocr_entropy and ppl_score)

        Args:
            image_path: Path to the input image.
            ground_truth_text: Ground truth text for the image.
            image_id: Optional unique identifier (auto-generated if None).

        Returns:
            List of metadata dictionaries, one per valid character crop.
            Each dict follows L2W1-MOD-004 JSON Schema with ocr_entropy and ppl_score fields.

        Raises:
            FileNotFoundError: If image_path does not exist.
            ValueError: If OCR returns no results.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if image_id is None:
            image_id = image_path.stem

        # === Step 1: Load image ===
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        img_height, img_width = image.shape[:2]  # [H, W, C]

        # === Step 2: OCR Inference via Agent A (with entropy) ===
        ocr_results, entropy_map, full_text = self._run_ocr_with_entropy(image_path)

        logger.debug(f"OCR results count: {len(ocr_results)}")
        logger.debug(f"Full text from OCR: '{full_text}'")
        
        if not ocr_results:
            error_msg = (
                f"OCR returned no results for image: {image_path}. "
                "Cannot proceed with alignment. "
                "Possible causes: "
                "1. Image is too small or low quality "
                "2. No text detected in image "
                "3. OCR model initialization failed"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # === Step 2.5: Compute PPL via Router (L2W1-MOD-004) ===
        # PPL is computed at line level, then assigned to all character crops
        ppl_score = 0.0
        if self.router is not None and full_text:
            try:
                ppl_score = self.router.compute_ppl(full_text)
                logger.debug(f"Computed PPL for '{full_text}': {ppl_score:.2f}")
            except Exception as e:
                logger.warning(f"Failed to compute PPL: {e}. Using default 0.0")
                ppl_score = 0.0
        else:
            if self.router is None:
                logger.debug("Router not available, skipping PPL calculation")
            if not full_text:
                logger.debug("No OCR text found, skipping PPL calculation")

        # === Step 3: DTW Alignment ===
        # DTWAligner expects: [{'char': 'X', 'box': [...], 'conf': 0.99}, ...]
        logger.debug(f"Running DTW alignment: OCR={len(ocr_results)} chars, GT={len(ground_truth_text)} chars")
        aligned_pairs = self.aligner.align(ocr_results, ground_truth_text)
        logger.debug(f"DTW alignment produced {len(aligned_pairs)} pairs")

        # === Step 4 & 5: Crop, Pad, Save ===
        metadata_list: List[Dict[str, Any]] = []
        
        # Count alignment types for debugging
        align_type_counts = {}
        for pair in aligned_pairs:
            align_type = pair.align_type.value
            align_type_counts[align_type] = align_type_counts.get(align_type, 0) + 1
        logger.debug(f"Alignment type distribution: {align_type_counts}")

        for idx, pair in enumerate(aligned_pairs):
            # Skip IGNORE samples
            if pair.align_type == AlignmentType.IGNORE:
                logger.debug(f"Skipping IGNORE sample at index {idx}: GT='{pair.gt_char}', Pred='{pair.pred_char}'")
                continue

            # Get context (previous and next characters)
            context_left = ground_truth_text[:idx] if idx > 0 else ""
            context_right = (
                ground_truth_text[idx + 1 :] if idx < len(ground_truth_text) - 1 else ""
            )

            # Limit context length for efficiency
            context_left = context_left[-5:] if len(context_left) > 5 else context_left
            context_right = (
                context_right[:5] if len(context_right) > 5 else context_right
            )

            # Expand and crop
            expanded_box = self._expand_box(
                box=pair.box,
                img_width=img_width,
                img_height=img_height,
            )

            # Skip invalid boxes (after expansion)
            if expanded_box is None:
                logger.debug(f"Skipping invalid box at index {idx} (expansion failed)")
                continue

            x1, y1, x2, y2 = expanded_box
            crop = image[y1:y2, x1:x2]

            # Skip empty crops
            if crop.size == 0:
                logger.debug(f"Skipping empty crop at index {idx}")
                continue

            # Normalize to square canvas
            normalized_crop = self._normalize_to_square(crop)

            # Generate unique filename
            crop_id = f"{image_id}_char{idx:04d}_{uuid.uuid4().hex[:8]}"
            crop_filename = f"{crop_id}.jpg"
            crop_path = self.output_dir / crop_filename

            # Save cropped image
            cv2.imwrite(str(crop_path), normalized_crop)

            # Determine sample type
            sample_type = self._get_sample_type(pair.align_type)

            # L2W1-MOD-002: Get entropy for this character
            # Use predicted char to look up entropy, or default to 0.0
            ocr_entropy = entropy_map.get(pair.pred_char, {}).get(idx, 0.0)
            
            # If no exact match, try to find by position
            if ocr_entropy == 0.0 and idx < len(ocr_results):
                ocr_entropy = ocr_results[idx].get("entropy", 0.0)

            # Build metadata (ensure all values are JSON-serializable)
            metadata = {
                "id": crop_id,
                "image_path": str(crop_path.relative_to(self.output_dir.parent)),
                "label_gt": pair.gt_char,
                "ocr_pred": pair.pred_char,
                "context_left": context_left,
                "context_right": context_right,
                "sample_type": sample_type,
                "confidence": self._to_native(pair.conf),
                # L2W1-MOD-002: New field - ocr_entropy
                "ocr_entropy": self._to_native(ocr_entropy),
                # L2W1-MOD-004: New field - ppl_score (line-level, shared by all chars in line)
                "ppl_score": self._to_native(ppl_score),
                "original_box": self._to_native(pair.box),
                "expanded_box": self._to_native(list(expanded_box)),
            }

            metadata_list.append(metadata)
            logger.debug(f"Added crop {idx}: GT='{pair.gt_char}', Pred='{pair.pred_char}', Type={sample_type}")

        logger.info(f"Generated {len(metadata_list)} valid crops from {len(aligned_pairs)} aligned pairs")
        
        if len(metadata_list) == 0 and len(aligned_pairs) > 0:
            logger.warning(
                f"All {len(aligned_pairs)} aligned pairs were filtered out. "
                f"Check: IGNORE samples, invalid boxes, or empty crops."
            )

        return metadata_list

    def _run_ocr_with_entropy(
        self,
        image_path: Path,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[int, float]], str]:
        """Run Agent A OCR and extract character-level entropy.

        L2W1-MOD-002: Uses AgentA.inference_with_char_boxes() to get
        both OCR results and entropy values.
        L2W1-MOD-004: Also returns full text line for PPL calculation.

        Args:
            image_path: Path to input image.

        Returns:
            Tuple of:
            - List of OCR results in DTWAligner-compatible format:
              [{'char': 'X', 'box': [x1, y1, x2, y2], 'conf': 0.99, 'entropy': 0.05}, ...]
            - Entropy map for quick lookup: {char: {index: entropy}}
            - Full text line (concatenated from all characters) for PPL calculation
        """
        logger.debug(f"Running OCR on: {image_path}")
        
        # Get line-level results first (for PPL calculation)
        try:
            line_results = self.agent_a.inference(str(image_path))
            logger.debug(f"Line-level OCR results: {len(line_results)} lines")
        except Exception as e:
            logger.error(f"Agent A inference failed: {e}")
            return [], {}, ""
        
        # Extract full text from line results
        full_text = ""
        if line_results:
            # Concatenate all detected text lines
            texts = [line.get("text", "") for line in line_results if line.get("text")]
            full_text = "".join(texts)  # Remove space between lines for Chinese
            logger.debug(f"Extracted full text: '{full_text}' (length: {len(full_text)})")
        else:
            logger.warning("No line-level OCR results returned")
        
        # Use Agent A's inference_with_char_boxes for character-level results
        try:
            char_results = self.agent_a.inference_with_char_boxes(str(image_path))
            logger.debug(f"Character-level OCR results: {len(char_results)} characters")
        except Exception as e:
            logger.error(f"Agent A inference_with_char_boxes failed: {e}")
            return [], {}, full_text

        if not char_results:
            logger.warning(f"No character-level OCR results for {image_path}")
            return [], {}, full_text

        # Convert to DTWAligner-compatible format and build entropy map
        ocr_results: List[Dict[str, Any]] = []
        entropy_map: Dict[str, Dict[int, float]] = {}

        for i, char_result in enumerate(char_results):
            # Adapt format: AgentA uses 'score', DTWAligner needs 'conf'
            ocr_entry = {
                "char": char_result["char"],
                "box": char_result["box"],
                "conf": char_result["score"],  # Rename score -> conf
                "entropy": char_result.get("entropy", 0.0),  # Preserve entropy
            }
            ocr_results.append(ocr_entry)

            # Build entropy lookup map
            char = char_result["char"]
            if char not in entropy_map:
                entropy_map[char] = {}
            entropy_map[char][i] = char_result.get("entropy", 0.0)

        logger.debug(f"Converted {len(ocr_results)} OCR results for DTW alignment")
        logger.debug(f"OCR characters: {''.join([r['char'] for r in ocr_results])}")

        return ocr_results, entropy_map, full_text

    def _to_native(self, obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization.

        Args:
            obj: Object to convert.

        Returns:
            Python native type equivalent.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return [self._to_native(x) for x in obj]
        elif isinstance(obj, (list, tuple)):
            return [self._to_native(x) for x in obj]
        else:
            return obj

    def _expand_box(
        self,
        box: List[float],
        img_width: int,
        img_height: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Expand bounding box with context padding.

        Implements L2W1-DE-002 Context Expansion:
        x_new = x_center ± (0.5 + α) × w

        Args:
            box: Original box [x1, y1, x2, y2].
            img_width: Image width for boundary check.
            img_height: Image height for boundary check.

        Returns:
            Expanded box as (x1, y1, x2, y2) integers, or None if invalid.
        """
        x1, y1, x2, y2 = box

        # Calculate dimensions
        w = x2 - x1
        h = y2 - y1

        # Gate 3: Skip boxes with width < 8px
        if w < 8:
            return None

        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        # Context Expansion: L2W1-DE-002
        # x_new = x_center ± (0.5 + α) × w
        expansion_factor = 0.5 + self.context_alpha

        new_half_w = expansion_factor * w
        new_half_h = expansion_factor * h

        new_x1 = x_center - new_half_w
        new_x2 = x_center + new_half_w
        new_y1 = y_center - new_half_h
        new_y2 = y_center + new_half_h

        # Boundary Check: Clamp to image bounds
        new_x1 = max(0, int(new_x1))
        new_y1 = max(0, int(new_y1))
        new_x2 = min(img_width, int(new_x2))
        new_y2 = min(img_height, int(new_y2))

        # Validate result
        if new_x2 <= new_x1 or new_y2 <= new_y1:
            return None

        return (new_x1, new_y1, new_x2, new_y2)

    def _normalize_to_square(
        self,
        crop: np.ndarray,
    ) -> np.ndarray:
        """Normalize crop to square canvas with padding.

        Implements L2W1-DE-002 Visual Normalization:
        1. Create 336×336 black canvas
        2. Resize crop maintaining aspect ratio (long edge = 336)
        3. Center-paste onto canvas

        Args:
            crop: Input crop image. Shape: [H, W, C]

        Returns:
            Normalized square image. Shape: [336, 336, 3]
        """
        h, w = crop.shape[:2]
        target = self.target_size

        # Calculate scale factor (long edge = target_size)
        scale = target / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize maintaining aspect ratio
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create black canvas (L2W1-DE-002: pure black RGB (0, 0, 0))
        canvas = np.zeros((target, target, 3), dtype=np.uint8)

        # Calculate center paste position
        x_offset = (target - new_w) // 2
        y_offset = (target - new_h) // 2

        # Paste resized image onto canvas
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return canvas

    def _get_sample_type(
        self,
        align_type: AlignmentType,
    ) -> str:
        """Map alignment type to sample type for training.

        Args:
            align_type: Alignment result type from DTWAligner.

        Returns:
            Sample type string as defined in L2W1-DE-002 Schema.
        """
        type_mapping = {
            AlignmentType.MATCH: SampleType.POSITIVE_ANCHOR,
            AlignmentType.SUBSTITUTION: SampleType.HARD_NEGATIVE_SUB,
            AlignmentType.DELETION_RECOVERY: SampleType.HARD_NEGATIVE_DEL,
            AlignmentType.INSERTION: SampleType.IGNORED,
            AlignmentType.IGNORE: SampleType.IGNORED,
        }
        return type_mapping.get(align_type, SampleType.IGNORED)

    def process_batch(
        self,
        image_gt_pairs: List[Tuple[str | Path, str]],
        progress_callback: Optional[callable] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process a batch of images.

        Args:
            image_gt_pairs: List of (image_path, ground_truth_text) tuples.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            Tuple of (successful_results, failed_items).
            - successful_results: Flattened list of all metadata dicts
            - failed_items: List of {'path': ..., 'error': ...} for failures
        """
        all_results: List[Dict[str, Any]] = []
        failed_items: List[Dict[str, Any]] = []
        total = len(image_gt_pairs)

        for idx, (image_path, gt_text) in enumerate(image_gt_pairs):
            if progress_callback:
                progress_callback(idx + 1, total)

            try:
                results = self.process_single_image(image_path, gt_text)
                all_results.extend(results)
            except Exception as e:
                failed_items.append(
                    {
                        "path": str(image_path),
                        "error": str(e),
                    }
                )

        return all_results, failed_items

    def get_entropy_statistics(
        self,
        metadata_list: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute entropy statistics for processed samples.

        Useful for understanding uncertainty distribution and
        tuning routing thresholds.

        Args:
            metadata_list: Output from process_single_image or process_batch.

        Returns:
            Statistics dict with entropy metrics.
        """
        if not metadata_list:
            return {
                "total_samples": 0,
                "avg_entropy": 0.0,
                "max_entropy": 0.0,
                "min_entropy": 0.0,
                "high_entropy_count": 0,
                "high_entropy_ratio": 0.0,
            }

        entropies = [m.get("ocr_entropy", 0.0) for m in metadata_list]
        
        # High entropy threshold (50% of max possible entropy ~0.346)
        high_entropy_threshold = 0.346
        high_entropy_count = sum(1 for e in entropies if e > high_entropy_threshold)

        return {
            "total_samples": len(metadata_list),
            "avg_entropy": sum(entropies) / len(entropies),
            "max_entropy": max(entropies),
            "min_entropy": min(entropies),
            "high_entropy_count": high_entropy_count,
            "high_entropy_ratio": high_entropy_count / len(metadata_list),
        }
