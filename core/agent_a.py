# -*- coding: utf-8 -*-
"""Agent A: The Scout - PaddleOCR wrapper with entropy calculation.

Spec Reference: L2W1-MOD-001 (Agent A & Entropy)

This module provides:
1. Unified OCR inference interface via PaddleOCR
2. Character-level confidence extraction
3. Visual entropy calculation to measure recognition uncertainty

The entropy tells us "how uncertain" Agent A is about each character.
High entropy = Agent A is "panicking" = likely needs Agent B's help.
"""

import logging
import math
import os
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)


# =============================================================================
# Entropy Calculation Utilities
# =============================================================================


def binary_entropy(p: float, eps: float = 1e-10) -> float:
    """Calculate binary entropy for a probability value.

    Implements the visual entropy formula from L2W1-MOD-001:

    $$U_{vis} = - (p \\log p + (1-p) \\log (1-p))$$

    Properties:
    - When p ≈ 1.0 (very confident): U_vis ≈ 0
    - When p ≈ 0.5 (uncertain): U_vis is maximum (= 1.0 for log base 2)
    - When p ≈ 0.0 (confident it's wrong): U_vis ≈ 0

    Args:
        p: Probability/confidence value in [0, 1].
        eps: Small epsilon to avoid log(0).

    Returns:
        Binary entropy value. Range: [0, 1] when using log base 2,
        or [0, ~0.693] when using natural log.

    Note:
        We use natural log (ln) for consistency with PyTorch/NumPy.
        The maximum value is ln(2) ≈ 0.693 when p = 0.5.
    """
    # Clamp p to avoid numerical issues
    p = max(eps, min(1.0 - eps, p))

    # Binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
    entropy = -p * math.log(p) - (1 - p) * math.log(1 - p)

    return entropy


def compute_char_entropies(scores: List[float]) -> List[float]:
    """Compute entropy for each character score.

    Args:
        scores: List of confidence scores for each character.

    Returns:
        List of entropy values, same length as scores.
    """
    return [binary_entropy(s) for s in scores]


def pad_square_image_for_ocr(
    image: np.ndarray,
    padding_ratio: float = 0.3,
    fill_color: tuple = (255, 255, 255),
) -> np.ndarray:
    """Pad a square image with white borders to make it look like a text line.

    When using PaddleOCR with det=False (recognition-only mode) on cropped
    single-character images, the recognition model expects a horizontally
    elongated text line image. Square images get distorted.

    This function adds white padding to the left and right sides of square
    images to simulate a short text line format.

    Args:
        image: Input image as numpy array [H, W, C].
        padding_ratio: Ratio of padding width relative to image width.
            Default 0.3 means 30% padding on each side.
        fill_color: RGB color for padding. Default is white (255, 255, 255).

    Returns:
        Padded image as numpy array.
    """
    h, w = image.shape[:2]

    # Check if image is roughly square (aspect ratio close to 1:1)
    aspect_ratio = w / h if h > 0 else 1.0

    # Only pad if image is square-ish (aspect ratio between 0.7 and 1.5)
    if 0.7 <= aspect_ratio <= 1.5:
        # Calculate padding width
        pad_width = int(w * padding_ratio)

        # Create padded image using cv2.copyMakeBorder
        if len(image.shape) == 3:
            # Color image
            padded = cv2.copyMakeBorder(
                image,
                top=0,
                bottom=0,
                left=pad_width,
                right=pad_width,
                borderType=cv2.BORDER_CONSTANT,
                value=fill_color,
            )
        else:
            # Grayscale image
            padded = cv2.copyMakeBorder(
                image,
                top=0,
                bottom=0,
                left=pad_width,
                right=pad_width,
                borderType=cv2.BORDER_CONSTANT,
                value=fill_color[0],  # Use single value for grayscale
            )

        logger.debug(
            f"Padded square image: {w}x{h} -> {padded.shape[1]}x{padded.shape[0]} "
            f"(added {pad_width}px on each side)"
        )
        return padded

    return image


# =============================================================================
# Agent A: The Scout
# =============================================================================


class AgentA:
    """PaddleOCR wrapper with uncertainty estimation.

    Agent A is "The Scout" in the L2W1 hierarchy:
    - Fast, lightweight OCR using PaddleOCR
    - Provides not just predictions, but also uncertainty metrics
    - High entropy characters are candidates for Agent B review

    Attributes:
        ocr_engine: Underlying PaddleOCR instance.
        use_gpu: Whether GPU acceleration is enabled.

    Example:
        >>> agent_a = AgentA(use_gpu=True)
        >>> results = agent_a.inference(image)
        >>> for line in results:
        ...     print(f"Text: {line['text']}, Avg Entropy: {line['avg_entropy']:.3f}")
        ...     for char, entropy in zip(line['text'], line['entropies']):
        ...         if entropy > 0.3:
        ...             print(f"  ⚠️ '{char}' is uncertain (entropy={entropy:.3f})")

    Note on Character-Level Scores:
        PaddleOCR's standard interface returns line-level confidence scores.
        To obtain true character-level confidence, we would need to:
        1. Access the recognition model's softmax output
        2. Extract per-character probabilities from the CTC/Attention decoder

        Current implementation provides two modes:
        - Default: Distributes line-level score to all characters (approximation)
        - Advanced: Attempts to extract character-level scores from model internals

        For production use, consider fine-tuning PaddleOCR's rec model to
        expose character-level logits.
    """

    # Maximum entropy value (ln(2) for natural log)
    MAX_ENTROPY = math.log(2)  # ≈ 0.693

    def __init__(
        self,
        use_gpu: bool = True,
        lang: str = "ch",
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
        show_log: bool = False,
    ) -> None:
        """Initialize Agent A with PaddleOCR.

        Args:
            use_gpu: Whether to use GPU for inference.
            lang: OCR language, "ch" for Chinese.
            det_model_dir: Custom detection model path.
            rec_model_dir: Custom recognition model path.
            show_log: Whether to show PaddleOCR logs.
        """
        self.use_gpu = use_gpu
        self.lang = lang

        # Set device parameter for PaddleOCR
        # PaddleOCR 3.x+ uses "device" parameter instead of "use_gpu"
        device = "gpu" if use_gpu else "cpu"

        # If using CPU, set environment variables to prevent cudnn loading
        if not use_gpu:
            # Disable GPU-related environment variables to prevent cudnn errors
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            os.environ.setdefault(
                "FLAGS_use_mkldnn", "0"
            )  # Disable MKLDNN to avoid instruction set issues
            logger.info("Using CPU mode - GPU and cudnn will be disabled")

        # Build OCR kwargs (compatible with different PaddleOCR versions)
        # PaddleOCR 3.x+ has different API, so we use minimal config first
        ocr_kwargs: Dict[str, Any] = {
            "lang": lang,
            "device": device,  # Explicitly set device to avoid cudnn issues
        }

        # Add optional parameters only if model dirs are provided
        if det_model_dir:
            ocr_kwargs["det_model_dir"] = det_model_dir
        if rec_model_dir:
            ocr_kwargs["rec_model_dir"] = rec_model_dir

        # Try to add use_angle_cls if supported (some versions don't support it)
        # Initialize PaddleOCR engine with progressive fallback
        try:
            # First try with use_angle_cls
            test_kwargs = {**ocr_kwargs, "use_angle_cls": True}
            self.ocr_engine = PaddleOCR(**test_kwargs)
            logger.info(
                f"PaddleOCR initialized with use_angle_cls=True, device={device}"
            )
        except (TypeError, AttributeError) as e:
            # If that fails, try without use_angle_cls
            logger.warning(f"PaddleOCR init with use_angle_cls failed: {e}")
            logger.warning("Trying without use_angle_cls...")
            try:
                self.ocr_engine = PaddleOCR(**ocr_kwargs)
                logger.info(
                    f"PaddleOCR initialized with minimal config, device={device}"
                )
            except Exception as e2:
                # Check if it's a cudnn error and we're trying to use GPU
                error_str = str(e2).lower()
                if use_gpu and ("cudnn" in error_str or "cuda" in error_str):
                    logger.error(f"GPU initialization failed: {e2}")
                    logger.warning(
                        "GPU is not available or cudnn is not properly configured."
                    )
                    logger.warning("Falling back to CPU mode...")
                    # Retry with CPU
                    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
                    os.environ.setdefault("FLAGS_use_mkldnn", "0")
                    device = "cpu"
                    self.use_gpu = False
                    ocr_kwargs["device"] = device
                    try:
                        self.ocr_engine = PaddleOCR(**ocr_kwargs)
                        logger.info(
                            "PaddleOCR initialized with CPU (fallback from GPU)"
                        )
                    except Exception as e3:
                        logger.error(f"CPU initialization also failed: {e3}")
                        raise
                else:
                    # Last resort: try with just lang and device
                    logger.warning(f"PaddleOCR init failed: {e2}")
                    logger.warning("Trying with lang and device only...")
                    try:
                        self.ocr_engine = PaddleOCR(lang=lang, device=device)
                        logger.info(
                            f"PaddleOCR initialized with lang and device={device} only"
                        )
                    except Exception as e3:
                        logger.error(
                            f"All PaddleOCR initialization attempts failed. Last error: {e3}"
                        )
                        raise

    def inference(
        self,
        image: Union[np.ndarray, str],
        return_raw: bool = False,
        skip_detection: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run OCR inference and compute entropy metrics.

        This is the main interface for Agent A, as defined in L2W1-MOD-001.

        Args:
            image: Input image as numpy array [H, W, C] or file path string.
            return_raw: If True, also return raw PaddleOCR output.
            skip_detection: If True, skip text detection (det=False).
                Use this for pre-cropped single-character images.

        Returns:
            List of dicts, where each dict represents a text line:
            {
                "text": "阿莫西林",
                "box": [[x1,y1], [x2,y1], [x2,y2], [x1,y2]],
                "line_score": 0.95,           # Original line confidence
                "scores": [0.95, 0.95, ...],  # Per-character scores
                "entropies": [0.01, 0.01, ...], # Per-character entropy
                "avg_entropy": 0.01,          # Average entropy for the line
                "max_entropy": 0.01,          # Max entropy (most uncertain char)
                "uncertain_chars": []         # Chars with high entropy
            }

        Note:
            Currently uses line-level score distributed to characters.
            See class docstring for details on obtaining true char-level scores.
        """
        # =====================================================================
        # Step 1: Load and preprocess image
        # =====================================================================
        if isinstance(image, str):
            # Load image from file path
            img_array = cv2.imread(image)
            if img_array is None:
                logger.error(f"Failed to load image: {image}")
                return []
            # Convert BGR to RGB (PaddleOCR expects RGB)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        elif hasattr(image, "mode"):
            # PIL Image - convert to numpy array
            from PIL import Image as PILImage

            if isinstance(image, PILImage.Image):
                img_array = np.array(image)
                # Ensure RGB format
                if len(img_array.shape) == 2:
                    # Grayscale, convert to RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                elif img_array.shape[2] == 4:
                    # RGBA, convert to RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            else:
                img_array = image
        else:
            img_array = image

        # =====================================================================
        # Step 2: Smart Detection Mode Selection
        # =====================================================================
        # [FIX] Intelligent detection mode: Force detection for non-square images
        # This prevents recognition head from processing full images with background
        h, w = img_array.shape[:2]
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
        
        # Heuristic: If image is wide (width > 64) or aspect ratio > 1.5, force detection
        # This ensures real-world long text images use detection mode
        should_use_detection = w > 64 or aspect_ratio > 1.5
        
        # Override skip_detection if image suggests it needs detection
        # Only skip detection if explicitly requested AND image is square/small
        actual_skip_detection = skip_detection and not should_use_detection
        
        if actual_skip_detection:
            # Pad square images to make them look like text lines
            # This prevents distortion in the recognition model
            img_array = pad_square_image_for_ocr(
                img_array,
                padding_ratio=0.3,  # 30% padding on each side
                fill_color=(255, 255, 255),  # White background
            )
            logger.debug("Running OCR in recognition-only mode (det=False)")
        elif should_use_detection and skip_detection:
            logger.info(
                f"Auto-enabling detection mode for image {w}x{h} "
                f"(aspect_ratio={aspect_ratio:.2f})"
            )

        # =====================================================================
        # Step 3: Run PaddleOCR
        # =====================================================================
        try:
            raw_results = self.ocr_engine.ocr(
                img_array,
                cls=not actual_skip_detection,
                det=not actual_skip_detection,  # Use actual_skip_detection instead of skip_detection
                rec=True,
            )
        except Exception as e:
            logger.error(f"PaddleOCR inference failed: {e}")
            return []

        # Handle empty results
        if raw_results is None or len(raw_results) == 0:
            logger.debug("PaddleOCR returned None or empty results")
            return []

        if raw_results[0] is None:
            logger.debug("PaddleOCR first result is None")
            return []

        # =====================================================================
        # Step 4: Parse results (different structure for det=True vs det=False)
        # =====================================================================
        results: List[Dict[str, Any]] = []

        # When det=False (skip_detection=True):
        #   Return structure: [[(text, score), (text2, score2), ...]]
        # When det=True:
        #   Return structure: [[[box, (text, score)], [box2, (text2, score2)], ...]]

        if skip_detection:
            # Recognition-only mode: result is [[(text, score), ...]]
            rec_results = raw_results[0]

            logger.debug(
                f"rec_results type: {type(rec_results)}, content: {rec_results}"
            )

            # Handle various possible return formats
            if rec_results is None:
                return []

            # Sometimes it's nested differently, try to extract (text, score) tuples
            items_to_process = []

            if isinstance(rec_results, list):
                if len(rec_results) == 0:
                    return []

                # Check first element to determine structure
                first_elem = rec_results[0]

                if isinstance(first_elem, tuple) and len(first_elem) == 2:
                    # Format: [(text, score), ...]
                    items_to_process = rec_results
                elif isinstance(first_elem, list):
                    # Format: [[(text, score), ...]] - nested
                    items_to_process = first_elem
                else:
                    # Unknown format, try to use as-is
                    items_to_process = rec_results
            elif isinstance(rec_results, tuple) and len(rec_results) == 2:
                # Single result: (text, score)
                items_to_process = [rec_results]

            # Process each (text, score) tuple
            for item in items_to_process:
                if item is None:
                    continue

                # Extract text and score
                if isinstance(item, tuple) and len(item) >= 2:
                    text = item[0]
                    line_score = item[1]
                elif isinstance(item, list) and len(item) >= 2:
                    text = item[0]
                    line_score = item[1]
                else:
                    logger.warning(f"Unexpected item format in rec results: {item}")
                    continue

                if not text:
                    continue

                # For rec-only mode, create a dummy box covering the whole image
                h, w = img_array.shape[:2]
                box_points = [[0, 0], [w, 0], [w, h], [0, h]]

                # Build result
                result_dict = self._build_result_dict(
                    text, line_score, box_points, return_raw, item
                )
                if result_dict:
                    results.append(result_dict)
                    logger.debug(f"Recognized text: '{text}' (score: {line_score:.4f})")

        else:
            # Detection mode: result is [[[box, (text, score)], ...]]
            for line_result in raw_results[0]:
                if line_result is None:
                    continue

                try:
                    box_points, (text, line_score) = line_result
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Failed to unpack line result: {line_result}, error: {e}"
                    )
                    continue

                if not text:
                    continue

                # Build result
                result_dict = self._build_result_dict(
                    text, line_score, box_points, return_raw, line_result
                )
                if result_dict:
                    results.append(result_dict)

        return results

    def _build_result_dict(
        self,
        text: str,
        line_score: float,
        box_points: List,
        return_raw: bool,
        raw_item: Any,
    ) -> Optional[Dict[str, Any]]:
        """Build a result dictionary for a recognized text line.

        Args:
            text: Recognized text string.
            line_score: Confidence score for the line.
            box_points: Bounding box points.
            return_raw: Whether to include raw result.
            raw_item: Raw result item for debugging.

        Returns:
            Result dictionary or None if invalid.
        """
        if not text:
            return None

        # Get character-level scores
        char_scores = self._get_char_scores(text, line_score)

        # Compute entropies
        char_entropies = compute_char_entropies(char_scores)
        avg_entropy = (
            sum(char_entropies) / len(char_entropies) if char_entropies else 0.0
        )
        max_entropy = max(char_entropies) if char_entropies else 0.0

        # Find uncertain characters (entropy > 50% of max possible)
        entropy_threshold = self.MAX_ENTROPY * 0.5  # ≈ 0.346
        uncertain_chars = [
            {"char": text[i], "index": i, "entropy": e}
            for i, e in enumerate(char_entropies)
            if e > entropy_threshold
        ]

        # Build result dict
        result_dict: Dict[str, Any] = {
            "text": text,
            "box": box_points,
            "line_score": float(line_score),
            "scores": char_scores,
            "entropies": char_entropies,
            "avg_entropy": avg_entropy,
            "max_entropy": max_entropy,
            "uncertain_chars": uncertain_chars,
        }

        if return_raw:
            result_dict["raw"] = raw_item

        return result_dict

    def _get_char_scores(
        self,
        text: str,
        line_score: float,
    ) -> List[float]:
        """Extract or estimate character-level confidence scores.

        Current Implementation (Approximation):
            Distributes line-level score uniformly to all characters.
            This is a simplification - in reality, some characters in a
            line may be more uncertain than others.

        Future Enhancement:
            Access PaddleOCR's recognition model internals to get
            per-character softmax probabilities from the CTC decoder.

        Args:
            text: Recognized text string.
            line_score: Line-level confidence score.

        Returns:
            List of per-character confidence scores.

        LIMITATION:
            This is an approximation. True character-level scores would
            require modifying PaddleOCR to expose CTC decoder outputs.
            See: https://github.com/PaddlePaddle/PaddleOCR/issues/XXX
        """
        # Simple approximation: all chars get the same score
        # This will result in uniform entropy across the line
        n_chars = len(text)

        # Alternative: Add slight variation based on character position
        # Characters at edges are typically less reliable
        # This is a heuristic until we have true char-level scores
        scores = []
        for i, char in enumerate(text):
            # Base score from line confidence
            base_score = line_score

            # Edge penalty: first and last characters slightly less reliable
            edge_factor = 1.0
            if n_chars > 2:
                if i == 0 or i == n_chars - 1:
                    edge_factor = 0.98

            # Apply edge penalty
            char_score = min(0.999, base_score * edge_factor)
            scores.append(char_score)

        return scores

    def get_uncertainty_summary(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Summarize uncertainty across all detection results.

        Useful for deciding whether to escalate to Agent B.

        Args:
            results: Output from inference() method.

        Returns:
            Summary dict with overall uncertainty metrics.
        """
        if not results:
            return {
                "total_chars": 0,
                "total_lines": 0,
                "overall_avg_entropy": 0.0,
                "max_entropy_char": None,
                "high_uncertainty_count": 0,
                "needs_agent_b": False,
            }

        all_entropies: List[float] = []
        max_entropy_info: Optional[Dict[str, Any]] = None
        max_entropy_val = 0.0

        for line in results:
            for i, entropy in enumerate(line["entropies"]):
                all_entropies.append(entropy)
                if entropy > max_entropy_val:
                    max_entropy_val = entropy
                    max_entropy_info = {
                        "char": line["text"][i],
                        "entropy": entropy,
                        "line_text": line["text"],
                    }

        total_chars = len(all_entropies)
        overall_avg = sum(all_entropies) / total_chars if total_chars > 0 else 0.0

        # Count high-uncertainty characters
        threshold = self.MAX_ENTROPY * 0.5
        high_uncertainty_count = sum(1 for e in all_entropies if e > threshold)

        # Decision: escalate to Agent B if significant uncertainty exists
        # Heuristic: >10% of characters are uncertain, or avg entropy > 0.2
        uncertainty_ratio = (
            high_uncertainty_count / total_chars if total_chars > 0 else 0.0
        )
        needs_agent_b = uncertainty_ratio > 0.1 or overall_avg > 0.2

        return {
            "total_chars": total_chars,
            "total_lines": len(results),
            "overall_avg_entropy": overall_avg,
            "max_entropy_char": max_entropy_info,
            "high_uncertainty_count": high_uncertainty_count,
            "uncertainty_ratio": uncertainty_ratio,
            "needs_agent_b": needs_agent_b,
        }

    def inference_with_char_boxes(
        self,
        image: Union[np.ndarray, str],
        skip_detection: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run inference and return character-level bounding boxes.

        This method splits line-level boxes into character-level boxes
        using uniform width distribution. Useful for visualization and
        downstream processing.

        Args:
            image: Input image as numpy array or file path.

        Returns:
            List of character-level results:
            {
                "char": "阿",
                "box": [x1, y1, x2, y2],
                "score": 0.95,
                "entropy": 0.05,
                "line_text": "阿莫西林",
                "char_index": 0
            }
        """
        line_results = self.inference(image, skip_detection=skip_detection)
        char_results: List[Dict[str, Any]] = []

        for line in line_results:
            text = line["text"]
            box = line["box"]  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            scores = line["scores"]
            entropies = line["entropies"]

            if len(text) == 0:
                continue

            # Extract bounding box coordinates
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # Distribute width uniformly among characters
            char_width = (x_max - x_min) / len(text)

            for i, char in enumerate(text):
                char_x1 = x_min + i * char_width
                char_x2 = char_x1 + char_width

                char_results.append(
                    {
                        "char": char,
                        "box": [char_x1, y_min, char_x2, y_max],
                        "score": scores[i],
                        "entropy": entropies[i],
                        "line_text": text,
                        "char_index": i,
                    }
                )

        return char_results
