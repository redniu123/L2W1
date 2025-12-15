#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Long Text Sliding Window Inference for L2W1.

This script implements context-aware sliding window inference for
long handwritten text images with extreme aspect ratios.

Key Features:
- Height normalization to 48px for consistent OCR
- Sliding window with overlap to prevent character cutting
- Accumulated context passed to each window for Agent B correction
- Progressive text assembly with context awareness

Usage:
    python scripts/06_long_text_inference.py --image_path my_test_image.jpg
    python scripts/06_long_text_inference.py --image_path data/long_text.jpg --window_size 320 --stride 280
"""

# =============================================================================
# Path Patching
# =============================================================================
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Imports
# =============================================================================
import argparse
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================

# Default parameters
DEFAULT_WINDOW_SIZE = 320  # pixels
DEFAULT_STRIDE = 280  # pixels (40px overlap)
DEFAULT_TARGET_HEIGHT = 48  # pixels

# Thresholds
PPL_THRESHOLD = 100.0
ENTROPY_THRESHOLD = 0.3

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Image Preprocessing
# =============================================================================

def load_and_normalize_image(
    image_path: Path,
    target_height: int = DEFAULT_TARGET_HEIGHT,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Load image and normalize height while preserving aspect ratio.
    
    Args:
        image_path: Path to input image.
        target_height: Target height in pixels.
        
    Returns:
        Tuple of (normalized_image, original_size).
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    original_size = (img.shape[1], img.shape[0])  # (width, height)
    
    # Calculate new dimensions
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    new_h = target_height
    
    # Resize
    normalized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    logger.info(f"Image normalized: {w}x{h} -> {new_w}x{new_h} (scale: {scale:.3f})")
    logger.debug(f"[DEBUG] Resized Image Shape: {normalized.shape}")
    
    return normalized, original_size


def generate_windows(
    image_width: int,
    window_size: int,
    stride: int,
) -> List[Tuple[int, int]]:
    """Generate sliding window positions.
    
    Args:
        image_width: Total image width.
        window_size: Window width.
        stride: Step size between windows.
        
    Returns:
        List of (start_x, end_x) tuples.
    """
    windows = []
    x = 0
    
    while x < image_width:
        end_x = min(x + window_size, image_width)
        windows.append((x, end_x))
        
        if end_x >= image_width:
            break
        
        x += stride
    
    return windows


def extract_window(
    image: np.ndarray,
    start_x: int,
    end_x: int,
    pad_to_size: Optional[int] = None,
) -> np.ndarray:
    """Extract a window from the image.
    
    Args:
        image: Full image array.
        start_x: Start x coordinate.
        end_x: End x coordinate.
        pad_to_size: If provided, pad window to this width.
        
    Returns:
        Window image array.
    """
    window = image[:, start_x:end_x].copy()
    
    # Pad if needed (for last window)
    if pad_to_size is not None and window.shape[1] < pad_to_size:
        pad_width = pad_to_size - window.shape[1]
        # Pad with white on the right
        if len(window.shape) == 3:
            padding = np.full((window.shape[0], pad_width, 3), 255, dtype=np.uint8)
        else:
            padding = np.full((window.shape[0], pad_width), 255, dtype=np.uint8)
        window = np.hstack([window, padding])
    
    return window


# =============================================================================
# Text Deduplication
# =============================================================================

def deduplicate_overlap(
    accumulated: str,
    new_text: str,
    overlap_chars: int = 2,
) -> str:
    """Remove overlapping characters between accumulated and new text.
    
    Simple heuristic: check if end of accumulated matches start of new_text.
    
    Args:
        accumulated: Previously accumulated text.
        new_text: New text from current window.
        overlap_chars: Max characters to check for overlap.
        
    Returns:
        Deduplicated new text to append.
    """
    if not accumulated or not new_text:
        return new_text
    
    # Check for overlap at the boundary
    for i in range(min(overlap_chars, len(accumulated), len(new_text)), 0, -1):
        if accumulated[-i:] == new_text[:i]:
            logger.debug(f"Found overlap of {i} chars: '{accumulated[-i:]}'")
            return new_text[i:]
    
    return new_text


# =============================================================================
# L2W1 Pipeline Wrapper
# =============================================================================

class SlidingWindowPipeline:
    """L2W1 Pipeline wrapper for sliding window inference."""
    
    def __init__(
        self,
        ppl_threshold: float = PPL_THRESHOLD,
        entropy_threshold: float = ENTROPY_THRESHOLD,
        use_gpu: bool = True,
    ):
        self.ppl_threshold = ppl_threshold
        self.entropy_threshold = entropy_threshold
        self.agent_a = None
        self.router = None
        self.agent_b = None
        
        self._init_agents(use_gpu)
    
    def _init_agents(self, use_gpu: bool) -> None:
        """Initialize all agents."""
        # Agent A
        logger.info("Loading Agent A (PaddleOCR)...")
        try:
            from core.agent_a import AgentA
            self.agent_a = AgentA(use_gpu=use_gpu)
            logger.info("  ✓ Agent A ready")
        except Exception as e:
            logger.warning(f"  ✗ Agent A failed: {e}")
        
        # Router
        logger.info("Loading Router (Qwen2.5-0.5B)...")
        try:
            from core.router import Router
            self.router = Router()
            logger.info("  ✓ Router ready")
        except Exception as e:
            logger.warning(f"  ✗ Router failed: {e}")
        
        # Agent B
        logger.info("Loading Agent B (Qwen2-VL-2B)...")
        try:
            from core.agent_b import AgentB
            self.agent_b = AgentB(
                model_path="Qwen/Qwen2-VL-2B-Instruct",
                load_in_4bit=True,
            )
            logger.info("  ✓ Agent B ready")
        except Exception as e:
            logger.warning(f"  ✗ Agent B failed: {e}")
    
    def process_window(
        self,
        window_image: np.ndarray,
        context: str = "",
    ) -> Dict[str, Any]:
        """Process a single window with context awareness.
        
        Args:
            window_image: Window image (BGR numpy array).
            context: Accumulated text from previous windows.
            
        Returns:
            Result dictionary with OCR and L2W1 predictions.
        """
        result = {
            "ocr_pred": "",
            "l2w1_pred": "",
            "ppl_score": 0.0,
            "entropy": 0.0,
            "is_routed": False,
            "context_used": context,
        }
        
        # Convert BGR to RGB
        window_rgb = cv2.cvtColor(window_image, cv2.COLOR_BGR2RGB)
        
        # ===== Agent A: OCR =====
        if self.agent_a is not None:
            try:
                # Use detection mode for text line segments
                line_results = self.agent_a.inference(window_rgb, skip_detection=False)
                
                if line_results:
                    ocr_text = "".join([r.get("text", "") for r in line_results])
                    entropy = max(r.get("max_entropy", 0) for r in line_results)
                else:
                    ocr_text = ""
                    entropy = 0.0
                
                result["ocr_pred"] = ocr_text
                result["l2w1_pred"] = ocr_text  # Default
                result["entropy"] = entropy
                
            except Exception as e:
                logger.warning(f"Agent A error: {e}")
        
        # ===== Router: PPL Analysis with Context =====
        full_text_for_ppl = context + result["ocr_pred"]
        
        if self.router is not None and full_text_for_ppl:
            try:
                # Compute PPL for the combined text (context + current)
                ppl_score = self.router.compute_ppl(full_text_for_ppl)
                result["ppl_score"] = ppl_score
                
                # Routing decision
                should_route = (
                    ppl_score > self.ppl_threshold or
                    result["entropy"] > self.entropy_threshold
                )
                result["is_routed"] = should_route
                
            except Exception as e:
                logger.warning(f"Router error: {e}")
        
        # ===== Agent B: Visual-Semantic Correction =====
        if result["is_routed"] and self.agent_b is not None and result["ocr_pred"]:
            try:
                # Convert to PIL Image
                pil_image = Image.fromarray(window_rgb)
                
                # Use accumulated context as left context
                corrected = self.agent_b.inference(
                    crop_image=pil_image,
                    context_left=context[-20:] if len(context) > 20 else context,  # Last 20 chars
                    context_right="",  # We don't know future text
                    ocr_pred=result["ocr_pred"],
                )
                
                if corrected and corrected != "KEEP":
                    result["l2w1_pred"] = corrected
                    
            except Exception as e:
                logger.warning(f"Agent B error: {e}")
        
        return result


# =============================================================================
# Main Inference Logic
# =============================================================================

def run_sliding_window_inference(
    image_path: Path,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    target_height: int = DEFAULT_TARGET_HEIGHT,
) -> Dict[str, Any]:
    """Run sliding window inference on a long text image.
    
    Args:
        image_path: Path to input image.
        window_size: Width of each window.
        stride: Step size between windows.
        target_height: Target height for normalization.
        
    Returns:
        Full results dictionary.
    """
    # Load and normalize image
    print()
    print("-" * 70)
    print("  [Step 1] Loading and Normalizing Image")
    print("-" * 70)
    
    normalized_img, original_size = load_and_normalize_image(image_path, target_height)
    img_height, img_width = normalized_img.shape[:2]
    
    print(f"  Original size: {original_size[0]}x{original_size[1]}")
    print(f"  Normalized size: {img_width}x{img_height}")
    print(f"  Window size: {window_size}px, Stride: {stride}px")
    
    # Generate windows
    windows = generate_windows(img_width, window_size, stride)
    num_windows = len(windows)
    
    print(f"  Number of windows: {num_windows}")
    if num_windows == 1:
        print("  (Image fits in single window)")
    else:
        print(f"  Overlap: {window_size - stride}px per window")
    
    # Initialize pipeline
    print()
    print("-" * 70)
    print("  [Step 2] Initializing L2W1 Pipeline")
    print("-" * 70)
    
    pipeline = SlidingWindowPipeline(use_gpu=True)
    
    # Process windows
    print()
    print("-" * 70)
    print("  [Step 3] Sliding Window Inference")
    print("-" * 70)
    print()
    
    full_text_ocr = ""
    full_text_l2w1 = ""
    window_results = []
    
    for i, (start_x, end_x) in enumerate(windows):
        # Extract window
        window_img = extract_window(normalized_img, start_x, end_x)
        
        # Process with context
        result = pipeline.process_window(
            window_image=window_img,
            context=full_text_l2w1,  # Pass accumulated L2W1 text as context
        )
        
        # Get predictions
        ocr_pred = result["ocr_pred"]
        l2w1_pred = result["l2w1_pred"]
        
        # Deduplicate overlapping characters
        ocr_to_add = deduplicate_overlap(full_text_ocr, ocr_pred)
        l2w1_to_add = deduplicate_overlap(full_text_l2w1, l2w1_pred)
        
        # Accumulate
        full_text_ocr += ocr_to_add
        full_text_l2w1 += l2w1_to_add
        
        # Store result
        result["window_index"] = i
        result["window_range"] = (start_x, end_x)
        result["text_added_ocr"] = ocr_to_add
        result["text_added_l2w1"] = l2w1_to_add
        window_results.append(result)
        
        # Log
        routed_flag = "→B" if result["is_routed"] else "  "
        context_preview = result["context_used"][-10:] if result["context_used"] else ""
        
        print(f"  [Window {i+1}/{num_windows}] x={start_x}-{end_x}")
        print(f"    OCR:  '{ocr_pred}'")
        print(f"    L2W1: '{l2w1_pred}' {routed_flag}")
        print(f"    PPL:  {result['ppl_score']:.2f} | Entropy: {result['entropy']:.4f}")
        print(f"    Context: '...{context_preview}'")
        print(f"    [DEBUG] Window {i+1} OCR added: '{ocr_to_add}'")
        print(f"    [DEBUG] Window {i+1} L2W1 added: '{l2w1_to_add}'")
        print()
    
    # Final results
    results = {
        "image_path": str(image_path),
        "original_size": original_size,
        "normalized_size": (img_width, img_height),
        "num_windows": num_windows,
        "window_size": window_size,
        "stride": stride,
        "full_text_ocr": full_text_ocr,
        "full_text_l2w1": full_text_l2w1,
        "window_results": window_results,
    }
    
    # Cleanup
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def print_final_results(results: Dict[str, Any]) -> None:
    """Print final results in a formatted way."""
    print()
    print("=" * 70)
    print("  📊 FINAL RESULTS")
    print("=" * 70)
    print()
    
    print(f"  Image: {results['image_path']}")
    print(f"  Windows processed: {results['num_windows']}")
    print()
    
    print("-" * 70)
    print("  OCR Result (Agent A only):")
    print("-" * 70)
    print(f"  '{results['full_text_ocr']}'")
    print()
    
    print("-" * 70)
    print("  L2W1 Result (with context-aware correction):")
    print("-" * 70)
    print(f"  '{results['full_text_l2w1']}'")
    print()
    
    # Summary statistics
    total_routed = sum(1 for w in results["window_results"] if w["is_routed"])
    avg_ppl = sum(w["ppl_score"] for w in results["window_results"]) / len(results["window_results"])
    
    print("-" * 70)
    print("  Statistics:")
    print("-" * 70)
    print(f"  Total characters (OCR):  {len(results['full_text_ocr'])}")
    print(f"  Total characters (L2W1): {len(results['full_text_l2w1'])}")
    print(f"  Windows routed to Agent B: {total_routed}/{results['num_windows']}")
    print(f"  Average PPL: {avg_ppl:.2f}")
    print()
    
    if results['full_text_ocr'] != results['full_text_l2w1']:
        print("  ✅ L2W1 made corrections to the OCR result!")
    else:
        print("  ➡️  OCR and L2W1 results are identical")
    
    print()
    print("=" * 70)
    print()


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="L2W1 Sliding Window Inference for Long Text Images",
    )
    parser.add_argument(
        "--image_path",
        type=Path,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Window width in pixels (default: {DEFAULT_WINDOW_SIZE})",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help=f"Stride between windows (default: {DEFAULT_STRIDE})",
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=DEFAULT_TARGET_HEIGHT,
        help=f"Target height for normalization (default: {DEFAULT_TARGET_HEIGHT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print()
    print("=" * 70)
    print("  L2W1 Long Text Sliding Window Inference")
    print("=" * 70)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Resolve image path
    image_path = args.image_path
    if not image_path.exists():
        # Try relative to project root
        alt_path = PROJECT_ROOT / args.image_path
        if alt_path.exists():
            image_path = alt_path
        else:
            print(f"  ❌ Image not found: {args.image_path}")
            return
    
    print(f"  Image: {image_path}")
    print(f"  Window: {args.window_size}px, Stride: {args.stride}px")
    print(f"  Target Height: {args.target_height}px")
    
    # Run inference
    start_time = time.time()
    
    results = run_sliding_window_inference(
        image_path=image_path,
        window_size=args.window_size,
        stride=args.stride,
        target_height=args.target_height,
    )
    
    elapsed = time.time() - start_time
    
    # Print results
    print_final_results(results)
    
    print(f"  Total time: {elapsed:.2f}s")
    print()


if __name__ == "__main__":
    main()

