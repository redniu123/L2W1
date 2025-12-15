#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Synthesize Text Line Test for L2W1.

This script creates synthetic text lines by horizontally concatenating
single character images, then tests the L2W1 pipeline on the result.

Strategy:
- Concatenate cropped single-character images into a text line
- This restores proper aspect ratio for OCR (solves shape problem)
- Enables Router's context-aware PPL analysis
- Demonstrates L2W1's strength in context-based correction

Usage:
    python scripts/06_synthesize_line_test.py
    python scripts/06_synthesize_line_test.py --num_chars 10 --spacing 15
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
import json
import logging
import os
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "casia_sample"
DEFAULT_OUTPUT_IMAGE = PROJECT_ROOT / "data" / "synthetic_text_line.jpg"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "data" / "synthetic_line_test.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CER Calculation
# =============================================================================


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER) using Levenshtein distance.

    CER = (S + D + I) / N
    where:
        S = substitutions
        D = deletions
        I = insertions
        N = length of reference

    Args:
        reference: Ground truth text.
        hypothesis: Predicted text.

    Returns:
        CER value (0.0 = perfect, 1.0 = completely wrong).
    """
    if len(reference) == 0:
        return 1.0 if len(hypothesis) > 0 else 0.0

    # Dynamic programming for edit distance
    m, n = len(reference), len(hypothesis)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # deletion
                    dp[i][j - 1],  # insertion
                    dp[i - 1][j - 1],  # substitution
                )

    edit_distance = dp[m][n]
    cer = edit_distance / len(reference)

    return min(cer, 1.0)  # Cap at 1.0


def calculate_accuracy(reference: str, hypothesis: str) -> float:
    """Calculate character-level accuracy.

    Args:
        reference: Ground truth text.
        hypothesis: Predicted text.

    Returns:
        Accuracy (0.0-1.0).
    """
    if len(reference) == 0:
        return 1.0 if len(hypothesis) == 0 else 0.0

    # Simple character-by-character comparison
    correct = sum(1 for r, h in zip(reference, hypothesis) if r == h)
    accuracy = correct / max(len(reference), len(hypothesis))

    return accuracy


# =============================================================================
# Label Extraction
# =============================================================================

CHINESE_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def extract_label_from_filename(filename: str) -> Optional[str]:
    """Extract Chinese character label from filename."""
    name = Path(filename).stem
    matches = CHINESE_CHAR_PATTERN.findall(name)
    if matches:
        return matches[0]
    return None


# =============================================================================
# Image Synthesis
# =============================================================================


def load_char_images(
    input_dir: Path,
    num_chars: Optional[int] = None,
    shuffle: bool = True,
) -> List[Tuple[np.ndarray, str]]:
    """Load character images and their labels from directory.

    Args:
        input_dir: Directory containing character images.
        num_chars: Maximum number of characters to load.
        shuffle: Whether to shuffle the selection.

    Returns:
        List of (image_array, label) tuples.
    """
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return []

    # Find all image files
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))

    image_files = sorted(set(image_files))

    if not image_files:
        logger.warning(f"No images found in {input_dir}")
        return []

    logger.info(f"Found {len(image_files)} images in {input_dir}")

    # Shuffle if requested
    if shuffle:
        random.shuffle(image_files)

    # Limit number of characters
    if num_chars is not None:
        image_files = image_files[:num_chars]

    # Load images and extract labels
    char_images = []
    for img_path in image_files:
        # Extract label
        label = extract_label_from_filename(img_path.name)
        if label is None:
            logger.warning(f"Cannot extract label from: {img_path.name}, skipping")
            continue

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Failed to load image: {img_path}")
            continue

        char_images.append((img, label))

    logger.info(f"Loaded {len(char_images)} character images")
    return char_images


def normalize_char_height(
    images: List[Tuple[np.ndarray, str]],
    target_height: int = 64,
) -> List[Tuple[np.ndarray, str]]:
    """Normalize all character images to the same height.

    Args:
        images: List of (image, label) tuples.
        target_height: Target height in pixels.

    Returns:
        Normalized images with same height.
    """
    normalized = []

    for img, label in images:
        h, w = img.shape[:2]
        if h == 0:
            continue

        # Calculate new width maintaining aspect ratio
        scale = target_height / h
        new_w = int(w * scale)
        new_h = target_height

        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        normalized.append((resized, label))

    return normalized


def synthesize_text_line(
    char_images: List[Tuple[np.ndarray, str]],
    spacing_range: Tuple[int, int] = (10, 20),
    padding: int = 20,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[np.ndarray, str]:
    """Synthesize a text line by horizontally concatenating character images.

    Args:
        char_images: List of (image, label) tuples.
        spacing_range: (min, max) spacing between characters in pixels.
        padding: Padding around the text line.
        background_color: Background color (BGR).

    Returns:
        Tuple of (synthesized_image, ground_truth_text).
    """
    if not char_images:
        return np.zeros((64, 200, 3), dtype=np.uint8), ""

    # Normalize heights
    char_images = normalize_char_height(char_images, target_height=64)

    if not char_images:
        return np.zeros((64, 200, 3), dtype=np.uint8), ""

    # Calculate total width
    total_width = padding * 2  # Left and right padding
    for img, _ in char_images:
        total_width += img.shape[1]
        total_width += random.randint(*spacing_range)  # Random spacing

    # Remove last spacing
    total_width -= random.randint(*spacing_range)

    # Get height (all should be same after normalization)
    height = char_images[0][0].shape[0] + padding * 2

    # Create canvas
    canvas = np.full((height, total_width, 3), background_color, dtype=np.uint8)

    # Place characters
    x_offset = padding
    ground_truth = ""

    for i, (img, label) in enumerate(char_images):
        h, w = img.shape[:2]
        y_offset = padding

        # Place character
        canvas[y_offset : y_offset + h, x_offset : x_offset + w] = img

        # Update ground truth
        ground_truth += label

        # Move to next position
        x_offset += w
        if i < len(char_images) - 1:
            x_offset += random.randint(*spacing_range)

    logger.info(f"Synthesized text line: {total_width}x{height}, GT: '{ground_truth}'")

    return canvas, ground_truth


# =============================================================================
# Pipeline Inference
# =============================================================================


def run_l2w1_inference(
    image: np.ndarray,
    ground_truth: str,
) -> Dict[str, Any]:
    """Run L2W1 pipeline on the synthesized text line.

    Args:
        image: Synthesized text line image (BGR numpy array).
        ground_truth: Ground truth text.

    Returns:
        Results dictionary.
    """
    results = {
        "ground_truth": ground_truth,
        "agent_a_pred": "",
        "final_pred": "",
        "ppl_score": 0.0,
        "cer_agent_a": 1.0,
        "cer_final": 1.0,
        "accuracy_agent_a": 0.0,
        "accuracy_final": 0.0,
        "is_improved": False,
    }

    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Initialize agents
    agent_a = None
    router = None
    agent_b = None

    # Agent A
    print()
    print("-" * 70)
    print("  [Stage 1] Agent A - OCR Recognition")
    print("-" * 70)

    try:
        from core.agent_a import AgentA

        agent_a = AgentA(use_gpu=True)
        logger.info("Agent A initialized")

        # Run OCR (with detection enabled for text line)
        line_results = agent_a.inference(image_rgb, skip_detection=False)

        if line_results:
            # Concatenate all detected text
            agent_a_text = "".join([r.get("text", "") for r in line_results])
            results["agent_a_pred"] = agent_a_text
            results["final_pred"] = agent_a_text  # Default

            # Get entropy
            avg_entropy = sum(r.get("avg_entropy", 0) for r in line_results) / len(
                line_results
            )

            print(f"  Agent A Result: '{agent_a_text}'")
            print(f"  Avg Entropy: {avg_entropy:.4f}")

            # Calculate Agent A metrics
            results["cer_agent_a"] = calculate_cer(ground_truth, agent_a_text)
            results["accuracy_agent_a"] = calculate_accuracy(ground_truth, agent_a_text)
        else:
            print("  Agent A: No text detected")

    except Exception as e:
        logger.error(f"Agent A failed: {e}")
        print(f"  Agent A Error: {e}")

    # Router
    print()
    print("-" * 70)
    print("  [Stage 2] Router - Semantic Perplexity Analysis")
    print("-" * 70)

    agent_a_text = results["agent_a_pred"]
    should_route = False

    if agent_a_text:
        try:
            from core.router import Router

            router = Router()
            logger.info("Router initialized")

            ppl_score = router.compute_ppl(agent_a_text)
            results["ppl_score"] = ppl_score

            print(f"  Text: '{agent_a_text}'")
            print(f"  PPL Score: {ppl_score:.2f}")

            # Routing decision
            PPL_THRESHOLD = 100.0
            should_route = ppl_score > PPL_THRESHOLD

            if should_route:
                print(f"  Decision: PPL > {PPL_THRESHOLD} -> ROUTE TO AGENT B")
            else:
                print(f"  Decision: PPL <= {PPL_THRESHOLD} -> KEEP AGENT A RESULT")

        except Exception as e:
            logger.error(f"Router failed: {e}")
            print(f"  Router Error: {e}")

    # Agent B (if routed)
    if should_route and agent_a_text:
        print()
        print("-" * 70)
        print("  [Stage 3] Agent B - Visual-Semantic Correction")
        print("-" * 70)

        try:
            from core.agent_b import AgentB

            agent_b = AgentB(
                model_path="Qwen/Qwen2-VL-2B-Instruct",
                load_in_4bit=True,
            )
            logger.info("Agent B initialized")

            # For text line, we process character by character
            # or we can send the whole line for holistic correction
            corrected_text = ""

            # Simple approach: process the whole image with context
            print("  Processing full text line with V-CoT...")

            # Use full text as context
            corrected = agent_b.inference(
                crop_image=pil_image,
                context_left="",
                context_right="",
                ocr_pred=agent_a_text,
            )

            if corrected and corrected != "KEEP":
                corrected_text = corrected
                results["final_pred"] = corrected_text
                print(f"  Agent B Correction: '{corrected_text}'")
            else:
                print(f"  Agent B: Kept original prediction")

        except Exception as e:
            logger.error(f"Agent B failed: {e}")
            print(f"  Agent B Error: {e}")

    # Calculate final metrics
    final_pred = results["final_pred"]
    results["cer_final"] = calculate_cer(ground_truth, final_pred)
    results["accuracy_final"] = calculate_accuracy(ground_truth, final_pred)
    results["is_improved"] = results["cer_final"] < results["cer_agent_a"]

    # Cleanup
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# =============================================================================
# Pretty Print Results
# =============================================================================


def print_results(results: Dict[str, Any]) -> None:
    """Print results in a formatted way."""
    print()
    print("=" * 70)
    print("  📊 SYNTHESIS LINE TEST RESULTS")
    print("=" * 70)
    print()

    gt = results["ground_truth"]
    pred_a = results["agent_a_pred"]
    pred_final = results["final_pred"]

    print(f"  Ground Truth:      '{gt}'")
    print(f"  Agent A (OCR):     '{pred_a}'")
    print(f"  L2W1 Final:        '{pred_final}'")
    print()

    print("-" * 70)
    print("  Metrics Comparison")
    print("-" * 70)
    print()
    print("  | Metric | Agent A | L2W1 (Final) | Δ |")
    print("  |--------|---------|--------------|---|")

    cer_a = results["cer_agent_a"]
    cer_f = results["cer_final"]
    cer_delta = cer_f - cer_a
    cer_sign = "↓" if cer_delta < 0 else ("↑" if cer_delta > 0 else "=")

    acc_a = results["accuracy_agent_a"]
    acc_f = results["accuracy_final"]
    acc_delta = acc_f - acc_a
    acc_sign = "↑" if acc_delta > 0 else ("↓" if acc_delta < 0 else "=")

    print(
        f"  | CER | {cer_a * 100:.2f}% | {cer_f * 100:.2f}% | {cer_sign} {abs(cer_delta) * 100:.2f}% |"
    )
    print(
        f"  | Accuracy | {acc_a * 100:.2f}% | {acc_f * 100:.2f}% | {acc_sign} {abs(acc_delta) * 100:.2f}% |"
    )
    print()

    print(f"  PPL Score: {results['ppl_score']:.2f}")
    print()

    if results["is_improved"]:
        print("  ✅ L2W1 IMPROVED the recognition!")
    elif cer_a == cer_f:
        print("  ➡️  No change in accuracy")
    else:
        print("  ⚠️  L2W1 did not improve (or made it worse)")

    print()
    print("=" * 70)
    print()


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize text line and test L2W1 pipeline",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory with character images (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output_image",
        type=Path,
        default=DEFAULT_OUTPUT_IMAGE,
        help=f"Output synthesized image path (default: {DEFAULT_OUTPUT_IMAGE})",
    )
    parser.add_argument(
        "--num_chars",
        type=int,
        default=10,
        help="Number of characters to synthesize (default: 10)",
    )
    parser.add_argument(
        "--spacing",
        type=int,
        default=15,
        help="Average spacing between characters (default: 15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print()
    print("=" * 70)
    print("  L2W1 Synthetic Text Line Test")
    print("=" * 70)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Input: {args.input_dir}")
    print(f"  Num Chars: {args.num_chars}")
    print(f"  Spacing: {args.spacing}px (±5)")
    print()

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Check input directory
    if not args.input_dir.exists():
        print()
        print("  ❌ Input directory not found!")
        print(f"     Path: {args.input_dir}")
        print()
        print("  To use this script:")
        print(f"  1. Create directory: mkdir -p {args.input_dir}")
        print("  2. Add character images with format: 字_序号.jpg")
        print("  3. Run this script again")
        print()
        return

    # Load character images
    print("-" * 70)
    print("  Loading character images...")
    print("-" * 70)

    char_images = load_char_images(
        args.input_dir,
        num_chars=args.num_chars,
        shuffle=True,
    )

    if not char_images:
        print("  ❌ No character images loaded!")
        return

    # Synthesize text line
    print()
    print("-" * 70)
    print("  Synthesizing text line...")
    print("-" * 70)

    spacing_range = (args.spacing - 5, args.spacing + 5)
    synthetic_image, ground_truth = synthesize_text_line(
        char_images,
        spacing_range=spacing_range,
        padding=20,
    )

    # Save synthesized image
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_image), synthetic_image)
    print(f"  Saved: {args.output_image}")
    print(f"  Size: {synthetic_image.shape[1]}x{synthetic_image.shape[0]}")
    print(f"  Ground Truth: '{ground_truth}'")

    # Run L2W1 inference
    print()
    print("=" * 70)
    print("  Running L2W1 Pipeline on Synthetic Text Line")
    print("=" * 70)

    results = run_l2w1_inference(synthetic_image, ground_truth)

    # Print results
    print_results(results)

    # Save results to JSON
    results_json = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "num_chars": args.num_chars,
            "spacing": args.spacing,
            "image_path": str(args.output_image),
        },
        "results": {
            "ground_truth": results["ground_truth"],
            "agent_a_pred": results["agent_a_pred"],
            "final_pred": results["final_pred"],
            "ppl_score": results["ppl_score"],
            "cer_agent_a": results["cer_agent_a"],
            "cer_final": results["cer_final"],
            "accuracy_agent_a": results["accuracy_agent_a"],
            "accuracy_final": results["accuracy_final"],
            "is_improved": results["is_improved"],
        },
    }

    with open(DEFAULT_OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

    print(f"  Results saved: {DEFAULT_OUTPUT_JSON}")
    print()


if __name__ == "__main__":
    main()
