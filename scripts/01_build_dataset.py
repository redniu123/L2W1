#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build Agent B training dataset from raw images.

Spec Reference: L2W1-DE-003 (The Builder Script)

This script:
1. Scans RAW_IMAGE_DIR for images
2. Processes each image through DataProcessor pipeline
3. Aggregates results into agent_b_train.json

Usage:
    python scripts/01_build_dataset.py

Demo Mode:
    If RAW_IMAGE_DIR is empty or doesn't exist, a dummy test image
    will be auto-generated to verify the pipeline works correctly.
"""

# =============================================================================
# Environment Setup (MUST be before any Paddle imports)
# =============================================================================
import os

# Set environment variables to prevent cudnn loading when using CPU
# This must be done BEFORE importing any PaddleOCR/PaddlePaddle modules
# Default to GPU mode, disable only if USE_CPU=true or USE_GPU=false
use_cpu = (
    os.getenv("USE_CPU", "false").lower() == "true"
    or os.getenv("USE_GPU", "true").lower() == "false"
)
if use_cpu:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    # Prevent PaddlePaddle from trying to load cudnn on CPU-only systems
    os.environ.setdefault("FLAGS_cudnn_deterministic", "1")

# =============================================================================
# Path Patching (L2W1-DE-003 Requirement)
# Since this script is in scripts/ while modules are in data/,
# we must add the project root to sys.path
# =============================================================================
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Standard Imports (after path patching)
# =============================================================================
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# =============================================================================
# Configuration Area (L2W1-DE-003)
# =============================================================================

# Input: Directory containing raw images with GT annotations
RAW_IMAGE_DIR = PROJECT_ROOT / "raw_data" / "images"

# Output: Directory for processed crops and metadata
OUTPUT_DIR = PROJECT_ROOT / "output"

# Output JSON file for Agent B training
OUTPUT_JSON = OUTPUT_DIR / "agent_b_train.json"

# Crops will be saved here
CROPS_DIR = OUTPUT_DIR / "crops"

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Demo mode settings
DEMO_IMAGE_PATH = OUTPUT_DIR / "demo_test_image.png"
DEMO_GT_TEXT = "测试样本123"

# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Demo Mode: Generate Dummy Image
# =============================================================================


def generate_demo_image(
    output_path: Path,
    text: str = "测试样本123",
    size: Tuple[int, int] = (800, 100),
) -> Path:
    """Generate a dummy test image with Chinese text for demo mode.

    Creates a white background image with black text for testing
    the pipeline without real data.

    Args:
        output_path: Path to save the generated image.
        text: Text to render on the image.
        size: Image size as (width, height).

    Returns:
        Path to the generated image.
    """
    width, height = size

    # Create white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Add black text (simple approach - actual rendering may vary)
    # Using OpenCV's putText with a basic font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 2
    color = (0, 0, 0)  # Black

    # Calculate text position (centered)
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = (width - text_size[0]) // 2
    y = (height + text_size[1]) // 2

    cv2.putText(image, text, (x, y), font, font_scale, color, thickness)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save image
    cv2.imwrite(str(output_path), image)
    logger.info(f"[DEMO] Generated dummy image: {output_path}")

    return output_path


def load_gt_annotations(image_dir: Path) -> List[Tuple[Path, str]]:
    """Load image paths and their ground truth annotations.

    This function looks for a gt.txt file in the image directory.
    Format: image_filename<tab>ground_truth_text

    If no gt.txt exists, returns empty list (will trigger demo mode).

    Args:
        image_dir: Directory containing images and gt.txt.

    Returns:
        List of (image_path, gt_text) tuples.
    """
    gt_file = image_dir / "gt.txt"
    pairs: List[Tuple[Path, str]] = []

    if not gt_file.exists():
        logger.warning(f"No gt.txt found in {image_dir}")
        return pairs

    with open(gt_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse: filename<tab>gt_text
            parts = line.split("\t", 1)
            if len(parts) != 2:
                logger.warning(f"Skipping malformed line {line_num}: {line}")
                continue

            filename, gt_text = parts
            image_path = image_dir / filename

            if image_path.exists():
                pairs.append((image_path, gt_text))
            else:
                logger.warning(f"Image not found: {image_path}")

    return pairs


def scan_images_without_gt(image_dir: Path) -> List[Path]:
    """Scan directory for images (fallback when no gt.txt).

    Args:
        image_dir: Directory to scan.

    Returns:
        List of image paths found.
    """
    if not image_dir.exists():
        return []

    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(image_dir.glob(f"*{ext}"))
        images.extend(image_dir.glob(f"*{ext.upper()}"))

    return sorted(images)


# =============================================================================
# Main Processing Logic
# =============================================================================


def run_pipeline(
    image_gt_pairs: List[Tuple[Path, str]],
    output_dir: Path,
    crops_dir: Path,
    use_gpu: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run the full data processing pipeline.

    Args:
        image_gt_pairs: List of (image_path, gt_text) tuples.
        output_dir: Base output directory.
        crops_dir: Directory for cropped images.
        use_gpu: Whether to use GPU for OCR.

    Returns:
        Tuple of (all_results, failed_items).
    """
    # Lazy import to avoid import errors if dependencies missing
    from data.preprocess import DataProcessor

    logger.info("=" * 60)
    logger.info("Initializing DataProcessor (Agent A: PaddleOCR)...")
    logger.info("=" * 60)

    try:
        processor = DataProcessor(
            output_dir=crops_dir,
            target_size=336,
            context_alpha=0.3,
            use_gpu=use_gpu,
        )
    except Exception as e:
        logger.error(f"Failed to initialize DataProcessor: {e}")
        raise

    logger.info(f"DataProcessor initialized successfully.")
    logger.info(f"  - Output crops: {crops_dir}")
    logger.info(f"  - Target size: 336x336")
    logger.info(f"  - Context alpha: 0.3")
    logger.info("")

    all_results: List[Dict[str, Any]] = []
    failed_items: List[Dict[str, Any]] = []
    total = len(image_gt_pairs)

    logger.info(f"Processing {total} image(s)...")
    logger.info("-" * 60)

    for idx, (image_path, gt_text) in enumerate(image_gt_pairs, 1):
        logger.info(f"[{idx}/{total}] Processing: {image_path.name}")
        logger.info(f"         GT: '{gt_text}'")

        try:
            results = processor.process_single_image(
                image_path=image_path,
                ground_truth_text=gt_text,
            )
            all_results.extend(results)
            logger.info(f"         ✓ Generated {len(results)} character crops")

        except Exception as e:
            logger.error(f"         ✗ Failed: {e}")
            failed_items.append(
                {
                    "path": str(image_path),
                    "gt_text": gt_text,
                    "error": str(e),
                }
            )

    return all_results, failed_items


def save_results(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Save processing results to JSON file.

    Args:
        results: List of metadata dictionaries.
        output_path: Path to save JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata header
    output_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_samples": len(results),
            "spec_version": "L2W1-DE-002",
        },
        "samples": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved results to: {output_path}")


def print_summary(
    results: List[Dict[str, Any]],
    failed: List[Dict[str, Any]],
    crops_dir: Path,
    output_json: Path,
) -> None:
    """Print processing summary.

    Args:
        results: Successful processing results.
        failed: Failed items.
        crops_dir: Directory containing crops.
        output_json: Path to output JSON.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info("")

    # Count sample types
    type_counts: Dict[str, int] = {}
    for r in results:
        sample_type = r.get("sample_type", "UNKNOWN")
        type_counts[sample_type] = type_counts.get(sample_type, 0) + 1

    logger.info(f"Total character crops generated: {len(results)}")
    logger.info("")
    logger.info("Sample type distribution:")
    for sample_type, count in sorted(type_counts.items()):
        logger.info(f"  - {sample_type}: {count}")

    if failed:
        logger.info("")
        logger.warning(f"Failed items: {len(failed)}")
        for item in failed:
            logger.warning(f"  - {item['path']}: {item['error']}")

    logger.info("")
    logger.info("Output locations:")
    logger.info(f"  📁 Crops directory: {crops_dir}")
    logger.info(f"  📄 Training JSON:   {output_json}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review generated crops in the crops directory")
    logger.info("  2. Use agent_b_train.json for Agent B fine-tuning")
    logger.info("=" * 60)


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the build script."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("L2W1 Dataset Builder - Phase 1")
    logger.info("Spec: L2W1-DE-003 (The Builder Script)")
    logger.info("=" * 60)
    logger.info("")

    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    # Try to load real data
    image_gt_pairs: List[Tuple[Path, str]] = []
    demo_mode = False

    if RAW_IMAGE_DIR.exists():
        # First try to load from gt.txt
        image_gt_pairs = load_gt_annotations(RAW_IMAGE_DIR)

        if not image_gt_pairs:
            # Fallback: scan for images (user will need to provide GT)
            images = scan_images_without_gt(RAW_IMAGE_DIR)
            if images:
                logger.warning(
                    f"Found {len(images)} images but no gt.txt. "
                    "Please create raw_data/images/gt.txt with format: "
                    "filename<tab>ground_truth_text"
                )

    # Demo mode: generate dummy image if no real data
    if not image_gt_pairs:
        logger.info("")
        logger.info("=" * 60)
        logger.info("[DEMO MODE] No input data found!")
        logger.info("Generating dummy test image for pipeline verification...")
        logger.info("=" * 60)
        logger.info("")

        demo_mode = True
        demo_image = generate_demo_image(DEMO_IMAGE_PATH, DEMO_GT_TEXT)
        image_gt_pairs = [(demo_image, DEMO_GT_TEXT)]

        logger.info(f"Demo image: {demo_image}")
        logger.info(f"Demo GT text: '{DEMO_GT_TEXT}'")
        logger.info("")
        logger.info("To use real data, create:")
        logger.info(f"  1. Directory: {RAW_IMAGE_DIR}")
        logger.info(f"  2. Place images in the directory")
        logger.info(f"  3. Create gt.txt with: filename<tab>ground_truth_text")
        logger.info("")
    else:
        logger.info(f"Found {len(image_gt_pairs)} image-GT pairs to process.")

    # Determine GPU usage: default to True, disable if USE_CPU=true or USE_GPU=false
    use_gpu = not (
        os.getenv("USE_CPU", "false").lower() == "true"
        or os.getenv("USE_GPU", "true").lower() == "false"
    )

    # Run the pipeline
    try:
        results, failed = run_pipeline(
            image_gt_pairs=image_gt_pairs,
            output_dir=OUTPUT_DIR,
            crops_dir=CROPS_DIR,
            use_gpu=use_gpu,  # Default: True (use GPU), set USE_CPU=true to disable
        )
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please ensure all dependencies are installed:")
        logger.error("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

    # Save results
    if results:
        save_results(results, OUTPUT_JSON)
    else:
        logger.warning("No results to save (all processing failed or no valid crops)")

    # Print summary
    print_summary(results, failed, CROPS_DIR, OUTPUT_JSON)

    if demo_mode:
        logger.info("")
        logger.info("💡 TIP: This was a DEMO run with synthetic data.")
        logger.info("   The OCR results may not be meaningful for the dummy image.")
        logger.info("   Use real handwritten text images for actual training data.")


if __name__ == "__main__":
    main()
