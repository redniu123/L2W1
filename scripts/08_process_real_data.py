#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Process Real Data Sources for L2W1 Evaluation.

This script converts two types of real-world data sources into the standard
L2W1 test set format (JSON) for batch inference:

1. VisCGEC Dataset: Official benchmark with JSON metadata
2. Custom Handwritten Data: User-provided images with corresponding .txt labels

Usage:
    # Process VisCGEC dataset
    python scripts/08_process_real_data.py --viscgec

    # Process custom handwritten data
    python scripts/08_process_real_data.py --custom

    # Process both
    python scripts/08_process_real_data.py --viscgec --custom
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
from datetime import datetime
from typing import Any, Dict, List, Optional

# =============================================================================
# Configuration
# =============================================================================

# Default paths
DEFAULT_VISCGEC_JSON = PROJECT_ROOT / "data" / "viscgec" / "VisCGEC.json"
DEFAULT_VISCGEC_IMAGES_DIR = PROJECT_ROOT / "data" / "viscgec" / "images"
DEFAULT_VISCGEC_OUTPUT = PROJECT_ROOT / "data" / "viscgec_test_set.json"

DEFAULT_CUSTOM_IMAGES_DIR = PROJECT_ROOT / "data" / "real_handwritten"
DEFAULT_CUSTOM_OUTPUT = PROJECT_ROOT / "data" / "real_test_set.json"

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# VisCGEC Dataset Processing
# =============================================================================


def process_viscgec_dataset(
    viscgec_json: Path,
    images_dir: Path,
    output_json: Path,
) -> List[Dict[str, Any]]:
    """Process VisCGEC dataset from JSON metadata.

    Args:
        viscgec_json: Path to VisCGEC.json metadata file.
        images_dir: Directory containing VisCGEC images.
        output_json: Output JSON path for test set.

    Returns:
        List of sample dictionaries in standard format.
    """
    if not viscgec_json.exists():
        logger.error("VisCGEC JSON file not found: %s", viscgec_json)
        return []

    if not images_dir.exists():
        logger.error("VisCGEC images directory not found: %s", images_dir)
        return []

    logger.info("Loading VisCGEC metadata from: %s", viscgec_json)
    with viscgec_json.open("r", encoding="utf-8") as f:
        metadata_list: List[Dict[str, Any]] = json.load(f)

    logger.info("Found %d entries in VisCGEC metadata", len(metadata_list))

    samples: List[Dict[str, Any]] = []
    matched_count = 0
    skipped_count = 0

    for idx, entry in enumerate(metadata_list):
        img_id = entry.get("img_id")
        if not img_id:
            logger.warning("Entry %d missing img_id, skipping", idx)
            skipped_count += 1
            continue

        # Extract ground truth label (source_ground_truth field)
        label_gt = entry.get("source_ground_truth", "")
        if not label_gt:
            logger.warning("Entry %s missing source_ground_truth, skipping", img_id)
            skipped_count += 1
            continue

        # Try to find matching image file
        image_path: Optional[Path] = None
        for ext in IMAGE_EXTENSIONS:
            candidate = images_dir / f"{img_id}{ext}"
            if candidate.exists():
                image_path = candidate
                break
            # Also try uppercase extensions
            candidate_upper = images_dir / f"{img_id}{ext.upper()}"
            if candidate_upper.exists():
                image_path = candidate_upper
                break

        if image_path is None:
            logger.debug("No image file found for img_id=%s, skipping", img_id)
            skipped_count += 1
            continue

        # Create relative path
        try:
            rel_path = image_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = image_path

        sample: Dict[str, Any] = {
            "id": f"viscgec_{img_id}",
            "image_path": str(rel_path),
            "label_gt": label_gt,
            "ocr_pred": "",  # Will be filled by Agent A during inference
            "context": "",  # Empty for line-level datasets
            "context_left": "",
            "context_right": "",
            "ocr_entropy": 0.0,  # Will be computed during inference
            "ppl_score": 0.0,  # Will be computed during inference
        }
        samples.append(sample)
        matched_count += 1

        # Progress indicator
        if (matched_count + skipped_count) % 50 == 0:
            logger.info(
                "Processed %d/%d entries (matched: %d, skipped: %d)",
                matched_count + skipped_count,
                len(metadata_list),
                matched_count,
                skipped_count,
            )

    logger.info(
        "VisCGEC processing complete: %d matched, %d skipped",
        matched_count,
        skipped_count,
    )

    # Save test set
    if samples:
        save_test_set(samples, output_json, source="VisCGEC Official Benchmark")
        logger.info("VisCGEC test set saved to: %s", output_json)
    else:
        logger.warning("No samples generated for VisCGEC dataset")

    return samples


# =============================================================================
# Custom Handwritten Data Processing
# =============================================================================


def process_custom_handwritten(
    images_dir: Path,
    output_json: Path,
) -> List[Dict[str, Any]]:
    """Process custom handwritten data with .txt label files.

    Args:
        images_dir: Directory containing images and corresponding .txt files.
        output_json: Output JSON path for test set.

    Returns:
        List of sample dictionaries in standard format.
    """
    if not images_dir.exists():
        logger.error("Custom handwritten images directory not found: %s", images_dir)
        return []

    logger.info("Scanning custom handwritten images from: %s", images_dir)

    # Find all image files
    image_files: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))

    # Also check subdirectories (one level deep)
    for subdir in images_dir.iterdir():
        if subdir.is_dir():
            for ext in IMAGE_EXTENSIONS:
                image_files.extend(subdir.glob(f"*{ext}"))
                image_files.extend(subdir.glob(f"*{ext.upper()}"))

    # Remove duplicates and sort
    image_files = sorted(set(image_files))

    if not image_files:
        logger.warning("No images found in %s", images_dir)
        logger.info("Supported extensions: %s", IMAGE_EXTENSIONS)
        return []

    logger.info("Found %d image files", len(image_files))

    samples: List[Dict[str, Any]] = []
    processed_count = 0
    skipped_count = 0

    for img_path in image_files:
        # Find corresponding .txt file
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            logger.warning(
                "Label file not found for %s, skipping",
                img_path.name,
            )
            skipped_count += 1
            continue

        # Read label from .txt file
        try:
            with txt_path.open("r", encoding="utf-8") as f:
                label_gt = f.read().strip()
        except Exception as e:
            logger.warning(
                "Failed to read label file %s: %s, skipping",
                txt_path,
                e,
            )
            skipped_count += 1
            continue

        if not label_gt:
            logger.warning("Empty label in %s, skipping", txt_path)
            skipped_count += 1
            continue

        # Create relative path
        try:
            rel_path = img_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = img_path

        # Generate sample ID from filename
        sample_id = f"custom_{img_path.stem}"

        sample: Dict[str, Any] = {
            "id": sample_id,
            "image_path": str(rel_path),
            "label_gt": label_gt,
            "ocr_pred": "",  # Will be filled by Agent A during inference
            "context": "",  # Empty for line-level datasets
            "context_left": "",
            "context_right": "",
            "ocr_entropy": 0.0,  # Will be computed during inference
            "ppl_score": 0.0,  # Will be computed during inference
        }
        samples.append(sample)
        processed_count += 1

        # Progress indicator
        if processed_count % 20 == 0:
            logger.info(
                "Processed %d/%d images (skipped: %d)",
                processed_count,
                len(image_files),
                skipped_count,
            )

    logger.info(
        "Custom handwritten processing complete: %d processed, %d skipped",
        processed_count,
        skipped_count,
    )

    # Save test set
    if samples:
        save_test_set(samples, output_json, source="Custom Handwritten Data")
        logger.info("Custom handwritten test set saved to: %s", output_json)
    else:
        logger.warning("No samples generated for custom handwritten data")

    return samples


# =============================================================================
# JSON Generation
# =============================================================================


def save_test_set(
    samples: List[Dict[str, Any]],
    output_path: Path,
    source: str = "Real Data Source",
) -> None:
    """Save test set to JSON file.

    Args:
        samples: List of sample dictionaries.
        output_path: Output JSON path.
        source: Source description for metadata.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create output structure
    output_data: Dict[str, Any] = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_samples": len(samples),
            "source": source,
            "description": "Real-world test set for L2W1 evaluation",
        },
        "samples": samples,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info("Test set saved to: %s", output_path)


def print_summary(samples: List[Dict[str, Any]], source_name: str) -> None:
    """Print summary of generated test set.

    Args:
        samples: List of sample dictionaries.
        source_name: Name of the data source.
    """
    print()
    print("=" * 70)
    print(f"  📊 {source_name} Summary")
    print("=" * 70)

    total = len(samples)
    print(f"  Total samples:   {total}")

    if samples:
        # Show label length statistics
        label_lengths = [len(s["label_gt"]) for s in samples]
        avg_length = sum(label_lengths) / len(label_lengths) if label_lengths else 0
        max_length = max(label_lengths) if label_lengths else 0
        min_length = min(label_lengths) if label_lengths else 0

        print(f"  Avg label length: {avg_length:.1f} chars")
        print(f"  Min label length: {min_length} chars")
        print(f"  Max label length: {max_length} chars")

        # Show sample preview
        print()
        print("  📋 Sample Preview (first 3):")
        print("-" * 70)
        for i, sample in enumerate(samples[:3]):
            print(f"  [{i + 1}] {sample['id']}")
            print(f"      Image: {sample['image_path']}")
            print(f"      Label: '{sample['label_gt'][:50]}...'")
            print()

    print("=" * 70)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process real data sources for L2W1 evaluation",
    )
    parser.add_argument(
        "--viscgec",
        action="store_true",
        help="Process VisCGEC dataset",
    )
    parser.add_argument(
        "--custom",
        action="store_true",
        help="Process custom handwritten data",
    )
    parser.add_argument(
        "--viscgec_json",
        type=Path,
        default=DEFAULT_VISCGEC_JSON,
        help=f"Path to VisCGEC.json (default: {DEFAULT_VISCGEC_JSON})",
    )
    parser.add_argument(
        "--viscgec_images",
        type=Path,
        default=DEFAULT_VISCGEC_IMAGES_DIR,
        help=f"VisCGEC images directory (default: {DEFAULT_VISCGEC_IMAGES_DIR})",
    )
    parser.add_argument(
        "--viscgec_output",
        type=Path,
        default=DEFAULT_VISCGEC_OUTPUT,
        help=f"VisCGEC output JSON (default: {DEFAULT_VISCGEC_OUTPUT})",
    )
    parser.add_argument(
        "--custom_images",
        type=Path,
        default=DEFAULT_CUSTOM_IMAGES_DIR,
        help=f"Custom handwritten images directory (default: {DEFAULT_CUSTOM_IMAGES_DIR})",
    )
    parser.add_argument(
        "--custom_output",
        type=Path,
        default=DEFAULT_CUSTOM_OUTPUT,
        help=f"Custom handwritten output JSON (default: {DEFAULT_CUSTOM_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for real data processing."""
    args = parse_args()

    print()
    print("=" * 70)
    print("  L2W1 Real Data Processor")
    print("=" * 70)
    print()

    if not args.viscgec and not args.custom:
        print("  ⚠️  No processing mode specified!")
        print()
        print("  Usage:")
        print("    python scripts/08_process_real_data.py --viscgec")
        print("    python scripts/08_process_real_data.py --custom")
        print("    python scripts/08_process_real_data.py --viscgec --custom")
        print()
        return

    # Process VisCGEC dataset
    if args.viscgec:
        print("-" * 70)
        print("  Processing VisCGEC Dataset...")
        print("-" * 70)
        print(f"  JSON file:      {args.viscgec_json}")
        print(f"  Images dir:     {args.viscgec_images}")
        print(f"  Output JSON:    {args.viscgec_output}")
        print()

        viscgec_samples = process_viscgec_dataset(
            viscgec_json=args.viscgec_json,
            images_dir=args.viscgec_images,
            output_json=args.viscgec_output,
        )

        if viscgec_samples:
            print_summary(viscgec_samples, "VisCGEC Dataset")
        print()

    # Process custom handwritten data
    if args.custom:
        print("-" * 70)
        print("  Processing Custom Handwritten Data...")
        print("-" * 70)
        print(f"  Images dir:     {args.custom_images}")
        print(f"  Output JSON:    {args.custom_output}")
        print()

        custom_samples = process_custom_handwritten(
            images_dir=args.custom_images,
            output_json=args.custom_output,
        )

        if custom_samples:
            print_summary(custom_samples, "Custom Handwritten Data")
        print()

    # Final summary
    print("=" * 70)
    print("  ✓ Real data processing complete!")
    print("=" * 70)
    print()
    print("  Next steps:")
    print("  1. Review the generated JSON files")
    print("  2. Run batch inference on the test sets:")
    if args.viscgec:
        print(
            f"     python scripts/04_batch_inference.py --test_set {args.viscgec_output}"
        )
    if args.custom:
        print(
            f"     python scripts/04_batch_inference.py --test_set {args.custom_output}"
        )
    print()


if __name__ == "__main__":
    main()
