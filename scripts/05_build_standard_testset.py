#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build Standard Test Set from CASIA-HWDB or Similar Dataset.

This script scans a directory of handwritten character images and generates
a test set JSON file for batch evaluation.

Expected filename format: 字符_序号.jpg (e.g., 测_123.jpg)

Usage:
    python scripts/05_build_standard_testset.py
    python scripts/05_build_standard_testset.py --input_dir data/casia_sample
    python scripts/05_build_standard_testset.py --auto_unknown  # Skip manual input
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
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Configuration
# =============================================================================

# Default paths
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "casia_sample"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "data" / "casia_test_set.json"

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Regex pattern for Chinese characters
CHINESE_CHAR_PATTERN = re.compile(r'[\u4e00-\u9fff]')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Label Extraction
# =============================================================================

def extract_label_from_filename(filename: str) -> Optional[str]:
    """Extract label from filename (Universal Support).
    
    Logic:
    1. Split filename by underscore '_'.
    2. The first part is ALWAYS the label, regardless of language.
    
    Examples:
    - 测_123.jpg -> "测"
    - h_00001.jpg -> "h"
    - ~_00201.jpg -> "~"
    
    Args:
        filename: Image filename (without path).
    
    Returns:
        Extracted single-character label, or None if not found.
    """
    # Remove extension
    name = os.path.splitext(os.path.basename(filename))[0]
    
    # Strategy: Split by underscore
    # This assumes format "Label_ID.jpg" or "Label_Anything.jpg"
    parts = name.split('_')
    
    if len(parts) >= 1:
        candidate = parts[0]
        # Valid label check: strictly length 1 (single char)
        if len(candidate) == 1:
            return candidate
            
    # Fallback (Legacy): Regex for Chinese if underscore logic fails
    chinese_match = CHINESE_CHAR_PATTERN.search(name)
    if chinese_match:
        return chinese_match.group(0)
        
    return None


def prompt_for_label(image_path: Path, auto_unknown: bool = False) -> str:
    """Prompt user to input label for an image.
    
    Args:
        image_path: Path to the image.
        auto_unknown: If True, skip prompt and return "UNKNOWN".
        
    Returns:
        User-provided label or "UNKNOWN".
    """
    if auto_unknown:
        return "UNKNOWN"
    
    print()
    print(f"  ⚠️  Cannot extract label from: {image_path.name}")
    print(f"      Full path: {image_path}")
    print()
    
    try:
        user_input = input("      Enter the correct label (or press Enter for UNKNOWN): ").strip()
        if user_input:
            # Take only the first character if multiple entered
            if len(user_input) > 1:
                print(f"      Note: Using first character '{user_input[0]}' from input")
            return user_input[0]
        else:
            return "UNKNOWN"
    except (EOFError, KeyboardInterrupt):
        print()
        return "UNKNOWN"


# =============================================================================
# Directory Scanning
# =============================================================================

def scan_image_directory(
    input_dir: Path,
    auto_unknown: bool = False,
) -> List[Dict[str, Any]]:
    """Scan directory for images and extract labels.
    
    Args:
        input_dir: Directory containing images.
        auto_unknown: If True, mark unparseable labels as UNKNOWN without prompting.
        
    Returns:
        List of sample dictionaries.
    """
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        logger.info(f"Please create the directory and add images:")
        logger.info(f"  mkdir -p {input_dir}")
        return []
    
    samples = []
    unknown_count = 0
    
    # Find all image files
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    # Also check subdirectories (one level deep)
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            for ext in IMAGE_EXTENSIONS:
                image_files.extend(subdir.glob(f"*{ext}"))
                image_files.extend(subdir.glob(f"*{ext.upper()}"))
    
    # Remove duplicates and sort
    image_files = sorted(set(image_files))
    
    if not image_files:
        logger.warning(f"No images found in {input_dir}")
        logger.info(f"Supported extensions: {IMAGE_EXTENSIONS}")
        return []
    
    logger.info(f"Found {len(image_files)} images in {input_dir}")
    print()
    
    for i, img_path in enumerate(image_files):
        # Extract label from filename
        label = extract_label_from_filename(img_path.name)
        
        if label is None:
            label = prompt_for_label(img_path, auto_unknown)
            if label == "UNKNOWN":
                unknown_count += 1
        
        # Create relative path from data directory
        try:
            rel_path = img_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = img_path
        
        sample = {
            "id": f"casia_{i:05d}",
            "image_path": str(rel_path),
            "label_gt": label,
            "ocr_pred": "",  # Will be filled by Agent A during inference
            "context": "",  # Empty for single-character datasets
            "context_left": "",
            "context_right": "",
            "ocr_entropy": 0.0,  # Will be computed during inference
            "ppl_score": 0.0,  # Will be computed during inference
        }
        samples.append(sample)
        
        # Progress indicator
        if (i + 1) % 50 == 0 or (i + 1) == len(image_files):
            print(f"  Processed {i + 1}/{len(image_files)} images...")
    
    if unknown_count > 0:
        logger.warning(f"{unknown_count} images marked as UNKNOWN")
    
    return samples


# =============================================================================
# JSON Generation
# =============================================================================

def save_test_set(
    samples: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Save test set to JSON file.
    
    Args:
        samples: List of sample dictionaries.
        output_path: Output JSON path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create output structure
    output_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_samples": len(samples),
            "source": "CASIA-HWDB or similar",
            "description": "Standard handwritten character test set",
        },
        "samples": samples,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Test set saved to: {output_path}")


def print_summary(samples: List[Dict[str, Any]]) -> None:
    """Print summary of generated test set."""
    print()
    print("=" * 70)
    print("  📊 Test Set Summary")
    print("=" * 70)
    
    total = len(samples)
    unknown = sum(1 for s in samples if s["label_gt"] == "UNKNOWN")
    valid = total - unknown
    
    # Count unique labels
    labels = [s["label_gt"] for s in samples if s["label_gt"] != "UNKNOWN"]
    unique_labels = set(labels)
    
    print(f"  Total samples:   {total}")
    print(f"  Valid labels:    {valid}")
    print(f"  Unknown labels:  {unknown}")
    print(f"  Unique chars:    {len(unique_labels)}")
    print()
    
    if unique_labels:
        # Show label distribution (top 10)
        from collections import Counter
        label_counts = Counter(labels)
        top_labels = label_counts.most_common(10)
        
        print("  Top 10 characters:")
        for label, count in top_labels:
            print(f"    '{label}': {count}")
    
    print()
    print("=" * 70)
    
    if unknown > 0:
        print()
        print("  ⚠️  Note: Some images have UNKNOWN labels.")
        print("     You may want to manually edit the JSON file to fix them.")
        print()


def print_sample_preview(samples: List[Dict[str, Any]], n: int = 5) -> None:
    """Print preview of first n samples."""
    print()
    print("  📋 Sample Preview (first 5):")
    print("-" * 70)
    
    for i, sample in enumerate(samples[:n]):
        print(f"  [{i}] {sample['id']}")
        print(f"      Image: {sample['image_path']}")
        print(f"      Label: '{sample['label_gt']}'")
        print()


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build standard test set from CASIA-HWDB or similar dataset",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory with images (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--auto_unknown",
        action="store_true",
        help="Automatically mark unparseable labels as UNKNOWN (no prompts)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print()
    print("=" * 70)
    print("  L2W1 Standard Test Set Builder")
    print("=" * 70)
    print()
    print(f"  Input directory:  {args.input_dir}")
    print(f"  Output JSON:      {args.output}")
    print(f"  Auto UNKNOWN:     {args.auto_unknown}")
    print()
    
    # Check if input directory exists
    if not args.input_dir.exists():
        print()
        print("  ❌ Input directory does not exist!")
        print()
        print("  To use this script:")
        print(f"  1. Create the directory: mkdir -p {args.input_dir}")
        print(f"  2. Add images with format: 字符_序号.jpg (e.g., 测_001.jpg)")
        print(f"  3. Run this script again")
        print()
        print("  Expected filename formats:")
        print("    - 测_123.jpg  -> label: 测")
        print("    - 测.png      -> label: 测")
        print("    - char_测.jpg -> label: 测")
        print()
        
        # Create directory for user convenience
        try:
            args.input_dir.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created empty directory: {args.input_dir}")
            print("    Please add images and run again.")
        except Exception as e:
            print(f"  Failed to create directory: {e}")
        
        return
    
    # Scan directory and extract labels
    print("-" * 70)
    print("  Scanning images and extracting labels...")
    print("-" * 70)
    
    samples = scan_image_directory(args.input_dir, args.auto_unknown)
    
    if not samples:
        print()
        print("  ❌ No samples generated!")
        print("     Please add images to the input directory.")
        return
    
    # Save test set
    save_test_set(samples, args.output)
    
    # Print summary
    print_summary(samples)
    print_sample_preview(samples)
    
    print("  ✓ Test set generation complete!")
    print()
    print("  Next steps:")
    print(f"  1. Review the JSON file: {args.output}")
    print(f"  2. Fix any UNKNOWN labels manually if needed")
    print(f"  3. Run batch inference:")
    print(f"     python scripts/04_batch_inference.py --test_set {args.output}")
    print()


if __name__ == "__main__":
    main()

