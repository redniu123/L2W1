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

# Determine if we should use CPU mode
# Default to GPU mode, disable only if USE_CPU=true or USE_GPU=false
use_cpu_env = (
    os.getenv("USE_CPU", "false").lower() == "true"
    or os.getenv("USE_GPU", "true").lower() == "false"
)

# Set environment variables to prevent cudnn loading when using CPU
# This must be done BEFORE importing any PaddleOCR/PaddlePaddle modules
if use_cpu_env:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    # Prevent PaddlePaddle from trying to load cudnn on CPU-only systems
    os.environ.setdefault("FLAGS_cudnn_deterministic", "1")
    # Disable MKLDNN to avoid instruction set compatibility issues
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    # Use basic CPU operations
    os.environ.setdefault("FLAGS_cpu_deterministic", "1")

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
import os
import platform
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


def load_chinese_font(font_size: int = 40) -> Optional[ImageFont.FreeTypeFont]:
    """Load a Chinese font for text rendering.

    Tries multiple common font paths on different platforms:
    - Windows: simhei.ttf, msyh.ttf
    - Linux/Mac: NotoSansCJK, WenQuanYi, etc.

    Args:
        font_size: Font size in pixels.

    Returns:
        ImageFont object, or None if no font found.
    """
    system = platform.system()
    font_paths = []

    if system == "Windows":
        # Windows common Chinese fonts
        windows_font_dir = os.path.join(
            os.environ.get("WINDIR", "C:\\Windows"), "Fonts"
        )
        font_paths = [
            os.path.join(windows_font_dir, "simhei.ttf"),  # 黑体
            os.path.join(windows_font_dir, "msyh.ttf"),  # 微软雅黑
            os.path.join(windows_font_dir, "simsun.ttc"),  # 宋体
            os.path.join(windows_font_dir, "msyhbd.ttf"),  # 微软雅黑 Bold
        ]
    elif system == "Linux":
        # Linux common Chinese fonts
        font_paths = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "~/.fonts/wqy-microhei.ttc",
        ]
    elif system == "Darwin":  # macOS
        # macOS common Chinese fonts
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/Library/Fonts/Microsoft/msyh.ttf",
            "~/Library/Fonts/NotoSansCJK-Regular.ttc",
        ]

    # Try to load each font
    for font_path in font_paths:
        # Expand user path (~)
        font_path = os.path.expanduser(font_path)

        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                logger.info(f"Loaded Chinese font: {font_path}")
                return font
            except Exception as e:
                logger.debug(f"Failed to load font {font_path}: {e}")
                continue

    # Fallback: try to use default font (may not support Chinese)
    try:
        font = ImageFont.load_default()
        logger.warning(
            "No Chinese font found. Using default font (Chinese may display as ?). "
            "Please install a Chinese font for proper display."
        )
        return font
    except Exception as e:
        logger.error(f"Failed to load default font: {e}")
        return None


def generate_demo_image(
    output_path: Path,
    text: str = "测试样本123",
    size: Tuple[int, int] = (800, 100),
) -> Path:
    """Generate a dummy test image with Chinese text for demo mode.

    Creates a white background image with black text for testing
    the pipeline without real data.

    Fixed: Uses PIL ImageDraw with Chinese font support to prevent
    character display issues (previously showed as ?).

    Args:
        output_path: Path to save the generated image.
        text: Text to render on the image.
        size: Image size as (width, height).

    Returns:
        Path to the generated image.
    """
    width, height = size

    # Create white background using PIL (supports TrueType fonts)
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Load Chinese font (40px for good OCR recognition)
    font_size = 40
    font = load_chinese_font(font_size=font_size)

    if font is None:
        logger.warning("No font available, text may not render correctly")

    # Calculate text position (centered)
    if font:
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        # Fallback estimation
        text_width = len(text) * font_size * 0.6  # Approximate
        text_height = font_size

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    # Draw text
    color = (0, 0, 0)  # Black
    draw.text((x, y), text, font=font, fill=color)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save image
    image.save(str(output_path), "PNG")
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

    # Try to initialize with requested GPU setting, with automatic fallback
    processor = None
    actual_use_gpu = use_gpu

    if use_gpu:
        logger.info("Attempting to initialize with GPU...")
        try:
            processor = DataProcessor(
                output_dir=crops_dir,
                target_size=336,
                context_alpha=0.3,
                use_gpu=True,
            )
            logger.info("✓ Successfully initialized with GPU")
        except Exception as e:
            logger.warning(f"GPU initialization failed: {e}")
            logger.warning("Falling back to CPU mode...")
            actual_use_gpu = False
            # Set CPU environment variables before retry
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            os.environ.setdefault("FLAGS_cudnn_deterministic", "1")
            os.environ.setdefault("FLAGS_use_mkldnn", "0")
            os.environ.setdefault("FLAGS_cpu_deterministic", "1")

    if processor is None:
        logger.info("Initializing with CPU mode...")
        try:
            processor = DataProcessor(
                output_dir=crops_dir,
                target_size=336,
                context_alpha=0.3,
                use_gpu=False,
            )
            logger.info("✓ Successfully initialized with CPU")
        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"Failed to initialize DataProcessor (CPU mode): {e}")

            if "illegal instruction" in error_str or "sigill" in error_str:
                logger.error("=" * 60)
                logger.error("CPU INSTRUCTION SET INCOMPATIBILITY DETECTED")
                logger.error("=" * 60)
                logger.error(
                    "Your CPU does not support the instruction sets required by"
                )
                logger.error("the installed PaddlePaddle version.")
                logger.error("")
                logger.error("Solutions:")
                logger.error("1. Install CPU-compatible PaddlePaddle:")
                logger.error("   pip uninstall paddlepaddle-gpu")
                logger.error("   pip install paddlepaddle")
                logger.error("")
                logger.error("2. Use GPU mode (if available):")
                logger.error("   python scripts/01_build_dataset.py")
                logger.error("")
                logger.error("3. Use a system with a newer CPU that supports AVX/AVX2")
                logger.error("=" * 60)
            elif "cudnn" in error_str:
                logger.error("CUDA/cudnn error detected. Make sure:")
                logger.error("1. CUDA and cudnn are properly installed")
                logger.error("2. GPU drivers are up to date")
                logger.error(
                    "3. Or use CPU mode: USE_CPU=true python scripts/01_build_dataset.py"
                )
            else:
                logger.error(
                    "Try installing a compatible PaddlePaddle version or use GPU mode."
                )
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

            if not results:
                logger.warning(
                    f"         ⚠ No valid crops generated. This may be due to:"
                )
                logger.warning(f"           - OCR failed to recognize text")
                logger.warning(
                    f"           - All characters were filtered (IGNORE type)"
                )
                logger.warning(f"           - DTW alignment produced no valid pairs")
                failed_items.append(
                    {
                        "path": str(image_path),
                        "gt_text": gt_text,
                        "error": "No valid crops generated (empty results)",
                    }
                )
            else:
                all_results.extend(results)
                logger.info(f"         ✓ Generated {len(results)} character crops")

        except Exception as e:
            import traceback

            logger.error(f"         ✗ Failed: {e}")
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
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
        logger.info("Generating demo test image...")
        demo_image = generate_demo_image(DEMO_IMAGE_PATH, DEMO_GT_TEXT)

        # Verify image was created
        if not demo_image.exists():
            logger.error(f"Failed to generate demo image at {demo_image}")
            logger.error("Cannot proceed without demo image.")
            return
        else:
            logger.info(f"✓ Demo image created: {demo_image}")
            logger.info(f"  Image size: {demo_image.stat().st_size} bytes")

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

        # Diagnostic information
        logger.info("")
        logger.info("=" * 60)
        logger.info("Pipeline Execution Summary")
        logger.info("=" * 60)
        logger.info(f"Total images processed: {len(image_gt_pairs)}")
        logger.info(f"Successful crops: {len(results)}")
        logger.info(f"Failed items: {len(failed)}")

        if len(results) == 0:
            logger.warning("")
            logger.warning("⚠️  NO VALID CROPS GENERATED!")
            logger.warning("")
            logger.warning("Possible causes:")
            logger.warning("  1. OCR failed to recognize any text")
            logger.warning("  2. All characters were filtered (IGNORE type)")
            logger.warning("  3. DTW alignment produced no valid pairs")
            logger.warning("  4. All boxes were invalid after expansion")
            logger.warning("")
            logger.warning("Check the logs above for detailed error messages.")
            logger.warning("")

            if failed:
                logger.warning("Failed items details:")
                for item in failed:
                    logger.warning(
                        f"  - {item.get('path', 'N/A')}: {item.get('error', 'Unknown error')}"
                    )

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please ensure all dependencies are installed:")
        logger.error("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        import traceback

        logger.error(f"Pipeline failed: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
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
