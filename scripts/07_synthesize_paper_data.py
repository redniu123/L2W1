#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Synthesize hard benchmark data for paper evaluation.

This script builds a small but challenging synthetic dataset by
concatenating single-character images from CASIA-style data and
injecting visual noise to intentionally cause OCR (Agent A) failures.

The output JSON format follows `casia_test_set.json`:

{
  "metadata": {...},
  "samples": [
    {
      "id": "synthetic_00000",
      "image_path": "data/synthetic_benchmark/xxx.png",
      "label_gt": "目标文本",
      "ocr_pred": "",
      "context": "",
      "context_left": "",
      "context_right": "",
      "ocr_entropy": 0.0,
      "ppl_score": 0.0
    },
    ...
  ]
}
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
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SINGLE_CHAR_DIR = PROJECT_ROOT / "data" / "casia_sample"
DEFAULT_CASIA_JSON = PROJECT_ROOT / "data" / "casia_test_set.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic_benchmark"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "data" / "synthetic_test.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def _scan_single_char_images(input_dir: Path) -> List[Tuple[Path, str]]:
    """Scan directory for single-character images and infer labels from filenames.

    This is a lightweight fallback when `casia_test_set.json` does not exist.
    It mirrors the filename assumption used by `05_build_standard_testset.py`.

    Args:
        input_dir: Directory that contains CASIA-style single-character images.

    Returns:
        List of (image_path, label) tuples. Images with unparseable labels are skipped.
    """
    def _extract_label_from_filename(filename: str) -> Optional[str]:
        """Extract single-character label from filename.

        Priority:
        1) Split by "_" and, if the first segment has length 1 (any character),
           return it directly as the label.
        2) Otherwise, take the first character of the stem.

        This is intentionally more permissive than the main builder script,
        because the synthetic benchmark can include non-Chinese symbols.

        Args:
            filename: Image filename (without path).

        Returns:
            Extracted single-character label, or None if not found.
        """
        stem: str = Path(filename).stem
        parts: List[str] = stem.split("_")
        if parts and len(parts[0]) == 1:
            return parts[0]
        if stem:
            return stem[0]
        return None

    if not input_dir.exists():
        logger.error("Single-character input directory does not exist: %s", input_dir)
        return []

    image_files: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))

    # One-level subdirectories
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            for ext in IMAGE_EXTENSIONS:
                image_files.extend(subdir.glob(f"*{ext}"))
                image_files.extend(subdir.glob(f"*{ext.upper()}"))

    image_files = sorted(set(image_files))

    pairs: List[Tuple[Path, str]] = []
    for img_path in image_files:
        label = _extract_label_from_filename(img_path.name)
        if label is None:
            continue
        pairs.append((img_path, label))

    logger.info("Scanned %d single-character images from %s", len(pairs), input_dir)
    return pairs


def load_single_char_pool(
    casia_json: Path,
    fallback_dir: Path,
) -> List[Tuple[Path, str]]:
    """Load single-character image pool from JSON or fallback directory scan.

    Priority:
    1) If `casia_test_set.json` exists, load samples from there.
    2) Otherwise, rescan `data/casia_sample/` and infer labels from filenames.

    Args:
        casia_json: Path to `casia_test_set.json`.
        fallback_dir: Fallback directory with single-character images.

    Returns:
        List of (image_path, label) tuples.
    """
    if casia_json.exists():
        logger.info("Loading single-character pool from JSON: %s", casia_json)
        with casia_json.open("r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        samples: List[Dict[str, Any]] = data.get("samples", [])
        pairs: List[Tuple[Path, str]] = []
        for sample in samples:
            image_rel = sample.get("image_path")
            label = sample.get("label_gt")
            if not image_rel or not isinstance(image_rel, str):
                continue
            if not label or not isinstance(label, str):
                continue
            image_path = PROJECT_ROOT / image_rel
            if not image_path.exists():
                continue
            pairs.append((image_path, label))
        logger.info(
            "Loaded %d single-character samples from %s", len(pairs), casia_json
        )
        if pairs:
            return pairs

    logger.warning(
        "Falling back to directory scan because JSON is missing or empty: %s",
        casia_json,
    )
    return load_single_char_pool_from_dir(fallback_dir)


def load_single_char_pool_from_dir(fallback_dir: Path) -> List[Tuple[Path, str]]:
    """Wrapper around `_scan_single_char_images` for clarity.

    Args:
        fallback_dir: Directory with single-character images.

    Returns:
        List of (image_path, label) tuples.
    """
    return _scan_single_char_images(fallback_dir)


# =============================================================================
# Image Composition & Augmentation
# =============================================================================

def resize_to_height(img: Image.Image, target_height: int) -> Image.Image:
    """Resize image to a fixed height while preserving aspect ratio.

    Tensor-wise, this corresponds to resizing from `[H, W]` to `[target_H, new_W]`.

    Args:
        img: PIL image.
        target_height: Target height in pixels.

    Returns:
        Resized PIL image.
    """
    w, h = img.size
    if h == 0:
        raise ValueError("Image height is zero, cannot resize.")
    scale: float = float(target_height) / float(h)
    new_w: int = max(1, int(round(w * scale)))
    return img.resize((new_w, target_height), Image.BILINEAR)


def concatenate_horizontally(images: Sequence[Image.Image]) -> Image.Image:
    """Concatenate images horizontally into a long strip.

    All images are assumed to have the same height: `[H, W_i] -> [H, sum_i W_i]`.

    Args:
        images: Sequence of PIL images with the same height.

    Returns:
        Concatenated PIL image.
    """
    if not images:
        raise ValueError("No images provided for concatenation.")

    heights = {img.size[1] for img in images}
    if len(heights) != 1:
        raise ValueError(
            f"All images must share the same height, got heights={heights}"
        )

    total_width: int = sum(img.size[0] for img in images)
    height: int = images[0].size[1]

    canvas = Image.new("L", (total_width, height), color=255)
    x_offset = 0
    for img in images:
        canvas.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    return canvas


def apply_gaussian_blur(img: Image.Image) -> Image.Image:
    """Apply Gaussian blur with random small kernel.

    Kernel size is approximated by radius in PIL:
    - 3x3 -> radius ~1
    - 5x5 -> radius ~2

    Args:
        img: PIL image.

    Returns:
        Blurred image.
    """
    radius: int = random.choice([1, 2])
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_salt_pepper_noise(img: Image.Image, amount: float = 0.02) -> Image.Image:
    """Apply salt-and-pepper noise to a grayscale image.

    This operates on the `[H, W]` array and randomly sets a proportion of
    pixels to 0 (pepper) or 255 (salt).

    Args:
        img: Grayscale PIL image.
        amount: Proportion of pixels to corrupt, in [0, 1].

    Returns:
        Noised image.
    """
    if amount <= 0.0:
        return img

    arr: np.ndarray = np.array(img, dtype=np.uint8)  # [H, W]
    h, w = arr.shape
    num_pixels: int = int(h * w * amount)
    if num_pixels <= 0:
        return img

    # Randomly choose coordinates
    ys = np.random.randint(0, h, size=num_pixels)
    xs = np.random.randint(0, w, size=num_pixels)

    # Half salt, half pepper
    half: int = num_pixels // 2
    arr[ys[:half], xs[:half]] = 0
    arr[ys[half:], xs[half:]] = 255

    return Image.fromarray(arr, mode="L")


def apply_low_contrast(img: Image.Image) -> Image.Image:
    """Lower image contrast.

    Args:
        img: PIL image.

    Returns:
        Contrast-reduced image.
    """
    enhancer = ImageEnhance.Contrast(img)
    factor: float = random.uniform(0.3, 0.7)
    return enhancer.enhance(factor)


def apply_random_noise(img: Image.Image, noise_prob: float) -> Image.Image:
    """Apply one of several random noise types with probability `noise_prob`.

    Args:
        img: PIL image (grayscale).
        noise_prob: Probability in [0, 1] to apply any noise.

    Returns:
        Possibly noised image.
    """
    if random.random() >= noise_prob:
        return img

    choice: str = random.choice(["blur", "s&p", "low_contrast"])
    if choice == "blur":
        return apply_gaussian_blur(img)
    if choice == "s&p":
        return apply_salt_pepper_noise(img)
    if choice == "low_contrast":
        return apply_low_contrast(img)

    return img


# =============================================================================
# Synthetic Sample Generation
# =============================================================================

def synthesize_samples(
    pool: List[Tuple[Path, str]],
    output_dir: Path,
    num_samples: int = 100,
    min_len: int = 3,
    max_len: int = 6,
    target_height: int = 48,
    noise_prob: float = 0.6,
    seed: Optional[int] = 42,
) -> List[Dict[str, Any]]:
    """Synthesize a list of multi-character samples for evaluation.

    Args:
        pool: List of (image_path, label) tuples for single characters.
        output_dir: Directory to save synthesized images.
        num_samples: Number of synthetic sequences to generate.
        min_len: Minimum sequence length (inclusive).
        max_len: Maximum sequence length (inclusive).
        target_height: Normalized height of the output images.
        noise_prob: Probability of applying any noise to a sample.
        seed: Optional random seed for reproducibility.

    Returns:
        List of sample dictionaries with fields compatible with
        `casia_test_set.json`.
    """
    if not pool:
        raise ValueError("Single-character pool is empty, cannot synthesize samples.")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    samples: List[Dict[str, Any]] = []

    for idx in range(num_samples):
        seq_len: int = random.randint(min_len, max_len)
        chosen = random.choices(pool, k=seq_len)

        chars: List[str] = [label for _, label in chosen]
        text: str = "".join(chars)

        pil_images: List[Image.Image] = []
        for img_path, _ in chosen:
            img = Image.open(img_path).convert("L")
            img = resize_to_height(img, target_height)
            pil_images.append(img)

        concat_img = concatenate_horizontally(pil_images)
        concat_img = apply_random_noise(concat_img, noise_prob=noise_prob)

        image_filename = f"synthetic_{idx:05d}.png"
        save_path = output_dir / image_filename
        concat_img.save(save_path)

        try:
            rel_path = save_path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = save_path

        sample: Dict[str, Any] = {
            "id": f"synthetic_{idx:05d}",
            "image_path": str(rel_path),
            "label_gt": text,
            "ocr_pred": "",
            # For paper evaluation, we can keep context empty to focus on OCR.
            "context": "",
            "context_left": "",
            "context_right": "",
            "ocr_entropy": 0.0,
            "ppl_score": 0.0,
        }
        samples.append(sample)

        if (idx + 1) % 20 == 0 or (idx + 1) == num_samples:
            logger.info(
                "Synthesized %d/%d samples (last text='%s')",
                idx + 1,
                num_samples,
                text,
            )

    return samples


def save_synthetic_test_set(
    samples: List[Dict[str, Any]],
    output_json: Path,
) -> None:
    """Save synthetic benchmark samples to JSON file.

    Args:
        samples: List of sample dictionaries.
        output_json: Output JSON path.
    """
    output_json.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_samples": len(samples),
            "source": "Synthetic CASIA-based benchmark",
            "description": (
                "Hard synthetic benchmark by concatenating CASIA characters "
                "and injecting visual noise to stress-test OCR and L2W1."
            ),
        },
        "samples": samples,
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("Synthetic test set saved to: %s", output_json)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Synthesize hard benchmark data for L2W1 paper evaluation.",
    )
    parser.add_argument(
        "--casia_json",
        type=Path,
        default=DEFAULT_CASIA_JSON,
        help=f"Path to casia_test_set.json (default: {DEFAULT_CASIA_JSON})",
    )
    parser.add_argument(
        "--single_char_dir",
        type=Path,
        default=DEFAULT_SINGLE_CHAR_DIR,
        help=(
            "Fallback directory with single-character images if JSON is missing "
            f"(default: {DEFAULT_SINGLE_CHAR_DIR})"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save synthesized images (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of synthetic sequences to generate (default: 100)",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=3,
        help="Minimum number of characters per sequence (default: 3)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=6,
        help="Maximum number of characters per sequence (default: 6)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=48,
        help="Target image height in pixels (default: 48)",
    )
    parser.add_argument(
        "--noise_prob",
        type=float,
        default=0.6,
        help="Probability to apply noise to each sample (default: 0.6)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry for synthetic benchmark generation."""
    args = parse_args()

    print()
    print("=" * 70)
    print("  L2W1 Synthetic Benchmark Builder")
    print("=" * 70)
    print()
    print(f"  CASIA JSON:         {args.casia_json}")
    print(f"  Single-char dir:    {args.single_char_dir}")
    print(f"  Output image dir:   {args.output_dir}")
    print(f"  Output JSON:        {args.output_json}")
    print(f"  Num samples:        {args.num_samples}")
    print(f"  Sequence length:    {args.min_len} ~ {args.max_len}")
    print(f"  Target height:      {args.height}")
    print(f"  Noise probability:  {args.noise_prob}")
    print(f"  Seed:               {args.seed}")
    print()

    pool = load_single_char_pool(
        casia_json=args.casia_json,
        fallback_dir=args.single_char_dir,
    )
    if not pool:
        print("  ❌ No single-character images available to build synthetic data.")
        print(f"     Please check {args.casia_json} or {args.single_char_dir}.")
        return

    print("-" * 70)
    print("  Synthesizing multi-character benchmark samples...")
    print("-" * 70)

    samples = synthesize_samples(
        pool=pool,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        min_len=args.min_len,
        max_len=args.max_len,
        target_height=args.height,
        noise_prob=args.noise_prob,
        seed=args.seed,
    )

    save_synthetic_test_set(samples, args.output_json)

    print()
    print("  ✓ Synthetic benchmark generation complete!")
    print()
    print("  Next steps:")
    print(f"  1. Review the JSON file: {args.output_json}")
    print("  2. Run batch inference on synthetic data, then compare:")
    print("       - Pure OCR (Agent A)")
    print("       - L2W1 (Agent A + Router + Agent B)")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Synthesize Noisy Text-Line Data for L2W1 Paper Benchmarks.

Goal:
    Generate synthetic long text-line images with controlled noise so that
    Agent A (OCR) accuracy is ~80%, highlighting Agent B's correction power.

Data Source:
    - Single-character images from data/casia_sample/

Output:
    - Images in data/synthetic_benchmark/
    - Index JSON: data/synthetic_test.json
      Each sample:
        {
          "id": "synt_00001",
          "image_path": "data/synthetic_benchmark/synt_00001.png",
          "label_gt": "测试样本",
          "context": ""
        }
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

INPUT_DIR = PROJECT_ROOT / "data" / "casia_sample"
OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic_benchmark"
OUTPUT_JSON = PROJECT_ROOT / "data" / "synthetic_test.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

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
    """Extract Chinese character label from filename.

    Expected formats:
        - 测_123.jpg -> 测
        - 测.png -> 测
    """
    stem = Path(filename).stem
    # Take first non-ASCII char as label (simple heuristic)
    for ch in stem:
        if ord(ch) > 127:
            return ch
    return None


# =============================================================================
# Image Loading & Line Synthesis
# =============================================================================

def load_char_images(input_dir: Path) -> List[Tuple[np.ndarray, str]]:
    """Load all character images and labels from directory."""
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return []

    files: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(input_dir.glob(f"*{ext}"))
        files.extend(input_dir.glob(f"*{ext.upper()}"))

    files = sorted(set(files))
    if not files:
        logger.error(f"No images found in {input_dir}")
        return []

    chars: List[Tuple[np.ndarray, str]] = []
    for path in files:
        label = extract_label_from_filename(path.name)
        if not label:
            logger.warning(f"Skip (no label): {path.name}")
            continue
        img = cv2.imread(str(path))
        if img is None:
            logger.warning(f"Skip (load fail): {path}")
            continue
        chars.append((img, label))

    logger.info(f"Loaded {len(chars)} character images from {input_dir}")
    return chars


def normalize_height(
    images: List[Tuple[np.ndarray, str]],
    target_height: int = 64,
) -> List[Tuple[np.ndarray, str]]:
    """Normalize all character images to same height while preserving aspect."""
    norm: List[Tuple[np.ndarray, str]] = []
    for img, label in images:
        h, w = img.shape[:2]
        if h <= 0:
            continue
        scale = target_height / h
        new_w = max(1, int(w * scale))
        resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
        norm.append((resized, label))
    return norm


def synthesize_line(
    pool: List[Tuple[np.ndarray, str]],
    min_chars: int = 3,
    max_chars: int = 6,
    spacing_max: int = 10,
    padding: int = 16,
) -> Tuple[np.ndarray, str]:
    """Synthesize a single text-line from random characters."""
    if not pool:
        raise ValueError("Character pool is empty.")

    num_chars = random.randint(min_chars, max_chars)
    chosen = random.choices(pool, k=num_chars)

    # Normalize height
    chosen = normalize_height(chosen, target_height=64)
    if not chosen:
        raise ValueError("Failed to normalize characters.")

    # Compute total width
    total_w = padding * 2
    for img, _ in chosen:
        total_w += img.shape[1]
        total_w += random.randint(0, spacing_max)
    total_w -= random.randint(0, spacing_max)  # remove last spacing

    h = chosen[0][0].shape[0] + padding * 2
    canvas = np.full((h, total_w, 3), 255, dtype=np.uint8)

    x = padding
    y = padding
    text = ""
    for i, (img, label) in enumerate(chosen):
        hh, ww = img.shape[:2]
        canvas[y : y + hh, x : x + ww] = img
        text += label
        x += ww
        if i < len(chosen) - 1:
            x += random.randint(0, spacing_max)

    return canvas, text


# =============================================================================
# Noise Injection
# =============================================================================

def add_gaussian_blur(image: np.ndarray) -> np.ndarray:
    ksize = random.choice([3, 5])
    sigma = random.uniform(0.5, 1.5)
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma)


def add_salt_pepper(image: np.ndarray, amount: float = 0.01) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    num_pixels = int(amount * h * w)
    for _ in range(num_pixels):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        if random.random() < 0.5:
            out[y, x] = 0
        else:
            out[y, x] = 255
    return out


def add_random_lines(image: np.ndarray, num_lines: int = 5) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    for _ in range(num_lines):
        x1 = random.randint(0, w - 1)
        y1 = random.randint(0, h - 1)
        x2 = random.randint(0, w - 1)
        y2 = random.randint(0, h - 1)
        color = (0, 0, 0) if random.random() < 0.5 else (128, 128, 128)
        thickness = random.randint(1, 2)
        cv2.line(out, (x1, y1), (x2, y2), color, thickness)
    return out


def adjust_brightness_contrast(
    image: np.ndarray,
    alpha_range: Tuple[float, float] = (0.6, 0.9),
    beta_range: Tuple[int, int] = (-40, 0),
) -> np.ndarray:
    alpha = random.uniform(*alpha_range)  # contrast
    beta = random.randint(*beta_range)  # brightness
    out = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return out


def apply_random_noise(image: np.ndarray) -> np.ndarray:
    """Apply one random noise type to the image."""
    noise_funcs = [
        add_gaussian_blur,
        add_salt_pepper,
        add_random_lines,
        adjust_brightness_contrast,
    ]
    func = random.choice(noise_funcs)
    return func(image)


# =============================================================================
# Main Generation Logic
# =============================================================================

def generate_benchmark(
    num_samples: int,
    noise_ratio: float = 0.5,
    seed: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Generate synthetic benchmark samples."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    chars = load_char_images(INPUT_DIR)
    if not chars:
        return []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples: List[Dict[str, str]] = []
    num_noisy = 0

    for i in range(num_samples):
        try:
            img, text = synthesize_line(chars)
        except Exception as e:
            logger.warning(f"Skip sample {i} (synthesis error): {e}")
            continue

        # Decide whether to add noise
        add_noise = random.random() < noise_ratio
        if add_noise:
            img = apply_random_noise(img)
            num_noisy += 1

        sample_id = f"synt_{i:05d}"
        filename = f"{sample_id}.png"
        img_path = OUTPUT_DIR / filename
        cv2.imwrite(str(img_path), img)

        rel_path = img_path.relative_to(PROJECT_ROOT)
        samples.append(
            {
                "id": sample_id,
                "image_path": str(rel_path).replace("\\", "/"),
                "label_gt": text,
                "context": "",
            }
        )

    logger.info(f"Generated {len(samples)} samples ({num_noisy} with noise)")
    return samples


def save_index(samples: List[Dict[str, str]], output_json: Path) -> None:
    """Save index JSON file with metadata."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_samples": len(samples),
            "source": "CASIA synthetic benchmark",
            "noise_note": "50% samples contain blur / noise / lines / low contrast",
        },
        "samples": samples,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Index JSON saved to: {output_json}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize noisy text-line data for L2W1 paper benchmarks",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of synthetic lines to generate (default: 200)",
    )
    parser.add_argument(
        "--noise_ratio",
        type=float,
        default=0.5,
        help="Fraction of samples with injected noise (default: 0.5)",
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

    print("=" * 70)
    print(" L2W1 Synthetic Benchmark Generator")
    print("=" * 70)
    print(f"  Input dir   : {INPUT_DIR}")
    print(f"  Output dir  : {OUTPUT_DIR}")
    print(f"  Output JSON : {OUTPUT_JSON}")
    print(f"  Num samples : {args.num_samples}")
    print(f"  Noise ratio : {args.noise_ratio:.2f}")
    print()

    samples = generate_benchmark(
        num_samples=args.num_samples,
        noise_ratio=args.noise_ratio,
        seed=args.seed,
    )

    if not samples:
        print("❌ No samples generated. Please check input directory.")
        return

    save_index(samples, OUTPUT_JSON)

    print()
    print("-" * 70)
    print(f"✅ Synthetic benchmark created with {len(samples)} samples.")
    print("-" * 70)
    print("Next step:")
    print(f"  Run batch inference on the generated dataset, e.g.:")
    print(f"  python scripts/04_batch_inference.py --test_set {OUTPUT_JSON}")
    print()


if __name__ == "__main__":
    main()


