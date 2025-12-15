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


