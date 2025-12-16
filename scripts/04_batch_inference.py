#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch Inference for Paper Experiments with CER Evaluation.

Generates core metrics for the L2W1 paper experiments section using
Character Error Rate (CER) instead of exact match.

Metrics:
1. Total Samples
2. Baseline Accuracy (Agent A only) - CER-based
3. L2W1 Accuracy (Ours) - CER-based
4. Router Activation Rate
5. Correction Success Rate

Output:
- CSV: output/batch_eval_details.csv
- Console: Markdown table for paper

Usage:
    python scripts/04_batch_inference.py
    python scripts/04_batch_inference.py --test_set data/test_set.json
"""

import os

# --- 【添加这两行】强制指定模型路径 & 离线模式 ---
# 1. 指定你解压出来的 my_models 文件夹的绝对路径
os.environ["HF_HOME"] = "/home/coder/project/L2W1/my_models"

# 2. 告诉 Hugging Face 不要联网，直接用本地的
os.environ["HF_HUB_OFFLINE"] = "1"
# ------------------------------------------------
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
import csv
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[WARN] tqdm not installed. Progress bar disabled.")

try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    logger.warning("python-Levenshtein not installed. Installing...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-Levenshtein"])
    from Levenshtein import distance as levenshtein_distance

# =============================================================================
# Configuration
# =============================================================================

# Default paths
DEFAULT_TEST_SET = PROJECT_ROOT / "data" / "test_set.json"
FALLBACK_TEST_SET = PROJECT_ROOT / "output" / "agent_b_train.json"
OUTPUT_CSV = PROJECT_ROOT / "output" / "batch_eval_details.csv"

# Thresholds
PPL_THRESHOLD = 100.0
ENTROPY_THRESHOLD = 0.3
CER_THRESHOLD = 0.1  # CER < 0.1 means "basically correct" (90% accuracy)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Text Normalization
# =============================================================================


def normalize_text(text: str) -> str:
    """Normalize text for CER calculation.

    Steps:
    1. Remove all whitespace
    2. Convert English punctuation to Chinese punctuation

    Args:
        text: Input text string.

    Returns:
        Normalized text string.
    """
    if not text:
        return ""

    # Remove all whitespace
    normalized = re.sub(r"\s+", "", text)

    # Convert English punctuation to Chinese punctuation
    punctuation_map = {
        ",": "，",
        ".": "。",
        ":": "：",
        ";": "；",
        "!": "！",
        "?": "？",
    }

    for en_punct, zh_punct in punctuation_map.items():
        normalized = normalized.replace(en_punct, zh_punct)

    return normalized


# =============================================================================
# CER Calculation
# =============================================================================


def compute_cer(pred: str, gt: str) -> float:
    """Compute Character Error Rate (CER).

    CER = Levenshtein.distance(pred, gt) / len(gt)

    Args:
        pred: Predicted text.
        gt: Ground truth text.

    Returns:
        CER value (0.0 = perfect match, 1.0 = completely wrong).
    """
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(gt)

    if len(gt_norm) == 0:
        # If GT is empty, CER is 0 if pred is also empty, else 1.0
        return 0.0 if len(pred_norm) == 0 else 1.0

    distance = levenshtein_distance(pred_norm, gt_norm)
    cer = distance / len(gt_norm)
    return cer


def is_correct_by_cer(pred: str, gt: str, threshold: float = CER_THRESHOLD) -> bool:
    """Check if prediction is correct based on CER threshold.

    Args:
        pred: Predicted text.
        gt: Ground truth text.
        threshold: CER threshold (default: 0.1, meaning 90% accuracy).

    Returns:
        True if CER < threshold (basically correct).
    """
    cer = compute_cer(pred, gt)
    return cer < threshold


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SampleResult:
    """Result for a single sample."""

    id: str
    gt: str  # Ground truth
    pred_a: str  # Agent A prediction
    pred_final: str  # Final prediction (after Agent B if routed)
    is_routed: bool  # Whether routed to Agent B
    is_correct: bool  # Whether final prediction is correct (CER-based)
    cer_a: float = 0.0  # CER for Agent A
    cer_final: float = 0.0  # CER for final prediction
    error: str = ""  # Error message if any


@dataclass
class PaperMetrics:
    """Core metrics for paper."""

    total_samples: int = 0
    baseline_accuracy: float = 0.0  # Agent A only (CER-based)
    l2w1_accuracy: float = 0.0  # Our method (CER-based)
    router_activation_rate: float = 0.0  # Efficiency metric
    correction_success_rate: float = 0.0  # Agent B efficacy
    avg_cer_baseline: float = 0.0  # Average CER for baseline
    avg_cer_l2w1: float = 0.0  # Average CER for L2W1


# =============================================================================
# Data Loading
# =============================================================================


def load_test_set(test_set_path: Path) -> List[Dict[str, Any]]:
    """Load test set from JSON file.

    Args:
        test_set_path: Path to JSON file.

    Returns:
        List of sample dictionaries.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    if not test_set_path.exists():
        raise FileNotFoundError(
            f"Test set not found: {test_set_path}\n"
            f"Please create the test set file or use --test_set to specify a different path."
        )

    with open(test_set_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both formats: list or {"samples": [...]}
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        samples = data.get("samples", [])
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")

    logger.info(f"Loaded {len(samples)} samples from {test_set_path.name}")
    return samples


def resolve_image_path(image_path_str: str, base_dir: Path) -> Optional[Path]:
    """Resolve image path trying multiple locations."""
    candidates = [
        base_dir / image_path_str,
        base_dir / "crops" / Path(image_path_str).name,
        PROJECT_ROOT / "output" / image_path_str,
        PROJECT_ROOT / "output" / "crops" / Path(image_path_str).name,
        PROJECT_ROOT / image_path_str,
        Path(image_path_str),
    ]

    for path in candidates:
        if path.exists():
            return path
    return None


# =============================================================================
# L2W1 Pipeline
# =============================================================================


class L2W1Pipeline:
    """Unified L2W1 Pipeline for batch inference."""

    def __init__(
        self,
        ppl_threshold: float = PPL_THRESHOLD,
        entropy_threshold: float = ENTROPY_THRESHOLD,
        use_gpu: bool = True,
        skip_detection: bool = False,
    ):
        self.ppl_threshold = ppl_threshold
        self.entropy_threshold = entropy_threshold
        self.skip_detection = skip_detection
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

    def should_route(self, entropy: float, ppl: float) -> bool:
        """Determine if sample should be routed to Agent B."""
        return ppl > self.ppl_threshold or entropy > self.entropy_threshold

    def infer(
        self,
        sample: Dict[str, Any],
        image: Optional[Image.Image] = None,
    ) -> SampleResult:
        """Run inference on a single sample.

        Args:
            sample: Sample dict with keys: id, label_gt, ocr_pred, context, etc.
            image: Pre-loaded PIL Image (optional).

        Returns:
            SampleResult with predictions and routing info.
        """
        sample_id = sample.get("id", "unknown")
        gt = sample.get("label_gt", "")
        pred_a = sample.get("ocr_pred", "")
        entropy = sample.get("ocr_entropy", 0.0)
        ppl = sample.get("ppl_score", 0.0)

        # Handle context - default to empty string if missing
        context = sample.get("context", "")
        if context is None:
            context = ""
        context_left = sample.get("context_left", context)
        context_right = sample.get("context_right", "")
        if context_left is None:
            context_left = ""
        if context_right is None:
            context_right = ""

        # Determine if we should skip detection based on image size
        # For real-world long text images, force detection mode (skip_detection=False)
        skip_detection = self.skip_detection  # Start with pipeline setting
        if image is not None:
            w, h = image.size
            # Force detection for wide images (width > 64) or non-square images
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
            if w > 64 or aspect_ratio > 1.5:
                skip_detection = False  # Enable detection (don't skip)
                logger.debug(
                    f"Auto-enabling detection for image {w}x{h} "
                    f"(aspect_ratio={aspect_ratio:.2f})"
                )

        result = SampleResult(
            id=sample_id,
            gt=gt,
            pred_a=pred_a,
            pred_final=pred_a,  # Default to Agent A
            is_routed=False,
            is_correct=False,
            cer_a=0.0,
            cer_final=0.0,
        )

        try:
            # If no OCR prediction, run Agent A
            if (
                (pred_a is None or pred_a == "")
                and self.agent_a is not None
                and image is not None
            ):
                try:
                    # Use detection mode for real-world images
                    line_results = self.agent_a.inference(
                        image,
                        skip_detection=skip_detection,
                    )
                    if line_results:
                        # For detection mode, concatenate all detected text lines
                        if not skip_detection:
                            # Detection mode: multiple lines
                            pred_a = "".join([line.get("text", "") for line in line_results])
                            # Use max entropy across all lines
                            entropy = max(
                                [line.get("max_entropy", 0.0) for line in line_results],
                                default=0.0,
                            )
                        else:
                            # Recognition-only mode: single line
                            first_line = line_results[0]
                            pred_a = first_line.get("text", pred_a)
                            entropy = first_line.get("max_entropy", entropy)

                        # Update in result
                        result.pred_a = pred_a
                        result.pred_final = pred_a

                        # Recompute ppl based on new text
                        if self.router is not None and pred_a:
                            try:
                                ppl = self.router.compute_ppl(pred_a)
                            except Exception:
                                pass
                except Exception as e:
                    result.error = f"Agent A error: {e}"

            # Compute CER for Agent A
            result.cer_a = compute_cer(result.pred_a, gt)

            # Routing decision
            is_routed = self.should_route(entropy, ppl)
            result.is_routed = is_routed

            # Agent B correction if needed
            if is_routed and self.agent_b is not None and image is not None:
                try:
                    corrected = self.agent_b.inference(
                        crop_image=image,
                        context_left=context_left,
                        context_right=context_right,
                        ocr_pred=pred_a,
                    )
                    result.pred_final = corrected
                except Exception as e:
                    result.error = f"Agent B error: {e}"

            # Compute CER for final prediction
            result.cer_final = compute_cer(result.pred_final, gt)

            # Check correctness based on CER threshold
            result.is_correct = is_correct_by_cer(result.pred_final, gt, CER_THRESHOLD)

        except Exception as e:
            result.error = str(e)

        return result


# =============================================================================
# Metrics Calculation
# =============================================================================


def compute_paper_metrics(results: List[SampleResult]) -> PaperMetrics:
    """Compute the 5 core metrics for paper using CER.

    Args:
        results: List of sample results.

    Returns:
        PaperMetrics with all metrics.
    """
    metrics = PaperMetrics()

    n = len(results)
    if n == 0:
        return metrics

    metrics.total_samples = n

    # 1. Baseline Accuracy (Agent A only) - CER-based
    agent_a_correct = sum(1 for r in results if is_correct_by_cer(r.pred_a, r.gt))
    metrics.baseline_accuracy = agent_a_correct / n

    # 2. L2W1 Accuracy (Ours - final prediction) - CER-based
    final_correct = sum(1 for r in results if r.is_correct)
    metrics.l2w1_accuracy = final_correct / n

    # 3. Router Activation Rate
    routed = sum(1 for r in results if r.is_routed)
    metrics.router_activation_rate = routed / n

    # 4. Correction Success Rate
    # = (Agent A wrong AND final correct) / (Agent A wrong AND routed)
    routed_and_a_wrong = [
        r
        for r in results
        if r.is_routed and not is_correct_by_cer(r.pred_a, r.gt)
    ]
    if len(routed_and_a_wrong) > 0:
        fixed = sum(1 for r in routed_and_a_wrong if r.is_correct)
        metrics.correction_success_rate = fixed / len(routed_and_a_wrong)

    # 5. Average CER
    metrics.avg_cer_baseline = sum(r.cer_a for r in results) / n
    metrics.avg_cer_l2w1 = sum(r.cer_final for r in results) / n

    return metrics


# =============================================================================
# Output Functions
# =============================================================================


def save_csv(results: List[SampleResult], output_path: Path) -> None:
    """Save results to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "gt",
                "pred_a",
                "pred_final",
                "is_routed",
                "is_correct",
                "cer_a",
                "cer_final",
                "error",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    logger.info(f"CSV saved: {output_path}")


def print_markdown_table(metrics: PaperMetrics) -> None:
    """Print metrics as Markdown table for paper."""
    improvement = (metrics.l2w1_accuracy - metrics.baseline_accuracy) * 100

    print()
    print("=" * 70)
    print("  📊 PAPER METRICS TABLE (CER-Based Evaluation)")
    print("=" * 70)
    print()
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Total Samples | {metrics.total_samples} |")
    print(
        f"| Baseline Accuracy (Agent A only) | {metrics.baseline_accuracy * 100:.2f}% |"
    )
    print(f"| **L2W1 Accuracy (Ours)** | **{metrics.l2w1_accuracy * 100:.2f}%** |")
    print(f"| Accuracy Improvement | +{improvement:.2f}% |")
    print(f"| Router Activation Rate | {metrics.router_activation_rate * 100:.2f}% |")
    print(
        f"| Correction Success Rate | {metrics.correction_success_rate * 100:.2f}% |"
    )
    print(f"| Avg CER (Baseline) | {metrics.avg_cer_baseline:.4f} |")
    print(f"| Avg CER (L2W1) | {metrics.avg_cer_l2w1:.4f} |")
    print()
    print("=" * 70)
    print()

    # Also print LaTeX format
    print("  📄 LaTeX Table Format:")
    print("-" * 70)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{L2W1 Evaluation Results (CER-Based)}")
    print(r"\begin{tabular}{lc}")
    print(r"\toprule")
    print(r"Metric & Value \\")
    print(r"\midrule")
    print(f"Total Samples & {metrics.total_samples} \\\\")
    print(
        f"Baseline Accuracy (Agent A) & {metrics.baseline_accuracy * 100:.2f}\\% \\\\"
    )
    print(
        f"\\textbf{{L2W1 Accuracy (Ours)}} & \\textbf{{{metrics.l2w1_accuracy * 100:.2f}\\%}} \\\\"
    )
    print(f"Improvement & +{improvement:.2f}\\% \\\\")
    print(
        f"Router Activation Rate & {metrics.router_activation_rate * 100:.2f}\\% \\\\"
    )
    print(
        f"Correction Success Rate & {metrics.correction_success_rate * 100:.2f}\\% \\\\"
    )
    print(f"Avg CER (Baseline) & {metrics.avg_cer_baseline:.4f} \\\\")
    print(f"Avg CER (L2W1) & {metrics.avg_cer_l2w1:.4f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("-" * 70)
    print()


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="L2W1 Batch Inference for Paper Experiments (CER-Based)",
    )
    parser.add_argument(
        "--test_set",
        type=Path,
        default=None,
        help=f"Path to test set JSON (default: {DEFAULT_TEST_SET})",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=OUTPUT_CSV,
        help=f"Output CSV path (default: {OUTPUT_CSV})",
    )
    parser.add_argument(
        "--ppl_threshold",
        type=float,
        default=PPL_THRESHOLD,
        help=f"PPL threshold (default: {PPL_THRESHOLD})",
    )
    parser.add_argument(
        "--rec_only",
        action="store_true",
        help="Recognition-only mode (skip detection) for cropped single-character images",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU only",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit samples (for debugging)",
    )
    parser.add_argument(
        "--cer_threshold",
        type=float,
        default=CER_THRESHOLD,
        help=f"CER threshold for correctness (default: {CER_THRESHOLD})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Update global CER threshold if provided
    global CER_THRESHOLD
    CER_THRESHOLD = args.cer_threshold

    print()
    print("=" * 70)
    print("  L2W1 Batch Inference - Paper Experiments (CER-Based)")
    print("=" * 70)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  CER Threshold: {CER_THRESHOLD} (CER < {CER_THRESHOLD} = correct)")
    print()

    # Find test set
    test_set_path = args.test_set
    if test_set_path is None:
        if DEFAULT_TEST_SET.exists():
            test_set_path = DEFAULT_TEST_SET
        elif FALLBACK_TEST_SET.exists():
            test_set_path = FALLBACK_TEST_SET
            logger.warning(f"Using fallback: {FALLBACK_TEST_SET}")
        else:
            logger.error(f"Test set not found!")
            logger.error(f"  Expected: {DEFAULT_TEST_SET}")
            logger.error(f"  Fallback: {FALLBACK_TEST_SET}")
            logger.error(
                f"Please create the test set or use --test_set to specify a path."
            )
            return

    # Load test set
    try:
        samples = load_test_set(test_set_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    if args.limit:
        samples = samples[: args.limit]
        logger.info(f"Limited to {len(samples)} samples")

    # Initialize pipeline
    print()
    print("-" * 70)
    print("  Initializing Pipeline...")
    print("-" * 70)

    # Enable rec_only automatically for CASIA single-character set
    # For real-world data (viscgec, real_handwritten), force detection mode
    auto_rec_only = "casia" in test_set_path.name.lower()
    rec_only_flag = args.rec_only or auto_rec_only

    if auto_rec_only and not args.rec_only:
        logger.info("Auto-enabling rec_only mode for CASIA test set (skip detection).")
    elif not auto_rec_only:
        logger.info(
            "Real-world dataset detected. Detection mode will be auto-enabled for wide images."
        )

    pipeline = L2W1Pipeline(
        ppl_threshold=args.ppl_threshold,
        use_gpu=not args.cpu,
        skip_detection=rec_only_flag,
    )

    # Run inference
    print()
    print("-" * 70)
    print("  Running Batch Inference...")
    print("-" * 70)
    print()

    results: List[SampleResult] = []

    iterator = (
        tqdm(samples, desc="Processing", unit="sample") if TQDM_AVAILABLE else samples
    )
    if not TQDM_AVAILABLE:
        logger.info(f"Processing {len(samples)} samples...")

    for sample in iterator:
        try:
            # Load image
            image_path = resolve_image_path(
                sample.get("image_path", ""),
                test_set_path.parent,
            )

            image = None
            if image_path and image_path.exists():
                try:
                    image = Image.open(image_path)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                except Exception as e:
                    logger.warning(f"Failed to load {image_path}: {e}")

            # Run inference
            result = pipeline.infer(sample, image)
            results.append(result)

        except Exception as e:
            logger.error(f"Error on {sample.get('id', '?')}: {e}")
            results.append(
                SampleResult(
                    id=sample.get("id", "unknown"),
                    gt=sample.get("label_gt", ""),
                    pred_a=sample.get("ocr_pred", ""),
                    pred_final=sample.get("ocr_pred", ""),
                    is_routed=False,
                    is_correct=False,
                    cer_a=1.0,
                    cer_final=1.0,
                    error=str(e),
                )
            )

    # Compute metrics
    metrics = compute_paper_metrics(results)

    # Save CSV
    save_csv(results, args.output_csv)

    # Print Markdown table
    print_markdown_table(metrics)

    # Summary
    print("  📁 Output Files:")
    print(f"     CSV: {args.output_csv}")
    print()

    # Cleanup
    logger.info("Cleaning up GPU memory...")
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=" * 70)
    print("  ✓ Evaluation Complete")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
