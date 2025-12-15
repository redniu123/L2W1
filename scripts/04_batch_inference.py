#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch Inference for Paper Experiments.

Generates core metrics for the L2W1 paper experiments section.

Metrics:
1. Total Samples
2. Baseline Accuracy (Agent A only)
3. L2W1 Accuracy (Ours)
4. Router Activation Rate
5. Correction Success Rate

Output:
- CSV: output/batch_eval_details.csv
- Console: Markdown table for paper

Usage:
    python scripts/04_batch_inference.py
    python scripts/04_batch_inference.py --test_set data/test_set.json
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
import csv
import json
import logging
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

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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
    is_correct: bool  # Whether final prediction matches GT
    error: str = ""  # Error message if any


@dataclass 
class PaperMetrics:
    """Core metrics for paper."""
    total_samples: int = 0
    baseline_accuracy: float = 0.0  # Agent A only
    l2w1_accuracy: float = 0.0  # Our method
    router_activation_rate: float = 0.0  # Efficiency metric
    correction_success_rate: float = 0.0  # Agent B efficacy


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
        
        result = SampleResult(
            id=sample_id,
            gt=gt,
            pred_a=pred_a,
            pred_final=pred_a,  # Default to Agent A
            is_routed=False,
            is_correct=False,
        )

        try:
            # If no OCR prediction, run Agent A (supports rec-only mode)
            if (pred_a is None or pred_a == "") and self.agent_a is not None and image is not None:
                try:
                    line_results = self.agent_a.inference(
                        image,
                        skip_detection=self.skip_detection,
                    )
                    if line_results:
                        # Use first line result for single-character images
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

            # Check correctness
            result.is_correct = (result.pred_final == gt)

        except Exception as e:
            result.error = str(e)
        
        return result


# =============================================================================
# Metrics Calculation
# =============================================================================

def compute_paper_metrics(results: List[SampleResult]) -> PaperMetrics:
    """Compute the 5 core metrics for paper.
    
    Args:
        results: List of sample results.
        
    Returns:
        PaperMetrics with all 5 metrics.
    """
    metrics = PaperMetrics()
    
    n = len(results)
    if n == 0:
        return metrics
    
    metrics.total_samples = n
    
    # 1. Baseline Accuracy (Agent A only)
    agent_a_correct = sum(1 for r in results if r.pred_a == r.gt)
    metrics.baseline_accuracy = agent_a_correct / n
    
    # 2. L2W1 Accuracy (Ours - final prediction)
    final_correct = sum(1 for r in results if r.is_correct)
    metrics.l2w1_accuracy = final_correct / n
    
    # 3. Router Activation Rate
    routed = sum(1 for r in results if r.is_routed)
    metrics.router_activation_rate = routed / n
    
    # 4. Correction Success Rate
    # = (Agent A wrong AND final correct) / (Agent A wrong AND routed)
    routed_and_a_wrong = [r for r in results if r.is_routed and r.pred_a != r.gt]
    if len(routed_and_a_wrong) > 0:
        fixed = sum(1 for r in routed_and_a_wrong if r.is_correct)
        metrics.correction_success_rate = fixed / len(routed_and_a_wrong)
    
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
            fieldnames=["id", "gt", "pred_a", "pred_final", "is_routed", "is_correct", "error"],
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
    print("  📊 PAPER METRICS TABLE (Copy to Paper)")
    print("=" * 70)
    print()
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Total Samples | {metrics.total_samples} |")
    print(f"| Baseline Accuracy (Agent A only) | {metrics.baseline_accuracy * 100:.2f}% |")
    print(f"| **L2W1 Accuracy (Ours)** | **{metrics.l2w1_accuracy * 100:.2f}%** |")
    print(f"| Accuracy Improvement | +{improvement:.2f}% |")
    print(f"| Router Activation Rate | {metrics.router_activation_rate * 100:.2f}% |")
    print(f"| Correction Success Rate | {metrics.correction_success_rate * 100:.2f}% |")
    print()
    print("=" * 70)
    print()
    
    # Also print LaTeX format
    print("  📄 LaTeX Table Format:")
    print("-" * 70)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{L2W1 Evaluation Results}")
    print(r"\begin{tabular}{lc}")
    print(r"\toprule")
    print(r"Metric & Value \\")
    print(r"\midrule")
    print(f"Total Samples & {metrics.total_samples} \\\\")
    print(f"Baseline Accuracy (Agent A) & {metrics.baseline_accuracy * 100:.2f}\\% \\\\")
    print(f"\\textbf{{L2W1 Accuracy (Ours)}} & \\textbf{{{metrics.l2w1_accuracy * 100:.2f}\\%}} \\\\")
    print(f"Improvement & +{improvement:.2f}\\% \\\\")
    print(f"Router Activation Rate & {metrics.router_activation_rate * 100:.2f}\\% \\\\")
    print(f"Correction Success Rate & {metrics.correction_success_rate * 100:.2f}\\% \\\\")
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
        description="L2W1 Batch Inference for Paper Experiments",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print()
    print("=" * 70)
    print("  L2W1 Batch Inference - Paper Experiments")
    print("=" * 70)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
            logger.error(f"Please create the test set or use --test_set to specify a path.")
            return
    
    # Load test set
    try:
        samples = load_test_set(test_set_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    if args.limit:
        samples = samples[:args.limit]
        logger.info(f"Limited to {len(samples)} samples")
    
    # Initialize pipeline
    print()
    print("-" * 70)
    print("  Initializing Pipeline...")
    print("-" * 70)
    
    # Enable rec_only automatically for CASIA single-character set
    auto_rec_only = "casia" in test_set_path.name.lower()
    rec_only_flag = args.rec_only or auto_rec_only

    if auto_rec_only and not args.rec_only:
        logger.info("Auto-enabling rec_only mode for CASIA test set (skip detection).")

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
    
    iterator = tqdm(samples, desc="Processing", unit="sample") if TQDM_AVAILABLE else samples
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
            results.append(SampleResult(
                id=sample.get("id", "unknown"),
                gt=sample.get("label_gt", ""),
                pred_a=sample.get("ocr_pred", ""),
                pred_final=sample.get("ocr_pred", ""),
                is_routed=False,
                is_correct=False,
                error=str(e),
            ))
    
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
