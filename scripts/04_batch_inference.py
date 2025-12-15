#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch Inference Script for L2W1 Hierarchical Multi-Agent Framework.

This script runs batch evaluation on a test set and generates metrics for paper.

Metrics computed:
- Total Samples
- Overall Accuracy (final pipeline accuracy)
- Agent A Only Accuracy (baseline without Router/Agent B)
- Router Activation Rate
- Correction Efficacy (Agent B correctly fixes Agent A errors)
- False Alarm Rate (Agent A correct but routed to Agent B)

Output:
- CSV file with detailed per-sample results
- Console summary table with all metrics
"""

# =============================================================================
# Path Patching
# =============================================================================
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Standard Imports
# =============================================================================
import argparse
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

# Try to import pandas and tqdm
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not installed. CSV output will be limited.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not installed. Progress bar will be disabled.")

# =============================================================================
# Configuration
# =============================================================================

# Default paths
DEFAULT_TEST_SET_PATH = PROJECT_ROOT / "output" / "agent_b_train.json"
ALTERNATIVE_TEST_SET_PATHS = [
    PROJECT_ROOT / "data" / "test_set.json",
    PROJECT_ROOT / "output" / "test_set.json",
    PROJECT_ROOT / "raw_data" / "test_set.json",
]
OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_OUTPUT_CSV = OUTPUT_DIR / "batch_eval_results.csv"

# Thresholds
DEFAULT_PPL_THRESHOLD = 100.0
DEFAULT_ENTROPY_THRESHOLD = 0.3

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SampleResult:
    """Result for a single sample inference."""
    sample_id: str
    image_path: str
    ground_truth: str
    agent_a_pred: str
    final_pred: str
    confidence: float = 0.0
    ocr_entropy: float = 0.0
    ppl_score: float = 0.0
    is_routed: bool = False
    agent_a_correct: bool = False
    final_correct: bool = False
    agent_b_correction: str = ""
    error_message: str = ""
    inference_time_ms: float = 0.0


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""
    total_samples: int = 0
    successful_samples: int = 0
    failed_samples: int = 0
    
    # Accuracy metrics
    agent_a_correct: int = 0
    final_correct: int = 0
    agent_a_accuracy: float = 0.0
    overall_accuracy: float = 0.0
    
    # Router metrics
    routed_samples: int = 0
    router_activation_rate: float = 0.0
    
    # Agent B efficacy
    agent_b_fixed_errors: int = 0  # A was wrong, B made it right
    agent_b_introduced_errors: int = 0  # A was right, B made it wrong
    agent_b_no_change: int = 0  # B kept A's prediction
    correction_efficacy: float = 0.0  # Fixed / (Routed and A was wrong)
    false_alarm_rate: float = 0.0  # (A correct but routed) / (A correct)
    
    # Timing
    avg_inference_time_ms: float = 0.0
    total_time_seconds: float = 0.0


# =============================================================================
# Pretty Printing Utilities
# =============================================================================

def print_banner(text: str, char: str = "=", width: int = 70) -> None:
    """Print a banner with the given text."""
    print()
    print(char * width)
    print(f"  {text}")
    print(char * width)


def print_metric_row(name: str, value: Any, width: int = 50) -> None:
    """Print a single metric row."""
    if isinstance(value, float):
        value_str = f"{value:.4f}" if value < 1 else f"{value:.2f}"
    else:
        value_str = str(value)
    print(f"  │ {name:<40} │ {value_str:>8} │")


def print_metrics_table(metrics: EvaluationMetrics) -> None:
    """Print a beautiful summary table of metrics."""
    print()
    print("  ┌" + "─" * 42 + "┬" + "─" * 10 + "┐")
    print("  │" + " EVALUATION METRICS".center(42) + "│" + " VALUE".center(10) + "│")
    print("  ├" + "─" * 42 + "┼" + "─" * 10 + "┤")
    
    # Sample counts
    print_metric_row("Total Samples", metrics.total_samples)
    print_metric_row("Successful Samples", metrics.successful_samples)
    print_metric_row("Failed Samples", metrics.failed_samples)
    
    print("  ├" + "─" * 42 + "┼" + "─" * 10 + "┤")
    
    # Accuracy metrics
    print_metric_row("Agent A Only Accuracy", f"{metrics.agent_a_accuracy * 100:.2f}%")
    print_metric_row("Overall Accuracy (with Agent B)", f"{metrics.overall_accuracy * 100:.2f}%")
    print_metric_row("Accuracy Improvement", f"{(metrics.overall_accuracy - metrics.agent_a_accuracy) * 100:+.2f}%")
    
    print("  ├" + "─" * 42 + "┼" + "─" * 10 + "┤")
    
    # Router metrics
    print_metric_row("Routed to Agent B", metrics.routed_samples)
    print_metric_row("Router Activation Rate", f"{metrics.router_activation_rate * 100:.2f}%")
    
    print("  ├" + "─" * 42 + "┼" + "─" * 10 + "┤")
    
    # Agent B metrics
    print_metric_row("Agent B Corrections (fixed errors)", metrics.agent_b_fixed_errors)
    print_metric_row("Agent B False Corrections (made worse)", metrics.agent_b_introduced_errors)
    print_metric_row("Correction Efficacy", f"{metrics.correction_efficacy * 100:.2f}%")
    print_metric_row("False Alarm Rate", f"{metrics.false_alarm_rate * 100:.2f}%")
    
    print("  ├" + "─" * 42 + "┼" + "─" * 10 + "┤")
    
    # Timing
    print_metric_row("Avg Inference Time (ms)", f"{metrics.avg_inference_time_ms:.1f}")
    print_metric_row("Total Time (seconds)", f"{metrics.total_time_seconds:.1f}")
    
    print("  └" + "─" * 42 + "┴" + "─" * 10 + "┘")
    print()


# =============================================================================
# Data Loading
# =============================================================================

def find_test_set(specified_path: Optional[Path] = None) -> Path:
    """Find the test set file.
    
    Args:
        specified_path: User-specified path, if any.
        
    Returns:
        Path to the test set file.
        
    Raises:
        FileNotFoundError: If no test set file is found.
    """
    paths_to_try = []
    
    if specified_path is not None:
        paths_to_try.append(specified_path)
    
    paths_to_try.append(DEFAULT_TEST_SET_PATH)
    paths_to_try.extend(ALTERNATIVE_TEST_SET_PATHS)
    
    for path in paths_to_try:
        if path.exists():
            logger.info(f"Found test set at: {path}")
            return path
    
    raise FileNotFoundError(
        f"Test set not found. Tried:\n" +
        "\n".join(f"  - {p}" for p in paths_to_try)
    )


def load_test_set(json_path: Path) -> List[Dict[str, Any]]:
    """Load test set from JSON file.
    
    Args:
        json_path: Path to the JSON file.
        
    Returns:
        List of sample dictionaries.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle both flat list and nested format
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        samples = data.get("samples", [])
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")
    
    logger.info(f"Loaded {len(samples)} samples from {json_path.name}")
    return samples


def resolve_image_path(
    image_path_str: str,
    base_dir: Path,
) -> Optional[Path]:
    """Resolve image path, trying multiple locations.
    
    Args:
        image_path_str: Image path from JSON (may be relative).
        base_dir: Base directory for relative paths.
        
    Returns:
        Resolved absolute path, or None if not found.
    """
    possible_paths = [
        base_dir / image_path_str,
        base_dir / "crops" / Path(image_path_str).name,
        OUTPUT_DIR / image_path_str,
        OUTPUT_DIR / "crops" / Path(image_path_str).name,
        PROJECT_ROOT / image_path_str,
        Path(image_path_str),  # Absolute path
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


# =============================================================================
# Pipeline Components
# =============================================================================

class L2W1Pipeline:
    """Unified pipeline for L2W1 hierarchical multi-agent inference.
    
    This class encapsulates:
    - Agent A (PaddleOCR): Initial OCR recognition
    - Router: PPL-based routing decision
    - Agent B (VLM): Visual-semantic correction
    """
    
    def __init__(
        self,
        ppl_threshold: float = DEFAULT_PPL_THRESHOLD,
        entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
        use_gpu: bool = True,
        load_agent_b: bool = True,
    ):
        """Initialize the pipeline.
        
        Args:
            ppl_threshold: PPL threshold for routing to Agent B.
            entropy_threshold: Entropy threshold for routing.
            use_gpu: Whether to use GPU for inference.
            load_agent_b: Whether to load Agent B (set False for Agent A-only testing).
        """
        self.ppl_threshold = ppl_threshold
        self.entropy_threshold = entropy_threshold
        self.agent_a = None
        self.router = None
        self.agent_b = None
        self.use_gpu = use_gpu
        
        self._init_components(load_agent_b)
    
    def _init_components(self, load_agent_b: bool) -> None:
        """Initialize pipeline components."""
        # Agent A
        logger.info("Initializing Agent A (PaddleOCR)...")
        try:
            from core.agent_a import AgentA
            self.agent_a = AgentA(use_gpu=self.use_gpu)
            logger.info("✓ Agent A initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Agent A: {e}")
            logger.warning("Agent A will be skipped - using pre-computed predictions")
        
        # Router
        logger.info("Initializing Router (Qwen2.5-0.5B)...")
        try:
            from core.router import Router
            self.router = Router()
            logger.info("✓ Router initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Router: {e}")
            logger.warning("Router will be skipped - using pre-computed PPL scores")
        
        # Agent B
        if load_agent_b:
            logger.info("Initializing Agent B (Qwen2-VL-2B)...")
            try:
                from core.agent_b import AgentB
                self.agent_b = AgentB(
                    model_path="Qwen/Qwen2-VL-2B-Instruct",
                    load_in_4bit=True,
                )
                logger.info("✓ Agent B initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Agent B: {e}")
                logger.warning("Agent B will be simulated using ground truth")
    
    def should_route_to_agent_b(
        self,
        entropy: float,
        ppl_score: float,
    ) -> bool:
        """Determine if sample should be routed to Agent B.
        
        Args:
            entropy: Visual entropy from Agent A.
            ppl_score: Perplexity score from Router.
            
        Returns:
            True if sample should be routed to Agent B.
        """
        return ppl_score > self.ppl_threshold or entropy > self.entropy_threshold
    
    def run_inference(
        self,
        sample: Dict[str, Any],
        crop_image: Optional[Image.Image] = None,
    ) -> SampleResult:
        """Run full pipeline inference on a sample.
        
        Args:
            sample: Sample dictionary from test set.
            crop_image: Pre-loaded crop image (optional).
            
        Returns:
            SampleResult with inference details.
        """
        start_time = time.time()
        
        # Extract sample info
        sample_id = sample.get("id", "unknown")
        image_path = sample.get("image_path", "")
        ground_truth = sample.get("label_gt", "")
        agent_a_pred = sample.get("ocr_pred", "")
        confidence = sample.get("confidence", 0.0)
        ocr_entropy = sample.get("ocr_entropy", 0.0)
        ppl_score = sample.get("ppl_score", 0.0)
        context_left = sample.get("context_left", "") or ""
        context_right = sample.get("context_right", "") or ""
        
        result = SampleResult(
            sample_id=sample_id,
            image_path=image_path,
            ground_truth=ground_truth,
            agent_a_pred=agent_a_pred,
            final_pred=agent_a_pred,  # Default to Agent A prediction
            confidence=confidence,
            ocr_entropy=ocr_entropy,
            ppl_score=ppl_score,
        )
        
        try:
            # Check if should route to Agent B
            is_routed = self.should_route_to_agent_b(ocr_entropy, ppl_score)
            result.is_routed = is_routed
            
            if is_routed and self.agent_b is not None and crop_image is not None:
                # Run Agent B inference
                try:
                    corrected = self.agent_b.inference(
                        crop_image=crop_image,
                        context_left=context_left,
                        context_right=context_right,
                        ocr_pred=agent_a_pred,
                    )
                    result.final_pred = corrected
                    result.agent_b_correction = corrected
                except Exception as e:
                    logger.warning(f"Agent B inference failed for {sample_id}: {e}")
                    result.error_message = f"Agent B error: {str(e)}"
            elif is_routed:
                # Agent B not available, keep Agent A prediction
                result.error_message = "Agent B not available"
            
            # Compute correctness
            result.agent_a_correct = (agent_a_pred == ground_truth)
            result.final_correct = (result.final_pred == ground_truth)
            
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Inference failed for {sample_id}: {e}")
        
        # Record timing
        result.inference_time_ms = (time.time() - start_time) * 1000
        
        return result


def compute_metrics(results: List[SampleResult]) -> EvaluationMetrics:
    """Compute evaluation metrics from results.
    
    Args:
        results: List of sample results.
        
    Returns:
        Aggregated evaluation metrics.
    """
    metrics = EvaluationMetrics()
    
    metrics.total_samples = len(results)
    metrics.successful_samples = sum(1 for r in results if not r.error_message)
    metrics.failed_samples = metrics.total_samples - metrics.successful_samples
    
    if metrics.total_samples == 0:
        return metrics
    
    # Accuracy
    metrics.agent_a_correct = sum(1 for r in results if r.agent_a_correct)
    metrics.final_correct = sum(1 for r in results if r.final_correct)
    metrics.agent_a_accuracy = metrics.agent_a_correct / metrics.total_samples
    metrics.overall_accuracy = metrics.final_correct / metrics.total_samples
    
    # Router
    metrics.routed_samples = sum(1 for r in results if r.is_routed)
    metrics.router_activation_rate = metrics.routed_samples / metrics.total_samples
    
    # Agent B efficacy
    for r in results:
        if r.is_routed:
            if not r.agent_a_correct and r.final_correct:
                metrics.agent_b_fixed_errors += 1
            elif r.agent_a_correct and not r.final_correct:
                metrics.agent_b_introduced_errors += 1
            elif r.final_pred == r.agent_a_pred:
                metrics.agent_b_no_change += 1
    
    # Correction efficacy: Fixed / (Routed samples where A was wrong)
    routed_and_a_wrong = sum(1 for r in results if r.is_routed and not r.agent_a_correct)
    if routed_and_a_wrong > 0:
        metrics.correction_efficacy = metrics.agent_b_fixed_errors / routed_and_a_wrong
    
    # False alarm rate: (A correct but routed) / (A correct total)
    if metrics.agent_a_correct > 0:
        a_correct_but_routed = sum(1 for r in results if r.is_routed and r.agent_a_correct)
        metrics.false_alarm_rate = a_correct_but_routed / metrics.agent_a_correct
    
    # Timing
    total_time = sum(r.inference_time_ms for r in results)
    metrics.avg_inference_time_ms = total_time / metrics.total_samples if metrics.total_samples > 0 else 0
    metrics.total_time_seconds = total_time / 1000
    
    return metrics


def save_results_to_csv(
    results: List[SampleResult],
    output_path: Path,
) -> None:
    """Save results to CSV file.
    
    Args:
        results: List of sample results.
        output_path: Path to output CSV file.
    """
    if PANDAS_AVAILABLE:
        # Use pandas for nice CSV output
        df = pd.DataFrame([asdict(r) for r in results])
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"Results saved to: {output_path}")
    else:
        # Fallback: manual CSV writing
        import csv
        
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
                writer.writeheader()
                for r in results:
                    writer.writerow(asdict(r))
        logger.info(f"Results saved to: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="L2W1 Batch Inference and Evaluation Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--test-set",
        type=Path,
        default=None,
        help="Path to test set JSON file",
    )
    
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to output CSV file",
    )
    
    parser.add_argument(
        "--ppl-threshold",
        type=float,
        default=DEFAULT_PPL_THRESHOLD,
        help="PPL threshold for routing to Agent B",
    )
    
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=DEFAULT_ENTROPY_THRESHOLD,
        help="Entropy threshold for routing to Agent B",
    )
    
    parser.add_argument(
        "--no-agent-b",
        action="store_true",
        help="Skip Agent B (Agent A only evaluation)",
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU only (no GPU)",
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (for debugging)",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print_banner("L2W1 Batch Inference & Evaluation", "=")
    print()
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PPL Threshold: {args.ppl_threshold}")
    print(f"  Entropy Threshold: {args.entropy_threshold}")
    print(f"  Agent B Enabled: {not args.no_agent_b}")
    print(f"  Device: {'CPU' if args.cpu else 'GPU'}")
    print()
    
    # Find and load test set
    try:
        test_set_path = find_test_set(args.test_set)
        samples = load_test_set(test_set_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    if args.limit is not None:
        samples = samples[:args.limit]
        logger.info(f"Limited to {len(samples)} samples")
    
    # Initialize pipeline
    print_banner("Initializing Pipeline", "-")
    pipeline = L2W1Pipeline(
        ppl_threshold=args.ppl_threshold,
        entropy_threshold=args.entropy_threshold,
        use_gpu=not args.cpu,
        load_agent_b=not args.no_agent_b,
    )
    
    # Run inference
    print_banner("Running Batch Inference", "-")
    results: List[SampleResult] = []
    
    # Create iterator with progress bar
    if TQDM_AVAILABLE:
        iterator = tqdm(samples, desc="Processing", unit="sample")
    else:
        iterator = samples
        logger.info(f"Processing {len(samples)} samples...")
    
    for sample in iterator:
        try:
            # Load crop image if available
            image_path = resolve_image_path(
                sample.get("image_path", ""),
                test_set_path.parent,
            )
            
            crop_image = None
            if image_path is not None and image_path.exists():
                try:
                    crop_image = Image.open(image_path)
                except Exception as e:
                    logger.warning(f"Failed to load image {image_path}: {e}")
            
            # Run inference
            result = pipeline.run_inference(sample, crop_image)
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing sample {sample.get('id', 'unknown')}: {e}")
            logger.debug(traceback.format_exc())
            
            # Create error result
            results.append(SampleResult(
                sample_id=sample.get("id", "unknown"),
                image_path=sample.get("image_path", ""),
                ground_truth=sample.get("label_gt", ""),
                agent_a_pred=sample.get("ocr_pred", ""),
                final_pred=sample.get("ocr_pred", ""),
                error_message=str(e),
            ))
    
    # Compute metrics
    print_banner("Evaluation Results", "=")
    metrics = compute_metrics(results)
    print_metrics_table(metrics)
    
    # Save results
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    save_results_to_csv(results, args.output_csv)
    
    # Print sample details for debugging
    if args.verbose:
        print_banner("Sample Details (First 10)", "-")
        for i, r in enumerate(results[:10]):
            status = "✓" if r.final_correct else "✗"
            routed = "→B" if r.is_routed else "  "
            print(f"  {status} {routed} GT: '{r.ground_truth}' | A: '{r.agent_a_pred}' | Final: '{r.final_pred}'")
    
    # Cleanup
    print()
    logger.info("Cleaning up GPU memory...")
    if hasattr(pipeline, "agent_b") and pipeline.agent_b is not None:
        del pipeline.agent_b
    if hasattr(pipeline, "router") and pipeline.router is not None:
        del pipeline.router
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print_banner("Evaluation Complete", "=")
    print()


if __name__ == "__main__":
    main()

