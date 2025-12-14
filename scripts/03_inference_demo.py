#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-End Inference Demo for L2W1 Hierarchical Multi-Agent Framework.

Spec Reference: L2W1-TEST-001 (End-to-End Inference Demo)

This script demonstrates the complete L2W1 pipeline:
1. Agent A (The Scout) - Initial OCR with entropy calculation
2. Router (The Gatekeeper) - Semantic perplexity check
3. Agent B (The Judge) - Visual-semantic correction via V-CoT

The demo loads a high-risk sample from agent_b_train.json and processes it
through the full pipeline, printing each decision step.
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
import json
import logging
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================

# File paths
TRAIN_JSON_PATH = PROJECT_ROOT / "output" / "agent_b_train.json"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Thresholds
PPL_THRESHOLD = 100.0
ENTROPY_THRESHOLD = 0.3

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Pretty Printing Utilities
# =============================================================================

def print_banner(text: str, char: str = "=", width: int = 70) -> None:
    """Print a banner with the given text."""
    print()
    print(char * width)
    print(f"  {text}")
    print(char * width)


def print_stage(stage_num: int, title: str) -> None:
    """Print a stage header."""
    print()
    print(f"{'─' * 70}")
    print(f"  [Stage {stage_num}] {title}")
    print(f"{'─' * 70}")


def print_result_box(before: str, after: str) -> None:
    """Print the final result in a box."""
    print()
    print("+" + "-" * 68 + "+")
    print("|" + " " * 68 + "|")
    print("|" + f"   BEFORE (Agent A):  '{before}'".ljust(68) + "|")
    print("|" + f"   AFTER  (Agent B):  '{after}'".ljust(68) + "|")
    print("|" + " " * 68 + "|")
    if before != after:
        print("|" + "   [CORRECTED]".center(68) + "|")
    else:
        print("|" + "   [NO CHANGE NEEDED]".center(68) + "|")
    print("|" + " " * 68 + "|")
    print("+" + "-" * 68 + "+")


# =============================================================================
# Data Loading
# =============================================================================

def load_training_data(json_path: Path) -> List[Dict[str, Any]]:
    """Load training data from JSON file."""
    # Try multiple possible paths
    possible_paths = [
        json_path,
        PROJECT_ROOT / json_path.name,
        PROJECT_ROOT / "output" / json_path.name,
        OUTPUT_DIR / json_path.name,
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"Loading training data from: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("samples", [])
    
    raise FileNotFoundError(
        f"Training data not found. Tried: {possible_paths}"
    )


def find_high_risk_sample(
    samples: List[Dict[str, Any]],
    ppl_threshold: float = PPL_THRESHOLD,
) -> Optional[Dict[str, Any]]:
    """Find a high-risk sample for demo.
    
    Priority:
    1. HARD_NEGATIVE_SUB with ppl_score > threshold
    2. HARD_NEGATIVE_SUB with highest ppl_score
    3. Any sample with highest ppl_score
    """
    # Filter HARD_NEGATIVE_SUB samples
    hard_neg_samples = [
        s for s in samples 
        if s.get("sample_type") == "HARD_NEGATIVE_SUB"
    ]
    
    # Find samples with high PPL
    high_ppl_samples = [
        s for s in hard_neg_samples 
        if s.get("ppl_score", 0) > ppl_threshold
    ]
    
    if high_ppl_samples:
        # Return the one with highest PPL
        return max(high_ppl_samples, key=lambda x: x.get("ppl_score", 0))
    
    if hard_neg_samples:
        # Return HARD_NEGATIVE_SUB with highest PPL
        return max(hard_neg_samples, key=lambda x: x.get("ppl_score", 0))
    
    if samples:
        # Return any sample with highest PPL
        return max(samples, key=lambda x: x.get("ppl_score", 0))
    
    return None


# =============================================================================
# Main Demo
# =============================================================================

def run_demo() -> None:
    """Run the end-to-end inference demo."""
    
    print_banner("L2W1 End-to-End Inference Demo", "=")
    print()
    print("  This demo showcases the hierarchical multi-agent pipeline:")
    print("  Agent A (Scout) -> Router (Gatekeeper) -> Agent B (Judge)")
    print()
    
    # =========================================================================
    # Step 0: Load Training Data
    # =========================================================================
    print_stage(0, "Loading Training Data")
    
    try:
        samples = load_training_data(TRAIN_JSON_PATH)
        print(f"  Loaded {len(samples)} samples from {TRAIN_JSON_PATH.name}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Please run scripts/01_build_dataset.py first!")
        return
    
    # Find high-risk sample
    target_sample = find_high_risk_sample(samples)
    
    if target_sample is None:
        print("  ERROR: No suitable sample found for demo.")
        return
    
    print()
    print("  Selected high-risk sample:")
    print(f"    ID:          {target_sample.get('id', 'N/A')}")
    print(f"    OCR Pred:    '{target_sample.get('ocr_pred', 'N/A')}'")
    print(f"    Label GT:    '{target_sample.get('label_gt', 'N/A')}'")
    print(f"    Sample Type: {target_sample.get('sample_type', 'N/A')}")
    print(f"    Confidence:  {target_sample.get('confidence', 0):.4f}")
    print(f"    Entropy:     {target_sample.get('ocr_entropy', 0):.4f}")
    print(f"    PPL Score:   {target_sample.get('ppl_score', 0):.2f}")
    
    # =========================================================================
    # Step 1: Agent A Analysis (already done in preprocessing)
    # =========================================================================
    print_stage(1, "Agent A (The Scout) - Initial OCR Recognition")
    
    ocr_pred = target_sample.get("ocr_pred", "?")
    confidence = target_sample.get("confidence", 0)
    entropy = target_sample.get("ocr_entropy", 0)
    
    print()
    print(f"  Agent A Recognition Result:")
    print(f"    Predicted Character: '{ocr_pred}'")
    print(f"    Confidence Score:    {confidence:.4f}")
    print(f"    Visual Entropy:      {entropy:.4f}")
    print()
    
    if entropy > ENTROPY_THRESHOLD:
        print(f"  [!] Visual Entropy ({entropy:.4f}) > Threshold ({ENTROPY_THRESHOLD})")
        print("      Agent A is UNCERTAIN about this character.")
    else:
        print(f"  [OK] Visual Entropy ({entropy:.4f}) <= Threshold ({ENTROPY_THRESHOLD})")
        print("       Agent A is CONFIDENT about this character.")
    
    # =========================================================================
    # Step 2: Router Semantic Analysis
    # =========================================================================
    print_stage(2, "Router (The Gatekeeper) - Semantic Perplexity Check")
    
    ppl_score = target_sample.get("ppl_score", 0)
    
    print()
    print(f"  Router Semantic Analysis:")
    print(f"    Perplexity Score (PPL): {ppl_score:.2f}")
    print()
    
    if ppl_score > PPL_THRESHOLD:
        print(f"  [ALERT] PPL ({ppl_score:.2f}) > Threshold ({PPL_THRESHOLD})")
        print("          HIGH PERPLEXITY DETECTED!")
        print("          The text is semantically INCOHERENT.")
        print("          -> Routing to Agent B for correction...")
        route_to_agent_b = True
    else:
        print(f"  [OK] PPL ({ppl_score:.2f}) <= Threshold ({PPL_THRESHOLD})")
        print("       Semantic coherence is acceptable.")
        route_to_agent_b = False
    
    # For demo purposes, always proceed to Agent B if it's a hard negative
    if target_sample.get("sample_type") == "HARD_NEGATIVE_SUB":
        print()
        print("  [INFO] Sample is HARD_NEGATIVE_SUB. Proceeding to Agent B for demo.")
        route_to_agent_b = True
    
    if not route_to_agent_b:
        print()
        print("  [SKIP] Agent B not needed. Using Agent A's prediction.")
        print_result_box(ocr_pred, ocr_pred)
        return
    
    # =========================================================================
    # Step 3: Agent B Visual Correction
    # =========================================================================
    print_stage(3, "Agent B (The Judge) - Visual Chain-of-Thought Correction")
    
    # Load the crop image (handle various path formats)
    image_path_rel = target_sample.get("image_path", "")
    
    # Try multiple possible paths
    possible_image_paths = [
        OUTPUT_DIR.parent / image_path_rel,  # Relative from project root
        OUTPUT_DIR / Path(image_path_rel).name,  # Just filename in output dir
        PROJECT_ROOT / image_path_rel,  # Direct relative path
        Path(image_path_rel),  # Absolute path
    ]
    
    image_path = None
    for path in possible_image_paths:
        if path.exists():
            image_path = path
            break
    
    if image_path is None:
        # Last resort: try to find in output/crops
        filename = Path(image_path_rel).name
        crops_dir = OUTPUT_DIR / "crops"
        if crops_dir.exists():
            image_path = crops_dir / filename
    
    print()
    print(f"  Loading crop image: {image_path}")
    
    if not image_path.exists():
        print(f"  ERROR: Image not found at {image_path}")
        print("  Cannot proceed with Agent B correction.")
        return
    
    # Load image
    try:
        crop_image = Image.open(image_path)
        print(f"  Image loaded successfully: {crop_image.size} {crop_image.mode}")
    except Exception as e:
        print(f"  ERROR loading image: {e}")
        return
    
    # Initialize Agent B
    print()
    print("  Initializing Agent B (VLM)...")
    print("  This may take a moment to load the model...")
    print()
    
    try:
        from core.agent_b import AgentB
        agent_b = AgentB(
            model_path="Qwen/Qwen2-VL-2B-Instruct",
            load_in_4bit=True,
        )
    except Exception as e:
        print(f"  ERROR: Failed to initialize Agent B: {e}")
        print()
        print("  [SIMULATION MODE]")
        print("  Agent B not available. Simulating correction...")
        
        # Simulate correction based on GT
        gt_char = target_sample.get("label_gt", ocr_pred)
        print()
        print("  V-CoT Prompt:")
        context_left = target_sample.get("context_left", "")
        context_right = target_sample.get("context_right", "")
        print(f"    Context: [{context_left}] <target> [{context_right}]")
        print(f"    OCR says: '{ocr_pred}'")
        print()
        print(f"  Simulated Correction: '{gt_char}'")
        print_result_box(ocr_pred, gt_char)
        return
    
    # Get context
    context_left = target_sample.get("context_left", "")
    context_right = target_sample.get("context_right", "")
    
    print("  Building V-CoT Prompt...")
    print()
    print("  +--- V-CoT Prompt ---+")
    print(f"  | Context: [{context_left}] <target> [{context_right}]")
    print(f"  | OCR says: '{ocr_pred}'")
    print("  +--------------------+")
    print()
    print("  Running Agent B inference...")
    
    # Run inference
    try:
        corrected_char = agent_b.inference(
            crop_image=crop_image,
            context_left=context_left,
            context_right=context_right,
            ocr_pred=ocr_pred,
        )
        print(f"  Agent B Response: '{corrected_char}'")
    except Exception as e:
        print(f"  ERROR during inference: {e}")
        corrected_char = ocr_pred
    
    # =========================================================================
    # Final Result
    # =========================================================================
    print_banner("FINAL RESULT", "*")
    
    gt_char = target_sample.get("label_gt", "?")
    
    print_result_box(ocr_pred, corrected_char)
    
    print()
    print("  Ground Truth: '{}'".format(gt_char))
    print()
    
    if corrected_char == gt_char:
        print("  [SUCCESS] Agent B correctly identified the character!")
    elif corrected_char == ocr_pred:
        print("  [NO CHANGE] Agent B kept the original prediction.")
        if ocr_pred == gt_char:
            print("              (which is correct!)")
        else:
            print("              (which is incorrect - GT: '{}')".format(gt_char))
    else:
        if corrected_char != gt_char:
            print("  [PARTIAL] Agent B changed the prediction, but result differs from GT.")
            print(f"            Predicted: '{corrected_char}' vs GT: '{gt_char}'")
        else:
            print("  [SUCCESS] Agent B correctly corrected the OCR error!")
    
    print()
    print_banner("Demo Complete", "=")
    
    # Cleanup
    if 'agent_b' in dir():
        del agent_b
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    run_demo()

