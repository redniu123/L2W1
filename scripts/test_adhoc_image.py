#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ad-hoc Image Testing Script for L2W1 Pipeline.

A simple script to test the L2W1 hierarchical multi-agent pipeline
on any image without needing a dataset JSON file.

Usage:
    python scripts/test_adhoc_image.py --image_path /path/to/image.jpg
    python scripts/test_adhoc_image.py --image_path test.png --context "这是测试"

Example:
    python scripts/test_adhoc_image.py --image_path output/crops/demo_test_image_char0000_371d09fe.jpg
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
import logging
import time
from typing import Optional

import torch
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================

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


def print_key_value(key: str, value: str, indent: int = 4) -> None:
    """Print a key-value pair with indentation."""
    prefix = " " * indent
    print(f"{prefix}{key}: {value}")


def print_result_box(ocr_pred: str, final_pred: str) -> None:
    """Print the final result in a box."""
    print()
    print("  +" + "-" * 66 + "+")
    print("  |" + " " * 66 + "|")
    print("  |" + f"    OCR Prediction:    '{ocr_pred}'".ljust(66) + "|")
    print("  |" + f"    Final Prediction:  '{final_pred}'".ljust(66) + "|")
    print("  |" + " " * 66 + "|")
    if ocr_pred != final_pred:
        print("  |" + "    [CORRECTED BY AGENT B]".center(66) + "|")
    else:
        print("  |" + "    [NO CORRECTION NEEDED]".center(66) + "|")
    print("  |" + " " * 66 + "|")
    print("  +" + "-" * 66 + "+")
    print()


# =============================================================================
# Main Logic
# =============================================================================

def run_pipeline(
    image_path: Path,
    context: Optional[str] = None,
    ppl_threshold: float = PPL_THRESHOLD,
    entropy_threshold: float = ENTROPY_THRESHOLD,
) -> None:
    """Run the L2W1 pipeline on a single image.
    
    Args:
        image_path: Path to the input image.
        context: Optional context text for Agent B.
        ppl_threshold: PPL threshold for routing.
        entropy_threshold: Entropy threshold for routing.
    """
    total_start_time = time.time()
    
    print_banner("L2W1 Ad-hoc Image Test", "=")
    print()
    print(f"  Image Path: {image_path}")
    print(f"  Context:    {context if context else '(none provided)'}")
    print(f"  PPL Threshold: {ppl_threshold}")
    print(f"  Entropy Threshold: {entropy_threshold}")
    
    # =========================================================================
    # Step 0: Load Image
    # =========================================================================
    print_stage(0, "Loading Image")
    
    if not image_path.exists():
        print(f"  ❌ ERROR: Image not found at {image_path}")
        return
    
    try:
        image = Image.open(image_path)
        print(f"  ✓ Image loaded successfully")
        print(f"    Size: {image.size}")
        print(f"    Mode: {image.mode}")
    except Exception as e:
        print(f"  ❌ ERROR loading image: {e}")
        return
    
    # =========================================================================
    # Step 1: Agent A - OCR Recognition
    # =========================================================================
    print_stage(1, "Agent A (The Scout) - OCR Recognition")
    
    agent_a = None
    ocr_text = ""
    ocr_confidence = 0.0
    ocr_entropy = 0.0
    char_results = []
    
    try:
        print()
        print("  Initializing Agent A (PaddleOCR)...")
        from core.agent_a import AgentA
        agent_a = AgentA(use_gpu=torch.cuda.is_available())
        print("  ✓ Agent A initialized")
        
        print()
        print("  Running OCR inference...")
        
        # Get line-level results
        line_results = agent_a.inference(str(image_path))
        
        if line_results:
            print()
            print("  📝 Line-level Results:")
            for i, line in enumerate(line_results):
                text = line.get("text", "")
                score = line.get("score", 0)
                avg_entropy = line.get("avg_entropy", 0)
                print(f"    [{i}] Text: '{text}'")
                print(f"        Confidence: {score:.4f}")
                print(f"        Avg Entropy: {avg_entropy:.4f}")
                ocr_text += text
                ocr_confidence = max(ocr_confidence, score)
                ocr_entropy = max(ocr_entropy, avg_entropy)
        else:
            print("  ⚠️ No text detected by Agent A")
        
        # Get character-level results
        char_results = agent_a.inference_with_char_boxes(str(image_path))
        
        if char_results:
            print()
            print("  📊 Character-level Results:")
            for i, char_result in enumerate(char_results):
                char = char_result.get("char", "?")
                score = char_result.get("score", 0)
                entropy = char_result.get("entropy", 0)
                entropy_flag = "⚠️" if entropy > entropy_threshold else "✓"
                print(f"    [{i}] '{char}' | conf: {score:.3f} | entropy: {entropy:.4f} {entropy_flag}")
        
    except Exception as e:
        print(f"  ❌ ERROR: Agent A failed: {e}")
        import traceback
        print(f"  {traceback.format_exc()}")
        # Continue without Agent A - use placeholder
        ocr_text = "?"
    
    print()
    print("  " + "─" * 50)
    print(f"  📋 Agent A Summary:")
    print(f"     OCR Text:    '{ocr_text}'")
    print(f"     Confidence:  {ocr_confidence:.4f}")
    print(f"     Max Entropy: {ocr_entropy:.4f}")
    
    if ocr_entropy > entropy_threshold:
        print(f"     [!] High entropy detected (> {entropy_threshold}) - Agent A is uncertain")
    
    # =========================================================================
    # Step 2: Router - Semantic Perplexity Check
    # =========================================================================
    print_stage(2, "Router (The Gatekeeper) - Semantic Perplexity Check")
    
    router = None
    ppl_score = 0.0
    route_to_agent_b = False
    
    try:
        print()
        print("  Initializing Router (Qwen2.5-0.5B)...")
        from core.router import Router
        router = Router()
        print("  ✓ Router initialized")
        
        print()
        print("  Computing perplexity...")
        
        if ocr_text:
            ppl_score = router.compute_ppl(ocr_text)
            print(f"  📊 PPL Score: {ppl_score:.2f}")
            
            # Get token-level losses for detailed analysis
            token_losses = router.get_token_losses(ocr_text)
            if token_losses:
                print()
                print("  📈 Token-level Analysis:")
                for token, loss in token_losses:
                    loss_flag = "⚠️" if loss > 5.0 else "  "
                    print(f"    {loss_flag} '{token}': loss = {loss:.2f}")
        else:
            print("  ⚠️ No text to analyze - skipping PPL computation")
        
    except Exception as e:
        print(f"  ❌ ERROR: Router failed: {e}")
        import traceback
        print(f"  {traceback.format_exc()}")
    
    # Routing decision
    print()
    print("  " + "─" * 50)
    print("  🚦 Routing Decision:")
    
    if ppl_score > ppl_threshold:
        print(f"     PPL ({ppl_score:.2f}) > Threshold ({ppl_threshold})")
        print(f"     → HIGH PERPLEXITY DETECTED")
        route_to_agent_b = True
    elif ocr_entropy > entropy_threshold:
        print(f"     Entropy ({ocr_entropy:.4f}) > Threshold ({entropy_threshold})")
        print(f"     → HIGH UNCERTAINTY DETECTED")
        route_to_agent_b = True
    else:
        print(f"     PPL ({ppl_score:.2f}) <= Threshold ({ppl_threshold})")
        print(f"     Entropy ({ocr_entropy:.4f}) <= Threshold ({entropy_threshold})")
        print(f"     → Text appears coherent, no correction needed")
    
    if route_to_agent_b:
        print()
        print("     ⚡ ROUTING TO AGENT B FOR CORRECTION")
    else:
        print()
        print("     ✓ KEEPING AGENT A'S PREDICTION")
    
    # =========================================================================
    # Step 3: Agent B - Visual Correction (if needed)
    # =========================================================================
    final_prediction = ocr_text
    agent_b_correction = None
    
    if route_to_agent_b:
        print_stage(3, "Agent B (The Judge) - Visual Chain-of-Thought Correction")
        
        try:
            print()
            print("  Initializing Agent B (Qwen2-VL-2B)...")
            from core.agent_b import AgentB
            agent_b = AgentB(
                model_path="Qwen/Qwen2-VL-2B-Instruct",
                load_in_4bit=True,
            )
            print("  ✓ Agent B initialized")
            
            # Prepare context
            context_left = context if context else ""
            context_right = ""
            
            # If context provided, try to split it
            if context and " " in context:
                parts = context.split(" ", 1)
                context_left = parts[0]
                context_right = parts[1] if len(parts) > 1 else ""
            
            print()
            print("  📝 V-CoT Prompt:")
            print(f"     Context Left:  '{context_left}'")
            print(f"     Context Right: '{context_right}'")
            print(f"     OCR Prediction: '{ocr_text}'")
            
            print()
            print("  Running Agent B inference...")
            
            # Ensure image is in correct format
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Run inference for each character or for the whole text
            if char_results:
                # Process each character crop
                corrections = []
                for i, char_result in enumerate(char_results):
                    char = char_result.get("char", "?")
                    entropy = char_result.get("entropy", 0)
                    
                    # Only correct high-entropy characters
                    if entropy > entropy_threshold:
                        print(f"     Correcting character '{char}' (entropy={entropy:.4f})...")
                        
                        # Use the full image for now (ideally we'd crop)
                        try:
                            corrected = agent_b.inference(
                                crop_image=image,
                                context_left=context_left,
                                context_right=context_right,
                                ocr_pred=char,
                            )
                            print(f"     Agent B: '{char}' → '{corrected}'")
                            corrections.append((i, char, corrected))
                        except Exception as e:
                            print(f"     ⚠️ Failed to correct '{char}': {e}")
                            corrections.append((i, char, char))
                    else:
                        corrections.append((i, char, char))
                
                # Build final text from corrections
                final_prediction = "".join([c[2] for c in corrections])
                agent_b_correction = final_prediction
            else:
                # Process the whole text at once
                corrected = agent_b.inference(
                    crop_image=image,
                    context_left=context_left,
                    context_right=context_right,
                    ocr_pred=ocr_text,
                )
                final_prediction = corrected
                agent_b_correction = corrected
            
            print()
            print(f"  ✓ Agent B Correction: '{agent_b_correction}'")
            
            # Cleanup
            del agent_b
            
        except Exception as e:
            print(f"  ❌ ERROR: Agent B failed: {e}")
            import traceback
            print(f"  {traceback.format_exc()}")
            print()
            print("  ⚠️ Using Agent A's prediction as fallback")
    else:
        print()
        print("  [Agent B skipped - not needed]")
    
    # =========================================================================
    # Final Results
    # =========================================================================
    print_banner("FINAL RESULTS", "*")
    
    total_time = time.time() - total_start_time
    
    print()
    print("  📊 Pipeline Summary:")
    print(f"     Input Image:      {image_path.name}")
    print(f"     Agent A Result:   '{ocr_text}'")
    print(f"     OCR Confidence:   {ocr_confidence:.4f}")
    print(f"     OCR Entropy:      {ocr_entropy:.4f}")
    print(f"     PPL Score:        {ppl_score:.2f}")
    print(f"     Routed to B:      {'Yes' if route_to_agent_b else 'No'}")
    if agent_b_correction:
        print(f"     Agent B Result:   '{agent_b_correction}'")
    print(f"     Total Time:       {total_time:.2f}s")
    
    print_result_box(ocr_text, final_prediction)
    
    # Cleanup GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print_banner("Test Complete", "=")


# =============================================================================
# Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test L2W1 pipeline on a single image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_adhoc_image.py --image_path test.png
  python scripts/test_adhoc_image.py --image_path output/demo_test_image.png --context "这是测试"
  python scripts/test_adhoc_image.py --image_path my_image.jpg --ppl_threshold 50
        """,
    )
    
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image",
    )
    
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Optional context text (e.g., surrounding characters)",
    )
    
    parser.add_argument(
        "--ppl_threshold",
        type=float,
        default=PPL_THRESHOLD,
        help=f"PPL threshold for routing to Agent B (default: {PPL_THRESHOLD})",
    )
    
    parser.add_argument(
        "--entropy_threshold",
        type=float,
        default=ENTROPY_THRESHOLD,
        help=f"Entropy threshold for routing (default: {ENTROPY_THRESHOLD})",
    )
    
    parser.add_argument(
        "--force_agent_b",
        action="store_true",
        help="Force routing to Agent B regardless of thresholds",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Resolve image path
    image_path = Path(args.image_path)
    
    # Try to find the image
    if not image_path.exists():
        # Try relative to project root
        alt_path = PROJECT_ROOT / args.image_path
        if alt_path.exists():
            image_path = alt_path
        else:
            # Try in output directory
            alt_path = PROJECT_ROOT / "output" / args.image_path
            if alt_path.exists():
                image_path = alt_path
            else:
                # Try in output/crops
                alt_path = PROJECT_ROOT / "output" / "crops" / Path(args.image_path).name
                if alt_path.exists():
                    image_path = alt_path
    
    # Adjust thresholds if force_agent_b is set
    ppl_threshold = args.ppl_threshold
    entropy_threshold = args.entropy_threshold
    
    if args.force_agent_b:
        # Set thresholds to 0 to always trigger Agent B
        ppl_threshold = 0.0
        entropy_threshold = 0.0
        print("  [INFO] --force_agent_b enabled: Always routing to Agent B")
    
    # Run the pipeline
    run_pipeline(
        image_path=image_path,
        context=args.context,
        ppl_threshold=ppl_threshold,
        entropy_threshold=entropy_threshold,
    )


if __name__ == "__main__":
    main()

