#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Long Text Sliding Window Inference (Tuned for Small Text Strips).

TARGET IMAGE: 268x41 pixels
LOGIC:
- Resize height to 48px (Width becomes ~313px)
- Window Size 128px -> Forces ~3 overlapping windows
- Detection OFF (Rec Only) -> Essential for tightly cropped text
"""

import sys
from pathlib import Path

# Path Patching
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
import cv2
import numpy as np
from PIL import Image

# =============================================================================
# TUNED CONFIGURATION
# =============================================================================

# [CRITICAL] Small window to force sliding on small images
DEFAULT_WINDOW_SIZE = 128  
DEFAULT_STRIDE = 96        # 32px overlap (25%)
DEFAULT_TARGET_HEIGHT = 48 

PPL_THRESHOLD = 100.0
ENTROPY_THRESHOLD = 0.3

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Helpers
# =============================================================================

def simple_resize(image: np.ndarray, target_height: int = 48):
    """Resize preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    new_h = target_height
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    print(f"  [Preprocess] Resize: {w}x{h} -> {new_w}x{new_h} (Scale: {scale:.2f})")
    return resized

def extract_window(image: np.ndarray, start_x: int, end_x: int, pad_to_size: int = None) -> np.ndarray:
    """Extract window and pad if needed (for the last chunk)."""
    window = image[:, start_x:end_x].copy()
    h, w = window.shape[:2]
    
    if pad_to_size is not None and w < pad_to_size:
        pad_w = pad_to_size - w
        # Pad with white
        if len(window.shape) == 3:
            pad_img = np.full((h, pad_w, 3), 255, dtype=np.uint8)
        else:
            pad_img = np.full((h, pad_w), 255, dtype=np.uint8)
        window = np.hstack([window, pad_img])
        
    return window

def deduplicate_overlap(accumulated: str, new_text: str, overlap_chars: int = 2) -> str:
    """Simple suffix-prefix matching."""
    if not accumulated or not new_text:
        return new_text
    
    # Check overlap of 1..overlap_chars
    check_len = min(overlap_chars + 1, len(accumulated), len(new_text))
    for i in range(check_len, 0, -1):
        if accumulated[-i:] == new_text[:i]:
            return new_text[i:]
    return new_text

# =============================================================================
# Pipeline Class
# =============================================================================

class SlidingWindowPipeline:
    def __init__(self):
        print("  [Init] Loading Models...")
        self._init_agents()
    
    def _init_agents(self):
        # 1. Agent A
        try:
            from core.agent_a import AgentA
            self.agent_a = AgentA(use_gpu=True)
        except Exception as e:
            print(f"    ! Agent A Error: {e}")
            self.agent_a = None

        # 2. Router
        try:
            from core.router import Router
            self.router = Router()
        except:
            self.router = None
            
        # 3. Agent B
        try:
            from core.agent_b import AgentB
            self.agent_b = AgentB(model_path="Qwen/Qwen2-VL-2B-Instruct", load_in_4bit=True)
        except:
            self.agent_b = None

    def process_window(self, window_image: np.ndarray, context: str = "") -> dict:
        result = {
            "ocr_pred": "", 
            "l2w1_pred": "", 
            "is_routed": False, 
            "ppl_score": 0.0
        }
        
        window_rgb = cv2.cvtColor(window_image, cv2.COLOR_BGR2RGB)
        
        # --- Step 1: Agent A (FORCE REC ONLY) ---
        if self.agent_a:
            try:
                # skip_detection=True is CRITICAL here
                raw_res = self.agent_a.inference(window_rgb, skip_detection=True)
                if raw_res:
                    parts = [r.get("text", "") for r in raw_res]
                    result["ocr_pred"] = "".join(parts)
                    result["l2w1_pred"] = result["ocr_pred"]
            except Exception as e:
                print(f"    ! OCR Failed: {e}")

        # --- Step 2: Router ---
        full_text = context + result["ocr_pred"]
        if self.router and full_text and len(full_text) > 1:
            try:
                ppl = self.router.compute_ppl(full_text)
                result["ppl_score"] = ppl
                # Route if PPL is high
                if ppl > PPL_THRESHOLD:
                    result["is_routed"] = True
            except: pass

        # --- Step 3: Agent B ---
        if result["is_routed"] and self.agent_b and result["ocr_pred"]:
            try:
                pil_img = Image.fromarray(window_rgb)
                correction = self.agent_b.inference(
                    crop_image=pil_img, 
                    context_left=context[-10:], # Minimal context for small windows
                    context_right="", 
                    ocr_pred=result["ocr_pred"]
                )
                if correction and correction != "KEEP":
                    result["l2w1_pred"] = correction
            except: pass
            
        return result

# =============================================================================
# Main
# =============================================================================

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=Path, required=True)
    # Allow override, but default is tuned for small image
    parser.add_argument("--window_size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    args = parser.parse_args()
    
    print("-" * 60)
    print(f"L2W1 Inference | Image: {args.image_path.name}")
    print("-" * 60)

    # 1. Load
    path = args.image_path
    if not path.exists():
        path = PROJECT_ROOT / args.image_path
        if not path.exists():
            print(f"File not found: {args.image_path}")
            return

    img = cv2.imread(str(path))
    
    # 2. Resize (Height 48)
    img_resized = simple_resize(img, target_height=48)
    h, w = img_resized.shape[:2]
    
    # 3. Generate Windows
    windows = []
    for x in range(0, w, args.stride):
        end = min(x + args.window_size, w)
        windows.append((x, end))
        if end >= w: break
    
    print(f"  [Plan] Image Width: {w}px -> {len(windows)} Windows")
    print(f"         Window Size: {args.window_size}, Stride: {args.stride}")
    
    if len(windows) <= 1:
        print("  [Warning] Window too large for this image! Try reducing --window_size")

    print("-" * 60)

    # 4. Run Pipeline
    pipeline = SlidingWindowPipeline()
    
    full_ocr = ""
    full_l2w1 = ""
    
    print("\nProcessing Windows:")
    for i, (sx, ex) in enumerate(windows):
        window = extract_window(img_resized, sx, ex, pad_to_size=args.window_size)
        
        # Run L2W1
        res = pipeline.process_window(window, context=full_l2w1)
        
        # Deduplicate (overlap=1 char usually safe for tight stride)
        ocr_chunk = deduplicate_overlap(full_ocr, res["ocr_pred"], overlap_chars=1)
        l2w1_chunk = deduplicate_overlap(full_l2w1, res["l2w1_pred"], overlap_chars=1)
        
        full_ocr += ocr_chunk
        full_l2w1 += l2w1_chunk
        
        tag = " [ROUTED]" if res["is_routed"] else ""
        print(f"  Win {i+1}: OCR='{res['ocr_pred']}' -> L2W1='{res['l2w1_pred']}'{tag}")

    print("-" * 60)
    print("FINAL RESULT:")
    print(f"OCR  : {full_ocr}")
    print(f"L2W1 : {full_l2w1}")
    print("-" * 60)

if __name__ == "__main__":
    run()

