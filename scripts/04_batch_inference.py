#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch Inference: Line-Level Visual Correction (MVP Style).

RESCUE PLAN:
1. Agent A detects LINES (not chars).
2. If Line Entropy is high -> Agent B reads the WHOLE LINE crop.
3. Use CER for evaluation.
"""

import os

# --- 【添加这两行】强制指定模型路径 & 离线模式 ---
# 1. 指定你解压出来的 my_models 文件夹的绝对路径
os.environ["HF_HOME"] = "/home/coder/project/L2W1/my_models"

# 2. 告诉 Hugging Face 不要联网，直接用本地的
os.environ["HF_HUB_OFFLINE"] = "1"
# ------------------------------------------------


# ... 原本的代码从这里继续 ...

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import Levenshtein
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.agent_a import AgentA
from core.agent_b import AgentB

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for CER calculation."""
    if not text:
        return ""
    # Remove whitespace and common punctuation noise
    return text.strip().replace(" ", "").replace("\n", "")


class L2W1Pipeline:
    def __init__(self):
        logger.info("Initializing L2W1 Pipeline (Line-Level Mode)...")
        # 1. Agent A (The Scout) - Force Detection ON
        self.agent_a = AgentA(use_gpu=True)

        # 2. Agent B (The Judge)
        self.agent_b = AgentB(model_path="Qwen/Qwen2-VL-2B-Instruct", load_in_4bit=True)

    def run(self, image_path: str, gt_text: str = "") -> Dict:
        """Process a single image."""
        # --- Step 1: Agent A (Detection + Recognition) ---
        # We use standard inference, which returns LINES
        # skip_detection=False is CRITICAL here
        results = self.agent_a.inference(image_path, skip_detection=False)

        # Merge all lines into one prediction (for full text evaluation)
        full_pred_a = ""
        full_pred_final = ""

        line_logs = []
        is_routed_any = False

        # Load original image for cropping
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Image load failed"}

        for line in results:
            text_a = line["text"]
            box = line["box"]
            avg_entropy = line["avg_entropy"]

            # --- Step 2: Router Strategy (Line Level) ---
            # If the line is uncertain (Entropy > 0.001) or contains specific keywords
            # For this rescue run, we use a low threshold to force Agent B to work
            is_routed = avg_entropy > 0.001 or len(text_a) < 2

            final_text = text_a

            if is_routed:
                is_routed_any = True
                # --- Step 3: Agent B (Visual Correction) ---
                # Crop the WHOLE line
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))

                # Padding to capture context
                h_img, w_img = img.shape[:2]
                pad = 10
                y_min = max(0, y_min - pad)
                y_max = min(h_img, y_max + pad)
                x_min = max(0, x_min - pad)
                x_max = min(w_img, x_max + pad)

                line_crop = img[y_min:y_max, x_min:x_max]

                # Convert to PIL for Agent B
                from PIL import Image

                line_crop_pil = Image.fromarray(
                    cv2.cvtColor(line_crop, cv2.COLOR_BGR2RGB)
                )

                # Agent B Prompt: Ask to read the line
                # We use a custom call to Agent B's internal model or modify the prompt slightly
                # Here we reuse the interface but treat context as empty
                try:
                    # Construct a prompt that asks to read the text in the image
                    # We hijack the inference method slightly
                    prompt = f"图片中的手写文字OCR识别为：'{text_a}'，可能存在错误。请仔细观察图片，直接输出图片中正确的完整文字。"

                    # Call Agent B logic manually to bypass the "single char" prompt restriction if needed
                    # But let's try using the existing inference first
                    corrected = self.agent_b.inference(
                        crop_image=line_crop_pil,
                        context_left="",  # No external context needed for line
                        context_right="",
                        ocr_pred=text_a,
                    )

                    if (
                        corrected
                        and corrected != text_a
                        and "<UNKNOWN>" not in corrected
                    ):
                        final_text = corrected

                except Exception as e:
                    logger.error(f"Agent B failed: {e}")

            full_pred_a += text_a
            full_pred_final += final_text

            line_logs.append({"ocr": text_a, "l2w1": final_text, "routed": is_routed})

        # --- Step 4: Evaluation ---
        norm_gt = normalize_text(gt_text)
        norm_a = normalize_text(full_pred_a)
        norm_final = normalize_text(full_pred_final)

        cer_a = Levenshtein.distance(norm_a, norm_gt) / len(norm_gt) if norm_gt else 1.0
        cer_final = (
            Levenshtein.distance(norm_final, norm_gt) / len(norm_gt) if norm_gt else 1.0
        )

        return {
            "id": Path(image_path).name,
            "gt": gt_text,
            "pred_a": full_pred_a,
            "pred_final": full_pred_final,
            "cer_a": cer_a,
            "cer_final": cer_final,
            "is_routed": is_routed_any,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="output/rescue_result.csv")
    args = parser.parse_args()

    pipeline = L2W1Pipeline()

    with open(args.test_set, "r") as f:
        data = json.load(f)
        samples = data.get("samples", [])

    results = []
    print(f"\n🚀 Running Rescue Protocol on {len(samples)} samples...\n")

    for sample in tqdm(samples):
        res = pipeline.run(sample["image_path"], sample["label_gt"])
        results.append(res)

        # Print Bad Cases (where Baseline failed)
        if res["cer_a"] > 0.1:
            print(f"\n[Bad Case] {res['id']}")
            print(f"  GT  : {res['gt']}")
            print(f"  OCR : {res['pred_a']}")
            print(f"  L2W1: {res['pred_final']}")

    # Calculate Stats
    avg_cer_a = np.mean([r["cer_a"] for r in results])
    avg_cer_final = np.mean([r["cer_final"] for r in results])

    print("\n" + "=" * 50)
    print(" 📊 RESCUE RESULTS (Line-Level Logic)")
    print("=" * 50)
    print(f" Total Samples   : {len(samples)}")
    print(f" Baseline CER    : {avg_cer_a:.4f}")
    print(f" L2W1 CER        : {avg_cer_final:.4f}")
    print(f" Improvement     : {(avg_cer_a - avg_cer_final) * 100:.2f}%")
    print("=" * 50)

    # Save CSV
    import pandas as pd

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved results to {args.output_csv}")


if __name__ == "__main__":
    main()
