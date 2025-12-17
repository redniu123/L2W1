#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch Inference: Line-Level Visual Correction (MVP Style).

RESCUE PLAN:
1. Agent A detects LINES (not chars).
2. If Line Entropy is high -> Agent B reads the WHOLE LINE crop.
3. Use CER for evaluation.
"""

import os
import sys
from pathlib import Path

# Add project root to path FIRST
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure Logging EARLY (before other imports that might use logger)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --- HuggingFace 模型路径和镜像配置 ---
# 自动检测项目根目录下的 my_models 文件夹
MY_MODELS_DIR = PROJECT_ROOT / "my_models"

# 如果 my_models 目录存在，设置 HF_HOME
if MY_MODELS_DIR.exists():
    os.environ["HF_HOME"] = str(MY_MODELS_DIR)
    logger.info(f"✅ 使用本地模型目录: {MY_MODELS_DIR}")
else:
    # 如果不存在，尝试使用默认的 HuggingFace cache 目录
    # 不设置 HF_HOME，让 HuggingFace 使用默认位置
    logger.info("⚠️  my_models 目录未找到，使用默认 HuggingFace 缓存目录")

# 配置镜像站点（如果未设置）
# 支持的镜像：hf-mirror.com（推荐）、openxlab、modelscope
if "HF_ENDPOINT" not in os.environ:
    # 默认使用 hf-mirror.com（国内访问友好）
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    logger.info("✅ 已设置 HuggingFace 镜像: https://hf-mirror.com")
    logger.info("💡 如需使用其他镜像，请设置环境变量: export HF_ENDPOINT=<镜像URL>")
else:
    logger.info(f"✅ 使用已配置的镜像: {os.environ['HF_ENDPOINT']}")

# 离线模式：默认禁用，允许在线下载模型
# 只有在明确设置环境变量时才启用离线模式
HF_OFFLINE = os.environ.get("HF_HUB_OFFLINE", "")
if HF_OFFLINE == "" or HF_OFFLINE == "0":
    # 确保未设置离线模式，允许在线下载
    os.environ.pop("HF_HUB_OFFLINE", None)
    logger.info("🌐 在线模式已启用（将从镜像站点下载模型）")
else:
    logger.info(f"📦 离线模式已启用: HF_HUB_OFFLINE={HF_OFFLINE}")

# Now import other modules
import argparse
import json
from typing import Dict

import cv2
import numpy as np
import Levenshtein
from tqdm import tqdm

from core.agent_a import AgentA
from core.agent_b import AgentB


def normalize_text(text: str) -> str:
    """Normalize text for CER calculation."""
    if not text:
        return ""
    # Remove whitespace and common punctuation noise
    return text.strip().replace(" ", "").replace("\n", "")


class L2W1Pipeline:
    def __init__(self, agent_b_model_path: str = None):
        """Initialize L2W1 Pipeline.

        Args:
            agent_b_model_path: Agent B 模型路径。
                - 如果为 None，使用默认 "openbmb/MiniCPM-V-4_5" (SOTA model)
                - 如果是本地路径，会自动检测并使用离线模式
                - 可以通过环境变量 HF_HOME 指定模型目录
        """
        logger.info("Initializing L2W1 Pipeline (Line-Level Mode)...")
        # 1. Agent A (The Scout) - Force Detection ON
        self.agent_a = AgentA(use_gpu=True)

        # 2. Agent B (The Judge)
        # 优先使用参数指定的路径，否则使用默认路径
        if agent_b_model_path is None:
            # 尝试从环境变量或本地目录查找模型
            if MY_MODELS_DIR.exists():
                # 查找 my_models 目录下的 MiniCPM-V 模型（优先）或 Qwen2-VL 模型（兼容）
                potential_paths = list(MY_MODELS_DIR.glob("*MiniCPM*V*"))
                if not potential_paths:
                    # Fallback: 查找 Qwen2-VL 模型（向后兼容）
                    potential_paths = list(MY_MODELS_DIR.glob("*Qwen*VL*"))
                if potential_paths:
                    agent_b_model_path = str(potential_paths[0])
                    logger.info(f"✅ 自动检测到本地模型: {agent_b_model_path}")
                else:
                    # [FIX] 使用 SOTA 模型 MiniCPM-V-4_5 作为默认值
                    agent_b_model_path = "openbmb/MiniCPM-V-4_5"
            else:
                # [FIX] 使用 SOTA 模型 MiniCPM-V-4_5 作为默认值
                agent_b_model_path = "openbmb/MiniCPM-V-4_5"

        self.agent_b = AgentB(model_path=agent_b_model_path, load_in_4bit=True)

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
                    # Call Agent B to correct the OCR prediction
                    # Agent B will use its V-CoT prompt internally
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
    parser.add_argument(
        "--agent_b_model",
        type=str,
        default=None,
        help="Agent B 模型路径（本地路径或 HuggingFace ID）。如果未指定，将自动检测 my_models 目录",
    )
    args = parser.parse_args()

    pipeline = L2W1Pipeline(agent_b_model_path=args.agent_b_model)

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
