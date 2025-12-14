#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pre-download models for offline use.

This script helps download required models before running in offline mode.
Run this script once when you have internet connection, then you can use
the models offline.

Usage:
    python scripts/00_download_models.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def download_router_model():
    """Download Router model (Qwen2.5-0.5B)."""
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    logger.info(f"Downloading Router model: {model_name}")
    logger.info("This may take a few minutes...")
    
    try:
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        logger.info("Tokenizer downloaded successfully")
        
        # Download model
        logger.info("Downloading model (this is the large file)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        logger.info("Model downloaded successfully")
        
        logger.info(f"Router model cached at: {tokenizer.cache_dir or 'default cache'}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download Router model: {e}")
        return False


def download_agent_b_model():
    """Download Agent B model (Qwen2-VL-2B)."""
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    
    logger.info(f"Downloading Agent B model: {model_name}")
    logger.info("This may take a few minutes...")
    
    try:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        
        # Download processor
        logger.info("Downloading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        logger.info("Processor downloaded successfully")
        
        # Download model
        logger.info("Downloading model (this is the large file)...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        logger.info("Model downloaded successfully")
        
        logger.info(f"Agent B model cached at: {processor.cache_dir or 'default cache'}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download Agent B model: {e}")
        return False


def main():
    """Main function."""
    print("=" * 70)
    print("L2W1 Model Downloader")
    print("=" * 70)
    print()
    print("This script will download required models for offline use.")
    print("Make sure you have internet connection!")
    print()
    
    import time
    
    # Download Router model
    print("-" * 70)
    print("Step 1: Downloading Router model (Qwen2.5-0.5B)")
    print("-" * 70)
    router_ok = download_router_model()
    print()
    
    time.sleep(2)
    
    # Download Agent B model (optional)
    print("-" * 70)
    print("Step 2: Downloading Agent B model (Qwen2-VL-2B) [Optional]")
    print("-" * 70)
    print("Press Ctrl+C to skip Agent B model download...")
    try:
        agent_b_ok = download_agent_b_model()
    except KeyboardInterrupt:
        logger.info("Skipping Agent B model download")
        agent_b_ok = False
    print()
    
    # Summary
    print("=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"Router model:  {'✓ Success' if router_ok else '✗ Failed'}")
    print(f"Agent B model: {'✓ Success' if agent_b_ok else '✗ Skipped/Failed'}")
    print()
    
    if router_ok:
        print("✓ Router model is ready for offline use!")
        print("  You can now run scripts with Router enabled.")
    else:
        print("✗ Router model download failed.")
        print("  Router will be disabled (PPL scores will be 0.0)")
    
    print()
    print("Note: Models are cached in HuggingFace cache directory.")
    print("      Default location: ~/.cache/huggingface/")
    print()


if __name__ == "__main__":
    main()

