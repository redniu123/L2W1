#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载 HuggingFace 模型到本地，用于离线部署。

使用方法：
    # 下载 Agent B 模型
    python scripts/09_download_models.py --model Qwen/Qwen2-VL-2B-Instruct --output my_models

    # 下载 Router 模型（如果需要）
    python scripts/09_download_models.py --model Qwen/Qwen2.5-0.5B-Instruct --output my_models

    # 使用镜像站点下载（推荐）
    python scripts/09_download_models.py --model Qwen/Qwen2-VL-2B-Instruct --output my_models --mirror hf-mirror.com
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# 支持的镜像站点
MIRROR_SITES = {
    "hf-mirror.com": "https://hf-mirror.com",
    "hf-mirror": "https://hf-mirror.com",
    "openxlab": "https://code.openxlab.org.cn",
    "modelscope": "https://www.modelscope.cn",
}


def setup_mirror(mirror_name: str = None) -> None:
    """配置 HuggingFace 镜像站点。
    
    Args:
        mirror_name: 镜像站点名称，支持: hf-mirror.com, openxlab, modelscope
    """
    if mirror_name and mirror_name in MIRROR_SITES:
        mirror_url = MIRROR_SITES[mirror_name]
        os.environ["HF_ENDPOINT"] = mirror_url
        logger.info(f"✅ 已设置 HuggingFace 镜像: {mirror_url}")
    elif mirror_name:
        # 自定义镜像 URL
        os.environ["HF_ENDPOINT"] = mirror_name
        logger.info(f"✅ 已设置自定义镜像: {mirror_name}")
    else:
        # 默认使用 hf-mirror.com
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logger.info("✅ 已设置默认镜像: https://hf-mirror.com")


def download_model(model_name: str, output_dir: Path, use_mirror: bool = True) -> None:
    """下载 HuggingFace 模型到指定目录。
    
    Args:
        model_name: HuggingFace 模型标识符，如 "Qwen/Qwen2-VL-2B-Instruct"
        output_dir: 输出目录路径
        use_mirror: 是否使用镜像站点
    """
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
        from transformers import Qwen2VLForConditionalGeneration
    except ImportError:
        logger.error("❌ 请先安装 transformers: pip install transformers")
        sys.exit(1)
    
    # 设置镜像（如果需要）
    if use_mirror:
        setup_mirror()
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 模型保存路径
    model_dir = output_dir / model_name.replace("/", "_")
    
    logger.info(f"📥 开始下载模型: {model_name}")
    logger.info(f"📁 保存路径: {model_dir}")
    
    try:
        # 判断模型类型
        is_vlm = "VL" in model_name or "vision" in model_name.lower()
        
        if is_vlm:
            # VLM 模型需要下载 processor 和 model
            logger.info("检测到视觉语言模型，下载 Processor 和 Model...")
            
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=str(output_dir),
            )
            processor.save_pretrained(str(model_dir))
            logger.info("✅ Processor 下载完成")
            
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=str(output_dir),
                torch_dtype="auto",
            )
            model.save_pretrained(str(model_dir))
            logger.info("✅ Model 下载完成")
            
        else:
            # 普通语言模型
            logger.info("检测到语言模型，下载 Tokenizer 和 Model...")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=str(output_dir),
            )
            tokenizer.save_pretrained(str(model_dir))
            logger.info("✅ Tokenizer 下载完成")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=str(output_dir),
                torch_dtype="auto",
            )
            model.save_pretrained(str(model_dir))
            logger.info("✅ Model 下载完成")
        
        logger.info(f"\n🎉 模型下载完成！")
        logger.info(f"📦 模型路径: {model_dir}")
        logger.info(f"\n💡 使用方法:")
        logger.info(f"   1. 将整个 '{output_dir.name}' 文件夹上传到服务器")
        logger.info(f"   2. 在服务器上设置环境变量: export HF_HOME=/path/to/{output_dir.name}")
        logger.info(f"   3. 或者修改代码中的 model_path 为本地路径")
        
    except Exception as e:
        logger.error(f"❌ 下载失败: {e}")
        logger.error("\n💡 提示:")
        logger.error("   1. 检查网络连接")
        logger.error("   2. 尝试使用镜像站点: --mirror hf-mirror.com")
        logger.error("   3. 如果镜像也失败，可以手动从 https://huggingface.co 下载")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="下载 HuggingFace 模型到本地，用于离线部署"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="HuggingFace 模型标识符（默认: Qwen/Qwen2-VL-2B-Instruct）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="my_models",
        help="输出目录（默认: my_models）",
    )
    parser.add_argument(
        "--mirror",
        type=str,
        default="hf-mirror.com",
        help="镜像站点（默认: hf-mirror.com，可选: openxlab, modelscope）",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="不使用镜像，直接连接 HuggingFace（需要科学上网）",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output).resolve()
    
    download_model(
        model_name=args.model,
        output_dir=output_dir,
        use_mirror=not args.no_mirror,
    )


if __name__ == "__main__":
    main()

