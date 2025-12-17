# -*- coding: utf-8 -*-
"""Agent B: The Judge - VLM-based visual-semantic correction.

Spec Reference: L2W1-MOD-005 (Agent B & V-CoT)

This module implements the final defense line in L2W1 hierarchy:
- Uses VLM (Visual Language Model) for character-level correction
- Implements Visual Chain-of-Thought (V-CoT) prompting strategy
- Handles image preprocessing for Qwen2-VL format
- Cleans output to extract single corrected character

Agent B is "The Judge" that reviews uncertain OCR results flagged by Router.
"""

import logging
import re
from typing import Optional, Any, Dict, List

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer  # [FIX] 改用通用 AutoModel 以支持 MiniCPM

# Initialize logger FIRST (before using it in import blocks)
logger = logging.getLogger(__name__)


class AgentB:
    """VLM-based character correction agent.
    
    The Judge in the L2W1 hierarchy:
    - Receives character crops flagged by Router (high PPL/entropy)
    - Uses V-CoT prompting to force explicit reasoning
    - Outputs corrected single character
    
    Attributes:
        model: Loaded VLM model (MiniCPM-V-4_5).
        tokenizer: Tokenizer for text encoding.
        device: Device where model runs.
    
    Example:
        >>> agent_b = AgentB(model_path="openbmb/MiniCPM-V-4_5", load_in_4bit=True)
        >>> crop_image = Image.open("crop_336x336.jpg")
        >>> corrected = agent_b.inference(
        ...     crop_image=crop_image,
        ...     context_left="测试",
        ...     context_right="样本",
        ...     ocr_pred="5"
        ... )
        >>> print(f"Corrected: {corrected}")  # "莫"
    """
    
    # System prompt for V-CoT (L2W1-MOD-005)
    SYSTEM_PROMPT = """你是一个中文手写字识别专家。你需要纠正OCR的错误识别。
请按照以下格式输出：
【分析】观察字形结构、笔画细节，并结合上下文推断该字最可能是什么。
【结果】输出最终确定的单个汉字。如果无法确定，输出 <UNKNOWN>。"""
    
    def __init__(
        self,
        model_path: str = "openbmb/MiniCPM-V-4_5",  # [FIX] 切换模型
        load_in_4bit: bool = True,
    ) -> None:
        """Initialize Agent B with VLM model.
        
        Args:
            model_path: HuggingFace model identifier or local path.
                - HuggingFace ID: "openbmb/MiniCPM-V-4_5" (SOTA model)
                - Local path: "/path/to/my_models/MiniCPM-V-4_5"
                Default: "openbmb/MiniCPM-V-4_5" (SOTA model per spec).
            load_in_4bit: Whether to use 4-bit quantization via bitsandbytes.
                Saves VRAM significantly.
        
        Raises:
            RuntimeError: If model loading fails.
            ImportError: If bitsandbytes not available when load_in_4bit=True.
        """
        logger.info(f"Loading Agent B (SOTA): {model_path}")
        
        # 检测是否为本地路径
        from pathlib import Path
        model_path_obj = Path(model_path)
        is_local_path = model_path_obj.exists() and model_path_obj.is_dir()
        
        if is_local_path:
            logger.info(f"✅ 检测到本地模型路径: {model_path}")
            logger.info("📦 使用离线模式加载（local_files_only=True）")
            local_files_only = True
        else:
            logger.info(f"🌐 使用在线模式加载（将从 HuggingFace 或镜像站点下载）")
            local_files_only = False
        
        try:
            # [FIX] 1. 加载 Tokenizer (MiniCPM 使用 AutoTokenizer)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            logger.info("✅ Tokenizer loaded")
            
            # [FIX] 2. 加载模型 (使用 AutoModel 以支持 MiniCPM)
            # [FIX] 解决 BFloat16/Byte 冲突：改用 float16（兼容性更好）
            dtype = torch.float16  # 使用 float16 避免与 bitsandbytes 的 Byte 类型冲突
            
            load_kwargs = {
                "trust_remote_code": True,
                "attn_implementation": "sdpa",  # Flash Attention 2 (faster)
                "torch_dtype": dtype,
                "device_map": "auto",
                "local_files_only": local_files_only,
            }
            
            if load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=dtype,  # 确保计算类型匹配（float16）
                        bnb_4bit_quant_type="nf4",  # 使用 NF4 量化类型（标准配置）
                    )
                    load_kwargs["quantization_config"] = quantization_config
                    logger.info("Using 4-bit quantization to save VRAM (float16 compute dtype)")
                except ImportError:
                    logger.warning(
                        "bitsandbytes not available. Install with: pip install bitsandbytes"
                    )
                    logger.warning("Falling back to full precision (may use more VRAM)")
            
            self.model = AutoModel.from_pretrained(
                model_path,
                **load_kwargs,
            )
            
            self.model.eval()  # Set to evaluation mode
            
            # Determine device
            if hasattr(self.model, "device"):
                self.device = self.model.device
            elif hasattr(self.model, "hf_device_map"):
                self.device = list(self.model.hf_device_map.values())[0]
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            logger.info(f"Agent B (MiniCPM-V-4_5) loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Agent B: {e}")
            raise RuntimeError(f"Cannot initialize Agent B: {e}")
    
    def _build_vcot_prompt(
        self,
        context_left: str,
        context_right: str,
        ocr_pred: str,
    ) -> str:
        """Build Visual Chain-of-Thought (V-CoT) prompt.
        
        Implements L2W1-MOD-005 V-CoT Prompt Template:
        - Forces explicit reasoning steps with 【分析】and 【结果】format
        - Embeds context and OCR prediction
        - Enables explicit reasoning chain
        
        Args:
            context_left: Left context text.
            context_right: Right context text.
            ocr_pred: OCR predicted character.
        
        Returns:
            Formatted user prompt string.
        """
        # L2W1-MOD-005 User Prompt Template with explicit reasoning
        prompt = f"""上下文：{context_left}[?]{context_right}
OCR 认为这是：'{ocr_pred}'

请按照系统指令的格式输出：
【分析】观察字形结构、笔画细节，并结合上下文 "{context_left}[?]{context_right}" 推断该字最可能是什么。
【结果】输出最终确定的单个汉字。如果无法确定，输出 <UNKNOWN>。"""
        
        return prompt
    
    def _preprocess_image(
        self,
        crop_image: Image.Image,
        target_size: int = 336,
    ) -> Image.Image:
        """Preprocess image for MiniCPM-V with keep-ratio padding.
        
        Implements L2W1-DE-002: Keep-Ratio Resize + Center Padding.
        This prevents character distortion by maintaining aspect ratio.
        
        Args:
            crop_image: Input PIL Image (can be any size).
            target_size: Target canvas size (default: 336).
        
        Returns:
            Preprocessed PIL Image in RGB format, size (target_size, target_size).
        """
        # Ensure RGB format
        if crop_image.mode != "RGB":
            crop_image = crop_image.convert("RGB")
        
        # If already target size, return as-is
        if crop_image.size == (target_size, target_size):
            return crop_image
        
        # Get original dimensions
        orig_w, orig_h = crop_image.size
        
        # Calculate scale factor to fit within target_size while keeping aspect ratio
        # Scale based on the longer dimension
        scale = target_size / max(orig_w, orig_h)
        
        # Calculate new dimensions (maintaining aspect ratio)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Resize image maintaining aspect ratio
        resized_image = crop_image.resize(
            (new_w, new_h),
            Image.Resampling.LANCZOS,
        )
        
        # Create black background canvas (336x336)
        canvas = Image.new("RGB", (target_size, target_size), color=(0, 0, 0))
        
        # Calculate paste position (center the resized image)
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        
        # Paste resized image onto canvas (centered)
        canvas.paste(resized_image, (paste_x, paste_y))
        
        logger.debug(
            f"Preprocessed image: {orig_w}x{orig_h} -> {new_w}x{new_h} "
            f"(scale={scale:.3f}) -> {target_size}x{target_size} canvas "
            f"(pasted at {paste_x}, {paste_y})"
        )
        
        return canvas
    
    def _debug_log_messages(self, messages: List[Dict[str, Any]], label: str = "") -> None:
        """Debug log messages structure without printing huge image bytes.
        
        Args:
            messages: Messages list to debug.
            label: Label for the debug output.
        """
        logger.debug(f"[DEBUG] {label} Messages structure:")
        if messages is None:
            logger.debug(f"  [!] messages is None!")
            return
            
        for i, msg in enumerate(messages):
            if msg is None:
                logger.debug(f"  [{i}] Message is None!")
                continue
            role = msg.get("role", "UNKNOWN")
            content = msg.get("content", [])
            logger.debug(f"  [{i}] role={role}, content type={type(content).__name__}")
            
            if isinstance(content, list):
                for j, item in enumerate(content):
                    if item is None:
                        logger.debug(f"      [{j}] Content item is None!")
                        continue
                    item_type = item.get("type", "UNKNOWN")
                    if item_type == "image":
                        img = item.get("image")
                        if img is None:
                            logger.debug(f"      [{j}] type=image, image=None (!)")
                        elif isinstance(img, Image.Image):
                            logger.debug(f"      [{j}] type=image, PIL.Image size={img.size}, mode={img.mode}")
                        else:
                            logger.debug(f"      [{j}] type=image, image_type={type(img).__name__}")
                    elif item_type == "text":
                        text = item.get("text", "")
                        text_preview = text[:50] + "..." if len(text) > 50 else text
                        logger.debug(f"      [{j}] type=text, len={len(text)}, preview='{text_preview}'")
                    else:
                        logger.debug(f"      [{j}] type={item_type}")
            elif isinstance(content, str):
                logger.debug(f"      content (str): '{content[:50]}...'")

    @torch.no_grad()
    def inference(
        self,
        crop_image: Image.Image,
        context_left: str,
        context_right: str,
        ocr_pred: str,
    ) -> str:
        """Run VLM inference to correct OCR prediction using MiniCPM-V chat interface.
        
        L2W1-MOD-005: Uses V-CoT prompting to force explicit reasoning.
        
        [FIX] Uses MiniCPM-V native `.chat()` interface for stable inference.
        
        Args:
            crop_image: Character crop image (PIL Image, will be preprocessed).
            context_left: Left context text.
            context_right: Right context text.
            ocr_pred: OCR predicted character to correct.
        
        Returns:
            Corrected character string (single character).
            Returns ocr_pred if model outputs <UNKNOWN> or fails.
        """
        # ========================================================================
        # Step 0: Input Validation
        # ========================================================================
        logger.debug(f"[AgentB.inference] Starting inference for ocr_pred='{ocr_pred}'")
        
        if crop_image is None:
            logger.error("[AgentB.inference] crop_image is None! Cannot proceed.")
            return ocr_pred
        
        if not isinstance(crop_image, Image.Image):
            logger.error(f"[AgentB.inference] crop_image is not PIL.Image, got: {type(crop_image)}")
            return ocr_pred
        
        logger.debug(f"[AgentB.inference] Input image: size={crop_image.size}, mode={crop_image.mode}")
        logger.debug(f"[AgentB.inference] Context: left='{context_left}', right='{context_right}'")
        
        # Validate context strings
        context_left = context_left if context_left is not None else ""
        context_right = context_right if context_right is not None else ""
        ocr_pred = ocr_pred if ocr_pred is not None else ""
        
        try:
            # ====================================================================
            # Step 1: Preprocess Image
            # ====================================================================
            processed_image = self._preprocess_image(crop_image)
            if processed_image is None:
                logger.error("[AgentB.inference] _preprocess_image returned None!")
                return ocr_pred
            
            logger.debug(f"[AgentB.inference] Preprocessed image: size={processed_image.size}")
            
            # ====================================================================
            # Step 2: Build V-CoT Prompt
            # ====================================================================
            user_prompt = self._build_vcot_prompt(context_left, context_right, ocr_pred)
            logger.debug(f"[AgentB.inference] User prompt length: {len(user_prompt)}")
            
            # ====================================================================
            # Step 3: Prepare Messages for MiniCPM-V chat interface
            # ====================================================================
            # MiniCPM-V uses simple message format: [{'role': 'user', 'content': prompt}]
            # The image is passed separately to .chat() method
            msgs = [{'role': 'user', 'content': user_prompt}]
            
            # ====================================================================
            # Step 4: Call MiniCPM-V .chat() interface
            # ====================================================================
            try:
                logger.debug("[AgentB.inference] Calling MiniCPM-V .chat() interface...")
                
                # [FIX] 使用 MiniCPM 原生 chat 接口
                res = self.model.chat(
                    image=processed_image,  # 传入 PIL Image
                    msgs=msgs,
                    tokenizer=self.tokenizer,
                    sampling=False,  # 贪婪解码保证稳定性
                    max_new_tokens=128,
                )
                
                logger.debug(f"[AgentB.inference] MiniCPM-V chat returned: '{res}'")
                
            except Exception as e:
                logger.error(f"[AgentB.inference] MiniCPM-V chat failed: {e}")
                import traceback
                logger.error(f"[AgentB.inference] Full traceback:\n{traceback.format_exc()}")
                return ocr_pred
            
            # ====================================================================
            # Step 5: Clean Output
            # ====================================================================
            corrected_char = self._clean_output(res, ocr_pred)
            logger.info(f"[AgentB.inference] Result: '{ocr_pred}' -> '{corrected_char}'")
            
            return corrected_char
            
        except Exception as e:
            # 捕获所有未预期的异常
            logger.error(f"[AgentB.inference] Unexpected error: {e}")
            import traceback
            logger.error(f"[AgentB.inference] Full traceback:\n{traceback.format_exc()}")
            return ocr_pred  # 永远返回一个值，不让脚本崩溃
    
    def _clean_output(
        self,
        raw_output: str,
        ocr_pred: str,
    ) -> str:
        """Clean VLM output to extract single corrected character.
        
        L2W1-MOD-005: Extract character from 【结果】tag or fallback to last Chinese char.
        Handles cases like:
        - "【分析】...【结果】莫" -> return "莫"
        - "【结果】莫" -> return "莫"
        - "【结果】<UNKNOWN>" -> return ocr_pred
        - "莫" (no tags) -> return "莫" (fallback)
        
        Args:
            raw_output: Raw model output string.
            ocr_pred: Original OCR prediction (fallback).
        
        Returns:
            Single corrected character, or ocr_pred if cleaning fails.
        """
        if not raw_output:
            return ocr_pred
        
        # Normalize whitespace
        raw_output = raw_output.strip()
        
        # Strategy 1: Extract from 【结果】tag using regex
        # Pattern: 【结果】followed by optional whitespace, then capture the first non-whitespace character
        result_pattern = r"【结果】\s*(\S)"
        result_match = re.search(result_pattern, raw_output)
        
        if result_match:
            result_char = result_match.group(1)
            # Check if it's <UNKNOWN>
            if result_char == "<" or "UNKNOWN" in raw_output.upper():
                logger.debug(f"Model output <UNKNOWN>, using original prediction: '{ocr_pred}'")
                return ocr_pred
            # Check if it's a Chinese character
            chinese_char_pattern = r'[\u4e00-\u9fff]'
            if re.match(chinese_char_pattern, result_char):
                logger.debug(f"Extracted character from 【结果】tag: '{result_char}'")
                return result_char
            else:
                logger.warning(
                    f"【结果】tag found but character '{result_char}' is not Chinese. "
                    f"Falling back to last Chinese character search."
                )
        
        # Strategy 2: Fallback - extract last Chinese character in output
        # This handles cases where model doesn't follow the format exactly
        chinese_char_pattern = r'[\u4e00-\u9fff]'
        matches = re.findall(chinese_char_pattern, raw_output)
        
        if matches:
            last_char = matches[-1]  # Use last character (most likely the final answer)
            logger.debug(
                f"No 【结果】tag found, extracted last Chinese character: '{last_char}'"
            )
            return last_char
        
        # Strategy 3: Final fallback - return original prediction
        logger.warning(
            f"Failed to extract Chinese character from output: '{raw_output}'. "
            f"Using original OCR prediction: '{ocr_pred}'"
        )
        return ocr_pred
    
    def batch_inference(
        self,
        crop_images: list[Image.Image],
        contexts: list[tuple[str, str]],  # List of (left, right) tuples
        ocr_preds: list[str],
    ) -> list[str]:
        """Run batch inference for multiple character crops.
        
        Args:
            crop_images: List of character crop images.
            contexts: List of (context_left, context_right) tuples.
            ocr_preds: List of OCR predicted characters.
        
        Returns:
            List of corrected characters.
        """
        if len(crop_images) != len(contexts) or len(crop_images) != len(ocr_preds):
            raise ValueError(
                f"Mismatched lengths: images={len(crop_images)}, "
                f"contexts={len(contexts)}, ocr_preds={len(ocr_preds)}"
            )
        
        results = []
        for crop_image, (left, right), ocr_pred in zip(crop_images, contexts, ocr_preds):
            try:
                corrected = self.inference(crop_image, left, right, ocr_pred)
                results.append(corrected)
            except Exception as e:
                logger.error(f"Batch inference failed for '{ocr_pred}': {e}")
                results.append(ocr_pred)  # Fallback to original
        
        return results

