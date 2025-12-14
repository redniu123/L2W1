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
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

logger = logging.getLogger(__name__)


class AgentB:
    """VLM-based character correction agent.
    
    The Judge in the L2W1 hierarchy:
    - Receives character crops flagged by Router (high PPL/entropy)
    - Uses V-CoT prompting to force explicit reasoning
    - Outputs corrected single character
    
    Attributes:
        model: Loaded VLM model (Qwen2VLForConditionalGeneration).
        processor: Image and text processor for Qwen2-VL.
        device: Device where model runs.
    
    Example:
        >>> agent_b = AgentB(model_path="Qwen/Qwen2-VL-2B-Instruct", load_in_4bit=True)
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
    SYSTEM_PROMPT = "你是一个中文古籍/手写体识别专家。你的任务是根据图像和上下文，纠正 OCR 的错误识别。"
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2-VL-2B-Instruct",
        load_in_4bit: bool = True,
    ) -> None:
        """Initialize Agent B with VLM model.
        
        Args:
            model_path: HuggingFace model identifier.
                Default: "Qwen/Qwen2-VL-2B-Instruct" (lightweight, fast).
            load_in_4bit: Whether to use 4-bit quantization via bitsandbytes.
                Saves VRAM significantly (recommended for 2B model).
        
        Raises:
            RuntimeError: If model loading fails.
            ImportError: If bitsandbytes not available when load_in_4bit=True.
        """
        logger.info(f"Loading Agent B (VLM): {model_path}")
        
        try:
            # Load processor (handles image and text preprocessing)
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            
            # Load model with optional 4-bit quantization
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": "auto",
                "device_map": "auto",
            }
            
            if load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    load_kwargs["quantization_config"] = quantization_config
                    logger.info("Using 4-bit quantization to save VRAM")
                except ImportError:
                    logger.warning(
                        "bitsandbytes not available. Install with: pip install bitsandbytes"
                    )
                    logger.warning("Falling back to full precision (may use more VRAM)")
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
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
            
            logger.info(f"Agent B loaded successfully on {self.device}")
            
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
        - Forces explicit reasoning steps
        - Embeds context and OCR prediction
        - Instructs model to output only final character
        
        Args:
            context_left: Left context text.
            context_right: Right context text.
            ocr_pred: OCR predicted character.
        
        Returns:
            Formatted user prompt string.
        """
        # L2W1-MOD-005 User Prompt Template
        prompt = f"""上下文：[{context_left}] <目标字符> [{context_right}]
OCR 认为这是：'{ocr_pred}'

请执行以下步骤：
1. 观察图像中的笔画结构和部首。
2. 结合上下文判断语义是否通顺。
3. 如果 OCR 正确，输出 "KEEP"；如果错误，输出正确的单个汉字。

不要输出任何解释，只输出最终字符。"""
        
        return prompt
    
    def _preprocess_image(
        self,
        crop_image: Image.Image,
    ) -> Image.Image:
        """Preprocess image for Qwen2-VL.
        
        Ensures image is in correct format (RGB, 336x336).
        
        Args:
            crop_image: Input PIL Image (should be 336x336 from DataProcessor).
        
        Returns:
            Preprocessed PIL Image in RGB format.
        """
        # Ensure RGB format
        if crop_image.mode != "RGB":
            crop_image = crop_image.convert("RGB")
        
        # Resize to 336x336 if needed (should already be from DataProcessor)
        if crop_image.size != (336, 336):
            crop_image = crop_image.resize((336, 336), Image.Resampling.LANCZOS)
        
        return crop_image
    
    @torch.no_grad()
    def inference(
        self,
        crop_image: Image.Image,
        context_left: str,
        context_right: str,
        ocr_pred: str,
    ) -> str:
        """Run VLM inference to correct OCR prediction.
        
        L2W1-MOD-005: Uses V-CoT prompting to force explicit reasoning.
        
        Args:
            crop_image: Character crop image (336x336 PIL Image).
            context_left: Left context text.
            context_right: Right context text.
            ocr_pred: OCR predicted character to correct.
        
        Returns:
            Corrected character string (single character).
            Returns ocr_pred if model outputs "KEEP" or fails.
        """
        # Preprocess image
        processed_image = self._preprocess_image(crop_image)
        
        # Build V-CoT prompt
        user_prompt = self._build_vcot_prompt(context_left, context_right, ocr_pred)
        
        # Prepare messages for Qwen2-VL format
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.SYSTEM_PROMPT},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": processed_image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        
        # Process inputs using Qwen2-VL processor
        # The processor handles image preprocessing and text tokenization
        try:
            # Try Qwen2-VL specific API first
            if hasattr(self.processor, "process_vision_info"):
                image_inputs, video_inputs = self.processor.process_vision_info(messages)
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                # Fallback: standard processor API
                inputs = self.processor(
                    text=user_prompt,
                    images=processed_image,
                    padding=True,
                    return_tensors="pt",
                )
        except Exception as e:
            logger.warning(f"Qwen2-VL specific API failed: {e}. Trying standard API...")
            # Fallback to standard processor
            inputs = self.processor(
                text=user_prompt,
                images=processed_image,
                padding=True,
                return_tensors="pt",
            )
        
        # Move inputs to model device
        inputs = inputs.to(self.device)
        
        # Generate with constraint decoding (L2W1-MOD-005: max_new_tokens=10)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=10,  # Prevent verbose output
            do_sample=False,  # Deterministic
        )
        
        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        # Clean output to extract single character
        corrected_char = self._clean_output(output_text, ocr_pred)
        
        return corrected_char
    
    def _clean_output(
        self,
        raw_output: str,
        ocr_pred: str,
    ) -> str:
        """Clean VLM output to extract single corrected character.
        
        L2W1-MOD-005: Extract first Chinese character from output.
        Handles cases like:
        - "KEEP" -> return ocr_pred
        - "我认为是：莫" -> return "莫"
        - "莫" -> return "莫"
        - "莫，因为..." -> return "莫"
        
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
        
        # Check for "KEEP" (case-insensitive)
        if raw_output.upper() == "KEEP" or "KEEP" in raw_output.upper():
            return ocr_pred
        
        # Extract first Chinese character using regex
        # Pattern matches any single Chinese character (CJK Unified Ideographs)
        chinese_char_pattern = r'[\u4e00-\u9fff]'
        matches = re.findall(chinese_char_pattern, raw_output)
        
        if matches:
            return matches[0]  # Return first Chinese character
        
        # Fallback: if no Chinese character found, return original prediction
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

