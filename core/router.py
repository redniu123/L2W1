# -*- coding: utf-8 -*-
"""Router: The Gatekeeper - Entropy + PPL based routing.

Spec Reference: L2W1-MOD-003 (Router & PPL)

This module implements semantic perplexity calculation using a lightweight LLM
to determine whether OCR results need Agent B's visual-semantic correction.

The Router uses:
- Visual Entropy (from Agent A): Measures OCR confidence uncertainty
- Perplexity (PPL): Measures semantic coherence of the text sequence

High PPL = text is semantically incoherent = likely needs Agent B review.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class Router:
    """Router for entropy + PPL based decision making.
    
    The Gatekeeper in the L2W1 hierarchy:
    - Uses lightweight LLM (Qwen2.5-0.5B) to compute semantic perplexity
    - Combines with visual entropy from Agent A to route samples
    - High PPL + High Entropy = Send to Agent B
    
    Attributes:
        model: Loaded language model (AutoModelForCausalLM).
        tokenizer: Tokenizer for text encoding.
        device: Device where model runs (cuda/cpu).
    
    Example:
        >>> router = Router(model_name="Qwen/Qwen2.5-0.5B-Instruct")
        >>> ppl = router.compute_ppl("阿莫西林 0.5g")
        >>> print(f"PPL: {ppl:.2f}")
        >>> if ppl > 20.0:
        ...     print("High PPL - needs Agent B review")
        
        >>> # Advanced: Get token-level losses
        >>> token_losses = router.get_token_losses("阿莫西林 0.5g")
        >>> for token, loss in token_losses:
        ...     if loss > 5.0:
        ...         print(f"  ⚠️ '{token}' has high loss: {loss:.2f}")
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: Optional[str] = None,
    ) -> None:
        """Initialize Router with language model.
        
        Args:
            model_name: HuggingFace model identifier.
                Default: "Qwen/Qwen2.5-0.5B-Instruct" (lightweight, fast).
            device: Device to run model on. If None, uses device_map="auto".
                For memory efficiency, leave as None to use auto device mapping.
        
        Raises:
            RuntimeError: If model loading fails.
        """
        logger.info(f"Loading Router model: {model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with memory-efficient settings
            # Use torch_dtype="auto" and device_map="auto" to save VRAM
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": "auto",  # Automatically choose float16/float32
                "device_map": "auto",   # Automatically distribute across devices
            }
            
            # If device is explicitly specified and not CUDA, override device_map
            if device is not None and device != "cuda":
                if device == "cpu":
                    load_kwargs["device_map"] = None
                    load_kwargs["torch_dtype"] = torch.float32
                else:
                    load_kwargs["device_map"] = device
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs,
            )
            
            # Set to evaluation mode (no dropout, etc.)
            self.model.eval()
            
            # Determine actual device from model
            if hasattr(self.model, "device"):
                self.device = self.model.device
            elif hasattr(self.model, "hf_device_map"):
                # Multi-device model, use first device
                self.device = list(self.model.hf_device_map.values())[0]
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            logger.info(f"Router model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Router model: {e}")
            raise RuntimeError(f"Cannot initialize Router: {e}")
    
    @torch.no_grad()
    def compute_ppl(self, text: str) -> float:
        """Compute perplexity (PPL) for a text sequence.
        
        Implements L2W1-MOD-003 PPL formula:
        
        $$PPL(X) = \exp \left( - \frac{1}{t} \sum_{i=1}^{t} \log P_{\theta}(x_i | x_{<i}) \right)$$
        
        Implementation: Use model(input_ids, labels=input_ids) to get loss directly.
        
        Args:
            text: Input text string to evaluate.
                Example: "阿莫西林 0.5g"
        
        Returns:
            Perplexity value as float.
            - Low PPL (< 10): Text is semantically coherent
            - Medium PPL (10-50): Some uncertainty
            - High PPL (> 50): Likely semantic error, needs Agent B
        
        Note:
            Empty string returns high PPL (1e6) as undefined case.
        """
        # Handle empty string boundary case
        if not text or not text.strip():
            return 1e6
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        )
        
        input_ids = inputs["input_ids"]  # [1, seq_len]
        
        # Move to model device if needed
        if hasattr(self.model, "device"):
            input_ids = input_ids.to(self.model.device)
        else:
            input_ids = input_ids.to(self.device)
        
        # Execute forward pass with labels to get loss directly
        # model(input_ids, labels=input_ids) computes cross-entropy loss internally
        outputs = self.model(input_ids, labels=input_ids)
        loss = outputs.loss  # Mean cross-entropy loss
        
        # Convert loss to perplexity: PPL = exp(loss)
        ppl = torch.exp(loss).item()
        
        return ppl
    
    @torch.no_grad()
    def get_token_losses(
        self,
        text: str,
    ) -> List[Tuple[str, float]]:
        """Get per-token loss values for detailed analysis.
        
        Advanced method to identify which specific tokens contribute
        most to high perplexity. Useful for debugging and visualization.
        
        Args:
            text: Input text string.
        
        Returns:
            List of (token, loss) tuples.
            - token: Decoded token string (may be subword)
            - loss: Cross-entropy loss for predicting this token
            Example: [('阿', 0.1), ('莫', 0.2), ..., ('5', 9.8), ('g', 0.1)]
        
        Note:
            Empty string returns empty list.
            Tokens are decoded from model's tokenizer, which may split
            Chinese characters into subwords. Loss values are in log space.
        """
        # Handle empty string boundary case
        if not text or not text.strip():
            return []
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        )
        
        input_ids = inputs["input_ids"]  # [1, seq_len]
        
        # Move to model device if needed
        if hasattr(self.model, "device"):
            input_ids = input_ids.to(self.model.device)
        else:
            input_ids = input_ids.to(self.device)
        
        # Get model outputs (logits)
        outputs = self.model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]
        
        # Shift for next-token prediction alignment
        # Input sequence:  [x0, x1, x2, ..., x_t]
        # Logits predict: [P(x1|x0), P(x2|x0,x1), ..., P(x_{t+1}|x_0..x_t)]
        # We want to align: logits[i] predicts input_ids[i+1]
        shift_logits = logits[..., :-1, :].contiguous()  # [1, seq_len-1, vocab_size]
        shift_labels = input_ids[..., 1:].contiguous()  # [1, seq_len-1]
        
        # Compute per-token losses using CrossEntropyLoss(reduction='none')
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),  # [seq_len-1, vocab_size]
            shift_labels.view(-1),  # [seq_len-1]
        )  # [seq_len-1]
        
        # Decode tokens and pair with losses
        token_losses: List[Tuple[str, float]] = []
        
        # Get token IDs and loss values
        token_ids = shift_labels[0].cpu().tolist()  # [seq_len-1]
        loss_values = losses.cpu().tolist()  # [seq_len-1]
        
        # Align tokens with their losses
        for token_id, loss_val in zip(token_ids, loss_values):
            # Decode token (skip special tokens)
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=True)
            token_losses.append((token_str, loss_val))
        
        return token_losses
    
    def should_route_to_agent_b(
        self,
        text: str,
        visual_entropy: float,
        ppl_threshold: float = 20.0,
        entropy_threshold: float = 0.3,
    ) -> Tuple[bool, Dict[str, float]]:
        """Combined routing decision using both PPL and visual entropy.
        
        This is the main routing logic for L2W1:
        - High PPL (semantic incoherence) OR
        - High visual entropy (OCR uncertainty)
        -> Route to Agent B
        
        Args:
            text: OCR recognized text.
            visual_entropy: Visual entropy from Agent A (0-1 range).
            ppl_threshold: PPL threshold for routing (default 20.0).
            entropy_threshold: Entropy threshold for routing (default 0.3).
        
        Returns:
            Tuple of (should_route, metrics_dict):
            - should_route: True if should send to Agent B
            - metrics_dict: Contains ppl, visual_entropy, and decision factors
        """
        ppl = self.compute_ppl(text)
        
        # Decision logic: route if either metric exceeds threshold
        high_ppl = ppl > ppl_threshold
        high_entropy = visual_entropy > entropy_threshold
        
        should_route = high_ppl or high_entropy
        
        metrics = {
            "ppl": ppl,
            "visual_entropy": visual_entropy,
            "high_ppl": high_ppl,
            "high_entropy": high_entropy,
            "should_route": should_route,
        }
        
        return should_route, metrics
