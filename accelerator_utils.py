"""Accelerator utilities for BAGEL model optimization.

This module provides wrapper functions to enable various accelerators
(xFormers, Flash Attention, SageAttention, Triton) for the BAGEL model.
"""

import os
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check available accelerators
ACCELERATORS_AVAILABLE = {
    "xformers": False,
    "flash_attn": False,
    "sageattention": False,
    "triton": False,
}

# Try importing accelerators
try:
    import xformers
    import xformers.ops
    ACCELERATORS_AVAILABLE["xformers"] = True
    logger.info("✓ xFormers available")
except ImportError:
    logger.info("✗ xFormers not available")

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    ACCELERATORS_AVAILABLE["flash_attn"] = True
    logger.info("✓ Flash Attention available")
except ImportError:
    logger.info("✗ Flash Attention not available")

try:
    import sageattention
    from sageattention import sageattn
    ACCELERATORS_AVAILABLE["sageattention"] = True
    logger.info("✓ SageAttention available")
except ImportError:
    logger.info("✗ SageAttention not available")

try:
    import triton
    ACCELERATORS_AVAILABLE["triton"] = True
    logger.info("✓ Triton available")
except ImportError:
    logger.info("✗ Triton not available")


class AcceleratorConfig:
    """Configuration for accelerator selection and optimization."""

    def __init__(self,
                 use_xformers=True,
                 use_flash_attn=True,
                 use_sageattention=False,
                 use_triton=True,
                 attention_backend="auto"):
        """
        Args:
            use_xformers: Enable xFormers optimizations
            use_flash_attn: Enable Flash Attention
            use_sageattention: Enable SageAttention (experimental)
            use_triton: Enable Triton optimizations
            attention_backend: "auto", "flash", "xformers", "sage", or "native"
        """
        self.use_xformers = use_xformers and ACCELERATORS_AVAILABLE["xformers"]
        self.use_flash_attn = use_flash_attn and ACCELERATORS_AVAILABLE["flash_attn"]
        self.use_sageattention = use_sageattention and ACCELERATORS_AVAILABLE["sageattention"]
        self.use_triton = use_triton and ACCELERATORS_AVAILABLE["triton"]

        # Auto-select best available backend
        if attention_backend == "auto":
            if self.use_flash_attn:
                self.attention_backend = "flash"
            elif self.use_xformers:
                self.attention_backend = "xformers"
            elif self.use_sageattention:
                self.attention_backend = "sage"
            else:
                self.attention_backend = "native"
        else:
            self.attention_backend = attention_backend

        logger.info(f"Accelerator config: backend={self.attention_backend}, "
                   f"xformers={self.use_xformers}, flash={self.use_flash_attn}, "
                   f"sage={self.use_sageattention}, triton={self.use_triton}")


def patch_attention_forward(model, config: AcceleratorConfig):
    """Patch the model's attention mechanism with accelerated implementations."""

    # Import the specific attention implementations we'll use
    if config.attention_backend == "flash" and config.use_flash_attn:
        from flash_attn import flash_attn_varlen_func
        logger.info("Patching model with Flash Attention")
        _patch_flash_attention(model)

    elif config.attention_backend == "xformers" and config.use_xformers:
        import xformers.ops as xops
        logger.info("Patching model with xFormers attention")
        _patch_xformers_attention(model)

    elif config.attention_backend == "sage" and config.use_sageattention:
        from sageattention import sageattn
        logger.info("Patching model with SageAttention")
        _patch_sage_attention(model)

    else:
        logger.info("Using native PyTorch attention")

    return model


def _patch_flash_attention(model):
    """Replace attention layers with Flash Attention implementation."""
    from flash_attn import flash_attn_varlen_func

    # Find and patch Qwen2 attention layers
    for name, module in model.named_modules():
        if "Qwen2NaiveFlashAttention" in module.__class__.__name__:
            # Already using flash attention
            logger.debug(f"Module {name} already using Flash Attention")
            continue

        if "Qwen2Attention" in module.__class__.__name__:
            logger.debug(f"Patching {name} with Flash Attention")
            # The model already imports flash_attn, just ensure it's enabled
            if hasattr(module, 'use_flash_attention'):
                module.use_flash_attention = True


def _patch_xformers_attention(model):
    """Replace attention layers with xFormers implementation."""
    import xformers.ops as xops

    for name, module in model.named_modules():
        if "Attention" in module.__class__.__name__:
            original_forward = module.forward

            def xformers_forward(self, *args, **kwargs):
                # Extract Q, K, V from the standard attention input
                hidden_states = args[0]

                # Check if we can extract attention weights
                if hasattr(self, 'q_proj') and hasattr(self, 'k_proj') and hasattr(self, 'v_proj'):
                    batch_size, seq_len, _ = hidden_states.size()

                    # Project to Q, K, V
                    query = self.q_proj(hidden_states)
                    key = self.k_proj(hidden_states)
                    value = self.v_proj(hidden_states)

                    # Reshape for multi-head attention
                    if hasattr(self, 'num_heads'):
                        head_dim = query.size(-1) // self.num_heads
                        query = query.view(batch_size, seq_len, self.num_heads, head_dim)
                        key = key.view(batch_size, seq_len, self.num_heads, head_dim)
                        value = value.view(batch_size, seq_len, self.num_heads, head_dim)

                        # Use xFormers memory efficient attention
                        attn_output = xops.memory_efficient_attention(
                            query, key, value,
                            attn_bias=None,  # Can add attention mask here if needed
                            op=None  # Let xFormers choose the best kernel
                        )

                        # Reshape back
                        attn_output = attn_output.reshape(batch_size, seq_len, -1)

                        # Apply output projection if exists
                        if hasattr(self, 'o_proj'):
                            attn_output = self.o_proj(attn_output)

                        return (attn_output,) + args[1:] if len(args) > 1 else attn_output

                # Fallback to original implementation
                return original_forward(*args, **kwargs)

            module.forward = xformers_forward.__get__(module, module.__class__)
            logger.debug(f"Patched {name} with xFormers attention")


def _patch_sage_attention(model):
    """Replace attention layers with SageAttention implementation."""
    from sageattention import sageattn

    for name, module in model.named_modules():
        if "Attention" in module.__class__.__name__:
            original_forward = module.forward

            def sage_forward(self, *args, **kwargs):
                hidden_states = args[0]

                if hasattr(self, 'q_proj') and hasattr(self, 'k_proj') and hasattr(self, 'v_proj'):
                    batch_size, seq_len, _ = hidden_states.size()

                    # Project to Q, K, V
                    query = self.q_proj(hidden_states)
                    key = self.k_proj(hidden_states)
                    value = self.v_proj(hidden_states)

                    # Reshape for multi-head attention
                    if hasattr(self, 'num_heads'):
                        head_dim = query.size(-1) // self.num_heads
                        query = query.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
                        key = key.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
                        value = value.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)

                        # Use SageAttention
                        attn_output = sageattn(query, key, value, is_causal=False)

                        # Reshape back
                        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)

                        # Apply output projection
                        if hasattr(self, 'o_proj'):
                            attn_output = self.o_proj(attn_output)

                        return (attn_output,) + args[1:] if len(args) > 1 else attn_output

                # Fallback to original
                return original_forward(*args, **kwargs)

            module.forward = sage_forward.__get__(module, module.__class__)
            logger.debug(f"Patched {name} with SageAttention")


def optimize_model_with_accelerators(model, attention_backend="auto"):
    """
    Optimize BAGEL model with available accelerators.

    Args:
        model: The BAGEL model to optimize
        attention_backend: Which attention to use ("auto", "flash", "xformers", "sage", "native")

    Returns:
        Optimized model with accelerators applied
    """
    config = AcceleratorConfig(attention_backend=attention_backend)

    # Apply attention optimizations
    model = patch_attention_forward(model, config)

    # Enable Triton if available (for other optimizations)
    if config.use_triton:
        logger.info("Triton enabled for kernel optimizations")
        # Triton kernels are typically auto-selected by PyTorch/other libraries

    # Set model to use memory efficient settings
    if hasattr(model, 'gradient_checkpointing_enable'):
        try:
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for memory efficiency")
        except Exception as e:
            logger.debug(f"Gradient checkpointing not supported: {e}")

    return model


def benchmark_attention_backends(model, input_tensor, num_iterations=10):
    """Benchmark different attention backends."""
    import time

    results = {}
    backends = ["native", "flash", "xformers", "sage"]

    for backend in backends:
        if backend != "native" and not ACCELERATORS_AVAILABLE.get(backend.replace("flash", "flash_attn"), False):
            logger.info(f"Skipping {backend} (not available)")
            continue

        logger.info(f"Benchmarking {backend} attention...")

        # Apply the backend
        optimized_model = optimize_model_with_accelerators(model.clone() if hasattr(model, 'clone') else model,
                                                          attention_backend=backend)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = optimized_model(input_tensor)

        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = optimized_model(input_tensor)

        torch.cuda.synchronize()
        end_time = time.time()

        avg_time = (end_time - start_time) / num_iterations
        results[backend] = avg_time
        logger.info(f"{backend}: {avg_time:.4f}s per iteration")

    return results