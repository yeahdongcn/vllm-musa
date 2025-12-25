# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM MUSA Platform Plugin

This plugin enables vLLM to run on Moore Threads MUSA GPUs.
It provides a MUSAPlatform implementation that integrates with vLLM's
platform abstraction layer.

Usage:
    Install this package alongside vLLM, and the MUSA platform will be
    automatically detected and used when running on Moore Threads hardware.
"""

# Import torchada early to ensure torch.device patching happens before
# any torch.device("cuda:X") calls in vLLM
try:
    import torchada  # noqa: F401
except ImportError:
    pass

__all__ = ["MUSAPlatform", "musa_platform_plugin"]
__version__ = "0.1.0"


def _patch_torch_musa_compat() -> None:
    """
    Patch torch_musa to add missing compatibility methods.

    vLLM v0.13+ checks for torch.cuda._is_compiled() which doesn't exist
    in torch_musa. We add it here to ensure compatibility.
    """
    try:
        import torch_musa

        # Add _is_compiled if missing (returns True since torch_musa is compiled)
        if not hasattr(torch_musa, "_is_compiled"):
            torch_musa._is_compiled = lambda: True
    except ImportError:
        pass

    # Patch torch.backends.cuda.matmul.fp32_precision for PyTorch <2.9
    try:
        import torch

        matmul = torch.backends.cuda.matmul

        # Check if fp32_precision is already supported
        try:
            _ = matmul.fp32_precision
        except AttributeError:
            # Add fp32_precision property to the matmul module
            # This is a no-op shim for compatibility
            class MatmulShim:
                """Shim to add fp32_precision attribute to torch.backends.cuda.matmul."""

                def __init__(self, original):
                    self._original = original
                    self._fp32_precision = "highest"

                def __getattr__(self, name):
                    if name == "fp32_precision":
                        return self._fp32_precision
                    if name in ("_original", "_fp32_precision"):
                        return object.__getattribute__(self, name)
                    return getattr(self._original, name)

                def __setattr__(self, name, value):
                    if name == "fp32_precision":
                        object.__setattr__(self, "_fp32_precision", value)
                    elif name in ("_original", "_fp32_precision"):
                        object.__setattr__(self, name, value)
                    else:
                        setattr(self._original, name, value)

            torch.backends.cuda.matmul = MatmulShim(matmul)
    except Exception:
        pass


def _apply_vllm_patches() -> None:
    """Apply vLLM source patches for MUSA compatibility."""
    try:
        from .patches import apply_patches

        apply_patches()
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(f"Failed to apply vLLM patches: {e}")


def musa_platform_plugin() -> str | None:
    """
    vLLM platform plugin entry point for MUSA.

    This function is called by vLLM to check if the MUSA platform is available.
    Returns the qualified class name if MUSA is available, None otherwise.
    """
    import logging

    logger = logging.getLogger(__name__)
    logger.debug("Checking if MUSA platform is available.")

    try:
        # Check if torchada detects MUSA platform
        import torchada

        if torchada.is_musa_platform():
            # Apply compatibility patches
            _patch_torch_musa_compat()
            _apply_vllm_patches()
            logger.debug("Confirmed MUSA platform is available via torchada.")
            return "vllm_musa_platform.musa.MUSAPlatform"
    except ImportError:
        logger.debug("torchada not available, trying torch_musa directly.")

    try:
        # Fallback: check if torch_musa is available
        import torch_musa  # noqa: F401

        # Apply compatibility patches
        _patch_torch_musa_compat()
        _apply_vllm_patches()
        logger.debug("Confirmed MUSA platform is available via torch_musa.")
        return "vllm_musa_platform.musa.MUSAPlatform"
    except ImportError:
        pass

    logger.debug("MUSA platform is not available.")
    return None


# Lazy import of MUSAPlatform for direct usage
def __getattr__(name: str):
    if name == "MUSAPlatform":
        from .musa import MUSAPlatform

        return MUSAPlatform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
