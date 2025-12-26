# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MUSA (Meta-computing Unified System Architecture) Platform implementation
for vLLM.

Code inside this file can safely assume MUSA platform, e.g. importing
pymtml. However, it should not initialize MUSA context.

This module provides the MUSAPlatform class that enables vLLM to run on
Moore Threads MUSA GPUs. It uses torchada for CUDA→MUSA compatibility
and pymtml (pynvml-compatible) for device management.
"""

import os
from collections.abc import Callable
from functools import cache, wraps
from typing import TYPE_CHECKING, Optional, TypeVar

import torch
from typing_extensions import ParamSpec
from vllm.logger import init_logger

try:
    import pymtml as pynvml  # MUSA equivalent of pynvml
except ImportError:
    pynvml = None  # type: ignore

__all__ = [
    # Main platform class (auto-selected based on mtml availability)
    "MUSAPlatform",
    # Platform implementations
    "MUSAPlatformBase",
    "MtmlMUSAPlatform",
    "NonMtmlMUSAPlatform",
    # Utilities
    "with_nvml_context",
    "mtml_available",
]

if TYPE_CHECKING:
    from vllm.attention.selector import AttentionSelectorConfig
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _import_torchada():
    """Import torchada and apply patches for CUDA→MUSA compatibility."""
    try:
        import torchada

        return torchada
    except ImportError:
        logger.warning("torchada not found. MUSA platform may not work correctly.")
        return None


# Import torchada early to apply patches
torchada = _import_torchada()


def _apply_musa_patches():
    """Apply MUSA compatibility patches to vLLM.

    See vllm_musa_platform/patches/README.md for details.
    """
    try:
        from .patches import apply_patches

        apply_patches()
    except Exception as e:
        logger.warning(f"Failed to apply MUSA patches: {e}")


# Apply patches early
_apply_musa_patches()


_musa_cache_ops_registered = False


def _reshape_and_cache_flash_pytorch(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Pure PyTorch implementation of reshape_and_cache_flash.

    This function reshapes and caches key/value tensors into the paged KV cache.

    Cache layout for flash attention: [num_blocks, block_size, num_kv_heads, head_size]

    Args:
        key: [num_tokens, num_kv_heads, head_size]
        value: [num_tokens, num_kv_heads, head_size]
        key_cache: [num_blocks, block_size, num_kv_heads, head_size]
        value_cache: [num_blocks, block_size, num_kv_heads, head_size]
        slot_mapping: [num_tokens] - maps each token to a slot in the cache
        kv_cache_dtype: dtype string for the cache (e.g., "auto", "fp8")
        k_scale: scale for key quantization (unused for now)
        v_scale: scale for value quantization (unused for now)
    """
    num_tokens = key.shape[0]
    block_size = key_cache.shape[1]

    # Flatten slot_mapping if needed
    slot_mapping_flat = slot_mapping.flatten()

    # Calculate block indices and offsets for all tokens at once
    block_indices = slot_mapping_flat // block_size
    block_offsets = slot_mapping_flat % block_size

    # Create valid mask for non-negative slots
    valid_mask = slot_mapping_flat >= 0

    # Use scatter to update the cache
    # key_cache and value_cache are [num_blocks, block_size, num_kv_heads, head_size]
    # We need to scatter key/value [num_tokens, num_kv_heads, head_size] into the cache

    for i in range(num_tokens):
        if valid_mask[i]:
            block_idx = block_indices[i]
            block_offset = block_offsets[i]
            key_cache[block_idx, block_offset] = key[i]
            value_cache[block_idx, block_offset] = value[i]


def _reshape_and_cache_pytorch(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Pure PyTorch implementation of reshape_and_cache.

    This function reshapes and caches key/value tensors into the paged KV cache.

    Cache layout for paged attention:
        key_cache: [num_blocks, num_kv_heads, head_size // x, block_size, x]
        value_cache: [num_blocks, num_kv_heads, head_size, block_size]

    Args:
        key: [num_tokens, num_kv_heads, head_size]
        value: [num_tokens, num_kv_heads, head_size]
        key_cache: [num_blocks, num_kv_heads, head_size // x, block_size, x]
        value_cache: [num_blocks, num_kv_heads, head_size, block_size]
        slot_mapping: [num_tokens] - maps each token to a slot in the cache
        kv_cache_dtype: dtype string for the cache
        k_scale: scale for key quantization (unused for now)
        v_scale: scale for value quantization (unused for now)
    """
    num_tokens = key.shape[0]
    num_kv_heads = key.shape[1]
    head_size = key.shape[2]
    block_size = value_cache.shape[3]  # value_cache layout gives us block_size
    x = key_cache.shape[4]  # x from key_cache layout

    slot_mapping_flat = slot_mapping.flatten()
    block_indices = slot_mapping_flat // block_size
    block_offsets = slot_mapping_flat % block_size

    for i in range(num_tokens):
        slot = slot_mapping_flat[i]
        if slot >= 0:
            block_idx = block_indices[i]
            block_offset = block_offsets[i]
            # For key: reshape from [num_kv_heads, head_size]
            # to [num_kv_heads, head_size // x, x]
            # key_cache layout: [num_blocks, num_kv_heads, head_size // x, block_size, x]
            key_reshaped = key[i].view(num_kv_heads, head_size // x, x)
            key_cache[block_idx, :, :, block_offset, :] = key_reshaped
            # For value: direct copy
            # value_cache layout: [num_blocks, num_kv_heads, head_size, block_size]
            value_cache[block_idx, :, :, block_offset] = value[i]


def _register_musa_cache_ops():
    """Register MUSA-compatible cache operations using PyTorch.

    This replaces the CUDA C++ ops with PyTorch implementations that work on MUSA.
    Must be called after vLLM platform initialization is complete.
    """
    global _musa_cache_ops_registered
    if _musa_cache_ops_registered:
        return True

    try:
        # Define the custom ops using torch.library

        # Register reshape_and_cache_flash
        @torch.library.custom_op(
            "_C_cache_ops::reshape_and_cache_flash",
            mutates_args=("key_cache", "value_cache"),
        )
        def reshape_and_cache_flash(
            key: torch.Tensor,
            value: torch.Tensor,
            key_cache: torch.Tensor,
            value_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
            kv_cache_dtype: str,
            k_scale: torch.Tensor,
            v_scale: torch.Tensor,
        ) -> None:
            """Reshape and cache key/value tensors using PyTorch."""
            _reshape_and_cache_flash_pytorch(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
                v_scale,
            )

        # Register reshape_and_cache
        @torch.library.custom_op(
            "_C_cache_ops::reshape_and_cache", mutates_args=("key_cache", "value_cache")
        )
        def reshape_and_cache(
            key: torch.Tensor,
            value: torch.Tensor,
            key_cache: torch.Tensor,
            value_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
            kv_cache_dtype: str,
            k_scale: torch.Tensor,
            v_scale: torch.Tensor,
        ) -> None:
            """Reshape and cache key/value tensors using PyTorch."""
            _reshape_and_cache_pytorch(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
                v_scale,
            )

        logger.info("Registered MUSA-compatible cache ops (PyTorch implementation)")
        _musa_cache_ops_registered = True
        return True
    except Exception as e:
        logger.warning(f"Failed to register MUSA cache ops: {e}")
        return False


def with_nvml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        pynvml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            pynvml.nvmlShutdown()

    return wrapper


# Import vLLM platform interface
try:
    from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum
except ImportError:
    # Fallback for testing without full vLLM installation
    from enum import Enum, auto
    from typing import NamedTuple

    class PlatformEnum(Enum):
        CUDA = auto()
        ROCM = auto()
        TPU = auto()
        XPU = auto()
        CPU = auto()
        OOT = auto()
        UNSPECIFIED = auto()

    class DeviceCapability(NamedTuple):
        major: int
        minor: int

        def to_int(self) -> int:
            return self.major * 10 + self.minor

        def as_version_str(self) -> str:
            return f"{self.major}.{self.minor}"

    class Platform:
        pass


class MUSAPlatformBase(Platform):
    """Base MUSA platform implementation."""

    _enum = PlatformEnum.OOT  # Out-of-tree platform
    device_name: str = "musa"
    device_type: str = "musa"  # MUSA device type (torchada handles compatibility)
    dispatch_key: str = "CUDA"  # Use CUDA dispatch key via torchada
    ray_device_key: str = "GPU"
    dist_backend: str = "mccl"  # MUSA's NCCL equivalent
    device_control_env_var: str = "MUSA_VISIBLE_DEVICES"

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        """Returns the supported dtypes for MUSA platform."""
        # MUSA GPUs support BF16 and FP16
        return [torch.bfloat16, torch.float16, torch.float32]

    def is_cuda_alike(self) -> bool:
        """
        MUSA is NOT CUDA-alike for custom ops purposes.

        This returns False because vLLM uses is_cuda_alike() to determine
        whether to use CUDA custom ops (torch.ops._C.*), which are not
        available on MUSA. By returning False, vLLM will use native PyTorch
        implementations instead.

        Note: For device initialization in gpu_worker.py, we patch the
        device type check to include "musa" separately.
        """
        return False

    def is_sleep_mode_available(self) -> bool:
        """MUSA does not currently support sleep mode."""
        return False

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        """
        Check if the current platform supports async output.

        MUSA supports async output processing when CUDA graphs are enabled
        (i.e., when enforce_eager is False or None).
        """
        if enforce_eager:
            logger.warning(
                "To see benefits of async output processing, enable CUDA "
                "graph. Since enforce-eager is enabled, async output "
                "processor cannot be used"
            )
            return False
        return True

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """Set the device for the current platform."""
        # Extract device index from the device
        if isinstance(device, torch.device):
            device_id = device.index if device.index is not None else 0
        else:
            device_id = int(device)
        # Use torch.musa.set_device directly for MUSA platform
        torch.musa.set_device(device_id)
        # Force eager device initialization
        _ = torch.zeros(1, device=f"musa:{device_id}")

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        raise NotImplementedError

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_fully_connected(cls, device_ids: list[int]) -> bool:
        raise NotImplementedError

    @classmethod
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        """
        Check if the device has a given capability.

        MUSA GPUs use a different capability scheme than CUDA. This method
        maps CUDA capability requirements to MUSA capabilities:
        - CUDA 8.0+ (BFloat16 support) -> MUSA 2.2+
        - CUDA 8.9+ (FP8 support) -> MUSA 3.1+

        This allows vLLM's CUDA-centric capability checks to work on MUSA.
        """
        # Convert capability to integer format
        if isinstance(capability, tuple):
            cap_int = capability[0] * 10 + capability[1]
        else:
            cap_int = capability

        # Get MUSA device capability
        musa_cap = cls.get_device_capability(device_id)
        if musa_cap is None:
            return False
        musa_cap_int = musa_cap.major * 10 + musa_cap.minor

        # Map CUDA capability requirements to MUSA capabilities
        # MUSA 3.1+ supports both BFloat16 and FP8
        if cap_int >= 89:  # FP8 requirement (CUDA 8.9+)
            # MUSA 3.1+ supports FP8
            return musa_cap_int >= 31
        elif cap_int >= 80:  # BFloat16 requirement (CUDA 8.0+)
            # MUSA 2.2+ supports BFloat16
            return musa_cap_int >= 22
        else:
            # For other capabilities, use direct comparison
            return musa_cap_int >= cap_int

    @classmethod
    def log_warnings(cls):
        pass

    @classmethod
    def supports_v1(cls, model_config: "ModelConfig") -> bool:
        """MUSA platform supports V1 engine."""
        return True

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Check and update the configuration for MUSA platform."""
        from vllm import envs
        from vllm.config import CUDAGraphMode

        # Try to import CompilationMode (vLLM 0.13+) or CompilationLevel (older)
        try:
            from vllm.config import CompilationMode

            NO_COMPILATION = CompilationMode.NONE
        except ImportError:
            from vllm.config import CompilationLevel

            NO_COMPILATION = CompilationLevel.NO_COMPILATION

        # Register MUSA cache ops (must be done after platform init is complete)
        _register_musa_cache_ops()

        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config

        # MUSA always uses V1 worker since V0 worker depends on libcuda.so
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        # Disable torch.compile for MUSA (device type not fully supported)
        # In vLLM 0.13+, V1 is the default engine; in older versions check VLLM_USE_V1
        use_v1 = getattr(envs, "VLLM_USE_V1", True)  # Default to True for 0.13+
        if use_v1 and model_config is not None:
            # Use 'level' for older vLLM, 'mode' for newer vLLM 0.13+
            if hasattr(vllm_config.compilation_config, "mode"):
                vllm_config.compilation_config.mode = NO_COMPILATION
            else:
                vllm_config.compilation_config.level = NO_COMPILATION

        # Disable CUDA graphs for MUSA (no CUDA runtime)
        compilation_config = vllm_config.compilation_config
        if (
            compilation_config.cudagraph_mode is None
            or compilation_config.cudagraph_mode.max_cudagraph_mode()
            != CUDAGraphMode.NONE
        ):
            logger.info_once(
                "[MUSA] CUDA graph is not supported on MUSA, " "disabling cudagraphs."
            )
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        # Ensure compile_sizes is initialized (since MUSA is not cuda_alike,
        # _set_cudagraph_sizes is not called, leaving compile_sizes as None)
        if compilation_config.compile_sizes is None:
            compilation_config.compile_sizes = []
        if compilation_config.cudagraph_capture_sizes is None:
            compilation_config.cudagraph_capture_sizes = []

        # For MUSA, use Triton attention backend
        # We leave backend=None here and let get_attn_backend_cls handle the default
        # This allows users to explicitly override via command line if needed

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        """Return the current memory usage in bytes."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.max_memory_allocated(device)

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "_Backend",
        attn_selector_config: Optional["AttentionSelectorConfig"] = None,
        # Legacy parameters for vLLM < 0.13 compatibility
        head_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        kv_cache_dtype: Optional[str] = None,
        block_size: Optional[int] = None,
        use_v1: Optional[bool] = None,
        use_mla: Optional[bool] = None,
        has_sink: Optional[bool] = None,
    ) -> str:
        """Get the attention backend class for MUSA.

        MUSA uses Triton-based attention backends since FlashAttention
        and other CUDA-specific backends are not available.
        On MUSA, we always use the Triton V1 attention backend regardless
        of V0/V1 engine mode, as it's the only backend that works with
        the MUSA Triton implementation.

        Supports both vLLM 0.10.x (legacy params) and 0.13+ (attn_selector_config)
        """
        # Always use Triton V1 attention backend for MUSA
        # This works because the V1 Triton backend uses pure Triton kernels
        # which are compatible with MUSA Triton
        TRITON_ATTN_V1 = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"  # noqa: E501

        logger.info_once("Using Triton Attention backend for MUSA platform.")
        return TRITON_ATTN_V1

    @classmethod
    def get_supported_vit_attn_backends(cls) -> list["AttentionBackendEnum"]:
        from vllm.attention.backends.registry import AttentionBackendEnum

        return [AttentionBackendEnum.TORCH_SDPA]

    @classmethod
    def get_vit_attn_backend(
        cls,
        head_size: int,
        dtype: torch.dtype,
        backend: Optional["AttentionBackendEnum"] = None,
    ) -> "AttentionBackendEnum":
        from vllm.attention.backends.registry import AttentionBackendEnum

        if backend is not None:
            assert backend in cls.get_supported_vit_attn_backends(), (
                f"Backend {backend} is not supported for vit attention. "
                f"Supported backends are: {cls.get_supported_vit_attn_backends()}"
            )
            return backend

        return AttentionBackendEnum.TORCH_SDPA

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """Get device communicator class - use CUDA communicator via torchada."""
        return (
            "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"
        )

    @classmethod
    def supports_fp8(cls) -> bool:
        """MUSA supports FP8 from device capability 3.1."""
        return cls.has_device_capability(31)

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        return True

    @classmethod
    def opaque_attention_op(cls) -> bool:
        """
        Return False to use direct attention calls instead of vLLM's
        opaque attention custom op, which is only registered for CUDA.
        """
        return False

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        """Use CUDA graph wrapper via torchada."""
        return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    @classmethod
    def device_count(cls) -> int:
        """Get the number of MUSA devices."""
        return torch.cuda.device_count()

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype):
        """Check if the dtype is supported by MUSA."""
        if dtype == torch.bfloat16:
            # MUSA GPUs generally support BF16
            pass

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True


# MTML utils
# Note that MTML is not affected by `MUSA_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# The major benefit of using MTML is that it will not initialize MUSA context.
class MtmlMUSAPlatform(MUSAPlatformBase):
    """MUSA platform using MTML (pymtml) for device queries.

    This is equivalent to NvmlCudaPlatform in cuda.py, using pymtml's
    pynvml-compatible API.
    """

    @classmethod
    @cache
    @with_nvml_context
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        try:
            physical_device_id = cls.device_id_to_physical_device_id(device_id)
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            return DeviceCapability(major=major, minor=minor)
        except RuntimeError:
            return None

    @classmethod
    @with_nvml_context
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        try:
            return super().has_device_capability(capability, device_id)
        except RuntimeError:
            return False

    @classmethod
    @with_nvml_context
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        return cls._get_physical_device_name(physical_device_id)

    @classmethod
    @with_nvml_context
    def get_device_uuid(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return pynvml.nvmlDeviceGetUUID(handle)

    @classmethod
    @with_nvml_context
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)

    @classmethod
    @with_nvml_context
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        """Query if the set of gpus are fully connected by MtLink (1 hop)."""
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                            handle,
                            peer_handle,
                            pynvml.NVML_P2P_CAPS_INDEX_NVLINK,
                        )
                        if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                            return False
                    except pynvml.NVMLError:
                        logger.exception(
                            "MtLink detection failed. This is normal if "
                            "your machine has no MtLink equipped."
                        )
                        return False
        return True

    @classmethod
    def _get_physical_device_name(cls, device_id: int = 0) -> str:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return pynvml.nvmlDeviceGetName(handle)

    @classmethod
    @with_nvml_context
    def log_warnings(cls):
        device_count: int = pynvml.nvmlDeviceGetCount()
        if device_count > 1:
            device_names = [
                cls._get_physical_device_name(i) for i in range(device_count)
            ]
            if (
                len(set(device_names)) > 1
                and os.environ.get("MUSA_DEVICE_ORDER") != "PCI_BUS_ID"
            ):
                logger.warning(
                    "Detected different devices in the system: %s. Please "
                    "make sure to set `MUSA_DEVICE_ORDER=PCI_BUS_ID` to "
                    "avoid unexpected behavior.",
                    ", ".join(device_names),
                )


class NonMtmlMUSAPlatform(MUSAPlatformBase):
    """MUSA platform without MTML - uses PyTorch APIs via torchada."""

    @classmethod
    @cache
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        """Get device capability using PyTorch."""
        # Use torch.cuda APIs via torchada
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def is_fully_connected(cls, _physical_device_ids: list[int]) -> bool:
        """Without MTML, we cannot detect MtLink connectivity."""
        logger.warning(
            "MtLink detection not possible without MTML. "
            "Assuming no MtLink available."
        )
        return False


# Autodetect MTML availability
# Note: We don't call nvmlShutdown() here because pymtml has a bug where
# nvmlInit() fails after nvmlShutdown() has been called.
mtml_available = False
try:
    pynvml.nvmlInit()
    mtml_available = True
except Exception:
    mtml_available = False

MUSAPlatform = MtmlMUSAPlatform if mtml_available else NonMtmlMUSAPlatform

MUSAPlatform.log_warnings()
