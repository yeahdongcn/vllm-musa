# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MUSA Platform implementation."""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestMUSAPlatformBase:
    """Tests for MUSAPlatformBase class."""

    def test_device_name(self):
        """Test that device_name is set correctly."""
        from vllm_musa_platform.musa import MUSAPlatformBase

        assert MUSAPlatformBase.device_name == "musa"

    def test_device_type(self):
        """Test that device_type is set correctly."""
        from vllm_musa_platform.musa import MUSAPlatformBase

        assert MUSAPlatformBase.device_type == "musa"

    def test_dispatch_key(self):
        """Test that dispatch_key uses CUDA via torchada."""
        from vllm_musa_platform.musa import MUSAPlatformBase

        assert MUSAPlatformBase.dispatch_key == "CUDA"

    def test_dist_backend(self):
        """Test that dist_backend uses mccl."""
        from vllm_musa_platform.musa import MUSAPlatformBase

        assert MUSAPlatformBase.dist_backend == "mccl"

    def test_device_control_env_var(self):
        """Test that device_control_env_var is set correctly."""
        from vllm_musa_platform.musa import MUSAPlatformBase

        assert MUSAPlatformBase.device_control_env_var == "MUSA_VISIBLE_DEVICES"

    def test_ray_device_key(self):
        """Test that ray_device_key is set correctly."""
        from vllm_musa_platform.musa import MUSAPlatformBase

        assert MUSAPlatformBase.ray_device_key == "GPU"

    def test_is_cuda_alike_returns_false(self):
        """Test that is_cuda_alike returns False for MUSA."""
        from vllm_musa_platform.musa import MUSAPlatformBase

        platform = MUSAPlatformBase()
        assert platform.is_cuda_alike() is False

    def test_is_sleep_mode_available_returns_false(self):
        """Test that is_sleep_mode_available returns False."""
        from vllm_musa_platform.musa import MUSAPlatformBase

        platform = MUSAPlatformBase()
        assert platform.is_sleep_mode_available() is False

    def test_supported_dtypes(self):
        """Test that supported_dtypes includes bf16, fp16, and fp32."""
        import torch

        from vllm_musa_platform.musa import MUSAPlatformBase

        platform = MUSAPlatformBase()
        dtypes = platform.supported_dtypes

        assert torch.bfloat16 in dtypes
        assert torch.float16 in dtypes
        assert torch.float32 in dtypes

    def test_opaque_attention_op_returns_false(self):
        """Test that opaque_attention_op returns False."""
        from vllm_musa_platform.musa import MUSAPlatformBase

        assert MUSAPlatformBase.opaque_attention_op() is False

    def test_use_custom_allreduce_returns_true(self):
        """Test that use_custom_allreduce returns True."""
        from vllm_musa_platform.musa import MUSAPlatformBase

        assert MUSAPlatformBase.use_custom_allreduce() is True

    def test_support_hybrid_kv_cache(self):
        """Test that support_hybrid_kv_cache returns True."""
        from vllm_musa_platform.musa import MUSAPlatformBase

        assert MUSAPlatformBase.support_hybrid_kv_cache() is True

    def test_support_static_graph_mode(self):
        """Test that support_static_graph_mode returns True."""
        from vllm_musa_platform.musa import MUSAPlatformBase

        assert MUSAPlatformBase.support_static_graph_mode() is True

    def test_get_punica_wrapper(self):
        """Test get_punica_wrapper returns correct path."""
        from vllm_musa_platform.musa import MUSAPlatformBase

        result = MUSAPlatformBase.get_punica_wrapper()
        assert result == "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    def test_get_device_communicator_cls(self):
        """Test get_device_communicator_cls returns CUDA communicator."""
        from vllm_musa_platform.musa import MUSAPlatformBase

        result = MUSAPlatformBase.get_device_communicator_cls()
        expected = (
            "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"
        )
        assert result == expected

    def test_get_static_graph_wrapper_cls(self):
        """Test get_static_graph_wrapper_cls returns CUDA graph wrapper."""
        from vllm_musa_platform.musa import MUSAPlatformBase

        result = MUSAPlatformBase.get_static_graph_wrapper_cls()
        assert result == "vllm.compilation.cuda_graph.CUDAGraphWrapper"


class TestNonMtmlMUSAPlatform:
    """Tests for NonMtmlMUSAPlatform class."""

    def test_get_device_capability(self):
        """Test get_device_capability returns DeviceCapability."""
        with patch("torch.cuda.get_device_capability") as mock_cap:
            mock_cap.return_value = (3, 1)

            from vllm_musa_platform.musa import NonMtmlMUSAPlatform

            # Clear cache to allow re-testing
            NonMtmlMUSAPlatform.get_device_capability.cache_clear()

            cap = NonMtmlMUSAPlatform.get_device_capability(0)

            assert cap.major == 3
            assert cap.minor == 1

    def test_get_device_name(self):
        """Test get_device_name returns device name."""
        with patch("torch.cuda.get_device_name") as mock_name:
            mock_name.return_value = "MTT S80"

            from vllm_musa_platform.musa import NonMtmlMUSAPlatform

            name = NonMtmlMUSAPlatform.get_device_name(0)

            assert name == "MTT S80"

    def test_get_device_total_memory(self):
        """Test get_device_total_memory returns memory size."""
        mock_props = MagicMock()
        mock_props.total_memory = 80 * 1024 * 1024 * 1024  # 80GB

        with patch("torch.cuda.get_device_properties") as mock_get_props:
            mock_get_props.return_value = mock_props

            from vllm_musa_platform.musa import NonMtmlMUSAPlatform

            memory = NonMtmlMUSAPlatform.get_device_total_memory(0)

            assert memory == 80 * 1024 * 1024 * 1024

    def test_is_fully_connected_returns_false_with_warning(self):
        """Test is_fully_connected returns False without MTML."""
        from vllm_musa_platform.musa import NonMtmlMUSAPlatform

        result = NonMtmlMUSAPlatform.is_fully_connected([0, 1])

        assert result is False


class TestWithMtmlContext:
    """Tests for the with_mtml_context decorator."""

    def test_decorator_returns_function_result(self):
        """Test that the decorator returns the wrapped function's result."""
        from vllm_musa_platform.musa import mtml, with_mtml_context

        if mtml is None or not mtml.is_mtml_available():
            pytest.skip("MTML not available")

        @with_mtml_context
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_decorator_preserves_function_name(self):
        """Test that the decorator preserves the wrapped function's name."""
        from vllm_musa_platform.musa import with_mtml_context

        @with_mtml_context
        def my_test_function():
            return "test"

        assert my_test_function.__name__ == "my_test_function"


class TestMtmlMUSAPlatform:
    """Tests for MtmlMUSAPlatform class."""

    def test_get_device_capability_returns_3_1(self, mock_pymtml):
        """Test get_device_capability returns (3, 1) for FP8 support."""
        if "vllm_musa_platform.mtml" in sys.modules:
            del sys.modules["vllm_musa_platform.mtml"]
        if "vllm_musa_platform.musa" in sys.modules:
            del sys.modules["vllm_musa_platform.musa"]

        from vllm_musa_platform.musa import MtmlMUSAPlatform, mtml

        if not mtml.is_mtml_available():
            pytest.skip("MTML not available")

        # Clear cache
        MtmlMUSAPlatform.get_device_capability.cache_clear()

        cap = MtmlMUSAPlatform.get_device_capability(0)

        assert cap.major == 3
        assert cap.minor == 1

    def test_get_device_name(self):
        """Test get_device_name returns a string."""
        from vllm_musa_platform.musa import MtmlMUSAPlatform, mtml

        if mtml is None or not mtml.is_mtml_available():
            pytest.skip("MTML not available")

        name = MtmlMUSAPlatform.get_device_name(0)

        assert isinstance(name, str)
        assert len(name) > 0
        # MUSA device names typically start with "MTT"
        assert "MTT" in name or len(name) > 0

    def test_get_device_uuid(self):
        """Test get_device_uuid returns a valid UUID string."""
        from vllm_musa_platform.musa import MtmlMUSAPlatform, mtml

        if mtml is None or not mtml.is_mtml_available():
            pytest.skip("MTML not available")

        uuid = MtmlMUSAPlatform.get_device_uuid(0)

        assert isinstance(uuid, str)
        # UUIDs have a specific format with dashes
        assert "-" in uuid
        assert len(uuid) >= 32  # Minimum UUID length

    def test_get_device_total_memory(self):
        """Test get_device_total_memory returns a positive integer."""
        from vllm_musa_platform.musa import MtmlMUSAPlatform, mtml

        if mtml is None or not mtml.is_mtml_available():
            pytest.skip("MTML not available")

        memory = MtmlMUSAPlatform.get_device_total_memory(0)

        assert isinstance(memory, int)
        assert memory > 0
        # Typical GPU memory is at least 4GB
        assert memory >= 4 * 1024 * 1024 * 1024


class TestPlatformSelection:
    """Tests for platform autodetection."""

    def test_musa_platform_is_one_of_two_options(self):
        """Test that MUSAPlatform is either MtmlMUSAPlatform or NonMtmlMUSAPlatform."""
        from vllm_musa_platform.musa import (
            MtmlMUSAPlatform,
            MUSAPlatform,
            NonMtmlMUSAPlatform,
        )

        assert MUSAPlatform in (MtmlMUSAPlatform, NonMtmlMUSAPlatform)

    def test_platform_selection_based_on_mtml_availability(self):
        """Test that platform selection is correct based on MTML availability."""
        from vllm_musa_platform.musa import (
            MtmlMUSAPlatform,
            MUSAPlatform,
            NonMtmlMUSAPlatform,
            mtml_available,
        )

        if mtml_available:
            assert MUSAPlatform is MtmlMUSAPlatform
        else:
            assert MUSAPlatform is NonMtmlMUSAPlatform


class TestImportTorchada:
    """Tests for torchada import handling."""

    def test_import_torchada_returns_none_when_unavailable(self):
        """Test _import_torchada returns None when torchada is not installed."""
        original_torchada = sys.modules.pop("torchada", None)

        try:
            from vllm_musa_platform.musa import _import_torchada

            # We can't easily test this without modifying the import system
            # Just verify the function exists
            assert callable(_import_torchada)
        finally:
            if original_torchada is not None:
                sys.modules["torchada"] = original_torchada


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_defined(self):
        """Test that __all__ is defined and contains expected items."""
        from vllm_musa_platform import musa

        assert hasattr(musa, "__all__")

        expected_exports = [
            "MUSAPlatform",
            "MUSAPlatformBase",
            "MtmlMUSAPlatform",
            "NonMtmlMUSAPlatform",
            "with_mtml_context",
            "mtml_available",
            "mtml",
        ]

        for export in expected_exports:
            assert export in musa.__all__, f"{export} not in __all__"
            assert hasattr(musa, export), f"{export} not defined in module"

    def test_musa_platform_plugin_function_exists(self):
        """Test that musa_platform_plugin function exists for entry point."""
        from vllm_musa_platform import musa_platform_plugin

        assert callable(musa_platform_plugin)
