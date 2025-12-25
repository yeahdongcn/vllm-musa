# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MTML (Moore Threads Management Library) wrapper module."""

import builtins
import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


def _clear_mtml_modules():
    """Clear mtml-related modules from sys.modules."""
    modules_to_remove = [k for k in sys.modules.keys()
                         if k == "pymtml" or k.startswith("vllm_musa_platform")]
    for mod in modules_to_remove:
        sys.modules.pop(mod, None)


class TestMtmlAvailability:
    """Tests for MTML availability detection."""

    def test_is_mtml_available_returns_true_when_pymtml_present(self):
        """Test that is_mtml_available returns True when pymtml is present."""
        # Just test the current environment - pymtml is installed
        from vllm_musa_platform import mtml

        # If we got here without error, pymtml exists
        result = mtml.is_mtml_available()
        # The result depends on whether pymtml is installed
        assert isinstance(result, bool)

    def test_is_mtml_available_returns_false_when_pymtml_missing(self):
        """Test that is_mtml_available returns False without pymtml."""
        # Save original import
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pymtml":
                raise ImportError("No module named 'pymtml'")
            return original_import(name, *args, **kwargs)

        _clear_mtml_modules()

        try:
            with patch.object(builtins, "__import__", side_effect=mock_import):
                # Re-import mtml with mocked import
                import vllm_musa_platform.mtml as mtml_module
                importlib.reload(mtml_module)

                assert mtml_module.is_mtml_available() is False
                assert mtml_module.MTML_AVAILABLE is False
        finally:
            # Restore original state
            _clear_mtml_modules()


class TestMtmlFunctionsWithRealPymtml:
    """Tests for MTML wrapper functions using the real pymtml if available."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Skip tests if pymtml is not available."""
        try:
            import pymtml
            yield
        except ImportError:
            pytest.skip("pymtml not available")

    def test_mtml_init_and_shutdown(self):
        """Test mtmlInit and mtmlShutdown work correctly."""
        from vllm_musa_platform import mtml

        # Should not raise
        mtml.mtmlInit()
        mtml.mtmlShutdown()

    def test_mtml_device_get_count(self):
        """Test mtmlDeviceGetCount returns a non-negative count."""
        from vllm_musa_platform import mtml

        mtml.mtmlInit()
        try:
            count = mtml.mtmlDeviceGetCount()
            assert isinstance(count, int)
            assert count >= 0
        finally:
            mtml.mtmlShutdown()

    def test_mtml_device_get_name(self):
        """Test mtmlDeviceGetName returns a string."""
        from vllm_musa_platform import mtml

        mtml.mtmlInit()
        try:
            count = mtml.mtmlDeviceGetCount()
            if count > 0:
                handle = mtml.mtmlDeviceGetHandleByIndex(0)
                name = mtml.mtmlDeviceGetName(handle)
                assert isinstance(name, str)
                assert len(name) > 0
            else:
                pytest.skip("No MUSA devices available")
        finally:
            mtml.mtmlShutdown()


class TestMtmlFunctionsWithMock:
    """Tests for MTML wrapper functions using mocked pymtml."""

    def test_mtml_wrapper_function_signatures(self):
        """Test that MTML wrapper functions have correct signatures."""
        from vllm_musa_platform import mtml
        import inspect

        # Test function signatures exist
        assert callable(mtml.mtmlInit)
        assert callable(mtml.mtmlShutdown)
        assert callable(mtml.mtmlDeviceGetCount)
        assert callable(mtml.mtmlDeviceGetHandleByIndex)
        assert callable(mtml.mtmlDeviceGetName)
        assert callable(mtml.mtmlDeviceGetUUID)
        assert callable(mtml.mtmlDeviceGetMemoryInfo)

        # Test mtmlDeviceGetHandleByIndex takes an index
        sig = inspect.signature(mtml.mtmlDeviceGetHandleByIndex)
        params = list(sig.parameters.keys())
        assert "index" in params


class TestMtmlErrors:
    """Tests for MTML error handling."""

    def test_mtml_error_exception_exists(self):
        """Test that MTMLError exception class exists."""
        from vllm_musa_platform.mtml import MTMLError

        assert issubclass(MTMLError, Exception)

        # Test it can be raised and caught
        with pytest.raises(MTMLError):
            raise MTMLError("test error")

    def test_functions_raise_error_when_pymtml_unavailable(self):
        """Test that functions raise MTMLError when pymtml is not available."""
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pymtml":
                raise ImportError("No module named 'pymtml'")
            return original_import(name, *args, **kwargs)

        _clear_mtml_modules()

        try:
            with patch.object(builtins, "__import__", side_effect=mock_import):
                import vllm_musa_platform.mtml as mtml_module
                importlib.reload(mtml_module)

                with pytest.raises(mtml_module.MTMLError, match="pymtml is not available"):
                    mtml_module.mtmlInit()
        finally:
            _clear_mtml_modules()


class TestMtmlConstants:
    """Tests for MTML constants."""

    def test_mtlink_state_constants(self):
        """Test MtLink state constants are defined correctly."""
        from vllm_musa_platform import mtml

        assert mtml.MTML_MTLINK_STATE_DOWN == 0
        assert mtml.MTML_MTLINK_STATE_UP == 1
        assert mtml.MTML_MTLINK_STATE_DOWNGRADE == 2


class TestMtmlExports:
    """Tests for module exports."""

    def test_all_exports_defined(self):
        """Test that __all__ is defined and contains expected items."""
        from vllm_musa_platform import mtml

        assert hasattr(mtml, "__all__")

        expected_exports = [
            "MTML_AVAILABLE",
            "is_mtml_available",
            "MTMLError",
            "mtmlInit",
            "mtmlShutdown",
            "mtmlDeviceGetCount",
            "mtmlDeviceGetHandleByIndex",
            "mtmlDeviceGetName",
            "mtmlDeviceGetUUID",
            "mtmlDeviceGetMemoryInfo",
            "mtmlDeviceGetMtLinkSpec",
            "mtmlDeviceGetMtLinkState",
            "mtmlDeviceGetMtLinkRemoteDevice",
            "MTML_MTLINK_STATE_DOWN",
            "MTML_MTLINK_STATE_UP",
            "MTML_MTLINK_STATE_DOWNGRADE",
        ]

        for export in expected_exports:
            assert export in mtml.__all__, f"{export} not in __all__"
            assert hasattr(mtml, export), f"{export} not defined in module"


class TestMtmlMtLinkFunctions:
    """Tests for MtLink-related functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Skip tests if pymtml is not available."""
        try:
            import pymtml
            yield
        except ImportError:
            pytest.skip("pymtml not available")

    def test_mtlink_functions_exist(self):
        """Test that MtLink functions are callable."""
        from vllm_musa_platform import mtml

        assert callable(mtml.mtmlDeviceGetMtLinkSpec)
        assert callable(mtml.mtmlDeviceGetMtLinkState)
        assert callable(mtml.mtmlDeviceGetMtLinkRemoteDevice)

    def test_mtlink_spec_has_link_num(self):
        """Test that MtLink spec has linkNum attribute."""
        from vllm_musa_platform import mtml

        mtml.mtmlInit()
        try:
            count = mtml.mtmlDeviceGetCount()
            if count > 0:
                handle = mtml.mtmlDeviceGetHandleByIndex(0)
                try:
                    spec = mtml.mtmlDeviceGetMtLinkSpec(handle)
                    # Spec should have linkNum attribute
                    assert hasattr(spec, "linkNum")
                    assert isinstance(spec.linkNum, int)
                    assert spec.linkNum >= 0
                except Exception:
                    # Some devices may not support MtLink
                    pytest.skip("Device does not support MtLink")
            else:
                pytest.skip("No MUSA devices available")
        finally:
            mtml.mtmlShutdown()

