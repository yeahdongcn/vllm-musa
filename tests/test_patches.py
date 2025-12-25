# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MUSA platform patches module."""

import importlib
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestPatchFileNaming:
    """Tests for patch file naming convention."""

    def test_get_patch_files_returns_correct_module_names(self):
        """Test that patch file names are correctly converted to module names."""
        from vllm_musa_platform.patches import _get_patch_files
        
        patch_files = _get_patch_files()
        
        # Should find the triton unified attention patch
        module_names = [name for name, path in patch_files]
        
        assert "vllm.attention.ops.triton_unified_attention" in module_names

    def test_naming_convention_double_underscore_to_dot(self):
        """Test that double underscores are converted to dots."""
        from vllm_musa_platform.patches import _get_patch_files
        
        patch_files = _get_patch_files()
        
        for module_name, path in patch_files:
            # Module names should not contain double underscores
            assert "__" not in module_name
            # Should have proper Python module path format
            assert module_name.startswith("vllm")


class TestPatchFileLoading:
    """Tests for patch file content loading."""

    def test_load_patch_config_returns_patches_list(self):
        """Test that _load_patch_config extracts PATCHES list."""
        from vllm_musa_platform.patches import _load_patch_config, _get_patch_files
        
        patch_files = _get_patch_files()
        
        # Find the triton patch file
        for module_name, patch_path in patch_files:
            if "triton_unified_attention" in module_name:
                patches = _load_patch_config(patch_path)
                
                assert isinstance(patches, list)
                assert len(patches) > 0
                
                # Each patch should be a tuple of (old, new)
                for old, new in patches:
                    assert isinstance(old, str)
                    assert isinstance(new, str)

    def test_load_patch_config_handles_missing_patches_list(self, tmp_path):
        """Test that _load_patch_config handles files without PATCHES."""
        from vllm_musa_platform.patches import _load_patch_config
        
        # Create a temporary patch file without PATCHES
        patch_file = tmp_path / "test.patch.py"
        patch_file.write_text("# No PATCHES defined\nFOO = 'bar'\n")
        
        patches = _load_patch_config(patch_file)
        
        assert patches == []


class TestTritonPatch:
    """Tests for the Triton unified attention patch."""

    def test_patch_file_exists(self):
        """Test that the triton patch file exists."""
        from vllm_musa_platform.patches import _get_patch_files
        
        patch_files = _get_patch_files()
        module_names = [name for name, path in patch_files]
        
        assert "vllm.attention.ops.triton_unified_attention" in module_names

    def test_patch_contains_annotated_assignment_fix(self):
        """Test that patch contains the annotated assignment fix."""
        from vllm_musa_platform.patches import _get_patch_files, _load_patch_config
        
        patch_files = _get_patch_files()
        
        for module_name, patch_path in patch_files:
            if "triton_unified_attention" in module_name:
                patches = _load_patch_config(patch_path)
                
                # Should have the fix for "left: tl.int32 = 0"
                old_strs = [old for old, new in patches]
                
                assert "left: tl.int32 = 0" in old_strs


class TestApplyPatches:
    """Tests for the apply_patches function."""

    def test_apply_patches_is_idempotent(self):
        """Test that apply_patches can be called multiple times safely."""
        from vllm_musa_platform import patches
        
        # Reset the flag
        patches._patches_applied = False
        
        # First call
        patches.apply_patches()
        assert patches._patches_applied is True
        
        # Second call should be a no-op
        patches.apply_patches()
        assert patches._patches_applied is True

    def test_apply_patches_handles_missing_module(self):
        """Test that apply_patches handles non-existent modules gracefully."""
        from vllm_musa_platform import patches
        
        # Reset state
        patches._patches_applied = False
        
        # Create a mock patch file for a non-existent module
        with patch.object(patches, '_get_patch_files') as mock_get:
            mock_get.return_value = [("non.existent.module", Path("/fake/path"))]
            
            with patch.object(patches, '_load_patch_config') as mock_load:
                mock_load.return_value = [("old", "new")]
                
                # Should not raise
                patches.apply_patches()


class TestPatchesReadme:
    """Tests for patches documentation."""

    def test_readme_exists(self):
        """Test that README.md exists in patches directory."""
        patches_dir = Path(__file__).parent.parent / "vllm_musa_platform" / "patches"
        readme_path = patches_dir / "README.md"
        
        assert readme_path.exists()

    def test_readme_documents_naming_convention(self):
        """Test that README explains the naming convention."""
        patches_dir = Path(__file__).parent.parent / "vllm_musa_platform" / "patches"
        readme_path = patches_dir / "README.md"
        
        content = readme_path.read_text()
        
        # Should document the double underscore convention
        assert "__" in content or "double underscore" in content.lower()
        assert ".patch.py" in content

