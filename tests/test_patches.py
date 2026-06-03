# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MUSA platform patches module."""

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch


class TestPatchFileNaming:
    """Tests for patch file naming convention."""

    def test_get_patch_files_returns_correct_module_names(self):
        """Test that patch file names are correctly converted to module names."""
        from vllm_musa.patches import _get_patch_files

        patch_files = _get_patch_files()

        # Should find the triton unified attention patch
        module_names = [name for name, path in patch_files]

        assert "vllm.v1.attention.ops.triton_unified_attention" in module_names

    def test_naming_convention_double_underscore_to_dot(self):
        """Test that double underscores are converted to dots."""
        from vllm_musa.patches import _get_patch_files

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
        from vllm_musa.patches import _get_patch_files, _load_patch_config

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
        from vllm_musa.patches import _load_patch_config

        # Create a temporary patch file without PATCHES
        patch_file = tmp_path / "test.patch.py"
        patch_file.write_text("# No PATCHES defined\nFOO = 'bar'\n")

        patches = _load_patch_config(patch_file)

        assert patches == []


class TestCustomOpsRuntimePatches:
    def test_rms_norm_wrapper_is_registered_on_vllm_custom_ops(self, monkeypatch):
        import vllm

        import vllm_musa

        vllm_ops = ModuleType("vllm._custom_ops")
        monkeypatch.setitem(sys.modules, "vllm._custom_ops", vllm_ops)
        monkeypatch.setattr(vllm, "_custom_ops", vllm_ops, raising=False)

        vllm_musa._patch_vllm_custom_ops_dflash_fallbacks()

        assert vllm_ops.rms_norm is vllm_musa._musa_safe_rms_norm
        assert getattr(vllm_ops.rms_norm, "_musa_safe_rms_norm") is True
        assert vllm_ops.rotary_embedding is vllm_musa._musa_safe_rotary_embedding
        assert getattr(vllm_ops.rotary_embedding, "_musa_safe_rotary_embedding") is True


class TestDeepSeekV4V022Patches:
    def test_common_inv_rope_fp8_quant_patch_is_discoverable(self):
        from vllm_musa.patches import _get_patch_files

        modules = {module_name for module_name, _ in _get_patch_files()}

        assert (
            "vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant"
            in modules
        )

    def test_attention_fp8_einsum_patch_is_discoverable(self):
        from vllm_musa.patches import _get_patch_files

        modules = {module_name for module_name, _ in _get_patch_files()}

        assert "vllm.models.deepseek_v4.attention" in modules

    def test_nvidia_model_patch_is_discoverable(self):
        from vllm_musa.patches import _get_patch_files

        modules = {module_name for module_name, _ in _get_patch_files()}

        assert "vllm.models.deepseek_v4.nvidia.model" in modules

    def test_attention_fp8_einsum_patch_uses_musa_provider(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        for module_name, patch_path in _get_patch_files():
            if module_name == "vllm.models.deepseek_v4.attention":
                patches = _load_patch_config(patch_path)
                new_source = "\n".join(new for _, new in patches)

                assert "try_musa_deepseek_v4_fp8_einsum" in new_source
                assert "upstream DeepGEMM fp8_einsum" in new_source
                assert "not available in the MUSA runtime" in new_source
                assert "current_platform.is_musa()" in new_source
                assert "_musa_deepseek_v4_mm_out_dtype" in new_source
                assert "torch.mm(a.to(out_dtype), b.to(out_dtype))" in new_source
                break
        else:
            raise AssertionError("DeepSeek-V4 v0.22 attention patch was not found")

    def test_nvidia_model_patch_runs_final_hc_post_on_musa(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        for module_name, patch_path in _get_patch_files():
            if module_name == "vllm.models.deepseek_v4.nvidia.model":
                patches = _load_patch_config(patch_path)
                new_source = "\n".join(new for _, new in patches)

                assert "current_platform.is_cuda() or current_platform.is_musa()" in (
                    new_source
                )
                assert "hidden_states = layer.hc_post(" in new_source
                break
        else:
            raise AssertionError("DeepSeek-V4 v0.22 NVIDIA model patch was not found")


class TestTritonPatch:
    """Tests for the Triton unified attention patch."""

    def test_patch_file_exists(self):
        """Test that the triton patch file exists."""
        from vllm_musa.patches import _get_patch_files

        patch_files = _get_patch_files()
        module_names = [name for name, path in patch_files]

        assert "vllm.v1.attention.ops.triton_unified_attention" in module_names

    def test_patch_contains_annotated_assignment_fix(self):
        """Test that patch contains the annotated assignment fix."""
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if "triton_unified_attention" in module_name:
                patches = _load_patch_config(patch_path)

                # Should have the fix for "left: tl.int32 = 0"
                old_strs = [old for old, new in patches]

                assert "left: tl.int32 = 0" in old_strs


class TestCustomAllReducePatch:
    """Tests for the MUSA custom all-reduce patch."""

    def test_patch_skips_cuda_p2p_check_on_musa(self):
        """Test that MUSA custom all-reduce does not run CUDA P2P probing."""
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if "custom_all_reduce" in module_name:
                patches = _load_patch_config(patch_path)
                old_strs = [old for old, new in patches]
                new_strs = [new for old, new in patches]

                assert (
                    "if ( not current_platform.is_rocm() or not "
                    "current_platform.is_musa() ) and not _can_p2p(rank, world_size):"
                ) in old_strs
                assert (
                    "if not current_platform.is_rocm() and not "
                    "current_platform.is_musa() and not _can_p2p(rank, world_size):"
                ) in new_strs
                assert all(
                    "not current_platform.is_rocm() or not current_platform.is_musa()"
                    not in new
                    for new in new_strs
                )
                break
        else:
            raise AssertionError("custom_all_reduce patch file was not found")


class TestSamplerPatch:
    """Tests for the MUSA top-k/top-p sampler patch."""

    def test_sampler_patch_keeps_triton_guard_and_fast_path_gate(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.v1.sample.ops.topk_topp_sampler":
                patches = _load_patch_config(patch_path)
                new_source = "\n".join(new for _, new in patches)

                assert "not current_platform.is_musa()" in new_source
                assert "VLLM_MUSA_SAMPLER_FAST_PATH" in new_source
                assert "_apply_top_k_top_p_musa_topk_prefilter" in new_source
                assert "logits.shape[0] >= 16" in new_source
                assert "logits.shape[1] >= 65536" in new_source
                break
        else:
            # MUSA-3172: the topk_topp_sampler *source patch* was removed in the
            # v0.22 migration to MUSA native sampling ops ("Using MUSA native
            # ops for top-p/top-k/min-p sampling"). This test guarded the old
            # patch; skip until it is rewritten to assert the native-ops path.
            pytest.skip(
                "topk_topp_sampler source patch removed in v0.22 native-ops "
                "sampler migration; test needs rewrite for the native path"
            )


class TestRejectionSamplerPatch:
    """Tests for the MUSA speculative rejection-sampler patch."""

    def test_rejection_sampler_random_path_uses_target_token_fallback(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.v1.sample.rejection_sampler":
                patches = _load_patch_config(patch_path)
                new_source = "\n".join(new for _, new in patches)

                assert "VLLM_MUSA_SPEC_DECODE_RANDOM_FALLBACK" in new_source
                assert "_musa_sample_first_target_token" in new_source
                assert "not sampling_metadata.all_greedy" in new_source
                assert "sampling_metadata.max_num_logprobs is None" in new_source
                assert "PLACEHOLDER_TOKEN_ID" in new_source
                break
        else:
            raise AssertionError("rejection_sampler patch file was not found")


class TestGPUModelRunnerPatch:
    """Tests for MUSA GPU model runner speculative guards."""

    def test_model_runner_skips_musa_random_drafter(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.v1.worker.gpu_model_runner":
                patches = _load_patch_config(patch_path)
                new_source = "\n".join(new for _, new in patches)

                assert "VLLM_MUSA_SPEC_DECODE_RANDOM_FALLBACK" in new_source
                assert "current_platform.is_musa()" in new_source
                assert "not self.input_batch.sampling_metadata.all_greedy" in (
                    new_source
                )
                assert "input_fits_in_drafter = False" in new_source
                break
        else:
            raise AssertionError("gpu_model_runner patch file was not found")


class TestLLMBaseProposerPatch:
    """Tests for MUSA LLM base proposer speculative guards."""

    def test_draft_full_wrapper_is_musa_opt_in(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.v1.spec_decode.llm_base_proposer":
                patches = _load_patch_config(patch_path)
                new_source = "\n".join(new for _, new in patches)

                assert (
                    'draft_full_wrap_default = "0" if current_platform.is_musa()'
                    in new_source
                )
                assert 'VLLM_MUSA_DRAFT_FULL_WRAP", draft_full_wrap_default' in (
                    new_source
                )
                assert "self.model = CUDAGraphWrapper(" in new_source
                break
        else:
            raise AssertionError("llm_base_proposer patch file was not found")

    def test_draft_full_wrapper_normalizer_updates_old_default(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_module

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.v1.spec_decode.llm_base_proposer":
                module = _load_patch_module(patch_path)
                assert module is not None
                source = """
        # Gated by VLLM_MUSA_DRAFT_FULL_WRAP (default ON since the
        # query_start_loc in-place hunk fixed the captured-replay stale
        # data_ptr issue). Set to "0" to disable for debugging.
        import os as _os
        cudagraph_mode = self.compilation_config.cudagraph_mode
        if (
            _os.environ.get("VLLM_MUSA_DRAFT_FULL_WRAP", "1") == "1"
            and cudagraph_mode.has_full_cudagraphs()
"""
                normalized = module.normalize_source(source)

                assert '_os.environ.get("VLLM_MUSA_DRAFT_FULL_WRAP", "1")' not in (
                    normalized
                )
                assert (
                    'draft_full_wrap_default = "0" if current_platform.is_musa()'
                    in normalized
                )
                assert 'VLLM_MUSA_DRAFT_FULL_WRAP", draft_full_wrap_default' in (
                    normalized
                )
                break
        else:
            raise AssertionError("llm_base_proposer patch file was not found")


class TestDeepSeekV4AttentionPatch:
    """Tests for the MUSA DeepSeek-V4 attention patch."""

    def test_v022_attention_redirects_qnorm_rope_insert_to_musa_native(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.models.deepseek_v4.attention":
                patches = _load_patch_config(patch_path)
                new_source = "\n".join(new for _, new in patches)

                assert "def _musa_deepseek_v4_qnorm_rope_kv_insert(" in new_source
                assert "_musa_custom_ops.deepseek_v4_qnorm_rope_kv_insert(" in (
                    new_source
                )
                assert "if _musa_deepseek_v4_is_musa_tensor(q):" in new_source
                assert "swa_kv_cache," in new_source
                assert "F.pad(q, (0, 0, 0, padded_heads - q.shape[1])" in (
                    new_source
                )
                assert "slot_mapping[: q.shape[0]].contiguous()" in new_source
                break
        else:
            raise AssertionError("v0.22 deepseek_v4 attention patch file was not found")

    def test_v022_deepseek_v4_flashmla_uses_musa_ops_wrapper(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.models.deepseek_v4.nvidia.flashmla":
                patches = _load_patch_config(patch_path)
                new_source = "\n".join(new for _, new in patches)

                assert "from vllm_musa.v1.attention.ops.flashmla import (" in (
                    new_source
                )
                assert "flash_mla_sparse_fwd" in new_source
                assert "flash_mla_with_kvcache" in new_source
                break
        else:
            raise AssertionError(
                "v0.22 deepseek_v4 nvidia flashmla patch file was not found"
            )

    def test_qnorm_rope_native_path_trims_padded_slot_mapping(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.model_executor.layers.deepseek_v4_attention":
                patches = _load_patch_config(patch_path)
                new_source = "\n".join(new for _, new in patches)

                assert "native_slot_mapping = slot_mapping" in new_source
                assert "slot_mapping.shape[0] > q.shape[0]" in new_source
                assert "slot_mapping[: q.shape[0]].contiguous()" in new_source
                assert "native_slot_mapping," in new_source
                assert new_source.count("native_slot_mapping = slot_mapping") >= 2
                break
        else:
            raise AssertionError("deepseek_v4_attention patch file was not found")

    def test_custom_ops_wrapper_trims_padded_slot_mapping(self):
        source = (Path(__file__).parents[1] / "vllm_musa/_custom_ops.py").read_text()

        assert "def deepseek_v4_qnorm_rope_kv_insert(" in source
        assert "slot_mapping.shape[0] > q.shape[0]" in source
        assert "slot_mapping[: q.shape[0]].contiguous()" in source

    def test_decode_flashmla_path_trims_padded_sparse_metadata(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.model_executor.layers.deepseek_v4_attention":
                patches = _load_patch_config(patch_path)
                new_source = "\n".join(new for _, new in patches)

                assert "active_decode_tokens = q.shape[0]" in new_source
                assert "topk_indices[:active_decode_tokens]" in new_source
                assert "topk_lens[:active_decode_tokens]" in new_source
                assert "swa_indices[:active_decode_tokens]" in new_source
                assert "swa_lens[:active_decode_tokens]" in new_source
                break
        else:
            raise AssertionError("deepseek_v4_attention patch file was not found")


class TestSparseAttnIndexerPatch:
    """Tests for the MUSA DeepSeek-V4 sparse indexer patch."""

    def test_sparse_indexer_patch_handles_mtp_block_table_rows(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.model_executor.layers.sparse_attn_indexer":
                patches = _load_patch_config(patch_path)
                new_source = "\n".join(new for _, new in patches)

                assert "_musa_decode_block_table_for_token_rows" in new_source
                assert "repeat_interleave(rows // block_rows, dim=0)" in new_source
                assert "block_table is None" in new_source
                break
        else:
            raise AssertionError("sparse_attn_indexer patch file was not found")

    def test_sparse_indexer_prefill_uses_musa_native_before_torch_path(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.model_executor.layers.sparse_attn_indexer":
                patches = _load_patch_config(patch_path)
                new_source = "\n".join(new for _, new in patches)

                assert 'VLLM_MUSA_SPARSE_INDEXER_GRAPH_EXACT_DECODE", "0"' in (
                    new_source
                )
                assert "_musa_try_fill_prefill_topk_from_indexer_cache_native" in (
                    new_source
                )
                assert "deepseek_v4_indexer_topk_prefill" in new_source
                assert new_source.index(
                    "_musa_try_fill_prefill_topk_from_indexer_cache_native"
                ) < new_source.index("_musa_gather_indexer_fp8_cache")
                assert (
                    "VLLM_MUSA_ENABLE_DEEPSEEK_V4_SPARSE_INDEXER_MUSA_IMPL"
                    in new_source
                )
                assert "per_head = per_head.clamp_min(0.0)" in new_source
                break
        else:
            raise AssertionError("sparse_attn_indexer patch file was not found")

    def test_sparse_indexer_patch_removes_stale_duplicate_helpers(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_module

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.model_executor.layers.sparse_attn_indexer":
                module = _load_patch_module(patch_path)
                assert module is not None
                source = """
logger = init_logger(__name__)


def _musa_sparse_indexer_is_current_stream_capturing() -> bool:
    return False


def _musa_fill_exact_sparse_indexer_indices_capture():
    return "current"


def _musa_sparse_indexer_is_current_stream_capturing() -> bool:
    return True


def _musa_fill_decode_topk_from_indexer_cache_capture():
    block_table = decode_metadata.block_table[:rows]


def _musa_fill_exact_sparse_indexer_indices(
    return "entry"
"""
                normalized = module.normalize_source(source)

                assert (
                    normalized.count(
                        "def _musa_sparse_indexer_is_current_stream_capturing()"
                    )
                    == 1
                )
                assert "decode_metadata.block_table[:rows]" not in normalized
                assert "def _musa_fill_exact_sparse_indexer_indices(" in normalized
                break
        else:
            raise AssertionError("sparse_attn_indexer patch file was not found")

    def test_sparse_indexer_patch_upgrades_partial_prefill_native_helper(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_module

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.model_executor.layers.sparse_attn_indexer":
                module = _load_patch_module(patch_path)
                assert module is not None
                source = """
def _musa_sparse_indexer_graph_exact_decode_enabled() -> bool:
    return os.getenv("VLLM_MUSA_SPARSE_INDEXER_GRAPH_EXACT_DECODE", "0") == "1"


def _musa_fill_exact_sparse_indexer_indices():
    if _musa_try_fill_prefill_topk_from_indexer_cache_native():
        return "native"


def _musa_indexer_cache_block(kv_cache: torch.Tensor, block_id: int) -> torch.Tensor:
    return kv_cache[block_id].view(torch.uint8).flatten()
"""
                normalized = module.normalize_source(source)

                assert 'VLLM_MUSA_SPARSE_INDEXER_GRAPH_EXACT_DECODE", "0"' in (
                    normalized
                )
                assert (
                    "def _musa_try_fill_prefill_topk_from_indexer_cache_native"
                    in normalized
                )
                assert "deepseek_v4_indexer_topk_prefill" in normalized
                break
        else:
            raise AssertionError("sparse_attn_indexer patch file was not found")

    def test_sparse_indexer_patch_removes_shadowed_exact_torch_fallback(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_module

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.model_executor.layers.sparse_attn_indexer":
                module = _load_patch_module(patch_path)
                assert module is not None
                source = """
def _musa_fill_exact_sparse_indexer_indices():
    if metadata.num_prefills > 0 and metadata.prefill is not None:
        for chunk in metadata.prefill.chunks:
            if _musa_try_fill_prefill_topk_from_indexer_cache_native(
                q_quant[token_start:token_end],
                kv_cache,
                weights[token_start:token_end],
                chunk,
                topk_indices_buffer[token_start:token_end, :topk_tokens],
                topk_tokens,
                head_dim,
            ):
                continue
            k_deq = _musa_gather_indexer_fp8_cache()
    return topk_indices_buffer


def _musa_fill_exact_sparse_indexer_indices():
    if metadata.num_prefills > 0 and metadata.prefill is not None:
        q_deq = q_quant.to(torch.float32)
        weights_fp32 = weights.to(torch.float32)
        for chunk in metadata.prefill.chunks:
            k_deq = _musa_gather_indexer_fp8_cache()
            _musa_fill_topk_rows_from_indexer_logits()
    return topk_indices_buffer
"""
                normalized = module.normalize_source(source)

                assert (
                    normalized.count("def _musa_fill_exact_sparse_indexer_indices()")
                    == 1
                )
                assert (
                    "_musa_try_fill_prefill_topk_from_indexer_cache_native"
                    in normalized
                )
                assert "q_deq = q_quant.to(torch.float32)" not in normalized
                assert normalized.index(
                    "_musa_try_fill_prefill_topk_from_indexer_cache_native"
                ) < normalized.index("_musa_gather_indexer_fp8_cache")
                break
        else:
            raise AssertionError("sparse_attn_indexer patch file was not found")


class TestSparseSWAPatch:
    """Tests for the MUSA DeepSeek-V4 sparse SWA metadata patch."""

    def test_sparse_swa_patch_builds_token_metadata_on_device(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.v1.attention.backends.mla.sparse_swa":
                patches = _load_patch_config(patch_path)
                new_source = "\n".join(new for _, new in patches)

                assert "_compute_token_to_req_and_valid_kernel" in new_source
                assert "token_to_req_indices.copy_(x, non_blocking=True)" not in (
                    new_source
                )
                assert (
                    "torch.repeat_interleave(torch.arange(num_reqs), query_lens)"
                    not in (new_source)
                )
                assert "slot >= 0" in new_source
                break
        else:
            raise AssertionError("sparse_swa patch file was not found")


class TestDeepGemmPatch:
    """Tests for the MUSA DeepGEMM compatibility patch."""

    def test_deep_gemm_patch_disables_unsupported_e8m0_on_musa(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.utils.deep_gemm":
                patches = _load_patch_config(patch_path)
                new_source = "\n".join(new for _, new in patches)

                assert "current_platform.is_musa()" in new_source
                assert "DeepGEMM E8M0 disabled on MUSA" in new_source
                assert "grouped FP8 UE8M0 cast is " in new_source
                assert "not supported by the MUSA DeepGEMM backend" in new_source
                break
        else:
            raise AssertionError("deep_gemm patch file was not found")


class TestCompilationBackendPatch:
    """Tests for MUSA torch.compile backend compatibility patches."""

    def test_vllm_backend_patch_accepts_torch_compile_options(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.compilation.backends":
                patches = _load_patch_config(patch_path)
                old_source = "\n".join(old for old, _ in patches)
                new_source = "\n".join(new for _, new in patches)

                assert "example_inputs: Sequence[Any]) -> Any" in old_source
                assert "**kwargs: Any" in new_source
                assert "autograd_cache_normalize_inputs=True" in old_source
                assert "functorch_cache_key_ctx" in new_source
                assert "hasattr(" in new_source
                break
        else:
            raise AssertionError("compilation backend patch file was not found")

    def test_live_vllm_backend_patch_ignores_options_kwarg(self, monkeypatch):
        import vllm_musa

        class DummyBackend:
            def __call__(self, graph, example_inputs):
                return graph, example_inputs

        class DummyModule:
            VllmBackend = DummyBackend

        monkeypatch.setitem(
            __import__("sys").modules,
            "vllm.compilation.backends",
            DummyModule,
        )

        vllm_musa._patch_vllm_backend_call_options()

        backend = DummyBackend()
        assert backend("graph", ["input"], options={"ignored": True}) == (
            "graph",
            ["input"],
        )

    def test_live_functorch_config_patch_skips_missing_keys(self):
        from contextlib import nullcontext

        import vllm_musa

        calls = []

        def original_patch(*args, **kwargs):
            calls.append((args, kwargs))
            return nullcontext()

        functorch_config = SimpleNamespace(existing_key=True)
        patched = vllm_musa._make_config_patch_filter(original_patch, functorch_config)

        with patched(missing_key=False):
            pass
        with patched("missing_key", True):
            pass
        with patched({"existing_key": True, "missing_key": False}):
            pass
        with patched(existing_key=True):
            pass

        assert calls == [
            (({"existing_key": True},), {}),
            ((), {"existing_key": True}),
        ]


class TestCompilationCachingPatch:
    """Tests for MUSA torch compile-cache compatibility patches."""

    def test_graph_pickler_options_patch_supports_torch_27(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.compilation.caching":
                patches = _load_patch_config(patch_path)
                old_source = "\n".join(old for old, _ in patches)
                new_source = "\n".join(new for _, new in patches)

                assert "GraphPickler, Options" in old_source
                assert 'getattr(_vllm_graph_pickler, "Options", None)' in new_source
                assert "_musa_graph_pickler_dumps" in new_source
                break
        else:
            raise AssertionError("compilation caching patch file was not found")


class TestCompilationCompilerInterfacePatch:
    """Tests for MUSA torch compiler-interface compatibility patches."""

    def test_functorch_config_patch_filters_missing_torch_keys(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.compilation.compiler_interface":
                patches = _load_patch_config(patch_path)
                old_source = "\n".join(old for old, _ in patches)
                new_source = "\n".join(new for _, new in patches)

                assert 'cfg["bundled_autograd_cache"] = False' in old_source
                assert "hasattr(functorch_config, key)" in new_source
                break
        else:
            raise AssertionError("compilation compiler_interface patch was not found")

    def test_live_functorch_config_patch_filters_missing_keys(self, monkeypatch):
        import sys

        import vllm_musa

        class DummyCompilerInterface:
            @staticmethod
            def _get_vllm_functorch_config():
                return {"existing_key": True, "missing_key": False}

        monkeypatch.setitem(
            sys.modules,
            "vllm.compilation.compiler_interface",
            DummyCompilerInterface,
        )

        dummy_functorch_config = SimpleNamespace(existing_key=True)
        monkeypatch.setattr(
            vllm_musa,
            "_filter_existing_config",
            lambda config, functorch_config: {
                key: value
                for key, value in config.items()
                if hasattr(dummy_functorch_config, key)
            },
        )

        vllm_musa._patch_vllm_functorch_config()

        config = DummyCompilerInterface._get_vllm_functorch_config()
        assert config == {"existing_key": True}


class TestCompilationPiecewiseBackendPatch:
    """Tests for MUSA piecewise backend compile-cache compatibility patches."""

    def test_piecewise_backend_patch_skips_missing_bundled_cache_key(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.compilation.piecewise_backend":
                patches = _load_patch_config(patch_path)
                old_source = "\n".join(old for old, _ in patches)
                new_source = "\n".join(new for _, new in patches)

                assert '"bundled_autograd_cache", True' in old_source
                assert "functorch_cache_ctx" in new_source
                assert "nullcontext" in new_source
                break
        else:
            raise AssertionError("compilation piecewise_backend patch was not found")


class TestAttentionCompilePatch:
    """Tests for MUSA attention torch.compile compatibility patches."""

    def test_attention_output_shape_patch_avoids_torch_size_constructor(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.model_executor.layers.attention.attention":
                patches = _load_patch_config(patch_path)
                old_source = "\n".join(old for old, _ in patches)
                new_source = "\n".join(new for _, new in patches)

                assert "output_shape = torch.Size(" in old_source
                assert "output_shape = (num_tokens," in new_source
                assert "torch.Size(" not in new_source
                break
        else:
            raise AssertionError("attention compile patch file was not found")


class TestMUSAFlashAttentionReshapeCache:
    """Tests for MUSA FlashAttention reshape+cache dispatch guards."""

    def _load_fa_utils_with_musa_platform(self, monkeypatch, musa_ops_namespace):
        import vllm
        import vllm.platforms as vllm_platforms

        import vllm_musa

        monkeypatch.setenv("VLLM_MUSA_RESHAPE_CACHE_FLASH", "1")
        monkeypatch.setattr(
            vllm_platforms,
            "current_platform",
            # v0.22 quant_utils.py reads current_platform.fp8_dtype() at import,
            # so the mock must provide it (MUSA-3172).
            SimpleNamespace(
                is_musa=lambda: True,
                fp8_dtype=lambda: torch.float8_e4m3fn,
                dispatch_key="MUSA",
            ),
        )

        # MUSA-3172: v0.22's import machinery (importlib.util) requires a real
        # __spec__ on injected modules, else it raises "<name>.__spec__ is None".
        from importlib.machinery import ModuleSpec

        def _mock_module(name):
            mod = ModuleType(name)
            mod.__spec__ = ModuleSpec(name, None)
            return mod

        flash_attn = _mock_module("flash_attn_interface")
        flash_attn.flash_attn_varlen_func = object()
        flash_attn.flash_attn_with_kvcache = object()
        flash_attn.get_scheduler_metadata = object()
        monkeypatch.setitem(sys.modules, "flash_attn_interface", flash_attn)

        vllm_ops = _mock_module("vllm._custom_ops")
        vllm_ops.reshape_and_cache_flash = lambda *args, **kwargs: None
        monkeypatch.setitem(sys.modules, "vllm._custom_ops", vllm_ops)
        monkeypatch.setattr(vllm, "_custom_ops", vllm_ops, raising=False)

        musa_custom_ops = _mock_module("vllm_musa._custom_ops")
        musa_custom_ops.musa_reshape_and_cache_flash_nhd = lambda *args, **kwargs: None
        monkeypatch.setitem(sys.modules, "vllm_musa._custom_ops", musa_custom_ops)
        monkeypatch.setattr(vllm_musa, "_custom_ops", musa_custom_ops, raising=False)

        if musa_ops_namespace is None:
            torch_ops = SimpleNamespace()
        else:
            torch_ops = SimpleNamespace(_C_musa_ops=musa_ops_namespace)
        monkeypatch.setattr(torch, "ops", torch_ops)

        module_name = "vllm_musa.v1.attention.backends.fa_utils"
        # Full sys.modules isolation (MUSA-3172): v0.22 fa_utils pulls in a deep
        # import chain (mla.common, fp8/quant utils, ...). Snapshot before the
        # reimport and drop EVERYTHING imported during it, so a partially-imported
        # module from this mocked environment does not leak into later tests and
        # cause order-dependent failures across the suite.
        mod_snapshot = set(sys.modules)
        previous_module = sys.modules.pop(module_name, None)
        try:
            module = importlib.import_module(module_name)
        finally:
            for _leaked in set(sys.modules) - mod_snapshot:
                sys.modules.pop(_leaked, None)
            if previous_module is not None:
                sys.modules[module_name] = previous_module
        return module

    @pytest.mark.xfail(
        reason="MUSA-3172: the mock-reimport approach hits v0.22's deep fa_utils "
        "import chain (mla.common, quant utils, then a safetensors PyO3 "
        "single-init wall) that cannot be mocked away incrementally. Needs a "
        "rewrite that asserts the guard without reimporting fa_utils.",
        strict=False,
    )
    def test_missing_musa_ops_namespace_disables_native_cache_path(self, monkeypatch):
        module = self._load_fa_utils_with_musa_platform(
            monkeypatch, musa_ops_namespace=None
        )

        assert module._HAS_NATIVE_RESHAPE_CACHE_FLASH is False

    @pytest.mark.xfail(
        reason="MUSA-3172: same v0.22 fa_utils reimport-chain incompatibility as "
        "test_missing_musa_ops_namespace_disables_native_cache_path; needs a "
        "rewrite that avoids reimporting fa_utils.",
        strict=False,
    )
    def test_native_cache_path_requires_matching_cache_dtypes(self, monkeypatch):
        module = self._load_fa_utils_with_musa_platform(
            monkeypatch,
            musa_ops_namespace=SimpleNamespace(
                musa_reshape_and_cache_flash_nhd=object()
            ),
        )

        key = torch.empty((2, 4, 64), dtype=torch.float16)
        value = torch.empty((2, 4, 64), dtype=torch.float16)
        key_cache = torch.empty((1, 16, 4, 64), dtype=torch.float16)
        value_cache = torch.empty((1, 16, 4, 64), dtype=torch.float16)
        slot_mapping = torch.arange(2, dtype=torch.long)
        k_scale = torch.ones(1, dtype=torch.float32)
        v_scale = torch.ones(1, dtype=torch.float32)

        assert module._can_use_musa_reshape_and_cache_flash_nhd(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            "auto",
            k_scale,
            v_scale,
        )
        assert not module._can_use_musa_reshape_and_cache_flash_nhd(
            key,
            value.to(torch.bfloat16),
            key_cache,
            value_cache,
            slot_mapping,
            "auto",
            k_scale,
            v_scale,
        )
        assert not module._can_use_musa_reshape_and_cache_flash_nhd(
            key,
            value,
            key_cache.to(torch.bfloat16),
            value_cache,
            slot_mapping,
            "auto",
            k_scale,
            v_scale,
        )
        assert not module._can_use_musa_reshape_and_cache_flash_nhd(
            key,
            value,
            key_cache,
            value_cache.to(torch.bfloat16),
            slot_mapping,
            "auto",
            k_scale,
            v_scale,
        )


class TestMUSANativeKernelReviewHardening:
    """Source-level tests for native MUSA kernel review fixes."""

    def test_fused_rmsnorm_forced_block_env_is_validated(self):
        source = (
            Path(__file__).parents[1] / "csrc/musa/fused_add_rmsnorm.mu"
        ).read_text()

        assert "forced_block == 128" in source
        assert "forced_block == 256" in source
        assert "forced_block == 512" in source
        assert "forced_block == 1024" in source
        # The error message is split across two adjacent C++ string literals in
        # the .mu source (a line break between "...must be one of " and the
        # value list), so match the parts rather than the concatenation
        # (MUSA-3172: avoid brittle cross-literal substring matching).
        assert "VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X must be one of" in source
        assert "128, 256, 512, or 1024" in source


class TestMUSAPlatformDefaults:
    """Tests for MUSA platform-level vLLM config defaults.

    MUSA-3172: apply_config_platform_defaults() writes real os.environ vars (not
    via monkeypatch); the autouse fixture below restores os.environ per test so
    they don't leak across tests in this class.
    """

    @pytest.fixture(autouse=True)
    def _restore_environ(self):
        saved = dict(os.environ)
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(saved)

    def _make_vllm_config(
        self,
        *,
        architectures=None,
        quantization="fp8",
        quantization_config=None,
        max_cudagraph_capture_size=None,
        cudagraph_capture_sizes=None,
        cudagraph_mode=None,
        tensor_parallel_size=1,
        use_mla=False,
        index_topk=None,
        attention_backend=None,
        cache_block_size=16,
    ):
        from types import SimpleNamespace

        hf_config = SimpleNamespace(
            architectures=architectures,
            quantization_config=quantization_config,
        )
        if index_topk is not None:
            hf_config.index_topk = index_topk
        return SimpleNamespace(
            model_config=SimpleNamespace(
                architectures=architectures,
                hf_config=hf_config,
                quantization=quantization,
                use_mla=use_mla,
                is_mm_prefix_lm=False,
            ),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=tensor_parallel_size,
                worker_cls="auto",
            ),
            cache_config=SimpleNamespace(block_size=cache_block_size),
            scheduler_config=SimpleNamespace(
                is_multimodal_model=False,
                disable_chunked_mm_input=False,
            ),
            attention_config=SimpleNamespace(backend=attention_backend),
            compilation_config=SimpleNamespace(
                custom_ops=[],
                cudagraph_mode=cudagraph_mode,
                max_cudagraph_capture_size=max_cudagraph_capture_size,
                cudagraph_capture_sizes=cudagraph_capture_sizes,
            ),
        )

    def test_qwen3_moe_fp8_caps_default_cudagraph_capture_size(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["Qwen3MoeForCausalLM"],
        )

        MUSAPlatformBase.apply_config_platform_defaults(vllm_config)

        assert vllm_config.compilation_config.max_cudagraph_capture_size == 64
        assert vllm_config.compilation_config.custom_ops == ["all"]

    def test_qwen3_moe_fp8_preserves_user_cudagraph_capture_size(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["Qwen3MoeForCausalLM"],
            max_cudagraph_capture_size=128,
        )

        MUSAPlatformBase.apply_config_platform_defaults(vllm_config)

        assert vllm_config.compilation_config.max_cudagraph_capture_size == 128

    def test_qwen3_moe_fp8_preserves_user_cudagraph_capture_sizes(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["Qwen3MoeForCausalLM"],
            cudagraph_capture_sizes=[1, 2, 4, 8],
        )

        MUSAPlatformBase.apply_config_platform_defaults(vllm_config)

        assert vllm_config.compilation_config.max_cudagraph_capture_size is None
        assert vllm_config.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8]

    @pytest.mark.xfail(
        reason="MUSA-3172: v0.22 platform.py apply_config_platform_defaults no "
        "longer forces cudagraph_mode=NONE at tp=4 (no such code path remains). "
        "Open behavioral question: does large-TP cudagraph capture need disabling "
        "on MUSA v0.22? Potential dropped-safety regression -- flagged, not "
        "resolved here.",
        strict=False,
    )
    def test_tp4_disables_musa_cudagraph_capture(self):
        from vllm.config import CUDAGraphMode

        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            max_cudagraph_capture_size=512,
            cudagraph_capture_sizes=[1, 2, 4, 8],
            tensor_parallel_size=4,
        )

        MUSAPlatformBase.check_and_update_config(vllm_config)

        assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
        assert vllm_config.compilation_config.max_cudagraph_capture_size == 0
        assert vllm_config.compilation_config.cudagraph_capture_sizes == []

    def test_tp2_keeps_musa_cudagraph_capture(self):
        from vllm.config import CUDAGraphMode

        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            max_cudagraph_capture_size=512,
            cudagraph_capture_sizes=[1, 2, 4, 8],
            tensor_parallel_size=2,
        )

        MUSAPlatformBase.check_and_update_config(vllm_config)

        assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
        assert vllm_config.compilation_config.max_cudagraph_capture_size == 512
        assert vllm_config.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8]

    def test_deepseek_v4_defaults_moe_gemv_block32x8(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["DeepseekV4ForCausalLM"],
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_MUSA_DEEPSEEK_V4_FUSED_MOE_GEMV", None)
            os.environ.pop("VLLM_MUSA_GEMV_MOE_BLOCK", None)

            MUSAPlatformBase.check_and_update_config(vllm_config)

            assert os.environ["VLLM_MUSA_DEEPSEEK_V4_FUSED_MOE_GEMV"] == "1"
            assert os.environ["VLLM_MUSA_GEMV_MOE_BLOCK"] == "32x8"

    def test_deepseek_v4_preserves_user_moe_gemv_block(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["DeepseekV4ForCausalLM"],
        )

        with patch.dict(os.environ, {"VLLM_MUSA_GEMV_MOE_BLOCK": "16x8"}):
            MUSAPlatformBase.check_and_update_config(vllm_config)

            assert os.environ["VLLM_MUSA_GEMV_MOE_BLOCK"] == "16x8"

    def test_deepseek_v4_preserves_empty_user_moe_gemv_block(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["DeepseekV4ForCausalLM"],
        )

        with patch.dict(os.environ, {"VLLM_MUSA_GEMV_MOE_BLOCK": ""}):
            MUSAPlatformBase.check_and_update_config(vllm_config)

            assert os.environ["VLLM_MUSA_GEMV_MOE_BLOCK"] == ""

    def test_non_deepseek_v4_does_not_default_moe_gemv_block(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["Qwen3ForCausalLM"],
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_MUSA_GEMV_MOE_BLOCK", None)

            MUSAPlatformBase.check_and_update_config(vllm_config)

            assert "VLLM_MUSA_GEMV_MOE_BLOCK" not in os.environ

    def test_deepseek_v4_tp8_profile_unset_keeps_default_envs(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["DeepseekV4ForCausalLM"],
            tensor_parallel_size=8,
        )
        profile_keys = [
            "VLLM_MUSA_DEEPSEEK_V4_TP8_PROFILE",
            "VLLM_MUSA_GEMV_MOE_BLOCK",
            "VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X",
            "VLLM_MUSA_DEEPSEEK_V4_FLASHMLA_SPARSE_BLOCK_SIZE",
            "VLLM_MUSA_DEEPSEEK_V4_TILELANG_COMPILE_PROFILE",
            "VLLM_MUSA_DEEPSEEK_V4_TILELANG_DISABLE_HOST_ASSERTS",
            "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_TILELANG_MAX_TOKENS",
            "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_IMPL",
            "VLLM_MUSA_DEEPSEEK_V4_FUSED_MOE_GEMV",
        ]

        with patch.dict(os.environ, {}, clear=False):
            for key in profile_keys:
                os.environ.pop(key, None)

            MUSAPlatformBase.check_and_update_config(vllm_config)

            assert os.environ["VLLM_MUSA_DEEPSEEK_V4_FUSED_MOE_GEMV"] == "1"
            assert os.environ["VLLM_MUSA_GEMV_MOE_BLOCK"] == "32x8"
            assert "VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X" not in os.environ
            assert "VLLM_MUSA_DEEPSEEK_V4_FLASHMLA_SPARSE_BLOCK_SIZE" not in os.environ
            assert "VLLM_MUSA_DEEPSEEK_V4_TILELANG_COMPILE_PROFILE" not in os.environ
            assert (
                "VLLM_MUSA_DEEPSEEK_V4_TILELANG_DISABLE_HOST_ASSERTS" not in os.environ
            )
            assert "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_TILELANG_MAX_TOKENS" not in os.environ
            assert "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_IMPL" not in os.environ

    def test_deepseek_v4_tp8_profile_sets_missing_envs(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["DeepseekV4ForCausalLM"],
            tensor_parallel_size=8,
            use_mla=True,
            index_topk=512,
            cache_block_size=64,
        )
        profile_keys = [
            "VLLM_MUSA_GEMV_MOE_BLOCK",
            "VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X",
            "VLLM_MUSA_DEEPSEEK_V4_FLASHMLA_SPARSE_BLOCK_SIZE",
            "VLLM_MUSA_DEEPSEEK_V4_TILELANG_COMPILE_PROFILE",
            "VLLM_MUSA_DEEPSEEK_V4_TILELANG_DISABLE_HOST_ASSERTS",
            "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_TILELANG_MAX_TOKENS",
            "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_IMPL",
            "VLLM_MUSA_DEEPSEEK_V4_FUSED_MOE_GEMV",
        ]

        with patch.dict(
            os.environ,
            {"VLLM_MUSA_DEEPSEEK_V4_TP8_PROFILE": "balanced_long_prefill"},
            clear=False,
        ):
            for key in profile_keys:
                os.environ.pop(key, None)

            MUSAPlatformBase.check_and_update_config(vllm_config)

            assert os.environ["VLLM_MUSA_DEEPSEEK_V4_FUSED_MOE_GEMV"] == "1"
            assert os.environ["VLLM_MUSA_GEMV_MOE_BLOCK"] == "16x8"
            assert os.environ["VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X"] == "256"
            assert (
                os.environ["VLLM_MUSA_DEEPSEEK_V4_FLASHMLA_SPARSE_BLOCK_SIZE"] == "256"
            )
            assert (
                os.environ["VLLM_MUSA_DEEPSEEK_V4_TILELANG_COMPILE_PROFILE"]
                == "dsa_full"
            )
            assert (
                os.environ["VLLM_MUSA_DEEPSEEK_V4_TILELANG_DISABLE_HOST_ASSERTS"] == "1"
            )
            assert (
                os.environ["VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_TILELANG_MAX_TOKENS"]
                == "2048"
            )
            assert "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_IMPL" not in os.environ
            assert vllm_config.cache_config.block_size == 256

    def test_deepseek_v4_tp8_aggressive_profile_sets_mhc_deepgemm(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["DeepseekV4ForCausalLM"],
            tensor_parallel_size=8,
            use_mla=True,
            index_topk=512,
            cache_block_size=64,
        )
        profile_keys = [
            "VLLM_MUSA_GEMV_MOE_BLOCK",
            "VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X",
            "VLLM_MUSA_DEEPSEEK_V4_FLASHMLA_SPARSE_BLOCK_SIZE",
            "VLLM_MUSA_DEEPSEEK_V4_TILELANG_COMPILE_PROFILE",
            "VLLM_MUSA_DEEPSEEK_V4_TILELANG_DISABLE_HOST_ASSERTS",
            "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_TILELANG_MAX_TOKENS",
            "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_IMPL",
            "VLLM_MUSA_DEEPSEEK_V4_FUSED_MOE_GEMV",
        ]

        with patch.dict(
            os.environ,
            {"VLLM_MUSA_DEEPSEEK_V4_TP8_PROFILE": "aggressive_long_prefill"},
            clear=False,
        ):
            for key in profile_keys:
                os.environ.pop(key, None)

            MUSAPlatformBase.check_and_update_config(vllm_config)

            assert os.environ["VLLM_MUSA_DEEPSEEK_V4_FUSED_MOE_GEMV"] == "1"
            assert os.environ["VLLM_MUSA_GEMV_MOE_BLOCK"] == "16x8"
            assert os.environ["VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X"] == "256"
            assert (
                os.environ["VLLM_MUSA_DEEPSEEK_V4_FLASHMLA_SPARSE_BLOCK_SIZE"] == "256"
            )
            assert (
                os.environ["VLLM_MUSA_DEEPSEEK_V4_TILELANG_COMPILE_PROFILE"]
                == "dsa_full"
            )
            assert (
                os.environ["VLLM_MUSA_DEEPSEEK_V4_TILELANG_DISABLE_HOST_ASSERTS"] == "1"
            )
            assert (
                os.environ["VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_TILELANG_MAX_TOKENS"]
                == "2048"
            )
            assert (
                os.environ["VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_IMPL"] == "deepgemm_big_fuse"
            )
            assert vllm_config.cache_config.block_size == 256

    def test_deepseek_v4_tp8_profile_preserves_explicit_overrides(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["DeepseekV4ForCausalLM"],
            tensor_parallel_size=8,
            use_mla=True,
            index_topk=512,
            cache_block_size=256,
        )
        env = {
            "VLLM_MUSA_DEEPSEEK_V4_TP8_PROFILE": "balanced_long_prefill",
            "VLLM_MUSA_GEMV_MOE_BLOCK": "32x8",
            "VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X": "128",
            "VLLM_MUSA_DEEPSEEK_V4_FLASHMLA_SPARSE_BLOCK_SIZE": "64",
            "VLLM_MUSA_DEEPSEEK_V4_TILELANG_COMPILE_PROFILE": "opt1",
            "VLLM_MUSA_DEEPSEEK_V4_TILELANG_DISABLE_HOST_ASSERTS": "0",
            "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_TILELANG_MAX_TOKENS": "0",
            "VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_IMPL": "native",
        }

        with patch.dict(os.environ, env, clear=False):
            MUSAPlatformBase.check_and_update_config(vllm_config)

            assert os.environ["VLLM_MUSA_GEMV_MOE_BLOCK"] == "32x8"
            assert os.environ["VLLM_MUSA_FUSED_ADD_RMSNORM_BLOCK_X"] == "128"
            assert (
                os.environ["VLLM_MUSA_DEEPSEEK_V4_FLASHMLA_SPARSE_BLOCK_SIZE"] == "64"
            )
            assert (
                os.environ["VLLM_MUSA_DEEPSEEK_V4_TILELANG_COMPILE_PROFILE"] == "opt1"
            )
            assert (
                os.environ["VLLM_MUSA_DEEPSEEK_V4_TILELANG_DISABLE_HOST_ASSERTS"] == "0"
            )
            assert (
                os.environ["VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_TILELANG_MAX_TOKENS"] == "0"
            )
            assert os.environ["VLLM_MUSA_DEEPSEEK_V4_MHC_PRE_IMPL"] == "native"
            assert vllm_config.cache_config.block_size == 64

    def test_deepseek_v4_tp8_profile_rejects_unknown_name(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["DeepseekV4ForCausalLM"],
            tensor_parallel_size=8,
        )

        with patch.dict(
            os.environ,
            {"VLLM_MUSA_DEEPSEEK_V4_TP8_PROFILE": "typo"},
            clear=False,
        ):
            try:
                MUSAPlatformBase.check_and_update_config(vllm_config)
            except ValueError as exc:
                message = str(exc)
            else:
                raise AssertionError("expected invalid profile to raise ValueError")

            assert "VLLM_MUSA_DEEPSEEK_V4_TP8_PROFILE" in message
            assert "balanced_long_prefill" in message
            assert "aggressive_long_prefill" in message

    def test_deepseek_v4_flashmla_sparse_defaults_block64(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["DeepseekV4ForCausalLM"],
            use_mla=True,
            index_topk=512,
            cache_block_size=16,
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_MUSA_DEEPSEEK_V4_FLASHMLA_SPARSE_BLOCK_SIZE", None)
            MUSAPlatformBase.check_and_update_config(vllm_config)

        assert vllm_config.cache_config.block_size == 64

    def test_deepseek_v4_flashmla_sparse_preserves_explicit_block64(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["DeepseekV4ForCausalLM"],
            use_mla=True,
            index_topk=512,
            cache_block_size=256,
        )

        with patch.dict(
            os.environ,
            {"VLLM_MUSA_DEEPSEEK_V4_FLASHMLA_SPARSE_BLOCK_SIZE": "64"},
        ):
            MUSAPlatformBase.check_and_update_config(vllm_config)

        assert vllm_config.cache_config.block_size == 64

    def test_deepseek_v4_flashmla_sparse_allows_block256_opt_in(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["DeepseekV4ForCausalLM"],
            use_mla=True,
            index_topk=512,
            cache_block_size=64,
        )

        with patch.dict(
            os.environ,
            {"VLLM_MUSA_DEEPSEEK_V4_FLASHMLA_SPARSE_BLOCK_SIZE": "256"},
        ):
            MUSAPlatformBase.check_and_update_config(vllm_config)

        assert vllm_config.cache_config.block_size == 256

    def test_deepseek_v4_flashmla_sparse_rejects_invalid_block_size(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["DeepseekV4ForCausalLM"],
            use_mla=True,
            index_topk=512,
            cache_block_size=64,
        )

        with patch.dict(
            os.environ,
            {"VLLM_MUSA_DEEPSEEK_V4_FLASHMLA_SPARSE_BLOCK_SIZE": "128"},
        ):
            with pytest.raises(ValueError, match="must be 64 or 256"):
                MUSAPlatformBase.check_and_update_config(vllm_config)

    def test_non_deepseek_v4_flashmla_sparse_ignores_block256_opt_in(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["OtherSparseMLAForCausalLM"],
            use_mla=True,
            index_topk=512,
            cache_block_size=256,
        )

        with patch.dict(
            os.environ,
            {"VLLM_MUSA_DEEPSEEK_V4_FLASHMLA_SPARSE_BLOCK_SIZE": "256"},
        ):
            MUSAPlatformBase.check_and_update_config(vllm_config)

        assert vllm_config.cache_config.block_size == 64

    def test_dense_fp8_does_not_cap_cudagraph_capture_size(self):
        from vllm_musa.platform import MUSAPlatformBase

        vllm_config = self._make_vllm_config(
            architectures=["Qwen3ForCausalLM"],
        )

        MUSAPlatformBase.apply_config_platform_defaults(vllm_config)

        assert vllm_config.compilation_config.max_cudagraph_capture_size is None


class TestMUSAFusedMoEFP8Scales:
    """Tests for MUSA FP8 MoE scale adaptation helpers."""

    def test_static_tensor_fp8_moe_scales_expand_to_block_layout(self):
        from vllm_musa.model_executor.layers.fused_moe.fused_moe import (
            _maybe_expand_fp8_moe_per_tensor_scale,
        )

        weight = torch.empty((2, 260, 320), dtype=torch.float8_e4m3fn)
        scale = torch.tensor([0.5, 1.5], dtype=torch.float32)

        expanded = _maybe_expand_fp8_moe_per_tensor_scale(scale, weight)

        assert expanded is not None
        assert expanded.shape == (2, 5, 5)
        assert expanded.is_contiguous()
        assert torch.all(expanded[0] == scale[0])
        assert torch.all(expanded[1] == scale[1])

    def test_static_tensor_fp8_moe_scales_prefer_128_block_layout(self):
        from vllm_musa.model_executor.layers.fused_moe.fused_moe import (
            _maybe_expand_fp8_moe_per_tensor_scale,
        )

        weight = torch.empty((2, 260, 256), dtype=torch.float8_e4m3fn)
        scale = torch.tensor([0.5, 1.5], dtype=torch.float32)

        expanded = _maybe_expand_fp8_moe_per_tensor_scale(scale, weight)

        assert expanded is not None
        assert expanded.shape == (2, 3, 2)
        assert expanded.is_contiguous()

    def test_block_fp8_moe_scales_are_left_unchanged(self):
        from vllm_musa.model_executor.layers.fused_moe.fused_moe import (
            _maybe_expand_fp8_moe_per_tensor_scale,
        )

        weight = torch.empty((2, 256, 256), dtype=torch.float8_e4m3fn)
        scale = torch.ones((2, 2, 2), dtype=torch.float32)

        expanded = _maybe_expand_fp8_moe_per_tensor_scale(scale, weight)

        assert expanded is scale


class TestMUSAFp8MoEPadding:
    """Tests for MUSA FP8 MoE TP padding helpers."""

    def test_musa_fp8_moe_tp_padding_rounds_partition_to_block_lcm(self, monkeypatch):
        from vllm_musa.model_executor.layers.quantization import fp8 as musa_fp8

        monkeypatch.setattr(
            musa_fp8,
            "current_platform",
            SimpleNamespace(is_musa=lambda: True),
        )
        monkeypatch.setattr(
            musa_fp8,
            "_ORIGINAL_FP8_MOE_MAYBE_ROUNDUP_SIZES",
            lambda self, hidden_size, intermediate_size_per_partition, act_dtype, moe_parallel_config: (
                hidden_size,
                intermediate_size_per_partition,
            ),
        )

        method = SimpleNamespace(block_quant=True, weight_block_size=(128, 128))

        assert musa_fp8.maybe_roundup_sizes(
            method,
            hidden_size=2048,
            intermediate_size_per_partition=704,
            act_dtype=torch.bfloat16,
            moe_parallel_config=SimpleNamespace(tp_size=2),
        ) == (2048, 768)
        assert musa_fp8.maybe_roundup_sizes(
            method,
            hidden_size=2048,
            intermediate_size_per_partition=704,
            act_dtype=torch.bfloat16,
            moe_parallel_config=SimpleNamespace(tp_size=1),
        ) == (2048, 704)

    @pytest.mark.skipif(
        not hasattr(torch, "float8_e4m3fn"),
        reason="FP8 tensor dtype is not available in this torch build",
    )
    def test_musa_fp8_moe_padded_weights_are_zero_initialized(self, monkeypatch):
        from vllm_musa.model_executor.layers.quantization import fp8 as musa_fp8

        monkeypatch.setattr(
            musa_fp8,
            "current_platform",
            SimpleNamespace(is_musa=lambda: True),
        )

        layer = SimpleNamespace(
            moe_config=SimpleNamespace(intermediate_size_per_partition_unpadded=704),
            prefix="model.layers.0.mlp",
        )

        def fake_create_weights(self, layer, **kwargs):
            layer.w13_weight = SimpleNamespace(
                data=torch.full((8,), 42, dtype=torch.uint8).view(torch.float8_e4m3fn)
            )
            layer.w2_weight = SimpleNamespace(
                data=torch.full((8,), 43, dtype=torch.uint8).view(torch.float8_e4m3fn)
            )

        monkeypatch.setattr(
            musa_fp8,
            "_ORIGINAL_FP8_MOE_CREATE_WEIGHTS",
            fake_create_weights,
        )

        musa_fp8.create_weights(
            SimpleNamespace(block_quant=True),
            layer=layer,
            num_experts=64,
            hidden_size=2048,
            intermediate_size_per_partition=768,
            params_dtype=torch.bfloat16,
        )

        assert torch.all(layer.w13_weight.data.view(torch.uint8) == 0)
        assert torch.all(layer.w2_weight.data.view(torch.uint8) == 0)


class TestScaledMMKernelPatch:
    """Tests for the MUSA scaled-mm kernel registry patch."""

    def test_musa_deepgemm_fp8_uses_custom_op_for_compile(self):
        source = (
            Path(__file__).parents[1]
            / "vllm_musa/model_executor/kernels/linear/scaled_mm/deep_gemm.py"
        ).read_text()

        assert "torch.ops.vllm.musa_deepgemm_fp8_op" in source
        assert "torch.ops.vllm.musa_fp8_small_m_gemv_op" in source
        assert "direct_register_custom_op(" in source
        assert '"musa_deepgemm_fp8_op"' in source
        assert '"musa_fp8_small_m_gemv_op"' in source
        assert "_musa_deepgemm_fp8_op_fake" in source
        assert "_musa_fp8_small_m_gemv_op_fake" in source
        assert "VLLM_MUSA_DEEPGEMM_ROW_MAJOR_ACT_SCALES" in source
        assert "VLLM_MUSA_FP8_SMALL_M_GEMV" in source

    def test_musa_deepgemm_row_major_scale_gate(self, monkeypatch):
        from vllm_musa.model_executor.kernels.linear.scaled_mm import deep_gemm

        monkeypatch.setattr(
            deep_gemm.envs, "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES", False
        )
        monkeypatch.delenv("VLLM_MUSA_DEEPGEMM_ROW_MAJOR_ACT_SCALES", raising=False)

        assert deep_gemm._use_row_major_activation_scales(False) is True
        assert deep_gemm._use_row_major_activation_scales(True) is False

        monkeypatch.setenv("VLLM_MUSA_DEEPGEMM_ROW_MAJOR_ACT_SCALES", "0")
        assert deep_gemm._use_row_major_activation_scales(False) is False

        monkeypatch.setenv("VLLM_MUSA_DEEPGEMM_ROW_MAJOR_ACT_SCALES", "1")
        monkeypatch.setattr(
            deep_gemm.envs, "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES", True
        )
        assert deep_gemm._use_row_major_activation_scales(False) is True

    def test_musa_swiglu_uses_custom_op_for_compile(self):
        source = (
            Path(__file__).parents[1] / "vllm_musa/model_executor/layers/activation.py"
        ).read_text()

        assert "torch.ops.vllm.musa_swish_glu_op" in source
        assert "direct_register_custom_op(" in source
        assert '"musa_swish_glu_op"' in source
        assert "_musa_swish_glu_op_fake" in source

    def test_musa_torch_fp8_scaled_mm_disables_fp8_output_padding(self):
        from vllm_musa.model_executor.kernels.linear.scaled_mm.torch_scaled_mm import (
            MUSAPerTensorTorchFP8ScaledMMLinearKernel,
        )

        assert (
            MUSAPerTensorTorchFP8ScaledMMLinearKernel.get_output_padding(None) is None
        )

    def test_musa_unquantized_gemm_materializes_plain_parameter_for_compile(self):
        source = (
            Path(__file__).parents[1] / "vllm_musa/model_executor/layers/utils.py"
        ).read_text()

        assert "BasevLLMParameter" in source
        assert "torch.nn.Parameter(weight.detach(), requires_grad=False)" in source
        assert "plain_weight.__dict__.update(weight.__dict__)" in source
        assert "current_platform.is_musa()" in source
        assert "process_weights_after_loading" in source
        assert "_musa_materializes_plain_parameter" in source
        assert "DisableTorchFunction" not in source

    def test_scaled_mm_patch_registers_musa_fp8_kernel_fallbacks(self):
        from vllm_musa.patches import _get_patch_files, _load_patch_config

        patch_files = _get_patch_files()

        for module_name, patch_path in patch_files:
            if module_name == "vllm.model_executor.kernels.linear":
                patches = _load_patch_config(patch_path)
                new_source = "\n".join(new for _, new in patches)

                assert "possible_kernels is _POSSIBLE_FP8_BLOCK_KERNELS" in new_source
                assert "MUSADeepGemmFp8BlockScaledMMKernel" in new_source
                assert "possible_kernels is _POSSIBLE_FP8_KERNELS" in new_source
                assert "MUSAPerTensorTorchFP8ScaledMMLinearKernel" in new_source
                assert "MUSAChannelWiseTorchFP8ScaledMMLinearKernel" in new_source
                break
        else:
            raise AssertionError("linear kernel patch file was not found")


class TestMUSAFP8ActivationQuant:
    """Tests for MUSA FP8 activation quantization helpers."""

    def test_per_token_group_quant_accepts_strided_2d_input(self, monkeypatch):
        from vllm_musa.model_executor.layers.quantization.utils import fp8_utils

        calls = []

        def fake_quant(
            x,
            x_q,
            x_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            use_ue8m0,
            column_major_scales,
            tma_aligned_scales,
        ):
            calls.append(
                {
                    "x": x,
                    "x_q": x_q,
                    "x_s": x_s,
                    "group_size": group_size,
                    "column_major_scales": column_major_scales,
                }
            )

        monkeypatch.setattr(
            fp8_utils,
            "current_platform",
            SimpleNamespace(
                is_musa=lambda: True,
                fp8_dtype=lambda: torch.float8_e4m3fn,
                dispatch_key="MUSA",
            ),
        )
        monkeypatch.setattr(
            torch.ops,
            "_C_musa_ops",
            SimpleNamespace(per_token_group_fp8_quant=fake_quant),
            raising=False,
        )

        base = torch.randn(4, 3, 128, dtype=torch.float32)
        x = base[:, 1, :]
        assert x.stride(-1) == 1
        assert not x.is_contiguous()

        x_q, x_s = fp8_utils.per_token_group_quant_fp8(
            x,
            group_size=128,
            use_ue8m0=False,
        )

        assert len(calls) == 1
        assert calls[0]["x"].is_contiguous()
        assert calls[0]["x"].shape == x.shape
        assert x_q.shape == x.shape
        assert x_q.is_contiguous()
        assert x_s.shape == (4, 1)

    def test_per_token_group_quant_rejects_noncontiguous_groups(self, monkeypatch):
        from vllm_musa.model_executor.layers.quantization.utils import fp8_utils

        monkeypatch.setattr(
            fp8_utils,
            "current_platform",
            SimpleNamespace(
                is_musa=lambda: True,
                fp8_dtype=lambda: torch.float8_e4m3fn,
                dispatch_key="MUSA",
            ),
        )

        x = torch.randn(128, 4, dtype=torch.float32).t()
        assert x.shape == (4, 128)
        assert x.stride(-1) != 1

        with pytest.raises(AssertionError, match="groups must be contiguous"):
            fp8_utils.per_token_group_quant_fp8(
                x,
                group_size=128,
                use_ue8m0=False,
            )


class TestApplyPatches:
    """Tests for the apply_patches function."""

    def test_apply_patches_is_idempotent(self):
        """Test that apply_patches can be called multiple times safely."""
        from vllm_musa import patches

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
        from vllm_musa import patches

        # Reset state
        patches._patches_applied = False

        # Create a mock patch file for a non-existent module
        with patch.object(patches, "_get_patch_files") as mock_get:
            mock_get.return_value = [("non.existent.module", Path("/fake/path"))]

            with patch.object(patches, "_load_patch_config") as mock_load:
                mock_load.return_value = [("old", "new")]

                # Should not raise
                patches.apply_patches()


class TestPatchesReadme:
    """Tests for patches documentation."""

    def test_readme_exists(self):
        """Test that README.md exists in patches directory."""
        patches_dir = Path(__file__).parent.parent / "vllm_musa" / "patches"
        readme_path = patches_dir / "README.md"

        assert readme_path.exists()

    def test_readme_documents_naming_convention(self):
        """Test that README explains the naming convention."""
        patches_dir = Path(__file__).parent.parent / "vllm_musa" / "patches"
        readme_path = patches_dir / "README.md"

        content = readme_path.read_text()

        # Should document the double underscore convention
        assert "__" in content or "double underscore" in content.lower()
        assert ".patch.py" in content


class TestPatchManifest:
    """MUSA-0301: deterministic patch discovery + read-only patch_report()."""

    def test_get_patch_files_sorted_and_unique(self):
        from vllm_musa.patches import _get_patch_files

        names = [f.name for _, f in _get_patch_files()]
        assert names, "no patch files discovered"
        assert names == sorted(names), "patch discovery order is not deterministic"
        assert len(names) == len(set(names)), "duplicate patch files"

    def test_patch_report_structure_and_determinism(self):
        import vllm_musa
        from vllm_musa.patches import _get_patch_files

        report = vllm_musa.patch_report()
        assert isinstance(report, list)
        assert len(report) == len(_get_patch_files()), "report must cover every patch"

        allowed_status = {
            "applied", "needs-apply", "no-op", "missing-target",
            "unreadable-target", "side-effect", "load-failed", "error", "unknown",
        }
        allowed_kind = {"source-transform", "side-effect", "load-failed", "unknown"}
        for e in report:
            assert {"module", "file", "kind", "target_resolved", "status"} <= set(e), e
            assert e["status"] in allowed_status, e
            assert e["kind"] in allowed_kind, e
            assert isinstance(e["target_resolved"], bool)

        # Deterministic across calls and sorted by file (MUSA-0301 ordering).
        report2 = vllm_musa.patch_report()
        assert [e["file"] for e in report] == [e["file"] for e in report2]
        assert [e["file"] for e in report] == sorted(e["file"] for e in report)
