# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.attention.ops.triton_unified_attention

This patch fixes Triton syntax incompatibility with MUSA's Triton version.

Issue:
------
The MUSA Triton compiler doesn't support Python's annotated assignment syntax
(PEP 526) in Triton kernels. Specifically, the syntax:

    left: tl.int32 = 0

causes a compilation error:
    AttributeError("'AnnAssign' object has no attribute 'targets'")

Solution:
---------
Replace annotated assignments with plain assignments:

    left = 0

The type annotation is not required for the Triton kernel to function
correctly, as Triton infers types from usage.

Affected function:
------------------
find_seq_idx() - Binary search helper for unified attention kernel
"""

# List of (old_string, new_string) replacements
PATCHES = [
    # Fix annotated assignment in find_seq_idx function
    ("left: tl.int32 = 0", "left = 0"),
]

