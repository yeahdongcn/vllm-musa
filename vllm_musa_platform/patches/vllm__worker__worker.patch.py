# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.worker.worker to support MUSA device type.

The vLLM worker only checks for device_config.device.type == "cuda",
which doesn't match for MUSA devices. This patch extends the check
to also accept "musa" device type.
"""

PATCHES = [
    # Patch init_device to support musa device type (V0 worker)
    (
        'if self.device_config.device.type == "cuda":',
        'if self.device_config.device.type in ("cuda", "musa"):',
    ),
    # Use platform-specific device string instead of hardcoded "cuda"
    (
        'self.device = torch.device(f"cuda:{self.local_rank}")',
        'self.device = torch.device(f"{self.device_config.device.type}:{self.local_rank}")',
    ),
]

