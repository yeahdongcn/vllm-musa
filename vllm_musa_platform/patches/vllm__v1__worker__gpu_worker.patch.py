# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.v1.worker.gpu_worker to support MUSA device type.

The vLLM V1 GPU worker only checks for device_config.device.type == "cuda",
which doesn't match for MUSA devices. This patch extends the check
to also accept "musa" device type.
"""

PATCHES = [
    # Patch init_device to support musa device type (V1 worker)
    # For vLLM 0.10.x
    (
        'if self.device_config.device.type == "cuda":',
        'if self.device_config.device.type in ("cuda", "musa"):',
    ),
    # For vLLM 0.13+
    (
        'if isinstance(device, torch.device) and device.type == "cuda":',
        'if isinstance(device, torch.device) and device.type in ("cuda", "musa"):',
    ),
    # Use platform-specific device string instead of hardcoded "cuda"
    (
        'self.device = torch.device(f"cuda:{self.local_rank}")',
        'self.device = torch.device(f"{self.device_config.device.type}:{self.local_rank}")',
    ),
]

