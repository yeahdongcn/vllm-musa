# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.worker.worker to support MUSA device type.

The vLLM V0 worker only checks for device_config.device.type == "cuda",
which doesn't match for MUSA devices. This patch extends the check
to also accept "musa" device type.

Note: No patch is needed for torch.device("cuda:X") because torchada
automatically aliases it to torch.device("musa:X") when imported.
"""

PATCHES = [
    # Patch init_device to support musa device type (V0 worker)
    (
        'if self.device_config.device.type == "cuda":',
        'if self.device_config.device.type in ("cuda", "musa"):',
    ),
]

