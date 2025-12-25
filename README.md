# vLLM MUSA Platform Plugin

A vLLM platform plugin that enables running vLLM on Moore Threads MUSA GPUs.

## Overview

This plugin provides MUSA (Moore Threads Unified Software Architecture) support for vLLM through:

- **torchada**: CUDA→MUSA compatibility layer for PyTorch
- **pymtml**: Moore Threads Management Library for device queries
- **Triton patches**: Compatibility fixes for MUSA's Triton compiler

## Requirements

- Python 3.9+
- vLLM
- Moore Threads GPU with MUSA toolkit installed
- torchada (CUDA→MUSA compatibility)
- mthreads-ml-py (pymtml - MTML bindings)

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/vllm-project/vllm-musa.git
cd vllm-musa

# Install in development mode
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

### From PyPI (when published)

```bash
pip install vllm-musa
```

## Verification

After installation, verify the plugin is registered:

```bash
python -c "from vllm_musa_platform import musa_platform_plugin; print('Plugin loaded successfully')"
```

Check if MTML (device management) is available:

```bash
python -c "from vllm_musa_platform import mtml; print(f'MTML available: {mtml.is_mtml_available()}')"
```

## Usage

Once installed, the plugin is automatically detected by vLLM. Simply run vLLM as usual:

```python
from vllm import LLM, SamplingParams

# vLLM will automatically use the MUSA platform
llm = LLM(model="your-model-path", trust_remote_code=True)

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)
outputs = llm.generate(["Hello, how are you?"], sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### Environment Variables

- `MUSA_VISIBLE_DEVICES`: Control which MUSA devices are visible (similar to `CUDA_VISIBLE_DEVICES`)
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`: Recommended for multi-process workers

### Example

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn python -c "
from vllm import LLM, SamplingParams

llm = LLM(model='/path/to/model', trust_remote_code=True, enforce_eager=True)
outputs = llm.generate(['Hello!'], SamplingParams(max_tokens=20))
print(outputs[0].outputs[0].text)
"
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_mtml.py -v
pytest tests/test_musa.py -v
pytest tests/test_patches.py -v

# Run with coverage
pytest tests/ -v --cov=vllm_musa_platform --cov-report=term-missing
```

## Project Structure

```
vllm-musa/
├── pyproject.toml              # Project configuration
├── README.md                   # This file
├── vllm_musa_platform/         # Main package
│   ├── __init__.py             # Plugin entry point
│   ├── mtml.py                 # MTML wrapper (device management)
│   ├── musa.py                 # MUSA platform implementation
│   └── patches/                # Triton compatibility patches
│       ├── __init__.py         # Patch application logic
│       ├── README.md           # Patch documentation
│       └── vllm__attention__ops__triton_unified_attention.patch.py
└── tests/                      # Test suite
    ├── conftest.py             # Pytest fixtures
    ├── test_mtml.py            # MTML wrapper tests
    ├── test_musa.py            # Platform tests
    └── test_patches.py         # Patch system tests
```

## Patches

The plugin includes runtime patches for vLLM compatibility with MUSA's Triton compiler. See [patches/README.md](vllm_musa_platform/patches/README.md) for details.

## License

Apache-2.0

## Contributing

Contributions are welcome! Please ensure all tests pass before submitting:

```bash
# Run tests
pytest tests/ -v

# Run linter (if ruff is installed)
ruff check .

# Run type checker (if mypy is installed)
mypy vllm_musa_platform/
```

