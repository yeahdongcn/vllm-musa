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

### Unit Tests

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

### Supported vLLM Versions

This plugin supports multiple vLLM versions:

| vLLM Version | PyTorch Version | Engine | Status |
|--------------|-----------------|--------|--------|
| 0.10.1.1     | 2.7.1           | V0/V1  | ✅ Supported |
| 0.13.0       | 2.7.1           | V1 only | ✅ Supported |

### Testing with Different vLLM Versions

#### vLLM 0.10.1.1 (with torch 2.7.1)

```bash
# Install the plugin (vLLM 0.10.1.1 is installed automatically as a dependency)
pip install -e .

# Start the server
vllm serve /path/to/model/

# In another terminal, test inference
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/path/to/model/", "prompt": "Hello!", "max_tokens": 50}'
```

#### vLLM 0.13.0 (with torch 2.7.1)

> **Important:** Use `--no-deps` when upgrading vLLM to prevent torch from being replaced.
> The MUSA container includes a pre-configured torch 2.7.1 that must not be overwritten.

```bash
# Install the plugin (vLLM 0.10.1.1 is installed automatically as a dependency)
pip install -e .

# Upgrade to vLLM 0.13.0 without reinstalling dependencies
pip install vllm==0.13.0 --no-deps --upgrade

# Install additional dependencies required by vLLM 0.13.0
pip install 'depyf==0.20.0' 'llguidance>=1.3.0,<1.4.0' \
            'lm-format-enforcer==0.11.3' 'outlines_core==0.2.11' \
            'xgrammar==0.1.27' 'compressed-tensors==0.12.2' \
            'model-hosting-container-standards<1.0.0,>=0.1.9' \
            ijson anthropic mcp

# Start the server
vllm serve /path/to/model/

# In another terminal, test inference
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/path/to/model/", "prompt": "Hello!", "max_tokens": 50}'

# Test chat completions
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/path/to/model/", "messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 50}'
```

### Version-Specific Notes

#### vLLM 0.10.x
- Supports both V0 and V1 engines
- Uses `VLLM_USE_V1=1` environment variable to enable V1 engine
- The `vllm.worker.worker` module exists for V0 engine support

#### vLLM 0.13.x
- V1 is the default (and only) engine
- The `vllm.worker` module was removed (V0 engine deprecated)
- Requires additional dependencies: `depyf`, `llguidance`, `lm-format-enforcer`, `outlines_core`, `xgrammar`, `compressed-tensors`

### Docker Testing

For containerized testing with MUSA GPUs:

```bash
# Start a container with MUSA support
docker run -d --net host --privileged --pid=host --shm-size 500g \
  -v $PWD:/ws -w /ws \
  -v /data/vllm:/home/dist \
  --name musa-test \
  sh-harbor.mthreads.com/mcctest/musa-compile:rc4.3.3-torch2.7-20251120 \
  sleep infinity

# Enter the container
docker exec -it musa-test bash

# Inside the container, install and test
pip install -e /ws
vllm serve /home/dist/your-model/
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

