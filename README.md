# pylocalvqe

C++ and Python interfaces for [LocalVQE](https://github.com/localai-org/LocalVQE),
a ggml-based voice quality enhancement engine providing acoustic echo
cancellation, noise suppression and dereverberation.

- Stream 16 kHz mono audio in 256-sample (16 ms) hops.
- Use v1.3, v1.4-AEC 200K or v1.4-AEC 2.7K models.
- Read each model's algorithmic delay for downstream compensation.
- Use the same shared core from Python or C++20 through a flat C ABI.
- Select CPU or CUDA at runtime with one installed package.
- Download model weights separately and verify their SHA-256 digests.

## Install

Install the wheel from [PyPI](https://pypi.org/project/pylocalvqe/) into a
CPython 3.12 or later environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: pylocalvqe
```

Pin a release when reproducibility matters, for example `pylocalvqe==0.4.0`.
One Linux x86_64 wheel contains both CPU and CUDA 13 backends. It uses
CPython's stable ABI and the `manylinux_2_28_x86_64` platform tag, for example
`pylocalvqe-0.4.0-cp312-abi3-manylinux_2_28_x86_64.whl`.

The host needs glibc 2.28 or later and a CPU with SSE4.2, AVX, AVX2, FMA,
F16C and BMI2. AVX-512 is unnecessary. NumPy 1.26 or later and the CUDA
user-space libraries are installed automatically as Python dependencies,
including when only CPU processing is used. The pinned CUDA distributions
are `nvidia-cuda-runtime==13.3.29` and `nvidia-cublas==13.6.0.2`; cuBLAS also
brings its NVRTC dependency. No compiler, CMake, CUDA toolkit or OpenMP
runtime is required to install or use the package.

Import and `device="cpu"` processing work without an NVIDIA driver or GPU.
To use `device="cuda"`, provide a supported NVIDIA GPU and a compatible host
driver. The supported Linux driver floor is 610.43.02 for CUDA 13.3; see the
[NVIDIA release notes](https://docs.nvidia.com/cuda/archive/13.3.1/cuda-toolkit-release-notes/index.html)
and the selected pylocalvqe release's validated driver information. WSL2
uses the compatible NVIDIA driver installed on Windows. The wheel contains
cubins for `sm_86`, `sm_89` and `sm_120a`, without PTX. CUDA user-space and
NVIDIA driver libraries are not bundled inside the pylocalvqe wheel;
user-space libraries come from the pip dependencies.

[GitHub Releases](https://github.com/sync-dev-org/pylocalvqe/releases) contain
the same wheel bytes as PyPI and `SHA256SUMS`. To verify a pinned download:

```bash
tag=v0.4.0
mkdir -p pylocalvqe-release
gh release download "$tag" --repo sync-dev-org/pylocalvqe \
  --pattern '*.whl' --pattern SHA256SUMS --dir pylocalvqe-release
(cd pylocalvqe-release && sha256sum --check SHA256SUMS)
```

Neither the shared core nor the extension loads external ggml backend files.

## Models

Weights are fetched from a pinned revision of Hugging Face
[`LocalAI-io/LocalVQE`](https://huggingface.co/LocalAI-io/LocalVQE), checked
against the model's SHA-256 digest, and stored outside the package:

```python
from pylocalvqe.assets import MODEL_REVISION, download_model, verify_model

print(MODEL_REVISION)
model = download_model("v1.4-aec-200k", "models")
verify_model(model, "v1.4-aec-200k")
```

Available variant names are `"v1.3"`, `"v1.4-aec-200k"` and
`"v1.4-aec-2.7k"`. `download_model` reuses a verified local copy; after
downloading, processing can run offline. The model repository's card states
Apache-2.0 terms for the weights.

## Python usage

```python
import numpy as np
from pylocalvqe import LocalVQE
from pylocalvqe.assets import download_model

model = download_model("v1.4-aec-200k", "models")
with LocalVQE(model, device="cpu", threads=1) as engine:
    mic = np.zeros(engine.hop_length, dtype=np.float32)
    ref = np.zeros(engine.hop_length, dtype=np.float32)
    enhanced = engine.process(mic, ref)
    delay_samples = engine.algorithmic_delay
    engine.reset()  # clear streaming state before another utterance
```

`mic` is one near-end microphone hop and `ref` is the corresponding far-end
reference hop. Each must be a one-dimensional floating-point array containing
exactly 256 samples. Output is a NumPy `float32` array. Shape and type errors
raise `ValueError` or `TypeError`; native processing failures raise
`LocalVQEError`. `sample_rate`, `hop_length`, `algorithmic_delay` and `variant`
are available on the engine. The algorithmic delay is 256 samples for v1.3
and v1.4-AEC 200K, and zero for v1.4-AEC 2.7K. Thread counts range from 0 to
32, with 0 selecting the native default. `close()` releases resources and is
safe to call repeatedly; the context manager closes automatically.

Use `device="cuda"` to select CUDA. Without a compatible NVIDIA driver or
GPU, this request raises `LocalVQEError`; the same package can still process
on CPU. The v1.4-AEC 2.7K model stores context
and weights on the selected CUDA device but computes frames on the CPU;
v1.3 and v1.4-AEC 200K run their frame graphs on CUDA when selected.

When migrating from v0.3, remove the `library_path` argument from `LocalVQE`:
the wheel supplies its own shared core. No separate native library download
or path argument is needed.

## C++ usage

The same wheel contains public C and C++ headers, the shared core and a
CMake package. Install it as above, then point CMake at the installed
`pylocalvqe` directory. Only your application is compiled; the supplied
core is already built. C++ consumers need a C++20 compiler and CMake 3.20
or later. Model download is provided by Python; C++ can verify and use an
existing GGUF file.

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(example LANGUAGES CXX)
find_package(pylocalvqe CONFIG REQUIRED)
add_executable(example main.cpp)
target_link_libraries(example PRIVATE pylocalvqe::core)
```

`main.cpp`:

```cpp
#include <pylocalvqe/pylocalvqe.hpp>
#include <array>
#include <cmath>

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    pylocalvqe::Engine engine(argv[1], {.threads = 1});
    std::array<float, 256> mic{}, ref{};
    for (std::size_t i = 0; i < mic.size(); ++i)
        mic[i] = 0.1f * std::sin(0.17f * i);
    for (int hop = 0; hop < 8; ++hop)
        for (float sample : engine.process(mic, ref))
            if (!std::isfinite(sample)) return 2;
    return 0;
}
```

```bash
prefix=$(.venv/bin/python -c \
  'from pathlib import Path; import pylocalvqe; print(Path(pylocalvqe.__file__).parent)')
cmake -S . -B build -DCMAKE_PREFIX_PATH="$prefix"
cmake --build build
./build/example models/localvqe-v1.4-aec-200K-f32.gguf
```

`pylocalvqe::core` supplies the C++20 requirement, include paths and the
shared library. The wrapper owns engine resources and reports failures as
`pylocalvqe::Error` (or a derived exception). C callers can include
`<pylocalvqe/pylocalvqe.h>` to use the flat C ABI directly.

## License

pylocalvqe's own code is licensed under MIT. Wheels include the full license
texts and copyright notices for bundled third-party components, along with
a notice identifying NVIDIA CUDA terms for the CUDA code. The wheel's
license metadata is `MIT AND Apache-2.0 AND BSD-3-Clause`. CUDA runtime
libraries and the NVIDIA driver have their own terms.
