# pylocalvqe

C++ and Python interfaces for [LocalVQE](https://github.com/localai-org/LocalVQE),
a ggml-based voice quality enhancement engine providing acoustic echo
cancellation, noise suppression and dereverberation. Processing is **CPU Only**.

- Stream 16 kHz mono audio in 256-sample (16 ms) hops.
- Use v1.3, v1.4-AEC 200K or v1.4-AEC 2.7K models.
- Read each model's algorithmic delay for downstream compensation.
- Use the same shared core from Python or C++20 through a flat C ABI.
- Download model weights separately and verify their SHA-256 digests.

## Install

Install the wheel from [PyPI](https://pypi.org/project/pylocalvqe/) into a
CPython 3.12 or later environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: pylocalvqe
```

Pin a release when reproducibility matters, for example `pylocalvqe==0.5.0`.
Linux x86_64 and Windows x86_64 wheels are available on PyPI. Both contain
only the CPU backend and use CPython's stable ABI, with the tags
`cp312-abi3-manylinux_2_28_x86_64` and `cp312-abi3-win_amd64`, respectively.
On Windows, install from PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --only-binary=:all: pylocalvqe
```

Linux needs glibc 2.28 or later. Windows needs the UCRT supplied by Windows;
a separate Visual C++ Redistributable package is unnecessary. Both platforms
need a CPU with SSE4.2, AVX, AVX2, FMA,
F16C and BMI2. AVX-512 is unnecessary. NumPy 1.26 or later is installed
automatically as a Python dependency. No compiler, CMake or OpenMP runtime
is required to install or use the package.

[GitHub Releases](https://github.com/sync-dev-org/pylocalvqe/releases) contain
the same two wheels as PyPI and `SHA256SUMS`. To verify a pinned download:

```bash
tag=v0.5.0
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
with LocalVQE(model, threads=1) as engine:
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

When migrating from v0.3, remove the `library_path` argument from `LocalVQE`:
the wheel supplies its own shared core. No separate native library download
or path argument is needed.

When migrating from v0.4, remove the `device` argument from `LocalVQE`.
Processing is CPU Only; CUDA can no longer be selected. The C ABI's
`pylocalvqe_engine_config` also removes its `device` field, and the core's
SONAME changes to `libpylocalvqe.so.2`. Rebuild C++ applications against the
updated headers and core.

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

pylocalvqe's own code is licensed under MIT
([`LICENSE`](https://github.com/sync-dev-org/pylocalvqe/blob/main/LICENSE)).
The wheel also bundles the third-party components below, so the wheel's
license metadata is `MIT AND Apache-2.0 AND BSD-3-Clause`. The wheel carries
the full license texts and copyright notices, and the same files are in
[`licenses/`](https://github.com/sync-dev-org/pylocalvqe/tree/main/licenses).

| Component | License | Files |
|---|---|---|
| LocalVQE | Apache-2.0 | `licenses/localvqe.txt`, `licenses/localvqe-notice.txt` |
| ggml | MIT | `licenses/ggml.txt` |
| llamafile code in ggml | MIT | `licenses/llamafile.txt` |
| YaRN code in ggml | MIT | `licenses/yarn.txt` |
| nanobind | BSD-3-Clause | `licenses/nanobind.txt` |
| robin-map (used by nanobind) | MIT | `licenses/robin-map.txt` |
| DLPack (used by nanobind) | Apache-2.0 | `licenses/dlpack.txt` |
