[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Main CI](https://github.com/gnuradio/gnuradio4-library/actions/workflows/ci-main.yml/badge.svg?branch=main)](https://github.com/gnuradio/gnuradio4-library/actions/workflows/ci-main.yml)
[![SDK Image](https://github.com/gnuradio/gnuradio4-library/actions/workflows/sdk-image.yml/badge.svg?branch=main)](https://github.com/gnuradio/gnuradio4-library/actions/workflows/sdk-image.yml)

# GNU Radio 4.0 Library

`gnuradio4-library` provides reusable, non-block signal-processing algorithms
for GNU Radio 4 applications and block libraries. It builds on the installed
[`gnuradio4-core`](https://github.com/gnuradio/gnuradio4-core) SDK and supplies
the common DSP building blocks used by downstream repositories without adding
them to the core runtime.

This repository is one part of the split GNU Radio 4 source tree:

- [`gnuradio4-core`](https://github.com/gnuradio/gnuradio4-core): runtime,
  scheduler, graph and block model, plugin support, and the block-development SDK
- `gnuradio4-library`: reusable, non-block DSP algorithms built on core
- [`gnuradio4-blocks`](https://github.com/gnuradio/gnuradio4-blocks): standard
  GNU Radio 4 blocks built on core and this library

## What's Included?

The public C++23 library is currently header-oriented and includes:

- Fourier transforms and window functions, including SIMD FFT support
- random-number, Gaussian-noise, tone, noise, and signal generators
- filtering utilities, including Savitzky-Golay and SVD filters
- data-set estimation, mathematics, and transformation helpers
- file I/O helpers for native and Emscripten builds
- burst tapering, sample-rate estimation, and Schmitt-trigger utilities
- plotting data models for canvases, charts, and graphs

Public headers are installed under `include/gnuradio-4.0/algorithm`. The project
also builds benchmarks and tests for the supplied algorithms.

## Building

The library requires CMake 3.27 or newer, a C++23 compiler, and an installed
GNU Radio 4 core SDK. Configure it by adding the core installation prefix to
`CMAKE_PREFIX_PATH`:

```bash
git clone https://github.com/gnuradio/gnuradio4-library.git
cd gnuradio4-library

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH=/path/to/gnuradio4-core-prefix
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Install the library into the same prefix used by the rest of a GNU Radio 4
development environment:

```bash
cmake --install build --prefix /path/to/gnuradio4-prefix
```

Useful configuration options are:

- `ENABLE_EXAMPLES` (default: `ON`): build example and benchmark programs
- `ENABLE_TESTING` (default: `ON`): build and register tests
- `ENABLE_COVERAGE` (default: `OFF`): add coverage-report targets
- `GR_USE_FETCHCONTENT_DEPS` (default: `OFF`): allow CMake to fetch selected
  development dependencies
- `USE_CCACHE` (default: `ON`): use `ccache` when it is available

## Using the Library

An installation exports the `gnuradio4Library` CMake package and targets in the
`gnuradio4::` namespace. A downstream CMake project can consume the algorithm
library with:

```cmake
find_package(gnuradio4Library CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE gnuradio4::gnuradio-algorithm)
```

The source tree uses the compatibility target name `gnuradio-algorithm`; the
public C++ API is in the `gr::algorithm` namespace.

## SDK Images

Pushes to `main` publish profile-specific SDK images to the GitHub Container
Registry. Each image contains compatible core and library installations for use
by `gnuradio4-blocks` and other downstream projects:

```text
ghcr.io/gnuradio/gnuradio4-library-sdk:main-<platform>-<compiler>-<profile>
```

Use a profile-specific tag such as `main-ubuntu-24.04-gcc14-release`; a generic
`main` tag is not published.

## License

Unless otherwise noted, this project is licensed under the [MIT License](LICENSE).

Copyright (C) The GNU Radio Authors<br>
Copyright (C) Contributors to the GNU Radio Project<br>
Copyright (C) FAIR - Facility for Antiproton & Ion Research, Darmstadt, Germany

## Helpful Links

- [GNU Radio website](https://gnuradio.org/)
- [GNU Radio wiki](https://wiki.gnuradio.org/)
- [Issue tracker](https://github.com/gnuradio/gnuradio4-library/issues)
- [GNU Radio Matrix chat](https://chat.gnuradio.org/)
- [GNU Radio mailing list](https://lists.gnu.org/mailman/listinfo/discuss-gnuradio)
