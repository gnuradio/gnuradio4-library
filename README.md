# GNU Radio 4 Library

This repository contains the reusable non-block DSP and algorithm libraries and tools that can be used across the ecosystem.

It builds against an installed `gnuradio4-core` SDK and installs the CMake package `gnuradio4Library`.

## Build model

- configure against the installed core prefix
- use `GR_USE_FETCHCONTENT_DEPS=ON` when test-only dependencies should be fetched automatically
- install into the shared local prefix used by the GNU Radio 4 umbrella workspace

## Package naming

- repository name: `gnuradio4-library`
- CMake package name: `gnuradio4Library`
- exported target namespace: `gnuradio4::`
- compatibility include directory, target, and namespace: `algorithm`, `gnuradio-algorithm`, and `gr::algorithm`
