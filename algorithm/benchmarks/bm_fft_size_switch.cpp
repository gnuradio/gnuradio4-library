#include <chrono>
#include <complex>
#include <cstddef>
#include <format>
#include <malloc.h>
#include <numbers>
#include <print>
#include <vector>

#include <benchmark.hpp>

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

// one FFT instance serving several transform lengths (display + audio + calibration off one thread)
// pays the per-length setup again on every change; a dedicated instance per length pays it once

template<typename T>
std::vector<T, gr::allocator::Aligned<T>> tone(std::size_t N) {
    std::vector<T, gr::allocator::Aligned<T>> signal(N);
    for (std::size_t i = 0; i < N; i++) {
        const double v = std::sin(2. * std::numbers::pi * 5. * static_cast<double>(i) / static_cast<double>(N));
        if constexpr (gr::meta::complex_like<T>) {
            signal[i] = {static_cast<typename T::value_type>(v), 0.};
        } else {
            signal[i] = static_cast<T>(v);
        }
    }
    return signal;
}

#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33))
#include <malloc.h>
std::size_t heapInUse() {
    const auto info = mallinfo2();
    return info.uordblks + info.hblkhd;
}
#else
// no allocator statistics on this C library; the heap columns read zero
std::size_t heapInUse() { return 0UZ; }
#endif

template<std::size_t nRepetitions, typename T>
void benchSizePair(std::size_t nA, std::size_t nB) {
    using namespace benchmark;
    using Out = std::complex<typename T::value_type>;
    using Fft = gr::algorithm::FFT<T, Out>;

    const auto signalA = tone<T>(nA);
    const auto signalB = tone<T>(nB);

    std::vector<Out, gr::allocator::Aligned<Out>> outA(nA);
    std::vector<Out, gr::allocator::Aligned<Out>> outB(nB);

    Fft dedicatedA;
    Fft dedicatedB;
    Fft shared;
    dedicatedA.compute(signalA, outA);
    dedicatedB.compute(signalB, outB);

    const std::size_t heapIdle = heapInUse();
    shared.compute(signalA, outA);
    const std::size_t heapOneSize = heapInUse();
    shared.compute(signalB, outB);
    shared.compute(signalA, outA);
    const std::size_t heapBothSizes = heapInUse();

    const std::string nameA      = std::format("dedicated   N = {:7}", nA);
    const std::string nameB      = std::format("dedicated   N = {:7}", nB);
    const std::string nameShared = std::format("alternating N = {:7} + {:7}", nA, nB);

    ::benchmark::benchmark<nRepetitions>(std::string_view(nameA))      = [&dedicatedA, &signalA, &outA] { dedicatedA.compute(signalA, outA); };
    ::benchmark::benchmark<nRepetitions>(std::string_view(nameB))      = [&dedicatedB, &signalB, &outB] { dedicatedB.compute(signalB, outB); };
    ::benchmark::benchmark<nRepetitions>(std::string_view(nameShared)) = [&shared, &signalA, &signalB, &outA, &outB] {
        shared.compute(signalA, outA);
        shared.compute(signalB, outB);
    };

    std::chrono::nanoseconds spentA{0};
    std::chrono::nanoseconds spentB{0};
    for (std::size_t i = 0; i < nRepetitions; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        shared.compute(signalA, outA);
        const auto t1 = std::chrono::steady_clock::now();
        shared.compute(signalB, outB);
        const auto t2 = std::chrono::steady_clock::now();
        spentA += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
        spentB += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);
    }

    const auto perCompute = [](std::chrono::nanoseconds total) { return 1e-3 * static_cast<double>(total.count()) / static_cast<double>(nRepetitions); };
    std::println("  alternating per compute(): N = {:7} {:10.1f} us, N = {:7} {:10.1f} us", nA, perCompute(spentA), nB, perCompute(spentB));
    std::println("  heap in use: idle {} kB, after N = {} {} kB, after both {} kB (delta {} kB)", //
        heapIdle / 1024, nA, heapOneSize / 1024, heapBothSizes / 1024, (heapBothSizes - heapOneSize) / 1024);

    ::benchmark::results::add_separator();
}

inline const boost::ut::suite<"FFT transform-length switching"> _fft_size_switch_bm = [] {
    benchSizePair<500UZ, std::complex<float>>(1024UZ, 4096UZ);
    benchSizePair<20UZ, std::complex<float>>(8192UZ, 1048576UZ);
    benchSizePair<200UZ, std::complex<float>>(1009UZ, 4001UZ); // Bluestein: prime lengths
};

int main() { /* not needed by the UT framework */ }
