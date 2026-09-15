#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <format>
#include <limits>
#include <numbers>
#include <print>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/algorithm/signal/Phasor.hpp>

namespace {

constexpr double kTwoPi = 2. * std::numbers::pi_v<double>;

struct WindowError {
    double first = 0.; // worst phase error over the first window generated
    double last  = 0.; // ... and over the last one, however long the stream ran
};

/// Worst deviation of fill()'s output from exp(j*k*increment), measured on a stride within the first and the
/// last window of an nSamples-long stream. The two figures together are the no-drift property: the error is
/// bounded by the re-seed rounding and does not grow with stream length.
template<std::floating_point F>
[[nodiscard]] WindowError fillError(double increment, std::size_t nSamples, std::size_t window, std::size_t stride) {
    gr::signal::Phasor<F> phasor;
    phasor.configure(increment, 0.);

    std::vector<std::complex<F>> buffer(window);
    WindowError                  error;
    std::size_t                  produced = 0UZ;
    while (produced < nSamples) {
        const std::size_t n = std::min(window, nSamples - produced);
        phasor.fill(std::span<std::complex<F>>(buffer.data(), n));

        double worst = 0.;
        for (std::size_t k = 0UZ; k < n; k += stride) {
            const std::complex<double> want = std::polar(1., std::remainder(static_cast<double>(produced + k) * increment, kTwoPi));
            const std::complex<double> have = std::complex<double>(buffer[k]);
            worst                           = std::max(worst, std::abs(std::arg(have / want)));
        }
        if (produced == 0UZ) {
            error.first = worst;
        }
        error.last = worst;
        produced += n;
    }
    return error;
}

/// The modulated path's contract restated independently: exp(j*phase_k), where phase_k is the prefix sum of
/// the increments reduced to (-pi, pi] after each step.
template<std::floating_point F, std::floating_point FIncrement>
[[nodiscard]] std::vector<std::complex<F>> modulatedOracle(std::span<const FIncrement> increments, double initialPhase) {
    std::vector<std::complex<F>> out(increments.size());
    double                       phase = std::remainder(initialPhase, kTwoPi);
    for (std::size_t k = 0UZ; k < increments.size(); ++k) {
        out[k] = std::complex<F>(static_cast<F>(std::cos(phase)), static_cast<F>(std::sin(phase)));
        phase += static_cast<double>(increments[k]);
        if (phase > std::numbers::pi_v<double> || phase < -std::numbers::pi_v<double>) {
            phase = std::remainder(phase, kTwoPi);
        }
    }
    return out;
}

/// Deterministic non-trivial increment sequence; no RNG dependency, and the values span both signs.
[[nodiscard]] std::vector<float> rampIncrements(std::size_t n) {
    std::vector<float> increments(n);
    for (std::size_t k = 0UZ; k < n; ++k) {
        increments[k] = static_cast<float>(0.31 * std::sin(0.017 * static_cast<double>(k)) + 0.05);
    }
    return increments;
}

/// Worst spur outside the main lobe, in dB relative to the tone.
///
/// The window and the transform are BOTH double: a 4-term Blackman-Harris floors at its own -92 dB sidelobe
/// and a float transform at about -132 dBc, so either would measure the instrument rather than the phasor.
/// The 7-term minimum-sidelobe window below is at -180 dB, past where the float tone's own quantization sits.
[[nodiscard]] double worstSpurDbc(std::span<const std::complex<float>> tone) {
    const std::size_t n = tone.size();
    // 7-term minimum-sidelobe Blackman-Harris
    constexpr std::array<double, 7UZ> a{0.27105140069342, 0.43329793923448, 0.21812299954311, 0.06592544638803, 0.01081174209837, 0.00077658482522, 0.00001388721735};
    std::vector<std::complex<double>> windowed(n);
    for (std::size_t i = 0UZ; i < n; ++i) {
        const double x = kTwoPi * static_cast<double>(i) / static_cast<double>(n);
        double       w = a[0];
        for (std::size_t term = 1UZ; term < a.size(); ++term) {
            w += (term % 2UZ == 1UZ ? -1. : 1.) * a[term] * std::cos(static_cast<double>(term) * x);
        }
        windowed[i] = std::complex<double>(tone[i]) * w;
    }

    gr::algorithm::FFT<std::complex<double>> fft;
    std::vector<std::complex<double>>        spectrum(n);
    fft.compute(windowed, spectrum);

    std::size_t peak = 0UZ;
    for (std::size_t k = 1UZ; k < n; ++k) {
        if (std::abs(spectrum[k]) > std::abs(spectrum[peak])) {
            peak = k;
        }
    }

    // the 7-term window's main lobe is +-7 bins wide; everything beyond it is spur
    constexpr std::size_t kMainLobe = 8UZ;
    double                worst     = 0.;
    for (std::size_t k = 0UZ; k < n; ++k) {
        const std::size_t distance = std::min((k + n - peak) % n, (peak + n - k) % n);
        if (distance <= kMainLobe) {
            continue;
        }
        worst = std::max(worst, std::abs(spectrum[k]));
    }
    return 20. * std::log10(worst / std::abs(spectrum[peak]));
}

} // namespace

const boost::ut::suite<"Phasor"> phasorTests = [] {
    using namespace boost::ut;
    using gr::signal::Phasor;

    constexpr auto kFloatingTypes = std::tuple<float, double>{};

    // 10^8 samples is 24414 re-seed intervals; the increment is irrational so no re-seed ever
    // lands on a repeated phase
    "fill stays on the exact rotation and does not drift"_test = []<std::floating_point F> {
        constexpr std::size_t nSamples  = 100'000'000UZ;
        constexpr double      increment = 0.2718281828459045;
        // the worst over every increment swept below is 9.5e-7 rad in float and 4.4e-9 in double. The double
        // figure is the oracle's floor rather than the kernel's, since remainder(k*inc, 2pi) at k = 1e8 carries
        // about 3e-9 rad of its own rounding, so the float column is the one that shows the no-drift property.
        constexpr double bound = std::is_same_v<F, float> ? 3e-6 : 1e-7;

        const auto error = fillError<F>(increment, nSamples, 65536UZ, 997UZ);
        expect(lt(error.first, bound)) << std::format("first-window phase error {:g}", error.first);
        expect(lt(error.last, bound)) << std::format("last-window phase error {:g} after {} samples", error.last, nSamples);
        std::println("Phasor<{}> fill phase error over {} samples: first window {:g} rad, last window {:g} rad", //
            std::is_same_v<F, float> ? "float" : "double", nSamples, error.first, error.last);
    } | kFloatingTypes;

    "fill matches the definition across increment magnitudes"_test = []<std::floating_point F> {
        constexpr double bound = std::is_same_v<F, float> ? 3e-6 : 1e-8;
        for (const double increment : {1e-6, 1e-3, 0.05, 1.0, 3.0, -2.7, 6.0}) {
            const auto error = fillError<F>(increment, 1'000'000UZ, 65536UZ, 313UZ);
            expect(lt(error.last, bound)) << std::format("increment {}: phase error {:g}", increment, error.last);
        }
    } | kFloatingTypes;

    // the modulated path against an independent statement of the same accumulation rule
    "fillModulated is the reduced prefix sum, bit for bit"_test = []<std::floating_point F> {
        const auto increments = rampIncrements(10'000UZ);
        const auto want       = modulatedOracle<F>(std::span<const float>(increments), 0.4);

        Phasor<F> phasor;
        phasor.configure(0., 0.4);
        std::vector<std::complex<F>> have(increments.size());
        phasor.fillModulated(std::span<const float>(increments), std::span<std::complex<F>>(have));

        expect(eq(have.size(), want.size()));
        bool identical = true;
        for (std::size_t k = 0UZ; k < want.size(); ++k) {
            identical = identical && have[k].real() == want[k].real() && have[k].imag() == want[k].imag();
        }
        expect(identical) << "fillModulated must reproduce the prefix-sum oracle exactly";
    } | kFloatingTypes;

    "mixModulated multiplies the same phases into the stream"_test = []<std::floating_point F> {
        const auto increments = rampIncrements(4096UZ);
        const auto phases     = modulatedOracle<F>(std::span<const float>(increments), -1.1);

        std::vector<std::complex<F>> in(increments.size());
        for (std::size_t k = 0UZ; k < in.size(); ++k) {
            in[k] = std::complex<F>(static_cast<F>(0.5 + 0.25 * std::cos(0.003 * static_cast<double>(k))), static_cast<F>(-0.75));
        }

        Phasor<F> phasor;
        phasor.configure(0., -1.1);
        std::vector<std::complex<F>> have(in.size());
        phasor.mixModulated(std::span<const float>(increments), std::span<const std::complex<F>>(in), std::span<std::complex<F>>(have));

        for (std::size_t k = 0UZ; k < have.size(); ++k) {
            const std::complex<double> want = std::complex<double>(in[k]) * std::complex<double>(phases[k]);
            expect(lt(std::abs(std::complex<double>(have[k]) - want), 1e-6)) << std::format("mixModulated mismatch at {}", k);
        }
    } | kFloatingTypes;

    // the re-seed grid is stream-absolute, so a call boundary anywhere must change nothing
    "fill and mix are chunk independent, bit for bit"_test = []<std::floating_point F> {
        constexpr std::size_t nSamples  = 20'000UZ;
        constexpr double      increment = 0.2718281828459045;

        std::vector<std::complex<F>> in(nSamples);
        for (std::size_t k = 0UZ; k < nSamples; ++k) {
            in[k] = std::complex<F>(static_cast<F>(std::cos(0.00037 * static_cast<double>(k))), static_cast<F>(std::sin(0.00037 * static_cast<double>(k))));
        }

        const auto generateInChunks = [&](std::size_t chunk, bool doMix) {
            Phasor<F> phasor;
            phasor.configure(increment, 0.25);
            std::vector<std::complex<F>> out(nSamples);
            for (std::size_t i = 0UZ; i < nSamples; i += chunk) {
                const std::size_t n = std::min(chunk, nSamples - i);
                if (doMix) {
                    phasor.mix(std::span<const std::complex<F>>(in.data() + i, n), std::span<std::complex<F>>(out.data() + i, n));
                } else {
                    phasor.fill(std::span<std::complex<F>>(out.data() + i, n));
                }
            }
            return out;
        };

        for (const bool doMix : {false, true}) {
            const auto reference = generateInChunks(nSamples, doMix);
            for (const std::size_t chunk : {1UZ, 3UZ, 64UZ, 1000UZ, 4095UZ, 4096UZ, 4097UZ}) {
                const auto split     = generateInChunks(chunk, doMix);
                bool       identical = true;
                for (std::size_t k = 0UZ; k < nSamples; ++k) {
                    identical = identical && split[k].real() == reference[k].real() && split[k].imag() == reference[k].imag();
                }
                expect(identical) << std::format("{} in {}-sample calls must be bit-identical to one call", doMix ? "mix" : "fill", chunk);
            }
        }
    } | kFloatingTypes;

    "fillModulated is chunk independent, bit for bit"_test = []<std::floating_point F> {
        const auto increments = rampIncrements(9'999UZ);

        const auto generateInChunks = [&](std::size_t chunk) {
            Phasor<F> phasor;
            phasor.configure(0., 0.4);
            std::vector<std::complex<F>> out(increments.size());
            for (std::size_t i = 0UZ; i < increments.size(); i += chunk) {
                const std::size_t n = std::min(chunk, increments.size() - i);
                phasor.fillModulated(std::span<const float>(increments.data() + i, n), std::span<std::complex<F>>(out.data() + i, n));
            }
            return out;
        };

        const auto reference = generateInChunks(increments.size());
        for (const std::size_t chunk : {1UZ, 7UZ, 512UZ, 4096UZ}) {
            const auto split     = generateInChunks(chunk);
            bool       identical = true;
            for (std::size_t k = 0UZ; k < reference.size(); ++k) {
                identical = identical && split[k].real() == reference[k].real() && split[k].imag() == reference[k].imag();
            }
            expect(identical) << std::format("fillModulated in {}-sample calls must be bit-identical to one call", chunk);
        }
    } | kFloatingTypes;

    "a chirp agrees with the modulated path it replaces, and is chunk independent"_test = []<std::floating_point F> {
        // A linear frequency sweep produces an arithmetic increment schedule, which is exactly what
        // fillModulated would be handed sample by sample. The chirp entry has to reproduce it without
        // evaluating a transcendental per sample, so the two are compared over the same schedule.
        constexpr std::size_t kSamples = 4096UZ;
        constexpr double      kStart   = 0.31;
        constexpr double      kStep    = 1.7e-6;

        std::vector<float> increments(kSamples);
        for (std::size_t k = 0UZ; k < kSamples; ++k) {
            increments[k] = static_cast<float>(kStart + static_cast<double>(k) * kStep);
        }

        Phasor<F> modulated;
        modulated.configure(0., 0.);
        std::vector<std::complex<F>> viaModulate(kSamples);
        modulated.fillModulated(std::span<const float>(increments), std::span<std::complex<F>>(viaModulate));

        const auto chirpInChunks = [&](std::size_t chunk) {
            Phasor<F> phasor;
            phasor.configure(kStart, 0.);
            std::vector<std::complex<F>> out(kSamples);
            for (std::size_t i = 0UZ; i < kSamples; i += chunk) {
                const std::size_t n = std::min(chunk, kSamples - i);
                phasor.fillChirp(kStep, std::span<std::complex<F>>(out.data() + i, n));
            }
            return out;
        };

        const auto reference = chirpInChunks(kSamples);

        // the two paths round differently - one accumulates a phase, the other a phasor - so they agree to a
        // tolerance rather than bit for bit, and the tolerance is what a float phasor can hold over the run
        double worst = 0.;
        for (std::size_t k = 0UZ; k < kSamples; ++k) {
            worst = std::max(worst, std::abs(std::complex<double>(reference[k]) - std::complex<double>(viaModulate[k])));
        }
        expect(lt(worst, 1e-4)) << std::format("the chirp and the modulated schedule differ by at most {:.3e}", worst);

        // chunk independence is the stronger property and does hold bit for bit
        for (const std::size_t chunk : {1UZ, 7UZ, 256UZ, 511UZ, 4096UZ}) {
            const auto split     = chirpInChunks(chunk);
            bool       identical = true;
            for (std::size_t k = 0UZ; k < kSamples; ++k) {
                identical = identical && split[k].real() == reference[k].real() && split[k].imag() == reference[k].imag();
            }
            expect(identical) << std::format("fillChirp in {}-sample calls must be bit-identical to one call", chunk);
        }
    } | kFloatingTypes;

    "mix equals fill multiplied into the stream"_test = []<std::floating_point F> {
        constexpr std::size_t nSamples  = 8192UZ;
        constexpr double      increment = 0.13;

        std::vector<std::complex<F>> in(nSamples, std::complex<F>(F(0.75), F(-0.25)));

        Phasor<F> generator;
        generator.configure(increment, 1.0);
        std::vector<std::complex<F>> phases(nSamples);
        generator.fill(std::span<std::complex<F>>(phases));

        Phasor<F> mixer;
        mixer.configure(increment, 1.0);
        std::vector<std::complex<F>> mixed(nSamples);
        mixer.mix(std::span<const std::complex<F>>(in), std::span<std::complex<F>>(mixed));

        expect(approx(mixer.phase(), generator.phase(), 1e-12)) << "both paths must leave the same accumulated phase";
        for (std::size_t k = 0UZ; k < nSamples; ++k) {
            const std::complex<double> want = std::complex<double>(in[k]) * std::complex<double>(phases[k]);
            expect(lt(std::abs(std::complex<double>(mixed[k]) - want), 1e-6)) << std::format("mix mismatch at {}", k);
        }
    } | kFloatingTypes;

    "advance moves the phase by whole increments and is reversible"_test = [] {
        Phasor<double> phasor;
        phasor.configure(0.37, 0.1);

        phasor.advance(1);
        expect(approx(phasor.phase(), 0.47, 1e-12));
        phasor.advance(-1);
        expect(approx(phasor.phase(), 0.1, 1e-12)) << "stepping one increment forward and back must return the phase";

        phasor.advance(1000);
        expect(approx(phasor.phase(), std::remainder(0.1 + 1000. * 0.37, kTwoPi), 1e-12));
    };

    "a generated span leaves the phase where the next sample continues"_test = [] {
        constexpr double increment = 0.37;
        Phasor<double>   phasor;
        phasor.configure(increment, 0.);

        std::vector<std::complex<double>> first(100UZ);
        phasor.fill(std::span<std::complex<double>>(first));
        expect(approx(phasor.phase(), std::remainder(100. * increment, kTwoPi), 1e-12));

        std::vector<std::complex<double>> second(1UZ);
        phasor.fill(std::span<std::complex<double>>(second));
        const std::complex<double> want = std::polar(1., std::remainder(100. * increment, kTwoPi));
        expect(lt(std::abs(second[0] - want), 1e-12)) << "the sample after a span continues the same rotation";
    };

    "configure refuses non-finite values"_test = [] {
        Phasor<float> phasor;
        expect(throws([&] { phasor.configure(std::numeric_limits<double>::quiet_NaN(), 0.); }));
        expect(throws([&] { phasor.configure(0.1, std::numeric_limits<double>::infinity()); }));
        expect(nothrow([&] { phasor.configure(0.1, 0.2); }));
    };

    "spectral honesty: the worst spur stays below the stated floor"_test = [] {
        constexpr std::size_t nSamples = 65536UZ;
        // deliberately off-bin: 1000.37 cycles over the transform, so no bin holds the tone exactly
        const double increment = kTwoPi * 1000.37 / static_cast<double>(nSamples);

        gr::signal::Phasor<float> phasor;
        phasor.configure(increment, 0.);
        std::vector<std::complex<float>> tone(nSamples);
        phasor.fill(std::span<std::complex<float>>(tone));

        const double spurDbc = worstSpurDbc(tone);
        std::println("Phasor<float> worst spur at an off-bin tone: {:.1f} dBc", spurDbc);
        // the recurrence holds the worst spur near -147 dBc; the bound leaves margin for libm and transform
        // differences across platforms
        expect(lt(spurDbc, -130.)) << std::format("worst spur {:.1f} dBc", spurDbc);
    };
};

int main() { /* not needed for UT */ }
