#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <numbers>
#include <print>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/algorithm/digital/AdaptiveFir.hpp>
#include <gnuradio-4.0/algorithm/digital/Constellation.hpp>
#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>

namespace {

using gr::digital::AdaptiveAlgorithm;
using gr::digital::AdaptiveFir;
using gr::digital::modulusReference;

using Complex = std::complex<double>;

/// A three-tap channel with a real echo and a quadrature one. Both of its zeros lie inside the unit circle, so the
/// inverse is causal and decays; the slower zero, at radius 0.62, sets how much of it an eleven-tap filter holds.
constexpr std::array<Complex, 3UZ> kChannel{Complex(1., 0.), Complex(0.4, 0.), Complex(0., 0.2)};

constexpr std::size_t   kEqualizerTaps   = 11UZ;
constexpr std::size_t   kTrainSymbols    = 20000UZ;
constexpr std::size_t   kMeasureTail     = 5000UZ;
constexpr std::size_t   kMaxTaps         = AdaptiveFir<double>::kMaxTaps;
constexpr double        kDivergenceBound = 100.;
constexpr std::uint64_t kSymbolSeed      = 0x5EEDU;

[[nodiscard]] constexpr std::string_view algorithmName(AdaptiveAlgorithm algorithm) noexcept {
    switch (algorithm) {
    case AdaptiveAlgorithm::Lms: return "lms";
    case AdaptiveAlgorithm::Nlms: return "nlms";
    case AdaptiveAlgorithm::Cma: return "cma";
    }
    return "unknown";
}

/// Unit-power QPSK from a deterministic engine, so every run of the suite sees the same sequence.
template<std::floating_point F = double>
[[nodiscard]] std::vector<std::complex<F>> qpskSymbols(std::size_t n, std::uint64_t seed) {
    gr::rng::Xoshiro256pp        rng(seed);
    const F                      level = std::numbers::sqrt2_v<F> / F{2};
    std::vector<std::complex<F>> symbols(n);
    for (std::size_t i = 0UZ; i < n; ++i) {
        const std::uint64_t bits = rng() & 3ULL;
        symbols[i]               = std::complex<F>((bits & 1ULL) != 0ULL ? level : -level, (bits & 2ULL) != 0ULL ? level : -level);
    }
    return symbols;
}

/// The symbol stream convolved with the channel, one received sample per symbol.
[[nodiscard]] std::vector<Complex> throughChannel(std::span<const Complex> symbols, std::span<const Complex> channel) {
    std::vector<Complex> received(symbols.size(), Complex{});
    for (std::size_t k = 0UZ; k < symbols.size(); ++k) {
        Complex sum{};
        for (std::size_t j = 0UZ; j < channel.size() && j <= k; ++j) {
            sum += channel[j] * symbols[k - j];
        }
        received[k] = sum;
    }
    return received;
}

/// The channel and the equalizer taken end to end, which is the response training drives toward a single spike.
[[nodiscard]] std::vector<Complex> combinedResponse(std::span<const Complex> taps, std::span<const Complex> channel) {
    std::vector<Complex> response(taps.size() + channel.size() - 1UZ, Complex{});
    for (std::size_t i = 0UZ; i < taps.size(); ++i) {
        for (std::size_t j = 0UZ; j < channel.size(); ++j) {
            response[i + j] += taps[i] * channel[j];
        }
    }
    return response;
}

[[nodiscard]] std::size_t peakIndex(std::span<const Complex> response) noexcept {
    std::size_t peak = 0UZ;
    for (std::size_t i = 1UZ; i < response.size(); ++i) {
        if (std::norm(response[i]) > std::norm(response[peak])) {
            peak = i;
        }
    }
    return peak;
}

/// Residual intersymbol interference: the power of the combined response away from its peak, against the peak.
[[nodiscard]] double residualIsiDb(std::span<const Complex> response) noexcept {
    double total = 0.;
    double peak  = 0.;
    for (const Complex& tap : response) {
        const double power = std::norm(tap);
        total += power;
        peak = std::max(peak, power);
    }
    return peak > 0. ? 10. * std::log10(std::max(total - peak, 1e-300) / peak) : 0.;
}

[[nodiscard]] bool isFinite(Complex value) noexcept { return std::isfinite(value.real()) && std::isfinite(value.imag()); }

/// The tap the center spike occupies, which is the whole of a filter that has not adapted.
template<std::floating_point F>
[[nodiscard]] std::complex<F> spikeTap(std::size_t index, std::size_t groupDelay) noexcept {
    return index == groupDelay ? std::complex<F>(F{1}, F{0}) : std::complex<F>{};
}

struct TrainedRun {
    double      meanSquaredError = 0.;
    double      isiDb            = 0.;
    std::size_t peak             = 0UZ;
    double      tapEnergy        = 0.;
    std::size_t resets           = 0UZ;
};

/// One trained pass over the channel, with the reference taken `referenceDelay` symbols behind the current input.
[[nodiscard]] TrainedRun runTrained(AdaptiveAlgorithm algorithm, double stepSize, std::size_t referenceDelay) {
    AdaptiveFir<double> filter;
    filter.configure(kEqualizerTaps, algorithm, stepSize, 1., kDivergenceBound);

    const auto symbols  = qpskSymbols(kTrainSymbols, kSymbolSeed);
    const auto received = throughChannel(symbols, kChannel);

    TrainedRun run;
    for (std::size_t k = 0UZ; k < kTrainSymbols; ++k) {
        filter.push(received[k]);
        const Complex y         = filter.output();
        const Complex reference = k >= referenceDelay ? symbols[k - referenceDelay] : Complex{};
        if (k >= kTrainSymbols - kMeasureTail) {
            run.meanSquaredError += std::norm(y - reference);
        }
        if (filter.adapt(y, reference)) {
            ++run.resets;
        }
    }

    const auto response = combinedResponse(filter.taps(), kChannel);
    run.meanSquaredError /= static_cast<double>(kMeasureTail);
    run.isiDb     = residualIsiDb(response);
    run.peak      = peakIndex(response);
    run.tapEnergy = filter.tapEnergy();
    return run;
}

} // namespace

const boost::ut::suite<"AdaptiveFir response"> adaptiveFirResponseTests = [] {
    using namespace boost::ut;

    "the unadapted filter is a delay of groupDelay samples"_test = []<std::floating_point F> {
        for (const std::size_t nTaps : {1UZ, 3UZ, 11UZ, 31UZ, 127UZ}) {
            AdaptiveFir<F> filter;
            filter.configure(nTaps, AdaptiveAlgorithm::Lms, 0.01, 1.);

            const std::size_t delay = filter.groupDelay();
            expect(eq(filter.size(), nTaps)) << std::format("{} taps", nTaps);
            expect(eq(filter.taps().size(), nTaps)) << std::format("{} taps", nTaps);
            expect(eq(delay, (nTaps - 1UZ) / 2UZ)) << std::format("{} taps", nTaps);

            for (std::size_t k = 0UZ; k < 2UZ * nTaps + 4UZ; ++k) {
                filter.push(k == 0UZ ? std::complex<F>(F{1}, F{0}) : std::complex<F>{});

                const std::complex<F> want = spikeTap<F>(k, delay);
                const std::complex<F> have = filter.output();
                expect(have == want) << std::format("{} taps: the impulse response at {} is ({}, {}) against ({}, {})", nTaps, k, have.real(), have.imag(), want.real(), want.imag());
            }
        }
    } | std::tuple<float, double>{};

    "output is the convolution of the taps with the history"_test = [] {
        constexpr std::size_t nTaps    = 17UZ;
        constexpr std::size_t nSamples = 400UZ;

        gr::rng::Xoshiro256pp rng(0xC0FFEEU);
        AdaptiveFir<double>   filter;
        filter.configure(nTaps, AdaptiveAlgorithm::Lms, 0.01, 1.);
        for (Complex& tap : filter._taps) {
            tap = Complex(rng.uniformM11<double>(), rng.uniformM11<double>());
        }

        std::vector<Complex> input(nSamples);
        for (Complex& sample : input) {
            sample = Complex(rng.uniformM11<double>(), rng.uniformM11<double>());
        }

        double worst = 0.;
        for (std::size_t k = 0UZ; k < nSamples; ++k) {
            filter.push(input[k]);

            // the direct sum runs from the oldest tap forward, so its rounding is arrived at independently
            Complex want{};
            for (std::size_t i = std::min(nTaps, k + 1UZ); i-- > 0UZ;) {
                want += filter.taps()[i] * input[k - i];
            }

            const Complex have      = filter.output();
            const double  deviation = std::abs(have - want) / std::max(std::abs(want), 1e-12);
            expect(lt(deviation, 1e-6)) << std::format("sample {}: output ({:g}, {:g}) against the direct sum ({:g}, {:g})", k, have.real(), have.imag(), want.real(), want.imag());
            worst = std::max(worst, deviation);
        }
        std::println("AdaptiveFir convolution: worst relative deviation from a direct sum over {} taps {:g}", nTaps, worst);
    };
};

const boost::ut::suite<"AdaptiveFir adaptation"> adaptiveFirAdaptationTests = [] {
    using namespace boost::ut;

    // The reference alignment picks which inverse the filter is asked for. At the group delay it keeps the center
    // spike's own lag and holds the six inverse terms that fit past it; at lag zero it is the causal inverse, which
    // this minimum-phase channel supplies across all eleven taps and which therefore reaches a far lower floor.
    "lms and nlms invert a fixed channel against a reference"_test = [] {
        struct Case {
            AdaptiveAlgorithm algorithm;
            double            stepSize;
            std::size_t       referenceDelay;
            double            errorBound;
        };
        constexpr std::size_t           groupDelay = (kEqualizerTaps - 1UZ) / 2UZ;
        constexpr std::array<Case, 4UZ> cases{Case{AdaptiveAlgorithm::Lms, 0.01, groupDelay, 3e-3}, Case{AdaptiveAlgorithm::Lms, 0.01, 0UZ, 5e-5}, Case{AdaptiveAlgorithm::Nlms, 0.3, groupDelay, 3e-3}, Case{AdaptiveAlgorithm::Nlms, 0.3, 0UZ, 5e-5}};

        for (const Case& one : cases) {
            const TrainedRun run   = runTrained(one.algorithm, one.stepSize, one.referenceDelay);
            const auto       label = std::format("{} mu {} with the reference at lag {}", algorithmName(one.algorithm), one.stepSize, one.referenceDelay);

            std::println("AdaptiveFir trained inversion, {}: mean squared error {:.4e} over the last {} of {} symbols, residual ISI {:.2f} dB, peak at {}, tap energy {:.4f}", label, run.meanSquaredError, kMeasureTail, kTrainSymbols, run.isiDb, run.peak, run.tapEnergy);

            expect(lt(run.meanSquaredError, one.errorBound)) << std::format("{}: mean squared error {:.4e}", label, run.meanSquaredError);
            expect(lt(run.isiDb, -20.)) << std::format("{}: residual ISI {:.2f} dB", label, run.isiDb);
            expect(eq(run.peak, one.referenceDelay)) << std::format("{}: the combined response peaks at {}", label, run.peak);
            expect(eq(run.resets, 0UZ)) << std::format("{}: {} divergence resets", label, run.resets);
        }
    };

    // Constant-modulus adaptation carries no reference and settles up to an arbitrary phase rotation, so the
    // modulus of the output and the shape of the combined response are what the run is measured on.
    "constant modulus converges on the same channel without a reference"_test = [] {
        constexpr double stepSize = 0.003;

        AdaptiveFir<double> filter;
        filter.configure(kEqualizerTaps, AdaptiveAlgorithm::Cma, stepSize, 1., kDivergenceBound);

        const auto symbols  = qpskSymbols(kTrainSymbols, kSymbolSeed);
        const auto received = throughChannel(symbols, kChannel);

        double      modulusError = 0.;
        std::size_t converged    = kTrainSymbols;
        std::size_t resets       = 0UZ;

        for (std::size_t k = 0UZ; k < kTrainSymbols; ++k) {
            filter.push(received[k]);
            const Complex y = filter.output();
            if (k >= kTrainSymbols - kMeasureTail) {
                const double slack = std::norm(y) - filter.referenceRatio();
                modulusError += slack * slack;
            }
            if (filter.adapt(y, Complex{})) {
                ++resets;
            }
            if (converged == kTrainSymbols && residualIsiDb(combinedResponse(filter.taps(), kChannel)) < -15.) {
                converged = k;
            }
        }
        modulusError /= static_cast<double>(kMeasureTail);

        const auto   response = combinedResponse(filter.taps(), kChannel);
        const double isi      = residualIsiDb(response);

        std::println("AdaptiveFir blind convergence, cma mu {}: mean (|y|^2 - R2)^2 {:.4e} over the last {} of {} symbols, residual ISI {:.2f} dB, first below -15 dB after {} symbols, peak at {}", stepSize, modulusError, kMeasureTail, kTrainSymbols, isi, converged, peakIndex(response));

        expect(lt(modulusError, 1e-2)) << std::format("the converged modulus error is {:.4e}", modulusError);
        expect(lt(isi, -20.)) << std::format("the residual ISI is {:.2f} dB", isi);
        expect(lt(converged, 3000UZ)) << std::format("the response first crosses -15 dB after {} symbols", converged);
        expect(eq(resets, 0UZ)) << std::format("{} divergence resets", resets);
    };

    // A step of 0.5 is several times the largest the eleven-tap window of this channel supports, so the taps run
    // away between resets and the bound is exercised over and over.
    "a step too large for the channel is reported and recovered"_test = [] {
        constexpr double stepSize = 0.5;

        AdaptiveFir<double> filter;
        filter.configure(kEqualizerTaps, AdaptiveAlgorithm::Lms, stepSize, 1., kDivergenceBound);
        const std::size_t delay = filter.groupDelay();

        const auto symbols  = qpskSymbols(kTrainSymbols, kSymbolSeed);
        const auto received = throughChannel(symbols, kChannel);

        std::size_t resets        = 0UZ;
        std::size_t firstReset    = kTrainSymbols;
        double      largestOutput = 0.;
        bool        outputsFinite = true;
        bool        tapsFinite    = true;
        bool        tapsRespiked  = true;
        bool        energyBounded = true;

        for (std::size_t k = 0UZ; k < kTrainSymbols; ++k) {
            filter.push(received[k]);

            const Complex y = filter.output();
            outputsFinite   = outputsFinite && isFinite(y);
            largestOutput   = std::max(largestOutput, std::abs(y));

            if (filter.adapt(y, k >= delay ? symbols[k - delay] : Complex{})) {
                ++resets;
                firstReset = std::min(firstReset, k);
                for (std::size_t i = 0UZ; i < filter.size(); ++i) {
                    tapsRespiked = tapsRespiked && filter.taps()[i] == spikeTap<double>(i, delay);
                }
            }
            energyBounded = energyBounded && filter.tapEnergy() <= kDivergenceBound;
            for (const Complex& tap : filter.taps()) {
                tapsFinite = tapsFinite && isFinite(tap);
            }
        }

        std::println("AdaptiveFir divergence, lms mu {}: {} resets over {} symbols, the first at symbol {}, largest |y| {:.4g}", stepSize, resets, kTrainSymbols, firstReset, largestOutput);

        expect(gt(resets, 100UZ)) << std::format("a step of {} must drive the taps past the bound repeatedly, and it reset {} times", stepSize, resets);
        expect(tapsRespiked) << "every reset must leave the taps on the center spike";
        expect(energyBounded) << std::format("the tap energy must stay within {:g} after every update", kDivergenceBound);
        expect(tapsFinite) << "every tap must stay finite";
        expect(outputsFinite) << std::format("every output must stay finite, and the largest was {:.4g}", largestOutput);
    };
};

const boost::ut::suite<"AdaptiveFir state"> adaptiveFirStateTests = [] {
    using namespace boost::ut;

    constexpr std::size_t kStateSymbols = 600UZ;

    "reset restores the spike and empties the history"_test = [] {
        const auto symbols  = qpskSymbols(kStateSymbols, kSymbolSeed);
        const auto received = throughChannel(symbols, kChannel);

        AdaptiveFir<double> filter;
        filter.configure(kEqualizerTaps, AdaptiveAlgorithm::Lms, 0.01, 1.);
        const std::size_t delay = filter.groupDelay();

        const auto run = [&] {
            std::vector<Complex> outputs(kStateSymbols);
            for (std::size_t k = 0UZ; k < kStateSymbols; ++k) {
                filter.push(received[k]);
                outputs[k]  = filter.output();
                std::ignore = filter.adapt(outputs[k], k >= delay ? symbols[k - delay] : Complex{});
            }
            return outputs;
        };

        const std::vector<Complex> first   = run();
        const std::vector<Complex> carried = run();
        filter.reset();
        const std::vector<Complex> restored = run();

        expect(std::ranges::equal(restored, first)) << "a reset filter must reproduce the first run bit for bit";
        expect(!std::ranges::equal(carried, first)) << "a continued run must carry the adapted taps and the history it left";

        filter.reset();
        for (std::size_t i = 0UZ; i < filter.size(); ++i) {
            expect(filter.taps()[i] == spikeTap<double>(i, delay)) << std::format("tap {} after a reset", i);
        }
        expect(eq(filter.tapEnergy(), 1.)) << std::format("the tap energy after a reset is {:g}", filter.tapEnergy());
        expect(filter.output() == Complex{}) << "the history is empty after a reset";
    };

    "respike restores the taps and keeps the history"_test = [] {
        const auto symbols  = qpskSymbols(kStateSymbols, kSymbolSeed);
        const auto received = throughChannel(symbols, kChannel);

        AdaptiveFir<double> filter;
        filter.configure(kEqualizerTaps, AdaptiveAlgorithm::Lms, 0.01, 1.);
        const std::size_t delay = filter.groupDelay();

        for (std::size_t k = 0UZ; k < kStateSymbols; ++k) {
            filter.push(received[k]);
            std::ignore = filter.adapt(filter.output(), k >= delay ? symbols[k - delay] : Complex{});
        }
        expect(gt(std::abs(filter.taps()[0]), 0.)) << "the run must move the taps off the spike for the restoration to mean anything";

        filter.respike();
        for (std::size_t i = 0UZ; i < filter.size(); ++i) {
            expect(filter.taps()[i] == spikeTap<double>(i, delay)) << std::format("tap {} after a respike", i);
        }
        expect(eq(filter.tapEnergy(), 1.)) << std::format("the tap energy after a respike is {:g}", filter.tapEnergy());
        expect(filter.output() == received[kStateSymbols - 1UZ - delay]) << "the sample pushed groupDelay samples ago is still in the history";
    };
};

const boost::ut::suite<"AdaptiveFir modulus reference"> adaptiveFirModulusTests = [] {
    using namespace boost::ut;

    "a unit-power constant-modulus alphabet has a modulus reference of one"_test = [] {
        using Alphabet = gr::digital::Constellation<double>;

        const Alphabet bpsk = Alphabet::bpsk();
        const Alphabet qpsk = Alphabet::qpsk();
        const Alphabet psk8 = Alphabet::psk8();

        const double bpskRatio = modulusReference(bpsk.points());
        const double qpskRatio = modulusReference(qpsk.points());
        const double psk8Ratio = modulusReference(psk8.points());

        std::println("AdaptiveFir modulus reference: bpsk {:.17g}, qpsk {:.17g}, 8psk {:.17g}", bpskRatio, qpskRatio, psk8Ratio);

        expect(lt(std::abs(bpskRatio - 1.), 1e-12)) << std::format("bpsk reports {:.17g}", bpskRatio);
        expect(lt(std::abs(qpskRatio - 1.), 1e-12)) << std::format("qpsk reports {:.17g}", qpskRatio);
        expect(lt(std::abs(psk8Ratio - 1.), 1e-12)) << std::format("8psk reports {:.17g}", psk8Ratio);
    };

    "a unit-power 16qam alphabet has a modulus reference of 1.32"_test = [] {
        const auto   qam16 = gr::digital::Constellation<double>::qam(16UZ);
        const double ratio = modulusReference(qam16.points());

        std::println("AdaptiveFir modulus reference: unit-power 16qam {:.17g}", ratio);

        expect(lt(std::abs(ratio - 1.32), 5e-4)) << std::format("16qam reports {:.17g}", ratio);
    };

    "the modulus reference needs an alphabet that carries power"_test = [] {
        expect(throws<std::invalid_argument>([] { std::ignore = modulusReference(std::span<const Complex>{}); })) << "an empty alphabet";

        const std::vector<Complex> silent(4UZ, Complex{});
        expect(throws<std::invalid_argument>([&silent] { std::ignore = modulusReference(std::span<const Complex>(silent)); })) << "an alphabet of zeros";
    };
};

const boost::ut::suite<"AdaptiveFir configuration"> adaptiveFirConfigurationTests = [] {
    using namespace boost::ut;

    "configure refuses tap counts outside the odd range"_test = [] {
        AdaptiveFir<double> filter;

        for (const std::size_t nTaps : {0UZ, 2UZ, 10UZ, kMaxTaps + 1UZ, kMaxTaps + 2UZ, 1000UZ}) {
            expect(throws<std::invalid_argument>([&filter, nTaps] { filter.configure(nTaps, AdaptiveAlgorithm::Lms, 0.01, 1.); })) << std::format("{} taps", nTaps);
        }
        for (const std::size_t nTaps : {1UZ, 3UZ, kMaxTaps}) {
            expect(nothrow([&filter, nTaps] { filter.configure(nTaps, AdaptiveAlgorithm::Lms, 0.01, 1.); })) << std::format("{} taps", nTaps);
        }
    };

    "configure refuses step sizes outside the open unit interval"_test = [] {
        AdaptiveFir<double> filter;

        for (const double stepSize : {0., 1., -0.1, 1.5, std::numeric_limits<double>::quiet_NaN()}) {
            expect(throws<std::invalid_argument>([&filter, stepSize] { filter.configure(kEqualizerTaps, AdaptiveAlgorithm::Lms, stepSize, 1.); })) << std::format("step size {}", stepSize);
        }
        expect(nothrow([&filter] { filter.configure(kEqualizerTaps, AdaptiveAlgorithm::Lms, 0.5, 1.); })) << "a step size in range";
    };

    "configure refuses a modulus reference and a divergence bound the rules cannot use"_test = [] {
        AdaptiveFir<double> filter;

        for (const double referenceRatio : {0., -1., std::numeric_limits<double>::quiet_NaN()}) {
            expect(throws<std::invalid_argument>([&filter, referenceRatio] { filter.configure(kEqualizerTaps, AdaptiveAlgorithm::Cma, 0.01, referenceRatio); })) << std::format("modulus reference {}", referenceRatio);
        }
        for (const double bound : {1., 0.5, 0., -10.}) {
            expect(throws<std::invalid_argument>([&filter, bound] { filter.configure(kEqualizerTaps, AdaptiveAlgorithm::Lms, 0.01, 1., bound); })) << std::format("divergence bound {}", bound);
        }
        expect(nothrow([&filter] { filter.configure(kEqualizerTaps, AdaptiveAlgorithm::Lms, 0.01, 1., 1.0001); })) << "a divergence bound just above the spike's own energy";
    };

    "configure reports back what the filter was given"_test = [] {
        AdaptiveFir<double> filter;
        filter.configure(31UZ, AdaptiveAlgorithm::Cma, 0.02, 1.32, 50.);

        expect(eq(filter.size(), 31UZ));
        expect(eq(filter.groupDelay(), 15UZ));
        expect(filter.algorithm() == AdaptiveAlgorithm::Cma) << "the algorithm the filter runs";
        expect(eq(filter.stepSize(), 0.02));
        expect(eq(filter.referenceRatio(), 1.32));
        expect(eq(filter.tapEnergy(), 1.)) << std::format("a freshly configured filter carries the spike's energy, and it reports {:g}", filter.tapEnergy());
    };

    "adaptiveAlgorithmFrom maps the algorithm names"_test = [] {
        using gr::digital::adaptiveAlgorithmFrom;

        expect(adaptiveAlgorithmFrom("lms") == AdaptiveAlgorithm::Lms);
        expect(adaptiveAlgorithmFrom("nlms") == AdaptiveAlgorithm::Nlms);
        expect(adaptiveAlgorithmFrom("cma") == AdaptiveAlgorithm::Cma);

        for (const std::string_view name : {"", "LMS", "Lms", "rls", "cma ", " lms", "constant_modulus"}) {
            expect(throws<std::invalid_argument>([name] { std::ignore = adaptiveAlgorithmFrom(name); })) << std::format("'{}' is not an algorithm name", name);
        }
    };
};

int main() { /* not needed for UT */ }
