#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iterator>
#include <numbers>
#include <print>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <gnuradio-4.0/algorithm/channel/NoisePower.hpp>
#include <gnuradio-4.0/algorithm/digital/CpmPulse.hpp>
#include <gnuradio-4.0/algorithm/filter/FilterDesign.hpp>
#include <gnuradio-4.0/algorithm/sync/MmseInterpolator.hpp>
#include <gnuradio-4.0/algorithm/sync/PreambleTone.hpp>
#include <gnuradio-4.0/algorithm/sync/TimingErrorDetector.hpp>

// The kernel is one dot product and one recursion, and neither is interesting on its own. What has to be pinned is
// that the model it assumes is the signal an AIS receiver actually carries, that the phase it returns points the way
// a timing tag says it does, and that its accuracy sits at the Cramer-Rao bound rather than near it. So this file
// builds an FSK demodulator's chain out of the kernels such a chain is made of - CpmPulse for the modulator,
// gr::filter::design for both filters - and measures the estimator on that.

namespace {

using gr::sync::PreambleToneEstimator;
using gr::sync::PreambleToneFit;

using CD = std::complex<double>;

/// ITU-R M.1371 Annex 2 section 3 at a 48 kHz discriminator rate: five samples a symbol exactly.
constexpr double      kSampleRate = 48000.0;
constexpr double      kSymbolRate = 9600.0;
constexpr double      kModIndex   = 0.5;
constexpr double      kBt         = 0.4;
constexpr std::size_t kSps        = 5UZ;
constexpr std::size_t kSpan       = 3UZ; ///< symbols the Gaussian frequency pulse spans
constexpr std::size_t kPreamble   = 24UZ;
constexpr std::size_t kWindow     = kPreamble * kSps;
constexpr double      kToneBin    = 0.5 / static_cast<double>(kSps);

constexpr double kPi = std::numbers::pi;

/// @brief The recipe's discriminator gain: a full-response symbol then averages to its own amplitude.
[[nodiscard]] constexpr double discriminatorGain() noexcept { return static_cast<double>(kSps) / (kPi * kModIndex); }

/// @brief The training sequence as an AIS scene builds it: alternating, opening on the level an NRZI encoder starts at.
[[nodiscard]] std::vector<float> alternating(std::size_t count) {
    std::vector<float> symbols(count);
    for (std::size_t k = 0UZ; k < count; ++k) {
        symbols[k] = (k % 2UZ) == 0UZ ? -1.f : 1.f;
    }
    return symbols;
}

[[nodiscard]] std::vector<CD> modulate(std::span<const float> symbols) {
    gr::digital::CpmPulse<float> pulse;
    pulse.configure(gr::digital::CpmPulseShape::Gaussian, kSpan, kSps, kModIndex, kBt);

    std::vector<double> increments(symbols.size() * kSps);
    std::ignore = pulse.incrementsFor(symbols, std::span<double>(increments));

    std::vector<CD> baseband(increments.size());
    double          phase = 0.0;
    for (std::size_t k = 0UZ; k < increments.size(); ++k) {
        phase += increments[k];
        baseband[k] = std::polar(1.0, phase);
    }
    return baseband;
}

[[nodiscard]] std::vector<float> designedLowpass(double cutoffSymbolRates) {
    gr::filter::fir::design::FilterSpec spec;
    spec.sampleRate      = kSampleRate;
    spec.cutoff          = cutoffSymbolRates * kSymbolRate;
    spec.transitionWidth = 0.5 * cutoffSymbolRates * kSymbolRate;
    return gr::filter::fir::design::designLowpass(spec);
}

/// @brief The taps' response at @p frequency cycles per sample, referred to the design's own center tap.
[[nodiscard]] CD responseAt(std::span<const float> taps, double frequency) {
    const auto center = static_cast<double>(taps.size() - 1UZ) / 2.0;
    CD         sum{};
    for (std::size_t k = 0UZ; k < taps.size(); ++k) {
        sum += static_cast<double>(taps[k]) * std::polar(1.0, -2.0 * kPi * frequency * (static_cast<double>(k) - center));
    }
    return sum;
}

/// @brief Linear-phase filtering with the design's own group delay removed, so a cascade shares one time axis.
template<typename T>
[[nodiscard]] std::vector<T> filtered(std::span<const T> signal, std::span<const float> taps) {
    const std::size_t lead = (taps.size() - 1UZ) / 2UZ;
    std::vector<T>    out(signal.size());
    for (std::size_t n = 0UZ; n < signal.size(); ++n) {
        const std::size_t index = n + lead;
        const std::size_t first = index + 1UZ > signal.size() ? index + 1UZ - signal.size() : 0UZ;
        const std::size_t last  = std::min(taps.size(), index + 1UZ);
        T                 sum{};
        for (std::size_t k = first; k < last; ++k) {
            sum += static_cast<double>(taps[k]) * signal[index - k];
        }
        out[n] = sum;
    }
    return out;
}

/// @brief The recipe's discriminator: the phase advance between consecutive samples, scaled onto the symbol grid.
[[nodiscard]] std::vector<double> discriminate(std::span<const CD> baseband) {
    std::vector<double> out(baseband.size(), 0.0);
    for (std::size_t n = 1UZ; n < baseband.size(); ++n) {
        out[n] = discriminatorGain() * std::arg(baseband[n] * std::conj(baseband[n - 1UZ]));
    }
    return out;
}

struct Rng {
    std::mt19937_64                  engine;
    std::normal_distribution<double> normal{0.0, 1.0};

    explicit Rng(std::uint64_t seed) : engine(seed) {}

    [[nodiscard]] double        gaussian() { return normal(engine); }
    [[nodiscard]] std::uint64_t bits() { return engine(); }
    [[nodiscard]] CD            circular(double power) {
        const double scale = std::sqrt(0.5 * power);
        return {scale * gaussian(), scale * gaussian()};
    }
};

/// What one pass of the recipe's chain leaves at each of the two taps the estimator may sit at.
struct Chain {
    std::vector<double> discriminator{}; ///< before the post-detection lowpass
    std::vector<double> lowpass{};       ///< after it, which is where the block sits
};

[[nodiscard]] Chain runChain(std::span<const CD> baseband, double esN0Db, Rng* rng, std::span<const float> channel, std::span<const float> post) {
    std::vector<CD> noisy(baseband.begin(), baseband.end());
    if (rng != nullptr) {
        const double power = gr::channel::noisePowerFor(esN0Db, 1.0, static_cast<double>(kSps));
        for (CD& sample : noisy) {
            sample += rng->circular(power);
        }
    }
    const std::vector<CD> band = filtered<CD>(std::span<const CD>(noisy), channel);

    Chain chain;
    chain.discriminator = discriminate(std::span<const CD>(band));
    chain.lowpass       = filtered<double>(std::span<const double>(chain.discriminator), post);
    return chain;
}

/// @brief The single-bin fit `X = (2/N) sum v[k] exp(-j 2 pi f k)` over a named window, at any frequency.
[[nodiscard]] CD bin(std::span<const double> window, double frequency) {
    CD sum{};
    for (std::size_t k = 0UZ; k < window.size(); ++k) {
        sum += window[k] * std::polar(1.0, -2.0 * kPi * frequency * static_cast<double>(k));
    }
    return (2.0 / static_cast<double>(window.size())) * sum;
}

[[nodiscard]] double meanOf(std::span<const double> values) {
    double sum = 0.0;
    for (const double v : values) {
        sum += v;
    }
    return sum / static_cast<double>(values.size());
}

[[nodiscard]] double standardDeviation(std::span<const double> values) {
    const double mean = meanOf(values);
    double       sum  = 0.0;
    for (const double v : values) {
        sum += (v - mean) * (v - mean);
    }
    return std::sqrt(sum / static_cast<double>(values.size() - 1UZ));
}

/// @brief (13.3d): the one-sigma timing error a preamble of @p symbols affords at `Es/N0 = gamma`, in symbols.
[[nodiscard]] double timingBound(double symbols, double gamma) { return 0.8418 / std::sqrt(symbols * gamma); }

/// @brief NRZI: hold the level on a one, transition on a zero, so a run of zeros is an alternating channel pattern.
[[nodiscard]] std::vector<float> nrziData(std::size_t count, Rng& rng) {
    std::vector<float> symbols(count);
    std::uint8_t       level = 0U;
    std::uint64_t      pool  = 0ULL;
    for (std::size_t k = 0UZ; k < count; ++k) {
        if (k % 64UZ == 0UZ) {
            pool = rng.bits();
        }
        const auto bit = static_cast<std::uint8_t>((pool >> (k % 64UZ)) & 1ULL);
        level          = static_cast<std::uint8_t>(bit != 0U ? level : (level ^ 1U));
        symbols[k]     = level != 0U ? 1.f : -1.f;
    }
    return symbols;
}

} // namespace

const boost::ut::suite<"preamble tone"> preambleToneTests = [] {
    using namespace boost::ut;

    const std::vector<float> channel = designedLowpass(0.6);
    const std::vector<float> post    = designedLowpass(0.5);

    "the training tone is the model the estimator assumes"_test = [&channel, &post] {
        // Every figure comes from CpmPulse's own pulse and gr::filter::design's own designs; the amplitudes are what
        // the model rests on, and a pulse or a design that changed would move them.
        const std::vector<float>  symbols  = alternating(200UZ);
        const std::vector<CD>     baseband = modulate(std::span<const float>(symbols));
        const std::vector<double> bare     = discriminate(std::span<const CD>(baseband));

        const auto   window      = std::span<const double>(bare).subspan(200UZ, kWindow);
        const double fundamental = std::abs(bin(window, kToneBin));
        const double third       = std::abs(bin(window, 3.0 * kToneBin));
        const double fifth       = std::abs(bin(window, 5.0 * kToneBin));
        double       peak        = 0.0;
        for (const double v : window) {
            peak = std::max(peak, std::abs(v));
        }
        std::println("[record] alternating tone at the discriminator: fundamental {:.6f}, third {:.6f}, fifth {:.6f}, symbol-instant value {:.6f}", fundamental, third, fifth, peak);
        expect(lt(std::abs(fundamental / 0.754481 - 1.0), 0.01)) << "the fundamental at Rs/2";
        expect(lt(std::abs(third / 0.003655 - 1.0), 0.01)) << "the third harmonic";
        expect(lt(std::abs(fifth / 0.000248 - 1.0), 0.01)) << "the fifth harmonic";
        expect(lt(std::abs(peak / 0.750700 - 1.0), 0.01)) << "a BT 0.4 Gaussian takes a quarter off the alternating pattern";

        const double channelResponse = std::abs(responseAt(std::span<const float>(channel), kToneBin));
        const double postResponse    = std::abs(responseAt(std::span<const float>(post), kToneBin));
        std::println("[record] channel filter {} taps, |H(f0)| {:.6f}; post-detection lowpass {} taps, |H(f0)| {:.6f} ({:.3f} dB)", channel.size(), channelResponse, post.size(), postResponse, 20.0 * std::log10(postResponse));
        expect(eq(channel.size(), 61UZ));
        expect(eq(post.size(), 73UZ));
        expect(lt(std::abs(postResponse / 0.499766 - 1.0), 1e-4)) << "the timing tone sits exactly on the lowpass's stated -6 dB point";
        expect(lt(std::abs(channelResponse / 0.950401 - 1.0), 1e-4)) << "and inside the channel filter's passband";

        const Chain  clean        = runChain(std::span<const CD>(baseband), 0.0, nullptr, std::span<const float>(channel), std::span<const float>(post));
        const double afterChannel = std::abs(bin(std::span<const double>(clean.discriminator).subspan(200UZ, kWindow), kToneBin));
        const double afterPost    = std::abs(bin(std::span<const double>(clean.lowpass).subspan(200UZ, kWindow), kToneBin));
        std::println("[record] tone amplitude after the channel filter {:.6f}, after the post-detection lowpass {:.6f}", afterChannel, afterPost);
        expect(lt(std::abs(afterChannel / 0.706962 - 1.0), 1e-4)) << "the channel filter costs 6.3 per cent of the tone";
        expect(lt(std::abs(afterPost / 0.353316 - 1.0), 1e-4)) << "and the post-detection lowpass halves what is left";
    };

    "the lattice reproduces the offset to the next extremum, and its sign"_test = [] {
        // This is the test that would catch a sign flip between the estimator and a timing tag, and nothing
        // downstream would: a lattice reflected about the window's first sample still decodes, one symbol late.
        for (const double sps : {4.0, 5.0, 4.3}) {
            PreambleToneEstimator estimator;
            estimator.configure(sps, static_cast<double>(kPreamble));
            for (int step = 0; step < 21; ++step) {
                const double phase   = -kPi + 2.0 * kPi * static_cast<double>(step) / 20.0;
                const double instant = estimator.instantAfter(phase, 0.0);
                // the instant the lattice names is where the continuous tone's own derivative vanishes
                const double carrier = std::cos(2.0 * kPi * (0.5 / sps) * instant + phase);
                expect(ge(instant, -1e-12)) << "the first instant at or after the window's first sample";
                expect(lt(instant, sps + 1e-9)) << "and inside one symbol of it";
                expect(lt(std::abs(std::abs(carrier) - 1.0), 1e-9)) << "sps=" << sps << " phase=" << phase;
            }
        }

        const std::vector<float>  symbols  = alternating(200UZ);
        const std::vector<CD>     baseband = modulate(std::span<const float>(symbols));
        const std::vector<double> bare     = discriminate(std::span<const CD>(baseband));

        PreambleToneEstimator estimator;
        estimator.configure(static_cast<double>(kSps), static_cast<double>(kPreamble));

        std::vector<std::size_t> offsets;
        std::string              phases;
        for (std::size_t start = 200UZ; start <= 205UZ; ++start) {
            const PreambleToneFit fit  = estimator.fit(std::span<const double>(bare).subspan(start, kWindow));
            const double          next = estimator.instantAfter(fit.phase, 0.0);
            // an instant on the window's own first sample is equally the zeroth lattice point and the sps-th, and
            // which of the two a phase of exactly pi lands on is a rounding decision, so the offset is read modulo
            offsets.push_back(static_cast<std::size_t>(std::llround(next)) % kSps);
            std::format_to(std::back_inserter(phases), "{}{:.5f}", phases.empty() ? "" : ", ", fit.phase);
            expect(lt(std::abs(next - std::round(next)), 1e-6)) << "the noiseless preamble's extrema sit on samples";
        }
        std::println("[record] windows at n0 = 200..205: phases {}; offset to the next extremum {}", phases, offsets);
        const std::vector<std::size_t> expected{2UZ, 1UZ, 0UZ, 4UZ, 3UZ, 2UZ};
        expect(that % (offsets == expected)) << "the AIS preamble's extrema fall on absolute samples congruent to 2 modulo 5";
    };

    "the estimator meets the Cramer-Rao bound"_test = [&channel, &post] {
        // The reference is the noiseless fit of the same window, so what is measured is the estimator's own error and
        // not the chain's group delay.
        constexpr std::size_t kTrials = 1000UZ;
        constexpr std::size_t kStart  = 200UZ;

        const std::vector<float> symbols  = alternating(80UZ);
        const std::vector<CD>    baseband = modulate(std::span<const float>(symbols));

        Rng         rng(0xF6E57ULL);
        const Chain clean     = runChain(std::span<const CD>(baseband), 0.0, nullptr, std::span<const float>(channel), std::span<const float>(post));
        const CD    reference = bin(std::span<const double>(clean.lowpass).subspan(kStart, kWindow), kToneBin);

        for (const double esN0Db : {20.0, 15.0, 10.0, 5.0}) {
            std::vector<double> errors(kTrials);
            for (std::size_t trial = 0UZ; trial < kTrials; ++trial) {
                const Chain noisy = runChain(std::span<const CD>(baseband), esN0Db, &rng, std::span<const float>(channel), std::span<const float>(post));
                const CD    fit   = bin(std::span<const double>(noisy.lowpass).subspan(kStart, kWindow), kToneBin);
                errors[trial]     = std::arg(fit * std::conj(reference));
            }
            const double spread = standardDeviation(std::span<const double>(errors));
            const double gamma  = std::pow(10.0, esN0Db / 10.0);
            const double bound  = timingBound(static_cast<double>(kPreamble), gamma);
            const double tau    = spread / kPi;
            std::println("[record] Es/N0 {:.0f} dB over {} trials: sd(phi) {:.5f} rad, sd(tau) {:.5f} symbol, bound {:.5f}, ratio {:.3f}", esN0Db, kTrials, spread, tau, bound, tau / bound);

            if (esN0Db > 7.0) {
                expect(lt(std::abs(tau / bound - 1.0), 0.15)) << "the estimator is at the bound from 20 dB down to 10 dB";
            } else {
                expect(gt(tau / bound, 1.05)) << "below the FM threshold the departure is an excess; an estimator that beats its own bound is a broken measurement";
            }
            const double measured = esN0Db > 17.0 ? 0.01669 : (esN0Db > 12.0 ? 0.03227 : (esN0Db > 7.0 ? 0.05515 : 0.12097));
            expect(lt(std::abs(tau / measured - 1.0), 0.20)) << "against the spread this chain measures at " << esN0Db << " dB";
        }
    };

    "the recursion decides where to fit and the fit decides what the answer is"_test = [] {
        // The sliding form sits on the unit circle and so is marginally stable by construction; what has to hold is
        // that its drift never reaches an estimate, and that it stays good enough to pick a window.
        const std::vector<float>  symbols = alternating(400UZ);
        const std::vector<CD>     wave    = modulate(std::span<const float>(symbols));
        const std::vector<double> stream  = discriminate(std::span<const CD>(wave));

        PreambleToneFit first{};
        for (const std::size_t chunk : {1UZ, 7UZ, 120UZ, 4096UZ}) {
            PreambleToneEstimator estimator;
            estimator.configure(static_cast<double>(kSps), static_cast<double>(kPreamble));
            for (std::size_t base = 0UZ; base < stream.size(); base += chunk) {
                for (std::size_t k = base; k < std::min(base + chunk, stream.size()); ++k) {
                    std::ignore = estimator.push(stream[k]);
                }
            }
            const PreambleToneFit fit = estimator.fit(std::span<const double>(stream).subspan(300UZ, kWindow));
            if (chunk == 1UZ) {
                first = fit;
            }
            expect(that % (fit.phase == first.phase)) << "the refit is bit-identical however the stream was chunked, chunk " << chunk;
            expect(that % (fit.amplitude == first.amplitude));
            expect(that % (fit.statistic == first.statistic));
        }

        constexpr std::size_t kLong = 1000000UZ;
        Rng                   noise(0xD21F7ULL);

        PreambleToneEstimator estimator;
        estimator.configure(static_cast<double>(kSps), static_cast<double>(kPreamble));
        std::vector<double> history(kWindow, 0.0);
        double              worst = 0.0;
        for (std::size_t n = 0UZ; n < kLong; ++n) {
            const double sample    = std::cos(2.0 * kPi * kToneBin * static_cast<double>(n)) + 0.5 * noise.gaussian();
            const double recursive = estimator.push(sample);
            history[n % kWindow]   = sample;
            if (n + 1UZ >= kWindow && (n % 1024UZ) == 0UZ) {
                std::vector<double> ordered(kWindow);
                for (std::size_t k = 0UZ; k < kWindow; ++k) {
                    ordered[k] = history[(n + 1UZ + k) % kWindow];
                }
                const double exact = estimator.fit(std::span<const double>(ordered)).statistic;
                worst              = std::max(worst, std::abs(recursive - exact) / std::max(exact, 1e-12));
            }
        }
        std::println("[record] the sliding statistic against an O(N) recomputation over {} samples: worst relative deviation {:.3e}", kLong, worst);
        expect(lt(worst, 1e-6)) << "the recursion stays good enough to choose a window";
    };

    "the post-detection lowpass is transparent to the estimator"_test = [&channel, &post] {
        // The filter scales the tone by |H(f0)| and the noise power at f0 by |H(f0)|^2, and the bound is their ratio,
        // so the six decibels the amplitude loses cost nothing and the block may sit on either side.
        constexpr std::size_t kTrials = 800UZ;
        constexpr std::size_t kStart  = 200UZ;

        const std::vector<float> symbols  = alternating(80UZ);
        const std::vector<CD>    baseband = modulate(std::span<const float>(symbols));
        const Chain              clean    = runChain(std::span<const CD>(baseband), 0.0, nullptr, std::span<const float>(channel), std::span<const float>(post));

        const CD     beforeFit = bin(std::span<const double>(clean.discriminator).subspan(kStart, kWindow), kToneBin);
        const CD     afterFit  = bin(std::span<const double>(clean.lowpass).subspan(kStart, kWindow), kToneBin);
        const double turned    = std::arg(afterFit * std::conj(beforeFit));
        std::println("[record] the same window fitted before and after the post-detection lowpass turns by {:.3e} rad", turned);
        expect(lt(std::abs(turned), 1e-6)) << "a linear-phase design with its group delay removed turns the tone by nothing";

        Rng rng(0xB0BULL);
        for (const double esN0Db : {20.0, 10.0}) {
            std::vector<double> before(kTrials);
            std::vector<double> after(kTrials);
            for (std::size_t trial = 0UZ; trial < kTrials; ++trial) {
                const Chain noisy = runChain(std::span<const CD>(baseband), esN0Db, &rng, std::span<const float>(channel), std::span<const float>(post));
                before[trial]     = std::arg(bin(std::span<const double>(noisy.discriminator).subspan(kStart, kWindow), kToneBin) * std::conj(beforeFit));
                after[trial]      = std::arg(bin(std::span<const double>(noisy.lowpass).subspan(kStart, kWindow), kToneBin) * std::conj(afterFit));
            }
            const double spreadBefore = standardDeviation(std::span<const double>(before));
            const double spreadAfter  = standardDeviation(std::span<const double>(after));
            std::println("[record] Es/N0 {:.0f} dB: sd(phi) before the lowpass {:.5f}, after {:.5f}, ratio {:.3f}", esN0Db, spreadBefore, spreadAfter, spreadAfter / spreadBefore);
            expect(lt(std::abs(spreadAfter / spreadBefore - 1.0), 0.10)) << "the two taps carry the same accuracy to within a tenth";
        }
    };

    "the rate the preamble affords is not worth estimating"_test = [] {
        // Estimating the frequency jointly costs a factor of four in phase variance and buys a period five times too
        // coarse for the frame that follows, which is why the tag carries the nominal period instead.
        constexpr double kFitted = 0.706962; // the tone's amplitude after the channel filter
        // S(f0) * gamma at the AIS constants: the discriminator differences consecutive samples and scales by
        // G = sps/(pi*h), which takes the carrier's phase-noise density to G^2 * 4 sin^2(pi*f0) * |H_ch(f0)|^2 / 2.
        constexpr double kDensity   = 8.7402;
        constexpr double kFrameNeed = 5e-4; // the fractional period error 200 symbols of drift under 0.1 symbol admit

        const double gamma   = 100.0; // Es/N0 = 20 dB
        const double samples = static_cast<double>(kWindow);
        const double density = kDensity / gamma;
        const double omega   = 24.0 * density / (kFitted * kFitted * samples * (samples * samples - 1.0));
        const double drift   = std::sqrt(omega) / (kPi / static_cast<double>(kSps));
        std::println("[record] a joint rate estimate over {} symbols at 20 dB: sd(omega) {:.4e} rad/sample, sd(period)/period {:.4e} ({:.0f} ppm)", kPreamble, std::sqrt(omega), drift, drift * 1e6);
        expect(lt(std::abs(drift / 2.4e-3 - 1.0), 0.05)) << "the joint estimate's own variance at N = 120 and Es/N0 = 20 dB";
        expect(lt(kFrameNeed, drift)) << "what a 200-symbol frame needs is finer than 24 alternating symbols can measure";

        const double jointPhase = 4.0 * density * (2.0 * samples - 1.0) / (kFitted * kFitted * samples * (samples + 1.0));
        const double phaseOnly  = 2.0 * density / (kFitted * kFitted * samples);
        std::println("[record] estimating the frequency too costs a factor {:.2f} in phase variance", jointPhase / phaseOnly);
        expect(lt(std::abs(jointPhase / phaseOnly - 4.0), 0.1)) << "a factor of four, six decibels, for a frequency the receiver already knows";
    };

    "the null distribution runs heavier than the white-noise model, and the threshold clears it anyway"_test = [&channel, &post] {
        // Beta(1, N/2 - 1) is the distribution of one of N/2 *independent* periodogram bins against their sum, and
        // the chain's noise at this tap is not white: the post-detection lowpass has its own -6 dB point at the
        // tone's frequency, so the window's energy sits closer to that bin than a white model puts it and the tail
        // runs heavier. What the default threshold rests on is therefore the measured tail, not the model's
        // extrapolation of it, and the long sweep below is where that margin is read off.
        constexpr std::size_t kWindows = 200000UZ;
        constexpr std::size_t kLongRun = 400000UZ;

        const auto sweep = [&channel, &post](std::size_t count, std::uint64_t seed, std::span<const double> levels, std::span<std::size_t> exceeded) {
            Rng                   rng(seed);
            const std::vector<CD> silence(count + 4UZ * kWindow, CD{});
            const Chain           noise = runChain(std::span<const CD>(silence), 20.0, &rng, std::span<const float>(channel), std::span<const float>(post));

            PreambleToneEstimator estimator;
            estimator.configure(static_cast<double>(kSps), static_cast<double>(kPreamble));

            double      total   = 0.0;
            double      largest = 0.0;
            std::size_t counted = 0UZ;
            for (std::size_t n = 2UZ * kWindow; n < noise.lowpass.size() && counted < count; ++n) {
                const double value = estimator.push(noise.lowpass[n]);
                if (!estimator.filled()) {
                    continue;
                }
                ++counted;
                total += value;
                largest = std::max(largest, value);
                for (std::size_t which = 0UZ; which < levels.size(); ++which) {
                    exceeded[which] += value > levels[which] ? 1UZ : 0UZ;
                }
            }
            return std::array<double, 3UZ>{total / static_cast<double>(counted), largest, static_cast<double>(counted)};
        };

        constexpr std::array<double, 3UZ> kLevels{0.05, 0.10, 0.15};
        std::array<std::size_t, 3UZ>      exceeded{};
        const auto                        summary = sweep(kWindows, 0x0E15EULL, std::span<const double>(kLevels), std::span<std::size_t>(exceeded));

        const double model = 2.0 / static_cast<double>(kWindow);
        std::println("[record] {:.0f} noise-only windows: mean Lambda {:.5f} against Beta(1, N/2-1)'s {:.5f}, maximum {:.4f}", summary[2], summary[0], model, summary[1]);
        expect(lt(std::abs(summary[0] / model - 1.0), 0.15)) << "the coloring leaves the mean near the model, which is what makes Lambda's scale invariance usable";

        for (std::size_t which = 0UZ; which < kLevels.size(); ++which) {
            const double measured = static_cast<double>(exceeded[which]) / summary[2];
            const double tail     = std::pow(1.0 - kLevels[which], static_cast<double>(kWindow) / 2.0 - 1.0);
            std::println("[record] P(Lambda > {:.2f}): measured {:.3e}, model {:.3e}, ratio {:.2f}", kLevels[which], measured, tail, measured / tail);
            expect(gt(measured, 0.8 * tail)) << "colored noise concentrates energy toward the bin, so the measured tail is never the lighter one, at " << kLevels[which];
        }

        constexpr std::array<double, 3UZ> kNear = {0.25, 0.30, 0.35};
        std::array<std::size_t, 3UZ>      reached{};
        const auto                        longRun = sweep(kLongRun, 0xF00DULL, std::span<const double>(kNear), std::span<std::size_t>(reached));
        std::println("[record] {:.0f} further noise-only windows: maximum Lambda {:.4f}; windows above 0.25, 0.30 and 0.35: {}, {}, {}", longRun[2], longRun[1], reached[0], reached[1], reached[2]);
        // A rate rather than a count of exactly zero: the noise comes from std::normal_distribution, whose stream
        // is not specified by the standard, so an exact extreme value is an assertion about one library's sequence.
        const double rate = static_cast<double>(reached[2]) / longRun[2];
        expect(lt(rate, 1.0e-4)) << "the default threshold sits above the noise-only tail: " << reached[2] << " windows of " << longRun[2] << " reached it";
    };

    "the preamble is declared wherever the chain can decode at all"_test = [&channel, &post] {
        // What the threshold has to exclude is a preset taken from a region with no tone: a preamble-shaped data
        // region is harmless, the tone being genuinely there and its phase genuinely the symbol phase.
        constexpr std::size_t kWindows = 2472UZ;
        constexpr double      kDefault = 0.35;

        const std::vector<float> symbols  = alternating(kPreamble + 32UZ);
        const std::vector<CD>    baseband = modulate(std::span<const float>(symbols));
        const std::size_t        start    = 16UZ * kSps;

        Rng rng(0xDE7ECULL);
        for (const double esN0Db : {20.0, 18.0, 16.0, 14.0, 10.0}) {
            PreambleToneEstimator estimator;
            estimator.configure(static_cast<double>(kSps), static_cast<double>(kPreamble));

            std::size_t declared = 0UZ;
            double      total    = 0.0;
            double      smallest = 1.0;
            for (std::size_t trial = 0UZ; trial < kWindows; ++trial) {
                const Chain           noisy = runChain(std::span<const CD>(baseband), esN0Db, &rng, std::span<const float>(channel), std::span<const float>(post));
                const PreambleToneFit fit   = estimator.fit(std::span<const double>(noisy.lowpass).subspan(start, kWindow));
                declared += fit.statistic > kDefault ? 1UZ : 0UZ;
                total += fit.statistic;
                smallest = std::min(smallest, fit.statistic);
            }
            const double rate = static_cast<double>(declared) / static_cast<double>(kWindows);
            std::println("[record] Es/N0 {:.0f} dB over {} aligned preamble windows: mean Lambda {:.4f}, minimum {:.4f}, P(Lambda > 0.35) = {:.4f}", esN0Db, kWindows, total / static_cast<double>(kWindows), smallest, rate);
            if (esN0Db > 12.0) {
                expect(eq(declared, kWindows)) << "every aligned window at " << esN0Db << " dB";
            } else {
                expect(ge(rate, 0.93)) << "and all but a few per cent at 10 dB, which is already below the chain's own noise knee";
            }
        }

        constexpr std::size_t    kDataWindows = 200000UZ;
        Rng                      data(0xDA7AULL);
        const std::vector<float> line      = nrziData(kDataWindows / kSps + 4UZ * kPreamble, data);
        const std::vector<CD>    modulated = modulate(std::span<const float>(line));
        const Chain              stream    = runChain(std::span<const CD>(modulated), 20.0, &data, std::span<const float>(channel), std::span<const float>(post));

        PreambleToneEstimator estimator;
        estimator.configure(static_cast<double>(kSps), static_cast<double>(kPreamble));
        double      largest = 0.0;
        std::size_t counted = 0UZ;
        for (std::size_t n = 2UZ * kWindow; n + 2UZ * kWindow < stream.lowpass.size(); ++n) {
            const double value = estimator.push(stream.lowpass[n]);
            if (estimator.filled()) {
                largest = std::max(largest, value);
                ++counted;
            }
        }
        std::println("[record] {} windows of NRZI-coded random data at 20 dB: maximum Lambda {:.4f}", counted, largest);
        expect(lt(largest, 0.30)) << "the default threshold stands clear of the highest data window measured";
    };

    "the loop's own jitter is what remains once the preset removes acquisition"_test = [&channel, &post] {
        // With the preset there is no acquisition transient left, so tracking jitter alone decides a slot, and
        // measuring the detector on this chain says which loop bandwidths the preset can rescue and which it cannot.
        // The S-curve's own zero crossing is the instant, so nothing here assumes where the pulse peaks.
        constexpr std::size_t kSymbols = 20000UZ;

        Rng                      data(0x7ED0ULL);
        const std::vector<float> line      = nrziData(kSymbols + 64UZ, data);
        const std::vector<CD>    modulated = modulate(std::span<const float>(line));
        const Chain              clean     = runChain(std::span<const CD>(modulated), 0.0, nullptr, std::span<const float>(channel), std::span<const float>(post));

        Rng         noisy(0x9A15ULL);
        const Chain loud = runChain(std::span<const CD>(modulated), 20.0, &noisy, std::span<const float>(channel), std::span<const float>(post));

        const gr::sync::MmseInterpolatorBank bank(8, 128, 0.25, false);
        const auto                           lead = static_cast<double>(bank.delay());
        // the pulse is causal over three symbols, so a symbol peaks near s*sps + (span*sps - 1)/2; the two
        // linear-phase designs carry no delay once theirs is removed, and the S-curve says where the instant is
        const double anchor = static_cast<double>(kSpan * kSps - 1UZ) / 2.0;

        const auto errorsAt = [&](double offset, const Chain& source) {
            std::vector<double> errors;
            errors.reserve(kSymbols);
            double previous         = 0.0;
            double previousDecision = 0.0;
            for (std::size_t s = 8UZ; s + 8UZ < kSymbols; ++s) {
                const double position = anchor + (static_cast<double>(s) + offset) * static_cast<double>(kSps) - lead;
                const double base     = std::floor(position);
                const auto   index    = static_cast<std::size_t>(base);
                const auto   row      = static_cast<std::size_t>((position - base) * static_cast<double>(bank.steps()) + 0.5);
                const double value    = bank.interpolate(source.lowpass.data() + index, std::min(row, bank.steps()));
                const double decision = value > 0.0 ? 1.0 : -1.0;
                if (s > 8UZ) {
                    errors.push_back(gr::sync::muellerMullerError(value, previous, decision, previousDecision));
                }
                previous         = value;
                previousDecision = decision;
            }
            return errors;
        };

        // a least-squares line through the S-curve's central five points: its slope is Kted, its root the instant
        constexpr std::array<double, 5UZ> kOffsets{-0.10, -0.05, 0.0, 0.05, 0.10};
        double                            sumX  = 0.0;
        double                            sumY  = 0.0;
        double                            sumXX = 0.0;
        double                            sumXY = 0.0;
        for (const double offset : kOffsets) {
            const std::vector<double> errors = errorsAt(offset, clean);
            const double              mean   = meanOf(std::span<const double>(errors));
            sumX += offset;
            sumY += mean;
            sumXX += offset * offset;
            sumXY += offset * mean;
        }
        const auto   points    = static_cast<double>(kOffsets.size());
        const double slope     = (points * sumXY - sumX * sumY) / (points * sumXX - sumX * sumX);
        const double intercept = (sumY - slope * sumX) / points;
        const double instant   = -intercept / slope;

        const double selfNoise = standardDeviation(std::span<const double>(errorsAt(instant, clean)));
        const double withNoise = standardDeviation(std::span<const double>(errorsAt(instant, loud)));
        std::println("[record] Mueller & Muller on this chain: Kted {:.4f} per symbol, S-curve zero at {:+.4f} symbol, sd(e) {:.4f} noiseless and {:.4f} at Es/N0 20 dB", slope, instant, selfNoise, withNoise);
        // The error is added to the tracked period, and a period made longer moves the next instant later, so a
        // sampler that is already late has to produce a negative error: the S-curve falls with the offset, and the
        // gain the loop is given is its magnitude.
        expect(lt(slope, 0.0)) << "the detector's sign convention closes the loop on this chain";
        expect(lt(std::abs(instant), 0.25)) << "and its zero crossing is where the pulse peaks";
        expect(gt(selfNoise, 0.0)) << "a BT 0.4 discriminator stream is not an ideal Nyquist channel, so the self-noise is not zero";

        std::string rows;
        double      widest = 0.0;
        for (const double bandwidth : {0.002, 0.010, 0.020, 0.050, 0.100}) {
            const double jitter    = std::sqrt(2.0 * bandwidth) * withNoise / std::abs(slope);
            const double perSymbol = std::erfc(0.25 / (jitter * std::numbers::sqrt2));
            const double lost      = 200.0 * (1.0 - std::pow(1.0 - perSymbol, 200.0));
            std::format_to(std::back_inserter(rows), "{}Bn*T {:.3f}: sd(tau) {:.4f}, P(|tau| > 0.25) {:.2e}, {:.1f} slots lost of 200", rows.empty() ? "" : "; ", bandwidth, jitter, perSymbol, lost);
            widest = lost;
        }
        std::println("[record] tracking jitter alone predicts: {}", rows);
        expect(gt(widest, 40.0)) << "the widest bandwidth's loss is tracking jitter, which a phase preset cannot touch";
        expect(lt(widest, 130.0));
    };

    "the chain's tolerance for a static timing error is a quarter symbol"_test = [&channel, &post] {
        // What the preset has to beat is this window, not zero: a slot survives when its 201 channel bits do, and
        // NRZI turns any single channel error into a decoded one, so a bit error rate is a slot loss rate. The
        // sampler is the same interpolator the block runs, held off the optimum rather than tracking it.
        constexpr std::size_t kSymbols  = 400000UZ;
        constexpr double      kSlotBits = 201.0;

        Rng                      data(0x7015ULL);
        const std::vector<float> line      = nrziData(kSymbols + 64UZ, data);
        const std::vector<CD>    modulated = modulate(std::span<const float>(line));

        const Chain clean = runChain(std::span<const CD>(modulated), 0.0, nullptr, std::span<const float>(channel), std::span<const float>(post));
        Rng         noise(0xC0DEULL);
        const Chain loud = runChain(std::span<const CD>(modulated), 20.0, &noise, std::span<const float>(channel), std::span<const float>(post));

        const gr::sync::MmseInterpolatorBank bank(8, 128, 0.25, false);
        const auto                           lead = static_cast<double>(bank.delay());
        // the pulse is causal over three symbols, so a symbol peaks near s*sps + (span*sps - 1)/2; where exactly the
        // eye is widest is measured rather than assumed, because the whole table is offsets from that instant
        const double anchor = static_cast<double>(kSpan * kSps - 1UZ) / 2.0;

        const auto sampleAt = [&](const Chain& source, std::size_t symbol, double offset) {
            const double position = anchor + (static_cast<double>(symbol) + offset) * static_cast<double>(kSps) - lead;
            const double base     = std::floor(position);
            const auto   index    = static_cast<std::size_t>(base);
            const auto   row      = static_cast<std::size_t>((position - base) * static_cast<double>(bank.steps()) + 0.5);
            return bank.interpolate(source.lowpass.data() + index, std::min(row, bank.steps()));
        };

        double optimum = 0.0;
        double widest  = 0.0;
        for (int step = -30; step <= 30; ++step) {
            const double offset  = 0.01 * static_cast<double>(step);
            double       opening = 1e18;
            for (std::size_t s = 8UZ; s < 4008UZ; ++s) {
                opening = std::min(opening, std::abs(sampleAt(clean, s, offset)));
            }
            if (opening > widest) {
                widest  = opening;
                optimum = offset;
            }
        }
        std::println("[record] the eye is widest at {:+.2f} symbol from the pulse's own center, opening {:.4f}", optimum, widest);
        expect(lt(std::abs(optimum), 0.25)) << "the chain's optimum instant is where the causal pulse peaks, to within a quarter symbol";

        std::string rows;
        double      atQuarter = 0.0;
        double      atThird   = 0.0;
        for (const double offset : {0.00, 0.05, 0.15, 0.20, 0.25, 0.30, 0.35}) {
            std::size_t wrong   = 0UZ;
            std::size_t decided = 0UZ;
            for (std::size_t s = 8UZ; s + 8UZ < kSymbols; ++s) {
                const double value = sampleAt(loud, s, optimum + offset);
                wrong += (value > 0.0) == (line[s] > 0.f) ? 0UZ : 1UZ;
                ++decided;
            }
            const double rate = static_cast<double>(wrong) / static_cast<double>(decided);
            const double lost = 200.0 * (1.0 - std::pow(1.0 - rate, kSlotBits));
            std::format_to(std::back_inserter(rows), "{}{:.2f}: {} of {} bits, BER {:.2e}, {:.2f} slots of 200", rows.empty() ? "" : "; ", offset, wrong, decided, rate, lost);
            if (offset > 0.24 && offset < 0.26) {
                atQuarter = lost;
            }
            if (offset > 0.34) {
                atThird = lost;
            }
        }
        std::println("[record] static timing offset from the optimum at Es/N0 20 dB, {} symbols a point: {}", kSymbols, rows);
        expect(lt(atQuarter, 6.0)) << "a quarter symbol of static error costs a handful of slots in two hundred";
        expect(gt(atThird, 20.0 * std::max(atQuarter, 0.05))) << "and the window shuts fast past it, which is what makes a quarter symbol the tolerance";
    };
};

int main() { /* tests are automatically registered and run */ }
