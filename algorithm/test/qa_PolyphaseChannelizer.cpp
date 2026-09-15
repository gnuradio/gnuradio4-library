#ifdef __GNUC__
#pragma GCC diagnostic push
// GCC 16 raises -Werror=null-dereference inside boost/ut.hpp's own inlined string handling at -O2, the
// optimization level this file is pinned to for SimdFFT's inliner depth; the diagnostic is in a header the
// project vendors rather than writes, so it is suppressed around the include rather than for the whole tree.
#pragma GCC diagnostic ignored "-Wnull-dereference"
#endif
#include <boost/ut.hpp>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <format>
#include <numbers>
#include <print>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/FilterDesign.hpp>
#include <gnuradio-4.0/algorithm/filter/PolyphaseChannelizer.hpp>
#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>

namespace {

using gr::filter::PolyphaseChannelizer;
using gr::filter::PolyphaseSynthesizer;
using gr::filter::fir::design::polyphasePartition;

using Complex = std::complex<float>;

constexpr double kTwoPi = 2. * std::numbers::pi_v<double>;

/// The stopband every Kaiser prototype in this file is cut for.
constexpr double kAttenuationDb = 80.;

/// The channel counts the placement and isolation sweeps run over, and the offsets they place a tone at, in
/// channel widths. The offsets stay inside a quarter of a channel, where the wanted channel's own response is
/// still within a fraction of a dB of its peak.
constexpr std::array<std::size_t, 3UZ> kChannelCounts{4UZ, 16UZ, 64UZ};
constexpr std::array<double, 6UZ>      kOffsets{0.1, -0.1, 0.2, -0.2, 0.24, -0.24};

/// The Kaiser stopband ripples about its design figure, and the widest offset in the sweep lands on a lobe near
/// the transition, so the isolation is held to the design attenuation less this much.
constexpr double kIsolationMarginDb = 2.;

/// A tone whose frequency is recovered exactly rotates by a constant angle per channel sample. This bounds the
/// worst departure from that constant rate, which a mirrored channel map would push to a radian or more.
constexpr double kPhaseDriftTolerance = 1e-4;

/// The run lengths the streaming tests cut a stream into, in steps.
constexpr std::array<std::size_t, 7UZ> kStreamGroups{3UZ, 1UZ, 7UZ, 2UZ, 11UZ, 1UZ, 5UZ};

[[nodiscard]] std::complex<double> widen(Complex z) noexcept { return {static_cast<double>(z.real()), static_cast<double>(z.imag())}; }

[[nodiscard]] std::complex<double> turn(double angle) noexcept { return {std::cos(angle), std::sin(angle)}; }

[[nodiscard]] Complex phasor(double angle) noexcept { return {static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(angle))}; }

/// A lowpass whose -6 dB point sits half a channel width from DC. At twelve taps per channel the transition is
/// closed well inside the neighboring channel, so the sweep's widest offset is already in the stopband.
[[nodiscard]] std::vector<float> kaiserPrototype(std::size_t channels, std::size_t tapsPerChannel = 12UZ) { return gr::filter::fir::design::kaiserLowpass(tapsPerChannel * channels - 1UZ, 0.5 / static_cast<double>(channels), kAttenuationDb); }

/// A square-root-Nyquist prototype spanning `symbols` channel widths. The length asked for is one tap short of a
/// whole number of widths, which is what the designer's odd-length rounding pads back to exactly that number.
[[nodiscard]] std::vector<float> rootNyquistPrototype(std::size_t channels, std::size_t symbols, double alpha) { return gr::filter::fir::design::rootRaisedCosine(symbols * channels - 1UZ, static_cast<double>(channels), alpha, 1.0); }

[[nodiscard]] std::vector<double> asDouble(std::span<const float> taps) {
    std::vector<double> out(taps.size());
    std::ranges::transform(taps, out.begin(), [](float v) { return static_cast<double>(v); });
    return out;
}

/// The prototype's magnitude response at `frequency` cycles per input sample, in dB relative to DC.
[[nodiscard]] double responseDb(std::span<const float> prototype, double frequency) {
    std::complex<double> sum{};
    double               dc = 0.;
    for (std::size_t i = 0UZ; i < prototype.size(); ++i) {
        sum += static_cast<double>(prototype[i]) * turn(-kTwoPi * frequency * static_cast<double>(i));
        dc += static_cast<double>(prototype[i]);
    }
    return 20. * std::log10(std::abs(sum) / std::abs(dc));
}

/// The worst leakage the prototype alone accounts for when a tone sits `offset` channel widths above the center
/// of channel `wanted`. Every other channel sees the same tone through the same prototype shifted to its own
/// center, and the ratio is taken against what the wanted channel sees, which is a fraction of a dB off its peak.
[[nodiscard]] double predictedLeakageDb(std::span<const float> prototype, std::size_t channels, std::size_t wanted, double offset) {
    const double width  = 1. / static_cast<double>(channels);
    const double inBand = responseDb(prototype, offset * width);
    double       worst  = -1e9;
    for (std::size_t c = 0UZ; c < channels; ++c) {
        if (c == wanted) {
            continue;
        }
        double away = (static_cast<double>(wanted) - static_cast<double>(c) + offset) * width;
        away -= std::round(away); // the channel a full turn away is the same channel
        worst = std::max(worst, responseDb(prototype, away) - inBand);
    }
    return worst;
}

struct ToneResponse {
    std::size_t peakChannel{};  ///< the channel carrying the most power
    double      leakageDb{};    ///< the worst other channel, relative to the peak
    double      basebandRate{}; ///< cycles per channel sample, from the lag-one correlation
    double      phaseDrift{};   ///< the worst departure from a constant rotation rate, in radians
    double      coherence{};    ///< one when the channel output is a pure tone at the expected rate
};

/// Runs a unit tone `wanted + offset` channel widths above DC through the bank and measures where it lands. The
/// warm-up covers twice the steps the prototype needs to fill, so the measured window sees only steady state.
[[nodiscard]] ToneResponse runTone(std::size_t channels, std::size_t oversample, std::span<const float> prototype, std::size_t wanted, double offset) {
    PolyphaseChannelizer<float> bank;
    bank.configure(channels, prototype, oversample);

    const std::size_t stride = bank.stride();
    const std::size_t warmUp = 2UZ * (bank.prototypeLength() / stride) + 8UZ;
    const std::size_t steps  = warmUp + 256UZ;
    const double      cycles = (static_cast<double>(wanted) + offset) / static_cast<double>(channels);

    std::vector<double>               power(channels, 0.);
    std::vector<std::complex<double>> wantedOut;
    wantedOut.reserve(steps - warmUp);
    std::vector<Complex> in(stride);
    std::vector<Complex> out(channels);

    for (std::size_t n = 0UZ; n < steps; ++n) {
        for (std::size_t i = 0UZ; i < stride; ++i) {
            in[i] = phasor(kTwoPi * cycles * static_cast<double>(n * stride + i));
        }
        bank.step(in, out);
        if (n < warmUp) {
            continue;
        }
        for (std::size_t c = 0UZ; c < channels; ++c) {
            power[c] += std::norm(widen(out[c]));
        }
        wantedOut.push_back(widen(out[wanted]));
    }

    ToneResponse response{};
    response.peakChannel = static_cast<std::size_t>(std::ranges::distance(power.begin(), std::ranges::max_element(power)));

    double other = 0.;
    for (std::size_t c = 0UZ; c < channels; ++c) {
        if (c != response.peakChannel) {
            other = std::max(other, power[c]);
        }
    }
    response.leakageDb = 10. * std::log10(std::max(other, 1e-300) / power[response.peakChannel]);

    std::complex<double> lagOne{};
    for (std::size_t i = 0UZ; i + 1UZ < wantedOut.size(); ++i) {
        lagOne += wantedOut[i + 1UZ] * std::conj(wantedOut[i]);
    }
    response.basebandRate = std::arg(lagOne) / kTwoPi;

    // The channel runs `oversample` times faster than the channel spacing, so an offset of `offset` channel
    // widths is `offset / oversample` cycles per channel sample.
    const double         rate = offset / static_cast<double>(oversample);
    std::complex<double> aligned{};
    double               magnitude = 0.;
    for (std::size_t i = 0UZ; i < wantedOut.size(); ++i) {
        aligned += wantedOut[i] * turn(-kTwoPi * rate * static_cast<double>(i));
        magnitude += std::abs(wantedOut[i]);
    }
    response.coherence = std::abs(aligned) / magnitude;

    const std::complex<double> reference = aligned / std::abs(aligned);
    for (std::size_t i = 0UZ; i < wantedOut.size(); ++i) {
        const std::complex<double> residual = wantedOut[i] * turn(-kTwoPi * rate * static_cast<double>(i)) * std::conj(reference);
        response.phaseDrift                 = std::max(response.phaseDrift, std::abs(std::arg(residual)));
    }
    return response;
}

/// The unnormalized inverse transform across the branches followed by the step's leading phase, which is the
/// second half of the bank's stated arithmetic:
///
///     y_k[n] = exp(-j*2*pi*k*((n*S) mod M)/M) * sum_r exp(+j*2*pi*k*r/M) * v_r[n]
[[nodiscard]] std::vector<std::complex<double>> branchesToChannels(std::span<const std::complex<double>> branch, std::size_t stride, std::size_t step) {
    const std::size_t                 channels = branch.size();
    const std::size_t                 phase    = (step * stride) % channels;
    std::vector<std::complex<double>> out(channels, std::complex<double>{});
    for (std::size_t k = 0UZ; k < channels; ++k) {
        std::complex<double> sum{};
        for (std::size_t r = 0UZ; r < channels; ++r) {
            sum += branch[r] * turn(kTwoPi * static_cast<double>((k * r) % channels) / static_cast<double>(channels));
        }
        out[k] = sum * turn(-kTwoPi * static_cast<double>((k * phase) % channels) / static_cast<double>(channels));
    }
    return out;
}

/// The index of the newest input sample at step `n`, on a stream numbered from zero: step `n` consumes
/// `x[n*S] ... x[n*S + S - 1]`, so the sample the commutator calls branch zero is the last of those.
[[nodiscard]] std::ptrdiff_t newestAt(std::size_t step, std::size_t stride) noexcept { return static_cast<std::ptrdiff_t>((step + 1UZ) * stride) - 1; }

/// The branch filters written straight from `v_r[n] = sum_j h[j*M + r] * x[n*S - j*M - r]`, with `n*S` read as
/// `newestAt(n)` on a stream numbered from zero and samples before the start of the stream reading as zero.
[[nodiscard]] std::vector<std::complex<double>> branchSum(std::span<const Complex> input, std::span<const double> taps, std::size_t channels, std::size_t stride, std::size_t step) {
    std::vector<std::complex<double>> branch(channels, std::complex<double>{});
    const std::ptrdiff_t              newest = newestAt(step, stride);
    for (std::size_t j = 0UZ; j * channels < taps.size(); ++j) {
        for (std::size_t r = 0UZ; r < channels; ++r) {
            const std::ptrdiff_t at = newest - static_cast<std::ptrdiff_t>(j * channels + r);
            if (at >= 0 && at < static_cast<std::ptrdiff_t>(input.size())) {
                branch[r] += taps[j * channels + r] * widen(input[static_cast<std::size_t>(at)]);
            }
        }
    }
    return branch;
}

/// The same branch vector for a lone unit impulse at input index `at`, read off the branch assignment instead of
/// the sum: a sample of age `d` sits in branch `d mod M` weighted by `h[d]`, so one branch responds and the rest
/// stay silent.
[[nodiscard]] std::vector<std::complex<double>> branchImpulse(std::size_t at, std::span<const double> taps, std::size_t channels, std::size_t stride, std::size_t step) {
    std::vector<std::complex<double>> branch(channels, std::complex<double>{});
    const std::ptrdiff_t              age = newestAt(step, stride) - static_cast<std::ptrdiff_t>(at);
    if (age >= 0 && age < static_cast<std::ptrdiff_t>(taps.size())) {
        branch[static_cast<std::size_t>(age) % channels] = taps[static_cast<std::size_t>(age)];
    }
    return branch;
}

struct Reconstruction {
    double         residualDb{};  ///< the residual power against the fitted signal power
    std::ptrdiff_t lag{};         ///< the input samples of delay the fit chose
    std::ptrdiff_t expectedLag{}; ///< the delay the synthesizer reports, in input samples
    std::size_t    length{};      ///< the padded prototype length both banks run
};

/// Analysis into synthesis on a white input, scored by the residual left after fitting an integer lag and a
/// single complex gain over a window clear of both banks' fill.
[[nodiscard]] Reconstruction reconstruct(std::size_t channels, std::span<const float> prototype, std::size_t oversample) {
    constexpr std::size_t kSamples = 1UZ << 14;

    PolyphaseChannelizer<float> analysis;
    PolyphaseSynthesizer<float> synthesis;
    analysis.configure(channels, prototype, oversample);
    synthesis.configure(channels, prototype, oversample);

    const std::size_t stride = analysis.stride();
    const std::size_t length = analysis.prototypeLength();

    gr::rng::Xoshiro256pp rng(0xA11CEU);
    std::vector<Complex>  in(kSamples);
    for (Complex& sample : in) {
        sample = Complex(rng.uniformM11<float>(), rng.uniformM11<float>());
    }

    std::vector<Complex> channelSamples(channels);
    std::vector<Complex> out(kSamples);
    for (std::size_t n = 0UZ; (n + 1UZ) * stride <= kSamples; ++n) {
        analysis.step(std::span<const Complex>(in).subspan(n * stride, stride), channelSamples);
        synthesis.step(channelSamples, std::span<Complex>(out).subspan(n * stride, stride));
    }

    const std::size_t guard = 2UZ * length + 2UZ * channels;
    Reconstruction    best{1e9, 0, static_cast<std::ptrdiff_t>(synthesis.pipelineDelay() * stride), length};
    for (std::ptrdiff_t lag = 0; lag <= static_cast<std::ptrdiff_t>(length); ++lag) {
        std::complex<double> cross{};
        double               inputPower = 0.;
        double               outPower   = 0.;
        for (std::size_t i = guard; i + guard < kSamples; ++i) {
            const std::complex<double> a = widen(in[i]);
            const std::complex<double> b = widen(out[i + static_cast<std::size_t>(lag)]);
            cross += b * std::conj(a);
            inputPower += std::norm(a);
            outPower += std::norm(b);
        }
        const double fitted   = std::norm(cross) / inputPower;
        const double residual = outPower - fitted;
        const double db       = 10. * std::log10(std::max(residual, 1e-300) / fitted);
        if (db < best.residualDb) {
            best.residualDb = db;
            best.lag        = lag;
        }
    }
    return best;
}

} // namespace

const boost::ut::suite<"PolyphaseChannelizer placement"> polyphaseChannelizerPlacementTests = [] {
    using namespace boost::ut;

    "a tone inside a channel lands in that channel and nowhere else"_test = [] {
        for (const std::size_t channels : kChannelCounts) {
            const auto prototype      = kaiserPrototype(channels);
            double     worstOther     = -1e9;
            double     worstDrift     = 0.;
            double     worstRate      = 0.;
            double     leastCoherence = 1.;

            for (const std::size_t wanted : {0UZ, 1UZ, channels / 2UZ, channels - 1UZ}) {
                for (const double offset : kOffsets) {
                    const auto response = runTone(channels, 1UZ, prototype, wanted, offset);
                    const auto label    = std::format("M {} channel {} offset {:+.2f}", channels, wanted, offset);

                    expect(eq(response.peakChannel, wanted)) << std::format("{}: the peak is in channel {}", label, response.peakChannel);
                    expect(lt(response.leakageDb, -(kAttenuationDb - kIsolationMarginDb))) << std::format("{}: the worst other channel is {:.2f} dB down", label, response.leakageDb);
                    expect(lt(response.phaseDrift, kPhaseDriftTolerance)) << std::format("{}: the baseband tone drifts by {:g} rad against a constant {:+.4f} cycles per channel sample", label, response.phaseDrift, offset);
                    expect(lt(std::abs(response.basebandRate - offset), 1e-6)) << std::format("{}: the recovered rate is {:+.6f} cycles per channel sample", label, response.basebandRate);

                    worstOther     = std::max(worstOther, response.leakageDb);
                    worstDrift     = std::max(worstDrift, response.phaseDrift);
                    worstRate      = std::max(worstRate, std::abs(response.basebandRate - offset));
                    leastCoherence = std::min(leastCoherence, response.coherence);
                }
            }
            std::println("PolyphaseChannelizer placement, M = {:2}: worst other channel {:.2f} dB, worst rate error {:.3g} cycles, worst phase drift {:.3g} rad, least coherence {:.9f}", channels, worstOther, worstRate, worstDrift, leastCoherence);
        }
    };

    // The oversampled commutator advances half a channel per step, so the channel carries twice the bandwidth
    // and the same input frequency reads as half as many cycles per channel sample.
    "an oversampled bank places the same tone at half the normalized rate"_test = [] {
        for (const std::size_t channels : kChannelCounts) {
            const auto prototype  = kaiserPrototype(channels);
            double     worstOther = -1e9;
            double     worstRate  = 0.;
            double     worstDrift = 0.;

            for (const std::size_t wanted : {0UZ, 1UZ, channels / 2UZ, channels - 1UZ}) {
                for (const double offset : kOffsets) {
                    const auto   response = runTone(channels, 2UZ, prototype, wanted, offset);
                    const double expected = 0.5 * offset;
                    const auto   label    = std::format("M {} channel {} offset {:+.2f} oversampled", channels, wanted, offset);

                    expect(eq(response.peakChannel, wanted)) << std::format("{}: the peak is in channel {}", label, response.peakChannel);
                    expect(lt(response.leakageDb, -(kAttenuationDb - kIsolationMarginDb))) << std::format("{}: the worst other channel is {:.2f} dB down", label, response.leakageDb);
                    expect(lt(std::abs(response.basebandRate - expected), 1e-6)) << std::format("{}: the recovered rate is {:+.6f} against {:+.6f} cycles per channel sample", label, response.basebandRate, expected);
                    expect(lt(response.phaseDrift, kPhaseDriftTolerance)) << std::format("{}: the baseband tone drifts by {:g} rad", label, response.phaseDrift);

                    worstOther = std::max(worstOther, response.leakageDb);
                    worstRate  = std::max(worstRate, std::abs(response.basebandRate - expected));
                    worstDrift = std::max(worstDrift, response.phaseDrift);
                }
            }
            std::println("PolyphaseChannelizer placement oversampled, M = {:2}: worst other channel {:.2f} dB, worst rate error {:.3g} cycles, worst phase drift {:.3g} rad", channels, worstOther, worstRate, worstDrift);
        }
    };
};

const boost::ut::suite<"PolyphaseChannelizer isolation"> polyphaseChannelizerIsolationTests = [] {
    using namespace boost::ut;

    // The bank cannot separate channels better than the prototype separates frequencies, so the leakage is
    // scored against the prototype's own response at each channel's view of the tone rather than against a fixed
    // number. A prototype change moves both sides together and the margin still holds; a change to the bank's
    // arithmetic moves only the measurement.
    "leakage tracks the prototype's response at the offending offset"_test = [] {
        constexpr double kAgreementDb = 0.5;

        for (const std::size_t channels : kChannelCounts) {
            const auto prototype = kaiserPrototype(channels);
            for (const std::size_t oversample : {1UZ, 2UZ}) {
                double worstGap = 0.;
                for (const std::size_t wanted : {0UZ, 1UZ, channels / 2UZ, channels - 1UZ}) {
                    for (const double offset : kOffsets) {
                        const auto   response  = runTone(channels, oversample, prototype, wanted, offset);
                        const double predicted = predictedLeakageDb(prototype, channels, wanted, offset);
                        const double gap       = std::abs(response.leakageDb - predicted);

                        // Both sides are ratios against the channel the tone belongs to, so they only compare
                        // once that channel is the one carrying the power.
                        expect(eq(response.peakChannel, wanted)) << std::format("M {} channel {} offset {:+.2f} oversample {}: the peak is in channel {}", channels, wanted, offset, oversample, response.peakChannel);
                        expect(lt(gap, kAgreementDb)) << std::format("M {} channel {} offset {:+.2f} oversample {}: leakage {:.3f} dB against the prototype's {:.3f} dB", channels, wanted, offset, oversample, response.leakageDb, predicted);
                        worstGap = std::max(worstGap, gap);
                    }
                }
                std::println("PolyphaseChannelizer isolation, M = {:2} oversample {}: worst departure from the prototype's own response {:.4f} dB", channels, oversample, worstGap);
            }
        }
    };

    "the sixteen-channel default reaches its design attenuation"_test = [] {
        constexpr std::size_t kChannels  = 16UZ;
        const auto            prototype  = gr::filter::fir::design::kaiserLowpass(127, 0.5 / static_cast<double>(kChannels), kAttenuationDb);
        const double          neighborDb = responseDb(prototype, 0.9 / static_cast<double>(kChannels));

        for (const std::size_t wanted : {0UZ, 1UZ, 5UZ, 8UZ, 12UZ, 15UZ}) {
            const auto response = runTone(kChannels, 1UZ, prototype, wanted, 0.1);
            std::println("PolyphaseChannelizer 127-tap default, channel {:2}: peak channel {:2}, worst other {:.2f} dB, recovered rate {:+.6f} cycles per channel sample", wanted, response.peakChannel, response.leakageDb, response.basebandRate);

            expect(eq(response.peakChannel, wanted)) << std::format("channel {}: the peak is in channel {}", wanted, response.peakChannel);
            expect(lt(response.leakageDb, -(kAttenuationDb - kIsolationMarginDb))) << std::format("channel {}: the worst other channel is {:.2f} dB down", wanted, response.leakageDb);
            expect(lt(std::abs(response.leakageDb - neighborDb), 0.5)) << std::format("channel {}: leakage {:.2f} dB against the prototype's {:.2f} dB one tenth of a channel inside its neighbor", wanted, response.leakageDb, neighborDb);
        }
        std::println("PolyphaseChannelizer 127-tap default: the prototype is {:.2f} dB down where the neighboring channel sees the tone", neighborDb);
    };
};

const boost::ut::suite<"PolyphaseChannelizer commutator"> polyphaseChannelizerCommutatorTests = [] {
    using namespace boost::ut;

    // The bank's arithmetic recomputed in double straight from its stated equations. This is the anchor for the
    // index convention: it fixes which input sample reaches which branch, which way the transform across the
    // branches runs, and what the step's leading phase is.
    "an impulse produces the branch response the index convention predicts"_test = [] {
        constexpr std::size_t kSteps = 12UZ;

        for (const std::size_t channels : {4UZ, 8UZ}) {
            for (const std::size_t oversample : {1UZ, 2UZ}) {
                std::vector<float> prototype(3UZ * channels + 3UZ);
                for (std::size_t i = 0UZ; i < prototype.size(); ++i) {
                    prototype[i] = static_cast<float>(1UZ + i) * 0.125f; // every tap distinct, so no branch can stand in for another
                }

                PolyphaseChannelizer<float> bank;
                bank.configure(channels, prototype, oversample);
                const std::size_t stride = bank.stride();
                const auto        taps   = asDouble(bank.prototype());

                for (const std::size_t at : {0UZ, 1UZ, stride - 1UZ, stride, stride + 1UZ, 2UZ * stride + 3UZ}) {
                    bank.reset();
                    std::vector<Complex> input(kSteps * stride, Complex{});
                    input[at] = Complex(1.f, 0.f);

                    double               worst = 0.;
                    std::vector<Complex> out(channels);
                    for (std::size_t n = 0UZ; n < kSteps; ++n) {
                        bank.step(std::span<const Complex>(input).subspan(n * stride, stride), out);
                        const auto branch   = branchImpulse(at, taps, channels, stride, n);
                        const auto expected = branchesToChannels(branch, stride, n);
                        for (std::size_t k = 0UZ; k < channels; ++k) {
                            worst = std::max(worst, std::abs(widen(out[k]) - expected[k]));
                        }
                    }
                    expect(lt(worst, 1e-6)) << std::format("M {} oversample {}: an impulse at input index {} departs from the predicted branch response by {:g}", channels, oversample, at, worst);
                }
            }
        }
    };

    "an arbitrary stream reproduces the branch filters and the leading phase"_test = [] {
        constexpr std::size_t kSteps        = 24UZ;
        double                worstRelative = 0.;

        for (const std::size_t channels : {4UZ, 8UZ}) {
            for (const std::size_t oversample : {1UZ, 2UZ}) {
                std::vector<float> prototype(3UZ * channels + 3UZ);
                for (std::size_t i = 0UZ; i < prototype.size(); ++i) {
                    prototype[i] = static_cast<float>(1UZ + i) * 0.125f;
                }

                PolyphaseChannelizer<float> bank;
                bank.configure(channels, prototype, oversample);
                const std::size_t stride = bank.stride();
                const auto        taps   = asDouble(bank.prototype());

                gr::rng::Xoshiro256pp rng(0x5EEDU);
                std::vector<Complex>  input(kSteps * stride);
                for (Complex& sample : input) {
                    sample = Complex(rng.uniformM11<float>(), rng.uniformM11<float>());
                }

                double               worst = 0.;
                double               scale = 0.;
                std::vector<Complex> out(channels);
                for (std::size_t n = 0UZ; n < kSteps; ++n) {
                    bank.step(std::span<const Complex>(input).subspan(n * stride, stride), out);
                    const auto branch   = branchSum(input, taps, channels, stride, n);
                    const auto expected = branchesToChannels(branch, stride, n);
                    for (std::size_t k = 0UZ; k < channels; ++k) {
                        worst = std::max(worst, std::abs(widen(out[k]) - expected[k]));
                        scale = std::max(scale, std::abs(expected[k]));
                    }
                }
                const double relative = worst / scale;
                expect(lt(relative, 1e-6)) << std::format("M {} oversample {}: the bank departs from its stated equations by {:g} of full scale", channels, oversample, relative);
                worstRelative = std::max(worstRelative, relative);
            }
        }
        std::println("PolyphaseChannelizer commutator: worst departure from the stated equations {:.3g} of full scale", worstRelative);
    };
};

const boost::ut::suite<"PolyphaseChannelizer round trip"> polyphaseChannelizerRoundTripTests = [] {
    using namespace boost::ut;

    // How well the cascade reconstructs is a property of the prototype, not of the bank. A plain lowpass is not
    // power complementary, so the aliasing the analysis bank folds into each channel does not cancel when the
    // synthesizer adds the channels back together, and the residual sits above -20 dB however long the filter
    // is. A square-root-Nyquist prototype makes the squared responses sum flat across the channel grid, and an
    // oversampled commutator keeps the aliasing out of the band where that flatness is used, so the residual
    // then falls to the prototype's own truncation floor.
    "the residual follows the prototype, not the bank"_test = [] {
        constexpr std::size_t kChannels = 16UZ;

        struct Design {
            std::string_view   label;
            std::vector<float> prototype;
        };

        std::vector<Design> designs;
        designs.emplace_back("kaiser 127, 0.5/M cutoff, 80 dB", gr::filter::fir::design::kaiserLowpass(127, 0.5 / static_cast<double>(kChannels), kAttenuationDb));
        designs.emplace_back("root-Nyquist alpha 0.25, L =  8", rootNyquistPrototype(kChannels, 8UZ, 0.25));
        designs.emplace_back("root-Nyquist alpha 0.50, L =  8", rootNyquistPrototype(kChannels, 8UZ, 0.5));
        designs.emplace_back("root-Nyquist alpha 0.50, L = 16", rootNyquistPrototype(kChannels, 16UZ, 0.5));
        designs.emplace_back("root-Nyquist alpha 0.90, L = 32", rootNyquistPrototype(kChannels, 32UZ, 0.9));

        std::println("PolyphaseChannelizer round trip at M = {}: residual power after fitting the best lag and one complex gain", kChannels);
        std::println("  {:32}  {:>5}  {:>21}  {:>21}", "prototype", "taps", "critically sampled", "oversampled");

        std::vector<Reconstruction> critical;
        std::vector<Reconstruction> oversampled;
        for (const Design& design : designs) {
            critical.push_back(reconstruct(kChannels, design.prototype, 1UZ));
            oversampled.push_back(reconstruct(kChannels, design.prototype, 2UZ));
            std::println("  {:32}  {:5}  {:11.1f} dB at {:4}  {:11.1f} dB at {:4}", design.label, critical.back().length, critical.back().residualDb, critical.back().lag, oversampled.back().residualDb, oversampled.back().lag);
        }

        expect(gt(critical[0].residualDb, -20.)) << std::format("the plain lowpass must not reconstruct: {:.1f} dB critically sampled", critical[0].residualDb);
        expect(gt(oversampled[0].residualDb, -20.)) << std::format("the plain lowpass must not reconstruct: {:.1f} dB oversampled", oversampled[0].residualDb);
        expect(lt(oversampled[3].residualDb, -55.)) << std::format("the alpha 0.5, L = 16 root-Nyquist prototype must reconstruct: {:.1f} dB oversampled", oversampled[3].residualDb);

        for (std::size_t i = 0UZ; i < designs.size(); ++i) {
            expect(lt(oversampled[i].residualDb, critical[i].residualDb)) << std::format("{}: oversampling must not make reconstruction worse, {:.1f} dB against {:.1f} dB", designs[i].label, oversampled[i].residualDb, critical[i].residualDb);

            // Where the cascade reconstructs, the delay the fit settles on is the synthesizer's own pipeline
            // delay carried from steps into input samples.
            if (oversampled[i].residualDb < -30.) {
                expect(eq(oversampled[i].lag, oversampled[i].expectedLag)) << std::format("{}: the fit chose a delay of {} input samples against the reported {}", designs[i].label, oversampled[i].lag, oversampled[i].expectedLag);
            }
        }
    };
};

const boost::ut::suite<"PolyphaseChannelizer streaming"> polyphaseChannelizerStreamingTests = [] {
    using namespace boost::ut;

    constexpr std::size_t kChannels = 16UZ;
    constexpr std::size_t kSteps    = 200UZ;

    "the analysis bank is indifferent to how the stream is cut up"_test = [] {
        const auto prototype = kaiserPrototype(kChannels);

        for (const std::size_t oversample : {1UZ, 2UZ}) {
            PolyphaseChannelizer<float> bank;
            bank.configure(kChannels, prototype, oversample);
            const std::size_t stride = bank.stride();

            gr::rng::Xoshiro256pp rng(0xBEEFU);
            std::vector<Complex>  input(kSteps * stride);
            for (Complex& sample : input) {
                sample = Complex(rng.uniformM11<float>(), rng.uniformM11<float>());
            }

            std::vector<Complex> whole(kSteps * kChannels);
            for (std::size_t n = 0UZ; n < kSteps; ++n) {
                bank.step(std::span<const Complex>(input).subspan(n * stride, stride), std::span<Complex>(whole).subspan(n * kChannels, kChannels));
            }

            // A step takes the samples it consumes off the front of whatever it is handed, so the irregular runs
            // pass the whole remaining stream and let each step take its own share.
            bank.reset();
            std::vector<Complex> cut(kSteps * kChannels);
            std::size_t          done  = 0UZ;
            std::size_t          group = 0UZ;
            while (done < kSteps) {
                const std::size_t take = std::min(kStreamGroups[group % kStreamGroups.size()], kSteps - done);
                for (std::size_t n = done; n < done + take; ++n) {
                    bank.step(std::span<const Complex>(input).subspan(n * stride), std::span<Complex>(cut).subspan(n * kChannels));
                }
                done += take;
                ++group;
            }
            expect(std::ranges::equal(cut, whole)) << std::format("oversample {}: an irregularly cut stream must give bit-identical channels", oversample);

            std::vector<Complex> carried(kSteps * kChannels);
            for (std::size_t n = 0UZ; n < kSteps; ++n) {
                bank.step(std::span<const Complex>(input).subspan(n * stride, stride), std::span<Complex>(carried).subspan(n * kChannels, kChannels));
            }
            expect(!std::ranges::equal(carried, whole)) << std::format("oversample {}: a continued run must see the history the first run left", oversample);

            bank.reset();
            std::vector<Complex> restarted(kSteps * kChannels);
            for (std::size_t n = 0UZ; n < kSteps; ++n) {
                bank.step(std::span<const Complex>(input).subspan(n * stride, stride), std::span<Complex>(restarted).subspan(n * kChannels, kChannels));
            }
            expect(std::ranges::equal(restarted, whole)) << std::format("oversample {}: a reset bank must reproduce its first run", oversample);
        }
    };

    "the synthesis bank is indifferent to how the channels are cut up"_test = [] {
        const auto prototype = kaiserPrototype(kChannels);

        for (const std::size_t oversample : {1UZ, 2UZ}) {
            PolyphaseSynthesizer<float> bank;
            bank.configure(kChannels, prototype, oversample);
            const std::size_t stride = bank.stride();

            gr::rng::Xoshiro256pp rng(0xBEEFU);
            std::vector<Complex>  input(kSteps * kChannels);
            for (Complex& sample : input) {
                sample = Complex(rng.uniformM11<float>(), rng.uniformM11<float>());
            }

            std::vector<Complex> whole(kSteps * stride);
            for (std::size_t n = 0UZ; n < kSteps; ++n) {
                bank.step(std::span<const Complex>(input).subspan(n * kChannels, kChannels), std::span<Complex>(whole).subspan(n * stride, stride));
            }

            bank.reset();
            std::vector<Complex> cut(kSteps * stride);
            std::size_t          done  = 0UZ;
            std::size_t          group = 0UZ;
            while (done < kSteps) {
                const std::size_t take = std::min(kStreamGroups[group % kStreamGroups.size()], kSteps - done);
                for (std::size_t n = done; n < done + take; ++n) {
                    bank.step(std::span<const Complex>(input).subspan(n * kChannels), std::span<Complex>(cut).subspan(n * stride));
                }
                done += take;
                ++group;
            }
            expect(std::ranges::equal(cut, whole)) << std::format("oversample {}: irregularly cut channels must give a bit-identical stream", oversample);

            std::vector<Complex> carried(kSteps * stride);
            for (std::size_t n = 0UZ; n < kSteps; ++n) {
                bank.step(std::span<const Complex>(input).subspan(n * kChannels, kChannels), std::span<Complex>(carried).subspan(n * stride, stride));
            }
            expect(!std::ranges::equal(carried, whole)) << std::format("oversample {}: a continued run must see the overlap the first run left", oversample);

            bank.reset();
            std::vector<Complex> restarted(kSteps * stride);
            for (std::size_t n = 0UZ; n < kSteps; ++n) {
                bank.step(std::span<const Complex>(input).subspan(n * kChannels, kChannels), std::span<Complex>(restarted).subspan(n * stride, stride));
            }
            expect(std::ranges::equal(restarted, whole)) << std::format("oversample {}: a reset bank must reproduce its first run", oversample);
        }
    };
};

const boost::ut::suite<"polyphasePartition"> polyphasePartitionTests = [] {
    using namespace boost::ut;

    "twelve taps across three arms come out branch-major and reversed"_test = [] {
        const std::array<float, 12UZ> prototype{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f};

        // Branch b holds p[(Lb - 1 - m)*arms + b] with Lb = 4, so branch 0 reads p[9], p[6], p[3], p[0] and the
        // other two follow one tap along.
        const std::array<float, 12UZ> expected{10.f, 7.f, 4.f, 1.f, //
            11.f, 8.f, 5.f, 2.f,                                    //
            12.f, 9.f, 6.f, 3.f};

        const auto branches = polyphasePartition(prototype, 3UZ);
        expect(eq(branches.size(), 12UZ));
        expect(std::ranges::equal(branches, expected)) << "the twelve-tap partition must match the hand-computed branches";
    };

    "a short prototype is zero-padded to an even branch length"_test = [] {
        const std::array<float, 7UZ> prototype{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};

        // Seven taps across three arms need a branch length of three, rounded up to four, so each branch reads
        // one position past the end and the two highest arms read two.
        const std::array<float, 12UZ> expected{0.f, 7.f, 4.f, 1.f, //
            0.f, 0.f, 5.f, 2.f,                                    //
            0.f, 0.f, 6.f, 3.f};

        const auto branches = polyphasePartition(prototype, 3UZ);
        expect(eq(branches.size(), 12UZ)) << "the branch length must be even";
        expect(std::ranges::equal(branches, expected)) << "the padding must land at the head of each branch";
    };

    "the branch length is even for every arm count"_test = [] {
        std::vector<float> prototype(37UZ);
        for (std::size_t i = 0UZ; i < prototype.size(); ++i) {
            prototype[i] = static_cast<float>(i);
        }

        for (const std::size_t arms : {1UZ, 2UZ, 3UZ, 4UZ, 5UZ, 8UZ, 16UZ, 64UZ}) {
            const auto        branches     = polyphasePartition(prototype, arms);
            const std::size_t branchLength = branches.size() / arms;
            expect(eq(branches.size() % arms, 0UZ)) << std::format("{} arms: the partition must divide evenly", arms);
            expect(eq(branchLength % 2UZ, 0UZ)) << std::format("{} arms: branch length {} must be even", arms, branchLength);
            expect(ge(branchLength * arms, prototype.size())) << std::format("{} arms: the partition must hold every tap", arms);

            double placed = 0.;
            for (const float tap : branches) {
                placed += static_cast<double>(tap);
            }
            double total = 0.;
            for (const float tap : prototype) {
                total += static_cast<double>(tap);
            }
            expect(eq(placed, total)) << std::format("{} arms: every tap must appear exactly once", arms);
        }
    };

    "a partition needs at least one branch"_test = [] {
        const std::array<float, 4UZ> prototype{1.f, 2.f, 3.f, 4.f};
        expect(throws<std::invalid_argument>([&] { std::ignore = polyphasePartition(prototype, 0UZ); })) << "zero arms";
        expect(nothrow([&] { std::ignore = polyphasePartition(prototype, 1UZ); })) << "one arm";
    };
};

const boost::ut::suite<"PolyphaseChannelizer configuration"> polyphaseChannelizerConfigurationTests = [] {
    using namespace boost::ut;

    "configure refuses geometry the commutator cannot run"_test = [] {
        const auto prototype = kaiserPrototype(16UZ);

        const auto check = [&prototype](auto bank, std::string_view name) {
            expect(throws<std::invalid_argument>([&] { bank.configure(0UZ, prototype); })) << std::format("{}: zero channels", name);
            expect(throws<std::invalid_argument>([&] { bank.configure(1UZ, prototype); })) << std::format("{}: one channel", name);
            expect(throws<std::invalid_argument>([&] { bank.configure(decltype(bank)::kMaxChannels + 1UZ, prototype); })) << std::format("{}: one channel past the widest bank", name);
            expect(nothrow([&] { bank.configure(2UZ, prototype); })) << std::format("{}: the narrowest bank", name);
            expect(nothrow([&] { bank.configure(decltype(bank)::kMaxChannels, prototype); })) << std::format("{}: the widest bank", name);

            expect(throws<std::invalid_argument>([&] { bank.configure(16UZ, prototype, 0UZ); })) << std::format("{}: oversample zero", name);
            expect(throws<std::invalid_argument>([&] { bank.configure(16UZ, prototype, 3UZ); })) << std::format("{}: oversample three", name);
            expect(throws<std::invalid_argument>([&] { bank.configure(15UZ, prototype, 2UZ); })) << std::format("{}: an odd channel count oversampled", name);
            expect(nothrow([&] { bank.configure(15UZ, prototype, 1UZ); })) << std::format("{}: an odd channel count critically sampled", name);
            expect(nothrow([&] { bank.configure(16UZ, prototype, 2UZ); })) << std::format("{}: an even channel count oversampled", name);

            expect(throws<std::invalid_argument>([&] { bank.configure(16UZ, std::span<const float>{}); })) << std::format("{}: an empty prototype", name);
        };

        check(PolyphaseChannelizer<float>{}, "channelizer");
        check(PolyphaseSynthesizer<float>{}, "synthesizer");
    };

    "the configured geometry is reported back"_test = [] {
        const auto prototype = kaiserPrototype(16UZ);

        const auto check = [&prototype](auto bank, std::string_view name) {
            for (const std::size_t channels : {2UZ, 16UZ, 64UZ}) {
                for (const std::size_t oversample : {1UZ, 2UZ}) {
                    bank.configure(channels, prototype, oversample);
                    const auto label = std::format("{} M {} oversample {}", name, channels, oversample);
                    expect(eq(bank.channels(), channels)) << label;
                    expect(eq(bank.oversample(), oversample)) << label;
                    expect(eq(bank.stride(), channels / oversample)) << label;
                    expect(eq(bank.prototypeLength() % channels, 0UZ)) << std::format("{}: the prototype must pad to a whole number of branches", label);
                    expect(ge(bank.prototypeLength(), prototype.size())) << std::format("{}: the padding must hold every tap", label);
                    expect(eq(bank.prototype().size(), bank.prototypeLength())) << label;
                }
            }
        };

        check(PolyphaseChannelizer<float>{}, "channelizer");
        check(PolyphaseSynthesizer<float>{}, "synthesizer");
    };

    "a prototype shorter than one branch is padded up to the channel count"_test = [] {
        const std::array<float, 3UZ> prototype{1.f, 2.f, 3.f};

        PolyphaseChannelizer<float> bank;
        bank.configure(8UZ, prototype);
        expect(eq(bank.prototypeLength(), 8UZ)) << "the padded prototype must fill one branch per channel";
        expect(eq(bank.prototype()[0], 1.f));
        expect(eq(bank.prototype()[3], 0.f));
    };
};

int main() { /* not needed for UT */ }
