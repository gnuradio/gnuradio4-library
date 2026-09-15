#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <numbers>
#include <numeric>
#include <print>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/algorithm/digital/CpmPulse.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>

namespace {

using gr::digital::CpmPulse;
using gr::digital::CpmPulseShape;

constexpr double kTwoPi = 2. * std::numbers::pi_v<double>;

constexpr std::array<CpmPulseShape, 3UZ> kShapes{CpmPulseShape::Rect, CpmPulseShape::RaisedCosine, CpmPulseShape::Gaussian};

[[nodiscard]] constexpr std::string_view shapeName(CpmPulseShape shape) noexcept {
    switch (shape) {
    case CpmPulseShape::Rect: return "rect";
    case CpmPulseShape::RaisedCosine: return "raised_cosine";
    case CpmPulseShape::Gaussian: return "gaussian";
    }
    return "unknown";
}

/// The bt values worth sweeping for a shape: the parameter reaches the design only through the Gaussian family.
[[nodiscard]] std::span<const double> btValuesFor(CpmPulseShape shape) noexcept {
    static constexpr std::array<double, 3UZ> gaussian{0.2, 0.3, 0.5};
    static constexpr std::array<double, 1UZ> single{0.3};
    return shape == CpmPulseShape::Gaussian ? std::span<const double>(gaussian) : std::span<const double>(single);
}

/// Relative agreement with an absolute floor, so a reference tap of zero still compares. The widest Gaussians
/// have tail taps that underflow the single-precision designer they are sampled from.
[[nodiscard]] bool closeTo(double have, double want, double tolerance) noexcept { return std::abs(have - want) <= tolerance * std::max(std::abs(want), 1e-12); }

/// A deterministic sequence on the odd PAM grid `±1, ±3, …, ±(levels-1)`. The draws depend only on the seed, so
/// every symbol type sees the same sequence.
template<std::floating_point T = double>
[[nodiscard]] std::vector<T> pamSymbols(std::size_t n, int levels, std::uint64_t seed) {
    gr::rng::Xoshiro256pp rng(seed);
    std::vector<T>        symbols(n);
    for (std::size_t i = 0UZ; i < n; ++i) {
        const auto level = static_cast<int>(rng() % static_cast<std::uint64_t>(levels));
        symbols[i]       = static_cast<T>(2 * level - levels + 1);
    }
    return symbols;
}

/// Every increment a configured kernel produces for the given symbols, taken in one call.
template<std::floating_point T>
[[nodiscard]] std::vector<double> incrementsOf(CpmPulse<T>& kernel, std::span<const T> symbols) {
    std::vector<double> increments(symbols.size() * kernel.samplesPerSymbol());
    kernel.incrementsFor(symbols, std::span<double>(increments));
    return increments;
}

/// The power spectrum averaged over half-overlapping windowed segments, arranged from -fs/2 to +fs/2.
///
/// The window and the transform are both double. A 4-term Blackman-Harris floors at its own -92 dB sidelobe,
/// which sits inside the skirt an occupied-bandwidth figure has to resolve; the 7-term minimum-sidelobe window
/// used here is at -180 dB, far below the modulation's own spectrum.
[[nodiscard]] std::vector<double> averagedSpectrum(std::span<const std::complex<double>> signal, std::size_t nfft) {
    constexpr std::array<double, 7UZ> a{0.27105140069342, 0.43329793923448, 0.21812299954311, 0.06592544638803, 0.01081174209837, 0.00077658482522, 0.00001388721735};
    std::vector<double>               window(nfft);
    for (std::size_t i = 0UZ; i < nfft; ++i) {
        const double x = kTwoPi * static_cast<double>(i) / static_cast<double>(nfft);
        double       w = a[0];
        for (std::size_t term = 1UZ; term < a.size(); ++term) {
            w += (term % 2UZ == 1UZ ? -1. : 1.) * a[term] * std::cos(static_cast<double>(term) * x);
        }
        window[i] = w;
    }

    gr::algorithm::FFT<std::complex<double>> fft;
    std::vector<std::complex<double>>        segment(nfft);
    std::vector<std::complex<double>>        transform(nfft);
    std::vector<double>                      power(nfft, 0.);

    std::size_t segments = 0UZ;
    for (std::size_t at = 0UZ; at + nfft <= signal.size(); at += nfft / 2UZ) {
        for (std::size_t i = 0UZ; i < nfft; ++i) {
            segment[i] = signal[at + i] * window[i];
        }
        fft.compute(segment, transform);
        for (std::size_t i = 0UZ; i < nfft; ++i) {
            power[i] += std::norm(transform[i]);
        }
        ++segments;
    }

    std::vector<double> centered(nfft);
    for (std::size_t i = 0UZ; i < nfft; ++i) {
        centered[i] = power[(i + nfft / 2UZ) % nfft] / static_cast<double>(segments);
    }
    return centered;
}

/// The width in bins of the band that holds the given fraction of the total power, with the remainder split
/// equally between the two skirts. Both crossings are interpolated inside the bin they land in.
[[nodiscard]] double occupiedBandwidthBins(std::span<const double> spectrum, double fraction) {
    const double total = std::accumulate(spectrum.begin(), spectrum.end(), 0.);

    const auto crossing = [spectrum](double target) {
        double below = 0.;
        for (std::size_t i = 0UZ; i < spectrum.size(); ++i) {
            if (below + spectrum[i] >= target) {
                return static_cast<double>(i) + (target - below) / spectrum[i];
            }
            below += spectrum[i];
        }
        return static_cast<double>(spectrum.size());
    };

    const double excluded = 0.5 * (1. - fraction);
    return crossing((1. - excluded) * total) - crossing(excluded * total);
}

constexpr std::size_t kSpectrumSamplesPerSymbol = 8UZ;
constexpr std::size_t kSpectrumSymbols          = 16384UZ;
constexpr std::size_t kSpectrumTransform        = 8192UZ;
constexpr double      kOccupiedFraction         = 0.99;

/// The 99% occupied bandwidth of a two-level h = 0.5 signal, as a multiple of the bit rate.
///
/// The signal is built from the kernel alone: the increments are accumulated into a phase in double and turned
/// into `exp(j*phi)` here, so the figure describes the frequency pulse rather than anything downstream of it.
[[nodiscard]] double occupiedBandwidth(CpmPulseShape shape, std::size_t symbolSpan, double bt) {
    CpmPulse<double> kernel;
    kernel.configure(shape, symbolSpan, kSpectrumSamplesPerSymbol, 0.5, bt);

    const auto symbols    = pamSymbols(kSpectrumSymbols, 2, 0xC0FFEEU);
    const auto increments = incrementsOf(kernel, std::span<const double>(symbols));

    std::vector<std::complex<double>> signal(increments.size());
    double                            phase = 0.;
    for (std::size_t k = 0UZ; k < increments.size(); ++k) {
        signal[k] = std::complex<double>(std::cos(phase), std::sin(phase));
        phase += increments[k];
        if (phase > std::numbers::pi_v<double> || phase < -std::numbers::pi_v<double>) {
            phase = std::remainder(phase, kTwoPi);
        }
    }

    const auto   spectrum = averagedSpectrum(signal, kSpectrumTransform);
    const double bins     = occupiedBandwidthBins(spectrum, kOccupiedFraction);
    return bins * static_cast<double>(kSpectrumSamplesPerSymbol) / static_cast<double>(kSpectrumTransform);
}

} // namespace

const boost::ut::suite<"CpmPulse design"> cpmPulseDesignTests = [] {
    using namespace boost::ut;

    "the pulse carries area one half and is symmetric on its grid"_test = [] {
        double worstArea = 0.;
        double worstSkew = 0.;

        for (const auto shape : kShapes) {
            for (const std::size_t span : {1UZ, 2UZ, 3UZ, 4UZ, 5UZ, 8UZ}) {
                for (const std::size_t sps : {2UZ, 3UZ, 4UZ, 5UZ, 8UZ, 10UZ}) {
                    for (const double bt : btValuesFor(shape)) {
                        CpmPulse<double> kernel;
                        kernel.configure(shape, span, sps, 0.5, bt);

                        const auto pulse = kernel.pulse();
                        const auto label = std::format("{} span {} sps {} bt {}", shapeName(shape), span, sps, bt);
                        expect(eq(pulse.size(), span * sps)) << label;
                        expect(eq(kernel.symbolSpan(), span)) << label;
                        expect(eq(kernel.samplesPerSymbol(), sps)) << label;

                        bool   usable = true;
                        double area   = 0.;
                        for (const double tap : pulse) {
                            usable = usable && std::isfinite(tap) && tap >= 0.;
                            area += tap;
                        }
                        expect(usable) << std::format("{}: every tap must be finite and non-negative", label);
                        expect(lt(std::abs(area - 0.5), 1e-14)) << std::format("{}: area {:.17g}", label, area);
                        worstArea = std::max(worstArea, std::abs(area - 0.5));

                        for (std::size_t k = 0UZ; k < pulse.size(); ++k) {
                            const double skew = std::abs(pulse[k] - pulse[pulse.size() - 1UZ - k]);
                            expect(lt(skew, 1e-14)) << std::format("{}: taps {} and {} differ by {:g}", label, k, pulse.size() - 1UZ - k, skew);
                            worstSkew = std::max(worstSkew, skew);
                        }
                    }
                }
            }
        }
        std::println("CpmPulse design sweep: worst area error {:g}, worst asymmetry {:g}", worstArea, worstSkew);
    };

    "rect is uniform over its whole span"_test = [] {
        for (const std::size_t span : {1UZ, 2UZ, 3UZ, 4UZ, 5UZ, 8UZ}) {
            for (const std::size_t sps : {2UZ, 3UZ, 4UZ, 5UZ, 8UZ, 10UZ}) {
                CpmPulse<double> kernel;
                kernel.configure(CpmPulseShape::Rect, span, sps, 0.5, 0.3);

                const double want = 1. / (2. * static_cast<double>(span * sps));
                for (const double tap : kernel.pulse()) {
                    expect(lt(std::abs(tap - want), 1e-15)) << std::format("rect span {} sps {}: tap {:.17g} against {:.17g}", span, sps, tap, want);
                }
            }
        }
    };
};

const boost::ut::suite<"CpmPulse increments"> cpmPulseIncrementTests = [] {
    using namespace boost::ut;

    "a lone symbol traces 2*pi*h*g sample for sample"_test = [] {
        for (const auto shape : kShapes) {
            for (const double h : {0.25, 0.5, 0.7, 1.0}) {
                for (const std::size_t span : {1UZ, 2UZ, 3UZ, 5UZ, 8UZ}) {
                    for (const std::size_t sps : {2UZ, 4UZ, 8UZ}) {
                        CpmPulse<double> kernel;
                        kernel.configure(shape, span, sps, h, 0.3);

                        std::vector<double> symbols{1.};
                        symbols.resize(span, 0.);
                        const auto increments = incrementsOf(kernel, std::span<const double>(symbols));
                        const auto pulse      = kernel.pulse();

                        expect(eq(increments.size(), pulse.size()));
                        for (std::size_t k = 0UZ; k < increments.size(); ++k) {
                            const double want = kTwoPi * h * pulse[k];
                            expect(closeTo(increments[k], want, 1e-15)) << std::format("{} span {} sps {} h {}: increment {} is {:.17g} against {:.17g}", shapeName(shape), span, sps, h, k, increments[k], want);
                        }
                    }
                }
            }
        }
    };

    "one symbol turns the carrier by pi*h*a whatever the pulse spreads it over"_test = [] {
        double worst = 0.;

        for (const auto shape : kShapes) {
            for (const double h : {0.25, 0.5, 0.7, 1.0}) {
                for (const std::size_t span : {1UZ, 2UZ, 3UZ, 5UZ, 8UZ}) {
                    for (const std::size_t sps : {2UZ, 5UZ, 8UZ}) {
                        CpmPulse<double> kernel;
                        kernel.configure(shape, span, sps, h, 0.3);

                        for (const int levels : {2, 4, 8}) {
                            for (int level = 0; level < levels; ++level) {
                                const double amplitude = static_cast<double>(2 * level - levels + 1);
                                kernel.reset();

                                std::vector<double> symbols{amplitude};
                                symbols.resize(span, 0.);
                                const auto increments = incrementsOf(kernel, std::span<const double>(symbols));

                                const double turned = std::accumulate(increments.begin(), increments.end(), 0.);
                                const double want   = std::numbers::pi_v<double> * h * amplitude;
                                const auto   label  = std::format("{} span {} sps {} h {} amplitude {}", shapeName(shape), span, sps, h, amplitude);

                                expect(lt(std::abs(turned - want), 1e-12)) << std::format("{}: the span turns the carrier by {:.17g} against {:.17g}", label, turned, want);
                                expect(lt(std::abs(kernel.symbolPhase(amplitude) - want), 1e-12)) << std::format("{}: symbolPhase reports {:.17g} against {:.17g}", label, kernel.symbolPhase(amplitude), want);
                                worst = std::max(worst, std::abs(turned - want));
                            }
                        }
                    }
                }
            }
        }
        std::println("CpmPulse per-symbol phase: worst deviation from pi*h*a {:g} rad", worst);
    };
};

const boost::ut::suite<"CpmPulse streaming"> cpmPulseStreamingTests = [] {
    using namespace boost::ut;

    constexpr auto        kSymbolTypes = std::tuple<float, double>{};
    constexpr std::size_t kSpan        = 3UZ;
    constexpr std::size_t kSps         = 8UZ;

    "a split symbol stream produces the same increments, bit for bit"_test = []<std::floating_point T> {
        const auto symbols = pamSymbols<T>(200UZ, 4, 0x5EEDU);

        const std::array<std::vector<std::size_t>, 2UZ> patterns{std::vector<std::size_t>{1UZ, 5UZ, 2UZ, 11UZ, 3UZ, 17UZ, 1UZ}, std::vector<std::size_t>{7UZ, 1UZ, 23UZ, 4UZ, 1UZ, 2UZ, 13UZ}};

        for (const auto shape : kShapes) {
            CpmPulse<T> whole;
            whole.configure(shape, kSpan, kSps, 0.5, 0.3);
            const auto reference = incrementsOf(whole, std::span<const T>(symbols));

            for (const auto& pattern : patterns) {
                CpmPulse<T> split;
                split.configure(shape, kSpan, kSps, 0.5, 0.3);

                std::vector<double> increments(symbols.size() * kSps);
                std::size_t         consumed = 0UZ;
                std::size_t         step     = 0UZ;
                while (consumed < symbols.size()) {
                    const std::size_t n = std::min(pattern[step % pattern.size()], symbols.size() - consumed);
                    split.incrementsFor(std::span<const T>(symbols.data() + consumed, n), std::span<double>(increments.data() + consumed * kSps, n * kSps));
                    consumed += n;
                    ++step;
                }
                expect(std::ranges::equal(increments, reference)) << std::format("{}: increments taken in chunks of {} must be bit-identical to one call", shapeName(shape), pattern.front());
            }
        }
    } | kSymbolTypes;

    "reset restarts the stream from silence"_test = []<std::floating_point T> {
        const auto symbols = pamSymbols<T>(200UZ, 4, 0x5EEDU);

        for (const auto shape : kShapes) {
            CpmPulse<T> kernel;
            kernel.configure(shape, kSpan, kSps, 0.5, 0.3);

            const auto first    = incrementsOf(kernel, std::span<const T>(symbols));
            const auto carried  = incrementsOf(kernel, std::span<const T>(symbols));
            const auto restored = [&] {
                kernel.reset();
                return incrementsOf(kernel, std::span<const T>(symbols));
            }();

            expect(std::ranges::equal(restored, first)) << std::format("{}: a reset kernel must reproduce the first run", shapeName(shape));
            expect(!std::ranges::equal(carried, first)) << std::format("{}: a continued run must overlap the symbols already in the history", shapeName(shape));
        }
    } | kSymbolTypes;
};

const boost::ut::suite<"CpmPulse configuration"> cpmPulseConfigurationTests = [] {
    using namespace boost::ut;

    "configure refuses geometry outside the supported range"_test = [] {
        CpmPulse<double> kernel;

        expect(throws<std::invalid_argument>([&] { kernel.configure(CpmPulseShape::Rect, 1UZ, 0UZ, 0.5, 0.3); })) << "zero samples per symbol";
        expect(throws<std::invalid_argument>([&] { kernel.configure(CpmPulseShape::Rect, 1UZ, 1UZ, 0.5, 0.3); })) << "one sample per symbol";
        expect(nothrow([&] { kernel.configure(CpmPulseShape::Rect, 1UZ, 2UZ, 0.5, 0.3); })) << "two samples per symbol";

        expect(throws<std::invalid_argument>([&] { kernel.configure(CpmPulseShape::Rect, 0UZ, 4UZ, 0.5, 0.3); })) << "zero symbol span";
        expect(throws<std::invalid_argument>([&] { kernel.configure(CpmPulseShape::Rect, CpmPulse<double>::kMaxSymbolSpan + 1UZ, 4UZ, 0.5, 0.3); })) << "one symbol past the longest span";
        expect(nothrow([&] { kernel.configure(CpmPulseShape::Rect, CpmPulse<double>::kMaxSymbolSpan, 4UZ, 0.5, 0.3); })) << "the longest supported span";
    };

    "configure refuses modulation indices outside (0, 4]"_test = [] {
        CpmPulse<double> kernel;

        expect(throws<std::invalid_argument>([&] { kernel.configure(CpmPulseShape::Rect, 2UZ, 4UZ, 0., 0.3); })) << "modulation index zero";
        expect(throws<std::invalid_argument>([&] { kernel.configure(CpmPulseShape::Rect, 2UZ, 4UZ, -1., 0.3); })) << "negative modulation index";
        expect(throws<std::invalid_argument>([&] { kernel.configure(CpmPulseShape::Rect, 2UZ, 4UZ, 4.5, 0.3); })) << "modulation index past the ceiling";
        expect(nothrow([&] { kernel.configure(CpmPulseShape::Rect, 2UZ, 4UZ, CpmPulse<double>::kMaxModulationIndex, 0.3); })) << "the largest accepted modulation index";
        expect(eq(kernel.modulationIndex(), CpmPulse<double>::kMaxModulationIndex));
    };

    "bt is validated where the design reads it"_test = [] {
        CpmPulse<double> kernel;

        expect(throws<std::invalid_argument>([&] { kernel.configure(CpmPulseShape::Gaussian, 3UZ, 4UZ, 0.5, 0.); })) << "gaussian with a zero bandwidth-symbol-time product";
        expect(throws<std::invalid_argument>([&] { kernel.configure(CpmPulseShape::Gaussian, 3UZ, 4UZ, 0.5, -1.); })) << "gaussian with a negative bandwidth-symbol-time product";

        for (const double bt : {0., -1.}) {
            expect(nothrow([&] { kernel.configure(CpmPulseShape::Rect, 3UZ, 4UZ, 0.5, bt); })) << std::format("rect with bt {}", bt);
            expect(nothrow([&] { kernel.configure(CpmPulseShape::RaisedCosine, 3UZ, 4UZ, 0.5, bt); })) << std::format("raised cosine with bt {}", bt);
        }
    };

    "cpmPulseShapeFrom maps the shape names"_test = [] {
        using gr::digital::cpmPulseShapeFrom;

        expect(cpmPulseShapeFrom("rect") == CpmPulseShape::Rect);
        expect(cpmPulseShapeFrom("raised_cosine") == CpmPulseShape::RaisedCosine);
        expect(cpmPulseShapeFrom("gaussian") == CpmPulseShape::Gaussian);

        for (const std::string_view name : {"", "Rect", "RECT", "gauss", "raised cosine", "raisedcosine", "gmsk"}) {
            expect(throws<std::invalid_argument>([name] { std::ignore = cpmPulseShapeFrom(name); })) << std::format("'{}' is not a pulse name", name);
        }
    };
};

const boost::ut::suite<"CpmPulse spectrum"> cpmPulseSpectrumTests = [] {
    using namespace boost::ut;

    // The measured figures are 0.920 for GMSK at bt 0.3, 1.030 at bt 0.5 and 1.200 for MSK, and they move by
    // under 0.01 across seeds, record lengths, transform sizes and oversampling ratios. The last two land within
    // a few thousandths of the classic 1.04 and 1.20, which is what calibrates the estimator; the 0.86 usually
    // quoted alongside them belongs to bt 0.25 rather than bt 0.3. Truncating the Gaussian to three symbols
    // accounts for about 0.007 of the bt 0.3 figure, which converges to 0.915 by a span of four.
    "the 99% occupied bandwidth tightens with bt and against MSK"_test = [] {
        const double gaussian03   = occupiedBandwidth(CpmPulseShape::Gaussian, 3UZ, 0.3);
        const double gaussian05   = occupiedBandwidth(CpmPulseShape::Gaussian, 3UZ, 0.5);
        const double minimumShift = occupiedBandwidth(CpmPulseShape::Rect, 1UZ, 0.3);

        std::println("CpmPulse 99% occupied bandwidth as a multiple of the bit rate: GMSK bt 0.3 {:.3f}, GMSK bt 0.5 {:.3f}, MSK {:.3f}", gaussian03, gaussian05, minimumShift);

        expect(lt(gaussian03, gaussian05)) << std::format("a tighter Gaussian must occupy less: {:.3f} against {:.3f}", gaussian03, gaussian05);
        expect(lt(gaussian05, minimumShift)) << std::format("shaping must occupy less than the rectangle: {:.3f} against {:.3f}", gaussian05, minimumShift);

        expect(gt(gaussian03, 0.85) and lt(gaussian03, 1.00)) << std::format("GMSK bt 0.3 occupies {:.3f} of the bit rate", gaussian03);
        expect(gt(gaussian05, 0.95) and lt(gaussian05, 1.15)) << std::format("GMSK bt 0.5 occupies {:.3f} of the bit rate", gaussian05);
        expect(gt(minimumShift, 1.10) and lt(minimumShift, 1.35)) << std::format("MSK occupies {:.3f} of the bit rate", minimumShift);
    };
};

int main() { /* not needed for UT */ }
