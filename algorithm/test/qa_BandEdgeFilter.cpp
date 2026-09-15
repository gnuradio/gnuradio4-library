#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <print>
#include <random>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/FilterDesign.hpp>
#include <gnuradio-4.0/algorithm/sync/BandEdgeFilter.hpp>
#include <gnuradio-4.0/algorithm/sync/ControlLoop.hpp>

namespace {

using gr::sync::bandEdgeCenter;
using gr::sync::bandEdgeDetectorGain;
using gr::sync::bandEdgeDetectorGainPerRadian;
using gr::sync::BandEdgeDiscriminant;
using gr::sync::BandEdgeFilters;
using gr::sync::BandEdgeForm;
using gr::sync::bandEdgeFrequencyLimit;
using gr::sync::bandEdgePrototype;
using gr::sync::ControlLoop;
using gr::sync::designBandEdgeFilters;
using gr::sync::discriminant;
using gr::sync::LoopOrder;
using gr::sync::normalizedDiscriminant;

using CF = std::complex<float>;
using CD = std::complex<double>;

constexpr double kPi = std::numbers::pi;

/// @brief `sum_n taps[n] exp(-j 2 pi f n)`, in double, from the stored float taps.
[[nodiscard]] CD responseAt(std::span<const CF> taps, double f) {
    CD acc{};
    for (std::size_t n = 0UZ; n < taps.size(); ++n) {
        acc += CD(taps[n]) * std::polar(1.0, -2.0 * kPi * f * static_cast<double>(n));
    }
    return acc;
}

struct Geometry {
    double peak  = 0.0; /// the frequency at which |H| is largest
    double width = 0.0; /// the -3 dB width about it
};

[[nodiscard]] Geometry geometryOf(std::span<const CF> taps) {
    constexpr int       kGrid = 50000; /// 1e-5 cycles/sample, ten times finer than the 1e-4 the peak is asserted to
    std::vector<double> magnitude(static_cast<std::size_t>(kGrid) + 1UZ);
    double              best  = -1.0;
    double              where = 0.0;
    for (int i = 0; i <= kGrid; ++i) {
        const double f                         = 0.5 * static_cast<double>(i) / static_cast<double>(kGrid);
        magnitude[static_cast<std::size_t>(i)] = std::abs(responseAt(taps, f));
        if (magnitude[static_cast<std::size_t>(i)] > best) {
            best  = magnitude[static_cast<std::size_t>(i)];
            where = f;
        }
    }
    const double threshold = best / std::numbers::sqrt2;
    int          low       = 0;
    int          high      = kGrid;
    for (int i = 0; i <= kGrid; ++i) {
        if (magnitude[static_cast<std::size_t>(i)] >= threshold) {
            low = i;
            break;
        }
    }
    for (int i = kGrid; i >= 0; --i) {
        if (magnitude[static_cast<std::size_t>(i)] >= threshold) {
            high = i;
            break;
        }
    }
    return {where, 0.5 * static_cast<double>(high - low) / static_cast<double>(kGrid)};
}

/**
 * @brief Root-raised-cosine shaped QPSK at `sps`, scaled to a stated mean power, at zero offset.
 *
 * The offset is applied by the caller as it feeds the discriminant, so one shaped stream serves
 * every point of an S-curve; the signal generation is the expensive half and does not depend on
 * the offset.
 */
[[nodiscard]] std::vector<CF> shapedQpsk(std::size_t nSymbols, int sps, double rolloff, double power, std::uint32_t seed) {
    const std::vector<float> pulse = gr::filter::fir::design::rootRaisedCosine(static_cast<std::size_t>(11 * sps + 1), static_cast<double>(sps), rolloff, 1.0);
    std::mt19937             rng(seed);

    const double    half = std::numbers::sqrt2 / 2.0;
    std::vector<CD> symbols(nSymbols * static_cast<std::size_t>(sps), CD{});
    for (std::size_t k = 0UZ; k < nSymbols; ++k) {
        const std::uint64_t bits                   = rng();
        symbols[k * static_cast<std::size_t>(sps)] = {(bits & 1U) != 0U ? half : -half, (bits & 2U) != 0U ? half : -half};
    }

    std::vector<CD> shaped(symbols.size(), CD{});
    for (std::size_t n = 0UZ; n < symbols.size(); ++n) {
        CD acc{};
        for (std::size_t t = 0UZ; t < pulse.size(); ++t) {
            if (n >= t) {
                acc += static_cast<double>(pulse[t]) * symbols[n - t];
            }
        }
        shaped[n] = acc;
    }

    double            mean = 0.0;
    const std::size_t skip = pulse.size();
    for (std::size_t n = skip; n < shaped.size(); ++n) {
        mean += std::norm(shaped[n]);
    }
    mean /= static_cast<double>(shaped.size() - skip);

    const double    gain = std::sqrt(power / mean);
    std::vector<CF> out(shaped.size(), CF{});
    for (std::size_t n = 0UZ; n < shaped.size(); ++n) {
        out[n] = {static_cast<float>(gain * shaped[n].real()), static_cast<float>(gain * shaped[n].imag())};
    }
    return out;
}

/// @brief The number of leading samples dropped before an average is taken: filter rise plus loop settling.
constexpr std::size_t kSettle = 600UZ;

/// @brief `E[e]` for a shaped stream offset by @p offset cycles/sample.
template<BandEdgeForm Form = BandEdgeForm::RealTaps, typename Error>
[[nodiscard]] double meanError(std::span<const CF> signal, int nTaps, double sps, double rolloff, double offset, Error&& error) {
    BandEdgeDiscriminant<Form> detector(nTaps, sps, rolloff);
    double                     accumulated = 0.0;
    std::size_t                counted     = 0UZ;
    for (std::size_t n = 0UZ; n < signal.size(); ++n) {
        const CD   rotated = CD(signal[n]) * std::polar(1.0, 2.0 * kPi * offset * static_cast<double>(n));
        const auto powers  = detector.step(CF{static_cast<float>(rotated.real()), static_cast<float>(rotated.imag())});
        if (n >= kSettle) {
            accumulated += static_cast<double>(error(powers));
            ++counted;
        }
    }
    return accumulated / static_cast<double>(counted);
}

/// @brief The S-curve slope at zero, from a symmetric chord, the definition of `Kdet`.
template<typename Error>
[[nodiscard]] double detectorSlope(std::span<const CF> signal, int nTaps, double sps, double rolloff, Error&& error, double chord = 0.0025) {
    return (meanError(signal, nTaps, sps, rolloff, +chord, error) - meanError(signal, nTaps, sps, rolloff, -chord, error)) / (2.0 * chord);
}

/// One row of the geometry and gain tables.
struct DesignRow {
    int    nTaps   = 0;
    int    sps     = 0;
    double rolloff = 0.0;
    double value   = 0.0;
};

/// The derivation says the pair peaks at `(1+a)/(2 sps)` and is `a/sps` wide.
constexpr DesignRow kGeometry[] = {{45, 4, 0.35, 0.0}, {88, 4, 0.35, 0.0}, {22, 2, 0.35, 0.0}, {45, 4, 0.22, 0.0}, {89, 8, 0.50, 0.0}};

/// `sum(b^2)` before normalization, the divisor that makes `Kdet` a constant times `sps`.
constexpr DesignRow kEnergy[] = {{23, 2, 0.35, 5.7128}, {45, 4, 0.35, 11.4251}, {89, 8, 0.35, 22.8499}, {45, 4, 0.22, 18.1622}, {45, 4, 0.50, 7.9992}};

/// The measured discriminant gain per cycle/sample at unit input power.
constexpr DesignRow kDetectorGain[] = {{23, 2, 0.35, 1.9936}, {45, 4, 0.35, 3.9899}, {89, 8, 0.35, 7.9806}, {45, 4, 0.22, 3.9543}, {45, 4, 0.50, 3.9687}, {33, 4, 0.35, 4.0160}, {65, 4, 0.35, 3.9841}};

/// An off-center grid, for comparison: the prototype on `rint(N/sps)` and the spin on `(N-1)/2`.
[[nodiscard]] BandEdgeFilters uncenteredBandEdgeFilters(int nTaps, double sps, double rolloff) {
    BandEdgeFilters   out;
    const std::size_t len      = static_cast<std::size_t>(nTaps);
    const double      protoMid = 0.5 * static_cast<double>(nTaps) / sps * 2.0; // the grid advances by 2/sps per tap, centered on rint(N/sps)
    const double      spinMid  = 0.5 * static_cast<double>(nTaps - 1);
    const double      centerOf = std::rint(static_cast<double>(nTaps) / sps);

    std::vector<double> b(len);
    double              energy = 0.0;
    for (std::size_t i = 0UZ; i < len; ++i) {
        const double grid = 2.0 * static_cast<double>(i) / sps - centerOf;
        b[i]              = gr::filter::fir::design::sincPi(rolloff * grid - 0.5) + gr::filter::fir::design::sincPi(rolloff * grid + 0.5);
        energy += b[i] * b[i];
    }
    (void)protoMid;
    out.center          = bandEdgeCenter(sps, rolloff);
    out.prototypeEnergy = energy;
    out.prototype.resize(len);
    out.upper.resize(len);
    out.lower.resize(len);
    for (std::size_t i = 0UZ; i < len; ++i) {
        const double turn = 2.0 * kPi * out.center * (static_cast<double>(i) - spinMid);
        const CD     spin = (b[i] / energy) * std::polar(1.0, turn);
        out.prototype[i]  = static_cast<float>(b[i] / energy);
        out.upper[i]      = {static_cast<float>(spin.real()), static_cast<float>(spin.imag())};
        out.lower[i]      = std::conj(out.upper[i]);
    }
    return out;
}

} // namespace

const boost::ut::suite<"band-edge filter"> bandEdgeTests = [] {
    using namespace boost::ut;

    "the pair lands where the derivation says it does"_test = [] {
        for (const DesignRow& row : kGeometry) {
            const BandEdgeFilters filters  = designBandEdgeFilters(row.nTaps, row.sps, row.rolloff);
            const Geometry        measured = geometryOf(filters.upper);
            const double          center   = bandEdgeCenter(row.sps, row.rolloff);
            const double          width    = row.rolloff / static_cast<double>(row.sps);

            expect(approx(measured.peak, center, 1.0e-4)) << std::format("sps={} a={} N={}: peak {:.5f} against (1+a)/(2 sps) = {:.5f}", row.sps, row.rolloff, row.nTaps, measured.peak, center);
            expect(lt(std::abs(measured.width - width) / width, 0.06)) << std::format("sps={} a={} N={}: -3 dB width {:.5f} against a/sps = {:.5f}", row.sps, row.rolloff, row.nTaps, measured.width, width);
            expect(eq(filters.center, center)) << "the designed center is the derivation's, not a fit";
            std::println("band edge sps={} a={:.2f} N={}: peak {:.5f} (want {:.5f}), -3 dB width {:.5f} (want {:.5f})", row.sps, row.rolloff, row.nTaps, measured.peak, center, measured.width, width);
        }
    };

    "the grid is centered for every length and every samples-per-symbol"_test = [] {
        for (int nTaps : {22, 23, 32, 44, 45, 64, 88, 89}) {
            for (double sps : {2.0, 4.0, 8.0}) {
                const BandEdgeFilters filters = designBandEdgeFilters(nTaps, sps, 0.35);
                const std::size_t     len     = filters.prototype.size();

                float asymmetry = 0.0f;
                float conjugacy = 0.0f;
                for (std::size_t i = 0UZ; i < len; ++i) {
                    asymmetry = std::max(asymmetry, std::abs(filters.prototype[i] - filters.prototype[len - 1UZ - i]));
                    conjugacy = std::max(conjugacy, std::abs(filters.lower[i] - std::conj(filters.upper[i])));
                }
                expect(eq(asymmetry, 0.0f)) << std::format("N={} sps={}: max|b - reverse(b)| must be exactly zero", nTaps, sps);
                expect(eq(conjugacy, 0.0f)) << std::format("N={} sps={}: lower must be conj(upper) elementwise and exactly", nTaps, sps);
            }
        }
    };

    "the uncentered grid is asymmetric for six lengths in eight, which is what the centering fixes"_test = [] {
        // A counterexample rather than a property of this implementation: it records what an
        // off-center construction does, so that "centered for any N" is a claim with evidence.
        int broken = 0;
        for (int nTaps : {22, 23, 32, 44, 45, 64, 88, 89}) {
            const BandEdgeFilters legacy = uncenteredBandEdgeFilters(nTaps, 4.0, 0.35);
            float                 worst  = 0.0f;
            for (std::size_t i = 0UZ; i < legacy.prototype.size(); ++i) {
                worst = std::max(worst, std::abs(legacy.prototype[i] - legacy.prototype[legacy.prototype.size() - 1UZ - i]));
            }
            if (worst > 1.0e-6f) {
                ++broken;
            }
        }
        expect(ge(broken, 5)) << "the grid whose center is rint(N/sps) is asymmetric for most tap counts at sps = 4";
    };

    "the prototype is divided by its energy, and the energy is the table"_test = [] {
        for (const DesignRow& row : kEnergy) {
            double                    energy = 0.0;
            const std::vector<double> b      = bandEdgePrototype(row.nTaps, row.sps, row.rolloff, &energy);
            expect(approx(energy, row.value, 5.0e-4)) << std::format("sps={} a={} N={}: sum(b^2) = {:.4f}", row.sps, row.rolloff, row.nTaps, energy);

            double normalized = 0.0;
            for (double tap : b) {
                normalized += tap * tap;
            }
            expect(approx(normalized, 1.0 / energy, 1.0e-12)) << "dividing by the energy leaves sum(b^2) = 1/energy — not unit energy, deliberately";
        }
        double     energy    = 0.0;
        const auto prototype = bandEdgePrototype(45, 4.0, 0.35, &energy);
        expect(approx(prototype[22] * energy, 4.0 / kPi, 1.0e-12)) << "the center tap is sinc(-1/2) + sinc(+1/2) = 4/pi for every alpha";
    };

    "the real-tap identity is exact, and it is what halves the multiply count"_test = [] {
        constexpr int nTaps   = 45;
        constexpr int sps     = 4;
        const double  rolloff = 0.35;

        double                    energy = 0.0;
        const std::vector<double> b      = bandEdgePrototype(nTaps, sps, rolloff, &energy);
        const double              center = bandEdgeCenter(sps, rolloff);
        const double              middle = 0.5 * static_cast<double>(nTaps - 1);

        std::vector<CD> upper(b.size());
        for (std::size_t i = 0UZ; i < b.size(); ++i) {
            upper[i] = b[i] * std::polar(1.0, 2.0 * kPi * center * (static_cast<double>(i) - middle));
        }

        const std::vector<CF> signal = shapedQpsk(600UZ, sps, rolloff, 1.0, 3U);
        double                worst  = 0.0;
        for (std::size_t m = b.size(); m < signal.size(); ++m) {
            CD complexTaps{};
            CD realTaps{};
            for (std::size_t n = 0UZ; n < b.size(); ++n) {
                const CD sample = CD(signal[m - n]);
                complexTaps += upper[n] * sample;
                realTaps += b[n] * (sample * std::polar(1.0, -2.0 * kPi * center * static_cast<double>(m - n)));
            }
            worst = std::max(worst, std::abs(std::abs(complexTaps) - std::abs(realTaps)));
        }
        expect(lt(worst, 1.0e-12)) << std::format("|sum upper[n] y[m-n]| against |sum b[n] y'[m-n]|: worst {:.3g}", worst);
        std::println("real-tap identity: max |difference of magnitudes| = {:.3g} over {} outputs", worst, signal.size() - b.size());
    };

    "the shipped real-tap form tracks the complex-tap reference over a hundred thousand samples"_test = [] {
        const std::vector<CF>                           signal = shapedQpsk(25000UZ, 4, 0.35, 1.0, 5U);
        BandEdgeDiscriminant<BandEdgeForm::RealTaps>    shipped(45, 4.0, 0.35);
        BandEdgeDiscriminant<BandEdgeForm::ComplexTaps> reference(45, 4.0, 0.35);

        double      upperEnergy = 0.0;
        double      lowerEnergy = 0.0;
        double      worstUpper  = 0.0;
        double      worstLower  = 0.0;
        std::size_t counted     = 0UZ;
        for (std::size_t n = 0UZ; n < signal.size(); ++n) {
            const CD rotated = CD(signal[n]) * std::polar(1.0, 2.0 * kPi * 0.01 * static_cast<double>(n));
            const CF sample{static_cast<float>(rotated.real()), static_cast<float>(rotated.imag())};

            const auto a = shipped.step(sample);
            const auto b = reference.step(sample);
            if (n > 100UZ) {
                worstUpper = std::max(worstUpper, std::abs(std::sqrt(static_cast<double>(a.upper)) - std::sqrt(static_cast<double>(b.upper))));
                worstLower = std::max(worstLower, std::abs(std::sqrt(static_cast<double>(a.lower)) - std::sqrt(static_cast<double>(b.lower))));
                upperEnergy += static_cast<double>(b.upper);
                lowerEnergy += static_cast<double>(b.lower);
                ++counted;
            }
        }
        const double upperRms = std::sqrt(upperEnergy / static_cast<double>(counted));
        const double lowerRms = std::sqrt(lowerEnergy / static_cast<double>(counted));

        // Against the RMS of the sequence rather than the instantaneous value: |yl| passes close to
        // zero many times in 1e5 samples, where a relative bound is beyond float arithmetic.
        expect(lt(worstUpper / upperRms, 1.0e-5)) << std::format("|yu|: worst {:.3g} against rms {:.4f}", worstUpper, upperRms);
        expect(lt(worstLower / lowerRms, 1.0e-5)) << std::format("|yl|: worst {:.3g} against rms {:.4f}", worstLower, lowerRms);
        expect(gt(counted, 99000UZ)) << "the recursive rotator has to run the whole way";
    };

    "the detector gain is sps times the input power"_test = [] {
        for (const DesignRow& row : kDetectorGain) {
            const std::vector<CF> signal = shapedQpsk(8000UZ, row.sps, row.rolloff, 1.0, 999U);
            const double          slope  = detectorSlope(signal, row.nTaps, row.sps, row.rolloff, discriminant);

            expect(lt(std::abs(slope - row.value) / row.value, 0.03)) << std::format("sps={} a={} N={}: Kdet {:.4f} against the measured {:.4f}", row.sps, row.rolloff, row.nTaps, slope, row.value);
            const double predicted = bandEdgeDetectorGain(row.sps, 1.0);
            expect(lt(std::abs(slope - predicted) / predicted, 0.03)) << std::format("sps={} a={} N={}: Kdet/sps = {:.4f}, against the default sps*P", row.sps, row.rolloff, row.nTaps, slope / static_cast<double>(row.sps));
        }
    };

    "the detector gain is proportional to the input power, which is why an AGC is a precondition"_test = [] {
        double gains[3] = {};
        int    at       = 0;
        for (double power : {0.25, 1.0, 4.0}) {
            const std::vector<CF> signal = shapedQpsk(6000UZ, 4, 0.35, power, 7U);
            gains[at++]                  = detectorSlope(signal, 45, 4.0, 0.35, discriminant);
        }
        expect(approx(gains[1] / gains[0], 4.0, 0.04)) << std::format("Kdet at P = 0.25 and 1.0: {:.4f}, {:.4f}", gains[0], gains[1]);
        expect(approx(gains[2] / gains[1], 4.0, 0.04)) << std::format("Kdet at P = 1.0 and 4.0: {:.4f}, {:.4f}", gains[1], gains[2]);
        expect(approx(gains[1], bandEdgeDetectorGain(4.0, 1.0), 0.12)) << "sps * P, at unit power";
    };

    "the normalized discriminant has the same gain at every input power"_test = [] {
        double gains[3] = {};
        int    at       = 0;
        for (double power : {0.25, 1.0, 4.0}) {
            const std::vector<CF> signal = shapedQpsk(6000UZ, 4, 0.35, power, 7U);
            gains[at++]                  = detectorSlope(signal, 45, 4.0, 0.35, [](auto powers) { return normalizedDiscriminant(powers); });
        }
        // The signal differs only by a scalar, so the ratio is exactly one until float rounding says
        // otherwise; five figures is what is asserted.
        expect(lt(std::abs(gains[0] / gains[1] - 1.0), 1.0e-5)) << std::format("normalized Kdet at P = 0.25 and 1.0: {:.5f}, {:.5f}", gains[0], gains[1]);
        expect(lt(std::abs(gains[2] / gains[1] - 1.0), 1.0e-5)) << std::format("normalized Kdet at P = 4.0 and 1.0: {:.5f}, {:.5f}", gains[2], gains[1]);
        expect(gt(gains[1], 40.0)) << "and it is an order of magnitude steeper than the unnormalized one, in its own units";
    };

    "the S-curve is odd, crosses zero once, and flattens at the pull-in range"_test = [] {
        const std::vector<CF> signal     = shapedQpsk(10000UZ, 4, 0.35, 1.0, 12345U);
        constexpr double      kOffsets[] = {-0.175, -0.15, -0.10, -0.05, -0.025, 0.0, 0.025, 0.05, 0.10, 0.15, 0.175};

        double curve[std::size(kOffsets)] = {};
        for (std::size_t i = 0UZ; i < std::size(kOffsets); ++i) {
            curve[i] = meanError(signal, 45, 4.0, 0.35, kOffsets[i], discriminant);
        }

        expect(lt(std::abs(curve[5]), 1.0e-3)) << std::format("S(0) = {:.5f}: a spectrally symmetric signal at zero offset gives zero error", curve[5]);
        for (std::size_t i = 0UZ; i < 5UZ; ++i) {
            expect(lt(std::abs(curve[i] + curve[std::size(kOffsets) - 1UZ - i]), 0.02)) << std::format("odd at +/-{:.4f}: {:+.5f} against {:+.5f}", kOffsets[i], curve[i], curve[std::size(kOffsets) - 1UZ - i]);
        }
        for (std::size_t i = 5UZ; i + 1UZ < 9UZ; ++i) {
            expect(gt(curve[i + 1UZ], curve[i])) << "monotone out to the peak of the curve";
        }
        const double pullIn = bandEdgeCenter(4.0, 0.35);
        expect(approx(pullIn, 0.16875, 1.0e-12)) << "the pull-in range and the band edge are the same number";
        expect(lt(std::abs(curve[10] - curve[9]) / std::abs(curve[9]), 0.05)) << "flat beyond (1+a)/(2 sps), which is the reason for the clamp";
        expect(gt(curve[6], 0.0)) << "positive above the carrier: Kdet is positive and the derotation carries the minus sign";
        expect(lt(curve[4], 0.0)) << "and negative below it";

        for (std::size_t i = 0UZ; i < std::size(kOffsets); ++i) {
            std::println("band edge S({:+.4f} cycles/sample) = {:+.5f}", kOffsets[i], curve[i]);
        }
    };

    "an off-center grid still gives S(0) = 0, so that is not what the centering fixes"_test = [] {
        // The two tap vectors are conjugates whatever the prototype's symmetry, so
        // |H_upper(f)| = |H_lower(-f)| identically and a symmetric input at zero offset gives zero
        // error on either grid. What the asymmetric grid costs is phase linearity.
        const std::vector<CF> signal = shapedQpsk(8000UZ, 4, 0.35, 1.0, 21U);
        const BandEdgeFilters legacy = uncenteredBandEdgeFilters(44, 4.0, 0.35);

        double          accumulated = 0.0;
        std::size_t     counted     = 0UZ;
        std::vector<CF> history(legacy.prototype.size(), CF{});
        for (std::size_t n = 0UZ; n < signal.size(); ++n) {
            for (std::size_t k = history.size() - 1UZ; k > 0UZ; --k) {
                history[k] = history[k - 1UZ];
            }
            history[0] = signal[n];

            CF upper{};
            CF lower{};
            for (std::size_t k = 0UZ; k < history.size(); ++k) {
                upper += legacy.upper[k] * history[k];
                lower += legacy.lower[k] * history[k];
            }
            if (n >= kSettle) {
                accumulated += static_cast<double>(std::norm(upper) - std::norm(lower));
                ++counted;
            }
        }
        expect(lt(std::abs(accumulated / static_cast<double>(counted)), 1.0e-3)) << "zero error at zero offset on the uncentered grid too";
    };

    "the first-order gain is the exact inversion, and the bandwidth setter leaves no proportional arm"_test = [] {
        struct GainRow {
            double sps            = 0.0;
            double noiseBandwidth = 0.0;
            double beta           = 0.0;
        };
        constexpr GainRow kRows[] = {{4.0, 0.01, 0.061600}, {2.0, 0.0628, 0.701109}, {8.0, 0.005, 0.015552}};

        for (const GainRow& row : kRows) {
            ControlLoop<double> loop(row.noiseBandwidth, 1.0, bandEdgeDetectorGainPerRadian(row.sps, 1.0), -bandEdgeFrequencyLimit(row.sps, 0.35), bandEdgeFrequencyLimit(row.sps, 0.35), LoopOrder::First);
            expect(approx(loop.beta(), row.beta, 1.0e-6)) << std::format("sps={} Bn*T={}: beta = {:.6f}", row.sps, row.noiseBandwidth, loop.beta());
            expect(eq(loop.alpha(), 0.0)) << "first order means no proportional arm, before and after any setter";

            loop.setNoiseBandwidth(2.0 * row.noiseBandwidth);
            expect(eq(loop.alpha(), 0.0)) << "changing the bandwidth recomputes beta only";
            expect(gt(loop.beta(), row.beta)) << "and it does recompute it";
        }

        // The small-bandwidth approximation: 2% high at 0.01 and 25% at 0.1.
        for (double bandwidth : {0.01, 0.1}) {
            const double exact       = 4.0 * bandwidth / (1.0 + 2.0 * bandwidth);
            const double approximate = 4.0 * bandwidth;
            expect(gt(approximate / exact - 1.0, 0.019)) << std::format("Bn*T={}: 4*Bn*T is {:.1f}% high", bandwidth, 100.0 * (approximate / exact - 1.0));
        }
    };

    "the clamp is the discriminant's own pull-in range"_test = [] {
        for (double sps : {2.0, 4.0, 8.0}) {
            for (double rolloff : {0.22, 0.35, 0.5}) {
                const double limit = bandEdgeFrequencyLimit(sps, rolloff);
                expect(approx(limit / (2.0 * kPi), bandEdgeCenter(sps, rolloff), 1.0e-15)) << "the clamp in cycles/sample is the band edge, which is where the S-curve goes flat";
                expect(lt(limit / (2.0 * kPi), 2.0 / sps)) << std::format("sps={} a={}: {:.5f} against the 2/sps it replaces", sps, rolloff, limit / (2.0 * kPi));
                expect(approx((2.0 / sps) / (limit / (2.0 * kPi)), 4.0 / (1.0 + rolloff), 1.0e-12)) << "narrower by 4/(1+a) — about three times at every rolloff in range";
            }
        }
    };

    "the loop pulls in from inside the range and saturates outside it"_test = [] {
        constexpr int    nTaps   = 45;
        constexpr int    sps     = 4;
        constexpr double rolloff = 0.35;
        const double     limit   = bandEdgeFrequencyLimit(sps, rolloff);

        const auto settle = [&](double offset, double noiseBandwidth, std::size_t nSymbols) {
            const std::vector<CF>                           signal = shapedQpsk(nSymbols, sps, rolloff, 1.0, 4242U);
            BandEdgeDiscriminant<BandEdgeForm::RealTaps>    detector(nTaps, static_cast<double>(sps), rolloff);
            ControlLoop<double, gr::sync::PhaseWrap::TwoPi> loop(noiseBandwidth, 1.0, bandEdgeDetectorGainPerRadian(sps, 1.0), -limit, limit, LoopOrder::First);

            double      tail    = 0.0;
            std::size_t counted = 0UZ;
            for (std::size_t n = 0UZ; n < signal.size(); ++n) {
                const CD offsetted = CD(signal[n]) * std::polar(1.0, 2.0 * kPi * offset * static_cast<double>(n));
                // The header's sign: the error is |yu|^2 - |yl|^2, so the derotation runs against it.
                const CD   derotated = offsetted * std::polar(1.0, -loop.phase());
                const auto powers    = detector.step(CF{static_cast<float>(derotated.real()), static_cast<float>(derotated.imag())});
                (void)loop.step(static_cast<double>(discriminant(powers)));
                if (n + signal.size() / 4UZ >= signal.size()) {
                    tail += loop.frequency();
                    ++counted;
                }
            }
            return std::pair{tail / static_cast<double>(counted), loop.saturated()};
        };

        struct ClosedLoopRow {
            double offset    = 0.0;
            double tolerance = 0.0;
        };
        constexpr ClosedLoopRow kRows[] = {{0.005, 0.01}, {0.020, 0.01}, {0.050, 0.01}};
        for (const ClosedLoopRow& row : kRows) {
            const auto   result = settle(row.offset, 0.01, 12000UZ);
            const double ideal  = 2.0 * kPi * row.offset;
            expect(lt(std::abs(result.first - ideal) / ideal, row.tolerance)) << std::format("offset {:+.4f} cycles/sample: settled {:+.5f} rad/sample against {:+.5f}", row.offset, result.first, ideal);
            expect(!result.second) << "and it is tracking, not parked against a bound";
        }

        const double pullIn  = bandEdgeCenter(sps, rolloff);
        const auto   inside  = settle(0.9 * pullIn, 0.02, 20000UZ);
        const auto   outside = settle(1.5 * pullIn, 0.02, 20000UZ);
        expect(lt(std::abs(inside.first - 2.0 * kPi * 0.9 * pullIn) / (2.0 * kPi * 0.9 * pullIn), 0.05)) << std::format("0.9 of the pull-in range converges: {:+.5f}", inside.first);
        expect(outside.second) << std::format("1.5 of it saturates against the clamp: {:+.5f}", outside.first);
    };

    "a redesign with the same parameters is bit-identical"_test = [] {
        // Two designs with the same parameters must agree bit for bit, so that recomputing a gain
        // with the value already in force cannot change the loop.
        const BandEdgeFilters first  = designBandEdgeFilters(45, 4.0, 0.35);
        const BandEdgeFilters second = designBandEdgeFilters(45, 4.0, 0.35);
        expect(that % (first.prototype == second.prototype)) << "taps";
        expect(that % (first.upper == second.upper)) << "upper";
        expect(that % (first.lower == second.lower)) << "lower";
        expect(eq(first.center, second.center));
        expect(eq(first.prototypeEnergy, second.prototypeEnergy));

        expect(eq(bandEdgeDetectorGain(4.0, 1.0), 4.0));
        expect(approx(bandEdgeDetectorGainPerRadian(4.0, 1.0), 4.0 / (2.0 * kPi), 1.0e-15));
    };

    "the discriminant has no chunk-dependent state"_test = [] {
        const std::vector<CF> signal = shapedQpsk(2000UZ, 4, 0.35, 1.0, 77U);

        std::vector<float>                           whole;
        BandEdgeDiscriminant<BandEdgeForm::RealTaps> one(45, 4.0, 0.35);
        for (const CF& sample : signal) {
            whole.push_back(discriminant(one.step(sample)));
        }

        for (std::size_t chunk : {1UZ, 3UZ, 17UZ, 4096UZ}) {
            std::vector<float>                           pieces;
            BandEdgeDiscriminant<BandEdgeForm::RealTaps> many(45, 4.0, 0.35);
            for (std::size_t at = 0UZ; at < signal.size(); at += chunk) {
                const std::size_t take = std::min(chunk, signal.size() - at);
                for (std::size_t i = 0UZ; i < take; ++i) {
                    pieces.push_back(discriminant(many.step(signal[at + i])));
                }
            }
            expect(that % (pieces == whole)) << std::format("chunk {}: bit for bit", chunk);
        }
    };

    "degenerate parameters throw rather than designing something meaningless"_test = [] {
        expect(throws<std::invalid_argument>([] { (void)bandEdgePrototype(2, 4.0, 0.35); }));
        expect(throws<std::invalid_argument>([] { (void)bandEdgePrototype(45, 0.0, 0.35); }));
        expect(throws<std::invalid_argument>([] { (void)bandEdgePrototype(45, -4.0, 0.35); }));
        expect(throws<std::invalid_argument>([] { (void)bandEdgePrototype(45, 4.0, 1.5); }));
        expect(throws<std::invalid_argument>([] { (void)bandEdgePrototype(45, 4.0, -0.1); }));

        // Even and non-dividing lengths are accepted by the centered grid.
        expect(nothrow([] { (void)designBandEdgeFilters(44, 4.0, 0.35); }));
        expect(nothrow([] { (void)designBandEdgeFilters(23, 2.5, 0.35); }));
    };
};

int main() { /* tests are automatically registered and run */ }
