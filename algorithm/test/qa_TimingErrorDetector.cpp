#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <print>
#include <random>
#include <vector>

#include <gnuradio-4.0/algorithm/sync/TimingErrorDetector.hpp>

namespace {

using gr::sync::detectorGainPerSample;
using gr::sync::earlyLateError;
using gr::sync::earlyLateGain;
using gr::sync::gardnerError;
using gr::sync::gardnerGain;
using gr::sync::modifiedMuellerMullerError;
using gr::sync::modifiedMuellerMullerGain;
using gr::sync::muellerMullerError;
using gr::sync::muellerMullerGain;
using gr::sync::signalTimesSlopeError;
using gr::sync::signalTimesSlopeGain;
using gr::sync::signumTimesSlopeError;
using gr::sync::signumTimesSlopeGain;
using gr::sync::TimingDetector;
using gr::sync::traitsOf;
using gr::sync::zeroCrossingError;
using gr::sync::zeroCrossingGain;

using CD = std::complex<double>;

constexpr double kPi = std::numbers::pi;

/// @brief The raised-cosine channel pulse, `q(t)`, with the symbol period as the time unit.
[[nodiscard]] double raisedCosineAt(double t, double rolloff) { return gr::sync::raisedCosinePulse(t, rolloff); }

/// @brief `q'(t)` in the same units, by a four-point central difference the pulse is smooth enough for.
[[nodiscard]] double raisedCosineSlopeAt(double t, double rolloff) {
    constexpr double h = 1.0e-4;
    return (raisedCosineAt(t - 2.0 * h, rolloff) - 8.0 * raisedCosineAt(t - h, rolloff) //
               + 8.0 * raisedCosineAt(t + h, rolloff) - raisedCosineAt(t + 2.0 * h, rolloff)) /
           (12.0 * h);
}

enum class Modulation { Bpsk, Qpsk };

/**
 * @brief An ideal Nyquist channel sampled at a stated timing error, with everything a detector reads.
 *
 * `tau` is the timing error in symbol periods, positive when the estimate is early, so the sample is
 * taken at `(k - tau)T`. The pulse is evaluated directly rather than through a filter chain and an
 * interpolator, so the gains here are a property of the detectors and not of an interpolator's
 * band-edge droop.
 */
struct NyquistChannel {
    std::vector<CD> sample;    /// y[k]
    std::vector<CD> midpoint;  /// y[k-1/2]
    std::vector<CD> lookahead; /// y[k+1/2]
    std::vector<CD> slope;     /// y'[k], per symbol period, as the maximum-likelihood forms read it
    std::vector<CD> decision;  /// the nearest constellation point of y[k]

    static constexpr int kSpan = 40; /// symbols of pulse tail kept either side
};

[[nodiscard]] NyquistChannel makeChannel(std::size_t nSymbols, double rolloff, double tau, Modulation modulation, double amplitude, std::uint32_t seed) {
    std::mt19937    rng(seed);
    const double    half = amplitude * std::numbers::sqrt2 / 2.0;
    std::vector<CD> symbols(nSymbols);
    for (std::size_t k = 0UZ; k < nSymbols; ++k) {
        const std::uint64_t bits = rng();
        symbols[k]               = modulation == Modulation::Qpsk ? CD{(bits & 1U) != 0U ? half : -half, (bits & 2U) != 0U ? half : -half} //
                                                                  : CD{(bits & 1U) != 0U ? amplitude : -amplitude, 0.0};
    }

    const std::size_t   width = static_cast<std::size_t>(2 * NyquistChannel::kSpan + 1);
    std::vector<double> onGrid(width);
    std::vector<double> earlyGrid(width);
    std::vector<double> lateGrid(width);
    std::vector<double> slopeGrid(width);
    for (int n = -NyquistChannel::kSpan; n <= NyquistChannel::kSpan; ++n) {
        const std::size_t i = static_cast<std::size_t>(n + NyquistChannel::kSpan);
        onGrid[i]           = raisedCosineAt(static_cast<double>(n) - tau, rolloff);
        earlyGrid[i]        = raisedCosineAt(static_cast<double>(n) - 0.5 - tau, rolloff);
        lateGrid[i]         = raisedCosineAt(static_cast<double>(n) + 0.5 - tau, rolloff);
        slopeGrid[i]        = raisedCosineSlopeAt(static_cast<double>(n) - tau, rolloff);
    }

    NyquistChannel out;
    out.sample.assign(nSymbols, CD{});
    out.midpoint.assign(nSymbols, CD{});
    out.lookahead.assign(nSymbols, CD{});
    out.slope.assign(nSymbols, CD{});
    out.decision.assign(nSymbols, CD{});
    for (std::size_t k = static_cast<std::size_t>(NyquistChannel::kSpan); k + static_cast<std::size_t>(NyquistChannel::kSpan) < nSymbols; ++k) {
        CD on{};
        CD early{};
        CD late{};
        CD derivative{};
        for (int n = -NyquistChannel::kSpan; n <= NyquistChannel::kSpan; ++n) {
            const CD          symbol = symbols[k - static_cast<std::size_t>(n)];
            const std::size_t i      = static_cast<std::size_t>(n + NyquistChannel::kSpan);
            on += symbol * onGrid[i];
            early += symbol * earlyGrid[i];
            late += symbol * lateGrid[i];
            derivative += symbol * slopeGrid[i];
        }
        out.sample[k]    = on;
        out.midpoint[k]  = early;
        out.lookahead[k] = late;
        out.slope[k]     = derivative;
        // Sliced, not the transmitted symbol: a detector that needs decisions gets the ones a
        // receiver would have, so the S-curve folds back rather than running away beyond a quarter
        // symbol.
        out.decision[k] = modulation == Modulation::Qpsk ? CD{on.real() >= 0.0 ? half : -half, on.imag() >= 0.0 ? half : -half} //
                                                         : CD{on.real() >= 0.0 ? amplitude : -amplitude, 0.0};
    }
    return out;
}

[[nodiscard]] double errorAt(const NyquistChannel& channel, std::size_t k, TimingDetector detector) {
    switch (detector) {
    case TimingDetector::MuellerMuller: return muellerMullerError(channel.sample[k], channel.sample[k - 1UZ], channel.decision[k], channel.decision[k - 1UZ]);
    case TimingDetector::ModifiedMuellerMuller: return modifiedMuellerMullerError(channel.sample[k], channel.sample[k - 1UZ], channel.sample[k - 2UZ], channel.decision[k], channel.decision[k - 1UZ], channel.decision[k - 2UZ], 1.0e30);
    case TimingDetector::ZeroCrossing: return zeroCrossingError(channel.midpoint[k], channel.decision[k], channel.decision[k - 1UZ]);
    case TimingDetector::Gardner: return gardnerError(channel.sample[k], channel.sample[k - 1UZ], channel.midpoint[k]);
    case TimingDetector::EarlyLate: return earlyLateError(channel.sample[k], channel.midpoint[k], channel.lookahead[k]);
    // The clip is opened out so the gain measured here is the S-curve slope rather than the clip's.
    case TimingDetector::SignalTimesSlopeMl: return signalTimesSlopeError(channel.sample[k], channel.slope[k], 1.0e30);
    case TimingDetector::SignumTimesSlopeMl: return signumTimesSlopeError(channel.sample[k], channel.slope[k], 1.0e30);
    }
    return 0.0;
}

struct Statistics {
    double mean      = 0.0;
    double deviation = 0.0;
    double worst     = 0.0;
};

[[nodiscard]] Statistics statisticsOf(const NyquistChannel& channel, TimingDetector detector) {
    const std::size_t first      = static_cast<std::size_t>(NyquistChannel::kSpan) + 4UZ;
    double            sum        = 0.0;
    double            sumSquares = 0.0;
    double            worst      = 0.0;
    std::size_t       count      = 0UZ;
    for (std::size_t k = first; k + static_cast<std::size_t>(NyquistChannel::kSpan) < channel.sample.size(); ++k) {
        const double error = errorAt(channel, k, detector);
        sum += error;
        sumSquares += error * error;
        worst = std::max(worst, std::abs(error));
        ++count;
    }
    const double mean = sum / static_cast<double>(count);
    return {mean, std::sqrt(std::max(0.0, sumSquares / static_cast<double>(count) - mean * mean)), worst};
}

/// The measured `Kted` table, one row per detector and modulation.
struct GainRow {
    TimingDetector detector   = TimingDetector::MuellerMuller;
    Modulation     modulation = Modulation::Bpsk;
    double         at022      = 0.0;
    double         at035      = 0.0;
    double         at050      = 0.0;
};

constexpr GainRow kGainTable[] = {
    {TimingDetector::MuellerMuller, Modulation::Bpsk, 1.89141, 1.76397, 1.55758},         //
    {TimingDetector::MuellerMuller, Modulation::Qpsk, 1.90283, 1.77355, 1.56435},         //
    {TimingDetector::ModifiedMuellerMuller, Modulation::Bpsk, 3.78285, 3.52798, 3.11518}, //
    {TimingDetector::ModifiedMuellerMuller, Modulation::Qpsk, 3.80566, 3.54714, 3.12875}, //
    {TimingDetector::ZeroCrossing, Modulation::Bpsk, 2.53867, 2.57977, 2.64744},          //
    {TimingDetector::ZeroCrossing, Modulation::Qpsk, 2.56109, 2.60278, 2.67137},          //
    {TimingDetector::Gardner, Modulation::Bpsk, 0.67408, 1.05626, 1.47926},               //
    {TimingDetector::Gardner, Modulation::Qpsk, 0.68004, 1.06739, 1.49611},               //
    {TimingDetector::EarlyLate, Modulation::Bpsk, 0.67397, 1.05616, 1.47918},             //
    {TimingDetector::EarlyLate, Modulation::Qpsk, 0.68008, 1.06743, 1.49614},             //
};

constexpr double kRolloffs[] = {0.22, 0.35, 0.50};
constexpr double kChord      = 0.02; /// the symmetric offset the S-curve slope is taken over

[[nodiscard]] const char* nameOf(TimingDetector detector) {
    switch (detector) {
    case TimingDetector::MuellerMuller: return "M&M";
    case TimingDetector::ModifiedMuellerMuller: return "modified M&M";
    case TimingDetector::ZeroCrossing: return "zero crossing";
    case TimingDetector::Gardner: return "Gardner";
    case TimingDetector::EarlyLate: return "early-late";
    case TimingDetector::SignalTimesSlopeMl: return "signal x slope";
    case TimingDetector::SignumTimesSlopeMl: return "signum x slope";
    }
    return "?";
}

} // namespace

const boost::ut::suite<"timing error detectors"> timingDetectorTests = [] {
    using namespace boost::ut;

    "the gain table, measured on an ideal Nyquist channel"_test = [] {
        constexpr std::size_t nSymbols = 20000UZ;

        for (std::size_t r = 0UZ; r < std::size(kRolloffs); ++r) {
            for (Modulation modulation : {Modulation::Bpsk, Modulation::Qpsk}) {
                const NyquistChannel forward  = makeChannel(nSymbols, kRolloffs[r], +kChord, modulation, 1.0, 5150U);
                const NyquistChannel backward = makeChannel(nSymbols, kRolloffs[r], -kChord, modulation, 1.0, 5150U);

                for (const GainRow& row : kGainTable) {
                    if (row.modulation != modulation) {
                        continue;
                    }
                    const double expected = r == 0UZ ? row.at022 : (r == 1UZ ? row.at035 : row.at050);
                    const double measured = (statisticsOf(forward, row.detector).mean - statisticsOf(backward, row.detector).mean) / (2.0 * kChord);

                    expect(gt(measured, 0.0)) << std::format("{} {}: Kted must be positive under the early-is-positive convention", nameOf(row.detector), modulation == Modulation::Qpsk ? "QPSK" : "BPSK");
                    expect(lt(std::abs(measured - expected) / expected, 0.03)) << std::format("{} {} rolloff {}: Kted {:.5f} against {:.5f}", nameOf(row.detector), modulation == Modulation::Qpsk ? "QPSK" : "BPSK", kRolloffs[r], measured, expected);
                    if (r == 1UZ && modulation == Modulation::Qpsk) {
                        std::println("Kted {:>14} QPSK rolloff 0.35: {:.5f} (table {:.5f})", nameOf(row.detector), measured, expected);
                    }
                }
            }
        }
    };

    "Mueller and Muller has a closed form, singularity and all"_test = [] {
        expect(approx(muellerMullerGain(0.0), 2.0, 1.0e-12)) << "2 E|a|^2 at zero rolloff";
        expect(approx(muellerMullerGain(0.22), 1.91100, 1.0e-5));
        expect(approx(muellerMullerGain(0.35), 1.78035, 1.0e-5));
        expect(approx(muellerMullerGain(0.50), kPi / 2.0, 1.0e-12)) << "the removable singularity at alpha = 1/2 is exactly pi/2";

        // Approached rather than hit, the plain expression has to agree with its own limit.
        for (double epsilon : {1.0e-3, 1.0e-5, 1.0e-7}) {
            expect(lt(std::abs(muellerMullerGain(0.5 + epsilon) - kPi / 2.0), 1.0e-2)) << std::format("alpha = 0.5 + {:.0e}", epsilon);
            expect(lt(std::abs(muellerMullerGain(0.5 - epsilon) - kPi / 2.0), 1.0e-2)) << std::format("alpha = 0.5 - {:.0e}", epsilon);
        }
        expect(that % std::isfinite(muellerMullerGain(0.5)));

        const std::size_t nSymbols = 20000UZ;
        for (double rolloff : {0.22, 0.35, 0.50}) {
            const NyquistChannel forward  = makeChannel(nSymbols, rolloff, +kChord, Modulation::Qpsk, 1.0, 5150U);
            const NyquistChannel backward = makeChannel(nSymbols, rolloff, -kChord, Modulation::Qpsk, 1.0, 5150U);
            const double         measured = (statisticsOf(forward, TimingDetector::MuellerMuller).mean - statisticsOf(backward, TimingDetector::MuellerMuller).mean) / (2.0 * kChord);
            expect(lt(std::abs(measured - muellerMullerGain(rolloff)) / muellerMullerGain(rolloff), 0.015)) << std::format("rolloff {}: measured {:.5f} against the closed form {:.5f}", rolloff, measured, muellerMullerGain(rolloff));

            const double modified = (statisticsOf(forward, TimingDetector::ModifiedMuellerMuller).mean - statisticsOf(backward, TimingDetector::ModifiedMuellerMuller).mean) / (2.0 * kChord);
            expect(lt(std::abs(modified / measured - 2.0), 0.01)) << std::format("rolloff {}: the modified form is exactly twice, {:.5f} against {:.5f}", rolloff, modified, measured);
            expect(approx(modifiedMuellerMullerGain(rolloff), 2.0 * muellerMullerGain(rolloff), 1.0e-15));
        }
    };

    "every gain function agrees with the channel it is quoted on"_test = [] {
        // The gain functions evaluate an expectation over the pulse; this measures the same quantity by running the
        // kernels over a modulated channel and taking the secant of the S-curve. Agreement across all seven is what
        // proves the two share a sign convention and a scale, and it is the only check the two maximum-likelihood
        // forms have, their gains reaching no closed form the recorded table could hold.
        constexpr TimingDetector kAll[] = {TimingDetector::MuellerMuller, TimingDetector::ModifiedMuellerMuller, TimingDetector::ZeroCrossing, //
            TimingDetector::Gardner, TimingDetector::EarlyLate, TimingDetector::SignalTimesSlopeMl, TimingDetector::SignumTimesSlopeMl};

        // The measurement is a mean over a finite draw, so its own error sets the tolerance: the blind forms on BPSK
        // are the noisiest of the twenty-one cases and the symbol count is chosen to bring them inside 2%.
        double worst = 0.0;
        for (double rolloff : kRolloffs) {
            for (Modulation modulation : {Modulation::Bpsk, Modulation::Qpsk}) {
                // The signum form is the one gain that scales with the constellation's mean axis magnitude rather
                // than with its power, and a real-valued symbol leaves the imaginary term of the kernel at zero.
                const double axis   = modulation == Modulation::Qpsk ? std::numbers::sqrt2 / 2.0 : 0.5;
                const auto   gainOf = [rolloff, axis](TimingDetector detector) {
                    switch (detector) {
                    case TimingDetector::MuellerMuller: return muellerMullerGain(rolloff);
                    case TimingDetector::ModifiedMuellerMuller: return modifiedMuellerMullerGain(rolloff);
                    case TimingDetector::ZeroCrossing: return zeroCrossingGain(rolloff);
                    case TimingDetector::Gardner: return gardnerGain(rolloff);
                    case TimingDetector::EarlyLate: return earlyLateGain(rolloff);
                    case TimingDetector::SignalTimesSlopeMl: return signalTimesSlopeGain(rolloff);
                    case TimingDetector::SignumTimesSlopeMl: return signumTimesSlopeGain(rolloff, axis);
                    }
                    return 0.0;
                };
                const NyquistChannel forward  = makeChannel(400000UZ, rolloff, +kChord, modulation, 1.0, 5150U);
                const NyquistChannel backward = makeChannel(400000UZ, rolloff, -kChord, modulation, 1.0, 5150U);

                for (TimingDetector detector : kAll) {
                    const double measured = (statisticsOf(forward, detector).mean - statisticsOf(backward, detector).mean) / (2.0 * kChord);
                    const double stated   = gainOf(detector);
                    const double error    = std::abs(measured - stated) / stated;
                    worst                 = std::max(worst, error);
                    expect(lt(error, 0.02)) << std::format("{} {} rolloff {}: measured {:.5f} against the stated {:.5f}", nameOf(detector), modulation == Modulation::Qpsk ? "QPSK" : "BPSK", rolloff, measured, stated);
                    if (rolloff == 0.35 && modulation == Modulation::Qpsk) {
                        std::println("Kted {:>14} QPSK rolloff 0.35: stated {:.5f}, measured {:.5f}", nameOf(detector), stated, measured);
                    }
                }
            }
        }
        std::println("the worst of the twenty-one gain comparisons is {:.2f}%", 100.0 * worst);
    };

    "the gains that reach a closed form hit it exactly"_test = [] {
        // Anchors that hold whatever the channel does: two limits of the zero-crossing gain, the linearity of the
        // maximum-likelihood form in the excess bandwidth, and the identity between Gardner and early-late.
        expect(approx(zeroCrossingGain(0.0), 8.0 / kPi, 1.0e-8)) << "sinc leaves 8/pi";
        expect(approx(zeroCrossingGain(1.0), 3.0, 1.0e-7)) << "and full excess bandwidth leaves exactly 3";

        for (double rolloff : {0.0, 0.13, 0.35, 0.5, 0.77, 1.0}) {
            expect(approx(signalTimesSlopeGain(rolloff), kPi * kPi * rolloff / 4.0, 1.0e-15)) << std::format("rolloff {}", rolloff);
            expect(approx(earlyLateGain(rolloff), gardnerGain(rolloff), 1.0e-15)) << std::format("rolloff {}: one S-curve, one gain", rolloff);
            expect(approx(gardnerGain(rolloff, 4.0), 4.0 * gardnerGain(rolloff), 1.0e-12)) << "and the blind gains carry the symbol power";
        }

        // A pulse confined to the Nyquist band carries no timing line for a detector that reads no decisions.
        expect(eq(gardnerGain(0.0), 0.0));
        expect(eq(earlyLateGain(0.0), 0.0));
        expect(eq(signalTimesSlopeGain(0.0), 0.0));
        // Taking a sign is itself a decision, so that one form keeps its gain there.
        expect(approx(signumTimesSlopeGain(0.0, 1.0), kPi * kPi / 3.0, 1.0e-15));
    };

    "the gain per input sample is Kted divided by the samples per symbol"_test = [] {
        struct SampleRow {
            double samplesPerSymbol = 0.0;
            double muellerMuller    = 0.0;
            double gardner          = 0.0;
            double zeroCrossing     = 0.0;
            double modified         = 0.0;
        };
        constexpr SampleRow kRows[] = {{2.0, 0.88678, 0.53370, 1.30139, 1.77357}, {4.0, 0.44339, 0.26685, 0.65070, 0.88679}, {8.0, 0.22169, 0.13342, 0.32535, 0.44339}};

        for (const SampleRow& row : kRows) {
            expect(approx(detectorGainPerSample(1.77355, row.samplesPerSymbol), row.muellerMuller, 1.0e-5)) << std::format("sps={}", row.samplesPerSymbol);
            expect(approx(detectorGainPerSample(1.06739, row.samplesPerSymbol), row.gardner, 1.0e-5));
            expect(approx(detectorGainPerSample(2.60278, row.samplesPerSymbol), row.zeroCrossing, 1.0e-5));
            expect(approx(detectorGainPerSample(3.54714, row.samplesPerSymbol), row.modified, 1.0e-5));
        }
        expect(eq(detectorGainPerSample(1.0, 1.0), 1.0)) << "and it is a division";
    };

    "the S-curve is odd, crosses zero once, and returns to zero at the half symbol"_test = [] {
        constexpr double kTaus[] = {-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
        struct CurveRow {
            double muellerMuller = 0.0;
            double gardner       = 0.0;
            double zeroCrossing  = 0.0;
            double earlyLate     = 0.0;
        };
        constexpr CurveRow kCurve[] = {
            {-0.31262, -0.10010, -0.45958, -0.10012}, //
            {-0.50819, -0.16199, -0.71047, -0.16201}, //
            {-0.34763, -0.16199, -0.49940, -0.16201}, //
            {-0.17649, -0.10012, -0.25769, -0.10014}, //
            {0.00000, -0.00001, -0.00001, -0.00002},  //
            {0.17649, 0.10011, 0.25768, 0.10010},     //
            {0.34763, 0.16199, 0.49939, 0.16199},     //
            {0.50818, 0.16200, 0.71047, 0.16199},     //
            {0.31845, 0.10013, 0.46648, 0.10012},     //
            {-0.00050, 0.00001, -0.00090, 0.00001},   //
        };

        for (std::size_t i = 0UZ; i < std::size(kTaus); ++i) {
            const NyquistChannel channel   = makeChannel(20000UZ, 0.35, kTaus[i], Modulation::Qpsk, 1.0, 5150U);
            const double         muller    = statisticsOf(channel, TimingDetector::MuellerMuller).mean;
            const double         gardner   = statisticsOf(channel, TimingDetector::Gardner).mean;
            const double         crossing  = statisticsOf(channel, TimingDetector::ZeroCrossing).mean;
            const double         earlyLate = statisticsOf(channel, TimingDetector::EarlyLate).mean;

            // Gardner and early-late are two arrangements of the same measurement.
            expect(lt(std::abs(gardner - earlyLate), 1.0e-3)) << std::format("tau={:+.3f}: Gardner {:+.5f} against early-late {:+.5f}", kTaus[i], gardner, earlyLate);

            const auto close = [](double measured, double expected) { return std::abs(measured - expected) < 0.03 * std::abs(expected) + 0.006; };
            expect(that % close(muller, kCurve[i].muellerMuller)) << std::format("tau={:+.3f}: M&M {:+.5f} against {:+.5f}", kTaus[i], muller, kCurve[i].muellerMuller);
            expect(that % close(gardner, kCurve[i].gardner)) << std::format("tau={:+.3f}: Gardner {:+.5f} against {:+.5f}", kTaus[i], gardner, kCurve[i].gardner);
            expect(that % close(crossing, kCurve[i].zeroCrossing)) << std::format("tau={:+.3f}: zero crossing {:+.5f} against {:+.5f}", kTaus[i], crossing, kCurve[i].zeroCrossing);

            if (kTaus[i] == 0.0 || kTaus[i] == 0.5) {
                expect(lt(std::abs(muller), 0.01)) << "zero at the lock point and at the half-symbol false lock";
                expect(lt(std::abs(gardner), 0.01));
                expect(lt(std::abs(crossing), 0.01));
                expect(lt(std::abs(earlyLate), 0.01));
            }
            std::println("S-curve tau={:+.3f}: M&M {:+.5f}  Gardner {:+.5f}  zero crossing {:+.5f}  early-late {:+.5f}", kTaus[i], muller, gardner, crossing, earlyLate);
        }
    };

    "the sign is the one the loop needs, for every detector"_test = [] {
        // One assertion per detector; the wrong sign makes the loop diverge rather than converge.
        const NyquistChannel early = makeChannel(8000UZ, 0.35, +0.02, Modulation::Qpsk, 1.0, 77U);
        const NyquistChannel late  = makeChannel(8000UZ, 0.35, -0.02, Modulation::Qpsk, 1.0, 77U);
        for (TimingDetector detector : {TimingDetector::MuellerMuller, TimingDetector::ModifiedMuellerMuller, TimingDetector::ZeroCrossing, TimingDetector::Gardner, TimingDetector::EarlyLate}) {
            expect(gt(statisticsOf(early, detector).mean, 0.0)) << std::format("{}: E[e] at tau = +0.02 is positive", nameOf(detector));
            expect(lt(statisticsOf(late, detector).mean, 0.0)) << std::format("{}: E[e] at tau = -0.02 is negative", nameOf(detector));
        }
    };

    "self-noise, including the exact zero that makes M&M the default where decisions exist"_test = [] {
        struct NoiseRow {
            TimingDetector detector = TimingDetector::Gardner;
            double         bpsk     = 0.0;
            double         qpsk     = 0.0;
        };
        constexpr NoiseRow kNoise[] = {{TimingDetector::Gardner, 0.34341, 0.24574}, {TimingDetector::ZeroCrossing, 0.34341, 0.24574}, {TimingDetector::EarlyLate, 1.14566, 0.81221}};

        for (Modulation modulation : {Modulation::Bpsk, Modulation::Qpsk}) {
            const NyquistChannel channel = makeChannel(20000UZ, 0.35, 0.0, modulation, 1.0, 5150U);

            for (TimingDetector detector : {TimingDetector::MuellerMuller, TimingDetector::ModifiedMuellerMuller}) {
                const Statistics statistics = statisticsOf(channel, detector);
                expect(lt(statistics.worst, 1.0e-6)) << std::format("{}: every single-symbol error is zero on an ideal Nyquist channel with correct decisions, worst {:.3g}", nameOf(detector), statistics.worst);
            }

            for (const NoiseRow& row : kNoise) {
                const double measured = statisticsOf(channel, row.detector).deviation;
                const double expected = modulation == Modulation::Qpsk ? row.qpsk : row.bpsk;
                expect(lt(std::abs(measured - expected) / expected, 0.05)) << std::format("{} {}: self-noise {:.5f} against {:.5f}", nameOf(row.detector), modulation == Modulation::Qpsk ? "QPSK" : "BPSK", measured, expected);
            }

            const double gardner  = statisticsOf(channel, TimingDetector::Gardner).deviation;
            const double crossing = statisticsOf(channel, TimingDetector::ZeroCrossing).deviation;
            expect(lt(std::abs(gardner - crossing), 1.0e-9)) << "Gardner and zero crossing coincide exactly at tau = 0 with correct decisions, so their raw self-noise is the same number";

            const double earlyLate = statisticsOf(channel, TimingDetector::EarlyLate).deviation;
            expect(gt(earlyLate / gardner, 3.2)) << std::format("early-late costs {:.2f}x Gardner's self-noise for an identical S-curve", earlyLate / gardner);
            expect(lt(earlyLate / gardner, 3.4));
        }
    };

    "Gardner and early-late are blind, and M&M is not"_test = [] {
        const NyquistChannel channel = makeChannel(4000UZ, 0.35, 0.05, Modulation::Qpsk, 1.0, 5150U);
        const double         gardner = statisticsOf(channel, TimingDetector::Gardner).mean;
        const double         late    = statisticsOf(channel, TimingDetector::EarlyLate).mean;
        const double         muller  = statisticsOf(channel, TimingDetector::MuellerMuller).mean;

        std::mt19937                           rng(9U);
        std::uniform_real_distribution<double> angle(0.0, 2.0 * kPi);
        double                                 mullerDrift = 0.0;
        for (int trial = 0; trial < 10; ++trial) {
            const CD       rotation = std::polar(1.0, angle(rng));
            NyquistChannel rotated  = channel;
            for (CD& v : rotated.sample) {
                v *= rotation;
            }
            for (CD& v : rotated.midpoint) {
                v *= rotation;
            }
            for (CD& v : rotated.lookahead) {
                v *= rotation;
            }
            // The decisions are left unrotated, as they stand before a carrier loop runs.
            expect(lt(std::abs(statisticsOf(rotated, TimingDetector::Gardner).mean - gardner), 1.0e-6)) << "a common rotation leaves Re{(u-v)conj(w)} unchanged";
            expect(lt(std::abs(statisticsOf(rotated, TimingDetector::EarlyLate).mean - late), 1.0e-6));
            mullerDrift = std::max(mullerDrift, std::abs(statisticsOf(rotated, TimingDetector::MuellerMuller).mean - muller));
        }
        expect(gt(mullerDrift, 0.01)) << std::format("and M&M's error does move, by {:.4f}", mullerDrift);
    };

    "the gain is proportional to the symbol power, which is why the AGC is mandatory"_test = [] {
        double gains[3] = {};
        int    at       = 0;
        for (double amplitude : {0.5, 1.0, 2.0}) {
            const NyquistChannel forward  = makeChannel(20000UZ, 0.35, +kChord, Modulation::Qpsk, amplitude, 5150U);
            const NyquistChannel backward = makeChannel(20000UZ, 0.35, -kChord, Modulation::Qpsk, amplitude, 5150U);
            gains[at++]                   = (statisticsOf(forward, TimingDetector::MuellerMuller).mean - statisticsOf(backward, TimingDetector::MuellerMuller).mean) / (2.0 * kChord);
        }
        expect(lt(std::abs(gains[1] / gains[0] - 4.0), 0.04)) << std::format("Kted at A = 0.5 and 1.0: {:.5f}, {:.5f}", gains[0], gains[1]);
        expect(lt(std::abs(gains[2] / gains[1] - 4.0), 0.04)) << std::format("Kted at A = 1.0 and 2.0: {:.5f}, {:.5f}", gains[1], gains[2]);
        expect(lt(std::abs(gains[1] - muellerMullerGain(0.35)) / muellerMullerGain(0.35), 0.015)) << "and the unit-power value is the closed form";
    };

    "the kernels are pure, and their traits describe what a scheduler needs"_test = [] {
        expect(eq(traitsOf(TimingDetector::MuellerMuller).inputsPerSymbol, 1));
        expect(that % traitsOf(TimingDetector::MuellerMuller).needsDecisions);
        expect(eq(traitsOf(TimingDetector::ModifiedMuellerMuller).errorDepth, 3)) << "it spans two symbol intervals, so it reads three samples";
        expect(eq(traitsOf(TimingDetector::Gardner).inputsPerSymbol, 2));
        expect(that % !traitsOf(TimingDetector::Gardner).needsDecisions);
        expect(that % !traitsOf(TimingDetector::Gardner).needsLookahead);
        expect(that % traitsOf(TimingDetector::EarlyLate).needsLookahead) << "early-late alone reads a sample after the symbol instant";
        expect(that % traitsOf(TimingDetector::SignalTimesSlopeMl).needsDerivative);
        expect(that % traitsOf(TimingDetector::SignumTimesSlopeMl).needsDerivative);

        // Real input is the same formula with the imaginary term absent, and it must compile as such.
        expect(approx(muellerMullerError(0.5f, 0.25f, 1.0f, 1.0f), 0.25f, 1.0e-7f));
        expect(approx(gardnerError(1.0, 0.0, 0.5), -0.5, 1.0e-15));
        expect(approx(earlyLateError(1.0, 0.25, 0.75), 0.5, 1.0e-15));
        expect(approx(zeroCrossingError(0.5, 1.0, -1.0), -1.0, 1.0e-15));

        // The three that clip do so symmetrically; the rest run unbounded.
        expect(approx(modifiedMuellerMullerError(CD{50.0, 0.0}, CD{}, CD{}, CD{1.0, 0.0}, CD{1.0, 0.0}, CD{}), 1.0, 1.0e-15));
        expect(approx(modifiedMuellerMullerError(CD{-50.0, 0.0}, CD{}, CD{}, CD{1.0, 0.0}, CD{1.0, 0.0}, CD{}), -1.0, 1.0e-15));
        expect(approx(signalTimesSlopeError(CD{4.0, 0.0}, CD{4.0, 0.0}), 1.0, 1.0e-15));
        expect(approx(signalTimesSlopeError(CD{1.0, 0.0}, CD{1.0, 0.0}), 0.5, 1.0e-15));
        expect(approx(signumTimesSlopeError(CD{7.0, 0.0}, CD{0.5, 0.0}), 0.25, 1.0e-15)) << "the signum removes one factor of amplitude";
        expect(approx(signumTimesSlopeError(CD{0.0, 0.0}, CD{0.5, 0.5}), 0.0, 1.0e-15)) << "and a zero sample contributes nothing rather than a sign";
        expect(gt(std::abs(gardnerError(CD{100.0, 0.0}, CD{}, CD{100.0, 0.0})), 100.0)) << "Gardner does not clip";
    };

    "the maximum-likelihood forms slope the right way against a real derivative"_test = [] {
        // These carry no published Kted, so what is asserted is the sign convention and the two
        // amplitude laws, which are what the loop depends on.
        for (double amplitude : {0.5, 1.0, 2.0}) {
            const CD sample{amplitude, 0.0};
            const CD rising{amplitude, 0.0};
            expect(gt(signalTimesSlopeError(sample, rising, 1.0e30), 0.0)) << "on the rising edge before the peak, which is where an early estimate lands";
            expect(lt(signalTimesSlopeError(sample, CD{-amplitude, 0.0}, 1.0e30), 0.0));
            expect(approx(signalTimesSlopeError(sample, rising, 1.0e30), amplitude * amplitude / 2.0, 1.0e-12)) << "signal x slope carries the amplitude twice";
            expect(approx(signumTimesSlopeError(sample, rising, 1.0e30), amplitude / 2.0, 1.0e-12)) << "signum x slope carries it once";
        }
    };
};

int main() { /* tests are automatically registered and run */ }
