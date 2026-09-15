#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/FilterDesign.hpp>

namespace {

[[nodiscard]] double tapSum(const std::vector<float>& taps) {
    return std::accumulate(taps.begin(), taps.end(), 0.0, [](double acc, float v) { return acc + static_cast<double>(v); });
}

template<typename T>
[[nodiscard]] bool identical(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

[[nodiscard]] double worstAsymmetry(const std::vector<float>& taps) {
    double worst = 0.0;
    for (std::size_t i = 0UZ; i < taps.size(); ++i) {
        worst = std::max(worst, std::abs(static_cast<double>(taps[i]) - static_cast<double>(taps[taps.size() - 1UZ - i])));
    }
    return worst;
}

} // namespace

const boost::ut::suite<"FIR lowpass design"> filterDesignTests = [] {
    using namespace boost::ut;
    using namespace gr::filter::fir::design;

    "the Kaiser length estimate"_test = [] {
        expect(eq(kaiserLength(80.0, 0.0), 3UZ)) << "a zero transition width has no estimate";
        expect(gt(kaiserLength(80.0, 0.025), kaiserLength(80.0, 0.05))) << "a narrower transition costs taps";
        expect(gt(kaiserLength(80.0, 0.05), kaiserLength(60.0, 0.05))) << "a deeper stopband costs taps";
    };

    "kaiserLowpass tap-set invariants"_test = [] {
        for (const std::size_t requested : {3UZ, 4UZ, 31UZ, 32UZ, 101UZ}) {
            const std::vector<float> taps = kaiserLowpass(requested, 0.2, 60.0);
            expect(eq(taps.size() % 2UZ, 1UZ)) << "odd length keeps the group delay a whole sample";
            expect(ge(taps.size(), requested));
            expect(le(taps.size(), requested + 1UZ));
            expect(approx(tapSum(taps), 1.0, 1e-6)) << "unit gain at DC";
            expect(eq(worstAsymmetry(taps), 0.0)) << "the second half is copied, not re-evaluated: symmetric to the bit";
        }
        expect(eq(kaiserLowpass(1, 0.2, 60.0).size(), 3UZ)) << "three taps is the floor";
        expect(eq(kaiserLowpass(0, 0.2, 60.0).size(), 3UZ));
    };

    "kaiserLowpass numeric anchors"_test = [] {
        const std::vector<float> taps = kaiserLowpass(31, 0.25, 60.0);
        expect(eq(taps.size(), 31UZ));
        expect(approx(static_cast<double>(taps[15UZ]), 0.5, 1e-4)) << "a halfband's center tap is one half";
        expect(approx(static_cast<double>(taps[15UZ]), 0.49998709559440613, 1e-7));
        expect(approx(static_cast<double>(taps[14UZ]), 0.31469336152076721, 1e-7));
        expect(approx(static_cast<double>(taps[0UZ]), -0.00043263562838546932, 1e-9));
    };

    "a halfband's alternate taps are zeros"_test = [] {
        // cutoff at a quarter of the rate makes the sinc sin(pi t / 2) / (pi t): every tap at an
        // even offset from the center lands on a rounded multiple of pi and is zero to the last
        // bits the window is computed in. These are the taps a halving stage skips.
        for (const int n : {31, 63}) {
            const std::vector<float> taps = kaiserLowpass(static_cast<std::size_t>(n), 0.25, 80.0);
            const int                mid  = (static_cast<int>(taps.size()) - 1) / 2;

            double peak = 0.0;
            for (const float v : taps) {
                peak = std::max(peak, std::abs(static_cast<double>(v)));
            }
            double worstZero = 0.0;
            for (int i = 0; i < static_cast<int>(taps.size()); ++i) {
                const int offset = i - mid;
                if (offset != 0 && (offset % 2) == 0) {
                    worstZero = std::max(worstZero, std::abs(static_cast<double>(taps[static_cast<std::size_t>(i)])));
                }
            }
            const double belowPeakDb = 20.0 * std::log10(std::max(worstZero, 1e-300) / peak);
            expect(lt(belowPeakDb, -300.0)) << "worst alternate tap sits " << belowPeakDb << " dB under the peak";
        }
    };

    "halfResponse is the tap set read back"_test = [] {
        const std::vector<float> taps = kaiserLowpass(101, 0.225, 80.0);
        std::vector<double>      mag;
        halfResponse(taps, mag);

        expect(eq(mag.size(), kDesignGrid / 2UZ + 1UZ));
        expect(approx(mag[0UZ], tapSum(taps), 1e-12)) << "DC is the sum of the taps";

        double alternating = 0.0;
        for (std::size_t i = 0UZ; i < taps.size(); ++i) {
            alternating += ((i % 2UZ) == 0UZ ? 1.0 : -1.0) * static_cast<double>(taps[i]);
        }
        expect(approx(mag.back(), std::abs(alternating), 1e-12)) << "Nyquist is the alternating sum";
    };

    "the design cutoff is where the response is half"_test = [] {
        // scanLowpass over [0, cutoff] spans exactly the 6.02 dB of half amplitude, which pins the
        // edge's position rather than a flatness the design does not claim.
        constexpr double         cutoff = 0.225;
        const std::vector<float> taps   = kaiserLowpass(101, cutoff, 80.0);
        expect(approx(scanLowpass(taps, cutoff, 0.25).rippleDb, 6.02, 0.05));
        expect(lt(scanLowpass(taps, 0.15, 0.25).rippleDb, 1.0)) << "flat to a dB over the inner two thirds";
    };

    "scanLowpass measures what the design was asked for"_test = [] {
        for (const double attenDb : {60.0, 80.0}) {
            constexpr double         passEdge = 0.20;
            constexpr double         stopEdge = 0.25;
            const LowpassSearch      found    = searchLowpass(passEdge, stopEdge, attenDb, 0.1, 501);
            const std::vector<float> taps     = kaiserLowpass(found.taps, 0.5 * (passEdge + stopEdge), attenDb);
            const LowpassScan        scan     = scanLowpass(taps, passEdge, stopEdge);

            expect(that % found.ok) << "a length under the cap delivers " << attenDb << " dB";
            expect(le(scan.stopbandDb, -attenDb)) << "stopband measured at " << scan.stopbandDb << " dB";
            expect(le(scan.rippleDb, 0.1)) << "passband ripple measured at " << scan.rippleDb << " dB";
            expect(eq(scan.stopbandDb, found.scan.stopbandDb)) << "the search reports the scan it ran";
            expect(eq(scan.rippleDb, found.scan.rippleDb));
        }
    };

    "searchLowpass returns the shortest length that passes"_test = [] {
        constexpr double passEdge    = 0.20;
        constexpr double stopEdge    = 0.25;
        constexpr double attenDb     = 60.0;
        constexpr double maxRippleDb = 0.1;

        const LowpassSearch found = searchLowpass(passEdge, stopEdge, attenDb, maxRippleDb, 501);
        expect(that % found.ok);
        expect(eq(found.taps % 2UZ, 1UZ)) << "odd, so the group delay stays a whole sample";
        expect(lt(found.taps, 501UZ));
        expect(gt(found.taps, 5UZ));

        const double      cutoff  = 0.5 * (passEdge + stopEdge);
        const LowpassScan shorter = scanLowpass(kaiserLowpass(found.taps - 2UZ, cutoff, attenDb), passEdge, stopEdge);
        expect(that % (shorter.stopbandDb > -attenDb || shorter.rippleDb > maxRippleDb)) << "two taps shorter must miss a target";

        expect(le(found.taps, kaiserLength(attenDb, stopEdge - passEdge) * 2UZ)) << "the search stays within twice the estimate";
    };

    "searchLowpass reports failure when no length meets the target"_test = [] {
        const LowpassSearch found = searchLowpass(0.20, 0.25, 60.0, 0.0, 101);
        expect(that % !found.ok) << "a windowed design always has some passband ripple";
        expect(eq(found.taps, 101UZ)) << "the longest length tried is reported";
        expect(gt(found.scan.rippleDb, 0.0));
    };

    "oddLength rounds up and respects the floor"_test = [] {
        expect(eq(oddLength(31UZ), 31UZ));
        expect(eq(oddLength(32UZ), 33UZ));
        expect(eq(oddLength(0UZ), 3UZ));
        expect(eq(oddLength(4UZ, 5UZ), 5UZ));
        expect(eq(oddLength(6UZ, 5UZ), 7UZ));
    };

    "fromEdges is the other spelling of the same transition"_test = [] {
        const TransitionBand band = fromEdges(5000.0, 7000.0);
        expect(approx(band.cutoff, 6000.0, 1e-12)) << "the -6 dB point is the midpoint of the two edges";
        expect(approx(band.width, 2000.0, 1e-12));
        expect(approx(fromEdges(7000.0, 5000.0).cutoff, 6000.0, 1e-12)) << "a high-pass states its edges the other way round";
        expect(approx(fromEdges(7000.0, 5000.0).width, 2000.0, 1e-12));
    };

    "amplitudeAt is the tap set read back at one frequency"_test = [] {
        const std::vector<float> taps = kaiserLowpass(101, 0.225, 80.0);
        std::vector<double>      amp;
        halfAmplitude(taps, amp);

        expect(approx(amplitudeAt(taps, 0.0), tapSum(taps), 1e-12)) << "DC is the sum of the taps";
        expect(approx(amplitudeAt(taps, 0.0), amp[0UZ], 1e-12));

        double worst = 0.0;
        for (std::size_t i = 0UZ; i <= kDesignGrid / 2UZ; i += 337UZ) {
            const double f = static_cast<double>(i) / static_cast<double>(kDesignGrid);
            worst          = std::max(worst, std::abs(amplitudeAt(taps, f) - amp[i]));
        }
        expect(lt(worst, 1e-12)) << "the cosine sum and the table-stepped grid agree to " << worst;

        std::vector<double> mag;
        halfResponse(taps, mag);
        bool signChanges = false;
        for (std::size_t i = 0UZ; i < amp.size(); ++i) {
            expect(eq(mag[i], std::abs(amp[i])));
            signChanges = signChanges || amp[i] < 0.0;
        }
        expect(that % signChanges) << "halfAmplitude keeps the sign halfResponse throws away";
    };

    "normalizeAt puts the stated gain at the stated frequency"_test = [] {
        for (const double gain : {1.0, 0.5, 1000.0}) {
            std::vector<double> taps(31UZ, 0.0);
            for (std::size_t i = 0UZ; i < taps.size(); ++i) {
                taps[i] = static_cast<double>(kaiserLowpass(31, 0.25, 60.0)[i]);
            }
            normalizeAt(taps, 0.1, gain);
            expect(approx(amplitudeAt(taps, 0.1), gain, 1e-12 * gain));
        }
        std::vector<double> nulled{1.0, -2.0, 1.0};
        expect(throws<std::invalid_argument>([&] { normalizeAt(nulled, 0.0, 1.0); })) << "a design cannot be normalized at one of its own nulls";
    };

    "scanBand reads the exact edges the grid steps over"_test = [] {
        // A stopband edge sits at the foot of the transition, where the response moves fastest, so a
        // grid that never lands on the stated edge reports a stopband better than the design has.
        const std::vector<float> taps  = kaiserLowpass(87, 0.125, 60.0);
        const BandLevels         band  = scanBand(taps, 0.15, 0.5);
        const LowpassScan        plain = scanLowpass(taps, 0.10, 0.15);

        expect(gt(band.peakDb(), plain.stopbandDb)) << "the exact edge is worse than anything the grid found: " << band.peakDb() << " against " << plain.stopbandDb;
        expect(lt(band.peakDb() - plain.stopbandDb, 0.5)) << "and only by the response's move over one grid step";

        const BandLevels point = scanBand(taps, 0.125, 0.125);
        expect(approx(20.0 * std::log10(point.maxMag), -6.0206, 0.01)) << "a band of zero width is the design's own cutoff, 6 dB down";
    };

    "a fixed-shape window's length buys transition width, not attenuation"_test = [] {
        using gr::algorithm::window::Type;
        // The attenuation is a property of the window; only the length is free. At the length the
        // window's measured transition constant asks for, the design delivers the window's figure.
        constexpr double width  = 0.02;
        constexpr double cutoff = 0.25;
        for (const Type type : {Type::Rectangular, Type::Bartlett, Type::Welch, Type::Hann, Type::Hamming, Type::Parzen, Type::Blackman, Type::FlatTop, Type::BlackmanHarris, Type::Nuttall, Type::BlackmanNuttall}) {
            const WindowFigures      figures  = windowFigures(type);
            const std::size_t        n        = oddLength(windowLength(type, width));
            const std::vector<float> taps     = lowpass(n, cutoff, {type, std::numeric_limits<double>::quiet_NaN()});
            const double             measured = -scanBand(taps, cutoff + 0.5 * width, 0.5).peakDb();

            expect(gt(figures.transitionConstant, 0.0));
            expect(approx(measured, figures.attenuationDb, 1.5)) << "window " << static_cast<int>(type) << " at " << n << " taps delivers " << measured << " dB against a tabled " << figures.attenuationDb;
        }
        expect(throws<std::invalid_argument>([] { [[maybe_unused]] const WindowFigures unused = windowFigures(Type::Kaiser); })) << "asking a shape-parameter window for a fixed figure is a category error";
        expect(throws<std::invalid_argument>([] { [[maybe_unused]] const WindowFigures unused = windowFigures(Type::Gaussian); }));
    };

    "the tap-count estimate and the search pin each other"_test = [] {
        // Kaiser's estimate marks where a design first touches its target. These are the measured
        // shortfalls, pinned rather than asserted away: a future improvement to the estimator
        // breaks this test deliberately.
        struct Row {
            double      attenDb;
            double      width;
            std::size_t estimated;
            double      deliveredDb;
            std::size_t shortest;
        };
        constexpr std::array<Row, 12> table{{{40.0, 0.010, 223, -39.07, 225}, {40.0, 0.020, 113, -40.10, 113}, {40.0, 0.050, 45, -38.21, 47}, //
            {60.0, 0.010, 363, -59.89, 417}, {60.0, 0.020, 183, -59.83, 209}, {60.0, 0.050, 73, -58.44, 85},                                  //
            {80.0, 0.010, 503, -79.67, 545}, {80.0, 0.020, 251, -79.43, 273}, {80.0, 0.050, 101, -79.07, 111},                                //
            {100.0, 0.010, 641, -98.73, 643}, {100.0, 0.020, 321, -98.72, 323}, {100.0, 0.050, 129, -98.65, 131}}};

        constexpr double cutoff = 0.25;
        for (const Row& row : table) {
            expect(eq(oddLength(kaiserLength(row.attenDb, row.width)), row.estimated)) << "estimate at " << row.attenDb << " dB";

            const double delivered = scanBand(kaiserLowpass(row.estimated, cutoff, row.attenDb), cutoff + 0.5 * row.width, 0.5).peakDb();
            expect(approx(delivered, row.deliveredDb, 0.05)) << "the estimated length delivers " << delivered << " dB against a requested " << row.attenDb;

            const LowpassSearch found = searchLowpass(cutoff - 0.5 * row.width, cutoff + 0.5 * row.width, row.attenDb, 1.0, 701);
            expect(that % found.ok);
            expect(eq(found.taps, row.shortest)) << "shortest length that measurably meets " << row.attenDb << " dB";
        }
    };

    "a grid too coarse for the length over-reports the stopband"_test = [] {
        // The warning kDesignGrid documents, made a check: a coarse grid steps over the stopband
        // peaks it is meant to find, and reports a filter as clearing a target it does not meet.
        const std::vector<float> taps   = kaiserLowpass(101, 0.225, 80.0);
        const double             fine   = scanLowpass(taps, 0.20, 0.25).stopbandDb;
        const double             coarse = scanLowpass(taps, 0.20, 0.25, 256).stopbandDb;
        expect(lt(coarse, fine)) << "coarse " << coarse << " dB flatters the design against fine " << fine << " dB";
        expect(gt(fine, -80.0)) << "the fine grid finds that 101 taps does not in fact reach 80 dB here";
        expect(lt(coarse, -80.0)) << "the coarse grid would have passed it";
    };
};

const boost::ut::suite<"FIR band transform design"> bandDesignTests = [] {
    using namespace boost::ut;
    using namespace gr::filter::fir::design;

    const WindowSpec kaiser60 = kaiserFor(60.0);

    "every design returns an odd count, and respects its floor"_test = [&] {
        for (const std::size_t requested : {1UZ, 2UZ, 3UZ, 4UZ, 30UZ, 31UZ, 32UZ}) {
            expect(eq(lowpass(requested, 0.2, kaiser60).size() % 2UZ, 1UZ));
            expect(eq(highpass(requested, 0.2, kaiser60).size() % 2UZ, 1UZ));
            expect(eq(bandpass(requested, 0.1, 0.2, kaiser60).size() % 2UZ, 1UZ));
            expect(eq(bandstop(requested, 0.1, 0.2, kaiser60).size() % 2UZ, 1UZ));
        }
        expect(eq(lowpass(32, 0.2, kaiser60).size(), 33UZ)) << "an even request is rounded up";
        expect(eq(highpass(1, 0.2, kaiser60).size(), 3UZ)) << "three taps is the floor";
        expect(eq(bandpass(1, 0.1, 0.2, kaiser60).size(), 5UZ)) << "a band design needs a center tap and one pair";
        expect(eq(bandstop(4, 0.1, 0.2, kaiser60).size(), 5UZ));
    };

    "real designs are symmetric to the bit"_test = [&] {
        expect(eq(worstAsymmetry(lowpass(87, 0.125, kaiser60)), 0.0));
        expect(eq(worstAsymmetry(highpass(87, 0.125, kaiser60)), 0.0));
        expect(eq(worstAsymmetry(bandpass(101, 0.1, 0.2, kaiser60)), 0.0));
        expect(eq(worstAsymmetry(bandstop(101, 0.1, 0.2, kaiser60)), 0.0));
    };

    "each design carries its gain to its own reference frequency"_test = [&] {
        for (const double gain : {1.0, 0.5, 1000.0}) {
            const std::vector<float> low  = lowpass(87, 0.125, kaiser60, gain);
            const std::vector<float> high = highpass(87, 0.125, kaiser60, gain);
            const std::vector<float> pass = bandpass(101, 0.1, 0.2, kaiser60, gain);
            const std::vector<float> stop = bandstop(101, 0.1, 0.2, kaiser60, gain);

            expect(approx(amplitudeAt(low, 0.0), gain, 1e-6 * gain)) << "low-pass at DC";
            expect(approx(std::abs(amplitudeAt(high, 0.5)), gain, 1e-6 * gain)) << "high-pass at Nyquist";
            expect(approx(amplitudeAt(pass, 0.15), gain, 1e-6 * gain)) << "band-pass at the arithmetic center";
            expect(approx(amplitudeAt(stop, 0.0), gain, 1e-6 * gain)) << "band-stop at DC";

            // the taps scale exactly linearly, gain being one multiply at the end
            const std::vector<float> unit  = lowpass(87, 0.125, kaiser60, 1.0);
            double                   worst = 0.0;
            for (std::size_t i = 0UZ; i < unit.size(); ++i) {
                worst = std::max(worst, std::abs(static_cast<double>(low[i]) - gain * static_cast<double>(unit[i])));
            }
            expect(lt(worst, 1e-6 * gain)) << "taps scale with gain, worst departure " << worst;
        }
    };

    "the high-pass is normalized at Nyquist and its sign follows M"_test = [&] {
        // A(pi) is the reference, so the alternating sum is (-1)^M times the gain: the check takes
        // the absolute value, since asserting the raw sum would fail on parity alone.
        for (const std::size_t n : {87UZ, 89UZ}) {
            const std::vector<float> taps        = highpass(n, 0.125, kaiser60);
            double                   alternating = 0.0;
            for (std::size_t i = 0UZ; i < taps.size(); ++i) {
                alternating += ((i % 2UZ) == 0UZ ? 1.0 : -1.0) * static_cast<double>(taps[i]);
            }
            const std::size_t mid = (n - 1UZ) / 2UZ;
            expect(approx(std::abs(alternating), 1.0, 1e-5));
            expect(that % ((alternating < 0.0) == ((mid % 2UZ) == 1UZ))) << "n = " << n << " sign follows the parity of M";
            expect(approx(amplitudeAt(taps, 0.5), 1.0, 1e-5)) << "and A(pi) itself is +gain by construction";
        }
    };

    "the complements are exact before normalization"_test = [&] {
        // w*2f*sinc(2fk) plus w*(delta(k) - 2f*sinc(2fk)) is w*delta(k) term by term, with nothing
        // left to round. This catches a sign error that the response checks would not.
        for (const std::size_t n : {31UZ, 87UZ, 175UZ}) {
            const std::vector<double> low  = lowpassKernel(n, 0.125, kaiser60);
            const std::vector<double> high = highpassKernel(n, 0.125, kaiser60);
            const std::vector<double> pass = bandpassKernel(n, 0.1, 0.2, kaiser60);
            const std::vector<double> stop = bandstopKernel(n, 0.1, 0.2, kaiser60);
            const std::size_t         mid  = (n - 1UZ) / 2UZ;

            for (std::size_t i = 0UZ; i < low.size(); ++i) {
                const double expected = (i == mid) ? low[mid] + high[mid] : 0.0;
                expect(eq(low[i] + high[i], expected)) << "low + high at " << i;
                expect(eq(pass[i] + stop[i], (i == mid) ? pass[mid] + stop[mid] : 0.0)) << "band-pass + band-stop at " << i;
            }
            expect(approx(low[mid] + high[mid], 1.0, 1e-15)) << "and the surviving center term is the window's own center value";
            expect(approx(pass[mid] + stop[mid], 1.0, 1e-15));
        }
    };

    "after normalization the complement residual is the two factors' departure from unity"_test = [&] {
        struct Pair {
            std::size_t taps;
            double      lowEdge;
            double      highEdge;
            double      residual;
        };
        // low-pass against high-pass, at one cutoff
        for (const Pair& p : std::array<Pair, 3>{{{87, 0.125, 0.0, 7.703e-05}, {175, 0.25, 0.0, 6.007e-05}, {31, 0.1, 0.0, 2.073e-04}}}) {
            const std::vector<float> low   = lowpass(p.taps, p.lowEdge, kaiser60);
            const std::vector<float> high  = highpass(p.taps, p.lowEdge, kaiser60);
            double                   worst = 0.0;
            for (std::size_t i = 0UZ; i < low.size(); ++i) {
                const double sum = static_cast<double>(low[i]) + static_cast<double>(high[i]) - (i == (p.taps - 1UZ) / 2UZ ? 1.0 : 0.0);
                worst            = std::max(worst, std::abs(sum));
            }
            expect(approx(worst, p.residual, 1e-6)) << "low/high at " << p.taps << " taps, cutoff " << p.lowEdge;
        }
        // band-pass against band-stop, over one pair of edges
        for (const Pair& p : std::array<Pair, 2>{{{175, 5000.0 / 48000.0, 9000.0 / 48000.0, 1.470e-04}, {101, 0.1, 0.2, 8.017e-05}}}) {
            const std::vector<float> pass  = bandpass(p.taps, p.lowEdge, p.highEdge, kaiser60);
            const std::vector<float> stop  = bandstop(p.taps, p.lowEdge, p.highEdge, kaiser60);
            double                   worst = 0.0;
            for (std::size_t i = 0UZ; i < pass.size(); ++i) {
                const double sum = static_cast<double>(pass[i]) + static_cast<double>(stop[i]) - (i == (p.taps - 1UZ) / 2UZ ? 1.0 : 0.0);
                worst            = std::max(worst, std::abs(sum));
            }
            expect(approx(worst, p.residual, 1e-6)) << "band pair at " << p.taps << " taps";
        }
    };

    "high-pass measured anchors"_test = [] {
        constexpr double fs = 48000.0;
        const FilterSpec spec{.sampleRate = fs, .cutoff = 6000.0, .transitionWidth = 2000.0, .attenuationDb = 60.0};

        expect(approx(gr::filter::fir::kaiserBeta(60.0), 5.65326, 1e-5));
        expect(eq(tapCountOf(spec), 87UZ));

        const std::vector<float> taps = designHighpass(spec);
        expect(eq(taps.size(), 87UZ));
        expect(approx(static_cast<double>(taps[43UZ]), 0.749970810986, 1e-7)) << "center tap";
        expect(approx(static_cast<double>(taps[0UZ]), -1.067147377416e-04, 1e-10));
        expect(approx(std::abs(amplitudeAt(taps, 0.0)), 1.914094e-04, 2e-6)) << "the design is not blind at DC, only 74 dB down";
        expect(approx(20.0 * std::log10(std::abs(amplitudeAt(taps, 6000.0 / fs))), -6.020141, 1e-4)) << "the stated cutoff is the -6 dB point";

        const BandLevels stopband = scanBand(taps, 0.0, 5000.0 / fs);
        const BandLevels passband = scanBand(taps, 7000.0 / fs, 0.5);
        expect(approx(stopband.peakDb(), -56.578, 0.01)) << "the estimate's own shortfall against a requested 60 dB, pinned rather than asserted away";
        expect(approx(passband.minMag, 0.998561420, 1e-6));
        expect(approx(passband.maxMag, 1.001043240, 1e-6));
        expect(approx(passband.rippleDb(), 0.0216, 1e-3));

        // the two ways of stating the same transition band
        constexpr TransitionBand edges = fromEdges(7000.0, 5000.0);
        const std::vector<float> same  = designHighpass({.sampleRate = fs, .cutoff = edges.cutoff, .transitionWidth = edges.width, .attenuationDb = 60.0});
        expect(that % identical(same, taps)) << "(passEdge, stopEdge) and (edge, transitionWidth) are the same design";
    };

    "band-pass measured anchors"_test = [] {
        constexpr double fs = 48000.0;
        const FilterSpec spec{.sampleRate = fs, .cutoff = 5000.0, .highCutoff = 9000.0, .transitionWidth = 1000.0, .attenuationDb = 60.0};

        expect(eq(tapCountOf(spec, 5UZ), 175UZ));
        const std::vector<float> taps = designBandpass(spec);
        expect(eq(taps.size(), 175UZ));
        expect(approx(static_cast<double>(taps[87UZ]), 0.166579470121, 1e-7)) << "center tap";
        expect(approx(amplitudeAt(taps, 7000.0 / fs), 1.0, 1e-6)) << "unit gain at the arithmetic center, not the geometric one";
        expect(approx(tapSum(taps), 2.808e-04, 1e-6)) << "and near-blind at DC";

        double alternating = 0.0;
        for (std::size_t i = 0UZ; i < taps.size(); ++i) {
            alternating += ((i % 2UZ) == 0UZ ? 1.0 : -1.0) * static_cast<double>(taps[i]);
        }
        expect(approx(alternating, -8.447e-06, 5e-7)) << "and at Nyquist";

        expect(approx(20.0 * std::log10(std::abs(amplitudeAt(taps, 5000.0 / fs))), -6.0244, 1e-3)) << "both stated edges are -6 dB points";
        expect(approx(20.0 * std::log10(std::abs(amplitudeAt(taps, 9000.0 / fs))), -6.0264, 1e-3));
        expect(approx(scanBand(taps, 0.0, 4500.0 / fs).peakDb(), -60.669, 0.01));
        expect(approx(scanBand(taps, 9500.0 / fs, 0.5).peakDb(), -59.539, 0.01));
    };

    "band-stop measured anchors"_test = [] {
        constexpr double fs = 48000.0;
        const FilterSpec spec{.sampleRate = fs, .cutoff = 5000.0, .highCutoff = 9000.0, .transitionWidth = 1000.0, .attenuationDb = 60.0};

        const std::vector<float> taps = designBandstop(spec);
        expect(eq(taps.size(), 175UZ));
        expect(approx(static_cast<double>(taps[87UZ]), 0.833567515119, 1e-7)) << "center tap";
        expect(approx(tapSum(taps), 1.0, 1e-6)) << "unit gain at DC";
        expect(approx(20.0 * std::log10(std::abs(amplitudeAt(taps, 7000.0 / fs))), -65.620, 0.02)) << "the notch center";
        expect(approx(scanBand(taps, 5500.0 / fs, 8500.0 / fs).peakDb(), -60.026, 0.01));

        const BandLevels lower  = scanBand(taps, 0.0, 4500.0 / fs);
        const BandLevels upper  = scanBand(taps, 9500.0 / fs, 0.5);
        const double     ripple = 20.0 * std::log10(std::max(lower.maxMag, upper.maxMag) / std::min(lower.minMag, upper.minMag));
        expect(approx(ripple, 0.0171, 1e-3));
    };

    "two calls with the same parameters return the same taps"_test = [&] {
        expect(that % identical(lowpass(87, 0.125, kaiser60, 2.0), lowpass(87, 0.125, kaiser60, 2.0)));
        expect(that % identical(highpass(87, 0.125, kaiser60), highpass(87, 0.125, kaiser60)));
        expect(that % identical(bandpass(101, 0.1, 0.2, kaiser60), bandpass(101, 0.1, 0.2, kaiser60)));
        expect(that % identical(bandstop(101, 0.1, 0.2, kaiser60), bandstop(101, 0.1, 0.2, kaiser60)));
    };
};

const boost::ut::suite<"complex FIR band design"> complexDesignTests = [] {
    using namespace boost::ut;
    using namespace gr::filter::fir::design;

    constexpr double fs       = 48000.0;
    const WindowSpec kaiser60 = kaiserFor(60.0);

    // the real low-pass prototype the complex pair is built from: same length, half the band width
    const auto prototype = [&](std::size_t n, double lowCutoff, double highCutoff) { return lowpass(n, 0.5 * (highCutoff - lowCutoff) / fs, kaiser60); };

    // |A_proto| read off the prototype's own half turn, folded by symmetry; the modulation
    // identity turns every complex band measurement into a prototype one
    const auto protoAmplitude = [](const std::vector<double>& amp, std::size_t grid, double relative) {
        std::size_t index = static_cast<std::size_t>(std::llround(std::abs(relative) * static_cast<double>(grid))) & (grid - 1UZ);
        if (index > grid / 2UZ) {
            index = grid - index;
        }
        return amp[index];
    };

    "complex designs are conjugate-symmetric to the bit"_test = [&] {
        for (const std::size_t n : {31UZ, 32UZ, 175UZ}) {
            for (const auto& taps : {complexBandpass(n, 2000.0 / fs, 6000.0 / fs, kaiser60), complexBandstop(n, 2000.0 / fs, 6000.0 / fs, kaiser60)}) {
                expect(eq(taps.size() % 2UZ, 1UZ));
                double worst = 0.0;
                for (std::size_t i = 0UZ; i < taps.size(); ++i) {
                    worst = std::max(worst, std::abs(std::complex<double>(taps[i]) - std::conj(std::complex<double>(taps[taps.size() - 1UZ - i]))));
                }
                expect(eq(worst, 0.0)) << "the second half is the first half's conjugate, copied";
            }
        }
        expect(eq(complexBandpass(1, 0.1, 0.2, kaiser60).size(), 5UZ)) << "a band design's floor is five taps";
    };

    "a complex band-pass centered on DC is the real low-pass"_test = [&] {
        // f1 = -f2 is not special-cased: the general formula gives a zero phase ramp and real taps.
        constexpr double         cutoff      = 0.2;
        const std::vector<float> real        = kaiserLowpass(101, cutoff, 60.0);
        const auto               complexTaps = complexBandpass(101, -cutoff, cutoff, kaiser60);

        double worstReal = 0.0;
        double worstImag = 0.0;
        for (std::size_t i = 0UZ; i < real.size(); ++i) {
            worstReal = std::max(worstReal, std::abs(static_cast<double>(complexTaps[i].real()) - static_cast<double>(real[i])));
            worstImag = std::max(worstImag, std::abs(static_cast<double>(complexTaps[i].imag())));
        }
        expect(lt(worstReal, 1e-7)) << "real part departs by " << worstReal;
        expect(lt(worstImag, 1e-12)) << "imaginary part is " << worstImag;
    };

    "complex band-pass measured anchors"_test = [&] {
        const FilterSpec spec{.sampleRate = fs, .cutoff = 2000.0, .highCutoff = 6000.0, .transitionWidth = 1000.0, .attenuationDb = 60.0};
        expect(eq(tapCountOf(spec, 5UZ), 175UZ));

        const auto taps = designComplexBandpass(spec);
        expect(eq(taps.size(), 175UZ));
        expect(approx(static_cast<double>(taps[87UZ].real()), 0.083290454007, 1e-7)) << "center tap";
        expect(eq(static_cast<double>(taps[87UZ].imag()), 0.0)) << "and the phase ramp is exactly zero there";

        expect(approx(std::abs(responseAt(taps, 4000.0 / fs)), 1.0, 1e-6)) << "unit gain at the band center";
        expect(approx(20.0 * std::log10(std::abs(responseAt(taps, 2000.0 / fs))), -6.0253, 1e-3)) << "both stated edges are -6 dB points";
        expect(approx(20.0 * std::log10(std::abs(responseAt(taps, 6000.0 / fs))), -6.0253, 1e-3));
        expect(approx(std::abs(responseAt(taps, 0.0)), 3.277782e-04, 2e-6)) << "and DC sits in the stopband";

        // the stopband is the prototype's, on both sides of the center by the modulation identity
        const std::vector<float> proto = prototype(175UZ, 2000.0, 6000.0);
        expect(approx(scanBand(proto, 2500.0 / fs, 0.5).peakDb(), -60.038, 0.01));

        double worst = 0.0;
        for (std::size_t i = 0UZ; i < kDesignGrid; i += 331UZ) {
            const double f = static_cast<double>(i) / static_cast<double>(kDesignGrid);
            worst          = std::max(worst, std::abs(std::abs(responseAt(taps, f)) - std::abs(amplitudeAt(proto, f - 4000.0 / fs))));
        }
        expect(lt(worst, 1e-6)) << "abs(H(f)) == abs(A_proto(f - fc)) to " << worst;
    };

    "complex band-stop measured anchors"_test = [&] {
        const FilterSpec spec{.sampleRate = fs, .cutoff = 2000.0, .highCutoff = 6000.0, .transitionWidth = 1000.0, .attenuationDb = 60.0};
        const auto       taps = designComplexBandstop(spec);
        expect(eq(taps.size(), 175UZ));
        expect(approx(static_cast<double>(taps[87UZ].real()), 0.916709545993, 1e-7)) << "center tap, one minus the band-pass's";

        // The notch is exact by construction rather than deep to a level: the band-pass has gain
        // exactly one at fc, so delta minus it cancels. What survives in the returned taps is the
        // float rounding of the storage, not anything the design did, so the depth is asserted as
        // "below any float signal path" rather than pinned to a figure.
        expect(approx(amplitudeAt(prototype(175UZ, 2000.0, 6000.0), 0.0), 1.0, 1e-6)) << "the prototype's DC gain is what the notch cancels against";
        expect(lt(20.0 * std::log10(std::max(std::abs(responseAt(taps, 4000.0 / fs)), 1e-300)), -150.0)) << "notch depth from float taps";
        expect(approx(std::abs(responseAt(taps, 0.0)), 1.000327778, 1e-5)) << "and DC sits in the passband";

        // abs(H) = abs(1 - A_proto(f - fc)) exactly, so both bands are read off the prototype
        const std::vector<float> proto = prototype(175UZ, 2000.0, 6000.0);
        std::vector<double>      amp;
        halfAmplitude(proto, amp);

        double notch = std::abs(1.0 - amplitudeAt(proto, 1500.0 / fs));
        for (std::size_t i = 0UZ; i <= static_cast<std::size_t>(std::floor(1500.0 / fs * static_cast<double>(kDesignGrid))); ++i) {
            notch = std::max(notch, std::abs(1.0 - protoAmplitude(amp, kDesignGrid, static_cast<double>(i) / static_cast<double>(kDesignGrid))));
        }
        expect(approx(20.0 * std::log10(notch), -56.959, 0.01)) << "worst level over the notch";

        double lowest  = std::abs(1.0 - amplitudeAt(proto, 2500.0 / fs));
        double highest = lowest;
        for (std::size_t i = static_cast<std::size_t>(std::ceil(2500.0 / fs * static_cast<double>(kDesignGrid))); i <= kDesignGrid / 2UZ; ++i) {
            const double v = std::abs(1.0 - protoAmplitude(amp, kDesignGrid, static_cast<double>(i) / static_cast<double>(kDesignGrid)));
            lowest         = std::min(lowest, v);
            highest        = std::max(highest, v);
        }
        expect(approx(20.0 * std::log10(highest / lowest), 0.0164, 1e-3)) << "passband ripple over the whole circle outside the notch";
    };

    "each complex design carries its gain, and two calls agree"_test = [&] {
        for (const double gain : {1.0, 0.5, 1000.0}) {
            const auto pass = complexBandpass(175, 2000.0 / fs, 6000.0 / fs, kaiser60, gain);
            const auto stop = complexBandstop(175, 2000.0 / fs, 6000.0 / fs, kaiser60, gain);
            expect(approx(std::abs(responseAt(pass, 4000.0 / fs)), gain, 1e-6 * gain)) << "band-pass at its center";
            expect(approx(std::abs(responseAt(stop, 0.0)) / gain, 1.000327778, 1e-5)) << "band-stop away from its notch";
        }
        expect(that % identical(complexBandpass(101, 0.1, 0.2, kaiser60), complexBandpass(101, 0.1, 0.2, kaiser60)));
        expect(that % identical(complexBandstop(101, 0.1, 0.2, kaiser60), complexBandstop(101, 0.1, 0.2, kaiser60)));
    };
};

const boost::ut::suite<"FIR pulse-shaping design"> pulseDesignTests = [] {
    using namespace boost::ut;
    using namespace gr::filter::fir::design;

    "the root-raised-cosine's two closed forms"_test = [] {
        struct Row {
            double alpha;
            double atZero;
            double atCorner;
        };
        // h(0) = 1 - a + 4a/pi, and h(T/(4a)) where the plain expression is 0/0. Two of these are
        // exact: at alpha 0.2 the argument pi/(4a) is 5pi/4, where sine and cosine are equal and the
        // bracket collapses to 2 cos(5pi/4); at alpha 1 it is pi/4 and the same collapse gives 1.
        for (const Row& row : std::array<Row, 5>{{{0.20, 1.054647908947033, -0.2}, {0.25, 1.068309886183791, -0.0642371557769986}, {0.35, 1.095633840657307, 0.26060346093755}, {0.50, 1.136619772367581, 0.57863246963255}, {1.00, 1.273239544735163, 1.0}}}) {
            expect(approx(rootRaisedCosineAt(0.0, row.alpha), row.atZero, 1e-12)) << "h(0) at alpha " << row.alpha;
            expect(approx(rootRaisedCosineAt(1.0 / (4.0 * row.alpha), row.alpha), row.atCorner, 1e-12)) << "h(T/4a) at alpha " << row.alpha;
        }
        expect(approx(rootRaisedCosineAt(1.0 / 0.8, 0.2), -0.2, 1e-15)) << "alpha 0.2 is exactly -0.2 at the singular point";
        expect(approx(rootRaisedCosineAt(0.25, 1.0), 1.0, 1e-15)) << "alpha 1 is exactly 1 there";

        // one expression across the range: at alpha = 0 it reduces to sinc(t) and stays finite
        expect(approx(rootRaisedCosineAt(0.0, 0.0), 1.0, 1e-15));
        for (int k = -32; k <= 32; ++k) {
            const double t = static_cast<double>(k) / 8.0;
            const double s = (t == 0.0) ? 1.0 : std::sin(std::numbers::pi * t) / (std::numbers::pi * t);
            expect(approx(rootRaisedCosineAt(t, 0.0), s, 1e-15)) << "alpha 0 at t = " << t;
        }
        expect(that % !std::isnan(rootRaisedCosineAt(0.5, 0.0)));

        // the guard band substitutes the limit rather than covering a discontinuity: just outside
        // it the plain expression is already within a few parts in 1e5 of the limit, and the
        // departure falls linearly with the offset from the singular point
        constexpr double alpha = 0.35;
        const double     limit = rootRaisedCosineAt(1.0 / (4.0 * alpha), alpha);
        for (const double offset : {2e-5, 2e-4, 2e-3}) {
            const double t = std::sqrt(1.0 - offset) / (4.0 * alpha);
            expect(approx(std::abs(rootRaisedCosineAt(t, alpha) - limit) / std::abs(limit), 2.1 * offset, 0.1 * offset)) << "continuity at an offset of " << offset;
        }
    };

    "the RDS case hits the singularity on integer arithmetic"_test = [] {
        // sps = 8, alpha = 1 puts 4*a*t at exactly 1 on k = +/-2, so the singular
        // case is reached in normal use.
        double unnormalized = 0.0;
        for (int k = -32; k <= 32; ++k) {
            unnormalized += rootRaisedCosineAt(static_cast<double>(k) / 8.0, 1.0);
        }
        expect(approx(unnormalized, 7.996057881613, 1e-11));

        const std::vector<float> taps = designRootRaisedCosine(65, 19000.0, 2375.0, 1.0);
        expect(eq(taps.size(), 65UZ));
        expect(eq(worstAsymmetry(taps), 0.0));
        expect(approx(static_cast<double>(taps[32UZ]), 0.159233407710, 1e-7)) << "center tap";
        expect(approx(static_cast<double>(taps[30UZ]), 0.125061625967, 1e-7)) << "the singular tap at k = -2";
        expect(approx(static_cast<double>(taps[34UZ]), 0.125061625967, 1e-7)) << "and at k = +2";
        expect(approx(tapSum(taps), 1.0, 1e-6)) << "normalized to DC gain, not to unit energy";
    };

    "the RRC is its own matched filter"_test = [] {
        // Convolved with itself it is a raised cosine, whose defining property is a zero at every
        // non-zero integer multiple of sps from the center. Truncation leaves a residue that depends
        // strongly on alpha, so the table is pinned: one blanket 1e-3 would pass alpha 0.35 and 1.0
        // and fail alpha 0.2.
        struct Row {
            double alpha;
            int    sps;
            double residue;
            double peak;
        };
        for (const Row& row : std::array<Row, 6>{{{0.20, 4, 1.743e-03, 0.248655515}, {0.20, 8, 1.797e-03, 0.124192346}, {0.35, 4, 3.918e-04, 0.250231763}, {0.35, 8, 4.522e-04, 0.125229537}, {1.00, 4, 4.915e-05, 0.250145006}, {1.00, 8, 5.956e-05, 0.125034273}}}) {
            const std::size_t        n    = static_cast<std::size_t>(16 * row.sps + 1);
            const std::vector<float> taps = rootRaisedCosine(n, static_cast<double>(row.sps), row.alpha);

            std::vector<double> self(2UZ * taps.size() - 1UZ, 0.0);
            for (std::size_t i = 0UZ; i < taps.size(); ++i) {
                for (std::size_t j = 0UZ; j < taps.size(); ++j) {
                    self[i + j] += static_cast<double>(taps[i]) * static_cast<double>(taps[j]);
                }
            }
            const std::size_t center = self.size() / 2UZ;
            expect(approx(self[center], row.peak, 1e-6)) << "self-convolution peak at alpha " << row.alpha << " sps " << row.sps;

            double worst = 0.0;
            for (int j = 1; j <= 7; ++j) {
                worst = std::max(worst, std::abs(self[center + static_cast<std::size_t>(j * row.sps)]));
            }
            expect(approx(worst / self[center], row.residue, 5e-3 * row.residue)) << "worst crossing residue at alpha " << row.alpha << " sps " << row.sps;
        }
    };

    "the Gaussian pulse sits on a centered grid"_test = [] {
        expect(approx(gaussianSigma(0.35), 0.378586234, 1e-9));
        expect(approx(gaussianSigma(0.30), 0.441683939, 1e-9));

        // N = 9, sps = 4, bt = 0.35: symmetric to the bit, sums to one, group delay exactly 4. A
        // half-integer grid drops the center tap, is asymmetric by 9.4e-2 against a peak of 0.264,
        // and lands the delay half a sample short of the integer a symbol chain assumes.
        constexpr std::array<double, 9> want{0.008067176918, 0.037115256631, 0.110408650330, 0.212360373672, 0.264097084898, 0.212360373672, 0.110408650330, 0.037115256631, 0.008067176918};
        const std::vector<float>        taps = gaussianPulse(9, 4.0, 0.35);

        expect(eq(taps.size(), 9UZ));
        for (std::size_t i = 0UZ; i < want.size(); ++i) {
            expect(approx(static_cast<double>(taps[i]), want[i], 1e-7)) << "tap " << i << " to float's own resolution";
        }
        expect(eq(worstAsymmetry(taps), 0.0)) << "symmetry fixes the group delay at exactly (N-1)/2, with no phase fit needed";
        expect(approx(tapSum(taps), 1.0, 1e-7));

        for (const std::size_t n : {9UZ, 17UZ, 33UZ}) {
            expect(eq(worstAsymmetry(gaussianPulse(n, 4.0, 0.35)), 0.0)) << "at " << n << " taps";
            expect(approx(tapSum(gaussianPulse(n, 8.0, 0.30, 1000.0)), 1000.0, 1e-3));
        }
    };

    "pulse designs round up, floor and repeat"_test = [] {
        expect(eq(rootRaisedCosine(32, 4.0, 0.35).size(), 33UZ));
        expect(eq(gaussianPulse(32, 4.0, 0.35).size(), 33UZ));
        expect(eq(rootRaisedCosine(0, 4.0, 0.35).size(), 3UZ));
        expect(eq(gaussianPulse(1, 4.0, 0.35).size(), 3UZ));
        expect(that % identical(rootRaisedCosine(65, 8.0, 0.35, 2.0), rootRaisedCosine(65, 8.0, 0.35, 2.0)));
        expect(that % identical(gaussianPulse(33, 8.0, 0.30), gaussianPulse(33, 8.0, 0.30)));
    };
};

const boost::ut::suite<"Hilbert transformer design"> hilbertDesignTests = [] {
    using namespace boost::ut;
    using namespace gr::filter::fir::design;
    using gr::algorithm::window::Type;

    const std::array<WindowSpec, 3> windows{WindowSpec{Type::Hamming, std::numeric_limits<double>::quiet_NaN()}, WindowSpec{Type::BlackmanHarris, std::numeric_limits<double>::quiet_NaN()}, WindowSpec{Type::Kaiser, 7.0}};

    "half a Hilbert transformer's taps are exactly zero"_test = [&] {
        for (const int n : {11, 31, 63, 127}) {
            for (const WindowSpec& window : windows) {
                const std::vector<float> taps = hilbert(static_cast<std::size_t>(n), window);
                const int                mid  = (n - 1) / 2;

                expect(eq(static_cast<double>(taps[static_cast<std::size_t>(mid)]), 0.0)) << "the center tap";
                for (int i = 0; i < n; ++i) {
                    if (((i - mid) % 2) == 0) {
                        expect(eq(static_cast<double>(taps[static_cast<std::size_t>(i)]), 0.0)) << "every even offset, at n = " << n;
                    }
                }
                double worst = 0.0;
                for (std::size_t i = 0UZ; i < taps.size(); ++i) {
                    worst = std::max(worst, std::abs(static_cast<double>(taps[i]) + static_cast<double>(taps[taps.size() - 1UZ - i])));
                }
                expect(eq(worst, 0.0)) << "antisymmetric to the bit";
            }
        }
    };

    "the response is structurally zero at DC and at Nyquist"_test = [&] {
        for (const int n : {31, 63}) {
            const std::vector<float> taps = hilbert(static_cast<std::size_t>(n), windows[0]);
            expect(lt(std::abs(amplitudeAtOdd(taps, 0.0)), 1e-12)) << "type III: H(0) = 0 whatever the window";
            expect(lt(std::abs(amplitudeAtOdd(taps, 0.5)), 1e-12)) << "and H(fs/2) = 0";
        }
    };

    "the analytic 2/(pi k) scaling is unit gain without renormalization"_test = [&] {
        // The alternative convention divides by an accumulated normalizer that works out to abs(H)
        // at exactly fs/4. These are what the un-renormalized design gives there, so they are how
        // far the two conventions differ: 1.3% at eleven taps, under 0.16% from thirty-one up.
        struct Row {
            std::size_t taps;
            double      hamming;
            double      blackmanHarris;
            double      kaiser7;
        };
        for (const Row& row : std::array<Row, 4>{{{11, 1.012577880, 0.967036063, 1.001004939}, {31, 0.996902114, 0.999996931, 1.000114019}, {63, 0.998472051, 0.999998654, 0.999974949}, {127, 0.999236854, 0.999999393, 0.999964121}}}) {
            const std::array<double, 3> want{row.hamming, row.blackmanHarris, row.kaiser7};
            for (std::size_t w = 0UZ; w < windows.size(); ++w) {
                const double measured = std::abs(amplitudeAtOdd(hilbert(row.taps, windows[w]), 0.25));
                expect(approx(measured, want[w], 5e-6)) << "abs(H(fs/4)) at " << row.taps << " taps, window " << w;
            }
        }
    };

    "the usable band is whatever is left between the two structural zeros"_test = [&] {
        // At eleven taps there is no usable band: 0.51 to 1.01 across the span. The table records
        // that rather than pinning a passband the design does not have.
        struct Row {
            std::size_t taps;
            double      lo[3];
            double      hi[3];
        };
        for (const Row& row : std::array<Row, 4>{{{11, {0.514, 0.348, 0.442}, {1.013, 0.967, 1.001}}, //
                 {31, {0.978, 0.831, 0.938}, {1.006, 1.000, 1.001}},                                  //
                 {63, {0.997, 0.998, 0.9996}, {1.003, 1.000, 1.0003}},                                //
                 {127, {0.998, 0.99999, 0.9999}, {1.002, 1.00001, 1.0001}}}}) {
            for (std::size_t w = 0UZ; w < windows.size(); ++w) {
                const BandLevels band = scanHilbert(hilbert(row.taps, windows[w]), 0.05);
                expect(approx(band.minMag, row.lo[w], 1e-3)) << "band floor at " << row.taps << " taps, window " << w;
                expect(approx(band.maxMag, row.hi[w], 1e-3)) << "band ceiling at " << row.taps << " taps, window " << w;
            }
        }
        expect(lt(scanHilbert(hilbert(11, windows[0]), 0.05).minMag, 0.6)) << "eleven taps is not usable across 0.05 .. 0.45";
        expect(gt(scanHilbert(hilbert(127, windows[0]), 0.05).minMag, 0.99)) << "127 taps is";
    };

    "halfResponseOdd is the sine sum halfResponse is not"_test = [&] {
        const std::vector<float> taps = hilbert(63, windows[2]);
        std::vector<double>      mag;
        halfResponseOdd(taps, mag);

        expect(eq(mag.size(), kDesignGrid / 2UZ + 1UZ));
        double worst = 0.0;
        for (std::size_t i = 0UZ; i <= kDesignGrid / 2UZ; i += 337UZ) {
            const double f = static_cast<double>(i) / static_cast<double>(kDesignGrid);
            worst          = std::max(worst, std::abs(std::abs(amplitudeAtOdd(taps, f)) - mag[i]));
        }
        expect(lt(worst, 1e-12)) << "the table-stepped sine sum and the direct one agree to " << worst;
        expect(lt(mag.front(), 1e-12));
        expect(lt(mag.back(), 1e-12));
    };

    "the length that wastes no taps is N mod 4 == 3"_test = [&] {
        expect(eq(hilbertLength(61UZ), 63UZ));
        expect(eq(hilbertLength(62UZ), 63UZ));
        expect(eq(hilbertLength(63UZ), 63UZ));
        expect(eq(hilbertLength(64UZ), 67UZ));
        expect(eq(hilbertLength(65UZ), 67UZ));
        expect(eq(hilbertLength(1UZ), 3UZ));

        // at M even the two outermost taps are among the zeros: two taps of arithmetic thrown away
        for (const std::size_t n : {61UZ, 65UZ}) {
            const std::vector<float> wasteful = hilbert(n, windows[0]);
            expect(eq(static_cast<double>(wasteful.front()), 0.0)) << n << " spends its outermost pair on zeros";
            expect(that % (static_cast<double>(hilbert(hilbertLength(n), windows[0]).front()) != 0.0)) << hilbertLength(n) << " does not";
        }
    };

    "gain scales, and two calls agree"_test = [&] {
        for (const double gain : {1.0, 0.5, 1000.0}) {
            const std::vector<float> taps = hilbert(63, windows[2], gain);
            expect(approx(std::abs(amplitudeAtOdd(taps, 0.25)) / gain, 0.999974949, 1e-5));
        }
        expect(that % identical(hilbert(63, windows[0]), hilbert(63, windows[0])));
    };
};

const boost::ut::suite<"manchesterMatchedFilter"> manchesterMatchedFilterTests = [] {
    using namespace boost::ut;
    using namespace gr::filter::fir::design;

    "the difference of the pulse against its half-symbol shift, DC-free"_test = [] {
        constexpr std::size_t kTaps = 151UZ;
        constexpr double      kRate = 38000.0; // RDS's biphase grid: 16 samples per 2375-baud symbol
        constexpr double      kBaud = 2375.0;

        const std::vector<float> pulse = designRootRaisedCosine(kTaps, kRate, kBaud, 1.0);
        const std::vector<float> taps  = manchesterMatchedFilter(kTaps, kRate, kBaud, 1.0);

        expect(eq(taps.size(), pulse.size() - 8UZ)) << "eight samples is the half symbol at sixteen per";
        double worst = 0.0;
        double sum   = 0.0;
        for (std::size_t n = 0UZ; n < taps.size(); ++n) {
            worst = std::max(worst, static_cast<double>(std::abs(taps[n] - (pulse[n] - pulse[n + 8UZ]))));
            sum += static_cast<double>(taps[n]);
        }
        expect(that % worst == 0.0) << "the definition, tap for tap";
        expect(lt(std::abs(sum), 1e-6)) << "the two copies cancel: no DC passes";
    };

    "the filter reads a biphase symbol at full strength and the un-split pulse at none"_test = [] {
        constexpr std::size_t    kSps  = 16UZ;
        constexpr double         kBaud = 2375.0;
        constexpr double         kRate = static_cast<double>(kSps) * kBaud;
        const std::vector<float> taps  = manchesterMatchedFilter(12UZ * kSps + 1UZ, kRate, kBaud, 1.0);
        const std::vector<float> pulse = designRootRaisedCosine(12UZ * kSps + 1UZ, kRate, kBaud, 1.0);

        // The biphase symbol IS the filter's generating difference, so the aligned correlation is
        // the filter's own energy, 2(R(0) - R(T/2)). The un-split pulse does best a quarter symbol
        // off, read by both halves at once as R(T/4) - R(3T/4) — at alpha 1 that is 0.68 R(0)
        // against the energy's 1.0 R(0), a discrimination of 1.47 the bound below pins.
        double energy = 0.0;
        for (const float tap : taps) {
            energy += static_cast<double>(tap) * static_cast<double>(tap);
        }
        double onPulse = 0.0;
        for (int lag = -16; lag <= 16; ++lag) {
            double dot = 0.0;
            for (std::size_t n = 0UZ; n < taps.size(); ++n) {
                const auto at = static_cast<std::ptrdiff_t>(n) + lag;
                if (at >= 0 && at < static_cast<std::ptrdiff_t>(pulse.size())) {
                    dot += static_cast<double>(taps[n]) * static_cast<double>(pulse[static_cast<std::size_t>(at)]);
                }
            }
            onPulse = std::max(onPulse, std::abs(dot));
        }
        expect(gt(energy, 1.4 * onPulse)) << "the split shape outreads the un-split one at its best alignment";
    };

    "refusals fire by name"_test = [] {
        expect(throws([] { std::ignore = manchesterMatchedFilter(151, 38000.0, 2400.0, 1.0); })) << "a fractional half symbol is a resampler's job";
        expect(throws([] { std::ignore = manchesterMatchedFilter(8, 38000.0, 2375.0, 1.0); })) << "too short to span the shift";
    };
};

int main() { /* tests are automatically registered and run */ }
