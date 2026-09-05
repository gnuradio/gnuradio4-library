#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/HalfbandCascade.hpp>
#include <gnuradio-4.0/algorithm/filter/PolyphaseResampler.hpp>

namespace {

using namespace gr::filter;

/// The charge the design tables are quoted under: half the length plus the center. It is the true
/// live count only where `(N-1)/2` is even; at N of 19 and 23 a halfband keeps 11 and 13 taps, not
/// 10 and 12. Both are carried here and the difference is asserted rather than averaged over.
[[nodiscard]] double tabulatedMacsPerInput(const StagedDecimatorDesign& design) {
    double macs = 0.0;
    for (const DecimatorStage& s : design.stages) {
        const std::size_t charge = s.halfband ? s.taps.size() / 2UZ + 1UZ : s.taps.size();
        macs += static_cast<double>(charge) / static_cast<double>(s.stride * s.decimation);
    }
    return macs;
}

[[nodiscard]] StagedDecimatorDesign singleStage(std::size_t decimation, double width) {
    const std::size_t ladder[] = {decimation};
    return designDecimatorLadder(std::span<const std::size_t>(ladder), decimation, width);
}

/// Where a frequency lands after a decimation by @p d, in cycles per sample of the slower rate.
[[nodiscard]] double foldedBy(double f, std::size_t d) {
    const double scaled = f * static_cast<double>(d);
    return std::abs(scaled - std::round(scaled));
}

/// The cascade's gain at input frequency @p f, following the alias down the ladder: every stage is
/// read at the frequency the stages above it have folded @p f to.
[[nodiscard]] double cascadeGain(const StagedDecimatorDesign& design, double f) {
    double gain = 1.0;
    double at   = f;
    for (const DecimatorStage& s : design.stages) {
        gain *= std::abs(fir::design::amplitudeAt(s.taps, at));
        at = foldedBy(at, s.decimation);
    }
    return gain;
}

/// @brief The worst gain over the frequencies whose image lands in the kept band, and the passband ripple.
struct FoldReading {
    double worstFoldDb = -1000.0;
    double rippleDb    = 0.0;
};

[[nodiscard]] FoldReading readFold(const StagedDecimatorDesign& design, double width, int grid) {
    const double half = 0.5 * width / static_cast<double>(design.decimation); // W/2 at the input rate

    FoldReading out;
    double      passMax = -1000.0;
    double      passMin = 1000.0;
    for (int i = 0; i <= grid; ++i) {
        const double f  = 0.5 * static_cast<double>(i) / static_cast<double>(grid);
        const double db = 20.0 * std::log10(std::max(cascadeGain(design, f), 1.0e-300));
        if (f <= half) {
            passMax = std::max(passMax, db);
            passMin = std::min(passMin, db);
        } else if (foldedBy(f, design.decimation) <= half * static_cast<double>(design.decimation)) {
            out.worstFoldDb = std::max(out.worstFoldDb, db);
        }
    }
    out.rippleDb = passMax - passMin;
    return out;
}

/// The tables are quoted to two decimal places, so a value ending in `.xx5` sits exactly on the
/// rounding boundary, 15.375 against a tabulated 15.38, and half a quantum is not enough slack.
constexpr double kTabulated = 0.0051;

/// One row of the design tables: MAC per input sample under the tabulated charge, at 85 dB and
/// 0.05 dB of total ripple.
struct Row {
    std::size_t decimation;
    std::size_t singleTaps;
    double      single;
    double      oneOddStage;
    double      oddFactored;
};

constexpr Row kWidth90[] = {
    {8UZ, 431UZ, 53.88, 15.38, 15.38},   //
    {16UZ, 863UZ, 53.94, 12.69, 12.69},  //
    {25UZ, 1347UZ, 53.88, 53.88, 17.84}, //
    {32UZ, 1717UZ, 53.66, 9.84, 9.84},   //
    {50UZ, 2681UZ, 53.62, 30.44, 12.42}, //
    {64UZ, 3431UZ, 53.61, 8.42, 8.42},   //
    {100UZ, 5359UZ, 53.59, 18.72, 9.71}, //
    {128UZ, 6859UZ, 53.59, 7.71, 7.71},  //
    {512UZ, 27427UZ, 53.57, 7.18, 7.18}, //
};

constexpr Row kWidth80[] = {
    {8UZ, 223UZ, 27.88, 11.88, 11.88},   //
    {16UZ, 447UZ, 27.94, 10.94, 10.94},  //
    {25UZ, 697UZ, 27.88, 27.88, 12.64},  //
    {32UZ, 891UZ, 27.84, 8.97, 8.97},    //
    {50UZ, 1393UZ, 27.86, 17.44, 9.82},  //
    {64UZ, 1779UZ, 27.80, 7.98, 7.98},   //
    {100UZ, 2769UZ, 27.69, 12.22, 8.41}, //
    {128UZ, 3433UZ, 26.82, 7.49, 7.49},  //
    {512UZ, 13729UZ, 26.81, 7.12, 7.12}, //
};

/// The rows above `D = 50` are a length search over thousands of candidates apiece, and the
/// single-stage arm of `D = 512` at `width = 0.90` searches out to 13729 taps, so it is gated.
[[nodiscard]] bool longRun() { return std::getenv("ENABLE_LONG_TESTS") != nullptr; }

} // namespace

const boost::ut::suite<"StagedDecimator planner"> _stagedDecimator = [] {
    using namespace boost::ut;

    "the ladder is halvings first, then the odd factors largest first"_test = [] {
        const auto expect_ladder = [](std::size_t d, bool factorOdd, const std::vector<std::size_t>& want) { expect(that % (stagedDecimatorLadder(d, factorOdd) == want)) << "D = " << d << (factorOdd ? " factored" : " one odd stage"); };

        expect_ladder(1UZ, true, {});
        expect_ladder(1UZ, false, {});
        expect_ladder(2UZ, true, {2UZ});
        expect_ladder(3UZ, true, {3UZ});
        expect_ladder(4UZ, true, {2UZ, 2UZ});
        expect_ladder(6UZ, true, {2UZ, 3UZ});
        expect_ladder(8UZ, true, {2UZ, 2UZ, 2UZ});
        expect_ladder(12UZ, true, {2UZ, 2UZ, 3UZ});
        expect_ladder(16UZ, true, {2UZ, 2UZ, 2UZ, 2UZ});
        expect_ladder(24UZ, true, {2UZ, 2UZ, 2UZ, 3UZ});
        expect_ladder(25UZ, true, {5UZ, 5UZ});
        expect_ladder(25UZ, false, {25UZ});
        expect_ladder(27UZ, true, {3UZ, 3UZ, 3UZ});
        expect_ladder(27UZ, false, {27UZ});
        expect_ladder(32UZ, true, {2UZ, 2UZ, 2UZ, 2UZ, 2UZ});
        expect_ladder(45UZ, true, {5UZ, 3UZ, 3UZ});
        expect_ladder(49UZ, true, {7UZ, 7UZ});
        expect_ladder(50UZ, true, {2UZ, 5UZ, 5UZ});
        expect_ladder(50UZ, false, {2UZ, 25UZ});
        expect_ladder(64UZ, true, {2UZ, 2UZ, 2UZ, 2UZ, 2UZ, 2UZ});
        expect_ladder(100UZ, true, {2UZ, 2UZ, 5UZ, 5UZ});
        expect_ladder(100UZ, false, {2UZ, 2UZ, 25UZ});
        expect_ladder(128UZ, true, {2UZ, 2UZ, 2UZ, 2UZ, 2UZ, 2UZ, 2UZ});
        expect_ladder(512UZ, true, std::vector<std::size_t>(9UZ, 2UZ));

        // An odd prime is one stage either way.
        for (const std::size_t d : {7UZ, 11UZ, 13UZ}) {
            expect(that % (stagedDecimatorLadder(d, true) == std::vector<std::size_t>{d}));
            expect(that % (stagedDecimatorLadder(d, false) == std::vector<std::size_t>{d}));
        }
    };

    "a decimation of one designs nothing at all"_test = [] {
        const StagedDecimatorDesign d = designStagedDecimator(1UZ, 0.90);
        expect(eq(d.stages.size(), 0UZ));
        expect(eq(d.groupDelaySamples, 0ULL)) << "nothing folds at a decimation of one, so nothing is filtered";
        expect(eq(d.macsPerInput, 0.0));
        expect(d.ok);
    };

    "a decimation of two is the cascade with one element"_test = [] {
        const StagedDecimatorDesign d = designStagedDecimator(2UZ, 0.90);
        expect(eq(d.stages.size(), 1UZ));
        expect(d.stages[0UZ].halfband);
        expect(eq(d.stages[0UZ].decimation, 2UZ));
        expect(eq(d.stages[0UZ].stride, 1UZ));
        expect(eq(d.stages[0UZ].taps.size(), 117UZ));
        expect(eq(d.groupDelaySamples, 58ULL));
        expect(d.ok);
    };

    "the D = 64 ladder, stage by stage"_test = [] {
        const StagedDecimatorDesign d = designStagedDecimator(64UZ, 0.90);
        expect(eq(d.stages.size(), 6UZ)) << "six halvings and no odd tail";

        constexpr std::size_t kTaps[]     = {13UZ, 13UZ, 19UZ, 19UZ, 23UZ, 117UZ};
        constexpr std::size_t kLive[]     = {7UZ, 7UZ, 11UZ, 11UZ, 13UZ, 59UZ};
        constexpr double      kPassEdge[] = {0.007031250, 0.014062500, 0.028125000, 0.056250000, 0.112500000, 0.225000000};
        constexpr double      kStopEdge[] = {0.492968750, 0.485937500, 0.471875000, 0.443750000, 0.387500000, 0.275000000};
        constexpr double      kStopband[] = {-85.77, -85.77, -97.38, -91.13, -87.11, -85.39};

        for (std::size_t i = 0UZ; i < d.stages.size(); ++i) {
            const DecimatorStage& s = d.stages[i];
            expect(eq(s.decimation, 2UZ));
            expect(eq(s.stride, 1UZ << i));
            expect(eq(s.taps.size(), kTaps[i])) << "stage " << i << " length";
            expect(eq(s.liveTaps(), kLive[i])) << "stage " << i << " live taps";
            expect(approx(s.passEdge, kPassEdge[i], 1e-9)) << "stage " << i << " pass edge";
            expect(approx(s.stopEdge, kStopEdge[i], 1e-9)) << "stage " << i << " stop edge";
            expect(approx(s.stopbandDb, kStopband[i], 0.005)) << "stage " << i << " stopband";
            expect(s.rippleDb <= 0.001) << "stage " << i << " ripple";
            expect(s.ok);
        }

        // The last stage is nine times the length of the first and costs a quarter as much, because
        // it runs at a sixty-fourth of the rate.
        expect(approx(d.stages[0UZ].macsPerInput(), 3.500, 1e-9));
        expect(approx(d.stages[5UZ].macsPerInput(), 0.921875, 1e-9));

        expect(approx(d.macsPerInput, 8.640625, 1e-9));
        expect(approx(tabulatedMacsPerInput(d), 8.421875, 1e-9));
        expect(approx(d.rippleSumDb, 0.0023, 0.0005));
        expect(approx(d.worstStopbandDb, -85.39, 0.005)) << "the least attenuated stage, which is the last";
        expect(eq(d.groupDelaySamples, 2158ULL)) << "6*1 + 6*2 + 9*4 + 9*8 + 11*16 + 58*32";
    };

    "the group delay is the stated sum over the stages"_test = [] {
        for (const std::size_t decimation : {1UZ, 2UZ, 3UZ, 8UZ, 16UZ, 25UZ, 32UZ, 64UZ}) {
            const StagedDecimatorDesign d = designStagedDecimator(decimation, 0.90);

            std::uint64_t want = 0ULL;
            for (const DecimatorStage& s : d.stages) {
                want += ((s.taps.size() - 1UZ) / 2UZ) * s.stride;
            }
            expect(eq(d.groupDelaySamples, want)) << "D = " << decimation;
        }
        expect(eq(designStagedDecimator(64UZ, 0.90).groupDelaySamples, 2158ULL));
    };

    "a halving stage's even-offset taps are zeros and its live count says so"_test = [] {
        const StagedDecimatorDesign d = designStagedDecimator(64UZ, 0.90);

        std::vector<std::vector<float>> stages;
        for (const DecimatorStage& s : d.stages) {
            expect(s.halfband);

            const auto  n         = static_cast<int>(s.taps.size());
            const int   mid       = (n - 1) / 2;
            double      peak      = 0.0;
            double      worstZero = 0.0;
            std::size_t live      = 0UZ;
            for (int i = 0; i < n; ++i) {
                const double v      = std::abs(static_cast<double>(s.taps[static_cast<std::size_t>(i)]));
                const int    offset = i - mid;
                peak                = std::max(peak, v);
                if (offset != 0 && (offset % 2) == 0) {
                    worstZero = std::max(worstZero, v);
                } else {
                    ++live;
                }
            }
            // Not zero to the bit: the tap is `sin` of a rounded multiple of pi, which lets the
            // kernel skip it by index rather than by testing against zero.
            expect(20.0 * std::log10(std::max(worstZero, 1e-300) / peak) < -300.0) << n << " taps";
            expect(eq(s.liveTaps(), live)) << n << " taps: the count the kernel will actually pay";
            stages.push_back(s.taps);
        }

        // The kernel agrees, which is why the count is taken this way rather than as N/2 + 1.
        HalfbandCascade cascade{std::span<const std::vector<float>>(stages)};
        for (std::size_t i = 0UZ; i < d.stages.size(); ++i) {
            expect(eq(cascade.liveTaps(i), d.stages[i].liveTaps())) << "stage " << i;
        }
    };

    "the MAC tables, staged"_test = [] {
        const auto check = [](const Row& row, double width) {
            const StagedDecimatorDesign one      = designStagedDecimator(row.decimation, width, 85.0, 0.05, false);
            const StagedDecimatorDesign factored = designStagedDecimator(row.decimation, width, 85.0, 0.05, true);

            expect(approx(tabulatedMacsPerInput(one), row.oneOddStage, kTabulated)) << "D = " << row.decimation << " at " << width << ", one odd stage";
            expect(approx(tabulatedMacsPerInput(factored), row.oddFactored, kTabulated)) << "D = " << row.decimation << " at " << width << ", odd factored";
            expect(factored.macsPerInput <= one.macsPerInput + 1e-9) << "factoring never costs more";
        };

        for (const Row& row : kWidth90) {
            check(row, 0.90);
        }
        for (const Row& row : kWidth80) {
            check(row, 0.80);
        }

        // At D = 25 one polyphase stage reclaims exactly nothing, because the ladder degenerates
        // to the filter it is compared with.
        expect(approx(tabulatedMacsPerInput(designStagedDecimator(25UZ, 0.90, 85.0, 0.05, false)), 53.88, kTabulated));
        expect(approx(tabulatedMacsPerInput(designStagedDecimator(25UZ, 0.90, 85.0, 0.05, true)), 17.84, kTabulated));
    };

    "the MAC tables, against the single-stage design"_test = [] {
        const auto check = [](const Row& row, double width) {
            const StagedDecimatorDesign single = singleStage(row.decimation, width);
            expect(eq(single.stages.size(), 1UZ));
            expect(eq(single.stages[0UZ].taps.size(), row.singleTaps)) << "D = " << row.decimation << " at " << width << ", single-stage length";
            expect(approx(single.macsPerInput, row.single, kTabulated)) << "D = " << row.decimation << " at " << width << ", single-stage cost";

            const double gain = single.macsPerInput / tabulatedMacsPerInput(designStagedDecimator(row.decimation, width, 85.0, 0.05, true));
            expect(gain >= 0.999) << "D = " << row.decimation << ": staging never costs more than one stage";
        };

        for (std::size_t i = 0UZ; i < 5UZ; ++i) { // D up to 50; the rest is a search over thousands of candidates
            check(kWidth90[i], 0.90);
            check(kWidth80[i], 0.80);
        }
        if (!longRun()) {
            expect(true) << "set ENABLE_LONG_TESTS for the D = 64 to 512 single-stage rows";
            return;
        }
        for (std::size_t i = 5UZ; i < std::size(kWidth90); ++i) {
            check(kWidth90[i], 0.90);
            check(kWidth80[i], 0.80);
        }
    };

    "the fold contract holds for the ladder and for the single stage alike"_test = [] {
        constexpr double kAttenuationDb = 85.0;
        constexpr double kRippleDb      = 0.05;

        for (const std::size_t decimation : {8UZ, 16UZ, 25UZ}) {
            const StagedDecimatorDesign ladder = designStagedDecimator(decimation, 0.90);
            const StagedDecimatorDesign single = singleStage(decimation, 0.90);

            for (const StagedDecimatorDesign& d : {ladder, single}) {
                const FoldReading r = readFold(d, 0.90, 40000);
                expect(r.worstFoldDb <= -(kAttenuationDb - 1.0)) << "D = " << decimation << ": a frequency folding into the kept band";
                expect(r.rippleDb <= kRippleDb) << "D = " << decimation << ": in-band ripple";
            }
        }
    };

    "degenerate parameters"_test = [] {
        expect(throws<std::invalid_argument>([] { std::ignore = designStagedDecimator(0UZ, 0.90); }));
        expect(throws<std::invalid_argument>([] { std::ignore = stagedDecimatorLadder(0UZ); }));
        expect(throws<std::invalid_argument>([] { std::ignore = designStagedDecimator(8UZ, 0.0); }));
        expect(throws<std::invalid_argument>([] { std::ignore = designStagedDecimator(8UZ, 1.0); }));
        expect(throws<std::invalid_argument>([] { std::ignore = designStagedDecimator(8UZ, -0.5); }));

        // An odd prime above the bound is placed anyway and reported: refusing a legal decimation is
        // worse than a long last stage.
        const StagedDecimatorDesign oversized = designStagedDecimator(29UZ, 0.90, 85.0, 0.05, true, 25UZ);
        expect(eq(oversized.stages.size(), 1UZ));
        expect(eq(oversized.oversizedOddFactor, 29UZ));
        expect(oversized.ok);

        expect(eq(designStagedDecimator(25UZ, 0.90, 85.0, 0.05, true, 25UZ).oversizedOddFactor, 0UZ));
        expect(eq(designStagedDecimator(25UZ, 0.90, 85.0, 0.05, false, 25UZ).oversizedOddFactor, 0UZ));
    };

    "a ladder already built is served from the memo"_test = [] {
        const StagedDecimatorDesign first  = designStagedDecimator(16UZ, 0.90);
        const StagedDecimatorDesign second = designStagedDecimator(16UZ, 0.90);

        expect(eq(first.stages.size(), second.stages.size()));
        for (std::size_t i = 0UZ; i < first.stages.size(); ++i) {
            expect(that % (first.stages[i].taps == second.stages[i].taps)) << "stage " << i << " is the same tap set, to the bit";
        }
        expect(eq(first.groupDelaySamples, second.groupDelaySamples));
    };
};

int main() { /* tests are automatically registered and run */ }
