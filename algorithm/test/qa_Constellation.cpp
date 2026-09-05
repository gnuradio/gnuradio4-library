#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <numbers>
#include <print>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <vector>

#include <gnuradio-4.0/algorithm/digital/Constellation.hpp>

namespace {

using gr::digital::Constellation;
using gr::digital::DecisionStrategy;
using gr::digital::grayDecode;
using gr::digital::grayEncode;
using gr::digital::Normalization;
using gr::digital::SoftAlgorithm;

using C  = Constellation<double>;
using CD = std::complex<double>;

constexpr double kPi = std::numbers::pi;

[[nodiscard]] std::vector<C> theSixConstellations(Normalization mode = Normalization::Power) {
    std::vector<C> set;
    set.push_back(C::bpsk(mode));
    set.push_back(C::qpsk(mode));
    set.push_back(C::psk8(mode));
    set.push_back(C::psk(16UZ, 0.0, 0U, mode));
    set.push_back(C::qam(16UZ, mode));
    set.push_back(C::qam(64UZ, mode));
    return set;
}

constexpr const char* kSixNames[] = {"BPSK", "QPSK", "8PSK", "16-PSK", "16QAM", "64QAM"};

/// @brief The operational definition of Gray: pairs at the minimum distance, and how many break it.
struct GrayCensus {
    std::size_t pairs      = 0UZ;
    std::size_t violations = 0UZ;
};

[[nodiscard]] GrayCensus grayCensus(const C& constellation) {
    const auto points = constellation.points();
    double     best   = std::numeric_limits<double>::infinity();
    for (std::size_t a = 0UZ; a + 1UZ < points.size(); ++a) {
        for (std::size_t b = a + 1UZ; b < points.size(); ++b) {
            best = std::min(best, std::abs(points[a] - points[b]));
        }
    }
    GrayCensus census;
    for (std::size_t a = 0UZ; a + 1UZ < points.size(); ++a) {
        for (std::size_t b = a + 1UZ; b < points.size(); ++b) {
            if (std::abs(std::abs(points[a] - points[b]) - best) > 1.0e-9 * best) {
                continue;
            }
            ++census.pairs;
            if (std::popcount(static_cast<std::uint32_t>(a ^ b)) != 1) {
                ++census.violations;
            }
        }
    }
    return census;
}

[[nodiscard]] double qFunction(double x) { return 0.5 * std::erfc(x / std::numbers::sqrt2); }

/// @brief Craig's finite-limit form of the exact M-PSK symbol error probability.
[[nodiscard]] double pskSymbolErrorProbability(std::size_t arity, double esOverN0) {
    const double upper = static_cast<double>(arity - 1UZ) * kPi / static_cast<double>(arity);
    const double gain  = std::sin(kPi / static_cast<double>(arity)) * std::sin(kPi / static_cast<double>(arity));

    constexpr std::size_t kSteps = 40000UZ; // Simpson, even count
    const double          step   = upper / static_cast<double>(kSteps);
    double                sum    = 0.0;
    for (std::size_t i = 0UZ; i <= kSteps; ++i) {
        const double theta  = step * static_cast<double>(i);
        const double sine   = std::sin(theta);
        const double value  = i == 0UZ ? 0.0 : std::exp(-gain * esOverN0 / (sine * sine));
        const double weight = (i == 0UZ || i == kSteps) ? 1.0 : (i % 2UZ == 1UZ ? 4.0 : 2.0);
        sum += weight * value;
    }
    return sum * step / 3.0 / kPi;
}

/// @brief Square QAM as its two independent PAM decisions.
[[nodiscard]] double qamSymbolErrorProbability(std::size_t arity, double esOverN0) {
    const double side = std::sqrt(static_cast<double>(arity));
    const double p    = 2.0 * (1.0 - 1.0 / side) * qFunction(std::sqrt(3.0 * esOverN0 / static_cast<double>(arity - 1UZ)));
    return 1.0 - (1.0 - p) * (1.0 - p);
}

struct ErrorRates {
    double symbol = 0.0;
    double bit    = 0.0;
};

[[nodiscard]] ErrorRates measureErrorRates(const C& constellation, double esOverN0dB, std::size_t count, std::uint64_t seed) {
    std::mt19937_64                              engine{seed};
    std::uniform_int_distribution<std::uint32_t> symbols{0U, static_cast<std::uint32_t>(constellation.size() - 1UZ)};
    const double                                 noisePower = std::pow(10.0, -esOverN0dB / 10.0);
    std::normal_distribution<double>             noise{0.0, std::sqrt(noisePower / 2.0)};

    std::size_t symbolErrors = 0UZ;
    std::size_t bitErrors    = 0UZ;
    for (std::size_t i = 0UZ; i < count; ++i) {
        const auto sent     = static_cast<std::uint8_t>(symbols(engine));
        const CD   received = constellation.point(sent) + CD(noise(engine), noise(engine));
        const auto decided  = constellation.hardDecision(received);
        if (decided != sent) {
            ++symbolErrors;
            bitErrors += static_cast<std::size_t>(std::popcount(static_cast<std::uint32_t>(sent ^ decided)));
        }
    }
    return {static_cast<double>(symbolErrors) / static_cast<double>(count), static_cast<double>(bitErrors) / static_cast<double>(count * constellation.bitsPerSymbol())};
}

/// @brief The per-axis minimum over the four 16QAM levels, written from the definition rather than the closed form.
void qam16AxisReference(double x, double spacing, double noisePower, double& high, double& low) {
    const std::array<double, 4> level{-1.5 * spacing, -0.5 * spacing, 0.5 * spacing, 1.5 * spacing};
    const std::array<int, 4>    label{0, 1, 3, 2};

    for (int bit = 1; bit >= 0; --bit) {
        double zero = std::numeric_limits<double>::infinity();
        double one  = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0UZ; i < 4UZ; ++i) {
            const double distance = (x - level[i]) * (x - level[i]);
            double&      target   = ((label[i] >> bit) & 1) != 0 ? one : zero;
            target                = std::min(target, distance);
        }
        (bit == 1 ? high : low) = (zero - one) / noisePower;
    }
}

/// @brief The level index a square-QAM symbol puts one axis at: that axis's label field, Gray-decoded.
template<std::floating_point F>
[[nodiscard]] std::uint32_t axisLevelOf(const Constellation<F>& constellation, std::uint8_t symbol, std::uint32_t levels, bool imaginary) {
    const auto bits  = static_cast<std::uint32_t>(constellation.bitsPerSymbol() / 2UZ);
    const auto field = imaginary ? static_cast<std::uint32_t>(symbol) >> bits : static_cast<std::uint32_t>(symbol);
    return grayDecode(field & (levels - 1U));
}

/**
 * @brief Whether `Nearest` is choosing by the tie rule rather than by distance.
 *
 * `|z - c|` absorbs a coordinate smaller than half an ulp of the level it is measured against, so below
 * that the reference has two bit-identical distances and falls back to the lowest symbol without
 * resolving the coordinate. The agreement with the closed form is a statement about the closed form, so it
 * is asserted wherever the reference can still resolve the difference.
 */
template<std::floating_point F>
[[nodiscard]] bool nearestIsBreakingATie(const Constellation<F>& constellation, std::complex<F> z) {
    F           best  = std::numeric_limits<F>::infinity();
    std::size_t count = 0UZ;
    for (const std::complex<F>& c : constellation.points()) {
        const F distance = std::norm(z - c);
        if (distance < best) {
            best  = distance;
            count = 1UZ;
        } else if (distance == best) {
            ++count;
        }
    }
    return count > 1UZ;
}

} // namespace

const boost::ut::suite<"constellation"> constellationTests = [] {
    using namespace boost::ut;

    "the strategy follows from the geometry, and is resolved once"_test = [] {
        expect(C::bpsk().strategy() == DecisionStrategy::Psk) << "BPSK is a two-point circle";
        expect(C::qpsk().strategy() == DecisionStrategy::SquareQam) << "QPSK is the smallest square QAM, so it takes the per-axis slicer and no atan2";
        expect(C::psk8().strategy() == DecisionStrategy::Psk);
        expect(C::psk(16UZ).strategy() == DecisionStrategy::Psk);
        expect(C::qam(16UZ).strategy() == DecisionStrategy::SquareQam);
        expect(C::qam(64UZ).strategy() == DecisionStrategy::SquareQam);

        const std::array<CD, 8> apsk{CD{1, 0}, CD{0, 1}, CD{-1, 0}, CD{0, -1}, CD{2, 0}, CD{0, 2}, CD{-2, 0}, CD{0, -2}};
        expect(C::custom(apsk).strategy() == DecisionStrategy::Nearest) << "two amplitude rings are neither a circle nor an axis product";

        expect(throws([] { (void)C::psk(6UZ); })) << "arity must be a power of two";
        expect(throws([] { (void)C::qam(8UZ); })) << "a square QAM arity must be a power of four";
        expect(throws([] { (void)C::psk(512UZ); })) << "symbols are bytes";
    };

    "the point tables, to nine figures"_test = [] {
        constexpr double kA = 0.707106781;
        constexpr double kI = 0.316227766;
        constexpr double kO = 0.948683298;

        const std::array<CD, 2>  bpsk{CD{-1, 0}, CD{+1, 0}};
        const std::array<CD, 4>  qpsk{CD{-kA, -kA}, CD{+kA, -kA}, CD{-kA, +kA}, CD{+kA, +kA}};
        const std::array<CD, 8>  psk8{CD{1, 0}, CD{kA, kA}, CD{-kA, kA}, CD{0, 1}, CD{kA, -kA}, CD{0, -1}, CD{-1, 0}, CD{-kA, -kA}};
        const std::array<CD, 16> qam16{CD{-kO, -kO}, CD{-kI, -kO}, CD{+kO, -kO}, CD{+kI, -kO}, CD{-kO, -kI}, CD{-kI, -kI}, CD{+kO, -kI}, CD{+kI, -kI}, //
            CD{-kO, +kO}, CD{-kI, +kO}, CD{+kO, +kO}, CD{+kI, +kO}, CD{-kO, +kI}, CD{-kI, +kI}, CD{+kO, +kI}, CD{+kI, +kI}};

        const auto compare = [](const C& constellation, std::span<const CD> table, const char* name) {
            expect(eq(constellation.size(), table.size()));
            double worst = 0.0;
            for (std::size_t s = 0UZ; s < table.size(); ++s) {
                worst = std::max(worst, std::abs(constellation.point(static_cast<std::uint8_t>(s)) - table[s]));
            }
            expect(lt(worst, 1.0e-9)) << std::format("{}: worst point error {:.3e}", name, worst);
        };
        compare(C::bpsk(), bpsk, "BPSK");
        compare(C::qpsk(), qpsk, "QPSK");
        compare(C::psk8(), psk8, "8PSK");
        compare(C::qam(16UZ), qam16, "16QAM");

        expect(approx(C::bpsk().minimumDistance(), 2.0, 1.0e-9));
        expect(approx(C::qpsk().minimumDistance(), 1.414213562, 1.0e-9));
        expect(approx(C::psk8().minimumDistance(), 0.765366865, 1.0e-9));
        expect(approx(C::qam(16UZ).minimumDistance(), 0.632455532, 1.0e-9));

        std::size_t index = 0UZ;
        for (const C& constellation : theSixConstellations()) {
            double power = 0.0;
            for (const CD& c : constellation.points()) {
                power += std::norm(c);
            }
            power /= static_cast<double>(constellation.size());
            expect(lt(std::abs(power - 1.0), 1.0e-12)) << std::format("{}: E|s|^2 = {:.15f}", kSixNames[index], power);
            ++index;
        }
    };

    "the normalization table, and that the ratio is the minimum-distance ratio"_test = [] {
        struct Row {
            std::size_t arity;
            double      meanMagnitude;
            double      rmsMagnitude;
            double      amplitudeScale;
            double      powerScale;
            double      ratio;
            double      decibels;
        };
        const std::array<Row, 2> rows{Row{16UZ, 2.995352392, 3.162277660, 0.333850535, 0.316227766, 1.055728090, 0.471042}, //
            Row{64UZ, 6.086890469, 6.480740698, 0.164287497, 0.154303350, 1.064704668, 0.544583}};

        for (const Row& row : rows) {
            const C raw          = C::qam(row.arity, Normalization::None);
            double  sumMagnitude = 0.0;
            double  sumPower     = 0.0;
            for (const CD& c : raw.points()) {
                sumMagnitude += std::abs(c);
                sumPower += std::norm(c);
            }
            const double meanMagnitude = sumMagnitude / static_cast<double>(row.arity);
            const double rmsMagnitude  = std::sqrt(sumPower / static_cast<double>(row.arity));
            expect(lt(std::abs(meanMagnitude - row.meanMagnitude), 1.0e-9)) << std::format("{}QAM mean|c| {:.9f}", row.arity, meanMagnitude);
            expect(lt(std::abs(rmsMagnitude - row.rmsMagnitude), 1.0e-9)) << std::format("{}QAM rms|c| {:.9f}", row.arity, rmsMagnitude);

            const C amplitude = C::qam(row.arity, Normalization::Amplitude);
            const C power     = C::qam(row.arity, Normalization::Power);
            expect(lt(std::abs(amplitude.normalizationScale() - row.amplitudeScale), 1.0e-9));
            expect(lt(std::abs(power.normalizationScale() - row.powerScale), 1.0e-9));

            const double ratio = amplitude.normalizationScale() / power.normalizationScale();
            expect(lt(std::abs(ratio - row.ratio), 1.0e-9)) << std::format("{}QAM amplitude/power {:.9f}", row.arity, ratio);
            expect(lt(std::abs(20.0 * std::log10(ratio) - row.decibels), 1.0e-6)) << std::format("{}QAM {:.6f} dB", row.arity, 20.0 * std::log10(ratio));

            const double distanceRatio = amplitude.minimumDistance() / power.minimumDistance();
            expect(lt(std::abs(distanceRatio - ratio), 1.0e-12)) << "the scale ratio is the minimum-distance ratio";
        }

        for (std::size_t arity : {2UZ, 4UZ, 8UZ, 16UZ}) {
            const C constellation = C::psk(arity, 0.0, 0U, Normalization::Amplitude);
            expect(lt(std::abs(constellation.normalizationScale() - 1.0), 1.0e-12)) << "a constant-modulus constellation cannot tell the two conventions apart";
        }
    };

    "every minimum-distance pair differs in exactly one bit"_test = [] {
        constexpr std::size_t kExpectedPairs[] = {1UZ, 4UZ, 8UZ, 16UZ, 24UZ, 112UZ};
        std::size_t           index            = 0UZ;
        for (const C& constellation : theSixConstellations()) {
            const GrayCensus census = grayCensus(constellation);
            expect(eq(census.pairs, kExpectedPairs[index])) << std::format("{}: {} minimum-distance pairs", kSixNames[index], census.pairs);
            expect(eq(census.violations, 0UZ)) << std::format("{}: {} Gray violations", kSixNames[index], census.violations);
            ++index;
        }
    };

    "rotation order is a permutation, and it is gray(r) ^ labelXor for the psk family"_test = [] {
        struct Family {
            std::size_t  arity;
            std::uint8_t labelXor;
        };
        const std::array<Family, 6> families{Family{2UZ, 1U}, Family{4UZ, 3U}, Family{8UZ, 0U}, Family{16UZ, 0U}, Family{8UZ, 5U}, Family{16UZ, 9U}};

        for (const Family& family : families) {
            const C    constellation = family.arity == 2UZ && family.labelXor == 1U ? C::bpsk() : (family.arity == 4UZ ? C::qpsk() : C::psk(family.arity, 0.0, family.labelXor));
            const auto forward       = constellation.rotationToGray();
            const auto reverse       = constellation.grayToRotation();
            expect(forward.has_value() && reverse.has_value()) << std::format("psk({}) sits on a circle", family.arity);

            for (std::uint32_t r = 0U; r < family.arity; ++r) {
                expect(eq(static_cast<std::uint32_t>((*forward)[r]), grayEncode(r) ^ family.labelXor)) << std::format("psk({}, xor {}) rotationToGray({})", family.arity, family.labelXor, r);
                expect(eq(static_cast<std::uint32_t>((*reverse)[(*forward)[r]]), r)) << "grayToRotation inverts rotationToGray";
            }

            // The definition the permutation exists for: stepping r by one rotates the point by 2*pi/M.
            const double step  = 2.0 * kPi / static_cast<double>(family.arity);
            double       worst = 0.0;
            for (std::uint32_t r = 0U; r < family.arity; ++r) {
                const CD from = constellation.point((*forward)[r]);
                const CD to   = constellation.point((*forward)[(r + 1U) % family.arity]);
                worst         = std::max(worst, std::abs(to - from * std::polar(1.0, step)));
            }
            expect(lt(worst, 1.0e-12)) << std::format("psk({}): rotation ordering off by {:.3e}", family.arity, worst);
        }

        expect(!C::qam(16UZ).rotationToGray().has_value()) << "rotation order is meaningless off a circle";
        expect(!C::qam(64UZ).grayToRotation().has_value());

        // qam(4) is QPSK, with the same points and the same labels, so its points do lie on a circle
        // and the permutation is meaningful, so it is documented rather than special-cased away.
        const auto squareFour = C::qam(4UZ).rotationToGray();
        expect(squareFour.has_value()) << "qam(4) is QPSK by another name";
        expect(eq((*squareFour)[0], (*C::qpsk().rotationToGray())[0]));
        expect(eq((*squareFour)[1], (*C::qpsk().rotationToGray())[1]));

        expect(eq(C::bpsk().rotationalSymmetry(), 2UZ));
        expect(eq(C::psk8().rotationalSymmetry(), 8UZ));
        expect(eq(C::qam(16UZ).rotationalSymmetry(), 4UZ));
        expect(eq(C::qam(64UZ).rotationalSymmetry(), 4UZ));
    };

    "every closed form agrees with exhaustive nearest-point search"_test = [] {
        const auto sweep = [](const C& constellation, const char* name) {
            std::mt19937_64                  engine{0x5EEDU};
            std::normal_distribution<double> gaussian{0.0, 1.3};
            std::size_t                      mismatches = 0UZ;
            for (std::size_t i = 0UZ; i < 400000UZ; ++i) {
                const CD z = CD(gaussian(engine), gaussian(engine));
                mismatches += constellation.hardDecision(z) != constellation.nearestDecision(z) ? 1UZ : 0UZ;
            }
            expect(eq(mismatches, 0UZ)) << std::format("{}: {} mismatches over 400000 Gaussian samples", name, mismatches);

            constexpr std::size_t kGrid          = 1001UZ;
            constexpr double      kSpan          = 2.5;
            std::size_t           gridMismatches = 0UZ;
            for (std::size_t a = 0UZ; a < kGrid; ++a) {
                const double re = -kSpan + 2.0 * kSpan * static_cast<double>(a) / static_cast<double>(kGrid - 1UZ);
                for (std::size_t b = 0UZ; b < kGrid; ++b) {
                    const double im = -kSpan + 2.0 * kSpan * static_cast<double>(b) / static_cast<double>(kGrid - 1UZ);
                    const CD     z  = CD(re, im);
                    gridMismatches += constellation.hardDecision(z) != constellation.nearestDecision(z) ? 1UZ : 0UZ;
                }
            }
            expect(eq(gridMismatches, 0UZ)) << std::format("{}: {} mismatches over a {}x{} grid spanning +/-{}", name, gridMismatches, kGrid, kGrid, kSpan);
        };

        sweep(C::bpsk(), "BPSK");
        sweep(C::qpsk(), "QPSK");
        sweep(C::psk8(), "8PSK");
        sweep(C::qam(16UZ), "16QAM");
        sweep(C::qam(64UZ), "64QAM");

        // The reported defect, pinned: forming `(x*scale + (L-1))*0.5 + 0.5` in `float` collapses every
        // coordinate within an ulp of `L-1` onto the tie, and the tie steps to the negative level. One
        // sample of 200000 complex Gaussian draws at standard deviation 1.3 lands in that band, the same
        // sample for all three orders.
        constexpr std::complex<float> reported{-0x1.9800b6p-1f, 0x1.df17f4p-25f};
        for (const std::uint32_t levels : {2U, 4U, 8U}) {
            const auto qam = Constellation<float>::qam(static_cast<std::size_t>(levels) * levels);
            expect(eq(qam.hardDecision(reported), qam.nearestDecision(reported))) << std::format("{}QAM at the reported sample ({:a}, {:a})", levels * levels, reported.real(), reported.imag());
            expect(ge(axisLevelOf(qam, qam.hardDecision(reported), levels, true), levels / 2U)) << std::format("{}QAM: an imaginary part of {:.3e} is above the axis zero", levels * levels, reported.imag());
        }

        // A few ulp either side of every inter-level boundary, and the whole binade ladder either side of
        // the axis zero, both signs and both axes. Across a boundary band the level index rises once and
        // never falls; at the axis zero the tie still goes to the negative level, every coordinate either
        // side of it lands on the level its sign says, and the closed form is still the nearest point
        // wherever the reference can resolve one.
        const auto bandSweep = [](auto probe, const char* precision) {
            using F                = decltype(probe);
            constexpr int kUlps    = 6;
            const F       infinity = std::numeric_limits<F>::infinity();

            for (const std::uint32_t levels : {2U, 4U, 8U}) {
                const auto qam      = Constellation<F>::qam(static_cast<std::size_t>(levels) * levels);
                const F    other    = static_cast<F>(0.2);
                const auto disagree = [&](std::complex<F> z) { return !nearestIsBreakingATie(qam, z) && qam.hardDecision(z) != qam.nearestDecision(z); };

                for (const bool imaginary : {false, true}) {
                    const auto at = [&](F x) { return imaginary ? std::complex<F>(other, x) : std::complex<F>(x, other); };

                    // An off-center boundary is not exactly representable in either domain, since the slicer
                    // rounds in the raw lattice and `Nearest` measures against the normalized points, so the
                    // two put it a couple of ulp apart and differ by one level between the two places. Each
                    // form is required to cross once, inside the band, and never back: a form that crossed
                    // early would hand a coordinate to the level on the far side.
                    for (std::uint32_t i = 1U; i < levels; ++i) {
                        if (i == levels / 2U) {
                            continue; // the axis zero, which is exact in both domains and is asserted below
                        }
                        F x = (static_cast<F>(2U * i) - static_cast<F>(levels)) * qam.minimumDistance() * F{0.5};
                        for (int step = 0; step < kUlps; ++step) {
                            x = std::nextafter(x, -infinity);
                        }

                        std::array<std::uint32_t, 2UZ> previous{i - 1U, i - 1U};
                        std::array<std::size_t, 2UZ>   crossings{};
                        for (int step = -kUlps; step <= kUlps; ++step) {
                            const std::array<std::uint8_t, 2UZ> decided{qam.hardDecision(at(x)), qam.nearestDecision(at(x))};
                            for (std::size_t form = 0UZ; form < 2UZ; ++form) {
                                const std::uint32_t level = axisLevelOf(qam, decided[form], levels, imaginary);
                                expect(level == i - 1U || level == i) << std::format("{} {}QAM: {} ulp off boundary {} decided level {}", precision, levels * levels, step, i, level);
                                expect(ge(level, previous[form])) << std::format("{} {}QAM: the level fell from {} to {} at {} ulp off boundary {}", precision, levels * levels, previous[form], level, step, i);
                                crossings[form] += level == i && previous[form] == i - 1U ? 1UZ : 0UZ;
                                previous[form] = level;
                            }
                            x = std::nextafter(x, infinity);
                        }
                        expect(eq(crossings[0UZ], 1UZ)) << std::format("{} {}QAM: the closed form crosses boundary {} {} times inside +/-{} ulp", precision, levels * levels, i, crossings[0UZ], kUlps);
                        expect(eq(crossings[1UZ], 1UZ)) << std::format("{} {}QAM: the nearest point crosses boundary {} {} times inside +/-{} ulp", precision, levels * levels, i, crossings[1UZ], kUlps);
                    }

                    // The axis zero and the whole binade ladder either side of it, down to the smallest
                    // subnormal. This is the band the affine map collapses onto the tie, and it is the one
                    // boundary that is exact in both domains, so here the agreement is unconditional wherever
                    // the reference can still resolve the coordinate.
                    expect(eq(axisLevelOf(qam, qam.hardDecision(at(F{0})), levels, imaginary), levels / 2U - 1U)) << std::format("{} {}QAM: the axis zero ties to the negative level", precision, levels * levels);
                    expect(!disagree(at(F{0}))) << std::format("{} {}QAM: the axis zero is exact in both domains", precision, levels * levels);

                    std::size_t absorbed = 0UZ;
                    for (int exponent = -1; exponent >= -1080; --exponent) {
                        const F magnitude = static_cast<F>(std::ldexp(1.0, exponent));
                        if (!(magnitude > F{0})) {
                            break;
                        }
                        expect(ge(axisLevelOf(qam, qam.hardDecision(at(magnitude)), levels, imaginary), levels / 2U)) << std::format("{} {}QAM: 2^{} is above the axis zero", precision, levels * levels, exponent);
                        expect(lt(axisLevelOf(qam, qam.hardDecision(at(-magnitude)), levels, imaginary), levels / 2U)) << std::format("{} {}QAM: -2^{} is below the axis zero", precision, levels * levels, exponent);
                        expect(!disagree(at(magnitude)) && !disagree(at(-magnitude))) << std::format("{} {}QAM: +/-2^{} disagrees with the nearest point", precision, levels * levels, exponent);
                        absorbed += nearestIsBreakingATie(qam, at(magnitude)) ? 1UZ : 0UZ;
                    }
                    expect(gt(absorbed, 0UZ)) << std::format("{} {}QAM: the ladder must reach below half an ulp of the level, where the reference stops resolving", precision, levels * levels);
                }
            }
        };
        bandSweep(float{}, "float ");
        bandSweep(double{}, "double");
    };

    "ties and boundaries resolve the way the definition does"_test = [] {
        // The origin is an exact tie for every constant-modulus constellation, so `Nearest` decides it
        // on the last bit of each point's magnitude, since `cos^2 + sin^2` is not 1 for an eighth of a
        // turn. Where the magnitudes are bit-identical the two forms agree; where they are not, the
        // closed form answers `sectorSymbol[0]`, deterministically, and that is what is pinned.
        const auto magnitudesAreIdentical = [](const C& constellation) {
            const auto points = constellation.points();
            return std::ranges::all_of(points, [&](const CD& c) { return std::norm(c) == std::norm(points[0]); });
        };

        std::size_t index = 0UZ;
        for (const C& constellation : theSixConstellations()) {
            const auto closed = constellation.hardDecision(CD{0.0, 0.0});
            if (constellation.strategy() == DecisionStrategy::SquareQam || magnitudesAreIdentical(constellation)) {
                expect(eq(closed, constellation.nearestDecision(CD{0.0, 0.0}))) << std::format("{} at the origin", kSixNames[index]);
            } else {
                expect(eq(closed, std::uint8_t{0})) << std::format("{} at the origin answers the symbol at its own phase offset", kSixNames[index]);
            }
            ++index;
        }

        // An axis zero is the one decision boundary that is exactly representable in both the raw
        // lattice the slicer rounds in and the normalized points `Nearest` measures against.
        for (std::size_t arity : {4UZ, 16UZ, 64UZ}) {
            const C constellation = C::qam(arity);
            for (const CD z : {CD{0.0, 0.37}, CD{0.37, 0.0}, CD{0.0, -0.37}, CD{-0.37, 0.0}}) {
                expect(eq(constellation.hardDecision(z), constellation.nearestDecision(z))) << std::format("{}QAM at an axis zero", arity);
            }
        }

        // Every other boundary is approached rather than hit: the two forms round in different domains
        // and an exactly-tabulated midpoint is not exactly a midpoint in either.
        for (std::size_t arity : {4UZ, 16UZ, 64UZ}) {
            const C           constellation = C::qam(arity);
            const std::size_t levels        = static_cast<std::size_t>(std::sqrt(static_cast<double>(arity)));
            const double      spacing       = constellation.minimumDistance();
            std::size_t       mismatches    = 0UZ;
            for (std::size_t i = 0UZ; i + 1UZ < levels; ++i) {
                const double boundary = (-static_cast<double>(levels - 1UZ) + 2.0 * static_cast<double>(i) + 1.0) * spacing / 2.0;
                for (const double nudge : {-1.0e-9, +1.0e-9}) {
                    for (const CD z : {CD{boundary + nudge, 0.31}, CD{0.31, boundary + nudge}, CD{boundary + nudge, boundary + nudge}}) {
                        mismatches += constellation.hardDecision(z) != constellation.nearestDecision(z) ? 1UZ : 0UZ;
                    }
                }
            }
            expect(eq(mismatches, 0UZ)) << std::format("{}QAM: {} mismatches either side of an axis boundary", arity, mismatches);
        }

        // The four diagonals, the points a natural-order 8PSK labeling places 157.5 degrees away.
        const C psk8 = C::psk8();
        for (const CD z : {CD{1, 1}, CD{-1, 1}, CD{1, -1}, CD{-1, -1}}) {
            expect(eq(psk8.hardDecision(z), psk8.nearestDecision(z))) << std::format("8PSK at ({}, {})", z.real(), z.imag());
        }
        expect(eq(psk8.hardDecision(CD{-1, 1}), std::uint8_t{2})) << "135 degrees is symbol 2, whose point is at 135 degrees";
    };

    "the round trip is exact, and 20 dB is inside the error floor"_test = [] {
        std::size_t index = 0UZ;
        for (const C& constellation : theSixConstellations()) {
            for (std::uint32_t s = 0U; s < constellation.size(); ++s) {
                const auto symbol = static_cast<std::uint8_t>(s);
                expect(eq(constellation.hardDecision(constellation.point(symbol)), symbol)) << std::format("{} symbol {}", kSixNames[index], s);
            }
            ++index;
        }

        expect(lt(measureErrorRates(C::qpsk(), 20.0, 100000UZ, 11U).symbol, 5.390296e-07 + 1.0e-9));
        expect(lt(measureErrorRates(C::psk8(), 20.0, 100000UZ, 12U).symbol, 6.679677e-03));
        expect(lt(measureErrorRates(C::qam(16UZ), 20.0, 100000UZ, 13U).symbol, 3.715085e-02));
    };

    "the symbol error rate against its exact formula"_test = [] {
        struct Row {
            const char* name;
            std::size_t arity;
            bool        square;
            double      at10dB;
            double      at14dB;
        };
        const std::array<Row, 3> rows{Row{"QPSK", 4UZ, true, 1.564790e-03, 5.390296e-07}, //
            Row{"8PSK", 8UZ, false, 8.700476e-02, 6.679677e-03},                          //
            Row{"16QAM", 16UZ, true, 2.220309e-01, 3.715085e-02}};

        for (const Row& row : rows) {
            for (const double decibels : {10.0, 14.0}) {
                const double linear = std::pow(10.0, decibels / 10.0);
                const double theory = row.square ? qamSymbolErrorProbability(row.arity, linear) : pskSymbolErrorProbability(row.arity, linear);
                const double tabled = decibels == 10.0 ? row.at10dB : row.at14dB;
                std::println("SER theory {:>5} at {:>2.0f} dB: {:.6e} (tabulated {:.6e}, ratio {:.6f})", row.name, decibels, theory, tabled, theory / tabled);
                expect(lt(std::abs(theory / tabled - 1.0), 1.0e-3)) << std::format("{} at {} dB: {:.6e} against the tabulated {:.6e}", row.name, decibels, theory, tabled);
            }
        }

        // The measured rate is a Monte Carlo estimate: over kSymbols symbols a rate p carries a relative
        // standard error of 1/sqrt(kSymbols*p), so a row is asserted only where the run is long enough to
        // resolve it at all, and the tolerance is four of those standard errors rather than a fixed figure.
        constexpr std::size_t kSymbols = 200'000UZ;
        std::size_t           asserted = 0UZ;
        for (const Row& row : rows) {
            const C constellation = row.square ? C::qam(row.arity) : C::psk8();
            for (const double decibels : {10.0, 14.0}) {
                const double linear         = std::pow(10.0, decibels / 10.0);
                const double theory         = row.square ? qamSymbolErrorProbability(row.arity, linear) : pskSymbolErrorProbability(row.arity, linear);
                const double expectedErrors = theory * static_cast<double>(kSymbols);
                if (expectedErrors < 100.0) {
                    continue;
                }
                const double measured  = measureErrorRates(constellation, decibels, kSymbols, 4242U).symbol;
                const double tolerance = 4.0 / std::sqrt(expectedErrors);
                ++asserted;
                expect(lt(std::abs(measured / theory - 1.0), tolerance)) << std::format("{} at {} dB: measured {:.6e} against theory {:.6e}, tolerance {:.3f}", row.name, decibels, measured, theory, tolerance);
            }
        }
        std::println("SER measured over {} symbols: {} of the {} rows carry enough errors to resolve", kSymbols, asserted, 2UZ * rows.size());
        expect(gt(asserted, 0UZ)) << "no row resolves at this length, so the measured leg asserts nothing";
    };

    "Gray labeling costs about one bit per symbol error"_test = [] {
        const auto check = [](const C& constellation, double decibels, const char* name) {
            const ErrorRates rates    = measureErrorRates(constellation, decibels, 200'000UZ, 909U);
            const double     expected = rates.symbol / static_cast<double>(constellation.bitsPerSymbol());
            expect(lt(std::abs(rates.bit / expected - 1.0), 0.10)) << std::format("{}: BER {:.6e} against SER/m {:.6e}", name, rates.bit, expected);
        };
        check(C::psk8(), 10.0, "8PSK");
        check(C::qam(16UZ), 10.0, "16QAM");
        check(C::qpsk(), 6.0, "QPSK");
    };

    "the soft closed forms are the closed forms they claim to be"_test = [] {
        std::mt19937_64                  engine{0xC0FFEEU};
        std::normal_distribution<double> gaussian{0.0, 1.0};
        std::array<double, 6>            soft{};

        for (const double noisePower : {0.1, 0.5, 2.0}) {
            const C      bpsk      = C::bpsk();
            const C      qpsk      = C::qpsk();
            const double axis      = 1.0 / std::numbers::sqrt2;
            double       worstBpsk = 0.0;
            double       worstQpsk = 0.0;
            for (std::size_t i = 0UZ; i < 20000UZ; ++i) {
                const CD z = CD(gaussian(engine), gaussian(engine));

                bpsk.softDecisionsExhaustive(z, noisePower, soft, SoftAlgorithm::Exact);
                worstBpsk = std::max(worstBpsk, std::abs(soft[0] - 4.0 * z.real() / noisePower));

                qpsk.softDecisionsExhaustive(z, noisePower, soft, SoftAlgorithm::Exact);
                worstQpsk = std::max(worstQpsk, std::abs(soft[0] - 4.0 * axis * z.imag() / noisePower));
                worstQpsk = std::max(worstQpsk, std::abs(soft[1] - 4.0 * axis * z.real() / noisePower));
            }
            expect(lt(worstBpsk, 1.0e-11)) << std::format("BPSK 4*Re/N0 at N0 = {}: worst {:.3e}", noisePower, worstBpsk);
            expect(lt(worstQpsk, 1.0e-11)) << std::format("QPSK 4*a*Im/N0, 4*a*Re/N0 at N0 = {}: worst {:.3e}", noisePower, worstQpsk);
        }

        const C      qam16   = C::qam(16UZ);
        const double spacing = qam16.minimumDistance();
        for (const double noisePower : {0.05, 0.2, 1.0}) {
            double worstHigh = 0.0;
            double worstLow  = 0.0;
            for (std::size_t i = 0UZ; i <= 40000UZ; ++i) {
                const double x          = -4.0 + 8.0 * static_cast<double>(i) / 40000.0;
                const double magnitude  = std::abs(x);
                const double closedHigh = (magnitude <= spacing ? 2.0 * spacing * x : 4.0 * spacing * x - 2.0 * spacing * spacing * (x < 0.0 ? -1.0 : 1.0)) / noisePower;
                const double closedLow  = 2.0 * spacing * (spacing - magnitude) / noisePower;

                double referenceHigh = 0.0;
                double referenceLow  = 0.0;
                qam16AxisReference(x, spacing, noisePower, referenceHigh, referenceLow);
                worstHigh = std::max(worstHigh, std::abs(closedHigh - referenceHigh));
                worstLow  = std::max(worstLow, std::abs(closedLow - referenceLow));
            }
            expect(lt(worstHigh, 1.0e-12)) << std::format("16QAM L_hi at N0 = {}: worst {:.3e}", noisePower, worstHigh);
            expect(lt(worstLow, 1.0e-12)) << std::format("16QAM L_lo at N0 = {}: worst {:.3e}", noisePower, worstLow);
        }

        struct Sample {
            double x;
            double high;
            double low;
        };
        const std::array<Sample, 7> samples{Sample{-1.600, -3.247715, -1.223858}, Sample{-0.800, -1.223858, -0.211929}, Sample{-0.400, -0.505964, +0.294036}, //
            Sample{0.000, 0.000000, +0.800000}, Sample{+0.400, +0.505964, +0.294036}, Sample{+0.800, +1.223858, -0.211929}, Sample{+1.600, +3.247715, -1.223858}};
        for (const Sample& sample : samples) {
            // The sample rows are stated per axis, so drive the I axis and hold Q at a level.
            qam16.softDecisions(CD(sample.x, 0.5 * spacing), 1.0, soft);
            expect(lt(std::abs(soft[2] - sample.high), 1.0e-6)) << std::format("16QAM L_hi at x = {:.3f}: {:.6f} against {:.6f}", sample.x, soft[2], sample.high);
            expect(lt(std::abs(soft[3] - sample.low), 1.0e-6)) << std::format("16QAM L_lo at x = {:.3f}: {:.6f} against {:.6f}", sample.x, soft[3], sample.low);
        }
    };

    "the max-log sign is the hard decision"_test = [] {
        std::mt19937_64                  engine{0xBEEFU};
        std::normal_distribution<double> gaussian{0.0, 1.1};
        std::array<double, 8>            soft{};

        std::size_t index = 0UZ;
        for (const C& constellation : theSixConstellations()) {
            const std::size_t bits          = constellation.bitsPerSymbol();
            std::size_t       disagreements = 0UZ;
            std::size_t       exactZeros    = 0UZ;
            for (std::size_t i = 0UZ; i < 50000UZ; ++i) {
                const CD   z      = CD(gaussian(engine), gaussian(engine));
                const auto symbol = constellation.hardDecision(z);
                constellation.softDecisions(z, 0.3, soft, SoftAlgorithm::MaxLog);
                for (std::size_t k = 0UZ; k < bits; ++k) {
                    const std::uint32_t bit = (static_cast<std::uint32_t>(symbol) >> (bits - 1UZ - k)) & 1U;
                    if (soft[k] == 0.0) {
                        ++exactZeros;
                    } else if ((soft[k] > 0.0) != (bit == 1U)) {
                        ++disagreements;
                    }
                }
            }
            expect(eq(disagreements, 0UZ)) << std::format("{}: {} sign disagreements", kSixNames[index], disagreements);
            expect(eq(exactZeros, 0UZ)) << std::format("{}: {} exactly-zero LLRs", kSixNames[index], exactZeros);
            ++index;
        }
    };

    "max-log against the exact log-likelihood ratio"_test = [] {
        struct Row {
            const char* name;
            std::size_t arity;
            bool        square;
            double      at005;
            double      at020;
            double      at100;
        };
        const std::array<Row, 4> rows{Row{"QPSK", 4UZ, true, 0.0, 0.0, 0.0}, Row{"8PSK", 8UZ, false, 0.6943, 0.7011, 0.6815}, //
            Row{"16QAM", 16UZ, true, 0.6226, 0.6747, 0.5061}, Row{"64QAM", 64UZ, true, 0.6922, 0.9558, 0.6870}};

        constexpr std::size_t kGrid = 241UZ;
        constexpr double      kSpan = 1.6;
        std::array<double, 8> maxLog{};
        std::array<double, 8> exact{};

        for (const Row& row : rows) {
            const C constellation = row.square ? C::qam(row.arity) : C::psk8();
            for (const double noisePower : {0.05, 0.2, 1.0}) {
                double worst = 0.0;
                double range = 0.0;
                for (std::size_t a = 0UZ; a < kGrid; ++a) {
                    const double re = -kSpan + 2.0 * kSpan * static_cast<double>(a) / static_cast<double>(kGrid - 1UZ);
                    for (std::size_t b = 0UZ; b < kGrid; ++b) {
                        const double im = -kSpan + 2.0 * kSpan * static_cast<double>(b) / static_cast<double>(kGrid - 1UZ);
                        const CD     z  = CD(re, im);
                        constellation.softDecisions(z, noisePower, maxLog, SoftAlgorithm::MaxLog);
                        constellation.softDecisionsExhaustive(z, noisePower, exact, SoftAlgorithm::Exact);
                        for (std::size_t k = 0UZ; k < constellation.bitsPerSymbol(); ++k) {
                            worst = std::max(worst, std::abs(maxLog[k] - exact[k]));
                            range = std::max(range, std::abs(exact[k]));
                        }
                    }
                }
                const double tabled = noisePower == 0.05 ? row.at005 : (noisePower == 0.2 ? row.at020 : row.at100);
                std::println("max-log vs exact {:>5} at N0 = {:.2f}: {:.4f} (tabulated {:.4f}), dynamic range {:.2f}", row.name, noisePower, worst, tabled, range);
                if (row.square && row.arity == 4UZ) {
                    expect(lt(worst, 1.0e-12)) << "max-log is exact for QPSK, and that is asserted rather than measured";
                } else {
                    expect(lt(std::abs(worst - tabled), 0.05 * tabled)) << std::format("{} at N0 = {}: {:.4f} against {:.4f}", row.name, noisePower, worst, tabled);
                    expect(lt(worst, 1.0)) << "the max-log error is bounded well inside one nat everywhere on the grid";
                }
            }
        }
    };

    "the square-QAM max-log separates per axis, exactly"_test = [] {
        constexpr std::size_t kGrid = 201UZ;
        constexpr double      kSpan = 1.8;
        std::array<double, 8> perAxis{};
        std::array<double, 8> full{};

        for (std::size_t arity : {16UZ, 64UZ}) {
            const C constellation = C::qam(arity);
            for (const double noisePower : {0.05, 0.2, 1.0}) {
                double worst = 0.0;
                for (std::size_t a = 0UZ; a < kGrid; ++a) {
                    const double re = -kSpan + 2.0 * kSpan * static_cast<double>(a) / static_cast<double>(kGrid - 1UZ);
                    for (std::size_t b = 0UZ; b < kGrid; ++b) {
                        const double im = -kSpan + 2.0 * kSpan * static_cast<double>(b) / static_cast<double>(kGrid - 1UZ);
                        const CD     z  = CD(re, im);
                        constellation.softDecisions(z, noisePower, perAxis, SoftAlgorithm::MaxLog);
                        constellation.softDecisionsExhaustive(z, noisePower, full, SoftAlgorithm::MaxLog);
                        for (std::size_t k = 0UZ; k < constellation.bitsPerSymbol(); ++k) {
                            worst = std::max(worst, std::abs(perAxis[k] - full[k]));
                        }
                    }
                }
                expect(lt(worst, 1.0e-12)) << std::format("{}QAM at N0 = {}: per-axis against the full two-dimensional form, worst {:.3e}", arity, noisePower, worst);
            }
        }
    };

    "a stream byte cannot index past the point list"_test = [] {
        const C bpsk = C::bpsk();
        for (std::uint32_t value = 0U; value < 256U; ++value) {
            const auto symbol = static_cast<std::uint8_t>(value);
            expect(eq(std::abs(bpsk.point(symbol) - bpsk.point(static_cast<std::uint8_t>(value % 2U))), 0.0)) << std::format("BPSK point({}) is point({} mod 2)", value, value);
        }
        const C qam16 = C::qam(16UZ);
        for (std::uint32_t value = 0U; value < 256U; ++value) {
            expect(eq(std::abs(qam16.point(static_cast<std::uint8_t>(value)) - qam16.point(static_cast<std::uint8_t>(value % 16U))), 0.0));
        }
    };

    "the noise power is linear, positive, and scales the soft values by its inverse"_test = [] {
        const C               qam16 = C::qam(16UZ);
        std::array<double, 4> soft{};
        expect(throws([&] { qam16.softDecisions(CD{0.3, -0.2}, 0.0, soft); })) << "zero noise power throws";
        expect(throws([&] { qam16.softDecisions(CD{0.3, -0.2}, -1.0, soft); })) << "a negative noise power throws";
        expect(throws([&] { qam16.softDecisions(CD{0.3, -0.2}, std::numeric_limits<double>::quiet_NaN(), soft); }));

        std::array<double, 4> half{};
        qam16.softDecisions(CD{0.31, -0.22}, 0.4, soft);
        qam16.softDecisions(CD{0.31, -0.22}, 0.8, half);
        for (std::size_t k = 0UZ; k < 4UZ; ++k) {
            expect(lt(std::abs(half[k] * 2.0 - soft[k]), 1.0e-12)) << "doubling N0 halves every soft value";
            expect((half[k] > 0.0) == (soft[k] > 0.0)) << "and changes no sign";
        }

        std::array<double, 3> tooShort{};
        expect(throws([&] { qam16.softDecisions(CD{0.3, -0.2}, 1.0, tooShort); })) << "the output span must hold bitsPerSymbol() values";
    };

    "the normalization is applied once, at construction"_test = [] {
        for (const Normalization mode : {Normalization::Power, Normalization::Amplitude, Normalization::None}) {
            const C first  = C::qam(16UZ, mode);
            const C second = C::qam(16UZ, mode);
            for (std::size_t s = 0UZ; s < 16UZ; ++s) {
                expect(eq(std::abs(first.points()[s] - second.points()[s]), 0.0)) << "two identical constructions give identical points";
            }
            expect(eq(first.normalizationScale(), second.normalizationScale()));
        }
    };

    "the span form is the per-symbol form, and float is registered"_test = [] {
        const C                          qam16 = C::qam(16UZ);
        std::mt19937_64                  engine{7U};
        std::normal_distribution<double> gaussian{0.0, 0.9};

        std::vector<CD> input(1000UZ);
        for (CD& z : input) {
            z = CD(gaussian(engine), gaussian(engine));
        }
        std::vector<std::uint8_t> bulk(input.size());
        qam16.hardDecisions(input, bulk);
        std::vector<double> bulkSoft(input.size() * qam16.bitsPerSymbol());
        qam16.softDecisions(input, 0.3, bulkSoft);

        std::array<double, 4> one{};
        std::size_t           mismatches = 0UZ;
        for (std::size_t i = 0UZ; i < input.size(); ++i) {
            mismatches += bulk[i] != qam16.hardDecision(input[i]) ? 1UZ : 0UZ;
            qam16.softDecisions(input[i], 0.3, one);
            for (std::size_t k = 0UZ; k < 4UZ; ++k) {
                mismatches += bulkSoft[i * 4UZ + k] != one[k] ? 1UZ : 0UZ;
            }
        }
        expect(eq(mismatches, 0UZ)) << "the span form is bit-identical to the per-symbol form, which is chunk independence for a memoryless block";

        const auto single = Constellation<float>::qam(16UZ);
        expect(eq(single.size(), 16UZ));
        expect(single.strategy() == DecisionStrategy::SquareQam);
        expect(eq(single.hardDecision(std::complex<float>{0.9f, 0.9f}), std::uint8_t{10}));
    };
};

int main() { /* tests are automatically registered and run */ }
