#include <boost/ut.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

#include <gnuradio-4.0/algorithm/sync/ControlLoop.hpp>

namespace {

using gr::sync::ControlLoop;
using gr::sync::designLoopGains;
using gr::sync::LoopGains;
using gr::sync::LoopOrder;
using gr::sync::PhaseWrap;
using gr::sync::wrapPhase;
using gr::sync::wrapPhaseBounded;

constexpr double kPi    = std::numbers::pi;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kFlat  = std::numbers::sqrt2 / 2.0; /// the classical maximally flat damping

/// One row of Rice's gain design at full precision, which an eight-decimal tabulation rounds.
struct GainRow {
    double noiseBandwidth = 0.0;
    double damping        = 0.0;
    double theta          = 0.0;
    double alpha          = 0.0;
    double beta           = 0.0;
    double poleRadius     = 0.0;
};

constexpr GainRow kGainTable[] = {
    {0.001, std::numbers::sqrt2 / 2.0, 0.0009428090415820633, 0.0026631134814793767, 3.5508179753058353e-06, 0.998667555555161}, //
    {0.010, std::numbers::sqrt2 / 2.0, 0.0094280904158206330, 0.0263134812735724940, 3.5084641698096656e-04, 0.986755551657262}, //
    {0.100, std::numbers::sqrt2 / 2.0, 0.0942809041582063400, 0.2334630350194552600, 3.1128404669260700e-02, 0.875520967756081}, //
    {0.010, 1.0, 0.0080000000000000000, 0.0314940791131267300, 2.5195263290501383e-04, 0.984126984126984},                       //
};

constexpr double kDampings[] = {0.5, 0.7071, 1.0, 2.0};

/// The closed-loop noise bandwidth the design actually delivers, from integrating `|H|^2`.
struct DeliveredRow {
    double requested    = 0.0;
    double delivered[4] = {}; /// one per entry of kDampings
};

constexpr DeliveredRow kDeliveredTable[] = {
    {0.001, {0.001001001, 0.001000889, 0.001000640, 0.001000221}}, //
    {0.005, {0.005025062, 0.005022259, 0.005016016, 0.005005538}}, //
    {0.010, {0.010100500, 0.010089186, 0.010064128, 0.010022158}}, //
    {0.050, {0.052562500, 0.052259274, 0.051616000, 0.050555262}}, //
    {0.100, {0.110500000, 0.109185248, 0.106528000, 0.102227560}}, //
};

/// The exact first-order inversion against the `4*Bn*T` approximation in common use.
struct FirstOrderRow {
    double requested        = 0.0;
    double approximateGain  = 0.0; /// lambda = 4*Bn*T
    double approximateGives = 0.0;
    double exactGain        = 0.0; /// lambda = 4*Bn*T / (1 + 2*Bn*T)
};

constexpr FirstOrderRow kFirstOrderTable[] = {
    {0.001, 0.004, 0.001002004008016032, 0.003992015968063872}, //
    {0.005, 0.020, 0.005050505050505051, 0.019801980198019802}, //
    {0.010, 0.040, 0.010204081632653062, 0.039215686274509800}, //
    {0.050, 0.200, 0.055555555555555560, 0.181818181818181820}, //
    {0.100, 0.400, 0.125000000000000000, 0.333333333333333370}, //
};

[[nodiscard]] std::complex<double> closedLoop(std::complex<double> z, double a, double b) { return ((a + b) * z - a) / (z * z + (a + b - 2.0) * z + (1.0 - a)); }

/**
 * @brief `Bn*T = (1/(2*pi)) * integral_0^pi |H(e^{jw})|^2 dw`, midpoint rule.
 *
 * `2^16` points reproduce a `2^20` reference to better than `1e-14` on every row of the table, the
 * integrand being smooth and periodic, which is the case the midpoint rule converges fastest on, so
 * the grid is well inside the `1e-6` the acceptance criterion asks for, at negligible cost.
 */
[[nodiscard]] double deliveredNoiseBandwidth(double a, double b, std::size_t grid = 1UZ << 16) {
    double total = 0.0;
    for (std::size_t k = 0UZ; k < grid; ++k) {
        const double               w = kPi * (static_cast<double>(k) + 0.5) / static_cast<double>(grid);
        const std::complex<double> z{std::cos(w), std::sin(w)};
        total += std::norm(((a + b) * z - a) / (z * z + (a + b - 2.0) * z + (1.0 - a)));
    }
    return total / (2.0 * static_cast<double>(grid));
}

/// The same integral for the first-order loop, whose closed form is `lambda / (2 * (2 - lambda))`.
[[nodiscard]] double deliveredNoiseBandwidthFirstOrder(double lambda, std::size_t grid = 1UZ << 16) {
    double total = 0.0;
    for (std::size_t k = 0UZ; k < grid; ++k) {
        const double               w = kPi * (static_cast<double>(k) + 0.5) / static_cast<double>(grid);
        const std::complex<double> z{std::cos(w), std::sin(w)};
        total += std::norm(lambda / (z - (1.0 - lambda)));
    }
    return total / (2.0 * static_cast<double>(grid));
}

/// The larger of the two pole moduli of `z^2 + (a+b-2) z + (1-a)`.
[[nodiscard]] double largestPoleModulus(double a, double b) {
    const double p    = a + b - 2.0;
    const double q    = 1.0 - a;
    const double disc = p * p - 4.0 * q;
    if (disc > 0.0) { // two real roots
        return std::max(std::abs(0.5 * (-p + std::sqrt(disc))), std::abs(0.5 * (-p - std::sqrt(disc))));
    }
    return std::sqrt(q); // a conjugate pair or a double root: their product is 1-a, so their common modulus is sqrt(1-a)
}

/// The subtractive phase wrap, under an iteration cap, so a stall is observable instead of fatal.
struct SubtractiveWrap {
    float         value      = 0.0f;
    std::uint64_t iterations = 0U;
    bool          stalled    = false; /// the subtrahend fell below the argument's ULP: the original spins here forever
};

[[nodiscard]] SubtractiveWrap subtractiveWrap(float phase, std::uint64_t cap) {
    constexpr float twoPi = 2.0f * std::numbers::pi_v<float>;

    SubtractiveWrap out{phase, 0U, false};
    while (out.value > twoPi) {
        const float next = out.value - twoPi;
        if (next == out.value || out.iterations >= cap) {
            out.stalled = true;
            return out;
        }
        out.value = next;
        ++out.iterations;
    }
    while (out.value < -twoPi) {
        const float next = out.value + twoPi;
        if (next == out.value || out.iterations >= cap) {
            out.stalled = true;
            return out;
        }
        out.value = next;
        ++out.iterations;
    }
    return out;
}

/**
 * @brief Close a second-order loop on a phase ramp: the reference advances by @p rate every step.
 *
 * A frequency step at the loop's input is a ramp in phase, and the phase is what a second-order loop
 * tracks; driving it with a frequency error instead exercises the integrator arm alone and settles
 * at `beta*Kdet` per step rather than at the loop bandwidth. The error is formed from the phase
 * before the update, which is the convention the closed-loop transfer function is written in.
 */
template<typename Loop>
void trackRamp(Loop& loop, double rate, double detectorGain, std::size_t steps, double* peakPhaseError = nullptr) {
    using State      = typename Loop::value_type;
    double reference = 0.0;
    for (std::size_t n = 0UZ; n < steps; ++n) {
        const double difference = wrapPhase(reference - static_cast<double>(loop.phase()));
        if (peakPhaseError != nullptr) {
            *peakPhaseError = std::max(*peakPhaseError, std::abs(difference));
        }
        [[maybe_unused]] const State instant = loop.step(static_cast<State>(detectorGain * difference));
        reference                            = wrapPhase(reference + rate);
    }
}

/// The mis-ordered loop the anti-windup criterion exists to reject: a hidden integrator, clamped only on the way out.
struct WoundUpLoop {
    double alpha        = 0.0;
    double beta         = 0.0;
    double minFrequency = -std::numeric_limits<double>::infinity();
    double maxFrequency = std::numeric_limits<double>::infinity();

    double hidden    = 0.0;
    double frequency = 0.0;
    double phase     = 0.0;

    double step(double error) {
        hidden += beta * error;
        frequency            = std::min(std::max(hidden, minFrequency), maxFrequency);
        const double instant = frequency + alpha * error;
        phase                = wrapPhase(phase + instant);
        return instant;
    }
};

} // namespace

const boost::ut::suite<"control loop"> controlLoopTests = [] {
    using namespace boost::ut;

    "Rice's gains are the table"_test = [] {
        for (const GainRow& row : kGainTable) {
            const LoopGains gains = designLoopGains(row.noiseBandwidth, row.damping, 1.0);
            expect(approx(gains.theta, row.theta, 1e-9 * row.theta)) << row.noiseBandwidth << " at zeta " << row.damping << ": theta";
            expect(approx(gains.alpha, row.alpha, 1e-9 * row.alpha)) << row.noiseBandwidth << " at zeta " << row.damping << ": alpha";
            expect(approx(gains.beta, row.beta, 1e-9 * row.beta)) << row.noiseBandwidth << " at zeta " << row.damping << ": beta";

            // Both poles are a conjugate pair here, or equal, so their common modulus is sqrt(1-alpha):
            // their product is the polynomial's constant term.
            expect(approx(std::sqrt(1.0 - gains.alpha), row.poleRadius, 1e-9 * row.poleRadius)) << row.noiseBandwidth << " at zeta " << row.damping << ": pole radius";
            expect(le(largestPoleModulus(gains.alpha, gains.beta), 1.0));
        }

        const LoopGains gains = designLoopGains(0.01, kFlat, 1.0);
        expect(approx(0.01, gains.theta * (kFlat + 0.25 / kFlat), 1e-15)) << "theta is wn*T/2";

        const LoopGains scaled = designLoopGains(0.01, kFlat, 4.0);
        expect(approx(scaled.alpha * 4.0, gains.alpha, 1e-15)) << "Kdet divides both arms";
        expect(approx(scaled.beta * 4.0, gains.beta, 1e-15));
        expect(approx(scaled.theta, gains.theta, 1e-15)) << "and leaves theta unchanged";

        const LoopGains critical = designLoopGains(0.010, 1.0, 1.0);
        const double    p        = critical.alpha + critical.beta - 2.0;
        expect(lt(std::abs(p * p - 4.0 * (1.0 - critical.alpha)), 1e-15)) << "the discriminant vanishes at zeta = 1";
        expect(approx(std::sqrt(1.0 - critical.alpha), 0.992 / 1.008, 1e-12)) << "leaving a double real pole at (1-theta)/(1+theta)";
    };

    "H(1) is one, which is what says the arms are not swapped"_test = [] {
        std::mt19937                           rng(20260821U);
        std::uniform_real_distribution<double> logBandwidth(std::log(0.005), std::log(0.2));
        std::uniform_real_distribution<double> damping(0.3, 4.0);
        std::uniform_real_distribution<double> detectorGain(0.1, 10.0);

        // The identity cancels 1 against 1 in the denominator, so what is left is the denominator's own
        // rounding divided by b rather than anything about the design, and it therefore grows as 1/b as the
        // bandwidth falls. Multiplying it back by b states the identity without the tolerance having to track
        // the loop bandwidth, which is what the closed-form statement below does over the full range.
        for (int trial = 0; trial < 20; ++trial) {
            const double    bandwidth = std::exp(logBandwidth(rng));
            const double    zeta      = damping(rng);
            const double    kdet      = detectorGain(rng);
            const LoopGains gains     = designLoopGains(bandwidth, zeta, kdet);

            const double a        = gains.alpha * kdet;
            const double b        = gains.beta * kdet;
            const double residual = std::abs(closedLoop(std::complex<double>{1.0, 0.0}, a, b) - 1.0) * b;
            expect(lt(residual, 1e-13)) << "Bn*T " << bandwidth << ", zeta " << zeta << ", Kdet " << kdet << ": scaled residual " << residual;
        }

        // Numerator and denominator both reduce to b, so what is left is eps/b.
        for (const double bandwidth : {1e-6, 1e-4, 1e-2, 0.2}) {
            const LoopGains gains = designLoopGains(bandwidth, kFlat, 1.0);
            const double    value = ((gains.alpha + gains.beta) - gains.alpha) / (1.0 + (gains.alpha + gains.beta - 2.0) + (1.0 - gains.alpha));
            expect(lt(std::abs(value - 1.0) * gains.beta, 1e-15)) << "Bn*T " << bandwidth;
        }
    };

    "the delivered noise bandwidth is the measured table"_test = [] {
        for (const DeliveredRow& row : kDeliveredTable) {
            for (std::size_t d = 0UZ; d < std::size(kDampings); ++d) {
                const LoopGains gains     = designLoopGains(row.requested, kDampings[d], 1.0);
                const double    delivered = deliveredNoiseBandwidth(gains.alpha, gains.beta);
                expect(approx(delivered, row.delivered[d], 1e-6)) << row.requested << " at zeta " << kDampings[d] << ": delivered " << delivered;
            }
        }

        // The approximation in full: exact asymptotically, 1% at 0.01, 10.5% at 0.1.
        const auto relative = [](double requested, double zeta) {
            const LoopGains gains = designLoopGains(requested, zeta, 1.0);
            return std::abs(deliveredNoiseBandwidth(gains.alpha, gains.beta) - requested) / requested;
        };
        expect(approx(relative(0.001, 0.5), 0.001001, 1e-6)) << "a tenth of a percent at Bn*T = 0.001";
        expect(approx(relative(0.010, 0.5), 0.010050, 1e-5)) << "one percent at Bn*T = 0.01";
        expect(approx(relative(0.100, 0.5), 0.105000, 1e-4)) << "and 10.5% at Bn*T = 0.1, where a second-order loop is not a useful object";
        expect(lt(relative(0.100, 2.0), relative(0.100, 0.5))) << "heavier damping is closer";
    };

    "the poles stay inside the unit circle over the whole parameter range"_test = [] {
        constexpr int kBandwidthSteps = 121;
        constexpr int kDampingSteps   = 38;

        double worst = 0.0;
        for (int i = 0; i < kBandwidthSteps; ++i) {
            const double bandwidth = std::pow(10.0, -5.0 + (std::log10(2.0) + 5.0) * static_cast<double>(i) / static_cast<double>(kBandwidthSteps - 1));
            for (int j = 0; j < kDampingSteps; ++j) {
                const double    zeta  = 0.3 + 3.7 * static_cast<double>(j) / static_cast<double>(kDampingSteps - 1);
                const LoopGains gains = designLoopGains(bandwidth, zeta, 1.0);
                worst                 = std::max(worst, largestPoleModulus(gains.alpha, gains.beta));
            }
        }
        expect(lt(worst, 1.0)) << "worst pole modulus over Bn*T in [1e-5, 2], zeta in [0.3, 4] is " << worst;

        const LoopGains wide = designLoopGains(2.0, kFlat, 1.0);
        expect(approx(largestPoleModulus(wide.alpha, wide.beta), 0.511408, 1e-6)) << "at Bn*T = 2 the poles are back inside";
    };

    "the first-order inversion is exact where 4*Bn*T is two percent high"_test = [] {
        for (const FirstOrderRow& row : kFirstOrderTable) {
            const LoopGains gains = designLoopGains(row.requested, 1.0, 1.0, LoopOrder::First);
            expect(eq(gains.alpha, 0.0)) << "a first-order loop has no proportional arm";
            expect(eq(gains.theta, 0.0)) << "and no natural frequency to report";
            expect(approx(gains.beta, row.exactGain, 1e-15));

            expect(approx(deliveredNoiseBandwidthFirstOrder(gains.beta), row.requested, 1e-9)) << row.requested << ": the exact inversion delivers what was asked";
            expect(approx(deliveredNoiseBandwidthFirstOrder(row.approximateGain), row.approximateGives, 1e-9)) << row.requested << ": and 4*Bn*T does not";
        }

        expect(approx(kFirstOrderTable[2].approximateGives / 0.010, 1.0204, 1e-4)) << "2% high at Bn*T = 0.01";
        expect(approx(kFirstOrderTable[4].approximateGives / 0.100, 1.25, 1e-9)) << "25% high at Bn*T = 0.1";

        expect(approx(designLoopGains(0.01, 0.3, 5.0, LoopOrder::First).beta * 5.0, designLoopGains(0.01, 4.0, 1.0, LoopOrder::First).beta, 1e-15)) << "Kdet divides the first-order gain too, and damping is not consulted";

        ControlLoop<double> loop(0.01, kFlat, 1.0, -1.0, 1.0, LoopOrder::First);
        expect(eq(loop.alpha(), 0.0));
        expect(throws<std::invalid_argument>([&] { loop.setGains(0.1, 0.01); })) << "setGains cannot install a proportional arm on a first-order loop";
        loop.setNoiseBandwidth(0.05);
        expect(eq(loop.alpha(), 0.0)) << "and a live bandwidth change does not re-derive one";
        expect(approx(loop.beta(), 0.2 / 1.1, 1e-15));
    };

    "a frequency step leaves no steady-state error"_test = [] {
        struct Case {
            double bandwidth = 0.0;
            double damping   = 0.0;
            double detector  = 0.0;
            double rate      = 0.0;
        };
        const Case kCases[] = {{0.010, kFlat, 1.0, 0.002}, {0.001, kFlat, 1.0, 0.0002}, {0.010, 1.0, 2.5, 0.002}, {0.050, 0.5, 0.4, 0.010}};

        for (const Case& item : kCases) {
            ControlLoop<double> loop(item.bandwidth, item.damping, item.detector);
            double              peak = 0.0;
            trackRamp(loop, item.rate, item.detector, static_cast<std::size_t>(20.0 / item.bandwidth), &peak);
            expect(approx(loop.frequency(), item.rate, 1e-9)) << "Bn*T " << item.bandwidth << ": frequency " << loop.frequency();
            expect(lt(peak, kPi)) << "and the transient slipped no cycle";
        }

        // The first-order loop is driven by its own detector, which measures the frequency directly.
        for (const double bandwidth : {0.001, 0.010, 0.100}) {
            constexpr double    kDetector = 3.0;
            constexpr double    kTarget   = 0.02;
            ControlLoop<double> loop(bandwidth, 1.0, kDetector, -1.0, 1.0, LoopOrder::First);
            for (std::size_t n = 0UZ; n < static_cast<std::size_t>(20.0 / bandwidth); ++n) {
                [[maybe_unused]] const double instant = loop.step(kDetector * (kTarget - loop.frequency()));
            }
            expect(approx(loop.frequency(), kTarget, 1e-9)) << "first order, Bn*T " << bandwidth;
        }

        // A float loop cannot reach 1e-9 and is not asked to: the integrator stops moving once
        // beta*error falls under half an ulp of the state. That dead band is what double is for.
        ControlLoop<float> narrow(0.010, kFlat, 1.0);
        trackRamp(narrow, 0.002, 1.0, 2000UZ);
        expect(approx(static_cast<double>(narrow.frequency()), 0.002, 1e-5));
    };

    "the clamp is on the integrator, so saturation leaves no windup"_test = [] {
        constexpr double kBandwidth = 0.01;
        constexpr double kLimit     = 0.01;

        const auto saturate = static_cast<std::size_t>(10.0 / kBandwidth);
        const auto recover  = static_cast<std::size_t>(20.0 / kBandwidth);

        ControlLoop<double> loop(kBandwidth, kFlat, 1.0, -kLimit, kLimit);
        for (std::size_t n = 0UZ; n < saturate; ++n) {
            [[maybe_unused]] const double instant = loop.step(1.0);
        }
        expect(eq(loop.frequency(), kLimit)) << "parked against the bound";
        expect(that % loop.saturated());

        double peak = 0.0;
        trackRamp(loop, 0.0, 1.0, recover, &peak); // the target drops to zero: a constant input phase
        expect(lt(std::abs(loop.frequency()), 1e-9)) << "frequency " << loop.frequency();
        expect(that % !loop.saturated());
        expect(lt(peak, kPi)) << "and the recovery slipped no cycle";

        constexpr LoopGains gains = designLoopGains(kBandwidth, kFlat, 1.0);
        WoundUpLoop         wrong{gains.alpha, gains.beta, -kLimit, kLimit, 0.0, 0.0, 0.0};
        for (std::size_t n = 0UZ; n < saturate; ++n) {
            [[maybe_unused]] const double instant = wrong.step(1.0);
        }
        expect(gt(wrong.hidden, 30.0 * kLimit)) << "the mis-ordered integrator ran away behind the limit: " << wrong.hidden;

        const double reference = wrong.phase;
        for (std::size_t n = 0UZ; n < recover; ++n) {
            [[maybe_unused]] const double instant = wrong.step(wrapPhase(reference - wrong.phase));
        }
        expect(gt(std::abs(wrong.frequency), 1e-3)) << "and is still pinned after the same recovery: " << wrong.frequency;
        expect(gt(std::abs(wrong.frequency), 1e6 * std::abs(loop.frequency()))) << "orders of magnitude, as the criterion says";
    };

    "the wrap is remainder, and remainder terminates"_test = [] {
        expect(eq(wrapPhase(0.0), 0.0));
        expect(eq(wrapPhase(kPi), kPi));
        expect(eq(wrapPhase(-kPi), -kPi));
        expect(eq(wrapPhase(kTwoPi), 0.0));
        expect(approx(wrapPhase(3.0 * kPi), -kPi, 1e-15)) << "ties resolve to the negative end";
        expect(lt(wrapPhase(3.0 * kPi), 0.0));
        expect(approx(wrapPhase(-3.0 * kPi), kPi, 1e-15));
        expect(gt(wrapPhase(-3.0 * kPi), 0.0));
        expect(approx(wrapPhase(100.0), -0.530964914873383, 1e-12));
        expect(approx(wrapPhase(-1.0e6), 0.357564167046875, 1e-12));
        expect(approx(wrapPhase(1.0e30), 0.027836527240, 1e-12));

        expect(gt(std::fmod(100.0, kTwoPi), kPi)) << "fmod takes the dividend's sign and is two turns wide: not an angle";

        // The statement here is that the reduction terminates, which is a property of the arithmetic and not
        // of the clock: std::remainder is one operation per call, so a thousand calls is a thousand operations
        // whatever the argument, and what is asserted is that every one of them came back inside the range.
        bool bounded = true;
        for (int repeat = 0; repeat < 1000; ++repeat) {
            bounded = bounded && std::abs(wrapPhase(1.0e30 * (1.0 + 1e-6 * static_cast<double>(repeat)))) <= kPi;
        }
        expect(that % bounded) << "every one of the thousand came back inside [-pi, pi]";

        expect(that % std::isnan(wrapPhase(std::numeric_limits<double>::quiet_NaN()))) << "NaN propagates rather than trapping";
        expect(that % std::isnan(wrapPhase(std::numeric_limits<double>::infinity()))) << "and infinity returns";
        expect(le(std::abs(wrapPhase(1.0e30f)), std::numbers::pi_v<float>)) << "float too";

        // Repeated subtraction, which is what this replaces. Every row run here is bounded; the row
        // that is not bounded is asserted through the float arithmetic instead of by running it.
        struct WrapRow {
            float         phase       = 0.0f;
            std::uint64_t iterations  = 0U;
            float         subtractive = 0.0f;
            double        exact       = 0.0;
            double        discrepancy = 0.0;
        };
        constexpr WrapRow kRows[] = {
            {7.0f, 1U, 0.7168145f, 0.7168147, 1.74846e-07},         //
            {100.0f, 15U, 5.7522111f, -0.5309649, 9.2984e-06},      //
            {10000.0f, 1591U, 3.4257035f, -2.8310090, 2.64728e-02}, //
            {1000000.0f, 158827U, 2.7226090f, -0.3575642, 3.08017}, //
            {10000000.0f, 1583682U, 0.0019083f, 2.7075436, 2.70564} //
        };
        for (const WrapRow& row : kRows) {
            const SubtractiveWrap got   = subtractiveWrap(row.phase, 4'000'000U);
            const double          exact = wrapPhase(static_cast<double>(row.phase));
            expect(that % !got.stalled);
            expect(eq(got.iterations, row.iterations)) << row.phase << ": iterations";
            expect(approx(static_cast<double>(got.value), static_cast<double>(row.subtractive), 1e-6)) << row.phase << ": subtractive result";
            expect(approx(exact, row.exact, 1e-6)) << row.phase << ": exact wrap";

            const double discrepancy = std::abs(wrapPhase(static_cast<double>(got.value) - exact));
            expect(approx(discrepancy, row.discrepancy, 1e-3 * row.discrepancy)) << row.phase << ": discrepancy modulo 2*pi is " << discrepancy;
        }

        // A float reduction is exact for the modulus it is given, and 2*pi in float is 2.8e-8 away
        // from 2*pi. That costs pi*2.8e-8 on a phase the loop keeps bounded, and 0.28 rad on one that
        // has been allowed to reach 1e7, small beside repeated subtraction's 2.71 rad at the same
        // argument, and one more reason the reduction happens every step.
        expect(approx(static_cast<double>(wrapPhase(1.0e7f)), 2.4292684, 1e-6));
        expect(approx(wrapPhase(1.0e7), 2.7075436, 1e-6)) << "where the double reduction is the angle itself";
        expect(lt(std::abs(static_cast<double>(wrapPhase(3.0f)) - wrapPhase(3.0)), 1e-7)) << "and a bounded phase cannot tell the two apart";

        // Where it stops terminating, asserted through the arithmetic that stalls it.
        constexpr float twoPi     = 2.0f * std::numbers::pi_v<float>;
        constexpr float kStalls   = 134217744.0f; // the first float above 2^27
        constexpr float kAdvances = 134217728.0f; // 2^27 itself
        expect(eq(kAdvances, static_cast<float>(1UL << 27)));
        expect(eq(kStalls - twoPi, kStalls)) << "subtraction stops making progress at 1.342177440e+08";
        expect(neq(kAdvances - twoPi, kAdvances)) << "and still advances at 1.342177280e+08";
        expect(that % subtractiveWrap(kStalls, 4'000'000U).stalled) << "so a subtractive loop stays there";
        expect(eq(subtractiveWrap(kStalls, 4'000'000U).iterations, 0U)) << "not even once";
        expect(that % subtractiveWrap(1.0e9f, 4'000'000U).stalled) << "and a tag payload of 1e9 hangs the flowgraph";
        expect(le(std::abs(wrapPhase(1.0e9f)), std::numbers::pi_v<float>)) << "where remainder answers immediately";
    };

    "the step's wrap is remainder, bit for bit, over the envelope the clamps allow"_test = [] {
        constexpr float kPiF    = std::numbers::pi_v<float>;
        constexpr float kTwoPiF = 2.0f * kPiF;

        // `std::remainder` is what the wrap means; the step computes it as `x - 2*pi*nearbyint(x/2*pi)`,
        // which is a fifth of the cost and the same number. What makes that legal is that the step's
        // argument is bounded before the reduction is ever formed: `phase` is whatever the previous
        // reduction returned, so |phase| <= pi, and the increment is `frequency + alpha*error` with the
        // frequency clamped on the line above. The envelope is therefore
        //
        //     pi + max(|minFrequency|, |maxFrequency|) + |alpha| * max|error|
        //
        // and every term of it is read off the object below rather than written down here. The clamp is
        // the one a carrier loop has: the frequency is radians per sample, so +/-pi is Nyquist and a
        // rate past it is aliased. The error is a phase detector's output, so max|error| is pi.
        // Bn*T = 0.1 is the widest bandwidth this design stays honest at, which is what makes
        // alpha as large as this class produces.
        const ControlLoop<float> carrier(0.100, kFlat, 1.0, -kPiF, kPiF);

        const float maxIncrement = std::max(std::abs(carrier.minFrequency()), std::abs(carrier.maxFrequency())) + carrier.alpha() * kPiF;
        const float envelope     = kPiF + maxIncrement;
        expect(approx(carrier.alpha(), 0.23346303f, 1e-7f)) << "alpha at the widest bandwidth the design is honest at";
        expect(approx(maxIncrement, 3.87503862f, 1e-7f)) << "so a step advances the phase by at most this";
        expect(approx(envelope, 7.01663113f, 1e-7f)) << "and the wrap is never handed more than this";
        expect(lt(envelope, 3.0f * kPiF)) << "which clears 3*pi = 9.42478, the first argument at which the two forms part company";

        const auto agree = [](float x) { return std::bit_cast<std::uint32_t>(wrapPhaseBounded(x)) == std::bit_cast<std::uint32_t>(std::remainder(x, kTwoPiF)); };

        // Uniform across the whole envelope, both signs: 2^22 points, a step of 3.3 microradians, which
        // is seven ULP at the top of the range and finer than that everywhere below it.
        constexpr std::size_t kPoints       = 1UZ << 22;
        std::size_t           mismatches    = 0UZ;
        float                 firstMismatch = 0.0f;
        for (std::size_t n = 0UZ; n <= kPoints; ++n) {
            const float x = -envelope + 2.0f * envelope * static_cast<float>(n) / static_cast<float>(kPoints);
            if (!agree(x)) {
                if (mismatches == 0UZ) {
                    firstMismatch = x;
                }
                ++mismatches;
            }
        }
        expect(eq(mismatches, 0UZ)) << "of " << kPoints << " points across [-" << envelope << ", " << envelope << "], first at " << firstMismatch;

        // Exhaustive where it could plausibly break: |x| = pi, the only argument inside 3*pi at which
        // the quotient crosses a half-integer and nearbyint changes its answer, and |x| = 2*pi, where
        // the result passes through zero. 16384 ULP either side of each, both signs.
        std::size_t nearMismatches = 0UZ;
        for (const float center : {kPiF, kTwoPiF}) {
            float x = center;
            for (int back = 0; back < 8192; ++back) {
                x = std::nextafter(x, 0.0f);
            }
            for (int n = 0; n < 16384; ++n, x = std::nextafter(x, 20.0f)) {
                nearMismatches += (agree(x) ? 0UZ : 1UZ) + ((agree(-x) || -x == -kTwoPiF) ? 0UZ : 1UZ); // -2*pi is the signed zero pinned below
            }
        }
        expect(eq(nearMismatches, 0UZ)) << "16384 ULP either side of pi and of 2*pi, both signs";

        // The boundaries, named.
        expect(eq(wrapPhaseBounded(0.0f), 0.0f));
        expect(eq(wrapPhaseBounded(kPiF), kPiF));
        expect(eq(wrapPhaseBounded(-kPiF), -kPiF));
        expect(eq(wrapPhaseBounded(kTwoPiF), 0.0f));
        expect(that % agree(envelope));
        expect(that % agree(-envelope));
        expect(that % agree(std::nextafter(kPiF, 0.0f)));
        expect(that % agree(std::nextafter(kPiF, 4.0f)));
        expect(that % agree(std::nextafter(kTwoPiF, 4.0f)));
        expect(that % agree(std::nextafter(kTwoPiF, 20.0f)));

        // The only two arguments in the envelope where they differ at all: the exact remainder is a
        // negative zero, and a subtraction in round-to-nearest cannot produce one. The same angle, equal
        // under every comparison, and distinguishable only by the sign bit.
        for (const float zero : {-0.0f, -kTwoPiF}) {
            expect(eq(wrapPhaseBounded(zero), std::remainder(zero, kTwoPiF))) << "equal as values";
            expect(that % !agree(zero)) << "and not as bits";
            expect(that % !std::signbit(wrapPhaseBounded(zero)));
            expect(that % std::signbit(std::remainder(zero, kTwoPiF)));
        }

        // Outside the envelope the cheap form is a different number, not a worse one: at exactly 3*pi
        // the division lands on the tie 1.5, nearbyint takes it up where the exact quotient rounds it
        // down, and the answers are a whole turn apart. That is why the bound above is a requirement on
        // the clamp rather than a tolerance.
        expect(approx(std::remainder(3.0f * kPiF, kTwoPiF), kPiF, 1e-5f)) << "remainder resolves the tie to the negative end of the quotient";
        expect(approx(wrapPhaseBounded(3.0f * kPiF), -kPiF, 1e-5f)) << "nearbyint resolves it to the even one";
        expect(gt(std::abs(wrapPhaseBounded(3.0f * kPiF) - std::remainder(3.0f * kPiF, kTwoPiF)), 6.0f)) << "a whole turn";

        // This is why the entry points that take a phase from outside the loop keep std::remainder:
        // nothing bounds a stream tag's payload.
        ControlLoop<float> injected(0.100, kFlat, 1.0, -kPiF, kPiF);
        injected.setPhase(1.0e7f);
        expect(eq(injected.phase(), wrapPhase(1.0e7f))) << "setPhase reduces with remainder, not with the step's form";
        expect(le(std::abs(injected.phase()), kPiF));
        injected.reset(1.0e7f, 0.0f);
        expect(eq(injected.phase(), wrapPhase(1.0e7f))) << "and so does reset";

        // End to end: the loop against a reference recursion written with std::remainder, bit identical
        // at every step. The drive is hard enough to park the frequency on both clamps, so the wrap sees
        // the top of the envelope and not just the middle of it.
        ControlLoop<float>                    loop(0.100, kFlat, 1.0, -kPiF, kPiF);
        const float                           alpha              = loop.alpha();
        const float                           beta               = loop.beta();
        float                                 referenceFrequency = 0.0f;
        float                                 referencePhase     = 0.0f;
        std::mt19937                          rng(20260821U);
        std::uniform_real_distribution<float> draw(-kPiF, kPiF);
        bool                                  identical = true;
        bool                                  canonical = true;
        bool                                  parked    = false;
        for (int n = 0; n < 200'000; ++n) {
            const float                  e       = draw(rng);
            [[maybe_unused]] const float instant = loop.step(e);

            referenceFrequency           = std::min(std::max(referenceFrequency + beta * e, -kPiF), kPiF);
            const float referenceInstant = referenceFrequency + alpha * e;
            referencePhase               = std::remainder(referencePhase + referenceInstant, kTwoPiF);

            identical = identical && std::bit_cast<std::uint32_t>(loop.phase()) == std::bit_cast<std::uint32_t>(referencePhase);
            canonical = canonical && std::abs(loop.phase()) <= kPiF;
            parked    = parked || loop.saturated();
        }
        expect(that % identical) << "200000 steps against a remainder reference, bit for bit";
        expect(that % canonical) << "and never outside [-pi, pi]";
        expect(that % parked) << "with the frequency against a clamp at some point, which is where the envelope is widest";
    };

    "the wrap is idempotent"_test = [] {
        std::mt19937                          rng(4711U);
        std::uniform_real_distribution<float> uniform(-1.0e8f, 1.0e8f);
        for (int n = 0; n < 1'000'000; ++n) {
            const float once  = wrapPhase(uniform(rng));
            const float twice = wrapPhase(once);
            if (once != twice) {
                expect(eq(once, twice)) << "wrap(wrap(x)) is not wrap(x)";
                break;
            }
        }
        expect(le(std::abs(wrapPhase(std::numbers::pi_v<float>)), std::numbers::pi_v<float>));
        expect(le(std::abs(wrapPhase(std::nextafter(std::numbers::pi_v<float>, 4.0f))), std::numbers::pi_v<float>));
    };

    "crossed bounds throw, and the limit cycle is unrepresentable"_test = [] {
        expect(throws<std::invalid_argument>([] { ControlLoop<double>(0.01, kFlat, 1.0, 0.5, 0.1); })) << "0.5, 0.1, 0.5, 0.1, ... once per sample is not a clamp";

        ControlLoop<double> loop(0.01, kFlat, 1.0, -1.0, 1.0);
        expect(throws<std::invalid_argument>([&] { loop.setMinFrequency(2.0); }));
        expect(throws<std::invalid_argument>([&] { loop.setMaxFrequency(-2.0); }));
        expect(eq(loop.minFrequency(), -1.0)) << "and a rejected bound changes nothing";
        expect(eq(loop.maxFrequency(), 1.0));

        loop.setFrequency(0.8);
        expect(eq(loop.frequency(), 0.8));
        loop.setMaxFrequency(0.5);
        expect(eq(loop.frequency(), 0.5)) << "changing a bound re-clamps the state immediately";
        expect(that % loop.saturated());
        loop.setMaxFrequency(1.0);
        expect(eq(loop.frequency(), 0.5)) << "and widening it back does not restore what was clamped away";
        expect(that % !loop.saturated());

        loop.setFrequency(9.0);
        expect(eq(loop.frequency(), 1.0)) << "setting the frequency clamps the assignment";
        expect(that % loop.saturated());
        loop.setFrequency(-9.0);
        expect(eq(loop.frequency(), -1.0));

        ControlLoop<double> stepping(0.05, 1.0, 1.0, -0.2, 0.2);
        bool                held = true;
        for (int n = 0; n < 1000; ++n) {
            [[maybe_unused]] const double instant = stepping.step(1.0);
            held                                  = held && stepping.frequency() <= 0.2;
        }
        expect(that % held) << "the clamp is inside the step and no caller can forget it";
        expect(eq(stepping.frequency(), 0.2));

        ControlLoop<double> unbounded(0.01, 1.0, 1.0);
        expect(that % !unbounded.saturated()) << "infinite bounds are a loop with no clamp, not one with a forgotten clamp";
        expect(throws<std::invalid_argument>([] { [[maybe_unused]] const LoopGains unused = designLoopGains(0.0, 1.0, 1.0); })) << "a bandwidth of zero designs nothing";
        expect(throws<std::invalid_argument>([] { [[maybe_unused]] const LoopGains unused = designLoopGains(0.01, 0.0, 1.0); }));
        expect(throws<std::invalid_argument>([] { [[maybe_unused]] const LoopGains unused = designLoopGains(0.01, 1.0, 0.0); })) << "and a detector gain of zero is the misconfiguration the mandatory argument exists to prevent";
        expect(throws<std::invalid_argument>([&] { unbounded.setGains(-0.1, 0.01); })) << "negative gains are a sign error in the detector, not a loop setting";
        unbounded.setGains(1.33, 2.5);
        expect(eq(unbounded.alpha(), 1.33)) << "gains outside [0, 1] are the caller's business: a range check here would exclude this design's own output";
    };

    "reset is the whole of the state"_test = [] {
        ControlLoop<double> loop(0.01, kFlat, 1.0, -0.4, 0.4);
        trackRamp(loop, 0.003, 1.0, 5000UZ);
        expect(gt(std::abs(loop.frequency()), 1e-6));

        loop.reset(0.25, 0.1);
        expect(eq(loop.phase(), 0.25));
        expect(eq(loop.frequency(), 0.1));

        loop.reset(7.0, 9.0);
        expect(approx(loop.phase(), wrapPhase(7.0), 1e-15)) << "the phase is wrapped";
        expect(eq(loop.frequency(), 0.4)) << "and the frequency clamped";

        // Two loops given the same reset are the same loop: the reset sets the whole of the state.
        ControlLoop<double> fresh(0.01, kFlat, 1.0, -0.4, 0.4);
        fresh.reset(0.25, 0.1);
        ControlLoop<double> used(0.01, kFlat, 1.0, -0.4, 0.4);
        trackRamp(used, 0.003, 1.0, 5000UZ);
        used.reset(0.25, 0.1);
        for (int n = 0; n < 200; ++n) {
            const double error = 0.01 * static_cast<double>(n % 7) - 0.02;
            expect(eq(fresh.step(error), used.step(error)));
        }
        expect(eq(fresh.frequency(), used.frequency()));
        expect(eq(fresh.phase(), used.phase()));

        ControlLoop<double, PhaseWrap::None> unwrapped(0.01, 1.0, 1.0);
        unwrapped.setPhase(1000.0);
        expect(eq(unwrapped.phase(), 1000.0)) << "the wrap mode is compile time, so the step keeps no branch for it";

        ControlLoop<double, PhaseWrap::Period> timing(0.01, 1.0, 1.0, 0.5, 8.0);
        timing.reset(0.0, 4.0);
        expect(eq(timing.frequency(), 4.0));
        timing.setPhase(9.5);
        expect(approx(timing.phase(), 1.5, 1e-15)) << "a timing loop wraps modulo its own period, not modulo 2*pi";
    };

    "a live bandwidth change moves no state"_test = [] {
        ControlLoop<double> loop(0.002, kFlat, 1.0, -0.5, 0.5);
        trackRamp(loop, 0.001, 1.0, 20000UZ);

        const double frequency = loop.frequency();
        const double phase     = loop.phase();
        const double alpha     = loop.alpha();

        loop.setNoiseBandwidth(0.02);
        expect(eq(loop.frequency(), frequency)) << "widening the loop to reacquire keeps the frequency it found";
        expect(eq(loop.phase(), phase));
        expect(gt(loop.alpha(), alpha)) << "and only the gains move";
        expect(eq(loop.noiseBandwidth(), 0.02));

        loop.setDamping(2.0);
        expect(eq(loop.frequency(), frequency));
        loop.setDetectorGain(4.0);
        expect(eq(loop.frequency(), frequency));
        expect(approx(loop.alpha(), designLoopGains(0.02, 2.0, 4.0).alpha, 1e-15));

        loop.setOrder(LoopOrder::First);
        expect(eq(loop.frequency(), frequency)) << "and changing the order only zeroes alpha";
        expect(eq(loop.phase(), phase));
        expect(eq(loop.alpha(), 0.0));
        expect(approx(loop.beta(), designLoopGains(0.02, 2.0, 4.0, LoopOrder::First).beta, 1e-15));

        expect(throws<std::invalid_argument>([&] { loop.setNoiseBandwidth(-1.0); }));
        expect(eq(loop.noiseBandwidth(), 0.02)) << "a rejected setting leaves the gains alone";
    };

    "the step has no chunk-dependent state"_test = [] {
        // The step is a scalar recursion over two members and one argument, so the only thing a block
        // holding it can get wrong is the order it feeds the errors in. Carrying one sequence across
        // arbitrary group boundaries is bit identical to one pass; the blocks that hold this carry
        // their own bit-identical-under-chunking test over their input spans.
        std::mt19937                           rng(99U);
        std::uniform_real_distribution<double> uniform(-0.3, 0.3);
        std::vector<double>                    errors(10000UZ);
        for (double& e : errors) {
            e = uniform(rng);
        }

        ControlLoop<float> whole(0.01, kFlat, 1.0, -0.5f, 0.5f);
        std::vector<float> reference(errors.size());
        for (std::size_t n = 0UZ; n < errors.size(); ++n) {
            reference[n] = whole.step(static_cast<float>(errors[n]));
        }

        ControlLoop<float>                         chunked(0.01, kFlat, 1.0, -0.5f, 0.5f);
        std::uniform_int_distribution<std::size_t> group(1UZ, 137UZ);
        std::size_t                                at        = 0UZ;
        bool                                       identical = true;
        while (at < errors.size()) {
            const std::size_t take = std::min(group(rng), errors.size() - at);
            for (std::size_t n = at; n < at + take; ++n) {
                identical = identical && chunked.step(static_cast<float>(errors[n])) == reference[n];
            }
            at += take;
        }
        expect(that % identical) << "every step's return, bit for bit";
        expect(eq(chunked.frequency(), whole.frequency()));
        expect(eq(chunked.phase(), whole.phase()));
    };
};

int main() { /* tests are automatically registered and run */ }
