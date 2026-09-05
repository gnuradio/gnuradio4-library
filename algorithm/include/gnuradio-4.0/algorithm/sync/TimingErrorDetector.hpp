#ifndef GNURADIO_ALGORITHM_TIMING_ERROR_DETECTOR_HPP
#define GNURADIO_ALGORITHM_TIMING_ERROR_DETECTOR_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <type_traits>

/**
 * @brief The timing error detectors as pure kernels.
 *
 * The sign convention; the wrong sign here makes the loop diverge:
 *
 * ```
 * tau_e  is the timing error of the sampling instant, in symbol periods,
 *        positive when the estimate is early (the true instant is later).
 * ```
 *
 * With that sign every S-curve here slopes upward through zero, every `Kted` is positive, and a
 * loop whose integral update is `averagePeriod += beta * error` lengthens the period and pushes the
 * next instant later, which is the correction an early estimate needs. A detector implemented with
 * the opposite sign is negated in its own definition, never by negating the loop gains.
 *
 * `y[k]` is the interpolated sample at symbol instant `k`, `y[k-1/2]` the sample half a symbol
 * earlier, `y'[k]` the derivative at the instant, and `a[k]` the decision for `y[k]`. Real and
 * imaginary contributions are added, which for a real input is just the real term. The caller
 * supplies the decisions: these kernels take `a[k]` as an argument and are independent of the constellation.
 *
 * ```
 *   detector          in/sym  decisions  derivative  error
 *   Mueller & Muller     1       yes         no      Re{a[k-1]}Re{y[k]} - Re{a[k]}Re{y[k-1]}, + Im
 *   modified M&M         1       yes         no      Re{(y[k]-y[k-2])conj(a[k-1]) - (a[k]-a[k-2])conj(y[k-1])}
 *   zero crossing        2       yes         no      (Re{a[k-1]} - Re{a[k]})Re{y[k-1/2]}, + Im
 *   Gardner              2       no          no      (Re{y[k-1]} - Re{y[k]})Re{y[k-1/2]}, + Im
 *   early-late           2       no          no      (Re{y[k+1/2]} - Re{y[k-1/2]})Re{y[k]}, + Im
 *   signal x slope ML    1       no          yes     (Re{y}Re{y'} + Im{y}Im{y'}) / 2
 *   signum x slope ML    1       no          yes     (sgn(Re{y})Re{y'} + sgn(Im{y})Im{y'}) / 2
 * ```
 *
 * The modified M&M and the two maximum-likelihood forms clip symmetrically at `+/-1`; the others do
 * not. Two properties decide most choices:
 *
 * - Gardner and early-late are blind. No decisions, so no dependence on the carrier phase: both are
 *   `Re{(u - v) conj(w)}` with `u`, `v` and `w` drawn from the same stream, and a common rotation of
 *   all three leaves the real part unchanged. They work before the carrier loop has locked.
 * - M&M and zero crossing need decisions, hence a locked carrier and a known constellation.
 *
 * `Kted`, the detector gain, is the slope of `E[e]` against `tau_e` at zero, in error units per
 * symbol period on an ideal Nyquist channel at unit average symbol power. Every form but one is
 * proportional to that power, so an AGC in front is mandatory, as it is for the band-edge
 * discriminant; the signum form scales with the constellation's mean axis magnitude instead. Each
 * detector has a gain function below, evaluated on the raised-cosine pulse a matched pair of
 * root-raised-cosine filters leaves. Only Gardner and early-late need the pulse summed; the rest
 * reduce to a closed form or to a single derivative of the pulse.
 *
 * A loop needs `Kted` per input sample, not per symbol, because a timing kernel's period state is in
 * input samples. `detectorGainPerSample` is the conversion and is the one place it should happen;
 * getting it wrong scales the loop bandwidth by `sps`.
 *
 * M&M, zero crossing, Gardner and early-late have odd S-curves that cross zero only at `tau_e = 0`
 * within `+/-0.5` and return to zero at the half-symbol boundary, an unstable false lock point.
 * Gardner and early-late share an S-curve; what separates them is the self-noise about it, which is
 * pure data-pattern jitter on an ideal channel. M&M and modified M&M have none of it there, since
 * `y[k] = a[k]` at `tau_e = 0` and the expression cancels identically, so M&M is the detector of
 * choice wherever decisions exist.
 * Early-late carries roughly three times Gardner's self-noise for the same S-curve and needs a sample
 * after the symbol instant, which costs an extra interpolation.
 *
 * Every kernel is a handful of multiplies with no branch except the clip and the signum, is
 * `constexpr`, and takes its arguments by value; all seven cost a rounding error against the
 * interpolation that feeds them. Which detector to run is a question about noise, not about cost.
 *
 * @see Mueller, K. H., Muller, M., "Timing Recovery in Digital Synchronous Data Receivers", IEEE
 *      Transactions on Communications, vol. COM-24, no. 5, May 1976, pp. 516-531, equation 49.
 * @see Gardner, F. M., "A BPSK/QPSK Timing-Error Detector for Sampled Receivers", IEEE Transactions
 *      on Communications, vol. COM-34, no. 5, pp. 423-429, 1986.
 * @see Danesfahani, G. R., Jeans, T. G., "Optimisation of modified Mueller and Muller algorithm",
 *      Electronics Letters, vol. 31, no. 13, 22 June 1995, pp. 1032-1033.
 */
namespace gr::sync {

/// @brief The sample types a detector kernel is written for.
template<typename T>
concept TimingSample = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;

/// @brief Which detector, chosen once at construction and never per symbol.
enum class TimingDetector { MuellerMuller, ModifiedMuellerMuller, ZeroCrossing, Gardner, EarlyLate, SignalTimesSlopeMl, SignumTimesSlopeMl };

/**
 * @brief What a scheduler needs to know about a detector, and nothing about how the error is computed.
 *
 * `errorDepth` counts the interpolated samples at symbol instants the formula reads, the current one
 * included; `inputsPerSymbol` counts the interpolations a symbol costs, so a midpoint detector is 2.
 * `needsLookahead` is true only for early-late, which reads a sample after the symbol instant and is
 * therefore the case most likely to break at a span boundary.
 */
struct TimingDetectorTraits {
    int  inputsPerSymbol = 1;
    int  errorDepth      = 1;
    bool needsDecisions  = false;
    bool needsDerivative = false;
    bool needsLookahead  = false;
};

[[nodiscard]] inline constexpr TimingDetectorTraits traitsOf(TimingDetector detector) noexcept {
    switch (detector) {
    case TimingDetector::MuellerMuller: return {1, 2, true, false, false};
    case TimingDetector::ModifiedMuellerMuller: return {1, 3, true, false, false};
    case TimingDetector::ZeroCrossing: return {2, 2, true, false, false};
    case TimingDetector::Gardner: return {2, 2, false, false, false};
    case TimingDetector::EarlyLate: return {2, 1, false, false, true};
    case TimingDetector::SignalTimesSlopeMl: return {1, 1, false, true, false};
    case TimingDetector::SignumTimesSlopeMl: return {1, 1, false, true, false};
    }
    return {};
}

namespace detail {

template<typename T>
struct TimingScalar {
    using type = T;
};

template<typename T>
struct TimingScalar<std::complex<T>> {
    using type = T;
};

template<TimingSample T>
using TimingScalarT = typename TimingScalar<T>::type;

template<TimingSample T>
[[nodiscard]] inline constexpr TimingScalarT<T> realPart(T v) noexcept {
    if constexpr (std::is_floating_point_v<T>) {
        return v;
    } else {
        return v.real();
    }
}

template<TimingSample T>
[[nodiscard]] inline constexpr TimingScalarT<T> imagPart(T v) noexcept {
    if constexpr (std::is_floating_point_v<T>) {
        return TimingScalarT<T>{0};
    } else {
        return v.imag();
    }
}

/// @brief `-1`, `0` or `+1`; zero maps to zero, so the signum detector is unbiased on a zero sample.
template<std::floating_point S>
[[nodiscard]] inline constexpr S signum(S v) noexcept {
    return static_cast<S>((v > S{0}) - (v < S{0}));
}

} // namespace detail

/// @brief The symmetric clip the modified M&M and the two maximum-likelihood forms carry.
template<std::floating_point S>
[[nodiscard]] inline constexpr S clipTimingError(S error, S limit) noexcept {
    return std::min(std::max(error, -limit), limit);
}

/// @brief Mueller & Muller: one interpolation per symbol, decisions required, exactly zero self-noise on an ideal channel.
template<TimingSample T>
[[nodiscard]] inline constexpr detail::TimingScalarT<T> muellerMullerError(T current, T previous, T currentDecision, T previousDecision) noexcept {
    using detail::imagPart;
    using detail::realPart;
    return realPart(previousDecision) * realPart(current) - realPart(currentDecision) * realPart(previous) //
           + imagPart(previousDecision) * imagPart(current) - imagPart(currentDecision) * imagPart(previous);
}

/// @brief The modified M&M: the same measurement over two symbol intervals, hence exactly twice the gain, clipped.
template<TimingSample T>
[[nodiscard]] inline constexpr detail::TimingScalarT<T> modifiedMuellerMullerError(T current, T previous, T older, T currentDecision, T previousDecision, T olderDecision, detail::TimingScalarT<T> limit = detail::TimingScalarT<T>{1}) noexcept {
    using detail::imagPart;
    using detail::realPart;
    const auto sampleDifferenceRe   = realPart(current) - realPart(older);
    const auto sampleDifferenceIm   = imagPart(current) - imagPart(older);
    const auto decisionDifferenceRe = realPart(currentDecision) - realPart(olderDecision);
    const auto decisionDifferenceIm = imagPart(currentDecision) - imagPart(olderDecision);

    const auto error = sampleDifferenceRe * realPart(previousDecision) + sampleDifferenceIm * imagPart(previousDecision) //
                       - decisionDifferenceRe * realPart(previous) - decisionDifferenceIm * imagPart(previous);
    return clipTimingError(error, limit);
}

/// @brief Zero crossing: the midpoint sample weighted by the decision transition. Two interpolations, decisions required.
template<TimingSample T>
[[nodiscard]] inline constexpr detail::TimingScalarT<T> zeroCrossingError(T midpoint, T currentDecision, T previousDecision) noexcept {
    using detail::imagPart;
    using detail::realPart;
    return (realPart(previousDecision) - realPart(currentDecision)) * realPart(midpoint) //
           + (imagPart(previousDecision) - imagPart(currentDecision)) * imagPart(midpoint);
}

/// @brief Gardner: the midpoint sample weighted by the sample transition. Blind: no decisions, no carrier.
template<TimingSample T>
[[nodiscard]] inline constexpr detail::TimingScalarT<T> gardnerError(T current, T previous, T midpoint) noexcept {
    using detail::imagPart;
    using detail::realPart;
    return (realPart(previous) - realPart(current)) * realPart(midpoint) //
           + (imagPart(previous) - imagPart(current)) * imagPart(midpoint);
}

/// @brief Early-late: the same S-curve as Gardner, 3.3x the self-noise, and a sample after the symbol instant.
template<TimingSample T>
[[nodiscard]] inline constexpr detail::TimingScalarT<T> earlyLateError(T current, T early, T late) noexcept {
    using detail::imagPart;
    using detail::realPart;
    return (realPart(late) - realPart(early)) * realPart(current) //
           + (imagPart(late) - imagPart(early)) * imagPart(current);
}

/// @brief Signal times slope: the maximum-likelihood form, needing the interpolating differentiator and no decisions.
template<TimingSample T>
[[nodiscard]] inline constexpr detail::TimingScalarT<T> signalTimesSlopeError(T sample, T slope, detail::TimingScalarT<T> limit = detail::TimingScalarT<T>{1}) noexcept {
    using detail::imagPart;
    using detail::realPart;
    using S          = detail::TimingScalarT<T>;
    const auto error = (realPart(sample) * realPart(slope) + imagPart(sample) * imagPart(slope)) / S{2};
    return clipTimingError(error, limit);
}

/// @brief Signum times slope: the same, with the sample replaced by its sign, so the gain does not scale with amplitude twice.
template<TimingSample T>
[[nodiscard]] inline constexpr detail::TimingScalarT<T> signumTimesSlopeError(T sample, T slope, detail::TimingScalarT<T> limit = detail::TimingScalarT<T>{1}) noexcept {
    using detail::imagPart;
    using detail::realPart;
    using detail::signum;
    using S          = detail::TimingScalarT<T>;
    const auto error = (signum(realPart(sample)) * realPart(slope) + signum(imagPart(sample)) * imagPart(slope)) / S{2};
    return clipTimingError(error, limit);
}

/**
 * @brief `Kted` for Mueller & Muller on a raised-cosine channel, in closed form, per symbol period.
 *
 * `2 E[|a|^2] cos(pi*alpha)/(1 - 4*alpha^2)`, with the `0/0` at `alpha = 1/2` replaced by its limit
 * `E[|a|^2] * pi/2`. `alpha = 0.5` is a common setting and is reached exactly, so the guard band on
 * `|1 - 4*alpha^2|` is two orders of magnitude wider than the plain expression needs.
 */
[[nodiscard]] inline double muellerMullerGain(double rolloff, double symbolPower = 1.0) {
    const double denominator = 1.0 - 4.0 * rolloff * rolloff;
    if (std::abs(denominator) < 1.0e-6) {
        return symbolPower * std::numbers::pi / 2.0;
    }
    return 2.0 * symbolPower * std::cos(std::numbers::pi * rolloff) / denominator;
}

/// @brief The modified M&M spans two symbol intervals instead of one, so its gain is exactly twice.
[[nodiscard]] inline double modifiedMuellerMullerGain(double rolloff, double symbolPower = 1.0) { return 2.0 * muellerMullerGain(rolloff, symbolPower); }

/**
 * @brief The raised-cosine channel pulse at unit peak, with the symbol period as the time unit.
 *
 * A matched pair of root-raised-cosine filters leaves this pulse, so it is the channel every gain
 * below is quoted on. It is one at the origin and zero at every other integer, which is what reduces
 * the decision-directed gains to a pair of pulse samples.
 */
[[nodiscard]] inline double raisedCosinePulse(double time, double rolloff) noexcept {
    const double phase = std::numbers::pi * time;
    const double sinc  = std::abs(time) < 1.0e-12 ? 1.0 : std::sin(phase) / phase;
    if (!(rolloff > 0.0)) {
        return sinc;
    }
    const double corner = 1.0 - 4.0 * rolloff * rolloff * time * time;
    if (std::abs(corner) < 1.0e-9) {
        return (std::numbers::pi / 4.0) * sinc; // the removable 0/0 at +/-1/(2*alpha)
    }
    return sinc * std::cos(std::numbers::pi * rolloff * time) / corner;
}

namespace detail {

/// The step a four-point central difference takes over the pulse; it holds eight digits of the slope.
inline constexpr double kSlopeStep = 1.0e-4;

/// Symbol periods of pulse tail the blind S-curve keeps either side of the instant.
inline constexpr int kPulseSpan = 256;

[[nodiscard]] inline double pulseSlope(double time, double rolloff) noexcept {
    const double h = kSlopeStep;
    return (raisedCosinePulse(time - 2.0 * h, rolloff) - 8.0 * raisedCosinePulse(time - h, rolloff) //
               + 8.0 * raisedCosinePulse(time + h, rolloff) - raisedCosinePulse(time + 2.0 * h, rolloff)) /
           (12.0 * h);
}

/// Gardner's S-curve at unit symbol power: the pulse's symbol-interval difference against its half-symbol shift.
[[nodiscard]] inline double gardnerSCurve(double timingError, double rolloff) noexcept {
    double sum = 0.0;
    for (int n = -kPulseSpan; n <= kPulseSpan; ++n) {
        const double at = static_cast<double>(n) - timingError;
        sum += (raisedCosinePulse(at - 1.0, rolloff) - raisedCosinePulse(at, rolloff)) * raisedCosinePulse(at - 0.5, rolloff);
    }
    return sum;
}

} // namespace detail

/**
 * @brief `Kted` for zero crossing on a raised-cosine channel, per symbol period.
 *
 * The decisions collapse the expectation onto two pulse samples half a symbol either side of the
 * instant, `E[|a|^2] (q(1/2 - tau_e) - q(-1/2 - tau_e))`, whose slope is `-2 E[|a|^2] q'(1/2)`. It
 * runs from `8/pi` at zero rolloff to exactly 3 at unit rolloff, so decisions buy a detector its
 * timing information back where a blind one has none. The derivative is taken numerically: the
 * closed form differences two terms that each diverge as the rolloff approaches one, and canceling
 * them costs more accuracy than the difference does.
 */
[[nodiscard]] inline double zeroCrossingGain(double rolloff, double symbolPower = 1.0) noexcept { return -2.0 * symbolPower * detail::pulseSlope(0.5, rolloff); }

/**
 * @brief `Kted` for Gardner on a raised-cosine channel, per symbol period.
 *
 * Blind, so nothing collapses the expectation and the whole pulse stays in it:
 * `E[|a|^2] sum_n (q(n-1-tau_e) - q(n-tau_e)) q(n-1/2-tau_e)`, differentiated at the origin. The sum
 * is truncated at 256 symbol periods either side, which holds eight digits at a rolloff of 0.05 and
 * above and loosens as the rolloff falls, the tail decaying only as its own reciprocal at zero. A
 * pulse confined to the Nyquist band carries no timing line for a detector that reads no decisions,
 * so a zero rolloff is answered with the exact zero rather than from the truncated sum.
 */
[[nodiscard]] inline double gardnerGain(double rolloff, double symbolPower = 1.0) noexcept {
    if (!(rolloff > 0.0)) {
        return 0.0;
    }
    const double h = detail::kSlopeStep;
    return symbolPower *
           (detail::gardnerSCurve(-2.0 * h, rolloff) - 8.0 * detail::gardnerSCurve(-h, rolloff) //
               + 8.0 * detail::gardnerSCurve(h, rolloff) - detail::gardnerSCurve(2.0 * h, rolloff)) /
           (12.0 * h);
}

/// @brief `Kted` for early-late, which shares Gardner's S-curve and so shares its gain; only the self-noise differs.
[[nodiscard]] inline double earlyLateGain(double rolloff, double symbolPower = 1.0) noexcept { return gardnerGain(rolloff, symbolPower); }

/**
 * @brief `Kted` for signal times slope on a raised-cosine channel, per symbol period: `pi^2 alpha / 4`.
 *
 * The expectation is `E[|a|^2] sum_n q(n-tau_e) q'(n-tau_e) / 2`, whose slope at the origin is
 * `-(E[|a|^2]/2) (q''(0) + sum_n q'(n)^2)`. Those two terms are each `pi^2/3` at zero rolloff and
 * cancel: the maximum-likelihood form is as blind to a Nyquist-band pulse as Gardner is. Their
 * difference is `pi^2 alpha / 2` thereafter, which is the one gain here that is linear in the
 * excess bandwidth.
 */
[[nodiscard]] inline constexpr double signalTimesSlopeGain(double rolloff, double symbolPower = 1.0) noexcept { return symbolPower * std::numbers::pi * std::numbers::pi * rolloff / 4.0; }

/**
 * @brief `Kted` for signum times slope on a raised-cosine channel, per symbol period.
 *
 * Replacing the sample by its sign takes the constellation's power out of the gain and puts its mean
 * axis magnitude in, so this is the one form whose gain is not proportional to `E[|a|^2]`.
 * `meanAxisMagnitude` is the average of `E[|Re{a}|]` and `E[|Im{a}|]`: `1/sqrt(2)` for unit-power
 * QPSK, `1/sqrt(5)` for unit-power 4-PAM, and `1/2` for unit-power BPSK, where the imaginary term
 * contributes nothing and the kernel still halves. The bracket is `-q''(0)`, which stays finite as
 * the rolloff falls, so unlike the other blind forms this one keeps a gain at zero excess bandwidth:
 * taking a sign is itself a decision.
 */
[[nodiscard]] inline constexpr double signumTimesSlopeGain(double rolloff, double meanAxisMagnitude) noexcept {
    constexpr double piSquared = std::numbers::pi * std::numbers::pi;
    return meanAxisMagnitude * (piSquared / 3.0 + rolloff * rolloff * (piSquared - 8.0));
}

/**
 * @brief `Kted` per input sample from `Kted` per symbol, as a timing loop's kernel requires.
 *
 * The kernel's frequency state is a clock period in input samples, so its detector gain must be in
 * error units per input sample. A detector's gain is quoted per symbol, since that is the figure
 * which does not change when `sps` does; passing it straight through scales the loop bandwidth by
 * `sps`.
 */
[[nodiscard]] inline constexpr double detectorGainPerSample(double gainPerSymbol, double samplesPerSymbol) noexcept { return gainPerSymbol / samplesPerSymbol; }

} // namespace gr::sync

#endif // GNURADIO_ALGORITHM_TIMING_ERROR_DETECTOR_HPP
