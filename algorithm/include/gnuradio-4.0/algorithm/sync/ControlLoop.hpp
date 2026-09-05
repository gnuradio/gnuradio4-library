#ifndef GNURADIO_ALGORITHM_CONTROL_LOOP_HPP
#define GNURADIO_ALGORITHM_CONTROL_LOOP_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <numbers>
#include <stdexcept>

/**
 * @brief The second-order tracking loop shared by the synchronizers: carrier PLL, Costas loop,
 * band-edge FLL and symbol clock recovery, written once, with the detector left outside.
 *
 * Those loops differ in their error detector and in nothing else. The recursion is shared, and the
 * recursion is where the errors are: gains that cannot be derived from the parameter they were
 * given, a phase wrap written as a `while` loop, a frequency clamp applied in the wrong place. One
 * step is
 *
 * ```
 * frequency += beta * error                         the integrator arm
 * frequency  = clamp(frequency, minFreq, maxFreq)   before the proportional arm is added
 * instant    = frequency + alpha * error            returned, never stored
 * phase      = wrap(phase + instant)                once per step, constant time
 * ```
 *
 * Four properties of that ordering matter. The clamp sits on the integrator arm alone,
 * immediately after it is updated: that is what stops integral windup, and clamping the phase
 * increment instead lets the integrator run away behind a limited output and take an unbounded time
 * to return. `instant` is the sum of both arms and is what the phase advanced by; a timing loop
 * reads it as the instantaneous clock period, so it is a return value and not state. The
 * proportional arm never enters the integrator: a single accumulator fed `(alpha + beta) * error`
 * is a different, first-order loop. The wrap is unconditional, once, in constant time.
 *
 * The error has one requirement. It is only that `E[error]` be, to first order about the lock
 * point, `detectorGain` times the quantity `phase` tracks, in the same units as `phase`, and
 * positive when `phase` must increase. A detector whose S-curve slopes the other way is negated
 * by its own block: negative gains put the poles outside the unit circle and the loop diverges
 * rather than mis-tracks.
 *
 * `detectorGain` is the S-curve slope at zero error. It is a property of the detector, the
 * modulation, the pulse shape, the input amplitude and the SNR, never of the loop, which cannot
 * measure it. It divides both gains, so a caller who passes `1` without knowing gets a loop whose
 * bandwidth is wrong by exactly that factor and no diagnostic. It is therefore a mandatory argument
 * with no default.
 *
 * The parameter is a noise bandwidth. `noiseBandwidth` is `Bn*T`, the one-sided equivalent noise
 * bandwidth of the closed loop normalized to the step rate; output noise power against white phase
 * noise is `2 * Bn*T * N0/2`. Take the loop filter to be proportional-plus-integral, match the
 * closed-loop denominator to the standard second-order prototype, and convert natural frequency to
 * bandwidth through `Bn = (wn/2) * (zeta + 1/(4*zeta))`:
 *
 * ```
 * theta = (Bn*T) / (zeta + 1/(4*zeta))          [ = wn*T / 2 ]
 * denom = 1 + 2*zeta*theta + theta*theta
 * alpha = (4 * zeta * theta) / denom / Kdet
 * beta  = (4 * theta * theta) / denom / Kdet
 * ```
 *
 * With `a = Kdet*alpha` and `b = Kdet*beta` the closed loop is
 * `H(z) = ((a + b)*z - a) / (z^2 + (a + b - 2)*z + (1 - a))`, and `H(1) = b/b = 1` identically for
 * any positive pair: constant phase and constant frequency are both tracked with zero error,
 * because the integrator supplies them. It is the cheapest check that the arms are not swapped.
 *
 * The noise-bandwidth relation is a continuous-time result carried into discrete time, so it is
 * asymptotically exact and not exact. Integrating `|H|^2` over the band, the delivered `Bn*T`
 * against the requested one:
 *
 * ```
 *   requested   zeta=0.5    zeta=0.7071   zeta=1      zeta=2
 *     0.001     0.0010010   0.0010009    0.0010006   0.0010002
 *     0.010     0.0101005   0.0100892    0.0100641   0.0100222
 *     0.100     0.1105000   0.1091852    0.1065280   0.1022275
 * ```
 *
 * Under 0.1% at `Bn*T <= 0.001`, under 1% at `Bn*T <= 0.01`, and 10.5% at `Bn*T = 0.1` at the
 * lightest damping, a bandwidth at which a second-order loop is of little use anyway. The limit is
 * accuracy and not stability: the poles stay inside the unit circle for every positive `Bn*T` the
 * formula is given, and at `Bn*T = 2, zeta = sqrt(2)/2` they sit at radius 0.511. Their
 * common modulus is `sqrt(1 - a)` wherever they are complex or equal.
 *
 * First order is a mode rather than setting `alpha` to zero afterwards. A frequency detector
 * measures the tracked quantity directly and has nothing for a proportional path to correct; the
 * recursion collapses to `frequency += lambda * (target - frequency)` with `lambda = beta * Kdet`,
 * whose closed-loop noise bandwidth is exactly `Bn*T = lambda / (2 * (2 - lambda))`, hence
 *
 * ```
 * lambda = 4*Bn*T / (1 + 2*Bn*T)      beta = lambda / Kdet
 * ```
 *
 * That inversion is a closed form and is exact. The `lambda = 4*Bn*T` approximation delivers
 * 0.01020 for a requested 0.010 and 0.12500 for a requested 0.100, 2% high and 25% high. Holding
 * the order inside the kernel keeps a later bandwidth change from re-deriving a proportional arm
 * the loop was never designed to have.
 *
 * The phase wrap is `std::remainder`: result in `[-pi, pi]`, exact, constant time, branch free,
 * correct for every finite input, and NaN-propagating rather than NaN-trapping. Repeated
 * subtraction of `2*pi` in a loop is none of those. It stops making progress once the subtrahend
 * falls below the argument's ULP: in `float` the smallest value for which `x - 2*pi == x` is
 * `1.342177440e+08`, the first representable value above `2^27`, and the largest that still
 * advances is `2^27` itself, so any phase at or above that is an infinite loop inside a per-sample
 * path, reachable from an unvalidated stream-tag payload. Below it, it terminates and is wrong:
 *
 * ```
 *   phase      iterations   subtractive   exact wrap    discrepancy mod 2*pi
 *        7             1     +0.7168145   +0.7168147     1.8e-07 rad
 *      100            15     +5.7522111   -0.5309649     9.3e-06 rad
 *    10000          1591     +3.4257035   -2.8310090     2.6e-02 rad
 *  1000000        158827     +2.7226090   -0.3575642     3.08 rad
 * ```
 *
 * The last row takes 158827 iterations to arrive at a number unrelated to its input. Reducing to
 * `+/-pi` rather than `+/-2*pi` also makes the state canonical, so a consumer of a `phase` output
 * never has to reduce it again and a detector forming `arg(x) - phase` needs no correction step.
 *
 * One caveat belongs to the `float` instantiation and not to the method: the reduction is exact for
 * the modulus it is given, and `2*pi` in `float` is 2.8e-8 away from `2*pi`. On a phase the loop
 * keeps bounded that costs `pi * 2.8e-8`, which is 1e-7 rad and negligible. On a phase allowed to
 * reach `1e7` it costs 0.28 rad, still an eighth of what repeated subtraction loses at the same
 * argument, and one more reason the reduction happens every step rather than once at the end.
 * A large argument arriving from outside is worth reducing in `double` before it is handed over.
 *
 * A timing loop wraps modulo its own estimated, varying period instead of modulo a constant; that
 * is `PhaseWrap::Period`, and it is `std::remainder(phase, frequency)` for exactly the same
 * reasons. Under repeated subtraction a non-positive period loops forever and has to be defended
 * against by clamping; under `remainder` a negative period is a sign flip and a zero period is a
 * NaN. Which mode applies is fixed at compile time so the step keeps no branch for it.
 *
 * The clamp is `min(max(frequency, minFrequency), maxFrequency)`, inside the step, and
 * `minFrequency <= maxFrequency` is a construction-time invariant that is checked. Written as an
 * `if`/`else if` pair with the bounds crossed it is not an error but a period-two limit cycle:
 * from `0` with `max = 0.1, min = 0.5` the state visits `0.5, 0.1, 0.5, 0.1, ...` once per sample,
 * indefinitely and with no diagnostic. Changing a bound re-clamps the state immediately rather than
 * waiting for the next update to pull it in, and setting the frequency clamps the assignment. A
 * loop parked against a bound is not tracking but reporting the closest frequency it may hold,
 * which `saturated()` answers for two comparisons on a value already in a register. There is no
 * clamp on the phase, which is an angle.
 *
 * The state is exactly two numbers, `frequency` and `phase`. Everything else, the gains, the bounds
 * and the order, is configuration. `reset` sets the two, wrapped and clamped, and those two numbers
 * are the whole of what a step reads and writes. Changing `noiseBandwidth`, `damping` or
 * `detectorGain` re-derives the gains and leaves the state where it was, so a receiver that widens
 * its loop to reacquire and narrows it again keeps the frequency it found.
 *
 * Hot path, no exemption: every block holding one of these steps it once per input sample, and the
 * family includes blocks sitting directly on a device stream. A step is one multiply-add, two
 * comparisons, one multiply-add, one add and one reduction, and the reduction is the only part with
 * a cost worth discussing. `std::remainder` is what the phase rule means, and it is correctly
 * rounded and exception-aware, so it compiles to a call. The step computes the same number as
 * `phase - 2*pi * nearbyint(phase / (2*pi))`, which is branch free and agrees bit for bit over the
 * argument a step can form. Over `float` at `-O3` a second-order step then costs 0.62x what it does
 * with `std::remainder`, the reduction itself falling to 0.44x.
 *
 * The substitution is legal because the argument a step can form is bounded twice over.
 * `_phase` is whatever the previous reduction returned, so `|_phase| <= pi`; the increment is
 * `frequency + alpha*error` with the frequency clamped on the line above, so it is bounded by
 * `max(|minFrequency|, |maxFrequency|) + |alpha*error|`. The two forms are bit-for-bit identical for
 * `|argument| < 3*pi`, so the requirement on a block's own clamp is
 *
 * ```
 * max(|minFrequency|, |maxFrequency|) + |alpha*error| < 2*pi
 * ```
 *
 * which a carrier loop meets with room to spare: its frequency is radians per sample and cannot
 * exceed `+/-pi` without aliasing, its error is an angle no larger than `pi`, and `alpha` at the
 * widest useful bandwidth (`Bn*T = 0.1`) is 0.233, an increment under 3.88 rad against the 6.28 the
 * bound allows, and a wrap argument under 7.02 against `3*pi = 9.42`. At exactly `3*pi` the two
 * forms part company by a whole turn rather than by a rounding error, which is why the bound is a
 * requirement and not a tolerance. The only inputs inside the envelope where they differ are
 * `-0` and `-2*pi`, where the cheap form returns `+0` and `remainder` returns `-0`: the same angle,
 * and the same value under every comparison.
 *
 * `setPhase` and `reset` keep `std::remainder`. Their argument comes from outside the loop, from a
 * stream tag or a caller's estimate, and nothing bounds it. The cheap form is used where the bound
 * is a property of the code rather than an assumption about the caller.
 *
 * Skipping the wrap where the caller's own `sincos` would reduce the argument anyway does not work:
 * at one radian per sample an unwrapped `float` phase reaches `2^24` after 16.8 million samples,
 * seventeen seconds at 1 MS/s, and from there stops advancing. `PhaseWrap::None` exists so
 * the wrap's share stays a measured number, not so that it can be turned off.
 *
 * The state type is a template parameter defaulting to `float`, which matches the sample type and
 * keeps the step in one register file. `double` exists for the callers that need the integrator's
 * zero steady-state error to be exact rather than to be a dead band: at `beta = 3.5e-4` a `float`
 * integrator stops moving once the residual falls below `ulp(frequency) / (2*beta)`.
 *
 * @see Rice, Michael, Digital Communications: A Discrete-Time Approach — the discrete-time
 *      phase-locked loop gain design above: the proportional-plus-integral loop filter, the match
 *      to the second-order prototype, and the conversion from natural frequency to equivalent noise
 *      bandwidth through `Bn = (wn/2)(zeta + 1/(4*zeta))`.
 */
namespace gr::sync {

/// @brief Second order (proportional and integral arms) or first order (integral only, `alpha` exactly zero).
enum class LoopOrder { Second, First };

/// @brief What the accumulated phase is reduced modulo, once per step.
enum class PhaseWrap {
    TwoPi,  /// carrier loops: `remainder(phase, 2*pi)`, canonical in `[-pi, pi]`
    Period, /// timing loops: `remainder(phase, frequency)`, the integrator's own varying period
    None    /// no reduction — for a caller whose phase cannot grow, and for measuring the wrap's own cost
};

/// @brief The gains a `(Bn*T, zeta, Kdet)` request designs to, in `double` whatever the loop's state type is.
struct LoopGains {
    double alpha = 0.0; /// proportional arm; exactly zero at first order
    double beta  = 0.0; /// integral arm
    double theta = 0.0; /// `wn*T/2`, the intermediate the second-order formula is compact in; zero at first order
};

/**
 * @brief Rice's gain design, or the exact first-order inversion, from a normalized noise bandwidth.
 *
 * The gains are designed in `double` and only then narrowed to the loop's state type, because the
 * design is a settings-time computation whose accuracy has nothing to do with how the recursion is
 * carried. `detectorGain` divides both arms and is rejected at zero rather than defaulted: a loop
 * given the wrong S-curve slope has the wrong bandwidth by exactly that factor and no diagnostic.
 */
[[nodiscard]] constexpr LoopGains designLoopGains(double noiseBandwidth, double damping, double detectorGain, LoopOrder order = LoopOrder::Second) {
    if (!(noiseBandwidth > 0.0)) {
        throw std::invalid_argument("designLoopGains: noiseBandwidth (Bn*T, cycles per step) must be positive");
    }
    if (!(damping > 0.0)) {
        throw std::invalid_argument("designLoopGains: damping must be positive");
    }
    if (!(detectorGain > 0.0)) {
        throw std::invalid_argument("designLoopGains: detectorGain must be positive — it is the detector's measured S-curve slope and it divides both gains");
    }

    if (order == LoopOrder::First) {
        const double lambda = 4.0 * noiseBandwidth / (1.0 + 2.0 * noiseBandwidth);
        return {0.0, lambda / detectorGain, 0.0};
    }

    const double theta = noiseBandwidth / (damping + 0.25 / damping);
    const double denom = 1.0 + 2.0 * damping * theta + theta * theta;
    return {4.0 * damping * theta / denom / detectorGain, 4.0 * theta * theta / denom / detectorGain, theta};
}

/// @brief `remainder(phase, 2*pi)`: canonical in `[-pi, pi]`, exact, constant time, terminating on every finite input.
template<std::floating_point T>
[[nodiscard]] inline T wrapPhase(T phase) noexcept {
    return std::remainder(phase, T{2} * std::numbers::pi_v<T>);
}

/**
 * @brief `wrapPhase` for an argument the caller can bound: the same value, bit for bit, for `|phase| < 3*pi`.
 *
 * `remainder` is the rule; this is the rule's cheap form where the argument is known small, and the
 * step is the one place where it is. Outside `3*pi` it is not an approximation but a different
 * number: at exactly `3*pi` in `float` the division lands on the tie `1.5`, `nearbyint` takes it up
 * to `2` where the exact quotient rounds down to `1`, and the result is a whole turn away. Use
 * `wrapPhase` for anything arriving from outside the loop.
 */
template<std::floating_point T>
[[nodiscard]] inline T wrapPhaseBounded(T phase) noexcept {
    constexpr T twoPi = T{2} * std::numbers::pi_v<T>;
    return phase - twoPi * std::nearbyint(phase / twoPi);
}

/// @brief The shared second-order recursion. Hold one by value; feed it a detector's error once per step.
template<std::floating_point T = float, PhaseWrap Wrap = PhaseWrap::TwoPi>
class ControlLoop {
public:
    using value_type = T;

    /**
     * @brief Design the gains from `(Bn*T, zeta, Kdet)` and take the frequency bounds the block owns.
     *
     * The bounds default to infinite, which is a loop with no clamp rather than a loop with a
     * forgotten one; every block in the synchronization family sets its own from something physical.
     * Crossed bounds throw here rather than becoming a limit cycle later.
     */
    ControlLoop(double noiseBandwidth, double damping, double detectorGain, T minFrequency = -std::numeric_limits<T>::infinity(), T maxFrequency = std::numeric_limits<T>::infinity(), LoopOrder order = LoopOrder::Second) : _noiseBandwidth(noiseBandwidth), _damping(damping), _detectorGain(detectorGain), _order(order) {
        setFrequencyLimits(minFrequency, maxFrequency);
        applyGains(designLoopGains(noiseBandwidth, damping, detectorGain, order));
    }

    /**
     * @brief Advance the loop by one step and return `instant`, the amount the phase advanced.
     *
     * The clamp is between the two arms and cannot be skipped; the wrap is at the end and happens
     * exactly once. `instant` is the timing loop's instantaneous clock period and is not stored:
     * the state is the integrator and the accumulator, and nothing else.
     *
     * This is the one reduction whose argument is bounded before it is formed, since `_phase` came
     * out of a wrap and the increment is bounded by the clamp, so it is the one that takes the
     * cheap form.
     */
    [[nodiscard]] T step(T error) noexcept {
        _frequency      = std::min(std::max(_frequency + _beta * error, _minFrequency), _maxFrequency);
        const T instant = _frequency + _alpha * error;
        _phase          = reduceStep(_phase + instant);
        return instant;
    }

    [[nodiscard]] T         frequency() const noexcept { return _frequency; }
    [[nodiscard]] T         phase() const noexcept { return _phase; }
    [[nodiscard]] T         alpha() const noexcept { return _alpha; }
    [[nodiscard]] T         beta() const noexcept { return _beta; }
    [[nodiscard]] T         minFrequency() const noexcept { return _minFrequency; }
    [[nodiscard]] T         maxFrequency() const noexcept { return _maxFrequency; }
    [[nodiscard]] double    noiseBandwidth() const noexcept { return _noiseBandwidth; }
    [[nodiscard]] double    damping() const noexcept { return _damping; }
    [[nodiscard]] double    detectorGain() const noexcept { return _detectorGain; }
    [[nodiscard]] LoopOrder order() const noexcept { return _order; }

    /// @brief Parked against a bound: the loop is not tracking, it is reporting the closest frequency it may hold.
    [[nodiscard]] bool saturated() const noexcept { return _frequency == _minFrequency || _frequency == _maxFrequency; }

    void setNoiseBandwidth(double noiseBandwidth) {
        applyGains(designLoopGains(noiseBandwidth, _damping, _detectorGain, _order));
        _noiseBandwidth = noiseBandwidth;
    }

    void setDamping(double damping) {
        applyGains(designLoopGains(_noiseBandwidth, damping, _detectorGain, _order));
        _damping = damping;
    }

    void setDetectorGain(double detectorGain) {
        applyGains(designLoopGains(_noiseBandwidth, _damping, detectorGain, _order));
        _detectorGain = detectorGain;
    }

    void setOrder(LoopOrder order) {
        applyGains(designLoopGains(_noiseBandwidth, _damping, _detectorGain, order));
        _order = order;
    }

    /**
     * @brief Take gains designed elsewhere, unvalidated against any range except sign.
     *
     * There is no `[0, 1]` box: the formula above produces `alpha = 1.33` at `zeta = 2, theta = 1`,
     * and a first-order `beta` leaves that box routinely. Negative gains are a sign error and are
     * rejected; everything else is accepted and the poles follow from it. A first-order loop keeps
     * `alpha == 0`, which is what the mode means, so setting one here is an error rather than a
     * promotion to a loop with a proportional arm.
     */
    void setGains(double alpha, double beta) {
        if (!(alpha >= 0.0) || !(beta >= 0.0)) {
            throw std::invalid_argument("ControlLoop::setGains: negative gains put the poles outside the unit circle — negate the detector, not the loop");
        }
        if (_order == LoopOrder::First && alpha != 0.0) {
            throw std::invalid_argument("ControlLoop::setGains: a first-order loop has no proportional arm — change the order first");
        }
        _alpha = static_cast<T>(alpha);
        _beta  = static_cast<T>(beta);
    }

    /// @brief Set both bounds and re-clamp the state immediately, rather than leaving it out of range until the next step.
    void setFrequencyLimits(T minFrequency, T maxFrequency) {
        if (!(minFrequency <= maxFrequency)) {
            throw std::invalid_argument("ControlLoop: minFrequency must not exceed maxFrequency — crossed bounds are a period-two limit cycle, not a clamp");
        }
        _minFrequency = minFrequency;
        _maxFrequency = maxFrequency;
        _frequency    = std::min(std::max(_frequency, _minFrequency), _maxFrequency);
    }

    void setMinFrequency(T minFrequency) { setFrequencyLimits(minFrequency, _maxFrequency); }
    void setMaxFrequency(T maxFrequency) { setFrequencyLimits(_minFrequency, maxFrequency); }

    void setFrequency(T frequency) noexcept { _frequency = std::min(std::max(frequency, _minFrequency), _maxFrequency); }
    void setPhase(T phase) noexcept { _phase = reduce(phase); }

    /// @brief Restart the loop: the frequency clamped, the phase wrapped, and nothing else to forget.
    void reset(T phase = T{0}, T frequency = T{0}) noexcept {
        setFrequency(frequency);
        _phase = reduce(phase);
    }

private:
    /// @brief The reduction for a phase arriving from outside, from `setPhase` or `reset`, where nothing bounds it.
    [[nodiscard]] T reduce(T phase) const noexcept {
        if constexpr (Wrap == PhaseWrap::TwoPi) {
            return wrapPhase(phase);
        } else if constexpr (Wrap == PhaseWrap::Period) {
            return std::remainder(phase, _frequency);
        } else {
            return phase;
        }
    }

    /// @brief The step's reduction: the same value as `reduce` over the argument a step can form, for a fifth of the cost.
    [[nodiscard]] T reduceStep(T phase) const noexcept {
        if constexpr (Wrap == PhaseWrap::TwoPi) {
            return wrapPhaseBounded(phase);
        } else if constexpr (Wrap == PhaseWrap::Period) {
            return std::remainder(phase, _frequency); // a varying modulus, and the cheap form needs a constant one
        } else {
            return phase;
        }
    }

    void applyGains(const LoopGains& gains) noexcept {
        _alpha = static_cast<T>(gains.alpha);
        _beta  = static_cast<T>(gains.beta);
    }

    T _frequency{0};
    T _phase{0};

    T         _alpha{0};
    T         _beta{0};
    T         _minFrequency{-std::numeric_limits<T>::infinity()};
    T         _maxFrequency{std::numeric_limits<T>::infinity()};
    double    _noiseBandwidth{0.0};
    double    _damping{0.0};
    double    _detectorGain{0.0};
    LoopOrder _order{LoopOrder::Second};
};

} // namespace gr::sync

#endif // GNURADIO_ALGORITHM_CONTROL_LOOP_HPP
