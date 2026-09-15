#ifndef GNURADIO_ALGORITHM_BAND_EDGE_FILTER_HPP
#define GNURADIO_ALGORITHM_BAND_EDGE_FILTER_HPP

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/FilterDesign.hpp>

/**
 * @brief The band-edge filter pair and its power discriminant: a carrier-frequency error estimate
 * that needs no carrier, no data decisions and no symbol timing.
 *
 * A signal shaped by a root-raised-cosine of excess bandwidth `alpha` and symbol period `T` is flat
 * out to `(1-alpha)/(2T)`, falls to zero at `(1+alpha)/(2T)`, and is symmetric about zero. An
 * offset in frequency breaks that symmetry by an amount monotone in the offset over roughly the
 * transition band. Two matched filters sitting on the two edges measure the break. The taps follow
 * from the derivation:
 *
 * 1. The discriminant is the matched filter's derivative in frequency. What changes when the signal
 *    shifts is the energy at the edges, so the sensitivity is `dH/df`: zero wherever the response
 *    is flat, nonzero only across the transition.
 * 2. Across the transition the amplitude is a quarter cosine, `cos((pi*T/(2*alpha))*(|f| -
 *    (1-alpha)/(2T)))`, falling from 1 to 0. Its derivative is a quarter sine, zero at the inner
 *    edge and largest at the outer edge `(1+alpha)/(2T)`.
 * 3. Mirror that quarter sine about the outer edge and it is a half cycle: a half cosine centered on
 *    the band edge, of total width `2*alpha/T`.
 * 4. Invert it. A half cosine `cos(pi*(f-f0)/W)` over `|f-f0| <= W/2` transforms to
 *    `(W/2)*[sinc(W*t - 1/2) + sinc(W*t + 1/2)]*exp(j*2*pi*f0*t)`, with `sinc(u) = sin(pi u)/(pi u)`.
 *    The bracket is a real, symmetric, low-pass prototype; the exponential puts it on the edge.
 *
 * With `t` in samples, `W = 2*alpha/sps` cycles per sample and `f0 = (1+alpha)/(2*sps)`:
 *
 * ```
 * n        = i - (N-1)/2                                   i = 0 .. N-1
 * b[n]     = sinc(2*alpha*n/sps - 1/2) + sinc(2*alpha*n/sps + 1/2)
 * b        = b / sum(b^2)                                  see normalization below
 * upper[n] = b[n] * exp(+j*2*pi*f0*n)
 * lower[n] = conj(upper[n])
 * ```
 *
 * The `-3 dB` width of a half cosine of total width `W` is `W/2`, so the filters peak at
 * `(1+alpha)/(2*sps)` and are `alpha/sps` wide, the width converging on `alpha/sps` from above as
 * `N` grows.
 *
 * The grid is centered, for any `N` and any `sps`. Both the prototype and the spin are evaluated on
 * `n = i - (N-1)/2` in floating point, so `N` need not be odd and `sps` need not divide anything.
 * Building the prototype on a grid whose center is `rint(N/sps)` and spinning it on one centered at
 * the integer `(N-1)/2` gives the same result only when `N` is odd and `(N-1)/sps` is an integer;
 * elsewhere the prototype loses its symmetry, by up to 60% of the peak tap at tap counts an ordinary
 * caller would choose. That does not bias the discriminant, since the two tap vectors are elementwise
 * conjugates and `|H_upper(f)| = |H_lower(-f)|` whatever the prototype's symmetry, but it costs the
 * phase linearity a band-edge discriminant rests on, and with it error variance.
 *
 * Normalization: the prototype is divided by the sum of its squared taps, by its energy rather than
 * by its root energy, which makes the detector gain independent of `alpha` and of the tap count and
 * equal to a value a caller can write down,
 *
 * ```
 * Kdet = sps * P            error units per cycle/sample of offset
 *      = sps * P / (2*pi)   error units per rad/sample, the units the loop gains are designed in
 * ```
 *
 * to within 1.2% over `sps` from 2 to 8, `alpha` from 0.22 to 0.5 and `N` from 33 to 89, with `P`
 * the mean input power. `Kdet` is proportional to that power, so a block built on this requires an
 * AGC in front of it. `normalizedDiscriminant` serves a caller who cannot promise one: bounded in
 * `[-1, 1]` and independent of the input power, at the cost of a gain that depends on `sps`, `alpha`
 * and `N` together and has to be measured per design.
 *
 * The sign. The error is `|yu|^2 - |yl|^2`: a signal that has drifted up puts more energy in the
 * upper filter, so the error is positive and `Kdet` is positive, as the loop kernel requires of
 * every detector in this family. A block closing the loop on it therefore derotates by
 * `exp(-j*phase)`.
 *
 * The S-curve in that sign is odd, close to linear inside about a third of the pull-in range, and
 * flat beyond `(1+alpha)/(2*sps)`, which is where a frequency clamp belongs;
 * `bandEdgeFrequencyLimit` returns it in rad/sample. A wider clamp lets a loop wander into the flat
 * part of its own S-curve and sit there.
 *
 * Cost. The direct form is two complex-tap FIRs over a complex stream, `2*4*N` real multiplies per
 * sample, which dominates everything else a frequency-locked loop does. Because
 * `upper[n] = b[n]*exp(+j*2*pi*f0*n)` with `b` real,
 *
 * ```
 * |sum_n upper[n] * y[m-n]|  ==  |sum_n b[n] * y'[m-n]|      with  y'[k] = y[k]*exp(-j*2*pi*f0*k)
 * ```
 *
 * The residual `exp(-j*2*pi*f0*m)` is common to the whole sum and only the magnitude is wanted, so
 * rotating the stream up and down by `f0` and filtering each with the real prototype gives the same
 * output for about a third less work. The two conjugate rotations share their four products, and a
 * recursive rotator is admissible here, unlike in a carrier loop, because `f0` is a constant.
 * `BandEdgeForm::ComplexTaps` is the reference the QA holds `RealTaps` against; `RealTaps` is the
 * default. Even so this is the most expensive stage in a synchronization chain, so a receiver able to
 * confine a frequency-locked loop to acquisition should do so.
 *
 * The rotator carries its phase in `double` and is renormalized by one Newton step every 256 samples.
 * A `float` rotator would also hold the identity above, its differential phase error across the
 * window random-walking to a few parts in `1e7` over `1e5` samples, but `double` costs four
 * multiplies against `4*N` `float` ones, so the drift is bought out cheaply.
 *
 * Each history is a doubled ring, every sample written twice, at `w` and `w+N`, so the window handed
 * to a dot product is always a contiguous run of exactly `N` in the same order. That fixes the
 * accumulation order at `N` alone, whatever a call carries, so the output is bit-identical under any
 * chunking. Redesign on a parameter change is `O(N)` and belongs in the settings path.
 *
 * @see harris, f. j., Multirate Signal Processing for Communication Systems, on the band-edge
 *      frequency-locked loop: the band-edge filters as the derivative of the matched filter's
 *      frequency response, the half-wave approximation to that derivative, and the power-difference
 *      discriminant. The step-by-step inversion above is this header's own working of it.
 */
namespace gr::sync {

/// @brief Complex taps and two filters, or one rotator and real taps: the same magnitudes, about 1.6x apart in cost.
enum class BandEdgeForm { RealTaps, ComplexTaps };

/// @brief A designed pair, plus the two numbers a caller needs to check it against the derivation.
struct BandEdgeFilters {
    std::vector<float>               prototype;            /// `b`, real, symmetric about `(N-1)/2`, energy-normalized
    std::vector<std::complex<float>> upper;                /// `b` spun up to `+center`
    std::vector<std::complex<float>> lower;                /// `conj(upper)`, exactly
    double                           center{0.0};          /// `f0 = (1+alpha)/(2*sps)`, cycles/sample
    double                           prototypeEnergy{0.0}; /// `sum(b^2)` before normalization, the divisor
};

/// @brief The band edge the filters sit on, `(1+alpha)/(2*sps)`, in cycles per sample.
[[nodiscard]] inline constexpr double bandEdgeCenter(double samplesPerSymbol, double rolloff) noexcept { return (1.0 + rolloff) / (2.0 * samplesPerSymbol); }

/// @brief The discriminant's own pull-in range as a frequency clamp, `pi*(1+alpha)/sps` rad/sample.
[[nodiscard]] inline constexpr double bandEdgeFrequencyLimit(double samplesPerSymbol, double rolloff) noexcept { return std::numbers::pi * (1.0 + rolloff) / samplesPerSymbol; }

/// @brief `Kdet = sps * P` in error units per cycle/sample of offset, with `P` the mean input power an AGC has to hold.
[[nodiscard]] inline constexpr double bandEdgeDetectorGain(double samplesPerSymbol, double power = 1.0) noexcept { return samplesPerSymbol * power; }

/// @brief The same gain in the units `ControlLoop` designs against, error per rad/sample: `sps*P/(2*pi)`.
[[nodiscard]] inline constexpr double bandEdgeDetectorGainPerRadian(double samplesPerSymbol, double power = 1.0) noexcept { return samplesPerSymbol * power / (2.0 * std::numbers::pi); }

/**
 * @brief The real, symmetric, energy-normalized prototype `b`, on the centered grid.
 *
 * @param nTaps            `N`; need not be odd, unlike every design in `FilterDesign.hpp`, because the
 *                         linear phase here comes from symmetry about `(N-1)/2` and that holds for even
 *                         `N` with the center falling between taps
 * @param samplesPerSymbol `sps`; need not be an integer and need not divide `N-1`
 * @param rolloff          `alpha` in `[0, 1]`. At exactly zero the transition band has no width, the
 *                         prototype degenerates to the constant `4/pi`, and what is left is a
 *                         length-`N` sinc at the band edge rather than a band-edge filter
 * @param energy           receives `sum(b^2)` before normalization, the divisor
 */
[[nodiscard]] inline std::vector<double> bandEdgePrototype(int nTaps, double samplesPerSymbol, double rolloff, double* energy = nullptr) {
    if (nTaps < 3) {
        throw std::invalid_argument("bandEdgePrototype: nTaps must be at least 3");
    }
    if (!(samplesPerSymbol > 0.0)) {
        throw std::invalid_argument("bandEdgePrototype: samplesPerSymbol must be positive");
    }
    if (!(rolloff >= 0.0) || !(rolloff <= 1.0)) {
        throw std::invalid_argument("bandEdgePrototype: rolloff must lie in [0, 1]");
    }

    const std::size_t len    = static_cast<std::size_t>(nTaps);
    const double      center = 0.5 * static_cast<double>(nTaps - 1);
    const double      slope  = 2.0 * rolloff / samplesPerSymbol;

    // Every tap is evaluated from its own `n`, never mirrored from its partner: the symmetry the
    // discriminant needs is then a property of the grid that a test can fail, not of the fill order.
    std::vector<double> b(len);
    double              sumSquares = 0.0;
    for (std::size_t i = 0UZ; i < len; ++i) {
        const double n = static_cast<double>(i) - center;
        b[i]           = gr::filter::fir::design::sincPi(slope * n - 0.5) + gr::filter::fir::design::sincPi(slope * n + 0.5);
        sumSquares += b[i] * b[i];
    }
    if (energy != nullptr) {
        *energy = sumSquares;
    }
    for (double& tap : b) {
        tap /= sumSquares;
    }
    return b;
}

/// @brief The prototype and both spins. `lower` is `conj(upper)` elementwise and exactly.
[[nodiscard]] inline BandEdgeFilters designBandEdgeFilters(int nTaps, double samplesPerSymbol, double rolloff) {
    BandEdgeFilters           out;
    const std::vector<double> b = bandEdgePrototype(nTaps, samplesPerSymbol, rolloff, &out.prototypeEnergy);

    out.center             = bandEdgeCenter(samplesPerSymbol, rolloff);
    const std::size_t len  = b.size();
    const double      turn = 2.0 * std::numbers::pi * out.center;
    const double      mid  = 0.5 * static_cast<double>(nTaps - 1);

    out.prototype.resize(len);
    out.upper.resize(len);
    out.lower.resize(len);
    for (std::size_t i = 0UZ; i < len; ++i) {
        const double               n    = static_cast<double>(i) - mid;
        const std::complex<double> spin = b[i] * std::polar(1.0, turn * n);
        out.prototype[i]                = static_cast<float>(b[i]);
        out.upper[i]                    = {static_cast<float>(spin.real()), static_cast<float>(spin.imag())};
        out.lower[i]                    = std::conj(out.upper[i]);
    }
    return out;
}

/// @brief The two band-edge powers of one sample, `|yu|^2` and `|yl|^2`, from which every error form is built.
struct BandEdgePowers {
    float upper = 0.0f;
    float lower = 0.0f;
};

/// @brief `|yu|^2 - |yl|^2`: positive above the carrier, so `Kdet` is positive and the derotation carries the minus sign.
[[nodiscard]] inline constexpr float discriminant(BandEdgePowers powers) noexcept { return powers.upper - powers.lower; }

/**
 * @brief The same difference divided by the sum: bounded in `[-1, 1]` and independent of the input power.
 *
 * One divide per sample removes the AGC precondition and costs the `Kdet = sps*P` rule: the
 * normalized gain depends on `sps`, `alpha` and `N` together and has to be measured per design.
 */
[[nodiscard]] inline constexpr float normalizedDiscriminant(BandEdgePowers powers, float epsilon = 1.0e-20f) noexcept { return (powers.upper - powers.lower) / (powers.upper + powers.lower + epsilon); }

/**
 * @brief Both band-edge filters, run over one stream, one sample at a time.
 *
 * The input is the derotated signal: a loop closes around this, so it measures the residual offset
 * of the block's own output and its `Kdet` is the small-signal gain about lock.
 */
template<BandEdgeForm Form = BandEdgeForm::RealTaps>
class BandEdgeDiscriminant {
public:
    using value_type = std::complex<float>;

    /// @brief Samples between renormalizations of the recursive rotator; a fixed interval, not a magnitude test.
    static constexpr std::size_t kRenormalizeInterval = 256UZ;

    BandEdgeDiscriminant(int nTaps, double samplesPerSymbol, double rolloff) : _filters(designBandEdgeFilters(nTaps, samplesPerSymbol, rolloff)), _length(_filters.prototype.size()) {
        _increment = std::polar(1.0, -2.0 * std::numbers::pi * _filters.center);
        if constexpr (Form == BandEdgeForm::RealTaps) {
            // Reversed, so the dot product walks the window oldest to newest. `b` is symmetric, so the
            // reversal is an identity; it is written out anyway, because relying on it silently lets an
            // index error survive a symmetric test case.
            _taps.assign(_filters.prototype.rbegin(), _filters.prototype.rend());
        } else {
            _upperTaps.assign(_filters.upper.rbegin(), _filters.upper.rend());
            _lowerTaps.assign(_filters.lower.rbegin(), _filters.lower.rend());
        }
        reset();
    }

    /// @brief One sample in, two band-edge powers out. No branch depends on how many samples the caller has.
    [[nodiscard]] BandEdgePowers step(value_type sample) noexcept {
        std::complex<float> yu{};
        std::complex<float> yl{};

        if constexpr (Form == BandEdgeForm::RealTaps) {
            const float rotorRe = static_cast<float>(_rotor.real());
            const float rotorIm = static_cast<float>(_rotor.imag());
            const float ac      = sample.real() * rotorRe;
            const float bd      = sample.imag() * rotorIm;
            const float ad      = sample.real() * rotorIm;
            const float bc      = sample.imag() * rotorRe;

            push(_upperWindow, {ac - bd, ad + bc}); // sample * exp(-j*2*pi*f0*k)
            push(_lowerWindow, {ac + bd, bc - ad}); // sample * exp(+j*2*pi*f0*k)

            const std::complex<float>* const upperRun = _upperWindow.data() + _write + 1UZ;
            const std::complex<float>* const lowerRun = _lowerWindow.data() + _write + 1UZ;
            for (std::size_t k = 0UZ; k < _length; ++k) {
                yu += _taps[k] * upperRun[k];
                yl += _taps[k] * lowerRun[k];
            }
            advanceRotor();
        } else {
            push(_upperWindow, sample);
            const std::complex<float>* const run = _upperWindow.data() + _write + 1UZ;
            for (std::size_t k = 0UZ; k < _length; ++k) {
                yu += _upperTaps[k] * run[k];
                yl += _lowerTaps[k] * run[k];
            }
        }
        advanceWrite();
        return {std::norm(yu), std::norm(yl)};
    }

    /// @brief Drop the history and restart the rotator at zero phase. Nothing else is state.
    void reset() noexcept {
        _upperWindow.assign(2UZ * _length, value_type{});
        if constexpr (Form == BandEdgeForm::RealTaps) {
            _lowerWindow.assign(2UZ * _length, value_type{});
        }
        _write = 0UZ;
        _rotor = {1.0, 0.0};
        _since = 0UZ;
    }

    [[nodiscard]] const BandEdgeFilters& filters() const noexcept { return _filters; }
    [[nodiscard]] std::size_t            size() const noexcept { return _length; }
    [[nodiscard]] double                 center() const noexcept { return _filters.center; }

private:
    void push(std::vector<value_type>& window, value_type v) noexcept {
        window[_write]           = v;
        window[_write + _length] = v;
    }

    void advanceWrite() noexcept {
        ++_write;
        if (_write == _length) {
            _write = 0UZ;
        }
    }

    void advanceRotor() noexcept {
        _rotor *= _increment;
        ++_since;
        if (_since == kRenormalizeInterval) {
            _rotor *= 1.5 - 0.5 * std::norm(_rotor); // one Newton step towards unit modulus
            _since = 0UZ;
        }
    }

    BandEdgeFilters                  _filters;
    std::size_t                      _length{0UZ};
    std::vector<float>               _taps;      /// `RealTaps` only
    std::vector<std::complex<float>> _upperTaps; /// `ComplexTaps` only
    std::vector<std::complex<float>> _lowerTaps; /// `ComplexTaps` only

    std::vector<value_type> _upperWindow; /// the whole history in `ComplexTaps`, the up-rotated one in `RealTaps`
    std::vector<value_type> _lowerWindow; /// `RealTaps` only
    std::size_t             _write{0UZ};

    std::complex<double> _rotor{1.0, 0.0};
    std::complex<double> _increment{1.0, 0.0};
    std::size_t          _since{0UZ};
};

} // namespace gr::sync

#endif // GNURADIO_ALGORITHM_BAND_EDGE_FILTER_HPP
