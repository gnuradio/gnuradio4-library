#ifndef GNURADIO_ALGORITHM_MMSE_INTERPOLATOR_HPP
#define GNURADIO_ALGORITHM_MMSE_INTERPOLATOR_HPP

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/FilterDesign.hpp>

/**
 * @brief The minimum-mean-square-error fractional-delay interpolator, generated in closed form from
 * its own normal equations.
 *
 * Given a contiguous window of `L` input samples `x[i] .. x[i+L-1]` and a fraction `mu` in `[0, 1)`,
 * it estimates the underlying continuous signal at
 *
 * ```
 * position = i + L/2 - 1 + mu           absolute, in input samples
 * ```
 *
 * For `L = 8` that is `i + 3 + mu`: the window's center pair is `x[i+3]` and `x[i+4]`, and `mu`
 * runs from the first of them to the second. Booking the sample one input later places every tag
 * derived from it one sample late.
 *
 * The objective. Writing `d_k = k - L/2` for `k = 0 .. L-1`, so `d` runs `-L/2 .. L/2-1`, the filter
 * is `y = sum_k h_k x[m - d_k]` approximating `x(m + mu)`, and the taps minimize
 *
 * ```
 * J(h) = integral over |w| <= 2*pi*B of | exp(j*w*mu) - sum_k h_k exp(-j*w*d_k) |^2 dw
 * ```
 *
 * with `B` the one-sided band of interest as a fraction of the sample rate. `J` is a
 * positive-definite quadratic form and its minimizer solves an `L x L` linear system whose entries
 * are elementary integrals:
 *
 * ```
 * integral over |w| <= W of cos(w*a) dw  =  2*W*sinc(W*a/pi)        W = 2*pi*B
 *
 * J / (2*W)  =  1 - 2*h^T p + h^T R h        R[k][l] = sinc(2*B*(d_k - d_l))
 *                                            p[k]    = sinc(2*B*(mu + d_k))
 * R h = p
 * ```
 *
 * `R` is real, symmetric, positive definite and Toeplitz, and it does not depend on `mu`, so one
 * factorization serves the whole bank. Its condition number at `B = 0.25` leaves about eleven
 * significant digits in a `double` solve and too few in a `float` one: compute in `double`, store
 * `float`. Two structural identities catch an index error: `h(1 - mu) == reverse(h(mu))`, and `h(0)`
 * is a unit impulse at index `L/2`.
 *
 * Outside the design band the response attenuates in magnitude only, the fit being symmetric, so a
 * caller has to check that the signal's own band edge, `(1+rolloff)/(2*sps)`, sits inside `B`. At
 * `sps = 2, rolloff = 0.35` it does not quite, which costs a fraction of a dB of band-edge droop;
 * `L = 8` at `B = 0.25` is a clean choice from `sps = 2.5` up. That droop is the first thing to check
 * when a two-samples-per-symbol receiver has an unexplained error-vector floor. The remedies, in
 * order: run at `sps >= 2.5`; raise `nTaps` to 12; or use a matched-filter polyphase bank designed at
 * the actual signal bandwidth. Raising `B` to `1/3` buys `sps = 2` about a third of a dB and costs
 * more than 20 dB of in-band accuracy everywhere else.
 *
 * The differentiator comes from the same objective with the ideal response replaced by that of a
 * band-limited differentiator at the same fractional advance, `H(w) = j*w*exp(j*w*mu)`. The
 * quadratic term is unchanged, so the same factorization serves:
 *
 * ```
 * R'[k][l] = 2*sin(W*(d_k - d_l)) / (d_k - d_l)   =  2*W * R[k][l]
 * q[k]     = -2*[ sin(W*a_k)/a_k^2 - W*cos(W*a_k)/a_k ]          a_k = mu + d_k
 * R h_d    = q / (2*W)
 * ```
 *
 * `h_d` is antisymmetric only at `mu = 0.5`, the sample window `m - d_k` spanning
 * `m - L/2 + 1 .. m + L/2`, centered half a sample above `m`; the identity that holds everywhere is
 * `h_d(1 - mu) == -reverse(h_d(mu))`. The fit leaves `H(0)` free, so the differentiator keeps a
 * residual DC response of order `1e-3`, which is the bound to assert on `sum h_d`. Its scale is the
 * derivative per input sample; a detector that needs it per symbol multiplies by the current period.
 *
 * The bank is generated at construction, on a uniform grid of `nSteps + 1` values of `mu` from 0 to 1
 * inclusive, selected by `row`, as `nSteps + 1` solves against the prefactorized `R`. `L` and `B` are
 * therefore parameters rather than constants.
 *
 * Quantizing `mu` to the grid is itself a timing error of up to `1/(2*nSteps)` samples, which at the
 * default 128 steps is below the timing loop's own jitter; do not raise it without a measurement. A
 * `mu` landing exactly halfway between two rows is at that bound whichever row it is given, so `row`
 * is free to pick the tie rule that stays clear of a libm call.
 *
 * Rows are stored reversed, so `interpolate` walks the window forward from `x[i]` and the tap index
 * needs no arithmetic. The accumulation order is a function of `L` alone, never of how many outputs a
 * call produces, so a caller's output is bit-identical under any chunking. The derivative bank very
 * nearly doubles the cost per output and is generated only when a detector asks for it.
 *
 * @see Meyr, H., Moeneclaey, M., Fechtel, S. L., Digital Communication Receivers: Synchronization,
 *      Channel Estimation and Signal Processing, Wiley, 1998, equation 9-7, the mean-squared
 *      frequency-response error criterion above.
 */
namespace gr::sync {

/// @brief The sample types an interpolation is written for. Taps are `float` in every case.
template<typename T>
concept InterpolatorSample = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;

namespace detail {

template<typename T>
struct InterpolatorScalar {
    using type = T;
};

template<typename T>
struct InterpolatorScalar<std::complex<T>> {
    using type = T;
};

/**
 * @brief `LDL^T` of a small symmetric positive-definite matrix, factored once and solved many times.
 *
 * `R` does not depend on `mu`, so one factorization is amortized over the whole bank and over both
 * right-hand sides.
 */
class SymmetricFactorization {
public:
    SymmetricFactorization(std::span<const double> matrix, std::size_t order) : _order(order), _lower(order * order, 0.0), _diagonal(order, 0.0) {
        for (std::size_t j = 0UZ; j < order; ++j) {
            double pivot = matrix[j * order + j];
            for (std::size_t k = 0UZ; k < j; ++k) {
                pivot -= _lower[j * order + k] * _lower[j * order + k] * _diagonal[k];
            }
            if (!(pivot > 0.0)) {
                throw std::invalid_argument("SymmetricFactorization: the matrix is not positive definite");
            }
            _diagonal[j]          = pivot;
            _lower[j * order + j] = 1.0;
            for (std::size_t i = j + 1UZ; i < order; ++i) {
                double off = matrix[i * order + j];
                for (std::size_t k = 0UZ; k < j; ++k) {
                    off -= _lower[i * order + k] * _lower[j * order + k] * _diagonal[k];
                }
                _lower[i * order + j] = off / pivot;
            }
        }
    }

    void solve(std::span<const double> rhs, std::span<double> out) const {
        for (std::size_t i = 0UZ; i < _order; ++i) {
            double value = rhs[i];
            for (std::size_t k = 0UZ; k < i; ++k) {
                value -= _lower[i * _order + k] * out[k];
            }
            out[i] = value;
        }
        for (std::size_t i = 0UZ; i < _order; ++i) {
            out[i] /= _diagonal[i];
        }
        for (std::size_t i = _order; i-- > 0UZ;) {
            double value = out[i];
            for (std::size_t k = i + 1UZ; k < _order; ++k) {
                value -= _lower[k * _order + i] * out[k];
            }
            out[i] = value;
        }
    }

    [[nodiscard]] std::size_t order() const noexcept { return _order; }

private:
    std::size_t         _order;
    std::vector<double> _lower;
    std::vector<double> _diagonal;
};

/**
 * @brief `integral over |w| <= W of w sin(w a) dw / 2`, i.e. `sin(W a)/a^2 - W cos(W a)/a`.
 *
 * Removable at `a = 0`, where both terms are `W/a` and the difference is `W^3 a/3 - W^5 a^3/30`.
 * The guard is reached in normal use: `a_k = mu + d_k` is exactly zero at `k = L/2` for `mu = 0`,
 * and every bank includes `mu = 0`.
 */
[[nodiscard]] inline double sineMoment(double bandRadians, double a) noexcept {
    const double w = bandRadians;
    if (std::abs(w * a) < 1.0e-3) {
        const double cube = w * w * w;
        return cube * a / 3.0 - cube * w * w * a * a * a / 30.0;
    }
    return std::sin(w * a) / (a * a) - w * std::cos(w * a) / a;
}

} // namespace detail

/// @brief `R[k][l] = sinc(2*B*(d_k - d_l))`, row-major and `nTaps` square. Independent of `mu`.
[[nodiscard]] inline std::vector<double> mmseCorrelation(int nTaps, double band) {
    if (nTaps < 2 || (nTaps % 2) != 0) {
        throw std::invalid_argument("mmseCorrelation: nTaps must be even and at least 2 — the window straddles the interpolation instant");
    }
    if (!(band > 0.0) || !(band < 0.5)) {
        throw std::invalid_argument("mmseCorrelation: band must lie in (0, 0.5)");
    }
    const std::size_t   order = static_cast<std::size_t>(nTaps);
    std::vector<double> matrix(order * order);
    for (std::size_t k = 0UZ; k < order; ++k) {
        for (std::size_t l = 0UZ; l < order; ++l) {
            matrix[k * order + l] = gr::filter::fir::design::sincPi(2.0 * band * (static_cast<double>(k) - static_cast<double>(l)));
        }
    }
    return matrix;
}

/// @brief `p[k] = sinc(2*B*(mu + d_k))`, the fractional-advance right-hand side.
[[nodiscard]] inline std::vector<double> mmseTarget(int nTaps, double band, double mu) {
    const std::size_t   order  = static_cast<std::size_t>(nTaps);
    const double        center = 0.5 * static_cast<double>(nTaps);
    std::vector<double> target(order);
    for (std::size_t k = 0UZ; k < order; ++k) {
        target[k] = gr::filter::fir::design::sincPi(2.0 * band * (mu + static_cast<double>(k) - center));
    }
    return target;
}

/// @brief `q[k] / (2*W)`, the band-limited-differentiator right-hand side scaled onto the same `R`.
[[nodiscard]] inline std::vector<double> mmseDerivativeTarget(int nTaps, double band, double mu) {
    const std::size_t   order   = static_cast<std::size_t>(nTaps);
    const double        center  = 0.5 * static_cast<double>(nTaps);
    const double        radians = 2.0 * std::numbers::pi * band;
    std::vector<double> target(order);
    for (std::size_t k = 0UZ; k < order; ++k) {
        const double a = mu + static_cast<double>(k) - center;
        target[k]      = -2.0 * detail::sineMoment(radians, a) / (2.0 * radians);
    }
    return target;
}

/// @brief The least-squares fractional-advance taps, `h`, in the natural order `k = 0 .. L-1`.
[[nodiscard]] inline std::vector<double> mmseTaps(int nTaps, double band, double mu) {
    const std::vector<double>            matrix = mmseCorrelation(nTaps, band);
    const detail::SymmetricFactorization factored(matrix, static_cast<std::size_t>(nTaps));
    std::vector<double>                  taps(static_cast<std::size_t>(nTaps));
    factored.solve(mmseTarget(nTaps, band, mu), taps);
    return taps;
}

/// @brief The interpolating differentiator's taps, scaled as a derivative per input sample.
[[nodiscard]] inline std::vector<double> mmseDifferentiatorTaps(int nTaps, double band, double mu) {
    const std::vector<double>            matrix = mmseCorrelation(nTaps, band);
    const detail::SymmetricFactorization factored(matrix, static_cast<std::size_t>(nTaps));
    std::vector<double>                  taps(static_cast<std::size_t>(nTaps));
    factored.solve(mmseDerivativeTarget(nTaps, band, mu), taps);
    return taps;
}

/// @brief `J(h) = 2*W*(1 - 2*h^T p + h^T R h)`, the objective the taps minimize, at its own optimum or anywhere else.
[[nodiscard]] inline double mmseObjective(std::span<const double> taps, double band, double mu) {
    const int                 nTaps  = static_cast<int>(taps.size());
    const std::vector<double> matrix = mmseCorrelation(nTaps, band);
    const std::vector<double> target = mmseTarget(nTaps, band, mu);

    double quadratic = 0.0;
    double linear    = 0.0;
    for (std::size_t k = 0UZ; k < taps.size(); ++k) {
        linear += taps[k] * target[k];
        for (std::size_t l = 0UZ; l < taps.size(); ++l) {
            quadratic += taps[k] * matrix[k * taps.size() + l] * taps[l];
        }
    }
    return 2.0 * (2.0 * std::numbers::pi * band) * (1.0 - 2.0 * linear + quadratic);
}

/**
 * @brief The tap bank, generated at construction and selected by `row`.
 *
 * Rows are stored reversed against `mmseTaps`, so a dot product walks the window forward from
 * `x[i]`: `interpolate(window, mu) = sum_j row[j] * window[j]` estimates the signal at
 * `i + L/2 - 1 + mu`.
 */
class MmseInterpolatorBank {
public:
    /**
     * @param nTaps          `L`, even; 8 is the default, 6 and 12 the other useful choices
     * @param nSteps         the `mu` grid, a power of two; the bank holds `nSteps + 1` rows
     * @param band           `B`, the one-sided design band as a fraction of the sample rate
     * @param withDerivative also generate the interpolating differentiator, which the two
     *                       maximum-likelihood timing detectors need and nothing else does
     */
    explicit MmseInterpolatorBank(int nTaps = 8, int nSteps = 128, double band = 0.25, bool withDerivative = true) : _nTaps(static_cast<std::size_t>(nTaps)), _nSteps(static_cast<std::size_t>(nSteps)), _band(band) {
        if (nSteps < 1 || (nSteps & (nSteps - 1)) != 0) {
            throw std::invalid_argument("MmseInterpolatorBank: nSteps must be a positive power of two");
        }
        const std::vector<double>            matrix = mmseCorrelation(nTaps, band);
        const detail::SymmetricFactorization factored(matrix, _nTaps);

        const std::size_t rows = _nSteps + 1UZ;
        _taps.resize(rows * _nTaps);
        if (withDerivative) {
            _derivativeTaps.resize(rows * _nTaps);
        }

        std::vector<double> solution(_nTaps);
        for (std::size_t row = 0UZ; row < rows; ++row) {
            const double mu = static_cast<double>(row) / static_cast<double>(_nSteps);

            factored.solve(mmseTarget(nTaps, band, mu), solution);
            for (std::size_t k = 0UZ; k < _nTaps; ++k) {
                _taps[row * _nTaps + k] = static_cast<float>(solution[_nTaps - 1UZ - k]);
            }
            if (withDerivative) {
                factored.solve(mmseDerivativeTarget(nTaps, band, mu), solution);
                for (std::size_t k = 0UZ; k < _nTaps; ++k) {
                    _derivativeTaps[row * _nTaps + k] = static_cast<float>(solution[_nTaps - 1UZ - k]);
                }
            }
        }
    }

    /**
     * @brief The row a fraction selects: `mu * nSteps` to nearest, ties up.
     *
     * Adding a half and truncating inlines, where `std::nearbyint` has to honor the dynamic rounding
     * mode and stays an out-of-line libm call at every `-march`. The two agree bit for bit except at
     * a tie, where they name the two rows that are equally far away; `nSteps` is a power of two, so
     * the scaling is exact and ties are reachable.
     *
     * `mu` outside `[0, 1]` is a caller error and is clamped, not wrapped; NaN selects row 0. The
     * clamp compares in `double` so that an arbitrarily large `mu` never reaches the conversion.
     */
    [[nodiscard]] std::size_t row(double mu) const noexcept {
        const double steps  = static_cast<double>(_nSteps);
        const double scaled = mu * steps + 0.5;
        if (!(scaled > 0.0)) {
            return 0UZ;
        }
        if (!(scaled < steps + 1.0)) {
            return _nSteps;
        }
        return static_cast<std::size_t>(static_cast<int>(scaled));
    }

    [[nodiscard]] std::span<const float> tapsFor(std::size_t row) const noexcept { return {_taps.data() + row * _nTaps, _nTaps}; }
    [[nodiscard]] std::span<const float> derivativeTapsFor(std::size_t row) const noexcept { return {_derivativeTaps.data() + row * _nTaps, _nTaps}; }

    /// @brief The signal at `i + L/2 - 1 + mu`, from `window = x[i .. i+L-1]`.
    template<InterpolatorSample T>
    [[nodiscard]] T interpolate(const T* window, std::size_t row) const noexcept {
        using Scalar            = typename detail::InterpolatorScalar<T>::type;
        const float* const taps = _taps.data() + row * _nTaps;
        T                  accumulator{};
        for (std::size_t k = 0UZ; k < _nTaps; ++k) {
            accumulator += static_cast<Scalar>(taps[k]) * window[k];
        }
        return accumulator;
    }

    /// @brief The derivative per input sample at the same instant, from the same window.
    template<InterpolatorSample T>
    [[nodiscard]] T differentiate(const T* window, std::size_t row) const noexcept {
        using Scalar            = typename detail::InterpolatorScalar<T>::type;
        const float* const taps = _derivativeTaps.data() + row * _nTaps;
        T                  accumulator{};
        for (std::size_t k = 0UZ; k < _nTaps; ++k) {
            accumulator += static_cast<Scalar>(taps[k]) * window[k];
        }
        return accumulator;
    }

    [[nodiscard]] std::size_t size() const noexcept { return _nTaps; }
    [[nodiscard]] std::size_t steps() const noexcept { return _nSteps; }
    [[nodiscard]] double      band() const noexcept { return _band; }
    [[nodiscard]] bool        hasDerivative() const noexcept { return !_derivativeTaps.empty(); }

    /// @brief The window index the interpolated sample sits at when `mu` is zero: `L/2 - 1`.
    [[nodiscard]] std::size_t delay() const noexcept { return _nTaps / 2UZ - 1UZ; }

private:
    std::size_t        _nTaps;
    std::size_t        _nSteps;
    double             _band;
    std::vector<float> _taps;
    std::vector<float> _derivativeTaps;
};

} // namespace gr::sync

#endif // GNURADIO_ALGORITHM_MMSE_INTERPOLATOR_HPP
