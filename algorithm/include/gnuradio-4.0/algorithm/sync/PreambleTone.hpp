#ifndef GNURADIO_ALGORITHM_PREAMBLE_TONE_HPP
#define GNURADIO_ALGORITHM_PREAMBLE_TONE_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <format>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

/**
 * @brief The symbol phase of a burst, measured from the tone an alternating training sequence leaves.
 *
 * A two-level training sequence of alternating symbols, seen through an FM discriminator, is a real
 * tone at half the symbol rate: with `a_m = (-1)^m` the discriminator's output is the symbol pulse
 * `g` driven by a square wave of period two symbols, so it is periodic with `2*sps` samples and its
 * fundamental sits at
 *
 * ```
 * f0 = 1 / (2 * samplesPerSymbol)          cycles per sample
 * ```
 *
 * The frequency being known, the phase of that tone *is* the symbol phase, and one complex dot
 * product measures it — no loop, no acquisition transient, and an accuracy at the Cramer-Rao bound
 * for a real tone in Gaussian noise. The data-aided estimate and that bound are the standard results;
 * see Mengali, U., and D'Andrea, A. N., *Synchronization Techniques for Digital Receivers*, Plenum
 * Press, 1997.
 *
 * Three responsibilities, and the split between the first two is the whole design:
 *
 * 1. `push` runs a sliding single-bin DFT and returns the fraction of the window's energy lying in
 *    the `f0` bin. It is `O(1)` a sample, by the recursion
 *    `S <- exp(j*w) * (S - v[n-N+1]) + v[n+1] * exp(-j*w*(N-1))`, which sits on the unit circle and
 *    so accumulates rounding without bound. Its only job is to decide *where* to fit.
 * 2. `fit` is an `O(N)` dot product over one named window and is what an estimate is taken from, so
 *    the recursion's drift never reaches a caller's answer.
 * 3. `instantAfter` turns a fitted phase into the symbol lattice, once, so no caller re-derives the
 *    sign.
 *
 * The statistic. Over a window of `N` samples with `X = (2/N) * sum_k v[k] * exp(-j*2*pi*f0*k)`,
 *
 * ```
 * Lambda = (N/2) * |X|^2 / sum_k v[k]^2
 * ```
 *
 * is the fraction of the window's energy in the `f0` bin: one on a pure tone, and invariant to the
 * stream's scale, so a threshold on it needs no AGC and no absolute level. Under `N` samples of
 * Gaussian noise `Lambda` is the ratio of one of `N/2` independent periodogram bins to their sum, so
 * it is `Beta(1, N/2 - 1)` with mean `2/N` and `P(Lambda > t) = (1-t)^(N/2-1)` — which is what turns
 * a wanted false-alarm rate into a threshold.
 *
 * The lattice. The symbol instants are the tone's extrema, `2*pi*f0*(n - n0) + phi = k*pi`, so with
 * `1/(2*f0) = sps`
 *
 * ```
 * n = n0 + (k - phi/pi) * sps            k integer
 * ```
 *
 * and the offset from the window's first sample to an instant is `-(phi/pi)*sps` reduced modulo
 * `sps`. It is positive toward later samples, which is the sense a `time_est` tag carries.
 */
namespace gr::sync {

/// @brief One window's fit: the tone's amplitude, its phase at the window's first sample, and the statistic.
struct PreambleToneFit {
    double amplitude = 0.0; ///< `|X|`, in the stream's own units
    double phase     = 0.0; ///< `arg(X)`, radians, referred to the window's first sample
    double statistic = 0.0; ///< `Lambda`, the fraction of the window's energy in the `f0` bin
};

struct PreambleToneEstimator {
    /**
     * @brief Sets the tone's frequency and the window, and clears the recursion.
     *
     * @param samplesPerSymbol  input samples per symbol; the tone sits at `1/(2*samplesPerSymbol)`
     * @param preambleSymbols   alternating symbols the window spans; `N = round(product)` samples
     */
    void configure(double samplesPerSymbol, double preambleSymbols) {
        if (!(samplesPerSymbol > 1.0) || !std::isfinite(samplesPerSymbol)) {
            throw std::invalid_argument(std::format("gr::sync::PreambleToneEstimator: {} samples per symbol; the tone at half the symbol rate needs more than two samples a period", samplesPerSymbol));
        }
        if (!(preambleSymbols > 0.0) || !std::isfinite(preambleSymbols)) {
            throw std::invalid_argument(std::format("gr::sync::PreambleToneEstimator: {} preamble symbols; the window has to hold at least one", preambleSymbols));
        }
        const auto length = static_cast<std::size_t>(std::llround(preambleSymbols * samplesPerSymbol));
        if (length < 2UZ) {
            throw std::invalid_argument(std::format("gr::sync::PreambleToneEstimator: a window of {} samples cannot carry a period of the tone", length));
        }

        _samplesPerSymbol = samplesPerSymbol;
        _length           = length;
        _frequency        = 0.5 / samplesPerSymbol;

        const double omega = 2.0 * std::numbers::pi * _frequency;
        _step              = std::polar(1.0, omega);
        _entry             = std::polar(1.0, -omega * static_cast<double>(_length - 1UZ));
        _history.assign(_length, 0.0);
        reset();
    }

    /// @brief Forgets the window: the recursion restarts from silence and `filled()` is false again.
    void reset() noexcept {
        std::ranges::fill(_history, 0.0);
        _sum    = {};
        _energy = 0.0;
        _cursor = 0UZ;
        _seen   = 0UZ;
    }

    [[nodiscard]] std::size_t windowLength() const noexcept { return _length; }
    [[nodiscard]] double      frequency() const noexcept { return _frequency; }
    [[nodiscard]] double      samplesPerSymbol() const noexcept { return _samplesPerSymbol; }
    /// @brief Whether `windowLength()` samples have been pushed, which is when the statistic means anything.
    [[nodiscard]] bool filled() const noexcept { return _seen >= _length; }

    /**
     * @brief Slides the window on by one sample and returns `Lambda` over the window that now ends at it.
     *
     * The returned value is zero until the window is full, and zero for a window carrying no energy at
     * all, which is the answer a ratio has where its denominator vanishes.
     */
    [[nodiscard]] double push(double sample) noexcept {
        const double leaving = _history[_cursor];
        _history[_cursor]    = sample;
        _cursor              = _cursor + 1UZ == _length ? 0UZ : _cursor + 1UZ;

        _sum = _step * (_sum - leaving) + _entry * sample;
        _energy += sample * sample - leaving * leaving;
        if (_seen < _length) {
            ++_seen;
        }
        if (_seen < _length || !(_energy > 0.0)) {
            return 0.0;
        }
        return 2.0 * std::norm(_sum) / (static_cast<double>(_length) * _energy);
    }

    /**
     * @brief The exact fit over one window of exactly `windowLength()` samples.
     *
     * The `O(N)` form, not the recursion's running value: an estimate that reaches a tag is always
     * recomputed here, so the marginally stable recursion decides only which window to fit.
     */
    template<std::floating_point TSample>
    [[nodiscard]] PreambleToneFit fit(std::span<const TSample> window) const {
        if (window.size() != _length) {
            throw std::invalid_argument(std::format("gr::sync::PreambleToneEstimator::fit: {} samples against a window of {}", window.size(), _length));
        }
        const double         omega = 2.0 * std::numbers::pi * _frequency;
        std::complex<double> sum{};
        double               energy = 0.0;
        for (std::size_t k = 0UZ; k < _length; ++k) {
            const double value = static_cast<double>(window[k]);
            sum += value * std::polar(1.0, -omega * static_cast<double>(k));
            energy += value * value;
        }

        const std::complex<double> bin = (2.0 / static_cast<double>(_length)) * sum;
        PreambleToneFit            fitted;
        fitted.amplitude = std::abs(bin);
        fitted.phase     = std::arg(bin);
        fitted.statistic = energy > 0.0 ? 2.0 * std::norm(sum) / (static_cast<double>(_length) * energy) : 0.0;
        return fitted;
    }

    /**
     * @brief The first symbol instant at or after @p from, in samples from the window's first sample.
     *
     * The instants are `(k - phase/pi) * samplesPerSymbol` for integer `k`, so this is the one place
     * the sign of the phase becomes a direction in time: positive is later.
     */
    [[nodiscard]] double instantAfter(double phase, double from) const noexcept {
        const double shift = phase / std::numbers::pi; // the lattice's offset, in symbols
        const double index = std::ceil(from / _samplesPerSymbol + shift);
        return (index - shift) * _samplesPerSymbol;
    }

private:
    double               _samplesPerSymbol = 2.0;
    double               _frequency        = 0.25;
    std::size_t          _length           = 2UZ;
    std::complex<double> _step{1.0, 0.0};  ///< `exp(j*w)`: the whole window's advance by one sample
    std::complex<double> _entry{1.0, 0.0}; ///< `exp(-j*w*(N-1))`: the arriving sample's own phase

    /// The window, `_length` samples of it. Two elements by default, matching the default `_length`, so
    /// `push()` is in bounds on a default-constructed estimator rather than only after `configure()`.
    std::vector<double>  _history{0.0, 0.0};
    std::complex<double> _sum{};
    double               _energy = 0.0;
    std::size_t          _cursor = 0UZ;
    std::size_t          _seen   = 0UZ;
};

} // namespace gr::sync

#endif // GNURADIO_ALGORITHM_PREAMBLE_TONE_HPP
