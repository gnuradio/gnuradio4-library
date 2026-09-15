#ifndef GNURADIO_ALGORITHM_CPM_PULSE_HPP
#define GNURADIO_ALGORITHM_CPM_PULSE_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <format>
#include <numbers>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/FilterDesign.hpp>

namespace gr::digital {

/// The frequency-pulse families continuous-phase modulation is built from.
enum class CpmPulseShape {
    Rect,         ///< uniform over the pulse's whole span; one symbol long it is CPFSK, and on two levels classic FSK
    RaisedCosine, ///< a raised cosine spanning the whole pulse, the partial-response form
    Gaussian      ///< a Gaussian of bandwidth-symbol-time product bt convolved with one symbol of rectangle: GFSK
};

[[nodiscard]] inline CpmPulseShape cpmPulseShapeFrom(std::string_view name) {
    if (name == "rect") {
        return CpmPulseShape::Rect;
    }
    if (name == "raised_cosine") {
        return CpmPulseShape::RaisedCosine;
    }
    if (name == "gaussian") {
        return CpmPulseShape::Gaussian;
    }
    throw std::invalid_argument(std::format("gr::digital::CpmPulse: unknown pulse '{}'; rect, raised_cosine or gaussian", name));
}

/**
 * @brief The frequency pulse of a continuous-phase modulator, and the convolution that turns symbols into
 * per-sample phase increments.
 *
 * Symbols arrive on the odd PAM grid `±1, ±3, …, ±(M-1)`. At `samplesPerSymbol` samples per symbol the phase
 * increment of sample `k` is
 *
 *     dphi[k] = 2*pi*h * sum_m a_m * g[k - m*samplesPerSymbol]
 *
 * for modulation index `h` and frequency pulse `g` normalized so `sum_k g[k] = 1/2`. That normalization is the
 * convention every published modulation index assumes: one symbol of amplitude `a` turns the carrier by exactly
 * `pi*h*a` radians, whatever the pulse's shape or length. Feeding the increments to a phase accumulator gives a
 * constant-envelope signal whose phase is continuous by construction.
 *
 * A pulse spanning more than one symbol overlaps its neighbors, so the increments of a symbol depend on the
 * `symbolSpan - 1` symbols before it. Those are the only state: the increment stream is a pure function of the
 * carried history and the symbols handed in, so splitting a symbol stream anywhere produces bit-identical output.
 * The pulse is causal, each symbol's own span beginning at the first sample it is offered.
 */
template<std::floating_point T>
struct CpmPulse {
    using value_type = T;

    /// The longest pulse and the deepest history: beyond eight symbols the intersymbol interference is no longer
    /// what any consumer of this modulation asks for, and the sweeps that check it stop being cheap.
    static constexpr std::size_t kMaxSymbolSpan = 8UZ;

    /// The largest modulation index accepted. Indices above 1 are used - Bluetooth's is 0.32, GMSK's 0.5, some
    /// telemetry links run above 1 - but four rotations of the carrier per symbol is past any of them.
    static constexpr double kMaxModulationIndex = 4.;

    std::vector<double>                _pulse{};   ///< g, normalized to sum 1/2
    std::vector<double>                _taps{};    ///< 2*pi*h*g, the form the convolution uses
    std::array<double, kMaxSymbolSpan> _history{}; ///< _history[i] is the symbol i places back, newest first
    std::size_t                        _samplesPerSymbol = 1UZ;
    std::size_t                        _symbolSpan       = 1UZ;
    double                             _modulationIndex  = 0.;

    /**
     * @brief Designs the frequency pulse and clears the symbol history.
     *
     * @param shape             which family the pulse comes from
     * @param symbolSpan        L, the symbols the pulse spans; 1 is full response
     * @param samplesPerSymbol  output samples per symbol, at least 2
     * @param modulationIndex   h, in (0, kMaxModulationIndex]
     * @param bt                bandwidth-symbol-time product, read only by the Gaussian family
     */
    void configure(CpmPulseShape shape, std::size_t symbolSpan, std::size_t samplesPerSymbol, double modulationIndex, double bt) {
        if (samplesPerSymbol < 2UZ) {
            throw std::invalid_argument(std::format("gr::digital::CpmPulse: {} samples per symbol; two is the fewest a phase trajectory can be drawn on", samplesPerSymbol));
        }
        if (symbolSpan < 1UZ || symbolSpan > kMaxSymbolSpan) {
            throw std::invalid_argument(std::format("gr::digital::CpmPulse: pulse spans {} symbols; the range is 1 to {}", symbolSpan, kMaxSymbolSpan));
        }
        if (!(modulationIndex > 0.) || !(modulationIndex <= kMaxModulationIndex)) {
            throw std::invalid_argument(std::format("gr::digital::CpmPulse: modulation index {} is outside (0, {}]", modulationIndex, kMaxModulationIndex));
        }
        if (shape == CpmPulseShape::Gaussian && !(bt > 0.)) {
            throw std::invalid_argument(std::format("gr::digital::CpmPulse: bandwidth-symbol-time product {} must be positive", bt));
        }

        _samplesPerSymbol = samplesPerSymbol;
        _symbolSpan       = symbolSpan;
        _modulationIndex  = modulationIndex;
        _pulse            = buildPulse(shape, symbolSpan, samplesPerSymbol, bt);

        _taps.resize(_pulse.size());
        const double scale = 2. * std::numbers::pi_v<double> * modulationIndex;
        std::ranges::transform(_pulse, _taps.begin(), [scale](double v) { return scale * v; });
        reset();
    }

    /// Forgets the symbols the next increments would have overlapped: the stream restarts from silence.
    void reset() noexcept { _history.fill(0.); }

    [[nodiscard]] std::span<const double> pulse() const noexcept { return _pulse; }
    [[nodiscard]] std::size_t             samplesPerSymbol() const noexcept { return _samplesPerSymbol; }
    [[nodiscard]] std::size_t             symbolSpan() const noexcept { return _symbolSpan; }
    [[nodiscard]] double                  modulationIndex() const noexcept { return _modulationIndex; }

    /// The phase one symbol of the given amplitude contributes in total, whatever the pulse spreads it over.
    [[nodiscard]] double symbolPhase(double symbol) const noexcept { return std::numbers::pi_v<double> * _modulationIndex * symbol; }

    /**
     * @brief Converts symbols to per-sample phase increments, samplesPerSymbol of them per symbol.
     *
     * Consumes as many symbols as both spans allow, leaves the history holding the last of them, and returns how many
     * it took. A caller reading fewer than it offered has to carry the rest to the next call, because the history the
     * increments are built from has already moved on.
     */
    std::size_t incrementsFor(std::span<const T> symbols, std::span<double> increments) noexcept {
        if (_taps.empty()) {
            return 0UZ;
        }
        const std::size_t sps      = _samplesPerSymbol;
        const std::size_t span     = _symbolSpan;
        const std::size_t nSymbols = std::min(symbols.size(), increments.size() / sps);

        for (std::size_t s = 0UZ; s < nSymbols; ++s) {
            for (std::size_t i = span - 1UZ; i > 0UZ; --i) {
                _history[i] = _history[i - 1UZ];
            }
            _history[0] = static_cast<double>(symbols[s]);

            for (std::size_t p = 0UZ; p < sps; ++p) {
                double weighted = 0.;
                for (std::size_t i = 0UZ; i < span; ++i) {
                    weighted += _history[i] * _taps[i * sps + p];
                }
                increments[s * sps + p] = weighted;
            }
        }
        return nSymbols;
    }

private:
    [[nodiscard]] static std::vector<double> buildPulse(CpmPulseShape shape, std::size_t symbolSpan, std::size_t samplesPerSymbol, double bt) {
        const std::size_t   nTaps = symbolSpan * samplesPerSymbol;
        std::vector<double> g(nTaps);

        switch (shape) {
        case CpmPulseShape::Rect: std::ranges::fill(g, 1.); break;
        case CpmPulseShape::RaisedCosine:
            // Sampled at the midpoint of each sample interval rather than at its left edge, which is where the
            // rectangle and the Gaussian already sit. The left-edge grid would put a zero tap at the start, none at
            // the end, and the pulse's center half a sample after the grid's.
            for (std::size_t k = 0UZ; k < nTaps; ++k) {
                g[k] = 1. - std::cos(2. * std::numbers::pi_v<double> * (static_cast<double>(k) + 0.5) / static_cast<double>(nTaps));
            }
            break;
        case CpmPulseShape::Gaussian: {
            // The Gaussian spans the pulse less the one symbol the rectangle adds back, so the moving sum below comes
            // out at exactly nTaps. Its taps sit on a grid centered on the pulse's own center, on a whole sample or
            // between two as the length parity requires, which is what keeps the result symmetric for every span and
            // rate. At a span of one symbol the Gaussian is a single tap and the pulse is the rectangle alone.
            const std::size_t   nGaussian = (symbolSpan - 1UZ) * samplesPerSymbol + 1UZ;
            const double        sigma     = gr::filter::fir::design::gaussianSigma(bt);
            const double        center    = 0.5 * static_cast<double>(nGaussian - 1UZ);
            std::vector<double> gaussian(nGaussian);
            for (std::size_t i = 0UZ; i < nGaussian; ++i) {
                const double t = (static_cast<double>(i) - center) / (static_cast<double>(samplesPerSymbol) * sigma);
                gaussian[i]    = std::exp(-0.5 * t * t);
            }

            for (std::size_t k = 0UZ; k < nTaps; ++k) {
                double moving = 0.;
                for (std::size_t j = 0UZ; j < samplesPerSymbol && j <= k; ++j) {
                    const std::size_t i = k - j;
                    if (i < nGaussian) {
                        moving += gaussian[i];
                    }
                }
                g[k] = moving;
            }
            break;
        }
        }

        const double total = std::accumulate(g.begin(), g.end(), 0.);
        if (!(total > 0.)) {
            throw std::invalid_argument("gr::digital::CpmPulse: the designed pulse carries no area to normalize");
        }
        const double norm = 1. / (2. * total);
        std::ranges::transform(g, g.begin(), [norm](double v) { return norm * v; });
        return g;
    }
};

} // namespace gr::digital

#endif // GNURADIO_ALGORITHM_CPM_PULSE_HPP
