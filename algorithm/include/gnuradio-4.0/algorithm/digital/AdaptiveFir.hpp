#ifndef GNURADIO_ALGORITHM_ADAPTIVE_FIR_HPP
#define GNURADIO_ALGORITHM_ADAPTIVE_FIR_HPP

#include <algorithm>
#include <complex>
#include <concepts>
#include <cstddef>
#include <format>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace gr::digital {

/// The adaptation rules an AdaptiveFir carries.
enum class AdaptiveAlgorithm {
    Lms,  ///< least mean squares against a reference symbol
    Nlms, ///< the same, with the step normalized by the energy in the tap window
    Cma   ///< constant modulus, blind: the reference is the constellation's own modulus statistic
};

[[nodiscard]] inline AdaptiveAlgorithm adaptiveAlgorithmFrom(std::string_view name) {
    if (name == "lms") {
        return AdaptiveAlgorithm::Lms;
    }
    if (name == "nlms") {
        return AdaptiveAlgorithm::Nlms;
    }
    if (name == "cma") {
        return AdaptiveAlgorithm::Cma;
    }
    throw std::invalid_argument(std::format("gr::digital::AdaptiveFir: unknown algorithm '{}'; lms, nlms or cma", name));
}

/**
 * @brief A complex FIR whose taps adapt, by least mean squares against a reference or blindly on the modulus.
 *
 * The filter is `y[k] = sum_i w_i * x[k-i]` over `nTaps` taps and the matching history. One update follows each
 * output:
 *
 *     lms   w_i += mu * e * conj(x[k-i])                       e = reference - y
 *     nlms  the same, with mu divided by the window's energy    mu_eff = mu / (eps + sum |x[k-i]|^2)
 *     cma   w_i += mu * e * conj(x[k-i])                       e = y * (R2 - |y|^2)
 *
 * `R2 = E[|a|^4] / E[|a|^2]` over the transmitted alphabet, which is one for any unit-power constant-modulus
 * constellation. The taps start as a unit spike at the center, so an unadapted filter is a delay of `(nTaps-1)/2`
 * samples rather than silence, and that delay is what a converged equalizer keeps.
 *
 * Adaptation can walk away from a solution when the step is too large for the channel. Rather than stream a growing
 * signal, the filter watches its own tap energy and, past a stated bound, re-initializes to the center spike and
 * says so, leaving the count to whatever owns it.
 *
 * The history is stored twice so the tap window is contiguous whichever sample is newest; nothing about the result
 * depends on how the input was divided into calls.
 */
template<std::floating_point F>
struct AdaptiveFir {
    using value_type = F;
    using Complex    = std::complex<F>;

    /// The longest filter accepted, and odd so a spike has a center. Past this the sweeps that check adaptation stop
    /// being cheap and no consumer asks.
    static constexpr std::size_t kMaxTaps = 127UZ;

    // The default state is a valid one-tap pass-through, so every entry point is in bounds before configure runs.
    std::vector<Complex> _taps{Complex(F{1}, F{0})};     ///< w_0 .. w_{nTaps-1}, newest input first
    std::vector<Complex> _history{Complex{}, Complex{}}; ///< 2*nTaps entries, each sample written to two slots
    std::size_t          _nTaps           = 1UZ;
    std::size_t          _position        = 0UZ; ///< slot holding the newest sample; walks backwards
    AdaptiveAlgorithm    _algorithm       = AdaptiveAlgorithm::Lms;
    double               _stepSize        = 0.01;
    double               _referenceRatio  = 1.;
    double               _divergenceBound = 100.;

    /**
     * @brief Sizes the filter, sets its rule, and initializes the taps to the center spike.
     *
     * @param nTaps            odd, 1 to kMaxTaps; the center-spike delay is (nTaps-1)/2 samples
     * @param algorithm        which update rule to run
     * @param stepSize         mu, in (0, 1)
     * @param referenceRatio   R2, read by the constant-modulus rule alone; must be positive
     * @param divergenceBound  tap energy past which the filter re-initializes
     */
    void configure(std::size_t nTaps, AdaptiveAlgorithm algorithm, double stepSize, double referenceRatio, double divergenceBound = 100.) {
        if (nTaps < 1UZ || nTaps > kMaxTaps || (nTaps % 2UZ) == 0UZ) {
            throw std::invalid_argument(std::format("gr::digital::AdaptiveFir: {} taps; the range is 1 to {} and the count must be odd so the spike has a center", nTaps, kMaxTaps));
        }
        if (!(stepSize > 0.) || !(stepSize < 1.)) {
            throw std::invalid_argument(std::format("gr::digital::AdaptiveFir: step size {} is outside (0, 1)", stepSize));
        }
        if (!(referenceRatio > 0.)) {
            throw std::invalid_argument(std::format("gr::digital::AdaptiveFir: the modulus reference {} must be positive", referenceRatio));
        }
        if (!(divergenceBound > 1.)) {
            throw std::invalid_argument(std::format("gr::digital::AdaptiveFir: the divergence bound {} must exceed the energy of the initial spike", divergenceBound));
        }

        _nTaps           = nTaps;
        _algorithm       = algorithm;
        _stepSize        = stepSize;
        _referenceRatio  = referenceRatio;
        _divergenceBound = divergenceBound;
        _taps.assign(nTaps, Complex{});
        _history.assign(2UZ * nTaps, Complex{});
        reset();
    }

    /// Returns the taps to the center spike and empties the history: the filter is a delay again.
    void reset() noexcept {
        std::ranges::fill(_taps, Complex{});
        _taps[(_nTaps - 1UZ) / 2UZ] = Complex(F{1}, F{0});
        std::ranges::fill(_history, Complex{});
        _position = 0UZ;
    }

    /// Returns the taps to the spike but leaves the history, which is what a divergence recovery wants.
    void respike() noexcept {
        std::ranges::fill(_taps, Complex{});
        _taps[(_nTaps - 1UZ) / 2UZ] = Complex(F{1}, F{0});
    }

    [[nodiscard]] std::span<const Complex> taps() const noexcept { return _taps; }
    [[nodiscard]] std::size_t              size() const noexcept { return _nTaps; }
    [[nodiscard]] std::size_t              groupDelay() const noexcept { return (_nTaps - 1UZ) / 2UZ; }
    [[nodiscard]] AdaptiveAlgorithm        algorithm() const noexcept { return _algorithm; }
    [[nodiscard]] double                   stepSize() const noexcept { return _stepSize; }
    [[nodiscard]] double                   referenceRatio() const noexcept { return _referenceRatio; }

    /// Accepts one input sample; the tap window moves with it.
    void push(Complex sample) noexcept {
        _position                    = _position == 0UZ ? _nTaps - 1UZ : _position - 1UZ;
        _history[_position]          = sample;
        _history[_position + _nTaps] = sample;
    }

    /// The filter's output for the window as it stands, accumulated in double.
    [[nodiscard]] Complex output() const noexcept {
        double re = 0.;
        double im = 0.;
        for (std::size_t i = 0UZ; i < _nTaps; ++i) {
            const double wr = static_cast<double>(_taps[i].real());
            const double wi = static_cast<double>(_taps[i].imag());
            const double xr = static_cast<double>(_history[_position + i].real());
            const double xi = static_cast<double>(_history[_position + i].imag());
            re += wr * xr - wi * xi;
            im += wr * xi + wi * xr;
        }
        return Complex(static_cast<F>(re), static_cast<F>(im));
    }

    /**
     * @brief Applies one update and reports whether the taps had to be re-initialized.
     *
     * @param y          the filter output the update is built from
     * @param reference  the symbol the output is compared against; the constant-modulus rule does not read it
     */
    [[nodiscard]] bool adapt(Complex y, Complex reference) noexcept {
        double errorRe = 0.;
        double errorIm = 0.;
        double step    = _stepSize;

        if (_algorithm == AdaptiveAlgorithm::Cma) {
            const double yr    = static_cast<double>(y.real());
            const double yi    = static_cast<double>(y.imag());
            const double slack = _referenceRatio - (yr * yr + yi * yi);
            errorRe            = yr * slack;
            errorIm            = yi * slack;
        } else {
            errorRe = static_cast<double>(reference.real()) - static_cast<double>(y.real());
            errorIm = static_cast<double>(reference.imag()) - static_cast<double>(y.imag());
            if (_algorithm == AdaptiveAlgorithm::Nlms) {
                double energy = 0.;
                for (std::size_t i = 0UZ; i < _nTaps; ++i) {
                    const double xr = static_cast<double>(_history[_position + i].real());
                    const double xi = static_cast<double>(_history[_position + i].imag());
                    energy += xr * xr + xi * xi;
                }
                step = _stepSize / (1e-10 + energy);
            }
        }

        const double gainRe = step * errorRe;
        const double gainIm = step * errorIm;
        double       energy = 0.;
        for (std::size_t i = 0UZ; i < _nTaps; ++i) {
            const double xr = static_cast<double>(_history[_position + i].real());
            const double xi = static_cast<double>(_history[_position + i].imag());
            // the conjugate of the input, so the update walks against the gradient of |e|^2
            const double wr = static_cast<double>(_taps[i].real()) + gainRe * xr + gainIm * xi;
            const double wi = static_cast<double>(_taps[i].imag()) + gainIm * xr - gainRe * xi;
            _taps[i]        = Complex(static_cast<F>(wr), static_cast<F>(wi));
            energy += wr * wr + wi * wi;
        }

        if (!(energy <= _divergenceBound)) { // the negation also catches a NaN
            respike();
            return true;
        }
        return false;
    }

    /// The energy in the taps, which is what the divergence bound is stated against.
    [[nodiscard]] double tapEnergy() const noexcept {
        double energy = 0.;
        for (const Complex& tap : _taps) {
            const double re = static_cast<double>(tap.real());
            const double im = static_cast<double>(tap.imag());
            energy += re * re + im * im;
        }
        return energy;
    }
};

/// `R2 = E[|a|^4] / E[|a|^2]` over an alphabet, the modulus statistic the constant-modulus rule drives toward.
template<std::floating_point F>
[[nodiscard]] double modulusReference(std::span<const std::complex<F>> alphabet) {
    if (alphabet.empty()) {
        throw std::invalid_argument("gr::digital::modulusReference: the alphabet is empty");
    }
    double second = 0.;
    double fourth = 0.;
    for (const std::complex<F>& point : alphabet) {
        const double re      = static_cast<double>(point.real());
        const double im      = static_cast<double>(point.imag());
        const double squared = re * re + im * im;
        second += squared;
        fourth += squared * squared;
    }
    if (!(second > 0.)) {
        throw std::invalid_argument("gr::digital::modulusReference: the alphabet carries no power");
    }
    return fourth / second;
}

} // namespace gr::digital

#endif // GNURADIO_ALGORITHM_ADAPTIVE_FIR_HPP
