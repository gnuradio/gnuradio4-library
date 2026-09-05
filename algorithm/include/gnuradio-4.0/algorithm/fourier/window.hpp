
#ifndef GNURADIO_ALGORITHM_WINDOW_HPP
#define GNURADIO_ALGORITHM_WINDOW_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <format>
#ifdef __GNUC__
#pragma GCC diagnostic push // ignore warning of external libraries that from this lib-context we do not have any control over
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
#include <magic_enum.hpp>
#include <magic_enum_utility.hpp>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::algorithm::window {

/**
 * Implementation of window function (also known as an apodization function or tapering function).
 * See Wikipedia for more info: https://en.wikipedia.org/wiki/Window_function
 *
 * Kaiser (beta), Tukey (alpha), Gaussian (sigma) and Exponential (decay in dB) read the single scalar
 * `param` of create(); every other window ignores it. See create() for the per-window defaults and ranges.
 */
// HannExp names the Hann window and holds its ordinal, so that a setting stored under that name still resolves and the enumerators after it keep their values.
enum class Type : int { None, Rectangular, Hamming, Hann, HannExp [[deprecated("HannExp is the Hann window")]], Blackman, Nuttall, BlackmanHarris, BlackmanNuttall, FlatTop, Exponential, Kaiser, Bartlett, Welch, Parzen, Tukey, Gaussian };
using enum Type;
inline static constexpr gr::meta::fixed_string TypeNames = "[None, Rectangular, Hamming, Hann, HannExp, Blackman, Nuttall, BlackmanHarris, BlackmanNuttall, FlatTop, Exponential, Kaiser, Bartlett, Welch, Parzen, Tukey, Gaussian]";

namespace detail {
template<typename T>
requires std::is_floating_point_v<T>
constexpr T bessel_i0(const T x) noexcept {
    T   sum  = 1;
    T   term = 1;
    int k    = 1;

    const T x_half = x / 2;

    do {
        term *= (x_half / static_cast<T>(k));
        sum += term * term;
        ++k;
    } while (term * term > sum * std::numeric_limits<T>::epsilon());

    return sum;
}

// the Kaiser default is a mild beta near the low end of the useful range; a caller who wants a stated
// sidelobe level passes kaiserBeta(attenuationDb)
template<typename T>
requires std::is_floating_point_v<T>
[[nodiscard]] constexpr T defaultParameter(Type windowFunction) noexcept {
    switch (windowFunction) {
    case Type::Kaiser: return static_cast<T>(1.6);
    case Type::Tukey: return static_cast<T>(0.5);
    case Type::Gaussian: return static_cast<T>(0.4);
    case Type::Exponential: return static_cast<T>(60);
    default: return static_cast<T>(0);
    }
}

template<typename T>
requires std::is_floating_point_v<T>
void validateParameter(Type windowFunction, T param) {
    switch (windowFunction) {
    case Type::Kaiser:
        if (param < static_cast<T>(0)) {
            throw std::invalid_argument(std::format("Kaiser beta must be non-negative, is {}", param));
        }
        return;
    case Type::Tukey:
        if (param < static_cast<T>(0) || param > static_cast<T>(1)) {
            throw std::invalid_argument(std::format("Tukey alpha must be within [0, 1], is {}", param));
        }
        return;
    case Type::Gaussian:
        if (param <= static_cast<T>(0) || param > static_cast<T>(0.5)) {
            throw std::invalid_argument(std::format("Gaussian sigma must be within (0, 0.5], is {}", param));
        }
        return;
    case Type::Exponential:
        if (param < static_cast<T>(0)) {
            throw std::invalid_argument(std::format("Exponential decay [dB] must be non-negative, is {}", param));
        }
        return;
    default: return;
    }
}

// every window here is symmetric about (n - 1)/2; mirroring the first half instead of evaluating the
// second makes that symmetry bit-exact -- 1 - abs(i - m)/m is not bit-identical at mirrored i -- and
// halves the transcendental count
template<typename T, typename ContainerType, typename TShape>
void fillSymmetric(ContainerType& container, std::size_t n, TShape&& shape) {
    for (std::size_t i = 0UZ; i < (n + 1UZ) / 2UZ; ++i) {
        const T value          = shape(i);
        container[i]           = value;
        container[n - 1UZ - i] = value;
    }
}
} // namespace detail

/**
 * @brief Creates in-place a window function (mathematically aka. 'apodisation function') of a specified type and size.
 *
 * This function generates various window functions used in digital signal processing.
 * See Wikipedia for more info: https://en.wikipedia.org/wiki/Window_function
 *
 * Generation belongs on the settings path, so each angle is evaluated directly; an angle recurrence
 * would drift past the sidelobe floor these windows are specified at.
 *
 * @tparam T The floating-point type to use for the window function values.
 * @param container std::vector or std::array containing the values of the window function.
 * @param windowFunction The type of window function to create.
 * @param param The window's shape parameter: Kaiser beta [0, inf) default 1.6, Tukey alpha [0, 1] default 0.5,
 *              Gaussian sigma (0, 0.5] default 0.4 as a fraction of the half-length, Exponential decay from
 *              center to edge in dB [0, inf) default 60. NaN selects the default; out of range throws.
 */
template<gr::meta::array_or_vector_type ContainerType, typename T = ContainerType::value_type>
requires std::is_floating_point_v<T>
void create(ContainerType& container, Type windowFunction, const T param = std::numeric_limits<T>::quiet_NaN()) {
    constexpr T       pi2 = 2 * std::numbers::pi_v<T>;
    const std::size_t n   = container.size();
    if (n == 0) {
        return;
    }
    const T p = std::isnan(param) ? detail::defaultParameter<T>(windowFunction) : param;
    detail::validateParameter<T>(windowFunction, p);

    // every cosine-sum window scales by 1/(n - 1); Kaiser rejects n <= 1 below
    if (n == 1 && windowFunction != Type::Kaiser) {
        container[0] = static_cast<T>(1);
        return;
    }

    using enum Type;
    switch (windowFunction) {
    case None:
    case Rectangular: {
        std::ranges::fill(container, 1);
        return;
    }
    case Hamming: {
        // formula: w(n) = 0.53836 - 0.46164 * cos((2 * pi * n) / (N - 1))
        // equiripple-optimal coefficients; the textbook 0.54/0.46 rounding raises the peak sidelobe by ~0.5 dB
        const T a = pi2 / static_cast<T>(n - 1);
        detail::fillSymmetric<T>(container, n, [a](const std::size_t i) { return static_cast<T>(0.53836) - static_cast<T>(0.46164) * std::cos(a * static_cast<T>(i)); });
        return;
    }
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    case HannExp:
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    case Hann: {
        // formula: w(n) = 0.5 - 0.5 * cos((2 * pi * n) / (N - 1))
        // reference: von Hann, J. (1901). Über den Durchgang einer elektrischen Welle längs der Erdoberfläche. Elektrische Nachrichtentechnik, 17, 421-424.
        const T a = pi2 / static_cast<T>(n - 1);
        detail::fillSymmetric<T>(container, n, [a](const std::size_t i) { return static_cast<T>(.5) - static_cast<T>(.5) * std::cos(a * static_cast<T>(i)); });
        return;
    }
    case Blackman: {
        // formula: w(n) = 0.42 - 0.5 * cos((2 * pi * n) / (N - 1)) + 0.08 * cos((4 * pi * n) / (N - 1))
        // reference: Blackman, R. B., & Tukey, J. W. (1958). The measurement of power spectra from the point of view of communications engineering. Dover Publications.
        const T a = pi2 / static_cast<T>(n - 1);
        detail::fillSymmetric<T>(container, n, [a](const std::size_t i) {
            const T ai = a * static_cast<T>(i);
            return static_cast<T>(0.42) - static_cast<T>(0.5) * std::cos(ai) + static_cast<T>(0.08) * std::cos(static_cast<T>(2.) * ai);
        });
        return;
    }
    case Nuttall: {
        // Formula: w(n) = a0 - a1 * cos((2 * pi * n) / (N - 1)) + a2 * cos((4 * pi * n) / (N - 1)) - a3 * cos((6 * pi * n) / (N - 1))
        // Reference: Nuttall, A. (1981). Some Windows with Very Good Sidelobe Behavior. IEEE Transactions on Acoustics, Speech, and Signal Processing, 29(1), 84-91.
        const T a = pi2 / static_cast<T>(n - 1);
        detail::fillSymmetric<T>(container, n, [a](const std::size_t i) {
            constexpr std::array<T, 4> coeff = {static_cast<T>(0.355768), static_cast<T>(0.487396), static_cast<T>(0.144232), static_cast<T>(0.012604)};
            const T                    ai    = a * static_cast<T>(i);
            return coeff[0] - coeff[1] * std::cos(ai) + coeff[2] * std::cos(2 * ai) - coeff[3] * std::cos(3 * ai);
        });
        return;
    }
    case BlackmanHarris: {
        // formula: w(n) = a0 - a1 * cos((2 * pi * n) / (N - 1)) + a2 * cos((4 * pi * n) / (N - 1)) - a3 * cos((6 * pi * n) / (N - 1))
        // reference: Harris, F. J. (1978). On the use of windows for harmonic analysis with the discrete Fourier transform. Proceedings of the IEEE, 66(1), 51-83.
        const T a = pi2 / static_cast<T>(n - 1);
        detail::fillSymmetric<T>(container, n, [a](const std::size_t i) {
            constexpr std::array<T, 4> coeff = {static_cast<T>(0.35875), static_cast<T>(0.48829), static_cast<T>(0.14128), static_cast<T>(0.01168)};
            const T                    ai    = a * static_cast<T>(i);
            return coeff[0] - coeff[1] * std::cos(ai) + coeff[2] * std::cos(2 * ai) - coeff[3] * std::cos(3 * ai);
        });
        return;
    }
    case BlackmanNuttall: {
        // formula: w(n) = 0.3635819 - 0.4891775 * cos(2*pi*n/(N-1)) + 0.1365995 * cos(4*pi*n/(N-1)) - 0.0106411 * cos(6*pi*n/(N-1))
        // reference: Generalized from Nuttall, A. (1981) and Harris, F. J. (1978).
        const T a = pi2 / static_cast<T>(n - 1);
        detail::fillSymmetric<T>(container, n, [a](const std::size_t i) {
            const T ai = a * static_cast<T>(i);
            return static_cast<T>(0.3635819) - static_cast<T>(0.4891775) * std::cos(ai) + static_cast<T>(0.1365995) * std::cos(static_cast<T>(2.) * ai) - static_cast<T>(0.0106411) * std::cos(static_cast<T>(3.) * ai);
        });
        return;
    }
    case FlatTop: {
        // formula: w(n) = (a0 - a1 * cos((2 * pi * n) / (N - 1)) + a2 * cos((4 * pi * n) / (N - 1)) - a3 * cos((6 * pi * n) / (N - 1)) + a4 * cos((8 * pi * n) / (N - 1))) / sum(a)
        // a4 = 0.028 makes the coefficients sum to zero at the boundary: peak sidelobe
        // -76.6 dB against -68.9 dB for the a4 = 0.032 variant, at the same scalloping loss
        const T a = pi2 / static_cast<T>(n - 1);
        detail::fillSymmetric<T>(container, n, [a](const std::size_t i) {
            constexpr std::array<T, 5> coeff = {static_cast<T>(1.0), static_cast<T>(1.93), static_cast<T>(1.29), static_cast<T>(0.388), static_cast<T>(0.028)};
            constexpr T                norm  = static_cast<T>(1) / (coeff[0] + coeff[1] + coeff[2] + coeff[3] + coeff[4]);
            const T                    ai    = a * static_cast<T>(i);
            return norm * (coeff[0] - coeff[1] * std::cos(ai) + coeff[2] * std::cos(2 * ai) - coeff[3] * std::cos(3 * ai) + coeff[4] * std::cos(4 * ai));
        });
        return;
    }
    case Exponential: {
        // formula: w(n) = exp(-|n - (N-1)/2| / tau), tau = (N-1)/2 * (8.69/param) -- symmetric Poisson window decaying by `param` dB at the edges
        // reference: Harris, F. J. (1978). On the use of windows for harmonic analysis with the discrete Fourier transform. Proceedings of the IEEE, 66(1), 51-83.
        // its transform has no main-lobe null within tens of bins, which suits spectral apodization but leaves FIR design without a transition-width bound
        const T center = static_cast<T>(n - 1) / static_cast<T>(2);
        const T tau    = center * (static_cast<T>(8.69) / p);
        detail::fillSymmetric<T>(container, n, [center, tau](const std::size_t i) { return std::exp(-std::abs(static_cast<T>(i) - center) / tau); });
        return;
    }
    case Kaiser: {
        // formula: w(n) = I0(beta * sqrt(1 - ((2*n/(N-1)) - 1)^2)) / I0(beta)
        // reference: J. F. Kaiser and R. W. Schafer. On the use of the i0-sinh window for spectrum analysis. IEEE Transactions on Acoustics, Speech, and Signal Processing, 28(1):105–107, 1980.
        if (n <= 1) {
            throw std::invalid_argument("n must be larger than one");
        }

        const T factor = static_cast<T>(1) / static_cast<T>(n - 1);
        const T i0Beta = detail::bessel_i0(p); // Compute the zeroth order modified Bessel function of the first kind for beta
        detail::fillSymmetric<T>(container, n, [p, factor, i0Beta](const std::size_t i) {
            const T term = (static_cast<T>(2 * i) * factor) - static_cast<T>(1);
            return detail::bessel_i0(p * std::sqrt(std::abs(static_cast<T>(1) - term * term))) / i0Beta;
        });
        return;
    }
    case Bartlett: {
        // formula: w(n) = 1 - |n - M| / M, M = (N - 1)/2 -- triangular with both ends exactly zero
        const T m = static_cast<T>(n - 1) / static_cast<T>(2);
        detail::fillSymmetric<T>(container, n, [m](const std::size_t i) { return static_cast<T>(1) - std::abs(static_cast<T>(i) - m) / m; });
        return;
    }
    case Welch: {
        // formula: w(n) = 1 - ((n - M) / M)^2, M = (N - 1)/2 -- parabolic
        const T m = static_cast<T>(n - 1) / static_cast<T>(2);
        detail::fillSymmetric<T>(container, n, [m](const std::size_t i) {
            const T x = (static_cast<T>(i) - m) / m;
            return static_cast<T>(1) - x * x;
        });
        return;
    }
    case Parzen: {
        // formula: r = |n - M| / (N/2), M = (N - 1)/2; w = 1 - 6 r^2 (1 - r) for r <= 1/2, else 2 (1 - r)^3
        // the N/2 denominator leaves the ends at 2/N^3 rather than at zero
        const T m    = static_cast<T>(n - 1) / static_cast<T>(2);
        const T half = static_cast<T>(n) / static_cast<T>(2);
        detail::fillSymmetric<T>(container, n, [m, half](const std::size_t i) {
            const T r = std::abs(static_cast<T>(i) - m) / half;
            const T s = static_cast<T>(1) - r;
            return r <= static_cast<T>(0.5) ? static_cast<T>(1) - static_cast<T>(6) * r * r * s : static_cast<T>(2) * s * s * s;
        });
        return;
    }
    case Tukey: {
        // formula: x = n/(N - 1); w = 0.5 (1 - cos(2 pi x / alpha)) for x < alpha/2, 1 in between, mirrored at the far end
        // alpha == 0 is Rectangular and alpha == 1 is Hann, both exactly
        if (p <= static_cast<T>(0)) {
            std::ranges::fill(container, 1);
            return;
        }
        const T taper = static_cast<T>(n - 1) * p / static_cast<T>(2); // taper length in samples
        const T a     = pi2 / (p * static_cast<T>(n - 1));
        detail::fillSymmetric<T>(container, n, [taper, a](const std::size_t i) { return static_cast<T>(i) < taper ? static_cast<T>(0.5) * (static_cast<T>(1) - std::cos(a * static_cast<T>(i))) : static_cast<T>(1); });
        return;
    }
    case Gaussian: {
        // formula: w(n) = exp(-0.5 ((n - M) / (sigma M))^2), M = (N - 1)/2
        // sigma is a fraction of the half-length rather than a count of samples: the shape -- and with it the sidelobe
        // level and the ENBW -- is then independent of N. Translation from a samples-based setting: sigma = std/M.
        const T m     = static_cast<T>(n - 1) / static_cast<T>(2);
        const T sigma = p * m;
        detail::fillSymmetric<T>(container, n, [m, sigma](const std::size_t i) {
            const T x = (static_cast<T>(i) - m) / sigma;
            return std::exp(static_cast<T>(-0.5) * x * x);
        });
        return;
    }
    }
}

/**
 * @brief Creates a new window function (mathematically aka. 'apodisation function') of a specified type and size.
 *
 * This function generates various window functions used in digital signal processing.
 * See Wikipedia for more info: https://en.wikipedia.org/wiki/Window_function
 *
 * @tparam T The floating-point type to use for the window function values.
 * @param windowFunction The type of window function to create.
 * @param n The size of the window function.
 * @param param The window's shape parameter, see the in-place overload; NaN selects the window's own default.
 * @return A std::vector<T> containing the values of the window function.
 */
template<typename T = float>
requires std::is_floating_point_v<T>
[[nodiscard]] auto create(Type windowFunction, const std::size_t n, const T param = std::numeric_limits<T>::quiet_NaN()) {
    std::vector<T> container(n);
    create(container, windowFunction, param);
    return container;
}

} // namespace gr::algorithm::window

#endif // GNURADIO_ALGORITHM_WINDOW_HPP
