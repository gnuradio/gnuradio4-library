#ifndef GNURADIO_ALGORITHM_FILTER_DESIGN_HPP
#define GNURADIO_ALGORITHM_FILTER_DESIGN_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <format>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/FilterTool.hpp>
#include <gnuradio-4.0/algorithm/fourier/window.hpp>

/**
 * @brief Windowed-sinc FIR design, and measurement of what a design delivers.
 *
 * Design and verification are separate. `kaiserLowpass` computes a tap set for a length, a cutoff
 * and an attenuation. `scanLowpass` reads the response of a tap set and reports what it delivers,
 * and `searchLowpass` uses that to find the shortest length that meets a target.
 *
 * The search costs a response scan per candidate; a caller with a deadline uses
 * `kaiserLength`'s estimate instead.
 *
 * Design is setup path. Nothing here runs per sample: taps are computed when a setting changes.
 * The cost is O(N) transcendental calls for a design and O(N * grid) for a scan. The hot path is
 * whatever consumes the taps, where every extra tap is a multiply-accumulate per output sample.
 * All internal evaluation is `double`; the returned taps are `float`, rounded once.
 */
namespace gr::filter::fir::design {

/**
 * @brief Default number of points a response is read on over a full turn.
 *
 * A tap set's finest feature is about one over its length wide in normalized frequency, so this
 * leaves several hundred points across the narrowest lobe of a filter of a hundred taps. A grid
 * too coarse for the length steps over the stopband peaks it is meant to find and reports a
 * filter as clearing a target it does not meet: keep the grid above about thirty-two times the
 * tap count.
 */
inline constexpr std::size_t kDesignGrid = 1UZ << 15;

/**
 * @brief Kaiser's length estimate for a transition @p width cycles per sample wide.
 *
 * `fir::estimateNumberOfTapsKaiser` is the same estimate stated in radians per sample; this converts
 * the unit and applies the floor of three taps every design here needs. The estimate marks where a
 * design first touches its target rather than where it clears it, and `searchLowpass` gives the exact
 * length.
 *
 * Measured against the shortest length that meets the target, at cutoff 0.25: the estimate falls
 * between 0.0 and 1.8 dB short, needing 0 to 54 extra taps. A design that must meet a stated
 * attenuation runs `searchLowpass`; a design with a deadline takes this estimate and accepts up
 * to two dB.
 */
[[nodiscard]] inline std::size_t kaiserLength(double attenDb, double width) {
    // checked in double: a width that is not positive, an attenuation the formula cannot resolve, and
    // anything that is not a number at all never reach the conversion to std::size_t
    if (!(width > 0.0) || !(attenDb > 8.0)) {
        return 3UZ;
    }
    return std::max(estimateNumberOfTapsKaiser(attenDb, 2.0 * std::numbers::pi * width), 3UZ);
}

/**
 * @brief The length used for a requested one: rounded up to odd, never below the floor.
 *
 * Odd length gives type-I linear phase with a group delay of exactly (N-1)/2 whole samples, which
 * keeps a cascade's total delay an integer number of input samples. A type-II design has
 * `H(fs/2) = 0` identically, so a high-pass cannot be normalized at Nyquist and cannot exist at
 * even length. Odd length also lets the second half be copied rather than re-evaluated, which
 * makes the taps symmetric to the bit.
 *
 * @param floorTaps 3 in general, 5 for the band designs, which need a center tap and one pair
 */
[[nodiscard]] inline constexpr std::size_t oddLength(std::size_t requested, std::size_t floorTaps = 3UZ) noexcept {
    const std::size_t n = std::max(requested, floorTaps);
    return ((n % 2UZ) == 0UZ) ? n + 1UZ : n;
}

/// @brief The window a design is shaped by: a type and the single shape parameter that type reads.
struct WindowSpec {
    gr::algorithm::window::Type type  = gr::algorithm::window::Type::Kaiser;
    double                      param = std::numeric_limits<double>::quiet_NaN(); /// NaN selects the window's own default
};

/// @brief A Kaiser window shaped for @p attenDb; Kaiser is the only window whose attenuation is a parameter.
[[nodiscard]] inline WindowSpec kaiserFor(double attenDb) { return {gr::algorithm::window::Type::Kaiser, kaiserBeta(attenDb)}; }

/**
 * @brief The measured figures of a window without a shape parameter.
 *
 * With any window but Kaiser the stopband attenuation is a property of the window, not a parameter: only
 * the length is free, and it buys transition width. The entry points therefore return the attenuation
 * instead of accepting it.
 *
 * `transitionConstant` is the length-transition product C_w, the measured transition width in
 * cycles per sample multiplied by the length, so a design needing width `df` needs C_w/df taps.
 * `attenuationDb` is the stopband attenuation of a windowed-sinc lowpass built with the window,
 * reported positive. Both are measured figures at N = 1023, from the window set's own
 * characterization.
 *
 * A single `A * fs / (22 * df)` length rule, common in older design tools, is short against C_w by
 * 1.3x to 3.2x depending on the window: a design built that way reaches its tabled attenuation with
 * a transition band that much wider than the caller asked for. Lengths here follow C_w, and are
 * correspondingly larger.
 */
struct WindowFigures {
    double transitionConstant = 0.0;
    double attenuationDb      = 0.0;
};

/// @brief The measured figures of a window whose shape is fixed. Throws for the four that take a shape parameter.
[[nodiscard]] inline WindowFigures windowFigures(gr::algorithm::window::Type type) {
    using enum gr::algorithm::window::Type;
    switch (type) {
    case None:
    case Rectangular: return {0.920, 20.96};
    case Bartlett: return {3.762, 26.26};
    case Welch: return {1.866, 31.38};
    case Hann: return {3.127, 43.94};
    case Hamming: return {3.319, 53.40};
    case Parzen: return {7.860, 56.64};
    case Blackman: return {5.593, 75.29};
    case FlatTop: return {9.150, 95.70};
    case BlackmanHarris: return {7.838, 109.29};
    case Nuttall: return {7.757, 111.96};
    case BlackmanNuttall: return {7.819, 114.90};
    default: break;
    }
    throw std::invalid_argument("FilterDesign: Kaiser, Tukey, Gaussian and Exponential have a shape-dependent transition constant — state the tap count, or use kaiserLength for Kaiser");
}

/// @brief The length a fixed-shape window needs for a transition @p width cycles per sample wide.
[[nodiscard]] inline std::size_t windowLength(gr::algorithm::window::Type type, double width) {
    if (width <= 0.0) {
        return 3UZ;
    }
    const double estimate = std::ceil(windowFigures(type).transitionConstant / width);
    return estimate > 3.0 ? static_cast<std::size_t>(estimate) : 3UZ;
}

/**
 * @brief A transition band stated as its two band edges rather than as a midpoint and a width.
 *
 * A stated cutoff is the -6 dB point of the design, the sinc's own argument, not the passband
 * edge. It is the midpoint of the transition band, so passing the stated passband edge as a
 * cutoff leaves the filter 6 dB down at the edge of the band it claims. Both conventions are in
 * use, so both are offered; this converts one into the other, in whatever unit both share.
 */
struct TransitionBand {
    double cutoff = 0.0;
    double width  = 0.0;
};

/// @brief The -6 dB point and full width of the transition between a pass edge and a stop edge, either order.
[[nodiscard]] inline constexpr TransitionBand fromEdges(double passEdge, double stopEdge) noexcept { return {0.5 * (passEdge + stopEdge), (stopEdge > passEdge) ? stopEdge - passEdge : passEdge - stopEdge}; }

/**
 * @brief A Kaiser-windowed lowpass of odd length at unit DC gain.
 *
 * Odd length makes the group delay a whole sample, which keeps a cascade's total delay an exact
 * integer number of input samples. The second half is copied from the first rather than evaluated
 * again, so the taps are symmetric to the bit and the response may be read as a cosine sum.
 *
 * @param n      requested length; rounded up to the next odd value, and never below three
 * @param cutoff the -6 dB point in cycles per sample, the sinc's own argument, not the passband
 *               edge. Passing the passband edge here leaves the filter 6 dB down at the edge of
 *               the band it claims.
 * @param attenDb stopband attenuation the Kaiser window is shaped for
 */
[[nodiscard]] inline std::vector<float> kaiserLowpass(std::size_t n, double cutoff, double attenDb) {
    constexpr double pi = std::numbers::pi;
    if ((n % 2UZ) == 0UZ) {
        ++n;
    }
    n                      = std::max(n, 3UZ);
    const std::size_t mid  = (n - 1UZ) / 2UZ;
    const double      beta = kaiserBeta(attenDb);
    const double      i0b  = gr::algorithm::window::besselI0(beta);

    std::vector<double> h(n);
    for (std::size_t i = 0UZ; i <= mid; ++i) {
        const double t = static_cast<double>(i) - static_cast<double>(mid);
        const double s = (i == mid) ? 2.0 * cutoff : std::sin(2.0 * pi * cutoff * t) / (pi * t);
        const double r = t / static_cast<double>(mid);
        const double w = gr::algorithm::window::besselI0(beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / i0b;

        h[i]           = s * w;
        h[n - 1UZ - i] = h[i];
    }

    double sum = 0.0;
    for (const double v : h) {
        sum += v;
    }
    std::vector<float> out(n);
    for (std::size_t i = 0UZ; i < n; ++i) {
        out[i] = static_cast<float>(h[i] / sum);
    }
    return out;
}

namespace detail {

/// @brief How many designs one memo holds; a new entry past that drops the oldest.
inline constexpr std::size_t kDesignMemoEntries = 8UZ;

/// @brief How many grid sizes `cosineTable` holds; a caller alternating between two sizes still hits.
inline constexpr std::size_t kCosineTableGrids = 2UZ;

/**
 * @brief cos(2*pi*j/grid) for every j, built once per grid size and held for the last two sizes.
 *
 * At the default grid the table is 32768 doubles, 256 kB, which is worth keeping across calls but not
 * worth keeping one of per size ever asked for: a caller sweeping the grid would leave every table
 * alive for the life of the process. The table is handed out as a shared pointer rather than as a
 * reference because an eviction on one thread would otherwise dangle a reference another thread is
 * part way through reading.
 */
[[nodiscard]] inline std::shared_ptr<const std::vector<double>> cosineTable(std::size_t grid) {
    static std::mutex                                                                      mutex;
    static std::vector<std::pair<std::size_t, std::shared_ptr<const std::vector<double>>>> tables;

    const std::lock_guard<std::mutex> lock(mutex);
    for (const auto& [size, table] : tables) {
        if (size == grid) {
            return table;
        }
    }

    std::vector<double> built(grid);
    for (std::size_t j = 0UZ; j < grid; ++j) {
        built[j] = std::cos(2.0 * std::numbers::pi * static_cast<double>(j) / static_cast<double>(grid));
    }
    if (tables.size() >= kCosineTableGrids) {
        tables.erase(tables.begin());
    }
    tables.emplace_back(grid, std::make_shared<const std::vector<double>>(std::move(built)));
    return tables.back().second;
}

/**
 * @brief Publish @p computed under @p key, returning whatever the memo ends up holding for that key.
 *
 * A design's look-up releases the lock while the design runs, so two callers asking for one key both
 * compute it. This re-checks under the lock and returns the entry the other caller published, so every
 * caller of one key sees one value and the memo never holds a duplicate the linear scan cannot reach.
 * The memo is bounded at @p Capacity entries, oldest dropped, which is what stops a process sweeping a
 * rate from growing without limit.
 */
template<std::size_t Capacity, typename Key, typename Value>
[[nodiscard]] inline Value keepDesign(std::mutex& mutex, std::vector<std::pair<Key, Value>>& memo, const Key& key, Value computed) {
    const std::lock_guard<std::mutex> lock(mutex);
    for (const auto& [cached, value] : memo) {
        if (cached == key) {
            return value;
        }
    }
    if (memo.size() >= Capacity) {
        memo.erase(memo.begin());
    }
    memo.emplace_back(key, std::move(computed));
    return memo.back().second;
}

} // namespace detail

/**
 * @brief The signed zero-phase amplitude A at i / grid cycles per sample, for i over half a turn.
 *
 * For symmetric taps of odd length `N = 2M+1`, `A(w) = h[M] + 2 * sum_k h[M+k] cos(k w)` is real
 * and `H(e^{jw}) = e^{-jwM} A(w)`. The sign is kept because normalization needs it: a design
 * normalized at Nyquist against `abs(A)` would silently flip.
 *
 * Each cosine is read out of the table at an index stepped in integer arithmetic, so the result
 * is exact. A running phasor would accumulate error over tens of thousands of
 * steps, which is inside the depth a search has to resolve.
 *
 * The index step is taken modulo the grid by mask, so the grid must be a power of two, and one that is not
 * is refused.
 *
 * A tap set of no taps has no center tap to read and no amplitude anywhere: the result is the grid
 * filled with zero, which is the reading `halfResponseOdd` already gives an empty tap set.
 */
inline void halfAmplitude(const std::vector<float>& taps, std::vector<double>& amp, std::size_t grid = kDesignGrid) {
    if (grid == 0UZ || (grid & (grid - 1UZ)) != 0UZ) {
        throw std::invalid_argument(std::format("FilterDesign: the index step is taken modulo the grid by mask, so the grid must be a power of two, got {}", grid));
    }
    const std::size_t n    = taps.size();
    const std::size_t half = grid / 2UZ;
    if (n == 0UZ) {
        amp.assign(half + 1UZ, 0.0);
        return;
    }
    const std::size_t mid = (n - 1UZ) / 2UZ;
    amp.assign(half + 1UZ, static_cast<double>(taps[mid]));

    const std::shared_ptr<const std::vector<double>> table = detail::cosineTable(grid);
    const std::vector<double>&                       cs    = *table;
    for (std::size_t k = 1UZ; k <= mid; ++k) {
        const double c   = 2.0 * static_cast<double>(taps[mid - k]);
        std::size_t  idx = 0UZ;
        for (std::size_t i = 0UZ; i <= half; ++i) {
            amp[i] += c * cs[idx];
            idx = (idx + k) & (grid - 1UZ);
        }
    }
}

/// @brief |A| at i / grid cycles per sample, for i over half a turn.
inline void halfResponse(const std::vector<float>& taps, std::vector<double>& mag, std::size_t grid = kDesignGrid) {
    halfAmplitude(taps, mag, grid);
    for (double& v : mag) {
        v = std::abs(v);
    }
}

/**
 * @brief A(2*pi*f) at one stated frequency, read by `gr::filter::signedAmplitudeAt`.
 *
 * Signed, which is why every normalization in this header is expressed as a value of A at a frequency
 * rather than as a magnitude off a grid: `calculateResponse<Normalised, Magnitude>` returns |A|, and a
 * design normalized against that at a frequency where A is negative comes out inverted.
 *
 * `H = e^{-jwM} A`, so the raw alternating sum `sum_n h[n] (-1)^n` is `(-1)^M A(pi)`: a
 * Nyquist-normalized design has `abs(sum_n h[n] (-1)^n) == gain`, the sign following M's parity.
 */
template<typename T>
requires std::is_floating_point_v<T>
[[nodiscard]] inline double amplitudeAt(const std::vector<T>& taps, double f) {
    const gr::filter::FilterCoefficients<double> coefficients{std::vector<double>(taps.begin(), taps.end())};
    return gr::filter::signedAmplitudeAt(coefficients, f);
}

/**
 * @brief |B| at i / grid cycles per sample, for i over half a turn, for antisymmetric taps.
 *
 * A type-III design has `H(e^{jw}) = -j e^{-jwM} B(w)` with `B(w) = 2 sum_k h[M+k] sin(k w)`, so
 * reading it needs a sine sum where `halfResponse` uses a cosine one. The same table serves both:
 * `sin(x) = cos(x - pi/2)`, which on a power-of-two grid is an integer index offset, so this keeps
 * the same drift-free integer index stepping.
 */
inline void halfResponseOdd(const std::vector<float>& taps, std::vector<double>& mag, std::size_t grid = kDesignGrid) {
    if (grid == 0UZ || (grid & (grid - 1UZ)) != 0UZ) {
        throw std::invalid_argument(std::format("FilterDesign: the index step is taken modulo the grid by mask, so the grid must be a power of two, got {}", grid));
    }
    const std::size_t n       = taps.size();
    const std::size_t mid     = n == 0UZ ? 0UZ : (n - 1UZ) / 2UZ;
    const std::size_t half    = grid / 2UZ;
    const std::size_t quarter = 3UZ * (grid / 4UZ); // cos(x + 3pi/2) == sin(x)
    mag.assign(half + 1UZ, 0.0);

    const std::shared_ptr<const std::vector<double>> table = detail::cosineTable(grid);
    const std::vector<double>&                       cs    = *table;
    for (std::size_t k = 1UZ; k <= mid; ++k) {
        const double c   = 2.0 * static_cast<double>(taps[mid + k]);
        std::size_t  idx = quarter;
        for (std::size_t i = 0UZ; i <= half; ++i) {
            mag[i] += c * cs[idx];
            idx = (idx + k) & (grid - 1UZ);
        }
    }
    for (double& v : mag) {
        v = std::abs(v);
    }
}

/// @brief B(2*pi*f) at one stated frequency, by the sine sum, for antisymmetric taps of odd length.
template<typename T>
requires std::is_floating_point_v<T>
[[nodiscard]] inline double amplitudeAtOdd(const std::vector<T>& taps, double f) {
    const std::size_t n   = taps.size();
    const std::size_t mid = n == 0UZ ? 0UZ : (n - 1UZ) / 2UZ;
    const double      w   = 2.0 * std::numbers::pi * f;

    double b = 0.0;
    for (std::size_t k = 1UZ; k <= mid; ++k) {
        b += 2.0 * static_cast<double>(taps[mid + k]) * std::sin(w * static_cast<double>(k));
    }
    return b;
}

/// @brief Scale @p taps so that `A(2*pi*fRef) == gain`. Throws where the reference value is zero.
inline void normalizeAt(std::vector<double>& taps, double fRef, double gain) {
    gr::filter::FilterCoefficients<double> coefficients{taps};
    const std::pair<bool, double>          normalized = gr::filter::normaliseFilterCoefficients(coefficients, fRef, gain, gr::filter::NormalizationReference::SignedAmplitude);
    if (!normalized.first) {
        throw std::invalid_argument("FilterDesign: the normalization reference is zero — a design cannot be normalized at one of its own nulls");
    }
    taps = std::move(coefficients.b);
}

/// @brief The response of one tap set across its own passband and its own stopband.
struct LowpassScan {
    double stopbandDb = 0.0; /// worst level anywhere at or above the stop edge, in dB and negative
    double rippleDb   = 0.0; /// peak to trough across the passband
};

/// @brief Measure @p taps between DC and @p passEdge, and from @p stopEdge to Nyquist.
[[nodiscard]] inline LowpassScan scanLowpass(const std::vector<float>& taps, double passEdge, double stopEdge, std::size_t grid = kDesignGrid) {
    std::vector<double> mag;
    halfResponse(taps, mag, grid);

    LowpassScan s;
    double      passMax = -1000.0;
    double      passMin = 1000.0;
    s.stopbandDb        = -1000.0;

    const std::size_t half = grid / 2UZ;
    for (std::size_t i = 0UZ; i <= half; ++i) {
        const double f  = static_cast<double>(i) / static_cast<double>(grid);
        const double db = 20.0 * std::log10(std::max(mag[i], 1.0e-300));
        if (f <= passEdge) {
            passMax = std::max(passMax, db);
            passMin = std::min(passMin, db);
        } else if (f >= stopEdge) {
            s.stopbandDb = std::max(s.stopbandDb, db);
        }
    }
    s.rippleDb = passMax - passMin;
    return s;
}

/// @brief The response of one tap set over one closed band.
struct BandLevels {
    double minMag = 0.0;
    double maxMag = 0.0;

    /// @brief The worst level in the band, in dB; the figure a stopband is judged by.
    [[nodiscard]] double peakDb() const { return 20.0 * std::log10(std::max(maxMag, 1.0e-300)); }
    /// @brief Peak to trough across the band, in dB; the figure a passband is judged by.
    [[nodiscard]] double rippleDb() const { return 20.0 * std::log10(std::max(maxMag, 1.0e-300) / std::max(minMag, 1.0e-300)); }
};

/**
 * @brief |A| over the closed band [@p lowEdge, @p highEdge], on the design grid and at both exact edges.
 *
 * A band edge is where the response is steepest; the stopband edge sits at the foot of the
 * transition. A grid that never lands on the stated edge therefore reports the design as better
 * than it is, by however much the response moves in one grid step. On an 87-tap high-pass that
 * bias is 0.11 dB of stopband peak. Evaluating the two edges exactly costs two cosine sums and
 * removes it.
 *
 * `scanLowpass` keeps its grid-only reading: it is what `searchLowpass` compares lengths with, and
 * changing it would change every length that search returns.
 */
[[nodiscard]] inline BandLevels scanBand(const std::vector<float>& taps, double lowEdge, double highEdge, std::size_t grid = kDesignGrid) {
    BandLevels out;
    out.minMag = std::min(std::abs(amplitudeAt(taps, lowEdge)), std::abs(amplitudeAt(taps, highEdge)));
    out.maxMag = std::max(std::abs(amplitudeAt(taps, lowEdge)), std::abs(amplitudeAt(taps, highEdge)));

    const double      lowest = std::ceil(lowEdge * static_cast<double>(grid));
    const std::size_t first  = lowest > 0.0 ? static_cast<std::size_t>(lowest) : 0UZ;
    const double      top    = std::floor(highEdge * static_cast<double>(grid));
    const std::size_t last   = top > 0.0 ? std::min(grid / 2UZ, static_cast<std::size_t>(top)) : 0UZ;
    if (first > last || !(top >= 0.0)) {
        return out;
    }

    std::vector<double> mag;
    halfResponse(taps, mag, grid);
    for (std::size_t i = first; i <= last; ++i) {
        out.minMag = std::min(out.minMag, mag[i]);
        out.maxMag = std::max(out.maxMag, mag[i]);
    }
    return out;
}

/**
 * @brief |B| over the usable band of a Hilbert transformer, `[lowEdge, 0.5 - lowEdge]`.
 *
 * A Hilbert transformer is structurally zero at DC and at Nyquist, and the passband is what lies
 * between, so it has no cutoff and no stopband to scan. What matters is the worst departure from
 * one across the band a caller intends to use, which is `max(maxMag - 1, 1 - minMag)`. Both
 * exact edges are read as well as the grid.
 *
 * At short lengths there is no usable band: eleven taps span 0.51 to 1.01 over `0.05 .. 0.45`,
 * which is not a passband a test can pin.
 */
[[nodiscard]] inline BandLevels scanHilbert(const std::vector<float>& taps, double lowEdge, std::size_t grid = kDesignGrid) {
    const double highEdge = 0.5 - lowEdge;

    BandLevels   out;
    const double atLow  = std::abs(amplitudeAtOdd(taps, lowEdge));
    const double atHigh = std::abs(amplitudeAtOdd(taps, highEdge));
    out.minMag          = std::min(atLow, atHigh);
    out.maxMag          = std::max(atLow, atHigh);

    const double      lowest = std::ceil(lowEdge * static_cast<double>(grid));
    const std::size_t first  = lowest > 0.0 ? static_cast<std::size_t>(lowest) : 0UZ;
    const double      top    = std::floor(highEdge * static_cast<double>(grid));
    const std::size_t last   = top > 0.0 ? std::min(grid / 2UZ, static_cast<std::size_t>(top)) : 0UZ;
    if (first > last || !(top >= 0.0)) {
        return out;
    }

    std::vector<double> mag;
    halfResponseOdd(taps, mag, grid);
    for (std::size_t i = first; i <= last; ++i) {
        out.minMag = std::min(out.minMag, mag[i]);
        out.maxMag = std::max(out.maxMag, mag[i]);
    }
    return out;
}

/// @brief The shortest odd length whose measured response meets the target, and what it delivers.
struct LowpassSearch {
    std::size_t taps = 0UZ;
    LowpassScan scan;
    bool        ok = false; /// false where no length under the limit met both targets, @c taps then being the longest tried
};

/**
 * @brief The shortest odd length whose measured response meets @p attenDb and @p maxRippleDb
 *        between @p passEdge and @p stopEdge.
 *
 * The -6 dB point goes midway between the two edges rather than on either, which puts a filter's
 * stated edge at its stated level.
 *
 * Kaiser's estimate marks where a design first touches its target rather than where it clears it,
 * and for wide transitions it is several taps out in either direction, so the length is walked up
 * until a scan says it delivers and then back down while it still does. Stepping two at a time
 * keeps the length odd, which keeps the group delay a whole sample.
 *
 * The cost is a response scan per candidate, and a scan is proportional to the length times the
 * grid.
 */
[[nodiscard]] inline LowpassSearch searchLowpass(double passEdge, double stopEdge, double attenDb, double maxRippleDb, std::size_t maxTaps, std::size_t grid = kDesignGrid) {
    const double cutoff = 0.5 * (passEdge + stopEdge);

    LowpassSearch out;
    const auto    delivers = [&](std::size_t len) {
        const LowpassScan r = scanLowpass(kaiserLowpass(len, cutoff, attenDb), passEdge, stopEdge, grid);
        return r.stopbandDb <= -attenDb && r.rippleDb <= maxRippleDb;
    };

    std::size_t use = kaiserLength(attenDb, stopEdge - passEdge) | 1UZ;
    while (use < maxTaps && !delivers(use)) {
        use += 2UZ;
    }
    while (use > 5UZ && delivers(use - 2UZ)) {
        use -= 2UZ;
    }

    out.taps = use;
    out.scan = scanLowpass(kaiserLowpass(use, cutoff, attenDb), passEdge, stopEdge, grid);
    out.ok   = out.scan.stopbandDb <= -attenDb && out.scan.rippleDb <= maxRippleDb;
    return out;
}

/// @brief sin(pi x) / (pi x), one at zero. Public because filter designs in other headers build on it.
[[nodiscard]] inline double sincPi(double x) { return x == 0.0 ? 1.0 : std::sin(std::numbers::pi * x) / (std::numbers::pi * x); }

namespace detail {

/// @brief The ideal lowpass impulse response `2f sinc(2 f k)` at integer offset @p k.
[[nodiscard]] inline double idealLowpass(double f, double k) { return 2.0 * f * sincPi(2.0 * f * k); }

/// @brief 1 at the center tap and 0 elsewhere.
[[nodiscard]] inline constexpr double delta(double k) noexcept { return k == 0.0 ? 1.0 : 0.0; }

/// @brief Window @p term over an odd length, second half copied so the taps are symmetric to the bit.
template<typename Term>
[[nodiscard]] inline std::vector<double> symmetricKernel(std::size_t n, WindowSpec window, Term&& term) {
    const std::vector<double> w   = gr::algorithm::window::create<double>(window.type, n, window.param);
    const std::size_t         mid = (n - 1UZ) / 2UZ;

    std::vector<double> h(n);
    for (std::size_t i = 0UZ; i <= mid; ++i) {
        const double v = w[i] * term(static_cast<double>(i) - static_cast<double>(mid));
        h[i]           = v;
        h[n - 1UZ - i] = v;
    }
    return h;
}

[[nodiscard]] inline std::vector<float> narrow(const std::vector<double>& h) {
    std::vector<float> out(h.size());
    for (std::size_t i = 0UZ; i < h.size(); ++i) {
        out[i] = static_cast<float>(h[i]);
    }
    return out;
}

[[nodiscard]] inline std::vector<std::complex<float>> narrow(const std::vector<std::complex<double>>& h) {
    std::vector<std::complex<float>> out(h.size());
    for (std::size_t i = 0UZ; i < h.size(); ++i) {
        out[i] = std::complex<float>(static_cast<float>(h[i].real()), static_cast<float>(h[i].imag()));
    }
    return out;
}

/**
 * @brief Rotate a real symmetric prototype to center frequency @p fc, the ramp centered on the center tap.
 *
 * Centering the phase ramp on the center tap (`k = n - M`, zero at `n = M`) makes the result
 * conjugate-symmetric, and conjugate symmetry gives the design exactly linear phase. The second half is the first half's conjugate, copied, so the invariant holds to the
 * bit rather than to the accuracy of two sine calls.
 *
 * A prototype of no taps has no center tap for the ramp to be zero at, and rotates to a design of no
 * taps.
 */
[[nodiscard]] inline std::vector<std::complex<double>> rotate(const std::vector<double>& proto, double fc) {
    const std::size_t n = proto.size();
    if (n == 0UZ) {
        return {};
    }
    const std::size_t mid = (n - 1UZ) / 2UZ;

    std::vector<std::complex<double>> h(n);
    for (std::size_t i = 0UZ; i < mid; ++i) {
        const std::complex<double> v = proto[i] * std::polar(1.0, 2.0 * std::numbers::pi * fc * (static_cast<double>(i) - static_cast<double>(mid)));
        h[i]                         = v;
        h[n - 1UZ - i]               = std::conj(v);
    }
    h[mid] = std::complex<double>(proto[mid], 0.0); // the ramp is zero at the center tap
    return h;
}

} // namespace detail

/**
 * @brief The windowed sinc of each real design, before normalization, in double.
 *
 * Exposed because the four are exact complements of one another term by term: a low-pass plus a
 * high-pass at the same cutoff is the windowed delta, and so is a band-pass plus a band-stop over
 * the same edges. That identity is bit-exact here and can be asserted as such. After each side is
 * normalized at its own reference the identity holds only to the two normalization factors'
 * departure from unity, which is a few parts in ten thousand.
 *
 * Frequencies are cycles per sample; a stated cutoff is the -6 dB point.
 */
[[nodiscard]] inline std::vector<double> lowpassKernel(std::size_t n, double cutoff, WindowSpec window) {
    return detail::symmetricKernel(oddLength(n), window, [cutoff](double k) { return detail::idealLowpass(cutoff, k); });
}

/// @copydoc lowpassKernel
[[nodiscard]] inline std::vector<double> highpassKernel(std::size_t n, double cutoff, WindowSpec window) {
    return detail::symmetricKernel(oddLength(n), window, [cutoff](double k) { return detail::delta(k) - detail::idealLowpass(cutoff, k); });
}

/// @copydoc lowpassKernel
[[nodiscard]] inline std::vector<double> bandpassKernel(std::size_t n, double lowCutoff, double highCutoff, WindowSpec window) {
    return detail::symmetricKernel(oddLength(n, 5), window, [lowCutoff, highCutoff](double k) { return detail::idealLowpass(highCutoff, k) - detail::idealLowpass(lowCutoff, k); });
}

/// @copydoc lowpassKernel
[[nodiscard]] inline std::vector<double> bandstopKernel(std::size_t n, double lowCutoff, double highCutoff, WindowSpec window) {
    return detail::symmetricKernel(oddLength(n, 5), window, [lowCutoff, highCutoff](double k) { return detail::delta(k) - detail::idealLowpass(highCutoff, k) + detail::idealLowpass(lowCutoff, k); });
}

/// @brief A windowed-sinc low-pass, normalized to `A(0) = gain`. @p cutoff is the -6 dB point in cycles per sample.
[[nodiscard]] inline std::vector<float> lowpass(std::size_t n, double cutoff, WindowSpec window, double gain = 1.0) {
    std::vector<double> h = lowpassKernel(n, cutoff, window);
    normalizeAt(h, 0.0, gain);
    return detail::narrow(h);
}

/**
 * @brief A windowed-sinc high-pass, normalized to `abs(A(pi)) = gain`.
 *
 * This is the spectral complement of the low-pass at the same cutoff, `delta(k)` minus the
 * low-pass, not a low-pass at the mirror frequency with alternating signs normalized somewhere
 * inside its passband. The two have the same shape, but the latter puts the gain reference at an
 * arbitrary point and is off by whatever the passband ripple is there.
 *
 * Odd length is mandatory: a type-II design has `A(pi) = 0` identically and the normalization
 * would divide by zero.
 */
[[nodiscard]] inline std::vector<float> highpass(std::size_t n, double cutoff, WindowSpec window, double gain = 1.0) {
    std::vector<double> h = highpassKernel(n, cutoff, window);
    normalizeAt(h, 0.5, gain);
    return detail::narrow(h);
}

/**
 * @brief A windowed-sinc band-pass, normalized to `A(2*pi*fc) = gain` at the arithmetic center.
 *
 * The arithmetic mean of the two edges is where a symmetric design's passband peak lies. The
 * geometric mean agrees for a narrow band and does not for a wide one.
 *
 * @p transitionWidth, where a caller sizes from one, applies to both edges and is taken once.
 */
[[nodiscard]] inline std::vector<float> bandpass(std::size_t n, double lowCutoff, double highCutoff, WindowSpec window, double gain = 1.0) {
    std::vector<double> h = bandpassKernel(n, lowCutoff, highCutoff, window);
    normalizeAt(h, 0.5 * (lowCutoff + highCutoff), gain);
    return detail::narrow(h);
}

/**
 * @brief A windowed-sinc band-stop, normalized to `A(0) = gain`.
 *
 * The center term sits inside the window multiply. That is identical to adding it afterwards
 * whenever the window is one at its own center, and it degrades gracefully where a window is not.
 */
[[nodiscard]] inline std::vector<float> bandstop(std::size_t n, double lowCutoff, double highCutoff, WindowSpec window, double gain = 1.0) {
    std::vector<double> h = bandstopKernel(n, lowCutoff, highCutoff, window);
    normalizeAt(h, 0.0, gain);
    return detail::narrow(h);
}

/**
 * @brief A complex band-pass: a real low-pass prototype at half the band width, rotated to center.
 *
 * Negative edges are legal, so a band that straddles DC, or sits entirely below it, is one call.
 * The gain is inherited exactly from the prototype's `A(0) = gain`, so `abs(H(fc)) = gain`.
 *
 * `f1 = -f2` gives a real-valued low-pass with a zero phase ramp. There is no special case for it;
 * the general formula produces exactly that.
 *
 * Verification needs no new scan machinery: the modulation identity
 * `abs(H(f)) == abs(A_proto(f - fc))` holds exactly, so scanning the real prototype and asserting
 * the identity and the conjugate symmetry covers what a direct complex scan would.
 *
 * @param lowCutoff  lower -6 dB point in cycles per sample, within [-0.5, 0.5]
 * @param highCutoff upper -6 dB point in cycles per sample, greater than @p lowCutoff
 */
[[nodiscard]] inline std::vector<std::complex<float>> complexBandpass(std::size_t n, double lowCutoff, double highCutoff, WindowSpec window, double gain = 1.0) {
    std::vector<double> proto = lowpassKernel(oddLength(n, 5), 0.5 * (highCutoff - lowCutoff), window);
    normalizeAt(proto, 0.0, gain);
    return detail::narrow(detail::rotate(proto, 0.5 * (lowCutoff + highCutoff)));
}

/**
 * @brief A complex band-stop: the delta minus a unit-gain complex band-pass, times the gain.
 *
 * `abs(H(f)) = gain * abs(1 - H_bp(f))`, so the gain is @p gain across the passband and exactly
 * zero at the band-pass's peak. This is the same convention the real band-stop uses. Building the
 * band-stop instead by rotating a high-pass prototype normalized at that prototype's own Nyquist
 * puts the gain reference at the single frequency diametrically opposite the notch, so taps from
 * such a design differ from these by the passband ripple at that far point.
 *
 * One length is computed once and the container sized from it, so the length the taps are written
 * to is the length the container was reserved at. A band-pass of no taps has no center tap to place
 * the delta on, and the band-stop is empty in turn.
 */
[[nodiscard]] inline std::vector<std::complex<float>> complexBandstop(std::size_t n, double lowCutoff, double highCutoff, WindowSpec window, double gain = 1.0) {
    std::vector<double> proto = lowpassKernel(oddLength(n, 5), 0.5 * (highCutoff - lowCutoff), window);
    normalizeAt(proto, 0.0, 1.0);

    std::vector<std::complex<double>> h = detail::rotate(proto, 0.5 * (lowCutoff + highCutoff));
    if (h.empty()) {
        return {};
    }
    const std::size_t mid = (h.size() - 1UZ) / 2UZ;
    for (std::complex<double>& v : h) {
        v = -v;
    }
    h[mid] += 1.0;
    for (std::complex<double>& v : h) {
        v *= gain;
    }
    return detail::narrow(h);
}

/**
 * @brief The root-raised-cosine impulse response at time @p t in symbol periods.
 *
 * ```
 * h(t) = [ sin(pi t (1-a)) + 4 a t cos(pi t (1+a)) ] / [ pi t (1 - (4 a t)^2) ]
 * ```
 *
 * with both removable singularities in closed form: `h(0) = 1 - a + 4a/pi`, and at
 * `abs(4 a t) = 1`, `h = (a/sqrt2) [ (1 + 2/pi) sin(pi/(4a)) + (1 - 2/pi) cos(pi/(4a)) ]`.
 *
 * One expression covers the whole range of `alpha`: at zero it reduces to `sinc(t)`, and at one the
 * first numerator term vanishes and the second singularity's closed form gives exactly 1. A kernel
 * algebraically rearranged into a form that divides by `4 a t` needs a separate branch at each end.
 *
 * The second singularity is reached exactly in normal use: an RDS receiver at
 * `sps = 8, alpha = 1` puts `4 a t = 1` on `k = +/-2` in integer arithmetic. The
 * guard band of 1e-6 on `abs(1 - (4at)^2)` is two orders of magnitude wider than the plain
 * expression needs; its relative error there is 3e-12.
 */
[[nodiscard]] inline double rootRaisedCosineAt(double t, double alpha) {
    constexpr double pi = std::numbers::pi;
    if (t == 0.0) {
        return 1.0 - alpha + 4.0 * alpha / pi;
    }
    const double x = 4.0 * alpha * t;
    if (std::abs(1.0 - x * x) < 1.0e-6) {
        const double corner = pi / (4.0 * alpha);
        return (alpha / std::numbers::sqrt2) * ((1.0 + 2.0 / pi) * std::sin(corner) + (1.0 - 2.0 / pi) * std::cos(corner));
    }
    return (std::sin(pi * t * (1.0 - alpha)) + x * std::cos(pi * t * (1.0 + alpha))) / (pi * t * (1.0 - x * x));
}

/**
 * @brief A root-raised-cosine pulse, normalized to DC gain: `sum_n h[n] = gain`.
 *
 * DC gain rather than unit energy. The textbook convention is unit energy, which is a different
 * tap set by about `sqrt(sps)`, and a receiver whose downstream gains were set against
 * DC-normalized taps would come up wrong by that factor with no other symptom. A unit-energy
 * variant belongs behind its own name rather than behind a flag on this one.
 *
 * The length is rounded up to odd by `oddLength`, so @p n is a request and not a guarantee: an even
 * @p n comes back one tap longer. A polyphase caller therefore cannot ask for exactly `arms *
 * branchLength` taps whenever that product is even. Ask for one fewer, `arms * branchLength - 1`,
 * and let the partition append the single trailing zero it pads with anyway. The prototype's center
 * then sits half a tap off the bank's, under `1/arms` of an input sample, and the branch length is
 * the one that was asked for. Asking for the product itself adds a whole extra tap, which pushes
 * the partition to the next branch length and lengthens every branch's dot product.
 *
 * @param samplesPerSymbol sample rate over symbol rate; need not be an integer
 * @param alpha            excess bandwidth in [0, 1]
 */
[[nodiscard]] inline std::vector<float> rootRaisedCosine(std::size_t n, double samplesPerSymbol, double alpha, double gain = 1.0) {
    const std::size_t len = oddLength(n);
    const std::size_t mid = (len - 1UZ) / 2UZ;

    std::vector<double> h(len);
    for (std::size_t i = 0UZ; i <= mid; ++i) {
        const double v   = rootRaisedCosineAt((static_cast<double>(i) - static_cast<double>(mid)) / samplesPerSymbol, alpha);
        h[i]             = v;
        h[len - 1UZ - i] = v;
    }
    normalizeAt(h, 0.0, gain);
    return detail::narrow(h);
}

/// @brief A root-raised-cosine from rates in Hz rather than a ratio.
[[nodiscard]] inline std::vector<float> designRootRaisedCosine(std::size_t nTaps, double sampleRate, double symbolRate, double alpha, double gain = 1.0) { return rootRaisedCosine(nTaps, sampleRate / symbolRate, alpha, gain); }

/**
 * @brief The matched filter for a Manchester (biphase) symbol: a root-raised-cosine differenced
 *        against itself by half a symbol.
 *
 * A biphase symbol is one shaped pulse minus the same pulse half a symbol later, so its matched
 * filter is the shaping pulse differenced at that offset — `h[n] = p[n] - p[n + T/2]` — rather
 * than the pulse itself, which would match the un-split shape and read a biphase transition at
 * half strength. The half symbol must be a whole number of samples: the difference is a shift,
 * and a fractional shift belongs to a resampler, not a tap table. The taps sum to zero exactly
 * (the two copies cancel), so the filter passes no DC and a bias ahead of it costs nothing.
 * RDS's 57 kHz biphase at 16 samples per symbol is the driving consumer, alpha 1 as annex C
 * shapes it.
 */
[[nodiscard]] inline std::vector<float> manchesterMatchedFilter(std::size_t nTaps, double sampleRate, double symbolRate, double alpha) {
    const double half = sampleRate / (2.0 * symbolRate);
    if (!(half > 0.0) || std::floor(half) != half) {
        throw std::invalid_argument(std::format("manchesterMatchedFilter: half a symbol is {} samples; the difference is a shift and needs a whole number", half));
    }
    const auto hop = static_cast<std::size_t>(half);
    if (nTaps <= hop) {
        throw std::invalid_argument(std::format("manchesterMatchedFilter: {} taps do not span the {}-sample half-symbol offset", nTaps, hop));
    }
    const auto         pulse = designRootRaisedCosine(nTaps, sampleRate, symbolRate, alpha);
    std::vector<float> taps(pulse.size() - hop);
    for (std::size_t n = 0UZ; n < taps.size(); ++n) {
        taps[n] = pulse[n] - pulse[n + hop];
    }
    return taps;
}

/// @brief The standard deviation in symbol periods that makes @p bt the pulse's -3 dB bandwidth-time product.
[[nodiscard]] inline double gaussianSigma(double bt) { return std::sqrt(std::numbers::ln2) / (2.0 * std::numbers::pi * bt); }

/**
 * @brief A Gaussian pulse on a centered time grid, normalized to `sum_n h[n] = gain`.
 *
 * `t = (n - (N-1)/2) / sps`, so odd `N` has a center tap at `t = 0`, the taps are symmetric, and
 * the group delay is exactly `(N-1)/2` whole samples. Evaluating the same Gaussian on
 * `t = (n + 1 - N/2)/sps` gives a half-integer grid for odd `N`: the pulse sits half a sample late,
 * carries one extra sample on the right, comes out asymmetric by 36% of its own peak, and has the
 * truncation drag its delay off `M - 0.5` by an amount that moves with the tap count.
 *
 * This is the Gaussian pulse alone. A GMSK transmit filter is this convolved with a one-symbol
 * rectangle, and that convolution belongs to the modulator.
 *
 * @param bt bandwidth-symbol-time product; 0.3 and 0.35 are the usual settings
 */
[[nodiscard]] inline std::vector<float> gaussianPulse(std::size_t n, double samplesPerSymbol, double bt, double gain = 1.0) {
    const std::size_t len   = oddLength(n);
    const std::size_t mid   = (len - 1UZ) / 2UZ;
    const double      sigma = gaussianSigma(bt);

    std::vector<double> h(len);
    for (std::size_t i = 0UZ; i <= mid; ++i) {
        const double x   = ((static_cast<double>(i) - static_cast<double>(mid)) / samplesPerSymbol) / sigma;
        const double v   = std::exp(-0.5 * x * x);
        h[i]             = v;
        h[len - 1UZ - i] = v;
    }
    normalizeAt(h, 0.0, gain);
    return detail::narrow(h);
}

/**
 * @brief The next length at or above @p n at which no tap of a Hilbert transformer is wasted.
 *
 * Half the taps of a Hilbert transformer are zero, at every even offset from the center. Where M
 * is even the two outermost taps are among them, so the design is two taps of arithmetic shorter
 * than the length it claims: 61 and 65 buy nothing over 59 and 63. `N mod 4 == 3` is the length
 * that wastes no taps. The design entry point does not force it, because a caller may have a
 * length constraint of its own; this is what a caller with a free choice asks for.
 */
[[nodiscard]] inline constexpr std::size_t hilbertLength(std::size_t n) noexcept {
    const std::size_t len = oddLength(n);
    return ((len % 4UZ) == 3UZ) ? len : len + 2UZ;
}

/**
 * @brief A windowed Hilbert transformer, unit gain by construction rather than by renormalization.
 *
 * ```
 * h[n] = gain * w[k] * 2/(pi*k)   for odd k,   0 for even k including k = 0
 * ```
 *
 * A Hilbert transformer covers the whole open band, so the length, the window and @p gain are the
 * whole of the design. The `2/pi` factor is what makes the untruncated design unit gain:
 * `sum_{odd k>0} sin(kw)/k` is `pi/4` across that band, so `abs(H)` is one by construction, the
 * gain is a property of the design rather than of an accumulation loop, and @p gain means what it
 * says. Taps of `1/k` divided by an accumulated normalizer land instead on `abs(H)` at exactly
 * `fs/4`, correcting for the window's own error at one frequency; the two conventions differ by
 * 1.3% at eleven taps and under 0.16% from thirty-one up.
 *
 * The taps are antisymmetric (type III), so `H(0) = 0` and `H(fs/2) = 0` are structural and
 * `H = -j` across the band: positive frequencies are rotated by -90 degrees. An analytic signal is
 * `x[n-M] + j*y[n]`, this output paired with the input delayed by M samples.
 *
 * @param n length; `hilbertLength` rounds to the parity that spends every tap
 */
[[nodiscard]] inline std::vector<float> hilbert(std::size_t n, WindowSpec window, double gain = 1.0) {
    const std::size_t         len = oddLength(n);
    const std::size_t         mid = (len - 1UZ) / 2UZ;
    const std::vector<double> w   = gr::algorithm::window::create<double>(window.type, len, window.param);

    std::vector<double> h(len, 0.0);
    for (std::size_t i = 0UZ; i < mid; ++i) {
        const std::size_t back = mid - i; // the distance from the center tap; the offset itself is negative on this half
        if ((back % 2UZ) == 0UZ) {
            continue;
        }
        const double v   = -gain * w[i] * 2.0 / (std::numbers::pi * static_cast<double>(back));
        h[i]             = v;
        h[len - 1UZ - i] = -v;
    }
    return detail::narrow(h);
}

/// @brief H(2*pi*f) of an arbitrary complex tap set, by direct evaluation; the reading a complex design is judged by.
[[nodiscard]] inline std::complex<double> responseAt(const std::vector<std::complex<float>>& taps, double f) {
    const double         w = 2.0 * std::numbers::pi * f;
    std::complex<double> h{0.0, 0.0};
    for (std::size_t i = 0UZ; i < taps.size(); ++i) {
        h += std::complex<double>(static_cast<double>(taps[i].real()), static_cast<double>(taps[i].imag())) * std::polar(1.0, -w * static_cast<double>(i));
    }
    return h;
}

/**
 * @brief One windowed-sinc design, in the units a caller already has.
 *
 * Frequencies are Hz against `sampleRate`; the conversion to cycles per sample happens once, at
 * the entry point, and the two are never mixed within a call. Every cutoff is a -6 dB point.
 *
 * The length is estimated from `transitionWidth` unless `taps` is stated. With Kaiser the estimate
 * is Kaiser's own and `attenuationDb` is what the window is shaped for; with any other window the
 * attenuation is a property of the window rather than a parameter, `attenuationDb` is ignored, and
 * `windowFigures` reports what the design will deliver.
 *
 * `fromEdges` states the same transition band as a pass edge and a stop edge.
 */
struct FilterSpec {
    double                      sampleRate      = 1.0;
    double                      cutoff          = 0.0;                                      /// -6 dB point [Hz]; the lower edge of a band form
    double                      highCutoff      = 0.0;                                      /// -6 dB point [Hz], band forms only
    double                      transitionWidth = 0.0;                                      /// [Hz], applied to every edge
    double                      gain            = 1.0;                                      /// the value of the design's own normalization reference
    double                      attenuationDb   = 60.0;                                     /// Kaiser only: what the window is shaped for
    gr::algorithm::window::Type window          = gr::algorithm::window::Type::Kaiser;      //
    double                      windowParam     = std::numeric_limits<double>::quiet_NaN(); /// NaN: Kaiser takes beta from attenuationDb, others their own default
    std::size_t                 taps            = 0UZ;                                      /// 0: estimated from transitionWidth
};

/// @brief The window @p spec asks for, Kaiser's beta following from the attenuation where none is stated.
[[nodiscard]] inline WindowSpec windowOf(const FilterSpec& spec) {
    if (spec.window == gr::algorithm::window::Type::Kaiser && std::isnan(spec.windowParam)) {
        return kaiserFor(spec.attenuationDb);
    }
    return {spec.window, spec.windowParam};
}

/// @brief The odd length @p spec asks for, estimated from the transition width where none is stated.
[[nodiscard]] inline std::size_t tapCountOf(const FilterSpec& spec, std::size_t floorTaps = 3UZ) {
    if (spec.taps > 0UZ) {
        return oddLength(spec.taps, floorTaps);
    }
    const double width = spec.transitionWidth / spec.sampleRate;
    if (spec.window == gr::algorithm::window::Type::Kaiser) {
        return oddLength(kaiserLength(spec.attenuationDb, width), floorTaps);
    }
    return oddLength(windowLength(spec.window, width), floorTaps);
}

/// @brief A low-pass from @p spec: `cutoff` is the -6 dB point.
[[nodiscard]] inline std::vector<float> designLowpass(const FilterSpec& spec) { return lowpass(tapCountOf(spec), spec.cutoff / spec.sampleRate, windowOf(spec), spec.gain); }

/// @brief A high-pass from @p spec: `cutoff` is the -6 dB point.
[[nodiscard]] inline std::vector<float> designHighpass(const FilterSpec& spec) { return highpass(tapCountOf(spec), spec.cutoff / spec.sampleRate, windowOf(spec), spec.gain); }

/// @brief A band-pass from @p spec: `cutoff` and `highCutoff` are the two -6 dB points.
[[nodiscard]] inline std::vector<float> designBandpass(const FilterSpec& spec) { return bandpass(tapCountOf(spec, 5), spec.cutoff / spec.sampleRate, spec.highCutoff / spec.sampleRate, windowOf(spec), spec.gain); }

/// @brief A band-stop from @p spec: `cutoff` and `highCutoff` are the two -6 dB points.
[[nodiscard]] inline std::vector<float> designBandstop(const FilterSpec& spec) { return bandstop(tapCountOf(spec, 5), spec.cutoff / spec.sampleRate, spec.highCutoff / spec.sampleRate, windowOf(spec), spec.gain); }

/// @brief A complex band-pass from @p spec: either edge may be negative.
[[nodiscard]] inline std::vector<std::complex<float>> designComplexBandpass(const FilterSpec& spec) { return complexBandpass(tapCountOf(spec, 5), spec.cutoff / spec.sampleRate, spec.highCutoff / spec.sampleRate, windowOf(spec), spec.gain); }

/// @brief A complex band-stop from @p spec: either edge may be negative.
[[nodiscard]] inline std::vector<std::complex<float>> designComplexBandstop(const FilterSpec& spec) { return complexBandstop(tapCountOf(spec, 5), spec.cutoff / spec.sampleRate, spec.highCutoff / spec.sampleRate, windowOf(spec), spec.gain); }

/**
 * @brief Split a prototype into `arms` branches, branch-major and reversed.
 *
 * Branch `b` holds `p[(Lb - 1 - m)*arms + b]`, so `sum_m branch[m] * x[i + m]` is the zero-stuffed convolution
 * evaluated at output phase `b`. The prototype is zero-padded to an even branch length, which puts the interpolated
 * instant at `base + Lb/2 - 1 + mu`.
 *
 * This is the layout an interpolating bank wants, where one branch is chosen per output. A channelizer runs every
 * branch on every step and reads the prototype in its natural `h[j*M + r]` order instead, which is contiguous in
 * both the taps and the history and needs no separate copy.
 */
[[nodiscard]] inline std::vector<float> polyphasePartition(std::span<const float> prototype, std::size_t arms) {
    if (arms == 0UZ) {
        throw std::invalid_argument("gr::filter::fir::design::polyphasePartition: a polyphase bank needs at least one branch");
    }
    std::size_t branchLength = std::max((prototype.size() + arms - 1UZ) / arms, 2UZ);
    branchLength += branchLength % 2UZ;

    std::vector<float> branches(arms * branchLength, 0.f);
    for (std::size_t branch = 0UZ; branch < arms; ++branch) {
        for (std::size_t m = 0UZ; m < branchLength; ++m) {
            const std::size_t at                = (branchLength - 1UZ - m) * arms + branch;
            branches[branch * branchLength + m] = at < prototype.size() ? prototype[at] : 0.f;
        }
    }
    return branches;
}

} // namespace gr::filter::fir::design

#endif // GNURADIO_ALGORITHM_FILTER_DESIGN_HPP
