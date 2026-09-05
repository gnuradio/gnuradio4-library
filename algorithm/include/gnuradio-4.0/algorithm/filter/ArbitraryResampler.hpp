#ifndef GNURADIO_ALGORITHM_ARBITRARY_RESAMPLER_HPP
#define GNURADIO_ALGORITHM_ARBITRARY_RESAMPLER_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <numbers>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/PolyphaseResampler.hpp>

/**
 * @brief A rate change by a real factor, realized as a stated rational that never drifts.
 *
 * `PolyphaseResampler` covers only an exact `L/M`, and the tap count of an exact
 * rational design grows like `max(l, m)` after reduction: 48 kHz to 44.1 kHz is `147/160` and
 * costs 5880 taps. Two free-running device clocks stand in an effectively irrational ratio. This
 * block bounds the filter instead: the prototype is designed once for a bank of `L` arms and its
 * length per arm follows the attenuation, the rolloff and the direction of the rate change alone.
 * Measured at `rolloff = 0.2` and 60 dB, taps per arm is 37 at every
 * bank size from 8 to 128; doubling `L` doubles the prototype and halves the share each arm gets.
 *
 * The cost is exactly `q+1` times the arithmetic of an exact rational resampler, `q` being the
 * interpolation order, in exchange for a tap count set by the attenuation and the rolloff alone
 * and a ratio free to be irrational. Use `RationalResampler` for a ratio that is a small fraction:
 * `1/5` and `6` and `19/24` are exactly rational, cost half the arithmetic there and are exact.
 *
 * The prototype's cutoff scales with the rate below one. Arm `p` produces the signal at input time offset `p/L`, so a
 * prototype cut at the interpolated Nyquist over `L` leaves the bank passing the input band
 * through; stepping the anchor by more than one input sample then folds everything above the
 * output Nyquist straight in. Measured at `L = 32`, `r = 0.2`, a tone at 0.25 cycles per input
 * sample:
 *
 * ```
 *   prototype                          length   the tone, at the output
 *   scaled   (stopEdge = 0.5*r/L)       5787         -82.93 dB
 *   unscaled (stopEdge = 0.5/L)         1159           0.00 dB
 * ```
 *
 * The unscaled prototype passes the alias at full amplitude, 83 dB above the scaled one.
 *
 * One bank serves every order. It is `L + 2` arms of `B + 1` taps, `L + 4` at order 3, with
 * `arm[p][r] = h[p + (r-1)*L]` and every arm reading the same window. The `r = 0` term is zero for
 * every arm below `L` and the first real tap for the wrap arms, which is what makes the branch-free
 * arm wrap fall out of one formula. Written instead as a second filterbank of difference taps, the
 * interpolation doubles the tap memory and fixes the order at one.
 *
 * The phase advances in fixed point throughout. The step through the interpolated grid is
 * `round(L/r * 2^32)`, so the realized ratio is exactly `L * 2^32 / step`, a rational the kernel
 * reports and holds at any stream length and on any schedule, within `2^-33 * r / L` of the
 * request. Measured at `L = 32`: `2.7e-12` relative at `19/24`, or 0.011 samples a day at 48 kHz.
 * Deriving the same step from a `float` rate costs `4.0e-08` and 165 samples a day at that rate, so
 * it is the rate's own precision, rather than the accumulator's, that sets the drift: a `float`
 * accumulator contributes `3e-2` interpolated samples per million outputs, a fortieth of it.
 *
 * This is the hot path. `(q+1)` dot products of `B+1` taps per output, `(q+1)*(B+1)*r` per input
 * sample, flat at `(q+1)*(B+1)` below unity because `B` scales as `1/r`: 76 multiply-accumulates
 * per input sample at the defaults against `RationalResampler`'s 36.75 at the same design targets.
 * The reduction tree is `PolyphaseResampler`'s, eight independent accumulator chains fixed by `B`
 * alone and worth fourteen times against one running sum, and adjacent arms are adjacent in memory,
 * so the `q+1` arms an output needs are one contiguous run.
 *
 * @see harris, f. j., Multirate Signal Processing for Communication Systems, chapter 7.5 — the
 *      polyphase filterbank read as an interpolator with a fractional-delay commutator.
 */
namespace gr::filter {

/// @brief Fraction bits in the phase step. Bounds the realized-ratio error at `1.2e-10` relative and keeps `step` inside a `uint64` for every `r > L*2^-32`.
inline constexpr int kArbitraryFractionBits = 32;

inline constexpr std::uint64_t kArbitraryOne  = 1ULL << kArbitraryFractionBits;
inline constexpr std::uint64_t kArbitraryMask = kArbitraryOne - 1ULL;

/**
 * @brief The worst deviation from an ideal fractional delay that interpolating between arms leaves.
 *
 * For a signal band-limited to `f_max = (1-rolloff)/2` cycles per input sample and arms spaced
 * `1/L` apart, `h = 2*pi*f_max/L` and the error is `h/2` at order 0, `h^2/8` at order 1 and
 * `h^4 * 9/384` at order 3. `mu` cycles at the resampling rate, so the
 * error is a spur and is sized against the stopband target rather than against the passband ripple.
 *
 * The floor is the prototype's own delivered passband ripple, `1.385e-3` at a
 * 60 dB design, which is why order 3 and order 1 measure the same at `L = 64`.
 */
[[nodiscard]] inline double arbitraryInterpolationError(std::size_t bankSize, double rolloff = 0.2, int order = 1) noexcept {
    const double h = 2.0 * std::numbers::pi * (1.0 - rolloff) * 0.5 / static_cast<double>(bankSize);
    if (order >= 3) {
        return h * h * h * h * 9.0 / 384.0;
    }
    if (order <= 0) {
        return 0.5 * h;
    }
    return h * h / 8.0;
}

/**
 * @brief The smallest power-of-two bank whose interpolation spur sits at or below @p attenuationDb.
 *
 * Evaluated at `r = 1`, so the bank size is settled once and holds across rate changes. At `rolloff = 0.2` that is
 * 16, 32, 128 and 512 arms at 40, 60, 80 and 100 dB for order 1, and 4, 8, 16 and 32 for order 3.
 * Raising the attenuation therefore costs memory, and past a point order 3 is the cheaper of the
 * two: at 100 dB the order-1 bank is 76 KiB against 5.3.
 *
 * A bank pinned at 32 arms with linear interpolation — a common fixed choice elsewhere — delivers
 * -54.6 dB, which meets a 60 dB design and misses an 80 dB one. That measurement is why the size is
 * derived from the attenuation here rather than fixed.
 */
[[nodiscard]] inline std::size_t arbitraryBankSize(double attenuationDb, double rolloff = 0.2, int order = 1) noexcept {
    const double target = std::pow(10.0, -attenuationDb / 20.0);
    std::size_t  size   = 1UZ;
    while (size < (1UZ << 16) && arbitraryInterpolationError(size, rolloff, order) > target) {
        size *= 2UZ;
    }
    return size;
}

namespace detail {

struct ArbitraryDesignKey {
    std::size_t bankSize      = 0UZ;
    double      minRate       = 0.0;
    double      rolloff       = 0.0;
    double      attenuationDb = 0.0;
    double      maxRippleDb   = 0.0;
    std::size_t maxTaps       = 0UZ;

    [[nodiscard]] bool operator==(const ArbitraryDesignKey&) const = default;
};

/// a 64-bit position quotient narrowed to an index; a template, so the cast stands where `std::size_t` is 32 bits and is
/// not a useless cast where it is the same 64-bit type
template<std::integral To, std::integral From>
[[nodiscard]] constexpr To narrowIndex(From value) noexcept {
    return static_cast<To>(value);
}

struct ShiftedQuotient {
    std::uint64_t quotient  = 0ULL;
    std::uint64_t remainder = 0ULL;
    bool          negative  = false; /// the numerator was below zero, both fields then being zero
};

/**
 * @brief `floor((delta * 2^F - frac) / step)` and its remainder, in 64-bit arithmetic throughout.
 *
 * The numerator runs to `2^96` and every offset in this header passes through it, so it cannot be
 * a `double` and it cannot be a 128-bit type either: the tree compiles `-Wpedantic -Werror`, and
 * `__int128` is not ISO C++. It uses `mapResampledOffset`'s decomposition instead: the whole
 * part of `delta/step` is a plain division, and the `F` fractional bits are a shift-subtract loop
 * over a remainder that is by construction below `step`, so nothing ever needs a 65th bit. That is
 * 32 iterations, once per call rather than once per sample.
 */
[[nodiscard]] inline ShiftedQuotient shiftedDivide(std::uint64_t delta, std::uint64_t frac, std::uint64_t step) noexcept {
    const std::uint64_t whole = delta / step;

    std::uint64_t q = 0ULL;
    std::uint64_t r = delta % step;
    for (int i = 0; i < kArbitraryFractionBits; ++i) {
        r <<= 1;
        q <<= 1;
        if (r >= step) {
            r -= step;
            q |= 1ULL;
        }
    }
    q += whole << kArbitraryFractionBits;

    if (r >= frac) {
        return {q, r - frac, false};
    }
    const std::uint64_t borrow = (frac - r + step - 1ULL) / step;
    if (borrow > q) {
        return {0ULL, 0ULL, true};
    }
    return {q - borrow, borrow * step + r - frac, false};
}

} // namespace detail

/**
 * @brief The default prototype for a bank of @p bankSize arms cut for rates down to @p minRate.
 *
 * ```
 * stopEdge = 0.5 * min(1, minRate) / L
 * passEdge = (1 - rolloff) * stopEdge
 * gain     = L
 * ```
 *
 * which is `designResampler`'s law with `max(l, m)` replaced by `L / min(1, r)`: the same single
 * statement, that the stopband starts at the Nyquist frequency of whichever rate is lower. The
 * `gain = L` restores what zero-stuffing divides away.
 *
 * The length is searched, for `designResampler`'s reason: Kaiser's estimate is where a design first
 * touches its target rather than where it clears it. gqrx reaches the scaling law empirically,
 * `cutoff = 0.4*rate` and `trans = 0.2*rate` below unity, "avoids phantom signals", with a
 * 385-tap Hamming prototype, 13 per arm, where this law asks for 1159 and 37.
 *
 * `minRate` is what makes a rise in rate free and a fall a rebuild: raising the rate inside the
 * designed band leaves the prototype narrower than it needs to be, which costs bandwidth and never
 * aliases. Memoized, because the search at `minRate = 0.2` is over ~5800-tap candidates; the memo
 * holds the last eight designs and drops the oldest.
 */
[[nodiscard]] inline ResamplerDesign designArbitraryResampler(std::size_t bankSize, double minRate, double rolloff = 0.2, double attenuationDb = 60.0, double maxRippleDb = 0.1, std::size_t maxTaps = 1UZ << 16) {
    if (bankSize == 0UZ) {
        throw std::invalid_argument("designArbitraryResampler: the bank must have at least one arm");
    }
    if (!(minRate > 0.0)) {
        throw std::invalid_argument("designArbitraryResampler: the rate must be positive");
    }

    const detail::ArbitraryDesignKey key = {bankSize, std::min(1.0, minRate), rolloff, attenuationDb, maxRippleDb, maxTaps};

    static std::mutex                                                          mutex;
    static std::vector<std::pair<detail::ArbitraryDesignKey, ResamplerDesign>> cache;
    {
        const std::lock_guard<std::mutex> lock(mutex);
        for (const auto& [cached, value] : cache) {
            if (cached == key) {
                return value;
            }
        }
    }

    const double stopEdge = 0.5 * key.minRate / static_cast<double>(bankSize);
    const double passEdge = (1.0 - rolloff) * stopEdge;
    const double cutoff   = 0.5 * (passEdge + stopEdge);

    const auto delivers = [&](std::size_t n) {
        const auto [stopbandDb, rippleDb] = detail::polyphaseEdgeScan(fir::design::kaiserLowpass(n, cutoff, attenuationDb), passEdge, stopEdge, fir::design::kDesignGrid);
        return stopbandDb <= -attenuationDb && rippleDb <= maxRippleDb;
    };

    std::size_t use = fir::design::kaiserLength(attenuationDb, stopEdge - passEdge) | 1UZ;
    while (use < maxTaps && !delivers(use)) {
        use += 2;
    }
    while (use > 5 && delivers(use - 2)) {
        use -= 2;
    }

    ResamplerDesign out;
    out.taps                               = fir::design::kaiserLowpass(use, cutoff, attenuationDb);
    out.designLength                       = static_cast<int>(out.taps.size());
    std::tie(out.stopbandDb, out.rippleDb) = detail::polyphaseEdgeScan(out.taps, passEdge, stopEdge, fir::design::kDesignGrid);
    out.ok                                 = out.stopbandDb <= -attenuationDb && out.rippleDb <= maxRippleDb;

    const auto gain = static_cast<float>(bankSize);
    for (float& v : out.taps) {
        v *= gain;
    }

    return fir::design::detail::keepDesign<fir::design::detail::kDesignMemoEntries>(mutex, cache, key, std::move(out));
}

/**
 * @brief The output offset an input offset maps to, `round((i*L*2^F - phase) / step)`, in integer arithmetic.
 *
 * The exact counterpart of `mapResampledOffset`, and it reduces to it whenever `step` is a whole
 * number of interpolated samples. Tag offsets never pass through a float, and the half rounds up,
 * as it does there.
 *
 * @param phase the interpolated position of the regime's first output, in `2^-F` interpolated
 *              samples relative to its first input sample; zero for a stream that has not
 *              changed rate.
 */
[[nodiscard]] inline std::uint64_t mapArbitraryOffset(std::uint64_t offset, std::uint64_t bankSize, std::uint64_t step, std::int64_t phase) noexcept {
    const std::int64_t  whole = phase >> kArbitraryFractionBits;
    const std::uint64_t frac  = static_cast<std::uint64_t>(phase) & kArbitraryMask;
    const std::uint64_t grid  = offset * bankSize;

    std::uint64_t delta = 0ULL;
    if (whole < 0) {
        delta = grid + static_cast<std::uint64_t>(-whole);
    } else {
        if (grid < static_cast<std::uint64_t>(whole)) {
            return 0ULL;
        }
        delta = grid - static_cast<std::uint64_t>(whole);
    }

    const detail::ShiftedQuotient at = detail::shiftedDivide(delta, frac, step);
    if (at.negative) {
        return 0ULL;
    }
    return at.quotient + ((at.remainder >= step - at.remainder) ? 1ULL : 0ULL);
}

/**
 * @brief The arbitrary-rate kernel: the bank, the fixed-point phase and the dot products per output.
 *
 * The taps are the prototype at the interpolated rate `L*fs_in` and are used as given, the caller
 * owning the gain including the factor of `L`. `L` is not inferred from them: a prototype is
 * meaningless without the bank size it was designed for.
 */
template<typename T>
requires PolyphaseSample<T>
class ArbitraryResampler {
    using Traits      = detail::PolyphaseTraits<T, float>;
    using SampleValue = typename Traits::SampleValue;

    static constexpr std::size_t kSampleLanes = Traits::kSampleLanes;

public:
    using sample_type = T;

    /**
     * @brief A bank of @p bankSize arms at interpolation @p order over @p taps, stepping at @p rate outputs per input.
     *
     * @p order is 0 (nearest arm), 1 (linear) or 3 (cubic Lagrange). Nothing else is offered: the
     * even orders put the interpolation nodes off-center for no gain, and the weights above order 3
     * cost more than the extra arms they save.
     */
    ArbitraryResampler(double rate, std::size_t bankSize, int order, std::span<const float> taps) : _bankSize(bankSize), _order(order) {
        if (bankSize == 0UZ) {
            throw std::invalid_argument("ArbitraryResampler: the bank must have at least one arm");
        }
        if (order != 0 && order != 1 && order != 3) {
            throw std::invalid_argument("ArbitraryResampler: the interpolation order is 0, 1 or 3");
        }
        if (taps.empty()) {
            throw std::invalid_argument("ArbitraryResampler: no taps");
        }

        build(taps);
        setRate(rate);
        reset();
    }

    /// @brief Forget the history and the phase: the next process() starts a new stream on silence.
    void reset() {
        std::ranges::fill(_window, T{});
        _ahead    = _bankSize; // the first output's window ends one input sample ahead of its anchor
        _frac     = 0ULL;
        _produced = 0ULL;
    }

    /**
     * @brief Change the rate, keeping the position, the window and the bank.
     *
     * Changing the step changes the spacing of future outputs, not where the kernel currently is.
     * The prototype is the caller's to reconsider: the design law depends on the rate only through
     * `min(1, r)`, and only a rate that falls below the one the prototype was cut for under-filters.
     */
    void setRate(double rate) {
        if (!(rate > 0.0)) {
            throw std::invalid_argument("ArbitraryResampler: the rate must be positive");
        }
        const double scaled = static_cast<double>(_bankSize) / rate * static_cast<double>(kArbitraryOne);
        if (!(scaled >= 1.0)) {
            throw std::invalid_argument("ArbitraryResampler: the rate exceeds L*2^32, at which the step rounds to nothing");
        }
        if (scaled >= 9.2233720368547758e18) {
            throw std::invalid_argument("ArbitraryResampler: the rate is too small for a 64-bit step");
        }
        _step = static_cast<std::uint64_t>(std::round(scaled));
    }

    /**
     * @brief Install a new prototype, keeping the phase and as much of the window as the new bank holds.
     *
     * The position is denominated in arms and the arm count has not changed, so it survives a change
     * of prototype length untouched; the window is carried newest-first and any deficit is
     * zero-filled, which is the same transient the kernel has at stream start.
     */
    void setTaps(std::span<const float> taps) {
        if (taps.empty()) {
            throw std::invalid_argument("ArbitraryResampler: no taps");
        }
        const std::vector<T> carried = _window;
        const std::size_t    was     = _windowLength;

        build(taps);

        const std::size_t keep = std::min(was, _windowLength);
        for (std::size_t i = 0UZ; i < keep; ++i) {
            _window[_windowLength - 1UZ - i] = carried[was - 1UZ - i];
        }
    }

    [[nodiscard]] std::size_t bankSize() const noexcept { return _bankSize; }
    [[nodiscard]] int         order() const noexcept { return _order; }
    /// @brief `B = ceil(N/L)`; an arm stores `B+1`, the extra one being the wrap term that is zero below arm `L`.
    [[nodiscard]] std::size_t tapsPerArm() const noexcept { return _windowLength; }
    /// @brief `N`, the prototype as supplied, before padding to a multiple of `L`.
    [[nodiscard]] std::size_t prototypeLength() const noexcept { return _prototypeLength; }
    /// @brief `(N-1)/(2L)` input samples, stated and not compensated. The Lagrange weights add none of their own, being exact at their nodes.
    [[nodiscard]] double groupDelaySamples() const noexcept { return 0.5 * static_cast<double>(_prototypeLength - 1UZ) / static_cast<double>(_bankSize); }
    /// @brief `L * 2^F / step`, the rational the kernel runs at.
    [[nodiscard]] double        realizedRate() const noexcept { return static_cast<double>(_bankSize) * static_cast<double>(kArbitraryOne) / static_cast<double>(_step); }
    [[nodiscard]] std::uint64_t step() const noexcept { return _step; }
    /// @brief `P`, the next output's interpolated position, in `2^-F` interpolated samples relative to the next unconsumed input sample.
    [[nodiscard]] std::int64_t  phase() const noexcept { return (static_cast<std::int64_t>(_ahead) - static_cast<std::int64_t>(_bankSize)) * static_cast<std::int64_t>(kArbitraryOne) + static_cast<std::int64_t>(_frac); }
    [[nodiscard]] std::uint64_t produced() const noexcept { return _produced; }

    /// @brief How many outputs @p nInput further samples yield, from the phase the kernel is in now.
    [[nodiscard]] std::size_t outputsFor(std::size_t nInput) const noexcept {
        const std::uint64_t available = static_cast<std::uint64_t>(nInput) * _bankSize;
        if (available <= _ahead) {
            return 0UZ;
        }
        const detail::ShiftedQuotient at = detail::shiftedDivide(available - _ahead, _frac, _step);
        if (at.negative) {
            return 0UZ;
        }
        return static_cast<std::size_t>(at.quotient + ((at.remainder > 0ULL) ? 1ULL : 0ULL));
    }

    /**
     * @brief How many input samples @p nOutput outputs need, from the phase the kernel is in now.
     *
     * The step's whole and fractional halves carry separately, so the interpolated position lands
     * back in `u * 2^F + v` with `v` below `2^F`, and `floor((u*2^F + v) / (L*2^F))` is then `u/L`
     * outright. This needs no wide arithmetic and is exact.
     */
    [[nodiscard]] std::size_t inputsFor(std::size_t nOutput) const noexcept {
        if (nOutput == 0UZ) {
            return 0UZ;
        }
        const std::size_t   behind = nOutput - 1UZ;
        const std::uint64_t carry  = _frac + behind * (_step & kArbitraryMask);
        const std::uint64_t at     = _ahead + behind * (_step >> kArbitraryFractionBits) + (carry >> kArbitraryFractionBits);
        return static_cast<std::size_t>(at / _bankSize + 1ULL);
    }

    /**
     * @brief Consume all of @p in and write `outputsFor(in.size())` samples to @p out.
     *
     * The stream produced is the one a single call over the whole input would have produced, to the
     * bit. Every sample an output reads comes from @p in: the window runs from `anchor - B` to
     * `anchor`, `anchor` included and accounted for by `outputsFor`, so the span the caller hands
     * over bounds what the kernel touches and `outputsFor` bounds what it reads.
     */
    std::size_t process(std::span<const T> in, std::span<T> out) {
        const std::size_t made = outputsFor(in.size());
        if (out.size() < made) {
            throw std::invalid_argument("ArbitraryResampler: the output span is shorter than outputsFor() reports");
        }

        const std::size_t history = _windowLength;
        const std::size_t head    = std::min(in.size(), history);
        if (head > 0UZ) {
            std::copy_n(in.begin(), head, _window.begin() + static_cast<std::ptrdiff_t>(history));
        }

        const float* const       bank    = _bank.data();
        const SampleValue* const windowF = reinterpret_cast<const SampleValue*>(_window.data());
        const SampleValue* const inF     = reinterpret_cast<const SampleValue*>(in.data());
        const std::size_t        armStep = history + 1UZ;

        for (std::size_t k = 0UZ; k < made; ++k) {
            const std::size_t        oldest = detail::narrowIndex<std::size_t>(_ahead / _bankSize);
            const std::size_t        arm    = _ahead % _bankSize;
            const SampleValue* const x      = (oldest < history) ? windowF + kSampleLanes * oldest : inF + kSampleLanes * (oldest - history);

            const double mu = static_cast<double>(_frac) / static_cast<double>(kArbitraryOne);
            out[k]          = blend(bank + (arm + 1UZ) * armStep, x, armStep, mu);

            _frac += _step & kArbitraryMask;
            _ahead += _step >> kArbitraryFractionBits;
            if (_frac >= kArbitraryOne) {
                _frac -= kArbitraryOne;
                ++_ahead;
            }
        }

        if (history > 0UZ) {
            if (in.size() >= history) {
                std::copy_n(in.end() - static_cast<std::ptrdiff_t>(history), history, _window.begin());
            } else {
                std::copy_n(_window.begin() + static_cast<std::ptrdiff_t>(in.size()), history, _window.begin());
            }
        }
        _ahead -= in.size() * _bankSize;
        _produced += made;
        return made;
    }

private:
    /**
     * @brief `arm[p][r] = h[p + (r-1)*L]` from `p = -1` up, each arm reversed, arms adjacent.
     *
     * The `r = 0` column is `h[p-L]`: zero for every arm below `L`, and the first real tap of the
     * wrap arms, which is what makes the wrap branch free. Substituting `r' = r-1` leaves
     * `sum_r' h[p + r'*L] * x[a-1-r']` for `p < L`, the arm-`p` output anchored at `a-1`, and for
     * `p >= L` the same expression is the wrapped arm reading the next input sample.
     *
     * The arm count follows the node set: `{0,1}` at orders 0 and 1 reaches `p = L`, and
     * `{-1,0,1,2}` at order 3 reaches `p = L+1` from `p = -1`. Order 0 needs the second arm because
     * it rounds rather than truncates.
     */
    void build(std::span<const float> taps) {
        _prototypeLength = taps.size();
        _windowLength    = (taps.size() + _bankSize - 1UZ) / _bankSize;

        const std::size_t armStep = _windowLength + 1UZ;
        const std::size_t arms    = _bankSize + ((_order == 3) ? 4UZ : 2UZ);
        _bank.assign(arms * armStep, 0.0f);
        for (std::size_t a = 0UZ; a < arms; ++a) {
            const std::ptrdiff_t p = static_cast<std::ptrdiff_t>(a) - 1;
            for (std::size_t j = 0UZ; j <= _windowLength; ++j) {
                const std::ptrdiff_t at = p + (static_cast<std::ptrdiff_t>(_windowLength - j) - 1) * static_cast<std::ptrdiff_t>(_bankSize);
                if (at >= 0 && static_cast<std::size_t>(at) < taps.size()) {
                    _bank[a * armStep + j] = taps[static_cast<std::size_t>(at)];
                }
            }
        }
        _window.assign(2UZ * _windowLength, T{});
    }

    /**
     * @brief `sum_s w_s(mu) * dot(arm[p+s], window)`, the Lagrange weights on the nodes `s`.
     *
     * `q+1` dot products of the same window against `q+1` consecutive arms, which are one
     * contiguous run of taps. The weights are a few multiplies of the fraction, once per output,
     * and the node set and the summation order are fixed at construction rather than per call, so
     * that the result does not depend on how the stream is chunked.
     */
    [[nodiscard]] T blend(const float* arm, const SampleValue* x, std::size_t n, double mu) const noexcept {
        if (_order == 0) {
            // the nearest arm, not the arm below: rounding halves the delay error to the h/2 the
            // error bound states, where truncation delivers h
            return detail::polyphaseDot<T, float>((mu < 0.5) ? arm : arm + n, x, n);
        }
        if (_order == 1) {
            const T first  = detail::polyphaseDot<T, float>(arm, x, n);
            const T second = detail::polyphaseDot<T, float>(arm + n, x, n);
            return static_cast<SampleValue>(1.0 - mu) * first + static_cast<SampleValue>(mu) * second;
        }

        const double weight[4] = {-mu * (mu - 1.0) * (mu - 2.0) / 6.0, (mu + 1.0) * (mu - 1.0) * (mu - 2.0) / 2.0, -(mu + 1.0) * mu * (mu - 2.0) / 2.0, (mu + 1.0) * mu * (mu - 1.0) / 6.0};

        T sum{};
        for (std::ptrdiff_t s = 0; s < 4; ++s) {
            sum += static_cast<SampleValue>(weight[s]) * detail::polyphaseDot<T, float>(arm + (s - 1) * static_cast<std::ptrdiff_t>(n), x, n);
        }
        return sum;
    }

    std::size_t        _bankSize;
    int                _order;
    std::size_t        _prototypeLength = 0UZ;
    std::size_t        _windowLength    = 0UZ; /// `B`
    std::vector<float> _bank;                  /// arm-major, each arm reversed, adjacent arms adjacent
    std::vector<T>     _window;                /// the B samples before the call, then room for the head of it
    std::uint64_t      _step     = 0ULL;       /// `round(L/r * 2^F)`
    std::uint64_t      _ahead    = 0ULL;       /// the next output's integer interpolated position, from the next unconsumed sample
    std::uint64_t      _frac     = 0ULL;       /// in `[0, 2^F)`
    std::uint64_t      _produced = 0ULL;
};

} // namespace gr::filter

#endif // GNURADIO_ALGORITHM_ARBITRARY_RESAMPLER_HPP
