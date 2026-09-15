#ifndef GNURADIO_ALGORITHM_POLYPHASE_RESAMPLER_HPP
#define GNURADIO_ALGORITHM_POLYPHASE_RESAMPLER_HPP

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <numeric>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/FilterDesign.hpp>

/**
 * @brief An exact `L/M` rate change: `L` outputs for every `M` inputs.
 *
 * The naive construction inserts `L-1` zeros between input samples, filters at the resulting rate
 * and keeps every `M`th sample. The polyphase form below computes exactly that result without
 * materializing the zeros or the intermediate stream. Only the taps at indices congruent to `k*M`
 * modulo `L` meet a non-zero sample, so an output costs one dot product of `N/L` taps instead of
 * `N`. At `L = 147, M = 160` over 5880 taps that is 36.75 multiply-accumulates per input sample
 * against the naive 864360, a factor of `L*M`.
 *
 * With `h` the prototype padded to a multiple of `L` and `B = N/L` the branch length, output `k`
 * counted from the start of the stream is
 *
 * ```
 * p_k  = (k * M) mod L            the branch
 * i_k  = floor(k * M / L)         the input anchor
 * y[k] = sum_{r=0..B-1} h[p_k + r*L] * x[i_k - r]
 * ```
 *
 * which the implementation walks as an integer phase accumulator: emit at branch `ctr`, add `M`,
 * and advance the input while `ctr >= L`. The two forms agree exactly. The accumulator carries
 * across calls, so `ctr` and the input history are the whole of the state, the counter being an
 * integer and the history a plain shift register.
 *
 * The history starts at zero, for the reason `HalfbandCascade::primeWithSilence` exists: a block
 * with a fixed input-to-output ratio has to deliver that ratio from its first sample. The first `B`
 * outputs are the filter's rise out of a zeroed window, which is what a filter starting on silence
 * does. The alternative, waiting for a full window, offsets the whole stream by the group delay and
 * breaks the tag map.
 *
 * Chunk independence is a correctness property. The values summed for output `k` are fixed by `k`
 * alone, but their order is not automatic: floating-point addition is not associative, so an
 * implementation that takes four outputs at a time when four are available and one otherwise
 * returns different last bits for the same input under different chunking. The dot product here
 * accumulates into a fixed number of lanes with a fixed tail and a fixed lane reduction, all three
 * determined by the branch length alone, whatever a call's length or span boundaries are.
 *
 * This is hot path code. The inner loop is one contiguous-tap dot product over a contiguous input
 * run, which vectorizes with no gather; the branch table is stored branch-major, each branch
 * reversed, so selecting a branch is a pointer add and the samples are read forward. Allocation,
 * locking and dispatch all happen at construction.
 *
 * Each output is a plain dot product. `HalfbandCascade`'s shape, sixteen outputs at a time so that a
 * tap becomes a scalar broadcast, suits a cascade whose consecutive outputs share a branch; here
 * they use different branches. Measured over `std::complex<float>` on one core with 65536 input
 * samples, relative to the cheapest row:
 *
 * ```
 *   L/M       taps/branch   multiplies per input sample   cost
 *   1/2            77                  38.5               1.16
 *   3/2            37                  55.5               1.94
 *   1/25          907                  36.3               1.00
 *   147/160        40                  36.8               1.22
 * ```
 *
 * The cost tracks the multiplies per input sample (`N/M`) rather than the branch length: 907 taps a
 * branch at `1/25` is the cheapest row here. Time per complex multiply-accumulate is flat to within
 * 1.3x across the four at the baseline `x86-64` ISA, and a further 1.25x to 1.4x is available
 * where the wider ISA is allowed.
 *
 * The tap type costs `4 : 2 : 2 : 1` in real multiplies (complex taps on complex samples, either
 * mixed pair, two reals) and delivers it: measured per tap and output at the baseline ISA, the
 * first costs about 4.5x the last and the mixed pairs about 2.4x. A real-valued tap set carried in
 * a complex container therefore costs exactly double for the same arithmetic, and a downconverter
 * that rotates first and filters with a real prototype is the cheaper of the two routes to a
 * translating filter.
 *
 * Two cases belong elsewhere. `L = M = 1` after reduction is a pass-through, which `designResampler`
 * answers with a single unit tap rather than a filter. `L = 1` with `M` a power of two is handled by
 * `HalfbandCascade` several times faster, and the caller selects that route explicitly.
 *
 * @see harris, f. j., Multirate Signal Processing for Communication Systems: the polyphase
 *      decomposition of an interpolate-by-L / decimate-by-M chain and the commutator formulation
 *      the phase accumulator implements.
 */
namespace gr::filter {

/// @brief The sample types a polyphase branch's dot product is written for.
template<typename T>
concept PolyphaseSample = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;

/**
 * @brief The tap types, which are the same set as the sample types.
 *
 * A real-valued tap set belongs in a real tap type. A complex container of real values costs four
 * real multiplies a tap for two multiplies of arithmetic, and the type is the only thing that tells
 * the kernel which of the two it has.
 */
template<typename T>
concept PolyphaseTap = PolyphaseSample<T>;

namespace detail {

template<typename T>
struct PolyphaseScalar {
    using type = T;
};

template<typename T>
struct PolyphaseScalar<std::complex<T>> {
    using type = T;
};

/**
 * @brief The lanes, the accumulator count and the output type of one `(sample, tap)` combination.
 *
 * `kAcc` is what a tap costs in real multiplies, `1 : 2 : 2 : 4` over the four combinations, with
 * real taps the cheapest. The unroll is `8 / kAcc`, so one step drives eight independent
 * accumulator chains in every combination. That count determines the throughput: a single running
 * sum measures the latency of a dependent add chain instead, the same cost per tap in all four
 * combinations and an order of magnitude above the unrolled real-tap form.
 */
template<typename TSample, typename TTap>
struct PolyphaseTraits {
    using SampleValue = typename PolyphaseScalar<TSample>::type;
    using TapValue    = typename PolyphaseScalar<TTap>::type;
    using Value       = std::common_type_t<SampleValue, TapValue>;

    static constexpr std::size_t kSampleLanes = sizeof(TSample) / sizeof(SampleValue);
    static constexpr std::size_t kTapLanes    = sizeof(TTap) / sizeof(TapValue);
    static constexpr std::size_t kAcc         = kSampleLanes * kTapLanes;
    static constexpr std::size_t kBlock       = 8UZ / kAcc;

    using Output = std::conditional_t<kAcc == 1UZ, Value, std::complex<Value>>;
};

/**
 * @brief One branch's dot product, in a reduction tree fixed by @p n alone.
 *
 * `kBlock` taps a step into `kBlock * kAcc` independent lanes, a scalar tail of fewer than `kBlock`,
 * and a lane reduction in index order. All three depend on the branch length alone,
 * so the same input gives the same bits under any chunking.
 *
 * Both sides are flat interleaved reals rather than `std::complex` objects because a shifted-span
 * loop over complex objects does not vectorize. The four products a complex tap needs are four
 * independent accumulations that combine once at the end rather than per tap.
 */
template<typename TSample, typename TTap>
[[nodiscard]] inline typename PolyphaseTraits<TSample, TTap>::Output polyphaseDot(const typename PolyphaseTraits<TSample, TTap>::TapValue* h, const typename PolyphaseTraits<TSample, TTap>::SampleValue* x, std::size_t n) noexcept {
    using Traits = PolyphaseTraits<TSample, TTap>;
    using Value  = typename Traits::Value;
    using Output = typename Traits::Output;

    constexpr std::size_t kSampleLanes = Traits::kSampleLanes;
    constexpr std::size_t kTapLanes    = Traits::kTapLanes;
    constexpr std::size_t kAcc         = Traits::kAcc;
    constexpr std::size_t kBlock       = Traits::kBlock;

    Value             acc[kBlock * kAcc]{};
    const std::size_t bulk = n - (n % kBlock);
    for (std::size_t r = 0UZ; r < bulk; r += kBlock) {
        for (std::size_t b = 0UZ; b < kBlock; ++b) {
            for (std::size_t t = 0UZ; t < kTapLanes; ++t) {
                const Value w = static_cast<Value>(h[kTapLanes * (r + b) + t]);
                for (std::size_t s = 0UZ; s < kSampleLanes; ++s) {
                    acc[b * kAcc + t * kSampleLanes + s] += w * static_cast<Value>(x[kSampleLanes * (r + b) + s]);
                }
            }
        }
    }

    Value tail[kAcc]{};
    for (std::size_t r = bulk; r < n; ++r) {
        for (std::size_t t = 0UZ; t < kTapLanes; ++t) {
            const Value w = static_cast<Value>(h[kTapLanes * r + t]);
            for (std::size_t s = 0UZ; s < kSampleLanes; ++s) {
                tail[t * kSampleLanes + s] += w * static_cast<Value>(x[kSampleLanes * r + s]);
            }
        }
    }

    Value sum[kAcc];
    for (std::size_t u = 0UZ; u < kAcc; ++u) {
        Value s = acc[u];
        for (std::size_t b = 1UZ; b < kBlock; ++b) {
            s += acc[b * kAcc + u];
        }
        sum[u] = s + tail[u];
    }

    if constexpr (kAcc == 1UZ) {
        return sum[0];
    } else if constexpr (kAcc == 2UZ) {
        return Output{sum[0], sum[1]};
    } else {
        return Output{sum[0] - sum[3], sum[1] + sum[2]};
    }
}

} // namespace detail

/**
 * @brief The output offset an input offset maps to, `floor(i*L/M + 1/2)`, in integer arithmetic that cannot overflow.
 *
 * Tag offsets are carried in integer arithmetic end to end. Scaling an offset in floating point
 * goes silently off by one from `2^24` in `float`, about seventeen seconds into a 1 MS/s run, and
 * from `2^53` in `double`.
 *
 * The decomposition matters as well. The direct form `(2*i*L + M) / (2*M)` overflows a 64-bit
 * offset for large `i`: at `L = 3` and `i = 2^62 + 1`, `2*i*L` is 27670116110564327430, which does
 * not fit. Splitting `i` into `q*M + r` multiplies only `r < M` by `2L`, so the form is safe
 * whenever `2*M*L` fits, 2e12 at `L = M = 10^6`.
 *
 * Decimation maps several inputs to one output offset. Two tags landing on the same output offset
 * both belong there, and are kept in input order.
 */
[[nodiscard]] inline constexpr std::uint64_t mapResampledOffset(std::uint64_t offset, std::uint64_t interpolation, std::uint64_t decimation) noexcept {
    const std::uint64_t q = offset / decimation;
    const std::uint64_t r = offset % decimation;
    return q * interpolation + (2ULL * r * interpolation + decimation) / (2ULL * decimation);
}

/// @brief A designed prototype and what it was measured to deliver, at unit DC gain.
struct ResamplerDesign {
    std::vector<float> taps;                 /// at the interpolated rate `L*fs_in`, DC gain `l`
    int                designLength = 0;     /// the length before any padding; the group delay is `(designLength - 1) / (2*L)` input samples
    double             stopbandDb   = 0.0;   /// worst level from the stop edge to Nyquist, relative to unit DC gain
    double             rippleDb     = 0.0;   /// peak to trough from DC to the pass edge
    bool               ok           = false; /// false where no length under the cap met both targets
};

namespace detail {

struct PolyphaseDesignKey {
    std::size_t interpolation = 0UZ;
    std::size_t decimation    = 0UZ;
    double      rolloff       = 0.0;
    double      attenuationDb = 0.0;
    double      maxRippleDb   = 0.0;
    std::size_t maxTaps       = 0UZ;

    [[nodiscard]] bool operator==(const PolyphaseDesignKey&) const = default;
};

/**
 * @brief What a candidate delivers, with both band edges evaluated exactly rather than off the grid.
 *
 * `fir::design::scanLowpass` reads its grid only, deliberately: it is what `fir::design::searchLowpass`
 * compares lengths with, and changing it would change every length that search has returned. That
 * reading is adequate while the grid resolves the response, and it is not adequate here, because a
 * resampler's transition band narrows with `max(l, m)` while the grid does not follow. The stopband
 * peak sits at the stop edge, at the foot of the transition, where the response is steepest, so a
 * grid that never lands on the stated edge flatters the design by however far the nearest grid point
 * has fallen. Measured at the default `2^15` grid: a 903-tap design for `1/25` reads -60.00 dB and
 * is truly -59.42 dB, and a 5767-tap design for `147/160` reads -60.01 dB and is truly -59.19 dB.
 * Both would be accepted for a 60 dB request. `fir::design::scanBand` evaluates the stated edges as well
 * as the grid for this reason.
 */
[[nodiscard]] inline std::pair<double, double> polyphaseEdgeScan(const std::vector<float>& taps, double passEdge, double stopEdge, std::size_t grid) { return {fir::design::scanBand(taps, stopEdge, 0.5, grid).peakDb(), fir::design::scanBand(taps, 0.0, passEdge, grid).rippleDb()}; }

/**
 * @brief The shortest odd length that measurably delivers, with the -6 dB point stated rather than implied.
 *
 * `fir::design::searchLowpass` puts the cutoff midway between the two edges and reads the response on the
 * grid alone; neither is right here. The cutoff has to be settable because a halving stage's cutoff
 * must be exactly `0.25` for its even-offset taps to vanish, and the reading has to be
 * `polyphaseEdgeScan`'s because a narrow transition's stopband peak sits at the stop edge, between
 * grid points, where a grid-only reading flatters the design and lets the walk-down run past the
 * shortest length that truly delivers. Measured on the single-stage decimators below: 3431 taps at
 * `D = 64` against the grid-only reading's 3393, and 6859 against 6779 at `D = 128`.
 */
[[nodiscard]] inline std::size_t searchStage(double passEdge, double stopEdge, double cutoff, double attenuationDb, double maxRippleDb, std::size_t maxTaps) {
    const auto delivers = [&](std::size_t n) {
        const auto [stopbandDb, rippleDb] = polyphaseEdgeScan(fir::design::kaiserLowpass(n, cutoff, attenuationDb), passEdge, stopEdge, fir::design::kDesignGrid);
        return stopbandDb <= -attenuationDb && rippleDb <= maxRippleDb;
    };

    std::size_t use = fir::design::kaiserLength(attenuationDb, stopEdge - passEdge) | 1UZ;
    while (use < maxTaps && !delivers(use)) {
        use += 2UZ; // two at a time keeps the length odd, which keeps the group delay a whole sample
    }
    while (use > 5UZ && delivers(use - 2UZ)) {
        use -= 2UZ;
    }
    return use;
}

} // namespace detail

/**
 * @brief The default prototype for an `L/M` rate change, at the interpolated rate `L*fs_in`.
 *
 * ```
 * g        = gcd(L, M);  l = L/g;  m = M/g
 * stopEdge = 0.5 / max(l, m)
 * passEdge = (1 - rolloff) * stopEdge
 * gain     = l
 * ```
 *
 * `stopEdge` is the Nyquist frequency of the slower of the two rates expressed at the interpolated
 * rate: above it lie the images upsampling creates (`l > m`) or the frequencies decimation would
 * fold in (`m > l`), and either way they are removed. `gain = l` restores what zero-stuffing
 * divides away, so a DC input of amplitude `a` leaves at amplitude `a`. The reported `stopbandDb`
 * is relative to unit DC gain, before that factor.
 *
 * Note that `passEdge` and `stopEdge` depend only on `max(l, m)`, so `L/M` and `M/L` design the
 * same prototype and differ only in padding and branch count; and that the length grows like
 * `max(l, m)`, which is why 48 kHz to 44.1 kHz costs 5880 taps and why a caller who cannot afford
 * that wants a two-stage design this does not build.
 *
 * The length is searched rather than estimated. `fir::design::kaiserLength` gives the length where a
 * design first touches its target rather than where it clears it, up to 1.8 dB short, and the
 * search costs a response scan per candidate, which is a settings-time cost and never a per-sample
 * one. At `147/160` that is a few seconds over ~5800-tap candidates, so the result is memoized by
 * `(l, m, rolloff, attenuationDb, maxRippleDb, maxTaps)`; the first design of a ratio still pays. The
 * memo holds the last eight designs and drops the oldest, so a stream sweeping its rate cannot grow it.
 */
[[nodiscard]] inline ResamplerDesign designResampler(std::size_t interpolation, std::size_t decimation, double rolloff = 0.2, double attenuationDb = 60.0, double maxRippleDb = 0.1, std::size_t maxTaps = 1UZ << 16) {
    if (interpolation == 0UZ || decimation == 0UZ) {
        throw std::invalid_argument("designResampler: interpolation and decimation must both be at least one");
    }

    const std::size_t                g   = std::gcd(interpolation, decimation);
    const detail::PolyphaseDesignKey key = {interpolation / g, decimation / g, rolloff, attenuationDb, maxRippleDb, maxTaps};

    static std::mutex                                                          mutex;
    static std::vector<std::pair<detail::PolyphaseDesignKey, ResamplerDesign>> cache;
    {
        const std::lock_guard<std::mutex> lock(mutex);
        for (const auto& [cached, value] : cache) {
            if (cached == key) {
                return value;
            }
        }
    }

    const std::size_t l = key.interpolation;
    const std::size_t m = key.decimation;

    ResamplerDesign out;
    if (l == 1UZ && m == 1UZ) { // no rate change: a pass-through, not a delta convolved at full cost
        out.taps         = {1.0f};
        out.designLength = 1;
        out.ok           = true;
    } else {
        const double stopEdge = 0.5 / static_cast<double>(std::max(l, m));
        const double passEdge = (1.0 - rolloff) * stopEdge;
        const double cutoff   = 0.5 * (passEdge + stopEdge);

        const std::size_t use = detail::searchStage(passEdge, stopEdge, cutoff, attenuationDb, maxRippleDb, maxTaps);

        out.taps                               = fir::design::kaiserLowpass(use, cutoff, attenuationDb);
        out.designLength                       = static_cast<int>(out.taps.size());
        std::tie(out.stopbandDb, out.rippleDb) = detail::polyphaseEdgeScan(out.taps, passEdge, stopEdge, fir::design::kDesignGrid);
        out.ok                                 = out.stopbandDb <= -attenuationDb && out.rippleDb <= maxRippleDb;

        const auto gain = static_cast<float>(l);
        for (float& v : out.taps) {
            v *= gain;
        }
    }

    return fir::design::detail::keepDesign<fir::design::detail::kDesignMemoEntries>(mutex, cache, key, std::move(out));
}

/**
 * @brief The list of stage decimations a decimation by @p decimation is performed as.
 *
 * `D = 2^k * m` with `m` odd gives `[2] * k`, then `m`'s prime factors in descending order where
 * @p factorOdd, or `m` itself where not. `D = 1` gives an empty ladder: nothing folds at a
 * decimation of one, so there is nothing to filter.
 *
 * Both orderings are load bearing. Halvings go first because the design law makes the earliest
 * stages the cheapest and they are the ones paid at the highest rate. Odd factors go largest first
 * because the last stage carries the narrowest transition and is therefore the longest, so it is
 * the one that should be given the smallest decimation.
 *
 * Arithmetic only: no design runs here, so a caller can assert the ladder without paying for a tap
 * search.
 */
[[nodiscard]] inline std::vector<std::size_t> stagedDecimatorLadder(std::size_t decimation, bool factorOdd = true) {
    if (decimation == 0UZ) {
        throw std::invalid_argument("stagedDecimatorLadder: the decimation must be at least one");
    }

    std::vector<std::size_t> out;
    std::size_t              m = decimation;
    for (; (m % 2UZ) == 0UZ; m /= 2UZ) {
        out.push_back(2UZ);
    }
    if (m == 1UZ) {
        return out;
    }
    if (!factorOdd) {
        out.push_back(m);
        return out;
    }

    std::vector<std::size_t> odd;
    for (std::size_t p = 3UZ; p * p <= m; p += 2UZ) {
        for (; (m % p) == 0UZ; m /= p) {
            odd.push_back(p);
        }
    }
    if (m > 1UZ) {
        odd.push_back(m);
    }
    std::ranges::sort(odd, std::greater<>{});
    out.insert(out.end(), odd.begin(), odd.end());
    return out;
}

/// @brief One stage of a decimation ladder: what it decimates by, the taps designed for it, and what they measured.
struct DecimatorStage {
    std::size_t        decimation = 0UZ;
    std::size_t        stride     = 1UZ;   /// `p_{i-1}`, the product of the decimations above it
    std::vector<float> taps;               /// at this stage's own input rate, unit DC gain
    bool               halfband   = false; /// its two edges stand symmetrically about a quarter of its own rate
    double             passEdge   = 0.0;   /// cycles per sample of this stage's input, not of the cascade's
    double             stopEdge   = 0.0;
    double             stopbandDb = 0.0;
    double             rippleDb   = 0.0;
    bool               ok         = false;

    /**
     * @brief The multiplies one output actually costs.
     *
     * A halfband's even-offset taps are zero and are skipped by index, so the count is the odd
     * offsets plus the center: `2*ceil(M/2) + 1` for `M = (N-1)/2`, which is `M + 1` where `M` is
     * even and `M + 2` where it is odd. The half-length `N/2 + 1` is the same number only in the
     * first case: at `N` of 19 and 23 a halfband keeps 11 and 13 taps, not 10 and 12, because its
     * outermost pair sits at an odd offset and survives. `HalfbandCascade::liveTaps` counts the same.
     */
    [[nodiscard]] std::size_t liveTaps() const noexcept {
        if (!halfband || taps.empty()) {
            return taps.size();
        }
        const std::size_t mid = (taps.size() - 1UZ) / 2UZ;
        return 2UZ * ((mid + 1UZ) / 2UZ) + 1UZ;
    }

    /// @brief Multiplies per sample of the cascade's input, `liveTaps / p_i`, not of this stage's input.
    [[nodiscard]] double macsPerInput() const noexcept { return static_cast<double>(liveTaps()) / static_cast<double>(stride * decimation); }
};

/// @brief A designed ladder and what it was measured to deliver.
struct StagedDecimatorDesign {
    std::vector<DecimatorStage> stages;
    std::size_t                 decimation         = 1UZ;
    double                      macsPerInput       = 0.0; /// summed over the stages
    double                      worstStopbandDb    = 0.0; /// the least attenuated stage, which is the one a fold path can cross
    double                      rippleSumDb        = 0.0;
    std::uint64_t               groupDelaySamples  = 0ULL; /// input samples, stated and never compensated
    std::size_t                 oversizedOddFactor = 0UZ;  /// an odd factor placed above the bound asked for, 0 where none was
    bool                        ok                 = false;
};

namespace detail {

struct DecimatorLadderKey {
    std::vector<std::size_t> ladder;
    std::size_t              decimation    = 0UZ;
    double                   passbandWidth = 0.0;
    double                   attenuationDb = 0.0;
    double                   rippleDb      = 0.0;
    std::size_t              maxTaps       = 0UZ;

    [[nodiscard]] bool operator==(const DecimatorLadderKey&) const = default;
};

} // namespace detail

/**
 * @brief Design a stated ladder: every stage is given the same passband in hertz.
 *
 * A stage whose output rate is `Ro` need only reject above `Ro - W/2`, that being the lowest
 * frequency its own decimation folds inside the `+/-W/2` the caller asked to keep. At the top of
 * the ladder, where a tap is paid at the device's rate, `W` is a rounding error against the rate:
 * the transition band is nearly half the rate wide and thirteen taps deliver 85 dB. The length
 * migrates to the last stage, which runs at a small fraction of the rate and is where a long
 * filter is nearly free.
 *
 * With `fs = 1`, `W = passbandWidth / decimation` is the width the output keeps, expressed at the
 * input. Stage `i` has input rate `r_i = 1 / p_{i-1}` and edges, in cycles per sample of its own
 * input,
 *
 * ```
 * passEdge_i = (W/2) / r_i
 * stopEdge_i = (r_i/d_i - W/2) / r_i
 * ```
 *
 * A halving stage's two edges then stand symmetrically about `0.25`, which is the definition of a
 * halfband and the reason its even-offset taps vanish. Its design cutoff is set to exactly `0.25`
 * rather than to the midpoint the search computed, so the zeros are exact to the bits the window is
 * evaluated in, 322 to 332 dB under the peak, `sin` of a rounded multiple of pi, and the kernel may
 * skip them by index rather than by testing.
 *
 * Each stage's length is searched against its realized response, for @p attenuationDb and for
 * @p rippleDb divided evenly among the stages. The window is Kaiser because a windowed design's
 * floor is the window's: a Blackman-Harris sidelobe does not go below about 92 dB however many taps
 * are spent on it, and a Kaiser's beta follows the attenuation asked for.
 *
 * `D = 64`, `passbandWidth = 0.90`, the ladder in full:
 *
 * ```
 *   stage  decim   input rate   passEdge   stopEdge     N   live   MAC/input   stopband
 *     0      2      1.000000    0.007031   0.492969    13     7      3.500     -85.77 dB
 *     1      2      0.500000    0.014063   0.485938    13     7      1.750     -85.77 dB
 *     2      2      0.250000    0.028125   0.471875    19    11      1.375     -97.38 dB
 *     3      2      0.125000    0.056250   0.443750    19    11      0.688     -91.13 dB
 *     4      2      0.062500    0.112500   0.387500    23    13      0.406     -87.11 dB
 *     5      2      0.031250    0.225000   0.275000   117    59      0.922     -85.39 dB
 * ```
 *
 * 8.64 MAC per input sample against a single 3431-tap decimator's 53.61, ripple sum 0.0023 dB,
 * group delay 2158 input samples. The last stage is nine times the length of the first and costs a
 * quarter of it.
 *
 * The gain is not uniform. The single-stage cost barely moves with `D`, 53.6 to 53.9 MAC/input
 * across a 64-fold range, its length growing in proportion to `D` while the cost is `N/D`, so all
 * of staging's gain is the staged figure falling, and it falls with the number of halvings, not
 * with `D`. At `D = 25` with one polyphase stage the ladder degenerates to the single filter it is
 * being compared with and reclaims nothing.
 *
 * Measured, MAC per input sample, at 85 dB and 0.05 dB of total ripple:
 *
 * ```
 *      passbandWidth = 0.90                    passbandWidth = 0.80
 *   D    single   one odd   factored        single   one odd   factored
 *   8     53.88     16.12     16.12          27.88     12.62     12.62
 *  16     53.94     13.56     13.56          27.94     11.81     11.81
 *  25     53.88     53.88     17.84          27.88     27.88     12.64
 *  32     53.66     10.28     10.28          27.84      9.41      9.41
 *  50     53.62     30.44     12.42          27.86     17.44      9.82
 *  64     53.61      8.64      8.64          27.80      8.20      8.20
 * 100     53.59     18.72      9.71          27.69     12.22      8.41
 * 128     53.59      7.82      7.82          26.82      7.60      7.60
 * 512     53.57      7.21      7.21          26.81      7.15      7.15
 * ```
 *
 * Factoring the odd part is worth 3x at the shape the simple rule gives up on: `D = 25` as one
 * stage of 25 is 53.88 MAC/input and as two stages of 5 is 17.84. It cannot help with an odd prime
 * `D`, which is one stage either way and 1.00x. `passbandWidth` matters as much as `D`: a passband
 * filling 90 % of the output rate leaves a transition of `(1 - width)/D` and gains 3.3x to 7.4x,
 * where at 80 % the single stage is half as long to begin with and the gains are 2.2x to 3.7x.
 *
 * `worstStopbandDb` is the least attenuated stage, which is what the acceptance rule asks for: a
 * frequency reaches the output band only if every stage passes it and folds into `+/-W/2` only from
 * above some stage's own stop edge, so a fold path crosses one stopband and several passbands, and
 * what bounds it is the worst any single stage delivers over its own fold region. Measured across
 * the sweep at `passbandWidth = 0.90` it is -85.39 dB for every ladder ending in a halving stage,
 * the last stage being the one designed closest to its target, and -85.09 dB for `D = 25` as a
 * single stage of 25. Every ripple sum comes out around twenty times inside the 0.05 dB budget: the
 * budget is divided as a worst case on the assumption that stage ripples add, and measured they
 * largely do not.
 *
 * This is setup path and costs one length search per stage, each a response scan per candidate, so
 * the result is memoized by the whole parameter tuple: a retune returning to a ladder already built
 * pays nothing. The memo holds the last eight ladders and drops the oldest.
 *
 * @param ladder        the stage decimations, from `stagedDecimatorLadder` or stated; `{D}` is the
 *                      single-stage design the ladder is judged against
 * @param decimation    the product the ladder performs, which fixes `W` and therefore every edge
 * @param passbandWidth the fraction of the output rate the passband occupies, in `(0, 1)`; not of
 *                      the input rate, and not a frequency: every edge in the law is a ratio, and a
 *                      caller thinking in hertz divides once at the call site where the rate is known
 */
[[nodiscard]] inline StagedDecimatorDesign designDecimatorLadder(std::span<const std::size_t> ladder, std::size_t decimation, double passbandWidth = 0.8, double attenuationDb = 85.0, double rippleDb = 0.05, std::size_t maxTaps = 1UZ << 16) {
    if (decimation == 0UZ) {
        throw std::invalid_argument("designDecimatorLadder: the decimation must be at least one");
    }
    if (!(passbandWidth > 0.0) || !(passbandWidth < 1.0)) {
        throw std::invalid_argument("designDecimatorLadder: the passband width is a fraction of the output rate and must lie in (0, 1)");
    }

    const detail::DecimatorLadderKey key = {std::vector<std::size_t>(ladder.begin(), ladder.end()), decimation, passbandWidth, attenuationDb, rippleDb, maxTaps};

    static std::mutex                                                                mutex;
    static std::vector<std::pair<detail::DecimatorLadderKey, StagedDecimatorDesign>> cache;
    {
        const std::lock_guard<std::mutex> lock(mutex);
        for (const auto& [cached, value] : cache) {
            if (cached == key) {
                return value;
            }
        }
    }

    StagedDecimatorDesign out;
    out.decimation = decimation;
    out.ok         = true;

    const double width         = passbandWidth / static_cast<double>(decimation);
    const double stageRippleDb = ladder.empty() ? rippleDb : rippleDb / static_cast<double>(ladder.size());

    std::size_t stride = 1UZ;
    for (const std::size_t d : ladder) {
        if (d < 2UZ) {
            throw std::invalid_argument("designDecimatorLadder: a stage decimation must be at least two");
        }

        DecimatorStage stage;
        stage.decimation = d;
        stage.stride     = stride;
        stage.halfband   = d == 2UZ;

        const double rate = 1.0 / static_cast<double>(stride);
        stage.passEdge    = (width / 2.0) / rate;
        stage.stopEdge    = (rate / static_cast<double>(d) - width / 2.0) / rate;
        if (!(stage.stopEdge > stage.passEdge)) {
            throw std::invalid_argument("designDecimatorLadder: the passband is wider than a stage's own output rate leaves room for");
        }

        const double      cutoff = stage.halfband ? 0.25 : 0.5 * (stage.passEdge + stage.stopEdge);
        const std::size_t n      = detail::searchStage(stage.passEdge, stage.stopEdge, cutoff, attenuationDb, stageRippleDb, maxTaps);

        stage.taps                                 = fir::design::kaiserLowpass(n, cutoff, attenuationDb);
        std::tie(stage.stopbandDb, stage.rippleDb) = detail::polyphaseEdgeScan(stage.taps, stage.passEdge, stage.stopEdge, fir::design::kDesignGrid);
        stage.ok                                   = stage.stopbandDb <= -attenuationDb && stage.rippleDb <= stageRippleDb;

        out.macsPerInput += stage.macsPerInput();
        out.rippleSumDb += stage.rippleDb;
        out.worstStopbandDb = out.stages.empty() ? stage.stopbandDb : std::max(out.worstStopbandDb, stage.stopbandDb);
        out.groupDelaySamples += ((stage.taps.size() - 1UZ) / 2UZ) * stride;
        out.ok = out.ok && stage.ok;

        stride *= d;
        out.stages.push_back(std::move(stage));
    }

    return fir::design::detail::keepDesign<fir::design::detail::kDesignMemoEntries>(mutex, cache, key, std::move(out));
}

/**
 * @brief The default ladder for a decimation, designed: halvings, then the odd part factored into primes.
 *
 * `factorOdd` defaults to true on measurement. At `D = 25` one polyphase stage of 25 costs 53.88
 * MAC per input sample and two stages of 5 cost 17.84, the difference between reclaiming nothing
 * and reclaiming 3.02x, at the shape the simple ladder gives up on; `D = 50` and `D = 100` show
 * the same. `false` is kept so the simple ladder is one argument away; both columns are tabulated
 * above.
 *
 * @p maxOddFactor bounds the closing stage's length. An odd factor above it is placed anyway and
 * reported in `oversizedOddFactor` rather than refused: an odd prime `D` cannot be factored, and
 * refusing a legal decimation is worse than a long last stage.
 */
[[nodiscard]] inline StagedDecimatorDesign designStagedDecimator(std::size_t decimation, double passbandWidth = 0.8, double attenuationDb = 85.0, double rippleDb = 0.05, bool factorOdd = true, std::size_t maxOddFactor = 25UZ, std::size_t maxTaps = 1UZ << 16) {
    const std::vector<std::size_t> ladder = stagedDecimatorLadder(decimation, factorOdd);

    StagedDecimatorDesign out = designDecimatorLadder(std::span<const std::size_t>(ladder), decimation, passbandWidth, attenuationDb, rippleDb, maxTaps);
    for (const std::size_t d : ladder) {
        if (d > 2UZ && d > maxOddFactor) {
            out.oversizedOddFactor = std::max(out.oversizedOddFactor, d);
        }
    }
    return out;
}

/**
 * @brief The polyphase `L/M` kernel: state, branch table and the dot product per output.
 *
 * `L` and `M` are used as given and are not reduced by their gcd. The taps are defined against a
 * specific interpolated rate `L*fs_in`, so reducing `L` changes that rate and turns the same numbers
 * into a different filter. A caller designing its own prototype reduces first, which is what
 * `designResampler` does; a caller supplying taps owns the ratio it supplied them for, and the gain
 * with it, including the factor of `L`.
 *
 * At `L = 1` this is a decimating FIR, which is why the taps carry their own type. One branch of
 * `N` taps, phase zero at every output, and the accumulator reduced to advancing the input by `M`:
 * the same window, the same reversed store, the same reduction tree. A second FIR kernel would
 * repeat all four and the offset map with them. `TTap` is needed by the FIR case: a channel
 * filter's passband is not symmetric about DC, and a complex band-pass is the only single-filter
 * expression of it.
 */
template<typename TSample, typename TTap = float>
requires PolyphaseSample<TSample> && PolyphaseTap<TTap>
class PolyphaseResampler {
    using Traits      = detail::PolyphaseTraits<TSample, TTap>;
    using SampleValue = typename Traits::SampleValue;
    using TapValue    = typename Traits::TapValue;

    static constexpr std::size_t kSampleLanes = Traits::kSampleLanes;
    static constexpr std::size_t kTapLanes    = Traits::kTapLanes;

public:
    using sample_type = TSample;
    using tap_type    = TTap;
    /// @brief `decltype(TSample{} * TTap{})`: complex where either side is.
    using output_type = typename Traits::Output;

    /**
     * @brief Take @p taps as the prototype at the interpolated rate, zero-padded up to a multiple of @p interpolation.
     *
     * Padding is appended, never prepended: a tap's branch is its index modulo `L`, so leading zeros
     * would move every real tap into a different branch, which is a delay rather than padding.
     * Appending changes nothing about the response, a zero tap contributing no term to `H(z)`; it
     * only makes the stored vector asymmetric, which affects a naive group-delay formula alone. The
     * zero taps do cost multiplies, and a length already a multiple of `L` costs none.
     */
    PolyphaseResampler(std::size_t interpolation, std::size_t decimation, std::span<const TTap> taps) : _interpolation(interpolation), _decimation(decimation) {
        if (interpolation == 0UZ || decimation == 0UZ) {
            throw std::invalid_argument("PolyphaseResampler: interpolation and decimation must both be at least one");
        }
        if (taps.empty()) {
            throw std::invalid_argument("PolyphaseResampler: no taps — the pass-through's own prototype is a single unit tap");
        }

        _branchLength = (taps.size() + interpolation - 1UZ) / interpolation;
        _branch.assign(interpolation * _branchLength, TTap{});
        for (std::size_t p = 0UZ; p < interpolation; ++p) {
            for (std::size_t r = 0UZ; r < _branchLength; ++r) {
                const std::size_t at = p + r * interpolation;
                // Branch-major, and reversed within the branch, so one output reads a contiguous run
                // of taps against a contiguous run of samples read forward, with no gather either side.
                _branch[p * _branchLength + (_branchLength - 1UZ - r)] = at < taps.size() ? taps[at] : TTap{};
            }
        }
        _window.assign(2UZ * (_branchLength - 1UZ), TSample{});
    }

    /// @brief Forget the history and the phase: the next process() starts a new stream on silence.
    void reset() {
        std::ranges::fill(_window, TSample{});
        _phase    = 0UZ;
        _skip     = 0UZ;
        _produced = 0UZ;
    }

    /**
     * @brief The `B-1` samples preceding the next call, oldest first: the whole of what a tap change has to carry.
     *
     * A tap set is fixed at construction: the branch table is built once, branch-major and reversed,
     * which is where the dot product's speed comes from, so a new tap set is a new kernel and the
     * only thing the old one holds that the new one needs is this window. Reading it, rather than
     * replaying the history through the new kernel and discarding the outputs, is the difference
     * between a copy and a filter length of arithmetic: at 11001 taps the replay is 121 million
     * multiply-accumulates against 88 kB of `std::copy`.
     */
    [[nodiscard]] std::span<const TSample> window() const noexcept { return std::span<const TSample>(_window.data(), _branchLength - 1UZ); }

    /**
     * @brief Seed the history from @p history, newest sample last, older positions zero-filled.
     *
     * `primeWith(old.window())` is the tap change, and it preserves the input-to-output alignment
     * exactly: output `k` continues to be the filter applied at input `k*M` from the first output
     * after the swap. Where @p history is shorter than this kernel needs, as for a tap set that
     * grew, the older positions are zeros and the first `B-1` outputs are the filter's rise out of
     * them, the same transient a stream start has rather than a discontinuity in timing. Where it is
     * longer, the extra oldest samples never reached this filter's window and are dropped.
     *
     * The history is the whole of what is carried. A caller changing taps holds `L/M` fixed, and a
     * call chunked to `M` leaves the commutator phase where a freshly constructed kernel starts it;
     * a caller changing the ratio is moving the phase origin and owns that itself.
     */
    void primeWith(std::span<const TSample> history) noexcept {
        const std::size_t room = _branchLength - 1UZ;
        const std::size_t take = std::min(history.size(), room);
        std::ranges::fill(_window, TSample{});
        std::copy_n(history.end() - static_cast<std::ptrdiff_t>(take), take, _window.begin() + static_cast<std::ptrdiff_t>(room - take));
    }

    [[nodiscard]] std::size_t interpolation() const noexcept { return _interpolation; }
    [[nodiscard]] std::size_t decimation() const noexcept { return _decimation; }
    /// @brief `B = N/L`, the taps one output actually multiplies.
    [[nodiscard]] std::size_t branchLength() const noexcept { return _branchLength; }
    /// @brief `N`, the padded prototype length.
    [[nodiscard]] std::size_t taps() const noexcept { return _interpolation * _branchLength; }
    /// @brief The branch the next output will use: `(k*M) mod L` for the k-th output of the stream.
    [[nodiscard]] std::size_t phase() const noexcept { return _phase; }
    /// @brief Outputs delivered since construction or the last reset().
    [[nodiscard]] std::uint64_t produced() const noexcept { return _produced; }

    /// @brief How many outputs @p nInput further samples yield, from the phase the kernel is in now.
    [[nodiscard]] std::size_t outputsFor(std::size_t nInput) const noexcept {
        if (nInput <= _skip) {
            return 0UZ;
        }
        return (_interpolation * (nInput - _skip) - _phase + _decimation - 1UZ) / _decimation;
    }

    /// @brief How many input samples @p nOutput outputs need, from the phase the kernel is in now.
    [[nodiscard]] std::size_t inputsFor(std::size_t nOutput) const noexcept {
        if (nOutput == 0UZ) {
            return 0UZ;
        }
        return _skip + (_phase + (nOutput - 1UZ) * _decimation) / _interpolation + 1UZ;
    }

    /**
     * @brief Consume all of @p in and write `outputsFor(in.size())` samples to @p out.
     *
     * The stream produced is the one a single call over the whole input would have produced, to the
     * bit. @p out is sized by the caller through `outputsFor`; this never truncates, and throws once
     * per call rather than risking a write past the end.
     */
    std::size_t process(std::span<const TSample> in, std::span<output_type> out) {
        const std::size_t made = outputsFor(in.size());
        if (out.size() < made) {
            throw std::invalid_argument("PolyphaseResampler: the output span is shorter than outputsFor() reports");
        }

        const std::size_t history = _branchLength - 1UZ;
        const std::size_t head    = std::min(in.size(), history);
        if (head > 0UZ) {
            // The window holds the samples that preceded this call followed by the head of it, so an
            // output whose window straddles the call boundary still reads one contiguous run.
            std::copy_n(in.begin(), head, _window.begin() + static_cast<std::ptrdiff_t>(history));
        }

        const TapValue* const    branch  = reinterpret_cast<const TapValue*>(_branch.data());
        const SampleValue* const windowF = reinterpret_cast<const SampleValue*>(_window.data());
        const SampleValue* const inF     = reinterpret_cast<const SampleValue*>(in.data());

        std::size_t anchor = _skip;
        for (std::size_t k = 0UZ; k < made; ++k) {
            const SampleValue* x = (anchor < history) ? windowF + kSampleLanes * anchor : inF + kSampleLanes * (anchor - history);
            out[k]               = detail::polyphaseDot<TSample, TTap>(branch + kTapLanes * _phase * _branchLength, x, _branchLength);

            _phase += _decimation;
            while (_phase >= _interpolation) {
                _phase -= _interpolation;
                ++anchor;
            }
        }

        if (history > 0UZ) {
            if (in.size() >= history) {
                std::copy_n(in.end() - static_cast<std::ptrdiff_t>(history), history, _window.begin());
            } else {
                std::copy_n(_window.begin() + static_cast<std::ptrdiff_t>(in.size()), history, _window.begin());
            }
        }
        _skip = anchor - in.size();
        _produced += made;
        return made;
    }

private:
    std::size_t          _interpolation;
    std::size_t          _decimation;
    std::size_t          _branchLength = 0UZ;
    std::vector<TTap>    _branch;         /// branch-major, each branch reversed
    std::vector<TSample> _window;         /// the B-1 samples before the call, then room for the head of it
    std::size_t          _phase    = 0UZ; /// the commutator's `ctr`
    std::size_t          _skip     = 0UZ; /// inputs the next output's anchor sits ahead of the next unconsumed sample
    std::uint64_t        _produced = 0UZ;
};

} // namespace gr::filter

#endif // GNURADIO_ALGORITHM_POLYPHASE_RESAMPLER_HPP
