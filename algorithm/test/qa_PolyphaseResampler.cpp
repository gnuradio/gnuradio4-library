#include <boost/ut.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <numeric>
#include <print>
#include <random>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/PolyphaseResampler.hpp>

namespace {

using gr::filter::PolyphaseResampler;

using CF = std::complex<float>;

struct Ratio {
    std::size_t l = 1UZ;
    std::size_t m = 1UZ;
};

/// One `(sample, tap)` pair, named the way a FIR cost table states it: sample first, tap second.
template<typename S, typename H>
struct Combo {
    using Sample = S;
    using Tap    = H;

    [[nodiscard]] static constexpr const char* name() {
        if constexpr (std::is_same_v<S, float>) {
            return std::is_same_v<H, float> ? "ff" : "fc";
        } else {
            return std::is_same_v<H, float> ? "cf" : "cc";
        }
    }
};

using Combos = std::tuple<Combo<float, float>, Combo<CF, float>, Combo<float, CF>, Combo<CF, CF>>;

template<typename T>
[[nodiscard]] constexpr T makeValue(float re, float im) {
    if constexpr (std::is_same_v<T, float>) {
        return re;
    } else {
        return T{re, im};
    }
}

template<typename T>
[[nodiscard]] constexpr float realOf(T v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else {
        return v.real();
    }
}

template<typename T>
[[nodiscard]] constexpr float imagOf(T v) {
    if constexpr (std::is_same_v<T, float>) {
        return 0.0f;
    } else {
        return v.imag();
    }
}

template<typename T>
[[nodiscard]] std::vector<T> asStream(const std::vector<float>& re, const std::vector<float>& im) {
    std::vector<T> out(re.size());
    for (std::size_t i = 0UZ; i < re.size(); ++i) {
        out[i] = makeValue<T>(re[i], im[i]);
    }
    return out;
}

[[nodiscard]] std::complex<double> widen(float v) { return {static_cast<double>(v), 0.0}; }
[[nodiscard]] std::complex<double> widen(CF v) { return {static_cast<double>(v.real()), static_cast<double>(v.imag())}; }

template<typename Body>
void forEachCombo(Body&& body) {
    std::apply([&](auto... combo) { (body(combo), ...); }, Combos{});
}

/// One row of the default-design table: what the design surface is asserted to return for a ratio.
struct DesignRow {
    std::size_t l          = 1UZ;
    std::size_t m          = 1UZ;
    double      passEdge   = 0.0;
    double      stopEdge   = 0.0;
    int         estimated  = 0; /// Kaiser's own estimate, rounded odd
    int         searched   = 0; /// the shortest length that measurably delivers
    int         padded     = 0; /// rounded up to a multiple of l
    int         perBranch  = 0;
    double      stopbandDb = 0.0;
    double      rippleDb   = 0.0;
};

constexpr DesignRow kDefaults[] = {
    {1UZ, 2UZ, 0.2, 0.25, 73, 77, 77, 77, -60.192, 0.0156},                //
    {3UZ, 2UZ, 2.0 / 15.0, 1.0 / 6.0, 109, 111, 111, 37, -60.087, 0.0155}, //
    {2UZ, 3UZ, 2.0 / 15.0, 1.0 / 6.0, 109, 111, 112, 56, -60.087, 0.0155}, //
    {1UZ, 4UZ, 0.1, 0.125, 145, 147, 147, 147, -60.357, 0.0160},           //
    {1UZ, 25UZ, 0.016, 0.02, 907, 907, 907, 907, -60.194, 0.0166},         //
    {4UZ, 5UZ, 0.08, 0.1, 183, 183, 184, 46, -60.396, 0.0161},             //
};

[[nodiscard]] std::vector<float> noise(std::size_t n, std::uint32_t seed) {
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> uniform(-0.7f, 0.7f);
    std::vector<float>                    x(n);
    for (float& v : x) {
        v = uniform(rng);
    }
    return x;
}

/// The whole stream through one call, which is the reference every chunked run is compared against.
[[nodiscard]] std::vector<float> resampleWhole(std::size_t l, std::size_t m, const std::vector<float>& taps, const std::vector<float>& x) {
    PolyphaseResampler<float> resampler(l, m, std::span<const float>(taps));
    std::vector<float>        y(resampler.outputsFor(x.size()));
    resampler.process(std::span<const float>(x), std::span<float>(y));
    return y;
}

/**
 * @brief The definition itself, in double: zero-stuff by l, convolve, keep every m-th sample.
 *
 * Deliberately not a second polyphase implementation. A branch-ordering error reproduced in both
 * would pass a comparison against another polyphase form and fails against this one.
 */
[[nodiscard]] std::vector<double> byDefinition(std::size_t l, std::size_t m, const std::vector<float>& taps, const std::vector<float>& x, std::size_t nOutput) {
    std::vector<double> stuffed(l * x.size(), 0.0);
    for (std::size_t i = 0UZ; i < x.size(); ++i) {
        stuffed[l * i] = static_cast<double>(x[i]);
    }

    std::vector<double> y(nOutput);
    for (std::size_t k = 0UZ; k < nOutput; ++k) {
        const std::size_t at  = k * m;
        double            acc = 0.0;
        for (std::size_t i = 0UZ; i < taps.size(); ++i) {
            if (i <= at && (at - i) < stuffed.size()) {
                acc += static_cast<double>(taps[i]) * stuffed[at - i];
            }
        }
        y[k] = acc;
    }
    return y;
}

/// The whole stream through one call at `L = 1`, which is a decimating FIR of these taps.
template<typename S, typename H>
[[nodiscard]] std::vector<typename PolyphaseResampler<S, H>::output_type> firWhole(std::size_t m, const std::vector<H>& taps, const std::vector<S>& x) {
    using Out = typename PolyphaseResampler<S, H>::output_type;
    PolyphaseResampler<S, H> filter(1UZ, m, std::span<const H>(taps));
    std::vector<Out>         y(filter.outputsFor(x.size()));
    filter.process(std::span<const S>(x), std::span<Out>(y));
    return y;
}

/**
 * @brief `y[n] = sum_i taps[i]*x[n-i]`, `out[k] = y[k*m]`, in double.
 *
 * The definition, written so that `taps` is the impulse response: no reversal, no conjugation,
 * `taps[0]` on the newest sample. Not a second polyphase implementation.
 */
template<typename S, typename H>
[[nodiscard]] std::vector<std::complex<double>> firByDefinition(std::size_t m, const std::vector<H>& taps, const std::vector<S>& x, std::size_t nOutput) {
    std::vector<std::complex<double>> y(nOutput);
    for (std::size_t k = 0UZ; k < nOutput; ++k) {
        const std::size_t    at = k * m;
        std::complex<double> acc{};
        for (std::size_t i = 0UZ; i < taps.size(); ++i) {
            if (i <= at && (at - i) < x.size()) {
                acc += widen(taps[i]) * widen(x[at - i]);
            }
        }
        y[k] = acc;
    }
    return y;
}

/// One DFT bin of a real record, normalized so a unit-amplitude cosine at that bin reads 1.
[[nodiscard]] double binAmplitude(const std::vector<float>& y, std::size_t bin) {
    const double         n = static_cast<double>(y.size());
    std::complex<double> acc{};
    for (std::size_t i = 0UZ; i < y.size(); ++i) {
        acc += static_cast<double>(y[i]) * std::polar(1.0, -2.0 * std::numbers::pi * static_cast<double>(bin) * static_cast<double>(i) / n);
    }
    return 2.0 * std::abs(acc) / n;
}

} // namespace

const boost::ut::suite<"polyphase resampler"> polyphaseResamplerTests = [] {
    using namespace boost::ut;
    using gr::filter::designResampler;
    using gr::filter::mapResampledOffset;
    using gr::filter::ResamplerDesign;

    constexpr Ratio kRatios[] = {{1UZ, 1UZ}, {1UZ, 2UZ}, {2UZ, 1UZ}, {3UZ, 2UZ}, {2UZ, 3UZ}, {1UZ, 25UZ}, {147UZ, 160UZ}};

    "the default design is the table"_test = [] {
        const auto check = [](const DesignRow& row) {
            const double width = row.stopEdge - row.passEdge;
            expect(eq(gr::filter::fir::design::kaiserLength(60.0, width) | 1UZ, static_cast<std::size_t>(row.estimated))) << row.l << "/" << row.m << ": Kaiser's estimate";

            const ResamplerDesign design = designResampler(row.l, row.m);
            expect(that % design.ok) << row.l << "/" << row.m << ": the search met both targets";
            expect(eq(design.designLength, row.searched)) << row.l << "/" << row.m << ": searched length";
            expect(approx(design.stopbandDb, row.stopbandDb, 0.05)) << row.l << "/" << row.m << ": stopband " << design.stopbandDb;
            expect(approx(design.rippleDb, row.rippleDb, 0.002)) << row.l << "/" << row.m << ": ripple " << design.rippleDb;

            const PolyphaseResampler<float> resampler(row.l, row.m, std::span<const float>(design.taps));
            expect(eq(resampler.taps(), static_cast<std::size_t>(row.padded))) << row.l << "/" << row.m << ": padded to a multiple of l";
            expect(eq(resampler.branchLength(), static_cast<std::size_t>(row.perBranch))) << row.l << "/" << row.m << ": taps per branch";
            expect(eq(resampler.taps() % row.l, 0UZ));
        };

        for (const DesignRow& row : kDefaults) {
            check(row);
        }
    };

    "L/M and M/L design the same prototype"_test = [] {
        // The band edges depend on max(l, m) alone, so only the padding and the branch count differ.
        const ResamplerDesign up   = designResampler(4UZ, 5UZ);
        const ResamplerDesign down = designResampler(5UZ, 4UZ);
        expect(eq(up.designLength, down.designLength));
        expect(approx(up.stopbandDb, down.stopbandDb, 1e-9));
        expect(eq(designResampler(6UZ, 4UZ).designLength, designResampler(3UZ, 2UZ).designLength)) << "and the ratio is reduced before it is designed";
    };

    "a rate change of one is a pass-through, not a delta"_test = [] {
        const ResamplerDesign design = designResampler(1UZ, 1UZ);
        expect(eq(design.taps.size(), 1UZ));
        expect(eq(design.taps[0UZ], 1.0f));
        expect(eq(designResampler(7UZ, 7UZ).taps.size(), 1UZ)) << "after reduction";

        const std::vector<float>  x = noise(1000UZ, 11U);
        PolyphaseResampler<float> resampler(1UZ, 1UZ, std::span<const float>(design.taps));
        std::vector<float>        y(x.size());
        expect(eq(resampler.process(std::span<const float>(x), std::span<float>(y)), x.size()));
        expect(that % (y == x)) << "bit for bit";
    };

    "the accumulator and the closed form are one schedule"_test = [&] {
        // One input sample at a time makes the accumulator state readable: the phase the kernel holds
        // is the branch the next output takes, and every output a single sample yields is anchored
        // on that sample. A divergence from the closed form means the state carry is wrong.
        for (const Ratio& ratio : kRatios) {
            const std::vector<float>  taps = noise(4UZ * ratio.l, 3U);
            PolyphaseResampler<float> resampler(ratio.l, ratio.m, std::span<const float>(taps));

            const std::vector<float> one(1UZ, 0.0f);
            std::vector<float>       out(ratio.l + 1UZ);
            std::size_t              fed = 0UZ;
            std::size_t              k   = 0UZ;
            while (k < 10000UZ) {
                expect(eq(resampler.phase(), (k * ratio.m) % ratio.l)) << ratio.l << "/" << ratio.m << " k=" << k << ": branch";

                const std::size_t made = resampler.outputsFor(1UZ);
                expect(eq(resampler.process(std::span<const float>(one), std::span<float>(out.data(), made)), made));
                ++fed;
                for (std::size_t j = 0UZ; j < made; ++j) {
                    expect(eq(((k + j) * ratio.m) / ratio.l, fed - 1UZ)) << ratio.l << "/" << ratio.m << " k=" << (k + j) << ": input anchor";
                }
                k += made;
            }
            expect(eq(resampler.produced(), static_cast<std::uint64_t>(k)));
        }
    };

    "the L=3 M=2 schedule is the table"_test = [] {
        constexpr std::size_t kBranch[] = {0UZ, 2UZ, 1UZ, 0UZ, 2UZ, 1UZ, 0UZ, 2UZ, 1UZ};
        constexpr std::size_t kAnchor[] = {0UZ, 0UZ, 1UZ, 2UZ, 2UZ, 3UZ, 4UZ, 4UZ, 5UZ};
        for (std::size_t k = 0UZ; k < 9UZ; ++k) {
            expect(eq((k * 2UZ) % 3UZ, kBranch[k])) << "k=" << k;
            expect(eq((k * 2UZ) / 3UZ, kAnchor[k])) << "k=" << k;
        }
    };

    "the ratio is exact from the first sample"_test = [&] {
        for (const Ratio& ratio : kRatios) {
            const std::vector<float>  taps = noise(7UZ * ratio.l, 5U);
            PolyphaseResampler<float> resampler(ratio.l, ratio.m, std::span<const float>(taps));

            const std::vector<float> x(ratio.m, 0.5f);
            std::vector<float>       y(ratio.l);
            for (std::size_t n = 1UZ; n <= 1000UZ; ++n) {
                expect(eq(resampler.process(std::span<const float>(x), std::span<float>(y)), ratio.l)) << ratio.l << "/" << ratio.m << " at chunk " << n;
            }
            expect(eq(resampler.produced(), 1000ULL * ratio.l)) << ratio.l << "/" << ratio.m << ": no drift over a thousand chunks";
            expect(eq(resampler.phase(), 0UZ)) << "and the phase closes on itself";
        }
    };

    "one call or many, bit for bit"_test = [] {
        for (const Ratio& ratio : {Ratio{1UZ, 1UZ}, Ratio{3UZ, 2UZ}, Ratio{2UZ, 3UZ}, Ratio{1UZ, 25UZ}}) {
            const std::vector<float> taps  = noise(11UZ * ratio.l + 3UZ, 7U);
            const std::vector<float> x     = noise(1UZ << 13, 13U + static_cast<std::uint32_t>(ratio.m));
            const std::vector<float> whole = resampleWhole(ratio.l, ratio.m, taps, x);
            expect(gt(whole.size(), 0UZ));

            for (const std::size_t chunk : {1UZ, 3UZ, 17UZ, 4096UZ}) {
                PolyphaseResampler<float> resampler(ratio.l, ratio.m, std::span<const float>(taps));
                std::vector<float>        y;
                for (std::size_t at = 0UZ; at < x.size(); at += chunk) {
                    const std::size_t take = std::min(chunk, x.size() - at);
                    const std::span   in(x.data() + at, take);
                    const std::size_t made = resampler.outputsFor(take);
                    y.resize(y.size() + made);
                    resampler.process(in, std::span<float>(y.data() + y.size() - made, made));
                }
                expect(that % (y == whole)) << ratio.l << "/" << ratio.m << " in chunks of " << chunk;
            }

            // and the same under chunk sizes that never repeat, where an accumulation tree chosen
            // from the call's own length would show
            PolyphaseResampler<float> resampler(ratio.l, ratio.m, std::span<const float>(taps));
            std::mt19937              rng(99U);
            std::vector<float>        y;
            for (std::size_t at = 0UZ; at < x.size();) {
                const std::size_t take = std::min<std::size_t>(1UZ + (rng() % 1500U), x.size() - at);
                const std::size_t made = resampler.outputsFor(take);
                y.resize(y.size() + made);
                resampler.process(std::span(x.data() + at, take), std::span<float>(y.data() + y.size() - made, made));
                at += take;
            }
            expect(that % (y == whole)) << ratio.l << "/" << ratio.m << " in random chunks";

            resampler.reset();
            std::vector<float> again(whole.size());
            resampler.process(std::span<const float>(x), std::span<float>(again));
            expect(that % (again == whole)) << "reset starts a new stream";
        }
    };

    "the output is the zero-stuff / convolve / decimate chain"_test = [] {
        for (const Ratio& ratio : {Ratio{1UZ, 1UZ}, Ratio{3UZ, 2UZ}, Ratio{2UZ, 3UZ}, Ratio{4UZ, 5UZ}, Ratio{1UZ, 4UZ}}) {
            const std::vector<float> taps = noise(9UZ * ratio.l + 2UZ, 17U);
            const std::vector<float> x    = noise(600UZ, 19U + static_cast<std::uint32_t>(ratio.l));
            const std::vector<float> got  = resampleWhole(ratio.l, ratio.m, taps, x);

            // the padding the kernel applies is part of the prototype the definition is fed
            std::vector<float> padded = taps;
            padded.resize(((taps.size() + ratio.l - 1UZ) / ratio.l) * ratio.l, 0.0f);
            const std::vector<double> want = byDefinition(ratio.l, ratio.m, padded, x, got.size());

            double worst = 0.0;
            double scale = 0.0;
            for (std::size_t k = 0UZ; k < got.size(); ++k) {
                worst = std::max(worst, std::abs(static_cast<double>(got[k]) - want[k]));
                scale = std::max(scale, std::abs(want[k]));
            }
            expect(gt(scale, 0.0));
            expect(lt(worst / scale, 1e-6)) << ratio.l << "/" << ratio.m << ": worst relative departure " << (worst / scale);
        }
    };

    "the designed filter has unit gain at DC"_test = [] {
        for (const DesignRow& row : kDefaults) {
            const ResamplerDesign     design = designResampler(row.l, row.m);
            PolyphaseResampler<float> resampler(row.l, row.m, std::span<const float>(design.taps));

            const std::vector<float> x(4UZ * resampler.branchLength() * row.m + 4UZ * row.m, 1.0f);
            std::vector<float>       y(resampler.outputsFor(x.size()));
            resampler.process(std::span<const float>(x), std::span<float>(y));

            // past the rise out of the zeroed history, a constant must come out a constant of the
            // same size: this is the test that catches a dropped gain factor of l
            double worst = 0.0;
            for (std::size_t k = 2UZ * resampler.branchLength(); k < y.size(); ++k) {
                worst = std::max(worst, std::abs(static_cast<double>(y[k]) - 1.0));
            }
            const double worstDb = 20.0 * std::log10(1.0 + worst);
            expect(lt(worstDb, 0.1)) << row.l << "/" << row.m << ": settles " << worstDb << " dB off unity";
        }
    };

    "a tone survives and its image does not"_test = [] {
        for (const DesignRow& row : kDefaults) {
            const ResamplerDesign     design = designResampler(row.l, row.m);
            PolyphaseResampler<float> resampler(row.l, row.m, std::span<const float>(design.taps));

            // an integer number of cycles per output record and per input record, so a bare DFT is
            // leakage-free and every bin below is the true level rather than a window's skirt
            const std::size_t nOut = row.l * (1024UZ / row.l + 1UZ);
            const std::size_t nIn  = nOut * row.m / row.l;
            const std::size_t bin  = static_cast<std::size_t>(0.3 * static_cast<double>(std::min(nOut, nIn)));

            std::vector<float> x(2UZ * nIn);
            for (std::size_t i = 0UZ; i < x.size(); ++i) {
                x[i] = static_cast<float>(std::cos(2.0 * std::numbers::pi * static_cast<double>(bin) * static_cast<double>(i) / static_cast<double>(nIn)));
            }
            std::vector<float> y(resampler.outputsFor(x.size()));
            resampler.process(std::span<const float>(x), std::span<float>(y));

            // the second record only: the first carries the filter's rise out of silence
            const std::vector<float> steady(y.end() - static_cast<std::ptrdiff_t>(nOut), y.end());
            expect(eq(steady.size(), nOut));

            const double tone = binAmplitude(steady, bin);
            expect(approx(tone, 1.0, 0.02)) << row.l << "/" << row.m << ": the tone arrives at " << tone;

            double worstOther = 0.0;
            for (std::size_t b = 1UZ; b < nOut / 2UZ; ++b) {
                if (b != bin) {
                    worstOther = std::max(worstOther, binAmplitude(steady, b));
                }
            }
            const double rejectionDb = 20.0 * std::log10(std::max(worstOther, 1e-300) / tone);
            expect(lt(rejectionDb, -59.0)) << row.l << "/" << row.m << ": worst image or alias at " << rejectionDb << " dB";
        }
    };

    "tag offsets are integer arithmetic"_test = [] {
        constexpr std::uint64_t kThreeTwo[]     = {0ULL, 2ULL, 3ULL, 5ULL, 6ULL, 8ULL, 9ULL, 11ULL};
        constexpr std::uint64_t kOneThree[]     = {0ULL, 0ULL, 1ULL, 1ULL, 1ULL, 2ULL, 2ULL, 2ULL};
        constexpr std::uint64_t kOneFourSeven[] = {0ULL, 1ULL, 2ULL, 3ULL, 4ULL, 5ULL, 6ULL, 6ULL};
        for (std::uint64_t i = 0ULL; i < 8ULL; ++i) {
            expect(eq(mapResampledOffset(i, 3ULL, 2ULL), kThreeTwo[i])) << "3/2 at " << i;
            expect(eq(mapResampledOffset(i, 1ULL, 3ULL), kOneThree[i])) << "1/3 at " << i;
            expect(eq(mapResampledOffset(i, 147ULL, 160ULL), kOneFourSeven[i])) << "147/160 at " << i;
        }
        expect(eq(mapResampledOffset(0ULL, 147ULL, 160ULL), 0ULL)) << "offset zero stays offset zero";

        // Why integer arithmetic: the offset does not survive the conversion, whatever is done to
        // it afterwards. 2^24+1 is not a float and 2^53+1 is not a double; each comes back one
        // sample short of where it was, and the offset it comes back as is a different output. At
        // 1 MS/s the float failure begins about seventeen seconds in.
        constexpr std::uint64_t k2p24 = (1ULL << 24) + 1ULL;
        constexpr std::uint64_t k2p53 = (1ULL << 53) + 1ULL;
        expect(eq(mapResampledOffset(k2p24, 1ULL, 3ULL), 5592406ULL));
        expect(eq(mapResampledOffset(k2p53, 1ULL, 3ULL), 3002399751580331ULL));

        expect(eq(static_cast<std::uint64_t>(static_cast<float>(k2p24)), k2p24 - 1ULL)) << "the float has already lost the offset";
        expect(eq(mapResampledOffset(k2p24 - 1ULL, 1ULL, 3ULL), 5592405ULL)) << "and the one it lost it to is an output short";
        expect(eq(static_cast<std::uint64_t>(static_cast<double>(k2p53)), k2p53 - 1ULL)) << "and a double from 2^53 in its turn";
        expect(eq(mapResampledOffset(k2p53 - 1ULL, 1ULL, 2ULL), 4503599627370496ULL));
        expect(eq(mapResampledOffset(k2p53, 1ULL, 2ULL), 4503599627370497ULL)) << "one apart, silently";

        // and where the undecomposed integer form overflows: 2*i*L is 27670116110564327430, which no
        // 64-bit register holds
        constexpr std::uint64_t kHuge  = (1ULL << 62) + 1ULL;
        const auto              direct = [](std::uint64_t i, std::uint64_t l, std::uint64_t m) { return (2ULL * i * l + m) / (2ULL * m); };
        expect(eq(mapResampledOffset(kHuge, 3ULL, 2ULL), 6917529027641081858ULL));
        expect(that % (direct(kHuge, 3ULL, 2ULL) != 6917529027641081858ULL)) << "the direct form has wrapped and answers with the wrapped value";
        expect(eq(direct(11ULL, 3ULL, 2ULL), mapResampledOffset(11ULL, 3ULL, 2ULL))) << "the two agree wherever the direct form still fits";

        // decimation puts several inputs on one output offset; they are not merged
        expect(eq(mapResampledOffset(3ULL, 1ULL, 3ULL), mapResampledOffset(4ULL, 1ULL, 3ULL)));
        for (std::uint64_t i = 1ULL; i < 500ULL; ++i) {
            expect(ge(mapResampledOffset(i, 147ULL, 160ULL), mapResampledOffset(i - 1ULL, 147ULL, 160ULL))) << "monotone at " << i;
        }
    };

    "an output costs one dot product of N/L taps"_test = [] {
        struct Cost {
            std::size_t l, m, taps, perOutput;
            double      perInput;
        };
        constexpr Cost kCosts[] = {{3UZ, 2UZ, 111UZ, 37UZ, 55.5}, {1UZ, 25UZ, 907UZ, 907UZ, 36.28}, {147UZ, 160UZ, 5880UZ, 40UZ, 36.75}};
        for (const Cost& cost : kCosts) {
            const std::vector<float>  taps = noise(cost.taps, 23U);
            PolyphaseResampler<float> resampler(cost.l, cost.m, std::span<const float>(taps));
            expect(eq(resampler.branchLength(), cost.perOutput)) << cost.l << "/" << cost.m;
            expect(approx(static_cast<double>(resampler.taps()) / static_cast<double>(cost.m), cost.perInput, 0.005)) << cost.l << "/" << cost.m;

            // the naive chain filters at the interpolated rate: N per interpolated sample, L*N per
            // input sample, so the decomposition is worth exactly L*M
            const double naivePerInput = static_cast<double>(cost.l) * static_cast<double>(resampler.taps());
            expect(approx(naivePerInput / (static_cast<double>(resampler.taps()) / static_cast<double>(cost.m)), static_cast<double>(cost.l * cost.m), 1e-9));
        }
    };

    "degenerate settings"_test = [] {
        const std::vector<float> unit{1.0f};
        expect(throws<std::invalid_argument>([&] { PolyphaseResampler<float>(0UZ, 1UZ, std::span<const float>(unit)); }));
        expect(throws<std::invalid_argument>([&] { PolyphaseResampler<float>(1UZ, 0UZ, std::span<const float>(unit)); }));
        expect(throws<std::invalid_argument>([&] { PolyphaseResampler<float>(1UZ, 1UZ, std::span<const float>()); }));
        expect(throws<std::invalid_argument>([&] { std::ignore = designResampler(0UZ, 4UZ); }));

        // fewer taps than branches: padded, and every branch that has no tap contributes zero
        const std::vector<float>  three{0.25f, 0.5f, 0.25f};
        PolyphaseResampler<float> resampler(4UZ, 1UZ, std::span<const float>(three));
        expect(eq(resampler.taps(), 4UZ));
        expect(eq(resampler.branchLength(), 1UZ));
        const std::vector<float> x{1.0f, 2.0f};
        std::vector<float>       y(8UZ);
        expect(eq(resampler.process(std::span<const float>(x), std::span<float>(y)), 8UZ));
        expect(eq(y[0UZ], 0.25f));
        expect(eq(y[3UZ], 0.0f)) << "the padded branch";

        // and an output span the caller sized short is refused rather than written past
        PolyphaseResampler<float> other(1UZ, 1UZ, std::span<const float>(unit));
        std::vector<float>        tooSmall(1UZ);
        expect(throws<std::invalid_argument>([&] { std::ignore = other.process(std::span<const float>(x), std::span<float>(tooSmall)); }));
    };

    "complex samples take the same schedule"_test = [] {
        const std::vector<float> taps = noise(23UZ, 29U);
        const std::vector<float> re   = noise(2000UZ, 31U);
        const std::vector<float> im   = noise(2000UZ, 37U);

        std::vector<CF> x(re.size());
        for (std::size_t i = 0UZ; i < x.size(); ++i) {
            x[i] = CF{re[i], im[i]};
        }

        PolyphaseResampler<CF> resampler(3UZ, 2UZ, std::span<const float>(taps));
        std::vector<CF>        y(resampler.outputsFor(x.size()));
        resampler.process(std::span<const CF>(x), std::span<CF>(y));

        // The two parts are the two real streams: same schedule, same branches, same taps. Not bit
        // for bit against the real kernel, though: a complex step carries four taps in eight lanes
        // where a real one carries eight in eight, so the two reduction trees differ. Each is fixed
        // for its own sample type, which is what chunk independence asks for.
        const std::vector<float> wantRe = resampleWhole(3UZ, 2UZ, taps, re);
        const std::vector<float> wantIm = resampleWhole(3UZ, 2UZ, taps, im);
        expect(eq(y.size(), wantRe.size()));

        double worst = 0.0;
        for (std::size_t k = 0UZ; k < y.size(); ++k) {
            worst = std::max({worst, std::abs(static_cast<double>(y[k].real() - wantRe[k])), std::abs(static_cast<double>(y[k].imag() - wantIm[k]))});
        }
        expect(lt(worst, 1e-6)) << "worst departure from the two real streams " << worst;

        // and the complex path is chunk-independent in its own right
        PolyphaseResampler<CF> chunked(3UZ, 2UZ, std::span<const float>(taps));
        std::vector<CF>        piecewise;
        for (std::size_t at = 0UZ; at < x.size(); at += 7UZ) {
            const std::size_t take = std::min<std::size_t>(7UZ, x.size() - at);
            const std::size_t made = chunked.outputsFor(take);
            piecewise.resize(piecewise.size() + made);
            chunked.process(std::span<const CF>(x.data() + at, take), std::span<CF>(piecewise.data() + piecewise.size() - made, made));
        }
        expect(that % (piecewise == y)) << "one call or many, bit for bit";
    };

    "taps are the impulse response"_test = [] {
        forEachCombo([](auto c) {
            using S   = typename decltype(c)::Sample;
            using H   = typename decltype(c)::Tap;
            using Out = typename PolyphaseResampler<S, H>::output_type;

            // asymmetric, so neither a stray reversal nor a stray conjugation can pass
            std::vector<H> taps(5UZ);
            for (std::size_t i = 0UZ; i < taps.size(); ++i) {
                taps[i] = makeValue<H>(static_cast<float>(i + 1UZ), static_cast<float>(2UZ * i + 1UZ));
            }

            for (const std::size_t p : {0UZ, 1UZ, 7UZ, 1000UZ}) {
                std::vector<S> x(p + taps.size() + 3UZ, S{});
                x[p] = makeValue<S>(1.0f, 0.0f);

                const std::vector<Out> y = firWhole<S, H>(1UZ, taps, x);
                expect(eq(y.size(), x.size())) << decltype(c)::name();
                for (std::size_t k = 0UZ; k < p; ++k) {
                    expect(that % (y[k] == Out{})) << decltype(c)::name() << ": before the impulse at " << p;
                }
                for (std::size_t i = 0UZ; i < taps.size(); ++i) {
                    expect(that % (y[p + i] == makeValue<Out>(realOf(taps[i]), imagOf(taps[i])))) << decltype(c)::name() << ": impulse at " << p << ", tap " << i;
                }
            }
        });
    };

    "the decimating FIR is the definition"_test = [] {
        forEachCombo([](auto c) {
            using S   = typename decltype(c)::Sample;
            using H   = typename decltype(c)::Tap;
            using Out = typename PolyphaseResampler<S, H>::output_type;

            const std::vector<H> taps = asStream<H>(noise(37UZ, 51U), noise(37UZ, 53U));
            const std::vector<S> x    = asStream<S>(noise(600UZ, 57U), noise(600UZ, 59U));

            for (const std::size_t m : {1UZ, 2UZ, 10UZ, 64UZ}) {
                const std::vector<Out>                  got  = firWhole<S, H>(m, taps, x);
                const std::vector<std::complex<double>> want = firByDefinition<S, H>(m, taps, x, got.size());

                double worst = 0.0;
                double scale = 0.0;
                for (std::size_t k = 0UZ; k < got.size(); ++k) {
                    worst = std::max(worst, std::abs(widen(got[k]) - want[k]));
                    scale = std::max(scale, std::abs(want[k]));
                }
                expect(gt(scale, 0.0));
                expect(lt(worst / scale, 1e-6)) << decltype(c)::name() << " at M=" << m << ": worst relative departure " << (worst / scale);
            }
        });
    };

    "one call or many, bit for bit, whatever the tap type"_test = [] {
        forEachCombo([](auto c) {
            using S   = typename decltype(c)::Sample;
            using H   = typename decltype(c)::Tap;
            using Out = typename PolyphaseResampler<S, H>::output_type;

            const std::vector<H> taps = asStream<H>(noise(53UZ, 61U), noise(53UZ, 67U));
            const std::vector<S> x    = asStream<S>(noise(1UZ << 13, 71U), noise(1UZ << 13, 73U));

            for (const std::size_t m : {1UZ, 10UZ}) {
                const std::vector<Out> whole = firWhole<S, H>(m, taps, x);
                for (const std::size_t chunk : {1UZ, 3UZ, 17UZ, 4096UZ}) {
                    PolyphaseResampler<S, H> filter(1UZ, m, std::span<const H>(taps));
                    std::vector<Out>         y;
                    for (std::size_t at = 0UZ; at < x.size(); at += chunk) {
                        const std::size_t take = std::min(chunk, x.size() - at);
                        const std::size_t made = filter.outputsFor(take);
                        y.resize(y.size() + made);
                        filter.process(std::span<const S>(x.data() + at, take), std::span<Out>(y.data() + y.size() - made, made));
                    }
                    expect(that % (y == whole)) << decltype(c)::name() << " at M=" << m << " in chunks of " << chunk;
                }
            }
        });
    };

    "a tap change through window() and primeWith() is the shadow replay"_test = [] {
        using S   = CF;
        using H   = float;
        using Out = PolyphaseResampler<S, H>::output_type;

        const std::vector<H> before = noise(151UZ, 811U);
        const std::vector<H> after  = noise(93UZ, 821U);
        const std::vector<S> x      = asStream<S>(noise(4096UZ, 823U), noise(4096UZ, 827U));

        for (const std::size_t m : {1UZ, 3UZ, 8UZ}) {
            const std::size_t    total = (x.size() / m) * m;
            const std::size_t    at    = (2048UZ / m) * m;
            const std::size_t    rest  = total - at;
            const std::vector<S> xs(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(total));

            PolyphaseResampler<S, H> old(1UZ, m, std::span<const H>(before));
            std::vector<Out>         head(old.outputsFor(at));
            old.process(std::span<const S>(xs.data(), at), std::span<Out>(head));

            PolyphaseResampler<S, H> primed(1UZ, m, std::span<const H>(after));
            expect(eq(primed.window().size(), primed.branchLength() - 1UZ));
            primed.primeWith(old.window());

            // The alternative route: feed the history back through the new kernel and throw the
            // outputs away. Aligned to the chunk size, because that is how the block is fed.
            PolyphaseResampler<S, H> replayed(1UZ, m, std::span<const H>(after));
            const std::size_t        history = replayed.branchLength() - 1UZ;
            const std::size_t        back    = ((history + m - 1UZ) / m) * m;
            std::vector<Out>         sink(replayed.outputsFor(back));
            replayed.process(std::span<const S>(xs.data() + at - back, back), std::span<Out>(sink));

            std::vector<Out> a(primed.outputsFor(rest));
            std::vector<Out> b(replayed.outputsFor(rest));
            primed.process(std::span<const S>(xs.data() + at, rest), std::span<Out>(a));
            replayed.process(std::span<const S>(xs.data() + at, rest), std::span<Out>(b));
            expect(that % (a == b)) << "M=" << m << ": the accessor path and the replay are the same stream";

            // Both are what the new taps would have produced had they run from the start, which is
            // the alignment claim: output k is still the filter applied at input k*M.
            const std::vector<Out> whole = firWhole<S, H>(m, after, xs);
            expect(ge(whole.size(), a.size()));
            expect(that % std::equal(a.begin(), a.end(), whole.begin() + static_cast<std::ptrdiff_t>(whole.size() - a.size()))) << "M=" << m << ": alignment across the change";
        }
    };

    "a tap set that grew starts on zeros, exactly as a stream start does"_test = [] {
        using S   = CF;
        using H   = float;
        using Out = PolyphaseResampler<S, H>::output_type;

        const std::vector<H> taps = noise(201UZ, 829U);
        const std::vector<S> x    = asStream<S>(noise(1024UZ, 839U), noise(1024UZ, 853U));

        PolyphaseResampler<S, H> shortHistory(1UZ, 1UZ, std::span<const H>(noise(31UZ, 857U)));
        std::vector<Out>         head(shortHistory.outputsFor(512UZ));
        shortHistory.process(std::span<const S>(x.data(), 512UZ), std::span<Out>(head));

        PolyphaseResampler<S, H> grown(1UZ, 1UZ, std::span<const H>(taps));
        grown.primeWith(shortHistory.window());

        std::vector<S> padded(grown.branchLength() - 1UZ, S{});
        std::ranges::copy(shortHistory.window(), padded.end() - static_cast<std::ptrdiff_t>(shortHistory.window().size()));
        expect(that % std::equal(grown.window().begin(), grown.window().end(), padded.begin())) << "the newest samples are kept and the older positions are zeros";
    };

    "priming reaches the state a filter length of replay reaches"_test = [] {
        using S   = CF;
        using H   = float;
        using Out = PolyphaseResampler<S, H>::output_type;

        const std::vector<H> taps = noise(11001UZ, 907U);
        const std::vector<S> x    = asStream<S>(noise(11001UZ, 911U), noise(11001UZ, 913U));

        PolyphaseResampler<S, H> source(1UZ, 1UZ, std::span<const H>(taps));
        std::vector<Out>         seen(source.outputsFor(x.size()));
        source.process(std::span<const S>(x), std::span<Out>(seen));

        PolyphaseResampler<S, H> primed(1UZ, 1UZ, std::span<const H>(taps));
        const auto               t0 = std::chrono::steady_clock::now();
        primed.primeWith(source.window());
        const auto t1 = std::chrono::steady_clock::now();

        PolyphaseResampler<S, H> replayed(1UZ, 1UZ, std::span<const H>(taps));
        const std::size_t        history = replayed.branchLength() - 1UZ;
        std::vector<Out>         discard(replayed.outputsFor(history));
        const auto               t2 = std::chrono::steady_clock::now();
        replayed.process(std::span<const S>(x.data() + x.size() - history, history), std::span<Out>(discard));
        const auto t3 = std::chrono::steady_clock::now();

        expect(that % std::equal(primed.window().begin(), primed.window().end(), replayed.window().begin())) << "the two reach the same state";

        const double copyNs   = std::max(std::chrono::duration<double, std::nano>(t1 - t0).count(), 1.0);
        const double replayNs = std::chrono::duration<double, std::nano>(t3 - t2).count();
        std::println("tap change at {} taps: primeWith {:.1f} us, shadow replay {:.1f} us over {} samples", taps.size(), copyNs / 1e3, replayNs / 1e3, history);
    };

    "decimation reads a phase, not a stride"_test = [] {
        forEachCombo([](auto c) {
            using S   = typename decltype(c)::Sample;
            using H   = typename decltype(c)::Tap;
            using Out = typename PolyphaseResampler<S, H>::output_type;

            constexpr std::size_t kM = 10UZ;

            const std::vector<H> unit{makeValue<H>(1.0f, 0.0f)};
            for (const std::size_t q : {0UZ, 5UZ, 10UZ, 37UZ, 40UZ}) {
                std::vector<S> x(64UZ, S{});
                x[q] = makeValue<S>(1.0f, 0.0f);

                const std::vector<Out> y = firWhole<S, H>(kM, unit, x);
                expect(eq(y.size(), 7UZ)) << decltype(c)::name();
                for (std::size_t k = 0UZ; k < y.size(); ++k) {
                    const bool hit = (q % kM == 0UZ) && (k == q / kM);
                    expect(that % (y[k] == (hit ? makeValue<Out>(1.0f, 0.0f) : Out{}))) << decltype(c)::name() << ": impulse at " << q << ", output " << k;
                }
            }

            // an N-tap set against an impulse at zero puts taps[k*M] in output k, which pins the
            // phase and the tap indexing together
            const std::vector<H> taps = asStream<H>(noise(83UZ, 77U), noise(83UZ, 79U));
            std::vector<S>       impulse(200UZ, S{});
            impulse[0UZ] = makeValue<S>(1.0f, 0.0f);

            const std::vector<Out> y = firWhole<S, H>(kM, taps, impulse);
            for (std::size_t k = 0UZ; k < y.size(); ++k) {
                const Out want = (k * kM < taps.size()) ? makeValue<Out>(realOf(taps[k * kM]), imagOf(taps[k * kM])) : Out{};
                expect(that % (y[k] == want)) << decltype(c)::name() << ": output " << k;
            }
        });
    };

    "a composite band-pass is a rotation and a real prototype"_test = [] {
        // freq_xlating(x)[k] == exp(-j*w0*(N-1)/2) * (Rotator -> FIR -> decimate)[k], exactly. The
        // fold and the composition are the same filter, so the tap type decides between them and no
        // third block is needed to hold the choice.
        constexpr int         kN = 143;
        constexpr std::size_t kD = 10UZ;
        const double          w0 = 2.0 * std::numbers::pi * 57000.0 / 240000.0;

        const std::vector<float> proto = gr::filter::fir::design::kaiserLowpass(kN, 0.03125, 60.0);
        std::vector<CF>          composite(proto.size());
        for (std::size_t i = 0UZ; i < proto.size(); ++i) {
            const std::complex<double> v = static_cast<double>(proto[i]) * std::polar(1.0, w0 * static_cast<double>(i));
            composite[i]                 = CF{static_cast<float>(v.real()), static_cast<float>(v.imag())};
        }

        const std::vector<float> x = noise(4096UZ, 83U);

        const std::vector<CF>      folded   = firWhole<float, CF>(kD, composite, x);
        const std::complex<double> constant = std::polar(1.0, -w0 * 0.5 * static_cast<double>(kN - 1));

        std::vector<CF> rotated(x.size());
        for (std::size_t n = 0UZ; n < x.size(); ++n) {
            const std::complex<double> v = static_cast<double>(x[n]) * std::polar(1.0, -w0 * static_cast<double>(n));
            rotated[n]                   = CF{static_cast<float>(v.real()), static_cast<float>(v.imag())};
        }
        const std::vector<CF> composed = firWhole<CF, float>(kD, proto, rotated);
        expect(eq(folded.size(), composed.size()));

        double worst = 0.0;
        double power = 0.0;
        for (std::size_t k = 0UZ; k < composed.size(); ++k) {
            const std::complex<double> fold = widen(folded[k]) * constant * std::polar(1.0, -w0 * static_cast<double>(kD) * static_cast<double>(k));
            worst                           = std::max(worst, std::abs(fold - constant * widen(composed[k])));
            power += std::norm(widen(composed[k]));
        }
        const double rms = std::sqrt(power / static_cast<double>(composed.size()));
        expect(gt(rms, 0.0));
        expect(lt(worst / rms, 1e-5)) << "the two routes differ by " << (worst / rms) << " of output rms";
    };

    "complexBandpass is the folded prototype up to the stated constant"_test = [] {
        constexpr int kN = 143;
        const double  f1 = 0.05;
        const double  f2 = 0.15;
        const double  w0 = 2.0 * std::numbers::pi * 0.5 * (f1 + f2);

        const auto                 window   = gr::filter::fir::design::kaiserFor(60.0);
        const std::vector<CF>      centered = gr::filter::fir::design::complexBandpass(kN, f1, f2, window, 1.0);
        const std::vector<float>   proto    = gr::filter::fir::design::lowpass(kN, 0.5 * (f2 - f1), window, 1.0);
        const std::complex<double> constant = std::polar(1.0, -w0 * 0.5 * static_cast<double>(kN - 1));

        expect(eq(centered.size(), proto.size()));
        double worst = 0.0;
        double peak  = 0.0;
        for (std::size_t i = 0UZ; i < centered.size(); ++i) {
            const std::complex<double> folded = static_cast<double>(proto[i]) * std::polar(1.0, w0 * static_cast<double>(i));
            worst                             = std::max(worst, std::abs(widen(centered[i]) - constant * folded));
            peak                              = std::max(peak, std::abs(widen(centered[i])));
        }
        expect(gt(peak, 0.0));
        expect(lt(worst / peak, 1e-6)) << "the two designs differ by " << (worst / peak) << " of the peak tap";
    };
};

int main() { /* tests are automatically registered and run */ }
