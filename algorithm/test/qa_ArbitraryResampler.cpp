#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <random>
#include <span>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/ArbitraryResampler.hpp>

namespace {

using gr::filter::ArbitraryResampler;
using gr::filter::kArbitraryMask;
using gr::filter::kArbitraryOne;

constexpr double kTwoPi = 2.0 * std::numbers::pi;

[[nodiscard]] std::vector<float> noise(std::size_t n, std::uint32_t seed) {
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> uniform(-0.7f, 0.7f);
    std::vector<float>                    x(n);
    for (float& v : x) {
        v = uniform(rng);
    }
    return x;
}

[[nodiscard]] std::uint64_t stepFor(std::size_t bank, double rate) { return static_cast<std::uint64_t>(std::round(static_cast<double>(bank) / rate * static_cast<double>(kArbitraryOne))); }

/**
 * @brief The definition itself, in double: zero-stuff by L, convolve with the padded prototype, read
 *        the interpolated grid at the same fixed-point positions with the same Lagrange weights.
 *
 * Deliberately not a second bank: an arm-index or window-offset error reproduced in both would pass
 * a comparison against one, and fails against this.
 */
[[nodiscard]] std::vector<double> arbitraryByDefinition(std::size_t bank, std::uint64_t step, int order, const std::vector<float>& taps, const std::vector<float>& x, std::size_t nOutput) {
    const std::size_t   b = (taps.size() + bank - 1UZ) / bank;
    std::vector<double> h(bank * b, 0.0);
    for (std::size_t i = 0UZ; i < taps.size(); ++i) {
        h[i] = static_cast<double>(taps[i]);
    }

    std::vector<double> u(bank * (b + x.size()), 0.0);
    for (std::size_t i = 0UZ; i < x.size(); ++i) {
        u[(b + i) * bank] = static_cast<double>(x[i]);
    }

    const auto interpolated = [&](std::size_t t) {
        double acc = 0.0;
        for (std::size_t m = 0UZ; m < h.size(); ++m) {
            if (m <= t && (t - m) < u.size()) {
                acc += h[m] * u[t - m];
            }
        }
        return acc;
    };

    std::vector<double> y(nOutput);
    std::uint64_t       ahead = bank;
    std::uint64_t       frac  = 0ULL;
    for (std::size_t k = 0UZ; k < nOutput; ++k) {
        // the kernel's anchor is floor(pos/L), one input sample behind the window's newest sample,
        // which the wrap arms read and every arm below L multiplies by a zero tap
        const std::size_t grid = b * bank + static_cast<std::size_t>(ahead) - bank;
        const double      mu   = static_cast<double>(frac) / static_cast<double>(kArbitraryOne);

        if (order == 0) {
            y[k] = interpolated(grid);
        } else if (order == 1) {
            y[k] = (1.0 - mu) * interpolated(grid) + mu * interpolated(grid + 1UZ);
        } else {
            const double w[4] = {-mu * (mu - 1.0) * (mu - 2.0) / 6.0, (mu + 1.0) * (mu - 1.0) * (mu - 2.0) / 2.0, -(mu + 1.0) * mu * (mu - 2.0) / 2.0, (mu + 1.0) * mu * (mu - 1.0) / 6.0};
            y[k]              = w[0] * interpolated(grid - 1UZ) + w[1] * interpolated(grid) + w[2] * interpolated(grid + 1UZ) + w[3] * interpolated(grid + 2UZ);
        }

        frac += step & kArbitraryMask;
        ahead += step >> gr::filter::kArbitraryFractionBits;
        if (frac >= kArbitraryOne) {
            frac -= kArbitraryOne;
            ++ahead;
        }
    }
    return y;
}

/// One DFT bin of a real record, normalized so a unit-amplitude cosine at that bin reads 1.
[[nodiscard]] double binAmplitude(const std::vector<float>& y, std::size_t bin) {
    const double         n = static_cast<double>(y.size());
    std::complex<double> acc{};
    for (std::size_t i = 0UZ; i < y.size(); ++i) {
        acc += static_cast<double>(y[i]) * std::polar(1.0, -kTwoPi * static_cast<double>(bin) * static_cast<double>(i) / n);
    }
    return 2.0 * std::abs(acc) / n;
}

/// `sum_r t[r] * exp(-j*2*pi*f*r)`, the response of one arm read as a filter at the input rate.
[[nodiscard]] std::complex<double> armResponse(const std::vector<double>& t, double f) {
    std::complex<double> acc{};
    for (std::size_t r = 0UZ; r < t.size(); ++r) {
        acc += t[r] * std::polar(1.0, -kTwoPi * f * static_cast<double>(r));
    }
    return acc;
}

[[nodiscard]] std::vector<float> run(ArbitraryResampler<float>& kernel, const std::vector<float>& x) {
    std::vector<float> y(kernel.outputsFor(x.size()));
    kernel.process(std::span<const float>(x), std::span<float>(y));
    return y;
}

} // namespace

const boost::ut::suite<"arbitrary resampler"> arbitraryResamplerTests = [] {
    using namespace boost::ut;
    using gr::filter::arbitraryBankSize;
    using gr::filter::designArbitraryResampler;
    using gr::filter::mapArbitraryOffset;
    using gr::filter::mapResampledOffset;
    using gr::filter::ResamplerDesign;

    "the bank is the definition"_test = [] {
        struct Row {
            std::size_t bank;
            double      rate;
            int         order;
            std::size_t outputs;
        };
        constexpr Row kRows[] = {{32UZ, 19.0 / 24.0, 1, 475UZ}, {16UZ, 1.7, 1, 1019UZ}, {32UZ, 0.2, 1, 120UZ}, {32UZ, 1.31, 3, 785UZ}};

        for (const Row& row : kRows) {
            // a noise prototype rather than a designed one: a smooth low-pass hides an arm-index
            // error that random taps do not
            const std::vector<float>   taps = noise(12UZ * row.bank + 5UZ, 3U);
            const std::vector<float>   x    = noise(1200UZ, 7U + static_cast<std::uint32_t>(row.bank));
            ArbitraryResampler<double> kernel(row.rate, row.bank, row.order, std::span<const float>(taps));

            const std::size_t made = std::min(row.outputs, kernel.outputsFor(x.size()));
            expect(eq(made, row.outputs)) << row.bank << " at " << row.rate;

            std::vector<double> in(x.size());
            for (std::size_t i = 0UZ; i < x.size(); ++i) {
                in[i] = static_cast<double>(x[i]);
            }
            std::vector<double> got(kernel.outputsFor(x.size()));
            kernel.process(std::span<const double>(in), std::span<double>(got));

            const std::vector<double> want = arbitraryByDefinition(row.bank, stepFor(row.bank, row.rate), row.order, taps, x, made);

            double worst = 0.0;
            double scale = 0.0;
            for (std::size_t k = 0UZ; k < made; ++k) {
                worst = std::max(worst, std::abs(got[k] - want[k]));
                scale = std::max(scale, std::abs(want[k]));
            }
            expect(gt(scale, 0.0));
            expect(lt(worst / scale, 1e-12)) << row.bank << " arms at " << row.rate << ", order " << row.order << ": worst relative departure " << (worst / scale);
        }
    };

    "the realized ratio is a rational the kernel reports"_test = [] {
        const std::vector<float> taps = noise(320UZ, 11U);

        for (const double rate : {0.2, 19.0 / 24.0, 48.0 / 44.1, 0.7912345678, 1.0 / std::numbers::pi}) {
            const ArbitraryResampler<float> kernel(rate, 32UZ, 1, std::span<const float>(taps));

            const double realized = kernel.realizedRate();
            expect(eq(realized, 32.0 * static_cast<double>(kArbitraryOne) / static_cast<double>(kernel.step()))) << "reported as L*2^F/step";

            // |step/2^F - L/r| <= 1/2, so the realized ratio sits inside 2^-(F+1)*r/L of the request
            const double relative = std::abs(realized - rate) / rate;
            expect(le(relative, std::ldexp(1.0, -33) * rate / 32.0)) << "requested " << rate << ", realized " << realized << " at " << relative;
        }

        // L/r a whole number is exactly realizable, and 0.2 at 32 arms is one
        const ArbitraryResampler<float> exact(0.2, 32UZ, 1, std::span<const float>(taps));
        expect(eq(exact.step(), 160ULL * kArbitraryOne));
        expect(eq(exact.realizedRate(), 0.2)) << "to the last bit";

        // 19/24 is the worst row for a float-derived step: 0.011 samples a day at 48 kHz here
        // against 165 for the float form
        const ArbitraryResampler<float> rds(19.0 / 24.0, 32UZ, 1, std::span<const float>(taps));
        const double                    relative = std::abs(rds.realizedRate() - 19.0 / 24.0) / (19.0 / 24.0);
        expect(approx(relative, 2.728e-12, 0.03e-12)) << "realized-ratio error at 19/24: " << relative;
        expect(lt(relative * 48000.0 * 86400.0, 0.02)) << "samples a day at 48 kHz";
    };

    "the counts are exact in both directions"_test = [] {
        const std::vector<float> taps = noise(320UZ, 13U);
        for (const double rate : {0.05, 0.2, 19.0 / 24.0, 1.7, 20.0}) {
            const ArbitraryResampler<float> kernel(rate, 32UZ, 1, std::span<const float>(taps));
            expect(eq(kernel.outputsFor(0UZ), 0UZ)) << rate;

            for (std::size_t n = 1UZ; n <= 100000UZ; n = (n < 2000UZ) ? n + 1UZ : n + 997UZ) {
                const std::size_t m = kernel.outputsFor(n);
                if (m > 0UZ) {
                    expect(le(kernel.inputsFor(m), n + 1UZ)) << rate << " at n=" << n;
                }
                expect(ge(kernel.outputsFor(kernel.inputsFor(n)), n)) << rate << " at m=" << n;
            }
        }
    };

    "the ratio holds over a long stream"_test = [] {
        const std::vector<float>  taps = noise(320UZ, 17U);
        ArbitraryResampler<float> kernel(19.0 / 24.0, 32UZ, 1, std::span<const float>(taps));

        const std::vector<float> block(4096UZ, 0.25f);
        std::vector<float>       out(kernel.outputsFor(block.size()) + 4UZ);

        std::size_t fed = 0UZ;
        for (int i = 0; i < 256; ++i) {
            const std::size_t made = kernel.outputsFor(block.size());
            expect(eq(kernel.process(std::span<const float>(block), std::span<float>(out.data(), made)), made));
            fed += block.size();
        }

        // the count a chunked run delivers is the count one call would have, exactly
        ArbitraryResampler<float> whole(19.0 / 24.0, 32UZ, 1, std::span<const float>(taps));
        expect(eq(kernel.produced(), std::uint64_t{whole.outputsFor(fed)}));

        const double drift = static_cast<double>(kernel.produced()) - static_cast<double>(fed) * kernel.realizedRate();
        expect(lt(std::abs(drift), 1.0)) << "departure from the realized ratio over " << fed << " samples: " << drift;
    };

    "one call or many, bit for bit"_test = [] {
        for (const double rate : {0.2, 19.0 / 24.0, 1.7}) {
            for (const int order : {1, 3}) {
                const std::vector<float> taps = noise(320UZ, 19U);
                const std::vector<float> x    = noise(1UZ << 13, 23U);

                ArbitraryResampler<float> reference(rate, 32UZ, order, std::span<const float>(taps));
                const std::vector<float>  whole = run(reference, x);
                expect(gt(whole.size(), 0UZ));

                for (const std::size_t chunk : {1UZ, 3UZ, 17UZ, 4096UZ}) {
                    ArbitraryResampler<float> kernel(rate, 32UZ, order, std::span<const float>(taps));
                    std::vector<float>        y;
                    for (std::size_t at = 0UZ; at < x.size(); at += chunk) {
                        const std::size_t take = std::min(chunk, x.size() - at);
                        const std::size_t made = kernel.outputsFor(take);
                        y.resize(y.size() + made);
                        kernel.process(std::span<const float>(x.data() + at, take), std::span<float>(y.data() + y.size() - made, made));
                    }
                    expect(that % (y == whole)) << rate << " order " << order << " in chunks of " << chunk;
                }

                ArbitraryResampler<float> ragged(rate, 32UZ, order, std::span<const float>(taps));
                std::mt19937              rng(29U);
                std::vector<float>        y;
                for (std::size_t at = 0UZ; at < x.size();) {
                    const std::size_t take = std::min<std::size_t>(1UZ + (rng() % 1500U), x.size() - at);
                    const std::size_t made = ragged.outputsFor(take);
                    y.resize(y.size() + made);
                    ragged.process(std::span<const float>(x.data() + at, take), std::span<float>(y.data() + y.size() - made, made));
                    at += take;
                }
                expect(that % (y == whole)) << rate << " order " << order << " in random chunks";
            }
        }
    };

    "the bank never reads outside the window"_test = [] {
        // The span the caller handed over is the whole of what is touched, so poisoning the memory
        // just past it cannot move an output. A kernel reading forward from each anchor instead
        // would consume past the span and this would catch it.
        const std::vector<float> taps = noise(320UZ, 31U);
        const std::vector<float> body = noise(3000UZ, 37U);

        for (const double rate : {0.2, 19.0 / 24.0, 1.7, 20.0}) {
            for (const int order : {1, 3}) {
                std::vector<float> quiet = body;
                std::vector<float> loud  = body;
                quiet.resize(body.size() + 64UZ, 0.0f);
                loud.resize(body.size() + 64UZ, 1.0e9f);

                ArbitraryResampler<float> a(rate, 32UZ, order, std::span<const float>(taps));
                ArbitraryResampler<float> b(rate, 32UZ, order, std::span<const float>(taps));

                const std::size_t  made = a.outputsFor(body.size());
                std::vector<float> ya(made);
                std::vector<float> yb(made);
                a.process(std::span<const float>(quiet.data(), body.size()), std::span<float>(ya));
                b.process(std::span<const float>(loud.data(), body.size()), std::span<float>(yb));
                expect(that % (ya == yb)) << rate << " order " << order << ": something past the span was read";
            }
        }
    };

    "the bank size follows the attenuation target"_test = [] {
        expect(eq(arbitraryBankSize(40.0, 0.2, 1), 16UZ));
        expect(eq(arbitraryBankSize(60.0, 0.2, 1), 32UZ));
        expect(eq(arbitraryBankSize(80.0, 0.2, 1), 128UZ));
        expect(eq(arbitraryBankSize(100.0, 0.2, 1), 512UZ));
        expect(eq(arbitraryBankSize(40.0, 0.2, 3), 4UZ));
        expect(eq(arbitraryBankSize(60.0, 0.2, 3), 8UZ));
        expect(eq(arbitraryBankSize(80.0, 0.2, 3), 16UZ));
        expect(eq(arbitraryBankSize(100.0, 0.2, 3), 32UZ));
    };

    "the prototype's length per arm does not move with the bank"_test = [] {
        struct Row {
            std::size_t bank;
            double      rate;
            double      stopEdge;
            int         estimated;
            int         searched;
            std::size_t perArm;
        };
        constexpr Row kRows[] = {
            {8UZ, 1.0, 0.0625, 291, 291, 37UZ},               //
            {16UZ, 1.0, 0.03125, 581, 581, 37UZ},             //
            {32UZ, 1.0, 0.015625, 1161, 1159, 37UZ},          //
            {32UZ, 2.0, 0.015625, 1161, 1159, 37UZ},          //
            {32UZ, 19.0 / 24.0, 0.0123698, 1465, 1463, 46UZ}, //
        };
        const auto check = [](const Row& row) {
            const double stopEdge = 0.5 * std::min(1.0, row.rate) / static_cast<double>(row.bank);
            expect(approx(stopEdge, row.stopEdge, 1e-7)) << row.bank << " at " << row.rate;
            expect(eq(gr::filter::fir::design::kaiserLength(60.0, 0.2 * stopEdge) | 1UZ, static_cast<std::size_t>(row.estimated))) << row.bank << " at " << row.rate << ": Kaiser's estimate";

            const ResamplerDesign design = designArbitraryResampler(row.bank, row.rate);
            expect(that % design.ok) << row.bank << " at " << row.rate << ": the search met both targets";
            expect(eq(design.designLength, row.searched)) << row.bank << " at " << row.rate << ": searched length";
            expect(lt(design.stopbandDb, -60.0)) << row.bank << " at " << row.rate << ": stopband " << design.stopbandDb;

            const std::vector<float>  taps = design.taps;
            ArbitraryResampler<float> kernel(std::max(row.rate, 0.2), row.bank, 1, std::span<const float>(taps));
            expect(eq(kernel.tapsPerArm(), row.perArm)) << row.bank << " at " << row.rate << ": taps per arm";
        };

        for (const Row& row : kRows) {
            check(row);
        }
    };

    "a constant comes out a constant"_test = [] {
        for (const double rate : {19.0 / 24.0, 1.0, 48.0 / 44.1, 1.7}) {
            const ResamplerDesign     design = designArbitraryResampler(32UZ, rate);
            ArbitraryResampler<float> kernel(rate, 32UZ, 1, std::span<const float>(design.taps));

            const std::vector<float> x(6UZ * kernel.tapsPerArm() + 4096UZ, 1.0f);
            const std::vector<float> y = run(kernel, x);

            double worst = 0.0;
            for (std::size_t k = 2UZ * kernel.tapsPerArm(); k < y.size(); ++k) {
                worst = std::max(worst, std::abs(static_cast<double>(y[k]) - 1.0));
            }
            const double worstDb = 20.0 * std::log10(1.0 + worst);
            expect(lt(worstDb, 0.1)) << rate << ": settles " << worstDb << " dB off unity";
        }
    };

    "the prototype's cutoff scales with the rate"_test = [] {
        // A tone inside the input band and above the output Nyquist has to be removed before the
        // rate change. The scaled prototype removes it; the r >= 1 law passes it at full amplitude,
        // and that counter-test is what stops the scaling being dropped.
        constexpr std::size_t kBank = 32UZ;
        constexpr double      kRate = 0.2;
        constexpr std::size_t kOut  = 1024UZ;

        const std::size_t  nIn = 5UZ * (kOut + 512UZ);
        std::vector<float> x(nIn);
        for (std::size_t i = 0UZ; i < nIn; ++i) {
            x[i] = static_cast<float>(std::cos(kTwoPi * 0.25 * static_cast<double>(i)));
        }

        const auto alias = [&](const ResamplerDesign& design) {
            ArbitraryResampler<float> kernel(kRate, kBank, 1, std::span<const float>(design.taps));
            const std::vector<float>  y = run(kernel, x);
            expect(ge(y.size(), kOut));
            const std::vector<float> steady(y.end() - static_cast<std::ptrdiff_t>(kOut), y.end());
            return 20.0 * std::log10(std::max(binAmplitude(steady, kOut / 4UZ), 1e-300));
        };

        const double scaled = alias(designArbitraryResampler(kBank, kRate));
        expect(lt(scaled, -75.0)) << "the scaled prototype leaves the alias at " << scaled << " dB";

        const double unscaled = alias(designArbitraryResampler(kBank, 1.0));
        expect(gt(unscaled, -3.0)) << "the unscaled prototype passes the alias at " << unscaled << " dB";
        expect(gt(unscaled - scaled, 70.0)) << "the scaling law is worth " << (unscaled - scaled) << " dB";
    };

    "arm zero is the decimation filter"_test = [] {
        constexpr std::size_t kBank  = 32UZ;
        const ResamplerDesign design = designArbitraryResampler(kBank, 0.2);

        std::vector<double> arm;
        for (std::size_t r = 0UZ; r * kBank < design.taps.size(); ++r) {
            arm.push_back(static_cast<double>(design.taps[r * kBank]));
        }

        const double atDc = std::abs(armResponse(arm, 0.0));
        expect(lt(std::abs(20.0 * std::log10(atDc)), 0.05)) << "arm 0 at DC: " << atDc;

        double worst = 0.0;
        for (int i = 0; i <= 800; ++i) {
            const double f = 0.1 + 0.4 * static_cast<double>(i) / 800.0;
            worst          = std::max(worst, std::abs(armResponse(arm, f)));
        }
        const double worstDb = 20.0 * std::log10(worst / atDc);
        expect(lt(worstDb, -59.0)) << "arm 0 above the r=0.2 output Nyquist: " << worstDb << " dB";
    };

    "tag offsets are integer arithmetic"_test = [] {
        // Wherever the rate makes the step a whole number of interpolated samples, the map is
        // mapResampledOffset, which is the check that the two families cannot drift apart.
        const auto agrees = [](std::size_t bank, std::uint64_t l, std::uint64_t m) {
            const std::uint64_t step = stepFor(bank, static_cast<double>(l) / static_cast<double>(m));
            expect(eq(step & kArbitraryMask, 0ULL)) << l << "/" << m << " at " << bank << " arms: a whole number of interpolated samples";
            for (std::uint64_t i = 0ULL; i < 500ULL; ++i) {
                expect(eq(mapArbitraryOffset(i, bank, step, 0), mapResampledOffset(i, l, m))) << l << "/" << m << " at " << i;
            }
        };
        agrees(32UZ, 1ULL, 5ULL);
        agrees(19UZ, 19ULL, 24ULL);
        agrees(38UZ, 19ULL, 24ULL);

        const std::uint64_t slow = stepFor(32UZ, 0.2);
        expect(eq(mapArbitraryOffset(0ULL, 32UZ, slow, 0), 0ULL)) << "offset zero stays offset zero";
        for (std::uint64_t i = 3ULL; i <= 7ULL; ++i) { // five consecutive inputs on one output offset
            expect(eq(mapArbitraryOffset(i, 32UZ, slow, 0), 1ULL)) << "at " << i;
        }

        const std::uint64_t fast = stepFor(32UZ, 1.7);
        for (std::uint64_t i = 1ULL; i < 200ULL; ++i) {
            expect(gt(mapArbitraryOffset(i, 32UZ, fast, 0), mapArbitraryOffset(i - 1ULL, 32UZ, fast, 0))) << "strictly increasing above unity, at " << i;
        }

        // and the offset survives where a float or a double would already have lost it
        constexpr std::uint64_t k2p24 = (1ULL << 24) + 1ULL;
        expect(eq(mapArbitraryOffset(k2p24, 32UZ, stepFor(32UZ, 1.0 / 3.0), 0), mapResampledOffset(k2p24, 1ULL, 3ULL)));
        expect(eq(static_cast<std::uint64_t>(static_cast<float>(k2p24)), k2p24 - 1ULL)) << "the float has already lost it";
    };

    "a rate change keeps the position"_test = [] {
        const std::vector<float>  taps = noise(320UZ, 41U);
        ArbitraryResampler<float> kernel(19.0 / 24.0, 32UZ, 1, std::span<const float>(taps));

        const std::vector<float> x = noise(2000UZ, 43U);
        std::ignore                = run(kernel, x);

        const std::int64_t before = kernel.phase();
        kernel.setRate(1.31);
        expect(eq(kernel.phase(), before));
        expect(eq(kernel.realizedRate(), 32.0 * static_cast<double>(kArbitraryOne) / static_cast<double>(kernel.step())));
        expect(approx(kernel.realizedRate(), 1.31, 1e-9));

        const std::size_t made = kernel.outputsFor(x.size());
        expect(gt(made, 0UZ));
        std::vector<float> more(made);
        expect(eq(kernel.process(std::span<const float>(x), std::span<float>(more)), made)) << "and the stream continues";
    };

    "a prototype change keeps the window and the phase"_test = [] {
        // QA for the rebuild a lowered rate forces: same taps in, and the output is unchanged.
        const std::vector<float> taps = noise(320UZ, 47U);
        const std::vector<float> x    = noise(4000UZ, 53U);

        ArbitraryResampler<float> plain(0.7, 32UZ, 1, std::span<const float>(taps));
        ArbitraryResampler<float> rebuilt(0.7, 32UZ, 1, std::span<const float>(taps));

        const std::size_t  half = 1500UZ;
        std::vector<float> a(plain.outputsFor(half));
        std::vector<float> b(rebuilt.outputsFor(half));
        plain.process(std::span<const float>(x.data(), half), std::span<float>(a));
        rebuilt.process(std::span<const float>(x.data(), half), std::span<float>(b));

        rebuilt.setTaps(std::span<const float>(taps));
        expect(eq(rebuilt.phase(), plain.phase())) << "the phase survives the rebuild";

        const std::size_t  rest = x.size() - half;
        std::vector<float> c(plain.outputsFor(rest));
        std::vector<float> d(rebuilt.outputsFor(rest));
        expect(eq(c.size(), d.size()));
        plain.process(std::span<const float>(x.data() + half, rest), std::span<float>(c));
        rebuilt.process(std::span<const float>(x.data() + half, rest), std::span<float>(d));
        expect(that % (c == d)) << "and so does the window, to the bit";

        // a longer prototype keeps what the old window held and zero-fills the deficit
        const std::vector<float> longer = noise(640UZ, 59U);
        rebuilt.setTaps(std::span<const float>(longer));
        expect(eq(rebuilt.tapsPerArm(), 20UZ));
        expect(eq(rebuilt.phase(), plain.phase()));
    };

    "a rate of one is not a pass-through"_test = [] {
        // At rate one the step is L*2^F, mu is identically zero and only arm 0 runs, which is a
        // unit-gain filter rather than the identity. Asserted so that a fast path added later has
        // to keep the same answer; a caller at exactly one is better served without a resampler.
        const ResamplerDesign     design = designArbitraryResampler(32UZ, 1.0);
        ArbitraryResampler<float> kernel(1.0, 32UZ, 1, std::span<const float>(design.taps));
        expect(eq(kernel.step(), 32ULL * kArbitraryOne));

        constexpr std::size_t kOut = 1024UZ;
        constexpr std::size_t kBin = 51UZ;

        std::vector<float> x(3UZ * kOut);
        for (std::size_t i = 0UZ; i < x.size(); ++i) {
            x[i] = static_cast<float>(std::cos(kTwoPi * static_cast<double>(kBin) * static_cast<double>(i) / static_cast<double>(kOut)));
        }
        const std::vector<float> y = run(kernel, x);
        expect(eq(y.size(), x.size() - 1UZ)) << "one output per input, after one sample of latency";

        const std::vector<float> steady(y.end() - static_cast<std::ptrdiff_t>(kOut), y.end());
        const double             gainDb = 20.0 * std::log10(binAmplitude(steady, kBin));
        expect(lt(std::abs(gainDb), 0.1)) << "arm 0 delivers " << gainDb << " dB";

        expect(that % !std::ranges::equal(std::span<const float>(y.data() + 512UZ, 256UZ), std::span<const float>(x.data() + 513UZ, 256UZ))) << "and it is not the input";
    };

    "degenerate settings"_test = [] {
        const std::vector<float> taps = noise(320UZ, 67U);
        expect(throws<std::invalid_argument>([&] { ArbitraryResampler<float>(0.0, 32UZ, 1, std::span<const float>(taps)); }));
        expect(throws<std::invalid_argument>([&] { ArbitraryResampler<float>(-1.0, 32UZ, 1, std::span<const float>(taps)); }));
        expect(throws<std::invalid_argument>([&] { ArbitraryResampler<float>(1.0, 0UZ, 1, std::span<const float>(taps)); }));
        expect(throws<std::invalid_argument>([&] { ArbitraryResampler<float>(1.0, 32UZ, 2, std::span<const float>(taps)); }));
        expect(throws<std::invalid_argument>([&] { ArbitraryResampler<float>(1.0, 32UZ, 1, std::span<const float>()); }));
        expect(throws<std::invalid_argument>([&] { std::ignore = designArbitraryResampler(0UZ, 1.0); }));

        // a rate above L*2^F drives the step to zero, and the guard exists to stop the loop there
        expect(throws<std::invalid_argument>([&] { ArbitraryResampler<float>(1.5e11, 32UZ, 1, std::span<const float>(taps)); }));

        // one arm is a plain fractional-delay filter, and it is legal
        const std::vector<float>  three{0.25f, 0.5f, 0.25f};
        ArbitraryResampler<float> single(0.5, 1UZ, 1, std::span<const float>(three));
        expect(eq(single.bankSize(), 1UZ));
        expect(eq(single.tapsPerArm(), 3UZ));
        const std::vector<float> x = noise(64UZ, 71U);
        expect(gt(run(single, x).size(), 0UZ));

        // a prototype shorter than the bank is zero-padded and works
        ArbitraryResampler<float> stubby(0.9, 32UZ, 1, std::span<const float>(three));
        expect(eq(stubby.tapsPerArm(), 1UZ));
        expect(gt(run(stubby, x).size(), 0UZ));
    };

    // an arm-by-arm sweep over 33 offsets and 257 frequencies, per bank and order
    "the interpolation error follows the order and the bank"_test = [] {
        struct Row {
            std::size_t bank;
            int         order;
            double      db;
        };
        constexpr Row kRows[] = {{8UZ, 0, -16.33}, {8UZ, 1, -37.50}, {8UZ, 3, -57.50}, {16UZ, 1, -47.70}, {32UZ, 1, -54.62}, {32UZ, 3, -57.23}};

        for (const Row& row : kRows) {
            const ResamplerDesign design = designArbitraryResampler(row.bank, 1.0);
            const std::size_t     b      = (design.taps.size() + row.bank - 1UZ) / row.bank;
            const double          dg     = 0.5 * static_cast<double>(design.designLength - 1) / static_cast<double>(row.bank);

            const auto armTaps = [&](std::ptrdiff_t p) {
                std::vector<double> t(b + 1UZ, 0.0);
                for (std::size_t r = 0UZ; r <= b; ++r) {
                    const std::ptrdiff_t at = p + (static_cast<std::ptrdiff_t>(r) - 1) * static_cast<std::ptrdiff_t>(row.bank);
                    if (at >= 0 && static_cast<std::size_t>(at) < design.taps.size()) {
                        t[r] = static_cast<double>(design.taps[static_cast<std::size_t>(at)]);
                    }
                }
                return t;
            };

            double worst = 0.0;
            for (std::size_t p = 0UZ; p < row.bank; ++p) {
                for (int j = 0; j < 33; ++j) {
                    const double mu   = static_cast<double>(j) / 32.0;
                    const double w[4] = {-mu * (mu - 1.0) * (mu - 2.0) / 6.0, (mu + 1.0) * (mu - 1.0) * (mu - 2.0) / 2.0, -(mu + 1.0) * mu * (mu - 2.0) / 2.0, (mu + 1.0) * mu * (mu - 1.0) / 6.0};

                    // order 0 rounds to the nearer arm rather than truncating to the one below,
                    // which is what makes its error h/2 and not h
                    const double nearest[2] = {(mu < 0.5) ? 1.0 : 0.0, (mu < 0.5) ? 0.0 : 1.0};

                    std::vector<double> blended(b + 1UZ, 0.0);
                    for (int s = (row.order == 3 ? -1 : 0); s <= (row.order == 3 ? 2 : 1); ++s) {
                        const double              weight = (row.order == 3) ? w[s + 1] : ((row.order == 0) ? nearest[s] : (s == 0 ? 1.0 - mu : mu));
                        const std::vector<double> t      = armTaps(static_cast<std::ptrdiff_t>(p) + s);
                        for (std::size_t r = 0UZ; r <= b; ++r) {
                            blended[r] += weight * t[r];
                        }
                    }

                    for (int i = 0; i <= 256; ++i) {
                        const double f     = 0.4 * static_cast<double>(i) / 256.0;
                        const auto   ideal = std::polar(1.0, -kTwoPi * f * (dg - (static_cast<double>(p) + mu) / static_cast<double>(row.bank)));
                        worst              = std::max(worst, std::abs(armResponse(blended, f) * std::polar(1.0, kTwoPi * f) - ideal));
                    }
                }
            }
            const double db = 20.0 * std::log10(worst);
            expect(approx(db, row.db, 0.6)) << row.bank << " arms at order " << row.order << ": " << db << " dB";
        }
    };
};

int main() { /* tests are automatically registered and run */ }
