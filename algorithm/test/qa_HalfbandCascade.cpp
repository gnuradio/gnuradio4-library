#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/FilterDesign.hpp>
#include <gnuradio-4.0/algorithm/filter/HalfbandCascade.hpp>

namespace {

using CF = std::complex<float>;
using CD = std::complex<double>;

/// Noise with a fixed seed: every check here is about arithmetic, so the signal only has to be
/// one whose every sample differs from its neighbors.
[[nodiscard]] std::vector<CF> noise(std::size_t n, std::uint32_t seed) {
    std::mt19937                          rng(seed);
    std::uniform_real_distribution<float> uniform(-0.7f, 0.7f);
    std::vector<CF>                       x(n);
    for (CF& v : x) {
        v = CF{uniform(rng), uniform(rng)};
    }
    return x;
}

/// Halving tap sets: a lowpass at a quarter of the rate is a halfband, which makes every tap at
/// an even offset from the center a zero the cascade may skip.
[[nodiscard]] std::vector<std::vector<float>> halfbandStages(std::span<const int> lengths) {
    std::vector<std::vector<float>> stages;
    for (const int n : lengths) {
        stages.push_back(gr::filter::fir::design::kaiserLowpass(static_cast<std::size_t>(n), 0.25, 80.0));
    }
    return stages;
}

/// One halving stage written out as its own definition, in double: output j is the taps read
/// forward from the sample it is handed. This is the reference the fast kernel is checked against.
[[nodiscard]] std::vector<CD> stageByDefinition(const std::vector<float>& taps, const std::vector<CD>& x) {
    std::vector<CD> y;
    if (x.size() < taps.size()) {
        return y;
    }
    y.resize((x.size() - taps.size()) / 2UZ + 1UZ);
    for (std::size_t j = 0UZ; j < y.size(); ++j) {
        CD acc{};
        for (std::size_t i = 0UZ; i < taps.size(); ++i) {
            acc += static_cast<double>(taps[i]) * x[2UZ * j + i];
        }
        y[j] = acc;
    }
    return y;
}

[[nodiscard]] std::vector<CD> asDouble(const std::vector<CF>& x) {
    std::vector<CD> out(x.size());
    std::ranges::transform(x, out.begin(), [](CF v) { return CD(static_cast<double>(v.real()), static_cast<double>(v.imag())); });
    return out;
}

[[nodiscard]] double worstDeviation(const std::vector<CF>& got, const std::vector<CD>& want, std::size_t count) {
    double worst = 0.0;
    for (std::size_t i = 0UZ; i < count; ++i) {
        worst = std::max(worst, std::abs(CD(static_cast<double>(got[i].real()), static_cast<double>(got[i].imag())) - want[i]));
    }
    return worst;
}

[[nodiscard]] std::vector<CF> filterWhole(const std::vector<std::vector<float>>& stages, const std::vector<CF>& x) {
    gr::filter::HalfbandCascade cascade{std::span<const std::vector<float>>(stages)};
    std::vector<CF>             out;
    cascade.push(std::span<const CF>(x), out);
    return out;
}

} // namespace

const boost::ut::suite<"halfband cascade"> halfbandCascadeTests = [] {
    using namespace boost::ut;
    using gr::filter::HalfbandCascade;

    constexpr int kLengths[] = {31, 23, 15};

    "output is the convolution computed by definition"_test = [&] {
        for (std::size_t stageCount = 1UZ; stageCount <= 3UZ; ++stageCount) {
            const std::vector<std::vector<float>> stages = halfbandStages(std::span<const int>(kLengths, stageCount));
            const std::vector<CF>                 x      = noise(1UZ << 14, 3U + static_cast<std::uint32_t>(stageCount));

            const std::vector<CF> got = filterWhole(stages, x);

            std::vector<CD> want = asDouble(x);
            for (const std::vector<float>& taps : stages) {
                want = stageByDefinition(taps, want);
            }

            expect(ge(got.size(), want.size())) << "the cascade delivers at least what the definition does";
            expect(le(got.size(), want.size() + 1UZ)) << "and at most one more, the trailing zero taps needing no sample";
            expect(gt(want.size(), 0UZ));

            // A float accumulation of seventeen live products of order one: the bound is a few
            // times the last bit of the running total. A tighter number would pin the summation
            // order rather than the arithmetic.
            const double worst = worstDeviation(got, want, want.size());
            expect(lt(worst, 5e-7)) << stageCount << " stage(s): worst deviation " << worst;
        }
    };

    "the halfband zeros are skipped by index, not by value"_test = [] {
        const std::vector<float> designed = gr::filter::fir::design::kaiserLowpass(31, 0.25, 80.0);
        const int                mid      = (static_cast<int>(designed.size()) - 1) / 2;

        std::vector<float> zeroed   = designed; // the skipped taps forced to exactly zero
        std::vector<float> poisoned = designed; // and to a value no correct output may contain
        for (int i = 0; i < static_cast<int>(designed.size()); ++i) {
            const int offset = i - mid;
            if (offset != 0 && (offset % 2) == 0) {
                zeroed[static_cast<std::size_t>(i)]   = 0.0f;
                poisoned[static_cast<std::size_t>(i)] = 1.0f;
            }
        }

        const std::vector<CF> x            = noise(1UZ << 13, 99U);
        const std::vector<CF> fromDesigned = filterWhole({designed}, x);
        const std::vector<CF> fromZeroed   = filterWhole({zeroed}, x);
        const std::vector<CF> fromPoisoned = filterWhole({poisoned}, x);

        expect(gt(fromDesigned.size(), 0UZ));
        expect(that % (fromZeroed == fromPoisoned)) << "a skipped tap is never read";
        expect(that % (fromZeroed == fromDesigned)) << "and the designed near-zeros are not read either";
    };

    "state carries across block boundaries"_test = [&] {
        const std::vector<std::vector<float>> stages = halfbandStages(std::span<const int>(kLengths, 3UZ));
        const std::vector<CF>                 x      = noise(1UZ << 14, 23U);
        const std::vector<CF>                 whole  = filterWhole(stages, x);
        expect(gt(whole.size(), 0UZ));

        HalfbandCascade cascade{std::span<const std::vector<float>>(stages)};
        std::vector<CF> piecewise;
        std::size_t     at = 0UZ;
        while (at < x.size()) {
            // uneven, and mostly straddling the sixteen-output inner block
            const std::size_t take = std::min<std::size_t>(1UZ + ((at * 37UZ) % 997UZ), x.size() - at);
            cascade.push(std::span<const CF>(x.data() + at, take), piecewise);
            at += take;
        }
        expect(that % (piecewise == whole)) << "one call or many, bit for bit";

        cascade.reset();
        std::vector<CF> afterReset;
        cascade.push(std::span<const CF>(x), afterReset);
        expect(that % (afterReset == whole)) << "reset starts a new stream";
    };

    "priming with silence delivers the ratio the decimation implies"_test = [&] {
        for (const int longest : {31, 29}) { // an odd and an even center index
            for (std::size_t stageCount = 1UZ; stageCount <= 3UZ; ++stageCount) {
                std::vector<int> lengths;
                for (std::size_t i = 0UZ; i < stageCount; ++i) {
                    lengths.push_back(longest - 8 * static_cast<int>(i));
                }
                const std::vector<std::vector<float>> stages = halfbandStages(lengths);
                const std::vector<CF>                 x      = noise(1UZ << 13, 41U + static_cast<std::uint32_t>(stageCount));

                HalfbandCascade cascade{std::span<const std::vector<float>>(stages)};
                expect(eq(cascade.stages(), stageCount));
                cascade.primeWithSilence();
                std::vector<CF> got;
                cascade.push(std::span<const CF>(x), got);

                // The same stream through the definition, each stage handed the leading silence the
                // priming gave it: that stage's own designed length less two.
                std::vector<CD> want = asDouble(x);
                for (std::size_t i = 0UZ; i < stageCount; ++i) {
                    std::vector<CD> padded(stages[i].size() - 2UZ, CD{});
                    padded.insert(padded.end(), want.begin(), want.end());
                    want = stageByDefinition(stages[i], padded);
                }

                const std::size_t implied = x.size() >> stageCount;
                expect(eq(got.size(), implied)) << "priming removes the group-delay shortfall and adds nothing";
                expect(eq(got.size(), want.size())) << "and the count is the zero-padded definition's own";

                const double worst = worstDeviation(got, want, want.size());
                expect(lt(worst, 5e-7)) << longest << "/" << stageCount << ": worst deviation " << worst;
            }
        }
    };

    "a stage whose center index is odd is not primed one sample too deep"_test = [&] {
        // 19 and 23 taps keep their first and last taps, both sitting at an odd offset from the
        // center, so they hold one live tap more than 17 or 29 do, and a length taken back out of
        // the live count reads two taps long. A stage primed to that length is a sample deep, and
        // every stage below it inherits the extra output.
        for (const std::vector<int>& lengths : {std::vector<int>{19}, std::vector<int>{23}, std::vector<int>{17}, std::vector<int>{23, 19}}) {
            const std::vector<std::vector<float>> stages = halfbandStages(lengths);
            const std::vector<CF>                 x      = noise(1UZ << 13, 71U + static_cast<std::uint32_t>(lengths.front()));

            HalfbandCascade cascade{std::span<const std::vector<float>>(stages)};
            cascade.primeWithSilence();
            std::vector<CF> got;
            cascade.push(std::span<const CF>(x), got);

            std::vector<CD> want = asDouble(x);
            for (const std::vector<float>& taps : stages) {
                std::vector<CD> padded(taps.size() - 2UZ, CD{});
                padded.insert(padded.end(), want.begin(), want.end());
                want = stageByDefinition(taps, padded);
            }

            expect(eq(got.size(), x.size() >> lengths.size())) << lengths.front() << " taps: exactly input / 2^stages";
            expect(eq(got.size(), want.size())) << lengths.front() << " taps: the zero-padded definition's own count";
            expect(lt(worstDeviation(got, want, want.size()), 5e-7)) << lengths.front() << " taps: and its samples";
        }
    };

    "liveTaps counts only what is multiplied"_test = [&] {
        const std::vector<int>                lengths{15, 23, 29, 31, 33};
        const std::vector<std::vector<float>> stages = halfbandStages(lengths);
        HalfbandCascade                       cascade{std::span<const std::vector<float>>(stages)};

        expect(eq(cascade.stages(), lengths.size()));
        for (std::size_t s = 0UZ; s < lengths.size(); ++s) {
            const int   n         = lengths[s];
            const int   mid       = (n - 1) / 2;
            std::size_t oddOffset = 0UZ;
            for (int i = 0; i < n; ++i) {
                if (((i - mid) % 2) != 0) {
                    ++oddOffset;
                }
            }
            expect(eq(cascade.liveTaps(s), oddOffset + 1UZ)) << "the odd-offset taps plus the center";
            expect(lt(cascade.liveTaps(s), static_cast<std::size_t>(n))) << "fewer than designed";
            expect(gt(2UZ * cascade.liveTaps(s), static_cast<std::size_t>(n))) << "a little over half";
            // The relation runs one way only: at an odd center index the live count takes the
            // length back two taps long, which is why priming keeps the designed length instead.
            expect(eq(2UZ * cascade.liveTaps(s) - 1UZ, static_cast<std::size_t>(n) + (((mid % 2) != 0) ? 2UZ : 0UZ))) << n << " taps: the length taken back out of the live count";
        }
    };

    "a cascade with no stages passes its input through"_test = [] {
        const std::vector<std::vector<float>> none;
        HalfbandCascade                       cascade{std::span<const std::vector<float>>(none)};
        const std::vector<CF>                 x = noise(64UZ, 5U);

        std::vector<CF> out;
        expect(eq(cascade.push(std::span<const CF>(x), out), x.size()));
        expect(that % (out == x));
        expect(eq(cascade.stages(), 0UZ));
    };
};

int main() { /* tests are automatically registered and run */ }
