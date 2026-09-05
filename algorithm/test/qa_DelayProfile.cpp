#include <boost/ut.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/algorithm/channel/DelayProfile.hpp>
#include <gnuradio-4.0/algorithm/channel/NoisePower.hpp>

/*
 * Both functions are value functions with no state, so what has to be pinned is the arithmetic: which sample each
 * published delay rounds onto, that two paths landing on one sample add in power rather than in amplitude, that the
 * normalization is one factor over the whole tap set, and that an Es/N0 turns into the power the channel model takes.
 * The refusals are pinned beside them, since a malformed profile is what a graph hands this function.
 */

namespace {

using gr::channel::noisePowerFor;
using gr::channel::tapsFromProfile;

/// @brief The total power a tap set carries, which is what `normalize` fixes at one.
[[nodiscard]] double totalPower(const std::vector<std::complex<float>>& taps) {
    double total = 0.0;
    for (const std::complex<float>& tap : taps) {
        total += static_cast<double>(std::norm(tap));
    }
    return total;
}

} // namespace

const boost::ut::suite<"delay profile"> delayProfileTests = [] {
    using namespace boost::ut;

    "a profile that cannot be read is refused"_test = [] {
        const std::vector<double> delays{0.0, 1.0e-6};
        const std::vector<double> powers{0.0, -3.0};
        const std::vector<double> backwards{0.0, -1.0e-6};

        expect(throws<std::invalid_argument>([&] { std::ignore = tapsFromProfile(std::span<const double>(delays), std::span<const double>(powers).first(1UZ), 1.0e6); })) << "the delay and power tables must be the same length";
        expect(throws<std::invalid_argument>([] { std::ignore = tapsFromProfile(std::span<const double>{}, std::span<const double>{}, 1.0e6); })) << "an empty profile names no path";
        expect(throws<std::invalid_argument>([&] { std::ignore = tapsFromProfile(std::span<const double>(delays), std::span<const double>(powers), 0.0); })) << "a sample rate of zero has no sample to round a delay onto";
        expect(throws<std::invalid_argument>([&] { std::ignore = tapsFromProfile(std::span<const double>(delays), std::span<const double>(powers), -1.0e6); })) << "nor a negative one";
        expect(throws<std::invalid_argument>([&] { std::ignore = tapsFromProfile(std::span<const double>(backwards), std::span<const double>(powers), 1.0e6); })) << "no path arrives before the first";

        expect(nothrow([&] { std::ignore = tapsFromProfile(std::span<const double>(delays), std::span<const double>(powers), 1.0e6); })) << "and the profile those four are variations of is accepted";
    };

    "each delay rounds onto a sample, and the tap set spans the longest"_test = [] {
        // At 1 MS/s a delay of 2.4 us rounds onto sample 2 and 2.6 us onto sample 3, so the set is four taps
        // long and the sample no path reaches is a zero.
        const std::vector<double> delays{0.0, 2.4e-6, 2.6e-6};
        const std::vector<double> powers{0.0, -3.0, -6.0};
        const auto                taps = tapsFromProfile(std::span<const double>(delays), std::span<const double>(powers), 1.0e6, false);

        expect(eq(taps.size(), 4UZ)) << "the set spans the first sample to the longest rounded delay";
        expect(eq(std::abs(taps[1UZ]), 0.f)) << "nothing rounds onto sample one";

        expect(approx(static_cast<double>(std::norm(taps[0UZ])), 1.0, 1e-6)) << "0 dB is unit power";
        expect(approx(static_cast<double>(std::norm(taps[2UZ])), std::pow(10.0, -0.3), 1e-6)) << "a level in dB is a power, not an amplitude";
        expect(approx(static_cast<double>(std::norm(taps[3UZ])), std::pow(10.0, -0.6), 1e-6));
    };

    "two paths that round onto one sample add in power"_test = [] {
        // Both of these round onto sample one at 1 MS/s. Summing the amplitudes instead would put the tap 1.5 dB
        // high for two equal paths, which is the error every published tabulation is read with.
        const std::vector<double> together{1.0e-6, 1.2e-6};
        const std::vector<double> equal{-3.0, -3.0};
        const auto                summed = tapsFromProfile(std::span<const double>(together), std::span<const double>(equal), 1.0e6, false);

        expect(eq(summed.size(), 2UZ));
        expect(approx(static_cast<double>(std::norm(summed[1UZ])), 2.0 * std::pow(10.0, -0.3), 1e-6)) << "the two powers add";
        expect(lt(static_cast<double>(std::abs(summed[1UZ])), 2.0 * std::sqrt(std::pow(10.0, -0.3)))) << "and the two amplitudes do not";
    };

    "normalization is one factor over the whole set, and is a request"_test = [] {
        const std::vector<double> delays{0.0, 1.0e-6, 3.0e-6};
        const std::vector<double> powers{0.0, -3.0, -9.0};

        const auto normalized = tapsFromProfile(std::span<const double>(delays), std::span<const double>(powers), 1.0e6);
        const auto raw        = tapsFromProfile(std::span<const double>(delays), std::span<const double>(powers), 1.0e6, false);

        expect(eq(normalized.size(), raw.size()));
        expect(approx(totalPower(normalized), 1.0, 1e-6)) << "a normalized profile leaves the level a receiver sees where it was";
        expect(gt(totalPower(raw), 1.0)) << "and an unnormalized one carries the profile's own total";

        const double scale = static_cast<double>(normalized[0UZ].real()) / static_cast<double>(raw[0UZ].real());
        for (std::size_t k = 0UZ; k < raw.size(); ++k) {
            expect(approx(static_cast<double>(normalized[k].real()), scale * static_cast<double>(raw[k].real()), 1e-6)) << "tap " << k << " is the same shape scaled";
        }
    };

    "noisePowerFor is the definition, in the units the literature states"_test = [] {
        expect(approx(noisePowerFor(0.0), 1.0, 1e-12)) << "0 dB is equal symbol energy and noise power";
        expect(approx(noisePowerFor(10.0), 0.1, 1e-12));
        expect(approx(noisePowerFor(20.0), 0.01, 1e-12));
        expect(approx(noisePowerFor(-3.0), std::pow(10.0, 0.3), 1e-12)) << "a negative Es/N0 is more noise than signal";

        expect(approx(noisePowerFor(10.0, 2.0), 0.2, 1e-12)) << "the mean symbol energy scales it";
        expect(approx(noisePowerFor(10.0, 1.0, 4.0), 0.4, 1e-12)) << "and so does the oversampling the noise is added at";
        expect(lt(noisePowerFor(30.0), noisePowerFor(20.0))) << "a higher operating point is less noise at the same rate";

        expect(approx(static_cast<double>(noisePowerFor(10.0f)), 0.1, 1e-6)) << "float reaches the same value";
    };
};

int main() { /* tests are automatically registered and run */ }
