#ifndef GNURADIO_ALGORITHM_CHANNEL_DELAY_PROFILE_HPP
#define GNURADIO_ALGORITHM_CHANNEL_DELAY_PROFILE_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <span>
#include <stdexcept>
#include <vector>

namespace gr::channel {

/**
 * @brief Complex FIR taps for a published delay/power profile at a sample rate.
 *
 * A profile states delays in seconds and average powers in dB. Rounding each delay to the nearest sample
 * gives the tap positions; powers convert to amplitudes and, when `normalize` is set, are scaled so the
 * profile carries unit mean power — which keeps a channel from quietly changing the level a receiver sees.
 * Paths that round onto the same sample are summed in power, as they are in every published tabulation.
 *
 * Static multipath is this function plus an FIR filter. A time-varying channel needs tap gains that are
 * processes rather than constants, which is what a fading-channel block covers.
 */
[[nodiscard]] inline std::vector<std::complex<float>> tapsFromProfile(std::span<const double> delaysSeconds, std::span<const double> powersDb, double sampleRate, bool normalize = true) {
    if (delaysSeconds.size() != powersDb.size()) {
        throw std::invalid_argument("tapsFromProfile: the delay and power tables must be the same length");
    }
    if (delaysSeconds.empty()) {
        throw std::invalid_argument("tapsFromProfile: the profile is empty");
    }
    if (!(sampleRate > 0.)) {
        throw std::invalid_argument("tapsFromProfile: the sample rate must be positive");
    }

    std::size_t span = 0UZ;
    for (const double delay : delaysSeconds) {
        if (delay < 0.) {
            throw std::invalid_argument("tapsFromProfile: delays must not be negative");
        }
        span = std::max(span, static_cast<std::size_t>(std::llround(delay * sampleRate)));
    }

    std::vector<double> power(span + 1UZ, 0.);
    for (std::size_t p = 0UZ; p < delaysSeconds.size(); ++p) {
        power[static_cast<std::size_t>(std::llround(delaysSeconds[p] * sampleRate))] += std::pow(10., powersDb[p] / 10.);
    }

    double total = 0.;
    for (const double value : power) {
        total += value;
    }
    const double scale = (normalize && total > 0.) ? 1. / total : 1.;

    std::vector<std::complex<float>> taps(power.size());
    for (std::size_t k = 0UZ; k < power.size(); ++k) {
        taps[k] = std::complex<float>(static_cast<float>(std::sqrt(power[k] * scale)), 0.f);
    }
    return taps;
}

} // namespace gr::channel

#endif // GNURADIO_ALGORITHM_CHANNEL_DELAY_PROFILE_HPP
