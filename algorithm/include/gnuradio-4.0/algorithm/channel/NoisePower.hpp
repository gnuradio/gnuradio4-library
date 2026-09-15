#ifndef GNURADIO_ALGORITHM_CHANNEL_NOISE_POWER_HPP
#define GNURADIO_ALGORITHM_CHANNEL_NOISE_POWER_HPP

#include <cmath>
#include <concepts>

namespace gr::channel {

/**
 * @brief Noise power for a wanted Es/N0, so a graph can state its operating point in the units the
 * literature uses while the channel model itself stays parameterized by power.
 *
 * `esN0_db` is per symbol, `symbolEnergy` the mean energy of one transmitted symbol (1 for a unit-power
 * constellation), and `samplesPerSymbol` the oversampling the noise is added at — noise added at the sample
 * rate spreads across `samplesPerSymbol` samples per symbol, so the per-sample power rises with it.
 */
template<std::floating_point F>
[[nodiscard]] F noisePowerFor(F esN0_db, F symbolEnergy = F(1), F samplesPerSymbol = F(1)) noexcept {
    return symbolEnergy * samplesPerSymbol / std::pow(F(10), esN0_db / F(10));
}

} // namespace gr::channel

#endif // GNURADIO_ALGORITHM_CHANNEL_NOISE_POWER_HPP
