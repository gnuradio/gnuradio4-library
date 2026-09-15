#ifndef GNURADIO_ALGORITHM_MEASUREMENT_SLOT_HPP
#define GNURADIO_ALGORITHM_MEASUREMENT_SLOT_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace gr::measurement {

/**
 * @brief A seqlock over a fixed set of doubles and a fill count, the way a measurement sink publishes.
 *
 * The writer never blocks and the reader retries while the counter is odd or has moved between its two reads, so a
 * poll from another thread beside a settings change sees a whole window or tries again, never half of two.
 *
 * The fill count says how much of the nominal window the values cover, so that a partial window can be reported as
 * partial rather than suppressed. What the values mean is the publisher's business; this type only carries them
 * across the thread boundary intact.
 */
template<std::size_t kValues>
struct MeasurementSlot {
    std::atomic<std::uint32_t>               sequence{0U};
    std::array<std::atomic<double>, kValues> values{};
    std::atomic<std::uint64_t>               filled{0ULL};

    void publish(const std::array<double, kValues>& next, std::uint64_t count) noexcept {
        const std::uint32_t at = sequence.load(std::memory_order_relaxed);
        sequence.store(at + 1U, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        for (std::size_t i = 0UZ; i < kValues; ++i) {
            values[i].store(next[i], std::memory_order_relaxed);
        }
        filled.store(count, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        sequence.store(at + 2U, std::memory_order_relaxed);
    }

    [[nodiscard]] std::pair<std::array<double, kValues>, std::uint64_t> read() const noexcept {
        for (;;) {
            const std::uint32_t before = sequence.load(std::memory_order_relaxed);
            if ((before & 1U) != 0U) {
                continue;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            std::array<double, kValues> out{};
            for (std::size_t i = 0UZ; i < kValues; ++i) {
                out[i] = values[i].load(std::memory_order_relaxed);
            }
            const std::uint64_t count = filled.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (sequence.load(std::memory_order_relaxed) == before) {
                return {out, count};
            }
        }
    }
};

} // namespace gr::measurement

#endif // GNURADIO_ALGORITHM_MEASUREMENT_SLOT_HPP
