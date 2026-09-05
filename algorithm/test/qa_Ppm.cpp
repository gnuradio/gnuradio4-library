#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <numbers>
#include <span>
#include <string_view>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/algorithm/digital/Ppm.hpp>
#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>
#include <gnuradio-4.0/algorithm/sync/MmseInterpolator.hpp>

namespace {

using gr::digital::PpmConfig;
using gr::digital::PpmCounters;
using gr::digital::PpmFrame;
using gr::digital::PpmScanner;

/// One hexadecimal frame as octets, first bit received in the first octet's most significant bit.
[[nodiscard]] std::vector<std::uint8_t> octetsOf(std::string_view hex) {
    std::vector<std::uint8_t> octets;
    octets.reserve(hex.size() / 2UZ);
    const auto nibble = [](char c) { return static_cast<std::uint8_t>(c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10); };
    for (std::size_t i = 0UZ; i + 1UZ < hex.size(); i += 2UZ) {
        octets.push_back(static_cast<std::uint8_t>((nibble(hex[i]) << 4U) | nibble(hex[i + 1UZ])));
    }
    return octets;
}

/// Grid points a slot is rendered on, before the front end's filter and the sampler read it.
constexpr std::size_t kGrid = 32UZ;
/// Slots of silence the renderer puts before and after the burst, so a scan has positions on either side of it.
constexpr std::size_t kPad = 8UZ;
/// The carrier phase the rendered pulses sit on; nothing downstream reads it, and a non-zero one keeps that honest.
constexpr float kCarrier = 0.7F;

/// @brief One complex Gaussian sample, @p sigma the standard deviation of each component, by the Box-Muller transform.
[[nodiscard]] std::complex<float> gaussian(gr::rng::Xoshiro256pp& source, double sigma) {
    const double uniform = std::max(source.uniform01<double>(), 1.0e-12); // the transform's logarithm has no value at zero
    const double angle   = source.uniform01<double>();
    const double radius  = sigma * std::sqrt(-2.0 * std::log(uniform));
    return {static_cast<float>(radius * std::cos(2.0 * std::numbers::pi * angle)), static_cast<float>(radius * std::sin(2.0 * std::numbers::pi * angle))};
}

/// @brief One rendered burst: the complex baseband, the magnitudes taken from it, and where the preamble starts.
struct Rendered {
    std::vector<std::complex<float>> samples{};
    std::vector<float>               magnitudes{};
    std::size_t                      position = 0UZ; ///< sample index of the preamble's first slot, the position a scan should nominate
};

/**
 * @brief A Mode S reply as complex baseband at @p samplesPerSlot samples a slot, sampled @p tau samples late.
 *
 * Pulses are one slot wide, unit amplitude and rectangular on a `kGrid` times finer grid; a moving average one slot
 * wide stands for a front end about as wide as the slot rate; and the samples are taken at the centers of the
 * scanner's own sample bins, displaced by @p tau samples. The filtered grid is piecewise linear between its points,
 * every breakpoint falling on one, so reading it linearly is exact rather than an approximation under test.
 *
 * @p sigma is the standard deviation of each component of the additive Gaussian noise, zero for a clean burst;
 * @p seed makes that noise a fixed stream. A scanner deciding this burst should report `phase` near `-tau`.
 */
[[nodiscard]] Rendered renderModeS(std::span<const std::uint8_t> octets, std::size_t samplesPerSlot, double tau, double sigma = 0.0, std::uint64_t seed = 1ULL) {
    const std::size_t         bits  = 8UZ * octets.size();
    const std::size_t         slots = 2UZ * kPad + 16UZ + 2UZ * bits;
    std::vector<std::uint8_t> pulses(slots, std::uint8_t{0});
    for (const std::size_t slot : {0UZ, 2UZ, 7UZ, 9UZ}) { // the preamble's leading edges at 0.0, 1.0, 3.5 and 4.5 microseconds
        pulses[kPad + slot] = 1U;
    }
    for (std::size_t bit = 0UZ; bit < bits; ++bit) { // a one puts its pulse in the first half of the bit period, a zero in the second
        const bool one                                      = (octets[bit / 8UZ] & (0x80U >> (bit % 8UZ))) != 0U;
        pulses[kPad + 16UZ + 2UZ * bit + (one ? 0UZ : 1UZ)] = 1U;
    }

    // one slot of headroom at either end of the grid, so the filter and an offset below zero stay inside it
    std::vector<float> fine((slots + 2UZ) * kGrid, 0.F);
    for (std::size_t slot = 0UZ; slot < slots; ++slot) {
        if (pulses[slot] != 0U) {
            std::fill_n(fine.begin() + static_cast<std::ptrdiff_t>((slot + 1UZ) * kGrid), kGrid, 1.F);
        }
    }
    std::vector<float> grid(fine.size(), 0.F);
    for (std::size_t point = kGrid / 2UZ; point + kGrid / 2UZ < fine.size(); ++point) {
        float total = 0.F;
        for (std::size_t k = 0UZ; k < kGrid; ++k) {
            total += fine[point - kGrid / 2UZ + k];
        }
        grid[point] = total / static_cast<float>(kGrid); // centered, so a pulse keeps the slot it was rendered on
    }

    Rendered          out;
    const std::size_t count = slots * samplesPerSlot;
    out.samples.resize(count);
    out.magnitudes.resize(count);
    out.position = kPad * samplesPerSlot;

    const std::complex<float> carrier = std::polar(1.F, kCarrier);
    gr::rng::Xoshiro256pp     source(seed);
    for (std::size_t m = 0UZ; m < count; ++m) {
        const double at    = static_cast<double>(kGrid) * ((static_cast<double>(m) + 0.5 + tau) / static_cast<double>(samplesPerSlot) + 1.0);
        const double whole = std::floor(at);
        const auto   point = static_cast<std::size_t>(whole);
        const double frac  = at - whole;
        const auto   value = static_cast<float>((1.0 - frac) * static_cast<double>(grid[point]) + frac * static_cast<double>(grid[point + 1UZ]));
        out.samples[m]     = value * carrier;
        if (sigma > 0.0) {
            out.samples[m] += gaussian(source, sigma);
        }
        out.magnitudes[m] = std::abs(out.samples[m]);
    }
    return out;
}

/// @brief @p count samples of complex Gaussian noise and no burst at all.
[[nodiscard]] Rendered noiseOnly(std::size_t count, double sigma, std::uint64_t seed) {
    Rendered              out;
    gr::rng::Xoshiro256pp source(seed);
    out.samples.resize(count);
    out.magnitudes.resize(count);
    for (std::size_t m = 0UZ; m < count; ++m) {
        out.samples[m]    = gaussian(source, sigma);
        out.magnitudes[m] = std::abs(out.samples[m]);
    }
    return out;
}

/// @brief What one scan reported, kept past the scanner's own buffers.
struct Scan {
    std::vector<std::vector<std::uint8_t>> octets{};    ///< every nomination's frame
    std::vector<std::size_t>               positions{}; ///< where each was nominated
    std::vector<float>                     phases{};    ///< the offset each was decided at
    std::vector<float>                     ratios{};    ///< the least pulse slot over the greatest gap, at that offset
    std::vector<bool>                      admitted{};  ///< whether the parity vouched for each
    PpmCounters                            counters{};
    std::size_t                            consumed = 0UZ;

    /// @brief The index of the one admitted nomination, or the size when there is none.
    [[nodiscard]] std::size_t hit() const { return static_cast<std::size_t>(std::ranges::find(admitted, true) - admitted.begin()); }

    /// @brief Whether exactly one nomination was admitted, its octets are @p frame, and it sits within a sample of @p position.
    [[nodiscard]] bool admits(std::span<const std::uint8_t> frame, std::size_t position) const {
        std::size_t hits = 0UZ;
        for (std::size_t k = 0UZ; k < octets.size(); ++k) {
            if (!admitted[k]) {
                continue;
            }
            // an offset of exactly half a sample is the same instant read from the next position, so the position may slip by one
            const std::size_t distance = positions[k] > position ? positions[k] - position : position - positions[k];
            if (!std::ranges::equal(octets[k], frame) || distance > 1UZ) {
                return false;
            }
            ++hits;
        }
        return hits == 1UZ;
    }

    /// @brief The shape ratio of the admitted nomination, or the greatest of every nomination when none was admitted.
    [[nodiscard]] float bestRatio() const {
        const std::size_t which = hit();
        if (which < ratios.size()) {
            return ratios[which];
        }
        return ratios.empty() ? 0.F : *std::ranges::max_element(ratios);
    }

    [[nodiscard]] float phaseOf() const { return phases[hit()]; }
};

/// @brief One pass of @p scanner over @p burst, through the phase-aligned call when @p aligned and the plain one otherwise.
[[nodiscard]] Scan scanOnce(PpmScanner& scanner, const Rendered& burst, bool aligned) {
    Scan       seen;
    const auto sink = [&seen](const PpmFrame& frame) {
        seen.octets.emplace_back(frame.octets.begin(), frame.octets.end());
        seen.positions.push_back(frame.position);
        seen.phases.push_back(frame.phase);
        seen.ratios.push_back(frame.weak > 0.F ? frame.strong / frame.weak : std::numeric_limits<float>::max());
        seen.admitted.push_back(frame.admitted());
    };
    seen.consumed = aligned ? scanner.consume(burst.magnitudes, burst.samples, sink) : scanner.consume(burst.magnitudes, sink);
    seen.counters = scanner.counters;
    return seen;
}

/// @brief A scanner carrying `modeS()` at @p samplesPerSlot samples a slot, searching @p phaseSteps offsets.
[[nodiscard]] PpmScanner scannerFor(std::size_t samplesPerSlot, std::size_t phaseSteps) {
    PpmScanner scanner;
    scanner.phaseSteps = phaseSteps;
    scanner.prepare(gr::digital::modeS(), samplesPerSlot);
    return scanner;
}

/// A scanner carrying `modeS()` at one sample a slot, which is all `remainderOf` needs.
[[nodiscard]] PpmScanner modeSScanner() { return scannerFor(1UZ, 1UZ); }

/// The eight sub-sample offsets the tests render at, `-0.5 + k / 8` samples.
[[nodiscard]] constexpr double offsetOf(std::size_t k) noexcept { return -0.5 + static_cast<double>(k) / 8.0; }

/// @brief The greatest preamble shape ratio the integer test finds at @p burst's own position or a neighbor of it.
[[nodiscard]] float integerRatio(const Rendered& burst, std::size_t samplesPerSlot) {
    const PpmConfig cfg  = gr::digital::modeS();
    float           best = 0.F;
    for (std::ptrdiff_t slip = -1; slip <= 1; ++slip) {
        const auto position = static_cast<std::size_t>(static_cast<std::ptrdiff_t>(burst.position) + slip);
        float      pulse    = std::numeric_limits<float>::max();
        float      gap      = 0.F;
        for (std::size_t slot = 0UZ; slot < cfg.preambleSlots; ++slot) {
            float total = 0.F;
            for (std::size_t j = 0UZ; j < samplesPerSlot; ++j) {
                total += burst.magnitudes[position + slot * samplesPerSlot + j];
            }
            if (std::ranges::find(cfg.pulseSlots, slot) != cfg.pulseSlots.end()) {
                pulse = std::min(pulse, total);
            } else {
                gap = std::max(gap, total);
            }
        }
        best = std::max(best, gap > 0.F ? pulse / gap : std::numeric_limits<float>::max());
    }
    return best;
}

/**
 * @brief The greatest preamble shape ratio a search over the eight offsets finds at @p burst's own position.
 *
 * It reads the bank directly rather than through the scanner, which holds its band as a constant, so that one band
 * can be measured against another on the same burst.
 */
[[nodiscard]] float searchedRatio(const Rendered& burst, std::size_t samplesPerSlot, double band) {
    const gr::sync::MmseInterpolatorBank bank(PpmScanner::kInterpolatorTaps, 8, band, false);
    const PpmConfig                      cfg = gr::digital::modeS();

    float best = 0.F;
    for (std::ptrdiff_t slip = -1; slip <= 1; ++slip) { // the same instants are reachable from either neighboring position
        const auto position = static_cast<std::size_t>(static_cast<std::ptrdiff_t>(burst.position) + slip);
        for (std::size_t k = 0UZ; k < 8UZ; ++k) {
            const double      tau     = offsetOf(k);
            const std::size_t row     = bank.row(tau < 0.0 ? tau + 1.0 : tau);
            const std::size_t lead    = tau < 0.0 ? 4UZ : 3UZ;
            const auto        slotSum = [&](std::size_t slot) {
                float total = 0.F;
                for (std::size_t j = 0UZ; j < samplesPerSlot; ++j) {
                    total += std::abs(bank.interpolate<std::complex<float>>(burst.samples.data() + position + slot * samplesPerSlot + j - lead, row));
                }
                return total;
            };
            float pulse = std::numeric_limits<float>::max();
            float gap   = 0.F;
            for (std::size_t slot = 0UZ; slot < cfg.preambleSlots; ++slot) {
                const float value = slotSum(slot);
                if (std::ranges::find(cfg.pulseSlots, slot) != cfg.pulseSlots.end()) {
                    pulse = std::min(pulse, value);
                } else {
                    gap = std::max(gap, value);
                }
            }
            best = std::max(best, gap > 0.F ? pulse / gap : std::numeric_limits<float>::max());
        }
    }
    return best;
}

/// The five published extended squitters, whose parity fields are the bare CRC.
constexpr std::array<std::string_view, 5> kPublished{"8D4840D6202CC371C32CE0576098", "8D40621D58C382D690C8AC2863A7", "8D40621D58C386435CC412692AD6", "8D485020994409940838175B284F", "8D40621D58C3862590C412D77427"};

} // namespace

const boost::ut::suite<"ppm"> ppmTests = [] {
    using namespace boost::ut;

    "modeS states Annex 10's geometry, length rule and parity generator"_test = [] {
        const PpmConfig cfg = gr::digital::modeS();
        expect(that % (cfg.pulseSlots == std::vector<std::size_t>{0UZ, 2UZ, 7UZ, 9UZ})) << "pulses at 0.0, 1.0, 3.5 and 4.5 microseconds, in half-microsecond slots";
        expect(eq(cfg.preambleSlots, 16UZ)) << "eight microseconds of preamble";
        expect(eq(cfg.formatBits, 5UZ));
        expect(eq(cfg.longBits, 112UZ));
        expect(eq(cfg.shortBits, 56UZ));
        expect(eq(static_cast<unsigned>(cfg.crcWidth), 24U));
        expect(eq(cfg.crcPoly, 0xFFF409ULL));
        expect(eq(cfg.crcInit, 0ULL));
        expect(eq(cfg.crcFinalXor, 0ULL));
        expect(!cfg.crcInputReflected);
        expect(!cfg.crcResultReflected);
    };

    "the published frames reduce to zero, and one flipped bit does not"_test = [] {
        const PpmScanner scanner = modeSScanner();
        for (const std::string_view hex : kPublished) {
            const std::vector<std::uint8_t> frame = octetsOf(hex);
            expect(eq(frame.size(), 14UZ)) << hex;
            expect(eq(scanner.remainderOf(std::span<const std::uint8_t>(frame)), 0ULL)) << hex << " is a bare-CRC frame";
        }

        std::vector<std::uint8_t> damaged = octetsOf(kPublished[0UZ]);
        damaged[0UZ] ^= 0x01U;
        expect(neq(scanner.remainderOf(std::span<const std::uint8_t>(damaged)), 0ULL)) << "a single flipped bit leaves the remainder";
    };

    "a parity field XORed with an address leaves that address as the remainder"_test = [] {
        const PpmScanner          scanner = modeSScanner();
        std::vector<std::uint8_t> message{0x20U, 0x00U, 0x11U, 0x22U}; // downlink format 4, a surveillance altitude reply
        message.insert(message.end(), 3UZ, std::uint8_t{0});
        const std::uint64_t parity = scanner.remainderOf(std::span<const std::uint8_t>(message));

        for (const std::uint64_t address : {0x4840D6ULL, 0xABCDEFULL}) {
            const std::uint64_t       field = parity ^ address;
            std::vector<std::uint8_t> frame(message.begin(), message.begin() + 4);
            frame.push_back(static_cast<std::uint8_t>(field >> 16U));
            frame.push_back(static_cast<std::uint8_t>(field >> 8U));
            frame.push_back(static_cast<std::uint8_t>(field));
            expect(eq(scanner.remainderOf(std::span<const std::uint8_t>(frame)), address)) << "the reduction is linear, so the XORed address survives it";
        }
    };

    "validatePpmConfig refuses every geometry the scanner cannot read"_test = [] {
        const auto refused = [](auto&& edit) {
            PpmConfig cfg = gr::digital::modeS();
            edit(cfg);
            return throws([&cfg] { gr::digital::validatePpmConfig(cfg); });
        };

        expect(nothrow([] { gr::digital::validatePpmConfig(gr::digital::modeS()); }));
        expect(refused([](PpmConfig& c) { c.pulseSlots.clear(); })) << "no pulse to measure";
        expect(refused([](PpmConfig& c) { c.pulseSlots.push_back(16UZ); })) << "a pulse slot outside the preamble";
        expect(refused([](PpmConfig& c) { c.pulseSlots.push_back(2UZ); })) << "a pulse slot named twice";
        expect(refused([](PpmConfig& c) { c.preambleSlots = 4UZ; })) << "every slot a pulse leaves no gap";
        expect(refused([](PpmConfig& c) { c.preambleSlots = 0UZ; }));
        expect(refused([](PpmConfig& c) { c.formatBits = 0UZ; }));
        expect(refused([](PpmConfig& c) { c.formatBits = 33UZ; })) << "a format field wider than the word that reads it, whose top bit could not be shifted out";
        expect(nothrow([] {
            PpmConfig cfg  = gr::digital::modeS();
            cfg.formatBits = 32UZ;
            gr::digital::validatePpmConfig(cfg);
        })) << "the widest field the word still reads";
        expect(refused([](PpmConfig& c) { c.shortBits = 120UZ; })) << "the short frame is longer than the long one";
        expect(refused([](PpmConfig& c) { c.shortBits = 52UZ; })) << "a frame length that is not whole octets";
        expect(refused([](PpmConfig& c) { c.longBits = 116UZ; }));
        expect(refused([](PpmConfig& c) { c.crcWidth = 2U; })) << "below the register width the CRC kernel accepts";
        expect(refused([](PpmConfig& c) { c.crcWidth = 20U; })) << "a parity field that is not whole octets";
        expect(refused([](PpmConfig& c) {
            c.shortBits = 24UZ;
            c.longBits  = 24UZ;
        })) << "a parity field that leaves no message";

        PpmScanner scanner;
        expect(throws([&scanner] { scanner.prepare(gr::digital::modeS(), 0UZ); })) << "a slot spans at least one sample";
        expect(!scanner.configured()) << "a refused preparation leaves the scanner inert";
    };

    "at one sample a slot the search reproduces the integer decision, and raises its ratio"_test = [] {
        const std::vector<std::uint8_t> frame = octetsOf(kPublished[0UZ]);

        std::vector<std::size_t> integerAdmitted;
        std::vector<std::size_t> searchedAdmitted;
        for (std::size_t k = 0UZ; k < 8UZ; ++k) {
            const std::string at    = std::format("offset {:+.3f} samples", offsetOf(k));
            const Rendered    burst = renderModeS(std::span<const std::uint8_t>(frame), 1UZ, offsetOf(k));

            PpmScanner integer = scannerFor(1UZ, 1UZ);
            if (scanOnce(integer, burst, false).admits(std::span<const std::uint8_t>(frame), burst.position)) {
                integerAdmitted.push_back(k);
            }

            PpmScanner aligned  = scannerFor(1UZ, 8UZ);
            const Scan searched = scanOnce(aligned, burst, true);
            if (searched.admits(std::span<const std::uint8_t>(frame), burst.position)) {
                searchedAdmitted.push_back(k);
            }
            expect(ge(searched.counters.candidates, searched.counters.nominations)) << at;
        }

        // measured: a Mode S preamble at one sample a slot holds its shape over the middle five of the eight
        // offsets and no more, half a pulse landing in the neighboring gap slot costing the ratio at either end
        expect(that % (integerAdmitted == std::vector<std::size_t>{2UZ, 3UZ, 4UZ, 5UZ, 6UZ})) << "the offsets an integer decision admits";
        // measured: and the search admits exactly those, a pulse one sample wide leaving nothing to interpolate;
        // it never admits fewer, a zero offset being in the set and reproducing the integer decision exactly
        expect(that % (searchedAdmitted == integerAdmitted)) << "the search admits neither fewer nor, at this rate, more";
    };

    "what the search buys at one sample a slot is ratio, measured where the integer decision has none"_test = [] {
        const std::vector<std::uint8_t> frame = octetsOf(kPublished[0UZ]);
        // the worst offset there is: the pulses split evenly over two samples, so every pulse slot reads what the
        // gap slot beside it reads and the integer ratio is exactly one
        const Rendered burst = renderModeS(std::span<const std::uint8_t>(frame), 1UZ, -0.5);

        PpmScanner integer     = scannerFor(1UZ, 1UZ);
        integer.threshold      = 1.3F;
        const Scan integerSeen = scanOnce(integer, burst, false);
        expect(!integerSeen.admits(std::span<const std::uint8_t>(frame), burst.position)) << "an integer decision cannot reach 1.3 on a ratio of one";

        PpmScanner aligned      = scannerFor(1UZ, 8UZ);
        aligned.threshold       = 1.3F;
        const Scan searchedSeen = scanOnce(aligned, burst, true);
        expect(searchedSeen.admits(std::span<const std::uint8_t>(frame), burst.position)) << "the search reaches it, and reads the frame's octets back";
        // measured: 1.353, against 1.000 at integer positions and 2.0 at the default threshold, which is why one
        // sample a slot is where this stops paying
        expect(gt(searchedSeen.bestRatio(), 1.3F)) << std::format("the ratio the search found was {:.3f}", searchedSeen.bestRatio());
        expect(lt(searchedSeen.bestRatio(), 2.F)) << "and short of what the default threshold asks, at this rate";
    };

    "three samples a slot are decided at every offset, and the search raises the ratio at each"_test = [] {
        const std::vector<std::uint8_t> frame = octetsOf(kPublished[1UZ]);
        for (std::size_t k = 0UZ; k < 8UZ; ++k) {
            const std::string at    = std::format("offset {:+.3f} samples", offsetOf(k));
            const Rendered    burst = renderModeS(std::span<const std::uint8_t>(frame), 3UZ, offsetOf(k));

            PpmScanner integer = scannerFor(3UZ, 1UZ);
            expect(scanOnce(integer, burst, false).admits(std::span<const std::uint8_t>(frame), burst.position)) << at << ": an integer decision already holds at three samples a slot";

            PpmScanner aligned  = scannerFor(3UZ, 8UZ);
            const Scan searched = scanOnce(aligned, burst, true);
            expect(searched.admits(std::span<const std::uint8_t>(frame), burst.position)) << at << ": and the search reads the same octets back";
            expect(lt(std::abs(static_cast<double>(searched.phaseOf())), 0.5 + 1.0e-6)) << at << ": the offset reported spans one sample and no more";

            // at one and the same position the search can only raise the ratio, a zero offset being in the set and
            // an interpolation at zero being the sample itself
            const float plain = integerRatio(burst, 3UZ);
            const float found = searchedRatio(burst, 3UZ, PpmScanner::kInterpolatorBand);
            expect(ge(found, plain)) << at << std::format(": searched {:.3f} against integer {:.3f} at the burst's own position", found, plain);
        }

        // measured: the worst of the eight is the burst rendered half a sample early, where the search lifts the
        // ratio at its own position from 2.600 to 3.470
        const Rendered worst = renderModeS(std::span<const std::uint8_t>(frame), 3UZ, -0.5);
        expect(gt(searchedRatio(worst, 3UZ, PpmScanner::kInterpolatorBand), integerRatio(worst, 3UZ))) << std::format("half a sample early: searched {:.3f} against integer {:.3f}", searchedRatio(worst, 3UZ, PpmScanner::kInterpolatorBand), integerRatio(worst, 3UZ));
    };

    "the ratio a scan reports is the first position that cleared the threshold, not the best one"_test = [] {
        // measured: at three samples a slot the search lifts positions a sample before the burst over the threshold
        // too, and containment gives the window to whichever cleared it first, so the position and the ratio the
        // scan reports can both be worse than what an integer decision reports on the same burst. The frame is the
        // same either way, which is what the parity vouches for; the sub-sample offset is not a best timing estimate.
        const std::vector<std::uint8_t> frame = octetsOf(kPublished[1UZ]);
        const Rendered                  burst = renderModeS(std::span<const std::uint8_t>(frame), 3UZ, -0.25);

        PpmScanner integer      = scannerFor(3UZ, 1UZ);
        const Scan integerSeen  = scanOnce(integer, burst, false);
        PpmScanner aligned      = scannerFor(3UZ, 8UZ);
        const Scan searchedSeen = scanOnce(aligned, burst, true);

        expect(integerSeen.admits(std::span<const std::uint8_t>(frame), burst.position));
        expect(searchedSeen.admits(std::span<const std::uint8_t>(frame), burst.position));
        expect(that % (integerSeen.octets == searchedSeen.octets)) << "the frame read out is the same";
        expect(ge(searchedSeen.counters.candidates, searchedSeen.counters.nominations));
        // and at the burst's own position the search is still the better of the two, which is the invariant that holds
        expect(ge(searchedRatio(burst, 3UZ, PpmScanner::kInterpolatorBand), integerRatio(burst, 3UZ)));
    };

    "under noise the search admits at least what an integer decision does, and an integer decision misses some"_test = [] {
        constexpr double                kSigma = 0.09; // per component, against unit-amplitude pulses
        const std::vector<std::uint8_t> frame  = octetsOf(kPublished[0UZ]);

        for (const std::size_t slot : {1UZ, 3UZ}) {
            std::size_t plainAdmitted    = 0UZ;
            std::size_t searchedAdmitted = 0UZ;
            for (std::size_t k = 0UZ; k < 8UZ; ++k) {
                const Rendered burst = renderModeS(std::span<const std::uint8_t>(frame), slot, offsetOf(k), kSigma, 0xC0FFEEULL + k);

                PpmScanner integer = scannerFor(slot, 1UZ);
                plainAdmitted += scanOnce(integer, burst, false).admits(std::span<const std::uint8_t>(frame), burst.position) ? 1UZ : 0UZ;

                PpmScanner aligned = scannerFor(slot, 8UZ);
                searchedAdmitted += scanOnce(aligned, burst, true).admits(std::span<const std::uint8_t>(frame), burst.position) ? 1UZ : 0UZ;
            }
            expect(ge(searchedAdmitted, plainAdmitted)) << std::format("at {} samples a slot the search admitted {} of eight against {}", slot, searchedAdmitted, plainAdmitted);
            if (slot == 1UZ) {
                expect(lt(plainAdmitted, 8UZ)) << "an integer decision still misses the offsets it cannot reach";
            }
        }
    };

    "candidates are never fewer than nominations, and one offset is the plain decision exactly"_test = [] {
        const std::vector<std::uint8_t> frame = octetsOf(kPublished[0UZ]);
        const Rendered                  burst = renderModeS(std::span<const std::uint8_t>(frame), 1UZ, 0.1875, 0.09, 0x5EEDULL);
        const Rendered                  noise = noiseOnly(4096UZ, 0.09, 0xBEEFULL);

        for (const Rendered& input : {burst, noise}) {
            PpmScanner aligned = scannerFor(1UZ, 8UZ);
            const Scan seen    = scanOnce(aligned, input, true);
            expect(ge(seen.counters.candidates, seen.counters.nominations)) << "every nomination passed the coarse test first";
            expect(eq(seen.counters.nominations, seen.counters.admitted + seen.counters.crcFailed + seen.counters.shortFormat));

            PpmScanner one     = scannerFor(1UZ, 1UZ);
            const Scan single  = scanOnce(one, input, true);
            PpmScanner plainer = scannerFor(1UZ, 1UZ);
            const Scan plain   = scanOnce(plainer, input, false);
            expect(eq(single.counters.candidates, single.counters.nominations)) << "one offset is no search, so a candidate is a nomination";
            expect(eq(single.consumed, plain.consumed)) << "and the whole span is decided the same way";
            expect(that % (single.positions == plain.positions)) << "the same positions";
            expect(that % (single.octets == plain.octets)) << "the same frames";
            expect(eq(single.counters.nominations, plain.counters.nominations));
            expect(eq(single.counters.admitted, plain.counters.admitted));
            expect(std::ranges::all_of(single.phases, [](float phase) { return phase == 0.F; })) << "an integer decision reports no offset";
        }

        // the magnitude-only call is the same decision whatever the search would have been set to
        PpmScanner eight = scannerFor(1UZ, 8UZ);
        PpmScanner one   = scannerFor(1UZ, 1UZ);
        const Scan a     = scanOnce(eight, burst, false);
        const Scan b     = scanOnce(one, burst, false);
        expect(eq(a.consumed, b.consumed));
        expect(that % (a.positions == b.positions));
        expect(that % (a.octets == b.octets));
    };

    "a span shorter than one window decides nothing, and a shorter sample span is refused"_test = [] {
        const std::vector<std::uint8_t> frame   = octetsOf(kPublished[0UZ]);
        const Rendered                  burst   = renderModeS(std::span<const std::uint8_t>(frame), 1UZ, 0.0);
        PpmScanner                      aligned = scannerFor(1UZ, 8UZ);

        expect(eq(aligned.guardSamples(), 4UZ)) << "an eight-tap window reads four samples before the instant it lands on";
        expect(eq(aligned.phasedWindowSamples(), aligned.windowSamples() + 8UZ)) << "a guard at either end";

        const auto                                 nothing = [](const PpmFrame&) {};
        const std::span<const float>               short_(burst.magnitudes.data(), aligned.phasedWindowSamples() - 1UZ);
        const std::span<const std::complex<float>> shortSamples(burst.samples.data(), aligned.phasedWindowSamples() - 1UZ);
        expect(eq(aligned.consume(short_, shortSamples, nothing), 0UZ)) << "one sample short of a window decides nothing";

        PpmScanner other = scannerFor(1UZ, 8UZ);
        expect(throws([&other, &burst] {
            const auto nothingElse = [](const PpmFrame&) {};
            std::ignore            = other.consume(burst.magnitudes, std::span<const std::complex<float>>(burst.samples.data(), burst.samples.size() - 1UZ), nothingElse);
        })) << "samples shorter than the magnitudes would misalign the two";
    };

    "the interpolator's band is chosen on the ratio the search delivers, and the choice is measured"_test = [] {
        const std::vector<std::uint8_t> frame   = octetsOf(kPublished[0UZ]);
        constexpr double                kNarrow = 0.25;

        // the criterion: the worst of the eight offsets, which is what decides whether a burst is framed at all
        const auto worstRatio = [&frame](std::size_t slot, double band) {
            float worst = std::numeric_limits<float>::max();
            for (std::size_t k = 0UZ; k < 8UZ; ++k) {
                const Rendered burst = renderModeS(std::span<const std::uint8_t>(frame), slot, offsetOf(k));
                worst                = std::min(worst, searchedRatio(burst, slot, band));
            }
            return worst;
        };
        for (const std::size_t slot : {1UZ, 3UZ}) {
            const float chosen = worstRatio(slot, PpmScanner::kInterpolatorBand);
            const float narrow = worstRatio(slot, kNarrow);
            expect(gt(chosen, narrow)) << std::format("at {} samples a slot the worst of eight offsets reaches {:.3f} at band {:.2f}, against {:.3f} at {:.2f}", slot, chosen, PpmScanner::kInterpolatorBand, narrow, kNarrow);
        }

        // and the reason the reconstruction error is not the criterion: it prefers the opposite band, a narrow fit
        // winning on error energy by attenuating the content past Nyquist that it cannot reproduce either way
        const Rendered here       = renderModeS(std::span<const std::uint8_t>(frame), 1UZ, 0.0);
        const Rendered later      = renderModeS(std::span<const std::uint8_t>(frame), 1UZ, 0.5);
        const auto     delayError = [&here, &later](double band) {
            const gr::sync::MmseInterpolatorBank bank(PpmScanner::kInterpolatorTaps, 8, band, false);
            const std::size_t                    row       = bank.row(0.5);
            double                               error     = 0.0;
            double                               reference = 0.0;
            for (std::size_t n = 3UZ; n + 4UZ < here.samples.size(); ++n) {
                const std::complex<float> got = bank.interpolate<std::complex<float>>(here.samples.data() + n - 3UZ, row);
                error += static_cast<double>(std::norm(got - later.samples[n]));
                reference += static_cast<double>(std::norm(later.samples[n]));
            }
            return std::sqrt(error / reference);
        };
        const double chosenError = delayError(PpmScanner::kInterpolatorBand);
        const double narrowError = delayError(kNarrow);
        // measured: 0.466 against 0.288, half a sample of delay as a fraction of the waveform's own mean square
        expect(gt(chosenError, narrowError)) << std::format("half a sample of delay costs {:.3f} at band {:.2f} and {:.3f} at {:.2f}", chosenError, PpmScanner::kInterpolatorBand, narrowError, kNarrow);
        expect(lt(chosenError, 1.0)) << "and both are a fit rather than a failure to reproduce anything at all";
    };
};

int main() { /* not needed for UT */ }
