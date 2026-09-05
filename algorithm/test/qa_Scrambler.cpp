#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <format>
#include <print>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/algorithm/digital/BitPacking.hpp>
#include <gnuradio-4.0/algorithm/digital/Scrambler.hpp>

namespace {

using gr::digital::applyProfile;
using gr::digital::BitOrder;
using gr::digital::BitRepack;
using gr::digital::configureExplicit;
using gr::digital::configureMonitor;
using gr::digital::configureRepack;
using gr::digital::configureScrambler;
using gr::digital::degreeOf;
using gr::digital::delayListFromTaps;
using gr::digital::kProfiles;
using gr::digital::minDelayOf;
using gr::digital::profileByName;
using gr::digital::ProfileFamily;
using gr::digital::repack;
using gr::digital::resetScrambler;
using gr::digital::scramble;
using gr::digital::ScramblerConfig;
using gr::digital::ScramblerMode;
using gr::digital::seedFromBitString;
using gr::digital::SequenceForm;
using gr::digital::sequenceFromHex;
using gr::digital::SequenceSource;
using gr::digital::tapCountOf;
using gr::digital::TapMask;
using gr::digital::tapsFromDelayList;

constexpr BitOrder kOrders[] = {BitOrder::MsbFirst, BitOrder::LsbFirst};

constexpr ScramblerMode kModes[] = {ScramblerMode::Additive, ScramblerMode::MultiplicativeScramble, ScramblerMode::MultiplicativeDescramble};

/// One row of the named-polynomial table, with the sequence it produces under this header's convention.
struct Entry {
    std::string_view name;
    std::string_view taps;
    std::string_view seed;
    std::size_t      period;
    std::string_view bytes;
};

constexpr Entry kAdditive[] = {{"CCSDS", "1,3,5,8", "11111111", 255UZ, "FF 48 0E C0 9A 0D 70 BC"}, //
    {"IEEE 802.11", "4,7", "0000111", 127UZ, "0E F2 C9 02 26 2E B6 0C"},                           //
    {"DVB", "14,15", "000000111111011", 32767UZ, "03 F6 08 34 30 B8 A3 93"},                       //
    {"ITU form A", "18,23", "11111111111111111111111", 8388607UZ, "FF FF FE 00 00 7C 00 1F"},      //
    {"ITU V.22", "14,17", "11111111111111111", 131071UZ, "FF FF 80 01 C0 07 E0 1C"},               //
    {"ITU form B", "5,23", "11111111111111111111111", 8388607UZ, "FF FF FE 0F 83 E3 07 3E"}};

/// The self-synchronizing entries, plus a four-tap set so error multiplication is measured at two tap counts.
constexpr std::string_view kMultiplicative[] = {"18,23", "5,23", "14,17", "4,7", "39,58", "1,2,5,7"};

[[nodiscard]] ScramblerConfig configured(std::string_view taps, std::uint64_t seed, ScramblerMode mode, unsigned width, BitOrder order) {
    ScramblerConfig cfg{};
    configureScrambler(cfg, tapsFromDelayList(taps), seed, mode, width, order);
    return cfg;
}

[[nodiscard]] ScramblerConfig additive(std::string_view taps, std::string_view seed, unsigned width = 8U, BitOrder order = BitOrder::MsbFirst) { return configured(taps, seedFromBitString(seed, degreeOf(tapsFromDelayList(taps))), ScramblerMode::Additive, width, order); }

[[nodiscard]] std::uint64_t maskOf(std::uint8_t degree) { return degree == 64U ? ~std::uint64_t{0} : ((std::uint64_t{1} << degree) - std::uint64_t{1}); }

[[nodiscard]] std::vector<std::uint8_t> through(ScramblerConfig& cfg, std::span<const std::uint8_t> in) {
    std::vector<std::uint8_t> out(in.size(), std::uint8_t{0xAA});
    scramble(cfg, in, out);
    return out;
}

/// The generator's own output, which is what an all-zero input scrambles to.
[[nodiscard]] std::vector<std::uint8_t> sequenceItems(ScramblerConfig& cfg, std::size_t count) {
    const std::vector<std::uint8_t> zeros(count, std::uint8_t{0});
    return through(cfg, zeros);
}

/// @p count sequence bits, one to an item.
[[nodiscard]] std::vector<std::uint8_t> sequenceBits(std::string_view taps, std::string_view seed, std::size_t count) {
    ScramblerConfig cfg = additive(taps, seed, 1U);
    return sequenceItems(cfg, count);
}

[[nodiscard]] std::string hexOf(std::span<const std::uint8_t> bytes) {
    std::string text;
    for (const std::uint8_t byte : bytes) {
        if (!text.empty()) {
            text.push_back(' ');
        }
        text += std::format("{:02X}", byte);
    }
    return text;
}

[[nodiscard]] std::vector<std::uint8_t> randomItems(std::size_t count, unsigned width, std::uint64_t seed) {
    std::mt19937_64           engine{seed};
    const std::uint32_t       mask = (1U << width) - 1U;
    std::vector<std::uint8_t> items(count);
    for (std::uint8_t& item : items) {
        item = static_cast<std::uint8_t>(static_cast<std::uint32_t>(engine() & 0xFFULL) & mask);
    }
    return items;
}

/// The first differing index, or the common size when the two agree.
[[nodiscard]] std::size_t firstDifference(std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) {
    if (a.size() != b.size()) {
        return std::min(a.size(), b.size());
    }
    for (std::size_t i = 0UZ; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            return i;
        }
    }
    return a.size();
}

[[nodiscard]] std::size_t agreementCount(std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) { //
    std::size_t agree = 0UZ;
    for (std::size_t i = 0UZ; i < std::min(a.size(), b.size()); ++i) {
        agree += a[i] == b[i] ? 1UZ : 0UZ;
    }
    return agree;
}

/// The longest run of @p value in the periodic sequence, taken over one period and wrapping.
[[nodiscard]] std::size_t longestRun(std::span<const std::uint8_t> bits, std::uint8_t value, std::size_t period) {
    std::size_t best    = 0UZ;
    std::size_t current = 0UZ;
    for (std::size_t i = 0UZ; i < period + 64UZ && i < bits.size(); ++i) {
        current = bits[i] == value ? current + 1UZ : 0UZ;
        best    = std::max(best, current);
    }
    return best;
}

[[nodiscard]] std::vector<std::size_t> primeDivisors(std::size_t value) {
    std::vector<std::size_t> primes;
    for (std::size_t factor = 2UZ; factor * factor <= value; ++factor) {
        if (value % factor == 0UZ) {
            primes.push_back(factor);
            while (value % factor == 0UZ) {
                value /= factor;
            }
        }
    }
    if (value > 1UZ) {
        primes.push_back(value);
    }
    return primes;
}

/// The stream divided into calls of @p chunk items, with a reset every @p period items when that is not zero.
[[nodiscard]] std::vector<std::uint8_t> inChunks(ScramblerConfig cfg, std::span<const std::uint8_t> in, std::size_t chunk, std::size_t period) {
    std::vector<std::uint8_t> out(in.size(), std::uint8_t{0xAA});
    std::size_t               index = 0UZ;
    std::size_t               since = 0UZ;
    while (index < in.size()) {
        if (period != 0UZ && since == period) {
            resetScrambler(cfg);
            since = 0UZ;
        }
        std::size_t take = std::min(chunk, in.size() - index);
        if (period != 0UZ) {
            take = std::min(take, period - since);
        }
        scramble(cfg, in.subspan(index, take), std::span<std::uint8_t>{out}.subspan(index, take));
        index += take;
        since += take;
    }
    return out;
}

/// One bit to a byte, packed most significant bit first, which is how `sequenceFromHex` spells a mask.
[[nodiscard]] std::vector<std::uint8_t> packMsbFirst(std::span<const std::uint8_t> bits) {
    std::vector<std::uint8_t> bytes((bits.size() + 7UZ) / 8UZ, std::uint8_t{0});
    for (std::size_t i = 0UZ; i < bits.size(); ++i) {
        bytes[i >> 3U] = static_cast<std::uint8_t>(bytes[i >> 3U] | ((bits[i] & 1U) << (7UZ - (i & 7UZ))));
    }
    return bytes;
}

/// What the forced-transition rule did, recomputed from the transmitted stream alone: the monitored comparison at every
/// bit and the bits at which the forcing fired. Nothing here reads the scrambler, which is the property being asserted.
struct Replay {
    std::vector<std::uint8_t> differs;
    std::vector<std::uint8_t> fired;
    std::size_t               firings = 0UZ;
};

[[nodiscard]] Replay replayForcing(std::span<const std::uint8_t> transmitted, std::uint32_t modulus, unsigned nearer, unsigned further) {
    Replay        replay{std::vector<std::uint8_t>(transmitted.size(), std::uint8_t{0}), std::vector<std::uint8_t>(transmitted.size(), std::uint8_t{0}), 0UZ};
    std::uint64_t window  = 0ULL;
    std::uint32_t counter = 0U;
    for (std::size_t k = 0UZ; k < transmitted.size(); ++k) {
        const std::uint8_t force = counter == modulus - 1U ? std::uint8_t{1} : std::uint8_t{0};
        replay.fired[k]          = force;
        replay.firings += force;

        const std::uint8_t differs = static_cast<std::uint8_t>(((window >> (nearer - 1U)) ^ (window >> (further - 1U))) & 1ULL);
        replay.differs[k]          = differs;
        counter                    = differs != 0U ? 0U : (counter + 1U) % modulus;
        window                     = (window << 1U) | static_cast<std::uint64_t>(transmitted[k] & 1U);
    }
    return replay;
}

/// The input that holds a multiplicative scrambler's output at all zeros: a descrambler of the same shape run over
/// zeros produces exactly it, whatever the seed. `[in]` carries the same configuration the scrambler will use.
[[nodiscard]] std::vector<std::uint8_t> lockUpInput(std::string_view taps, std::uint64_t seed, bool invert, std::uint32_t modulus, std::size_t count) {
    ScramblerConfig cfg = configured(taps, seed, ScramblerMode::MultiplicativeDescramble, 1U, BitOrder::MsbFirst);
    configureMonitor(cfg, invert, modulus, modulus == 0U ? "" : "1,2");
    const std::vector<std::uint8_t> zeros(count, std::uint8_t{0});
    return through(cfg, zeros);
}

/// The length of the leading run of items equal to the first one.
[[nodiscard]] std::size_t leadingRun(std::span<const std::uint8_t> items) {
    std::size_t run = 0UZ;
    while (run < items.size() && items[run] == items[0]) {
        ++run;
    }
    return run;
}

} // namespace

const boost::ut::suite<"scrambler"> scramblerTests = [] {
    using namespace boost::ut;

    // The strongest clean-room evidence in the family: the sequence CCSDS 131.0-B publishes.
    "CCSDS reproduces the sequence its standard publishes"_test = [] {
        ScramblerConfig cfg = additive("1,3,5,8", "11111111");
        expect(eq(hexOf(sequenceItems(cfg, 8UZ)), std::string{"FF 48 0E C0 9A 0D 70 BC"})) << "the first five bytes are the value the standard publishes";
        expect(that % (cfg.form == SequenceForm::Table)) << "degree 8 takes the table form";
        expect(eq(gr::digital::standard::ccsds131, tapsFromDelayList("1,3,5,8"))) << "the named constant carries the reading that reproduces the check value";

        const std::vector<std::uint8_t> plain{0x47, 0x53, 0x4F, 0x43, 0x21, 0x21, 0x21, 0x21};
        resetScrambler(cfg);
        const auto scrambled = through(cfg, plain);
        expect(eq(hexOf(scrambled), std::string{"B8 1B 41 83 BB 2C 51 9D"})) << "the worked example";
        resetScrambler(cfg);
        expect(eq(firstDifference(through(cfg, scrambled), plain), plain.size())) << "scrambling twice returns the plain text";

        // the sequence does not depend on the data, which is the whole of what synchronous buys
        constexpr std::array<std::uint8_t, 8> kSequence{0xFF, 0x48, 0x0E, 0xC0, 0x9A, 0x0D, 0x70, 0xBC};
        const auto                            noise = randomItems(kSequence.size(), 8U, 0x243F6A8885A308D3ULL);
        resetScrambler(cfg);
        const auto over = through(cfg, noise);
        for (std::size_t i = 0UZ; i < noise.size(); ++i) {
            expect(eq(static_cast<unsigned>(over[i]), static_cast<unsigned>(noise[i] ^ kSequence[i]))) << std::format("item {}", i);
        }
    };

    // The other published check value, and the period it is drawn from.
    "IEEE 802.11 reproduces its published sequence and repeats at 127"_test = [] {
        ScramblerConfig cfg = additive("4,7", "0000111");
        expect(eq(hexOf(sequenceItems(cfg, 8UZ)), std::string{"0E F2 C9 02 26 2E B6 0C"})) << "all 127 bits the standard publishes for the all-ones initial state";
        expect(eq(gr::digital::standard::ieee80211, tapsFromDelayList("4,7")));

        const auto  bits    = sequenceBits("4,7", "0000111", 1200UZ);
        std::size_t offside = 0UZ;
        for (std::size_t i = 0UZ; i + 127UZ < bits.size(); ++i) {
            offside += bits[i] == bits[i + 127UZ] ? 0UZ : 1UZ;
        }
        expect(eq(offside, 0UZ)) << "bit k + 127 is bit k over more than a thousand bits";
    };

    // Section 1.5's trap: the reciprocal delay set is a different sequence, so a tap parser that reverses
    //    the set is caught rather than passing a symmetric round trip.
    "the reciprocal tap reading is a different sequence"_test = [] {
        struct Pair {
            std::string_view name;
            std::string_view taps;
            std::string_view reciprocal;
            std::string_view seed;
            std::string_view bytes;
            std::string_view reciprocalBytes;
        };
        constexpr Pair kPairs[] = {{"CCSDS", "1,3,5,8", "3,5,7,8", "11111111", "FF 48 0E C0 9A 0D 70 BC", "FF 1A AF 66 52 23 1E 10"}, //
            {"IEEE 802.11", "4,7", "3,7", "0000111", "0E F2 C9 02 26 2E B6 0C", "0F E3 B1 4B EA 85 BC E5"},                           //
            {"DVB", "14,15", "1,15", "000000111111011", "03 F6 08 34 30 B8 A3 93", "03 F7 FA B5 59 B3 22 44"}};

        for (const Pair& pair : kPairs) {
            ScramblerConfig chosen     = additive(pair.taps, pair.seed);
            ScramblerConfig reciprocal = additive(pair.reciprocal, pair.seed);
            const auto      wanted     = hexOf(sequenceItems(chosen, 8UZ));
            const auto      other      = hexOf(sequenceItems(reciprocal, 8UZ));

            expect(eq(wanted, std::string{pair.bytes})) << pair.name;
            expect(eq(other, std::string{pair.reciprocalBytes})) << std::format("{} reciprocal", pair.name);
            expect(that % (wanted != other)) << std::format("{}: the two readings must not agree, or the check value proves nothing", pair.name);
        }

        // DVB carries the other trap as well, the one the output-tap convention decides: its standard loads
        // `100101010000000` into the register but emits the feedback bit, so reading the register's contents out first
        // is a third sequence again, and it is the one a seed copied straight from the initialization text produces.
        ScramblerConfig emitted    = additive("14,15", "100101010000000");
        const auto      wrongPhase = hexOf(sequenceItems(emitted, 8UZ));
        expect(eq(wrongPhase, std::string{"95 01 7E 07 04 12 18 6C"})) << "emitting the register's contents first";

        ScramblerConfig figure = additive("14,15", "000000111111011");
        expect(that % (hexOf(sequenceItems(figure, 8UZ)) != wrongPhase)) << "and that is not the sequence the standard's figure prints";
    };

    // Every named additive generator is maximal length, walked over one full period.
    "every named generator is maximal length"_test = [] {
        for (const Entry& entry : kAdditive) {
            const std::uint8_t degree = degreeOf(tapsFromDelayList(entry.taps));
            const std::size_t  period = entry.period;

            expect(that % (maskOf(degree) == period)) << std::format("{}: the period must be 2^{} - 1", entry.name, degree);

            const auto  bits = sequenceBits(entry.taps, entry.seed, period + 400UZ);
            std::size_t ones = 0UZ;
            for (std::size_t i = 0UZ; i < period; ++i) {
                ones += bits[i];
            }
            expect(eq(ones, period / 2UZ + 1UZ)) << std::format("{}: a maximal sequence carries 2^(n-1) ones", entry.name);
            expect(eq(period - ones, period / 2UZ)) << std::format("{}: and 2^(n-1) - 1 zeros", entry.name);
            expect(eq(longestRun(bits, std::uint8_t{1}, period), static_cast<std::size_t>(degree))) << std::format("{}: longest run of ones", entry.name);
            expect(eq(longestRun(bits, std::uint8_t{0}, period), static_cast<std::size_t>(degree) - 1UZ)) << std::format("{}: longest run of zeros", entry.name);

            // the sequence repeats at `period`, and at no proper divisor of it, which fixes the period exactly
            std::size_t offside = 0UZ;
            for (std::size_t i = 0UZ; i < 128UZ; ++i) {
                offside += bits[i] == bits[i + period] ? 0UZ : 1UZ;
            }
            expect(eq(offside, 0UZ)) << std::format("{}: the sequence repeats at its period", entry.name);
            for (const std::size_t prime : primeDivisors(period)) {
                const std::size_t shorter = period / prime;
                bool              differs = false;
                for (std::size_t i = 0UZ; i < 128UZ && !differs; ++i) {
                    differs = bits[i] != bits[i + shorter];
                }
                expect(differs) << std::format("{}: {} is not a period, so the period is not a proper divisor", entry.name, shorter);
            }
        }
    };

    // Two-valued periodic autocorrelation, the m-sequence property and the check that the walk is right.
    "periodic autocorrelation is exactly -1 at every non-zero shift"_test = [] {
        for (const Entry& entry : std::span<const Entry>{kAdditive}.first(3UZ)) {
            const std::size_t period = entry.period;
            const auto        bits   = sequenceBits(entry.taps, entry.seed, period + 400UZ);

            std::size_t offside = 0UZ;
            for (std::size_t shift = 1UZ; shift < std::min(400UZ, period); ++shift) {
                std::ptrdiff_t correlation = 0;
                for (std::size_t i = 0UZ; i < period; ++i) {
                    correlation += bits[i] == bits[i + shift] ? 1 : -1;
                }
                offside += correlation == -1 ? 0UZ : 1UZ;
            }
            expect(eq(offside, 0UZ)) << std::format("{}: every non-zero shift correlates to exactly -1", entry.name);
        }
    };

    // XOR is an involution, which is why one block serves both directions of the additive family.
    "the additive scrambler is its own inverse"_test = [] {
        std::size_t cases    = 0UZ;
        std::size_t failures = 0UZ;
        for (const Entry& entry : kAdditive) {
            for (const unsigned width : {1U, 2U, 3U, 4U, 8U}) {
                for (const BitOrder order : kOrders) {
                    const auto      source = randomItems(4096UZ, width, 0x13198A2E03707344ULL + cases);
                    ScramblerConfig cfg    = additive(entry.taps, entry.seed, width, order);
                    const auto      once   = through(cfg, source);
                    resetScrambler(cfg);
                    const auto twice = through(cfg, once);

                    ++cases;
                    const bool ok = firstDifference(twice, source) == source.size();
                    failures += ok ? 0UZ : 1UZ;
                    expect(ok) << std::format("{} at {} bits, {} differs at item {}", entry.name, width, order == BitOrder::MsbFirst ? "msb_first" : "lsb_first", firstDifference(twice, source));
                }
            }
        }
        expect(eq(cases, 60UZ));
        expect(eq(failures, 0UZ));
    };

    // The two multiplicative modes invert each other exactly, from item zero, when both carry the same seed.
    "the multiplicative pair round-trips from item zero"_test = [] {
        std::size_t cases    = 0UZ;
        std::size_t failures = 0UZ;
        for (const std::string_view taps : kMultiplicative) {
            const std::uint8_t degree = degreeOf(tapsFromDelayList(taps));
            for (const unsigned width : {1U, 8U}) {
                for (const BitOrder order : kOrders) {
                    const std::size_t   items  = 8000UZ / width + 1UZ;
                    const auto          source = randomItems(items, width, 0xA4093822299F31D0ULL + cases);
                    const std::uint64_t seed   = std::uint64_t{0x9E3779B97F4A7C15} & maskOf(degree);

                    ScramblerConfig transmit = configured(taps, seed, ScramblerMode::MultiplicativeScramble, width, order);
                    ScramblerConfig receive  = configured(taps, seed, ScramblerMode::MultiplicativeDescramble, width, order);
                    const auto      channel  = through(transmit, source);
                    const auto      received = through(receive, channel);

                    ++cases;
                    const bool ok = firstDifference(received, source) == source.size();
                    failures += ok ? 0UZ : 1UZ;
                    expect(ok) << std::format("taps {} at {} bits differs at item {}", taps, width, firstDifference(received, source));
                }
            }
        }
        expect(eq(cases, 24UZ));
        expect(eq(failures, 0UZ));
    };

    // The self-synchronizing property itself: the register holds only received bits, so it is data after `degree`
    //    of them whatever it started as.
    "a wrong seed costs a descrambler exactly its first degree bits"_test = [] {
        std::mt19937_64 engine{0x082EFA98EC4E6C89ULL};
        for (const std::string_view taps : kMultiplicative) {
            const std::uint8_t degree  = degreeOf(tapsFromDelayList(taps));
            const auto         source  = randomItems(4000UZ, 1U, 0x452821E638D01377ULL);
            ScramblerConfig    tx      = configured(taps, 0ULL, ScramblerMode::MultiplicativeScramble, 1U, BitOrder::MsbFirst);
            const auto         channel = through(tx, source);

            std::size_t lateFailures = 0UZ;
            std::size_t worst        = 0UZ;
            for (std::size_t trial = 0UZ; trial < 50UZ; ++trial) {
                const std::uint64_t wrong = engine() & maskOf(degree);
                ScramblerConfig     rx    = configured(taps, wrong, ScramblerMode::MultiplicativeDescramble, 1U, BitOrder::MsbFirst);
                const auto          out   = through(rx, channel);

                std::size_t firstCorrect = 0UZ;
                for (std::size_t i = 0UZ; i < out.size(); ++i) {
                    if (out[i] != source[i]) {
                        firstCorrect = i + 1UZ;
                        lateFailures += i >= static_cast<std::size_t>(degree) ? 1UZ : 0UZ;
                    }
                }
                worst = std::max(worst, firstCorrect);
            }
            expect(eq(lateFailures, 0UZ)) << std::format("taps {}: every bit from index {} onward must be correct", taps, degree);
            expect(eq(worst, static_cast<std::size_t>(degree))) << std::format("taps {}: and the bound is tight, not vacuous", taps);
        }
    };

    // The same theorem stated as a mid-stream join, which is what a receiver switched on mid-burst does.
    "a mid-stream join converges in degree bits"_test = [] {
        for (const std::string_view taps : kMultiplicative) {
            const std::uint8_t degree  = degreeOf(tapsFromDelayList(taps));
            const auto         source  = randomItems(4000UZ, 1U, 0xBE5466CF34E90C6CULL);
            ScramblerConfig    tx      = configured(taps, std::uint64_t{0x2B7E151628AED2A6} & maskOf(degree), ScramblerMode::MultiplicativeScramble, 1U, BitOrder::MsbFirst);
            const auto         channel = through(tx, source);

            for (const std::size_t join : {1UZ, 7UZ, 137UZ, 1999UZ}) {
                ScramblerConfig rx  = configured(taps, std::uint64_t{0xABF7158809CF4F3C} & maskOf(degree), ScramblerMode::MultiplicativeDescramble, 1U, BitOrder::MsbFirst);
                const auto      out = through(rx, std::span<const std::uint8_t>{channel}.subspan(join));

                std::size_t offside = 0UZ;
                for (std::size_t i = static_cast<std::size_t>(degree); i < out.size(); ++i) {
                    offside += out[i] == source[join + i] ? 0UZ : 1UZ;
                }
                expect(eq(offside, 0UZ)) << std::format("taps {} joined at {} must be correct from bit {}", taps, join, degree);
            }
        }
    };

    // The engineering trade itself, both halves in one place: 1 against 1 + tapCount.
    "error multiplication: one bit becomes 1, or 1 + tapCount"_test = [] {
        std::mt19937_64 engine{0x3F84D5B5B5470917ULL};
        const auto      source = randomItems(4000UZ, 1U, 0x9216D5D98979FB1BULL);

        for (const std::string_view taps : kMultiplicative) {
            const TapMask      mask   = tapsFromDelayList(taps);
            const std::uint8_t degree = degreeOf(mask);
            const std::size_t  wanted = 1UZ + static_cast<std::size_t>(tapCountOf(mask));

            ScramblerConfig tx      = configured(taps, 0ULL, ScramblerMode::MultiplicativeScramble, 1U, BitOrder::MsbFirst);
            const auto      channel = through(tx, source);

            std::size_t offside = 0UZ;
            for (std::size_t trial = 0UZ; trial < 50UZ; ++trial) {
                std::vector<std::uint8_t> corrupted = channel;
                const std::size_t         position  = static_cast<std::size_t>(degree) + 8UZ + engine() % (corrupted.size() - 2UZ * static_cast<std::size_t>(degree) - 16UZ);
                corrupted[position] ^= std::uint8_t{1};

                ScramblerConfig rx  = configured(taps, 0ULL, ScramblerMode::MultiplicativeDescramble, 1U, BitOrder::MsbFirst);
                const auto      out = through(rx, corrupted);
                offside += (out.size() - agreementCount(out, source)) == wanted ? 0UZ : 1UZ;
            }
            expect(eq(offside, 0UZ)) << std::format("taps {}: one channel error must produce exactly {} output errors", taps, wanted);
        }

        for (const Entry& entry : kAdditive) {
            ScramblerConfig tx      = additive(entry.taps, entry.seed, 1U);
            const auto      channel = through(tx, source);

            std::size_t offside = 0UZ;
            for (std::size_t trial = 0UZ; trial < 50UZ; ++trial) {
                std::vector<std::uint8_t> corrupted = channel;
                corrupted[engine() % corrupted.size()] ^= std::uint8_t{1};

                ScramblerConfig rx  = additive(entry.taps, entry.seed, 1U);
                const auto      out = through(rx, corrupted);
                offside += (out.size() - agreementCount(out, source)) == 1UZ ? 0UZ : 1UZ;
            }
            expect(eq(offside, 0UZ)) << std::format("{}: an additive descrambler multiplies a channel error by exactly one", entry.name);
        }
    };

    // Why the additive family needs an explicit boundary at all: a phase error is total, not recoverable.
    "an additive descrambler out of phase agrees at chance"_test = [] {
        constexpr std::size_t kFrame = 8160UZ; // one CCSDS transfer frame's worth of bits
        const auto            source = randomItems(kFrame, 1U, 0xC0AC29B7C97C50DDULL);

        ScramblerConfig tx      = additive("1,3,5,8", "11111111", 1U);
        const auto      channel = through(tx, source);

        for (const std::size_t shift : {1UZ, 2UZ, 8UZ}) {
            ScramblerConfig                 rx = additive("1,3,5,8", "11111111", 1U);
            const std::vector<std::uint8_t> lost(shift, std::uint8_t{0});
            (void)through(rx, lost); // the phase the descrambler is running at is `shift` bits ahead

            const std::size_t agree = agreementCount(through(rx, channel), source);
            expect(that % (agree > kFrame * 45UZ / 100UZ && agree < kFrame * 55UZ / 100UZ)) << std::format("a {}-bit phase error left {} of {} bits correct, which should be chance", shift, agree, kFrame);
        }

        ScramblerConfig                 rx = additive("1,3,5,8", "11111111", 1U);
        const std::vector<std::uint8_t> whole(255UZ, std::uint8_t{0});
        (void)through(rx, whole);
        expect(eq(agreementCount(through(rx, channel), source), kFrame)) << "only a shift by a whole period comes back into step";
    };

    // The reset boundary as the kernel offers it, and what it is worth on each side.
    "reset restores the epoch, and a round trip through resets"_test = [] {
        constexpr std::size_t kEpoch  = 1504UZ; // eight 188-byte DVB packets
        constexpr std::size_t kEpochs = 12UZ;
        const auto            source  = randomItems(kEpoch * kEpochs, 8U, 0x9CC4F4C9C6C1F0B1ULL);

        ScramblerConfig cfg   = additive("14,15", "000000111111011");
        const auto      first = through(cfg, source);
        resetScrambler(cfg);
        expect(eq(firstDifference(through(cfg, source), first), source.size())) << "reset puts the register back to the seed exactly";

        // an epoch is independent of every other epoch
        std::vector<std::uint8_t> epochwise;
        epochwise.reserve(source.size());
        for (std::size_t epoch = 0UZ; epoch < kEpochs; ++epoch) {
            ScramblerConfig fresh = additive("14,15", "000000111111011");
            const auto      part  = through(fresh, std::span<const std::uint8_t>{source}.subspan(epoch * kEpoch, kEpoch));
            epochwise.insert(epochwise.end(), part.begin(), part.end());
        }
        ScramblerConfig periodic = additive("14,15", "000000111111011");
        const auto      channel  = inChunks(periodic, source, 4096UZ, kEpoch);
        expect(eq(firstDifference(channel, epochwise), source.size())) << "a reset every epoch is the same as running each epoch from the seed";

        // the round trip, both halves resetting
        ScramblerConfig receive   = additive("14,15", "000000111111011");
        const auto      recovered = inChunks(receive, channel, 4096UZ, kEpoch);
        expect(eq(firstDifference(recovered, source), source.size())) << "scrambler and descrambler both at the epoch: exact";

        // and the negative half, which is what makes the positive one mean something
        ScramblerConfig   adrift    = additive("14,15", "000000111111011");
        const auto        unreset   = through(adrift, channel);
        const std::size_t lateItems = source.size() - kEpoch;
        const std::size_t lateAgree = agreementCount(std::span<const std::uint8_t>{unreset}.subspan(kEpoch), std::span<const std::uint8_t>{source}.subspan(kEpoch));
        expect(eq(firstDifference(std::span<const std::uint8_t>{unreset}.first(kEpoch), std::span<const std::uint8_t>{source}.first(kEpoch)), kEpoch)) << "the first epoch is right whatever happens after it";
        expect(that % (lateAgree * 20UZ < lateItems)) << std::format("a descrambler that misses the boundary kept {} of {} items, and should keep almost none", lateAgree, lateItems);

        // CCSDS's transfer frame is exactly 32 periods of its sequence, so there the boundary costs nothing
        const auto      frame   = randomItems(1020UZ * 4UZ, 8U, 0x1B0FA2C1E29A1D3EULL);
        ScramblerConfig ccsdsTx = additive("1,3,5,8", "11111111");
        const auto      framed  = inChunks(ccsdsTx, frame, 4096UZ, 1020UZ);
        ScramblerConfig ccsdsRx = additive("1,3,5,8", "11111111");
        expect(eq(firstDifference(through(ccsdsRx, framed), frame), frame.size())) << "1020 items is 8160 bits, which is 32 whole periods of the 255-bit sequence";
    };

    // Forgetting to write the register back at the end of a call is the classic bug, and it is right for
    //     large chunks and wrong for small ones.
    "the output does not depend on how the stream is divided into calls"_test = [] {
        constexpr std::size_t kEpoch   = 977UZ; // deliberately coprime with every chunk size below
        std::size_t           cases    = 0UZ;
        std::size_t           failures = 0UZ;

        for (const std::string_view taps : {"1,3,5,8", "14,15", "18,23", "39,58", "1,2,5,7"}) {
            const std::uint8_t degree = degreeOf(tapsFromDelayList(taps));
            for (const unsigned width : {1U, 8U}) {
                for (const ScramblerMode mode : kModes) {
                    const auto      source = randomItems(5000UZ, width, 0xDA1D8CB0DB0C0EA6ULL + cases);
                    ScramblerConfig cfg    = configured(taps, std::uint64_t{0x5AA5F00F} & maskOf(degree), mode, width, BitOrder::MsbFirst);
                    const auto      whole  = inChunks(cfg, source, source.size(), kEpoch);

                    for (const std::size_t chunk : {1UZ, 3UZ, 17UZ, 4096UZ}) {
                        const auto part = inChunks(cfg, source, chunk, kEpoch);
                        ++cases;
                        const bool ok = firstDifference(part, whole) == whole.size();
                        failures += ok ? 0UZ : 1UZ;
                        expect(ok) << std::format("taps {} at {} bits, chunks of {}, differs at item {}", taps, width, chunk, firstDifference(part, whole));
                    }
                }
            }
        }
        expect(eq(cases, 120UZ));
        expect(eq(failures, 0UZ));
    };

    // The scrambler and the packer sit next to each other in every chain that has one, and they share a
    //     bit-stream model rather than each carrying its own.
    "composition with the bit-packing conventions"_test = [] {
        constexpr std::size_t kBytes = 768UZ; // whole periods for every width below
        const auto            source = randomItems(kBytes, 8U, 0x6C44198C4A475817ULL);

        std::size_t cases = 0UZ;
        for (const BitOrder order : kOrders) {
            ScramblerConfig direct = additive("1,3,5,8", "11111111", 8U, order);
            const auto      wanted = through(direct, source);

            for (const unsigned width : {1U, 2U, 3U, 4U, 8U}) {
                BitRepack down{};
                configureRepack(down, 8U, width, order, order);
                BitRepack up{};
                configureRepack(up, width, 8U, order, order);

                std::vector<std::uint8_t> fields(kBytes * 8UZ / width, std::uint8_t{0});
                repack(down, source, fields);

                ScramblerConfig stepped   = additive("1,3,5,8", "11111111", width, order);
                const auto      scrambled = through(stepped, fields);

                std::vector<std::uint8_t> packed(kBytes, std::uint8_t{0});
                repack(up, scrambled, packed);

                ++cases;
                expect(eq(firstDifference(packed, wanted), kBytes)) << std::format("scrambling {} bits at a time under {} differs from scrambling whole bytes", width, order == BitOrder::MsbFirst ? "msb_first" : "lsb_first");
            }
        }
        expect(eq(cases, 10UZ));

        // and the negative half: the two orders are not interchangeable
        ScramblerConfig   msb   = additive("1,3,5,8", "11111111", 8U, BitOrder::MsbFirst);
        ScramblerConfig   lsb   = additive("1,3,5,8", "11111111", 8U, BitOrder::LsbFirst);
        const std::size_t agree = agreementCount(through(msb, source), through(lsb, source));
        expect(that % (agree * 4UZ < kBytes)) << std::format("the two orders agreed on {} of {} bytes, which should be only the bit-palindromes", agree, kBytes);
    };

    // Configuration is the only thing that can fail, and it fails before a sample moves.
    "validation rejects, and leaves the configuration intact"_test = [] {
        for (const std::string_view bad : {"", "0,3", "1,65", "3,3", "1,3;5", "1,3,", ",1", "1,,3", "  "}) {
            expect(throws<std::invalid_argument>([bad] { [[maybe_unused]] const TapMask ignored = tapsFromDelayList(bad); })) << std::format("taps '{}'", bad);
        }
        expect(eq(tapsFromDelayList(" 8 , 5 ,3, 1 "), tapsFromDelayList("1,3,5,8"))) << "spaces and order do not change a delay set";
        expect(eq(tapsFromDelayList("64"), TapMask{1} << 63U)) << "the widest register the kernel offers";

        expect(throws<std::invalid_argument>([] { [[maybe_unused]] const std::uint64_t ignored = seedFromBitString("1111", 8U); })) << "a seed shorter than the degree";
        expect(throws<std::invalid_argument>([] { [[maybe_unused]] const std::uint64_t ignored = seedFromBitString("111111111", 8U); })) << "and one longer";
        expect(throws<std::invalid_argument>([] { [[maybe_unused]] const std::uint64_t ignored = seedFromBitString("1111211", 7U); })) << "and one that is not bits";
        expect(eq(seedFromBitString("10000000", 8U), std::uint64_t{0x80})) << "the leftmost character is the most significant of the degree significant bits";

        ScramblerConfig       cfg    = additive("1,3,5,8", "11111111");
        const ScramblerConfig before = cfg;
        const TapMask         ccsds  = tapsFromDelayList("1,3,5,8");

        expect(throws<std::invalid_argument>([&cfg] { configureScrambler(cfg, 0ULL, 0ULL, ScramblerMode::Additive, 8U, BitOrder::MsbFirst); })) << "an empty tap set";
        for (const unsigned bad : {0U, 9U, 16U}) {
            expect(throws<std::invalid_argument>([&cfg, bad] { configureScrambler(cfg, ccsds, 0xFFULL, ScramblerMode::Additive, bad, BitOrder::MsbFirst); })) << std::format("bitsPerItem = {}", bad);
        }
        expect(throws<std::invalid_argument>([&cfg] { configureScrambler(cfg, ccsds, 0x100ULL, ScramblerMode::Additive, 8U, BitOrder::MsbFirst); })) << "a seed carrying bits above the degree";

        expect(eq(cfg.taps, before.taps)) << "a rejected setting leaves the previous configuration standing";
        expect(eq(cfg.seed, before.seed));
        expect(eq(static_cast<unsigned>(cfg.bitsPerItem), 8U));
        expect(eq(cfg.tablePeriod, before.tablePeriod));
        expect(eq(cfg.reg, before.reg));
        expect(eq(hexOf(sequenceItems(cfg, 8UZ)), std::string{"FF 48 0E C0 9A 0D 70 BC"})) << "and it still produces its sequence";

        bool quoted = false;
        try {
            [[maybe_unused]] const TapMask ignored = tapsFromDelayList("1,65");
        } catch (const std::invalid_argument& error) {
            quoted = std::string(error.what()).find("'1,65'") != std::string::npos;
        }
        expect(quoted) << "the offending value is quoted back in the message";

        // an all-zero seed is a dead generator, not an error: it is a legitimate history for the two
        // multiplicative modes, and refusing it belongs to whatever declares the setting
        for (const ScramblerMode mode : kModes) {
            expect(nothrow([mode] {
                ScramblerConfig zero{};
                configureScrambler(zero, ccsds, 0ULL, mode, 8U, BitOrder::MsbFirst);
            }));
        }
    };

    // A scrambler is a bijection on the bit stream: there is no value an item can carry that it rejects.
    "a stream value cannot stop the kernel"_test = [] {
        constexpr std::size_t           kItems = 100000UZ;
        const std::vector<std::uint8_t> loud(kItems, std::uint8_t{0xFF});
        const std::vector<std::uint8_t> quiet(kItems, std::uint8_t{0x07});

        for (const ScramblerMode mode : kModes) {
            for (const BitOrder order : kOrders) {
                ScramblerConfig cfg      = configured("1,3,5,8", 0xFFULL, mode, 3U, order);
                const auto      fromLoud = through(cfg, loud);
                resetScrambler(cfg);
                const auto fromQuiet = through(cfg, quiet);

                expect(eq(firstDifference(fromLoud, fromQuiet), kItems)) << "all 0xFF and all 0x07 are the same three-bit stream";
                expect(std::ranges::all_of(fromLoud, [](std::uint8_t value) { return value < 8U; })) << "and nothing is written above the field";
            }
        }
    };

    // The shape the hot-path rule asks for, measured against the definition it replaces. Whichever form is
    //     fastest is then safe to keep, and the table-form threshold can be moved with evidence.
    "the fast forms agree with the definition, bit for bit and in state"_test = [] {
        std::mt19937_64 engine{0xB3EE1411636FBC2BULL};
        std::size_t     comparisons   = 0UZ;
        std::size_t     disagreements = 0UZ;

        for (const std::string_view taps : {"1,3,5,8", "4,7", "14,15", "18,23", "5,23", "39,58", "1,2,5,7"}) {
            const TapMask      mask   = tapsFromDelayList(taps);
            const std::uint8_t degree = degreeOf(mask);
            for (unsigned width = 1U; width <= 8U; ++width) {
                for (const BitOrder order : kOrders) {
                    for (const ScramblerMode mode : kModes) {
                        const auto          source = randomItems(1000UZ, width, engine());
                        const std::uint64_t seed   = engine() & maskOf(degree);

                        ScramblerConfig fast = configured(taps, seed, mode, width, order);
                        ScramblerConfig ref  = fast;
                        const auto      want = through(fast, source);

                        std::vector<std::uint8_t> got(source.size(), std::uint8_t{0xAA});
                        gr::digital::detail::scrambleReference(ref, source, got);
                        ++comparisons;
                        disagreements += (firstDifference(got, want) == want.size() && ref.reg == fast.reg && ref.phase == fast.phase) ? 0UZ : 1UZ;
                        expect(eq(firstDifference(got, want), want.size())) << std::format("taps {} at {} bits, mode {}: the fast form left the definition", taps, width, static_cast<int>(mode));

                        if (mode == ScramblerMode::Additive && fast.form == SequenceForm::Table) {
                            ScramblerConfig recurrence = configured(taps, seed, mode, width, order);
                            recurrence.form            = SequenceForm::Recurrence;
                            const auto other           = through(recurrence, source);
                            ++comparisons;
                            disagreements += (firstDifference(other, want) == want.size() && recurrence.reg == fast.reg && recurrence.phase == fast.phase) ? 0UZ : 1UZ;
                            expect(eq(firstDifference(other, want), want.size())) << std::format("taps {} at {} bits: the table and the recurrence disagree", taps, width);
                        }
                        if (mode == ScramblerMode::MultiplicativeDescramble && width == 8U) {
                            ScramblerConfig           items = configured(taps, seed, mode, width, order);
                            std::vector<std::uint8_t> other(source.size(), std::uint8_t{0xAA});
                            gr::digital::detail::descrambleItems(items, source, other);
                            ++comparisons;
                            disagreements += (firstDifference(other, want) == want.size() && items.reg == fast.reg) ? 0UZ : 1UZ;
                            expect(eq(firstDifference(other, want), want.size())) << std::format("taps {}: the windowed and the per-item descramblers disagree", taps);
                        }
                    }
                }
            }
        }
        expect(eq(disagreements, 0UZ));
        expect(that % (comparisons > 400UZ)) << std::format("{} form comparisons", comparisons);
    };

    // What configure derives, which is everything the sample path indexes.
    "what configure derives, and the delay-list spelling"_test = [] {
        struct Derived {
            std::string_view taps;
            unsigned         degree;
            unsigned         minDelay;
            unsigned         tapCount;
            unsigned         stepBits;
            std::size_t      tableBytes; // zero when the recurrence form applies
        };
        constexpr Derived kDerived[] = {{"1,3,5,8", 8U, 1U, 4U, 1U, 33UZ}, //
            {"4,7", 7U, 4U, 2U, 4U, 17UZ},                                 //
            {"14,15", 15U, 14U, 2U, 8U, 4097UZ},                           //
            {"1,3,12,16", 16U, 1U, 4U, 1U, 8193UZ},                        //
            {"18,23", 23U, 18U, 2U, 8U, 0UZ},                              //
            {"14,17", 17U, 14U, 2U, 8U, 0UZ},                              //
            {"5,23", 23U, 5U, 2U, 5U, 0UZ},                                //
            {"39,58", 58U, 39U, 2U, 8U, 0UZ}};

        for (const Derived& row : kDerived) {
            const TapMask   mask = tapsFromDelayList(row.taps);
            ScramblerConfig cfg{};
            configureScrambler(cfg, mask, 0x1ULL, ScramblerMode::Additive, 8U, BitOrder::MsbFirst);

            expect(eq(static_cast<unsigned>(degreeOf(mask)), row.degree)) << row.taps;
            expect(eq(static_cast<unsigned>(minDelayOf(mask)), row.minDelay)) << row.taps;
            expect(eq(static_cast<unsigned>(tapCountOf(mask)), row.tapCount)) << row.taps;
            expect(eq(static_cast<unsigned>(cfg.stepBits), row.stepBits)) << std::format("{}: bits per branch-free step", row.taps);
            expect(eq(cfg.table.size(), row.tableBytes)) << std::format("{}: the sequence table is bounded by construction", row.taps);
            expect(that % ((cfg.form == SequenceForm::Table) == (row.tableBytes != 0UZ))) << row.taps;
            expect(eq(delayListFromTaps(mask), std::string{row.taps})) << "the delay list round-trips through the mask";

            // a multiplicative configuration holds nothing but its register
            ScramblerConfig self{};
            configureScrambler(self, mask, 0x1ULL, ScramblerMode::MultiplicativeScramble, 8U, BitOrder::MsbFirst);
            expect(eq(self.table.size(), 0UZ)) << std::format("{}: only the additive mode has a sequence to tabulate", row.taps);
        }

        // the named constants are the readings the table records
        expect(eq(gr::digital::standard::ccsds131, tapsFromDelayList("1,3,5,8")));
        expect(eq(gr::digital::standard::dvb, tapsFromDelayList("14,15")));
        expect(eq(gr::digital::standard::ieee80211, tapsFromDelayList("4,7")));
        expect(eq(gr::digital::standard::itu_18_23, tapsFromDelayList("18,23")));
        expect(eq(gr::digital::standard::itu_5_23, tapsFromDelayList("5,23")));
        expect(eq(gr::digital::standard::itu_14_17, tapsFromDelayList("14,17")));
        expect(eq(gr::digital::standard::ieee8023_64b66b, tapsFromDelayList("39,58")));
        expect(eq(static_cast<unsigned>(degreeOf(gr::digital::standard::ieee8023_64b66b)), 58U)) << "the widest named entry, and the reason degree runs to 64";
    };

    // What a scrambler does to a stream, and the two inputs that defeat one. Neither is a defect, and a
    //     scrambler is not a cipher.
    "whitening, the dead seed, and the two inputs that defeat a scrambler"_test = [] {
        constexpr std::size_t kBits = 100000UZ;

        const auto  ccsdsBits = sequenceBits("1,3,5,8", "11111111", kBits);
        std::size_t ones      = 0UZ;
        for (const std::uint8_t bit : ccsdsBits) {
            ones += bit;
        }
        expect(that % (ones > kBits * 49UZ / 100UZ && ones < kBits * 51UZ / 100UZ)) << std::format("100000 bits of all-zero input came out {} ones", ones);
        expect(eq(longestRun(ccsdsBits, std::uint8_t{1}, kBits - 64UZ), 8UZ)) << "and the longest run is the degree, whatever the input was";

        // a zero seed is a dead generator, which is why an additive block refuses one
        for (const Entry& entry : kAdditive) {
            const std::uint8_t degree = degreeOf(tapsFromDelayList(entry.taps));
            ScramblerConfig    dead   = configured(entry.taps, 0ULL, ScramblerMode::Additive, 1U, BitOrder::MsbFirst);
            const auto         bits   = sequenceItems(dead, 64UZ);
            expect(std::ranges::all_of(bits, [](std::uint8_t bit) { return bit == 0U; })) << std::format("{} at degree {} generates nothing from an all-zero seed", entry.name, degree);
        }

        // the one input an additive scrambler cannot whiten is its own sequence, which needs the seed and the phase
        ScramblerConfig ccsds  = additive("1,3,5,8", "11111111", 1U);
        const auto      folded = through(ccsds, std::span<const std::uint8_t>{ccsdsBits}.first(20000UZ));
        expect(std::ranges::all_of(folded, [](std::uint8_t bit) { return bit == 0U; })) << "feeding the generator its own sequence gives zeros";

        // the multiplicative equivalent needs neither: a descrambler run over zeros produces exactly the input
        // that holds a scrambler's output at a constant, from any seed
        for (const std::string_view taps : {"18,23", "4,7"}) {
            const std::uint8_t                 degree = degreeOf(tapsFromDelayList(taps));
            const std::array<std::uint64_t, 2> seeds{std::uint64_t{0}, std::uint64_t{0x71574E69A458FEA3} & maskOf(degree)};
            for (const std::uint64_t seed : seeds) {
                ScramblerConfig                 shape = configured(taps, seed, ScramblerMode::MultiplicativeDescramble, 1U, BitOrder::MsbFirst);
                const std::vector<std::uint8_t> zeros(20000UZ, std::uint8_t{0});
                const auto                      crafted = through(shape, zeros);

                ScramblerConfig tx  = configured(taps, seed, ScramblerMode::MultiplicativeScramble, 1U, BitOrder::MsbFirst);
                const auto      out = through(tx, crafted);
                expect(std::ranges::all_of(out, [](std::uint8_t bit) { return bit == 0U; })) << std::format("taps {}: a crafted input holds the output at a constant, and no run-length monitor is offered", taps);
            }
        }
    };

    // The profile table's openings, every one of them regenerated here from the recursion rather than compared
    //     against a value the test also fed in.
    "every profile regenerates the opening its document publishes"_test = [] {
        struct Variant {
            std::string_view what;
            std::string_view taps;
            BitOrder         order;
            std::string_view opening;
        };
        // pn9's three choices are confirmed together by one string: taps, seed and bit order each have an alternative
        // here, and each alternative produces its own bytes and not the vendor's.
        constexpr Variant kPn9[] = {{"{4,9} lsb_first, the profile", "4,9", BitOrder::LsbFirst, "FF E1 1D 9A ED 85 33 24"}, //
            {"{4,9} msb_first", "4,9", BitOrder::MsbFirst, "FF 87 B8 59 B7 A1 CC 24"},                                      //
            {"{5,9} lsb_first, the reciprocal reading", "5,9", BitOrder::LsbFirst, "FF C1 FB E8 4C 90 72 8B"},              //
            {"{5,9} msb_first", "5,9", BitOrder::MsbFirst, "FF 83 DF 17 32 09 4E D1"}};

        std::vector<std::string> openings;
        for (const Variant& variant : kPn9) {
            ScramblerConfig cfg  = additive(variant.taps, "111111111", 8U, variant.order);
            const auto      seen = hexOf(sequenceItems(cfg, 8UZ));
            expect(eq(seen, std::string{variant.opening})) << variant.what;
            openings.push_back(seen);
        }
        std::ranges::sort(openings);
        expect(eq(std::ranges::unique(openings).size(), 0UZ)) << "the four readings are four different sequences, so the check value pins all three choices at once";

        ScramblerConfig pn9 = additive("4,9", "111111111", 8U, BitOrder::LsbFirst);
        expect(eq(hexOf(sequenceItems(pn9, 16UZ)), std::string{"FF E1 1D 9A ED 85 33 24 EA 7A D2 39 70 97 57 0A"})) << "the sixteen bytes the CC11xx and SX12xx data sheets describe";

        // si4463: pn9's recursion at the phase the chip's register model emits — all effective stages seeded ones,
        // the Galois form of the documented tap mask, output at bit 0
        ScramblerConfig si4463 = additive("4,9", "111100001", 8U, BitOrder::LsbFirst);
        expect(eq(hexOf(sequenceItems(si4463, 8UZ)), std::string{"0F EF D0 6C 2F 9C 21 51"})) << "pn9's sequence advanced five steps, not pn9's own phase";

        // g3ruh: nothing offline separates the two readings, so both are recorded and a swapped one is visible
        ScramblerConfig delayReading   = additive("12,17", "11111111111111111");
        ScramblerConfig characteristic = additive("5,17", "11111111111111111");
        expect(eq(hexOf(sequenceItems(delayReading, 8UZ)), std::string{"FF FF 80 07 C0 7F E7 C1"})) << "1 + X^12 + X^17, the recursion the modem description states";
        expect(eq(hexOf(sequenceItems(characteristic, 8UZ)), std::string{"FF FF 83 E0 C7 CE 13 7C"})) << "x^17 + x^12 + 1 read as a characteristic polynomial";

        // CCSDS 131.0-B-5 10.4.1's degree-17 primary: the first forty bits are 10.4.3 note 2's published value,
        // which pins the characteristic reading and the time-order seed together; a swapped reading parts from it
        // at the third byte.
        ScramblerConfig primary    = additive("3,17", "00011100011100011");
        ScramblerConfig reciprocal = additive("14,17", "00011100011100011");
        const auto      primaryHex = hexOf(sequenceItems(primary, 8UZ));
        expect(eq(primaryHex, std::string{"1C 71 B9 1B A9 BA 84 57"})) << "the first five bytes are the forty bits the standard publishes";
        expect(eq(hexOf(sequenceItems(reciprocal, 8UZ)), std::string{"1C 71 FF FF 00 03 80 0F"})) << "the reciprocal reading, parting at the third byte";

        // and the legacy degree-8 sequence is exactly where it was
        ScramblerConfig legacy = additive("1,3,5,8", "11111111");
        expect(eq(hexOf(sequenceItems(legacy, 5UZ)), std::string{"FF 48 0E C0 9A"})) << "10.4.2's legacy sequence, unmoved by the arrival of the primary";
    };

    // The period is the closed form, walked and not stored.
    "every profile's period is 2^degree - 1"_test = [] {
        struct Row {
            std::string_view name;
            std::string_view taps;
            std::string_view seed;
            std::size_t      period;
        };
        constexpr Row kRows[] = {{"ccsds131", "1,3,5,8", "11111111", 255UZ}, //
            {"ccsds131_17", "3,17", "00011100011100011", 131071UZ},          //
            {"pn9", "4,9", "111111111", 511UZ},                              //
            {"si4463", "4,9", "111100001", 511UZ},                           //
            {"g3ruh", "12,17", "11111111111111111", 131071UZ}};

        for (const Row& row : kRows) {
            const std::uint8_t degree = degreeOf(tapsFromDelayList(row.taps));
            expect(that % (maskOf(degree) == row.period)) << std::format("{}: the period must be 2^{} - 1", row.name, degree);

            const auto  bits    = sequenceBits(row.taps, row.seed, row.period + 128UZ);
            std::size_t offside = 0UZ;
            for (std::size_t i = 0UZ; i < 128UZ; ++i) {
                offside += bits[i] == bits[i + row.period] ? 0UZ : 1UZ;
            }
            expect(eq(offside, 0UZ)) << std::format("{}: the sequence repeats at its period", row.name);
            for (const std::size_t prime : primeDivisors(row.period)) {
                const std::size_t shorter = row.period / prime;
                bool              differs = false;
                for (std::size_t i = 0UZ; i < 128UZ && !differs; ++i) {
                    differs = bits[i] != bits[i + shorter];
                }
                expect(differs) << std::format("{}: {} is not a period, so the period is exactly the closed form", row.name, shorter);
            }
        }
    };

    // A maximal-length recursion has one period, not one per seed, so every seed names a phase of one
    // sequence — and si4463's own seed is the phase five steps on from pn9's.
    "every seed of the pn9 recursion names a phase of one sequence"_test = [] {
        constexpr std::size_t kPeriod = 511UZ;

        const auto  reference = sequenceBits("4,9", "111111111", 2UZ * kPeriod);
        std::size_t checked   = 0UZ;
        std::size_t shifted   = 0UZ;
        for (std::uint64_t seed = 1ULL; seed <= kPeriod; ++seed) {
            ScramblerConfig cfg  = configured("4,9", seed, ScramblerMode::Additive, 1U, BitOrder::MsbFirst);
            const auto      bits = sequenceItems(cfg, kPeriod);

            bool found = false;
            for (std::size_t shift = 0UZ; shift < kPeriod && !found; ++shift) {
                found = true;
                for (std::size_t i = 0UZ; i < kPeriod && found; ++i) {
                    found = bits[i] == reference[i + shift];
                }
            }
            ++checked;
            shifted += found ? 1UZ : 0UZ;
        }
        expect(eq(checked, kPeriod));
        expect(eq(shifted, kPeriod)) << "every non-zero seed of {4,9} is a cyclic shift of the same 511 bits";

        // and the profile's phase in particular: its seed emits pn9's sequence five steps on
        const auto chip    = sequenceBits("4,9", "111100001", kPeriod);
        bool       matches = true;
        for (std::size_t i = 0UZ; i < kPeriod && matches; ++i) {
            matches = chip[i] == reference[i + 5UZ];
        }
        expect(matches) << "si4463's phase is pn9's sequence advanced five steps";
        expect(eq(profileByName("si4463").taps, tapsFromDelayList("4,9")));
        expect(that % profileByName("si4463").unrecorded.empty()) << "the profile's constants are recorded, so it applies";
    };

    // Scramble then descramble is the identity, per profile and per family.
    "every profile round-trips, and g3ruh does so from a wrong seed"_test = [] {
        constexpr std::size_t kItems = 100000UZ;

        for (const std::string_view name : {"ccsds131", "ccsds131_17", "pn9", "si4463"}) {
            const auto& entry = profileByName(name);
            for (const unsigned width : {1U, 8U}) {
                ScramblerConfig cfg{};
                configureScrambler(cfg, entry.taps, entry.seed, ScramblerMode::Additive, width, entry.order);

                const auto source = randomItems(kItems, width, 0x9E3779B97F4A7C15ULL + width);
                const auto once   = through(cfg, source);
                resetScrambler(cfg);
                const auto twice = through(cfg, once);
                expect(eq(firstDifference(twice, source), source.size())) << std::format("{} at {} bits per item", name, width);
            }
        }

        // the multiplicative profile: correct from item `degree` whatever the descrambler started from, which is the
        // self-synchronizing property g3ruh rests on
        const auto  source = randomItems(20000UZ, 1U, 0xC2B2AE3D27D4EB4FULL);
        const auto& g3ruh  = profileByName("g3ruh");
        for (const std::uint64_t seed : {std::uint64_t{0}, std::uint64_t{0x1FFFF}, std::uint64_t{0x0ACE1}}) {
            ScramblerConfig tx{};
            configureScrambler(tx, g3ruh.taps, g3ruh.seed, ScramblerMode::MultiplicativeScramble, 1U, g3ruh.order);

            ScramblerConfig rx = configured("12,17", seed, ScramblerMode::MultiplicativeDescramble, 1U, BitOrder::MsbFirst);
            const auto      y  = through(tx, source);
            const auto      x  = through(rx, y);

            std::size_t bad = 0UZ;
            for (std::size_t i = 17UZ; i < source.size(); ++i) {
                bad += x[i] != source[i] ? 1UZ : 0UZ;
            }
            expect(eq(bad, 0UZ)) << std::format("g3ruh converges in 17 bits from seed {:#x}", seed);
        }
    };

    // An explicit sequence is the same mask from a different source, and what it cannot cover it counts.
    "the explicit sequence agrees with the recursion, and counts what it cannot cover"_test = [] {
        constexpr std::size_t kPeriod = 511UZ;

        // (a) one period of pn9, supplied instead of walked, over ten periods
        const auto      onePeriod = sequenceBits("4,9", "111111111", kPeriod);
        const auto      packed    = packMsbFirst(onePeriod);
        ScramblerConfig walked    = ScramblerConfig{};
        applyProfile(walked, "pn9", ScramblerMode::Additive);
        ScramblerConfig supplied = ScramblerConfig{};
        configureExplicit(supplied, packed, kPeriod, 8U, BitOrder::LsbFirst, true);

        const auto source = randomItems(kPeriod * 10UZ / 8UZ, 8U, 0x452821E638D01377ULL);
        expect(eq(firstDifference(through(walked, source), through(supplied, source)), source.size())) << "the two mask sources agree bit for bit over ten periods";
        expect(that % (supplied.source == SequenceSource::Explicit && supplied.taps == 0ULL && supplied.tablePeriod == kPeriod)) << "an explicit sequence is the table form with the table handed over";

        // (b) the counted refusal, as a number
        constexpr std::size_t kOverrun = 37UZ;
        ScramblerConfig       once     = ScramblerConfig{};
        configureExplicit(once, packed, kPeriod, 1U, BitOrder::MsbFirst, false);
        const std::vector<std::uint8_t> ones(kPeriod + kOverrun, std::uint8_t{1});
        const auto                      covered = through(once, ones);
        expect(eq(once.nUnscrambledItems, kOverrun)) << "every item the sequence could not reach is counted";

        std::size_t wrong = 0UZ;
        for (std::size_t i = 0UZ; i < covered.size(); ++i) {
            const unsigned mask = i < kPeriod ? onePeriod[i] : 0U;
            wrong += covered[i] == (1U ^ mask) ? 0UZ : 1UZ;
        }
        expect(eq(wrong, 0UZ)) << "what the sequence reached it scrambled, and what it did not reach passed through";

        // repeating instead covers the whole epoch and counts nothing
        ScramblerConfig tiled = ScramblerConfig{};
        configureExplicit(tiled, packed, kPeriod, 1U, BitOrder::MsbFirst, true);
        const auto whole = through(tiled, ones);
        expect(eq(tiled.nUnscrambledItems, 0UZ)) << "the tiling source leaves nothing uncovered";
        expect(that % (firstDifference(whole, covered) == kPeriod)) << "and the two sources part exactly where the sequence ran out";

        // (d) the spelling and its refusals
        expect(eq(hexOf(sequenceFromHex("FF E1 1D 9A")), std::string{"FF E1 1D 9A"}));
        expect(eq(hexOf(sequenceFromHex("ffe11d9a")), std::string{"FF E1 1D 9A"})) << "either case, spaces optional";
        expect(throws<std::invalid_argument>([] { [[maybe_unused]] const auto ignored = sequenceFromHex("FF E"); })) << "an odd number of digits";
        expect(throws<std::invalid_argument>([] { [[maybe_unused]] const auto ignored = sequenceFromHex("FF ZZ"); })) << "a character that is not a hexadecimal digit";
        expect(throws<std::invalid_argument>([] { [[maybe_unused]] const auto ignored = sequenceFromHex("   "); })) << "an empty sequence";
        expect(throws<std::invalid_argument>([&packed] {
            ScramblerConfig cfg{};
            configureExplicit(cfg, packed, 0UZ, 8U, BitOrder::MsbFirst, true);
        })) << "a zero-length mask";
        expect(throws<std::invalid_argument>([&packed] {
            ScramblerConfig cfg{};
            configureExplicit(cfg, packed, (1UZ << 20) + 1UZ, 8U, BitOrder::MsbFirst, true);
        })) << "a mask above 2^20 bits";
        expect(throws<std::invalid_argument>([&packed] {
            ScramblerConfig cfg{};
            configureExplicit(cfg, packed, 9U * 1024UZ, 8U, BitOrder::MsbFirst, true);
        })) << "more bits claimed than supplied";
        expect(throws<std::invalid_argument>([&packed] {
            ScramblerConfig cfg{};
            configureExplicit(cfg, packed, 511UZ, 9U, BitOrder::MsbFirst, true);
        })) << "an item width outside [1, 8]";

        // configuring a tap set afterwards puts the mask source back, so nothing of the explicit configuration leaks
        ScramblerConfig reused = supplied;
        configureScrambler(reused, tapsFromDelayList("4,9"), seedFromBitString("111111111", 9U), ScramblerMode::Additive, 8U, BitOrder::LsbFirst);
        expect(that % (reused.source == SequenceSource::Lfsr && reused.nUnscrambledItems == 0UZ)) << "a reconfiguration is whole";
    };

    // The forced-transition rule is invertible, and the two counters agree item by item because the monitor
    // reads only the transmitted stream.
    "the forced-transition rule inverts exactly, and both ends compute the same counter"_test = [] {
        constexpr std::size_t  kTrials = 400UZ;
        constexpr std::size_t  kBits   = 4000UZ;
        constexpr std::uint8_t kDegree = 20U;

        std::size_t cases    = 0UZ;
        std::size_t failures = 0UZ;
        for (const std::uint32_t modulus : {8U, 16U, 32U}) {
            for (std::size_t trial = 0UZ; trial < kTrials; ++trial) {
                const std::uint64_t seed   = (0x243F6A8885A308D3ULL * (trial + 1UZ)) & maskOf(kDegree);
                const auto          source = randomItems(kBits, 1U, 0xB5026F5AA96619E9ULL + trial);

                ScramblerConfig tx = configured("3,20", seed, ScramblerMode::MultiplicativeScramble, 1U, BitOrder::MsbFirst);
                ScramblerConfig rx = configured("3,20", seed, ScramblerMode::MultiplicativeDescramble, 1U, BitOrder::MsbFirst);
                configureMonitor(tx, true, modulus, "1,2");
                configureMonitor(rx, true, modulus, "1,2");

                const auto y = through(tx, source);
                const auto x = through(rx, y);

                std::size_t bad = 0UZ;
                for (std::size_t i = kDegree; i < kBits; ++i) {
                    bad += x[i] != source[i] ? 1UZ : 0UZ;
                }
                ++cases;
                failures += bad == 0UZ && tx.nForcedTransitions == rx.nForcedTransitions ? 0UZ : 1UZ;
            }
        }
        expect(eq(cases, 3UZ * kTrials));
        expect(eq(failures, 0UZ)) << "descramble o scramble is the identity from item degree at every N and seed";

        // the counter itself, compared at every item rather than at the end of the run
        for (const std::uint32_t modulus : {8U, 32U}) {
            const auto      source = randomItems(4000UZ, 1U, 0x2545F4914F6CDD1DULL + modulus);
            ScramblerConfig tx     = configured("3,20", 0x5A5A5ULL, ScramblerMode::MultiplicativeScramble, 1U, BitOrder::MsbFirst);
            ScramblerConfig rx     = configured("3,20", 0x0F0F0ULL, ScramblerMode::MultiplicativeDescramble, 1U, BitOrder::MsbFirst);
            configureMonitor(tx, true, modulus, "1,2");
            configureMonitor(rx, true, modulus, "1,2");

            std::size_t apart = 0UZ;
            for (std::size_t i = 0UZ; i < source.size(); ++i) {
                std::array<std::uint8_t, 1> sent{};
                std::array<std::uint8_t, 1> back{};
                scramble(tx, std::span<const std::uint8_t>{source}.subspan(i, 1UZ), sent);
                scramble(rx, std::span<const std::uint8_t>{sent}, back);
                apart += tx.forceCounter == rx.forceCounter ? 0UZ : 1UZ;
            }
            expect(eq(apart, 0UZ)) << std::format("N = {}: the two counters agree at every item, from seeds that do not", modulus);
        }
    };

    // The firing bound is combinatorial, so it is an equality and not a tolerance, and the reference count is
    // computed from the emitted stream alone.
    "no window of N items runs with the comparison zero and no firing"_test = [] {
        constexpr std::size_t kBits = 1000000UZ;

        for (const std::uint32_t modulus : {8U, 32U}) {
            const auto      source = randomItems(kBits, 1U, 0x8AED2A6ABF715880ULL + modulus);
            ScramblerConfig tx     = configured("3,20", 0x13579ULL, ScramblerMode::MultiplicativeScramble, 1U, BitOrder::MsbFirst);
            configureMonitor(tx, true, modulus, "1,2");
            const auto y = through(tx, source);

            const Replay replay = replayForcing(y, modulus, 1U, 2U);
            expect(eq(replay.firings, tx.nForcedTransitions)) << std::format("N = {}: the count a reference loop reads off the emitted stream", modulus);

            std::size_t violations = 0UZ;
            std::size_t run        = 0UZ;
            for (std::size_t k = 0UZ; k < y.size(); ++k) {
                run = replay.fired[k] != 0U ? 0UZ : (replay.differs[k] != 0U ? 0UZ : run + 1UZ);
                violations += run >= static_cast<std::size_t>(modulus) ? 1UZ : 0UZ;
            }
            expect(eq(violations, 0UZ)) << std::format("N = {}: no run of {} bits with the comparison zero throughout carries no firing", modulus, modulus);

            // What the rate is, rather than only that the bound holds. A scrambled stream looks random, so the
            // comparison is zero at a bit with probability 1/2 and the counter reaches N - 1 about once in 2^N bits:
            // 3906 expected over a million at N = 8, and 0.0002 at N = 32. The rule is a lock-up destroyer and not a
            // transition source, and a chain whose count is zero is a chain that could have disabled it.
            const double expected = static_cast<double>(kBits) / static_cast<double>(1ULL << modulus);
            std::println("forced transitions at N = {:2}: {} over {} bits, against {:.4f} expected of a random stream", modulus, replay.firings, kBits, expected);
            if (modulus == 8U) {
                expect(that % (replay.firings > kBits / 8192UZ)) << "at N = 8 the rule fires often enough that its count is evidence the chain reached it";
            }
        }
    };

    // The reason the rule exists, put as the strict comparison against the same chain without it.
    "the rule destroys the plain scrambler's lock-up input"_test = [] {
        constexpr std::size_t kBits = 4000UZ;

        for (const std::uint32_t modulus : {8U, 16U, 32U}) {
            for (const std::uint64_t seed : {std::uint64_t{0}, std::uint64_t{0x13579}, std::uint64_t{0xFFFFF}}) {
                const auto crafted = lockUpInput("3,20", seed, true, 0U, kBits);

                ScramblerConfig plain = configured("3,20", seed, ScramblerMode::MultiplicativeScramble, 1U, BitOrder::MsbFirst);
                configureMonitor(plain, true, 0U, "");
                const auto held = through(plain, crafted);
                expect(eq(leadingRun(held), kBits)) << std::format("seed {:#x}: the plain scrambler's output is constant for the whole run", seed);

                ScramblerConfig forced = configured("3,20", seed, ScramblerMode::MultiplicativeScramble, 1U, BitOrder::MsbFirst);
                configureMonitor(forced, true, modulus, "1,2");
                const auto broken = through(forced, crafted);
                expect(that % (leadingRun(broken) < kBits)) << std::format("N = {}, seed {:#x}: the forced scrambler's is not", modulus, seed);
                expect(eq(leadingRun(broken), static_cast<std::size_t>(modulus) - 1UZ)) << std::format("N = {}, seed {:#x}: and the constant run ends at the N-th item", modulus, seed);
                expect(that % (forced.nForcedTransitions >= 1UZ)) << "the forcing is what ended it";
            }
        }
    };

    // The bijection, pinned as a counterexample so no run-length guarantee is ever added on top of a rule that
    // cannot keep one. The forcing term is a function of the output alone, so the output can be anything.
    "the forced pair is still a bijection, constant output included"_test = [] {
        constexpr std::size_t   kBits  = 4000UZ;
        constexpr std::uint32_t kAfter = 32U;

        const auto crafted = lockUpInput("3,20", 0x13579ULL, true, kAfter, kBits);

        ScramblerConfig tx = configured("3,20", 0x13579ULL, ScramblerMode::MultiplicativeScramble, 1U, BitOrder::MsbFirst);
        configureMonitor(tx, true, kAfter, "1,2");
        const auto held = through(tx, crafted);
        expect(eq(leadingRun(held), kBits)) << "an input constructed against the rule holds the output at a constant for the whole run";
        expect(that % (tx.nForcedTransitions > 0UZ)) << "and the forcing fired throughout, which is what makes it a counterexample rather than a disabled rule";

        ScramblerConfig rx = configured("3,20", 0x13579ULL, ScramblerMode::MultiplicativeDescramble, 1U, BitOrder::MsbFirst);
        configureMonitor(rx, true, kAfter, "1,2");
        expect(eq(firstDifference(through(rx, held), crafted), crafted.size())) << "and it descrambles exactly, from item zero";
    };

    // What the resolver refuses, and that naming nothing changes nothing.
    "profile resolution, and the four refusals"_test = [] {
        expect(eq(kProfiles.size(), 5UZ)) << "ccsds131, ccsds131_17, pn9, si4463, g3ruh";

        ScramblerConfig cfg = ScramblerConfig{};
        applyProfile(cfg, "pn9", ScramblerMode::Additive);
        expect(that % (cfg.taps == tapsFromDelayList("4,9") && cfg.seed == 511ULL && cfg.bitsPerItem == 8U && cfg.order == BitOrder::LsbFirst)) << "a profile fills the four settings the generator is built from";
        expect(that % (cfg.forceAfter == 0U && !cfg.invertOutput && cfg.sequenceRepeat)) << "and names no other setting";

        expect(throws<std::invalid_argument>([] {
            ScramblerConfig c{};
            applyProfile(c, "g3ruh", ScramblerMode::Additive);
        })) << "a self-synchronizing agreement run additively is a silent wrong answer";
        expect(throws<std::invalid_argument>([] {
            ScramblerConfig c{};
            applyProfile(c, "pn9", ScramblerMode::MultiplicativeScramble);
        })) << "and an additive one run self-synchronizingly likewise";
        expect(throws<std::invalid_argument>([] {
            ScramblerConfig c{};
            applyProfile(c, "pn9", ScramblerMode::MultiplicativeDescramble);
        }));
        expect(nothrow([] {
            ScramblerConfig c{};
            applyProfile(c, "si4463", ScramblerMode::Additive);
        })) << "a profile with recorded constants applies";
        expect(throws<std::invalid_argument>([] { [[maybe_unused]] const auto& ignored = profileByName("ccsds"); })) << "an unknown name lists the ones there are";
        expect(throws<std::invalid_argument>([] { [[maybe_unused]] const auto& ignored = profileByName(""); })) << "and so does the empty one, which is how a caller selects nothing";

        // the forced-transition group's own refusals
        ScramblerConfig pair = configured("3,20", 0ULL, ScramblerMode::MultiplicativeScramble, 1U, BitOrder::MsbFirst);
        expect(throws<std::invalid_argument>([&pair] { configureMonitor(pair, false, 1U, "1,2"); })) << "N below 2";
        expect(throws<std::invalid_argument>([&pair] { configureMonitor(pair, false, 4097U, "1,2"); })) << "N above 4096";
        expect(throws<std::invalid_argument>([&pair] { configureMonitor(pair, false, 0U, "1,2"); })) << "delays describing a disabled mechanism";
        expect(throws<std::invalid_argument>([&pair] { configureMonitor(pair, false, 32U, "1"); })) << "one delay where two are needed";
        expect(throws<std::invalid_argument>([&pair] { configureMonitor(pair, false, 32U, "1,2,3"); })) << "three delays where two are needed";
        expect(throws<std::invalid_argument>([&pair] { configureMonitor(pair, false, 32U, "1,65"); })) << "a monitored delay above 64";

        ScramblerConfig additiveCfg = configured("4,9", 511ULL, ScramblerMode::Additive, 8U, BitOrder::LsbFirst);
        expect(throws<std::invalid_argument>([&additiveCfg] { configureMonitor(additiveCfg, false, 32U, "1,2"); })) << "the rule belongs to the family that has feedback for it to act on";
        expect(throws<std::invalid_argument>([&additiveCfg] { configureMonitor(additiveCfg, true, 0U, ""); })) << "and so does the inversion";

        // with the group off, the multiplicative pair is bit for bit what it was before the rule existed
        const auto      source = randomItems(8192UZ, 8U, 0xA54FF53A5F1D36F1ULL);
        ScramblerConfig plain  = configured("18,23", 0x7FFFFFULL, ScramblerMode::MultiplicativeScramble, 8U, BitOrder::MsbFirst);
        const auto      before = through(plain, source);
        resetScrambler(plain);
        configureMonitor(plain, false, 0U, "");
        expect(eq(firstDifference(through(plain, source), before), source.size())) << "naming the disabled group changes no bit";
    };
};

int main() { /* tests are automatically registered and run */ }
