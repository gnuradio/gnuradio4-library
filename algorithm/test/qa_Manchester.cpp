#include <boost/ut.hpp>

#include <magic_enum.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/algorithm/digital/BitPacking.hpp>
#include <gnuradio-4.0/algorithm/digital/Manchester.hpp>

namespace {

using gr::digital::BitOrder;
using gr::digital::BitRepack;
using gr::digital::configureManchester;
using gr::digital::configureRepack;
using gr::digital::conventionFromName;
using gr::digital::conventionName;
using gr::digital::conventionXor;
using gr::digital::decodeManchester;
using gr::digital::encodeManchester;
using gr::digital::ManchesterConfig;
using gr::digital::ManchesterConvention;
using gr::digital::realignManchester;
using gr::digital::repack;
using gr::digital::resetManchester;

constexpr BitOrder kOrders[] = {BitOrder::MsbFirst, BitOrder::LsbFirst};

constexpr ManchesterConvention kConventions[] = {ManchesterConvention::Ieee8023, ManchesterConvention::GeThomas};

[[nodiscard]] ManchesterConfig configured(ManchesterConvention convention, unsigned width = 8U, BitOrder order = BitOrder::MsbFirst, unsigned phase = 0U) {
    ManchesterConfig cfg{};
    configureManchester(cfg, convention, width, order, phase);
    return cfg;
}

[[nodiscard]] std::vector<std::uint8_t> encoded(const ManchesterConfig& cfg, std::span<const std::uint8_t> in) {
    std::vector<std::uint8_t> out(2UZ * in.size(), std::uint8_t{0xEE});
    encodeManchester(cfg, in, out);
    return out;
}

/// The decoded items and the violation flags, which always come in step and are always the same length.
struct Decoded {
    std::vector<std::uint8_t> items;
    std::vector<std::uint8_t> flags;
};

[[nodiscard]] Decoded decoded(ManchesterConfig& cfg, std::span<const std::uint8_t> in) {
    Decoded result{std::vector<std::uint8_t>(in.size() / 2UZ, std::uint8_t{0xEE}), std::vector<std::uint8_t>(in.size() / 2UZ, std::uint8_t{0xEE})};
    decodeManchester(cfg, in, result.items, result.flags);
    return result;
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

/// The chip or bit stream of @p items, one to an element, which is how a property about positions is stated.
[[nodiscard]] std::vector<std::uint8_t> streamOf(std::span<const std::uint8_t> items, unsigned width, BitOrder order) {
    std::vector<std::uint8_t> stream;
    stream.reserve(items.size() * static_cast<std::size_t>(width));
    for (const std::uint8_t item : items) {
        for (unsigned i = 0U; i < width; ++i) {
            const unsigned position = order == BitOrder::MsbFirst ? width - 1U - i : i;
            stream.push_back(static_cast<std::uint8_t>((static_cast<unsigned>(item) >> position) & 1U));
        }
    }
    return stream;
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

[[nodiscard]] std::size_t countOnes(std::span<const std::uint8_t> stream) {
    std::size_t ones = 0UZ;
    for (const std::uint8_t bit : stream) {
        ones += bit;
    }
    return ones;
}

/// @p items decoded in chunks of @p chunk input items, which must leave the same trail as one call.
[[nodiscard]] Decoded decodedInChunks(ManchesterConfig& cfg, std::span<const std::uint8_t> in, std::size_t chunk) {
    Decoded result{std::vector<std::uint8_t>(in.size() / 2UZ, std::uint8_t{0xEE}), std::vector<std::uint8_t>(in.size() / 2UZ, std::uint8_t{0xEE})};
    for (std::size_t index = 0UZ; index < in.size(); index += chunk) {
        const std::size_t take = std::min(chunk, in.size() - index);
        decodeManchester(cfg, in.subspan(index, take), std::span<std::uint8_t>{result.items}.subspan(index / 2UZ, take / 2UZ), std::span<std::uint8_t>{result.flags}.subspan(index / 2UZ, take / 2UZ));
    }
    return result;
}

/// The chip stream `(out, violation)` says it came from: `c0 = bit XOR k`, and `c1` repeats `c0` exactly where the flag is set.
[[nodiscard]] std::vector<std::uint8_t> chipsFrom(std::span<const std::uint8_t> bits, std::span<const std::uint8_t> flags, unsigned k) {
    std::vector<std::uint8_t> chips;
    chips.reserve(2UZ * bits.size());
    for (std::size_t i = 0UZ; i < bits.size(); ++i) {
        const unsigned head = static_cast<unsigned>(bits[i]) ^ k;
        chips.push_back(static_cast<std::uint8_t>(head));
        chips.push_back(static_cast<std::uint8_t>(flags[i] != 0U ? head : head ^ 1U));
    }
    return chips;
}

} // namespace

const boost::ut::suite<"manchester"> manchesterTests = [] {
    using namespace boost::ut;

    // The IEEE 802.3 column of the chip table, and the standard's own preamble and start-of-frame
    //    delimiter, whose two waveform properties are what the convention is checked against.
    "ieee802_3 reproduces its table and IEEE 802.3's preamble and delimiter"_test = [] {
        struct Row {
            std::uint8_t     item;
            std::string_view ieee;
            std::string_view thomas;
        };
        constexpr Row kTable[] = {{0x00U, "AA AA", "55 55"}, {0xFFU, "55 55", "AA AA"}, {0x0FU, "AA 55", "55 AA"}, {0x55U, "99 99", "66 66"}, {0xAAU, "66 66", "99 99"}, {0xA5U, "66 99", "99 66"}, {0x47U, "9A 95", "65 6A"}};

        const ManchesterConfig ieee = configured(ManchesterConvention::Ieee8023);
        for (const Row& row : kTable) {
            expect(eq(hexOf(encoded(ieee, std::span{&row.item, 1UZ})), std::string{row.ieee})) << std::format("{:02X} under ieee802_3", row.item);
        }
        expect(eq(static_cast<unsigned>(conventionXor(ManchesterConvention::Ieee8023)), 1U)) << "the standard's complement-then-true bit cell is k = 1";
        expect(eq(static_cast<unsigned>(ieee.k), 1U));

        // the preamble octet and the delimiter are transmitted least-significant-bit first
        const ManchesterConfig lsb      = configured(ManchesterConvention::Ieee8023, 8U, BitOrder::LsbFirst);
        const std::uint8_t     preamble = 0x55U;
        const std::uint8_t     sfd      = 0xD5U;
        expect(eq(hexOf(encoded(lsb, std::span{&preamble, 1UZ})), std::string{"66 66"})) << "the preamble octet 0x55";
        expect(eq(hexOf(encoded(lsb, std::span{&sfd, 1UZ})), std::string{"66 A6"})) << "the start-of-frame delimiter 0xD5";

        // the same two bit sequences read the other way round, which is the same chip stream repacked
        const std::uint8_t preambleMsb = 0xAAU;
        const std::uint8_t sfdMsb      = 0xABU;
        expect(eq(hexOf(encoded(ieee, std::span{&preambleMsb, 1UZ})), std::string{"66 66"}));
        expect(eq(hexOf(encoded(ieee, std::span{&sfdMsb, 1UZ})), std::string{"66 65"}));

        // property one: the preamble is a square wave with exactly one transition per bit time, so the
        // mid-bit transitions are all of them and no bit boundary carries one
        const auto  preambleChips = streamOf(encoded(lsb, std::span{&preamble, 1UZ}), 8U, BitOrder::LsbFirst);
        std::size_t transitions   = 0UZ;
        for (std::size_t i = 1UZ; i < preambleChips.size(); ++i) {
            transitions += preambleChips[i] != preambleChips[i - 1UZ] ? 1UZ : 0UZ;
        }
        expect(eq(transitions, 8UZ)) << "eight bit times, eight transitions";
        for (std::size_t j = 0UZ; j + 1UZ < 8UZ; ++j) {
            expect(eq(static_cast<unsigned>(preambleChips[2UZ * j + 1UZ]), static_cast<unsigned>(preambleChips[2UZ * j + 2UZ]))) << std::format("no transition at preamble bit boundary {}", j);
        }

        // property two: the delimiter breaks that square wave at exactly one place, the boundary
        // between its final two ones, because that is the only place two adjacent bits are equal
        const auto               sfdChips = streamOf(encoded(lsb, std::span{&sfd, 1UZ}), 8U, BitOrder::LsbFirst);
        std::vector<std::size_t> broken;
        for (std::size_t j = 0UZ; j + 1UZ < 8UZ; ++j) {
            if (sfdChips[2UZ * j + 1UZ] != sfdChips[2UZ * j + 2UZ]) {
                broken.push_back(j);
            }
        }
        expect(eq(broken.size(), 1UZ)) << "exactly one break";
        expect(eq(broken.front(), 6UZ)) << "and it is the boundary between the delimiter's final two ones";
    };

    // The other convention, anchored to the one standard that publishes the chip pair itself rather
    //    than a waveform, and the identity that makes the whole table checkable in one line.
    "ge_thomas reproduces MIL-STD-1553's own statement, and the columns are complements"_test = [] {
        const ManchesterConfig thomas = configured(ManchesterConvention::GeThomas);
        expect(eq(static_cast<unsigned>(conventionXor(ManchesterConvention::GeThomas)), 0U)) << "a logic one as 1/0 and a logic zero as 0/1 is k = 0";

        // the standard's sentence, reproduced from the chip map at one bit per item
        const ManchesterConfig bitwise = configured(ManchesterConvention::GeThomas, 1U);
        const std::uint8_t     one     = 1U;
        const std::uint8_t     zero    = 0U;
        expect(eq(hexOf(encoded(bitwise, std::span{&one, 1UZ})), std::string{"01 00"})) << "a logic one is a positive pulse followed by a negative one";
        expect(eq(hexOf(encoded(bitwise, std::span{&zero, 1UZ})), std::string{"00 01"})) << "and a logic zero is its inverse";

        struct Row {
            std::uint8_t     item;
            std::string_view thomas;
        };
        constexpr Row kTable[] = {{0x00U, "55 55"}, {0xFFU, "AA AA"}, {0x0FU, "55 AA"}, {0x55U, "66 66"}, {0xAAU, "99 99"}, {0xA5U, "99 66"}, {0x47U, "65 6A"}};
        for (const Row& row : kTable) {
            expect(eq(hexOf(encoded(thomas, std::span{&row.item, 1UZ})), std::string{row.thomas})) << std::format("{:02X} under ge_thomas", row.item);
        }

        // the cheapest possible check that the whole table is right, over its whole domain
        const ManchesterConfig ieee     = configured(ManchesterConvention::Ieee8023);
        std::size_t            agreeing = 0UZ;
        for (unsigned value = 0U; value < 256U; ++value) {
            const std::uint8_t item = static_cast<std::uint8_t>(value);
            const auto         a    = encoded(ieee, std::span{&item, 1UZ});
            const auto         b    = encoded(thomas, std::span{&item, 1UZ});
            agreeing += (a[0] == static_cast<std::uint8_t>(~b[0]) && a[1] == static_cast<std::uint8_t>(~b[1])) ? 1UZ : 0UZ;
        }
        expect(eq(agreeing, 256UZ)) << "every ge_thomas entry is the bitwise complement of its ieee802_3 neighbor";

        expect(eq(conventionName(ManchesterConvention::Ieee8023), std::string_view{"ieee802_3"}));
        expect(eq(conventionName(ManchesterConvention::GeThomas), std::string_view{"ge_thomas"}));
        expect(that % (conventionFromName("ieee802_3") == ManchesterConvention::Ieee8023));
        expect(that % (conventionFromName("ge_thomas") == ManchesterConvention::GeThomas));
        expect(that % (gr::digital::standard::ethernet == ManchesterConvention::Ieee8023)) << "IEEE 802.3 and IEC 62386 are on one side";
        expect(that % (gr::digital::standard::dali == ManchesterConvention::Ieee8023));
        expect(that % (gr::digital::standard::mil1553 == ManchesterConvention::GeThomas)) << "and MIL-STD-1553 is on the other, which is the whole point of the setting";
    };

    // The trap, both halves. A test that only asserted a difference would pass on a decoder that
    //    flagged everything, so the all-zero violation stream is the assertion that matters.
    "the wrong convention is the exact complement and raises nothing"_test = [] {
        struct Row {
            std::uint8_t     item;
            std::string_view chips;
            std::uint8_t     wrong;
        };
        constexpr Row kRows[] = {{0x47U, "9A 95", 0xB8U}, {0x00U, "AA AA", 0xFFU}, {0xAAU, "66 66", 0x55U}, {0xA5U, "66 99", 0x5AU}};

        const ManchesterConfig ieee = configured(ManchesterConvention::Ieee8023);
        for (const Row& row : kRows) {
            const auto chips = encoded(ieee, std::span{&row.item, 1UZ});
            expect(eq(hexOf(chips), std::string{row.chips})) << std::format("{:02X}", row.item);

            ManchesterConfig right  = configured(ManchesterConvention::Ieee8023);
            const Decoded    honest = decoded(right, chips);
            expect(eq(static_cast<unsigned>(honest.items.front()), static_cast<unsigned>(row.item))) << "the right convention returns the data";
            expect(eq(static_cast<unsigned>(honest.flags.front()), 0U));

            ManchesterConfig wrong  = configured(ManchesterConvention::GeThomas);
            const Decoded    silent = decoded(wrong, chips);
            expect(that % (silent.items.front() != row.item)) << std::format("{:02X}: the wrong convention must not return the data", row.item);
            expect(eq(static_cast<unsigned>(silent.items.front()), static_cast<unsigned>(row.wrong))) << "and what it returns is the exact complement";
            expect(eq(static_cast<unsigned>(silent.items.front()), static_cast<unsigned>(static_cast<std::uint8_t>(~row.item))));
            expect(eq(static_cast<unsigned>(silent.flags.front()), 0U)) << std::format("{:02X}: and it says nothing at all about it", row.item);
        }

        // the same thing over a stream, so that the silence is not an artifact of one byte
        const auto       source = randomItems(4096UZ, 8U, 0x243F6A8885A308D3ULL);
        const auto       chips  = encoded(ieee, source);
        ManchesterConfig wrong  = configured(ManchesterConvention::GeThomas);
        const Decoded    silent = decoded(wrong, chips);
        std::size_t      exact  = 0UZ;
        for (std::size_t i = 0UZ; i < source.size(); ++i) {
            exact += silent.items[i] == static_cast<std::uint8_t>(~source[i]) ? 1UZ : 0UZ;
        }
        expect(eq(exact, source.size())) << "every byte complemented";
        expect(eq(countOnes(silent.flags), 0UZ)) << "and not one violation flag raised over 4096 items";
    };

    // The round trip, over every width, order and convention this kernel offers.
    "the round trip is exact from item 0 with no violations"_test = [] {
        constexpr unsigned kWidths[] = {1U, 2U, 3U, 4U, 8U};

        std::size_t cases = 0UZ;
        std::size_t exact = 0UZ;
        for (const ManchesterConvention convention : kConventions) {
            for (const BitOrder order : kOrders) {
                for (const unsigned width : kWidths) {
                    ManchesterConfig cfg    = configured(convention, width, order);
                    const auto       source = randomItems(4096UZ, width, 0x13198A2E03707344ULL + width);
                    ManchesterConfig sink   = configured(convention, width, order);
                    const Decoded    trip   = decoded(sink, encoded(cfg, source));
                    ++cases;
                    exact += (firstDifference(trip.items, source) == source.size() && countOnes(trip.flags) == 0UZ) ? 1UZ : 0UZ;
                }
            }
        }
        expect(eq(cases, 20UZ));
        expect(eq(exact, 20UZ)) << "20 of 20 round trips exact from item 0, with an all-zero violation stream";
    };

    // What the code buys, asserted as properties of every input rather than as sample statistics.
    "exact DC balance, a mid-bit transition always, and a run of at most two"_test = [] {
        for (const ManchesterConvention convention : kConventions) {
            const ManchesterConfig cfg    = configured(convention, 1U);
            const auto             source = randomItems(4096UZ, 1U, 0x852835481D8E5A15ULL);
            const auto             chips  = streamOf(encoded(cfg, source), 1U, BitOrder::MsbFirst);
            const std::string      name{conventionName(convention)};

            expect(eq(chips.size(), 8192UZ)) << "the chip rate is exactly twice the bit rate";
            expect(eq(countOnes(chips), 4096UZ)) << std::format("{}: exactly half the chips are ones, and never merely close", name);

            int         disparity            = 0;
            std::size_t offBalanceBoundaries = 0UZ;
            std::size_t centers              = 0UZ;
            for (std::size_t j = 0UZ; j < source.size(); ++j) {
                centers += chips[2UZ * j] != chips[2UZ * j + 1UZ] ? 1UZ : 0UZ;
                disparity += (chips[2UZ * j] != 0U ? 1 : -1) + (chips[2UZ * j + 1UZ] != 0U ? 1 : -1);
                offBalanceBoundaries += disparity != 0 ? 1UZ : 0UZ;
            }
            expect(eq(centers, source.size())) << std::format("{}: a transition at every bit center, unconditionally", name);
            expect(eq(offBalanceBoundaries, 0UZ)) << std::format("{}: the running disparity is back to zero at every bit boundary", name);

            std::size_t longest = 1UZ;
            std::size_t run     = 1UZ;
            for (std::size_t i = 1UZ; i < chips.size(); ++i) {
                run     = chips[i] == chips[i - 1UZ] ? run + 1UZ : 1UZ;
                longest = std::max(longest, run);
            }
            expect(eq(longest, 2UZ)) << std::format("{}: never more than two like chips in a row, and two are attained", name);

            std::size_t boundaryRule = 0UZ;
            std::size_t fewerThanOne = 0UZ;
            std::size_t moreThanTwo  = 0UZ;
            for (std::size_t j = 0UZ; j + 1UZ < source.size(); ++j) {
                const bool boundary = chips[2UZ * j + 1UZ] != chips[2UZ * j + 2UZ];
                boundaryRule += boundary == (source[j] == source[j + 1UZ]) ? 1UZ : 0UZ;
                const std::size_t perBit = 1UZ + (boundary ? 1UZ : 0UZ);
                fewerThanOne += perBit < 1UZ ? 1UZ : 0UZ;
                moreThanTwo += perBit > 2UZ ? 1UZ : 0UZ;
            }
            expect(eq(boundaryRule, source.size() - 1UZ)) << std::format("{}: a boundary transition exactly where the adjacent data bits are equal", name);
            expect(eq(fewerThanOne + moreThanTwo, 0UZ)) << std::format("{}: between one and two transitions per bit time, never zero", name);
        }
    };

    // A chip pair of 00 or 11 is information in MIL-STD-1553 and IEEE 802.5, so it is reported and
    //    not absorbed, and above all it does not move the grid behind the caller's back.
    "deliberate violations are flagged, carried and never realigned on"_test = [] {
        for (const ManchesterConvention convention : kConventions) {
            const unsigned         k     = static_cast<unsigned>(conventionXor(convention));
            const ManchesterConfig cfg   = configured(convention, 1U);
            const auto             data  = randomItems(64UZ, 1U, 0xA4093822299F31D0ULL);
            auto                   chips = streamOf(encoded(cfg, data), 1U, BitOrder::MsbFirst);

            // the MIL-STD-1553 command sync's shape: three bit times wide, one polarity for the first
            // half of it and the other for the second, which is three chip pairs of which two are invalid
            const std::size_t                     at = chips.size();
            constexpr std::array<std::uint8_t, 6> kSync{1U, 1U, 1U, 0U, 0U, 0U};
            chips.insert(chips.end(), kSync.begin(), kSync.end());
            const auto tail = streamOf(encoded(cfg, data), 1U, BitOrder::MsbFirst);
            chips.insert(chips.end(), tail.begin(), tail.end());

            ManchesterConfig sink   = configured(convention, 1U);
            const Decoded    result = decoded(sink, chips);

            expect(eq(result.items.size(), chips.size() / 2UZ)) << "exactly one item per two input items, nothing dropped";
            expect(eq(result.flags.size(), result.items.size()));

            std::vector<std::size_t> flagged;
            for (std::size_t i = 0UZ; i < result.flags.size(); ++i) {
                if (result.flags[i] != 0U) {
                    flagged.push_back(i);
                }
            }
            const std::size_t base = at / 2UZ;
            expect(eq(flagged.size(), 2UZ)) << "the flag is set at exactly the invalid pairs and nowhere else";
            expect(eq(flagged.front(), base)) << "the sync's first pair";
            expect(eq(flagged.back(), base + 2UZ)) << "and its third";
            expect(eq(static_cast<unsigned>(result.items[base]), 1U ^ k)) << "the decoded bit at a flagged position is the common chip corrected by the convention";
            expect(eq(static_cast<unsigned>(result.items[base + 2UZ]), 0U ^ k));
            expect(eq(static_cast<unsigned>(result.items[base + 1UZ]), 1U ^ k)) << "and the sync's one valid pair decodes as any other";

            expect(eq(static_cast<unsigned>(sink.gridParity), 0U)) << "a violation must not move the pairing grid";
            expect(eq(sink.nOrphanItems, 0ULL));
            expect(eq(sink.nDroppedChips, 0ULL));

            // the data on either side of the sync is untouched by it
            expect(eq(firstDifference(std::span<const std::uint8_t>{result.items}.first(data.size()), data), data.size())) << "the run before the sync";
            expect(eq(firstDifference(std::span<const std::uint8_t>{result.items}.last(data.size()), data), data.size())) << "and the run after it";
        }
    };

    // The criterion that makes "not absorbed" mean something: the two output streams and the
    //    convention are the chip stream, exactly, with nothing discarded on the way.
    "(out, violation) is a lossless recoding of the chip pair"_test = [] {
        for (const ManchesterConvention convention : kConventions) {
            for (const BitOrder order : kOrders) {
                const unsigned k = static_cast<unsigned>(conventionXor(convention));

                // arbitrary chips, not encoder output, so roughly half the pairs are violations
                const auto       chips  = randomItems(8192UZ, 8U, 0x082EFA98EC4E6C89ULL);
                ManchesterConfig sink   = configured(convention, 8U, order);
                const Decoded    result = decoded(sink, chips);

                const auto stream        = streamOf(chips, 8U, order);
                const auto reconstructed = chipsFrom(streamOf(result.items, 8U, order), streamOf(result.flags, 8U, order), k);
                expect(eq(firstDifference(reconstructed, stream), stream.size())) << std::format("{} at {}: the chip stream is recoverable from the two outputs", conventionName(convention), magic_enum::enum_name(order));

                const std::size_t violations = countOnes(streamOf(result.flags, 8U, order));
                expect(that % (violations > stream.size() / 8UZ)) << "arbitrary chips must actually contain violations, or the criterion is vacuous";
            }
        }
    };

    // What Manchester detects and what it does not, with the undetectable case asserted as a
    //    negative so that nobody improves the decoder into claiming a guarantee it does not have.
    "every single-chip error is flagged, and a whole pair is silent"_test = [] {
        for (const ManchesterConvention convention : kConventions) {
            const ManchesterConfig cfg   = configured(convention, 1U);
            const auto             data  = randomItems(1024UZ, 1U, 0x9216D5D98979FB1BULL);
            const auto             clean = encoded(cfg, data);

            std::size_t unflagged     = 0UZ;
            std::size_t headCorrect   = 0UZ;
            std::size_t tailCorrupted = 0UZ;
            std::size_t pairFlagged   = 0UZ;
            std::size_t pairUnchanged = 0UZ;
            for (std::size_t j = 0UZ; j < data.size(); ++j) {
                for (const std::size_t which : {0UZ, 1UZ}) {
                    auto broken = clean;
                    broken[2UZ * j + which] ^= std::uint8_t{1};
                    ManchesterConfig sink   = configured(convention, 1U);
                    const Decoded    result = decoded(sink, broken);
                    unflagged += result.flags[j] == 0U ? 1UZ : 0UZ;
                    if (which == 1UZ) {
                        headCorrect += result.items[j] == data[j] ? 1UZ : 0UZ;
                    } else {
                        tailCorrupted += result.items[j] != data[j] ? 1UZ : 0UZ;
                    }
                }

                auto both = clean;
                both[2UZ * j] ^= std::uint8_t{1};
                both[2UZ * j + 1UZ] ^= std::uint8_t{1};
                ManchesterConfig sink   = configured(convention, 1U);
                const Decoded    result = decoded(sink, both);
                pairFlagged += result.flags[j] != 0U ? 1UZ : 0UZ;
                pairUnchanged += result.items[j] == data[j] ? 1UZ : 0UZ;
            }

            const std::string name{conventionName(convention)};
            expect(eq(unflagged, 0UZ)) << std::format("{}: every single-chip error makes its pair invalid, and the detection is unconditional", name);
            expect(eq(headCorrect, data.size())) << std::format("{}: flipping the second chip flags the pair and leaves the bit correct", name);
            expect(eq(tailCorrupted, data.size())) << std::format("{}: flipping the first chip flags the pair and inverts the bit", name);
            expect(eq(pairFlagged, 0UZ)) << std::format("{}: flipping both chips of a pair is not detected, and there is no smaller undetectable error", name);
            expect(eq(pairUnchanged, 0UZ)) << std::format("{}: and it inverts the bit silently", name);
        }
    };

    // Misalignment, which is loud in proportion to transition density and silent on a constant
    //    stream, where it is the wrong-convention symptom exactly.
    "one chip out of phase: the rows, the rule, and the silent case"_test = [] {
        struct Row {
            std::uint8_t     item;
            std::string_view bits;
            std::string_view flags;
            std::size_t      count;
        };
        constexpr Row kRows[] = {{0x47U, "5C", "E4", 4UZ}, {0x00U, "7F", "80", 1UZ}, {0xAAU, "AA", "FF", 8UZ}};

        const ManchesterConfig ieee = configured(ManchesterConvention::Ieee8023);
        for (const Row& row : kRows) {
            const auto       chips  = encoded(ieee, std::span{&row.item, 1UZ});
            ManchesterConfig offset = configured(ManchesterConvention::Ieee8023, 8U, BitOrder::MsbFirst, 1U);
            const Decoded    result = decoded(offset, chips);
            expect(eq(hexOf(result.items), std::string{row.bits})) << std::format("{:02X} decoded one chip out of phase", row.item);
            expect(eq(hexOf(result.flags), std::string{row.flags}));
            expect(eq(countOnes(streamOf(result.flags, 8U, BitOrder::MsbFirst)), row.count)) << std::format("{:02X}: flagged of 8", row.item);
        }

        // the general rule, independent of the convention: a misaligned decoder always emits NOT b[j]
        // and flags exactly where the adjacent data bits differ
        for (const ManchesterConvention convention : kConventions) {
            const ManchesterConfig cfg   = configured(convention, 1U);
            const auto             data  = randomItems(4096UZ, 1U, 0xB3EE1411636FBC2BULL);
            const auto             chips = encoded(cfg, data);

            ManchesterConfig offset = configured(convention, 1U, BitOrder::MsbFirst, 1U);
            const Decoded    result = decoded(offset, chips);

            std::size_t wrongBit  = 0UZ;
            std::size_t wrongFlag = 0UZ;
            for (std::size_t j = 0UZ; j + 1UZ < data.size(); ++j) {
                wrongBit += result.items[j + 1UZ] != (data[j] ^ 1U) ? 1UZ : 0UZ;
                wrongFlag += result.flags[j + 1UZ] != (data[j] != data[j + 1UZ] ? 1U : 0U) ? 1UZ : 0UZ;
            }
            expect(eq(wrongBit, 0UZ)) << "the misaligned decoder emits the complement of the earlier bit throughout";
            expect(eq(wrongFlag, 0UZ)) << "and flags exactly where the two adjacent data bits differ";

            // data-dependent, so a band rather than a count
            const std::size_t flagged = countOnes(result.flags);
            expect(that % (flagged > result.flags.size() * 45UZ / 100UZ && flagged < result.flags.size() * 55UZ / 100UZ)) << std::format("uniform random data flagged {} of {} positions, which should be about half", flagged, result.flags.size());
        }

        // and the half that matters: on a constant stream nothing but the orphan is flagged, and the
        // output is the wrong-convention output in all but the first item
        const std::vector<std::uint8_t> constant(64UZ, std::uint8_t{0x00});
        const auto                      chips  = encoded(ieee, constant);
        ManchesterConfig                offset = configured(ManchesterConvention::Ieee8023, 8U, BitOrder::MsbFirst, 1U);
        const Decoded                   quiet  = decoded(offset, chips);
        expect(eq(countOnes(streamOf(quiet.flags, 8U, BitOrder::MsbFirst)), 1UZ)) << "one flag in the whole stream, and it is the orphan";
        expect(eq(static_cast<unsigned>(quiet.flags.front()), 0x80U)) << "at the very first position";

        ManchesterConfig wrong      = configured(ManchesterConvention::GeThomas);
        const Decoded    complement = decoded(wrong, chips);
        expect(eq(firstDifference(std::span<const std::uint8_t>{quiet.items}.subspan(1UZ), std::span<const std::uint8_t>{complement.items}.subspan(1UZ)), quiet.items.size() - 1UZ)) << "the two faults are indistinguishable after the first item, which is why the convention has no default";
        expect(that % (quiet.items.front() != complement.items.front())) << "and distinguishable only in that one item";
    };

    // The alignment contract. Off by one here is the whole failure, so the item index is asserted.
    "the pairing grid, what a change costs, and what does not move it"_test = [] {
        // (a) a chip_phase = 1 run begins with exactly one flagged orphan and spends nothing else
        const ManchesterConfig cfg   = configured(ManchesterConvention::Ieee8023, 1U);
        const auto             data  = randomItems(2048UZ, 1U, 0x8A5EDAFDA9C4C61EULL);
        const auto             chips = encoded(cfg, data);

        ManchesterConfig offset = configured(ManchesterConvention::Ieee8023, 1U, BitOrder::MsbFirst, 1U);
        const Decoded    late   = decoded(offset, chips);
        expect(eq(static_cast<unsigned>(late.flags.front()), 1U)) << "the first output item is flagged";
        expect(eq(offset.nOrphanItems, 1ULL)) << "exactly one item is fabricated";
        expect(eq(offset.nDroppedChips, 0ULL)) << "and no chip is lost";
        expect(that % offset.hasCarry) << "steady phase 1 costs one chip of state and nothing else";

        ManchesterConfig aligned = configured(ManchesterConvention::Ieee8023, 1U);
        const Decoded    shifted = decoded(aligned, std::span<const std::uint8_t>{chips}.subspan(1UZ));
        expect(eq(firstDifference(std::span<const std::uint8_t>{late.items}.subspan(1UZ), shifted.items), shifted.items.size())) << "from item 1 a phase-1 decode is a phase-0 decode of the same chips shifted by one chip";
        expect(eq(firstDifference(std::span<const std::uint8_t>{late.flags}.subspan(1UZ), shifted.flags), shifted.flags.size()));

        // (b) the parity a chip index requests, and the consequence at an even width
        ManchesterConfig grid = configured(ManchesterConvention::Ieee8023, 1U);
        expect(eq(realignManchester(grid, 8ULL), 0UZ)) << "an even chip index is the natural grid";
        expect(eq(static_cast<unsigned>(grid.gridParity), 0U));
        expect(eq(realignManchester(grid, 9ULL), 0UZ));
        expect(eq(static_cast<unsigned>(grid.gridParity), 1U)) << "an odd chip index moves it";
        for (const std::uint64_t item : {0ULL, 1ULL, 7ULL, 8ULL, 1000ULL}) {
            ManchesterConfig wide = configured(ManchesterConvention::Ieee8023, 8U);
            expect(eq(realignManchester(wide, item * 8ULL), 0UZ));
            expect(eq(static_cast<unsigned>(wide.gridParity), 0U)) << std::format("at bits_per_item = 8 an item offset of {} can only ever request parity 0", item);
        }

        // (e) the grid-change budget: every call publishes one item per two input items, in every row
        ManchesterConfig                budget = configured(ManchesterConvention::Ieee8023, 1U);
        const std::vector<std::uint8_t> chunk{1U, 0U};
        std::vector<std::uint8_t>       out(1UZ, 0U);
        std::vector<std::uint8_t>       flags(1UZ, 0U);

        decodeManchester(budget, chunk, out, flags);
        expect(eq(budget.nOrphanItems, 0ULL)) << "steady on parity 0 spends nothing";
        expect(eq(budget.nDroppedChips, 0ULL));
        expect(eq(realignManchester(budget, 0ULL), 0UZ)) << "asking for the parity already in force does nothing";
        expect(eq(budget.nOrphanItems, 0ULL));
        expect(eq(budget.nDroppedChips, 0ULL));

        expect(eq(realignManchester(budget, 1ULL), 0UZ)) << "moving to parity 1 drops no chip";
        decodeManchester(budget, chunk, out, flags);
        expect(eq(budget.nOrphanItems, 1ULL)) << "it fabricates exactly one item instead";
        expect(eq(budget.nDroppedChips, 0ULL));
        expect(eq(static_cast<unsigned>(flags.front()), 1U)) << "and that item is flagged in band as well as counted";
        expect(that % budget.hasCarry);

        decodeManchester(budget, chunk, out, flags);
        expect(eq(budget.nOrphanItems, 1ULL)) << "steady on parity 1 spends nothing further";
        expect(eq(realignManchester(budget, 1ULL), 0UZ));
        expect(eq(budget.nOrphanItems, 1ULL));

        expect(eq(realignManchester(budget, 2ULL), 1UZ)) << "moving back to parity 0 drops the held chip";
        expect(eq(budget.nDroppedChips, 1ULL)) << "and counts it";
        expect(eq(budget.nOrphanItems, 1ULL)) << "and fabricates nothing";
        expect(that % !budget.hasCarry);
        decodeManchester(budget, chunk, out, flags);
        expect(eq(budget.nDroppedChips, 1ULL));

        // (f) a long deliberate violation run moves nothing
        ManchesterConfig                still = configured(ManchesterConvention::GeThomas, 1U);
        const std::vector<std::uint8_t> flat(4096UZ, std::uint8_t{1});
        const Decoded                   all = decoded(still, flat);
        expect(eq(countOnes(all.flags), all.flags.size())) << "every pair of a constant chip run is a violation";
        expect(eq(static_cast<unsigned>(still.gridParity), 0U)) << "and none of them realigns the decoder";
        expect(eq(still.nOrphanItems, 0ULL));
        expect(eq(still.nDroppedChips, 0ULL));
    };

    // The line coder sits directly beside a packer in every chain that has one, so the two must
    //     compose, and the two orders must not be interchangeable.
    "bit-order interop, composed with the bit-packing conventions"_test = [] {
        constexpr std::size_t kBytes = 1024UZ;

        std::size_t identical = 0UZ;
        for (const ManchesterConvention convention : kConventions) {
            for (const BitOrder order : kOrders) {
                const auto source = randomItems(kBytes, 8U, 0xC97C50DDA0B5C136ULL);

                BitRepack unpack{};
                configureRepack(unpack, 8U, 1U, order, order);
                std::vector<std::uint8_t> bits(8UZ * kBytes, std::uint8_t{0});
                repack(unpack, source, bits);

                const ManchesterConfig perBit = configured(convention, 1U, order);
                const auto             chips  = encoded(perBit, bits);
                BitRepack              pack{};
                configureRepack(pack, 1U, 8U, order, order);
                std::vector<std::uint8_t> packed(2UZ * kBytes, std::uint8_t{0});
                repack(pack, chips, packed);

                const ManchesterConfig whole  = configured(convention, 8U, order);
                const auto             direct = encoded(whole, source);
                identical += firstDifference(packed, direct) == direct.size() ? 1UZ : 0UZ;
            }
        }
        expect(eq(identical, 4UZ)) << "4 of 4: unpacking, coding one bit at a time and repacking is coding whole items";

        // and the negative half: the two orders are not interchangeable
        const ManchesterConfig msb   = configured(ManchesterConvention::Ieee8023, 8U, BitOrder::MsbFirst);
        const ManchesterConfig lsb   = configured(ManchesterConvention::Ieee8023, 8U, BitOrder::LsbFirst);
        std::size_t            agree = 0UZ;
        for (unsigned value = 0U; value < 256U; ++value) {
            const std::uint8_t item = static_cast<std::uint8_t>(value);
            agree += encoded(msb, std::span{&item, 1UZ}) == encoded(lsb, std::span{&item, 1UZ}) ? 1UZ : 0UZ;
        }
        expect(eq(agree, 16UZ)) << "the two orders agree on one byte in sixteen and differ on the rest";
    };

    // A stream value cannot stop the kernel: unused high bits are masked, never rejected, and an
    //     invalid chip pair is a legal value rather than an error.
    "unused high bits are masked and nothing is written above the field"_test = [] {
        constexpr std::size_t kItems = 100000UZ;
        constexpr unsigned    kWidth = 3U;

        const std::vector<std::uint8_t> loud(kItems, std::uint8_t{0xFF});
        const std::vector<std::uint8_t> quiet(kItems, std::uint8_t{0x07});

        for (const ManchesterConvention convention : kConventions) {
            const ManchesterConfig cfg       = configured(convention, kWidth);
            const auto             fromLoud  = encoded(cfg, loud);
            const auto             fromQuiet = encoded(cfg, quiet);
            expect(eq(firstDifference(fromLoud, fromQuiet), fromLoud.size())) << "all 0xFF and all 0x07 are the same three-bit stream to the encoder";
            expect(std::ranges::all_of(fromLoud, [](std::uint8_t value) { return value < 8U; })) << "and nothing is written above the field";

            ManchesterConfig noisy   = configured(convention, kWidth);
            ManchesterConfig clean   = configured(convention, kWidth);
            const Decoded    fromAll = decoded(noisy, loud);
            const Decoded    fromLow = decoded(clean, quiet);
            expect(eq(firstDifference(fromAll.items, fromLow.items), fromAll.items.size())) << "and the same stream to the decoder";
            expect(eq(firstDifference(fromAll.flags, fromLow.flags), fromAll.flags.size()));
            expect(std::ranges::all_of(fromAll.items, [](std::uint8_t value) { return value < 8U; }));
            expect(std::ranges::all_of(fromAll.flags, [](std::uint8_t value) { return value < 8U; }));
        }
    };

    // Forgetting to write the carry back at the end of a call is the classic bug in a block that
    //     holds one chip, and it produces output that is right for large chunks and wrong for small ones.
    "chunk independence, bit-identical, across a grid change"_test = [] {
        constexpr std::size_t kChunks[] = {2UZ, 4UZ, 18UZ, 4096UZ};
        constexpr unsigned    kWidths[] = {1U, 8U};

        std::size_t cases     = 0UZ;
        std::size_t divergent = 0UZ;
        for (const ManchesterConvention convention : kConventions) {
            for (const BitOrder order : kOrders) {
                for (const unsigned width : kWidths) {
                    const auto source = randomItems(2304UZ, width, 0xBE5466CF34E90C6CULL);

                    // the encoder holds nothing, so every division of the stream must give one answer
                    const ManchesterConfig cfg   = configured(convention, width, order);
                    const auto             whole = encoded(cfg, source);
                    for (const std::size_t chunk : kChunks) {
                        std::vector<std::uint8_t> parts(whole.size(), std::uint8_t{0xEE});
                        for (std::size_t index = 0UZ; index < source.size(); index += chunk) {
                            const std::size_t take = std::min(chunk, source.size() - index);
                            encodeManchester(cfg, std::span<const std::uint8_t>{source}.subspan(index, take), std::span<std::uint8_t>{parts}.subspan(2UZ * index, 2UZ * take));
                        }
                        ++cases;
                        divergent += firstDifference(parts, whole) == whole.size() ? 0UZ : 1UZ;
                    }

                    // and the decoder holds one chip, at both phases
                    for (const unsigned phase : {0U, 1U}) {
                        ManchesterConfig one       = configured(convention, width, order, phase);
                        const Decoded    reference = decoded(one, whole);
                        for (const std::size_t chunk : kChunks) {
                            ManchesterConfig many  = configured(convention, width, order, phase);
                            const Decoded    parts = decodedInChunks(many, whole, chunk);
                            ++cases;
                            divergent += (firstDifference(parts.items, reference.items) == reference.items.size() && firstDifference(parts.flags, reference.flags) == reference.flags.size() && many.hasCarry == one.hasCarry && many.carry == one.carry) ? 0UZ : 1UZ;
                        }
                    }

                    // and a grid change part-way through, at an item boundary every chunk size shares
                    constexpr std::size_t kSplit = 36UZ;
                    for (const std::size_t chunk : {2UZ, 4UZ, 18UZ}) {
                        ManchesterConfig split = configured(convention, width, order);
                        Decoded          run{std::vector<std::uint8_t>(whole.size() / 2UZ, std::uint8_t{0xEE}), std::vector<std::uint8_t>(whole.size() / 2UZ, std::uint8_t{0xEE})};
                        for (std::size_t index = 0UZ; index < whole.size(); index += chunk) {
                            if (index == kSplit) {
                                [[maybe_unused]] const std::size_t droppedInChunks = realignManchester(split, 1ULL);
                            }
                            const std::size_t take = std::min(chunk, whole.size() - index);
                            decodeManchester(split, std::span<const std::uint8_t>{whole}.subspan(index, take), std::span<std::uint8_t>{run.items}.subspan(index / 2UZ, take / 2UZ), std::span<std::uint8_t>{run.flags}.subspan(index / 2UZ, take / 2UZ));
                        }

                        ManchesterConfig two = configured(convention, width, order);
                        Decoded          want{std::vector<std::uint8_t>(whole.size() / 2UZ, std::uint8_t{0xEE}), std::vector<std::uint8_t>(whole.size() / 2UZ, std::uint8_t{0xEE})};
                        decodeManchester(two, std::span<const std::uint8_t>{whole}.first(kSplit), std::span<std::uint8_t>{want.items}.first(kSplit / 2UZ), std::span<std::uint8_t>{want.flags}.first(kSplit / 2UZ));
                        [[maybe_unused]] const std::size_t droppedInOne = realignManchester(two, 1ULL);
                        decodeManchester(two, std::span<const std::uint8_t>{whole}.subspan(kSplit), std::span<std::uint8_t>{want.items}.subspan(kSplit / 2UZ), std::span<std::uint8_t>{want.flags}.subspan(kSplit / 2UZ));

                        ++cases;
                        divergent += (firstDifference(run.items, want.items) == want.items.size() && firstDifference(run.flags, want.flags) == want.flags.size() && split.nOrphanItems == two.nOrphanItems) ? 0UZ : 1UZ;
                    }
                }
            }
        }
        expect(eq(cases, 120UZ));
        expect(eq(divergent, 0UZ)) << "no division of the stream into calls changes one bit of the answer";
    };

    // The fast forms exist so that a benchmark can choose, which only works if they are exactly
    //     the definition. Every width, order, convention and phase, in both directions.
    "the fast forms agree with the definition bit for bit"_test = [] {
        std::size_t comparisons   = 0UZ;
        std::size_t disagreements = 0UZ;
        for (const ManchesterConvention convention : kConventions) {
            for (const BitOrder order : kOrders) {
                for (unsigned width = 1U; width <= 8U; ++width) {
                    const auto             source = randomItems(2048UZ, width, 0x2FFD72DBD01ADFB7ULL + width);
                    const ManchesterConfig cfg    = configured(convention, width, order);

                    const auto                table = encoded(cfg, source);
                    std::vector<std::uint8_t> loop(table.size(), std::uint8_t{0xEE});
                    gr::digital::detail::encodeReference(cfg, source, loop);
                    ++comparisons;
                    disagreements += firstDifference(table, loop) == table.size() ? 0UZ : 1UZ;

                    for (const unsigned phase : {0U, 1U}) {
                        ManchesterConfig fast = configured(convention, width, order, phase);
                        ManchesterConfig slow = configured(convention, width, order, phase);
                        const Decoded    got  = decoded(fast, table);

                        Decoded want{std::vector<std::uint8_t>(source.size(), std::uint8_t{0xEE}), std::vector<std::uint8_t>(source.size(), std::uint8_t{0xEE})};
                        gr::digital::detail::decodeReference(slow, table, want.items, want.flags);

                        ++comparisons;
                        const bool same = firstDifference(got.items, want.items) == want.items.size() && firstDifference(got.flags, want.flags) == want.flags.size() && fast.carry == slow.carry && fast.hasCarry == slow.hasCarry && fast.nOrphanItems == slow.nOrphanItems;
                        disagreements += same ? 0UZ : 1UZ;
                    }
                }
            }
        }
        expect(eq(comparisons, 96UZ));
        expect(eq(disagreements, 0UZ)) << "the table, the de-interleave and the two reference loops are one function";
    };

    // What configureManchester() refuses, and the guarantee that a refusal cannot half-apply.
    "validation rejects, and the previous configuration stands"_test = [] {
        ManchesterConfig cfg = configured(ManchesterConvention::GeThomas, 5U, BitOrder::LsbFirst, 1U);

        for (const unsigned bad : {0U, 9U, 255U}) {
            expect(throws<std::invalid_argument>([&cfg, bad] { configureManchester(cfg, ManchesterConvention::Ieee8023, bad, BitOrder::MsbFirst); })) << std::format("bitsPerItem = {}", bad);
        }
        for (const unsigned bad : {2U, 100U}) {
            expect(throws<std::invalid_argument>([&cfg, bad] { configureManchester(cfg, ManchesterConvention::Ieee8023, 8U, BitOrder::MsbFirst, bad); })) << std::format("chipPhase = {}", bad);
        }

        expect(that % (cfg.convention == ManchesterConvention::GeThomas)) << "a rejected setting leaves the previous configuration standing";
        expect(eq(static_cast<unsigned>(cfg.bitsPerItem), 5U));
        expect(that % (cfg.order == BitOrder::LsbFirst));
        expect(eq(static_cast<unsigned>(cfg.chipPhase), 1U));
        expect(eq(static_cast<unsigned>(cfg.k), 0U));
        const std::uint8_t item = 0x1FU;
        expect(eq(hexOf(encoded(cfg, std::span{&item, 1UZ})), std::string{"15 0A"})) << "and it still codes";

        // the names the blocks expose, and the empty string that stands for "not set"
        for (const std::string_view bad : {"", "ieee802.3", "IEEE802_3", "thomas", "manchester"}) {
            expect(throws<std::invalid_argument>([bad] { [[maybe_unused]] const ManchesterConvention ignored = conventionFromName(bad); })) << std::format("convention '{}'", bad);
        }
        bool quoted = false;
        try {
            [[maybe_unused]] const ManchesterConvention ignored = conventionFromName("differential");
        } catch (const std::invalid_argument& error) {
            quoted = std::string_view{error.what()}.find("differential") != std::string_view::npos;
        }
        expect(quoted) << "the offending value is quoted back in the message";

        expect(nothrow([] { [[maybe_unused]] const ManchesterConfig ok = configured(ManchesterConvention::Ieee8023, 1U, BitOrder::MsbFirst, 1U); })) << "the edges of both ranges are accepted";
        expect(nothrow([] { [[maybe_unused]] const ManchesterConfig ok = configured(ManchesterConvention::GeThomas, 8U, BitOrder::LsbFirst, 0U); }));
    };

    // The state a caller has to account for, which is bounded by construction rather than by a threshold.
    "the configuration's shape, and what it carries between calls"_test = [] {
        const ManchesterConfig cfg = configured(ManchesterConvention::Ieee8023);
        expect(eq(cfg.table.size(), 256UZ)) << "the chip table covers every item value at every width";
        expect(eq(sizeof(cfg.table), 512UZ)) << "512 bytes, fixed by the item type and never resized";
        expect(eq(static_cast<unsigned>(cfg.gridParity), 0U));
        expect(that % !cfg.hasCarry) << "the encoder's direction carries nothing at all";

        // resetManchester() puts the grid back where the phase says and leaves the totals alone
        ManchesterConfig                live = configured(ManchesterConvention::Ieee8023, 1U, BitOrder::MsbFirst, 1U);
        const std::vector<std::uint8_t> chips(64UZ, std::uint8_t{1});
        [[maybe_unused]] const Decoded  ignored = decoded(live, chips);
        expect(that % live.hasCarry);
        expect(eq(live.nOrphanItems, 1ULL));
        resetManchester(live);
        expect(that % !live.hasCarry) << "reset clears the held chip";
        expect(eq(static_cast<unsigned>(live.gridParity), 1U)) << "and restores the configured chip phase";
        expect(eq(live.nOrphanItems, 1ULL)) << "and leaves the counters standing, so a settings change does not erase a run's totals";

        // the two directions of the rate change, which are properties of the code and not of a setting
        const auto source = randomItems(1000UZ, 8U, 0xF12C7F995D0EB1D9ULL);
        expect(eq(encoded(cfg, source).size(), 2000UZ)) << "one input item to two output items";
        ManchesterConfig sink   = configured(ManchesterConvention::Ieee8023);
        const Decoded    result = decoded(sink, encoded(cfg, source));
        expect(eq(result.items.size(), 1000UZ)) << "and two input items to one output item";
        expect(eq(result.flags.size(), 1000UZ)) << "and one violation item beside it";
    };
};

int main() { /* tests are automatically registered and run */ }
