#include <boost/ut.hpp>

#include <magic_enum.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iterator>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <gnuradio-4.0/algorithm/digital/BitPacking.hpp>

namespace {

using gr::digital::BitOrder;
using gr::digital::BitRepack;
using gr::digital::configureRepack;
using gr::digital::mapRepackedOffset;
using gr::digital::repack;
using gr::digital::repackTail;

constexpr BitOrder kOrders[] = {BitOrder::MsbFirst, BitOrder::LsbFirst};

[[nodiscard]] std::size_t inChunkOf(const BitRepack& cfg) { return static_cast<std::size_t>(cfg.inChunk); }
[[nodiscard]] std::size_t outChunkOf(const BitRepack& cfg) { return static_cast<std::size_t>(cfg.outChunk); }

[[nodiscard]] BitRepack configured(unsigned bitsIn, unsigned bitsOut, BitOrder orderIn, BitOrder orderOut) {
    BitRepack cfg{};
    configureRepack(cfg, bitsIn, bitsOut, orderIn, orderOut);
    return cfg;
}

/// The definition of the bit stream itself: one item becomes its `width` bits, taken from whichever
/// end the order names. Deliberately not a second shift table — a table indexed wrongly would agree
/// with itself and disagree with this.
[[nodiscard]] std::vector<std::uint8_t> toBits(std::span<const std::uint8_t> items, unsigned width, BitOrder order) {
    std::vector<std::uint8_t> bits;
    bits.reserve(items.size() * width);
    for (const std::uint8_t item : items) {
        for (unsigned step = 0U; step < width; ++step) {
            const unsigned position = order == BitOrder::MsbFirst ? width - 1U - step : step;
            bits.push_back(static_cast<std::uint8_t>((item >> position) & 1U));
        }
    }
    return bits;
}

/// The inverse, over whole items only; a trailing part-item is dropped.
[[nodiscard]] std::vector<std::uint8_t> fromBits(std::span<const std::uint8_t> bits, unsigned width, BitOrder order) {
    std::vector<std::uint8_t> items(bits.size() / width, std::uint8_t{0});
    for (std::size_t b = 0UZ; b < items.size() * width; ++b) {
        const unsigned step     = static_cast<unsigned>(b % width);
        const unsigned position = order == BitOrder::MsbFirst ? width - 1U - step : step;
        items[b / width]        = static_cast<std::uint8_t>(items[b / width] | (bits[b] << position));
    }
    return items;
}

[[nodiscard]] std::vector<std::uint8_t> byDefinition(std::span<const std::uint8_t> in, unsigned bitsIn, unsigned bitsOut, BitOrder orderIn, BitOrder orderOut) { //
    return fromBits(toBits(in, bitsIn, orderIn), bitsOut, orderOut);
}

[[nodiscard]] std::vector<std::uint8_t> randomItems(std::size_t count, unsigned width, std::uint64_t seed) {
    std::mt19937_64                              engine{seed};
    std::uniform_int_distribution<std::uint32_t> byte{0U, 255U};
    const std::uint32_t                          mask = (1U << width) - 1U;
    std::vector<std::uint8_t>                    items(count);
    for (std::uint8_t& item : items) {
        item = static_cast<std::uint8_t>(byte(engine) & mask);
    }
    return items;
}

[[nodiscard]] std::vector<std::uint8_t> throughKernel(const BitRepack& cfg, std::span<const std::uint8_t> in) {
    const std::size_t         periods = in.size() / inChunkOf(cfg);
    std::vector<std::uint8_t> out(periods * outChunkOf(cfg), std::uint8_t{0xAA});
    repack(cfg, in.first(periods * inChunkOf(cfg)), out);
    return out;
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

[[nodiscard]] std::string describe(unsigned bitsIn, unsigned bitsOut, BitOrder orderIn, BitOrder orderOut) { //
    return std::format("({} -> {} bits, {} -> {})", bitsIn, bitsOut, magic_enum::enum_name(orderIn), magic_enum::enum_name(orderOut));
}

[[nodiscard]] std::uint8_t reverseByte(std::uint8_t value) {
    std::uint8_t reversed = 0U;
    for (unsigned bit = 0U; bit < 8U; ++bit) {
        reversed = static_cast<std::uint8_t>(reversed | (((value >> bit) & 1U) << (7U - bit)));
    }
    return reversed;
}

} // namespace

const boost::ut::suite<"bit packing"> bitPackingTests = [] {
    using namespace boost::ut;

    // Round-trip identity over the whole parameter space: 64 width pairs times four order pairs.
    "every width pair and order pair round-trips exactly"_test = [] {
        std::size_t cases    = 0UZ;
        std::size_t failures = 0UZ;
        for (unsigned bitsIn = 1U; bitsIn <= 8U; ++bitsIn) {
            for (unsigned bitsOut = 1U; bitsOut <= 8U; ++bitsOut) {
                for (const BitOrder orderIn : kOrders) {
                    for (const BitOrder orderOut : kOrders) {
                        const BitRepack forward = configured(bitsIn, bitsOut, orderIn, orderOut);
                        const BitRepack reverse = configured(bitsOut, bitsIn, orderOut, orderIn);

                        const auto source    = randomItems(32UZ * inChunkOf(forward), bitsIn, 0x9E3779B97F4A7C15ULL + cases);
                        const auto symbols   = throughKernel(forward, source);
                        const auto recovered = throughKernel(reverse, symbols);

                        ++cases;
                        const bool ok = symbols.size() == 32UZ * outChunkOf(forward) && firstDifference(recovered, source) == source.size();
                        failures += ok ? 0UZ : 1UZ;
                        expect(ok) << std::format("{} lost bits at item {}", describe(bitsIn, bitsOut, orderIn, orderOut), firstDifference(recovered, source));
                    }
                }
            }
        }
        expect(eq(cases, 256UZ)) << "the parameter space is 8 x 8 widths times 2 x 2 orders";
        expect(eq(failures, 0UZ));
    };

    // The same space against the definition, so a round trip wrong in both directions cannot pass.
    "every configuration matches the bit-stream definition"_test = [] {
        std::size_t failures = 0UZ;
        for (unsigned bitsIn = 1U; bitsIn <= 8U; ++bitsIn) {
            for (unsigned bitsOut = 1U; bitsOut <= 8U; ++bitsOut) {
                for (const BitOrder orderIn : kOrders) {
                    for (const BitOrder orderOut : kOrders) {
                        const BitRepack cfg      = configured(bitsIn, bitsOut, orderIn, orderOut);
                        const auto      source   = randomItems(16UZ * inChunkOf(cfg), bitsIn, 0xD1B54A32D192ED03ULL + bitsIn * 8U + bitsOut);
                        const auto      kernel   = throughKernel(cfg, source);
                        const auto      expected = byDefinition(source, bitsIn, bitsOut, orderIn, orderOut);

                        const std::size_t at = firstDifference(kernel, expected);
                        failures += at == expected.size() ? 0UZ : 1UZ;
                        expect(eq(at, expected.size())) << std::format("{} differs from the definition at item {}", describe(bitsIn, bitsOut, orderIn, orderOut), at);
                    }
                }
            }
        }
        expect(eq(failures, 0UZ));
    };

    // Eight bits both ways is either the identity or a per-byte bit reversal, and nothing else.
    "eight to eight is the identity or a per-byte bit reversal"_test = [] {
        std::vector<std::uint8_t> all(256UZ);
        for (std::size_t i = 0UZ; i < all.size(); ++i) {
            all[i] = static_cast<std::uint8_t>(i);
        }

        for (const BitOrder order : kOrders) {
            const auto same = throughKernel(configured(8U, 8U, order, order), all);
            expect(eq(firstDifference(same, all), all.size())) << std::format("equal orders ({}) must copy every byte", magic_enum::enum_name(order));
        }

        std::vector<std::uint8_t> reversed(all.size());
        std::ranges::transform(all, reversed.begin(), reverseByte);
        for (const BitOrder orderIn : kOrders) {
            const BitOrder orderOut = orderIn == BitOrder::MsbFirst ? BitOrder::LsbFirst : BitOrder::MsbFirst;
            const auto     mirrored = throughKernel(configured(8U, 8U, orderIn, orderOut), all);
            expect(eq(firstDifference(mirrored, reversed), all.size())) << std::format("unequal orders ({} -> {}) must reverse every byte", magic_enum::enum_name(orderIn), magic_enum::enum_name(orderOut));
        }

        const std::array<std::uint8_t, 6> anchor{0x00U, 0x01U, 0x80U, 0xC5U, 0x3AU, 0xFFU};
        const std::array<std::uint8_t, 6> mirroredAnchor{0x00U, 0x80U, 0x01U, 0xA3U, 0x5CU, 0xFFU};
        expect(eq(firstDifference(throughKernel(configured(8U, 8U, BitOrder::MsbFirst, BitOrder::LsbFirst), anchor), mirroredAnchor), mirroredAnchor.size()));
    };

    // The two eight-bit shapes carry a 64-bit form, which has to be the same answer as the general loop.
    "the 64-bit forms agree with the general loop bit for bit"_test = [] {
        std::vector<std::uint8_t> bytes(256UZ);
        for (std::size_t i = 0UZ; i < bytes.size(); ++i) {
            bytes[i] = static_cast<std::uint8_t>(i);
        }
        const auto bits = randomItems(8UZ * 512UZ, 1U, 0x243F6A8885A308D3ULL);

        for (const BitOrder order : kOrders) {
            const BitRepack           unpack = configured(8U, 1U, order, BitOrder::MsbFirst);
            std::vector<std::uint8_t> fastBits(bytes.size() * 8UZ, std::uint8_t{0xAA});
            std::vector<std::uint8_t> generalBits(fastBits.size(), std::uint8_t{0x55});
            repack(unpack, bytes, fastBits);
            gr::digital::detail::repackGeneral(unpack, bytes, generalBits);
            expect(eq(firstDifference(fastBits, generalBits), generalBits.size())) << std::format("unpacking {} differs at bit {}", magic_enum::enum_name(order), firstDifference(fastBits, generalBits));

            const BitRepack           pack = configured(1U, 8U, BitOrder::MsbFirst, order);
            std::vector<std::uint8_t> fastBytes(bits.size() / 8UZ, std::uint8_t{0xAA});
            std::vector<std::uint8_t> generalBytes(fastBytes.size(), std::uint8_t{0x55});
            repack(pack, bits, fastBytes);
            gr::digital::detail::repackGeneral(pack, bits, generalBytes);
            expect(eq(firstDifference(fastBytes, generalBytes), generalBytes.size())) << std::format("packing {} differs at byte {}", magic_enum::enum_name(order), firstDifference(fastBytes, generalBytes));
        }
    };

    // The boundary widths, and the packed and unpacked byte.
    "one-bit fields, and packing and unpacking whole bytes"_test = [] {
        const auto                items = randomItems(64UZ, 8U, 0xB7E151628AED2A6BULL);
        std::vector<std::uint8_t> lowBits(items.size());
        std::ranges::transform(items, lowBits.begin(), [](std::uint8_t value) { return static_cast<std::uint8_t>(value & 1U); });

        for (const BitOrder orderIn : kOrders) {
            for (const BitOrder orderOut : kOrders) {
                const auto copied = throughKernel(configured(1U, 1U, orderIn, orderOut), items);
                expect(eq(firstDifference(copied, lowBits), lowBits.size())) << std::format("{} must copy the low bit of every item", describe(1U, 1U, orderIn, orderOut));
            }
        }

        const std::array<std::uint8_t, 8> stream{1U, 0U, 1U, 1U, 0U, 0U, 1U, 0U};
        expect(eq(static_cast<unsigned>(throughKernel(configured(1U, 8U, BitOrder::MsbFirst, BitOrder::MsbFirst), stream).at(0UZ)), 0xB2U)) << "1,0,1,1,0,0,1,0 packs msb_first to 0xB2";
        expect(eq(static_cast<unsigned>(throughKernel(configured(1U, 8U, BitOrder::MsbFirst, BitOrder::LsbFirst), stream).at(0UZ)), 0x4DU)) << "and lsb_first to 0x4D, its reversal";

        const std::array<std::uint8_t, 2> source{0xF5U, 0x08U};
        const std::array<std::uint8_t, 4> nibblesMsb{0xFU, 0x5U, 0x0U, 0x8U};
        const std::array<std::uint8_t, 4> nibblesLsb{0x5U, 0xFU, 0x8U, 0x0U};
        expect(eq(firstDifference(throughKernel(configured(8U, 4U, BitOrder::MsbFirst, BitOrder::MsbFirst), source), nibblesMsb), 4UZ)) << "0xF5 0x08 to msb_first nibbles";
        expect(eq(firstDifference(throughKernel(configured(8U, 4U, BitOrder::LsbFirst, BitOrder::LsbFirst), source), nibblesLsb), 4UZ)) << "0xF5 0x08 to lsb_first nibbles";

        const std::array<std::uint8_t, 16> bitsMsb{1U, 1U, 1U, 1U, 0U, 1U, 0U, 1U, 0U, 0U, 0U, 0U, 1U, 0U, 0U, 0U};
        const std::array<std::uint8_t, 16> bitsLsb{1U, 0U, 1U, 0U, 1U, 1U, 1U, 1U, 0U, 0U, 0U, 1U, 0U, 0U, 0U, 0U};
        expect(eq(firstDifference(throughKernel(configured(8U, 1U, BitOrder::MsbFirst, BitOrder::MsbFirst), source), bitsMsb), 16UZ)) << "0xF5 0x08 to msb_first bits";
        expect(eq(firstDifference(throughKernel(configured(8U, 1U, BitOrder::LsbFirst, BitOrder::LsbFirst), source), bitsLsb), 16UZ)) << "0xF5 0x08 to lsb_first bits";
    };

    // Three bytes are exactly one period at eight to three, and the inverse recovers them.
    "eight to three over three bytes, and back"_test = [] {
        const std::array<std::uint8_t, 3> source{0xC5U, 0x3AU, 0x9FU};
        const std::array<std::uint8_t, 8> msbSymbols{6U, 1U, 2U, 3U, 5U, 2U, 3U, 7U};
        const std::array<std::uint8_t, 8> lsbSymbols{5U, 0U, 3U, 5U, 3U, 6U, 7U, 4U};

        const auto msb = throughKernel(configured(8U, 3U, BitOrder::MsbFirst, BitOrder::MsbFirst), source);
        const auto lsb = throughKernel(configured(8U, 3U, BitOrder::LsbFirst, BitOrder::LsbFirst), source);
        expect(eq(firstDifference(msb, msbSymbols), 8UZ)) << "0xC5 0x3A 0x9F to msb_first octal 6 1 2 3 5 2 3 7";
        expect(eq(firstDifference(lsb, lsbSymbols), 8UZ)) << "0xC5 0x3A 0x9F to lsb_first octal 5 0 3 5 3 6 7 4";

        expect(eq(firstDifference(throughKernel(configured(3U, 8U, BitOrder::MsbFirst, BitOrder::MsbFirst), msb), source), 3UZ)) << "and back, msb_first";
        expect(eq(firstDifference(throughKernel(configured(3U, 8U, BitOrder::LsbFirst, BitOrder::LsbFirst), lsb), source), 3UZ)) << "and back, lsb_first";
    };

    // Periods are independent, so how a stream is divided into calls cannot reach the result.
    "the output does not depend on how the stream is divided into calls"_test = [] {
        constexpr std::size_t kPeriodsPerCall[] = {1UZ, 3UZ, 17UZ};
        for (const auto& [bitsIn, bitsOut] : {std::pair{8U, 3U}, std::pair{3U, 8U}}) {
            for (const BitOrder order : kOrders) {
                const BitRepack   cfg      = configured(bitsIn, bitsOut, order, order);
                const std::size_t inChunk  = inChunkOf(cfg);
                const std::size_t outChunk = outChunkOf(cfg);
                const std::size_t periods  = 419UZ;
                const auto        source   = randomItems(periods * inChunk, bitsIn, 0xCBBB9D5DC1059ED8ULL + bitsIn);
                const auto        whole    = throughKernel(cfg, source);

                for (const std::size_t perCall : kPeriodsPerCall) {
                    std::vector<std::uint8_t> pieced;
                    pieced.reserve(whole.size());
                    for (std::size_t done = 0UZ; done < periods; done += perCall) {
                        const std::size_t         take = std::min(perCall, periods - done);
                        std::vector<std::uint8_t> out(take * outChunk, std::uint8_t{0xAA});
                        repack(cfg, std::span<const std::uint8_t>(source).subspan(done * inChunk, take * inChunk), out);
                        pieced.insert(pieced.end(), out.begin(), out.end());
                    }
                    expect(eq(firstDifference(pieced, whole), whole.size())) << std::format("{} at {} periods per call differs", describe(bitsIn, bitsOut, order, order), perCall);
                }

                const std::size_t         itemsPerCall = 4096UZ - (4096UZ % inChunk);
                std::vector<std::uint8_t> pieced;
                pieced.reserve(whole.size());
                for (std::size_t done = 0UZ; done < source.size(); done += itemsPerCall) {
                    const std::size_t         take = std::min(itemsPerCall, source.size() - done);
                    std::vector<std::uint8_t> out(take / inChunk * outChunk, std::uint8_t{0xAA});
                    repack(cfg, std::span<const std::uint8_t>(source).subspan(done, take), out);
                    pieced.insert(pieced.end(), out.begin(), out.end());
                }
                expect(eq(firstDifference(pieced, whole), whole.size())) << std::format("{} at 4096-item calls differs", describe(bitsIn, bitsOut, order, order));
            }
        }
    };

    // The chunk counts a rate-changing block declares, and the period they come from.
    "the derived chunk counts and period"_test = [] {
        struct Row {
            unsigned bitsIn;
            unsigned bitsOut;
            unsigned inChunk;
            unsigned outChunk;
            unsigned period;
        };
        constexpr Row kRows[] = {{1U, 8U, 8U, 1U, 8U}, {8U, 1U, 1U, 8U, 8U}, {8U, 3U, 3U, 8U, 24U}, {3U, 8U, 8U, 3U, 24U}, {2U, 8U, 4U, 1U, 8U}, //
            {6U, 4U, 2U, 3U, 12U}, {5U, 7U, 7U, 5U, 35U}, {8U, 7U, 7U, 8U, 56U}, {8U, 8U, 1U, 1U, 8U}, {1U, 1U, 1U, 1U, 1U}};

        for (const Row& row : kRows) {
            const BitRepack cfg = configured(row.bitsIn, row.bitsOut, BitOrder::MsbFirst, BitOrder::MsbFirst);
            expect(eq(static_cast<unsigned>(cfg.inChunk), row.inChunk)) << std::format("({} -> {}) input chunk", row.bitsIn, row.bitsOut);
            expect(eq(static_cast<unsigned>(cfg.outChunk), row.outChunk)) << std::format("({} -> {}) output chunk", row.bitsIn, row.bitsOut);
            expect(eq(static_cast<unsigned>(cfg.period), row.period)) << std::format("({} -> {}) period", row.bitsIn, row.bitsOut);
        }

        std::size_t worst = 0UZ;
        for (unsigned bitsIn = 1U; bitsIn <= 8U; ++bitsIn) {
            for (unsigned bitsOut = 1U; bitsOut <= 8U; ++bitsOut) {
                worst = std::max(worst, static_cast<std::size_t>(configured(bitsIn, bitsOut, BitOrder::MsbFirst, BitOrder::MsbFirst).period));
            }
        }
        expect(eq(worst, gr::digital::kMaxRepackPeriod)) << "the shift table is sized for the longest period in the space";

        const BitRepack defaults{};
        const BitRepack rebuilt = configured(defaults.bitsIn, defaults.bitsOut, defaults.orderIn, defaults.orderOut);
        expect(eq(static_cast<unsigned>(defaults.inChunk), static_cast<unsigned>(rebuilt.inChunk)));
        expect(eq(static_cast<unsigned>(defaults.outChunk), static_cast<unsigned>(rebuilt.outChunk)));
        expect(eq(static_cast<unsigned>(defaults.period), static_cast<unsigned>(rebuilt.period)));
        expect(that % std::ranges::equal(defaults.moves, rebuilt.moves)) << "a default-constructed configuration already carries the table its defaults imply";
    };

    // A width outside the stated set is rejected, and rejects without half-applying.
    "validation rejects, and leaves the configuration intact"_test = [] {
        BitRepack cfg = configured(3U, 5U, BitOrder::LsbFirst, BitOrder::MsbFirst);

        for (const unsigned bad : {0U, 9U, 16U}) {
            expect(throws<std::invalid_argument>([&] { configureRepack(cfg, bad, 8U, BitOrder::MsbFirst, BitOrder::MsbFirst); })) << std::format("bitsIn = {}", bad);
            expect(throws<std::invalid_argument>([&] { configureRepack(cfg, 8U, bad, BitOrder::MsbFirst, BitOrder::MsbFirst); })) << std::format("bitsOut = {}", bad);
        }
        expect(eq(static_cast<unsigned>(cfg.bitsIn), 3U)) << "a rejected width leaves the previous configuration standing";
        expect(eq(static_cast<unsigned>(cfg.bitsOut), 5U));
        expect(eq(static_cast<unsigned>(cfg.inChunk), 5U));
        expect(eq(static_cast<unsigned>(cfg.outChunk), 3U));
        expect(that % (cfg.orderIn == BitOrder::LsbFirst));
    };

    // A stream value is masked to its field, so no stream value can change the answer or stop a graph.
    "an item's unused high bits are masked, not read"_test = [] {
        constexpr std::size_t kItems = 100000UZ;
        struct Case {
            unsigned bitsIn;
            unsigned bitsOut;
        };
        constexpr Case kCases[] = {{3U, 8U}, {8U, 3U}, {1U, 8U}, {8U, 1U}, {5U, 7U}, {6U, 4U}, {2U, 3U}, {8U, 8U}};

        for (const Case& shape : kCases) {
            for (const BitOrder orderIn : kOrders) {
                const BitRepack cfg    = configured(shape.bitsIn, shape.bitsOut, orderIn, BitOrder::MsbFirst);
                const auto      masked = randomItems(kItems - (kItems % inChunkOf(cfg)), shape.bitsIn, 0x510E527FADE682D1ULL + shape.bitsIn);

                const std::uint8_t        aboveField = static_cast<std::uint8_t>(~((1U << shape.bitsIn) - 1U));
                std::vector<std::uint8_t> dirty(masked.size());
                std::ranges::transform(masked, dirty.begin(), [aboveField](std::uint8_t value) { return static_cast<std::uint8_t>(value | aboveField); });

                const auto fromMasked = throughKernel(cfg, masked);
                const auto fromDirty  = throughKernel(cfg, dirty);
                expect(eq(firstDifference(fromMasked, fromDirty), fromMasked.size())) << std::format("{} reads a bit above its field", describe(shape.bitsIn, shape.bitsOut, orderIn, BitOrder::MsbFirst));

                const unsigned limit = 1U << shape.bitsOut;
                expect(std::ranges::all_of(fromDirty, [limit](std::uint8_t value) { return value < limit; })) << std::format("{} wrote above its output field", describe(shape.bitsIn, shape.bitsOut, orderIn, BitOrder::MsbFirst));
            }
        }

        const BitRepack                 cfg = configured(3U, 8U, BitOrder::MsbFirst, BitOrder::MsbFirst);
        const std::vector<std::uint8_t> ones(inChunkOf(cfg), std::uint8_t{0xFFU});
        const std::vector<std::uint8_t> threeBits(inChunkOf(cfg), std::uint8_t{0x07U});
        expect(eq(firstDifference(throughKernel(cfg, ones), throughKernel(cfg, threeBits)), outChunkOf(cfg))) << "all 0xFF and all 0x07 are the same three-bit stream";
    };

    // The tail, which is the only place a partial output item exists.
    "the trailing input, flushed and dropped"_test = [] {
        const std::array<std::uint8_t, 2> trailing{0xC5U, 0xBBU}; // 16 stranded bits, five whole symbols and one bit over

        const BitRepack                   msb = configured(8U, 3U, BitOrder::MsbFirst, BitOrder::MsbFirst);
        const std::array<std::uint8_t, 6> expectedMsb{6U, 1U, 3U, 3U, 5U, 4U};
        std::vector<std::uint8_t>         padded(6UZ, std::uint8_t{0xAA});
        expect(eq(repackTail(msb, trailing, padded, true), 6UZ)) << "flushing emits the five whole symbols and one padded";
        expect(eq(firstDifference(padded, expectedMsb), 6UZ)) << "the flushed symbols, msb_first";
        expect(eq(padded[5UZ] & 0x03U, 0U)) << "msb_first output pads the low positions";

        std::vector<std::uint8_t> dropped(6UZ, std::uint8_t{0xAA});
        expect(eq(repackTail(msb, trailing, dropped, false), 5UZ)) << "dropping emits the whole symbols only";
        expect(eq(firstDifference(std::span(dropped).first(5UZ), std::span(expectedMsb).first(5UZ)), 5UZ));
        expect(eq(static_cast<unsigned>(dropped[5UZ]), 0xAAU)) << "nothing past the returned count is written";

        const BitRepack                   lsb = configured(8U, 3U, BitOrder::LsbFirst, BitOrder::LsbFirst);
        const std::array<std::uint8_t, 6> expectedLsb{5U, 0U, 7U, 5U, 3U, 1U};
        std::vector<std::uint8_t>         lsbPadded(6UZ, std::uint8_t{0xAA});
        expect(eq(repackTail(lsb, trailing, lsbPadded, true), 6UZ));
        expect(eq(firstDifference(lsbPadded, expectedLsb), 6UZ)) << "the flushed symbols, lsb_first";
        expect(eq(lsbPadded[5UZ] & 0x06U, 0U)) << "lsb_first output pads the high positions";

        // the real bit of a padded item is the stream's own next bit, at the end the order names
        expect(eq((padded[5UZ] >> 2U) & 1U, static_cast<unsigned>(toBits(trailing, 8U, BitOrder::MsbFirst)[15UZ])));
        expect(eq(lsbPadded[5UZ] & 1U, static_cast<unsigned>(toBits(trailing, 8U, BitOrder::LsbFirst)[15UZ])));

        // the longest trailing input each shape admits, the items it flushes, and the pad it leaves
        struct Row {
            unsigned bitsIn;
            unsigned bitsOut;
            unsigned trailing;
            unsigned whole;
            unsigned padBits;
        };
        constexpr Row kRows[] = {{1U, 8U, 7U, 0U, 1U}, {8U, 3U, 2U, 5U, 2U}, {3U, 8U, 7U, 2U, 3U}, {5U, 7U, 6U, 4U, 5U}, //
            {7U, 5U, 4U, 5U, 2U}, {2U, 3U, 2U, 1U, 2U}, {6U, 4U, 1U, 1U, 2U}, {4U, 8U, 1U, 0U, 4U}};

        for (const Row& row : kRows) {
            const BitRepack cfg = configured(row.bitsIn, row.bitsOut, BitOrder::MsbFirst, BitOrder::MsbFirst);
            expect(eq(static_cast<unsigned>(cfg.inChunk), row.trailing + 1U)) << std::format("({} -> {}) the longest trailing input is one item short of a period", row.bitsIn, row.bitsOut);

            const auto                source = randomItems(row.trailing, row.bitsIn, 0x9B05688C2B3E6C1FULL + row.bitsIn * 8U + row.bitsOut);
            std::vector<std::uint8_t> flushed(row.whole + 1UZ, std::uint8_t{0xAA});
            expect(eq(repackTail(cfg, source, flushed, true), std::size_t{row.whole} + 1UZ)) << std::format("({} -> {}) flushed count", row.bitsIn, row.bitsOut);

            const unsigned padMask = (1U << row.padBits) - 1U;
            expect(eq(flushed[row.whole] & padMask, 0U)) << std::format("({} -> {}) the pad bits are zero", row.bitsIn, row.bitsOut);
            expect(eq(firstDifference(std::span(flushed).first(row.whole), byDefinition(source, row.bitsIn, row.bitsOut, BitOrder::MsbFirst, BitOrder::MsbFirst)), std::size_t{row.whole})) //
                << std::format("({} -> {}) the whole items match the definition", row.bitsIn, row.bitsOut);

            std::vector<std::uint8_t> truncated(row.whole + 1UZ, std::uint8_t{0xAA});
            expect(eq(repackTail(cfg, source, truncated, false), std::size_t{row.whole})) << std::format("({} -> {}) dropped count", row.bitsIn, row.bitsOut);
            expect(eq(static_cast<unsigned>(truncated[row.whole]), 0xAAU)) << std::format("({} -> {}) dropping writes no partial item", row.bitsIn, row.bitsOut);
        }

        // where one input item is a whole period there is no trailing input, and the setting is inert
        for (const auto& [bitsIn, bitsOut] : {std::pair{8U, 1U}, std::pair{8U, 4U}, std::pair{8U, 8U}, std::pair{6U, 3U}}) {
            const BitRepack cfg = configured(bitsIn, bitsOut, BitOrder::MsbFirst, BitOrder::MsbFirst);
            expect(eq(static_cast<unsigned>(cfg.inChunk), 1U)) << std::format("({} -> {}) one input item is a whole period", bitsIn, bitsOut);
            std::vector<std::uint8_t> out(4UZ, std::uint8_t{0xAA});
            expect(eq(repackTail(cfg, {}, out, true), 0UZ)) << "there is nothing to flush";
            expect(eq(repackTail(cfg, {}, out, false), 0UZ));
        }
    };

    // The offset map, which is integer arithmetic and nothing else.
    "the offset map floors, and never passes through a float"_test = [] {
        constexpr std::uint64_t kPackOffsets[]  = {0ULL, 3ULL, 7ULL, 8ULL, 15ULL, 16ULL, 23ULL};
        constexpr std::uint64_t kPackMapped[]   = {0ULL, 0ULL, 0ULL, 1ULL, 1ULL, 2ULL, 2ULL};
        constexpr std::uint64_t kEightToThree[] = {0ULL, 2ULL, 5ULL, 8ULL, 10ULL, 13ULL, 16ULL, 18ULL, 21ULL};
        constexpr std::uint64_t kThreeToEight[] = {0ULL, 0ULL, 0ULL, 1ULL, 1ULL, 1ULL, 2ULL, 2ULL, 3ULL, 3ULL};

        for (std::size_t i = 0UZ; i < std::size(kPackOffsets); ++i) {
            expect(eq(mapRepackedOffset(kPackOffsets[i], 1ULL, 8ULL), kPackMapped[i])) << std::format("packing eight, input offset {}", kPackOffsets[i]);
        }
        for (std::uint64_t t = 0ULL; t < 6ULL; ++t) {
            expect(eq(mapRepackedOffset(t, 8ULL, 1ULL), 8ULL * t)) << std::format("unpacking eight, input offset {}", t);
        }
        for (std::size_t t = 0UZ; t < std::size(kEightToThree); ++t) {
            expect(eq(mapRepackedOffset(t, 8ULL, 3ULL), kEightToThree[t])) << std::format("eight to three, input offset {}", t);
        }
        for (std::size_t t = 0UZ; t < std::size(kThreeToEight); ++t) {
            expect(eq(mapRepackedOffset(t, 3ULL, 8ULL), kThreeToEight[t])) << std::format("three to eight, input offset {}", t);
        }
        for (std::uint64_t t = 0ULL; t < 32ULL; ++t) {
            expect(eq(mapRepackedOffset(t, 8ULL, 8ULL), t)) << "eight to eight is the identity map";
        }

        // the split form against the direct product, wherever the direct product still fits
        std::mt19937_64                              engine{0x1F83D9ABFB41BD6BULL};
        std::uniform_int_distribution<std::uint64_t> offsets{0ULL, (1ULL << 58) - 1ULL};
        std::uniform_int_distribution<std::uint64_t> widths{1ULL, 8ULL};
        std::size_t                                  disagreements = 0UZ;
        for (std::size_t trial = 0UZ; trial < 200000UZ; ++trial) {
            const std::uint64_t offset  = offsets(engine);
            const std::uint64_t bitsIn  = widths(engine);
            const std::uint64_t bitsOut = widths(engine);
            disagreements += mapRepackedOffset(offset, bitsIn, bitsOut) == offset * bitsIn / bitsOut ? 0UZ : 1UZ;
        }
        expect(eq(disagreements, 0UZ)) << "the split form and the direct product agree wherever both are defined";

        // above 2^53 the integers are still exact and a double is not
        constexpr std::uint64_t kBeyondDouble = 9007199254740993ULL;
        expect(eq(mapRepackedOffset(kBeyondDouble, 1ULL, 3ULL), 3002399751580331ULL)) << "packing three bits, exactly";
        expect(eq(mapRepackedOffset(kBeyondDouble, 3ULL, 1ULL), 3ULL * kBeyondDouble)) << "unpacking three bits, exactly";
        expect(that % (static_cast<std::uint64_t>(static_cast<double>(kBeyondDouble) / 3.0) != mapRepackedOffset(kBeyondDouble, 1ULL, 3ULL))) //
            << "a double path is off by one here, which is why an offset never becomes one";

        expect(eq(mapRepackedOffset(~0ULL, 1ULL, 8ULL), (~0ULL) / 8ULL)) << "the offset space is exhausted before the decomposition is";
    };
};

int main() { /* tests are automatically registered and run */ }
