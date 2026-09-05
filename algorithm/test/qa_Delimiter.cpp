#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/algorithm/digital/Delimiter.hpp>

namespace {

using gr::digital::DelimiterConfig;
using gr::digital::FrameOpen;
using gr::digital::FrameOutcome;
using gr::digital::ScanEvent;
using gr::digital::Transparency;

struct Rng {
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;

    [[nodiscard]] std::uint64_t next() noexcept {
        state ^= state << 13U;
        state ^= state >> 7U;
        state ^= state << 17U;
        return state;
    }
    [[nodiscard]] std::size_t below(std::size_t bound) noexcept { return next() % bound; }
};

/// One item per '0'/'1' character, which is what a bit-level stream carries.
[[nodiscard]] std::vector<std::uint8_t> itemsOf(std::string_view bits) {
    std::vector<std::uint8_t> items;
    items.reserve(bits.size());
    for (const char character : bits) {
        items.push_back(static_cast<std::uint8_t>(character == '1' ? 1 : 0));
    }
    return items;
}

/// What one drive of the scanner produced, item by item.
template<typename T>
struct Run {
    std::vector<std::vector<T>>    payloads{};
    std::vector<FrameOutcome>      outcomes{};
    std::vector<std::size_t>       starts{};
    std::vector<std::size_t>       removals{};
    std::vector<std::size_t>       matches{};
    gr::digital::DelimiterCounters counters{};
};

/// Drives the scanner one item at a time, recording every match position and every record.
template<typename T>
[[nodiscard]] Run<T> drive(DelimiterConfig cfg, std::span<const T> items) {
    gr::digital::configureDelimiter(cfg);
    gr::digital::DelimiterScanner<T> scanner;
    scanner.prepare(std::move(cfg));

    Run<T> result;
    for (std::size_t i = 0UZ; i < items.size(); ++i) {
        const gr::digital::ScanStep step = scanner.push(items[i]);
        if (step.matched) {
            result.matches.push_back(i);
        }
        if (step.event == ScanEvent::Record) {
            const std::span<const T> frame = scanner.frame();
            result.payloads.emplace_back(frame.begin(), frame.end());
            result.outcomes.push_back(scanner.frameOutcome());
            result.starts.push_back(scanner.frameStart());
            result.removals.push_back(scanner.frameRemoved());
        }
    }
    result.counters = scanner.counters;
    return result;
}

/// The bit-stuffing encoder the decoder is the inverse of: a zero is inserted after every run of @p k ones.
[[nodiscard]] std::vector<std::uint8_t> stuff(std::span<const std::uint8_t> bits, unsigned k) {
    std::vector<std::uint8_t> coded;
    coded.reserve(bits.size() + bits.size() / k + 1UZ);
    unsigned ones = 0U;
    for (const std::uint8_t bit : bits) {
        coded.push_back(bit);
        if (bit != 0U) {
            ++ones;
            if (ones == k) {
                coded.push_back(0U);
                ones = 0U;
            }
        } else {
            ones = 0U;
        }
    }
    return coded;
}

/// The delimiter for a stuffing width: one more one than coded data can carry, so it is unforgeable and invariant.
[[nodiscard]] std::string flagFor(unsigned k) { return "0" + std::string(k + 1UZ, '1') + "0"; }

/// The HDLC configuration of anchor A, with the bound the anchor's frames fit inside.
[[nodiscard]] DelimiterConfig hdlcAt(std::size_t bound) {
    DelimiterConfig cfg = gr::digital::hdlc();
    cfg.maxPayloadItems = bound;
    return cfg;
}

} // namespace

const boost::ut::suite<"delimiter scanner"> delimiterTests = [] {
    using namespace boost::ut;

    "the HDLC frame, item by item"_test = [] {
        // anchor A: flag, then 0x3F transmitted LSB first with one stuffed zero, then flag
        const std::vector<std::uint8_t> stream = itemsOf(std::string("01111110") + "111110100" + "01111110");
        expect(eq(stream.size(), 25UZ));

        const Run<std::uint8_t> run = drive<std::uint8_t>(hdlcAt(1024UZ), std::span<const std::uint8_t>(stream));
        expect(that % (run.matches == std::vector<std::size_t>{7UZ, 24UZ})) << "the register equals the flag at exactly two of the eighteen full windows";
        expect(eq(run.payloads.size(), 1UZ));
        if (run.payloads.size() == 1UZ) {
            const std::vector<std::uint8_t>& payload = run.payloads[0UZ];
            expect(eq(payload.size(), 8UZ));
            const std::vector<std::uint8_t> wanted{1U, 1U, 1U, 1U, 1U, 1U, 0U, 0U};
            for (std::size_t i = 0UZ; i < std::min(payload.size(), wanted.size()); ++i) {
                expect(eq(payload[i], wanted[i])) << std::format("payload item {}", i);
            }
            expect(eq(run.starts[0UZ], 8UZ));
            expect(eq(run.removals[0UZ], 1UZ));
            expect(run.outcomes[0UZ] == FrameOutcome::Emitted);
        }

        // extended to a second frame whose payload is 0x00 and needs no stuffing
        const std::vector<std::uint8_t> two = itemsOf(std::string("01111110") + "111110100" + "01111110" + "00000000" + "01111110");
        expect(eq(two.size(), 41UZ));
        const Run<std::uint8_t> both = drive<std::uint8_t>(hdlcAt(1024UZ), std::span<const std::uint8_t>(two));
        expect(that % (both.matches == std::vector<std::size_t>{7UZ, 24UZ, 40UZ}));
        expect(eq(both.payloads.size(), 2UZ));
        if (both.payloads.size() == 2UZ) {
            expect(that % (both.starts == std::vector<std::size_t>{8UZ, 25UZ}));
            expect(that % (both.removals == std::vector<std::size_t>{1UZ, 0UZ}));
            expect(that % (both.payloads[1UZ] == std::vector<std::uint8_t>(8UZ, 0U)));
        }
    };

    "an abort run no longer than the flag's own is refused"_test = [] {
        // the scanner counts the closing flag's ones while the frame is still open, so an abort at six abandons every
        // HDLC frame at the delimiter that closes it
        const auto atAbort = [](unsigned abortOnes) {
            DelimiterConfig cfg = hdlcAt(1024UZ);
            cfg.abortOnes       = abortOnes;
            gr::digital::configureDelimiter(cfg);
            return cfg;
        };
        expect(throws<std::invalid_argument>([&atAbort] { atAbort(6U); })) << "six is the flag's own run of ones";
        expect(nothrow([&atAbort] { atAbort(7U); })) << "seven, HDLC's abort, exceeds it";

        // anchor A again, under the abort the configuration does accept
        const std::vector<std::uint8_t> stream = itemsOf(std::string("01111110") + "111110100" + "01111110");
        const Run<std::uint8_t>         run    = drive<std::uint8_t>(atAbort(7U), std::span<const std::uint8_t>(stream));
        expect(eq(run.payloads.size(), 1UZ));
        expect(eq(run.counters.nAborts, 0ULL));
        if (run.payloads.size() == 1UZ) {
            expect(run.outcomes[0UZ] == FrameOutcome::Emitted);
        }
    };

    "the shared flag is idle rather than a seven-item frame"_test = [] {
        // anchor B: three flags sharing their common zero, 22 items
        const std::vector<std::uint8_t> shared = itemsOf(std::string("01111110") + "1111110" + "1111110");
        expect(eq(shared.size(), 22UZ));
        const Run<std::uint8_t> run = drive<std::uint8_t>(hdlcAt(1024UZ), std::span<const std::uint8_t>(shared));
        expect(that % (run.matches == std::vector<std::size_t>{7UZ, 14UZ, 21UZ}));
        expect(eq(run.payloads.size(), 0UZ)) << "the delay ring is cleared before any of its items is evicted";
        expect(eq(run.counters.nIdleDelimiters, 2ULL));

        // the non-shared form reaches the same answer by the same mechanism, with seven ring items instead of six
        const std::vector<std::uint8_t> apart = itemsOf(std::string("01111110") + "01111110" + "01111110");
        expect(eq(apart.size(), 24UZ));
        const Run<std::uint8_t> spaced = drive<std::uint8_t>(hdlcAt(1024UZ), std::span<const std::uint8_t>(apart));
        expect(that % (spaced.matches == std::vector<std::size_t>{7UZ, 15UZ, 23UZ}));
        expect(eq(spaced.payloads.size(), 0UZ));
        expect(eq(spaced.counters.nIdleDelimiters, 2ULL));
    };

    "the SLIP frame, item by item"_test = [] {
        // anchor C: a payload carrying both reserved values, encoded and framed
        const std::vector<std::uint8_t> stream{0xC0U, 0xDBU, 0xDCU, 0x01U, 0xDBU, 0xDDU, 0x02U, 0xC0U};
        DelimiterConfig                 cfg = gr::digital::slip();
        cfg.maxPayloadItems                 = 256UZ;

        const Run<std::uint8_t> run = drive<std::uint8_t>(cfg, std::span<const std::uint8_t>(stream));
        expect(that % (run.matches == std::vector<std::size_t>{0UZ, 7UZ})) << "the register never held 0xC0 between the two delimiters";
        expect(eq(run.payloads.size(), 1UZ));
        if (run.payloads.size() == 1UZ) {
            expect(that % (run.payloads[0UZ] == std::vector<std::uint8_t>{0xC0U, 0x01U, 0xDBU, 0x02U}));
            expect(eq(run.starts[0UZ], 1UZ));
            expect(eq(run.removals[0UZ], 2UZ));
        }
    };

    "an escaped item with no map entry is emitted and counted"_test = [] {
        // anchor D: the frame is emitted, not refused; whatever checks the payload is what judges it
        const std::vector<std::uint8_t> stream{0xC0U, 0xDBU, 0x41U, 0xC0U};
        DelimiterConfig                 cfg = gr::digital::slip();
        cfg.maxPayloadItems                 = 256UZ;

        const Run<std::uint8_t> run = drive<std::uint8_t>(cfg, std::span<const std::uint8_t>(stream));
        expect(eq(run.payloads.size(), 1UZ));
        if (run.payloads.size() == 1UZ) {
            expect(that % (run.payloads[0UZ] == std::vector<std::uint8_t>{0x41U}));
            expect(eq(run.removals[0UZ], 1UZ));
            expect(run.outcomes[0UZ] == FrameOutcome::Emitted);
        }
        expect(eq(run.counters.nEscapeViolations, 1ULL));
    };

    "the unforgeability check refuses what is provably wrong and warns about the rest"_test = [] {
        const auto stuffed = [](std::string_view delimiter, unsigned k) {
            DelimiterConfig cfg;
            cfg.endDelimiter    = std::string(delimiter);
            cfg.transparency    = Transparency::BitStuffing;
            cfg.stuffAfterOnes  = k;
            cfg.abortOnes       = k + 2U;
            cfg.maxPayloadItems = 256UZ;
            gr::digital::configureDelimiter(cfg);
        };
        expect(nothrow([&stuffed] { stuffed("01111110", 5U); })) << "the HDLC flag is invariant under the decoder and carries six ones";
        expect(throws([&stuffed] { stuffed("0111110", 5U); })) << "a zero arriving at exactly five ones is removed, so the delimiter is not invariant";
        expect(throws([&stuffed] { stuffed("0111010", 5U); })) << "no run longer than five, so a conforming encoder can produce it";
        expect(throws([&stuffed] { stuffed("101111110", 5U); })) << "invariant from a zero entering count but not behind four payload ones, whose fifth one makes the second bit a stuffed zero";
        expect(nothrow([&stuffed] { stuffed("0111111", 5U); })) << "a leading zero absorbs any entering count before the run begins, so trailing ones alone do not break invariance";

        const auto escaped = [](std::uint8_t delimiterValue) {
            DelimiterConfig cfg;
            cfg.endDelimiter = std::string();
            for (unsigned bit = 8U; bit-- > 0U;) {
                cfg.endDelimiter.push_back(((delimiterValue >> bit) & 1U) != 0U ? '1' : '0');
            }
            cfg.bitsPerItem     = 8U;
            cfg.transparency    = Transparency::ByteEscape;
            cfg.escapeItem      = 0xDBU;
            cfg.escapeMap       = {{0xDCU, 0xC0U}, {0xDDU, 0xDBU}};
            cfg.maxPayloadItems = 256UZ;
            gr::digital::configureDelimiter(cfg);
            return cfg;
        };
        expect(throws([&escaped] { std::ignore = escaped(0xDBU); })) << "the introducer would be removed from inside the delimiter";
        expect(throws([&escaped] { std::ignore = escaped(0xDCU); })) << "an escaped value belongs to a pair rather than standing alone";
        expect(nothrow([&escaped] { std::ignore = escaped(0xC0U); }));
        expect(gr::digital::unforgeabilityWarning(escaped(0xC0U)).empty()) << "0xC0 is an original in the map, so a conforming encoder escapes it";
        expect(!gr::digital::unforgeabilityWarning(escaped(0x41U)).empty()) << "0x41 is not an original, so the local half of the guarantee is absent";

        // anchor F: the four published framings all pass, and only the escape-based ones can be warned about
        expect(nothrow([] { std::ignore = gr::digital::hdlc(); }));
        expect(nothrow([] { std::ignore = gr::digital::slip(); }));
        expect(nothrow([] { std::ignore = gr::digital::pppAsync(); }));
        expect(nothrow([] { std::ignore = gr::digital::nmea0183(); }));
        expect(gr::digital::unforgeabilityWarning(gr::digital::hdlc()).empty());
        expect(gr::digital::unforgeabilityWarning(gr::digital::slip()).empty());
        expect(gr::digital::unforgeabilityWarning(gr::digital::pppAsync()).empty());
        expect(gr::digital::unforgeabilityWarning(gr::digital::nmea0183()).empty());
    };

    "bit stuffing is unforgeable"_test = [] {
        constexpr std::size_t     kBits = 500'000UZ;
        Rng                       rng{};
        std::vector<std::uint8_t> bits(kBits);
        for (std::uint8_t& bit : bits) {
            bit = static_cast<std::uint8_t>(rng.next() & 1ULL);
        }
        const std::vector<std::uint8_t> coded = stuff(std::span<const std::uint8_t>(bits), 5U);

        std::size_t ones    = 0UZ;
        std::size_t longest = 0UZ;
        for (const std::uint8_t bit : coded) {
            ones    = bit != 0U ? ones + 1UZ : 0UZ;
            longest = std::max(longest, ones);
        }
        expect(eq(longest, 5UZ)) << "coded data carries at most five consecutive ones, so the flag's six cannot occur";

        const Run<std::uint8_t> run = drive<std::uint8_t>(hdlcAt(1024UZ), std::span<const std::uint8_t>(coded));
        expect(eq(run.matches.size(), 0UZ)) << "and the scanner finds no flag anywhere in it";
    };

    "the expansion matches the closed form"_test = [] {
        for (const unsigned k : {3U, 4U, 5U, 6U, 8U}) {
            constexpr std::size_t           kOnes = 100'000UZ;
            const std::vector<std::uint8_t> ones(kOnes, 1U);
            const std::vector<std::uint8_t> coded = stuff(std::span<const std::uint8_t>(ones), k);
            expect(eq(coded.size() - kOnes, kOnes / k)) << std::format("all ones at k = {} stuffs exactly floor(n/k)", k);
        }

        // The closed form is exact; its variance under a renewal process is not, so the spread is estimated from
        // twenty independent batches and the mean is tested against three standard errors of that estimate.
        constexpr std::size_t kBatches   = 20UZ;
        constexpr std::size_t kBatchBits = 50'000UZ;
        Rng                   rng{};
        for (const unsigned k : {4U, 5U, 6U}) {
            std::vector<double> fractions;
            fractions.reserve(kBatches);
            for (std::size_t batch = 0UZ; batch < kBatches; ++batch) {
                std::vector<std::uint8_t> bits(kBatchBits);
                for (std::uint8_t& bit : bits) {
                    bit = static_cast<std::uint8_t>(rng.next() & 1ULL);
                }
                const std::vector<std::uint8_t> coded = stuff(std::span<const std::uint8_t>(bits), k);
                fractions.push_back(static_cast<double>(coded.size() - kBatchBits) / static_cast<double>(kBatchBits));
            }
            const double mean     = std::accumulate(fractions.begin(), fractions.end(), 0.0) / static_cast<double>(kBatches);
            double       variance = 0.0;
            for (const double value : fractions) {
                variance += (value - mean) * (value - mean);
            }
            variance /= static_cast<double>(kBatches - 1UZ);
            const double standardError = std::sqrt(variance / static_cast<double>(kBatches));
            const double theory        = 1.0 / (2.0 * (std::pow(2.0, static_cast<double>(k)) - 1.0));
            expect(std::abs(mean - theory) <= 3.0 * standardError) << std::format("k = {}: measured {:.6f} against {:.6f}, three standard errors being {:.6f}", k, mean, theory, 3.0 * standardError);
        }
    };

    "stuff then destuff is the identity"_test = [] {
        Rng rng{};
        for (const unsigned k : {3U, 4U, 5U, 6U, 8U}) {
            const std::string flag = flagFor(k);
            DelimiterConfig   cfg;
            cfg.endDelimiter    = flag;
            cfg.transparency    = Transparency::BitStuffing;
            cfg.stuffAfterOnes  = k;
            cfg.abortOnes       = 0U; // an all-ones payload is legal data here rather than an abort
            cfg.maxPayloadItems = 4096UZ;

            for (std::size_t which = 0UZ; which < 10'000UZ; ++which) {
                const std::size_t         length = 1UZ + rng.below(500UZ);
                std::vector<std::uint8_t> payload(length);
                for (std::uint8_t& bit : payload) {
                    bit = static_cast<std::uint8_t>(rng.next() & 1ULL);
                }
                const std::vector<std::uint8_t> coded = stuff(std::span<const std::uint8_t>(payload), k);

                std::vector<std::uint8_t> stream = itemsOf(flag);
                stream.insert(stream.end(), coded.begin(), coded.end());
                const std::vector<std::uint8_t> closing = itemsOf(flag);
                stream.insert(stream.end(), closing.begin(), closing.end());

                const Run<std::uint8_t> run = drive<std::uint8_t>(cfg, std::span<const std::uint8_t>(stream));
                if (run.payloads.size() != 1UZ) {
                    expect(eq(run.payloads.size(), 1UZ)) << std::format("k = {}, case {}, length {}", k, which, length);
                    break;
                }
                if (run.payloads[0UZ] != payload || run.removals[0UZ] != coded.size() - length) {
                    expect(that % (run.payloads[0UZ] == payload)) << std::format("k = {}, case {}", k, which);
                    expect(eq(run.removals[0UZ], coded.size() - length)) << std::format("k = {}, case {}: the stuffed length is the original plus the stuff count", k, which);
                    break;
                }
            }
        }
    };

    "escapeMapFromXor reproduces PPP's control character map"_test = [] {
        // the 34 originals, written out here rather than generated, so the helper is checked against the standard
        const std::vector<std::uint8_t> originals{0x00U, 0x01U, 0x02U, 0x03U, 0x04U, 0x05U, 0x06U, 0x07U, //
            0x08U, 0x09U, 0x0AU, 0x0BU, 0x0CU, 0x0DU, 0x0EU, 0x0FU,                                       //
            0x10U, 0x11U, 0x12U, 0x13U, 0x14U, 0x15U, 0x16U, 0x17U,                                       //
            0x18U, 0x19U, 0x1AU, 0x1BU, 0x1CU, 0x1DU, 0x1EU, 0x1FU, 0x7DU, 0x7EU};
        const std::vector<std::uint8_t> escapes{0x20U, 0x21U, 0x22U, 0x23U, 0x24U, 0x25U, 0x26U, 0x27U, //
            0x28U, 0x29U, 0x2AU, 0x2BU, 0x2CU, 0x2DU, 0x2EU, 0x2FU,                                     //
            0x30U, 0x31U, 0x32U, 0x33U, 0x34U, 0x35U, 0x36U, 0x37U,                                     //
            0x38U, 0x39U, 0x3AU, 0x3BU, 0x3CU, 0x3DU, 0x3EU, 0x3FU, 0x5DU, 0x5EU};

        const auto pairs = gr::digital::escapeMapFromXor(0x20U, std::span<const std::uint8_t>(originals));
        expect(eq(pairs.size(), 34UZ));
        for (std::size_t i = 0UZ; i < std::min(pairs.size(), escapes.size()); ++i) {
            expect(eq(pairs[i].first, escapes[i])) << std::format("pair {} escaped value", i);
            expect(eq(pairs[i].second, originals[i])) << std::format("pair {} original value", i);
            expect(neq(pairs[i].first, std::uint8_t{0x7EU})) << "no escaped value is the delimiter";
            expect(neq(pairs[i].first, std::uint8_t{0x7DU})) << "no escaped value is the introducer";
        }
        const DelimiterConfig ppp = gr::digital::pppAsync();
        expect(eq(ppp.escapeMap.size(), 34UZ));
        expect(gr::digital::unforgeabilityWarning(ppp).empty());
    };

    "delimiter length bounds, and the mask at the register's full width"_test = [] {
        const auto lengthOf = [](std::size_t bits, unsigned width) {
            DelimiterConfig cfg;
            cfg.endDelimiter    = std::string(bits, '0');
            cfg.bitsPerItem     = width;
            cfg.maxPayloadItems = 256UZ;
            gr::digital::configureDelimiter(cfg);
            return cfg;
        };
        expect(nothrow([&lengthOf] { std::ignore = lengthOf(64UZ, 1U); })) << "64 items at one bit each";
        expect(throws([&lengthOf] { std::ignore = lengthOf(65UZ, 1U); })) << "65 does not fit one 64-bit register";
        expect(nothrow([&lengthOf] { std::ignore = lengthOf(64UZ, 8U); })) << "8 items at eight bits each";
        expect(throws([&lengthOf] { std::ignore = lengthOf(72UZ, 8U); })) << "9 byte items do not fit either";
        expect(throws([&lengthOf] { std::ignore = lengthOf(12UZ, 8U); })) << "a delimiter must land on item boundaries";

        const DelimiterConfig full = lengthOf(64UZ, 1U);
        expect(eq(full.mask, ~0ULL)) << "at the register's full width the mask is all ones, computed without a 64-bit shift";
        expect(eq(full.endItems, 64UZ));

        const DelimiterConfig eight = lengthOf(8UZ, 1U);
        expect(eq(eight.mask, 0xFFULL));
    };

    "the framed HDLC frame, item by item"_test = [] {
        // anchor G: two payload bytes through the HDLC profile, hand-walked -- the flag, 0x7E least significant bit
        // first with one stuffed zero after its fifth one, 0xFF with one more, then the flag
        DelimiterConfig cfg = gr::digital::hdlc();
        cfg.maxPayloadItems = 1024UZ;
        cfg.packBits        = 8U;
        cfg.packOrder       = gr::digital::BitOrder::LsbFirst;
        gr::digital::framerConfigure(cfg);

        gr::digital::DelimiterEncoder encoder;
        encoder.prepare(cfg);

        const std::vector<std::uint8_t> bytes{0x7EU, 0xFFU};
        std::vector<std::uint8_t>       wire;
        const gr::digital::EncodeResult framed = encoder.encode(std::span<const std::uint8_t>(bytes), wire);

        const std::vector<std::uint8_t> wanted = itemsOf(std::string("01111110") + "011111010" + "111110111" + "01111110");
        expect(eq(wanted.size(), 34UZ));
        expect(eq(wire.size(), 34UZ));
        expect(that % (wire == wanted)) << "the wire items, compared one at a time against the hand-walked string";
        expect(eq(framed.emitted, 34UZ));
        expect(eq(framed.inserted, 2UZ));
        expect(eq(framed.forged, 0UZ));
        expect(eq(framed.emitted, 8UZ + 8UZ + 16UZ + framed.inserted)) << "opening delimiter, closing delimiter, unpacked payload items and the insertions";
        expect(eq(gr::digital::framedItemsBound(cfg, 16UZ), 35UZ)) << "the worst case is one insertion every five items, which this payload comes one short of";

        const Run<std::uint8_t> run = drive<std::uint8_t>(cfg, std::span<const std::uint8_t>(wire));
        expect(eq(run.payloads.size(), 1UZ));
        if (run.payloads.size() == 1UZ) {
            expect(that % (run.payloads[0UZ] == bytes)) << "and the scanner reads the two bytes back out of it";
            expect(eq(run.removals[0UZ], framed.inserted)) << "the removed count is the inserted count, frame by frame";
        }
    };

    "the encoder's ones counter starts a frame where the decoder starts it"_test = [] {
        // a delimiter ending in ones, which the invariance rule admits and which the two formulations of the counter
        // disagree about: the machine clears the count as the opening delimiter opens the frame, so the encoder does
        DelimiterConfig cfg;
        cfg.endDelimiter    = "0111111";
        cfg.transparency    = Transparency::BitStuffing;
        cfg.stuffAfterOnes  = 5U;
        cfg.abortOnes       = 0U; // the delimiter's own run would otherwise abandon the frame carrying it
        cfg.maxPayloadItems = 4096UZ;
        gr::digital::framerConfigure(cfg);

        gr::digital::DelimiterEncoder encoder;
        encoder.prepare(cfg);

        Rng                       rng{};
        std::vector<std::uint8_t> wire;
        for (std::size_t which = 0UZ; which < 2000UZ; ++which) {
            const std::size_t         length = 1UZ + rng.below(300UZ);
            std::vector<std::uint8_t> payload(length);
            for (std::uint8_t& bit : payload) {
                bit = static_cast<std::uint8_t>(rng.next() & 1ULL);
            }

            const gr::digital::EncodeResult framed = encoder.encode(std::span<const std::uint8_t>(payload), wire);
            const Run<std::uint8_t>         run    = drive<std::uint8_t>(cfg, std::span<const std::uint8_t>(wire));
            if (run.payloads.size() != 1UZ) {
                expect(eq(run.payloads.size(), 1UZ)) << std::format("case {}, length {}", which, length);
                break;
            }
            if (run.payloads[0UZ] != payload || run.removals[0UZ] != framed.inserted) {
                expect(that % (run.payloads[0UZ] == payload)) << std::format("case {}, length {}", which, length);
                expect(eq(run.removals[0UZ], framed.inserted)) << std::format("case {}, length {}", which, length);
                break;
            }
        }

        // the same frame with the count carried over the opening delimiter's six ones instead: the encoder inserts
        // nothing where the decoder removes one, and the frame comes back a bit short
        const std::vector<std::uint8_t> payload{1U, 1U, 1U, 1U, 1U, 0U};
        std::vector<std::uint8_t>       carried = itemsOf(cfg.endDelimiter);
        carried.insert(carried.end(), payload.begin(), payload.end()); // no insertion: the count is already past five
        const std::vector<std::uint8_t> closing = itemsOf(cfg.endDelimiter);
        carried.insert(carried.end(), closing.begin(), closing.end());

        const Run<std::uint8_t> diverged = drive<std::uint8_t>(cfg, std::span<const std::uint8_t>(carried));
        expect(eq(diverged.payloads.size(), 1UZ));
        if (diverged.payloads.size() == 1UZ) {
            expect(that % (diverged.payloads[0UZ] != payload)) << "the count carried over the delimiter loses the frame's last item";
            expect(eq(diverged.payloads[0UZ].size(), 5UZ));
        }
    };

    "the framer refuses an escape map that cannot keep the delimiter off the wire"_test = [] {
        const auto framed = [](std::uint8_t introducer, std::vector<std::pair<std::uint8_t, std::uint8_t>> pairs) {
            DelimiterConfig cfg;
            cfg.endDelimiter    = "11000000"; // 0xC0
            cfg.bitsPerItem     = 8U;
            cfg.transparency    = Transparency::ByteEscape;
            cfg.escapeItem      = introducer;
            cfg.escapeMap       = std::move(pairs);
            cfg.maxPayloadItems = 256UZ;
            gr::digital::framerConfigure(cfg);
            return cfg;
        };

        expect(nothrow([&framed] { std::ignore = framed(0xDBU, {{0xDCU, 0xC0U}, {0xDDU, 0xDBU}}); })) << "SLIP's map covers both the delimiter and the introducer";
        expect(throws([&framed] { std::ignore = framed(0xDBU, {{0xDCU, 0xC0U}}); })) << "the introducer has no pair, so the encoder cannot escape one in a payload";
        expect(throws([&framed] { std::ignore = framed(0xDBU, {{0xDDU, 0xDBU}}); })) << "the delimiter has no pair, which the receiver could only warn about";

        // the same check over a start delimiter, which is an item value the encoder emits and the receiver matches
        const auto opened = [](std::vector<std::pair<std::uint8_t, std::uint8_t>> pairs) {
            DelimiterConfig cfg;
            cfg.endDelimiter    = "11000000"; // 0xC0
            cfg.startDelimiter  = "00100100"; // '$'
            cfg.frameOpen       = FrameOpen::Start;
            cfg.bitsPerItem     = 8U;
            cfg.transparency    = Transparency::ByteEscape;
            cfg.escapeItem      = 0xDBU;
            cfg.escapeMap       = std::move(pairs);
            cfg.maxPayloadItems = 256UZ;
            gr::digital::framerConfigure(cfg);
        };
        expect(throws([&opened] { opened({{0xDCU, 0xC0U}, {0xDDU, 0xDBU}}); })) << "'$' is not an original, so a payload carrying it would open a frame in the middle of this one";
        expect(nothrow([&opened] { opened({{0xDCU, 0xC0U}, {0xDDU, 0xDBU}, {0x5EU, 0x24U}}); }));

        // the four published framings pass the transmit-side table as they pass the receive-side one
        const auto bounded = [](DelimiterConfig cfg) {
            cfg.maxPayloadItems = 1024UZ;
            gr::digital::framerConfigure(cfg);
        };
        expect(nothrow([&bounded] { bounded(gr::digital::hdlc()); }));
        expect(nothrow([&bounded] { bounded(gr::digital::slip()); }));
        expect(nothrow([&bounded] { bounded(gr::digital::pppAsync()); }));
        expect(nothrow([&bounded] { bounded(gr::digital::nmea0183()); }));

        // a repeated original is accepted and encodes by the last pair naming it, both spellings decoding alike
        const DelimiterConfig         twice = framed(0xDBU, {{0xDCU, 0xC0U}, {0xDDU, 0xDBU}, {0x5EU, 0xC0U}});
        gr::digital::DelimiterEncoder encoder;
        encoder.prepare(twice);
        const std::vector<std::uint8_t> payload{0xC0U, 0x41U};
        std::vector<std::uint8_t>       wire;
        const gr::digital::EncodeResult result = encoder.encode(std::span<const std::uint8_t>(payload), wire);
        expect(that % (wire == std::vector<std::uint8_t>{0xC0U, 0xDBU, 0x5EU, 0x41U, 0xC0U}));
        expect(eq(result.inserted, 1UZ));

        const Run<std::uint8_t> run = drive<std::uint8_t>(twice, std::span<const std::uint8_t>(wire));
        expect(eq(run.payloads.size(), 1UZ));
        if (run.payloads.size() == 1UZ) {
            expect(that % (run.payloads[0UZ] == payload)) << "the later spelling decodes to the same original the earlier one does";
        }
    };

    "a delimiter of '0' and '1' characters only"_test = [] {
        const auto spelled = [](std::string_view text) {
            DelimiterConfig cfg;
            cfg.endDelimiter    = std::string(text);
            cfg.maxPayloadItems = 256UZ;
            gr::digital::configureDelimiter(cfg);
        };
        expect(nothrow([&spelled] { spelled("11000000"); }));
        expect(throws([&spelled] { spelled("1100000O"); })) << "a typo is a different delimiter rather than a low-bit reading";
        expect(throws([&spelled] { spelled("0xC0"); }));
    };
};

int main() { /* not needed for UT */ }
