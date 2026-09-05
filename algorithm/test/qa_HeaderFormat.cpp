#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <print>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>
#include <variant>
#include <vector>

#include <gnuradio-4.0/algorithm/digital/HeaderFormat.hpp>
#include <gnuradio-4.0/algorithm/fec/Golay.hpp>

namespace {

using gr::digital::CountCovers;
using gr::digital::countFromPayloadItems;
using gr::digital::Crc;
using gr::digital::FixedLengthHeader;
using gr::digital::FlagsPosition;
using gr::digital::formatHeader;
using gr::digital::headerBit;
using gr::digital::HeaderByteOrder;
using gr::digital::HeaderFormat;
using gr::digital::headerItem;
using gr::digital::headerItemsOf;
using gr::digital::kMaxHeaderItems;
using gr::digital::LengthCount;
using gr::digital::LengthCrcHeader;
using gr::digital::LengthGolay24Header;
using gr::digital::LengthPlainHeader;
using gr::digital::LengthRepeatedHeader;
using gr::digital::parseHeader;
using gr::digital::payloadItemsFromCount;
using gr::digital::readHeaderField;
using gr::digital::writeHeaderField;

using gr::property_map;

using Clock = std::chrono::steady_clock;

/// The count arithmetic written out by hand from the specification, so the class is compared against the
/// specification and not against itself.
[[nodiscard]] std::optional<std::size_t> handCountedItems(CountCovers covers, std::size_t unit, std::size_t check, std::size_t prefix, std::uint64_t length) {
    const std::uint64_t product = static_cast<std::uint64_t>(unit) * length;
    std::uint64_t       items   = product;
    switch (covers) {
    case CountCovers::Payload: items = product + check; break;
    case CountCovers::PayloadAndCheck: break;
    case CountCovers::WholeFrame:
        if (product <= prefix) {
            return std::nullopt;
        }
        items = product - prefix;
        break;
    }
    if (items == 0ULL || items > std::numeric_limits<std::size_t>::max()) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(items);
}

/// The 24 wire items of a Golay header, written from section 4.1's layout: the first transmitted item is bit 23.
[[nodiscard]] std::array<std::uint8_t, 24> handBuiltGolayHeader(std::uint32_t word) {
    std::array<std::uint8_t, 24> items{};
    for (std::size_t i = 0UZ; i < items.size(); ++i) {
        items[i] = static_cast<std::uint8_t>((word >> (23U - static_cast<std::uint32_t>(i))) & 1U);
    }
    return items;
}

/// Every error pattern of weight one, two or three over 24 bits: 24 + 276 + 2024 = 2324 of them.
[[nodiscard]] std::vector<std::uint32_t> lowWeightPatterns() {
    std::vector<std::uint32_t> patterns;
    for (unsigned a = 0U; a < 24U; ++a) {
        patterns.push_back(1U << a);
        for (unsigned b = a + 1U; b < 24U; ++b) {
            patterns.push_back((1U << a) | (1U << b));
            for (unsigned c = b + 1U; c < 24U; ++c) {
                patterns.push_back((1U << a) | (1U << b) | (1U << c));
            }
        }
    }
    return patterns;
}

/// @brief `length_crc`'s wire layout, written here from the specification rather than from the class.
[[nodiscard]] std::vector<std::uint8_t> handBuiltCrcHeader(std::uint32_t length, std::uint32_t number) {
    const Crc                         crc{8U, 0x07ULL, 0xFFULL, 0x00ULL, false, false};
    const std::array<std::uint8_t, 4> covered{static_cast<std::uint8_t>(length & 0xFFU), static_cast<std::uint8_t>(length >> 8U), //
        static_cast<std::uint8_t>(number & 0xFFU), static_cast<std::uint8_t>(number >> 8U)};
    const auto                        check = static_cast<std::uint32_t>(crc.compute(covered));

    std::vector<std::uint8_t> header(32UZ);
    writeHeaderField(std::span<std::uint8_t>(header), 0UZ, 12UZ, length);
    writeHeaderField(std::span<std::uint8_t>(header), 12UZ, 12UZ, number);
    writeHeaderField(std::span<std::uint8_t>(header), 24UZ, 8UZ, check);
    return header;
}

} // namespace

const boost::ut::suite<"header formats"> headerFormatTests = [] {
    using namespace boost::ut;

    "an item is one bit, and the slicing rule is pinned rather than incidental"_test = [] {
        expect(that % !headerBit(std::uint8_t{0U}));
        expect(that % headerBit(std::uint8_t{1U}));
        expect(that % headerBit(std::uint8_t{0xFFU})) << "an unpacked-bit chain that emits 0xFF for a one is not misread";
        expect(that % headerBit(std::uint8_t{0x02U}));

        expect(that % headerBit(0.5F));
        expect(that % headerBit(+0.0F)) << "a soft bit slices at zero with x >= 0 meaning one";
        expect(that % headerBit(-0.0F)) << "and negative zero compares equal to zero, so it is also a one";
        expect(that % !headerBit(-0.5F));
        expect(that % !headerBit(std::numeric_limits<float>::quiet_NaN())) << "every IEEE comparison with NaN is false, so a NaN slices to zero";
        expect(that % !headerBit(std::numeric_limits<double>::quiet_NaN()));

        expect(eq(headerItem<std::uint8_t>(true), std::uint8_t{1U}));
        expect(eq(headerItem<std::uint8_t>(false), std::uint8_t{0U}));
        expect(eq(headerItem<float>(true), 1.0F));
        expect(eq(headerItem<float>(false), -1.0F));
    };

    "fields are MSB first, in layout order"_test = [] {
        std::array<std::uint8_t, 16> bits{};
        writeHeaderField(std::span<std::uint8_t>(bits), 0UZ, 16UZ, 0xB39CU);
        expect(eq(bits[0], std::uint8_t{1U})) << "bit 15 of 0xB39C is one and it is written first";
        expect(eq(bits[1], std::uint8_t{0U}));
        expect(eq(bits[15], std::uint8_t{0U}));
        expect(eq(readHeaderField(std::span<const std::uint8_t>(bits), 0UZ, 16UZ), 0xB39CU));
    };

    "fixed_length carries no header"_test = [] {
        const FixedLengthHeader layout{800UZ};
        expect(eq(layout.headerItems(), 0UZ));
        const auto parsed = layout.parse(std::span<const std::uint8_t>{});
        expect(that % parsed.has_value());
        expect(eq(parsed->payloadItems, 800UZ));
        expect(that % parsed->meta.empty());

        std::span<std::uint8_t> nothing{};
        expect(that % layout.format(800UZ, property_map{}, nothing));
        expect(that % !layout.format(799UZ, property_map{}, nothing));
    };

    "length_repeated accepts a matching pair and rejects a mismatched one"_test = [] {
        const LengthRepeatedHeader layout;
        expect(eq(layout.headerItems(), 32UZ));

        std::array<std::uint8_t, 32> header{};
        std::size_t                  roundTripFailures = 0UZ;
        for (std::uint32_t length = 0U; length <= 0xFFFFU; length += 7U) {
            expect(that % layout.format(length, property_map{}, std::span<std::uint8_t>(header)));
            const auto parsed = layout.parse(std::span<const std::uint8_t>(header));
            roundTripFailures += (!parsed.has_value() || parsed->payloadItems != length) ? 1UZ : 0UZ;
        }
        expect(eq(roundTripFailures, 0UZ)) << "the 16-bit reading round trips for every length up to 65535";

        expect(that % !layout.format(0x10000UZ, property_map{}, std::span<std::uint8_t>(header))) << "a length past 16 bits is reported, never truncated";

        expect(that % layout.format(4321UZ, property_map{}, std::span<std::uint8_t>(header)));
        std::size_t detected = 0UZ;
        for (std::size_t bit = 0UZ; bit < 32UZ; ++bit) {
            auto corrupted = header;
            corrupted[bit] = static_cast<std::uint8_t>(corrupted[bit] ^ 1U);
            detected += layout.parse(std::span<const std::uint8_t>(corrupted)).has_value() ? 0UZ : 1UZ;
        }
        expect(eq(detected, 32UZ)) << "repetition detects every single-bit error; it is the correlated ones it lets through";

        // The failure mode that makes it the weaker code: the same error in both copies is accepted.
        auto burst          = header;
        burst[3]            = static_cast<std::uint8_t>(burst[3] ^ 1U);
        burst[19]           = static_cast<std::uint8_t>(burst[19] ^ 1U);
        const auto accepted = layout.parse(std::span<const std::uint8_t>(burst));
        expect(that % accepted.has_value()) << "two copies corrupted identically agree, which is why this is not the default";
        expect(neq(accepted->payloadItems, 4321UZ));
    };

    "length_crc accepts a valid header, rejects every single-bit corruption, and reports the number"_test = [] {
        const LengthCrcHeader layout;
        expect(eq(layout.headerItems(), 32UZ));

        std::array<std::uint8_t, 32> header{};
        for (const std::uint32_t number : {0U, 1U, 1234U, 4095U}) {
            expect(that % layout.format(2000UZ, property_map{{"packet_number", static_cast<std::uint64_t>(number)}}, std::span<std::uint8_t>(header)));
            const auto parsed = layout.parse(std::span<const std::uint8_t>(header));
            expect(that % parsed.has_value());
            expect(eq(parsed->payloadItems, 2000UZ));
            const auto entry = parsed->meta.find("packet_number");
            expect(that % (entry != parsed->meta.end()));
            const auto* carried = entry->second.get_if<std::uint64_t>();
            expect(that % (carried != nullptr));
            expect(eq(*carried, static_cast<std::uint64_t>(number)));

            std::size_t detected = 0UZ;
            for (std::size_t bit = 0UZ; bit < 32UZ; ++bit) {
                auto corrupted = header;
                corrupted[bit] = static_cast<std::uint8_t>(corrupted[bit] ^ 1U);
                detected += layout.parse(std::span<const std::uint8_t>(corrupted)).has_value() ? 0UZ : 1UZ;
            }
            expect(eq(detected, 32UZ)) << std::format("packet number {}: all 32 single-bit corruptions rejected", number);
        }

        std::size_t roundTripFailures = 0UZ;
        for (std::size_t length = 1UZ; length <= 4095UZ; ++length) {
            if (!layout.format(length, property_map{}, std::span<std::uint8_t>(header))) {
                ++roundTripFailures;
                continue;
            }
            const auto parsed = layout.parse(std::span<const std::uint8_t>(header));
            roundTripFailures += (!parsed.has_value() || parsed->payloadItems != length) ? 1UZ : 0UZ;
        }
        expect(eq(roundTripFailures, 0UZ)) << "format then parse is the identity for every length in [1, 4095]";

        expect(that % !layout.format(4096UZ, property_map{}, std::span<std::uint8_t>(header))) << "the 12-bit field caps the payload at 4095 and format reports it rather than truncating";
        expect(that % !layout.format(100UZ, property_map{{"packet_number", std::uint64_t{4096U}}}, std::span<std::uint8_t>(header))) << "so does an out-of-range packet number";
    };

    "the CRC-8 is the Crc kernel under the stated six-tuple, not a reimplementation"_test = [] {
        const LengthCrcHeader layout;
        expect(eq(layout.crc().width(), std::uint8_t{8U}));
        expect(eq(layout.crc().polynomial(), 0x07ULL));
        expect(eq(layout.crc().initialValue(), 0xFFULL));
        expect(eq(layout.crc().finalXor(), 0x00ULL));
        expect(that % !layout.crc().inputReflected());
        expect(that % !layout.crc().resultReflected());

        std::array<std::uint8_t, 32> written{};
        std::size_t                  mismatches = 0UZ;
        for (const std::uint32_t length : {1U, 17U, 255U, 256U, 4095U}) {
            for (const std::uint32_t number : {0U, 7U, 4095U}) {
                expect(that % layout.format(length, property_map{{"packet_number", static_cast<std::uint64_t>(number)}}, std::span<std::uint8_t>(written)));
                const auto expected = handBuiltCrcHeader(length, number);
                mismatches += std::ranges::equal(written, expected) ? 0UZ : 1UZ;
            }
        }
        expect(eq(mismatches, 0UZ)) << "the wire bits agree with the layout written out by hand from the specification";
    };

    "a soft-bit header round trips through the same layouts"_test = [] {
        const LengthCrcHeader        layout;
        std::array<float, 32>        soft{};
        std::array<std::uint8_t, 32> hard{};

        std::size_t mismatches = 0UZ;
        for (std::size_t length = 1UZ; length <= 4095UZ; length += 13UZ) {
            expect(that % layout.format(length, property_map{{"packet_number", std::uint64_t{42U}}}, std::span<float>(soft)));
            expect(that % layout.format(length, property_map{{"packet_number", std::uint64_t{42U}}}, std::span<std::uint8_t>(hard)));
            for (std::size_t bit = 0UZ; bit < 32UZ; ++bit) {
                mismatches += (soft[bit] >= 0.0F) != (hard[bit] != 0U) ? 1UZ : 0UZ;
            }
            const auto parsed = layout.parse(std::span<const float>(soft));
            mismatches += (!parsed.has_value() || parsed->payloadItems != length) ? 1UZ : 0UZ;
        }
        expect(eq(mismatches, 0UZ)) << "a soft header carries the same bits and parses to the same length";

        // A soft header whose confidence is low is still a header; only the sign is read.
        std::array<float, 32> faint{};
        for (std::size_t bit = 0UZ; bit < 32UZ; ++bit) {
            faint[bit] = soft[bit] * 1.0e-9F;
        }
        const auto parsedFaint = layout.parse(std::span<const float>(faint));
        expect(that % parsedFaint.has_value());
    };

    "the variant dispatches once per packet and its buffer is fixed at compile time"_test = [] {
        const std::array<HeaderFormat, 3> formats{HeaderFormat{LengthCrcHeader{}}, HeaderFormat{LengthRepeatedHeader{}}, HeaderFormat{FixedLengthHeader{640UZ}}};
        const std::array<std::size_t, 3>  expectedItems{32UZ, 32UZ, 0UZ};

        std::array<std::uint8_t, kMaxHeaderItems> buffer{};
        for (std::size_t i = 0UZ; i < formats.size(); ++i) {
            const std::size_t items = headerItemsOf(formats[i]);
            expect(eq(items, expectedItems[i]));
            expect(le(items, kMaxHeaderItems)) << "kMaxHeaderItems bounds every layout in the closed set";

            const std::size_t length = i == 2UZ ? 640UZ : 300UZ;
            expect(that % formatHeader(formats[i], length, property_map{}, std::span<std::uint8_t>(buffer).first(items)));
            const auto parsed = parseHeader(formats[i], std::span<const std::uint8_t>(buffer).first(items));
            expect(that % parsed.has_value());
            expect(eq(parsed->payloadItems, length));
        }

        expect(that % std::holds_alternative<LengthCrcHeader>(HeaderFormat{})) << "length_crc is the default alternative, not the repeated one";
    };

    // The three conventions differ only in what the transmitter wrote, and the consequence of reading one as
    //    another is pinned as a number rather than warned about.
    "the three count conventions, against the worked table"_test = [] {
        // one wire frame: a 34-byte payload region whose last two bytes are a CRC-16, behind a 16-bit length field
        constexpr std::size_t kPayloadItems = 272UZ;
        struct Row {
            std::string_view name;
            CountCovers      covers;
            std::size_t      check;
            std::size_t      prefix;
            std::uint32_t    field;
        };
        constexpr Row kRows[] = {{"payload", CountCovers::Payload, 16UZ, 0UZ, 32U}, //
            {"payload_and_check", CountCovers::PayloadAndCheck, 0UZ, 0UZ, 34U},     //
            {"whole_frame", CountCovers::WholeFrame, 0UZ, 16UZ, 36U}};

        std::array<std::uint8_t, 16> wire{};
        for (const Row& row : kRows) {
            const LengthPlainHeader layout{16UZ, HeaderByteOrder::Big, LengthCount{row.covers, 8UZ, row.check, row.prefix}};
            expect(that % layout.format(kPayloadItems, property_map{}, std::span<std::uint8_t>(wire))) << row.name;
            expect(eq(readHeaderField(std::span<const std::uint8_t>(wire), 0UZ, 16UZ), row.field)) << std::format("{}: the field value the transmitter writes", row.name);

            const auto parsed = layout.parse(std::span<const std::uint8_t>(wire));
            expect(that % parsed.has_value());
            expect(eq(parsed->payloadItems, kPayloadItems)) << std::format("{}: and all three extract the same 272 items", row.name);
            expect(that % parsed->meta.empty()) << "a bare length field carries nothing else";
        }

        // reading a payload frame as payload_and_check drops the check field; reading a whole_frame frame as payload
        // over-runs into the next one. Both are arithmetic and both are asserted as the numbers they are.
        const LengthPlainHeader asPayloadAndCheck{16UZ, HeaderByteOrder::Big, LengthCount{CountCovers::PayloadAndCheck, 8UZ, 0UZ, 0UZ}};
        const LengthPlainHeader asPayload{16UZ, HeaderByteOrder::Big, LengthCount{CountCovers::Payload, 8UZ, 16UZ, 0UZ}};

        writeHeaderField(std::span<std::uint8_t>(wire), 0UZ, 16UZ, 32U);
        expect(eq(asPayloadAndCheck.parse(std::span<const std::uint8_t>(wire))->payloadItems, 256UZ)) << "a payload frame read as payload_and_check is two bytes short, and the CRC is what fails";
        writeHeaderField(std::span<std::uint8_t>(wire), 0UZ, 16UZ, 36U);
        expect(eq(asPayload.parse(std::span<const std::uint8_t>(wire))->payloadItems, 304UZ)) << "a whole_frame frame read as payload runs into the next frame";
    };

    // Byte order, worked from the wire items rather than from the class.
    "length_plain reads its field in the stated byte order"_test = [] {
        const LengthCount            plain{CountCovers::PayloadAndCheck, 1UZ, 0UZ, 0UZ};
        const LengthPlainHeader      big{16UZ, HeaderByteOrder::Big, plain};
        const LengthPlainHeader      little{16UZ, HeaderByteOrder::Little, plain};
        std::array<std::uint8_t, 16> items{0U, 0U, 0U, 0U, 0U, 0U, 1U, 0U, 0U, 0U, 1U, 0U, 0U, 0U, 0U, 0U}; // bytes 0x02, 0x20

        expect(eq(big.parse(std::span<const std::uint8_t>(items))->payloadItems, 0x0220UZ)) << "544: the field's bits are the integer's, most significant first";
        expect(eq(little.parse(std::span<const std::uint8_t>(items))->payloadItems, 0x2002UZ)) << "8194: the field's bytes are least significant first, its bits still most significant first";

        for (const HeaderByteOrder order : {HeaderByteOrder::Big, HeaderByteOrder::Little}) {
            const LengthPlainHeader layout{16UZ, order, plain};
            std::size_t             failures = 0UZ;
            for (std::size_t length = 1UZ; length <= 0xFFFFUZ; length += 37UZ) {
                std::array<std::uint8_t, 16> wire{};
                if (!layout.format(length, property_map{}, std::span<std::uint8_t>(wire))) {
                    ++failures;
                    continue;
                }
                const auto parsed = layout.parse(std::span<const std::uint8_t>(wire));
                failures += (!parsed.has_value() || parsed->payloadItems != length) ? 1UZ : 0UZ;
            }
            expect(eq(failures, 0UZ)) << "format then parse is the identity under both orders";
        }

        // a byte order over a field that is not whole bytes has no meaning, so it is refused where it is stated
        for (const std::size_t bits : {1UZ, 7UZ, 12UZ, 31UZ}) {
            expect(throws<std::invalid_argument>([bits, plain] { [[maybe_unused]] const LengthPlainHeader bad{bits, HeaderByteOrder::Little, plain}; })) << std::format("little at {} bits", bits);
        }
        expect(nothrow([plain] { [[maybe_unused]] const LengthPlainHeader ok{24UZ, HeaderByteOrder::Little, plain}; })) << "and accepted at a whole number of bytes";
    };

    // The round trip over the whole parameter surface, with the unrepresentable cases asserted to be exactly the
    //    ones the arithmetic names and no others.
    "length_plain round-trips, and refuses exactly what cannot be framed"_test = [] {
        constexpr CountCovers kCovers[] = {CountCovers::Payload, CountCovers::PayloadAndCheck, CountCovers::WholeFrame};

        std::size_t                  cases         = 0UZ;
        std::size_t                  roundTrips    = 0UZ;
        std::size_t                  disagreements = 0UZ;
        std::array<std::uint8_t, 32> wire{};
        for (std::size_t bits = 1UZ; bits <= 32UZ; ++bits) {
            for (const HeaderByteOrder order : {HeaderByteOrder::Big, HeaderByteOrder::Little}) {
                if (order == HeaderByteOrder::Little && bits % 8UZ != 0UZ) {
                    continue;
                }
                for (const CountCovers covers : kCovers) {
                    for (const std::size_t unit : {1UZ, 8UZ}) {
                        const std::size_t       check  = covers == CountCovers::Payload ? 16UZ : 0UZ;
                        const std::size_t       prefix = covers == CountCovers::WholeFrame ? 16UZ : 0UZ;
                        const LengthPlainHeader layout{bits, order, LengthCount{covers, unit, check, prefix}};

                        const std::uint64_t                top = (1ULL << bits) - 1ULL;
                        const std::array<std::uint64_t, 6> fields{0ULL, 1ULL, static_cast<std::uint64_t>(unit), static_cast<std::uint64_t>(unit) + 1ULL, top / 2ULL, top};
                        for (const std::uint64_t field : fields) {
                            if (field > top) {
                                continue;
                            }
                            ++cases;
                            const auto wanted = handCountedItems(covers, unit, check, prefix, field);

                            writeHeaderField(std::span<std::uint8_t>(wire), 0UZ, bits, static_cast<std::uint32_t>(field));
                            const LengthPlainHeader bigOnly{bits, HeaderByteOrder::Big, LengthCount{covers, unit, check, prefix}};
                            const auto              parsed = bigOnly.parse(std::span<const std::uint8_t>(wire).first(bits));
                            disagreements += parsed.has_value() == wanted.has_value() && (!wanted.has_value() || parsed->payloadItems == *wanted) ? 0UZ : 1UZ;

                            if (!wanted.has_value()) {
                                continue;
                            }
                            // and the same count, put back through the configured order, returns the same field
                            std::array<std::uint8_t, 32> written{};
                            if (!layout.format(*wanted, property_map{}, std::span<std::uint8_t>(written).first(bits))) {
                                continue;
                            }
                            const auto back = layout.parse(std::span<const std::uint8_t>(written).first(bits));
                            roundTrips += back.has_value() && back->payloadItems == *wanted ? 1UZ : 0UZ;
                        }
                    }
                }
            }
        }
        expect(that % (cases > 600UZ)) << std::format("{} parameter combinations swept", cases);
        expect(eq(disagreements, 0UZ)) << "the class agrees with the arithmetic written out by hand, refusals included";
        expect(that % (roundTrips > 500UZ)) << std::format("{} of them round-tripped", roundTrips);

        // what format refuses, and that it refuses by returning rather than by throwing
        const LengthPlainHeader byte{8UZ, HeaderByteOrder::Big, LengthCount{CountCovers::Payload, 8UZ, 16UZ, 0UZ}};
        expect(that % !byte.format(15UZ, property_map{}, std::span<std::uint8_t>(wire).first(8UZ))) << "a payload shorter than the check field it carries";
        expect(that % !byte.format(20UZ, property_map{}, std::span<std::uint8_t>(wire).first(8UZ))) << "a payload that is not a whole number of counted units";
        expect(that % byte.format(24UZ, property_map{}, std::span<std::uint8_t>(wire).first(8UZ))) << "and one that is";
        expect(that % !byte.format(8UZ * 256UZ + 16UZ, property_map{}, std::span<std::uint8_t>(wire).first(8UZ))) << "a length past the field's width is reported, never truncated";
        expect(that % byte.format(8UZ * 255UZ + 16UZ, property_map{}, std::span<std::uint8_t>(wire).first(8UZ))) << "and the largest that fits is accepted";
        expect(that % !byte.format(24UZ, property_map{}, std::span<std::uint8_t>(wire).first(7UZ))) << "a span shorter than the header is refused rather than written past";

        // the construction-time refusals, which are the only throws here
        const LengthCount ok{CountCovers::PayloadAndCheck, 8UZ, 0UZ, 0UZ};
        expect(throws<std::invalid_argument>([ok] { [[maybe_unused]] const LengthPlainHeader bad{0UZ, HeaderByteOrder::Big, ok}; })) << "a zero-width field";
        expect(throws<std::invalid_argument>([ok] { [[maybe_unused]] const LengthPlainHeader bad{33UZ, HeaderByteOrder::Big, ok}; })) << "and one past the header buffer";
        expect(throws<std::invalid_argument>([] { [[maybe_unused]] const LengthPlainHeader bad{8UZ, HeaderByteOrder::Big, LengthCount{CountCovers::PayloadAndCheck, 0UZ, 0UZ, 0UZ}}; })) << "a zero counting unit";
        expect(throws<std::invalid_argument>([] { [[maybe_unused]] const LengthPlainHeader bad{8UZ, HeaderByteOrder::Big, LengthCount{CountCovers::PayloadAndCheck, 65UZ, 0UZ, 0UZ}}; })) << "and one past 64";
        expect(throws<std::invalid_argument>([] { [[maybe_unused]] const LengthPlainHeader bad{8UZ, HeaderByteOrder::Big, LengthCount{CountCovers::PayloadAndCheck, 8UZ, 16UZ, 0UZ}}; })) << "check_items where the field already counts the check";
        expect(throws<std::invalid_argument>([] { [[maybe_unused]] const LengthPlainHeader bad{8UZ, HeaderByteOrder::Big, LengthCount{CountCovers::Payload, 8UZ, 0UZ, 16UZ}}; })) << "frame_prefix_items where the field does not count the frame";
    };

    // The overflow test is made before the multiplication, at the exact value where the product would wrap.
    "the count arithmetic does not wrap, and says so instead"_test = [] {
        if constexpr (sizeof(std::size_t) != 8UZ) {
            return; // the boundary asserted here is a 64-bit std::size_t's
        }
        constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();

        // a 32-bit field at u = 64 does not reach the wrapping point: 64 * (2^32 - 1) is 2^38, well inside 2^64. So
        // the refusal at the top of a 32-bit field belongs to max_payload_items and not to the arithmetic, which is
        // what the specification says and what is asserted here rather than assumed.
        const LengthCount wide{CountCovers::PayloadAndCheck, 64UZ, 0UZ, 0UZ};
        expect(that % payloadItemsFromCount(wide, 0xFFFFFFFFU, 32UZ).has_value());
        expect(eq(*payloadItemsFromCount(wide, 0xFFFFFFFFU, 32UZ), 64UZ * 0xFFFFFFFFUZ));

        // the boundary itself, one below and one at it, on the condition L > (SIZE_MAX - c) / u the specification
        // states. The count is a value type, so the boundary is reachable through it even though no 32-bit field can
        // spell it, and that is the honest place to assert the guard.
        for (const std::size_t unit : {3UZ, 8UZ, 64UZ}) {
            const LengthCount count{CountCovers::Payload, unit, 16UZ, 0UZ};
            const std::size_t last = (kMax - 16UZ) / unit;
            expect(that % (last > 0xFFFFFFFFUZ)) << std::format("u = {}: the wrapping point is far above anything a 32-bit field can say", unit);
            expect(that % payloadItemsFromCount(count, 0xFFFFFFFFU, 32UZ).has_value()) << std::format("u = {}", unit);
            expect(eq(*payloadItemsFromCount(count, 0xFFFFFFFFU, 32UZ), unit * 0xFFFFFFFFUZ + 16UZ));
        }

        // zero, and a whole_frame count no larger than its own prefix, are refusals rather than answers
        expect(that % !payloadItemsFromCount(LengthCount{CountCovers::PayloadAndCheck, 8UZ, 0UZ, 0UZ}, 0U, 16UZ).has_value()) << "a zero length is not a packet";
        expect(that % !payloadItemsFromCount(LengthCount{CountCovers::WholeFrame, 8UZ, 0UZ, 16UZ}, 2U, 16UZ).has_value()) << "a frame whose whole content is its own header carries no payload";
        expect(that % !payloadItemsFromCount(LengthCount{CountCovers::WholeFrame, 8UZ, 0UZ, 16UZ}, 1U, 16UZ).has_value()) << "and one shorter than its header is not a frame";
        expect(eq(*payloadItemsFromCount(LengthCount{CountCovers::WholeFrame, 8UZ, 0UZ, 16UZ}, 3U, 16UZ), 8UZ));
    };

    // Golay(24,12,8) on the header field, which is the property the realization exists for.
    "length_golay24 corrects three header bit errors and detects four"_test = [] {
        const LengthCount         count{CountCovers::PayloadAndCheck, 8UZ, 0UZ, 0UZ};
        const LengthGolay24Header layout{8UZ, FlagsPosition::High, count};
        expect(eq(layout.headerItems(), 24UZ));
        expect(eq(layout.flagBits(), 4UZ)) << "twelve information bits, eight of length and the rest flags";

        const auto patterns = lowWeightPatterns();
        expect(eq(patterns.size(), 2324UZ)) << "24 + 276 + 2024 error patterns of weight one, two and three";

        // Zero errors first: the key is present and reads zero, and the length and flags come back
        {
            const auto clean  = handBuiltGolayHeader(gr::fec::golay24Encode(0x3ADU)); // flags 3, length 0xAD
            const auto parsed = layout.parse(std::span<const std::uint8_t>(clean));
            expect(that % parsed.has_value());
            expect(eq(parsed->payloadItems, 8UZ * 0xADUZ));
            const auto corrected = parsed->meta.find("header_corrected_errors");
            expect(that % (corrected != parsed->meta.end())) << "the key is present on a clean header, not only on a corrected one";
            expect(eq(*corrected->second.get_if<std::uint64_t>(), std::uint64_t{0}));
            expect(eq(*parsed->meta.find("header_flags")->second.get_if<std::uint64_t>(), std::uint64_t{3}));
        }

        // Corrects three. A parse costs about 266 ns against the bare decode's 12, because it builds a
        // two-entry property_map, so the information words are strided rather than swept: one in eight of
        // the 4096 against all 2324 patterns of weight one to three.
        constexpr std::uint32_t stride  = 8U;
        const auto              started = Clock::now();
        std::size_t             decodes = 0UZ;
        std::size_t             wrong   = 0UZ;
        for (std::uint32_t info = 0U; info < 4096U; info += stride) {
            const std::uint32_t length = info & 0xFFU;
            const std::uint32_t flags  = info >> 8U;
            if (length == 0U) {
                continue; // a zero length is the count arithmetic's refusal, asserted in its own case
            }
            const std::uint32_t word = gr::fec::golay24Encode(static_cast<std::uint16_t>(info));
            for (const std::uint32_t pattern : patterns) {
                const auto wire   = handBuiltGolayHeader(word ^ pattern);
                const auto parsed = layout.parse(std::span<const std::uint8_t>(wire));
                ++decodes;
                if (!parsed.has_value() || parsed->payloadItems != 8UZ * length || *parsed->meta.find("header_flags")->second.get_if<std::uint64_t>() != flags || *parsed->meta.find("header_corrected_errors")->second.get_if<std::uint64_t>() != static_cast<std::uint64_t>(std::popcount(pattern))) {
                    ++wrong;
                }
            }
        }
        expect(eq(wrong, 0UZ)) << std::format("{} decodes of weight one to three, every one recovering the length, the flags and the injected weight", decodes);

        // Detects four: every weight-4 pattern over a sampled set of information words.
        constexpr std::uint32_t words    = 64U;
        std::size_t             accepted = 0UZ;
        std::size_t             detected = 0UZ;
        for (std::uint32_t info = 0U; info < words; ++info) {
            const std::uint32_t word = gr::fec::golay24Encode(static_cast<std::uint16_t>(info));
            for (unsigned a = 0U; a < 24U; ++a) {
                for (unsigned b = a + 1U; b < 24U; ++b) {
                    for (unsigned c = b + 1U; c < 24U; ++c) {
                        for (unsigned d = c + 1U; d < 24U; ++d) {
                            const std::uint32_t pattern = (1U << a) | (1U << b) | (1U << c) | (1U << d);
                            const auto          wire    = handBuiltGolayHeader(word ^ pattern);
                            (layout.parse(std::span<const std::uint8_t>(wire)).has_value() ? accepted : detected) += 1UZ;
                        }
                    }
                }
            }
        }
        expect(eq(accepted, 0UZ)) << "a header carrying four bit errors is refused, never framed";
        expect(eq(detected, static_cast<std::size_t>(words) * 10626UZ)) << "C(24,4) = 10626 patterns per word";
        std::println("golay header sweep: {} weight-1..3 parses and {} weight-4 parses in {:.0f} ms", decodes, detected, std::chrono::duration<double, std::milli>(Clock::now() - started).count());
    };

    // The flag split, both ways, and the encode-then-parse identity over the whole field.
    "flags_position is a shift, and the round trip holds either way"_test = [] {
        const LengthCount count{CountCovers::Payload, 8UZ, 16UZ, 0UZ};

        for (const FlagsPosition where : {FlagsPosition::High, FlagsPosition::Low}) {
            const LengthGolay24Header    layout{8UZ, where, count};
            std::array<std::uint8_t, 24> wire{};

            std::size_t failures = 0UZ;
            for (std::uint32_t length = 1U; length <= 0xFFU; ++length) {
                for (const std::uint32_t flags : {0U, 1U, 9U, 15U}) {
                    const std::size_t items = 8UZ * static_cast<std::size_t>(length) + 16UZ;
                    if (!layout.format(items, property_map{{"header_flags", static_cast<std::uint64_t>(flags)}}, std::span<std::uint8_t>(wire))) {
                        ++failures;
                        continue;
                    }
                    const auto parsed = layout.parse(std::span<const std::uint8_t>(wire));
                    failures += parsed.has_value() && parsed->payloadItems == items && *parsed->meta.find("header_flags")->second.get_if<std::uint64_t>() == flags ? 0UZ : 1UZ;
                }
            }
            expect(eq(failures, 0UZ)) << "encode then parse is the identity over every length and flag value";
        }

        // the two positions are different wire words for the same content, which is what makes the setting load-bearing
        const LengthGolay24Header    high{8UZ, FlagsPosition::High, count};
        const LengthGolay24Header    low{8UZ, FlagsPosition::Low, count};
        std::array<std::uint8_t, 24> a{};
        std::array<std::uint8_t, 24> b{};
        const property_map           flags{{"header_flags", std::uint64_t{5}}};
        expect(that % high.format(8UZ * 0xC3UZ + 16UZ, flags, std::span<std::uint8_t>(a)));
        expect(that % low.format(8UZ * 0xC3UZ + 16UZ, flags, std::span<std::uint8_t>(b)));
        expect(that % !std::ranges::equal(a, b)) << "the length in the low bits and the length in the high bits are different headers";
        expect(that % (low.parse(std::span<const std::uint8_t>(a))->payloadItems != 8UZ * 0xC3UZ + 16UZ)) << "and reading one as the other is a wrong length, not a refusal";

        // a twelve-bit length leaves no flag bits, and the key is then absent rather than a constant zero
        const LengthGolay24Header full{12UZ, FlagsPosition::High, LengthCount{CountCovers::PayloadAndCheck, 8UZ, 0UZ, 0UZ}};
        expect(eq(full.flagBits(), 0UZ));
        std::array<std::uint8_t, 24> wide{};
        expect(that % full.format(8UZ * 4095UZ, property_map{}, std::span<std::uint8_t>(wide)));
        const auto parsed = full.parse(std::span<const std::uint8_t>(wide));
        expect(eq(parsed->payloadItems, 8UZ * 4095UZ));
        expect(that % (parsed->meta.find("header_flags") == parsed->meta.end())) << "no wire bit backs a flags key here, so there is no flags key";
        expect(that % !full.format(8UZ * 4096UZ, property_map{}, std::span<std::uint8_t>(wide))) << "and the twelve-bit field's cap is reported rather than truncated";

        expect(that % !high.format(8UZ * 0x10UZ + 16UZ, property_map{{"header_flags", std::uint64_t{16}}}, std::span<std::uint8_t>(a))) << "flags too wide for the field are refused";
        expect(that % !high.format(8UZ * 0x10UZ + 16UZ, property_map{{"header_flags", std::uint64_t{0}}}, std::span<std::uint8_t>(a).first(23UZ))) << "and so is a span one item short of the codeword";

        expect(throws<std::invalid_argument>([count] { [[maybe_unused]] const LengthGolay24Header bad{0UZ, FlagsPosition::High, count}; })) << "a zero-width length field";
        expect(throws<std::invalid_argument>([count] { [[maybe_unused]] const LengthGolay24Header bad{13UZ, FlagsPosition::High, count}; })) << "and one past the twelve information bits";
    };

    // The variant grew and its buffer did not.
    "the widened variant still fits kMaxHeaderItems"_test = [] {
        const LengthCount                 count{CountCovers::PayloadAndCheck, 8UZ, 0UZ, 0UZ};
        const std::array<HeaderFormat, 5> formats{HeaderFormat{LengthCrcHeader{}}, HeaderFormat{LengthRepeatedHeader{}}, HeaderFormat{FixedLengthHeader{640UZ}}, //
            HeaderFormat{LengthPlainHeader{32UZ, HeaderByteOrder::Big, count}}, HeaderFormat{LengthGolay24Header{12UZ, FlagsPosition::High, count}}};
        const std::array<std::size_t, 5>  expectedItems{32UZ, 32UZ, 0UZ, 32UZ, 24UZ};

        std::array<std::uint8_t, kMaxHeaderItems> buffer{};
        for (std::size_t i = 0UZ; i < formats.size(); ++i) {
            const std::size_t items = headerItemsOf(formats[i]);
            expect(eq(items, expectedItems[i]));
            expect(le(items, kMaxHeaderItems)) << "kMaxHeaderItems bounds every layout in the closed set, the two new ones included";

            const std::size_t length = i == 2UZ ? 640UZ : 8UZ * 40UZ;
            expect(that % formatHeader(formats[i], length, property_map{}, std::span<std::uint8_t>(buffer).first(items))) << std::format("layout {}", i);
            const auto parsed = parseHeader(formats[i], std::span<const std::uint8_t>(buffer).first(items));
            expect(that % parsed.has_value());
            expect(eq(parsed->payloadItems, length));
        }

        expect(that % std::holds_alternative<LengthCrcHeader>(HeaderFormat{})) << "and length_crc is still the default alternative";
    };

    // A soft header reaches the new layouts too: the Golay decode reads signs, so a faint header is still a header.
    "the new layouts read soft items by their sign"_test = [] {
        const LengthCount            count{CountCovers::PayloadAndCheck, 8UZ, 0UZ, 0UZ};
        const LengthGolay24Header    layout{8UZ, FlagsPosition::High, count};
        std::array<float, 24>        soft{};
        std::array<std::uint8_t, 24> hard{};

        expect(that % layout.format(8UZ * 200UZ, property_map{{"header_flags", std::uint64_t{7}}}, std::span<float>(soft)));
        expect(that % layout.format(8UZ * 200UZ, property_map{{"header_flags", std::uint64_t{7}}}, std::span<std::uint8_t>(hard)));
        std::size_t mismatches = 0UZ;
        for (std::size_t bit = 0UZ; bit < 24UZ; ++bit) {
            mismatches += (soft[bit] >= 0.0F) != (hard[bit] != 0U) ? 1UZ : 0UZ;
        }
        expect(eq(mismatches, 0UZ));

        std::array<float, 24> faint{};
        for (std::size_t bit = 0UZ; bit < 24UZ; ++bit) {
            faint[bit] = soft[bit] * 1.0e-9F;
        }
        const auto parsed = layout.parse(std::span<const float>(faint));
        expect(that % parsed.has_value());
        expect(eq(parsed->payloadItems, 8UZ * 200UZ)) << "only the sign is read, so a low-confidence header parses to the same length";

        // and one soft item flipped is a corrected header, not a refused one
        std::array<float, 24> nicked = soft;
        nicked[5]                    = -nicked[5];
        const auto corrected         = layout.parse(std::span<const float>(nicked));
        expect(that % corrected.has_value());
        expect(eq(corrected->payloadItems, 8UZ * 200UZ));
        expect(eq(*corrected->meta.find("header_corrected_errors")->second.get_if<std::uint64_t>(), std::uint64_t{1}));
    };
};

int main() { /* tests are automatically registered and run */ }
