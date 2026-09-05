#ifndef GNURADIO_ALGORITHM_HEADER_FORMAT_HPP
#define GNURADIO_ALGORITHM_HEADER_FORMAT_HPP

#include <gnuradio-4.0/Tag.hpp>

#include <gnuradio-4.0/algorithm/digital/Crc.hpp>
#include <gnuradio-4.0/algorithm/fec/Golay.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>

/**
 * @brief The five packet-header layouts, as small value types rather than a virtual hierarchy.
 *
 * A header carries how long the packet is. A framer collects `headerItems()` items, calls `parse`, and
 * either gets a length back or gets `std::nullopt` and goes back to searching. A modulator calls
 * `format`. The set is closed, a `std::variant`, so the framer's header buffer is a fixed array sized
 * at compile time from `kMaxHeaderItems` and nothing is allocated per packet. A user-supplied layout
 * needs a code change; that is the same trade `Constellation` makes for its decision strategies.
 *
 * An item is one bit. For `std::uint8_t` an item is one iff it is non-zero, which accepts both the 0/1
 * convention and the 0/0xFF one an unpacked-bit chain emits; for a floating-point item it is a soft bit
 * sliced at zero, `x >= 0` meaning one, matching the soft-decision sign convention. A NaN slices to
 * zero, because every IEEE comparison with NaN is false. One bit per item keeps the layouts free of
 * a bits-per-item mask table, which is a table whose entries have to be transcribed correctly: a
 * six-bit mask written `0x2F` where `0x3F` is meant drops bit 4 on both the format and the parse
 * path, so a packet round-trips through the same corruption and looks intact.
 *
 * `parse` takes a span sized exactly `headerItems()`. The framer has already established the count, so
 * a format cannot read past it and cannot report having consumed a different number. That is structural
 * rather than a bounds check. It rules out a parser that reads `input[nbits_in]`, one item past what
 * the scheduler provided, and then returns `nbits_in + 1` as its consumed count.
 *
 * The five layouts:
 *
 * ```
 *   fixed_length     no header at all; the length is a setting
 *   length_repeated  | length (16) | length (16) |            32 items, accepted when the copies agree
 *   length_crc       | length (12) | packet number (12) | CRC-8 (8) |   32 items      <- the default
 *   length_plain     | length (length_bits) |                1 to 32 items, in a stated byte order
 *   length_golay24   | Golay(24,12,8) over | flags | length | |  24 items, correcting three bit errors
 * ```
 *
 * The last two carry a number rather than an item count, and what that number counts is a setting with no default:
 * `LengthCount` states the arithmetic once and both share it.
 *
 * Fields are MSB first, in layout order. `length_crc` is the default because repetition is a poor code:
 * it detects any single error and accepts any error pattern that hits both copies identically, which
 * for a burst spanning the field boundary is not rare, and its detection capability against a CRC-8
 * over the same 32 bits is strictly worse. `length_repeated` is carried as an interoperability
 * format, and its length field is read as 16 bits, which is the reading its published description
 * states; implementations that read it as 12 disagree with that above 4095.
 *
 * `length_crc`'s CRC-8 covers the four bytes `{len & 0xFF, len >> 8, num & 0xFF, num >> 8}` under
 * `width = 8, poly = 0x07, init = 0xFF, finalXor = 0x00, refin = false, refout = false`. Those
 * parameters are kept for interoperability and computed through the same `Crc` kernel everything
 * else in this domain uses. They are their own parameter set rather than CRC-8/ATM despite the
 * shared `0x07` polynomial: ITU-T I.432.1 seeds with `0x00` and finishes with `0x55`. The 12-bit length caps a payload at 4095 items, and
 * `format` reports that rather than truncating silently.
 *
 * No layout here carries a sync word. A sync word is an interoperability constant belonging to whatever
 * detector precedes the framer, and there is no default one.
 */
namespace gr::digital {

/// @brief The item types a header is carried in: hard bits, or soft bits sliced at zero.
template<typename T>
concept HeaderItem = std::same_as<T, std::uint8_t> || std::floating_point<T>;

/// @brief The largest `headerItems()` any layout in the variant can return, so a framer's buffer is fixed.
inline constexpr std::size_t kMaxHeaderItems = 32UZ;

/// @brief What a successful parse yields: the payload length, plus whatever else the layout carried.
struct ParsedHeader {
    std::size_t  payloadItems = 0UZ;
    property_map meta{};
};

[[nodiscard]] inline constexpr bool headerBit(std::uint8_t item) noexcept { return item != 0U; }

template<std::floating_point T>
[[nodiscard]] inline constexpr bool headerBit(T item) noexcept {
    return item >= T{0};
}

template<HeaderItem T>
[[nodiscard]] inline constexpr T headerItem(bool bit) noexcept {
    if constexpr (std::same_as<T, std::uint8_t>) {
        return static_cast<std::uint8_t>(bit ? 1U : 0U);
    } else {
        return bit ? T{1} : T{-1};
    }
}

template<HeaderItem T>
[[nodiscard]] inline constexpr std::uint32_t readHeaderField(std::span<const T> header, std::size_t offset, std::size_t bits) noexcept {
    std::uint32_t value = 0U;
    for (std::size_t i = 0UZ; i < bits; ++i) {
        value = (value << 1U) | (headerBit(header[offset + i]) ? 1U : 0U);
    }
    return value;
}

template<HeaderItem T>
inline constexpr void writeHeaderField(std::span<T> header, std::size_t offset, std::size_t bits, std::uint32_t value) noexcept {
    for (std::size_t i = 0UZ; i < bits; ++i) {
        header[offset + i] = headerItem<T>(((value >> static_cast<std::uint32_t>(bits - 1UZ - i)) & 1U) != 0U);
    }
}

/// @brief No header at all: the length is a setting. The degenerate case; a fixed-length protocol needs no second block.
class FixedLengthHeader {
public:
    explicit constexpr FixedLengthHeader(std::size_t items = 1UZ) noexcept : payloadItems_(items) {}

    [[nodiscard]] static constexpr std::size_t headerItems() noexcept { return 0UZ; }

    template<HeaderItem T>
    [[nodiscard]] std::optional<ParsedHeader> parse(std::span<const T>) const {
        return ParsedHeader{payloadItems_, {}};
    }

    template<HeaderItem T>
    [[nodiscard]] constexpr bool format(std::size_t requested, const property_map&, std::span<T>) const noexcept {
        return requested == payloadItems_;
    }

    [[nodiscard]] constexpr std::size_t payloadItems() const noexcept { return payloadItems_; }

private:
    std::size_t payloadItems_;
};

/// @brief An interoperability layout in wide use: the 16-bit length written twice.
class LengthRepeatedHeader {
public:
    [[nodiscard]] static constexpr std::size_t headerItems() noexcept { return 32UZ; }

    template<HeaderItem T>
    [[nodiscard]] std::optional<ParsedHeader> parse(std::span<const T> header) const {
        const std::uint32_t first  = readHeaderField(header, 0UZ, 16UZ);
        const std::uint32_t second = readHeaderField(header, 16UZ, 16UZ);
        if (first != second) {
            return std::nullopt;
        }
        return ParsedHeader{static_cast<std::size_t>(first), {}};
    }

    template<HeaderItem T>
    [[nodiscard]] bool format(std::size_t payloadItems, const property_map&, std::span<T> out) const {
        if (payloadItems > 0xFFFFUZ || out.size() < headerItems()) {
            return false;
        }
        writeHeaderField(out, 0UZ, 16UZ, static_cast<std::uint32_t>(payloadItems));
        writeHeaderField(out, 16UZ, 16UZ, static_cast<std::uint32_t>(payloadItems));
        return true;
    }
};

/// @brief The default: a 12-bit length and a 12-bit packet number under a CRC-8 that covers both.
class LengthCrcHeader {
public:
    static constexpr std::size_t   kLengthBits = 12UZ;
    static constexpr std::size_t   kNumberBits = 12UZ;
    static constexpr std::size_t   kCrcBits    = 8UZ;
    static constexpr std::uint32_t kMaxLength  = (1U << kLengthBits) - 1U;

    LengthCrcHeader() : crc_(8U, 0x07ULL, 0xFFULL, 0x00ULL, false, false) {}

    [[nodiscard]] static constexpr std::size_t headerItems() noexcept { return kLengthBits + kNumberBits + kCrcBits; }

    template<HeaderItem T>
    [[nodiscard]] std::optional<ParsedHeader> parse(std::span<const T> header) const {
        const std::uint32_t length  = readHeaderField(header, 0UZ, kLengthBits);
        const std::uint32_t number  = readHeaderField(header, kLengthBits, kNumberBits);
        const std::uint32_t carried = readHeaderField(header, kLengthBits + kNumberBits, kCrcBits);
        if (carried != static_cast<std::uint32_t>(crc_.compute(coveredBytes(length, number)))) {
            return std::nullopt;
        }
        return ParsedHeader{static_cast<std::size_t>(length), property_map{{"packet_number", static_cast<std::uint64_t>(number)}}};
    }

    template<HeaderItem T>
    [[nodiscard]] bool format(std::size_t payloadItems, const property_map& extra, std::span<T> out) const {
        if (payloadItems > kMaxLength || out.size() < headerItems()) {
            return false;
        }
        std::uint32_t number = 0U;
        if (const auto entry = extra.find("packet_number"); entry != extra.end()) {
            const auto* carried = entry->second.get_if<std::uint64_t>();
            if (carried == nullptr || *carried > kMaxLength) {
                return false;
            }
            number = static_cast<std::uint32_t>(*carried);
        }
        const auto length = static_cast<std::uint32_t>(payloadItems);
        writeHeaderField(out, 0UZ, kLengthBits, length);
        writeHeaderField(out, kLengthBits, kNumberBits, number);
        writeHeaderField(out, kLengthBits + kNumberBits, kCrcBits, static_cast<std::uint32_t>(crc_.compute(coveredBytes(length, number))));
        return true;
    }

    [[nodiscard]] const Crc& crc() const noexcept { return crc_; }

private:
    [[nodiscard]] static constexpr std::array<std::uint8_t, 4> coveredBytes(std::uint32_t length, std::uint32_t number) noexcept { return {static_cast<std::uint8_t>(length & 0xFFU), static_cast<std::uint8_t>(length >> 8U), static_cast<std::uint8_t>(number & 0xFFU), static_cast<std::uint8_t>(number >> 8U)}; }

    Crc crc_;
};

/// @brief What a length field's number counts. Required, with no default: the three give different answers for the
/// same wire bits and a silent choice is a silent interoperability assumption.
enum class CountCovers : std::uint8_t { Payload, PayloadAndCheck, WholeFrame };

/// @brief Whether a length field's bytes run most or least significant first. The bits inside a byte are MSB first
/// either way, which is the whole of the difference.
enum class HeaderByteOrder : std::uint8_t { Big, Little };

/// @brief Which end of a Golay header's twelve information bits the flags occupy.
enum class FlagsPosition : std::uint8_t { High, Low };

/**
 * @brief The count arithmetic, stated once and shared by every layout that carries a length field.
 *
 * A field carries a number `L` in units of `unitItems` items, and three conventions differ in what that number counts:
 *
 * ```
 *   payloadItems = u * L            payload_and_check   the field counts the whole extracted region
 *                = u * L + c        payload             the field excludes a trailing check field of c items
 *                = u * L - p        whole_frame         the field includes the p items the frame carries before the payload
 * ```
 *
 * `unitItems` is 8 by default because a framer item is one bit and a length field counts bytes, which is a unit
 * conversion and not an interoperability assumption; `covers` has no default for the opposite reason. A worked
 * example, from one 34-byte payload region whose last two bytes are a CRC-16, behind a 16-bit length field: at
 * `u = 8`, `c = 16`, `p = 16` the three conventions write 32, 34 and 36 into the field and every one of them extracts
 * 272 items. Reading a `payload` frame as `payload_and_check` extracts 256 and hands the checker a record two bytes
 * short; reading a `whole_frame` frame as `payload` extracts 304 and runs into the next frame.
 *
 * `framePrefixItems` of zero means the layout's own `headerItems()`, which is the ordinary case and what makes the
 * default correct without stating a number twice.
 */
struct LengthCount {
    CountCovers covers           = CountCovers::PayloadAndCheck;
    std::size_t unitItems        = 8UZ; ///< `u`, items per counted unit, in `[1, 64]`
    std::size_t checkItems       = 0UZ; ///< `c`, trailing check items the field does not count; only under `Payload`
    std::size_t framePrefixItems = 0UZ; ///< `p`, frame items ahead of the payload the field does count; zero means `headerItems()`
};

/// @brief Refuse a count whose parameters do not describe the convention they are given with, naming the setting.
inline void validateLengthCount(const LengthCount& count) {
    if (count.unitItems < 1UZ || count.unitItems > 64UZ) {
        throw std::invalid_argument("gr::digital::LengthCount: count_unit_items must be in [1, 64], got " + std::to_string(count.unitItems));
    }
    if (count.checkItems != 0UZ && count.covers != CountCovers::Payload) {
        throw std::invalid_argument("gr::digital::LengthCount: check_items is only meaningful when the field counts the payload alone, got " + std::to_string(count.checkItems));
    }
    if (count.framePrefixItems != 0UZ && count.covers != CountCovers::WholeFrame) {
        throw std::invalid_argument("gr::digital::LengthCount: frame_prefix_items is only meaningful when the field counts the whole frame, got " + std::to_string(count.framePrefixItems));
    }
}

/// @brief The payload item count a field value @p length names, or `std::nullopt` when the arithmetic has no answer.
///
/// The overflow test is made before the multiplication and never after it: `L > (SIZE_MAX - c) / u` is the exact
/// condition under which `u * L + c` would wrap, and testing the product is testing a value that has already been
/// lost. A `whole_frame` count no larger than the prefix describes a frame whose entire content is its own header, and
/// a result of zero is a packet with no payload; both are refusals here as well as in the framer above.
[[nodiscard]] inline constexpr std::optional<std::size_t> payloadItemsFromCount(const LengthCount& count, std::uint32_t length, std::size_t headerItems) noexcept {
    const std::size_t unit   = count.unitItems;
    const std::size_t check  = count.checkItems;
    const std::size_t prefix = count.framePrefixItems != 0UZ ? count.framePrefixItems : headerItems;
    const std::size_t value  = static_cast<std::size_t>(length);

    if (unit == 0UZ || value > (std::numeric_limits<std::size_t>::max() - check) / unit) {
        return std::nullopt;
    }

    std::size_t items = unit * value;
    switch (count.covers) {
    case CountCovers::Payload: items += check; break;
    case CountCovers::PayloadAndCheck: break;
    case CountCovers::WholeFrame:
        if (items <= prefix) {
            return std::nullopt;
        }
        items -= prefix;
        break;
    }
    return items == 0UZ ? std::nullopt : std::optional<std::size_t>{items};
}

/// @brief The field value naming @p payloadItems, or `std::nullopt` when no whole value in `[0, maxValue]` does.
///
/// The inversion is a division and it is checked rather than assumed: a payload whose item count is not a whole number
/// of counted units cannot be framed by a length field at all, and saying so is what `format` returning `false` means.
[[nodiscard]] inline constexpr std::optional<std::uint32_t> countFromPayloadItems(const LengthCount& count, std::size_t payloadItems, std::size_t headerItems, std::uint32_t maxValue) noexcept {
    const std::size_t unit   = count.unitItems;
    const std::size_t check  = count.checkItems;
    const std::size_t prefix = count.framePrefixItems != 0UZ ? count.framePrefixItems : headerItems;
    if (unit == 0UZ) {
        return std::nullopt;
    }

    std::size_t counted = payloadItems;
    switch (count.covers) {
    case CountCovers::Payload:
        if (payloadItems < check) {
            return std::nullopt;
        }
        counted = payloadItems - check;
        break;
    case CountCovers::PayloadAndCheck: break;
    case CountCovers::WholeFrame:
        if (payloadItems > std::numeric_limits<std::size_t>::max() - prefix) {
            return std::nullopt;
        }
        counted = payloadItems + prefix;
        break;
    }

    if (counted % unit != 0UZ) {
        return std::nullopt;
    }
    const std::size_t value = counted / unit;
    if (value > static_cast<std::size_t>(maxValue)) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(value);
}

/// @brief A bare length field of a stated width, in a stated byte order, counting one of the three things.
///
/// ```
///   | length (length_bits) |
/// ```
///
/// The width caps at 32 because `kMaxHeaderItems` is 32 and a framer's header buffer is sized from it; that is an
/// exact constraint rather than a comfortable one, and it is why the field is not 64 bits wide. `little` reads the
/// field's bytes least significant first with each byte's bits still most significant first, so it requires a width
/// that is a whole number of bytes: a byte order over a field that is not whole bytes has no meaning, and refusing is
/// better than doing something. Worked, at 16 bits over the wire items of `0x02`, `0x20`: `big` reads 544 and
/// `little` reads 8194.
///
/// `meta` is empty. A bare length field carries no packet number, no flags and no status, and a key no wire bit backs
/// is worse than no key.
class LengthPlainHeader {
public:
    static constexpr std::size_t kMaxLengthBits = 32UZ;

    LengthPlainHeader(std::size_t lengthBits, HeaderByteOrder order, LengthCount count) : lengthBits_(lengthBits), order_(order), count_(count) {
        if (lengthBits < 1UZ || lengthBits > kMaxLengthBits) {
            throw std::invalid_argument("gr::digital::LengthPlainHeader: length_bits must be in [1, 32], got " + std::to_string(lengthBits));
        }
        if (order == HeaderByteOrder::Little && lengthBits % 8UZ != 0UZ) {
            throw std::invalid_argument("gr::digital::LengthPlainHeader: a little-endian byte order needs a whole number of bytes, got length_bits " + std::to_string(lengthBits));
        }
        validateLengthCount(count);
        maxValue_ = lengthBits == 32UZ ? ~std::uint32_t{0} : ((std::uint32_t{1} << lengthBits) - 1U);
    }

    [[nodiscard]] std::size_t        headerItems() const noexcept { return lengthBits_; }
    [[nodiscard]] std::size_t        lengthBits() const noexcept { return lengthBits_; }
    [[nodiscard]] HeaderByteOrder    byteOrder() const noexcept { return order_; }
    [[nodiscard]] const LengthCount& count() const noexcept { return count_; }

    template<HeaderItem T>
    [[nodiscard]] std::optional<ParsedHeader> parse(std::span<const T> header) const {
        const std::optional<std::size_t> items = payloadItemsFromCount(count_, readField(header), headerItems());
        if (!items.has_value()) {
            return std::nullopt;
        }
        return ParsedHeader{*items, {}};
    }

    template<HeaderItem T>
    [[nodiscard]] bool format(std::size_t payloadItems, const property_map&, std::span<T> out) const {
        if (out.size() < headerItems()) {
            return false;
        }
        const std::optional<std::uint32_t> value = countFromPayloadItems(count_, payloadItems, headerItems(), maxValue_);
        if (!value.has_value()) {
            return false;
        }
        writeField(out, *value);
        return true;
    }

private:
    template<HeaderItem T>
    [[nodiscard]] std::uint32_t readField(std::span<const T> header) const noexcept {
        if (order_ == HeaderByteOrder::Big) {
            return readHeaderField(header, 0UZ, lengthBits_);
        }
        std::uint32_t value = 0U;
        for (std::size_t byte = 0UZ; byte * 8UZ < lengthBits_; ++byte) {
            value |= readHeaderField(header, byte * 8UZ, 8UZ) << static_cast<std::uint32_t>(byte * 8UZ);
        }
        return value;
    }

    template<HeaderItem T>
    void writeField(std::span<T> out, std::uint32_t value) const noexcept {
        if (order_ == HeaderByteOrder::Big) {
            writeHeaderField(out, 0UZ, lengthBits_, value);
            return;
        }
        for (std::size_t byte = 0UZ; byte * 8UZ < lengthBits_; ++byte) {
            writeHeaderField(out, byte * 8UZ, 8UZ, (value >> static_cast<std::uint32_t>(byte * 8UZ)) & 0xFFU);
        }
    }

    std::size_t     lengthBits_;
    HeaderByteOrder order_;
    LengthCount     count_;
    std::uint32_t   maxValue_ = 0U;
};

/// @brief A length and some flags under the extended Golay(24,12,8) code, so a header that took three bit errors still
/// frames its packet and one that took four is refused rather than framing garbage.
///
/// ```
///   | Golay(24,12,8) codeword (24) |
///           information bits (12) = | flags (12 - length_bits) | length (length_bits) |
/// ```
///
/// The 24 wire items are read most significant bit first into a 24-bit word, so the first transmitted item lands in
/// bit 23 and the overall parity in bit 0 — which is exactly the layout `gr::fec::golay24Decode` documents, so the
/// mapping is stated once and shared rather than restated here. The correcting properties are the code's and are
/// inherited: minimum distance 8, three errors corrected, a fourth detected and reported as `valid == false`, which is
/// what makes `std::nullopt` the right answer to a header the code refused.
///
/// `flags_position` is a setting because both orderings occur and neither is derivable; `high` is the default, which
/// is the reading in which the length occupies the field's low bits. A `length_bits` of 12 leaves no flag bits, and
/// then `header_flags` is omitted rather than written as a constant zero.
///
/// The correction count is published as `header_corrected_errors` and deliberately not as the record vocabulary's
/// `corrected_errors`, which accumulates across a chain: a header correction and a payload correction are corrections
/// in different fields, and summing them would make a record whose header took three errors look like one whose
/// payload did.
class LengthGolay24Header {
public:
    static constexpr std::size_t kCodewordItems = 24UZ;
    static constexpr std::size_t kInfoBits      = 12UZ;

    LengthGolay24Header(std::size_t lengthBits, FlagsPosition flags, LengthCount count) : lengthBits_(lengthBits), flags_(flags), count_(count) {
        if (lengthBits < 1UZ || lengthBits > kInfoBits) {
            throw std::invalid_argument("gr::digital::LengthGolay24Header: length_bits must be in [1, 12], got " + std::to_string(lengthBits));
        }
        validateLengthCount(count);
        flagBits_ = kInfoBits - lengthBits;
        maxValue_ = (std::uint32_t{1} << lengthBits) - 1U;
        maxFlags_ = flagBits_ == 0UZ ? 0U : ((std::uint32_t{1} << flagBits_) - 1U);
    }

    [[nodiscard]] static constexpr std::size_t headerItems() noexcept { return kCodewordItems; }
    [[nodiscard]] std::size_t                  lengthBits() const noexcept { return lengthBits_; }
    [[nodiscard]] std::size_t                  flagBits() const noexcept { return flagBits_; }
    [[nodiscard]] FlagsPosition                flagsPosition() const noexcept { return flags_; }
    [[nodiscard]] const LengthCount&           count() const noexcept { return count_; }

    template<HeaderItem T>
    [[nodiscard]] std::optional<ParsedHeader> parse(std::span<const T> header) const {
        const gr::fec::GolayResult decoded = gr::fec::golay24Decode(readHeaderField(header, 0UZ, kCodewordItems));
        if (!decoded.valid) {
            return std::nullopt;
        }

        const std::uint32_t info   = static_cast<std::uint32_t>(decoded.info);
        const std::uint32_t length = flags_ == FlagsPosition::High ? (info & maxValue_) : (info >> static_cast<std::uint32_t>(flagBits_));
        const std::uint32_t flags  = flags_ == FlagsPosition::High ? (info >> static_cast<std::uint32_t>(lengthBits_)) : (info & maxFlags_);

        const std::optional<std::size_t> items = payloadItemsFromCount(count_, length, headerItems());
        if (!items.has_value()) {
            return std::nullopt;
        }

        property_map meta{{"header_corrected_errors", static_cast<std::uint64_t>(decoded.errors)}};
        if (flagBits_ != 0UZ) {
            meta["header_flags"] = static_cast<std::uint64_t>(flags);
        }
        return ParsedHeader{*items, std::move(meta)};
    }

    template<HeaderItem T>
    [[nodiscard]] bool format(std::size_t payloadItems, const property_map& extra, std::span<T> out) const {
        if (out.size() < headerItems()) {
            return false;
        }
        const std::optional<std::uint32_t> value = countFromPayloadItems(count_, payloadItems, headerItems(), maxValue_);
        if (!value.has_value()) {
            return false;
        }

        std::uint32_t flags = 0U;
        if (const auto entry = extra.find("header_flags"); entry != extra.end()) {
            const auto* carried = entry->second.get_if<std::uint64_t>();
            if (carried == nullptr || *carried > static_cast<std::uint64_t>(maxFlags_)) {
                return false;
            }
            flags = static_cast<std::uint32_t>(*carried);
        }

        const std::uint32_t info = flags_ == FlagsPosition::High ? ((flags << static_cast<std::uint32_t>(lengthBits_)) | *value) : ((*value << static_cast<std::uint32_t>(flagBits_)) | flags);
        writeHeaderField(out, 0UZ, kCodewordItems, gr::fec::golay24Encode(static_cast<std::uint16_t>(info)));
        return true;
    }

private:
    std::size_t   lengthBits_;
    FlagsPosition flags_;
    LengthCount   count_;
    std::size_t   flagBits_ = 0UZ;
    std::uint32_t maxValue_ = 0U;
    std::uint32_t maxFlags_ = 0U;
};

using HeaderFormat = std::variant<LengthCrcHeader, LengthRepeatedHeader, FixedLengthHeader, LengthPlainHeader, LengthGolay24Header>;

static_assert(LengthPlainHeader::kMaxLengthBits <= kMaxHeaderItems, "a plain length field must fit the framer's fixed header buffer");
static_assert(LengthGolay24Header::kCodewordItems <= kMaxHeaderItems, "a Golay-coded header must fit the framer's fixed header buffer");

[[nodiscard]] inline std::size_t headerItemsOf(const HeaderFormat& format) {
    return std::visit([](const auto& layout) { return layout.headerItems(); }, format);
}

template<HeaderItem T>
[[nodiscard]] inline std::optional<ParsedHeader> parseHeader(const HeaderFormat& format, std::span<const T> header) {
    return std::visit([header](const auto& layout) { return layout.template parse<T>(header); }, format);
}

template<HeaderItem T>
[[nodiscard]] inline bool formatHeader(const HeaderFormat& format, std::size_t payloadItems, const property_map& extra, std::span<T> out) {
    return std::visit([payloadItems, &extra, out](const auto& layout) { return layout.template format<T>(payloadItems, extra, out); }, format);
}

} // namespace gr::digital

#endif // GNURADIO_ALGORITHM_HEADER_FORMAT_HPP
