#ifndef GNURADIO_ALGORITHM_DELIMITER_HPP
#define GNURADIO_ALGORITHM_DELIMITER_HPP

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gnuradio-4.0/algorithm/digital/BitPacking.hpp>

/**
 * @brief Frames delimited by a reserved item pattern, the oldest way a protocol says where a frame ends.
 *
 * HDLC, AX.25 and synchronous PPP put a flag byte `01111110` between frames and keep it out of the payload by bit
 * stuffing. SLIP and KISS put `0xC0` between frames and keep it out by byte escaping. NMEA 0183 opens with `$`,
 * closes with `CR LF` and relies on a printable-ASCII payload. None of them carries a length field, because the
 * length is not known until the frame ends.
 *
 * A delimiter is spelled once, as a bit string of `'0'` and `'1'` characters, most significant bit first in wire
 * order. An input item carries `bitsPerItem` bits, so a bit-level and a byte-level delimiter differ in exactly one
 * number: HDLC is `bitsPerItem = 1` with an eight-item delimiter, SLIP is `bitsPerItem = 8` with a one-item one.
 * The scanner shifts each item's bits into one `std::uint64_t` register and compares for equality, which bounds a
 * delimiter at 64 bits and costs one shift, one OR, one AND and one compare per item.
 *
 * The register sees every raw item and transparency decoding never touches it. That is uniform across both
 * transparency codes and it is deliberate: excluding an escaped pair from matching would leave a hole in the
 * register's history, and the items either side of the hole become adjacent, so a two-item delimiter `16 16` would
 * be forged by the wire sequence `16 ESC X 16`, which contains no delimiter at all. `configureDelimiter()` establishes the
 * same guarantee by a settings-time check instead, at no runtime cost and with no hole.
 *
 * A match is known only at the delimiter's last item, by which time its earlier items have already been seen, so
 * the scanner holds the newest `endItems - 1` decoded items in a ring and appends an item to the frame buffer only
 * once it is that many items old. On a match the ring is cleared, which discards exactly the delimiter's body. The
 * frame buffer is therefore exactly `maxPayloadItems` items and the length test is on the number actually appended.
 * The ring is empty for every one-item delimiter, so SLIP, KISS and PPP pay nothing for it.
 *
 * `configureDelimiter()` validates and is the only function here that can throw a settings error. It throws
 * `std::invalid_argument` and leaves the configuration as it was, so a rejected setting cannot half-apply.
 * `DelimiterScanner::push()` allocates nothing: every buffer is sized once by `prepare()`, from the stated bound
 * and never from anything the stream said.
 */
namespace gr::digital {

/// @brief How a protocol keeps its delimiter out of the payload.
enum class Transparency : std::uint8_t { None, BitStuffing, ByteEscape };

/// @brief What opens a frame: the closing delimiter itself, a separate opening delimiter, or nothing at all.
enum class FrameOpen : std::uint8_t { Delimiter, Start, Immediate };

/// @brief Which port a closed region belongs on, and under which refusal.
enum class FrameOutcome : std::uint8_t { Emitted, Overrun, Undersize, Unaligned };

/// @brief What one input item did to the machine.
enum class ScanEvent : std::uint8_t {
    None,    ///< the item was consumed and no region ended
    Record,  ///< a region ended and a record is ready; `frameOutcome()` says which port it belongs on
    Idle,    ///< a delimiter closed a region carrying no items, which is the absence of a frame rather than a short one
    Abort,   ///< an abort run abandoned the frame in progress
    Restart, ///< a start delimiter abandoned the frame in progress and opened a new one
};

/// @brief The name a block exposes for @p mode.
[[nodiscard]] constexpr std::string_view transparencyName(Transparency mode) noexcept {
    switch (mode) {
    case Transparency::BitStuffing: return std::string_view{"bit_stuffing"};
    case Transparency::ByteEscape: return std::string_view{"byte_escape"};
    default: break;
    }
    return std::string_view{"none"};
}

/// @brief The mode @p name selects; throws `std::invalid_argument`, quoting @p name, for anything but the three names.
[[nodiscard]] inline Transparency transparencyFromName(std::string_view name) {
    if (name == transparencyName(Transparency::None)) {
        return Transparency::None;
    }
    if (name == transparencyName(Transparency::BitStuffing)) {
        return Transparency::BitStuffing;
    }
    if (name == transparencyName(Transparency::ByteEscape)) {
        return Transparency::ByteEscape;
    }
    throw std::invalid_argument(std::format("gr::digital::transparencyFromName: transparency must be 'none', 'bit_stuffing' or 'byte_escape', got '{}'", name));
}

/// @brief The name a block exposes for @p opening.
[[nodiscard]] constexpr std::string_view frameOpenName(FrameOpen opening) noexcept {
    switch (opening) {
    case FrameOpen::Start: return std::string_view{"start"};
    case FrameOpen::Immediate: return std::string_view{"immediate"};
    default: break;
    }
    return std::string_view{"delimiter"};
}

/// @brief The opening rule @p name selects; throws `std::invalid_argument`, quoting @p name, for anything but the three names.
[[nodiscard]] inline FrameOpen frameOpenFromName(std::string_view name) {
    if (name == frameOpenName(FrameOpen::Delimiter)) {
        return FrameOpen::Delimiter;
    }
    if (name == frameOpenName(FrameOpen::Start)) {
        return FrameOpen::Start;
    }
    if (name == frameOpenName(FrameOpen::Immediate)) {
        return FrameOpen::Immediate;
    }
    throw std::invalid_argument(std::format("gr::digital::frameOpenFromName: frame_open must be 'delimiter', 'start' or 'immediate', got '{}'", name));
}

/// @brief The refusal @p outcome names, as a record's `discard_reason`; empty for a frame that is not refused.
[[nodiscard]] constexpr std::string_view discardReasonName(FrameOutcome outcome) noexcept {
    switch (outcome) {
    case FrameOutcome::Overrun: return std::string_view{"overrun"};
    case FrameOutcome::Undersize: return std::string_view{"undersize"};
    case FrameOutcome::Unaligned: return std::string_view{"unaligned"};
    default: break;
    }
    return std::string_view{};
}

/**
 * @brief The `(escaped, original)` pairs a substitution map needs in order to express an XOR escape.
 *
 * RFC 1662 escapes each control character by XOR with `0x20`, and PPP's default asynchronous control character map
 * covers `0x00` to `0x1F` plus `0x7D` and `0x7E`, which is 34 entries. Written out as a literal list of 68 numbers
 * that would be a transcription hazard; written as one call it is checkable by eye. SLIP's own escape is not any
 * XOR, so the substitution map is the only runtime mechanism and this helper builds one rather than being a second.
 */
[[nodiscard]] inline std::vector<std::pair<std::uint8_t, std::uint8_t>> escapeMapFromXor(std::uint8_t xorMask, std::span<const std::uint8_t> originals) {
    std::vector<std::pair<std::uint8_t, std::uint8_t>> pairs;
    pairs.reserve(originals.size());
    for (const std::uint8_t original : originals) {
        pairs.emplace_back(static_cast<std::uint8_t>(original ^ xorMask), original);
    }
    return pairs;
}

namespace detail {

/// @brief One iff non-zero for a byte item, `x >= 0` for a soft one; a NaN slices to zero, every IEEE comparison with it being false.
[[nodiscard]] inline constexpr bool sliceBit(std::uint8_t item) noexcept { return item != 0U; }

template<std::floating_point T>
[[nodiscard]] inline constexpr bool sliceBit(T item) noexcept {
    return item >= T{0};
}

/// @brief The low @p width bits of @p value, in place under `MsbFirst` and reversed under `LsbFirst`.
[[nodiscard]] inline constexpr std::uint8_t orderedLowBits(unsigned value, unsigned width, BitOrder order) noexcept {
    const unsigned low = value & ((1U << width) - 1U);
    if (order == BitOrder::MsbFirst) {
        return static_cast<std::uint8_t>(low);
    }
    unsigned reversed = 0U;
    for (unsigned bit = 0U; bit < width; ++bit) {
        reversed = (reversed << 1U) | ((low >> bit) & 1U);
    }
    return static_cast<std::uint8_t>(reversed);
}

/// @brief The `width`-bit item values a delimiter's bit string spells, most significant bit of each item first.
[[nodiscard]] inline std::vector<std::uint8_t> delimiterItemValues(std::string_view bits, unsigned width) {
    std::vector<std::uint8_t> values;
    values.reserve(bits.size() / width);
    for (std::size_t at = 0UZ; at + width <= bits.size(); at += width) {
        unsigned value = 0U;
        for (unsigned bit = 0U; bit < width; ++bit) {
            value = (value << 1U) | (bits[at + bit] == '1' ? 1U : 0U);
        }
        values.push_back(static_cast<std::uint8_t>(value));
    }
    return values;
}

/// @brief The item values a delimiter's bit string is put on the wire as, each item's bits laid out in @p order.
[[nodiscard]] inline std::vector<std::uint8_t> delimiterEmitValues(std::string_view bits, unsigned width, BitOrder order) {
    std::vector<std::uint8_t> values = delimiterItemValues(bits, width);
    for (std::uint8_t& value : values) {
        value = orderedLowBits(value, width, order);
    }
    return values;
}

} // namespace detail

/**
 * @brief One delimiter framing: what a protocol states, and what `configureDelimiter()` derives from it.
 *
 * The derived members are written by `configureDelimiter()` and are never set directly. An empty `endDelimiter` and a
 * `maxPayloadItems` of `0` are the two unset states rather than errors: there is no universal delimiter and no
 * universal bound, so a scanner carrying either is inert and the block that owns it refuses to start.
 */
struct DelimiterConfig {
    std::string                                        endDelimiter{};   ///< the closing delimiter, `'0'`/`'1'` characters, MSB first in wire order
    std::string                                        startDelimiter{}; ///< the opening delimiter, required and non-empty iff `frameOpen == Start`
    FrameOpen                                          frameOpen      = FrameOpen::Delimiter;
    unsigned                                           bitsPerItem    = 1U;                 ///< bits each input item carries, `[1, 8]`
    BitOrder                                           bitOrder       = BitOrder::MsbFirst; ///< order of an item's bits on the wire, meaningful only above one bit per item
    Transparency                                       transparency   = Transparency::None;
    unsigned                                           stuffAfterOnes = 5U;                  ///< bit stuffing: a `0` arriving at exactly this consecutive-ones count is removed
    unsigned                                           abortOnes      = 7U;                  ///< bit stuffing: this many consecutive `1`s abandons the frame; exceeds `stuffAfterOnes` and the delimiters' longest run of ones; `0` disables
    std::uint8_t                                       escapeItem     = 0U;                  ///< byte escaping: the introducer's item value
    std::vector<std::pair<std::uint8_t, std::uint8_t>> escapeMap{};                          ///< byte escaping: `(escaped, original)` pairs
    std::size_t                                        maxPayloadItems = 0UZ;                ///< the largest payload a frame may carry, in decoded items; `0` is unset
    std::size_t                                        minPayloadItems = 1UZ;                ///< a closed region shorter than this is refused as undersize
    unsigned                                           packBits        = 0U;                 ///< `0` emits decoded items unchanged; `1..8` repacks the decoded bits into items of that width
    BitOrder                                           packOrder       = BitOrder::MsbFirst; ///< the order the pack stage assembles bits in

    std::uint64_t                 endBits       = 0ULL;  ///< the closing delimiter's bits, right-aligned
    std::uint64_t                 startBits     = 0ULL;  ///< the opening delimiter's bits, right-aligned
    std::uint64_t                 mask          = 0ULL;  ///< the low `endDelimiter.size()` bits
    std::uint64_t                 startMask     = 0ULL;  ///< the low `startDelimiter.size()` bits
    std::size_t                   endItems      = 0UZ;   ///< the closing delimiter's length in items; `0` is the unset state
    std::size_t                   startItems    = 0UZ;   ///< the opening delimiter's length in items
    bool                          distinctStart = false; ///< whether a second register and compare run beside the first
    std::array<std::uint8_t, 256> itemTable{};           ///< an item's register bits, masked to `bitsPerItem` and ordered by `bitOrder`
    std::array<std::uint8_t, 256> escapeTable{};         ///< the identity, overwritten at each escaped value
    std::array<bool, 256>         escapeMapped{};        ///< whether a value appears as an `escaped` value in the map
    std::array<bool, 256>         escapeOriginal{};      ///< whether a value appears as an `original` value in the map
    BitRepack                     pack{};                ///< the pack stage's conversion, configured when `packBits` is not zero
};

/**
 * @brief Validates @p cfg and derives the registers, the masks and the tables the scanner runs on.
 *
 * Everything the settings can get wrong is refused here, where it costs one pass over at most 64 bits, rather than
 * on the item path. Three of the refusals are what make the framing correct rather than merely consistent. Under bit
 * stuffing the delimiter must be invariant under the destuffing rule, so that clearing the delay ring discards
 * exactly the delimiter's body; it must carry a run of more than `stuffAfterOnes` ones, so that a conforming
 * encoder can never produce it inside a payload; and `abortOnes`, where it is not zero, must exceed the delimiters'
 * own longest run of ones, or the abort fires inside the delimiter closing a frame. The last two properties are
 * arithmetic and are therefore refusals; under byte escaping the unforgeability equivalent depends on a peer this
 * code cannot inspect, so it is a warning from `unforgeabilityWarning()` instead.
 *
 * @throws std::invalid_argument leaving @p cfg as it was, so a rejected setting cannot half-apply.
 */
inline void configureDelimiter(DelimiterConfig& cfg) {
    const std::string_view endText(cfg.endDelimiter);
    const std::string_view startText(cfg.startDelimiter);
    const unsigned         width = cfg.bitsPerItem;

    if (width < 1U || width > 8U) {
        throw std::invalid_argument(std::format("gr::digital::configureDelimiter: bitsPerItem must be in [1, 8], got {}", width));
    }
    if (cfg.packBits > 8U) {
        throw std::invalid_argument(std::format("gr::digital::configureDelimiter: packBits must be 0, for no packing, or in [1, 8], got {}", cfg.packBits));
    }
    if (cfg.packBits != 0U && width != 1U) {
        throw std::invalid_argument(std::format("gr::digital::configureDelimiter: the pack stage assembles bits into items and needs bitsPerItem 1, got {}", width));
    }
    if (cfg.minPayloadItems == 0UZ) {
        throw std::invalid_argument("gr::digital::configureDelimiter: minPayloadItems must be at least 1, a region with no items being the absence of a frame rather than a short one");
    }
    if (cfg.maxPayloadItems > 2147483647UZ) {
        throw std::invalid_argument(std::format("gr::digital::configureDelimiter: maxPayloadItems is {}; a record states its extent as a signed 32-bit count, so a longer frame could not describe itself", cfg.maxPayloadItems));
    }
    if (cfg.maxPayloadItems != 0UZ && cfg.minPayloadItems > cfg.maxPayloadItems) {
        throw std::invalid_argument(std::format("gr::digital::configureDelimiter: minPayloadItems {} exceeds maxPayloadItems {}", cfg.minPayloadItems, cfg.maxPayloadItems));
    }
    if (cfg.stuffAfterOnes < 2U || cfg.stuffAfterOnes > 62U) {
        throw std::invalid_argument(std::format("gr::digital::configureDelimiter: stuffAfterOnes must be in [2, 62], got {}", cfg.stuffAfterOnes));
    }
    if (cfg.abortOnes != 0U && cfg.abortOnes <= cfg.stuffAfterOnes) {
        throw std::invalid_argument(std::format("gr::digital::configureDelimiter: abortOnes {} must exceed stuffAfterOnes {}, or an abort would fire before a stuff and make the code unusable", cfg.abortOnes, cfg.stuffAfterOnes));
    }
    if (cfg.frameOpen == FrameOpen::Start && startText.empty()) {
        throw std::invalid_argument("gr::digital::configureDelimiter: frameOpen 'start' needs a startDelimiter");
    }
    if (cfg.frameOpen != FrameOpen::Start && !startText.empty()) {
        throw std::invalid_argument(std::format("gr::digital::configureDelimiter: frameOpen '{}' consults no startDelimiter, so setting one is a settings error rather than a harmless extra", frameOpenName(cfg.frameOpen)));
    }
    if (cfg.transparency == Transparency::BitStuffing && width != 1U) {
        throw std::invalid_argument(std::format("gr::digital::configureDelimiter: bit stuffing is a bit-level code and needs bitsPerItem 1, got {}", width));
    }
    if (cfg.transparency == Transparency::ByteEscape && width != 8U) {
        throw std::invalid_argument(std::format("gr::digital::configureDelimiter: byte escaping needs a byte, so bitsPerItem must be 8, got {}", width));
    }

    std::array<std::uint8_t, 256> escapeTable{};
    std::array<bool, 256>         escapeMapped{};
    std::array<bool, 256>         escapeOriginal{};
    for (unsigned value = 0U; value < 256U; ++value) {
        escapeTable[value] = static_cast<std::uint8_t>(value);
    }
    if (cfg.escapeMap.size() > 256UZ) {
        throw std::invalid_argument(std::format("gr::digital::configureDelimiter: the escape map holds {} pairs; one per byte value is the most that can be distinguished", cfg.escapeMap.size()));
    }
    for (const auto& [escaped, original] : cfg.escapeMap) {
        if (escapeMapped[escaped]) {
            throw std::invalid_argument(std::format("gr::digital::configureDelimiter: the escape map lists the escaped value {} twice, so the decode of that pair is ambiguous", escaped));
        }
        escapeMapped[escaped]    = true;
        escapeTable[escaped]     = original;
        escapeOriginal[original] = true;
    }

    const auto parse = [width](std::string_view text, std::string_view which) -> std::pair<std::uint64_t, std::uint64_t> {
        if (text.size() > 64UZ) {
            throw std::invalid_argument(std::format("gr::digital::configureDelimiter: {} is {} bits; the match register is one 64-bit word", which, text.size()));
        }
        if (text.size() % width != 0UZ) {
            throw std::invalid_argument(std::format("gr::digital::configureDelimiter: {} is {} bits, which is not a whole number of {}-bit items", which, text.size(), width));
        }
        std::uint64_t word = 0ULL;
        for (const char character : text) {
            if (character != '0' && character != '1') {
                throw std::invalid_argument(std::format("gr::digital::configureDelimiter: {} must be '0' and '1' characters only, got '{}' in \"{}\"", which, character, text));
            }
            word = (word << 1U) | (character == '1' ? 1ULL : 0ULL);
        }
        // the shift count never equals the operand's width, which is undefined behavior rather than an all-ones mask
        const std::uint64_t bitMask = text.size() == 64UZ ? ~0ULL : ((1ULL << text.size()) - 1ULL);
        return {word & bitMask, bitMask};
    };

    const auto longestOneRun = [](std::string_view text) {
        unsigned run     = 0U;
        unsigned longest = 0U;
        for (const char character : text) {
            run     = character == '1' ? run + 1U : 0U;
            longest = std::max(longest, run);
        }
        return longest;
    };

    const auto [endWord, endMask]     = parse(endText, "endDelimiter");
    const auto [startWord, startMask] = parse(startText, "startDelimiter");

    // The abort counter runs over the delimiter's own ones, the frame the delimiter closes still being open, so an
    // abort run no longer than the delimiter's fires inside every closing delimiter and abandons every frame. The
    // bound is therefore the delimiters' longest run of ones and not stuffAfterOnes alone.
    if (cfg.transparency == Transparency::BitStuffing && cfg.abortOnes != 0U) {
        const unsigned delimiterOnes = std::max(longestOneRun(startText), longestOneRun(endText));
        if (cfg.abortOnes <= delimiterOnes) {
            throw std::invalid_argument(std::format("gr::digital::configureDelimiter: abortOnes {} does not exceed the run of {} consecutive ones the delimiters carry, so the abort would fire inside the delimiter that closes a frame and abandon every frame", cfg.abortOnes, delimiterOnes));
        }
    }

    if (!endText.empty() && cfg.transparency == Transparency::BitStuffing) {
        const unsigned longest = longestOneRun(endText);
        if (longest <= cfg.stuffAfterOnes) {
            throw std::invalid_argument(std::format("gr::digital::configureDelimiter: coded data may carry runs of up to {} consecutive ones and \"{}\" carries at most {}, so a conforming encoder can produce this delimiter inside a payload", cfg.stuffAfterOnes, endText, longest));
        }
        // A closing delimiter is entered carrying the payload's trailing run of ones — anywhere from zero to
        // stuffAfterOnes - 1 on a conforming coded stream — and the decoder's counter cannot know the delimiter has
        // begun. Invariance must therefore hold from every reachable entering count, not only from zero:
        // "101111110" at stuffAfterOnes = 5 survives the zero-entry walk yet loses its second bit when entered
        // behind four ones.
        for (unsigned entering = 0U; entering < cfg.stuffAfterOnes; ++entering) {
            unsigned ones = entering;
            for (const char character : endText) {
                if (character == '1') {
                    ++ones;
                } else {
                    if (ones == cfg.stuffAfterOnes) {
                        throw std::invalid_argument(std::format("gr::digital::configureDelimiter: the destuffing rule removes a zero arriving at {} consecutive ones and \"{}\" contains one when entered behind a payload ending in {} ones, so the delimiter is not invariant under the decoder", cfg.stuffAfterOnes, endText, entering));
                    }
                    ones = 0U;
                }
            }
        }
    }
    if (!endText.empty() && cfg.transparency == Transparency::ByteEscape) {
        for (const std::uint8_t value : detail::delimiterItemValues(endText, width)) {
            if (value == cfg.escapeItem) {
                throw std::invalid_argument(std::format("gr::digital::configureDelimiter: the delimiter carries the escape introducer {}, which the decoder removes, so the delimiter is not invariant under the decoder", value));
            }
            if (escapeMapped[value]) {
                throw std::invalid_argument(std::format("gr::digital::configureDelimiter: the delimiter carries {}, which the escape map lists as an escaped value", value));
            }
        }
    }

    cfg.endBits        = endWord;
    cfg.mask           = endMask;
    cfg.endItems       = endText.size() / width;
    cfg.startBits      = startWord;
    cfg.startMask      = startMask;
    cfg.startItems     = startText.size() / width;
    cfg.distinctStart  = cfg.frameOpen == FrameOpen::Start;
    cfg.escapeTable    = escapeTable;
    cfg.escapeMapped   = escapeMapped;
    cfg.escapeOriginal = escapeOriginal;
    for (unsigned value = 0U; value < 256U; ++value) {
        cfg.itemTable[value] = detail::orderedLowBits(value, width, cfg.bitOrder);
    }
    if (cfg.packBits != 0U) {
        configureRepack(cfg.pack, 1U, cfg.packBits, BitOrder::MsbFirst, cfg.packOrder);
    }
}

/**
 * @brief What `configureDelimiter()` can only suspect, stated rather than refused.
 *
 * Under byte escaping the delimiter's absence from a payload depends on the peer's encoder escaping it, which
 * nothing here can verify or compel. The local half is checkable — every delimiter item should appear as an
 * `original` in the escape map, meaning a conforming encoder would have escaped it — and it is only a warning
 * because a protocol may legitimately guarantee the delimiter's absence by other means, such as a seven-bit payload
 * under a high-bit delimiter.
 *
 * @return the warning, or an empty string when there is nothing to warn about.
 */
[[nodiscard]] inline std::string unforgeabilityWarning(const DelimiterConfig& cfg) {
    if (cfg.transparency != Transparency::ByteEscape || cfg.endDelimiter.empty()) {
        return {};
    }
    for (const std::uint8_t value : detail::delimiterItemValues(cfg.endDelimiter, cfg.bitsPerItem)) {
        if (!cfg.escapeOriginal[value]) {
            return std::format("the delimiter carries {}, which the escape map does not list as an original value, so a conforming encoder would not escape it and a payload may forge the delimiter", value);
        }
    }
    return {};
}

/**
 * @brief Validates @p cfg for the transmit side: `configureDelimiter()`'s whole table, and then what only an encoder can prove.
 *
 * A receiver cannot check that its peer escapes the delimiter, which is why `unforgeabilityWarning()` states that half
 * rather than refusing it. An encoder is that peer, so here the property is local and the answer is a refusal: under
 * byte escaping, every item value the encoder puts on the wire for a delimiter — and the introducer itself — must
 * appear as an `original` in the escape map. A value the map does not name is a value the encoder emits unescaped, and
 * a payload carrying it would cut the frame in half at the receiver. SLIP and PPP pass; a map that drops the
 * introducer's own pair is the misconfiguration this catches.
 *
 * A repeated `original` is accepted, where a repeated `escaped` is refused. The asymmetry is deliberate: two spellings
 * of one original decode to the same value, so either round-trips and the encoder takes the last one the map names,
 * while two originals under one escaped value have no unambiguous decode at all.
 *
 * @throws std::invalid_argument leaving @p cfg as it was, which is `configureDelimiter()`'s own guarantee over both stages.
 */
inline void framerConfigure(DelimiterConfig& cfg) {
    DelimiterConfig staged = cfg;
    configureDelimiter(staged);

    if (staged.transparency == Transparency::ByteEscape) {
        const auto require = [&staged](std::uint8_t value, std::string_view which) {
            if (!staged.escapeOriginal[value]) {
                throw std::invalid_argument(std::format("gr::digital::framerConfigure: {} is {}, which the escape map does not list as an original value, so an encoder would emit it unescaped and a payload could forge the delimiter", which, value));
            }
        };
        for (const std::uint8_t value : detail::delimiterEmitValues(staged.endDelimiter, staged.bitsPerItem, staged.bitOrder)) {
            require(value, "an item of the end delimiter");
        }
        if (staged.frameOpen == FrameOpen::Start) {
            for (const std::uint8_t value : detail::delimiterEmitValues(staged.startDelimiter, staged.bitsPerItem, staged.bitOrder)) {
                require(value, "an item of the start delimiter");
            }
        }
        require(staged.escapeItem, "the escape introducer");
    }

    cfg = std::move(staged);
}

/// @brief What one record's encode produced, in the numbers a caller reports.
struct EncodeResult {
    std::size_t emitted  = 0UZ; ///< items written, both delimiters included
    std::size_t inserted = 0UZ; ///< items the transparency encoder added: stuffed zeros, or escape introducers
    std::size_t forged   = 0UZ; ///< payload items at which the match register held the closing delimiter, under `None` only
};

/**
 * @brief The largest framed size a payload of @p payloadItems wire items can reach, delimiters included.
 *
 * Bit stuffing's worst case is one insertion every `stuffAfterOnes` items, which an all-ones payload reaches exactly;
 * byte escaping's is one insertion per item, so twice the payload; `None` expands by nothing. The bound sizes a
 * buffer once and never sizes it from anything a payload said.
 */
[[nodiscard]] inline std::size_t framedItemsBound(const DelimiterConfig& cfg, std::size_t payloadItems) noexcept {
    std::size_t coded = payloadItems;
    if (cfg.transparency == Transparency::BitStuffing) {
        coded += payloadItems / cfg.stuffAfterOnes;
    } else if (cfg.transparency == Transparency::ByteEscape) {
        coded += payloadItems;
    }
    const std::size_t opening = cfg.frameOpen == FrameOpen::Start ? cfg.startItems : (cfg.frameOpen == FrameOpen::Delimiter ? cfg.endItems : 0UZ);
    return coded + opening + cfg.endItems;
}

/**
 * @brief The scanner's inverse: one payload becomes one delimited, transparency-coded frame.
 *
 * The assembly is the opening delimiter — the start delimiter, the closing delimiter, or nothing at all, as
 * `frameOpen` says — then the coded payload, then the closing delimiter. Delimiter items are emitted verbatim: they
 * are never stuffed and never escaped, which is what leaves the receiver's match register the only thing that has to
 * recognize them, and it is why `framerConfigure()` refuses a framing whose delimiter the encoder could not keep out
 * of a payload.
 *
 * Under bit stuffing the encoder runs the decoder's own counter of consecutive ones over the payload it emits, and
 * inserts a zero immediately after the payload one that brings the count to exactly `stuffAfterOnes`. The counter
 * starts each frame at zero, which is where the decoder starts it too: the machine clears it as the opening delimiter
 * opens the frame. The insertion after the payload's last item is the one the closing delimiter depends on — without
 * it a payload ending in exactly `stuffAfterOnes` ones would have the delimiter's leading zero removed as a stuffed
 * bit, and one payload item would be discarded with the delimiter's body.
 *
 * Under `None` nothing is coded and nothing can be, so the encoder instead runs the receiver's own match register over
 * the payload items it emits and counts every match. Each one is a delimiter the receiver will honor inside what was
 * meant to be one frame.
 *
 * `prepare()` sizes the pack stage's scratch from the bound, and `encode()` allocates only where the caller's buffer
 * is smaller than `framedItemsBound()` says a frame can be.
 */
struct DelimiterEncoder {
    DelimiterConfig config{};

    std::array<std::uint8_t, 256> _escapeReverse{}; ///< the identity, overwritten at each `original` with its escaped spelling
    std::vector<std::uint8_t>     _opening{};       ///< the opening delimiter's item values, in wire order
    std::vector<std::uint8_t>     _closing{};       ///< the closing delimiter's, the same way
    BitRepack                     _unpack{};        ///< the pack stage's conversion in reverse, configured when `packBits` is not zero
    std::vector<std::uint8_t>     _payload{};       ///< the unpacked payload, when that stage runs
    std::size_t                   _endBitLength = 0UZ;

    /// @brief Adopts an already validated framing and derives the reverse table, the delimiter items and the unpack stage.
    void prepare(DelimiterConfig configured) {
        config = std::move(configured);
        for (unsigned value = 0U; value < 256U; ++value) {
            _escapeReverse[value] = static_cast<std::uint8_t>(value);
        }
        for (const auto& [escaped, original] : config.escapeMap) {
            _escapeReverse[original] = escaped; // a repeated original encodes by the last pair naming it
        }

        _closing      = detail::delimiterEmitValues(config.endDelimiter, config.bitsPerItem, config.bitOrder);
        _opening      = config.frameOpen == FrameOpen::Start ? detail::delimiterEmitValues(config.startDelimiter, config.bitsPerItem, config.bitOrder) : (config.frameOpen == FrameOpen::Delimiter ? _closing : std::vector<std::uint8_t>{});
        _endBitLength = config.endItems * config.bitsPerItem;

        _payload.clear();
        if (config.packBits != 0U) {
            configureRepack(_unpack, config.packBits, 1U, config.packOrder, BitOrder::MsbFirst);
            _payload.reserve(config.maxPayloadItems);
        }
    }

    /// @brief Whether a delimiter and a bound have both been stated; an encoder missing either is inert.
    [[nodiscard]] bool configured() const noexcept { return config.endItems != 0UZ && config.maxPayloadItems != 0UZ; }

    /// @brief The wire items a record of @p recordItems items carries, which is what the bound is measured against.
    [[nodiscard]] std::size_t unpackedItems(std::size_t recordItems) const noexcept { return config.packBits == 0U ? recordItems : recordItems * config.packBits; }

    /// @brief Frames @p record into @p out, replacing whatever it held.
    [[nodiscard]] EncodeResult encode(std::span<const std::uint8_t> record, std::vector<std::uint8_t>& out) {
        std::span<const std::uint8_t> payload = record;
        if (config.packBits != 0U) {
            // the same conversion the scanner's pack stage runs, read the other way round, so no unpacking code is written twice
            _payload.resize(record.size() * config.packBits);
            repack(_unpack, record, std::span<std::uint8_t>(_payload));
            payload = std::span<const std::uint8_t>(_payload);
        }

        EncodeResult result;
        out.clear();
        out.reserve(framedItemsBound(config, payload.size()));
        out.insert(out.end(), _opening.begin(), _opening.end());

        // under `None` the receiver's own register runs here, seeded by the opening delimiter exactly as the receiver seeds it
        std::uint64_t match   = 0ULL;
        std::size_t   shifted = 0UZ;
        if (config.transparency == Transparency::None) {
            for (const std::uint8_t item : _opening) {
                advance(match, shifted, item);
            }
        }

        unsigned ones = 0U;
        for (const std::uint8_t item : payload) {
            switch (config.transparency) {
            case Transparency::BitStuffing:
                out.push_back(item);
                if (detail::sliceBit(item)) {
                    ++ones;
                    if (ones == config.stuffAfterOnes) {
                        out.push_back(0U);
                        ++result.inserted;
                        ones = 0U;
                    }
                } else {
                    ones = 0U;
                }
                break;
            case Transparency::ByteEscape:
                if (config.escapeOriginal[item]) {
                    out.push_back(config.escapeItem);
                    out.push_back(_escapeReverse[item]);
                    ++result.inserted;
                } else {
                    out.push_back(item);
                }
                break;
            default:
                out.push_back(item);
                advance(match, shifted, item);
                if (shifted >= _endBitLength && match == config.endBits) {
                    ++result.forged;
                }
                break;
            }
        }

        out.insert(out.end(), _closing.begin(), _closing.end());
        result.emitted = out.size();
        return result;
    }

private:
    /// @brief Shifts one item into the match register, exactly as the scanner does, and counts the bits it has seen.
    void advance(std::uint64_t& match, std::size_t& shifted, std::uint8_t item) const noexcept {
        const std::uint64_t bits = config.bitsPerItem == 1U ? (detail::sliceBit(item) ? 1ULL : 0ULL) : static_cast<std::uint64_t>(config.itemTable[item]);
        match                    = ((match << config.bitsPerItem) | bits) & config.mask;
        if (shifted < 64UZ) {
            shifted += config.bitsPerItem;
        }
    }
};

/// @brief What one item did to the machine, and where in the record it belongs.
struct ScanStep {
    ScanEvent   event   = ScanEvent::None;
    bool        matched = false; ///< the closing delimiter's register matched at this item, whatever the state was
    bool        inFrame = false; ///< the item lay inside an open frame's raw region when it arrived
    bool        removed = false; ///< the transparency decoder removed the item
    std::size_t index   = 0UZ;   ///< the decoded index this item's offset maps to, meaningful when `inFrame`
};

/// @brief Every event the scanner counts. `nItemsMasked` is a configuration warning; the rest are data the framing lost.
struct DelimiterCounters {
    std::uint64_t nOverrunFrames    = 0ULL; ///< frames that reached `maxPayloadItems` with no closing delimiter
    std::uint64_t nUndersizeFrames  = 0ULL; ///< closed frames carrying between one item and `minPayloadItems - 1`
    std::uint64_t nUnalignedFrames  = 0ULL; ///< closed frames whose decoded count is not a multiple of `packBits`
    std::uint64_t nIdleDelimiters   = 0ULL; ///< delimiters closing a zero-item region, which an idle link produces continuously
    std::uint64_t nAborts           = 0ULL; ///< abort runs under bit stuffing
    std::uint64_t nRestarts         = 0ULL; ///< start delimiters arriving while a frame was open
    std::uint64_t nEscapeViolations = 0ULL; ///< escaped items with no entry in the escape map
    std::uint64_t nItemsMasked      = 0ULL; ///< items whose bits above `bitsPerItem` were not zero
    std::uint64_t nItemsDiscarded   = 0ULL; ///< items consumed outside a frame, or belonging to a frame that was abandoned
};

/**
 * @brief The delimiter machine: an idle state, an open state, one match register per delimiter and one delay ring.
 *
 * In the open state the frame buffer holds exactly the transparency-decoded items strictly between the end of the
 * opening delimiter and the current item, less the newest `endItems - 1` of them, which are still in the delay ring
 * because they might turn out to be the closing delimiter's body. Every other property follows from that invariant.
 *
 * The per-item order is fixed and it matters: the item's bits enter the registers before the transparency decoder
 * sees the item, and a match is acted on before the item is decoded at all. That is what makes the ring exact — on
 * a match it holds the delimiter's first `endItems - 1` decoded items and nothing else — and it holds only because
 * `configureDelimiter()` has already established that the delimiter is invariant under the decoder.
 *
 * The scanner owns absolute input offsets, so a caller reads a record's first-item offset and an interior tag's
 * record index off it rather than repeating the arithmetic. `seek()` states where the next item sits in the stream.
 */
template<typename T>
requires(std::same_as<T, std::uint8_t> || std::floating_point<T>)
struct DelimiterScanner {
    DelimiterConfig   config{};
    DelimiterCounters counters{};

    enum class State : std::uint8_t { Idle, Open };

    State         _state         = State::Idle;
    std::uint64_t _register      = 0ULL;  ///< the closing delimiter's shift register
    std::uint64_t _startRegister = 0ULL;  ///< the opening delimiter's, when it is a distinct pattern
    std::size_t   _shifted       = 0UZ;   ///< bits shifted in since the last resynchronization, capped at the register's width
    std::size_t   _offset        = 0UZ;   ///< the absolute input offset of the next item
    std::size_t   _sampleStart   = 0UZ;   ///< the absolute input offset of the open frame's first raw payload item
    std::size_t   _removed       = 0UZ;   ///< items the decoder removed from the open frame so far
    std::size_t   _rawInFrame    = 0UZ;   ///< raw items consumed since the open frame's first payload item
    unsigned      _ones          = 0U;    ///< consecutive one items on the raw stream, for bit stuffing
    bool          _escaped       = false; ///< whether the previous item was the escape introducer

    std::vector<T> _frame{};  ///< the frame in progress, filled by eviction from the ring
    std::vector<T> _record{}; ///< the frame that closed, held until the next item arrives
    std::vector<T> _packed{}; ///< the pack stage's output, when the stage is on and the frame is aligned
    std::vector<T> _ring{};   ///< the newest `endItems - 1` decoded items, which may yet be the delimiter's body
    std::size_t    _ringFill  = 0UZ;
    std::size_t    _ringHead  = 0UZ;
    std::size_t    _ringItems = 0UZ; ///< the ring's capacity, `endItems - 1`, and zero for every one-item delimiter

    FrameOutcome _outcome      = FrameOutcome::Emitted;
    std::size_t  _frameStart   = 0UZ;   ///< the closed frame's first raw payload item, as an absolute input offset
    std::size_t  _frameRemoved = 0UZ;   ///< the closed frame's removed-item count
    std::size_t  _frameDecoded = 0UZ;   ///< the closed frame's decoded item count, before the pack stage
    bool         _framePacked  = false; ///< whether `frame()` reads the pack stage's output
    bool         _holding      = false; ///< whether a closed frame is still readable

    std::size_t _endBitLength   = 0UZ; ///< bits that must have been shifted in before the closing compare is meaningful
    std::size_t _startBitLength = 0UZ; ///< the same for the opening compare

    /// @brief Adopts an already configured framing, sizes every buffer from it, and clears the machine and the counters.
    ///
    /// A soft item carries one sliced bit and nothing else, so any other `bitsPerItem` is refused here: the scanner
    /// would otherwise run with a register that shifts in a zero for every item and match nothing, silently.
    void prepare(DelimiterConfig configured) {
        if constexpr (!std::same_as<T, std::uint8_t>) {
            if (configured.bitsPerItem != 1U) {
                throw std::invalid_argument(std::format("gr::digital::DelimiterScanner: a soft item carries one sliced bit, so bitsPerItem must be 1, got {}", configured.bitsPerItem));
            }
        }
        config          = std::move(configured);
        _endBitLength   = config.endItems * config.bitsPerItem;
        _startBitLength = config.startItems * config.bitsPerItem;
        _ringItems      = config.endItems == 0UZ ? 0UZ : config.endItems - 1UZ;
        _ring.assign(_ringItems, T{});
        _frame.clear();
        _record.clear();
        _packed.clear();
        // two frame buffers rather than one: the frame that closed stays readable while the next one accumulates,
        // and swapping them keeps that free of any allocation after prepare()
        _frame.reserve(config.maxPayloadItems);
        _record.reserve(config.maxPayloadItems);
        if (config.packBits != 0U) {
            _packed.reserve(config.maxPayloadItems / config.packBits);
        }
        counters = DelimiterCounters{};
        reset();
    }

    /// @brief Returns the machine to its initial state, leaving the configuration and the counters alone.
    void reset() noexcept {
        _state         = State::Idle;
        _register      = 0ULL;
        _startRegister = 0ULL;
        _shifted       = 0UZ;
        _sampleStart   = 0UZ;
        _removed       = 0UZ;
        _rawInFrame    = 0UZ;
        _ones          = 0U;
        _escaped       = false;
        _ringFill      = 0UZ;
        _ringHead      = 0UZ;
        _holding       = false;
        _frame.clear();
        _record.clear();
        _packed.clear();
    }

    /// @brief States the absolute input offset of the next item, which a caller resuming a span already knows.
    void seek(std::size_t absolute) noexcept { _offset = absolute; }

    /// @brief Whether a delimiter and a bound have both been stated; a scanner missing either is inert.
    [[nodiscard]] bool configured() const noexcept { return config.endItems != 0UZ && config.maxPayloadItems != 0UZ; }

    [[nodiscard]] bool        isOpen() const noexcept { return _state == State::Open; }
    [[nodiscard]] std::size_t offset() const noexcept { return _offset; }

    /// @brief The closed frame's payload, packed when the stage ran. Valid until the next `push()`.
    [[nodiscard]] std::span<const T> frame() const noexcept { return _framePacked ? std::span<const T>(_packed) : std::span<const T>(_record); }
    [[nodiscard]] FrameOutcome       frameOutcome() const noexcept { return _outcome; }
    [[nodiscard]] std::size_t        frameStart() const noexcept { return _frameStart; }
    [[nodiscard]] std::size_t        frameRemoved() const noexcept { return _frameRemoved; }
    /// @brief The closed frame's decoded item count before the pack stage, which an interior tag's index is measured against.
    [[nodiscard]] std::size_t frameDecodedItems() const noexcept { return _frameDecoded; }
    [[nodiscard]] bool        framePacked() const noexcept { return _framePacked; }

    /**
     * @brief Abandons any frame in progress and returns the machine to its idle state, counting the items lost.
     *
     * This is what a detector's trigger tag does. It is a resynchronization point rather than a gate: it guarantees
     * that a frame cannot straddle two bursts, and the registers are cleared with everything else because the
     * previous burst's bits are not part of this burst's delimiter.
     */
    void resynchronize() noexcept {
        abandon();
        _state         = State::Idle;
        _register      = 0ULL;
        _startRegister = 0ULL;
        _shifted       = 0UZ;
    }

    /// @brief Reads one input item and reports what it did.
    [[nodiscard]] ScanStep push(T item) {
        if (_holding) { // the frame the previous item closed has been read by now
            _record.clear();
            _packed.clear();
            _holding = false;
        }

        ScanStep          step{};
        const std::size_t at = _offset++;
        if (!configured()) { // no delimiter or no bound: inert rather than matching or accumulating something arbitrary
            ++counters.nItemsDiscarded;
            return step;
        }

        if (_state == State::Idle && config.frameOpen == FrameOpen::Immediate) {
            openAt(at); // there is no opening delimiter, so the machine opens at the first item and again after every close
        }
        const bool open = _state == State::Open;
        step.inFrame    = open;
        if (open) {
            step.index = (at - _sampleStart) - _removed;
        }

        const std::uint64_t bits = itemBits(item);
        _register                = ((_register << config.bitsPerItem) | bits) & config.mask;
        if (config.distinctStart) {
            _startRegister = ((_startRegister << config.bitsPerItem) | bits) & config.startMask;
        }
        if (_shifted < 64UZ) {
            _shifted += config.bitsPerItem;
        }

        const bool endMatch = _shifted >= _endBitLength && _register == config.endBits;
        step.matched        = endMatch;
        if (open && endMatch) {
            close(step, at);
            return step;
        }

        const bool startMatch = config.frameOpen == FrameOpen::Delimiter ? endMatch : (config.distinctStart && _shifted >= _startBitLength && _startRegister == config.startBits);
        if (startMatch) {
            if (open) { // a mid-frame start abandons rather than being read as data, which would cost every frame after a lost end delimiter
                abandon();
                ++counters.nRestarts;
                step.event = ScanEvent::Restart;
            }
            openAt(at + 1UZ);
            return step;
        }
        if (!open) {
            ++counters.nItemsDiscarded;
            return step;
        }

        ++_rawInFrame;
        T    value   = item;
        bool removed = false;
        if (config.transparency == Transparency::BitStuffing) {
            if (bits != 0ULL) {
                ++_ones;
                if (config.abortOnes != 0U && _ones == config.abortOnes) {
                    abandon();
                    ++counters.nAborts;
                    _state     = State::Idle;
                    step.event = ScanEvent::Abort;
                    return step;
                }
            } else {
                // exactly, not at or above: a zero arriving at a higher count is a delimiter's or an abort's tail
                removed = _ones == config.stuffAfterOnes;
                _ones   = 0U;
            }
        } else if constexpr (std::same_as<T, std::uint8_t>) {
            if (config.transparency == Transparency::ByteEscape) {
                if (_escaped) {
                    _escaped = false;
                    if (!config.escapeMapped[item]) {
                        ++counters.nEscapeViolations; // one corrupt byte costs one byte; the payload's own CRC is what is qualified to judge it
                    }
                    value = config.escapeTable[item];
                } else if (item == config.escapeItem) {
                    removed  = true;
                    _escaped = true; // the item after an introducer is never itself read as one, so two removals are never adjacent
                }
            }
        }

        if (removed) {
            ++_removed;
            step.removed = true;
            return step;
        }

        if (_ringItems == 0UZ) {
            append(step, value);
            return step;
        }
        if (_ringFill < _ringItems) { // the head is zero while the ring fills, so the next free slot is the fill count
            _ring[_ringFill] = value;
            ++_ringFill;
            return step;
        }
        const T evicted  = _ring[_ringHead];
        _ring[_ringHead] = value;
        _ringHead        = _ringHead + 1UZ == _ringItems ? 0UZ : _ringHead + 1UZ;
        append(step, evicted);
        return step;
    }

private:
    /// @brief The item's bits as the register reads them: one sliced bit, or the low `bitsPerItem` bits in `bitOrder`.
    [[nodiscard]] std::uint64_t itemBits(T item) noexcept {
        if (config.bitsPerItem == 1U) {
            return detail::sliceBit(item) ? 1ULL : 0ULL;
        }
        if constexpr (std::same_as<T, std::uint8_t>) {
            if ((static_cast<unsigned>(item) >> config.bitsPerItem) != 0U) {
                ++counters.nItemsMasked; // a mis-wired chain is a counted event rather than a working one that decodes nothing
            }
            return config.itemTable[item];
        } else {
            return 0ULL; // unreachable: prepare() refuses any width but one for a soft item
        }
    }

    /// @brief Opens a frame whose first raw payload item is at @p firstPayloadItem. The registers stand, so overlapping delimiters both match.
    void openAt(std::size_t firstPayloadItem) noexcept {
        _state       = State::Open;
        _sampleStart = firstPayloadItem;
        _ones        = 0U;
        _escaped     = false;
        discardFrame();
    }

    /// @brief Empties the frame in progress and its ring, without counting anything.
    void discardFrame() noexcept {
        _rawInFrame = 0UZ;
        _removed    = 0UZ;
        _ringFill   = 0UZ;
        _ringHead   = 0UZ;
        _frame.clear();
    }

    /// @brief Discards the frame in progress, counting its raw items. The caller states what the machine does next.
    void abandon() noexcept {
        counters.nItemsDiscarded += _rawInFrame;
        discardFrame();
    }

    /// @brief Appends one decoded item, or closes the frame at the bound when the item would carry it past.
    void append(ScanStep& step, T value) {
        if (_frame.size() >= config.maxPayloadItems) {
            ++counters.nItemsDiscarded; // the item that tripped the bound; everything after it is counted while idle
            ++counters.nOverrunFrames;
            publish(FrameOutcome::Overrun);
            step.event = ScanEvent::Record;
            discardFrame();
            _state = State::Idle; // under 'immediate' the next item re-opens; otherwise the next delimiter does
            return;
        }
        _frame.push_back(value);
    }

    /// @brief Closes the region the delimiter ending at @p at delimits, classifies it, and states what the machine does next.
    void close(ScanStep& step, std::size_t at) {
        _ringFill = 0UZ; // the ring holds exactly the delimiter's body, so clearing it discards exactly that
        _ringHead = 0UZ;

        const std::size_t decoded = _frame.size();
        if (decoded == 0UZ) {
            // an idle link is back-to-back delimiters, and one rejected record per pair would flood a link carrying no data
            ++counters.nIdleDelimiters;
            step.event = ScanEvent::Idle;
        } else {
            FrameOutcome outcome = FrameOutcome::Emitted;
            if (decoded < config.minPayloadItems) {
                outcome = FrameOutcome::Undersize;
                ++counters.nUndersizeFrames;
            } else if (config.packBits != 0U && decoded % config.packBits != 0UZ) {
                outcome = FrameOutcome::Unaligned;
                ++counters.nUnalignedFrames;
            }
            publish(outcome);
            step.event = ScanEvent::Record;
        }

        if (config.frameOpen == FrameOpen::Delimiter) {
            openAt(at + 1UZ); // the delimiter that closed this frame opens the next one
            return;
        }
        discardFrame();
        _state = State::Idle;
    }

    /// @brief Hands the frame buffer over as the closed record, running the pack stage where the decoded count admits it.
    void publish(FrameOutcome outcome) {
        _record.swap(_frame); // both were reserved to the bound by prepare(), so neither ever reallocates
        _frame.clear();
        _outcome      = outcome;
        _frameStart   = _sampleStart;
        _frameRemoved = _removed;
        _frameDecoded = _record.size();
        _framePacked  = false;
        _holding      = true;

        if constexpr (std::same_as<T, std::uint8_t>) {
            // an unaligned frame carries its unpacked bits instead, so that nothing is lost to a partial item
            if (config.packBits != 0U && _frameDecoded % config.packBits == 0UZ) {
                _packed.resize(_frameDecoded / config.packBits);
                repack(config.pack, std::span<const std::uint8_t>(_record), std::span<std::uint8_t>(_packed));
                _framePacked = true;
            }
        }
    }
};

/**
 * @brief ISO/IEC 13239 framing: the `01111110` flag, a zero removed after five ones, seven ones an abort.
 *
 * Serves AX.25 version 2.2 and synchronous PPP unchanged, both of which use this framing and differ only in the
 * payload. This is an interoperability constant and not a default: a caller states the bound and, where it wants
 * octets rather than bits, the pack width and order.
 */
[[nodiscard]] inline DelimiterConfig hdlc() {
    DelimiterConfig cfg;
    cfg.endDelimiter   = "01111110";
    cfg.frameOpen      = FrameOpen::Delimiter;
    cfg.bitsPerItem    = 1U;
    cfg.transparency   = Transparency::BitStuffing;
    cfg.stuffAfterOnes = 5U;
    cfg.abortOnes      = 7U;
    configureDelimiter(cfg);
    return cfg;
}

/**
 * @brief RFC 1055 framing: `0xC0` between frames, `0xDB` the escape introducer, `0xDC` and `0xDD` its two escapes.
 *
 * Serves KISS byte for byte: FEND, FESC, TFEND and TFESC are these four values under different names, and the KISS
 * command byte is the first payload item rather than part of the framing.
 */
[[nodiscard]] inline DelimiterConfig slip() {
    DelimiterConfig cfg;
    cfg.endDelimiter = "11000000"; // 0xC0
    cfg.frameOpen    = FrameOpen::Delimiter;
    cfg.bitsPerItem  = 8U;
    cfg.transparency = Transparency::ByteEscape;
    cfg.escapeItem   = 0xDBU;
    cfg.escapeMap    = {{0xDCU, 0xC0U}, {0xDDU, 0xDBU}};
    configureDelimiter(cfg);
    return cfg;
}

/// @brief RFC 1662 section 4 framing: `0x7E` between frames, `0x7D` the introducer, and the default control character map.
[[nodiscard]] inline DelimiterConfig pppAsync() {
    std::vector<std::uint8_t> originals;
    originals.reserve(34UZ);
    for (unsigned value = 0U; value < 32U; ++value) {
        originals.push_back(static_cast<std::uint8_t>(value));
    }
    originals.push_back(0x7DU);
    originals.push_back(0x7EU);

    DelimiterConfig cfg;
    cfg.endDelimiter = "01111110"; // 0x7E
    cfg.frameOpen    = FrameOpen::Delimiter;
    cfg.bitsPerItem  = 8U;
    cfg.transparency = Transparency::ByteEscape;
    cfg.escapeItem   = 0x7DU;
    cfg.escapeMap    = escapeMapFromXor(0x20U, std::span<const std::uint8_t>(originals));
    configureDelimiter(cfg);
    return cfg;
}

/**
 * @brief NMEA 0183 sentences: `$` opens, `CR LF` closes, and nothing is escaped.
 *
 * The example of `Transparency::None`, which is correct here because the payload alphabet is printable ASCII and is
 * wrong wherever it is not: an eight-bit delimiter over unconstrained binary data is forged once every 256 bit
 * positions, which at 9600 bit/s is 37.5 times a second.
 */
[[nodiscard]] inline DelimiterConfig nmea0183() {
    DelimiterConfig cfg;
    cfg.endDelimiter   = "0000110100001010"; // CR LF
    cfg.startDelimiter = "00100100";         // $
    cfg.frameOpen      = FrameOpen::Start;
    cfg.bitsPerItem    = 8U;
    cfg.transparency   = Transparency::None;
    configureDelimiter(cfg);
    return cfg;
}

} // namespace gr::digital

#endif // GNURADIO_ALGORITHM_DELIMITER_HPP
