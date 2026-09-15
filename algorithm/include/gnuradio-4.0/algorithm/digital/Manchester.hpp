#ifndef GNURADIO_ALGORITHM_MANCHESTER_HPP
#define GNURADIO_ALGORITHM_MANCHESTER_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

#include <gnuradio-4.0/algorithm/digital/BitPacking.hpp>

/**
 * @brief The Manchester line code, both live conventions, from one XOR constant.
 *
 * One data bit becomes two chips of opposite value. Writing the convention as a constant `k` in
 * GF(2), the whole code is one line:
 *
 *     chipPair(b) = (b XOR k, b XOR k XOR 1)
 *
 * so there is a transition at the center of every bit, the running disparity returns to zero at every
 * bit boundary, no more than two like chips ever occur in a row, and the chip rate is twice the bit
 * rate. None of that depends on the data, on a seed or on any state.
 *
 * `k` is the whole of the difference between the two conventions in use. `Ieee8023` is `k = 1`: IEEE
 * Std 802.3's 10 Mbit/s PLS carries the complement of the data bit in the first half of the bit cell
 * and the true bit in the second, so a mid-bit low-to-high transition is a one and a zero encodes to
 * the chips `10`. `GeThomas` is `k = 0`: MIL-STD-1553 states the chip pair itself, a logic one as a
 * positive pulse followed by a negative one and a logic zero as its inverse, so a one encodes to `10`.
 * The two are exact chipwise complements — `chipPair` under one convention is the bitwise complement
 * of the other's — and that is the trap the required setting exists for: **a stream read under the
 * wrong convention decodes to the exact bitwise complement of the data with no violation raised
 * anywhere**, so nothing short of a downstream CRC can tell. There is no defensible default.
 *
 * The decoder is the same line read backwards, and the rule is uniform:
 *
 *     bit       = c0 XOR k
 *     violation = NOT (c0 XOR c1)
 *
 * The decoded bit comes from the first chip alone; the second chip only sets the flag. On a valid
 * pair that is the same answer as any other reading, since `c1 = NOT c0` there. On a chip pair of
 * `00` or `11` — which the encoder cannot produce, and which MIL-STD-1553's syncs and IEEE 802.5's
 * `J`/`K` delimiters are on purpose — it is a stated convention, it removes the only branch the
 * decoder would otherwise have, and it makes `(bit, violation)` a lossless recoding of the chip pair:
 * the chips were `(bit XOR k, bit XOR k XOR 1)` when the flag is clear and `(bit XOR k, bit XOR k)`
 * when it is set. Nothing is dropped, corrected, or resynchronized on, because a decoder that
 * realigned itself on a violation would fail hardest on the protocols that use violations as data.
 *
 * A 2:1 decoder also has to decide which chip starts a pair. `gridParity` is that decision and one
 * bit is the whole of it: `0` pairs chips `(0,1), (2,3), …` and holds nothing between calls, `1`
 * pairs `(1,2), (3,4), …` and holds exactly one chip. `realignManchester()` moves it, and each direction costs
 * exactly one chip: moving to `1` emits the chip at the boundary as a flagged orphan, moving to `0`
 * drops the held chip. That asymmetry is forced rather than chosen — it is the only assignment that
 * keeps the decoder exactly one output item per two input items in both directions. A decoder one
 * chip out of phase is loud in proportion to the data's transition density and completely silent on a
 * constant stream, where it produces the complement of the data: the wrong-convention symptom
 * exactly, which is the second reason the convention is not guessed.
 *
 * An item is `BitPacking.hpp`'s item: a `std::uint8_t` carrying `bitsPerItem` significant bits in its
 * low positions, whose stream order inside the item is `MsbFirst` or `LsbFirst`. Bits above
 * `bitsPerItem` are masked away on input and left zero on output, so no stream value can fail.
 * `BitOrder` is that header's enumeration rather than a second one with the same two members. One
 * input item of `bitsPerItem` bits becomes exactly two output items of `bitsPerItem` chips for every
 * width, so the ratio does not move with the width; at odd widths a single bit's two chips fall on
 * either side of the boundary between those two items, which is the same statement about the chip
 * stream and not a special case.
 *
 * `configureManchester()` validates, derives `k`, builds the encoder's chip table and calls `resetManchester()`. It is
 * the only function here that can throw. It throws `std::invalid_argument`, not a graph exception,
 * because this header depends on the standard library alone. `encodeManchester()` is `const` on the
 * configuration and carries no state whatever; `decodeManchester()` is the only stateful entry and its state is
 * one chip; both are `noexcept` and allocate nothing.
 *
 * Two shapes carry the sample path. Encoding is a pure function of the input item, so `configureManchester()`
 * resolves it into a 512-byte table of the two output items, and the sample path is one indexed load
 * and two stores with the field mask already folded in. Decoding is bit-parallel and has no
 * loop-carried dependency: gathering the even chip positions of an item pair into `E` and the odd into
 * `O` gives the decoded field as `E XOR k` and the flags as `NOT (E XOR O)` in three operations for a
 * whole item, so the fast form runs whenever the grid is aligned and no chip is held.
 * `detail::encodeReference()` and `detail::decodeReference()` are the bit-at-a-time definitions both
 * are measured against; they agree bit for bit and leave identical state, so either can be deleted if
 * a benchmark says it is not worth its lines.
 */
namespace gr::digital {

/// @brief Which chip pair a data bit becomes, which is the whole of the difference between the two.
enum class ManchesterConvention : std::uint8_t { Ieee8023, GeThomas };

/// @brief The name a block exposes for @p convention.
[[nodiscard]] constexpr std::string_view conventionName(ManchesterConvention convention) noexcept { return convention == ManchesterConvention::GeThomas ? std::string_view{"ge_thomas"} : std::string_view{"ieee802_3"}; }

/// @brief The convention @p name selects; throws `std::invalid_argument`, quoting @p name, for anything but the two names.
[[nodiscard]] constexpr ManchesterConvention conventionFromName(std::string_view name) {
    if (name == conventionName(ManchesterConvention::Ieee8023)) {
        return ManchesterConvention::Ieee8023;
    }
    if (name == conventionName(ManchesterConvention::GeThomas)) {
        return ManchesterConvention::GeThomas;
    }
    throw std::invalid_argument("gr::digital::conventionFromName: convention must be 'ieee802_3' or 'ge_thomas', got '" + std::string(name) + "'");
}

/// @brief The `k` of `chipPair(b) = (b XOR k, b XOR k XOR 1)`, and the only thing the two conventions disagree about.
[[nodiscard]] constexpr std::uint8_t conventionXor(ManchesterConvention convention) noexcept { return static_cast<std::uint8_t>(convention == ManchesterConvention::Ieee8023 ? 1 : 0); }

/**
 * @brief The conventions of the standards that state one, as interoperability constants.
 *
 * Which convention a protocol uses is an agreement between two ends rather than a free choice, and no
 * formula produces it, so each is named here for a caller who wants the identifier instead of the
 * spelling. None of them is a default: a receiver on the wrong one is silently handed the complement
 * of the data. Each entry records what its standard says and the check value that statement produces.
 */
namespace standard {

/// @brief IEEE Std 802.3, the Manchester encoding of the 10 Mbit/s PLS, and the AUI. The bit cell carries the complement
/// of the data bit in its first half and the true bit in its second, so a mid-bit low-to-high transition is a one. The
/// check value is the standard's own preamble and start-of-frame delimiter: the preamble octet `0x55` and the delimiter
/// `0xD5`, taken least-significant-bit first, encode to the item pairs `66 66` and `66 A6`. Their chip sequences carry the
/// two properties the standard's waveform description states — the preamble is a square wave with exactly one transition
/// per bit time, and the delimiter breaks it at exactly one place, the boundary between its final two ones.
inline constexpr ManchesterConvention ethernet = ManchesterConvention::Ieee8023;

/// @brief MIL-STD-1553, "Manchester II bi-phase level". The standard states the mapping at bit level: a logic one is a
/// bipolar coded signal `1/0`, a positive pulse followed by a negative pulse, and a logic zero is `0/1`. That is `k = 0`,
/// and it is the strongest anchor available because it publishes the chip pair itself rather than a waveform. The same
/// standard's command and data syncs are, in its own words, invalid Manchester waveforms, which is why a violation is
/// reported and never absorbed.
inline constexpr ManchesterConvention mil1553 = ManchesterConvention::GeThomas;

/// @brief IEC 62386, digital addressable lighting. A logical one is low during the first half-bit and high during the
/// second, so a rising mid-bit edge is a one, which is the same reading as `ethernet`.
inline constexpr ManchesterConvention dali = ManchesterConvention::Ieee8023;

} // namespace standard

/**
 * @brief One configured line coder: the four settings, what `configureManchester()` derives from them, and the decoder's one chip.
 *
 * The derived members are written by `configureManchester()` and are never set directly. The two counters are
 * the opposite: they are cumulative, are left alone by both `configureManchester()` and `resetManchester()`, and belong
 * to whoever owns the configuration, so that a settings change in the middle of a run does not erase
 * the totals a caller is about to report.
 */
struct ManchesterConfig {
    ManchesterConvention convention  = ManchesterConvention::Ieee8023;
    std::uint8_t         bitsPerItem = 8U;
    BitOrder             order       = BitOrder::MsbFirst;
    std::uint8_t         chipPhase   = 0U; ///< the pairing grid `resetManchester()` restores, 0 or 1

    std::uint8_t                     k     = 1U; ///< the XOR constant of the convention
    std::array<std::uint16_t, 256UZ> table = {}; ///< each item's two output items, the first in the high byte

    std::uint8_t gridParity = 0U;    ///< a pair begins at every chip whose absolute chip index is congruent to this modulo 2
    std::uint8_t carry      = 0U;    ///< the held chip, meaningful only while `hasCarry`
    bool         hasCarry   = false; ///< whether a chip is held, which is the grid parity realized at a call boundary

    std::uint64_t nOrphanItems  = 0ULL; ///< items fabricated by a grid change to parity 1
    std::uint64_t nDroppedChips = 0ULL; ///< chips dropped by a grid change to parity 0
};

/// @brief Puts the pairing grid back to the configured chip phase and clears the held chip. Leaves the counters alone. Allocates nothing.
inline void resetManchester(ManchesterConfig& cfg) noexcept {
    cfg.gridParity = cfg.chipPhase;
    cfg.carry      = 0U;
    cfg.hasCarry   = false;
}

namespace detail {

/// @brief The bit position of the @p index-th bit of a @p width-bit item under @p order, which is the whole of what an order means.
[[nodiscard]] constexpr unsigned streamPosition(unsigned index, unsigned width, BitOrder order) noexcept { return order == BitOrder::MsbFirst ? width - 1U - index : index; }

/// @brief The two output items @p item encodes to, the first in the high byte, packed in @p order and masked to @p width.
[[nodiscard]] constexpr std::uint16_t chipPairItems(unsigned item, unsigned width, unsigned k, BitOrder order) noexcept {
    unsigned first  = 0U;
    unsigned second = 0U;
    for (unsigned i = 0U; i < width; ++i) {
        const unsigned bit  = (item >> streamPosition(i, width, order)) & 1U;
        const unsigned high = bit ^ k;

        // chips 2i and 2i+1 of the item's 2*width chips; the first width of them make the first output
        // item, so at an odd width a bit's two chips fall on either side of that boundary
        for (unsigned half = 0U; half < 2U; ++half) {
            const unsigned index = 2U * i + half;
            if (index < width) {
                first |= (high ^ half) << streamPosition(index, width, order);
            } else {
                second |= (high ^ half) << streamPosition(index - width, width, order);
            }
        }
    }
    return static_cast<std::uint16_t>((first << 8U) | second);
}

} // namespace detail

/**
 * @brief Sets @p cfg to the stated code, deriving the XOR constant and the encoder's chip table.
 *
 * A @p bitsPerItem outside `[1, 8]` and a @p chipPhase above `1` throw `std::invalid_argument` and
 * leave @p cfg as it was, so a rejected setting cannot half-apply. Both conventions are always
 * acceptable here; refusing an unstated one belongs to whatever declares the setting, since this
 * function takes an enumerator and not a name.
 *
 * The table covers all 256 item values at every width, so the field mask is folded into it and the
 * sample path carries none. It is 512 bytes, fixed by the item type rather than by a threshold, and
 * is built once here and never resized.
 */
inline void configureManchester(ManchesterConfig& cfg, ManchesterConvention convention, unsigned bitsPerItem, BitOrder order, unsigned chipPhase = 0U) {
    if (bitsPerItem < 1U || bitsPerItem > 8U) {
        throw std::invalid_argument("gr::digital::configureManchester: bitsPerItem must be in [1, 8], got " + std::to_string(bitsPerItem));
    }
    if (chipPhase > 1U) {
        throw std::invalid_argument("gr::digital::configureManchester: chipPhase must be 0 or 1, got " + std::to_string(chipPhase));
    }

    const unsigned k = conventionXor(convention);
    for (std::size_t value = 0UZ; value < cfg.table.size(); ++value) {
        cfg.table[value] = detail::chipPairItems(static_cast<unsigned>(value), bitsPerItem, k, order);
    }

    cfg.convention  = convention;
    cfg.bitsPerItem = static_cast<std::uint8_t>(bitsPerItem);
    cfg.order       = order;
    cfg.chipPhase   = static_cast<std::uint8_t>(chipPhase);
    cfg.k           = static_cast<std::uint8_t>(k);

    resetManchester(cfg);
}

namespace detail {

/// @brief The definition itself, one bit at a time, written independently of the table the sample path reads.
inline void encodeReference(const ManchesterConfig& cfg, std::span<const std::uint8_t> in, std::span<std::uint8_t> out) noexcept {
    const unsigned    width = static_cast<unsigned>(cfg.bitsPerItem);
    const unsigned    k     = static_cast<unsigned>(cfg.k);
    const BitOrder    order = cfg.order;
    const std::size_t count = std::min(in.size(), out.size() / 2UZ);

    for (std::size_t n = 0UZ; n < count; ++n) {
        unsigned first  = 0U;
        unsigned second = 0U;
        for (unsigned i = 0U; i < width; ++i) {
            const unsigned bit  = (static_cast<unsigned>(in[n]) >> streamPosition(i, width, order)) & 1U;
            const unsigned high = bit ^ k;
            for (unsigned half = 0U; half < 2U; ++half) {
                const unsigned chip  = high ^ half;
                const unsigned index = 2U * i + half;
                if (index < width) {
                    first |= chip << streamPosition(index, width, order);
                } else {
                    second |= chip << streamPosition(index - width, width, order);
                }
            }
        }
        out[2UZ * n]       = static_cast<std::uint8_t>(first);
        out[2UZ * n + 1UZ] = static_cast<std::uint8_t>(second);
    }
}

/// @brief Bits 0, 2, 4 … of @p word gathered into positions 0, 1, 2 …, which is one half of a chip de-interleave.
[[nodiscard]] inline unsigned compactEvenBits(std::uint16_t word) noexcept {
    unsigned bits = static_cast<unsigned>(word) & 0x5555U;
    bits          = (bits | (bits >> 1U)) & 0x3333U;
    bits          = (bits | (bits >> 2U)) & 0x0F0FU;
    return (bits | (bits >> 4U)) & 0x00FFU;
}

/// @brief Bits 1, 3, 5 … of @p word gathered into positions 0, 1, 2 …, which is the other half.
[[nodiscard]] inline unsigned compactOddBits(std::uint16_t word) noexcept { return compactEvenBits(static_cast<std::uint16_t>(word >> 1U)); }

/**
 * @brief The bit-parallel decoder, for an aligned grid with no chip held: three operations per output item.
 *
 * Two input items are `2 * width` chips and one output item. Laying them out so that the chip index
 * runs the same way as the bit index puts every first chip of a pair on one parity of bit position and
 * every second chip on the other, so one de-interleave gives the fields `E` and `O` and the decode
 * rule is `E XOR k` and `NOT (E XOR O)` over the whole item at once. There is no loop-carried
 * dependency, which is the point of the form.
 */
inline void decodeAligned(const ManchesterConfig& cfg, std::span<const std::uint8_t> in, std::span<std::uint8_t> out, std::span<std::uint8_t> violation, std::size_t count) noexcept {
    const unsigned width     = static_cast<unsigned>(cfg.bitsPerItem);
    const unsigned mask      = (1U << width) - 1U;
    const unsigned broadcast = cfg.k != 0U ? mask : 0U;
    const bool     msbFirst  = cfg.order == BitOrder::MsbFirst;

    for (std::size_t i = 0UZ; i < count; ++i) {
        const unsigned early = static_cast<unsigned>(in[2UZ * i]) & mask;
        const unsigned late  = static_cast<unsigned>(in[2UZ * i + 1UZ]) & mask;

        // chip j at bit 2*width-1-j under MsbFirst and at bit j under LsbFirst, so the first chip of a
        // pair takes odd bit positions in the first layout and even ones in the second
        const std::uint16_t chips = static_cast<std::uint16_t>(msbFirst ? (early << width) | late : early | (late << width));
        const unsigned      heads = msbFirst ? compactOddBits(chips) : compactEvenBits(chips);
        const unsigned      tails = msbFirst ? compactEvenBits(chips) : compactOddBits(chips);

        out[i]       = static_cast<std::uint8_t>(heads ^ broadcast);
        violation[i] = static_cast<std::uint8_t>(~(heads ^ tails) & mask);
    }
}

/**
 * @brief The definition itself, one chip at a time through the held chip, for every grid and width.
 *
 * A grid parity of `1` with nothing held is the one chip a move to that grid owes: the first chip of
 * the call has no partner, so it leaves as a flagged orphan carrying `chip XOR k` and pairing resumes
 * behind it. Every other case is the plain pairing loop. All three leave exactly one output item per
 * two input items, which is what keeps the ratio constant across a grid change.
 */
inline void decodeReference(ManchesterConfig& cfg, std::span<const std::uint8_t> in, std::span<std::uint8_t> out, std::span<std::uint8_t> violation) noexcept {
    const unsigned    width = static_cast<unsigned>(cfg.bitsPerItem);
    const unsigned    k     = static_cast<unsigned>(cfg.k);
    const BitOrder    order = cfg.order;
    const std::size_t count = std::min({in.size() / 2UZ, out.size(), violation.size()});

    std::uint8_t carry    = cfg.carry;
    bool         hasCarry = cfg.hasCarry;
    bool         orphan   = cfg.gridParity == 1U && !hasCarry;

    std::size_t index = 0UZ;
    unsigned    slot  = 0U;
    unsigned    bits  = 0U;
    unsigned    flags = 0U;

    for (std::size_t n = 0UZ; n < 2UZ * count; ++n) {
        for (unsigned i = 0U; i < width; ++i) {
            const unsigned chip = (static_cast<unsigned>(in[n]) >> streamPosition(i, width, order)) & 1U;

            unsigned bit  = 0U;
            unsigned flag = 0U;
            if (orphan) {
                bit    = chip ^ k;
                flag   = 1U;
                orphan = false;
                ++cfg.nOrphanItems;
            } else if (!hasCarry) {
                carry    = static_cast<std::uint8_t>(chip);
                hasCarry = true;
                continue;
            } else {
                bit      = static_cast<unsigned>(carry) ^ k;
                flag     = 1U - (static_cast<unsigned>(carry) ^ chip);
                hasCarry = false;
            }

            bits |= bit << streamPosition(slot, width, order);
            flags |= flag << streamPosition(slot, width, order);
            if (++slot == width) {
                out[index]       = static_cast<std::uint8_t>(bits);
                violation[index] = static_cast<std::uint8_t>(flags);
                ++index;
                bits  = 0U;
                flags = 0U;
                slot  = 0U;
            }
        }
    }

    // a chip that is not held is cleared rather than left standing, so the two decode forms leave
    // identical state and are interchangeable at any item boundary
    cfg.carry    = hasCarry ? carry : std::uint8_t{0};
    cfg.hasCarry = hasCarry;
}

} // namespace detail

/**
 * @brief Encodes @p in into @p out, two output items per input item, advancing nothing.
 *
 * `out.size()` is twice `in.size()`; a caller that supplies more of either converts only the items
 * both spans hold. Each input item carries `cfg.bitsPerItem` significant bits, unused high bits are
 * masked rather than rejected, and the output's unused high bits are zero. No state, no allocation,
 * no branch on data and no throw, so the encoder is a pure function of its input and there is no tail
 * or flush of any kind.
 */
inline void encodeManchester(const ManchesterConfig& cfg, std::span<const std::uint8_t> in, std::span<std::uint8_t> out) noexcept {
    const std::size_t count = std::min(in.size(), out.size() / 2UZ);
    for (std::size_t n = 0UZ; n < count; ++n) {
        const std::uint16_t pair = cfg.table[in[n]];
        out[2UZ * n]             = static_cast<std::uint8_t>(pair >> 8U);
        out[2UZ * n + 1UZ]       = static_cast<std::uint8_t>(pair & 0x00FFU);
    }
}

/**
 * @brief Decodes @p in into @p out and @p violation, one item of each per two input items, advancing the held chip.
 *
 * `in.size()` is twice `out.size()` and twice `violation.size()`; a caller that supplies more of any
 * of them converts only what all three hold. Every decoded item bit is accompanied by a flag bit at
 * the same position in the same order, an invalid chip pair still carries its decoded bit, and
 * nothing is dropped, corrected or realigned on, so `(out, violation)` reconstructs the chip stream
 * exactly. No allocation, no branch on data and no throw.
 */
inline void decodeManchester(ManchesterConfig& cfg, std::span<const std::uint8_t> in, std::span<std::uint8_t> out, std::span<std::uint8_t> violation) noexcept {
    const std::size_t count = std::min({in.size() / 2UZ, out.size(), violation.size()});
    if (!cfg.hasCarry && cfg.gridParity == 0U) {
        detail::decodeAligned(cfg, in, out, violation, count);
        return;
    }
    detail::decodeReference(cfg, in, out, violation);
}

/**
 * @brief Re-anchors the pairing grid so that the chip at absolute chip index @p chipIndex starts a pair.
 *
 * The parity of @p chipIndex is the whole of the request, so a caller converts an item offset by
 * multiplying it by `bitsPerItem` first; at an even width that product is always even and the grid
 * can only ever be asked for parity zero, which is correct rather than broken, since at an even width
 * every item boundary is already a pair boundary.
 *
 * Called at a call boundary the move costs exactly one chip in either direction. Moving to parity one
 * leaves the next `decodeManchester()` owing a flagged orphan, which costs an item of the output and no chips;
 * moving to parity zero drops the held chip, which costs a chip and fabricates nothing. Both are
 * counted. Requesting the parity already in force does nothing at all.
 *
 * @return the chips dropped, `0` or `1`.
 */
[[nodiscard]] inline std::size_t realignManchester(ManchesterConfig& cfg, std::uint64_t chipIndex) noexcept {
    const std::uint8_t wanted = static_cast<std::uint8_t>(chipIndex & 1ULL);
    if (wanted == cfg.gridParity) {
        return 0UZ;
    }

    cfg.gridParity = wanted;
    if (wanted == 0U && cfg.hasCarry) {
        cfg.carry    = 0U;
        cfg.hasCarry = false;
        ++cfg.nDroppedChips;
        return 1UZ;
    }
    return 0UZ;
}

} // namespace gr::digital

#endif // GNURADIO_ALGORITHM_MANCHESTER_HPP
