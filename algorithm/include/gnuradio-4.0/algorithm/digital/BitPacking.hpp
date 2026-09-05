#ifndef GNURADIO_ALGORITHM_BIT_PACKING_HPP
#define GNURADIO_ALGORITHM_BIT_PACKING_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>

/**
 * @brief The conversion between a stream of `bitsIn`-bit fields and a stream of `bitsOut`-bit fields.
 *
 * An item is a `std::uint8_t` carrying `w` significant bits in its low `w` positions, `1 <= w <= 8`;
 * positions `w..7` are ignored on input and zero on output. The bit stream of a port is the
 * concatenation, over items in order, of each item's `w` bits taken in that port's bit order:
 * `MsbFirst` emits position `w-1` first and counts down, `LsbFirst` emits position `0` first and
 * counts up. The conversion is then one sentence — the output stream's bit sequence is the input
 * stream's bit sequence. No bit is created, dropped, or moved relative to another bit, and
 * everything below is a consequence of that.
 *
 * The two sides carry independent orders, so an MSB-first byte stream becomes LSB-first symbols in
 * one pass. Bit order is not byte order: it names the traversal of a field inside one item, the
 * order of items is always stream order, and machine endianness never enters because an item is one
 * byte.
 *
 * The conversion is periodic with period `lcm(bitsIn, bitsOut)` bits, which is `bitsOut/g` input
 * items and `bitsIn/g` output items for `g = gcd(bitsIn, bitsOut)`. After a whole period the bit
 * cursor stands at zero on both sides, so no period carries state into the next: `repack()` takes
 * whole periods only, periods are independent, and the result therefore does not depend on how a
 * caller divides the stream into calls. Over `[1,8] x [1,8]` the period is at most `lcm(7,8) = 56`
 * bits and both chunk counts are at most 8, so `configureRepack()` resolves the whole conversion into a
 * fixed 56-entry table of source item, source shift, destination item and destination shift. What
 * is left on the sample path is a shift, a mask, a shift and an OR per bit, with no division, no
 * modulo, no allocation and no branch on data.
 *
 * `bitsIn == 1, bitsOut == 8` and `bitsIn == 8, bitsOut == 1` are the shapes a bit path meets most
 * and take an exact branch-free 64-bit form instead. Eight one-bit items read as a little-endian
 * word, multiplied by `0x8040201008040201`, gather into one byte: the multiplier's set bits sit at
 * `2^(9i)` for `i` in `[0, 8)`, and a spacing of nine keeps the eight shifted copies of the operand
 * from ever sharing a bit position, so the product carries nowhere and one shift with one mask reads
 * the answer out. Unpacking is the same constant read the other way round. Reversing the eight items
 * — one `std::byteswap` — is what turns the MSB-first form into the LSB-first one, so a single
 * constant serves all four cases.
 *
 * `configureRepack()` validates and is the only function here that can throw. It throws
 * `std::invalid_argument`, not a graph exception, because this header depends on the standard
 * library alone. `repack()` and `repackTail()` are `noexcept`, read and write nothing outside their
 * spans, and hold no state, so one configured `BitRepack` is safe to call concurrently.
 *
 * The one place a partial output item can exist is the end of a stream, where fewer than `inChunk`
 * items remain. `repackTail()` covers exactly that: it emits the output items the stranded bits fill
 * and, on request, one more carrying the remainder, with the positions the later bits would have
 * occupied left zero — the low ones under `MsbFirst` output, the high ones under `LsbFirst`.
 *
 * Widths above 8 are not offered. A field wider than a byte needs an item type wider than a byte and
 * brings the order of the bytes inside that item with it, which is machine endianness and is exactly
 * what the bit order here is not.
 */
namespace gr::digital {

/// @brief Which end of a `w`-bit field the stream's first bit occupies.
enum class BitOrder : std::uint8_t { MsbFirst, LsbFirst };

/// @brief The longest period over `[1, 8] x [1, 8]`, `lcm(7, 8)` bits.
inline constexpr std::size_t kMaxRepackPeriod = 56UZ;

/// @brief Where one bit of a period is read and where it is written, both as an item index and a shift within that item.
struct BitMove {
    std::uint8_t sourceItem       = 0U;
    std::uint8_t sourceShift      = 0U;
    std::uint8_t destinationItem  = 0U;
    std::uint8_t destinationShift = 0U;

    [[nodiscard]] bool operator==(const BitMove&) const = default;
};

namespace detail {

/// @brief One period of the definition, resolved into shifts. Bit `b` of the period is item `b/bits` at position `b%bits`, counted from whichever end the order names.
[[nodiscard]] constexpr std::array<BitMove, kMaxRepackPeriod> bitMoves(unsigned bitsIn, unsigned bitsOut, BitOrder orderIn, BitOrder orderOut) noexcept {
    std::array<BitMove, kMaxRepackPeriod> moves{};
    const unsigned                        period = bitsIn * bitsOut / std::gcd(bitsIn, bitsOut);
    for (unsigned b = 0U; b < period; ++b) {
        const unsigned sourcePosition      = b % bitsIn;
        const unsigned destinationPosition = b % bitsOut;
        moves[b]                           = BitMove{static_cast<std::uint8_t>(b / bitsIn),                                                     //
            static_cast<std::uint8_t>(orderIn == BitOrder::MsbFirst ? bitsIn - 1U - sourcePosition : sourcePosition), //
            static_cast<std::uint8_t>(b / bitsOut),                                                                   //
            static_cast<std::uint8_t>(orderOut == BitOrder::MsbFirst ? bitsOut - 1U - destinationPosition : destinationPosition)};
    }
    return moves;
}

} // namespace detail

/**
 * @brief One configured conversion: the four settings, and what `configureRepack()` derives from them.
 *
 * The derived members are written by `configureRepack()` and are never set directly; a default-constructed
 * instance already carries the ones that belong to its default widths and orders.
 */
struct BitRepack {
    std::uint8_t bitsIn   = 1U;
    std::uint8_t bitsOut  = 8U;
    BitOrder     orderIn  = BitOrder::MsbFirst;
    BitOrder     orderOut = BitOrder::MsbFirst;

    std::uint8_t                          inChunk  = 8U; ///< `M = bitsOut / gcd`, input items in one period
    std::uint8_t                          outChunk = 1U; ///< `L = bitsIn / gcd`, output items in one period
    std::uint8_t                          period   = 8U; ///< `lcm(bitsIn, bitsOut)`, at most 56 bits
    std::array<BitMove, kMaxRepackPeriod> moves    = detail::bitMoves(1U, 8U, BitOrder::MsbFirst, BitOrder::MsbFirst);
};

/**
 * @brief Sets @p cfg to the stated conversion, deriving the chunk counts, the period and the shift table.
 *
 * Widths outside `[1, 8]` throw `std::invalid_argument` and leave @p cfg as it was, so a rejected
 * setting cannot half-apply. This is the only function here that can throw and it never runs on the
 * sample path.
 */
inline void configureRepack(BitRepack& cfg, unsigned bitsIn, unsigned bitsOut, BitOrder orderIn, BitOrder orderOut) {
    if (bitsIn < 1U || bitsIn > 8U) {
        throw std::invalid_argument("gr::digital::configureRepack: bitsIn must be in [1, 8], got " + std::to_string(bitsIn));
    }
    if (bitsOut < 1U || bitsOut > 8U) {
        throw std::invalid_argument("gr::digital::configureRepack: bitsOut must be in [1, 8], got " + std::to_string(bitsOut));
    }

    const unsigned common = std::gcd(bitsIn, bitsOut);
    cfg.bitsIn            = static_cast<std::uint8_t>(bitsIn);
    cfg.bitsOut           = static_cast<std::uint8_t>(bitsOut);
    cfg.orderIn           = orderIn;
    cfg.orderOut          = orderOut;
    cfg.inChunk           = static_cast<std::uint8_t>(bitsOut / common);
    cfg.outChunk          = static_cast<std::uint8_t>(bitsIn / common);
    cfg.period            = static_cast<std::uint8_t>(bitsIn * bitsOut / common);
    cfg.moves             = detail::bitMoves(bitsIn, bitsOut, orderIn, orderOut);
}

namespace detail {

static_assert(std::endian::native == std::endian::little || std::endian::native == std::endian::big, "the 64-bit forms read eight items as one word and need a defined byte order");

/// @brief Eight copies of the operand at a spacing of nine bits, which is what keeps them from overlapping in the product.
inline constexpr std::uint64_t kBitSpread = 0x8040201008040201ULL;

/// @brief The low bit of each of eight items held in one word.
inline constexpr std::uint64_t kItemLowBits = 0x0101010101010101ULL;

/// @brief Eight items as one word, item 0 in the low byte, whatever the machine's byte order.
[[nodiscard]] inline std::uint64_t loadItemWord(const std::uint8_t* items) noexcept {
    std::uint64_t word = 0ULL;
    std::memcpy(&word, items, sizeof(word));
    if constexpr (std::endian::native == std::endian::big) {
        word = std::byteswap(word);
    }
    return word;
}

/// @brief The inverse of `loadItemWord`.
inline void storeItemWord(std::uint8_t* items, std::uint64_t word) noexcept {
    if constexpr (std::endian::native == std::endian::big) {
        word = std::byteswap(word);
    }
    std::memcpy(items, &word, sizeof(word));
}

/// @brief Eight one-bit items to one byte. The gathered byte holds item 0 at bit 7 under `MsbFirst`, and reversing the items moves it to bit 0.
inline void packEightBits(std::span<const std::uint8_t> in, std::span<std::uint8_t> out, BitOrder orderOut) noexcept {
    const std::size_t count = std::min(in.size() / 8UZ, out.size());
    if (orderOut == BitOrder::MsbFirst) {
        for (std::size_t i = 0UZ; i < count; ++i) {
            const std::uint64_t word = loadItemWord(in.data() + 8UZ * i) & kItemLowBits;
            out[i]                   = static_cast<std::uint8_t>((word * kBitSpread) >> 56U);
        }
    } else {
        for (std::size_t i = 0UZ; i < count; ++i) {
            const std::uint64_t word = std::byteswap(loadItemWord(in.data() + 8UZ * i) & kItemLowBits);
            out[i]                   = static_cast<std::uint8_t>((word * kBitSpread) >> 56U);
        }
    }
}

/// @brief One byte to eight one-bit items, the same constant read the other way round: output item `i` carries bit `7-i` under `MsbFirst`.
inline void unpackEightBits(std::span<const std::uint8_t> in, std::span<std::uint8_t> out, BitOrder orderIn) noexcept {
    const std::size_t count = std::min(in.size(), out.size() / 8UZ);
    if (orderIn == BitOrder::MsbFirst) {
        for (std::size_t i = 0UZ; i < count; ++i) {
            storeItemWord(out.data() + 8UZ * i, ((static_cast<std::uint64_t>(in[i]) * kBitSpread) >> 7U) & kItemLowBits);
        }
    } else {
        for (std::size_t i = 0UZ; i < count; ++i) {
            storeItemWord(out.data() + 8UZ * i, std::byteswap(((static_cast<std::uint64_t>(in[i]) * kBitSpread) >> 7U) & kItemLowBits));
        }
    }
}

/// @brief The general form: whole periods off the shift table, one period at a time, each independent of every other.
inline void repackGeneral(const BitRepack& cfg, std::span<const std::uint8_t> in, std::span<std::uint8_t> out) noexcept {
    const std::size_t inChunk  = static_cast<std::size_t>(cfg.inChunk);
    const std::size_t outChunk = static_cast<std::size_t>(cfg.outChunk);
    const std::size_t period   = static_cast<std::size_t>(cfg.period);
    const std::size_t periods  = std::min(in.size() / inChunk, out.size() / outChunk);

    for (std::size_t p = 0UZ; p < periods; ++p) {
        const std::uint8_t* source      = in.data() + p * inChunk;
        std::uint8_t*       destination = out.data() + p * outChunk;
        for (std::size_t j = 0UZ; j < outChunk; ++j) {
            destination[j] = 0U;
        }
        for (std::size_t b = 0UZ; b < period; ++b) {
            const BitMove& move               = cfg.moves[b];
            destination[move.destinationItem] = static_cast<std::uint8_t>(destination[move.destinationItem] | (((source[move.sourceItem] >> move.sourceShift) & 1U) << move.destinationShift));
        }
    }
}

} // namespace detail

/**
 * @brief Whole periods of @p in to whole periods of @p out.
 *
 * `in.size()` is `periods * cfg.inChunk` and `out.size()` is `periods * cfg.outChunk`; a caller that
 * supplies more of either converts only the periods both spans hold. Unused high bits of an input
 * item are masked away rather than rejected, so no stream value can fail here. No state, no
 * allocation, no branch on data, and periods are independent of one another.
 */
inline void repack(const BitRepack& cfg, std::span<const std::uint8_t> in, std::span<std::uint8_t> out) noexcept {
    if (cfg.bitsIn == std::uint8_t{1} && cfg.bitsOut == std::uint8_t{8}) {
        detail::packEightBits(in, out, cfg.orderOut);
        return;
    }
    if (cfg.bitsIn == std::uint8_t{8} && cfg.bitsOut == std::uint8_t{1}) {
        detail::unpackEightBits(in, out, cfg.orderIn);
        return;
    }
    detail::repackGeneral(cfg, in, out);
}

/**
 * @brief The trailing input a whole period cannot cover, `in.size() < cfg.inChunk`.
 *
 * For `n` trailing items the stranded bits are `b = n * cfg.bitsIn`. `b / cfg.bitsOut` whole output
 * items are always written. When @p pad is set and `b % cfg.bitsOut` is not zero, one further item
 * carries the remainder, with the positions the later bits would have occupied left zero — the low
 * `cfg.bitsOut - b % cfg.bitsOut` under `MsbFirst` output, the high ones under `LsbFirst`. When
 * @p pad is clear those remaining bits are dropped.
 *
 * @return the items written, at most `out.size()`.
 */
[[nodiscard]] inline std::size_t repackTail(const BitRepack& cfg, std::span<const std::uint8_t> in, std::span<std::uint8_t> out, bool pad) noexcept {
    const std::size_t bitsOut   = static_cast<std::size_t>(cfg.bitsOut);
    const std::size_t bits      = in.size() * static_cast<std::size_t>(cfg.bitsIn);
    const std::size_t remainder = bits % bitsOut;
    const std::size_t written   = std::min(bits / bitsOut + ((pad && remainder != 0UZ) ? 1UZ : 0UZ), out.size());

    const std::uint8_t* source      = in.data();
    std::uint8_t*       destination = out.data();
    for (std::size_t j = 0UZ; j < written; ++j) {
        destination[j] = 0U;
    }

    // A trailing input is shorter than one period, so every bit of it indexes the period's own table.
    // The clamp to the period is what makes that a bound this function enforces rather than one it states:
    // `moves` holds one period and nothing above it.
    const std::size_t usable = std::min({bits, written * bitsOut, static_cast<std::size_t>(cfg.period)});
    for (std::size_t b = 0UZ; b < usable; ++b) {
        const BitMove& move               = cfg.moves[b];
        destination[move.destinationItem] = static_cast<std::uint8_t>(destination[move.destinationItem] | (((source[move.sourceItem] >> move.sourceShift) & 1U) << move.destinationShift));
    }
    return written;
}

/**
 * @brief The output item holding the first bit of input item @p offset, `floor(offset * bitsIn / bitsOut)`, in arithmetic that cannot overflow.
 *
 * A tag marks an item, that item's first bit is bit `offset * bitsIn` of the stream, and this is the
 * output item that bit lands in. Flooring is the rule: rounding books a tag onto the item after the
 * one holding its bit whenever that bit falls in the second half of an item.
 *
 * Offsets stay in integer arithmetic end to end. A `double` is exact on integers only to `2^53`,
 * where `floor(t/3)` first disagrees at `t = 9007199254740993`. The direct product `offset * bitsIn`
 * overflows a 64-bit offset above `2^64/8`, so `offset` is split into `q * bitsOut + r` and only
 * `r < bitsOut <= 8` is multiplied; the remaining `q * bitsIn` overflows only once the output stream
 * has itself passed `2^64` items, which is the same event as exhausting the offset space.
 *
 * Packing maps several input offsets onto one output offset. Each tag belongs there, and they keep
 * their input order.
 */
[[nodiscard]] inline constexpr std::uint64_t mapRepackedOffset(std::uint64_t offset, std::uint64_t bitsIn, std::uint64_t bitsOut) noexcept {
    const std::uint64_t whole     = offset / bitsOut;
    const std::uint64_t remainder = offset % bitsOut;
    return whole * bitsIn + (remainder * bitsIn) / bitsOut;
}

} // namespace gr::digital

#endif // GNURADIO_ALGORITHM_BIT_PACKING_HPP
