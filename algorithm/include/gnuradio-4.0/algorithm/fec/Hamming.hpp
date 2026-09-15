#ifndef GNURADIO_ALGORITHM_FEC_HAMMING_HPP
#define GNURADIO_ALGORITHM_FEC_HAMMING_HPP

#include <array>
#include <cstddef>
#include <cstdint>

/**
 * The shortened Hamming codes P25 Phase 1 transmits (TIA-102.BAAA).
 *
 * Both are systematic: the information bits occupy the high end of the codeword and four
 * parity bits the low end, and each information bit contributes a fixed four-bit column to
 * the parity. A code is therefore completely described by its list of columns.
 *
 *   Hamming(15,11,3)  11 information + 4 parity, columns F E D C B A 9 7 6 5 3
 *   Hamming(10,6,3)    6 information + 4 parity, columns E D B 7 3 C
 *
 * listed from the first information bit transmitted. The (15,11) columns are every four-bit
 * pattern of weight two or more, in descending order, which is what makes that code perfect:
 * its fifteen nonzero syndromes name its fifteen bit positions and nothing is left over. The
 * (10,6) code uses six of those eleven columns, so five of the fifteen syndromes — 5, 6, 9,
 * A and F — name no single position at all.
 *
 * That difference is the whole of the practical distinction between them. Hamming(15,11,3)
 * can never report a failure: two errors always look like some other single error and the
 * decoder returns the wrong word with a clear conscience. Hamming(10,6,3) reports a failure
 * on five syndromes out of fifteen, and passing that signal upward is worth doing, because a
 * symbol known to be doubtful is worth more to the code above than one silently guessed.
 */
namespace gr::fec {

struct HammingResult {
    std::uint16_t info{0U};     ///< the recovered information bits, right-aligned
    unsigned      errors{0U};   ///< bits corrected
    bool          valid{false}; ///< the syndrome named a position this code contains
};

namespace detail {

/// The parity columns of Hamming(15,11,3), first information bit transmitted first.
inline constexpr std::array<std::uint8_t, 11UZ> kHamming1511Columns{0xFU, 0xEU, 0xDU, 0xCU, 0xBU, 0xAU, 0x9U, 0x7U, 0x6U, 0x5U, 0x3U};

/// The parity columns of Hamming(10,6,3), first information bit transmitted first.
inline constexpr std::array<std::uint8_t, 6UZ> kHamming1063Columns{0xEU, 0xDU, 0xBU, 0x7U, 0x3U, 0xCU};

/// The parity a run of information bits produces, the first column belonging to the most
/// significant of them.
template<std::size_t N>
[[nodiscard]] constexpr std::uint8_t hammingParity(const std::array<std::uint8_t, N>& columns, std::uint16_t info) noexcept {
    std::uint8_t p = 0U;
    for (std::size_t i = 0UZ; i < N; ++i) {
        if ((info >> (N - 1UZ - i)) & 1U) {
            p ^= columns[i];
        }
    }
    return p;
}

/// Decode one systematic codeword against a column list: `info` information bits above four
/// parity bits.
template<std::size_t N>
[[nodiscard]] constexpr HammingResult hammingDecode(const std::array<std::uint8_t, N>& columns, std::uint16_t info, std::uint8_t parity) noexcept {
    HammingResult r;
    r.info                      = static_cast<std::uint16_t>(info & ((1U << N) - 1U));
    const std::uint8_t syndrome = static_cast<std::uint8_t>(hammingParity(columns, r.info) ^ (parity & 0x0FU));

    if (syndrome == 0U) {
        r.valid = true;
        return r;
    }
    for (std::size_t i = 0UZ; i < N; ++i) {
        if (columns[i] == syndrome) {
            r.info   = static_cast<std::uint16_t>(r.info ^ (1U << (N - 1UZ - i)));
            r.errors = 1U;
            r.valid  = true;
            return r;
        }
    }
    // A syndrome of weight one names one of the four parity bits, which carry nothing.
    if ((syndrome & (syndrome - 1U)) == 0U) {
        r.errors = 1U;
        r.valid  = true;
        return r;
    }
    r.errors = 2U;
    return r;
}

} // namespace detail

/// Encode 11 information bits, producing a 15-bit codeword with the information in bits 14..4.
[[nodiscard]] inline constexpr std::uint16_t hamming1511Encode(std::uint16_t info) noexcept {
    const std::uint16_t data = static_cast<std::uint16_t>(info & 0x07FFU);
    return static_cast<std::uint16_t>((data << 4) | detail::hammingParity(detail::kHamming1511Columns, data));
}

/// Encode 6 information bits, producing a 10-bit codeword with the information in bits 9..4.
[[nodiscard]] inline constexpr std::uint16_t hamming1063Encode(std::uint8_t info) noexcept {
    const std::uint16_t data = static_cast<std::uint16_t>(info & 0x3FU);
    return static_cast<std::uint16_t>((data << 4) | detail::hammingParity(detail::kHamming1063Columns, data));
}

/// Decode a 15-bit codeword, its first transmitted bit in bit 14.
///
/// The code is perfect, so `valid` is always true: every one of the fifteen nonzero syndromes
/// names a bit position, and a word carrying two errors decodes to a neighboring codeword
/// rather than reporting anything.
[[nodiscard]] inline constexpr HammingResult hamming1511Decode(std::uint16_t cw) noexcept { return detail::hammingDecode(detail::kHamming1511Columns, static_cast<std::uint16_t>((cw >> 4) & 0x07FFU), static_cast<std::uint8_t>(cw & 0x0FU)); }

/// Decode a 10-bit codeword, its first transmitted bit in bit 9.
///
/// Five of the fifteen nonzero syndromes name no position in this shortened code and are
/// reported as a failure rather than corrected.
[[nodiscard]] inline constexpr HammingResult hamming1063Decode(std::uint16_t cw) noexcept { return detail::hammingDecode(detail::kHamming1063Columns, static_cast<std::uint16_t>((cw >> 4) & 0x3FU), static_cast<std::uint8_t>(cw & 0x0FU)); }

} // namespace gr::fec

#endif // GNURADIO_ALGORITHM_FEC_HAMMING_HPP
