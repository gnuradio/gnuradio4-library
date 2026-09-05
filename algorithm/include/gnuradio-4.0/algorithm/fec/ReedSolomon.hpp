#ifndef GNURADIO_ALGORITHM_FEC_REED_SOLOMON_HPP
#define GNURADIO_ALGORITHM_FEC_REED_SOLOMON_HPP

#include <array>
#include <cstddef>
#include <cstdint>

/**
 * Reed-Solomon over GF(2^M), shortened, with the generator's roots at alpha^1 upward.
 *
 * A code is fixed by its field — the symbol width M and the primitive polynomial — and by how
 * many parity symbols R it carries; the natural block length is 2^M - 1 symbols and the code
 * corrects R/2 symbol errors. The driving instantiations are P25 Phase 1's three codes over
 * GF(64) on x^6 + x + 1 (TIA-102.BAAA), reached through the `ReedSolomon6` alias:
 *
 *   RS(24,12,13)  R = 12, corrects 6   Link Control, in a voice frame carrying it
 *   RS(24,16,9)   R =  8, corrects 4   Encryption Sync, in a voice frame carrying it
 *   RS(36,20,17)  R = 16, corrects 8   the header that opens a transmission
 *
 * All three are the 63-symbol code SHORTENED: the transmitter and receiver both know that
 * the leading symbols are zero and neither sends them. A shortened code is decoded by
 * putting those zeros back, which is why every entry point here works on a full block and
 * takes the count of leading symbols the air did not carry.
 *
 * Shortening is also a free check, and it is taken. A decoder that places an error inside
 * the padding is saying the received word is not a word of the shortened code at all — the
 * implementations this behavior was compared against skip such a correction and return the
 * remaining symbols as though nothing had happened, which is the silent-wrong-answer case in
 * its purest form. Here the correction is applied and the padding is then checked, so the
 * word is reported invalid instead. The same reasoning covers the final syndrome check: after
 * correction the syndromes must all vanish, and if they do not, the decoder has found a
 * locator polynomial whose roots do not describe the received word and has no business
 * returning symbols.
 *
 * The decoder is Berlekamp-Massey for the error locator, a Chien search for its roots and
 * Forney's formula for the magnitudes, with the field's logarithm and antilogarithm tables
 * carrying every multiplication. Erasures are not supported: no consumer knows which symbols
 * are doubtful in the sense the erasure path needs, and the code that is not written cannot
 * be the code that is wrong. The generator's first root is alpha^1, the convention every
 * validated instantiation shares; a code family rooted elsewhere brings its own oracle when
 * it brings its first consumer.
 */
namespace gr::fec {

struct RsResult {
    unsigned errors{0U};           ///< symbols corrected
    bool     valid{false};         ///< the corrected block is a word of the shortened code
    bool     pad_corrupted{false}; ///< a correction fell in the padding the air never carried
};

namespace detail {

template<std::size_t SymbolBits, unsigned FieldPoly>
struct GfTables {
    static constexpr std::size_t kElements = 1UZ << SymbolBits;
    static constexpr std::size_t kNonzero  = kElements - 1UZ;

    std::array<std::uint8_t, kElements> exp{}; ///< exp[i] = alpha^i, with exp[kNonzero] = 0 as the log-of-zero sentinel's image
    std::array<std::uint8_t, kElements> log{}; ///< log[x] = i where alpha^i = x, kNonzero for 0
};

template<std::size_t SymbolBits, unsigned FieldPoly>
[[nodiscard]] constexpr GfTables<SymbolBits, FieldPoly> buildGfTables() noexcept {
    GfTables<SymbolBits, FieldPoly> t{};
    constexpr std::size_t           nonzero = GfTables<SymbolBits, FieldPoly>::kNonzero;
    t.log[0]                                = static_cast<std::uint8_t>(nonzero);
    t.exp[nonzero]                          = 0U;

    unsigned sr = 1U;
    for (std::size_t i = 0UZ; i < nonzero; ++i) {
        t.exp[i]  = static_cast<std::uint8_t>(sr);
        t.log[sr] = static_cast<std::uint8_t>(i);
        sr <<= 1U;
        if (sr & (1U << SymbolBits)) {
            sr ^= FieldPoly;
        }
        sr &= static_cast<unsigned>(nonzero);
    }
    return t;
}

template<std::size_t SymbolBits, unsigned FieldPoly>
inline constexpr GfTables<SymbolBits, FieldPoly> kGf = buildGfTables<SymbolBits, FieldPoly>();

} // namespace detail

/**
 * @brief The 2^M - 1 symbol Reed-Solomon code over GF(2^M) with `Roots` parity symbols.
 *
 * Blocks are laid out with the information symbols first and the parity last, and a
 * shortened code is used by holding the first `pad` symbols at zero. Symbols are one byte
 * each, which serves every field through GF(256).
 */
template<std::size_t SymbolBits, unsigned FieldPoly, std::size_t Roots>
struct ReedSolomon {
    static_assert(SymbolBits >= 2UZ && SymbolBits <= 8UZ, "a symbol is carried in one byte");
    static_assert((FieldPoly >> SymbolBits) == 1U, "the field polynomial carries the x^M term and nothing above it");

    static constexpr std::size_t  kBlock       = (1UZ << SymbolBits) - 1UZ;
    static constexpr std::size_t  kRoots       = Roots;
    static constexpr unsigned     kCorrectable = static_cast<unsigned>(Roots / 2UZ);
    static constexpr std::uint8_t kLogZero     = static_cast<std::uint8_t>(kBlock); ///< the logarithm zero does not have

    using Block = std::array<std::uint8_t, kBlock>;

    /// Reduce an exponent into 0..kBlock-1. Exponents arrive as sums, differences and small
    /// multiples of table indices, so the reduction is a plain remainder rather than a subtraction.
    [[nodiscard]] static constexpr unsigned modExp(unsigned x) noexcept { return x % static_cast<unsigned>(kBlock); }

    /// The generator polynomial in logarithm form, constant term first; roots alpha^1 .. alpha^Roots.
    [[nodiscard]] static constexpr std::array<std::uint8_t, Roots + 1UZ> generator() noexcept {
        constexpr auto& gf = detail::kGf<SymbolBits, FieldPoly>;

        std::array<std::uint8_t, Roots + 1UZ> g{};
        g[0] = 1U;
        for (std::size_t i = 0UZ; i < Roots; ++i) {
            const unsigned root = static_cast<unsigned>(i + 1UZ);
            g[i + 1UZ]          = 1U;
            for (std::size_t j = i; j > 0UZ; --j) {
                if (g[j] != 0U) {
                    g[j] = static_cast<std::uint8_t>(g[j - 1UZ] ^ gf.exp[modExp(gf.log[g[j]] + root)]);
                } else {
                    g[j] = g[j - 1UZ];
                }
            }
            g[0] = gf.exp[modExp(gf.log[g[0]] + root)];
        }
        for (std::size_t i = 0UZ; i <= Roots; ++i) {
            g[i] = gf.log[g[i]];
        }
        return g;
    }

    /// Write the parity symbols over the information symbols already in `block`, leaving a
    /// codeword. The first `pad` symbols are the shortening and must be zero.
    static constexpr void encode(Block& block, std::size_t pad = 0UZ) noexcept {
        constexpr auto& gf = detail::kGf<SymbolBits, FieldPoly>;
        constexpr auto  g  = generator();

        std::array<std::uint8_t, Roots> parity{};
        for (std::size_t i = pad; i < kBlock - Roots; ++i) {
            const std::uint8_t feedbackSymbol = static_cast<std::uint8_t>(block[i] ^ parity[0]);
            const std::uint8_t feedback       = gf.log[feedbackSymbol];
            if (feedback != kLogZero) {
                for (std::size_t j = 1UZ; j < Roots; ++j) {
                    parity[j] ^= gf.exp[modExp(feedback + g[Roots - j])];
                }
            }
            for (std::size_t j = 0UZ; j + 1UZ < Roots; ++j) {
                parity[j] = parity[j + 1UZ];
            }
            parity[Roots - 1UZ] = (feedback != kLogZero) ? gf.exp[modExp(feedback + g[0])] : 0U;
        }
        for (std::size_t j = 0UZ; j < Roots; ++j) {
            block[kBlock - Roots + j] = parity[j];
        }
    }

    /// Correct `block` in place. The first `pad` symbols are the shortening the air did not
    /// carry and must be zero on entry.
    static RsResult decode(Block& block, std::size_t pad = 0UZ) noexcept {
        constexpr auto& gf = detail::kGf<SymbolBits, FieldPoly>;
        RsResult        result;

        // Syndromes: the received polynomial evaluated at alpha^1 .. alpha^Roots.
        std::array<std::uint8_t, Roots> syndrome{};
        std::uint8_t                    any = 0U;
        for (std::size_t i = 0UZ; i < Roots; ++i) {
            std::uint8_t s = block[0];
            for (std::size_t j = 1UZ; j < kBlock; ++j) {
                s = static_cast<std::uint8_t>(block[j] ^ ((s == 0U) ? 0U : gf.exp[modExp(static_cast<unsigned>(gf.log[s]) + static_cast<unsigned>(i + 1UZ))]));
            }
            syndrome[i] = s;
            any         = static_cast<std::uint8_t>(any | s);
        }
        if (any == 0U) {
            result.valid = true;
            return result;
        }

        std::array<std::uint8_t, Roots> syndromeLog{};
        for (std::size_t i = 0UZ; i < Roots; ++i) {
            syndromeLog[i] = gf.log[syndrome[i]];
        }

        // Berlekamp-Massey for the error locator.
        std::array<std::uint8_t, Roots + 1UZ> lambda{};
        std::array<std::uint8_t, Roots + 1UZ> back{};
        std::array<std::uint8_t, Roots + 1UZ> next{};
        lambda[0] = 1U;
        for (std::size_t i = 0UZ; i <= Roots; ++i) {
            back[i] = gf.log[lambda[i]];
        }

        std::size_t length = 0UZ;
        for (std::size_t r = 1UZ; r <= Roots; ++r) {
            std::uint8_t discrepancy = 0U;
            for (std::size_t i = 0UZ; i < r; ++i) {
                if (lambda[i] != 0U && syndromeLog[r - i - 1UZ] != kLogZero) {
                    discrepancy ^= gf.exp[modExp(static_cast<unsigned>(gf.log[lambda[i]]) + syndromeLog[r - i - 1UZ])];
                }
            }
            const std::uint8_t discrepancyLog = gf.log[discrepancy];

            if (discrepancyLog == kLogZero) {
                for (std::size_t i = Roots; i > 0UZ; --i) {
                    back[i] = back[i - 1UZ];
                }
                back[0] = kLogZero;
                continue;
            }

            next[0] = lambda[0];
            for (std::size_t i = 0UZ; i < Roots; ++i) {
                next[i + 1UZ] = (back[i] != kLogZero) ? static_cast<std::uint8_t>(lambda[i + 1UZ] ^ gf.exp[modExp(discrepancyLog + back[i])]) : lambda[i + 1UZ];
            }
            if (2UZ * length <= r - 1UZ) {
                length = r - length;
                for (std::size_t i = 0UZ; i <= Roots; ++i) {
                    back[i] = (lambda[i] == 0U) ? kLogZero : static_cast<std::uint8_t>(modExp(static_cast<unsigned>(gf.log[lambda[i]]) + static_cast<unsigned>(kBlock) - discrepancyLog));
                }
            } else {
                for (std::size_t i = Roots; i > 0UZ; --i) {
                    back[i] = back[i - 1UZ];
                }
                back[0] = kLogZero;
            }
            lambda = next;
        }

        std::array<std::uint8_t, Roots + 1UZ> lambdaLog{};
        std::size_t                           degree = 0UZ;
        for (std::size_t i = 0UZ; i <= Roots; ++i) {
            lambdaLog[i] = gf.log[lambda[i]];
            if (lambdaLog[i] != kLogZero) {
                degree = i;
            }
        }
        if (degree == 0UZ) {
            return result; // syndromes nonzero but no locator: not a correctable word
        }

        // Chien search: alpha^-i is a root of lambda exactly when symbol i is in error.
        std::array<unsigned, Roots>       rootExp{};
        std::array<unsigned, Roots>       location{};
        std::array<unsigned, Roots + 1UZ> reg{};
        for (std::size_t i = 1UZ; i <= Roots; ++i) {
            reg[i] = lambdaLog[i];
        }

        std::size_t found = 0UZ;
        for (std::size_t i = 1UZ; i <= kBlock; ++i) {
            std::uint8_t q = 1U;
            for (std::size_t j = degree; j > 0UZ; --j) {
                if (reg[j] != kLogZero) {
                    reg[j] = modExp(reg[j] + static_cast<unsigned>(j));
                    q ^= gf.exp[reg[j]];
                }
            }
            if (q != 0U) {
                continue;
            }
            rootExp[found]  = static_cast<unsigned>(i);
            location[found] = static_cast<unsigned>(i - 1UZ);
            if (++found == degree) {
                break;
            }
        }
        if (found != degree) {
            return result; // the locator's roots are not all in the field: uncorrectable
        }

        // Forney: omega = syndrome * lambda truncated, and each magnitude is omega over the
        // formal derivative of lambda, both evaluated at the root.
        const std::size_t                     omegaDegree = degree - 1UZ;
        std::array<std::uint8_t, Roots + 1UZ> omegaLog{};
        for (std::size_t i = 0UZ; i <= omegaDegree; ++i) {
            std::uint8_t      acc = 0U;
            const std::size_t top = (degree < i) ? degree : i;
            for (std::size_t j = top + 1UZ; j-- > 0UZ;) {
                if (syndromeLog[i - j] != kLogZero && lambdaLog[j] != kLogZero) {
                    acc ^= gf.exp[modExp(static_cast<unsigned>(syndromeLog[i - j]) + lambdaLog[j])];
                }
            }
            omegaLog[i] = gf.log[acc];
        }

        for (std::size_t k = found; k-- > 0UZ;) {
            std::uint8_t numerator = 0U;
            for (std::size_t i = omegaDegree + 1UZ; i-- > 0UZ;) {
                if (omegaLog[i] != kLogZero) {
                    numerator ^= gf.exp[modExp(omegaLog[i] + static_cast<unsigned>(i) * rootExp[k])];
                }
            }
            // The formal derivative of a polynomial over a field of characteristic two keeps
            // only its odd-numbered coefficients, so the sum steps by two.
            std::uint8_t denominator = 0U;
            const int    cap         = static_cast<int>((degree < Roots - 1UZ) ? degree : Roots - 1UZ);
            for (int i = cap & ~1; i >= 0; i -= 2) {
                if (lambdaLog[static_cast<std::size_t>(i) + 1UZ] != kLogZero) {
                    denominator ^= gf.exp[modExp(lambdaLog[static_cast<std::size_t>(i) + 1UZ] + static_cast<unsigned>(i) * rootExp[k])];
                }
            }
            if (numerator == 0U || denominator == 0U) {
                continue;
            }
            const unsigned magnitude = modExp(static_cast<unsigned>(gf.log[numerator]) + static_cast<unsigned>(kBlock) - gf.log[denominator]);
            block[location[k]] ^= gf.exp[magnitude];
            ++result.errors;
        }

        for (std::size_t i = 0UZ; i < pad; ++i) {
            if (block[i] != 0U) {
                result.pad_corrupted = true;
            }
        }

        // The corrected block must now be a codeword. Where it is not, the locator described
        // some other word and the symbols handed back would be an invention.
        for (std::size_t i = 0UZ; i < Roots; ++i) {
            std::uint8_t s = block[0];
            for (std::size_t j = 1UZ; j < kBlock; ++j) {
                s = static_cast<std::uint8_t>(block[j] ^ ((s == 0U) ? 0U : gf.exp[modExp(static_cast<unsigned>(gf.log[s]) + static_cast<unsigned>(i + 1UZ))]));
            }
            if (s != 0U) {
                return result;
            }
        }

        result.valid = !result.pad_corrupted;
        return result;
    }
};

/// The GF(64) family on x^6 + x + 1, the field P25 Phase 1's three codes share.
template<std::size_t Roots>
using ReedSolomon6 = ReedSolomon<6UZ, 0x43U, Roots>;

} // namespace gr::fec

#endif // GNURADIO_ALGORITHM_FEC_REED_SOLOMON_HPP
