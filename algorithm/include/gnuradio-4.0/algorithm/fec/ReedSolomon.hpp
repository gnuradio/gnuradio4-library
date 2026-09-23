#ifndef GNURADIO_ALGORITHM_FEC_REED_SOLOMON_HPP
#define GNURADIO_ALGORITHM_FEC_REED_SOLOMON_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

/**
 * Reed-Solomon over GF(2^M), shortened, with the generator's roots at a stated power of a stated
 * primitive element upward.
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
 * carrying every multiplication. `decodeWithErasures` also takes the positions of erased
 * symbols, such as the least reliable symbols a soft-decision demodulator marks. The product of
 * `(1 + X x)` over the erased symbols, `X = beta^d` for the symbol at power `d`, seeds
 * Berlekamp-Massey, which extends it by the error locator, and a word is corrected whenever
 * twice its errors plus its erasures is at most R. `decode` is the same decoder with no erasures.
 *
 * **Where the generator's roots start, and how far apart they are.** A Reed-Solomon code is fixed
 * by its field and by the `Roots` consecutive powers its generator vanishes at, and two families
 * disagree about which powers those are. The roots are `beta^f .. beta^(f+Roots-1)` with
 * `beta = alpha^PrimitiveStep` and `f = FirstConsecutiveRoot`; both default to 1, which is the plain
 * convention and leaves every existing instantiation bit-identical. CCSDS 131.0-B-5 4.3.4
 * writes its generator as the product over `j = 128-E .. 127+E` of `(x - alpha^(11j))`, so its two
 * numbers are read straight off the product limits: `PrimitiveStep = 11` and
 * `FirstConsecutiveRoot = 128 - E`, which is 112 at `E = 16` and 120 at `E = 8`.
 *
 * Five places carry the two parameters and every one of them reduces to the plain expression at
 * `f = 1, PrimitiveStep = 1`: the generator's roots, the syndrome evaluation points, the erasure locator's `X`, the Chien
 * search's step, and Forney's magnitude, which gains the factor `X^(1-f)`. That factor is the one
 * that is easy to leave out, and leaving it out produces a decoder that locates errors correctly
 * and corrects them wrongly — caught here by the post-correction syndrome check, which then fails
 * on every correctable word rather than on none.
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

/// The greatest common divisor, for the coprimality a primitive step has to satisfy.
[[nodiscard]] constexpr unsigned greatestCommonDivisor(unsigned a, unsigned b) noexcept {
    while (b != 0U) {
        const unsigned rest = a % b;
        a                   = b;
        b                   = rest;
    }
    return a;
}

/// The product of two field elements through the logarithm tables, zero absorbing.
template<std::size_t SymbolBits, unsigned FieldPoly>
[[nodiscard]] constexpr std::uint8_t gfMultiply(std::uint8_t a, std::uint8_t b) noexcept {
    constexpr auto&       gf      = kGf<SymbolBits, FieldPoly>;
    constexpr std::size_t nonzero = GfTables<SymbolBits, FieldPoly>::kNonzero;
    if (a == 0U || b == 0U) {
        return 0U;
    }
    return gf.exp[(static_cast<unsigned>(gf.log[a]) + static_cast<unsigned>(gf.log[b])) % static_cast<unsigned>(nonzero)];
}

} // namespace detail

/**
 * @brief The 2^M - 1 symbol Reed-Solomon code over GF(2^M) with `Roots` parity symbols.
 *
 * Blocks are laid out with the information symbols first and the parity last, and a
 * shortened code is used by holding the first `pad` symbols at zero. Symbols are one byte
 * each, which serves every field through GF(256).
 */
template<std::size_t SymbolBits, unsigned FieldPoly, std::size_t Roots, unsigned FirstConsecutiveRoot = 1U, unsigned PrimitiveStep = 1U>
struct ReedSolomon {
    static_assert(SymbolBits >= 2UZ && SymbolBits <= 8UZ, "a symbol is carried in one byte");
    static_assert((FieldPoly >> SymbolBits) == 1U, "the field polynomial carries the x^M term and nothing above it");

    static constexpr std::size_t  kBlock       = (1UZ << SymbolBits) - 1UZ;
    static constexpr std::size_t  kRoots       = Roots;
    static constexpr unsigned     kCorrectable = static_cast<unsigned>(Roots / 2UZ);
    static constexpr std::uint8_t kLogZero     = static_cast<std::uint8_t>(kBlock); ///< the logarithm zero does not have

    static_assert(PrimitiveStep >= 1U && detail::greatestCommonDivisor(PrimitiveStep, static_cast<unsigned>(kBlock)) == 1U, "the primitive step must be coprime with the block length, or the powers of beta do not run over the whole field and the roots repeat");
    static_assert(FirstConsecutiveRoot < static_cast<unsigned>(kBlock), "the first consecutive root is an exponent of beta and lives below the block length");

    using Block = std::array<std::uint8_t, kBlock>;

    /// Reduce an exponent into 0..kBlock-1. Exponents arrive as sums, differences and small
    /// multiples of table indices, so the reduction is a plain remainder rather than a subtraction.
    [[nodiscard]] static constexpr unsigned modExp(unsigned x) noexcept { return x % static_cast<unsigned>(kBlock); }

    /// The alpha exponent of the `index`-th generator root, `beta^(FirstConsecutiveRoot + index)`.
    [[nodiscard]] static constexpr unsigned rootExponent(std::size_t index) noexcept { return modExp(PrimitiveStep * (FirstConsecutiveRoot + static_cast<unsigned>(index))); }

    /// The generator polynomial in logarithm form, constant term first; its roots are the `Roots`
    /// consecutive powers `beta^FirstConsecutiveRoot` upward.
    [[nodiscard]] static constexpr std::array<std::uint8_t, Roots + 1UZ> generator() noexcept {
        constexpr auto& gf = detail::kGf<SymbolBits, FieldPoly>;

        std::array<std::uint8_t, Roots + 1UZ> g{};
        g[0] = 1U;
        for (std::size_t i = 0UZ; i < Roots; ++i) {
            const unsigned root = rootExponent(i);
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
    static RsResult decode(Block& block, std::size_t pad = 0UZ) noexcept { return correct(block, {}, pad); }

    /// Correct `block` in place, the symbols at the block indices in `erasures` taken as erased:
    /// their positions are known and their values are not. `e` errors are corrected together with
    /// the erasures whenever `2e + erasures.size() <= Roots`. The result's `errors` counts the
    /// symbols changed, which leaves out an erased symbol that already held the transmitted value.
    /// A position outside `[pad, kBlock)`, a position given twice or more than `Roots` positions is
    /// refused: the result is not `valid` and `block` is untouched. No erasures is `decode`.
    static RsResult decodeWithErasures(Block& block, std::span<const std::size_t> erasures, std::size_t pad = 0UZ) noexcept {
        if (erasures.size() > Roots) {
            return {};
        }
        std::array<bool, kBlock> named{};
        for (const std::size_t position : erasures) {
            if (position < pad || position >= kBlock || named[position]) {
                return {};
            }
            named[position] = true;
        }
        return correct(block, erasures, pad);
    }

private:
    static RsResult correct(Block& block, std::span<const std::size_t> erasures, std::size_t pad) noexcept {
        constexpr auto& gf = detail::kGf<SymbolBits, FieldPoly>;
        RsResult        result;

        // Syndromes: the received polynomial evaluated at the generator's own roots.
        std::array<std::uint8_t, Roots> syndrome{};
        std::uint8_t                    any = 0U;
        for (std::size_t i = 0UZ; i < Roots; ++i) {
            const unsigned point = rootExponent(i);
            std::uint8_t   s     = block[0];
            for (std::size_t j = 1UZ; j < kBlock; ++j) {
                s = static_cast<std::uint8_t>(block[j] ^ ((s == 0U) ? 0U : gf.exp[modExp(static_cast<unsigned>(gf.log[s]) + point)]));
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

        // Berlekamp-Massey for the error locator, from the erasure locator.
        std::array<std::uint8_t, Roots + 1UZ> lambda{};
        std::array<std::uint8_t, Roots + 1UZ> back{};
        std::array<std::uint8_t, Roots + 1UZ> next{};
        lambda[0] = 1U;
        for (std::size_t k = 0UZ; k < erasures.size(); ++k) {
            const unsigned locator = modExp(PrimitiveStep * static_cast<unsigned>(kBlock - 1UZ - erasures[k]));
            for (std::size_t j = k + 1UZ; j > 0UZ; --j) {
                if (lambda[j - 1UZ] != 0U) {
                    lambda[j] ^= gf.exp[modExp(locator + static_cast<unsigned>(gf.log[lambda[j - 1UZ]]))];
                }
            }
        }
        for (std::size_t i = 0UZ; i <= Roots; ++i) {
            back[i] = gf.log[lambda[i]];
        }

        const std::size_t erased = erasures.size();
        std::size_t       length = erased;
        for (std::size_t r = erased + 1UZ; r <= Roots; ++r) {
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
            if (2UZ * length <= r + erased - 1UZ) {
                length = r + erased - length;
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

        // Chien search: beta^-i is a root of lambda exactly when symbol i is in error or erased, so the
        // register steps by PrimitiveStep in the alpha exponent rather than by one.
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
                    reg[j] = modExp(reg[j] + static_cast<unsigned>(j) * PrimitiveStep);
                    q ^= gf.exp[reg[j]];
                }
            }
            if (q != 0U) {
                continue;
            }
            rootExp[found]  = modExp(PrimitiveStep * static_cast<unsigned>(i));
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
            // e = X^(1-f) * omega(X^-1) / lambda'(X^-1). `rootExp[k]` is the alpha exponent of
            // X^-1, so X is alpha^(kBlock - rootExp[k]) and the factor's exponent is that times
            // 1 - f reduced into the exponent group. At f = 1 the factor is one and drops out.
            const unsigned inverseExp = modExp(static_cast<unsigned>(kBlock) - rootExp[k]);
            const unsigned factorExp  = modExp(inverseExp * modExp(static_cast<unsigned>(kBlock) + 1U - FirstConsecutiveRoot));
            const unsigned magnitude  = modExp(factorExp + static_cast<unsigned>(gf.log[numerator]) + static_cast<unsigned>(kBlock) - gf.log[denominator]);
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
            const unsigned point = rootExponent(i);
            std::uint8_t   s     = block[0];
            for (std::size_t j = 1UZ; j < kBlock; ++j) {
                s = static_cast<std::uint8_t>(block[j] ^ ((s == 0U) ? 0U : gf.exp[modExp(static_cast<unsigned>(gf.log[s]) + point)]));
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

/// The GF(256) family CCSDS 131.0-B-5 section 4 specifies: `F(x) = x^8 + x^7 + x^2 + x + 1` as the
/// integer `0x187`, and the generator's roots stepping by `alpha^11`. `Fcr` is `128 - E`, read off
/// 4.3.4's lower product limit, so the two codes are `ReedSolomonCcsds<32UZ, 112U>` at `E = 16`,
/// which is RS(255,223), and `ReedSolomonCcsds<16UZ, 120U>` at `E = 8`, which is RS(255,239).
template<std::size_t Roots, unsigned Fcr>
using ReedSolomonCcsds = ReedSolomon<8UZ, 0x187U, Roots, Fcr, 11U>;

using ReedSolomonCcsds255_223 = ReedSolomonCcsds<32UZ, 112U>;
using ReedSolomonCcsds255_239 = ReedSolomonCcsds<16UZ, 120U>;

/**
 * @brief The dual (Berlekamp) basis of GF(2^M), as the two element-wise recodings between it and the
 * conventional basis.
 *
 * The conventional basis is `(1, alpha, ..., alpha^(M-1))`, so an element is
 * `u_(M-1) alpha^(M-1) + ... + u_1 alpha + u_0`. A second basis is `(1, b, b^2, ..., b^(M-1))` with
 * `b = alpha^BasisExponent`, and the **dual** basis `(l_0, ..., l_(M-1))` is defined by
 * `Tr(l_i b^j) = 1` when `i == j` and `0` otherwise, where the trace is
 * `Tr(z) = sum over k = 0..M-1 of z^(2^k)`. A symbol in the dual representation is
 * `z_0 l_0 + ... + z_(M-1) l_(M-1)`.
 *
 * Everything follows from those two sentences in one line. Taking the trace of `w b^j` where
 * `w = sum_i z_i l_i` gives `Tr(w b^j) = sum_i z_i Tr(l_i b^j) = z_j`, so
 *
 *     z_j = Tr(w * b^j) = sum over k of u_k * T[j][k],   T[j][k] = Tr(alpha^k * alpha^(BasisExponent*j))
 *
 * which is an M by M matrix over GF(2) computed from the field polynomial alone. The map is a
 * GF(2)-linear bijection, so a table of `2^M` entries realizes it and its inverse is the scatter of
 * that table — built here with a written-count check, on the pattern `Golay.hpp` uses for its coset
 * table, so a `BasisExponent` whose first M powers are not independent leaves every lookup zero and
 * `isBijective()` false. Each named basis asserts `isBijective()` where it is named, so such an
 * exponent is a build failure there rather than a quietly wrong table.
 *
 * **Octet packing.** CCSDS 131.0-B-5 1.6.3 numbers the first transmitted bit of a field as the most
 * significant, and 4.3.9.2 transmits `z_0` first, so `z_j` occupies bit `M-1-j` of the dual octet
 * while `u_k` occupies bit `k` of the conventional one in the ordinary way. Both packings are stated
 * because they are the only remaining place a correct transformation still produces a wrong byte.
 *
 * `T` is GF(2)-linear but is **not** a field isomorphism — it does not preserve multiplication — so
 * every arithmetic step stays in the conventional representation and the recoding is applied to a
 * whole block before and after a call, never inside the decoder's loops.
 */
template<std::size_t SymbolBits, unsigned FieldPoly, unsigned BasisExponent>
struct DualBasis {
    static_assert(SymbolBits >= 2UZ && SymbolBits <= 8UZ, "a symbol is carried in one byte");

    static constexpr std::size_t kElements = 1UZ << SymbolBits;
    static constexpr std::size_t kNonzero  = kElements - 1UZ;

    /// `Tr(z) = sum over k of z^(2^k)`, which lands in the prime subfield and so is 0 or 1.
    [[nodiscard]] static constexpr std::uint8_t trace(std::uint8_t z) noexcept {
        std::uint8_t sum   = 0U;
        std::uint8_t power = z;
        for (std::size_t k = 0UZ; k < SymbolBits; ++k) {
            sum   = static_cast<std::uint8_t>(sum ^ power);
            power = detail::gfMultiply<SymbolBits, FieldPoly>(power, power);
        }
        return sum;
    }

    /// Row `j` of `T` as a bitmask, bit `k` holding `T[j][k] = Tr(alpha^k * b^j)`.
    [[nodiscard]] static constexpr std::array<std::uint8_t, SymbolBits> traceMatrix() noexcept {
        constexpr auto&                      gf = detail::kGf<SymbolBits, FieldPoly>;
        std::array<std::uint8_t, SymbolBits> rows{};
        for (std::size_t j = 0UZ; j < SymbolBits; ++j) {
            const unsigned basis = (BasisExponent * static_cast<unsigned>(j)) % static_cast<unsigned>(kNonzero);
            std::uint8_t   row   = 0U;
            for (std::size_t k = 0UZ; k < SymbolBits; ++k) {
                const std::uint8_t element = gf.exp[(basis + static_cast<unsigned>(k)) % static_cast<unsigned>(kNonzero)];
                row                        = static_cast<std::uint8_t>(row | (trace(element) << k));
            }
            rows[j] = row;
        }
        return rows;
    }

    /// The conventional-to-dual table, `z_j` at bit `M-1-j`.
    [[nodiscard]] static constexpr std::array<std::uint8_t, kElements> forwardTable() noexcept {
        constexpr auto                      rows = traceMatrix();
        std::array<std::uint8_t, kElements> table{};
        for (std::size_t value = 0UZ; value < kElements; ++value) {
            std::uint8_t dual = 0U;
            for (std::size_t j = 0UZ; j < SymbolBits; ++j) {
                const unsigned parity = static_cast<unsigned>(std::popcount(static_cast<unsigned>(rows[j]) & static_cast<unsigned>(value))) & 1U;
                dual                  = static_cast<std::uint8_t>(dual | (parity << (SymbolBits - 1UZ - j)));
            }
            table[value] = dual;
        }
        return table;
    }

    /// The scatter of `forwardTable()`, with the coverage check that says the basis is a basis.
    [[nodiscard]] static constexpr std::array<std::uint8_t, kElements> inverseTable() noexcept {
        constexpr auto                      forward = forwardTable();
        std::array<std::uint8_t, kElements> table{};
        std::array<std::uint8_t, kElements> written{};
        for (std::size_t value = 0UZ; value < kElements; ++value) {
            table[forward[value]] = static_cast<std::uint8_t>(value);
            ++written[forward[value]];
        }
        for (const std::uint8_t count : written) {
            if (count != 1U) {
                return {}; // not a bijection: `isBijective()` reports it and every lookup is zero
            }
        }
        return table;
    }

    static constexpr std::array<std::uint8_t, kElements> kToDual   = forwardTable();
    static constexpr std::array<std::uint8_t, kElements> kFromDual = inverseTable();

    [[nodiscard]] static constexpr std::uint8_t toDual(std::uint8_t conventional) noexcept { return kToDual[conventional]; }
    [[nodiscard]] static constexpr std::uint8_t fromDual(std::uint8_t dual) noexcept { return kFromDual[dual]; }

    /// Whether the two tables invert each other over the whole field. A caller has no reason to ask;
    /// a test does, and so does a `BasisExponent` nobody has checked.
    [[nodiscard]] static constexpr bool isBijective() noexcept {
        for (std::size_t value = 0UZ; value < kElements; ++value) {
            const std::uint8_t element = static_cast<std::uint8_t>(value);
            if (fromDual(toDual(element)) != element || toDual(fromDual(element)) != element) {
                return false;
            }
        }
        return true;
    }

    /// The dual basis element `l_i`, in the conventional representation: column `i` of `T^-1`, which
    /// is what the dual octet carrying only `z_i` maps back to.
    [[nodiscard]] static constexpr std::uint8_t basisElement(std::size_t index) noexcept { return fromDual(static_cast<std::uint8_t>(1U << (SymbolBits - 1UZ - index))); }
};

/// The basis CCSDS 131.0-B-5 4.4.2 names: GF(256) on `0x187` with `b = alpha^117`. That element has
/// multiplicative order 85 and so is not primitive, and nothing requires it to be — only that its
/// first eight powers are linearly independent, which `isBijective()` is the check of.
using CcsdsDualBasis = DualBasis<8UZ, 0x187U, 117U>;
static_assert(CcsdsDualBasis::isBijective(), "gr::fec::CcsdsDualBasis: the first eight powers of alpha^117 must be linearly independent, or the dual-basis tables do not invert each other");

/**
 * @brief The symbol interleaving of CCSDS 131.0-B-5 4.4.1, as the index map it is.
 *
 * A commutator distributes successive codeblock symbols across `depth` encoders and a second
 * commutator reassembles them in step, which stated as an index map is: codeblock symbol `s` belongs
 * to codeword `s mod depth` at position `s div depth` within it. At `depth = 1` the map is the
 * identity and the codeblock is one codeword, which is the standard's own statement that `I = 1` is
 * the absence of interleaving.
 *
 * Shortening composes with it exactly as the standard states: the fill `Q` is a multiple of `depth`
 * (4.3.7.3), so every codeword carries the same `pad = Q / depth` leading zeros and a shortened
 * transmitted codeblock is `(kBlock - pad) * depth` symbols. That makes this map exactly
 * `Interleaver<std::uint8_t>::block(kBlock - pad, depth)` written out, which is what the qa proves
 * rather than what this comment asserts. It is written out because it is two lines of index
 * arithmetic belonging to one codeblock, and reaching the interleaver for them would make this
 * header depend on `fec/Interleaver.hpp`; the qa holds the two against each other instead.
 *
 * The code itself stays a one-codeword entity: these are free functions over a whole codeblock, not
 * members, because interleaving is a property of how codewords are laid out on the wire and not of
 * the code.
 */
[[nodiscard]] inline constexpr bool ccsdsInterleaveDepthAllowed(std::size_t depth) noexcept { return depth == 1UZ || depth == 2UZ || depth == 3UZ || depth == 4UZ || depth == 5UZ || depth == 8UZ; }

/// Gather `depth` contiguous codewords of `rows` symbols each into one interleaved codeblock.
inline void interleaveCodewords(std::span<const std::uint8_t> codewords, std::span<std::uint8_t> codeblock, std::size_t rows, std::size_t depth) noexcept {
    const std::size_t count = rows * depth;
    if (codewords.size() < count || codeblock.size() < count) {
        return;
    }
    for (std::size_t s = 0UZ; s < count; ++s) {
        codeblock[s] = codewords[(s % depth) * rows + (s / depth)];
    }
}

/// The inverse gather: one interleaved codeblock back into `depth` contiguous codewords.
inline void deinterleaveCodewords(std::span<const std::uint8_t> codeblock, std::span<std::uint8_t> codewords, std::size_t rows, std::size_t depth) noexcept {
    const std::size_t count = rows * depth;
    if (codeblock.size() < count || codewords.size() < count) {
        return;
    }
    for (std::size_t s = 0UZ; s < count; ++s) {
        codewords[(s % depth) * rows + (s / depth)] = codeblock[s];
    }
}

} // namespace gr::fec

#endif // GNURADIO_ALGORITHM_FEC_REED_SOLOMON_HPP
