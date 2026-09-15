#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <random>
#include <span>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/algorithm/fec/Interleaver.hpp>
#include <gnuradio-4.0/algorithm/fec/ReedSolomon.hpp>

/*
 * The field, the generator polynomial and the decoder are all built here rather than carried,
 * so these tests pin each layer against something that is not the construction restated.
 *
 *   - the field: alpha has order 63 exactly, and the logarithm and antilogarithm tables are
 *     inverses over every element. A field polynomial that is irreducible but not primitive
 *     passes neither;
 *   - the generator: alpha^1 through alpha^R are roots of it and alpha^(R+1) is not, which is
 *     what fixes both the number of roots and where they start. A generator built for a
 *     different first root produces a code that still encodes and decodes self-consistently
 *     and agrees with nothing else in the world, which is exactly why this is checked;
 *   - the codewords: three parity vectors that are the known parity of these codes for the
 *     information below. These are the values nothing here measured, and they pin the field, the
 *     generator, the first root and the shortening together.
 *
 * Correction is then checked at its exact boundary and past it. Up to t symbol errors must
 * come back as the block that was encoded. PAST t the decoder may legitimately fail, and it
 * may legitimately land on a different codeword when the received block really is closer to
 * one -- no decoder can do better than that. What it may not do is return a block that is not
 * a codeword at all, or report a correction count that does not match what it changed. Those
 * two are asserted on every trial, at every error weight, because they are the shape a silent
 * wrong answer would take.
 *
 * The three driving instantiations are P25 Phase 1's codes over GF(64) on x^6 + x + 1
 * (TIA-102.BAAA), reached through the ReedSolomon6 alias:
 *
 *   ReedSolomon6<12>  RS(24,12,13)  corrects 6   Link Control, in a voice frame carrying it
 *   ReedSolomon6<8>   RS(24,16,9)   corrects 4   Encryption Sync, in a voice frame carrying it
 *   ReedSolomon6<16>  RS(36,20,17) corrects 8   the header that opens a transmission
 */
namespace {

using gr::fec::CcsdsDualBasis;
using gr::fec::ccsdsInterleaveDepthAllowed;
using gr::fec::deinterleaveCodewords;
using gr::fec::interleaveCodewords;
using gr::fec::Interleaver;
using gr::fec::ReedSolomon6;
using gr::fec::ReedSolomonCcsds255_223;
using gr::fec::ReedSolomonCcsds255_239;
using gr::fec::RsResult;

/// The GF(256) field CCSDS 131.0-B-5 4.3.3 names, on `F(x) = x^8 + x^7 + x^2 + x + 1`.
constexpr auto& kGf256 = gr::fec::detail::kGf<8UZ, 0x187U>;

/// A codeword evaluated at `alpha^exp`, its first symbol the highest power, written here rather than
/// taken from the decoder so a syndrome check is a statement about the encoder and not about itself.
template<typename Code>
[[nodiscard]] std::uint8_t evaluateCcsds(const typename Code::Block& block, unsigned exp) {
    std::uint8_t value = 0U;
    for (const std::uint8_t symbol : block) {
        const std::uint8_t scaled = value == 0U ? std::uint8_t{0} : kGf256.exp[(static_cast<unsigned>(kGf256.log[value]) + exp) % 255U];
        value                     = static_cast<std::uint8_t>(symbol ^ scaled);
    }
    return value;
}

/// @p count distinct symbol positions of a 255-symbol block, at or above @p from.
[[nodiscard]] std::vector<std::size_t> distinctPositions(std::mt19937_64& engine, std::size_t count, std::size_t from) {
    std::vector<std::size_t> chosen;
    while (chosen.size() < count) {
        const std::size_t position = from + (engine() % (255UZ - from));
        if (std::ranges::find(chosen, position) == chosen.end()) {
            chosen.push_back(position);
        }
    }
    return chosen;
}

/// The field the three P25 codes share: GF(64) on x^6 + x + 1.
constexpr auto& kGf64 = gr::fec::detail::kGf<6UZ, 0x43U>;

/// Zero has no logarithm; every ReedSolomon6<R> instantiation carries the same sentinel because
/// they all share this field.
constexpr std::uint8_t kRsLogZero = ReedSolomon6<12UZ>::kLogZero;

[[nodiscard]] constexpr std::uint8_t gfMul(std::uint8_t a, std::uint8_t b) noexcept {
    if (a == 0U || b == 0U) {
        return 0U;
    }
    return kGf64.exp[(static_cast<unsigned>(kGf64.log[a]) + static_cast<unsigned>(kGf64.log[b])) % 63U];
}

/// Evaluate a block as a polynomial, its first symbol the highest power, at alpha^exp.
[[nodiscard]] std::uint8_t evaluate(const std::array<std::uint8_t, 63UZ>& block, unsigned exp) {
    std::uint8_t acc = 0U;
    for (std::size_t i = 0UZ; i < block.size(); ++i) {
        acc = static_cast<std::uint8_t>(gfMul(acc, kGf64.exp[exp]) ^ block[i]);
    }
    return acc;
}

/// Every property that must hold whatever the decoder decided.
template<typename Code>
void checkInvariants(const std::array<std::uint8_t, 63UZ>& received, const std::array<std::uint8_t, 63UZ>& corrected, const RsResult& r, std::size_t pad, const char* what) {
    using namespace boost::ut;
    unsigned distance = 0U;
    for (std::size_t i = 0UZ; i < 63UZ; ++i) {
        if (received[i] != corrected[i]) {
            ++distance;
        }
    }
    if (r.valid) {
        expect(eq(distance, r.errors)) << what;
        for (std::size_t i = 0UZ; i < pad; ++i) {
            expect(eq(corrected[i], std::uint8_t{0U})) << "an accepted block leaves the shortening's padding at zero";
        }
        for (std::size_t i = 0UZ; i < Code::kRoots; ++i) {
            expect(eq(evaluate(corrected, static_cast<unsigned>(i + 1UZ)), std::uint8_t{0U})) << "an accepted block really is a codeword: every generator root annihilates it";
        }
    }
}

std::uint64_t rng = 0xB5026F5AA96619E9ULL;

std::uint64_t next() {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return rng >> 17U;
}

template<typename Code>
void exercise(const char* name, std::size_t pad, unsigned trials) {
    using namespace boost::ut;
    constexpr unsigned t    = Code::kCorrectable;
    const std::size_t  info = 63UZ - Code::kRoots;

    long exactAtLimit = 0, acceptedBeyond = 0, wrongBeyond = 0, trialsBeyond = 0;

    for (unsigned k = 0U; k < trials; ++k) {
        std::array<std::uint8_t, 63UZ> clean{};
        for (std::size_t i = pad; i < info; ++i) {
            clean[i] = static_cast<std::uint8_t>(next() & 0x3FU);
        }
        Code::encode(clean, pad);

        // a clean block must decode with nothing corrected
        {
            std::array<std::uint8_t, 63UZ> out = clean;
            const RsResult                 r   = Code::decode(out, pad);
            expect(that % (r.valid && r.errors == 0U && out == clean)) << "a clean codeword decodes untouched";
        }

        // at and under the limit: exact recovery, every time
        for (unsigned e = 1U; e <= t; ++e) {
            std::array<std::uint8_t, 63UZ> received = clean;
            std::array<bool, 63UZ>         hit{};
            for (unsigned q = 0U; q < e; ++q) {
                std::size_t p = 0UZ;
                do {
                    p = pad + next() % (63UZ - pad);
                } while (hit[p]);
                hit[p]         = true;
                std::uint8_t d = 0U;
                do {
                    d = static_cast<std::uint8_t>(next() & 0x3FU);
                } while (d == 0U);
                received[p] = static_cast<std::uint8_t>(received[p] ^ d);
            }
            std::array<std::uint8_t, 63UZ> out = received;
            const RsResult                 r   = Code::decode(out, pad);
            checkInvariants<Code>(received, out, r, pad, "the reported correction count is what the decoder changed");
            if (r.valid && out == clean && r.errors == e) {
                ++exactAtLimit;
            } else {
                expect(false) << "a block inside the correction limit came back wrong";
            }
        }

        // past the limit: whatever it decides, the invariants above still hold
        {
            const unsigned                 e        = t + 1U + static_cast<unsigned>(next() % 3U);
            std::array<std::uint8_t, 63UZ> received = clean;
            std::array<bool, 63UZ>         hit{};
            for (unsigned q = 0U; q < e && q < (63UZ - pad); ++q) {
                std::size_t p = 0UZ;
                do {
                    p = pad + next() % (63UZ - pad);
                } while (hit[p]);
                hit[p]         = true;
                std::uint8_t d = 0U;
                do {
                    d = static_cast<std::uint8_t>(next() & 0x3FU);
                } while (d == 0U);
                received[p] = static_cast<std::uint8_t>(received[p] ^ d);
            }
            std::array<std::uint8_t, 63UZ> out = received;
            const RsResult                 r   = Code::decode(out, pad);
            checkInvariants<Code>(received, out, r, pad, "the reported correction count is what the decoder changed");
            ++trialsBeyond;
            if (r.valid) {
                ++acceptedBeyond;
                if (out != clean) {
                    ++wrongBeyond;
                }
            }
        }
    }
    expect(eq(exactAtLimit, static_cast<long>(trials) * static_cast<long>(t))) << name << ": every block inside the limit was recovered exactly (beyond it: accepted " << acceptedBeyond << " of " << trialsBeyond << ", of which not the transmitted word " << wrongBeyond << ")";
}

} // namespace

const boost::ut::suite<"reed-solomon"> reedSolomonTests = [] {
    using namespace boost::ut;

    "the field"_test = [] {
        expect(eq(kGf64.exp[0], std::uint8_t{1U})) << "alpha^0 is one";
        bool                  distinct = true;
        std::array<bool, 64U> seen{};
        for (unsigned i = 0U; i < 63U; ++i) {
            const std::uint8_t v = kGf64.exp[i];
            if (v == 0U || seen[v]) {
                distinct = false;
            }
            seen[v] = true;
        }
        expect(that % distinct) << "the 63 powers of alpha are the 63 nonzero elements, so the polynomial is primitive";
        expect(eq(gfMul(kGf64.exp[62], kGf64.exp[1]), std::uint8_t{1U})) << "alpha has order exactly 63";

        bool inverses = true;
        for (unsigned v = 1U; v < 64U; ++v) {
            if (kGf64.exp[kGf64.log[v]] != v) {
                inverses = false;
            }
        }
        expect(that % inverses) << "the logarithm and antilogarithm tables are inverses over every nonzero element";
        expect(eq(kGf64.log[0], kRsLogZero)) << "zero has the sentinel logarithm";
    };

    "the generator's roots are alpha^1 .. alpha^R and stop there"_test = [] {
        const auto checkRoots = [](const auto& g, std::size_t roots, const char* name) {
            // g is in logarithm form, constant term first; evaluate it at alpha^e.
            const auto eval = [&g, roots](unsigned e) {
                std::uint8_t acc = 0U;
                for (std::size_t i = 0UZ; i <= roots; ++i) {
                    if (g[i] != kRsLogZero) {
                        acc = static_cast<std::uint8_t>(acc ^ kGf64.exp[(g[i] + e * static_cast<unsigned>(i)) % 63U]);
                    }
                }
                return acc;
            };
            bool ok = true;
            for (std::size_t i = 1UZ; i <= roots; ++i) {
                if (eval(static_cast<unsigned>(i)) != 0U) {
                    ok = false;
                }
            }
            expect(that % ok) << name;
            expect(neq(eval(static_cast<unsigned>(roots) + 1U), 0U)) << "and the next power of alpha is not a root, which fixes how many there are";
            expect(neq(eval(0U), 0U)) << "alpha^0 is not a root, which fixes where they start";
        };
        constexpr auto g12 = ReedSolomon6<12UZ>::generator();
        constexpr auto g8  = ReedSolomon6<8UZ>::generator();
        constexpr auto g16 = ReedSolomon6<16UZ>::generator();
        checkRoots(g12, 12UZ, "RS(24,12,13)'s generator has roots alpha^1 .. alpha^12");
        checkRoots(g8, 8UZ, "RS(24,16,9)'s generator has roots alpha^1 .. alpha^8");
        checkRoots(g16, 16UZ, "RS(36,20,17)'s generator has roots alpha^1 .. alpha^16");
    };

    "the known parity of each code, for one stated information word"_test = [] {
        const auto pin = [](auto code, std::size_t pad, const std::vector<std::uint8_t>& expectVals, const char* name) {
            using Code = decltype(code);
            std::array<std::uint8_t, 63UZ> block{};
            const std::size_t              info = 63UZ - Code::kRoots;
            for (std::size_t i = pad; i < info; ++i) {
                block[i] = static_cast<std::uint8_t>((7UZ * (i - pad) + 5UZ) & 0x3FUZ);
            }
            Code::encode(block, pad);
            bool same = true;
            for (std::size_t i = 0UZ; i < Code::kRoots; ++i) {
                if (block[info + i] != expectVals[i]) {
                    same = false;
                }
            }
            expect(that % same) << name;
        };
        pin(ReedSolomon6<12UZ>{}, 39UZ, {0x1AU, 0x3FU, 0x3AU, 0x3AU, 0x0DU, 0x03U, 0x05U, 0x21U, 0x1FU, 0x1DU, 0x17U, 0x19U}, "RS(24,12,13) parity is the known parity for this code");
        pin(ReedSolomon6<8UZ>{}, 39UZ, {0x24U, 0x0BU, 0x1EU, 0x3BU, 0x0AU, 0x0AU, 0x09U, 0x0DU}, "RS(24,16,9) parity is the known parity for this code");
        pin(ReedSolomon6<16UZ>{}, 27UZ, {0x3FU, 0x38U, 0x0DU, 0x31U, 0x1FU, 0x27U, 0x05U, 0x06U, 0x04U, 0x25U, 0x09U, 0x05U, 0x2BU, 0x1CU, 0x2BU, 0x1EU}, "RS(36,20,17) parity is the known parity for this code");
    };

    "correction at the limit and past it"_test = [] {
        exercise<ReedSolomon6<12UZ>>("RS(24,12,13)", 39UZ, 4000U);
        exercise<ReedSolomon6<8UZ>>("RS(24,16,9)", 39UZ, 4000U);
        exercise<ReedSolomon6<16UZ>>("RS(36,20,17)", 27UZ, 4000U);
    };

    "a correction landing in the shortening's padding is refused"_test = [] {
        // A receiver builds the padding itself and always builds it as zeros, so the case to
        // catch is the decoder PUTTING something there. Provoke it: encode a codeword of the
        // full-length code carrying a nonzero symbol in the padding, then hand the decoder the
        // block a receiver would actually have -- the same symbols with the padding zeroed.
        // The nearest codeword is the one that was encoded, so the decoder restores the symbol
        // the air could not have carried, and that is what must be refused.
        std::array<std::uint8_t, 63UZ> full{};
        full[10] = 0x2AU;
        for (std::size_t i = 39UZ; i < 51UZ; ++i) {
            full[i] = static_cast<std::uint8_t>((3UZ * i) & 0x3FUZ);
        }
        ReedSolomon6<12UZ>::encode(full, 0UZ);

        std::array<std::uint8_t, 63UZ> received = full;
        for (std::size_t i = 0UZ; i < 39UZ; ++i) {
            received[i] = 0U;
        }
        const RsResult r = ReedSolomon6<12UZ>::decode(received, 39UZ);
        expect(that % r.pad_corrupted) << "a correction landing inside the shortening's padding is noticed";
        expect(that % !r.valid) << "and the block is refused rather than handed back with the padding quietly rewritten";
        expect(eq(received[10], 0x2AU)) << "the decoder really did place the symbol there, so the check has something to catch";
    };

    "a locator whose corrections do not produce a codeword is refused"_test = [] {
        // Past the correction limit the decoder can find a locator polynomial with the right
        // number of roots in the field and still be describing no codeword at all -- the
        // magnitudes it then applies leave a block that satisfies nothing. It is rare: it
        // happens on the order of once in ten thousand blocks past the limit, so a random
        // sweep of any affordable size will usually miss it. These are received blocks that
        // provoke it, each one refused by the syndrome check after correction and by nothing
        // else. Without that check every one of them would come back as a decoded message.
        const auto refuses = [](auto code, std::size_t pad, const std::vector<std::uint8_t>& transmitted, const char* name) {
            using Code = decltype(code);
            typename Code::Block block{};
            for (std::size_t i = 0UZ; i < transmitted.size(); ++i) {
                block[pad + i] = transmitted[i];
            }
            const typename Code::Block received = block;
            const RsResult             r        = Code::decode(block, pad);
            expect(that % (block != received)) << "the decoder did get past its root search and correct something";
            expect(that % !r.pad_corrupted) << "and it was not the shortening's padding that gave it away";
            expect(that % !r.valid) << name;
        };
        refuses(ReedSolomon6<12UZ>{}, 39UZ, {0x3FU, 0x21U, 0x3FU, 0x03U, 0x35U, 0x03U, 0x2CU, 0x19U, 0x33U, 0x3CU, 0x00U, 0x3FU, 0x3AU, 0x24U, 0x25U, 0x0DU, 0x20U, 0x09U, 0x0FU, 0x0DU, 0x34U, 0x1AU, 0x2CU, 0x3EU}, "RS(24,12,13) refuses a corrected block that is not a codeword");
        refuses(ReedSolomon6<8UZ>{}, 39UZ, {0x20U, 0x3AU, 0x2DU, 0x20U, 0x20U, 0x0CU, 0x2AU, 0x20U, 0x2DU, 0x0AU, 0x16U, 0x0BU, 0x07U, 0x24U, 0x0BU, 0x2BU, 0x19U, 0x10U, 0x0AU, 0x2DU, 0x22U, 0x0EU, 0x01U, 0x0DU}, "RS(24,16,9) refuses a corrected block that is not a codeword");
        refuses(ReedSolomon6<16UZ>{}, 27UZ, {0x22U, 0x26U, 0x13U, 0x39U, 0x0AU, 0x1CU, 0x20U, 0x37U, 0x0CU, 0x3DU, 0x0DU, 0x2CU, 0x31U, 0x11U, 0x06U, 0x15U, 0x07U, 0x09U, 0x23U, 0x1AU, 0x1EU, 0x2DU, 0x14U, 0x0DU, 0x09U, 0x03U, 0x38U, 0x30U, 0x3AU, 0x26U, 0x28U, 0x26U, 0x24U, 0x09U, 0x3FU, 0x3CU}, "RS(36,20,17) refuses a corrected block that is not a codeword");
    };

    // The CCSDS generator, from 4.3.4's product limits rather than from any table.
    "the CCSDS generator's roots are the product limits, and it is self-reciprocal"_test = [] {
        using Code = ReedSolomonCcsds255_223;
        expect(eq(Code::kBlock, 255UZ));
        expect(eq(Code::kRoots, 32UZ));
        expect(eq(Code::kCorrectable, 16U)) << "255 - 223 = 32 = 2E, so t = E = 16";

        // 4.3.4's product runs over j = 128-E .. 127+E of (x - alpha^(11j)), so the roots written as
        // powers of alpha are 11j mod 255. The first four and the last are the values that derivation
        // gives, and all 32 are distinct because gcd(11, 255) = 1 makes alpha^11 primitive.
        expect(eq(Code::rootExponent(0UZ), 212U)) << "11 * 112 mod 255";
        expect(eq(Code::rootExponent(1UZ), 223U));
        expect(eq(Code::rootExponent(2UZ), 234U));
        expect(eq(Code::rootExponent(3UZ), 245U));
        expect(eq(Code::rootExponent(4UZ), 1U)) << "11 * 116 = 1276, which is 1 past five turns of 255";
        expect(eq(Code::rootExponent(31UZ), 43U)) << "11 * 143 mod 255";

        std::vector<unsigned> exponents;
        for (std::size_t i = 0UZ; i < Code::kRoots; ++i) {
            exponents.push_back(Code::rootExponent(i));
        }
        std::ranges::sort(exponents);
        expect(eq(std::ranges::unique(exponents).size(), 0UZ)) << "32 distinct roots, which is what the coprimality buys";

        // Annex F states the generator is self-reciprocal, which is a property of the code and needs no
        // table at all, and annex G's coefficient table is not reproduced here, so the derivation is
        // checked against the property it must satisfy on its own.
        constexpr auto g          = Code::generator();
        std::size_t    asymmetric = 0UZ;
        for (std::size_t i = 0UZ; i <= Code::kRoots; ++i) {
            asymmetric += g[i] == g[Code::kRoots - i] ? 0UZ : 1UZ;
        }
        expect(eq(asymmetric, 0UZ)) << "G_i equals G_(2E-i) for every i";
        expect(eq(g[0], std::uint8_t{0})) << "and both end coefficients are alpha^0, which is one";

        // The generator vanishes at its own roots and nowhere adjacent to them, which is what fixes the
        // count and the starting power together. A generator built for a different first root encodes
        // and decodes self-consistently and agrees with nothing else in the world.
        typename Code::Block asPolynomial{};
        for (std::size_t i = 0UZ; i <= Code::kRoots; ++i) {
            asPolynomial[Code::kBlock - 1UZ - i] = g[i] == Code::kLogZero ? std::uint8_t{0} : kGf256.exp[g[i]];
        }
        std::size_t nonZeroAtRoot = 0UZ;
        for (std::size_t i = 0UZ; i < Code::kRoots; ++i) {
            nonZeroAtRoot += evaluateCcsds<Code>(asPolynomial, Code::rootExponent(i)) == 0U ? 0UZ : 1UZ;
        }
        expect(eq(nonZeroAtRoot, 0UZ)) << "every one of the 32 roots is a root of the generator";
        expect(that % (evaluateCcsds<Code>(asPolynomial, Code::modExp(11U * 111U)) != 0U)) << "and the power one below the first is not";
        expect(that % (evaluateCcsds<Code>(asPolynomial, Code::modExp(11U * 144U)) != 0U)) << "nor the one above the last";

        expect(eq(ReedSolomonCcsds255_239::kRoots, 16UZ));
        expect(eq(ReedSolomonCcsds255_239::rootExponent(0UZ), Code::modExp(11U * 120U))) << "E = 8 starts at 128 - 8";
    };

    // The dual basis, from 4.4.2's definition alone.
    "the dual basis is the definition, and it is a bijection"_test = [] {
        expect(that % CcsdsDualBasis::isBijective()) << "the eight powers of alpha^117 are linearly independent, which is all the basis needs";

        // (b) Tr(l_i * b^j) is one when i equals j and zero otherwise, over all 64 pairs. This is the
        //     definition 4.4.2 states, asserted directly rather than through the tables it produced.
        std::size_t offside = 0UZ;
        for (std::size_t i = 0UZ; i < 8UZ; ++i) {
            for (std::size_t j = 0UZ; j < 8UZ; ++j) {
                const std::uint8_t power   = kGf256.exp[(117U * static_cast<unsigned>(j)) % 255U];
                const std::uint8_t product = gr::fec::detail::gfMultiply<8UZ, 0x187U>(CcsdsDualBasis::basisElement(i), power);
                offside += CcsdsDualBasis::trace(product) == (i == j ? 1U : 0U) ? 0UZ : 1UZ;
            }
        }
        expect(eq(offside, 0UZ)) << "Tr(l_i b^j) is the Kronecker delta over all 64 pairs";

        // (c) the round trip over all 256 elements, both ways, and GF(2)-linearity over all 65536 pairs
        std::size_t roundTrips = 0UZ;
        for (unsigned value = 0U; value < 256U; ++value) {
            const std::uint8_t element = static_cast<std::uint8_t>(value);
            roundTrips += CcsdsDualBasis::fromDual(CcsdsDualBasis::toDual(element)) == element && CcsdsDualBasis::toDual(CcsdsDualBasis::fromDual(element)) == element ? 1UZ : 0UZ;
        }
        expect(eq(roundTrips, 256UZ));

        std::size_t nonLinear = 0UZ;
        for (unsigned a = 0U; a < 256U; ++a) {
            for (unsigned b = 0U; b < 256U; ++b) {
                const std::uint8_t left  = CcsdsDualBasis::toDual(static_cast<std::uint8_t>(a ^ b));
                const std::uint8_t right = static_cast<std::uint8_t>(CcsdsDualBasis::toDual(static_cast<std::uint8_t>(a)) ^ CcsdsDualBasis::toDual(static_cast<std::uint8_t>(b)));
                nonLinear += left == right ? 0UZ : 1UZ;
            }
        }
        expect(eq(nonLinear, 0UZ)) << "the map is GF(2)-linear over all 65536 pairs, which is what makes a 256-entry table right";

        // the anchors the derivation produces, which need no published matrix: 4.3.9.3's two matrices are
        // not reproduced here, and these four values are the derivation's own outputs.
        expect(eq(CcsdsDualBasis::toDual(0x00U), std::uint8_t{0x00U})) << "linearity fixes zero";
        expect(eq(CcsdsDualBasis::toDual(0x01U), std::uint8_t{0x7BU}));
        expect(eq(CcsdsDualBasis::toDual(0x02U), std::uint8_t{0xAFU}));
        expect(eq(CcsdsDualBasis::toDual(0xFFU), std::uint8_t{0xBFU}));

        // the map is not a field isomorphism, which is why the recoding never moves inside the decoder
        expect(that % (CcsdsDualBasis::toDual(gr::fec::detail::gfMultiply<8UZ, 0x187U>(0x03U, 0x05U)) != gr::fec::detail::gfMultiply<8UZ, 0x187U>(CcsdsDualBasis::toDual(0x03U), CcsdsDualBasis::toDual(0x05U)))) << "linear, not multiplicative";
    };

    // 3 and 5. A codeword has zero syndromes, and shortening is padding then discarding.
    "a CCSDS codeword evaluates to zero at every root, at every pad"_test = [] {
        using Code = ReedSolomonCcsds255_223;
        std::mt19937_64 engine{0xA5A5A5A5ULL};

        std::size_t nonZero = 0UZ;
        for (const std::size_t pad : {0UZ, 1UZ, 17UZ, 100UZ, 222UZ}) {
            typename Code::Block block{};
            for (std::size_t i = pad; i < 223UZ; ++i) {
                block[i] = static_cast<std::uint8_t>(engine());
            }
            Code::encode(block, pad);
            for (std::size_t i = 0UZ; i < Code::kRoots; ++i) {
                nonZero += evaluateCcsds<Code>(block, Code::rootExponent(i)) == 0U ? 0UZ : 1UZ;
            }
        }
        expect(eq(nonZero, 0UZ)) << "the encoder's own output vanishes at all 32 roots, so the generator and the evaluation agree";

        // Encoding k - p symbols at pad = p gives the last n - p symbols of the same block preceded
        //    by p zeros and encoded at pad = 0, which is the identity 4.3.7.3 states.
        std::size_t mismatches = 0UZ;
        for (const std::size_t pad : {0UZ, 1UZ, 5UZ, 100UZ, 200UZ}) {
            typename Code::Block shortened{};
            typename Code::Block filled{};
            for (std::size_t i = pad; i < 223UZ; ++i) {
                const std::uint8_t symbol = static_cast<std::uint8_t>(engine());
                shortened[i]              = symbol;
                filled[i]                 = symbol;
            }
            Code::encode(shortened, pad);
            Code::encode(filled, 0UZ);
            for (std::size_t i = pad; i < Code::kBlock; ++i) {
                mismatches += shortened[i] == filled[i] ? 0UZ : 1UZ;
            }
        }
        expect(eq(mismatches, 0UZ)) << "shortening is virtual fill, which is exactly padding then discarding";
    };

    // Correction to the radius and refusal beyond it, counted.
    "the CCSDS code corrects sixteen symbols and refuses seventeen"_test = [] {
        using Code = ReedSolomonCcsds255_223;
        std::mt19937_64 engine{0x0DDBA11ULL};

        const auto trial = [&engine](unsigned errors) {
            typename Code::Block information{};
            for (std::size_t i = 0UZ; i < 223UZ; ++i) {
                information[i] = static_cast<std::uint8_t>(engine());
            }
            typename Code::Block sent = information;
            Code::encode(sent);

            typename Code::Block received = sent;
            for (const std::size_t position : distinctPositions(engine, errors, 0UZ)) {
                std::uint8_t noise = 0U;
                while (noise == 0U) {
                    noise = static_cast<std::uint8_t>(engine());
                }
                received[position] = static_cast<std::uint8_t>(received[position] ^ noise);
            }

            const RsResult result = Code::decode(received);
            const bool     exact  = std::equal(received.begin(), received.begin() + 223, information.begin());
            return std::tuple{result.valid, result.errors, exact};
        };

        for (unsigned errors = 0U; errors <= 16U; ++errors) {
            std::size_t recovered = 0UZ;
            for (std::size_t i = 0UZ; i < 20UZ; ++i) {
                const auto [valid, corrected, exact] = trial(errors);
                recovered += valid && corrected == errors && exact ? 1UZ : 0UZ;
            }
            expect(eq(recovered, 20UZ)) << std::format("{} injected symbol errors, all recovered with the count reported exactly", errors);
        }

        // the specification's own reference shape, at the radius and one past it
        std::size_t correctedAtLimit = 0UZ;
        std::size_t refusedBeyond    = 0UZ;
        std::size_t acceptedWrong    = 0UZ;
        for (std::size_t i = 0UZ; i < 200UZ; ++i) {
            const auto [valid, corrected, exact] = trial(16U);
            correctedAtLimit += valid && corrected == 16U && exact ? 1UZ : 0UZ;
        }
        for (std::size_t i = 0UZ; i < 200UZ; ++i) {
            const auto [valid, corrected, exact] = trial(17U);
            refusedBeyond += valid ? 0UZ : 1UZ;
            acceptedWrong += valid && !exact ? 1UZ : 0UZ;
        }
        expect(eq(correctedAtLimit, 200UZ)) << "200 of 200 corrected at sixteen errors";
        expect(eq(refusedBeyond, 200UZ)) << "200 of 200 refused at seventeen";
        expect(eq(acceptedWrong, 0UZ)) << "and none accepted but wrong";
    };

    // The interleaving is one index map, and it is the map the Interleaver kernel already has.
    "interleaving at depth one is the identity, and at depth I is the block permutation"_test = [] {
        std::mt19937_64 engine{0x5EEDULL};

        for (const std::size_t depth : {1UZ, 2UZ, 3UZ, 4UZ, 5UZ, 8UZ}) {
            expect(that % ccsdsInterleaveDepthAllowed(depth)) << std::format("4.3.5.1 allows depth {}", depth);
            for (const std::size_t pad : {0UZ, 7UZ, 200UZ}) {
                const std::size_t         rows  = 255UZ - pad;
                const std::size_t         count = rows * depth;
                std::vector<std::uint8_t> codewords(count);
                for (std::uint8_t& symbol : codewords) {
                    symbol = static_cast<std::uint8_t>(engine());
                }

                std::vector<std::uint8_t> codeblock(count, 0U);
                interleaveCodewords(codewords, codeblock, rows, depth);
                if (depth == 1UZ) {
                    expect(that % std::ranges::equal(codeblock, codewords)) << "depth one is the absence of interleaving, exactly";
                }

                std::vector<std::uint8_t> back(count, 0U);
                deinterleaveCodewords(codeblock, back, rows, depth);
                expect(that % std::ranges::equal(back, codewords)) << std::format("depth {}, pad {}: the two maps invert each other", depth, pad);

                // Interleaver's `block` kind at rows = 255 - pad and cols = depth takes a transmitted
                // codeblock to contiguous codewords, which is the deinterleave; its inverse is the
                // interleave. Proved against that kernel rather than believed.
                Interleaver<std::uint8_t> reference = Interleaver<std::uint8_t>::block(rows, depth);
                std::vector<std::uint8_t> gathered(count, 0U);
                expect(eq(reference.interleave(codeblock, gathered), count));
                expect(that % std::ranges::equal(gathered, codewords)) << std::format("depth {}, pad {}: the deinterleave is the block permutation", depth, pad);

                std::vector<std::uint8_t> scattered(count, 0U);
                expect(eq(reference.deinterleave(codewords, scattered), count));
                expect(that % std::ranges::equal(scattered, codeblock)) << std::format("depth {}, pad {}: and the interleave is its inverse", depth, pad);
            }
        }

        expect(that % !ccsdsInterleaveDepthAllowed(6UZ)) << "4.3.5.1 omits six";
        expect(that % !ccsdsInterleaveDepthAllowed(7UZ)) << "and seven";
        expect(that % !ccsdsInterleaveDepthAllowed(0UZ));
        expect(that % !ccsdsInterleaveDepthAllowed(9UZ));

        // a burst of E * I consecutive codeblock symbols is one error per codeword per E, which is the
        // reason the standard has interleaving at all, stated as the count it is
        constexpr std::size_t     kDepth = 5UZ;
        std::vector<std::uint8_t> codewords(255UZ * kDepth, 0U);
        std::vector<std::uint8_t> codeblock(codewords.size(), 0U);
        interleaveCodewords(codewords, codeblock, 255UZ, kDepth);
        for (std::size_t i = 0UZ; i < 16UZ * kDepth; ++i) {
            codeblock[i] = 0xFFU;
        }
        deinterleaveCodewords(codeblock, codewords, 255UZ, kDepth);
        for (std::size_t word = 0UZ; word < kDepth; ++word) {
            std::size_t hit = 0UZ;
            for (std::size_t i = 0UZ; i < 255UZ; ++i) {
                hit += codewords[word * 255UZ + i] != 0U ? 1UZ : 0UZ;
            }
            expect(eq(hit, 16UZ)) << std::format("codeword {}: a burst of 80 consecutive symbols is 16 per codeword, exactly the radius", word);
        }
    };
};

int main() { /* tests are automatically registered and run */ }
