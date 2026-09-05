#include <boost/ut.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

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

using gr::fec::ReedSolomon6;
using gr::fec::RsResult;

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
};

int main() { /* tests are automatically registered and run */ }
