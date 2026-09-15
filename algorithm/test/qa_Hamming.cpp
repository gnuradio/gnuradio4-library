#include <boost/ut.hpp>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

#include <gnuradio-4.0/algorithm/fec/Hamming.hpp>

/*
 * Both shortened Hamming codes are described entirely by their lists of parity columns, so
 * these tests pin the columns and then pin what the columns imply.
 *
 * The two codes behave differently at their limit and the difference is the point.
 * Hamming(15,11,3) is perfect: its fifteen nonzero syndromes name its fifteen bit positions,
 * nothing is left over, and two errors are therefore turned silently into some other single
 * error. Hamming(10,6,3) uses only ten of those fifteen syndromes, so the other five are
 * proof of two or more errors and are reported. Both facts are asserted here by enumeration
 * rather than assumed, because a column list that drifted would change which of them holds.
 */
namespace {

using gr::fec::hamming1063Decode;
using gr::fec::hamming1063Encode;
using gr::fec::hamming1511Decode;
using gr::fec::hamming1511Encode;
using gr::fec::HammingResult;

/// The syndrome a single error at `position` produces, counting the first transmitted bit as
/// position 0. Derived from the code's own encoder rather than from its column list, so the
/// two have to agree for anything below to pass.
template<typename Encode>
std::uint8_t syndromeOfSingleError(Encode encode, std::uint16_t info, std::size_t infoBits, std::size_t position) {
    const std::uint16_t cw   = encode(info);
    const std::uint16_t bad  = static_cast<std::uint16_t>(cw ^ (1U << ((infoBits + 4U) - 1U - position)));
    const std::uint16_t data = static_cast<std::uint16_t>((bad >> 4) & ((1U << infoBits) - 1U));
    return static_cast<std::uint8_t>((encode(data) ^ bad) & 0x0FU);
}

} // namespace

const boost::ut::suite<"hamming"> hammingTests = [] {
    using namespace boost::ut;

    "codewords P25 equipment transmits"_test = [] {
        expect(eq(hamming1511Encode(0x001U), 0x0013U)) << "the (15,11) codeword for the least significant information bit";
        expect(eq(hamming1511Encode(0x400U), 0x400FU)) << "the (15,11) codeword for the most significant information bit";
        expect(eq(hamming1063Encode(0x01U), 0x001CU)) << "the (10,6) codeword for the least significant information bit";
        expect(eq(hamming1063Encode(0x20U), 0x020EU)) << "the (10,6) codeword for the most significant information bit";
        expect(eq(hamming1063Encode(0x3FU), 0x03F0U)) << "the (10,6) codeword whose columns cancel exactly";
    };

    "the (15,11) columns are every four-bit pattern of weight two or more"_test = [] {
        std::array<bool, 16U> seen{};
        for (std::size_t i = 0U; i < 11U; ++i) {
            const std::uint8_t s = syndromeOfSingleError(hamming1511Encode, 0U, 11U, i);
            expect(ge(std::popcount(static_cast<unsigned>(s)), 2)) << "an information bit's column has weight two or more";
            expect(that % !seen[s]) << "no two information bits share a column";
            seen[s] = true;
        }
        for (unsigned v = 0U; v < 16U; ++v) {
            const bool heavy = std::popcount(v) >= 2;
            expect(eq(seen[v], heavy)) << "the columns are exactly the patterns of weight two or more";
        }
    };

    "every single error is corrected, over every word, in both codes"_test = [] {
        int wrong = 0;
        for (unsigned info = 0U; info < 2048U; ++info) {
            const std::uint16_t cw = hamming1511Encode(static_cast<std::uint16_t>(info));
            for (int b = -1; b < 15; ++b) {
                const std::uint16_t received = static_cast<std::uint16_t>(cw ^ ((b < 0) ? 0U : (1U << b)));
                const HammingResult r        = hamming1511Decode(received);
                if (!r.valid || r.info != info || r.errors != ((b < 0) ? 0U : 1U)) {
                    ++wrong;
                }
            }
        }
        expect(eq(wrong, 0)) << "Hamming(15,11,3) corrects every single error exactly";

        wrong = 0;
        for (unsigned info = 0U; info < 64U; ++info) {
            const std::uint16_t cw = hamming1063Encode(static_cast<std::uint8_t>(info));
            expect(eq(static_cast<unsigned>(cw >> 10), 0U)) << "a (10,6) codeword fits in 10 bits";
            for (int b = -1; b < 10; ++b) {
                const std::uint16_t received = static_cast<std::uint16_t>(cw ^ ((b < 0) ? 0U : (1U << b)));
                const HammingResult r        = hamming1063Decode(received);
                if (!r.valid || r.info != info || r.errors != ((b < 0) ? 0U : 1U)) {
                    ++wrong;
                }
            }
        }
        expect(eq(wrong, 0)) << "Hamming(10,6,3) corrects every single error exactly";
    };

    "the perfect code cannot say no; the shortened one can, on exactly five syndromes"_test = [] {
        long accepted1511 = 0;
        long total1511    = 0;
        for (unsigned info = 0U; info < 2048U; ++info) {
            const std::uint16_t cw = hamming1511Encode(static_cast<std::uint16_t>(info));
            for (int a = 0; a < 15; ++a) {
                for (int b = a + 1; b < 15; ++b) {
                    ++total1511;
                    if (hamming1511Decode(static_cast<std::uint16_t>(cw ^ (1U << a) ^ (1U << b))).valid) {
                        ++accepted1511;
                    }
                }
            }
        }
        expect(eq(total1511, accepted1511)) << "Hamming(15,11,3) accepts every double error, being perfect and having no syndrome left to refuse with";

        // Which syndromes the (10,6) code refuses, taken from the decoder itself: a received
        // word is built for each syndrome by starting from a codeword and forcing the parity.
        std::array<bool, 16U> refused{};
        for (unsigned s = 0U; s < 16U; ++s) {
            const std::uint16_t cw       = hamming1063Encode(0x15U);
            const std::uint16_t received = static_cast<std::uint16_t>((cw & 0x3F0U) | ((cw ^ s) & 0x0FU));
            refused[s]                   = !hamming1063Decode(received).valid;
        }
        expect(that % (!refused[0x0U] && !refused[0x1U] && !refused[0x2U] && !refused[0x4U] && !refused[0x8U])) << "a clean word and the four single parity errors are accepted";
        expect(that % (refused[0x5U] && refused[0x6U] && refused[0x9U] && refused[0xAU] && refused[0xFU])) << "the five syndromes naming no position in the shortened code are refused";
        unsigned refusedCount = 0U;
        for (const bool r : refused) {
            if (r) {
                ++refusedCount;
            }
        }
        expect(eq(refusedCount, 5U)) << "and exactly five of the sixteen syndromes are refused";
    };

    "double errors in the shortened code: how many are caught, absolutely"_test = [] {
        long total         = 0;
        long caught        = 0;
        long silentlyWrong = 0;
        for (unsigned info = 0U; info < 64U; ++info) {
            const std::uint16_t cw = hamming1063Encode(static_cast<std::uint8_t>(info));
            for (int a = 0; a < 10; ++a) {
                for (int b = a + 1; b < 10; ++b) {
                    const HammingResult r = hamming1063Decode(static_cast<std::uint16_t>(cw ^ (1U << a) ^ (1U << b)));
                    ++total;
                    if (!r.valid) {
                        ++caught;
                    } else if (r.info != info) {
                        ++silentlyWrong;
                    }
                }
            }
        }
        // 45 pairs over each of 64 words. Twenty-one of the pairs land on a syndrome this
        // shortened code has no position for and are reported; the other twenty-four are
        // turned into some other single error. Both counts are fixed by the column list and
        // by nothing else, which is what makes them worth asserting exactly.
        expect(eq(total, 2880L)) << "the double-error sweep covers every pair over every word";
        expect(eq(caught, 1344L)) << "Hamming(10,6,3) reports 21 of the 45 double errors";
        expect(eq(silentlyWrong, 1536L)) << "and mis-decodes the remaining 24";
        expect(eq(caught + silentlyWrong, total)) << "no double error comes back quietly correct: four of them disturb only parity bits and leave the "
                                                     "information intact, but a distance-3 code cannot tell that case from a heavier one and reports it";
    };
};

int main() { /* tests are automatically registered and run */ }
