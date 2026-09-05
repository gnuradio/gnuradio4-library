#include <boost/ut.hpp>

#include <algorithm>
#include <bit>
#include <cstdint>

#include <gnuradio-4.0/algorithm/fec/Golay.hpp>

/*
 * The decoder's table is built from the generator polynomial rather than carried, so these
 * tests pin the construction rather than repeat it. Four ways, none of which is the
 * derivation restated:
 *
 *   - the coset table is a bijection between the 2048 syndromes and the 2048 error patterns
 *     of weight three or less. That is what "perfect" means for this code, and a table built
 *     from a wrong generator leaves slots empty and writes others twice;
 *   - the generator divides x^23 - 1, which every cyclic code's generator must;
 *   - the lightest nonzero codeword weighs exactly 7 over 23 bits and exactly 8 over 24,
 *     taken by enumerating all 4096 codewords. For a linear code the minimum weight IS the
 *     minimum distance, so this is the property the correcting power is claimed from rather
 *     than a proxy for it;
 *   - two codewords equal what P25 equipment transmits.
 *
 * Correction is then checked at its exact boundary and past it. Every pattern of up to three
 * wrong bits must come back as the word that was encoded, with the right count. Every pattern
 * of four must be reported -- that is the whole reason the extended codes carry an overall
 * parity bit, and a decoder that quietly returns a neighboring codeword instead is the defect
 * these tests exist to catch.
 */
namespace {

using gr::fec::golay18Decode;
using gr::fec::golay18Encode;
using gr::fec::golay23Decode;
using gr::fec::golay23Encode;
using gr::fec::golay24Decode;
using gr::fec::golay24Encode;
using gr::fec::golayCosetTableIsBijective;
using gr::fec::GolayResult;
using gr::fec::kGolayCorrectable;
using gr::fec::kGolayGenerator;
using gr::fec::kGolayParityBits;

} // namespace

const boost::ut::suite<"golay"> golayTests = [] {
    using namespace boost::ut;

    "the coset table is the bijection a perfect code demands"_test = [] { expect(that % golayCosetTableIsBijective()) << "every syndrome names exactly one error pattern of weight <= 3"; };

    "the generator divides x^23 - 1"_test = [] {
        // Reduce x^23 + 1 by the generator using plain polynomial division over GF(2), held as
        // bits 23 and 0 of a 24-bit remainder accumulator.
        std::uint32_t poly = (1U << 23) | 1U;
        for (int bit = 23; bit >= static_cast<int>(kGolayParityBits); --bit) {
            if ((poly >> bit) & 1U) {
                poly ^= kGolayGenerator << (bit - static_cast<int>(kGolayParityBits));
            }
        }
        expect(eq(poly, 0U)) << "the generator divides x^23 - 1";
    };

    "the minimum weight of the code, enumerated"_test = [] {
        unsigned lightest23 = 99U;
        unsigned lightest24 = 99U;
        for (unsigned info = 1U; info < 4096U; ++info) {
            const unsigned w23 = static_cast<unsigned>(std::popcount(golay23Encode(static_cast<std::uint16_t>(info))));
            const unsigned w24 = static_cast<unsigned>(std::popcount(golay24Encode(static_cast<std::uint16_t>(info))));
            lightest23         = std::min(lightest23, w23);
            lightest24         = std::min(lightest24, w24);
        }
        expect(eq(lightest23, 7U)) << "the lightest nonzero Golay(23,12) codeword weighs 7";
        expect(eq(lightest24, 8U)) << "the lightest nonzero Golay(24,12) codeword weighs 8";
    };

    "codewords P25 equipment transmits"_test = [] {
        expect(eq(golay24Encode(0x001U), 0x0018EBU)) << "the codeword for the least significant information bit";
        expect(eq(golay24Encode(0x800U), 0x800C75U)) << "the codeword for the most significant information bit";
        expect(eq(golay23Encode(0x001U), 0x000C75U)) << "the unextended codeword is the extended one without its parity";
    };

    "every pattern of weight <= 3 is corrected exactly, over every codeword"_test = [] {
        long       checked = 0;
        int        wrong   = 0;
        const auto trial   = [&checked, &wrong](std::uint32_t cw, unsigned info, std::uint32_t e) {
            const unsigned    weight = static_cast<unsigned>(std::popcount(e));
            const GolayResult r      = golay24Decode(cw ^ e);
            ++checked;
            if (!r.valid || r.info != info || r.errors != weight) {
                ++wrong;
            }
        };
        for (unsigned info = 0U; info < 4096U; ++info) {
            const std::uint32_t cw = golay24Encode(static_cast<std::uint16_t>(info));
            trial(cw, info, 0U);
            for (int a = 0; a < 24; ++a) {
                trial(cw, info, 1U << a);
                for (int b = a + 1; b < 24; ++b) {
                    trial(cw, info, (1U << a) | (1U << b));
                    for (int c = b + 1; c < 24; ++c) {
                        trial(cw, info, (1U << a) | (1U << b) | (1U << c));
                    }
                }
            }
        }
        // 1 + 24 + 276 + 2024 patterns over each of 4096 codewords.
        expect(eq(checked, 4096L * 2325L)) << "the sweep covered every codeword and every pattern of weight <= 3";
        expect(eq(wrong, 0)) << "Golay(24,12,8) corrects every pattern of weight <= 3 exactly";
    };

    "every pattern of weight 4 is reported, never silently mis-decoded"_test = [] {
        long total = 0;
        long quiet = 0;
        for (unsigned info = 0U; info < 128U; ++info) {
            const std::uint32_t cw = golay24Encode(static_cast<std::uint16_t>(info));
            for (int a = 0; a < 24; ++a) {
                for (int b = a + 1; b < 24; ++b) {
                    for (int c = b + 1; c < 24; ++c) {
                        for (int d = c + 1; d < 24; ++d) {
                            const std::uint32_t e = (1U << a) | (1U << b) | (1U << c) | (1U << d);
                            ++total;
                            if (golay24Decode(cw ^ e).valid) {
                                ++quiet;
                            }
                        }
                    }
                }
            }
        }
        expect(gt(total, 1000000L)) << "the weight-4 sweep was large enough to mean something";
        expect(eq(quiet, 0L)) << "Golay(24,12,8) reports every pattern of weight 4 rather than inventing an answer";
    };

    "the unextended code cannot say no, which is why the extended one is used"_test = [] {
        // Four errors on a (23,12) codeword decode to some codeword with no complaint. The
        // point of the check is that the decoder does not pretend otherwise.
        const std::uint32_t cw = golay23Encode(0x5A5U);
        const std::uint32_t e  = (1U << 0) | (1U << 5) | (1U << 11) | (1U << 19);
        const GolayResult   r  = golay23Decode(cw ^ e);
        expect(that % r.valid) << "Golay(23,12,7) always reports success, having no way to detect a fourth error";
        expect(neq(r.info, 0x5A5U)) << "and four errors do take it to a different word";
    };

    "the shortened code corrects every pattern of weight <= 3, over every one of its 64 words"_test = [] {
        long       checked = 0;
        int        wrong   = 0;
        const auto trial   = [&checked, &wrong](std::uint32_t cw, unsigned info, std::uint32_t e) {
            const unsigned    weight = static_cast<unsigned>(std::popcount(e));
            const GolayResult r      = golay18Decode(cw ^ e);
            ++checked;
            if (!r.valid || r.info != info || r.errors != weight) {
                ++wrong;
            }
        };
        for (unsigned info = 0U; info < 64U; ++info) {
            const std::uint32_t cw = golay18Encode(static_cast<std::uint8_t>(info));
            expect(eq(cw >> 18, 0U)) << "a shortened codeword fits in 18 bits";
            trial(cw, info, 0U);
            for (int a = 0; a < 18; ++a) {
                trial(cw, info, 1U << a);
                for (int b = a + 1; b < 18; ++b) {
                    trial(cw, info, (1U << a) | (1U << b));
                    for (int c = b + 1; c < 18; ++c) {
                        trial(cw, info, (1U << a) | (1U << b) | (1U << c));
                    }
                }
            }
        }
        // 1 + 18 + 153 + 816 patterns over each of the code's 64 words.
        expect(eq(checked, 64L * 988L)) << "the shortened sweep covered every word and every pattern of weight <= 3";
        expect(eq(wrong, 0)) << "Golay(18,6,8) corrects every pattern of weight <= 3 exactly";
    };

    "the shortened code reports every pattern of weight 4"_test = [] {
        long total = 0;
        long quiet = 0;
        for (unsigned info = 0U; info < 64U; ++info) {
            const std::uint32_t cw = golay18Encode(static_cast<std::uint8_t>(info));
            for (int a = 0; a < 18; ++a) {
                for (int b = a + 1; b < 18; ++b) {
                    for (int c = b + 1; c < 18; ++c) {
                        for (int d = c + 1; d < 18; ++d) {
                            ++total;
                            if (golay18Decode(cw ^ (1U << a) ^ (1U << b) ^ (1U << c) ^ (1U << d)).valid) {
                                ++quiet;
                            }
                        }
                    }
                }
            }
        }
        expect(gt(total, 190000L)) << "the shortened weight-4 sweep was large enough to mean something";
        expect(eq(quiet, 0L)) << "Golay(18,6,8) reports every pattern of weight 4";
    };

    "a word whose nearest codeword lies outside the shortened code is refused"_test = [] {
        // The extended parity catches every fourth error, so it alone covers the shortened
        // code out to its distance. What it does not cover is a word that decodes cleanly to a
        // codeword the shortening cannot contain. Build one: take a codeword whose information
        // uses the six bits the shortening holds at zero, keep only the 18 bits the air would
        // carry, and hand that over. The extended decoder finds its way back to the full
        // codeword and is perfectly happy; only the shortening's own constraint says no.
        long provoked = 0;
        for (unsigned info = 0x040U; info < 0x1000U; ++info) {
            if ((info & 0x0FC0U) == 0U) {
                continue; // this word is inside the shortened code
            }
            const std::uint32_t cw = golay24Encode(static_cast<std::uint16_t>(info));
            if (std::popcount(cw >> 18) > static_cast<int>(kGolayCorrectable)) {
                continue; // too far from any 18-bit word for the decoder to reach it
            }
            const std::uint32_t received = cw & 0x3FFFFU;
            const GolayResult   extended = golay24Decode(received);
            if (!extended.valid || extended.info != info) {
                continue;
            }
            ++provoked;
            expect(that % !golay18Decode(received).valid) << "a word decoding to a codeword the shortening cannot contain is refused";
        }
        expect(gt(provoked, 100L)) << "the construction really did produce words the extended decoder accepts";
    };
};

int main() { /* tests are automatically registered and run */ }
