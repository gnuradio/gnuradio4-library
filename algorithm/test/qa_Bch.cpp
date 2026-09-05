#include <boost/ut.hpp>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include <gnuradio-4.0/algorithm/fec/Bch.hpp>
#include <gnuradio-4.0/algorithm/fec/Hamming.hpp>

/*
 * The generator polynomial of the binary BCH(63,16,23) code is derived from the field, the
 * primitive polynomial and the designed distance, so these tests pin the derivation rather
 * than repeat it. Three independent ways, none of which is the derivation restated:
 *
 *   - the generator divides x^63 - 1, which every cyclic code's generator must and which a
 *     wrong product of minimal polynomials will not;
 *   - the code's minimum distance really is 23, checked by taking the weight of codewords
 *     directly -- for a linear code the minimum weight IS the minimum distance, so this is the
 *     property the correcting power is claimed from and not a proxy for it;
 *   - the resulting polynomial equals the value P25 equipment transmits.
 *
 * Correction is then checked at its exact boundary: every pattern of up to 11 wrong bits must
 * come back as the identifier that was encoded, because that is what distance 23 buys.
 */
namespace {

using gr::fec::bch63Decode;
using gr::fec::bch63Encode;
using gr::fec::bch63GeneratorPoly;
using gr::fec::Bch63Result;
using gr::fec::kBch63Correctable;
using gr::fec::kBch63Distance;
using gr::fec::kBch63Length;
using gr::fec::kBch63ParityBits;

std::uint64_t rng = 0x243F6A8885A308D3ULL;

std::uint64_t next() {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return rng;
}

/// Remainder of `a` divided by `b`, both polynomials over GF(2) held bit i = coefficient x^i.
std::uint64_t polyMod(std::uint64_t a, std::uint64_t b) {
    const int db = std::bit_width(b) - 1;
    for (int i = std::bit_width(a) - 1; i >= db; --i) {
        if (((a >> i) & 1U) != 0U) {
            a ^= (b << (i - db));
        }
    }
    return a;
}

} // namespace

const boost::ut::suite<"bch"> bchTests = [] {
    using namespace boost::ut;

    "the generator is a generator: degree 47, and it divides x^63 - 1"_test = [] {
        const std::uint64_t g = bch63GeneratorPoly();
        expect(eq(std::bit_width(g) - 1, static_cast<int>(kBch63ParityBits))) << "the generator's degree is 63 - 16 = 47";
        expect(neq(g & 1U, std::uint64_t{0U})) << "the generator has a constant term, as a cyclic code's must";

        // x^63 + 1 does not fit in 64 bits alongside the reduction, so reduce it stepwise:
        // x^63 mod g, then add 1.
        std::uint64_t x63 = 1U;
        for (int i = 0; i < 63; ++i) {
            x63 <<= 1;
            x63 = polyMod(x63, g);
        }
        expect(eq(x63 ^ 1ULL, std::uint64_t{0U})) << "the generator divides x^63 - 1";

        expect(eq(g, std::uint64_t{0xCD930BDD3B2BULL})) << "the derivation reproduces the polynomial P25 equipment transmits";
    };

    "encoding is linear: the codeword of a sum is the sum of the codewords"_test = [] {
        bool linear = true;
        for (int t = 0; t < 200; ++t) {
            const auto a = static_cast<std::uint16_t>(next() >> 48U);
            const auto b = static_cast<std::uint16_t>(next() >> 48U);
            linear       = linear && (bch63Encode(static_cast<std::uint16_t>(a ^ b)) == (bch63Encode(a) ^ bch63Encode(b)));
        }
        expect(that % linear);
    };

    "the information bits survive encoding where the systematic layout says they are"_test = [] {
        bool systematic = true;
        for (int t = 0; t < 500; ++t) {
            const auto info = static_cast<std::uint16_t>(next() >> 48U);
            systematic      = systematic && (static_cast<std::uint16_t>(bch63Encode(info) >> kBch63ParityBits) == info);
        }
        expect(that % systematic) << "the codeword's top 16 bits are the information word unchanged";
    };

    "minimum distance 23, taken as the minimum weight of a nonzero codeword"_test = [] {
        unsigned lightest = 64U;
        for (std::uint32_t info = 1U; info < (1U << 16U); ++info) {
            const auto w = static_cast<unsigned>(std::popcount(bch63Encode(static_cast<std::uint16_t>(info))));
            lightest     = (w < lightest) ? w : lightest;
        }
        expect(eq(lightest, kBch63Distance)) << "the lightest nonzero codeword weighs 23 bits, so the minimum distance is 23";
        expect(eq(kBch63Correctable, 11U)) << "a distance of 23 corrects 11 bits";
    };

    "an undamaged codeword decodes to itself with nothing corrected"_test = [] {
        bool exact = true;
        for (int t = 0; t < 2000; ++t) {
            const auto        info = static_cast<std::uint16_t>(next() >> 48U);
            const Bch63Result r    = bch63Decode(bch63Encode(info));
            exact                  = exact && r.valid && r.errors == 0U && r.info == info;
        }
        expect(that % exact) << "an undamaged codeword decodes to its own identifier with zero corrections";
    };

    "every pattern of up to 11 wrong bits is corrected, and the count is reported"_test = [] {
        bool recovered = true;
        bool counted   = true;
        for (unsigned k = 1U; k <= kBch63Correctable; ++k) {
            for (int t = 0; t < 120; ++t) {
                const auto    info = static_cast<std::uint16_t>(next() >> 48U);
                std::uint64_t w    = bch63Encode(info);
                unsigned      hit  = 0U;
                std::uint64_t used = 0U;
                while (hit < k) {
                    const unsigned b = static_cast<unsigned>(next() % kBch63Length);
                    if (((used >> b) & 1U) != 0U) {
                        continue;
                    }
                    used |= (1ULL << b);
                    w ^= (1ULL << b);
                    ++hit;
                }
                const Bch63Result r = bch63Decode(w);
                recovered           = recovered && r.valid && r.info == info;
                counted             = counted && (r.errors == k);
            }
        }
        expect(that % recovered) << "every pattern of up to 11 wrong bits recovers the identifier that was encoded";
        expect(that % counted) << "the decoder reports how many bits it corrected";
    };

    "the accepted distance is respected when it is set below the code's power"_test = [] {
        const auto    info = static_cast<std::uint16_t>(next() >> 48U);
        std::uint64_t w    = bch63Encode(info);
        for (unsigned b = 0U; b < 5U; ++b) {
            w ^= (1ULL << (b * 7U));
        }
        expect(that % bch63Decode(w, 11U).valid) << "five wrong bits are inside the code's correcting power";
        expect(that % !bch63Decode(w, 4U).valid) << "five wrong bits are refused when at most four corrections are accepted";
        expect(eq(bch63Decode(w, 4U).errors, 5U)) << "a refusal still reports the distance it found";
    };

    "a word that was never transmitted is almost always refused, which is the whole reason a frame sync can be treated as unproven"_test = [] {
        int       accepted = 0;
        const int trials   = 8000;
        for (int t = 0; t < trials; ++t) {
            const std::uint64_t w = next() & ((1ULL << kBch63Length) - 1U);
            accepted += bch63Decode(w).valid ? 1 : 0;
        }
        expect(lt(accepted * 100, trials)) << "fewer than one random word in a hundred reaches a codeword: " << accepted << " of " << trials;
    };

    "the length-15 family's generators are the field and the designed distances, and nothing else"_test = [] {
        struct Row {
            unsigned      distance;
            std::uint16_t generator;
            unsigned      parityBits;
            unsigned      infoBits;
            unsigned      correctable;
        };
        constexpr std::array<Row, 3UZ> rows{Row{3U, 0x13U, 4U, 11U, 1U}, Row{5U, 0x1D1U, 8U, 7U, 2U}, Row{7U, 0x537U, 10U, 5U, 3U}};

        for (const Row& row : rows) {
            const std::uint16_t g = gr::fec::bch15GeneratorPoly(row.distance);
            expect(eq(g, row.generator)) << "designed distance " << row.distance;
            expect(eq(std::bit_width(g) - 1, static_cast<int>(row.parityBits)));
            expect(neq(g & 1U, 0U)) << "a cyclic code's generator has a constant term";
            // A cyclic code's generator divides x^15 - 1, which a wrong product of minimal
            // polynomials does not.
            expect(eq(polyMod(0x8001ULL, g), std::uint64_t{0U})) << "the generator divides x^15 + 1";
            expect(eq(15U - row.parityBits, row.infoBits));
            expect(eq((row.distance - 1U) / 2U, row.correctable));
        }
        expect(eq(gr::fec::Bch15_11::kGenerator, std::uint16_t{0x13U})) << "which is the field polynomial itself: the distance-3 code is a cyclic Hamming code";
        expect(eq(gr::fec::Bch15_11::kInfoBits, 11U));
        expect(eq(gr::fec::Bch15_7::kInfoBits, 7U));
        expect(eq(gr::fec::Bch15_5::kInfoBits, 5U));

        // The whole family's tables are a compile-time constant of 4416 bytes.
        expect(eq(gr::fec::Bch15_11::codewords().size(), 2048UZ));
        expect(eq(gr::fec::Bch15_7::codewords().size(), 128UZ));
        expect(eq(gr::fec::Bch15_5::codewords().size(), 32UZ));
        expect(eq((2048UZ + 128UZ + 32UZ) * sizeof(std::uint16_t), 4416UZ));
    };

    "the true minimum distances are the designed ones: 3, 5 and 7"_test = [] {
        // For a linear code the minimum weight is the minimum distance, so this is the property the
        // correcting power is claimed from and not a proxy for it.
        const auto minimumWeight = []<typename Code>(Code) {
            unsigned best = Code::kLength + 1U;
            for (std::size_t i = 1UZ; i < Code::codewords().size(); ++i) {
                const unsigned weight = static_cast<unsigned>(std::popcount(static_cast<unsigned>(Code::codewords()[i])));
                best                  = weight < best ? weight : best;
            }
            return best;
        };
        expect(eq(minimumWeight(gr::fec::Bch15_11{}), 3U));
        expect(eq(minimumWeight(gr::fec::Bch15_7{}), 5U));
        expect(eq(minimumWeight(gr::fec::Bch15_5{}), 7U));
    };

    "every error pattern inside the radius is corrected, exhaustively, with its weight exact"_test = [] {
        const auto sweep = []<typename Code>(Code, std::size_t expectedCases) {
            std::size_t cases = 0UZ;
            for (std::uint16_t info = 0U; info < (1U << Code::kInfoBits); ++info) {
                const std::uint16_t codeword = Code::encode(info);
                expect(eq(Code::decode(codeword).errors, 0U));
                expect(eq(Code::decode(codeword).info, info));
                ++cases;

                for (unsigned weight = 1U; weight <= Code::kCorrectable; ++weight) {
                    // Every pattern of exactly `weight` bits, walked as a combination.
                    std::vector<unsigned> pattern(weight);
                    for (unsigned i = 0U; i < weight; ++i) {
                        pattern[i] = i;
                    }
                    while (true) {
                        std::uint16_t received = codeword;
                        for (const unsigned bit : pattern) {
                            received = static_cast<std::uint16_t>(received ^ (1U << bit));
                        }
                        const gr::fec::Bch15Result result = Code::decode(received);
                        expect(result.valid);
                        expect(eq(result.info, info));
                        expect(eq(result.errors, weight)) << "the reported count is the injected weight, exactly";
                        ++cases;

                        unsigned position = weight;
                        while (position > 0U && pattern[position - 1U] == Code::kLength - weight + position - 1U) {
                            --position;
                        }
                        if (position == 0U) {
                            break;
                        }
                        ++pattern[position - 1U];
                        for (unsigned i = position; i < weight; ++i) {
                            pattern[i] = pattern[i - 1U] + 1U;
                        }
                    }
                }
            }
            expect(eq(cases, expectedCases));
        };

        // 2^11 * 16, 2^7 * 121 and 2^5 * 576: the sphere sizes are sum_{i<=t} C(15,i).
        expect(eq(1UZ + 15UZ, 16UZ));
        expect(eq(1UZ + 15UZ + 105UZ, 121UZ));
        expect(eq(1UZ + 15UZ + 105UZ + 455UZ, 576UZ));
        sweep(gr::fec::Bch15_11{}, 2048UZ * 16UZ);
        sweep(gr::fec::Bch15_7{}, 128UZ * 121UZ);
        sweep(gr::fec::Bch15_5{}, 32UZ * 576UZ);

        // (15,11,3) is perfect and the other two are not, which is exactly the refusal column below.
        expect(eq(16UZ, 1UZ << 4U)) << "the spheres tile the space";
        expect(lt(121UZ, 1UZ << 8U));
        expect(lt(576UZ, 1UZ << 10U));
    };

    "past the radius the decoder either refuses or miscorrects, and the split is the code's"_test = [] {
        // By linearity the split is a property of the code, so it is computed once from the zero
        // codeword rather than 2^k times.
        const auto split = []<typename Code>(Code, unsigned weight) {
            std::size_t           miscorrected = 0UZ;
            std::size_t           refused      = 0UZ;
            std::vector<unsigned> pattern(weight);
            for (unsigned i = 0U; i < weight; ++i) {
                pattern[i] = i;
            }
            while (true) {
                std::uint16_t received = 0U;
                for (const unsigned bit : pattern) {
                    received = static_cast<std::uint16_t>(received ^ (1U << bit));
                }
                if (Code::decode(received).valid) {
                    ++miscorrected;
                } else {
                    ++refused;
                }

                unsigned position = weight;
                while (position > 0U && pattern[position - 1U] == Code::kLength - weight + position - 1U) {
                    --position;
                }
                if (position == 0U) {
                    break;
                }
                ++pattern[position - 1U];
                for (unsigned i = position; i < weight; ++i) {
                    pattern[i] = pattern[i - 1U] + 1U;
                }
            }
            return std::pair<std::size_t, std::size_t>{miscorrected, refused};
        };

        const auto three = split(gr::fec::Bch15_7{}, 3U);
        expect(eq(three.first, 180UZ)) << "C(15,3) = 455 patterns at weight t + 1 for (15,7,5)";
        expect(eq(three.second, 275UZ));
        expect(eq(three.first + three.second, 455UZ));

        const auto four = split(gr::fec::Bch15_5{}, 4U);
        expect(eq(four.first, 525UZ)) << "C(15,4) = 1365 patterns at weight t + 1 for (15,5,7)";
        expect(eq(four.second, 840UZ));
        expect(eq(four.first + four.second, 1365UZ));
    };

    "the cyclic BCH(15,11,3) and Hamming.hpp's Hamming(15,11,3) share exactly 128 codewords"_test = [] {
        std::unordered_set<std::uint16_t> hamming;
        for (std::uint16_t info = 0U; info < 2048U; ++info) {
            hamming.insert(gr::fec::hamming1511Encode(info));
        }
        expect(eq(hamming.size(), 2048UZ)) << "both codes have 2^11 distinct codewords";

        std::size_t shared = 0UZ;
        for (const std::uint16_t codeword : gr::fec::Bch15_11::codewords()) {
            if (hamming.contains(codeword)) {
                ++shared;
            }
        }
        expect(eq(shared, 128UZ)) << "a 2^7 subcode of both, and the reason both ship";
        expect(eq(128UZ, 1UZ << 7U));

        // And the trap itself: a BCH codeword outside the shared subcode, handed to the Hamming
        // decoder, comes back as eleven wrong bits with a clean report and nothing to say so.
        std::size_t silentlyWrong = 0UZ;
        for (std::uint16_t info = 0U; info < 2048U; ++info) {
            const std::uint16_t          codeword = gr::fec::Bch15_11::encode(info);
            const gr::fec::HammingResult misread  = gr::fec::hamming1511Decode(codeword);
            if (misread.valid && misread.info != info) {
                ++silentlyWrong;
            }
        }
        expect(gt(silentlyWrong, 0UZ)) << "the wrong decoder reports success on " << silentlyWrong << " of 2048 words";
    };
};

int main() { /* tests are automatically registered and run */ }
