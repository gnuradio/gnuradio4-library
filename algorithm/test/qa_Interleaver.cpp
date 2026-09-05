#include <boost/ut.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <numeric>
#include <print>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include <gnuradio-4.0/algorithm/fec/Interleaver.hpp>
#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>

using namespace boost::ut;
using gr::fec::Interleaver;
using gr::fec::InterleaverKind;

namespace {

[[nodiscard]] std::vector<std::uint8_t> seededBytes(std::size_t n, std::uint64_t seed) {
    gr::rng::Xoshiro256pp     rng(seed);
    std::vector<std::uint8_t> data(n);
    for (std::uint8_t& value : data) {
        value = static_cast<std::uint8_t>(rng() & 0xffULL);
    }
    return data;
}

/// A seeded permutation of `n`, built by a Fisher-Yates shuffle so that the table is a real permutation and the
/// kernel's validation is being handed something it must accept rather than something it must reject.
[[nodiscard]] std::vector<std::size_t> seededPermutation(std::size_t n, std::uint64_t seed) {
    gr::rng::Xoshiro256pp    rng(seed);
    std::vector<std::size_t> table(n);
    std::iota(table.begin(), table.end(), 0UZ);
    for (std::size_t i = n; i > 1UZ; --i) {
        const std::size_t j = std::uniform_int_distribution<std::size_t>{0UZ, i - 1UZ}(rng);
        std::swap(table[i - 1UZ], table[j]);
    }
    return table;
}

/// Where each input item of a framed interleaver lands, read off the kernel's own gather: `map[i]` is the input
/// index that output position `i` takes, so this inverts it.
[[nodiscard]] std::vector<std::size_t> outputPositions(const Interleaver<std::uint8_t>& interleaver) {
    std::vector<std::size_t> position(interleaver.frameSize(), std::numeric_limits<std::size_t>::max());
    const auto               map = interleaver.indexMap();
    for (std::size_t i = 0UZ; i < map.size(); ++i) {
        position[map[i]] = i;
    }
    return position;
}

/// The convolutional index map, recovered from the kernel rather than asserted about in prose: a ramp of
/// distinct non-zero values goes in, and the position each value comes out at is where that input index went.
/// `direction` picks which of the two commutator sets is being read.
enum class Direction : std::uint8_t { Interleave, Deinterleave };

[[nodiscard]] std::vector<std::size_t> convolutionalPositions(std::size_t branches, std::size_t unitDelay, std::size_t nItems, Direction direction) {
    Interleaver<float> kernel = Interleaver<float>::convolutional(branches, unitDelay, 0.f);

    const std::size_t  span = nItems + kernel.latency() + branches * branches * unitDelay;
    std::vector<float> in(span, 0.f);
    for (std::size_t k = 0UZ; k < nItems; ++k) {
        in[k] = static_cast<float>(k + 1UZ); // 0 is the fill value, so a real item is never mistaken for one
    }
    std::vector<float> out(span, 0.f);
    if (direction == Direction::Interleave) {
        (void)kernel.interleave(in, out);
    } else {
        (void)kernel.deinterleave(in, out);
    }

    std::vector<std::size_t> position(nItems, std::numeric_limits<std::size_t>::max());
    for (std::size_t p = 0UZ; p < span; ++p) {
        if (out[p] != 0.f) {
            const std::size_t index = static_cast<std::size_t>(out[p]) - 1UZ;
            if (index < nItems) {
                position[index] = p;
            }
        }
    }
    return position;
}

/// The closest two members of a burst come to each other once the interleaver has moved them.
[[nodiscard]] std::size_t minimumSeparation(std::span<const std::size_t> position, std::size_t first, std::size_t length) {
    std::size_t worst = std::numeric_limits<std::size_t>::max();
    for (std::size_t i = 0UZ; i < length; ++i) {
        for (std::size_t j = i + 1UZ; j < length; ++j) {
            const std::size_t a = position[first + i];
            const std::size_t b = position[first + j];
            worst               = std::min(worst, (a > b) ? a - b : b - a);
        }
    }
    return worst;
}

} // namespace

const boost::ut::suite<"Interleaver"> interleaverTests = [] {
    // The family's contract is the spread, and it is exact: two items adjacent at the
    // input and in the same row land exactly `rows` apart. The pair that straddles a row boundary is the one
    // exception and it is arithmetic, not an accident — input `r*cols + (cols-1)` reads out at
    // `(cols-1)*rows + r` and the next input at `r + 1`, which is `(cols-1)*rows - 1` back the other way — so
    // the assertion states the within-row case and pins the boundary case to its closed form as well.
    "the block map is the column-major readout, and its spread is exactly rows"_test = [] {
        for (const auto& [rows, cols] : {std::pair{1UZ, 7UZ}, std::pair{7UZ, 1UZ}, std::pair{3UZ, 5UZ}, std::pair{5UZ, 3UZ}, std::pair{8UZ, 8UZ}, std::pair{4UZ, 16UZ}}) {
            const Interleaver<std::uint8_t> interleaver = Interleaver<std::uint8_t>::block(rows, cols);
            expect(eq(interleaver.frameSize(), rows * cols));
            expect(eq(interleaver.interleavedSize(), rows * cols));
            expect(eq(interleaver.latency(), 0UZ)) << "a framed kind produces a frame from a frame";
            expect(that % interleaver.covers());

            const auto map = interleaver.indexMap();
            for (std::size_t r = 0UZ; r < rows; ++r) {
                for (std::size_t c = 0UZ; c < cols; ++c) {
                    expect(eq(map[c * rows + r], r * cols + c)) << std::format("out[c*rows + r] = in[r*cols + c] at {}x{}", rows, cols);
                }
            }

            const std::vector<std::size_t> position = outputPositions(interleaver);
            for (std::size_t r = 0UZ; r < rows; ++r) {
                for (std::size_t c = 0UZ; c + 1UZ < cols; ++c) {
                    const std::size_t here = position[r * cols + c];
                    const std::size_t next = position[r * cols + c + 1UZ];
                    expect(eq(next - here, rows)) << std::format("adjacent inputs {}x{} are not rows apart", rows, cols);
                }
                if (r + 1UZ < rows && cols > 1UZ) {
                    const std::size_t last  = position[r * cols + cols - 1UZ];
                    const std::size_t after = position[(r + 1UZ) * cols];
                    expect(eq(last - after, (cols - 1UZ) * rows - 1UZ)) << "the row boundary is the one pair that is not rows apart";
                }
            }
        }
    };

    // A grid of shapes including both degenerate rectangles and a table permutation,
    // several frames per call so a per-frame base that was wrong would show.
    "a framed interleave and its deinterleave are the identity"_test = [] {
        for (const auto& [rows, cols] : {std::pair{1UZ, 7UZ}, std::pair{7UZ, 1UZ}, std::pair{3UZ, 5UZ}, std::pair{5UZ, 3UZ}, std::pair{16UZ, 16UZ}, std::pair{2UZ, 1024UZ}}) {
            Interleaver<std::uint8_t> interleaver = Interleaver<std::uint8_t>::block(rows, cols);

            constexpr std::size_t           kFrames = 3UZ;
            const std::vector<std::uint8_t> data    = seededBytes(kFrames * rows * cols, 0x51ee'd0'01ULL + rows * 1000UZ + cols);
            std::vector<std::uint8_t>       wire(data.size());
            std::vector<std::uint8_t>       back(data.size());

            expect(eq(interleaver.interleave(data, wire), data.size()));
            expect(eq(interleaver.deinterleave(wire, back), data.size()));
            expect(that % std::ranges::equal(back, data)) << std::format("block {}x{} did not round trip", rows, cols);
            const bool moved = (rows == 1UZ) || (cols == 1UZ) || !std::ranges::equal(wire, data);
            expect(that % moved) << "a non-degenerate block must actually move something";
        }

        for (const std::size_t n : {1UZ, 2UZ, 17UZ, 64UZ, 4095UZ}) {
            const std::vector<std::size_t> table       = seededPermutation(n, 0xfeedULL + n);
            Interleaver<std::uint8_t>      interleaver = Interleaver<std::uint8_t>::permutation(table);
            expect(eq(interleaver.frameSize(), n));
            expect(that % std::ranges::equal(interleaver.indexMap(), table)) << "the table is the gather, verbatim";

            const std::vector<std::uint8_t> data = seededBytes(2UZ * n, 0xabcdULL + n);
            std::vector<std::uint8_t>       wire(data.size());
            std::vector<std::uint8_t>       back(data.size());
            expect(eq(interleaver.interleave(data, wire), data.size()));
            expect(eq(interleaver.deinterleave(wire, back), data.size()));
            expect(that % std::ranges::equal(back, data)) << std::format("permutation of {} did not round trip", n);

            for (std::size_t i = 0UZ; i < n; ++i) {
                expect(eq(wire[i], data[table[i]])) << "out[i] = in[table[i]]";
            }
        }
    };

    // The delay identity and the round trip are one assertion: the recovered
    // stream is the original shifted by exactly `B*(B-1)*M`, and the items before that are the stated fill.
    // Both branch pairs sum to `(B-1)*M` cells whatever the branch is, and a cell holds its item for a whole
    // revolution of the commutator, so the delay is `B*(B-1)*M` items and is the same for every item.
    "the convolutional round trip closes at exactly B*(B-1)*M"_test = [] {
        for (const auto& [branches, unitDelay] : {std::pair{2UZ, 1UZ}, std::pair{2UZ, 4UZ}, std::pair{3UZ, 1UZ}, std::pair{5UZ, 2UZ}, std::pair{12UZ, 17UZ}, std::pair{16UZ, 1UZ}}) {
            Interleaver<std::uint8_t> forward = Interleaver<std::uint8_t>::convolutional(branches, unitDelay, 0xa5U);
            Interleaver<std::uint8_t> reverse = Interleaver<std::uint8_t>::convolutional(branches, unitDelay, 0xa5U);

            const std::size_t latency = branches * (branches - 1UZ) * unitDelay;
            expect(eq(forward.latency(), latency));
            expect(eq(forward.frameSize(), 0UZ)) << "the convolutional kind is stream-shaped and has no frame";

            const std::size_t               n    = latency + 4096UZ;
            const std::vector<std::uint8_t> data = seededBytes(n, 0xc0'ffeeULL + branches * 100UZ + unitDelay);
            std::vector<std::uint8_t>       wire(n);
            std::vector<std::uint8_t>       back(n);
            expect(eq(forward.interleave(data, wire), n));
            expect(eq(reverse.deinterleave(wire, back), n));

            for (std::size_t k = 0UZ; k < latency; ++k) {
                expect(eq(back[k], std::uint8_t{0xa5U})) << std::format("B={} M={}: the initial fill is not the stated value", branches, unitDelay);
            }
            for (std::size_t k = latency; k < n; ++k) {
                expect(eq(back[k], data[k - latency])) << std::format("B={} M={}: the round trip did not close at {}", branches, unitDelay, latency);
            }

            // One item earlier or later and the identity fails, which is what makes the latency a number rather
            // than a bound.
            expect(that % !std::ranges::equal(std::span(back).subspan(latency + 1UZ, 64UZ), std::span(data).first(64UZ)));

            forward.reset();
            std::vector<std::uint8_t> again(n);
            expect(eq(forward.interleave(data, again), n));
            expect(that % std::ranges::equal(again, wire)) << "reset did not return the lines to their fill state";
        }
    };

    // The convolutional spread, asserted from the index map the kernel actually realizes rather than
    // sampled. The map is `k -> k + (k mod B) * M * B`: a cell holds its item for one revolution of the
    // commutator, so `b*M` cells are `b*M*B` items of stream, and the shift being a multiple of B is what makes
    // the map injective.
    //
    // The burst figure follows from that map and is not `M + 1`. On the interleaver, a burst of `L <= B` items
    // starting at a commutator-aligned index crosses no wrap, so two members `d` apart land `d * (1 + M*B)`
    // apart and the closest pair is `1 + M*B`. A burst that does straddle the wrap has `d - B` branches between
    // its members instead of `d`, giving `|d*(1 + M*B) - B*M*B|`, which for `d = B-1, M = 1` is 1 — so the
    // interleaver-side spread is a property of aligned bursts only, and the assertion is stated that way.
    //
    // The guarantee the family exists for lives on the other side: a channel burst is adjacent at the
    // deinterleaver *input*, and there the map is `n -> n + (B - 1 - n mod B) * M * B`, so two members `d`
    // apart land `d*(M*B - 1)` apart without a wrap and `d*(1 - M*B) + M*B*B` with one, whose minimum over
    // `d < B` is `M*B - 1`. That is the number asserted. It exceeds `M + 1` for every shape except
    // `B = 2, M = 1`, where it is 1 — the degenerate interleaver spreads nothing, which is honest.
    "the convolutional map spreads a burst by the amount its arithmetic says"_test = [] {
        for (const auto& [branches, unitDelay] : {std::pair{2UZ, 1UZ}, std::pair{3UZ, 1UZ}, std::pair{4UZ, 3UZ}, std::pair{8UZ, 2UZ}, std::pair{16UZ, 4UZ}}) {
            constexpr std::size_t kItems = 512UZ;

            const std::vector<std::size_t> forward = convolutionalPositions(branches, unitDelay, kItems, Direction::Interleave);
            for (std::size_t k = 0UZ; k < kItems; ++k) {
                expect(eq(forward[k], k + (k % branches) * unitDelay * branches)) << std::format("B={} M={}: the interleave map is not k + (k mod B)*M*B", branches, unitDelay);
            }

            const std::size_t forwardBound = 1UZ + unitDelay * branches;
            for (std::size_t start = 0UZ; start + branches <= kItems; start += branches) {
                for (std::size_t length = 2UZ; length <= branches; ++length) {
                    expect(ge(minimumSeparation(forward, start, length), forwardBound)) << std::format("B={} M={}: an aligned burst of {} is closer than 1 + M*B", branches, unitDelay, length);
                }
            }

            const std::vector<std::size_t> reverse = convolutionalPositions(branches, unitDelay, kItems, Direction::Deinterleave);
            for (std::size_t n = 0UZ; n < kItems; ++n) {
                expect(eq(reverse[n], n + (branches - 1UZ - (n % branches)) * unitDelay * branches)) << std::format("B={} M={}: the deinterleave map is not n + (B-1-n mod B)*M*B", branches, unitDelay);
            }

            const std::size_t reverseBound = unitDelay * branches - 1UZ;
            std::size_t       worst        = std::numeric_limits<std::size_t>::max();
            for (std::size_t start = 0UZ; start + branches <= kItems; ++start) {
                for (std::size_t length = 2UZ; length <= branches; ++length) {
                    worst = std::min(worst, minimumSeparation(reverse, start, length));
                }
            }
            std::println("qa_Interleaver: B={:2} M={:2}: a channel burst of at most B lands at least {} apart, bound M*B - 1 = {}", branches, unitDelay, worst, reverseBound);
            expect(eq(worst, reverseBound)) << std::format("B={} M={}: the channel-burst spread is not M*B - 1", branches, unitDelay);
        }
    };

    // The record boundary is transparent to a convolutional interleaver, which is the one claim
    // about it a graph depends on: a block whose buffer size changes must not change its output.
    "convolutional state does not depend on where a call ended"_test = [] {
        constexpr std::size_t kItems = 4096UZ;

        for (const auto& [branches, unitDelay] : {std::pair{2UZ, 1UZ}, std::pair{5UZ, 3UZ}, std::pair{16UZ, 7UZ}}) {
            const std::vector<std::uint8_t> data = seededBytes(kItems, 0x5911ULL);
            std::vector<std::uint8_t>       whole(kItems);
            Interleaver<std::uint8_t>       reference = Interleaver<std::uint8_t>::convolutional(branches, unitDelay);
            expect(eq(reference.interleave(data, whole), kItems));

            for (const std::size_t chunk : {1UZ, 7UZ, 64UZ, 1000UZ}) {
                Interleaver<std::uint8_t> pieces = Interleaver<std::uint8_t>::convolutional(branches, unitDelay);
                std::vector<std::uint8_t> out(kItems);
                for (std::size_t done = 0UZ; done < kItems; done += chunk) {
                    const std::size_t take = std::min(chunk, kItems - done);
                    expect(eq(pieces.interleave(std::span(data).subspan(done, take), std::span(out).subspan(done, take)), take));
                }
                expect(that % std::ranges::equal(out, whole)) << std::format("B={} M={}: chunk size {} changed the stream", branches, unitDelay, chunk);
            }
        }
    };

    // The §47 addition. A framing that spends part of its interleaved frame on synchronization reads out a
    // window of it, and the deinterleaver has to put that window back where it came from and say, in the value
    // it writes, that the rest was never received.
    "a windowed block hands back its window and marks the rest"_test = [] {
        constexpr std::size_t kRows   = 4UZ;
        constexpr std::size_t kCols   = 5UZ;
        constexpr std::size_t kOffset = 3UZ;
        constexpr std::size_t kLength = 12UZ;

        Interleaver<std::uint8_t> interleaver = Interleaver<std::uint8_t>::block(kRows, kCols, kOffset, kLength, std::uint8_t{0x7fU});
        expect(eq(interleaver.frameSize(), kRows * kCols));
        expect(eq(interleaver.interleavedSize(), kLength));
        expect(that % !interleaver.covers());
        expect(that % interleaver.inverseMap().empty()) << "a partial map has no inverse gather to hold";

        const Interleaver<std::uint8_t> full = Interleaver<std::uint8_t>::block(kRows, kCols);
        for (std::size_t i = 0UZ; i < kLength; ++i) {
            expect(eq(interleaver.indexMap()[i], full.indexMap()[kOffset + i])) << "the window is a stretch of the full readout, not a second map";
        }

        const std::vector<std::uint8_t> data = seededBytes(2UZ * kRows * kCols, 0x0a4'0ULL);
        std::vector<std::uint8_t>       wire(2UZ * kLength);
        std::vector<std::uint8_t>       back(2UZ * kRows * kCols);
        expect(eq(interleaver.interleave(data, wire), wire.size()));
        expect(eq(interleaver.deinterleave(wire, back), back.size()));

        std::vector<bool> covered(kRows * kCols, false);
        for (const std::size_t source : interleaver.indexMap()) {
            covered[source] = true;
        }
        for (std::size_t frame = 0UZ; frame < 2UZ; ++frame) {
            for (std::size_t i = 0UZ; i < kRows * kCols; ++i) {
                const std::size_t at = frame * kRows * kCols + i;
                if (covered[i]) {
                    expect(eq(back[at], data[at])) << "a windowed position did not come back";
                } else {
                    expect(eq(back[at], std::uint8_t{0x7fU})) << "an unreached position must read the stated fill";
                }
            }
        }
        expect(eq(std::ranges::count(covered, true), static_cast<std::ptrdiff_t>(kLength)));
    };

    // The families are index maps and never read a value, so a soft decision rides them unchanged. This is the
    // instantiation the gr-satellites wave needs and the reason the kernel is templated at all.
    "soft items ride the same kernel"_test = [] {
        constexpr std::size_t kRows = 8UZ;
        constexpr std::size_t kCols = 13UZ;

        std::vector<float> soft(kRows * kCols);
        for (std::size_t i = 0UZ; i < soft.size(); ++i) {
            soft[i] = -3.375f + 0.25f * static_cast<float>(i); // exact in float, and never zero, which is the erasure value below
        }

        Interleaver<float> interleaver = Interleaver<float>::block(kRows, kCols);
        std::vector<float> wire(soft.size());
        std::vector<float> back(soft.size());
        expect(eq(interleaver.interleave(soft, wire), soft.size()));
        expect(eq(interleaver.deinterleave(wire, back), soft.size()));
        expect(that % std::ranges::equal(back, soft));

        // An erasure written into a windowed frame is a soft zero, which is exactly what a decoder wants told.
        Interleaver<float> windowed = Interleaver<float>::block(kRows, kCols, 8UZ, 40UZ, 0.f);
        std::vector<float> partial(40UZ);
        std::vector<float> recovered(soft.size());
        expect(eq(windowed.interleave(soft, partial), partial.size()));
        expect(eq(windowed.deinterleave(partial, recovered), recovered.size()));
        expect(eq(std::ranges::count(recovered, 0.f), static_cast<std::ptrdiff_t>(soft.size() - 40UZ)));

        const std::vector<std::size_t> table   = seededPermutation(97UZ, 7ULL);
        Interleaver<float>             byTable = Interleaver<float>::permutation(table);
        std::vector<float>             softShort(soft.begin(), soft.begin() + 97);
        std::vector<float>             tableWire(97UZ);
        std::vector<float>             tableBack(97UZ);
        expect(eq(byTable.interleave(softShort, tableWire), 97UZ));
        expect(eq(byTable.deinterleave(tableWire, tableBack), 97UZ));
        expect(that % std::ranges::equal(tableBack, softShort));
    };

    "a configuration that is not an interleaver is refused"_test = [] {
        expect(throws<std::invalid_argument>([] { (void)Interleaver<std::uint8_t>::block(0UZ, 8UZ); })) << "rows of 0";
        expect(throws<std::invalid_argument>([] { (void)Interleaver<std::uint8_t>::block(8UZ, 0UZ); })) << "cols of 0";
        expect(throws<std::invalid_argument>([] { (void)Interleaver<std::uint8_t>::block(1UZ << 11, 1UZ << 11); })) << "a frame past the ceiling";
        expect(nothrow([] { (void)Interleaver<std::uint8_t>::block(1UZ << 10, 1UZ << 10); })) << "a frame at the ceiling";
        expect(throws<std::invalid_argument>([] { (void)Interleaver<std::uint8_t>::block(4UZ, 5UZ, 0UZ, 0UZ, std::uint8_t{}); })) << "a window of no length";
        expect(throws<std::invalid_argument>([] { (void)Interleaver<std::uint8_t>::block(4UZ, 5UZ, 20UZ, 1UZ, std::uint8_t{}); })) << "a window starting past the frame";
        expect(throws<std::invalid_argument>([] { (void)Interleaver<std::uint8_t>::block(4UZ, 5UZ, 12UZ, 9UZ, std::uint8_t{}); })) << "a window running past the frame";
        expect(nothrow([] { (void)Interleaver<std::uint8_t>::block(4UZ, 5UZ, 12UZ, 8UZ, std::uint8_t{}); })) << "a window ending exactly at the frame";

        expect(throws<std::invalid_argument>([] { (void)Interleaver<std::uint8_t>::convolutional(0UZ, 1UZ); })) << "branches of 0";
        expect(throws<std::invalid_argument>([] { (void)Interleaver<std::uint8_t>::convolutional(1UZ, 1UZ); })) << "one branch interleaves nothing";
        expect(throws<std::invalid_argument>([] { (void)Interleaver<std::uint8_t>::convolutional(4UZ, 0UZ); })) << "unit_delay of 0";
        expect(throws<std::invalid_argument>([] { (void)Interleaver<std::uint8_t>::convolutional(1UZ << 12, 1UZ); })) << "a latency past the ceiling";
        expect(nothrow([] { (void)Interleaver<std::uint8_t>::convolutional(2UZ, 1UZ); })) << "two branches and one cell is the degenerate case, not an error";

        expect(throws<std::invalid_argument>([] { (void)Interleaver<std::uint8_t>::permutation(std::span<const std::size_t>{}); })) << "no table at all";
        expect(throws<std::invalid_argument>([] {
            const std::vector<std::size_t> table{0UZ, 1UZ, 3UZ};
            (void)Interleaver<std::uint8_t>::permutation(table);
        })) << "an index outside the frame";
        expect(throws<std::invalid_argument>([] {
            const std::vector<std::size_t> table{0UZ, 1UZ, 1UZ};
            (void)Interleaver<std::uint8_t>::permutation(table);
        })) << "an index used twice";
        expect(throws<std::invalid_argument>([] {
            std::vector<std::size_t> table(Interleaver<std::uint8_t>::kMaxFrame + 1UZ);
            std::iota(table.begin(), table.end(), 0UZ);
            (void)Interleaver<std::uint8_t>::permutation(table);
        })) << "a table past the ceiling";
        expect(nothrow([] {
            const std::vector<std::size_t> table{0UZ};
            (void)Interleaver<std::uint8_t>::permutation(table);
        })) << "the one-item permutation is a permutation";
    };

    "a partial frame is left where it is"_test = [] {
        Interleaver<std::uint8_t> interleaver = Interleaver<std::uint8_t>::block(4UZ, 5UZ);

        const std::vector<std::uint8_t> data = seededBytes(37UZ, 3ULL);
        std::vector<std::uint8_t>       wire(37UZ, 0xffU);
        expect(eq(interleaver.interleave(data, wire), 20UZ)) << "one whole frame of the 37 items offered";
        expect(eq(wire[20UZ], std::uint8_t{0xffU})) << "the kernel does not write past the frames it took";

        std::vector<std::uint8_t> tight(19UZ);
        expect(eq(interleaver.interleave(data, tight), 0UZ)) << "an output that cannot hold a frame gets none";
    };
};

int main() { /* not needed for UT */ }
