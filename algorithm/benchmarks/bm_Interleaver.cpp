#include <benchmark.hpp>

#include <gnuradio-4.0/algorithm/fec/Interleaver.hpp>
#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>

#include <cstddef>
#include <cstdint>
#include <format>
#include <numeric>
#include <random>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kItems   = 65'536UZ;
constexpr std::size_t kRepeats = 200UZ;

[[nodiscard]] std::vector<std::uint8_t> seededBytes(std::size_t n) {
    gr::rng::Xoshiro256pp     rng(0xb0'0bULL);
    std::vector<std::uint8_t> data(n);
    for (std::uint8_t& value : data) {
        value = static_cast<std::uint8_t>(rng() & 0xffULL);
    }
    return data;
}

[[nodiscard]] std::vector<std::size_t> seededPermutation(std::size_t n) {
    gr::rng::Xoshiro256pp    rng(0xb0'0bULL);
    std::vector<std::size_t> table(n);
    std::iota(table.begin(), table.end(), 0UZ);
    for (std::size_t i = n; i > 1UZ; --i) {
        const std::size_t j = std::uniform_int_distribution<std::size_t>{0UZ, i - 1UZ}(rng);
        std::swap(table[i - 1UZ], table[j]);
    }
    return table;
}

} // namespace

/// Every row is one item's worth of work, so the four rows are directly comparable. What separates them is
/// only how the address of the next item is arrived at: the block and permutation kinds read it out of a
/// resolved table and gather, and the two differ solely in how far apart the loads land — the block's stride is
/// `cols` items and the permutation's is a shuffle, so the row pair measures what the cache thinks of a random
/// gather. The convolutional kind reads no table at all and instead advances a commutator and one ring pointer,
/// which is the cheapest of the three per item and the only one carrying state between calls.
void benchInterleave() {
    using namespace benchmark;
    using boost::ut::operator""_test;

    "interleave: one item's work, by kind"_test = [] {
        const std::vector<std::uint8_t> data = seededBytes(kItems);
        std::vector<std::uint8_t>       out(kItems);

        {
            gr::fec::Interleaver<std::uint8_t> kernel                        = gr::fec::Interleaver<std::uint8_t>::block(64UZ, 64UZ);
            const auto                         name                          = std::format("block         64x64,   uint8, N={}", kItems);
            ::benchmark::benchmark<kRepeats>(std::string_view(name), kItems) = [&] { (void)kernel.interleave(data, out); };
        }
        {
            gr::fec::Interleaver<std::uint8_t> kernel                        = gr::fec::Interleaver<std::uint8_t>::block(256UZ, 16UZ);
            const auto                         name                          = std::format("block         256x16,  uint8, N={}", kItems);
            ::benchmark::benchmark<kRepeats>(std::string_view(name), kItems) = [&] { (void)kernel.interleave(data, out); };
        }
        {
            const std::vector<std::size_t>     table                         = seededPermutation(4'096UZ);
            gr::fec::Interleaver<std::uint8_t> kernel                        = gr::fec::Interleaver<std::uint8_t>::permutation(table);
            const auto                         name                          = std::format("permutation   4096,    uint8, N={}", kItems);
            ::benchmark::benchmark<kRepeats>(std::string_view(name), kItems) = [&] { (void)kernel.interleave(data, out); };
        }
        {
            gr::fec::Interleaver<std::uint8_t> kernel                        = gr::fec::Interleaver<std::uint8_t>::convolutional(12UZ, 17UZ);
            const auto                         name                          = std::format("convolutional B=12 M=17, uint8, N={}", kItems);
            ::benchmark::benchmark<kRepeats>(std::string_view(name), kItems) = [&] { (void)kernel.interleave(data, out); };
        }
        ::benchmark::results::add_separator();
    };

    // The soft-decision instantiation is the same code over a four-times-wider item, so the pair says how much
    // of the cost is the addressing and how much is moving the item.
    "interleave: the soft-decision item"_test = [] {
        std::vector<float> soft(kItems);
        for (std::size_t i = 0UZ; i < kItems; ++i) {
            soft[i] = static_cast<float>(i % 251UZ);
        }
        std::vector<float> out(kItems);

        {
            gr::fec::Interleaver<float> kernel                               = gr::fec::Interleaver<float>::block(64UZ, 64UZ);
            const auto                  name                                 = std::format("block         64x64,   float, N={}", kItems);
            ::benchmark::benchmark<kRepeats>(std::string_view(name), kItems) = [&] { (void)kernel.interleave(soft, out); };
        }
        {
            gr::fec::Interleaver<float> kernel                               = gr::fec::Interleaver<float>::convolutional(12UZ, 17UZ);
            const auto                  name                                 = std::format("convolutional B=12 M=17, float, N={}", kItems);
            ::benchmark::benchmark<kRepeats>(std::string_view(name), kItems) = [&] { (void)kernel.interleave(soft, out); };
        }
        ::benchmark::results::add_separator();
    };

    // Deinterleaving a full map is the inverse gather and should cost what interleaving costs; a windowed block
    // has no inverse gather to hold, so it clears the frame and scatters the window back into it, which is the
    // one path here that touches an item twice.
    "deinterleave: the inverse gather against the windowed scatter"_test = [] {
        const std::vector<std::uint8_t> wire = seededBytes(kItems);
        std::vector<std::uint8_t>       out(kItems);

        {
            gr::fec::Interleaver<std::uint8_t> kernel                        = gr::fec::Interleaver<std::uint8_t>::block(64UZ, 64UZ);
            const auto                         name                          = std::format("deinterleave  64x64 full,   N={}", kItems);
            ::benchmark::benchmark<kRepeats>(std::string_view(name), kItems) = [&] { (void)kernel.deinterleave(wire, out); };
        }
        {
            gr::fec::Interleaver<std::uint8_t> kernel                        = gr::fec::Interleaver<std::uint8_t>::block(64UZ, 64UZ, 512UZ, 3'072UZ, std::uint8_t{});
            const auto                         name                          = std::format("deinterleave  64x64 window, N={}", kItems);
            ::benchmark::benchmark<kRepeats>(std::string_view(name), kItems) = [&] { (void)kernel.deinterleave(wire, out); };
        }
        ::benchmark::results::add_separator();
    };
}

inline const boost::ut::suite<"interleaver benchmarks"> _interleaver_bm = [] { benchInterleave(); };

int main() { /* not needed by the UT framework */ }
