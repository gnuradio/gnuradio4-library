#include <cstdint>
#include <random>

#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>

int main() {
    gr::rng::Xoshiro256pp rng(0);
    return rng() == std::uint64_t{0x53175d61490b23dfULL} ? 0 : 1;
}
