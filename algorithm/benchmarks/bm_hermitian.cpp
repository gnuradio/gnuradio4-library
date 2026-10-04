#include <benchmark.hpp>

#include <complex>
#include <format>
#include <random>
#include <tuple>

#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/algorithm/math/Hermitian.hpp>
#include <gnuradio-4.0/algorithm/math/LinearSolve.hpp>

/// R = X X^H / (2n) + I for an n x 2n matrix X of normal samples: Hermitian and positive definite.
template<typename T>
gr::Tensor<T> hermitianPositiveDefinite(std::size_t n, std::mt19937& rng) {
    using RealT = gr::meta::fundamental_base_value_type_t<T>;
    std::normal_distribution<RealT> normal;
    gr::Tensor<T>                   X({n, 2UZ * n});
    for (auto& x : X) {
        x = T{normal(rng), normal(rng)};
    }
    gr::Tensor<T> R({n, n});
    for (std::size_t i = 0UZ; i < n; ++i) {
        for (std::size_t j = 0UZ; j < n; ++j) {
            T sum{0};
            for (std::size_t k = 0UZ; k < 2UZ * n; ++k) {
                sum += X[i, k] * std::conj(X[j, k]);
            }
            R[i, j] = sum / static_cast<RealT>(2UZ * n);
        }
        R[i, i] += T{1};
    }
    return R;
}

template<typename T, std::size_t NRepetitions>
void benchmarkSize(std::size_t n) {
    using namespace boost::ut;
    using namespace boost::ut::reflection;
    using RealT = gr::meta::fundamental_base_value_type_t<T>;

    std::mt19937                    rng(7U);
    const gr::Tensor<T>             A = hermitianPositiveDefinite<T>(n, rng);
    gr::Tensor<T>                   b({n});
    std::normal_distribution<RealT> normal;
    for (auto& v : b) {
        v = T{normal(rng), normal(rng)};
    }
    gr::Tensor<T>     x;
    gr::Tensor<RealT> values;
    gr::Tensor<T>     vectors;
    expect(gr::math::solveHermitian(x, A, b) == gr::math::solve::Status::Success) << fatal;
    expect(gr::math::eigh(values, vectors, A) == gr::math::eig::Status::Success) << fatal;

    ::benchmark::benchmark<NRepetitions>(std::format("solveHermitian {:22} N={}", type_name<T>(), n)) = [&] { std::ignore = gr::math::solveHermitian(x, A, b); };
    ::benchmark::benchmark<NRepetitions>(std::format("eigh           {:22} N={}", type_name<T>(), n)) = [&] { std::ignore = gr::math::eigh(values, vectors, A); };
}

inline const boost::ut::suite<"Hermitian solve and eigendecomposition benchmark"> _hermitian_bm_tests = [] {
    using boost::ut::operator""_test;

    "std::complex<float>"_test = [] {
        benchmarkSize<std::complex<float>, 10000UZ>(4UZ);
        benchmarkSize<std::complex<float>, 2000UZ>(8UZ);
        benchmarkSize<std::complex<float>, 200UZ>(16UZ);
        ::benchmark::results::add_separator();
    };

    "std::complex<double>"_test = [] {
        benchmarkSize<std::complex<double>, 10000UZ>(4UZ);
        benchmarkSize<std::complex<double>, 2000UZ>(8UZ);
        benchmarkSize<std::complex<double>, 200UZ>(16UZ);
        ::benchmark::results::add_separator();
    };
};

int main() { /* not needed by the UT framework */ }
