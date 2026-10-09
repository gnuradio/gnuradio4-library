#include <boost/ut.hpp>

#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/algorithm/math/Hermitian.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <tuple>
#include <utility>

namespace {

using gr::Tensor;
using namespace gr::math;

using ElementTypes = std::tuple<float, double, std::complex<float>, std::complex<double>>;

template<typename T>
using Real = gr::meta::fundamental_base_value_type_t<T>;

template<typename T>
std::string typeName() {
    if constexpr (std::is_same_v<T, float>) {
        return "float";
    } else if constexpr (std::is_same_v<T, double>) {
        return "double";
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return "complex<float>";
    } else {
        return "complex<double>";
    }
}

/// An element from a real and an imaginary part; a real element type drops the imaginary part.
template<typename T>
T element(double re, double im = 0.0) {
    if constexpr (gr::meta::complex_like<T>) {
        return T{static_cast<Real<T>>(re), static_cast<Real<T>>(im)};
    } else {
        return static_cast<T>(re);
    }
}

template<typename T>
T conjugate(const T& x) {
    if constexpr (gr::meta::complex_like<T>) {
        return std::conj(x);
    } else {
        return x;
    }
}

/// The error bound of a decomposition of an n x n matrix relative to its largest entry: a multiple of n * eps.
template<typename T>
Real<T> tolerance(std::size_t n) {
    return Real<T>{64} * static_cast<Real<T>>(n) * std::numeric_limits<Real<T>>::epsilon();
}

template<typename T>
Tensor<T> matrix(std::size_t rows, std::size_t cols, std::initializer_list<std::pair<double, double>> values) {
    Tensor<T> M({rows, cols});
    auto      it = M.begin();
    for (const auto& [re, im] : values) {
        *it++ = element<T>(re, im);
    }
    return M;
}

template<typename T>
Tensor<T> multiply(const Tensor<T>& A, const Tensor<T>& B) {
    Tensor<T> C({A.extent(0), B.extent(1)});
    C.fill(T{0});
    for (std::size_t i = 0UZ; i < A.extent(0); ++i) {
        for (std::size_t j = 0UZ; j < B.extent(1); ++j) {
            for (std::size_t k = 0UZ; k < A.extent(1); ++k) {
                C[i, j] += A[i, k] * B[k, j];
            }
        }
    }
    return C;
}

template<typename T>
Tensor<T> adjoint(const Tensor<T>& A) {
    Tensor<T> H({A.extent(1), A.extent(0)});
    for (std::size_t i = 0UZ; i < A.extent(0); ++i) {
        for (std::size_t j = 0UZ; j < A.extent(1); ++j) {
            H[j, i] = conjugate(A[i, j]);
        }
    }
    return H;
}

template<typename T>
Tensor<T> identity(std::size_t n) {
    Tensor<T> I({n, n});
    I.fill(T{0});
    for (std::size_t i = 0UZ; i < n; ++i) {
        I[i, i] = T{1};
    }
    return I;
}

/// The largest element-wise distance between two tensors of equal size, relative to the largest magnitude in `reference`.
template<typename T>
Real<T> relativeError(const Tensor<T>& actual, const Tensor<T>& reference) {
    Real<T> scale{0};
    Real<T> error{0};
    for (std::size_t k = 0UZ; k < reference.size(); ++k) {
        scale = std::max(scale, static_cast<Real<T>>(std::abs(reference.data()[k])));
        error = std::max(error, static_cast<Real<T>>(std::abs(actual.data()[k] - reference.data()[k])));
    }
    return error / std::max(scale, std::numeric_limits<Real<T>>::min());
}

/// A Hermitian 5 x 5 matrix written by its upper triangle.
template<typename T>
Tensor<T> chosenHermitian() {
    Tensor<T> A = matrix<T>(5, 5,
        {{4, 0}, {1, 2}, {0, -1}, {2, 0}, {-1, 1},    //
            {0, 0}, {-3, 0}, {1, 1}, {0, 2}, {3, 0},  //
            {0, 0}, {0, 0}, {2, 0}, {-2, 1}, {1, -1}, //
            {0, 0}, {0, 0}, {0, 0}, {5, 0}, {0, 1},   //
            {0, 0}, {0, 0}, {0, 0}, {0, 0}, {-1, 0}});
    for (std::size_t i = 0UZ; i < 5UZ; ++i) {
        for (std::size_t j = i + 1UZ; j < 5UZ; ++j) {
            A[j, i] = conjugate(A[i, j]);
        }
    }
    return A;
}

template<typename T>
Tensor<T> diagonal(const Tensor<Real<T>>& values) {
    Tensor<T> D({values.size(), values.size()});
    D.fill(T{0});
    for (std::size_t i = 0UZ; i < values.size(); ++i) {
        D[i, i] = T{values[i]};
    }
    return D;
}

/// ||M^H a|| / ||a||: the share of a vector that the columns of M capture.
template<typename T>
Real<T> projectionNorm(const Tensor<T>& M, const Tensor<T>& a) {
    const Tensor<T> projection = multiply(adjoint(M), a);
    Real<T>         numerator{0};
    Real<T>         denominator{0};
    for (const T& p : projection) {
        numerator += static_cast<Real<T>>(std::norm(p));
    }
    for (const T& x : a) {
        denominator += static_cast<Real<T>>(std::norm(x));
    }
    return std::sqrt(numerator / denominator);
}

} // namespace

const boost::ut::suite<"eigh"> _eigh = [] {
    using namespace boost::ut;

    "reconstruction A = V diag(values) V^H and orthonormal V"_test = []<typename T> {
        const Tensor<T> A = chosenHermitian<T>();
        Tensor<Real<T>> values;
        Tensor<T>       V;
        expect(eigh(values, V, A) == eig::Status::Success) << typeName<T>();
        expect(eq(values.size(), 5UZ) && eq(V.extent(0), 5UZ) && eq(V.extent(1), 5UZ)) << typeName<T>();
        expect(le(relativeError(multiply(multiply(V, diagonal<T>(values)), adjoint(V)), A), tolerance<T>(5))) << typeName<T>() << "A = V diag V^H";
        expect(le(relativeError(multiply(adjoint(V), V), identity<T>(5)), tolerance<T>(5))) << typeName<T>() << "V^H V = I";
        expect(std::ranges::is_sorted(values)) << typeName<T>() << "ascending";
    } | ElementTypes{};

    "a matrix near either end of the range decomposes as at unit scale"_test = []<typename T> {
        // Above sqrt(max) the square of an entry overflows; below sqrt(denorm_min) it underflows to zero.
        const Real<T> large = std::sqrt(std::numeric_limits<Real<T>>::max()) * Real<T>{4};
        const Real<T> small = std::sqrt(std::numeric_limits<Real<T>>::denorm_min()) / Real<T>{16};
        for (const Real<T> scale : {large, small}) {
            Tensor<T> A = chosenHermitian<T>();
            for (auto& a : A) {
                a *= scale;
            }
            Tensor<Real<T>> values;
            Tensor<T>       V;
            expect(eigh(values, V, A) == eig::Status::Success) << typeName<T>() << scale << fatal;
            expect(le(relativeError(multiply(multiply(V, diagonal<T>(values)), adjoint(V)), A), tolerance<T>(5))) << typeName<T>() << scale << "A = V diag V^H";
            expect(le(relativeError(multiply(adjoint(V), V), identity<T>(5)), tolerance<T>(5))) << typeName<T>() << scale << "V^H V = I";
        }

        Tensor<T> beyond = chosenHermitian<T>(); // finite entries whose Frobenius norm exceeds max
        for (auto& a : beyond) {
            a *= std::numeric_limits<Real<T>>::max() / Real<T>{8};
        }
        Tensor<Real<T>> values;
        Tensor<T>       V;
        expect(eigh(values, V, beyond) == eig::Status::InvalidInput) << typeName<T>();
    } | ElementTypes{};

    "a pair whose diagonal difference overflows decomposes as it does at unit scale"_test = []<typename T> {
        // [[-s, g], [conj(g), s]] with |g| = s / 8 has eigenvalues -s sqrt(65) / 8 and s sqrt(65) / 8.
        // For s = 2^(max_exponent - 1) the eigenvalues and the Frobenius norm are finite, and 2 s overflows.
        const Real<T>                   top    = std::ldexp(Real<T>{1}, std::numeric_limits<Real<T>>::max_exponent - 1);
        const std::pair<double, double> g      = gr::meta::complex_like<T> ? std::pair{0.6, 0.8} : std::pair{1.0, 0.0};
        const Real<T>                   lambda = std::sqrt(Real<T>{65}) / Real<T>{8};
        for (const Real<T> scale : {Real<T>{1}, top}) {
            Tensor<T> A = matrix<T>(2, 2, {{-1, 0}, {g.first / 8, g.second / 8}, {g.first / 8, -g.second / 8}, {1, 0}});
            for (auto& a : A) {
                a *= scale;
            }
            Tensor<Real<T>> values;
            Tensor<T>       V;
            expect(eigh(values, V, A) == eig::Status::Success) << typeName<T>() << scale << fatal;

            // the residual A V - V diag(values) relative to s, with each factor divided by s, a power of two, before the product
            Real<T> residual{0};
            for (std::size_t i = 0UZ; i < 2UZ; ++i) {
                for (std::size_t k = 0UZ; k < 2UZ; ++k) {
                    T av{0};
                    for (std::size_t j = 0UZ; j < 2UZ; ++j) {
                        av += (A[i, j] / scale) * V[j, k];
                    }
                    residual = std::max(residual, static_cast<Real<T>>(std::abs(av - V[i, k] * (values[k] / scale))));
                }
            }
            expect(le(residual, tolerance<T>(2))) << typeName<T>() << scale << "A V = V diag(values)";
            expect(le(std::abs(values[0] / scale + lambda), tolerance<T>(2))) << typeName<T>() << scale << values[0];
            expect(le(std::abs(values[1] / scale - lambda), tolerance<T>(2))) << typeName<T>() << scale << values[1];
        }
    } | ElementTypes{};

    "eigenvalues worked by hand"_test = []<typename T> {
        // [[2, g], [conj(g), 2]] with |g| = 1 has eigenvalues 2 - |g| and 2 + |g|
        const std::pair<double, double> g = gr::meta::complex_like<T> ? std::pair{0.6, 0.8} : std::pair{1.0, 0.0};
        const Tensor<T>                 A = matrix<T>(2, 2, {{2, 0}, g, {g.first, -g.second}, {2, 0}});
        Tensor<Real<T>>                 values;
        Tensor<T>                       V;
        expect(eigh(values, V, A) == eig::Status::Success) << typeName<T>();
        expect(le(std::abs(values[0] - Real<T>{1}), tolerance<T>(2) * Real<T>{3})) << typeName<T>() << values[0];
        expect(le(std::abs(values[1] - Real<T>{3}), tolerance<T>(2) * Real<T>{3})) << typeName<T>() << values[1];

        // a diagonal matrix is sorted, and its eigenvectors are the permuted unit vectors
        const Tensor<T> D = matrix<T>(3, 3, {{3, 0}, {0, 0}, {0, 0}, {0, 0}, {-1, 0}, {0, 0}, {0, 0}, {0, 0}, {2, 0}});
        expect(eigh(values, V, D) == eig::Status::Success) << typeName<T>();
        expect(values[0] == Real<T>{-1} && values[1] == Real<T>{2} && values[2] == Real<T>{3}) << typeName<T>();
        expect(V[1, 0] == T{1} && V[2, 1] == T{1} && V[0, 2] == T{1}) << typeName<T>();
    } | ElementTypes{};

    "only the upper triangle is read"_test = []<typename T> {
        const Tensor<T> A     = chosenHermitian<T>();
        Tensor<T>       upper = A;
        for (std::size_t i = 0UZ; i < 5UZ; ++i) {
            for (std::size_t j = 0UZ; j < i; ++j) {
                upper[i, j] = element<T>(100, -50);
            }
        }
        Tensor<Real<T>> values;
        Tensor<Real<T>> valuesUpper;
        Tensor<T>       V;
        expect(eigh(values, V, A) == eig::Status::Success) << typeName<T>();
        expect(eigh(valuesUpper, V, upper) == eig::Status::Success) << typeName<T>();
        expect(std::ranges::equal(values, valuesUpper)) << typeName<T>();
    } | ElementTypes{};

    "too few sweeps report MaxIterations; invalid input is refused"_test = []<typename T> {
        const Tensor<T> A = chosenHermitian<T>();
        Tensor<Real<T>> values;
        Tensor<T>       V;
        expect(eigh(values, V, A, eig::Config<Real<T>>{.maxIterations = 1UZ}) == eig::Status::MaxIterations) << typeName<T>();
        expect(eq(values.size(), 5UZ)) << typeName<T>() << "the estimate is returned";

        const Tensor<T> rectangular({2UZ, 3UZ});
        expect(eigh(values, V, rectangular) == eig::Status::InvalidInput) << typeName<T>();
        Tensor<T> withInf = A;
        withInf[0, 1]     = element<T>(std::numeric_limits<double>::infinity());
        expect(eigh(values, V, withInf) == eig::Status::InvalidInput) << typeName<T>();
    } | ElementTypes{};
};

const boost::ut::suite<"subspace"> _subspace = [] {
    using namespace boost::ut;

    "two plane waves: the noise subspace is orthogonal to both steering vectors"_test = []<typename T> {
        // six elements; a1 advances the phase by 90 degrees per element and a2 by 180 degrees.
        // A real element type keeps the real parts, (1, 0, -1, 0, 1, 0) and (1, -1, 1, -1, 1, -1).
        const Tensor<T> a1 = matrix<T>(6, 1, {{1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 0}, {0, 1}});
        const Tensor<T> a2 = matrix<T>(6, 1, {{1, 0}, {-1, 0}, {1, 0}, {-1, 0}, {1, 0}, {-1, 0}});

        // R = a1 a1^H + 2 a2 a2^H + noisePower I
        const Real<T> noisePower = Real<T>{0.01f};
        Tensor<T>     R          = multiply(a1, adjoint(a1));
        const auto    R2         = multiply(a2, adjoint(a2));
        for (std::size_t i = 0UZ; i < 6UZ; ++i) {
            for (std::size_t j = 0UZ; j < 6UZ; ++j) {
                R[i, j] += T{2} * R2[i, j];
            }
            R[i, i] += T{noisePower};
        }

        Tensor<Real<T>> values;
        Tensor<T>       V;
        expect(eigh(values, V, R) == eig::Status::Success) << typeName<T>();
        for (std::size_t k = 0UZ; k < 4UZ; ++k) {
            expect(le(std::abs(values[k] - noisePower), tolerance<T>(6) * Real<T>{12})) << typeName<T>() << "noise eigenvalue" << k;
        }
        expect(gt(values[4], Real<T>{1})) << typeName<T>();

        Tensor<T> signal;
        Tensor<T> noise;
        expect(subspace(signal, noise, values, V, eig::Rank{2UZ}) == eig::Status::Success) << typeName<T>();
        expect(eq(signal.extent(1), 2UZ) && eq(noise.extent(1), 4UZ) && eq(noise.extent(0), 6UZ)) << typeName<T>();
        const Real<T> bound = tolerance<T>(6) * Real<T>{12} / (values[4] - values[3]); // the error of R over the eigenvalue gap bounds the subspace error
        expect(le(projectionNorm(noise, a1), bound)) << typeName<T>() << "En^H a1 = 0";
        expect(le(projectionNorm(noise, a2), bound)) << typeName<T>() << "En^H a2 = 0";
        expect(le(std::abs(projectionNorm(signal, a1) - Real<T>{1}), bound)) << typeName<T>() << "a1 lies in the signal subspace";
        expect(le(std::abs(projectionNorm(signal, a2) - Real<T>{1}), bound)) << typeName<T>() << "a2 lies in the signal subspace";

        Tensor<T> signalByThreshold;
        Tensor<T> noiseByThreshold;
        expect(subspace(signalByThreshold, noiseByThreshold, values, V, eig::Threshold{Real<T>{10} * noisePower}) == eig::Status::Success) << typeName<T>();
        expect(std::ranges::equal(signalByThreshold, signal) && std::ranges::equal(noiseByThreshold, noise)) << typeName<T>() << "a threshold above the noise floor splits as rank 2";
    } | ElementTypes{};

    "an out-of-range rank is refused"_test = []<typename T> {
        Tensor<Real<T>> values;
        Tensor<T>       V;
        expect(eigh(values, V, identity<T>(3)) == eig::Status::Success) << typeName<T>();
        Tensor<T> signal;
        Tensor<T> noise;
        expect(subspace(signal, noise, values, V, eig::Rank{4UZ}) == eig::Status::InvalidInput) << typeName<T>();
        expect(subspace(signal, noise, values, V, eig::Rank{3UZ}) == eig::Status::Success && eq(noise.extent(1), 0UZ)) << typeName<T>();
        expect(subspace(signal, noise, values, V, eig::Threshold{Real<T>{2}}) == eig::Status::Success && eq(signal.extent(1), 0UZ)) << typeName<T>();
    } | ElementTypes{};
};

int main() { /* tests are automatically registered and executed */ return 0; }
