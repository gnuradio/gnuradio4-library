#include <boost/ut.hpp>

#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/algorithm/math/Covariance.hpp>
#include <gnuradio-4.0/algorithm/math/Hermitian.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

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

/// The error bound of an n x n estimate relative to its largest entry: a multiple of n * eps.
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
Tensor<T> vector(std::initializer_list<std::pair<double, double>> values) {
    Tensor<T> v({values.size()});
    auto      it = v.begin();
    for (const auto& [re, im] : values) {
        *it++ = element<T>(re, im);
    }
    return v;
}

/// The reference covariance sum_k weight_k x_k x_k^H / sum_k weight_k, written as plain loops.
template<typename T>
Tensor<T> weightedCovariance(const std::vector<Tensor<T>>& snapshots, const std::vector<double>& weights) {
    const std::size_t n = snapshots.front().size();
    Tensor<T>         R({n, n});
    R.fill(T{0});
    double total = 0.0;
    for (std::size_t k = 0UZ; k < snapshots.size(); ++k) {
        total += weights[k];
        for (std::size_t i = 0UZ; i < n; ++i) {
            for (std::size_t j = 0UZ; j < n; ++j) {
                R[i, j] += static_cast<Real<T>>(weights[k]) * snapshots[k][i] * conjugate(snapshots[k][j]);
            }
        }
    }
    for (auto& r : R) {
        r /= static_cast<Real<T>>(total);
    }
    return R;
}

template<typename T>
bool isExactlyHermitian(const Tensor<T>& R) {
    for (std::size_t i = 0UZ; i < R.extent(0); ++i) {
        if (std::imag(R[i, i]) != Real<T>{0}) {
            return false;
        }
        for (std::size_t j = i + 1UZ; j < R.extent(1); ++j) {
            if (R[j, i] != conjugate(R[i, j])) {
                return false;
            }
        }
    }
    return true;
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

} // namespace

const boost::ut::suite<"vdot and outer"> _products = [] {
    using namespace boost::ut;

    "vdot conjugates its first argument"_test = []<typename T> {
        // complex: conj(1+i) * 3 + 2 * (1-i) = 5 - 5i; real: 1 * 3 + 2 * 1 = 5
        const Tensor<T> a = vector<T>({{1, 1}, {2, 0}});
        const Tensor<T> b = vector<T>({{3, 0}, {1, -1}});
        expect(vdot(a, b) == element<T>(5, -5)) << typeName<T>();
        expect(vdot(a, a) == element<T>(gr::meta::complex_like<T> ? 6.0 : 5.0)) << typeName<T>() << "vdot(a, a) = ||a||^2";
        expect(throws([&] { std::ignore = vdot(a, vector<T>({{1, 0}})); })) << typeName<T>();
    } | ElementTypes{};

    "outer is a b^H"_test = []<typename T> {
        // complex: [[(1+i)*3, (1+i)*conj(1-i)], [2*3, 2*conj(1-i)]] = [[3+3i, 2i], [6, 2+2i]]; real: [[3, 1], [6, 2]]
        const Tensor<T> a        = vector<T>({{1, 1}, {2, 0}});
        const Tensor<T> b        = vector<T>({{3, 0}, {1, -1}});
        const Tensor<T> expected = gr::meta::complex_like<T> ? matrix<T>(2, 2, {{3, 3}, {0, 2}, {6, 0}, {2, 2}}) : matrix<T>(2, 2, {{3, 0}, {1, 0}, {6, 0}, {2, 0}});
        Tensor<T>       C;
        outer(C, a, b);
        expect(eq(C.extent(0), 2UZ) && eq(C.extent(1), 2UZ)) << typeName<T>();
        expect(std::ranges::equal(C, expected)) << typeName<T>();
    } | ElementTypes{};
};

const boost::ut::suite<"CovarianceAccumulator"> _accumulator = [] {
    using namespace boost::ut;

    "block averaging completes on the last snapshot of a block"_test = []<typename T> {
        const std::vector<Tensor<T>> snapshots{vector<T>({{1, 0}, {0, 1}, {2, -1}}), vector<T>({{-1, 1}, {3, 0}, {0, 2}}), vector<T>({{0, -2}, {1, 1}, {1, 0}})};
        CovarianceAccumulator<T>     accumulator(3UZ, 3UZ);
        expect(!accumulator.add(snapshots[0]) && !accumulator.add(snapshots[1])) << typeName<T>();
        expect(accumulator.add(snapshots[2])) << typeName<T>() << "the third snapshot completes the block";
        expect(eq(accumulator.count(), 3UZ)) << typeName<T>();

        const Tensor<T> R = accumulator.matrix();
        expect(le(relativeError(R, weightedCovariance(snapshots, {1.0, 1.0, 1.0})), tolerance<T>(3))) << typeName<T>();
        expect(isExactlyHermitian(R)) << typeName<T>() << "upper triangle mirrored, real diagonal";

        const Tensor<T> next = vector<T>({{2, 0}, {0, 0}, {0, -1}});
        expect(!accumulator.add(next)) << typeName<T>();
        expect(eq(accumulator.count(), 1UZ)) << typeName<T>() << "the next snapshot starts a new block";
        expect(le(relativeError(accumulator.matrix(), weightedCovariance<T>({next}, {1.0})), tolerance<T>(3))) << typeName<T>();
    } | ElementTypes{};

    "exponential averaging weights by powers of the forgetting factor"_test = []<typename T> {
        const std::vector<Tensor<T>> snapshots{vector<T>({{0, 1}, {1, 0}}), vector<T>({{2, 0}, {0, 0}}), vector<T>({{1, -1}, {-1, 0}})};
        CovarianceAccumulator<T>     accumulator(2UZ);
        expect(accumulator.setExponentialAveraging(Real<T>{0.5f})) << typeName<T>();
        expect(accumulator.add(snapshots[0]) && accumulator.add(snapshots[1])) << typeName<T>() << "every snapshot completes an estimate";

        // (0.5 x1 x1^H + x2 x2^H) / 1.5, worked by hand
        const Tensor<T> afterTwo = gr::meta::complex_like<T> ? matrix<T>(2, 2, {{3, 0}, {0, 1.0 / 3.0}, {0, -1.0 / 3.0}, {1.0 / 3.0, 0}}) //
                                                             : matrix<T>(2, 2, {{8.0 / 3.0, 0}, {0, 0}, {0, 0}, {1.0 / 3.0, 0}});
        expect(le(relativeError(accumulator.matrix(), afterTwo), tolerance<T>(2))) << typeName<T>();

        std::ignore       = accumulator.add(snapshots[2]);
        const Tensor<T> R = accumulator.matrix();
        expect(le(relativeError(R, weightedCovariance(snapshots, {0.25, 0.5, 1.0})), tolerance<T>(2))) << typeName<T>();
        expect(isExactlyHermitian(R)) << typeName<T>();
    } | ElementTypes{};

    "reset, a wrong length and an invalid forgetting factor"_test = []<typename T> {
        CovarianceAccumulator<T> accumulator(2UZ, 4UZ);
        std::ignore = accumulator.add(vector<T>({{1, 0}, {2, 0}}));
        expect(!accumulator.add(vector<T>({{1, 0}, {2, 0}, {3, 0}}))) << typeName<T>();
        expect(eq(accumulator.count(), 1UZ)) << typeName<T>() << "a snapshot of another length is not counted";

        accumulator.reset();
        expect(eq(accumulator.count(), 0UZ)) << typeName<T>();
        const Tensor<T> R = accumulator.matrix();
        expect(eq(R.size(), 4UZ) && std::ranges::all_of(R, [](const T& r) { return r == T{0}; })) << typeName<T>();

        expect(!accumulator.setExponentialAveraging(Real<T>{0}) && !accumulator.setExponentialAveraging(Real<T>{1.5f})) << typeName<T>();
        expect(accumulator.averaging() == CovarianceAccumulator<T>::Averaging::Block && eq(accumulator.blockSize(), 4UZ)) << typeName<T>();
    } | ElementTypes{};

    "two plane waves give a rank-2 covariance whose noise subspace is orthogonal to both"_test = []<typename T> {
        // four elements; a1 advances the phase by 90 degrees per element and a2 by 180 degrees.
        // A real element type keeps the real parts.
        const Tensor<T> a1 = vector<T>({{1, 0}, {0, 1}, {-1, 0}, {0, -1}});
        const Tensor<T> a2 = vector<T>({{1, 0}, {-1, 0}, {1, 0}, {-1, 0}});

        // the four sign pairs of the two sources cancel the cross terms: R = a1 a1^H + a2 a2^H
        CovarianceAccumulator<T> accumulator(4UZ, 4UZ);
        bool                     complete = false;
        for (const auto& [s1, s2] : std::array{std::pair{1, 1}, std::pair{1, -1}, std::pair{-1, 1}, std::pair{-1, -1}}) {
            Tensor<T> x({4UZ});
            for (std::size_t i = 0UZ; i < 4UZ; ++i) {
                x[i] = static_cast<Real<T>>(s1) * a1[i] + static_cast<Real<T>>(s2) * a2[i];
            }
            complete = accumulator.add(x);
        }
        expect(complete) << typeName<T>();
        const Tensor<T> R        = accumulator.matrix();
        Tensor<T>       expected = weightedCovariance<T>({a1, a2}, {1.0, 1.0}); // (a1 a1^H + a2 a2^H) / 2
        for (auto& e : expected) {
            e *= Real<T>{2};
        }
        expect(le(relativeError(R, expected), tolerance<T>(4))) << typeName<T>();

        Tensor<Real<T>> values;
        Tensor<T>       V;
        expect(eigh(values, V, R) == eig::Status::Success) << typeName<T>();
        Tensor<T> signal;
        Tensor<T> noise;
        expect(subspace(signal, noise, values, V, eig::Rank{2UZ}) == eig::Status::Success) << typeName<T>();
        const Real<T> scale = std::max(std::abs(vdot(a1, a1)), std::abs(vdot(a2, a2)));
        for (std::size_t k = 0UZ; k < noise.extent(1); ++k) {
            Tensor<T> column({4UZ});
            for (std::size_t i = 0UZ; i < 4UZ; ++i) {
                column[i] = noise[i, k];
            }
            expect(le(std::abs(vdot(column, a1)), tolerance<T>(4) * scale)) << typeName<T>() << "noise column" << k << "against a1";
            expect(le(std::abs(vdot(column, a2)), tolerance<T>(4) * scale)) << typeName<T>() << "noise column" << k << "against a2";
        }
    } | ElementTypes{};
};

int main() { /* tests are automatically registered and executed */ return 0; }
