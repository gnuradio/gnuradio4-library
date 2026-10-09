#include <boost/ut.hpp>

#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/algorithm/math/LinearSolve.hpp>

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

/// The error bound of a factorization or a solve of a well-conditioned n x n system: a multiple of n * eps.
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

/// A lower-triangular factor with a positive diagonal that dominates its rows, so L L^H is well conditioned.
template<typename T>
Tensor<T> chosenFactor() {
    return matrix<T>(4, 4,
        {{4, 0}, {0, 0}, {0, 0}, {0, 0},      //
            {1, 1}, {5, 0}, {0, 0}, {0, 0},   //
            {-1, 0}, {2, -1}, {6, 0}, {0, 0}, //
            {0, 1}, {1, 0}, {-2, 1}, {7, 0}});
}

/// A tall 6 x 3 matrix of full column rank.
template<typename T>
Tensor<T> tallMatrix() {
    return matrix<T>(6, 3,
        {{3, 1}, {1, 0}, {0, -1},    //
            {1, 0}, {4, 2}, {1, 1},  //
            {0, 2}, {1, -1}, {5, 0}, //
            {2, 0}, {0, 1}, {1, 0},  //
            {-1, 1}, {2, 0}, {0, 3}, //
            {1, -2}, {-1, 0}, {2, 1}});
}

} // namespace

const boost::ut::suite<"Cholesky"> _cholesky = [] {
    using namespace boost::ut;

    "the factor of L0 L0^H is L0"_test = []<typename T> {
        const Tensor<T> L0 = chosenFactor<T>();
        const Tensor<T> A  = multiply(L0, adjoint(L0));
        Tensor<T>       L;
        expect(cholesky(L, A) == solve::Status::Success) << typeName<T>();
        expect(le(relativeError(L, L0), tolerance<T>(4))) << typeName<T>();
        expect(L[0, 1] == T{0} && L[0, 3] == T{0} && L[2, 3] == T{0}) << typeName<T>() << "zeros above the diagonal";
    } | ElementTypes{};

    "solve recovers a chosen solution"_test = []<typename T> {
        const Tensor<T> L0 = chosenFactor<T>();
        const Tensor<T> A  = multiply(L0, adjoint(L0));
        const Tensor<T> x0 = matrix<T>(4, 2, {{1, 0}, {0, 1}, {-2, 1}, {3, 0}, {0, -1}, {1, 1}, {4, 2}, {-1, 0}});
        const Tensor<T> b  = multiply(A, x0);

        Tensor<T> x;
        expect(solveHermitian(x, A, b) == solve::Status::Success) << typeName<T>();
        expect(eq(x.rank(), 2UZ) && eq(x.extent(1), 2UZ)) << typeName<T>();
        expect(le(relativeError(x, x0), tolerance<T>(4))) << typeName<T>() << "two right-hand sides";

        Tensor<T> b1({4UZ});
        Tensor<T> x1Expected({4UZ});
        for (std::size_t i = 0UZ; i < 4UZ; ++i) {
            b1[i]         = b[i, 0UZ];
            x1Expected[i] = x0[i, 0UZ];
        }
        Tensor<T> x1;
        expect(solveHermitian(x1, A, b1) == solve::Status::Success) << typeName<T>();
        expect(eq(x1.rank(), 1UZ)) << typeName<T>() << "a vector right-hand side gives a vector";
        expect(le(relativeError(x1, x1Expected), tolerance<T>(4))) << typeName<T>() << "one right-hand side";
    } | ElementTypes{};

    "an indefinite matrix is refused; diagonal loading makes it definite"_test = []<typename T> {
        // eigenvalues -1 and 3
        const Tensor<T> A = matrix<T>(2, 2, {{1, 0}, {2, 0}, {2, 0}, {1, 0}});
        Tensor<T>       L;
        expect(cholesky(L, A) == solve::Status::NotPositiveDefinite) << typeName<T>();

        expect(cholesky(L, A, Real<T>{2}) == solve::Status::Success) << typeName<T>();
        Tensor<T> loaded = A;
        loaded[0, 0] += T{2};
        loaded[1, 1] += T{2};
        expect(le(relativeError(multiply(L, adjoint(L)), loaded), tolerance<T>(2))) << typeName<T>() << "L L^H = A + 2 I";
    } | ElementTypes{};

    "loading equals adding to the diagonal"_test = []<typename T> {
        const Tensor<T> L0     = chosenFactor<T>();
        const Tensor<T> A      = multiply(L0, adjoint(L0));
        const Tensor<T> b      = matrix<T>(4, 1, {{1, 0}, {2, -1}, {0, 1}, {-1, 0}});
        Tensor<T>       loaded = A;
        for (std::size_t i = 0UZ; i < 4UZ; ++i) {
            loaded[i, i] += T{3};
        }
        Tensor<T> x;
        Tensor<T> xLoaded;
        expect(solveHermitian(x, A, b, Real<T>{3}) == solve::Status::Success) << typeName<T>();
        expect(solveHermitian(xLoaded, loaded, b) == solve::Status::Success) << typeName<T>();
        expect(le(relativeError(x, xLoaded), tolerance<T>(4))) << typeName<T>();
    } | ElementTypes{};

    "invalid extents and non-finite entries are refused"_test = []<typename T> {
        Tensor<T>       L;
        Tensor<T>       x;
        const Tensor<T> rectangular({3UZ, 2UZ});
        expect(cholesky(L, rectangular) == solve::Status::InvalidInput) << typeName<T>();

        const Tensor<T> A = identity<T>(3);
        Tensor<T>       wrongLength({4UZ});
        wrongLength.fill(T{1});
        expect(solveHermitian(x, A, wrongLength) == solve::Status::InvalidInput) << typeName<T>();

        Tensor<T> withNaN = A;
        withNaN[2, 1]     = element<T>(std::numeric_limits<double>::quiet_NaN());
        expect(cholesky(L, withNaN) == solve::Status::InvalidInput) << typeName<T>();
        expect(x.size() == 0UZ) << typeName<T>() << "a refused solve leaves x untouched";
    } | ElementTypes{};

    "a factor with a non-finite entry is refused and leaves x as it was"_test = []<typename T> {
        const Tensor<T> L0 = chosenFactor<T>();
        const Tensor<T> x0 = matrix<T>(4, 1, {{1, 0}, {-2, 1}, {0, -1}, {3, 0}});
        const Tensor<T> b  = multiply(multiply(L0, adjoint(L0)), x0);

        Tensor<T> x;
        expect(choleskySolve(x, L0, b) == solve::Status::Success) << typeName<T>() << "a finite factor";
        expect(le(relativeError(x, x0), tolerance<T>(4))) << typeName<T>() << "a finite factor";

        const Tensor<T> solved = x;
        for (const double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()}) {
            for (const auto& [i, j] : {std::pair{2UZ, 1UZ}, std::pair{3UZ, 3UZ}}) { // below the diagonal and on it
                Tensor<T> L = L0;
                L[i, j]     = element<T>(bad);
                expect(choleskySolve(x, L, b) == solve::Status::InvalidInput) << typeName<T>() << bad << "at" << i << j;
                expect(std::ranges::equal(x, solved)) << typeName<T>() << bad << "at" << i << j << "x is left as it was";
            }
        }

        // the solve reads the lower triangle only
        Tensor<T> upper = L0;
        upper[0, 3]     = element<T>(std::numeric_limits<double>::quiet_NaN());
        expect(choleskySolve(x, upper, b) == solve::Status::Success) << typeName<T>() << "a non-finite entry above the diagonal";
        expect(le(relativeError(x, x0), tolerance<T>(4))) << typeName<T>() << "a non-finite entry above the diagonal";
    } | ElementTypes{};
};

const boost::ut::suite<"Householder QR and least squares"> _qr = [] {
    using namespace boost::ut;

    "Q R reconstructs A, Q has orthonormal columns, R is upper triangular"_test = []<typename T> {
        const Tensor<T> A = tallMatrix<T>();
        Tensor<T>       Q;
        Tensor<T>       R;
        expect(householderQr(Q, R, A) == solve::Status::Success) << typeName<T>();
        expect(eq(Q.extent(0), 6UZ) && eq(Q.extent(1), 3UZ) && eq(R.extent(0), 3UZ) && eq(R.extent(1), 3UZ)) << typeName<T>();
        expect(le(relativeError(multiply(Q, R), A), tolerance<T>(6))) << typeName<T>() << "A = Q R";
        expect(le(relativeError(multiply(adjoint(Q), Q), identity<T>(3)), tolerance<T>(6))) << typeName<T>() << "Q^H Q = I";
        expect(R[1, 0] == T{0} && R[2, 0] == T{0} && R[2, 1] == T{0}) << typeName<T>() << "R upper triangular";
    } | ElementTypes{};

    "a matrix scaled far below eps factors and solves as accurately as at unit scale"_test = []<typename T> {
        // At eps^2 the first entry of a reflector lies below eps. Below sqrt(denorm_min) the square of every entry underflows to zero.
        const Real<T>   eps = std::numeric_limits<Real<T>>::epsilon();
        const Tensor<T> x0  = matrix<T>(3, 1, {{1, -1}, {-2, 0}, {0, 3}});
        for (const Real<T> scale : {eps * eps, std::sqrt(std::numeric_limits<Real<T>>::denorm_min()) / Real<T>{16}}) {
            Tensor<T> A = tallMatrix<T>();
            for (auto& a : A) {
                a *= scale;
            }
            Tensor<T> Q;
            Tensor<T> R;
            expect(householderQr(Q, R, A) == solve::Status::Success) << typeName<T>() << scale;
            expect(le(relativeError(multiply(Q, R), A), tolerance<T>(6))) << typeName<T>() << scale << "A = Q R";
            expect(le(relativeError(multiply(adjoint(Q), Q), identity<T>(3)), tolerance<T>(6))) << typeName<T>() << scale << "Q^H Q = I";

            Tensor<T> x;
            expect(leastSquares(x, A, multiply(A, x0)) == solve::Status::Success) << typeName<T>() << scale;
            expect(le(relativeError(x, x0), tolerance<T>(6))) << typeName<T>() << scale << "the chosen solution";
        }
    } | ElementTypes{};

    "a consistent tall system gives its chosen solution"_test = []<typename T> {
        const Tensor<T> A  = tallMatrix<T>();
        const Tensor<T> x0 = matrix<T>(3, 1, {{1, -1}, {-2, 0}, {0, 3}});
        const Tensor<T> b  = multiply(A, x0);
        Tensor<T>       x;
        expect(leastSquares(x, A, b) == solve::Status::Success) << typeName<T>();
        expect(le(relativeError(x, x0), tolerance<T>(6))) << typeName<T>();
    } | ElementTypes{};

    "an inconsistent tall system satisfies the normal equations"_test = []<typename T> {
        const Tensor<T> A = tallMatrix<T>();
        const Tensor<T> b = matrix<T>(6, 1, {{1, 0}, {0, 1}, {2, 0}, {-1, 1}, {0, 0}, {3, -2}});
        Tensor<T>       x;
        expect(leastSquares(x, A, b) == solve::Status::Success) << typeName<T>();
        Tensor<T> residual = multiply(A, x);
        for (std::size_t i = 0UZ; i < 6UZ; ++i) {
            residual[i, 0UZ] -= b[i, 0UZ];
        }
        const Tensor<T> gradient = multiply(adjoint(A), residual); // A^H (A x - b) vanishes at the minimum
        const Tensor<T> scale    = multiply(adjoint(A), b);
        Real<T>         largest{0};
        Real<T>         reference{0};
        for (std::size_t i = 0UZ; i < 3UZ; ++i) {
            largest   = std::max(largest, static_cast<Real<T>>(std::abs(gradient[i, 0UZ])));
            reference = std::max(reference, static_cast<Real<T>>(std::abs(scale[i, 0UZ])));
        }
        expect(le(largest / reference, tolerance<T>(6))) << typeName<T>();
    } | ElementTypes{};

    "a rank-deficient matrix is reported singular"_test = []<typename T> {
        Tensor<T> A = tallMatrix<T>();
        for (std::size_t i = 0UZ; i < 6UZ; ++i) {
            A[i, 2UZ] = A[i, 0UZ] + A[i, 1UZ]; // the third column is the sum of the first two
        }
        Tensor<T> b({6UZ});
        b.fill(T{1});
        Tensor<T> x;
        expect(leastSquares(x, A, b) == solve::Status::Singular) << typeName<T>();

        const Tensor<T> wide({2UZ, 3UZ});
        Tensor<T>       Q;
        Tensor<T>       R;
        expect(householderQr(Q, R, wide) == solve::Status::InvalidInput) << typeName<T>() << "a wide matrix";
    } | ElementTypes{};

    "inverse times A is the identity"_test = []<typename T> {
        const Tensor<T> L0 = chosenFactor<T>();
        Tensor<T>       A  = multiply(L0, adjoint(L0));
        A[0, 3] += element<T>(2, 1); // A is not Hermitian
        Tensor<T> Ainv;
        expect(inverse(Ainv, A) == solve::Status::Success) << typeName<T>();
        expect(le(relativeError(multiply(A, Ainv), identity<T>(4)), tolerance<T>(4))) << typeName<T>() << "A A^-1 = I";
        expect(le(relativeError(multiply(Ainv, A), identity<T>(4)), tolerance<T>(4))) << typeName<T>() << "A^-1 A = I";

        const Tensor<T> singular = matrix<T>(2, 2, {{1, 1}, {2, 0}, {2, 2}, {4, 0}});
        expect(inverse(Ainv, singular) == solve::Status::Singular) << typeName<T>();
    } | ElementTypes{};
};

int main() { /* tests are automatically registered and executed */ return 0; }
