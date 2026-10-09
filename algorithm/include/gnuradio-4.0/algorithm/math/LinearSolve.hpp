#ifndef GNURADIO_LINEAR_SOLVE_HPP
#define GNURADIO_LINEAR_SOLVE_HPP

#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/algorithm/math/SVD.hpp>
#include <gnuradio-4.0/algorithm/math/TensorMath.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

/**
 * Dense linear solvers over gr::Tensor for real and complex floating-point element types.
 *
 * A matrix is a rank-2 tensor. A right-hand side `b` is a rank-1 tensor of n elements or a rank-2
 * tensor of n rows, one system per column; the solution `x` takes the rank and the column count of
 * `b`. Every function reports a bad input through its solve::Status. An output is written only when
 * the status is Success.
 */
namespace gr::math {

namespace solve {
enum class Status : std::uint8_t {
    Success,
    InvalidInput,        /// wrong tensor rank, mismatched extents or a non-finite entry
    NotPositiveDefinite, /// a Cholesky pivot is zero, negative or not finite
    Singular             /// a triangular factor has a diagonal entry that is zero or negligible
};
} // namespace solve

namespace detail {

template<TensorLike TensorA>
[[nodiscard]] bool isSquareMatrix(const TensorA& A) noexcept {
    return A.rank() == 2UZ && A.extent(0) == A.extent(1);
}

/// Returns the number of systems in a right-hand side for n unknowns, or 0 when its extents do not fit.
template<TensorLike TensorB>
[[nodiscard]] std::size_t rhsColumns(const TensorB& b, std::size_t n) noexcept {
    if (b.rank() == 1UZ && b.extent(0) == n) {
        return 1UZ;
    }
    if (b.rank() == 2UZ && b.extent(0) == n) {
        return b.extent(1);
    }
    return 0UZ;
}

/// Copies a right-hand side into an n x k work matrix.
template<typename T, TensorLike TensorB>
[[nodiscard]] Tensor<T> rhsMatrix(const TensorB& b, std::size_t n, std::size_t k) {
    Tensor<T> X({n, k});
    for (std::size_t i = 0UZ; i < n; ++i) {
        for (std::size_t c = 0UZ; c < k; ++c) {
            if (b.rank() == 1UZ) {
                X[i, c] = b[i];
            } else {
                X[i, c] = b[i, c];
            }
        }
    }
    return X;
}

/// Writes the first rows of an n x k work matrix into x, with the rank of the right-hand side.
template<typename T>
void storeSolution(Tensor<T>& x, const Tensor<T>& X, std::size_t rows, std::size_t rhsRank) {
    const std::size_t k = X.extent(1);
    if (rhsRank == 1UZ) {
        x.resize({rows});
        for (std::size_t i = 0UZ; i < rows; ++i) {
            x[i] = X[i, 0UZ];
        }
    } else {
        x.resize({rows, k});
        for (std::size_t i = 0UZ; i < rows; ++i) {
            for (std::size_t c = 0UZ; c < k; ++c) {
                x[i, c] = X[i, c];
            }
        }
    }
}

/// Solves R x = y in place for the leading n x n upper triangle of R, column by column of Y.
template<typename T>
void backSubstitute(const Tensor<T>& R, Tensor<T>& Y, std::size_t n) {
    const std::size_t k = Y.extent(1);
    for (std::size_t c = 0UZ; c < k; ++c) {
        for (std::size_t i = n; i-- > 0UZ;) {
            T sum = Y[i, c];
            for (std::size_t j = i + 1UZ; j < n; ++j) {
                sum -= R[i, j] * Y[j, c];
            }
            Y[i, c] = sum / R[i, i];
        }
    }
}

/// Factors an m x n work matrix with m >= n in place: R above the diagonal and the Householder tails below it.
template<typename T, typename RealT = gr::meta::fundamental_base_value_type_t<T>>
void householderFactor(Tensor<T>& W, std::vector<RealT>& tau, std::vector<T>& rDiagonal) {
    const std::size_t m = W.extent(0);
    const std::size_t n = W.extent(1);
    tau.assign(n, RealT{0});
    rDiagonal.assign(n, T{0});
    std::vector<T> column(m);

    for (std::size_t k = 0UZ; k < n; ++k) {
        const std::size_t length = m - k;
        for (std::size_t i = 0UZ; i < length; ++i) {
            column[i] = W[k + i, k];
        }
        const T x0  = column[0];
        std::ignore = householderVector(column.data(), length, tau[k]);
        // householderVector stores v0 = x0 - beta in column[0], so the diagonal entry of R is x0 - v0
        rDiagonal[k] = (tau[k] == RealT{0}) ? x0 : x0 - column[0];

        W[k, k] = column[0];
        for (std::size_t i = 1UZ; i < length; ++i) {
            W[k + i, k] = column[i];
        }
        if (k + 1UZ < n && length > 1UZ) {
            applyHouseholderLeft(W, column.data() + 1, tau[k], k, k + 1UZ, length - 1UZ);
        }
    }
}

} // namespace detail

/**
 * @brief Cholesky factor A + loading * I = L L^H of a Hermitian positive-definite matrix.
 *
 * Reads the lower triangle of A and the real part of its diagonal. `loading` is added to every
 * diagonal entry before the factorization. L is n x n and lower triangular with a real, positive
 * diagonal and zeros above it.
 */
template<TensorLike TensorA, typename T = typename std::remove_cvref_t<TensorA>::value_type>
requires detail::FloatingElement<T>
[[nodiscard]] solve::Status cholesky(Tensor<T>& L, const TensorA& A, gr::meta::fundamental_base_value_type_t<T> loading = {}) {
    using RealT = gr::meta::fundamental_base_value_type_t<T>;
    if (!detail::isSquareMatrix(A) || !isFinite(A) || !std::isfinite(loading)) {
        return solve::Status::InvalidInput;
    }
    const std::size_t n = A.extent(0);
    Tensor<T>         F({n, n});
    F.fill(T{0});

    for (std::size_t j = 0UZ; j < n; ++j) {
        RealT pivot = std::real(A[j, j]) + loading;
        for (std::size_t k = 0UZ; k < j; ++k) {
            pivot -= squaredMagnitude(F[j, k]);
        }
        if (!(pivot > RealT{0}) || !std::isfinite(pivot)) {
            return solve::Status::NotPositiveDefinite;
        }
        const RealT diagonal = std::sqrt(pivot);
        F[j, j]              = T{diagonal};
        for (std::size_t i = j + 1UZ; i < n; ++i) {
            T sum = A[i, j];
            for (std::size_t k = 0UZ; k < j; ++k) {
                sum -= F[i, k] * detail::conj(F[j, k]);
            }
            F[i, j] = sum / diagonal;
        }
    }
    L = std::move(F);
    return solve::Status::Success;
}

namespace detail {

/// Solves L L^H x = b for the lower triangle of an n x n factor L and k right-hand sides, and writes x with the rank of b.
template<typename T, TensorLike TensorL, TensorLike TensorB>
void choleskySubstitute(Tensor<T>& x, const TensorL& L, const TensorB& b, std::size_t k) {
    const std::size_t n = L.extent(0);
    Tensor<T>         X = rhsMatrix<T>(b, n, k);
    for (std::size_t c = 0UZ; c < k; ++c) {
        for (std::size_t i = 0UZ; i < n; ++i) { // L y = b
            T sum = X[i, c];
            for (std::size_t j = 0UZ; j < i; ++j) {
                sum -= L[i, j] * X[j, c];
            }
            X[i, c] = sum / L[i, i];
        }
        for (std::size_t i = n; i-- > 0UZ;) { // L^H x = y
            T sum = X[i, c];
            for (std::size_t j = i + 1UZ; j < n; ++j) {
                sum -= conj(L[j, i]) * X[j, c];
            }
            X[i, c] = sum / conj(L[i, i]);
        }
    }
    storeSolution(x, X, n, b.rank());
}

} // namespace detail

/**
 * @brief Solves L L^H x = b for a Cholesky factor L.
 *
 * Forward substitution with L, then back substitution with L^H. Reads the lower triangle of L.
 * Returns InvalidInput when an entry of that triangle is not finite.
 */
template<TensorLike TensorL, TensorLike TensorB, typename T = typename std::remove_cvref_t<TensorL>::value_type>
requires detail::FloatingElement<T>
[[nodiscard]] solve::Status choleskySolve(Tensor<T>& x, const TensorL& L, const TensorB& b) {
    if (!detail::isSquareMatrix(L)) {
        return solve::Status::InvalidInput;
    }
    const std::size_t n = L.extent(0);
    const std::size_t k = detail::rhsColumns(b, n);
    if (k == 0UZ || !isFinite(b)) {
        return solve::Status::InvalidInput;
    }
    for (std::size_t i = 0UZ; i < n; ++i) {
        for (std::size_t j = 0UZ; j <= i; ++j) {
            const T entry = L[i, j];
            if (!std::isfinite(std::real(entry)) || !std::isfinite(std::imag(entry))) {
                return solve::Status::InvalidInput;
            }
        }
    }
    for (std::size_t i = 0UZ; i < n; ++i) {
        if (L[i, i] == T{0}) {
            return solve::Status::Singular;
        }
    }
    detail::choleskySubstitute(x, L, b, k);
    return solve::Status::Success;
}

/**
 * @brief Solves (A + loading * I) x = b for a Hermitian positive-definite A through its Cholesky factor.
 */
template<TensorLike TensorA, TensorLike TensorB, typename T = typename std::remove_cvref_t<TensorA>::value_type>
requires detail::FloatingElement<T>
[[nodiscard]] solve::Status solveHermitian(Tensor<T>& x, const TensorA& A, const TensorB& b, gr::meta::fundamental_base_value_type_t<T> loading = {}) {
    Tensor<T>           L;
    const solve::Status status = cholesky(L, A, loading);
    if (status != solve::Status::Success) {
        return status;
    }
    const std::size_t k = detail::rhsColumns(b, L.extent(0));
    if (k == 0UZ || !isFinite(b)) {
        return solve::Status::InvalidInput;
    }
    // a factor from cholesky has finite entries and a positive diagonal
    detail::choleskySubstitute(x, L, b, k);
    return solve::Status::Success;
}

/**
 * @brief Thin Householder QR factorization A = Q R of an m x n matrix with m >= n.
 *
 * Q is m x n with orthonormal columns and R is n x n upper triangular. A rank-deficient A still
 * factors; leastSquares reports it as Singular.
 */
template<TensorLike TensorA, typename T = typename std::remove_cvref_t<TensorA>::value_type>
requires detail::FloatingElement<T>
[[nodiscard]] solve::Status householderQr(Tensor<T>& Q, Tensor<T>& R, const TensorA& A) {
    using RealT = gr::meta::fundamental_base_value_type_t<T>;
    if (A.rank() != 2UZ || A.extent(0) < A.extent(1) || !isFinite(A)) {
        return solve::Status::InvalidInput;
    }
    const std::size_t m = A.extent(0);
    const std::size_t n = A.extent(1);
    Tensor<T>         W({m, n});
    for (std::size_t i = 0UZ; i < m; ++i) {
        for (std::size_t j = 0UZ; j < n; ++j) {
            W[i, j] = A[i, j];
        }
    }
    std::vector<RealT> tau;
    std::vector<T>     rDiagonal;
    detail::householderFactor(W, tau, rDiagonal);

    Tensor<T> Rout({n, n});
    Rout.fill(T{0});
    for (std::size_t i = 0UZ; i < n; ++i) {
        Rout[i, i] = rDiagonal[i];
        for (std::size_t j = i + 1UZ; j < n; ++j) {
            Rout[i, j] = W[i, j];
        }
    }
    detail::accumulateU(W, tau, Q);
    R = std::move(Rout);
    return solve::Status::Success;
}

/**
 * @brief Least-squares solution of A x = b for an m x n matrix A with m >= n.
 *
 * Minimizes ||A x - b|| through a Householder QR factorization: x solves R x = Q^H b. A square A
 * gives the exact solution. Returns Singular when a diagonal entry of R is at or below
 * eps * max(m, n) times the largest one, that is when A is rank-deficient to working precision.
 */
template<TensorLike TensorA, TensorLike TensorB, typename T = typename std::remove_cvref_t<TensorA>::value_type>
requires detail::FloatingElement<T>
[[nodiscard]] solve::Status leastSquares(Tensor<T>& x, const TensorA& A, const TensorB& b) {
    using RealT = gr::meta::fundamental_base_value_type_t<T>;
    if (A.rank() != 2UZ || A.extent(0) < A.extent(1) || !isFinite(A)) {
        return solve::Status::InvalidInput;
    }
    const std::size_t m = A.extent(0);
    const std::size_t n = A.extent(1);
    const std::size_t k = detail::rhsColumns(b, m);
    if (k == 0UZ || !isFinite(b)) {
        return solve::Status::InvalidInput;
    }

    Tensor<T> W({m, n});
    for (std::size_t i = 0UZ; i < m; ++i) {
        for (std::size_t j = 0UZ; j < n; ++j) {
            W[i, j] = A[i, j];
        }
    }
    std::vector<RealT> tau;
    std::vector<T>     rDiagonal;
    detail::householderFactor(W, tau, rDiagonal);

    RealT largest{0};
    for (const T& r : rDiagonal) {
        largest = std::max(largest, static_cast<RealT>(std::abs(r)));
    }
    const RealT negligible = std::numeric_limits<RealT>::epsilon() * static_cast<RealT>(m) * largest;
    for (const T& r : rDiagonal) {
        if (!(static_cast<RealT>(std::abs(r)) > negligible)) {
            return solve::Status::Singular;
        }
    }

    // Y = Q^H b: apply H_0, H_1, ... in order; each reflector is Hermitian
    Tensor<T> Y = detail::rhsMatrix<T>(b, m, k);
    for (std::size_t j = 0UZ; j < n; ++j) {
        if (tau[j] == RealT{0}) {
            continue;
        }
        for (std::size_t c = 0UZ; c < k; ++c) {
            T dot = Y[j, c];
            for (std::size_t i = j + 1UZ; i < m; ++i) {
                dot += detail::conj(W[i, j]) * Y[i, c];
            }
            const T scale = tau[j] * dot;
            Y[j, c] -= scale;
            for (std::size_t i = j + 1UZ; i < m; ++i) {
                Y[i, c] -= scale * W[i, j];
            }
        }
    }

    for (std::size_t i = 0UZ; i < n; ++i) {
        W[i, i] = rDiagonal[i];
    }
    detail::backSubstitute(W, Y, n);
    detail::storeSolution(x, Y, n, b.rank());
    return solve::Status::Success;
}

/**
 * @brief Inverse of a square matrix, solved column by column against the identity through leastSquares.
 *
 * Returns Singular when A is singular to working precision.
 */
template<TensorLike TensorA, typename T = typename std::remove_cvref_t<TensorA>::value_type>
requires detail::FloatingElement<T>
[[nodiscard]] solve::Status inverse(Tensor<T>& Ainv, const TensorA& A) {
    if (!detail::isSquareMatrix(A)) {
        return solve::Status::InvalidInput;
    }
    const std::size_t n = A.extent(0);
    Tensor<T>         identity({n, n});
    identity.fill(T{0});
    for (std::size_t i = 0UZ; i < n; ++i) {
        identity[i, i] = T{1};
    }
    return leastSquares(Ainv, A, identity);
}

} // namespace gr::math

#endif // GNURADIO_LINEAR_SOLVE_HPP
