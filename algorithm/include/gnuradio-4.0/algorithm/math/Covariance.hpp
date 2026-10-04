#ifndef GNURADIO_COVARIANCE_HPP
#define GNURADIO_COVARIANCE_HPP

#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/algorithm/math/SVD.hpp>
#include <gnuradio-4.0/algorithm/math/TensorMath.hpp>

#include <cmath>
#include <complex>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

/**
 * The conjugated dot product, the outer product a b^H and a sample covariance accumulator
 * R = sum x x^H, for real and complex floating-point element types.
 */
namespace gr::math {

/**
 * @brief Conjugated dot product sum_i conj(a_i) * b_i of two rank-1 tensors of equal length.
 *
 * Equals dot(a, b) for a real element type. Throws std::runtime_error on another rank or length,
 * as dot does.
 */
template<TensorLike TensorA, TensorLike TensorB, typename T = typename std::remove_cvref_t<TensorA>::value_type>
requires TensorOf<TensorB, T>
[[nodiscard]] T vdot(const TensorA& a, const TensorB& b) {
    if (a.rank() != 1UZ || b.rank() != 1UZ || a.size() != b.size()) {
        throw std::runtime_error("vdot requires two rank-1 tensors of equal length");
    }
    T sum{0};
    for (std::size_t i = 0UZ; i < a.size(); ++i) {
        sum += detail::conj(a[i]) * b[i];
    }
    return sum;
}

/**
 * @brief Outer product C = a b^H of two rank-1 tensors, C[i, j] = a_i * conj(b_j).
 *
 * C is resized to a.size() x b.size(). Throws std::runtime_error when a or b is not rank 1.
 */
template<TensorLike TensorA, TensorLike TensorB, typename T = typename std::remove_cvref_t<TensorA>::value_type>
requires TensorOf<TensorB, T>
void outer(Tensor<T>& C, const TensorA& a, const TensorB& b) {
    if (a.rank() != 1UZ || b.rank() != 1UZ) {
        throw std::runtime_error("outer requires two rank-1 tensors");
    }
    C.resize({a.size(), b.size()});
    for (std::size_t i = 0UZ; i < a.size(); ++i) {
        for (std::size_t j = 0UZ; j < b.size(); ++j) {
            C[i, j] = a[i] * detail::conj(b[j]);
        }
    }
}

/**
 * @brief Accumulates the sample covariance R = sum x x^H of snapshots across N elements.
 *
 * Each snapshot x holds one sample per element, and R[i, j] estimates E{x_i conj(x_j)}. Only the
 * upper triangle is accumulated; matrix() mirrors it, so the result is Hermitian with a real
 * diagonal.
 *
 * Block averaging (the default) averages `blockSize` snapshots with equal weight. add() returns
 * true on the snapshot that completes a block, and the next add() starts a new block. A block size
 * of 0 never completes and averages every snapshot since the last reset().
 *
 * Exponential averaging with a forgetting factor lambda in (0, 1] weights the snapshot k places
 * back by lambda^k and divides by the sum of the weights, so the first snapshots are not biased
 * toward zero. add() returns true on every snapshot.
 */
template<typename T>
requires detail::FloatingElement<T>
class CovarianceAccumulator {
public:
    using value_type = T;
    using real_type  = gr::meta::fundamental_base_value_type_t<T>;

    enum class Averaging : std::uint8_t { Block, Exponential };

    explicit CovarianceAccumulator(std::size_t nElements = 0UZ, std::size_t blockSize = 0UZ) : _n(nElements), _blockSize(blockSize), _sum(nElements * nElements, T{0}) {}

    [[nodiscard]] std::size_t size() const noexcept { return _n; }
    [[nodiscard]] Averaging   averaging() const noexcept { return _averaging; }
    [[nodiscard]] std::size_t blockSize() const noexcept { return _blockSize; }
    [[nodiscard]] real_type   forgetting() const noexcept { return _forgetting; }

    /// Snapshots in the current estimate.
    [[nodiscard]] std::size_t count() const noexcept { return _count; }

    /// Selects block averaging over `blockSize` snapshots and resets.
    void setBlockAveraging(std::size_t blockSize) noexcept {
        _averaging  = Averaging::Block;
        _blockSize  = blockSize;
        _forgetting = real_type{1};
        reset();
    }

    /// Selects exponential averaging and resets. Returns false and changes nothing for a factor outside (0, 1].
    bool setExponentialAveraging(real_type forgetting) noexcept {
        if (!(forgetting > real_type{0} && forgetting <= real_type{1})) {
            return false;
        }
        _averaging  = Averaging::Exponential;
        _forgetting = forgetting;
        reset();
        return true;
    }

    /// Adds one snapshot of size() samples. Returns whether an estimate is complete; a snapshot of another length is ignored and returns false.
    bool add(std::span<const T> x) noexcept {
        if (x.size() != _n) {
            return false;
        }
        const bool exponential = _averaging == Averaging::Exponential;
        if (!exponential && _blockComplete) {
            reset();
        }
        const real_type decay = exponential ? _forgetting : real_type{1};
        for (std::size_t i = 0UZ; i < _n; ++i) {
            const T xi  = x[i];
            T*      row = _sum.data() + i * _n;
            row[i]      = decay * row[i] + T{squaredMagnitude(xi)};
            for (std::size_t j = i + 1UZ; j < _n; ++j) {
                row[j] = decay * row[j] + xi * detail::conj(x[j]);
            }
        }
        _weight = decay * _weight + real_type{1};
        ++_count;
        if (exponential) {
            return true;
        }
        _blockComplete = _blockSize != 0UZ && _count == _blockSize;
        return _blockComplete;
    }

    /// Adds one snapshot held in a contiguous rank-1 tensor.
    template<TensorLike TensorX>
    requires TensorOf<TensorX, T>
    bool add(const TensorX& x) noexcept {
        if (x.rank() != 1UZ || !x.is_contiguous()) {
            return false;
        }
        return add(std::span<const T>(x.data(), x.size()));
    }

    /// Writes the N x N Hermitian estimate into R; all zeros before the first snapshot.
    void matrix(Tensor<T>& R) const {
        R.resize({_n, _n});
        if (_count == 0UZ) {
            R.fill(T{0});
            return;
        }
        const real_type scale = real_type{1} / _weight;
        for (std::size_t i = 0UZ; i < _n; ++i) {
            const T* row = _sum.data() + i * _n;
            R[i, i]      = T{std::real(row[i]) * scale};
            for (std::size_t j = i + 1UZ; j < _n; ++j) {
                const T value = row[j] * scale;
                R[i, j]       = value;
                R[j, i]       = detail::conj(value);
            }
        }
    }

    [[nodiscard]] Tensor<T> matrix() const {
        Tensor<T> R;
        matrix(R);
        return R;
    }

    /// Clears the estimate and keeps the averaging mode.
    void reset() noexcept {
        std::ranges::fill(_sum, T{0});
        _weight        = real_type{0};
        _count         = 0UZ;
        _blockComplete = false;
    }

private:
    std::size_t    _n;
    std::size_t    _blockSize;
    real_type      _forgetting{1};
    Averaging      _averaging{Averaging::Block};
    std::vector<T> _sum; // n x n row-major; the upper triangle holds the weighted sum
    real_type      _weight{0};
    std::size_t    _count{0UZ};
    bool           _blockComplete{false};
};

} // namespace gr::math

#endif // GNURADIO_COVARIANCE_HPP
