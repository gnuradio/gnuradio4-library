# Tensor mathematics and linear algebra

GNU Radio's `gr::Tensor` and `gr::TensorView` storage types are provided by
`gnuradio4-core`. Tensor arithmetic, linear algebra, and matrix decompositions
are provided by `gnuradio4-library` through the `gnuradio-algorithm` target.

```cmake
find_package(gnuradio4Library CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE gnuradio4::gnuradio-algorithm)
```

```cpp
#include <gnuradio-4.0/algorithm/math/SVD.hpp>
#include <gnuradio-4.0/algorithm/math/TensorMath.hpp>

using gr::math::TensorOps;
```

## GEMM and GEMV

The library supplies cache-blocked, vectorised matrix-matrix and matrix-vector
multiplication for small-to-medium dense tensors.

```cpp
gr::Tensor<float, 64, 64> A, B, C;

// C = A * B
gr::math::gemm(C, A, B);

// C = alpha * A * B + beta * C
gr::math::gemm(C, A, B, 1.0f, 0.5f);

// Explicit execution policy
gr::math::gemm<gr::math::TransposeOp::NoTrans,
               gr::math::TransposeOp::NoTrans>(
    gr::math::cpu_policy{}, C, A, B, 1.0f, 0.0f);

gr::Tensor<float, 64> x, y;

// y = A * x
gr::math::gemv(y, A, x);
```

## Reductions and element-wise operations

```cpp
gr::Tensor<double> t{{2, 3}, {1, 2, 3, 4, 5, 6}};

auto total  = TensorOps<double>::sum(t);
auto product = TensorOps<double>::product(t);
auto average = TensorOps<double>::mean(t);
auto variance = TensorOps<double>::variance(t);
auto standardDeviation = TensorOps<double>::std_dev(t);

gr::Tensor<float> a{{3}, {1, 2, 3}};
gr::Tensor<float> b{{3}, {4, 5, 6}};

TensorOps<float>::add_inplace(a, b);
TensorOps<float>::multiply_elementwise_inplace(a, b);
auto scaled = TensorOps<float>::multiply_scalar(a, 3.0f);
```

`TensorMath.hpp` also provides axis reductions, transpose and concatenation,
complex conjugation, vector and Frobenius norms, Hankel construction and
reconstruction, and Givens rotations.

## Singular value decomposition

`SVD.hpp` provides Jacobi and Golub-Reinsch implementations through
`gr::math::gesvd`. It supports full or truncated decompositions and
singular-values-only operation.

```cpp
gr::Tensor<double> A{{3, 2}, {1, 2, 3, 4, 5, 6}};
gr::Tensor<double> U, singularValues, V;

auto status = gr::math::gesvd(U, singularValues, V, A);
if (status != gr::math::svd::Status::Success) {
    // Handle non-convergence or invalid input.
}
```
