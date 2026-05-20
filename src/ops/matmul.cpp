#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "../broadcast.h"
#include "../ops.h"
#include "../tensor_impl.h"

namespace tae::ops {

namespace {

// Blocked GEMM. Input strides are arbitrary so we can matmul transposed views
// without first materializing a contiguous copy. The output is always
// freshly-allocated and contiguous, so its strides are (N, 1).
//
// Loop nest: outer block over (i, j, k); inner loop has K as the innermost
// axis with the accumulator pinned in a register, which is the one trick that
// makes a hand-rolled GEMM not embarrassing.
void matmul_2d(const float* A, const float* B, float* C,
               std::int64_t M, std::int64_t K, std::int64_t N,
               std::int64_t a_sM, std::int64_t a_sK,
               std::int64_t b_sK, std::int64_t b_sN,
               std::int64_t c_sM, std::int64_t c_sN) {
    constexpr std::int64_t BM = 64, BN = 64, BK = 32;

    // Zero the output block.
    for (std::int64_t i = 0; i < M; ++i)
        for (std::int64_t j = 0; j < N; ++j)
            C[i * c_sM + j * c_sN] = 0.0f;

    for (std::int64_t i0 = 0; i0 < M; i0 += BM) {
        const std::int64_t i1 = std::min(i0 + BM, M);
        for (std::int64_t j0 = 0; j0 < N; j0 += BN) {
            const std::int64_t j1 = std::min(j0 + BN, N);
            for (std::int64_t k0 = 0; k0 < K; k0 += BK) {
                const std::int64_t k1 = std::min(k0 + BK, K);
                for (std::int64_t i = i0; i < i1; ++i) {
                    for (std::int64_t j = j0; j < j1; ++j) {
                        float acc = C[i * c_sM + j * c_sN];
                        for (std::int64_t k = k0; k < k1; ++k) {
                            acc += A[i * a_sM + k * a_sK] * B[k * b_sK + j * b_sN];
                        }
                        C[i * c_sM + j * c_sN] = acc;
                    }
                }
            }
        }
    }
}

}  // namespace

TensorImpl* matmul(const TensorImpl* a, const TensorImpl* b) {
    if (a->dtype() != DType::Float32 || b->dtype() != DType::Float32) {
        throw std::invalid_argument("matmul: only Float32 is supported in v1");
    }
    if (a->ndim() < 2 || b->ndim() < 2) {
        throw std::invalid_argument("matmul: inputs must be at least 2D in v1");
    }
    const std::int64_t M  = a->shape()[static_cast<std::size_t>(a->ndim() - 2)];
    const std::int64_t Ka = a->shape()[static_cast<std::size_t>(a->ndim() - 1)];
    const std::int64_t Kb = b->shape()[static_cast<std::size_t>(b->ndim() - 2)];
    const std::int64_t N  = b->shape()[static_cast<std::size_t>(b->ndim() - 1)];
    if (Ka != Kb) {
        throw std::invalid_argument("matmul: inner dimensions must match");
    }

    // Split off batch shapes (everything but the last two dims).
    Dims a_batch, b_batch;
    for (std::size_t d = 0; d + 2 < static_cast<std::size_t>(a->ndim()) + 0; ++d) {
        a_batch.push_back(a->shape()[d]);
    }
    for (std::size_t d = 0; d + 2 < static_cast<std::size_t>(b->ndim()) + 0; ++d) {
        b_batch.push_back(b->shape()[d]);
    }
    const Dims batch_shape = broadcast_shape(a_batch, b_batch);

    // Build the broadcast targets — batch_shape + (M, K) for `a`, batch_shape + (K, N) for `b`.
    Dims a_target = batch_shape;
    a_target.push_back(M);
    a_target.push_back(Ka);
    Dims b_target = batch_shape;
    b_target.push_back(Kb);
    b_target.push_back(N);

    TensorImpl* a_view = (a->shape() == a_target) ? const_cast<TensorImpl*>(a)
                                                  : broadcast_to(a, a_target);
    TensorImpl* b_view = (b->shape() == b_target) ? const_cast<TensorImpl*>(b)
                                                  : broadcast_to(b, b_target);
    if (a_view == a) a_view->incref();
    if (b_view == b) b_view->incref();

    Dims out_shape = batch_shape;
    out_shape.push_back(M);
    out_shape.push_back(N);
    auto* out = TensorImpl::make_contiguous(out_shape, DType::Float32);

    const auto* a_data   = static_cast<const float*>(a_view->storage()->data());
    const auto* b_data   = static_cast<const float*>(b_view->storage()->data());
    auto*       out_data = static_cast<float*>(out->storage()->data());

    const std::int64_t batch_rank = static_cast<std::int64_t>(batch_shape.size());
    std::int64_t batch_total = 1;
    for (auto s : batch_shape) batch_total *= s;

    const std::int64_t a_sM = a_view->strides()[static_cast<std::size_t>(a_view->ndim() - 2)];
    const std::int64_t a_sK = a_view->strides()[static_cast<std::size_t>(a_view->ndim() - 1)];
    const std::int64_t b_sK = b_view->strides()[static_cast<std::size_t>(b_view->ndim() - 2)];
    const std::int64_t b_sN = b_view->strides()[static_cast<std::size_t>(b_view->ndim() - 1)];
    const std::int64_t c_sM = out->strides()[static_cast<std::size_t>(out->ndim() - 2)];
    const std::int64_t c_sN = out->strides()[static_cast<std::size_t>(out->ndim() - 1)];

    Dims batch_idx(static_cast<std::size_t>(batch_rank), 0);
    for (std::int64_t k = 0; k < batch_total; ++k) {
        std::int64_t a_base = a_view->offset();
        std::int64_t b_base = b_view->offset();
        std::int64_t c_base = out->offset();
        for (std::int64_t d = 0; d < batch_rank; ++d) {
            a_base += batch_idx[static_cast<std::size_t>(d)]
                      * a_view->strides()[static_cast<std::size_t>(d)];
            b_base += batch_idx[static_cast<std::size_t>(d)]
                      * b_view->strides()[static_cast<std::size_t>(d)];
            c_base += batch_idx[static_cast<std::size_t>(d)]
                      * out->strides()[static_cast<std::size_t>(d)];
        }

        matmul_2d(a_data + a_base, b_data + b_base, out_data + c_base,
                  M, Ka, N, a_sM, a_sK, b_sK, b_sN, c_sM, c_sN);

        for (std::int64_t d = batch_rank - 1; d >= 0; --d) {
            if (++batch_idx[static_cast<std::size_t>(d)]
                < batch_shape[static_cast<std::size_t>(d)]) break;
            batch_idx[static_cast<std::size_t>(d)] = 0;
        }
    }

    a_view->decref();
    b_view->decref();
    return out;
}

}  // namespace tae::ops
