#include "test_main.h"

#include <cstdint>

#include "ops.h"
#include "tensor_impl.h"

using namespace tae;
using namespace tae::ops;

namespace {

TensorImpl* make_filled(const Dims& shape, std::initializer_list<float> vals) {
    auto* t = TensorImpl::make_contiguous(shape, DType::Float32);
    auto* data = static_cast<float*>(t->storage()->data());
    std::int64_t i = 0;
    for (float v : vals) data[i++] = v;
    return t;
}

}  // namespace

// @spec OPS-050, OPS-051
TEST(matmul, 2d_basic) {
    // A: 2x3, B: 3x2
    auto* A = make_filled(Dims{2, 3}, {1, 2, 3, 4, 5, 6});
    auto* B = make_filled(Dims{3, 2}, {7, 8, 9, 10, 11, 12});
    auto* C = matmul(A, B);
    CHECK(C->shape() == (Dims{2, 2}));
    const auto* d = static_cast<const float*>(C->storage()->data());
    // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    CHECK_NEAR(d[0],  58.0f, 1e-4f);
    CHECK_NEAR(d[1],  64.0f, 1e-4f);
    CHECK_NEAR(d[2], 139.0f, 1e-4f);
    CHECK_NEAR(d[3], 154.0f, 1e-4f);
    C->decref(); B->decref(); A->decref();
}

// @spec OPS-052
TEST(matmul, mismatched_inner_dims_raise) {
    auto* A = TensorImpl::make_contiguous(Dims{2, 3}, DType::Float32);
    auto* B = TensorImpl::make_contiguous(Dims{4, 2}, DType::Float32);
    CHECK_THROWS(matmul(A, B));
    B->decref(); A->decref();
}

// @spec OPS-050 (matmul with transposed view — covers the strided GEMM path)
TEST(matmul, transposed_view_input) {
    // A: 3x2 = [[1,2],[3,4],[5,6]]; A.T: 2x3 = [[1,3,5],[2,4,6]]
    // B: 3x2 = [[7,8],[9,10],[11,12]]
    // A.T @ B: 2x2 = [[1*7+3*9+5*11, 1*8+3*10+5*12], [2*7+4*9+6*11, 2*8+4*10+6*12]]
    //              = [[89, 98], [116, 128]]
    auto* A  = make_filled(Dims{3, 2}, {1, 2, 3, 4, 5, 6});
    auto* AT = A->transpose(0, 1);
    auto* B  = make_filled(Dims{3, 2}, {7, 8, 9, 10, 11, 12});
    auto* C  = matmul(AT, B);
    CHECK(C->shape() == (Dims{2, 2}));
    const auto* d = static_cast<const float*>(C->storage()->data());
    CHECK_NEAR(d[0],  89.0f, 1e-4f);
    CHECK_NEAR(d[1],  98.0f, 1e-4f);
    CHECK_NEAR(d[2], 116.0f, 1e-4f);
    CHECK_NEAR(d[3], 128.0f, 1e-4f);
    C->decref(); B->decref(); AT->decref(); A->decref();
}

// @spec OPS-050 (batched matmul)
TEST(matmul, batched_2x2x3_times_2x3x2) {
    // Two parallel matmuls; each 2x3 @ 3x2 → 2x2.
    auto* A = make_filled(Dims{2, 2, 3}, {
        1, 2, 3, 4, 5, 6,        // batch 0
        7, 8, 9, 10, 11, 12,     // batch 1
    });
    auto* B = make_filled(Dims{2, 3, 2}, {
        1, 0, 0, 1, 1, 1,        // batch 0
        2, 0, 0, 2, 1, 1,        // batch 1
    });
    auto* C = matmul(A, B);
    CHECK(C->shape() == (Dims{2, 2, 2}));
    const auto* d = static_cast<const float*>(C->storage()->data());
    // batch 0: A0 = [[1,2,3],[4,5,6]]; B0 = [[1,0],[0,1],[1,1]]
    //   row 0: [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
    //   row 1: [4*1+5*0+6*1, 4*0+5*1+6*1] = [10, 11]
    CHECK_NEAR(d[0], 4.0f, 1e-4f);
    CHECK_NEAR(d[1], 5.0f, 1e-4f);
    CHECK_NEAR(d[2], 10.0f, 1e-4f);
    CHECK_NEAR(d[3], 11.0f, 1e-4f);
    // batch 1: A1 = [[7,8,9],[10,11,12]]; B1 = [[2,0],[0,2],[1,1]]
    //   row 0: [7*2+8*0+9*1, 7*0+8*2+9*1] = [23, 25]
    //   row 1: [10*2+11*0+12*1, 10*0+11*2+12*1] = [32, 34]
    CHECK_NEAR(d[4], 23.0f, 1e-4f);
    CHECK_NEAR(d[5], 25.0f, 1e-4f);
    CHECK_NEAR(d[6], 32.0f, 1e-4f);
    CHECK_NEAR(d[7], 34.0f, 1e-4f);
    C->decref(); B->decref(); A->decref();
}

// @spec OPS-050 (batched × 2D — leading dims broadcast)
TEST(matmul, batched_a_times_2d_b) {
    // A: (2, 2, 3), B: (3, 2). Result: (2, 2, 2) — B is broadcast across batch.
    auto* A = make_filled(Dims{2, 2, 3}, {
        1, 2, 3, 4, 5, 6,
        1, 1, 1, 1, 1, 1,
    });
    auto* B = make_filled(Dims{3, 2}, {1, 0, 0, 1, 1, 1});
    auto* C = matmul(A, B);
    CHECK(C->shape() == (Dims{2, 2, 2}));
    const auto* d = static_cast<const float*>(C->storage()->data());
    // batch 0: as test above → [4, 5, 10, 11]
    CHECK_NEAR(d[0], 4.0f, 1e-4f);
    CHECK_NEAR(d[3], 11.0f, 1e-4f);
    // batch 1: A1 = [[1,1,1],[1,1,1]], B = [[1,0],[0,1],[1,1]] → [[2,2],[2,2]]
    CHECK_NEAR(d[4], 2.0f, 1e-4f);
    CHECK_NEAR(d[7], 2.0f, 1e-4f);
    C->decref(); B->decref(); A->decref();
}

// Large enough to stress the blocking (>BM, >BN, >BK).
TEST(matmul, large_dims_exercise_blocking) {
    const std::int64_t M = 100, K = 80, N = 60;
    auto* A = TensorImpl::make_contiguous(Dims{M, K}, DType::Float32);
    auto* B = TensorImpl::make_contiguous(Dims{K, N}, DType::Float32);
    auto* Ad = static_cast<float*>(A->storage()->data());
    auto* Bd = static_cast<float*>(B->storage()->data());
    // A[i,k] = 1.0; B[k,j] = 1.0 → C[i,j] = K
    for (std::int64_t i = 0; i < M * K; ++i) Ad[i] = 1.0f;
    for (std::int64_t i = 0; i < K * N; ++i) Bd[i] = 1.0f;
    auto* C = matmul(A, B);
    const auto* Cd = static_cast<const float*>(C->storage()->data());
    for (std::int64_t i = 0; i < M * N; ++i) CHECK_NEAR(Cd[i], static_cast<float>(K), 1e-3f);
    C->decref(); B->decref(); A->decref();
}
