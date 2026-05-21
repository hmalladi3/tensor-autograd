#include "test_main.h"

#include <cstdint>

#include "ops.h"
#include "tensor_impl.h"

using namespace tae;
using namespace tae::ops;

// @spec OPS-070
TEST(constructors, full_fills_value) {
    auto* t = full(Dims{2, 3}, DType::Float32, 7.5);
    CHECK(t->shape() == (Dims{2, 3}));
    CHECK(t->is_contiguous());
    const auto* d = static_cast<const float*>(t->storage()->data());
    for (int i = 0; i < 6; ++i) CHECK_NEAR(d[i], 7.5f, 1e-6f);
    t->decref();
}

// @spec OPS-071
TEST(constructors, arange_produces_stepped_sequence) {
    auto* t = arange(0.0, 5.0, 1.0, DType::Float32);
    CHECK(t->shape() == (Dims{5}));
    const auto* d = static_cast<const float*>(t->storage()->data());
    for (int i = 0; i < 5; ++i) CHECK_NEAR(d[i], static_cast<float>(i), 1e-6f);
    t->decref();

    auto* t2 = arange(2.0, 10.0, 2.0, DType::Float32);
    CHECK(t2->shape() == (Dims{4}));
    const auto* d2 = static_cast<const float*>(t2->storage()->data());
    CHECK_NEAR(d2[0], 2.0f, 1e-6f);
    CHECK_NEAR(d2[3], 8.0f, 1e-6f);
    t2->decref();
}

// @spec OPS-072, FFI-055, FFI-057
TEST(random, uniform_in_range_and_reproducible_via_seed) {
    seed(42);
    auto* t1 = random(OP_UNIFORM, Dims{1000}, DType::Float32, 0.0, 1.0);
    const auto* d1 = static_cast<const float*>(t1->storage()->data());
    for (int i = 0; i < 1000; ++i) {
        CHECK(d1[i] >= 0.0f);
        CHECK(d1[i] < 1.0f);
    }
    // Reseed → same numbers.
    seed(42);
    auto* t2 = random(OP_UNIFORM, Dims{1000}, DType::Float32, 0.0, 1.0);
    const auto* d2 = static_cast<const float*>(t2->storage()->data());
    for (int i = 0; i < 1000; ++i) CHECK_NEAR(d1[i], d2[i], 0.0f);

    // No reseed between calls → different numbers.
    auto* t3 = random(OP_UNIFORM, Dims{1000}, DType::Float32, 0.0, 1.0);
    const auto* d3 = static_cast<const float*>(t3->storage()->data());
    bool any_different = false;
    for (int i = 0; i < 1000; ++i) if (d1[i] != d3[i]) { any_different = true; break; }
    CHECK(any_different);

    t3->decref(); t2->decref(); t1->decref();
}

// @spec OPS-073
TEST(random, normal_mean_and_std_are_in_ballpark) {
    seed(7);
    const std::int64_t n = 50000;
    auto* t = random(OP_NORMAL, Dims{n}, DType::Float32, 0.0, 1.0);
    const auto* d = static_cast<const float*>(t->storage()->data());
    double sum = 0, sq = 0;
    for (std::int64_t i = 0; i < n; ++i) { sum += d[i]; sq += d[i] * d[i]; }
    const double mean = sum / static_cast<double>(n);
    const double var  = sq / static_cast<double>(n) - mean * mean;
    // Generous bounds — sampling noise.
    CHECK(std::fabs(mean) < 0.05);
    CHECK(std::fabs(var - 1.0) < 0.05);
    t->decref();
}

// @spec OPS-060 (cast same dtype is no-op view)
TEST(cast, same_dtype_returns_same_object) {
    auto* a = TensorImpl::make_contiguous(Dims{4}, DType::Float32);
    auto* c = cast(a, DType::Float32);
    CHECK_EQ(c, a);
    CHECK_EQ(a->refcount(), 2);
    c->decref(); a->decref();
}

// @spec TENS-050, TENS-051
TEST(cast, converts_elementwise) {
    auto* f = full(Dims{4}, DType::Float32, 3.7);
    auto* i = cast(f, DType::Int64);
    CHECK_EQ(i->dtype(), DType::Int64);
    const auto* d = static_cast<const std::int64_t*>(i->storage()->data());
    for (int k = 0; k < 4; ++k) CHECK_EQ(d[k], static_cast<std::int64_t>(3));
    i->decref(); f->decref();
}

// @spec OPS-062
TEST(gather, picks_per_row_indices) {
    auto* src = TensorImpl::make_contiguous(Dims{3, 4}, DType::Float32);
    auto* sd  = static_cast<float*>(src->storage()->data());
    // src = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    for (int i = 0; i < 12; ++i) sd[i] = static_cast<float>(i);

    auto* idx = TensorImpl::make_contiguous(Dims{3, 2}, DType::Int64);
    auto* id  = static_cast<std::int64_t*>(idx->storage()->data());
    // For each row, pick columns: row 0 → (0,3); row 1 → (2,1); row 2 → (3,0)
    id[0] = 0; id[1] = 3;
    id[2] = 2; id[3] = 1;
    id[4] = 3; id[5] = 0;

    auto* out = gather(src, /*dim=*/1, idx);
    CHECK(out->shape() == (Dims{3, 2}));
    const auto* od = static_cast<const float*>(out->storage()->data());
    CHECK_NEAR(od[0], 0.0f, 1e-6f);  CHECK_NEAR(od[1], 3.0f, 1e-6f);
    CHECK_NEAR(od[2], 6.0f, 1e-6f);  CHECK_NEAR(od[3], 5.0f, 1e-6f);
    CHECK_NEAR(od[4], 11.0f, 1e-6f); CHECK_NEAR(od[5], 8.0f, 1e-6f);
    out->decref(); idx->decref(); src->decref();
}

// @spec OPS-063
TEST(gather, out_of_range_index_throws) {
    auto* src = TensorImpl::make_contiguous(Dims{3, 4}, DType::Float32);
    auto* idx = TensorImpl::make_contiguous(Dims{3, 1}, DType::Int64);
    static_cast<std::int64_t*>(idx->storage()->data())[0] = 99;
    CHECK_THROWS(gather(src, 1, idx));
    idx->decref(); src->decref();
}

// @spec FFI-050, FFI-051 (op_inplace AXPY semantics)
TEST(inplace_axpy, mutates_dst_in_place_and_preserves_handle) {
    auto* dst = full(Dims{4}, DType::Float32, 1.0);
    auto* src = full(Dims{4}, DType::Float32, 2.0);
    auto* dst_handle_before = dst;
    inplace(OP_AXPY, dst, src, 3.0);  // dst += 3 * src → 1 + 6 = 7
    CHECK_EQ(dst, dst_handle_before);   // handle unchanged
    const auto* d = static_cast<const float*>(dst->storage()->data());
    for (int i = 0; i < 4; ++i) CHECK_NEAR(d[i], 7.0f, 1e-6f);
    src->decref(); dst->decref();
}

// @spec FFI-052
TEST(inplace_axpy, rejects_shape_mismatch) {
    auto* dst = full(Dims{4}, DType::Float32, 0.0);
    auto* src = full(Dims{3}, DType::Float32, 1.0);
    CHECK_THROWS(inplace(OP_AXPY, dst, src, 1.0));
    src->decref(); dst->decref();
}
