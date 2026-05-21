// Tests for the public C ABI. This file deliberately includes ONLY
// tensor_engine.h — no tae:: internals — so it exercises the surface a real
// C client (or the Python ctypes binding) would use.

#include "test_main.h"

#include <cstdint>
#include <cstring>

#include "tensor_engine.h"

// ============================================================================
// Construction + metadata + lifetime
// ============================================================================

// @spec FFI-005, FFI-020, FFI-023
TEST(ffi_lifetime, full_construct_metadata_and_decref) {
    int64_t shape[2] = {2, 3};
    tensor_handle_t t = nullptr;
    REQUIRE(tensor_full(shape, 2, DTYPE_FLOAT32, 1.5, &t) == TENSOR_OK);
    REQUIRE(t != nullptr);

    CHECK_EQ(tensor_ndim(t), static_cast<int64_t>(2));
    CHECK_EQ(tensor_numel(t), static_cast<int64_t>(6));
    CHECK_EQ(tensor_dtype(t), static_cast<tensor_dtype_t>(DTYPE_FLOAT32));

    int64_t s[2] = {0, 0};
    tensor_shape(t, s);
    CHECK_EQ(s[0], static_cast<int64_t>(2));
    CHECK_EQ(s[1], static_cast<int64_t>(3));

    int64_t st[2] = {0, 0};
    tensor_strides(t, st);
    CHECK_EQ(st[0], static_cast<int64_t>(3));
    CHECK_EQ(st[1], static_cast<int64_t>(1));

    tensor_decref(t);
}

// @spec FFI-022, FFI-023
TEST(ffi_lifetime, incref_then_decref_pair_does_not_free) {
    int64_t shape[1] = {4};
    tensor_handle_t t = nullptr;
    REQUIRE(tensor_full(shape, 1, DTYPE_FLOAT32, 0.0, &t) == TENSOR_OK);
    tensor_incref(t);   // refcount 2
    tensor_decref(t);   // refcount 1
    // Still alive — can still query.
    CHECK_EQ(tensor_numel(t), static_cast<int64_t>(4));
    tensor_decref(t);   // refcount 0, freed
}

// @spec FFI-030
TEST(ffi_construct, from_buffer_copies_data) {
    float src[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int64_t shape[2] = {2, 3};
    tensor_handle_t t = nullptr;
    REQUIRE(tensor_from_buffer(src, shape, 2, DTYPE_FLOAT32, &t) == TENSOR_OK);

    // Mutating the original buffer must not affect the tensor's storage.
    src[0] = 99.0f;
    float dst[6] = {0};
    REQUIRE(tensor_copy_to_buffer(t, dst, sizeof(dst)) == TENSOR_OK);
    CHECK_NEAR(dst[0], 1.0f, 1e-6f);
    CHECK_NEAR(dst[5], 6.0f, 1e-6f);

    tensor_decref(t);
}

// @spec FFI-031
TEST(ffi_egress, copy_to_buffer_rejects_too_small_dst) {
    int64_t shape[1] = {4};
    tensor_handle_t t = nullptr;
    REQUIRE(tensor_full(shape, 1, DTYPE_FLOAT32, 0.0, &t) == TENSOR_OK);
    float small[2] = {0};
    CHECK_EQ(tensor_copy_to_buffer(t, small, sizeof(small)),
             static_cast<tensor_status_t>(TENSOR_INVALID_ARGUMENT));
    tensor_decref(t);
}

// @spec FFI-032
TEST(ffi_egress, copy_to_buffer_walks_strided_source_in_c_order) {
    // 2x3 source filled with [0..5]
    float src_data[6] = {0, 1, 2, 3, 4, 5};
    int64_t shape[2] = {2, 3};
    tensor_handle_t a = nullptr;
    REQUIRE(tensor_from_buffer(src_data, shape, 2, DTYPE_FLOAT32, &a) == TENSOR_OK);

    tensor_handle_t at = nullptr;
    REQUIRE(tensor_transpose(a, 0, 1, &at) == TENSOR_OK);  // shape (3,2), non-contig
    CHECK_EQ(tensor_numel(at), static_cast<int64_t>(6));

    float dst[6] = {0};
    REQUIRE(tensor_copy_to_buffer(at, dst, sizeof(dst)) == TENSOR_OK);

    // Expected C-order of the transposed view: [0,3, 1,4, 2,5]
    const float expected[6] = {0, 3, 1, 4, 2, 5};
    for (int i = 0; i < 6; ++i) CHECK_NEAR(dst[i], expected[i], 1e-6f);

    tensor_decref(at);
    tensor_decref(a);
}

// ============================================================================
// Views
// ============================================================================

TEST(ffi_views, reshape_transpose_slice_contiguous_cast) {
    int64_t shape[2] = {2, 3};
    tensor_handle_t a = nullptr;
    REQUIRE(tensor_full(shape, 2, DTYPE_FLOAT32, 1.0, &a) == TENSOR_OK);

    int64_t new_shape[1] = {6};
    tensor_handle_t r = nullptr;
    REQUIRE(tensor_reshape(a, new_shape, 1, &r) == TENSOR_OK);
    CHECK_EQ(tensor_ndim(r), static_cast<int64_t>(1));
    CHECK_EQ(tensor_numel(r), static_cast<int64_t>(6));
    tensor_decref(r);

    tensor_handle_t t = nullptr;
    REQUIRE(tensor_transpose(a, 0, 1, &t) == TENSOR_OK);
    int64_t ts[2] = {0, 0};
    tensor_shape(t, ts);
    CHECK_EQ(ts[0], static_cast<int64_t>(3));
    CHECK_EQ(ts[1], static_cast<int64_t>(2));
    tensor_decref(t);

    tensor_handle_t sl = nullptr;
    REQUIRE(tensor_slice(a, 1, 0, 2, 1, &sl) == TENSOR_OK);
    int64_t ss[2] = {0, 0};
    tensor_shape(sl, ss);
    CHECK_EQ(ss[0], static_cast<int64_t>(2));
    CHECK_EQ(ss[1], static_cast<int64_t>(2));
    tensor_decref(sl);

    tensor_handle_t c = nullptr;
    REQUIRE(tensor_contiguous(a, &c) == TENSOR_OK);
    // Contiguous on contiguous → same handle (refcount bumped).
    CHECK_EQ(c, a);
    tensor_decref(c);

    tensor_handle_t casted = nullptr;
    REQUIRE(tensor_cast(a, DTYPE_INT64, &casted) == TENSOR_OK);
    CHECK_EQ(tensor_dtype(casted), static_cast<tensor_dtype_t>(DTYPE_INT64));
    tensor_decref(casted);

    tensor_decref(a);
}

// ============================================================================
// Ops dispatch
// ============================================================================

// @spec FFI-040
TEST(ffi_ops, op_binary_add_with_broadcast) {
    int64_t row_shape[2] = {1, 3};
    int64_t col_shape[2] = {2, 1};
    float row_data[3] = {10, 20, 30};
    float col_data[2] = {1, 2};
    tensor_handle_t row = nullptr, col = nullptr, out = nullptr;
    REQUIRE(tensor_from_buffer(row_data, row_shape, 2, DTYPE_FLOAT32, &row) == TENSOR_OK);
    REQUIRE(tensor_from_buffer(col_data, col_shape, 2, DTYPE_FLOAT32, &col) == TENSOR_OK);
    REQUIRE(op_binary(TENSOR_OP_ADD, row, col, &out) == TENSOR_OK);

    int64_t os[2] = {0, 0};
    tensor_shape(out, os);
    CHECK_EQ(os[0], static_cast<int64_t>(2));
    CHECK_EQ(os[1], static_cast<int64_t>(3));

    float dst[6] = {0};
    REQUIRE(tensor_copy_to_buffer(out, dst, sizeof(dst)) == TENSOR_OK);
    const float expected[6] = {11, 21, 31, 12, 22, 32};
    for (int i = 0; i < 6; ++i) CHECK_NEAR(dst[i], expected[i], 1e-6f);

    tensor_decref(out); tensor_decref(col); tensor_decref(row);
}

// @spec FFI-041
TEST(ffi_ops, op_unary_relu) {
    int64_t shape[1] = {5};
    float data[5] = {-2, -1, 0, 1, 2};
    tensor_handle_t a = nullptr, r = nullptr;
    REQUIRE(tensor_from_buffer(data, shape, 1, DTYPE_FLOAT32, &a) == TENSOR_OK);
    REQUIRE(op_unary(TENSOR_OP_RELU, a, &r) == TENSOR_OK);

    float dst[5] = {0};
    REQUIRE(tensor_copy_to_buffer(r, dst, sizeof(dst)) == TENSOR_OK);
    const float expected[5] = {0, 0, 0, 1, 2};
    for (int i = 0; i < 5; ++i) CHECK_NEAR(dst[i], expected[i], 1e-6f);

    tensor_decref(r); tensor_decref(a);
}

// @spec FFI-042
TEST(ffi_ops, op_reduce_sum_over_axis) {
    int64_t shape[2] = {2, 3};
    float data[6] = {1, 2, 3, 4, 5, 6};
    tensor_handle_t a = nullptr, s = nullptr;
    REQUIRE(tensor_from_buffer(data, shape, 2, DTYPE_FLOAT32, &a) == TENSOR_OK);

    int64_t axes[1] = {1};
    REQUIRE(op_reduce(TENSOR_OP_SUM, a, axes, 1, /*keepdim=*/0, &s) == TENSOR_OK);
    CHECK_EQ(tensor_numel(s), static_cast<int64_t>(2));
    float dst[2] = {0};
    REQUIRE(tensor_copy_to_buffer(s, dst, sizeof(dst)) == TENSOR_OK);
    CHECK_NEAR(dst[0], 6.0f, 1e-6f);
    CHECK_NEAR(dst[1], 15.0f, 1e-6f);

    tensor_decref(s); tensor_decref(a);
}

TEST(ffi_ops, op_matmul_2d) {
    int64_t as[2] = {2, 3};
    int64_t bs[2] = {3, 2};
    float ad[6] = {1, 2, 3, 4, 5, 6};
    float bd[6] = {7, 8, 9, 10, 11, 12};
    tensor_handle_t a = nullptr, b = nullptr, c = nullptr;
    REQUIRE(tensor_from_buffer(ad, as, 2, DTYPE_FLOAT32, &a) == TENSOR_OK);
    REQUIRE(tensor_from_buffer(bd, bs, 2, DTYPE_FLOAT32, &b) == TENSOR_OK);
    REQUIRE(op_matmul(a, b, &c) == TENSOR_OK);

    float dst[4] = {0};
    REQUIRE(tensor_copy_to_buffer(c, dst, sizeof(dst)) == TENSOR_OK);
    CHECK_NEAR(dst[0],  58.0f, 1e-4f);
    CHECK_NEAR(dst[3], 154.0f, 1e-4f);

    tensor_decref(c); tensor_decref(b); tensor_decref(a);
}

// @spec FFI-050, FFI-051
TEST(ffi_ops, op_inplace_axpy_preserves_handle) {
    int64_t shape[1] = {4};
    tensor_handle_t dst = nullptr, src = nullptr;
    REQUIRE(tensor_full(shape, 1, DTYPE_FLOAT32, 1.0, &dst) == TENSOR_OK);
    REQUIRE(tensor_full(shape, 1, DTYPE_FLOAT32, 2.0, &src) == TENSOR_OK);
    tensor_handle_t dst_before = dst;

    REQUIRE(op_inplace(TENSOR_OP_AXPY, dst, src, 3.0) == TENSOR_OK);
    CHECK_EQ(dst, dst_before);  // handle identity preserved

    float read[4] = {0};
    REQUIRE(tensor_copy_to_buffer(dst, read, sizeof(read)) == TENSOR_OK);
    for (int i = 0; i < 4; ++i) CHECK_NEAR(read[i], 7.0f, 1e-6f);  // 1 + 3*2

    tensor_decref(src); tensor_decref(dst);
}

// @spec FFI-052
TEST(ffi_ops, op_inplace_axpy_shape_mismatch) {
    int64_t s4[1] = {4}, s3[1] = {3};
    tensor_handle_t dst = nullptr, src = nullptr;
    REQUIRE(tensor_full(s4, 1, DTYPE_FLOAT32, 0.0, &dst) == TENSOR_OK);
    REQUIRE(tensor_full(s3, 1, DTYPE_FLOAT32, 1.0, &src) == TENSOR_OK);
    CHECK_EQ(op_inplace(TENSOR_OP_AXPY, dst, src, 1.0),
             static_cast<tensor_status_t>(TENSOR_INVALID_ARGUMENT));
    tensor_decref(src); tensor_decref(dst);
}

// @spec FFI-043 (unknown op_id)
TEST(ffi_ops, unknown_op_id_returns_error) {
    int64_t shape[1] = {3};
    tensor_handle_t a = nullptr, out = nullptr;
    REQUIRE(tensor_full(shape, 1, DTYPE_FLOAT32, 0.0, &a) == TENSOR_OK);
    CHECK_EQ(op_unary(999, a, &out),
             static_cast<tensor_status_t>(TENSOR_INVALID_ARGUMENT));
    CHECK_EQ(out, static_cast<tensor_handle_t>(nullptr));  // FFI-011
    tensor_decref(a);
}

// ============================================================================
// Error model
// ============================================================================

// @spec FFI-011, FFI-012, FFI-013
TEST(ffi_errors, last_error_populated_then_cleared) {
    // Force an error: reshape to a wrong size.
    int64_t shape[2] = {2, 3};
    tensor_handle_t a = nullptr;
    REQUIRE(tensor_full(shape, 2, DTYPE_FLOAT32, 0.0, &a) == TENSOR_OK);

    int64_t bad[2] = {2, 5};   // 10 != 6
    tensor_handle_t r = nullptr;
    const tensor_status_t st = tensor_reshape(a, bad, 2, &r);
    CHECK(st != TENSOR_OK);
    CHECK_EQ(r, static_cast<tensor_handle_t>(nullptr));   // FFI-011
    const char* msg = tensor_last_error();
    CHECK(msg != nullptr);
    CHECK(std::strlen(msg) > 0);                          // FFI-012

    // A successful call clears the message.
    int64_t good[1] = {6};
    REQUIRE(tensor_reshape(a, good, 1, &r) == TENSOR_OK);
    CHECK_EQ(std::strlen(tensor_last_error()), static_cast<size_t>(0));   // FFI-013
    tensor_decref(r);

    tensor_decref(a);
}

// ============================================================================
// Random / seed reproducibility
// ============================================================================

// @spec FFI-055, FFI-057
TEST(ffi_random, seed_makes_random_reproducible) {
    int64_t shape[1] = {16};
    tensor_seed(123);
    tensor_handle_t a = nullptr;
    REQUIRE(tensor_random(TENSOR_OP_UNIFORM, shape, 1, DTYPE_FLOAT32, 0.0, 1.0, &a) == TENSOR_OK);
    float ad[16] = {0};
    REQUIRE(tensor_copy_to_buffer(a, ad, sizeof(ad)) == TENSOR_OK);

    tensor_seed(123);
    tensor_handle_t b = nullptr;
    REQUIRE(tensor_random(TENSOR_OP_UNIFORM, shape, 1, DTYPE_FLOAT32, 0.0, 1.0, &b) == TENSOR_OK);
    float bd[16] = {0};
    REQUIRE(tensor_copy_to_buffer(b, bd, sizeof(bd)) == TENSOR_OK);

    for (int i = 0; i < 16; ++i) CHECK_NEAR(ad[i], bd[i], 0.0f);
    tensor_decref(b); tensor_decref(a);
}
