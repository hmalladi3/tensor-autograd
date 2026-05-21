#include "test_main.h"

#include "broadcast.h"

using tae::broadcast_shape;
using tae::broadcast_to;
using tae::Dims;
using tae::DType;
using tae::TensorImpl;

// @spec TENS-040
TEST(broadcast, same_rank_same_shape_is_identity) {
    CHECK(broadcast_shape(Dims{3, 4}, Dims{3, 4}) == (Dims{3, 4}));
}

// @spec TENS-041
TEST(broadcast, ones_expand_to_other_size) {
    CHECK(broadcast_shape(Dims{3, 1}, Dims{1, 4}) == (Dims{3, 4}));
    CHECK(broadcast_shape(Dims{1, 3, 1}, Dims{2, 1, 4}) == (Dims{2, 3, 4}));
}

// @spec TENS-040 (right-alignment)
TEST(broadcast, shorter_shape_right_aligns_with_implicit_leading_ones) {
    CHECK(broadcast_shape(Dims{3}, Dims{2, 3}) == (Dims{2, 3}));
    CHECK(broadcast_shape(Dims{4}, Dims{2, 3, 4}) == (Dims{2, 3, 4}));
    CHECK(broadcast_shape(Dims{}, Dims{5}) == (Dims{5}));
}

// @spec TENS-042
TEST(broadcast, incompatible_shapes_throw) {
    CHECK_THROWS(broadcast_shape(Dims{3}, Dims{4}));
    CHECK_THROWS(broadcast_shape(Dims{2, 3}, Dims{3, 2}));
    CHECK_THROWS(broadcast_shape(Dims{3, 1}, Dims{4, 2}));
}

// @spec TENS-043, TENS-044
TEST(broadcast_to, padding_dim_gets_stride_zero) {
    auto* t = TensorImpl::make_contiguous(Dims{3}, DType::Float32);
    auto* b = broadcast_to(t, Dims{4, 3});
    CHECK(b->shape()   == (Dims{4, 3}));
    CHECK(b->strides() == (Dims{0, 1}));
    CHECK_EQ(b->storage(), t->storage());  // no copy
    b->decref();
    t->decref();
}

// @spec TENS-043
TEST(broadcast_to, broadcast_dim_gets_stride_zero) {
    auto* t = TensorImpl::make_contiguous(Dims{1, 3, 1}, DType::Float32);
    auto* b = broadcast_to(t, Dims{2, 3, 4});
    CHECK(b->shape()   == (Dims{2, 3, 4}));
    CHECK(b->strides() == (Dims{0, 1, 0}));
    b->decref();
    t->decref();
}

TEST(broadcast_to, kept_dim_preserves_source_stride) {
    auto* t  = TensorImpl::make_contiguous(Dims{2, 3}, DType::Float32);  // strides (3, 1)
    auto* b  = broadcast_to(t, Dims{2, 3});
    CHECK(b->shape()   == (Dims{2, 3}));
    CHECK(b->strides() == (Dims{3, 1}));
    b->decref();
    t->decref();
}

TEST(broadcast_to, broadcast_view_is_not_contiguous) {
    auto* t = TensorImpl::make_contiguous(Dims{3}, DType::Float32);
    auto* b = broadcast_to(t, Dims{4, 3});
    CHECK(!b->is_contiguous());
    b->decref();
    t->decref();
}

TEST(broadcast_to, rejects_source_with_more_dims_than_target) {
    auto* t = TensorImpl::make_contiguous(Dims{2, 3, 4}, DType::Float32);
    CHECK_THROWS(broadcast_to(t, Dims{3, 4}));
    t->decref();
}

TEST(broadcast_to, rejects_incompatible_source) {
    auto* t = TensorImpl::make_contiguous(Dims{3, 2}, DType::Float32);
    CHECK_THROWS(broadcast_to(t, Dims{3, 4}));  // 2 vs 4, neither is 1
    t->decref();
}
