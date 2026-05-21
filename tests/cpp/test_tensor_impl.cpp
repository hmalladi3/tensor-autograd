#include "test_main.h"

#include "tensor_impl.h"

using tae::contiguous_strides_for;
using tae::Dims;
using tae::DType;
using tae::is_contiguous_layout;
using tae::normalize_dim;
using tae::TensorImpl;

// @spec TENS-010 (computation), TENS-002 (units are elements)
TEST(tensor_helpers, contiguous_strides_are_row_major) {
    CHECK(contiguous_strides_for(Dims{2, 3, 4}) == (Dims{12, 4, 1}));
    CHECK(contiguous_strides_for(Dims{5}) == (Dims{1}));
    CHECK(contiguous_strides_for(Dims{}).empty());
}

// @spec TENS-010
TEST(tensor_helpers, is_contiguous_layout_recognizes_c_order) {
    CHECK(is_contiguous_layout(Dims{2, 3, 4}, Dims{12, 4, 1}));
    CHECK(!is_contiguous_layout(Dims{2, 3, 4}, Dims{1, 2, 6}));
    CHECK(is_contiguous_layout(Dims{}, Dims{}));               // 0-dim scalar
    CHECK(is_contiguous_layout(Dims{1, 3}, Dims{0, 1}));       // size-1 stride doesn't matter
    CHECK(!is_contiguous_layout(Dims{3, 4}, Dims{0, 1}));      // broadcast view (stride 0)
}

// @spec TENS-008, TENS-009
TEST(tensor_helpers, normalize_dim_handles_negatives_and_bounds) {
    CHECK_EQ(normalize_dim(0, 3), 0);
    CHECK_EQ(normalize_dim(2, 3), 2);
    CHECK_EQ(normalize_dim(-1, 3), 2);
    CHECK_EQ(normalize_dim(-3, 3), 0);
    CHECK_THROWS(normalize_dim(3, 3));
    CHECK_THROWS(normalize_dim(-4, 3));
}

// @spec TENS-006, TENS-007
TEST(tensor_impl, refcount_initializes_to_one_and_decrefs_to_zero) {
    auto* t = TensorImpl::make_contiguous(Dims{2, 3}, DType::Float32);
    CHECK_EQ(t->refcount(), 1);
    t->incref();
    CHECK_EQ(t->refcount(), 2);
    CHECK(!t->decref());
    CHECK_EQ(t->refcount(), 1);
    CHECK(t->decref());
}

// @spec TENS-004, TENS-005
TEST(tensor_impl, view_construction_shares_storage_and_manages_refcount) {
    auto* a = TensorImpl::make_contiguous(Dims{4, 5}, DType::Float32);
    auto* s = a->storage();
    REQUIRE(s != nullptr);
    CHECK_EQ(s->refcount(), 1);  // owned by `a`

    auto* b = a->transpose(0, 1);  // new view, increfs storage
    CHECK_EQ(s->refcount(), 2);
    CHECK_EQ(b->storage(), s);

    b->decref();
    CHECK_EQ(s->refcount(), 1);
    a->decref();  // last view → storage freed
}

// @spec TENS-001, TENS-011 (cached contiguity)
TEST(tensor_impl, fresh_tensor_is_contiguous_with_row_major_strides) {
    auto* t = TensorImpl::make_contiguous(Dims{3, 4, 5}, DType::Float32);
    CHECK(t->is_contiguous());
    CHECK(t->strides() == (Dims{20, 5, 1}));
    CHECK_EQ(t->offset(), 0);
    CHECK_EQ(t->numel(), static_cast<std::int64_t>(60));
    t->decref();
}

// @spec TENS-020
TEST(tensor_impl, transpose_swaps_dims_and_breaks_contiguity) {
    auto* t = TensorImpl::make_contiguous(Dims{2, 3, 4}, DType::Float32);
    auto* tt = t->transpose(0, 2);
    CHECK(tt->shape()   == (Dims{4, 3, 2}));
    CHECK(tt->strides() == (Dims{1, 4, 12}));
    CHECK(!tt->is_contiguous());
    tt->decref();
    t->decref();
}

// @spec TENS-008 (negative axis)
TEST(tensor_impl, transpose_accepts_negative_dims) {
    auto* t = TensorImpl::make_contiguous(Dims{2, 3, 4}, DType::Float32);
    auto* tt = t->transpose(-1, -2);
    CHECK(tt->shape() == (Dims{2, 4, 3}));
    tt->decref();
    t->decref();
}

// @spec TENS-021
TEST(tensor_impl, permute_reorders_shape_and_strides) {
    auto* t = TensorImpl::make_contiguous(Dims{2, 3, 4}, DType::Float32);
    auto* p = t->permute(Dims{2, 0, 1});
    CHECK(p->shape()   == (Dims{4, 2, 3}));
    CHECK(p->strides() == (Dims{1, 12, 4}));
    p->decref();
    t->decref();
}

TEST(tensor_impl, permute_rejects_duplicate_or_wrong_rank) {
    auto* t = TensorImpl::make_contiguous(Dims{2, 3, 4}, DType::Float32);
    CHECK_THROWS(t->permute(Dims{0, 0, 1}));
    CHECK_THROWS(t->permute(Dims{0, 1}));
    t->decref();
}

// @spec TENS-009 (permute accepts negative axis indices)
TEST(tensor_impl, permute_accepts_negative_dims) {
    auto* t = TensorImpl::make_contiguous(Dims{2, 3, 4}, DType::Float32);
    auto* p = t->permute(Dims{-1, 0, -2});  // == permute(2, 0, 1)
    CHECK(p->shape() == (Dims{4, 2, 3}));
    p->decref();
    t->decref();
}

// @spec TENS-022
TEST(tensor_impl, squeeze_removes_size_one_dim) {
    auto* t = TensorImpl::make_contiguous(Dims{2, 1, 4}, DType::Float32);
    auto* s = t->squeeze(1);
    CHECK(s->shape() == (Dims{2, 4}));
    s->decref();
    t->decref();
}

TEST(tensor_impl, squeeze_rejects_nonunit_dim) {
    auto* t = TensorImpl::make_contiguous(Dims{2, 3, 4}, DType::Float32);
    CHECK_THROWS(t->squeeze(0));
    t->decref();
}

// @spec TENS-023
TEST(tensor_impl, unsqueeze_inserts_size_one_dim) {
    auto* t = TensorImpl::make_contiguous(Dims{2, 4}, DType::Float32);
    auto* s = t->unsqueeze(1);
    CHECK(s->shape() == (Dims{2, 1, 4}));
    s->decref();
    auto* end = t->unsqueeze(2);  // insert past the end
    CHECK(end->shape() == (Dims{2, 4, 1}));
    end->decref();
    auto* neg = t->unsqueeze(-1);
    CHECK(neg->shape() == (Dims{2, 4, 1}));
    neg->decref();
    t->decref();
}

// @spec TENS-030 (view path)
TEST(tensor_impl, reshape_contiguous_returns_view) {
    auto* t = TensorImpl::make_contiguous(Dims{2, 3, 4}, DType::Float32);
    auto* r = t->reshape(Dims{6, 4});
    CHECK(r->shape()   == (Dims{6, 4}));
    CHECK(r->strides() == (Dims{4, 1}));
    CHECK_EQ(r->storage(), t->storage());  // shared
    r->decref();
    t->decref();
}

// @spec TENS-031 (copy fallback)
TEST(tensor_impl, reshape_noncontiguous_materializes_copy) {
    auto* t  = TensorImpl::make_contiguous(Dims{2, 3, 4}, DType::Float32);
    auto* tt = t->transpose(0, 1);  // non-contiguous
    auto* r  = tt->reshape(Dims{3, 8});
    CHECK(r->shape() == (Dims{3, 8}));
    CHECK(r->is_contiguous());
    CHECK(r->storage() != t->storage());  // fresh Storage
    r->decref();
    tt->decref();
    t->decref();
}

// @spec TENS-033
TEST(tensor_impl, reshape_rejects_size_mismatch) {
    auto* t = TensorImpl::make_contiguous(Dims{2, 3, 4}, DType::Float32);
    CHECK_THROWS(t->reshape(Dims{5, 5}));
    t->decref();
}

// @spec TENS-032
TEST(tensor_impl, slice_adjusts_shape_stride_offset) {
    auto* t = TensorImpl::make_contiguous(Dims{2, 6}, DType::Float32);
    auto* s = t->slice(1, /*start=*/1, /*stop=*/5, /*step=*/2);
    CHECK(s->shape()   == (Dims{2, 2}));
    CHECK(s->strides() == (Dims{6, 2}));
    CHECK_EQ(s->offset(), 1);
    s->decref();
    t->decref();
}

// @spec TENS-012
TEST(tensor_impl, contiguous_on_contiguous_returns_same_object) {
    auto* t = TensorImpl::make_contiguous(Dims{2, 3}, DType::Float32);
    auto* c = t->contiguous();
    CHECK_EQ(c, t);
    CHECK_EQ(t->refcount(), 2);
    c->decref();
    t->decref();
}

// @spec TENS-013
TEST(tensor_impl, contiguous_on_noncontiguous_copies_data_in_c_order) {
    auto* t = TensorImpl::make_contiguous(Dims{2, 3}, DType::Float32);
    // Fill source buffer: [[0, 1, 2], [3, 4, 5]]
    auto* src = static_cast<float*>(t->storage()->data());
    for (int i = 0; i < 6; ++i) src[i] = static_cast<float>(i);

    auto* tt = t->transpose(0, 1);  // shape (3,2), strides (1,3), non-contig
    auto* c  = tt->contiguous();
    CHECK(c->is_contiguous());
    CHECK(c->shape() == (Dims{3, 2}));

    // Expected: [[0, 3], [1, 4], [2, 5]]
    const auto* out = static_cast<const float*>(c->storage()->data());
    const float expected[6] = {0, 3, 1, 4, 2, 5};
    for (int i = 0; i < 6; ++i) CHECK_NEAR(out[i], expected[i], 1e-6f);

    c->decref();
    tt->decref();
    t->decref();
}
