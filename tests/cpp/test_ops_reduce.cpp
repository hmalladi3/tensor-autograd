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

// @spec OPS-040, OPS-043 (empty axes reduces all)
TEST(reduce_sum, all_dims_to_scalar) {
    auto* a = make_filled(Dims{2, 3}, {1, 2, 3, 4, 5, 6});
    auto* s = reduce(OP_SUM, a, Dims{}, /*keepdim=*/false);
    CHECK(s->shape().empty());
    CHECK_NEAR(static_cast<const float*>(s->storage()->data())[0], 21.0f, 1e-6f);
    s->decref(); a->decref();
}

// @spec OPS-040, OPS-042
TEST(reduce_sum, single_axis_drops_dim) {
    auto* a = make_filled(Dims{2, 3}, {1, 2, 3, 4, 5, 6});
    auto* s = reduce(OP_SUM, a, Dims{0}, false);
    CHECK(s->shape() == (Dims{3}));
    const auto* d = static_cast<const float*>(s->storage()->data());
    CHECK_NEAR(d[0], 5.0f, 1e-6f);   // 1 + 4
    CHECK_NEAR(d[1], 7.0f, 1e-6f);   // 2 + 5
    CHECK_NEAR(d[2], 9.0f, 1e-6f);   // 3 + 6
    s->decref(); a->decref();
}

// @spec OPS-041
TEST(reduce_sum, keepdim_preserves_size_one_dim) {
    auto* a = make_filled(Dims{2, 3}, {1, 2, 3, 4, 5, 6});
    auto* s = reduce(OP_SUM, a, Dims{1}, /*keepdim=*/true);
    CHECK(s->shape() == (Dims{2, 1}));
    const auto* d = static_cast<const float*>(s->storage()->data());
    CHECK_NEAR(d[0], 6.0f, 1e-6f);   // row 0
    CHECK_NEAR(d[1], 15.0f, 1e-6f);  // row 1
    s->decref(); a->decref();
}

// @spec OPS-046 (negative axes)
TEST(reduce_sum, negative_axis) {
    auto* a = make_filled(Dims{2, 3}, {1, 2, 3, 4, 5, 6});
    auto* s = reduce(OP_SUM, a, Dims{-1}, false);
    CHECK(s->shape() == (Dims{2}));
    const auto* d = static_cast<const float*>(s->storage()->data());
    CHECK_NEAR(d[0], 6.0f, 1e-6f);
    CHECK_NEAR(d[1], 15.0f, 1e-6f);
    s->decref(); a->decref();
}

// @spec OPS-040 (mean)
TEST(reduce_mean, computes_average) {
    auto* a = make_filled(Dims{4}, {2, 4, 6, 8});
    auto* m = reduce(OP_MEAN, a, Dims{}, false);
    CHECK_NEAR(static_cast<const float*>(m->storage()->data())[0], 5.0f, 1e-6f);
    m->decref(); a->decref();
}

// @spec OPS-040
TEST(reduce_max, picks_largest) {
    auto* a = make_filled(Dims{2, 3}, {3, 1, 4, 1, 5, 9});
    auto* m = reduce(OP_MAX_R, a, Dims{1}, false);
    CHECK(m->shape() == (Dims{2}));
    const auto* d = static_cast<const float*>(m->storage()->data());
    CHECK_NEAR(d[0], 4.0f, 1e-6f);
    CHECK_NEAR(d[1], 9.0f, 1e-6f);
    m->decref(); a->decref();
}

// @spec OPS-044 (argmax dtype is Int64)
TEST(reduce_argmax, returns_int64_indices) {
    auto* a = make_filled(Dims{2, 3}, {3, 1, 4, 1, 5, 9});
    auto* am = reduce(OP_ARGMAX, a, Dims{1}, false);
    CHECK_EQ(am->dtype(), DType::Int64);
    const auto* d = static_cast<const std::int64_t*>(am->storage()->data());
    CHECK_EQ(d[0], static_cast<std::int64_t>(2));  // argmax of [3,1,4]
    CHECK_EQ(d[1], static_cast<std::int64_t>(2));  // argmax of [1,5,9]
    am->decref(); a->decref();
}

TEST(reduce, works_on_strided_input) {
    auto* a  = make_filled(Dims{2, 3}, {1, 2, 3, 4, 5, 6});
    auto* at = a->transpose(0, 1);          // shape (3,2), non-contig
    auto* s  = reduce(OP_SUM, at, Dims{0}, false);
    // Sum across the "rows" of the transposed view = sum across original cols
    CHECK(s->shape() == (Dims{2}));
    const auto* d = static_cast<const float*>(s->storage()->data());
    CHECK_NEAR(d[0], 6.0f, 1e-6f);   // 1 + 2 + 3
    CHECK_NEAR(d[1], 15.0f, 1e-6f);  // 4 + 5 + 6
    s->decref(); at->decref(); a->decref();
}
