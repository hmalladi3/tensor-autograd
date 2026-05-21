#include "test_main.h"

#include <cmath>
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

// @spec OPS-020, OPS-002 (out-of-place)
TEST(binary_add, elementwise_same_shape) {
    auto* a = make_filled(Dims{2, 2}, {1, 2, 3, 4});
    auto* b = make_filled(Dims{2, 2}, {10, 20, 30, 40});
    auto* c = binary(OP_ADD, a, b);
    const auto* d = static_cast<const float*>(c->storage()->data());
    CHECK_NEAR(d[0], 11.0f, 1e-6f); CHECK_NEAR(d[1], 22.0f, 1e-6f);
    CHECK_NEAR(d[2], 33.0f, 1e-6f); CHECK_NEAR(d[3], 44.0f, 1e-6f);
    // Original inputs untouched (out-of-place).
    const auto* ad = static_cast<const float*>(a->storage()->data());
    CHECK_NEAR(ad[0], 1.0f, 1e-6f);
    c->decref(); b->decref(); a->decref();
}

// @spec OPS-001, OPS-021
TEST(binary, broadcast_row_plus_col) {
    auto* row = make_filled(Dims{1, 3}, {10, 20, 30});       // (1,3)
    auto* col = make_filled(Dims{2, 1}, {1, 2});             // (2,1)
    auto* c   = binary(OP_ADD, row, col);
    CHECK(c->shape() == (Dims{2, 3}));
    const auto* d = static_cast<const float*>(c->storage()->data());
    // [[11, 21, 31], [12, 22, 32]]
    const float expected[6] = {11, 21, 31, 12, 22, 32};
    for (int i = 0; i < 6; ++i) CHECK_NEAR(d[i], expected[i], 1e-6f);
    c->decref(); col->decref(); row->decref();
}

// @spec OPS-020
TEST(binary, mul_sub_div_pow) {
    auto* a = make_filled(Dims{4}, {1, 2, 3, 4});
    auto* b = make_filled(Dims{4}, {2, 2, 2, 2});

    auto* m = binary(OP_MUL, a, b);
    const auto* md = static_cast<const float*>(m->storage()->data());
    CHECK_NEAR(md[2], 6.0f, 1e-6f);

    auto* s = binary(OP_SUB, a, b);
    CHECK_NEAR(static_cast<const float*>(s->storage()->data())[0], -1.0f, 1e-6f);

    auto* d = binary(OP_DIV, a, b);
    CHECK_NEAR(static_cast<const float*>(d->storage()->data())[3], 2.0f, 1e-6f);

    auto* p = binary(OP_POW, a, b);
    CHECK_NEAR(static_cast<const float*>(p->storage()->data())[3], 16.0f, 1e-6f);

    p->decref(); d->decref(); s->decref(); m->decref(); b->decref(); a->decref();
}

// @spec OPS-020, OPS-022 (comparison ops produce Bool)
TEST(binary, comparisons_produce_bool) {
    auto* a = make_filled(Dims{3}, {1, 5, 3});
    auto* b = make_filled(Dims{3}, {2, 5, 3});
    auto* eq = binary(OP_EQ, a, b);
    auto* lt = binary(OP_LT, a, b);
    auto* gt = binary(OP_GT, a, b);
    CHECK_EQ(eq->dtype(), DType::Bool);
    CHECK_EQ(lt->dtype(), DType::Bool);
    CHECK_EQ(gt->dtype(), DType::Bool);
    const auto* eqd = static_cast<const std::uint8_t*>(eq->storage()->data());
    const auto* ltd = static_cast<const std::uint8_t*>(lt->storage()->data());
    const auto* gtd = static_cast<const std::uint8_t*>(gt->storage()->data());
    CHECK_EQ(eqd[0], 0u); CHECK_EQ(eqd[1], 1u); CHECK_EQ(eqd[2], 1u);
    CHECK_EQ(ltd[0], 1u); CHECK_EQ(ltd[1], 0u); CHECK_EQ(ltd[2], 0u);
    CHECK_EQ(gtd[0], 0u); CHECK_EQ(gtd[1], 0u); CHECK_EQ(gtd[2], 0u);
    gt->decref(); lt->decref(); eq->decref(); b->decref(); a->decref();
}

// @spec OPS-004 (dtype mismatch raises)
TEST(binary, dtype_mismatch_raises) {
    auto* a = TensorImpl::make_contiguous(Dims{3}, DType::Float32);
    auto* b = TensorImpl::make_contiguous(Dims{3}, DType::Int64);
    CHECK_THROWS(binary(OP_ADD, a, b));
    b->decref(); a->decref();
}

// @spec OPS-030, OPS-032
TEST(unary_relu, clamps_below_zero) {
    auto* a = make_filled(Dims{5}, {-2, -1, 0, 1, 2});
    auto* r = unary(OP_RELU, a);
    const auto* d = static_cast<const float*>(r->storage()->data());
    CHECK_NEAR(d[0], 0.0f, 1e-6f); CHECK_NEAR(d[1], 0.0f, 1e-6f);
    CHECK_NEAR(d[2], 0.0f, 1e-6f); CHECK_NEAR(d[3], 1.0f, 1e-6f);
    CHECK_NEAR(d[4], 2.0f, 1e-6f);
    r->decref(); a->decref();
}

// @spec OPS-030 (sqrt included)
TEST(unary, sqrt_exp_log_neg_sigmoid_tanh) {
    auto* a = make_filled(Dims{3}, {1.0f, 4.0f, 9.0f});

    auto* sq = unary(OP_SQRT, a);
    CHECK_NEAR(static_cast<const float*>(sq->storage()->data())[1], 2.0f, 1e-5f);
    sq->decref();

    auto* lg = unary(OP_LOG, a);
    CHECK_NEAR(static_cast<const float*>(lg->storage()->data())[0], 0.0f, 1e-6f);
    lg->decref();

    auto* ng = unary(OP_NEG, a);
    CHECK_NEAR(static_cast<const float*>(ng->storage()->data())[2], -9.0f, 1e-6f);
    ng->decref();

    auto* ex = unary(OP_EXP, a);
    CHECK_NEAR(static_cast<const float*>(ex->storage()->data())[0], std::exp(1.0f), 1e-5f);
    ex->decref();

    auto* sg = unary(OP_SIGMOID, a);
    CHECK_NEAR(static_cast<const float*>(sg->storage()->data())[0],
               1.0f / (1.0f + std::exp(-1.0f)), 1e-6f);
    sg->decref();

    auto* th = unary(OP_TANH, a);
    CHECK_NEAR(static_cast<const float*>(th->storage()->data())[0], std::tanh(1.0f), 1e-6f);
    th->decref();

    a->decref();
}

// @spec OPS-005 (contiguous fast path), OPS-007 (respects contiguity)
TEST(binary, works_on_transposed_inputs_via_strided_path) {
    auto* a = make_filled(Dims{2, 3}, {1, 2, 3, 4, 5, 6});  // [[1,2,3],[4,5,6]]
    auto* at = a->transpose(0, 1);                          // [[1,4],[2,5],[3,6]]
    auto* b  = make_filled(Dims{3, 2}, {10, 20, 30, 40, 50, 60});
    auto* c  = binary(OP_ADD, at, b);
    CHECK(c->shape() == (Dims{3, 2}));
    const auto* d = static_cast<const float*>(c->storage()->data());
    const float expected[6] = {11, 24, 32, 45, 53, 66};
    for (int i = 0; i < 6; ++i) CHECK_NEAR(d[i], expected[i], 1e-6f);
    c->decref(); b->decref(); at->decref(); a->decref();
}
