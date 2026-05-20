#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "../broadcast.h"
#include "../dispatch.h"
#include "../iter.h"
#include "../ops.h"
#include "../tensor_impl.h"

namespace tae::ops {

namespace {

// Returns true if the op produces a Bool output (comparisons); false otherwise.
bool is_comparison(BinaryOp op) noexcept {
    return op == OP_EQ || op == OP_LT || op == OP_GT;
}

template <typename TIn, typename TOut, typename Kernel>
void run_binary(const TensorImpl* a, const TensorImpl* b, TensorImpl* out, Kernel kernel) {
    const auto* a_data   = static_cast<const TIn*>(a->storage()->data());
    const auto* b_data   = static_cast<const TIn*>(b->storage()->data());
    auto*       out_data = static_cast<TOut*>(out->storage()->data());

    // Fast path: all three contiguous and same shape (no broadcast happened).
    if (a->is_contiguous() && b->is_contiguous() && out->is_contiguous()
        && a->shape() == b->shape() && a->shape() == out->shape()) {
        const std::int64_t n      = out->numel();
        const std::int64_t a_base = a->offset();
        const std::int64_t b_base = b->offset();
        const std::int64_t o_base = out->offset();
        for (std::int64_t k = 0; k < n; ++k) {
            out_data[o_base + k] = kernel(a_data[a_base + k], b_data[b_base + k]);
        }
        return;
    }

    iterate_binary(a, b, out,
        [&](std::int64_t /*k*/, std::int64_t a_off, std::int64_t b_off, std::int64_t o_off) {
            out_data[o_off] = kernel(a_data[a_off], b_data[b_off]);
        });
}

}  // namespace

TensorImpl* binary(BinaryOp op, const TensorImpl* a, const TensorImpl* b) {
    if (a->dtype() != b->dtype()) {
        throw std::invalid_argument("binary: inputs must have the same dtype");
    }
    const Dims  out_shape = broadcast_shape(a->shape(), b->shape());
    const DType out_dtype = is_comparison(op) ? DType::Bool : a->dtype();
    auto*       out       = TensorImpl::make_contiguous(out_shape, out_dtype);

    // Broadcast inputs to the output shape. The broadcast view shares storage
    // and adds a stride-0 dimension where applicable.
    TensorImpl* a_view = (a->shape() == out_shape) ? const_cast<TensorImpl*>(a)
                                                   : broadcast_to(a, out_shape);
    TensorImpl* b_view = (b->shape() == out_shape) ? const_cast<TensorImpl*>(b)
                                                   : broadcast_to(b, out_shape);
    if (a_view == a) a_view->incref();
    if (b_view == b) b_view->incref();

    dispatch_floating(a->dtype(), [&](auto tag) {
        using T = decltype(tag);
        // Integer dtype check would normally be here too; v1 only supports
        // Float32 in arithmetic ops, Int64 is reserved for indices.
        switch (op) {
            case OP_ADD: run_binary<T, T>(a_view, b_view, out,
                              [](T x, T y) { return static_cast<T>(x + y); }); break;
            case OP_SUB: run_binary<T, T>(a_view, b_view, out,
                              [](T x, T y) { return static_cast<T>(x - y); }); break;
            case OP_MUL: run_binary<T, T>(a_view, b_view, out,
                              [](T x, T y) { return static_cast<T>(x * y); }); break;
            case OP_DIV: run_binary<T, T>(a_view, b_view, out,
                              [](T x, T y) { return static_cast<T>(x / y); }); break;
            case OP_POW: run_binary<T, T>(a_view, b_view, out,
                              [](T x, T y) { return static_cast<T>(std::pow(x, y)); }); break;
            case OP_MAX: run_binary<T, T>(a_view, b_view, out,
                              [](T x, T y) { return std::max(x, y); }); break;
            case OP_MIN: run_binary<T, T>(a_view, b_view, out,
                              [](T x, T y) { return std::min(x, y); }); break;
            case OP_EQ:  run_binary<T, std::uint8_t>(a_view, b_view, out,
                              [](T x, T y) -> std::uint8_t { return x == y ? 1 : 0; }); break;
            case OP_LT:  run_binary<T, std::uint8_t>(a_view, b_view, out,
                              [](T x, T y) -> std::uint8_t { return x <  y ? 1 : 0; }); break;
            case OP_GT:  run_binary<T, std::uint8_t>(a_view, b_view, out,
                              [](T x, T y) -> std::uint8_t { return x >  y ? 1 : 0; }); break;
            default: throw std::invalid_argument("binary: unknown op_id");
        }
    });

    a_view->decref();
    b_view->decref();
    return out;
}

}  // namespace tae::ops
