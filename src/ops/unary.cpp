#include <cmath>
#include <stdexcept>

#include "../dispatch.h"
#include "../iter.h"
#include "../ops.h"
#include "../tensor_impl.h"

namespace tae::ops {

namespace {

template <typename T, typename Kernel>
void run_unary(const TensorImpl* a, TensorImpl* out, Kernel kernel) {
    const auto* a_data   = static_cast<const T*>(a->storage()->data());
    auto*       out_data = static_cast<T*>(out->storage()->data());

    if (a->is_contiguous() && out->is_contiguous() && a->shape() == out->shape()) {
        const std::int64_t n      = out->numel();
        const std::int64_t a_base = a->offset();
        const std::int64_t o_base = out->offset();
        for (std::int64_t k = 0; k < n; ++k) {
            out_data[o_base + k] = kernel(a_data[a_base + k]);
        }
        return;
    }
    iterate_unary(a, out, [&](std::int64_t /*k*/, std::int64_t a_off, std::int64_t o_off) {
        out_data[o_off] = kernel(a_data[a_off]);
    });
}

}  // namespace

TensorImpl* unary(UnaryOp op, const TensorImpl* a) {
    auto* out = TensorImpl::make_contiguous(a->shape(), a->dtype());

    dispatch_floating(a->dtype(), [&](auto tag) {
        using T = decltype(tag);
        switch (op) {
            case OP_NEG:     run_unary<T>(a, out, [](T x) { return static_cast<T>(-x); }); break;
            case OP_EXP:     run_unary<T>(a, out, [](T x) { return static_cast<T>(std::exp(x)); }); break;
            case OP_LOG:     run_unary<T>(a, out, [](T x) { return static_cast<T>(std::log(x)); }); break;
            case OP_SQRT:    run_unary<T>(a, out, [](T x) { return static_cast<T>(std::sqrt(x)); }); break;
            case OP_RELU:    run_unary<T>(a, out, [](T x) { return x > T(0) ? x : T(0); }); break;
            case OP_SIGMOID: run_unary<T>(a, out, [](T x) {
                                 return static_cast<T>(T(1) / (T(1) + std::exp(-x)));
                             }); break;
            case OP_TANH:    run_unary<T>(a, out, [](T x) { return static_cast<T>(std::tanh(x)); }); break;
            default: throw std::invalid_argument("unary: unknown op_id");
        }
    });
    return out;
}

}  // namespace tae::ops
