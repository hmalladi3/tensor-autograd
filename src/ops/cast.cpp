#include <cstdint>

#include "../dispatch.h"
#include "../iter.h"
#include "../ops.h"
#include "../tensor_impl.h"

namespace tae::ops {

TensorImpl* cast(const TensorImpl* a, DType dtype) {
    // Same-dtype cast: return the same TensorImpl with refcount bumped. The
    // const_cast is safe — refcount mutation doesn't change logical state.
    if (a->dtype() == dtype) {
        const_cast<TensorImpl*>(a)->incref();
        return const_cast<TensorImpl*>(a);
    }

    auto* out = TensorImpl::make_contiguous(a->shape(), dtype);
    dispatch_all(a->dtype(), [&](auto in_tag) {
        using TIn = decltype(in_tag);
        const auto* in_data = static_cast<const TIn*>(a->storage()->data());
        dispatch_all(dtype, [&](auto out_tag) {
            using TOut = decltype(out_tag);
            auto* out_data = static_cast<TOut*>(out->storage()->data());
            iterate_unary(a, out,
                [&](std::int64_t /*k*/, std::int64_t a_off, std::int64_t o_off) {
                    out_data[o_off] = static_cast<TOut>(in_data[a_off]);
                });
        });
    });
    return out;
}

}  // namespace tae::ops
