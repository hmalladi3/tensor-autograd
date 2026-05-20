#include <stdexcept>

#include "../dispatch.h"
#include "../iter.h"
#include "../ops.h"
#include "../tensor_impl.h"

namespace tae::ops {

void inplace(InplaceOp op, TensorImpl* dst, const TensorImpl* src, double alpha) {
    if (dst->dtype() != src->dtype()) {
        throw std::invalid_argument("inplace: dst/src dtype mismatch");
    }
    if (dst->shape() != src->shape()) {
        throw std::invalid_argument("inplace: dst/src shape mismatch");
    }

    dispatch_floating(dst->dtype(), [&](auto tag) {
        using T = decltype(tag);
        auto*       dst_data = static_cast<T*>(dst->storage()->data());
        const auto* src_data = static_cast<const T*>(src->storage()->data());
        const T     a        = static_cast<T>(alpha);

        switch (op) {
            case OP_AXPY:
                iterate_binary(src, dst, dst,
                    [&](std::int64_t /*k*/, std::int64_t s_off, std::int64_t d_off,
                        std::int64_t /*o_off equals d_off*/) {
                        dst_data[d_off] = static_cast<T>(dst_data[d_off] + a * src_data[s_off]);
                    });
                return;
        }
        throw std::invalid_argument("inplace: unknown op_id");
    });
}

}  // namespace tae::ops
