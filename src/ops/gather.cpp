#include <cstdint>
#include <stdexcept>

#include "../dispatch.h"
#include "../ops.h"
#include "../tensor_impl.h"

namespace tae::ops {

// gather(input, dim, indices):
//   output[i0, i1, ..., id, ...] = input[i0, i1, ..., indices[i0, i1, ..., id, ...], ...]
// Output has the same shape as `indices`. The non-`dim` dimensions of input
// must match indices in the corresponding positions.
TensorImpl* gather(const TensorImpl* a, std::int64_t dim, const TensorImpl* indices) {
    if (indices->dtype() != DType::Int64) {
        throw std::invalid_argument("gather: indices must be Int64");
    }
    const std::int64_t ndim = a->ndim();
    if (indices->ndim() != ndim) {
        throw std::invalid_argument("gather: indices must have same rank as input");
    }
    dim = normalize_dim(dim, ndim);

    // Output has the shape of `indices`. Validate non-`dim` dims match input.
    for (std::int64_t d = 0; d < ndim; ++d) {
        if (d == dim) continue;
        if (indices->shape()[static_cast<std::size_t>(d)]
            != a->shape()[static_cast<std::size_t>(d)]) {
            throw std::invalid_argument("gather: non-dim shapes must match");
        }
    }

    auto* out = TensorImpl::make_contiguous(indices->shape(), a->dtype());

    const auto&        out_shape = out->shape();
    const std::int64_t total     = out->numel();
    const auto&        a_str     = a->strides();
    const auto&        idx_str   = indices->strides();
    const std::int64_t a_base    = a->offset();
    const std::int64_t idx_base  = indices->offset();
    const std::int64_t dim_size  = a->shape()[static_cast<std::size_t>(dim)];

    dispatch_floating(a->dtype(), [&](auto tag) {
        using T = decltype(tag);
        const auto* a_data       = static_cast<const T*>(a->storage()->data());
        const auto* idx_data     = static_cast<const std::int64_t*>(indices->storage()->data());
        auto*       out_data     = static_cast<T*>(out->storage()->data());
        const std::int64_t a_dim_stride = a_str[static_cast<std::size_t>(dim)];

        Dims idx(static_cast<std::size_t>(ndim), 0);
        for (std::int64_t k = 0; k < total; ++k) {
            // Compute offsets into `a` (skipping the gathered dim) and `indices`.
            std::int64_t a_off = a_base;
            std::int64_t i_off = idx_base;
            for (std::int64_t d = 0; d < ndim; ++d) {
                i_off += idx[d] * idx_str[d];
                if (d != dim) a_off += idx[d] * a_str[d];
            }
            const std::int64_t gathered = idx_data[i_off];
            if (gathered < 0 || gathered >= dim_size) {
                throw std::out_of_range("gather: index out of range");
            }
            a_off += gathered * a_dim_stride;
            out_data[k] = a_data[a_off];

            for (std::int64_t d = ndim - 1; d >= 0; --d) {
                if (++idx[d] < out_shape[d]) break;
                idx[d] = 0;
            }
        }
    });
    return out;
}

}  // namespace tae::ops
