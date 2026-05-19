#include "broadcast.h"

#include <algorithm>
#include <stdexcept>

namespace tae {

Dims broadcast_shape(const Dims& a, const Dims& b) {
    const std::size_t ra = a.size();
    const std::size_t rb = b.size();
    const std::size_t rr = std::max(ra, rb);
    Dims result(rr, 1);

    for (std::size_t i = 0; i < rr; ++i) {
        const std::int64_t da = (i < ra) ? a[ra - 1 - i] : 1;
        const std::int64_t db = (i < rb) ? b[rb - 1 - i] : 1;
        if (da != db && da != 1 && db != 1) {
            throw std::invalid_argument("broadcast_shape: shapes are not broadcastable");
        }
        result[rr - 1 - i] = std::max(da, db);
    }
    return result;
}

TensorImpl* broadcast_to(const TensorImpl* t, const Dims& target_shape) {
    const std::size_t rt = static_cast<std::size_t>(t->ndim());
    const std::size_t rr = target_shape.size();
    if (rt > rr) {
        throw std::invalid_argument("broadcast_to: source has more dims than target");
    }
    const std::size_t pad = rr - rt;

    Dims new_strides(rr, 0);
    for (std::size_t i = 0; i < rr; ++i) {
        const std::int64_t tgt_size = target_shape[i];

        if (i < pad) {
            // Padding dim: source is implicitly size 1 here. Stride 0 — the
            // expanded dimension reads the same element for every index.
            continue;
        }

        const std::size_t  src_dim    = i - pad;
        const std::int64_t src_size   = t->shape()[src_dim];
        const std::int64_t src_stride = t->strides()[src_dim];

        if (src_size == tgt_size) {
            new_strides[i] = src_stride;
        } else if (src_size == 1) {
            // Broadcast this dimension: stride 0 so all expanded indices alias.
            new_strides[i] = 0;
        } else {
            throw std::invalid_argument("broadcast_to: incompatible source shape");
        }
    }
    return new TensorImpl(t->storage(), target_shape, std::move(new_strides),
                          t->offset(), t->dtype());
}

}  // namespace tae
