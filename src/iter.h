#ifndef TENSOR_ITER_H
#define TENSOR_ITER_H

#include <cstdint>

#include "tensor_impl.h"

namespace tae {

// Iteration helpers shared across ops. They operate on type-erased buffers
// (char*); ops are expected to compute byte offsets per element from the
// element-unit strides held by TensorImpl, and to reinterpret_cast inside the
// kernel.
//
// Two access patterns are common:
//   - Unary / output ops walk one tensor + one output of the same shape.
//   - Binary ops walk two inputs (after broadcast) + one output.

// Walks `out` in C-order. At each step, calls `body` with the running output
// index (k) and the per-tensor element offsets (a_off, out_off). All offsets
// include the tensors' own .offset() base.
template <typename Body>
void iterate_unary(const TensorImpl* a, TensorImpl* out, Body body) {
    const auto&        shape    = out->shape();
    const std::int64_t ndim     = out->ndim();
    const auto&        a_str    = a->strides();
    const auto&        out_str  = out->strides();
    const std::int64_t total    = out->numel();
    const std::int64_t a_base   = a->offset();
    const std::int64_t out_base = out->offset();

    if (total == 0) return;

    Dims idx(static_cast<std::size_t>(ndim), 0);
    for (std::int64_t k = 0; k < total; ++k) {
        std::int64_t a_off = a_base, out_off = out_base;
        for (std::int64_t d = 0; d < ndim; ++d) {
            a_off   += idx[d] * a_str[d];
            out_off += idx[d] * out_str[d];
        }
        body(k, a_off, out_off);
        for (std::int64_t d = ndim - 1; d >= 0; --d) {
            if (++idx[d] < shape[d]) break;
            idx[d] = 0;
        }
    }
}

template <typename Body>
void iterate_binary(const TensorImpl* a, const TensorImpl* b, TensorImpl* out, Body body) {
    const auto&        shape    = out->shape();
    const std::int64_t ndim     = out->ndim();
    const auto&        a_str    = a->strides();
    const auto&        b_str    = b->strides();
    const auto&        out_str  = out->strides();
    const std::int64_t total    = out->numel();
    const std::int64_t a_base   = a->offset();
    const std::int64_t b_base   = b->offset();
    const std::int64_t out_base = out->offset();

    if (total == 0) return;

    Dims idx(static_cast<std::size_t>(ndim), 0);
    for (std::int64_t k = 0; k < total; ++k) {
        std::int64_t a_off = a_base, b_off = b_base, out_off = out_base;
        for (std::int64_t d = 0; d < ndim; ++d) {
            a_off   += idx[d] * a_str[d];
            b_off   += idx[d] * b_str[d];
            out_off += idx[d] * out_str[d];
        }
        body(k, a_off, b_off, out_off);
        for (std::int64_t d = ndim - 1; d >= 0; --d) {
            if (++idx[d] < shape[d]) break;
            idx[d] = 0;
        }
    }
}

}  // namespace tae

#endif  // TENSOR_ITER_H
