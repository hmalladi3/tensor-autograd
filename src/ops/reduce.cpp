#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "../dispatch.h"
#include "../ops.h"
#include "../tensor_impl.h"

namespace tae::ops {

namespace {

// Builds the output shape after reducing the given axes.
//   keepdim = true:  reduced dims become size 1
//   keepdim = false: reduced dims are removed
Dims build_output_shape(const Dims& in_shape, const Dims& axes, bool keepdim) {
    const std::int64_t ndim = static_cast<std::int64_t>(in_shape.size());
    SmallVec<bool, kMaxRank> is_reduced(in_shape.size(), false);
    if (axes.empty()) {
        // Empty axes list reduces over everything (OPS-043). Mirrors split_layout.
        for (auto& f : is_reduced) f = true;
    } else {
        for (auto a : axes) {
            const std::int64_t d = normalize_dim(a, ndim);
            is_reduced[static_cast<std::size_t>(d)] = true;
        }
    }
    Dims out;
    for (std::int64_t d = 0; d < ndim; ++d) {
        if (is_reduced[static_cast<std::size_t>(d)]) {
            if (keepdim) out.push_back(1);
        } else {
            out.push_back(in_shape[static_cast<std::size_t>(d)]);
        }
    }
    return out;
}

// Splits dims into kept vs reduced and computes the per-kept and per-reduced
// shape/stride lists for `in`. Used by the actual reduction loop.
struct ReductionLayout {
    Dims kept_shape;     Dims kept_strides;
    Dims reduced_shape;  Dims reduced_strides;
};

ReductionLayout split_layout(const TensorImpl* a, const Dims& axes) {
    const std::int64_t ndim = a->ndim();
    SmallVec<bool, kMaxRank> is_reduced(static_cast<std::size_t>(ndim), false);
    if (axes.empty()) {
        // Empty axes list reduces over everything.
        for (auto& f : is_reduced) f = true;
    } else {
        for (auto axis : axes) {
            is_reduced[static_cast<std::size_t>(normalize_dim(axis, ndim))] = true;
        }
    }
    ReductionLayout layout;
    for (std::int64_t d = 0; d < ndim; ++d) {
        if (is_reduced[static_cast<std::size_t>(d)]) {
            layout.reduced_shape.push_back(a->shape()[static_cast<std::size_t>(d)]);
            layout.reduced_strides.push_back(a->strides()[static_cast<std::size_t>(d)]);
        } else {
            layout.kept_shape.push_back(a->shape()[static_cast<std::size_t>(d)]);
            layout.kept_strides.push_back(a->strides()[static_cast<std::size_t>(d)]);
        }
    }
    return layout;
}

// Walks each kept-cell, accumulating across the reduced dims using `accumulate`,
// then writing the result via `finalize`. Both kernels run inside a dispatch.
template <typename T, typename TOut, typename Init, typename Accum, typename Finalize>
void run_reduce(const TensorImpl* a, TensorImpl* out, const ReductionLayout& L,
                Init init, Accum accumulate, Finalize finalize) {
    const auto* in_data  = static_cast<const T*>(a->storage()->data());
    auto*       out_data = static_cast<TOut*>(out->storage()->data());

    std::int64_t kept_total = 1;
    for (auto s : L.kept_shape) kept_total *= s;
    std::int64_t red_total = 1;
    for (auto s : L.reduced_shape) red_total *= s;

    Dims kept_idx(L.kept_shape.size(), 0);
    Dims red_idx (L.reduced_shape.size(), 0);

    for (std::int64_t k = 0; k < kept_total; ++k) {
        std::int64_t base = a->offset();
        for (std::size_t d = 0; d < L.kept_shape.size(); ++d) {
            base += kept_idx[d] * L.kept_strides[d];
        }

        auto acc = init();
        std::int64_t arg = 0;
        std::fill(red_idx.begin(), red_idx.end(), 0);
        for (std::int64_t r = 0; r < red_total; ++r) {
            std::int64_t off = base;
            for (std::size_t d = 0; d < L.reduced_shape.size(); ++d) {
                off += red_idx[d] * L.reduced_strides[d];
            }
            accumulate(acc, arg, in_data[off], r);
            for (std::int64_t d = static_cast<std::int64_t>(L.reduced_shape.size()) - 1;
                 d >= 0; --d) {
                if (++red_idx[static_cast<std::size_t>(d)]
                    < L.reduced_shape[static_cast<std::size_t>(d)]) break;
                red_idx[static_cast<std::size_t>(d)] = 0;
            }
        }
        out_data[k] = finalize(acc, arg, red_total);

        for (std::int64_t d = static_cast<std::int64_t>(L.kept_shape.size()) - 1; d >= 0; --d) {
            if (++kept_idx[static_cast<std::size_t>(d)]
                < L.kept_shape[static_cast<std::size_t>(d)]) break;
            kept_idx[static_cast<std::size_t>(d)] = 0;
        }
    }
}

}  // namespace

TensorImpl* reduce(ReduceOp op, const TensorImpl* a, const Dims& axes, bool keepdim) {
    const ReductionLayout L = split_layout(a, axes);
    Dims out_shape = build_output_shape(a->shape(), axes, keepdim);
    const DType out_dtype = (op == OP_ARGMAX) ? DType::Int64 : a->dtype();
    auto* out = TensorImpl::make_contiguous(out_shape, out_dtype);

    dispatch_floating(a->dtype(), [&](auto tag) {
        using T = decltype(tag);
        switch (op) {
            case OP_SUM:
                run_reduce<T, T>(a, out, L,
                    [] { return T{0}; },
                    [](T& acc, std::int64_t& /*arg*/, T v, std::int64_t /*r*/) { acc += v; },
                    [](T acc, std::int64_t /*arg*/, std::int64_t /*n*/) { return acc; });
                return;
            case OP_MEAN:
                run_reduce<T, T>(a, out, L,
                    [] { return T{0}; },
                    [](T& acc, std::int64_t& /*arg*/, T v, std::int64_t /*r*/) { acc += v; },
                    [](T acc, std::int64_t /*arg*/, std::int64_t n) {
                        return static_cast<T>(acc / static_cast<T>(n));
                    });
                return;
            case OP_MAX_R:
                run_reduce<T, T>(a, out, L,
                    [] { return std::numeric_limits<T>::lowest(); },
                    [](T& acc, std::int64_t& /*arg*/, T v, std::int64_t /*r*/) {
                        if (v > acc) acc = v;
                    },
                    [](T acc, std::int64_t /*arg*/, std::int64_t /*n*/) { return acc; });
                return;
            case OP_ARGMAX:
                run_reduce<T, std::int64_t>(a, out, L,
                    [] { return std::numeric_limits<T>::lowest(); },
                    [](T& acc, std::int64_t& arg, T v, std::int64_t r) {
                        if (v > acc) { acc = v; arg = r; }
                    },
                    [](T /*acc*/, std::int64_t arg, std::int64_t /*n*/) { return arg; });
                return;
        }
        throw std::invalid_argument("reduce: unknown op_id");
    });
    return out;
}

}  // namespace tae::ops
