#include "tensor_engine.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "dtype.h"
#include "op_ids.h"
#include "ops.h"
#include "small_vec.h"
#include "tensor_impl.h"

// Mark every C ABI entry point as default-visibility so it survives
// -fvisibility=hidden. Internal symbols stay hidden, keeping the export table
// small and preventing accidental ABI commitments.
#if defined(__GNUC__) || defined(__clang__)
    #define TAE_API __attribute__((visibility("default")))
#else
    #define TAE_API
#endif

namespace {

using tae::DType;
using tae::Dims;
using tae::TensorImpl;

// Per-thread error string. Spec: cleared at the start of every fallible call,
// populated on exception, retrievable via tensor_last_error().
thread_local std::string g_last_error;

inline TensorImpl*     to_impl  (tensor_handle_t h) noexcept {
    return reinterpret_cast<TensorImpl*>(h);
}
inline tensor_handle_t to_handle(TensorImpl* t)     noexcept {
    return reinterpret_cast<tensor_handle_t>(t);
}

// Wraps an FFI entry point: clears last_error, runs `fn`, and translates any
// C++ exception into a status code + last_error message. C++ exceptions never
// propagate across the boundary.
template <typename Fn>
tensor_status_t guarded(Fn fn) noexcept {
    g_last_error.clear();
    try {
        return fn();
    } catch (const std::bad_alloc&) {
        g_last_error = "out of memory";
        return TENSOR_OOM;
    } catch (const std::out_of_range& e) {
        g_last_error = e.what();
        return TENSOR_INDEX_OUT_OF_RANGE;
    } catch (const std::invalid_argument& e) {
        g_last_error = e.what();
        return TENSOR_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return TENSOR_INTERNAL_ERROR;
    } catch (...) {
        g_last_error = "unknown C++ exception";
        return TENSOR_INTERNAL_ERROR;
    }
}

Dims dims_from(const int64_t* shape, int64_t ndim) {
    Dims d(static_cast<std::size_t>(ndim));
    for (int64_t i = 0; i < ndim; ++i) {
        d[static_cast<std::size_t>(i)] = shape[i];
    }
    return d;
}

}  // namespace

extern "C" {

// ============================================================================
// Error reporting
// ============================================================================

TAE_API const char* tensor_last_error(void) {
    return g_last_error.c_str();
}

// ============================================================================
// Lifetime
// ============================================================================

TAE_API void tensor_incref(tensor_handle_t t) {
    if (t != nullptr) to_impl(t)->incref();
}

TAE_API void tensor_decref(tensor_handle_t t) {
    if (t != nullptr) to_impl(t)->decref();
}

// ============================================================================
// Construction
// ============================================================================

TAE_API tensor_status_t tensor_empty(const int64_t* shape, int64_t ndim, tensor_dtype_t dtype,
                                     tensor_handle_t* out) {
    return guarded([&]() {
        *out = nullptr;
        *out = to_handle(TensorImpl::make_contiguous(dims_from(shape, ndim),
                                                     static_cast<DType>(dtype)));
        return TENSOR_OK;
    });
}

TAE_API tensor_status_t tensor_full(const int64_t* shape, int64_t ndim, tensor_dtype_t dtype,
                                    double value, tensor_handle_t* out) {
    return guarded([&]() {
        *out = nullptr;
        *out = to_handle(tae::ops::full(dims_from(shape, ndim),
                                        static_cast<DType>(dtype), value));
        return TENSOR_OK;
    });
}

TAE_API tensor_status_t tensor_from_buffer(const void* data, const int64_t* shape, int64_t ndim,
                                           tensor_dtype_t dtype, tensor_handle_t* out) {
    return guarded([&]() {
        *out = nullptr;
        auto* t = TensorImpl::make_contiguous(dims_from(shape, ndim),
                                              static_cast<DType>(dtype));
        const std::size_t bytes = static_cast<std::size_t>(t->numel())
                                  * tae::dtype_size(t->dtype());
        if (bytes > 0) std::memcpy(t->storage()->data(), data, bytes);
        *out = to_handle(t);
        return TENSOR_OK;
    });
}

TAE_API tensor_status_t tensor_arange(double start, double stop, double step,
                                      tensor_dtype_t dtype, tensor_handle_t* out) {
    return guarded([&]() {
        *out = nullptr;
        *out = to_handle(tae::ops::arange(start, stop, step, static_cast<DType>(dtype)));
        return TENSOR_OK;
    });
}

TAE_API tensor_status_t tensor_random(int32_t op_id, const int64_t* shape, int64_t ndim,
                                      tensor_dtype_t dtype, double p1, double p2,
                                      tensor_handle_t* out) {
    return guarded([&]() {
        *out = nullptr;
        *out = to_handle(tae::ops::random(static_cast<tae::RandomOp>(op_id),
                                          dims_from(shape, ndim),
                                          static_cast<DType>(dtype), p1, p2));
        return TENSOR_OK;
    });
}

TAE_API void tensor_seed(uint64_t seed) {
    tae::ops::seed(seed);
}

// ============================================================================
// Metadata
// ============================================================================

TAE_API int64_t tensor_ndim(tensor_handle_t t) {
    return to_impl(t)->ndim();
}

TAE_API int64_t tensor_numel(tensor_handle_t t) {
    return to_impl(t)->numel();
}

TAE_API tensor_dtype_t tensor_dtype(tensor_handle_t t) {
    return static_cast<tensor_dtype_t>(to_impl(t)->dtype());
}

TAE_API void tensor_shape(tensor_handle_t t, int64_t* out) {
    const auto& s = to_impl(t)->shape();
    for (std::size_t i = 0; i < s.size(); ++i) out[i] = s[i];
}

TAE_API void tensor_strides(tensor_handle_t t, int64_t* out) {
    const auto& s = to_impl(t)->strides();
    for (std::size_t i = 0; i < s.size(); ++i) out[i] = s[i];
}

// ============================================================================
// Data egress
// ============================================================================

TAE_API tensor_status_t tensor_copy_to_buffer(tensor_handle_t t, void* dst, size_t dst_nbytes) {
    return guarded([&]() {
        auto* impl = to_impl(t);
        const std::size_t needed = static_cast<std::size_t>(impl->numel())
                                   * tae::dtype_size(impl->dtype());
        if (dst_nbytes < needed) {
            throw std::invalid_argument("tensor_copy_to_buffer: dst buffer too small");
        }
        if (needed == 0) return TENSOR_OK;

        if (impl->is_contiguous()) {
            const auto elem = tae::dtype_size(impl->dtype());
            std::memcpy(dst,
                        static_cast<const char*>(impl->storage()->data())
                            + static_cast<std::size_t>(impl->offset()) * elem,
                        needed);
        } else {
            // Walk in C-order regardless of source layout — reuse contiguous().
            auto* contig = impl->contiguous();
            std::memcpy(dst, contig->storage()->data(), needed);
            contig->decref();
        }
        return TENSOR_OK;
    });
}

// ============================================================================
// Views
// ============================================================================

TAE_API tensor_status_t tensor_reshape(tensor_handle_t t, const int64_t* shape, int64_t ndim,
                                       tensor_handle_t* out) {
    return guarded([&]() {
        *out = nullptr;
        *out = to_handle(to_impl(t)->reshape(dims_from(shape, ndim)));
        return TENSOR_OK;
    });
}

TAE_API tensor_status_t tensor_transpose(tensor_handle_t t, int64_t dim_a, int64_t dim_b,
                                         tensor_handle_t* out) {
    return guarded([&]() {
        *out = nullptr;
        *out = to_handle(to_impl(t)->transpose(dim_a, dim_b));
        return TENSOR_OK;
    });
}

TAE_API tensor_status_t tensor_slice(tensor_handle_t t, int64_t dim, int64_t start, int64_t stop,
                                     int64_t step, tensor_handle_t* out) {
    return guarded([&]() {
        *out = nullptr;
        *out = to_handle(to_impl(t)->slice(dim, start, stop, step));
        return TENSOR_OK;
    });
}

TAE_API tensor_status_t tensor_contiguous(tensor_handle_t t, tensor_handle_t* out) {
    return guarded([&]() {
        *out = nullptr;
        *out = to_handle(to_impl(t)->contiguous());
        return TENSOR_OK;
    });
}

TAE_API tensor_status_t tensor_cast(tensor_handle_t t, tensor_dtype_t dtype,
                                    tensor_handle_t* out) {
    return guarded([&]() {
        *out = nullptr;
        *out = to_handle(tae::ops::cast(to_impl(t), static_cast<DType>(dtype)));
        return TENSOR_OK;
    });
}

// ============================================================================
// Ops
// ============================================================================

TAE_API tensor_status_t op_binary(int32_t op_id, tensor_handle_t a, tensor_handle_t b,
                                  tensor_handle_t* out) {
    return guarded([&]() {
        *out = nullptr;
        *out = to_handle(tae::ops::binary(static_cast<tae::BinaryOp>(op_id),
                                          to_impl(a), to_impl(b)));
        return TENSOR_OK;
    });
}

TAE_API tensor_status_t op_unary(int32_t op_id, tensor_handle_t a, tensor_handle_t* out) {
    return guarded([&]() {
        *out = nullptr;
        *out = to_handle(tae::ops::unary(static_cast<tae::UnaryOp>(op_id), to_impl(a)));
        return TENSOR_OK;
    });
}

TAE_API tensor_status_t op_reduce(int32_t op_id, tensor_handle_t a, const int64_t* axes,
                                  int64_t naxes, int32_t keepdim, tensor_handle_t* out) {
    return guarded([&]() {
        *out = nullptr;
        Dims ax(static_cast<std::size_t>(naxes));
        for (int64_t i = 0; i < naxes; ++i) {
            ax[static_cast<std::size_t>(i)] = axes[i];
        }
        *out = to_handle(tae::ops::reduce(static_cast<tae::ReduceOp>(op_id),
                                          to_impl(a), ax, keepdim != 0));
        return TENSOR_OK;
    });
}

TAE_API tensor_status_t op_matmul(tensor_handle_t a, tensor_handle_t b, tensor_handle_t* out) {
    return guarded([&]() {
        *out = nullptr;
        *out = to_handle(tae::ops::matmul(to_impl(a), to_impl(b)));
        return TENSOR_OK;
    });
}

TAE_API tensor_status_t op_gather(tensor_handle_t a, int64_t dim, tensor_handle_t indices,
                                  tensor_handle_t* out) {
    return guarded([&]() {
        *out = nullptr;
        *out = to_handle(tae::ops::gather(to_impl(a), dim, to_impl(indices)));
        return TENSOR_OK;
    });
}

TAE_API tensor_status_t op_inplace(int32_t op_id, tensor_handle_t dst, tensor_handle_t src,
                                   double alpha) {
    return guarded([&]() {
        tae::ops::inplace(static_cast<tae::InplaceOp>(op_id),
                          to_impl(dst), to_impl(src), alpha);
        return TENSOR_OK;
    });
}

}  // extern "C"
