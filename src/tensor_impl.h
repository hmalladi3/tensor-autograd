#ifndef TENSOR_TENSOR_IMPL_H
#define TENSOR_TENSOR_IMPL_H

#include <atomic>
#include <cstdint>

#include "dtype.h"
#include "small_vec.h"
#include "storage.h"

namespace tae {

// Maximum rank a tensor can have in v1. Picked to keep shape/stride vectors
// inline in two cache lines while comfortably covering every realistic model.
inline constexpr std::size_t kMaxRank = 8;

using Dims = SmallVec<std::int64_t, kMaxRank>;

// Computes C-contiguous strides for the given shape.
Dims contiguous_strides_for(const Dims& shape);

// Returns true iff (strides, shape) describes a C-contiguous tensor.
bool is_contiguous_layout(const Dims& shape, const Dims& strides);

// Resolves a possibly-negative dim index against `ndim`. Throws std::out_of_range
// if the resolved index falls outside [0, ndim).
std::int64_t normalize_dim(std::int64_t dim, std::int64_t ndim);

// TensorImpl is the metadata layer of the engine. It describes how to read a
// region of a Storage as a strided n-d array. Multiple TensorImpls can share
// the same Storage — that's how views work.
//
// TensorImpl is heap-allocated and refcounted. Construct with `new`, manage
// lifetime with `incref` / `decref`. The destructor is private; `decref` is
// the only way the object goes away.
class TensorImpl {
public:
    // Constructs a fresh contiguous TensorImpl with a freshly allocated Storage.
    // Returns a TensorImpl with refcount 1; the new Storage also has refcount 1.
    static TensorImpl* make_contiguous(Dims shape, DType dtype);

    // View constructor. Adopts the supplied shape/strides/offset and increments
    // the supplied Storage's refcount. Caller retains its own Storage reference.
    TensorImpl(Storage* storage, Dims shape, Dims strides, std::int64_t offset, DType dtype);

    TensorImpl(const TensorImpl&)            = delete;
    TensorImpl& operator=(const TensorImpl&) = delete;
    TensorImpl(TensorImpl&&)                 = delete;
    TensorImpl& operator=(TensorImpl&&)      = delete;

    void incref() noexcept;
    bool decref() noexcept;  // returns true if last reference dropped

    // ---- View operations.
    // Each returns a fresh TensorImpl* with refcount 1. Caller must decref.

    TensorImpl* transpose(std::int64_t dim_a, std::int64_t dim_b) const;
    TensorImpl* permute(const Dims& dims) const;
    TensorImpl* squeeze(std::int64_t dim) const;
    TensorImpl* unsqueeze(std::int64_t dim) const;
    TensorImpl* slice(std::int64_t dim, std::int64_t start, std::int64_t stop,
                      std::int64_t step) const;

    // reshape returns a view when the new shape is stride-compatible with the
    // current layout. When not compatible, the engine materializes a contiguous
    // copy first and reshapes that. The result is always a fresh TensorImpl*.
    TensorImpl* reshape(const Dims& new_shape) const;

    // contiguous() on a contiguous tensor returns `this` with refcount bumped
    // (no allocation). On a non-contiguous tensor, returns a fresh tensor over
    // a fresh Storage containing the elements in C-order.
    TensorImpl* contiguous() const;

    // ---- Metadata accessors.
    std::int64_t  ndim()  const noexcept { return static_cast<std::int64_t>(shape_.size()); }
    std::int64_t  numel() const noexcept;
    const Dims&   shape()   const noexcept { return shape_; }
    const Dims&   strides() const noexcept { return strides_; }
    std::int64_t  offset()  const noexcept { return offset_; }
    DType         dtype()   const noexcept { return dtype_; }
    bool          is_contiguous() const noexcept { return is_contiguous_; }
    Storage*      storage() const noexcept { return storage_; }

    // Test-only: inspect the TensorImpl's own refcount.
    std::int32_t refcount() const noexcept {
        return refcount_.load(std::memory_order_relaxed);
    }

private:
    ~TensorImpl();

    Storage*                  storage_;
    std::int64_t              offset_;
    Dims                      shape_;
    Dims                      strides_;
    DType                     dtype_;
    bool                      is_contiguous_;
    std::atomic<std::int32_t> refcount_;
};

}  // namespace tae

#endif  // TENSOR_TENSOR_IMPL_H
