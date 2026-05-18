#ifndef TENSOR_STORAGE_H
#define TENSOR_STORAGE_H

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "dtype.h"

namespace tae {

// Storage owns one aligned buffer and tags it with a dtype. Refcounted so that
// multiple TensorImpl views can share the same bytes without copies. The
// refcount is atomic because backward() can touch the same Storage from
// multiple Tensor objects in arbitrary order, and a future move to release
// the Python GIL during op execution should not introduce a race.
class Storage {
public:
    // Allocates `numel * dtype_size(dtype)` bytes via the aligned allocator.
    // Throws std::bad_alloc on failure.
    Storage(std::int64_t numel, DType dtype);

    Storage(const Storage&)            = delete;
    Storage& operator=(const Storage&) = delete;
    Storage(Storage&&)                 = delete;
    Storage& operator=(Storage&&)      = delete;

    void incref() noexcept;

    // Returns true if the call dropped the last reference (and the Storage
    // has been destroyed). After a true return, the pointer is dangling.
    bool decref() noexcept;

    void*         data()       noexcept { return data_; }
    const void*   data() const noexcept { return data_; }
    std::int64_t  numel()  const noexcept { return numel_; }
    std::size_t   nbytes() const noexcept { return nbytes_; }
    DType         dtype()  const noexcept { return dtype_; }

    // Test-only inspection of the refcount. Not part of the public API.
    std::int32_t refcount() const noexcept {
        return refcount_.load(std::memory_order_relaxed);
    }

private:
    ~Storage();  // private — destruction goes through decref()

    void*                data_;
    std::size_t          nbytes_;
    std::int64_t         numel_;
    DType                dtype_;
    std::atomic<std::int32_t> refcount_;
};

}  // namespace tae

#endif  // TENSOR_STORAGE_H
