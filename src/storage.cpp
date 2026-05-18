#include "storage.h"

#include <new>

#include "allocator.h"

namespace tae {

Storage::Storage(std::int64_t numel, DType dtype)
    : data_(nullptr),
      nbytes_(static_cast<std::size_t>(numel) * dtype_size(dtype)),
      numel_(numel),
      dtype_(dtype),
      refcount_(1) {
    if (nbytes_ > 0) {
        data_ = aligned_alloc(nbytes_);
        if (data_ == nullptr) throw std::bad_alloc();
    }
}

Storage::~Storage() {
    aligned_free(data_);
}

void Storage::incref() noexcept {
    refcount_.fetch_add(1, std::memory_order_relaxed);
}

bool Storage::decref() noexcept {
    // Decrementing requires acquire semantics so that any reads of the buffer
    // by other threads happen-before the destructor runs here.
    const std::int32_t prev = refcount_.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) {
        delete this;
        return true;
    }
    return false;
}

}  // namespace tae
