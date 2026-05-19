#include "tensor_impl.h"

#include <cstring>
#include <stdexcept>

namespace tae {

// --------- free helpers ---------

Dims contiguous_strides_for(const Dims& shape) {
    Dims strides(shape.size());
    if (shape.empty()) return strides;
    strides[shape.size() - 1] = 1;
    for (std::int64_t i = static_cast<std::int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

bool is_contiguous_layout(const Dims& shape, const Dims& strides) {
    if (shape.size() != strides.size()) return false;
    if (shape.empty()) return true;  // 0-dim is trivially contiguous

    std::int64_t expected = 1;
    for (std::int64_t i = static_cast<std::int64_t>(shape.size()) - 1; i >= 0; --i) {
        // Size-1 dimensions impose no constraint on the stride — any value works
        // because the dimension contributes one offset (zero). PyTorch follows
        // the same convention.
        if (shape[i] != 1 && strides[i] != expected) return false;
        expected *= shape[i];
    }
    return true;
}

std::int64_t normalize_dim(std::int64_t dim, std::int64_t ndim) {
    const std::int64_t resolved = dim < 0 ? dim + ndim : dim;
    if (resolved < 0 || resolved >= ndim) {
        throw std::out_of_range("dim out of range");
    }
    return resolved;
}

// Walks `src` in C-order via strided access and copies its elements densely
// into `dst`. Used by contiguous() and reshape's fallback path.
static void copy_strided_to_contiguous(const TensorImpl* src, void* dst) {
    const std::int64_t ndim     = src->ndim();
    const auto&        shape    = src->shape();
    const auto&        strides  = src->strides();
    const std::size_t  elem     = dtype_size(src->dtype());
    const char*        src_base = static_cast<const char*>(src->storage()->data())
                                  + static_cast<std::size_t>(src->offset()) * elem;
    char*              out      = static_cast<char*>(dst);
    const std::int64_t total    = src->numel();

    if (total == 0) return;

    Dims idx(static_cast<std::size_t>(ndim), 0);
    for (std::int64_t k = 0; k < total; ++k) {
        std::int64_t src_off = 0;
        for (std::int64_t d = 0; d < ndim; ++d) src_off += idx[d] * strides[d];
        std::memcpy(out + static_cast<std::size_t>(k) * elem,
                    src_base + static_cast<std::size_t>(src_off) * elem,
                    elem);
        for (std::int64_t d = ndim - 1; d >= 0; --d) {
            if (++idx[d] < shape[d]) break;
            idx[d] = 0;
        }
    }
}

// --------- TensorImpl ---------

TensorImpl::TensorImpl(Storage* storage, Dims shape, Dims strides, std::int64_t offset,
                       DType dtype)
    : storage_(storage),
      offset_(offset),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      dtype_(dtype),
      is_contiguous_(is_contiguous_layout(shape_, strides_)),
      refcount_(1) {
    if (shape_.size() != strides_.size()) {
        throw std::invalid_argument("shape and strides must have the same rank");
    }
    if (storage_ != nullptr) storage_->incref();
}

TensorImpl::~TensorImpl() {
    if (storage_ != nullptr) storage_->decref();
}

TensorImpl* TensorImpl::make_contiguous(Dims shape, DType dtype) {
    Dims strides = contiguous_strides_for(shape);
    std::int64_t total = 1;
    for (auto s : shape) total *= s;

    auto* storage = new Storage(total, dtype);  // refcount 1
    auto* t = new TensorImpl(storage, std::move(shape), std::move(strides), 0, dtype);
    storage->decref();  // TensorImpl's ctor incref'd; we drop our extra reference.
    return t;
}

void TensorImpl::incref() noexcept {
    refcount_.fetch_add(1, std::memory_order_relaxed);
}

bool TensorImpl::decref() noexcept {
    const std::int32_t prev = refcount_.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) {
        delete this;
        return true;
    }
    return false;
}

std::int64_t TensorImpl::numel() const noexcept {
    if (shape_.empty()) return 1;  // 0-dim scalar holds exactly one element
    std::int64_t n = 1;
    for (auto s : shape_) n *= s;
    return n;
}

TensorImpl* TensorImpl::transpose(std::int64_t dim_a, std::int64_t dim_b) const {
    const std::int64_t a = normalize_dim(dim_a, ndim());
    const std::int64_t b = normalize_dim(dim_b, ndim());
    Dims new_shape   = shape_;
    Dims new_strides = strides_;
    std::swap(new_shape[a],   new_shape[b]);
    std::swap(new_strides[a], new_strides[b]);
    return new TensorImpl(storage_, std::move(new_shape), std::move(new_strides), offset_, dtype_);
}

TensorImpl* TensorImpl::permute(const Dims& dims) const {
    if (static_cast<std::int64_t>(dims.size()) != ndim()) {
        throw std::invalid_argument("permute: dims must have same rank as tensor");
    }
    Dims new_shape(dims.size());
    Dims new_strides(dims.size());
    // Each output position i takes from input position dims[i].
    // We also require that each input dimension is listed exactly once.
    SmallVec<bool, kMaxRank> seen(dims.size(), false);
    for (std::size_t i = 0; i < dims.size(); ++i) {
        const std::int64_t d = normalize_dim(dims[i], ndim());
        if (seen[static_cast<std::size_t>(d)]) {
            throw std::invalid_argument("permute: duplicate dim");
        }
        seen[static_cast<std::size_t>(d)] = true;
        new_shape[i]   = shape_[static_cast<std::size_t>(d)];
        new_strides[i] = strides_[static_cast<std::size_t>(d)];
    }
    return new TensorImpl(storage_, std::move(new_shape), std::move(new_strides), offset_, dtype_);
}

TensorImpl* TensorImpl::squeeze(std::int64_t dim) const {
    const std::int64_t d = normalize_dim(dim, ndim());
    if (shape_[static_cast<std::size_t>(d)] != 1) {
        throw std::invalid_argument("squeeze: dimension is not size 1");
    }
    Dims new_shape;
    Dims new_strides;
    for (std::int64_t i = 0; i < ndim(); ++i) {
        if (i == d) continue;
        new_shape.push_back(shape_[static_cast<std::size_t>(i)]);
        new_strides.push_back(strides_[static_cast<std::size_t>(i)]);
    }
    return new TensorImpl(storage_, std::move(new_shape), std::move(new_strides), offset_, dtype_);
}

TensorImpl* TensorImpl::unsqueeze(std::int64_t dim) const {
    // unsqueeze accepts dim in [0, ndim] — one past the end is "insert at the back."
    const std::int64_t resolved = dim < 0 ? dim + ndim() + 1 : dim;
    if (resolved < 0 || resolved > ndim()) {
        throw std::out_of_range("unsqueeze: dim out of range");
    }
    Dims new_shape;
    Dims new_strides;
    for (std::int64_t i = 0, j = 0; j < ndim() + 1; ++j) {
        if (j == resolved) {
            new_shape.push_back(1);
            new_strides.push_back(0);  // size-1 stride is unconstrained; 0 is the neutral choice
        } else {
            new_shape.push_back(shape_[static_cast<std::size_t>(i)]);
            new_strides.push_back(strides_[static_cast<std::size_t>(i)]);
            ++i;
        }
    }
    return new TensorImpl(storage_, std::move(new_shape), std::move(new_strides), offset_, dtype_);
}

TensorImpl* TensorImpl::slice(std::int64_t dim, std::int64_t start, std::int64_t stop,
                              std::int64_t step) const {
    const std::int64_t d = normalize_dim(dim, ndim());
    if (step <= 0) throw std::invalid_argument("slice: step must be positive");

    const std::int64_t size = shape_[static_cast<std::size_t>(d)];
    if (start < 0) start += size;
    if (stop  < 0) stop  += size;
    start = std::max<std::int64_t>(0, std::min(start, size));
    stop  = std::max<std::int64_t>(0, std::min(stop,  size));

    const std::int64_t new_size = (stop > start) ? (stop - start + step - 1) / step : 0;

    Dims new_shape   = shape_;
    Dims new_strides = strides_;
    new_shape[static_cast<std::size_t>(d)]   = new_size;
    new_strides[static_cast<std::size_t>(d)] = strides_[static_cast<std::size_t>(d)] * step;
    const std::int64_t new_offset = offset_ + start * strides_[static_cast<std::size_t>(d)];

    return new TensorImpl(storage_, std::move(new_shape), std::move(new_strides), new_offset, dtype_);
}

TensorImpl* TensorImpl::reshape(const Dims& new_shape) const {
    std::int64_t product = 1;
    for (auto s : new_shape) {
        if (s < 0) throw std::invalid_argument("reshape: negative dimension");
        product *= s;
    }
    if (product != numel()) {
        throw std::invalid_argument("reshape: total element count must be preserved");
    }

    // Fast path: a contiguous tensor reshapes to any compatible shape as a view.
    if (is_contiguous_) {
        Dims new_strides = contiguous_strides_for(new_shape);
        return new TensorImpl(storage_, new_shape, std::move(new_strides), offset_, dtype_);
    }

    // Fallback: materialize a contiguous copy, then view-reshape that. The
    // intermediate is dropped after the new view increfs the freshly allocated
    // Storage.
    TensorImpl* contig = contiguous();
    Dims new_strides   = contiguous_strides_for(new_shape);
    auto* result       = new TensorImpl(contig->storage(), new_shape, std::move(new_strides),
                                        contig->offset(), dtype_);
    contig->decref();
    return result;
}

TensorImpl* TensorImpl::contiguous() const {
    if (is_contiguous_) {
        const_cast<TensorImpl*>(this)->incref();
        return const_cast<TensorImpl*>(this);
    }

    const std::int64_t total = numel();
    auto* new_storage = new Storage(total, dtype_);  // refcount 1
    copy_strided_to_contiguous(this, new_storage->data());

    Dims new_strides = contiguous_strides_for(shape_);
    auto* result = new TensorImpl(new_storage, shape_, std::move(new_strides), 0, dtype_);
    new_storage->decref();  // ctor incref'd; drop the construction reference
    return result;
}

}  // namespace tae
