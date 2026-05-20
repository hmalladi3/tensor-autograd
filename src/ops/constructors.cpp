#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "../dispatch.h"
#include "../ops.h"
#include "../tensor_impl.h"

namespace tae::ops {

TensorImpl* full(const Dims& shape, DType dtype, double value) {
    auto* t = TensorImpl::make_contiguous(shape, dtype);
    const std::int64_t n = t->numel();
    dispatch_numeric(dtype, [&](auto tag) {
        using T = decltype(tag);
        auto* data = static_cast<T*>(t->storage()->data());
        const T v  = static_cast<T>(value);
        for (std::int64_t i = 0; i < n; ++i) data[i] = v;
    });
    return t;
}

TensorImpl* arange(double start, double stop, double step, DType dtype) {
    if (step == 0.0) throw std::invalid_argument("arange: step must be non-zero");
    const double range = stop - start;
    if ((range > 0 && step < 0) || (range < 0 && step > 0)) {
        return TensorImpl::make_contiguous(Dims{0}, dtype);
    }
    const std::int64_t n = static_cast<std::int64_t>(std::ceil(range / step));
    auto* t = TensorImpl::make_contiguous(Dims{n}, dtype);
    dispatch_numeric(dtype, [&](auto tag) {
        using T = decltype(tag);
        auto* data = static_cast<T*>(t->storage()->data());
        for (std::int64_t i = 0; i < n; ++i) {
            data[i] = static_cast<T>(start + static_cast<double>(i) * step);
        }
    });
    return t;
}

}  // namespace tae::ops
