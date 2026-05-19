#ifndef TENSOR_DISPATCH_H
#define TENSOR_DISPATCH_H

#include <cstdint>
#include <stdexcept>
#include <utility>

#include "dtype.h"

namespace tae {

// Tag-dispatch helpers. Each takes a callable that receives a default-constructed
// instance of the C++ type matching the dtype tag, letting the body deduce `T`
// via a generic lambda: `dispatch_floating(t->dtype(), [&](auto tag) { using T = decltype(tag); ... });`

template <typename Fn>
void dispatch_floating(DType dt, Fn&& fn) {
    switch (dt) {
        case DType::Float32: std::forward<Fn>(fn)(float{}); return;
        default: throw std::invalid_argument("op requires a floating dtype");
    }
}

template <typename Fn>
void dispatch_numeric(DType dt, Fn&& fn) {
    switch (dt) {
        case DType::Float32: std::forward<Fn>(fn)(float{}); return;
        case DType::Int64:   std::forward<Fn>(fn)(std::int64_t{}); return;
        default: throw std::invalid_argument("op requires a numeric dtype");
    }
}

template <typename Fn>
void dispatch_all(DType dt, Fn&& fn) {
    switch (dt) {
        case DType::Float32: std::forward<Fn>(fn)(float{}); return;
        case DType::Int64:   std::forward<Fn>(fn)(std::int64_t{}); return;
        case DType::Bool:    std::forward<Fn>(fn)(std::uint8_t{}); return;
    }
    throw std::invalid_argument("unknown dtype");
}

}  // namespace tae

#endif  // TENSOR_DISPATCH_H
