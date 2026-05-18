#ifndef TENSOR_DTYPE_H
#define TENSOR_DTYPE_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "tensor_engine.h"

namespace tae {

enum class DType : std::uint8_t {
    Float32 = DTYPE_FLOAT32,
    Int64   = DTYPE_INT64,
    Bool    = DTYPE_BOOL,
};

constexpr std::size_t dtype_size(DType d) {
    switch (d) {
        case DType::Float32: return sizeof(float);
        case DType::Int64:   return sizeof(std::int64_t);
        case DType::Bool:    return sizeof(std::uint8_t);
    }
    throw std::invalid_argument("dtype_size: unknown dtype");
}

}  // namespace tae

#endif  // TENSOR_DTYPE_H
