#ifndef TENSOR_BROADCAST_H
#define TENSOR_BROADCAST_H

#include "tensor_impl.h"

namespace tae {

// Computes the broadcast shape of `a` and `b` per numpy right-alignment rules:
//   1. Right-align the two shapes (pad the shorter on the left with 1s).
//   2. At each aligned position, the two sizes must be equal OR one must be 1.
//   3. The result size is the maximum of the two.
// Throws std::invalid_argument on incompatibility.
Dims broadcast_shape(const Dims& a, const Dims& b);

// Returns a view of `t` reshaped to `target_shape` via stride-zero expansion.
// `t`'s shape must be broadcastable to `target_shape` (after right-alignment).
// No data is copied — the returned TensorImpl shares t's Storage.
TensorImpl* broadcast_to(const TensorImpl* t, const Dims& target_shape);

}  // namespace tae

#endif  // TENSOR_BROADCAST_H
