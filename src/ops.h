#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include <cstdint>

#include "op_ids.h"
#include "tensor_impl.h"

// Public C++ API for engine ops. Every function returns a fresh TensorImpl*
// with refcount 1 (caller is responsible for `decref`), except for op_inplace
// which mutates its `dst` argument in place and returns nothing.

namespace tae::ops {

// ---- Construction.
TensorImpl* full  (const Dims& shape, DType dtype, double value);
TensorImpl* arange(double start, double stop, double step, DType dtype);
TensorImpl* random(RandomOp op, const Dims& shape, DType dtype, double p1, double p2);
void        seed  (std::uint64_t seed);

// ---- Elementwise.
TensorImpl* unary (UnaryOp  op, const TensorImpl* a);
TensorImpl* binary(BinaryOp op, const TensorImpl* a, const TensorImpl* b);

// ---- Reductions.
TensorImpl* reduce(ReduceOp op, const TensorImpl* a, const Dims& axes, bool keepdim);

// ---- Matmul.
TensorImpl* matmul(const TensorImpl* a, const TensorImpl* b);

// ---- Movement.
TensorImpl* cast(const TensorImpl* a, DType dtype);

// ---- Indexing.
TensorImpl* gather(const TensorImpl* a, std::int64_t dim, const TensorImpl* indices);

// ---- In-place mutation.
void inplace(InplaceOp op, TensorImpl* dst, const TensorImpl* src, double alpha);

}  // namespace tae::ops

#endif  // TENSOR_OPS_H
