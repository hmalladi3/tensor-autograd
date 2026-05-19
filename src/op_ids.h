#ifndef TENSOR_OP_IDS_H
#define TENSOR_OP_IDS_H

#include <cstdint>

namespace tae {

// Stable op_id values used by the polymorphic FFI dispatch families.
// Numerically pinned because they cross the C ABI; adding new ops appends
// to the end of a family rather than renumbering.

enum BinaryOp : std::int32_t {
    OP_ADD = 0, OP_SUB = 1, OP_MUL = 2, OP_DIV = 3, OP_POW = 4,
    OP_MAX = 5, OP_MIN = 6, OP_EQ  = 7, OP_LT  = 8, OP_GT  = 9,
};

enum UnaryOp : std::int32_t {
    OP_NEG     = 0,
    OP_EXP     = 1,
    OP_LOG     = 2,
    OP_SQRT    = 3,
    OP_RELU    = 4,
    OP_SIGMOID = 5,
    OP_TANH    = 6,
};

enum ReduceOp : std::int32_t {
    OP_SUM    = 0,
    OP_MEAN   = 1,
    OP_MAX_R  = 2,
    OP_ARGMAX = 3,
};

enum RandomOp : std::int32_t {
    OP_UNIFORM = 0,
    OP_NORMAL  = 1,
};

enum InplaceOp : std::int32_t {
    OP_AXPY = 0,  // dst += alpha * src
};

}  // namespace tae

#endif  // TENSOR_OP_IDS_H
