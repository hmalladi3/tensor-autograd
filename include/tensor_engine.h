/*
 * tensor_engine.h — public C ABI of the tensor engine.
 *
 * Refcount discipline (the rule sheet):
 *   1. Returned handles have refcount 1. Callers must `tensor_decref` once.
 *   2. Argument handles are borrowed. Engine functions do not change their refcount.
 *   3. Views share Storage, not TensorImpl. Each view is a fresh handle (refcount 1).
 *   4. `tensor_from_buffer` always copies. The engine retains no pointer into caller memory.
 *   5. `tensor_copy_to_buffer` always copies. The engine never lets a raw Storage pointer escape.
 *
 * Error model:
 *   Every fallible function returns tensor_status_t. 0 = success; nonzero = error.
 *   On error, output parameters are not written (handle outputs are nulled).
 *   `tensor_last_error()` returns a thread-local string set by the most recent engine call.
 */

#ifndef TENSOR_ENGINE_H
#define TENSOR_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque tag. The actual type behind the pointer is tae::TensorImpl (C++),
 * but the tag is intentionally distinct so the C name does not collide with
 * the C++ class when tests or other internal code use both headers. */
typedef struct tensor_handle_opaque* tensor_handle_t;
typedef int32_t                      tensor_status_t;
typedef int32_t                      tensor_dtype_t;

/* Status codes. */
enum {
    TENSOR_OK                 = 0,
    TENSOR_OOM                = 1,
    TENSOR_SHAPE_MISMATCH     = 2,
    TENSOR_DTYPE_MISMATCH     = 3,
    TENSOR_INDEX_OUT_OF_RANGE = 4,
    TENSOR_INVALID_ARGUMENT   = 5,
    TENSOR_INTERNAL_ERROR     = 6,
};

/* Dtype tags. Numerically stable across the boundary. */
enum {
    DTYPE_FLOAT32 = 0,
    DTYPE_INT64   = 1,
    DTYPE_BOOL    = 2,
};

/* Op IDs for the polymorphic op families. Values match src/op_ids.h exactly;
 * the C names carry a TENSOR_ prefix to avoid colliding with the internal C++
 * enums of the same numeric meaning when both headers are visible (tests). */
enum { TENSOR_OP_ADD = 0, TENSOR_OP_SUB = 1, TENSOR_OP_MUL = 2, TENSOR_OP_DIV = 3,
       TENSOR_OP_POW = 4, TENSOR_OP_MAX = 5, TENSOR_OP_MIN = 6, TENSOR_OP_EQ = 7,
       TENSOR_OP_LT  = 8, TENSOR_OP_GT  = 9 };
enum { TENSOR_OP_NEG = 0, TENSOR_OP_EXP = 1, TENSOR_OP_LOG = 2, TENSOR_OP_SQRT = 3,
       TENSOR_OP_RELU = 4, TENSOR_OP_SIGMOID = 5, TENSOR_OP_TANH = 6 };
enum { TENSOR_OP_SUM = 0, TENSOR_OP_MEAN = 1, TENSOR_OP_MAX_R = 2, TENSOR_OP_ARGMAX = 3 };
enum { TENSOR_OP_UNIFORM = 0, TENSOR_OP_NORMAL = 1 };
enum { TENSOR_OP_AXPY = 0 };

/* ---- Error reporting ---- */
const char* tensor_last_error(void);

/* ---- Lifetime ---- */
void tensor_incref(tensor_handle_t t);
void tensor_decref(tensor_handle_t t);

/* ---- Construction ---- */
tensor_status_t tensor_empty      (const int64_t* shape, int64_t ndim, tensor_dtype_t dtype,
                                   tensor_handle_t* out);
tensor_status_t tensor_full       (const int64_t* shape, int64_t ndim, tensor_dtype_t dtype,
                                   double value, tensor_handle_t* out);
tensor_status_t tensor_from_buffer(const void* data, const int64_t* shape, int64_t ndim,
                                   tensor_dtype_t dtype, tensor_handle_t* out);
tensor_status_t tensor_arange     (double start, double stop, double step, tensor_dtype_t dtype,
                                   tensor_handle_t* out);
tensor_status_t tensor_random     (int32_t op_id, const int64_t* shape, int64_t ndim,
                                   tensor_dtype_t dtype, double p1, double p2,
                                   tensor_handle_t* out);

void tensor_seed(uint64_t seed);

/* ---- Metadata ---- */
int64_t        tensor_ndim   (tensor_handle_t t);
int64_t        tensor_numel  (tensor_handle_t t);
tensor_dtype_t tensor_dtype  (tensor_handle_t t);
void           tensor_shape  (tensor_handle_t t, int64_t* out);   /* caller-allocated, length ndim */
void           tensor_strides(tensor_handle_t t, int64_t* out);

/* ---- Data egress ---- */
tensor_status_t tensor_copy_to_buffer(tensor_handle_t t, void* dst, size_t dst_nbytes);

/* ---- Views ---- */
tensor_status_t tensor_reshape   (tensor_handle_t t, const int64_t* shape, int64_t ndim,
                                  tensor_handle_t* out);
tensor_status_t tensor_transpose (tensor_handle_t t, int64_t dim_a, int64_t dim_b,
                                  tensor_handle_t* out);
tensor_status_t tensor_slice     (tensor_handle_t t, int64_t dim, int64_t start, int64_t stop,
                                  int64_t step, tensor_handle_t* out);
tensor_status_t tensor_contiguous(tensor_handle_t t, tensor_handle_t* out);
tensor_status_t tensor_cast      (tensor_handle_t t, tensor_dtype_t dtype,
                                  tensor_handle_t* out);

/* ---- Ops ---- */
tensor_status_t op_binary (int32_t op_id, tensor_handle_t a, tensor_handle_t b,
                           tensor_handle_t* out);
tensor_status_t op_unary  (int32_t op_id, tensor_handle_t a, tensor_handle_t* out);
tensor_status_t op_reduce (int32_t op_id, tensor_handle_t a, const int64_t* axes, int64_t naxes,
                           int32_t keepdim, tensor_handle_t* out);
tensor_status_t op_matmul (tensor_handle_t a, tensor_handle_t b, tensor_handle_t* out);
tensor_status_t op_gather (tensor_handle_t a, int64_t dim, tensor_handle_t indices,
                           tensor_handle_t* out);
tensor_status_t op_inplace(int32_t op_id, tensor_handle_t dst, tensor_handle_t src, double alpha);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_ENGINE_H */
