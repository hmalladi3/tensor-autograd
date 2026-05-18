#ifndef TENSOR_ALLOCATOR_H
#define TENSOR_ALLOCATOR_H

#include <cstddef>

namespace tae {

// All engine buffers are aligned to this boundary (one x86 cache line; comfortable
// for AVX-512 and ARM NEON). Smaller dtypes still get this alignment — wasted only
// on the tail of the buffer.
inline constexpr std::size_t kEngineAlignment = 64;

// Returns nullptr on allocation failure. The caller is responsible for releasing
// the returned pointer via aligned_free.
void* aligned_alloc(std::size_t bytes, std::size_t align = kEngineAlignment);

void aligned_free(void* ptr);

}  // namespace tae

#endif  // TENSOR_ALLOCATOR_H
