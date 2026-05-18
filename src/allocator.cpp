#include "allocator.h"

#include <cstdlib>

#if defined(_WIN32)
    #include <malloc.h>
#endif

namespace tae {

void* aligned_alloc(std::size_t bytes, std::size_t align) {
    if (bytes == 0) return nullptr;

    // Many platforms require the request size to be a multiple of the alignment.
    // Round up — the caller's logical byte count is whatever it asked for.
    const std::size_t rounded = (bytes + align - 1) & ~(align - 1);

#if defined(_WIN32)
    return _aligned_malloc(rounded, align);
#elif defined(__APPLE__)
    // posix_memalign is the portable POSIX answer. macOS lacks std::aligned_alloc
    // until very recent SDKs and even then the size/alignment constraints differ.
    void* p = nullptr;
    if (posix_memalign(&p, align, rounded) != 0) return nullptr;
    return p;
#else
    return std::aligned_alloc(align, rounded);
#endif
}

void aligned_free(void* ptr) {
    if (ptr == nullptr) return;
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

}  // namespace tae
