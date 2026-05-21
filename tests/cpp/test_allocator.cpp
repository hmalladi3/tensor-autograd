#include "test_main.h"

#include <cstdint>

#include "allocator.h"

// `tae::aligned_alloc` deliberately shares its name with libc's. Callers must
// qualify to disambiguate; we never `using`-import it for that reason.

// @spec MEM-001, MEM-003
TEST(allocator, returns_aligned_pointer) {
    void* p = tae::aligned_alloc(128);
    REQUIRE(p != nullptr);
    const auto addr = reinterpret_cast<std::uintptr_t>(p);
    CHECK_EQ(addr % tae::kEngineAlignment, 0u);
    tae::aligned_free(p);
}

// @spec MEM-003
TEST(allocator, aligns_small_allocations_too) {
    for (std::size_t bytes : {1u, 7u, 17u, 63u}) {
        void* p = tae::aligned_alloc(bytes);
        REQUIRE(p != nullptr);
        const auto addr = reinterpret_cast<std::uintptr_t>(p);
        CHECK_EQ(addr % tae::kEngineAlignment, 0u);
        tae::aligned_free(p);
    }
}

// @spec MEM-001
TEST(allocator, zero_size_returns_null) {
    CHECK_EQ(tae::aligned_alloc(0), static_cast<void*>(nullptr));
}

// @spec MEM-004
TEST(allocator, free_null_is_safe) {
    tae::aligned_free(nullptr);  // must not crash
}

TEST(allocator, allocations_are_writable) {
    constexpr std::size_t bytes = 1024;
    auto* p = static_cast<unsigned char*>(tae::aligned_alloc(bytes));
    REQUIRE(p != nullptr);
    for (std::size_t i = 0; i < bytes; ++i) p[i] = static_cast<unsigned char>(i & 0xFF);
    for (std::size_t i = 0; i < bytes; ++i) CHECK_EQ(p[i], static_cast<unsigned char>(i & 0xFF));
    tae::aligned_free(p);
}
