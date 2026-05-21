#include "test_main.h"

#include <cstdint>

#include "dtype.h"
#include "storage.h"

using tae::DType;
using tae::dtype_size;
using tae::Storage;

// @spec MEM-010, MEM-011
TEST(storage, constructed_with_correct_size_and_refcount) {
    auto* s = new Storage(/*numel=*/100, DType::Float32);
    REQUIRE(s != nullptr);
    CHECK_EQ(s->numel(),  static_cast<std::int64_t>(100));
    CHECK_EQ(s->nbytes(), 100u * sizeof(float));
    CHECK_EQ(s->dtype(),  DType::Float32);
    CHECK_EQ(s->refcount(), 1);
    s->decref();
}

// @spec MEM-006 (translated: nbytes == numel * dtype_size)
TEST(storage, nbytes_matches_numel_times_dtype_size) {
    for (auto dt : {DType::Float32, DType::Int64, DType::Bool}) {
        auto* s = new Storage(13, dt);
        CHECK_EQ(s->nbytes(), 13u * dtype_size(dt));
        s->decref();
    }
}

// @spec MEM-020, MEM-021, MEM-022, MEM-023
TEST(storage, refcount_lifecycle) {
    auto* s = new Storage(8, DType::Float32);
    REQUIRE(s->refcount() == 1);

    s->incref();
    CHECK_EQ(s->refcount(), 2);

    const bool freed1 = s->decref();
    CHECK(!freed1);
    CHECK_EQ(s->refcount(), 1);

    const bool freed2 = s->decref();
    CHECK(freed2);
    // s is now dangling; do not touch.
}

TEST(storage, multiple_increfs_and_decrefs_balance) {
    auto* s = new Storage(4, DType::Int64);
    for (int i = 0; i < 9; ++i) s->incref();    // refcount now 10
    int freed_count = 0;
    for (int i = 0; i < 10; ++i) {
        if (s->decref()) ++freed_count;
    }
    CHECK_EQ(freed_count, 1);
}

TEST(storage, zero_numel_does_not_allocate) {
    auto* s = new Storage(0, DType::Float32);
    CHECK_EQ(s->nbytes(), 0u);
    CHECK_EQ(s->data(),   static_cast<void*>(nullptr));
    s->decref();
}

TEST(storage, data_buffer_is_writable_and_readable) {
    auto* s    = new Storage(16, DType::Float32);
    auto* data = static_cast<float*>(s->data());
    for (int i = 0; i < 16; ++i) data[i] = static_cast<float>(i) * 0.5f;
    for (int i = 0; i < 16; ++i) CHECK_NEAR(data[i], static_cast<float>(i) * 0.5f, 1e-6);
    s->decref();
}
