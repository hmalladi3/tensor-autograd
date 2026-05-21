#include "test_main.h"

#include "small_vec.h"

using tae::SmallVec;

TEST(small_vec, default_construct_is_empty) {
    SmallVec<int, 8> v;
    CHECK_EQ(v.size(), 0u);
    CHECK(v.empty());
}

TEST(small_vec, initializer_list_construct) {
    SmallVec<int, 8> v{1, 2, 3};
    CHECK_EQ(v.size(), 3u);
    CHECK_EQ(v[0], 1);
    CHECK_EQ(v[1], 2);
    CHECK_EQ(v[2], 3);
}

TEST(small_vec, fill_construct) {
    SmallVec<int, 8> v(4, 7);
    CHECK_EQ(v.size(), 4u);
    for (auto x : v) CHECK_EQ(x, 7);
}

TEST(small_vec, push_back_then_resize) {
    SmallVec<int, 8> v;
    v.push_back(10);
    v.push_back(20);
    CHECK_EQ(v.size(), 2u);
    v.resize(5, -1);
    CHECK_EQ(v.size(), 5u);
    CHECK_EQ(v[2], -1);
    CHECK_EQ(v[4], -1);
}

TEST(small_vec, equality) {
    SmallVec<int, 8> a{1, 2, 3};
    SmallVec<int, 8> b{1, 2, 3};
    SmallVec<int, 8> c{1, 2, 4};
    SmallVec<int, 8> d{1, 2};
    CHECK(a == b);
    CHECK(a != c);
    CHECK(a != d);
}

TEST(small_vec, overflow_throws) {
    SmallVec<int, 2> v{1, 2};
    CHECK_THROWS(v.push_back(3));
    CHECK_THROWS((SmallVec<int, 2>{1, 2, 3}));
    CHECK_THROWS(v.resize(5));
}
