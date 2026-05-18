#ifndef TENSOR_SMALL_VEC_H
#define TENSOR_SMALL_VEC_H

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>

namespace tae {

// Fixed-capacity inline vector. Shape and stride vectors live here — they are
// short (≤8 dims in every realistic case) and never need heap allocation at
// that size. Overflow is a hard error rather than a silent fallback, because
// crossing this threshold means a model with >N dimensions which v1 does not
// claim to support.
template <typename T, std::size_t N>
class SmallVec {
public:
    SmallVec() noexcept : size_(0) {}

    SmallVec(std::initializer_list<T> init) : size_(init.size()) {
        if (init.size() > N) throw std::length_error("SmallVec capacity exceeded");
        std::copy(init.begin(), init.end(), data_);
    }

    explicit SmallVec(std::size_t n, T fill = T{}) : size_(n) {
        if (n > N) throw std::length_error("SmallVec capacity exceeded");
        std::fill(data_, data_ + n, fill);
    }

    std::size_t          size()  const noexcept { return size_; }
    bool                 empty() const noexcept { return size_ == 0; }
    static constexpr std::size_t capacity() noexcept { return N; }

    T*       data()       noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    T&       operator[](std::size_t i)       noexcept { return data_[i]; }
    const T& operator[](std::size_t i) const noexcept { return data_[i]; }

    T*       begin()       noexcept { return data_; }
    T*       end()         noexcept { return data_ + size_; }
    const T* begin() const noexcept { return data_; }
    const T* end()   const noexcept { return data_ + size_; }

    void push_back(T v) {
        if (size_ >= N) throw std::length_error("SmallVec capacity exceeded");
        data_[size_++] = v;
    }

    void resize(std::size_t n, T fill = T{}) {
        if (n > N) throw std::length_error("SmallVec capacity exceeded");
        if (n > size_) std::fill(data_ + size_, data_ + n, fill);
        size_ = n;
    }

    bool operator==(const SmallVec& other) const noexcept {
        if (size_ != other.size_) return false;
        return std::equal(begin(), end(), other.begin());
    }
    bool operator!=(const SmallVec& other) const noexcept { return !(*this == other); }

private:
    T           data_[N];
    std::size_t size_;
};

}  // namespace tae

#endif  // TENSOR_SMALL_VEC_H
