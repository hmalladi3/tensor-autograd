#include "prng.h"

namespace tae {

namespace {

inline std::uint64_t rotl(std::uint64_t x, int k) noexcept {
    return (x << k) | (x >> (64 - k));
}

// SplitMix64 — seed expander. Standard pairing with xoshiro to derive a full
// 256-bit state from a single 64-bit seed (the constructor's input).
inline std::uint64_t splitmix64(std::uint64_t& x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

}  // namespace

void Xoshiro256pp::reseed(std::uint64_t seed) noexcept {
    std::uint64_t s = seed;
    for (auto& word : s_) word = splitmix64(s);
}

std::uint64_t Xoshiro256pp::next_u64() noexcept {
    const std::uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
    const std::uint64_t t = s_[1] << 17;

    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];

    s_[2] ^= t;
    s_[3]  = rotl(s_[3], 45);

    return result;
}

double Xoshiro256pp::next_double() noexcept {
    // Standard "53-bit float" extraction: top 53 bits → [0, 1).
    const std::uint64_t bits = next_u64() >> 11;
    return static_cast<double>(bits) * (1.0 / (1ULL << 53));
}

Xoshiro256pp& engine_prng() noexcept {
    // Default-seed value is arbitrary; the choice doesn't matter as long as it's
    // fixed across runs (so absent `tensor_seed`, random construction is
    // reproducible). Picked to be visually distinct from common zeros.
    thread_local Xoshiro256pp instance{0xA4C1A29DEADBEEF0ULL};
    return instance;
}

}  // namespace tae
