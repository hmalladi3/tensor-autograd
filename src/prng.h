#ifndef TENSOR_PRNG_H
#define TENSOR_PRNG_H

#include <cstdint>

namespace tae {

// xoshiro256++ — Vigna & Blackman, 2018. Public domain reference algorithm.
// Period 2^256 - 1, statistically excellent, fast. Picked over std::mt19937
// because (a) zero-deps story and (b) state fits in 32 bytes vs ~2.5 KiB,
// which matters for the per-thread state we hold.
class Xoshiro256pp {
public:
    explicit Xoshiro256pp(std::uint64_t seed) noexcept { reseed(seed); }

    void reseed(std::uint64_t seed) noexcept;

    // Returns a uniform 64-bit value.
    std::uint64_t next_u64() noexcept;

    // Returns a uniform double in [0, 1).
    double next_double() noexcept;

private:
    std::uint64_t s_[4];
};

// Thread-local engine PRNG. All `tensor_random` / `random_*` ops draw from this.
// Reset via `tensor_seed` (or `tae::ops::seed`).
Xoshiro256pp& engine_prng() noexcept;

}  // namespace tae

#endif  // TENSOR_PRNG_H
