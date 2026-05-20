#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "../ops.h"
#include "../prng.h"
#include "../tensor_impl.h"

namespace tae::ops {

void seed(std::uint64_t s) {
    engine_prng().reseed(s);
}

TensorImpl* random(RandomOp op, const Dims& shape, DType dtype, double p1, double p2) {
    if (dtype != DType::Float32) {
        throw std::invalid_argument("random: only Float32 is supported in v1");
    }
    auto* t = TensorImpl::make_contiguous(shape, dtype);
    auto* data = static_cast<float*>(t->storage()->data());
    const std::int64_t n = t->numel();
    auto& prng = engine_prng();

    switch (op) {
        case OP_UNIFORM: {
            const double lo = p1, hi = p2;
            if (!(hi > lo)) throw std::invalid_argument("random uniform: requires hi > lo");
            const double span = hi - lo;
            for (std::int64_t i = 0; i < n; ++i) {
                data[i] = static_cast<float>(lo + prng.next_double() * span);
            }
            return t;
        }
        case OP_NORMAL: {
            const double mean = p1, std_dev = p2;
            if (!(std_dev > 0)) throw std::invalid_argument("random normal: requires std > 0");
            // Box–Muller. Two uniform samples → two normal samples per pair.
            std::int64_t i = 0;
            while (i + 1 < n) {
                double u1 = prng.next_double();
                if (u1 < 1e-300) u1 = 1e-300;  // guard log(0)
                const double u2  = prng.next_double();
                const double mag = std::sqrt(-2.0 * std::log(u1));
                const double z0  = mag * std::cos(2.0 * M_PI * u2);
                const double z1  = mag * std::sin(2.0 * M_PI * u2);
                data[i]     = static_cast<float>(mean + std_dev * z0);
                data[i + 1] = static_cast<float>(mean + std_dev * z1);
                i += 2;
            }
            if (i < n) {  // odd tail
                double u1 = prng.next_double();
                if (u1 < 1e-300) u1 = 1e-300;
                const double u2  = prng.next_double();
                const double mag = std::sqrt(-2.0 * std::log(u1));
                const double z0  = mag * std::cos(2.0 * M_PI * u2);
                data[i] = static_cast<float>(mean + std_dev * z0);
            }
            return t;
        }
    }
    t->decref();
    throw std::invalid_argument("random: unknown op_id");
}

}  // namespace tae::ops
