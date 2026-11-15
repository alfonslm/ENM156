#include "denoiser/core.h"

#include <algorithm>

namespace denoiser {

const char* version() noexcept { return DENOISER_VERSION; }

void passThrough(const float* in, float* out, std::size_t n) noexcept {
    if (in != out)
        std::copy_n(in, n, out);
}

}  // namespace denoiser
