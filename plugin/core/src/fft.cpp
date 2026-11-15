#include "denoiser/fft.h"

#include <kiss_fftr.h>

#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace denoiser {

// kiss_fft_cpx is {float r, i}; std::complex<float> is guaranteed to have the
// same layout as float[2], so spectra can be passed through without copying
static_assert(sizeof(kiss_fft_cpx) == sizeof(std::complex<float>), "complex layout");
static_assert(sizeof(kiss_fft_scalar) == sizeof(float), "KISS FFT must be built for float");

RealFft::RealFft(int n) : n_(n) {
    if (n < 2 || n % 2 != 0)
        throw std::invalid_argument("RealFft: size must be even and >= 2, got " + std::to_string(n));
    forwardCfg_ = kiss_fftr_alloc(n, 0, nullptr, nullptr);
    inverseCfg_ = kiss_fftr_alloc(n, 1, nullptr, nullptr);
    if (!forwardCfg_ || !inverseCfg_) {
        release();
        throw std::bad_alloc();
    }
}

RealFft::~RealFft() { release(); }

RealFft::RealFft(RealFft&& other) noexcept
    : n_(std::exchange(other.n_, 0)),
      forwardCfg_(std::exchange(other.forwardCfg_, nullptr)),
      inverseCfg_(std::exchange(other.inverseCfg_, nullptr)) {}

RealFft& RealFft::operator=(RealFft&& other) noexcept {
    if (this != &other) {
        release();
        n_ = std::exchange(other.n_, 0);
        forwardCfg_ = std::exchange(other.forwardCfg_, nullptr);
        inverseCfg_ = std::exchange(other.inverseCfg_, nullptr);
    }
    return *this;
}

void RealFft::release() noexcept {
    kiss_fftr_free(forwardCfg_);
    kiss_fftr_free(inverseCfg_);
    forwardCfg_ = inverseCfg_ = nullptr;
}

void RealFft::forward(const float* in, std::complex<float>* out) noexcept {
    kiss_fftr(static_cast<kiss_fftr_cfg>(forwardCfg_), in, reinterpret_cast<kiss_fft_cpx*>(out));
}

void RealFft::inverse(const std::complex<float>* in, float* out) noexcept {
    kiss_fftri(static_cast<kiss_fftr_cfg>(inverseCfg_), reinterpret_cast<const kiss_fft_cpx*>(in), out);
    const float scale = 1.0f / static_cast<float>(n_);
    for (int i = 0; i < n_; ++i)
        out[i] *= scale;
}

}  // namespace denoiser
