#pragma once
// Real FFT of any even size, with the conventions of torch.fft.rfft / irfft.
//
//   denoiser::RealFft fft(1280);            // allocates; do this at setup
//   fft.forward(samples, spectrum);         // 1280 floats -> 641 bins
//   fft.inverse(spectrum, samples);         // 641 bins -> 1280 floats
//
// forward() and inverse() do not allocate, lock or throw, so they may be
// called on the audio thread. One RealFft holds scratch memory: don't call it
// from two threads at once (use one per thread instead).

#include <complex>
#include <cstddef>

namespace denoiser {

class RealFft {
public:
    // n: transform size, even and >= 2 (e.g. n_fft = 128, long window = 1280).
    // Throws std::invalid_argument otherwise, std::bad_alloc if out of memory.
    explicit RealFft(int n);
    ~RealFft();

    RealFft(const RealFft&) = delete;
    RealFft& operator=(const RealFft&) = delete;
    RealFft(RealFft&& other) noexcept;
    RealFft& operator=(RealFft&& other) noexcept;

    int size() const noexcept { return n_; }
    int bins() const noexcept { return n_ / 2 + 1; }

    // in: size() samples -> out: bins() complex values, unnormalised
    // (out[k] = sum_t in[t] * exp(-2 pi i k t / n), like torch.fft.rfft)
    void forward(const float* in, std::complex<float>* out) noexcept;

    // in: bins() complex values -> out: size() samples, scaled by 1 / n so
    // that inverse(forward(x)) == x (like torch.fft.irfft(in, n)). The
    // imaginary parts of the first and last bin are ignored.
    void inverse(const std::complex<float>* in, float* out) noexcept;

private:
    void release() noexcept;

    int n_ = 0;
    void* forwardCfg_ = nullptr;  // kiss_fftr_cfg, kept out of this header
    void* inverseCfg_ = nullptr;
};

}  // namespace denoiser
