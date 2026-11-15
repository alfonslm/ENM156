#pragma once
// One STFT frame in, one model input frame out; one model output frame in,
// one windowed time frame out. Both framings and the streaming processor
// (stream.h) are built from this. Follows plugin/tools/README.md, "Features
// and synthesis", in float32 like PyTorch.
//
//   denoiser::FrameCodec codec(model.info());               // allocates
//   const float* x = codec.analyse(shortSamples, longSamples);   // n_in floats
//   network.process(x, pred);                                     // n_out floats
//   const float* frame = codec.synthesise(pred);                  // n_fft floats, irfft(...) * window
//
// analyse() and synthesise() don't allocate, lock or throw. A FrameCodec
// holds FFT scratch memory: one per thread.

#include "denoiser/fft.h"
#include "denoiser/model_file.h"

#include <complex>
#include <vector>

namespace denoiser {

// torch.hann_window(n): the periodic Hann window, computed in float32 the way
// PyTorch does (cos(i * float(2 pi / n)) * -0.5 + 0.5)
std::vector<float> hannWindow(int n);

class FrameCodec {
public:
    explicit FrameCodec(const ModelInfo& info);

    const ModelInfo& info() const noexcept { return info_; }

    // shortFrame: n_fft samples; longFrame: long_n_fft samples (ignored, may be
    // nullptr, without a long window). Returns the model input frame (n_in
    // floats), valid until the next analyse().
    const float* analyse(const float* shortFrame, const float* longFrame) noexcept;

    // pred: the model's output for the frame of the last analyse() (n_out
    // floats). Returns irfft(denoised spectrum, n_fft) * window: n_fft
    // samples, to be overlap-added; valid until the next synthesise().
    const float* synthesise(const float* pred) noexcept;

    // Like synthesise(), from a spectrum (n_freq bins) instead of a model
    // output: synthesiseSpectrum(spectrum()) is the analysis/synthesis
    // round trip without a network ("reconstruction")
    const float* synthesiseSpectrum(const std::complex<float>* spec) noexcept;

    // Intermediate results of the last analyse()/synthesise(), for tests and
    // inspection (vectors file names in brackets)
    const std::complex<float>* spectrum() const noexcept { return spec_.data(); }   // n_freq bins [spec]
    const float* longMagnitude() const noexcept { return longMag_.data(); }         // n_freq; empty without a long window [long_mag]
    const float* input() const noexcept { return x_.data(); }                       // n_in [x]
    const std::complex<float>* outSpectrum() const noexcept { return outSpec_.data(); }  // n_freq bins [out_spec]
    const float* window() const noexcept { return window_.data(); }                 // n_fft [window]
    const float* longWindow() const noexcept { return longWindow_.data(); }         // long_n_fft [long_window]

private:
    ModelInfo info_;
    RealFft fft_;
    RealFft longFft_;  // size 2 (unused) without a long window
    std::vector<float> window_, longWindow_;
    std::vector<float> buffer_;  // windowed samples, max(n_fft, long_n_fft)
    std::vector<std::complex<float>> spec_, longSpec_, outSpec_;
    std::vector<float> mag_;               // |spec|
    std::vector<float> compReal_, compImag_;  // compressed spectrum (complex_mask)
    std::vector<float> longMag_;
    std::vector<float> x_;
    std::vector<float> frame_;  // n_fft output samples
};

}  // namespace denoiser
