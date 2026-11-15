#pragma once
// Whole-clip denoising, pipeline.enhance(): the same length out as in, no
// delay. For the command-line tool and ARA analysis, not for the audio thread
// (allocates, may throw).
//
//   std::vector<float> y = denoiser::denoiseClip(model.info(), network, wav.data(), wav.size());
//
// Both framings:
// - causal: the stream (StreamProcessor) run over the clip and enough zeros
//   after it, shifted back by n_fft - hop. clipFrames() frames.
// - centred (realtime = false models): torch.stft's centring (the clip
//   reflect-padded by n_fft / 2, the long window by long_n_fft / 2) and
//   torch.istft's overlap-add (divided by the window envelope, trimmed by
//   n_fft / 2).
//
// The network is reset first and then given every frame once, in order.

#include "denoiser/model_file.h"
#include "denoiser/network.h"

#include <cstddef>
#include <functional>
#include <vector>

namespace denoiser {

// Number of frames the network sees for a clip of n samples:
// causal floor((n + n_fft - hop) / hop) + 1, centred floor(n / hop) + 1
std::size_t clipFrames(const ModelInfo& info, std::size_t n);

// Shortest clip the framing can take: 0 for causal; for centred, reflect
// padding needs more samples than half the widest window
std::size_t minClipSamples(const ModelInfo& info);

// Called after each frame with (frames done, frames in all); return false to
// cancel (denoiseClip then returns an empty vector). Optional.
using ClipProgress = std::function<bool(std::size_t done, std::size_t total)>;

// in: n samples at info.sampleRate -> n denoised samples. Throws
// std::invalid_argument if n < minClipSamples(info).
std::vector<float> denoiseClip(const ModelInfo& info, FrameNetwork& network, const float* in, std::size_t n,
                               const ClipProgress& progress = nullptr);

}  // namespace denoiser
