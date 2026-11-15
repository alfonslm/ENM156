#pragma once
// Whole clips at any supported sample rate, for offline use (ARA analysis):
// resample a clip to the model's 16 kHz and back with the plugin's filter
// (resample.h), aligned with the input (no delay) and of the right length,
// and denoise a whole clip at its own rate.
//
//   auto filter = denoiser::ResampleFilter::create(rate);
//   std::vector<float> x16 = denoiser::clipToModelRate(*filter, x, n);       // modelRateLength(rate, n) samples
//   std::vector<float> y16 = denoiser::denoiseClip(info, net, x16.data(), x16.size());
//   std::vector<float> y   = denoiser::clipFromModelRate(*filter, y16.data(), y16.size(), n);
//
// or all of it (the reference the plugin's ARA analysis is tested against):
//
//   std::vector<float> y = denoiser::denoiseClipAtRate(info, net, rate, x, n, HighBand::Drop);
//
// Alignment: 16 kHz sample j is the band-limited input at time j / 16000 s;
// output sample m is the band-limited 16 kHz signal at time m / rate. The
// filters' delays (kResampleHalfLength each way) are removed; the signal is
// taken as silence before the clip and after it. At 16 kHz both directions
// copy samples unchanged.
//
// Not for the audio thread: these allocate (and may throw).

#include "denoiser/host_rate.h"
#include "denoiser/model_file.h"
#include "denoiser/network.h"
#include "denoiser/offline.h"
#include "denoiser/resample.h"

#include <cstddef>
#include <vector>

namespace denoiser {

// Samples at 16 kHz covering n samples at `rate`: ceil(n * 16000 / rate).
// Throws std::invalid_argument for an unsupported rate.
std::size_t modelRateLength(int rate, std::size_t n);

// n samples at filter.hostRate() -> modelRateLength(rate, n) samples at 16 kHz
std::vector<float> clipToModelRate(const ResampleFilter& filter, const float* in, std::size_t n);

// n16 samples at 16 kHz -> n samples at filter.hostRate() (n16 is normally
// modelRateLength(rate, n); samples past n16 are taken as silence)
std::vector<float> clipFromModelRate(const ResampleFilter& filter, const float* in16, std::size_t n16,
                                     std::size_t n);

// The whole offline path at `rate`: to 16 kHz, denoiseClip, back, aligned and
// the same length as the input. With HighBand::PassThrough the input minus its
// own round trip through 16 kHz is added (as the real-time plugin does), so
// the band above ~7.5 kHz passes unchanged. At 16 kHz this is denoiseClip.
// Throws std::invalid_argument for an unsupported rate or a clip too short
// for the model (modelRateLength(rate, n) < minClipSamples(info)).
std::vector<float> denoiseClipAtRate(const ModelInfo& info, FrameNetwork& network, int rate, const float* in,
                                     std::size_t n, HighBand highBand = HighBand::Drop,
                                     const ClipProgress& progress = nullptr);

}  // namespace denoiser
