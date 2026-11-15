#include "denoiser/stream.h"

#include <algorithm>
#include <stdexcept>

namespace denoiser {

namespace {
const ModelInfo& requireCausal(const ModelInfo& info) {
    if (info.framing != Framing::Causal)
        throw std::invalid_argument("this model uses centred frames (realtime = false): it can't run live");
    return info;
}
}  // namespace

StreamProcessor::StreamProcessor(const ModelInfo& info)
    : codec_(requireCausal(info)),
      history_(static_cast<std::size_t>(info.widestWindow())),
      pending_(static_cast<std::size_t>(info.nFft)),
      envelope_(static_cast<std::size_t>(info.nFft)),
      windowSq_(static_cast<std::size_t>(info.nFft)),
      pred_(static_cast<std::size_t>(info.nOut)) {
    for (std::size_t i = 0; i < windowSq_.size(); ++i)
        windowSq_[i] = codec_.window()[i] * codec_.window()[i];
    reset();
}

void StreamProcessor::reset() noexcept {
    std::fill(history_.begin(), history_.end(), 0.0f);
    std::fill(pending_.begin(), pending_.end(), 0.0f);
    std::fill(envelope_.begin(), envelope_.end(), 0.0f);
    std::fill(pred_.begin(), pred_.end(), 0.0f);
    samplesOut_ = 0;
}

const float* StreamProcessor::analyse(const float* in) noexcept {
    const int h = hop();
    const auto size = static_cast<std::ptrdiff_t>(history_.size());
    // Oldest hop samples out, new ones in at the end
    std::copy(history_.begin() + h, history_.end(), history_.begin());
    std::copy(in, in + h, history_.begin() + (size - h));
    const ModelInfo& info = codec_.info();
    const float* end = history_.data() + size;
    return codec_.analyse(end - info.nFft, info.longFactor ? end - info.longNFft : nullptr);
}

int StreamProcessor::overlapAdd(const float* frame, float* out) noexcept {
    const int h = hop(), n = codec_.info().nFft;
    for (int i = 0; i < n; ++i) {
        pending_[i] += frame[i];
        envelope_[i] += windowSq_[i];
    }
    // The oldest hop samples get no more frames: they are final
    for (int i = 0; i < h; ++i)
        out[i] = pending_[i] / std::max(envelope_[i], 1e-8f);
    std::copy(pending_.begin() + h, pending_.end(), pending_.begin());
    std::copy(envelope_.begin() + h, envelope_.end(), envelope_.begin());
    std::fill(pending_.end() - h, pending_.end(), 0.0f);
    std::fill(envelope_.end() - h, envelope_.end(), 0.0f);

    const std::int64_t left = std::max<std::int64_t>(delay() - samplesOut_, 0);
    samplesOut_ += h;
    return static_cast<int>(std::min<std::int64_t>(left, h));
}

int StreamProcessor::synthesise(const float* pred, float* out) noexcept {
    return overlapAdd(codec_.synthesise(pred), out);
}

int StreamProcessor::synthesisePassThrough(float* out) noexcept {
    return overlapAdd(codec_.synthesiseSpectrum(codec_.spectrum()), out);
}

int StreamProcessor::process(const float* in, float* out, FrameNetwork& network) noexcept {
    network.process(analyse(in), pred_.data());
    return synthesise(pred_.data(), out);
}

}  // namespace denoiser
