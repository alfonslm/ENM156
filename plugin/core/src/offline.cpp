#include "denoiser/offline.h"

#include "denoiser/features.h"
#include "denoiser/stream.h"

#include <stdexcept>
#include <string>

namespace denoiser {

std::size_t clipFrames(const ModelInfo& info, std::size_t n) {
    const auto hop = static_cast<std::size_t>(info.hop);
    if (info.framing == Framing::Causal)
        return (n + static_cast<std::size_t>(info.nFft - info.hop)) / hop + 1;
    return n / hop + 1;
}

std::size_t minClipSamples(const ModelInfo& info) {
    if (info.framing == Framing::Causal)
        return 0;
    return static_cast<std::size_t>(info.widestWindow() / 2 + 1);
}

namespace {

std::vector<float> causalClip(const ModelInfo& info, FrameNetwork& network, const float* in, std::size_t n,
                              const ClipProgress& progress) {
    StreamProcessor stream(info);
    const auto hop = static_cast<std::size_t>(info.hop);
    const auto delay = static_cast<std::size_t>(stream.delay());
    const std::size_t frames = clipFrames(info, n);  // enough to flush the last sample out
    std::vector<float> result(n), chunk(hop), out(hop);
    std::size_t produced = 0;  // stream samples so far
    network.reset();
    for (std::size_t t = 0; t < frames; ++t) {
        for (std::size_t i = 0; i < hop; ++i) {
            const std::size_t s = t * hop + i;
            chunk[i] = s < n ? in[s] : 0.0f;
        }
        stream.process(chunk.data(), out.data(), network);
        for (std::size_t i = 0; i < hop; ++i, ++produced)
            if (produced >= delay && produced - delay < n)
                result[produced - delay] = out[i];
        if (progress && !progress(t + 1, frames))
            return {};
    }
    return result;
}

// numpy.pad(x, p, mode="reflect"): mirrored without repeating the edge sample
std::vector<float> reflectPad(const float* x, std::size_t n, std::size_t p) {
    std::vector<float> out(n + 2 * p);
    const auto last = static_cast<std::ptrdiff_t>(n) - 1;
    for (std::size_t i = 0; i < out.size(); ++i) {
        std::ptrdiff_t j = static_cast<std::ptrdiff_t>(i) - static_cast<std::ptrdiff_t>(p);
        if (j < 0) j = -j;
        if (j > last) j = 2 * last - j;
        out[i] = x[j];
    }
    return out;
}

std::vector<float> centredClip(const ModelInfo& info, FrameNetwork& network, const float* in, std::size_t n,
                               const ClipProgress& progress) {
    FrameCodec codec(info);
    const auto hop = static_cast<std::size_t>(info.hop);
    const auto nFft = static_cast<std::size_t>(info.nFft);
    const auto longN = static_cast<std::size_t>(info.longNFft);
    const std::size_t frames = clipFrames(info, n);
    const std::size_t length = (frames - 1) * hop + nFft;
    const std::size_t start = nFft / 2;

    // torch.istft's window envelope; it refuses a clip where it is ~0
    std::vector<float> envelope(length, 0.0f);
    const float* w = codec.window();
    for (std::size_t t = 0; t < frames; ++t)
        for (std::size_t i = 0; i < nFft; ++i)
            envelope[t * hop + i] += w[i] * w[i];
    for (std::size_t i = start; i < start + n; ++i)
        if (!(envelope[i] > 1e-11f))
            throw std::invalid_argument("the window envelope is zero at sample " + std::to_string(i - start) +
                                        " (hop too large for n_fft)");

    const std::vector<float> shortPadded = reflectPad(in, n, nFft / 2);
    const std::vector<float> longPadded = info.longFactor ? reflectPad(in, n, longN / 2) : std::vector<float>();
    std::vector<float> y(length, 0.0f), pred(static_cast<std::size_t>(info.nOut));
    network.reset();
    for (std::size_t t = 0; t < frames; ++t) {
        const float* x = codec.analyse(shortPadded.data() + t * hop,
                                       info.longFactor ? longPadded.data() + t * hop : nullptr);
        network.process(x, pred.data());
        const float* frame = codec.synthesise(pred.data());
        for (std::size_t i = 0; i < nFft; ++i)
            y[t * hop + i] += frame[i];
        if (progress && !progress(t + 1, frames))
            return {};
    }
    std::vector<float> result(n);
    for (std::size_t i = 0; i < n; ++i)
        result[i] = y[start + i] / envelope[start + i];
    return result;
}

}  // namespace

std::vector<float> denoiseClip(const ModelInfo& info, FrameNetwork& network, const float* in, std::size_t n,
                               const ClipProgress& progress) {
    if (n < minClipSamples(info))
        throw std::invalid_argument("clip of " + std::to_string(n) + " samples is too short for centred frames (needs " +
                                    std::to_string(minClipSamples(info)) + ")");
    if (info.framing == Framing::Causal)
        return causalClip(info, network, in, n, progress);
    return centredClip(info, network, in, n, progress);
}

}  // namespace denoiser
