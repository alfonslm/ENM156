#include "denoiser/offline_rate.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace denoiser {

std::size_t modelRateLength(int rate, std::size_t n) {
    const RateRatio r = hostRateRatio(rate);  // throws for an unsupported rate
    // 16 kHz sample j lies at grid time j * M; the clip ends at grid time n * L
    const auto L = static_cast<std::size_t>(r.up), M = static_cast<std::size_t>(r.down);
    return (n * L + M - 1) / M;
}

std::vector<float> clipToModelRate(const ResampleFilter& filter, const float* in, std::size_t n) {
    const std::size_t n16 = modelRateLength(filter.hostRate(), n);
    std::vector<float> out(n16);
    if (filter.bypass()) {
        std::copy(in, in + n, out.begin());
        return out;
    }
    // Downsampler output j is the band-limited input at 16 kHz time j - Q:
    // drop the first Q outputs
    Downsampler down;
    down.prepare(std::shared_ptr<const ResampleFilter>(&filter, [](const ResampleFilter*) {}));
    const std::size_t skip = static_cast<std::size_t>(kResampleHalfLength);
    std::size_t produced = 0;  // Downsampler outputs so far
    for (std::size_t i = 0; produced < n16 + skip; ++i) {
        float y = 0.0f;
        if (down.push(i < n ? in[i] : 0.0f, &y)) {
            if (produced >= skip)
                out[produced - skip] = y;
            ++produced;
        }
    }
    return out;
}

std::vector<float> clipFromModelRate(const ResampleFilter& filter, const float* in16, std::size_t n16,
                                     std::size_t n) {
    std::vector<float> out(n, 0.0f);
    if (filter.bypass()) {
        std::copy(in16, in16 + std::min(n, n16), out.begin());
        return out;
    }
    // Upsampler output k is the 16 kHz signal at grid time k L + offset - Q M.
    // Output m must be the signal at grid time m L: drop d = floor(Q M / L)
    // outputs, with offset = Q M - d L (0 <= offset < L).
    const std::int64_t L = filter.up(), M = filter.down();
    const std::int64_t QM = std::int64_t(kResampleHalfLength) * M;
    const std::int64_t d = QM / L;
    const int offset = int(QM - d * L);
    Upsampler up;
    up.prepare(std::shared_ptr<const ResampleFilter>(&filter, [](const ResampleFilter*) {}), offset);
    std::size_t pulled = 0;
    const std::size_t total = n + static_cast<std::size_t>(d);
    for (std::size_t j = 0; pulled < total; ++j) {
        up.push(j < n16 ? in16[j] : 0.0f);
        while (pulled < total && up.ready()) {
            const float y = up.pull();
            if (pulled >= static_cast<std::size_t>(d))
                out[pulled - static_cast<std::size_t>(d)] = y;
            ++pulled;
        }
    }
    return out;
}

std::vector<float> denoiseClipAtRate(const ModelInfo& info, FrameNetwork& network, int rate, const float* in,
                                     std::size_t n, HighBand highBand, const ClipProgress& progress) {
    if (info.sampleRate != kModelRate)
        throw std::invalid_argument("denoiseClipAtRate: the model's rate is " + std::to_string(info.sampleRate) +
                                    " Hz, not " + std::to_string(kModelRate));
    if (rate == kModelRate)
        return denoiseClip(info, network, in, n, progress);
    const auto filter = ResampleFilter::create(rate);  // throws for an unsupported rate
    const std::vector<float> x16 = clipToModelRate(*filter, in, n);
    const std::vector<float> y16 = denoiseClip(info, network, x16.data(), x16.size(), progress);
    if (y16.size() != x16.size())
        return {};  // cancelled
    std::vector<float> y = clipFromModelRate(*filter, y16.data(), y16.size(), n);
    if (highBand == HighBand::PassThrough) {
        const std::vector<float> round = clipFromModelRate(*filter, x16.data(), x16.size(), n);
        for (std::size_t i = 0; i < n; ++i)
            y[i] += in[i] - round[i];
    }
    return y;
}

}  // namespace denoiser
