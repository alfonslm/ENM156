#include "RealtimeEngine.h"

#include "denoiser/resample.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace enm156 {

namespace {

// Exact comparison without -Wfloat-equal noise
inline bool same(float a, float b) noexcept { return std::equal_to<float>()(a, b); }

}  // namespace

int passThroughLatency(int hostRate, ModelShape shape) noexcept {
    if (!denoiser::isSupportedHostRate(hostRate))
        return 0;
    if (!shape.valid())
        shape = ModelShape{};
    try {
        return denoiser::hostLatencySamples(hostRate, shape.hop, shape.nFft - shape.hop);
    } catch (const std::exception&) {
        return 0;
    }
}

std::unique_ptr<Engine> Engine::create(std::shared_ptr<const denoiser::Model> model, int hostRate, int channels,
                                       ModelShape shape) {
    std::unique_ptr<Engine> e(new Engine());
    e->hostRate_ = hostRate;
    e->channels_ = std::clamp(channels, 1, kMaxChannels);

    const bool rateOk = denoiser::isSupportedHostRate(hostRate);
    if (!rateOk) {
        e->message_ = "The host sample rate (" + std::to_string(hostRate) +
                      " Hz) is not supported: use 16 kHz or 22.05 to 384 kHz (44.1, 48, 88.2, 96 kHz, ...). "
                      "Audio passes through with no latency.";
    } else if (model && !model->info().realtime) {
        e->message_ = "This model uses centred (non-causal) frames, so it can't run in real time; audio passes "
                      "through. It can be used in ARA mode.";
    } else if (model) {
        try {
            for (int c = 0; c < e->channels_; ++c)
                e->denoisers_.push_back(std::make_unique<denoiser::ChannelDenoiser>(model));
            const denoiser::HostRateConfig config =
                e->denoisers_[0]->hostRateConfig(hostRate, e->channels_, denoiser::HighBand::Drop);
            e->adapter_.prepare(config);
            for (int c = 0; c < e->channels_; ++c)
                e->adapter_.setProcessor(c, e->denoisers_[size_t(c)].get());
            if (hostRate != denoiser::kModelRate)
                e->nullRoundTrip_.prepare(config);  // no processors: the built-in pass-through
            e->latency_ = e->adapter_.latencySamples();
            e->model_ = std::move(model);
        } catch (const std::exception& error) {
            // Not expected for a model that loaded (the rate was checked above), or out of memory
            e->denoisers_.clear();
            e->message_ = std::string("The model can't run at this rate: ") + error.what() + ". Audio passes through.";
        }
    }

    if (!e->denoising())
        e->latency_ = rateOk ? passThroughLatency(hostRate, shape) : 0;

    // Crossfade timing: the output depends only on input since the reset
    // once the latency has passed and the analysis window is full
    int widest = 0;
    if (e->model_) {
        const denoiser::ModelInfo& info = e->model_->info();
        widest = int(std::ceil(double(info.widestWindow()) * hostRate / info.sampleRate));
    }
    e->warmUp_ = e->latency_ + widest;

    e->delay_.assign(size_t(e->channels_), std::vector<float>(size_t(e->latency_), 0.0f));
    e->dry_.assign(size_t(e->channels_), std::vector<float>(kChunk, 0.0f));
    if (e->denoising()) {
        e->wet_.assign(size_t(e->channels_), std::vector<float>(kChunk, 0.0f));
        if (e->hasHighBand())
            e->null_.assign(size_t(e->channels_), std::vector<float>(kChunk, 0.0f));
    }
    return e;
}

Engine::~Engine() = default;

void Engine::reset() noexcept {
    for (auto& line : delay_)
        std::fill(line.begin(), line.end(), 0.0f);
    delayPos_ = 0;
    if (adapter_.prepared())
        adapter_.reset();  // also resets the ChannelDenoisers
    if (nullRoundTrip_.prepared())
        nullRoundTrip_.reset();
}

void Engine::process(const float* const* in, float* const* out, int n, const float* wetGain,
                     const float* highGain) noexcept {
    n = std::clamp(n, 0, kChunk);
    if (n == 0)
        return;

    // 1. Dry: the input delayed by exactly latency_ samples. Reads all of the
    //    input before anything is written (in may equal out).
    float* dryPtr[kMaxChannels] = {};
    const int length = latency_;
    for (int c = 0; c < channels_; ++c) {
        float* dry = dry_[size_t(c)].data();
        const float* x = in[c];
        if (length == 0) {
            std::copy(x, x + n, dry);
        } else {
            float* ring = delay_[size_t(c)].data();
            int pos = delayPos_;
            for (int i = 0; i < n; ++i) {
                dry[i] = ring[pos];
                ring[pos] = x[i];
                if (++pos == length)
                    pos = 0;
            }
        }
        dryPtr[c] = dry;
    }
    if (length > 0)
        delayPos_ = int((delayPos_ + n) % length);

    if (!denoising()) {
        for (int c = 0; c < channels_; ++c)
            std::copy(dryPtr[c], dryPtr[c] + n, out[c]);
        return;
    }

    // 2. Wet: the denoiser (and the high band, if any)
    float* wetPtr[kMaxChannels] = {};
    for (int c = 0; c < channels_; ++c)
        wetPtr[c] = wet_[size_t(c)].data();
    adapter_.process(in, wetPtr, n);

    if (hasHighBand()) {
        float* nullPtr[kMaxChannels] = {};
        for (int c = 0; c < channels_; ++c)
            nullPtr[c] = null_[size_t(c)].data();
        nullRoundTrip_.process(in, nullPtr, n);  // always, so its state stays continuous
        for (int c = 0; c < channels_; ++c)
            for (int i = 0; i < n; ++i)
                if (!same(highGain[i], 0.0f))
                    wetPtr[c][i] += highGain[i] * (dryPtr[c][i] - nullPtr[c][i]);
    }

    // 3. Mix (exact copies at g = 0 and g = 1)
    for (int c = 0; c < channels_; ++c) {
        float* y = out[c];
        const float* dry = dryPtr[c];
        const float* wet = wetPtr[c];
        for (int i = 0; i < n; ++i) {
            const float g = wetGain[i];
            y[i] = same(g, 1.0f) ? wet[i] : same(g, 0.0f) ? dry[i] : (1.0f - g) * dry[i] + g * wet[i];
        }
    }
}

}  // namespace enm156
