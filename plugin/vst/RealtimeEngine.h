#pragma once
// One complete, prepared real-time processing state of the plugin: the model,
// one ChannelDenoiser per channel behind a HostRateAdapter, the dry delay
// line and the dry/wet + high-band mixing, for one host rate and channel
// count. No JUCE here, so it can be tested and reused on its own.
//
// Built (allocating, maybe slowly) off the audio thread by Engine::create();
// handed to the audio thread as a whole (see PluginProcessor.h); used there
// only through process() / reset(), which don't allocate, lock or throw;
// destroyed off the audio thread again.
//
// Every engine has a fixed latency L (host samples), and its output is
//
//   (1 - g) * dry + g * wet,      dry = input delayed by exactly L samples
//   wet = denoised + h * (dry - nullRoundTrip)
//
// per sample, with g = mix (0 when bypassed) and h = 1 for "pass the band
// above 8 kHz through" (0 = drop it). nullRoundTrip is the input sent through
// the same resampling chain without a model (the core's own pass-through
// processor), so dry - nullRoundTrip is exactly what the 16 kHz round trip
// removes: the band above ~7.5 kHz. That is the core's HighBand::PassThrough
// (option B) computed outside the adapter, which makes the high band an
// ordinary smooth, automatable switch instead of a re-prepare.
//
// At g == 0 the output is the delayed input copied bit for bit (bypass and
// mix 0), at g == 1 and h == 0 it is the denoiser's output copied bit for
// bit. An engine without a running model (no model, an offline-only model)
// is a pure delay line of L samples (pass-through, exact); one for an
// unsupported host rate has L = 0 and copies its input.

#include "denoiser/denoiser.h"

#include <memory>
#include <string>
#include <vector>

namespace enm156 {

// Mono or stereo (the plugin's bus layouts)
constexpr int kMaxChannels = 2;
// process() takes at most this many samples; the processor splits host
// blocks into chunks of this size, so nothing depends on the host's block size
constexpr int kChunk = 256;

// The STFT shape (hop, n_fft) that decides the latency. The engine without a
// running model reports the latency of this shape, so that loading a model
// of that shape later doesn't change the latency the host compensates.
struct ModelShape {
    int hop = 80;     // config.toml's defaults
    int nFft = 128;
    bool valid() const noexcept { return hop >= 1 && nFft >= hop && nFft <= 65536; }
};

class Engine {
public:
    // Allocates; never throws for bad input (rate, model): the reason is in
    // message() and the engine passes audio through (also if building the
    // denoisers runs out of memory; std::bad_alloc can still escape from the
    // small pass-through buffers). `model` may be null.
    static std::unique_ptr<Engine> create(std::shared_ptr<const denoiser::Model> model, int hostRate,
                                          int channels, ModelShape shape);

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    ~Engine();

    int hostRate() const noexcept { return hostRate_; }
    int channels() const noexcept { return channels_; }
    // Host samples; what the plugin reports with setLatencySamples
    int latencySamples() const noexcept { return latency_; }
    // True if a model is running (else: pass-through)
    bool denoising() const noexcept { return !denoisers_.empty(); }
    // True if the host rate is not 16 kHz and a model runs, i.e. the high-band switch does anything
    bool hasHighBand() const noexcept { return nullRoundTrip_.prepared(); }
    // Samples after a reset before the output depends only on input since the reset
    // (latency + the widest analysis window), used to time a crossfade
    int warmUpSamples() const noexcept { return warmUp_; }
    // Why the engine passes audio through (empty if it denoises)
    const std::string& message() const noexcept { return message_; }
    const std::shared_ptr<const denoiser::Model>& model() const noexcept { return model_; }

    // Back to silence: delay line, resamplers, STFT, network state. Real-time safe.
    void reset() noexcept;

    // channels() planar pointers each way, n <= kChunk. in[c] == out[c] is
    // allowed. wetGain[i] = g and highGain[i] = h for each sample (see top).
    // Real-time safe.
    void process(const float* const* in, float* const* out, int n, const float* wetGain,
                 const float* highGain) noexcept;

private:
    Engine() = default;

    std::shared_ptr<const denoiser::Model> model_;
    std::vector<std::unique_ptr<denoiser::ChannelDenoiser>> denoisers_;
    denoiser::HostRateAdapter adapter_;       // the denoiser
    denoiser::HostRateAdapter nullRoundTrip_; // same chain, no model (only with hasHighBand())
    int hostRate_ = 0;
    int channels_ = 0;
    int latency_ = 0;
    int warmUp_ = 0;
    std::string message_;

    // Dry delay line: one ring of latency_ samples per channel
    std::vector<std::vector<float>> delay_;
    int delayPos_ = 0;

    // Scratch, kChunk samples per channel
    std::vector<std::vector<float>> dry_, wet_, null_;
};

// The latency an engine for this rate and shape without a model reports
// (0 for an unsupported rate)
int passThroughLatency(int hostRate, ModelShape shape) noexcept;

}  // namespace enm156
