#pragma once
// Runs a 16 kHz, hop-by-hop processor (the denoiser) inside a host that
// calls with blocks of any size at its own sample rate, one independent
// processor per channel.
//
// Per channel, per host sample:
//   host rate --Downsampler--> 16 kHz --groups of `hop`--> processor
//   --> 16 kHz --Upsampler--> host rate
//
// The whole chain is a fixed delay of hostLatencySamples(...) host samples
// (for a processor that is itself a pure delay): it does not depend on the
// block sizes or on how calls are split, because the chain runs sample by
// sample on a fixed schedule. Output for the same input is identical
// whatever the block sizes.
//
//   denoiser::HostRateAdapter adapter;
//   // prepareToPlay (allocates; throws std::invalid_argument for a bad rate):
//   adapter.prepare({hostRate, channels, hop, nFft - hop, HighBand::Drop});
//   for (int c = 0; c < channels; ++c) adapter.setProcessor(c, &denoisers[c]);
//   setLatencySamples(adapter.latencySamples());
//   // processBlock (no allocation, locks or exceptions):
//   adapter.process(inputs, outputs, numSamples);   // in == out is fine
//   // reset: clears the adapter and calls reset() on every processor
//   adapter.reset();

#include "denoiser/resample.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace denoiser {

// What happens to the band above 8 kHz, which the model never sees
enum class HighBand {
    // Option A: output only the 16 kHz result, resampled back. Nothing above
    // ~7.5 kHz (stopband from 8 kHz). Speech sounds duller than the dry signal.
    Drop,
    // Option B: also add back what the 16 kHz round trip removes, i.e. the
    // input delayed by the latency minus its own (unprocessed) round trip.
    // The band above 8 kHz passes through unchanged and in time (noise there
    // stays). With an identity processor the output is exactly the delayed
    // input. Costs one more Upsampler per channel. Ignored at 16 kHz.
    PassThrough,
};

// One channel's 16 kHz processor, called with exactly `hop` samples in and
// `hop` out. Called on the audio thread: processHop() and reset() must not
// allocate, lock or throw. `in` and `out` never overlap.
class HopProcessor {
public:
    virtual ~HopProcessor() = default;
    virtual void processHop(const float* in, float* out, int hop) noexcept = 0;
    // Called by HostRateAdapter::reset() / resetChannel()
    virtual void reset() noexcept {}
};

struct HostRateConfig {
    int hostRate = 48000;     // Hz; see isSupportedHostRate()
    int channels = 2;         // one processor per channel
    int hop = 80;             // 16 kHz samples per processor call (stft.hop)
    int processorDelay = 48;  // the processor's own delay in 16 kHz samples
                              // (StreamingDenoiser: n_fft - hop)
    HighBand highBand = HighBand::Drop;
};

// THE latency to report to the host, in host samples, for a processor that
// delays by `processorDelay` samples at 16 kHz and is called with `hop`:
//
//   16 kHz host:  processorDelay + hop - 1                    (no resampling)
//   otherwise:    floor((2 Q + processorDelay + hop - 1) * hostRate / 16000)
//
// where Q = kResampleHalfLength is the delay of each resampler (in 16 kHz
// samples) and hop - 1 is the regrouping of samples into hops. The fraction
// floor() drops is absorbed by the Upsampler's offset, so the result is
// exact. Throws std::invalid_argument for an unsupported rate, hop < 1 or
// processorDelay < 0.
int hostLatencySamples(int hostRate, int hop, int processorDelay);

class HostRateAdapter {
public:
    HostRateAdapter() = default;
    HostRateAdapter(const HostRateAdapter&) = delete;
    HostRateAdapter& operator=(const HostRateAdapter&) = delete;

    // Allocates everything and resets. Not real-time safe. Throws
    // std::invalid_argument for an unsupported rate or bad sizes. Keeps
    // processors already set for channels that still exist.
    void prepare(const HostRateConfig& config);
    const HostRateConfig& config() const noexcept { return config_; }
    bool prepared() const noexcept { return !channels_.empty(); }

    // == hostLatencySamples(config.hostRate, config.hop, config.processorDelay)
    int latencySamples() const noexcept { return latency_; }

    // The processor for one channel (not owned). nullptr (the default) is a
    // built-in pass-through that delays by processorDelay, so the latency is
    // the same with and without a model. Swapping between process() calls is
    // allowed; the new processor takes over at the next hop boundary. Call
    // from the audio thread (or while it is stopped): no synchronisation here.
    void setProcessor(int channel, HopProcessor* processor) noexcept;

    // Back to silence, as after prepare(); also calls reset() on the channel's
    // processor(s). Real-time safe.
    void reset() noexcept;
    void resetChannel(int channel) noexcept;

    // Planar audio, config().channels pointers each way. in[c] == out[c] is
    // allowed. numSamples may be anything >= 0, varying from call to call.
    void process(const float* const* in, float* const* out, int numSamples) noexcept;
    void processChannel(int channel, const float* in, float* out, int numSamples) noexcept;

    // Scheduling errors since reset (always 0 unless there is a bug)
    std::int64_t underruns() const noexcept;

private:
    struct Channel {
        Downsampler down;
        Upsampler upWet;
        Upsampler upDry;                // HighBand::PassThrough only
        HopProcessor* processor = nullptr;
        std::vector<float> history16;   // last processorDelay + hop samples at 16 kHz
        int historyWrite = 0;
        std::vector<float> hopIn, hopOut;
        int hopFill = 0;
        std::vector<float> fifo;        // processor output waiting for the Upsampler
        int fifoRead = 0, fifoCount = 0;
        std::vector<float> hostDelay;   // HighBand::PassThrough: input delayed by the latency
        int hostDelayPos = 0;
    };

    void stepModelRate(Channel& ch, float y) noexcept;

    HostRateConfig config_;
    std::shared_ptr<const ResampleFilter> filter_;
    int latency_ = 0;
    bool passHigh_ = false;
    std::vector<Channel> channels_;
};

// Offline (e.g. ARA analysis): one channel of a whole clip, output aligned
// with the input (the latency is removed) and the same length. Resets the
// channel, feeds the clip followed by latencySamples() zeros and drops the
// first latencySamples() outputs, so the result equals a real-time render
// shifted back by the reported latency. Does not allocate. in == out is not
// allowed.
void processClipAligned(HostRateAdapter& adapter, int channel, const float* in, float* out,
                        std::int64_t numSamples) noexcept;

}  // namespace denoiser
