#include "denoiser/host_rate.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace denoiser {

namespace {

void checkSizes(int hop, int processorDelay) {
    if (hop < 1)
        throw std::invalid_argument("hop must be >= 1, got " + std::to_string(hop));
    if (processorDelay < 0)
        throw std::invalid_argument("processor delay must be >= 0, got " + std::to_string(processorDelay));
}

// Delay of the whole chain before rounding down to host samples, in grid
// steps (one host sample = L steps, one 16 kHz sample = M steps):
// down filter centre + up filter centre + (processor + regrouping) * M
std::int64_t chainDelayGrid(const RateRatio& r, bool bypass, int hop, int processorDelay) {
    const std::int64_t q = bypass ? 0 : kResampleHalfLength;
    return std::int64_t(r.down) * (2 * q + processorDelay + hop - 1);
}

}  // namespace

int hostLatencySamples(int hostRate, int hop, int processorDelay) {
    checkSizes(hop, processorDelay);
    const RateRatio r = hostRateRatio(hostRate);  // throws if unsupported
    return int(chainDelayGrid(r, hostRate == kModelRate, hop, processorDelay) / r.up);
}

void HostRateAdapter::prepare(const HostRateConfig& config) {
    checkSizes(config.hop, config.processorDelay);
    if (config.channels < 1)
        throw std::invalid_argument("channels must be >= 1");
    auto filter = ResampleFilter::create(config.hostRate);  // throws if unsupported

    const RateRatio r{filter->up(), filter->down()};
    const std::int64_t delay = chainDelayGrid(r, filter->bypass(), config.hop, config.processorDelay);
    // The Upsampler samples `offset` grid steps late, so that the total delay
    // (delay - offset) is a whole number of host samples. offset < L keeps
    // every 16 kHz sample it needs available by the host sample it is due at.
    const int offset = int(delay % r.up);
    const int latency = hostLatencySamples(config.hostRate, config.hop, config.processorDelay);

    std::vector<HopProcessor*> keep;
    for (const Channel& ch : channels_)
        keep.push_back(ch.processor);

    std::vector<Channel> channels(config.channels);
    const bool passHigh = config.highBand == HighBand::PassThrough && !filter->bypass();
    for (int c = 0; c < config.channels; ++c) {
        Channel& ch = channels[c];
        ch.down.prepare(filter);
        ch.upWet.prepare(filter, offset);
        if (passHigh) {
            ch.upDry.prepare(filter, offset);
            ch.hostDelay.assign(std::size_t(std::max(latency, 1)), 0.0f);
        }
        ch.history16.assign(std::size_t(config.processorDelay) + config.hop, 0.0f);
        ch.hopIn.assign(config.hop, 0.0f);
        ch.hopOut.assign(config.hop, 0.0f);
        ch.fifo.assign(2 * std::size_t(config.hop), 0.0f);
        ch.processor = c < int(keep.size()) ? keep[c] : nullptr;
    }

    config_ = config;
    filter_ = std::move(filter);
    latency_ = latency;
    passHigh_ = passHigh;
    channels_ = std::move(channels);
    reset();
}

void HostRateAdapter::setProcessor(int channel, HopProcessor* processor) noexcept {
    if (channel >= 0 && channel < int(channels_.size()))
        channels_[channel].processor = processor;
}

void HostRateAdapter::reset() noexcept {
    for (int c = 0; c < int(channels_.size()); ++c)
        resetChannel(c);
}

void HostRateAdapter::resetChannel(int channel) noexcept {
    if (channel < 0 || channel >= int(channels_.size()))
        return;
    Channel& ch = channels_[channel];
    ch.down.reset();
    ch.upWet.reset();
    if (passHigh_)
        ch.upDry.reset();
    std::fill(ch.history16.begin(), ch.history16.end(), 0.0f);
    ch.historyWrite = 0;
    std::fill(ch.hopIn.begin(), ch.hopIn.end(), 0.0f);
    std::fill(ch.hopOut.begin(), ch.hopOut.end(), 0.0f);
    ch.hopFill = 0;
    // hop - 1 samples of silence ahead of the first hop's output: the
    // regrouping delay, which lets every 16 kHz output sample leave as soon as
    // the 16 kHz input sample with the same index has arrived
    std::fill(ch.fifo.begin(), ch.fifo.end(), 0.0f);
    ch.fifoRead = 0;
    ch.fifoCount = config_.hop - 1;
    std::fill(ch.hostDelay.begin(), ch.hostDelay.end(), 0.0f);
    ch.hostDelayPos = 0;
    if (ch.processor)
        ch.processor->reset();
}

// One new sample y[n] at 16 kHz: regroup, process, and pass one sample on
void HostRateAdapter::stepModelRate(Channel& ch, float y) noexcept {
    const int hop = config_.hop;
    const int size = int(ch.history16.size());  // processorDelay + hop
    ch.history16[ch.historyWrite] = y;
    if (++ch.historyWrite == size)
        ch.historyWrite = 0;
    // y[n - (processorDelay + hop - 1)]: the oldest sample kept, which is the
    // dry signal with exactly the delay of the processor path
    const float dry = ch.history16[ch.historyWrite];

    ch.hopIn[ch.hopFill] = y;
    if (++ch.hopFill == hop) {
        ch.hopFill = 0;
        if (ch.processor) {
            ch.processor->processHop(ch.hopIn.data(), ch.hopOut.data(), hop);
        } else {
            // Built-in pass-through: out[i] = y[n - (hop - 1) + i - processorDelay]
            for (int i = 0; i < hop; ++i) {
                int slot = ch.historyWrite - 1 - (config_.processorDelay + hop - 1 - i);
                if (slot < 0)
                    slot += size;
                ch.hopOut[i] = ch.history16[slot];
            }
        }
        const int cap = int(ch.fifo.size());
        int w = ch.fifoRead + ch.fifoCount;
        for (int i = 0; i < hop; ++i, ++w)
            ch.fifo[w % cap] = ch.hopOut[i];
        ch.fifoCount += hop;
    }
    const float wet = ch.fifo[ch.fifoRead];
    if (++ch.fifoRead == int(ch.fifo.size()))
        ch.fifoRead = 0;
    --ch.fifoCount;

    ch.upWet.push(wet);
    if (passHigh_)
        ch.upDry.push(dry);
}

void HostRateAdapter::processChannel(int channel, const float* in, float* out, int numSamples) noexcept {
    if (channel < 0 || channel >= int(channels_.size()))
        return;
    Channel& ch = channels_[channel];
    for (int i = 0; i < numSamples; ++i) {
        const float x = in[i];
        float y;
        if (ch.down.push(x, &y))
            stepModelRate(ch, y);
        const float wet = ch.upWet.pull();
        if (passHigh_) {
            const float dry = ch.upDry.pull();
            const float delayed = ch.hostDelay[ch.hostDelayPos];
            ch.hostDelay[ch.hostDelayPos] = x;
            if (++ch.hostDelayPos == int(ch.hostDelay.size()))
                ch.hostDelayPos = 0;
            // The input minus its own band-limited round trip is the band the
            // model cannot see; add the processed low band to it
            out[i] = delayed + (wet - dry);
        } else {
            out[i] = wet;
        }
    }
}

void HostRateAdapter::process(const float* const* in, float* const* out, int numSamples) noexcept {
    for (int c = 0; c < int(channels_.size()); ++c)
        processChannel(c, in[c], out[c], numSamples);
}

std::int64_t HostRateAdapter::underruns() const noexcept {
    std::int64_t n = 0;
    for (const Channel& ch : channels_)
        n += ch.upWet.underruns() + (passHigh_ ? ch.upDry.underruns() : 0);
    return n;
}

void processClipAligned(HostRateAdapter& adapter, int channel, const float* in, float* out,
                        std::int64_t numSamples) noexcept {
    adapter.resetChannel(channel);
    const std::int64_t latency = adapter.latencySamples();
    const std::int64_t total = numSamples + latency;
    constexpr int kChunk = 256;
    float inBuf[kChunk];
    float outBuf[kChunk];
    for (std::int64_t pos = 0; pos < total; pos += kChunk) {
        const int n = int(std::min<std::int64_t>(kChunk, total - pos));
        for (int i = 0; i < n; ++i)
            inBuf[i] = pos + i < numSamples ? in[pos + i] : 0.0f;
        adapter.processChannel(channel, inBuf, outBuf, n);
        for (int i = 0; i < n; ++i) {
            const std::int64_t at = pos + i - latency;
            if (at >= 0)
                out[at] = outBuf[i];
        }
    }
}

}  // namespace denoiser
