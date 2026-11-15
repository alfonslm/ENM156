#include "AraEngine.h"

#include "denoiser/offline_rate.h"
#include "denoiser/resample.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <new>
#include <stdexcept>
#include <thread>
#include <utility>

namespace enm156::ara {

// ============================================================================
// AudioSlot: a "left-right" scheme (two reader counters). A reader registers
// on the current side, then loads the pointer. The writer stores the new
// pointer, waits until the other side has no readers, makes it current, then
// waits until the previous side has no readers. After that no reader can
// hold the old pointer: anyone who registered after the store loads the new
// one, and everyone registered before has left. All operations are
// sequentially consistent.

void AudioSlot::publish(std::shared_ptr<const SourceAudio> next) {
    std::shared_ptr<const SourceAudio> old;
    {
        std::lock_guard<std::mutex> lock(writeMutex_);
        current_.store(next.get());
        const unsigned prev = side_.load();
        const unsigned other = prev ^ 1u;
        const auto drain = [this](unsigned side) {
            while (readers_[side].load() != 0)
                std::this_thread::sleep_for(std::chrono::microseconds(50));
        };
        drain(other);
        side_.store(other);
        drain(prev);
        old = std::exchange(owner_, std::move(next));
    }
    // `old` is freed here (if this was its last owner), outside the lock
}

std::shared_ptr<const SourceAudio> AudioSlot::current() const {
    std::lock_guard<std::mutex> lock(writeMutex_);
    return owner_;
}

AudioSlot::Reader::Reader(const AudioSlot& slot) noexcept
    : slot_(slot), side_(slot.side_.load()), audio_(nullptr) {
    slot_.readers_[side_].fetch_add(1);
    audio_ = slot_.current_.load();
}

AudioSlot::Reader::~Reader() {
    slot_.readers_[side_].fetch_sub(1);
}

// ============================================================================
// Analysis

bool isSupportedRate(double rate) noexcept {
    return denoiser::isSupportedHostRate(rate);
}

namespace {

std::shared_ptr<const denoiser::ResampleFilter> filterFor(double rate) {
    return denoiser::ResampleFilter::create(int(std::lround(rate)));
}

}  // namespace

void SourceAnalysis::setInput(double sampleRate, Planar input) {
    rate_ = sampleRate;
    if (int(input.size()) > kMaxSourceChannels)
        input.resize(size_t(kMaxSourceChannels));
    input_ = std::make_shared<const Planar>(std::move(input));
    hasInput_ = true;
    x16_.clear();
    y16_.clear();
    message_.clear();
    if (isSupportedRate(rate_)) {
        const auto filter = filterFor(rate_);
        for (const auto& channel : *input_)
            x16_.push_back(denoiser::clipToModelRate(*filter, channel.data(), channel.size()));
    }
}

Outcome SourceAnalysis::denoise(const std::shared_ptr<const denoiser::Model>& model, const Progress& progress) {
    y16_.clear();
    message_.clear();
    if (!model) {
        message_ = "No model chosen: the original plays.";
        return Outcome::NoModel;
    }
    if (length() == 0 || channels() == 0) {
        message_ = "Empty audio source.";
        return Outcome::Empty;
    }
    if (x16_.empty()) {
        const double rounded = std::round(rate_);
        const std::string rate = std::fabs(rounded - rate_) < 1e-9 && rounded < 1e9 ? std::to_string(std::llround(rate_))
                                                                   : std::to_string(rate_);
        message_ = "Sample rate " + rate +
                   " Hz is not supported (16 to 384 kHz, e.g. 44.1, 48, 96 kHz): the original plays.";
        return Outcome::UnsupportedRate;
    }
    const denoiser::ModelInfo& info = model->info();
    const std::size_t n16 = x16_[0].size();
    if (n16 < denoiser::minClipSamples(info)) {
        message_ = "Too short for this model (it needs at least " +
                   std::to_string(denoiser::minClipSamples(info)) + " samples at 16 kHz, this source has " +
                   std::to_string(n16) + "): the original plays.";
        return Outcome::TooShort;
    }
    try {
        Planar y16;
        const std::size_t channels = x16_.size();
        for (std::size_t c = 0; c < channels; ++c) {
            auto network = model->makeNetwork();
            const auto frameProgress = [&](std::size_t done, std::size_t total) {
                if (!progress)
                    return true;
                // Report about every 1 % (and every frame is a cancellation point)
                const std::size_t step = std::max<std::size_t>(1, total / 100);
                if (done % step != 0 && done != total)
                    return progress(-1.0f);  // negative: a cancellation check only
                return progress((float(c) + float(done) / float(std::max<std::size_t>(total, 1))) / float(channels));
            };
            std::vector<float> y = denoiser::denoiseClip(info, *network, x16_[c].data(), n16, frameProgress);
            if (y.size() != n16)
                return Outcome::Cancelled;
            y16.push_back(std::move(y));
        }
        y16_ = std::move(y16);
        return Outcome::Denoised;
    } catch (const std::bad_alloc&) {
        message_ = "Out of memory while denoising: the original plays.";
    } catch (const std::exception& e) {
        message_ = std::string("Denoising failed: ") + e.what() + ". The original plays.";
    }
    return Outcome::Failed;
}

std::shared_ptr<const SourceAudio> SourceAnalysis::atSourceRate(denoiser::HighBand highBand) const {
    auto audio = std::make_shared<SourceAudio>();
    audio->sampleRate = rate_;
    audio->channels = channels();
    audio->length = length();
    audio->dry = input_ ? input_ : std::make_shared<const Planar>();
    if (!y16_.empty()) {
        const auto filter = filterFor(rate_);
        const bool passHigh = highBand == denoiser::HighBand::PassThrough && !filter->bypass();
        for (std::size_t c = 0; c < input_->size(); ++c) {
            const std::vector<float>& x = (*input_)[c];
            std::vector<float> y = denoiser::clipFromModelRate(*filter, y16_[c].data(), y16_[c].size(), x.size());
            if (passHigh) {
                const std::vector<float> round =
                    denoiser::clipFromModelRate(*filter, x16_[c].data(), x16_[c].size(), x.size());
                for (std::size_t i = 0; i < x.size(); ++i)
                    y[i] += x[i] - round[i];
            }
            audio->wet.push_back(std::move(y));
        }
    }
    return audio;
}

std::shared_ptr<const SourceAudio> SourceAnalysis::atRate(int rate) const {
    if (x16_.empty() || !isSupportedRate(rate))
        return nullptr;
    const auto filter = filterFor(rate);
    auto audio = std::make_shared<SourceAudio>();
    audio->sampleRate = rate;
    audio->channels = channels();
    // The source's duration at the new rate
    const std::size_t n = std::size_t(std::llround(double(length()) * rate / rate_));
    audio->length = std::int64_t(n);
    Planar dry;
    for (std::size_t c = 0; c < x16_.size(); ++c) {
        dry.push_back(denoiser::clipFromModelRate(*filter, x16_[c].data(), x16_[c].size(), n));
        if (!y16_.empty())
            audio->wet.push_back(denoiser::clipFromModelRate(*filter, y16_[c].data(), y16_[c].size(), n));
    }
    audio->dry = std::make_shared<const Planar>(std::move(dry));
    return audio;
}

// ============================================================================
// Rendering

RenderResult renderPlan(const RenderPlan& plan, std::int64_t position, int n, float* const* out, int outChannels,
                        const float* wetGain) noexcept {
    RenderResult result;
    for (int c = 0; c < outChannels; ++c)
        std::fill(out[c], out[c] + n, 0.0f);
    const std::int64_t blockEnd = position + n;
    for (const RegionEntry& region : plan.regions) {
        std::int64_t from = std::max(position, region.start);
        std::int64_t to = std::min(blockEnd, region.end);
        if (from >= to || region.slot == nullptr)
            continue;
        const AudioSlot::Reader reader(*region.slot);
        const SourceAudio* audio = reader.get();
        if (audio == nullptr || audio->channels < 1 || !audio->dry || int(audio->dry->size()) < audio->channels) {
            ++result.regionsSilent;
            continue;
        }
        // Clip to the stored audio
        from = std::max(from, -region.offset);
        to = std::min(to, audio->length - region.offset);
        if (from >= to) {
            ++result.regionsRendered;
            continue;
        }
        const bool hasWet = int(audio->wet.size()) == audio->channels;
        if (hasWet)
            ++result.regionsRendered;
        else
            ++result.regionsOriginal;
        const int srcChannels = audio->channels;
        for (int c = 0; c < outChannels; ++c) {
            float* o = out[c] + (from - position);
            const int count = int(to - from);
            const float* g = wetGain + (from - position);
            const std::size_t start = std::size_t(from + region.offset);
            if (srcChannels >= 2 && outChannels == 1) {
                // Stereo source on a mono output: the mean of both channels
                const float* d0 = (*audio->dry)[0].data() + start;
                const float* d1 = (*audio->dry)[1].data() + start;
                if (hasWet) {
                    const float* w0 = audio->wet[0].data() + start;
                    const float* w1 = audio->wet[1].data() + start;
                    for (int i = 0; i < count; ++i) {
                        const float d = 0.5f * (d0[i] + d1[i]), w = 0.5f * (w0[i] + w1[i]);
                        o[i] += (1.0f - g[i]) * d + g[i] * w;
                    }
                } else {
                    for (int i = 0; i < count; ++i)
                        o[i] += 0.5f * (d0[i] + d1[i]);
                }
                continue;
            }
            // Mono source: on every output channel; stereo: channel by channel
            const std::size_t s = std::size_t(std::min(c, srcChannels - 1));
            const float* d = (*audio->dry)[s].data() + start;
            if (hasWet) {
                const float* w = audio->wet[s].data() + start;
                for (int i = 0; i < count; ++i)
                    o[i] += (1.0f - g[i]) * d[i] + g[i] * w[i];
            } else {
                for (int i = 0; i < count; ++i)
                    o[i] += d[i];
            }
        }
    }
    return result;
}

bool planIsWorking(const RenderPlan& plan, std::int64_t position, int n) noexcept {
    for (const RegionEntry& region : plan.regions)
        if (region.slot != nullptr && region.start < position + n && region.end > position &&
            region.slot->working.load())
            return true;
    return false;
}

}  // namespace enm156::ara
