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

std::shared_ptr<PlayCursor> AudioSlot::addCursor() {
    auto cursor = std::make_shared<PlayCursor>();
    std::lock_guard<std::mutex> lock(cursorMutex_);
    cursors_.erase(std::remove_if(cursors_.begin(), cursors_.end(),
                                  [](const std::weak_ptr<PlayCursor>& c) { return c.expired(); }),
                   cursors_.end());
    cursors_.push_back(cursor);
    return cursor;
}

void AudioSlot::cursorStates(std::vector<CursorState>& out) const {
    out.clear();
    std::lock_guard<std::mutex> lock(cursorMutex_);
    for (const auto& weak : cursors_)
        if (const auto cursor = weak.lock()) {
            CursorState state;
            state.position = cursor->position.load();
            state.lead = cursor->lead.load();
            if (state.position >= 0)
                out.push_back(state);
        }
}

// ============================================================================
// SourceStream: a pool of blocks shared by the streaming thread (writer) and
// audio threads (readers), per block entry an atomic tag (which block it
// holds) and a reader count. A reader registers on the entry, then checks
// the tag; the writer clears the tag, then waits until no reader is
// registered, fills the entry, then sets the tag. All sequentially
// consistent: a reader either sees the cleared tag and skips the entry, or
// is registered before the writer looks and is waited for. The writer waits
// at most for one copy of up to 256 samples per reader.

namespace {

std::int64_t floorDiv(std::int64_t a, std::int64_t b) {
    return a >= 0 ? a / b : -((-a + b - 1) / b);
}

std::int64_t ceilDiv(std::int64_t a, std::int64_t b) {
    return -floorDiv(-a, b);
}

std::int64_t rateRound(double rate) {
    return std::int64_t(std::llround(rate));
}

}  // namespace

SourceStream::SourceStream(double sourceRate, int channels, std::int64_t sourceLength, int playbackRate)
    : sourceRate_(sourceRate),
      channels_(std::clamp(channels, 1, kMaxSourceChannels)),
      sourceLength_(std::max<std::int64_t>(sourceLength, 0)),
      playbackRate_(playbackRate) {
    if (rateRound(sourceRate) == playbackRate) {
        length_ = sourceLength_;
    } else {
        if (!isSupportedRate(sourceRate) || !isSupportedRate(playbackRate))
            throw std::invalid_argument("streaming: unsupported sample rate");
        in_ = denoiser::ResampleFilter::create(int(rateRound(sourceRate)));
        out_ = denoiser::ResampleFilter::create(playbackRate);
        length_ = std::int64_t(std::llround(double(sourceLength_) * playbackRate / sourceRate));
    }
    blocks_ = (length_ + kBlockFrames - 1) / kBlockFrames;
    entries_ = std::make_unique<Entry[]>(size_t(kPoolBlocks));
    pool_.assign(size_t(kPoolBlocks) * size_t(channels_) * size_t(kBlockFrames), 0.0f);
    loaded_.assign(size_t(kPoolBlocks), -1);
    loadedAt_.assign(size_t(kPoolBlocks), 0);
    scratch_.assign(size_t(channels_), std::vector<float>(size_t(kBlockFrames)));
}

float* SourceStream::data(int entry, int channel) noexcept {
    return pool_.data() + (size_t(entry) * size_t(channels_) + size_t(channel)) * size_t(kBlockFrames);
}

const float* SourceStream::data(int entry, int channel) const noexcept {
    return pool_.data() + (size_t(entry) * size_t(channels_) + size_t(channel)) * size_t(kBlockFrames);
}

int SourceStream::mixInto(std::int64_t position, int n, float* const* out, int outChannels) const noexcept {
    int missing = 0;
    std::int64_t pos = std::max<std::int64_t>(position, 0);
    const std::int64_t end = std::min(position + n, length_);
    missing += int(pos - position);
    if (!valid_.load())
        return missing + int(std::max<std::int64_t>(end - pos, 0));
    while (pos < end) {
        const std::int64_t block = pos / kBlockFrames;
        const std::int64_t blockEnd = std::min(end, (block + 1) * kBlockFrames);
        const int count = int(blockEnd - pos);
        const int at = int(pos - position);           // in out
        const auto in = size_t(pos - block * kBlockFrames);      // in the block
        bool found = false;
        for (int e = 0; e < kPoolBlocks && !found; ++e) {
            Entry& entry = entries_[size_t(e)];
            if (entry.block.load(std::memory_order_relaxed) != block)
                continue;
            entry.readers.fetch_add(1);
            if (entry.block.load() == block) {
                found = true;
                if (channels_ >= 2 && outChannels == 1) {
                    const float* d0 = data(e, 0) + in;
                    const float* d1 = data(e, 1) + in;
                    float* o = out[0] + at;
                    for (int i = 0; i < count; ++i)
                        o[i] += 0.5f * (d0[i] + d1[i]);
                } else {
                    for (int c = 0; c < outChannels; ++c) {
                        const float* d = data(e, std::min(c, channels_ - 1)) + in;
                        float* o = out[c] + at;
                        for (int i = 0; i < count; ++i)
                            o[i] += d[i];
                    }
                }
            }
            entry.readers.fetch_sub(1);
        }
        if (!found)
            missing += count;
        pos = blockEnd;
    }
    return missing;
}

bool SourceStream::ready(std::int64_t position, std::int64_t n) const noexcept {
    const std::int64_t from = std::max<std::int64_t>(position, 0);
    const std::int64_t to = std::min(position + n, length_);
    if (!valid_.load())
        return from >= to;
    for (std::int64_t block = from / kBlockFrames; from < to && block * kBlockFrames < to; ++block) {
        bool found = false;
        for (int e = 0; e < kPoolBlocks && !found; ++e)
            found = entries_[size_t(e)].block.load() == block;
        if (!found)
            return false;
    }
    return true;
}

std::int64_t SourceStream::urgency(std::int64_t block, const std::vector<CursorState>& cursors) const noexcept {
    // Frames until the earliest cursor plays this block; INT64_MAX: none wants it
    std::int64_t best = INT64_MAX;
    const std::int64_t start = block * kBlockFrames, end = start + kBlockFrames;
    for (const CursorState& c : cursors) {
        if (c.position < 0 || end <= c.position)
            continue;  // behind the cursor (or idle)
        const std::int64_t until = std::max<std::int64_t>(c.lead, 0) + std::max<std::int64_t>(0, start - c.position);
        if (until < kReadAheadFrames)
            best = std::min(best, until);
    }
    return best;
}

SourceStream::Fill SourceStream::fillOne(const std::vector<CursorState>& cursors, const SourceReader& read) {
    if (!valid_.load() || (failing_.load() && std::chrono::steady_clock::now() < retryAt_))
        return Fill::Idle;
    // The most urgent block that is wanted and not in the pool
    std::int64_t want = -1, wantUrgency = INT64_MAX;
    for (const CursorState& c : cursors) {
        if (c.position < 0 || c.position >= length_)
            continue;
        for (std::int64_t block = c.position / kBlockFrames; block < blocks_; ++block) {
            const std::int64_t u = urgency(block, cursors);
            if (u == INT64_MAX)
                break;
            if (u < wantUrgency && std::find(loaded_.begin(), loaded_.end(), block) == loaded_.end()) {
                want = block;
                wantUrgency = u;
            }
        }
    }
    if (want < 0)
        return Fill::Idle;
    // The entry to replace: a free one, else the oldest one no cursor wants,
    // else the one wanted latest (only if later than this block)
    // Ranked by (free, then not wanted, then wanted), then by urgency, then oldest
    int victim = -1;
    int victimTier = -1;
    std::int64_t victimUrgency = 0;
    std::uint64_t victimAge = 0;
    for (int e = 0; e < kPoolBlocks; ++e) {
        const auto i = size_t(e);
        const std::int64_t u = loaded_[i] < 0 ? INT64_MAX : urgency(loaded_[i], cursors);
        const int tier = loaded_[i] < 0 ? 2 : u == INT64_MAX ? 1 : 0;
        const bool better = tier != victimTier ? tier > victimTier
                            : u != victimUrgency ? u > victimUrgency
                                                 : loadedAt_[i] < victimAge;
        if (victim < 0 || better) {
            victim = e;
            victimTier = tier;
            victimUrgency = u;
            victimAge = loadedAt_[i];
        }
    }
    if (victim < 0 || (victimTier == 0 && victimUrgency <= wantUrgency))
        return Fill::Idle;  // the pool is full of blocks needed sooner

    const ReadStatus status = compute(want, read);
    if (status != ReadStatus::Ok) {
        failing_ = true;
        retryAt_ = std::chrono::steady_clock::now() + kRetryDelay;
        return status == ReadStatus::NoAccess ? Fill::NoAccess : Fill::Failed;
    }
    failing_ = false;
    Entry& entry = entries_[size_t(victim)];
    entry.block.store(-1);
    while (entry.readers.load() != 0)
        std::this_thread::yield();
    for (int c = 0; c < channels_; ++c)
        std::copy(scratch_[size_t(c)].begin(), scratch_[size_t(c)].end(), data(victim, c));
    entry.block.store(want);
    loaded_[size_t(victim)] = want;
    loadedAt_[size_t(victim)] = ++loads_;
    return Fill::Loaded;
}

ReadStatus SourceStream::compute(std::int64_t block, const SourceReader& read) {
    const std::int64_t m0 = block * kBlockFrames;
    const std::int64_t m1 = std::min(length_, m0 + kBlockFrames);
    for (auto& channel : scratch_)
        std::fill(channel.begin(), channel.end(), 0.0f);
    if (m1 <= m0)
        return ReadStatus::Ok;
    if (!in_) {
        // The source's own rate: the samples as read
        float* ptrs[kMaxSourceChannels] = {};
        for (int c = 0; c < channels_; ++c)
            ptrs[c] = scratch_[size_t(c)].data();
        const ReadStatus s = read(m0, m1 - m0, ptrs);
        if (s == ReadStatus::Ok)
            framesRead_ += m1 - m0;
        return s;
    }
    // Another rate: source -> 16 kHz -> playback rate over a window with
    // margins, its ends on the filters' periods, so that every sample of the
    // block is computed from the same inputs, with the same filter phases, as
    // by clipToModelRate + clipFromModelRate over the whole source.
    // 16 kHz sample j is at source time j * Mi / Li and output time j * Mo / Lo.
    const denoiser::RateRatio ri = denoiser::hostRateRatio(in_->hostRate());
    const denoiser::RateRatio ro = denoiser::hostRateRatio(out_->hostRate());
    const std::int64_t Li = ri.up, Mi = ri.down, Lo = ro.up, Mo = ro.down;
    const std::int64_t margin = denoiser::kResampleHalfLength + 8;
    const std::int64_t n16 = std::int64_t(denoiser::modelRateLength(in_->hostRate(), size_t(sourceLength_)));
    // 16 kHz range [j0, j1) for output [mStart, m1), j0 on a multiple of Lo
    std::int64_t j0 = std::max<std::int64_t>(0, floorDiv(m0 * Lo, Mo) - margin);
    j0 = j0 / Lo * Lo;
    const std::int64_t mStart = j0 * Mo / Lo;
    const std::int64_t j1 = std::min(n16, ceilDiv(m1 * Lo, Mo) + margin);
    if (j1 <= j0)
        return ReadStatus::Ok;
    // Source range [s0, s1) for that, s0 on a multiple of Mi
    std::int64_t s0 = std::max<std::int64_t>(0, floorDiv((j0 - margin) * Mi, Li));
    s0 = s0 / Mi * Mi;
    const std::int64_t jStart = s0 * Li / Mi;
    const std::int64_t s1 = std::min(sourceLength_, ceilDiv((j1 + margin) * Mi, Li));
    if (s1 <= s0)
        return ReadStatus::Ok;
    sourceScratch_.resize(size_t(channels_));
    float* ptrs[kMaxSourceChannels] = {};
    for (int c = 0; c < channels_; ++c) {
        sourceScratch_[size_t(c)].assign(size_t(s1 - s0), 0.0f);
        ptrs[c] = sourceScratch_[size_t(c)].data();
    }
    const ReadStatus s = read(s0, s1 - s0, ptrs);
    if (s != ReadStatus::Ok)
        return s;
    framesRead_ += s1 - s0;
    for (int c = 0; c < channels_; ++c) {
        const std::vector<float> x16 =
            denoiser::clipToModelRate(*in_, sourceScratch_[size_t(c)].data(), size_t(s1 - s0));
        const std::int64_t first = j0 - jStart;
        const std::int64_t count = std::min<std::int64_t>(j1 - j0, std::int64_t(x16.size()) - first);
        if (count <= 0)
            continue;
        const std::vector<float> y =
            denoiser::clipFromModelRate(*out_, x16.data() + first, size_t(count), size_t(m1 - mStart));
        std::copy(y.begin() + (m0 - mStart), y.end(), scratch_[size_t(c)].begin());
    }
    return ReadStatus::Ok;
}

// ============================================================================
// Reading a whole source

LoadResult readWholeSource(const SourceReader& read, int channels, std::int64_t length, Planar& out,
                           const std::function<bool(float)>& goOn) {
    out.clear();
    const int kept = std::clamp(channels, 1, kMaxSourceChannels);
    if (channels < 1 || length <= 0) {
        out.assign(size_t(kept), {});
        return LoadResult::Ok;
    }
    if (std::int64_t(kept) * length > kMaxSourceSamples)
        return LoadResult::TooLong;
    try {
        out.reserve(size_t(kept));
        for (int c = 0; c < kept; ++c)
            out.emplace_back(size_t(length));
    } catch (const std::bad_alloc&) {
        Planar().swap(out);
        return LoadResult::OutOfMemory;
    }
    float* ptrs[kMaxSourceChannels] = {};
    for (std::int64_t pos = 0; pos < length; pos += kReadChunk) {
        if (goOn && !goOn(float(double(pos) / double(length))))
            return LoadResult::Cancelled;
        const std::int64_t n = std::min(kReadChunk, length - pos);
        for (int c = 0; c < kept; ++c)
            ptrs[c] = out[size_t(c)].data() + pos;
        const ReadStatus s = read(pos, n, ptrs);
        if (s == ReadStatus::NoAccess)
            return LoadResult::NoAccess;
        if (s == ReadStatus::Failed)
            return LoadResult::Failed;
    }
    if (goOn)
        goOn(1.0f);
    return LoadResult::Ok;
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
        if (audio != nullptr && audio->stream) {
            // The original, streamed: whatever the mix (there is no denoised signal)
            const SourceStream& stream = *audio->stream;
            from = std::max(from, -region.offset);
            to = std::min(to, stream.length() - region.offset);
            ++result.regionsOriginal;
            ++result.regionsStreamed;
            if (from >= to)
                continue;
            float* o[kMaxSourceChannels] = {};
            for (int c = 0; c < outChannels && c < kMaxSourceChannels; ++c)
                o[c] = out[c] + (from - position);
            result.framesMissing +=
                stream.mixInto(from + region.offset, int(to - from), o, std::min(outChannels, kMaxSourceChannels));
            continue;
        }
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

void notePlayPosition(const RenderPlan& plan, std::int64_t position) noexcept {
    for (const RegionEntry& region : plan.regions) {
        if (region.cursor == nullptr)
            continue;
        if (position >= region.end || region.end <= region.start) {
            region.cursor->position.store(-1);  // played: nothing more to read ahead
            continue;
        }
        const std::int64_t from = std::max(position, region.start);
        // Where the source starts later than the region: from the source's start
        const std::int64_t at = std::max<std::int64_t>(from + region.offset, 0);
        region.cursor->lead.store(at - region.offset - position);
        region.cursor->position.store(at);
    }
}

StreamState planStreamState(const RenderPlan& plan, std::int64_t position, int n) noexcept {
    StreamState state = StreamState::Ready;
    for (const RegionEntry& region : plan.regions) {
        const std::int64_t from = std::max(position, region.start);
        const std::int64_t to = std::min(position + n, region.end);
        if (from >= to || region.slot == nullptr)
            continue;
        const AudioSlot::Reader reader(*region.slot);
        const SourceAudio* audio = reader.get();
        if (audio == nullptr || !audio->stream || audio->stream->ready(from + region.offset, to - from))
            continue;
        if (audio->stream->failing())
            state = StreamState::Failing;
        else
            return StreamState::Pending;
    }
    return state;
}

}  // namespace enm156::ara
