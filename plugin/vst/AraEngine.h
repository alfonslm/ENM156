#pragma once
// The ARA side of the plugin below the ARA interface: no JUCE and no ARA SDK
// here, so it can be tested on its own (denoiser_ara_engine_test). See
// DocumentController.h for how the ARA classes use it, and README.md "ARA".
//
// - SourceAudio: what one audio source plays, at one sample rate: the
//   original samples and the denoised ones (or none yet). Immutable once
//   published.
// - AudioSlot: where the stored audio of one audio source (at one playback
//   rate) is published by the analysis thread and read by the audio thread,
//   without locks or allocation on the audio thread. A new SourceAudio
//   replaces the old one atomically; the old one is freed by the publishing
//   thread once no audio thread can still be reading it.
// - SourceAnalysis: the offline denoising of one audio source (whole clip,
//   any supported rate, causal or centred models), keeping the 16 kHz
//   intermediate results so that a high-band change or another playback
//   rate needs no new network run.
// - RenderPlan / renderPlan(): the playback regions of one renderer as plain
//   numbers (timeline range -> position in the stored audio), and the render
//   call: for each output sample, the sum over the regions covering it of
//   (1 - g) dry + g wet, g = the mix (0 when bypassed): exactly the original
//   at g = 0, exactly the denoised audio at g = 1.

#include "denoiser/denoiser.h"
#include "denoiser/host_rate.h"
#include "denoiser/offline.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace enm156::ara {

using Planar = std::vector<std::vector<float>>;

// At most this many channels of a source are stored and played (mono or
// stereo, like the plugin's buses); further channels are ignored
constexpr int kMaxSourceChannels = 2;
// Largest source the analysis takes, in samples (all channels together):
// 400 M = 69 minutes of 48 kHz stereo, 2.3 hours of 48 kHz mono. Everything
// is kept in memory as float32: the original and the denoised audio at the
// source's rate plus the 16 kHz input and output (a third each at 48 kHz),
// so about 2.7 x 4 bytes per sample: up to 4.3 GB for a source at the limit
// (an hour of 48 kHz stereo: 3.7 GB). Longer sources are not read: they play
// silence and the editor says why.
constexpr std::int64_t kMaxSourceSamples = 400'000'000;

struct SourceAudio {
    double sampleRate = 0.0;  // of these samples
    int channels = 0;         // 1 or 2
    std::int64_t length = 0;  // samples per channel
    // The original (at another rate than the source's: through 16 kHz).
    // Shared with the analysis and between publications, not copied.
    std::shared_ptr<const Planar> dry;
    Planar wet;               // denoised; empty: not available (yet), the original plays
};

class AudioSlot {
public:
    AudioSlot() = default;
    AudioSlot(const AudioSlot&) = delete;
    AudioSlot& operator=(const AudioSlot&) = delete;

    // Any thread except the audio thread (here: the analysis thread). Makes
    // `next` (may be null) the audio readers see; waits until no reader can
    // still hold the previous one, then frees it (if this was its last owner).
    void publish(std::shared_ptr<const SourceAudio> next);
    // The current audio (for the analysis thread and the UI; not real-time safe)
    std::shared_ptr<const SourceAudio> current() const;

    // Audio thread: a scoped read. get() stays valid until the Reader is
    // destroyed. Real-time safe (two atomic increments and decrements).
    class Reader {
    public:
        explicit Reader(const AudioSlot& slot) noexcept;
        ~Reader();
        Reader(const Reader&) = delete;
        Reader& operator=(const Reader&) = delete;
        const SourceAudio* get() const noexcept { return audio_; }

    private:
        const AudioSlot& slot_;
        unsigned side_;
        const SourceAudio* audio_;
    };

    // True while the analysis has work queued or running that will change
    // what this slot holds. Non-realtime renders wait for it (see renderer).
    std::atomic<bool> working{false};

private:
    std::atomic<const SourceAudio*> current_{nullptr};
    mutable std::atomic<unsigned> side_{0};
    mutable std::array<std::atomic<int>, 2> readers_{};
    mutable std::mutex writeMutex_;
    std::shared_ptr<const SourceAudio> owner_;  // writeMutex_
};

// ---- Analysis ----

enum class Outcome {
    Denoised,         // wet is available
    NoModel,          // the original plays
    UnsupportedRate,  // the source's rate can't be resampled to 16 kHz: the original plays
    TooShort,         // shorter than a centred model needs: the original plays
    Empty,            // zero-length source
    Cancelled,
    Failed,           // an exception (e.g. out of memory): the original plays
};

// Called with a fraction 0..1 (or a negative value: only asking whether to
// go on); return false to cancel
using Progress = std::function<bool(float)>;

class SourceAnalysis {
public:
    // The source's samples (at most kMaxSourceChannels channels, equal
    // lengths) at its own rate. Resamples to 16 kHz if the rate is supported.
    // Allocates; may throw std::bad_alloc.
    void setInput(double sampleRate, Planar input);

    // Runs the network over each channel at 16 kHz (whole clip). Cancellable.
    // Keeps the result for wet() / rendition(). model may be null.
    Outcome denoise(const std::shared_ptr<const denoiser::Model>& model, const Progress& progress);

    // The audio at the source's own rate: dry = the input, wet = the
    // denoised clip resampled back (+ the input's band above ~7.5 kHz with
    // HighBand::PassThrough), or no wet if not denoised. Allocates.
    std::shared_ptr<const SourceAudio> atSourceRate(denoiser::HighBand highBand) const;
    // The audio at another playback rate, through 16 kHz (dry too, so it is
    // band-limited to ~7.5 kHz; the high band is not available here). Null
    // if the source rate or `rate` is not supported. Allocates.
    std::shared_ptr<const SourceAudio> atRate(int rate) const;

    double sampleRate() const noexcept { return rate_; }
    int channels() const noexcept { return input_ ? int(input_->size()) : 0; }
    std::int64_t length() const noexcept {
        return input_ && !input_->empty() ? std::int64_t((*input_)[0].size()) : 0;
    }
    bool hasInput() const noexcept { return hasInput_; }
    bool denoised() const noexcept { return !y16_.empty(); }
    // Why there is no wet audio (empty if denoised)
    const std::string& message() const noexcept { return message_; }

private:
    double rate_ = 0.0;
    bool hasInput_ = false;
    std::shared_ptr<const Planar> input_;  // at rate_ (also the stored audio's dry)
    Planar x16_;    // input at 16 kHz (empty if rate_ is unsupported)
    Planar y16_;    // denoised at 16 kHz (empty if not denoised)
    std::string message_;
};

// True if `rate` is a rate the analysis can resample from/to (an integer
// rate isSupportedHostRate accepts)
bool isSupportedRate(double rate) noexcept;

// ---- Rendering ----

struct RegionEntry {
    std::int64_t start = 0, end = 0;  // timeline samples [start, end) at the render rate
    std::int64_t offset = 0;          // position in the slot's audio = timeline sample + offset
    AudioSlot* slot = nullptr;
};

struct RenderPlan {
    std::vector<RegionEntry> regions;
    std::vector<std::shared_ptr<AudioSlot>> keepAlive;  // owns every slot in `regions`
};

// What a render call found (for tests and the editor)
struct RenderResult {
    int regionsRendered = 0;  // regions overlapping the block that had audio
    int regionsSilent = 0;    // overlapping regions with nothing stored yet (silence)
    int regionsOriginal = 0;  // overlapping regions playing the original (no wet yet / not denoised)
};

// Renders timeline samples [position, position + n) into out[0..outChannels)
// (overwritten): the sum over the regions of (1 - g) dry + g wet, g =
// wetGain[i] (0 = original, 1 = denoised). Real-time safe: no allocation, locks or system calls.
RenderResult renderPlan(const RenderPlan& plan, std::int64_t position, int n, float* const* out, int outChannels,
                        const float* wetGain) noexcept;

// True if a region overlapping [position, position + n) has analysis work pending
bool planIsWorking(const RenderPlan& plan, std::int64_t position, int n) noexcept;

}  // namespace enm156::ara
