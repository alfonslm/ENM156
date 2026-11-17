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
// - SourceStream: the original of a source that is not held in memory (too
//   long, out of memory, or the host failed to read it whole), read from the
//   host ahead of the play position by a background thread into a fixed pool
//   of preallocated blocks, and played from there by the audio thread
//   without locks or allocation. PlayCursor: where one region wants it.

#include "denoiser/denoiser.h"
#include "denoiser/host_rate.h"
#include "denoiser/offline.h"
#include "denoiser/resample.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace enm156::ara {

using Planar = std::vector<std::vector<float>>;

static_assert(std::atomic<std::int64_t>::is_always_lock_free, "the audio thread needs lock-free 64-bit atomics");

// At most this many channels of a source are stored and played (mono or
// stereo, like the plugin's buses); further channels are ignored
constexpr int kMaxSourceChannels = 2;
// Largest source the analysis takes, in samples (all channels together):
// 400 M = 69 minutes of 48 kHz stereo, 2.3 hours of 48 kHz mono. Everything
// is kept in memory as float32: the original and the denoised audio at the
// source's rate plus the 16 kHz input and output (a third each at 48 kHz),
// so about 2.7 x 4 bytes per sample: up to 4.3 GB for a source at the limit
// (an hour of 48 kHz stereo: 3.7 GB). Longer sources are not read into
// memory: their original is streamed from the host (SourceStream), not
// denoised, and the editor says why.
constexpr std::int64_t kMaxSourceSamples = 400'000'000;

// ---- Reading a source from the host ----

enum class ReadStatus { Ok, Failed, NoAccess };
// Reads source frames [position, position + n) (at the source's rate, inside
// the source) of the kept channels into out[0 .. kept). Not the audio thread.
using SourceReader = std::function<ReadStatus(std::int64_t position, std::int64_t n, float* const* out)>;

// ---- Streaming the original ----

// Where one region wants a streamed source's audio, written by the audio
// thread on every block (also while the transport is stopped), read by the
// streaming thread. Positions are in the stream's frames (the playback rate).
struct PlayCursor {
    std::atomic<std::int64_t> position{-1};  // the next frame to be played; -1: nothing wanted
    std::atomic<std::int64_t> lead{0};       // frames until the play position reaches it (0: playing it now)
};
struct CursorState {
    std::int64_t position = -1, lead = 0;
};

class SourceStream {
public:
    // Frames per block (at the playback rate); the pool holds kPoolBlocks of
    // them, all channels, allocated once (64 x 16384 x 2 x 4 bytes = 8 MB
    // for stereo). Blocks are read ahead until kReadAheadFrames past each
    // cursor (2.7 s at 48 kHz, 0.7 s at 192 kHz).
    static constexpr std::int64_t kBlockFrames = 16384;
    static constexpr int kPoolBlocks = 64;
    static constexpr std::int64_t kReadAheadFrames = 8 * kBlockFrames;
    // After a failed host read, the next attempt for this stream waits this long
    static constexpr std::chrono::milliseconds kRetryDelay{50};

    // The original of a source (sourceLength frames of `channels` <= 2 at
    // sourceRate) played at playbackRate: at the source's own rate the
    // samples as read; at another rate through 16 kHz, exactly what an
    // analysed source plays there (SourceAnalysis::atRate's dry). Allocates
    // the pool: may throw std::bad_alloc; throws std::invalid_argument if the
    // rates differ and one of them is not supported.
    SourceStream(double sourceRate, int channels, std::int64_t sourceLength, int playbackRate);
    SourceStream(const SourceStream&) = delete;
    SourceStream& operator=(const SourceStream&) = delete;

    int channels() const noexcept { return channels_; }
    int playbackRate() const noexcept { return playbackRate_; }
    std::int64_t length() const noexcept { return length_; }  // frames at the playback rate

    // Audio thread. Adds frames [position, position + n) (inside the stream)
    // to out[0 .. outChannels) (mono on every output channel; stereo on a
    // mono output: the mean), where they are in the pool; the rest stays as
    // it is (silence). Returns the number of frames that were not there.
    // Wait-free: atomic loads and increments, no lock, no allocation.
    int mixInto(std::int64_t position, int n, float* const* out, int outChannels) const noexcept;
    // True if all of [position, position + n) (clipped to the stream) is in the pool
    bool ready(std::int64_t position, std::int64_t n) const noexcept;
    // True while the latest host read for this stream failed (or had no access)
    bool failing() const noexcept { return failing_.load(); }
    // Any thread: the source's samples changed; nothing of this stream plays
    // from now on (a new stream replaces it)
    void invalidate() noexcept { valid_ = false; }
    bool valid() const noexcept { return valid_.load(); }

    // The streaming thread (one at a time): loads at most one block, the
    // most urgent one the cursors want and the pool lacks, evicting a block
    // no cursor needs (or one needed later than this one).
    enum class Fill { Idle, Loaded, Failed, NoAccess };
    Fill fillOne(const std::vector<CursorState>& cursors, const SourceReader& read);

    // Frames read from the host so far (for tests and statistics)
    std::int64_t framesRead() const noexcept { return framesRead_.load(); }

private:
    struct Entry {
        std::atomic<std::int64_t> block{-1};  // the block it holds; -1: none (or being replaced)
        std::atomic<int> readers{0};
    };
    float* data(int entry, int channel) noexcept;
    const float* data(int entry, int channel) const noexcept;
    ReadStatus compute(std::int64_t block, const SourceReader& read);
    std::int64_t urgency(std::int64_t block, const std::vector<CursorState>& cursors) const noexcept;

    int channels_;
    std::int64_t sourceLength_;
    int playbackRate_;
    std::int64_t length_;
    std::int64_t blocks_;
    std::shared_ptr<const denoiser::ResampleFilter> in_, out_;  // null at the source's own rate

    std::unique_ptr<Entry[]> entries_;
    std::vector<float> pool_;  // entry, channel, frame
    std::atomic<bool> failing_{false};
    std::atomic<bool> valid_{true};
    std::atomic<std::int64_t> framesRead_{0};

    // Streaming thread only
    std::vector<std::int64_t> loaded_;      // mirror of entries_[i].block
    std::vector<std::uint64_t> loadedAt_;   // when (a counter), for evicting the oldest
    std::uint64_t loads_ = 0;
    std::chrono::steady_clock::time_point retryAt_{};
    Planar scratch_;                        // one block, all channels
    Planar sourceScratch_;                  // source frames for a resampled block
};

struct SourceAudio {
    double sampleRate = 0.0;  // of these samples
    int channels = 0;         // 1 or 2
    std::int64_t length = 0;  // samples per channel
    // The original (at another rate than the source's: through 16 kHz).
    // Shared with the analysis and between publications, not copied.
    std::shared_ptr<const Planar> dry;
    Planar wet;               // denoised; empty: not available (yet), the original plays
    // Instead of dry and wet: the original, streamed from the host (no
    // denoised audio). Shared by every publication for this source and rate.
    std::shared_ptr<SourceStream> stream;
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

    // Control side: a cursor for one region playing this slot (kept alive by
    // the render plan; it stops counting when the last owner drops it)
    std::shared_ptr<PlayCursor> addCursor();
    // Streaming thread: the live cursors' states (replaces `out`)
    void cursorStates(std::vector<CursorState>& out) const;

private:
    std::atomic<const SourceAudio*> current_{nullptr};
    mutable std::atomic<unsigned> side_{0};
    mutable std::array<std::atomic<int>, 2> readers_{};
    mutable std::mutex writeMutex_;
    std::shared_ptr<const SourceAudio> owner_;  // writeMutex_
    mutable std::mutex cursorMutex_;
    mutable std::vector<std::weak_ptr<PlayCursor>> cursors_;  // cursorMutex_
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

// Reads a whole source into memory (kept channels, at most
// kMaxSourceSamples in all), in chunks of kReadChunk frames. `goOn` is asked
// before each chunk (false: Cancelled) and told the fraction read.
enum class LoadResult { Ok, Cancelled, NoAccess, Failed, TooLong, OutOfMemory };
constexpr std::int64_t kReadChunk = 1 << 16;
LoadResult readWholeSource(const SourceReader& read, int channels, std::int64_t length, Planar& out,
                           const std::function<bool(float)>& goOn);

// ---- Rendering ----

struct RegionEntry {
    std::int64_t start = 0, end = 0;  // timeline samples [start, end) at the render rate
    std::int64_t offset = 0;          // position in the slot's audio = timeline sample + offset
    AudioSlot* slot = nullptr;
    PlayCursor* cursor = nullptr;     // registered with `slot`; null: none
};

struct RenderPlan {
    std::vector<RegionEntry> regions;
    std::vector<std::shared_ptr<AudioSlot>> keepAlive;  // owns every slot in `regions`
    std::vector<std::shared_ptr<PlayCursor>> cursors;   // owns every cursor in `regions`
};

// What a render call found (for tests and the editor)
struct RenderResult {
    int regionsRendered = 0;  // regions overlapping the block that had audio
    int regionsSilent = 0;    // overlapping regions with nothing stored yet (silence)
    int regionsOriginal = 0;  // overlapping regions playing the original (no wet yet / not denoised)
    int regionsStreamed = 0;  // of those, streamed from the host
    int framesMissing = 0;    // streamed frames not read yet (silence)
};

// Renders timeline samples [position, position + n) into out[0..outChannels)
// (overwritten): the sum over the regions of (1 - g) dry + g wet, g =
// wetGain[i] (0 = original, 1 = denoised); a streamed source plays its
// original whatever g. Real-time safe: no allocation, locks or system calls.
RenderResult renderPlan(const RenderPlan& plan, std::int64_t position, int n, float* const* out, int outChannels,
                        const float* wetGain) noexcept;

// Audio thread, every block (playing or not): tells each region's cursor
// where the play position is. Real-time safe.
void notePlayPosition(const RenderPlan& plan, std::int64_t position) noexcept;

// True if a region overlapping [position, position + n) has analysis work pending
bool planIsWorking(const RenderPlan& plan, std::int64_t position, int n) noexcept;

// For non-realtime renders: whether the streamed audio for [position,
// position + n) is all there (Ready), still being read (Pending), or not
// coming now because the host's reads fail (Failing)
enum class StreamState { Ready, Pending, Failing };
StreamState planStreamState(const RenderPlan& plan, std::int64_t position, int n) noexcept;

}  // namespace enm156::ara
