#pragma once
// Streaming rational resampling between a host sample rate and the model's
// 16 kHz, in both directions. Most code wants denoiser/host_rate.h, which
// wraps these for a whole plugin; this header is the building block.
//
// Design (the same for every host rate):
//   - One linear-phase lowpass prototype per host rate: a Kaiser-windowed
//     sinc on the common grid F = M * 16000 = L * hostRate, where
//     16000 / hostRate = L / M in lowest terms (48 kHz: L 1, M 3;
//     44.1 kHz: L 160, M 441).
//   - Passband 0 .. 7 kHz, stopband from 8 kHz (cutoff 7.5 kHz), the same
//     whatever the host rate, so nothing aliases into or images out of the
//     band below 7 kHz, and the model's band edge (8 kHz) is in the stopband.
//   - Length 2 * M * Q + 1 taps on the grid, Q = kResampleHalfLength. Its
//     delay is therefore exactly Q samples at 16 kHz (M * Q grid steps), in
//     each direction, at every host rate.
//   - Applied as a polyphase filter: down (host -> 16 kHz) has L phases of
//     ceil(N / L) taps, up (16 kHz -> host) has M phases of ceil(N / M) taps.
//   - At a host rate of 16 kHz both directions are bypassed: samples are
//     copied unchanged, with no delay.
//
// Supported host rates (isSupportedHostRate): integers from 16000 to 384000
// Hz whose M (= hostRate / gcd(hostRate, 16000)) is at most 1000. That
// covers 22.05, 24, 32, 44.1, 48, 88.2, 96, 176.4, 192, 352.8 and 384 kHz.
// Rates below 16 kHz are refused (the model needs the band up to 8 kHz), as
// are rates whose ratio would need a huge table (e.g. 47999 Hz).
//
// Real-time safety: ResampleFilter::create, Downsampler::prepare and
// Upsampler::prepare allocate (do them at setup). push(), pull() and
// reset() do not allocate, lock or throw.

#include <cstdint>
#include <memory>
#include <vector>

namespace denoiser {

// The model's sample rate
constexpr int kModelRate = 16000;

// Filter design constants (see above)
constexpr double kResamplePassbandHz = 7000.0;
constexpr double kResampleStopbandHz = 8000.0;
constexpr double kResampleCutoffHz = 7500.0;
// Half the filter length, in 16 kHz samples: the delay of each direction.
// 48 gives a Kaiser design for ~94 dB stopband attenuation at a 1 kHz
// transition band; see plugin/README.md for the measured values.
constexpr int kResampleHalfLength = 48;

// True if `hostRate` is an integer this code supports (see above)
bool isSupportedHostRate(double hostRate) noexcept;

// 16000 / hostRate = up / down in lowest terms (L and M above), so that
// hostRate * up == 16000 * down. A host sample lasts `up` grid steps, a
// 16 kHz sample `down` grid steps. Throws std::invalid_argument if the rate
// is not supported.
struct RateRatio {
    int up = 1;    // L
    int down = 1;  // M
};
RateRatio hostRateRatio(int hostRate);

// The polyphase tables for one host rate. Immutable once made, so one can be
// shared by every channel's Downsampler and Upsampler (and across threads).
class ResampleFilter {
public:
    // Allocates. Throws std::invalid_argument for an unsupported rate.
    static std::shared_ptr<const ResampleFilter> create(int hostRate);

    int hostRate() const noexcept { return hostRate_; }
    bool bypass() const noexcept { return hostRate_ == kModelRate; }
    int up() const noexcept { return ratio_.up; }      // L
    int down() const noexcept { return ratio_.down; }  // M
    // Prototype length N and its centre (N - 1) / 2, in grid steps
    int length() const noexcept { return length_; }
    std::int64_t centre() const noexcept { return (length_ - 1) / 2; }

    // The prototype on the grid, normalised to a DC gain of 1. Empty when
    // bypassed. For tests and analysis.
    const std::vector<double>& prototype() const noexcept { return prototype_; }

    // Polyphase tables, each phase's taps stored oldest-sample-first
    int downTaps() const noexcept { return downTaps_; }  // per phase, L phases
    int upTaps() const noexcept { return upTaps_; }      // per phase, M phases
    const float* downPhase(int phase) const noexcept { return downTable_.data() + std::size_t(phase) * downTaps_; }
    const float* upPhase(int phase) const noexcept { return upTable_.data() + std::size_t(phase) * upTaps_; }

private:
    ResampleFilter() = default;

    int hostRate_ = kModelRate;
    RateRatio ratio_;
    int length_ = 1;
    int downTaps_ = 0;
    int upTaps_ = 0;
    std::vector<double> prototype_;
    std::vector<float> downTable_;
    std::vector<float> upTable_;
};

// Host rate -> 16 kHz, one channel, one sample at a time.
// Output 16 kHz sample n is the input at host time n * 16000 / hostRate,
// delayed by kResampleHalfLength samples at 16 kHz (exactly; bypass: none).
class Downsampler {
public:
    // Keeps a reference to `filter`; allocates the history, then reset()s.
    void prepare(std::shared_ptr<const ResampleFilter> filter);
    // Silence before the next sample; the next output is 16 kHz sample 0
    void reset() noexcept;

    // Takes one host-rate sample. Returns true and sets *out when that
    // completes a 16 kHz sample (at most one per input, as hostRate >= 16000).
    bool push(float in, float* out) noexcept;

private:
    std::shared_ptr<const ResampleFilter> filter_;
    std::vector<float> history_;  // mirrored ring, 2 * downTaps
    int write_ = 0;
    std::int64_t pushed_ = 0;     // host samples taken since reset
    std::int64_t nextTime_ = 0;   // grid time of the next output, n * M
};

// 16 kHz -> host rate, one channel. Output host sample m is the 16 kHz
// input at grid time m * L + offset, delayed by kResampleHalfLength samples
// at 16 kHz. `offset` (0 <= offset < L) lets the caller make the total
// delay of a down + up chain a whole number of host samples; HostRateAdapter
// works it out.
class Upsampler {
public:
    void prepare(std::shared_ptr<const ResampleFilter> filter, int offset);
    void reset() noexcept;

    // Takes one 16 kHz sample
    void push(float in) noexcept;
    // True if the next host sample can be computed from what was pushed
    bool ready() const noexcept;
    // The next host sample. Call only when ready(); otherwise returns 0 and
    // counts an underrun (a bug in the caller's scheduling).
    float pull() noexcept;

    std::int64_t underruns() const noexcept { return underruns_; }

private:
    std::shared_ptr<const ResampleFilter> filter_;
    int offset_ = 0;
    int capacity_ = 0;
    std::vector<float> history_;  // mirrored ring, 2 * capacity
    int write_ = 0;
    std::int64_t pushed_ = 0;     // 16 kHz samples taken since reset
    std::int64_t nextTime_ = 0;   // grid time of the next output, m * L + offset
    std::int64_t underruns_ = 0;
};

}  // namespace denoiser
