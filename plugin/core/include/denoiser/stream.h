#pragma once
// Live, causal processing one hop at a time: denoiser/stream.py's
// StreamingDenoiser without the network. Causal models only.
//
//   denoiser::StreamProcessor stream(model.info());   // allocates; throws for centred models
//   // per hop (hop = stream.hop() samples in, hop samples out):
//   const float* x = stream.analyse(in);              // model input frame, n_in floats
//   network.process(x, pred);                          // n_out floats
//   int preRoll = stream.synthesise(pred, out);        // hop output samples
//   // or, the same in one call:
//   int preRoll = stream.process(in, out, network);
//
// The output is the offline result delayed by delay() = n_fft - hop samples.
// The first delay() output samples after a reset (the "pre-roll") belong to
// time before the input started: they reproduce Python exactly but are the
// model's output for silence divided by the tiny start of the window
// envelope, so they can be large. synthesise() returns how many samples at
// the start of this hop's output are pre-roll; a plugin should mute them.
//
// analyse(), synthesise(), process() and reset() don't allocate, lock or
// throw. Not thread-safe: one instance per audio stream.

#include "denoiser/features.h"
#include "denoiser/model_file.h"
#include "denoiser/network.h"

#include <cstdint>
#include <vector>

namespace denoiser {

class StreamProcessor {
public:
    // Throws std::invalid_argument for a centred (realtime = false) model
    explicit StreamProcessor(const ModelInfo& info);

    int hop() const noexcept { return codec_.info().hop; }
    int inputSize() const noexcept { return codec_.info().nIn; }   // of a model input frame
    int outputSize() const noexcept { return codec_.info().nOut; } // of a model output frame
    // Output = offline result delayed by this many samples (n_fft - hop)
    int delay() const noexcept { return codec_.info().nFft - codec_.info().hop; }
    // Delay as a host sees it: a whole hop must arrive first (n_fft); stream.py's latency
    int latency() const noexcept { return codec_.info().nFft; }

    // Back to the just-constructed state: silent history, empty overlap-add
    // buffers, pre-roll counter restarted. Does not touch the network.
    void reset() noexcept;

    // Takes hop() new input samples; returns the model input frame (n_in
    // floats, valid until the next analyse()).
    const float* analyse(const float* in) noexcept;

    // Takes the model output for the frame of the last analyse() (n_out
    // floats) and writes hop() output samples. Returns the number of
    // pre-roll samples at the start of out (0 once delay() samples have
    // been output since the reset).
    int synthesise(const float* pred, float* out) noexcept;

    // analyse, network, synthesise. Uses an internal n_out buffer.
    int process(const float* in, float* out, FrameNetwork& network) noexcept;

    // Like synthesise(), with the noisy spectrum as the output spectrum
    // (no network): the output is the input delayed by delay()
    int synthesisePassThrough(float* out) noexcept;

    // The frame codec, for its intermediate results (spectrum(), ...)
    const FrameCodec& codec() const noexcept { return codec_; }

    // Output samples since the reset
    std::int64_t samplesOut() const noexcept { return samplesOut_; }

private:
    int overlapAdd(const float* frame, float* out) noexcept;

    FrameCodec codec_;
    std::vector<float> history_;   // last max(n_fft, long_n_fft) input samples
    std::vector<float> pending_;   // overlap-add of frames, n_fft
    std::vector<float> envelope_;  // overlap-added window^2, n_fft
    std::vector<float> windowSq_;  // window^2
    std::vector<float> pred_;      // for process()
    std::int64_t samplesOut_ = 0;
};

}  // namespace denoiser
