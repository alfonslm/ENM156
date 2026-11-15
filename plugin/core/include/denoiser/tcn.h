#pragma once
// The TCN (denoiser/nets.py: TCN, TCNBlock, FrameNorm), evaluated
// incrementally: one new frame per process() call at a constant cost, instead
// of recomputing the whole context (127 frames by default) like
// denoiser/stream.py does. Semantics: plugin/tools/README.md, "TCN".
//
//   auto net = denoiser::makeTcnNetwork(model);              // allocates, throws FormatError
//   net->reset();                                             // real-time safe
//   net->process(x, y);                                       // n_in floats -> n_out floats
//
//   // several instances (e.g. one per channel) sharing one copy of the weights:
//   auto weights = std::make_shared<const denoiser::TcnModel>(model);
//   denoiser::TcnNetwork left(weights), right(weights);
//
// State: per block, a ring of the last trim = (kernel - 1) * dilation frames
// of relu(LayerNorm(block input)); with the current frame, that is all the
// dilated convolution reads (the residual only needs the current frame), so
// the LayerNorm runs once per frame and block. Nothing else is kept. The start-up of
// FrameModel.forward (context - 1 zero *input* frames in front of the clip)
// is run once, when the TcnModel is built; reset() copies that snapshot back.
//
// TcnModel is immutable once built: share it between threads freely.
// TcnNetwork is one stream's state: one instance per audio stream, not
// thread-safe. reset() and process() don't allocate, lock or throw.

#include "denoiser/model_file.h"
#include "denoiser/network.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace denoiser {

class TcnNetwork;

// The ring buffers of all blocks, and where each block writes next
struct TcnState {
    std::vector<float> rings;      // block b: ringFrames(b) * channels floats at ringOffset(b)
    std::vector<int> position;     // per block: the ring slot of its oldest frame (the next one written)
};

// The weights, laid out for the kernels, plus the start-up snapshot
class TcnModel {
public:
    // Throws FormatError if the model is not a consistent TCN
    explicit TcnModel(const ModelFile& model);

    int inputSize() const noexcept { return nIn_; }
    int outputSize() const noexcept { return nOut_; }
    int channels() const noexcept { return channels_; }
    int kernel() const noexcept { return kernel_; }
    int blocks() const noexcept { return static_cast<int>(blocks_.size()); }
    int dilation(int b) const noexcept { return blocks_[static_cast<std::size_t>(b)].dilation; }
    int context() const noexcept { return context_; }
    // Past frames in block b's ring: trim = (kernel - 1) * dilation
    int ringFrames(int b) const noexcept { return blocks_[static_cast<std::size_t>(b)].ringFrames; }
    std::size_t ringOffset(int b) const noexcept { return blocks_[static_cast<std::size_t>(b)].ringOffset; }
    // Floats of state per instance (all rings)
    std::size_t stateFloats() const noexcept { return stateFloats_; }
    // Multiply-adds per frame of a dense evaluation (zeros not skipped)
    std::size_t macsPerFrame() const noexcept;

    // The state after the start-up: what reset() restores
    const TcnState& startState() const noexcept { return start_; }

    // Scratch memory of one evaluation (per instance)
    struct Work {
        std::vector<float> hidden;  // (blocks + 1) * C: h_0 .. h_B of the last frame
        std::vector<float> conv;    // C
        std::vector<float> current; // C: relu(LayerNorm(h_b)) of the current frame
        std::vector<float> normed;  // C (final norm)
        std::vector<float> out;     // n_out (start-up outputs, discarded)
        std::vector<float> zeros;   // n_in zeros: the start-up's input frame
        std::vector<int> index;     // a layer's nonzero inputs: up to max(n_in, kernel * C)
        std::vector<float> value;
    };
    Work makeWork() const;
    // A state of the right size, all zeros, positions 0
    TcnState makeState() const;

private:
    friend class TcnNetwork;

    // One frame: x (n_in) -> y (n_out), advancing the state
    void step(TcnState& state, Work& work, const float* x, float* y) const noexcept;

    struct Block {
        int dilation = 1;
        int ringFrames = 0;
        std::size_t ringOffset = 0;
        std::vector<float> normWeight, normBias;  // [C]
        std::vector<float> convWeight;            // [tap][in][out]: row (j * C + i) has C outputs
        std::vector<float> convBias;              // [C]
        std::vector<float> mixWeight;             // [in][out]
        std::vector<float> mixBias;               // [C]
    };

    int nIn_ = 0, nOut_ = 0, channels_ = 0, kernel_ = 0, context_ = 1;
    float eps_ = 1e-5f;
    Activation outActivation_ = Activation::None;
    std::vector<float> inWeight_, inBias_;        // [n_in][C], [C]
    std::vector<Block> blocks_;
    std::vector<float> finalWeight_, finalBias_;  // [C]
    std::vector<float> outWeight_, outBias_;      // [C][n_out], [n_out]
    std::size_t stateFloats_ = 0;
    TcnState start_;
};

class TcnNetwork final : public FrameNetwork {
public:
    explicit TcnNetwork(const ModelFile& model);  // its own TcnModel
    explicit TcnNetwork(std::shared_ptr<const TcnModel> model);

    // Copies the start-up snapshot back: O(state size), ~135 kB for the default TCN
    void reset() noexcept override;
    void process(const float* x, float* y) noexcept override;

    const TcnModel& model() const noexcept { return *model_; }

    // Channels of the last frame: hidden(0) after the input layer,
    // hidden(b + 1) after block b (the golden vectors' tcn.h*)
    const float* hidden(int i) const noexcept {
        return work_.hidden.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(model_->channels());
    }

    // ---- for tests (not real-time: start-up costs context - 1 frames) ------
    const TcnState& state() const noexcept { return state_; }
    // Ring contents in time order (oldest first) per block, independent of
    // where each ring starts
    std::vector<float> orderedState() const;
    // Fill the rings with finite garbage and put the write positions anywhere
    void scrambleState(std::uint32_t seed) noexcept;
    // The straightforward start-up from the current state: push context - 1
    // zero input frames, discarding the outputs
    void startUp() noexcept;

private:
    std::shared_ptr<const TcnModel> model_;
    TcnState state_;
    TcnModel::Work work_;
};

// The FrameNetwork for a model whose net.type is "tcn". Throws FormatError.
std::unique_ptr<FrameNetwork> makeTcnNetwork(const ModelFile& model);

}  // namespace denoiser
