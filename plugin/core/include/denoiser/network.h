#pragma once
// What sits between the two halves of the STFT code: one model input frame
// (ModelInfo::nIn floats) in, one model output frame (nOut floats) out, in
// frame order, carrying any state (SNN potentials, TCN history) itself.
//
// Implementations (MLP, SNN, TCN) are built from a ModelFile: its net() for
// the structure and tensors() for the weights. The STFT code never looks
// inside a network.

namespace denoiser {

class FrameNetwork {
public:
    virtual ~FrameNetwork() = default;

    // Back to the state of a just-constructed network: the start of a
    // recording (SNN potentials zero, TCN history = the zero input frames
    // FrameModel.forward pads a clip with). Real-time safe.
    virtual void reset() noexcept = 0;

    // One frame: x = nIn floats -> y = nOut floats, after the output
    // activation. Called once per frame, in order. Real-time safe: no
    // allocation, locks or exceptions.
    virtual void process(const float* x, float* y) noexcept = 0;
};

}  // namespace denoiser
