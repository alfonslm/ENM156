#pragma once
// The frame networks (FrameNetwork, network.h) built from a model file:
// the MLP of denoiser/nets.py and the spiking LIFNet of denoiser/snn/nets.py,
// following plugin/tools/README.md ("Tensors") and tools/reference.py.
//
//   auto net = denoiser::makeNetwork(modelFile);   // allocates; throws FormatError
//   net->reset();                                  // start of a recording
//   net->process(x, y);                            // per frame: n_in floats -> n_out floats
//
// Most code should use denoiser::Model (denoiser.h), which loads a file once
// and makes any number of networks that share its weights.
//
// Lifetime: a network reads the weights in place (MlpNetwork from the
// ModelFile's tensors; SnnNetwork from an SnnWeights that also points into
// the ModelFile), so the ModelFile must outlive every network made from it.
//
// Threads: weights are read-only after construction and can be shared by
// networks on any number of threads. A network itself (its state and
// scratch memory) belongs to one thread at a time. process() and reset()
// don't allocate, lock or throw.

#include "denoiser/model_file.h"
#include "denoiser/network.h"

#include <memory>
#include <vector>

namespace denoiser {

// Multilayer perceptron: Linear + ReLU per hidden layer, Linear, output
// activation. Stateless.
class MlpNetwork final : public FrameNetwork {
public:
    // Throws FormatError for missing tensors, wrong shapes or sizes that
    // don't match the model's n_in / n_out
    explicit MlpNetwork(const ModelFile& model);

    void reset() noexcept override {}
    void process(const float* x, float* y) noexcept override;

    int layers() const noexcept { return static_cast<int>(layers_.size()); }
    // Multiply-adds per frame
    long long macsPerFrame() const noexcept;

private:
    struct Layer {
        const float* weight = nullptr;  // [out, in], row-major, in the ModelFile
        const float* bias = nullptr;    // [out]
        int in = 0, out = 0;
        Activation activation = Activation::None;
    };
    std::vector<Layer> layers_;
    std::vector<float> ping_, pong_;  // the widest hidden layer each
};

// The SNN's weights, prepared once per model and shared read-only by all
// SnnNetworks of that model. Besides pointers into the ModelFile (which
// must outlive this) it holds the matrices fed by spikes transposed, so
// that the column of an input that spiked is contiguous.
class SnnWeights {
public:
    // Throws FormatError for missing tensors, wrong shapes or bad settings
    static std::shared_ptr<const SnnWeights> build(const ModelFile& model);

    struct Layer {
        int in = 0, size = 0;
        const float* weight = nullptr;     // [size, in] row-major (ModelFile)
        const float* bias = nullptr;       // [size]
        const float* decay = nullptr;      // [size], already through the sigmoid
        const float* recurrent = nullptr;  // [size, size] row-major, or nullptr
        std::vector<float> weightT;        // [in, size]: weight transposed (layers after the first)
        std::vector<float> recurrentT;     // [size, size]: recurrent transposed (if recurrent)
    };

    int nIn = 0, nOut = 0;
    int steps = 1;           // time steps per frame
    float threshold = 1.0f;
    bool recurrent = false;
    Activation outActivation = Activation::None;
    std::vector<Layer> layers;
    const float* readout = nullptr;       // [nOut, last size] row-major (ModelFile)
    std::vector<float> readoutT;          // [last size, nOut]
    const float* readoutDecay = nullptr;  // [nOut]
    const float* outputBias = nullptr;    // [nOut]

    // Multiply-adds per frame if every product were computed (the dense cost)
    long long denseMacsPerFrame() const noexcept;
};

// Leaky integrate-and-fire network with a leaky readout (LIFNet.step).
// State (membrane potentials, last spikes, readout potentials) carries over
// from frame to frame and is zero after reset().
class SnnNetwork final : public FrameNetwork {
public:
    // How the matrices fed by spikes (layers after the first, recurrent,
    // readout) are applied. Both give the same spikes on the golden cases;
    // the sums differ in rounding only.
    enum class Kernel {
        Spikes,  // add the (transposed) weight column of each input that spiked: no multiplies,
                 // work proportional to the spike rate. The default: faster than Dense at
                 // every spike rate measured (0.02 to 0.45 per neuron and step)
        Dense,   // full matrix-vector products with the 0/1 spike vectors, like PyTorch
    };

    // Called during process() with each layer's spikes of every time step
    // (layer 0 first; steps in order). For tests and statistics; nullptr = off.
    class SpikeObserver {
    public:
        virtual ~SpikeObserver() = default;
        virtual void spikes(int layer, const float* s, int size) noexcept = 0;
    };

    explicit SnnNetwork(std::shared_ptr<const SnnWeights> weights, Kernel kernel = Kernel::Spikes);
    // Builds its own SnnWeights
    explicit SnnNetwork(const ModelFile& model, Kernel kernel = Kernel::Spikes);

    void reset() noexcept override;
    void process(const float* x, float* y) noexcept override;

    void setSpikeObserver(SpikeObserver* observer) noexcept { observer_ = observer; }
    // Spikes since the last reset, all layers, and the time steps they were counted over
    long long spikeCount() const noexcept { return spikeCount_; }
    long long neuronSteps() const noexcept { return neuronSteps_; }

    const SnnWeights& weights() const noexcept { return *w_; }

private:
    struct State {
        std::vector<float> v;    // membrane potentials
        std::vector<float> s;    // spikes of the last time step (0 / 1)
        std::vector<int> active; // indices where s is 1
        int count = 0;           // ... how many
    };

    std::shared_ptr<const SnnWeights> w_;
    Kernel kernel_;
    std::vector<State> state_;
    std::vector<float> u_;      // readout potentials
    std::vector<float> first_;  // the first layer's current, once per frame
    std::vector<float> cur_;    // scratch: the widest layer
    std::vector<float> tmp_;    // scratch: max(widest layer, n_out)
    SpikeObserver* observer_ = nullptr;
    long long spikeCount_ = 0, neuronSteps_ = 0;
};

// The network a model file describes ("mlp" or "snn"). Throws FormatError
// for malformed weights or an unknown type, and std::runtime_error for
// "tcn", which this build can't run yet.
std::unique_ptr<FrameNetwork> makeNetwork(const ModelFile& model);

}  // namespace denoiser
