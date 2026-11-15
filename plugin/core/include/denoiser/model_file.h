#pragma once
// A model file ("kind": "denoiser-model", plugin/tools/README.md): the
// container, plus its "derived" settings parsed and checked for consistency.
//
//   auto model = denoiser::ModelFile::load("model.bin");     // throws FormatError
//   const denoiser::ModelInfo& info = model.info();          // n_fft, hop, ... for the STFT code
//   const denoiser::Json& net = model.net();                 // "type", "layers", ... for a network
//   const denoiser::Tensor& w = model.tensors().get("linear0.weight", {24, 260});
//
// Loading allocates and throws; afterwards everything is read-only.

#include "denoiser/format_error.h"
#include "denoiser/json.h"
#include "denoiser/tensor_file.h"

#include <string>
#include <utility>
#include <vector>

namespace denoiser {

enum class OutputKind { Magnitude, Mask, ComplexMask };
enum class Framing { Causal, Centred };
enum class Activation { None, Relu, Sigmoid };

// The parts of a model input frame (derived.input_layout); each is n_freq values
enum class InputPart {
    Log1pMag,        // log1p(|spec|)
    CompressedReal,  // |spec|^compress * cos(angle)
    CompressedImag,  // |spec|^compress * sin(angle)
    Log1pLongMag,    // log1p(pooled long-window magnitude)
};

// Everything the feature and synthesis code needs, from the header's
// "derived" and "realtime" (names as in the file, in camelCase)
struct ModelInfo {
    int sampleRate = 0;
    int nFft = 0;
    int hop = 0;
    int longFactor = 0;  // 0: no long window
    int longNFft = 0;    // nFft * longFactor
    int nFreq = 0;       // nFft / 2 + 1
    int nIn = 0;         // model input frame size
    int nOut = 0;        // model output frame size
    std::vector<std::pair<InputPart, int>> inputLayout;  // in order; sizes are all nFreq
    OutputKind output = OutputKind::Magnitude;
    Activation outActivation = Activation::None;  // applied by the network, last
    float compress = 0.0f;    // complex_mask only (else 0)
    float decompress = 0.0f;  // complex_mask only: 1 / compress computed in double, as float32
    float maskBound = 0.0f;   // complex_mask only; 0 = no bound
    int contextFrames = 1;    // input frames one output frame depends on
    Framing framing = Framing::Causal;
    int streamDelay = 0;      // nFft - hop for causal models, -1 for centred ones
    bool realtime = false;    // header "realtime": false = offline (ARA) only
    std::string netType;      // net.type: "mlp", "tcn", "snn", ...

    // Window used by analysis: the long one if there is one, else nFft
    int widestWindow() const noexcept { return longFactor ? longNFft : nFft; }
};

class ModelFile {
public:
    // Throws FormatError if the file is not a valid model file of kind_version 1
    static ModelFile load(const std::string& path);
    static ModelFile fromMemory(const void* bytes, std::size_t size);
    // From an already parsed container (e.g. one built in a test)
    static ModelFile fromTensorFile(TensorFile file);

    ModelFile(ModelFile&&) noexcept = default;
    ModelFile& operator=(ModelFile&&) noexcept = default;

    const ModelInfo& info() const noexcept { return info_; }
    // The header's "net" object, for building the network
    const Json& net() const { return file_.header().at("net"); }
    // The whole header (also "config", for information)
    const Json& header() const noexcept { return file_.header(); }
    // The weights, by name (TensorFile::get checks names and shapes)
    const TensorFile& tensors() const noexcept { return file_; }

private:
    explicit ModelFile(TensorFile file);

    TensorFile file_;
    ModelInfo info_;
};

// Parses and checks a model header (the JSON object). Throws FormatError
// naming the offending key. Exposed for tests and tools.
ModelInfo parseModelInfo(const Json& header);

}  // namespace denoiser
