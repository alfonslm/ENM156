#include "denoiser/denoiser.h"

#include "denoiser/networks.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace denoiser {

// ---- Model --------------------------------------------------------------------

Model::Model(ModelFile file) : file_(std::move(file)) {
    if (info().netType == "snn")
        snn_ = SnnWeights::build(file_);
    // Build one network now, so that a file that loads can always make networks
    // (missing tensors, wrong shapes and unsupported types fail here)
    makeNetwork();
}

Model::~Model() = default;

std::shared_ptr<const Model> Model::fromFile(ModelFile file) {
    return std::shared_ptr<const Model>(new Model(std::move(file)));
}

std::shared_ptr<const Model> Model::load(const std::string& path) { return fromFile(ModelFile::load(path)); }

std::shared_ptr<const Model> Model::fromMemory(const void* bytes, std::size_t size) {
    return fromFile(ModelFile::fromMemory(bytes, size));
}

std::unique_ptr<FrameNetwork> Model::makeNetwork() const {
    if (snn_)
        return std::make_unique<SnnNetwork>(snn_);
    return denoiser::makeNetwork(file_);
}

std::string Model::describe() const {
    const ModelInfo& m = info();
    const Json& net = file_.net();
    std::string s = m.netType + " " + std::to_string(m.nIn);
    if (m.netType == "mlp") {
        for (const Json& l : net.at("layers").array())
            s += " -> " + std::to_string(l.at("out").integer());
    } else if (m.netType == "snn") {
        for (const Json& h : net.at("hidden").array())
            s += " -> " + std::to_string(h.integer());
        s += " -> " + std::to_string(m.nOut) + " (spiking";
        if (net.at("recurrent").boolean())
            s += ", recurrent";
        s += ", " + std::to_string(net.at("steps_per_frame").integer()) + " step(s) per frame)";
    } else {
        s += " -> " + std::to_string(m.nOut);
    }
    s += m.output == OutputKind::Magnitude ? ", magnitude" : m.output == OutputKind::Mask ? ", mask" : ", complex_mask";
    s += "; n_fft " + std::to_string(m.nFft) + ", hop " + std::to_string(m.hop);
    if (m.longFactor)
        s += ", long window " + std::to_string(m.longNFft);
    s += "; " + std::to_string(m.sampleRate) + " Hz; ";
    s += m.realtime ? "real-time" : "centred frames: offline only";
    return s;
}

// ---- ChannelDenoiser ------------------------------------------------------------

namespace {
const ModelInfo& checkedInfo(const std::shared_ptr<const Model>& model) {
    if (!model)
        throw std::invalid_argument("ChannelDenoiser: no model");
    return model->info();
}
}  // namespace

ChannelDenoiser::ChannelDenoiser(std::shared_ptr<const Model> model)
    : model_(std::move(model)), stream_(checkedInfo(model_)), network_(model_->makeNetwork()) {
    reset();
}

void ChannelDenoiser::reset() noexcept {
    stream_.reset();
    network_->reset();
}

void ChannelDenoiser::processHop(const float* in, float* out, int hop) noexcept {
    if (hop != stream_.hop()) {
        std::fill(out, out + std::max(hop, 0), 0.0f);
        return;
    }
    const int preRoll = stream_.process(in, out, *network_);
    if (mutePreRoll_ && preRoll > 0)
        std::fill(out, out + preRoll, 0.0f);
}

HostRateConfig ChannelDenoiser::hostRateConfig(int hostRate, int channels, HighBand highBand) const noexcept {
    HostRateConfig c;
    c.hostRate = hostRate;
    c.channels = channels;
    c.hop = hop();
    c.processorDelay = processorDelay();
    c.highBand = highBand;
    return c;
}

}  // namespace denoiser
