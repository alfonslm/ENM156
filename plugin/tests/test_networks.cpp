// The MLP and SNN against every golden MLP / SNN case, each network on its
// own: the golden x in, frame by frame, pred compared with the golden pred
// (tolerance 1e-5 of max(1, max |golden|), tools/README.md) and the SNN's
// spikes compared with snn.spikes* exactly, for both SNN kernels. Also:
// reset() reproduces the first run bit for bit, the factory builds a TCN
// (tested on its own in test_tcn.cpp) and refuses broken weights, and Model
// shares one set of weights.
#include "check.h"
#include "golden.h"

#include "denoiser/denoiser.h"
#include "denoiser/networks.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <tuple>

using namespace denoiser;

namespace {

constexpr double kTol = 1e-5;

// Records every layer's spikes of every time step
class SpikeLog : public SnnNetwork::SpikeObserver {
public:
    explicit SpikeLog(std::size_t layers) : steps(layers) {}
    void spikes(int layer, const float* s, int size) noexcept override {
        steps[std::size_t(layer)].emplace_back(s, s + size);
    }
    std::vector<std::vector<std::vector<float>>> steps;  // [layer][step][neuron]
};

std::vector<float> runAll(FrameNetwork& net, const Tensor& x, std::size_t nOut) {
    const std::size_t frames = x.shape[0], nIn = x.shape[1];
    std::vector<float> y(frames * nOut);
    for (std::size_t t = 0; t < frames; ++t)
        net.process(x.data + t * nIn, y.data() + t * nOut);
    return y;
}

// Spike mismatches against snn.spikes{i} (all stored steps); -1 if the case stores none
long spikeMismatches(const golden::Case& c, const SpikeLog& log) {
    long mismatches = 0;
    bool any = false;
    for (std::size_t i = 0; i < log.steps.size(); ++i) {
        const std::string name = "snn.spikes" + std::to_string(i);
        const Tensor* ref = c.find(name.c_str());
        if (!ref)
            continue;
        any = true;
        const std::size_t steps = ref->shape[0], size = ref->shape[1];
        CHECK(log.steps[i].size() >= steps);
        for (std::size_t t = 0; t < steps && t < log.steps[i].size(); ++t) {
            CHECK(log.steps[i][t].size() == size);
            for (std::size_t k = 0; k < size; ++k)
                mismatches += log.steps[i][t][k] != ref->data[t * size + k];
        }
    }
    return any ? mismatches : -1;
}

void checkCase(const golden::Case& c) {
    const ModelInfo& info = c.info();
    const Tensor& x = c.get("x");
    const Tensor& pred = c.get("pred");
    const std::size_t nOut = std::size_t(info.nOut);
    CHECK(x.shape[1] == std::size_t(info.nIn) && pred.shape[1] == nOut);

    std::unique_ptr<FrameNetwork> net = makeNetwork(c.model);
    const std::vector<float> y = runAll(*net, x, nOut);
    const double err = golden::relErr(y.data(), pred.data, y.size());
    CHECK(err <= kTol);
    std::printf("%-28s pred %.1e", c.name.c_str(), err);

    // reset() gives the same output again, bit for bit (after other input)
    net->reset();
    const std::vector<float> again = runAll(*net, x, nOut);
    CHECK(std::memcmp(y.data(), again.data(), y.size() * sizeof(float)) == 0);

    if (info.netType == "snn") {
        const auto weights = SnnWeights::build(c.model);
        std::vector<std::vector<std::vector<float>>> spikesOfFirstKernel;
        for (SnnNetwork::Kernel kernel : {SnnNetwork::Kernel::Spikes, SnnNetwork::Kernel::Dense}) {
            SnnNetwork snn(weights, kernel);
            SpikeLog log(weights->layers.size());
            snn.setSpikeObserver(&log);
            const std::vector<float> ys = runAll(snn, x, nOut);
            const double e = golden::relErr(ys.data(), pred.data, ys.size());
            CHECK(e <= kTol);
            // every step of every frame, so the spike rates can be compared with the header's too
            const long mismatches = spikeMismatches(c, log);
            CHECK(mismatches <= 0);
            const Json& stats = c.vectors.header().at("snn");
            const auto& rates = stats.at("spike_rate_per_layer").array();
            for (std::size_t i = 0; i < log.steps.size(); ++i) {
                double sum = 0.0;
                for (const auto& step : log.steps[i])
                    for (float s : step)
                        sum += s;
                const double rate = sum / double(log.steps[i].size() * log.steps[i][0].size());
                CHECK_NEAR(rate, rates[i].number(), 1e-4);  // the header rounds to 4 digits
            }
            // both kernels: the same spikes at every step of every frame
            if (kernel == SnnNetwork::Kernel::Spikes)
                spikesOfFirstKernel = log.steps;
            else
                CHECK(log.steps == spikesOfFirstKernel);
            std::printf(" | %s pred %.1e, spike mismatches %s, rate %.3f", kernel == SnnNetwork::Kernel::Spikes ? "spikes" : "dense",
                        e, mismatches < 0 ? "(none stored)" : std::to_string(mismatches).c_str(),
                        double(snn.spikeCount()) / double(snn.neuronSteps()));
            snn.setSpikeObserver(nullptr);
            snn.reset();
            CHECK(snn.spikeCount() == 0);
            const std::vector<float> ys2 = runAll(snn, x, nOut);
            CHECK(std::memcmp(ys.data(), ys2.data(), ys.size() * sizeof(float)) == 0);
        }
    }
    std::printf("\n");
}

// A copy of a model file with one tensor renamed (so it is "missing")
std::vector<char> withRenamedTensor(std::vector<char> bytes, const std::string& from, const std::string& to) {
    CHECK(from.size() == to.size());
    const std::string needle = "\"" + from + "\"";
    const std::string with = "\"" + to + "\"";
    auto it = std::search(bytes.begin(), bytes.end(), needle.begin(), needle.end());
    CHECK(it != bytes.end());
    if (it != bytes.end())
        std::copy(with.begin(), with.end(), it);
    return bytes;
}

template <class E, class F>
bool throws(F&& f) {
    try {
        f();
    } catch (const E&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

void checkErrors() {
    // tcn: the factory and Model both build it (its output is checked in
    // test_tcn.cpp and, end to end, in test_denoiser_golden.cpp)
    const ModelFile tcn = ModelFile::load(golden::dir() + "/tcn_mask_short.model.bin");
    CHECK(makeNetwork(tcn) != nullptr);
    const std::vector<char> tcnBytes = golden::readBytes(golden::dir() + "/tcn_mask_short.model.bin");
    CHECK(Model::fromMemory(tcnBytes.data(), tcnBytes.size())->makeNetwork() != nullptr);

    // a missing weight: a FormatError when the model loads, naming the tensor
    for (const auto& [file, from, to] : {std::make_tuple("mlp_mask_default", "linear1.bias", "linear1.bia_"),
                                         std::make_tuple("snn_complex_recurrent", "layer1.recurrent", "layer1.recurrenX"),
                                         std::make_tuple("snn_complex_default", "readout.decay", "readout.decaX")}) {
        const std::vector<char> bad = withRenamedTensor(golden::readBytes(golden::dir() + "/" + file + ".model.bin"), from, to);
        bool named = false;
        try {
            Model::fromMemory(bad.data(), bad.size());
        } catch (const FormatError& e) {
            named = std::string(e.what()).find(from) != std::string::npos;
            std::printf("missing %s: \"%s\"\n", from, e.what());
        }
        CHECK(named);
    }
}

void checkModelSharing() {
    const std::vector<char> bytes = golden::readBytes(golden::dir() + "/snn_complex_recurrent.model.bin");
    const std::shared_ptr<const Model> model = Model::fromMemory(bytes.data(), bytes.size());
    std::printf("describe: %s\n", model->describe().c_str());
    ChannelDenoiser a(model), b(model);
    CHECK(&a.model() == &b.model() && model.use_count() == 3);
    auto& sa = dynamic_cast<SnnNetwork&>(a.network());
    auto& sb = dynamic_cast<SnnNetwork&>(b.network());
    CHECK(&sa.weights() == &sb.weights());  // one transposed copy for all channels
    CHECK(sa.weights().layers[1].weight == model->file().tensors().get("layer1.weight").data);  // read in place

    const std::vector<char> mlpBytes = golden::readBytes(golden::dir() + "/mlp_mask_default.model.bin");
    const auto mlp = Model::fromMemory(mlpBytes.data(), mlpBytes.size());
    std::printf("describe: %s\n", mlp->describe().c_str());
    // centred models can't run live
    const std::vector<char> centred = golden::readBytes(golden::dir() + "/offline_mlp_mask_centred.model.bin");
    const auto off = Model::fromMemory(centred.data(), centred.size());
    CHECK(!off->info().realtime);
    CHECK(throws<std::invalid_argument>([&] { ChannelDenoiser d(off); }));
    CHECK(off->makeNetwork() != nullptr);
}

}  // namespace

int main(int argc, char** argv) {
    golden::init(argc, argv);
    std::printf("pred error = max|c++ - golden| / max(1, max|golden|), from the golden x\n");
    int cases = 0;
    for (const std::string& name : golden::caseNames()) {
        const golden::Case c = golden::load(name);
        if (c.info().netType != "mlp" && c.info().netType != "snn")
            continue;
        checkCase(c);
        ++cases;
    }
    CHECK(cases == 12);
    checkErrors();
    checkModelSharing();
    return check::finish();
}
